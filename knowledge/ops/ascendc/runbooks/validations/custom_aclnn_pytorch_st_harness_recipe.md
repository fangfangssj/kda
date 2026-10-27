---
type: "Runbook"
title: 自定义 aclnn 算子 PyTorch 接入 ST 测试工程配方
description: 把已安装到 opp/vendors 的自定义 aclnn 算子接入 PyTorch 做上板 ST/精度验收：torch_adapter.so 桥接层（TORCH_LIBRARY 注册 + aclnn 两段式封装）+ CMake 链接 torch/torch_npu/libcust_opapi + test.py 先 CPU golden 自测再 NPU 全量比对（非计算类算子 bitwise 优先）。
tags: [task.testing, topic.implementation.api_layer.c_api, topic.precision]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# 自定义 aclnn 算子 PyTorch 接入 ST 测试工程配方

## 现象

一个 AscendC 自定义算子（Transpose 类纯搬运算子）已完成开发并通过 C++ ST，算子包已安装到 `$ASCEND_HOME_PATH/opp/vendors/<vendor名>/op_api/`（含 `aclnn_transpose.h` 与 `libcust_opapi.so`）。现在需要从 PyTorch 侧端到端验证它：在真实 NPU 上以 `torch` 张量为输入调用该自定义算子并与 CPU golden 比对。工程落在算子仓 `tests/st/torch/` 下，共 5 个文件（golden.py / compare.py / test.py / torch_adapter.cpp / CMakeLists.txt），构建产物 `libtorch_adapter.so`。数据流：

```
PyTorch Tensor (NPU) -> torch_adapter.so -> ACLNN 两段式接口 -> NPU 结果
CPU golden (golden.py)  <-compare.py->  NPU 实际结果（取回 CPU 后比对）
```

调用形态：测试进程用 `torch.ops.load_library("./build/libtorch_adapter.so")` 装载桥接库，然后 `torch.ops.cann_bench.transpose(x_npu, perm)` 触发 NPU 执行。构建与运行全部一次跑通：CPU golden 自测 12/12、L0 门槛 5/5、全量 49/49 PASS（0 FAIL 0 SKIP），最终精度验收在同一环境重跑 49/49 全部 bitwise 一致。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）交付/验收阶段的 PyTorch 接入 ST（也可用于黑盒精度验收，替代/补充 C++ ST）。
- API 路径：`torch.ops.load_library` + `TORCH_LIBRARY` 自定义命名空间注册；aclnn 两段式接口（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）；vendor 包 `op_api/include/aclnn_*.h` 与 `op_api/lib/libcust_opapi.so`；CANN `libascendcl.so`/`libopapi.so`/`libnnopbase.so`；`torch_npu`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35，1 卡）+ CANN 9.1.0 + PyTorch + torch_npu（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——任何已安装 vendor 包、暴露 aclnn 两段式接口的自定义算子，换算子只需改 adapter 内的接口封装与用例表。
- 路线优先级：recommended_fastpath（相比 C++ ST，能直接复用 torch 数据生成/golden/比对生态，且验证的是真实 aclnn 调用链）。
- 明确不适用 / 未验证场景：未安装到 `opp/vendors` 的算子（没有 `libcust_opapi.so` 可链）；GE/图模式调用路径未覆盖（本工程只走 aclnn 直调）；CPU-only 环境只能跑 golden 自测（NPU 用例自动 SKIP）。

## 系统信息

- 主机/工具链：x86_64 Ubuntu，cmake 3.20.5，g++ 13.3.0，python3.13（torch + torch_npu）。
- NPU：Ascend950PR（arch35），设备数 1；`ASCEND_HOME_PATH` 指向 CANN 9.1.0 安装目录。
- 目录布局（`operators/<op>/tests/st/torch/`）：

```
torch/
├── CMakeLists.txt          # 构建配置（本卡完整内联）
├── torch_adapter.cpp       # PyTorch 算子注册 + ACLNN 两段式封装（接口契约见下文）
├── golden.py               # CPU golden = torch.permute + 12 项自测
├── compare.py              # 精度比对（本卡内联关键实现）
├── test.py                 # 测试入口（本卡完整内联）
└── build/libtorch_adapter.so  # 编译产物
```

- vendor 包路径约定：头文件 `$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/include/aclnn_transpose.h`，库 `$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib/libcust_opapi.so`（vendor 名通过 CMake 变量 `OP_VENDOR_NAME` 传入，默认取安装目录名）。

## 完整代码

### CMakeLists.txt（构建配置，verbatim）

```cmake
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for BenchNoHarnessTranspose ST test (PyTorch adapter)
cmake_minimum_required(VERSION 3.10)
project(aclnn_transpose_torch_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 算子 vendor 包名 (安装后 opp/vendors/ 下的目录名)
set(OP_VENDOR_NAME "bench_no_harness_transpose_custom" CACHE STRING "Custom operator vendor package name")

# ============================================================================
# Find PyTorch
# ============================================================================
find_package(Torch REQUIRED)
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "PyTorch path: ${TORCH_INSTALL_PREFIX}")

# ============================================================================
# Find CANN
# ============================================================================
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH directory does not exist!")
endif()

set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)

message(STATUS "CANN_ROOT: ${CANN_ROOT}")

# Find torch_npu
execute_process(
    COMMAND python3 -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_NPU_ROOT)
    set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_ROOT}/include")
endif()

if(TORCH_NPU_INCLUDE_DIR)
    message(STATUS "torch_npu include dir: ${TORCH_NPU_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "torch_npu include directory not found.")
endif()

# Find torch_npu library
find_library(TORCH_NPU_LIBRARY torch_npu
    PATHS
        ${TORCH_NPU_ROOT}/lib
    PATH_SUFFIXES lib
)

if(TORCH_NPU_LIBRARY)
    message(STATUS "torch_npu library: ${TORCH_NPU_LIBRARY}")
else()
    message(FATAL_ERROR "torch_npu library not found.")
endif()

# Find custom operator (aclnn_transpose.h)
find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_transpose.h
    PATHS
        ${CANN_ROOT}/opp/vendors/${OP_VENDOR_NAME}/op_api/include
        /usr/local/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/include
)

if(CUSTOM_OP_INCLUDE_DIR)
    message(STATUS "Custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Custom operator include directory not found (vendor=${OP_VENDOR_NAME}). "
                        "Please install the operator first or set -DOP_VENDOR_NAME=<name>")
endif()

find_library(CUSTOM_OP_LIBRARY cust_opapi
    PATHS
        ${CANN_ROOT}/opp/vendors/${OP_VENDOR_NAME}/op_api/lib
        /usr/local/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/${OP_VENDOR_NAME}/op_api/lib
)

if(CUSTOM_OP_LIBRARY)
    message(STATUS "Custom operator library: ${CUSTOM_OP_LIBRARY}")
else()
    message(FATAL_ERROR "cust_opapi library not found (vendor=${OP_VENDOR_NAME}).")
endif()

# ============================================================================
# Build shared library
# ============================================================================
add_library(torch_adapter SHARED
    torch_adapter.cpp
)

target_include_directories(torch_adapter PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${ASCENDCL_INCLUDE_DIRS}
    ${CANN_INCLUDE_DIRS}
    ${CUSTOM_OP_INCLUDE_DIR}
)

target_link_libraries(torch_adapter PRIVATE
    ${TORCH_LIBRARIES}
    ${ASCENDCL_LIBRARIES}
    ${CANN_LIBRARIES}
    ${CANN_ROOT}/lib64/libnnopbase.so
    ${CUSTOM_OP_LIBRARY}
    ${TORCH_NPU_LIBRARY}
)

# Set rpath
get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
get_filename_component(TORCH_NPU_LIB_DIR ${TORCH_NPU_LIBRARY} DIRECTORY)
target_link_options(torch_adapter PRIVATE
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
    "-Wl,-rpath,${CANN_ROOT}/lib64"
    "-Wl,-rpath,${TORCH_NPU_LIB_DIR}"
)

install(TARGETS torch_adapter DESTINATION lib)

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build config:")
message(STATUS "  Output: libtorch_adapter.so")
message(STATUS "  Test:   python3 ../test.py --lib ./libtorch_adapter.so")
message(STATUS "========================================")
message(STATUS "")
```

### test.py（测试入口，verbatim，用例表节选结构、入口与执行逻辑全量）

```python
#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
ST 测试 - Transpose 算子 ACLNN 两段式接口 PyTorch 接入验证

数据流：
    PyTorch Tensor (NPU) -> torch_adapter.so -> ACLNN -> NPU 结果
    CPU golden (golden.py)  <-compare.py->  NPU 实际结果（取回 CPU 后比对）

测试用例：L0 (5 个门槛用例) + L1 (44 个功能用例) = 49 个
覆盖：7 dtype (fp16/fp32/bf16/int8/int16/int32/int64)
      ndim 2~8, perm 模式: identity/swap/cyclic/full-reverse
      边界: 空 tensor / size-1 / 质数非对齐 / 大 numel / 最大 ndim
      TilingKey: TK0/TK1/TK2 三分支

使用方法：
    python3 test.py --lib /path/to/libtorch_adapter.so
    python3 test.py --lib libtorch_adapter.so --case 0
    python3 test.py --lib libtorch_adapter.so --level L0
"""

import torch
import argparse
import sys
import os
import time

from golden import compute_golden_transpose, test_golden_correctness
from compare import compare_results

CASE_COOLDOWN_MS = int(os.environ.get("CASE_COOLDOWN_MS", 100))

# ============================================================================
# dtype 字符串 -> torch.dtype 映射
# ============================================================================

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


# ============================================================================
# 测试数据生成
# ============================================================================
# 与 C++ ST 测试保持一致的数据生成策略：
#   - 浮点 (fp16/fp32/bf16): (i + 1) * 0.5, i = 0..numel-1 (所有值在 fp16/bf16 中精确可表示)
#   - 整型 (int8/int16/int32/int64): i, i = 0..numel-1 (int8/int16 自动回绕)

def generate_data(numel: int, dtype: torch.dtype) -> torch.Tensor:
    """
    生成测试数据（1D tensor, 调用方负责 reshape）
    """
    if numel == 0:
        return torch.empty(0, dtype=dtype)

    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        # (i + 1) * 0.5: 0.5, 1.0, 1.5, 2.0, ... (fp16/bf16 精确可表示)
        indices = torch.arange(numel, dtype=torch.float32)
        data = (indices + 1) * 0.5
        return data.to(dtype)
    else:
        # 整型: 0, 1, 2, ... (int8/int16 自动回绕)
        indices = torch.arange(numel, dtype=torch.int64)
        return indices.to(dtype)


# ============================================================================
# 测试用例定义（结构示意；完整 49 例按同一 _make_case 模式覆盖下述维度组合）
# ============================================================================
# L0 门槛 5 例（全 fp16）：2D 基本转置 / 4D NCHW->NHWC / 4D perm[0,2,1,3] /
#                        2D identity（TK0 拷贝路径）/ 2D 全 1 最小 shape
# L1 功能 44 例：7 dtype × 2D 转置 / 4D 多 perm / 3D perm / TK2 通用 N-D（全反转、
#               5D/6D/8D 最大 ndim）/ TK0 恒等 / 空 tensor / size-1 维 / 质数非对齐维
#               （1009/1021/4001/1013）/ 大 numel（含 128M 元素）/ int64 8 字节对齐

def _make_case(case_id, dtype_str, shape, perm, description, level):
    """构建测试用例字典"""
    return {
        "case_id": case_id,
        "dtype": DTYPE_MAP[dtype_str],
        "dtype_str": dtype_str,
        "shape": tuple(shape),
        "perm": list(perm),
        "description": description,
        "level": level,
    }


def get_test_cases(level=None):
    """返回测试用例列表, 可按 level 过滤"""
    cases = get_l0_test_cases() + get_l1_test_cases()
    if level:
        cases = [c for c in cases if c["level"] == level]
    return cases


# ============================================================================
# 辅助函数
# ============================================================================

def compute_output_shape(shape, perm):
    """计算输出 shape: y.shape[i] = x.shape[perm[i]]"""
    return tuple(shape[perm[i]] for i in range(len(perm)))


def get_numel(shape):
    """计算元素总数"""
    n = 1
    for d in shape:
        n *= d
    return n


# ============================================================================
# 单条测试执行
# ============================================================================

def run_test(tc):
    """
    执行单条测试用例

    Returns:
        True: 通过, False: 失败, None: 跳过 (NPU 不可用)
    """
    shape = tc["shape"]
    perm = tc["perm"]
    dtype = tc["dtype"]
    numel = get_numel(shape)
    out_shape = compute_output_shape(shape, perm)

    print(f"\n[{tc['case_id']}] {tc['description']}")
    print(f"  dtype: {tc['dtype_str']}, shape: {list(shape)}, perm: {perm}")
    print(f"  output shape: {list(out_shape)}, numel: {numel}")

    # 1. 生成输入数据 (CPU)
    x_cpu = generate_data(numel, dtype).reshape(shape)

    # 2. 计算 golden (CPU)
    golden = compute_golden_transpose(x_cpu, perm)

    # 3. 验证 golden shape
    if golden.shape != torch.Size(out_shape):
        print(f"  [FAIL] Golden shape 不匹配: {golden.shape} vs {list(out_shape)}")
        return False

    # 4. 检查 NPU 可用性
    if not torch.npu.is_available():
        print("  [SKIP] NPU 不可用")
        return None

    # 5. 传输到 NPU 并执行
    try:
        x_npu = x_cpu.to("npu")
        result = torch.ops.cann_bench.transpose(x_npu, perm)
    except Exception as e:
        print(f"  [FAIL] NPU 执行异常: {e}")
        return False

    # 6. 取回结果 (cpu() 内部会隐式同步)
    result_cpu = result.cpu()

    # 7. 比对
    passed = compare_results(golden, result_cpu)

    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # 用例间冷却，避免 NPU 内存池复用导致的时序问题
    if CASE_COOLDOWN_MS > 0:
        time.sleep(CASE_COOLDOWN_MS / 1000.0)

    return passed


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ST 测试 - Transpose 算子 ACLNN 两段式接口 PyTorch 接入验证")
    parser.add_argument("--lib", required=True,
                        help="共享库路径 (libtorch_adapter.so)")
    parser.add_argument("--case", type=int, default=None,
                        help="执行指定测试用例编号 (0-based)")
    parser.add_argument("--level", choices=["L0", "L1"], default=None,
                        help="只执行指定级别的用例")
    args = parser.parse_args()

    print(f"\n加载共享库: {args.lib}")
    args.lib = os.path.realpath(args.lib)
    if not os.path.exists(args.lib):
        print(f"错误: 文件不存在 {args.lib}")
        sys.exit(1)

    torch.ops.load_library(args.lib)
    print("共享库加载成功")

    # 1. CPU Golden 自测
    print(f"\n{'=' * 60}")
    print("CPU Golden 正确性自测")
    print(f"{'=' * 60}")
    golden_passed = test_golden_correctness()
    print(f"\nGolden 自测: {'PASS' if golden_passed else 'FAIL'}")

    if not golden_passed:
        print("错误: Golden 自测失败, 终止测试")
        sys.exit(1)

    # 2. 获取测试用例
    test_cases = get_test_cases(level=args.level)

    if args.case is not None:
        if args.case < 0 or args.case >= len(test_cases):
            print(f"错误: 测试用例编号超出范围 (0-{len(test_cases)-1})")
            sys.exit(1)
        test_cases = [test_cases[args.case]]

    # 3. 执行测试
    print(f"\n{'=' * 60}")
    print(f"设备: {'NPU' if torch.npu.is_available() else 'CPU (NPU 不可用, 将跳过)'}")
    print(f"用例数: {len(test_cases)}")
    print(f"{'=' * 60}")

    passed_count = 0
    failed_count = 0
    skipped_count = 0
    failed_cases = []

    for tc in test_cases:
        result = run_test(tc)
        if result is None:
            skipped_count += 1
        elif result:
            passed_count += 1
        else:
            failed_count += 1
            failed_cases.append(tc["case_id"])

    # 4. 汇总报告
    print(f"\n{'=' * 60}")
    print("总体测试报告")
    print(f"{'=' * 60}")
    print(f"总计: {len(test_cases)}")
    print(f"通过: {passed_count}")
    print(f"失败: {failed_count}")
    print(f"跳过: {skipped_count}")
    print(f"{'=' * 60}")

    if failed_count > 0:
        print("\n失败的测试用例:")
        for case_id in failed_cases:
            print(f"  - {case_id}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

### compare.py（精度比对核心，verbatim）

非计算类（纯搬运）算子的比对标准：整型必须 bitwise；浮点预期 bitwise，未达 bitwise 时按 `MERE < 阈值 且 MARE < 10×阈值` 兜底（fp16 2^-10 ≈ 9.77e-4 / bf16 2^-7 ≈ 7.81e-3 / fp32 2^-13 ≈ 1.22e-4）。

```python
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
精度比对模块 - Transpose 算子

Transpose 为纯数据搬运（非计算类），依据 ops-precision-standard/non_compute.md：
  - 核心标准：二进制一致（Bitwise Match）
  - 整型：必须 bitwise，任何差异即判失败
  - 浮点：预期 bitwise；若因 DMA dummy padding 等原因未达 bitwise，
          以 MERE/MARE 阈值为兜底验收标准

误差指标：
  MERE = avg(|actual - golden| / (|golden| + 1e-7))
  MARE = max(|actual - golden| / (|golden| + 1e-7))
通过标准：MERE < Threshold 且 MARE < 10 * Threshold

阈值（社区标准）：
  fp16: 2^-10 ≈ 9.77e-4
  bf16: 2^-7  ≈ 7.81e-3
  fp32: 2^-13 ≈ 1.22e-4
  int8/int16/int32/int64: bitwise 精确匹配
"""

import torch


# ============================================================================
# 浮点精度阈值
# ============================================================================

FLOAT_THRESHOLDS = {
    torch.float16: 2 ** (-10),       # ~0.000977
    torch.bfloat16: 2 ** (-7),       # ~0.00781
    torch.float32: 2 ** (-13),       # ~0.000122
    torch.float64: 2 ** (-13),       # ~0.000122
}


def _get_threshold(dtype: torch.dtype) -> float:
    """根据 dtype 获取 MERE 阈值"""
    return FLOAT_THRESHOLDS.get(dtype, 2 ** (-13))


# ============================================================================
# 浮点比对：先 bitwise，后 MERE/MARE 兜底
# ============================================================================

def _compare_float(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """
    浮点精度比对（Transpose 非计算类）

    1. 先检查 bitwise（Transpose 预期 bitwise 匹配）
    2. 若 bitwise 失败，降级为 MERE/MARE 阈值比对
    """
    # 1. Bitwise 检查（预期通过）
    if torch.equal(golden, actual):
        return True

    # 2. 降级 MERE/MARE 比对
    print("    [INFO] Bitwise 不匹配, 降级为 MERE/MARE 比对")

    golden_f = golden.flatten().float()
    actual_f = actual.flatten().float()

    # MERE/MARE 公式: |actual - golden| / (|golden| + 1e-7)
    eps = 1e-7
    relative_errors = (actual_f - golden_f).abs() / (golden_f.abs() + eps)
    mere = relative_errors.mean().item()
    mare = relative_errors.max().item()

    threshold = _get_threshold(golden.dtype)
    mare_threshold = 10 * threshold

    mere_pass = mere < threshold
    mare_pass = mare < mare_threshold
    passed = mere_pass and mare_pass

    if not passed:
        print(f"    MERE: {mere:.6e} (阈值: {threshold:.6e}) {'PASS' if mere_pass else 'FAIL'}")
        print(f"    MARE: {mare:.6e} (阈值: {mare_threshold:.6e}) {'PASS' if mare_pass else 'FAIL'}")
        # 打印最大差异位置
        diff = (golden_f - actual_f).abs()
        max_idx = diff.argmax().item()
        print(f"    最大差异位置: [{max_idx}] Golden={golden_f[max_idx].item():.6f}, "
              f"Actual={actual_f[max_idx].item():.6f}")
        # 打印前 5 个不匹配
        mask = golden_f != actual_f
        mismatch_count = mask.sum().item()
        print(f"    不匹配元素数: {mismatch_count} / {golden_f.numel()}")
    else:
        print(f"    MERE: {mere:.6e} (阈值: {threshold:.6e}) PASS (兜底通过)")
        print(f"    MARE: {mare:.6e} (阈值: {mare_threshold:.6e}) PASS (兜底通过)")

    return passed


# ============================================================================
# 整型比对：bitwise 精确匹配
# ============================================================================

def _compare_integer(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """整数精度比对（二进制一致，必须精确匹配）"""
    passed = torch.equal(golden, actual)
    if not passed:
        mask = golden != actual
        mismatch = mask.sum().item()
        total = golden.numel()
        print(f"    不匹配元素数: {mismatch} / {total}")
        # 打印前 5 个不匹配
        flat_golden = golden.flatten()
        flat_actual = actual.flatten()
        indices = mask.flatten().nonzero()[:5].squeeze(-1)
        for idx in indices:
            i = idx.item()
            print(f"    [{i}] Golden={flat_golden[i].item()}, Actual={flat_actual[i].item()}")
    return passed


# ============================================================================
# 统一入口
# ============================================================================

def compare_results(golden: torch.Tensor, actual: torch.Tensor) -> bool:
    """
    比对 golden 与实际结果，失败时打印差异详情
    """
    # 形状检查
    if golden.shape != actual.shape:
        print(f"    [FAIL] Shape 不匹配: golden={golden.shape}, actual={actual.shape}")
        return False

    # dtype 检查
    if golden.dtype != actual.dtype:
        print(f"    [FAIL] Dtype 不匹配: golden={golden.dtype}, actual={actual.dtype}")
        return False

    # 空 tensor 直接通过
    if golden.numel() == 0:
        return True

    if golden.dtype.is_floating_point:
        return _compare_float(golden, actual)
    else:
        return _compare_integer(golden, actual)
```

### 构建与运行命令（verbatim，按本会话实际执行顺序）

```bash
# 1. CPU golden 自测（不依赖 NPU）
cd <op>/tests/st/torch && python3 -c "
from golden import test_golden_correctness
print('=== Golden Self-Test ===')
result = test_golden_correctness()
print(f'\nOverall: {\"PASS\" if result else \"FAIL\"}')
"

# 2. 取 PyTorch cmake 前缀路径（单独长超时执行，勿内嵌到 cmake 命令替换里）
python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
# 输出: /.../site-packages/torch/share/cmake

# 3. 配置 + 编译（用第 2 步的字面量结果）
cd <op>/tests/st/torch && rm -rf build && mkdir build && cd build && \
cmake .. -DCMAKE_PREFIX_PATH=/.../site-packages/torch/share/cmake && \
make -j$(nproc)
# 产物: build/libtorch_adapter.so

# 4. 跑 L0 门槛用例（快速冒烟）
cd <op>/tests/st/torch && python3 test.py --lib ./build/libtorch_adapter.so --level L0

# 5. 全量 49 用例（CASE_COOLDOWN_MS=0 关闭用例间冷却提速）
cd <op>/tests/st/torch && CASE_COOLDOWN_MS=0 python3 test.py --lib ./build/libtorch_adapter.so

# 6. 验收口径：显式导出 vendor 库路径与 CANN 环境后全量重跑
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/<vendor>/op_api/lib/:${LD_LIBRARY_PATH} && \
export ASCEND_HOME_PATH=/.../Ascend/cann-9.1.0 && \
cd <op>/tests/st/torch && python3 test.py --lib ./build/libtorch_adapter.so
```

说明：`torch_adapter.cpp`（`TORCH_LIBRARY(cann_bench, m)` 注册 `transpose`、内部做 aclnn 两段式封装）与 `golden.py`（CPU golden = `torch.permute` + 12 项自测）的终版写入发生在本段轨迹起点之前，未在本文内联；其对外契约从 test.py 调用点可见——`torch.ops.load_library(lib)` 后以 `torch.ops.cann_bench.transpose(x_npu, perm)` 调用，输入 NPU 张量 + Python list 形式的 perm。aclnn 两段式封装细节（workspace 申请、`aclIntArray` 构造、stream 获取）见 [PyTorch 适配层 aclnn 两段式调用完整配方](torch_adapter_aclnn_two_stage_recipe.md)。

## 设计决策

- **桥接层形态**：不侵入 torch_npu 插件机制，而是独立 `.so` 用 `TORCH_LIBRARY` 注册自定义命名空间算子，测试进程 `torch.ops.load_library` 显式装载——算子名冲突面小、加载时机可控，且能同时链 CANN 原生库与 vendor `libcust_opapi.so`。
- **链接与查找策略**：CMake 同时查找并链接 PyTorch（`find_package(Torch)`）、torch_npu（`execute_process` 定位包根 + `find_library`）、CANN（`libascendcl/libopapi/libnnopbase`）、vendor 自定义算子（`find_path` 头 + `find_library` cust_opapi，支持 `OP_VENDOR_NAME` 覆盖）；并用 `target_link_options -Wl,-rpath` 把 cust_opapi、CANN lib64、torch_npu lib 三个目录烧进 `.so`，减少对环境变量的依赖。
- **cmake 前缀路径预计算**：`cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; ...")` 的命令替换形式 60s 超时跑不完（torch_npu import 慢）；把前缀路径用长超时单独取出、以字面量传入后才成功。慢 import 不进命令替换。
- **测试分层**：先 CPU golden 自测（golden 本身错则立即终止，不浪费 NPU 时间）→ L0 门槛 5 例 → L1 全量；`--level` / `--case` 支持缩小重跑范围。
- **精度标准**：非计算类算子 bitwise 优先（整型强制），浮点 MERE/MARE 兜底且必须双指标（MERE<阈值 且 MARE<10×阈值），阈值按 dtype 社区标准；兜底通过时打印差异明细（最大差异位置、不匹配计数、前 5 个不匹配值）。
- **用例间冷却**：默认 `CASE_COOLDOWN_MS=100`（避免 NPU 内存池复用的时序问题），全量跑批可设 0 提速——本会话两种模式各跑过一遍全量，结果一致 49/49。
- **数据生成与 C++ ST 对齐**：浮点 `(i+1)*0.5`（fp16/bf16 精确可表示）、整型 `i`（自动回绕），保证 golden 与设备侧输入逐位确定。

## 性能影响

本卡为测试工程配方，无算子性能结论；未对被测算子做性能测量（该算子性能验收在本会话中被明确跳过）。

## 预防

- 上板跑批前先在纯 CPU 环境跑 golden 自测，把参考实现的错误挡在最前。
- adapter 编译报"找不到 cust_opapi/头文件"时，先确认算子包已安装且 `OP_VENDOR_NAME` 与 `opp/vendors/` 下目录名一致，再查 `ASCEND_HOME_PATH`。

## 验证信号

以下信号来自同一次算子验收会话（PyTorch ST 工程开发 + 最终精度验收两步，Ascend950PR + CANN 9.1.0）：

- 构建成功：`[100%] Linking CXX shared library libtorch_adapter.so`、`[100%] Built target torch_adapter`，产物 939584 字节 ELF 64-bit x86-64（来源平台：x86_64 构建主机；证据等级：tool_observed）。
- CPU golden 自测 12/12 PASS（2D/3D/4D、fp16 bitwise、bf16 bitwise、int8/int16/int32/int64 round-trip、非对齐质数维度、空 tensor）（来源平台：CPU；证据等级：tool_observed）。
- L0 门槛用例 5/5 PASS（来源平台：950；证据等级：tool_observed）。
- 全量 49/49 PASS、0 FAIL、0 SKIP：覆盖 7 dtype（fp16/fp32/bf16/int8/int16/int32/int64）、TK0/TK1/TK2 三分支、ndim 2~8、空 tensor / size-1 / 质数非对齐 / 大 numel（含 128M 元素）/ 最大 ndim 边界（来源平台：950；证据等级：tool_observed）。
- 最终精度验收（显式导出 vendor lib 路径后重跑）：49/49 全部 bitwise 一致，无任何降级 MERE/MARE 的 INFO 输出（来源平台：950；证据等级：tool_observed）。

## 失效边界

- `torch_adapter.cpp` 与 `golden.py` 的终版 verbatim 内容未在本段轨迹恢复（写入事件在本段起点之前）；复用本配方时需从工程目录取这两个文件，或按"TORCH_LIBRARY 注册 + aclnn 两段式"契约自行实现（实现细节见 [torch 适配层两段式调用卡](torch_adapter_aclnn_two_stage_recipe.md)）。
- 只验证 aclnn 直调路径；GE/图模式（proto + 图编译）调用未覆盖。
- 大 numel 用例（128M fp16）单例需数十秒（CPU 生成 + H2D + 执行 + D2H + 比对），默认 100ms 冷却下全量约需数分钟。
- 其他 NPU 平台与其他 CANN/torch_npu 版本未验证（未验证，推断可迁移：机制只依赖 torch_npu 通用 NPU 后端与 aclnn 标准两段式接口）。
- 本会话曾出现 NPU 设备对 `npu-smi` 与示例运行均超时无响应的时段（同日稍后恢复）；设备不可用时本工程 NPU 用例自动 SKIP，只有 golden 自测仍有效。

**未决项**：

- 用例失败时的最小化重跑入口（`--case N`）已在 test.py 实现，但未在本会话演练失败路径（无失败用例）。
## 坏实践

- `cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")`——命令替换里嵌慢 import，60s 超时失败且无任何输出（本会话实测踩坑）。
- 临时验证时直接覆盖共享目录里的 `CMakeLists.txt`，事后依赖 `git checkout` 恢复——目录不是 git 仓库时恢复失败；应先 `cp` 备份、验证后 `cp` 恢复。

## 误诊

- 无重大误诊；PyTorch ST 工程一次构建通过、全量用例一次通过。
