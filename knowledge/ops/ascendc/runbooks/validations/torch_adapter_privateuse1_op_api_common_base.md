---
type: "Runbook"
title: 自定义 aclnn 算子 PyTorch 适配层配方
description: "torch 2.7.1+cpu + torch_npu 2.7.1.post4 下为自定义 aclnn 算子构建 torch 适配层的可复用配方：TORCH_LIBRARY_IMPL 用 PrivateUse1、经 op_api_common_base.h 的 GetApiFunc/ConvertType/ExecuteApiFunc/Release/GetAclStream 直调两段式接口；规避 EXEC_NPU_CMD（g_hash_buf 弱符号）、torch/extension.h（Python 符号）、OpPreparation（未导出）与 CANN/torch_npu ACL 头冲突，输出与 workspace 用 at::empty。"
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-16T10:18:45Z'
updated_at: '2026-09-16T10:18:45Z'
platforms: ["950"]
---

# 自定义 aclnn 算子 PyTorch 适配层配方

## 现象

自定义 aclnn 算子（grid_sampler_3d 语义，Ascend950PR，CANN 9.1.0）已通过 C++ ST（Real 模式 99/99），需要再做 PyTorch 接入测试（L0+L1 全量、NPU 真实执行 + CPU golden 比对）。环境：torch 2.7.1+cpu、torch_npu 2.7.1.post4（python 3.13）、算子已装在 `opp/vendors/<op>_custom`（aclnn 符号在 `libcust_opapi.so`，安装头为规范化名——数字前下划线被构建系统去掉，见 [算子名数字段安装头卡](../compilation/opname_digit_snake_case_install_header_mismatch.md)）。

参照 torch_npu op-plugin 的 `EXEC_NPU_CMD` 宏写了第一版 `tests/st/torch/torch_adapter.cpp`（`torch/extension.h` + `op_plugin/utils/op_api_common.h` + `TORCH_LIBRARY_IMPL(..., NPU, m)`），随后在"编译→dlopen 加载"循环里连续踩了五个坑，每个都有明确报错：

1. 编译期：`fatal error: op_plugin/utils/op_api_common.h: No such file or directory`——op-plugin 头在 `torch_npu/include/third_party/op-plugin/` 下，不在 `torch_npu/include/` 顶层。
2. 编译期：`error: 'NPU' is not a member of 'c10::DispatchKey'; did you mean 'CPU'?`（`TORCH_LIBRARY_IMPL(grid_sampler_3d_ext, NPU, m)` 处）。
3. 加载期（`torch.ops.load_library`，内部 `ctypes.CDLL`）：`OSError: ... undefined symbol: g_hash_buf`——`nm -D libtorch_npu.so` 显示 `g_hash_buf`/`g_hash_offset` 仅为弱未定义（`w _ZTH10g_hash_buf`），它们的定义在 op-plugin 编译单元里，外部扩展解析不到。
4. 加载期：`OSError: ... undefined symbol: _PyThreadState_UncheckedGet`——`torch/extension.h` 拉进 pybind11/PYBIND11_MODULE，`ctypes.CDLL` 默认 RTLD_LOCAL 下 Python 符号不可见（注释掉 PYBIND11_MODULE 后仍报，说明 include 本身就引入符号引用）。
5. 加载期：`OSError: ... undefined symbol: _ZN6at_npu6native13OpPreparation22unsafe_empty_workspaceEm`——`nm -D libtorch_npu.so | grep OpPreparation` 无输出，OpPreparation 系列符号未导出。
6. 其后改用 `aclrtMalloc`/`aclrtFree` 分配 workspace 又引入 `#include <acl/acl_rt_api.h>`，与 torch_npu 自带的 `third_party/acl/inc` 头冲突。

最终形态：`torch/library.h + ATen/ATen.h + torch_npu NPUFunctions.h + op_plugin/utils/op_api_common_base.h`，PrivateUse1 注册，`GetApiFunc` 动态找符号、`ConvertType` 转 aclTensor、`ExecuteApiFunc` 执行、`Release` 清理，输出与 workspace 都用 `at::empty`——库加载成功（`Op registered: grid_sampler_3d_ext.forward`），NPU 全量 99/99 通过。

## 适用场景

- 工程形态：自定义 aclnn 算子（已安装进 `opp/vendors/<op>_custom`）的 PyTorch 接入测试/适配层开发——独立于 C++ ST，一次搭好后用 `torch.ops.load_library` + python 测试脚本驱动。
- API 路径：torch 2.7.1 `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL`；torch_npu 2.7.1.post4 `op_plugin/utils/op_api_common_base.h`（`GetApiFunc`/`ConvertType`/`ExecuteApiFunc`/`Release`/`GetAclStream`，均从 `libtorch_npu.so` 导出）；ACLNN 两段式（`aclnn<Op>GetWorkspaceSize` + `aclnn<Op>`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0，torch 2.7.1+cpu / torch_npu 2.7.1.post4 / python 3.13（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（同版 torch_npu 生态内所有自定义 aclnn 算子通用；算子签名不同只改函数指针类型与参数）。
- 路线优先级：recommended_fastpath。
- 明确不适用/未验证场景：
  - 需要 autograd 反向的适配（本卡只有 forward，无 autograd 注册）；
  - 走 `torch.utils.cpp_extension.load` 或 `import` 直用（需 PYBIND11_MODULE + 链接 libpython，本卡刻意回避）；
  - 其他 torch_npu 版本（符号导出面与 dispatch key 策略可能不同，未验证）。

## 根因/解决配方

六个坑逐一归因与修法（全部为 torch_npu 生态的符号/头文件事实，可用 `nm -D`/`grep -rn` 复核）：

| # | 报错 | 根因 | 修法 |
|---|------|------|------|
| 1 | `op_plugin/utils/op_api_common.h: No such file or directory` | op-plugin 头在 `torch_npu/include/third_party/op-plugin/` 子目录 | CMake `target_include_directories` 追加 `${TORCH_NPU_INCLUDE_DIR}/third_party/op-plugin` |
| 2 | `'NPU' is not a member of 'c10::DispatchKey'` | torch_npu 2.7 以 `PrivateUse1` 注册 NPU 后端（`torch_npu/csrc/core/npu/DeviceUtils.h` 返回 `c10::DeviceType::PrivateUse1`） | `TORCH_LIBRARY_IMPL(grid_sampler_3d_ext, PrivateUse1, m)` |
| 3 | `undefined symbol: g_hash_buf` | `op_api_common.h` 的 `EXEC_NPU_CMD` 依赖 `extern thread_local g_hash_buf` 等，定义在 op-plugin 编译单元，`libtorch_npu.so` 中仅弱未定义 | 改 include `op_plugin/utils/op_api_common_base.h`，用导出的 `GetApiFunc/ExecuteApiFunc` 直调 |
| 4 | `undefined symbol: _PyThreadState_UncheckedGet` | `torch/extension.h` 引入 pybind11（即使不写 PYBIND11_MODULE） | 换 `torch/library.h` + `ATen/ATen.h`；`torch.ops.load_library` 只需 TORCH_LIBRARY 注册，无需 pybind 模块入口 |
| 5 | `undefined symbol: at_npu::native::OpPreparation::unsafe_empty_workspace` | OpPreparation 符号未从 `libtorch_npu.so` 导出（nm -D 无输出） | 输出用 `at::empty(sizes, x.options())`；workspace 用 `at::empty(kLong)` 取 `storage().data()` |
| 6 | CANN `acl/acl_rt_api.h` 与 torch_npu `third_party/acl/inc` 头冲突 | 两套 ACL 头版本不一致 | 删除显式 `#include <acl/acl_rt_api.h>`，ACL 声明经 torch_npu 头链条获得 |

完整终版 `torch_adapter.cpp`（见《完整代码》）+ CMakeLists（见《系统信息》）即配方本体。

## 设计决策

- **直调导出函数而非复刻 EXEC_NPU_CMD**：`nm -D libtorch_npu.so` 确认 `GetApiFunc`/`RunAclCall`/`ConvertType`（多重载）/`GetAclStream`/`SetExecConfig`/`ExecuteApiFunc` 都是 `T`（强导出）符号；走 `op_api_common_base.h` 这层"无宏、无 thread_local 依赖"的头即可获得同等能力。
- **`GetApiFunc` 动态找符号而非编译期链接 cust_opapi**：算子包的 `libcust_opapi.so` 装在 vendors 目录，适配层不必硬链接；`TORCH_CHECK` 兜底"未安装自定义算子"的可读报错。
- **`workspace 用 at::empty(..., at::kLong)` 再取 `storage().data()`**：8 字节元素保证对齐，且 tensor 生命周期托管释放（替代 `OpPreparation::unsafe_empty_workspace` 与 `aclrtMalloc/aclrtFree`，同时消掉坑 5 与坑 6）。
- **不用 PYBIND11_MODULE**：`torch.ops.load_library` 只认 `TORCH_LIBRARY` 注册；pybind 入口只在你需要 `import torch_adapter` 直用时才加（届时须链接 libpython，注释里写明）。
- **连续输入先 `.contiguous()`**：ACLNN 接口要求连续 Tensor；输出 shape 由 x 的 N/C 与 grid 的 D_out/H_out/W_out 拼出。
- **当前方案路线档位**：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR，CANN 9.1.0（`ASCEND_HOME_PATH` 指向安装根），torch 2.7.1+cpu，torch_npu 2.7.1.post4（`python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"` 定位）。
- 目录布局（`tests/st/torch/`）：`torch_adapter.cpp`（适配层）、`CMakeLists.txt`（构建）、`golden.py`/`compare.py`/`test.py`（CPU golden + 用例驱动，`--lib ./build/libtorch_adapter.so` 加载适配层）。
- CMakeLists 关键段（`find_package(Torch)` + 动态定位 torch_npu/CANN，坑 1 的 include 修复已含）：

```cmake
execute_process(
    COMMAND python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    RESULT_VARIABLE TORCH_NPU_FOUND
    OUTPUT_VARIABLE TORCH_NPU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_DIR}/include")
set(TORCH_NPU_LIB_DIR "${TORCH_NPU_DIR}/lib")
set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
set(CANN_INCLUDE_DIR "${CANN_ROOT}/include")
set(CANN_LIB_DIR "${CANN_ROOT}/lib64")

add_library(torch_adapter SHARED torch_adapter.cpp)

target_include_directories(torch_adapter PRIVATE
    ${TORCH_NPU_INCLUDE_DIR}
    ${TORCH_NPU_INCLUDE_DIR}/third_party/op-plugin
    ${CANN_INCLUDE_DIR}
)

target_link_directories(torch_adapter PRIVATE
    ${TORCH_NPU_LIB_DIR}
    ${CANN_LIB_DIR}
)

target_link_libraries(torch_adapter
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    opapi
)

target_link_options(torch_adapter PRIVATE
    -Wl,--allow-shlib-undefined
    -Wl,-rpath,${TORCH_NPU_LIB_DIR}
    -Wl,-rpath,${CANN_LIB_DIR}
)
```

- 构建与加载命令：

```bash
cd tests/st/torch && mkdir -p build && cd build && \
  cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") && make -j$(nproc)
cd .. && python3 -c "
import os
os.environ['ACLNN_CACHE_LIMIT'] = '0'
import torch
import torch_npu
torch.ops.load_library('./build/libtorch_adapter.so')
print('Library loaded successfully!')
op = torch.ops.grid_sampler_3d_ext.forward
print(f'Op registered: {op}')
"
```

## 完整代码

终版 `torch_adapter.cpp`（verbatim，会话最终写入 + 两处收尾编辑后的状态；换成你的算子时只改注释、函数指针类型 `GetWorkspaceSizeFunc`、shape 拼装与 `GetApiFunc` 的两个符号名）：

```cpp
// ============================================================================
// torch_adapter.cpp — PyTorch 适配层 + ACLNN 两段式封装
//
// 功能：
//   - 注册自定义算子到 PyTorch（torch_npu PrivateUse1 dispatch）
//   - 封装 aclnnBenchNoHarnessGridSampler3dGetWorkspaceSize +
//     aclnnBenchNoHarnessGridSampler3d 两段式调用
//
// 实现说明：
//   使用 op_api_common_base.h 中的导出 API（ConvertType / GetApiFunc /
//   ExecuteApiFunc / Release / GetAclStream）直接调用 ACLNN 两段式接口。
//   不使用 EXEC_NPU_CMD 宏（避免 g_hash_buf 等 thread_local 符号依赖）。
//   不使用 OpPreparation（符号未导出），输出与 workspace 均由 at::empty 分配。
//   通过 GetApiFunc 动态查找 ACLNN 函数符号，无需编译期链接 cust_opapi 库。
// ============================================================================

#include <torch/library.h>
#include <ATen/ATen.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
// op_api_common_base.h 提供 ConvertType / GetApiFunc / ExecuteApiFunc / Release / GetAclStream
#include "op_plugin/utils/op_api_common_base.h"

// ACLNN 函数指针类型
using GetWorkspaceSizeFunc = int (*)(
    const aclTensor *, const aclTensor *,
    int64_t, int64_t, bool,
    aclTensor *, uint64_t *, aclOpExecutor **);

// ============================================================================
// GridSampler3D forward 实现（NPU）
// ============================================================================

at::Tensor grid_sampler_3d_forward(
    const at::Tensor &x,
    const at::Tensor &grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners)
{
    // 确保输入连续（ACLNN 要求连续 Tensor）
    auto x_contig = x.contiguous();
    auto grid_contig = grid.contiguous();

    // 输出 shape: (N, C, D_out, H_out, W_out)
    auto y_sizes = at::IntArrayRef({
        x_contig.size(0),       // N
        x_contig.size(1),       // C
        grid_contig.size(1),    // D_out
        grid_contig.size(2),    // H_out
        grid_contig.size(3)     // W_out
    });

    // 预分配输出 Tensor（与 x 同 dtype、同 device）
    at::Tensor y = at::empty(y_sizes, x_contig.options());

    // ---- at::Tensor → aclTensor* 转换 ----
    aclTensor *x_acl = ConvertType(x_contig);
    aclTensor *grid_acl = ConvertType(grid_contig);
    aclTensor *y_acl = ConvertType(y);

    // ---- 动态查找 ACLNN 函数符号 ----
    // CANN 构建系统将 grid_sampler_3d 规范化为 grid_sampler3d（去除数字前下划线）
    // ACLNN 函数名为 aclnnBenchNoHarnessGridSampler3d（无下划线）
    void *getWorkspaceSizeFuncAddr = nullptr;
    void *opApiFuncAddr = nullptr;
    GetApiFunc("aclnnBenchNoHarnessGridSampler3d",
               "aclnnBenchNoHarnessGridSampler3dGetWorkspaceSize",
               opApiFuncAddr, getWorkspaceSizeFuncAddr);
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnBenchNoHarnessGridSampler3d or its GetWorkspaceSize not found. "
                "Please install the custom operator first.");

    // ---- 第一阶段：GetWorkspaceSize ----
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    auto getWsFunc = reinterpret_cast<GetWorkspaceSizeFunc>(getWorkspaceSizeFuncAddr);
    auto ws_ret = getWsFunc(x_acl, grid_acl,
                            interpolation_mode, padding_mode, align_corners,
                            y_acl, &workspace_size, &executor);
    TORCH_CHECK(ws_ret == 0,
                "aclnnBenchNoHarnessGridSampler3dGetWorkspaceSize failed, error=", ws_ret);

    // ---- 分配 workspace（使用 at::empty 创建 NPU Tensor，避免 OpPreparation/ACL 依赖）----
    void *workspace_addr = nullptr;
    at::Tensor workspace_tensor;
    if (workspace_size > 0) {
        // 以 int64 元素分配，保证 8 字节对齐
        int64_t ws_count = (static_cast<int64_t>(workspace_size) + 7) / 8;
        workspace_tensor = at::empty({ws_count}, x_contig.options().dtype(at::kLong));
        workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
    }

    // ---- 第二阶段：Execute ----
    aclrtStream stream = GetAclStream();
    auto exec_ret = ExecuteApiFunc(opApiFuncAddr, stream, workspace_size, executor);
    TORCH_CHECK(exec_ret == 0,
                "aclnnBenchNoHarnessGridSampler3d execution failed, error=", exec_ret);

    // ---- 清理 ----
    // workspace_tensor 和 aclTensor 由析构/Release 自动清理
    Release(x_acl);
    Release(grid_acl);
    Release(y_acl);

    return y;
}

// ============================================================================
// PyTorch 算子注册
// ============================================================================

TORCH_LIBRARY(grid_sampler_3d_ext, m)
{
    m.def(
        "forward(Tensor x, Tensor grid, int interpolation_mode, "
        "int padding_mode, bool align_corners) -> Tensor");
}

// torch_npu 使用 PrivateUse1 作为 NPU dispatch key
TORCH_LIBRARY_IMPL(grid_sampler_3d_ext, PrivateUse1, m)
{
    m.impl("forward", grid_sampler_3d_forward);
}
```

## 预防

- 写 torch 适配层前先 `nm -D libtorch_npu.so | grep <symbol>` 确认依赖符号是否强导出：`GetApiFunc/ConvertType/ExecuteApiFunc/GetAclStream/Release` 可用，`OpPreparation::*`、`g_hash_buf` 不可用。
- 只 include `torch/library.h`，绝不 include `torch/extension.h`（除非接受链接 libpython）。
- dispatch key 写 `PrivateUse1`（torch 2.x + torch_npu 2.7 实测），不要写 `NPU`。
- 头文件搜索路径加 `third_party/op-plugin`；不给 CANN 的 `acl/` 头与 torch_npu 自带 ACL 头同场编译的机会（少 include 一个就少一次冲突）。

## 验证信号

以下信号来自该算子 PyTorch 接入测试开发会话：迭代编译/加载修复后全量执行。均为工具实测：

- 加载验证：`torch.ops.load_library('./build/libtorch_adapter.so')` 成功，`Op registered: grid_sampler_3d_ext.forward`（来源平台：950；证据等级：tool_observed）。
- 符号核查：`nm -D libtorch_npu.so` 显示 `g_hash_buf/g_hash_offset` 为弱未定义（`w`），`GetApiFunc/RunAclCall/ConvertType*/GetAclStream/SetExecConfig/ExecuteApiFunc` 为强导出（`T`），`OpPreparation` 无导出（来源平台：950；证据等级：tool_observed）。
- NPU 全量：基础+扩展共 99 条全部 PASS（如首条 MERE=1.336e-08 / MARE=2.552e-07，阈值 1.221e-04），Golden 自测 ALL PASS（来源平台：950；证据等级：tool_observed）。
- 最终精度验收复测：`bash run.sh --torch --level ALL` 99/99，测试结果 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 符号导出面与 PrivateUse1 策略绑定 torch_npu 2.7.1.post4 / torch 2.7.1：其他版本未验证（推断思路通用——先 nm -D 核符号、再定 include 与 dispatch key，具体结论需复核）。
- 本卡只注册 forward；autograd/反向、`torch.compile` 捕获行为未验证。
- `ACLNN_CACHE_LIMIT=0` 仍需设置（aclnn 执行器缓存不含 attr 的已知问题，属另一张卡的范畴）。
- `--allow-shlib-undefined` 链接选项在本配方里依赖运行期 libtorch_npu.so 已被 import torch_npu 加载；脱离 python 进程直接 dlopen 未验证。

**未决项**：

- 反向/autograd 注册未实现（本算子验收只需 forward）。
- `RunAclCall`/`SetExecConfig` 等其余导出工具的适用场景（如多输出、异步错误处理）未探索。
## 坏实践

- 照抄 op-plugin 的 `EXEC_NPU_CMD`/`op_api_common.h` 到独立扩展——`g_hash_buf` 等 thread_local 符号只在 op-plugin 编译单元有定义。
- include `torch/extension.h` 又只用 `torch.ops.load_library` 加载——pybind11 的 Python 符号引用在 RTLD_LOCAL 下解析失败。
- 用 `OpPreparation::apply_tensor_with_sizes`/`unsafe_empty_workspace`——符号未从 libtorch_npu.so 导出，链接能过（--allow-shlib-undefined）但 dlopen 必炸。
- 同时 include CANN 与 torch_npu 自带的 `acl/` 头——版本不一致直接编译冲突。

## 误诊

- 第一次 `_PyThreadState_UncheckedGet` 失败时先注释掉了 `PYBIND11_MODULE`——方向对但不彻底：`torch/extension.h` 这个 include 本身就拉 pybind11 符号，必须整体换成 `torch/library.h`（第二次加载失败后才定位到 include 层面）。
