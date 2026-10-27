---
type: "Runbook"
title: TTK e2e 自定义 aclnn 算子 Torch 路由配方
description: 自定义算子 Torch 路由完整配方：tests/torch 三件套（run.sh 构建 .run+安装+load_inline 编译 TORCH_LIBRARY adapter+直调、test_torch.py 带注解暴露函数、ttk_cases.csv manifest），脚本外用同一 CSV 跑 TTK e2e eager 与 --aclgraph 双模式并按预期拒绝核对 L2 负向。
tags: [task.testing, topic.implementation.operator_registration, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-14T12:12:55Z'
updated_at: '2026-09-14T12:12:55Z'
platforms: ["950"]
---

# TTK e2e 自定义 aclnn 算子 Torch 路由配方

## 现象

为一个已建成的 CANN 自定义算子（scatter/scatter-reduce，原型 `scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor y`，10 组 dtype 组合、5 种 reduce、dim 支持负索引）搭建 Torch 路由并在 Ascend950PR（CANN 9.2.0）上完整执行评测：要求生成 Torch manifest 与测试工程、adapter 构建并加载后用同一份 CSV 完整执行 TTK e2e 的 eager 与 ACLGraph 两种模式、manifest 与 proto/spec 一致、每个用例有可复核结果（不得把 NOT_RUN/N/A 记为 PASS）、脚本内不调用 TTK、结果写到测试目录外。本卡是这套流程跑通后的完整配方（含三个被修复的坑：aclnn String 属性 char* 编译失败、.run 重装只读目录权限、Meta 实现致 ACLGraph 浮点用例失败——分别见 [aclnn String 属性 char* 传 const 编译失败](../compilation/aclnn_string_attr_char_ptr_const_compile.md)、[.run 重装只读 scripts 目录 rm 权限拒绝](../compilation/run_package_reinstall_readonly_scripts_chmod.md)、[Meta 校验误传 data 致 ACLGraph 浮点全失败](../compilation/torch_meta_param_misalignment_aclgraph.md)）。

## 适用场景

- 工程形态：CANN 自定义算子（custom opp .run 包）的 Torch 路由开发/评测——TORCH_LIBRARY adapter 直调 aclnn C API + TTK e2e 双模式回归。
- API 路径：`torch.utils.cpp_extension.load_inline`（现场编译 adapter）；`TORCH_LIBRARY_FRAGMENT`/`TORCH_LIBRARY_IMPL`（Meta + PrivateUse1）；`aclnnXxxGetWorkspaceSize`/`aclnnXxx` + `at_npu::native::OpCommand::RunOpApiV2`；`python3 -m ttk e2e [--aclgraph]`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0；TTK 3.0.0；torch 2.x + torch_npu（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——对任何含张量输入+标量属性的 aclnn 自定义算子可直接套用，与算子语义无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：算子带输出型张量形参（out 参数，CSV 需配 output_tensor_indexes，本例 forward 返回值即输出、留空）；TF 路由；TTK 其他版本。

## 解决配方（完整流程）

1. **tests/torch/ 三件套**（版本管理范围内只维护这三个文件）：`run.sh`（构建+安装+adapter+直调+生成 golden 插件，**不调 TTK**）、`test_torch.py`（只加载 adapter、暴露带类型注解的 forward、直调一次，不实现算子语义）、`ttk_cases.csv`（api_name=test_torch.forward 的 manifest）。adapter 源码、动态库、安装包只生成到 gitignored 的 build/、install/。
2. **run.sh 五步**：① cmake 构建 .run 包（`cmake --build --target binary package`）→ ② **先 `chmod -R u+w` 再 `rm -rf`** 清旧安装并执行 `.run` 安装（安装器会建只读 scripts 目录，见 [.run 重装只读 scripts 目录 rm 权限拒绝](../compilation/run_package_reinstall_readonly_scripts_chmod.md)）→ ③ load_inline 编译 adapter：ToAclDataType 覆盖全部 5 种 dtype、CreateAclTensor（shape/stride/data_ptr → aclCreateTensor）、Invocation 结构（aclTensor 生命周期 + workspace + 属性字符串存 std::string 成员后以 `data()` 传 char*，见 [aclnn String 属性 char* 传 const 编译失败](../compilation/aclnn_string_attr_char_ptr_const_compile.md)）、Npu 实现（GetWorkspaceSize → workspace 按需分配 → `RunOpApiV2` 发射）、**Meta 实现转发真实参数做校验**（torch.compile 捕获会用 FakeTensor 走 Meta，见 [Meta 校验误传 data 致 ACLGraph 浮点全失败](../compilation/torch_meta_param_misalignment_aclgraph.md)）、`TORCH_LIBRARY_FRAGMENT` 定义 schema + Meta/PrivateUse1 双注册 → ④ 跑 test_torch.py 直调一次验证 → ⑤ 生成 build/torch_golden.py（e2e golden 插件，按 api_name 注册，机制见 [TTK e2e 按 api_name 注册 golden 插件](ttk_e2e_custom_api_name_golden_plugin.md)）。
3. **ttk_cases.csv manifest 设计**：`tensor_view_shapes` 给 3 个输入张量（data/indices/updates），`output_tensor_indexes` 留空（forward 返回值即输出）；`attributes` 给 `{"dim": .., "reduce": ..}`；**`input_data_ranges` 必须给 indices 指定值域**（如 (0,7)），否则随机整型索引越界直接报错；用例矩阵按 spec 覆盖：10 组 dtype 组合 × 代表性 reduce、dim 0/1/负值、2D/3D，外加少量 L2 负向用例（dtype 不匹配/indices dtype 非法/rank 不匹配——期望被拒，TTK 中呈现为 FAIL+明确错误信息即预期结果）。
4. **脚本外 TTK 执行**（结果写测试目录外，需 cd 到 TTK 根目录运行）：

```bash
cd "$TTK" && \
ASCEND_CUSTOM_OPP_PATH="<install>/vendors/<VENDOR>" \
TORCH_ADAPTER_PATH="<build>/torch_adapter/<adapter>.so" \
PYTHONPATH="<tests/torch 目录>" \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python3 -m ttk e2e -i ttk_cases.csv --plugin build/torch_golden.py \
    --device-whitelist 0 --pc 4 --proc-timeout 300 \
    -o <输出目录>/result_eager.csv --print full          # eager 模式
# ACLGraph 模式：同命令追加 --aclgraph，输出到 result_aclgraph.csv
# 冒烟：追加 -t <单个用例名> --pc 1
```

5. **验收核对**：① 两份结果 CSV 均 0 个 NOT_RUN/N/A 记为 PASS、每用例有可复核结果；② L2 负向用例的 FAIL 必须带明确拒绝信息（eager 为 TORCH_CHECK 文案，aclgraph 为 GRAPH_EXEC_FAILURE）；③ 证明命中本地安装包：ctypes 加载本地 vendor 的 libcust_opapi.so 确认 aclnn 符号存在，且 CANN 系统 libopapi.so/libnnopbase.so 中该符号计数为 0。

## 设计决策

- **属性映射**：reduce=None 或 "update" 传 nullptr（tiling 中 `MapReduceMode(nullptr)` 与 `"update"` 同为覆盖模式 0），add/multiply/amin/amax 传字符串——先 grep tiling 源码确认等价性后才合并传参。
- **golden 复用算子仓 numpy 参考实现**而非重写 torch 版：与 aclnn 黑盒基线同口径（fp16/bf16 add/multiply 的 fp32 累加单次舍入模型）；golden 插件由 run.sh 生成到 build/ 保证可复现。
- **adapter 用 load_inline 现场编译**而非预编译提交：adapter 是测试脚手架，绑定"当前安装包"的 aclnn 头文件与库（extra_include_paths/ldpaths 指向 install/vendors/<VENDOR>），保证每次评测命中刚安装的版本。
- **Meta 实现与 Npu 同校验**：图捕获期（FakeTensor）即可拒绝非法输入，eager 与 aclgraph 校验语义一致。
- **L2 负向用例按"预期拒绝"判读**：TTK 无负向标记机制，以 FAIL+错误文案核对（dtype/rank/indices-dtype 三类拒绝各一例）。
- **TORCH_DEVICE_BACKEND_AUTOLOAD=0**：执行环境 env.sh 的既定约定（注释称避免 GE-init 抢占 vendor tiling 注册导致 561002，机制见 [561002：torch_npu GE 初始化干扰 aclnn tiling](../compilation/561002_torch_npu_ge_init_tiling_interference.md)）；本配方全程携带执行成功，该变量的独立作用未在本轨迹做对照验证（agent_reported，承自环境文件）。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0（`source <cann>/set_env.sh`）；TTK 3.0.0（TTK=/tmp/ops-test-kit，需在其根目录运行）；torch 2.x + torch_npu（miniconda python 3.13）。
- 目录布局：`<op>_package/tests/torch/`（run.sh、test_torch.py、ttk_cases.csv；build/ 与 install/ 为生成物）。
- 相关文件：`run.sh`（本卡完整代码）；`test_torch.py`（暴露函数）；`ttk_cases.csv`（manifest）；`build/torch_golden.py`（run.sh 步骤 5 生成）；结果 CSV 写测试目录外（如 /tmp 下专用目录）。

## 完整代码

`tests/torch/run.sh` 终版（由初版完整写入叠加三处修复编辑逐字组装恢复；含 adapter C++ 源与 golden 插件生成）：

```bash
#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# BenchHarnessScatter — Torch route run script.
#
# This script does NOT call TTK. It only:
#   1. builds the operator .run package and installs it under install/
#   2. compiles a transient TORCH_LIBRARY adapter (build/torch_adapter/) that
#      binds torch.ops.glacier_bench_harness_scatter.forward to the
#      aclnnBenchHarnessScatter C API of the locally installed vendor package
#   3. runs test_torch.py (one direct call) to prove the adapter loads & the
#      aclnn API executes
#   4. emits build/torch_golden.py — the e2e golden plugin reused by the
#      external TTK eager / --aclgraph runs (generated, gitignored, not a TTK call)
#
# Operator prototype (proto.md):
#   scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor y
# ----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"
ADAPTER_DIR="${BUILD_DIR}/torch_adapter"
ADAPTER_PATH="${ADAPTER_DIR}/bench_harness_scatter_torch_adapter.so"
VENDOR_NAME="BenchHarnessScatter"
VENDOR_ROOT="${INSTALL_DIR}/vendors/${VENDOR_NAME}"

# ---- 1. Build the run package -------------------------------------------
cmake -S "${PACKAGE_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target binary package -j"$(nproc)"
RUN_PACKAGE="$(find "${BUILD_DIR}" -maxdepth 1 -type f -name '*.run' -print -quit)"
test -n "${RUN_PACKAGE}"

# ---- 2. Install it under this script's directory -----------------------
# The .run installer marks some files read-only; make them writable before rm.
chmod -R u+w "${INSTALL_DIR}" 2>/dev/null || true
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
bash "${RUN_PACKAGE}" --quiet --nox11 --install-path="${INSTALL_DIR}"
test -d "${VENDOR_ROOT}/op_api/include"
test -f "${VENDOR_ROOT}/op_api/include/aclnn_bench_harness_scatter.h"

# ---- 3. Build the transient Torch adapter (TORCH_LIBRARY) --------------
mkdir -p "${ADAPTER_DIR}"
ASCEND_CUSTOM_OPP_PATH="${VENDOR_ROOT}" \
    OP_VENDOR_ROOT="${VENDOR_ROOT}" \
    ADAPTER_DIR="${ADAPTER_DIR}" python3 - <<'PY'
import os
from pathlib import Path

import torch_npu
from torch.utils.cpp_extension import load_inline


cann_home = Path(os.environ.get("ASCEND_HOME_PATH") or os.environ["ASCEND_TOOLKIT_HOME"])
torch_npu_root = Path(torch_npu.__file__).parent
vendor_root = Path(os.environ["OP_VENDOR_ROOT"])
build_dir = Path(os.environ["ADAPTER_DIR"])
source = r"""
#include <memory>
#include <string>
#include <utility>

#include "acl/acl.h"
#include "aclnn_bench_harness_scatter.h"
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

aclDataType ToAclDataType(c10::ScalarType dtype)
{
    switch (dtype) {
        case c10::ScalarType::Float:    return ACL_FLOAT;
        case c10::ScalarType::Half:      return ACL_FLOAT16;
        case c10::ScalarType::BFloat16:  return ACL_BF16;
        case c10::ScalarType::Int:       return ACL_INT32;
        case c10::ScalarType::Long:      return ACL_INT64;
        default:                         return ACL_DT_UNDEFINED;
    }
}

aclTensor *CreateAclTensor(const torch::Tensor &tensor)
{
    const auto shape = tensor.sizes().vec();
    const auto strides = tensor.strides().vec();
    return aclCreateTensor(shape.data(), shape.size(), ToAclDataType(tensor.scalar_type()),
                           strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                           tensor.data_ptr());
}

// reduce attr mapping:
//   reduce is None  -> nullptr (optional attr, framework default "update" = overwrite)
//   reduce == "update" -> nullptr (same overwrite semantics)
//   reduce == add/multiply/amin/amax -> the C string (mutable, aclnn wants char*)
// The kernel tiling treats nullptr and "update" identically (mode 0).
char *ReduceToCstr(const c10::optional<std::string> &reduce, std::string &storage)
{
    if (!reduce.has_value()) {
        return nullptr;
    }
    const std::string &r = *reduce;
    if (r.empty() || r == "update") {
        return nullptr;
    }
    storage = r;
    return storage.data();  // C++17: non-const .data() on non-const std::string
}

struct Invocation {
    Invocation(torch::Tensor dataArg, torch::Tensor indicesArg, torch::Tensor updatesArg,
               c10::optional<std::string> reduceArg)
        : data(std::move(dataArg)), indices(std::move(indicesArg)),
          updates(std::move(updatesArg)), reduce(std::move(reduceArg)),
          reduceCstr(ReduceToCstr(this->reduce, this->reduceStorage)),
          output(torch::empty_like(this->data))
    {
        dataAcl = CreateAclTensor(data);
        indicesAcl = CreateAclTensor(indices);
        updatesAcl = CreateAclTensor(updates);
        outputAcl = CreateAclTensor(output);
    }

    ~Invocation()
    {
        if (dataAcl != nullptr) { aclDestroyTensor(dataAcl); }
        if (indicesAcl != nullptr) { aclDestroyTensor(indicesAcl); }
        if (updatesAcl != nullptr) { aclDestroyTensor(updatesAcl); }
        if (outputAcl != nullptr) { aclDestroyTensor(outputAcl); }
    }

    torch::Tensor data;
    torch::Tensor indices;
    torch::Tensor updates;
    c10::optional<std::string> reduce;
    std::string reduceStorage;
    char *reduceCstr;
    torch::Tensor output;
    torch::Tensor workspace;
    aclTensor *dataAcl = nullptr;
    aclTensor *indicesAcl = nullptr;
    aclTensor *updatesAcl = nullptr;
    aclTensor *outputAcl = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
};

void CheckInputs(const torch::Tensor &data, const torch::Tensor &indices,
                 const torch::Tensor &updates)
{
    TORCH_CHECK(data.scalar_type() == updates.scalar_type(),
                "data and updates must have the same dtype");
    TORCH_CHECK(indices.scalar_type() == c10::ScalarType::Int ||
                    indices.scalar_type() == c10::ScalarType::Long,
                "indices dtype must be int32 or int64");
    TORCH_CHECK(data.scalar_type() == c10::ScalarType::Float ||
                    data.scalar_type() == c10::ScalarType::Half ||
                    data.scalar_type() == c10::ScalarType::BFloat16 ||
                    data.scalar_type() == c10::ScalarType::Int ||
                    data.scalar_type() == c10::ScalarType::Long,
                "BenchHarnessScatter supports float16/float32/bfloat16/int32/int64 only");
    TORCH_CHECK(data.dim() == indices.dim() && data.dim() == updates.dim(),
                "data/indices/updates must have the same rank");
}

torch::Tensor Npu(const torch::Tensor &data, int64_t dim,
                  const torch::Tensor &indices, const torch::Tensor &updates,
                  c10::optional<std::string> reduce)
{
    CheckInputs(data, indices, updates);
    TORCH_CHECK(data.device().type() == c10::DeviceType::PrivateUse1 &&
                    indices.device().type() == c10::DeviceType::PrivateUse1 &&
                    updates.device().type() == c10::DeviceType::PrivateUse1,
                "BenchHarnessScatter expects NPU tensors");
    auto invocation = std::make_shared<Invocation>(data, indices, updates, reduce);
    TORCH_CHECK(invocation->dataAcl != nullptr && invocation->indicesAcl != nullptr &&
                    invocation->updatesAcl != nullptr && invocation->outputAcl != nullptr,
                "aclCreateTensor failed for BenchHarnessScatter");

    const aclnnStatus workspaceStatus = aclnnBenchHarnessScatterGetWorkspaceSize(
        invocation->dataAcl, invocation->indicesAcl, invocation->updatesAcl,
        dim, invocation->reduceCstr, invocation->outputAcl,
        &invocation->workspaceSize, &invocation->executor);
    TORCH_CHECK(workspaceStatus == ACL_SUCCESS,
                "aclnnBenchHarnessScatterGetWorkspaceSize failed with status ", workspaceStatus);

    void *workspacePtr = nullptr;
    if (invocation->workspaceSize > 0) {
        invocation->workspace = torch::empty(
            {static_cast<int64_t>(invocation->workspaceSize)},
            torch::TensorOptions().dtype(torch::kUInt8).device(data.device()));
        workspacePtr = invocation->workspace.data_ptr();
    }
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    auto launch = [invocation, workspacePtr, stream]() -> int {
        return aclnnBenchHarnessScatter(workspacePtr, invocation->workspaceSize,
                              invocation->executor, stream) == ACL_SUCCESS ? 0 : 1;
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnBenchHarnessScatter", launch);
    return invocation->output;
}

torch::Tensor Meta(const torch::Tensor &data, int64_t,
                   const torch::Tensor &indices, const torch::Tensor &updates,
                   c10::optional<std::string>)
{
    CheckInputs(data, indices, updates);  // rank/dtype sanity (works on FakeTensors)
    return torch::empty_like(data);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(glacier_bench_harness_scatter, module)
{
    module.def(
        "forward(Tensor data, int dim, Tensor indices, Tensor updates, "
        "str? reduce=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(glacier_bench_harness_scatter, Meta, module)
{
    module.impl("forward", TORCH_FN(Meta));
}

TORCH_LIBRARY_IMPL(glacier_bench_harness_scatter, PrivateUse1, module)
{
    module.impl("forward", TORCH_FN(Npu));
}
"""

load_inline(
    name="bench_harness_scatter_torch_adapter",
    cpp_sources=source,
    extra_cflags=["-std=c++17"],
    extra_include_paths=[
        str(torch_npu_root / "include"),
        str(torch_npu_root / "include/third_party/op-plugin"),
        str(cann_home / "include"),
        str(vendor_root / "op_api/include"),
    ],
    extra_ldflags=[
        str(torch_npu_root / "lib/libtorch_npu.so"),
        str(cann_home / "lib64/libascendcl.so"),
        str(cann_home / "lib64/libopapi.so"),
        str(cann_home / "lib64/libnnopbase.so"),
        str(vendor_root / "op_api/lib/libcust_opapi.so"),
        f"-Wl,-rpath,{torch_npu_root / 'lib'}:{cann_home / 'lib64'}:"
        f"{vendor_root / 'op_api/lib'}",
    ],
    build_directory=str(build_dir),
    is_python_module=False,
)
PY
test -f "${ADAPTER_PATH}"

# ---- 4. Direct call (smoke) through Torch ------------------------------
ASCEND_CUSTOM_OPP_PATH="${VENDOR_ROOT}" \
    TORCH_ADAPTER_PATH="${ADAPTER_PATH}" \
    python3 "${SCRIPT_DIR}/test_torch.py"

# ---- 5. Emit the e2e golden plugin (gitignored, reused by external TTK runs)
# This reuses the authoritative numpy golden (docs/develop/golden.py
# scatter_golden_impl) so the torch route's reference matches the blackbox
# aclnn baseline exactly (incl. fp16/bf16 accumulate-then-round for add/multiply).
# It is NOT a TTK invocation — just a generated helper file.
GOLDEN_PY="${BUILD_DIR}/torch_golden.py"
cat > "${GOLDEN_PY}" <<'GOLDEN'
#!/usr/bin/env python3
# Auto-generated by run.sh — e2e golden for test_torch.forward (BenchHarnessScatter).
# Reuses docs/develop/golden.py scatter_golden_impl (authoritative numpy golden).
import importlib.util as _ilu
import os as _os
import sys as _sys

_VENDOR_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # tests/torch
_PKG_DIR = _os.path.dirname(_os.path.dirname(_VENDOR_DIR))                  # BenchHarnessScatter_package
_DOCS_DEV = _os.path.join(_os.path.dirname(_os.path.dirname(_PKG_DIR)), "docs", "develop")
_spec = _ilu.spec_from_file_location("_bhs_golden_impl", _os.path.join(_DOCS_DEV, "golden.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
scatter_golden_impl = _mod.scatter_golden_impl
_to_numpy = _mod._to_numpy


def golden(data, dim, indices, updates, reduce=None, **kwargs):
    """e2e golden — same param order as test_torch.forward.

    Receives CPU torch tensors (data, indices, updates) + scalars dim, reduce.
    Delegates to the authoritative numpy impl for bit-exact parity with the
    aclnn blackbox golden. Returns a list of numpy arrays (golden outputs).
    """
    import numpy as _np
    d = _to_numpy(data)
    i = _to_numpy(indices)
    u = _to_numpy(updates)
    y = scatter_golden_impl(d, dim, i, u, reduce=reduce)
    return [y]


class BenchHarnessScatterTorchTestSpec:
    golden = staticmethod(golden)
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }


__spec__ = {"test_torch.forward": "BenchHarnessScatterTorchTestSpec"}
GOLDEN

echo "torch_route_ready: adapter=${ADAPTER_PATH} vendor=${VENDOR_ROOT} golden=${GOLDEN_PY}"
```

`tests/torch/test_torch.py`（终版，与初版一致）：

```python
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# BenchHarnessScatter — Torch route test harness.
#
# Only loads the transient adapter (TORCH_LIBRARY fragment) and exposes the
# function referenced by ttk_cases.csv (api_name = test_torch.forward). It does
# NOT implement operator semantics — it purely delegates to the registered
# torch.ops.glacier_bench_harness_scatter.forward, which in turn calls the
# aclnnBenchHarnessScatter C API of the locally installed vendor package.
#
# Operator prototype (proto.md / proto.yaml):
#   scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor y
# ----------------------------------------------------------------------------

import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401  registers PrivateUse1 (NPU) backend


_ADAPTER = Path(
    os.environ.get(
        "TORCH_ADAPTER_PATH",
        Path(__file__).parent / "build/torch_adapter/bench_harness_scatter_torch_adapter.so",
    )
)
torch.ops.load_library(str(_ADAPTER))


def forward(
    data: torch.Tensor,
    dim: int,
    indices: torch.Tensor,
    updates: torch.Tensor,
    reduce: str = None,
) -> torch.Tensor:
    """Delegate to the registered NPU adapter.

    Signature mirrors proto.md `scatter(data, dim, indices, updates, reduce=None)`.
    Type annotations let TTK e2e introspect tensor vs scalar parameters so the
    CSV tensor_view_shapes map to (data, indices, updates) and attributes to
    (dim, reduce).
    """
    return torch.ops.glacier_bench_harness_scatter.forward(
        data, dim, indices, updates, reduce
    )


if __name__ == "__main__":
    torch_npu.npu.set_device(0)

    # One direct call (smoke) — verifies adapter loads and the aclnn API runs.
    # reduce=None (overwrite), dim=0 on a tiny 1-D tensor.
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu", dtype=torch.float32)
    indices = torch.tensor([0, 2], device="npu", dtype=torch.int64)
    updates = torch.tensor([10.0, 30.0], device="npu", dtype=torch.float32)
    y = forward(data, 0, indices, updates, reduce=None)
    print("direct_call_output:", y.cpu().tolist())
    torch_npu.npu.synchronize()
```

`tests/torch/ttk_cases.csv`（终版，20 用例 = 17 正向 + 3 L2 负向）：

```text
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,tensor_formats,attributes,output_tensor_indexes,input_data_ranges
bhs_torch_fp16_i32_update,test_torch.forward,"((8,8),(4,4),(4,4))","('float16','int32','float16')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp16_i64_update,test_torch.forward,"((8,8),(4,4),(4,4))","('float16','int64','float16')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp32_i32_update,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int32','float32')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp32_i64_update,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int64','float32')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_bf16_i32_update,test_torch.forward,"((8,8),(4,4),(4,4))","('bfloat16','int32','bfloat16')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_bf16_i64_update,test_torch.forward,"((8,8),(4,4),(4,4))","('bfloat16','int64','bfloat16')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_int32_i32_update,test_torch.forward,"((8,8),(4,4),(4,4))","('int32','int32','int32')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-50,50,),(0,7,),(-50,50,),)"
bhs_torch_int32_i64_update,test_torch.forward,"((8,8),(4,4),(4,4))","('int32','int64','int32')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-50,50,),(0,7,),(-50,50,),)"
bhs_torch_int64_i32_update,test_torch.forward,"((8,8),(4,4),(4,4))","('int64','int32','int64')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-50,50,),(0,7,),(-50,50,),)"
bhs_torch_int64_i64_update,test_torch.forward,"((8,8),(4,4),(4,4))","('int64','int64','int64')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-50,50,),(0,7,),(-50,50,),)"
bhs_torch_fp32_i32_dim1_update,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int32','float32')",,"{""dim"": 1, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp32_i32_dim1_add,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int32','float32')",,"{""dim"": 1, ""reduce"": ""add""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp32_i32_dim1_multiply,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int32','float32')",,"{""dim"": 1, ""reduce"": ""multiply""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp32_i32_dim1_amin,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int32','float32')",,"{""dim"": 1, ""reduce"": ""amin""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp32_i32_dim1_amax,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int32','float32')",,"{""dim"": 1, ""reduce"": ""amax""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp32_i64_negdim,test_torch.forward,"((4,8),(2,8),(2,8))","('float32','int64','float32')",,"{""dim"": -1, ""reduce"": ""add""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_fp16_i32_3d,test_torch.forward,"((4,6,8),(2,3,8),(2,3,8))","('float16','int32','float16')",,"{""dim"": 1, ""reduce"": ""amax""}",,"((-1,1,),(0,5,),(-1,1,),)"
bhs_torch_L2_dtype_mismatch,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','int32','float16')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_L2_indices_dtype_invalid,test_torch.forward,"((8,8),(4,4),(4,4))","('float32','float32','float32')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
bhs_torch_L2_rank_mismatch,test_torch.forward,"((8,8),(4,),(4,))","('float32','int32','float32')",,"{""dim"": 0, ""reduce"": ""update""}",,"((-1,1,),(0,7,),(-1,1,),)"
```

## 预防

- manifest 的 indices 值域必须显式落在 `[0, data.size(dim))` 内——随机生成无值域约束的整型索引会直接越界报错。
- 每轮重装 .run 包前先 `chmod -R u+w`；adapter 涉及 String 属性时按 char* 传参（std::string 成员 + data()）。

## 验证信号

以下信号来自该算子 Torch 路由搭建会话的实测。

- run.sh 全链路通过：`.run` 包构建、安装 SUCCESS、adapter 编译加载、直调输出 `direct_call_output: [10.0, 2.0, 30.0, 4.0]`（reduce=None 覆盖语义正确）、torch_route_ready 打印 adapter/vendor/golden 路径（来源平台：950；证据等级：tool_observed）。
- golden 一致性：NPU 直调与 golden 插件输出逐位一致（[10.0, 2.0, 30.0, 4.0]）（来源平台：950；证据等级：tool_observed）。
- eager 全量：20 用例全执行，17 正向 PASS（fp16/fp32/bf16 mere=0.0/mare=0.0，int32/int64 binary_equal），3 个 L2 负向按预期拒绝（"data and updates must have the same dtype" / "indices dtype must be int32 or int64" / "data/indices/updates must have the same rank"）（来源平台：950；证据等级：tool_observed）。
- aclgraph 全量（Meta 修复后）：与 eager 逐用例一致——17 正向 PASS、3 负向预期 GRAPH_EXEC_FAILURE（来源平台：950；证据等级：tool_observed）。
- 验收核对：两模式结果 CSV 均 0 个 NOT_RUN/N/A 记为 PASS、0 个 NOT_RUN/N/A 条目、20/20 用例可复核（来源平台：950；证据等级：tool_observed）。
- 命中本地安装包：aclnnBenchHarnessScatter/GetWorkspaceSize 符号在本地 vendor libcust_opapi.so 中存在（ctypes hasattr=True），CANN 系统 libopapi.so/libnnopbase.so 中计数 0，vendor 安装目录含 10 个 kernel .o（来源平台：950；证据等级：tool_observed）。
- manifest 一致性：CSV 覆盖 10 个合法 dtype 组合、reduce {update,add,multiply,amin,amax}、dim {-1,0,1}，与 spec.yaml 逐项一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 配方绑定"TORCH_LIBRARY adapter + TTK e2e"形态；GEIR/TF 路由、TTK aclnn 模式不适用（各有自己的 manifest 与执行方式）。
- update 覆盖模式在重复索引下结果非确定（spec 明确不保证写序）：本批小用例实测未出现分歧，但大 shape/高重复率下 golden 对比可能偶发 FAIL，属语义边界而非缺陷。
- TTK 3.0.0 实测；TTK 升级后 e2e 参数/解析链可能变化（未验证）。
- bf16 走 torch CPU golden 时原生支持；若换 float8/int4 等 torch 原生不支持的 dtype，golden 收参形态改变（numpy fallback），本配方未覆盖。

**未决项**：

- update 覆盖模式在重复索引下的 kernel 写序与 numpy last-write-wins 是否总是偶合一致未做机制验证：本批 update 用例的 indices 形状 (4,4)、值域 [0,8)，大概率含重复索引，但 17 个正向用例 mere/mare 全 0.0，未出现分歧；TTK 随机输入的种子与实际索引值未核查。
- reduce 缺省（None 默认路径）在 aclgraph 捕获下未单独验证（调试日志显示捕获调用显式传 'update'）。
## 坏实践

- 在 test_torch.py 里实现算子/golden 语义——被测件被污染，golden 应独立成插件。
- run.sh 里直接调用 TTK——违反"脚本内不调 TTK"的路由契约（TTK 执行属脚本外步骤，结果写测试目录外）。
- 不给 indices 配 input_data_ranges 就跑随机用例——越界报错掩盖真实回归信号。
- 把 L2 负向用例的 FAIL 当作缺陷修复目标——负向用例被拒（FAIL+明确错误）才是正确行为；反过来把 NOT_RUN/N/A 记为 PASS 是验收红线。

## 误诊

- torch_npu 的 `Cannot create tensor with interal format while allow_internel_format=False` 告警在直调与 TTK 运行中反复出现，不要当成失败去排查：警告文本本身为工具输出实测（tool_observed），"良性"判断为 agent 归纳——依据是所有出现该告警的用例最终全部 PASS（tool_observed）。
