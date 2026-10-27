---
type: "Runbook"
title: TTK e2e 按 api_name 注册 golden 插件
description: TTK e2e 模式按 api_name 解析被测函数（__import__ + inspect.signature 注解分类）、按同一 api_name 查 golden 插件（__spec__ 字典键必须是完整点分名如 test_torch.forward），golden 按被测函数签名参数序收 CPU 张量并返回 list。
tags: [task.testing, topic.implementation, topic.implementation.api_layer.high_level_api]
status: stable
sources: []
created_at: '2026-09-14T12:12:55Z'
updated_at: '2026-09-14T12:12:55Z'
platforms: ["950"]
---

# TTK e2e 按 api_name 注册 golden 插件

## 现象

为一个无内置 torch.* 对应 API、NPU-only 的自定义 aclnn 算子（scatter/scatter-reduce，原型 `forward(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor`）接入 TTK e2e 模式（`python3 -m ttk e2e`，torch_npu 端到端）做精度回归：CSV 的 `api_name` 填的是自定义暴露函数 `test_torch.forward`。此时 golden 生成默认链路不可用——TTK 的 golden 三级回退里第 3 级"同 API 在 CPU 执行"对只注册了 Meta + PrivateUse1 实现的自定义 op 必然失败；必须用 `--plugin` 提供 golden，且注册方式有多个非显然的机制点（api_name 解析、参数分类、golden 查找键、golden 收参形态）。本卡记录这条从 TTK 源码逐层核实并在本算子上完整验证过的接线机制。

## 适用场景

- 工程形态：TTK e2e（Framework API 模式）评测自定义 aclnn 算子——算子经 TORCH_LIBRARY adapter 暴露为 `test_torch.forward` 之类的自定义函数，需要 golden 对比。
- API 路径：TTK 3.0.0（`ttk e2e -i <csv> --plugin <golden.py> [--aclgraph]`）；内部链路 `framework_api/api_resolver.py` → `utilities/simple_param_extractor.py`（inspect.signature）→ `testcase_manager/param_plan.py`（build_args）→ `framework_api/golden_generation.py`（三级回退）→ `plugin_loader` + `test_spec/`（`__spec__` 字典）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（TTK 3.0.0）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——TTK e2e 对任意自定义算子 Torch 路由的 golden 接线，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：TTK aclnn 模式（api_name=aclnnXxx，golden 按算子名注册，不走本链路）；TTK 其他版本的解析实现（仅核对当前 checkout）；TF 后端（tf. 前缀走 resolve_callable_str，未验证）。

## 判别方法（机制链，逐层源码核实）

1. **api_name 解析**：`resolve_api("test_torch.forward")` 对非 `torch.Tensor.*`/`tf.*` 的名字做 `__import__("test_torch")` + 逐级 `getattr`。因此被测暴露函数所在模块必须可导入——`PYTHONPATH` 须包含 test_torch.py 所在目录（或以该目录为 cwd）。
2. **参数分类**：`simple_param_extractor._extract_params_from_inspect` 用 `inspect.signature` + 函数 `__annotations__` 把参数分类为 Tensor / int / str / bool 等（无注解时按参数名推断）。因此**暴露函数必须带完整类型注解**（`data: torch.Tensor, dim: int, indices: torch.Tensor, updates: torch.Tensor, reduce: str = None`），否则张量/标量分类失败、用例校验不过。
3. **参数组装**：`ParamPlan.build_args` 按**签名顺序交错**填充：CSV `tensor_view_shapes` 的张量按序填 tensor 形参，`attributes` 列按形参名填标量形参（如 `{"dim": 1, "reduce": "add"}`），带默认值的形参缺省时用默认值。张量与标量在签名中交错（data, dim, indices, updates, reduce）不影响组装。
4. **golden 三级回退**（`golden_generation.generate_golden`）：① CSV `golden_api` 列指定替代 golden API → ② `--plugin` 的 TestSpec 插件：`get_plugin_function(api_name, "golden", "e2e", plugin_path)` 经 spec_loader 按**模块级 `__spec__` 字典**解析，**键必须是完整 api_name 字符串**（本例 `"test_torch.forward"`；命名约定兜底只对纯 snake_case 算子名有效，点分 api_name 必须显式注册）→ ③ 同 API 在 CPU 执行（NPU-only 自定义 op 必失败）。`__spec__` 由 AST 静态扫描收集，必须是模块级赋值、值为类名字符串。
5. **golden 收参形态**：`_call_plugin_with_plan` 用与被测函数相同的参数计划调 golden——即 **golden 按被测函数签名的参数序接收 CPU torch 张量**（`plan.build_args(testcase.tensors)`），返回值经 `result_to_numpy` 转换（单输出返回 `[y]` 列表）；golden 带 `**kwargs` 时还会收到 `tensor_dtypes`/`testcase_name`/`backend` 等附加上下文。

## 设计决策

- golden 复用算子仓既有的**纯 numpy 参考实现**（`scatter_golden_impl`，经 importlib 按文件路径加载）而不是重写 torch.scatter 版 golden：与 aclnn 黑盒基线同一参考口径（fp16/bf16 的 add/multiply 在 fp32 内部累加后一次性舍入的"数学真值"模型），避免 torch 版 golden 在重复索引下与 numpy 版产生 golden 口径漂移。
- `tolerance` 随插件声明（fp16/fp32/bf16 `stat_rel_err`、int32/int64 `binary_equal`），与算子 spec 的 numerical_tolerance 对齐——TTK 按 per-output 路由使用 Spec.tolerance（需 `--plugin`）。
- 注册类用 `golden = staticmethod(golden)` + 模块级 `__spec__ = {"test_torch.forward": "BenchHarnessScatterTorchTestSpec"}`：点分 api_name 无法命中命名约定兜底（snake_to_pascal 只适用于纯算子名），显式注册是唯一路径。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0；TTK 3.0.0（/tmp/ops-test-kit，需在其根目录运行 `python3 -m ttk`）；torch 2.x + torch_npu。
- 目录布局：`<op>_package/tests/torch/`（test_torch.py + ttk_cases.csv + build/torch_golden.py，golden 由 run.sh 生成到 gitignored 的 build/）；golden 委托的 numpy 实现位于算子仓 `docs/develop/golden.py`。
- 相关文件：`tests/torch/test_torch.py`（被测暴露函数）；`tests/torch/build/torch_golden.py`（golden 插件）；`ttk_cases.csv`（api_name=test_torch.forward）。

## 完整代码

被测暴露函数（test_torch.py，完整；只加载 adapter、带注解委托、直调一次，不实现算子语义）：

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

golden 插件（build/torch_golden.py，完整；按 api_name 注册、按被测函数参数序收 CPU 张量、委托 numpy 参考）：

```python
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
```

## 预防

- 写 e2e CSV 前先确认三件事：暴露模块可导入（PYTHONPATH）、函数带类型注解、golden 插件 `__spec__` 键与 CSV `api_name` 逐字符一致（含点号）。
- api_name 必须是 `module.func` 形态（解析器对单段名直接报 Invalid api_name）。
- 相邻机制卡区分：TTK kernel 模式的 golden 查找键是 CSV `op_name`（蛇形算子名），与 e2e 模式的点分 `api_name` 查找键不同，见 [TTK 白盒 CSV op_name 蛇形与 golden 注册键不一致](ttk_csv_op_name_snake_case_golden_registration_key.md)；TestSpec golden 函数以未绑定方式被调用、不得带 self 形参，见 [TTK TestSpec golden 函数不得带 self 形参](ttk_testspec_golden_no_self_unbound_call.md)。

## 验证信号

以下信号来自该算子 Torch 路由搭建会话：golden 插件接线后的实测结果。

- golden 插件加载与注册正确：`spec keys: ['test_torch.forward']`；NPU 直调与 golden 输出逐位一致 `npu: [10.0, 2.0, 30.0, 4.0] golden: [10.0, 2.0, 30.0, 4.0]`（来源平台：950；证据等级：tool_observed）。
- eager 冒烟：`bhs_torch_fp32_i32_update` 单用例 EAGER PASS，TTK Summary 1/1 PASS（来源平台：950；证据等级：tool_observed）。
- eager 全量 20 用例：17 正向全 PASS（浮点 mere=0.0/mare=0.0、整型 binary_equal '100%'），3 个 L2 负向按预期被拒（错误信息即拒绝证据）（来源平台：950；证据等级：tool_observed）。
- aclgraph 全量（torch.compile 捕获 + 同一 golden）：同样 17 正向 PASS / 3 负向预期拒绝，与 eager 逐用例一致（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 机制读自 TTK 3.0.0 当前 checkout 源码并在其上实测；TTK 升级后 api_resolver/spec_loader 的实现可能变化（其他版本未验证）。
- golden 收到的是 **CPU torch 张量**（标准 dtype 下 needs_numpy_fallback=False）；若用例 dtype 是 torch 原生不支持的类型（float8/int4 等），插件会改收 numpy 数组，golden 需按 `use_numpy` 分支处理（本例未涉及）。
- `__spec__` 的 AST 扫描只认**模块级**赋值且值为字符串类名；类内/函数内声明或非字符串值不被索引。

**未决项**：

- TTK 对 L2 负向用例无"预期失败"原生标记机制（只能以 FAIL + 错误信息人工判读），TTK 是否有 negative-case 支持未验证。
## 坏实践

- 在 test_torch.py（被测暴露函数模块）里内联实现 golden 语义——污染被测件，且 `__spec__` 应随 golden 插件走而非随被测模块走。
- golden 插件只按 aclnn 形参序写签名（如 `golden(data, indices, updates, dim, ...)`）却不改注册键——参数序与被测函数签名不一致时按位错位，产生系统性错配而非报错。

## 误诊

- "自定义 op 没有内置 torch.* API 所以 e2e 无 golden 可用、只能不比精度"——错：`--plugin` 的 TestSpec 插件按 api_name 注册即可提供 golden（本卡路径），三级回退只是找不到插件时才落到 CPU 执行。
