---
type: Runbook
title: "`at::Tensor` → `py::object` return needs explicit `py::cast` at every return site"
description: "`at::Tensor` → `py::object` return needs explicit `py::cast` at every return site"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["py::cast"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-210 — EC-210: A5Ops additive extension

> 关联：同一 EC-210 主题已由既有卡 [ec_210_at_tensor_py_object_return_needs_explicit_py_cas.md](ec_210_at_tensor_py_object_return_needs_explicit_py_cas.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all（宿主侧 pybind11 包装层问题，与芯片无关）；op_class=任何入口声明为 `py::object` 的 pybind 包装（tensor / tuple / 异构空 tensor 等变体返回）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（22_Nonzero 310p build iter-2，2026-08-15）。

**EC-210** — **`at::Tensor` → `py::object` return needs explicit `py::cast`**: a pybind11 wrapper declared `py::object run_X(...)` (variant returns: tensor / tuple / differently-constructed empty) gets "no viable conversion" at return statements — wrap every return site with `return py::cast(t);`. 22_Nonzero 310p build iter-2 2026-08-15 (2 sites). Predicted: `as_tuple` variants, empty-rank early-exits, optional aux outputs. Cross-ref [OL-36](../precision/ol_36_npukernelbench_pytorch_wrapper_is_cann_delegation.md). backend=ascendc.

- Former A5Ops ID: `EC-210`
- Fusion disposition: `additive_new`

`applies_to: soc=all (host-side pybind11 wrapper concern, silicon-independent); cann=n/a; bisheng=n/a; op_class=any pybind wrapper with a `py::object`-typed entry point (variant returns: tensor / tuple / differently-constructed empty)`

- **Error**: no viable conversion from `at::Tensor` to `py::object` at the return sites of a wrapper declared `py::object run_X(...)` — needed when the function must return either a real tensor or an empty tensor built via a different path.
- **Fix**: `return py::cast(t);` at every return site.
- **Evidence**: 22_Nonzero 310p build iter-2 (2026-08-15): both return sites wrapped (pre-fix 内部路径（不公开）pybind11.cpp:87,183), build PASS (内部路径（不公开）workspace/22_Nonzero/kernel/pybind11.cpp:84,180 — the shipped `py::cast` sites; `workspace/22_Nonzero/failures_ledger.md` iter 2).
- **Other instances (predicted)**: any wrapper returning polymorphic results — `as_tuple` variants, empty-rank early-exits, optional auxiliary outputs.
- Cross-ref [OL-36](../precision/ol_36_npukernelbench_pytorch_wrapper_is_cann_delegation.md) (pybind is metadata+alloc+launch only — a boundary typing detail of that layer). backend=ascendc.
