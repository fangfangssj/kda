---
type: Runbook
title: "Pybind module import fails with a torch symbol — link libtorch_python before diagnosing ABI"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; torch=2.9.0+cpu; op_class=all"
tags: [task.debugging, topic.precision, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-55-pybind-module-import-fails-with-undefined-symbol-t.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:00Z"
platforms: [a3]
aliases: ["EC-55"]
---

# Pybind module import fails with a torch symbol — link libtorch_python before diagnosing ABI

> 原 OKF v1 卡号：**EC-55**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - pybind module builds successfully but import <module> fails with undefined symbol: _ZTVN5torch8autograd12AutogradMetaE and/or undefined symbol: pybind11::detail

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382; cann=9.0.0; torch=2.9.0+cpu; op_class=all`
`verified_on: soc=Ascend910_9382; cann=9.0.0; torch=2.9.0+cpu`
`unverified_on: soc=Ascend950PR (verify the installed torch build before assuming the same ABI)`

(ds agent extraction 2026-05-19 from earlier ds-branch commit e13d49db; originally numbered EC-47 in ds branch but main now has different EC-47 — renumbered to EC-55.)

- **Error pattern**: the pybind module links successfully, but `import <module>`
  fails with a symbol such as `_ZTVN5torch8autograd12AutogradMetaE` or
  `pybind11::detail::type_caster<at::Tensor>::cast`.
- **First diagnosis**: the `type_caster<at::Tensor>` symbols are provided by
  `libtorch_python.so`; the `AutogradMeta` vtable is defined by PyTorch's core
  `libtorch_cpu.so`, normally reached through `libtorch.so`. Missing `torch`
  and/or `torch_python` from the extension's link set is therefore a
  link-configuration defect, not evidence of an ABI mismatch.
- **Fix and diagnosis order**:
  1. Inspect the generated `target_link_libraries`. The current `torch::Tensor`
     binding contract requires `torch`, `torch_python`, and `torch_npu`.
  2. Verify the selected torch library directory contains
     `libtorch_python.so*`. If it does not, report the environment dependency as
     missing; do not disguise it with a different binding contract.
  3. Fix the persistent `build_ascendc.py` generator, delete the affected build
     directory, and rebuild. Do not patch only an auto-generated CMake file.
  4. Use `readelf -d <module>.so` to inspect `DT_NEEDED`. Run `ldd -r` in the
     runtime environment and filter for the target torch symbols; a standalone
     `ldd -r` can legitimately report Python C-API symbols that the interpreter
     supplies. The decisive check is a fresh interpreter that imports `torch`
     and `torch_npu` before importing the extension.
  5. Only if all three libraries are linked and their symbols resolve should an
     incompatible torch/torch_npu/extension ABI pairing be investigated.
- **Current contract**: generated bindings use `torch::Tensor`. Do not replace
  them with raw Python-object operations or `py::object`; that bypasses the
  worker hooks and changes the runtime interface.
- **Corrected evidence (2026-07-30)**: a backward-generation extension initially
  failed on `AutogradMeta` while its generated CMake linked only `torch_npu`.
  Linking `torch`, `torch_python`, and `torch_npu`, then rebuilding cleanly,
  produced an importable module with resolved dependencies.
- **Historical evidence, not a current recommendation**: op#3 Add (2026-05-11)
  bypassed the tensor binding with raw pybind11/Python operations and imported,
  but only 44/50 precision cases passed. That single run does not establish an
  ABI root cause and its workaround is outside the current contract.

<!-- 迁移自 porter kb/target/ascendc/（EC-55，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
