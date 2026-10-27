---
type: Runbook
title: "pybind11 launcher: a `const at::Tensor&` operand rejects an explicit Python `None` (torch caster has no None conversion) — declare None-able operands `std::optional<at::Tensor>`; and `Tensor::is_npu()` does not exist in this torch — `!is_cpu()` is the portable device predicate"
description: "pybind 可选参数声明 `const at::Tensor&` 时传 Python None 会抛 caster 错误（要用 `c10::optional`）；且这个 torch 版本没有 `Tensor::is_npu()`。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-130", "None", "std::optional", "Tensor::is_npu"]
---

# pybind11 launcher: a `const at::Tensor&` operand rejects an explicit Python `None` (torch caster has no None conversion) — declare None-able operands `std::optional<at::Tensor>`; and `Tensor::is_npu()` does not exist in this torch — `!is_cpu()` is the portable device predicate

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=any pybind11 launcher with optional tensor operands (bias/weight/index) or a device-check predicate`
- **Symptom**: a launcher declaring an optional operand `const at::Tensor&` throws a pybind caster error when the caller passes an explicit Python `None` (the torch caster has no None→Tensor conversion). Separately, a device guard calling `x.is_npu()` fails to build — `is_npu()` is not defined on this torch's `Tensor`.
- **Root cause**: the `at::Tensor` pybind caster binds only real tensors, not `None`; and this torch_npu `Tensor` API exposes `is_cpu()`, not `is_npu()`.
- **Fix**: declare None-able operands `std::optional<at::Tensor>` (unpack with a guard / `.value_or()` before use); and gate device placement with `!x.is_cpu()` — the portable predicate — rather than `is_npu()`.
- **Evidence**: 6_ConvStandard1d 310p kw-1 iters 1-2 (2026-09-08) — the launcher surface whose discipline PB-93 documents.
- **Other instances (predicted)**: any port mirroring a CUDA op's optional-parameter surface (bias/weight/index operands), and any device-check that assumed `is_npu()` exists.
- **Cross-ref**: PB-93 (the CPU-staged-GM all-NaN discipline — this entry defines the `!is_cpu()` + `std::optional` launcher surface that discipline lives in), PB-78 (scalar GM loads are reliable). backend=ascendc.
