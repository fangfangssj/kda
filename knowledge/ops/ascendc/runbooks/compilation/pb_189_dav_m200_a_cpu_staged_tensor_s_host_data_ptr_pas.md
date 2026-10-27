---
type: Runbook
title: "dav-m200 — a CPU-staged tensor's host `data_ptr()` passed to the kernel as a GM operand is read as GM SILENTLY: output all-NaN, kernel completes, zero error codes (scalar GM loads perform no pointer-provenance validation)"
description: "kernel 全程跑完、输出稳定全 NaN、零错误码——只有 CPU 侧来的操作数坏了（GM 标量读 corruption 特征）。"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.scalar, topic.performance.analysis, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-189", data_ptr]
---

# dav-m200 — a CPU-staged tensor's host `data_ptr()` passed to the kernel as a GM operand is read as GM SILENTLY: output all-NaN, kernel completes, zero error codes (scalar GM loads perform no pointer-provenance validation)

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any pybind launcher that can receive a CPU-side tensor operand — weights, biases, indices, probe-script tensors)`
`verified_on: 6_ConvStandard1d 310p kw-1 Phase D iter 1 (2026-09-08): CPU-staged weight/bias host pointers passed as GM → all-NaN output, no error; device-place fix cleared it same iteration`
`unverified_on: other SoCs (A5/351x, A3/220x) — the silent-garbage-on-host-ptr-as-GM behavior is assumed chip-family-wide but only observed on dav-m200`

- **Severity**: HIGH — silent numerical corruption with normal kernel completion; the all-NaN pattern does NOT track the input, so it reads as an "algorithm bug" first.
- **Symptom**: kernel executes fully (canary depth trace shows every phase completing), output is ALL-NaN (stable across runs, does not track input values), zero error codes from launch/runtime. Only the operands that arrived CPU-side are corrupt: in the observed case `xRaw_` reads correct (x was `.npu()`'d) while `wRaw_[0..8]` read as NaN.
- **Mechanism**: the launcher passed a CPU tensor's HOST virtual address where the kernel expects a device GM pointer. dav-m200 scalar GM loads validate no pointer provenance — the address is dereferenced as GM and yields garbage (NaN-class in the observed mapping), silently.
- **Fix (launcher discipline)**: EVERY tensor operand whose tensor can arrive CPU-side must be device-placed before launch — `weight.is_cpu() ? weight.to(x.device()).contiguous() : weight.contiguous()` — plus a `TORCH_CHECK(!x.is_cpu(), ...)` guard on the primary input. Landed: `代码片段 6_ConvStandard1d/pybind11.cpp.snippet.cpp（见文末）`.
- **Diagnosis path (5 probe builds, reusable as a recipe)**: 3-primitive sentinel (Duplicate/DataCopy/Axpy/scalar-GM-load each PASS in isolation) → canary depth trace (kernel completes — not a fault) → value dump (image correct, Duplicate correct, Axpy NaN) → scalar-source fork (literal OK, UB-staged literal OK, GM-read NaN) → raw pointer dump (wRaw_ NaN, xRaw_ correct) → pointer provenance. The step-4 fork (UB-staged vs GM-read of the same value) is what isolates the pointer path from the value path.
- **Evidence**: 6_ConvStandard1d 310p kw-1 Phase D iter 1 (2026-09-08) `6_ConvStandard1d/failures_ledger.md` precision-failures iter-1 row; fix cleared the all-NaN in the same iteration.
- **Other instances (predicted)**: any op whose ModelNew stages weights on CPU (RNG-replay capture tables, key-seeded edge weights, probe scripts) — the CAND-AMBIENT-RNG-IN-FORWARD-REPLAY workflow makes CPU-staged weights ROUTINE, so this trap co-occurs with it.
- **Cross-ref**: [`PB-124`](pb_124_on_dav_m200_multi_core_scalar_gm_stores_are_sile.md) (scalar GM loads are themselves reliable — the LOADS are fine, the PROVENANCE is the bug), [`PB-186`](pb_186_dav_m200_out_of_range_gm_reads_do_not_fault_a_sc.md) (the same silent-misaddress family, OOB flavor), [`EC-130`](ec_130_pybind11_launcher_a_const_at_tensor_operand_reje.md) (the `!is_cpu()` predicate + `std::optional` launcher surface this discipline lives in). backend=ascendc.

## 内嵌代码片段

### 6_ConvStandard1d/pybind11.cpp.snippet.cpp

```cpp
                      int64_t dilation, int64_t groups)
{
    auto xc = x.contiguous();
    TORCH_CHECK(!xc.is_cpu(), "conv_standard1d: x must be an NPU tensor");
    // Weight/bias may arrive CPU-staged (ModelNew's RNG-replay weight table,
    // pass_b key-seeded weights): moving them to x's device here is a pure
    // data-placement step (same class as contiguous) — the launch below reads
    // them as GM. BUG CLASS (failures_ledger, kw-1): passing a CPU data_ptr()
    // to the kernel silently reads HOST memory as GM — no error, all-NaN
    // output (probed: wRaw_[0..8] = NaN while xRaw_ reads correct).
    auto wc = !weight.is_cpu() ? weight.contiguous() : weight.to(xc.device()).contiguous();
    const bool hasBias = bias.has_value();
    at::Tensor bc;
    if (hasBias) {
        bc = !bias->is_cpu() ? bias->contiguous() : bias->to(xc.device()).contiguous();
    }

    TORCH_CHECK(xc.dim() == 3, "conv_standard1d: x must be 3-D (N, C_in, L)");
```
