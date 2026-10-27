---
type: Optimization
title: "Aligned-base scratch via `Duplicate(0) + scalar SetValue` for inline VEC reductions with unaligned index offsets"
description: "### Trigger Per-row VEC reduction kernel where the inner accumulation is Axpy(acc, src[runtime_offset], scalar, len) (or equivalent FMA chain), and the runtime offset is data-dependent and frequently"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.atomic]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p76-aligned-base-scratch-via-duplicate-0-scalar-setv.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-19T13:26:04Z"
platforms: ["950"]
---

# Aligned-base scratch via `Duplicate(0) + scalar SetValue` for inline VEC reductions with unaligned index offsets

> 原 OKF v1 卡号：**P-P76**。

## 优化点

### Trigger

Per-row VEC reduction kernel where the inner accumulation is `Axpy(acc, src[runtime_offset], scalar, len)` (or equivalent FMA chain), and the runtime offset is data-dependent and frequently NOT 32B-aligned. Canonical case: 1D/2D/3D conv with kernel_size > 1 and padding > 0 — the per-`kernel_position` offset `op_start = max(0, ceil_div(-(kp*dilation - padding), stride))` is typically 1/2/3 for the boundary positions. Other instances: pooling with stride > 1, dilated patches, scatter-with-runtime-offset, any reduction where the source slice base is data-dependent.

### The trap

VEC instructions on AIV require **32B-aligned bases** for both src and dst LocalTensor offsets (8 fp32 / 16 fp16 elements). The "natural" formulation:

```cpp
// ❌ Crash: error code 340 ("UB address not aligned")
// op_start can be 1/2/3 (small unaligned offsets) for boundary kernel positions
const int op_start = max(0, ceil_div(-base, stride));
const int in_start = max(0, base);
Axpy(accLocal[op_start], inLocal[in_start], w, len);
```

compiles fine, runs cleanly when `op_start = in_start = 0`, and crashes hard when either becomes non-aligned mid-loop. The crash is at runtime (error 340), not at compile time, so it's only caught by actually running on NPU.

### The pattern

Build an **aligned-base tmp scratch** via VEC zero-fill + scalar fill of the valid range:

```cpp
// ✅ Aligned-base — Axpy operates on len_pad with zero-padded inactive elements
const int Lo_pad = round_up_to_8(Lo);   // 32B align
TBuf<TPosition::VECCALC> tmpBuf;        // sized Lo_pad fp32

LocalTensor<float> tmp = tmpBuf.Get<float>();
Duplicate(tmp, 0.0f, Lo_pad);            // VEC zero, aligned

const int op_start = max(0, ceil_div(-base, stride));
const int op_end   = min(Lo, ceil_div(L - base, stride));
for (int op = op_start; op < op_end; ++op) {
    tmp.SetValue(op, in.GetValue(op * stride + base));   // scalar fill, no align constraint
}
SetFlag<HardEvent::S_V>(evSV); WaitFlag<HardEvent::S_V>(evSV);
Axpy(acc, tmp, w, Lo_pad);              // all bases aligned, single FMA op over Lo_pad
SetFlag<HardEvent::V_S>(evVS); WaitFlag<HardEvent::V_S>(evVS);
```

The zero-padded inactive elements (positions outside `[op_start, op_end)`) contribute `0 * w = 0` to `acc`, so they don't change the result. Cost: `~Lo` scalar SetValues per iteration of the outer loop — negligible on AIV vs the alternative of crashing or computing a fully aligned manual unroll.

### When to use vs alternatives

| Alternative | When to prefer |
|---|---|
| **Aligned-base scratch (this pattern)** | Inner reduction loop with data-dependent offsets, len ≤ a few KB, AIV path |
| Manual unroll of unaligned head/tail | Compile-time-known shape with mostly-aligned offsets and a small unaligned region |
| im2col + cube path | Throughput-bound reduction where the data-staging cost amortizes; offsets become loop indices not src bases |
| `DataCopy` with non-32B-aligned base | NEVER — same alignment requirement as VEC ops |

### Critical companion: output-side EC-23 workaround

For the same op family, the output is also typically not 32B-aligned (e.g. conv output length can be any positive int). Combined pattern:

```cpp
// pybind11.cpp: over-allocate output so each row is aligned
const int Lo_pad = round_up_to_8(Lo);
auto out = torch::empty({B, Cout, Lo_pad}, opts_f32);
// ... launch kernel, kernel writes [B, Cout, Lo_pad] via aligned DataCopy ...
return out.narrow(2, 0, Lo).contiguous();  // discard junk tail
```

This trades ~7 fp32 (28 bytes) per row of GM for crash-free output. Same trade-off principle as the input-scratch pattern: pay a small bounded overhead to satisfy alignment, instead of paying a large bounded overhead (or a crash) to handle unaligned offsets in-place. See **EC-23** for the underlying `DataCopyPad UB→GM crash error 507035` this works around.

### Determinism (by-construction)

The scalar-fill is sequential per-AIV; `Duplicate` is deterministic; `Axpy` is single-round FMA bit-aligned with `fmaf`. Reduction order over the outer accumulation loop is fixed by the for-loop structure. No atomic / no cross-AIV merge.

### Evidence

- **6_ConvStandard1d (2026-04-29)** — direct-VEC 1D conv via per-(batch, out_ch) AIV core. Phase D iter 1 hit error 340 with offset-Axpy formulation; switched to this pattern + EC-23 output pre-pad → 50/50 Pass A bit-exact at harness tolerance (atol=rtol=0.01) + 16/16 Pass B + det 50/50 + perf median 0.654× (≥ 0.6 threshold). 0 build + 1 precision iter total.

### Other instances (predicted)

- 7_ConvStandard2d / 8_ConvStandard3d — same kernel-size + stride + padding boundary pattern, same alignment issue. Direct port.
- 9_ConvDepthwise2d — depthwise variant, same trap (groups=Cin → Cin_per_g=1 → tighter weight-stride alignment too; weight loaded via scalar GetValue per kernel position).
- 10_ConvTranspose2d — output position arithmetic is different but boundary patterns still produce unaligned offsets.
- Any 1D/2D/3D pooling with stride > 1 + non-trivial padding.
- Any "scatter with runtime offset" pattern where the destination index is data-dependent.

### Combine with other patterns

- **OL-93** (multi-instance partition-dispatch): per-output-row dispatch with `blockDim = B × Cout` is the uniform-partition variant.
- **EC-23** (DataCopyPad UB→GM crash 507035): output over-allocation + narrow-on-return is the symmetric companion for write side.
- **Axpy bit-FMA semantics** (catalog): the inner accumulation is single-round FMA bit-matching `fmaf`, so accuracy is `≤ 1 ULP per accumulation step` regardless of `Lo_pad` vs `Lo`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P76，convert_patterns_to_okf.py）。confidence 未升格。 -->

## 更新 (2026-09-19)

私有沉淀仓 PATTERN_INDEX.md（pinned commit cb9c78c85ead3088398f3455051355349298b3cb）同号模式行复核：与 v1 卡同源同题，正文无新增技术内容；补录索引行元数据——domain: `platform_compat`；trigger: When an inline VEC reduction (`Axpy(acc, src[off], scalar, len)` or analogous) has a runtime index offset that's not 32B-aligned (e.g. conv `op_start = max(0, ceil_div(-base, stride))` is 1/2/3 for typical kernel_size+padding combos), the natural offset-Axpy formulation triggers UB-alignment crash error 340. Workaround: build an aligned-base tmp scratch by `Duplicate(tmp, T(0), len_pad)` then a scalar-loop `tmp.SetValue(op, src.GetValue(op*stride + base))` for the valid range, then `Axpy(acc, tmp, scalar, len_pad)` with all-aligned bases. Cost: ~`len_pad` scalar SetValues per (ic, kp) iteration — negligible on AIV vs the alignment-crash alternative. Combine with EC-23 output over-allocation pattern (pybind allocates `[B, *, len_pad]` instead of `[B, *, len]`, narrows on return). Validated op#6 ConvStandard1d (2026-04-29) — direct-VEC 1D conv via per-(batch, out_ch) AIV core. Likely-applicable to any inline reduction where the per-iter src offset is data-dependent and can be unaligned: 1D/2D/3D conv, 1D/2D pooling with stride>1, dilated patches.；优先级: HIGH。

> 索引行原文：`| P-P76 | **Aligned-base scratch via `Duplicate(0) + scalar SetValue` for inline VEC reductions with unaligned index offsets** | platform_compat | When an inline VEC reduction (`Axpy(acc, src[off], scalar, len)` or analogous) has a runtime index offset that's not 32B-aligned (e.g. conv `op_start = max(0, ceil_div(-base, stride))` is 1/2/3 for typical kernel_size+padding combos), the natural offset-Axpy formulation triggers UB-alignment crash error 340. Workaround: build an **aligned-base tmp scratch** by `Duplicate(tmp, T(0), len_pad)` then a scalar-loop `tmp.SetValue(op, src.GetValue(op*stride + base))` for the valid range, then `Axpy(acc, tmp, scalar, len_pad)` with all-aligned bases. Cost: ~`len_pad` scalar SetValues per (ic, kp) iteration — negligible on AIV vs the alignment-crash alternative. Combine with EC-23 output over-allocation pattern (pybind allocates `[B, *, len_pad]` instead of `[B, *, len]`, narrows on return). Validated op#6 ConvStandard1d (2026-04-29) — direct-VEC 1D conv via per-(batch, out_ch) AIV core. Likely-applicable to any inline reduction where the per-iter src offset is data-dependent and can be unaligned: 1D/2D/3D conv, 1D/2D pooling with stride>1, dilated patches. | **HIGH** |`
