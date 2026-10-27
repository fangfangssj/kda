---
type: Guide
title: "Porting CUDA Benchmark Suites (BabelStream / STREAM / HPL-AI style)"
description: "CUDA benchmark 套件移植方法论：host C++ 按语义重写为 PyTorch harness（不逐字移植）、每个 kernel 独立 op-gen、真值来自 A100 CUDA 运行（含 nvcc 标志与 FMA 分组对齐）、case_gen 形状/分布覆盖矩阵强制。"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["agnostic"]
aliases: ["BabelStream", "HPL-AI"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Pattern: Porting CUDA Benchmark Suites (BabelStream / STREAM / HPL-AI style)

> 适用范围：源为 CUDA 时方法论按定义与目标平台无关；但 NPU 侧落地必须过目标平台能力门（a3/a2 为 SIMD-only）——platforms 取 agnostic，正文按此边界使用。

## 适用场景

- 适用平台与范围：soc=all（卡内 applies_to 块）——CUDA 多 kernel benchmark 套件移植方法论适用于任意 TARGET（来源为 CUDA 时），但 NPU 侧实现须遵守目标能力门（a3/a2 仅 SIMD）（卡内 reason 行）。
- 验证来源：卡内无 verified_on 行；codified 状态见正文 Status 行（BabelStream Triad pilot 2026-04-20；Dot pilot 验证后晋升 P-P63）。


**Status**: codified from BabelStream Triad pilot 2026-04-20. Promote to P-P63 after Dot (reduction) pilot validates the reduction-order variant.

## When to apply

Source is a **multi-kernel CUDA benchmark suite** (BabelStream, STREAM, HPL-AI, NPB-style) with:
- Kernel code in `.cu` / `.cuh` files
- Host-side driver in `main.cpp` doing init + iterate + time + verify
- Optionally a CMake build system
- No native PyTorch reference (PyTorch exists only as a pseudocode docstring if at all)

## Rule 1 — Host C++ code is NOT ported literally

BabelStream's `main.cpp`:
```cpp
for (size_t n = 0; n < num_times; n++) {
    auto t = std::chrono::high_resolution_clock::now();
    stream->triad();
    auto dt = std::chrono::high_resolution_clock::now() - t;
    timings.push_back(dt);
}
```

Do **not** translate this to AscendC C++. `cudaMalloc` / `cudaMemcpy` / `kernel<<<>>>` / `chrono` are runtime glue, not semantics. Replicate *semantics* (init buffers, iterate N times, measure time per iter, validate final state) as a **PyTorch Python harness on the A5 side**. Typical structure:

```python
# A5 harness (pytorch, NOT C++)
dev = torch.device("npu:0")
a = torch.full((N,), initA, dtype=torch.float32, device=dev)
b = torch.full((N,), initB, dtype=torch.float32, device=dev)
c = torch.full((N,), initC, dtype=torch.float32, device=dev)
scalar = startScalar

for _ in range(num_iters):
    torch.npu.synchronize(); t0 = time.time()
    a = model_new.triad(b, c, scalar)   # our AscendC kernel via pybind
    torch.npu.synchronize(); timings.append(time.time() - t0)
```

## Rule 2 — Each kernel is a separate `/ascendc-op-gen` invocation

Suite-level orchestration lives in the harness Python; each individual kernel
(Copy / Mul / Add / Triad / Dot / Nstream) is generated independently via
`/ascendc-op-gen path/to/<Kernel>.cu`. Pilot the simplest (pointwise) first,
then graduate to harder ones (reduction).

## Rule 3 — Ground truth comes from A100 CUDA run, not upstream bundled init values

BabelStream's upstream default is `startA=0.1, startB=0.2, startC=0.0`. For Triad
this makes `a = 0.2 + 0.4*0.0 = 0.2` — a constant value, no FMA rigor. Upstream
init is tuned for bandwidth-benchmark determinism, **not** numerical edge coverage.

Always extend with:
- `uniform` randn (typical case)
- `large_mag` / `small_mag` / `denormal` (dynamic-range paths)
- `cancellation` (worst-case rounding — `b ≈ -scalar*c` so `a ≈ 0`)
- `mixed_sign`, `scalar_zero`, `scalar_neg` (polarity / degenerate paths)

Reference is computed on **A100 with upstream-matching nvcc flags** (see Phase O2.5
checklist in `ascendc-op-gen/SKILL.md`). Compile flags matter:
- `--use_fast_math` implies `-ftz=true` → flushes denormals. Upstream BabelStream
  does NOT use this; don't add it.
- `--fmad=true` (nvcc default) enables fmaf contraction → single-round FMA. This
  is the reference behavior; kernel must align via `Axpy` (single-round) and not
  `Muls+Add` (two rounds).

## Rule 4 — Suite-level pilot first (simplest kernel before hardest)

Recommended pilot order for BabelStream-class suites:

1. **Triad / Add / Mul / Copy / Nstream** (pointwise, no reduction): validates
   Phase O2.5 reference pipeline. Expected: bit-exact PASS with `Axpy` primitive.
2. **Dot** (reduction): tests cross-platform reduction order. A100 CUDA uses
   block-reduce + host sum; AscendC uses per-core ReduceSum + inter-core add.
   Expected: NOT bit-exact at last 1-2 ULPs; need per-dtype tolerance OR match
   CUDA's tree-reduction structure in AscendC (deeper design question).

Do **not** port all 6 at once. If the Triad pilot shows unexpected issues, fix
them before burning worker budget on other kernels.

## Canonical primitive mapping

| CUDA pattern | AscendC idiom | Reason |
|---|---|---|
| `a[i] = b[i]` (Copy) | `DataCopy(a, b, count)` on queue tensors (NOT UB→UB — see PB-9) | literal |
| `c[i] = scalar * b[i]` (Mul) | `Muls(c, b, scalar, count)` | scalar broadcast |
| `c[i] = a[i] + b[i]` (Add) | `Add(c, a, b, count)` | pointwise 2-input |
| `a[i] = b[i] + scalar * c[i]` (Triad / FMA) | `Adds(a, b, 0.0f, count); Axpy(a, c, scalar, count)` | **Axpy = single-round hw FMA**, NOT Muls+Add |
| `a[i] += b[i] + scalar * c[i]` (Nstream) | `Adds(a, b, 0.0f); Axpy(a, c, scalar); Axpy(a, a_in, 1.0f)` | **rounding-order-matched**: seed with b → FMA s·c+b → add a_in. See §FMA grouping below for why the naive two-Axpy form drifts 1 ULP. |
| `sum = reduce(a[i] * b[i])` (Dot) | per-core: `Mul(prod, a, b, count); ReduceSum(partial, prod, count)`; host: sum partials | **Reduction order differs from CUDA** — expect 1-ULP drift, not bit-exact |

## Size caps for pilot

- Per-case tensor size ≤ 1M elements (4 MB fp32). Upstream default (e.g. BabelStream
  32M = 128 MB) is ONLY for performance stage, not precision pilot.
- Reasons: (1) WAN transfer between orchestrator local and A100/A5 is slow (~200 KB/s);
  (2) precision bit-exact at 1M already proves multi-tile multi-core paths.

## Shape coverage matrix (MANDATORY for sign_off tier)

The `case_gen` engine at `src/scripts/reference_provider/case_gen.py` auto-generates
shapes across these axes when `coverage_tier ∈ {sign_off, production}`. Do NOT
hand-curate shape lists — use the engine.

| Axis | sign_off shapes (A5, 56 cores, 32B-align) | Purpose |
|---|---|---|
| Degenerate | n ∈ {1, 7}                                       | fall-through paths, sub-alignment |
| Alignment boundary | n ∈ {8, 9}                                | 1 aligned unit, 1-tail |
| Core-partition boundary | n ∈ {55, 56*8=448, 56*8+3=451}      | fewer than cores / exact per-core / remainder |
| Tile boundary | n ∈ {4095, 4097, 8191}                         | ±1 around default UB tile (4096 fp32) |
| Prime / non-aligned | n ∈ {1009, 32771}                        | stress neither 8- nor 56-divisible |
| Typical | n ∈ {1024, 32768, 1048576}                         | 1 UB tile, 8 tiles, 256 tiles (multi-core, multi-tile) |
| 2D rank | [32,32], [64,16] (= 1024 flat)                     | flatness-equivalence check; catches pybind view bugs |

If your op class adds a NEW shape-relevant axis (e.g. reduction target length,
sort stability window), extend `case_gen._shape_plan` and add an axis row here.

## Distribution coverage matrix (MANDATORY for sign_off tier)

| Distribution | Purpose | Tier |
|---|---|---|
| `uniform` (randn) × n seeds | typical case, seed stability | all |
| `const_sanity`  | upstream-default init equivalent | all |
| `large_mag` [1e20, 1e30] | overflow / exponent alignment | all |
| `small_mag` [1e-30, 1e-20] | underflow | all |
| `denormal` [0.5–2× smallest-normal] | FTZ behavior | all |
| `mixed_sign` | add vs subtract path | all |
| `cancellation` (b ≈ -scalar·c) | FMA rounding worst case | sign_off+ |
| `const_near_zero` | boundary at IEEE zero | sign_off+ |
| `scalar_zero`, `scalar_neg` (via scalar probe) | degenerate scalar path | all |
| `sparse_nan`, `sparse_inf` (10% sprinkle) | NaN/Inf propagation | production |

Seeds per distribution: 1 (pilot) / 2 (sign_off) / 3 (production).

## Enforcement

`input_gen.py` MUST use `case_gen.generate_cases(SCHEMA, coverage_tier=...)`.
Hand-curated case lists are ANTI-PATTERN — the pilot's 12-case Triad list was a
pre-engine artifact and under-covered shape edges. Any future port MUST declare
schema + pick tier, NOT write a per-case loop.

`verification.json` MUST record `precision_coverage: {tier, n_cases, engine_sha}`.
A `tier=pilot` port that later needs sign_off MUST re-run through case_gen with
`tier=sign_off` and re-attest. "PASS at pilot tier" is not the same as "PASS at
sign_off tier".

## FMA grouping (P-P58.1 sub-pattern, bit-exact not reduction-only)

Discovered 2026-04-20 P.M. during Nstream port. Even for pointwise ops (no
reduction), the ORDER in which AscendC primitives are composed can diverge from
CUDA fmad=true's grouping — producing 1-ULP drift.

Example (Nstream, `a_out[i] = a_in[i] + b[i] + scalar * c[i]`):

- CUDA fmad=true compiles source `a[i] += b[i] + startScalar * c[i]` as
  `tmp = fmaf(s, c, b); a_out = a_in + tmp` — one FMA rounding (s·c fused with b)
  plus one add rounding.
- Naive AscendC two-Axpy: `Axpy(a, b, 1.0); Axpy(a, c, s)` computes
  `(a_in + b) + fma(s, c, 0)` — same two roundings, but GROUPED as `(a+b)+s·c`.
  Produces 4.77e-7 (≈ 1 fp32 ULP) drift on 29/34 sign_off cases.
- Correct AscendC: `Adds(a, b, 0.0); Axpy(a, c, s); Axpy(a, a_in, 1.0)` computes
  `a_in + (s·c + b)` — matches CUDA exactly, bit-exact 34/34.

**Rule**: when the CUDA source is `x = y + s*z + ... + u`, the compiler's
fmad=true pass will fuse the `+ s*z` into an fmaf with some neighboring addend.
AscendC kernel MUST use the same addend inside the FMA. Order primitives so:

1. Seed accumulator with the PARTNER-OF-FMA (not the outer-most operand).
2. FMA (`Axpy(acc, z, s)`) fuses s·z with the seeded value.
3. Final adds (`Axpy(acc, x, 1.0)`) bring in the remaining operands one at a time.

For 3-operand expressions `a + b + s*c` the CUDA compiler almost always picks
fmaf(s, c, b) (treating s*c as the "multiply-heavy" term worth fusing with its
arithmetically-adjacent addend). Port that way by default; the pilot's failed
naive form is the cautionary anti-pattern.

## Cross-refs

- `ascendc-op-gen/SKILL.md` Phase O2.5 — operational pipeline
- `ASCENDC_API_CATALOG.md` Axpy entry — single-round FMA note
- `precision.md` P-P58 — cross-platform precision RE (kicks in when this pattern's
  reduction-order cases drift); §P-P58.1 FMA grouping above is the bit-exact sub-case
- `PLATFORM_BUGS.md` PB-9 — UB→UB DataCopy silent corruption (why initial load uses
  `Adds(y, b, 0.0)` not raw DataCopy between UB buffers)
