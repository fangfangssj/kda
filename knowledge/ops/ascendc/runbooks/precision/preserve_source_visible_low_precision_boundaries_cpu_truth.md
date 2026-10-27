---
type: Runbook
title: "Preserve source-visible low-precision boundaries in CPU truth"
description: "Preserve source-visible low-precision boundaries in CPU truth"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-304"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-304 — OL-304: Preserve source-visible low-precision boundaries in CPU truth

## 适用场景

- 适用平台与范围：soc=all (host-side truth construction, not a silicon claim); cann=all; bisheng=n/a; op_class=graphs whose fp16/bf16 source exposes primitive or explicit-cast boundaries。
- 验证来源：2_FFN canonical eval (2026-07-25, 58 cases incl. 16 fp16 overflow-prone); 1_LSTM CPU-truth repair (2026-07-31, 60 cases); exact 34_NystromAttention source-literal BF16 adapter and retained fixed50 evidence (2026-09-14)。

**OL-304** — Widen an individual low-precision primitive only to the accumulator
dtype that the source contract gives it, then round exactly where that primitive
returns to the source-visible dtype.  Do not pre-upcast an entire graph merely to
make it CPU-runnable: doing so can erase intermediate rounding boundaries.  Keep
that compute rule separate from the oracle's storage dtype and the grader's
tolerance dtype.

- Former A5Ops ID: `OL-304`
- Fusion disposition: `additive_new`

`applies_to: soc=all (host-side truth construction, not a silicon claim); cann=all; bisheng=n/a; op_class=graphs whose fp16/bf16 source exposes primitive or explicit-cast boundaries`
`verified_on: 2_FFN canonical eval (2026-07-25, 58 cases incl. 16 fp16 overflow-prone); 1_LSTM CPU-truth repair (2026-07-31, 60 cases); exact 34_NystromAttention source-literal BF16 adapter and retained fixed50 evidence (2026-09-14)`
`unverified_on: genuine A3 behavior for 34_NystromAttention; any graph whose primitive accumulation contract is unknown; the unretained two-case Nystrom diagnostic output bytes`

**Decision rule (compute dtype and rounding boundary)**:

1. Read the source dtype at every primitive/module boundary and every explicit
   cast.  For a BF16 `Linear` whose contract is fp32 accumulation followed by a
   BF16 result, compute that `Linear` in fp32 and cast once to BF16 at its output.
2. Apply later source casts only after that boundary round.  A subsequent
   `.float()` does not retroactively license a fp32 `Linear`; it widens an already
   rounded BF16 value.
3. Whole-graph input promotion is valid only when it leaves every source-visible
   primitive and cast boundary unchanged.  Otherwise it is a semantic rewrite,
   not a CPU-truth portability shim.
4. Decide oracle storage separately.  The truth computation may need the source's
   intermediate rounds while the retained golden stays fp32 so the grader can
   compare genuine overflow positions.

For example, preserve two distinct BF16 `Linear` boundaries around a fp32
attention body:

```python
def native_bf16_linear(x, w):
    return linear(x.float(), w.float()).to(torch.bfloat16)
q = native_bf16_linear(x, wq).float()
attention = attention_body(q, k, v)  # source-explicit fp32 body
out = native_bf16_linear(attention.to(torch.bfloat16), wo)
```

The helper's internal `.float()` supplies the fp32 accumulator inputs; each
returned value still rounds once to BF16.

**Principle (oracle storage dtype vs the grader)**: the canonical grader's `compute_mere_mare` (precision_tier1.py) has NO Inf masking. An fp16-STORED golden carries genuine-overflow Infs; `inf − inf = NaN` → `ours_mere = cann_mere = NaN` → `NaN ≤ NaN = False` → every overflow case that can't pass strict T1 is an automatic FAIL regardless of vendor parity — the T2 leg is structurally dead. Keeping the oracle in **fp32 storage** for fp16-input cases makes T2 computable (`ours_mere=inf, cann_mere=inf` → `inf ≤ inf` parity pass) and is NOT a threshold loosening — the coarser-dtype tolerance table still applies fp16 thresholds. Companion rule for custom runners: key tolerance tables on the COARSER of (candidate, oracle) dtypes, not the oracle's. The grader is the shared Phase-O5 surface — do NOT patch it per-op; fix the oracle side.

**Evidence**: 2_FFN (2026-07-25): fp16-computed + fp16-stored oracle → 16/16 fp16 cases T2-dead (NaN) + eco_mare 0.03..81 oracle noise masking; fp32-compute + fp32-storage oracle → 14 grader-structural FAILs became adjudicated passes (v3 41/58 → v4 55/58 with 3 degenerate-draw skips).
- 1_LSTM (2026-07-31, a3/Ascend910_9382, CANN 9.0.0, benchmark mode): the same compute-in-fp32 rule applied to a MODULE reference. The vendor `model.py` (nn.LSTM) run on CPU crashes `mat1 and mat2 must have the same dtype` (Half≠Float — module weights fp32 vs fp16 inputs) → 32/60 fp16 cases EVAL_ERR, the entire fp16 axis ungradable. The Path-A `model_cpu_truth.py` oracle dtype-normalizes: upcast fp16 inputs → fp32 CPU nn.LSTM (compute in fp32, cast at storage) → EVAL_ERR 0 and the fp16 axis became gradable (canonical two-tier T1=34, T2=12). Confirms the principle transfers from hand-written matmul oracles to nn.Module references — and that a dtype-crashing vendor oracle is an oracle-design bug to fix (dtype-normalize), not a case set to drop.
- 34_NystromAttention (`output/npukernelbench/src/kernels/34_NystromAttention/accepted_a46cc6a1/model.py`, SHA-256 `029bb1b0ed987e66988132cba9b53e43667b6f61234df4eaacf9fe1debd88827`) constructs all four projection weights in the input dtype; q/k/v are widened only *after* their `Linear`, and the attention result is cast back before `out_proj`.  The source-literal adapter (`src/scripts/reference_provider/semantic_adapters/nystrom_attention_source_literal.py`, SHA-256 `c7cdbf4f6c59f87ed27a99f7189612079a980164f5dada08a6b6485b7390b9e5`) implements BF16 `Linear` as fp32 accumulation plus exactly one BF16 output round.  Its external vectors (`src/scripts/reference_provider/semantic_adapters/nystrom_attention_source_literal_conformance.json`, SHA-256 `de50283cbafc9c2ce49f27eebfcb50ae5868df108ef175f52b05533407448fae`) reject both an fp64-accumulation substitution and a whole-input-pre-upcast mutant in `src/scripts/reference_provider/tests/test_nystrom_attention_source_literal_adapter.py`.  The retained fixed50 receipt (`output/npukernelbench/src/kernels/34_NystromAttention/accepted_a46cc6a1/evidence/op34_controller_bf16_arm1_fixed50.json`, SHA-256 `7a7360c35be5478d29b5e281a08c9613697f0a58c9ce4eadb95c29195c0cd6a6`) binds the exact source, 50-case dataset, adapter and vectors; it reports 50/50 native controls and 50/50 strict-T1 cases with performance explicitly not run.  This is evidence for the truth rule, not a new product, performance, A3, selector, or KB-consumer closure.

**Causal negative control**: replacing the source-literal route with whole-input
pre-upcast makes the external `landmark_fp16_literal_chain` vector fail bitwise
conformance.  The separate fp64-accumulation mutant also fails.  Thus “use a wider
dtype” is not the rule; the accumulator width and every output-round boundary are
both load-bearing.

**Evidence boundary**: `diagnose_nystrom_bf16_accumulation.py` registered a
two-case prediction (case 12 sentinel; case 15 first-bad), but its run-output bytes
were not retained in the archive.  Those two diagnostic results are **UNKNOWN**
and are not load-bearing evidence for this entry.

**Other instances (predicted)**: attention projections, recurrent modules, or
other mixed-precision graphs where a low-precision module result is explicitly
widened for a middle computation and narrowed again before another module.  Each
prediction still requires its own source-bound primitive contract; do not infer
BF16 `Linear` semantics for a different primitive.

**Cross-ref**: OL-303 (the double-rounding this oracle design exposed), OL-262 (coverage gaps surface truth-source bugs — the degenerate-draw sibling), OL-104 / OL-89 (reference-as-truth methodology family). backend=ascendc.
