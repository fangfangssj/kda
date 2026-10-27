---
type: Runbook
title: "a GM-delta RMW design must initialize the delta from the (widened) base tensor — zero-init + post-add merge co"
description: "scatter 的 GM-delta 必须从 widen 后的基值初始化，不能零初始化——零初始化改变浮点结合顺序，cancellation 重的 case 超 MARE 门限"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-547"]
---

# a GM-delta RMW design must initialize the delta from the (widened) base tensor — zero-init + post-add merge co

**OL-547** — **a GM-delta RMW design must initialize the delta from the (widened) base tensor — zero-init + post-add merge computes `x+(a+b)` per slot while every chained reference computes `((x+a)+b)`; on cancellation-heavy duplicate slots this exceeds MARE bars with MERE ~1e-10 / max_abs ~1e-6 (MARE-only T1 fail)**. Fix: widen kernel writes `D[i]=float(x[i])`, RMW accumulates from that, narrow rounds once → per-slot chain IS the reference chain (bit-probe 0 mismatches). Distinguish from the reference's own chained floor ([`OL-553`](ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md)/[`OL-109`](ol_109_two_tier_precision_verdict.md)) by the bit-diff: association divergence = ours≠ref; floor = ours==ref bit-exact. 21_Scatter 310p 2026-08-15 (case 24 mare 1.64e-3 vs 1.22e-3 bar → 4 fails became T1/T2). Cross-ref [`OL-553`](ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md), [`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md), [`OL-109`](ol_109_two_tier_precision_verdict.md), [`P-P204`](../performance/p_p204_regime_split_for_duplicate_writer_scatter_by_out.md). backend=ascendc.

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=any scatter-accumulate / duplicate-writer merge factored as a separate accumulator (delta/scratch) merged back into the base tensor`
`verified_on: soc=Ascend310P3; cann=9.0.0 (21_Scatter, 2026-08-15)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the association argument is fp-arithmetic-general, but the T1-bar interaction was measured on 310p only`

**Principle**: when a scatter-accumulate (or any duplicate-writer merge) is factored as "accumulate contributions into a scratch delta, then merge the delta back into x", the per-slot floating-point ASSOCIATION is a first-class precision decision, not a perf detail. A zero-init delta + post-add merge computes `x+(a+b)` per slot; every chained reference (aten CPU, aclnn NPU, sequential loop) computes `((x+a)+b)`. On cancellation-heavy duplicate slots that association difference blows past MARE-based tolerance bars while max-abs stays microscopic — invisible to an abs-diff glance and to MERE.

**Failure signature**: T1 fails MARE-only on a subset of duplicate-heavy cases (mare just over bar) with MERE ~1e-10 and max_abs ~1e-6. Distinguish from the reference's own chained floor (OL-553's bf16 note, OL-109 T2 territory) by the bit-diff: association divergence shows ours != reference with one-association-apart values on cancellation slots; the reference floor shows ours == reference bit-exact.

**Fix**: initialize the delta FROM the (widened) base tensor so the RMW chain IS the reference chain — a widen kernel writes `D[i] = float(x[i])` (not 0), every RMW accumulates from that value, narrow rounds once at the end. Per-slot result = `narrow(((x+a)+b))` — exactly the chained reference, bit-probe-verifiable.

**Concrete anchor** (21_Scatter delta regime, `21_Scatter/kernel/scatter_kernel.h:735,809 [snippet-unavailable: source workspace not on this box]`):
```cpp
ob.SetValue(i, WidenToFloat(xt.GetValue(i)));        // delta init FROM widened x — NOT zero
/* per-hit 32B-block fp32 RMW: D[blk..blk+7] += src[p] */
ob.SetValue(i, NarrowFromFloat<T>(dt.GetValue(i)));  // single final round to T
```

**Evidence**:
- 21_Scatter 310p (2026-08-15): zero-init delta failed T1 MARE on fp32 DELTA-regime cases (case 24: mare 1.64e-3 vs 1.22e-3 bar, mere 4.2e-10, max_abs 1.2e-6). Widen-init fix → `probes/delta_bitprobe.py` = 0 mismatches vs CPU-native chained loop on fp32/fp16/bf16; the 4 failing cases became T1/T2. Root-caused per OL-85 logic-first (chain divergence, not a data bug). `21_Scatter/failures_ledger.md:10`.

**Other instances (predicted)**: any "accumulate into scratch then merge" factorization — segmented reduction with base offsets, histogram-weighted recombination onto an existing tensor, multi-launch partial-sum buffers added back to x, embedding-backward accumulate into a pre-loaded (not zeroed) weight delta, fused-optimizer two-pass updates. Applies whenever the delta is fp32-widened for fp16/bf16 outputs: the WIDENED init must come from x, narrowing once at the end.

**Cross-reference**: [`OL-553`](ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md) (which chain the graded reference uses — init-from-x makes your chain the chained reference regardless of accumulate dtype), [`OL-85`](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md) (logic-first), [`OL-109`](ol_109_two_tier_precision_verdict.md) (the T2 rung for the OTHER signature: ours==ref bit-exact residual floor), [`P-P204`](../performance/p_p204_regime_split_for_duplicate_writer_scatter_by_out.md) (the DELTA-regime design this rule governs).

## 补充证据（2026-10-10 合并自 `gm_delta_rmw_init_delta_from_widened_base.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-547 — OL-547: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=any scatter-accumulate / duplicate-writer merge factored as a separate accumulator (delta/scratch) merged back into the base tensor。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (21_Scatter, 2026-08-15)。

**OL-547** — **a GM-delta RMW design must initialize the delta from the (widened) base tensor — zero-init + post-add merge computes `x+(a+b)` per slot while every chained reference computes `((x+a)+b)`; on cancellation-heavy duplicate slots this exceeds MARE bars with MERE ~1e-10 / max_abs ~1e-6 (MARE-only T1 fail)**. Fix: widen kernel writes `D[i]=float(x[i])`, RMW accumulates from that, narrow rounds once → per-slot chain IS the reference chain (bit-probe 0 mismatches). Distinguish from the reference's own chained floor (OL-553/OL-109) by the bit-diff: association divergence = ours≠ref; floor = ours==ref bit-exact. 21_Scatter 310p 2026-08-15 (case 24 mare 1.64e-3 vs 1.22e-3 bar → 4 fails became T1/T2). Cross-ref [OL-553](ol_553_scatter_accumulate_numeric_semantics_are_dtype_a.md), [OL-85](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md), [OL-109](ol_109_two_tier_precision_verdict.md), P-P204（A5Ops 绑定外部引用；本语料未收录目标）. backend=ascendc.

- Former A5Ops ID: `OL-547`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=any scatter-accumulate / duplicate-writer merge factored as a separate accumulator (delta/scratch) merged back into the base tensor`
`verified_on: soc=Ascend310P3; cann=9.0.0 (21_Scatter, 2026-08-15)`
`unverified_on: soc=Ascend950PR / Ascend910_220x — the association argument is fp-arithmetic-general, but the T1-bar interaction was measured on 310p only`

**Principle**: when a scatter-accumulate (or any duplicate-writer merge) is factored as "accumulate contributions into a scratch delta, then merge the delta back into x", the per-slot floating-point ASSOCIATION is a first-class precision decision, not a perf detail. A zero-init delta + post-add merge computes `x+(a+b)` per slot; every chained reference (aten CPU, aclnn NPU, sequential loop) computes `((x+a)+b)`. On cancellation-heavy duplicate slots that association difference blows past MARE-based tolerance bars while max-abs stays microscopic — invisible to an abs-diff glance and to MERE.

**Failure signature**: T1 fails MARE-only on a subset of duplicate-heavy cases (mare just over bar) with MERE ~1e-10 and max_abs ~1e-6. Distinguish from the reference's own chained floor (OL-553's bf16 note, OL-109 T2 territory) by the bit-diff: association divergence shows ours != reference with one-association-apart values on cancellation slots; the reference floor shows ours == reference bit-exact.

**Fix**: initialize the delta FROM the (widened) base tensor so the RMW chain IS the reference chain — a widen kernel writes `D[i] = float(x[i])` (not 0), every RMW accumulates from that value, narrow rounds once at the end. Per-slot result = `narrow(((x+a)+b))` — exactly the chained reference, bit-probe-verifiable.

**Concrete anchor** (21_Scatter delta regime, 内部路径（不公开）workspace/21_Scatter/kernel/scatter_kernel.h:735,809):
```cpp
ob.SetValue(i, WidenToFloat(xt.GetValue(i)));        // delta init FROM widened x — NOT zero
/* per-hit 32B-block fp32 RMW: D[blk..blk+7] += src[p] */
ob.SetValue(i, NarrowFromFloat<T>(dt.GetValue(i)));  // single final round to T
```

**Evidence**:
- 21_Scatter 310p (2026-08-15): zero-init delta failed T1 MARE on fp32 DELTA-regime cases (case 24: mare 1.64e-3 vs 1.22e-3 bar, mere 4.2e-10, max_abs 1.2e-6). Widen-init fix → `probes/delta_bitprobe.py` = 0 mismatches vs CPU-native chained loop on fp32/fp16/bf16; the 4 failing cases became T1/T2. Root-caused per OL-85 logic-first (chain divergence, not a data bug). 内部路径（不公开）workspace/21_Scatter/failures_ledger.md:10.

**Other instances (predicted)**: any "accumulate into scratch then merge" factorization — segmented reduction with base offsets, histogram-weighted recombination onto an existing tensor, multi-launch partial-sum buffers added back to x, embedding-backward accumulate into a pre-loaded (not zeroed) weight delta, fused-optimizer two-pass updates. Applies whenever the delta is fp32-widened for fp16/bf16 outputs: the WIDENED init must come from x, narrowing once at the end.

**Cross-reference**: OL-553 (which chain the graded reference uses — init-from-x makes your chain the chained reference regardless of accumulate dtype), OL-85 (logic-first), OL-109 (the T2 rung for the OTHER signature: ours==ref bit-exact residual floor), P-P204 (the DELTA-regime design this rule governs).

---
