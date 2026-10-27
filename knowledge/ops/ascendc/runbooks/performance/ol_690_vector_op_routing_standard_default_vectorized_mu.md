---
type: Runbook
title: "Vector-op 性能为准 routing standard — default vectorized + multi-core; scalar/单核只在\"实测更快\"才允许 (VECTOR_REGIME gate)"
description: "VECTOR 类算子默认每个 case 都向量化+多核；只有同设备 A/B 实测证明标量/单核更快才允许降级，且不许为绕开 vector bug 而降单核（必须根因修掉）"
tags: [task.optimization, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar, topic.performance.analysis]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-690"]
---

# Vector-op 性能为准 routing standard — default vectorized + multi-core; scalar/单核只在"实测更快"才允许 (VECTOR_REGIME gate)

## 优化点

`applies_to: soc=Ascend310P3 (dav-m200) and all targets; cann=n/a; bisheng=n/a; op_class=AscendC VECTOR-class ops (elementwise / fused quant-activation / row-parallel chains)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (owner mandate 2026-09-03 — applies to swi_glu_quant + dequant_swiglu_quant archives rework and ALL future vector ops)`

**Principle**: an AscendC VECTOR-op delivery defaults to **vectorized + multi-core for EVERY case**. A case may be routed to scalar and/or single-core ONLY where a same-device, back-to-back A/B measurement shows the scalar/single-core path is FASTER than the vector+multi-core path for that exact shape. There is NO fixed numel/width "small-shape" shortcut; the exemption is perf-based and must carry evidence.

**Forbidden** (anti-patterns):
- scalar-because-precision-passes — PASS against the golden is not a license for scalar;
- routing non-exempt cases to scalar/single-core to dodge a vector bug (e.g. the `E%64!=0` single-core fallback in swi_glu_quant, motivated by the unverified CAND-DAV-M200-TAIL-LANES-E32MOD64 — routing around it is now illegal; it must be root-caused and the vector tail FIXED);
- whole-op scalar/single-core default (dequant_swiglu_quant 2026-09-02 shipped all 29 cases single-core scalar at ~0.2 s/case — the OL-127 nblk=1-final-state trap on dav-m200);
- "can't vectorize this shape" as a scalar license — an un-buildable vector path is a root-cause blocker, not an excuse.

**Exemption details**: M==1 single row exempts multi-core (no rows to split) but the row must still be vectorized unless A/B shows scalar faster. Every scalar/single-core-routed case needs a recorded `scalar_ab` (scalar_us vs vector_us, same device, back-to-back, winner=scalar).

**Harness enforcement**: archive verification.json carries a per-case `regime` table {vectorized, cores, entry}; finalize PASS-branch gate `VECTOR_REGIME` rejects any non-vector or cores<=1 case lacking scalar_ab, and rejects a kernel with no vector+multi-core entry at all. Full rule text (must-read, unconditional) is in `shared/ALWAYS_LOADED_RULES.md` OL-690.

**Evidence**:
- swi_glu_quant + dequant_swiglu_quant archives rework (owner mandate 2026-09-03) — the originating instances (see `verified_on` above).
- 30_TimeDecayExponentialStabilization (2026-09-06, dav-m200 benchmark) — compliance instance on a SEQUENTIAL-SCAN op, extending the rule's scope reading beyond elementwise/row-parallel chains: the sequential axis (S timesteps) lives INSIDE the vector lane loop while the independent B×H channel space tiles across all 7 vector cores (channel-tile grid, ~14-tile ladder; see CAND-DAV-M200-CHANNEL-TILE-SCAN). No scalar or single-core fallback triggered on any benchmark/edge case; per-case regime recorded in verification.json. Perf (kw-2 re-grade on op-def-faithful inputs): 64.03× mean / 48.60× median symmetric device-event, worst case 5.16× — kw-1's 60.49× mean / 4.69× worst was measured on a diverged randn surrogate and is superseded; the 8.41× vendored-asymmetric figure was taken pre-restoration on the data-oblivious kernel and stands.
