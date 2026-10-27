---
type: Runbook
title: "A geometric-mean speedup gate constrains neither the aggregate runtime nor the per-case distribution — a pass can hide a slower aggregate and regressed cases"
description: "A geometric-mean speedup gate constrains neither the aggregate runtime nor the per-case distribution — a pass can hide a slower aggregate and regressed cases"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-443"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-443 — OL-443: A geometric-mean speedup gate constrains neither the aggregate runtime nor the per-case distribution — a pass can hide a slower aggregate and regressed cases

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0。

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0`

**Principle.** Accepting performance on `geomean_i(reference_i / candidate_i) >= 1.0` is a statement
about the *central tendency of per-case ratios* and nothing else. Three quantities that readers
routinely assume it implies are all left free:

1. **The aggregate.** `Σ reference / Σ candidate` can be below 1.0 while the geomean is above it.
   The geomean weights every case equally; the aggregate weights each case by its runtime. A
   candidate that wins on many cheap cases and loses on a few expensive ones passes the gate and
   makes the total workload slower.
2. **The per-case distribution.** The gate says nothing about how many cases regressed, or by how
   much. A geomean of 1.09 is compatible with roughly half the cases below 1.0.
3. **Any production workload.** Benchmark cases are supplied at equal weight. Unless the case
   weights match the intended workload, neither statistic is a claim about it.

None of this makes the geomean wrong — it is a deliberate choice that resists domination by one
large case. It makes a geomean quoted *alone* an incomplete acceptance record.

**Rule.** When reporting or accepting a performance result, state the geomean, the aggregate ratio
and the count of cases at or above 1.0 together. If only one is available, say which quantity the
verdict is about. Do not describe a geomean pass as "faster" without qualification.

**Concrete anchor.** From a single measured window, all three quantities on the same 50 cases:

```
geometric mean (ref/cand)   1.0885    <- passes a >= 1.0 gate
aggregate ratio (Σref/Σcand) 0.8591   <- the workload as a whole is ~16% SLOWER
cases at or above 1.0        24/50    <- about half regressed
```

### Evidence
- 4-13 ScaledDotProductAttention row-sum arm, 2026-09-11 (Ascend950DT_9582, CANN 9.1.0, msprof
  device-time, 50 benchmark-supplied cases): the three figures above were produced by one run on one
  card and are recorded together in the archive's own result note
  (`output/npukernelbench/src/kernels/13_ScaledDotProductAttention/row_sum_9a91d1bf_20260911/RESULT.md`). The author stated the limit explicitly — "passing the geometric-mean criterion
  does not imply every case or aggregate workload is faster" — and also recorded the run as
  SINGLE_RUN with spread UNMEASURED. The arithmetic relationship is a property of the statistics,
  not of this operator or this silicon; the row is the witness, not the derivation.
- In-repo consumer, 2026-09-12: the benchmark-acceptance block forwarded `geomean_speedup`,
  `mean_speedup`, `n_cases_total`, `n_cases_valid` and `timing_method` and nothing else, so a reader
  of that block could not tell a candidate that won 8 of 10 cases while doubling the total runtime
  from one that won everywhere. `_run_acceptance_performance` in
  `src/scripts/orchestrator/benchmark_acceptance.py` (computed in
  `src/scripts/orchestrator/speedup_stats.py`) now also carries `aggregate_speedup`,
  `n_cases_ge_1x` and `speedup_stats_basis` (the one case set all three are measured over), derived
  from the per-case rows the perf script already wrote — or, when no usable per-case row exists,
  `speedup_stats_unavailable` with the reason, so the geomean is never left standing alone without
  one. Reporting only: no threshold, verdict or numerator moved.

### Other instances (predicted)
- Any acceptance that quotes a single ratio: per-case geomean, total-time ratio, or a median. Each
  constrains only itself.
- Any comparison where case cost is skewed — long-sequence attention, wide-K GEMM, variable-length
  batches — where equal-weight and runtime-weighted summaries diverge most.
- Any report that carries a geomean forward into a claim about a production workload whose case
  weights were never bound.

### Cross-references
- OL-230 (batching a per-row reduction) — the operator-side lever whose measurement surfaced this;
  its evidence row carries the same three figures for exactly this reason.
