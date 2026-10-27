---
type: Runbook
title: "Reference dispatch and dtype define performance comparison groups"
description: "Reference dispatch and dtype define performance comparison groups"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-580"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-580 — OL-580: Reference dispatch and dtype define performance comparison groups

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=matmul profiling。
- 验证来源：MatmulBothTrans TT, Ascend950DT_9582, CANN 9.1.0, historical run fd3c8cce-de33-4b13-9736-1338f69f5966。

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=n/a; op_class=matmul profiling`
`verified_on: MatmulBothTrans TT, Ascend950DT_9582, CANN 9.1.0, historical run fd3c8cce-de33-4b13-9736-1338f69f5966`
`unverified_on: genuine A3 execution; another CANN/runtime, input population or rebuilt candidate`

Before explaining a candidate/reference speed ratio, bind the reference implementation actually
executed for each shape and dtype. An operator name and a shared mathematical expression do
not identify a single implementation or a uniform performance target. Partition the measured
rows by observed dispatch and dtype, and retain both reference and candidate times.

The retained 50-case TT workload computes A[K,M]^T @ B[N,K]^T. Its profiler data reports
48 reference MatMulV3 cases and two Mul cases. The scalar shapes demonstrate why dtype is
part of the reference identity; the three large shapes show that a common dispatch label
also does not imply a common latency.

| Case | K/M/N | dtype | Reference dispatch | Reference median us | Candidate median us | ref/candidate |
|---|---|---|---|---:|---:|---:|
| 25 | 1/1/1 | float32 | Mul | 1.276 | 1.833 | 0.696127 |
| 26 | 1/1/1 | float16 | Mul | 1.28 | 1.837 | 0.696788 |
| 27 | 1/1/1 | bfloat16 | MatMulV3 | 2.474 | 1.836 | 1.347495 |
| 43 | 512/1024/768 | float32 | MatMulV3 | 38.945 | 50.036 | 0.778340 |
| 44 | 512/1024/768 | float16 | MatMulV3 | 7.396 | 11.621 | 0.636434 |
| 45 | 512/1024/768 | bfloat16 | MatMulV3 | 7.213 | 11.398 | 0.632830 |

Positive use: distinguish the measured scalar Mul/reference versus mixed-core candidate
comparison from a large MatMulV3 case before selecting a performance experiment.
Counterexample: scalar BF16 uses MatMulV3 here, so a rule routing every 1x1 matmul to Mul
is false. All three large rows use MatMulV3, so equal kernel names do not justify sharing
the FP32 reference latency with FP16/BF16.

These observations do not prove a launch-cost lower bound, identify the cause of the
large-shape gap, exclude a single tile strategy, or justify stopping optimization. No
pipeline, split-K, wider-BN or NZ-layout benefit is established by this table. A changed
source/DSO requires its own full-denominator precision and performance comparison.

Evidence and scope:

- Exact source/archive inspection: A5Ops `078016f022c85c8c6f0ffd4450fd887e6de67533`.
- At that immutable historical exact, 内部路径（不公开）output/npukernelbench-tilelang/src/kernels/3_MatmulBothTrans/retained_versions/kimi_v2_1a0eea33_current50_20260908/REPORT.md:18-22 binds the run, candidate, model, cases and captured-input/truth digests; `:31` identifies its original-dtype NPU reference. The `ext:` marker records that this path is not present in the current tree.
- At the same exact, 内部路径（不公开）output/npukernelbench-tilelang/src/kernels/3_MatmulBothTrans/retained_versions/kimi_v2_1a0eea33_current50_20260908/REPORT.md:23-25 binds the profile, hardware and runtime image.
- At the same exact, 内部路径（不公开）output/npukernelbench-tilelang/src/kernels/3_MatmulBothTrans/retained_versions/kimi_v2_1a0eea33_current50_20260908/REPORT.md:36 states five warmups/five measured samples per implementation/case, sample-major ordering, kernel_details Duration(us) and operator_details attribution.
- `output/npukernelbench-tilelang/src/kernels/3_MatmulBothTrans/retained_versions/kimi_v2_1a0eea33_current50_20260908/evidence/FINAL_RECOMPUTED.json` SHA256 `4c0f40961f419a13598fc39c3e4102ac8867bd7f95923a196e2f77491f8ab9e8` carries the 50 case identities, raw-member digests and 50 mapped DSOs.
- `output/npukernelbench-tilelang/src/kernels/3_MatmulBothTrans/retained_versions/kimi_v2_1a0eea33_current50_20260908/evidence/F_RAW.tar.gz` current Git SHA256 `c3ab9fe73d6869a32c58c8c0102898bd572814300d1ee7f716310b6d0e39c2a3` differs from the original outer-tar digest recorded in FINAL_RECOMPUTED. The cause of that packaging difference is not inferred. All 1,000 CSV members match their recorded digests; all 500 raw-duration sums and 100 sample medians were recomputed in this source-only audit.
- Full 50 historical GM `1.2566552196581078`, sum-of-case-medians ratio `1.057331328515303`, 42/50 ratios>=1 are separate summaries (OL-443), not a new measurement, formal O5 result or a claim about another workload.

Dedup: OL-328 discusses fixed-cost versus compute-dominated profiling; this note adds
observed dispatch/dtype partitioning without declaring which cause dominates. OL-443
owns aggregate-statistic interpretation. Neither is extended to a new hardware claim.
