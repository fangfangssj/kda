---
type: Runbook
title: "CANN aclnnIncreFlashAttention (torch_npu.npu_incre_flash_attention) aicore-TIME-OUTS after ~8 sequential invoc"
description: "Ascend310P3 上 aclnnIncreFlashAttention 连续调用约 8 次后 aicore 超时挂死——厂商 op 坑"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-111"]
---

# CANN aclnnIncreFlashAttention (torch_npu.npu_incre_flash_attention) aicore-TIME-OUTS after ~8 sequential invoc

**PB-111** — **CANN `aclnnIncreFlashAttention` (`torch_npu.npu_incre_flash_attention`) aicore-TIME-OUTS after ~8 sequential invocations on Ascend310P3** (persists device 0+1; CANN ref runs cases 0–6 then times out 7+). Likely a per-invocation resource leak (invocation-monotonic, recovers only on process reset). T2-vs-CANN is bounded to the cases the ref runs — precision still fully verified via T1 golden truth (910B2C). Distinct from [`PB-15`](pb_15_parallel_npu_launches_on_same_docker_container_pro.md) (cross-process concurrency). Do NOT loop the CANN ref unbounded in a verify harness — wrap with a reset/retry budget and treat monotonic timeout as THIS bug, not a kernel bug. backend=all. incre_flash_attention 2026-08-11. Cross-ref [`PB-15`](pb_15_parallel_npu_launches_on_same_docker_container_pro.md), [`OL-97`](../precision/ol_97_precision_target_cpu_truth_vs_cann_bit_match.md).

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=attention (incre_flash_attention CANN reference path)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu (aarch64 build)`
`unverified_on: soc=Ascend950PR, Ascend910 (may or may not reproduce — re-verify before assuming the CANN op is stream-stable on other SoCs)`

**Symptom**: calling `torch_npu.npu_incre_flash_attention` (CANN `aclnnIncreFlashAttention`) repeatedly on Ascend310P3 aicore-**time-outs after ~8 invocations**; the failure persists across device 0 and device 1. The CANN reference runs cases 0–6, then times out from case 7+. The hand-written AscendC kernel runs all 27 cases cleanly when the CANN reference is NOT invoked on the same process.

**Mechanism (hypothesis, not yet isolated)**: likely a per-invocation resource leak (events / scratch / stream-internal allocation) inside the CANN incre-FA op that accumulates until the aicore watchdog fires. Falsifiable claim: the timeout count is invocation-monotonic (~8), independent of input shape, and recovers only on process/device reset — NOT on retry within the same process.

**Impact on verification (measurement limit, NOT a pass-gate relaxation)**: the T2 comparison (our kernel vs the live CANN reference) cannot complete for all 27 cases on this box; T2 is bounded to the cases (0–6) where the CANN reference still runs. Precision is STILL fully verified via T1 (golden truth captured on Ascend910B2C) for all 27 — T1 is the load-bearing verdict here, T2 is a sanity cross-check on the subset that runs.

**Distinct from PB-15**: PB-15 is cross-PROCESS concurrent-launch stream-state corruption (parallel `docker exec &`); PB-111 is single-PROCESS sequential repeated-invocation timeout. Different signature — PB-15 = transient magnitude drift under concurrency; PB-111 = monotonic aicore timeout under sequential reuse of one CANN op.

**Workaround (current)**: rely on T1 golden-truth verification for full coverage; use the CANN reference only for the subset of cases it runs as a T2 cross-check. Do NOT loop `npu_incre_flash_attention` unbounded in a verification harness — wrap it with a per-call reset/retry budget and treat invocation-count-monotonic timeout as THIS bug, not a kernel-correctness bug.

**Evidence**: incre_flash_attention (2026-08-11, Ascend310P3 / dav-m200, CANN 9.0.0) — CANN ref times out case 7+ on both device 0 and device 1; our kernel 27/27 on T1 golden truth (`verification.json.precision`: 14/27 strict PASS, all `max_abs ≤ 0.000977`; `method` records "T2 attempted but CANN ref matches golden exactly (mere=0) for cases 0-4, so T2 cannot bridge gap").

**Cross-ref**: [`PB-15`](pb_15_parallel_npu_launches_on_same_docker_container_pro.md) (concurrency stream-state corruption — different mechanism), [`OL-97`](../precision/ol_97_precision_target_cpu_truth_vs_cann_bit_match.md) (CPU-truth vs CANN-bit-match target choice — here T1/golden-truth is the viable target precisely because T2 is reference-limited).

## 补充证据（2026-10-10 合并自 `aclnn_incre_flash_attention_sequential_invocation_timeout_310p3.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## PB-111 — PB-111: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=attention（incre_flash_attention CANN 参考路径）。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0；torch_npu（aarch64 构建）。

**PB-111** — **CANN `aclnnIncreFlashAttention` (`torch_npu.npu_incre_flash_attention`) aicore-TIME-OUTS after ~8 sequential invocations on Ascend310P3** (persists device 0+1; CANN ref runs cases 0–6 then times out 7+). Likely a per-invocation resource leak (invocation-monotonic, recovers only on process reset). T2-vs-CANN is bounded to the cases the ref runs — precision still fully verified via T1 golden truth (910B2C). Distinct from PB-15 (cross-process concurrency). Do NOT loop the CANN ref unbounded in a verify harness — wrap with a reset/retry budget and treat monotonic timeout as THIS bug, not a kernel bug. backend=all. incre_flash_attention 2026-08-11. Cross-ref [PB-15](pb_15_parallel_npu_launches_on_same_docker_container_pro.md), [OL-97](../precision/ol_97_precision_target_cpu_truth_vs_cann_bit_match.md).

- Former A5Ops ID: `PB-111`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=attention (incre_flash_attention CANN reference path)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; torch_npu (aarch64 build)`
`unverified_on: soc=Ascend950PR, Ascend910 (may or may not reproduce — re-verify before assuming the CANN op is stream-stable on other SoCs)`

**Symptom**: calling `torch_npu.npu_incre_flash_attention` (CANN `aclnnIncreFlashAttention`) repeatedly on Ascend310P3 aicore-**time-outs after ~8 invocations**; the failure persists across device 0 and device 1. The CANN reference runs cases 0–6, then times out from case 7+. The hand-written AscendC kernel runs all 27 cases cleanly when the CANN reference is NOT invoked on the same process.

**Mechanism (hypothesis, not yet isolated)**: likely a per-invocation resource leak (events / scratch / stream-internal allocation) inside the CANN incre-FA op that accumulates until the aicore watchdog fires. Falsifiable claim: the timeout count is invocation-monotonic (~8), independent of input shape, and recovers only on process/device reset — NOT on retry within the same process.

**Impact on verification (measurement limit, NOT a pass-gate relaxation)**: the T2 comparison (our kernel vs the live CANN reference) cannot complete for all 27 cases on this box; T2 is bounded to the cases (0–6) where the CANN reference still runs. Precision is STILL fully verified via T1 (golden truth captured on Ascend910B2C) for all 27 — T1 is the load-bearing verdict here, T2 is a sanity cross-check on the subset that runs.

**Distinct from PB-15**: PB-15 is cross-PROCESS concurrent-launch stream-state corruption (parallel `docker exec &`); PB-111 is single-PROCESS sequential repeated-invocation timeout. Different signature — PB-15 = transient magnitude drift under concurrency; PB-111 = monotonic aicore timeout under sequential reuse of one CANN op.

**Workaround (current)**: rely on T1 golden-truth verification for full coverage; use the CANN reference only for the subset of cases it runs as a T2 cross-check. Do NOT loop `npu_incre_flash_attention` unbounded in a verification harness — wrap it with a per-call reset/retry budget and treat invocation-count-monotonic timeout as THIS bug, not a kernel-correctness bug.

**Evidence**: incre_flash_attention (2026-08-11, Ascend310P3 / dav-m200, CANN 9.0.0) — CANN ref times out case 7+ on both device 0 and device 1; our kernel 27/27 on T1 golden truth (`verification.json.precision`: 14/27 strict PASS, all `max_abs ≤ 0.000977`; `method` records "T2 attempted but CANN ref matches golden exactly (mere=0) for cases 0-4, so T2 cannot bridge gap").

**Cross-ref**: PB-15 (concurrency stream-state corruption — different mechanism), OL-97 (CPU-truth vs CANN-bit-match target choice — here T1/golden-truth is the viable target precisely because T2 is reference-limited).
