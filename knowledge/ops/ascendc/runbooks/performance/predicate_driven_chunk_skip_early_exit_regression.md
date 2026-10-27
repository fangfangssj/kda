---
type: Runbook
title: "Predicate-driven chunk-skip early-exit on iterative merge regresses on Gaussian / well-spread input distributions (ANTI-PATTERN candidate)"
description: "Predicate-driven chunk-skip early-exit on iterative merge regresses on Gaussian / well-spread input distributions (ANTI-PATTERN candidate)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP77"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP77 — Predicate-driven chunk-skip early-exit on iterative merge regresses on Gaussian / well-spread input distributions (ANTI-PATTERN candidate)

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；观测 op#9 9_TopKTopP ko-1 Opt1（Ascend950PR_9579，卡内 Source 行），场景为 chunked Sort + 双指针 merge 的 top-K 早退谓词（正文 Hypothesis tested 段）。
- 验证来源：卡内无 verified_on 行；1 op REVERT 证据（Gaussian 分布 bf16 N=65536 k=1024 +5.0% 回归），反模式候选未 promote（正文 Validation status / Why it failed 段）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: op#9 9_TopKTopP ko-1 Opt1 (2026-05-02 Ascend950PR_9579) — REVERT.

**Validation status**: 1 op observed (REVERT evidence). Anti-pattern candidate, not promoted.

**Hypothesis tested**: For a chunked Sort + 2-pointer merge into a top-K buffer of capacity `K_MAX`, add a per-chunk early-exit `if (chunk_max < top_min) skip_merge` once the buffer is full. The intuition: if the new chunk's maximum is below the current top-K floor, no element in this chunk can survive the merge, so the entire `K_MAX`-iter merge loop is wasted work.

**Why it failed (Gaussian-distribution case)**:
- For random-normal input at N = 65536, the top-1088 floor stabilizes at ~2σ after the first 4–8 chunks
- New chunk maxes are routinely 3–4σ → the early-exit predicate `chunk_max < top_min` almost never fires
- Net cost added per chunk: 2× scalar `GetValue` + 1× explicit V→S sync, with zero merge-loop savings
- Wall-clock impact: bf16 N=65536 k=1024 +5.0 % regression; fp16 mid +3.9 % regression; fp32 small −0.4 % (noise)

**When the pattern *would* work** (predicted, not yet observed):
- Long-tailed distributions (real attention scores: a few large positions, many tiny tail)
- Inputs with provable structure (already partially sorted, sparse logits with many filler positions)
- Distributions where the top-K floor saturates fast and most subsequent chunks are below it

**Generalization**: any iterative merge / scan / accumulate with a "skip if this iter cannot contribute" predicate is data-distribution-dependent. The optimization is NOT a free win — it adds fixed predicate-test cost in exchange for a probabilistic savings that depends on input distribution.

**Promotion criteria**:
1. Demonstrate +5 % or better speedup on a real-workload distribution (e.g. captured attention scores or production logits) on ≥2 ops
2. Document a distribution test the kernel can apply at runtime to gate the early-exit (e.g. estimate input variance from the first chunk's max-to-mean ratio; only enable skip when the estimated tail is heavy)
3. Show that the predicate cost (`GetValue` + sync) is amortized across enough merges to net positive

**Until promotion**: aog-kernel-optimizer should NOT propose chunk-skip / iter-skip early-exits on merge-style hot loops without empirical evidence on the target workload's distribution. If proposed, treat as anti-pattern unless input distribution is documented to be long-tailed.

**Related**: P-P81 (runtime-bounded loop cap — the validated alternative for iter-count reduction; works on per-row k variance, NOT per-chunk data variance), OL-85 (anti-overfitting — distribution-specific kernel paths require the distribution probe in the kernel), failures_ledger.md tagging convention (REVERT entries become anti-pattern evidence).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP77，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
