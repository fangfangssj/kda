---
type: Runbook
title: "Top-K buffer `effective_kept` vs `buffer_len` — always compute effective_kept separately"
description: "Top-K buffer `effective_kept` vs `buffer_len` — always compute effective_kept separately"
tags: [task.debugging, topic.precision, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-32-top-k-buffer-effective-kept-vs-buffer-len-always-c.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-32", "effective_kept", "buffer_len"]
---

# Top-K buffer `effective_kept` vs `buffer_len` — always compute effective_kept separately

> 原 OKF v1 卡号：**EC-32**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Kernel with oversized top-K buffer (e.g. TOPK_CAP > actual row's count of v >= threshold) produces wrong cutsum cumulation, cutoff decision lands far from thres

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Kernel with oversized top-K buffer (e.g. TOPK_CAP > actual row's count of v >= threshold) produces wrong cutsum cumulation, cutoff decision lands far from threshold, precision all-fail with systemic offset.
- **Root cause**: Worker walks cumsum / reduce over the full TOPK_CAP-sized buffer assuming all positions contain valid data. When the row has fewer than TOPK_CAP values ≥ threshold (e.g. row with small N, or row where actual kept count < buffer cap), the walk processes uninitialized / padding positions, inflating the cum and preventing cutoff from triggering.
- **Fix**: After any top-K merge / compaction step, compute `effective_kept` separately:
  ```cpp
  int32_t effective_kept = 0;
  for (int32_t i = 0; i < TOPK_CAP; i++) {
    if (top_val[i] >= threshold) effective_kept++;
  }
  ```
  Then walk `[0 .. effective_kept - 1]` (or the equivalent ASC range), NOT `[0 .. TOPK_CAP - 1]`. Padding positions between effective_kept and TOPK_CAP must be held at a sentinel (e.g. -inf) that guarantees they would be rejected if processed — belt-and-suspenders.
- **Detection**: Small-N cases fail systemically while large-N cases (where effective_kept ≈ TOPK_CAP) pass. Or: cumsum values reach 1.0 far from the expected cutoff rank.
- **Evidence**: 9_TopKTopP cold-run Phase D iter 2 (2026-04-18). Worker walked from `topk_len - 1` down instead of `effective_kept - 1`; for rows with < TOPK_CAP kept values, iterated over non-kept pos → cum inflated → cutoff never set. Fix: explicit `effective_kept` scan, walk that range.
- **Related**: P-P59 (tied-threshold buffer truncation) — P-P59 assumes `effective_kept ≤ TOPK_CAP`; this EC is about the distinct implementation concern that `effective_kept` may be `< TOPK_CAP` for small rows.

<!-- 迁移自 porter kb/target/ascendc/（EC-32，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
