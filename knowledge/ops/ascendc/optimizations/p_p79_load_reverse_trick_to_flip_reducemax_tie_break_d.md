---
type: Optimization
title: "Load-reverse trick to flip ReduceMax tie-break direction (P-P57 + stable-sort tie compatibility)"
description: "When a kernel uses P-P57 SIMD ReduceMax(calcIndex=true) for top-k AND must match a reference that uses sort(stable=True, descending=False) + mask(value < kth_value) (stable-ascending sort + mask), the"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/operator-optimization/p-p79-load-reverse-trick-to-flip-reducemax-tie-break-d.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
---

# Load-reverse trick to flip ReduceMax tie-break direction (P-P57 + stable-sort tie compatibility)

> 原 OKF v1 卡号：**P-P79**。

## 优化点

When a kernel uses P-P57 SIMD ReduceMax(calcIndex=true) for top-k AND must match a reference that uses `sort(stable=True, descending=False) + mask(value < kth_value)` (stable-ascending sort + mask), the tie-break directions disagree: Ascend `ReduceMax<calcIndex>` returns the **lowest** index for ties; reference's stable-ASC sort + mask keeps the **largest-original-index** at the kth boundary. fp16/bf16 quantization creates many ties, so ours and reference pick a different SUBSET of survivors → MERE/MARE = inf at the boundary positions. **Fix**: in-place reverse the working buffer `xf[0..N_pad)` after the Cast/Adds load step (e.g., scalar-loop swap or vector-rotate idiom). Pad NEG_SENTINEL at the FRONT instead of tail. ReduceMax now returns reduced_idx; convert back to original via `orig_idx = N_pad - 1 - reduced_idx`. Mask via `xf.SetValue(reduced_idx, NEG_SENTINEL)` so the next iter picks the next-largest. The reversed-coordinate "lowest-reduced-idx" maps to "largest-original-idx" — matches reference's tie convention exactly. **Generalizes**: any P-P57 user that needs PyTorch stable-ASC compatibility. **Anti-patterns** that don't work: secondary-key sort by -orig_idx within ties (more expensive); flipping dtype sentinel (orthogonal); expanding tie-inclusion buffer K_MAX_TIE alone (only matters when k_input == K_MAX, refuted on op#9 TopKTopP a3 — tied 0/24 → 0/24 fp16 cases). Validated op#9 TopKTopP a3 V200 2026-04-30 (aog-precision-probe `topktopp-pp-1` 8-iter probe): 22/50 → 43/50 (+21 cases) with load-reverse alone, then +1 with ascending-cumsum top-p walk = 44/50. Compare to P-P60 which addresses the same tie-direction issue for the V220-only `Sort<>` API; P-P79 is the V200-portable counterpart for ReduceMax-based top-k.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P79，convert_patterns_to_okf.py）。confidence 未升格。 -->
