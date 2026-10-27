---
type: Guide
title: "Per-core GM workspace slab stride is a three-party contract (producer/consumer/pybind) — grep-audit cid-indexed expressions"
description: "Slab stride mismatch between producer class, consumer class and allocator is invisible to single-core numeric probes and multi-core marker probes (cid=0 coincides; markers are addressing-blind). Grep-audit every cid* expression; one multi-core probe must check numerics. Found on 11_BigBirdAttention ws2 (+4096 vs +2048)."
tags: [task.debugging, topic.implementation.pipeline, topic.memory]
status: draft
sources: []
created_at: "2026-09-26T12:23:46Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases:
- OL-614
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Per-core GM workspace slab stride is a three-party contract (producer/consumer/pybind) — grep-audit cid-indexed expressions

## Evidence

- **run_id**: 5450b083-4f53-4620-a1b2-8656e7bb9889
- **soc**: Ascend950DT_9582
- **clock**: 2026-09-26T12:23Z
- **trigger**: 11_BigBirdAttention: end-to-end kernel read found ws2 slab stride mismatch (cube+4096/vec+2048/pybind+4096)

## 现象 / 触发
MIX kernel with per-core GM workspace slabs: AIC read P tiles at
`cid*(slab+4096)+tile*2048` (its own constant) while the AIV wrote them at
`cid*(slab+2048)+tile*2048`, pybind allocating `grid*(slab+4096)`. cid=0 coincides, so
single-core probes pass numerically; cid>=1 the AIC reads zeros -> O'=0 -> zero output
rows for all cores>0. Multi-core probes that check only progress markers also pass.

## 修法 / 规则
One-line stride alignment across all three sites. The cheap detector is a grep-audit:
enumerate every `cid *`-indexed GM expression in producer, consumer, and allocator, and
assert the slab-stride expression is textually/semantically identical. At least one
multi-core probe must check NUMERICS of a cid>=1 core's output (markers are blind to
addressing bugs).
