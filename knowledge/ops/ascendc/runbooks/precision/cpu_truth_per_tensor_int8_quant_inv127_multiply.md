---
type: Runbook
title: "CPU-truth construction rule for per-tensor int8 dynamic quant — multiply by precomputed `_INV_127`, NOT divide by `127.0`"
description: "CPU-truth construction rule for per-tensor int8 dynamic quant — multiply by precomputed `_INV_127`, NOT divide by `127.0`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP103"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP103 — CPU-truth construction rule for per-tensor int8 dynamic quant — multiply by precomputed `_INV_127`, NOT divide by `127.0`

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=quant（卡内 applies_to 行）。
- 验证来源：soc=Ascend950PR; cann=9.0.0（卡内 verified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; op_class=quant`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: per-tensor int8 dynamic quant CPU-truth synthesis must mirror CANN's `Muls(absmax, DYNAMIC_QUANT_FACTOR)` semantics — multiply by precomputed `_INV_127 = 1.0/127.0` constant. Do NOT divide by `127.0` directly. Mathematically equivalent but LSB-different on fp32 → silent bit-exact drift when CPU truth uses divide.

Source: grouped_matmul_swiglu_quant_v2 kw-3 2026-05-24 (Path-B truth synthesis, 8/8 cases bit-exact post-fix).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP103，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
