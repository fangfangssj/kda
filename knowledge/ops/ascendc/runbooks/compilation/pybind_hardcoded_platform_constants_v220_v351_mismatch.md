---
type: Runbook
title: "Pybind hardcoded platform constants — mirror upstream runtime ascendcPlatform queries to avoid silent V220-vs-V351 mismatch"
description: "Pybind hardcoded platform constants — mirror upstream runtime ascendcPlatform queries to avoid silent V220-vs-V351 mismatch"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP99"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP99 — Pybind hardcoded platform constants — mirror upstream runtime ascendcPlatform queries to avoid silent V220-vs-V351 mismatch

## 适用场景

- 适用平台与范围：soc=all; cann=9.0.0+; op_class=all（卡内 applies_to 行）。
- 验证来源：soc=Ascend950PR; cann=9.0.0（卡内 verified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=9.0.0+; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Anti-pattern: pybind11.cpp hardcoding platform constants (`AIC_NUM`, `AIV_NUM`, `LIB_API_WS_BYTES`) at host-side. On V220→V351 ports these values silently mismatch (V220 hard-coded to V220 numbers, deployed on V351 → wrong workspace size + wrong block dim). Use runtime `ascendcPlatform` queries (`GetCoreNumAic()` / `GetCoreNumAiv()` / `GetLibApiWorkSpaceSize()`) mirroring upstream pybind11.cpp pattern.

Source: lightning_indexer_grad kw-NEW 2026-05-23 anti-patterns §3.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP99，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
