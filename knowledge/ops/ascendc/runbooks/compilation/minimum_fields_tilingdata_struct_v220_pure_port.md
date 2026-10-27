---
type: Runbook
title: "Minimum-fields `<Op>TilingData` struct in port_a3 V220-pure path"
description: "Minimum-fields `<Op>TilingData` struct in port_a3 V220-pure path"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP101"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP101 — Minimum-fields `<Op>TilingData` struct in port_a3 V220-pure path

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure。
- 验证来源：soc=Ascend950PR; cann=9.0.0（卡内 verified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: when porting V220 cube+vec fused op via thin-TU wrapping (CAND-PP100), grep `tilingData->` first in the worker TU + helpers, mirror only the accessed fields in your local `<Op>TilingData` struct. Skip the upstream `BEGIN_TILING_DATA_DEF` + nested `TCubeTiling` reconstruction.

Source: flat_quant kw-1 2026-05-23 (4-field struct sufficed where upstream had 11-field nested struct).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP101，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
