---
type: Runbook
title: "V351 single-launch FA-class kernel — all-zero output diagnosis checklist when build+dispatch PASS"
description: "V351 single-launch FA-class kernel — all-zero output diagnosis checklist when build+dispatch PASS"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP98"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP98 — V351 single-launch FA-class kernel — all-zero output diagnosis checklist when build+dispatch PASS

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.0.0; op_class=fused-attention（卡内 applies_to 行）。
- 验证来源：soc=Ascend950PR; cann=9.0.0（卡内 verified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=fused-attention`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Bisection: when a V351 single-launch FA-class kernel builds clean, dispatches clean, and produces all-zero output, walk these probes in order to localize the silent-no-exec:
1. Zero-pybind probe — confirm pybind wrapper actually marshals args + invokes kernel (insert printf in pybind, rerun)
2. Zero-VEC probe — invoke VEC-only path, check if any VEC write fires
3. Zero-MTE3 probe — instrument MTE3 emit count to verify the cube→GM writeback is happening
4. Fixpipe probe — N/A on V351 (V351 uses different cube writeback). On V220 this is the cube→L0C→GM step

Source: lightning_indexer_grad kw-NEW 2026-05-23.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP98，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
