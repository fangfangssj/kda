---
type: Runbook
title: "Two-kernel split (AIC_ONLY + AIV_ONLY via separate ACLRT_LAUNCH_KERNEL on same stream) as PB-34 mitigation Pattern C — empirically broken on V351"
description: "Two-kernel split (AIC_ONLY + AIV_ONLY via separate ACLRT_LAUNCH_KERNEL on same stream) as PB-34 mitigation Pattern C — empirically broken on V351"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-PP102"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-PP102 — Two-kernel split (AIC_ONLY + AIV_ONLY via separate ACLRT_LAUNCH_KERNEL on same stream) as PB-34 mitigation Pattern C — empirically broken on V351

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.0.0; op_class=all_cube_vec_fused（卡内 applies_to 行）。
- 验证来源：soc=Ascend950PR; cann=9.0.0（卡内 verified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all_cube_vec_fused`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Anti-pattern (verified broken): splitting a fused cube+vec op into two separate kernels (AIC_ONLY + AIV_ONLY) launched as two ACLRT_LAUNCH_KERNEL calls on the same stream — proposed as PB-34 manual-CrossCoreSetFlag deadlock mitigation. Empirically: AIV kernel returns ret=0 but writes nothing (matches `13_Cat` silent-no-exec signature). Probe required (constant-write probe to AIV-only entry) before declaring this infrastructure-unsupported vs algorithm-bug.

Source: grouped_matmul_swiglu_quant_v2 kw-3 2026-05-24.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP102，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
