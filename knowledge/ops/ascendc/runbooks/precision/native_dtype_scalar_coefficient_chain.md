---
type: Runbook
title: "Native-dtype constraint for scalar coefficient chain (fp16 normalization backward)"
description: "Native-dtype constraint for scalar coefficient chain (fp16 normalization backward)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["P-P51"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P51

## 适用场景

- 同号异题：optimizations/ 区既有 v1 卡 [p_p51_lift_nblk_to_runtime_aiv_count_for_v220_portabil.md](../../optimizations/p_p51_lift_nblk_to_runtime_aiv_count_for_v220_portabil.md) 与本卡同号 P-P51 但主题不同（lift-nblk 运行时分块）；引用 P-P51 时需按主题区分。
- 适用平台与范围：卡内无 applies_to 行；适用触发条件见正文模式索引行（trigger 列）。
- 验证来源：卡内无 verified_on/confirmed_on 行；本卡为 PATTERN_INDEX 模式行沉淀，未附验证记录。

| P-P51 | Native-dtype constraint for scalar coefficient chain (fp16 normalization backward) | precision | reduction result → multi-step scalar arithmetic → broadcast; every step must stay in native dtype (fp16/bf16), cannot do all math in fp32 and cast only at the end. **[CPU-truth audit 2026-04-29: ✅ VALIDATED — co-occurs with CANN-pass op 20_FusedRopeWithQkNormAndKvCacheUpdate. Do not downgrade.]** | **CRITICAL** |
