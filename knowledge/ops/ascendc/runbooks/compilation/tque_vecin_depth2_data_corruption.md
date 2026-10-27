---
type: Runbook
title: "TQue<VECIN,2> Data Corruption"
description: "TQue<VECIN,2> Data Corruption"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["TQue", "VECIN"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-2 — PB-2: TQue<VECIN,2> Data Corruption

> 关联：同一 PB-2 主题已由既有卡 [pb_2_tque_vecin_2_data_corruption.md](pb_2_tque_vecin_2_data_corruption.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：paradigm=ascendc（卡内 applies_to 仅声明范式；受影响版本等细节见正文 Affected 行）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行。

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: 99.5% elements corrupted when using TQue with depth 2
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use TQue<VECIN,4> (depth 4 works correctly)
- **Status**: OPEN
- **Evidence**: hardware/target/ascend950pr.md, E13 test data
