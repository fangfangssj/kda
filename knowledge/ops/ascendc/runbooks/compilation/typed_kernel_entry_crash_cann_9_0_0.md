---
type: Runbook
title: "Typed Kernel Entry Crash (CANN 9.0.0)"
description: "Typed Kernel Entry Crash (CANN 9.0.0)"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-1"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-1 — PB-1: Typed Kernel Entry Crash (CANN 9.0.0)

## 适用场景

- 适用平台与范围：paradigm=ascendc（卡内 applies_to 仅声明范式；受影响版本等细节见正文 Affected 行）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行。

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Error 507035 on kernel launch with typed entry points (e.g., `_fp32` suffix)
- **Affected**: CANN 9.0.0 with bisheng 2026-03-21
- **Workaround**: Use legacy untyped entry points (single dispatcher .cpp, cast inside kernel)
- **Status**: OPEN (not fixed in CANN 9.0.T501)
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-16
