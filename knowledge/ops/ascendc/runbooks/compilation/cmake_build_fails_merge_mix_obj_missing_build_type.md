---
type: Runbook
title: "CANN Build Fails at merge_mix_obj.sh (95%)"
description: "CANN Build Fails at merge_mix_obj.sh (95%)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-11"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-11 — EC-11: CANN Build Fails at merge_mix_obj.sh (95%)

## 适用场景

- 适用平台与范围：paradigm=ascendc（卡内 applies_to 仅声明范式；触发条件与受影响环境细节见正文 Error pattern / Root cause 行）。
- 验证来源：卡内无 verified_on 行；相关平台缺陷条目见正文 Related 行（PLATFORM_BUGS.md PB-7）。

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh`
- **Root cause**: `CMAKE_BUILD_TYPE` not set → cmake passes empty `--build-type` to `merge_mix_obj.sh` → `shift 2` fails
- **Fix**: Always pass `-DCMAKE_BUILD_TYPE=Release` to cmake
- **Related**: PB-7 in PLATFORM_BUGS.md

---