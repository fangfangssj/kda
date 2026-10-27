---
type: Runbook
title: "NPU native atomicAdd (no fastAtomicAdd)"
description: "NPU native atomicAdd (no fastAtomicAdd)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["P-P31"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# P-P31 — P-P31: A5Ops additive extension

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；domain 与 trigger 见正文首行（platform_compat；CUDA fastAtomicAdd / packed half2 atomics）。
- 验证来源：卡内无 verified_on/confirmed_on 行；正文未附验证记录。

**P-P31** — NPU native atomicAdd (no fastAtomicAdd) (domain: platform_compat; trigger: CUDA fastAtomicAdd / packed half2 atomics)

- Former A5Ops ID: `P-P31`
- Fusion disposition: `additive_new`

**P-P31** — NPU native atomicAdd (no fastAtomicAdd) (domain: platform_compat; trigger: CUDA fastAtomicAdd / packed half2 atomics)
