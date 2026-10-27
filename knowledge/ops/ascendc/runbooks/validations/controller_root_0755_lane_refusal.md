---
type: Runbook
title: "controller state root 若被 umask 默认建成 0755，lane 准入会正确拒绝（每次 build exit 2）——修法是显式 chmod 0700 根目录，不是放宽断言"
description: "controller state root 若被 umask 默认建成 0755，lane 准入会正确拒绝（每次 build exit 2）——修法是显式 chmod 0700 根目录，不是放宽断言"
tags: [task.debugging, topic.implementation]
status: draft
sources: []
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-INFRA-CONTROLLER-ROOT-0755"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-INFRA-CONTROLLER-ROOT-0755

## 适用场景

- 新账号/新主机首次跑 op-gen：worker build 在编译前 exit 2（lane-workspace admission refused, rc=4）。

## 结论（39_ChunkKdaFwd，2026-09-20）

`Path.mkdir(parents=True, mode=0o700)` 的 mode 只作用于叶子——隐式父目录拿 umask 默认（0755）。controller state root（~/.local/state/a5_ops/controller）一旦是 0755，`ensure_controller_host_task` 正确地拒绝（0755 会让 controller 权威变成写 gadget，有钉死的测试）。**修法：chmod 0700 根目录（恢复 harness 要求且测试钉住的不变量），叶子走 sanctioned creator 创建。不要放宽谓词、不要在 pre-O4 加创建调用**（会破坏恰好一次不变量）。

- **证据**：39 归档 knowledge_update.md finding 1-2（含 /tmp 阳性对照：叶子 0700、父 0775 复现）。
- **Cross-ref**：lane_workspace_admission.py / lane_workspace_provision.py。
