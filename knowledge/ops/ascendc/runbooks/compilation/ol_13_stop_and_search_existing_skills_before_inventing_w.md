---
type: Runbook
title: "STOP and search existing skills before inventing workarounds"
description: "STOP and search existing skills before inventing workarounds"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-13-stop-and-search-existing-skills-before-inventing-w.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["OL-13"]
---

# STOP and search existing skills before inventing workarounds

> 原 OKF v1 卡号：**OL-13**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - when hitting infrastructure problems (network, auth, deployment)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process
- **Loaded by**: Builder, Optimizer, QA
- **Trigger**: when hitting infrastructure problems (network, auth, deployment)
- **Lesson**: The A5 container has proxy access via `a5_exec.py --proxy` (sources `/home/npu_user/setup_proxy.sh`). The agent spent 30+ minutes on scp/tar/docker cp workarounds instead of checking the `/a5_op` skill documentation it was already using. The proxy flag enables git clone, pip install, and all internet access. Always check existing skill docs FIRST.
- **Evidence**: Session 2026-03-29, SG deployment failures

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-13（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
