---
type: Runbook
title: "`deploy_to_npu.sh` TARGET case statement rejects `a3-ds` and other alias targets"
description: "applies_to: soc=all; cann=all; bisheng=all; op_class=all"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-44-deploy-to-npu-sh-target-case-statement-rejects-a3.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: []
aliases: ["EC-44", "deploy_to_npu.sh", "a3-ds"]
---

# `deploy_to_npu.sh` TARGET case statement rejects `a3-ds` and other alias targets

> 原 OKF v1 卡号：**EC-44**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - deploy_to_npu.sh's case statement only accepts a5|a3|a2 for TARGET. When .ascendc_env sets TARGET=a3-ds (DS-backend isolation), the script exits with no matchin

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=all; bisheng=all; op_class=all`
`verified_on: soc=Ascend910_9382; cann=9.0.0-beta.2`
- **Severity**: LOW (deploy-time blocker, workaround exists)
- **Status**: OPEN
- **Symptom**: `deploy_to_npu.sh`'s case statement only accepts `a5|a3|a2` for TARGET. When `.ascendc_env` sets `TARGET=a3-ds` (DS-backend isolation), the script exits with no matching case and deploy is blocked.
- **Workaround**: manual `tar+scp+docker exec` deploy, or temporarily change TARGET in .ascendc_env.
- **Fix**: add case aliases in the script (`a3-ds|a3_kimi|a3`) all mapping to the a3 code path.
- **Detection rule**: if `deploy_to_npu.sh` exits silently without deploying, check that TARGET matches one of the script's expected values.
- **Evidence**: op#30 NMS a3 ds kw-1 (2026-05-07) — manual tar+scp+extract deploy required because script rejected TARGET=a3-ds.

<!-- 迁移自 porter kb/target/ascendc/（EC-44，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
