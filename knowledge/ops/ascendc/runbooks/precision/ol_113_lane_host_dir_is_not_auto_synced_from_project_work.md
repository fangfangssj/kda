---
type: Runbook
title: "Lane host dir is NOT auto-synced from project workspace — `deploy_to_npu_lane.sh` reads stale source unless caller manually copies first"
description: "Lane host dir is NOT auto-synced from project workspace — `deploy_to_npu_lane.sh` reads stale source unless caller manually copies first"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.data_movement]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-113-lane-host-dir-is-not-auto-synced-from-project-work.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-113", "deploy_to_npu_lane.sh"]
---

# Lane host dir is NOT auto-synced from project workspace — `deploy_to_npu_lane.sh` reads stale source unless caller manually copies first

> 原 OKF v1 卡号：**OL-113**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - any agent running on a non-default lane (LANE != "" / using _lane{N} paths)

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: process / tooling / parallel-lane orchestration
- **Loaded by**: aog-kernel-worker / aog-kernel-optimizer / aog-precision-probe (every iteration that uses `deploy_to_npu_lane.sh`)
- **Trigger**: any agent running on a non-default lane (`LANE != ""` / using `_lane{N}` paths)

### Principle

`src/scripts/deploy_to_npu_lane.sh` reads kernel files from `$LOCAL_TASK = ~/workspace/AscendOpGenAgent_lane{N}/current_task/` and tars+ships **that** directory. There is **no automatic rsync/cp** from the project workspace `workspace/{op}/` to `$LOCAL_TASK`. If the agent edits files only in `workspace/{op}/` and runs `deploy_to_npu_lane.sh`, the build runs against whatever was last manually copied into `$LOCAL_TASK` — silently stale.

### Mandatory protocol per deploy iteration

Before EVERY `deploy_to_npu_lane.sh --lane N --build`:
```bash
cp -r workspace/{op}/kernel/* /home/npu_user/workspace/AscendOpGenAgent_lane{N}/current_task/kernel/
cp workspace/{op}/model_new_ascendc.py /home/npu_user/workspace/AscendOpGenAgent_lane{N}/current_task/
# if model.py was edited: cp workspace/{op}/model.py $LOCAL_TASK/
```

### Detection signal

**Identical crash signature or precision pattern across iterations with very different code edits is a strong prior for the stale-source bug.** Verify by inspecting remote container source:
```bash
sshpass -p $A5_PASSWORD ssh $A5_USER@$A5_HOST \
  "docker exec $A5_CONTAINER stat -c '%y %n' /home/npu_user/workspace/AscendOpGenAgent_lane{N}/current_task/kernel/*.h"
```
If remote mtime predates your last edit by minutes/hours, the bug fired.

### Evidence

- 29_DynamicQuant kw-2 (2026-05-01): worker spent **4 build/probe cycles** debugging an unchanged kernel before noticing `$LOCAL_TASK/kernel/dynamicquant_kernel.h` mtime was 17:10 while `workspace/29_DynamicQuant/kernel/dynamicquant_kernel.h` mtime was 17:54. The probe-recommended two-pass tile loop fix had been edited into workspace but never reached the lane host. Once corrected, kernel passed precision on first deploy. Self-challenge protocol caught it on iter 5; could have been iter 1 with this OL.

### Other instances (predicted)

Any agent on lane > 0 (i.e. parallel-lane batches via `/aog-op-batch` or main-context multiplex per C12 carve-out). The lane=0 default path uses `$LOCAL_TASK = ~/workspace/AscendOpGenAgent/current_task/` which by convention some setup flows DO sync into; lane > 0 is uncovered.

### Permanent fix candidates (filed for V3.7.4 follow-up)

1. Modify `deploy_to_npu_lane.sh` to do `cp -r workspace/{op}/kernel/* $LOCAL_TASK/kernel/` automatically when invoked with `ASCENDC_WORKSPACE` set.
2. OR: change `$LOCAL_TASK` derivation to read directly from `$ASCENDC_WORKSPACE` when set.
3. Until either lands: this OL is the discipline.

### Related

- C13 (self-critic: claim runtime state without verification) — stale-source fits: agent claimed "fix didn't work" without verifying remote source matched local
- OL-85 (anti-overfit gate) — agent's anti-overfit "no regression" check is meaningless if running stale code

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-113（category=process / tooling / parallel-lane orchestration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
