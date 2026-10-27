---
type: Runbook
title: "`deploy_to_a5.sh` syncs `current_task/` not `workspace/<op>/kernel/`"
description: "`deploy_to_a5.sh` syncs `current_task/` not `workspace/<op>/kernel/`"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-30-deploy-to-a5-sh-syncs-current-task-not-workspace-o.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["EC-30", "deploy_to_a5.sh"]
---

# `deploy_to_a5.sh` syncs `current_task/` not `workspace/<op>/kernel/`

> 原 OKF v1 卡号：**EC-30**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Worker edits workspace/<op>/kernel/topktopp_kernel.h, runs bash src/scripts/deploy_to_a5.sh --build, build succeeds, precision result is unchanged from previous

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: Worker edits `workspace/<op>/kernel/topktopp_kernel.h`, runs `bash src/scripts/deploy_to_a5.sh --build`, build succeeds, precision result is **unchanged** from previous iteration despite the edit. Verification reports same FAIL signature as before the change.
- **Root cause**: `deploy_to_a5.sh` syncs `~/workspace/AscendOpGenAgent/current_task/` to A5 container for build. It does **not** read from `workspace/<op>/`. So edits to `workspace/<op>/kernel/*.h` are invisible to the build.
- **Fix**: Worker must manually copy before each build:
  ```bash
  cp $LOCAL_PROJECT/workspace/<op>/kernel/* \
     ${LOCAL_TASK:-$HOME/workspace/AscendOpGenAgent/current_task}/kernel/
  bash src/scripts/deploy_to_npu.sh --build
  ```
  Or fix the deploy script to accept `ASCENDC_WORKSPACE` env var and sync from that. Until fixed, worker/probe must always pre-copy.
- **Detection**: Verification run produces identical results across two consecutive iters despite kernel edits. Confirm by `md5sum workspace/<op>/kernel/topktopp_kernel.h` vs `md5sum <current_task on A5>/kernel/topktopp_kernel.h` — different md5 = phantom build.
- **Evidence**:
  - 9_TopKTopP V2 iter 1 (2026-04-17). Worker rewrote kernel, built, got identical 34/50 FAIL as v1 — phantom 30 minutes debugging before noticing A5 current_task/ had stale v1 kernel.
  - 9_TopKTopP cold-run probe (2026-04-18). Probe explicitly ran `cp workspace/topktopp_v31/kernel/* current_task/kernel/` before every `deploy_to_a5.sh --build` call — confirmed the workaround works and is load-bearing. Spawned inside aog-kernel-worker + aog-precision-probe specs as a standing requirement.

<!-- 迁移自 porter kb/target/ascendc/（EC-30，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
