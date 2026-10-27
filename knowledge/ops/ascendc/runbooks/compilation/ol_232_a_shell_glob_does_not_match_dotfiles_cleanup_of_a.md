---
type: Runbook
title: "A shell `*` glob does NOT match dotfiles — cleanup of a SHARED workspace must explicitly remove dotfile markers, or a prior op's stale marker silently contaminates the next op"
description: "<!-- applies_to_backend: all -->"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-232-a-shell-glob-does-not-match-dotfiles-cleanup-of-a.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-232"]
---

# A shell `*` glob does NOT match dotfiles — cleanup of a SHARED workspace must explicitly remove dotfile markers, or a prior op's stale marker silently contaminates the next op

> 原 OKF v1 卡号：**OL-232**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - <!-- applies_to_backend: all -->

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=n/a; bisheng=n/a; op_class=all`
`verified_on: host-tooling (soc-independent — this is POSIX shell glob behavior in the deploy/verify harness, not a hardware lesson)`

### Principle

`rm -rf $DIR/*` relies on the shell to expand `*`, and by default the glob **skips entries beginning with `.`** (dotfiles). So a "clean the workspace" step written as `rm -rf $REMOTE_TASK/*` leaves every `.foo` marker behind. When `$REMOTE_TASK` is a **shared** location reused across ops (e.g. a single remote `current_task` dir on a verify NPU host), a control dotfile dropped by op A persists into op B's run and silently changes B's behavior. The failure is invisible: no error, B just takes a different branch.

### Concrete anchor

```bash
# BUGGY — leaves .truth_source_override, .kb_merged, .agent_died_at_*, etc. behind:
rm -rf "$REMOTE_TASK"/*

# CORRECT — also remove dotfiles (guard against unset var; '.*' would match . and ..):
rm -rf "$REMOTE_TASK"/* "$REMOTE_TASK"/.[!.]* 2>/dev/null
# or: find "$REMOTE_TASK" -mindepth 1 -delete
```
Either delete the specific known markers explicitly before a run, or clean with a method that includes dotfiles (`find -mindepth 1 -delete`, or shell `dotglob`).

### Evidence
- MultiScaleDeformableAttnFunction port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b): `deploy_to_npu.sh` cleans the shared remote `current_task` with `rm -rf $REMOTE_TASK/*`, which does NOT match dotfiles. A `.truth_source_override` left by a PRIOR op (celu_v2) survived into the MSDA verify run, falsely tripped `is_synth_conflation` → all cases marked `A3_UNAVAILABLE` → T2 tier never engaged → false FAIL on the 9 cases T2 rescues. Deleting the stale `.truth_source_override` before verify restored correct tiering (inclusive 33/34).

### Other instances (predicted)
- Any harness that reuses a shared/remote scratch dir across ops and cleans with `rm -rf dir/*`: stale `.kb_merged`, `.agent_died_at_*`, `.workflow_exception_*`, `.truth_source_override`, lane-state dotfiles all leak forward. Audit every `rm -rf .../*` over a shared path; prefer `find -mindepth 1 -delete`.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-232（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
