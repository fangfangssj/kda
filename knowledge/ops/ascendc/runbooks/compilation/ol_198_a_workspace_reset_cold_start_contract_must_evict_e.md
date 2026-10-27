---
type: Runbook
title: "A workspace-reset (cold-start) contract must evict EVERY in-workspace backup/snapshot dir, not just one naming convention — any surviving snapshot is a restore-target that the next generator agent will resurrect"
description: "<!-- applies_to_backend: all -->"
tags: [task.debugging, topic.data_movement, topic.performance.analysis, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-198-a-workspace-reset-cold-start-contract-must-evict-e.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: []
aliases: ["OL-198"]
---

# A workspace-reset (cold-start) contract must evict EVERY in-workspace backup/snapshot dir, not just one naming convention — any surviving snapshot is a restore-target that the next generator agent will resurrect

> 原 OKF v1 卡号：**OL-198**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - <!-- applies_to_backend: all -->

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all; scope=op-gen harness / workspace-reset`
`verified_on: orchestrator _cold_start_reset_workspace (a5_ops), FA-class IL chain`

**Principle**: a "fresh evaluation" / clean-slate workspace reset that wipes worker outputs (kernel/, runners, state) but leaves ANY in-workspace backup or snapshot directory in place silently breaks its own contract: the next generation agent explores the workspace, finds the surviving snapshot, and **restores from it instead of regenerating** — so the run reproduces the OLD artifact rather than the design-intent one. The reset must evict every backup/snapshot dir family (move OUT of the workspace, audit-preserved), enumerated by a GLOB SET, not a single prefix. Eviction-by-single-prefix is the trap: a second backup convention added later (manual snapshots, per-iter attempt dirs, pre-restore saves) is invisible to the prefix and survives.

**Concrete anchor**: the reset migrated only `.pre-cold-start-*` out; the eviction set had to become `{.pre-cold-start-*, .empirical_backup*, .pre_kw*_restore_*, .pre_iter*, .iter*_attempt, .pattern_a*}` (dedup + is_dir guard). Decision rule: if a workspace dir holds a prior copy of generator output, it is a restore-target → evict on reset.

**Evidence**: FA-class IL `3_FusionAttention` (2026-05-29). cold-start wiped kernel/+runners but left `.empirical_backup_2026_05_26` (a block_N=64 kernel snapshot); the IL translator restored that kernel **byte-for-byte** (disk-verified: emitted kernel == backup) instead of translating fresh from the designer's `tile_level` block_N=128 — so op-gen never produced the design-intent (fast) kernel, plus it triggered a P94 self-citing-verifier rollback churn. Fix (a5_ops PR #261): eviction-glob-set + regression test `test_cold_start_migrates_empirical_backup_out_of_workspace`. After fix, a clean cold-start produced a fresh translate (file-naming + single-pass + decision_manifest all confirmed regenerate, not restore).

**Other instances (predicted)**: any op-gen mode with a cold-start/reset (benchmark / port_a3_to_a5 / backward); any harness that snapshots prior attempts into the working dir; backward/training op-gen feature reusing the same reset primitive.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-198（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
