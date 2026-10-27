---
type: Runbook
title: "On V351 (A5), the MIX_AIC_1_2 AIV→AIC cross-core flag AND-reduces across BOTH AIV sub-blocks — an idle sub-block that never SetFlags hangs the AIC forever"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=cube-MIX (MIX_AIC_1_2 cube↔vec)"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-212-on-v351-a5-the-mix-aic-1-2-aiv-aic-cross-core-flag.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-212"]
---

# On V351 (A5), the MIX_AIC_1_2 AIV→AIC cross-core flag AND-reduces across BOTH AIV sub-blocks — an idle sub-block that never SetFlags hangs the AIC forever

> 原 OKF v1 卡号：**OL-212**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=cube-MIX (MIX_AIC_1_2 cube↔vec)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=cube-MIX (MIX_AIC_1_2 cube↔vec)`
`verified_on: soc=Ascend950PR; cann=9.0.0; kernel_type=KERNEL_TYPE_MIX_AIC_1_2 (FA-A5 multi-core graybox 2026-06-07)`

**Principle**: under `KERNEL_TYPE_MIX_AIC_1_2` — the hardware-native ratio on A5, where each of the 36 AI subsystems is **1 Cube + 2 Vector** (whitepaper §3) — a `CrossCoreSetFlag<...,0x2>`-mode flag issued AIV→AIC **AND-reduces across the 2 AIV sub-blocks**: the AIC's matching `CrossCoreWaitFlag` completes ONLY when BOTH AIV sub-blocks have issued the matching `CrossCoreSetFlag`. An idle/inactive sub-block (one that takes an early-return branch, or is starved because `BLOCK_M` only fills sub-0) never SetFlags → the group-set never reaches count → **the AIC waits forever → kernel HANGS** (timeout-killed, zero output). **Mitigation: both sub-blocks must participate in EVERY handshake** — split the producer work by row/tile range so each sub-block writes its own disjoint slice then Sets, balancing the AND-reduce.

**Concrete anchor**: FA-A5 multi-core graybox 2026-06-07 — a standalone MIX kernel **hung even at C=1** (single attention column) until BOTH AIV sub-blocks were made to run every SIG handshake; gating the Set on `subBlockIdx==0` (natural when one column only fills one sub-block) reproduced the forever-wait. Same `if ASCEND_IS_AIV { work_my_slice(); CrossCoreSetFlag<0x2,PIPE_MTE3>(SIG); }` / `if ASCEND_IS_AIC { CrossCoreWaitFlag<0x2>(SIG); }` fix as the V220 case.

**Cross-ref**: OL-190 (the V220/A3 instance of the same group-barrier symmetric-participation rule — this OL is the **V351/A5 confirmation** that OL-190 left `unverified_on: A5`), OL-206 (the broader MIX cube↔vec RESULT-handshake hand-roll ladder — prefer the managed abstraction), OL-210 (global cross-group flags), `fa_class/cross_core_sync.md` §3 (single-flag-broadcast across AIV sub-blocks).

**Evidence**: FA-A5 multi-core graybox session 2026-06-07 — C=1 hang reproduced + fixed by symmetric sub-block participation; confirms OL-190's `MIX_AIC_1_2` AND-reduce barrier behavior holds on A5 V351 (previously verified only on A3 V220).

**Other instances (predicted)**: any A5 `MIX_AIC_1_2` cube↔vec op with an AIV→AIC handoff where the producer work is naturally single-sub-block (single-column FA, single-row reduction, tail-tile epilogue) — the early-return idle sub-block is the deadlock trap; split-by-range to keep both sub-blocks live.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-212（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
