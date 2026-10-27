---
type: Guide
title: "Unbounded-pending cross-core flag channel overflows mode-4 line counter — drain-credit ring bound (MIX deadlock)"
description: "arch35 MIX looped-handshake kernels: any CrossCoreSetFlag channel whose producer can run unboundedly ahead overflows the mode-4 line counter under GM contention -> non-deterministic 507014. Audit every channel's run-ahead; add drain-credit ring (ConsumerReleaseMte2 form). Verified on 11_BigBirdAttention (c12 4/4-wedge -> 8/8 clean @28 cores, 50/50 PASS)."
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector]
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-212-on-v351-a5-the-mix-aic-1-2-aiv-aic-cross-core-flag.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-34-matmulimpl-with-manual-crosscoresetflag-waitflag-m.md
  role: supporting
created_at: "2026-09-26T12:23:45Z"
updated_at: "2026-10-16T00:00:00Z"
platforms: ["950"]
aliases:
- OL-615
---

> **来源说明（2026-10-16）**：本卡核心主张的实测证据来自车队私有 run（见 Evidence 节 run_id / verified_on），sources 为同缺陷族公开先例（背景，不承载本卡具体主张）；按社区评审口径降为 draft，待补可复核来源后回升。

# Unbounded-pending cross-core flag channel overflows mode-4 line counter — drain-credit ring bound (MIX deadlock)

## Evidence

- **run_id**: 5450b083-4f53-4620-a1b2-8656e7bb9889
- **soc**: Ascend950DT_9582, CANN 9.2.0
- **clock**: 2026-09-26T12:23Z
- **trigger**: 11_BigBirdAttention port_a3_to_a5: non-deterministic 507014 root-caused via X-matrix + marker peek; credit-ring fix verified 8/8 at 28 cores + 50/50 pass_a

## 现象 / 触发
arch35 (Ascend950DT_9582, CANN 9.2.0) `KERNEL_TYPE_MIX_AIC_1_2` looped-handshake kernel:
single-core sequential run clean (41K handshakes), but at >=8 concurrent AIC cores and
>=~6K handshakes/core the kernel intermittently 507014 (aicore timeout), both engines
parked at uniform wait PCs. Probabilistic; frequency grows with cores x handshakes/core.

## 根因
A CrossCoreSetFlag channel whose producer can run arbitrarily far ahead of the consumer
accumulates pending sets on the mode-4 line. Instance: AIV pass-2 produced up to n_kblk
P tiles while the AIC's MM2 consumed them one by one; under GM contention (more cores)
the AIC's MM2 K/V/P L1 loads stall, P_READY pending grows up to n_kblk (40), the line
counter overflows, a set is lost/corrupted -> the AIC's wait never completes. Ring-shaped
channels (S with depth-2 credit ring) never overflow; the un-shaped P channel did.

## 判别
- Single-core full-item-count run CLEAN + multi-core wedge = cross-core delivery, not
  intra-core accounting.
- Marker readback: ALL engines completed work-loop phases (MarkAllDone reached) while
  host sync fails = lost set, not logic deadlock.
- Deterministic repro shape wedged at cap=28/16/8 -> not fixable by grid cap.

## 修法（已验证）
Give the unbounded channel a drain-credit ring (WorkspaceQueue `ConsumerReleaseMte2`
form): consumer pre-issues RING credits per line at init and returns one credit per
consumed tile on the pipe that drained it (PIPE_MTE2 after the GM read); producer waits
a credit on its store pipe (PIPE_MTE3) before each produce. Verified: previously
4/4-wedged shape went 8/8 clean at full 28-core concurrency; full 50-case suite PASS
with det 50/50.

## 规则
Audit EVERY CrossCoreSetFlag channel in a MIX kernel for the producer's run-ahead
distance. Any channel that is not structurally bounded (ring credit, strict alternation
with <=1 pending) needs a drain-credit ring — regardless of flag-id choice (id-aliasing
hypothesis falsified: shifting the id block changed nothing).

## 相关

- [OBSERVED — 507014 deadlock in MatmulImpl + manual CrossCore* + MIX_AIC_1_2](../runbooks/compilation/mix_aic_1_2_matmulimpl_manual_crosscore_507014_deadlock.md) — 同错误码同 kernel 族、机制 UNDETERMINED 的观测；本条的 unbounded-channel-overflow 机制与该观测的关系是候选归因（未经该 op 实测），不作为其机制断言。
- [PB-34 matmulimpl_with_manual_crosscoresetflag_waitflag_m](../runbooks/compilation/pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md) — related surface（MatmulImpl + manual flag chain）；PB-34 的 KFC FFTS 槽位机制与本条不同。
- [OL-212 MIX_AIC_1_2 AIV→AIC flag AND-reduces across both AIV sub-blocks](../runbooks/compilation/ol_212_on_v351_a5_the_mix_aic_1_2_aiv_aic_cross_core_flag.md) — 同 AIV→AIC 通道的另一失效模式（空闲 sub-block 不 Set 则 AIC 永等）；与本条的 run-ahead 溢出是同一通道上的两个对立方向（过少 Set vs 过多 pending）。
