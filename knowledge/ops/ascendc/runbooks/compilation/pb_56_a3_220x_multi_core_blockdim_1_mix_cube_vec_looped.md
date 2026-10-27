---
type: Runbook
title: "a3/220x multi-core (blockDim>1) MIX cube↔vec LOOPED bidirectional handshake deadlocks (507014)"
description: "a3/220x multi-core (blockDim>1) MIX cube↔vec LOOPED bidirectional handshake deadlocks (507014)"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-56-a3-220x-multi-core-blockdim-1-mix-cube-vec-looped.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:00Z"
platforms: [a3, "950"]
aliases: ["PB-56"]
---

# a3/220x multi-core (blockDim>1) MIX cube↔vec LOOPED bidirectional handshake deadlocks (507014)

> 原 OKF v1 卡号：**PB-56**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - a hand-rolled per-core cube↔vec handshake in the LOOPED bidirectional form hangs: torch.npu.synchronize() blocks >90s, no aicore exception, prints kernel-launch

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

```yaml
applies_to:
  paradigm: ascendc
  soc: Ascend910_9382
  cann: 9.0.0
```
- **Status**: RESOLVED (2026-07-20, device-measured — multi-core FA ACHIEVED, see Resolution). This PB documents that a **HAND-ROLLED** per-core cube↔vec cross-core LOOPED ring deadlocks (mode3/mode5) AND — the key point — that FA multi-core does NOT need it. NOT a hardware wall; NOT "perf N/A" (a3 vendor baseline = `npu_fusion_attention`, measured running ~102µs).
- **Affected**: only **hand-rolled** multi-core (blockDim>1) MIX 1:1 cube↔vec handshakes on a3/220x (`CrossCoreSetFlag`/GM-slot per-core). Single-core (blockDim=1) hand-rolled handshake is fine (works + verified 17/17). The library/KFC multi-core path is NOT affected — it does not hand-roll.
- **Symptom**: a **hand-rolled** per-core cube↔vec handshake in the **LOOPED bidirectional** form hangs: `torch.npu.synchronize()` blocks >90s, no aicore exception, prints kernel-launch line then no result. Device latches AICore=100% (recovers on process-kill / >120s idle). **1-shot (non-looped)** concurrent multi-group MIX runs clean.
- **Root cause (device-measured, by elimination)**: NOT distinct-vs-shared flag-id orphan-credit — BOTH distinct-id (mode3) AND the shipped c220 `pto::TSYNC_CVID` shared-flag+GM-slot recipe (mode5, **per-round** sync) deadlock. §4's arch35 mode-4 resolution does NOT port (`INTRA_MODE=4` collapses to mode-0 on c220 via `mode & 0x3` mask). Deeper than flag-id disambiguation. See [cross_core_sync §5](../../optimizations/fa_cross_core_sync_workspacequeue.md) for the 4-mode device table. **The deeper lesson: do not hand-roll multi-core cube↔vec sync — use the library.**
- **Resolution (device-measured 2026-07-20 — multi-core FA ACHIEVED)**: extend the VERIFIED single-core base (library `MatmulImpl IterateAll<sync=true>` + per-pair PB-55 handshake) to blockDim>1, ONE head-slice per core, keeping ONLY the per-pair `MODE2`/`CV_CORE_SYNC` handshake — NO cross-core ring. **Runs DEADLOCK-FREE, precision 20/20** (5 shapes × cores{1,2,4,20}, cos 0.999998+, deterministic, bit-identical to 1-core), perf **0.186× vendor** (550µs vs `npu_fusion_attention` ~102µs @ 20 cores; recovered 14.3× over single-core 7866µs). **Key insight: FA multi-core is per-head-INDEPENDENT** — each core is a self-contained single-core FA, `MODE2` scopes the handshake to that core's OWN AIC↔AIV group (zero cross-core interaction), so the LOOPED cross-core ring (mode3/mode5 deadlock) is UNNECESSARY for head-parallel FA. Perf gap is STRUCTURAL (non-flash: materializes full [S,S] score+prob to GM per head + row-serial softmax vs vendor's fused online-softmax L1/UB-resident); next lever = flash/online-softmax rework (cf independent generated-kernel witness 0.51×), NOT a sync fix. NOTE: the *async*-KFC path (`matmul::Matmul`+KfcServer) remains a3-standalone-BLOCKED (CAND-KFC-standalone-bootstrap-teardown / PB-53 / PB-54) — the `IterateAll<sync=true>` route sidesteps it. Artifacts: [fa_a3_multicore_result](../../examples/fa_a3_multicore_result.md)（含 20/20 验证结论；mc_verify/mc_perf JSON 原始资产未迁移，见 sources 主来源所在目录）。See [cross_core_sync §5](../../optimizations/fa_cross_core_sync_workspacequeue.md)。
- **DETECTION methodology (stomp-probe)**: to distinguish a real multi-core deadlock from a kernel bug, build a minimal blockDim=2 handshake probe with 4 modes (1-shot distinct / 1-shot shared / looped distinct / looped shared+GM-slot). Run the CANDIDATE mode FIRST on a **verified-clean** device (a hang LATCHES the device — budget for it), and the known-deadlock mode LAST as control. Verify device-clean (`npu-smi` AICore≈0%, no proc, no zombie) before EACH run; a hang is a valid negative verdict; do NOT hammer a latched device (>120s idle to recover). Write the run-log incrementally to disk (the probe can die on API error — disk is the salvage).
- **large-D note (507015 refuted for the device-side head-loop)**: the 507015 fault class (single-core, D>128, N>1) is for hand-rolled **HOST** multi-head tiling; a **device-side head-loop** kernel (`MatmulImpl IterateAll<sync=true>` per head, internal K-tiling) does NOT hit it — D=256/384/512/768/1024/1280 all computed + matched golden (cos≥0.99998) at single-core on a3/220x.
- **Evidence**: DS device build+probes 2026-07-19/20 → [fa_a3_multicore_gmslot_run_log](../../examples/fa_a3_multicore_gmslot_run_log.md) (hand-rolled mode 0/1/3/5 characterization) + [fa_a3_multicore_result](../../examples/fa_a3_multicore_result.md)（mc_verify/mc_perf JSON 资产未迁移）。 Grounded in c220 `pto/npu/a2a3` TSyncCVID.hpp / TSync_Custom.hpp / TPush.hpp. Cross-ref: [cross_core_sync §5](../../optimizations/fa_cross_core_sync_workspacequeue.md) (sibling to the §4 V351/A5 RUNNABLE handshake), PB-55 (single-core both-AIV-subblocks-must-Set).

<!-- 迁移自 porter kb/target/ascendc/（PB-56，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
