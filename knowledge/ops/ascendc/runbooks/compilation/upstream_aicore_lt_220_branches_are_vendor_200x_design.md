---
type: Runbook
title: "On a 220x→dav-m200 port the upstream #if __CCE_AICORE__ < 220 branches are the vendor's own 200x design — read them before designing anything"
description: "On a 220x→dav-m200 port the upstream #if __CCE_AICORE__ < 220 branches are the vendor's own 200x design — read them before designing anything"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["SoftSyncAllImpl", "SyncAll", "UB_SELF_LENGTH_310P"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-405 — OL-405: A5Ops additive extension

> 关联：同一 OL-405 主题已由既有卡 [ol_405_on_a_220x_dav_m200_port_the_upstream_if_cce_aico.md](../performance/ol_405_on_a_220x_dav_m200_port_the_upstream_if_cce_aico.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200 as TARGET of a 220x-family source port); cann=9.0.0; bisheng=n/a; op_class=all (port direction / source-reading order)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (histogram_v2 310p, 2026-08-09)。

**OL-405** — **on a 220x→dav-m200 port, the upstream `#if __CCE_AICORE__ < 220` branches ARE the vendor's own 200x design — read them before designing anything.** The 200x algorithm structure (per-core GM partials + SyncAll + block-0 fixed-order reduce, no SetAtomicAdd), 310p tiling constants (`UB_SELF_LENGTH_310P`, `SetScheduleMode(1)`, DAV_2002 workspace sizing), and workspace layout are inline in the 220x top-level file. Positive complement to the is_inf arch35-scope note: arch35/ is unusable on a non-regbase target, but the `< 220` branches are the intended port base. histogram_v2 310p 2026-08-09 (164/164 bit-exact, det 164/164, 2.20×/2.28× device-event vs torch.histc). Cross-ref [PB-105](dav_m200_datacopypad_unsupported_both_directions.md)/[PB-106](torch_npu_dispatcher_raw_launch_not_stream_ordered.md), P-P201（A5Ops 绑定外部引用；本语料未收录目标）, [OL-130](ol_130_v220_libascendcl_soname_conflict.md), CAND-DAV-M200-SCATTER-COUNT-GM-PARTIALS-SYNCALL-REDUCE.

- Former A5Ops ID: `OL-405`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200 as TARGET of a 220x-family source port); cann=9.0.0; bisheng=n/a; op_class=all (port direction / source-reading order)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (histogram_v2 310p, 2026-08-09)`
`unverified_on: soc=Ascend310B / other 200x-family SoCs (same `< 220` branch family, but tiling constants may differ — re-check before relying)`

**Principle**: upstream 220x-family op_kernel source frequently carries the dav-m200 (200x)
implementation INLINE as the `#if __CCE_AICORE__ < 220` conditional branches — not as a
separate file. On a TARGET=310p port those branches are the vendor's own 200x path, complete
with 310p tiling constants (`UB_SELF_LENGTH_310P`, `SetScheduleMode(1)`, DAV_2002 workspace
sizing in op_tiling). Read them FIRST, before designing anything: the algorithm structure
(here: per-core private GM partial histograms + `SyncAll` + block-0 fixed-order reduce —
deterministic by construction, NO `SetAtomicAdd`, which only the ≥220 branch needs), the
tiling constants, and the workspace layout are all already there. This is the positive
complement to the is_inf 310p scope note (2026-08-09, in the port-base entry above): arch35/
artifacts are unusable on a non-regbase target, but the `< 220` branches inside the 220x
top-level file ARE the intended port base.

**Concrete anchor**: histogram_v2 310p `workspace/histogram_v2/kernel/histogram_v2_kernel.h`
— ReduceYPart / ProcessBigBins ported from the upstream `< 220` branches (workspace scratch,
gitignored). dav-m200 `SyncAll` is the SDK's `SoftSyncAllImpl` (per-block GM slots, ungated
in CANN 安装树内 kernel_operator_block_sync_intf.h).

**Evidence**: histogram_v2 310p (2026-08-09, port_a3_to_a5 TARGET=310p, first instance): the
`< 220` branch design ported nearly verbatim → 164/164 bit-exact vs kernel-semantics CPU
truth, det 164/164, perf 2.20×/2.28× device-event median vs torch.histc (two back-to-back
runs). The iter-1 bigbins race was NOT a design flaw in the branch — it was three missing
MTE3_MTE2 interlocks (CAND-DAV-M200-MULTIPART-RMW-INTERLOCK-TRIO).

**Other instances (predicted)**: every 220x→dav-m200 port where the upstream op_kernel file
has `#if __CCE_AICORE__` conditionals — the `< 220` side is the 200x algorithm + tiling
reference; also applies when auditing which upstream behaviors are 220x-only (the `>= 220`
side, e.g. SetAtomicAdd histogram accumulation) and must be REPLACED rather than ported.

**Cross-ref**: PB-105 / PB-106 (dav-m200 platform constraints), P-P201 (do NOT inherit the
AlignUp copy lengths inside those branches), CAND-DAV-M200-SCATTER-COUNT-GM-PARTIALS-SYNCALL-REDUCE
(the histogram structure found there), OL-130 (SDK-header lookup chain). backend=ascendc.
