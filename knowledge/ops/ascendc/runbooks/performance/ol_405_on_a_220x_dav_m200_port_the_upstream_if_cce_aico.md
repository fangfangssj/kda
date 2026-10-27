---
type: Runbook
title: "on a 220x→dav-m200 port, the upstream #if __CCE_AICORE__ < 220 branches ARE the vendor's own 200x design — rea"
description: "220x→dav-m200 移植时，上游 #if __CCE_AICORE__ < 220 分支就是厂商自己的 200x 设计——设计前先读它们（含 tiling 常量与 workspace 布局）"
tags: [task.optimization, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-405"]
---

# on a 220x→dav-m200 port, the upstream #if __CCE_AICORE__ < 220 branches ARE the vendor's own 200x design — rea

## 优化点

**OL-405** — **on a 220x→dav-m200 port, the upstream `#if __CCE_AICORE__ < 220` branches ARE the vendor's own 200x design — read them before designing anything.** The 200x algorithm structure (per-core GM partials + SyncAll + block-0 fixed-order reduce, no SetAtomicAdd), 310p tiling constants (`UB_SELF_LENGTH_310P`, `SetScheduleMode(1)`, DAV_2002 workspace sizing), and workspace layout are inline in the 220x top-level file. Positive complement to the is_inf arch35-scope note: arch35/ is unusable on a non-regbase target, but the `< 220` branches are the intended port base. histogram_v2 310p 2026-08-09 (164/164 bit-exact, det 164/164, 2.20×/2.28× device-event vs torch.histc). Cross-ref [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md)/[`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md), [`P-P201`](p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md), [`OL-130`](../compilation/ol_130_v220_libascendcl_soname_conflict.md), CAND-DAV-M200-SCATTER-COUNT-GM-PARTIALS-SYNCALL-REDUCE.

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

**Concrete anchor**: histogram_v2 310p `代码片段 histogram_v2/histogram_v2_kernel.core.snippet.h（见文末）`
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

**Cross-ref**: [`PB-105`](../compilation/pb_105_datacopypad_datacopyextparams_are_not_supported.md) / [`PB-106`](../compilation/pb_106_torch_npu_dispatcher_ops_and_a_raw_aclrt_launch.md) (dav-m200 platform constraints), [`P-P201`](p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md) (do NOT inherit the
AlignUp copy lengths inside those branches), CAND-DAV-M200-SCATTER-COUNT-GM-PARTIALS-SYNCALL-REDUCE
(the histogram structure found there), OL-130 (SDK-header lookup chain). backend=ascendc.

## 内嵌代码片段

### histogram_v2/histogram_v2_kernel.core.snippet.h

```cpp
class HistogramV2M200 {
public:
    __aicore__ inline HistogramV2M200()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, int64_t numel, int64_t chunkElems, int64_t bins,
        int64_t slotElems, int64_t ubBinsLength, float minVal, float maxVal, TPipe* tPipe)
    {
        this->numel = numel;
        this->bins = bins;
        this->slotElems = slotElems;
        this->ubBinsLength = ubBinsLength;
        this->coreNum = GetBlockNum();
        this->coreIdx = GetBlockIdx();
        this->minValue = minVal;
        this->maxValue = maxVal;
        if (this->minValue == this->maxValue) {
            this->minValue = this->minValue - 1.0f;
            this->maxValue = this->maxValue + 1.0f;
        }
        int64_t start = this->coreIdx * chunkElems;
        int64_t len = 0;
        if (start < numel) {
            len = numel - start;
            if (len > chunkElems) {
                len = chunkElems;
            }
        }
        this->myLen = len;
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ MTE_T*>(x) + start, len > 0 ? len : 1);
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), bins);
        this->syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));
        this->partialsGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(workspace + SYNC_SLOT_BYTES * this->coreNum));
        this->pipe = tPipe;
        this->pipe->InitBuffer(this->xQue, 2, TILE_ELEMS * sizeof(MTE_T));
        this->pipe->InitBuffer(this->yQue, 1, (this->ubBinsLength + ALIGNED_NUM) * sizeof(int32_t));
        this->pipe->InitBuffer(this->y1Que, 1, (this->ubBinsLength + ALIGNED_NUM) * sizeof(int32_t));
        this->pipe->InitBuffer(this->syncQue, 1, SYNC_SLOT_BYTES * this->coreNum);
```

## 适用场景

- **工程形态**：220x→dav-m200 移植时，上游 #if __CCE_AICORE__ < 220 分支就是厂商自己的 200x 设计——设计前先读它们（含 tiling 常量与 workspace 布局）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
