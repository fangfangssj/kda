---
type: Runbook
title: "AscendC data-rearrangement vectorization — DataCopy(VECIN) load + byte-offset Gather reorder + K-group aligned store"
description: "纯数据重排 kernel 的三个阶段（加载/重排/存储）必须全部向量化：32B 对齐窗口 DataCopy 加载、字节偏移 Gather 重排、K 组批量存回——标量循环慢 2-5 倍"
tags: [task.optimization, topic.data_movement.data_copy, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P276"]
---

# AscendC data-rearrangement vectorization — DataCopy(VECIN) load + byte-offset Gather reorder + K-group aligned store

## 优化点

(2026-08-29, pixel_shuffle + pixel_unshuffle 310p — one op class, two entries)
`applies_to: soc=Ascend310P3 (dav-m200, __NPU_ARCH__==2002); cann=9.0.0; op_class=data-rearrangement (pixel_shuffle/unshuffle, permute, transpose, gather, scatter — pure data movement, zero arithmetic)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (pixel_shuffle + pixel_unshuffle 2026-08-29: 8/8 contract each {bf16×4+fp32×4, r∈{2,3,4}} bit-exact max_abs=0 + large-shape maxerr=0 up to 4.19M el/512 rows; single-core 2.27× aligned / 4.0× non-aligned over fully-scalar V1)`
`unverified_on: soc=Ascend950PR, Ascend910 (Gather byte-offset + VECIN TQue semantics should transfer; K-group math is perBlock-parameterized — re-probe before relying)`
**Principle**: a pure data-rearrangement kernel has no arithmetic — every cycle is data movement.
The scalar-first port (three nested loops: scalar GM load → scalar UB reorder → scalar GM store)
leaves ~2-5× on the table vs the vector primitives. Vectorize ALL THREE stages; none alone
suffices because each scalar loop is an independent ~N-element S-pipe bottleneck. The S-pipe
scalar GM read is the single worst offender (≈5 ns/elem vs ≈1.3 ns/elem for DataCopy).
**Recipe** (three stages; index/layout tensors built ONCE outside the row loop):
1. **Load — `TQue<VECIN,1>` + 32B-aligned window** (replace the `gm.GetValue` loop):
   `inLocal = inQue.AllocTensor<T>();` then per source row
   `DataCopy(inLocal[off], inG[aligned], loadLen)` where for a non-32B-aligned GM start
   `s0 = gmStart % perBlock; aligned = gmStart - s0; loadLen = ((s0+len+perBlock-1)/perBlock)*perBlock;`
   then `inQue.EnQue(inLocal); inLocal = inQue.DeQue<T>();` (MTE2→V sync). The row's valid data
   lands at `inLocal[off + s0 + e]`; the Gather index below adds s0 back. This is the
   gelu_kernel.h / joint_attention ApplyMaskRow canonical load. **Note**: DataCopy GM→UB
   requires 32B-aligned src/dst AND count — never index `inG[gmStart]` directly when unaligned.
2. **Reorder — byte-offset `Gather`** (replace `ub.GetValue`/`ub.SetValue`): build `idxT`
   (uint32) once: `idxT[ow] = srcElem(ow) * sizeof(T)` — Gather's `srcOffset` is **BYTE**
   offsets (not element), and `srcBaseAddr` is a byte offset added to the src base. Then
   `Gather(outLocal, inLocal, idxT, srcBaseAddr, Wout)`; srcBaseAddr absorbs the per-row
   alignment s0 (and per-row phase j for unshuffle). Count-mode Gather handles any non-32B
   count. S-pipe→V-pipe handoff of idxT needs one `SetFlag<S_V>`+`WaitFlag<S_V>` after
   building it (EC-73). Gather writes V-pipe UB → directly MTE3-visible, no Adds/Or
   materialization step.
3. **Store — K-group aligned DataCopy** (replace non-aligned scalar `gm.SetValue`):
   if `Wout % perBlock == 0` single-row `DataCopy` suffices (K=1). Else
   `K = perBlock / gcd(Wout, perBlock)` so `K*Wout % perBlock == 0`; per K-row group do K
   `Gather(outLocal[k*Wout], inLocal, idxT, srcBaseAddr_k, Wout)` with per-row `srcBaseAddr_k`
   (= `k*r*rowStride + s0` shuffle / `k*rowStride + s0 + j` unshuffle), then ONE
   `DataCopy(outG[row*Wout], outLocal, K*Wout)` — both sides 32B-aligned by construction
   (row*Wout % perBlock == 0 since row is K-aligned). Output `TQue<VECOUT,1>`
   Alloc/EnQue/DeQue/Free gives V→MTE3 sync.
**Boundary caveat (multi-core, hard)**: head/tail rows that don't form a full K-group still
need a store. Single-core: scalar store is fine. Multi-core: scalar GM `SetValue` is DROPPED
under concurrency on dav-m200 (PB-144, see P-P277) — so a multi-core data-movement kernel must
be DataCopy-only; pad/assign K-aligned ranges and route the residual with the P-P201
overlapping-32B-DataCopy-anchored-at-END trick, never a scalar GM write.
**Concrete anchor**: `代码片段 pixel_shuffle__opsmath/pixel_shuffle.core.snippet.h（见文末）`
(`PixelShuffleCompute`, mode 0/1 share one body; `PixelShuffleCompute` is the shuffle entry,
unshuffle is the same .h with the include guard + mode=1).
- pixel_shuffle + pixel_unshuffle 310p (2026-08-29): 8/8 contract each (bf16×4+fp32×4,
  r∈{2,3,4}) bit-exact (max_abs=0, pure rearrange); large-shape verify maxerr=0 up to
  4.19M el / 512 rows (also disproves the old "DataCopy load flaky ≥48 rows" quirk — that was
  the S-pipe read-BACK after DataCopy load, not the load itself; V-pipe Gather read is clean).
  Single-core vs fully-scalar V1: aligned 4.19M 10840→4785 µs (2.27×), non-aligned Win=129
  574→143 µs (4.0×). Caveat: non-aligned tiny-Win (Win=7) regresses 1.4× — the aligned-window
  load's padding dominates at 7-elem rows; not a contract shape.
**Cross-ref**: [`P-P201`](p_p201_in_kernel_ragged_tail_on_datacopypad_less_socs.md) (ragged tail + PB-144 no-scalar-store multi-core), P-P28 (TQue
auto-pipeline), P-P34 (per-element gather), P-P36 (TQueBind), EC-73 (scalar→vector handoff),
P-P277 (multi-core partitioning). backend=ascendc.

## 内嵌代码片段

### pixel_shuffle__opsmath/pixel_shuffle.core.snippet.h

```cpp

template <typename T>
__aicore__ inline void PixelShuffleCompute(GM_ADDR x, GM_ADDR y, const PixelShuffleTilingData* td)
{
    const int64_t useCoreNum = td->useCoreNum;
    const int64_t formerNum = td->formerNum;
    const int64_t formerLength = td->formerLength;
    const int64_t tailLength = td->tailLength;
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= useCoreNum) {
        return;
    }

    int64_t rowStart = 0;
    int64_t rowLen = 0;
    if (blockIdx < formerNum) {
        rowStart = blockIdx * formerLength;
        rowLen = formerLength;
    } else {
        rowStart = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
        rowLen = tailLength;
    }
    if (rowLen <= 0) {
        return;
    }
    const int64_t rowEnd = rowStart + rowLen;

    GlobalTensor<T> inG;
    GlobalTensor<T> outG;
    inG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    const int64_t N = td->N;
    const int64_t Cin = td->Cin;
    const int64_t Hin = td->Hin;
    const int64_t Win = td->Win;
    const int64_t Cout = td->Cout;
    const int64_t Hout = td->Hout;
    const int64_t Wout = td->Wout;
    const int64_t r = td->r;
```
