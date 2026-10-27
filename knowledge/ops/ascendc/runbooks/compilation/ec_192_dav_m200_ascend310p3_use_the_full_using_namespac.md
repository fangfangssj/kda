---
type: Runbook
title: "dav-m200 (Ascend310P3): use the FULL using namespace AscendC at kernel-namespace scope; selective using Ascend"
description: "kernel 命名空间内必须用完整 `using namespace AscendC`；选择性 using 会让 PIPE_V/PipeBarrier 解析失败（与 EC-6 的\"引入过多\"正好相反方向）。"
tags: [task.debugging, topic.implementation.synchronization, topic.compilation, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-192"]
---

# dav-m200 (Ascend310P3): use the FULL using namespace AscendC at kernel-namespace scope; selective using Ascend

**EC-192** — dav-m200 (Ascend310P3): use the FULL `using namespace AscendC` at kernel-namespace scope; selective `using AscendC::xxx` declarations leave the `PIPE_V` enum + `PipeBarrier<>` template unresolved (import too little — the OPPOSITE failure to [`EC-6`](ec_6_using_namespace_ascendc_simt_causes_getblockidx_am.md)'s `::Simt` ambiguity which imports too much). The kernel `.h` needs the full directive so unqualified `PipeBarrier<PIPE_V>()` / `GetBlockIdx()` resolve; the entry `.cpp` stays qualified (`AscendC::TPipe`, [`EC-191`](ec_191_duplicate_int8_t_has_no_dav_m200_overload.md)). Anchor: ge_glu_v2 310p `kernel/ge_glu_v2_kernel.h` (`namespace NsGeGluV2 { using namespace AscendC; }`), 9/9 T1+T2. Cross-ref [`EC-6`](ec_6_using_namespace_ascendc_simt_causes_getblockidx_am.md)/[`OL-14`](ol_14_using_namespace_ascendc_simt_causes_getblockidx_am.md), [`EC-191`](ec_191_duplicate_int8_t_has_no_dav_m200_overload.md). backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any AscendC kernel header authored for dav-m200)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (ge_glu_v2 310p, 2026-08-08)`
`unverified_on: soc=Ascend950PR (A5), Ascend910_220x (A3) — A5/A3 kernels in this repo use the full `using namespace AscendC` form already (EC-6/OL-14, EC-191), so the same resolution holds; the selective-using-insufficiency failure mode is dav-m200-observed and not yet reproduced on the other SoCs.`

- **Category**: build / compile
- **Loaded by**: Builder/Generator when authoring a dav-m200 (Ascend310P3) kernel header
- **Symptom**: a kernel header that imports AscendC symbols via SELECTIVE declarations (`using AscendC::GetBlockIdx; using AscendC::GetBlockNum; ...`) fails to compile — the `PIPE_V` enum value and the `PipeBarrier<>` template are reported undeclared / out-of-scope at `PipeBarrier<PIPE_V>();` call sites, even though the individually-named symbols resolve.
- **Mechanism**: a selective `using AscendC::xxx;` imports ONLY the named symbols. The `PIPE_V` enum and the `PipeBarrier<>` template are not carried by a `GetBlockIdx`/`GetBlockNum`-only import set, so they stay unresolved. This is the OPPOSITE failure to EC-6/OL-14 (where `using namespace AscendC::Simt` brings in TOO MUCH → `GetBlockIdx` ambiguity); here selective importing brings in TOO LITTLE → missing enum/template symbols.
- **Fix**: at kernel-namespace scope use the full directive `using namespace AscendC;`. This resolves `GetBlockIdx`/`GetBlockNum`, the `PIPE_V` enum, and the `PipeBarrier<>` template together. (The entry-point `.cpp` at global scope is the other case — there qualify explicitly as `AscendC::TPipe`, do NOT add a global using: see EC-191.)
- **Correct pattern** (anchor, `代码片段 ge_glu_v2/ge_glu_v2_kernel.core.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]` — `namespace NsGeGluV2 { using namespace AscendC; ... }` then unqualified `GetBlockIdx()`/`GetBlockNum()` + `PipeBarrier<PIPE_V>()`):
  ```cpp
  namespace NsGeGluV2 {
  using namespace AscendC;        // full directive — resolves PIPE_V + PipeBarrier + GetBlockIdx/Num
  template <typename T> class GeGluV2Kn {
      __aicore__ inline void ProcessTile(...) {
          // ...
          PipeBarrier<PIPE_V>();  // resolves only under full using namespace AscendC
          Mul(yMulOut, geluOut, ubA, count);
      }
  };
  }
  ```
- **Evidence**: ge_glu_v2 310p port (2026-08-08, Ascend310P3, CANN 9.0.0) — kernel header uses `using namespace AscendC;` at `namespace NsGeGluV2` scope; `PipeBarrier<PIPE_V>()` + `GetBlockIdx()`/`GetBlockNum()` resolve unqualified, kernel builds clean, 9/9 T1+T2 PASS. Peer 310p ops (celu, sin) use the same full-namespace form.
- **Cross-ref**: [`EC-6`](ec_6_using_namespace_ascendc_simt_causes_getblockidx_am.md) / [`OL-14`](ol_14_using_namespace_ascendc_simt_causes_getblockidx_am.md) (`using namespace AscendC::Simt` → `GetBlockIdx` AMBIGUITY — opposite failure, same recommended full directive), [`EC-191`](ec_191_duplicate_int8_t_has_no_dav_m200_overload.md) (entry-point `.cpp` at global scope → bare `TPipe` unresolved → qualify `AscendC::TPipe`, no global using). backend=ascendc.
- **Valid alternative to the full directive — selective `using` + explicit `AscendC::` qualification** (added 2026-08-09, scatter_list 310p, 2nd op after ge_glu_v2/ge_glu_grad_v2): the full `using namespace AscendC` (above) is sufficient but NOT necessary. A kernel that uses NONE of `PIPE_V` / `PipeBarrier<>` may instead (a) selectively `using AscendC::GetBlockIdx; using AscendC::GetBlockNum; using AscendC::DataCopy; using AscendC::GlobalTensor; ...` for the value/function symbols, AND (b) fully-qualify the enum/template symbols it does use as `AscendC::HardEvent::MTE2_MTE3`, `AscendC::SetFlag<AscendC::HardEvent::…>`, `AscendC::WaitFlag<…>`. This compiles clean on dav-m200 because every AscendC symbol still resolves — value-symbols via the selective `using`, enum/templates via explicit qualification. So EC-192's failure mode is narrower than "any selective using": it is specifically a selective-`using` set that OMITS the enum/template symbols AND leaves them unqualified. Either path works; do NOT mix a partial `using` with unqualified enum/template references.
- **Two placement facts NOT implied by either directive form** (scatter_list 310p, dav-m200):
  - **`GetTPipePtr()` lives at GLOBAL scope — write it bare or `::GetTPipePtr()`, NEVER `AscendC::GetTPipePtr()`** (which fails to compile). It is not carried by `using namespace AscendC` and is not in the `AscendC::` namespace. (EC-73's own AFTER snippet already uses the bare global form.)
  - **`GetTPipePtr()->FetchEventID(HardEvent::…)` returns `TEventID`; it MUST be `static_cast<event_t>(…)` before `SetFlag`/`WaitFlag`** — `event_t` (the flag handle) and `TEventID` (the raw id) are distinct types on dav-m200; passing the raw `TEventID` to `SetFlag` is a type error.
- **Evidence (2nd/3rd op)**: scatter_list 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — kernel header (`代码片段 scatter_list/scatter_list_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`) uses selective `using AscendC::` for value symbols + explicit `AscendC::HardEvent::`/`AscendC::SetFlag`/`AscendC::WaitFlag` qualification + bare global `GetTPipePtr()` + `static_cast<event_t>(FetchEventID(…))`; builds clean, 12/12 cases bit-exact (`reduce="update"` pure data movement, `max_abs_diff=0.0`). Confirms the symbol-resolution rules on a 2nd op family (scatter vs ge_glu) and validates the selective-using + explicit-qualification alternative. backend=ascendc.
- **Evidence (4th op) — `PIPE_V` is a GLOBAL enum, NOT an `AscendC::` member; `AscendC::PIPE_V` over-qualification fails (refines the line-above "fully-qualify enum/template symbols as `AscendC::xxx`" advice)**: group_norm_silu 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — kernel header (`代码片段 group_norm_silu/group_norm_silu_kernel.h.snippet.h（见文末）`) uses the selective-`using` alternative (`using AscendC::GlobalTensor; using AscendC::HardEvent; ...`, omitting `PIPE_V`/`PipeBarrier`) and qualifies functions explicitly (`AscendC::DataCopy`, `AscendC::PipeBarrier`, `AscendC::ReduceSum`). Iter-1 build FAILED at 内部路径（不公开）group_norm_silu_kernel.h:160 with `error: no member named 'PIPE_V' in namespace 'AscendC'` — the worker wrote `AscendC::PipeBarrier<AscendC::PIPE_V>()`, over-generalizing the "you may fully-qualify enum/template symbols as `AscendC::xxx`" note above to `PIPE_V`. **Root fact**: `PIPE_V` and the other `pipe_t` enum values (`PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_FIX` — cf. EC-15's source-derived correction) are GLOBAL enums; they are NOT members of `AscendC`, unlike `AscendC::HardEvent`. So `AscendC::PIPE_V` never compiles (any arch); reference it bare: `AscendC::PipeBarrier<PIPE_V>()` resolves because `PipeBarrier` IS in `AscendC` and `PIPE_V` resolves via global scope. Fix applied (replace all `AscendC::PIPE_V` → `PIPE_V`) → iter-2 BUILD SUCCESS (`_group_norm_silu_ext.so` produced). Peer deep_norm 310p uses the identical bare-enum form (`代码片段 deep_norm/deep_norm_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`, `AscendC::PipeBarrier<PIPE_V>()`). **Takeaway for briefs**: the `AscendC::` qualifier is valid for `HardEvent`/`TQue`/`SetFlag`/etc. (true `AscendC::` members) but FORBIDDEN for the global `pipe_t` enums — write `PIPE_V`/`PIPE_ALL`/`PIPE_MTE3` bare. COMPILE claim (build-host, device-independent), confirmed by the produced `.so`; the `device.runtime: NOT_RUN` provenance flag does not block it. backend=ascendc.
- **Evidence (5th op, primary-form confirmation on a norm class)**: layer_norm_v4 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — kernel header (`代码片段 layer_norm_v4/layer_norm_v4_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`) uses the PRIMARY EC-192 form (full `using namespace AscendC;` at kernel-namespace scope), so unqualified `PipeBarrier<PIPE_V>()` at ~7 intra-VEC call sites (`:186,189,192,195,200,217`) resolves clean; iter-1 also hit the over-qualification form `AscendC::PIPE_V` → dropped to bare `PIPE_V` per the 4th-op takeaway above. Builds clean, 9/9 T1 PASS. Confirms the bare-global-`PIPE_V` rule on a 2nd op family (norm/reduction vs scatter/ge_glu). COMPILE claim (build-host), device provenance UNVERIFIED (`device.runtime: NOT_RUN`) — does not block it. backend=ascendc.
- **Evidence (6th op — `AscendC::HardEvent::V_S` qualification + bare `EVENT_ID0` macro on a quant reduction)**: dynamic_quant 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — kernel header (`代码片段 dynamic_quant/dynamic_quant_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]`) uses the selective-`using` + explicit-qualification form at every V↔S handoff: `AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0)` / `AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0)`, passing the **bare `EVENT_ID0` macro directly** (no `static_cast<event_t>(FetchEventID(…))` indirection). Builds clean, 9/9 T1 PASS. Confirms on a 3rd op family (quant / scalar-VEC-reduction) that (a) `AscendC::HardEvent::*` qualification is the dav-m200-correct form, and (b) the `EVENT_ID0` macro (=0) is a valid direct argument to `SetFlag`/`WaitFlag` on dav-m200 — a lighter placement fact complementing the scatter_list `static_cast<event_t>(FetchEventID(…))` form (both compile; `EVENT_ID0` is the cheaper form when event-id 0 is reusable). COMPILE claim (build-host), device provenance UNVERIFIED (`device.runtime: NOT_RUN`) — does not block it. backend=ascendc.

## 内嵌代码片段

### ge_glu_v2/ge_glu_v2_kernel.core.snippet.h

```cpp
class GeGluV2Kn {
public:
    __aicore__ inline GeGluV2Kn() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR yGelu, GM_ADDR yMul,
                                int64_t splitSize, int64_t totalOutput);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(int64_t outOffset, int64_t count);

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGeluGm;
    GlobalTensor<T> yMulGm;

    int64_t m_splitSize = 0;
    int64_t m_totalOutput = 0;

    TPipe pipe;
    // Input queue: [B(0..TILE) | A(TILE..2*TILE)]
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    // Output queue: [gelu(0..TILE) | y_mul(TILE..2*TILE)]
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // fp32 computation buffers
    TBuf<QuePosition::VECCALC> fp32Buf;   // B in fp32 + GELU result
    TBuf<QuePosition::VECCALC> tmpBuf;    // intermediate polynomial
};

template <typename T>
__aicore__ inline void GeGluV2Kn<T>::Init(GM_ADDR x, GM_ADDR yGelu, GM_ADDR yMul,
                                           int64_t splitSize, int64_t totalOutput)
{
    m_splitSize = splitSize;
    m_totalOutput = totalOutput;

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGeluGm.SetGlobalBuffer((__gm__ T*)yGelu);
    yMulGm.SetGlobalBuffer((__gm__ T*)yMul);

    // Allocate UB: input 2*TILE elems, output 2*TILE elems, 2 fp32 bufs of TILE elems
    pipe.InitBuffer(inQueue, BUFFER_NUM, 2 * TILE * sizeof(T));
```

### scatter_list/scatter_list_kernel.h.snippet.h

```cpp
#ifndef SCATTER_LIST_KERNEL_H_
#define SCATTER_LIST_KERNEL_H_

#include "kernel_operator.h"

namespace scatter_list_ops {

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPosition;
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
        D_ = D;
        rowElems_ = S1 * D;     // elements per (b,n) row of var
        updateElems_ = S2 * D;  // elements per (b,n) row of updates

        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(varAddr));
        varOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(varOutAddr));
        updatesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updatesAddr));
        maskPtr_ = reinterpret_cast<__gm__ uint8_t*>(maskAddr);
        indicePtr_ = reinterpret_cast<__gm__ int32_t*>(indiceAddr);

        // UB buffer for data staging (max of row, update sizes, minimum 8 for overlap-tail)
        int64_t maxElems = rowElems_ > updateElems_ ? rowElems_ : updateElems_;
            int64_t n = idx % N_;
            ProcessPair(b, n);
        }
    }

private:
    // ── Pipeline sync helpers ──────────────────────────────────────────
    __aicore__ inline void SyncMte2ToMte3()
    {
        event_t e = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(e);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(e);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t e = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_S));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(e);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(e);
    }

    __aicore__ inline void SyncSToMte2()
    {
        event_t e = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(e);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(e);
    }
```

### group_norm_silu/group_norm_silu_kernel.h.snippet.h

```cpp
//   => fp32 pipeline is DS end-to-end:
//     pass 1: TwoSum-compensated lane sum of x           -> mean = DS / L
//     pass 2: d = x - mean_h; TwoSum-compensated sum d^2  -> var  = DS / L
//     rstd: DS Newton rsqrt from vector-Sqrt seed         (~1e-13 rel)
//     per segment: c1 = DS(rstd)*gamma, c2 = beta - c1*DS(mean) (scalar DS)
//     y = (x*c1h + c2h) + (x*c1l + c2l) with exact TwoProd/TwoSum residuals
//   DS residual error ~2^-45 relative — 4+ orders inside the compare.py
//   thresholds; verified against the vendored grader in sim_numerics.py.
//
// Algorithm skeleton (V220 g32 ancestor): per group row r (contiguous flat
// span [r*L,(r+1)*L), L=(C/G)*HW); y=(x*scale+bias), scale=rstd*gamma[c],
// bias=beta[c]-scale*mean (V220 fold, mirrors PyTorch CPU apply loop).
// fp16: fp32 compute; CAST_NONE (RNE on dav-m200, EC-104) both directions.
// No Compare/Select (PB-66), no ShiftLeft (PB-64), no DataCopyPad (PB-59),
// Adds(-s) instead of Subs (EC-93), count-form VEC ops (EC-96),
// WholeReduceSum single-repeat int32-mask (fp16 path, OL-347), full

private:
    // ---- double-single (float-float) scalar primitives ---------------------
    // TwoSum: exact a + b = s + e (branchless, 6 flops).
    __aicore__ inline void TwoSumSc(float a, float b, float& s, float& e)
    {
        s = a + b;
        float bp = s - a;
        e = (a - (s - bp)) + (b - bp);
    }

    // Veltkamp split: v = vh + vl exactly, vh/vl non-overlapping halves.
    __aicore__ inline void VeltkampSc(float v, float& vh, float& vl)
    {
        float t = v * 4097.0f;  // 2^12 + 1
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(ev);
        WaitFlag<HardEvent::MTE3_S>(ev);
    }

    __aicore__ inline void FenceSMTE3()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(ev);
        WaitFlag<HardEvent::S_MTE3>(ev);
    }

    // ---- scalar helpers ---------------------------------------------------
    __aicore__ inline float LoadScalarGB(const GlobalTensor<T>& gm, int64_t idx)
    {
```

### deep_norm/deep_norm_kernel.h.snippet.h

```cpp
        LocalTensor<float> normed = normedBuf_.Get<float>();

        AscendC::Cast(xF32, xLocal, RoundMode::CAST_NONE, tileLen);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(gxF32, gxLocal, RoundMode::CAST_NONE, tileLen);
        AscendC::PipeBarrier<PIPE_V>();
        xQue_.FreeTensor(xLocal);
    __aicore__ inline float ReduceTileSum(const LocalTensor<float> &src, int32_t count)
    {
        LocalTensor<float> dst = reduceBuf_.Get<float>();
        AscendC::ReduceSum(dst, src, dst, count);
        // Extra V_S sync to ensure final ReduceImpl write is visible to scalar read
        event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        AscendC::SetFlag<HardEvent::V_S>(evt);
        AscendC::WaitFlag<HardEvent::V_S>(evt);
        return dst.GetValue(0);
    }

    __aicore__ inline void ProcessRow(int64_t row)
```

### layer_norm_v4/layer_norm_v4_kernel.h.snippet.h

```cpp
#include "kernel_operator.h"

namespace NsLayerNormV4 {

using namespace AscendC;

constexpr int32_t LN_TILE = 2048;
    {
        LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
        DataCopy(xLocal, xGm_[gmOffset], alignedLen);
        inQueueX_.EnQue(xLocal);
        xLocal = inQueueX_.DeQue<float>();

        LocalTensor<float> gammaLocal = inQueueGamma_.AllocTensor<float>();
        DataCopy(gammaLocal, gammaGm_[gmOffset], alignedLen);
```

### dynamic_quant/dynamic_quant_kernel.h.snippet.h

```cpp
    __aicore__ inline DynamicQuantKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                int64_t totalLen, TPipe *pipe)
    {
        pipe_ = pipe;
        totalLen_ = totalLen;
        if (totalLen_ <= 0) {
            return;
        }

        alignedNum_ = (totalLen_ / DQ_ALIGN) * DQ_ALIGN;
        tileLen_ = DQ_TILE;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x), totalLen_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y), totalLen_);
        xRaw_ = reinterpret_cast<__gm__ half *>(x);
        pipe_->InitBuffer(workBuf_, tileLen_ * sizeof(half));
        pipe_->InitBuffer(f32Buf_, tileLen_ * sizeof(float));
        pipe_->InitBuffer(i32Buf_, tileLen_ * sizeof(int32_t));
        pipe_->InitBuffer(tailInBuf_, DQ_ALIGN * sizeof(half));
        pipe_->InitBuffer(tailOutBuf_, DQ_ALIGN * sizeof(int8_t));
    }

            half tailMax = ProcessTailMax();
            if (static_cast<float>(tailMax) > static_cast<float>(runningMax)) {
                runningMax = tailMax;
            }
        }

        // ── Compute scale ─────────────────────────────────────────────────
        LocalTensor<half> absLocal = absBuf_.Get<half>();
        LocalTensor<half> workLocal = workBuf_.Get<half>();

        AscendC::Abs(absLocal, inDeq, count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::ReduceMax(absLocal, absLocal, workLocal, count, false);
        AscendC::PipeBarrier<PIPE_V>();

    __aicore__ inline half ProcessTailMax()
    {
        const int32_t tailNum = static_cast<int32_t>(totalLen_ - alignedNum_);
        LocalTensor<half> tin = tailInBuf_.Get<half>();

        for (int32_t i = 0; i < tailNum; ++i) {
            tin.SetValue(i, xRaw_[alignedNum_ + i]);

        LocalTensor<half> absLocal = absBuf_.Get<half>();
        AscendC::Abs(absLocal, tin, DQ_ALIGN);
        AscendC::PipeBarrier<PIPE_V>();

        LocalTensor<half> workLocal = workBuf_.Get<half>();
        AscendC::ReduceMax(absLocal, absLocal, workLocal, DQ_ALIGN, false);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outLocal, inDeq, RoundMode::CAST_ROUND, count);
        AscendC::PipeBarrier<PIPE_V>();

        inQue_.FreeTensor(inDeq);
        outQue_.EnQue(outLocal);

        LocalTensor<int8_t> outDeq = outQue_.DeQue<int8_t>();
        AscendC::DataCopy(yGm_[offset], outDeq, count);
        outQue_.FreeTensor(outDeq);
    }

    __aicore__ inline void ProcessTailQuant(float scaleInv)
    {
        const int32_t tailNum = static_cast<int32_t>(totalLen_ - alignedNum_);
        LocalTensor<half> tin = tailInBuf_.Get<half>();
        LocalTensor<int8_t> tout = tailOutBuf_.Get<int8_t>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tf32, tf32, scaleInv, DQ_ALIGN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(tin, tf32, RoundMode::CAST_NONE, DQ_ALIGN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(tout, tin, RoundMode::CAST_ROUND, DQ_ALIGN);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
```

**孪生卡 ec-261 合并的独有内容**（ec-261-on-dav-m200-ascend310p3-use-the-full-using-names.md 已删除）：

> using namespace AscendC; // full directive — resolves PIPE_V + PipeBarrier + GetBlockIdx/Num

> PipeBarrier<PIPE_V>(); // resolves only under full using namespace AscendC

> - **Evidence (2nd/3rd op)**: scatter_list 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — kernel header (`target/ascendc/examples/scatter_list/snippets.md,62,87-98`) uses selective `using AscendC::` for value symbols + explicit `AscendC::HardEvent::`/`AscendC::SetFlag`/`AscendC::WaitFlag` qualification + bare global `GetTPipePtr()` + `static_cast<event_t>(FetchEventID(…))`; builds clean, 12/12 cases bit-exact (`reduce="update"` pure data movement, `max_abs_diff=0.0`). Confirms the symbol-resolution rules on a 2nd op family (scatter vs ge_glu) and validates the selective-using + explicit-qualification alternative. backend=ascendc.

> - **Evidence (4th op) — `PIPE_V` is a GLOBAL enum, NOT an `AscendC::` member; `AscendC::PIPE_V` over-qualification fails (refines the line-above "fully-qualify enum/template symbols as `AscendC::xxx`" advice)**: group_norm_silu 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — kernel header (`target/ascendc/examples/group_norm_silu/snippets.md`) uses the selective-`using` alternative (`using AscendC::GlobalTensor; using AscendC::HardEvent; ...`, omitting `PIPE_V`/`PipeBarrier`) and qualifies functions explicitly (`AscendC::DataCopy`, `AscendC::PipeBarrier`, `AscendC::ReduceSum`). Iter-1 build FAILED at 内部路径（不公开）group_norm_silu_kernel.h:160 with `error: no member named 'PIPE_V' in namespace 'AscendC'` — the worker wrote `AscendC::PipeBarrier<AscendC::PIPE_V>()`, over-generalizing the "you may fully-qualify enum/template symbols as `AscendC::xxx`" note above to `PIPE_V`. **Root fact**: `PIPE_V` and the other `pipe_t` enum values (`PIPE_S`, `PIPE_V`, `PIPE_M`, `PIPE_MTE1`, `PIPE_MTE2`, `PIPE_MTE3`, `PIPE_ALL`, `PIPE_FIX` — cf. EC-15's source-derived correction) are GLOBAL enums; they are NOT members of `AscendC`, unlike `AscendC::HardEvent`. So `AscendC::PIPE_V` never compiles (any arch); reference it bare: `AscendC::PipeBarrier<PIPE_V>()` resolves because `PipeBarrier` IS in `AscendC` and `PIPE_V` resolves via global scope. Fix applied (replace all `AscendC::PIPE_V` → `PIPE_V`) → iter-2 BUILD SUCCESS (`_group_norm_silu_ext.so` produced). Peer deep_norm 310p uses the identical bare-enum form (`target/ascendc/examples/deep_norm/snippets.md`, `AscendC::PipeBarrier<PIPE_V>()`). **Takeaway for briefs**: the `AscendC::` qualifier is valid for `HardEvent`/`TQue`/`SetFlag`/etc. (true `AscendC::` members) but FORBIDDEN for the global `pipe_t` enums — write `PIPE_V`/`PIPE_ALL`/`PIPE_MTE3` bare. COMPILE claim (build-host, device-independent), confirmed by the produced `.so`; the `device.runtime: NOT_RUN` provenance flag does not block it. backend=ascendc.

> - **Evidence (5th op, primary-form confirmation on a norm class)**: layer_norm_v4 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — kernel header (`target/ascendc/examples/layer_norm_v4/snippets.md`) uses the PRIMARY EC-192 form (full `using namespace AscendC;` at kernel-namespace scope), so unqualified `PipeBarrier<PIPE_V>()` at ~7 intra-VEC call sites (`:186,189,192,195,200,217`) resolves clean; iter-1 also hit the over-qualification form `AscendC::PIPE_V` → dropped to bare `PIPE_V` per the 4th-op takeaway above. Builds clean, 9/9 T1 PASS. Confirms the bare-global-`PIPE_V` rule on a 2nd op family (norm/reduction vs scatter/ge_glu). COMPILE claim (build-host), device provenance UNVERIFIED (`device.runtime: NOT_RUN`) — does not block it. backend=ascendc.

> - **Evidence (6th op — `AscendC::HardEvent::V_S` qualification + bare `EVENT_ID0` macro on a quant reduction)**: dynamic_quant 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, ACLRT_LAUNCH_KERNEL+pybind verify path) — kernel header (`target/ascendc/examples/dynamic_quant/snippets.md,144,160,161,171,172,209,210,217,218`) uses the selective-`using` + explicit-qualification form at every V↔S handoff: `AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0)` / `AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0)`, passing the **bare `EVENT_ID0` macro directly** (no `static_cast<event_t>(FetchEventID(…))` indirection). Builds clean, 9/9 T1 PASS. Confirms on a 3rd op family (quant / scalar-VEC-reduction) that (a) `AscendC::HardEvent::*` qualification is the dav-m200-correct form, and (b) the `EVENT_ID0` macro (=0) is a valid direct argument to `SetFlag`/`WaitFlag` on dav-m200 — a lighter placement fact complementing the scatter_list `static_cast<event_t>(FetchEventID(…))` form (both compile; `EVENT_ID0` is the cheaper form when event-id 0 is reusable). COMPILE claim (build-host), device provenance UNVERIFIED (`device.runtime: NOT_RUN`) — does not block it. backend=ascendc.

> - **Evidence (7th op — `FetchEventID` return type bites a greenfield scan kernel)**: cumsum 310p (2026-08-18, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane) — iter-1 build failed on `TPipe::FetchEventID`: it returns `TEventID` (signed char), not `event_t`, on this SDK; the `static_cast<event_t>(FetchEventID(…))` form in the rule above is the fix. COMPILE claim (build-host). `cumsum/failures_ledger.md` build iter 1. backend=ascendc.

## 适用场景

- **工程形态**：kernel 命名空间内必须用完整 `using namespace AscendC`。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
