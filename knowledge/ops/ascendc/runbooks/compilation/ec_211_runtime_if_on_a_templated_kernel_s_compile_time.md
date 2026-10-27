---
type: Runbook
title: "runtime if on a templated kernel's compile-time dtype flag type-checks BOTH branches"
description: "模板 kernel 的运行时 if 会对两个分支都做类型检查——按 dtype 分叉的计算链必须 `if constexpr`，否则死分支在 T=half 实例化时报 Adds/Div 推导冲突。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-211"]
---

# runtime if on a templated kernel's compile-time dtype flag type-checks BOTH branches

**EC-211** — **runtime `if` on a templated kernel's compile-time dtype flag type-checks BOTH branches** — per-dtype compute-chain branches in a templated `__aicore__` class need `if constexpr`, else bisheng reports `no matching function for call to Adds/Div` (half-vs-float deduction conflict) from the DEAD branch at the `T=half` instantiation. swi_glu 310p 2026-08-14 (build iter 1, 3 errors → green). Cross-ref [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.

`applies_to: soc=all (C++17 template semantics, compiler-level); cann=9.0.0; bisheng=CANN 9.0.0 toolchain; op_class=all (any templated per-dtype kernel class)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=CANN 9.0.0 (swi_glu 310p, 2026-08-14)`

- **Symptom**: `no matching function for call to 'Adds'` / `'Div'` (or similar template-deduction errors, half-vs-float operand conflict) in a `template <typename T, bool kFp32>`-style `__aicore__` kernel class — even though the failing branch is DEAD for the instantiation being compiled.
- **Root cause**: a runtime `if (kFp32)` statement requires BOTH branches to be well-formed for EVERY instantiation. For `T=half`, the fp32 branch's `Adds(LocalTensor<float>, ...)` / `Div(...)` calls fail deduction against half operands; the error fires at instantiation time, not execution time, so "the branch never runs" is no defense.
- **Fix**: `if constexpr (kFp32)` — the discarded branch is not instantiated for `T=half`, so its ill-formed-for-half calls are never checked.
- **Concrete anchor** (`代码片段 swi_glu/swi_glu_vec.core.snippet.h（见文末）`, Init+ComputeChain):
  ```cpp
  if constexpr (kFp32) { Div(y, x, denom, n); }   // fp32 chain — discarded for T=half
  else               { Adds(t, x, half(0), n); }  // half chain — discarded for T=float
  ```
- **Evidence**: swi_glu 310p (2026-08-14, failures_ledger build iter 1): 3 `Adds`/`Div` half-vs-float deduce errors under runtime `if (kFp32)`; `if constexpr` → rebuild green; acceptance 41/41.
- **Cross-ref**: [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (SDK-header lookup chain — check per-dtype API overloads before writing the branch). backend=ascendc.

## 内嵌代码片段

### swi_glu/swi_glu_vec.core.snippet.h

```cpp
    static constexpr uint16_t BUF_NUM = 2;                 // double-buffered queues

    __aicore__ inline SwiGluVec() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t runLen, uint64_t total,
                                uint64_t perCore)
    {
        L_ = runLen;
        uint64_t bid = GetBlockIdx();
        start_ = bid * perCore;
        end_ = start_ + perCore;
        if (end_ > total) {
            end_ = total;
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

        pipe_.InitBuffer(inQA_, BUF_NUM, TILE * sizeof(T));
        pipe_.InitBuffer(inQB_, BUF_NUM, TILE * sizeof(T));
        pipe_.InitBuffer(outQC_, BUF_NUM, TILE * sizeof(T));
        if constexpr (kFp32) {
            pipe_.InitBuffer(onesBuf_, TILE * sizeof(float));
            LocalTensor<float> ones = onesBuf_.Get<float>();
            Duplicate<float>(ones, 1.0f, static_cast<int32_t>(TILE));
        } else {
            pipe_.InitBuffer(calcABuf_, TILE * sizeof(float));
            pipe_.InitBuffer(calcCBuf_, TILE * sizeof(float));
        }
        svId_ = pipe_.FetchEventID(HardEvent::S_V);
        vsId_ = pipe_.FetchEventID(HardEvent::V_S);
    }

    __aicore__ inline void Process()
    {
        // Multi-run batching: when a contiguous half-run is smaller than half a tile and
        // run boundaries are 32B-aligned for this dtype, gather K whole runs per tile with
        // ONE strided multi-block DataCopy per operand (a-runs sit at stride 2L in GM;
        // output runs are contiguous). Small-L shapes (e.g. dim=-1 on [128,128] fp16,
        // L=64) are scalar-pipe bound on per-piece TQue ceremony otherwise (msprof:
        // scalar_ratio 0.745 on [256,256] fp16 dim=-1).
```

**孪生卡 ec-279 合并的独有内容**（ec-279-runtime-if-on-a-templated-kernel-s-compile-time.md 已删除）：

> if constexpr (kFp32) { Div(y, x, denom, n); } // fp32 chain — discarded for T=half

> else { Adds(t, x, half(0), n); } // half chain — discarded for T=float
