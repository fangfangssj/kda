---
type: Runbook
title: "helpers in a shared tiling header called from aicore code must be `__aicore__ inline` — a host-flavored `inline` helper fails the device compile"
description: "host 与 kernel 共享头文件里的普通 `inline` 辅助函数被 aicore 调用会报 host-from-aicore——共享头里的辅助函数都要标 `__aicore__`。"
tags: [task.debugging, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-298"]
---

# helpers in a shared tiling header called from aicore code must be `__aicore__ inline` — a host-flavored `inline` helper fails the device compile

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any kernel TU sharing helper functions between the host tiling pass and device code)`
- **Symptom**: device-pass compile error `call to [host] function from [aicore]` when aicore kernel code calls a helper defined as a plain `inline` function in a header shared with host code (e.g. a `*_tiling.h` consumed by both the host tiling pass and the kernel TU).
- **Root cause**: an unmarked `inline` helper compiles as a host function; the bisheng device pass rejects host-function calls from `__aicore__` code.
- **Fix**: mark every such helper `__aicore__ inline` — `代码片段 27_MultiMaskAttentionAggregation/mmagg_tiling.h.snippet.h（见文末）` (`__aicore__ inline int64_t QuantumOf(...)` / `RoundUp` / `RowPitchOf` / `BatchRowsOf`). Marked helpers remain callable from the host tiling pass.
- **Evidence**: 27_MultiMaskAttentionAggregation build iter 2a (2026-09-04): marking the 4 tiling-header helpers `__aicore__ inline` cleared the error in the same build round.
- **Other instances (predicted)**: any 310p port that factors layout arithmetic (round-up, pitch, row-batching) into a shared tiling header.
- **Cross-ref**: [`EC-290`](ec_290_bisheng_310p_dav_m200_backend_rejects_plain_name.md) (same lane, `constexpr` internal-linkage class). backend=ascendc.

## 内嵌代码片段

### 27_MultiMaskAttentionAggregation/mmagg_tiling.h.snippet.h

```cpp
// (Header-only consumers include kernel_operator.h first anyway; this makes
// the header self-contained for static analysis and standalone TUs.)
#include "kernel_operator.h"

namespace mmagg {

// ---- 32B alignment quantum (PB-22) --------------------------------------
// DataCopy GM<->UB addresses and counts must be 32B multiples.
__aicore__ inline int64_t QuantumOf(int64_t elemSize) { return 32 / elemSize; }

__aicore__ inline int64_t RoundUp(int64_t v, int64_t q) { return (v + q - 1) / q * q; }

// ---- row pitch (NP) rule -------------------------------------------------
// Softmax rows are reduced with repeat-mode WholeReduce (mask <= 64) plus
// BlockReduce folding levels (each level consumes 64-elem groups and shrinks
// the per-row count by 8). Folding groups must never straddle a row, so the
// row pitch must be:
//   multiple of 64            (level-0 groups tile rows exactly)
//   multiple of 512 when >448 (level-1 groups tile the NP/8 partial rows)
// Pads inside a row are exact reduction identities: -FLT_MAX for the rowmax
// pass (x buffer), +0.0 for the sum passes (e / prod buffers).
__aicore__ inline int64_t RowPitchOf(int64_t n)
{
    return (n > 448) ? RoundUp(n, 512) : RoundUp(n, 64);
}

// ---- batch (V) rule ------------------------------------------------------
// V rows of NP fp32 elems per softmax batch. 255 = repeatTime field width
// (8-bit, with margin — same cap class as 7_Sum's 248/255 note).
constexpr int64_t BATCH_F32_ELEMS = 4096; // V*NP cap (fp32 elems)
constexpr int64_t V_MAX = 255;
constexpr int64_t V_MIN = 1;
```
