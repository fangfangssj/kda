---
type: Runbook
title: "dav-m200 (Ascend310P3, arch 200x): the scalar-subtraction VEC primitive Subs is absent (error: no template nam"
description: "dav-m200 没有 `Subs`（张量减标量）原语；用 `Adds(dst, src, -scalar, n)` 恒等替代。"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-194"]
---

# dav-m200 (Ascend310P3, arch 200x): the scalar-subtraction VEC primitive Subs is absent (error: no template nam

**EC-194** — dav-m200 (Ascend310P3, arch 200x): the scalar-subtraction VEC primitive `Subs` is absent (`error: no template named 'Subs' in namespace 'AscendC'`); compute scalar subtraction via `Adds(dst, src, -scalar, count)` — the universal `a + (−s)` identity, a portable fallback whenever ANY scalar-subtract primitive is unavailable or dtype-restricted (tensor-tensor `Sub` IS present on dav-m200). Same arch-surface-reduction class as EC-88 (`Cast<float,int8_t>` no arch-220x overload). COMPILE claim (build-host), device provenance UNVERIFIED — does not block it. Anchor: `代码片段 layer_norm_v4/layer_norm_v4_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]` (mean-centering step), 9/9 T1. Cross-ref [`PB-23`](pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md) (Subs dtype-restriction on arches where it EXISTS), EC-88. backend=ascendc.

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=all (any dav-m200 kernel needing per-element subtract-a-scalar)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (layer_norm_v4 310p iter-1 build, 2026-08-09)`
`unverified_on: soc=Ascend950PR (A5, arch 351x), Ascend910_220x (A3) — on those arches `Subs` EXISTS (PB-23 catalogs its dtype-restricted support set); the template-absence is dav-m200/arch-200-observed. Do not assume `Subs` is absent on A5/A3.`

- **Category**: build / compile
- **Loaded by**: Generator/Worker when a dav-m200 kernel needs to subtract a broadcast scalar from every element of a vector (the `Subs(dst, src, scalar, count)` shape).
- **Symptom**: `AscendC::Subs<float>(dst, src, scalar, count)` fails to compile on dav-m200 with `error: no template named 'Subs' in namespace 'AscendC'` (the scalar-broadcast subtraction template is not in the dav-m200 primitive surface).
- **Mechanism**: dav-m200 (arch 200x) ships a reduced VEC primitive set vs A5/220x; `Subs` (scalar-broadcast subtraction) is among the absent templates. This is the same surface-reduction class as EC-88 (`Cast<float,int8_t>` no arch-220x overload) — a VEC primitive present on newer arches but missing on the older one.
- **Fix**: substitute the algebraic identity `a − s ≡ a + (−s)` — use `Adds(dst, src, -scalar, count)`. `Adds` IS present on dav-m200; negating the scalar is a host-side / scalar-pipe op. This identity is universal (holds on every arch), so `Adds(dst, src, -s)` is a portable fallback whenever ANY scalar-subtraction primitive is unavailable or dtype-restricted. For tensor-tensor subtraction, use `Sub(dst, src1, src2, count)` (which IS present) after broadcasting the scalar into a buffer if needed.
- **Correct pattern** (anchor, `代码片段 layer_norm_v4/layer_norm_v4_kernel.h.snippet.h（见文末） [snippet-unavailable: source workspace not on this box]` — centering `x` by the row mean, the classic norm step):
  ```cpp
  // x_centered = x - mean  (Subs not on dav-m200 -> Adds with -mean)
  Adds<float>(xLocal, xLocal, -meanScalar, alignedLen);   // a + (-mean), not Subs(a, mean)
  ```
- **Evidence**: layer_norm_v4 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 build rejected `Subs<float>`; the `Adds(dst, src, -meanScalar, …)` substitution is in the shipped kernel, which builds clean and reports 9/9 T1 PASS (CPU-truth, A3 ref unreachable — see OL-25 dav-m200 provenance note). Recorded in `layer_norm_v4/failures_ledger.md` iter 1. Device provenance UNVERIFIED ([target_ascend310p](../../../../common/platforms/concepts/target_ascend310p.md) `device.runtime: NOT_RUN`) — but this is a COMPILE claim (build-host phenomenon), confirmed by the produced `.so`, NOT a device-runtime claim, so the provenance flag does not block it.
- **Cross-ref**: [`PB-23`](pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md) (catalogs `Subs`/`Adds`/… dtype-restriction on arches where they EXIST — A5/220x; the absent-on-dav-m200 case here is a strictly-older-arch surface gap, not a dtype-pair rejection), the VEC-op naming convention (`s` suffix = scalar broadcast: `Adds/Add`, `Subs/Sub`; `Subs` absent on dav-m200 ⇒ fall back to `Adds` + negation), EC-88 (companion arch-220x-overload-absence class). backend=ascendc.

## 内嵌代码片段

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

**孪生卡 ec-263 合并的独有内容**（ec-263-on-dav-m200-ascend310p3-the-scalar-subtraction-v.md 已删除）：

> - **Correct pattern** (anchor, `target/ascendc/examples/layer_norm_v4/snippets.md` — centering `x` by the row mean, the classic norm step):

> // x_centered = x - mean (Subs not on dav-m200 -> Adds with -mean)

> Adds<float>(xLocal, xLocal, -meanScalar, alignedLen); // a + (-mean), not Subs(a, mean)
