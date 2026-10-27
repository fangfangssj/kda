---
type: Runbook
title: "lambdas are HOST-ONLY under bisheng"
description: "bisheng 下 lambda 是 host-only——`__aicore__` 成员函数里的 lambda 编译报错，要提为命名的 `__aicore__ inline` 成员函数。"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-209"]
---

# lambdas are HOST-ONLY under bisheng

**EC-209** — **lambdas are HOST-ONLY under bisheng**: a lambda inside an `__aicore__` kernel member function fails with "call to [host] function from [aicore] function" — hoist to a named `__aicore__ inline` member function, which also unblocks dependent-name `tile.template ReinterpretCast<half>()` lookup (rejected in the lambda body). Detection: grep kernel headers for `[&](` / `[](` before first build. 22_Nonzero 310p build iter-1 2026-08-15 (lambda→`FlushStream` member fn, next build PASS). Cross-ref [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.

`applies_to: soc=all (bisheng CCE front-end language restriction on aicore code, observed on arch-200); cann=9.0.0; bisheng=CCE bundled with CANN 9.0.0 (exact version unrecorded — language-surface restriction, not codegen); op_class=any aicore kernel phase using inline lambdas`

- **Error**: 内部路径（不公开）nonzero_kernel.h:335:28 note: candidate function not viable: call to [host] function from [aicore] function (lambda flushStream)` — lambdas carry a `[host]` attribute under bisheng and cannot be invoked from `[aicore]` code.
- **Fix**: convert the lambda to a named `__aicore__ inline` member function (`FlushStream(s, cap)`). Member-function scope also lets dependent-name template calls resolve with the explicit spelling `tile.template ReinterpretCast<half>()`, which the lambda body rejected (2 dependent-name errors in the same iter).
- **Detection**: grep kernel headers for `[&](` / `[](` before first build — every hit inside an aicore phase is this EC.
- **Evidence**: 22_Nonzero 310p build iter-1 (2026-08-15): lambda→`FlushStream` member fn + `.template` fix; next build PASS (`代码片段 22_Nonzero/nonzero_kernel.h.snippet.h（见文末）` — the shipped named member function; `22_Nonzero/failures_ledger.md` iter 1).
- **Other instances (predicted)**: any port that copies a host-side C++ idiom (immediately-invoked lambda, closure capture) into a kernel phase body.
- Cross-ref [`OL-80`](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md) (grep the SDK API surface before coding — lambdas appear in zero aicore usage patterns in the SDK headers). backend=ascendc.

## 内嵌代码片段

### 22_Nonzero/nonzero_kernel.h.snippet.h

```cpp
        // Position ramp [0..TILE) — reused by every tile's GatherMask
        // (mask=len bounds consumption to the first len entries).
        LocalTensor<int32_t> pos = posBuf_.Get<int32_t>();
        CreateVecIndex(pos, static_cast<int32_t>(0), static_cast<uint32_t>(NZ_TILE));
        // S-pipe (CreateVecIndex is __out_pipe__(V)) — first consumer is also
        // V-pipe (GatherMask), same-pipe ordered; no fence needed.
    }
        // tiling is kept for dLast < NZ_ROW_TILE_MIN; dLast >= it uses
        // per-row tiles exactly like the emit pass.
        int64_t tIdx = 0;  // per-core-relative tile index (count==emit seq)
        if (dLast_ >= NZ_ROW_TILE_MIN) {
            for (int64_t a = rowStart_; a < rowEnd_; ++a) {
                for (int64_t t = 0; t < dLast_; t += NZ_TILE) {
                    const int32_t len = static_cast<int32_t>(
```

## 适用场景

- **工程形态**：bisheng 下 lambda 是 host-only。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
