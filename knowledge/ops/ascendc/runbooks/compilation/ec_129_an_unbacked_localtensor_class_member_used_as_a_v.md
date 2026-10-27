---
type: Runbook
title: "an unbacked `LocalTensor` class member used as a VEC destination/source writes through a garbage UB address — silent, channel-selective corruption that neither the compiler nor the static checks flag"
description: "首次 NPU 运行只坏一个输出通道（fp32 flaky、fp16/bf16 确定性 ≈0、特定 shape NaN），编译期无任何告警，静态检查也抓不到——疑似初始化/同步类隐患的标志性症状。"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.vector, topic.implementation.masking]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-129", "LocalTensor"]
---

# an unbacked `LocalTensor` class member used as a VEC destination/source writes through a garbage UB address — silent, channel-selective corruption that neither the compiler nor the static checks flag

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a (C++ object-lifecycle discipline, arch-independent); op_class=all (any kernel holding LocalTensor members)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (23_HyenaFftSizePaddingRfft, 2026-09-05 — 27-cell S×dtype matrix post-fix at ≤1.5e-8)`
`unverified_on: soc=Ascend950PR, Ascend910_V220 (pattern is arch-independent C++ discipline, expected to transfer, unmeasured)`

- **Symptom**: first-ever NPU run corrupts ONE output channel while the other sits at 1.5e-8 — flaky across process states for fp32, deterministically ≈0-scale for fp16/bf16, NaN at particular shapes. Nothing flags it at compile time, and neither `ascendc_static_check.py` nor `pre_build_check.py` catches it.
- **Root cause**: a `LocalTensor<T> m_;` declared as a class member and passed to `Cast`/`Gather`/`DataCopy` without ever being assigned storage holds a garbage UB address. VEC writes land wherever that address points; which real buffer (twiddle tables, accumulator banks) gets clobbered depends on the garbage value — hence the channel-selective, flaky-looking signature that resembles a race but is not one.
- **Fix**: back every `LocalTensor` member before first VEC use — `m_ = tbuf.Get<T>()` or `m_ = scr_[SLOT].ReinterpretCast<T>()` in Init. Anchor: `23_HyenaFftSizePaddingRfft/kernel/hyenarfft_kernel.h:224 [snippet-unavailable: source workspace not on this box]` (`midxI_ = scr_[SS_MIDXI].ReinterpretCast<int32_t>();`; use sites at `:643-667`).
- **Suggested static-check rule (harness)**: every `LocalTensor<...> name_;` class member must appear on the LHS of an assignment from `<buf>.Get<...>` or `...ReinterpretCast<...>` before any use — candidate rule for `ascendc_static_check.py` / `pre_build_check.py` so the next customer catches this pre-silicon.
- **Evidence**: 23_HyenaFftSizePaddingRfft 310p (2026-09-05, failures_ledger precision iter 1): real channel max 9.75e-2 flaky f32 / NaN f16 while imag stayed at 1.5e-8; after the SS_MIDXI scratch backing, all three dtypes pass at ≤1.5e-8 over the 27-cell S×dtype matrix, then 49/49 PASS_T1 strict + 78/78 edge vs recomputed fp64 (`23_HyenaFftSizePaddingRfft/verification.json`; `23_HyenaFftSizePaddingRfft/failures_ledger.md:7`).
- **Other instances (predicted)**: any kernel refactoring VEC temporaries from locals into class members (scratch-pool style); kernels holding index-tensor members feeding `Gather`.
- **Cross-ref**: [`EC-62`](ec_62_tbuf_workspace_buffer_used_without_pipe_initbuffer.md) (TBuf without InitBuffer → 507035 — the loud sibling; this one is silent), [`PB-24`](pb_24_v220_tbuf_getvalue_returns_sequential_position_val.md) (TBuf GetValue corruption — different mechanism, same "storage layer corrupts silently" class). backend=ascendc.
