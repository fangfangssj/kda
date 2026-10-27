---
type: Runbook
title: "dav-m200 rejects half-typed arithmetic in ANY instantiated branch"
description: "dav-m200 在模板实例化时检查 half 算术——即使 half 分支在运行时 if 的死分支里也会编译报错；dtype 分支必须 `if constexpr`。"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-207"]
---

# dav-m200 rejects half-typed arithmetic in ANY instantiated branch

**EC-207** — **dav-m200 rejects half-typed arithmetic in ANY instantiated branch**: `half precision operation is not allowed in aicore function` fires for half+float / half+half inside a RUNTIME `if (std::is_same<T,...>)` / `if (kShadow)` dead branch — the dtype restriction is enforced at template instantiation (both sides of a plain `if` compile for T=half). Fix: every dtype-selecting arithmetic site behind `if constexpr` (`21_Scatter/kernel/scatter_kernel.h:455,480,491 [snippet-unavailable: source workspace not on this box]`, helpers `:164-190`). Detection: grep templated kernels for runtime dtype guards wrapping arithmetic. 21_Scatter 310p iters 1-2 2026-08-15 (`21_Scatter/failures_ledger.md:4-5`). Same aicore-C restriction family as CAND-AICORE-UINT-FLOAT-CAST-BAN; the distinct lesson is dead-branch instantiation. Cross-ref [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md). backend=ascendc.

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=any templated multi-dtype aicore kernel`

- **Error**: `half precision operation is not allowed in aicore function` — raised for `half+float` AND `half+half` expressions sitting inside a branch the current instantiation can never execute at runtime (e.g. `if (std::is_same<T, float>)` when `T=half`, or `if (kShadow)` in the non-shadow instantiation). The dtype restriction is enforced at TEMPLATE INSTANTIATION, not at runtime: both sides of a plain `if` are compiled for `T=half`, so the half arithmetic in the dead side is a hard compile error.
- **Fix**: make every dtype-selecting branch compile-time-discarding — `if constexpr (std::is_same<T, half>::value)` / `if constexpr (kShadow)` — so the dead branch is not instantiated. Shipped form: `21_Scatter/kernel/scatter_kernel.h:455,480,491 [snippet-unavailable: source workspace not on this box]` (the `kShadow` / widen-narrow dtype branches) and the `WidenToFloat`/`NarrowFromFloat` helpers at `:164-190`.
- **Detection**: grep templated kernels for runtime dtype guards (`if (std::is_same`, `if (k`) wrapping arithmetic (any `+ - * /` over GetValue/SetValue locals) — each is a latent EC-207 for the dtype whose arithmetic the guard excludes. Two build iterations were burned discovering this one site-class at a time.
- **Evidence**: 21_Scatter 310p iters 1-2 (2026-08-15): `xt.GetValue(i)+dt.GetValue(i)` (half+float) inside runtime `if (std::is_same<T,float>)` → error; after that fix, `stage.GetValue(o)+vt.GetValue(i)` (half+half) inside runtime `if (kShadow)` → same error. `21_Scatter/failures_ledger.md:4-5`.
- **Other instances (predicted)**: any one-template multi-dtype kernel on the 220x/dav-m200 family (fp16/bf16-as-int16/fp32) with per-dtype arithmetic forks. Same aicore-C dtype-restriction family as CAND-AICORE-UINT-FLOAT-CAST-BAN (unsigned→float cast ban), but the distinct lesson here is DEAD-BRANCH instantiation: the expression was never going to execute for `T=half`, yet still had to compile.
- Cross-ref [`EC-197`](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md) (dav-m200 signature surface), CAND-AICORE-UINT-FLOAT-CAST-BAN. backend=ascendc.

**孪生卡 ec-275 合并的独有内容**（ec-275-dav-m200-rejects-half-typed-arithmetic-in-any-in.md 已删除）：

> - **Fix**: make every dtype-selecting branch compile-time-discarding — `if constexpr (std::is_same<T, half>::value)` / `if constexpr (kShadow)` — so the dead branch is not instantiated. Shipped form: `21_Scatter/kernel/scatter_kernel.h:455,480,491 [snippet-unavailable: source workspace not on this box]` (the `kShadow` / widen-narrow dtype branches) and the `WidenToFloat`/`NarrowFromFloat` helpers at `:164-190`.

> - **Evidence**: 21_Scatter 310p iters 1-2 (2026-08-15): `xt.GetValue(i)+dt.GetValue(i)` (half+float) inside runtime `if (std::is_same<T,float>)` → error; after that fix, `stage.GetValue(o)+vt.GetValue(i)` (half+half) inside runtime `if (kShadow)` → same error. `21_Scatter/failures_ledger.md:4-5`.
