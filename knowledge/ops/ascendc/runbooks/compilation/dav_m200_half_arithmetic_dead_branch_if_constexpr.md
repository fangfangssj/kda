---
type: Runbook
title: "dav-m200 rejects half-typed arithmetic in any instantiated branch — dtype-selecting branches need `if constexpr`"
description: "dav-m200 rejects half-typed arithmetic in any instantiated branch — dtype-selecting branches need `if constexpr`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-207 — EC-207: A5Ops additive extension

> 关联：同一 EC-207 主题已由既有卡 [ec_207_dav_m200_rejects_half_typed_arithmetic_in_any_in.md](ec_207_dav_m200_rejects_half_typed_arithmetic_in_any_in.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3；CANN 9.0.0；op_class=任何模板化多 dtype aicore kernel。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（21_Scatter 310p iters 1-2，2026-08-15）。

**EC-207** — **dav-m200 rejects half-typed arithmetic in ANY instantiated branch**: `half precision operation is not allowed in aicore function` fires for half+float / half+half inside a RUNTIME `if (std::is_same<T,...>)` / `if (kShadow)` dead branch — the dtype restriction is enforced at template instantiation (both sides of a plain `if` compile for T=half). Fix: every dtype-selecting arithmetic site behind `if constexpr` (内部路径（不公开）workspace/21_Scatter/kernel/scatter_kernel.h:455,480,491, helpers `:164-190`). Detection: grep templated kernels for runtime dtype guards wrapping arithmetic. 21_Scatter 310p iters 1-2 2026-08-15 (内部路径（不公开）workspace/21_Scatter/failures_ledger.md:4-5). Same aicore-C restriction family as CAND-AICORE-UINT-FLOAT-CAST-BAN; the distinct lesson is dead-branch instantiation. Cross-ref [EC-197](cann_9_0_0_dav_m200_mask_form_vec_signatures.md). backend=ascendc.

- Former A5Ops ID: `EC-207`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=any templated multi-dtype aicore kernel`

- **Error**: `half precision operation is not allowed in aicore function` — raised for `half+float` AND `half+half` expressions sitting inside a branch the current instantiation can never execute at runtime (e.g. `if (std::is_same<T, float>)` when `T=half`, or `if (kShadow)` in the non-shadow instantiation). The dtype restriction is enforced at TEMPLATE INSTANTIATION, not at runtime: both sides of a plain `if` are compiled for `T=half`, so the half arithmetic in the dead side is a hard compile error.
- **Fix**: make every dtype-selecting branch compile-time-discarding — `if constexpr (std::is_same<T, half>::value)` / `if constexpr (kShadow)` — so the dead branch is not instantiated. Shipped form: 内部路径（不公开）workspace/21_Scatter/kernel/scatter_kernel.h:455,480,491 (the `kShadow` / widen-narrow dtype branches) and the `WidenToFloat`/`NarrowFromFloat` helpers at `:164-190`.
- **Detection**: grep templated kernels for runtime dtype guards (`if (std::is_same`, `if (k`) wrapping arithmetic (any `+ - * /` over GetValue/SetValue locals) — each is a latent EC-207 for the dtype whose arithmetic the guard excludes. Two build iterations were burned discovering this one site-class at a time.
- **Evidence**: 21_Scatter 310p iters 1-2 (2026-08-15): `xt.GetValue(i)+dt.GetValue(i)` (half+float) inside runtime `if (std::is_same<T,float>)` → error; after that fix, `stage.GetValue(o)+vt.GetValue(i)` (half+half) inside runtime `if (kShadow)` → same error. 内部路径（不公开）workspace/21_Scatter/failures_ledger.md:4-5.
- **Other instances (predicted)**: any one-template multi-dtype kernel on the 220x/dav-m200 family (fp16/bf16-as-int16/fp32) with per-dtype arithmetic forks. Same aicore-C dtype-restriction family as CAND-AICORE-UINT-FLOAT-CAST-BAN (unsigned→float cast ban), but the distinct lesson here is DEAD-BRANCH instantiation: the expression was never going to execute for `T=half`, yet still had to compile.
- Cross-ref [EC-197](cann_9_0_0_dav_m200_mask_form_vec_signatures.md) (dav-m200 signature surface), CAND-AICORE-UINT-FLOAT-CAST-BAN. backend=ascendc.
