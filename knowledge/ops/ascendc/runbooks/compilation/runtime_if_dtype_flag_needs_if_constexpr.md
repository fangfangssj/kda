---
type: Runbook
title: "runtime `if` on a templated kernel's compile-time dtype flag type-checks both branches — use `if constexpr`"
description: "runtime `if` on a templated kernel's compile-time dtype flag type-checks both branches — use `if constexpr`"
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
# EC-211 — EC-211: A5Ops additive extension

> 关联：同一 EC-211 主题已由既有卡 [ec_211_runtime_if_on_a_templated_kernel_s_compile_time.md](ec_211_runtime_if_on_a_templated_kernel_s_compile_time.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all（C++17 模板语义，编译器层面）；CANN 9.0.0；bisheng=CANN 9.0.0 toolchain；op_class=all（任何按 dtype 模板化的 kernel 类）。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0（swi_glu 310p，2026-08-14，build iter 1 三处错误改 `if constexpr` 后转绿，验收 41/41）。

**EC-211** — **runtime `if` on a templated kernel's compile-time dtype flag type-checks BOTH branches** — per-dtype compute-chain branches in a templated `__aicore__` class need `if constexpr`, else bisheng reports `no matching function for call to Adds/Div` (half-vs-float deduction conflict) from the DEAD branch at the `T=half` instantiation. swi_glu 310p 2026-08-14 (build iter 1, 3 errors → green). Cross-ref [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md). backend=ascendc.

- Former A5Ops ID: `EC-211`
- Fusion disposition: `additive_new`

`applies_to: soc=all (C++17 template semantics, compiler-level); cann=9.0.0; bisheng=CANN 9.0.0 toolchain; op_class=all (any templated per-dtype kernel class)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=CANN 9.0.0 (swi_glu 310p, 2026-08-14)`

- **Symptom**: `no matching function for call to 'Adds'` / `'Div'` (or similar template-deduction errors, half-vs-float operand conflict) in a `template <typename T, bool kFp32>`-style `__aicore__` kernel class — even though the failing branch is DEAD for the instantiation being compiled.
- **Root cause**: a runtime `if (kFp32)` statement requires BOTH branches to be well-formed for EVERY instantiation. For `T=half`, the fp32 branch's `Adds(LocalTensor<float>, ...)` / `Div(...)` calls fail deduction against half operands; the error fires at instantiation time, not execution time, so "the branch never runs" is no defense.
- **Fix**: `if constexpr (kFp32)` — the discarded branch is not instantiated for `T=half`, so its ill-formed-for-half calls are never checked.
- **Concrete anchor** (`workspace/swi_glu/kernel/swi_glu_vec.h`, Init+ComputeChain):
  ```cpp
  if constexpr (kFp32) { Div(y, x, denom, n); }   // fp32 chain — discarded for T=half
  else               { Adds(t, x, half(0), n); }  // half chain — discarded for T=float
  ```
- **Evidence**: swi_glu 310p (2026-08-14, failures_ledger build iter 1): 3 `Adds`/`Div` half-vs-float deduce errors under runtime `if (kFp32)`; `if constexpr` → rebuild green; acceptance 41/41.
- **Cross-ref**: OL-80 (SDK-header lookup chain — check per-dtype API overloads before writing the branch). backend=ascendc.
