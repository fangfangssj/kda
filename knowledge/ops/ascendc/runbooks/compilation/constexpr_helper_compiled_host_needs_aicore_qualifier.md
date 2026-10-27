---
type: Runbook
title: "An unqualified helper in a shared host/device header is compiled `__host__` under bisheng — qualify it `__aicore__ inline`; fires both for a `constexpr` tiling helper (non-constexpr on device) and for a plain `static inline` one (`call to __host__ function from __aicore__ function`)"
description: "An unqualified helper in a shared host/device header is compiled `__host__` under bisheng — qualify it `__aicore__ inline`; fires both for a `constexpr` tiling helper (non-constexpr on device) and for a plain `static inline` one (`call to __host__ function from __aicore__ function`)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["EC-173"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-173 — EC-173: An unqualified `constexpr` helper is compiled `__host__`, making a `constexpr` device cube-tiling value non-cons

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582；CANN 9.1.0；bisheng=clang15.0.5；op_class=cube-matmul；arch=arch35。
- 验证来源：soc=Ascend950DT_9582；CANN 9.1.0；bisheng clang15.0.5（3-1 BatchMatmul S7 A3→A5，2026-09-13；加 `__aicore__ inline` 后 BUILD_PASS，随后 P51 51/51）。

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; bisheng=clang15.0.5; op_class=cube-matmul; arch=arch35`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0; bisheng=clang15.0.5`
`unverified_on: soc=Ascend910_V220 (A3 chip family — A5 evidence does not transfer automatically)`

- **Symptom**: `kernels_preprocess` Error 2 — a static `MatmulApiStaticTiling` / `MM_CFG` value built inside an `__aicore__` context is rejected as non-constexpr on device, even though the helper that computes it is itself declared `constexpr`.
- **Mechanism**: bisheng compiles a bare `constexpr` free/template helper (e.g. `constexpr int32_t bmm_base_k()`) as `__host__`. When it is invoked from an `__aicore__` builder (`make_static_cfg` is `__aicore__`), the device translation unit has no constexpr definition in scope, so the tiling value it feeds is non-constexpr on device and the static config is rejected.
- **Fix**: qualify the helper for device — `__aicore__ inline constexpr int32_t bmm_base_k() {...}`. General rule: any `constexpr` helper that feeds an `__aicore__` static tiling / `MM_CFG` must carry `__aicore__` (with `inline`); a bare `constexpr` is host-only under bisheng.
- **Detection**: a `constexpr` free/template helper called from `make_static_cfg` / static-tiling construction in a hand-written cube kernel, without `__aicore__`.
- **Evidence**: 3-1 BatchMatmul S7 A3→A5 (2026-09-13, Ascend950DT_9582 / CANN 9.1.0 / bisheng clang15.0.5): build attempts 1–2 FAILED with the `bmm_base_k` template unqualified; adding `__aicore__ inline` → BUILD_PASS (candidate DSO `7f2026a4`), then canonical P51 51/51. Anchor: the `bmm_base_k` template in the 3-1 S7 archive kernel 内部路径（不公开）batchmatmul_kernel.h:79 (evidence archive, external to this repo). Single-instance, symptom-scoped (EC).
- **Wider trigger than `constexpr` (corroboration, 2026-09-26)**: the same root — an unqualified helper
  in a shared host/device header is compiled `__host__` under bisheng — also fires for a **plain,
  non-`constexpr`** `static inline` helper, and then presents with a *different* error text:
  `no matching function for call to 'X'` followed by
  `note: candidate function not viable: call to __host__ function from __aicore__ function`.
  So the trigger is not limited to `constexpr` helpers nor to static-tiling / `MM_CFG` construction:
  **any** helper defined unqualified in a header shared between host and device is unusable from an
  `__aicore__` context. Fix is the same shape — either qualify it `__aicore__ inline`, or keep the
  shared header declarative and define a device-local `__aicore__ inline` helper beside the kernel.
- **Evidence (second instance)**: 3-3 `MatmulBothTrans` A3→A5 `port_a3` (2026-09-26,
  `Ascend950DT_9582` / CANN 9.1.0, run `3a2afc29-151c-4b91-90df-5f3ce7313dc3`): a plain
  `static inline` min-helper in the shared tiling header was rejected at two kernel call sites with
  the `__host__`-from-`__aicore__` note above; replaced with a device-local `__aicore__ inline`
  helper, build then reached `BUILD OK` and the candidate measured precision 50/50 on a frozen
  50-case denominator. Compile-iteration 1 of 3 for that run. This instance is **not** constexpr-
  related, which is what widens the card.
- **Same-family sibling**: **EC-209** (`lambdas_host_only_under_bisheng_aicore`) — lambdas are host-only
  under bisheng and must be hoisted to a named `__aicore__ inline` member. Same underlying rule as this
  card at one more construct: **anything defined without an explicit `__aicore__` in a context bisheng
  compiles for host is unusable from device.** Between them they cover helpers (this card) and lambdas
  (EC-209); if you arrive with a host-only-construct error, check both.
- **Cross-ref**: EC-39 (Cube `MatmulImpl` `CFG_NORM` rejection — a different Cube static-config compile symptom); OL-485/OL-508 (the dbL0 sentinel + on-host library cube recipe this tiling helper participates in). backend=ascendc.
