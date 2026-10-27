---
type: Runbook
title: "dav-m200 has no `Subs` scalar-subtraction primitive — substitute `Adds(dst, src, -scalar, count)`"
description: "dav-m200 has no `Subs` scalar-subtraction primitive — substitute `Adds(dst, src, -scalar, count)`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["error: no template named 'Subs' in namespace 'AscendC'"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-194 — EC-194: A5Ops additive extension

> 关联：同一 EC-194 主题已由既有卡 [ec_194_dav_m200_ascend310p3_arch_200x_the_scalar_subtra.md](ec_194_dav_m200_ascend310p3_arch_200x_the_scalar_subtra.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，arch 200x）；CANN 9.0.0；op_class=all（任何需要逐元素减标量的 dav-m200 kernel；A5/220x 上 `Subs` 存在，勿外推缺失结论）。
- 验证来源：soc=Ascend310P3（dav-m200）；CANN 9.0.0（layer_norm_v4 310p iter-1 build，2026-08-09）。

**EC-194** — dav-m200 (Ascend310P3, arch 200x): the scalar-subtraction VEC primitive `Subs` is absent (`error: no template named 'Subs' in namespace 'AscendC'`); compute scalar subtraction via `Adds(dst, src, -scalar, count)` — the universal `a + (−s)` identity, a portable fallback whenever ANY scalar-subtract primitive is unavailable or dtype-restricted (tensor-tensor `Sub` IS present on dav-m200). Same arch-surface-reduction class as EC-88 (`Cast<float,int8_t>` no arch-220x overload). COMPILE claim (build-host), device provenance UNVERIFIED — does not block it. Anchor: 内部路径（不公开）workspace/layer_norm_v4/kernel/layer_norm_v4_kernel.h:184-185 (mean-centering step), 9/9 T1. Cross-ref [PB-23](pb_23_simd_binary_scalar_vec_ops_divs_muls_adds_subs_sub.md) (Subs dtype-restriction on arches where it EXISTS), EC-88. backend=ascendc.

- Former A5Ops ID: `EC-194`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=all (any dav-m200 kernel needing per-element subtract-a-scalar)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 (layer_norm_v4 310p iter-1 build, 2026-08-09)`
`unverified_on: soc=Ascend950PR (A5, arch 351x), Ascend910_220x (A3) — on those arches `Subs` EXISTS (PB-23 catalogs its dtype-restricted support set); the template-absence is dav-m200/arch-200-observed. Do not assume `Subs` is absent on A5/A3.`

- **Category**: build / compile
- **Loaded by**: Generator/Worker when a dav-m200 kernel needs to subtract a broadcast scalar from every element of a vector (the `Subs(dst, src, scalar, count)` shape).
- **Symptom**: `AscendC::Subs<float>(dst, src, scalar, count)` fails to compile on dav-m200 with `error: no template named 'Subs' in namespace 'AscendC'` (the scalar-broadcast subtraction template is not in the dav-m200 primitive surface).
- **Mechanism**: dav-m200 (arch 200x) ships a reduced VEC primitive set vs A5/220x; `Subs` (scalar-broadcast subtraction) is among the absent templates. This is the same surface-reduction class as EC-88 (`Cast<float,int8_t>` no arch-220x overload) — a VEC primitive present on newer arches but missing on the older one.
- **Fix**: substitute the algebraic identity `a − s ≡ a + (−s)` — use `Adds(dst, src, -scalar, count)`. `Adds` IS present on dav-m200; negating the scalar is a host-side / scalar-pipe op. This identity is universal (holds on every arch), so `Adds(dst, src, -s)` is a portable fallback whenever ANY scalar-subtraction primitive is unavailable or dtype-restricted. For tensor-tensor subtraction, use `Sub(dst, src1, src2, count)` (which IS present) after broadcasting the scalar into a buffer if needed.
- **Correct pattern** (anchor, 内部路径（不公开）workspace/layer_norm_v4/kernel/layer_norm_v4_kernel.h:184-185 — centering `x` by the row mean, the classic norm step):
  ```cpp
  // x_centered = x - mean  (Subs not on dav-m200 -> Adds with -mean)
  Adds<float>(xLocal, xLocal, -meanScalar, alignedLen);   // a + (-mean), not Subs(a, mean)
  ```
- **Evidence**: layer_norm_v4 310p port (2026-08-09, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5) — iter-1 build rejected `Subs<float>`; the `Adds(dst, src, -meanScalar, …)` substitution is in the shipped kernel, which builds clean and reports 9/9 T1 PASS (CPU-truth, A3 ref unreachable — see OL-25 dav-m200 provenance note). Recorded in `workspace/layer_norm_v4/failures_ledger.md` iter 1. Device provenance UNVERIFIED (`hardware/target/ascend310p.md` `device.runtime: NOT_RUN`) — but this is a COMPILE claim (build-host phenomenon), confirmed by the produced `.so`, NOT a device-runtime claim, so the provenance flag does not block it.
- **Cross-ref**: PB-23 (catalogs `Subs`/`Adds`/… dtype-restriction on arches where they EXIST — A5/220x; the absent-on-dav-m200 case here is a strictly-older-arch surface gap, not a dtype-pair rejection), the VEC-op naming convention (`s` suffix = scalar broadcast: `Adds/Add`, `Subs/Sub`; `Subs` absent on dav-m200 ⇒ fall back to `Adds` + negation), EC-88 (companion arch-220x-overload-absence class). backend=ascendc.
