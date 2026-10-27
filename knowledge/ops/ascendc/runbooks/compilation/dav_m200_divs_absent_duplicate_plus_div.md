---
type: Runbook
title: "dav-m200 arch impl has no `Divs` (tensor÷scalar) — substitute `Duplicate(const)` + tensor `Div` for bit-faithful division"
description: "dav-m200 arch impl has no `Divs` (tensor÷scalar) — substitute `Duplicate(const)` + tensor `Div` for bit-faithful division"
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
# EC-208 — EC-208: A5Ops additive extension

> 关联：同一 EC-208 主题已由既有卡 [ec_208_dav_m200_arch_impl_has_no_divs_tensor_scalar.md](ec_208_dav_m200_arch_impl_has_no_divs_tensor_scalar.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200，__CCE_AICORE__=200）；CANN 9.0.0；op_class=all。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（17_AdamW 310p build iter-1，2026-08-15，6 处替换后 fp32 路径 bit-exact）。

**EC-208** — **dav-m200 arch impl has NO `Divs` (tensor÷scalar)** (build error; Muls/Adds present in the same impl header): substitute `Duplicate(const)` + tensor `Div` — TRUE IEEE division, bit-faithful to torch `/ scalar`; do NOT reach for reciprocal+`Muls` on bit-exact paths (OL-256 is the perf-tolerant option on Divs-bearing arches — different scope). Detection: grep `Divs(` before first dav-m200 build. 17_AdamW 310p 2026-08-15 (6 sites, fp32 bit-exact); prior art: celu 310p (reciprocal, concedes 0–1 ULP), op#17 AdamW 220x 2026-05-02. Cross-ref [OL-127](ol_127_cann_api_surface_shrinks_across_versions_check_api.md), [OL-256](../../optimizations/ol_256_divs_to_muls_fp32_precision_safe.md), [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md), [EC-197](cann_9_0_0_dav_m200_mask_form_vec_signatures.md). backend=ascendc.

- Former A5Ops ID: `EC-208`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200); cann=9.0.0; bisheng=n/a; op_class=all`
`unverified_on: soc=Ascend950PR (a5 — Divs exists there, OL-80 catalog); soc=Ascend910_220x (Divs also absent per OL-127's checklist — different workaround era)`

- **Error**: build-time — `Divs` has zero instantiations in the dav-m200 arch impl (CANN 安装树内 asc/impl/basic_api/dav_m200/kernel_operator_vec_binary_scalar_impl.h; `Muls`/`Adds` present, `Divs` absent), so any `Divs(dst, src, scalar, n)` fails to compile.
- **Fix**: broadcast the divisor into a UB tensor once, then tensor `Div`:
  ```cpp
  Duplicate(b2, dM1_, n);   // divisor constant → UB tensor
  Div(t1, t3, b2, n);       // TRUE IEEE division, bit-faithful to torch `/ scalar`
  ```
  This preserves per-element IEEE division — bit-faithful to torch's tensor÷scalar — and is the preferred substitute whenever bit-exactness is the target (ALWAYS_LOADED iron law §5: no reciprocal strength-reduction). Anchor: 内部路径（不公开）workspace/17_AdamW/kernel/adamw_kernel.h:207-233 (6 sites).
- **Detection**: grep the kernel for `Divs(` before first build on a dav-m200 target; every hit is this EC.
- **Evidence**: 17_AdamW 310p build iter-1 (2026-08-15): 6× Divs → Duplicate+Div; fp32 path bit-exact vs the CPU chain (内部路径（不公开）workspace/17_AdamW/failures_ledger.md:9). Prior art, same arch-gap class: celu 310p (reciprocal workaround — concedes 0–1 ULP; superseded by this fix pattern where bit-exactness matters); op#17 AdamW 220x port 2026-05-02 (OL-127's Divs row — reciprocal-in-pybind+`Muls` era).
- **Tension note (deliberate, not a conflict)**: OL-256 recommends `Divs→Muls(1/scalar)` as a PRECISION-TOLERANT fp32 perf optimization on arches where Divs exists. That rule targets throughput under tolerance; THIS EC targets the Divs-absent BUILD ERROR on dav-m200 with bit-exactness preferred. Do not cite OL-256 to justify reciprocal-multiply on a bit-exact dav-m200 path.
- **Other instances (predicted)**: any dav-m200 kernel with tensor÷scalar sites — optimizer bias-correction divisions, normalization eps-divisions, LR-scaled updates.
- Cross-ref [OL-127](ol_127_cann_api_surface_shrinks_across_versions_check_api.md) (API surface shrinks across versions — the 220x Divs row this extends to arch-200x with a better workaround), OL-256 (reciprocal perf option, different scope), OL-80 (API-catalog lookup discipline), EC-197 (dav-m200 signature-surface gap class). backend=ascendc.
