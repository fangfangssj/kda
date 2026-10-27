---
type: Runbook
title: "CANN 9.0.0 (dav-m200) repeat-form VEC ops take a mask, not a count — verify every VEC/fixpipe call against the installed toolkit headers"
description: "CANN 9.0.0 (dav-m200) repeat-form VEC ops take a mask, not a count — verify every VEC/fixpipe call against the installed toolkit headers"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["BinaryRepeatParams", "SetFixpipeNz2ndFlag", "SetLoadDataPaddingValue", "SetNdpara"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-197 — EC-197: A5Ops additive extension

> 关联：同一 EC-197 主题已由既有卡 [ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md](ec_197_cann_9_0_0_dav_m200_vec_repeat_form_signatures_d.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=all（任何参考更新版本 CANN 源码 kernel 调用形式的移植）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（签名对本机 9.0.0 安装头文件核验；quant_batch_matmul_v3 310p iter 1 编译观测，2026-08-10；add_layer_norm_grad 第二实例见正文）。

**EC-197** — CANN 9.0.0 (dav-m200) VEC/repeat-form signatures DIFFER from newer-CANN source-kernel forms: repeat-form binary ops take a **MASK** not a count (`Add/Mul<T,isSetMask=true>(dst,src0,src1,uint64 mask,uint8 repeatTime,BinaryRepeatParams)`, verified CANN 安装树内 asc/include/interface/kernel_operator_vec_binary_intf.h:57-77; count-form `Add(dst,src0,src1,count)` + count-form `Cast(dst,src,RoundMode,count)` also exist); copy-paste from CANN ops-nn source (newer CANN) fails to compile with "no matching function". Plus: `TBuf` must be `.Get<T>()`-materialized before VEC/DataCopy use; fixpipe config is `SetLoadDataPaddingValue`/`SetFixpipeNz2ndFlag`, NOT the utils' `SetPadding`/`SetNdpara` aliases. Fix: verify every VEC/fixpipe call against the INSTALLED toolkit headers; prefer count-form (also the dav-m200-correct choice). quant_batch_matmul_v3 310p 2026-08-10 iter-1 all 4 surfaces, rewritten → 9/9 T1. Cross-ref [OL-80](../precision/ol_80_api_existence_must_be_looked_up_in_the_catalog_nev.md), [EC-191](duplicate_int8_no_dav_m200_overload.md)/91/93, CAND-DAV-M200-MASKFORM-VEC-UNRELIABLE. backend=ascendc.

- Former A5Ops ID: `EC-197`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any port reading newer-CANN source kernels for call-form reference)`
`verified_on: soc=Ascend310P3; cann=9.0.0 — signatures verified against the local 9.0.0 install headers (paths cited below); compile behavior observed in quant_batch_matmul_v3 310p iter 1`
`unverified_on: soc=Ascend950PR (arch 351x ships a NEWER CANN with the count-form source-kernel signatures — the 9.0.0 mask-form set below is the OLD surface); other 220x-family SoCs (same CANN 9.0.0 toolkit → expect the same mask-form surface, but only dav-m200 is measured)`

- **Category**: compile / authoring (API-surface, CANN-version drift)
- **Loaded by**: Generator/Worker during any port_a3 port whose reference kernels (CANN ops-nn source) use newer-CANN call forms than the installed toolkit — reading the reference for call-form is safe, copy-pasting its VEC/fixpipe calls verbatim is NOT.
- **Error pattern**: porting a kernel from CANN ops-nn source (newer CANN) to a CANN 9.0.0 install fails at compile with "no matching function" — the SOURCE kernels use `(dst, src0, src1, count, repeatTime, ...)`-style forms that do not exist in 9.0.0. The failure reads as a generic template-resolution error, not as a version mismatch.
- **Root cause**: the repeat-form binary VEC ops on CANN 9.0.0 take a **MASK**, not a count. Verified in the 9.0.0 install (CANN 安装树内 asc/include/interface/kernel_operator_vec_binary_intf.h:57-77, CANN 安装树内 asc/include/interface/kernel_operator_vec_binary_scalar_intf.h:52-63):
  ```cpp
  // CANN 9.0.0 (dav-m200) — repeat-form takes a uint64 mask + repeatTime + repeat params:
  Add<T, isSetMask=true>(dst, src0, src1, uint64_t mask, uint8_t repeatTime, BinaryRepeatParams);
  Muls<T, U, isSetMask=true>(dst, src, scalar, uint64_t mask, uint8_t repeatTime, UnaryRepeatParams);
  // count-form ALSO exists in 9.0.0 and is the portable default:
  Add<T>(dst, src0, src1, int32_t count);
  Cast<T, U>(dst, src, RoundMode, count);   // count-form Cast verified at CANN 安装树内 asc/include/interface/kernel_operator_vec_vconv_intf.h:73
  ```
  Additional 9.0.0-specific surfaces hit in the same iter: (a) `TBuf` must be materialized via `.Get<T>()` before VEC/DataCopy use; (b) fixpipe config APIs are `SetLoadDataPaddingValue` / `SetFixpipeNz2ndFlag` (CANN 安装树内 asc/include/basic_api/kernel_operator_mm_intf.h:353-361, CANN 安装树内 asc/include/basic_api/kernel_operator_fixpipe_intf.h:59), NOT the utils' `SetPadding`/`SetNdpara` aliases.
- **Fix**: verify every VEC/fixpipe call signature against the INSTALLED toolkit's headers (`tikcpp/tikcfw/interface/` equivalents under the CANN install `asc/include/interface/` and `asc/include/basic_api/`) before writing the call — do not trust the reference source kernel's call form when the reference targets a newer CANN. Prefer count-form ops when the mask form is not semantically needed (count-form is also the dav-m200-correct choice per CAND-DAV-M200-MASKFORM-VEC-UNRELIABLE).
- **Detection**: `grep` the kernel TU for repeat-form calls (`repeatTime`, `BinaryRepeatParams`, `UnaryRepeatParams`, `isSetMask`) and for fixpipe utils aliases (`SetPadding`, `SetNdpara`) — any hit that came from a CANN-source kernel needs re-verification against 9.0.0 headers.
- **Evidence**: quant_batch_matmul_v3 310p port (2026-08-10, Ascend310P3/dav-m200, CANN 9.0.0, port_a3_to_a5, CUBE_MIX) — iter-1 compile failed on all four surfaces above (内部路径（不公开）workspace/quant_batch_matmul_v3/failures_ledger.md:4 iter 1, logged `novel`); rewritten to the 9.0.0 forms (count-form Casts, mask-form replaced by count-form per CAND-DAV-M200-MASKFORM-VEC-UNRELIABLE, `.Get<T>()` TBuf materialization, 9.0.0 fixpipe config names) → build clean, 9/9 T1 PASS. COMPILE claim — device provenance `NOT_RUN` does not block it.
- **Evidence (2nd instance — absent OVERLOAD, same signature-drift class, authored call)**: add_layer_norm_grad cast_probe (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0) — `Adds(half-src, float-scalar, float-dst)` mixed-type overload is ABSENT from the 9.0.0 surface ("no matching function"); dropped the Adds-widen variant and spelled the widen as count-form `Cast` + `Adds` on the widened buffer (`workspace/add_layer_norm_grad/failures_ledger.md`, cast_probe compile row, logged `novel`). Extends the fix rule beyond source-copied calls: the verify-against-9.0.0-headers discipline applies to AUTHORED mixed-type calls too — a newer-CANN overload's existence is not a guarantee of its presence in 9.0.0.
- **Other instances (predicted)**: any port whose CANN reference sources were written for CANN ≥ 9.1/newer ops-nn — the ops-nn source tree moves ahead of installed toolkits; quant-matmul / dequant / norm-quant families (repeat-form-heavy) are the likely repeat victims.
- **Cross-ref**: OL-80 (API existence must be looked up in the catalog — this is the signature-level sibling: existence is not enough, the VERSION's signature must be verified); EC-191/91/93 (dav-m200 API-surface family); CAND-DAV-M200-MASKFORM-VEC-UNRELIABLE (count-form preference on dav-m200 — the semantic reason count-form is also the better choice). backend=ascendc.

---
