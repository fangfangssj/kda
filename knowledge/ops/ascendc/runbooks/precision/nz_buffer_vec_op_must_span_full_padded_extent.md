---
type: Runbook
title: "A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the full NZ-padded extent (roundM × roundSkv), not the logical ND extent (actualM × roundSkv)"
description: "A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the full NZ-padded extent (roundM × roundSkv), not the logical ND extent (actualM × roundSkv)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["Stage2ScaleSoftMax"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-545 — OL-545: A5Ops additive extension

> 关联：同一 OL-545 主题已由既有卡 [ol_545_a_scalar_vec_op_muls_adds_on_an_nz_format_buffer.md](../compilation/ol_545_a_scalar_vec_op_muls_adds_on_an_nz_format_buffer.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200) [and any arch using NZ-format cube operands]; cann=9.0.0; bisheng=n/a; op_class=attention-fwd / any cube+vec kernel where a scalar VEC op on an NZ buffer precedes a cube matmul。
- 验证来源：soc=Ascend310P3 (dav-m200); cann=9.0.0 — prompt_flash_attention, single op。

**OL-545** — **A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the FULL NZ-padded extent (roundM × roundSkv), NOT the logical ND extent (actualM × roundSkv).** When actualM < roundM (tail block, Sq<16) the cube still consumes roundM NZ rows; unscaled rows `actualM..roundM` get multiplied into the output → silent precision fail. NZ-layout analog of OL-221's ND-alignment rule ("cover what the next stage physically reads"). Evidence: prompt_flash_attention iter 1 2026-08-11 dav-m200 CANN 9.0.0 (large(>1) cross-attn fail → fix → 27/27 PASS). Cross-ref [OL-221](../../optimizations/ol_221_partial_vector_alignup_restrict_gm_access.md), [OL-544](../performance/ol_544_fp16_nz_format_cube_matmul_is_usable_on_ascend31.md), [OL-224](../../optimizations/ol_224_manual_mmad_real_k_and_transpose_gating.md). backend=ascendc.

- Former A5Ops ID: `OL-545`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200, __CCE_AICORE__=200) [and any arch using NZ-format cube operands]; cann=9.0.0; bisheng=n/a; op_class=attention-fwd / any cube+vec kernel where a scalar VEC op on an NZ buffer precedes a cube matmul`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — prompt_flash_attention, single op`
`unverified_on: expected to transfer to any NZ-cube arch (220x, V351) since it is a layout fact, but cross-verified only on dav-m200`

- **Category**: correctness / precision
- **Loaded by**: Generator/QA when a scalar/mask VEC op (Muls, Adds) touches an NZ-format UB buffer that is then read by a cube matmul
- **Trigger**: a scalar VEC op (e.g. `Muls` scaling softmax scores by an attention-mask / scale) is applied to a UB buffer holding NZ-format data, and that buffer is then fed back into a cube matmul (BMM2: P@V). When the logical M (`actualM`, e.g. a tail query-block `Sq < 16`) is smaller than the NZ-padded M (`roundM = RoundUp(actualM, 16)`), the NZ block physically contains `roundM` rows.
- **Principle**: the count argument to the scalar VEC op must be the FULL NZ-padded element count `roundM × roundSkv`, NOT the logical `actualM × roundSkv`. The cube consumes the whole NZ block (`roundM` rows); if the scalar op only covered `actualM` rows, rows `actualM..roundM` are left unscaled and the cube multiplies them into the output → silent precision failure (no error). Padding elements are zero / don't-care for a pure VEC consumer, but once a cube reads the block they MUST be scaled consistently with the valid rows. This is the NZ-layout analog of OL-221's ND-alignment rule — different layout, same "cover what the next stage physically reads" principle.
- **Concrete anchor** (内部路径（不公开）pfa_kernel.h):
  ```cpp
  // NZ score block is [roundM, roundSkv]; scale ALL of it before BMM2 reads it.
  Muls(scoreUb, scoreUb, scale, roundM * roundSkv);   // NOT actualM * roundSkv
  ```
- **Action**: any scalar/mask/bias VEC op whose NZ-format output feeds the next cube pass must use the padded NZ extent as its count; do likewise for the softmax max/sum reduction over the NZ block.
- **Evidence**: prompt_flash_attention iter 1 (2026-08-11, Ascend310P3/dav-m200, CANN 9.0.0) — large(>1) precision failures in cross-attention cases (`Sq ≠ Skv`, `actualM < 16`); root-caused to `Stage2ScaleSoftMax` passing `actualM*roundSkv` as the Muls count instead of `roundM*roundSkv`. Fix → full NZ coverage → 27/27 PASS. Source candidate: `CAND-NZ-MULS-COUNT`. Failures-ledger: `iter 1 | large(>1) | novel`.
- **Other instances (predicted)**: any cube+vec kernel where a scalar/mask VEC op touches an NZ buffer before the next cube pass — masked-attention scale, dropout-scale, bias-add on NZ score tiles; the softmax max/sum reduction over the NZ block spans `roundM` for the same reason.
- **Cross-ref**: OL-221 (ND VEC-op count alignment — the orthogonal layout case: AlignUp for issue, restrict GM to actual count), OL-544 (the FP16 NZ Cube attention composition this bug was found in), OL-224 (tail-K `Mmad` k-arg must be the real K not `kAlign` — sibling "next stage reads the padded extent" trap on the cube side). backend=ascendc.

---
