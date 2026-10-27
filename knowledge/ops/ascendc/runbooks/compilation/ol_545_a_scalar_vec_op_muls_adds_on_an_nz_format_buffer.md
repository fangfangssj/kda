---
type: Runbook
title: "A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the FULL NZ-padded extent (roun"
description: "喂给 cube 的 NZ 格式 buffer 上的标量 VEC 算（Muls/Adds）必须覆盖整个 NZ 填充区（roundM×roundSkv），否则尾部未缩放行混入输出"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-545"]
---

# A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the FULL NZ-padded extent (roun

**OL-545** — **A scalar VEC op (Muls/Adds) on an NZ-format buffer that feeds a cube must span the FULL NZ-padded extent (roundM × roundSkv), NOT the logical ND extent (actualM × roundSkv).** When actualM < roundM (tail block, Sq<16) the cube still consumes roundM NZ rows; unscaled rows `actualM..roundM` get multiplied into the output → silent precision fail. NZ-layout analog of OL-221's ND-alignment rule ("cover what the next stage physically reads"). Evidence: prompt_flash_attention iter 1 2026-08-11 dav-m200 CANN 9.0.0 (large(>1) cross-attn fail → fix → 27/27 PASS). Cross-ref OL-221, [`OL-544`](../performance/ol_544_fp16_nz_format_cube_matmul_is_usable_on_ascend31.md), OL-224. backend=ascendc.

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
- **Cross-ref**: OL-221 (ND VEC-op count alignment — the orthogonal layout case: AlignUp for issue, restrict GM to actual count), [`OL-544`](../performance/ol_544_fp16_nz_format_cube_matmul_is_usable_on_ascend31.md) (the FP16 NZ Cube attention composition this bug was found in), OL-224 (tail-K `Mmad` k-arg must be the real K not `kAlign` — sibling "next stage reads the padded extent" trap on the cube side). backend=ascendc.

**孪生卡 ol-649 合并的独有内容**（ol-649-a-scalar-vec-op-on-an-nz-format-buffer-that-feed.md 已删除）：

> // NZ score block is [roundM, roundSkv]; scale ALL of it before BMM2 reads it.

> Muls(scoreUb, scoreUb, scale, roundM * roundSkv); // NOT actualM * roundSkv

## 适用场景

- **工程形态**：喂给 cube 的 NZ 格式 buffer 上的标量 VEC 算（Muls/Adds）必须覆盖整个 NZ 填充区（roundM×roundSkv）。
- **验证过的平台**：310p（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
