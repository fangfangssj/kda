---
type: Runbook
title: "host-passed input strides are already element-level — re-multiplying by `inner_dim` double-scales the offset into a GM out-of-range crash"
description: "host-passed input strides are already element-level — re-multiplying by `inner_dim` double-scales the offset into a GM out-of-range crash"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["ComputeSrcRowOffset"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# EC-206 — EC-206: A5Ops additive extension

> 关联：同一 EC-206 主题已由既有卡 [ec_206_element_level_input_strides_from_the_host_are_al.md](ec_206_element_level_input_strides_from_the_host_are_al.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all（逻辑 bug，观测于 Ascend310P3/dav-m200/arch-200）；CANN 9.0.0；op_class=任何接收宿主预算 element 级 stride 且源偏移另含 inner-dim 因子的算子（repeat/tiling/broadcast、stride×count 的 strided gather）。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行（16_Repeat 310p，2026-08-14；去掉因子后 49/49 + 159/159 bit-exact）。

**EC-206** — element-level input strides from the host are ALREADY in element units (each = product of lower input dims); a repeat/tiling/broadcast source-offset kernel that re-multiplies the stride-accumulated offset by `inner_dim` double-scales → GM out-of-range crash on every 2D+ case (1D short-circuits to offset 0, so the bug hides until the first multi-dim case — easy to miss on a 1D smoke test). Fix: source row offset = `Σ (out_coord[d] % in_dims[d]) * in_strides[d]` over higher dims only, no inner-extent factor. Anchor 内部路径（不公开）workspace/16_Repeat/kernel/repeat_kernel.h:162-175 (`ComputeSrcRowOffset`). 16_Repeat 310p 2026-08-14 (49/49 + 159/159 bit-exact after dropping the factor). Arch-independent logic bug; EC-195 sibling. backend=ascendc.

- Former A5Ops ID: `EC-206`
- Fusion disposition: `additive_new`

`applies_to: soc=all (logic bug, observed on Ascend310P3/dav-m200/arch-200); cann=9.0.0; bisheng=n/a (arch-independent indexing arithmetic); op_class=any op that (a) receives host-precomputed element-level strides AND (b) computes a source offset that also involves an inner-dim extent — repeat/tiling/broadcast, strided gather with stride×count`
`applies_to_backend: ascendc`

- **Category**: logic / indexing (arch-independent author bug)
- **Loaded by**: Generator/Worker authoring an N-D repeat/tiling/broadcast or any stride×count source-offset kernel
- **Error pattern**: a multi-dim repeat/tiling kernel computes the source GM offset for an output row by decomposing the output row index into per-dim coordinates and accumulating `Σ in_coord[d] * stride[d]` — but then ADDITIONALLY multiplies the accumulated offset (or its inner term) by the innermost input dimension (`inner_dim`), as if the strides were not yet element-level. The result is a source offset scaled ~`inner_dim×` too large → GM read past the input buffer → runtime out-of-range crash (on 16_Repeat: GM OOB on every 2D+ case). 1D cases (`ndim_higher==0`) short-circuit to offset 0 and are unaffected, so the bug HIDES until the first multi-dim case — easy to miss on a 1D smoke test.
- **Root cause**: the host already passes `in_strides[d]` as ELEMENT-level strides (each = product of the lower input dims, so `in_strides[d]` already accounts for `inner_dim` and everything below it). The kernel's flat source offset is the dot product `Σ in_coord[d] * in_strides[d]` over the higher dims — NOTHING else. Multiplying that by `inner_dim` re-counts the inner dimension.
- **Fix**: drop the extra `inner_dim` factor. Compute the source row offset as `Σ (out_coord[d] % in_dims[d]) * in_strides[d]` over the higher dims only; the innermost dim's contribution is `+0` (contiguous) and is already encoded in the element-level strides. Anchor: 内部路径（不公开）workspace/16_Repeat/kernel/repeat_kernel.h:162-175 (`ComputeSrcRowOffset` — `src += in_coord * in_strides_[idx]`, no extra `inner_dim` factor).
- **Detection**: a repeat/tiling/broadcast op that passes a 1D smoke test but crashes with GM out-of-range on the first 2D+ case — grep the source-offset computation for any multiplication of an already-strided accumulator by an inner-dim extent.
- **Evidence**: 16_Repeat 310p (2026-08-14, Ascend310P3/dav-m200/arch-200, CANN 9.0.0, benchmark — `torch.Tensor.repeat`): the iter path's `ComputeSrcRowOffset` multiplied the stride-accumulated `src` by `inner_dim_`, crashing (GM OOB) on all 2D+ cases while 1D returned early; removing the factor → 49/49 benchmark + 159/159 edge bit-exact (fp32 20/20, fp16 19/19, bf16 10/10), det 49/49. Arch-independent logic bug — device provenance `NOT_RUN` does not block the diagnosis. Cross-ref [EC-195](flat_index_divmod_remainder_current_level_stride.md) (sibling multi-dim index-decomposition logic bug — the flat-index divmod family). backend=ascendc.
- **Other instances (predicted)**: tile/broadcast/expand ops, strided gather where a stride is multiplied by a repeat/count, any host-stride consumer that re-derives an inner term the strides already encode.

---