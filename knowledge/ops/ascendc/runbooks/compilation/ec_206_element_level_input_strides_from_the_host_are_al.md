---
type: Runbook
title: "element-level input strides from the host are ALREADY in element units (each = product of lower input dims); a"
description: "host 传来的元素级 strides 已是元素单位，再乘一次 inner_dim 会双重缩放导致 GM 越界崩溃；1D 冒烟测不出（恒短路 offset 0），首个多维 case 才爆。"
tags: [task.debugging, topic.data_movement.data_copy, topic.implementation.tiling, topic.compute.elementwise.type_conversion, topic.compilation, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-206"]
---

# element-level input strides from the host are ALREADY in element units (each = product of lower input dims); a

**EC-206** — element-level input strides from the host are ALREADY in element units (each = product of lower input dims); a repeat/tiling/broadcast source-offset kernel that re-multiplies the stride-accumulated offset by `inner_dim` double-scales → GM out-of-range crash on every 2D+ case (1D short-circuits to offset 0, so the bug hides until the first multi-dim case — easy to miss on a 1D smoke test). Fix: source row offset = `Σ (out_coord[d] % in_dims[d]) * in_strides[d]` over higher dims only, no inner-extent factor. Anchor `代码片段 16_Repeat/repeat_kernel.h.snippet.h（见文末）` (`ComputeSrcRowOffset`). 16_Repeat 310p 2026-08-14 (49/49 + 159/159 bit-exact after dropping the factor). Arch-independent logic bug; EC-195 sibling. backend=ascendc.

`applies_to: soc=all (logic bug, observed on Ascend310P3/dav-m200/arch-200); cann=9.0.0; bisheng=n/a (arch-independent indexing arithmetic); op_class=any op that (a) receives host-precomputed element-level strides AND (b) computes a source offset that also involves an inner-dim extent — repeat/tiling/broadcast, strided gather with stride×count`
`applies_to_backend: ascendc`

- **Category**: logic / indexing (arch-independent author bug)
- **Loaded by**: Generator/Worker authoring an N-D repeat/tiling/broadcast or any stride×count source-offset kernel
- **Error pattern**: a multi-dim repeat/tiling kernel computes the source GM offset for an output row by decomposing the output row index into per-dim coordinates and accumulating `Σ in_coord[d] * stride[d]` — but then ADDITIONALLY multiplies the accumulated offset (or its inner term) by the innermost input dimension (`inner_dim`), as if the strides were not yet element-level. The result is a source offset scaled ~`inner_dim×` too large → GM read past the input buffer → runtime out-of-range crash (on 16_Repeat: GM OOB on every 2D+ case). 1D cases (`ndim_higher==0`) short-circuit to offset 0 and are unaffected, so the bug HIDES until the first multi-dim case — easy to miss on a 1D smoke test.
- **Root cause**: the host already passes `in_strides[d]` as ELEMENT-level strides (each = product of the lower input dims, so `in_strides[d]` already accounts for `inner_dim` and everything below it). The kernel's flat source offset is the dot product `Σ in_coord[d] * in_strides[d]` over the higher dims — NOTHING else. Multiplying that by `inner_dim` re-counts the inner dimension.
- **Fix**: drop the extra `inner_dim` factor. Compute the source row offset as `Σ (out_coord[d] % in_dims[d]) * in_strides[d]` over the higher dims only; the innermost dim's contribution is `+0` (contiguous) and is already encoded in the element-level strides. Anchor: `代码片段 16_Repeat/repeat_kernel.h.snippet.h（见文末）` (`ComputeSrcRowOffset` — `src += in_coord * in_strides_[idx]`, no extra `inner_dim` factor).
- **Detection**: a repeat/tiling/broadcast op that passes a 1D smoke test but crashes with GM out-of-range on the first 2D+ case — grep the source-offset computation for any multiplication of an already-strided accumulator by an inner-dim extent.
- **Evidence**: 16_Repeat 310p (2026-08-14, Ascend310P3/dav-m200/arch-200, CANN 9.0.0, benchmark — `torch.Tensor.repeat`): the iter path's `ComputeSrcRowOffset` multiplied the stride-accumulated `src` by `inner_dim_`, crashing (GM OOB) on all 2D+ cases while 1D returned early; removing the factor → 49/49 benchmark + 159/159 edge bit-exact (fp32 20/20, fp16 19/19, bf16 10/10), det 49/49. Arch-independent logic bug — device provenance `NOT_RUN` does not block the diagnosis. Cross-ref [`EC-195`](ec_195_flat_index_multi_dim_divmod_chain_each_level_s_r.md) (sibling multi-dim index-decomposition logic bug — the flat-index divmod family). backend=ascendc.
- **Other instances (predicted)**: tile/broadcast/expand ops, strided gather where a stride is multiplied by a repeat/count, any host-stride consumer that re-derives an inner term the strides already encode.

## 内嵌代码片段

### 16_Repeat/repeat_kernel.h.snippet.h

```cpp
        int64_t rem = output_row;
        for (int32_t k = 0; k < tiling_.ndim_higher; k++) {
            int32_t idx = tiling_.ndim_higher - 1 - k;
            int64_t out_dim = tiling_.out_dims_higher[idx];
            int64_t in_dim = tiling_.in_dims_higher[idx];
            int64_t out_coord = rem % out_dim;
            rem /= out_dim;
            int64_t in_coord = out_coord % in_dim;
            src += in_coord * tiling_.in_strides_higher[idx];
        }
        return src;
    }

    __aicore__ inline void CopyTile(
        int32_t count, int64_t in_row_elem_offset, int32_t src_col, int64_t out_pos)
    {
        int32_t inner = static_cast<int32_t>(tiling_.inner_dim);
        int32_t batch_elements = inner;
        while (batch_elements % ALIGN != 0) batch_elements += inner;

        int32_t done = 0;
        int32_t cur_col = src_col;

        while (done < count) {
            int32_t remaining = count - done;
            int32_t batch_count = batch_elements;
            if (batch_count > remaining) {
                batch_count = (remaining / ALIGN) * ALIGN;
                if (batch_count == 0) break;
            }

            int32_t batch_done = 0;
            while (batch_done < batch_count) {
                int32_t chunk = batch_count - batch_done;
                int32_t row_avail = inner - cur_col;
                if (chunk > row_avail) chunk = row_avail;
                int32_t aligned = ((chunk + ALIGN - 1) / ALIGN) * ALIGN;

                LocalTensor<T> in = inQueue_.AllocTensor<T>();
                DataCopy(in, gmIn_[in_row_elem_offset + cur_col], aligned);
                inQueue_.EnQue(in);

                LocalTensor<T> in_dq = inQueue_.DeQue<T>();
                LocalTensor<T> out = outQueue_.AllocTensor<T>();
                DoCopy(out, in_dq, aligned);
                inQueue_.FreeTensor(in_dq);
                outQueue_.EnQue(out);

        // (overwritten by next tile). Overflow past output buffer is absorbed
        // by pybind padding. Inter-block boundaries limited by small nblk.
        while (done < count) {
            int32_t chunk = count - done;
            int32_t row_avail = inner - cur_col;
            if (chunk > row_avail) chunk = row_avail;
            int32_t aligned = ((chunk + ALIGN - 1) / ALIGN) * ALIGN;
            DataCopy(in, gmIn_[in_row_elem_offset + cur_col], aligned);
            inQueue_.EnQue(in);

            LocalTensor<T> in_dq = inQueue_.DeQue<T>();
            LocalTensor<T> out = outQueue_.AllocTensor<T>();
            DoCopy(out, in_dq, aligned);
            inQueue_.FreeTensor(in_dq);
            outQueue_.EnQue(out);

            LocalTensor<T> out_dq = outQueue_.DeQue<T>();
            DataCopy(gmOut_[out_pos + done], out_dq, aligned);
            outQueue_.FreeTensor(out_dq);

            done += chunk;
            cur_col = (cur_col + chunk) % inner;
        }
    }

    __aicore__ inline void DoCopy(LocalTensor<T>& dst, LocalTensor<T>& src, int32_t count)
    {
```
