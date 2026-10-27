---
type: Runbook
title: "Concatenated-pair UB layout for 2-input binary-reduction kernels — single LocalTensor of size `2N`, operand B at offset `N`"
description: "Concatenated-pair UB layout for 2-input binary-reduction kernels — single LocalTensor of size `2N`, operand B at offset `N`"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-RAU-4"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-RAU-4 — Concatenated-pair UB layout for 2-input binary-reduction kernels — single LocalTensor of size `2N`, operand B at offset `N`

## 适用场景

- 适用平台与范围：any SoC with public AscendC TQue/LocalTensor; cann=9.0.0+; op_class=binary_reduction / pairwise_merge / 2_input_pointwise（卡内 applies_to 行）。
- 验证来源：cann ops-transformer/attention/ring_attention_update.h + _tnd.h SoftmaxDataMoveIn/AttnDataMoveIn (source-structure-only; no a5_ops measurement)（卡内 verified_on 行）；未验证范围：a5_ops（卡内 unverified_on 行）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any SoC with public AscendC TQue/LocalTensor; cann=9.0.0+; op_class=binary_reduction / pairwise_merge / 2_input_pointwise`
`derived-from: cann-source (ring-attn-class update SBH + TND, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update.h + _tnd.h SoftmaxDataMoveIn/AttnDataMoveIn (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: Kernel implements a binary reduction `out = f(a, b)` where `a, b` have identical shape, both come from GM, and both are processed in the same Compute block. The naïve form allocates two separate queues / two separate LocalTensors. This pattern packs both operands into ONE LocalTensor of size `2 * N` and indexes operand B via offset `[N]`, halving queue count and easing buffer-budget pressure.

**Pattern**:
1. **Init**: Allocate one queue per pair (not two). Configure each queue's buffer size to `2 * N * sizeof(T)` (twice operand size). `tPipe->InitBuffer(pairQueue, BUFFER_NUM, 2 * N * sizeof(T))`.
2. **DataMoveIn**: AllocTensor returns a LocalTensor of full length `2N`. Issue two DataCopyPad's, first into `tensor` (offset 0), second into `tensor[N]` (offset N elements). Each DataCopyPad uses the SAME DataCopyExtParams (same shape, same stride).
3. **Compute**: Reference operand A as `tensor` and operand B as `tensor[N]` directly in the VEC primitive call. e.g. `Max(out, tensor, tensor[N], mask, repeat, params)`.
4. **Free**: One FreeTensor releases both operands together.

**Concrete anchor** (public-API; worker-local names):
```cpp
// Init phase
tPipe->InitBuffer(pairQueue, BUFFER_NUM, 2 * N * sizeof(float));

// DataMoveIn
auto pair = pairQueue.AllocTensor<float>();        // length 2N
AscendC::DataCopyPad(pair,        srcAGm[offset], copyParams, padParams);
AscendC::DataCopyPad(pair[N],     srcBGm[offset], copyParams, padParams);
pairQueue.EnQue<float>(pair);

// Compute
pair = pairQueue.DeQue<float>();
AscendC::Max(out, pair, pair[N], mask, repeatTimes, repeatPar);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Sub(scratchA, pair,    out, mask, repeatTimes, repeatPar);
AscendC::Sub(scratchB, pair[N], out, mask, repeatTimes, repeatPar);
// ... reuse pair[0..N) and pair[N..2N) for the full merge ...
pairQueue.FreeTensor<float>(pair);
```

**Benefits**:
- **Queue count halved**: for a 2-input op with 3 logical streams (e.g. softmax has prev_max, prev_sum, prev_out and cur_max, cur_sum, cur_out → six logical streams paired as three pairs), this saves three queues. Important when budget is tight (typical Atlas-class budget: 12-16 queues).
- **DMA bandwidth identical**: two DataCopyPad's go out either way; packing into one tensor doesn't merge them.
- **VEC primitive overhead unchanged**: Max/Sub/etc don't care that operands are offset-aliased; they just compute on the addresses they're handed.
- **Cache locality marginally better**: A and B end up in adjacent UB blocks; the VEC unit's load step crosses them in one fetch sometimes.

**Numerics**: No effect on semantics — identical to two-queue form.

**Hard do-not-apply**:
- Do NOT pack when operands have DIFFERENT shape: defeats the offset-indexing assumption.
- Do NOT pack when operand A and operand B have DIFFERENT lifetimes (e.g. A is used early, B is used in a downstream compute): the FreeTensor releases both together. Use separate queues if independent free is needed.
- Do NOT pack when buffer-budget is tight in absolute UB bytes: this form REQUIRES 2*N per queue slot; two separate queues would have N each. Total UB usage same per pair, but allocation is less flexible.
- Do NOT pack more than 2 operands this way without measurement: 3-way packing (`tensor`, `tensor[N]`, `tensor[2*N]`) is legal but harder to reason about; downstream readers struggle.

**Other instances predicted**:
- Any 2-input merge/reduce kernel: `max(a, b)`, `min(a, b)`, pairwise softmax-merge (this op), pairwise tree-reductions.
- Top-k merge kernels combining two sorted-and-padded chunks.
- KV-cache append where `[prev_kv, new_kv]` are concatenated then re-indexed.
- Two-stream fused element-wise: `out = α*a + β*b` with α, β scalar.

**Risks before promotion**:
- Buffer-budget reasoning becomes per-pair (in 2N units) instead of per-stream — analyzers (`probe_report.md` budget sections) must understand the convention. Document in the kernel header that the queue holds 2 operands packed.
- Some VEC primitives have alignment requirements on offsets; `tensor[N]` must satisfy them. For fp32 with 32-byte alignment, `N` must be a multiple of 8 — typically true (compute counts are usually mult of 64 = one repeat). Add a static_assert.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel that measurably benefits from the queue-count reduction (i.e. the un-packed form runs out of buffer slots or has measurably worse pipeline overlap).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-RAU-4，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
