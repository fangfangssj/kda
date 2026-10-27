---
type: Runbook
title: "Chunked cat (dim>0, small inner): 2D strided DataCopyParams amortizes per-row MTE2→MTE3 sync"
description: "Chunked cat (dim>0, small inner): 2D strided DataCopyParams amortizes per-row MTE2→MTE3 sync"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-CHUNKED-CAT-2D-STRIDED-DATACOPYPARAMS"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-CHUNKED-CAT-2D-STRIDED-DATACOPYPARAMS — Chunked cat (dim>0, small inner): 2D strided DataCopyParams amortizes per-row MTE2→MTE3 sync

## 适用场景

- 适用平台与范围：soc=Ascend950 (351x)；CANN 9.1.0；op_class=cat/split/chunked-copy 且内维较小；要求所有 chunk 32B 对齐、chunk <= TILE、stride <= 65535 blocks。
- 验证来源：13_Cat M-I run-1 ko-2 Opt3（cases 7/8/9：0.30-0.60x -> >=0.97x）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950 (351x); cann=9.1.0; op_class=cat/split/chunked-copy with small inner dim`
`verified_on: 13_Cat M-I run-1 ko-2 Opt3 (cases 7/8/9: 0.30-0.60x -> >=0.97x)`

2D strided DataCopy with `DataCopyParams{blockCount=K rows, blockLen=chunk_blocks, srcGap=(blk_num-1)*chunk_blocks, dstGap=blk_num*tci-chunk_blocks}`, K=TILE/chunk rows per single flag sync. **Requires**: all chunks 32B-aligned (slot prefix sums must stay aligned), chunk <= TILE, strides <= 65535 blocks (uint16). CANN 9.1.0 header facts: srcGap/dstGap are GAP semantics (srcStride/dstStride deprecated aliases); gap = step_blocks - blockLen; `SetFlag/WaitFlag<HardEvent::MTE2_MTE3|MTE3_MTE2>(int32 id)` with literal small ids 0-3 works on dav_3510 (AIV-only kernel).

- **This answers the M-F-era open question** ('the unaligned-interleave mechanism the reference uses is unknown') for the ALIGNED sub-family; unaligned chunked rows remain open (see CAND-BATCHED-UNALIGNED-STAGING-DEVICE-FAULT-507035).
- **Cross-ref**: CAND-COPY-MODE-SPECIFIC-TILE-BUDGET, CAND-QUEUE-PROTOCOL-IS-THE-MTE2-MTE3-ORDERING. backend=ascendc.
