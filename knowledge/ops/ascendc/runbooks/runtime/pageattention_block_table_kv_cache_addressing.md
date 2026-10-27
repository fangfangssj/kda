---
type: Runbook
title: "PageAttention block-table KV cache addressing — physical-block indirection for paged LLM inference"
description: "PageAttention block-table KV cache addressing — physical-block indirection for paged LLM inference"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-FA-CV-14"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-FA-CV-14 — PageAttention block-table KV cache addressing — physical-block indirection for paged LLM inference

## 适用场景

- 适用平台与范围：卡内无 applies_to 行；适用条件见正文（vLLM 式 paged KV cache 的 LLM 推理；block_table 逻辑→物理块间接寻址）。
- 验证来源：cv-agent sparse_flash_attn_mask_pa kernel（block_table indirection + sparse top-k indices + GQA）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent sparse_flash_attn_mask_pa kernel (block_table indirection + sparse top-k indices + GQA)`

**Pattern**: For LLM inference with paged KV cache (vLLM-style PageAttention), use a block table for logical→physical address translation: `physical_block = block_table[batch, token_idx // block_size]`; `block_offset = token_idx % block_size`. The kernel reads KV from physically non-contiguous blocks, assembling logical KV sequences on-the-fly in L1.

**Addressing mechanism**:
1. Input: `block_table[batch, max_blocks]` — maps logical block index → physical block index
2. For token at position `s`, compute `block_idx = s // block_size`, `offset = s % block_size`
3. Read KV from `kv_cache[block_table[batch, block_idx], offset, :]` — physical address via indirection
4. Assemble contiguous logical KV sequence in L1 buffer from scattered physical reads

**Why paging matters**: Without PageAttention, KV cache must be pre-allocated as max_seq_len × num_layers × 2 × d_model elements of contiguous memory — most of which is wasted for short sequences. Block-table paging allows physical KV cache blocks to be allocated on-demand, reducing memory by 3-20× for typical batch mixtures.

**Additional structural features (this specific variant)**:
- **Sparse top-k indices**: `indices[S, G, topk]` — attend only to top-k relevant tokens per query, not all KV
- **Split-dim**: Q/KV dim split into `dim + tail_dim` where tail_dim is handled differently
- **GQA**: `heads_per_group = heads // kv_group` — multiple Q heads share KV

**L1 challenge**: KV tokens are scattered across physical blocks → DataCopy must do N individual GM reads (one per logical token) rather than one contiguous transfer. Mitigation: if block_size is large enough, intra-block reads are contiguous; only cross-block transitions pay the scatter cost.

**Detection**: grep for `block_table\|page_table\|block_idx\|physical_block` in kernel .h/.cpp. Absence → contiguous KV cache (no paging). Presence → PageAttention pattern.

**Evidence**: cv-agent `sparse_flash_attn_mask_pa/model.py:6-11` — `_logical_pa_token()` function with `block_idx = token_idx // block_size` + `block_table[batch_idx, block_idx]` + `kv[physical_block, block_offset, 0]` — canonical PageAttention indirection.

**Cross-ref**: CAND-FA-CV-6 (MLA split-key architecture — same KV cache compression family, different mechanism: nope+rope decomposition vs block-table paging), CAND-FA-CV-1 (ring buffer with WorkspaceQueue — applies when KV iteration loop exists; PageAttention may use ring buffer for async block prefetch), CAND-NSA-1 (Matmul<>::IterateAll — matmul lib may accelerate the Q×K computation inside PageAttention)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-14，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
