---
type: Runbook
title: "Declare partition, ring-buffer, workspace, and sync contracts before AscendC kernel implementation"
description: "Declare partition, ring-buffer, workspace, and sync contracts before AscendC kernel implementation"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-FA-CV-2"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-FA-CV-2 — Declare partition, ring-buffer, workspace, and sync contracts before AscendC kernel implementation

## 适用场景

- 适用平台与范围：workflow=L4_fused_op_design；backend=ascendc——新 L4 融合算子在写 kernel 前的算法设计阶段。
- 验证来源：卡内无 verified_on 行；证据见正文 Evidence 段（FA 参考设计的 block/ring/workspace/flag 先行决策暴露了 one-step 生成漏掉的漂移）。

## 候选（未验证 —— 默认检索不返回，需 --status all 才可见）

`applies_to: workflow=L4_fused_op_design; backend=ascendc`
`derived-from: FA-class template-assembly and WorkspaceQueue evidence`

**Trigger**: a new L4 fused op needs algorithm design before AscendC coding. The worker
must decide block partitioning, tile sizes, ring-buffer depth, workspace tensors, and
cross-core synchronization before it starts emitting kernel bodies.

**Pattern**:

1. Record `block_num`, `block_M/N`, prelaunch/ring slots, workspace tensor
   names/shapes/dtypes, cube/vector ownership, and flag IDs in a structured
   decision manifest or Phase-A artifact.
2. Fill per-tile Load/Gemm/Softmax stages only after that contract is reviewable.
3. Mechanically verify that emitted tiling and workspace layout match the contract.

**Detection**: reject an L4 cube↔vec emission that lacks decisions for
`block_partition`, `cube_vec_split`, `cross_core_sync`, `ub_tiling`, and
`ring_slots`. Do not infer them from a monolithic header after generation.

**Evidence**: the FA reference design used block_M=64, block_N=64, prelaunch=2,
ring_slots=3, four workspace tensors, and explicit C/V ownership. Making those
decisions first exposed drift that a one-step model-to-code path had missed.

**Promote when**: a second L4 fused operator independently confirms the same contract
keys predict a correct multi-core AscendC assembly.
