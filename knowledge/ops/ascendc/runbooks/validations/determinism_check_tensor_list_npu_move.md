---
type: Runbook
title: "determinism_check.py shallow NPU move breaks tensor-list ops (cat/stack family)"
description: "determinism_check.py shallow NPU move breaks tensor-list ops (cat/stack family)"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-DETERMINISM-CHECK-TENSOR-LIST-NPU-MOVE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-DETERMINISM-CHECK-TENSOR-LIST-NPU-MOVE — determinism_check.py shallow NPU move breaks tensor-list ops (cat/stack family)

## 适用场景

- 适用平台与范围：op_class=cat/stack 等张量列表输入算子；harness=determinism_check.py。
- 验证来源：13_Cat M-I run-1（运行中已在 harness 侧修复）。

## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: op_class=cat/stack/tensor-list-input; harness=determinism_check.py`
`verified_on: 13_Cat M-I run-1 (fixed in-run, harness-side)`

determinism_check.py's shallow `[t.npu() if isinstance(t, torch.Tensor) else t for t in inp]` leaves a nested tensor LIST on CPU → any tensor-list op's ModelNew (correctly) raises "tensors must be on NPU" while verification_ascendc.py passes. Fixed harness-side: recursive `_to_npu` (list/tuple descent), mirroring verification_ascendc.py's `_move_to_device`. Without the fix, determinism measurement is impossible for the whole cat/stack op family.

- **Symptom signature**: `RuntimeError: tensors must be on NPU` from ModelNew during the det check only.
- **Note**: fixed in this run's harness; if the fix is lost, this card is the regression signature. backend=ascendc.
