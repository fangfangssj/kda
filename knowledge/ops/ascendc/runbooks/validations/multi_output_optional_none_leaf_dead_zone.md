---
type: Runbook
title: "多输出算子的设计性缺席输出（None 叶）与死区（如 Aqk 严格上三角）必须在测量侧显式排除——否则每个用例都会被误判"
description: "多输出算子的设计性缺席输出（None 叶）与死区（如 Aqk 严格上三角）必须在测量侧显式排除——否则每个用例都会被误判"
tags: [task.testing, topic.precision]
status: draft
sources: []
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["CAND-MULTI-OUTPUT-NONE-LEAF-DEAD-ZONE"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# CAND-MULTI-OUTPUT-NONE-LEAF-DEAD-ZONE

## 适用场景

- 多输出算子（KDA 族 72/73/88 等）：输出六元组含可选位（qg 在 disable_recompute=False 时双侧 None）与契约死区（Aqk 严格上三角 kernel 侧为未初始化垃圾，golden 置 0，不参与比较）。

## 结论（72 run，2026-09-20）

1. **None/None = 一致，不是 mismatch**：消费者（评分器/对比器）必须把「双侧均按设计缺席」当 agreement；单侧 None 才是结构分歧。踩坑实例：O5 评分器 `_profile_tree_passed` 无 None 分支 → 字节正确的候选被判 0/50（harness !1364 已修，对照 0/50→50/50）。
2. **死区排除是 op 契约的一部分**：worker 的 runner 遵守排除（50/50 PASS），通用复测路径若不做排除会把垃圾区当真差异。写 runner 时从 model docstring 里把「不参与比较」的区域显式列出。
3. **allclose 对该族 vacuous**：输出幅值远低于 atol 时 allclose 不可能 fail，MERE 承重（见 CAND-PRECISION-ALLCLOSE-VACUOUS-SMALL-OUTPUT）。

- **Evidence**：72 归档 REGRADE_EVIDENCE.md + ARCHIVE_RECEIPT.json（mfu56_closeloop 专用树）。
- **Cross-ref**：!1364（评分器修复）、CAND-PRECISION-ALLCLOSE-VACUOUS-SMALL-OUTPUT。
