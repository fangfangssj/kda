---
type: Runbook
title: 'KB 选择器编号在语料迁移后失配：换语料必须先重建编号对照表'
status: draft
description: 'OKF/语料迁移后卡片不带旧编号，KB 选择器按旧编号点卡会 spawn 失败（KB_SELECTED_ENTRY_UNRESOLVED）：spawn 前必须用 anchor map 生成器重建编号→卡片对照表并验证所需编号全解析'
sources: []
tags:
- task.debugging
- topic.architecture
platforms:
- '950'
created_at: '2026-09-18T00:00:00Z'
updated_at: "2026-10-15T00:00:00Z"
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# KB 选择器编号在语料迁移后失配（spawn 失败教训，2026-09-18）

**症状**：spawn aog-kernel-worker 连续失败，报错 `KB_SELECTED_ENTRY_UNRESOLVED: P-P200`（此前同一原因卡在 EC-39）。**根因**：OKF 转换后语料从旧 monolith KB 换成了 fork 仓（cannbot-knowledge，PR36 语义命名），卡片不再携带旧编号；但 KB 选择器仍按旧编号（EC-39、P-P200 等）点卡，而新语料缺少"旧编号 → 新卡片"的对照表（anchor map）和对应的 KB_INDEX 投影，于是选择器选中的编号在新语料里根本解析不到任何卡片。**修复**：泛化 `docs/design/okf_anchor_map_gen.py`（新增 `--corpus` / `--legacy-anchor-map` / `--disposition-map` 参数），基于 fork 语料重建对照表（550 个旧编号全部解析、0 遗漏），生成 411 行 KB_INDEX 投影；对照表中确实缺失的 P-P200、CAND-PP82 两张卡，从 legacy KB 迁移进 fork 语料补齐。**今后如何预防**：凡发生 KB 迁移或语料更换（fork、重命名、OKF 转换等），在 spawn 任何依赖 KB 选择器的 agent 之前，必须先用 anchor map 生成器重新生成"编号 → 卡片"对照表和 KB_INDEX 投影，并验证选择器本次运行所需的所有编号在新语料中全部解析（0 漏）；解析不全时先补卡或修对照表，再 spawn——不要把 spawn 失败当作重试信号，它是语料与索引失配的确定性信号。

## 适用场景

- **工程形态**：OKF/语料迁移后卡片不带旧编号。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
