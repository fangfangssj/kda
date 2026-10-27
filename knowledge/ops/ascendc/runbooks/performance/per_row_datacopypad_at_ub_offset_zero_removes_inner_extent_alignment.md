---
type: Runbook
title: 逐行 DataCopyPad 落在 UB 偏移 0 可消除内维对齐问题，无需 GM 补齐中转
description: 内维 N 全部 32B 不友好时，按 [Ct, N] 块布局会让多数行落在非对齐 UB 偏移；逐行拷入恒从偏移 0 起的缓冲使所有向量操作数对齐，并省掉一遍 GM 补齐中转。
tags:
- task.optimization
- topic.alignment
- topic.data_movement
status: draft
sources: []
created_at: '2026-09-25T08:59:06Z'
updated_at: "2026-10-15T00:00:00Z"
platforms:
- '950'
aliases: ["CAND-PP-ROW-LANDING-ALIGNMENT"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# 逐行 DataCopyPad 落在 UB 偏移 0 可消除内维对齐问题，无需 GM 补齐中转

## 证据

实测的每个 `N`（35 / 77 / 153 / 169 / 323 / 341 / 375 / 483 / 2048）都是 32B 不友好的，
因此 `[Ct, N]` 块布局会把**多数行**放到非对齐的 UB 偏移上。

把每行单独拷入一个**恒从偏移 0 开始**的缓冲，则无论 `N` 为何，所有向量操作数都是对齐的，
并且**完全省掉**补齐后的 GM 中转那一遍。

行缓冲在循环前**一次性**预清零到 `alignUp(N, 64)`：每行的 `Cast` 只写 `[0, N)`，
所以尾部保持干净且无需逐行清理。

实测来源：run `142408a6-72e4-45a7-adea-1c83590452ff`（算子 48_GCModule），soc=Ascend950DT_9582，cann=9.1.0。

## 与既有卡的边界（作者初判 PARTIAL）

同一个对齐危害已有两张卡，但给的是**另一种**治法，本卡与它们是姊妹关系而非重复：

- `runbooks/compilation/datacopypad_dst_32b.md`：多块 DataCopyPad 的 dst 基址累积到非 32B 对齐
  → MTE 地址未对齐、vector core exception。
- `runbooks/compilation/b16_datacopypad_ub_gm_32b_fault.md`：fp16/bf16 UB→GM 源地址非 32B 对齐
  直接硬件 fault；它的治本办法是**选整除输出边界的切分因子让源恒对齐**。

本卡的治法不同：**不去挑切分因子，而是让每行都落在偏移 0**，于是对齐与 `N` 无关。当 `N` 由外部
给定、无法选出整除的切分因子时（本例 9 个 `N` 全部 32B 不友好），上面那条治法不可用，本卡可用。
两者都不覆盖"本卡是否比补齐中转更快"——见下方边界，本卡不提供独立加速比。

## 适用范围与边界

- 适用：soc=Ascend950*；内维 32B 不友好的 row_reduction / normalization 类。
- **未验证**：soc=Ascend910_9382（arch22 的 `DataCopyPad` ext-params 形状不同）。
- **代价模型**：它把一次宽拷贝换成 `Ct` 次窄拷贝，因此只有在它替掉的对齐修补本身
  需要一遍同等数据量的中转时才划算。
- **本卡不提供该技术的独立加速比**。该 run 的性能腿整体为 median 1.99×
  （min 0.81×、max 3.26×、50/50 实测、门 1.0×），那是**算子整体结果**，
  不是本技术的消融实验。

## 适用场景

- **工程形态**：内维 N 全部 32B 不友好时，按 [Ct, N] 块布局会让多数行落在非对齐 UB 偏移；逐行拷入恒从偏移 0 起的。
- **验证过的平台**：Ascend950（以正文验证信号为准）。
- **适用粒度**：`operator_family`——与该卡所述 API/机制/缺陷族相同的场景。
- **不适用/未验证场景**：未在正文验证信号中列出的平台、dtype、shape 与该 API 变体；跨平台迁移须重新验证。
