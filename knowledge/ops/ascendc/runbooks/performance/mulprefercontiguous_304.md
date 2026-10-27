---
type: "Runbook"
title: "MulPreferContiguous 稀疏广播拦截失败且慢于 304"
description: "B 类稀疏广播（首轴稀疏、stride 大）触发 MulPreferContiguous 拦截转连续，baseline GetWorkspaceSize 返回 -1.0（schMode 空）或性能慢于 fixed 直接走 304 nddma_transpose_noLoop。"
tags: ["task.debugging", "topic.data_movement.layout_transform", "topic.implementation.tiling", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-18T01:26:49Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["schMode_304"]
---

# MulPreferContiguous 稀疏广播拦截失败且慢于 304

## 现象

在 Ascend950PR 真机对比 `aclnnMul` 非连续广播性能时（baseline = master，含 `MulPreferContiguous` 拦截；fixed = PR#2803 + PR#380，删除 `MulPreferContiguous`），构造 144 条白盒用例：A 类 136 条（非连续 vs 非连续），B 类 8 条（连续 vs 非连续，首轴稀疏广播）。B 类用例（如 `self=[1,N]` storage `[1,2N]` stride `[2N,2]`）出现显著性能差异与功能性失败：

- **baseline**：`MulPreferContiguous`（aclnn 层 4 阈值拦截）把 B 类稀疏广播用例拦截转连续，再走 `nddma_noLoop` / `nlast_ub_brc_r2` 模板。8 用例中全部转连续失败——`aclnnMulGetWorkspaceSize` 返回 -1.0，schMode 空。
- **fixed**：删除 `MulPreferContiguous`，B 类直接进入 atvoss 非连续 tiling，路由到 `schMode=304`（`nddma_transpose_noLoop`），全部成功。6 条可比用例 fixed 相对 baseline 延迟比 0.41~0.44（fixed 更快）。
- A 类 131 条 SAME（±5%），无系统性回归。

## 适用场景

- **工程形态**：aclnn 算子非连续广播性能对比（ttk 真机单跑，`-b=release`，自研 perf CSV 采集 tilingKey/schMode/耗时）。
- **API 路径**：`aclnnMul` → `aclnn_mul.cpp` 的 `MulPreferContiguous` 拦截分支 → atvoss `nddma_transpose_noLoop`（304）/ `nddma_noLoop`（1）/ `nlast_ub_brc`（102）模板。
- **验证过的平台与 CANN 版本**：Ascend 950PR（9589），CANN 9.2.0 weekly，`soc_version=Ascend950PR_9599`。仅 950 实测（ttk single_run）。
- **适用粒度**：`operator_family`（所有走 atvoss broadcast 非连续模板的二元算子，如 mul/add）。
- **路线优先级**：`recommended_fastpath`（B 类稀疏广播直接走非连续 304，不拦截转连续）。
- **不适用或未验证场景**：连续用例（baseline 与 fixed 均走连续 sch=0 模板，无差异）；A 类非连续 vs 非连续（无系统性差异，4~6 条 SLOWER 为噪声）；非 950 平台未验证；ttk single_run 非标准化几何平均。

## 判别方法

### 触发（trigger）

B 类稀疏广播用例（首轴稀疏，如 `self=[1,N]` storage `[1,2N]` stride `[2N,2]`）在 baseline（含 `MulPreferContiguous` 拦截）上触发 `aclnnMulGetWorkspaceSize` 返回 -1.0，schMode 空——ttk perf CSV 中 `us_median` 列为 -1.0，表示 GetWorkspaceSize 失败。8 条 B 类用例全部失败。

### 探针（probe）

用 ttk 真机 perf 对比工具在同一 Ascend950 上跑 baseline（commit `2cd7568`，含 `MulPreferContiguous`）与 fixed（PR#2803 + PR#380，删 `MulPreferContiguous`），采集 144 条白盒用例的 schMode/tilingKey/`us_median` CSV。重点检查 B 类（`compareType=CONT_vs_NC`）用例的 `schMode` 列与 `us_median` 列：baseline 行 schMode 是否空 + us_median 是否 -1.0；fixed 行 schMode 是否为 304。

### 观测（observation）

- B 类 8 用例：baseline `schMode` 列全部空，`us_median` 列全部 -1.0（GetWorkspaceSize 失败）；fixed `schMode` 列全部 304（模板 `nddma_transpose_noLoop`），`us_median` 列为正值（成功执行）。
- B 类 6 条可比用例：fixed(304 `nddma_transpose_noLoop`) 相对 baseline(转连续 `nddma_noLoop`/`nlast_ub_brc_r2`) 延迟比 0.41~0.44（fixed 更快）。
- A 类 131 条：SAME（±5%），4~6 条 SLOWER 为噪声。
- schMode 与模板映射：301=`nlast_nddma`，302=`nlast_ub_brc`，303=`last_transpose`，304=`nddma_transpose_noLoop`。

### 判读（interpretation）

`MulPreferContiguous` 是 aclnn 层 4 阈值拦截，把"看似可转连续"的非连续输入先转连续再走连续 kernel，避免非连续 tiling 复杂度。但对 B 类稀疏广播（首轴稀疏、stride 大），转连续需全量拷贝 `2N` 元素再走连续模板，开销大于直接走 304 NDDMA 转置无循环搬运；且转连续路径对部分 shape 的 `GetWorkspaceSize` 会失败（返回 -1.0）。删除拦截让非连续 kernel 直接处理，既快又更鲁棒。

## 验证信号

- B 类 8 用例 baseline `schMode` 空、`us_median`=-1.0（GetWorkspaceSize 失败），fixed `schMode`=304、`us_median` 正值（成功）（来源平台：950；证据等级：tool_observed）
- B 类 6 用例 fixed(304 `nddma_transpose_noLoop`)/baseline(转连续 `nddma_noLoop`/`nlast_ub_brc_r2`) 相对延迟比 0.41~0.44（来源平台：950；证据等级：tool_observed）
- A 类 131 用例 SAME（±5%），4~6 条 SLOWER 为噪声，PR#380 tiling 优化无系统性回归（来源平台：950；证据等级：tool_observed）
- schMode 分布：B 类 baseline schMode 空 / fixed schMode=304（模板 `nddma_transpose_noLoop`）；301=`nlast_nddma`，302=`nlast_ub_brc`，303=`last_transpose`，304=`nddma_transpose_noLoop`（来源平台：950；证据等级：tool_observed）
