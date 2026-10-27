---
type: "Runbook"
title: "MIX_AIC_1_1 仅启用 28/56 AIV 带宽不饱和"
description: "950（Ascend950PR，28 CubeCore:56 VectorCore）上 KERNEL_TYPE_MIX_AIC_1_1 强制 1:1 配比，blockNum 上限=28，只启用 28/56 AIV（50%），HBM 带宽无法饱和；测 HBM 带宽应回退到 AIV_ONLY 56 核路线，MIX_AIC_1_1 在此场景 banned。"
tags: ["task.optimization", "topic.programming_model.execution_unit.vector", "topic.programming_model.execution_unit.cube", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-17T02:40:47Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# MIX_AIC_1_1 仅启用 28/56 AIV 带宽不饱和

## 现象

在 Ascend950PR（950，NpuArch=3510）上把一个原为 A3/c220 设计的 L2 带宽测试工程（`KERNEL_TYPE_MIX_AIC_1_1`，blockNum=25）适配过来重测带宽。把 blockNum 从 25 调到 MIX 模式上限 28 后，纯读场景带宽仍远低于用户给出的 HBM 与 L2 规格（绝对值仅作 provenance，不进正文）——启用 L2 命中后远低于 L2 规格。排查发现 950 的核配比与 MIX kernel 的 1:1 约束冲突：950 有 28 个 CubeCore + 56 个 VectorCore（platform ini `cube_core_cnt=28`、`vector_core_cnt=56`），而 `KERNEL_TYPE_MIX_AIC_1_1` 强制 1 AIC:1 AIV，blockNum 上限受 CubeCore 数约束 = 28，于是只启用了 28 个 AIV，另外 28 个 AIV 空闲（核利用率 50%）。

## 适用场景

- **工程形态**：基于 Ascend C MIX Kernel（`KERNEL_TYPE_MIX_AIC_1_1`）的 GM↔UB/L2/HBM 带宽测量 kernel 直调工程。
- **API 路径**：`KERNEL_TYPE_MIX_AIC_1_1` / `KERNEL_TYPE_MIX_AIC_1_2` / `KERNEL_TYPE_AIV_ONLY` 宏；`GetBlockIdx()`、`DataCopy`、`SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE)`。
- **验证过的平台与 CANN 版本**：Ascend950PR（950，NpuArch=3510，CANN 9.1.0，platforms=950，28 CubeCore:56 VectorCore，L2=128MB）。
- **适用粒度**：`platform_wide`（950/3510 上凡用 MIX_AIC_1_1 测带宽均受此核配比约束）。
- **路线优先级**：`banned`（MIX_AIC_1_1 在 950 上测 HBM/L2 饱和带宽为 banned 路线；应回退 `KERNEL_TYPE_AIV_ONLY` 或 `KERNEL_TYPE_MIX_AIC_1_2`）。
- **不适用或未验证场景**：A3/c220（25 block、AIC:AIV=1:1 与本卡无关，platforms 不含 a3）；非带宽饱和场景（如只想各跑一次看通路）不在此限。

## 失败路线

**尝试**：沿用 A3 工程的 `KERNEL_TYPE_MIX_AIC_1_1`，把 blockNum 从 25 调到 MIX 模式上限 28，纯读场景跑 `DataCopy`（每 loop 读同地址，L2 命中）。期望：接近 HBM / L2 规格。

**失败/回退**：实测带宽仍不饱和，根因不是数据量也不是 L2 容量（每核仅 64KB、总量 1.6MB 远小于 L2 128MB），而是**核没占满**：MIX_AIC_1_1 的 1:1 约束让 blockNum 卡在 CubeCore 数（28），28 个 AIV 闲置。本路线对"HBM/L2 带宽饱和测量"判 `banned`。

**相对代价**：MIX_AIC_1_1 路线核利用率 28/56=50%；回退到 `KERNEL_TYPE_AIV_ONLY`（纯 AIV，可用全部 56 核 + `SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE)` 禁用 L2 + 每 loop 偏移地址避免命中）后 HBM 读带宽达到规格（参照路线：AIV_ONLY 56 核；来源平台：950）。即 MIX 路线相对 AIV_ONLY 路线少了约一半 AIV 参与搬运，故不饱和。

**边界**：MIX_AIC_1_1 本身不是错——它适合需要 AIC 与 AIV 协同计算（Cube+Vector 联合）的算子 kernel；只是不适合"用满全部 AIV 测带宽"这一场景。若要同时用 AIC+AIV 测双向带宽又想占满 AIV，应改用 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC:2 AIV，与 950 的 28:56 配比一致），本 trace 仅确认 `KERNEL_TYPE_MIX_AIC_1_2` 宏存在（预处理确认），未实测其带宽饱和度。

## 根因

950 的核配比是 28 CubeCore:56 VectorCore（1:2），而 `KERNEL_TYPE_MIX_AIC_1_1` 强制 1 AIC:1 AIV 配比，blockNum 上限受 CubeCore 数（28）约束，只启用 28 个 AIV，另外 28 个 AIV 空闲，带宽测量无法占满 AIV 通道。

## 验证信号

- 950 platform ini：`cube_core_cnt=28`、`vector_core_cnt=56`、`SoC_version=Ascend950PR_9579`、`NpuArch=3510`（来源平台：950；证据等级: tool_observed）
- MIX_AIC_1_1 模式 blockNum 调到 28 后 AIV 仍只 28/56（50%）参与，纯读带宽未达规格（来源平台：950；证据等级: tool_observed）
- 回退 `KERNEL_TYPE_AIV_ONLY` 56 核 + `SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE)` 后 HBM 读带宽达到规格（参照路线：AIV_ONLY 56 核）（来源平台：950；证据等级: tool_observed）

## 失效边界

- 本卡结论绑定 950/3510 的 28:56 配比；A3/c220 的核配比不同（platforms 不含 a3）。
- 推断 `KERNEL_TYPE_MIX_AIC_1_2`（1:2）可占满 56 AIV，但**本 trace 未实测其带宽饱和度，推断可迁移**，需独立验证后才能作为推荐路线。
- `SetL2CacheHint` 在 950/arch35 需带模板参数 `SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE)`（非模板形式搜不到定义）；本卡不展开其调用细节。