---
type: "Runbook"
title: "Matmul workspace 预偏移 totalN 回读 banned"
description: "Ascend910B 上 QuantBatchMatMulV3 反量化阶段把 INT32 workspace 指针预偏移到 mStart*N+nStart 并用 totalN 作行 stride 回读，期望 matmul 按全局 N 行步幅写入；但 IterateAll 紧凑写 [singleCoreM×singleCoreN] 到指针基，回读错位、精度偏差。本 trace 内反复切换 totalN/singleCoreN stride 未能闭环精度，该预偏移+totalN stride 回读路线 banned。"
tags: ["task.debugging", "topic.precision", "topic.programming_model.execution_unit.cube", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
aliases: ["MM_CFG_NO_PRELOAD", "enUnitFlag"]
---

# Matmul workspace 预偏移 totalN 回读 banned

## 现象

CANNJudge cube1（QuantBatchMatMulV3，INT8 cube matmul→INT32 workspace→AIV 反量化）在 Ascend910B 上本地 CPU 仿真精度正确，但 judge 提交后前 6 个用例 `Wrong Answer`、TC6 超时。失败用例的误差比例分布广泛（TC0≈0.057%、TC5=25%），提示计算大体正确但存在数据搬运错位。agent 定位到反量化阶段把 matmul 输出的 INT32 workspace 指针预偏移 `wsInt32GM = wsInt32GM[offsetWs_]`（`offsetWs_ = mStart * N + nStart`，按全局 N 行主序偏移），并按 `totalN` 作行 stride 回读，期望 matmul 把单核结果写入全局 `[M, N]` 布局的对应位置。

## 适用场景

- 工程形态：cube+vector 混合算子，AIC `Matmul` 高阶 API 把 INT32 中间结果经 `IterateAll` 写入 GM workspace，AIV 从 workspace 读回做反量化；多核切分单核 M/N。
- API 路径：`Matmul<...>::SetSingleShape / SetTensorA / SetTensorB / IterateAll(cGm[offset], enAtomic) / End`；workspace 大小 = `usedCoreNum × singleCoreM × singleCoreN × sizeof(int32_t)`。
- 验证过的平台与 CANN 版本：Ascend910B（a2），CANN 9.0.0。
- 适用粒度：`operator_family`（所有用 Matmul 高阶 API + workspace 中转 + 多核切分的 cube 算子）。
- 路线优先级：`banned`（预偏移 workspace 指针到全局 `(mStart, nStart)` 再用 `totalN` stride 回读这条路线在本 trace 内客观未闭环精度）。
- 不适用或未验证场景：单核整 M/N 输出（`singleCoreM=M`、`singleCoreN=N`）时紧凑布局即全局布局，不触发；Ascend950/a3 的 `IterateAll` 布局语义未独立验证。

## 失败路线

**尝试**：把 workspace INT32 指针预偏移到 `offsetWs_ = mStart * N + nStart`（全局行主序偏移）后传给 `mm.IterateAll(wsInt32GM)`，期望 matmul 把单核结果写入全局 `[M, N]` 布局的对应槽位；AIV 回读时用 `totalN` 作行 stride。精度不对后，在 `totalN` 与 `singleCoreN` 之间反复切换回读 stride，不做 dump 验证即盲调。

**失败**：

1. judge 前 6 用例持续 `Wrong Answer`（TC0 ratio=0.00057~0.058，TC1≈0.99%，TC5=25%），TC6 仍超时；CPU debug 路径用三重循环手填全局布局故不暴露本错位。
2. 在 `totalN`/`singleCoreN` stride 之间反复切换未能在 judge 上闭环精度——本 trace 内 `IterateAll` 紧凑布局是强候选根因但未闭环验证。
3. 真正根因在本 trace 结束时未闭合：并发的候选根因包括 `SetTensorA` 传转置标志致双重转置、`SetOrgShape` 配置（姊妹卡已判明非根因）等，offset/stride 单点修复不足以闭环。

**banned**：把 `wsInt32GM[offsetWs_]` 当作全局布局槽位传给 `IterateAll`、再用 `totalN` stride 回读这条路线——matmul 不感知 `totalN`，紧凑写入指针基导致回读错位；且盲切 stride 不做 dump 验证浪费提交次数。正确方向（本 trace 未落定干净修复）：workspace 按 `coreIdx * singleCoreM * singleCoreN * sizeof(int32_t)` 分区，每核独占连续段，回读用 `singleCoreN` 作 stride，再由 AIV 按全局 `(mStart, nStart)` + `totalN` stride 散布到输出 GM。

## 验证信号

- judge 提交后前 6 用例 `Wrong Answer`，TC0 ratio=0.00057~0.058、TC5 ratio=0.25（来源平台：a2；证据等级：tool_observed）。
- cannbot-skills ascendc-api-best-practices（webfetch 自 gitcode.com/cann/cannbot-skills）明确 `IterateAll(cGm[offsetC], enAtomic)` 自动遍历 `[singleCoreM × singleCoreN]` 内所有 baseBlock 并写回 GM、推荐 `SetTensorA(aGm_[offsetA], false)`、workspace 大小 = `usedCoreNum × baseM × baseN × sizeof(int32_t)`（来源平台：a2；证据等级：artifact_verified）。
- 本 trace 内 agent 在 `totalN`/`singleCoreN` stride 间反复切换后 judge 仍未通过，offset/stride 单点修复未闭环精度（来源平台：a2；证据等级：tool_observed）。

## 失效边界

- 单核整 M/N 输出不触发（紧凑布局即全局布局）。
- Ascend950/a3 的 `IterateAll` 布局语义未独立验证，推断可迁移（cannbot-skills 为通用口径），未实测。
- 本卡只否定"预偏移+`totalN` stride 回读"这一路线；`IterateAll` 紧凑写契约本身（artifact_verified）是设计正确用法的依据，不是被否定的对象。