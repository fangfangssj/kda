---
type: "Runbook"
title: "通算流水 tileCnt>1 性能反降 banned"
description: "matmul_reducescatter 通算流水 tileCnt>1 即使精度通过、死锁解除，端到端耗时随 tileCnt 反增；根因为 fixpipe 固定开销随 Blaze 调用次数线性累积 + Pattern B 依赖链限制流水窗口。tileCnt>1 路线 banned。"
tags: ["task.optimization", "topic.implementation.pipeline", "topic.implementation.synchronization", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# 通算流水 tileCnt>1 性能反降 banned

## 现象

在为 matmul_reducescatter（MC2 通算融合，4P SHMEM ReduceScatter）实现通算流水（AIC 逐 tile matmul ‖ AIV 逐 tile reduce，batch=1、tileCnt 控制 M 轴分块）后，精度全 PASS、死锁已解除，但性能不升反降：

- tileCnt=1（无流水，单次 batch matmul）是基线参照路线；
- tileCnt=2 端到端耗时相对 tileCnt=1 基线约 1.30x；
- tileCnt=4 相对 tileCnt=1 基线约 2.13x。

tileCnt 越大越慢，与"流水掩盖收益"的预期完全相反。

## 适用场景

- 工程形态：kernel 直调（ops-direct-invoke），MC2 多卡通算融合算子的通算流水（mLoop）改造。
- API 路径：Blaze matmul（`quantMatmulKernelImpl_`）/ CrossCore flag / SHMEM UDMA GET。
- 验证平台与版本：Atlas A2（ascend910b，4 卡），4P M=2048 N=3584 K=5120。
- 适用粒度：`same_operator`（本算子在 4P M=2048 N=3584 K=5120 shape 下的 tileCnt>1 流水路线判定）。
- 路线优先级：`banned`（tileCnt>1 通算流水路线在本算子本 shape 下 banned；推荐采用 tileCnt=1 batch matmul 参照路线）。
- 不适用或未验证场景：其它 shape（尤其通信量远大于计算量、Pattern A 先通后计或无依赖通信）下 tileCnt 流水是否有收益未验证；Ascend950（950）与 A3（a3）未实测。

## 失败路线

尝试：把 batch 优化（tileCnt=1 单次 batch matmul）改为 batch=1 + tileCnt 控制 M 轴分块流水，AIC 逐 tile 算 matmul、AIV 逐 tile reduce，ping-pong bufferSize=2，期望"AIC 算 tile t+1 ‖ AIV 搬 tile t"掩盖通信。

客观失败（来源平台：a2；证据等级: tool_observed）：

- tileCnt=2 端到端相对 tileCnt=1 基线约 1.30x（变慢）。
- tileCnt=4 相对 tileCnt=1 基线约 2.13x（变慢）。
- aiv_mte2_time 相对 tileCnt=1 基线无显著变化（UDMA 优化的搬运节省在所有 tileCnt 下都成立，不是 tileCnt 流水的收益来源）。

边界与回退：tileCnt>1 路线 banned；回退到 tileCnt=1 batch matmul（参照路线）。本算子是 Pattern B（先计后通），通信内容是 matmul 计算结果，AIV 必须等 AIC 算完该 tile 才能搬运+reduce，流水掩盖窗口仅 "AIC 算 tile t+1 ‖ AIV 搬 tile t"；而 ReduceScatter 总通信量固定（每卡必须读完其他 3 卡对应段做累加，总搬运量不随 tileCnt 减小），切 tileCnt 只改调度不改工作量，流水掩盖空间本就有限。

根因（机制链，不跳步）：tileCnt↑ 使 headMSize↓（headMSize = M/tileCnt）。曾一度归因为"headMSize 减小导致 Blaze 分不满 28 核"，但实测核数表证明该归因错误——

- tileCnt=1: headMSize=2048, mBlockCnt=8, nBlockCnt=14, totalBlockCnt=112（每核 4 块）
- tileCnt=2: headMSize=1024, mBlockCnt=4, nBlockCnt=14, totalBlockCnt=56（每核 2 块）
- tileCnt=4: headMSize=512, mBlockCnt=2, nBlockCnt=14, totalBlockCnt=28（每核 1 块）

28 核在所有 tileCnt 下均分满。真因是 **fixpipe 固定开销随 Blaze 调用次数线性累积**：tileCnt=2 时 Blaze 调用 2 次，fixpipe_time 相对 tileCnt=1 基线约 1.26x；tileCnt=4 时 4 次，累积更多。fixpipe 每次的初始化开销（`mmadOp_.Init()` 重复执行）随调用次数叠加，远超流水掩盖的有限收益。enableAtomicAdd 默认 false（`quant_matmul_mx_block_mmad_swat.h:204`），fixpipe 变大与 atomic add 无关。

## 性能影响

相对 tileCnt=1 batch matmul 参照路线（来源平台：a2）：

- tileCnt=2 端到端约 1.30x 参照路线（变慢）。
- tileCnt=4 端到端约 2.13x 参照路线（变慢）。
- fixpipe_time tileCnt=2 相对 tileCnt=1 约 1.26x，随 Blaze 调用次数线性累积。
- aiv_mte2_time 在所有 tileCnt 下相对 tileCnt=1 基线无显著变化（UDMA 搬运节省与 tileCnt 无关）。

## 验证信号

本卡验证信号来自 trace 中对 tileCnt=1/2/4 的 msprof task-based 采集（4 卡，最后 5 轮平均取 max，来源平台：a2）。

- tileCnt=2 端到端相对 tileCnt=1 基线约 1.30x（来源平台：a2；证据等级: tool_observed）
- tileCnt=4 相对 tileCnt=1 基线约 2.13x（来源平台：a2；证据等级: tool_observed）
- 28 核在 tileCnt=1/2/4 下均分满（totalBlockCnt=112/56/28），推翻"分不满核"归因（来源平台：a2；证据等级: tool_observed）
- enableAtomicAdd 默认 false，fixpipe 变大与 atomic add 无关（来源平台：a2；证据等级: tool_observed）
- 首轮冷启动长尾仅第 1 轮出现、mac_ratio 正常非死等，后续 9 轮稳定，统计取最后 5 轮可靠（来源平台：a2；证据等级: tool_observed）

## 坏实践

- 用"headMSize 减小分不满核"作为 tileCnt>1 变慢的归因而未先用核数表验证（本 trace 中该归因被实测推翻）。
- 把首轮长尾当死锁或性能结论——必须排除冷启动轮，取最后 5 轮平均。

## 预防

- tileCnt>1 流水前先算理论流水收益：Pattern B 下流水窗口受限于"AIC 算 tile t+1 ‖ AIV 搬 tile t"，而总通信量不随 tileCnt 减小，掩盖空间有限；若 fixpipe 等固定开销随调用次数线性累积超过流水收益，tileCnt 路线 banned。
- 用 msprof task-based 采集多 tileCnt 对比，确认端到端是否下降再决定是否采用流水。

## 失效边界

- 已验证：Atlas A2（ascend910b）4 卡，4P M=2048 N=3584 K=5120，tileCnt>1 流水 banned。
- 未验证，推断可迁移：同算子在通信量远大于计算量、或 Pattern A（先通后计/无依赖通信）的 shape 下，tileCnt 流水可能有收益，本卡结论不得外推到那些场景。
- Ascend950（950）与 A3（a3）未实测，推断可迁移。