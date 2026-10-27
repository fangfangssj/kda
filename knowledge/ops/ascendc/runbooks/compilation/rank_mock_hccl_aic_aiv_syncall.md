---
type: "Runbook"
title: "单 rank mock HCCL 致 AIC+AIV SyncAll 死锁"
description: "atcoss BlockPrologueAllGather 等 AIC+AIV 通算融合 kernel 在单 rank mock HCCL context 下执行超时，mock 不提供 SyncAll 跨核配对所需的状态标志写入流程致 ReadStatus/SyncAll 死锁；该 mock 路线 banned，分布式融合 kernel 须用真实多卡 HCCL。"
tags: ["task.debugging", "topic.implementation.synchronization"]
status: stable
sources: []
created_at: "2026-08-18T07:36:25Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# 单 rank mock HCCL 致 AIC+AIV SyncAll 死锁

## 现象

atcoss `all_gather_quant_matmul` kernel 接口的 ST 初版采用 mock HCCL context（`tests/st/common/mock_hccl_context.h` 的 `MockHcclContextBuilder`，所有 `windowsIn[i]` 指向同一 window 内存，单卡模拟分布式通信域）。在 Ascend950 单卡上以 `<<<totalCores, nullptr, stream>>>` 启动 kernel 后，kernel 执行超时无返回。

## 适用场景

- 工程形态：ST 测试，单卡 mock HCCL context 模拟分布式通信域。
- API 路径：`BlockPrologueAllGather`（atcoss `block/blaze_ext/prologue`）的 `ReadStatus`/`WriteStatusToWin`/`SyncAll<true>()` workspace 消息协议。
- 验证平台与版本：Ascend950（dav-3510/arch35）+ CANN 9.1.0。
- 适用粒度：`operator_family`（适用于 atcoss/mc2 用旧 `HcclCombinOpParam` workspace 消息协议的 blaze_ext prologue/epilogue 类通算融合 kernel）。
- 路线优先级：`banned`（mock HCCL 对 AIC+AIV SyncAll 配对 kernel 是死路）。
- 不适用或未验证场景：纯 AIV 或纯 AIC、无 SyncAll 配对的 kernel 未验证可能可用 mock；a2/a3 未验证。

## 失败路线

**banned 路线：单卡 + mock HCCL context（`MockHcclContextBuilder`，`windowsIn[]` 全指向同一 window 内存）启动 `BlockPrologueAllGather` 通算融合 kernel。**

- 尝试：在 Ascend950 单卡上用只填 `windowsIn[]` 同址指针的最简 mock context 启动 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV）的 `BlockPrologueAllGather` kernel。
- 客观失败：kernel 执行超时无返回（来源平台：950；证据等级：tool_observed）。
- 边界：该 mock 路线对 AIC+AIV SyncAll 配对的通算融合 kernel 是 banned 路线。用户明确指出环境有真实卡（4 卡 Ascend950），应改用真实多 rank HCCL（`HcclCommInitRootInfoConfig` + fork 多进程 + TCP 交换 rootInfo）。纯 AIV 或无 SyncAll 配对的 kernel 是否可用 mock 未验证。

## 根因

> 以下为 trace 内 assistant_message 的根因分析（interpretation），非来自非脱敏 tool_result 的逐字证据；确切的死锁堆栈/日志未从 trace 恢复。

atcoss `all_gather_quant_matmul` kernel 是 `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV），`totalCores = aivNum + aicNum = 8 + 4 = 12`。kernel 内 AIC 核走 `Run()` 的 matmul 路径、AIV 核走 prologue 路径，两者的 `SyncAll<true>()` 需要所有核都到达配对。`BlockPrologueAllGather` 的 `ReadStatus` 在 `WinStatusAddr(curRankId_)` 处轮询等待远端 rank 的状态标志，`WriteStatusToWin` 写入 `WinStatusAddr(remoteRankId_)`。单 rank（rankNum=1）时 `remoteRankId_ == curRankId_` 理论上写读同址，但 mock context 不提供完整的跨核同步信号写入流程，`SyncAll` 在 totalCores 不匹配/状态标志未由 mock server 写入时死锁，kernel 执行超时。

## 验证信号

- 单 rank mock HCCL context 下 `<<<totalCores, nullptr, stream>>>` 启动 `BlockPrologueAllGather` kernel 后执行超时无返回（来源平台：950；证据等级：tool_observed）。

## 失效边界

- a2/a3 的 AIC/AIV 核数配比不同，但"mock HCCL 对 SyncAll 配对 kernel 死锁"这一机制推断一致，未验证，推断可迁移。
- 若 mock context 实现完整的 `MockHcclServer`（`tests/mc2_mock_test_frame` 的 `MockHcclServer`）来模拟跨核同步信号，可能解锁；但本 trace 未实现该完整 mock server，未验证。

## 坏实践

- 用只填 `windowsIn[]` 同址指针的最简 mock context 测 AIC+AIV SyncAll 配对 kernel，忽略 `SyncAll` 的核间配对与状态标志写入流程。
- 单 rank 测试分布式 kernel 时假设 `remoteRankId_ == curRankId_` 就一定不死锁——`SyncAll` 的 totalCores 配对与 mock server 写标志流程是独立条件。

## 预防

- AIC+AIV 通算融合 kernel 的 ST 优先用真实多卡 HCCL（环境有卡时），不要用 mock。
- 若必须单卡 mock，需接入 `tests/mc2_mock_test_frame` 的完整 `MockHcclServer` 模拟跨核同步信号，并保证 `totalCores` 与 AIC/AIV 配比正确。