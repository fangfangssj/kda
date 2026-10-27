---
type: Runbook
title: "bare AscendC::Exp() defaults to ExpAlgo::INTRINSIC whose input-dependent error pushes softmax past frozen-case rtol on V351 — pass ExpConfig{PRECISION_1ULP_FTZ_TRUE} explicitly"
description: "On A5/CANN 9.2.0 an Exp call without an explicit ExpConfig uses ExpAlgo::INTRINSIC; its input-dependent error is large enough to fail strict NPUKernelBench matched_ratio gates even though the kernel math is correct (57-D4: fp32 matched_ratio=0.0 with small systematic bias diff 0.0036-0.91, fp16 12/19)."
tags: [task.debugging, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/v351-exp-intrinsic-precision-001.md
    role: primary
created_at: "2026-08-26T12:40:00Z"
updated_at: "2026-08-26T12:40:00Z"
platforms: [a3, "950"]
aliases: ["v351-exp-intrinsic-precision-001"]
---

# bare AscendC::Exp() defaults to ExpAlgo::INTRINSIC whose input-dependent error pushes softmax past frozen-case rtol on V351 — pass ExpConfig{PRECISION_1ULP_FTZ_TRUE} explicitly

> 原 OKF v1 卡号：**v351-exp-intrinsic-precision-001**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - large-area precision FAIL whose CPU simulation against the frozen reference matches — an execution defect, not an algorithm error
> - kernel contains bare Exp( calls without an explicit ExpConfig
> - fp32 cases show small systematic bias with matched_ratio=0.0 while fp16/bf16 partially pass

## 现象 / 触发

57_ParallelPolarizedSelfAttention_evo（Ascend950DT + CANN 9.2.0，iter D4）实测：

1. 不显式给 ExpConfig 时 `AscendC::Exp` 走默认 `ExpAlgo::INTRINSIC`，误差是**输入相关**的，足以把 softmax 输出推过冻结 NPUKernelBench 用例的严格 matched_ratio 门槛（fp32 rtol=2^-13、fp16 rtol=2^-10）。
2. CPU 仿真核对该 kernel 的意图 fp32 数学与冻结 reference 一致（fp16 用例 matched ≥0.99），但真实 kernel 执行 fp32 全 matched_ratio=0.0（diff 0.0036-0.91 小幅系统性偏差）、fp16 仅 12/19 PASS——执行缺陷不是算法错。

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

- A3 的 100% 是在项目自有 5% 容差 quick_precision 下达成的，~0.4% 的误差一直在，只是被容差掩盖；冻结用例的严格 rtol 把它暴露出来。
- 修复：`constexpr AscendC::ExpConfig EXP_CONFIG = {AscendC::ExpAlgo::PRECISION_1ULP_FTZ_TRUE}`，softmax 处全部改 `AscendC::Exp<float, EXP_CONFIG>`。

## 动作规则

1. 迁移/新写代码里出现裸 `Exp(`（未显式 ExpConfig）一律补 `ExpConfig{PRECISION_1ULP_FTZ_TRUE}`——机械规则（trap_scan 规则 12 直接扫裸 Exp）。
2. 排查顺序：首次大面积错先 CPU 仿真对齐冻结 reference 分算法/执行；确认是执行缺陷后 near-miss 类再考虑精度开关，别一上来就调容差。

## 证据

- 57_ParallelPolarizedSelfAttention_evo failures_ledger.md 行 12（iter D4）。
