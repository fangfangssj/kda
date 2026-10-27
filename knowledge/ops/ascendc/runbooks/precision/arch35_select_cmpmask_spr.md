---
type: "Runbook"
title: "arch35 Select cmpmask SPR 跨调用泄漏致非确定失败"
description: "arch35 Select(VSEL_CMPMASK_SPR) cmpmask SPR 跨调用残留致分支错乱；case 单独 PASS、套件内 FAIL 为判别信号。"
tags: ["task.debugging", "topic.precision", "topic.implementation.masking", "topic.compute.elementwise.comparison_logic"]
status: stable
sources: []
created_at: "2026-08-18T15:23:28Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# arch35 Select cmpmask SPR 跨调用泄漏致非确定失败

## 现象

在 Ascend950PR（arch35 / DAV_3510）RegBase 路线上调试条件分支类算子精度时，kernel 用 `Compare` 生成 mask、`Select(VSEL_CMPMASK_SPR)` 在线性/sigmoid 分支间选择。C++ ST 全过，但 PyTorch ST 套件内部分 case 失败，且失败**非确定**：同一 case 单独跑 PASS、在套件内 FAIL。典型 fail case 的 `Actual = 0.5 × Golden`，即 kernel 误走 sigmoid 分支并算出 `sigmoid(0) = 0.5`。

kernel 调用形态：
```cpp
AscendC::Compare<float, uint8_t>(bMask, bBx, bThr, CMPMODE::GT, count_padded);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(bBx, bMask, bG, bThr,
                                SELMODE::VSEL_CMPMASK_SPR, count_padded);
```

## 适用场景

- **工程形态**：kernel 直调（RegBase 路线，arch35/DAV_3510 async SIMT 向量 API）。
- **API 路径**：`AscendC::Compare<float,uint8_t>` + `AscendC::Select<float,uint8_t>(...,SELMODE::VSEL_CMPMASK_SPR,...)`；同步原语 `AscendC::PipeBarrier<PIPE_V>`。
- **验证过的平台与 CANN 版本**：Ascend950PR（`--soc=ascend950`），CANN 9.0.0-beta.2。
- **适用粒度**：`operator_family` —— 任何在 arch35 上用 `Select(VSEL_CMPMASK_SPR)` 做条件分支的 RegBase kernel。
- **路线优先级**：`banned` —— `VSEL_CMPMASK_SPR` 路径有跨调用 cmpmask SPR 泄漏风险，不应依赖其序列化语义做条件分支；应改用纯算术 blend 或张量路径 Select。
- **不适用或未验证场景**：非 arch35（如 a2/a3）的 Select 语义未验证，推断可迁移但未实测（见 `## 失效边界`）。

## 判别方法

**触发**：kernel 用 `Compare` + `Select(VSEL_CMPMASK_SPR)` 做条件分支，PyTorch ST 套件内部分 case 失败，且同一 case 单独跑 PASS。典型表现为 `Actual = 0.5 × Golden`（Select 误选 sigmoid 分支）。

**探针**：
1. 将失败的 case 单独跑（fresh process）——若 PASS，则指向跨调用状态泄漏而非逐 case 数值逻辑错。
2. 分别跑 `--level L1` 单独与 `L0+L1` 全量套件——若失败数不同，差值即为被前序调用污染后新失败的 case 数，确证跨调用泄漏。

**原始观测**：
- 套件内 baseline：578 用例，116 失败。
- 典型 fail case（套件内）：`Golden=0.681152 Actual=0.340332 mismatch=16632/1204224`（`0.340332 ≈ 0.5 × 0.681152`，sigmoid 分支被误选）。
- 同一 case 单独跑（fresh process）：`golden=0.500000 npu=0.500000 OK`。
- `--level L1` 单独：108 失败；`L0+L1` 全量：116 失败（差 8 例 = L0 前置调用污染致额外失败）。

**解释**（interpretation，属 agent 根因分析，非独立硬件验证）：arch35 向量算子底层为 async SIMT。`Select` 的 `VSEL_CMPMASK_SPR` 模式内部实现为 `set_cmpmask(sel) → PipeBarrier<PIPE_V> → vsel`，从 cmpmask SPR（特殊寄存器）读分支掩码。`PipeBarrier<PIPE_V>` 能同步 V→V 张量依赖（trace 中 L0 的 `Maxs/Mins` clamp 正常工作即证明），但**不能可靠序列化 cmpmask SPR 的写-读**：`vsel` 可能在 `set_cmpmask` 落盘前执行，读到**上一轮 kernel 调用残留的 cmpmask**，导致分支掩码错乱。L0 因 threshold 恒定、残留 cmpmask 恰好匹配而幸免；L1 的 threshold 横跨正负大小 → 分支选错。判别关键：失败用例单独跑 PASS、被 L0 前置后 FAIL，是典型跨调用状态泄漏，非逐 case 数值逻辑错误。

> 注：cmpmask SPR 写-读序列化失败的硬件机制未独立测量（未用 DumpTensor 在 set_cmpmask 后/vsel 前导出 SPR 值），属 agent 推断。

## 验证信号

- PyTorch ST baseline：578 用例，116 失败（来源平台：950；证据等级：tool_observed）
- 典型 fail case 套件内：Golden=0.681152 Actual=0.340332 mismatch=16632/1204224（来源平台：950；证据等级：tool_observed）
- 同一 case 单独跑（fresh process）：golden=0.500000 npu=0.500000 OK（来源平台：950；证据等级：tool_observed）
- `--level L1` 单独 108 失败 vs `L0+L1` 全量 116 失败（差 8 例 = L0 污染）（来源平台：950；证据等级：tool_observed）
- arch35 Select(VSEL_CMPMASK_SPR) 内部实现含 set_cmpmask → PipeBarrier<PIPE_V> → vsel 序列（来源平台：950；证据等级：artifact_verified）

## 失效边界

- 实测验证平台：仅 Ascend950PR（arch35/DAV_3510，CANN 9.0.0-beta.2）。
- **未验证，推断可迁移**：a2/a3 上若同样用 `Select(VSEL_CMPMASK_SPR)`，cmpmask SPR 的 async 语义可能不同（a2/a3 非 arch35 SIMT），需单独实测后方可采用。
- 本卡仅记录诊断规则（触发→探针→观测→解释），不提供 verbatim before/after 修复代码——cmpmask SPR 写-读序列化失败的硬件机制未独立验证，属 agent 推断。

## 坏实践

- 用"单独跑某个 case PASS"判定 kernel 正确——会漏掉跨调用状态泄漏类 bug（必须跑完整套件 + 单独 case 对比）。
- 见 `Select(VSEL_CMPMASK_SPR)` 注释只写"arch35 仅支持 VSEL_CMPMASK_SPR 模式"而未点明其跨调用泄漏风险，下个 agent 易直接照抄。
- 依赖 `PipeBarrier<PIPE_V>` 序列化 cmpmask SPR——该 barrier 同步 V→V 张量依赖可靠，但对 SPR 写-读不可靠。

## 误诊

- 早期把失败归因于"threshold 为负/极小值时 `Mins(bx, threshold)` clamp 逻辑错"——逐 case 数值推演后被证伪（clamp 逻辑对 mask=false 段正确）。
- 早期怀疑"threshold_ 标量传参错（被截断/符号翻转）"——tiling log 与 `OpDef` 核对后排错。
- 早期怀疑"`Duplicate(bThr, threshold_)` async 未完成致 `Compare` 读陈旧 bThr"——这是真实隐患但非本卡主因（即便 bThr 正确，cmpmask SPR 仍泄漏）。