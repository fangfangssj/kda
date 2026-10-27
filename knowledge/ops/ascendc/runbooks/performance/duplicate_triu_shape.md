---
type: "Runbook"
title: "Duplicate 掩码向量化未修复 triu 大 shape 失败"
description: "triu 大 shape Case9 失败初判为标量 SetValue 掩码瓶颈，改 Duplicate 0 填充向量化后仍失败；后续 timing 定位实为 per-tile event flag 耗尽的正确性问题（非超时），Duplicate 向量化路线 banned。"
tags: ["task.debugging", "topic.implementation.masking", "topic.implementation.synchronization", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-18T07:28:54Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["mask_generation", "HardEvent", "event_flag"]
---

# Duplicate 掩码向量化未修复 triu 大 shape 失败

## 现象

在 Ascend950PR（arch35/DAV_3510，CANN 9.1.0）上以 Regbase 直调模式开发 `triu` 算子 Branch-1（掩码 predicate VF 路线）。首轮实现用标量 `SetValue` 逐元素写 0/1 生成掩码，B1 ST 35/36 通过，仅 Case 9（`[4,8,1024,1024]` float16，diagonal=0，out-of-place）FAILED——无 561003 错误码、无显式超时标记，仅标记 FAILED。

初判为标量 `SetValue` 掩码生成的性能瓶颈：由 shape 推算 dim0=33,554,432、8 核每核 16384 tile × 每 tile 256 次 SetValue ≈ 33M 次标量 S 管线操作，假设使该 case 执行过慢被判 FAILED。据此尝试用 `Duplicate` 向量填充向量化掩码生成。本卡记录该优化路线及其变体——全部失败，并记录为何失败。

## 适用场景

- **工程形态**：kernel 直调（Regbase，Branch-1 掩码 predicate VF 路线）。
- **API 路径**：`AscendC::SetValue`（标量写 UB）、`Reg::Duplicate`（向量寄存器填充）、`AscendC::SetFlag`/`WaitFlag`（per-tile event flag）、`AscendC::PipeBarrier<PIPE_ALL>`、`AscendC::CrossCoreSetFlag`/`CrossCoreWaitFlag`。
- **验证过的平台与 CANN 版本**：Ascend950PR / DAV_3510 / arch35，CANN 9.1.0。仅 950 有实测。
- **适用粒度**：`platform_wide`——DAV_3510 上任何用标量循环生成 0/1 掩码、且大 shape ST 出现"无错误码 FAILED"的 kernel。
- **路线优先级**：**banned**——下列向量化/跨核同步路线均已实测无效，禁止作为"修复"产出正卡。
- **不适用或未验证场景**：a2/a3 未实测，async/event 语义可能不同；已用 timing 证实 kernel 确超时的场景不属本卡（彼时 Duplicate 0 填充向量化仍是合理方向）。

## 失败路线

### 路线 A（banned）：Duplicate 0 填充 + SetValue 仅写 1

把每 tile "逐元素 SetValue 写 0/1" 改为 "`Duplicate(int8_t(0))` 向量填 0 + 仅对 keep=true 位置 SetValue 写 1"，期望把每 tile O(tileSize) 标量降为 O(1) 向量 + O(置1数) 标量。

实测 rebuild + rerun 后 Case 9 **仍 FAILED**（"Still failing"）。向量化未修复大 shape 失败。

### 路线 B（banned）：行级 Duplicate，弃用 SetValue

按行段用 `Duplicate` 同时填 0 与 1，彻底移除 SetValue，并移除 Compute 中的 S→V 同步。

实测 B1 由 35/36 **跌至 1/36**——行级 `Duplicate` 对子张量/`GetWithOffset` 的写入行为与 S→V 同步缺失叠加，大面积回归。已 revert。

### 路线 C（banned）：GM mask + 跨核同步

让 core 0 把掩码生成写入 GM workspace，其余核用 `DataCopy`（MTE2 DMA）加载掩码，跨核用 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 同步。

实测两次失败：kernel **hang**（core 0 也 WaitFlag 自设的 flag 致死锁，改为仅非 core-0 等待后仍 B1 跌至 7）；且 `CrossCoreSetFlag` 的 modeId 须取 [2,5]∪[10,10]，`SYNC_MODE0=0` 非法，同步语义不可靠。已 revert。

### 为何三条路线都失败（boundary）

三条例证的前提"Case 9 是标量吞吐超时"是**误判**。回退纯 SetValue 后用 `timeout 120 ./run.sh` 复测：Case 9 **在 120s 内跑完**（未超时）但**输出错误**——这是 correctness 问题，不是性能超时。标量 `SetValue` 总操作数 33M 只是数量级旁证，并非 FAILED 的因果；向量化掩码生成无法修复一个根子在 per-tile event flag 耗尽/async 同步的正确性缺陷。真正修复是 `PipeBarrier<PIPE_ALL>()` 替换 per-tile event flag（独立路线，见验证信号对照），与本卡 banned 的向量化路线无关。

## 验证信号

- Case 9（`[4,8,1024,1024]` float16）B1 ST FAILED，无错误码，其余 35 用例 PASS（来源平台：950；证据等级：tool_observed）
- 路线 A（Duplicate 0 填充 + SetValue-1）后重跑，Case 9 仍 FAILED（来源平台：950；证据等级：tool_observed）
- 路线 B（行级 Duplicate 弃用 SetValue）B1 由 35/36 跌至 1/36，已 revert（来源平台：950；证据等级：tool_observed）
- 路线 C（GM mask + CrossCoreSetFlag）kernel hang，修复后 B1 跌至 7，已 revert（来源平台：950；证据等级：tool_observed）
- 回退纯 SetValue 后 `timeout 120 ./run.sh`：Case 9 在 120s 内跑完但输出错误（correctness，非超时）（来源平台：950；证据等级：tool_observed）
- 对照（独立路线）：`PipeBarrier<PIPE_ALL>()` 替换 per-tile event flag 后 B1 36/36 PASS、B0 16/16、TilingUT 95/95 无回归（来源平台：950；证据等级：tool_observed）

## 失效边界

- 实测验证平台：仅 Ascend950PR（arch35/DAV_3510，CANN 9.1.0）。
- **未验证，推断可迁移**：a2/a3 的 event flag / `CrossCoreSetFlag` 语义可能不同，本卡 banned 结论仅对 arch35 成立。
- 本卡 banned 的三条向量化路线仅针对"FAILED 实为 correctness"的情形；若 timing 已证实 kernel 确超时（非本 trace 情形），Duplicate 0 填充向量化仍是合理优化方向。

## 坏实践

- 见"无错误码 FAILED + 大 shape"即归因标量 `SetValue` 吞吐瓶颈，不经 timing 复测就上向量化——本 trace 证实此归因为误判。
- "加个 Duplicate 0 填充试试"式补丁——当根因是 event 同步时，向量化掩码不改善，且每次 rebuild+run 成本高。
- 用 per-tile `SetFlag`/`WaitFlag` 处理 ~131k tile 的 V→V 依赖——event 资源在大 tile 数下耗尽，应改 `PipeBarrier<PIPE_ALL>`。

## 误诊

- 初判"33M 次标量 SetValue 使 Case 9 超时 FAILED"——`timeout 120 ./run.sh` 复测推翻：kernel 在 120s 内跑完、输出错误，属 correctness 非超时。标量操作数 33M 仅是数量级旁证，非因果。
- 把路线 A 的"仍 FAILED"误读为"向量化力度不够"而升级到路线 B/C——实为方向错误，三条例证均未触及 event flag 耗尽的真实根因。