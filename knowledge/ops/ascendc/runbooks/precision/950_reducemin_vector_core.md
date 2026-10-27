---
type: "Runbook"
title: "950 循环内 ReduceMin 取索引致 vector core 异常"
description: "950(arch3510) 上 for 循环内反复调用 ReduceMin(calIndex=true)+reinterpret_cast 取最小值索引触发非确定 vector core exception(error 264/retCode 0x31)，崩溃率随循环次数升高(numSeqs>=12 全崩)；所有 Vector 路线修复(buffer 增大/WholeReduceMin/PipeBarrier/改同步)的 vec error info 全同 0x41003810003106ea，换 Scalar 循环 120/120 PASS。判据：vec error info 跨所有 Vector 修复一致 + numSeqs 相关 + Scalar 绕开即愈 -> Reg::ReduceMin 循环调用级问题，非软件层。"
tags: ["task.debugging", "topic.precision", "topic.compute.reduction", "topic.programming_model.execution_unit.vector"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# 950 循环内 ReduceMin 取索引致 vector core 异常

## 适用场景

- 工程形态：kernel 内 reduce-with-index 取最小值位置（如 `advance_step` 的 `ComputeInputTokens` 在 `for` 循环内逐行求"第一个 -1 的位置"）。
- API 路径：`ReduceMin(dst, src, workBuf, count, calIndex=true)`（Level2）；`WholeReduceMin`（Level1，底层同 `Reg::ReduceMin`）；`SetWaitFlag<HardEvent::*>`；`LocalTensor::GetValue/SetValue`。
- 验证过的平台与 CANN 版本：Ascend950（arch 3510）/ CANN 9.0.0。
- 适用粒度：`platform_wide`（任何在 950 上循环调用 `ReduceMin(calIndex=true)` 取索引的算子都会踩；单次调用如 ifmr 不受影响）。
- 路线优先级：`recommended_fastpath`（循环内取最小值索引改成纯 Scalar `GetValue` 循环；只在单次、大批量元素时才用 `ReduceMin(calIndex=true)`）。
- 明确不适用或未验证场景：A2/A3 上原 `ReduceMin(calIndex=true)` + `reinterpret_cast` 写法正常工作（910B 用 `SetMask` 限制读取范围，不越界），不需要换 Scalar 循环（未验证，推断 A2/A3 仍可用原写法）；950 单次调用（循环外）未观测到崩溃。

## 判别方法

触发（tool_observed）：在 950（arch 3510）上，kernel 的 `for` 循环内反复调用 `ReduceMin(dst, src, workBuf, count, calIndex=true)` + `reinterpret_cast<uint32_t*>(&tempIndex)` 取最小值索引时，plog 报 `AIV Kernel happen error, retCode=0x31`（vector core exception），对应 error code 264、subErrType 0x4（D-cache 读写 UB、总线返回非零）。同一 V2 用例、同一数据跑 10 次出现 3 PASS/7 FAIL 的非确定崩溃；`numSeqs` 越大（`ComputeInputTokens` 的 `for` 循环迭代越多、`ReduceMin` 调用越频繁）崩溃率越高。

探针（executable check）：

1. 对同一 V2 用例连跑 10 次，统计 PASS/FAIL 比例并取 plog 的 vec error info。
2. 扫 `numSeqs` ∈ {2, 4, 6, 8, 12, 24}，记录各档通过率。
3. 逐一尝试 Vector 路线修复并每次比对崩溃的 vec error info 是否一致：把 `sampledTokenIdsBuf_` 增大到 256B/512B；改用 `WholeReduceMin`（Level1，底层同 `Reg::ReduceMin`）；加 `PipeBarrier<PIPE_ALL>` 强同步；改同步 `MTE2_S`/`MTE2_V`。同时用 `maxCores=2` 限制核数看通过率是否变化。
4. 把循环内 `ReduceMin(calIndex=true)`+`reinterpret_cast` 替换为纯 Scalar `GetValue`+比较循环（遇第一个 -1 即 break 取前一个元素，若第一个即 -1 则取 -1），跑全量 V2。
5. 对照单次调用场景（如 ifmr 每核循环外只调 1 次 `ReduceMin(calIndex=true)`），看是否崩溃。

> 注：before/after 逐字代码未从 trace 恢复（edit/file_write 事件被 redact 为 `_see below_`），上述探针为 agent 在轨迹中叙述的实验步骤，落库前应回原 `op_kernel/advance_step_spec.h` 比对最终版。

观测（tool_observed）：

- 同 V2 用例 10x → 3 PASS/7 FAIL 非确定；崩溃与不崩溃交替无规律。
- `numSeqs` 通过率：2-4 约 66%，6-8 约 36%，`>=12` 全 0% PASS，`numSeqs=24` 全 0% PASS——崩溃率随循环迭代数单调上升。
- 所有 Vector 路线修复（buffer 256B/512B、`WholeReduceMin`、`PipeBarrier<PIPE_ALL>`、`MTE2_S`/`MTE2_V`）崩溃 error code 均为 264，vec error info 完全相同 `0x41003810003106ea`，指向同一条硬件指令；`maxCores=2` 限制核数后通过率 42/120 与无限制 40/120 相近，排除多核争抢。
- Scalar 循环替代 → 120/120 V2 PASS，全量 250/250（V1 130/130 + V2 120/120）PASS；对照 ifmr（每核 1 次调用）950 不崩溃。
- 原始 spec.h + V2 单独跑 3/5 PASS（ReduceMin 崩溃）；原始 spec.h + V1+V2 混合跑 V1 也 FAIL（V2 JIT 编译崩溃污染 `kernel_meta` 缓存）；修改后 spec.h + V1+V2 混合跑 10/10 PASS。

判别与解释（interpretation，agent 根因分析，非 API 规范主张）：四条 tool_observed 事实构成判据——(1) 所有 Vector 路线修复产生**完全相同**的 vec error info `0x41003810003106ea`，证明崩溃点是同一条 `Reg::ReduceMin` 硬件指令，而非 buffer 布局、同步或软件逻辑；(2) 崩溃**非确定**且通过率随 `numSeqs`（循环调用次数）单调下降，`>=12` 必现，单次调用（ifmr）不崩——触发条件是"循环内多次调用"，不是单次调用；(3) `maxCores=2` 与无限制通过率相近，排除多核竞争；(4) Scalar `GetValue` 循环绕开 `Reg::ReduceMin` 指令即 120/120 PASS，反向证实问题在该指令本身。综合判为：950 上 `Reg::ReduceMin` 指令在循环调用场景下存在硬件/微码级问题，非软件层。该结论基于排除法，确切微架构机制（pipeline hazard / 内部状态机竞争 / 指令缓存交互）无官方文档说明。曾提出但被证伪的假设：① 950 `ReduceIndexTemplate` 用 `Reg::LoadAlign` 读固定 256B 向量寄存器、buffer <256B 越界（seq 1480）——被证伪，buffer 增到 256B/512B 仍崩；② `ReduceMin` 索引格式在 950 与 A2/A3 不同致 `minIndex-1` 越界——被证伪，崩溃发生在 `GetValue` 之前的 `ReduceMin` 指令本身；③ 流水线同步不足——被证伪，`MTE2_S`/`MTE2_V`/`PipeBarrier<PIPE_ALL>` 均无效。

## 验证信号

来源：本 trace 在 Ascend950（arch 3510，CANN 9.0.0）上对 `advance_step` V2 用例做 10x 复跑、`numSeqs` 扫描、Vector 路线修复 A/B 与 Scalar 循环替代验证。

- 同 V2 用例 10x → 3 PASS/7 FAIL 非确定，plog `retCode=0x31`/error 264/vec error info `0x41003810003106ea`。（来源平台：950；证据等级：tool_observed）
- `numSeqs` 通过率 2-4 约 66%、6-8 约 36%、`>=12` 全 0%、24 全 0%——崩溃率随循环调用次数单调上升。（来源平台：950；证据等级：tool_observed）
- buffer 增到 256B/512B、`WholeReduceMin`、`PipeBarrier<PIPE_ALL>`、`MTE2_S`/`MTE2_V` 均 1-3/10 PASS 无效，vec error info 全同 `0x41003810003106ea`；`maxCores=2` 限制核数 42/120 与无限制 40/120 相近，排除多核争抢。（来源平台：950；证据等级：tool_observed）
- Scalar 循环替代 `ReduceMin(calIndex=true)` → 120/120 V2 PASS，全量 250/250（V1 130/130 + V2 120/120）PASS。（来源平台：950；证据等级：tool_observed）
- 原始 spec.h + V2 单独 3/5 PASS（ReduceMin 崩溃），原始 + V1+V2 混合 V1 也 FAIL（JIT 缓存污染），修改后 spec.h + V1+V2 混合 10/10 PASS。（来源平台：950；证据等级：tool_observed）

## 失效边界

- A2/A3：原 `ReduceMin(calIndex=true)` + `reinterpret_cast` 写法正常工作（910B 用 `SetMask` 限制读取范围不越界），本卡判据与 Scalar 替代路线均不适用（未验证，推断 A2/A3 仍可用原写法）。
- 950 单次调用 `ReduceMin(calIndex=true)`（循环外）：本卡不适用，单次调用未观测到崩溃（ifmr 每核 1 次佐证）。
- 未来若 CANN 修复 `Reg::ReduceMin` 在 950 循环调用的硬件/微码问题，可回退到 Vector 路线；届时本卡判据失效。

## 坏实践

- 在 `for` 循环内反复调用 `ReduceMin`/`ReduceMax` 等 Reduce 系列 API 的 `calIndex=true` 模式取索引，假设不同架构行为一致——950 上触发非确定 vector core 异常。该路线标记为 `banned`。
- 把 `ReduceMin` 索引的 float 内存用 `reinterpret_cast<uint32_t*>` 读为 uint32 是 910B 惯用法，950 上即使 `ReduceMin` 不崩，索引格式也可能与 910B 不同，避免依赖该 reinterpret 惯用法。