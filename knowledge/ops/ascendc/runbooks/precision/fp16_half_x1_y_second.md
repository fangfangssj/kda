---
type: "Runbook"
title: "FP16 half 就地覆写 x1 破坏 y_second 读依赖"
description: "FP16/BF16 half 模式 ComputeFp16 把 xFp32 同时当输入与输出，Sub 就地覆写 x1(xFp32[0:halfD]) 为 y_first 后，后续 Mul 读 xFp32 实得 y_first 而非 x1，y_second=x2*cos+y_first*sin 错误；判别信号为 FP16 half max_diff 异常离群（前几元素正确）而 FP32 同路径 max_diff=0，D=8 触发向量核心异常。"
tags: ["task.debugging", "topic.precision", "topic.implementation.memory_management", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-18T07:59:53Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a2"]
---

# FP16 half 就地覆写 x1 破坏 y_second 读依赖

## 现象

在 cann-bench 框架下开发 `apply_rotary_pos_emb`（RoPE）算子，目标芯片 Ascend 910B3 / NpuArch DAV_2201（注册平台 `a2`）。FP16/BF16 输入需先 `Cast` 到 FP32 在 UB 内计算再 `Cast` 回半精度。`ComputeFp16` 的 half 模式（LLaMA 风格，`y[:D/2]=x1*cos-x2*sin; y[D/2:]=x2*cos+x1*sin`）首次跑测试时 FP32 half 精度完美（max_diff=0），但 FP16 half 出现大异常值（max_diff=17.8），前几个元素匹配良好、少数元素严重偏离；D=8（D/2=4）用例直接触发向量核心异常。`xFp32` 缓冲布局为 `[x1(0:halfD)] [pad(halfD:halfDAligned)] [x2(halfDAligned:...)]`，`halfDAligned = alignUp(D/2, 8)`（FP32 32 字节对齐=8 元素）。

## 适用场景

- **工程形态**：kernel 直调（cann-bench direct-launch，bisheng 编译 kernel + g++ 编译 plugin），算子族=FusedComposite/RoPE。
- **API 路径**：`AscendC::Mul/Sub/Add`（UB 内 LocalTensor 逐元素，`count=halfD`）、`AscendC::Cast`（`CAST_NONE` 入 / `CAST_RINT` 出）、`AscendC::DataCopyPad`（GM↔UB）。
- **验证过的平台与 CANN 版本**：Ascend 910B3 / SocVersion ASCEND910B / NpuArch DAV_2201 / `--npu-arch dav-2201`；CANN 9.0.0；bisheng compiler。
- **适用粒度**：`same_operator`（RoPE half 模式 FP16/BF16 升精度计算路径）；就地缓冲复用模式可外推到 `operator_family`（任何把输入缓冲同时当输出缓冲、且后续还要读原输入的逐元素融合算子）。
- **路线优先级**：`recommended_fastpath`（先存 x1 的两个积到 tmp 再覆写 x1，是 UB 内多算子融合就地复用缓冲的正确范式）。
- **明确不适用/未验证场景**：FP32 直算路径（无 Cast、无 xFp32 缓冲）不受此问题影响；interleaved 模式走 Gather 到独立 xEven/xOdd 缓冲、不存在就地覆写；非 A2 平台未验证（见 `## 失效边界`）。

## 判别方法

**触发**：FP16/BF16 half 模式 `ComputeFp16` 跑出 max_diff 异常离群（实测 17.8），前几个元素匹配、少数元素严重偏离；D=8（D/2=4）用例直接触发向量核心异常。同一 kernel 的 FP32 half 路径 max_diff=0（控制路径正常）。

**探针**：
1. 对比 FP32 half 与 FP16 half 的 max_diff——若 FP32=0 而 FP16 异常离群，排除 Cast 舍入与 FP16 值域溢出（17.8 在 FP16 范围内），指向 half 模式升精度计算路径内的读-写顺序问题而非全局数值问题。
2. 检查 `ComputeFp16` half 分支内 `xFp32` 是否同时作为输入与输出缓冲（in-place 复用），逐元素画读-写依赖图：确认每个原输入元素（x1=`xFp32[0:halfD]`）在被覆写前，所有依赖它的积是否已落盘到 tmp。
3. D=8 用例单独验证 `halfDAligned` 对齐：`halfD=4` 是否被 `alignUp(., 8)` 抬到 8（FP32 32 字节=8 元素对齐）；若 `count` 误用 `halfDAligned`（对齐值）会读越界 padding 触发异常。

**原始观测**：
- FP16 half max_diff=17.8（前几元素正确、少数严重偏离）。
- FP32 half 同路径 max_diff=0。
- D=8（D/2=4）用例触发向量核心异常（`halfD=4` 不满足 FP32 8 元素对齐）。
- 重排读-写顺序（先存 x1 的两积再覆写 x1）后 FP32 max_diff=0 / FP16 max_diff=1.5e-2 / BF16 max_diff=6.25e-2，13 用例全 PASS——确认归因为读-写顺序，而非其他。

**解释**（interpretation，属 agent 根因分析，非独立硬件验证）：`xFp32` 布局为 `[x1(0:halfD)] [pad] [x2(halfDAligned:...)]`，half 模式把 `xFp32` 同时当输入与输出。原始顺序 `Sub(xFp32, tmp1, tmp2)` 把 `xFp32[0:halfD]` 就地覆写为 `y_first = x1*cos - x2*sin`，破坏了 x1；随后 `Mul(tmp2, xFp32, sinFp32)` 本应读 x1 算 `x1*sin`，实际读到第 3 步写入的 `y_first`，于是 `y_second = x2*cos + y_first*sin`（错值），偏差可达输入量级（17.8）。次要因素：UB 内 LocalTensor 之间不应使用 `DataCopy`（`DataCopy` 是 GM↔UB 通路，UB↔UB 应直接原地计算或用 `DataCopyPad` 的 UB 变体）；`halfDAligned` 必须把 `halfD` 对齐到 8，否则 `xFp32[halfDAligned]` 偏移越界触发核心异常。判别关键：FP32 同路径无误 → 排除 Cast；max_diff 离群且前几元素正确 → 指向特定元素的读-写顺序而非全局舍入；重排后即恢复 → 确认单点读-写依赖破坏。

> 注：读-写依赖破坏的机制层归因由代码顺序分析与"重排后实测恢复"共同支撑，属 agent 推断；未在 trace 中用逐元素 dump 独立验证 y_second 读到的就是 y_first。

## 验证信号

- FP16 half max_diff=17.8 异常离群（前几元素正确、少数严重偏离）（来源平台：a2；证据等级：tool_observed）
- FP32 half 同路径 max_diff=0（控制路径正常，排除 Cast 舍入/值域溢出）（来源平台：a2；证据等级：tool_observed）
- D=8（D/2=4）用例触发向量核心异常（来源平台：a2；证据等级：tool_observed）
- 重排读-写顺序后 FP32 max_diff=0 / FP16 max_diff=1.5e-2 / BF16 max_diff=6.25e-2，13 用例全 PASS（来源平台：a2；证据等级：tool_observed）

## 失效边界

- 实测验证平台：仅 A2（Ascend 910B3 / DAV_2201，CANN 9.0.0）上 FP32/FP16/BF16 half 模式。
- **未验证，推断可迁移**：A3 (950) / Ascend950 平台，FP32 8 元素对齐与 `Mul/Sub/Add`/`Cast` 语义在同架构族一致，本判别规则应适用；但 `halfDAligned` 的对齐粒子须按目标平台 FP32 对齐要求复核（A3 若 UB 对齐要求不同需调整 `alignElements`）。
- interleaved 模式不适用本卡（走 Gather 到独立缓冲，无就地覆写）。
- 本卡仅记录诊断规则（触发→探针→观测→解释），不提供 verbatim before/after 修复代码。

## 坏实践

- 把输入 LocalTensor 同时当输出缓冲的融合算子，不画"读-写依赖图"就安排计算顺序——任何在被覆写前其依赖积未落盘的原输入元素都会读到错值。
- half 模式升精度路径用 `Sub(输入, ...)` 这类就地覆写先于后续读原输入的写法。
- `count` 用 `halfDAligned`（对齐值）而非 `halfD`（有效值），会读越界 padding 区触发异常。
- 用 `AscendC::DataCopy` 在 UB 内部 LocalTensor 间复制（`DataCopy` 是 GM↔UB 通路）。

## 误诊

- 曾怀疑 FP16 值域溢出（max≈65504，乘法溢出）——不成立，17.8 在 FP16 范围内；真正原因是就地覆写后读到 `y_first` 而非 `x1`（agent 推断，已由代码顺序分析与重排后实测恢复共同推翻）。
- 曾试图用 `DataCopy` 在 UB 内把 yFirst 复制到 xFp32——API 误用，未解决问题（已弃用）。
