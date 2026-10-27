---
type: "Runbook"
title: GEIR 图模式 int dtype 用例静默逐位失配
description: 自定义算子 GEIR 图模式下 int32/int64 dtype 用例不报错但输出逐位错值（bitwise mismatch），同批浮点用例报 EZ9999、ACLNN 回归通过；结果比对明细判读与"静默错值"判别特征（含 int64 输出 32/64 位拼装位宽指纹分解法）。
tags: [task.debugging, topic.precision, topic.graph_execution, topic.compute.indexing]
status: stable
sources: []
created_at: '2026-09-14T12:09:22Z'
updated_at: '2026-09-14T12:09:22Z'
platforms: ["950"]
---

# GEIR 图模式 int dtype 用例静默逐位失配

## 现象

在 Ascend950PR（CANN 9.2.0）上验收自定义算子 BenchHarnessScatter（scatter / scatter-reduce，`reduce=update` 覆盖模式为主）的 GEIR 图模式时，TTK GEIR 静态用例 8 个全部失败，但失败形态分成两类：6 个浮点 dtype 用例（f16/f32/bf16 组合）直接 `EXEC_FAILURE`（RunGraph 报 EZ9999，见同批《GE 图路径 TBE 动态编译 blockDim 固化致 EZ9999》卡）；而 2 个整数 dtype 用例**不崩溃、正常跑完并产出输出，但输出逐位错误**：

- `static_i32_i32`（data/updates 为 int32，indices 为 int32，形状 (8,4)/(3,4)/(3,4)，dim=0，输入范围 (-5,5)/(0,7)/(-5,5)，比较标准 binary_equal）：精度 **59.375%**，`reason: 'bitwise mismatch'`，差异 13 处。比对明细（原文节选）：

```
Index: 000 RealIndex: 000000 Expected: 5.000000000000000000 Actual: 0.000000000000000000 Diff: -1.000000
Index: 002 RealIndex: 000003 Expected: 2.000000000000000000 Actual: -3.000000000000000000 Diff: -2.500000
Index: 005 RealIndex: 000006 Expected: 0.000000000000000000 Actual: 4.000000000000000000 Diff: inf
Index: 010 RealIndex: 000016 Expected: -1.000000000000000000 Actual: 4.000000000000000000 Diff: -5.000000
Index: 012 RealIndex: 000029 Expected: -1.000000000000000000 Actual: 0.000000000000000000 Diff: -1.000000
```

- `static_i64_i32`（data/updates 为 int64，indices 为 int32）：精度 **68.75%**，bitwise mismatch 10 处。其中一处出现明显的位宽指纹（原文）：

```
Index: 007 RealIndex: 000020 Expected: 5.000000000000000000 Actual: 21474836485.000000000000000000 Diff: 4294967296.000000
```

`21474836485 = 5 + 5×2^32`，即该 64 位输出的高 32 位与低 32 位各装了一个 5——与"两个 32 位值被拼进一个 64 位槽位"的元素位宽处理错误一致（此为对原始数值的算术分解，属未验证解释）。

当时的环境状态：同一隔离 staging、同一批次测试中，InferShape/InferDataType 10/10 PASS、Reject 1/1 PASS、ACLNN 回归（overwrite/add）PASSED；GEIR 路径下发的 kernel 是 TBE 动态编译产物（`te_benchharnessscatter_911e...`，blockDim=6），而非包内预编译二进制。

## 适用场景（必填 — 检索前的第一道滤网）

- 工程形态：自定义算子 GEIR 图模式（TTK GEIR / route 探针）执行后的精度结果判读；int dtype 输出算子的图模式验收。
- API 路径：TTK GEIR 测试（static.csv/dynamic.csv → static-result.csv）、binary_equal 逐位比较、TBE 动态编译执行路径（与同批 EZ9999 卡共用的回退机制）。
- 验证过的平台与 CANN 版本：Ascend950PR（Ascend950PR_9579），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：`operator_family`——凡 int dtype 输出、依赖索引散布/搬运的算子在 GEIR 动态编译路径下的结果判读（判别特征可迁移；具体失配模式随算子而异）。
- 路线优先级：`recommended_fastpath`（作为结果判读规则使用：先分类失败形态，再决定排查方向）。
- 明确不适用 / 未验证场景：
  - aclnn 直调路径（本会话实测通过）。
  - 浮点 dtype 用例（它们表现为 EZ9999 EXEC_FAILURE，不是本卡形态）。
  - 未走 TBE 动态编译回退（即二进制匹配命中）的 GEIR 执行——本卡形态仅在回退路径上观测到。

## 判别方法

1. **失败形态分类**（结果 CSV 的 `precision` / `precision_status` 列）：
   - `EXEC_FAILURE` + `ERROR: GEIR execution failed (rc=255): ... EZ9999` → kernel 执行崩溃（见同批 EZ9999 卡）。
   - `NN.NN%` + `FAIL` + `bitwise mismatch` → kernel 跑完但值错，即本卡形态。
   关键判别点：**int dtype 用例没有任何运行错误信号**，若只看退出码/错误日志会误判"整数用例正常"，必须看逐位比对结果。
2. **位宽指纹检查**：对 int64 输出的 Actual 值做分解——形如 `v = a + b×2^32`（高/低 32 位各含一个合理数据值）提示 32/64 位元素宽度在搬运或索引换算中被混淆。本次 `21474836485 = 5 + 5×2^32`。
3. **错值模式检查**：比对明细中 Expected 为"应被 update 覆盖后的值"而 Actual 为 0 或其他原始数据值的位置（如 RealIndex 000000 Expected 5 / Actual 0），提示散布目标位置的计算或索引换算错误，而非数值精度漂移——int 运算不存在舍入，bitwise mismatch 只能来自位置/位宽/搬运错误。
4. **路径对照**：同算子 aclnn 回归通过 + GEIR 浮点用例 EZ9999 → 失败集中在 GE 图模式动态编译路径，先按同批 EZ9999 卡的判别链确认是否走了动态编译回退。

## 根因

未追到。本会话为验收会话（禁止修改 kernel/tiling），只完成了现象取证与路径归属：失配只出现在 GEIR 图模式（动态编译 kernel `te_benchharnessscatter_911e...`）下，同算子 aclnn 回归通过；int dtype 用例不崩溃但逐位错值，i64 用例出现 32/64 位拼装的位宽指纹。int 运算无舍入，逐位失配的候选机制是索引/偏移换算或元素位宽在动态编译 kernel 的搬运路径中被错误处理，但确切机制（以及为何浮点用例崩溃而整数用例静默出错）未验证。同组另一会话对该形态的候选归因（多核拷贝写与单核散布读的跨核 GM 可见性/时序）亦未经实验证实，仅作线索。

## 设计决策

- 本卡按"结果判读规则"组织而非修复配方：验收语境下最有复用价值的是失败形态分类（崩溃 vs 静默错值）与位宽指纹检查，它们决定下一步排查方向（EZ9999 卡的判别链 vs kernel 索引/位宽逻辑）。
- 刻意不把"动态编译路径导致"写成确定根因：该归属对浮点用例有完整证据链（见同批 EZ9999 卡），对整数用例只是同批次相关性推断。

## 系统信息

- 环境：CANN 9.2.0，设备 Ascend950PR（Ascend950PR_9579），TTK 3.0.0（Mode: ASCEND_ONBOARD）。
- 隔离 staging：`/tmp/geir_opp.*/vendors/BenchHarnessScatter`。
- 相关文件：`tests/geir/build/ttk/static-result.csv`（比对明细来源）、`tests/geir/ttk/static.csv`（用例定义：dtype/shape/attr/精度容差 1e-08）。

## 预防

- 图模式验收不要只看执行是否报错：int dtype 用例可能静默产出错值，必须强制逐位比对（binary_equal）并把精度阈值当硬门槛。
- 对 int64 输出算子，结果检查时顺手做一次 `v mod 2^32` / `v div 2^32` 分解，可快速识别位宽拼装类错误。

## 验证信号

来源说明：本卡信号来自一次 BenchHarnessScatter GEIR 图模式实现验收会话（TTK GEIR 静态结果 CSV，逐条原文取证）；同组其他会话在同算子 GEIR 图路径上独立观测到同类整数用例 bitwise mismatch（精度数值略有差异），形态一致复现。

- `static_i32_i32`：精度 59.375%，binary_equal bitwise mismatch 13 处，明细含 Expected 5/Actual 0（RealIndex 000000）等（来源平台：950；证据等级：tool_observed）。
- `static_i64_i32`：精度 68.75%，bitwise mismatch 10 处，含 Expected 5 / Actual 21474836485（= 5 + 5×2^32）（来源平台：950；证据等级：tool_observed）。
- 同批对照组：6 个浮点 dtype 用例全部 EZ9999 EXEC_FAILURE；ACLNN 回归 overwrite=PASSED / add=PASSED（来源平台：950；证据等级：tool_observed）。
- 动态用例（dynamic-result.csv）因整批 600s 超时未获得结果，本卡不覆盖动态 shape 场景（来源平台：950；证据等级：tool_observed）。
- 跨会话复现：同算子 GEIR 图路径另一验收会话观测 `static_i32_i32` 59.375%（13 处差异，如 `RealIndex: 000001 Expected: -1.0 Actual: 0.0`）、`static_i64_i32` 53.125%（15 处差异）——形态一致（整数用例不崩、逐位错值），精度百分比随用例版本略有差异（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅 Ascend950PR + CANN 9.2.0 + 单算子（BenchHarnessScatter，update 模式为主）实测；其他平台/版本未验证，判别方法（形态分类 + 位宽分解）推断可迁移但未验证。
- 位宽指纹（`v = a + b×2^32`）只是线索不是证明：本会话只有 1 个样本点，未验证其成因。
- 本卡不覆盖 add/amultiply/amin/amax 等 reduce 模式与动态 shape 用例（未执行完成）。
- 失配的具体 kernel 机制未定位，不能据此直接给出代码级修复。

**未决项**：

1. 【优先】复现并追根因到机制层：定位动态编译 kernel 中 int32/int64 用例的索引/偏移/位宽处理错误（对照 DSL impl 与预编译 AscendC kernel 的差异），并解释"浮点崩溃、整数静默错值"的分化。
2. 位宽指纹 `v = a + b×2^32` 的成因验证（构造最小 int64 scatter 用例复现）。
3. 动态 shape 用例在该路径下的行为（本会话超时未取得结果）。
## 坏实践

- 只依据"用例执行完成、无报错"判定图模式整数用例通过——本会话中这两个用例正是以静默错值形态失败，若无逐位比对会被漏放。
- 把整数 bitwise mismatch 当"精度容差问题"调 tolerance 处理：int 运算无舍入，容差放松只会掩盖位置/位宽错误（本会话比对标准为 binary_equal，未放松）。

## 误诊

- "整数用例没报 EZ9999，所以图模式对整数 dtype 是正常的"——错误：不崩溃不等于结果正确，两个整数用例均逐位失配。
- "i64 Actual=21474836485 是数据越界读到了随机大数"——更可能是位宽拼装（5+5×2^32 的结构化值），但两者均未验证，此处仅作判读提示。
