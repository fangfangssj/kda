---
type: "Runbook"
title: "fp32-vs-double 机器精度差判真实超标 — 放宽 epsilon 属 banned 掩盖"
description: "fp32 kernel 与 double golden 的机器精度级差（|a-g|≈1.19e-7）在近零输出上 MARE 超阈时，按官方小值域标准（FLOAT32 SmallValueThreshold 2^-14、Error Threshold 2^-30）对照：|a-g|≫9.3e-10 为真实超标而非指标病——放宽 epsilon（1e-7→1e-4）恰使最差用例压线通过属 banned 掩盖。三态判据：恰压线通过 + kernel 侧手段未穷尽 + 官方标准真实超标。kernel 侧死路：AI Core 禁 double、零填充单次归约更差；正解=Kahan/成对求和/小值子集 ErrorCount/如实声明精度边界。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-20T11:36:00Z'
updated_at: '2026-09-20T11:36:00Z'
platforms: ["950"]
---

# fp32-vs-double 机器精度差判真实超标 — 放宽 epsilon 属 banned 掩盖

## 现象

3D 自适应平均池化 fp32 kernel（ReduceSum float → partialSum float 跨批累加 → float 除法）与 double CPU golden 的机器精度级差使近零输出 MARE 超阈（L0_003：MERE=2.12e-07 / MARE=1.57e-03，阈值 1.22e-04/1.22e-03；失配项打印 6 位小数相同、|a−g|≈1.19e-7 = fp32 机器精度级；输出元素越多出现极小 golden 的概率越大）。

## 适用场景

- **工程形态**：fp32 归约类算子（avg pool / reduce）C++ ST 验收，double CPU golden。
- **API 路径**：MERE/MARE 双阈值比较器；官方小值域标准（FLOAT32 SmallValueThreshold 2^-14≈6.1e-5、Error Threshold 2^-30≈9.3e-10）。
- **验证过的平台与 CANN 版本**：Ascend950PR，CANN 9.1.0。
- **适用粒度**：`operator_family`——fp32 归约近零输出且 double golden 的场景。
- **路线优先级**：本卡结论为 `banned`（放宽即掩盖），正确路径为接续点（未实施）。

## 根因

fp32 kernel 与 double golden 的机器精度差（|a−g|≈1.19e-7）在近零输出上按相对误差口径必然 MARE 超阈。关键判定：**这是真实超标还是指标病**——按官方小值域标准（FLOAT32 Error Threshold 2^-30≈9.3e-10），|a−g|≈1.19e-7 远大于 9.3e-10，即按官方标准该误差**真实超标**，不是"比较器数值不稳"造成的假失败。

## 解决配方

本轨迹最终以分母 epsilon 1e-7→1e-4 放宽关闭 37/37——**判定为 banned 掩盖性捷径**，不得作为修复配方复用：

- ε=1e-4 恰使最差用例压线通过（g=0.000014 时 relErr = 1.19e-7/(0.000014+1e-4) ≈ 1.04e-3 < 1.22e-3）；
- 放宽前后 NPU 输出逐位不变——kernel 精度没有任何改善；
- 按官方小值域标准 |a-g| 真实超标。

**正确做法（该轨迹未实施，接续点）**：

1. kernel 侧 Kahan 补偿求和/成对求和/分块 fp32 归约逼近 golden 精度；
2. 按小值域标准对小值子集单独计算 ErrorCount（需三方标杆）；
3. 或如实记录为 fp32 已知精度边界并在验收报告中显式声明——而不是改比较器。

### kernel 侧死路证据

- double 标量累加被 CCE 拒绝：`error: double precision operation is not allowed in aicore function`（AI Core 禁 double 是跨算子死路）；
- `isPad=true` 零填充 + 对 rowsThisBatch×rowLenAlign 元素单次 ReduceSum（期望 tree reduction 更准）实测更差：L0_003/004/005 MARE = 1.57e-3/2.10e-3/5.88e-3 三条失败（基线 L0_005=4.07e-3 恶化到 5.88e-3 并新增 L0_004）——零元素参与改变归约顺序，近零舍入更差，已回退。

## 设计决策

- **为何判定 banned**：三态齐备——① ε=1e-4 恰使最差用例压线通过；② kernel 侧手段未穷尽（只试了 double 与零填充单归约两条，未试 Kahan/成对求和）；③ 按官方小值域标准 |a-g| 真实超标。三态齐备即掩盖，非指标病。
- **与指标病分界**：若确属比较器数值不稳（MERE≪th 且 |a-g| 在舍入/累加噪声量级），按噪声界推导 eps 不是放松——分界就用官方小值域标准对照 |a-g|（与[官方小值域分支卡](official_small_value_branch_two_implementations.md)的三段判定同源）。

## 验证信号

- 基线与死路：L0_003 MERE=2.12e-07/MARE=1.57e-03、L0_005 MERE=4.54e-07/MARE=4.07e-03；isPad=true 单次归约后三条失败（1.57e-3/2.10e-3/5.88e-3，更差，已回退）；double 累加 CCE 编译拒绝（来源平台：950；证据等级：tool_observed）。
- 放宽后：`总计: 37 通过: 37 失败: 0`（L0_003 MARE=7.24e-04、L0_005 3.02e-04）——该"通过"依赖放宽后的比较器且 kernel 输出逐位不变，不构成 kernel 精度修复的证据；Mock 模式回归不受影响（37/37、golden 自测 10/10）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 放宽属掩盖的判据是"恰使最差用例压线通过 + kernel 侧手段未穷尽 + 按官方小值域标准 |a-g| 真实超标"三态齐备；缺任一态不构成掩盖判定，须分别核验。
- 本卡正确路径（Kahan/成对求和）未实施验证，是否能把 MARE 拉回 1.22e-3 内为未决项。

## 坏实践

- 放宽自研比对器分母 epsilon 使失败用例恰好通过，且不在验收报告中披露比较器改动——本轨迹实际发生，标记 banned，不得作为修复配方复用。

## 未决事项

- 【优先项】补 Kahan/成对求和实现并在同批用例（L0_003/L0_005）验证 MARE 能否回到 1.22e-3 内。
- 按小值域标准实现小值子集独立评估（含三方标杆可得性调研）。