---
type: "Runbook"
title: "Two-Product split 遇 inf 产 NaN 破坏 inf 传播"
description: "向量核引入 Dekker split（×4097）补偿乘法误差后，±inf 输入使 inf−inf=NaN 污染全路输出，原 Axpy 正确传播 inf 的边界用例反成\"NaN位置不匹配\"回归；EFT 算法须防非有限输入。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.arithmetic]
status: stable
sources: []
created_at: '2026-09-17T00:50:52Z'
updated_at: '2026-09-17T00:50:52Z'
platforms: ["950"]
---

# Two-Product split 遇 inf 产 NaN 破坏 inf 传播

## 现象

在 Ascend 950（arch35，CANN 9.2.0）的 DepthwiseConv2D 类算子 fp32 路径引入 Two-Product + Double-Double 补偿累加（目的是改善大值域用例的 MARE）之后，重跑 bench 全量 20 个精度用例：原本通过的 case 13——一个 value_range=[-inf, inf] 的 inf 边界用例（x=[2,63,31,31]、weight=[63,3,3]、bias=[63]）——出现新失败，报错原文：

```
[13/20] level3/depthwise_conv_2d_13: ❌ 精度不达标:
  - float32[y]: ❌ MERE=0.000000, MARE=0.000000 (threshold=1.220703e-04, mare_threshold=1.220703e-03), NaN位置不匹配
compare: 所有有效位置均为 NaN/Inf 且匹配；两个算子均输出 NaN 的极端情况可能掩盖独立的数值 bug，请人工核查 case 输入分布。
```

注意其特殊性：MERE/MARE 均为 0（有效位置全匹配），失败原因只有"NaN 位置不匹配"——即算子输出在某些位置产生了参考输出没有的 NaN。而引入补偿之前，原始 Axpy 实现下该用例是 success（MARE=7.608650e-04）。这是典型的"精度补偿改动引入正确性回归"。

## 适用场景

- 工程形态：AscendC 向量核 fp32 路径上叠加误差无关变换（EFT：Two-Product / Dekker split / Double-Double / Kahan 家族）的精度优化改动，且算子评测/业务输入可能包含 ±inf（value_range 含 inf 的边界用例、无界激活输出等）。
- API 路径：`Muls`/`Sub`/`Add` 向量指令组成的 split 序列；对照原始 `Axpy<float,float>` 的 inf 传播语义。
- 验证过的平台与 CANN 版本：Ascend 950（arch35/DAV_3510），CANN 9.2.0。
- 适用粒度：platform_wide（IEEE 浮点语义层面成立，与具体算子无关）。
- 路线优先级：banned（对输入可能含 ±inf/NaN 的路径，无有限值防护的 split/EFT 补偿序列禁止使用；原 Axpy 为 recommended_fastpath）。
- 明确不适用/未验证场景：输入保证有限值的路径（split 语义正常）；fp16/bf16 的 EFT（未尝试）；带 IsFinite 向量防护的混合方案（本轨迹未实现，成本未评估）。

## 根因

单句归因：Dekker split 的 `hi = t - (t - a)` 在 a=±inf 时计算 `inf - inf = NaN`，使补偿分支把原本正确传播的 inf 污染成 NaN 并扩散到整个输出。

机制链：

```cpp
// split(x): 向量运算
Muls(tmp, xIn, 4097.0f, count);          // tmp = x * 4097  →  inf * 4097 = inf
Sub(y, tmp, xIn, count);                 // y = tmp - x     →  inf - inf = NaN
Sub(xHi, tmp, y, count);                 // xHi = tmp - y   →  inf - NaN = NaN
Sub(xLo, xIn, xHi, count);               // xLo = x - xHi   →  NaN
```

对照原始 Axpy 的 inf 语义：`Axpy(acc, x, w, n)` 内部为 `acc += x * w`，`inf * w = inf`（w 非 0）、`inf + inf = inf`——inf 作为"合法的溢出表示"全程正确传播，不产生 NaN。所以补偿改动不是"精度不够"，而是把一条原本正确的语义路径改坏了。

## 判别方法

- 触发信号：精度评测中出现 `MERE=0.000000, MARE=0.000000` 但 `NaN位置不匹配` 的失败（误差指标全绿 + NaN 位置差异的组合是 EFT 改动污染 inf 的指纹）。
- 确认步骤：核对失败用例的 value_range 是否含 ±inf；对照引入补偿前的同一用例结果；定位 kernel 中 `a * 4097` / `t - (t - a)` 形态的 split 序列。
- 验证方式：回退补偿改动（或对 inf 值域用例走原始路径）后重跑全量用例，NaN 位置不匹配消失。

## 验证信号

本卡验证信号来自补偿版本与回退版本的同一 bench 全量评测对照：

- Two-Product + Double-Double 版本：case 13 报 `MERE=0.000000, MARE=0.000000 ... NaN位置不匹配`，同时 case 5 MARE=0.013790 仍未达标（20 用例中 2 个失败）（来源平台：950；证据等级：tool_observed）。
- 原始 Axpy 基线（引入补偿前）：case 13 success，MERE=2.068186e-07、MARE=7.608650e-04（来源平台：950；证据等级：tool_observed）。
- 回退为 Axpy 后：case 13 恢复通过，bench 恢复 19/20，UT 57/57、ST 53/53（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论覆盖"无防护的 split/EFT 序列 + 可能 inf 的输入"这一组合；对输入保证有限的路径不适用。
- `inf * 0 = NaN`、`inf - inf = NaN` 是 IEEE 语义，任何平台的浮点实现一致；但"评测框架把 NaN 位置差异单列为失败项"的行为取决于评测框架（本例评测框架会）。
- 未验证，推断可迁移：其他向量平台上的 EFT 实现同样需要有限值前提，结论按 IEEE 语义推断可迁移。

**未决项**：

- 向量级 IsFinite 防护 + 混合路径（有限值走补偿、非有限值走 Axpy）的可行性与开销未评估；若未来重试补偿路线，这是必要前置。
## 坏实践

- 给必须处理 inf 值域的算子路径直接套教科书 EFT 算法（Two-Product/Double-Double）而不检查输入有限性：教科书算法默认有限输入，"理论精度提升"会变成"边界用例回归"。
- 只看 MERE/MARE 指标不看 NaN/Inf 位置校验：本例若只看误差指标会误判为"全对"。

## 预防

引入任何含 `t - (t - a)`（split）或 `a*b` 误差分解的补偿算术前：

1. 盘点算子的值域契约：评测用例/上层调用是否可能传 ±inf、NaN（value_range 含 inf、无界激活等）；
2. 若可能：要么该路径保持原始乘加（inf 语义天然正确），要么实现向量级有限值判定 + 分支防护（并评估其指令开销）；
3. 回归时必须包含 inf 值域用例，并把"NaN 位置不匹配"作为独立失败信号盯住。

## 误诊

- "case 13 失败说明原算子 inf 处理也有问题"：误；原始 Axpy 下该用例通过，回归由补偿改动引入（tool_observed 前后对照）。
