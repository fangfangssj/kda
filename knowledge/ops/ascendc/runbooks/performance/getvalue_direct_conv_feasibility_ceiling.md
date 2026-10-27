---
type: "Runbook"
title: GetValue标量直卷大shape超时：最大用例不可行
description: 朴素直接卷积（GetValue 逐元素标量 GM 读 + 浮点累加）的单用例可行规模实测到约 154.6 GFLOPs（L0-032，修复多核并行度后在 10 分钟超时预算内 PASS），约 53.7 GFLOPs 用例稳定通过；104 用例 L1 全量批跑会因累积执行时间触发默认超时（SIGKILL 落在仅约 3.7 GFLOPs 的 L1-089 上，属批跑累积超时而非单用例规模超时）；验收要求的最大 shape（27.5 TFLOPs，约为已通过最大用例的 178 倍）不可行，须以代表性大 shape 用例替代验证并转向 Cube 路线。
tags: [task.optimization, topic.performance.analysis, topic.compute.neighborhood.convolution]
status: stable
sources: []
created_at: '2026-09-17T00:54:40Z'
updated_at: '2026-09-17T00:54:40Z'
platforms: ["950"]
---

# GetValue标量直卷大shape超时：最大用例不可行

## 现象

在 Ascend950PR（CANN 9.1.0）上开发自定义算子 BenchNoHarnessConv2d：当前 kernel 是朴素直接卷积——`GlobalTensor::GetValue` 逐元素从 GM 标量读 x 与 filter，寄存器浮点累加，`DataCopyPad` 批量写回（为正确性优先有意选择的实现，未走设计文档的 Load3D+MMAD+Fixpipe Cube 路线）。全量覆盖轮的验收计划要求验证最大 shape 边界用例：N=8, Cin=2048, Cout=2048, H=W=128, K=5×5。按 shape 计算其计算量为 27.5 TFLOPs（输出 268,435,456 元素 × 每元素 51,200 次乘加 × 2）；对照已通过的最大用例 L0-032（N=8, C=512, 64×64, K=3，约 154.6 GFLOPs，修复多核并行度前单跑即超时被杀、修复后在 10 分钟超时预算内完成）——量级差约 178 倍，标量路径无法在合理时间内完成，判定不可行，未实际运行。同会话中修复并行度后 L1 全量（104 用例）批跑仍被默认超时 SIGKILL（EXIT_CODE=137，kill 落在 L1-089——N=1, C=128, H=W=112, K=3，仅约 3.7 GFLOPs）：该次被杀是全量批跑的累积执行时间超出默认超时预算，而非该用例自身规模超时（会话内估算其单跑仅约数秒）；L1 全量最终以更长超时/分批完成。最终以三条代表性大 shape 用例（N=4, Cin=256, Cout=256, H=W=64, K=5×5，每条约 53.7 GFLOPs，fp16/fp32/bf16 各一）替代验证，3/3 全过；最大 shape 的性能缺口立项为 Cube 路线优化 issue。

## 适用场景

- 工程形态：AscendC 算子的 ST 边界/最大 shape 验证计划制定、大计算量用例的可行性评估与超时预算分配。
- API 路径：`GlobalTensor::GetValue`/`SetValue` 标量路径的吞吐特性；设计文档 Load3D（硬件 im2col）+ MMAD（矩阵乘累加）+ Fixpipe（L0C→UB+量化）Cube 路线作为替代。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——标量 GM 读取吞吐是架构级特性，对 950 上任何 GetValue 主导的 kernel 的规模规划适用；其他架构未验证。
- 路线优先级：safety_fallback——标量直卷路线是正确性优先的兜底路线，性能上明确劣于 Cube 路线；采用它验证大 shape 时必须接受本卡的规模边界。fallback 声明：本卡不是性能优化方案，只界定兜底路线的可行边界并给出验收替代口径；Real 性能目标须由 Cube 路线达成。
- 明确不适用 / 未验证场景：Cube/Load3D 路线实现后的规模上限（预期完全不同）；向量 DataCopy 主导的搬运型 kernel（瓶颈模型不同）。

## 判别方法（规模可行性估算）

对直接卷积类 kernel，用 shape 算两个数即可判断可行性：

- FLOPs = N × C_out × H_out × W_out × C_in × KH × KW × 2；
- 标量读取次数 ≈ FLOPs（每次乘加对应 x、filter 各一次 GetValue）。

本会话的分析输出原文（对四个 shape）：

```
dilation=16 K=3x3 H=32 pad=16: H_out=32, W_out=32
Max shape: H_out=128, W_out=128, y_elems=268435456, flops=27487.8G
  Input bytes: 536.9MB (fp16)
  Filter bytes: 209.7MB (fp16)
  Output bytes: 536.9MB (fp16)
Large feasible: N=4 Cin=256 Cout=256 H=64 W=64 K=5x5: H_out=64, W_out=64, flops=53.7G
```

经验边界（本会话实测锚定）：修复多核并行度后，单用例规模约 154.6 GFLOPs（L0-032）可在 10 分钟超时预算内完成，约 53.7 GFLOPs 的用例稳定通过；须区分单用例规模与批跑累积时间——104 用例全量批跑的累积执行时间会触发默认超时（kill 可落在仅约 3.7 GFLOPs 的用例上），全量验证须分批或加长超时；T（TFLOPs）级用例不可行（外推）。

## 尝试与观测

- 尝试：全量覆盖轮验收计划要求验证最大 shape（N=8, Cin=2048, Cout=2048, H=W=128, K=5×5，27.5 TFLOPs）。
- 观测一（单用例规模上界）：L0-032（N=8, C=512, 64×64, K=3，约 154.6 GFLOPs）在 tiling 多核切分修复（totalTasks 由 N×cOutTileCount 改为 N×C_out，全部核参与）前单跑即被超时 SIGKILL（137），修复后在 10 分钟超时预算内完成 PASS。
- 观测二（批跑累积超时）：修复并行度后 L1 全量（104 用例）批跑仍在默认超时被杀（137），kill 落在 L1-089（N=1, C=128, H=W=112, C_out=128, K=3；FLOPs = 1×128×112×112×128×9×2 ≈ 3.7 G）——被杀机制是全量批跑累积执行时间超限而非该用例自身规模（会话内估算其单跑约数秒）；L1 全量最终以更长超时/分批完成。
- 观测三（可行上界锚点）：L1-125/126/127（各约 53.7 GFLOPs，3 dtype）全部 PASS；L0-032 与 L1-089 计算量相差约 41.8 倍（154.6/3.7），二者结局不同源于执行方式（单跑 vs 全量批跑累积）而非仅规模。
- 观测四（量级差）：最大 shape 27.5 TFLOPs ≈ 178 × L0-032（154.6 GFLOPs）计算量，外推需数小时。

## 验证信号

以下信号来自该算子全量覆盖会话：对最大 shape 做可行性分析后，以代表性大 shape 用例替代验证并立项优化 issue。

- L0-032（N=8, C=512, 64×64, K=3，约 154.6 GFLOPs）修复并行度前单跑即被超时 SIGKILL（EXIT_CODE=137），修复后（totalTasks 改为 N×C_out、全部核参与）在 10 分钟超时预算内完成 PASS（MERE=1.76e-04，16,777,216 元素）（来源平台：950；证据等级：tool_observed）。
- L1-089（N=1, C=128, H=W=112, K=3，约 3.7 GFLOPs）在多核并行度修复后的 L1 全量（104 用例）批跑中被超时 SIGKILL（EXIT_CODE=137，kill 落在该用例）——属批跑累积执行时间超限而非单用例规模超时（会话内估算其单跑约数秒），L1 全量需更长超时/分批才完成（来源平台：950；证据等级：tool_observed）。
- L1-125/126/127（N=4, C=256, 64×64, K=5×5，各约 53.7 GFLOPs）3 dtype 全部 PASS：fp16 MERE=1.76e-04、fp32 MERE=1.11e-06、bf16 MERE=1.41e-03，各 4,194,304 元素（来源平台：950；证据等级：tool_observed）。
- 规模分析输出：Max shape flops=27487.8G（≈27.5 TFLOPs）、Large feasible flops=53.7G（来源平台：950；证据等级：tool_observed——shape 确定性算术）。

（最大 shape 用例未实际运行，"需数小时、不可行"为由量级外推得出的边界判断而非实测，见失效边界。）

## 失效边界

- 最大 shape 用例未实际运行（判定不可行后放弃），"需数小时"是外推结论而非实测。
- 边界数字（3.7/53.7/154.6 GFLOPs）依赖当时的多核并行度、超时预算与执行环境，只作量级参考；批跑累积超时阈值还依赖用例总数与顺序，换环境应重新以中等规模用例标定。
- L1-089 的被杀不能解读为"约 3.7 GFLOPs 即单用例规模超时"——它是 104 用例批跑的累积超时，单用例规模边界以 L0-032（约 154.6 GFLOPs，修复并行度后）为锚。
- 本卡只覆盖 GetValue 标量直卷路线；Cube 路线（Load3D+MMAD+Fixpipe）实现后边界完全不同（issue 中预估提升 100-1000 倍，未验证）。

**未决项**：

- Cube 路线（Load3D 硬件 im2col + MMAD + Fixpipe + Double Buffer + L1 复用）未实现，最大 shape 验证与性能目标待该路线落地后重开；其预估收益（100-1000 倍）未验证。
- 标量路线的精确吞吐（GetValue/秒）未测量——建议下轮对中等用例加计时输出，把单用例规模边界与批跑累积超时阈值从"量级参考"收紧为可计算公式。
## 性能影响

- 相对代价（同算子内部成对对比，来源平台：950）：单用例约 154.6 GFLOPs（L0-032，修复并行度后）是本会话实测通过的上界，约 53.7 GFLOPs 规模内可稳定作为正确性兜底；批跑累积时间会在更小用例（约 3.7 GFLOPs）上触发默认超时，全量批跑须分批/加长超时；若以 Cube 路线为参照，标量路线在百 GFLOPs 以上规模不可用（相对代价趋于无穷）。无可用的绝对单跑分数（未计时）。

## 坏实践

- 用朴素标量 kernel 直接跑验收要求的最大 shape 用例（会耗尽超时且无产出）。
- 修复并行度后误以为规模问题已解决——并行度只改善常数倍，标量读取的量级瓶颈仍在。
- 把全量批跑的累积超时（kill 落在小用例上）误读为该用例的单用例规模超时——判读超时先区分"单用例耗时"与"批跑累积耗时"，必要时单跑复现。

## 误诊

- 无（本卡是主动可行性评估，无错误归因过程）。
