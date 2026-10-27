---
type: "Runbook"
title: "int32 fp32累加reduce binary_equal不可达"
description: "int32 输入的 reduce 类算子内部 cast 到 fp32 累加，kernel 的 ReduceSum+bisection cache tree 与 golden 的 pairwise 折半是两棵不同 reduce 树；sum 超过 fp32 精确整数范围 2^24 时舍入路径不同致 1 ULP 差，cast 回 int32 后 bitwise 不一致，binary_equal 永不可达。R2（同算子 group tail-R 分支，CANN 9.1.0）：第二条出路——数据设计约束 n·v²≤2^24 使平方和 fp32 精确可表示，任意累加次序同值、bitwise 数学可达（SmallInt 1..8 循环模式 + host 侧预检探针）；并实测 CAST_TRUNC 把亚 ULP 差放大成 ±1 整数位差，device 可比 golden 更接近精确数学。"
tags: ["task.debugging", "topic.precision", "topic.compute.reduction", "topic.compute.elementwise.type_conversion"]
status: stable
sources: []
created_at: "2026-08-18T00:00:00Z"
updated_at: "2026-09-17T01:10:00Z"
platforms: ["950"]
aliases: ["bisection_cache_tree"]
---

# int32 fp32累加reduce binary_equal不可达

## 现象

EuclideanNorm（L2 范数）算子黑盒 ACLNN ST 全量 507 条正向用例（TTK aclnn 模式跑 `blackbox_all.csv`，soc=Ascend950PR_9579）跑出 31 条 FAIL。其中 27 条为 `int32` dtype、`binary_equal`（逐点 bitwise 对齐）精度标准下的 bitwise mismatch：用例的 `input_data_range` 命中 `INT_MIN=-2147483648` 或 `INT_MAX=2147483647` 边界，`x²` 高达 `2^62`，且 R 轴（归约轴）元素数大（如 `L0_037` shape `(345812,13)` axes=`[0]`，R=345812），平方和 `sum = 345812 × 2^62` 远超 fp32 精确整数范围 `2^24=16777216`。失败用例 precision 从 `0.0%` 到 `99.95%` 不等，逐点 bit 不一致。kernel-developer 经 3 轮修复（fix1 52.9m / fix2 5.2h / fix3 8.8h，被用户取消）均未能使 int32 bitwise 全绿；fix2 把 31→2（4 个 NO_OUTPUT 已修、27 int32 bitwise 收敛到 1），但残留 int32 `L1_345` 仍 bitwise mismatch，用户在 fix3 取消，停止反复修复。

## 适用场景

- **工程形态**：ACLNN 算子仓开发（ops-registry-invoke），黑盒 ACLNN ST 用 TTK aclnn 模式跑 `blackbox_all.csv`。
- **API 路径**：kernel 内部 `int32 → cast fp32 → square → AscendC::ReduceSum(chunk 内树形) → bisection cache tree(chunk 间合并) → Sqrt → cast int32`；golden `int32 → astype fp32 → square → _pairwise_sum_last(递归折半) → sqrt → astype int32`。
- **验证过的平台与 CANN 版本**：Ascend950（soc=Ascend950PR_9579，result.csv 的 soc 列）；CANN 版本未在 trace 内显式记录。
- **适用粒度**：`operator_family`——对任何 int32 输入、内部用 fp32 累加的 reduce 类算子（L2/Norm/SumSqrt 等）用 `binary_equal` 验收均成立。
- **路线优先级**：`banned`——「让 int32 fp32-累加 reduce 与不同 reduce 树的 golden 达成 bitwise 一致」是不可达路线；正确路线是在 spec 阶段把 int32 的 `numerical_tolerance` 从 `binary_equal` 改为 `atol`（如 `atol=1`），而非死磕 kernel reduce 顺序。
- **不适用或未验证场景**：fp16/fp32/bf16 输入（本开发会话内 fp16/fp32 零失败，bf16 仅 mare 超差非 bitwise）；int64 累加路径未在本开发会话验证。

## 根因

int32 用例用 `binary_equal` 精度标准（bit 级精确匹配）。kernel 与 golden 的计算路径是两棵不同的 reduce 树：

```
kernel: int32 → cast fp32 → square → ReduceSum(chunk) → bisection cache tree → sqrt → cast int32
golden: int32 → astype fp32 → square → _pairwise_sum_last(递归折半) → sqrt → astype int32
```

| | kernel | golden |
|---|---|---|
| **分块** | R 元素按 rUbFactor 分 chunk，每个 chunk 用 `AscendC::ReduceSum` 内部树形 reduce | 不分 chunk，全部 R 元素展平成一个一维数组 |
| **合并** | chunk 间用 bisection cache tree 合并 | 递归折半（`_pairwise_sum_last`） |
| **reduce 树结构** | 两级：chunk 内 ReduceSum + chunk 间 bisection | 单级：全量递归折半 |

**机制推导（不跳步）**：fp32 的精确整数范围为 `2^24 = 16777216`。当 sum 超过 `2^24` 时，fp32 浮点结合律不再成立——不同 reduce 顺序（chunk 树形 vs 递归折半）会产生不同的 fp32 舍入结果。具体地：`INT_MIN² = (-2147483648)² = 2^62 = 4611686018427387904`；`L0_037` 的 R=345812，`sum = 345812 × 2^62 ≈ 1.59e21`，远大于 `2^24`。fp32 累加在如此大量级下发生舍入，kernel 与 golden 的舍入路径不同 → fp32 sum 差 1 ULP → `sqrt` 后差 1 ULP → cast 回 int32 后 bitwise 不一致。

关键点：`spec.yaml` 定义 `accumulation_order = stable_in_axis`（归约轴不跨核），但 **`stable_in_axis` ≠ bitwise reproducible**——它只保证归约轴不被多核切分，不保证 reduce 树结构与 golden 一致。对 int32 用 `binary_equal` 精度标准本身就不合理：fp32 域浮点结合律不成立，不同 reduce 顺序不可能保证 bitwise 一致。

## 失败路线

本卡为 `negative_result`，记录 `banned` 路线——「让 kernel 的 fp32 累加 reduce 与 golden 的 pairwise reduce 达成 bitwise 一致」不可达。不暗示已有成功修复。

- **fix1（52.9m）**：kernel-developer 加了 `CompareScalar+Select` 处理 int32 最终 cast 回 int32 时的溢出（kernel 饱和 `INT_MAX` vs numpy 回绕 `INT_MIN`）。修的是**最终 cast 回 int32 时的溢出**，不是累加精度。且 fix1 修了 C++ aclnn 测试工程但**没重跑 TTK 黑盒**（`result.csv` 未更新），验收仍看旧 31 FAIL。52.9m 基本浪费。
- **fix2（5.2h）**：修 `CastSquareRintVfImpl` 模拟 round-to-nearest（int32→fp32 cast 的精度）。修的是**int32→fp32 cast 的精度**，不是累加精度。同时修了 NO_OUTPUT 崩溃。31→2 FAIL。
- **fix3（8.8h）**：尝试修最后 2 个（bf16 mare + int32 bitwise），被用户取消。
- **核心问题未触及**：kernel 的 `ReduceSum + bisection cache tree` 与 golden 的 `_pairwise_sum_last` 是**不同的 reduce 树**，当 sum 超过 `2^24` 时必然产生 ULP 差异。`AscendC::ReduceSum` 是硬件指令，内部 reduce 顺序不可控；golden 的 pairwise 是软件递归折半。两者不可能 bitwise 一致。

**结论**：`binary_equal` 对 int32-fp32-累加 reduce 不可达。正确做法是在 spec 阶段把 int32 dtype 的 `numerical_tolerance` 从 `binary_equal` 改为 `atol`（如 `atol=1`）。

## 误诊

kernel-developer 在 fix2/fix3 陷入精度调试死循环：反复改 kernel 的 cast 路径（`CastSquareRintVfImpl`）、pairwise reduction 模拟、tiling 切分（试图让 R 单 chunk），试图让 fp32 累加 bitwise 一致。这些修复方向均未触及根因——reduce 树结构不一致。诊断阶段才定位：**根因是 spec 容差设计问题（int32 用 binary_equal），不是 kernel bug**。kernel-developer 未在首轮加载 `ascendc-precision-debug` skill（违反分诊规则「归类故障→加载 skill→再动手」），直接凭自身推理调试，导致 14.9h 浪费。本条为 agent 诊断结论，不进验证信号。

## 预防

- spec 阶段对 int32 输入、内部 fp32 累加的 reduce 算子，`numerical_tolerance` 用 `atol`（如 `atol=1`），不用 `binary_equal`。
- spec.yaml 的 `accumulation_order: stable_in_axis` 只保证归约轴不跨核，**不等于 bitwise reproducible**——设计验收不应据此推断 bitwise 可达。
- kernel-developer 面对精度问题时先加载 `ascendc-precision-debug` skill 走诊断流程；当诊断表明根因是 spec/容差设计问题而非 kernel bug 时，在 LOG 标注 `SPEC_ISSUE` 并反馈，不得反复改 kernel 试图绕过。

## 验证信号

- 黑盒正向 507 条中 31 FAIL：int32 118 条中 29 失败（24.6%）、bf16 125 条中 2 失败（1.6%）、fp16 143 条 0 失败、fp32 121 条 0 失败。29/31 为 int32（27 bitwise + 2 NO_OUTPUT），2/31 为 bf16 NO_OUTPUT（来源平台：950；证据等级：tool_observed）
- 27 条 int32 bitwise mismatch 的逐用例明细：`L0_037` shape `(345812,13)` axes `[0]` data_range `(-2147483648,-2147483648)` precision 0.0%；`L1_345` shape `(25,36,2,68200)` axes `[0,-3]` data_range `(-1000,-10)` precision 99.95%；`L1_175` data_range `(2147483647,2147483647)` precision 0.0% 等，`precision_metrics` 统一为 `{standard: binary_equal, pass: False, reason: bitwise mismatch}`（来源平台：950；证据等级：tool_observed）
- fix2 修复后 31→2 FAIL（4 NO_OUTPUT 全部消除，27 int32 bitwise 收敛到 1 条 `L1_345`），残留 `L1_439` bf16 mare=0.00415 ≥ 阈值 0.00122 + `L1_345` int32 bitwise，用户在 fix3 取消（来源平台：950；证据等级：tool_observed）

## 失效边界

- a2/a3 平台未在本开发会话内实测——`AscendC::ReduceSum` 硬件指令的内部 reduce 顺序在不同 NPU 代际可能不同，`binary_equal` 不可达的结论推断一致（fp32 浮点结合律不成立是数学事实，与平台无关），未验证，推断可迁移。
- int64 累加路径（若 kernel 改用 int64 累加再开方）未在本开发会话验证；理论上 int64 精确整数范围 `2^63` 远大于 `2^62`，可能达成 bitwise，但需 kernel 与 golden 同时改用 int64，不在本卡修复范围。
- 若 spec 把 int32 tolerance 改为 `atol=1`，本卡所述「不可达」前提即被正确解除，应改走 atol 验收路线。

## 坏实践

- 对 int32 输入、fp32 累加的 reduce 用 `binary_equal` 精度标准。
- 把 `accumulation_order: stable_in_axis` 当作 bitwise reproducible 的保证。
- 修完代码不重跑原始验证命令（TTK 黑盒 `result.csv`），只跑 C++ 工程替代。
- 精度问题不加载诊断 skill，凭自身推理反复改 kernel。


## 独立复现 R2：数据约束 n·v² ≤ 2^24 使 bitwise 数学可达（group tail-R 分支，CANN 9.1.0）

同算子（euclidean_norm）group tail-R 分支（Phase 1 各核局部 R 段 fp32 归约 + Phase 2 汇总 + sqrt + CAST_TRUNC）的 int32 逐位（atol=0）用例失败，根因与本卡同源（fp32 精确整数域 2^24），但给出了**不改 spec 容差的第二条出路——数据设计约束**：

**失败与机制**：`[1,65536] axes=[-1] int32` 用例（数据为顺序整数 1..65536）报 `int32 bitwise mismatch (elem0 dev=9686441.000000 golden=9686442.000000)`。numpy 探针逐位复算：精确 sum(x²)=93,827,139,731,456（超 2^24 达 5,592,533 倍），sqrt=9686441.025，**CAST_TRUNC=9686441=device**（device 与精确数学一致）；golden 单遍线性 fp32 累加 sum 舍入到 9.382716e13（偏差 +2.03e7，约 23 ULP）→ 9686442；kernel 两段式累加 sum 舍入到 9.382714e13（数十 ULP）→ 9686441。两条路线各向相反方向舍入，精确 sqrt 距整数边界仅 0.975——截断把亚 ULP 差放大成 1 个整数的位级差异；逐位断言比较的是 fp32 舍入噪声而非 kernel 正确性，**device 反而是更精确的一方**。

**数据设计约束（充分条件）**：n 个元素、值域上界 v 时 sum(x²) ≤ n·v² ≤ 2^24 则全部部分和与总和都 fp32 精确可表示，任意累加次序同值、bitwise 数学可达。本例 n=65536 → v ≤ 16，取 1..8 循环（sum(x²)=1,671,168 < 2^24；sqrt=1292.736… 非整数，仍真实覆盖 CAST_TRUNC 向零截断路径）。修复为新增 SmallInt 数据模式（`v = (double)((i % 8) + 1)`）并用于 int32 逐位用例；kernel 零改动、断言强度（atol=0/rtol=0）保持——这是修测试数据的数学前提缺陷，不是测试放松。

**host 侧预检探针**（可直接复用）：

```python
import numpy as np
x = np.arange(1, 65537, dtype=np.int32)            # 或任意待测数据
exact = int(np.sum(x.astype(np.int64)**2))          # 精确平方和（int64 无舍入）
print(f"sum of squares = {exact} (2^24 = {2**24}, exact-in-fp32: {exact <= 2**24})")
```

- 修复前：该用例 FAILED（dev=9686441 vs golden=9686442），同套件其余 7/8 通过（来源平台：950；证据等级：tool_observed）。
- 探针：精确 CAST_TRUNC=9686441（=device）、fp32 线性累加=9686442（=golden）、两段式=9686441；sum 超 2^24 达 5,592,533 倍（来源平台：950 宿主侧复算；证据等级：tool_observed）。
- 修复后：该分支 8/8 PASSED；回归 Branch-0 5/5、Branch-1 18/18、Branch-2 14/14、Branch-4 8/8、public TilingUT 19/19 完好（来源平台：950；证据等级：tool_observed）。
- 新数据核验：`sum of squares = 1671168 (exact-in-fp32: True)`，sqrt=1292.736→CAST_TRUNC=1292（来源平台：950；证据等级：tool_observed）。

**R2 边界**：n·v² ≤ 2^24 是充分条件（整数和精确可表示）；容差型断言（rtol/atol>0）不受此约束；golden/kernel 改用 fp64/Kahan 累加则约束解除（本算子 spec 定义为 fp32 累加）；fp16/bf16 输出走 CAST_RINT 同为位级断言、超精确域数据同样可能位漂（本会话因数据量级未触发，未实测失配）；平方和非整数时判据需改为"所有部分和与和均 fp32 可精确表示"。与本卡主结论的关系：spec 侧把 binary_equal 改 atol 是验收口径出路，数据约束是不改口径时让 bitwise 前提成立的测试设计出路——先做 host 侧精确复算再定责（device 可能比 golden 更准）。
