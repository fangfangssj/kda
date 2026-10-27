---
type: "Runbook"
title: st-design 生成用例输出维数独立采样需显式约束
description: ascendc-st-design 求解器把无约束的输出 y.dimensions 当 Level 0 独立采样，而 y.shape 已可由 x.shape 派生——同 shape 输出算子的用例元数据会自相矛盾；显式加 y.dimensions=x.dimensions 约束后降至 Level 1 派生因子，anchor 只剩输入因子。
tags: [task.testing, topic.implementation]
status: stable
sources: []
created_at: '2026-09-17T04:06:11Z'
updated_at: '2026-09-17T04:06:11Z'
platforms: ["950"]
---

# st-design 生成用例输出维数独立采样需显式约束

## 现象

用 ascendc-st-design 脚本链为逐元素算子（输出 y 与输入 x 同 shape）生成 ST 用例：`03_参数定义.yaml` → `generate_test_factors.py`（04_测试因子）→ `generate_implicit_constraints.py`（05_约束定义）→ `generate_solver_config.py`（06_求解配置）→ `generate_factor_values.py` → L0/L1/L2 用例。检查生成的 `06_求解配置.yaml` 时发现：输出 `y.dimensions` 被列为 **Level 0 anchor（独立采样因子）**，而 `y.shape` 已由 `C-SHAPE-001`（`y.shape = x.shape`）从 `x.shape` 派生。若某用例采样出 `x.dimensions=3` 而 `y.dimensions=2`，生成的用例行元数据自相矛盾（shape 相同但维数不同在数学上不可能）。

## 适用场景

- 工程形态：算子 ST 测试设计（ascendc-st-design 脚本链：测试因子/约束定义/求解配置/用例生成）
- API 路径：`generate_solver_config.py` 的因子分层（anchor=Level 0 独立采样 / derived=由 calculate 约束派生）；`05_约束定义.yaml` 的 `calculate` 约束
- 验证过的平台与 CANN 版本：Ascend 950（ascend950）目标算子 + 脚本链实测（平台无关的用例生成工具）
- 适用粒度：`operator_family`（所有"输出与输入同 shape"的算子做 ST 测试设计）
- 路线优先级：recommended_fastpath
- 明确不适用 / 未验证场景：输出维度独立于输入的算子（reshape/broadcast/归约）**不应**加此约束；带属性的算子未验证

## 根因

约束求解器只按**显式 calculate 约束**建立因子依赖；"y.shape = x.shape" 不会隐式传递到 `y.dimensions`，输出维数因子缺约束时默认作为 anchor 独立采样。

## 解决配方

在 `05_约束定义.yaml` 的约束列表中，为输出维数补显式派生约束（before：只有 C-SHAPE-001 → after：新增 C-DIM-001）：

before：

```yaml
  # 输出y的shape必须与输入x的shape完全一致（不支持广播）
  - id: "C-SHAPE-001"
    type: calculate
    sources: ["x.shape"]
    target: "y.shape"
    expression: "sources[0]"
    description: "输出y的shape等于输入x的shape"
```

after：

```yaml
  # 输出y的维度数必须与输入x一致（shape完全一致的前提）
  - id: "C-DIM-001"
    type: calculate
    sources: ["x.dimensions"]
    target: "y.dimensions"
    expression: "sources[0]"
    description: "输出y的维度数等于输入x的维度数"

  # 输出y的shape必须与输入x的shape完全一致（不支持广播）
  - id: "C-SHAPE-001"
    type: calculate
    sources: ["x.shape"]
    target: "y.shape"
    expression: "sources[0]"
    description: "输出y的shape等于输入x的shape"
```

然后重新运行 `generate_solver_config.py` 覆盖 06_求解配置。

## 设计决策

- 选"补显式约束"而非"忽略不一致"：`y.shape` 由 C-SHAPE-001 最终覆盖，用例实际执行的 shape 值是对的，`y.dimensions` 只影响用例元数据——但脏元数据会误导后续用例筛选与统计，约束补全成本一行。
- 只约束 dimensions，不为 y.dtype 重复造约束：dtype 派生（y.dtype=x.dtype）已由参数定义/隐式约束链覆盖（本例 y.dtype 位于 Level 1 派生层）。

## 预防

- 写完 05_约束定义后，检查 06_求解配置的分层表：**每个输出 tensor 的 dimensions/shape/dtype 因子都不应是 Level 0 anchor**（除非语义上真的独立）；anchor 集合应只含真正自由的输入因子。
- "输出与输入同 shape"类接口，dimensions 与 shape 两条约束都要写（shape 约束不隐含维数约束）。

## 验证信号

以下信号来自该逐元素算子 ST 用例生成会话的求解配置检查与约束补全闭环。

- 修复前：06_求解配置中 `y.dimensions` 位于 Level 0 anchor、独立采样（来源平台：950；证据等级：tool_observed）。
- 修复后：重新生成求解配置，`y.dimensions` 降至 Level 1（由 x.dimensions 派生），anchor 只剩 x.dimensions；后续因子值与 L0/L1/L2 用例正常生成（L0 15 条、L1 500 条）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 该行为基于 ascendc-st-design 脚本链当前实现（按显式 calculate 约束建依赖）；脚本升级若引入 shape→dimensions 隐式推导，此约束变为冗余（无害）。
- 仅适用于"输出维数=输入维数"的接口契约；输出维数语义独立的算子加了反而锁死本应自由的因子。
- 平台边界：验证环境为面向 Ascend 950 的算子工程与 ascendc-st-design 脚本链（host 侧）；其他平台未验证（脚本行为平台无关，推断可迁移）。

**未决项**：

- 无。
