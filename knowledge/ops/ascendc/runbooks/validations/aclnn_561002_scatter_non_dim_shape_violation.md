---
type: "Runbook"
title: "ACLNN 561002：scatter 复现用例形状撞校验"
description: "自建 aclnn 复现程序调 GetWorkspaceSize 返回 ACLNN err 561002，两次都是用例形状撞上算子 TilingFunc 的全维 indices.shape[k] ≤ data.shape[k] 校验（比 torch.scatter 的 k≠dim 约束更严，dim 维也查）——先逐维核对用例形状与算子实际校验分支，再怀疑 kernel。"
tags: [task.testing, topic.compute.indexing, topic.implementation.operator_registration]
status: stable
sources: []
created_at: "2026-09-14T12:20:54Z"
updated_at: "2026-09-14T12:20:54Z"
platforms: ["950"]
---

# ACLNN 561002：scatter 复现用例形状撞校验

## 现象

在 Ascend950PR（CANN 9.2.0）上为自定义算子 BenchHarnessScatter 写自建 aclnn 复现程序时，两次不同的测试都在 `aclnnBenchHarnessScatterGetWorkspaceSize` 处拿到同一个错误码：

```
ACLNN err 561002 repro_failures.cpp:67
```

- 第一次（多核竞争复现）：data (8,4096)、indices/updates (2000,4096)、dim=1、reduce=add——indices.shape[0]=2000 远大于 data.shape[0]=8，本以为在做"重复索引累加"测试；
- 第二次（高重复累加复现）：data (100,)、indices (10000,)、dim=0、reduce=add——indices.shape[0]=10000 大于 data.shape[0]=100，本以为在做"万次重复命中同一位置"测试。

两次都误以为 kernel/tiling 出了问题，实际是**用例形状本身非法**，算子的 TilingFunc 形状校验把它拒了。

## 适用场景

- 工程形态：为 scatter 类自定义算子编写直连 aclnn 的 C++ 复现/测试程序（GetWorkspaceSize + aclnn 两段式调用）。
- API 路径：`aclnnXxxGetWorkspaceSize`；算子 TilingFunc 的 shape 校验（`gert::TilingContext`）。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有 scatter/scatter_nd/index_put 类算子（torch.scatter 同源约束）。
- 路线优先级：recommended_fastpath（诊断规则，先查用例再查 kernel）。
- 明确不适用 / 未验证场景：561002 在其他上下文（真 tiling bug、workspace 声明问题等）的语义未覆盖——本卡只断言"自建 scatter 用例 + 561002 → 先查形状约束"；其他算子的 TilingFunc 校验集各异。

## 判别方法

torch.scatter 的形状语义：`indices.shape == updates.shape`，且**对每一维 k ≠ dim 允许 indices.shape[k] ≤ data.shape[k]**（dim 维上 indices 的长度任意，它是散布操作的次数）。但**本算子 TilingFunc 的实际校验比 torch 更严**——逐维循环不跳过 dim，即 dim 维也要求 `indices.shape[dim] ≤ data.shape[dim]`（verbatim）：

```cpp
    // shape consistency: indices.shape == updates.shape, indices.shape[d] <= data.shape[d]
    for (int64_t k = 0; k < rank; k++) {
        if (indicesShape.GetDim(k) != updatesShape.GetDim(k)) {
            return ge::GRAPH_FAILED;
        }
        if (indicesShape.GetDim(k) > dataShape.GetDim(k)) {
            return ge::GRAPH_FAILED;
        }
    }
```

任一维违反即返回 GRAPH_FAILED，外层表现为 `GetWorkspaceSize` 返回 561002。上面两次触发分属两类：第一次（indices.shape[0]=2000 > data.shape[0]=8，dim=1）违反的是 k≠dim 维——按 torch 语义也非法；第二次（dim=0 下 indices.shape[0]=10000 > data.shape[0]=100）只违反 dim 维——按 torch.scatter 自由轴语义本是合法的"万次散布"用例，被本算子更严的全维 ≤ 校验拒绝。

可执行探针：逐维打印 `indices.shape[k]`、`data.shape[k]` 与 dim，核对目标算子 TilingFunc 的实际约束分支（读校验代码，不要按 torch 文档推断）；修正形状后同一测试应能走通（见验证信号）。

## 解释

561002 是 host 侧 TilingFunc 校验拒绝的对外表现，不是 kernel/tiling bug——自建复现用例撞上了算子的形状校验。构造"大量重复索引"类用例时，**按目标算子的实际校验设计形状**（本算子为全维 ≤）：重复命中应通过索引**值**（如全 0/全 t）配合各维 ≤ 的形状表达，而不是放大任一超界维；第一次那种放大非 dim 维的写法在 golden（numpy）侧同样非法（目标多维索引越界）。

## 预防

- 写 scatter 复现用例前先写下三件事：dim、indices.shape、data.shape，并**读目标算子 TilingFunc 的实际校验分支**逐维核对（不要按 torch 文档推断——算子侧校验可能更严）。
- 想构造 N 次重复命中同一目标：用各维 ≤ 的形状 + 重复的索引**值**（如 indices 与 data 同形、值全为 t，data.shape[dim] 只需 > t）；放大任何超界维（含 dim 维，对本算子）都会被 561002 拒绝。

## 验证信号

以下信号来自该算子的一次黑盒失败用例修复会话：

- 触发：两个独立自建用例（indices.shape[0]=2000 > data.shape[0]=8（dim=1）；indices.shape=(10000,) > data.shape=(100,)（dim=0））均在 GetWorkspaceSize 处返回 `ACLNN err 561002`（来源平台：950；证据等级：tool_observed）。
- 校验代码：TilingFunc shape 一致性循环（indices≠updates 或任一 k≠dim 维超界 → GRAPH_FAILED）在会话中读取算子源码核实（来源平台：950；证据等级：tool_observed）。
- 修正后：把重复索引测试改为合法形状（data(8,4096)、indices(8,4096) 全 0、dim=0，32768 次更新命中 4096 个目标），同用例 0/32768 失配，正常执行（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "561002 = 形状校验拒绝"的断言限于本算子（自研 TilingFunc 自带该校验）与本会话两次观测；同一错误码在其他算子/其他校验分支的语义未验证。
- 校验只查形状，不查 indices 的**值**范围（越界值由 kernel/GM 访问语义兜底）——不要把值越界问题也归到 561002。
- 本算子全维 ≤ 校验比 torch.scatter 语义更严（torch 的 dim 维是自由轴）——按 torch 语义设计的 dim 轴放大用例会被拒，属算子侧约束语义分歧，治理方向（校验跳过 dim 维）不在本卡；host 侧按 ≤ 语义正确实现校验的对照见 [Scatter 非散射维可缩小卡](../precision/scatter_update_count_must_follow_indices_shape.md)。
- 561002 的其他已见形态（占位张量 rank 不符、标量输入被赋非标量 shape、tiling 干扰）各有独立判别，见 [占位 1D 张量 rank 校验卡](aclnn_561002_rank1_placeholder_tensor_rank_check.md)、[标量输入非标量 shape 卡](../compilation/shape_561002.md)。

**未决项**：

- 无（诊断规则已闭环：触发→探针→修正→复跑）。
## 坏实践

- 见到 561002 就去翻 kernel/tiling 实现——先用一分钟逐维核对测试张量形状。
- 为凑重复索引把 indices 的非 dim 维拉大。
