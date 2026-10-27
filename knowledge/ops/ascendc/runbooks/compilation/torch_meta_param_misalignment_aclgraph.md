---
type: "Runbook"
title: Meta 校验误传 data 致 ACLGraph 浮点全失败
description: 自定义 torch op 的 Meta 实现把 data 同时当三个参数传给校验函数，ACLGraph（torch.compile FakeTensor 捕获）下 indices dtype 校验实际校验的是 data 的 dtype——整型用例碰巧通过、浮点用例全部 GRAPH_EXEC_FAILURE；Meta 须转发真实 indices/updates。
tags: [task.debugging, topic.graph_execution, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-14T12:12:55Z'
updated_at: '2026-09-14T12:12:55Z'
platforms: ["950"]
---

# Meta 校验误传 data 致 ACLGraph 浮点全失败

## 现象

在 Ascend950PR（CANN 9.2.0）上对自定义算子的 Torch 路由做 TTK e2e 双模式执行：同一个 ttk_cases.csv（20 用例：17 个正向 + 3 个 L2 负向），eager 模式 17 个正向全部 PASS（浮点 mere=0.0/mare=0.0，整型 binary_equal）；切到 ACLGraph 模式（`ttk e2e --aclgraph`，经 torch.compile reduce-overhead 捕获）后首轮结果变成 **20 total / 4 PASS / 16 FAIL**——通过的恰好只有 4 个整型 dtype 组合（int32/int32、int32/int64、int64/int32、int64/int64，均 100%），而**全部浮点（fp16/fp32/bf16）正向用例 GRAPH_EXEC_FAILURE**。

对单个 fp32 用例（data (8,8) fp32、indices (4,4) int32、updates (4,4) fp32、dim=0、reduce=update，eager 下已 PASS）单独重跑 aclgraph 抓日志，报错原文：

```text
E0904 20:49:36.615000 ... torch/_subclasses/fake_tensor.py:2431] [0/0] failed while attempting to run meta for glacier_bench_harness_scatter.forward.default
E0904 20:49:36.615000 ... RuntimeError: indices dtype must be int32 or int64
...
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function glacier_bench_harness_scatter.forward(*(FakeTensor(..., device='npu:0', size=(8, 8)), 0, FakeTensor(..., device='npu:0', size=(4, 4), dtype=torch.int32), FakeTensor(..., size=(4, 4)), 'update'), **{}): got RuntimeError('indices dtype must be int32 or int64')
```

indices 明明是 int32，报错却说 indices dtype 非法。当时 adapter 的 Meta 实现写的是 `CheckInputs(data, data, data)`。

## 适用场景

- 工程形态：aclnn 自定义算子的 TORCH_LIBRARY adapter（`torch.ops.load_library` 注册，Meta + PrivateUse1 双实现）；TTK e2e `--aclgraph` / 任何 torch.compile 图捕获路径。
- API 路径：`TORCH_LIBRARY_IMPL(<ns>, Meta, module)` 注册的 Meta kernel；torch._dynamo 以 FakeTensor 运行 Meta 做 shape 推导；eager 路径只走 PrivateUse1（NPU）实现。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（torch 2.x + torch_npu；TTK 3.0.0 e2e --aclgraph）（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——凡给自定义 torch op 写 Meta 实现并做参数校验的场景，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：只有 PrivateUse1 实现、从不进图捕获的 op（Meta 不被调用）；Meta 完全不做校验（只 empty_like）的写法不会触发本缺陷，但也丢失了捕获期前置校验；TF/GEIR 等非 torch.compile 捕获路径未验证。

## 根因

torch.compile/ACLGraph 捕获阶段，Dynamo 用 **FakeTensor 运行自定义 op 的 Meta 实现**做元数据推导；eager 模式不经过 Meta（直接调 PrivateUse1 NPU 实现），所以 eager 全过、图捕获才暴露。缺陷 Meta 实现把同一个 data 张量当作三个参数传给输入校验：

```cpp
CheckInputs(data, data, data);  // rank/dtype sanity only
```

`CheckInputs` 第 2 形参是 indices，其中校验 `indices.scalar_type() == Int || Long`。传 data 进去等于校验 **data 的 dtype** 是否 int32/int64：整型 data 碰巧通过（所以 4 个整型用例 PASS），浮点 data 必然失败（所以全部浮点用例 GRAPH_EXEC_FAILURE）——与"恰好只有整型组合通过"的失败分布完全吻合，报错文案"indices dtype"则来自校验函数的真实分支，具误导性。

## 解决配方

修复前（run.sh 内嵌 adapter 的 Meta 实现，函数级完整）：

```cpp
torch::Tensor Meta(const torch::Tensor &data, int64_t,
                   const torch::Tensor &, const torch::Tensor &,
                   c10::optional<std::string>)
{
    CheckInputs(data, data, data);  // rank/dtype sanity only
    return torch::empty_like(data);
}
```

修复后（转发真实 indices/updates；FakeTensor 支持 `scalar_type()`/`dim()` 等元数据查询，校验可直接复用）：

```cpp
torch::Tensor Meta(const torch::Tensor &data, int64_t,
                   const torch::Tensor &indices, const torch::Tensor &updates,
                   c10::optional<std::string>)
{
    CheckInputs(data, indices, updates);  // rank/dtype sanity (works on FakeTensors)
    return torch::empty_like(data);
}
```

## 设计决策

- Meta 保留与 Npu 相同的 CheckInputs（而不是删掉校验只留 empty_like）：图捕获期就能拒绝非法输入，错误在 trace 阶段暴露而不是运行期；代价为零（FakeTensor 元数据查询便宜）。
- 修复选择"转发真实参数"而不是"放宽 Meta 校验"：eager 与图捕获两条路径的校验语义保持一致，避免"图模式校验弱于 eager"的隐性分叉。
- 当前方案路线档位：recommended_fastpath（一处参数转发修正）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0；torch 2.x + torch_npu（PrivateUse1 后端）；TTK 3.0.0（e2e --aclgraph）。
- 相关文件：`<op>_package/tests/torch/run.sh`（内嵌 adapter 的 Meta 实现，本修复位置）；TTK `graph_execution.py`/`torch/_subclasses/fake_tensor.py`（捕获路径报错来源，只读）。

## 预防

- 自定义 torch op 的 Meta 实现必须与 Npu 实现做同样的参数校验，且**用真实形参**——图捕获（torch.compile/aclgraph/任何 FakeTensor trace）都会走 Meta，eager 测试通过不代表 Meta 正确。
- 现象判别：**图模式挂、eager 过、且失败按 data dtype 分布（整型过、浮点挂）**——优先怀疑 Meta 实现里参数错位/复用了错误张量做校验。
- 写 Meta 时把未具名参数（`const torch::Tensor &`）一律具名并使用，可让编译器/评审直接看出参数消费关系。
- 相邻坑区分：TORCH_LIBRARY 注册期因返回类型推断不符直接 abort（多输出返回 std::vector）是另一独立根因，见 [TORCH_LIBRARY 多输出返回须 std::tuple 非 vector](torch_library_multi_output_return_tuple_not_vector.md)——彼卡挂在 dlopen 注册期，本卡挂在 torch.compile 捕获期。

## 验证信号

以下信号来自该算子 Torch 路由搭建会话：Meta 修复前后的 TTK e2e aclgraph 实测。

- 修复前：aclgraph 全量 20 用例仅 4 个整型组合 PASS（binary_equal 100%），12 个浮点正向 + 3 个 L2 负向共 16 个 GRAPH_EXEC_FAILURE；单用例调试日志显示 Dynamo 以 FakeTensor 运行 `glacier_bench_harness_scatter.forward.default` 的 meta 失败于 `RuntimeError: indices dtype must be int32 or int64`（来源平台：950；证据等级：tool_observed）。
- 修复后：重建 adapter 直调通过（`direct_call_output: [10.0, 2.0, 30.0, 4.0]`）；原失败 fp32 用例 aclgraph 冒烟 `GRAPH ACLGRAPH: PASS / STATUS: PASS / PassRate: 100.00% (1/1)`（来源平台：950；证据等级：tool_observed）。
- 修复后：全量 aclgraph 重跑 20 total / 17 PASS / 3 FAIL——17 个正向全部 PASS（浮点 mere=0.0/mare=0.0、整型 binary_equal），与 eager 模式逐用例一致；3 个 L2 负向保持 GRAPH_EXEC_FAILURE（预期拒绝）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本例根因在 adapter（测试脚手架）的 Meta 实现参数错位，不在算子 kernel 或 torch_npu 图机制；若自定义 op 无 Meta 实现且 op 声明含自定义 schema，torch.compile 会用 schema 推导或直接报不支持，行为不同。
- FakeTensor 支持元数据查询（dtype/shape/dim），但不支持 data_ptr/真实数据访问——Meta 里做不了数值级校验。
- 机制是 torch Meta kernel 语义层面的，推断可迁移至任何平台的自定义 torch op adapter（未验证，推断可迁移）。

**未决项**：

- 无（根因已追到 Meta kernel 参数错位 + FakeTensor 捕获机制层面并闭环验证）。
## 坏实践

- Meta 实现里"偷懒"用 data（或任意已有张量）顶替 indices/updates 做校验——校验语义静默错位，只在图捕获下暴露。
- 图模式失败时不抓单用例调试日志就归因于"aclgraph 对自定义 op 不兼容"——本例同一 op 同一捕获路径整型用例全过，足以反证。

## 误诊

- "浮点用例失败 = kernel 对浮点 dtype 有问题"：eager 模式同用例全过（mere=0.0），纯图捕获路径差异即可排除。
- "GRAPH_EXEC_FAILURE = aclgraph/torch.compile 不支持自定义 TORCH_LIBRARY op"：4 个整型用例同一 op 同一捕获路径 PASS，说明捕获机制本身可用。
