---
type: "Runbook"
title: ValueDepend 虚影 dtype 组合与 Follow 输出冲突
description: "def 中给输入声明 ValueDepend(REQUIRED, DependScope::TILING) 后 opbuild 要求额外虚影 dtype 组合（仅该输入换 dtype、其余不变）；输出 Follow 输入时组合不存在，编译不过；移除 ValueDepend 解锁编译但 tiling 读数据有运行时风险，须配套 Device 侧计算。"
tags: [task.implementation, topic.compilation, topic.tiling]
status: stable
sources: []
created_at: '2026-09-16T10:17:47Z'
updated_at: '2026-09-16T10:17:47Z'
platforms: ["950"]
---

# ValueDepend 虚影 dtype 组合与 Follow 输出冲突

## 现象

MoE 重排自定义算子（BenchNoHarnessMoeReRouting，Ascend950PR/ascend950，CANN 9.1.0）的 tiling 函数需要读 `expert_token_num_per_rank`（(N,E) 计数张量）的**数据**来计算块调度（srcOffset/dstOffset 前缀和），因此在 `op_host/*_def.cpp` 中给该输入声明了数据依赖：

```cpp
.ValueDepend(REQUIRED, DependScope::TILING)
```

在把各输入/输出 DataType 列表修成 6 组等长组合（3 token dtype × 2 num dtype）之后执行 `bash build.sh --soc=ascend950 -j8`，opbuild 仍失败，报错原文：

```
opbuild ops error: One combination of input and output dtypes of op
BenchNoHarnessMoeReRouting is [tokens: ge::DT_FLOAT16,
expert_token_num_per_rank: ge::DT_INT32, per_token_scales: ge::DT_FLOAT,
permute_tokens: ge::DT_FLOAT16, permute_per_token_scales: ge::DT_FLOAT,
permute_token_idx: ge::DT_INT32, expert_token_num: ge::DT_INT32], the
combinations of input and output dtypes should add [tokens: ge::DT_FLOAT16,
expert_token_num_per_rank: ge::DT_INT64, per_token_scales: ge::DT_FLOAT,
permute_tokens: ge::DT_FLOAT16, permute_per_token_scales: ge::DT_FLOAT,
permute_token_idx: ge::DT_INT32, expert_token_num: ge::DT_INT32], because
of the ValueDepend input expert_token_num_per_rank.
```

关键点：要求补的组合里 `expert_token_num_per_rank` 换成 int64 而 `expert_token_num` 输出**仍是 int32**——而该算子语义要求输出 dtype 跟随输入（int64 输入 → int64 输出），这种"错位"组合在语义上本就不该存在。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子开发，tiling 阶段需要读输入张量数据（数据依赖）的算子。
- API 路径：`ops::OpParamDef::ValueDepend(Option, DependScope::TILING)`、`gert::TilingContext::GetInputTensor(i)->GetData<T>()`、`.Follow("input_name")`、opbuild 工具。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950/arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"tiling 读输入数据 + 该输入 dtype 被某输出 Follow"的算子。
- 路线优先级：safety_fallback（移除 ValueDepend 是为解锁编译的回退做法，带明确运行时风险，见下）。
- 明确不适用 / 未验证场景：ValueDepend 输入的 dtype 不影响任何输出 dtype 的算子（虚影组合可以显式补齐时也许能保住 ValueDepend，本轨迹未走通该路线）；没有 Follow 输出跟随该输入的算子；CANN 9.1.0 之外的版本（9.0/9.2 未验证）。

## 根因

CANN 9.1.0 的 opbuild 对 `ValueDepend` 输入做"虚影组合"展开：对每个已声明的 dtype 组合，框架额外要求存在**仅 ValueDepend 输入换 dtype、其余输入输出全部不变**的变体组合（因为数据依赖输入只影响 tiling 计算，框架默认它不影响其他张量的 dtype）。本算子 `expert_token_num` 输出用 `.Follow("expert_token_num_per_rank")` 让 dtype 跟随输入，于是"num=int64 输入 + expert_token_num=int32 输出"的虚影组合永远不可能出现在声明里——显式列出 6 组语义正确的组合（int64 输入必配 int64 输出）也满足不了，Follow 也不能消除该要求，编译必然失败。

tiling 侧读数据的代码（tiling.cpp，移除 ValueDepend 后这段就是风险点）：

```cpp
    // 3. 读取 expert_token_num_per_rank 数据
    auto numTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, numTensor);

    std::vector<int64_t> matrix(N * E);
    if (numDtype == ge::DT_INT32) {
        const int32_t* data = numTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, data);
        for (uint32_t i = 0; i < N * E; i++) {
            matrix[i] = static_cast<int64_t>(data[i]);
        }
    } else if (numDtype == ge::DT_INT64) {
        const int64_t* data = numTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, data);
        for (uint32_t i = 0; i < N * E; i++) {
            matrix[i] = data[i];
        }
    } else {
        OP_LOGE(context, "Unsupported numDtype: %d", static_cast<int>(numDtype));
        return ge::GRAPH_FAILED;
    }
```

## 解决配方

尝试一（失败）：给 `expert_token_num` 输出改用 Follow 跟随输入 dtype——

```cpp
        this->Output("expert_token_num")
            .ParamType(REQUIRED)
            .Follow("expert_token_num_per_rank")
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
```

重编后报错一字不差（虚影组合仍要求 num=int64 + 输出=int32）。Follow 无法表达"错位"组合，此路不通。

最终修复（移除 ValueDepend，safety_fallback）——把 `expert_token_num_per_rank` 输入声明里的数据依赖行删掉：

```cpp
        this->Input("expert_token_num_per_rank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
```

（保留 6 组等长 dtype 组合与 `expert_token_num` 输出的 Follow；只删 `.ValueDepend(REQUIRED, DependScope::TILING)` 一行。）

同时把该问题连同风险与后续计划写入算子仓 issue 文件（ValueDepend 导致 opbuild dtype 组合冲突）。

**fallback 声明：移除 ValueDepend 只是解锁编译，不是无损修复。** 按 TilingContext 语义（"若算子被配置为'tiling'数据依赖，则返回的 Tensor 对象中保存了 Host 内存地址；反之，内存地址为 nullptr"），没有 ValueDepend 时 `GetInputTensor(1)->GetData<T>()` 在真实运行时可能拿到 nullptr——tiling 读不到计数数据就无法算块调度。必须配套实施"Device 侧计算 schedule"方案（Host 仅传标量、kernel 从 GM 读计数自行算块调度），不能只删配置了事；若 tiling 侧仍保留 GetData 调用，会直接引入运行时段错误。

## 设计决策

- 为什么选移除而不是补 12 组错位组合：显式补齐虚影组合（num=int64+输出=int32 等）会让 op def 声明出语义上非法的调用组合，框架可能据此生成错误的 kernel 分发，代价大于收益。
- 为什么保留 Follow：输出 dtype 跟随输入是算子语义（int64 计数 → int64 输出），Follow 是表达该语义的正确方式，与 ValueDepend 冲突的是后者不是前者。
- 后备方案（设计文档既有）：Host 仅传标量、Device 侧从 GM 读 expert_token_num_per_rank 自行计算块调度（O(N×E) 标量计算，≤16384 次整数加法，可忽略）——若 ST Real 验证确认 tiling 读不到数据，则切换该路线。
- 当前方案路线档位：safety_fallback（编译解锁优先，运行时风险挂 issue 跟踪）。

## 预防

- def 中同时使用 ValueDepend 与 Follow 之前，先确认两者作用于不同输入/输出；同一输入既是 ValueDepend 又被输出 Follow 时，在 CANN 9.1.0 上必然冲突。
- tiling 要读输入数据 + 该输入 dtype 影响某输出 dtype 的算子，在 CANN 9.1.0 上不要直接上 `ValueDepend(REQUIRED, DependScope::TILING)`——先用最小 def 试编译，命中本冲突就早切后备方案（Device 侧读 GM）。
- 为 tiling 数据依赖设计备选方案（Device 侧计算）并写入设计文档，遇到工具链冲突时可直接切换。
- 遇到 "the combinations of input and output dtypes should add [...], because of the ValueDepend input X" 时，先看要求补的组合是否语义非法（错位 dtype）；非法则不要硬补。
- 移除 ValueDepend 后必须检查 tiling 函数中是否还有 `GetData` 调用——不检查会直接引入运行时段错误（未声明依赖时 TilingFunc 读输入数据的段错误形态见库内 TilingFunc 未声明依赖读 axes 致 segfault 卡）。

## 验证信号

以下信号来自该 MoE 重排算子骨架编译会话（Ascend950PR，CANN 9.1.0）。

- 冲突复现：6 组等长组合 + ValueDepend 声明下，opbuild 稳定报 "One combination of input and output dtypes ... should add [... int64 ... expert_token_num: ge::DT_INT32], because of the ValueDepend input expert_token_num_per_rank"（来源平台：950；证据等级：tool_observed）。
- Follow 尝试失败：expert_token_num 改 `.Follow("expert_token_num_per_rank")` 后重编，报错原文不变（来源平台：950；证据等级：tool_observed）。
- 移除 ValueDepend 后：`bash build.sh --make_clean && bash build.sh --soc=ascend950 -j8` 编译通过，生成 6 个 TilingKey（3 token dtype × 2 has_scales）的 kernel 二进制（`build/op_kernel/ascendc_kernels/binary/ascend950/.../*.o`）与自定义算子包 custom_opp_ubuntu_x86_64.run（来源平台：950；证据等级：tool_observed）。
  运行时 tiling 数据读取是否正常属**未验证**项（ST Real 模式在后续阶段；不能宣称移除 ValueDepend 无副作用，见失效边界与遗留问题）。

## 失效边界

- 本卡结论限定 CANN 9.1.0 的 opbuild 行为；后续版本若修复了 ValueDepend 与 Follow 的交互则不适用（未验证）。
- 移除 ValueDepend 后 tiling 读数据是否真的拿到 nullptr 取决于运行时链路（aclnn 单算子调用 vs 入图），本轨迹均未实测。
- "虚影组合"机制解释来自报错文本与两轮因果实验（Follow 不消错、移除即过），opbuild 源码未读。
- 仅覆盖「ValueDepend 输入被 Follow」这一冲突形态；ValueDepend 输入无 Follow 输出时是否正常工作，本轨迹未涉及。

**未决项**：

- 优先项：ST Real 模式上板验证移除 ValueDepend 后 `GetInputTensor(1)` 的数据指针是否为 nullptr；若为 nullptr，切换设计后备方案（Host 传标量 + Device 从 GM 读计数自行算块调度）。
- 次优项：验证"显式补齐 12 组含错位组合"是否真能保住 ValueDepend（若框架接受，可与运行时 dtype 校验配合使用）。
- 读 opbuild 源码确认虚影组合的确切生成规则；`const input` 标记等其他启用 tiling 数据依赖的方式未尝试。
## 坏实践

- 见到 "should add [...]" 就机械补齐要求的组合——补出来的错位组合（int64 输入 + int32 输出）在语义上是非法调用，等于向框架注册了不存在的算子行为。
- 移除 ValueDepend 后不挂 issue、不做运行时验证就宣称闭环——tiling 读数据空指针会在 ST 阶段才爆出来。

## 误诊

- 曾以为 Follow 能表达"输出跟随输入"从而消除组合需求——opbuild 对 ValueDepend 的虚影组合要求与 Follow 语义正交，改 Follow 前后报错逐字相同。
