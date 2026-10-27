---
type: "Runbook"
title: workspace 与 aclnn 保留名冲突：按孪生先例改名 storage
description: torch 侧 workspace 参数名与 CANN aclnn 二段接口的运行时保留形参 void *workspace 冲突；按 LSTM 孪生 aclnn 头与 910b 信息库先例改用 storage，需求期一次定名避免全链路返工。
tags: [task.design, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# workspace 与 aclnn 保留名冲突：按孪生先例改名 storage


## 现象

在为 ThnnFusedGruCellGrad（对标 PyTorch `aten::_thnn_fused_gru_cell_backward`）设计 CANN 侧接口时，撞上一个命名问题：torch 原算子的第二个输入参数名叫 `workspace`（本机 torch 2.7.1 头文件里的 schema 逐字为 `aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)`），而客户需求明确指出这个名字"与 CANN 内置参数冲突，须改用 storage"。

冲突的具体机制可以在族内先例——内置 LSTM 孪生算子的 aclnn 头文件（`<CANN>/x86_64-linux/include/aclnnop/level2/aclnn_thnn_fused_lstm_cell_backward.h`）里直接看到：该头文件的一段接口里算子自身的缓存输入就叫 `storage`，而二段接口的固定签名是 `aclnnThnnFusedLstmCellBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`——`workspace` 这个名字已经被 aclnn 二段式接口的**运行时 workspace 指针形参**占用了。如果算子输入也叫 `workspace`，生成的 aclnn 接口就会出现两个同名参数（算子输入 `aclTensor *workspace` 与运行时形参 `void *workspace`），语义与调用约定都无法区分。

进一步核对生态先例：ascend910b 信息库中 `ThnnFusedLstmCellGrad` 的 `input4` 名字就是 `storage`（dtype float32,float16, required）；本机安装的前代 ThnnFusedGruCellGrad vendor 包 IR 也是 `.INPUT(storage, ...)`——三处（aclnn 头、信息库、前代安装包）一致指向 `storage`。

## 适用场景

- **工程形态**：算子仓开发（aclnn 二段式接口）需求/规格期的输入命名决策；任何从 aten 算子移植到 CANN 时的参数名冲突排查。
- **API 路径**：aclnn 二段式接口（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）、`REG_OP` IR 定义、分 SoC 信息库、aten schema。
- **验证过的平台与 CANN 版本**：Ascend 950PR 本机 + CANN 9.0.0-beta.2（aclnn 头与信息库均为该安装内的包级事实；与 frontmatter `platforms` 一致；CANN 版本以正文实测记录为准）。
- **适用粒度（transfer_scope）**：`operator_family`（THNN fused cell 族的既成命名先例）；命名三查方法本身 `workflow_only` 可推广。
- **路线优先级**：`recommended_fastpath`（需求期一次定名）。
- **明确不适用/未验证场景**：非 aclnn 交付形态（如 Kernel 直调不走 aclnn 二段接口，无此冲突）；其他命名冲突类型（如与属性名、保留关键字的冲突）须另行排查。

## 判别方法

输入命名**三查**，缺一不可：

1. **查 aten 原名**：从本地 torch 头文件拿 schema 逐字原文（不要凭记忆写参数名）：
   ```bash
   cat /opt/buildtools/python-3.12.9/lib/python3.12/site-packages/torch/include/ATen/ops/_thnn_fused_gru_cell_backward.h
   # aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor×5)
   ```
2. **查 CANN 保留名**：看同族（或任意）aclnn 头的二段接口形参表，确认目标名字是否撞上 aclnn 运行时保留形参：
   ```bash
   cat <CANN>/x86_64-linux/include/aclnnop/level2/aclnn_thnn_fused_lstm_cell_backward.h
   ```
3. **查生态先例**：同族内置算子在 aclnn 头 + 信息库 + 已安装 vendor 包里实际用的名字：
   ```bash
   python3 -c "
   import json
   d=json.load(open('<...>/tbe/config/ascend910b/aic-ascend910b-ops-info-nn.json'))
   e=d['ThnnFusedLstmCellGrad']
   for k,v in sorted(e.items()):
       if k.startswith('input'): print(k, v.get('name'), v.get('dtype'), v.get('paramType'))
   "
   # 实测输出：input0 dhy ... / input4 storage float32,float16 required
   ```

## 根因

aclnn 二段式接口的第二段固定形参就是 `void *workspace, uint64_t workspaceSize`（运行时 workspace 指针，每个 aclnn 算子接口都有）；aten 侧恰好把融合前向的中间量缓存张量命名为 `workspace` 并作为反向的输入。两个命名空间在生成 C 接口时重叠：算子输入 `workspace` 与运行时形参 `workspace` 同名冲突——torch 的参数名不能原样进 CANN IR/aclnn 接口，必须在需求/规格期改名并锚定生态先例（`storage`），否则后期改名会沿真值文档向下游（spec/设计/kernel/测试/文档）全链路传播。

## 设计决策

- **为什么选 `storage` 而不是自造名**：`storage` 是族内（ThnnFusedLstmCell/Grad）已在 aclnn 头、910b 信息库、前代 vendor 安装三处物化的既成命名——跟随先例让新算子与 LSTM 系列在接口层可对照，也避免了"自造名后来发现与生态冲突再返工"。
- **放弃的候选**：保留 `workspace`（冲突，不可行）；自造名如 `gru_aux`（前次开发实际走过的一版名字——交接文档记载该次改名造成真值 3 文档→下游 8 路的全链路返工；该记载为本会话转述、未复现，但机制上说明了后期改名的代价结构）。
- **当前方案档位**：`recommended_fastpath`（需求期三查一次定名）。

## 系统信息

- 环境：Ascend 950PR + CANN 9.0.0-beta.2、torch 2.7.1+cpu。
- 关键文件：
  - `torch/include/ATen/ops/_thnn_fused_gru_cell_backward.h`（aten schema）
  - `<CANN>/x86_64-linux/include/aclnnop/level2/aclnn_thnn_fused_lstm_cell_backward.h`（冲突机制 + storage 先例）
  - `<CANN>/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info-nn.json`（input4 storage）
  - `$ASCEND_OPP_PATH/vendors/ThnnFusedGruCellGrad/op_proto/inc/op_proto.h`（前代安装包 `.INPUT(storage, ...)`）

## 完整代码

LSTM 孪生 aclnn 头关键段（逐字，脱敏路径）——一段接口的算子输入 `storage` 与二段接口的运行时形参 `void *workspace` 并存，即冲突机制的直接证据：

```c
ACLNN_API aclnnStatus aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(
    const aclTensor *gradHyOptional,
    const aclTensor *gradCOptional,
    const aclTensor *cx,
    const aclTensor *cy,
    const aclTensor *storage,
    bool hasBias,
    aclTensor *gradGatesOut,
    aclTensor *gradCxOut,
    aclTensor *gradBiasOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnThnnFusedLstmCellBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

前代 vendor 包 IR（逐字）——改名决策的第三处先例：

```cpp
REG_OP(ThnnFusedGruCellGrad)
    .INPUT(grad_hy, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(storage, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_hx, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_input_gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_hidden_gates, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_input_bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_hidden_bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(has_bias, Bool, false)
    .OP_END_FACTORY_REG(ThnnFusedGruCellGrad)
```

## 预防

- 移植 aten 算子时，参数名先过三查（aten 原名 → CANN/aclnn 保留名 → 族内生态先例），在需求文档里把改名决策连同先例证据一次写死，不要留到 spec 或实现期"发现冲突再说"。
- 高危名字清单意识：`workspace`（aclnn 二段接口保留）、以及其他与 aclnn 运行时概念可能撞名的词，宁可先查一遍族内先例。

## 验证信号

以下均为本次开发会话内、950PR 本机 CANN 9.0.0-beta.2 安装上的实测：

- aten schema 逐字核验：`(Tensor grad_hy, Tensor workspace, bool has_bias) -> 5 tensors`（来源平台：950（本机 torch 2.7.1 头文件）；证据等级：tool_observed）。
- LSTM 孪生 aclnn 头中算子输入名 `storage` 与二段接口形参 `void *workspace` 并存（来源平台：950（本机 CANN 9.0.0-beta.2 安装）；证据等级：tool_observed）。
- ascend910b 信息库 `ThnnFusedLstmCellGrad` 的 input4 为 `storage`（来源平台：950（本机 CANN 包内 ascend910b 信息库文件，未在 910b 实机验证）；证据等级：tool_observed）。
- 前代 vendor 安装包 IR `.INPUT(storage, ...)`（来源平台：950（本机 CANN 安装）；证据等级：tool_observed）。

## 失效边界

- 冲突机制绑定 aclnn 二段式接口形态；若交付形态不是 aclnn（如纯 Kernel 直调），`workspace` 参数名不构成同类冲突。
- `storage` 先例来自 THNN fused cell 族；移植其他族算子时先例须重新检索，不能默认 `storage` 通用。
- 前次开发"gru_aux 中间名→全链路返工"的代价记载为交接文档转述（本会话未复现），只作决策参考、不作实测信号。

## 坏实践

- 参数名直接照抄 aten 原名进 CANN IR，等编译/接口生成期才撞保留名。
- 改名时不查族内先例、自造名字，导致同一族算子（LSTM 用 `storage`、GRU 用别的）接口不可对照。

## 误诊

- 把冲突误解为"与某个内置**算子**同名冲突"——实际是与 aclnn **接口层的运行时保留形参**命名空间冲突（证据：LSTM 孪生头文件里两者并存）。

## 未决事项

- 无（本卡结论（改名 `storage` + 三查方法）在本会话内已闭环到需求文档的接口设计章节；唯一开放项是最终 aclnn 导出名 `aclnnThnnFusedGruCellGrad` 与 LSTM 孪生的 `...Backward` 后缀差异，属 spec 期命名规范决策，不属本卡根因）。
