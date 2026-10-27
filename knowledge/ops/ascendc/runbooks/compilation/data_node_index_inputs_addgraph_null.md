---
type: "Runbook"
title: Data 节点缺 index 未收集致 AddGraph 失败
description: "GE IR 构图时 op::Data 节点未 set_attr_index 也未收集进 inputs，main 的 SetInputs/SetOutputs 被空判断跳过，AddGraph 报 inner_session.cc SetSessionGraphId: null is invalid（error code 1343225857）；Data 节点补 index 属性并收集进 inputs、主算子节点交给 SetOutputs 隐式入图后 AddGraph 成功。"
tags: [task.debugging, topic.graph_execution]
status: stable
sources: []
created_at: '2026-09-17T04:13:22Z'
updated_at: '2026-09-17T04:13:22Z'
platforms: ["950"]
---

# Data 节点缺 index 未收集致 AddGraph 失败

## 现象

自定义算子（depthwise conv）的 GE IR 图模式调用示例（Ascend950PR，CANN 9.2.0）首次运行：编译通过、`GEInitialize` 成功，但 `session->AddGraph(graph_id, graph)` 失败返回。plog 日志（`~/ascend/log/debug/plog/plog-*.log`）里的报错完全没提"inputs/outputs"：

```text
[ERROR] GE(...,test_geir_...): ... [inner_session.cc:348]... SetSessionGraphId: ErrorNo: 4294967295(failed)[Check][Param:compute_graph]null is invalid
[ERROR] GE(...) [inner_session.cc:327]... AddGraph: ErrorNo: 4294967295(failed)Assert ((SetSessionGraphId(graph, session_id_, graph_id)) == ge::SUCCESS) failed
[ERROR] GE(...) [ge_api.cc:440]... AddGraph: ErrorNo: 4294967295(failed)Add graph failed, error code:1343225857, session_id:0, graph_id:0.
```

当时代码状态：构图函数 `CreateOppInGraph(...)` 里为 x/weight/bias 各建了 `op::Data` 节点并 `graph.AddOp(dataNode)`、`dwConvOp.set_input_x(xData)` 等，但 Data 节点没有 `set_attr_index(...)`，也没有任何节点被 push 进 `inputs` 参数列表；main 里的 `graph.SetInputs(inputs).SetOutputs(outputs)` 被 `if (!inputs.empty() && !outputs.empty())` 守卫，因 `inputs` 为空而整体跳过，提交给 AddGraph 的 compute_graph 是空的。

## 适用场景

- 工程形态：GE IR 图模式调用示例/独立图执行程序（`ge::Graph` + `op::Data` 输入节点 + 自定义算子节点 + `ge::Session::AddGraph`）。
- API 路径：`graph/graph.h`（`Graph::SetInputs/SetOutputs/AddOp`）、`op::Data::set_attr_index`、参考实现 add_example 的 ADD_INPUT/ADD_OUTPUT 宏模式。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——GE IR 构图契约，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：Const 输入（非 Data 节点）构图、多输出构图未在本例验证；报错同码但由 op_proto 未加载等其他原因使 compute_graph 为空的 AddGraph 失败不在本卡范围。

## 根因

构图时 `op::Data` 节点既没有 `set_attr_index(n)` 标记图输入序号、也没有被收集进 `inputs` 列表，导致 main 中 `graph.SetInputs(inputs).SetOutputs(outputs)` 因 inputs 为空被跳过、compute_graph 未成型，AddGraph 报 `SetSessionGraphId: null is invalid`。

## 解决配方

修复前（Data 节点无 index、不收集 inputs；主算子节点显式 AddOp）：

```cpp
    // 创建输入 Data 节点：x [N,C,H,W]
    auto xData = op::Data("x");
    vector<int64_t> xShape = {N, C, H, W};
    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);
    Tensor xTensor;
    if (GenFp16Data(xShape, xTensor, xDesc, -1.0f, 1.0f, 42) != SUCCESS) {
        LOG_PRINT("%s - ERROR: Generate x data failed\n", GetTime().c_str());
        return FAILED;
    }
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    input.push_back(xTensor);
    graph.AddOp(xData);
    dwConvOp.set_input_x(xData);

    // （weight/bias 同构，均无 set_attr_index、均不 push 进 inputs）

    outputs.push_back(dwConvOp);
    graph.AddOp(dwConvOp);

    return SUCCESS;
}
```

修复后（Data 节点带 index 属性并收集进 inputs；主算子节点不显式 AddOp，由 SetOutputs 隐式添加）：

```cpp
    // 创建输入 Data 节点：x [N,C,H,W]
    auto xData = op::Data("x").set_attr_index(0);
    vector<int64_t> xShape = {N, C, H, W};
    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, inDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    xDesc.SetFormat(FORMAT_ND);
    Tensor xTensor;
    if (GenFp16Data(xShape, xTensor, xDesc, -1.0f, 1.0f, 42) != SUCCESS) {
        LOG_PRINT("%s - ERROR: Generate x data failed\n", GetTime().c_str());
        return FAILED;
    }
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    input.push_back(xTensor);
    graph.AddOp(xData);
    dwConvOp.set_input_x(xData);
    inputs.push_back(xData);

    // 创建输入 Data 节点：weight [C,Kh,Kw]
    auto wData = op::Data("weight").set_attr_index(1);
    // ... 同构：GenFp16Data / update_input_desc_x / update_output_desc_y /
    //     input.push_back(wTensor) / graph.AddOp(wData) /
    //     dwConvOp.set_input_weight(wData) / inputs.push_back(wData)

    // 创建输入 Data 节点：bias [C]
    auto bData = op::Data("bias").set_attr_index(2);
    // ... 同构 ... inputs.push_back(bData);

    // 设置输出描述：y [N,C,H_out,W_out]
    vector<int64_t> yShape = {N, C, H_out, W_out};
    TensorDesc yDesc(ge::Shape(yShape), FORMAT_ND, inDtype);
    dwConvOp.update_output_desc_y(yDesc);

    outputs.push_back(dwConvOp);
    // 注意：不要显式 graph.AddOp(dwConvOp)，由 graph.SetOutputs(outputs) 隐式添加

    return SUCCESS;
}
```

main 侧（修复前后不变，但修复前因 inputs 为空而整体被跳过）：

```cpp
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    LOG_PRINT("%s - INFO: Create session\n", GetTime().c_str());
    std::map<AscendString, AscendString> build_options;
    ge::Session* session = new Session(build_options);
    ...
    ret = session->AddGraph(graph_id, graph);
```

## 设计决策

- 对照 add_example 参考实现的 ADD_INPUT 宏发现其模式为：`graph.AddOp(dataNode); op.set_input_x(dataNode);` 且 Data 节点带 index 属性并被收入 inputs；主算子节点从不显式 `graph.AddOp`，仅 push 进 outputs 由 `SetOutputs` 隐式入图——本例初版图省事漏掉了 index 属性与 inputs 收集，还多加了显式 AddOp。
- 当前方案（完全对齐参考模式）属 recommended_fastpath。

## 预防

- 写 GE IR 构图代码时按参考模式自查三件：每个 `op::Data` 节点 `set_attr_index(n)` 与输入张量序号一致；每个 Data 节点 push 进 `inputs`；主算子节点只 push 进 `outputs` 不显式 AddOp。
- 构图函数签名里带 `std::vector<Operator>& inputs` 出参就必须填充——依赖 main 侧空判断静默跳过会让错误延迟到 AddGraph 才暴露。

## 验证信号

来源说明：问题与验证来自该算子 GE IR 示例开发会话——AddGraph 失败后按 plog 定位、重构构图函数、重跑验证。

- 修复前 plog：`inner_session.cc:348 SetSessionGraphId: ... [Check][Param:compute_graph]null is invalid` + `ge_api.cc:440 Add graph failed, error code:1343225857`（来源平台：950；证据等级：tool_observed）。
- 修复后：`INFO: AddGraph success`（进入 RunGraph 阶段；RunGraph 的后续失败属另一独立问题，见缺 InferDataType 注册相关卡）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅在 Ascend950PR + CANN 9.2.0 实测；GE 构图 API 语义推断可迁移（未验证，推断可迁移）。
- `SetSessionGraphId: null is invalid` 也可能由其他原因使 compute_graph 为空（例如 Graph 对象构造失败）；本卡结论限定在"Data 节点缺 index/inputs 收集导致 SetInputs 被跳过"这一成因。
- 空判断 `if (!inputs.empty() && !outputs.empty())` 是参考示例写法，静默跳过 SetInputs/SetOutputs 会让错误延迟到 AddGraph 才暴露——排障时先检查该守卫是否被执行。

**未决项**：

- 无（根因已追到构图契约层）。
## 坏实践

- 构图函数签名里带着 `std::vector<Operator>& inputs` 参数却从不填充，依赖 main 侧空判断静默跳过。
- 只看 AddGraph 报错字面（"null is invalid"）去查 session/graph_id 参数，不看构图侧是否真的成型了 compute_graph。

## 误诊

- 中间曾怀疑 GE 找不到自定义算子 op_proto 库/`ASCEND_OPP_PATH` 未设置，检查后发现配置均正常，随即转向 plog 定位到构图侧。
