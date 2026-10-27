---
type: "Runbook"
title: "GetOutputViewTensor 私有致 op_host 编译失败"
description: "gert::TilingContext::GetOutputViewTensor 是私有成员，op_host TilingFunc 直接调它取输出 TensorV2 再读 dtype 会编译失败（退出码 1）；取输出 compile-time dtype 须走公有路径 ComputeNodeInfo::GetOutputTdInfo→CompileTimeTensorDesc::GetDataType。"
tags: ["task.debugging", "topic.implementation.api_layer.basic_api", "topic.implementation.tiling"]
status: stable
sources: []
created_at: "2026-08-18T04:27:46Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# GetOutputViewTensor 私有致 op_host 编译失败

## 现象

在 Ascend950 / cann-9.1.0 上为 TileWithAxis 算子的 aclnn ST 补充错误用例 E24（输出 dtype 不匹配）。在 `op_host` 侧 TilingFunc 中，为校验 E24 的输出 dtype，向 `gert::TilingContext* context` 直接调用 `context->GetOutputViewTensor(0)` 试图取输出 `TensorV2` 再读 dtype。`cd operators && ./build.sh TileWithAxis_package` 编译失败（退出码 1），构建日志报错符号指向 `const gert::TensorV2*`，即 `GetOutputViewTensor` 的返回类型——该方法是 `TilingContext` 的私有成员，外部 op_host 代码不可访问。

## 适用场景

- **工程形态**：aclnn 算子仓开发（ops-registry-invoke），op_host 侧 TilingFunc 内做输出 dtype/shape 校验（错误用例 raises_error）。
- **API 路径**：`gert::TilingContext`、`gert::ComputeNodeInfo`、`gert::CompileTimeTensorDesc`、`ge::DataType`（cann `exe_graph/runtime/tiling_context.h`、`compute_node_info.h`）。
- **验证过的平台与 CANN 版本**：Ascend950 / cann-9.1.0（toolkit `/workspace/chenjialiang/Ascend`）。
- **适用粒度**：`operator_family`——任何在 op_host TilingFunc 中需要读取输出 compile-time dtype 的算子都适用。
- **不适用/未验证场景**：运行期才确定的输出 dtype（依赖输入 shape/data 的动态推导）不适用——`GetOutputTdInfo` 返回的是 compile-time desc；这类场景未在本 trace 涉及。a3/a2 未验证。

## 判别方法

**触发条件**：op_host TilingFunc 为读输出 compile-time dtype，直接写 `context->GetOutputViewTensor(0)`（或其它 index）取输出 `TensorV2`。随后 `cd operators && ./build.sh <OpType>_package` 编译失败、退出码 1。

**可执行探针**：构建失败后，捕获并 grep 构建日志中的私有成员报错符号：

```bash
cd operators && ./build.sh <OpType>_package > /tmp/build_output.txt 2>&1; echo "Exit code: $?"
grep -n "TensorV2\|GetOutputViewTensor\|private" /tmp/build_output.txt | head
```

退出码非 0，且日志命中 `const gert::TensorV2*` 指向用户 TilingFunc 对 `GetOutputViewTensor` 的调用点，即为外部访问私有成员所致。

**原始观测**：构建退出码 1（tool_observed，trace seq 426 tool_result 记 `Exit code: 1`）。报错符号 `const gert::TensorV2*` 的逐字 grep 输出未在 trace 的 tool_result 中恢复（markdown_transcript 适配器对 grep 结果记为占位，仅 reasoning 事件复述该符号）。

**解释**（agent 根因分析）：`gert::TilingContext` 暴露的输出相关公有方法 `GetOutputShape(index) → const StorageShape*`、`GetOutputStride`、`GetOutputOffset`、`OutputIsView` 等，**内部**调用私有 `GetOutputViewTensor(index) → const TensorV2*`。即 `GetOutputViewTensor` 是 TilingContext 的私有实现细节，公有方法可以调它，但用户 op_host 代码不能直接调——直接调触发访问私有成员，编译失败。取输出 compile-time dtype 的公有路径是：`context->GetComputeNodeInfo()` 取 `ComputeNodeInfo*` → `ComputeNodeInfo::GetOutputTdInfo(index)` 取 `const CompileTimeTensorDesc*` → `CompileTimeTensorDesc::GetDataType()` 取 `ge::DataType`。三者均公有，`GetComputeNodeInfo()` 在 cann 自身 `TilingContext::GetOutputShape` 等公有方法内部已使用，对外可访问。

## 验证信号

来源：TileWithAxis aclnn ST 错误用例 E24 开发，op_host TilingFunc 调 `GetOutputViewTensor` 后构建。

- 在 TilingFunc 中加 `context->GetOutputViewTensor(0)` 调用后，`./build.sh TileWithAxis_package` 构建失败、退出码 1（来源平台：950；证据等级：tool_observed）

## 预防

- 在 op_host TilingFunc 中读输出 compile-time 属性时，优先用 `ComputeNodeInfo::GetOutputTdInfo` → `CompileTimeTensorDesc::GetDataType`/`GetShape` 等公有路径；不要直接调 `GetOutputViewTensor`。公有取输出 compile-time dtype 的用法：

```cpp
const auto* compute_node_info = context->GetComputeNodeInfo();
if (compute_node_info != nullptr) {
    const auto* outputTd = compute_node_info->GetOutputTdInfo(0);
    if (outputTd != nullptr && outputTd->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;  // dtype mismatch
    }
}
```

- 若只需输出 shape，直接用公有 `GetOutputShape(index)`；它内部已封装私有访问，对外公有。

## 失效边界

- 平台：Ascend950 / cann-9.1.0 实测；a3/a2 未验证，推断可迁移（cann 9.x 内 `ComputeNodeInfo`/`CompileTimeTensorDesc` 公有路径稳定）。
- 边界：本卡只覆盖"`GetOutputViewTensor` 私有、外部不可直接调"的判别与公有替代路径；运行期才确定的输出 dtype 不适用（`GetOutputTdInfo` 返回 compile-time desc）。
- `GetOutputViewTensor` 在 `tiling_context.h` 中的确切可见性修饰符（private vs protected）未在本 trace 从 cann 头逐行核验——仅从构建报错指向 `const gert::TensorV2*` 反推为私有。
