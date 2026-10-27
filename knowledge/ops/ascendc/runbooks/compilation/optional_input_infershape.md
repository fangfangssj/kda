---
type: "Runbook"
title: "optional input 未连接触发 InferShape 失败"
description: "OpDef 声明含 optional input、GEIR 图未连接该 input 时，本例在 ConstructInferShapeContextInputs 阶段 assert；用于确认失败发生在算子 InferShape 回调之前，并核对声明输入与实际连边。"
tags: ["task.debugging", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-18T12:30:47Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# optional input 未连接触发 InferShape 失败

## 现象

在 Ascend950PR（CANN 9.0.0）上用 GEIR 测试程序调用 `advance_step`（已补 Output 声明、kernel 签名、buffer 拷贝），`RunGraph` 仍在 InferShape pass 失败。报错原文（GE 工具输出，verbatim）：

```text
E19999[PID: 647236] 2026-08-06-15:04:23.258.077 (E19999):  Assert ((ConstructCompileKernelContextInputs(op, op_desc, inputs, ge_tensors_holder)) == ge::GRAPH_SUCCESS) failed[FUNC:ConstructInferShapeContextInputs][FILE:shape_inference.cc][LINE:180]
        TraceBack (most recent call last):
       [Construct][InferShapeContextInputs] failed, op_desc[AdvanceStep][FUNC:InferShapeOnCompile][FILE:shape_inference.cc][LINE:658]
       Call InferShapeAndType for node:AdvanceStep(AdvanceStep) failed. You can ignore this validation by exporting IGNORE_INFER_ERROR=1 if necessary, but it is highly recommended to fix this problem.[FUNC:CallInferFunc][FILE:op_desc_utils_ex.cc][LINE:183]
       ...
       Run graph failed, error code:1343242270, session_id:0, graph_id:0.[FUNC:RunGraph][FILE:ge_api.cc][LINE:575]
```

失败发生在 InferShape 函数被调用**之前**的 context 构造阶段（`ConstructInferShapeContextInputs`），即不是 `advance_step_infershape.cpp` 的实现问题，而是 GE 框架构造 context 时 input 数量与 OpDef 声明不匹配。

## 适用场景

- **工程形态**：op_host OpDef 注册 + GEIR 通路测试（`ge::Session`/`AddGraph`/`RunGraph`）。
- **API 路径**：`OpDef` 的 `this->Input(...).ParamType(OPTIONAL)` 声明 + GE `ConstructInferShapeContextInputs`（`shape_inference.cc:180`）。
- **验证过的平台与 CANN 版本**：Ascend950PR / CANN 9.0.0。
- **适用粒度**：`operator_family`（任何 OpDef 声明 optional input、且 GEIR 测试图未连接该 optional input 的算子均会复现）。
- **路线优先级**：`diagnostic_only`。本卡只定位 context 构造阶段的输入契约不一致，不提供跳过 InferShape 校验的绕过路线。
- **不适用/未验证场景**：aclnn 通路不经 InferShape pass，本卡不适用；连接 optional input 会触发 forward compatibility 错误（见误诊），不是解法。

## 判别方法

**触发条件**：OpDef 声明的 input 数 > GEIR 图实际连接的 input 数（典型：声明了 optional input 但测试图只连 required input）。

**可执行探针**：查安装目录下算子配置的 `optionalInputMode` 与 input 列表（工具输出，verbatim）：

```text
=== 950 config ===
optionalInputMode: gen_placeholder
inputs: 8
  input_tokens: required
  sampled_token_ids: required
  input_positions: required
  seq_lens: required
  slot_mapping: required
  block_tables: required
  spec_token: optional
  accepted_num: optional
```

`optionalInputMode: gen_placeholder` 说明 GE 设计上会为未连接的 optional input 生成 placeholder，但 `ConstructInferShapeContextInputs` 仍按声明的 8 个 input 构造 context，与实际连接的 6 个不一致 → assert 失败。

**原始观测**：`RunGraph` 报 `ConstructInferShapeContextInputs ... shape_inference.cc:180` assert 失败，且报错明示 `You can ignore this validation by exporting IGNORE_INFER_ERROR=1`。

**解释边界**：日志表明失败发生在真正的 InferShape 回调之前，因此修改该回调不能触及当前失败点。声明 8 个 input、图中连接 6 个 input 是直接可核的差异；至于 placeholder 生成与内置算子定义覆盖如何共同影响 context 构造，本 trace 没有隔离到框架内部的唯一机制。

## 验证信号

- 不带 `IGNORE_INFER_ERROR`，`RunGraph` 在 InferShape pass 失败、报 `ConstructInferShapeContextInputs ... shape_inference.cc:180` assert，error code 1343242270（来源平台：950；证据等级：tool_observed）。
- 探针读取 950 算子配置，确认 `optionalInputMode: gen_placeholder`、声明 8 input（6 required + spec_token/accepted_num 2 optional），与测试图连接的 6 个 input 不匹配（来源平台：950；证据等级：tool_observed）。

## 失效边界

- ascend910b / ascend910_93：未验证，推断可迁移（同一 GE 框架的 `ConstructInferShapeContextInputs` 机制）。
- 连接 optional input 试图"对齐 8 input"会触发 forward compatibility 错误（见误诊），不可行。

## 误诊

- **误诊 A：去掉 `.Version(ADVANCE_STEP_VERSION_TWO)` 声明以让 optional input 在 v1 可用**。尝试后 `ConstructInferShapeContextInputs` 仍失败——根因不在 Version 声明，而在框架层 input 数量不匹配。
- **误诊 B：在 GEIR 测试图中连接全部 8 个 input（含 spec_token/accepted_num）**。尝试后出现独立的 `forward compatibility` 错误，说明该对照没有验证 optional-input 缺失问题；内置与自定义定义的加载优先级需另行核实。
- **误诊 C：修改 `advance_step_infershape.cpp` 处理 optional 缺失**。无效——失败在 `ConstructInferShapeContextInputs`（context 构造阶段），InferShape 函数根本未被调用到。