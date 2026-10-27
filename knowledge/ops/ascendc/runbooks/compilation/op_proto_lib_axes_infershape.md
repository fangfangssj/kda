---
type: "Runbook"
title: "缺 OP_PROTO_LIB 致空 axes 命中内置 InferShape"
description: "cust_proto_obj 未加 OP_PROTO_LIB 时 OP_ADD 走默认分支，自定义 V2 gert InferShape 不注入 rt2 registry，框架回落 CANN 内置 EuclideanNormInferShape（reduce_ops.cc），全局 first-wins 使 COMMON_INFER_FUNC_REG 无法覆盖；内置把空 axes(shape=[0]) 当 passthrough 而非 full reduce，输出 shape 错误。"
tags: ["task.debugging", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-18T16:10:11Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["empty_axes", "full_reduce"]
---

# 缺 OP_PROTO_LIB 致空 axes 命中内置 InferShape

## 现象

在 Ascend950（arch35, CANN 9.0.0-beta.2）上以 GEIR 图模式开发 `EuclideanNorm` 算子。GEIR ST 静态用例 T-S10（"special_fullreduce"——空 axes(shape=[0]) 应做 full reduce，输入 x shape=[2,3] 时期望输出 shape=[]）持续失败：输出 shape 不是 full reduce 的结果，而是 passthrough（原输入 shape=[2,3]）。`GRAPH-RUN` 行报告 `value=FAIL`，`GEIR-SUMMARY total=35 pass=34 fail=1`。

## 适用场景

- **工程形态**：aclnn/GEIR 图模式算子开发（op_host + op_graph 注册层），属 op_host 构建配置 + InferShape 注册机制问题，非 kernel 计算逻辑。
- **API 路径**：CMake `target_compile_definitions(cust_proto_obj PRIVATE OP_PROTO_LIB ...)` + `OP_ADD`/`OP_END_FACTORY_REG` + `Operator::InferFuncRegister` + `COMMON_INFER_FUNC_REG` + gert V2 `IMPL_OP` InferShape/InferDataType。
- **验证平台与版本**：Ascend950，CANN 9.0.0-beta.2，GEIR 图模式 ST（test_geir）。
- **适用粒度**：`operator_family` —— 任何自定义算子名与 CANN 内置算子同名（EuclideanNorm 等 reduce 类内置）、或需要覆盖内置 InferShape 行为的算子都会踩。
- **不适用或未验证场景**：算子名不与任何 CANN 内置冲突、且自定义 InferShape 无需覆盖内置时不触发；非 GEIR/图模式路径不涉及。a2/a3 平台未验证。

## 判别方法

### 触发（tool_observed 失败）

GEIR ST 静态用例 T-S10（空 axes shape=[0]，输入 x shape=[2,3]，keep_dims=false 期望 full reduce 输出 shape=[]）在 `GRAPH-RUN mode=static` 下 `value=FAIL`：`expected_shape=[] actual_shape=[2,3] expected_numel=1 actual_numel=6`。即框架对空 axes 做了 passthrough（输出原 shape）而非 full reduce。`GEIR-SUMMARY total=35 pass=34 fail=1`。

### 探针（可执行检查）

1. **路由日志检查**：抓 GEIR 运行日志，grep `CallInferFunc` 与 `InferShapeFuncV`，看 T-S10 节点命中的是自定义 InferShape 还是 built-in。命中 `InferShapeFuncV1` 且不是开发者注册的自定义 V1 时，说明落到了 built-in。
2. **built-in 符号检查**：对 CANN built-in `libopsproto.so` 跑 `strings | grep <OpType>InferShape` 与 `nm -D | grep op_impl_register_infershape_<OpType>`，确认 built-in 是否为同名算子注册了 InferShape（若存在同名符号，全局 map 已被 built-in 先占）。
3. **shape flow 检查**：抓 T-S10 节点的 `infer_base_pass`/`infershape_pass` 日志，看 axes shape=[0] 时 `[after_infer]` 输出 shape 是否被改成 passthrough（与输入同 shape），并核对 `origin shape` 是否未被 reduce。

### 观测（tool_observed）

- `GRAPH-RUN ... case=T-S10 ... expected_shape=[] actual_shape=[2,3] ... value=FAIL` / `CASE T-S10 FAIL` / `GEIR-SUMMARY total=35 pass=34 fail=1`。
- `strings libopsproto.so` 命中 `EuclideanNormInferShape` 与 `Enter EuclideanNormInferShape proto inferfunction`；`nm -D` 命中 `ops::op_impl_register_infershape_EuclideanNorm`（built-in 已在全局 map 注册同名 InferShape）。
- T-S10 shape flow：`[before_infer]` axes output shape=[0]；`[after_infer]` 节点 output shape=[2,3]、`origin shape is 6`（即未 reduce，passthrough）。

### 解释（根因分析）

两层机制叠加：

1. **`cust_proto_obj` 未定义 `OP_PROTO_LIB`**：标准 CANN 模板（`npu_op_library` GRAPH 编译源时）会给 proto 目标加 `OP_PROTO_LIB`，使 `OP_ADD` 经 `OpImplRegisterV2` 把 OpDef 上 `SetInferShape`/`SetInferDataType` 注册的 gert V2 InferShape 注入 rt2 registry。未加时 `OP_ADD` 走默认分支（仅 OpDefFactory），V2 gert InferShape 永不注入 rt2 registry，框架回落到 built-in V1 `InferShapeAndType`。
2. **全局 `COMMON_INFER_FUNC_REG` 是 first-wins**：CANN built-in `EuclideanNormInferShape`（reduce_ops.cc）先注册到全局 `InferShapeFuncRegister` map，自定义算子用 `COMMON_INFER_FUNC_REG` 后注册无法覆盖（first-wins，后注册不生效）。built-in 对空 axes（shape=[0]）按 passthrough 处理，而非 full reduce。

判别要点：当且仅当「自定义算子名与 CANN built-in 同名」且「cust_proto_obj 缺 OP_PROTO_LIB 或仅靠 COMMON_INFER_FUNC_REG 注册」两个条件同时成立时，空 axes（及任何依赖自定义 InferShape 语义的用例）会落到 built-in 而失败。仅命中 built-in 符号（探针 2）但自定义 InferShape 已通过 `Operator::InferFuncRegister` 逐实例挂到 OpDesc 时，`CallInferFunc` 优先查 OpDesc 上的 InferFunc（先于全局 map），可不触发本失败。

## 验证信号

- T-S10 `GRAPH-RUN mode=static` 报告 `value=FAIL`：`expected_shape=[] actual_shape=[2,3] expected_numel=1 actual_numel=6`，`GEIR-SUMMARY total=35 pass=34 fail=1`（来源平台：950；证据等级：tool_observed）。
- `strings libopsproto.so` 与 `nm -D` 输出含 built-in `EuclideanNormInferShape`、`op_impl_register_infershape_EuclideanNorm` 符号，证明同名 built-in InferShape 已全局注册（来源平台：950；证据等级：tool_observed）。
- T-S10 shape flow 日志：axes output shape=[0]，`[after_infer]` 节点 output shape=[2,3]、`origin shape is 6`（passthrough，未 reduce）（来源平台：950；证据等级：tool_observed）。