---
type: "Runbook"
title: OpDef 骨架 ge::InferShape 无字面，绑定走 IMPL_OP
description: "proto OpDef 骨架行的 ge::InferShape/ge::InferDataType 在 CANN register 头文件 grep 不到字面符号属预期；实际 infershape/datatype 绑定经 IMPL_OP 宏在 op_host/op_graph 注册。"
tags: [task.implementation, topic.implementation.operator_registration, topic.compilation]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# OpDef 骨架 ge::InferShape 无字面，绑定走 IMPL_OP


## 现象

写算子原型文档（proto.md）的 OpDef 节时，按工程骨架要求写绑定行：

```cpp
this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend950");
```

随后到 CANN 安装目录核对语法，发现在 register 头文件里 grep 不到骨架记号：

```
grep -rn "ge::InferShape\|ge::InferDataType" \
  /home/<user>/Ascend/cann-9.0.0-beta.2/x86_64-linux/include/register/
# → 无任何命中（graph/infershape/ 目录同样无命中）
```

`ge::InferShape` / `ge::InferDataType` 在头文件中查无字面符号，容易怀疑骨架写法有误而"纠正"它（删行或改写成别的符号），或反过来把这一行照抄进 def.cpp 当真实绑定。

## 适用场景

- 工程形态：ops-registry-invoke 算子仓开发——proto.md OpDef 骨架书写与 op_host def.cpp 实际注册的分层理解。
- API 路径：`register/op_def.h`（OpDef::SetInferShape / SetInferDataType、OpAICoreDef::SetTiling / AddConfig）、`register/op_def_registry.h`（OP_ADD 宏）、op_host/op_graph 的 IMPL_OP_INFERSHAPE / IMPL_OP / IMPL_OP_OPTILING 宏。
- 验证过的平台与 CANN 版本：Ascend 950 系列（Ascend950PR/Ascend950DT 工具链）+ CANN 9.0.0-beta.2（host 侧头文件 grep 核验；未上板）。
- 适用粒度（transfer_scope）：platform_wide——CANN 9.0.0-beta.2 OpDef 注册机制，对所有自定义算子工程同构。
- 路线优先级：recommended_fastpath——proto 骨架行按钦定写法原样保留；实际 infershape/datatype 绑定在 op_host/op_graph 用 IMPL_OP 系宏注册，不照抄 proto 行进 def.cpp。
- 明确不适用 / 未验证场景：SetInferShape 传自定义函数指针（非骨架记号）的真实可用路径本会话未涉及；其他 CANN 版本的记号解析情况未验证。

## 判别方法

**根因（单句）**：CANN 的 OpDef 注册 API 里 `SetInferShape`/`SetInferDataType` 方法真实存在（形参类型是 `gert::OpImplRegisterV2::InferShapeKernelFunc` 等函数指针），但工程骨架中的 `ge::InferShape`/`ge::InferDataType` 是文档级记号而非 register 头文件里的可 grep 字面符号——工程惯例是 proto.md 保留该行作声明占位，真实 shape/datatype 推导绑定由 op_host/op_graph 的 IMPL_OP 系宏另行注册，因此"头文件查无字面"不等于"骨架写错"。

**解释链（每步有实证）**：

1. 方法存在：`op_def.h` 逐字命中 `OpDef &SetInferShape(gert::OpImplRegisterV2::InferShapeKernelFunc func)`、`OpDef &SetInferDataType(gert::OpImplRegisterV2::InferDataTypeKernelFunc func)`、`OpAICoreDef &SetTiling(gert::OpImplRegisterV2::TilingKernelFunc func)`、`OpAICoreDef &AddConfig(const char *soc)`（L490/L492/L422/L435）。
2. 记号无字面：register/ 与 graph/infershape/ 头文件 grep `ge::InferShape|ge::InferDataType` 均无命中。
3. 工程分层实证：同仓样例 add_custom 的 def.cpp 与前代 def.cpp 的 OpDef 构造函数都**没有** SetInferShape 行；前代 def.cpp 头注释明文"proto.md declares SetInferShape(ge::InferShape) ... those bindings are registered through the equivalent op_impl_registry macros"（IMPL_OP_INFERSHAPE 在 op_host/infershape.cpp、IMPL_OP 在 op_graph/graph_infer.cpp、IMPL_OP_OPTILING 在 op_host/arch35/tiling.cpp）。
4. 宏接线佐证：`op_def_registry.h` 的 OP_ADD 宏体含 `impl.InferShape(op.GetInferShape()).InferDataType(op.GetInferDataType())`——OpDef 侧的 GetInferShape 默认值被宏接进 impl registry，真实推导函数由 IMPL_OP 宏覆盖。

## 设计决策

- 骨架行原样保留（不因 grep 不到字面而删改）：proto.md 是接口契约文档，骨架一致性优先；前代 proto.md 同款写法且经全流程交付。
- 绑定分层：proto 的 `SetInferShape(ge::InferShape)` 行是声明占位；def.cpp 实际注册走 IMPL_OP 宏（以样例工程为准），两处不互相照抄。

## 系统信息

- toolchain：CANN 9.0.0-beta.2（`/home/<user>/Ascend/cann-9.0.0-beta.2`，x86_64-linux）。
- 相关文件（本会话实际读/检索/写）：
  - `$CANN/x86_64-linux/include/register/op_def.h`、`register/op_def_registry.h`（API 实况）
  - 当前工作区 `operators/add_custom_package/op_host/add_custom_def.cpp`（样例：无 SetInferShape 行）
  - 旧工作区 `operators/ThnnFusedGruCellGrad_package/op_host/thnn_fused_gru_cell_grad_def.cpp`（前代：宏注册注释存证）
  - 旧工作区 `docs/ThnnFusedGruCellGrad/develop/proto.md` L32（前代 proto 同款骨架行）
  - `docs/ThnnFusedGruCellGrad/develop/proto.md`（产出，骨架行保留）

## 完整代码

核对脚本（可直接复跑）：

```bash
H=/home/<user>/Ascend/cann-9.0.0-beta.2/x86_64-linux/include

## 1. 方法确实存在（grep 方法名，不是记号名）
grep -n "SetInferShape\|SetInferDataType\|SetTiling\|AddConfig\|class OpDef\|ValueDepend" \
  $H/register/op_def.h | head -20
## 命中（节选）：
##   195:  OpParamDef &ValueDepend(Option value_depend);
##   422:  OpAICoreDef &SetTiling(gert::OpImplRegisterV2::TilingKernelFunc func);
##   435:  OpAICoreDef &AddConfig(const char *soc);
##   490:  OpDef &SetInferShape(gert::OpImplRegisterV2::InferShapeKernelFunc func);
##   492:  OpDef &SetInferDataType(gert::OpImplRegisterV2::InferDataTypeKernelFunc func);

## 2. 骨架记号在 register/graph 头文件确无字面
grep -rn "ge::InferShape\|ge::InferDataType" $H/register/ | head -5    # → 空
grep -rn "InferDataType\b" $H/graph/infershape/* 2>/dev/null | head -5 # → 空

## 3. OP_ADD 宏内部接线
grep -n "InferShape\|InferDataType" $H/register/op_def_registry.h | head -10
## 命中（节选）：
##   23:    impl.InferShape(op.GetInferShape())
##   25:      .InferDataType(op.GetInferDataType());
```

产出 proto.md 中保留的骨架行（verbatim）：

```cpp
this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend950");
```

## 预防

- 核对 OpDef 骨架 API 时 grep **方法名**（SetInferShape/SetInferDataType/SetTiling/AddConfig），不要 grep **记号名**（ge::InferShape）——后者是骨架记号，头文件无字面是预期。
- 写 def.cpp 时不要从 proto.md 照抄 `SetInferShape(ge::InferShape)` 行——实际绑定走 IMPL_OP_INFERSHAPE/IMPL_OP 宏（以样例工程为准）。
- ValueDepend 只在有"值依赖输入"（host 需读输入值参与 shape/tiling 决策）时写；需求/规格文档无记载就不写（本例 grep REQUIREMENTS/spec 无值依赖记载，exit 1）。

## 验证信号

本卡信号来自一次 GRU Cell 融合反向算子原型提取会话中对 CANN 头文件与样例工程的核对（host 侧 grep 实测）：

- op_def.h 中 SetInferShape/SetInferDataType/SetTiling/AddConfig/ValueDepend 方法声明逐字命中（形参为 `gert::OpImplRegisterV2::*KernelFunc`，op_def.h L195/L422/L435/L490/L492）（来源平台：950——CANN 9.0.0-beta.2 头文件 host 侧 grep；证据等级：tool_observed）
- register/ 与 graph/infershape/ 目录 grep `ge::InferShape|ge::InferDataType` 无命中；同仓样例与前代 def.cpp 均无 SetInferShape 行、前代宏注册注释逐字可读；op_def_registry.h OP_ADD 宏体含 `impl.InferShape(op.GetInferShape())` 接线（来源平台：950——CANN 9.0.0-beta.2 头文件与工作区代码 artifact 核验；证据等级：tool_observed）
- 产出 proto.md 按骨架保留该行，验收 5 项 PASS（含 OP_ADD( / public OpDef 检索）（来源平台：950——host 侧；证据等级：tool_observed）

## 失效边界

- 结论绑定 CANN 9.0.0-beta.2 头文件实况与该工程骨架约定；其他 CANN 版本未验证（未验证，推断同构，推断可迁移）。
- `ge::InferShape` 记号的确切解析位置未追查（本会话 grep register/ 与 graph/infershape/ 均无命中；未做全 include 树扫描与编译验证）——本卡只证明"方法存在 + 记号无字面 + 工程惯例分层"三点。

## 坏实践

- 因 grep 不到字面就删改 proto 骨架行（破坏接口契约文档与骨架的一致性）。
- 把 proto 的 `SetInferShape(ge::InferShape)` 行照抄进 def.cpp 当真实绑定（实际绑定走 IMPL_OP 系宏，照抄不能编译出有效推导）。
- 核对 API 时只 grep 一处头文件路径就下结论——CANN 包内同 名头文件有多份（如 `x86_64-linux/include/register/op_def.h` 与 `include/aclnn/opdev/op_def.h`），先 find 全部再对目标逐一 grep。

## 误诊

- 无。

## 未决事项

- 优先项：`ge::InferShape`/`ge::InferDataType` 记号的确切定义位置与解析机制未追到底（下一步：全量 find CANN include 树 + 查生成代码/宏展开；若确认是纯文档记号，应在骨架文档中显式标注）。
- 未验证：SetInferShape 传自定义函数指针（非骨架记号）的真实可用路径与编译表现。
