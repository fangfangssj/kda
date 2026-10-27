---
type: "Runbook"
title: gert 无 TensorDesc：GetInputDesc 用 auto 接
description: "TilingFunc 中显式声明 const gert::TensorDesc* 接收 context->GetInputDesc(0) 会因该类型不存在而使整个声明语句失败，报错却指向使用点的 'inputDesc' was not declared；GetInputDesc 实际返回 const CompileTimeTensorDesc*，用 auto 接收即可。"
tags: [task.implementation, topic.compilation]
status: stable
sources: []
created_at: '2026-09-17T03:24:03Z'
updated_at: '2026-09-17T03:24:03Z'
platforms: ["950"]
---

# gert 无 TensorDesc：GetInputDesc 用 auto 接

## 现象

为 Ascend950PR（CANN 9.2.0）算子的 TilingFunc 读取输入 dtype 时写了：

```cpp
const gert::TensorDesc* inputDesc = context->GetInputDesc(0);
OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
*dataType = inputDesc->GetDataType();
```

cmake configure 阶段（npu_op_code_gen 编译 tiling.cpp）报：

```
op_host/arch35/<op>_tiling.cpp:164:5:
note: in expansion of macro 'OP_CHECK_NULL_WITH_CONTEXT'
    164 |     OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
op_host/arch35/<op>_tiling.cpp:165:17:
error: 'inputDesc' was not declared in this scope
    165 |     *dataType = inputDesc->GetDataType();
```

报错指向第 165 行的变量名而非第 163 行的类型声明，容易误判为作用域/宏问题。实际是 `gert::TensorDesc` 这个类型不存在，声明语句本身编译失败，导致变量"未声明"。

## 适用场景

- 工程形态：registry-invoke 算子 host 侧 TilingFunc 开发，需要读输入 dtype（`GetInputDesc(0)->GetDataType()`）。
- API 路径：`exe_graph/runtime/extended_kernel_context.h:25`（`const CompileTimeTensorDesc *GetInputDesc(const size_t index) const`）；`runtime/CompileTimeTensorDesc` 的 `GetDataType()`。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.2.0（与 frontmatter `platforms` 一致；验证等级为编译级）。
- 适用粒度：platform_wide——gert 头文件契约，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他 CANN 版本的描述符类型名未验证。

## 根因

`gert` 命名空间里不存在 `TensorDesc` 类型；`TilingContext`（经 `ExtendedKernelContext`）的 `GetInputDesc` 返回 `const CompileTimeTensorDesc*`。用不存在的类型显式声明变量时，整条声明语句编译失败，编译器随后在使用点报 "'inputDesc' was not declared in this scope"，错误位置与根因位置错位。

## 解决配方

修复前（显式写不存在的类型）：

```cpp
// op_host/arch35/<op>_tiling.cpp（修复前）
    // dtype
    const gert::TensorDesc* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
```

修复后（auto 推导正确返回类型）：

```cpp
// op_host/arch35/<op>_tiling.cpp（修复后）
    // dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
```

判别方法：`grep -n "GetInputDesc" $ASCEND_HOME_PATH/x86_64-linux/include/exe_graph/runtime/tiling_context.h extended_kernel_context.h` → `extended_kernel_context.h:25: const CompileTimeTensorDesc *GetInputDesc(const size_t index) const;`，再 `grep -rn "GetDataType" .../runtime/` 确认 `CompileTimeTensorDesc` 有 `GetDataType()`。凡是 gert 侧拿不准的返回类型，用 `auto` 或先查头文件，不要凭记忆写显式类型。

## 设计决策

- 选 `auto` 而非显式 `const gert::CompileTimeTensorDesc*`：add_example 参考实现即用 `auto inputDesc = context->GetInputDesc(0);`；该返回类型在头文件内无稳定公开文档名，显式拼写反而易错。

## 系统信息

- 硬件/软件：x86_64 主机，CANN 9.2.0，目标 ascend950。
- 相关文件：`op_host/arch35/<op>_tiling.cpp`（GetShapeAttrsInfo 的 dtype 段）；`exe_graph/runtime/extended_kernel_context.h`。
- 复现命令：`cd build && rm -f CMakeCache.txt && cmake -DASCEND_COMPUTE_UNIT=ascend950 ..`。

## 预防

- gert/exe_graph 侧 API 返回类型一律 `auto` 接收或查头文件，不写凭印象的显式类型。
- 看到 "'<var>' was not declared in this scope" 且该变量明明在上面声明过时，先检查声明行的类型名是否存在，而不是查作用域。

## 验证信号

以下信号来自该算子 host 侧首轮编译联调：修复前稳定复现，修复后该错误消失。

- 修复前：cmake configure 报 `error: 'inputDesc' was not declared in this scope`（tiling.cpp:165，note 指向 164 行宏展开），`compile ascend all ops failed`（来源平台：950；证据等级：tool_observed）。
- 修复后：同一 cmake configure 不再报此错，推进到下一独立错误（StorageShape GetDim，另行闭环，见库内 [StorageShape 需解包为 Shape 再用维度 API](gert_storageshape_unwrap_dim_api.md)）；全链路修复后全量编译通过并产出安装包（来源平台：950；证据等级：tool_observed）。
- 头文件事实锚点：`extended_kernel_context.h:25` `const CompileTimeTensorDesc *GetInputDesc(const size_t index) const;`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 以 CANN 9.2.0 头文件为准；若后续版本提供 `gert::TensorDesc` 别名或改名，本卡失效（未验证）。

**未决项**：

- 无。
## 坏实践

- 对不熟悉的框架类型写显式 `const gert::Xxx*` 声明（拼错类型名时错误位置错位，排障成本高）。

## 误诊

- 报错字面指向"变量未声明"，曾先怀疑 `OP_CHECK_NULL_WITH_CONTEXT` 宏展开/头文件包含问题：核对 include 链（platform_util.h → tiling_context.h → extended_kernel_context.h）完整后排除，真正根因是类型名不存在。
