---
type: "Runbook"
title: IMPL_OP_INFERSHAPE 双展开重定义须链式注册
description: 同一 infershape.cpp 中两次 IMPL_OP_INFERSHAPE(Op) 宏展开（一段挂 InferShape、一段挂 InferDataType）会重复定义全局注册变量 op_impl_register_infershape_<Op>，op_host autogen 编译失败、cmake configure 中断；InferShape/InferDataType/InferOutDataTypeSameWithFirstInput 须在同一次注册对象上链式调用。
tags: [task.debugging, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-17T04:13:22Z'
updated_at: '2026-09-17T04:13:22Z'
platforms: ["950"]
---

# IMPL_OP_INFERSHAPE 双展开重定义须链式注册

## 现象

为一个多输入自定义算子（MoE FinalizeRouting 类，7 输入 1 输出、含 5 个可选输入）从零搭建 registry-invoke 算子工程，首次执行 `cmake -DASCEND_COMPUTE_UNIT=ascend950 ..` configure 时，op_host 的 autogen 编译阶段（ascendc_all_ops.make 编译 infershape.o）直接失败。报错原文：

```text
${ASCEND_HOME_PATH}/aarch64-linux/include/register/op_impl_registry.h:157:37:
  note: 'gert::OpImplRegisterV2
  ops::op_impl_register_infershape_<Op>'
  previously declared here

    157 |   gert::OpImplRegisterV2 VAR_UNUSED op_impl_register_infershape_##op_type = gert::OpImplRegisterV2(#op_type)
        |                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /.../op_host/<op>_infershape.cpp:80:1:
  note: in expansion of macro 'IMPL_OP_INFERSHAPE'

     80 | IMPL_OP_INFERSHAPE(<Op>).InferShape(InferShape4<Op>);
        | ^~~~~~~~~~~~~~~~~~

make: *** [.../build/autogen/ascendc_all_ops.make:31: .../<op>_infershape.o] Error 1
RuntimeError: compile ascend all ops failed
Call Stack: op_host/CMakeLists.txt:49 (npu_op_code_gen)
```

当时 infershape.cpp 里写了两段注册：一段 `IMPL_OP_INFERSHAPE(Op).InferShape(...)` 注册形状推导，另一段 `IMPL_OP_INFERSHAPE(Op).InferDataType(...)` 注册输出 dtype 推导——两段宏展开生成了同名全局变量。

## 适用场景

- 工程形态：registry-invoke 自定义算子工程（op_host 侧 infershape.cpp / tiling.cpp 编写），首次 cmake configure 或全量构建。
- API 路径：`register/op_impl_registry.h` 的 `IMPL_OP_INFERSHAPE` / `IMPL_OP_OPTILING` 注册宏；`gert::OpImplRegisterV2` 的 `.InferShape()` / `.InferDataType()` / `.InferOutDataTypeSameWithFirstInput()` 链式接口。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510 / arch35），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——宏展开行为由 CANN 头文件决定，与具体算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：其他 CANN 版本未验证（宏定义行为可能变化）；`IMPL_OP_OPTILING` 的链式写法（`.Tiling(...).TilingParse<...>(...)`）在本会话一次通过、未经历失败闭环，仅作参照写法。

## 根因

`IMPL_OP_INFERSHAPE(Op)` 宏每次展开都会在当前翻译单元定义一个同名全局注册变量 `op_impl_register_infershape_##op_type`（op_impl_registry.h:157），同一 .cpp 展开两次即变量重定义、编译失败。InferShape、InferDataType、InferOutDataTypeSameWithFirstInput 等都是 `OpImplRegisterV2` 返回对象上的成员函数，必须在**一次**注册对象上链式调用。

## 解决配方

修复前（infershape.cpp，两段独立注册）：

```cpp
IMPL_OP_INFERSHAPE(<Op>).InferShape(InferShape4<Op>);

// ===== InferDataType =====
static ge::graphStatus InferDataType4<Op>(gert::InferDataTypeContext* context)
{
    // out.dtype = expanded_permuted_rows.dtype (输入0)
    const ge::DataType expandedPermutedDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, expandedPermutedDtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(<Op>).InferDataType(InferDataType4<Op>);

} // namespace ops
```

修复后（单次注册、链式挂接；输出 dtype 跟随输入 0 时直接用框架内置规则，无需自写 InferDataType 函数）：

```cpp
IMPL_OP_INFERSHAPE(<Op>)
    .InferShape(InferShape4<Op>)
    .InferOutDataTypeSameWithFirstInput();

} // namespace ops
```

若输出 dtype 推导不是"跟随第一个输入"，则保留自定义函数并链式挂 `.InferDataType(InferDataType4<Op>)`——同样只注册一次。

## 设计决策

- 选择 `InferOutDataTypeSameWithFirstInput()` 内置规则替代自写 InferDataType 函数：本算子输出 dtype 恰好等于输入 0 的 dtype，内置规则语义正确且少维护一段代码；自写函数仅在推导规则非平凡时保留。
- 本工程 tiling 注册采用同样的单次链式写法一次通过：`IMPL_OP_OPTILING(Op).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParseFunc);`——注册宏"一次展开、链式挂接"是统一模式。

## 预防

- 写 op_host 注册代码时自查：同一 .cpp 内每个 `IMPL_OP_*` 宏（INFERSHAPE / OPTILING）对同一算子只允许出现一次；多种实现挂接一律写成链式调用。
- 遇到 `op_impl_register_infershape_<Op>` / `op_impl_register_optiling_<Op>` 重定义报错，直接定位到第二次宏展开的行，合并注册即可，不必怀疑头文件包含顺序。

## 验证信号

来源说明：以下信号来自该算子迭代一骨架的首次构建会话——configure 失败于 infershape.o 重定义，合并注册后重新 configure 验证。

- 修复前：`cmake -DASCEND_COMPUTE_UNIT=ascend950 ..` 在 `op_host/CMakeLists.txt:49 (npu_op_code_gen)` 处报 `op_impl_register_infershape_<Op> previously declared here`，`RuntimeError: compile ascend all ops failed`（来源平台：950；证据等级：tool_observed）。
- 修复后：重新 configure 时 infershape.o 编译通过，错误阶段推进到 opbuild 的 dtype 校验（另一独立问题）；该轮问题全部修复后 configure 输出 `-- Configuring done / -- Generating done`（来源平台：950；证据等级：tool_observed）。
- 最终全量构建 `cmake --build . --target all binary package` 复跑 grep error/warning 为空、退出码 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡结论基于 CANN 9.1.0 的 `register/op_impl_registry.h:157` 宏定义；升级 CANN 版本后宏行为如变化需重新验证（未验证，机制推断可迁移）。
- `InferOutDataTypeSameWithFirstInput` 仅适用于"输出 dtype == 输入 0 dtype"的算子；多输出或 dtype 推导非平凡时须用自定义 InferDataType 函数（本卡链式写法仍适用）。

**未决项**：

- 无（宏展开机制明确，修复一次生效）。
## 坏实践

- 在一个 .cpp 里为同一算子写两段 `IMPL_OP_INFERSHAPE(Op)` 分别挂 InferShape 和 InferDataType——必然重定义。
- 把 InferShape 与 InferDataType 拆到两个 .cpp 各自注册：即便绕开单文件重定义，也会产生两个同名全局注册变量（链接期冲突），同样不可行。

## 误诊

- 无（报错 note 直接指向第二次宏展开行与宏定义行，一次定位）。
