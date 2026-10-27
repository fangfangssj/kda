---
type: "Runbook"
title: OpAttrDef 无 Int64()，int64 属性用 Int()
description: "op_def 中 this->Attr(\"dim\").Int64() 编译报 'class ops::OpAttrDef' has no member named 'Int64'；CANN register/op_def.h 提供 Int()，属性仍按 int64 语义由 tiling 侧 GetAttrPointer<int64_t> 读回。"
tags: [task.implementation, topic.compilation, topic.implementation.operator_registration]
status: stable
sources: []
created_at: '2026-09-17T03:04:16Z'
updated_at: '2026-09-17T03:04:16Z'
platforms: ["950"]
---

# OpAttrDef 无 Int64()，int64 属性用 Int()

## 现象

在 Ascend950PR（CANN 9.1.0）上搭建自定义算子 BenchNoHarnessCummin 的 op_host 骨架时，算子属性 `dim` 是 int64 语义（对标 torch.cummin 的 dim 参数，支持负数索引）。按"类型对齐"的直觉，在 `op_host/bench_no_harness_cummin_def.cpp` 的 OpDef 构造函数里写成 `this->Attr("dim").Int64();`。首次执行 `bash build.sh --soc=ascend950 -j8` 编译，`compile ascend_all_ops` 阶段失败，报错原文：

```text
op_host/bench_no_harness_cummin_def.cpp:44:27:
error: 'class ops::OpAttrDef' has no member named 'Int64'; did you mean 'Int'?
     44 |         this->Attr("dim").Int64();
         |                           ^~~~~
         |                           Int
```

同一份日志里还并列出现 `ge::DT_BFLOAT16` 与 `gert::TilingContext::GetAttrPointer` 两类错误（各有独立根因，不在本卡范围）；本卡只针对 Attr 链式方法名。

## 适用场景

- 工程形态：ops-registry-invoke 自定义算子开发（OpDef 注册）。
- API 路径：`op_host/*_def.cpp` 的 `this->Attr("<name>").<Type>()` 链式声明，依赖 `register/op_def_registry.h` → `register/op_def.h` 的 `ops::OpAttrDef`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（验证方式为编译通过；tiling 侧按 int64 读回同属性成功编译）。
- 适用粒度：platform_wide——所有声明整型属性的自定义算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：非整型属性（Float/Bool/String/List 等方法名以 op_def.h 为准，本卡未逐一验证）。

## 根因

`ops::OpAttrDef` 的整型属性声明方法名是 `Int()`，不存在 `Int64()`；这里编译器的 `did you mean 'Int'` 恰好是正确答案。

机制链：`Int()` 声明的属性在 tiling 侧仍按 int64 语义访问——本算子 tiling 用 `attrs->GetAttrPointer<int64_t>(0)` 取回 dim 指针并解引用为 int64，两端编译均通过，证明 `Int()` 与 int64 读法兼容；方法名说的是"整型属性"这一类别，不是存储宽度。

## 解决配方

修复前（`op_host/bench_no_harness_cummin_def.cpp` 构造函数的属性声明，verbatim）：

```cpp
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("dim").Int64();
```

修复后（仅方法名一改）：

```cpp
        this->Attr("dim").Int();
```

配套消费端（op_host/arch35/*_tiling.cpp，供核对语义未变）：

```cpp
    // 3. 获取 dim 属性
    const auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto dimScalar = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimScalar);
    int64_t dim = *dimScalar;
```

## 预防

- 写 `Attr(...)` 链前先 `grep -n "OpAttrDef &" $ASCEND_HOME_PATH/x86_64-linux/include/register/op_def.h` 查可用方法集（本机提供 `Int()` 等）。
- 记住命名口径：属性声明方法表达"类别"（Int/Float/...），宽度由访问端模板参数（`GetAttrPointer<int64_t>`）决定。

## 验证信号

以下信号来自该算子骨架搭建会话的编译修复循环。

- 首次编译：`error: 'class ops::OpAttrDef' has no member named 'Int64'; did you mean 'Int'?`（来源平台：950；证据等级：tool_observed）。
- 修复后重编译：`cust_optiling`/`cust_op_proto` 目标编译链接通过，tiling 侧 `GetAttrPointer<int64_t>(0)` 同一属性编译通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅核对 CANN 9.1.0 的 `register/op_def.h`；其他版本方法集可能变化（未验证，推断可迁移）。
- 未验证 Float/Bool/String 等其他属性方法的对应关系。

**未决项**：

无（该错误已闭环到编译通过）。
## 坏实践

- 因为属性语义是 int64 就臆造 `.Int64()` 方法名。
- 与 `aclScalar` 的取值方法（`ToInt64()`）混淆——那是 op_api 层另一套 API。

## 误诊

无。
