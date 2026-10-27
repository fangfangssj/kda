---
type: "Runbook"
title: OP_CHECK 第三参花括号逗号拆参致编译失败
description: L0 opapi 层把 return {nullptr, nullptr} 写进三参宏 OP_CHECK 第三参，花括号内顶层逗号被预处理器拆成第 4 个实参，报 macro "OP_CHECK" passed 4 arguments, but takes just 3（且该层宏可见性不可依赖）；tuple 返回的失败分支改用显式 if + OP_LOGE + return。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T01:39:38Z'
updated_at: '2026-09-17T01:39:38Z'
platforms: ["950"]
---

# OP_CHECK 第三参花括号逗号拆参致编译失败

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义算子 BenchNoHarnessCummin 编写 op_api 两层代码：`aclnn_bench_no_harness_cummin.cpp`（L2 aclnn 接口）与 `bench_no_harness_cummin.cpp`（L0 launcher，返回 `std::tuple<const aclTensor*, const aclTensor*>`）。L0 层函数失败时需要返回 `{nullptr, nullptr}`，最初照 aclnn 层的宏样式写：

```cpp
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessCumminAiCore failed."),
        return {nullptr, nullptr});
```

第二次执行 `bash build.sh --soc=ascend950 -j8` 编译到 `cust_opapi` 时失败，报错原文：

```text
op_api/bench_no_harness_cummin.cpp:68:34: error: macro "OP_CHECK" passed 4 arguments, but takes just 3
   68 |         return {nullptr, nullptr});
      |                                  ^
include/aclnn/opdev/op_log.h:227: note: macro "OP_CHECK" defined here
   227 | #define OP_CHECK(cond, log_func, return_expr) \

op_api/bench_no_harness_cummin.cpp:65:5: error: 'OP_CHECK' was not declared in this scope
   65 |     OP_CHECK(
```

同型错误在另一独立会话的 RoPE 双输出算子 L0 API（`op_api/bench_no_harness_apply_rotary_pos_emb.cpp`，返回 `std::tuple<const aclTensor*, const aclTensor*>`）同样出现：`77:34/98:39` 两处调用点（`ADD_TO_LAUNCHER_LIST_AICORE` 返回值检查与 `IsAiCoreSupport` 结果检查）均报 `macro "OP_CHECK" passed 4 arguments, but takes just 3`，并连带 `'OP_CHECK' was not declared in this scope`。两条 trace 共 5 处调用点同型。

同轮编译里 `OP_CHECK_SAME_SHAPE` 也被头文件核查发现不存在（该问题在编译前用 grep 预先修正为 `OP_CHECK_SHAPE_NOT_EQUAL`，未走到编译报错）。

## 适用场景

- 工程形态：自定义算子仓开发，op_api 层（L2 aclnn + L0 launcher）入参校验与错误返回；重点是返回 `std::tuple` 的多输出 L0 API。
- API 路径：`aclnn/opdev/op_log.h` 的 `OP_CHECK(cond, log_func, return_expr)` 三参宏；`aclnn_kernels/common/op_error_check.h` 的校验宏族（OP_CHECK_NULL / OP_CHECK_DTYPE_NOT_SUPPORT / OP_CHECK_SHAPE_NOT_EQUAL / OP_CHECK_MAX_DIM ...）、`OP_LOGE`、`ADD_TO_LAUNCHER_LIST_AICORE`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.1.0（两条独立 trace 编译闭环验证；未上板）。
- 适用粒度：platform_wide——所有在返回 tuple/braced-init 的函数里使用 CANN OP_CHECK 宏的 op_api 代码。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：返回单指针/bool/void 的函数（`return nullptr;`/`return false;` 无逗号）可继续用三参 OP_CHECK（两条 trace 的 aclnn 层 `return false` 校验均编译通过）；非 CANN 宏的普通函数调用不受此限；其他 CANN 版本的宏签名未核对。

## 根因

`OP_CHECK` 是三参函数式宏，预处理器按"未加括号保护的顶层逗号"切分实参；第三参 `return {nullptr, nullptr}` 的花括号初始化列表内含一个逗号，使宏实际收到 4 个实参（`{nullptr` 与 `nullptr}` 被拆开），触发 `passed 4 arguments, but takes just 3`。同时 L0 launcher 层的头文件包含链并不保证展开出 `OP_CHECK` 定义（同文件还出现 "not declared in this scope"），宏在该层的可见性不可依赖。

机制链：宏实参切分发生在语法分析之前，`{...}` 内的逗号对预处理器而言与普通逗号无区别；只有把表达式整体加圆括号（`return (std::tuple<...>{nullptr, nullptr})` 之类）才能护住逗号，但 L0 层更直接的做法是不用宏。展开失败进而连带报 `'OP_CHECK' was not declared in this scope`。

## 解决配方

tuple 返回的函数不用 OP_CHECK 包裹失败分支，改用普通 if + OP_LOGE + return；单指针/bool 返回的检查保持 OP_CHECK 不变。

修复前（L0 层两个函数，verbatim 取自首次写入）：

```cpp
static std::tuple<const aclTensor*, const aclTensor*> BenchNoHarnessCumminAiCore(
    const aclTensor* input, const int64_t dim,
    const aclTensor* values, const aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessCumminAiCore, input, values, indices);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessCummin,
        OP_INPUT(input), OP_OUTPUT(values, indices), OP_ATTR(dim));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessCumminAiCore failed."),
        return {nullptr, nullptr});
    return {values, indices};
}

std::tuple<const aclTensor*, const aclTensor*> BenchNoHarnessCummin(
    const aclTensor* input, const int64_t dim, aclOpExecutor* executor)
{
    Shape valuesShape;
    Shape indicesShape;
    OP_CHECK(BenchNoHarnessCumminInferShape(input->GetViewShape(), valuesShape, indicesShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return {nullptr, nullptr});

    auto values = executor->AllocTensor(valuesShape, input->GetDataType());
    auto indices = executor->AllocTensor(indicesShape, DataType::DT_INT64);

    OP_CHECK(IsAiCoreSupport(input),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return {nullptr, nullptr});

    return BenchNoHarnessCumminAiCore(input, dim, values, indices, executor);
}
```

修复后（L0 层展开为显式 if + OP_LOGE + return，不再依赖宏可见性）：

```cpp
static std::tuple<const aclTensor*, const aclTensor*> BenchNoHarnessCumminAiCore(
    const aclTensor* input, const int64_t dim,
    const aclTensor* values, const aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(BenchNoHarnessCumminAiCore, input, values, indices);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(BenchNoHarnessCummin,
        OP_INPUT(input), OP_OUTPUT(values, indices), OP_ATTR(dim));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "BenchNoHarnessCumminAiCore failed.");
        return {nullptr, nullptr};
    }
    return {values, indices};
}

std::tuple<const aclTensor*, const aclTensor*> BenchNoHarnessCummin(
    const aclTensor* input, const int64_t dim, aclOpExecutor* executor)
{
    Shape valuesShape;
    Shape indicesShape;
    if (!BenchNoHarnessCumminInferShape(input->GetViewShape(), valuesShape, indicesShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed.");
        return {nullptr, nullptr};
    }

    auto values = executor->AllocTensor(valuesShape, input->GetDataType());
    auto indices = executor->AllocTensor(indicesShape, DataType::DT_INT64);

    if (!IsAiCoreSupport(input)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed.");
        return {nullptr, nullptr};
    }

    return BenchNoHarnessCumminAiCore(input, dim, values, indices, executor);
}
```

同轮配套修正（aclnn 层 shape 校验宏名，`OP_CHECK_SAME_SHAPE` 不存在，grep `op_error_check.h` 确认正确名）：

```cpp
    // 修复前（宏不存在）:
    OP_CHECK_SAME_SHAPE(input, values, return false);
    OP_CHECK_SAME_SHAPE(input, indices, return false);
    // 修复后:
    OP_CHECK_SHAPE_NOT_EQUAL(input, values, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(input, indices, return false);
```

## 设计决策

- 为何选 if 替代而不是把返回值包进括号：`return ({nullptr, nullptr})` 这类写法不是合法 C++ 语句；`std::make_tuple(nullptr, nullptr)` 仍含顶层逗号，同样会被拆参。if + OP_LOGE 语义与 OP_CHECK 完全等价且无预处理器陷阱。
- 保留场景：单指针/bool 返回（`return nullptr;`/`return false;`）的检查继续用 OP_CHECK，保持与模板示例一致。
- 当前方案属 recommended_fastpath。

## 预防

- 在返回值为 tuple/结构体（braced-init 含逗号）的函数里，不要把 `return {...}` 放进任何函数式宏实参；展开为显式 if 最稳。
- 往任何 CANN 日志/检查宏里传含顶层逗号的表达式（braced-init-list、未加括号的模板实参列表）前，先确认宏是否为变参。
- 用 CANN 校验宏前先核对宏表：`grep -n "#define OP_CHECK" $ASCEND_HOME_PATH/x86_64-linux/include/aclnn_kernels/common/op_error_check.h`（本机提供 OP_CHECK_NULL / OP_CHECK_DTYPE_NOT_SUPPORT / OP_CHECK_SHAPE_NOT_EQUAL / OP_CHECK_MAX_DIM 等；没有 OP_CHECK_SAME_SHAPE）。
- L2 aclnn 层返回 bool 的校验（`return false`）可继续用三参 OP_CHECK。

## 验证信号

以下信号来自两个独立算子开发会话的 op_api 编译修复循环（cummin 双输出与 RoPE 双输出 L0 API，Ascend950PR，CANN 9.1.0）。

- cummin 会话第二次编译：`error: macro "OP_CHECK" passed 4 arguments, but takes just 3`（L0 层 3 处）+ `error: 'OP_CHECK' was not declared in this scope`（来源平台：950；证据等级：tool_observed）。
- cummin 会话宏定义核查：`op_log.h:227: #define OP_CHECK(cond, log_func, return_expr)`（来源平台：950；证据等级：tool_observed）。
- cummin 会话修复后第三次编译：`cust_opapi` 目标编译链接通过（`Linking CXX shared library libcust_opapi.so`）（来源平台：950；证据等级：tool_observed）。
- RoPE 会话修复前：`op_api/bench_no_harness_apply_rotary_pos_emb.cpp:77:34/98:39` 两处 `macro "OP_CHECK" passed 4 arguments, but takes just 3` + `'OP_CHECK' was not declared in this scope`，op_api 目标编译失败（来源平台：950；证据等级：tool_observed）。
- RoPE 会话修复后：op_host / op_api 目标全部编译通过，构建错误列表中只剩 kernel 侧问题（DataCopyPad 相关，另卡记录），最终全量构建输出 `[INFO] Build completed successfully!` 并生成算子包 `custom_opp_ubuntu_x86_64.run`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅验证 CANN 9.1.0 的宏签名与可见性；其他版本宏集可能变化（未验证，推断可迁移）。
- "L0 层宏不可见"基于本开发会话 的包含链观察（该文件同时报 not declared 与 4 参错误），未逐一排查是哪个传递包含缺失；显式展开的写法不依赖该结论。
- 两条 trace 均为编译闭环验证，未上板。

**未决项**：

- 无（两条轨迹均闭环到编译/构建通过）。
## 坏实践

- 在宏第三参里放 `return {a, b}` 这类含逗号的 braced-init。
- 臆造宏名 `OP_CHECK_SAME_SHAPE`（正确为 `OP_CHECK_SHAPE_NOT_EQUAL`，语义是"不相等才报错"）。
- 照搬单输出算子模板的 OP_CHECK 写法到 tuple 返回函数，不检查宏参数计数。
- 误以为报错是头文件缺失（`'OP_CHECK' was not declared` 具有误导性）而四处补 include——根因在宏实参拆分。

## 误诊

- RoPE 会话首轮看到 `'OP_CHECK' was not declared in this scope` 时曾怀疑 opdev 头文件未被包含；核对 `opdev/op_log.h` 中宏定义并观察 `passed 4 arguments` 报错后，确认为逗号拆参，而非头文件路径问题。
