---
type: "Runbook"
title: "OP_CHECK_NOTNULL 内嵌 int 返回，指针函数编译错"
description: opdev 的 OP_CHECK_NOTNULL(val) 判空失败时内嵌 return ACLNN_ERR_INNER_NULLPTR（int 561103），只能用于返回 aclnnStatus 的函数；在返回 const aclTensor*/const aclTensorList*/bool 的 L0 API 里使用报 invalid conversion from 'int' to 指针（报错位置落在 op_errno.h 而非业务行）。指针/bool 返回函数须用三参 OP_CHECK(cond, OP_LOGE(...), return <表达式>) 或显式 if。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T01:50:47Z'
updated_at: '2026-09-17T01:50:47Z'
platforms: ["950"]
---

# OP_CHECK_NOTNULL 内嵌 int 返回，指针函数编译错

## 现象

三条独立轨迹、同一宏契约的编译失败（Ascend950PR 目标编译，CANN 9.1.0，registry-invoke 算子 op_api 层）。L0 API 按惯例返回输出张量句柄（`const aclTensor*` / `const aclTensorList*`）或 `bool`，开发者做判空时先用了不存在的 `OP_CHECK_NULL(x, return nullptr)`，按编译器 "did you mean" 提示换成 `OP_CHECK_NOTNULL(x);` 后，报错变成与业务代码行号不对应的类型错误：

```text
$ASCEND_HOME_PATH/.../include/aclnn/opdev/op_errno.h:24:33: error: invalid conversion from ‘int’ to ‘const aclTensor*’ [-fpermissive]
   24 | #define ACLNN_ERR_INNER_NULLPTR 561103
      |                                 ^~~~~~
```

（bool 返回函数与 `const aclTensorList*` 返回函数同型报错：`invalid conversion from 'int' to 'const aclTensorList*'`。）

完整链路（以 resize_bilinear 轨迹为例，两轮失败）：

第一轮（宏不存在）：

```text
op_api/bench_no_harness_resize_bilinear.cpp:136:5: error: 'OP_CHECK_NULL' was not declared in this scope; did you mean 'OP_CHECK_NOTNULL'?
   136 |     OP_CHECK_NULL(x, return nullptr);
op_api/bench_no_harness_resize_bilinear.cpp:136:22: error: expected primary-expression before 'return'
```

第二轮（换成 OP_CHECK_NOTNULL 后）：

```text
$ASCEND_HOME_PATH/.../include/aclnn/opdev/op_errno.h:24:33: error: invalid conversion from 'int' to 'const aclTensor*' [-fpermissive]
```

`OP_CHECK_NOTNULL` 展开后内嵌 `return ACLNN_ERR_INNER_NULLPTR;`——一个 int。指针/bool 返回的 L0 函数里，宏里内嵌的 int return 与函数返回类型不匹配，直接编译失败。

## 适用场景

- 工程形态：手写自定义算子（registry-invoke）op_api 层的 L0 算子实现——L0 函数的返回类型通常是 `const aclTensor*`/`const aclTensorList*`（成功返回输出句柄）或 `bool`（校验辅助函数），而不是 `aclnnStatus`；L2 两段式接口（`aclnn<Op>GetWorkspaceSize`）返回 `aclnnStatus`。
- API 路径：`OP_CHECK_NOTNULL(val)`（`aclnn/opdev/op_log.h:202`，单参、内嵌 `return ACLNN_ERR_INNER_NULLPTR`）、`OP_CHECK(cond, log_func, return_expr)`（三参、可自定义返回动作）、`CHECK_COND`/`CHECK_RET`（aclnnStatus 语境）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950 目标编译），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——宏语义属于 opdev 头文件契约，与算子无关。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：返回 `aclnnStatus` 的函数（L2 两段式接口、CheckParams）——`OP_CHECK_NOTNULL`/`CHECK_COND` 在那里是正确用法（已验证）；`OP_CHECK_NULL` 两参宏定义在 `aclnn_kernels/common/op_error_check.h`（aclnn L2 层头），L0 未 include 时不可用，见同域 [OP_CHECK_NULL 宏层级可用性卡](acltensor_no_iscontiguous_op_check_null.md)。

## 根因

`OP_CHECK_NOTNULL` 是为 `aclnnStatus`（int 兼容）返回函数设计的速记宏：判空失败时 `return ACLNN_ERR_INNER_NULLPTR;`（= 561103，int）。宏体内嵌 return 语句，调用方函数的返回类型必须与之兼容。L0 层函数按 ACLNN 惯例返回输出张量句柄（指针）或 bool，int→指针 / int→bool 都是非法隐式转换，于是报错位置落在宏展开处（`op_errno.h:24`）而非业务代码行——第一次看到时容易误判成头文件问题。

配套事实：CANN 9.1.0 opdev 头文件里不存在带返回表达式的 `OP_CHECK_NULL(val, expr)` 宏（它定义在 aclnn 层的 op_error_check.h）；编译器的 "did you mean OP_CHECK_NOTNULL" 建议只保证名字相近，不保证语义兼容。

## 解决配方

按函数返回型选检查方式：int 返回型（aclnnStatus）用 `OP_CHECK_NOTNULL`；指针/bool 返回型用三参 `OP_CHECK`（返回动作显式写进宏参数）或显式 if。

修复前（L0 中误用，三轨迹同型；grid_sampler_3d 轨迹 verbatim）：

```cpp
const aclTensor* BenchNoHarnessGridSampler3d(
    const aclTensor* x, const aclTensor* grid,
    int64_t interpolationMode, int64_t paddingMode, bool alignCorners,
    aclOpExecutor* executor)
{
    OP_CHECK_NULL(x, return nullptr);        // 第一轮：宏不存在
    OP_CHECK_NULL(grid, return nullptr);

    // 形状推导
    Shape outShape;
    OP_CHECK(InferShape(x->GetViewShape(), grid->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."),
             return nullptr);

    // 分配输出 Tensor
    const aclTensor* y = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK_NULL(y, return nullptr);

    // 检查平台和 dtype 支持
    OP_CHECK(IsAiCoreSupport(x, grid),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return BenchNoHarnessGridSampler3dAiCore(x, grid, y,
        interpolationMode, paddingMode, alignCorners, executor);
}
```

修复后（指针返回型统一走三参 OP_CHECK，返回类型随宿主函数走）：

```cpp
const aclTensor* BenchNoHarnessGridSampler3d(
    const aclTensor* x, const aclTensor* grid,
    int64_t interpolationMode, int64_t paddingMode, bool alignCorners,
    aclOpExecutor* executor)
{
    // 形状推导
    Shape outShape;
    OP_CHECK(InferShape(x->GetViewShape(), grid->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."),
             return nullptr);

    // 分配输出 Tensor
    const aclTensor* y = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK(y != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed."),
             return nullptr);

    // 检查平台和 dtype 支持
    OP_CHECK(IsAiCoreSupport(x, grid),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return BenchNoHarnessGridSampler3dAiCore(x, grid, y,
        interpolationMode, paddingMode, alignCorners, executor);
}
```

bool 返回函数（foreach_addcdiv_scalar 轨迹 verbatim）：

```cpp
// 修复前（IsAiCoreSupport 返回 bool，宏内嵌 int return）：
static bool IsAiCoreSupport(const aclTensorList* x1)
{
    ...
    OP_CHECK(x1->Size() > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TensorList is empty"), return false);
    auto x1Tensor = (*x1)[0];
    OP_CHECK_NOTNULL(x1Tensor);
    ...
}

// 修复后：
static bool IsAiCoreSupport(const aclTensorList* x1)
{
    ...
    OP_CHECK(x1->Size() > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TensorList is empty"), return false);
    auto x1Tensor = (*x1)[0];
    OP_CHECK(x1Tensor != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "x1[0] is null"), return false);
    ...
}
```

显式 if 形态（resize_bilinear 轨迹，零宏语义歧义）：

```cpp
const aclTensor* BenchNoHarnessResizeBilinear(
    const aclTensor* x,
    const aclIntArray* outputSize,
    bool alignCorners,
    const aclFloatArray* scaleFactor,
    aclOpExecutor* executor)
{
    if (x == nullptr || executor == nullptr) {
        return nullptr;
    }
```

要点：`OP_CHECK(cond, OP_LOGE(...), <任意 return 语句>)` 的第三个参数就是失败时的返回动作，类型随宿主函数走（`return false` / `return nullptr`），指针、bool、自定义错误码函数都能用。L2（aclnnStatus）接口里 `OP_CHECK_NOTNULL(x);` 是合法用法，保留不动。

## 设计决策

- 指针返回函数统一走 `OP_CHECK(cond, OP_LOGE(<code>, <msg>), return nullptr)` 三参宏：返回类型、错误码、日志一次写清，不依赖宏内部的隐式 return；与同文件其余 `OP_CHECK` 风格一致。
- 不引入 `OP_CHECK_NULL`（曾以为存在该宏并被编译器提示 did you mean）：CANN 9.1.0 opdev 中不存在，`OP_CHECK` 组合即可覆盖。
- 不强行包一层"宏 + lambda 返回指针"的技巧：显式 if / 三参宏最直白且零宏语义歧义。
- 编译器 "did you mean" 建议只保证名字相近，不保证语义兼容；替换前先看宏定义体。

## 预防

- 写 ACLNN 判空前先确认函数返回型：返回 `aclnnStatus` 才能用 `OP_CHECK_NOTNULL`；返回指针/bool 的 L0 API 一律显式 `if (p == nullptr) { return nullptr; }` 或 `OP_CHECK` 条件宏。
- 记忆口诀：`OP_CHECK_NOTNULL` 自带 `return 561103`，落点必须是 int 返回型函数。
- 报错位置落在 CANN 头文件（op_errno.h）而非业务代码行时，先怀疑"宏内嵌 return 与宿主函数返回类型不匹配"，不是头文件问题。

## 验证信号

以下信号来自三条独立轨迹的 op_api 编译闭环（CANN 9.1.0，ascend950 目标编译），修复前稳定复现、修复后编译通过。

- resize_bilinear 轨迹：第一轮 `'OP_CHECK_NULL' was not declared in this scope; did you mean 'OP_CHECK_NOTNULL'?` + `expected primary-expression before 'return'`；第二轮 `op_errno.h:24:33: error: invalid conversion from 'int' to 'const aclTensor*'`；修复后 op_api 两个目标编译通过、全量 `build.sh --soc=ascend950 -j8` 成功产出 3 个 dtype 内核二进制与安装包（来源平台：950；证据等级：tool_observed）。
- grid_sampler_3d 轨迹：第一轮 `OP_CHECK_NULL was not declared`、第二轮 `invalid conversion from 'int' to 'const aclTensor*'`；修复后 host 全部目标编译通过（cust_opapi/cust_optiling/cust_op_proto 等 Built，进入 kernel 编译阶段）（来源平台：950；证据等级：tool_observed）。
- foreach_addcdiv_scalar 轨迹：`op_errno.h:24:33: error: invalid conversion from 'int' to 'const aclTensorList*'`（bool 与 TensorList 指针两处同型）；修复后构建通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 宏语义以 CANN 9.1.0 的 `opdev/op_log.h` 为准；若未来版本提供带自定义 return 的判空宏或改宏体，本卡结论需复核。
- `ACLNN_ERR_INNER_NULLPTR(561103)` 在本卡是宏展开上下文中出现的编译期常量，不是运行期返回码现象；按码检索运行期 561103 报错时应先查 tiling/executor 空指针路径，勿与本编译期问题混淆。
- grid_sampler_3d 轨迹的修复版 L0 入口删掉了 x/grid 的显式空指针检查、依赖下游 InferShape 间接暴露——正式上库前应回补显式 `OP_CHECK(x != nullptr && ...)`（该轨迹遗留项）。

**未决项**：

- 无（三条轨迹均闭环到编译通过）。
## 坏实践

- 把参考示例里的 `OP_CHECK_NULL(x, return false)`（示例环境自定义的辅助宏，或 aclnn 层宏）当作 opdev 标准宏直接照抄进 L0。
- 按编译器 "did you mean" 提示盲替换单参宏，不看宏体返回类型。
- 在指针返回函数里用 `OP_CHECK_NOTNULL` 后再加外层包装函数把 int 转 nullptr——多余一层间接，且丢掉具体错误码日志。

## 误诊

- 第一轮失败后曾以为是"宏名拼写/可用性问题"，直接按提示换成 `OP_CHECK_NOTNULL(y);`——该宏在指针返回函数里语义不兼容，导致第二轮类型错误。正确做法是先读宏定义再选型。
- 报错指向 `op_errno.h:24` 时一度像"CANN 头文件本身有错"——宏展开处的报错行号不代表错误源头在头文件。
