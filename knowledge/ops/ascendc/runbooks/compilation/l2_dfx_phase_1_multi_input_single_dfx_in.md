---
type: "Runbook"
title: L2_DFX_PHASE_1 只收 3 参：多输入并 DFX_IN
description: aclnn L2 API 的 L2_DFX_PHASE_1 宏签名是 (APIName, IN, OUT)；多个输入/属性必须合并在一个 DFX_IN(...) 里，写成两个 DFX_IN 会报 macro "L2_DFX_PHASE_1" passed 4 arguments, but takes just 3。
tags: [task.debugging, topic.compilation, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-17T03:04:16Z'
updated_at: '2026-09-17T03:04:16Z'
platforms: ["950"]
---

# L2_DFX_PHASE_1 只收 3 参：多输入并 DFX_IN

## 现象

算子骨架开发阶段，为 aclnn L2 API 的 `GetWorkspaceSize` 入口写 DFX 打点。算子有一个张量输入 `x`、一个 ListInt 属性 `outputSize`（`aclIntArray*`）、一个输出 `y`，打点宏按"每个输入一个 DFX_IN"的直觉写。执行 `bash build.sh --soc=ascend950 -j8` 时 op_api 编译失败，报错原文（路径前缀截断）：

```text
op_api/aclnn_bench_no_harness_adaptive_avg_pool_3d.cpp:149:99:
error: macro "L2_DFX_PHASE_1" passed 4 arguments, but takes just 3

  149 |     L2_DFX_PHASE_1(aclnnBenchNoHarnessAdaptiveAvgPool3d, DFX_IN(x), DFX_IN(outputSize), DFX_OUT(y));
      |                                                                                                   ^

In file included from $ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_executor.h:25, ...
$ASCEND_HOME_PATH/x86_64-linux/include/aclnn/opdev/op_dfx.h:428: note: macro "L2_DFX_PHASE_1" defined here

  428 | #define L2_DFX_PHASE_1(APIName, IN, OUT)                                              \
```

并伴生 `error: 'L2_DFX_PHASE_1' was not declared in this scope`（宏参数不匹配导致展开失败后的连锁报错）。

## 适用场景

- 工程形态：算子仓自定义算子 op_api 的 aclnn L2 封装（`aclnnXxxGetWorkspaceSize` 入口 DFX 打点）。
- API 路径：`aclnn/opdev/op_dfx.h` 的 `L2_DFX_PHASE_1` / `L2_DFX_PHASE_2` 宏族与 `DFX_IN` / `DFX_OUT` 变参宏。
- 验证过的平台与 CANN 版本：Ascend950PR + CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——所有写 aclnn L2 API 打点的算子工程。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：`L2_DFX_PHASE_2`（收尾打点，只收 APIName 一个参数）；其他 CANN 版本的 DFX 宏签名未逐一核对。

## 根因

`L2_DFX_PHASE_1` 的宏签名只有 3 个形参 `(APIName, IN, OUT)`；`DFX_IN` 本身是可变参宏，多个输入（含 `aclIntArray*` 属性）必须合并在同一个 `DFX_IN(...)` 内传入。写成两个 `DFX_IN` 后，预处理器按顶层逗号数出 4 个实参，与宏定义不匹配，展开失败。

## 解决配方

修复前（打点行，verbatim）：

```cpp
    L2_DFX_PHASE_1(aclnnBenchNoHarnessAdaptiveAvgPool3d, DFX_IN(x), DFX_IN(outputSize), DFX_OUT(y));
```

修复后（verbatim，输入与属性合并进一个 DFX_IN）：

```cpp
    L2_DFX_PHASE_1(aclnnBenchNoHarnessAdaptiveAvgPool3d, DFX_IN(x, outputSize), DFX_OUT(y));
```

修复后的完整函数（`aclnnBenchNoHarnessAdaptiveAvgPool3dGetWorkspaceSize`，verbatim）：

```cpp
extern "C" aclnnStatus aclnnBenchNoHarnessAdaptiveAvgPool3dGetWorkspaceSize(
    const aclTensor* x,
    const aclIntArray* outputSize,
    aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBenchNoHarnessAdaptiveAvgPool3d, DFX_IN(x, outputSize), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, outputSize, y, workspaceSize, const_cast<const aclOpExecutor**>(executor));
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空 Tensor 快速返回
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::BenchNoHarnessAdaptiveAvgPool3d(
        xContiguous, outputSize, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

## 设计决策

- 属性（`aclIntArray*` outputSize）与张量输入一起进 `DFX_IN`：参考工程 add_example 的用法是 `L2_DFX_PHASE_1(aclnnAddExample, DFX_IN(x1, x2), DFX_OUT(out))`——DFX 打点跟踪的是"进入算子的全部入参"，属性也是入参，实测编译通过。
- 收尾打点 `L2_DFX_PHASE_2(aclnnBenchNoHarnessAdaptiveAvgPool3d)` 只收 APIName 一个参数，不要与 PHASE_1 的传参方式混淆。

## 预防

- 写 L2_DFX_PHASE_1 前记住签名是三段式：API 名、一个 DFX_IN（全部输入）、一个 DFX_OUT（全部输出）。
- 报 `macro ... passed N arguments, but takes just M` 时，先数顶层逗号：变参宏（DFX_IN）内的逗号不计，顶层多写一个 DFX_IN/DFX_OUT 就多一个实参。

## 验证信号

以下信号来自该算子骨架开发阶段的构建循环。

- 修复后同一条构建命令中 op_api 编译通过，cust_opapi 目标 Built、libcust_opapi.so 链接成功（来源平台：950；证据等级：tool_observed）。
- 最终整包构建成功，产出 ascend950 kernel 二进制与 custom_opp_ubuntu_x86_64.run（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 仅验证 CANN 9.1.0 的 `op_dfx.h` 宏签名；DFX 宏族其他成员/其他版本的签名未逐一核对（未验证，推断可迁移）。
- 打点只影响 DFX 观测数据，不影响算子语义与精度。

**未决项**：

- 无。
## 坏实践

- 按每个输入一个 `DFX_IN` 的直觉展开传参——顶层逗号数超参直接编译失败。
