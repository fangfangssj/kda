---
type: "Runbook"
title: "ACLNN GetWorkspaceSize 缺 L0 调用致输出全零"
description: "aclnn L2 GetWorkspaceSize 未把 L0 算子加入 launcher list，executor 为空，CommonOpExecutorRun 不 dispatch kernel，NPU 输出全零；补齐 l0op::ApplyAdamV2(INFER_SHAPE + ADD_TO_LAUNCHER_LIST_AICORE) 后恢复。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ACLNN GetWorkspaceSize 缺 L0 调用致输出全零

## 现象

在 Ascend950 (arch35, DAV_3510) 上用 ops-registry-invoke-workflow 开发 ApplyAdamV2 算子。迭代一骨架阶段，Kernel 直调穿刺（`<<<>>>` 内核启动）在 NPU 上已通过（`m_out max_diff=9.31e-10, v_out max_diff=2.33e-10`），证明 kernel 计算逻辑正确。但经 aclnn 框架调用 `aclnnApplyAdamV2` 时，所有非空 tensor 用例输出全零；Tiling 已注册成功（plog 可见 `Op type[ApplyAdamV2] register OP_IMPL : [Tiling][TilingParse]`），但 tiling 函数的 `OP_LOGI` 日志未出现在 plog 中。L0 调用已声明但 GetWorkspaceSize 内 Contiguous 处理后直接返回，留有待实现注释。结论：`aclnnApplyAdamV2 → CommonOpExecutorRun → 执行空 executor → 无 kernel dispatch → 输出全零`。

## 适用场景

- 工程形态：aclnn L2 算子开发（手写 L2 + L0，`aclnn_exclude` 类型）。
- API 路径：`aclnnApplyAdamV2GetWorkspaceSize` → `aclnnApplyAdamV2` 两阶段接口；L0 侧 `INFER_SHAPE` + `ADD_TO_LAUNCHER_LIST_AICORE`。
- 验证平台与版本：Ascend950PR（arch35/DAV_3510），CANN 9.0.0。
- 适用粒度：`operator_family`（适用于所有手写 L2+L0 的 aclnn 算子，不止 ApplyAdamV2）。
- 路线优先级：`recommended_fastpath`（aclnn L2 必须显式调用 L0 把算子加入 launcher list；无替代）。
- 不适用或未验证场景：aclnn_include（自动生成 L2）类型不适用；a2/a3 平台未验证。

## 根因

`aclnnApplyAdamV2GetWorkspaceSize` 在 Contiguous 处理后未调用 L0 算子，executor 的 launcher list 为空；`CommonOpExecutorRun` 据此不 dispatch 任何 kernel，输出 tensor 保持初始零值。Kernel 直调穿刺绕过 aclnn 框架直接 launch kernel，所以穿刺通过而 aclnn 全零——这一对比正是定位该根因的关键判别信号。

## 解决配方

before（`op_api/aclnn_apply_adam_v2.cpp` 中 `aclnnApplyAdamV2GetWorkspaceSize`，骨架阶段，Contiguous 处理后直接返回）：

```cpp
    // Contiguous processing for element tensor inputs
    auto varContiguous = l0op::Contiguous(var, uniqueExecutor.get());
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mContiguous = l0op::Contiguous(m, uniqueExecutor.get());
    CHECK_RET(mContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vContiguous = l0op::Contiguous(v, uniqueExecutor.get());
    CHECK_RET(vContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Scalar tensors: pass through (shape=[1], already contiguous)
    // Optional tensors: pass through (may be nullptr)

    // Iteration 1 skeleton: the L0 operator call is not implemented.
    // Call l0op::ApplyAdamV2 with all inputs and executor here.

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

after（补齐 L0 调用 + 在 `namespace l0op` 内定义静态 L0 算子函数 `ApplyAdamV2`，执行 `INFER_SHAPE` 与 `ADD_TO_LAUNCHER_LIST_AICORE`）：

L0 算子函数（新增于 `namespace l0op`，`OP_TYPE_REGISTER(ApplyAdamV2)` 之后）：

```cpp
namespace l0op {
OP_TYPE_REGISTER(ApplyAdamV2);

/**
 * @brief L0 operator: configure kernel dispatch via INFER_SHAPE + ADD_TO_LAUNCHER_LIST_AICORE
 */
static aclnnStatus ApplyAdamV2(
    const aclTensor* var, const aclTensor* m, const aclTensor* v,
    const aclTensor* lr, const aclTensor* beta1, const aclTensor* beta2,
    const aclTensor* epsilon, const aclTensor* grad,
    const aclTensor* maxGradNorm, const aclTensor* globalGradNorm,
    const aclTensor* weightDecay, const aclTensor* stepSize,
    const char* adamMode,
    aclTensor* varOut, aclTensor* mOut, aclTensor* vOut,
    aclOpExecutor* executor)
{
    L0_DFX(ApplyAdamV2, var, m, v, lr, beta1, beta2, epsilon, grad,
           maxGradNorm, globalGradNorm, weightDecay, stepSize, adamMode,
           varOut, mOut, vOut);

    auto ret = INFER_SHAPE(ApplyAdamV2,
        OP_INPUT(var, m, v, lr, beta1, beta2, epsilon, grad,
                 maxGradNorm, globalGradNorm, weightDecay, stepSize),
        OP_OUTPUT(varOut, mOut, vOut),
        OP_ATTR(adamMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_INFERSHAPE_ERROR, "InferShape failed for ApplyAdamV2.");
        return ret;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(ApplyAdamV2,
        OP_INPUT(var, m, v, lr, beta1, beta2, epsilon, grad,
                 maxGradNorm, globalGradNorm, weightDecay, stepSize),
        OP_OUTPUT(varOut, mOut, vOut),
        OP_ATTR(adamMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID,
                "ADD_TO_LAUNCHER_LIST_AICORE failed for ApplyAdamV2.");
        return ret;
    }

    return ACLNN_SUCCESS;
}
} // namespace l0op
```

GetWorkspaceSize 内 Contiguous 后补齐的 L0 调用：

```cpp
    // Contiguous processing for element tensor inputs
    auto varContiguous = l0op::Contiguous(var, uniqueExecutor.get());
    CHECK_RET(varContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mContiguous = l0op::Contiguous(m, uniqueExecutor.get());
    CHECK_RET(mContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto vContiguous = l0op::Contiguous(v, uniqueExecutor.get());
    CHECK_RET(vContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Scalar tensors: pass through (shape=[1], already contiguous)
    // Optional tensors: pass through (may be nullptr)

    // L0 call: configure kernel dispatch
    auto l0Ret = l0op::ApplyAdamV2(
        varContiguous, mContiguous, vContiguous,
        lr, beta1, beta2, epsilon, gradContiguous,
        maxGradNorm, globalGradNorm, weightDecay, stepSize,
        adamMode,
        varOut, mOut, vOut,
        uniqueExecutor.get());
    CHECK_RET(l0Ret == ACLNN_SUCCESS, l0Ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
```

## 验证信号

补齐 L0 调用后，executor 的 launcher list 非空，`CommonOpExecutorRun` 正常 dispatch kernel；小 shape 用例由全零转为通过（`var_out: PASS | m_out: PASS | v_out: PASS`，`max_diff=2.98e-08`）。（来源平台：950；证据等级: tool_observed）

定位判据：Kernel 直调穿刺通过而 aclnn 框架调用输出全零，且 plog 中只见 TilingParse 注册、不见 tiling 函数 `OP_LOGI`，指向 executor 空 dispatch。（来源平台：950；证据等级: tool_observed）

## 失效边界

- 该修复针对手写 L2+L0（`aclnn_exclude`）类型的算子；aclnn_include 自动生成 L2 的算子由代码生成器填入 L0 调用，不适用本卡。
- 推断可迁移到 a2/a3 平台的手写 aclnn 算子（L0 launcher list 机制相同），但未验证，推断可迁移。
- 若补齐 L0 调用后仍全零，需进一步排查 kernel 二进制编译目标架构（见同 trace 的 ASCEND_COMPUTE_UNIT 卡）与 TilingData 结构一致性。

## 预防

- 写 aclnn L2 骨架时，`aclnnApplyAdamV2GetWorkspaceSize` 内的待实现占位必须在进入联调前落实为真实 L0 调用，禁止带未实现占位进 NPU 验证。
- 联调第一判据：穿刺（Kernel 直调）与 aclnn 框架调用必须同时验证；二者结果一致才算 kernel 真正被 dispatch。

## 误诊

初判怀疑 tiling 函数未执行（因 `OP_LOGI` 不在 plog），但 TilingParse 注册成功；真正未执行的是 kernel dispatch（executor 空），而非 tiling。`OP_LOGI` 不可见是 tiling 函数日志级别/路径问题，不是 tiling 未运行。
