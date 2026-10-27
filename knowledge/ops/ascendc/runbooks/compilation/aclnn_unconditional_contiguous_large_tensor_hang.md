---
type: "Runbook"
title: aclnn 无条件 Contiguous 致连续大张量挂起
description: "aclnn L2 层对 TensorList 每个张量无条件调用 l0op::Contiguous（含已连续张量）后，4M 元素连续大张量用例挂起超时；用 IsContiguous 守卫只对非连续张量转换后恢复通过。"
tags: [task.debugging, topic.data_movement.layout_transform, topic.implementation.api_layer.c_api]
status: stable
sources: []
created_at: '2026-09-16T10:20:34Z'
updated_at: '2026-09-16T10:20:34Z'
platforms: ["950"]
---

# aclnn 无条件 Contiguous 致连续大张量挂起

## 现象

在 Ascend950PR（CANN 9.1.0）上为 BenchNoHarnessForeachNorm 算子的 aclnn L2 层（`op_api/aclnn_bench_no_harness_foreach_norm.cpp`）增加非连续 Tensor 支持时，对输入 TensorList 的每个张量调用 `l0op::Contiguous` 转成连续后再传给 L0 算子。重新编译安装后跑 ST Real 全量回归：600 秒超时（exit code 124），输出显示前 18 条用例全部 PASS 后，4M 元素连续一维张量用例（FP32、4194304 元素、p=2）打印用例头后无结果——该用例在此前的所有轮次一直通过。改动前一轮同套件可完整跑完（73/74，唯一失败为非连续用例）。

## 适用场景

- 工程形态：aclnn 自定义算子 L2 API（op_api 层）开发 + ST Real NPU 回归。
- API 路径：`l0op::Contiguous(const aclTensor*, aclOpExecutor*)`（头文件 `aclnn_kernels/contiguous.h`）、`opdev/tensor_view_utils.h` 的 `IsContiguous(const aclTensor*)`、`aclCreateTensorList`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有在 aclnn 层对输入做 Contiguous 预处理的算子。
- 路线优先级：recommended_fastpath。
- 明确不适用 / 未验证场景：非连续张量经 Contiguous 转换后的正确性验证（本卡未完成，见失效边界）；`PickViewAsContiguous`/`CanOptimizeContiguous` 等替代 API 未尝试。

## 根因

对 TensorList 中**所有**张量（包括已连续张量）无条件调用 `l0op::Contiguous`，会在 executor 中为每个输入加入多余的数据搬运操作，连续大张量（4M 元素）用例因此挂起；`l0op::Contiguous` 对连续张量并非零开销直通的观测行为是该挂起的触发条件（其 executor 内部加入拷贝算子的机制为推断）。

## 解决配方

用 `IsContiguous` 守卫：已连续的张量直接透传，仅对非连续张量调用 `l0op::Contiguous`。

修复前（无条件转换）：

```cpp
    // 对 TensorList 中每个张量执行 Contiguous，确保非连续 Tensor 被转为连续
    // AutoContiguous() 标志在 aclnn 路径不自动生效，需显式调用
    uint64_t xSize = x->Size();
    std::vector<const aclTensor*> contiguousTensors(xSize);
    for (uint64_t i = 0; i < xSize; i++) {
        const aclTensor* xTensor = (*x)[i];
        contiguousTensors[i] = l0op::Contiguous(xTensor, uniqueExecutor.get());
        OP_CHECK_NULL(contiguousTensors[i], return ACLNN_ERR_INNER_NULLPTR);
    }
    aclTensorList* contiguousX = aclCreateTensorList(contiguousTensors.data(), xSize);
    OP_CHECK_NULL(contiguousX, return ACLNN_ERR_INNER_NULLPTR);

    const aclTensorList* opResult = l0op::BenchNoHarnessForeachNorm(
        contiguousX, scalar, y, uniqueExecutor.get());
```

修复后（IsContiguous 守卫，连续张量零拷贝透传）：

```cpp
    // 对 TensorList 中每个张量执行 Contiguous，确保非连续 Tensor 被转为连续
    // AutoContiguous() 标志在 aclnn 路径不自动生效，需显式调用
    // 已连续的张量直接使用，避免不必要的拷贝
    uint64_t xSize = x->Size();
    std::vector<const aclTensor*> contiguousTensors(xSize);
    for (uint64_t i = 0; i < xSize; i++) {
        const aclTensor* xTensor = (*x)[i];
        if (IsContiguous(xTensor)) {
            contiguousTensors[i] = xTensor;
        } else {
            contiguousTensors[i] = l0op::Contiguous(xTensor, uniqueExecutor.get());
            OP_CHECK_NULL(contiguousTensors[i], return ACLNN_ERR_INNER_NULLPTR);
        }
    }
    aclTensorList* contiguousX = aclCreateTensorList(contiguousTensors.data(), xSize);
    OP_CHECK_NULL(contiguousX, return ACLNN_ERR_INNER_NULLPTR);

    const aclTensorList* opResult = l0op::BenchNoHarnessForeachNorm(
        contiguousX, scalar, y, uniqueExecutor.get());
```

配套改动：`#include "opdev/tensor_view_utils.h"`（提供 `IsContiguous`）。另注意 `OP_CHECK_NULL` 宏只接受 2 个参数（条件 + 返回值），带日志的三参数写法会编译失败，需改用 `OP_CHECK` 包装日志。

## 设计决策

- 选 `IsContiguous` 前置守卫而不是依赖 `l0op::Contiguous` 内部跳过：实测无条件调用即触发大张量挂起，守卫把开销降到零；代价是每个输入多一次连续性判断（host 侧纯元数据检查，可忽略）。
- 备选 `PickViewAsContiguous`/`CanOptimizeContiguous` 未采用：行为语义（视图挑选）与本场景"必须得到连续数据"的需求不完全一致，且本卡未验证。

## 验证信号

本卡结论来自该算子后续联调的前后对比：

- 修复前（无条件 Contiguous）：ST Real 600s 超时（exit 124），最后输出停在 4M 元素连续一维张量用例（FP32、4194304 元素）的用例头、无结果，其前 18 条用例均 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后（IsContiguous 守卫）：全量回归中该用例 `[PASS] MERE=0.00e+00, MARE=0.00e+00`，L0 10/10 与 L1 前 24 条用例全部 PASS（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 只验证了"守卫消除连续大张量挂起"；非连续张量经 `l0op::Contiguous` 转换后的**正确性**（非连续用例能否通过）在本卡未完成验证，接续工作见遗留问题。
- 挂起机制（executor 中加入拷贝算子）为推断，未在 executor 层面直接观测。
- 多维非连续（stride>1 且 ndim>1）场景未验证。

**未决项**：

- `l0op::Contiguous` 对连续张量在 executor 中的实际行为（是否加入拷贝 op、耗时来源）未核实。
- 非连续输入经转换后的端到端精度验证未完成（接续工作，另见非连续诊断相关卡）。
## 坏实践

- 认为"Contiguous 对连续张量肯定是 no-op"而不加判断——本场景实测无条件调用即触发大张量挂起。
- aclnn 层改动后只跑小张量冒烟就放行：4M 级大张量用例才会暴露搬运开销问题。
