---
type: "Runbook"
title: "RunOpApi 延迟回调需按值捕获局部 launch vector"
description: "多 tensor 合并到一个 RunOpApi lambda 时，引用捕获局部 pointer/tiling vector 会悬空；按值捕获保住提交参数。"
tags: ["task.debugging", "topic.implementation.synchronization"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# RunOpApi 延迟回调需按值捕获局部 launch vector

## 现象

Host 为 TensorList 预先收集 GM pointer 和 tiling 参数，再用一个 `OpCommand::RunOpApi` 回调依次 launch 多个 kernel。构建通过，但多 tensor 冒烟在首组输出前静默挂起；相邻实现也曾表现为输出大幅错误或 NaN。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`RunOpApi`、`OpCommand`、`lambda`、`TensorList` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`exact_task`，只覆盖相同 RunOpApi 延迟回调与局部发射状态生命周期。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

唯一根因是延迟执行的 lambda 通过引用捕获局部 vector。包装函数结束后，vector 对象及其 storage 已失效；回调再读取 `size()`、pointer 或 tiling 参数即产生悬空访问。捕获标量按值而只对 vector 用引用，并不能保证 vector 生命周期。

## 解决配方

把回调执行期需要的 pointer/tiling vector 全部按值捕获。以下是同一完整 callable 的 before/after，launch 分支未改变。

### before

`csrc/ops/foreach_addcdiv_scalar/op_plugin/foreach_addcdiv_scalar_plugin.cpp`，`acl_call`，修复前：

```cpp
    auto acl_call = [=, &x1Ptrs, &x2Ptrs, &x3Ptrs, &yPtrs,
                        &totalLens, &numBlocksV, &blockLens, &tileSizes, &hostKeys]() -> int {
        for (size_t i = 0; i < x1Ptrs.size(); ++i) {
            if (dtype == torch::kFloat32) {
                launch_foreach_addcdiv_scalar_float(
                    x1Ptrs[i], x2Ptrs[i], x3Ptrs[i], yPtrs[i], scalar_f,
                    totalLens[i], numBlocksV[i], blockLens[i],
                    static_cast<uint32_t>(tileSizes[i]), hostKeys[i], stream);
            } else if (dtype == torch::kFloat16) {
                launch_foreach_addcdiv_scalar_half(
                    x1Ptrs[i], x2Ptrs[i], x3Ptrs[i], yPtrs[i], scalar_f,
                    totalLens[i], numBlocksV[i], blockLens[i],
                    static_cast<uint32_t>(tileSizes[i]), hostKeys[i], stream);
            } else if (dtype == torch::kBFloat16) {
                launch_foreach_addcdiv_scalar_bfloat16(
                    x1Ptrs[i], x2Ptrs[i], x3Ptrs[i], yPtrs[i], scalar_f,
                    totalLens[i], numBlocksV[i], blockLens[i],
                    static_cast<uint32_t>(tileSizes[i]), hostKeys[i], stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ForeachAddcdivScalar", acl_call);
    return out;
}
```

### after

同文件、同 callable，修复后：

```cpp
    auto acl_call = [=]() -> int {
        for (size_t i = 0; i < x1Ptrs.size(); ++i) {
            if (dtype == torch::kFloat32) {
                launch_foreach_addcdiv_scalar_float(
                    x1Ptrs[i], x2Ptrs[i], x3Ptrs[i], yPtrs[i], scalar_f,
                    totalLens[i], numBlocksV[i], blockLens[i],
                    static_cast<uint32_t>(tileSizes[i]), hostKeys[i], stream);
            } else if (dtype == torch::kFloat16) {
                launch_foreach_addcdiv_scalar_half(
                    x1Ptrs[i], x2Ptrs[i], x3Ptrs[i], yPtrs[i], scalar_f,
                    totalLens[i], numBlocksV[i], blockLens[i],
                    static_cast<uint32_t>(tileSizes[i]), hostKeys[i], stream);
            } else if (dtype == torch::kBFloat16) {
                launch_foreach_addcdiv_scalar_bfloat16(
                    x1Ptrs[i], x2Ptrs[i], x3Ptrs[i], yPtrs[i], scalar_f,
                    totalLens[i], numBlocksV[i], blockLens[i],
                    static_cast<uint32_t>(tileSizes[i]), hostKeys[i], stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ForeachAddcdivScalar", acl_call);
    return out;
}
```

## 设计决策

按值捕获复制的是 GM pointer 与 tiling 标量 vector，不复制 tensor payload。若要捕获 Tensor handle，也应按值复制 handle 以延长引用计数。只有 API 明确保证同步立即调用时，引用捕获局部对象才安全。

## 预防

- 所有交给异步/延迟 API 的 lambda 做 capture-lifetime 审查。
- 在回调入口检查 vector 长度与第一个 pointer，避免把 Host 悬空误诊为 kernel 算术错误。
- 多 tensor 合并 launch 后同时覆盖 L=1、L>1 和多 dtype 冒烟。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：运行只打印首个 fp32 多元素用例标题，未产出该用例结果或完成标记。（来源平台：a3；证据等级: tool_observed）。
- 修复后：fp32/fp16 定向矩阵每项 max_diff=0。（来源平台：a3；证据等级: tool_observed）。
- 修复后：完整运行 20/20，exit=0。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- vector 中若存裸指针，仍需保证底层 tensor/GM storage 生命周期覆盖回调。
- 长期缓存 callable 会延长所有按值捕获对象寿命，应定义释放时点。
- 其他平台未验证；核心结论由 Host 延迟执行契约触发。

## 坏实践

- 混合 `[=, &localVectors...]` 并假定 `RunOpApi` 一定立即执行。
- 看到 NaN 或随机大误差就修改 kernel 数学逻辑，未先确认 Host launch 参数仍有效。
- 只用 L=1 测试 TensorList wrapper；它不能覆盖 vector 遍历和生命周期问题。

## 误诊

多 dtype、多个输出同时异常且复现位置随运行变化时，应先检查 callback 生命周期；单个 AscendC 指令精度问题通常不会让 Host vector 的控制数据失效。