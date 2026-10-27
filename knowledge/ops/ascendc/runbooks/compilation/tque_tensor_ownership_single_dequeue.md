---
type: "Runbook"
title: "TQue tensor 所有权只能 DeQue 一次"
description: "FreeTensor 需要可写 LocalTensor lvalue，且同一队列元素只能 DeQue 一次；把队列所有权集中在明确的生产消费函数中可消除编译错误和重复出队。"
tags: ["task.debugging", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# TQue tensor 所有权只能 DeQue 一次

## 现象

在 `operators/rms_norm/rms_norm.asc` 的逐行处理循环里，gamma 队列最后写成：

```text
inQueueGamma.FreeTensor(inQueueGamma.DeQue<T>())
```

构建对 half、float、bf16 三个模板实例都报 `FreeTensor` 没有匹配重载，并指出候选函数要求第一个参数是 lvalue。与此同时，x 队列在 `Compute` 内已经 `DeQue`，循环结尾又执行一次 `DeQue`，形成同一元素的重复出队风险。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：卡内所述 Ascend C 接口链 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`exact_task`，只覆盖相同 TQue 生产、消费和释放结构。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

`FreeTensor` 的参数必须是一个可引用的 `LocalTensor<T>` lvalue，不能直接接收 `DeQue<T>()` 返回的临时表达式。队列所有权还要求一个已入队 tensor 只被一个消费者出队一次；如果 `Compute` 已消费 x，外层循环再次 `DeQue` 就破坏 `Alloc→EnQue→DeQue→Free` 的一一对应关系。

## 解决配方

```cpp
    __aicore__ inline void Process() {
        if (rowCount_ == 0) {
            return;
        }
        // gamma loaded once, resident across this core's rows.
        LocalTensor<T> gammaUb = inQueueGamma.AllocTensor<T>();
        CopyInGamma(gammaUb);
        inQueueGamma.EnQue(gammaUb);

        for (uint32_t r = 0; r < rowCount_; r++) {
            LocalTensor<T> xUb = inQueueX.AllocTensor<T>();
            CopyInRow(xUb, r);
            inQueueX.EnQue(xUb);

            LocalTensor<T> yUb = outQueueY.AllocTensor<T>();
            Compute(r, gammaUb, yUb);
            outQueueY.EnQue<T>(yUb);

            LocalTensor<T> yOut = outQueueY.DeQue<T>();
            CopyOutRow(yOut, r);
            outQueueY.FreeTensor(yOut);

            LocalTensor<T> xIn = inQueueX.DeQue<T>();
            inQueueX.FreeTensor(xIn);
        }
        inQueueGamma.FreeTensor(inQueueGamma.DeQue<T>());
    }
```

```cpp
    __aicore__ inline void Process() {
        if (rowCount_ == 0) {
            return;
        }
        // gamma loaded once, resident (DeQue'd once) across this core's rows.
        LocalTensor<T> gammaUb = inQueueGamma.AllocTensor<T>();
        CopyInGamma(gammaUb);
        inQueueGamma.EnQue(gammaUb);
        LocalTensor<T> gammaResident = inQueueGamma.DeQue<T>();

        for (uint32_t r = 0; r < rowCount_; r++) {
            CopyInRow(r);
            Compute(r, gammaResident);
            CopyOutRow(r);
        }
        inQueueGamma.FreeTensor(gammaResident);
    }
```

## 设计决策

- gamma 是每个核跨多行复用的只读数据：入队一次、出队一次，保存 `gammaResident` 到循环结束再释放。
- x 由 `CopyInRow` 生产、`Compute` 消费并释放；y 由 `Compute` 生产、`CopyOutRow` 消费并释放。每个 tensor 的所有权边界只有一处。
- 放弃在 `Process` 中同时手工管理所有队列对象，因为这容易与子函数内部 `DeQue` 重复。
- 路线属于 `recommended_fastpath`，生命周期明确且不增加额外队列操作。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：三处 FreeTensor 调用均报 no matching function，暴露重复 DeQue/错误 ownership。（来源平台：a3；证据等级: tool_observed）。
- 修复后：单次 DeQue/FreeTensor 后 BUILD_RC=0，共享库生成。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

- 本配方假设单 buffer、顺序逐行消费；双缓冲或异步流水需要重新检查每个队列槽位的生产消费次数。
- 未验证 Reg API 或不使用 TQue 的静态 Tensor 路径。

## 坏实践

- 把 `DeQue<T>()` 直接嵌在 `FreeTensor(...)` 参数中。
- 在外层和子函数中同时对同一队列负责出队，导致所有权重复。
- 只让编译过而不审计每个 `Alloc/EnQue/DeQue/Free` 是否一一配对。

## 预防

- 为每个队列写出唯一生产者和唯一消费者。
- code review 时按 tensor 画出 `Alloc→EnQue→DeQue→Free` 链，并核对循环次数。