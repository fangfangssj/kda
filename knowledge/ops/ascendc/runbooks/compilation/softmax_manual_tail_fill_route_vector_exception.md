---
type: "Runbook"
title: "Softmax 手工尾填充会触发 Vector 异常"
description: "将宽度补齐后用 Duplicate 或 SetValue 写入负无穷，非对齐宽度仍触发 vector core exception，而对齐控制组通过。"
tags: ["task.debugging", "topic.compute.normalization"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# Softmax 手工尾填充会触发 Vector 异常

## 现象

高阶 `SoftMax` 路线把真实宽度向上补齐，并尝试在 UB 的尾段填入负无穷。对齐宽度能运行，多个非对齐宽度却稳定返回 vector core exception。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程。
- API 路径：`SoftMax`、`Duplicate`、`SetValue`、`vector core exception` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`same_operator`，只禁止同一 SoftMax 非对齐尾部的机械手工填充路线。
- 路线优先级：`banned`。
- 不适用/停止边界：该失败路线为 `banned`；只有工程前提或 API 契约改变，并有新的对照证据时才重新评估。

## 根因

实验先使用从非对齐偏移开始的 `Duplicate`，后改为逐元素 `SetValue`；两种填充方式都未消除异常。这说明“仅替换尾部写法”不足以满足该高阶 API 的 shape、临时空间或 buffer 契约，不能把异常单归因于 Duplicate 的地址对齐。

## 失败路线

路线优先级：`banned`。在未核清 SoftMax tiling、临时空间与真实/补齐 K 契约前，禁止继续把“补齐到 Np 后手工填负无穷”作为可交付路线。由 `Duplicate` 换成 `SetValue` 也不构成修复。

### 两条已失败的完整尾填单元

先尝试从非对齐起点做 Vector Duplicate：

```cpp
        // Pad columns [N, Np) with -inf (only when padding is needed).
        if (Np_ > N_) {
            const T negInf = static_cast<T>(-std::numeric_limits<float>::infinity());
            AscendC::Duplicate<T>(xLocal[N_], negInf, Np_ - N_);
            AscendC::PipeBarrier<PIPE_V>();
        }
```

随后只把该单元逐字替换为标量 SetValue，异常仍未消除：

```cpp
        // Pad columns [N, Np) with -inf. Duplicate() needs a 32B-aligned dst, but
        // [N, Np) starts at a non-aligned offset when N is not mult-8; use SetValue
        // (element-wise, no alignment) instead. The pad is always <= 7 elements.
        if (Np_ > N_) {
            const T negInf = static_cast<T>(-std::numeric_limits<float>::infinity());
            for (int64_t i = N_; i < Np_; ++i) {
                xLocal.SetValue(static_cast<uint32_t>(i), negInf);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
```

## 验证信号

以下信号来自本次开发会话对两种尾填写法的实际运行输出。

- 使用 `Duplicate` 尾填时，多个非对齐用例返回 `retCode=0x31, vector core exception`，而两组对齐控制与两组特殊值控制均为 `RESULT PASS`（来源平台：a3；证据等级: tool_observed）。
- 把尾填单元替换为逐元素 `SetValue` 后，非对齐用例仍返回同一 `retCode=0x31`，对齐与特殊值控制仍通过（来源平台：a3；证据等级: tool_observed）。

## 失效边界

本卡不证明高阶 SoftMax 永远不能处理尾部；只有在使用与目标 CANN 匹配的正式 tiling/shape 契约并完成非对齐验证后，才可解除 banned。
