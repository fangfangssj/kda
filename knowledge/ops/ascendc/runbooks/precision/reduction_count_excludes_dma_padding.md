---
type: "Runbook"
title: "Reduce count 必须排除 DMA 对齐填充元素"
description: "count-form ReduceMax/ReduceSum 只应传逻辑归约长度；传 alignUp 后填充元素会参与 exp/sum 并改变 Softmax 分母。"
tags: ["task.debugging", "topic.compute.reduction", "topic.data_movement.memory_access", "topic.precision"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["a3"]
---

# Reduce count 必须排除 DMA 对齐填充元素

## 现象

非对齐行宽的 Softmax 把 reduce 的 `count` 传成对齐后长度。全零输入本应得到 `1/R`，实际却得到 `1/alignCount`，说明尾部填充值经过 `Exp` 后以 1 参与了分母。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的定向精度调试。
- API 路径：`ReduceMax`、`ReduceSum`、`count`、`DataCopyPad` 所在的 kernel、host 或构建接口链。
- 验证平台与版本：平台 a3；CANN：unknown。
- 适用粒度：`operator_family`，只迁移卡内同一机制，不外推到未验证 dtype、layout 或执行路径。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断。

## 根因

buffer 容量和 DMA 传输对齐长度是存储约束，不是数学归约范围。把 `alignCount` 传给 reduce 会显式要求 API 处理填充 lane；填充零在减最大值和指数后可能变成非零贡献。

## 解决配方

输入搬运仍按字节安全地容纳尾部，但 reduce 与逐元素计算统一使用真实长度：

```cpp
AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
const int32_t cnt = static_cast<int32_t>(R_);
```

后续 `ReduceMax`、`Adds`、`Exp`、`ReduceSum` 和 `Muls` 都传 `cnt`。不要为了规约而读取 `[R, alignUp(R))`。

## 验证信号

以下信号均来自本次开发会话修复前后的实际构建、运行或评测输出。

- 修复前：规约探针最大绝对差约为 0.017857，与逻辑长度参考不一致。（来源平台：a3；证据等级: tool_observed）。
- 修复后：规约 count 排除 DMA padding 后测试矩阵逐项 PASSED。（来源平台：a3；证据等级: tool_observed）。

## 失效边界

若使用的不是 count-form API，而是明确要求固定 block/mask 的接口，应按该接口契约提供尾部掩码或中性元。