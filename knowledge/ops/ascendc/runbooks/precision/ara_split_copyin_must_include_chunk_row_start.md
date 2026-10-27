---
type: "Runbook"
title: "ARA 分块重复首块：CopyIn 传递 rowStart"
description: "ARA_SPLIT 循环推进分块游标却未用于 GM 起点，导致每轮重复读取首块；将 rowStart 纳入地址并由各 pass 传入。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# ARA 分块重复首块：CopyIn 传递 rowStart

## 现象

在 AscendC kernel 直调工程中实现非末轴归约的三遍 `ARA_SPLIT` 路径时，归约轴只需一个 chunk 的输入正确；归约轴超过一个 `rChunk` 后，随机输入的元素分布明显偏离参考结果，但输出沿归约轴求和仍为 1。全零输入也会通过，因为重复读取首块仍能得到均匀分布，容易把问题误判成跨 chunk 的 max/sum 数值累积错误。

## 适用场景

- 工程形态：AscendC kernel 直调；将张量合轴为 `[A1, R, A0]`，对 `R` 做多遍分块归约。
- API 路径：`DataCopyPad` 的 GM→UB 跨步 gather；每个 pass 以 `r0` 推进当前 `rChunk`。
- 验证过的平台与 CANN 版本：Ascend 950，CANN 9.0.1；与 frontmatter `platforms: [950]` 一致。
- 适用粒度：`operator_family`，适用于把归约轴分成多个 chunk、且 helper 自行计算 GM 首地址的归约/归一化 kernel。
- 路线优先级：`recommended_fastpath`。它保留多遍分块路线，不引入标量搬运 fallback；本轨迹验证了正确性与完整评测，但未证明整个算子实现相对其他算法性能最优。
- 不适用或未验证：单 chunk 路径不会触发；未在 A2/A3 或不同 `DataCopyPad` 地址单位契约上验证。

## 根因

`ProcessBlockARASplit` 的三个循环都推进 `r0`，但旧版 `CopyInBlock` 的 `gmOffset` 只计算 `a1Base + a0Start`，没有包含 `r0 * A0_`，所以每个 pass 的所有迭代都重新 gather 归约轴首个 chunk。

## 解决配方

给 gather helper 增加 `rowStart` 参数，并把 GM 起点写成 `a1Base + rowStart * A0_ + a0Start`；整块路径传 `0`，分块路径的 max、sum、output 三个 pass 都传当前 `r0`。

### before

revision: trace before-fix snapshot
path: `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`
symbol: `KernelSoftmax<T>::CopyInBlock`
lines: 505-522（轨迹读取快照）
supports: GM 起点未消费分块游标这一根因。

```cpp
    __aicore__ inline void CopyInBlock(
        int64_t a1Idx, int64_t a0Start,
        int32_t tileA0Clamped, int32_t rows)
    {
        auto xLocal = inQueueX_.AllocTensor<T>();
        int64_t a1Base = a1Idx * R_ * A0_;
        int64_t gmOffset = a1Base + a0Start;
        // Multi-row strided gather: blockCount=rows, dstStride=0 (UB packed).
        // On 950: srcStride in bytes (GM), dstStride in 32 B blocks (UB).
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(tileA0Clamped * sizeof(T)),
            static_cast<uint32_t>((A0_ - tileA0Clamped) * static_cast<int64_t>(sizeof(T))),
            0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[gmOffset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }
```

### after

revision: trace after-fix snapshot
path: `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`
symbol: `KernelSoftmax<T>::CopyInBlock`
lines: 509-527（轨迹写入快照）
supports: `rowStart * A0_` 进入 GM 起点，允许每次 gather 读取当前 chunk。

```cpp
    // rowStart selects the first R-row of the [R, A0] block to gather. ARA_FULL
    // always loads the whole block (rowStart=0); ARA_SPLIT loads one rChunk per
    // call (rowStart=r0). Omitting rowStart made every SPLIT pass re-read the
    // first chunk, corrupting results whenever R spanned >1 rChunk.
    __aicore__ inline void CopyInBlock(
        int64_t a1Idx, int64_t a0Start,
        int32_t tileA0Clamped, int32_t rows, int32_t rowStart)
    {
        auto xLocal = inQueueX_.AllocTensor<T>();
        int64_t a1Base = a1Idx * R_ * A0_;
        int64_t gmOffset = a1Base
            + static_cast<int64_t>(rowStart) * A0_ + a0Start;
        // Multi-row strided gather: blockCount=rows, dstStride=0 (UB packed).
        // On 950: srcStride in bytes (GM), dstStride in 32 B blocks (UB).
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(tileA0Clamped * sizeof(T)),
            static_cast<uint32_t>((A0_ - tileA0Clamped) * static_cast<int64_t>(sizeof(T))),
            0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[gmOffset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }
```

调用契约也必须同步：整块 `ARA_FULL` 传 `rowStart=0`；`ARA_SPLIT` 三个 pass 都传本轮 `r0`。只改 helper 签名但漏改任一 pass，会让该 pass 继续读取错误 chunk。

## 设计决策

1. 选择把 `rowStart` 显式传给搬运 helper，因为 `r0` 属于调用侧的分块状态，helper 不能从 `rows` 或 `a0Start` 反推出当前 chunk。
2. 放弃“只修 max 或 sum 累积公式”的路线：全零输入通过、随机输入失败的信号说明三遍数学结构并非唯一问题；地址轨迹显示所有 pass 都重复首块。
3. 当前方案是 `recommended_fastpath`：保持原有多遍分块和多行 `DataCopyPad`，只修正 GM 起点，不退化为逐元素搬运。

## 系统信息

- 平台：Ascend 950 / CANN 9.0.1。
- 工程：AscendC kernel 直调，代码文件 `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`。
- 相关符号：`ProcessBlockARASplit`、`CopyInBlock`、`r0`、`rChunk_`、`A0_`、`gmOffset`。

## 预防

- 对每个分块循环列出地址公式，确认循环游标确实进入 GM/UB 首地址。
- 用一组 `R <= rChunk` 与一组 `R > rChunk` 输入做成对验证；不要只测全零或均匀输入。
- 三遍算法逐 pass 检查：max pass、sum pass、output pass 都必须读取同一轮对应的 chunk。
- 把“循环变量推进”与“地址推进”作为两个独立 review 项，避免代码看起来在分块、实际地址不变。

## 验证信号

来源说明：一次 AscendC Softmax 直调开发会话在真实重编译后先复现跨 chunk 错误，再修改 `CopyInBlock` 地址与四个调用点，并在 Ascend 950 上重新构建和运行三种 dtype probes。

- 修复前，两个跨 chunk 随机输入 probe 的 `max_err` 分别达到 0.926 与 0.320，同时归约轴求和检查仍通过（来源平台：950；证据等级：tool_observed）。
- 修复后，fp32 probes 最坏 `max_err=1.192e-07` 且全部通过（来源平台：950；证据等级：tool_observed）。
- 修复后，fp16 probes 最坏 `max_err=2.384e-04` 且全部通过（来源平台：950；证据等级：tool_observed）。
- 修复后，bf16 probes 最坏 `max_err=1.838e-03` 且全部通过（来源平台：950；证据等级：tool_observed）。
- 修复后的 wheel 经 `build.sh --soc=ascend950 --install` 完成真实编译与安装（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 单 chunk 场景不会暴露该错误，不能据此证明地址推进正确。
- 本结论假设逻辑布局为 `[A1, R, A0]`，相邻 `R` 行的 GM pitch 为 `A0_` 元素；其他布局必须重算地址式。
- A2/A3 未验证，推断可迁移；其 `DataCopyPad` 参数单位与地址限制仍须按对应版本文档复核。

## 坏实践

- 在 `for (r0 += rChunk)` 中只更新循环变量，却让搬运 helper 始终使用固定 GM 起点。
- 只用全零输入验证分块归约；重复首块仍可产生归一化和为 1 的假通过。
- 只改 output pass 的地址；max/sum pass 若仍重复首块，最终分布仍错误。

## 误诊

最初把“和为 1 但分布错误”归因于跨 chunk 的 max/sum 数值累积。全零通过与随机输入失败并不能单独证明数值问题；应先核对每个 pass 的实际 GM 起点。