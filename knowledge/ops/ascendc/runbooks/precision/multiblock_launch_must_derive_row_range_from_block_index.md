---
type: "Runbook"
title: "多核共享首段行区间导致部分输出未写"
description: "host 把首个行区间传给所有 block，多个核重复处理同一段；kernel 用 GetBlockIdx 推导本核区间后恢复覆盖。"
tags: ["task.debugging", "topic.implementation.synchronization", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 多核共享首段行区间导致部分输出未写

## 现象

在一个按外层行切到多个 AI Core 的前缀扫描 kernel 中，单行形态通过，而包含多条独立外层行的形态输出未初始化或错误。host 虽然算出了每个 block 的 `ranges`，launch 时却只把 `ranges[0].first/second` 作为标量参数传入；所有 block 因而重复处理第一段行，后续行没有任何 block 写回。

## 适用场景

- 工程形态：Ascend C kernel 直调，多 block 按外层逻辑行切分。
- API 路径：host `<<<numBlocks, ...>>>` launch + device `AscendC::GetBlockIdx()`。
- 验证过的平台与 CANN 版本：950（`dav-3510`）+ CANN 9.0.1。
- 适用粒度：`operator_family`，适用于 host 只能传统一 kernel 标量、每个 block 必须自行推导区间的多核算子。
- 路线优先级：`recommended_fastpath`。
- 不适用/未验证：每核参数由独立 tiling buffer 索引读取的工程不应照搬；其他平台未验证。

## 根因

一次 `<<<numBlocks>>>` launch 的函数实参对所有 block 相同。host 端即使构造了 `ranges[b]`，传 `ranges[0]` 仍会令每个 block 得到相同的 `mStart/mEnd`。正确映射是 `mStart = blockIdx * rowsPerCore`，`mEnd = min(mStart + rowsPerCore, M)`；这里 `rowsPerCore = ceil(M / numBlocks)`。device 还要在 `mStart >= M` 时早退，覆盖向上取整产生的空尾核。

## 解决配方

调用侧只传所有 block 共享的 `rowsPerCore`，由 kernel 使用真实 block id 计算本核区间。

### before

```cpp
template <typename InputT, typename ComputeT, bool kIsFloat>
__global__ __aicore__ __vector__ void cummin_kernel(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices,
    int64_t M, int64_t R, int64_t N, int64_t mStart, int64_t mEnd)
{
    KernelCummin<InputT, ComputeT, kIsFloat> op;
    op.Init(x, values, indices, M, R, N, mStart, mEnd);
    op.Process();
}
```

### after

```cpp
template <typename InputT, typename ComputeT, bool kIsFloat>
__global__ __aicore__ __vector__ void cummin_kernel(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices,
    int64_t M, int64_t R, int64_t N, int64_t rowsPerCore)
{
    int64_t bid = static_cast<int64_t>(AscendC::GetBlockIdx());
    int64_t mStart = bid * rowsPerCore;
    if (mStart >= M) { return; }
    int64_t mEnd = mStart + rowsPerCore;
    if (mEnd > M) { mEnd = M; }
    KernelCummin<InputT, ComputeT, kIsFloat> op;
    op.Init(x, values, indices, M, R, N, mStart, mEnd);
    op.Process();
}
```

## 设计决策

选择 device 侧 `GetBlockIdx()` 推导区间，是因为普通 launch 的标量实参不能随 block 改变。为每核在 host 构造 vector 却仍传首元素只是伪切分。当前方案为 `recommended_fastpath`；若已有正式 tiling buffer，可改为按 block id 索引 tiling，但必须保持每核区间互斥且覆盖完整。

## 预防

- 审核每个多核 launch：哪些参数是所有核共享，哪些必须由 `GetBlockIdx()` 派生。
- 用集合不变量检查区间：两两不重叠，且并集严格等于 `[0, M)`。
- 至少同时测试单行与多行形态；只测 `M=1` 会掩盖此错误。

## 验证信号

该结论来自 950 上的多核精度调试。修复前单行路径通过而二维/多行路径失败；切换到 `GetBlockIdx()` 后，同一套件通过数从 13 增到 29，剩余失败随后被定位为独立的寄存器 lane 宽度问题。再修复该独立问题后，float32/int32 失败计数为 0。

- 区间修复后通过数由 13 增到 29（950 / CANN 9.0.1，证据等级: tool_observed）。
- 后续独立 lane 修复后 float32/int32 失败计数为 0（950 / CANN 9.0.1，证据等级: tool_observed）。

## 失效边界

该配方只解决 block 到外层行区间的映射，不解决区间内部的寄存器 mask、dtype 或 DMA 错误。其他平台未验证，推断可迁移时应确认 `GetBlockIdx()` 与 launch 语义。

## 坏实践

在 host 构造 `ranges` 后把 `ranges[0]` 传给多 block launch，会制造“看起来已切分”的假象；标量参数不会按 block 自动索引 vector。

## 误诊

错误最初被归因于 value/index mask 配对；`M=1` 恰好通过而多行失败才暴露出另一条独立的多核覆盖根因。