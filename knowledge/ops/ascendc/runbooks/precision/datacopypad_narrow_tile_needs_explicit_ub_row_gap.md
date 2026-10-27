---
type: "Runbook"
title: "窄尾 tile 行距错位：补齐 UB gap"
description: "DataCopyPad 按实际窄块紧排 UB 行，而 Cast/计算按完整 tile 行距访问；在搬入 dstStride 与搬出 srcStride 中补齐 UB gap。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.tiling"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 窄尾 tile 行距错位：补齐 UB gap

## 现象

在 AscendC kernel 直调的非末轴归约中，完整 `tileA0_` 的 ARA 路径可以通过，但最后一个 `tileA0Clamped < tileA0_` 的窄尾 tile 会出现精度失败，严重时没有有效运行结果。失败跨越普通值与极值输入，表面像计算精度或数据范围问题，实际共同条件是多行 UB 中最后一个 A0 tile 的逻辑宽度小于固定计算行距。

## 适用场景

- 工程形态：AscendC kernel 直调；GM→UB 多行 gather、UB 内 Cast/计算、UB→GM 多行 scatter。
- API 路径：`DataCopyPad` + `DataCopyExtParams`，UB 侧 stride 单位为 32B data block；计算侧按 `rowBytesT_`/`rowElemsF_` 跨行。
- 验证过的平台与 CANN 版本：Ascend 950，CANN 9.0.1；与 frontmatter `platforms: [950]` 一致。
- 适用粒度：`operator_family`，适用于完整 tile 分配固定 UB 行距、尾 tile 仅搬运实际列数的非末轴归约/归一化 kernel。
- 路线优先级：`recommended_fastpath`。它保留多行 `DataCopyPad`，利用 stride 表达固定物理行距，不退化为逐元素尾部搬运；本轨迹验证了正确性与完整评测，但未与其他搬运算法做性能对照。
- 不适用或未验证：每行实际 padded bytes 恰好等于 `rowBytesT_` 时 gap 为 0；未验证 Compact padding mode、负 stride 或 A2/A3。

## 根因

`DataCopyPad` 会把每个 `blockLen` 的实际数据补齐到 32B；旧版搬入 `dstStride=0`、搬出 `srcStride=0`，因此窄尾 tile 的 UB 行距是 `Align32(tileA0Clamped*sizeof(T))`，而 Cast 与计算始终按完整 tile 的 `rowBytesT_` 跨行，二者相差 `rowBytesT_-Align32(actualBytes)`，导致第二行起读取或写回错误地址。

## 解决配方

分别在 GM→UB 的 `dstStride` 与 UB→GM 的 `srcStride` 中补上相同的 UB gap：

`ubGapBlocks = (rowBytesT_ - Align32(tileA0Clamped * sizeof(T))) / 32`

GM 侧 `srcStride`/`dstStride` 仍表示相邻逻辑行之间跳过的字节数，不要把 UB gap 与 GM gap 混用。

### before

revision: trace before-fix snapshot
path: `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`
symbol: `KernelSoftmax<T>::CopyInBlock`
lines: 509-527（轨迹写入快照）
supports: 搬入 `dstStride=0` 只形成实际窄块的 32B 对齐行距。

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

revision: trace before-fix snapshot
path: `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`
symbol: `KernelSoftmax<T>::CopyOutRows`
lines: 539-552（轨迹读取快照）
supports: 搬出 `srcStride=0` 假设窄尾结果在 UB 中紧排。

```cpp
    __aicore__ inline void CopyOutRows(
        int64_t gmOffset, int32_t tileA0Clamped,
        int32_t rows, AscendC::LocalTensor<T>& zLocal)
    {
        // Multi-row strided scatter: blockCount=rows, srcStride=0 (UB packed).
        // On 950 for UBToGM: srcStride in 32 B blocks (UB), dstStride in bytes (GM).
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(tileA0Clamped * sizeof(T)),
            0,
            static_cast<uint32_t>((A0_ - tileA0Clamped) * static_cast<int64_t>(sizeof(T))),
            0};
        AscendC::DataCopyPad(yGm_[gmOffset], zLocal, copyParams);
    }
```

### after

revision: trace after-fix snapshot
path: `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`
symbol: `KernelSoftmax<T>::CopyInBlock`
lines: function-level reconstruction from verbatim trace write
supports: `dstStride=ubGapBlocks` 把每个搬入行补到固定 `rowBytesT_`。

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
        // UB rows are laid out at the FULL-tile stride rowBytesT_ (compute and
        // CastBlockToFP32 stride by rowBytesT_). DataCopyPad pads each block to
        // 32 B (actualPaddedBytes) then inserts dstStride*32 B of gap. For a
        // full tile gap=0; for a narrow tail tile (tileA0Clamped<tileA0_) the
        // gap pads each row up to rowBytesT_ so compute reads valid data.
        int32_t actualPaddedBytes = Align32Bytes(
            tileA0Clamped * static_cast<int32_t>(sizeof(T)));
        int32_t ubGapBlocks = (rowBytesT_ - actualPaddedBytes) / 32;
        // Multi-row strided gather: blockCount=rows.
        // On 950: srcStride in bytes (GM), dstStride in 32 B blocks (UB).
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(tileA0Clamped * sizeof(T)),
            static_cast<uint32_t>((A0_ - tileA0Clamped) * static_cast<int64_t>(sizeof(T))),
            static_cast<uint32_t>(ubGapBlocks), 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm_[gmOffset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }
```

revision: trace after-fix snapshot
path: `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`
symbol: `KernelSoftmax<T>::CopyOutRows`
lines: function-level verbatim trace write
supports: `srcStride=ubGapBlocks` 让 scatter 从固定 `rowBytesT_` 的下一行继续读取。

```cpp
    __aicore__ inline void CopyOutRows(
        int64_t gmOffset, int32_t tileA0Clamped,
        int32_t rows, AscendC::LocalTensor<T>& zLocal)
    {
        // UB rows were written at the FULL-tile stride rowBytesT_ (compute and
        // CastBlockFromFP32 stride by rowBytesT_). srcStride (UB gap, 32 B
        // blocks) skips the padding tail of each narrow row so the scatter
        // reads the correct elements. GM-side dstStride (bytes) is unchanged.
        int32_t actualPaddedBytes = Align32Bytes(
            tileA0Clamped * static_cast<int32_t>(sizeof(T)));
        int32_t ubGapBlocks = (rowBytesT_ - actualPaddedBytes) / 32;
        // Multi-row strided scatter: blockCount=rows.
        // On 950 for UBToGM: srcStride in 32 B blocks (UB), dstStride in bytes (GM).
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(tileA0Clamped * sizeof(T)),
            static_cast<uint32_t>(ubGapBlocks),
            static_cast<uint32_t>((A0_ - tileA0Clamped) * static_cast<int64_t>(sizeof(T))),
            0};
        AscendC::DataCopyPad(yGm_[gmOffset], zLocal, copyParams);
    }
```

## 设计决策

1. 选择在 `DataCopyExtParams` 中表达 UB gap，因为计算与 Cast 已以完整 tile 规划固定行距，改变搬运 stride 是最小且保持批量搬运的修复。
2. 放弃“把计算行距改成每个尾 tile 动态变化”的路线：这会同时影响 fp32 buffer、原 dtype queue、Cast helper 和多个 pass，扩大状态组合。
3. 放弃逐元素标量尾部搬运：它可作为保守诊断，但会拆散多行 `DataCopyPad`；当前 `recommended_fastpath` 直接使用 API 已有的 UB stride 语义。

## 系统信息

- 平台：Ascend 950 / CANN 9.0.1。
- 工程：AscendC kernel 直调，代码文件 `example/csrc/ops/softmax/op_kernel/softmax_kernel.cpp`。
- 相关符号：`CopyInBlock`、`CopyOutRows`、`CastBlockToFP32`、`CastBlockFromFP32`、`rowBytesT_`、`tileA0Clamped`。

## 预防

- 同时写出三种行距：GM pitch、`DataCopyPad` 实际 padded bytes、计算/Cast 的 UB row stride。
- 对搬入检查 `dstStride`，对搬出检查 `srcStride`；两者都以 UB 侧 data block 为单位。
- 必测 `tileA0Clamped == tileA0_` 与 `tileA0Clamped < tileA0_`；尾 tile 最窄时最容易暴露跨行错位。
- 当 queue 按最大 tile 分配时，不要据此假设 `DataCopyPad(dstStride=0)` 会自动占满最大行距。

## 验证信号

来源说明：一次 AscendC Softmax 直调开发会话先在标准评测中定位两个窄尾 tile 失败，再同时修正 GM→UB 和 UB→GM 的 UB gap，真实重编译后运行定向 probes、原失败场景与完整评测。

- 修复前标准正确性评测通过 18/20，两个窄尾 tile 场景失败，其中一个没有有效运行结果、另一个精度误差为 1.0（来源平台：950；证据等级：tool_observed）。
- 修复后，fp32 窄尾 probes 最坏 `max_err=1.192e-07`，fp16 为 `2.430e-04`，bf16 为 `1.952e-03`，全部通过（来源平台：950；证据等级：tool_observed）。
- 修复后的边界值窄尾 probe `max_err=8.139e-05` 且无 NaN（来源平台：950；证据等级：tool_observed）。
- 修复后的大输入 bf16 定向验证 `max_err=0.001953125`，无 NaN 且归约轴求和检查通过（来源平台：950；证据等级：tool_observed）。
- 完整标准评测由 18/20 提升到 20/20（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 若计算/Cast 本身也使用 `Align32(actualBytes)` 的动态行距，则不应再补到完整 `rowBytesT_`；生产者与消费者必须采用同一物理布局。
- `ubGapBlocks` 要求差值非负且是 32B 的整数倍；若 buffer 布局不是 32B 对齐，须重新设计而不是截断除法。
- A2/A3 未验证，推断可迁移；必须复核对应产品的 `DataCopyExtParams` 类型、单位与范围。

## 坏实践

- 把 `dstStride=0`/`srcStride=0` 理解成“按已分配 queue 的最大行宽跨行”；API 实际按本次 blockLen 的 padded bytes 继续下一行。
- 只修搬入不修搬出，或只修搬出不修搬入；两侧任何一个物理行距不同都会破坏闭环。
- 看到极值输入失败就只调数值公式，而不比较完整 tile 与窄尾 tile 的布局差异。

## 误诊

两个失败场景的数据范围和 dtype 不同，容易被归因于 bf16 或极值精度；共同的结构信号是 `A0` 留下窄尾 tile，且完整 tile 路径已通过。