---
type: "Runbook"
title: CopyOut 改 blockCount 批量搬运回归回退
description: fp32/fp16/bf16 的 UB→GM CopyOut 由逐行 DataCopyPad 改为 DataCopyExtParams blockCount=rows 批量模式（srcStride 按 32B 块计 UB 行 padding 间隙）后通过率 71.4%→64.3%，L1_017/L1_023 重新失败；机制未定位即回退，批量模式在此场景判不可用（banned）。
tags: [task.implementation, topic.precision]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-16T11:20:30Z'
platforms: ["950"]
---

# CopyOut 改 blockCount 批量搬运回归回退

## 现象

BenchNoHarnessDepthwiseConv2d（Ascend950PR/arch35，CANN 9.2.0）迭代二联调中，fp32 路径的 V→MTE3 同步问题已用 PipeBarrier 修复（逐行 CopyOut + 屏障，通过率 71.4%，L1_017/L1_023 通过）。为减少逐行搬运的 DMA 次数，把三个 dtype 路径的 CopyOut 从"每行一次 DataCopyPad"改为"一次 DataCopyPad、blockCount=rows、srcStride 跳过 UB 行 padding"的批量模式，期望行为不变。结果通过率回落到 64.3%：L1_017 fp32 5x5 重新报 `[FAIL] MERE=8.06e-01 (thr 1.22e-04), MARE=2.02e+03`，L1_023 同类失败，L1_010 也失败（`MERE=8.68e-01, MARE=9.89e+01`）。回退逐行 + 保留 PipeBarrier 后恢复 71.4%。

## 适用场景

- 工程形态：registry-invoke 自定义算子仓开发（AscendC kernel，UB 中行带 padding 间隙、GM 中行连续的 CopyOut）。
- API 路径：`DataCopyPad(LocalTensor→GlobalTensor)` + `DataCopyExtParams{blockCount, blockLen, srcStride, dstStride}`；UB 侧 srcStride 单位为 32B 块、GM 侧为字节。
- 验证过的平台与 CANN 版本：Ascend950PR（arch35），CANN 9.2.0（上板 ST 实测）。
- 适用粒度：operator_family——任何"UB 行 stride ≠ GM 行 stride（padding 间隙）"的多行批量搬出尝试。
- 路线优先级：banned（本场景禁用 blockCount 批量 CopyOut；逐行 + PipeBarrier 为可用路线）。作为负结果保留：该路线可重建、失败为客观观测，但不暗示已有批量模式的修复。
- 明确不适用 / 未验证场景：UB 行间无 padding（行 stride 与 GM 完全一致）时的整块搬运未在本会话验证；其他 CANN 版本未验证。

## 尝试（可重建）

批量模式实现（fp32 路径；fp16/bf16 同型，仅元素尺寸不同）：

```cpp
// op_kernel/arch35/bench_no_harness_depthwise_conv_2d.h（尝试：批量 CopyOut）
// 3.3 fp32 无需 Cast，直接从 accBuf CopyOut 到 GM
//    使用 blockCount 模式批量搬运所有行，srcStride 处理 UB 行间距
//    UB 中每行 floatStride_ 个 float，有效数据 WOut_ 个，padding (floatStride_ - WOut_) 个
//    srcStride 单位是 32B 块 = (floatStride_ - WOut_) * sizeof(float) / 32
//    GM 中每行 WOut_ 个 float，连续存储，dstStride = 0
int64_t outHStart = hTileStart;
int64_t outOffset = yOffset + outHStart * WOut_;
uint32_t srcStride = static_cast<uint32_t>((floatStride_ - WOut_) * sizeof(float) / 32);
DataCopyExtParams outCopyParams{
    static_cast<uint16_t>(rows),
    static_cast<uint32_t>(WOut_ * sizeof(float)),
    srcStride, 0, 0};
DataCopyPad(yGm_[outOffset], accFloat, outCopyParams);
```

fp16 路径：`uint32_t srcStrideHalf = static_cast<uint32_t>((halfStride_ - WOut_) * sizeof(half) / 32);`，blockLen=`WOut_ * sizeof(half)`；bf16 同型（bf16Stride_）。

## 验证信号（成对回退观测）

以下信号来自该算子迭代二联调（Ascend950PR 上板 ST）；批量与逐行两路线成对对照，构成本负结果的客观观测。

- 批量模式全量 ST：`总计: 43 通过: 18 失败: 10 通过率: 64.3%`（来源平台：950；证据等级：tool_observed）。
- 对照逐行 + PipeBarrier 路线：`71.4%`（20 通过），L1_017（批量模式下 `[FAIL] MERE=8.06e-01 (thr 1.22e-04), MARE=2.02e+03`）、L1_023、L1_010（`MERE=8.68e-01, MARE=9.89e+01`）在批量模式重新失败（来源平台：950；证据等级：tool_observed）。
- 回退逐行（保留 PipeBarrier）后 L1_017/L1_023 恢复 PASS、通过率回到 71.4%——同代码两路线成对回退信号（来源平台：950；证据等级：tool_observed）。
- 机制未定位：blockLen=224B（56×4，32B 对齐）、srcStride=1 块（8 float 间隙=32B）参数表面合规，未见独立解释（见失效边界与遗留问题）。

## 失效边界

- 回退即终点：本会话未继续定位批量模式失败机制（候选解释：UB 侧 srcStride 语义、块计数模式下的 MTE 行为、或与屏障的交互），批量模式在本算子结构下判不可用。
- 可用路线（同会话验证）：逐行 `DataCopyPad` + `PipeBarrier<PIPE_ALL>()`，通过率 71.4% → 后续叠加 bf16 修复至 85.7% → 小值域判定后 100%。
- 相对代价：批量模式若可用可减少 DMA 次数（rows 次 → 1 次），但正确性不可用；逐行路线的每行一次 DMA 为已接受的成对代价（来源平台：950；相对参照：批量模式路线，未量化）。

**未决项**：

- 优先项：复现并追根因到机制层——用最小 kernel（固定 accFloat 内容 + 批量 CopyOut + GM 读回）逐参数（blockCount、srcStride 单位、blockLen 对齐）定位 arch35 上批量模式失败的确切机制；确认后本卡可升级为精确的修复卡或确认禁用。
- 块对齐（blockLen 为 32B 倍数）与 srcStride 单位（32B 块）的文档化语义需与 CANN 头文件实现核对（本会话仅凭经验注释）。
## 坏实践

- 在未定位失败机制前把"看起来等价"的搬运模式当作安全重构——批量与逐行的 srcStride/块语义差异足以改变正确性，重构必须全量回归。
- 一次重构三个 dtype 路径的 CopyOut 再统一验证——失败时无法隔离是哪个路径/哪种参数组合的问题（本会话幸运地全部回退成功）。
