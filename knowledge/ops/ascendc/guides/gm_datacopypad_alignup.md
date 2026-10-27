---
type: Guide
title: "GM 布局已内联零填充的操作数用整块紧凑拷贝载入，根除 DataCopyPad 跨步读 AlignUp 行距错位"
description: "（A3 侧行距公式一致为推断，未实测）A3→A5 port of chunked/padded-layout operands: when the GM operand already carries its zero padding inline, load the full padded region as ONE contiguous block copy (strides all 0) instead of a strided multi-block DataCopyPad whose UB row pitch (AlignUp(blockLen,32)+dstStride*32) silently mismatches the kernel's row-index constant whenever blockLen is not 32B-aligned. Verified on 14_ChunkScaledDotKktFwd (950DT, CANN 9.2.0) 50/50 PASS_T1."
tags: [task.implementation, topic.data_movement.data_copy, topic.memory, topic.precision]
status: draft
sources: []
created_at: "2026-09-24T21:42:50Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# GM 布局已内联零填充的操作数用整块紧凑拷贝载入，根除 DataCopyPad 跨步读 AlignUp 行距错位

## Evidence

- **op**: 14_ChunkScaledDotKktFwd
- **soc**: Ascend950DT_9582
- **cann**: 9.2.0
- **result**: 50/50 PASS_T1, 28 cases bit-exact
- **workspace**: workspace/14_ChunkScaledDotKktFwd

## 关联卡

`runbooks/precision/datacopypad_narrow_tile_needs_explicit_ub_row_gap`——**同一根因公式**（DataCopyPad 把 blockLen 补齐到 32B，dstStride=0 时 UB 行距为 Align32(实际字节) 而计算按固定行距取行）。它给的治法是"保留跨步、补 UB gap"；本卡是它的对偶场景：当 GM 已内联零填充时，**直接放弃跨步形式**改整块紧凑拷贝。`datacopypad_strided_gather` 与 `v351_datacopypad_extparams_stride_byte_unit_001` 为相关但不同失效。

## 现象 / 触发

A3→A5 移植 chunked/padded 布局算子（如 14_ChunkScaledDotKktFwd 的 kt 操作数，
GM 布局 [B,M,Hg,D,64]，chunk 轴在 GM 内联零填充到 stride 64）时，惯性写法是
DataCopyPad + DataCopyExtParams{blockCount=D, blockLen=chunk*2, srcStride=(64-chunk)*2,
dstStride=(64-chunk)*2/32}。UB 行距实际是 AlignUp(blockLen,32)+dstStride*32，
当 chunk*2 不是 32 整数倍（chunk%16!=0，如 chunk∈{8,24,40,56}）时行距 ≠ kMaxChunk，
kernel 按 kMaxChunk 行距读 UB → 行 1..N-1 静默错位（行 0 恒正确）。A3 源里这是
潜在 bug，仅因 benchmark 全 chunk=64 未暴露。

## 适用场景

- 技术/问题域：AscendC 数据搬运（GM→UB DataCopyPad 多块跨步读）。
- 平台：Atlas A3、Ascend 950 系列均适用（行距公式两边一致）。
- 观测置信度：single_run（14_ChunkScaledDotKktFwd kw-1，50/50 PASS_T1 实测）。

## 修复

GM 布局已内联零填充时，把整块（含填充）当一次紧凑拷贝载入：
DataCopyExtParams{blockCount=D, blockLen=kMaxChunk*sizeof(T), 0, 0, 0}——
srcStride=dstStride=0，UB 行距恒等于布局行距，与 chunk 取值无关。填充零
正是 kernel 想要的（对 KKT 贡献 0，且被 mask/store 边界覆盖），无需剔除。

## 动作规则

1. 审查移植源码里 DataCopyPad 多块读：若 GM 操作数布局已含对齐填充，
   优先改整块紧凑拷贝（一次 DataCopyPad，strides 全 0）。
2. 必须保留跨步读时，UB 侧行距按 AlignUp(blockLen,32)+dstStride*32 重算并
   与 kernel 的行索引常量核对——不要用"看起来等于布局 stride"的假设。

## 证据

- 14_ChunkScaledDotKktFwd kw-1（Ascend950DT_9582, CANN 9.2.0）：kt 载入改整块
  紧凑拷贝后 50/50 PASS_T1（28 case bit-exact）；A3 源跨步写法在 chunk%16!=0 时
  行距 144B≠128B 的分析记录于 workspace analysis.md / knowledge_update.md。
- 关联卡：v351-datacopypad-extparams-stride-byte-unit-001（GM/UB stride 单位规则，
  本卡是其"伴随 bug"（UB 行距 AlignUp）的根除配方）。
