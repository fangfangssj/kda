---
type: Runbook
title: "dav-m200: a pure 2-byte GM scalar store (`GlobalTensor<T>::SetValue`, T∈{half,int16}) sharing a 64B line with MTE3-DMA-written data zero-clobbers the PRECEDING 32B block, in either issue order"
description: "dav-m200 上 2 字节标量 GM 写若与 MTE3-DMA 数据共享 64B 行，会把前 32B 块清零（任意发射顺序、同核确定性）——PB 级硬件坑。"
tags: [task.debugging, topic.data_movement.data_copy, topic.compute.elementwise.type_conversion, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-296", "GlobalTensor"]
---

# dav-m200: a pure 2-byte GM scalar store (`GlobalTensor<T>::SetValue`, T∈{half,int16}) sharing a 64B line with MTE3-DMA-written data zero-clobbers the PRECEDING 32B block, in either issue order

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; op_class=all (any UB->GM store path with scalar + DMA on the same 64B line)`
**EC-296 — dav-m200: a pure 2-byte GM scalar store (`GlobalTensor<T>::SetValue`, T∈{half,int16}) sharing a 64B line with MTE3-DMA-written data zero-clobbers the PRECEDING 32B block, in either issue order** — observed 3 rebuilds (21_Scatter kw-2: M=216/217/221 fp16/bf16 lose exactly [M-24,M-8) while the scalar tail and earlier data stay correct; a standalone DMA of the same 416B is clean; fp32 scalar stores at the same geometry are clean). Mitigation: M-anchored overlapping 32B tail DataCopy [end-ALIGN,end) — never past valid range, idempotent; GM scalar stores ONLY in DMA-free regions; host nblk clamp forbids sub-ALIGN multi-core final slices. Supersedes the naive P-P123 scalar-SetValue tail read when DMA shares the line. Cross-ref CAND-DAV-M200-SCALAR-STORE-DMA-LINE-CLOBBER. backend=ascendc.
