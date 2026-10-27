---
type: Runbook
title: "every cross-dtype torch .to() costs ~145-167us wall on this torch_npu build, even for 1K elements"
description: "这个 torch_npu build 上每次跨 dtype .to() 花 145-167µs（哪怕 1K 元素）——bf16 路径两次 cast 可改用 int16 位视图零拷贝消除"
tags: [task.optimization, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-677"]
---

# every cross-dtype torch .to() costs ~145-167us wall on this torch_npu build, even for 1K elements

`applies_to: soc=Ascend310P3; cann=9.1.0; op_class=any bf16/mixed-dtype forward path with cross-dtype .to()`
**OL-677 — every cross-dtype torch .to() costs ~145-167us wall on this torch_npu build, even for 1K elements** (bf16→fp32 137us, fp32→bf16 149us; a same-dtype .to() is a 5us no-op) — a bf16-on-fp32 path is TWO casts + kernel. Bit-move via a zero-copy int16 bit-view removes both (CAND-DAV-M200-BF16-BITMOVE). 15_Pad ko-8 measured. backend=ascendc.
