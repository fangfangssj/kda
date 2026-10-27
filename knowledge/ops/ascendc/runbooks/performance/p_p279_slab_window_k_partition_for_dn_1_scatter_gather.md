---
type: Runbook
title: "Slab-window k-partition for dn>=1 scatter/gather-class ops"
description: "连续内存上 dn≥1 的 scatter/gather，利用地址可计算性让每个输出窗口只扫自己重叠的 k 区间——省掉全 K×核数重复扫描和逐命中 GM RMW，单所有者天然确定性"
tags: [task.optimization, topic.data_movement.data_copy, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["P-P279"]
---

# Slab-window k-partition for dn>=1 scatter/gather-class ops

## 优化点

For contiguous x with scatter dim dn>=1: `outer(p)` is a multiple of `stride[dn-1]` in the prefix part and `suffixPart < stride[dn]`, so `o = outer + idx*stride[dn]` stays inside the contiguous slab `[P*stride[dn-1], (P+1)*stride[dn-1])` of k's prefix P — and k's with prefix P are one contiguous k-range (row-major). A window over output can scan ONLY its own overlapping slabs' k-range.

**Win**: kills full-K×nblk scan replication AND per-hit GM RMW. Single-owner per output slot → P-P67 determinism WITHOUT atomics (contrast P-P2/P-P10/P-P17/P-P21/P-P25 atomic routes).

**Anchor**: 21_Scatter ko-1 ScatterSlabOp (2026-09-01): geomean 0.43→0.999x; case 21 [4096,11008] fp16 2161→233ms, case 42 201→22ms, case 45 1259→137ms, case 26 248→28ms; floors held (47/47 det, bit-exact vs CPU-native).

**Caveats**: window→k-range mapping uses Pidx of slab, NOT x-slab index, when index prefix dims < x prefix dims (nondim_reduced family — see OL-995/EC-252 sibling note). Cross-ref OL-995, CAND-DAV-M200-SLAB-WINDOW-KPARTITION, [`OL-90`](../precision/ol_90_pytorch_ub_class_ops_with_non_deterministic_refere.md), P-P67.
