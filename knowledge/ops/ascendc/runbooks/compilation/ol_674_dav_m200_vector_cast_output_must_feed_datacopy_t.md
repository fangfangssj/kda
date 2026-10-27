---
type: Runbook
title: "dav-m200: vector-Cast output MUST feed DataCopy through a VECOUT TQue"
description: "dav-m200 上 VEC Cast 的输出必须经 VECOUT TQue 喂给 DataCopy——裸 TBuf+S_MTE3 flag 不定序，首 tile 垃圾（~50% 稳定特征）。"
tags: [task.debugging, topic.data_movement.data_copy, topic.implementation.synchronization, topic.compute.elementwise.type_conversion, topic.programming_model.execution_unit.vector]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-674"]
---

# dav-m200: vector-Cast output MUST feed DataCopy through a VECOUT TQue

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; op_class=any kernel path feeding VEC Cast output into MTE3 DataCopy`
**OL-674 — dav-m200: vector-Cast output MUST feed DataCopy through a VECOUT TQue** — bare TBuf + SetFlag/WaitFlag(S_MTE3) does NOT order VEC-pipe Cast writes before an MTE3 DataCopy (first-tile-per-window garbage, stable ~50% signature). Correct idiom: outQue TQue<VECOUT,2> Alloc→Cast→EnQue→DeQue→DataCopy→Free; add V_S before scalar reads / S_V before VEC reads of scalar-written UB. Extends PB-74's cast whitelist with the pipe-ordering requirement. 21_Scatter ko-1 (case-by-case A/B). backend=ascendc.
