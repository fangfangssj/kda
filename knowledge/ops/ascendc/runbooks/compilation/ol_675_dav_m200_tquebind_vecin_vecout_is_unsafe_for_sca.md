---
type: Runbook
title: "dav-m200 TQueBind<VECIN,VECOUT> is UNSAFE for scalar-consumer RMW"
description: "dav-m200 上 TQueBind<VECIN,VECOUT> 对标量消费型 RMW 不安全——DeQue 只保 MTE2→MTE3 不保 MTE2→标量，21_Scatter 出现非确定丢更新。（注：正文自报 ID 误标为 OL-679，待修）"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-675"]
---

# dav-m200 TQueBind<VECIN,VECOUT> is UNSAFE for scalar-consumer RMW

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; op_class=any scalar consumer of a TQue<VECIN,VECOUT> buffer`
**OL-675 — dav-m200 TQueBind<VECIN,VECOUT> is UNSAFE for scalar-consumer RMW** — DeQue orders MTE2→MTE3, NOT MTE2→scalar; a scalar read-modify-write between DeQue and out-DataCopy races the incoming DMA (21_Scatter case-27 nondeterministic lost updates; PipeBarrier<PIPE_MTE3> before FreeTensor does not close the read side). Carry this caveat on OL-49's "TQueBind unverified" flag. backend=ascendc.
