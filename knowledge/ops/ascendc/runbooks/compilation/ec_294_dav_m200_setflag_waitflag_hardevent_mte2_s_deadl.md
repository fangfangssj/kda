---
type: Runbook
title: "dav-m200: `SetFlag`/`WaitFlag<HardEvent::MTE2_S>` DEADLOCKS for scalar consumers of a TQue-dequeued VECIN buffer"
description: "dav-m200 上对 TQue 出队的 VECIN buffer 做 `SetFlag/WaitFlag<MTE2_S>` 会死锁（标量消费者场景）；回退方案是标量 GM GetValue。"
tags: [task.debugging, topic.data_movement.data_copy, topic.implementation.synchronization, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-294", "SetFlag", "WaitFlag"]
---

# dav-m200: `SetFlag`/`WaitFlag<HardEvent::MTE2_S>` DEADLOCKS for scalar consumers of a TQue-dequeued VECIN buffer

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.1.0; op_class=any scalar consumer of a TQue-dequeued VECIN buffer`
**EC-294 — dav-m200: `SetFlag`/`WaitFlag<HardEvent::MTE2_S>` DEADLOCKS for scalar consumers of a TQue-dequeued VECIN buffer** (kernel hang at device sync; killed). The SDK's only MTE2_S precedent (adv_api matmul `data_copy_wrapper_using_ub_nd.h`) is cube-path code. Do NOT route scalar consumers through MTE2_S on arch 200; scalar GM `GetValue` (~100cyc) is the working fallback. 15_Pad ko-8 Opt4(a) 2026-09-01. Cross-ref [`EC-13`](ec_13_ascendc_syncfunc_does_not_exist.md), [`EC-73`](ec_73_a_scalar_read_getvalue_raw_gm_deref_of_data_just_d.md) (which needs MTE2_S — that is a DIFFERENT data-copy→scalar-read order on A5/950PR; arch-200 hangs). backend=ascendc.
