---
type: Guide
title: "Debug markers must not SetFlag a shared functional event at pending==1 (depth-2 excursion = the wedge class being diagnosed)"
description: "Progress markers SetFlag+WaitFlag on a functional event id whose pending is 1 at entry drive the counter 1->2 — same overflow class as the wedge under diagnosis; stripping markers changed a deterministic wedge to pass. Use a dedicated event id or assert pending==0; strip after localization. Found on 11_BigBirdAttention CubeMark_ eFixM_."
tags: [task.debugging, topic.implementation.synchronization]
status: draft
sources: []
created_at: "2026-09-26T12:23:46Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases:
- OL-611
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# Debug markers must not SetFlag a shared functional event at pending==1 (depth-2 excursion = the wedge class being diagnosed)

## Evidence

- **run_id**: 5450b083-4f53-4620-a1b2-8656e7bb9889
- **soc**: Ascend950DT_9582
- **clock**: 2026-09-26T12:23Z
- **trigger**: 11_BigBirdAttention: marker strip changed deterministic wedge to pass; CubeMark_ depth-2 eFixM_ excursion identified

## 现象 / 触发
Progress markers added to diagnose a MIX deadlock did `SetFlag<FIX_M>` + `WaitFlag<FIX_M>`
on the kernel's functional eFixM_ event. At several call sites the functional pending
count is 1 at entry, so the marker Set drives the counter 1->2 — the same
event-counter-overflow class as the AIV V_MTE2 wedge being diagnosed (5 excursions/item
on the AIC FIX pipe). With markers present a shape wedged deterministically; after
no-op'ing the marker bodies it passed.

## 规则
Kernel debug markers that Set/Wait must either use a DEDICATED event id (own FetchEventID,
never shared with functional sync) or assert pending==0 at entry. Strip instrumentation
after it has served its localization purpose — it perturbs pipe timing and can be the
residual wedge source itself.

## 相关

- [OBSERVED — 507014 deadlock in MatmulImpl + manual CrossCore* + MIX_AIC_1_2](../runbooks/compilation/mix_aic_1_2_matmulimpl_manual_crosscore_507014_deadlock.md) — 同错误码同 kernel 族的观测记录（机制 UNDETERMINED）；本条为 instrument 侧的独立已证机制。
