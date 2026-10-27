---
type: Runbook
title: "507015 aicore exception under NPU contention — distinguish a device-load crash from a kernel ADDR_MISALIGN bug"
description: "Runtime 507015 aicore exception (plog fixp_error, retCode 0x26) can be host NPU contention, not a kernel ADDR_MISALIGN bug — identical plog signature. Check npu-smi AICore, re-run on an idle host."
tags: [task.debugging, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-189-507015-aicore-exception-npu-contention.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-09-16T07:35:00Z"
platforms: [a3, "950"]
aliases: ["OL-189"]
---

# 507015 aicore exception under NPU contention — distinguish a device-load crash from a kernel ADDR_MISALIGN bug

> 原 OKF v1 卡号：**OL-189**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - 507015 aicore exception with plog fixp_error0/fixp_error1, retCode=0x26, subErrType:4

## 现象 / 触发
Runtime error `507015 aicore exception` with plog `fixp_error0`/`fixp_error1` and `retCode=0x26` (subErrType:4). The plog signature is **IDENTICAL** whether the cause is a genuine memory-access violation (kernel ADDR_MISALIGN) or **host-level NPU contention** (ray/VeRL training occupying the NPU, AICore >100%) — the error code alone cannot distinguish them.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
Before concluding a kernel bug, check contention:
```bash
npu-smi info | grep AICore
```
- AICore > 50% → NPU under load → contention is a candidate
- AICore > 100% → oversubscribed → contention is the **primary** suspect
- AICore = 0% (all idle) → contention unlikely → investigate the kernel

**Distinguishing test**: re-run the same kernel binary on an **uncontended** host (or the same host after load clears). Passes on the clean host → contention was the cause; the same 507015 reproduces on an idle host → investigate a real ADDR_MISALIGN.

**Evidence (2026-05-27)**: same FA kernel binary, same cann=9.0.0 — host `198.51.100.70` (16/16 chips AICore 162–171%, ray/VeRL) → 0/9 pass, every case 507015 (plog `fixp_error0=0xf7 fixp_error1=0xa9 subErrType:4`); host `198.51.100.92` (NPU idle) → 9/9 cases dispatched without crash. Same binary, different host state → 507015 is **device-state-dependent, not kernel-inherent**.

**Self-inflicted device-poison extension (FA-A5, 2026-06-02)**: the same 507015/hang regime also arises WITHOUT external contention — a kernel that hangs or faults **poisons the device for ~547s** (aicpu op-execute timeout), and the next case sees a 507015 cascade that is NOT its own bug. Two mitigations for crash-prone / multi-core verify: (1) run **each case in its own subprocess with a hard timeout** — a hang/507015 otherwise poisons the device for the rest of the process, producing a false cascade-FAIL; on wedge, record `WEDGED` and skip, do NOT retry-in-place. (2) `docker restart` does **NOT** clear a host-level stuck aicore (the 547s wedge persists across container restart) — swap to a genuinely idle card. So when triaging a multi-case crash run, isolate per-case before concluding any single case is a real kernel bug.

**Prevention**: pre-launch `npu-smi` check in the harness/orchestrator before kernel verify; if target AICore > 50%, warn or route to a different NPU.

Verified on soc=Ascend910_9382, cann=9.0.0.
