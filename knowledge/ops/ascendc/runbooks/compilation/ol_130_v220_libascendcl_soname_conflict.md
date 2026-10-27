---
type: Runbook
title: "V220 A3 container has a libascendcl.so SONAME conflict between CANN 9.0.0 and ascend-toolkit 8.3"
description: "The npu-a3 container ships two CANN installs whose libascendcl.so share a SONAME; the linker may resolve the wrong one, causing silent kernel-launch failures. Build+run against the same CANN torch_npu uses (toolkit 8.3)."
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-130-v220-libascendcl-soname-conflict.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: [a3]
aliases: ["OL-130"]
---

# V220 A3 container has a libascendcl.so SONAME conflict between CANN 9.0.0 and ascend-toolkit 8.3

> 原 OKF v1 卡号：**OL-130**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - double free or corruption (!prev) on process exit; runtime segfault (exit 139) on kernel launch; TQue<VECOUT> crash with error 507035 — all on the A3 npu-a3 container

## 现象 / 触发
On the A3 `npu-a3` container, kernels fail at launch in ways that look unrelated to the kernel algorithm (observed across the DS batch 2026-05-02):
- `double free or corruption (!prev)` on process exit (31_IOU)
- Runtime segfault (exit 139) on kernel launch (15_Pad)
- `TQue<VECOUT>` crash with error 507035 (25_NLLLoss)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训
The A3 container `npu-a3` has TWO CANN installations that both ship `libascendcl.so` with the **same SONAME**. CANN 9.0.0 has `aclrtLaunchKernelWithHostArgs`; toolkit 8.3 does not. When both are loaded, the linker may resolve to the wrong one → kernel launch fails silently (segfault, double-free, garbage output).

## 解决配方
Rebuild the kernel against the SAME CANN installation that torch_npu uses at runtime. For this container, use ascend-toolkit 8.3's headers + libs for BOTH the build AND `LD_LIBRARY_PATH` at runtime.

### Evidence
- op#31 IOU pp-1 (2026-05-02): a 3-iter bisection confirmed the kernel algorithm was correct; a library-isolation probe identified the SONAME conflict. op#31 kw-6: rebuilt against toolkit 8.3 → crash resolved, kernel runs clean.

### Cross-ref
- OL-127 (CANN API surface gaps)
- OL-128 (TBuf→MTE3 coherence)
- PB-24 (TBuf GetValue interleaving)
