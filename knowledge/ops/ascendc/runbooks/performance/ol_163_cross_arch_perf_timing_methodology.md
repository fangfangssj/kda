---
type: Runbook
title: "Cross-arch perf timing: device-event symmetric first, wrapper-inclusive fallback, N/A last resort"
description: "Timing symmetry is not API symmetry: with different A3/A5 APIs the perf ratio is still measurable by matching the timing-window, not the API surface. 'API differs' is not an escape to N/A."
tags: [task.optimization, topic.performance.analysis]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/perf/ol-163-cross-arch-perf-timing-methodology.md
    role: primary
created_at: "2026-07-12T16:00:00Z"
updated_at: "2026-07-12T16:00:00Z"
platforms: ["950"]
aliases: ["OL-163"]
---

# Cross-arch perf timing: device-event symmetric first, wrapper-inclusive fallback, N/A last resort

> 原 OKF v1 卡号：**OL-163**。
> 现象分类：perf_regression；观测置信度：single_run（单次观测）。
> 触发信号：
> - measuring a perf ratio across two backends with different APIs (A3 aclnn entry vs A5 ACLRT_LAUNCH_KERNEL macro) and tempted to declare performance NOT_VERIFIED because the APIs differ

## 现象 / 触发

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=all, mode=port_a3_to_a5`. Unverified on
Ascend910_V220 (patterns expected to transfer since `aclrtEvent` is shared CANN API, but not yet
verified end-to-end on A3).

Measuring a perf ratio across two hardware backends that ship different APIs (A3 ships an aclnn
entry; A5 ships only the `ACLRT_LAUNCH_KERNEL` macro path). The failure mode is escaping to N/A —
declaring performance unverifiable — just because the API surfaces differ.

## 适用场景

- 适用技术与问题域：AscendC 性能/测量问题（现象分类：perf_regression）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## 根因 / 教训

**Timing symmetry is NOT API symmetry.** What must be symmetric is the *timing-window definition* on
both sides, not the API surface being timed. Three options, in strict preference order:

1. **Device-event symmetric (PREFERRED)** — wrap each side's natural call shape with the platform's
   device-event API (`aclrtEvent` on A3, `torch.npu.Event` on A5). Events are recorded on the same
   stream as the kernels; `ElapsedTime` captures on-device kernel time only, EXCLUDING all host
   overhead (Python+pybind+alloc+aclnn host prep) by construction. Works regardless of how different
   the APIs look.
2. **Wrapper-inclusive symmetric (ACCEPTABLE FALLBACK)** — wrap each side's natural user-visible call
   with Python `perf_counter` + `torch.npu.synchronize` end-to-end. The wrappers need not be
   byte-equivalent — only the user-visible entry point on each side. The ratio captures wall-clock
   user-visible perf even when host-overhead composition differs. Must declare
   `option2_wrapper_composition` explicitly.
3. **Escape (LAST RESORT, gate-defended)** — declare `performance.status="NOT_VERIFIED_SAME_METHOD"`,
   ratio absent. Reserved for cases where BOTH Option 1 AND Option 2 are demonstrably infeasible WITH
   CODE/CANN-DOC EVIDENCE PER OPTION. "API differs" is NOT evidence of infeasibility — it is the
   default state of port_a3_to_a5 mode.

**Concrete anchor — Option 1 (C++ A3 side)**:
```cpp
aclrtEvent evt_start, evt_end;
aclrtCreateEventWithFlag(&evt_start, ACL_EVENT_TIME_LINE);
aclrtCreateEventWithFlag(&evt_end, ACL_EVENT_TIME_LINE);

// Build executor + alloc workspace BEFORE the timed window — host-side prep
// would re-introduce asymmetric host time.
aclnnFatreluMulGetWorkspaceSize(/*...*/, &ws_size, &executor);
aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_HUGE_FIRST);

// === timed window ===
aclrtRecordEvent(evt_start, stream);
aclnnFatreluMul(workspace, ws_size, executor, stream);
aclrtRecordEvent(evt_end, stream);
aclrtSynchronizeStream(stream);
// then read ElapsedTime(evt_start, evt_end)
```
