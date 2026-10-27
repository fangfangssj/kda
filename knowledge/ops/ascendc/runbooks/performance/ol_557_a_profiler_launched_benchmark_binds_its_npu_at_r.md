---
type: Runbook
title: "a profiler-launched benchmark binds its NPU at runtime init, BEFORE any wrapper-level env var applies — export"
description: "msprof 启动的 benchmark 在进程启动时就绑定 NPU，wrapper 里后设 ASCEND_RT_VISIBLE_DEVICES 无效——要在 profiler 自己的启动环境里 export（曾因此浪费 2.5h 测垃圾数据）"
tags: [task.optimization, topic.compilation, topic.performance.analysis]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-557"]
---

# a profiler-launched benchmark binds its NPU at runtime init, BEFORE any wrapper-level env var applies — export

**OL-557** — **a profiler-launched benchmark binds its NPU at runtime init, BEFORE any wrapper-level env var applies — export `ASCEND_RT_VISIBLE_DEVICES` in the profiler's OWN launch env.** msprof's injected profiler attaches to physical device 0 if the profiled wrapper sets the var only post-init (from `--device`); on a shared box every case then fails `rtSetDevice 507033` while the process looks alive — liveness ≠ output-validity (W5 checked `ps`, not per-case success). Fix landed: `benchmark_acceptance.py` perf_env carries the var like the precision path. reduce_sum 310p 2026-08-17 (2.5 h of garbage perf measurement, 507033 in every app_output.log). backend=all.

<!-- applies_to_backend: all -->

`applies_to: soc=all (msprof / profiler-injected measurement harnesses); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend310P3; cann=9.1.0 (reduce_sum 310p benchmark-acceptance lane, 2026-08-17)`

**Principle**: msprof's injected profiler initialises the runtime when the profiled
process STARTS. A wrapper script that sets `ASCEND_RT_VISIBLE_DEVICES` from a `--device`
argument does so AFTER that init — the profiler daemon has already attached to PHYSICAL
device 0. On a shared box, if device 0 has another tenant, every case then fails
`rtSetDevice 507033 (device retain)` — and a harness that only checks process liveness
(`ps` shows the process; critic W5 verified the orphan was ALIVE, not that its cases
were SUCCEEDING) will march through the whole case set fast-failing (~90 s/case) for
hours, producing a full set of garbage measurements that look like a completed run.
Two rules: (a) export `ASCEND_RT_VISIBLE_DEVICES` in the msprof process's OWN
environment, never rely on the profiled wrapper to set it post-init; (b) liveness is
not output-validity — a long-running measurement job must be checked on per-case
success signals (app_output.log verdicts), not on the process being alive.

**Concrete anchor**: `benchmark_acceptance.py:595`（a5_ops 路径，未迁移） — the
landed fix: `perf_env["ASCEND_RT_VISIBLE_DEVICES"] = str(lane)`, mirroring what the
precision path already did; covers the finalize-gate re-run too.

**Evidence**: reduce_sum 310p (2026-08-17, first instance): acceptance perf path
silently measured garbage for 2.5 h — every case's app_output.log showed `open device
0 failed 507033` while the parent process looked alive; discovered only on output
audit, not by any liveness check.

**Evidence (2nd instance — and a misdiagnosis trap)**: sort 310p (2026-08-18,
benchmark-acceptance lane, cannbot-skills msprof path, same shared host): kw-1's
all-fail perf loop (`rtSetDevice device retain error 507033`) was first logged in the
failures ledger as "transient device state" after a later probe passed — the flap-with-
device-0-occupancy symptom makes the transient reading plausible. kw-3 root-caused it
to this rule: `_run_msprof_quick`/`_run_msprof_standard` launched msprof WITHOUT
`ASCEND_RT_VISIBLE_DEVICES` (the wrapper app set it only in-process); msprof's
collector then probed PHYSICAL device 0, held by another tenant. Fixed at the harness
layer (cannbot-skills commit 3d94e270): export the var into the msprof subprocess env
in both launch paths. Diagnostic signature: app log shows `open device 0 failed,
runtime result = 507033` while npu-smi shows the TARGET device free — check who holds
device 0, not the target device. Post-fix: 10/10 cases measured both sides (geomean
0.814×). Lesson-in-lesson: "transient" on a shared box is a hypothesis to refute with
the device-0 check, not a verdict.

**Other instances (predicted)**: any msprof- or torch_npu.profiler-wrapped benchmark
on shared NPUs, any lane-allocated parallel run whose perf command differs from its
precision command in env setup, and any watchdog that supervises a measurement job by
process state alone.

**Cross-ref**: CLAUDE.md "Shared NPU — 性能测试前检查" (the shared-box pre-check this
env bug defeats), P9 in shared/ANTI_PRESSURE_PROTOCOLS.md (507033 is an infra signal —
here it was swallowed by a fast-failing loop instead of escalated). backend=all.
