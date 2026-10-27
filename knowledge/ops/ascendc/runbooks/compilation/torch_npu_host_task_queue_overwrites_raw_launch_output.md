---
type: Runbook
title: "torch_npu lazy host task queue vs raw `ACLRT_LAUNCH_KERNEL` — a still-queued vendor op can overwrite the raw kernel's output"
description: "torch_npu lazy host task queue vs raw `ACLRT_LAUNCH_KERNEL` — a still-queued vendor op can overwrite the raw kernel's output"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["aclrtSynchronizeStream"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-115 — PB-115: A5Ops additive extension

> 关联：同一 PB-115 主题已由既有卡 [pb_115_torch_npu_lazy_host_task_queue_vs_raw_aclrt_laun.md](pb_115_torch_npu_lazy_host_task_queue_vs_raw_aclrt_laun.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3（dav-m200）；CANN 9.0.0；op_class=all（任何 pybind raw-launch shim）。
- 验证来源：soc=Ascend310P3；CANN 9.0.0（gelu 310p，2026-08-12 — sentinel/canary 二分）。

**PB-115** — **torch_npu lazy host task queue vs raw `ACLRT_LAUNCH_KERNEL`: a still-queued vendor op can OVERWRITE the raw-launched kernel's output** (launch rc==0, sync rc==0, silent; y bitwise-equal to a vendor op's output or a fill sentinel). Stream-level sync does NOT drain the host queue. Fix: `c10_npu::npuSynchronizeDevice()` pre-launch + launch-rc TORCH_CHECK in the pybind shim. Suspect first for intermittent 'output equals the other side's data' in harnesses that interleave vendor + raw-launch ops in one process. gelu 310p 2026-08-12 (canary/vendor-first-ordering bisect → deterministic 40/40). Cross-ref P140, P-P203. [dav-m200/Ascend310P3, pybind-raw-launch, runtime]

- Former A5Ops ID: `PB-115`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any pybind raw-launch shim)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — sentinel/canary bisect)`
`unverified_on: soc=Ascend910, Ascend950PR (queue semantics may differ; the drain fix is harmless everywhere)`

- **Severity**: HIGH — silent wrong-result class: our kernel's output buffer is overwritten AFTER our kernel completes, with every return code green.
- **Symptom signature**: output `y` is bitwise-equal to a VENDOR op's output from an earlier call, or to a fill sentinel (e.g. 7.0) the test wrote into a recycled block — i.e. `y` contains whatever the previous owner of the recycled memory wrote, because our kernel's result was clobbered (or never landed). Correlates with vendor-runs-first ordering.
- **Mechanism**: torch_npu's host-side lazy task queue means a raw `ACLRT_LAUNCH_KERNEL` lands on the rt stream BEFORE still-queued torch tasks. If our output buffer recycles a just-freed block that a queued vendor op will write, that vendor op executes AFTER our kernel and OVERWRITES our output. `aclrtSynchronizeStream` / stream-level syncs do NOT drain the host-side queue.
- **Fix**: in the pybind shim, call `c10_npu::npuSynchronizeDevice()` (device-wide, drains the host queue — NOT just stream_synchronize) immediately before the raw launch, plus a TORCH_CHECK on the launch return code.
- **Evidence**: gelu 310p (2026-08-12) — acceptance-verifier fp16 cases intermittently returned vendor-equal/sentinel data; canary + vendor-first-ordering probes isolated the overwrite; the drain fix made all regimes pass deterministically (evaluate_ascendc.sh 40/40). Shim anchor `workspace/gelu/kernel/pybind11.cpp` (workspace scratch, gitignored). Cross-ref P140 (the pybind + ACLRT_LAUNCH_KERNEL verify pattern this fix lives in), P-P203.
- **Other instances (predicted)**: every raw-launch pybind shim that runs interleaved with vendor torch_npu ops in the same process (acceptance verifiers do exactly this — ref model then candidate model in one process). Any intermittent "output equals the OTHER side's data" bug in such harnesses should suspect this first.
