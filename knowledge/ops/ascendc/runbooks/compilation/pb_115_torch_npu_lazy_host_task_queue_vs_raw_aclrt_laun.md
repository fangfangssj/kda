---
type: Runbook
title: "torch_npu lazy host task queue vs raw ACLRT_LAUNCH_KERNEL: a still-queued vendor op can OVERWRITE the raw-laun"
description: "torch_npu 惰性 host 任务队列里未执行完的厂商 op 会覆盖 raw launch kernel 的输出——launch 成功、同步正常但结果被改"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-115"]
---

# torch_npu lazy host task queue vs raw ACLRT_LAUNCH_KERNEL: a still-queued vendor op can OVERWRITE the raw-laun

**PB-115** — **torch_npu lazy host task queue vs raw `ACLRT_LAUNCH_KERNEL`: a still-queued vendor op can OVERWRITE the raw-launched kernel's output** (launch rc==0, sync rc==0, silent; y bitwise-equal to a vendor op's output or a fill sentinel). Stream-level sync does NOT drain the host queue. Fix: `c10_npu::npuSynchronizeDevice()` pre-launch + launch-rc TORCH_CHECK in the pybind shim. Suspect first for intermittent 'output equals the other side's data' in harnesses that interleave vendor + raw-launch ops in one process. gelu 310p 2026-08-12 (canary/vendor-first-ordering bisect → deterministic 40/40). Cross-ref P140, [`P-P203`](../performance/p_p203_predicate_mask_cast_chain_on_dav_m200.md). [dav-m200/Ascend310P3, pybind-raw-launch, runtime]

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0; bisheng=n/a; op_class=all (any pybind raw-launch shim)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (gelu 310p, 2026-08-12 — sentinel/canary bisect)`
`unverified_on: soc=Ascend910, Ascend950PR (queue semantics may differ; the drain fix is harmless everywhere)`

- **Severity**: HIGH — silent wrong-result class: our kernel's output buffer is overwritten AFTER our kernel completes, with every return code green.
- **Symptom signature**: output `y` is bitwise-equal to a VENDOR op's output from an earlier call, or to a fill sentinel (e.g. 7.0) the test wrote into a recycled block — i.e. `y` contains whatever the previous owner of the recycled memory wrote, because our kernel's result was clobbered (or never landed). Correlates with vendor-runs-first ordering.
- **Mechanism**: torch_npu's host-side lazy task queue means a raw `ACLRT_LAUNCH_KERNEL` lands on the rt stream BEFORE still-queued torch tasks. If our output buffer recycles a just-freed block that a queued vendor op will write, that vendor op executes AFTER our kernel and OVERWRITES our output. `aclrtSynchronizeStream` / stream-level syncs do NOT drain the host-side queue.
- **Fix**: in the pybind shim, call `c10_npu::npuSynchronizeDevice()` (device-wide, drains the host queue — NOT just stream_synchronize) immediately before the raw launch, plus a TORCH_CHECK on the launch return code.
- **Evidence**: gelu 310p (2026-08-12) — acceptance-verifier fp16 cases intermittently returned vendor-equal/sentinel data; canary + vendor-first-ordering probes isolated the overwrite; the drain fix made all regimes pass deterministically (evaluate_ascendc.sh 40/40). Shim anchor `代码片段 gelu/pybind11.core.snippet.cpp（见文末）` (workspace scratch, gitignored). Cross-ref P140 (the pybind + ACLRT_LAUNCH_KERNEL verify pattern this fix lives in), [`P-P203`](../performance/p_p203_predicate_mask_cast_chain_on_dav_m200.md).
- **Other instances (predicted)**: every raw-launch pybind shim that runs interleaved with vendor torch_npu ops in the same process (acceptance verifiers do exactly this — ref model then candidate model in one process). Any intermittent "output equals the OTHER side's data" bug in such harnesses should suspect this first.

## 内嵌代码片段

### gelu/pybind11.core.snippet.cpp

```cpp
}

PYBIND11_MODULE(_gelu_ext, m)
{
    m.def("run_gelu", &run_gelu,
          pybind11::arg("x"), pybind11::arg("approximate") = "none",
          "gelu (approximate='none'|'tanh' — BOTH dispatch to the software tanh chain per owner 2026-08-12: 310P vendor ignores the flag; erf-exact chain retained in kernel but not dispatched) — Ascend310P3 AscendC kernel");
}
```

**孪生卡 pb-168 合并的独有内容**（pb-168-torch-npu-lazy-host-side-task-queue-vs-raw-aclrt.md 已删除）：

> - **Evidence**: gelu 310p (2026-08-12) — acceptance-verifier fp16 cases intermittently returned vendor-equal/sentinel data; canary + vendor-first-ordering probes isolated the overwrite; the drain fix made all regimes pass deterministically (evaluate_ascendc.sh 40/40). Shim anchor `代码片段 gelu/pybind11.core.snippet.cpp（见文末）` (workspace scratch, gitignored). Cross-ref P140 (the pybind + ACLRT_LAUNCH_KERNEL verify pattern this fix lives in), P-P125.
