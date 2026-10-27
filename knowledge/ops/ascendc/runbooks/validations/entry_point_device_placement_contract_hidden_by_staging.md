---
type: Runbook
title: "A canonical entry point with two benchmark-harness consumers carries a device-placement contract, and the consumer that stages inputs hides the violation the retained runner hits"
description: "A canonical entry point with two benchmark-harness consumers carries a device-placement contract, and the consumer that stages inputs hides the violation the retained runner hits"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-500"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-500 — OL-500: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=SpatialGroupEnhance; scope=benchmark-verification-harness。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0.B060。

**OL-500** — **A canonical entry point with two benchmark-harness consumers carries a device-placement contract, and the consumer that stages inputs for you HIDES the violation the retained runner hits.** On the retained SpatialGroupEnhance A5/B060 run, the inline verifier moved inputs to the device while Pass A reloaded with `map_location="cpu"`; a `forward` that assumed NPU input handed HOST memory to the kernel launch. Signature: `worst_abs_diff` ~`1e+37`, 0 bit-exact cases, and an NPU-tensor-saving warning. Fix has TWO clauses: device-stage inside forward without adding compute/fallback, then return on the caller's original device. Anchor: 50/50 inline vs **0/50 retained, `worst_abs_diff=3.21e+37`** on the same build. Sibling ops, other modes/backends, SoCs, and CANN releases remain unverified.

- Former A5Ops ID: `OL-500`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=SpatialGroupEnhance; scope=benchmark-verification-harness`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060`
`unverified_on: every sibling op, backend, SoC, CANN release, and non-benchmark op-gen mode (only the retained SpatialGroupEnhance A5/B060 benchmark run is empirical evidence)`

**Principle**: the canonical `ModelNew.forward` (OL-160) is called by more than one consumer, and
they do not agree on who moves the tensors. An inline benchmark verifier typically stages inputs
onto the device before calling. A **retained** Pass-A runner reloads a captured dataset with
`map_location="cpu"` and calls `ModelNew(*case["inputs"])` directly, so the same `forward` receives
CPU tensors. A wrapper that assumes "my input is already on the NPU" and reaches straight for
`data_ptr()` therefore hands **host memory to a kernel launch** — and passes every inline case while
failing every retained-artifact case. Neither consumer is wrong; the entry point simply has a
device-placement contract that nothing states, so it gets decided twice.

**Why it stays invisible until finalize**: the failing consumer runs LATE (at capture/finalize) and
the passing consumer is the one an author naturally re-runs. "50/50 inline, 0/50 retained" is not a
contradiction to be explained away — it is two different callers, and the retained one is the one
that ships.

**Signature**: `worst_abs_diff` in the region of `1e+37` — the uninitialised-memory marker, NOT a
precision error, so do not route it to a precision probe; zero bit-exact cases rather than a
degrading tail; and often a torch warning about saving NPU tensors, because a forward that assumed
NPU input also *returns* NPU output to a CPU-fed caller.

**Rule (two clauses, both required)**:
1. Stage defensively inside `forward`: `x if x.device.type == "npu" else x.npu().contiguous()`.
   This is device staging ONLY. It must never grow into a compute path — a `.cpu()` fallback is a
   CLAUDE.md-forbidden CPU fallback, and a `torch.*` compute call here is delegation.
2. Return on the CALLER's original device, so a CPU-fed consumer gets a CPU tensor back and an
   NPU-fed consumer is unaffected. Clause 1 without clause 2 fixes the crash and leaves the warning
   plus a device-type surprise for the caller.

**Concrete anchor** (small piece):
```python
@staticmethod
def _to_npu(x):                       # device staging ONLY — never compute
    return x if x.device.type == "npu" else x.npu().contiguous()

def forward(self, x, ...):
    orig_device = x.device
    out = self._launch(self._to_npu(x), ...)
    return out if out.device == orig_device else out.to(orig_device)
```

**Catch it BEFORE finalize**: run the retained-Pass-A path locally against the freshly built `.so`
instead of discovering it at capture time. On the originating op a workspace-local dry-run
reproduced the 0/50 in seconds; the same defect otherwise surfaces only after the entire precision
and performance loop has already been declared green.

**Evidence**: SpatialGroupEnhance (2026-08-21, Ascend950PR_9579 / CANN 9.1.0.B060, kw iter 2) — all
50 inline benchmark cases passed while the retained Pass-A dry-run scored **0/50 bit-exact with
`worst_abs_diff = 3.21e+37`**, plus the torch NPU-tensor-saving warning. Root cause was exactly the
above. After adding staging + device mirroring: 50/50 bit-exact on both consumers, MERE = MARE = 0.

**Other instances (predicted, NOT admitted by this entry)**: sibling benchmark ops whose retained
runner reloads a captured dataset with an explicit `map_location`; edge-case and determinism
re-run consumers may have the same shape. Non-benchmark op-gen modes and other backends remain
unverified even when they reuse the canonical entry-point name; a shared name is not evidence that
their staging or return-device contract is identical.

**Cross-reference**: OL-160 (canonical entry-point names — this entry is the *contract* those names
carry, and the reason a name-only convention is not sufficient); OL-222 (the other retained-runner
trap: NPU tensor accumulation across serial cases).
