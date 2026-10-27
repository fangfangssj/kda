---
type: Runbook
title: "An AICPU-only upstream source (no op_kernel/, no arch35/) is a VALID port_a3 input"
description: "上游只有 AICPU 实现（无 op_kernel）的算子也是合法的移植输入——AICPU 的 .cpp 就是算法规格书，照着生成 dav-native kernel 而不是找源码逐行搬"
tags: [task.optimization, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-555"]
---

# An AICPU-only upstream source (no op_kernel/, no arch35/) is a VALID port_a3 input

## 优化点

**OL-555** — **An AICPU-only upstream source (no `op_kernel/`, no arch35/) is a VALID port_a3 input** — the AICPU `op_kernel_aicpu/<op>_aicpu.cpp` is the ALGORITHM SPEC from which the dav-native AICore kernel is GENERATED (a P140 `ACLRT_LAUNCH_KERNEL` deliverable), not line-ported (no dav source exists to port); `op_host/{infershape,op_api}` + `op_kernel_aicpu/` are CARRY (arch-invariant GE-registration path). Complements (not contradicts) the author-from-A3-source / default-OFF-prestage rule — when the A3 source is AICPU-only, the AICPU `.cpp` IS that source. topk 310p 2026-08-14 (25/25 bit-exact vs CPU torch.topk values-only + 40/40 stress). backend=ascendc.

`applies_to: soc=all (source-shape / port_a3-workflow rule, not a silicon behavior — holds for ANY target where port_a3 runs); cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5 (AICPU-only upstream sources)`
`verified_on: soc=Ascend310P3 (dav-m200); cann=9.0.0 — topk, single op`
`unverified_on: the rule is expected to transfer to a3/a5 (the source shape is upstream-CANN-determined, not target-determined), but only dav-m200 evidence exists`

- **Category**: port_a3 source handling / harness workflow
- **Loaded by**: Generator (port_a3 mode), aog-kernel-worker, QA
- **Trigger**: an upstream op whose tree carries `op_kernel_aicpu/` + `op_host/` + `op_api/` but NO `op_kernel/` dav source (no `op_kernel/<op>.h` top-level dav algorithm, no arch35/).
- **Principle**: an AICPU-only upstream shape is a **valid** port_a3 input, not a "missing source" failure. The AICPU `op_kernel_aicpu/<op>_aicpu.cpp` is the **algorithm spec** (selection semantics, tie-break, dtype contract) from which the dav-native AICore kernel is **generated** (a P140 `ACLRT_LAUNCH_KERNEL` deliverable) rather than line-ported — there is no dav source to line-port. `op_host/{infershape,op_api}` + `op_kernel_aicpu/` are **CARRY** (the arch-invariant GE-registration path), not the compute to port. This complements (does not contradict) the "author from A3 algorithm source, not arch35/ wrapper" rule — when the A3 algorithm source is AICPU-only, the AICPU `.cpp` IS that algorithm source. The kernel is still authored by the worker from that spec; it is NOT a wrap of anything.
- **Concrete anchor**: `topk/` — upstream source `ops-math/math/topk` has `op_kernel_aicpu/` + `op_host/` + `op_api/`, no `op_kernel/`; the dav kernel was generated from `topk_aicpu.cpp` selection semantics (values-only output makes tie-break moot), with `op_host/{infershape,op_api}` + `op_kernel_aicpu/` carried.
- **Evidence**: topk 310p (2026-08-14, Ascend310P3/dav-m200, CANN 9.0.0, benchmark-acceptance lane, DEVICE-RUN): AICPU-only source accepted as port_a3 input; kernel generated from the AICPU spec → 25/25 bit-exact vs CPU `torch.topk` (values-only) + 40/40 stress; `op_host/` + `op_kernel_aicpu/` carried as the GE-registration path. First instance.
- **Other instances (predicted)**: any AICPU-only upstream op in port_a3 mode (many fallback-class / registered-only ops ship AICPU kernels with no dav counterpart).
- **Cross-ref**: CLAUDE.md "Default OFF on upstream-A5 prestage" (the author-from-A3-source rule this extends to the AICPU-only case), [`OL-160`](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md) (canonical entry-point names the AICPU-only deliverable must still emit), the "no CANN/CPU delegation" rule (the generated kernel is real AscendC, not a torch/aclnn wrap). backend=ascendc.
