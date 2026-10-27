---
type: Runbook
title: "The CPU-truth reference must replicate the kernel's out-of-range index policy — scatter/index kernels that skip OOB indices require the reference to mask them, never clamp"
description: "The CPU-truth reference must replicate the kernel's out-of-range index policy — scatter/index kernels that skip OOB indices require the reference to mask them, never clamp"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["310p"]
aliases: ["CPU-truth", "OOB"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-408 — OL-408: A5Ops additive extension

> 关联：同一 OL-408 主题已由既有卡 [ol_408_the_cpu_truth_reference_must_replicate_the_kerne.md](ol_408_the_cpu_truth_reference_must_replicate_the_kerne.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=scatter/index (scatter_elements, gather_elements, index_put, index_select, index_fill)。
- 验证来源：soc=Ascend310P3; cann=9.0.0 (scatter_elements_v2 310p 2026-08-08)。

**OL-408** — **the CPU-truth reference must replicate the kernel's out-of-range index policy: scatter/index kernels that SKIP OOB indices require the reference to MASK them, never clamp.** `.clamp(0,N-1)` remaps OOB indices onto a valid position the kernel never writes → precision mismatch on large-index cases (~1e17 from int32/int64 edge data). Fix = validity mask. Rule = *match the kernel* (skip-or-clamp, either is valid if both sides agree). Anchor `workspace/scatter_elements_v2/verification.json`. Evidence: scatter_elements_v2 310p iter-2, 9/9 T1 after fix. backend=ascendc.

- Former A5Ops ID: `OL-408`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend310P3; cann=9.0.0; bisheng=n/a; op_class=scatter/index (scatter_elements, gather_elements, index_put, index_select, index_fill)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend310P3; cann=9.0.0 (scatter_elements_v2 310p 2026-08-08)`
`unverified_on: other SoCs — the principle (reference must match kernel OOB policy) is arch-agnostic; the specific skip-vs-clamp policy is op/source-specific and must be read from each kernel.`

### Principle

When a scatter/index kernel's contract is to **skip** out-of-range indices (no write — the common case when the upstream/CANN source guards `if (idx >= 0 && idx < N)` and does nothing otherwise), the CPU-truth reference (`model.py`) MUST implement the identical policy. Writing the reference with `.clamp(0, N-1)` instead **remaps** every OOB index onto a valid position (typically the last element), so on any test case whose indices include large OOB values the reference performs a write the kernel never performs → a precision mismatch concentrated on that remapped position.

The pitfall is sharp because OOB indices arise naturally in edge datasets: random `int32`/`int64` index generation, integer overflow, or deliberately-stressful cases can yield indices ~1e17–1e18 that are trivially out of range. A clamped reference hides the divergence on small-index cases and exposes it only on the large-index case — easy to misread as "one flaky case" rather than a systematic policy mismatch.

### Concrete anchor
