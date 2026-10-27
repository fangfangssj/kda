---
type: Runbook
title: "Key the op-gen dedup gate on reference semantics, not the op slug — a pre-spawn sibling cross-check turns a greenfield re-derive into a verbatim own-archive adapt"
description: "Key the op-gen dedup gate on reference semantics, not the op slug — a pre-spawn sibling cross-check turns a greenfield re-derive into a verbatim own-archive adapt"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-229"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-229 — OL-229: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all (op-gen workflow / duplicate-spend avoidance)。
- 验证来源：workflow lesson (NPU-independent — reference-semantics dedup)。

**OL-229** — **A source-mode op-gen CLI can re-emit a duplicate of an already-archived op — key the dedup gate on reference *semantics*, not the op slug, and a pre-spawn sibling cross-check converts a greenfield re-derive into a verbatim own-archive adapt** — different CLI source-modes/slugs can map onto the same math (same grads, same forward, byte-identical reference JSON). Slug-keyed or archive-time-only dedup misses it → wasted spawn. Detect pre-spawn by comparing reference-JSON hash / forward+wrt signature across existing archives; if a passing sibling is our OWN archive, adapt it verbatim (asset-reuse, distinct from forbidden CANN-source copy) → clean first build + identical verdict. Anchor: selective_scan_bwd (port_cuda backward) ≡ archived selective_scan_full_grad, adapt → first-build 28/30 PASS_WITHIN_TOLERANCE 2026-06-18. Cross-ref [OL-27](../performance/ol_27_performance_claims_must_be_based_on_same_condition.md), [OL-103](../precision/ol_103_npu_transcendentals_fp16_precision.md)/109.

- Former A5Ops ID: `OL-229`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (op-gen workflow / duplicate-spend avoidance)`
`verified_on: workflow lesson (NPU-independent — reference-semantics dedup)`

### Principle

Different CLI source-mode entry points can map onto the **same mathematical op** under different slugs. A backward-mode port from a CUDA source and a previously-archived gradient op can be byte-identical (same gradients, same forward, byte-identical reference JSON) while carrying unrelated names. When the dedup gate keys only on the **op slug** or fires only at archive time (output-path collision), the spawn is already committed before the duplication is visible → wasted compute / tokens / device verification on an identical deliverable. The dedup key must be **reference-semantics** (reference-JSON hash, or forward formula + wrt-order signature), not the slug.

### Decision rule

- **Detect pre-spawn**: before committing a spawn, scan existing archives for an op whose reference computes the same math (compare `backward_ref.json` / reference-manifest hash, or forward + grad-wrt signature). The self-critic sibling cross-check (C19-class) is the place for this — surface a passing sibling *before* the worker is dispatched, not after archive.
- **If a passing sibling exists and is our OWN archive**: ADAPT it verbatim (asset-reuse — inherit the proven kernel structure + helpers) rather than greenfield re-deriving identical math. Expect a clean first-attempt build and an identical verdict at near-zero iteration cost.
- **Stay on the allowed side of the copy line**: reusing our own archived kernel is permitted asset-reuse (distinct from copying CANN source, which stays forbidden). Disclose provenance in `reference_manifest.jsonl` (sibling = peer_op, understand+adapt).

### Concrete anchor

`selective_scan_bwd` (spawned via the port_cuda backward CLI) is mathematically identical to the already-archived `selective_scan_full_grad`: byte-identical `backward_ref.json`, same 8 grads, same explicit-gy adjoint, same forward. The pre-spawn cross-check pointed at the passing sibling; adapting its proven OL-103 `SoftExp`/`SoftRecip`/`SoftLog` + scan structure yielded a first-attempt clean build and an identical `PASS_WITHIN_TOLERANCE` 28/30 (fp32 10/10, bf16 10/10, fp16 8/10). The spawn was nonetheless committed (slug-keyed gate did not catch the duplication earlier) — the wasted-spawn signal is the lesson.

### Evidence

- selective_scan_bwd (2026-06-18, A5 Ascend950PR): port_cuda backward CLI re-emitted the already-archived `selective_scan_full_grad` deliverable. C19 sibling cross-check caught it post-spawn; verbatim adapt → clean first build + identical 28/30 PASS_WITHIN_TOLERANCE. Flagged C17 duplicate-spend in `analysis.md`; recommendation: move the duplicate detection ahead of the spawn decision and key it on reference semantics.

### Other instances (predicted)

Any time two CLI invocations name the same math under different slugs — `port_cuda` vs `port_pytorch` of the same op; a `*_full_grad` vs `*_bwd` naming split; a benchmark op that duplicates an already-shipped fused op. Whenever the spawn cost is non-trivial, the dedup gate should compare reference-JSON / formula signatures across the existing archive set, not just compare slugs.

### Cross-references

- OL-27 (re-measure perf on byte-identical kernels — the adapted sibling's perf must be independently re-measured, not inherited from the sibling's archive).
- OL-103 / OL-109 (the per-dtype precision tiers the adapted kernel inherits unchanged).

---
