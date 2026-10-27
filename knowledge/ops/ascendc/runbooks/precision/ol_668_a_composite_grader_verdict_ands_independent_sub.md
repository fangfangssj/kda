---
type: Runbook
title: "A composite grader verdict ANDs independent sub-checks — when the statistical metric passes but the verdict fails, the per-element bound is the binding constraint; decompose the verdict before iterating"
description: "复合评分器 MERE 达标但 verdict=fail 时，约束在逐元素 allclose 的少数放大元素上——先分解 verdict 子项再迭代 kernel"
tags: [task.debugging, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-668"]
---

# A composite grader verdict ANDs independent sub-checks — when the statistical metric passes but the verdict fails, the per-element bound is the binding constraint; decompose the verdict before iterating

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (any acceptance/verifier emitting a statistical metric plus a boolean verdict); workflow=verification`
`verified_on: soc=Ascend310P3; cann=9.1.0 (instance_norm_grad 310p, 2026-09-02 — owner precision leg, 41 cases)`

**Principle**: composite verifiers report a statistical metric (MERE / matched-ratio class) AND a boolean verdict across several sub-checks. Reading only the metric misdiagnoses the failure class: `MERE < rel_thr` with `passed=False` means the binding constraint is the per-element allclose (atol/rtol) on a small number of amplified elements — matched_ratio can be ≥ 0.96 on every failing case while the verdict still fails. Before iterating the kernel, decompose the verdict: WHICH sub-check failed, on how many elements, at what magnitude.

**Concrete anchor**: instance_norm_grad owner leg — every failing fp32 case had MERE 2e-6..7e-5 (below threshold) and matched_ratio ≥ 0.962, yet `passed=False`: the per-element allclose (atol=1e-3, rtol=2^-13) on `pow(std,-3)`-amplified elements was the sole failing sub-check.

**Evidence**: instance_norm_grad 310p (2026-09-02): `instance_norm_grad/knowledge_update.md` operational learnings; authoritative owner-leg output recorded with the op's Phase D artifacts.

**Other instances (predicted)**: any two-tier or composite grader (compare.py small-value/cancellation carve-outs, owner benchmark allclose+MERE AND) — the "metric green, verdict red" signature appears whenever a per-element tail is amplified beyond the elementwise bound while the bulk statistic stays clean. backend=all.
