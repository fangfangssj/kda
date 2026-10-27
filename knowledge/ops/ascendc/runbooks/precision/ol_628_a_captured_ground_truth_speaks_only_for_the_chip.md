---
type: Runbook
title: "A captured ground truth speaks only for the chip it was captured on — audit the capture AND its device provenance before encoding semantics from it or from any advisory"
description: "捕获的 ground truth 只对捕获它的那颗芯片有效——移植时的语义声明要区分来源（记忆/他卡捕获/本卡实测），跨芯片引用要重新验证"
tags: [task.debugging, topic.compute.elementwise.type_conversion, topic.compilation, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-628"]
---

# A captured ground truth speaks only for the chip it was captured on — audit the capture AND its device provenance before encoding semantics from it or from any advisory

`applies_to: soc=all (workflow discipline — validated on the Ascend310P3 / Ascend910B2C pair); cann=9.0.0; bisheng=n/a; op_class=all (port/verify paths that encode semantics from a captured reference or a stored advisory)`
`verified_on: soc=Ascend310P3 (target box) + Ascend910B2C (capture host); cann=9.0.0`
**Principle**: a semantic claim entering a port ("op X maps NaN → Y") can come from three
sources, each fallible in a DIFFERENT direction — (a) memory / prior-session advisories /
self-critic repeats of them, which can carry **swapped case-family attributions**; (b) a live
probe, which describes the box it runs on; (c) a captured reference dataset, which describes
the chip the CAPTURE ran on — not necessarily the target box. Before encoding any semantics
into a kernel: (1) **audit the capture data itself** — per-case counts, not the advisory's
summary of them; (2) **check the capture's provenance metadata** (e.g.
`a3_baseline_perf.json.device` / the capture manifest) — a capture from chip X is silent
about chip Y's native vendor behavior; (3) if a live-probe claim exists, **re-run the probe**
(typically < 1 minute) before generalizing in EITHER direction. The failure mode is symmetric:
generalizing box behavior onto the capture, or capture data onto the box.
**Concrete anchor** (is_inf 310p, 2026-08-09): capture audit —
`edge_dataset.pt dist_sparse_nan_seed{0,1,2}` (102 NaNs each) → n_true=0;
`dist_sparse_inf_seed{0,1,2}` → n_true=102 (exact-inf semantics); capture metadata
`a3_baseline_perf.json: "device": "Ascend910B2C"` (A3 server, NOT the 310p box); live probe
on the 310p box, 6/6 reproducible — `torch.isinf([1, inf, -inf, nan, -3.5]).npu()` →
`[F, T, T, T, F]` (NaN → True, diverging from CPU AND from the A3 capture).
**Evidence**: is_inf 310p (2026-08-09, port_a3_to_a5 TARGET=310p) — THREE compounding
attribution errors around one semantic: (i) an older memory claimed the capture's NaN cases
have n_true=102 (they have 0 — wrong about the capture); (ii) the A3-capture author
live-probed the 310p box and assumed the capture matched (wrong about the capture's
provenance); (iii) kw-1 audited the A3 capture, then generalized it to declare the 310p BOX
exact-inf and requested a memory rewrite to that effect (wrong about the box — the post-worker
self-critic audit refuted it with the 30-second live probe above). Resolution encoded in
memory `project-310p-isinf-nan-semantic-split`: 310p-native `torch.isinf(nan)=True`, A3
capture is exact-inf, ported kernel is A3-faithful per the port contract and the divergence
from 310p-native is disclosed, not "fixed". Following the wrong advisory (either direction)
would have shipped a kernel failing 3×102 bit errors or an unrequested semantic flip.
**Other instances (predicted)**: every TARGET=310p port whose reference capture is produced
on the A3 server (the whole 310p port line runs this capture-host ≠ target-box asymmetry);
any semantic claim sourced from memory / self-critic advisory rather than fresh data; any
cross-host capture reused as a truth source.
**Cross-ref**: OL-321 (a result's scope must cover the claim — same family), [`OL-27`](../performance/ol_27_performance_claims_must_be_based_on_same_condition.md) / P141
(same-condition A/B — the perf-side analog), CAND-UPSTREAM-UT-GOLDEN-NOT-TRUTH (a golden is
an opinion to cross-check, not truth), CAND-DAV-M200-PERF-BASELINE-SAME-HOST-VENDOR (the
perf-side instance of capture-host ≠ target-box). backend=all.
