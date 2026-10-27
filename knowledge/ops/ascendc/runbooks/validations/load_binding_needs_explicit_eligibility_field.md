---
type: Runbook
title: "A load-binding recorded as free-text metadata does not mark the number it accompanies; give the receipt an explicit eligibility field"
description: "A load-binding recorded as free-text metadata does not mark the number it accompanies; give the receipt an explicit eligibility field"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-515"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-515 — OL-515: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=a lane-local in-process timing harness whose contract is ONE candidate .so per timing process; build=any。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — main 2026-09-05, two lane-local harnesses only (perf_op13_events.py, perf42b.py); the predicate was exercised against 6 constructed payloads。

**OL-515** — **A load-binding recorded as free-text metadata does not mark the number it accompanies; give the receipt an explicit eligibility FIELD.** A perf harness that notes `n_mapped` / `identity_agrees` and then emits the geomean with no flag on it hands a reader a product-shaped figure whose binding they must reconstruct by hand.

- Former A5Ops ID: `OL-515`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=a lane-local in-process timing harness whose contract is ONE candidate .so per timing process; build=any`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — main 2026-09-05, two lane-local harnesses only (perf_op13_events.py, perf42b.py); the predicate was exercised against 6 constructed payloads`
`unverified_on: any harness that legitimately maps several candidate .so files, loads out-of-process, or dlopens more than one generation — there `len == 1` is the WRONG contract and this entry does not apply; also unverified as a publication gate (see the scope bullet)`

- **Category**: measurement instrument / product authority
- **Trigger**: an in-process timing run whose candidate arrives through a `pybind11` extension **and whose harness contract is a single candidate**, especially after the lane has more than one build on disk.
- **Mechanism**: the probe reads `/proc/self/maps`, matches the candidate `.so` and compares its FD identity (`identity_agrees`). Recording that alongside the number leaves the *reader* to reconstruct it. The form used here is one predicate — `_bound = bool(entries) and len(entries) == 1 and entries[0].get("identity_agrees") is True` — driving `out["product_eligible"]`, a `WARNING` key in the payload, and `[UNBOUND -- NOT A PRODUCT FIGURE]` on the printed geomean.
- **⚠️ Scope of `len == 1`**: it encodes **these two harnesses' contract** — after path de-duplication they expect exactly one candidate mapping — so zero means unbound and two means the process is not the one they describe. It is **not** a general ruling that multiple maps entries disqualify a measurement; a harness that legitimately maps several `.so` files needs a different predicate.
- **⚠️ What `product_eligible` is and is not**: it is a **reporting marker on the binding dimension**. Both harnesses still compute, still return and still print the geomean when it is false; nothing downstream is known to refuse an ineligible receipt. Calling it a publication gate would be the OL-506 error — a gate is only real once a production caller acts on it, and that has not been demonstrated here.
- **The failure that motivated it**: the probe's match token was hardcoded (`cotattn`), so on a different op it returned `n_mapped=0 / AMBIGUOUS` while the harness still emitted a complete geomean. The token is now `LOADBIND_MATCH` with a default, hoisted out of the loop — but **the parameterisation is the shallow fix**; the intent of the marker is that an unbound run should not yield a number a reader can mistake for a product figure.
- **Related trap in the same lane**: on 4-13 the precision leg and the performance leg were built from the same sources at different paths and were in fact **two different binaries** — sha256 `dd0d9cfc…c109` and `96d06ccd…`. Note the direction of that evidence: the differing hashes are what proved they differed; **a different path does not by itself prove different bytes**, and a same-source rebuild need not differ (OL-505). Pairing legs is only a same-candidate double pass once the two hashes are compared and equal.
- **Detection**: put a known-bad control through the *same* invocation. A known-bad that fails to change the verdict tells you the chain is broken **somewhere** — it does not localise the fault to the load binding, since the same silence follows from a mis-built control, an unread output, or a tolerance that swallows the perturbation. Localise by comparing the recomputed hash of the mapped `.so` against the one you meant to measure.
- **Cross-ref**: OL-514 (the precision-leg twin); OL-505 (identity by recomputed content, not by path); OL-506 (a marker is not a gate until a production caller acts on it). backend=ascendc.
