---
type: Runbook
title: "A mandated artifact that no consumer reads drifts in shape and nothing goes red — name the consumer and the field, or drop the mandate"
description: "A mandated artifact that no consumer reads drifts in shape and nothing goes red — name the consumer and the field, or drop the mandate"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-504"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-504 — OL-504: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (brief/checklist items phrased MANDATORY across modes sharing a default block)。
- 验证来源：soc=Ascend950PR; archive corpus + harness gates, 2026-08-18。

**OL-504** — A mandated artifact that no consumer reads drifts in shape and nothing goes red — its shape is unconstrained, so every generation passes and divergence surfaces only when some later consumer parses them. Name the consumer and the field, or drop the mandate, scoping removal to the mode that lacks the consumer. Anchor: a `MANDATORY` binary-registration file whose only readers are another mode's brief/tests; 19/20 vs 0/6 field presence across modes. Does NOT assert generation scope: a tracker-column filter decides what the next generation produces; the reference matrix decides which existing archives may serve as a bounded reference. Read the scope list from its authority at time of use — its criterion and membership were both restated within hours on 2026-08-18, so no count is recorded here.

- Former A5Ops ID: `OL-504`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (brief/checklist items phrased MANDATORY across modes sharing a default block)`
`verified_on: soc=Ascend950PR; archive corpus + harness gates, 2026-08-18`

A brief can require a file, and workers will produce it. If no code path reads it, its **shape** is
unconstrained: each generation writes whatever its author assumed, every one passes, and the divergence is
discovered only when some later consumer finally tries to parse them.

**Rule.** Every mandated artifact must name the consumer that reads it and the field that consumer requires. If
no consumer exists, remove the mandate rather than keep producing unread files — and scope the removal to the
mode that lacks the consumer, because another mode may genuinely depend on it.

- **Scope boundary**: this entry governs archive/harness mechanics only; it does NOT determine generation eligibility. Read the current op scope from its authoritative tracker at time of use — it is a point-in-time list and was restated twice within hours on 2026-08-18.
- **positive**: mandate the artifact where a consumer parses it, and gate on the field that consumer reads.
- **anti-pattern**: a shared default block inherited by a mode with no consumer for it — the inheriting mode's
  workers are told the file is mandatory while nothing validates what they wrote.
- **falsifier**: find any production consumer on that mode's path that opens the file; the mandate is then
  justified and this entry does not apply.
- **Evidence**: 2026-08-18 — `op_host/config/ascend950/<op>_binary.json` is called
  "MANDATORY — A5 binary registration" by the shared default block. Repo-wide, its only readers are the
  port_a3 brief, port_a3 tests, and the legacy provider; the real benchmark builder never opens it. Measured
  shape drift: port_a3 19/20 configs carry `op_type`, benchmark 0/6 do, and one archive's file is a
  simplified-key map — 1 of 25 in the corpus.
- **Cross-reference**: OL-160 (mode entry-point name coupling), OL-506 (a gate nothing calls).
