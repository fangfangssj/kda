---
type: Runbook
title: "Bind identity by content the consumer recomputes — a timestamp proves ordering, never identity"
description: "Bind identity by content the consumer recomputes — a timestamp proves ordering, never identity"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-505"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-505 — OL-505: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (producer->consumer handoff of a built or merged artifact)。
- 验证来源：soc=Ascend950PR; archive corpus + harness gates, 2026-08-18。

**OL-505** — Bind identity by content the consumer recomputes — a timestamp proves ordering, never identity (`touch`, same-nanosecond windows, and other-source artifacts all satisfy it). Producer writes the sha atomically at the moment of the act; consumer recomputes at use time; never copy the producer's value into the consumer's field. Scope the comparison to one generation: a same-source rebuild need not be byte-identical. Does NOT assert generation scope: a tracker-column filter decides what the next generation produces; the reference matrix decides which existing archives may serve as a bounded reference. Read the scope list from its authority at time of use — its criterion and membership were both restated within hours on 2026-08-18, so no count is recorded here.

- Former A5Ops ID: `OL-505`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=n/a; bisheng=n/a; op_class=all (producer->consumer handoff of a built or merged artifact)`
`verified_on: soc=Ascend950PR; archive corpus + harness gates, 2026-08-18`

A producer-recorded hash that no one recomputes is a single-party assertion. A timestamp is weaker still: it
orders two events without saying they concern the same thing, so `touch`, a same-nanosecond window, or an
artifact built from different sources all satisfy it.

**Rule.** The producer writes the binding atomically at the moment of the act, naming the sha256 of the exact
bytes it acted on; the consumer **recomputes** those bytes at use time and compares. Missing, unreadable, or
other-content bindings mean "cannot prove it happened", which must read as NOT happened. Never copy the
producer's value into the consumer's field — that makes them agree by construction and proves nothing.

- **Scope boundary**: this entry governs archive/harness mechanics only; it does NOT determine generation eligibility. Read the current op scope from its authoritative tracker at time of use — it is a point-in-time list and was restated twice within hours on 2026-08-18.
- **positive**: build records `binary_sha256` after a successful build; the verifier recomputes from disk before
  measuring; finalize requires the two to agree.
- **anti-pattern**: `marker.mtime_ns >= source.mtime_ns` as proof of ingestion; binding an intermediate `-c`
  object that is never loaded or measured in place of the loadable artifact; or attributing a result to a
  binary by its path or filename instead of by the binary hash it carries, checked against a hash the
  consumer recomputes from that binary.
- **falsifier**: two artifacts with equal content hash that behave differently in the consumer would break the
  premise that content identity is sufficient here.
- ⚠️ **Scope the comparison to one generation.** A same-source rebuild need not be byte-identical (timestamps,
  link order), so requiring a future rebuild to reproduce a stored hash is a gate that fails on correct work.
- **Evidence**: 2026-08-18 — an mtime freshness rule was defeated by `touch` on a stale marker (the arm asserts
  the forgery is clock-valid before requiring it be refused); a host-only `-c` compile's `packaged.o` was found
  recorded where a loadable `.so` was assumed.
- **Evidence**: 2026-09-07, 4_GroupedMatmulSwigluQuant (SoC not stated in the record) — the path/filename
  anti-pattern above. Copying an arm with `cp -a` carries its parent's `evidence/` with it, and those result
  files are not re-run, so an arm's evidence directory can describe a different binary than the arm it sits
  in: `evidence/p_nw24.json` was present in 7 arms carrying `elf_sha256` `985895cd…`, the base arm's binary,
  not the arm's own. Independently, the `tile128` arm's own precision result was written under the name
  `p_nw24.json`, because `run_t128.sh` passed an `--output` path belonging to a different experiment.
  The filename pointed one way and the identity another, in both directions at once.
  The record's rule, attribute a result by its `elf_sha256` / `candidate_sha256` and never by its path or
  filename, still rests on a single-party assertion (above) until the consumer compares the carried sha
  with a hash it recomputes from the binary; the record does not say whether that comparison was made.
  Evidence (agent-team-wiki @d813788c): `projects/op4_4/FINDING_20260907T1735Z_DS_COPIED_ARM_BUILDS_INTO_ITS_BASE.md`.
- **Cross-reference**: OL-506 (wiring), OL-503 (layers), OL-548 (the build-side face of the same `cp -a`
  record), OL-514 (a grading runner identified by filename instead of sha256).
