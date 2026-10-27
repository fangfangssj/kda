---
type: Runbook
title: "A checkout without 9441a9e08 silently recasts the intake on --resume — detect by symbol presence before launching"
description: "A checkout without 9441a9e08 silently recasts the intake on --resume — detect by symbol presence before launching"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-421"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-421 — OL-421: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=any; scope=orchestrator-resume。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0.B060。

**OL-421** — A checkout without `9441a9e08` silently RECASTS the intake on `--resume`; detect by symbol presence before launching

- Former A5Ops ID: `OL-421`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=any; scope=orchestrator-resume`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060`
`unverified_on: other SoCs and CANN releases; not bisected beyond the landing commit named below`

**Trigger**: resuming an idle lane whose checkout is behind current main, especially a lane whose
workspace already holds a `verification.json` bound to the original run.

**Principle**: same-run continuation is carried by an explicit token plus a retained-intake claim.
When those are absent, `resume` still finds the workspace and still succeeds at the *routing* level,
but it invokes the child WITHOUT `--workspace` and without a token, so the child provisions a fresh
intake. The lane keeps its artifacts and loses its identity — and nothing reports that as an error
until a later consumer compares the two ids.

**Evidence (measured on the affected checkout)**: three symbols absent —
`AOG_SAME_RUN_RESUME_TOKEN` (0 files), `same_run_resume_authority.py` (absent),
`claim_retained_cli_intake` (0 files). The emitted child command was
`python3 .../orchestrator.py <op> --lane 0`, carrying neither `--workspace` nor a token. Result: new
`run_id`, `origin=raw_cli`, and `rc=4 SCHEMA REJECT` against the pre-existing `verification.json`.

**Landing commit (verified in-repo, not relayed)**: the mechanism arrives with `9441a9e08`
("repair op42 quota authority succession"), which adds `same_run_resume_authority.py` and its
child-identity test. A checkout is affected iff `git merge-base --is-ancestor 9441a9e08 <HEAD>`
is false. The affected lane in this instance based at a commit ~29 hours older than that.

**Signature**: `[resume] invoking: python3 .../orchestrator.py <op> --lane 0` with no `--workspace`
argument is the tell, visible in the orch log BEFORE any agent is spawned.

**Falsifier**: grep the checkout for the three symbols. If any is present, this entry does not
explain that tree's behaviour and the recast has another cause. If all three are absent yet the
`run_id` survives a real resume, the entry is refuted.

**Rule**: before resuming an idle lane, grep for the three symbols. All absent ⇒ do not resume for
identity-preserving purposes; use a current-main fresh run instead and do not inherit prior results.
The cost of skipping this check is a full agent execution whose output cannot be written back.
