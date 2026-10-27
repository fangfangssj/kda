---
type: Runbook
title: "A resume dry-run's action=resumable proves continuation, not run-identity preservation — compare run_id and intake_sha256 across the real launch"
description: "A resume dry-run's action=resumable proves continuation, not run-identity preservation — compare run_id and intake_sha256 across the real launch"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-420"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-420 — OL-420: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=any; scope=orchestrator-resume。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0.B060。

**OL-420** — A resume dry-run's `action=resumable` proves continuation, NOT run-identity preservation — compare `run_id`+`intake_sha256` across the real launch

- Former A5Ops ID: `OL-420`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR_9579; cann=9.1.0.B060; bisheng=n/a; op_class=any; scope=orchestrator-resume`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0.B060`
`unverified_on: other SoCs, CANN releases, and non-benchmark op-gen modes`

**Trigger**: about to report "same-run resume is supported" on the strength of
`src/scripts/orch <op> --resume --dry-run` exiting `rc=0` with
`action=resumable ... mid-flight; re-invoke orchestrator to resume`.

**Principle**: `rc=0 / resumable` answers *can this workspace continue?* The claim
"the run keeps its `run_id`" is a **different predicate**, decided later by whether the
resume path hands the child a same-run token. A tree can answer the first question
affirmatively and still mint a fresh intake when the real resume executes. Checking the
weaker property and reporting the stronger one is the defect.

**Evidence (measured, run 33abdbe3-5e07-47cd-83fd-2e3926b37e22)**: dry-run at T+0 reported
`rc=0 action=resumable state=await_fused_optimizer`, and `.aog_run_intake.json` still held the
prior `run_id`. The real resume launched 11 seconds later rewrote that same file with a NEW
`run_id` and a new `intake_sha256`, `origin=raw_cli`. The orchestrator then died `rc=4` on
`SCHEMA REJECT: verification.json already carries run_id='<prior>', but this workspace's run
identity is '<new>'` — after its spawned agent had already completed successfully.

**Signature**: a clean dry-run, then a late `SCHEMA REJECT` naming two different run ids for the
same workspace; `.aog_run_intake.json` `created_at` equal to the resume launch time rather than
the original run's.

**Falsifier**: run the dry-run, record `run_id` + `intake_sha256`; perform the real resume; re-read
both. If they are unchanged, this entry does not apply to that tree. The entry is refuted for any
tree where the resume path forwards a same-run token (see OL-421).

**Rule (load-bearing, states the required comparison)**: to assert that a run KEPT its identity you
must compare `.aog_run_intake.json`'s `run_id` **and** `intake_sha256` from before the real launch to
after it. `dry-run rc=0`, "the process is still alive", and "the state is resumable" are each a
DIFFERENT predicate and none of them substitutes for that identity check. The already-landed correct
path is `9441a9e08` on main (OL-421) — consult it rather than authoring a second fix.
