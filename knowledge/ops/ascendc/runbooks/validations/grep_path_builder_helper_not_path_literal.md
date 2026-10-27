---
type: Runbook
title: "To prove an artifact has no writer, grep the path-builder helper symbol, not the path literal — a reader-only grep is a false negative that manufactures phantom harness-wide defects"
description: "To prove an artifact has no writer, grep the path-builder helper symbol, not the path literal — a reader-only grep is a false negative that manufactures phantom harness-wide defects"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-402"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-402 — OL-402: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (investigation methodology, not silicon); cann=all; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0; op=43_CoordAtt 2026-08-14。

**OL-402** -- to prove an artifact has no writer, grep the path-BUILDER helper, not the path literal: in a codebase that centralises path construction the literal appears ONLY in the builder, so a reader-only grep is a FALSE NEGATIVE that manufactures phantom harness-wide defects. Recipe: read the predicate -> `ls` the artifact (absence != no writer) -> grep the helper SYMBOL incl. thin aliases -> only then may you claim reader-only. A writer that exists but did not run is a different escalation (provisioning, not code). Anchor: 43_CoordAtt 2026-08-14 -- `grep architecture_requirements` returned 2 readers -> "no producer exists" -> claimed harness-wide P0; the writer is `o17_contract.py:203` via `phase_o17_classify.py:659` (line 669 is a DIFFERENT call, `current_bound_classification_status`; an earlier draft cited 669 -- withdrawn). Cross-ref [OL-160](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md). backend=ascendc.

- Former A5Ops ID: `OL-402`
- Fusion disposition: `additive_new`

`applies_to: soc=all (investigation methodology, not silicon); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0; op=43_CoordAtt 2026-08-14`

**Principle**: "this gate reads a file nothing ever writes" is a high-consequence claim -- it
reclassifies a local failure as a harness-wide structural defect, escalates it past every worker,
and justifies a ROADMAP P0. It is also **easy to establish falsely**, because in any codebase that
centralises path construction, the path literal appears *only* in the builder. Writers reach the
file through the helper and match no grep for the path string. The absence of the string is not
the absence of the writer.

**Correct recipe** (each step is what the previous one misses):

1. Read the gate's failing **predicate**, not its remediation prose.
2. Resolve the exact path of every artifact in that predicate and `ls` it. *(absence != no writer)*
3. Find the **helper** that builds the path -- not the literal -- then grep for **that symbol**
   across the repo, including thin aliases that re-export it.
4. Only a hit set that is reader-only *after step 3* supports "no writer exists". A writer that
   exists but did not run is a **different** escalation with a different owner and a different fix
   (provisioning/execution, not code).

**Worked example -- the false negative this entry exists to prevent:**

```
grep -rn "architecture_requirements" --include=*.py . | grep -v /tests/
  -> 2 hits, both READERS, both in architecture_class_check.py   # "no writer exists" (WRONG)
```

The writer is `o17_contract.begin_current_architecture_contract`
(`src/scripts/orchestrator/o17_contract.py:203`), which calls
`_atomic_json(_controller_requirement_path(workspace), ...)` -- and `_atomic_json` `mkdir`s the
absent parent tree. It is invoked from `src/scripts/orchestrator/phase_o17_classify.py:659` via the
alias `_retain_contract_requirement` (`src/scripts/orchestrator/o17_contract.py:456`, a one-line
re-export of `begin_current_architecture_contract`). Cite **659**, not 669: line 659 is
`begin_error = _retain_contract_requirement(workspace, source_identity)`, whereas 669 is the closing
paren of the *different* call `current_bound_classification_status(workspace, source_identity)` at
line 667 -- an earlier draft of this entry cited 669 and that line number is withdrawn. The literal
`"architecture_requirements/"` occurs once in the repo, in the path builder at
`src/scripts/orchestrator/checks/architecture_class_check.py:125`.
