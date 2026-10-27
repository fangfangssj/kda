---
type: Runbook
title: "A rollback/verdict reason names the first failing check, not the failing set — re-run the whole check list read-only after remediation"
description: "A rollback/verdict reason names the first failing check, not the failing set — re-run the whole check list read-only after remediation"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-401"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-401 — OL-401: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (orchestrator discipline, not silicon); cann=all; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0; op=43_CoordAtt 2026-08-14。

**OL-401** -- a rollback/verdict reason names the FIRST failing check, not the failing set: first-failure-wins pipelines emit a PREFIX, so fixing the named check and handing off `done` is unsound by construction. Re-run the whole check list read-only and treat that set as the completion criterion. Recipe: `import finalize_pipeline` FIRST (breaks the import cycle), then `from finalize_dispatch import _pass_branch_gate_specs`, then call each `fn(workspace, verification)` (`None` == pass). Anchor: 43_CoordAtt 2026-08-14 -- reason named gate #11; two further gates (#15 + `kb_writeup`) were red and unreported. backend=ascendc.

- Former A5Ops ID: `OL-401`
- Fusion disposition: `additive_new`

`applies_to: soc=all (orchestrator discipline, not silicon); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0; op=43_CoordAtt 2026-08-14`

**Principle**: pipelines that early-return on the first failing check emit a reason that is a
**prefix** of the failing set. Fixing the named check and handing off `done` is therefore unsound
by construction -- the next check may already be red and was simply never reported. The reason
field answers "what stopped us first", never "what is wrong".

**Rule**: after remediating a named check, re-run the **entire** check list read-only and treat the
resulting set -- not the original reason -- as the completion criterion.

**Concrete anchor** (the import order is the load-bearing part -- importing the dispatch module
first raises a circular-import error):

```python
import finalize_pipeline                                   # FIRST, breaks the import cycle
from finalize_dispatch import _pass_branch_gate_specs
for gate_id, fn in _pass_branch_gate_specs():
    err = fn(workspace, verification_dict)                 # None == pass
```
