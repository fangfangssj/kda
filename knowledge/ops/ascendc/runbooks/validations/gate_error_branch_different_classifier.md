---
type: Runbook
title: "A gate's error branch and its success branch can classify the same artifact with different classifiers — the field is a real measurement of the wrong predicate"
description: "A gate's error branch and its success branch can classify the same artifact with different classifiers — the field is a real measurement of the wrong predicate"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-400"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-400 — OL-400: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (gate-triage discipline, not silicon); cann=all; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0; op=43_CoordAtt 2026-08-14。

**OL-400** -- a gate's ERROR branch and its SUCCESS branch can classify the same artifact with DIFFERENT classifiers, so the field is a real measurement of the WRONG predicate -- worse than a default, because it looks like evidence. NOT "the gate never inspected the artifact": `check_architecture_class`'s contract-error branch (architecture_class_check.py:626-633) does call `_classify_generated_arch`, which rglobs `kernel/`+`op_kernel/` and greps every `.h`/`.cpp`; but it uses the legacy `CUBE_MARKER_RE` (no `Mmad`), while the CV_COOPERATIVE branch uses `find_reachable_cube_path` (which accepts `AscendC::Mmad`). Detect by calling BOTH classifiers and comparing. Does NOT loosen OL-188. Harness fix: error branch should use the reachability scanner or emit `"not-evaluated"`. Anchor: 43_CoordAtt 2026-08-14 -- `generated_arch = vec-only` printed for a kernel with **1** executable `AscendC::Mmad` (`output/npukernelbench/src/kernels/43_CoordAtt/kernel/coordatt_cube.h:189`; an earlier draft said 3 -- withdrawn) on an `AIC_ONLY` entry, while `find_reachable_cube_path` returns `found=True`; 4 spawns misrouted at the kernel. ⚠️ Its anchor workspace is NOT a verified result -- `PRODUCT_CLOSED=1` but `HARNESS_FINALIZE_CLOSED=0`, `performance.status=N/A`, 4.443× withdrawn, nothing `measured_this_spawn`, no independent O5 Pass-B receipt; see the entry's status block before citing. Cross-ref [OL-188](../compilation/ol_188_op_family_architecture_requirement_classification.md), [OL-160](../compilation/ol_160_safety_net_file_name_coupling_is_an_architectural.md). backend=ascendc.

- Former A5Ops ID: `OL-400`
- Fusion disposition: `additive_new`

`applies_to: soc=all (gate-triage discipline, not silicon); cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0; op=43_CoordAtt 2026-08-14`

**Principle**: when a gate resolves a requirement and then classifies an artifact against it, the
error return path is often *not* a bare short-circuit -- it still fills the classification field,
but frequently from a **cheaper, older classifier** than the one the success path uses. The field is
then a real measurement of the *wrong predicate*. That is strictly worse than an obvious default,
because it looks like evidence and it is internally consistent.

**Do not restate this as "the gate never inspected the artifact."** In the anchor below the gate
demonstrably *did* inspect it. Read the return path.

**What `check_architecture_class` actually does** (`src/scripts/orchestrator/checks/architecture_class_check.py`):
