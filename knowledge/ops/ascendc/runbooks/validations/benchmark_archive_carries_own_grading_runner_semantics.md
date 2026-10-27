---
type: Runbook
title: "A benchmark archive can carry its own copy of the grading runner, and that copy can implement a different acceptance semantics from the repo's"
description: "A benchmark archive can carry its own copy of the grading runner, and that copy can implement a different acceptance semantics from the repo's"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-514"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-514 — OL-514: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any op graded by an ARCHIVE-LOCAL copy of precision_eval_two_tier.py predating the resolve_selection fix; build=any。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — 4-13 ScaledDotProductAttention, main 2026-09-05, against the archive runner sha256 5298f943…6a1f (2026-09-01, output/npukernelbench/.../13_ScaledDotProductAttention/runners/): the SAME DSO and the same archive graded 37 PASS_T1 / 8 PASS_T2 / 5 FAIL without the flag and 50/50 PASS_T1 with it。

**OL-514** — **A benchmark archive can carry its OWN copy of the grading runner, and that copy can implement a different acceptance semantics from the repo's.** In the archive copy used on 4-13 (`precision_eval_two_tier.py`, sha256 `5298f94347ceec99c834184c87b103c99bad1b414d9f9ac831f06d18b38c6a1f`, 2026-09-01), `selected_profile` stays `None` unless `--precision-profile-binding` is passed, and the run falls back to that script's own built-in MERE/MARE tiers with no warning. **This is NOT current `src/scripts/` behaviour** — see the correction note below.

- Former A5Ops ID: `OL-514`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any op graded by an ARCHIVE-LOCAL copy of precision_eval_two_tier.py predating the resolve_selection fix; build=any`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — 4-13 ScaledDotProductAttention, main 2026-09-05, against the archive runner sha256 5298f943…6a1f (2026-09-01, output/npukernelbench/.../13_ScaledDotProductAttention/runners/): the SAME DSO and the same archive graded 37 PASS_T1 / 8 PASS_T2 / 5 FAIL without the flag and 50/50 PASS_T1 with it`
`unverified_on: the repo runner at src/scripts/precision_eval_two_tier.py — it does NOT have this behaviour (see the correction note); and any other consumer, whose own runner copy must be checked before assuming this shape`

- **Category**: measurement instrument / acceptance semantics
- **Trigger**: you invoke a precision runner directly (rather than through the orchestrator phase that supplies the binding) and read its verdict as the product verdict.
- **Mechanism (in the archive copy)**: in `evaluate()`, `selected_profile = resolve_binding(...)` runs **only** under `if precision_profile_binding is not None`; both verdict sites are `if selected_profile is None: <built-in tiers> else: selected_profile.compare(...)`. The built-in Tier-1 rule is `ours_MERE < per-dtype threshold AND ours_MARE < 10x threshold`. The `npukernelbench_v1` profile is a **distinct algorithm** (`algorithm: "npukernelbench_v1"`, not `allclose_v1`, which is the separate user-profile algorithm): per-dtype `allclose` tolerances + `npu_limits` + `required_matched_ratio: 0.9`. It has **no MARE axis at all**, which is why the 5 unbound failures — all MARE-driven — could not exist under it.
- **Why it is dangerous**: the fallback is not a degraded mode, it is a *different standard*, and it can be either stricter or looser than the product one. The output is a normal-looking verdict table, so the result reads as a finding about the kernel.
- **⚠️ Already fixed in the repo — do not read this as current `main` behaviour**: `src/scripts/precision_eval_two_tier.py` now resolves an omitted binding through `resolve_selection()` to the registered default, and reaches the legacy classifier only when a binding explicitly names `a5_two_tier_v1` (`legacy_two_tier = profile_binding.profile_id == "a5_two_tier_v1"`). The entry is retained because the **archive-local copy** is what a lane actually invokes, and it can lag the repo by an arbitrary amount.
- **What it cost**: a "the candidate regressed on DT" report — and, built on top of it, an fp32-concentration reading, a MARE interpretation and an HF32 hypothesis, all of which had to be withdrawn. None of them were ever about the kernel.
- **Detection**: have the runner **echo the selected profile id and hash into the receipt**, and treat a receipt with no profile identity as ungraded; record the runner's own sha256, not just its filename. "I used the same runner and the same truth source" does not entail "I used the same criterion" — the truth source and the judge are two independent choices, and a same-named file at a different path may be a different generation.
- **Cross-ref**: OL-515 (the same class on the performance leg); OL-505 (bind identity by content the consumer recomputes, not by path or name). backend=ascendc.
