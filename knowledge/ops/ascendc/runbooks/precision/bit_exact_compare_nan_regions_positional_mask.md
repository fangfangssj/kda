---
type: Runbook
title: "Bit-exact comparison vs a reference with legitimate NaN regions needs positional NaN-mask agreement + nan_to_num bit-compare"
description: "Bit-exact comparison vs a reference with legitimate NaN regions needs positional NaN-mask agreement + nan_to_num bit-compare"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-549 — OL-549: A5Ops additive extension

> 关联：同一 OL-549 主题已由既有卡 [ol_549_bit_exact_comparison_vs_a_reference_with_legitim.md](ol_549_bit_exact_comparison_vs_a_reference_with_legitim.md) 收录；本卡不再声明该 alias，仅作补充证据。

## 适用场景

- 适用平台与范围：soc=all; cann=all; bisheng=n/a; op_class=all (verifier/runner-side)。
- 验证来源：卡内无 verified_on 行；验证证据见正文 Evidence 行。

**OL-549** — **bit-exact comparison vs a reference with legitimate NaN regions needs positional NaN-mask agreement + `nan_to_num` bit-compare**: `torch.equal` alone fails on ANY NaN (manufactures a fake "0/N bit-exact"); an all-NaN fallback clause silently mis-grades MIXED NaN/finite outputs (~half-NaN reference with bit-identical finite parts fails BOTH arms and reads as a kernel bug when the predicate is the bug). soc=all, verifier/runner-side. 17_AdamW 310p 2026-08-15: fake 0/18 scare → corrected predicate (内部路径（不公开）workspace/17_AdamW/pass_a_runner.py:71-93) → 13/18 + 605/645 bit-exact, residuals root-caused as hardware classes (PB-120 saturation, OL-147 FTZ). Cross-ref [OL-109](ol_109_two_tier_precision_verdict.md), [OL-85](ol_85_logic_first_precision_fix_anti_overfitting_anti_re.md), [OL-548](../validations/piped_build_verify_chain_grades_thought_artifact.md), [PB-120](../compilation/dav_m200_vconv_per_direction_cast_mode_whitelist.md). backend=all.

- Former A5Ops ID: `OL-549`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (verifier/runner-side)`
<!-- applies_to_backend: all -->

**Principle**: when a reference legitimately produces NaN in a SUBREGION of the output (Sqrt of negatives, 0/0, log of ≤0, inf−inf — e.g. an optimizer update whose v_hat < 0 feeds a Sqrt), a bit-exactness predicate must check TWO facts independently: (1) the NaN MASKS agree positionally, and (2) the finite values are bit-identical after `nan_to_num`. Two broken forms:
- `torch.equal(out, ref)` returns False if even ONE element is NaN on BOTH sides — an any-NaN reference can never pass, manufacturing a fake "0/N bit-exact" result that reads as a kernel precision bug.
- An "all-NaN everywhere" fallback clause (`torch.equal(o, r) or (o.isnan().all() and r.isnan().all())`) is silent poison for MIXED outputs: a reference that is ~half NaN with bit-identical finite parts fails BOTH arms, and the failure again reads as a kernel bug when the predicate is the bug.

**Concrete anchor** (correct form):
```python
an, bn = torch.isnan(a.float()), torch.isnan(b.float())
return torch.equal(an, bn) and bool(
    torch.equal(torch.nan_to_num(a.float(), 0.0),
                torch.nan_to_num(b.float(), 0.0)))
```
(内部路径（不公开）workspace/17_AdamW/pass_a_runner.py:71-93; same predicate in `pass_b_runner.py`.)

**Evidence**: 17_AdamW 310p (2026-08-15, benchmark): the broken predicate manufactured a fake "0/18 bit-exact" scare (failures_ledger iter-5); corrected predicate → pass_a 13/18, pass_b 605/645 bit-exact, with every residual class root-caused as hardware classes (fp16 overflow saturation per PB-120, denormal FTZ per OL-147), NOT predicate artifacts (内部路径（不公开）workspace/17_AdamW/failures_ledger.md:16).

**Other instances (predicted)**: any verifier/runner over references with legitimate NaN subregions — Sqrt/log-domain ops, 0/0 divisions, attention masks chaining −inf→NaN; also mixed inf/saturation outputs where mask-then-compare beats scalar equality.

**Cross-ref**: OL-109 (two-tier T1/T2 judging this predicate feeds), OL-85 (logic-first — predicate bugs are harness bugs, fixed harness-side), OL-548 (sibling "grade what you think you graded" class — here the grader itself lied), PB-120 (the NaN→65504 saturation that makes NaN-region outputs legitimate in the first place). backend=all.
