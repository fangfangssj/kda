---
type: Runbook
title: "Profile only in the compute-bound regime, and establish it for the baseline too — launch-bound shapes inverted three conclusions"
description: "Profile only in the compute-bound regime, and establish it for the baseline too — launch-bound shapes inverted three conclusions"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-328"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-328 — OL-328: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950PR_9579; cann=9.1.0-beta.1。

Profile only in the compute-bound regime, and establish it for the BASELINE too — launch-bound shapes inverted three conclusions

- Former A5Ops ID: `OL-328`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0-beta.1; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR_9579; cann=9.1.0-beta.1`

**Principle.** A profile taken where fixed cost dominates measures the fixed cost, not the kernel. The
failure is not that the numbers are noisy — they are stable and reproducible — but that **pipe ratios
measured in the launch-bound regime can invert when the shape grows**, so conclusions drawn there are not
merely imprecise, they are backwards.

**The measured instance.** Profiling at `T=256` (the largest shape in that operator's own case set) and then
re-profiling the same operator at `T=2048 H=32` inverted three conclusions:

```
cube utilisation        6.8%  ->  56.9%     "the cube is idle"        -> false
one stage's share       8.2%  ->  33.7%     "that stage is minor"     -> false
AIV-only share         10.6%  ->  36.5%     "this is a CV-fused op"   -> false at the real shape
```

**Decision rule.** Before quoting a profile, show that the shape is compute-bound **for both sides of any
comparison you intend to draw**. A cheap test is a scaling control: double the work and check the time
roughly doubles. If it does not, the measurement is dominated by fixed cost. This applies to the *reference*
too — a GPU baseline that is flat in the sequence length is not computing either, and a ratio taken against
it describes two fixed-cost regimes rather than two kernels.

- **Evidence**: FlashKDA A5 port, 2026-08-01. Root cause of the inversion was parallelism, not measurement
  error: block count derives from `B*H`, so the small shape used 2 of 28 cube cores.
- **Other instances (predicted)**: any operator whose case set was authored for precision coverage rather
  than perf — precision sets favour small shapes because goldens are cheap there, so the largest available
  case is routinely still launch-bound.
