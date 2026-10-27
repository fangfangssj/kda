---
type: Runbook
title: "An audit of a gate must reproduce the gate's headline number before its decomposition means anything — disagreeing with the authority means you are measuring your own golden"
description: "An audit of a gate must reproduce the gate's headline number before its decomposition means anything — disagreeing with the authority means you are measuring your own golden"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-693"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-693 — OL-296: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all (verification discipline)。
- 验证来源：FlashKDA A5 precision audit, 2026-07-30 (two wrong analyses, same cause)。

**OL-693** — **An audit of a gate must REPRODUCE the gate's headline number before its decomposition means anything; disagreeing with the authority means you are measuring your own golden** — graded runner said 17/56, my audit of the SAME build said 12/56 and I read its categories anyway; cause = runner grades vs `ref(*grp)` whose output is CAST TO BF16, my audit used raw fp64. A stricter golden is a DIFFERENT QUESTION, not a stricter audit. **Tell I ignored**: the kernel and a deliberately pessimistic all-bf16 model scored IDENTICALLY (12/56, 19/56 twice) — identical results from non-identical subjects means the number is set by something they SHARE. **Corollary**: refutation needs far less independence than corroboration — a same-author control can't carry "at the bf16 limit", but one fp32-endpoint run (56/56) killed it; prefer measuring an endpoint to constructing a bound. Cross-ref CAND-A5OPS-0016 (control-based calibration, complementary), OL-208, OL-295.

- Former A5Ops ID: `OL-296`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all (verification discipline)`
`verified_on: FlashKDA A5 precision audit, 2026-07-30 (two wrong analyses, same cause)`
**Principle**: when you write an instrument to explain *why* a gate fails — per-case
decomposition, per-stage attribution, failure taxonomy — the instrument must first
reproduce the gate's own headline number on the same artifact. If it does not, every
downstream category is a statement about your instrument's reference, not about the
thing under test. Reproducing the number is one line of output and it is not optional.
**The concrete miss**: the graded runner reported **17/56**. My audit of the same build
reported **12/56** and I read its decomposition anyway. Cause: the runner grades against
`ref(*grp)` — the reference module, whose output is **cast to bf16** — while my audit
compared against the raw fp64 tensor. A stricter golden is not a stricter audit; it is a
*different question*. After switching to the gate's golden the audit reproduced 17/56
and per-case values matched to every printed digit, and only then were the categories
usable.
**The tell I ignored**: two different implementations — the kernel and a deliberately
pessimistic all-bf16 model — scored *identically* (12/56 and 12/56), and produced
identical aggregate sub-counts (19/56 twice). **When two implementations of clearly
different quality score identically, the number is being set by something they share** —
here, my golden — not by either implementation. Identical results from non-identical
subjects is a bug signature, not a robustness signal.
**Corollary on evidence economics (same investigation)**: to *support* a claim like "the
kernel is at the bf16 limit" I built a reference implementation and then had to defend
its independence — a control written by the implementation's own author inherits that
author's misconceptions, so it cannot carry a positive limit claim. To *refute* the same
claim, one endpoint measurement sufficed: running the reference with **fp32**
intermediates scored 56/56, which proved the threshold reachable and killed the claim
outright. **Refutation demands far less independence than corroboration.** When tempted
to construct a bound, first ask whether measuring an endpoint would settle it.
**Cross-ref**: CAND-A5OPS-0016 (calibrate a probe's discriminating power with positive/negative
controls — the complementary check; this card is about agreeing with the authority on a
known point, that one is about separating two known-different inputs), OL-208
(fail-before/succeed-after on the same verifier-controlled build), OL-295.
---