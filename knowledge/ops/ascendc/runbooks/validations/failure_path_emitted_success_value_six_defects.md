---
type: Runbook
title: "Six observed defects in one session shared one shape: the failure path emitted the success value and no error"
description: "Six observed defects in one session shared one shape: the failure path emitted the success value and no error"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-510"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-510 — OL-510: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=any; cann=any; bisheng=n/a; op_class=any; build=any harness with gates, probes, or generated receipts。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention, main 2026-09-05 (six instances, each reproduced)。

**OL-510** — **six observed defects in one session shared one shape: the failure path emitted the SUCCESS value and no error.** ⚠️ Recorded as six observations, NOT as a measured claim that this family is the most common, that other defect classes are rarer, or that only a control can catch it — none of those was measured. The instances were: `BUILD_RC=0` while the `#ifdef` was never compiled (DSO byte-identical); assigning `allow_tf32=False` to a knob that is `None` on this build; a probe advertising a "hard timeout" via SIGALRM that HANGS instead of rejecting (Python signal handlers do not run inside native calls); `ADM=$?` after `... | grep VERDICT`, which reads GREP's status so `VERDICT=REJECT` matched and a fail-closed admission gate became fail-OPEN; `except Exception: pass` swallowing a `TypeError` so a receipt shipped with no identity while reporting success; a release script that ran cleanly but named the PREVIOUS window's container, leaving the card held. **Defence: make the control go RED on the broken version BEFORE trusting a green from the fixed one**, and prove the STIMULUS lands (a mutation control needs a positive control on the mutation). Cross-ref [OL-511](../runtime/tensor_reference_pins_allocation_not_contents.md), [OL-512](free_healthy_unclaimed_is_not_usable_live_execution.md), [OL-507](build_time_identity_cannot_be_reconstructed_afterwards.md), [P-P68](../../optimizations/p_p68_single_aic_gemm_with_constexpr_static_tiling_on.md). backend=all.

- Former A5Ops ID: `OL-510`
- Fusion disposition: `additive_new`

`applies_to: soc=any; cann=any; bisheng=n/a; op_class=any; build=any harness with gates, probes, or generated receipts`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention, main 2026-09-05 (six instances, each reproduced)`
`unverified_on: nothing — the pattern is language- and platform-independent; the six anchors are AscendC/bash/Python but the failure shape is not specific to them`

- **Category**: methodology / instrument integrity
- **Trigger**: you are about to believe a green result from a gate, probe, control or receipt.
- **Principle**: these are hard to notice because the artifact of failure is identical to the artifact of success. Two of the six were in fact caught by a reviewer re-reading artifacts — i.e. **not** only by a control; that is an observation about how they were found, not a claim that controls are the sole route.
- **What "the control works" means here, precisely**: a REGRESSION ASSERTION that FAILS against the old/broken version and PASSES against the fixed one. It does **not** mean a gate that wrongly let something through and was then called a success — a gate erroneously passing is the defect, not evidence of a working control.
- **Procedure (three steps, all required)**:
  1. construct the failing case (a rejecting probe, a hanging child, a missing define, a mutated input);
  2. run the CONTROL against the OLD/broken code and confirm the control **FAILS (goes red)** — that is what proves it can fail. (This step previously read "confirm it **passes**", which contradicted the definition one bullet above: a control is an assertion that FAILS on broken and PASSES on fixed. Read literally it endorsed the opposite of the rule it belongs to.);
  3. only then accept a green from the fixed code.
  A control that cannot fail the broken version proves nothing. Example of one that could not: `libc.pause()` used to simulate a "native hang" is interruptible by design, so SIGALRM woke it and a fake probe passed cleanly.
- **Corollaries**: verify a staged/instrumented file by READING it (`grep -c <marker>`), never by `py_compile` — compiling proves syntax, not presence. Verify a card release by re-scanning device mappings, never by the script's exit status. Prefer THREE outcome codes (ok / fail / instrument-broken) over two, so an instrument can report that it is broken instead of returning one of the two answers. Never wrap evidence-producing code in a bare `except` — the swallow IS the bug.
- **Evidence**: main 2026-09-05, 4-42 CoTAttention lane record `TERMINAL_42_PAIRED.md` ADDENDUM 24/28/29; each of the six reproduced, and the corrected controls demonstrated red-on-broken before green-on-fixed.
- **Other instances (predicted)**: any acceptance gate whose status comes through a pipe; any timeout implemented in-process; any cache/memo whose key omits a mutable input; any "verified" claim resting on a compile rather than a read.
- **Cross-ref**: OL-511 (cache content guard), OL-512 (card admission), OL-507, EC-68. backend=all.
