---
type: Runbook
title: "A convention mismatch between reference and kernel presents as a plausible precision defect, never as a crash — verify activation state, layout, and dtype casts per boundary tensor before diagnosing arithmetic"
description: "A convention mismatch between reference and kernel presents as a plausible precision defect, never as a crash — verify activation state, layout, and dtype casts per boundary tensor before diagnosing arithmetic"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-440"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-440 — OL-440: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=all。
- 验证来源：soc=Ascend950PR; cann=9.1.0。

**OL-440** — **A CONVENTION mismatch between reference and kernel presents as a plausible precision defect, never as a crash.** Activation state (logit vs already-activated), state LAYOUT, and silent dtype casts must be verified per boundary tensor BEFORE diagnosing arithmetic. Anchor: a `beta` double-sigmoid cost **0.342 → 0.00166 nRMSE** and was investigated as a kernel defect for rounds. ⚠️ Some instances are invisible in one regime and dominant in another (state layout: 0.61 at weak decay, 0.0019 at strong) — a test run only in the damping regime reports CLEAN.

- Former A5Ops ID: `OL-440`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.1.0`
`unverified_on: soc=Ascend910_220x (methodology is chip-independent, but not re-measured there)`

**Principle.** When a kernel is graded against a reference, the two sides must agree not only
on the MATH but on the CONVENTIONS of every tensor crossing the boundary: whether an input
arrives **already activated or as a raw logit**, the **layout** of a state/accumulator, and the
**dtype** it is expected in. A convention mismatch does not raise — both sides are internally
correct — so it surfaces as a **numerically stable, plausibly-scaled error** that is
indistinguishable from a genuine kernel defect, and it survives every sanity check that only
asks "is the output finite and roughly the right magnitude".

**Why it is dangerous rather than merely annoying.** The error it produces has the *texture* of
a real defect: constant across sequence length, scaling smoothly with a physical parameter, and
of a magnitude (10–40 % relative) that invites a mechanism story. An investigator who already
has a hypothesis will find the data fits it.

**Detection rule — before diagnosing any cross-implementation gap, verify per boundary tensor:**
```
activation state : does EITHER side apply sigmoid/softplus/exp internally? (grep the kernel
                   AND the reference for the activation, not just the reference)
layout           : is a [K,V] accumulator consumed as [V,K] anywhere? (look for `.t()` /
                   transpose on the state at its USE site, not only where it is stored)
dtype            : does the wrapper silently cast an fp32 input down (e.g. `.to(bfloat16)`)?
```
If any answer differs between the two sides, fix that FIRST — a convention gap will otherwise
be re-diagnosed as an arithmetic gap.

**Concrete anchor.** A gated-linear-attention forward whose `beta` is a **logit**: the reference
activates it internally —
```python
beta_act = torch.sigmoid(bc).unsqueeze(-1)   # output/flashkda_cuda_a5/src/kernels/flash_kda_fwd/model.py:138
```
— and so does the kernel, and so does upstream (内部路径（不公开）torch_ref.py:214, MoonshotAI/FlashKDA,
"beta is bf16 logits"). Feeding a pre-activated `sigmoid(x)` therefore computed
`sigmoid(sigmoid(x))`, compressing beta from (0,1) to ~(0.5,0.73).

**Regime-dependent visibility (the second trap).** The same class of mismatch can be **invisible
in one operating regime and dominant in another**. The state-layout instance below is only
visible at WEAK decay, because strong decay damps the initial state before it can express the
error:
```
state consumed transposed:  `vv = (vc - k_decayed @ st.t()) * beta_act`
                            output/flashkda_cuda_a5/src/kernels/flash_kda_fwd/model.py:152
initial_state supplied as-is  -> nRMSE 0.61 at weak decay, 0.0019 at strong decay
```
⇒ **A convention test run only in the strong-forgetting regime reports CLEAN.** Choose the
regime that lets the state persist.
