---
type: Runbook
title: "A reused buffer needs both direction hazards — the reverse edge is the forgotten one and omitting it returns plausible wrong numbers, not a crash"
description: "A reused buffer needs both direction hazards — the reverse edge is the forgotten one and omitting it returns plausible wrong numbers, not a crash"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-316"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-316 — OL-316: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=any tiled cube kernel; kernel_type=MIX。
- 验证来源：soc=Ascend950PR (957b); cann=9.1.0 — two independent instances in one kernel。

**OL-316** — **A reused buffer needs BOTH direction hazards; the REVERSE edge is the forgotten one and omitting it returns plausible wrong numbers, not a crash.** Forward (producer done → consumer may read) is always written because its absence breaks iteration 1; **reverse** (consumer done → producer may overwrite) only bites from iteration 2 and silently races. **Two instances in ONE kernel**: segment loop missing `FIX_MTE2` ⇒ segments of one length exact and others wrong (**length-dependent, not position-dependent**); K-tile loop missing `MTE1_MTE2` ⇒ 16306/16384 wrong at K>KT but correct whenever the loop ran once. ⭐ **Diagnostic**: a fault tracking a LOOP PARAMETER rather than position implies buffer reuse (only reuse makes iteration N depend on N-1); a kernel correct when its loop runs ONCE and wrong when twice is a missing reverse hazard until proven otherwise. ⚠️ Two capacity hypotheses were refuted first (L0A → M-tiling implemented, still failed; then L0B) — capacity stories are seductive because the failure IS about buffers, and absorb unlimited debugging; the control that localised it was **running the NEW tiled code at the KNOWN-GOOD size** (0/16384 ⇒ rewrite sound, fault specific to multi-tile). **Apply by reading, not measuring**: enumerate buffers reused across iterations and confirm both edges before any experiment.

- Former A5Ops ID: `OL-316`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=any tiled cube kernel; kernel_type=MIX`
`verified_on: soc=Ascend950PR (957b); cann=9.1.0 — two independent instances in one kernel`
**Principle.** Any buffer written by one pipe and read by another needs **two** `SetFlag`/
`WaitFlag` edges, not one: the **forward** edge (producer done → consumer may read) *and* the
**reverse** edge (consumer done → producer may overwrite). The forward edge is the one everybody
writes, because omitting it visibly breaks the first iteration. **The reverse edge only matters
from the second iteration onward, and omitting it does not crash — it silently races and returns
numbers of the right shape, dtype and magnitude.**
**Two instances in the same kernel, found weeks apart in effort:**
| loop | missing edge | symptom |
|---|---|---|
| per-expert segment loop | `FIX_MTE2` (Fixpipe drained L0C → next segment may refill L1/L0) | segments of one size exact, others wrong — **length-dependent, not position-dependent** |
| K-tile loop | `MTE1_MTE2` (LoadData read L1 → next tile may overwrite L1) | 16306/16384 wrong at K>KT; correct whenever the loop ran only once |
⇒ In both cases the forward edge (`MTE2_MTE1`) was present and looked complete.
**Diagnostic value of the symptom shape.** A fault that tracks a *loop parameter* (segment
length, tile count) rather than *position in the loop* points at buffer reuse, because only reuse
makes iteration N depend on iteration N-1's completion. That signature is what separated this
from ordering and offset bugs. Conversely, a kernel that is correct whenever its loop executes
**once** and wrong when it executes twice is a missing reverse hazard until proven otherwise.
⚠️ **Two capacity hypotheses were refuted before the real cause was found** — "L0A too small, so
tile M as well" (M-tiling was implemented; it still failed) and "L0B too small". Buffer-capacity
explanations are attractive because the failure *is* about buffers, and they will absorb any
amount of debugging effort. **The control that localised it: run the NEW tiled code at the
KNOWN-GOOD size.** It passed `0/16384`, proving the rewrite sound and the fault specific to
multi-tile operation — see the sibling rule that a bisect needs an arm which PASSES.
**How to apply.** Before debugging a tiled kernel, enumerate every buffer reused across
iterations and confirm **both** edges exist for each. This is cheaper than any measurement:
it is a read of the source, and it would have saved two device-experiment cycles here.
