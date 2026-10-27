---
type: Runbook
title: "OBSERVED — `507014` deadlock in a Layer-1 `MatmulImpl` + manual `CrossCore*` + `MIX_AIC_1_2` attention kernel on Ascend950DT_9582 (CANN 9.1.0)"
description: "OBSERVED — `507014` deadlock in a Layer-1 `MatmulImpl` + manual `CrossCore*` + `MIX_AIC_1_2` attention kernel on Ascend950DT_9582 (CANN 9.1.0)"
tags:
- task.debugging
- topic.compilation
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["PB-100"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# PB-100 — PB-100: OBSERVED — `507014` deadlock in a Layer-1 `MatmulImpl` + manual `CrossCore*` + `MIX_AIC_1_2` attention kernel on

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582；CANN 9.1.0；op_class=mixed_aic_aiv_with_high_level_matmul_library；macro=KERNEL_TYPE_MIX_AIC_1_2。
- 验证来源：a5_ops 45_CrossformerAttention run_id 1d9140e9-203c-4501-83b4-a2fb203aad2d（2026-08-30）— orchestrator 无关 verification.json sha256 90b83287…08405；archive id op45_archive_20260830T0625Z（42/42 OK）。

`applies_to: soc=Ascend950DT_9582; cann=9.1.0; op_class=mixed_aic_aiv_with_high_level_matmul_library; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: a5_ops:45_CrossformerAttention run_id 1d9140e9-203c-4501-83b4-a2fb203aad2d (2026-08-30) — orchestrator-independent verification.json sha256 90b832877680991152861d6304f07389df0e4892fa3b2afbe6ae5f88b8708405; archive id op45_archive_20260830T0625Z (MANIFEST.root.sha256 bf0759fd… → b4fd5d96…, 42/42 OK)`
`relation: RELATED SURFACE to PB-34 (same call-site shape). NOT a scope extension, NOT a root-cause identity.`
`mechanism: UNDETERMINED — see "What is NOT claimed"`

**This entry records an OBSERVATION. It does not identify a mechanism, and it is NOT a scope
extension of [PB-34](pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md).**

The first version of this entry attributed the hang to PB-34's mechanism — a KFC client claiming the
same FFTS sync slots as a user-owned flag chain. **That attribution was wrong and is withdrawn.** The
kernel's own header states it uses `matmul::MatmulImpl` as a **Layer-1 direct template with NO KFC,
NO `MatmulClient`, NO `REGIST_MATMUL_OBJ`** (内部路径（不公开）crossformerattention_kernel.h:5-6). With no KFC
present there is no such claimant, so PB-34's mechanism cannot be asserted here. PB-34 is cited only
as a **related surface** — the same call-site shape (`MatmulImpl` + manual `CrossCore*` in
`MIX_AIC_1_2`) — and its own `verified_does_not_reproduce_on` verdicts for `Ascend950PR_9579` /
`Ascend950PR` (including the full-op GDN witness) stand untouched and were not re-tested.

⚠️ **The difference is NOT the chip, and this entry must not be read as if it were.** This KB's own
canonical-naming glossary (`KB_INDEX.md`) states: *"product **Ascend950PR** (prefill; **Ascend950DT =
decode, same chip**)"* and that the `_NNNN` suffix is a **silicon bin, server-specific**. So
`Ascend950DT_9582` is not a different part from `Ascend950PR_9579` in any sense the KB recognises.
The reproduction difference therefore has to lie on some OTHER axis, and the honest position is that
**which axis is UNRESOLVED**. The candidates, none of them tested here:

  1. **CANN version** — PB-34's full-op no-reproduce witness ran on CANN **9.1.T500**; this run is CANN **9.1.0**.
  2. **Op structure** — GDN's handshake sits inside a different stage topology than this attention op's.
  3. **Silicon bin** — `_9579` vs `_9582`, which the glossary says is server-specific.
  4. **Driver / firmware / host-runtime / device state** — none of which were captured or compared for either
     witness. This is a real axis, not a formality: two hosts can differ here while agreeing on SoC id and CANN tag.

  **This list is NON-EXHAUSTIVE.** It enumerates the differences I can name, not the differences that exist.

Do not "resolve" this by asserting a 950PR/950DT hardware split; the glossary contradicts that, and I
have no measurement that separates the candidates below — and that list is non-exhaustive. What is measured is only: on THIS
configuration, the combination hangs. Anyone applying PB-34's "on A5 the DEFAULT route is a LIGHT PORT
(keep `MatmulImpl<>` + the manual flag chain)" should treat that default as verified on the
configurations PB-34 names, not on every V351 deployment.

- **Severity**: HIGH — identical consequence to PB-34: build PASS, DEBT-482 `RTLD_NOW` load gate PASS, kernel launches, then hangs. No precision sample is ever produced, so this presents as `precision ABSENT`, **not** as a precision mismatch. Classifying it as "accuracy not yet aligned" sends the next engineer to the wrong subsystem.
- **Symptom**: `ACL stream synchronize failed 507014 aicore timeout`; AIC core 18 + AIV cores 36/37 report "timeout or trap error"; stall PCs `0x120040888d08` / `0x12004088a6c0`.
- **What the three post-baseline attempts actually showed** — note they cover only **two distinct flag encodings**, not three: the baseline run used the bare form (`modeId=0`), attempt 1 moved to templated `mode 0x2`, attempt 2 to `mode 4`, and attempt 3 **returned to `mode 0x2`** — so `0x2` was exercised twice and `mode 4` once. Stated per attempt, because the run's own
  `error_signature` summary field says "stall PCs … invariant across 3 flag configurations" and the
  per-attempt records **do not support that**. Only attempt 3 records byte-identity; attempts 1 and 2
  record only that the error code was unchanged, which is a statement about `507014`, not about the PC:
  1. templated `CrossCoreWaitFlag<0x2>` (was bare, i.e. `modeId=0`) → **507014 unchanged** (no PC claim recorded)
  2. `cross_core_sync.md` S4: mode 4 + `id`/`id+16` disjoint ids + pinned pipes → **507014 unchanged**; the run notes S4's own scope is MIX 1:1 + user-`Mmad`, so it did not apply here. The mode-4 PC is recorded only as **shifted after recompile**, which a recompile alone can explain — it is not evidence either way.
  3. revert to the GDN-proven templated mode `0x2` shared ids → 507014 unchanged, **stall PC byte-identical to run 1**
  **Do NOT read this as "the failure is not in the flag encoding."** Three failing configurations cannot
  establish that; they establish that these three did not fix it. The load-bearing evidence is the control
  experiment below, and its scope is this op and this configuration.
- **Control experiment — what it does and does not license**: compiling out **only** the 6 `CrossCore*` calls, leaving **both compute paths intact**, makes all **50/50** cases run to completion with no 507014; case[0] `max_abs_diff = 1.64575`, finite. **What this licenses**: for THIS op in THIS configuration, the hang depends on the hand-written `CrossCore*` chain being present, and both compute paths can terminate without it. **What it does not license**: a claim about the mechanism inside the flag encoding, about other ops, or about other configurations. The 1.64575 is a finite-value observation from a diagnostic build that was reverted — it is **not** a precision result and must never be quoted as one.
- **Mechanism: UNDETERMINED.** What is established is a dependency, not a cause: the hang is present
  with the six hand-written `CrossCore*` calls and absent without them. That is all the control licenses.
  Candidate mechanisms, none tested: (a) something in Layer-1 `MatmulImpl`'s own synchronisation
  interacting with user `CrossCoreSetFlag<0x2>` ids; (b) a **count imbalance** in the flag protocol —
  the kernel documents a symmetric-participation contract where the AIC's single `Wait` clears only
  after BOTH AIV sub-blocks `Set` (PB-55 / OL-190, 内部路径（不公开）crossformerattention_kernel.h:349-353), and the
  AIC and AIV loop bounds are computed independently; (c) something not yet named.
  **Do not fill this in from PB-34.**
- **No fix is recommended here, and Pattern A is explicitly NOT recommended.** The earlier version of
  this entry pointed at PB-34's Pattern A (hand-rolled tile-MMAD keeping the manual chain). That is a
  documented forbidden zone on this silicon: **PB-35 is `confirmed_on: Ascend950PR` that a user-owned
  `Mmad` plus a hand-rolled flag chain DEADLOCKS**, and PB-45 records that a hand-rolled cube stays
  non-deterministic (内部路径（不公开）crossformerattention_kernel.h:7-9). Recommending it would trade a measured hang
  for a documented one. With the mechanism undetermined, this entry prescribes nothing.
- **Surface locator (NON-BLOCKING, not a diagnostic).** PB-34's static grep — same file mentions both `MatmulImpl<` and `CrossCoreSetFlag<0x2` — will also match this kernel, because it matches a **call-site shape**, not a mechanism. Use it only to LOCATE candidate files. It must **not** be used to diagnose, to route, or to prescribe: this entry's mechanism is undetermined, and a match here says nothing about why anything hangs. The runtime signature here is `507014` on a kernel that built and passed the `RTLD_NOW` load gate. **Do not use stall-PC movement as a discriminator.** An earlier draft of this entry did, and it was wrong: only one attempt recorded a byte-identical PC, and a recompile alone can move a PC, so neither movement nor stability separates this bug from another.
- **What is NOT claimed**: this entry does not re-test `Ascend950PR_9579` or `Ascend950PR`, does not touch the GDN full-op no-reproduce witness, and — per the glossary above — does **not** claim a 950PR-vs-950DT hardware distinction. It records exactly one thing: on this configuration (`Ascend950DT_9582`, CANN 9.1.0, this op), the combination reproduces. The axis that explains the divergence is **unresolved**; re-running the GDN light-port on CANN 9.1.0 on this same host is a **discriminator for axis 1 (and partly 2)** — it would not settle axes 3 or 4, and cannot settle an axis nobody has named yet.
- **Cross-reference**: PB-34 (V220 original + the Ascend950PR no-reproduce verdicts this entry deliberately leaves intact); `ascend950pr.md` §249 Cross-core sync; `cross_core_sync.md` S4 (scope: MIX 1:1 + user-`Mmad` — does not cover this case).

<!-- Compatibility anchor for previously published PB-126; canonical ID is PB-75. -->
<a id="PB-126"></a>
<a id="PB-75"></a>
