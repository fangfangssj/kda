---
type: Runbook
title: "A reassuring green/empty/zero must prove its scope covers the claim — positive control and report-your-standpoint mandatory"
description: "A reassuring green/empty/zero must prove its scope covers the claim — positive control and report-your-standpoint mandatory"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-321"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-321 — OL-321: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (verification methodology, not silicon); cann=all; bisheng=n/a; op_class=any — this is a criterion-level rule。
- 验证来源：2026-08-01 — eight distinct carriers, each an actual occurrence, across five agents in one day。

**OL-321** — a reassuring green/empty/zero must prove its scope covers the claim; 9 carriers + discriminators; positive control + report-your-standpoint mandatory

- Former A5Ops ID: `OL-321`
- Fusion disposition: `additive_new`

`applies_to: soc=all (verification methodology, not silicon); cann=all; bisheng=n/a; op_class=any — this is a criterion-level rule`
`verified_on: 2026-08-01 — eight distinct carriers, each an actual occurrence, across five agents in one day`

**Principle**: when a check's scope is NARROWER than the claim drawn from it, its output is
**byte-identical to a genuine pass**. So the first question about any green / empty / zero result is
not "what does it say" but **"what did it actually measure, and does that cover the sentence I am
about to write?"**

**Why it resists self-review**: the failure mode is not "the command errored". It is *the command
succeeded and the output looks like an answer*. Nothing will flag it, because flagging it requires
something that already knows the right answer — which is the thing being sought.

**Carriers observed (each one real, with its discriminator)**:

| carrier | what the false result looks like | discriminator |
|---|---|---|
| exit code not propagated | stage2 FAIL, script still `exit 0` | make the subject fail once; see if the code changes |
| vacuous universal over an empty set | `git log --branches --not --remotes` = 0 with 0 local branches | confirm the set is non-empty first |
| stale baseline | `ahead=3` from a local `main` 22 commits behind | `fetch` first; `branch -d` compares HEAD, not `origin/main` |
| repo-root detection failing | whole tree `NO_SEED`, identical to genuinely unseeded | tool must FAIL LOUD, never degrade to a normal verdict |
| over-specific literal search | grep with a date → 0 hits → read as "absent" | prove the pattern can hit, then narrow |
| check narrower than conclusion | `git log @{u}..HEAD` = 0 ⇒ "the whole directory is pushed" | say WHICH scope the zero belongs to |
| the invisible layer | `git status` hides ignored ⇒ 37G reported "clean" | `--ignored`; ref-based tools cannot see non-git dirs |
| gate reached ≠ work completed | "only PyYAML needed to reach O2.5" ⇒ "control plane needs only PyYAML", while CPU-truth bundling needs torch | push to the TERMINAL state before concluding scope |
| pipe eats the exit code | `cmd | head; echo $?` returns `head`'s status | read `$?` without a pipeline |

**Two mandatory actions**:
1. **Positive control** — prove the check CAN produce the other verdict. An always-green gate and a
   genuinely passing gate look identical; so do an always-rejecting gate and a correctly-rejecting one.
2. **Report the standpoint** — any triage/scan tool must emit *where it ran from and which paths it
   followed*. On 2026-08-01 that field is the only reason two agents who independently obtained the
   same false reading located the fault in their own invocation within ten minutes, instead of filing
   a bug against a colleague's correct fix.

⚠️ **Most expensive variant**: a false reading used to ACCUSE someone. Verify the instrument before
trusting any reading that points at a peer.

**Two refinements contributed by back, 2026-08-01, both from checking its own work:**

**(a) A BORROWED CHECKLIST is not coverage.** back merged `main` into a stale branch and
reported *"verified item by item what you asked me to preserve"*, listing 4 items. The
`check_merge_would_delete` tool showed **8 classes** had actually been at risk — the 4 named plus
EC-88, two CAND entries, three DEBT rows, and an entire file (`workspace_triage.py`). back had
checked exactly what my message named. ⇒ **Someone else's list is what THEY saw, not what is
complete.** The conclusion happened to be right — the merge really did repair all 8 — but that was
the merge's doing, not the verification's. ⭐ Adopting another party's enumeration as your own
verification scope is a specific, nameable mechanism for a check narrower than its claim.

**(b) ZERO HITS IS `UNDETERMINED`, NOT `NO` — until a control says otherwise.** Both of us hit
this while handling the other's report of it: back searched 129 refs through `head -40` and
concluded "on no remote ref"; I grepped `merge-would-delete` against a title containing
`check_merge_would_delete` and concluded "the PR was never created". Neither of us was careless;
**an empty result is simply not evidence of absence until the probe has been shown able to produce
a hit.** ⇒ Treat every `0` as pending a positive control, the same way an always-green gate is
pending one.

⭐ **Corollary on being right for the wrong reason**: back nearly reported "your tool is not on
main" from a `--diff-filter=A` probe that returned 0 for a KNOWN-present control file. That
statement was **true** — the tool genuinely was not on main — but the probe backing it was broken.
A broken instrument that lands on the correct answer is worse than one that is visibly wrong,
because it earns confidence it did not deserve and will be reused.

**Always-loaded short form**: `shared/ALWAYS_LOADED_RULES.md` §1 (this is the canonical entry).
