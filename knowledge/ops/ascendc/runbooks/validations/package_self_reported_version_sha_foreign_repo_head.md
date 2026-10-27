---
type: Runbook
title: "A package's self-reported version SHA can be a different repository's HEAD — tilelang/version.py appends the CWD's git HEAD at import with no cwd="
description: "A package's self-reported version SHA can be a different repository's HEAD — tilelang/version.py appends the CWD's git HEAD at import with no cwd="
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-299"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-299 — OL-299: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all; cann=all; op_class=all。
- 验证来源：MEASURED 2026-07-30 (scan, host-only, no NPU) -- tilelang/version.py read directly: cwd=<real checkout> -> 0.1.4+<that checkout's HEAD>; cwd=/tmp (non-git) -> bare 0.1.4; cwd=<freshly git-init'd DECOY repo> -> 0.1.4+<the DECOY's HEAD>. Dirty-state control: decoy made dirty (untracked AND tracked-modified) still reported its SHA.。

**OL-299** — **A package's self-reported version SHA can be a DIFFERENT repository's HEAD.** `tilelang/version.py` reads a **static `VERSION`** (`:27-28` → `0.1.4`) then appends a git SHA at import (`:46-47`) via `subprocess.check_output(['git','rev-parse','HEAD'], stderr=DEVNULL)` with **no `cwd=`** (`:31-42`) — so the SHA is **the CWD's git HEAD**, not the package's (`:9-16` computes the package dir for `VERSION` and never passes it to git). **MEASURED (scan, host-only)**: real checkout → its own HEAD; `/tmp` → bare `0.1.4`; **freshly `git init`'d DECOY repo → the DECOY's HEAD**, byte-indistinguishable from a genuine string. **⚠️ CORRECTED 2026-07-30 — the original "dirty tree ⇒ SHA disappears" mechanism is REFUTED, not narrowed**: holding CWD fixed and varying only dirtiness (untracked **and** tracked-modified) the SHA is **always retained** — dirtiness is irrelevant. Third mode: `git` absent from `PATH` raises an **uncaught `FileNotFoundError`** (`except subprocess.SubprocessError` does not catch it). ⇒ absent / **foreign** / crash, and **they are not equally bad — a foreign SHA is false information wearing the format of evidence**, inviting none of the suspicion a missing suffix invites. Hard form of the missed direction: **a SHA present is not evidence the SHA belongs to this package.** Classification: a **third pattern** — static `VERSION` + runtime CWD-derived HEAD, neither pure-static nor setuptools-scm; the clue was **no `.dirty` suffix**. Also records that the earlier "counterexample" was never one (two people measured from different CWDs ⇒ **two readings don't conflict until you establish they refer to the same object**) and that the candidate-cause list is now moot. Practice (**unchanged across both corrections, never depended on the mechanism**): record **base commit + diff + the diff's own hash**, resolved **relative to the package** (`git -C <pkg_dir>`), not the caller's CWD.

- Former A5Ops ID: `OL-299`
- Fusion disposition: `additive_new`

`applies_to: soc=all; cann=all; op_class=all`
`verified_on: MEASURED 2026-07-30 (scan, host-only, no NPU) -- tilelang/version.py read directly: cwd=<real checkout> -> 0.1.4+<that checkout's HEAD>; cwd=/tmp (non-git) -> bare 0.1.4; cwd=<freshly git-init'd DECOY repo> -> 0.1.4+<the DECOY's HEAD>. Dirty-state control: decoy made dirty (untracked AND tracked-modified) still reported its SHA.`
`superseded_mechanism: the original "dirty tree => SHA disappears" mechanism is REFUTED, not merely narrowed -- dirtiness is IRRELEVANT. See the CORRECTION block before citing this entry.`
**The mechanism** (source-verified, `tilelang/version.py`): `__version__` is read from a
**static `VERSION` file** (`:27-28` → `0.1.4`), then a git SHA is **appended at import
time** (`:46-47`) from `get_git_commit_id()` (`:31-42`), which runs
`subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL, ...)`
with **no `cwd=` argument**. The subprocess therefore inherits **the caller's working
directory**, and the reported SHA is *whatever git repository the importing process happens
to be standing in*.
The file makes the omission stark: `:9-16` carefully computes `current_dir`,
`develop_project_root_dir` and `installed_project_root_dir` **to locate its own `VERSION`
file** — and then passes none of them to the git call. The package knows where it lives and
does not use that knowledge for provenance.
**The two bad outcomes, in order of severity**:
1. **Absent** (imported from a non-git directory) — `0.1.4` is not false, it is *less
   specific*, and less-specific reads as fine in a log. A reviewer matches it to the tag and
   concludes the artefact is the release. Silent, because `stderr` is discarded.
2. **Foreign** (imported from any other git repository) — **strictly worse.** The string is
   precise, well-formed, and names a commit that has nothing to do with the artefact. It
   invites no question at all. **Provenance that is confidently wrong beats provenance that is
   missing, every time, in the wrong direction.**
**The replacement, and it is cheap** — and note it is now *better-founded* than when it was
first written, because the artefact's self-report can name **a repository you were merely
standing in**: record the **base commit** plus the **diff**, and carry the **diff's own hash**
(e.g. `sha256` of the patch file) next to it. The hash gives the appendix an identity, so a
reproduction attempt can verify that the pasted patch is the patch that was built instead of
trusting the paste to be faithful. Record the base commit even when the tree is clean, so the
provenance record has the same shape in both cases and its absence is visible. Capture it
**from the package's own directory** (`git -C <pkg_dir> rev-parse HEAD`), never from whatever
directory the process happens to be in.
> ⚠️ **CORRECTED 2026-07-30 — the MECHANISM was MISIDENTIFIED, not merely overstated. The
> real one is WORSE. The PRACTICE below is unchanged and is now better-founded than when it
> was written.**
>
> **What this entry originally said.** The heading read "Self-certifying provenance fails
> precisely in the scenario that needs it — a version string carrying its commit SHA
> **degrades to a bare version once the tree is dirty**", and the body asserted the mechanism
> "is generic to every scheme that derives identity from `git describe` …, **so this is not a
> property of one project**." Stated as a rule, in the present tense, from **one** observation.
>
> **What is actually true.**
>
> | | original claim | actual mechanism |
> |---|---|---|
> | what the SHA depends on | whether the working tree is **dirty** | the **CWD at import time** |
> | relation to dirtiness | causal | **none — dirtiness is irrelevant** |
> | failure modes | SHA **absent** ⇒ loss of information | SHA **absent** (imported from a non-git dir) **or FOREIGN** (imported from any other git repo, reporting *that* repo's HEAD) **or a hard import crash** |
> | severity | information is lost | **false information wearing the format of evidence** |
>
> **MEASURED by scan, host-only, no NPU** — `tilelang/version.py` executed directly with only
> CWD varied (same interpreter, same file):
>
> ```
> cwd=<real tilelang-ascend checkout>   -> 0.1.4+d1288c265e73a2215993a4dd75cb8568987f682f   (= that checkout's HEAD)
> cwd=/tmp                (not a git repo) -> 0.1.4                                          (bare)
> cwd=<DECOY repo, git init + 1 commit>  -> 0.1.4+1738ba3fa7ba740bec435bf22e9ffc6f2a8b073f   (= the DECOY's HEAD)
> ```
>
> **The decoy is the decisive test**: a freshly `git init`'d repository with one empty commit,
> unrelated to the package in every way, and the package **reports that repository's HEAD as
> its own provenance — in a string byte-indistinguishable from a genuine one.**
>
> **Dirty-state control (this is what refutes the original claim).** Holding CWD fixed inside
> the decoy repo and varying *only* dirtiness — the experiment the original mechanism was
> about, and which was never run:
>
> | decoy repo state | reported version |
> |---|---|
> | clean | `0.1.4+1738ba3f…` |
> | dirty — **untracked** file present | `0.1.4+1738ba3f…` (**unchanged**) |
> | dirty — **tracked file modified** | `0.1.4+705b1755…` (its new HEAD; **still present**) |
>
> ⇒ **A dirty tree does not drop the SHA, in either flavour of dirty.** The original mechanism
> is **refuted**, not narrowed.
>
> **A third failure mode, also measured**: `get_git_commit_id()` catches only
> `subprocess.SubprocessError`, and `issubclass(FileNotFoundError, subprocess.SubprocessError)`
> is **False**. So with `git` absent from `PATH` the import raises an **uncaught
> `FileNotFoundError`** — verified. The `stderr=subprocess.DEVNULL` hides git's
> "not a git repository" message, which is why the *absent* case is silent, but it does not
> make the *missing-binary* case safe. ⇒ absent / foreign / crash.
>
> **⇒ The load-bearing point: the two failure modes are NOT equally bad.** An **absent** SHA
> loses information — `0.1.4` is less specific, and a reader may notice the imprecision. A
> **foreign** SHA is **false information in the exact format of evidence**: it is precise,
> well-formed, plausible, and wrong. It does not invite the question that a missing suffix
> invites. Provenance that is confidently incorrect is worse than provenance that is missing.
>
> **The earlier "counterexample" was never a counterexample.** An intermediate revision of this
> entry recorded a peer's second-host reading (`0.1.4+38e6b38f…`, dirty tree, SHA retained) as a
> *counterexample* to the dirty-tree mechanism, and downgraded the mechanism to
> "host/config-dependent, trigger unidentified". That framing was wrong too: **two people
> measured from different working directories.** Neither reading was about the property being
> claimed, so they never actually conflicted. ⇒ **Two readings do not conflict until you have
> established that they refer to the same object.** Recorded in-entry because the disagreement
> consumed a full review cycle and a PR, and the resolution was not "who measured correctly"
> but "what were you each measuring."
>
> **Candidate causes now MOOT — listed so nobody re-derives them.** The same intermediate
> revision enumerated candidate triggers: git version, setuptools-scm version and config,
> sdist/wheel vs checkout build, packaging cache state, tracked-modified vs untracked edits.
> **All were candidates for a mechanism that does not exist.** Not silently deleted, because
> the list is a plausible thing to reconstruct from the original claim; anyone who does should
> stop here. (Note the last of them — tracked-vs-untracked — was actually *tested* above and
> makes no difference.)
>
> **Classification corrected — it is a THIRD pattern, not one of the usual two.** This is
> neither a pure-static version nor setuptools-scm. It is **a static `VERSION` file plus a
> runtime-appended, CWD-derived git HEAD**. The natural assumption is the two-way
> static-vs-scm split, and it is wrong here. **The clue that pointed at it: no `.dirty`
> suffix.** setuptools-scm marks a dirty tree (`+d<date>` / `.dirty`); a string that appends a
> bare SHA and never marks dirtiness is not scm-derived at all — which also means it was never
> *capable* of expressing the dirty-tree behaviour the original entry attributed to it.
>
> **Kept from the earlier revision** (both still stand): `38e6b38f…` is a real
> `xLLM-AI/tilelang-ascend` commit pinned in our own tree since 2026-07-19 —
> `docs/design/ROADMAP.md:146` (DEBT-225), `docs/design/ROUTE_A_LAYER_RESPONSIBILITY_DESIGN.md:41`
> / `:371` / `:398` / `:413-415` — so that SHA was not garbled or invented; **that is all it
> establishes.** And the peer's patch-presence check carried a **positive control of 26
> `set/wait_flag` occurrences** alongside `if (src == dst)` count **1**, new-comment **1**,
> old-comment **0**, `pipe_barrier(PIPE_ALL)` in both `.so` files — the control is what makes
> the count of 1 a measurement rather than an empty grep (OL-297's point).
>
> ### ⚠️ The direction the original entry MISSED — now in its hard, measured form
>
> The original guarded only against the SHA **disappearing**. The earlier revision upgraded
> that to "a SHA present is not evidence the tree was clean." **The decoy test makes it
> strictly stronger:**
>
> **A version string containing a SHA is NOT evidence that the SHA belongs to this package.**
>
> It may be the HEAD of a repository you were merely standing in. Anyone reasoning *"it reports
> a SHA, therefore it is a build of that commit"* can be wrong about **which repository the
> commit is even from** — not just about whether the tree was clean. This is no longer
> speculation from an asymmetry; it is demonstrated by construction.
>
> ### Meta — two errors, one week apart, in the entry that now carries the rule
>
> **First**: a single-point observation was amplified into a general mechanism, in the same
> breath as praising the observation. What was not licensed was "once the tree is dirty"
> (universal) and "not a property of one project" (explicitly asserting generality) from
> **n = 1**, with no second host and no attempt to vary conditions.
>
> **Second, and this one is about the correction rather than the claim**: a PR was opened to
> narrow a **merged** entry on the strength of **one** report — moving faster than
> verification, inside the very entry that had just gained the rule "before writing a mechanism
> in the present tense, ask what n is." **The narrowing was itself premature.** The correct
> first step was not to downgrade the mechanism but to establish **whether the two readings
> referred to the same object** — which is exactly what the CWD finding eventually did, and it
> replaced the mechanism outright rather than weakening it.
>
> ⇒ **The conclusion survived both times; the stated reason did not survive either time.** Same
> family as OL-300's METHOD section (a faithful observation generalised past what it licenses)
> and OL-297 (a record that reads the same in the good and the bad case carries no
> information). ⇒ **Before writing a mechanism in the present tense, ask what n is. Before
> "reconciling" two readings, ask whether they measure the same thing.**
**Rule** (unchanged across two corrections — it never depended on the mechanism at all):
never let a build's self-report be the sole provenance of a measurement. Provenance =
(base commit, diff, hash of the diff), recorded by the harness that ran the measurement,
resolved **relative to the package**, not to the caller's CWD.
That independence is the point worth noticing: the mechanism behind this entry was first
overstated and then **replaced outright**, and the recommendation needed no revision either
time — because it says **stop relying on the version string**, which holds whether the string
drops the SHA, keeps it, reports a stranger's SHA, or crashes the import. ⇒ **A
recommendation that does not depend on the mechanism it was motivated by survives the
mechanism being wrong.** Prefer that shape when writing KB advice.
**Cross-ref**: OL-297 (a record that reads the same in the good and the bad case carries
no information — the same defect one level up, applied to provenance instead of to a
test), CAND-PP110 (build-provenance finalize gates must enumerate all legitimate build
paths).
