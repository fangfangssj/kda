---
type: Runbook
title: "A piped build-verify chain grades the artifact it thinks it built"
description: "A piped build-verify chain grades the artifact it thinks it built"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-548"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-548 — OL-548: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (host-side shell harness concern, silicon-independent); cann=all; bisheng=n/a; op_class=any worker/orchestrator build+verify loop。
- 验证来源：21_Scatter 310p (2026-08-15) + FA-A5 pse (2026-06-02, OL-206 evidence — same class, different trigger)。

**OL-548** — **a piped build-verify chain grades the artifact it THINKS it built**: `deploy ... \

- Former A5Ops ID: `OL-548`
- Fusion disposition: `additive_new`

`applies_to: soc=all (host-side shell harness concern, silicon-independent); cann=all; bisheng=n/a; op_class=any worker/orchestrator build+verify loop`
`verified_on: 21_Scatter 310p (2026-08-15) + FA-A5 pse (2026-06-02, OL-206 evidence — same class, different trigger)`

**Principle**: in a chain like `deploy ... | tail -3 && cp ... && verify`, the `&&` keys on the exit of the PIPE TAIL (`tail`), not on the deploy. A deploy that fails mid-pipe still yields tail exit 0, the chain proceeds, and verification runs against the STALE previous artifact. The verdict is then a confident, well-formed PASS — for a binary that no longer corresponds to the edited source. This is the verification-integrity class of the FA stale-`.so` incident (OL-206 evidence: "a stale deployed .so spawns an entire chain of conclusions built on void data") with a NEW trigger: not a partial docker-cp, but shell pipe exit-code masking.

**Decision rule**: any build→verify chain must (a) run under `set -o pipefail` so a failed build stage fails the chain, (b) explicitly assert the artifact's identity before verifying — the `.so` mtime postdates the source edit (timestamp assert), or stronger, the compiled-source md5 matches the working tree (the FA incident's unblock) — and (c) pin WHICH artifact location and WHICH evaluator variant produced the verdict: the graders' import path (`workspace/{op}/kernel/build`) is not the deploy build path (`vendor/AscendOpGenAgent/current_task/kernel/build`), and md5-distinct copies of the canonical evaluator exist in-tree (archived per-op copies predate per-model input isolation); a build tree created by copying another tree (e.g. `cp -a`) can keep generated build configuration (here `_autogen_cmake`) that names the original tree's absolute paths, so its build can return rc=0 while compiling and linking into the original tree — for every copy-derived tree, bind the check to where the artifact landed and what else changed: before configuring, confirm no generated build file names the original tree and record the original tree's artifact hash; after building, confirm the new artifact differs from the original tree's artifact and the original artifact's hash is unchanged. Absent (a)+(b)+(c), a full-PASS count recorded after an edit is UNPROVEN, not verified — and so is a full-FAIL.

**Failure signature (21_Scatter)**: rebuild chain `deploy ... | tail -3 && cp && verify` returned 47/47 PASS on a kernel whose source had just been edited — the deploy had FAILED and the verify graded the OLD binary (still computing the pre-fix `x+(a+b)` association, OL-547). Caught only because the canonical seeded evaluator + `probes/delta_bitprobe.py` showed ours == old-binary semantics; an unseeded lucky draw had let the stale binary pass.

**Evidence**:
- 21_Scatter 310p (2026-08-15): 内部路径（不公开）workspace/21_Scatter/failures_ledger.md:7 (iter 4). Fix: `set -o pipefail` + explicit `.so` timestamp assert on every build+verify chain; re-verified on the fresh binary (Pass A 47/47 OVERALL_T2).
- 21_Scatter kw-2 310p (2026-08-15, two NEW triggers of the same class — decision rule (c)): (i) artifact-LOCATION mismatch: `deploy_to_npu_lane.sh` builds into `vendor/AscendOpGenAgent/current_task/kernel/build` while runner-based graders import the extension from `workspace/{op}/kernel/build` — grading without copying the fresh `.so` over silently measures the PREVIOUS binary (first pass_a/canonical numbers were the 03:08 binary's); (ii) script-VARIANT staleness: an archived copy of `precision_eval_two_tier.py` under `output/*/src/kernels/` predates the per-model `copy.deepcopy(raw)` isolation, so for an IN-PLACE reference (`scatter_` mutates its input) it double-scatters the shared input and returns a guaranteed 47/47 FAIL for ANY kernel. Both caught before acting on the numbers — a 47-FAIL artifact could otherwise have triggered a pointless kernel rewrite. Always invoke `src/scripts/precision_eval_two_tier.py`, never an archived copy.
- FA-A5 pse (2026-06-02): same class, docker-cp-partial-update trigger — recorded in OL-206's evidence; the md5-of-compiled-source check was the unblock.
- 4_GroupedMatmulSwigluQuant (2026-09-07; SoC and CANN not stated in the record — a NEW trigger of the same class, decision rule (c)): a new ablation arm made with `cp -a <base arm> <new arm>`, its kernel sources then edited and configured with `cmake -S <new arm>/.../build/_autogen_cmake -B <new arm>/.../build`, built with rc=0 — but it compiled the BASE arm's sources and linked the `.so` into the BASE arm's build directory, because `build/_autogen_cmake` contains absolute paths to the tree it was generated in and `cp -a` duplicates them verbatim. Two failures at once: the new arm's edited sources were never compiled (its `.so` remained the inherited copy), and the base arm's binary, whose hash is cited in receipts, was silently overwritten. A first attempt had failed loudly with `CMakeCache.txt directory is different …`; clearing the cache to get past it made the build "succeed" into the wrong tree, which is worse than the visible failure. The trap fires only when an arm inherits an unrewritten `_autogen_cmake`: the `novec2` arm, whose autogen had been regenerated to refer to itself, built correctly — so check each arm's autogen individually rather than assuming either way. Caught by the build wrapper, which refused to continue when the produced `.so` was identical to the base `.so` (failure 1), and by re-hashing the baseline afterwards (failure 2); the baseline was restored byte-for-byte from an existing run archive and re-verified against the hash in the published receipt. Checks for every arm build, in order: before configuring, no file under the arm's `_autogen_cmake` still names the base path, and the base `.so` hash is recorded; after building, the arm's `.so` exists and differs from the base `.so`, and the base `.so` hash is unchanged (both after-build checks are needed: a rebuild that is not byte-reproducible (OL-505) makes the overwritten base differ from the inherited copy). The record prescribes rewriting the inherited absolute paths before configuring as the fix, but does not record that fix being applied and verified; the observed correct build is `novec2`, whose autogen was regenerated. Evidence (agent-team-wiki @d813788c): `projects/op4_4/FINDING_20260907T1735Z_DS_COPIED_ARM_BUILDS_INTO_ITS_BASE.md`.

**Other instances (predicted)**: any `cmd | grep/tail/head && next` harness chain around builds; scp/docker-cp deploy steps followed by verify; benchmark loops that re-time a stale `.so` after a failed rebuild; piped log-filter chains in CI. Near-relative: scripted text-replacement edits that silently no-op (target string drifted) and surface only one build later as `use of undeclared identifier` — verify the edit APPLIED (grep -c the inserted symbol) before building (21_Scatter iter 3, 内部路径（不公开）workspace/21_Scatter/failures_ledger.md:6).

**Cross-reference**: OL-206 (stale-`.so` conclusions chain), OL-122 (file-CONTENT signal, not exists-and-nonempty, for build-status wait loops — the same verify-the-thing-itself family), OL-160 (the verification-fraud class this feeds), OL-547 (the precision fix the stale binary masked), OL-509 (a `build_overrides.json` define can silently fail to reach the device compile while the build returns rc=0 and the DSO is byte-identical — a different trigger).
