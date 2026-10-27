---
type: Runbook
title: "A per-source build_overrides.json define can silently fail to reach the device compile: the build still returns rc=0 and the emitted DSO is byte-identical, so every number then measured describes the old code path"
description: "A per-source build_overrides.json define can silently fail to reach the device compile: the build still returns rc=0 and the emitted DSO is byte-identical, so every number then measured describes the old code path"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-509"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-509 — OL-509: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any; build=build_ascendc.py with per_source_defines in build_overrides.json。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention, main 2026-09-05 (byte-identical DSO + invalid-field rc=0 control)。

**OL-509** — **a per-source `build_overrides.json` define can silently fail to reach the device compile: the build still returns rc=0 and the emitted DSO is BYTE-IDENTICAL, so every number you then measure describes the OLD code path.** I reported a complete set of results — correctness, per-stage timings, a full-denominator GM and a bandwidth explanation — for an `#ifdef` that was never compiled. Detection ladder (weakest → strongest): knob sweep whose values agree to 4-5 sig figs → DSO sha unchanged across a source edit → inject an invalid statement in the block and see rc=0 → **plant `#error <SENTINEL>` inside the `#ifdef`**. Workaround: a plain `#define` in the `.cpp` above the include. Cross-ref [OL-507](../validations/build_time_identity_cannot_be_reconstructed_afterwards.md), [OL-235](launch_lib_without_cann_host_tiling_lib_matmul_tcubetiling.md)/[OL-508](ol_508_matmul_matmul_does_build_and_run_inside_the_buil.md). backend=ascendc.

- Former A5Ops ID: `OL-509`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any; build=build_ascendc.py with per_source_defines in build_overrides.json`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — 4-42 CoTAttention, main 2026-09-05 (byte-identical DSO + invalid-field rc=0 control)`
`unverified_on: other build_overrides forms and other harness versions — this records ONE case, see the scope caveat below; do not generalise to all per-source defines`

- **Category**: build system / instrument integrity
- **Trigger**: you added a define through `build_overrides.json` (`per_source_defines`, or the per-pass `{"aic": [...]}` form) to switch on an `#ifdef` code path, and you are about to measure that path.
- **Principle**: the failure is silent and *convincing*. The build succeeds, the artifact is produced, the harness runs, and numbers come out — they are simply the numbers of the code you did not change. Nothing in the log says the define was dropped. **A build that succeeds is not evidence that your code was compiled.** Before attributing ANY measurement to a newly-gated path, prove the gate is open with a control that fails loudly when it is closed.
- **Concrete anchor**: adding `{"per_source_defines": {"cotattention_cube_kernels.cpp": ["COTATTN_GEMM_LIB"]}}` — and the per-pass `{"aic": [...]}` form — did not activate the macro for that file's compile. It survived my own review because the copied task dir happened to carry a tiling constant whose manual timing (53.12 ms) sat within noise of the "library" result (53.14 ms), so the two arms agreed and looked mutually confirming.
- **Detection, in increasing order of strength**:
  1. a knob sweep whose values agree to 4-5 significant figures (suspicious, not proof);
  2. **DSO sha byte-identical across a source edit** — strong that the ARTIFACT
     did not change, but it does NOT by itself distinguish "the edit never
     reached the compiler" from "it reached the compiler and was optimised
     away". Both produce an identical artifact. Decide activation by an actual
     compile diagnostic (step 4), not by the hash;
  3. inject a deliberately invalid statement inside the block — e.g. assign a struct field that does not exist — and build: **rc=0 means that code is not compiled**;
  4. cheapest and unambiguous: plant `#error <SENTINEL>` inside the `#ifdef` — only an actual RECOMPILE of that source that emits the specified `#ifdef` sentinel diagnostic proves the define is active; **no diagnostic proves nothing** — the source may simply not have been recompiled (cached object), so a clean build is not evidence either way. Check the log for that source's compile line before reading anything into a clean build.
- **Mitigation**: put a plain `#define` in the `.cpp` above the include. Verified to work in this case.
- **Scope caveat (do NOT over-read)**: this records that the override did not enter the compile **in this case**. The `global` form emits `set_source_files_properties(... PROPERTIES ...)` with no `TARGET_DIRECTORY`, so it would only apply in the calling directory scope while the device `aic_obj` sub-target lives in another sub-build — that is a PLAUSIBLE mechanism, **not a verified one**, and it does not explain why the `aic` form also failed. The mechanism is UNRESOLVED; only the detection ladder and the workaround are established.
- **Evidence**: main 2026-09-05, 4-42 CoTAttention, Ascend950DT_9582 / CANN 9.1.0. Discriminator chain that caught it: identical per-stage timings → identical DSO sha → `rc=0` after injecting `tiling.reserved_dummy_probe = 0;` (a field that does not exist).
- **Other instances (predicted)**: any conditional-compilation experiment routed through a build-config indirection rather than the source file — per-source defines, per-pass defines, cmake `add_definitions` in a sub-build, and toolchain-file flags. Generalisation: **when an experiment's independent variable travels through a config file, add a control that makes the build FAIL when the variable did not arrive.**
- **Cross-ref**: OL-507 (build-time identity cannot be reconstructed afterwards), OL-508/OL-235 (same lane), P-P68. backend=ascendc.
