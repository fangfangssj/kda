---
type: Runbook
title: "DEBT-091(c) compiled_provenance stem-check fails on the ACLRT_LAUNCH_KERNEL pybind path because the build wraps the kernel source as auto_gen_<name>.cpp"
description: "DEBT-091(c) compiled_provenance stem-check fails on the ACLRT_LAUNCH_KERNEL pybind path because the build wraps the kernel source as auto_gen_<name>.cpp"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-540"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-540 — OL-540: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all (harness/port_a3 DEBT-091(c) finalize provenance gate — chip-independent; the build wrapper is toolchain-level, not silicon); cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5。
- 验证来源：soc=Ascend310P3 (dav-m200, CANN 9.0.0)。

**OL-540** — DEBT-091(c) `compiled_provenance` stem-check fails on the ACLRT_LAUNCH_KERNEL pybind path because the build wraps the kernel source as `auto_gen_<name>.cpp`** — the real object is `auto_gen_<source>.cpp.o`, NOT `<source>.cpp.o`. A worker that sets `compiled_provenance.source` to the raw workspace filename (exactly what `kw_brief_pa3_phases.py:172` literally shows) makes the gate compute `auto_gen_… != <op>_kernels.cpp+'.o'` → **stem broken** → `binary_provenance` FAIL → rollback, kernel correct + built. Fix: set `source`/`object`/md5 to the auto_gen WRAPPER name + its md5 (NOT the workspace source's — the wrapper adds code). Discrimination vs OL-404 (different gate) + the other DEBT-091 legs (md5/stale-.o) in the body. Anchor `port_a3/__init__.py:538` + `:562`. Evidence: ge_glu_grad_v2 310p kw-3. **Brief gap flagged**: brief should warn about the auto_gen rename. backend=ascendc.

- Former A5Ops ID: `OL-540`
- Fusion disposition: `additive_new`

<!-- applies_to_backend: ascendc -->
`applies_to: soc=all (harness/port_a3 DEBT-091(c) finalize provenance gate — chip-independent; the build wrapper is toolchain-level, not silicon); cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5`
`verified_on: soc=Ascend310P3 (dav-m200, CANN 9.0.0)`
`unverified_on: soc=Ascend950PR (A5) / Ascend910_220x (A3) — the same _check_binary_provenance gate and the same vendored ACLRT_LAUNCH_KERNEL build wrapper serve port_a3 on those SoCs, so the principle transfers; the specific stem-mismatch was observed only on the 310P P140-pybind port so far. Re-confirm the wrapper-naming on A5/A3 before relying on the exact filename.`

**Principle**: The DEBT-091(c) `compiled_provenance` gate proves "the binary the runtime dispatches is exactly what we built" by validating a build CHAIN, not a basename+md5 equality (compiled bytes ≠ source bytes by construction). One leg of that chain is the **stem check**: `stem(object) == source + ".o"`. On the P140 pybind / `ACLRT_LAUNCH_KERNEL` path the build toolchain does NOT compile the workspace source directly — it generates a compilation WRAPPER named `auto_gen_<source>.cpp` (the wrapper `#include`s the workspace source, adds the `ACLRT_LAUNCH_KERNEL` device entry points, and compiles to `auto_gen_<source>.cpp.o`). So the real object on disk is `auto_gen_<source>.cpp.o`, NOT `<source>.cpp.o`. A worker that sets `compiled_provenance.source` to the raw workspace filename (`<op>_kernels.cpp`) — which is exactly what the worker-brief example literally shows — makes the stem check compute `auto_gen_<op>_kernels.cpp.o != <op>_kernels.cpp + ".o"` → **stem broken** → the `binary_provenance` finalize gate FAILs → rollback, even though the kernel itself is correct and fully built. Fix: set `source` to the ACTUAL compiled filename (`auto_gen_<source>`) and record the auto_gen wrapper's md5 as the source digest (NOT the workspace source's md5 — the wrapper adds code, so its md5 differs).

**Concrete anchor** (resolving citations):
- gate reads the block — `src/scripts/orchestrator/plugins/port_a3/__init__.py:538` (`cp = build_ev.get("compiled_provenance")`)
- the stem check that fails — `src/scripts/orchestrator/plugins/port_a3/__init__.py:562` (`stem broken: object {obj} != source {src}+'.o'`)
- the worker-brief instruction that is INCOMPLETE on this point — it shows `source=<op>_kernels.cpp`, `object=<source>.o`, and states the gate checks `stem(object==source+.o)`, but does NOT warn that the ACLRT_LAUNCH_KERNEL wrapper renames the compiled file to `auto_gen_<source>` — `src/scripts/orchestrator/briefs/kw_brief_pa3_phases.py:172`
- the rename is performed by the vendored cmake-based ACLRT_LAUNCH_KERNEL build wrapper (under `vendor/AscendOpGenAgent/`); confirmed empirically by this op's build output, which materializes `auto_gen_ge_glu_grad_v2_kernels.cpp` + `auto_gen_ge_glu_grad_v2_kernels.cpp.o` (observed in `workspace/ge_glu_grad_v2/kernel/build/auto_gen/kernels/`).

**Discrimination** (vs the sibling DEBT-091 / finalize-provenance failure classes — pick the branch by the gate message):
- THIS entry — gate message reads `stem broken: object 'auto_gen_<op>_kernels.cpp.o' != source '<op>_kernels.cpp'+'.o'`: the `source` field names the workspace file instead of the auto_gen wrapper. Fix: set source/object/md5 to the auto_gen (compiled) name + the wrapper's md5.
- OL-404 — a DIFFERENT gate (`phase_o5_mismatch`, `tier1_pass=0, total=0`): the worker never emitted the capture artifacts. Different layer (precision-grader input), not binary provenance.
- the OTHER legs of THIS same gate — `md5s not all 32-hex` (placeholder digests), `deploy!=workspace src md5`, `built_from!=deploy src md5` (stale `.o`): fix those by emitting real 32-hex digests of the right files.
