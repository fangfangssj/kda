---
type: Runbook
title: "Per-kernel KERNEL_TASK_TYPE(name, TYPE) is silently ignored by the auto_gen kernel build; only the file-level KERNEL_TASK_TYPE_DEFAULT reaches the generated engine config"
description: "Per-kernel KERNEL_TASK_TYPE(name, TYPE) is silently ignored by the auto_gen kernel build; only the file-level KERNEL_TASK_TYPE_DEFAULT reaches the generated engine config"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-516"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-516 — OL-516: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any kernel translation unit built through the ascendc auto_gen cmake flow (kernels_precompile / kernels_preprocess / auto_gen); build=any。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — 4-46 DAModule, main 2026-09-06, by direct comparison of the GENERATED build config across three builds of the same tree。

**OL-516** — **Per-kernel `KERNEL_TASK_TYPE(name, TYPE)` is silently ignored by the auto_gen kernel build; only the file-level `KERNEL_TASK_TYPE_DEFAULT` reaches the generated engine config.** The per-kernel form compiles, links, changes the DSO hash and produces a clean `rc=0` — and changes nothing about which engine the kernel is launched on.

- Former A5Ops ID: `OL-516`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=any kernel translation unit built through the ascendc auto_gen cmake flow (kernels_precompile / kernels_preprocess / auto_gen); build=any`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — 4-46 DAModule, main 2026-09-06, by direct comparison of the GENERATED build config across three builds of the same tree`
`unverified_on: other CANN versions and other build flows (a hand-written CMakeLists that does not go through auto_gen was not tested); whether a later toolkit honours the per-kernel form`

- **Category**: build system / silent no-op
- **Trigger**: a translation unit holding a mix of single-engine kernels, where you want each declared as AIV_ONLY or AIC_ONLY instead of inheriting one file-wide MIX default.
- **What was observed**: adding `KERNEL_TASK_TYPE(<name>, KERNEL_TYPE_AIV_ONLY)` / `..._AIC_ONLY` for 12 of 14 kernels left `build/auto_gen/kernels/aiv_config.cmake` **byte-identical** to the build without them — `MIX_SOURCES` still holding the generated source, `AIV_SOURCES` still empty, and every kernel still carrying `auto_gen_<name>_kernel=<name>_N_mix_aiv` with `__MIX_CORE_MACRO__=1`. The config was regenerated during that build (its mtime falls inside the build window), so this is not a stale artifact.
- **The positive control that makes it a finding rather than a guess**: changing the **file-level** `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` to `KERNEL_TYPE_AIV_ONLY` in the same tree flipped the same generated file decisively — `MIX_SOURCES` empty, `AIV_SOURCES` populated, and every `_mix_aiv` / `__MIX_CORE_MACRO__` definition gone. Same macro family, same build, same file: the mechanism works, the per-kernel key does not reach it.
- **⚠️ The obvious control does NOT work here**: planting a deliberately invalid value (`KERNEL_TASK_TYPE(k, KERNEL_TYPE_BOGUS_XYZ)`) and expecting a compile error proves nothing. That step's `CXX_FLAGS` contain **`-E`**, so **that stage performs no C++ name resolution** and an undeclared identifier cannot fail it there; this says nothing about other stages of the build. A control that cannot fail is not a control. Judge by the **generated config**, which is the artifact the rest of the build consumes: `46_DAModule/kernel/build/auto_gen/kernels/aiv_config.cmake`.
- **Why it bites**: the intended use is per-launch cost. A MIX launch brings up both engines for every kernel, including ones whose entire body on one engine is `if ASCEND_IS_AIC { return; }`. Believing the per-kernel declaration took effect turns a *no-op* into an apparent *refutation* — you measure "no change", and conclude the engine cost is not the problem, when in fact you never changed the engine.
- **What to do instead**: split the kernels into separate translation units, each with its own file-level `KERNEL_TASK_TYPE_DEFAULT`. This is the remedy **EC-42** already prescribes for a neighbouring symptom (mixing AIV and AIC kernels in one `.cpp` mis-registers all but the first); what this entry adds is *why annotating them in place does not work* — the per-kernel macro never reaches the generated config at all.
- **One shortcut was tried and yielded no measurement**: a build of this unit carrying `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` while cube kernels were still present faulted on **5 of 5** cases — `npuSynchronizeDevice ... device error type 3, error code 507035` — producing zero stage timings. Whatever the cause, the attempt returns `PROBE_ERROR` and can never serve as a cheap refutation of a per-launch engine-cost hypothesis.
- **⚠️ Do NOT read that fault as 'AIV_ONLY faults'.** That build also carried a second change inherited from the previous arm — the four pure-AIV kernels were launched with `blockDim` 48 instead of 24 — so the fault is **not attributable to the task-type declaration alone**, and **this entry claims no cause for it**. Isolating it needs a single-variable rebuild that was not run.
- **One narrow thing a neighbouring lane's measurement does settle.** 507035 is `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`, and a UB-budget overflow is a known producer of it on this part, so "the doubled block count blew the UB budget" is the obvious story. An AIV_ONLY lane on the same SoC raised `blockDim` from 1-8 to as much as 64 with **per-core UB accounting unchanged to the byte** and passed 49/49, because UB is allocated **per core** — so raising the block count does not *by itself* raise per-core UB. That is all it establishes. It is an AIV_ONLY lane and does not extrapolate to a MIX resource account, so it does **not** exonerate the block-count change in this MIX build, where the two variables were mixed. **No cause is claimed and none is excluded**; isolating it needs a single-variable rebuild that was not run.
- **Also distinct from the file's own header comment**, which describes *omitting* the macro entirely as "clean rc, outputs at allocation values" (PB-43 class). Omitted and explicitly-AIV_ONLY are different inputs; only the second was built here, and it did not reproduce that description.
- **Cross-ref**: EC-42 (same remedy, adjacent symptom: one `.cpp` mixing AIV and AIC kernels mis-registers all but the first); PB-44 (its 107000 register-FAIL was observed under CANN **9.0.0** and did not reproduce in a 9.1.T500 A/B — that is a scoped observation, NOT a claim that 9.1.x is generally safe); OL-506 (a declaration with no consumer is inert, and its own build success cannot tell you so). backend=ascendc.
