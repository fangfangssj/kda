---
type: Runbook
title: "matmul::Matmul<> DOES build and run inside the build_ascendc.py pybind harness when the CANN host tiling libs"
description: "`matmul::Matmul<>` 在 build_ascendc.py pybind harness 里能构建能跑（host tiling 库在场时）——边界在 MIX/KFC 不在 matmul 库本身；修正 OL-235 的\"结构性不可构建\"说法。"
tags: [task.debugging, topic.implementation.tiling, topic.compilation, topic.programming_model.execution_unit.cube]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-10-10T00:00:00Z"
platforms: ["950"]
aliases: ["OL-508"]
---

# matmul::Matmul<> DOES build and run inside the build_ascendc.py pybind harness when the CANN host tiling libs

**OL-508** — **`matmul::Matmul<>` DOES build and run inside the `build_ascendc.py` pybind harness when the CANN host tiling libs are present on the build host — the boundary is MIX/KFC, not the matmul library. Drive it with `KERNEL_TYPE_AIC_ONLY` + synchronous `IterateAll<true>` and a kernel-local on-stack `TCubeTiling`.** Scope-corrects OL-235 (whose "structurally unbuildable" is environment-conditional, not structural). Probe: 64×64×64 fp16 vs torch `max_abs 3.814697265625e-06`. But it did NOT win the one op it was ported into: 4-42 CoTAttention 39/50 + GM 0.149592 vs the manual-`Mmad` candidate's 50/50 + 0.256397. ⚠️ That is a **single-op, single-host** result: it says this port was not adopted for 4-42, and it does NOT by itself justify a manual-`Mmad` default for all cube-class ops — that default must rest on P-P102's own scoped evidence. Choose per op and per shape, by measurement. Cross-ref OL-235 (corrected), [`OL-220`](ol_220_mix_ascendc_library_build_recipe.md), [`OL-275`](../precision/ol_275_a_raw_hand_rolled_cube_is_deterministic_alone_but.md), P-P102, [`PB-34`](pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md), [`PB-35`](pb_35_event_t_0_for_cube_internal_pipe_sync_mte1_m_m_fix.md), [`EC-68`](ec_68_a_kernel_that_owns_its_workspace_via_getuserworksp.md), [`OL-91`](ol_91_hardware_accelerator_typed_config_struct_use_the_t.md), P-P68. backend=ascendc.

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=CUBE (any op considering the matmul library instead of manual Mmad); build=build_ascendc.py pybind/ACLRT_LAUNCH harness`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — standalone probe + 4-42 CoTAttention full port, main 2026-09-05 (BUILD rc=0, DEVICE-RUN, full-50 acceptance)`
`unverified_on: soc=Ascend950PR, Ascend910_9382, Ascend310P3 — the host-lib presence test and the MIX fault behaviour are per-host/per-install; re-probe before assuming either half transfers`

- **Category**: build linkage / cube path selection
- **Trigger**: you are about to reject the matmul library for a cube-class op on the authority of OL-235, or you need a cube GEMM and are deciding between manual `Mmad` and `matmul::Matmul<>`.
- **Principle**: OL-235's premise ("the pybind link line is fixed to `kernels torch_npu m dl`, so the host tiling lib can never be there") no longer matches the code, and its conclusion was over-stated as *structural*. Two independent facts break it: (a) the link line is extensible and DEBT-110's baseline capability already appends the CANN host tiling libs **conditionally on those libs existing on the build host**; (b) the library does not need a host-computed `TCubeTiling` at all — a kernel-local on-stack struct works. What actually failed in my probes was `MIX`, which pulls in the KFC client/server rendezvous. `AIC_ONLY` with a synchronous `IterateAll<true>` uses no KFC and runs. **So the decision "manual `Mmad` vs library" is a MEASUREMENT question per op/shape, not a buildability question.**
- **Working recipe (every element was load-bearing; a missing one produced either a build failure or silent garbage)**:
```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);   // NOT MIX — MIX brings KFC
AscendC::SetSysWorkspaceForce(ws);                // FIRST statement (EC-68)
TCubeTiling tiling{};                             // on-stack, runtime shape (OL-91)
MatmulImpl<MTA, MTB, MTC, MTC, CFG_NORM> mm;
mm.Init(&tiling, &pipe);                          // NON-const TCubeTiling* — see trap below
mm.SetOrgShape(M, pitch, K, K, pitch);            // argument mapping — see trap below
mm.SetTensorA(gA, false); mm.SetTensorB(gB, false);
mm.SetSingleShape(M, N, K);
mm.template IterateAll<true>(gC, 0, false, false, false);   // SYNC (PB-34: never async here)
mm.End();
```
  Also allocate a `TBuf`/`LocalTensor` so the static checker sees computation — it does not recognise `IterateAll` as compute (P-P68).
- **Trap 1 — `SetOrgShape` argument mapping (builds clean, computes garbage)**: per 内部路径（不公开）matmul.h:92-96 (CANN vendor header, not in this repo), `orgN` = **B's N-axis**, `orgKa` = A's K-axis, `orgKb` = **B's K-axis**, `orgKc` = C's N-axis; the 3-arg overload expands to `SetOrgShape(M, N, K, K, N)`. **A row pitch belongs in `orgN`/`orgKc`, NEVER in `orgKb`.** Wrong `SetOrgShape(M, N, K, pitch, pitch)` → builds clean, output ~uncorrelated. Single-variable bisect (pitch-only, everything else fixed): `max_abs 43.75 → 5.7e-06`; slab offsets were never the problem. In the ported op this one mapping was the whole difference between 1/50 and 39/50.
- **Trap 2 — `Init` overload ambiguity**: `Init(nullptr, &pipe)` is **ambiguous** (`matmul_client.h` takes `const TCubeTiling*`, `matmul_server_aux.h` takes non-const `TCubeTiling*`); casting to `const TCubeTiling*` then gives **no matching function**, because the template instantiates the server side too. Pass a **non-const** `TCubeTiling*`.
- **What this does NOT claim**: it does not say the library is faster for a given op, and it does not say MIX is unusable everywhere. In 4-42 CoTAttention the fixed-layout port was worse on BOTH gates than the hand-written manual-`Mmad` candidate — precision 39/50 vs 50/50 under the selected profile, and full-50 event-domain GM 0.149592 vs 0.256397. Per-GEMM the picture was mixed, which is the actionable part: mm1 16.41 → 9.65 ms (−6.76), but mm2 +7.88, mm3 +4.08, mm4 +10.92. Routing only mm1 to the library still lost on the full denominator (39/50, GM 0.250848). **Route selection must be measured per shape; do not adopt the library op-wide on the strength of one GEMM — and symmetrically, do not read this one failed port as evidence for a manual-`Mmad` default across cube-class ops. It is one op on one host; the standing default keeps whatever independent basis P-P102 gives it, which I did not re-verify.**
- **Evidence**: main 2026-09-05, Ascend950DT_9582 / CANN 9.1.0. Standalone probe `matmul::Matmul<> RAN. max_abs 3.814697265625e-06 mean_abs 3.5367730788493645e-07` (`BUILD_RC=0`, verdict `CORRECT`) — main lane `main_op42_fix_20260905/` (not in this repo): run receipt `MMRUN5.log` sha256 `8c348c8744395778`, driver `_mmrun5.sh` sha256 `228476c5da8a2c8b`, probe source `scripts/mmprobe.cpp` sha256 `87ee8cdcfd0f0d3e` (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` :19, on-stack `TCubeTiling tiling{}` :38, `IterateAll<true>` :58) and binding `scripts/mmprobe_bind.cpp` sha256 `7a4590cf3f92e40c`. ⚠️ Those hashes are of the **local staging copies**; the container copy at run time (2026-09-05 04:53Z) was never hashed, so byte-identity between the two is **not established** — the run's own log is the receipt, not the disk match. Repo-HEAD `4b8941b` source checks: `build_ascendc.py:307,356`; `build_capabilities/baseline_cann_host_tiling.py:41-45` + docstring. Production witness on the same host: Triton op-29 `flashattentionbwd_kernel.h`. Full paired record: 4-42 `TERMINAL_42_PAIRED.md`.
- **Other instances (predicted)**: any cube-class op whose lane has a full CANN host install rather than a clean-runtime/stub lane. The corollary is the general one — **a KB entry that says "impossible" should name the ENVIRONMENT CONDITION that makes it so, otherwise the next reader inherits a wall that no longer exists.**
- **Cross-ref**: OL-235 (scope-corrected by this entry), [`OL-220`](ol_220_mix_ascendc_library_build_recipe.md) (the `ascendc_library` flow where host tiling links), [`OL-275`](../precision/ol_275_a_raw_hand_rolled_cube_is_deterministic_alone_but.md) (cube determinism / managed lifecycle), P-P102 (manual-Mmad cube scaffold — still the default), [`PB-34`](pb_34_matmulimpl_with_manual_crosscoresetflag_waitflag_m.md) (sync `IterateAll`), [`PB-35`](pb_35_event_t_0_for_cube_internal_pipe_sync_mte1_m_m_fix.md), [`EC-68`](ec_68_a_kernel_that_owns_its_workspace_via_getuserworksp.md) (`SetSysWorkspaceForce` first), [`OL-91`](ol_91_hardware_accelerator_typed_config_struct_use_the_t.md) (runtime tiling struct), P-P68 (static checker), `CAND-KFC-standalone-bootstrap-teardown`. backend=ascendc.

## 补充证据（2026-10-10 合并自 `matmul_builds_in_pybind_harness_with_host_tiling_libs.md`）

> 原卡已按社区去重要求合并入本卡并删除；以下为该卡正文全文（verbatim）。

## OL-508 — OL-508: A5Ops additive extension


## 适用场景

- 适用平台与范围：soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=CUBE (any op considering the matmul library instead of manual Mmad); build=build_ascendc.py pybind/ACLRT_LAUNCH harness。
- 验证来源：soc=Ascend950DT_9582; cann=9.1.0 — standalone probe + 4-42 CoTAttention full port, main 2026-09-05 (BUILD rc=0, DEVICE-RUN, full-50 acceptance)。

**OL-508** — **`matmul::Matmul<>` DOES build and run inside the `build_ascendc.py` pybind harness when the CANN host tiling libs are present on the build host — the boundary is MIX/KFC, not the matmul library. Drive it with `KERNEL_TYPE_AIC_ONLY` + synchronous `IterateAll<true>` and a kernel-local on-stack `TCubeTiling`.** Scope-corrects OL-235 (whose "structurally unbuildable" is environment-conditional, not structural). Probe: 64×64×64 fp16 vs torch `max_abs 3.814697265625e-06`. But it did NOT win the one op it was ported into: 4-42 CoTAttention 39/50 + GM 0.149592 vs the manual-`Mmad` candidate's 50/50 + 0.256397. ⚠️ That is a **single-op, single-host** result: it says this port was not adopted for 4-42, and it does NOT by itself justify a manual-`Mmad` default for all cube-class ops — that default must rest on P-P102's own scoped evidence. Choose per op and per shape, by measurement. Cross-ref [OL-235](launch_lib_without_cann_host_tiling_lib_matmul_tcubetiling.md) (corrected), OL-220, OL-275, P-P102, PB-34, PB-35, EC-68, OL-91, P-P68. backend=ascendc.

- Former A5Ops ID: `OL-508`
- Fusion disposition: `additive_new`

`applies_to: soc=Ascend950DT_9582 (dav_3510); cann=9.1.0; bisheng=n/a; op_class=CUBE (any op considering the matmul library instead of manual Mmad); build=build_ascendc.py pybind/ACLRT_LAUNCH harness`
`verified_on: soc=Ascend950DT_9582; cann=9.1.0 — standalone probe + 4-42 CoTAttention full port, main 2026-09-05 (BUILD rc=0, DEVICE-RUN, full-50 acceptance)`
`unverified_on: soc=Ascend950PR, Ascend910_9382, Ascend310P3 — the host-lib presence test and the MIX fault behaviour are per-host/per-install; re-probe before assuming either half transfers`

- **Category**: build linkage / cube path selection
- **Trigger**: you are about to reject the matmul library for a cube-class op on the authority of OL-235, or you need a cube GEMM and are deciding between manual `Mmad` and `matmul::Matmul<>`.
- **Principle**: OL-235's premise ("the pybind link line is fixed to `kernels torch_npu m dl`, so the host tiling lib can never be there") no longer matches the code, and its conclusion was over-stated as *structural*. Two independent facts break it: (a) the link line is extensible and DEBT-110's baseline capability already appends the CANN host tiling libs **conditionally on those libs existing on the build host**; (b) the library does not need a host-computed `TCubeTiling` at all — a kernel-local on-stack struct works. What actually failed in my probes was `MIX`, which pulls in the KFC client/server rendezvous. `AIC_ONLY` with a synchronous `IterateAll<true>` uses no KFC and runs. **So the decision "manual `Mmad` vs library" is a MEASUREMENT question per op/shape, not a buildability question.**
- **Working recipe (every element was load-bearing; a missing one produced either a build failure or silent garbage)**:
```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);   // NOT MIX — MIX brings KFC
AscendC::SetSysWorkspaceForce(ws);                // FIRST statement (EC-68)
TCubeTiling tiling{};                             // on-stack, runtime shape (OL-91)
MatmulImpl<MTA, MTB, MTC, MTC, CFG_NORM> mm;
mm.Init(&tiling, &pipe);                          // NON-const TCubeTiling* — see trap below
mm.SetOrgShape(M, pitch, K, K, pitch);            // argument mapping — see trap below
mm.SetTensorA(gA, false); mm.SetTensorB(gB, false);
mm.SetSingleShape(M, N, K);
mm.template IterateAll<true>(gC, 0, false, false, false);   // SYNC (PB-34: never async here)
mm.End();
```
  Also allocate a `TBuf`/`LocalTensor` so the static checker sees computation — it does not recognise `IterateAll` as compute (P-P68).
- **Trap 1 — `SetOrgShape` argument mapping (builds clean, computes garbage)**: per 内部路径（不公开）matmul.h:92-96 (CANN vendor header, not in this repo), `orgN` = **B's N-axis**, `orgKa` = A's K-axis, `orgKb` = **B's K-axis**, `orgKc` = C's N-axis; the 3-arg overload expands to `SetOrgShape(M, N, K, K, N)`. **A row pitch belongs in `orgN`/`orgKc`, NEVER in `orgKb`.** Wrong `SetOrgShape(M, N, K, pitch, pitch)` → builds clean, output ~uncorrelated. Single-variable bisect (pitch-only, everything else fixed): `max_abs 43.75 → 5.7e-06`; slab offsets were never the problem. In the ported op this one mapping was the whole difference between 1/50 and 39/50.
- **Trap 2 — `Init` overload ambiguity**: `Init(nullptr, &pipe)` is **ambiguous** (`matmul_client.h` takes `const TCubeTiling*`, `matmul_server_aux.h` takes non-const `TCubeTiling*`); casting to `const TCubeTiling*` then gives **no matching function**, because the template instantiates the server side too. Pass a **non-const** `TCubeTiling*`.
- **What this does NOT claim**: it does not say the library is faster for a given op, and it does not say MIX is unusable everywhere. In 4-42 CoTAttention the fixed-layout port was worse on BOTH gates than the hand-written manual-`Mmad` candidate — precision 39/50 vs 50/50 under the selected profile, and full-50 event-domain GM 0.149592 vs 0.256397. Per-GEMM the picture was mixed, which is the actionable part: mm1 16.41 → 9.65 ms (−6.76), but mm2 +7.88, mm3 +4.08, mm4 +10.92. Routing only mm1 to the library still lost on the full denominator (39/50, GM 0.250848). **Route selection must be measured per shape; do not adopt the library op-wide on the strength of one GEMM — and symmetrically, do not read this one failed port as evidence for a manual-`Mmad` default across cube-class ops. It is one op on one host; the standing default keeps whatever independent basis P-P102 gives it, which I did not re-verify.**
- **Evidence**: main 2026-09-05, Ascend950DT_9582 / CANN 9.1.0. Standalone probe `matmul::Matmul<> RAN. max_abs 3.814697265625e-06 mean_abs 3.5367730788493645e-07` (`BUILD_RC=0`, verdict `CORRECT`) — main lane `main_op42_fix_20260905/` (not in this repo): run receipt `MMRUN5.log` sha256 `8c348c8744395778`, driver `_mmrun5.sh` sha256 `228476c5da8a2c8b`, probe source `scripts/mmprobe.cpp` sha256 `87ee8cdcfd0f0d3e` (`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY)` :19, on-stack `TCubeTiling tiling{}` :38, `IterateAll<true>` :58) and binding `scripts/mmprobe_bind.cpp` sha256 `7a4590cf3f92e40c`. ⚠️ Those hashes are of the **local staging copies**; the container copy at run time (2026-09-05 04:53Z) was never hashed, so byte-identity between the two is **not established** — the run's own log is the receipt, not the disk match. Repo-HEAD `4b8941b` source checks: `build_ascendc.py:307,356`; `build_capabilities/baseline_cann_host_tiling.py:41-45` + docstring. Production witness on the same host: Triton op-29 `flashattentionbwd_kernel.h`. Full paired record: 4-42 `TERMINAL_42_PAIRED.md`.
- **Other instances (predicted)**: any cube-class op whose lane has a full CANN host install rather than a clean-runtime/stub lane. The corollary is the general one — **a KB entry that says "impossible" should name the ENVIRONMENT CONDITION that makes it so, otherwise the next reader inherits a wall that no longer exists.**
- **Cross-ref**: OL-235 (scope-corrected by this entry), OL-220 (the `ascendc_library` flow where host tiling links), OL-275 (cube determinism / managed lifecycle), P-P102 (manual-Mmad cube scaffold — still the default), PB-34 (sync `IterateAll`), PB-35, EC-68 (`SetSysWorkspaceForce` first), OL-91 (runtime tiling struct), P-P68 (static checker), `CAND-KFC-standalone-bootstrap-teardown`. backend=ascendc.
