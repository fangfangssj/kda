---
type: Runbook
title: "`REGIST_MATMUL_OBJ` lands NO-SYNC branch with single `-DASCENDC_MATMUL_AICORE` — runtime MPU 507015 once cube uses the matmul object [V220+V351, ALL_MODES, build-config + KFC-sync-activation]"
description: "applies_to: soc=Ascend910_9382 (V220 A2/A3) + Ascend950PR_9579 (V351 A5); cann=9.0.0+; bisheng=ascendc.cmake DYNAMIC_MODE; op_class=mixed_aic_aiv_with_REGIST_MATMUL_OBJ; macro=KERNEL_TYPE_MIX_AIC_1_2"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-57-regist-matmul-obj-lands-no-sync-branch-with-single.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:00Z"
platforms: [a3, "950"]
aliases: ["EC-57", "REGIST_MATMUL_OBJ"]
---

# `REGIST_MATMUL_OBJ` lands NO-SYNC branch with single `-DASCENDC_MATMUL_AICORE` — runtime MPU 507015 once cube uses the matmul object [V220+V351, ALL_MODES, build-config + KFC-sync-activation]

> 原 OKF v1 卡号：**EC-57**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Mixed cube+vec kernel uses REGIST_MATMUL_OBJ(...) to instantiate the MatmulImpl<> library. Project sets a single global -DASCENDC_MATMUL_AICORE define (the "sim

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A2 系列、Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_9382 (V220 A2/A3) + Ascend950PR_9579 (V351 A5); cann=9.0.0+; bisheng=ascendc.cmake DYNAMIC_MODE; op_class=mixed_aic_aiv_with_REGIST_MATMUL_OBJ; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: a5_ops:3_FusionAttention 2026-05-21T19:53Z (independent CANN-source read of kfc_register_obj.h L262/322/368, confirmed empirically); also reproduced on V351 by probe_a5_v300_fa_sync 2026-05-23 — DEBT-20 per-source-define gap is cross-arch`

- **Severity**: HIGH (build + register-binary both succeed, kernel launches, AIC then traps with MPU error 507015 on first matmul usage — symptom looks like an MPU bug but is actually a missing-define misconfiguration).
- **Symptom**: Mixed cube+vec kernel uses `REGIST_MATMUL_OBJ(...)` to instantiate the `MatmulImpl<>` library. Project sets a single global `-DASCENDC_MATMUL_AICORE` define (the "simple" DEBT-20 approach). Build succeeds. `aclrtlaunch_<op>(...)` returns clean. AIC stage traps with `aicore exception 507015` immediately when cube enters `mm.IterateAll` / `GetTensorC`. No AIV-side error.
- **Root cause** (per independent CANN-source read of `kfc_register_obj.h:262/322/368`):
  - L368: `REGIST_MATMUL_OBJ` macro expansion checks `#if defined(SPLIT_CORE_CUBE)` → if undefined, takes the **NO-SYNC branch** which just runs `InitCurObj(...)` — no KfcServer init on AIC, no KfcCommClient on AIV, no mailbox topology.
  - L262: real KFC server init (the AIC side: `KfcServer.Init() + while-isRun loop`) is gated on `#if defined(SPLIT_CORE_CUBE) && !defined(SPLIT_CORE_VEC)`.
  - L322: real KFC client init (the AIV side: `KfcCommClient + CrossCoreWaitFlag` mailbox poll) is gated on `#if !defined(SPLIT_CORE_CUBE) && defined(SPLIT_CORE_VEC)`.
  - Single global `-DASCENDC_MATMUL_AICORE` doesn't satisfy any of L262/L322 — both passes fall through to L368 NO-SYNC. Cube launches matmul state machine with no server to handshake with → MPU traps when matmul tries to wait for AIV consumer ack.
- **Fix** — per-compile-pass defines:
  - **AIC pass** must receive `-DSPLIT_CORE_CUBE=1` (and `-DASCENDC_MATMUL_AICORE` as before).
  - **AIV pass** must receive `-DSPLIT_CORE_VEC=1` (and `-DASCENDC_MATMUL_AICORE` as before).
  - In CMake (cmake ≥ 3.18 required): `set_source_files_properties(<file>.cpp PROPERTIES TARGET_DIRECTORY aic_obj COMPILE_DEFINITIONS "SPLIT_CORE_CUBE=1")` + symmetric for `aiv_obj` + `SPLIT_CORE_VEC=1`. Apply per kernel source file.
  - `build_ascendc.py` schema extension (landed `66a4d985`): `per_source_defines` now accepts per-pass dict form `{"global": [...], "aic": [...], "aiv": [...]}` in addition to legacy flat list form.
- **Detection** (pre-build static check): if kernel file uses `REGIST_MATMUL_OBJ(...)` AND `build_overrides.json` declares only flat `per_source_defines` (no `aic` / `aiv` keys) AND no `-DSPLIT_CORE_*` in global compile flags → guaranteed runtime 507015 fault.
- **Evidence**:
  - 3_FusionAttention 2026-05-21 (independent prototype ar-2 KFC dispatch crash): AIC 507015 with REGIST_MATMUL_OBJ + single `-DASCENDC_MATMUL_AICORE`. Direct CANN-source read confirmed L262/322/368 branch logic.
  - probe_a5_v300_fa_sync 2026-05-23 (main A5 probe attempt 2): same Pattern B build pattern reproduces the L368 NO-SYNC fall-through on V351 — DEBT-20 per-source-define plumbing applies cross-arch (V220 + V351 both need per-pass defines).
- **Cross-ref**: PB-34 (Pattern A V220 sync conflict — different failure mode in same MIX_AIC_1_2 mode), OL-176 (matmul_intf.h non-transitive include — sibling DEBT-20 family), DEBT-20.1 implementation commit `66a4d985`.

<!-- 迁移自 porter kb/target/ascendc/（EC-57，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
