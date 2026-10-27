---
type: Runbook
title: "`.ascendc_env` `A5_CANN_PATH` pointed at the toolkit symlink-root (only `latest` + `set_env.sh` symlinks, no `tools/tikcpp`) → cmake `ascendc_kernel_cmake does not exist` — point it at the complete `.../ascend-toolkit/latest`"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.implementation.pipeline, topic.data_movement, topic.performance.analysis, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-75-ascendc-env-a5-cann-path-pointed-at-the-toolkit-sy.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-75", "A5_CANN_PATH", "latest", "set_env.sh"]
---

# `.ascendc_env` `A5_CANN_PATH` pointed at the toolkit symlink-root (only `latest` + `set_env.sh` symlinks, no `tools/tikcpp`) → cmake `ascendc_kernel_cmake does not exist` — point it at the complete `.../ascend-toolkit/latest`

> 原 OKF v1 卡号：**EC-75**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - build_ascendc.py (verify-only pybind/ACLRT_LAUNCH build) fails at cmake configure with a missing-directory error naming ascendc_kernel_cmake (e.g. ascendc_kerne

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: `build_ascendc.py` (verify-only pybind/ACLRT_LAUNCH build) fails at cmake configure with a missing-directory error naming `ascendc_kernel_cmake` (e.g. `ascendc_kernel_cmake does not exist` / `add_subdirectory given source ".../ascendc_kernel_cmake" which is not an existing directory`). The build does not even reach a compile step.

**Root cause**: the `A5_CANN_PATH` value in `workspace/.ascendc_env` was set to a CANN *symlink-root* directory that contains only the `latest` symlink and `set_env.sh` — NOT the actual toolkit tree. `build_ascendc.py` resolves the AscendC cmake module relative to `A5_CANN_PATH` (expects `tools/tikcpp/ascendc_kernel_cmake` under it), but a symlink-root has no `tools/`, so the path resolves to a non-existent directory.

**Fix**: set `A5_CANN_PATH` to the **complete toolkit directory** `.../ascend-toolkit/latest` (the resolved toolkit, which actually contains `tools/tikcpp/ascendc_kernel_cmake`), not the parent symlink-root. Verify with:
```bash
ls "$A5_CANN_PATH/tools/tikcpp/ascendc_kernel_cmake" || echo "EC-75: A5_CANN_PATH is not a complete toolkit"
```

**Distinct from EC-51**: EC-51 is `ASCEND_CANN_PACKAGE_PATH` unset → `find_package(ASC)` fails in the on-host ops-nn-build pipeline. EC-75 is the *workspace verify-build* `A5_CANN_PATH` config var pointing at the wrong (symlink-root) directory. Same `ascendc_kernel_cmake` token can appear in the trace, but the misconfigured variable and the fix are independent.

**Evidence**: iou_v2 kw-1 (2026-06-21, port_a3_to_a5, A5 Ascend950PR): `A5_CANN_PATH` at the toolkit symlink-root → cmake `ascendc_kernel_cmake does not exist`; repointing to `.../ascend-toolkit/latest` resolved it. (Same session also corrected two non-cmake `.ascendc_env`/lane setup issues: `A5_DEPLOY_STAGE_HOST/_CONTAINER` held host-IP/container-name where directory paths belong → silent empty deploy; and lane6 `BENCHMARK_ROOT` was missing `utils/build_ascendc.py` → `setup_lanes.sh` should copy `utils/` when provisioning a lane.)

**Other instances (predicted)**: any fresh A5 host bring-up or new lane where `.ascendc_env` is hand-filled — `A5_CANN_PATH` is the most error-prone field because the symlink-root and the resolved toolkit dir look interchangeable but only the latter has `tools/tikcpp`. Add the `ls .../tools/tikcpp/ascendc_kernel_cmake` check to lane/host preflight.

**Cross-reference**: EC-51 (different `ascendc_kernel_cmake`/ASC cmake failure — ops-nn-build pipeline, `ASCEND_CANN_PACKAGE_PATH`), OL-234 (build-only A5 host needs a complete-CANN runtime container — adjacent host-setup gotcha).

<!-- 迁移自 porter kb/target/ascendc/（EC-75，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
