---
type: Runbook
title: "A3→A5 port — PR4778 ship artifacts (`<op>_apt.cpp`, `arch35/<op>.h`) placed under `kernel/` instead of `op_kernel/` → `build_ascendc.py` glob picks them up and compile fails with `unknown type name '<Op>TilingData'` [V351, port_a3_to_a5, build-layout]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5"
tags: [task.debugging, topic.precision, topic.implementation.pipeline, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-54-a3-a5-port-pr4778-ship-artifacts-op-apt-cpp-arch35.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3, "950"]
aliases: ["EC-54", "build_ascendc.py"]
---

# A3→A5 port — PR4778 ship artifacts (`<op>_apt.cpp`, `arch35/<op>.h`) placed under `kernel/` instead of `op_kernel/` → `build_ascendc.py` glob picks them up and compile fails with `unknown type name '<Op>TilingData'` [V351, port_a3_to_a5, build-layout]

> 原 OKF v1 卡号：**EC-54**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - workspace-side build_ascendc.py verify-only build (the per-op pybind/ACLRT_LAUNCH_KERNEL build, NOT the on-host CANN ops-nn-build pipeline) fails with:

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: workspace-side `build_ascendc.py` verify-only build (the per-op pybind/ACLRT_LAUNCH_KERNEL build, NOT the on-host CANN ops-nn-build pipeline) fails with:
```
kernel/arch35/<op>.h:NN:NN: error: unknown type name '<Op>TilingData'
kernel/<op>_apt.cpp:NN:NN: error: use of undeclared identifier 'tilingData'
```

**Root cause**: `build_ascendc.py` globs `workspace/<op>/kernel/*.cpp` for compilation. The `<Op>TilingData` struct + `GET_TILING_DATA` macro are emitted ONLY by the CANN ops-nn-build pipeline (`build.sh --pkg --ops=<op> --soc=ascend950`) — they do not exist at `build_ascendc.py` time. When the worker places PR4778 ship artifacts (`<op>_apt.cpp` and/or `arch35/<op>.h`) under `kernel/` instead of `op_kernel/`, the verify-build glob picks them up and tries to compile them outside the auto-tiling pipeline → struct + macro are unresolved → compile error.

This is distinct from EC-52 (`<op>_apt.cpp` mistakenly `#include`s `<op>_tiling.h`). EC-52 = include resolution bug; EC-54 = layout misplacement bug. Both surface the same `unknown type name '<Op>TilingData'` token but the fix paths are independent.

**Fix**: enforce PB-33 layout split. `kernel/` is verify-only (pybind11.cpp + `<op>_kernels.cpp` + `<op>_kernel.h` — what `build_ascendc.py` compiles); `op_kernel/` is ship-only (`arch35/<op>.h` + `<op>_apt.cpp` — what `build.sh --pkg --ops=<op>` compiles).

```
workspace/<op>/
  kernel/                 ← build_ascendc.py glob picks these up
    pybind11.cpp
    <op>_kernel.h
    <op>_kernels.cpp
  op_kernel/              ← ops-nn-build pipeline picks these up
    arch35/<op>.h
    <op>_apt.cpp
  op_host/                ← ops-nn-build pipeline picks these up
    <op>_def.cpp, <op>_tiling.cpp, <op>_tiling.h, CMakeLists.txt, config/ascend950/...
```

Mechanical fix when misplacement is detected:
```bash
mv workspace/<op>/kernel/arch35/        workspace/<op>/op_kernel/arch35/
mv workspace/<op>/kernel/<op>_apt.cpp   workspace/<op>/op_kernel/<op>_apt.cpp
```

**Detection signature** (pre-build audit):
```bash
# Should ONLY see pybind11.cpp + <op>_kernel.h + <op>_kernels.cpp under kernel/
ls workspace/<op>/kernel/*.cpp 2>/dev/null | grep -E '_apt\.cpp$' && echo "EC-54: apt.cpp misplaced under kernel/"
[ -d workspace/<op>/kernel/arch35 ] && echo "EC-54: arch35/ misplaced under kernel/"
```

**Evidence**:
- fatrelu_mul kw-1 port_a3_to_a5 (2026-05-17): worker initially emitted `kernel/arch35/fatrelu_mul.h` + `kernel/fatrelu_mul_apt.cpp` + `kernel/fatrelu_mul_kernels.cpp`. Build globbed all three .cpp files; apt.cpp compile failed with `unknown type name 'FatreluMulTilingData'`. Fix was `mv` to `op_kernel/`. After move, the kernel/ glob only saw `fatrelu_mul_kernels.cpp` + `pybind11.cpp` and build proceeded cleanly. Iter-1 build PASS after layout correction → 8/8 T1_BIT_EXACT precision.

**Other instances (predicted)**: every port_a3_to_a5 op that emits BOTH a pybind/ACLRT_LAUNCH_KERNEL verify path AND PR4778 ship artifacts. Especially likely when the worker generates apt.cpp + arch35/ from scratch without consulting PB-33 layout.

**Mitigation candidates** (out of scope for this entry — would require harness change):
- (a) `build_ascendc.py` exclude-by-name pattern: explicitly skip `*_apt.cpp` and `arch35/` subdir (already excludes `pybind11.cpp`).
- (b) `finalize_pipeline` pre-build-gate: warn if `workspace/<op>/kernel/arch35/` exists or `kernel/<op>_apt.cpp` exists, before invoking `build_ascendc.py`.

**Cross-reference**:
- PB-33 — archive layout contract (kernel/ vs op_kernel/ split); EC-54 is the build-time failure mode when the contract is violated at workspace level
- EC-52 — different cause for the same `unknown type name '<Op>TilingData'` token (defensive `#include` of `<op>_tiling.h` from a correctly-placed apt.cpp)
- OL-141 — target `op_kernel/arch35/` is advisory layout evidence, not a body mirror source
- OL-160 — canonical entry-point names (`kernel/pybind11.cpp` + `kernel/<op>_kernels.cpp` are the verify-path canonical names; misplacing ship artifacts under `kernel/` violates this naming invariant)

<!-- 迁移自 porter kb/target/ascendc/（EC-54，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
