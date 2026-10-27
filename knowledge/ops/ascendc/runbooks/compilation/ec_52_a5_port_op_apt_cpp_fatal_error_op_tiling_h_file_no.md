---
type: Runbook
title: "A5 port `<op>_apt.cpp` `fatal error: '<op>_tiling.h' file not found` — must not `#include` the host-side tiling header"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5"
tags: [task.debugging, topic.implementation.pipeline, topic.data_movement.data_copy, topic.data_movement, topic.tiling, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-52-a5-port-op-apt-cpp-fatal-error-op-tiling-h-file-no.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a3, "950"]
aliases: ["EC-52"]
---

# A5 port `<op>_apt.cpp` `fatal error: '<op>_tiling.h' file not found` — must not `#include` the host-side tiling header

> 原 OKF v1 卡号：**EC-52**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - opc compile of A5 port's op_kernel/<op>_apt.cpp fails with fatal error: '<op>_tiling.h' file not found even though op_host/<op>_tiling.h exists in the source tr

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Symptom**: opc compile of A5 port's `op_kernel/<op>_apt.cpp` fails with `fatal error: '<op>_tiling.h' file not found` even though `op_host/<op>_tiling.h` exists in the source tree.

**Root cause**: `op_host/` is NOT in the kernel-side compile include path. The tiling-data struct definition (e.g. `FatreluMulTilingData`) is auto-injected into the kernel translation unit by the build pipeline's tiling-data header generation — driven by the `BEGIN_TILING_DATA_DEF` / `REGISTER_TILING_DATA_CLASS` macros in `op_host/<op>_tiling.h`, processed host-side, then re-emitted as a synthesized header that `GET_TILING_DATA()` resolves at kernel compile time. An explicit `#include "<op>_tiling.h"` from the apt.cpp searches the wrong path and trips the compile.

**Fix**: remove `#include "<op>_tiling.h"` (or any include of `op_host/<op>_tiling.h`) from the `<op>_apt.cpp`. The `GET_TILING_DATA(<varname>, tiling)` macro inside `Process()` resolves the struct through the auto-injected header — no manual include needed.

**Reference convention**: the A3 upstream `op_kernel/<op>.cpp` (e.g. `cann/ops-nn/activation/fatrelu_mul/op_kernel/fatrelu_mul.cpp`) also does NOT include `op_host/<op>_tiling.h`. The A5 `<op>_apt.cpp` should follow the same convention — adding the include "defensively" is a worker-side mistake.

**Evidence**: fatrelu_mul kw-1 port_a3_to_a5 (2026-05-17). Iter-2 tripped on this exact symptom after worker added the include as a "make-sure-it-resolves" measure; removing the line let `--opkernel` proceed cleanly to ELF emission. The A3 upstream pattern was the authoritative counter-example.

**Other instances (predicted)**: any greenfield A5 port (Mode B / Mode B-simple / Mode B-mechanical per OL-141 / OL-158). Especially likely when worker generates apt.cpp from scratch rather than copying-and-editing an existing arch35 sibling. Add to W11 apt.cpp emission checklist.

**Cross-reference**: OL-141 (target include structure is advisory; derive the task-owned include set
from current public APIs), W11 (arch35 apt.cpp emission gate).

<!-- 迁移自 porter kb/target/ascendc/（EC-52，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
