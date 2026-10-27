---
type: Runbook
title: "Explicit `kernel_operator_<adv>_intf.h` include fails — adv_api headers are pulled transitively via `kernel_operator.h`"
description: "applies_to: soc=all; cann=9.0.0; bisheng=2026-03-21; op_class=all"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-46-explicit-kernel-operator-adv-intf-h-include-fails.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-46", "kernel_operator_<adv>_intf.h", "kernel_operator.h"]
---

# Explicit `kernel_operator_<adv>_intf.h` include fails — adv_api headers are pulled transitively via `kernel_operator.h`

> 原 OKF v1 卡号：**EC-46**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - kernel explicitly adds #include "kernel_operator_<adv>_intf.h" (e.g. kernel_operator_sigmoid_intf.h) following ASCENDC_API_CATALOG.md's "Header: adv_api/<adv>/k

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=9.0.0; bisheng=2026-03-21; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`
- **Severity**: MEDIUM (compile-time hard fail, well-defined fix)
- **Status**: CONFIRMED 2026-05-13 group_norm_silu_quant build iter 1
- **Symptom**: kernel explicitly adds `#include "kernel_operator_<adv>_intf.h"` (e.g. `kernel_operator_sigmoid_intf.h`) following ASCENDC_API_CATALOG.md's "Header: adv_api/<adv>/kernel_operator_<adv>_intf.h" line. Build fails first preprocessor pass: `'kernel_operator_sigmoid_intf.h' file not found`. The header DOES exist on disk at `cann-{ver}/<arch>-linux/asc/include/adv_api/<adv>/kernel_operator_<adv>_intf.h` (EC-38 confirms presence), but the default build include path does NOT reach `adv_api/` — adv_api headers are intended to be pulled in transitively, not directly included.
- **Root cause**: ASCENDC_API_CATALOG.md's "Header: ..." annotation documents WHERE the API is defined for human reference; it is NOT a recommendation to add an explicit `#include`. `kernel_operator.h` is the canonical entry point that pulls in all adv_api headers transitively. Adv_api intf.h files reference internal types/macros assuming kernel_operator.h was loaded first; including them standalone breaks even if the path were on the search list.
- **Fix**: keep ONLY `#include "kernel_operator.h"` (or `<kernel_operator.h>`). Call the adv API directly — `Sigmoid(dst, src, count)` / `Tanh(dst, src, count)` / etc. — and trust the transitive include.
- **Signature note (Sigmoid specifically)**: prefer the 2-arg form `Sigmoid(dst, src, count)` (sizes its tmp buffer internally from `srcTensor.GetSize()`) over the older 3-arg form `Sigmoid(dst, src, tmpBuf, count)`. Cousin op `11_DequantSwigluQuant` uses the 2-arg variant and is the reference template for sigmoid-bearing kernels.
- **Detection rule**: compile error `'kernel_operator_<adv>_intf.h' file not found` + the offending #include line traces back to a worker reading ASCENDC_API_CATALOG.md's "Header:" annotation. Fix is mechanical: delete the explicit #include line.
- **Catalog edit candidate**: ASCENDC_API_CATALOG.md should re-annotate "Header: ..." entries as "(transitive via `kernel_operator.h` — do NOT add explicit `#include`)" or drop the field entirely. Filed as catalog improvement; not blocking this EC.
- **Related**: EC-38 (catalog miss ≠ API doesn't exist; the `ls adv_api/<name>/` step that EC-38 mandates verifies the API exists — but EC-46 says you still don't need to include it explicitly once verified).
- **Evidence**: group_norm_silu_quant (2026-05-13, A5 fused GroupNorm+SiLU+Quant port) — iter 1 added `#include "kernel_operator_sigmoid_intf.h"` per catalog line 234 advice → build failed `file not found`. Iter 2 dropped the explicit include, used `Sigmoid(...)` directly with `<kernel_operator.h>` already in scope → build PASS, all 8 cases bit-exact on Pass A. Cousin op `11_DequantSwigluQuant` archive validated the no-explicit-include pattern is the project precedent.

<!-- 迁移自 porter kb/target/ascendc/（EC-46，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
