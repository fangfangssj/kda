---
type: Runbook
title: "AscendC autogen K_TYPE single-match-per-cpp — mixing AIV and AIC kernels in one .cpp silently mis-registers all but the first"
description: "AscendC autogen K_TYPE single-match-per-cpp — mixing AIV and AIC kernels in one .cpp silently mis-registers all but the first"
tags: [task.debugging, topic.precision, topic.implementation.pipeline, topic.programming_model.execution_unit.cube, topic.data_movement, topic.memory]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-42-ascendc-autogen-k-type-single-match-per-cpp-mixing.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["EC-42"]
---

# AscendC autogen K_TYPE single-match-per-cpp — mixing AIV and AIC kernels in one .cpp silently mis-registers all but the first

> 原 OKF v1 卡号：**EC-42**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - build succeeds with no error, optionally a [WARNING]: Multiple kernel functions are detected. It is recommended to define only one kernel function per file. lin

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Severity**: CRITICAL (silent corruption, no runtime error)
- **Status**: OPEN (CANN 9.0.0 build pipeline)
- **Symptom**: build succeeds with no error, optionally a `[WARNING]: Multiple kernel functions are detected. It is recommended to define only one kernel function per file.` line in build log. At runtime, the kernels meant to be AIC (e.g. cube via `MatmulImpl`) silently produce no output — destination buffer reads as uninitialized memory. Same .cpp's first-declared kernel (whose `KERNEL_TASK_TYPE_DEFAULT(...)` ran first) runs correctly; subsequent kernels of a different task type write nothing.
- **Trigger pattern**: kernel author puts kernels of DIFFERENT task types (`KERNEL_TYPE_AIV_ONLY` and `KERNEL_TYPE_AIC_ONLY`) in the same .cpp file, expecting per-`extern "C"` `KERNEL_TASK_TYPE_DEFAULT(...)` to apply locally to each kernel. The macro looks like a per-function attribute but is actually a file-scope global.
- **Root cause**: `cann-9.0.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py::find_kernel_type_by_source` uses `re.search` (single-match) on `__enable_feature_for_compile_default = X;` (the macro injected by `KERNEL_TASK_TYPE_DEFAULT`). Only the FIRST match is taken and applied to ALL kernels in that file. Mixing types silently mis-registers all but the first.
- **Fix**: split kernels into separate .cpp files by task type. Build script `kernel_dir.glob("*.cpp")` (excluding `pybind11.cpp`) picks up multiple files independently, each with its own first-and-only `KERNEL_TASK_TYPE_DEFAULT(...)` line. Naming convention: `<op>_aiv_kernels.cpp` + `<op>_aic_kernels.cpp` per the op#7 ConvStandard2d precedent.
- **Detection rule**: if your kernel set has BOTH `KERNEL_TYPE_AIV_ONLY` and `KERNEL_TYPE_AIC_ONLY`, you MUST split. The build warning is logged but easy to miss.
- **Diagnostic when output is silent zeros**: re-deploy a minimal one-cube-call kernel in its own .cpp first to confirm the cube path itself works, before debugging algorithm. If isolated cube works but combined doesn't → K_TYPE trap is the root cause.
- **Evidence**:
  - op#7 ConvStandard2d Opt1 (aog-kernel-optimizer ko-1, 2026-04-29). Single combined `conv2d_kernels.cpp` with AIV im2col + AIC cube + AIV bias → 50/50 precision PASS appeared but cube output was zeros (debug `4×8 @ 8×16 = ones×8` returned zeros). Split into `conv2d_aiv_kernels.cpp` + `conv2d_aic_kernels.cpp` → cube ran correctly, returned 8.0 as expected, end-to-end 50/50 + 16/16 PASS, perf median 0.087× → 0.155× → 0.705×.
  - op#6 QuantMatmul kw (Phase E backfill 2026-05-07). Two-launch quant-matmul: AIC int8 GEMM → workspace int32 → AIV dequant → output T. Single `quantmatmul_kernels.cpp` with both `[aicore]` and `[aivec]` `extern "C"` entry points; `device_aiv.o` was empty post-build, AIV dequant never ran, output buffer was uninitialized. Detection: `ls kernels_aiv_device_dir/` empty when AIV launches were expected. Fix: split into `quantmatmul_kernels.cpp` (AIC) + `quantmatmul_aiv_kernels.cpp` (AIV) → 50/50 PASS. Confirms generalization beyond conv-shaped two-launch ops; applies to the textbook quant-matmul shape on Ascend950PR.
- **Cross-ref**: OL-91 (cube playbook), P-P68 (single-AIC GEMM template). Any kernel layering AIV stages around an AIC cube call needs split-cpp.

<!-- 迁移自 porter kb/target/ascendc/（EC-42，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
