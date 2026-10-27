---
type: Runbook
title: "ops-nn UT `add_modules_ut_sources` splits test modules by FILENAME GLOB — a test cpp mixing tiling + infershape + aclnn concerns links only against its glob-module's libs → undefined references at ld [ops-nn ut.cmake, host build]"
description: "ophost UT 链接缺 `InferShapeContextFaker`——一个测试文件只管一件事并按 glob 命名（test_<op>_tiling.cpp 等）。"
tags: [task.debugging, topic.implementation.tiling, topic.compilation]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["EC-281", add_modules_ut_sources]
---

# ops-nn UT `add_modules_ut_sources` splits test modules by FILENAME GLOB — a test cpp mixing tiling + infershape + aclnn concerns links only against its glob-module's libs → undefined references at ld [ops-nn ut.cmake, host build]

`applies_to: soc=all (host-side cmake, SoC-independent); cann=9.0.0 ops-nn UT harness (`ut.cmake` `add_modules_ut_sources`); bisheng=n/a; op_class=all ops-nn vendor ops with UT`
`verified_on: cann=9.0.0 ops-nn UT harness (conv2d_backprop_input 310p ops-math flow, 2026-08-26 — ophost UT link failure, iter 2→3)`
- **Symptom**: ld-time `undefined reference to InferShapeContextFaker` (or the tiling/opapi faker equivalents) when building the ophost UT — the test cpp compiles but does not link.
- **Root cause**: `add_modules_ut_sources` assigns test sources to modules BY FILENAME GLOB: `test_*_tiling.cpp` → tiling module, `test_*_infershape.cpp` → infershape module, `test_aclnn_*.cpp` → opapi module. A single test cpp mixing tiling + infershape tests is globbed into ONE module and links only against that module's libs — the other concern's faker/framework symbols are missing.
- **Fix**: one concern per test file, named per the glob (`test_<op>_tiling.cpp`, `test_<op>_infershape.cpp`, `test_aclnn_<op>.cpp`). No cmake edit needed — the split is the contract.
- **Evidence**: conv2d_backprop_input 310p (2026-08-26, ops-math flow): ophost UT link failure at iter 2→3; splitting the mixed test cpp per glob → ophost UT 11/11, opapi UT 14/14 (`conv2d_backprop_input__opsmath/evidence/ut_result.json`).
- **Cross-ref**: OL-369 (the UT-first methodology this UT layout serves). backend=ascendc.
