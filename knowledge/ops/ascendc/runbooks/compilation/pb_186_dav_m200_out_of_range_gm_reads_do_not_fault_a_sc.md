---
type: Runbook
title: "dav-m200 out-of-range GM reads do NOT fault — a scalar load at GM offset 2^34 silently returns 0xffff and the kernel completes; \"zero observable writes + no error\" can therefore be a MIS-ADDRESSED access, not only \"did not run\\"
description: "dav-m200 越界 GM 读**不报错**——偏移 2^34 的标量读静默返回 0xffff；\"零写入+无错误\"可能是地址算错了。"
tags: [task.debugging, topic.compilation, topic.programming_model.execution_unit.scalar]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["PB-186"]
---

# dav-m200 out-of-range GM reads do NOT fault — a scalar load at GM offset 2^34 silently returns 0xffff and the kernel completes; "zero observable writes + no error" can therefore be a MIS-ADDRESSED access, not only "did not run\

`applies_to: soc=Ascend310P3 (dav-m200, arch 200x); cann=9.0.0; bisheng=n/a; op_class=all (any kernel doing address arithmetic on GM pointers)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (29_TanhGatedResidualAddBackward 310p spawn 3, 2026-09-06, DEVICE-RUN probe)`
`unverified_on: soc=Ascend950PR (351x), Ascend910 (220x) — OOB-read behavior not probed there`
- **Behavior**: a scalar `GlobalTensor` load at GM offset 2^34 returned `0xffff` silently — kernel completed, no exception, no plog. Wild/out-of-range GM READS do not trap on this SoC.
- **Debug implication (why this is a PB, not a curiosity)**: on a SoC where OOB reads are silent, "outputs all zero/garbage AND no error" does NOT discriminate between "kernel never executed" and "kernel executed with mis-addressed reads/writes". Triage must FIRST prove execution (chrome-trace kernel duration, P-P131) and write-vs-garbage (dirty-allocator-block probe, P-P131) before concluding "did not run". This exact ambiguity cost op 29's spawn 2 a wrong belief — "kernels never execute" — when they executed and wrote nothing.
- **Evidence**: 29_TanhGatedResidualAddBackward 310p spawn 3 (2026-09-06, DEVICE-RUN): probe recorded in `29_TanhGatedResidualAddBackward/analysis.md` (wild GM read at offset 2^34 → 0xffff, kernel completed).
- **Other instances (predicted)**: any dav-m200 debug session leaning on "no crash ⇒ addresses were valid"; pointer-arithmetic bugs in multi-tensor offset kernels (4-slab offset-dst load shapes).
- **Cross-ref**: P-P131 (the probe triad that resolves the ambiguity this creates), [`PB-124`](pb_124_on_dav_m200_multi_core_scalar_gm_stores_are_sile.md) (dav-m200 silent-failure family — stores dropped under concurrency), CAND-DAV-M200-DATACOPY-STRIDE-SILENT-NOOP (strides silently ignored), [`PB-185`](pb_185_dav_m200_composed_function_vec_kernel_faults_det.md) (the fault this debugging uncovered). backend=ascendc.
