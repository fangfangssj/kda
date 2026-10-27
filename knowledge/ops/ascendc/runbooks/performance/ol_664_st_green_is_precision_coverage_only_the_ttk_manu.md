---
type: Runbook
title: "ST-green is precision-coverage only — the ttk manual-binary lane cannot ingest bf16 offline (no ml_dtypes/tensorflow on the box) and its tiny shapes mask slow kernels; move bf16 coverage to the case-pack runner and A/B a mid-size shape against the CANN built-in before any perf claim"
description: "ST 全绿只证明精度覆盖，ttk 手册的 ST 不含性能门禁——别把 ST 绿当性能健康的证据"
tags: [task.optimization, topic.compute.elementwise.type_conversion, topic.compilation, topic.implementation.masking, topic.performance.analysis, topic.precision]
status: stable
sources: []
created_at: "2026-09-15T16:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
platforms: ["310p"]
aliases: ["OL-664"]
---

# ST-green is precision-coverage only — the ttk manual-binary lane cannot ingest bf16 offline (no ml_dtypes/tensorflow on the box) and its tiny shapes mask slow kernels; move bf16 coverage to the case-pack runner and A/B a mid-size shape against the CANN built-in before any perf claim

`applies_to: soc=Ascend310P3 (dav-m200); cann=9.0.0 ops-math 2.1 suite flow (ttk ST + case-pack runner); bisheng=n/a; op_class=all (conv-class sharpest)`
`verified_on: soc=Ascend310P3; cann=9.0.0 (conv2d_backprop_input 310p, 2026-08-26; identical ST dtype constraint confirmed in finalized add + avg_pool3_d csvs)`
- **Category**: ops-math suite methodology — what each validation lane can and cannot prove
- **Trigger**: preparing to claim "ST green" or "perf healthy" for an op on the ops-math 2.1 suite flow.
- **Principle (two independent blind spots of the ST/ttk lane)**:
  1. **Dtype blind spot (environmental, not op-driven)**: ttk's `numpy_bfloat16()` needs `ml_dtypes` or `tensorflow`; neither is pip-installable offline on the 310P box, so ST csvs are fp32+fp16 only. bf16 tiling-key coverage does NOT live in ST — it moves to the case-pack runner, which dumps bf16 widened to fp32 (CAND-310P-OPSMATH-BF16-DUMP-FP32-WIDEN). Do not burn time trying to make ttk eat bf16 .bin.
  2. **Perf blind spot (shape distribution)**: ST kernel-only times (6–561 us) looked healthy, but all ST shapes are tiny; per-element scalar cost only surfaces at aclnn/case-runner scale or larger shapes. A v1 scalar conv kernel measured **0.0054× geomean vs the CANN built-in** over 14 func cases (msprof: 80.9% scalar_ratio, vec 0.2%, mac 0.0%) while its ST was green — func_float16_d4 took 29.2 s vs CANN 9.6 ms.
- **Decision rule**: for conv-class (any per-element-cost-scaling) ops — ST green ≠ perf healthy. Before claiming perf, A/B at least one mid-size shape against the CANN built-in on the same device, same session (per CLAUDE.md same-condition A/B rule). When the gap is algorithm-class (scalar where the reference is cube), the fix is algorithm selection (P-P78), not tuning.
- **Evidence**: conv2d_backprop_input 310p (2026-08-26, kw-1 respawn perf investigation, same-device A/B): geomean 0.0054× over 14 func cases; `conv2d_backprop_input__opsmath/evidence/` (case_runner_run.log, example_smoke.log). First instance for both blind spots.
- **Other instances (predicted)**: every op shipping manual ST cases on the 310P flow (bf16 blind spot is universal — confirmed identical in add + avg_pool3_d csvs); every conv/matmul-class op whose ST battery is small-shape-only.
- **Cross-ref**: (where bf16 coverage actually lives), (the torch-eager floor frame for the A/B), P-P78 (the algorithm-class fix), (the ST packaging contract). backend=ascendc.
