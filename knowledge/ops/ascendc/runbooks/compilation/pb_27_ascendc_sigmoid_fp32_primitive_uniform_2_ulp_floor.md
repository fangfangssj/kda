---
type: Runbook
title: "AscendC `Sigmoid<fp32>` primitive — uniform 2-ULP floor (slight correction to OL-103 \"1-ULP\" claim)"
description: "AscendC `Sigmoid<fp32>` primitive — uniform 2-ULP floor (slight correction to OL-103 \"1-ULP\" claim)"
tags: [task.debugging, topic.precision, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/pb-27-ascendc-sigmoid-fp32-primitive-uniform-2-ulp-floor.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: [a3, "950"]
aliases: ["PB-27", "Sigmoid<fp32>"]
---

# AscendC `Sigmoid<fp32>` primitive — uniform 2-ULP floor (slight correction to OL-103 "1-ULP" claim)

> 原 OKF v1 卡号：**PB-27**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - AscendC::Sigmoid<float> floor is 2 ULP uniformly across all bands (small / mid / transition / plateau). No degeneracy near zero (unlike Tanh in PB-24).

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **applies_to**: `soc=Ascend950PR_9579; cann=9.0.0 (V100R001C25B046); bisheng=15.0.5+2026-04-13`
- **applies_to**: `soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5+2026-01-28`
- **last_verified**: 2026-05-07 (A3 cross-arch confirmation)
- **status**: CONFIRMED on A5 AND A3 — uniform 2-ULP floor is **chip-family-wide** for Sigmoid. Both arches well-behaved.
- **Severity**: LOW (uniform 2-ULP across all bands — well-behaved, no failure mode on either arch).
- **Symptom (measured, both A5 and A3)**: `AscendC::Sigmoid<float>` floor is **2 ULP** uniformly across all bands (small / mid / transition / plateau). No degeneracy near zero (unlike Tanh in PB-24).
  - **A5**: max 2 ULP, mean 0.39 ULP, 99% of 14,349 points within 1 ULP, max abs err 1.04e-7. Histogram: 0→8941, 1→5210, 2→198, ≥4→0.
  - **A3**: max 2 ULP across ALL bands (including tiny + small), 62% bit-exact, **99.95% within 1 ULP** — even cleaner aggregate than A5. No points above 3 ULP.
- **Root cause hypothesis**: bisheng's `Sigmoid` polynomial is correctly-rounded-most-of-the-time, 2-ULP bound. Compared to `Tanh`, does NOT have a small-x failure mode — the `1/(1+exp(-x))` formulation preserves expected output magnitude near 0 (sigmoid(0) = 0.5), so the ULP measurement isn't pathological.
- **Workaround**: for sub-ULP fp32 sigmoid, implement via `Exp + Reciprocal + Add` (same primitives `Tanh` Cephes-form uses). For 2-ULP-tolerant uses (most ML inference paths), the primitive is fine on both arches.
- **Detection**: edge_dataset cases at the 2-ULP boundary may show 0/1/2 ULP scatter; distinguish primitive ceiling from kernel cancellation drift via this entry.
- **Evidence**:
  - A5: `workspace/_probes/tanh_sigmoid_precision_a5_cann9.0.0/PROBE_REPORT.md` (2026-05-06)
  - A3: `workspace/_probes/tanh_sigmoid_precision_a3/PROBE_REPORT.md` (2026-05-07, DS-side)
- **Cross-reference**: PB-24 (paired Tanh measurement, very different bimodal profile despite both being polynomial-evaluation primitives), OL-103 §Refined-statement (1-ULP floor → 2-ULP floor refinement, pending DS A3 + soften edit).

<!-- 迁移自 porter kb/target/ascendc/（PB-27，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
