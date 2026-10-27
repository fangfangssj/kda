---
type: Runbook
title: "API static-template-arg vs runtime-member ambiguity — trace the source to find the actual driver"
description: "API static-template-arg vs runtime-member ambiguity — trace the source to find the actual driver"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.cube, topic.data_movement, topic.tiling, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-92-api-static-template-arg-vs-runtime-member-ambiguit.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-92"]
---

# API static-template-arg vs runtime-member ambiguity — trace the source to find the actual driver

> 原 OKF v1 卡号：**OL-92**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - API has parallel config surfaces — e.g. a template parameter named <X> AND a member function Set<X>(bool). The naming suggests they're equivalent; the source ma

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Category**: platform_compat
- **Loaded by**: aog-kernel-worker / aog-precision-probe when an API exposes BOTH a compile-time template arg AND a runtime setter for the same conceptual property, and worker is choosing which to set
- **Trigger**: API has parallel config surfaces — e.g. a template parameter named `<X>` AND a member function `Set<X>(bool)`. The naming suggests they're equivalent; the source may say otherwise.
- **Principle**: When an API exposes a property through both a static template argument AND a runtime member function, **the actual driver at the operation site is determined by what the source consults** — not by the type's name and not by what the template arg is "obviously" for. Set both to the same value if you can, but if you must pick one, **trace the impl**:
  - Search for reads of the static value (`<TYPE>::<flag>` / `<TYPE>::is<X>`). If reads only appear in a path that's gated by a different feature (e.g. quantization scale, MX-FP8 tile format), the static arg is **not** the general driver — it's a sub-regime hint.
  - Search for reads of the runtime member (e.g. `IsTransposeA()` / `Get<X>()`). If those are at the operation site (the place that actually does the matmul / sort / etc.), the runtime member IS the driver.
- **The trap** (compiles, runs, produces wrong output): set the static template arg to "true" and the runtime member to "false" because you assumed the static arg is the canonical/preferred form. The op proceeds along the runtime-member-driven path and IGNORES the static arg → garbage output, no error.
- **Diagnostic signature**: precision FAIL with **algorithmic-magnitude residuals** (max_abs_diff in tens or hundreds, mean_abs_diff > 1, well above any rounding-class jitter). This is the signature of "the kernel computed a related-but-wrong formula", not a precision drift. If you see this signature on an op with parallel static+runtime config surfaces, OL-92 is the candidate root cause.
- **Detection workflow**:
  1. Identify the parallel static+runtime config surfaces in the API
  2. Open the impl source and `grep` for both
  3. Check what `Init() / IterateAll() / Compute()` (or whatever the operation entry is) actually consults
  4. Set the runtime member to the desired value; set the static arg to its **default** (often `false`) unless you know it engages a different path that you also want
- **Cube-unit instance** (canonical evidence): `MatmulType<..., ISTRANS>` template arg is stored as `A_TYPE::isTrans` but only read in the MX-FP8/scale path (`asc/impl/adv_api/detail/matmul/utils/mx_matmul_utils.h`); the ND→ND transpose decision reads `MatmulShapeInfoBase::isTransposeA_` set by `SetTensorA(_, bool)` runtime. Code template, 4-corner-lattice tiling field map, and cube-specific evidence live in `patterns/domains/platform_compat.md §P-P69`. **Five cube ops validated** the runtime-bool-as-driver across {none, A, B, both} transpose corners; rule is empirically settled for cube.
- **Other API regimes (predicted to have this trap)**: hardware sort engine if it has `Sort<DESC>` template + `SetSortDirection(bool)` runtime; reduction engines with template-vs-runtime accumulator-mode flags; quantization engines with template `<Symmetric>` + runtime `SetQuantMode()`. Apply the detection workflow before assuming the template arg is the driver.
- **Evidence**: 4_MatmulTransA Phase D iter 1 (2026-04-28) — algorithmic-magnitude precision FAIL (max_abs_diff 5–120) when worker followed orchestrator brief that said "use `MatmulType<…,ISTRANS=true>`". Worker self-corrected by reading `matmul_shape_info.h` directly (impl source trace), found runtime-bool path. Iter 2 with `ISTRANS=false` template + `SetTensorA(_, true)` runtime → 50/50 PASS fp32 bit-exact, 1.36× median. Subsequent ops 5_MatmulTransB / 3_MatmulBothTrans (0+0 iters each) confirmed the rule across all transpose corners.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-92（category=platform_compat，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
