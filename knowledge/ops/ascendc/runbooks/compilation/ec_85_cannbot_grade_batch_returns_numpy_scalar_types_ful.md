---
type: Runbook
title: "cannbot `grade_batch` returns numpy scalar types + full per-case arrays → `TypeError: not JSON serializable` and a 100s-of-MB verdict — coerce recursively to native + trim to a scalar summary before persisting"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (grading-harness plumbing, SoC-independent)"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-85-cannbot-grade-batch-returns-numpy-scalar-types-ful.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["EC-85", "grade_batch"]
---

# cannbot `grade_batch` returns numpy scalar types + full per-case arrays → `TypeError: not JSON serializable` and a 100s-of-MB verdict — coerce recursively to native + trim to a scalar summary before persisting

> 原 OKF v1 卡号：**EC-85**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - persisting verification.json after grading a batch through cannbot grade_batch raises TypeError: Object of type bool_ / float64 is not JSON serializable; separa

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (grading-harness plumbing, SoC-independent)`
`verified_on: soc=Ascend950PR (selective_scan_full_grad kw-1 2026-06-18 / A5)`

- **Symptom**: persisting `verification.json` after grading a batch through cannbot `grade_batch` raises `TypeError: Object of type bool_ / float64 is not JSON serializable`; separately, the raw grade result embeds full per-case arrays that bloat the verdict file (observed ~329 MB) with data that does not belong in a verdict artifact.
- **Root cause**: `grade_batch` returns numpy scalar types (`np.bool_`, `np.float64`) and full per-case numpy arrays inside its result dict; the stdlib `json` encoder cannot serialize numpy scalars, and the arrays are unbounded in size.
- **Fix**: (1) recursively coerce the result with a `_native()` helper before `json.dump` — `np.generic → .item()`, `np.ndarray → summary stats or drop`, recurse into `dict`/`list`; (2) trim per-case arrays to a scalar summary (pass/total counts + representative stats), not the raw arrays.
- **Detection**: a `json.dump(grade_result)` on a cannbot output raising the numpy-type `TypeError`, or a `verification.json` in the 100s-of-MB range.
- **Cross-ref**: OL-272 (backward-mode pybind build/deploy — same class of backward-mode harness plumbing carve-outs the worker must handle itself). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/（EC-85，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
