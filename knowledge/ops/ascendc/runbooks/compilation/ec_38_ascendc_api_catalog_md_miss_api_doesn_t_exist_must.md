---
type: Runbook
title: "ASCENDC_API_CATALOG.md miss ≠ API doesn't exist — must `ls` adv_api/ headers before falling back to manual decomposition"
description: "ASCENDC_API_CATALOG.md miss ≠ API doesn't exist — must `ls` adv_api/ headers before falling back to manual decomposition"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.cube, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-38-ascendc-api-catalog-md-miss-api-doesn-t-exist-must.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T08:14:40Z"
platforms: [a3, "950"]
aliases: ["EC-38"]
---

# ASCENDC_API_CATALOG.md miss ≠ API doesn't exist — must `ls` adv_api/ headers before falling back to manual decomposition

> 原 OKF v1 卡号：**EC-38**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Worker greps ASCENDC_API_CATALOG.md for an advanced API name (e.g. Sigmoid / Silu / Swish / Tanh) per OL-80 "API existence check". Catalog returns 0 hits → work

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Worker greps `ASCENDC_API_CATALOG.md` for an advanced API name (e.g. `Sigmoid` / `Silu` / `Swish` / `Tanh`) per OL-80 "API existence check". Catalog returns 0 hits → worker concludes "API not available on this chip". Worker falls back to manual decomposition (`Exp + Reciprocal + Mul` for sigmoid, hand-rolled polynomial for tanh). Resulting kernel: ULP-divergence from CANN reference, Pass A FAIL with `max_abs_diff ≈ 4e-7..1e-3` (polynomial-difference signature, not 1-ULP rounding drift).
- **Root cause**: `ASCENDC_API_CATALOG.md` is a HUMAN-MAINTAINED summary, not an auto-generated index — it lags behind real CANN releases. Many advanced API headers exist on the chip but are not (yet) listed in the catalog. The headers ARE present at `cann-{version}/aarch64-linux/asc/include/adv_api/<name>/kernel_operator_<name>_intf.h`. OL-80 grep is a cheap first-pass check, NOT a definitive existence check.
- **Fix (worker Phase A — MANDATORY when catalog grep miss)**: Before falling back to manual decomposition, run:
  ```bash
  # via /a3_op or /a5_op skill on the active container
  ls /usr/local/Ascend/cann-{ver}/aarch64-linux/asc/include/adv_api/ 2>/dev/null
  find /usr/local/Ascend/cann-{ver} -name "kernel_operator_<api>_intf.h" 2>/dev/null
  ```
  If the header exists → use the advanced API; expect bit-exact match against CANN reference (A-P35 advanced API regime). If the header genuinely doesn't exist → manual decomposition is the right path AND A-P35 contract softening (Pass B 1e-3 tolerance) applies.
- **Detection signal**: precision FAIL with residuals 1-ULP to 1e-3 in transcendental ops + worker's analysis.md cites OL-80 catalog grep but did NOT cite an `ls adv_api/` step — suspect catalog miss → manual decomposition trap.
- **Prevention (Phase A checklist addition)**: For every ascendc primitive used in the kernel, the analysis.md must list one of: (a) catalog §section it came from (existing OL-80 check), OR (b) actual `ls adv_api/<name>/` output showing the intf header.
- **Evidence**:
  - 2026-04-28 op#11 DequantSwigluQuant a3 cold-start: a different agent's worker grep'd catalog for `Sigmoid` → 0 hits → fell back to manual `Exp + Reciprocal + Mul` for silu → Pass A drifted from CANN reference. Fix: switched to advanced `Sigmoid()` API (header at `adv_api/sigmoid/kernel_operator_sigmoid_intf.h`) → bit-exact (a5 sibling op#11 archive kernel.h:309/328 is the precedent that surfaced the catalog gap).
  - General: 50+ ops over the project that touched activation / softmax / matmul historically had this trap; catalog gradually accreted §9.1 entries as ops surfaced them.
- **Related**: A-P35 (advanced API regime) — EC-38 is the discovery step that determines which A-P35 regime applies. OL-80 (API existence check) — EC-38 is the second-stage check after OL-80 grep miss. OL-91 / aog-self-critic C23 (bar-lowering verdicts without artifact evidence) — declaring "API doesn't exist" without `ls` is a C23 bar-lowering verdict labeled as authoritative narrative.

<!-- 迁移自 porter kb/target/ascendc/（EC-38，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
