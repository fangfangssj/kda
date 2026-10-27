---
type: Runbook
title: "prove a shared fa_class-template edit is \"FA-neutral\" with a 3-gate codegen-diff — `.text`-section md5 of the FA `.o` before/after (determinism-validated) + live bit-exact + downstream-compile — NOT logical reasoning"
description: "Provenance: distilled from the SFA a3→a5 forward port (2026-06-22, scan/pr/y1-fa-template-lift commit b56b3be3, \"lift sparse-gather generalization into the fa_class template\"). The SFA operator itself"
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.implementation.masking, topic.data_movement, topic.compilation, topic.precision]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-280-prove-a-shared-fa-class-template-edit-is-fa-neutra.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: ["950"]
aliases: ["OL-280"]
---

# prove a shared fa_class-template edit is "FA-neutral" with a 3-gate codegen-diff — `.text`-section md5 of the FA `.o` before/after (determinism-validated) + live bit-exact + downstream-compile — NOT logical reasoning

> 原 OKF v1 卡号：**OL-280**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - Provenance: distilled from the SFA a3→a5 forward port (2026-06-22, scan/pr/y1-fa-template-lift commit b56b3be3, "lift sparse-gather generalization into the fa_c

> Provenance: distilled from the SFA a3→a5 forward port (2026-06-22, `scan/pr/y1-fa-template-lift` commit `b56b3be3`, "lift sparse-gather generalization into the fa_class template"). The SFA operator itself was superseded/never-merged, but the FA-neutrality proof method is reusable and grounds the existing "shared-template edit must be FA-regression-gated" gesture. Re-anchored against latest main `e3e4b051` (2026-07-24): the two edited template files `wp_block_cube.h` / `wp_kernel_train.h` are byte-identical to the merge-base the lift was proven against, so the recipe applies verbatim.

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR (a5, arch 351x); cann=9.x; op_class=fa_class (shared wholeport template); kernel_type=fa_class template edit`
`verified_on: SFA forward a3→a5 port whitebox (2026-06-22, Ascend950PR) — FA TU .text md5 aic bf694f2c / aiv 036f01af reproduced identical before/after the sparse-gather lift`

**Principle**: when you add a new feature (block-sparse gather, a new mask mode, etc.) to a SHARED fa_class template that plain forward/training FA also instantiates, "it's guarded by `if constexpr(hasSparse)` so FA is unaffected" is a **claim to be PROVEN, not asserted** — a mis-scoped guard or a shared helper edit can leak into the FA codegen. Prove FA-neutrality with three gates, in order:
- **Gate ① codegen-diff (load-bearing, compile-only, NO NPU device needed)**: compile **only the FA translation unit** (all launchers `hasSparse=false`/`hasRope=false`/`isInfer=false`) with the a5 bisheng/ccec toolchain **before** and **after** the template edit; compare the **`.text`-section md5** of the FA kernel `.o` (strip metadata/`.comment`, e.g. `objdump -d` or `llvm-objcopy --only-section=.text` then md5). **Determinism-validate first**: a clean rebuild of the *same* source must reproduce the same md5 (else the md5 is noise, not a fingerprint). Identical before/after `.text` md5 = the new feature DCE's out of the FA path → codegen byte-identical → provably FA-neutral. This gate needs the **bisheng compiler only** (an a5 build host); it does **not** need a live NPU.
- **Gate ② live bit-exact**: build the FA `.so` from the edited template, run the FA op on representative shapes before vs after; require `max_abs_diff == 0.0`. Needs a real Ascend950PR NPU.
- **Gate ③ downstream-compile**: the op that motivated the edit still compiles+runs from the shared template.
- **Why not logical reasoning alone**: on the SFA lift, Gate① caught a real leak (an isInfer-guard in `Process()`) that by-eye "this is hasSparse-gated" reasoning had passed as clean.

**Concrete anchor**: the SFA lift touched `wp_block_cube.h` + `wp_kernel_train.h` (3-site topK packing-gather + `GetS2LoopRange` hasSparse-first branch, all `hasSparse`-gated). Gate① compiled the FA TU before/after → `.text` md5 `aic bf694f2c / aiv 036f01af` identical both ways (determinism pre-validated by a clean rebuild reproducing the md5) → FA codegen byte-identical. On latest main these two files are byte-identical to that proven base, so Gate① is re-runnable and expected to reproduce the fingerprint.

**Evidence**: SFA whitebox task#14 (`scan/home` whitebox_log). One refuted leak (Process() isInfer-guard) proved the gate necessary. **E2E status**: Gate① is compile-only (a5 build host, no device) and is the load-bearing check for the FA-neutrality claim; Gate② needs an Ascend950PR NPU (route allocation via the main hub per the NPU-usage protocol).

**Cross-ref**: OL-210 / the "fixing a shared `regbase_matmul.h` bug is NOT FA-neutral → must be FA-regression-gated, deferred to its own PR" gesture (this OL is the concrete recipe behind that principle), `patterns/domains/fa_class_template.md` (the shared FA-class template this guards), EC-86 (the uninitialized-`RunInfo<true>` latent hazard in the same template — a shared-template edit's other failure mode). backend=ascendc.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-280（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
