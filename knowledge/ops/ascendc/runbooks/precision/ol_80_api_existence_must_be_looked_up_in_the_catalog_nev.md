---
type: Runbook
title: "API existence MUST be looked up in the catalog — never invent a workaround from memory or guess"
description: "API existence MUST be looked up in the catalog — never invent a workaround from memory or guess"
tags: [task.debugging, topic.precision, topic.programming_model.execution_unit.vector, topic.data_movement, topic.memory]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/precision/ol-80-api-existence-must-be-looked-up-in-the-catalog-nev.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["OL-80"]
---

# API existence MUST be looked up in the catalog — never invent a workaround from memory or guess

> 原 OKF v1 卡号：**OL-80**。
> 现象分类：precision_issue；观测置信度：single_run（单次观测）。
> 触发信号：
> - when implementing an op you think of some operation ("divide each element by a scalar", "scalar broadcast", "tensor reduce") and you are not sure whether Ascend

## 适用场景

- 适用技术与问题域：AscendC 精度问题（现象分类：precision_issue）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 22_HybridAttentionMaskPreparation). Do not downgrade.
- **Category**: process_rules
- **Loaded by**: Generator (ALWAYS before writing VEC ops), Orchestrator (before editing kernel code)
- **Trigger**: when implementing an op you think of some operation ("divide each element by a scalar", "scalar broadcast", "tensor reduce") and you are not sure whether AscendC has a corresponding API
- **Lesson**: **Never say "AscendC does not have API X" from memory and then write a workaround**. All VEC APIs are in `src/skills/references/target/ascendc/API_CATALOG.md` (278 lines); grep finds them in 3 seconds. The hallucinated pattern "I assumed it doesn't exist → I used a workaround → I failed for multiple rounds" is a classic single-agent error mode under attention pressure. The correct flow:
  1. Think of an operation → immediately grep the catalog (not later, **immediately**)
  2. If not in the catalog → `fetch_ascendc_doc.py <ApiName>` to query hiascend.com official docs
  3. Only if neither finds it → consider a workaround, and mark "API missing" in knowledge_update.md
- **Typical failure case**:
  - 2026-04-16 op #14: orchestrator assumed `Divs` (scalar div) did not exist and used `Muls(x, 1/std)`. In reality the catalog line 54 clearly has `Divs`. Wasted 3 rounds on the wrong assumption.
  - Lesson consistency: this is not the model "unable to look up" — it is "forgetting to look up" under pressure. Therefore the skill prompt must treat "look up API" as a **mandatory pre-step**, not "look up when you hit a problem".
- **Anti-pattern** (never do this):
  ```
  // WRONG: imagining that an API is missing, going straight to a workaround
  // "AscendC doesn't have scalar div — use Muls(1/x)"
  Muls(dst, src, 1.0f / scalar, count);
  ```
- **Correct pattern**:
  ```
  // 1. Grep catalog first
  //    grep -i "Divs\|scalar div" src/skills/references/target/ascendc/API_CATALOG.md
  // 2. Found it → use it
  Divs(dst, src, scalar, count);
  ```
- **Evidence**: E2 level (observation of orchestrator's own behavior). Applies to all single-agent agents (worker and orchestrator). Corresponding skill improvement: SKILL.md Stage 1 `Write kernel code` step must have a sub-step "grep ASCENDC_API_CATALOG for all VEC ops needed" up front.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-80（category=process_rules，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
