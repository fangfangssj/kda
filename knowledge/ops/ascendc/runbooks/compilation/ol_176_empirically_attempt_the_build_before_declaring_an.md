---
type: Runbook
title: "Empirically attempt the build before declaring an L4 \"structural ceiling\" [V351/Ascend950PR, port_a3_to_a5, L0-agent-discipline + classifier-override]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.cube, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-176-empirically-attempt-the-build-before-declaring-an.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-176"]
---

# Empirically attempt the build before declaring an L4 "structural ceiling" [V351/Ascend950PR, port_a3_to_a5, L0-agent-discipline + classifier-override]

> 原 OKF v1 卡号：**OL-176**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`

**Principle**: Surface metrics (LOC, Mmad call-site count, CrossCoreSetFlag count, op-class taxonomy) are **necessary-not-sufficient** evidence for routing a port_a3_to_a5 op to L4 ("structurally untranslatable"). They are over-broad classifiers — high metrics correlate with hard ports but do not entail one. Before any L4 declaration, the worker MUST attempt the minimum-viable build once.

**Minimum-viable build procedure**: when upstream A3 lacks `op_kernel/arch35/`, author a thin worker TU (`kernels.cpp` + `pybind11.cpp`) using verbatim upstream headers + `KERNEL_TASK_TYPE_MIX_AIC_1_2` dispatch (no arch35 wrapping, default-OFF prestage per CLAUDE.md port_a3 rule). Run `deploy_to_npu_lane.sh --build` once.

**Decision rule from build outcome**:
- Build PASS → the port is L1/L2 wiring, not L4. The `__CCE_AICORE__==220` guards in V220 upstream are benign on V351 (early-return doesn't fire; code follows the V351 path).
- Build FAIL with arch-specific errors after 3-5 compile-fix iterations → escalate per OL-159 (FA-class structural rewrite) / OL-156 (V220→V351 paradigm rewrite).
- Build FAIL with environment errors (resolver, container, CANN install) → fix infra, NOT escalate.

**Evidence**: `lightning_indexer_grad` 2026-05-23: build PASS in 1 spawn disproved 4 prior kw spawns + 2 researchers + 1 probe that had declared L4 based on surface metrics (2476 LOC / 9 Mmad call-sites / 8 CrossCoreSetFlag — all "high" by V220 standards). Those metrics correctly identified that the port was non-trivial but incorrectly predicted that it was unbuildable. The probe-before-fix discipline (CLAUDE.md) applies here: 1 build attempt is cheaper than 4 reroute spawns + analysis docs.

**Cross-ref**: OL-156/OL-159/OL-141/OL-175, CLAUDE.md "Probe-Before-Fix" + "port_a3_to_a5 default-OFF prestage".

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-176（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
