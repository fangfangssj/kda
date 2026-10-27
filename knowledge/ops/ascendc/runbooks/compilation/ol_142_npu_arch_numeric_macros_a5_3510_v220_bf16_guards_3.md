---
type: Runbook
title: "`__NPU_ARCH__` numeric macros — A5 = 3510, V220 BF16 guards = 3003 / 3113 [V351, port_a3_to_a5]"
description: "applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=all"
tags: [task.debugging, topic.programming_model.execution_model.simt, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-142-npu-arch-numeric-macros-a5-3510-v220-bf16-guards-3.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3, "950"]
aliases: ["OL-142", "__NPU_ARCH__"]
---

# `__NPU_ARCH__` numeric macros — A5 = 3510, V220 BF16 guards = 3003 / 3113 [V351, port_a3_to_a5]

> 原 OKF v1 卡号：**OL-142**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=all

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 (Ascend/agent-skills) ascendc-operator-A5-migration SKILL.md §271-303 + §401`

**Rule**: A5 kernel code uses `__NPU_ARCH__ == 3510` for compile-time A5 detection. V220 (A3) BF16-or-feature guards in the A3 source use either `__NPU_ARCH__ == 3003` or `__NPU_ARCH__ == 3113`. When producing an `op_kernel/arch35/` port, every occurrence of these V220 guards MUST be REMOVED — they exclude valid A5 code paths.

**Why**: A3 sources liberally guard newer features (notably BF16) behind `#if !(__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113)`. On A5 these conditions are false (the macro doesn't match 3003 or 3113 because A5 is 3510), but ONLY by accident — the guard was written to MEAN "skip on old V220 silicon", not "include on A5". Leaving the guard in `arch35/` creates a fragile dependency on accidental macro equality; future macro changes break the port.

The correct A5 port:
- Removes V220-specific guards that excluded BF16 / new dtype paths (every BF16-only branch in A3 → unconditional in A5)
- Wraps A5-specific new code (SIMT, MicroAPI, overflow-mode SPR access) in `__NPU_ARCH__ == 3510`

**Concrete A3 → A5 transformations** (from PR 103 §289-303):

```cpp
// A3 source (V220) — BF16 path guarded
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
    op.Init(...);
    op.Process();
#endif

// A5 port (arch35/) — guard REMOVED, code unconditional
MoeFinalizeRouting::MoeFinalizeRoutingBf16CutK<bfloat16_t> op;
op.Init(...);
op.Process();
```

```cpp
// A5-only new code (SIMT / overflow-mode SPR) — guard with 3510
#if (__NPU_ARCH__ == 3510)
    int64_t saved = AscendC::GetCtrlSpr<60, 60>();
    AscendC::SetCtrlSpr<60, 60>(0);  // disable overflow check for perf
#endif
```

**Evidence**:
- PR 103 cites this exact pattern in §289-303 with `MoeFinalizeRoutingBf16CutK` as the canonical example
- Our `EC-A5-BF16-GUARD` would fire on every L1 port if not handled — applies to 14 ACTIONABLE ops in scan list

**Other instances (predicted)**: every A3 op with V220 macro guards needs the same removal. Apply mechanically during L1 (basic adapt) phase of any port.

**Cross-reference**:
- OL-141 (skip-if-upstream-present) — supersedes this when upstream exists
- EC-49 below (BF16 guard removal in arch35/)
- L1 tier of OL-143 (migration tier classifier)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-142（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
