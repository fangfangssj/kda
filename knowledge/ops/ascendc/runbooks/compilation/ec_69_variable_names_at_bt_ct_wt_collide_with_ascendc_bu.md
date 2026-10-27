---
type: Runbook
title: "Variable names `AT`, `BT`, `CT`, `WT` collide with AscendC builtin enums in `cce_aicore_intrinsics.h`"
description: "applies_to: soc=all; cann=all; bisheng=all; op_class=all"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-69-variable-names-at-bt-ct-wt-collide-with-ascendc-bu.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: []
aliases: ["EC-69"]
---

# Variable names `AT`, `BT`, `CT`, `WT` collide with AscendC builtin enums in `cce_aicore_intrinsics.h`

> 原 OKF v1 卡号：**EC-69**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - compile error when a local variable, template parameter, or struct member is named AT, BT, CT, or WT. The error typically surfaces as an ambiguous reference or

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=all; cann=all; bisheng=all; op_class=all`

- **Symptom**: compile error when a local variable, template parameter, or struct member is named `AT`, `BT`, `CT`, or `WT`. The error typically surfaces as an ambiguous reference or type-mismatch in code that otherwise looks valid — e.g. `error: expected ';' before '=' token` or `error: 'AT' was not declared in this scope` inside a function that included AscendC headers.

- **Root cause**: `cce_aicore_intrinsics.h` (included transitively by `kernel_operator.h`) defines `AT`, `BT`, `CT`, `WT` as enum values or macros in the global scope. Any user code that uses these names for variables or template parameters silently collides. The collision is namespace-less — both the builtin enum and the user symbol occupy the same name.

- **Fix**: rename user symbols away from the 4 reserved names. Recommended replacements: `MTA` (for A-transpose flag), `MTB` (for B-transpose flag), `MTC` (for C-type flag), `MTW` (for weight/workspace flag). Alternatives: prefix with a namespace-like tag (e.g. `kAT`, `kBT`).

```cpp
// BAD — collides with cce_aicore_intrinsics.h enum
bool AT = false, BT = true, CT = false, WT = false;

// GOOD — renamed away from the reserved names
bool MTA = false, MTB = true, MTC = false, MTW = false;
```

- **Detection**: grep kernel source for `\b(AT|BT|CT|WT)\b` used as a variable/parameter name (not inside a string or comment). The 4-letter token in a declaration context is the smoking gun.

- **Evidence**:
  - fused_quant_mat_mul kw-1 (2026-06-15): compile failed at `kernel/fused_quant_mat_mul_kernel.h:36` — `AT`, `BT`, `CT`, `WT` as bool flags collided with intrinsics enum. Renamed to `MTA`/`MTB`/`MTC`/`MTW` → compile passed.

<!-- 迁移自 porter kb/target/ascendc/（EC-69，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
