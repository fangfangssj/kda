---
type: Runbook
title: "AscendC API symbols are split between global scope and the `AscendC::` namespace — when a qualified or unqualified symbol isn't found, try the OTHER scope; the `did you mean simply X?` hint means the symbol lives at global scope"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.programming_model.execution_model.simt, topic.data_movement.data_copy, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ol-267-ascendc-api-symbols-are-split-between-global-scope.md
    role: primary
created_at: "2026-08-30T16:00:00Z"
updated_at: "2026-08-30T16:00:00Z"
platforms: [a3]
aliases: ["OL-267", "AscendC::"]
---

# AscendC API symbols are split between global scope and the `AscendC::` namespace — when a qualified or unqualified symbol isn't found, try the OTHER scope; the `did you mean simply X?` hint means the symbol lives at global scope

> 原 OKF v1 卡号：**OL-267**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=all

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend910_V220 (V220/910C); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5/V351) — the exact per-symbol split may differ by CANN/SDK version; the try-the-other-scope heuristic is version-agnostic`

**Principle**: the AscendC SDK does not export all of its API through one scope. Some symbols are at GLOBAL scope and some are in the `AscendC::` namespace, and the split is not documented per-symbol — so an unqualified name that "should" work fails, or an `AscendC::`-qualified name fails, depending on which bucket the symbol is in. The reliable resolution is empirical: when the compiler reports an unknown / unresolved symbol, try the OTHER scope before concluding the API is absent. A strong tell is the compiler's own hint — `did you mean simply X?` (i.e. dropping the `AscendC::`) reliably indicates the symbol is at GLOBAL scope.

**Concrete anchor (V220/910C observed split)**:
- GLOBAL scope (use unqualified): `CubeFormat`, `CFG_NORM`, `CFG_MDL`, the typed align-copy intrinsics `copy_gm_to_ubuf_align_{b8,b16,b32}` / `copy_ubuf_to_gm_align_{b8,b16,b32}`.
- `AscendC::` namespace (qualify, else it shadows / is unresolved): `TPosition`, `MatmulType`, `Matmul`, `TPipe`, `SyncAll`, `DataCopy`, `GlobalTensor`, `TBuf`, `TQue`, `HardEvent`.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Evidence
- A fused matmul-reduce kernel on V220/910C with CANN 9.0.0: qualifying format/config symbols (e.g. `AscendC::CFG_NORM`) failed while the unqualified global form compiled; the inverse held for the typed primitives. The `did you mean simply ...?` hint consistently pointed to global scope.

## Other instances (predicted)
Any AscendC kernel authoring on this SDK — the global-vs-namespace split bites whenever a symbol is taken from the "wrong" bucket. This is the third scope-flavored entry, but the only INCONSISTENT-EXPORT (symbol-at-one-scope-only) one; EC-6 and EC-69 are scope-COLLISION cases (a name that exists at two scopes, or a user name clashing with a global builtin).

Cross-ref: [[EC-6]] (`using namespace AscendC::Simt` → `GetBlockIdx` ambiguity — scope collision), [[EC-69]] (`AT`/`BT`/`CT`/`WT` collide with global builtin enums — scope collision); OL-267 is the missing-symbol-direction counterpart.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-267（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
