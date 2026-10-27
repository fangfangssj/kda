---
type: Runbook
title: "`const LocalTensor<T>` variable blocks Sort / write-through VEC API overload resolution"
description: "`const LocalTensor<T>` variable blocks Sort / write-through VEC API overload resolution"
tags: [task.debugging, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-34-const-localtensor-t-variable-blocks-sort-write-thr.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-34"]
---

# `const LocalTensor<T>` variable blocks Sort / write-through VEC API overload resolution

> 原 OKF v1 卡号：**EC-34**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Compile error no matching function for call to 'Sort' ... 1st argument ('const LocalTensor<float>') would lose const qualifier when the caller declared destinat

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**: Compile error `no matching function for call to 'Sort' ... 1st argument ('const LocalTensor<float>') would lose const qualifier` when the caller declared destination tensor locals as `const LocalTensor<float> x = ...;` (a common "value hint" pattern from earlier kernels).
- **Root cause**: Hardware `Sort` (and most write-through VEC APIs: `Cast`, `Duplicate`, `Adds`, `Muls`, `Select`, etc.) take `LocalTensor<T>&` (non-const) for their destination parameter because the API writes to them. A `const LocalTensor` cannot bind to the non-const reference.
- **Fix**: Drop `const` from any LocalTensor variable that will be passed as a destination to a write-through VEC API. `LocalTensor<T>` is a view/handle type with copy semantics, so dropping `const` does not weaken safety meaningfully.
- **Detection**: Compile error at the API call site, blaming "const qualifier" or "argument would lose const". Check the variable declaration, not the API signature.
- **Related**: general C++ overload resolution rule; applies to any write-through `LocalTensor` API, not just Sort. Documented here because it's a common newcomer gotcha when porting patterns from pure read-only kernels.
- **Evidence**: 9_TopKTopP cold-run round 2, Phase C iter 1 (2026-04-18). Dropped const → compile OK.
- **Status**: Low-severity general-C++ gotcha. Useful as preventive guidance in Phase B for kernels using hardware Sort / write-through APIs.

<!-- 迁移自 porter kb/target/ascendc/（EC-34，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
