---
type: Runbook
title: "`__VEC_SCOPE__` for-loop induction variable MUST be `uint16_t`"
description: "`__VEC_SCOPE__` for-loop induction variable MUST be `uint16_t`"
tags: [task.debugging, topic.implementation.masking, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-67-vec-scope-for-loop-induction-variable-must-be-uint.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:00Z"
platforms: ["950"]
aliases: ["EC-67", "__VEC_SCOPE__", "uint16_t"]
---

# `__VEC_SCOPE__` for-loop induction variable MUST be `uint16_t`

> 原 OKF v1 卡号：**EC-67**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - bisheng emits compile error inside __VEC_SCOPE__ { ... } block when the for-loop induction variable is anything other than uint16_t:

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A2 系列、Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

```yaml
applies_to:
  paradigm: ascendc
  arch_family: arch35
  bisheng: 2026-03-21+
```

- **Empirical anchor**: HW probe sub-agent `acd7700cc8182c637` 2026-05-28 20:14Z on Ascend950PR_9579 / arch35.

- **Symptom**: bisheng emits compile error inside `__VEC_SCOPE__ { ... }` block when the for-loop induction variable is anything other than `uint16_t`:
  ```
  regbase_probe_kernel.cpp:69:18: error: Induction variable must have a type uint16_t. Example:
   (uint16_t $var=0; $var< bound; $var++))
              for (int32_t i = 0; i < loops; ++i) {
  ```

- **Root cause**: bisheng's BuiltIn-API legality checker on the SIMD VF surface treats the loop counter as a `MaskReg`/`AddrReg` index source. `RegTensor` / `MaskReg` index registers are 16-bit by hardware design. Wider integer types fail the legality check.

- **Fix**: rewrite the loop to use `uint16_t`. Example:
  ```cpp
  // BAD — bisheng rejects
  __VEC_SCOPE__ {
      for (int32_t i = 0; i < repeatTimes; ++i) { /* ... */ }
  }

  // GOOD — bisheng accepts
  __VEC_SCOPE__ {
      for (uint16_t i = 0; i < repeatTimes; ++i) { /* ... */ }
  }
  ```
  The induction-variable rule applies ONLY inside `__VEC_SCOPE__`. Regular `__aicore__` code outside the scope can use any int type.

- **Scope qualifier**: bisheng version sealed at 2026-03-21 build; future bisheng may relax the constraint. Applies to arch35 / V351x / Ascend950PR family only — V351x (Atlas 200I/500 A2) does not surface `__VEC_SCOPE__` per its arch spec doc.

- **Detection (worker / translator emit-time)**: grep emitted kernel for `__VEC_SCOPE__` block, then for any `for (` line inside it. If the induction declaration is not exactly `uint16_t`, fail the local emit-time check.

- **Cross-ref**: OL-196 (Membase vs Regbase + `__VEC_SCOPE__` programming entry).

<!-- 迁移自 porter kb/target/ascendc/（EC-67，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
