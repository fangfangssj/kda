---
type: Runbook
title: "Cube `MatmulImpl<MM_CFG=CFG_NORM>` rejected — `MatmulConfig` lacks `usedCoreNum/M/N/Ka/...` fields"
description: "Cube `MatmulImpl<MM_CFG=CFG_NORM>` rejected — `MatmulConfig` lacks `usedCoreNum/M/N/Ka/...` fields"
tags: [task.debugging, topic.programming_model.execution_unit.cube, topic.data_movement.data_copy, topic.data_movement, topic.tiling, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-39-cube-matmulimpl-mm-cfg-cfg-norm-rejected-matmulcon.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-39", "MatmulConfig"]
---

# Cube `MatmulImpl<MM_CFG=CFG_NORM>` rejected — `MatmulConfig` lacks `usedCoreNum/M/N/Ka/...` fields

> 原 OKF v1 卡号：**EC-39**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Symptom (compile error tail)**:
  ```
  matmul_utils.h:435:26: error: no member named 'usedCoreNum' in 'MatmulConfig'
      if constexpr (MM_CFG.usedCoreNum == -1) { ... }
  matmul_utils.h:438:26: error: no member named 'M' in 'MatmulConfig'
  matmul_utils.h:486:39: error: invalid operands ('const IterateOrder' and 'int')
      if constexpr (MM_CFG.iterateOrder == -1) { ... }
  ```
  …20+ such errors before `-ferror-limit` kicks in. The compiler is trying to compare `MatmulConfig` fields to `−1` for the constexpr-vs-GM tiling decision, but `MatmulConfig` does not have those fields — they live on `MatmulApiStaticTiling` (CANN 9.0.0 `include/adv_api/matmul/tiling.h:431`).
- **Trigger pattern**: building `MatmulImpl<AT, BT, CT, BIAS, MM_CFG = CFG_NORM>` and calling `mm.Init(__gm__ TCubeTiling*, TPipe*)` (or the non-`__gm__` overload).
- **Fix**: Use `MatmulApiStaticTiling` (a struct that wraps `MatmulConfig`) as `MM_CFG`:
  ```cpp
  static constexpr MatmulApiStaticTiling MM_CFG_RUNTIME = []() {
      MatmulApiStaticTiling t{};   // every shape field defaults to −1 → use runtime tiling
      t.cfg = CFG_NORM;            // plain MatmulConfig becomes the .cfg member
      return t;
  }();
  MatmulImpl<AT, BT, CT, /*BIAS=*/CT, MM_CFG_RUNTIME> mm;
  ```
  Setting individual fields (e.g. `t.baseM = 128`) makes those constexpr; leaving them −1 reads from runtime tiling. This is the Opt2 unlock — see OL-91 step 3.
- **Rejected workaround**: switching to `Init(const TCubeTiling*, TPipe*)` (non-`__gm__`) and copying tiling field-by-field per OL-77 is NOT necessary — `MatmulImpl::Init` has a dedicated `__gm__` overload that does the slice-copy internally. The actual issue is the `MM_CFG` type, not the `Init` overload.
- **Evidence**: 1_BatchMatmul (2026-04-28) Phase C iter 1 — first level-3 cube op hit this on first build. Now amortized across 4 cube ops (op#1/#4/#5/#3 all cite OL-91 step 3 in analysis.md and built clean from iter 0).

<!-- 迁移自 porter kb/target/ascendc/（EC-39，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
