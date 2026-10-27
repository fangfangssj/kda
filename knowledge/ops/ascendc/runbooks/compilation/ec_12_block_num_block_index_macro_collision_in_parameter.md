---
type: Runbook
title: "`block_num` / `block_index` macro collision in parameter names"
description: "`block_num` / `block_index` macro collision in parameter names"
tags: [task.debugging, topic.data_movement, topic.compilation]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-12-block-num-block-index-macro-collision-in-parameter.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-12", "block_num", "block_index"]
---

# `block_num` / `block_index` macro collision in parameter names

> 原 OKF v1 卡号：**EC-12**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: cannot initialize a parameter of type 'int64_t (*)(void)' with an rvalue of type 'int64_t'
  note: expanded from macro 'block_num'
  #define block_num get_block_num()
  ```
- **Root cause**: CANN defines `block_num` as a macro expanding to `get_block_num()` (a function). When used as a function parameter name, `int64_t block_num` becomes `int64_t get_block_num()` -- a function declaration, not a parameter. Similarly, `block_index` may collide with other CANN macros.
- **Fix**: Rename parameters to avoid CANN macro names:
  ```cpp
  // BEFORE (fails):
  void Init(GM_ADDR x, int64_t block_index, int64_t block_num) { ... }

  // AFTER (compiles):
  void Init(GM_ADDR x, int64_t blk_idx, int64_t blk_cnt) { ... }
  ```
- **CANN macros to avoid as identifiers**: `block_num`, `block_idx`, and any other identifier in `__clang_cce_aicore_builtin_vars.h`.
- **Related**: OL-14 (namespace ambiguity)

<!-- 迁移自 porter kb/target/ascendc/（EC-12，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
