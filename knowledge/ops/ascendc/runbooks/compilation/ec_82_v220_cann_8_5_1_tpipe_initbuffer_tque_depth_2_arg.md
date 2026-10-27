---
type: Runbook
title: "V220/CANN 8.5.1 `TPipe::InitBuffer(TQue&, depth)` 2-arg form does not compile — use the 3-arg `(TQue&, depth, buf_size_bytes)` form"
description: "applies_to: soc=Ascend910_V220; cann=8.5.1; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-82-v220-cann-8-5-1-tpipe-initbuffer-tque-depth-2-arg.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: [a2, a3]
aliases: ["EC-82"]
---

# V220/CANN 8.5.1 `TPipe::InitBuffer(TQue&, depth)` 2-arg form does not compile — use the 3-arg `(TQue&, depth, buf_size_bytes)` form

> 原 OKF v1 卡号：**EC-82**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - build error no matching member function for call to 'InitBuffer' on a pipe->InitBuffer(que, depth) (2-arg) call that compiles fine on newer CANN.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Atlas A2 系列、Atlas A3 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend910_V220; cann=8.5.1; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend910B2C (V220); cann=8.5.1`

- **Symptom**: build error `no matching member function for call to 'InitBuffer'` on a `pipe->InitBuffer(que, depth)` (2-arg) call that compiles fine on newer CANN.
- **Fix**: pass the explicit buffer size as the third argument:
  ```cpp
  // BAD on V220/CANN 8.5.1 — 2-arg overload absent
  pipe->InitBuffer(inQueue_, PIPE_DEPTH);
  // GOOD — 3-arg form
  pipe->InitBuffer(inQueue_, PIPE_DEPTH, TILE_ELEMS * sizeof(float));
  ```
- **Scope**: distinct from EC-62 (TBuf with NO `InitBuffer` at all → 507035 vector core exception). Here the call IS present; only the arg count is wrong. The 2-arg overload may exist on newer CANN — this is the V220/8.5.1 fallback. Applies to `TQue` buffers; `TBuf` workspace uses the 2-arg `(name, size_bytes)` form as usual (EC-62).
- **Detection**: build log shows `no matching member function for call to 'InitBuffer'` pointing at a `InitBuffer(<TQue>, <int>)` line. Fix by adding the byte-size third arg.
- **Evidence**: 1_GELU kw-1 re-spawn (2026-06-23, Ascend910B2C V220, CANN 8.5.1): the 2-arg form was rejected by the compiler; switching to the 3-arg form cleared the build.
- **Cross-ref**: EC-62 (missing InitBuffer → 507035), OL-63 (TQue depth=4 for elementwise bandwidth), P-P28 (TQue depth + explicit buf-size pairing).

<!-- 迁移自 porter kb/target/ascendc/（EC-82，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
