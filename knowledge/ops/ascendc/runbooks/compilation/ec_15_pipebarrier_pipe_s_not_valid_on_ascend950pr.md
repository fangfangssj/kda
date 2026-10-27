---
type: Runbook
title: "`PipeBarrier<PIPE_S>` not valid on Ascend950PR"
description: "`PipeBarrier<PIPE_S>` not valid on Ascend950PR"
tags: [task.debugging, topic.implementation.synchronization, topic.programming_model.execution_unit.vector, topic.implementation.masking, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-15-pipebarrier-pipe-s-not-valid-on-ascend950pr.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: ["950"]
aliases: ["EC-15", "PipeBarrier<PIPE_S>"]
---

# `PipeBarrier<PIPE_S>` not valid on Ascend950PR

> 原 OKF v1 卡号：**EC-15**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

- **Error pattern**:
  ```
  error: the range of 1st parameter must be [4, 6]
  ```
  (from `kernel_reg.h`, triggered by `PipeBarrier<PIPE_S>()`)
- **Root cause**: On Ascend950PR, `pipe_barrier()` only accepts pipe values 4 (PIPE_MTE2), 5 (PIPE_V), 6 (PIPE_MTE3). The scalar pipe (PIPE_S) is not supported for PipeBarrier. To synchronize the scalar pipe, use `SetFlag`/`WaitFlag` with appropriate event types.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::PipeBarrier<PIPE_S>();

  // AFTER (S→MTE3 sync for scalar writes visible to MTE3 output):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(ev);

  // For S→V sync (scalar writes visible to VEC):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev);
  ```
- **Valid PipeBarrier pipes**: PIPE_MTE2 (4), PIPE_V (5), PIPE_MTE3 (6) — that's it.
- **Evidence**:
  - Sort V1 build failure (2026-04-09)
  - clipped_swiglu port_a3_to_a5 kw-1 (2026-05-17): scalar→vector gather pattern (interleaved A/B half extraction via `LocalTensor::SetValue(i, …)` feeding subsequent `Cast` / `Mins` / `Maxs`) tripped `kernel_reg.h:85` "range of 1st parameter must be [2,6],[10,10]" on AIC and "[4,6]" on AIV. Fixed in iter 2 by replacing `PipeBarrier<PIPE_S>()` with `SetFlag/WaitFlag<HardEvent::S_V>` at two sync sites (F32 interleaved branch + Half interleaved branch). 8/8 cases PASS post-fix. General trigger class: any kernel that scalar-gathers a strided/masked pattern (even/odd interleave, group indexing) into UB that subsequent VEC ops read — common in SwiGLU-family fused activations and small-shape Gather helpers.

<!-- 迁移自 porter kb/target/ascendc/（EC-15，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
