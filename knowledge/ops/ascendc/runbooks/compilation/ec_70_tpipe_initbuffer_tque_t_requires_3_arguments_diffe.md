---
type: Runbook
title: "`TPipe::InitBuffer(TQue<T>&)` requires 3 arguments — different from TBuf InitBuffer which takes 2"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all"
tags: [task.debugging, topic.implementation.synchronization, topic.data_movement, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-70-tpipe-initbuffer-tque-t-requires-3-arguments-diffe.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-70"]
---

# `TPipe::InitBuffer(TQue<T>&)` requires 3 arguments — different from TBuf InitBuffer which takes 2

> 原 OKF v1 卡号：**EC-70**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - compile error when calling pipe_.InitBuffer(tQue) with only 2 arguments. Error message: too few arguments to function call, expected 3 or similar.

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A3 chip family — CANN version may differ)`

- **Symptom**: compile error when calling `pipe_.InitBuffer(tQue)` with only 2 arguments. Error message: `too few arguments to function call, expected 3` or similar.

- **Root cause**: `TPipe::InitBuffer` has different signatures for TQue vs TBuf:
  - `InitBuffer(TBuf<T>&, uint8_t num_buffers)` — 2 args
  - `InitBuffer(TQue<T>&, uint8_t num_buffers, uint32_t len_per_buffer)` — 3 args

  The TQue variant requires an explicit per-buffer element length (in bytes) as the third argument. Workers familiar with the TBuf 2-arg form naturally write the TQue form with 2 args → compile fails.

- **Fix**: always provide the third argument for TQue InitBuffer. The `len_per_buffer` is the byte size of one buffer slot: typically `tileLength * sizeof(dtype)`.

```cpp
// TBuf — 2 args (OK)
pipe_.InitBuffer(tBuf, depth);

// TQue — 3 args REQUIRED
pipe_.InitBuffer(tQue, depth, tileLength * sizeof(half));
```

- **Detection**: grep for `InitBuffer(` calls near TQue variable declarations. If the call has exactly 2 args and the first arg is a `TQue<...>`, the third arg is missing.

- **Evidence**:
  - fused_quant_mat_mul kw-1 (2026-06-15): `kernel/fused_quant_mat_mul_kernels.cpp:69-70` — TQue InitBuffer called with 2 args. Added `BUF_LEN * sizeof(half)` as third arg → compile passed.

<!-- 迁移自 porter kb/target/ascendc/（EC-70，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
