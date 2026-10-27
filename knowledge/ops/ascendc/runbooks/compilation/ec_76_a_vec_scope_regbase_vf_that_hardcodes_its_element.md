---
type: Runbook
title: "A `__VEC_SCOPE__` regbase VF that hardcodes its element/tile count to a fixed inner-dim constant produces garbage above that constant — derive the count from the RUNTIME inner dim"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all"
tags: [task.debugging, topic.implementation.masking, topic.data_movement, topic.memory, topic.compilation]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-76-a-vec-scope-regbase-vf-that-hardcodes-its-element.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-76", "__VEC_SCOPE__"]
---

# A `__VEC_SCOPE__` regbase VF that hardcodes its element/tile count to a fixed inner-dim constant produces garbage above that constant — derive the count from the RUNTIME inner dim

> 原 OKF v1 卡号：**EC-76**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all`

A regbase MicroAPI VF that processes a `[outer, inner]` chunk must compute its loop/tile count from the **runtime** inner dim, not a value that happened to equal the inner dim on the shape it was first written for. A VF written for a fixed `inner=16` and coded as `ntFull = ceil(outer*16 / VL)` only covers the first 16 inner-lanes; for `inner>16` the upper `outer*(inner-16)` elements are NEVER written → the consumer reads uninitialized UB → garbage output. The bug is **masked whenever runtime inner == the hardcoded constant** (e.g. the original `dstate=16` customer), so it ships silently and only surfaces when a later caller uses a larger inner dim.

**Concrete anchor** (selective_scan fwd regbase build/prodC VFs): `uint16_t ntFull = (uint16_t)((cl * N + VLf - 1) / VLf);` — `N` is the runtime `dstate`, NOT a literal `16`. Tail over-process (when `cl*N` is not a multiple of VL=64) only touches the +64 buffer padding → identical semantics.

**Evidence**: selective_scan fwd-SIMD (2026-06-24, PR #52). `cl*16` → N=32 upper half garbage; `cl*N` → N∈{8,16,24,32,48,64} all 0-wrong at dtype floor. N=16 (customer) was correct either way (16==N).

**Other instances (predicted)**: any regbase/MicroAPI VF parameterized over a runtime inner/state/head dim — attention head-dim VFs, normalization feature-dim VFs, any `[L, D]` chunk VF — where an early single-shape author bakes the first D as a literal.

<!-- 迁移自 porter kb/target/ascendc/（EC-76，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
