---
type: Runbook
title: "DataCopy alignment overwrite in strided/chunked copies"
description: "DataCopy alignment overwrite in strided/chunked copies"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_model.simt, topic.implementation.atomic, topic.data_movement.data_copy]
status: draft
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-16-datacopy-alignment-overwrite-in-strided-chunked-co.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-09-16T07:35:48Z"
platforms: []
aliases: ["EC-16"]
---

# DataCopy alignment overwrite in strided/chunked copies

> 原 OKF v1 卡号：**EC-16**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - Precision failures on specific test cases where non-aligned chunk sizes cause data corruption in adjacent output regions

- **Error pattern**: Precision failures on specific test cases where non-aligned chunk sizes cause data corruption in adjacent output regions
- **Root cause**: DataCopy requires 32-byte aligned element counts. When `chunk_size % ALIGN != 0`, naively aligning up writes extra elements past the chunk boundary, corrupting adjacent tensor data.
- **Fix (overlapping tail write)**:
  ```
  1. Copy floor_aligned(chunk) elements normally
  2. Copy last ALIGN elements starting at (chunk - ALIGN), overlapping with already-written region
  ```
  The overlap is harmless (same values re-written), and tail elements are placed correctly without overflow.
- **Condition**: Strided/chunked DMA with non-aligned chunk boundaries (e.g., cat along non-last dim)
- **Evidence**: Cat V1 failed 3/51 cases, fixed in V2 (2026-04-09)

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：平台待核验（draft）；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

## Quick Lookup Table

| EC | Error keyword | One-line fix |
|----|--------------|--------------|
| EC-1 | `calling a __host__ function from __aicore__` | Add `__aicore__ inline` to helper |
| EC-2 | `cannot initialize '__gm__ T*' with 'GM_ADDR'` | `reinterpret_cast<__gm__ T*>(gm_addr)` |
| EC-3 | `LAUNCH_BOUND exceeds maximum` | Reduce to 512 |
| EC-4 | `redefinition of 'blockDim'` | Guard with `#if defined(ASCENDC_CPU_DEBUG)` |
| EC-5 | `not support bf16 type cast` | Use `simt_to_float()` bit-manipulation (P-P27) |
| EC-6 | `call to 'GetBlockIdx' is ambiguous` | Remove `using namespace AscendC::Simt;` (OL-14) |
| EC-7 | `no member 'atomicAdd' in 'Simt'` | Use unqualified `atomicAdd()` (global built-in) |
| EC-8 | `unknown type name 'GM_ADDR'` | Add `#include <kernel_operator.h>` as first include |
| EC-9 | `redefinition of 'ITER'` / missing symbols | Wrap all code in `namespace ascendc_ops {}` |
| EC-10 | `undefined reference to 'aclrtlaunch_'` | Add `extern "C" {}` around declaration |
| EC-11 | `merge_mix_obj.sh Error 1` at 95% | Add `-DCMAKE_BUILD_TYPE=Release` |
| EC-12 | `cannot initialize 'int64_t (*)(void)'` + `expanded from macro 'block_num'` | Rename param: `blk_idx`/`blk_cnt` |
| EC-13 | `no member named 'SyncFunc' in namespace 'AscendC'` | Use `SetFlag`/`WaitFlag` with `FetchEventID` |
| EC-14 | `static assertion failed: must use AllocTensor...depth is zero` | Change TQue depth from 0 to ≥1 |
| EC-15 | `the range of 1st parameter must be [4, 6]` | No `PipeBarrier<PIPE_S>`, use SetFlag/WaitFlag for S pipe |
| EC-16 | Non-aligned chunk DataCopy corrupts adjacent data | Overlapping tail write: copy last ALIGN elems separately |
| EC-17 | Sub-align chunk overwrite in compact output | nblk=1 + padded alloc + narrow view |

<!-- 迁移自 porter kb/target/ascendc/（EC-16，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
