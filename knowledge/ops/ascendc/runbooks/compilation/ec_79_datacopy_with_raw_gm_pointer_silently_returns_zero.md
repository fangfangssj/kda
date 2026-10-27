---
type: Runbook
title: "DataCopy with raw `__gm__` pointer silently returns zeros on Ascend950PR pure-AIV kernels"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (any pure-AIV SIMD kernel using DataCopy for GM->UB reads)"
tags: [task.debugging, topic.precision, topic.implementation.synchronization, topic.programming_model.execution_model.simt, topic.data_movement.data_copy, topic.data_movement]
status: stable
sources:
  - resource: https://gitcode.com/cann/cannbot-skills/blob/bb96c9c595c7a8e96745d365291769dc48319c22/plugins-community/ascendc-port-orchestrator/kb/okf/runbooks/field-notes/build/ec-79-datacopy-with-raw-gm-pointer-silently-returns-zero.md
    role: primary
created_at: "2026-07-09T16:00:00Z"
updated_at: "2026-07-09T16:00:00Z"
platforms: ["950"]
aliases: ["EC-79", "__gm__"]
---

# DataCopy with raw `__gm__` pointer silently returns zeros on Ascend950PR pure-AIV kernels

> 原 OKF v1 卡号：**EC-79**。
> 现象分类：build_failure；观测置信度：single_run（单次观测）。
> 触发信号：
> - DataCopy(local, reinterpret_cast<__gm__ float>(gm_addr), byte_count) compiles cleanly (no warning, no error) but at runtime the destination local tensor contain

## 适用场景

- 适用技术与问题域：AscendC 编译/构建问题（现象分类：build_failure）。
- 适用平台：Ascend 950 系列；本卡为单次观测沉淀（single_run），按卡内声明范围消费。

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (any pure-AIV SIMD kernel using DataCopy for GM->UB reads)`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 — GM->UB DataCopy with raw pointers on V220 unconfirmed)`

- **Error pattern**: `DataCopy(local, reinterpret_cast<__gm__ float*>(gm_addr), byte_count)` compiles cleanly (no warning, no error) but at runtime the destination `local` tensor contains **all zeros** — the DataCopy reads zero from GM instead of the actual tensor data. All downstream computation produces zero/near-zero output. No runtime error, no 507035 — silent data corruption.
- **Root cause**: On Ascend950PR, `DataCopy` (the high-level AscendC API) expects a `GlobalTensor<T>` (or `LocalTensor<T>` for UB->UB) as the source/destination argument, not a raw `__gm__ T*` pointer. When passed a `reinterpret_cast<__gm__ T*>(gm_addr)`, the template resolves to a path that does not actually perform the GM->UB DMA — it produces zero-filled output without any diagnostic. The raw `__gm__` pointer path works in SIMT VF kernels (scalar pipe) but fails silently in pure-AIV SIMD class kernels.
- **Fix**: Use `GlobalTensor<T>` with `SetGlobalBuffer` + `operator[]` for all GM addressing in pure-AIV kernels:
  ```cpp
  // BEFORE (silently returns zeros):
  __gm__ float* gm_ptr = reinterpret_cast<__gm__ float*>(gm_addr);
  DataCopy(local, gm_ptr, N);  // compiles, reads zeros

  // AFTER (works correctly):
  GlobalTensor<float> gm_tensor;
  gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_addr));
  DataCopy(local, gm_tensor[offset], N);  // correct GM->UB DMA
  ```
  This is the proven pattern from FusedAddRmsnorm and other pure-AIV class kernels.
- **Distinct from PB-20**: PB-20 covers GM WRITE paths (`GlobalTensor::SetValue` silent no-op + raw `__gm__` pointer writes failing in pure-AIV). EC-79 covers the GM->UB READ path via `DataCopy` with a raw pointer — different API call, different direction, same root class (raw pointer vs GlobalTensor).
- **Detection**: kernel compiles and produces **all-zero output across ALL cases** (not just edge cases). If the entire output tensor is zero/epsilon and the kernel uses `DataCopy` with `reinterpret_cast<__gm__ T*>` arguments, suspect this first. A one-line probe: replace one `DataCopy(local, raw_gm_ptr, N)` with `DataCopy(local, gm_tensor[0], N)` and check if output becomes non-zero.
- **Evidence**: add_rms_norm_quant (2026-06-23, Ascend950PR_9579, CANN 9.0.0): aog-precision-probe iter 0 — kernel produced all-zero x_out and y1 across all 196 cases. Switching to GlobalTensor + SetGlobalBuffer + operator[] fixed immediately. FusedAddRmsnorm (earlier) used the GlobalTensor pattern from the start — no such issue.
- **Cross-reference**: PB-20 (GlobalTensor::SetValue silent no-op — same raw-pointer-vs-GlobalTensor class but WRITE direction), OL-77 (byte-copy loop workaround for reinterpret_cast GM->non-GM — adjacent API-surface mismatch).

<!-- 迁移自 porter kb/target/ascendc/（EC-79，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
