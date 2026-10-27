---
type: "Runbook"
title: "TQue 重构不修复 at::zeros 跨 stream 覆盖"
description: "把 rms_norm_rope_k_store_kv_cache 的 TBuf+手动 fence 改成 TQue（bufNum=1/输入 double/全 double）4 个方案都没修复 kv_a_out new 区整行 0；全 double 反触发 507035 UB crash。根因在 host 侧 at::zeros 跨 stream 覆盖，不在 kernel 流水同步——把 TQue 重构当修复手段是 banned 路线。"
tags: ["task.debugging", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-18T07:45:35Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["DOUBLE_BUFFER"]
---

# TQue 重构不修复 at::zeros 跨 stream 覆盖

## 现象

`rms_norm_rope_k_store_kv_cache`（dav-3510 / Ascend950 / CANN 9.1.0）在批量 UT 下 `kv_a_out` new 区整行为 0，非确定性，bs=96 场景 `zero_rows=202447`，cache 断言始终通过。

初始假设是"kernel 内 TBuf + 手动 `SetFlag/WaitFlag`（V_MTE2 / MTE3_V / MTE2_V fence）在并发压力下偶发写丢失"。参考算子 `fused_k_rms_norm_rope_store_kv_cache_mx_quant` 用 TQue（`AllocTensor/EnQue/DeQue/FreeTensor` 自动管理 MTE2→V / V→MTE3 同步），于是把 `Process()` 的输入/输出 buffer 从 TBuf 改 TQue 作为"用重构治 bug"的尝试。

## 适用场景

- **工程形态**：kernel 直调，`Process()` 内 MTE2 加载（GatherCopyIn）→ V 计算（DoRmsNorm512 / DoRope64）→ MTE3 写出（DataCopyPad + StorePaCache），跨迭代有 WAR hazard。
- **API 路径**：`TQue<QuePosition::VECIN/VECOUT, bufNum>` + `AllocTensor/EnQue/DeQue/FreeTensor`；对照 `TBuf<TPosition::VECCALC>` + 手动 `SetFlag/WaitFlag`。
- **验证平台与版本**：Ascend950（dav-3510），CANN 9.1.0。仅本平台实测，`platforms: [950]`。
- **适用粒度**：`operator_family`——任何"输出 new 区整行 0、非确定性、仅批量触发、cache 断言通过"的 kernel 直调算子，在考虑用 TQue 重构修复前都适用本结论。
- **路线优先级**：`banned`——把"TBuf→TQue 重构"当作修复 kv_a_out 写丢失的手段是 banned；此类现象的根因在 host 侧输出分配 stream 与 kernel launch stream 不同步（`at::zeros` 异步零初始化跨 stream 覆盖 kernel 写出），不在 kernel 流水 fence。
- **不适用/未验证场景**：a2 / a3 未验证（推断可迁移性低，507035 的 UB 阈值平台相关）；本卡只证伪"TQue 重构修复本 bug"这一假设，不证伪"TQue 在其他场景下优于 TBuf"。

## 失败路线

> 来源：本 trace 在 Ascend950 + CANN 9.1.0 上对 `rms_norm_rope_k_store_kv_cache` 的 4 次 TQue 重构尝试，均未修复，全部回退。

**banned 路线：把 `Process()` 内 TBuf+手动 `SetFlag/WaitFlag` 改成 TQue（`AllocTensor/EnQue/DeQue/FreeTensor` 自动同步），意图修复 `kv_a_out` new 区整行 0 的写丢失。**

4 个 bufNum 方案逐个尝试，均未修复，全部回退：

1. **TQue bufNum=1**（输入/输出 queue 都单 buffer）：12 failed / 50 passed。性能相对 TBuf 版提升（耗时不升反降），但 bug 仍在。假设"bufNum=1 时 TQue 的 AllocTensor 等 FreeTensor，但不自动处理 V→MTE2 跨迭代同步"未得到证伪支撑——bug 不在跨迭代同步。
2. **TQue 输入 double buffer**（`inQueueKv_, inQueueSinCos_` bufNum=2，输出仍 single）：12 failed。double buffer 反而变慢，UB 受压。对应的成员声明改动（verbatim，from trace Edit）：

   before（TQue bufNum=1 声明）：
   ```cpp
       TQue<QuePosition::VECIN, 1> inQueueKv_, inQueueSinCos_, gammaQueue_;
       TQue<QuePosition::VECOUT, 1> outQueueNorm_, outQueueRope_;
   ```
   after（输入改 double）：
   ```cpp
       TQue<QuePosition::VECIN, 2> inQueueKv_, inQueueSinCos_;
       TQue<QuePosition::VECIN, 1> gammaQueue_;
       TQue<QuePosition::VECOUT, 1> outQueueNorm_, outQueueRope_;
   ```

3. **TQue 全 double**（输入 + 输出 queue 都 bufNum=2）：62 failed，硬件错误 **`507035`**（UB 溢出 crash）。UB 占用接近 192KB 上限，全 double 直接超。
4. **TQue 后再加诊断**：`zero_rows=254244 hist_zr=0 new_zr=254244`——0 行数量比 TBuf 版（202447）**更大**，且 100% 在 new 区。重构不仅没修 bug，new 区零行数反增。

`Process()` 内循环体 TQue 化的完整 Edit（`AllocTensor/EnQue/DeQue/FreeTensor` 替换手动 fence）内容被转写截断（`_see below_`），无法 verbatim 恢复；但失败信号与 banned 结论由上述 4 组实测数据充分支撑。真正修复来自 host 侧 `allocate_xpu_zeros_like` → `allocate_xpu_tensor_like`（`at::zeros`→`at::empty`，消除跨 stream 零初始化竞争），属 host 侧 `op_host` 层而非本卡 `kernel` 层。

## 验证信号

本段只收本 trace 内 `tool_observed` 实测；agent 推断的机制假设不进此段。

- TQue bufNum=1：12 failed / 50 passed（来源平台：950；证据等级：tool_observed）。
- TQue 输入 double buffer（`inQueueKv_/inQueueSinCos_` bufNum=2）：12 failed（来源平台：950；证据等级：tool_observed）。
- TQue 全 double（输入+输出都 bufNum=2）：62 failed，error 507035（UB crash）（来源平台：950；证据等级：tool_observed）。
- TQue 重构后 BUGDIAG：`zero_rows=254244 hist_zr=0 new_zr=254244`，new 区零行数比 TBuf 版（202447）更大（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡只证伪"TQue 重构修复本 bug"这一假设；不证伪"TQue 在其他场景下优于 TBuf"。若算子确实存在跨迭代 WAR hazard（确定性、边界 token 丢失、replay 可复现），TQue + 合理 bufNum 仍是正当方案——本 trace 的 bug 不属此类。
- `platforms` 仅 950（dav-3510），a2/a3 未验证；507035 的 UB 阈值在不同平台不同，全 double 的可行性边界平台相关。推断可迁移性低，不写入 `platforms`。
- 把整段算子包进 TQue + 合理 bufNum 后是否真能修复 host 侧 `at::zeros` 跨 stream 覆盖，未在本 trace 验证（根因在 host 侧，不在 kernel 流水），不作为本卡结论。

## 坏实践

- 未先定位根因就大改同步框架（TBuf→TQue），把性能/重构工作和 bug 修复混在一起，结果既没修 bug 又引入 507035 UB crash。
- 全 double buffer 不先算 UB 占用直接上，触发 507035。
- 把"TQue 重构"当成"治一切写丢失 bug 的银弹"。TQue 自动管理的是**单次迭代内** MTE2→V→MTE3 同步；它不会修复 host 侧 GM 写入竞争。
- 看到"输出 new 区整行 0 + 非确定性 + 仅批量触发 + cache/同类写出路径正常"时，直接归因 kernel 流水 fence 而不先查 host 侧输出 tensor 分配 stream 与 kernel launch stream 是否同步。