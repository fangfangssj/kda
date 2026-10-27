---
type: "Runbook"
title: "aclnnQuantMatmulV5 L0C2out 三约束判别"
description: "Ascend950/CANN 9.2.0 上 aclnnQuantMatmulV5 L0C→out 直出路径（int8 输入+uint64 scale）有三条硬约束：bias 须 int32、offset 须打包进 uint64 scale 不可单独传、offset 与 pertokenScale 互斥；传错均返回 161002，offset 打包则引入精度损失。"
tags: ["task.debugging", "topic.compute.quantization", "topic.implementation.api_layer.c_api"]
status: stable
sources: []
created_at: "2026-08-18T15:55:16Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["offset_packing"]
---

# aclnnQuantMatmulV5 L0C2out 三约束判别

## 现象

在 Ascend950（DAV_3510/arch35，CANN 9.2.0）上 `bench_no_harness_quant_matmul` 以 L2 API 委托 `aclnnQuantMatmulV5`，把 FP16 输出的 fp32 scale 转为 uint64 后，ST 通过率从 35/101 升到 71/101，但剩余 30 个用例仍失败：18 个 `GetWorkspaceSize 失败: ret=161002`，12 个精度不匹配（`max_diff` 超阈）。

失败用例经 SLOG 探针揭示后归类：
- 带单独 offset（`x2Offset` 非空）的 FP16 输出用例：161002；
- 带 float bias（fp16/bf16/fp32）+ uint64 scale 的 L0C→out 用例：161002（"the dtype of bias must be DT_INT32"）；
- offset + pertokenScale 同时存在的用例：161002；
- 部分 offset 打包进 uint64 scale 的用例：精度漂移（offset 编码进 uint64 量化格式的精度损失）。

## 适用场景

- **工程形态**：aclnn L2 API 委托内置 `aclnnQuantMatmulV5`（op_api 侧转调）。
- **API 路径**：`aclnnQuantMatmulV5GetWorkspaceSize` 参数 `x2Offset`/`x2Bias`/`x1Scale(pertokenScale)`；offset 打包经 `aclnnTransQuantParam`。
- **验证平台与版本**：Ascend950（DAV_3510/arch35），CANN 9.2.0。frontmatter `platforms: [950]` 与本场景严格一致。
- **适用粒度**：`operator_family`（int8 量化 matmul 族走 V5 L0C→out 直出路径）。
- **路线优先级**：`safety_fallback`——uint64 scale + 打包 offset 是 FP16 输出在 L0C→out 路径唯一可用形态，但牺牲 offset 精度且不支持 float bias；bf16 输出走 L0C→UB 路径（任意 bias + offset）约束更宽，是首选，L0C→out 为 fallback。
- **不适用/未验证场景**：a2/a3 未验证，不得据此刻入 `platforms`；bf16 输出走 L0C→UB 路径不受此三约束；自定义 Ascend C kernel 路径不适用。

## 判别方法

判定一次 `aclnnQuantMatmulV5GetWorkspaceSize` 返回 `161002`（或剩余精度漂移）是否属"L0C→out 直出路径参数合同违反"。

1. **触发条件（trigger）**：在 950 / CANN 9.2.0 上以 int8 输入 + uint64 scale 调 `aclnnQuantMatmulV5GetWorkspaceSize`，输出 dtype 在 int8 之外（如 FP16）；调用返回 `161002`，或打包 offset 后 `max_diff` 超阈。原始报错：
   ```
   GetWorkspaceSize 失败: ret=161002
   ```

2. **可执行探针（probe）**：在算子 ST 目录开 SLOG 抓路径 checker 信息：
   ```bash
   cd <operator_dir>/tests/st && ASCEND_SLOG_PRINT_TO_STDOUT=1 bash run.sh --real 2>&1 | grep -E "L0_012|L0_010|L0_015|L1_061" -A2 | head -20
   ```

3. **原始观测（observation）**：SLOG 输出三条约束原文（逐字）：
   - 约束 1——L0C→out 路径 bias 必须 int32：
     ```
     the dtype of bias must be DT_INT32
     ```
   - 约束 2——out 在 int8 之外时 offset 不可单独传（须打包进 uint64 scale）：
     ```
     when x1 and x2 are INT8 and the dtype of out is outside INT8, x2Offset(offset) must be null
     ```
   - 约束 3——offset 与 pertokenScale 互斥：
     ```
     when x1 and x2 are INT8 with x1Scale(pertokenScale), x2Offset(offset) must be null
     ```

4. **解释（interpretation）**：`aclnnQuantMatmulV5` 的 L0C→out 直出路径（int8 输入、uint64 scale、输出 dtype 在 int8 之外）是硬件直出量化路径，其参数合同为：
   - **bias 仅接受 int32**——float bias（fp16/bf16/fp32）在该路径不支持；float bias 场景须走 L0C→UB 路径（bf16 输出）或拆分计算（V5 用 int32 bias/空 bias 算 L0C→out 中间结果，再用 `aclnnAdds`/`aclnnAdd` 补 float bias）。
   - **offset 不可单独传**——offset 必须由 `aclnnTransQuantParam` 与 scale 一起打包进 uint64 量化参数，`x2Offset` 传非空即 161002；但打包会引入精度损失（uint64 量化格式的 offset 编码精度有限，对应 12 个精度漂移用例）。
   - **offset 与 pertoken 互斥**——带 pertokenScale 时 offset 必须为 null；offset+pertoken 组合在 L0C→out 路径无硬件支持。
   三条约束同源：均来自 L0C→out 直出路径的 `CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input` 系列 checker，是芯片路径级硬约束，非参数范围可绕过。

## 验证信号

本卡证据来自 `bench_no_harness_quant_matmul` 算子在 Ascend950 上的 V5 委托开发会话；scale 转 uint64 后用同一 ST 套件实测。

- scale 转 uint64 后 ST 71/101，剩余 30 个失败（含 18 个 `GetWorkspaceSize 失败: ret=161002` 与 12 个 `max_diff` 超阈精度漂移）（来源平台：950；证据等级：tool_observed）
- `ASCEND_SLOG_PRINT_TO_STDOUT=1` 探针输出三条约束原文（`the dtype of bias must be DT_INT32` / `x2Offset(offset) must be null` (out outside INT8) / `x2Offset(offset) must be null` (with pertokenScale)）（来源平台：950；证据等级：tool_observed）

## 失效边界

- a2/a3 未验证，不得刻入 `platforms`；V5 为内置算子，路径约束可能随芯片代际变化，上线前须以 SLOG 探针复核。
- 三条约束仅适用于 L0C→out 直出路径；bf16 输出走 L0C→UB 不受约束。
- offset 打包进 uint64 的精度边界未单独量化（12 个精度失败用例与此相关，编码损失范围未测定）。

## 坏实践

- 把 offset 单独作为 `x2Offset` 传给 V5 的 L0C→out 路径——必然 161002。
- 给 L0C→out 路径传 float bias（fp16/bf16/fp32）——必然 161002。
- offset + pertokenScale 同时传 L0C→out 路径——必然 161002。
- 把 offset 打包进 uint64 scale 后忽略精度损失——12 个用例精度漂移即此后果。