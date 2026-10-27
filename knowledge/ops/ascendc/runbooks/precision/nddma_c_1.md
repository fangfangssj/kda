---
type: "Runbook"
title: "NDDMA 快速路径 C=1 中间轴漏维判别"
description: "bn_training_update 的 NDDMA 快速路径以 isNHWC_=(chanStride_==1)||(cTotal==1) 判通道末轴，C=1 广播且 channel 在中间轴时误判为末轴，NDDMA dim0 loopSrcStride=1 假设 channel 在末轴致漏维，表现为 rank5 (5,1,3,4,4) C=1 sq=0 的 y 输出 FAIL（fp32 mere≈0.557、fp16 NaN/Inf），同输入 2D C=1 用例 PASS。判别=核 chanStride_==1 而非 cTotal==1。"
tags: ["task.debugging", "topic.precision", "topic.data_movement.data_copy", "topic.alignment"]
status: stable
sources: []
created_at: "2026-08-18T07:22:43Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# NDDMA 快速路径 C=1 中间轴漏维判别

## 现象

`bn_training_update` 算子精度调试会话中，blackbox TTK 用例 shape `(5,1,3,4,4)` fp16 C=1（`(C,)` 张量长度 1），`square_sum` 范围 `(0,0)`（sq=0）。C++ ST 全过，但 blackbox y 输出报 FAIL：fp32 mere≈0.557（约 20% 绝对误差），fp16 版 NaN/Inf mismatch。同输入的 2D `(8,1)`、`(8,2)` C=1 sq=0 用例 PASS。差异在 rank：2D 走 Branch-0（RANK4），5D 走 Branch-1（RANK8，`RANK>5` 分支）。机制链：`cTotal==1` 时 `isNHWC_ = (chanStride_==1) || (cTotal==1)` 为真，走 NHWC 路径；`RANK>5` 且 `tileNumM_==1`（全量加载）时进入 `CopyInXTileNDDMA()` 快速路径，但 NDDMA 的 dim 0 是 `cBlock_=1`、`loopSrcStride[0]=1`，硬编码假设 channel 在末轴（axis 7），对 `(5,1,3,4,4)` channel 实际在 axis 1（`chanStride_=48≠1`），NDDMA 漏掉中间维度，只搬了部分 x 数据到 UB，后续 y 计算在错误数据上进行，产生 mere≈0.557 / NaN。

## 适用场景

- **工程形态**：kernel 直调 + aclnn TTK 黑盒精度验证。
- **API 路径**：`aclnnBNTrainingUpdate` → `BNUpdateKernel::Process()` 的 S21 CopyIn 选择（`op_kernel/arch35/bn_training_update_kernel.h`，`RANK>5` 分支的 `CopyInXTileNDDMA` / `CopyInXTileRowByRow` 分发）。
- **验证过的平台与 CANN 版本**：Ascend950（Ascend950PR_9579），CANN 9.0.0-beta.2。
- **适用粒度**：`same_operator`（bn_training_update 的 NDDMA 快速路径选择）。
- **路线优先级**：`diagnostic_only`——本卡只给判别方法与根因解释；修复实现的终版代码（在 NDDMA 条件加 `chanStride_==1`、不满足回退逐行 `CopyInXTileRowByRow`）及修复后重跑结果未从 trace 逐字恢复。参照路线=NDDMA 快速路径在 channel 真末轴时更高效，channel 非末轴时回退逐行是正确性必需，来源平台 Ascend950。
- **不适用或未验证场景**：channel 真末轴（`chanStride_==1`）时 NDDMA 快速路径正确，不触发本 bug；`RANK≤5` 走 `CopyInXTileRowByRow` 也不触发。a2/a3 未验证。

## 根因

`isNHWC_ = (chanStride_ == 1) || (cTotal == 1)` 把 `cTotal==1`（C=1 广播）误判为 NHWC（channel 末轴），但 C=1 时 channel 可能在任意轴（如 `(5,1,3,4,4)` channel 在 axis 1，`chanStride_=48`）。`CopyInXTileNDDMA` 的 NDDMA 参数 dim 0 `loopSize=cBlock_=1`、`loopSrcStride=1` 硬编码假设 channel 在 axis RANK-1（末轴），对 channel 在中间轴的 C=1 shape 漏掉部分维度，只搬了部分数据。判别核心是 `isNHWC_` 与"NDDMA 可用"不能等价：`cTotal==1` 不是"channel 末轴"的可靠判据，`chanStride_==1` 才是。

## 判别方法

**触发条件**（tool_observed 失败症状）：`bn_training_update` blackbox TTK 中，C=1（`(C,)` 长度 1）且 `square_sum=0` 的 rank≥5 shape（如 `(5,1,3,4,4)`）y 输出 FAIL——fp32 大绝对误差（mere≈0.557）、fp16 NaN/Inf mismatch；而同输入的 2D C=1 用例（`(8,1)`/`(8,2)`）PASS。C++ ST 全过，排除 host/tiling 侧，指向设备 kernel 的 `RANK>5` 分支。

**可执行探针**：
1. 读 `op_kernel/arch35/bn_training_update_kernel.h` 的 `Process()` S21 CopyIn 分发，确认 `RANK>5` 分支是否以 `tileNumM_==1` 为唯一条件进入 `CopyInXTileNDDMA()`。
2. 读 `isNHWC_` 的定义，确认是否含 `(cTotal==1)` 这一支——若含，则 C=1 广播会被判为 NHWC，而不论 channel 实际轴位。
3. 对失败 shape 计算 `chanStride_`：`chanStride_ == 1` 当且仅当 channel 真在末轴；`(5,1,3,4,4)` channel 在 axis 1，`chanStride_=48≠1`。
4. 对照实验：把同输入降到 2D（如 `(8,1)`），若 PASS 则锁定触发条件为 `RANK>5` + C=1 中间轴 channel（2D 走 Branch-0 不进 NDDMA 快速路径）。

**原始观测**（Ascend950，CANN 9.0.0-beta.2，bn_training_update blackbox TTK）：
- C=1 sq=0 rank5 `(5,1,3,4,4)` y 输出 FAIL：fp32 mere≈0.557（约 20% 绝对误差），fp16 NaN/Inf mismatch。
- 同输入 2D `(8,1)` / `(8,2)` C=1 sq=0 用例 y PASS；C++ ST 全过。
- 失败仅出现在 `RANK>5` 分支（5D 走 Branch-1/RANK8），2D（Branch-0/RANK4）不触发。

**机制解释**（根因分析，属 interpretation）：`isNHWC_` 把 `cTotal==1`（C=1 广播）并入"channel 末轴"判据，又用 `isNHWC_` 作 NDDMA 快速路径开关；C=1 时 channel 位置不确定，NDDMA 的末轴假设（dim0 `loopSrcStride=1`）静默漏数据，表现为"部分 shape 错、2D 对、5D 错"的隐蔽 bug。判别核心是 `chanStride_==1`（channel 真末轴）而非 `cTotal==1`（C=1 广播）。

## 预防

- 凡用 NDDMA 多维 DataCopy 快速路径，其 dim 参数必须与实际 channel 轴位置匹配；不能用 `cTotal==1`（C=1 广播）作为"channel 末轴"的判据——C=1 时 channel 可在任意轴，`chanStride_==1` 才是 channel 真末轴的可靠判据。
- `isNHWC_` 与"NDDMA 可用"不能等价，NDDMA 可用性需单独判 `chanStride_==1`；不满足时回退逐行 `DataCopyPad`/`CopyInXTileRowByRow`。
- blackbox 覆盖必须含 rank≥5 + C=1 + channel 非末轴的组合，否则 2D C=1 用例全过仍漏掉本 bug。

## 坏实践

- 把 `cTotal==1`（C=1 广播）并入 `isNHWC_` 判据，又用 `isNHWC_` 作为 NDDMA 快速路径开关——C=1 时 channel 位置不确定，NDDMA 的末轴假设静默漏数据。
- 看到 mere≈0.557 / NaN 就去查 fp16 精度或 golden 公式，不先核 NDDMA 快速路径的 channel 轴假设；本 bug fp32 与 fp16 都错，不是 fp16 精度问题。

## 验证信号

- blackbox TTK 执行 C=1 sq=0 rank5 `(5,1,3,4,4)` 用例，y 输出 FAIL：fp32 mere≈0.557、fp16 NaN/Inf mismatch（来源平台：Ascend950；证据等级：tool_observed）
- 同输入 2D `(8,1)` / `(8,2)` C=1 sq=0 用例 y PASS；失败仅出现在 `RANK>5` 分支（来源平台：Ascend950；证据等级：tool_observed）

## 失效边界

- Ascend950 / CANN 9.0.0-beta.2 实测；a2/a3 未验证（NDDMA 行为可能随 SoC 代际不同），标注"未验证"。
- 仅适用 NDDMA 快速路径以 `cTotal==1` 作 channel 末轴判据的代码；若算子已用 `chanStride_==1` 单独判 NDDMA 可用性，本判别不触发。
- 本卡只给判别方法与根因解释，不内联修复终版代码。