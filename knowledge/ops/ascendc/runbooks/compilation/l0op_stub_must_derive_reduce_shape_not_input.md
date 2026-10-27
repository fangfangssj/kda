---
type: "Runbook"
title: "L0 op 须算归约后 shape（非复用输入）"
description: "L0 op stub（op_api 层）若用 x->GetViewShape() 分配输出而非按 axes/keepdim 算归约后 shape，ViewCopy shape 不匹配致 561103/161002；L0 op 与 InferShape 分离须各自推导。"
tags: ["task.debugging", "topic.implementation.operator_registration"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["reduce_shape"]
---

# L0 op 须算归约后 shape（非复用输入）

## 现象

`euclidean_norm`（归约范数，ascend950）冒烟测试 `tests/aclnn/run.sh` 在 `GetWorkspaceSize` 阶段报 561103。逐事件追溯：详细日志显示 L0 op `l0op::EuclideanNorm`（`op_api/euclidean_norm.cpp`）仍为 stub，用 `x->GetViewShape()`（[2,3]）分配输出，而 `y` 已是归约后 shape [1,3] → `ViewCopy(result=[2,3], y=[1,3])` shape 不匹配 → 161002 → 561103。`TilingFunc` 在 `out=[2,3]` 时已运行成功，仅 `ViewCopy([2,3]→[1,3])` 失败。L0 op（op_api 层）与 `InferShape`（op_host 层）是分离的两处，InferShape 已正确推导 y=[1,3]，但 L0 op stub 未镜像同一推导，仍用输入 shape 分配中间 result。

## 适用场景

- **工程形态**：aclnn 算子仓开发，op_api 层 L0 op stub + op_host 层 InferShape 分离实现。
- **API 路径**：`aclnnEuclideanNormGetWorkspaceSize` → L0 op（`l0op::EuclideanNorm`，op_api）→ `ViewCopy` → `aclnnEuclideanNorm` Execute。
- **验证过的平台与 CANN 版本**：Ascend950，CANN 9.1.0。
- **适用粒度**：`operator_family`——任何归约/规约类算子（output shape 由 axes/keepdim 推导，不等于输入 shape）。
- **路线优先级**：`recommended_fastpath`（L0 op stub 镜像 InferShape §3.1 推导归约后 shape）。
- **不适用或未验证场景**：输出 shape 等于输入 shape 的算子（无归约）；a2/a3 未验证（推断可迁移，L0 op/InferShape 分离跨平台一致）。

## 判别方法

**触发**：ST `GetWorkspaceSize` 报 561103，且算子为归约类（y shape ≠ x shape），且 L0 op 为 stub。

**探针**（可执行）：
1. `grep -n 'l0op::EuclideanNorm\|GetViewShape\|ViewCopy\|InferShape' op_api/euclidean_norm.cpp`——L0 op 是否用输入 shape 分配输出。
2. 启用详细日志：`ASCEND_GLOBAL_LOG_LEVEL=0 ./run.sh 2>&1 | grep -E 'ViewCopy|161002|561103|EuclideanNorm'`，定位 161002 的 ViewCopy result/y shape。
3. 对比日志中 `TilingFunc out=[...]` 与 `ViewCopy(result=[...], y=[...])`——若 TilingFunc 用 [2,3] 而 y=[1,3]，即 L0 op 未算归约 shape。

**原始观测**（本 trace，tool_observed）：
- L0 op stub 用 `x->GetViewShape()` [2,3] 分配输出。
- `TilingFunc` 在 `out=[2,3]` 运行成功；仅 `ViewCopy([2,3]→[1,3])` 失败 161002。
- 修复 L0 op 后 `GetWorkspaceSize` 成功，错误移至 Execute（361001，为独立根因，见对应卡）。

## 根因

aclnn 算子的 **L0 op（op_api 层）**与 **InferShape（op_host 层）**是两处独立代码：InferShape 推导输出 y 的最终 shape，L0 op 负责 GetWorkspaceSize 阶段的中间 result 分配。若 L0 op stub 图省事用 `x->GetViewShape()`（输入 shape）分配 result，而 y 已被 InferShape 推导为归约后 shape，则 `ViewCopy(result=输入shape, y=归约shape)` shape 不匹配 → 161002 → 561103。两处必须**各自独立推导**同一归约 shape 语义（axes 归一化、keepdim、全归约）。

## 解决配方

修复（agent_reported，逐字代码未恢复）：L0 op（`op_api/euclidean_norm.cpp`）按 axes/keepdim 计算真实归约后输出 shape（镜像 InferShape §3.1 推导），**不改 InferShape 自身**。修复后 `GetWorkspaceSize` 成功（ViewCopy shape 匹配）。

> op_api/euclidean_norm.cpp 修复后完整代码未在事件流保留可解析 tool_result（file_read 的 tool_result 未单独落盘或被转写摘要替代）；逐字代码不可恢复。机制（L0 op 与 InferShape 分离，各自推导归约 shape）由日志观测与 assistant_message 推理闭环。

## 系统信息

- 目标平台：Ascend950；CANN 9.1.0。
- 工具链：`./build.sh euclidean_norm_package`；ST：`tests/aclnn/run.sh`；详细日志 `ASCEND_GLOBAL_LOG_LEVEL=0`。
- 相关文件：`op_api/euclidean_norm.cpp`（L0 op）、`op_host/euclidean_norm_infershape.cpp`（InferShape）。

## 验证信号

- 修复 L0 op 后 `GetWorkspaceSize` 成功（ViewCopy shape 匹配，561103 消失）（来源平台：Ascend950；证据等级: tool_observed）
- 后续错误移至 Execute 阶段 361001（独立根因，见对应卡）（来源平台：Ascend950；证据等级: tool_observed）

## 失效边界

- a2/a3：未验证，推断可迁移（L0 op/InferShape 分离与 ViewCopy 语义跨平台一致）。
- 输出 shape 等于输入 shape 的算子（无归约）不适用。

## 预防

- L0 op stub 不要用 `x->GetViewShape()` 分配输出——必须按 axes/keepdim 推导归约后 shape。
- 把 InferShape §3.1 的推导逻辑抽成可被 op_api 与 op_host 共用的纯函数，避免两处各写一份漂移。
- ST 调试 561103 时优先 `ASCEND_GLOBAL_LOG_LEVEL=0` 看 ViewCopy 的 result/y shape，区分 L0 op shape bug 与 tiling 路由 bug。

## 坏实践

- 把 L0 op 的 result 分配改为输入 shape 后，再去改 y 的 shape 匹配 result——倒置因果，y 的 shape 由 InferShape 推导，不应迁就 L0 op。
- 把 ViewCopy 失败误判为 kernel 实现 bug 去改 kernel——根因在 op_api 层 L0 op shape 推导。

## 误诊

本 trace 早期用户报告"形状不匹配"被误判为完整诊断，实际 561103（L0 op shape）之后还藏着 361001（kernel 二进制未注册，为独立根因），是两个串联根因。
