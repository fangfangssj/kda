---
type: "Runbook"
title: "C1 Cube FP32 输出阻断 950 Mix合图 CV 通路"
description: "sparse_flash_attention_quant 的 C1 matmul 输出 FP32 阻断 950 Mix合图 L0C→UB CV 通路：扩展 mix合图到 V1+C2 反退化，C1 输出改 BF16 触发 ERR_PARAM_DTYPE_UNSUPPORTED，两条路线均 banned。"
tags: ["task.debugging", "topic.programming_model.execution_unit.cube", "topic.implementation"]
status: stable
sources: []
created_at: "2026-08-18T08:40:58Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["mix_graph", "CV_path"]
---

# C1 Cube FP32 输出阻断 950 Mix合图 CV 通路

## 现象

在 `sparse_flash_attention_quant_d_950`（bf16，Ascend950/950）Cube↔Vector 交替段 attention 上，为消除 C1→V1 的 DDR 往返，尝试两条改造路线均失败回退：(1) 把 `sg_set_scope=20001`（仅包裹 V0+C1）扩展到包裹 V0+C1+V1+C2（C2 后才 `sg_set_scope=-1` 复位）——先因 workspace 需求约 465183KB 超 `max_workspace_kb=458752` 编译失败，增大 workspace 后运行，AICore E2E 相对扩展前约 1.37× 退化；(2) 把 C1 matmul 输出 dtype 从 `DT_FP32` 改为 BF16 以打通 L0C→UB CV 通路——立即报 `VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED`（下游 vector mul/amax 不支持 BF16 输入），运行失败。两者均回退到仅 V0+C1 mix合图、C1 输出 FP32 的基线。

## 适用场景

- **工程形态**：kernel 直调（pypto-gym，Cube↔Vector 交替段 attention）。
- **API 路径**：`pypto.set_pass_options(sg_set_scope=...)` / `pypto.matmul(..., out_dtype)` / `pypto.mul` / `pypto.amax`。
- **验证过的平台**：Ascend950（950）。
- **适用粒度**：`operator_family`——任何 Cube 输出 FP32 且下游 Vector 需 FP32 输入的 CV 交替段。
- **路线优先级**：`banned`——对 C1 输出 FP32 的算子，mix合图扩展到 V1+C2 与 C1 输出降 BF16 均为禁用方向。
- **不适用/未验证场景**：Cube 输出本就是 BF16/FP16 的算子不受此约束，可正常扩展 mix合图覆盖 CV 通路；a2/a3 平台未验证。

## 失败路线

- **路线 A（mix合图 scope 扩展到 V1+C2）**：目标让 C1→V1 走 L0C→UB CV 通路消除 DDR 往返。先因 workspace 需求约 465183KB 超 `max_workspace_kb=458752` 编译失败；增大 max_workspace_kb 后运行，AICore E2E 相对扩展前约 1.37× 退化。根因：C1 输出仍为 FP32，L0C→UB 不支持 FP32，C1 数据仍回退 GM，CV 通路在 C1→V1 处断裂，扩展只增加同步开销与 workspace 需求而无通路收益。结论：banned。
- **路线 B（C1 matmul 输出 FP32→BF16）**：目标让 L0C→UB 可通。运行立即报 `VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED`（C1 下游 `pypto.mul(sij, softmax_scale)`、`pypto.amax(...)` 等 vector 操作不支持 BF16 输入）。根因：打通 L0C→UB 的代价是下游 vector 操作 dtype 不兼容，CV 通路被两侧 dtype 约束夹住。结论：banned。
- **路线 C（C1 后插 cast FP32→BF16 再进 V1）**：未运行即判明无效——matmul 仍输出 FP32 到 L0C，cast 在 UB 做无益于 L0C→UB 通路，C1 数据仍走 DDR。结论：banned。

## 根因

950 Mix合图 CV 通路（L0C→UB）只支持 FP16/BF16，不支持 FP32。本算子 C1 matmul `sij = pypto.matmul(qi, kj_view, pypto.DT_FP32, ...)` 输出 FP32（在线 softmax 精度所需），因此 C1→V1 这条 CV 链路无法走 L0C→UB，必经 DDR 往返。机制链：C1 out FP32 → L0C→UB 不支持 FP32 → C1 输出走 DDR → mix合图在 C1→V1 断裂 → scope 扩展无 CV 通路收益 + 增加同步/workspace → 退化；改 out BF16 → L0C→UB 可通但 vector mul/amax 不支持 BF16 → 报错。要消除 C1→V1 的 DDR 往返需从根本上改变 softmax 数据类型策略（如全程 BF16 softmax + 专用 reduce），本轨迹未实现。

## 验证信号

- mix合图扩展到 V0+C1+V1+C2 后 AICore E2E 相对仅 V0+C1 配置约 1.37× 退化，且 workspace 需求约 465183KB 超 `max_workspace_kb=458752`（来源平台：950；证据等级：tool_observed）
- C1 matmul 输出改 BF16 后报 `VectorErrorCode::ERR_PARAM_DTYPE_UNSUPPORTED`，运行失败（来源平台：950；证据等级：tool_observed）

## 失效边界

- Cube 输出本就是 BF16/FP16 的算子不受此约束，可正常扩展 mix合图覆盖 CV 通路。
- a2/a3 平台未验证（L0C→UB 的 FP32 限制是 950 架构特性，跨 A 系列一致的概率高，但未实测）。