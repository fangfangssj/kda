---
type: "Runbook"
title: "507035 — aten 与自定义命名空间路径对照"
description: "PyTorch NPU 扩展中，同类直调 kernel 在自定义命名空间下稳定触发下游 507035，而改为显式 aten 调用后错误消失并命中替换实现；用于做 dispatch 路径 A/B，不把 format 后处理机制当作已证实根因。"
tags: ["task.debugging", "topic.implementation.operator_registration", "topic.implementation"]
status: draft
sources: []
created_at: "2026-08-18T08:27:55Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: []
---

# 507035 — aten 与自定义命名空间路径对照

## 现象

某 AscendC kernel 通过 `<<<>>>` 写入 `at::empty`/`at::empty_like` 分配的输出并返回 PyTorch。注册为 `npu_test::npu_rms_norm` 时，下游 torch_npu op 读取输出会稳定出现一次 507035；同工程内注册在 `aten::*` 的同类直调算子未复现。

轮换 `at::empty`、`at::empty_like`、`new_empty`、`clone` 等分配方式没有改变错误。把实现注册到 `aten::rms_norm` 并让模型显式调用 `torch.ops.aten.rms_norm` 后，507035 计数降为 0，且替换 kernel 有明确命中计数。

## 适用场景

- **工程形态**：PyTorch/torch_npu 扩展，通过 `TORCH_LIBRARY` 注册并由 host 直调 AscendC kernel。
- **适用粒度**：`exact_task`。平台、CANN 版本和 format descriptor 未在 trace 中定位。
- **路线优先级**：`diagnostic_only`。命名空间切换只作为 dispatch 路径对照，不是通用修复规则。

## 判别方法

1. 固定 kernel、输出分配方式和模型输入，只改变注册/调用路径。
2. 分别统计 `grep -c '507035' test.log` 和 kernel replacement 命中数；错误消失但命中数为 0 不构成有效对照。
3. 已有 aten schema 时，确认模型实际调用 `torch.ops.aten.<op>`。本例中模型继续调用 `torch_npu.npu_rms_norm` 时，自定义 `aten::rms_norm` 命中数仍为 0。
4. 分配 API 轮换均无效时，可降低“单纯输出分配 API 错误”的优先级，但不能据此证明具体的 NPU format 内部机制。

## 验证信号

- 自定义命名空间路径下每次推理 `507035` 计数为 1；多种输出分配 API 下均不变（来源平台：未定位；证据等级：tool_observed）。
- 同工程 `aten::add` 同类直调路径的 507035 计数为 0（来源平台：未定位；证据等级：tool_observed）。
- 注册到 `aten::rms_norm` 且显式调用 `torch.ops.aten.rms_norm` 后：`507035: 0`、`KERNEL-REPLACE RmsNorm: 6998`、`ATEN-REPLACE: 4166`（来源平台：未定位；证据等级：tool_observed）。
- 注册 aten 实现但模型仍调用 `torch_npu.npu_rms_norm` 时，替换 kernel 命中数为 0（来源平台：未定位；证据等级：tool_observed）。

## 解释边界

这些 A/B 结果证明注册与实际调用路径会影响故障是否出现，也证明 `torch_npu.<op>` 不会自动命中本例的自定义 aten 实现。trace 没有直接观测 tensor format descriptor 或所谓 dispatch 后处理，因此不能断言 507035 一定由 format 标记修复/缺失导致，也不能推广成“所有自定义命名空间都不安全”。

## 失效边界

- 未记录芯片型号与 CANN 版本；不得自动推广到 950/a3/a2。
- 目标 aten schema 不存在或调用约定不兼容时，不能使用此 A/B。
- 显式改调用路径会改变集成契约，应用前需单独验证功能与兼容性。

## 坏实践

- 只看 507035 消失，不确认自定义 kernel 是否真正命中。
- 把相关性直接写成 tensor format 内部机制结论。
