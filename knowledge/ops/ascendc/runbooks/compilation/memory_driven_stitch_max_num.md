---
type: "Runbook"
title: "memory-driven stitch 与显式 max_num 冲突致挂起"
description: "在已启用 memory-driven stitch 自动合图的 sparse_flash_attention_quant _part 实现上，于 JIT runtime_options 显式设 stitch_function_max_num=128 试图进一步优化，运行后 5 分钟无任何输出、shell 在 300000ms 超时终止，判定为编译/运行挂起；回退该参数后恢复正常。memory-driven stitch 与显式 stitch_function_max_num 是两种互斥的合图决策路径，不可混用，banned。"
tags: ["task.debugging", "topic.implementation.pipeline"]
status: stable
sources: []
created_at: "2026-08-18T09:12:00Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["memory_driven_stitch"]
---

# memory-driven stitch 与显式 max_num 冲突致挂起

## 现象

在 Ascend950（950）上 `sparse_flash_attention_quant_d_950`（bf16）的 `_part` 参考实现已使用 memory-driven stitch 自动合图（隐式决定 stitch 合图数）。为尝试进一步优化，在 JIT `runtime_options` 中显式加入 `stitch_function_max_num: 128`（取与基线 stitch 数同值）。运行后 5 分钟无任何输出（正常完成约 2-3 秒），shell tool 在 300000ms 超时后终止，判定为编译/运行挂起。回退该参数后运行恢复正常。

## 适用场景

- **工程形态**：kernel 直调（pypto-gym JIT，已用 memory-driven stitch 自动合图的算子）。
- **API 路径**：JIT `runtime_options.stitch_function_max_num`（与 memory-driven stitch 模式互斥）。
- **验证过的平台与 CANN 版本**：Ascend950（950）。
- **适用粒度**：`workflow_only`（pypto-gym JIT stitch 配置组合行为，非 kernel 行为本身）。
- **路线优先级**：`banned`——memory-driven stitch 模式下显式设 `stitch_function_max_num` 为禁用组合。
- **不适用/未验证场景**：未启用 memory-driven stitch 的算子不受此约束；显式 stitch 模式下该参数可能合法（本 trace 未验证该路径）；a2/a3 未验证，不得据本卡推断可迁移。

## 失败路线

**banned 路线：在已启用 memory-driven stitch 自动合图的算子上，于 JIT `runtime_options` 显式设 `stitch_function_max_num`（如 128）。**

- **尝试**：`_part` 实现已由 memory-driven stitch 隐式决定 stitch 合图数；为追求进一步优化，照搬另一实现上正常的显式 `stitch_function_max_num=128` 写入 `runtime_options`。该 edit 的 tool_call 输入被 trace adapter 截断为 `_see below_`，verbatim before/after 未从 trace 恢复。
- **客观失败**：运行该配置后 5 分钟无任何输出（正常 2-3 秒完成），shell tool 在 300000ms 超时后终止、无错误退出码，属编译/运行挂起（来源平台：950；证据等级：tool_observed）。
- **边界**：仅验证了 memory-driven stitch + 显式 128 的冲突组合；未验证关闭 memory-driven stitch 后显式设值是否合法，未验证其他显式取值（如 256）在 memory-driven 模式下是否同样挂起。
- **收敛**：memory-driven stitch 与显式 `stitch_function_max_num` 是两条互斥的合图决策路径——前者由编译器按 memory 自动决定 stitch 合图数，后者显式给定上限；二者同时存在时编译器在两条决策路径间无法收敛，导致编译阶段挂起（无错误输出、纯超时）。该机制属 agent 推断（编译器是否报 internal error 或死循环未追到机制层），故本卡以失败路线收敛、不发布为修复。回退该参数后运行恢复正常（该恢复信号为 agent reasoning，未作为 tool_result 捕获，不进 `## 验证信号`）。

## 验证信号

本段只收本 trace 内 tool_observed 实测，agent 推断的机制假设与口述恢复不进此段。

- 在 memory-driven stitch 已启用的 `_part` 实现上显式设 `stitch_function_max_num=128` 运行 JIT，5 分钟无任何输出，shell tool 在 300000ms 超时后终止、无错误退出码（来源平台：950；证据等级：tool_observed）

## 预防

memory-driven stitch 模式下不要显式设 `stitch_function_max_num`；显式 stitch 与 memory-driven stitch 是两种互斥的合图决策路径，不可混用。把一个实现上正常的 JIT 参数值照搬到用不同 stitch 模式的另一实现，上下文不同、参数语义不同，属坏实践。
