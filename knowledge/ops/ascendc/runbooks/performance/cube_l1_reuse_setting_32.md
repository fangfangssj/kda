---
type: "Runbook"
title: "cube_l1_reuse_setting 32 非单调反退"
description: "Ascend950 上 cube_l1_reuse_setting={-1:32} 相对甜点 {-1:16} 略差，L1 占用上升挤压其他 cube 数据；banned，不得盲目增大 L1 复用次数。"
tags: ["task.optimization", "topic.memory", "topic.programming_model.execution_unit.cube", "topic.performance.analysis"]
status: stable
sources: []
created_at: "2026-08-18T08:43:30Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# cube_l1_reuse_setting 32 非单调反退

## 现象

`sparse_flash_attention_quant_d_950`（bf16，Ascend950）在 `s_kv_tile=512` + `ooo_sched_mode=GAPMIN` + `device_sched_mode=0` + `vec_nbuffer_setting={-1:8, -2:1}` 配置下，对 JIT `pass_options.cube_l1_reuse_setting` 做 sweep。当 `cube_l1_reuse_setting` 从甜点 `{-1:16}` 增大到 `{-1:32}` 后，profiler 输出显示 `EXIT=0`（精度通过）但 AICore 利用率持平略降，性能未提升反而略差。增大 L1 复用次数未能带来收益，呈非单调行为。

## 适用场景

- 工程形态：kernel 直调（pypto-gym，Cube matmul 为主的 attention）。
- API 路径：JIT `pass_options.cube_l1_reuse_setting`（`{-1: <int>}`）。
- 验证过的平台与 CANN 版本：Ascend950（950）；CANN 版本未记录。
- 适用粒度：`operator_family`（950 上 scope=20001 部分包裹结构的 cube matmul 算子）。
- 路线优先级：`banned`——`cube_l1_reuse_setting={-1:32}` 在本结构下未带来性能收益且略差，不得作为优化手段发布。
- 不适用/未验证场景：scope=5001 全包裹结构下 L1 复用受 mix合图统一管理，甜点值不同；a2/a3 未验证。

## 根因

`cube_l1_reuse_setting` 控制 AIC cube 子图的 L1 buffer 复用次数。在 scope=20001（仅 V0+C1）部分包裹结构下，L1 复用 16 次能覆盖多次 K 迭代、减少 L1 重载；但继续增大到 32 次时，L1 占用上升挤压其他 cube 数据的存放空间，调度开销增加，性能持平略降。甜点值随 mix合图 scope 策略变化，不可跨结构照搬。

## 失败路线

**banned：不得把 `cube_l1_reuse_setting={-1:32}` 作为已验证性能优化发布。**

尝试路径：在已确定 `{-1:16}` 为当前结构甜点后，将 `cube_l1_reuse_setting` 从 `{-1:16}` 增大到 `{-1:32}`，期望通过增加 L1 复用次数进一步减少 L1 重载。profiler 输出 `EXIT=0`、AICore 利用率 42.81%（持平略降），性能未改善，agent 随即回退到 `{-1:16}`。

判别方式：对同一算子同一 shape，只改 `cube_l1_reuse_setting` 单参数，对比 AICore 利用率与 EXIT 码。若增大复用次数后利用率不升反降，则已越过甜点。

```python
# banned 配置——不要照搬
pass_options = {
    "ooo_sched_mode": "GAPMIN",
    "vec_nbuffer_setting": {-1: 8, -2: 1},
    "cube_l1_reuse_setting": {-1: 32},  # ← banned：略差于 {-1:16}
}
```

## 验证信号

以下信号来自本次开发会话对 `cube_l1_reuse_setting={-1:32}` 的 profiler 实测。

- `cube_l1_reuse_setting={-1:32}`：`EXIT=0`（精度通过），AICore 利用率 42.81%（来源平台：950；证据等级: tool_observed）

## 失效边界

- mix合图结构变化时甜点值移动：scope=5001 全包裹下 `{-1:1}` 最优（agent 报告，未由本 trace 的 tool_result 直接验证）；不可跨结构照搬 16 或 32。
- a2/a3 未验证；L1 复用机制跨 A 系列一致但 L1 容量不同，甜点值可能移动。
- 本卡只覆盖 `{-1:32}` 单次实测的负面结果；`{-1:16}` 甜点主张与 `{-1:8}` 基线值未由本 trace 的非 redact tool_result 直接验证。