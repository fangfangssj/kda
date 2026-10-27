---
type: "Runbook"
title: "flash在线softmax JIT重排致精度超阈"
description: "flash 在线 softmax 的 mi/li/oi 循环依赖对执行顺序敏感；loop_unroll_list、device_sched_mode、cube_nbuffer_setting 任一调离默认值都会改变 C1 任务执行顺序，引入 BF16 ULP 级累积误差 0.00048828 超逐位置阈值，三参数对在线 softmax 算子 banned。"
tags: ["task.debugging", "topic.precision", "topic.compute.normalization"]
status: stable
sources: []
created_at: "2026-08-18T08:43:16Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
aliases: ["online_softmax", "loop_carried_dependency"]
---

# flash在线softmax JIT重排致精度超阈

## 现象

在 `sparse_flash_attention_quant_d_950`（bf16 非量化 flash 在线 softmax，Ascend950）JIT 调优过程中，把任一改变 C1/flash-update 任务执行顺序的 JIT 参数调离默认值——`loop_unroll_list` 从 `{1}` 改为 `{4}`、`runtime_options.device_sched_mode` 从 `0` 改为 `2/3`、`pass_options.cube_nbuffer_setting={-1:2}`——都会触发同一类精度失败：BF16 ULP 级（约 2^-11）误差 `0.00048828` 刚好超出逐位置阈值（约 `0.00042` 量级），出现在固定位置，无 NaN/Inf、无结构性错位。回退默认值后精度恢复通过。三者根因同一：在线 softmax 的 `mi/li/oi` 跨迭代循环依赖对执行顺序敏感，任何重排 C1 任务顺序的优化都会改变累积次序，引入 BF16 ULP 级累积误差并超阈值。

## 适用场景

- **工程形态**：kernel 直调（PyPTO gym，flash 在线 softmax 算子）。
- **API 路径**：`pypto.loop(..., unroll_list=...)`、JIT `runtime_options.device_sched_mode`、JIT `pass_options.cube_nbuffer_setting`。
- **验证过的平台与 CANN 版本**：Ascend950（950）。
- **适用粒度**：`operator_family`（任何采用 flash 在线 softmax、`mi/li/oi` 跨迭代传递的 attention 变体）。
- **路线优先级**：`banned`——对在线 softmax 算子，`loop_unroll_list={4}`（裸展开未守序）、`device_sched_mode=2/3`、`cube_nbuffer_setting={-1:2}` 属禁用路线；保持默认 `unroll_list={1}` / `device_sched_mode=0` / 不设 `cube_nbuffer_setting`。
- **不适用/未验证场景**：非在线 softmax（标准注意力、单块 softmax）不受此约束；用 `sg_set_scope=5001` 全包裹 CV 序列后再 unroll 的 `_part` 参考实现精度通过，说明禁用的是"未守序下的裸重排"而非 unroll 本身。

## 判别方法

### 触发信号

对 flash 在线 softmax 算子改了任一影响 C1/flash-update 任务执行顺序的 JIT 参数（unroll/sched_mode/cube_nbuffer）后，精度校验输出固定位置 `误差=0.00048828 | 阈值≈0.00042`（约 1 个 BF16 ULP @ 0.06 附近），且无 NaN/Inf、无结构性错位——纯逐位置 ULP 抖动超阈。本信号为上板实测（tool_observed），非 agent 推测。

### 探针

在同一算子、同一 tile 配置下分别跑两组精度校验：(A) 默认顺序（`unroll_list={1}`、`device_sched_mode=0`、不设 `cube_nbuffer_setting`）；(B) 任一参数调离默认值。对比两组的逐位置误差输出。若 (A) 全位置通过、(B) 出现同一批固定位置的 `误差=0.00048828` 超阈，即可判定为"执行顺序重排触发在线 softmax 累积误差"。进一步可检查该参数是否在编译期改变 C1 任务的合并/调度顺序：unroll 合并迭代、sched_mode 改设备侧调度、cube_nbuffer 合并相似 AIC 子图。

### 观测

实测中，默认顺序配置精度校验 `EXIT=0` 且 PRECISION 段无逐位置超阈项（基线通过）；将 `cube_nbuffer_setting={-1:2}`（同模式亦见于 `device_sched_mode=3`）后，精度校验输出固定位置超阈：

- 位置 `(0,1,5,190)`：`-0.06347656 vs -0.06396484 | 误差=0.00048828 | 阈值=0.00041962`
- 位置 `(2,0,38,55)`：`-0.07275391 vs -0.07226562 | 误差=0.00048828 | 阈值=0.00046158`
- 位置 `(2,1,86,55)`：`-0.06591797 vs -0.06542969 | 误差=0.00048828 | 阈值=0.00042725`

误差量级 `0.00048828 ≈ 2^-11`，恰为 BF16 在 `0.06` 量值附近的 1 ULP，且三处阈值均被同等超出——典型的"累积顺序变化导致的 ULP 级系统性抖动"，而非个别数值错误。

### 判读

根因是 flash 在线 softmax 的循环依赖：每次 s2 子块迭代更新 `mi_new = max(mi, tilda_mij)`、`li_new = ...`、`oi_tmp = ...`，`oi/li/mi` 跨迭代顺序传递。`oi` 累积形如 `oi = oi * exp(mi - mi_new) + oi_j * exp(mij - mi_new)`，`li` 同理。不同执行顺序下 `max/reduce` 的结合次序不同，BF16 下每步约 epsilon 量级抖动，跨多次迭代累积达 `2^-11` 量级，恰超逐位置阈值。`loop_unroll_list={4}` 把 4 次迭代合并展开、破坏 `s2_idx` 递增 1 的顺序假设，使 flash update 的 `is_loop_begin/is_loop_end` 守序逻辑失效；`device_sched_mode=2/3` 改变设备侧任务调度顺序；`cube_nbuffer_setting={-1:2}` 合并相似 AIC 子图，可能把不同 s2 迭代的 C1 任务重排执行。三者根因同一：任何改变 C1/flash-update 任务执行顺序的优化，都会改变在线 softmax 累积顺序。诊断结论：出现该触发信号时，先怀疑 JIT 重排类参数，而非算子逻辑或 tile 形状。

## 验证信号

- 默认顺序配置（`unroll_list={1}`、`device_sched_mode=0`、不设 `cube_nbuffer_setting`）精度校验 `EXIT=0`、PRECISION 段无逐位置超阈项，基线通过（来源平台：950；证据等级：tool_observed）
- `cube_nbuffer_setting={-1:2}`（与 `device_sched_mode=3` 同模式）精度校验输出固定位置 `误差=0.00048828` 超阈（阈值 `0.00041962`/`0.00046158`/`0.00042725`），精度失败（来源平台：950；证据等级：tool_observed）