---
type: "Runbook"
title: "PyPTO loop_unroll 串行 s2 循环诊断"
description: "PyPTO decode 算子 no_flash compute 路径 s2_kv tile 循环以 unroll_list={1} 串行执行 4 次独立迭代，AICore E2E 超 500us 目标而利用率已 93%+、0 瓶颈；用 perf-analyze 解析 bubble_analysis.log + 内核源码检视 unroll_list 与迭代依赖性二分定位为串行迭代链未并行化，而非调度/搬运瓶颈。"
tags: ["task.debugging", "topic.implementation.pipeline", "topic.performance.analysis", "topic.programming_model.execution_unit.cube"]
status: stable
sources: []
created_at: "2026-08-18T08:38:58Z"
updated_at: '2026-08-27T12:23:11Z'
platforms: ["950"]
---

# PyPTO loop_unroll 串行 s2 循环诊断

## 现象

在 Ascend950（950 die，96 核）/ CANN 9.1.0 环境下，用 `pypto-op-perf-tune` skill 对 DeepSeek V32 `sparse_flash_attention_quant` 算子的 decode 950 路径（`sparse_flash_attention_quant_d_950`，实际执行 no_flash compute 路径）做开箱性能调优，目标 AICore E2E ≤500us。基线（`unroll_list={1}`，`device_sched_mode=1`）用 `cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py` 解析 `bubble_analysis.log`：平均核心利用率 93.25%、平均气泡率 4.05%、负载均衡度 98.61%、AICore E2E 897.45us、0 个性能瓶颈——属 compute-bound 且调度搬运已较优，但 E2E 仍超目标。该 897.45us/500us 超限为本卡诊断的触发信号（tool_observed，非 agent 推测）。

## 适用场景

- **工程形态**：PyPTO `@pypto.frontend.jit` 装饰的 decode 算子直调，性能调优（非 aclnn 算子仓开发）。
- **API 路径**：`pypto.loop_unroll(start, end, step, name=, idx_name=, unroll_list=)`；`cannbot-skills/ops/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py`（`bubble_analysis.log` 解析）。
- **验证过的平台与 CANN 版本**：Ascend950（950 die），96 核，CANN 9.1.0，pypto build_out（arch32）。仅此平台实测。
- **适用粒度**：`operator_family`——任何 PyPTO 算子中"迭代间无数据依赖"的 `pypto.loop_unroll` 循环；最窄落到 `sparse_flash_attention_quant` 的 no_flash compute s2 循环。
- **路线优先级**：`recommended_fastpath`——高利用率 + 0 瓶颈 + compute-bound 基线上，迭代级并行是唯一有效杠杆（tile-shape 调整在本 shape 无正收益，见失效边界）。
- **明确不适用/未验证场景**：迭代间存在数据依赖的循环（如 flash 在线 softmax 归约路径 `FLASH_LOOP_L4_s2_SA`，其 `oi_update/li_update/mi_update` 跨迭代累积）不可套用本诊断的"迭代独立可并行"前提；a2/a3 平台未实测。

## 判别方法

### 触发信号

PyPTO decode 算子 perf-analyze 报告 AICore E2E 超目标，且同时满足：平均核心利用率高（本例 93.25%）、平均气泡率低（4.05%）、负载均衡度高（98.61%）、0 个性能瓶颈——即 compute-bound 且调度搬运已较优，常规瓶颈分析（瓶颈数=0）无法给出优化方向。本触发信号为上板实测（tool_observed），非 agent 推测。

### 探针：perf-analyze + 内核源码二分

1. **perf-analyze 解析**：对 `bubble_analysis.log` 运行 `analyze_perf.py`，读取核心利用率/气泡率/负载均衡度/瓶颈数/AICore E2E。若利用率高 + 0 瓶颈 + E2E 仍超目标 → 转向"循环级并行未利用"假设。
2. **内核源码检视**：grep / 读 kernel 实现，定位所有 `pypto.loop_unroll(...)` 声明，检查可疑循环的 `unroll_list` 是否为 `{1}`（串行）以及循环迭代数 `bn_per_batch`。
3. **迭代依赖性判定**：逐迭代分析循环体是否跨迭代累积同一变量。no_flash compute 的 s2 迭代写同一 `cur_offset`、末次写生效，迭代间无数据依赖 → 可并行；flash 在线 softmax 的 `oi/li/mi` 跨迭代累积 → 不可并行。

触发信号对应 perf tool_result（seq 445，verbatim 摘要）：

```
=== 性能指标摘要 ===
平均核心利用率: 93.25%
平均气泡率: 4.05%
核心负载均衡度: 98.61%
算子实际执行时间: 897.45 us
发现 0 个性能瓶颈
```

诊断目标——`LOOP_L4_s2_SA` 的串行声明（verbatim 取自 trace 内 git diff seq 572，before 行）：

```python
                    for s2_idx, _ in pypto.loop_unroll(0, bn_per_batch, 1,
                        name="LOOP_L4_s2_SA", idx_name="s2_idx", unroll_list={1}):
```

### 原始观测

- 基线 perf-analyze（seq 445）：利用率 93.25%、气泡率 4.05%、负载均衡度 98.61%、AICore E2E 897.45us、0 瓶颈。
- 内核源码（seq 272 file_read / seq 572 diff）：`LOOP_L4_s2_SA` 声明 `unroll_list={1}`，`bn_per_batch=ceil(2048/512)=4`，即 4 次 s2 迭代以单并行块串行展开。
- 反向验证（seq 523）：把 `unroll_list` 改为 `{4,2,1}` 后 perf-analyze 报告利用率 97.17%、气泡率 1.20%、AICore E2E 323.33us、0 瓶颈——E2E 大幅下降且利用率上升，反向印证"串行迭代链为 E2E 主导"。

### 解释

`unroll_list={1}` 表示只按 1 个并行块串行展开 4 次迭代；no_flash compute 路径每次 s2 迭代独立完成"gather K/V → Q@K^T → per-tile softmax → P@V → assemble 到同一 cur_offset"且末次写生效，迭代间无数据依赖，本可并行却被串行调度，AIC 等待串行链致 AICore E2E 偏高。`unroll_list` 数值集合代表可选并行块大小（按从大到小尝试），`{4,2,1}` 允许编译器把 4 次迭代作为 4 个并行任务调度到多核（无法整除时退化到 2 或 1）。高利用率 + 0 瓶颈 + compute-bound 的基线上，瓶颈不在调度搬运而在串行迭代链本身——这是 perf-analyze"0 瓶颈"报告的盲区，需结合内核源码的 `unroll_list` 检视才能定位。

**解释的不确定性**：最终 diff 同时改了 `unroll_list` 与 `device_sched_mode`（3→1），3→1 的约 2% 独立贡献为 agent 自述、无单独 perf tool_result，故"串行迭代链为 E2E 主导"的归因由 `{4,2,1}` 单变量带来的 E2E 量级下降（897→323us）支撑，`device_sched_mode` 的边际贡献未隔离。

## 验证信号

本段只收录本 trace 内 `tool_observed` / `artifact_verified` 信号；tile-shape 失败路线编辑原文被 redact（`_see below_`）、`device_sched_mode` 独立改善为 agent 自述，均不列入。

- 基线 perf-analyze（seq 445）：平均核心利用率 93.25%、平均气泡率 4.05%、负载均衡度 98.61%、AICore E2E 897.45us、0 个性能瓶颈（来源平台：950；证据等级：tool_observed）
- 内核源码（seq 272 file_read / seq 572 diff）`LOOP_L4_s2_SA` 声明 `unroll_list={1}`，`bn_per_batch=4`，4 次 s2 迭代串行展开（来源平台：950；证据等级：artifact_verified）
- 并行化后 perf-analyze（seq 523）：平均核心利用率 97.17%、平均气泡率 1.20%、AICore E2E 323.33us、0 瓶颈——反向验证"串行迭代链为 E2E 主导"归因（来源平台：950；证据等级：tool_observed）

## 失效边界

- **flash 在线归约路径不可套用**：`FLASH_LOOP_L4_s2_SA` 迭代间有 `oi_update/li_update/mi_update` 累积依赖，本诊断"迭代独立可并行"前提不成立，直接套 `unroll_list={4,2,1}` 会破坏归约语义；需改用归约友好的分块并行 + 末段归约策略，本 trace 未验证。
- **迭代数非 4 时**：`unroll_list` 集合首元素应取实际 `bn_per_batch`（如 2 次迭代用 `{2,1}`）；动态 `bn_per_batch` 场景需验证编译器能否按集合退化。
- **tile-shape 调整不是有效杠杆**：先后尝试 `c1/c2 kL1=576`、`C2 kL0=128 nL0=512`、`s_kv_tile=2048` 均无正收益或更差/超时——在 93% 利用率的 compute-bound 基线上调 tile-shape 不是有效杠杆；但相关编辑原文被 redact，结论为 agent_reported，不作为本卡诊断证据。
- **a2/a3 平台未实测**：PyPTO `loop_unroll` 语义应跨平台一致，但 950 的 `device_sched_mode`/调度行为与 a2/a3 可能不同，不得据本卡直接采用——需目标平台定向 perf 验证后再用。