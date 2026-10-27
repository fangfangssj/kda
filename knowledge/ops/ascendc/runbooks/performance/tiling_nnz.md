---
type: "Runbook"
title: "按行均分 tiling 致 nnz 倾斜时约半数核空闲"
description: "稀疏算子 Host 按行数均分多核（不按 nnz），nnz 前重后轻时后半核约半数空闲，逐核 cycle 差距 20×+；用 msprof sample-based + ai_vector_core.db task_cyc 判别。"
tags: ["task.debugging", "topic.tiling", "topic.performance.analysis", "topic.compute.indexing"]
status: stable
sources: []
created_at: "2026-08-15T00:00:00Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 按行均分 tiling 致 nnz 倾斜时约半数核空闲

## 现象

在 Ascend950PR 上对 CSR→COO 类稀疏算子做 msprof sample-based 采集，查逐核负载。该算子 Host 侧 `ComputeBlockSplit` 用 `numBlocks = min(m, GetAivCoreCount())`、`blockSize = ceil(m / numBlocks)` 按行数均分多核，每核处理连续的一段行。

large 场景（m=65536，nnz=32768，even 分布但 nnz 集中在前半行）BlockNum=56。从 sample-based 输出的 `ai_vector_core.db` 的 `task_cyc` 表逐核统计 cycles：28 个忙核对 28 个空闲核，max/min ≈ 26.2×，(max−min)/max = 96.32%。约半数核因分到的行 nnz 为 0 或极少而几乎空转（绝对 Task Duration 数值见 manifest provenance，不写入卡正文）。

## 适用场景

- **工程形态**：性能瓶颈定位（msprof sample-based 逐核分析），非实现修复。
- **API 路径**：`msprof --ai-core=on --aic-metrics=PipeUtilization --task-time=on --sample-based=on`（sample-based 模式）；解析 `device_0/sqlite/ai_vector_core.db` 的 `task_cyc` 表（按 coreid 聚合 pmucount）。
- **验证过的平台与 CANN 版本**：Ascend950PR（npu-smi 25.7.rc1，56 AIV 核），CANN 9.0.0-beta.2。
- **适用粒度**：`operator_family`——适用于"Host 按行数均分多核、每核工作量由 nnz 决定且 nnz 逐行不规则"的稀疏算子（CSR↔COO、SpMV、SpMM 等 rowPtr 遍历族）。
- **路线优先级**：`recommended_fastpath`——对疑似负载不均的稀疏算子，先用 sample-based + task_cyc 逐核判别，再决定是否重构 tiling。
- **不适用或未验证场景**：稠密算子（每行工作量等长，按行均分天然均衡）；超大单行场景（BlockNum=1，无多核不均衡）；非 AIV-only kernel（ai_vector_core.db 路径未验证）。

## 判别方法

**触发**（tool_observed 失败）：稀疏算子多核场景总效率低于单核理论（核数翻倍吞吐未翻倍）；或 msprof `op_summary.csv` 的 Task Duration 随核数增加未呈线性下降。本卡触发信号来自 csr2coo 算子 large 场景 sample-based 采集：逐核 (max−min)/max = 96.32%。

**可执行探针**（verbatim 取自 trace 内调用）：

```bash
msprof --output=<out_dir> --ai-core=on --aic-metrics=PipeUtilization --task-time=on \
  --sample-based=on --ascendcl=on <perf_app> <m> <nnz> <warmup> <iters> <idxBase> <dist>
```

随后查 `ai_vector_core.db`（注意：AIV-only kernel 的逐核数据在 `ai_vector_core.db`，不在 `aicore.db`——后者 `task_cyc` 表为空）：

```bash
sqlite3 <out_dir>/device_0/sqlite/ai_vector_core.db \
  "SELECT coreid, SUM(pmucount) FROM task_cyc GROUP BY coreid ORDER BY coreid;"
```

计算 `max(sum)/min(sum)` 与 `(max−min)/max`。后者 > 0.5 即判显著负载不均。

**原始观测**（tool_observed）：

| 维度 | 值 |
|------|------|
| BlockNum（large, m=65536） | 56 |
| 忙核数 / 空闲核数 | 28 / 28 |
| max/min（逐核 cycles） | ≈ 26.2× |
| (max−min)/max | 96.32% |

（绝对 Task Duration 数值见 manifest provenance，不写入卡正文。）

**解释**（interpretation，agent 根因分析）：Host `ComputeBlockSplit` 按 `numBlocks = min(m, GetAivCoreCount())`、`blockSize = ceil(m/numBlocks)` 把行按**计数**均分到各核，每核处理 `[blockIdx*blockSize, (blockIdx+1)*blockSize)` 连续行段。各核工作量 = 该段行 nnz 之和；当 nnz 前重后轻（even 分布的随机性 + rowPtr 单调递增使前段行密集）时，前半核分到的 nnz 远多于后半核，后半核处理完空行后空转，cycle 仅为初始化开销。Host 侧无法直接读 Device GM 的 rowPtr/nnz 做按 nnz 切分（Host-Device 边界，此为 agent 推理的通用架构知识，本 trace 内未做 probe 直接验证该约束），故 tiling 只能按行数。

## 根因

Host 多核 tiling 按"行数均分"而非"nnz 均分"，当输入矩阵 nnz 分布倾斜（前重后轻）时，后半核分到的行 nnz 极少而空转，逐核 cycle 差距 20×+。

## 验证信号

本段信号来自 csr2coo 算子在 Ascend950PR 上的 msprof sample-based 采集（CANN 9.0.0-beta.2）。

- large 场景 BlockNum=56，逐核 (max−min)/max = 96.32%（来源平台：950；证据等级：tool_observed）
- 28 个忙核 vs 28 个空闲核，逐核 max/min ≈ 26.2×（来源平台：950；证据等级：tool_observed）
- AIV-only kernel 逐核数据在 `ai_vector_core.db` 的 `task_cyc` 表，`aicore.db` 的 `task_cyc` 为空（来源平台：950；证据等级：tool_observed）
- Ascend950PR `GetAivCoreCount()` 返回 56，非仓内注释的 40（来源平台：950；证据等级：tool_observed）

## 预防

对"每核工作量由 nnz 决定且 nnz 不规则"的稀疏算子，在 Host tiling 用按行数均分后，务必用 sample-based + `ai_vector_core.db` task_cyc 逐核核查负载均衡；若 (max−min)/max > 0.5，考虑 Device 侧 prefix-sum 切分或 SIMT 多线程路径。

## 失效边界

- "Host 无法读 Device GM 做 nnz 切分"是 agent 推理（通用 AscendC 架构知识），本 trace 内未做 probe 直接验证该约束；若某平台/框架提供 Host 直读 Device GM 的能力，本结论的 tiling 限制不复成立。
- Ascend910B（dav-2201）未采集，56 核与 40 核的差异、逐核不均衡形态推断可迁移但未验证。
- 按 nnz 切分 / Device tiling sink / SIMT 多线程的实测收益未验证（未实现）。

## 坏实践

- 见多核场景总吞吐不达预期，不查逐核 task_cyc，直接归因为"核数不够"或"单核慢"——实际是约半数核空转。
- 把 `aicore.db` 的空 `task_cyc` 当作"采集失败"放弃分析——AIV-only kernel 的逐核数据在 `ai_vector_core.db`，需换库查询。