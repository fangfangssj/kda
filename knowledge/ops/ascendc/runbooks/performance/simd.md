---
type: "Runbook"
title: "稀疏不规则输出算子 SIMD 标量退化诊断"
description: "每行输出长度不规则的稀疏索引算子在 SIMD 模型下退化为逐行标量 GetValue 循环，scalar pipe 占比 80%+；用 msprof PipeUtilization 的 aiv_scalar_time ratio 判别。"
tags: ["task.debugging", "topic.performance.analysis", "topic.programming_model.execution_model.simd", "topic.compute.indexing"]
status: stable
sources: []
created_at: "2026-08-18T15:59:15Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
aliases: ["csr2coo"]
---

# 稀疏不规则输出算子 SIMD 标量退化诊断

## 现象

在 Ascend950PR 上对一个 CSR→COO 行索引展开类稀疏算子做性能验收时，用 msprof `--aic-metrics=PipeUtilization` 采集 kernel 级流水线占比。该算子每行输出长度等于该行 nnz，而 nnz 逐行不规则。kernel 实现先在 Vector pipe 用 `Subs`/`Sub` 向量化预计算 per-row nnz 与 start offset，`PipeBarrier<PIPE_ALL>` 同步 V→S 后，进入标量 for 循环逐行 `GetValue` 读预计算结果再 `Duplicate`+`DataCopyPad` 写出。

采集到 4 个分布式场景（行数 m 从 128 到 65536，even 分布每行约 0.5 nnz）和一个超大行场景（m=1, nnz=100000）。分布式场景的 `aiv_scalar_time` ratio 一致在 0.88-0.90；超大行场景 scalar 降至 0.63、vec 升至 0.24。结论：分布式场景被标量 GetValue 循环主导，属 SCALAR BOUND。

## 适用场景

- **工程形态**：性能验收 / 性能瓶颈定位（msprof 采集分析），非实现修复。
- **API 路径**：`msprof --ai-core=on --aic-metrics=PipeUtilization --task-time=on`；解析 `op_summary.csv` 的 `aiv_scalar_time(us)` + ratio 列（第 41 列）。
- **验证过的平台与 CANN 版本**：Ascend950PR（npu-smi 25.7.rc1），CANN 9.0.0-beta.2。`GetAivCoreCount()` 在该设备返回 56。
- **适用粒度**：`operator_family`——适用于"每行输出长度由 nnz 决定且 nnz 不规则"的稀疏索引/格式转换类算子（CSR↔COO、CSC↔COO、SpMV rowPtr 遍历族）。
- **路线优先级**：`recommended_fastpath`——对疑似 SCALAR BOUND 的稀疏算子，先用本判别方法定位 scalar 占比，再决定是否走 SIMT 混合路径（见 ## 失效边界）。
- **不适用或未验证场景**：稠密 elementwise 算子（每行输出等长，可纯向量化，不触发标量退化）；超大单行场景（vec pipe 已占 24%，瓶颈形态不同）；非 AIV-only kernel（PIPE_ALL 与 ai_vector_core.db 路径未验证）。

## 判别方法

- **触发（tool_observed）**：稀疏/索引类算子 kernel 时间与行数 m 线性正比、与 nnz 弱相关；分布式场景明显慢于超大行场景（同等 nnz 总量下行数多者远慢）。msprof PipeUtilization 采集后分布式场景 `aiv_scalar_time` ratio ≥ 0.80。
- **探针（executable check）**（verbatim 取自 trace 内 msprof 调用）：
  ```bash
  msprof --output=<out_dir> --ai-core=on --aic-metrics=PipeUtilization --task-time=on --ascendcl=on \
    <perf_app> <m> <nnz> <warmup> <iters> <idxBase> <dist>
  ```
  解析 `op_summary.csv`：`aiv_scalar_time(us)` 列（第 41 列）后的 ratio 即 scalar pipe 占比。分布式（even）场景 ratio ≥ 0.80 即判 SCALAR BOUND。对照采集一个超大行场景（m=1, 长 nnz），其 scalar ratio 明显下降、vec ratio 上升，可佐证瓶颈形态是"逐行标量循环"而非"搬运/写出"。
- **观测（tool_observed，本 trace 实测 950）**：

  | 场景 | m | nnz | 分布 | scalar ratio | vec ratio |
  |------|------|--------|--------|--------|--------|
  | small | 128 | 64 | even | 0.889 | 0.006 |
  | medium | 4096 | 2048 | even | 0.903 | 0.007 |
  | large | 65536 | 32768 | even | 0.901 | 0.007 |
  | ultrarow | 1 | 100000 | onerow | 0.628 | 0.237 |

  （绝对 Task Duration / aiv_time 数值见 manifest provenance，不写入卡正文。）
- **解释（interpretation）**：每行输出长度等于该行 nnz，而 nnz 逐行不规则，无法用单条 Vector 指令统一处理多行；kernel 已在 Vector pipe 用 `Subs`/`Sub` 把 `rowPtr[i+1]-rowPtr[i]` 与 `rowPtr[i]-idxBase` 向量化预计算进 UB buffer，但读取预计算结果仍需逐行标量 `GetValue`（标量从 UB 读，高延迟），随后 `Duplicate`+`DataCopyPad` 写出。标量 GetValue 循环随行数线性增长，占据 80%+ scalar pipe 时间。超大行场景只有 1 行，标量循环仅 1 次 GetValue，瓶颈转移到 vec Duplicate 写出，故 scalar 降、vec 升。

## 根因

每行输出长度不规则（nnz 逐行变化）使 SIMD 框架无法跨行向量化输出写入，被迫退化为"逐行标量 GetValue 读预计算结果 + 单行 Duplicate/DataCopyPad 写出"的标量循环路径；该路径随行数 m 线性增长，scalar pipe 占比 80%+。

## 验证信号

- 分布式（even）场景 scalar pipe ratio 0.88-0.90，4 个规模一致（来源平台：950；证据等级：tool_observed）
- 超大行场景 scalar ratio 0.63、vec ratio 0.24，瓶颈形态切换（来源平台：950；证据等级：tool_observed）
- kernel 时间与行数 m 线性正比、与 nnz 弱相关（来源平台：950；证据等级：tool_observed）

## 设计决策

kernel 已选"Vector pipe 向量化预计算 per-row nnz/start + 标量循环 GetValue 读取"的混合路径（仿 gtsv2_strided_batch 的小 m 标量回退写法），而非"循环内逐元素减法"。该决策把减法从标量循环移到 Vector pipe，但 GetValue 读取仍留在标量循环——因为输出写入长度不规则，Vector 指令无法统一处理多行。标量循环本身是 SIMD 框架的结构性限制，编码层优化（成员变量外提、循环内对象外提、idxBase 模板化等，参考 ascendc-performance-best-practices scalar P4/P9）预计只能降 15-30%，scalar 占比仍占 60%+。

## 性能影响

同一算子内部两条路径的相对代价：分布式场景标量循环路径（scalar 80%+）相对超大行场景的 Duplicate+DataCopyPad 路径（scalar 63%/vec 24%）显著更慢——同等 nnz 总量下，行数多者因逐行标量 GetValue 循环而呈线性退化（参照路线：超大行 SIMD 路径；来源平台:950）。

## 预防

对"每行输出长度由 nnz 决定且不规则"的稀疏算子，在 SIMD 实现完成后先用 msprof PipeUtilization 采集 scalar ratio；若 ≥ 0.80，不要在编码层优化（成员外提等 15-30% 收益）上过度投入，应直接评估 SIMT 混合路径或 Host 侧负载均衡重构。

## 失效边界

- Ascend910B（dav-2201/arch22）与 Ascend950DT 未在本 trace 内采集，SCALAR BOUND 形态推断可迁移（同为 AIV SIMD 模型，scalar pipe 语义一致），但 scalar ratio 具体值未验证。
- SIMT 混合路径的实测收益未验证（未实现）；repo-op-templates simt/sparse 模板存在但本 trace 内未跑通改写版。
- 非 AIV-only kernel（含 AIC cube pipeline）的 `op_summary.csv` 列布局与 `ai_vector_core.db` 逐核数据路径未验证。

## 坏实践

- 见 msprof 采集后 scalar ratio 80%+，不先评估 SIMT/负载均衡架构级路径，只在编码层做成员变量外提等微调，预期把 scalar 从 90% 降到 60% 仍是瓶颈。
- 把超大行场景（scalar 63%）的结论外推到分布式场景——两者瓶颈形态不同，优化方向相反（超大行应优化 Duplicate/写出，分布式应消除标量循环）。