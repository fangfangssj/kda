---
type: "Runbook"
title: "mte2/mte3 wait 双高示读写串行无重叠"
description: "纯搬运算子 CONTIGUOUS 场景无数据放大（1x）但聚合带宽仍只有峰值 ~16%：ResourceConflict 显示 mte2_wait 89.76%/mte3_wait 88.44%，读（MTE2）与写（MTE3）完全串行、无双缓冲流水重叠，且单条搬运仅 1.67KB；三探针判别。"
tags: [task.optimization, topic.implementation.pipeline, topic.data_movement.data_copy]
status: stable
sources: []
created_at: '2026-09-16T11:08:02Z'
updated_at: '2026-09-16T11:08:02Z'
platforms: ["950"]
---

# mte2/mte3 wait 双高示读写串行无重叠

## 现象

在 Ascend950PR 上对 BenchNoHarnessStridedSlice 做性能验收：CONTIGUOUS 场景（[1024,1024] fp16 strides=[1,1] 连续全量拷贝，无任何数据放大——理论 4MB、实测 4.03MB）聚合带宽仅约为 1.8TB/s 参考峰值的 ~16%（该峰值本身未经验证，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)），远低于 >60% 目标。msprof 汇总（summary.txt，源自 PipeUtilization/ResourceConflict CSV，节选判别信号行）显示：

```
--- PipeUtilization (56 cores, prefix=aiv) ---
  mte2_ratio%                 56.95      60.62      63.60
  mte3_ratio%                 28.66      30.44      32.99

--- Memory ---
  Avg MTE2 transfer: 1.67KB (1247 instructions total)

--- ResourceConflict ---
  vec_wait: 0.00% | mte2_wait: 89.76% | mte3_wait: 88.44%
```

MTE2（GM→UB 读）与 MTE3（UB→GM 写）的 wait 比例同时接近 90%：两条搬运队列都在互相等对方，读和写没有流水重叠。同批 STRIDED 场景（mte2_wait/mte3_wait 99.2%/99.2%）同样完全串行。

## 适用场景

- 工程形态：AscendC 向量核纯搬运/访存密集型 kernel 的性能瓶颈定位（msprof op 采集 + ResourceConflict/PipeUtilization 解读）。
- API 路径：kernel 侧 MTE2/MTE3 搬运组织（DataCopyPad/TBuf/SetFlag-WaitFlag 同步）；msprof `ResourceConflict.csv`（mte2_wait/mte3_wait）、`PipeUtilization.csv`（mte2/mte3 ratio）。
- 验证过的平台与 CANN 版本：Ascend950PR（ascend950），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——向量核读写搬运流水化的通用判定信号（与算子无关）。
- 路线优先级：recommended_fastpath（排查带宽不足时的第二步：先排除数据放大，再查流水串行）。
- 明确不适用 / 未验证场景：计算型算子（VEC 占比高）瓶颈不在本信号；Cube 路线未验证；双缓冲优化收益未实测（见遗留）。

## 判别方法

触发条件：纯搬运算子聚合带宽远低于峰值，且已排除数据放大（实际搬运量 ≈ 理论搬运量）。

可执行探针（msprof 汇总的三个量）：

1. `mte2_wait` 与 `mte3_wait` 同时高（本例 89.76% / 88.44%；STRIDED 场景 99.2% / 99.2%）——读队列与写队列都在空等，即读写无重叠。
2. `mte2_ratio` 与 `mte3_ratio` 交替占满（60.6% / 30.4%，相加 ~91%）——时间片上读一段、写一段地轮流执行，而不是同时进行。
3. `Avg MTE2 transfer` 偏小（1.67KB/指令，1247 条）——单块搬运粒度小，未合并成大块（大块连续搬运推荐 ≥16KB）。

解释：kernel 逐行搬运（每条 ~1 行 1024 元素 = 2KB）且读-写串行（MTE2 搬完等 MTE3 搬完），总线任何时刻只有一半时间在一个方向上传数据；再叠加小块高频发射的固定开销，聚合带宽被封在峰值的 ~16%。

## 设计决策

（本卡为诊断规则，无代码修复落地；优化方向按验收报告记录，供接续。）

- [P1] 合并连续段为大块 DataCopy：多维连续（本例整个 [1024,1024] 连续）时应合并为单次大块搬运（≥16KB），预期带宽利用率提升 3~5x（报告估值，未验证）。
- [P1] 启用 MTE2/MTE3 双缓冲流水：检查 `InitBuffer` 的 bufNum=2 是否生效，确保读块 N+1 与写块 N 重叠执行。
- 两项均未实施，本开发会话 无前后对比数据。

## 系统信息

- 环境：Ascend950PR（56 VectorCore，1650/1650MHz 满频），CANN 9.1.0，msprof op --warm-up=10。
- 采集物：`docs/perf/round_001/scene1_contiguous/`（8 CSV + summary.txt）；场景2/3 同构目录。
- 相关文件：kernel 实现 `op_kernel/`（TBuf + 手动 SetFlag/WaitFlag 同步、无 AllocTensor/FreeTensor——代码检视确认非 TQue 双缓冲形态）；汇总 `docs/perf/round_001/summary.txt`。

## 预防

- 纯搬运 kernel 的搬运组织默认按"大块合并 + 双缓冲流水"设计，逐行小块 + 串行同步的形态在验收前先做 wait 指标自检。

## 验证信号

来源说明：性能验收 round_001 三场景 msprof CSV 汇总（同机同日）。

- 场景1 CONTIGUOUS：mte2_wait 89.76%、mte3_wait 88.44%（56 核汇总）；mte2_ratio avg 60.62%、mte3_ratio avg 30.44%；Avg MTE2 transfer 1.67KB（1247 条指令）；聚合带宽 ≈ 1.8TB/s 参考峰值的 16.2%（峰值口径未验证，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)）；数据放大 1x（4.03MB vs 理论 4MB）（来源平台：950；证据等级：tool_observed）。
- 场景2/3 STRIDED：mte2_wait/mte3_wait 99.2%/99.2% 与 99.29%/99.29%（完全串行，与逐元素放大叠加）（来源平台：950；证据等级：tool_observed）。
- 头开销排除：Task Duration 相对最长核 aiv_time 的头开销占比仅 4.0%——瓶颈不是 launch 开销（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "mte2_wait/mte3_wait 双高 ⇒ 读写串行"的判定适用于纯搬运形态；若 vec_wait 也高，瓶颈可能在计算或同步，须先看 vec_wait。
- 1.67KB 粒度、3~5x 提升估值均为本用例（1024 宽 fp16 行）的观测/报告估值，不同 shape/dtype 不同。
- 双缓冲/大块合并的收益未实测，不能当已验证结论引用。
- 推断可迁移（未验证）：向量核平台的 mte2/mte3 wait 语义预计一致；其他芯片未实测。

**未决项**：

- 实施大块合并 + 双缓冲后复测 mte2_wait/mte3_wait 与聚合带宽，验证串行归因（优先项）。
- `InitBuffer bufNum=2` 在 TBuf 形态下是否等价于双缓冲流水未在代码层核实（代码检视为 TBuf + 手动同步）。
## 坏实践

- 把"带宽低"一律归因于数据放大或总线竞争——CONTIGUOUS 场景无放大仍只有 ~16%，不看 wait 指标会漏掉流水串行这一独立根因。

## 误诊

- 无（本闭合从 ResourceConflict/PipeUtilization 数据直接判定；优化未实施，无错误修复轮次）。
