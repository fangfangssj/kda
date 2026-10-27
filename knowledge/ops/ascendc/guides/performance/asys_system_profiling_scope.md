---
type: Guide
title: asys 系统采样的参数范围与 msprof 下发身份
description: 区分系统级采样和算子级度量，保留采集类型、单设备参数、PMU 条件及周期/时长表述的核验边界。
tags:
- task.optimization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/profiling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/tools/asys-toolkit/references/constraints.md
  role: supporting
created_at: '2026-09-14T07:58:31Z'
updated_at: '2026-09-14T07:58:31Z'
platforms: []
aliases:
- asys profiling
- sys-period
---

# asys 系统采样的参数范围与 msprof 下发身份

固定 reference 的 `asys profiling` 在底层下发 msprof，用于观测系统/设备资源情况。它可以辅助 Ascend C 性能排障，但输出不自动等于单个算子的调用耗时、端到端延迟或每 step 指标。本卡未运行采集，参数与产品范围需要对应真实工具版本。

## 类型与参数合同

命令形态为 `asys profiling -r=类型 -p=数值 -d=ID --aic_metrics=指标 --output=目录`：

| 参数 | 来源合同 |
|---|---|
| -r | 必填；可用逗号组合多个类型 |
| -p | 必填，来源单位秒，范围 `[1,30*24*3600]`，实际周期/采集时长含义见下文 |
| -d | 只接受一个 Device ID，默认 0；不因 -r 多选就接受设备列表 |
| --aic_metrics | 只有 -r 包含 aicore 时生效 |
| --output | 目录前缀，生成 `asys_profiling_result_timestamp`，默认执行目录；空/无效/无权限/创建失败会退出 |

| -r 类型 | 主要观测 |
|---|---|
| dvpp | DVPP 时间和利用率等 |
| aicore | Cube/Vector 指令及计算/搬运单元性能 |
| os | 系统内存、AI CPU、Ctrl CPU 等 |
| memory | 内存访问速率、带宽及片上/缓存相关信息 |
| link | 集合通信、PCIe 等带宽 |
| power | 来源归为低功耗数据 |

可选 PMU 为 PipeUtilization（默认）、ArithmeticUtilization、Memory、MemoryL0、MemoryUB、ResourceConflictRatio、L2Cache、MemoryAccess，分别定位计算/搬运利用、算术活动、不同层级读写、冲突、缓存命中和访存量。精确 CSV 字段及单位复用 [msprof 字段卡](msprof_csv_fields_units_and_diagnostic_limits.md)，不因枚举同名就推断不同版本的全部列完全一致。

## 用实际下发命令确认采集口径

来源示例在 -r=aicore、-p=10、-d=0 时回显：

```text
msprof --output=./ --sys-period=10 --sys-devices=0 --ai-core=on --aic-mode=sample-based --aic-metrics=PipeUtilization
```

该日志明确给出 sample-based 模式、设备和 PMU，是确认实际调用的入口。来源参数表把 -p 称为“采集间隔”，日志又提示大约等待相同秒数；仅凭它们不能断言 -p 是每个样本间隔、完整采集持续时间或任务超时。应核对本次下发的 --sys-period 定义、实际起止时间和导出记录，不能从 -p=10 算出每秒 0.1 个样本或 10 秒业务耗时。

输出日志中的 `Data is saved in` 指向实际 PROF 产物；外围目录创建成功不证明内层数据完整。对照运行任务、Device、采集模式、参数和有效时间范围，再分析资源变化。单算子 PMU/耗时需要匹配算子级采集，区别见 [msprof 模式与身份](msprof_capture_modes_and_trace_identity.md)；torch_npu 的 [op_statistic 每步聚合](torch_npu_profiler_statistic_aggregation.md)使用另一种分母，不能混用。

来源支持表限定该能力为 A2/A3 与 950，不支持 Atlas 训练/推理系列及 200I/500 A2；还需对应实际部署与权限核验。是否捕获到业务活动应由原始时间范围和任务身份确认，不能只由工具退出成功或高低利用率倒推。
