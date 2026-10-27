---
type: Guide
title: Profiler 指标口径与瓶颈假设
description: 从任务身份、采集范围和原始单位解释流水指标，区分利用率信号、理论估算与可验证的瓶颈原因。
tags:
- task.debugging
- topic.performance.analysis
- topic.implementation.pipeline
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/standards/csv_systematic_analysis_framework.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/standards/op_summary_header_guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-issues-general.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-issues-aiv.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-simulator/references/performance-issues-aic.md
  role: supporting
created_at: '2026-09-10T13:50:53Z'
updated_at: '2026-09-14T04:19:37Z'
platforms: []
---

# Profiler 指标口径与瓶颈假设

解释 Ascend C profiling 数据时，先确认设备、任务、stream、输入组合、执行模式和采集版本。来自不同任务或采集轮次的 AIC/AIV 指标不能仅按行号拼接；Task ID 也需结合其设备与上下文判断身份。

task duration、某条流水的活动时间、等待时间、利用率和理论耗时分别表达不同量。理论值由工作量与所用硬件模型估算，不能因字段名是 `aicore_time` 或 `aiv_time` 就称为理论值。百分比和 0–1 比例在使用前应核对原始字段单位，避免归一化两次。

AIC、AIV、MTE 等单元可以重叠执行，因此各活动时间或 ratio 通常不能相加作为整个任务耗时。混合执行下的 Block 与子执行单元应按实际采集格式关联；来源中 cube0/vector0/vector1 只是特定导出的标签，不能推广到每个平台或每种 Kernel 类型。

空值、未采集、该模式不适用和真实零值需要区分。汇总 CSV 没有 Block 级记录时，无法据它直接得到跨 Block 方差；需要相应粒度的采集。只有汇总均值也可能掩盖尾块或局部等待。

高 MTE 活动比例可提示搬运开销，仍需区分实际传输量、带宽、请求粒度和重复加载。低计算比例或低平均带宽不能单独排除局部饱和；高等待也不能唯一确定依赖链、碎片访问或同步错误。UB 带宽稳定并不证明 UB 无瓶颈，Vector 总冲突指标也不能无条件解释为 Bank 冲突。

把这些信号作为待验证假设时，应关联到具体源代码、地址/布局和流水依赖。例如检查较高 MTE2 时间是否伴随不必要的重载，检查跨 Block 耗时差异是否与尾块工作量一致，再用范围明确的改动和相同输入复测。理论搬运估算可参考[性能分析](../operator_practice/performance_analysis/analyze_performance_data.md)，其中平台参数必须重新核对适用范围。

判断优化效果以相同语义的实际耗时和正确性为依据。利用率增加不自动等于更快，减少同步也不自动保持正确；来源中 0.3/0.5/10% 等诊断分界是启发式例值，不能变成全平台的瓶颈判定规则。

## 比例变化、慢核和访存计数的反例

活动比例同时受分子和分母影响。例如 Scalar 忙时维持 10 ticks，kernel 由 100 降到 50 ticks，比例会从 10% 升到 20%，此时总时延改善，不能仅凭比例上升判定 Scalar 退化。相反，其他流水变慢也会让它的比例下降。来源把不同流水按固定顺序排查，适合作为导航，不能规定某一流水在所有 Kernel 中永远是主因或伴随结果。

某核 CUBE 利用率最低不等于该核最后完成：工作量更少、开始更晚、提前结束、等待以及采集窗口均会改变比例。识别慢核需实际任务/Block 身份、逐核起止与有效工作量；只有各核均值或缺失的 per_core 数据，不能推导是哪一核拖长临界路径。单核样本也不能证明其他核未启动。

Load/Store 与算术指令数之比可提示中间量往返，但输入加载、最终输出和合法 Gather 同样贡献计数；需证明确实写出再读回同一中间量。Scalar 访存还可能来自合法状态或参数读取，不能仅由次数判定寄存器溢出。Cache miss 为零或平均带宽低，既不证明不存在短时队列阻塞，也不能排除其它未被该指标观察的访问路径。
