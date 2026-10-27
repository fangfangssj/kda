---
type: Guide
title: torch_npu.profiler 的 op_statistic 聚合与采集窗口
description: 解释 Total Time(us) 求和、实际采集步数归一化、repeat 导出选择和 CSV 身份，区分 OP 累计时间与墙钟耗时。
tags:
- task.optimization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/skills/aog-perf-eval/references/REFERENCE_PROFILER_AND_METRICS.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/torch-ops-profiler/references/REFERENCE_PROFILER_AND_METRICS.md
  role: supporting
created_at: '2026-09-14T07:15:54Z'
updated_at: '2026-09-14T07:15:54Z'
platforms: []
aliases:
- op_statistic
- per_step_metric
---

# torch_npu.profiler 的 op_statistic 聚合与采集窗口

固定来源的性能基准使用 `torch_npu.profiler.profile`、`schedule` 和 `tensorboard_trace_handler`，导出路径形态为 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv`。这是该来源工具链的产物约定；是否导出、目录层级和字段名称仍需对应实际 torch_npu/Profiler 版本。本卡解释 Ascend C 算子基准中的统计量，不把导出成功视为采集对象和窗口已正确。

对同一用例、同一实现、同一次采集窗口，令 R 为纳入的 OP 类型行集合，N 为这些记录实际覆盖的 active 步数，定义：

```text
sum_total_us = sum(row["Total Time(us)"] for row in R)
per_step_metric = sum_total_us / N       # N > 0
```

来源默认计入每一行，若只保留目标 Kernel 或排除辅助 OP，必须说明行筛选条件。`Total Time(us)` 已是该行累计量，不能再按调用次数重复相乘；平均时间列、调用次数列和累计时间列也不能仅按名称近似替换。OP 时间可能重叠，CPU/NPU 活动和统计行是否嵌套取决于导出定义，因此该和表示所选 OP 的累计耗时，并不自动等于每步端到端墙钟。比较 custom/native 时应保持相同的行范围、算子工作量与输出语义。

## active、repeat 与多份导出

`schedule` 中 wait、warmup、active、repeat、skip_first 的作用应和循环及每轮末尾的 `prof.step()` 对齐。分母只数进入当前 CSV 统计窗口的有效采集步，不把 warmup、wait、skip_first 或未完成迭代算进去。来源的 `repeat*(wait+warmup+active)` 循环公式未覆盖任意非零 skip_first、截断运行或其他 schedule 行为，不能不核对实际窗口就机械使用。

只有收齐对应的全部 repeat 且每次均含 active 个有效步时，才可取 `N=active*repeat`。若多次回调生成多份 CSV，应记录每份的窗口、是否累计及去重方式。只按修改时间选最后一份文件可能只选到最后一个窗口，也可能选到旧运行；它不能证明分子包含所有 repeat。

例如两个窗口各采 5 步、累计 OP 时间各 50 us，全部窗口合并时为 `(50+50)/(5+5)=10 us/step`；只选择第二个窗口仍应为 `50/5=10 us/step`。把最后一份 50 us 除以全程 10 步会误报 5 us/step。窗口样本数不等时，整体统计应为 `sum(T_j)/sum(N_j)`，简单平均各窗口的 `T_j/N_j` 会改变权重。这些是口径反例，不是硬件实测。

## CSV 解析与测量身份

表头可能含 UTF-8 BOM。解析前显式处理 BOM 和外围空白，并按已确认的 schema 映射等价列名；不能同时累加同义列，也不能把缺列、空值、不可解析数值、无有效步或采集失败补成 0 us。保留原始单位，`us` 与 `ms` 换算不能只靠相似列名。

每份结果需关联设备、进程/任务、用例、输入 dtype/shape、实现、二进制及采集参数。不同实现或旧运行的目录混入后，即使 CSV 格式合法也不能用于本次比较。每次 profile 对应独立输出身份；是否保留、清理或归档旧产物属于运行器实现，判读只接受身份一致的数据。更多任务/采样边界见 [msprof 采集模式与结果身份](msprof_capture_modes_and_trace_identity.md)。

验证聚合时至少对照单窗口、多窗口、只选末窗口、不同有效步数、带 BOM 表头和缺列输入；重新按原始行求和应可复算报告值。固定来源的 warmup=5/active=5、Markdown 分节及文件命名属于基准约定，不是 Profiler 的固定能力限制。本卡未执行 NPU 采集，具体 SDK 调用仍以实际安装的签名和产物核验。
