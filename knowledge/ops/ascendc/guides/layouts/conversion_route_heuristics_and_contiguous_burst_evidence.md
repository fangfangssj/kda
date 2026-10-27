---
type: Guide
title: 重排路线启发式与连续搬运证据
description: 保存NLast和片上转置的设计选择，限定固定阈值与性能归因。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/conversion/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/conversion/fallback/index.md
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# 重排路线启发式与连续搬运证据

末轴参与转置时，可比较连续GM→UB、UB内VCONV/Gather/VF重排、连续UB→GM，与NDDMA多维stride搬运。末轴不转置且存在可合并的连续子轴时，N_LAST可减少短stride写回。需要真实perm/inv_perm和输入输出物理stride证明哪几轴可合并，不能仅按C大小或算子名决定。

VCONV_5HD的2D16bit、021的perm[0,2,1]、Gather的目标平台/使能条件都是来源特定模板筛选；其他dtype和特殊布局需重新判断。≤5维与BIG_DIM外循环是NDDMA能力组织，不能据source“C<32永不BIG_DIM”取消真实实现条件；合并维度也不必使所有读写完全连续。

原文4MiB→SIMT、更新建议约10KiB且C≥16、SIMT带宽6.25%或≥50%、VF派发约4us、慢3–10倍均未给本次完整trace/平台/数据证据，不当普适阈值。N_LAST并非始终优于SIMT；微小任务、协同线程访问、对齐、cache和寄存器成本不同可改变结果。线程逐元素访问也可能跨线程合并，不能从单线程load大小直接推带宽率。

源决策树可合并组数≥2与=0没有覆盖=1，且计数口径不明，不是完整可执行分发。应将每策略写成明确能力谓词与完整fallback，并保持输入输出逐坐标正确后测同一case完整延迟。Concat/Split在该fallback未实现，不能根据族入口自动宣称支持。
