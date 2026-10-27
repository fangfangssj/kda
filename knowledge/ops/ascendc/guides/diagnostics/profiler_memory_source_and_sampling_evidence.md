---
type: Guide
title: Profiler 不同观测层的数据不可替代性
description: 区分分配生命周期、访存计数、源码指令映射、PC 采样和时间线，明确各类性能证据的能力边界。
tags:
- task.debugging
- topic.performance.analysis
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/msopprof-visualization/references/data-contract.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-profiling/msopprof-visualization/references/visualization-contract.md
  role: supporting
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# Profiler 不同观测层的数据不可替代性

不同性能产物观测不同对象，不能因标题相似互相替代。固定来源将 MemoryDetail 计数器与 On-Chip Memory 的分配地址、容量、lifetime、bank/group 分开；在该工具中后者来自 `memory_info.json`。有某类内存读写计数，不足以推导具体 buffer 的地址区间、何时分配释放或哪两块内存同时存活。

源码热点需要与当前二进制一致的源码快照、文件和行号、指令地址/opcode/pipe，以及源码行到指令和反向关系。一个源码行可能对应多条指令，一条优化后指令也可能带复杂映射；仅列头文件或从二进制提取 printable strings 不提供该对应证据。调试信息缺失或制品版本不同，应报告无法定位，不能把最近的同名文件当作映射。

SIMT/PCSampling 的 Warp Stall 数据是相应执行与采样模式的证据，不等同于整个 kernel 的精确连续等待时间。样本数、采样对象、周期、支持的 stall 类别和缺采会限制解释；不能以 SIMD 指令数或通用 conflict ratio 填出未采集的 warp 状态。

指令时间线需要真实受支持的指令级事件，保留开始时间、时长、PID/TID/lane 和参数。汇总 duration 或周期比例不能重建唯一时间线，也不能凭视觉插值补出未记录事件。分配生命周期同样不证明这段时间里每个流水线都访问了该 buffer。

缺数据、有效零、部分数据和不支持需要区分。证据最低要求随待判断问题变化：定位容量峰值需真实同时存活区间，定位源码等待需可信映射与相应执行观测，分析流水重叠需时序；缺少其中一项就收窄结论。具体文件名和 schema 绑定来源工具版本，不作为跨所有 Profiler 的固定命名。
