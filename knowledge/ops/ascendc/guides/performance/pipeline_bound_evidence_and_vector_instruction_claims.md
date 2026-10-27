---
type: Guide
title: 流水瓶颈证据与向量指令说明边界
description: 保留访存、Scalar、Vector和低利用率诊断，修正计数与API推断。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/single-core-pipeline/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/single-core-pipeline/memory.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/single-core-pipeline/no-bound.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/single-core-pipeline/scalar.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/single-core-pipeline/vec.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/math_compute/exp_interface/exp.md
  role: constraint
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# 流水瓶颈证据与向量指令说明边界

高MTE2占比可能是实际搬运、排队或等待通信，MC2挂在MTE2上的CrossCoreWait尤其需分离；但确认存在通信等待也不代表所有内存优化都无效，真实路径可能混合受限。查看同任务的带宽、有效字节、计算强度与时间线，而非只按ratio路由。低带宽还可能来自小事务、stride、核数不足或工作集无复用，不能一律转L2缓存优化。

较大连续搬运、合法DataCopyPad批量行、消除中间GM往返、调整访问顺序与UB驻留可以成为候选；UB tile可放不代表全rank/多核工作集可放L2。相邻核访问相邻地址不证明共享同一cacheline或产生数据复用，块大小、bank冲突和cache容量需实际地址分析。

各单元利用率低可能来自关键依赖、Scalar/调度、短任务或采集口径，不是“没有瓶颈”。DoubleBuffer/Preload要有独立合法槽和生产消费顺序，计算时间≥搬运时间有助掩盖但不是任何收益的必要条件。MTE2/Vector交替出现不等于时间重叠，必须看跨tile时间区间；MMAD→FIX的UnitFlag仍是同步协议，不是无开销，也不能替代无关barrier。

Scalar优化可减少重复地址计算、循环/分支、无谓Scalar↔Vector读回与动态tiling解析；LocalTensor是视图，不自动替代TBuf分配，去TPipe/队列需显式承担内存与事件生命周期。aic_scalar_ratio≥0.15只是案例筛选，不保证常量折叠有收益；固化shape/上界需实际合法分发，循环展开也有代码体积/寄存器代价。

Vector ratio50/65/80/90%、Cast20%或超越函数30%是来源诊断阈值，不是硬件饱和线；高活跃不意味着有效lane/吞吐接近峰值。Chrome Trace不能固定pid5当Vector，应读取进程/线程metadata；Regbase可有RVECEX/RVECLD/RVECST/RVECSU等不同轨道，原始单元名按版本确认。

两段Compute之间的MTE3/MTE2不一定搬同一中间结果，要对照地址/数据流。源GM→UB→Compute1→GM→UB→Compute2→GM实际4次GM传输，而非所称6次；融合后2次的净收益还需中间结果是否有外部消费者。寄存器融合与UB驻留是不同层级，跨核依赖不能被核内融合省略。

VMULS通常表示标量乘，不能按来源表当VMUL+VSUB的乘减融合；VMULA/VMADD等需核实具体API公式、dst修改和舍入。Exp/Log等不都由数十FMA的Taylor实现，basic Exp与高级Exp不同；高级taylorExpandLevel0只表示不使用Taylor，现有API卡不支持断言必定LUT+插值，更不能推广Log/Tanh同参数。降低展开项数、LUT或混精度需明确输入域与误差验证。

192/248KiB UB及双缓冲总UB除2都只适用于对应资源模型，固定scratch/常驻区与各队列份数分别核算。性能归因须完整数值正确、同口径测量，来源流程的必须先后/报告格式保留在Skill。
