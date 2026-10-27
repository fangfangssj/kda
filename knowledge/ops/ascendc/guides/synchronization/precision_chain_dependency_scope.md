---
type: Guide
title: 精度转换链中的同步作用域
description: 区分 Cast 数值误差与缓冲依赖，明确同流水、跨流水和跨核同步不能按名称互换。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A4_pipeline_barrier.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A4_pipeline_barrier.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A4.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# 精度转换链中的同步作用域

Cast→Compute→Cast 是一条可能具有读写依赖的计算链，增加位宽与选择舍入并不能代替同步。应标明每步真正使用的执行流水、访问地址和存储生命期，再判断相邻操作是否存在生产后读取、读取完成前覆写或写写冲突。

普通向量 Cast 的名字不意味着其运行在 MTE2 或 MTE3。来源将向量 Cast 标为 MTE2、将下落 Cast 标为 MTE3 并配 MTE2_V/V_MTE3 的示例，不能确立真实完成关系。MTE2 CopyIn 完成后供 Vector 消费，以及 Vector 完成后供 MTE3 CopyOut 消费，才是需要按各自实际接口确认的两类跨流水交接。

PIPE_V 只处理其规定的 Vector 流水作用域，不会发布 GM 或跨核数据。跨流水 Set/Wait 事件与跨核同步各有合同，也不能一律互换为 SyncAll。RegBase VF 内的寄存器保序及 UB LocalMemBar 又有单独规则，见[同步层次](../regbase/synchronization_layers.md)。

独立操作没有共享存储依赖时，不因相邻就必须加屏障；有编译器或队列保障的依赖也应基于具体合同判断。循环间复用 buffer 时必须等最后一个消费者完成，不能只因为作用域跨循环就推断需要全核同步或全局可见性。删除重复屏障前应证明其对应依赖仍被满足。

来源“一条屏障固定 1–2 周期”、看到 grep 命中就已同步、没有 Cast 就无需检查同步的条件都不足以建立正确性或性能结论。验证不同 dtype、非整块尾部、多轮缓冲复用及重复调用；同步修改后误差消失表明时序相关性，仍需定位具体数据依赖，不能用它证明 RoundMode 或算法已正确。
