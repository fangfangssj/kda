---
type: Guide
title: UB 计算链融合与队列交接
description: 保留中间值免 GM 往返、混合类型与多队列所有权的完整条件。
tags:
- task.optimization
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P69_ub_fused_vector_chain.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P69.yaml
  role: supporting
created_at: '2026-09-11T02:07:16Z'
updated_at: '2026-09-11T02:07:16Z'
platforms: []
---

# UB 计算链融合与队列交接

若一条n阶段Vector链原本每阶段均把同一个中间值写GM再读回，且所有阶段能在同一有效tile内执行，可把中间结果保留在UB，仅首次输入和最终输出访问GM。2n次降至2次只计算这条单输入输出链的数据传递；其他输入、参数、必要中间输出与跨tile依赖仍需计入。

中间值可以位于同一允许原地计算的区域，也可以在多个独立UB区间传递。链长度不是UB容量的直接代理：空间取决于同时活跃的值、dtype、scratch、padding和buffer份数，不能以大于4步机械判定必须拆链。支持的Cast可位于链中间，不能把“类型匹配”误写成一律禁止混合FP32/FP16。融合应保留原有转换与舍入节点。

单输入/输出队列可围绕整条计算链发布结果。多队列级联时，生产者EnQue后应由相应消费者DeQue获得可消费句柄，再在最后使用后释放；来源直接使用已经EnQue的bufB并FreeTensor、没有相应DeQue，不能视为完整合法协议。input的Free与output的EnQue先后也不是独立于依赖的固定语法规则，应看输入最后消费者和发布合同。

每条操作的in-place允许性、Vector内部依赖和MTE2→V→MTE3交接均需保持；有正确队列保证时不必为形式添加多余屏障，手动管理时则不能省掉真实依赖。注释把outQueue分配的Tensor称作VECCALC不会改变其真实逻辑位置。链中间可观察输出、共享旧状态和跨任务消费者不能因融合而消失。

核对逐节点dtype及最后消费者，验证尾部、允许别名、分离buffer和多轮队列状态，计总GM读写、UB峰值及端到端时延。仅grep到VECCALC或连续函数调用既不证明融合已完成，也不证明输入已正确搬入。
