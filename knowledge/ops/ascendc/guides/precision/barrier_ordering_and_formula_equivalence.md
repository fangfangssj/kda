---
type: Guide
title: Barrier 顺序与公式等价性
description: 区分数据依赖和数值舍入，识别Rstd及Modulate示例语义差异。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_04_pipeline_barrier.md
  role: primary
created_at: '2026-09-11T05:41:33Z'
updated_at: '2026-09-11T05:41:33Z'
platforms: []
---

# Barrier 顺序与公式等价性

BF16→FP32→计算→最终BF16舍入可减少中间舍入，但插入PIPE_V barrier只是执行顺序约束，不增加算术位宽。是否需要每条Vector之间显式barrier取决于真实API与隐式顺序；没写barrier不能单独证明结果错误。PIPE_V也不能代替MTE2写完后Vector读取或Vector完成后MTE3读取的跨流水发布协议，队列与SetFlag/WaitFlag须按实际生产者消费者配对。

RMS reciprocal std应为1/sqrt(sum(x²)/N+epsilon)。来源对照段只做到sqrt，若输出即rstd则语义不完整，不能把差异归因于精度或同步。平方、reduce、比例、epsilon、开方、倒数的dtype/顺序各自会影响结果；并不因要求高精度就必需每两条同流水指令之间插一个barrier。

Modulate示例专家段算x*scale+shift，对照片段先乘scale再加x再加shift，算x*(scale+1)+shift。Gemma类合同可能有1+scale，两者不能当成同一公式的性能或同步对比。应先确认真实算子语义，再保留对应加法；没有合同不能把额外x视为冗余。

alignedD表示物理容量时，不得自动把它当有效opCopyLength；最后块、padding初始化、输出index与GM有效长度要一致。向量计算结束与最终Cast完成后，才能发布给MTE3，buffer也只能在实际搬出完成后复用。验证需分别隔离公式、dtype、同步和尾块，避免一次改变多个因素却归因于barrier。
