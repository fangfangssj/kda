---
type: Guide
title: FA 复合 Causal Varlen 与 Sink 数学
description: 修正局部列坐标、有限Mask与Sink分子缩放。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-compound-features.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/flash-attention.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/a2-a3-flash-attention-stage-design.md
  role: supporting
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# FA 复合 Causal Varlen 与 Sink 数学

等长左上causal为c0+j≤r0+i，因此局部最后可见列lim=r0+i-c0。来源vis0=r0-c0却写lim=i-vis0，符号反了；vis0≤0的注释也与c0≤r0矛盾。整块全可见需最大key≤最小query，整块全不可见需最小key>最大query，其余都需行mask。不能仅用某个对角chunk不等式覆盖边界等号或跨多个chunk的query块。

右下causal还需实际sk-sq偏移，varlen用各batch长度。因果可省约半矩阵仅对近似方形长序列且忽略块/启动开销，不能要求时间恰好减半。source NO_MASK0/MASK_SPEC1/MASK_CAUSUAL2是内部枚举，不等同外部sparse_mode2/3的全部语义。

无效K/V补0仍使score0对应exp非0；必须在softmax有效集合上屏蔽。-3e38是有限sentinel，对全部无效行会相减得0并产生概率，极大合法score/负scale也可破坏假设。若mask在Muls前写入，scale0会抹掉mask，负scale会翻转；按已缩放score施加真实mask并定义全mask/非有限策略更明确。

前缀varlen mask只限制KV，Q的实际长度及输出域仍需单独处理。预建每batch K长mask可减少Scalar构造，但需B*K*dtypeBytes容量及数据更新失效条件；逐batchlaunch、批界截断和连续TND是不同组织，不能无条件宣称完全等价。

learnable sink若定义为只进分母的虚拟logit z，无V贡献：mF=max(m,z)，alpha=exp(m-mF)，lF=alpha*l+exp(z-mF)，未归一化oF=alpha*o。来源向量链更新m/l却未展示o缩放，必须在RescaleO中确认仅一次对应缩放，否则z>m时结果错；也可对已归一化旧输出按等价权重校正。sink处于哪个scale单位由接口决定，不能把原始logit与已缩放logit混用。

aux smax/ssum来源按每行8FP32物理槽输出，实际有效槽/复制布局需匹配消费者；Brcb+slot0掩码写法与同文Brcb不可靠断言冲突，按具体重载验证。可选null输出应由分支避免写或分配真实内部缓冲，不能把null直接传给仍写aux的Kernel。
