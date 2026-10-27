---
type: Guide
title: Varlen 任务前缀与 Sparse 边界
description: 保存四类累计量、TND偏移与尾块双重约束。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-varlen-handcraft-recipe.md
  role: primary
created_at: '2026-09-11T05:21:50Z'
updated_at: '2026-09-11T05:21:50Z'
platforms: []
---

# Varlen 任务前缀与 Sparse 边界

该s1s2路径的长度输入为B项累加终点，差分得到qLen_b/kvLen_b。任务数sum_b ceil(qLen_b/s1Base)*n2G，其中n2G代表KV头与GQA组的组合数量；按实际长度切任务避免给短序列分配padding任务。整数范围、0长度batch和全部空输入需在ceil除法/索引前处理。

定位任务起点和跨batch推进时同时维护任务前缀s1OuterSizeAcc、Q token前缀s1SizeAcc、KV token前缀s2SizeAcc、显式逐batch mask元素前缀attenB1SSOffset=sum(qLen*kvLen)。跨多个空/短batch要while且守住boIdx<B，不能单次if。若mask是广播或压缩存储，最后那个乘积前缀应改用其真实布局，不能无条件使用。

局部task经qBlocks=ceil(qLen/s1Base)解码n2o=(idx-prefix)//qBlocks//g、go=(idx-prefix)//qBlocks%g、s1o=(idx-prefix)%qBlocks。Q地址为prefixQ*(Hq*D)+s1o*s1Base*(Hq*D)+(n2o*g+go)*D；K为prefixKV*(Hkv*D)+(s2Start+s2Loop*s2Base)*(Hkv*D)+n2o*D，V需用独立Dv与stride。不能对K/V使用Q前缀或b*固定长度。

纯varlen的s2范围[0,kvLen)；左上causal块上界min((s1o+1)*s1Base,kvLen)，右下另加kvLen-qLen，并夹到合法非负范围。BAND取max(qBlockStart-pre,0)与min(qBlockEnd+next,kvLen)，这些是块候选范围，行内仍要mask。来源空band退回全扫只可作为带完整mask的计算fallback，不能让原本无效key获得概率。

每次Matmul SetTail约束Q/KV/归约轴的实际尾部，softmax还要将无效score列置-inf或等价mask；SetTail不独自定义softmax分母。s2LoopLimit=ceil((end-start)/base)-1时空范围不进入循环，但finalize必须处理该任务空语义，不能读上个task状态。

extraInfo三槽沿全局已发布流水step递增，跨s2内层不能随意重置；每task首s2块才切其softmax状态，保留task身份。needNz2Nd、bmm1Nz与普通bmm1分发由dtype/尾部/布局真实条件决定，不能仅模板名字推断所有FP32/BF16尾都必走NZ。
