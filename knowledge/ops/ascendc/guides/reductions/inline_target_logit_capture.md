---
type: Guide
title: 归约扫描中的目标 logit 保存
description: 在已有输入 tile 中读取目标元素，保留原始值、有效索引、跨核归属及后续损失语义。
tags:
- task.optimization
- topic.compute.reduction
- topic.compute.indexing
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/discovered/disc_X1.md
  role: primary
created_at: '2026-09-11T02:29:01Z'
updated_at: '2026-09-11T02:29:01Z'
platforms: []
---

# 归约扫描中的目标 logit 保存

交叉熵等计算同时需要整行logsumexp和一个目标logit时，可在已有tile扫描中保存目标值。tileLen>0且0<=targetIdx<N时，所属tile为targetIdx/tileLen，片内偏移为targetIdx%tileLen；只由真正加载该位置的任务读取。应在减max、Exp或原地覆写前保留原始logit，必要时在已合法拓宽到FP32的输入区读取。

读取不是免费操作。若值由MTE或Vector刚写出，GetValue前仍需相应到Scalar的完成依赖；缓存到Scalar也不能绕过数据可见性。每行缓存的身份、有效标记、初始化和使用时刻必须一致。分片由不同核负责时，需要将目标值与归约状态交给最终消费者，不能假定每个核都有这个元素。

普通单标签无其他附加项时loss=logsumexp(logits)-targetLogit；class weight、label smoothing、ignore_index、reduction方式等须保留原算子公式。非法/忽略索引不能先执行除法寻址后再过滤；空类别、空行或无有效输入按算子合同定义，不能从未赋值缓存返回。

原实现若为目标元素额外做一次搬入，本方法可将该行K+1次输入搬运变为K次；它不自动减少别处为max/sum必须完成的扫描，也不保证新增同步成本小于省下的DMA。验证目标在首尾tile、非满tile、跨核归约、特殊值和ignore_index，并比较完整时延。来源2.58倍是未在本批复现的案例。
