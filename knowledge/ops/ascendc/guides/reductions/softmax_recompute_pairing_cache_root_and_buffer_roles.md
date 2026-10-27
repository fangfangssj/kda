---
type: Guide
title: Softmax Recompute配对、缓存根与缓冲角色
description: 保存AR和ARA三遍组织，区分buffer数量及尾部字段。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ar_recompute.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ara_recompute.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_base.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# Softmax Recompute配对、缓存根与缓冲角色

AR Recompute对每行先扫描所有chunk求全局max，再将主chunk exp与相隔basicBlockLoop的折叠chunk exp逐lane相加后局部ReduceSum，UpdateCache形成树根，最后重读输入输出exp/sum。U尾0表示整除无额外尾；basicBlockLoop严格小于ceil(R/U)的2幂、mainFoldCount=floor(R/U)-basicBlockLoop。仅一个chunk时basic0走独立局部sum，不能解释为缺失统计。

AR输入num3、输出num2、一个float tmp及max/sum/cache；代码逐次Alloc/DeQue/Compute/Free主和折叠块，并不因num3就证明两chunk同时驻留或DMA完全重叠。sum每层stride8float，2048B cache允许的层数及Reg全向量可读尾须一起核验。AR页说明和通用指南的“recompute均三缓冲”不适用于ARA。

ARA Recompute输入num2，yQueue也num2但每份按float；main exp存yMain_，fold exp加到其中后NlastReduceSum→temp→cache，归约阶段yMain_没有EnQue就Free，这是临时Vector存储角色；输出阶段再按T2正常EnQue/DeQue。不能为统一队列流程强行增加一次未消费的EnQue。默认bin128与基类Nlast上限吻合，改大bin不是单改Host常量。

ARA尾bin字段是最后bin真实长度，整除时等binFactor，不是0；完整折叠、部分折叠与未折叠块逻辑依赖此区别。a0尾输入仅curTileA列，但归约调用按tileA，未写padding列只能保持与有效列独立；最后输出仍按有效列。

两类树根cache层号对应完整2幂main序列，不能直接把所有chunk任意顺序UpdateCache后只取最高层。所有R块求max时必须同一有效组，三遍之间输入保持一致，若生产者仍修改GM则统计与输出可能来自不同数据。GM读取3遍是逻辑算法计数，不等于NPU实际流量。
