---
type: Guide
title: AR FullLoad归约临时区随批量行数增长
description: 保留一次GM载入的计算骨架，定位固定512字节临时区不足。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ar_full_load.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_base.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# AR FullLoad归约临时区随批量行数增长

AR FullLoad按aBlockFactor分行、ubFactor再分批；每批GM一次载入，Vector先逐行求max并写FP32 exp，随后求sum并输出。输入输出各双buffer，exp区域ubFactor*rAligned*4，紧随其后仅固定512B binaryTmp。Kernel用GetBlockNum判断末核，因此launch必须等于Host返回usedCoreNums，额外空核不能自动安全忽略。

R>128时SecondNormCompute把每行向量组折到P=FindNearestPower2(ceil(R/64))个标量，每行临时stride=align(P,8)，需要ubA*stride*4字节。CPU Host实例UB196608、32核、A4096、R129产生71行，P2、stride8，临时实际需2272B，超过声明512B。不能仅验证exp区和总UB公式；需动态分配临时区、减小ubFactor或逐行归约/输出。上述是源码算术反例，未在NPU执行。

最终Post仅支持P≤128，且原始R/rAligned又转uint16；任意扩大UB或R范围需同步扩展末端归约、临时空间、计数宽度及DataCopyParams重载限额。512B恰好够128个float不意味着够所有行。

rAligned仅32B/dtype对齐，DataCopyPad真实GM blockLen为R*sizeof(T)，UB行距含自动对齐加显式gap。FirstNorm先加载尾并将无效max lane保持-inf，再处理完整VL；不能把padding0参与max。FP16/BF16载入Cast到float、输出按CastTrait舍入。减max对有限非空输入稳定，NaN/Inf全掩码仍按接口另定义。
