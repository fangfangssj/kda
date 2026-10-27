---
type: Guide
title: 独立 Golden 与整数 Dot 的精确性条件
description: 保留阶段舍入与量化验证知识，修正最终幅值即可保证FP32精确的说法。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/precision-verification.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/verify_result_template.py
  role: supporting
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# 独立 Golden 与整数 Dot 的精确性条件

Golden应实现算子合同规定的数学和中间舍入，不能镜像候选实现的索引或数值bug。CPU NumPy/BLAS的FP32 reduction tree可能不同于Cube；同一Mul/Exp序列也不保证软件库与硬件近似bitwise一致。GELU tanh近似与来源常数-1.595769121、0.044715等具体变形需验证符号、系数、溢出与合同，不能只比较算法名字。

int8乘积可在FP32精确表示，但“最终整数sum绝对值<2^24就保证FP32 matmul精确”错误：中间大和经抵销也会丢单位。一个顺序反例是4096个4096累加到2^24，再加1，再累加4096个-4096，整数真值1，FP32此顺序得0。所有乘积绝对值之和≤2^24是对使用完整FP32乘加且无其他低精度约化路径的一种充分界限；只看最终sum不够。

若|a|、|b|≤8且K6144，则sum|a*b|≤393216，满足上述幅值充分条件，仍需确认实际BLAS/backend未使用不同乘法精度。更宽输入/长K宜用经防溢出的int64独立参考，INT32自身也有溢出上限。小shape交叉检查不能证明所有大shape的约化顺序和精确性。

将随机输入限制O(1)可用于正常域对照，但合法极端值、取消抵销、scale边界、0/Inf/NaN不能因此被删除。FP16最大有限值65504附近的overflow是否允许由接口定义。BF16/FP16中间Cast的时点会改变结果，全FP32与每阶段降精度属于不同语义候选。

FA来源atol0.02/rtol0.1、99.9%匹配且maxAbs0.05以及与aclnn差1e-4是经验配置/历史声称，未在本批实测，不能视为官方通用容差。GDN逐阶段比较要以独立正确的上游输入对齐，也要做完整链验证；基线不支持某输入不等于自定义实现失败，反之自定义更新后旧输出也不能作为新版本通过证据。
