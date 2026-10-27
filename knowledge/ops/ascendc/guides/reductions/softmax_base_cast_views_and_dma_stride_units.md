---
type: Guide
title: Softmax 公共Cast视图与DMA步长单位
description: 保存同字节布局重解释的约束，防止把空操作当FP32拷贝。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_base.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/kernel_utils.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# Softmax 公共Cast视图与DMA步长单位

基类CastToFp32From/CastFromFp32To仅对half/bfloat16的if constexpr分支执行转换；T=float时函数体不复制。只有调用方已使用同一物理视图或单独复制，才能称FP32直通；传两个不同float buffer会留下目的未写。其二维stride以FP32元素表达，16位侧用2*stride保持相同字节行距，这不是紧凑half行数不变的普通转换。

CastTrait指定16→32及32→16的布局、mask merge、饱和和舍入。转换遵循这些配置而非“自动高精度无损”；16位输出会舍入，BF16与FP16范围也不同。同一存储重解释不能覆盖尚待读取的源行，独立源/目的步长与生命周期需证明。

CopyIn/CopyOut二维helper的rowSize是块数、colSize是有效元素数，两端stride是整行元素距离。GM gap=gmStride*sizeof(T)-blockLen字节，UB gap=(ubStride*sizeof(T)-align(blockLen,UBblock))/UBblock。输入需正shape、物理stride覆盖对齐行，且字段范围合法。单块重载未显式填全部参数，SDK结构体默认初始化合同需核对，不能假定局部声明天然清0。

CopyUB2UB把count上对齐到UB block，会搬逻辑尾外字节，两端均须有空间；VectorAdd/Mul/BroadcastMul虽mask有效数，载入还可能整向量。Normalize按连续row*col索引，无独立padded stride参数，不能直接应用有行padding的源。Ceil/Aligned的(a+b-1)可溢出、除数0时一些helper返回a只是兜底而非合法tiling；所有shape/平台参数在Host先拒绝非法值。
