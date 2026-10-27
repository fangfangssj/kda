---
type: Guide
title: Gather 转置索引的完整向量空间与窄整数
description: 补充首向量无掩码写、相位间距及位宽转换的约束。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_gather.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_with_gather.template
  role: supporting
created_at: '2026-09-11T06:55:07Z'
updated_at: '2026-09-11T06:55:07Z'
platforms: []
---

# Gather 转置索引的完整向量空间与窄整数

RangeType在1/2B数据下为int16，其他为int32；IdxType是对应unsigned，1B CastType扩为16bit类型，再以DIST_PACK_B16保存原字节。这里是按位布局处理，不是把8位数值通用转成16位计算。真实DataCopyGatherImpl重载的地址单位、Reg重解释和mask lane含义须与SDK确认，不能按C++类型名推断所有1/8B数据均正确。

GenIndex4TwoDim/ThreeDim首轮使用ALL mask，无论逻辑totalDimSize是否小于idxVLSize都写一个完整向量。gIdxOffset仅对32B对齐，不自动覆盖256B完整VReg。例int16索引17项逻辑34B、相位步长64B，但首轮写256B，可越过相位或索引分配边界；即使Host通常大tile，尾相位也可能很小。解决需实际限定最小相位物理跨度、改首轮有效mask或显式预留并保护完整向量范围。GatherData读取idxOriReg也无局部mask，尾向量需合法可读空间，后续Gather的mask不替前一load授权越界。

源索引乘法、Div中间维度、外层标量offset经RangeType窄化；unsigned最终重解释可能有意使用位模式，但不能让有符号Div维度、进位比较和索引累加未经证明地溢出。对每相位验证最大实际输入索引与累加前中间范围，int16维度超过32767尤其不能仅看uint16最终上限65535。uint16循环次数和axis大小、int32产品也分别核验。

8B数据的vlSize与idxVLSize不同，index按32位生成而消费按数据VL分段，索引load可包含多余lane，必须保证有效lane对应及物理可读尾。1B路径maskValue乘2、Gather寄存器重解释、打包存储也需要按官方接口证明；本参考未带相应运行证据。以原始字节编号覆盖1/2/4/8B、三种索引维数、全部尾相位及最大索引，检查容量与输出bitpattern。
