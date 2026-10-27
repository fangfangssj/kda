---
type: Guide
title: Ascend C API 家族与存储位置决定的搬运
description: 补充向量标量分流、scope 搬运和重载身份，限制机械映射。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md
  role: primary
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# Ascend C API 家族与存储位置决定的搬运

高层copy不对应单一指令：GM→L1可能DataCopy+Nd2NzParams，L1→L0A/B用LoadData或LoadDataWithTranspose，L0C→GM用Fixpipe，GM↔UB用适合对齐/尾部/stride的DataCopy或DataCopyPad，UB同dtype复制与数值Cast属于不同操作。布局、物理stride、有效长度和目的存储位置共同确定重载，二维tile也不自动必须使用某一API。

来源DataCopyExtParams含blockCount/blockLen/srcStride/dstStride/rsv，blockLen为字节，stride单位按方向/GM或Local位置核对；isPad/leftPadding/rightPadding/padValue归DataCopyPadExtParams。其GM→UB四参数带pad、UB→GM三参数形式是所述重载，不是全API只有这两签名。TQue可管理搬运buffer生命周期，但合法TBuf LocalTensor配显式同步也能搬运，不能把“不要TBuf”当硬件限制。

向量对向量通常Add/Sub/Mul/Div/Max/Min，向量对标量对应Adds/Muls/Maxs/Mins等；减标量可加负数，除标量可乘倒数，但整型极小值取负/溢出、FP NaN/±0及舍入需保留。BufferLoad不自动是可直接给Scalar指令的Host标量，取值位置和同步需明确。

其他族包括Exp/Sigmoid/Ln/Abs/Power/Reciprocal/Sqrt/Rsqrt/Relu/LeakyRelu/Axpy/Sin/Cos，And/Or/Not/Xor/ShiftLeft/ShiftRight，Compare/CompareScalar与Select，Cast/Round，Transpose/Broadcast，CreateVecIndex/Duplicate/ArithProgression，以及ClampMin/Max、BilinearInterpolation、Brcb。名字对应只作检索索引，scratch、dtype、mask、broadcast轴、别名和参数列表需实际声明；逐段Clamp顺序还受上下界与NaN规则约束。

Sort/Sort32、MrgSort和Gather/Gatherb/GatherMask有不同数据结构。初始化Sort buffer可能由多次Duplicate完成，merge后是否copy由目标驻留位置决定；topK不是只调用MrgSort就一定完整，还需已排序候选、有效长度和索引。来源自定义gather pattern先乘元素字节数再Gather是索引表形式，不能与专用mask编码混用。

归约AR/RA按逻辑维度及物理布局选择，ReduceSum/Max/Min、BlockReduce*、WholeReduce*不是可无条件互换。底层矩阵乘Mmad还要求正确L0A/B/C及累加状态。来源高层DSL语法只用于定位对应Ascend C能力，编译器翻译步骤与Triton内容不迁入。
