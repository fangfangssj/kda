---
type: Guide
title: Gather 转置的相位布局与缓冲区计量
description: 保留预生成索引、Compact输入与两端DMA循环的完整契约。
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

# Gather 转置的相位布局与缓冲区计量

参考3510实现key10006受TRANSPOSE_ENABLE_GATHER及相应Reg/MicroAPI依赖保护，shape消除1轴/连续轴后末轴参与交换时尝试。资料给出VCONV→Gather→NDDMA顺序及32KiB最小tile、至少半数核、借轴至多3、CheckBC回退；这是该Host路由及可行性策略，不是所有Gather硬件或版本的普遍能力限制。不能从此分支只支持3510推断其他架构没有任何Gather。

GM输入与输出分别尽量选较大连续burst，其外侧仍可带stride，并非整个tile两端一定单次连续。in/out队列各depth1、num2、单份dataTensorSize；index单TBuf容量indexTensorSize，总数据预算4*dataTensorSize+indexTensorSize，还需平台保留空间和对齐。队列允许流水，是否实际充分重叠仍受循环依赖和瓶颈影响。

输入使用DataCopyPad PaddingMode::Compact：descriptor的blockCount块紧排，而LoopMode组边界才按32B对齐，CalcUbAxesInOffset专门表达这一布局，不能换成每行都自动32B补齐的普通公式。输出每连续outCube按32B行距，LoopMode更外层乘已对齐行距。GM gap为真实轴stride字节减有效cube字节；需非负及字段范围，两方向LoopMode均设置后恢复。

索引为最内1/2/3维输出坐标乘输入UB stride之和，二维q/r、三维商余分解，后续向量通过VL增量及进位更新坐标，外侧最多3轴只加标量基址以复用索引。索引必须计入Compact特殊padding。dimNum的else直接走3维，Host必须保证1..3，不能让0或4..6静默丢轴。

四相位固定编号0主、1输入尾、2输出尾、3双尾，gIdxOffset按主相位元素数对32B索引块对齐；只有输出尾也写到2*gIdxOffset，不能按实际生成两份就只分配2份空间。两个切点哨兵-1只能表示该方向不存在非整除尾，若整除仍标有效会算出0维尾并除0。各切轴size/factor正，生成完成恢复两侧主shape，每块按真实混合进制尾坐标选择相位。

block任务按标准每核blkFactor、最后核实际blkTailFactor，而非q/r的前r核加1。blkAxes两端绝对stride把同一坐标映到GM，源码CalcBlkAxesOffset虽返回int64内部乘积是int32，仍需约束后缀积或改宽中间量，返回类型不修复先前溢出。
