---
type: Guide
title: Transpose 策略ABI与同名字段的不同单位
description: 保留四类POD及路由关系，防止跨策略机械复用字段。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_tiling_data.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/transpose_tiling_data.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/templates/dav3510/transpose_base.template
  role: supporting
created_at: '2026-09-11T06:46:43Z'
updated_at: '2026-09-11T06:46:43Z'
platforms: []
---

# Transpose 策略ABI与同名字段的不同单位

该arch35参考按TENSOR_MOVE、SMALL_SHAPE、CUT_ONCE、CUT_TWICE、N_LAST、BIG_DIM组织通用TransposeOpTilingData，Gather、VCONV5HD和021 VCONV各有独立结构。策略键10000/10001/10002/10003/10005等是此实现分发，不是AscendC通用编号；GATHER宏TRANSPOSE_ENABLE_GATHER与__NPU_ARCH__3510限制需Host和Kernel同时匹配。路由表只给启发条件，未附全部Host实现，不能据表宣称可直接完成tiling。

POD原样下发需要同字段顺序、宽度、数组长度、padding和编译器布局；#pragma pack(push,8)是最大成员对齐控制，不能仅凭字段名相同推断与框架生成结构ABI一致。TransposeOp有19个int64标量、4组8元素int64数组、13组5元素int64数组，按给定声明总928B；不能套说明中972B旧blob大小。wrapper只有一成员仍需sizeof/offsetof对照。依赖头的相对路径从.template改.h后也须与真实目录一致，源码路径描述不是自动生成能力。

通用字段包含permSize/inCutIndex/outCutIndex、两侧UB factor/tail、realCoreNum、ubSize、四类区间和输入/输出/扩展shape。input/perm数组上限8，NDDMA描述数组5；perm必须为合法排列、维度正且乘积可表示，退化/空张量单独处理。#pragma pack与标量bool字段也需实际两端布局验证。

ParseMultiCoreRange以q=blkFactor、r=blkTailFactor表达T=C*q+r，前r核q+1任务，start=bid*q+min(bid,r)，end=start+q+[bid<r]；TENSOR_MOVE使用元素任务，CUT/BIG使用DMA任务。SMALL_SHAPE同字段却是标准每核元素数和最后一核有效数，不可调用同一q/r解析。Gather字段的语义也应阅读对应Kernel而非套通用表。

totalNddmaNum在BIG_DIM实际参与blockLen=该值*sizeof(T)，表示单次DMA主块元素数；文档总表说总块数不适用这一消费者。ubSize在TENSOR_MOVE表示该队列总预算再除2，单缓冲CUT/BIG直接全量分配；不应Host预除2后Kernel重复减半。in/outTailFactor0表示无额外尾块，其他族末块长度惯例可能整除时等full。

Gather另有dataTensorSize/indexTensorSize字节、usedCoreCnt、int8切轴位置(-1哨兵)、8维block轴及6维UB轴、最多3层GM步长；VCONV含R/C主尾核与UB分块，021含N/H分核及UseRConv/UseHSplit。记录逻辑长度、对齐长度与块数的区别，实际资源/能力未验证前保留draft。
