---
type: Guide
title: 注册 Add 的分区、空输入与队列缓冲语义
description: 抽取双输入分块算法并修正空路径除零、粒度单位与模板键假设。
tags:
- task.implementation
- topic.implementation.tiling
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_host/arch22/add_example_tiling.cpp
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_host/arch35/add_example_tiling.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/add_example_arch22.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/add_example_arch35.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/arch22/add_example.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/arch22/add_example_tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/arch22/add_example_tiling_key.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/arch35/add_example.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/arch35/add_example_tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/add_example/op_kernel/arch35/add_example_tiling_key.h
  role: supporting
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# 注册 Add 的分区、空输入与队列缓冲语义

来源TilingData三字段totalNum/blockFactor/ubFactor均为int64元素数，Host与Kernel必须共享布局、大小和含义。非空时按平台AIV核数C计算blockFactor=align_up(ceil(N/C),granularity)，启动ceil(N/blockFactor)个核；第i核长度min(blockFactor,N-i*blockFactor)，再按ubFactor循环并单独处理尾部。C、granularity、ubFactor须为正，乘积、ceildiv及窄化应有界。

来源GetUbBlockSize用于元素数对齐，但若返回单位为字节，需除sizeof(T)才得到所需元素粒度；直接按32元素对齐4B类型属于更粗分块，不应解释为唯一32B要求。DataCopyPad的blockLen=currentNum*sizeof(T)是本例字节长度，需核对字段范围与实际重载，不能泛称它必向上多写GM导致相邻核覆盖。物理UBpadding、GM合法span和有效计算长度分别证明。

N>1024时来源选BUFFER_MODE=1，输入X/Y和输出Z各分配两份，共6*ubFactor*4字节；否则三份。阈值1024、TYPE_SIZE=4仅适用当前FLOAT/INT32，扩dtype需更新预算和编译实例。全部UB用于三/六份分配还要计真实框架/scratch/对齐开销。ubSize非零不能保证向下对齐后ubFactor非零。

源码TQue的模板深度为1，同时InitBuffer可分配BUFFER_NUM=2；队列待处理深度与底层buffer份数是不同概念，不能机械要求模板数字等于分配份数。当前每轮各EnQue/DeQue一份、buffer可交替复用，是否真正并行仍看生命周期与目标API合同。input完成计算后回收、output搬出后按queue协议回收，不能仅计函数数量。

来源空Tensor分支把TilingData清零后设置1核返回，但Kernel仍Init并执行(0+0-1)/0计算loopCount，存在除零；还会用0字节InitBuffer。应在合法空路径避免资源初始化/循环，或建立有效非零tile并确保无需地址解引用。L2空输入绕过不能证明图模式或直接Tiling测试也安全。

D_T_X与BUFFER_MODE在ASCENDC_TPL_ARGS_DECL/SEL中声明并由SEL_PARAM按相同顺序选择，入口模板参数和对应dtype支持组合必须一致。BUFFER_MODE的8是声明位宽，不能当八份缓冲。arch35注释把BEGIN_TILING_DATA_DEF与TILING_KEY_IS称为普遍废弃，是模板规范；不能取代具体SDK能力。struct类型可见性（int64_t头文件）不应偶然依赖include顺序。

source workspace slot0设0和空CompileInfo用于此框架注册形态；声明0大小不保证返回非null可解引用指针。GET_TILING_DATA_WITH_STRUCT/REGISTER_TILING_DEFAULT解析结构必须匹配Host，workspace与tiling参数也不能交换。不同架构目录内容相似不等于资源和指令支持已在所有目标验证。
