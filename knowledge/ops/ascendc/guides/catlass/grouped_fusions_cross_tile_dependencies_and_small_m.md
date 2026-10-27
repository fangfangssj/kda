---
type: Guide
title: CATLASS 分组融合的跨块依赖与小 M 调度
description: 保留配对列、整行归约和跨核合并的完整数据所有权。
tags:
- task.design
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/grouped-matmul.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/mmad-epilogue-selection.md
  role: supporting
created_at: '2026-09-11T04:33:09Z'
updated_at: '2026-09-11T04:33:09Z'
platforms: []
---

# CATLASS 分组融合的跨块依赖与小 M 调度

Grouped Matmul把任务分解为组及组内tile；slice-M时连续A起址为Σ前组Mj*K个元素，B/C按各自组shape/布局计算，不能“同理”复用A公式。GroupList在Device动态生成时Host未必可得，需要Device寻址或明确数据传输。来源02/05/08分别slice-M/slice-K/通用，07为MoE per-token dequant，10为多阶段workspace候选；不同groupSize与空组需完整覆盖。

逐元素dequant/激活通常只依赖本C块；SwiGLU/GeGLU/ReGLU需要相距H=N/2的两路，per-row量化/softmax/norm/top-k则需要跨整行信息。单N块通过不能证明跨N块正确。SwiGLU可以按输出[M,H]调度，C左列c与右列c+H各做一次对应Matmul并在同一stage保留两份，两份均完成后发布ready；或者物化足够范围的C后读取配对列。N≤L1.N还不是全部充分条件，尾块、实际切块和输入布局仍需一致。

行归约可让同一row-block的全部N块归同一执行者：第一趟计算并保存S到该row-block workspace，同时累计rowMaxAbs/其他统计；第二趟用最终整行统计处理所有N块。量化scale=rowMaxAbs/127只是对称INT8合同例，零行、round/saturate和输出scale格式需另定义。每核rowBlockM*N*4只是FP32 S区，数MB是否可接受应按真实内存预算，不能以HBM占比小自动接受。

row-block总任务为Σceil(Mi/rowBlockM)，少于核数时可能有空核；M=1024、rowBlockM128的8块在20核上只有8个任务。减小M块可能增加B重读但并非必然更慢，来源+29%只约束该案例。切N可增加任务，保持M块128是候选，不是硬件不变量。

N切分后整行统计属于多个核，需要私有partial槽与全参与者同步后合并，合并统计发布后才做第二趟；AIV域barrier需核对实际参与集合、驻留和资源，不可把伪代码当通用跨核barrier。partial按(rowBlock,nBlock,AIV子块)真实归属分配，不能只用nBlocks*subblockNum作为全局容量。maxAbs空贡献可初始化0，其他归约需正确中性值；Host memset是可选方式，Device完整写入同样可建立初始化。

小M N-split可能需保存所有row-block的S及partial区，轮转per-core空间不能自动保持跨核后续消费者数据。调度模式与dtype组合key可用schedMode*DtypeCount+dtypeKey，但来源以K作组合基数应与矩阵K区分并保证无碰撞。独立Kernel文件不是正确性必需，真正要求dispatch覆盖且大shape未误选不支持变体。

A2/A3 GM交接与950某些L0C→UB能力不同；source“只能example10”过窄，其他满足同一硬件通路与依赖的实现也可成立。hand-launch裸workspace不能无证据经GetUserWorkspace重新解释；框架预留区则按其真实ABI处理。平台能力应从目标声明/实现确定，不能按example有无950前缀猜测。

性能比较需相同量化粒度、激活、dtype、layout及计时范围。ND输入先转NZ的成本是否计入要明确，不能跑不同格式后用估算折算成同条件结果；baseline不支持某shape与自实现失败是不同状态。
