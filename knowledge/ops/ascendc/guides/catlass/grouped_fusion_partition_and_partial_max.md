---
type: Guide
title: Grouped Matmul 融合的分区与 Partial Max
description: 补齐不均匀分组、小M二维调度和跨N统计索引。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/grouped-matmul.md
  role: primary
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# Grouped Matmul 融合的分区与 Partial Max

按M切组且K固定时A组基址为sum(M_j)*K；B为每组独立矩阵时其基址为sum(K_j*N_j)，C为sum(M_j*N_j)，不能把A/B偏移写成同一公式。空组、极度偏斜组和变化K/N必须遵循真实接口，groupList会改变任务映射和数据依赖。

A2来源选定路径将Cube结果经GM交给Vector，不意味着全部架构无L0C→UB路线。整行per-token amax需覆盖完整逻辑输出列；若SwiGLU把N列配为H=N/2列，S及后续quant的统计宽度是H。轮转S空间要含真实core/stage份数、rowBlockM、逻辑宽度和dtype。

M1024、rowBlockM128只有8行块，20核候选中会有12核没有该类任务。缩到32或按(rowBlock,nBlock)二维调度可增加并行，但额外A/B搬运、Scalar开销和归约同步可能抵销收益。来源小块快29%、某比例0.76→0.38为未附本批原始日志的历史测量，不是固定结论；totalRowBlocks<coreNum且nBlocks>1是候选条件，不是性能收益的充分必要条件。

跨N方案可写全局S及partialMax，然后合并行统计，再量化对应输出。partialMax索引应唯一标识rowBlock、nBlock、AIV子块和row lane，不能只用coreId或nBlocks*subblock而漏行；同一核可能承担多任务。abs的max可用0作为中性值，但所有被读slot必须已初始化/写完，空任务也需合法参与或明确排除。

跨半区SwiGLU需要C[...,c]和C[...,c+H]配对；单个普通N tile未包含配对半区时不能只在本地逐元素处理。可用配对生产者、完整GM结果再取配对、专门调度等方案，不能宣称只有一种实现；N≤tileWidth也不独自证明配对偏移、mask和尾部正确。来源N18432约44%错误、最大差171.5只记录为历史故障线索。

AIV域CrossCoreBarrier与AIC的ready/release握手是不同协议。域内所有实际参与者包括闲任务核需按约定到达，不能一部分提前return；不需要AIC参加同一AIV barrier并不等于AIC无同步需求。Host MemsetAsync须在生产消费stream顺序中，或由生产者显式初始化/validity mask；不能只清部分workspace而随后读取未写C/S。

分开小M/大M文件不能证明大M性能零回归，编译、dispatch和共享资源仍可能变化。比较基线需同公式、dtype、layout、分组分布、输入域；转换布局若是方案成本需显式计入，不能静默改变算子合同来匹配基线。子算子耗时求和只是含启动/cache/重叠假设的估计，roofline依赖平台资源；1.0–1.3倍不是全局达标标准。
