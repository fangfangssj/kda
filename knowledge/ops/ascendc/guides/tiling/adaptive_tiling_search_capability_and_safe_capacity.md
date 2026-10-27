---
type: Guide
title: 自适应分块搜索、可行性与安全容量
description: 保留多策略选择及二维搜索，明确示例谓词、启发式和失败回退。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_02_adaptive_tiling.md
  role: primary
created_at: '2026-09-11T06:10:16Z'
updated_at: '2026-09-11T06:10:16Z'
platforms: []
---

# 自适应分块搜索、可行性与安全容量

AdaptiveAvgPool3D示例doubleC>tileLen优先SplitC，cTile=min(alignC,tileLen)；否则inputTileNum<maxWindowWLength选SplitW，其余MultiW。需保证通道对齐、每窗口真实边界及全部活跃buffer可放。AdaptiveMaxPoolGrad Normal以CalUBTotalSize(1,1,1)判断可行，Scatter兜底IsCapable=true仍不代表索引/输出初始化/资源总是合法；0/100、2、102是局部key映射，重叠窗口的多次贡献仍需累加及精度协议。

AddRMSDynamicQuant依次CheckUbNormal、SingleRow、SliceD，Normal rowStep上限16，slice列长8864是该版本策略参数。片段CheckUbNormalTiling声明bool却未返回，不能直接编译为完整实现；rowStep0需失败转下一策略，不能选0行循环。多个“可行”分支的优先级只表达启发式，不证明性能全局最优。

DynamicBlockQuant用used=min(totalCores,rowBlocks*colBlocks)和FindUniqueCut枚举二维候选，n=used/m，delta=rowNormalBlock*colNormalBlock更接近最大矩形负担，不能单由片段声称已精确计算尾核额外量。colNormalBlock、尾核划分与候选是否覆盖全部整数方案均未展示；需检查m,n正、任务覆盖、空核、尾部及实际搬运/计算权重。

Embedding的uint64 ubSizeLeft-=reserve+indexBytes后再比较>0无法拦截下溢，应在减前判断。3份索引和FP32三份/FP16四份grad、20KiB预留是来源缓冲模型，maxFormerNum最终乘gradAlignNum得到对齐元素容量，不是最大行数。DeformableGrad公式中的3*levelsAlign+8*channels固定项与15+13*channels每query项以dtype元素表示；先证实各buffer实际dtype相同，否则应分别换算字节。向下data_align取整后0不能参与ceil/除法。

LayerNormV3 Welford的tileLength/updateTimes/tempSize与TwoPass的aFactor/aBlockFactor/对齐字段不通用，CanFitInBuffer必须计入真实API scratch与固定区。V4 SingleRead示例rowMax=(ub-8*BLOCK_SIZE)/4/2，只是两份float模型；Transpose条件rowSize≤64与另一段“适合大rowSize”冲突，以实际能力谓词为准。IsCapable省略成功return的代码不可直接采用。

CompileInfo可在TilingParse阶段缓存AIV核数、UB容量和Soc模式，再在Host Tiling读取；检查context/platform/compileInfo有效、数据属于同目标且构建缓存失效正确。ASCEND310P、ASCEND910_95是该SDK枚举，不能泛化所有同名营销平台或dtype能力。Softmax示例按x/y、FP32计算区、bool mask行跨度以及32/64KiB scratch算行数，scratch常量需服从实际API/版本，防混合signed/unsigned减法回绕。

MaxPool按N→H→W→C尝试切分，再必要时切kernel。二分需要可行谓词在搜索区间上具有所假设的单调性，UB与核利用率联合约束不当然单调；没有可行mid时不能把默认best=1当成功。内核维切分要合并max/argmax与tie规则，所有搜索结果仍需最终资源、覆盖和语义检查。
