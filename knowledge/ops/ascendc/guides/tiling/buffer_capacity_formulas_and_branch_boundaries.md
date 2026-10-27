---
type: Guide
title: 缓冲容量公式与分支边界
description: 保存量化及归一化的细粒度预算，并修正单位和下溢陷阱。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_01_double_buffer.md
  role: primary
created_at: '2026-09-11T06:10:16Z'
updated_at: '2026-09-11T06:10:16Z'
platforms: []
---

# 缓冲容量公式与分支边界

GroupedDynamicMXQuant把in、scale、out各分2份且按相同bufferSize预算，maxUbAvailable=ub/2/3只对应这六份等大区域；压缩后数据与scale并非自然等字节，额外scratch仍须计入。maxUbCol计算中的vfLen/blockSize/2需明确元素与字节单位，不能省dtypeSize后仍称精确容量。其baseline已有DB_BUFFER，不足支持新增双缓冲的收益对比。

ClippedSwiGLU以xBuffer=2*2*DTYPE_FACTOR、yBuffer=DTYPE_FACTOR及tmp1/tmp2系数计算pair容量。先扣reserve和group index，再除系数、按BLOCK_SIZE向下取整，再减BLOCK_SIZE-1、除2字节；每一步都应使用一致基础单位、检查减法非负及最终pair>0。正的可用字节在取整后仍可能小于31导致负容量；isLongH比较ubMaxPair*2与dim2H须宽域。dtype/排列/转换scratch变化需重算系数。

DynamicBlockQuant单行预算示例n*(inBytes+outBytes)+BLOCK_SIZE；FP16多行还包含另一输入副本及(BLOCK_SIZE/inBytes)*floatBytes，BF16多行包含n*floatBytes和2*BLOCK_SIZE。再以(ub-reserved)/2/perBlockBytes估双缓冲block数量；这只是所述路径的清单，perBlockUbSize初始化及其他活跃区须核对，结果0需改切分或拒绝。

Adaptive DoubleBuffer的示例条件是maxRow>rowPerHeadCore才转单缓冲，等于时仍走双缓冲；单双路径对应key0/1与2/3须与dtype/模板一致。multiRowNum若直接maxRow还应限制本核剩余行数；较大单缓冲tile与较小双缓冲tile应按完整时延选择。

AddRMS模式优先numCol>ubFactor走SplitD，再blockFactor1走SingleN且该示例排除ASCEND310P，half且对齐才Normal，其余MultiN需有效默认初始化。InplaceAddRMS另有MergeN等模式，不能把不同算子的条件合成统一表。

ScatterElements按indices/input是否各占小于半UB选择优先常驻和另一侧分块，否则均分。总字节须包含indices、updates、input及副本/对齐；先按原始元素计算inputEach再向上inputAlign可能超过预算，应回算真实分配，inputEach0不能进入(inputOnePiece-1)/inputEach。

AscendQuant片段将scale从sLocal Cast到castS，后续Mul却用sLocal：若无合法别名或匹配重载就是未消费转换结果，不能据此证明FP32 scale计算。SQRT_MODE示例重复乘scale得到scale²，其名字不代表开平方语义；与实际量化公式逐项匹配。
