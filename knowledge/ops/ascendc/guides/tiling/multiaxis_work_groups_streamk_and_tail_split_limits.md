---
type: Guide
title: 多轴任务组、Stream-K 与尾块再分裂
description: 保存层次化并行，并补齐分组索引与跨核合并边界。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_04_load_balance.md
  role: primary
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-09-11T06:15:28Z'
platforms: []
---

# 多轴任务组、Stream-K 与尾块再分裂

AdaptivePoolGrad可按NC以及D/H/W块构造总任务，nc/do/ho/wo的ceil分块乘积要宽域且尺寸正；preCoreNum0在示例表示整除同round路径，其他核用ncRoundTail，所有Host字段需统一。实际输出梯度重叠仍要按目标owner或安全合并，不因多维切分自动无冲突。

GatherElements按idxPre较多时分行组，否则每pre组再分post，former/tail group core数与组内尾部应同时校验。MergeAxis合并的是一段轴上x与index的对应维相等，轴与轴之间不必彼此等长；还需连续stride且不跨Gather轴，乘积不能溢出。来源j-i==1分支没有AppendDim，会丢掉一个孤立匹配轴，完整实现需保留该轴并同步index shape，而非照抄片段。

ScatterElements当times<C时每任务分eachPiece个核，否则每核处理多个任务。源码eachPiece=C/times、inputOnePiece=ceil(inputOneTime/eachPiece)，used=ceil(inputOneTime/inputOnePiece)*times，但Kernel仍按旧eachPiece求coreId/eachPiece：若实际所需piece减少会漏任务。例如times2,C8,inputOneTime2，初始eachPiece4而used4，4核均被映射到task0，task1缺失。应统一实际piece数和Host/Kernel映射，或保留原槽数并正确空任务；times0、input0先处理。

Stream-K把部分MN tile沿K拆开，DP保留完整K，SK输出需要安全合并。来源tail=(mTiles*nTiles)%used、total=DP+tail*skKTileNum只是本例；UpdateMNTileIdx中的curKTileNum、余used映射与K起点/长度尚不完整，不证明任意tilegrid都可直接均摊全部核。K切片必须无漏无重复，bias/激活只在正确阶段执行一次，Atomic需初始化和最终完成，确定性取决于合并顺序合同。

GroupedMatmul尾轮以mTailCnt*nTailCnt进一步切MN，mSplitIdx=(blockIdx%tailCnt)%mTailCnt、nSplitIdx=(blockIdx%tailCnt)/mTailCnt。源仅展示mSplitIdx越界检查，N越界也需处理；实际tailL1M/N取剩余与块长min，偏移按物理布局和Expert基址，NZ对齐不意味着可写入其他子块。空子块返回{0,0,0,0}须被调用方识别，不得参与无效DMA/Matmul。

上述方案各有额外索引、workspace、同步或格式成本，分组变长时均衡任务数不必然均衡FLOPs。用小整数枚举覆盖网格，再测真实短尾/多Expert与K拆分数，不能单凭示例实现宣称峰值性能。
