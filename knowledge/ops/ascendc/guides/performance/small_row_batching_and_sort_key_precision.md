---
type: Guide
title: 小行批处理与排序键精度
description: 保存SingleN和多行策略，补齐索引转Float的失真边界。
tags:
- task.optimization
- topic.memory
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_03_small_d_optimization.md
  role: primary
created_at: '2026-09-11T05:55:38Z'
updated_at: '2026-09-11T05:55:38Z'
platforms: []
---

# 小行批处理与排序键精度

小D可在一次搬运中合并多行，摊薄DMA启动、地址与标量控制；减少的是搬运次数，不一定减少有效字节。SingleN是每核一行的来源策略，不等于整个算子只有一行或所有小batch都应单核。

AddRMS SingleN给MAXBUF195584B=191KiB，half输入两份位于T视图0/n，FP32区域从float视图n/2n/3n起，换成byte后分别4n/8n/12n。若每区长n，则末端16n，尚须其他scratch/对齐；T不同、实际len不同要重新核对，不能按同一个offset数认定不重叠。此固定容量未验证通用平台，不可称单核性能极限。

ClippedSwiGLU小2H以batchSpace=SWI_FACTOR*DTYPE_FACTOR*AlignBytes(H*sizeof(bfloat16_t))计算可放batch数，大2H以ubMaxPair切H。DTYPE_FACTOR必须与真实输入/计算dtype对应，ubMaxBatch或ubMaxPair0要失败/回退，H0与batch0不能进入ceil除零或尾部减法。pair两半输入对应关系、尾pair和各batch起点一致。

Modulate候选isDSmall=未切D且DId0且opCopyLength*sizeof(T)<MIN_DLENGTH；handleL=min(剩余行数,ubLength/alignedD)，每行按alignedD物理stride计算，最终GM按有效长度写。handleL必须≥1，curL总推进handleL且offset随curL更新，scale/shift只可在其broadcast范围内复用。D<2048并不是该byte条件对所有dtype的等价阈值。

RMSGrad以colValAlign≤640的来源SmallD路径使用Broadcast构造2D、ReduceSumMultiN多行归约；需保持每行统计互相独立以及dgamma跨行归并。原始D与alignedD分界可能不同，padding须中性，不能把多行合并变成一行总reduce。

Embedding小维≤512只是候选预算：512FP32一行2KiB，完整成本还含索引/位置、排序结果/scratch及多个梯度行。Sort键需保持整数身份，来源先Cast索引FP32会把16777217舍入成16777216，若支持更大索引，可能把不同目标错误合并；须限制可精确键域或使用保持原整数身份的排序方案，Cast回整数不能恢复。

CreateVecIndex生成n个位置应为0..n-1，来源文字0..n多一项。Sort/Extract载荷必须跟随原始梯度位置，idxLocal/idxULocal若只是视图须实际同buffer且初始化，padding的-1只在所有合法键不与其冲突且sort方向/有效长度正确时可用。分块排序不建立全局相同键连续，跨块/核组要保留所有贡献；梯度地址仍可能间接随机，不能称所有访存都变顺序。稳定次序、最后组提交、empty和重复键均须验证，2–5x收益未由本批测量建立。
