---
type: Guide
title: Transpose融合的部分块与变长序列偏移
description: 补齐BNSD/TND布局写回、头尾重合与驻留状态条件。
tags:
- task.optimization
- topic.data_movement.layout_transform
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/conversion/transpose_fusion_design.md
  role: primary
created_at: '2026-09-11T06:46:43Z'
updated_at: '2026-09-11T06:46:43Z'
platforms: []
---

# Transpose融合的部分块与变长序列偏移

写回时把已有连续UB块直接放到目标布局可省独立Transpose及中间GM读写，仍发生数据搬运，并非零拷贝。NBSD中固定batch、相邻head起点跨度B*S*D元素；若当前连续块长sTile*D，则GM gap=(B*S-sTile)*D*sizeof(T)。相邻batch跨度只是S*D，来源文字混用；还需UB相邻head块确实连续，否则srcStride不能0。

DataCopy普通参数的32B单位与DataCopyPad扩展GM字节/UB dataBlock单位分别按重载，来源FAQ把Ext.dstStride也除32与其其他说明冲突。普通blockLen整除32前须确实对齐，不能整数除法丢尾；Variant B中间dstStride漏sizeof(T)，若直接用于普通字节换算会错位。任何结构体未填字段不当默认为0，窄化前检查实际接口上限。

头/中/尾拆分须精确交集：若gCount1且同一块既有头又有尾，source减去两个bool会负数/无符号下溢，分别处理还会重叠。主块数量可0，起点、UB消费长度及末有效S同步推进；不能只沿用同offset写三次。来源s1EndIdx是含尾还是开区间也要明确。

TND→NTD若actualSeqLengths是累计终点，T=最后前缀，当前tBase=前一前缀或0，局部s映为tBase+s，目标head跨度T*D。若接口给逐序列长度需先求前缀；T0时(T-1)无符号下溢，batch0也不能读batch-1。gCountOneS1块的源layout须确实每head连续D，变长元数据可合法提前传入/缓存，并非必须每tile GM读取。

TransDataTo5HD的16×8/16×16是特定类型布局，source模板T却硬调<float>，只有匹配32位表示及地址列表时可行；repeat和stride字段未完整初始化、整除row/col省尾，不是任意T通用转置。按位转换应覆盖全部元素与原始bitpattern。

Sinkhorn等迭代状态驻留可避免每轮GM往返，但colNorm/rowNorm必须都能在该固定布局正确取数，不能仅删除TransposeX/Back就维持语义。若同时保存两布局，每次更新需保持一致，或显式重排；预算tAlign*n*n*4是该float状态实例。迭代≥3、占UB≤50%和4–8倍等为经验，不是必要条件。Transpose→Gather→ReTranspose还需正确indices映射和实际producer/consumer同步，MTE3_MTE2不自动覆盖纯UB Vector依赖。
