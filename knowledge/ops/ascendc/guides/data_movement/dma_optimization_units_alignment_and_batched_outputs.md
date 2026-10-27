---
type: Guide
title: DMA 优化单位、地址对齐与批量输出
description: 保留搬运融合思路，纠正padding字段单位和向上移动数据起点。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/common/datacopy_optimization_design.md
  role: primary
created_at: '2026-09-11T06:42:38Z'
updated_at: '2026-09-30T01:53:56Z'
platforms: []
---

# DMA 优化单位、地址对齐与批量输出

减少短事务可从合并相邻输出、按行批量提取、GM→L1同时ND→NZ和片上Gather偏移表复用入手，但需保持真实布局/依赖。ND→NZ并不总需独立TransData，也不固定节省一次L1往返；取决于旧路径。Nd2NzParams示意缺完整字段，N方向padding与数据C0打包粒度不同，不能把dstNzC0Stride一律按FP16 16、FP8 32套用。

扩展DataCopyPad中GM侧间隙和UB侧间隙单位不同，UB stride通常按32B dataBlock；left/rightPadding按对应接口的元素单位与实际上限，uint8不推出可补255字节。blockLen字节、Tensor下标元素、padding/dummy及行pitch分别推导。源“列提取blockLen/srcStride必须32B对齐”与扩展非对齐能力不能混用，按实际方向/模式确认。

AlignUp(rawOffset,512)会跳过合法数据，且GlobalTensor<T>[offset]会再乘dtype字节。只能在内存规划中保证相同逻辑数据落到对齐物理起点，或安全地搬较大窗口再取正确子区；不能在Kernel随意移起点。32B是某些接口基本粒度，512B最优、70/90/100%带宽、>4KiB充分和Gather5–20倍均是未附本次trace的来源经验，需目标平台同工作量测量，70%→100%相对提升是约42.9%而非30%。

批量输出需每轮写累计UB的不同offset，所有结果存活到最终MTE3完成；仅把DataCopy移出循环但Compute总写同一位置会丢结果。若loopCount>0，总有效量=(loopCount-1)*full+tail；loopCount0须独立，tail可等full，乘积与字段窄化检查。改用Ext参数不能自动解除硬件描述符上限，仍需分批。

Gather偏移表是字节地址、范围/位宽/源UB区域合法，若逻辑索引变化不能永久复用。source pre/post表内j循环不改变curOffset，实际每行重复同一地址；只有目标本就需要复制该值时合理，不是一般连续子矩阵提取。Scatter逐行DataCopy是DMA行写，并不等于向量Scatter；重复索引需要确定覆盖/累加语义，单纯批量化不提供原子性。

非对齐padding进入ReduceSum可用0，Max需负无穷或有效mask，其他运算取其中性值；全零默认不能推广。验证位级往返、行列编码、不同dtype/stride/最后批和地址哨兵，再比较有效流量、事务与完整时延。

## 配套 API 与常见组合

- [DataCopy](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_gm_and_ub_continuous.md)
- [DataCopy](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_gm_and_ub_highdim_split.md)
- [DataCopyPad](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_gm_to_ub.md)
- [对齐单位、布局行距与填充值](../memory/alignment_units_layout_stride_and_padding_values.md)
