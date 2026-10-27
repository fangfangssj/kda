---
type: Guide
title: 量化模式分发与反量化广播坐标
description: 区分类型别名、实际低位计算、行列 scale 与高阶接口的空间和平台合同。
tags:
- task.implementation
- topic.compute.quantization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P48_multi_quant_dispatch.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P49_hw_dequantization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P48_multi_quant_dispatch.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P49_hw_dequantization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P48.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P49.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/quantization/ascend_dequant.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/memory_vector_compute/data_padding/brcb.md
  role: constraint
created_at: '2026-09-11T01:56:34Z'
updated_at: '2026-09-11T01:56:34Z'
platforms: []
---

# 量化模式分发与反量化广播坐标

来源量化分发示例分别列出A8W8的int8输入/权重及int32累加，A4W4的打包int4输入/权重及int32累加，A16W8的half输入、int8权重与float计算状态。A16W8还需要实际权重反量化或相应受支持计算路径，仅改using别名不会自动实现Antiquant；A8W4/MSD等扩展还需要对应分解、组合及中间buffer。

编译期宏决定编入哪组类型与实现，TilingKey选择已构建的合法实例。来源key=2/3/4只是项目映射，并非全局标准；Host注册、编译实例与Kernel分发要一致，未知组合应有明确行为。预处理的互斥分支不会在一个二进制中自动生成所有模式。静态特化可消除相应分支，但Host选择/启动仍有成本；实例组合增加编译及二进制成本，裁剪不能删掉已承诺支持的输入。

对int32矩阵累加值C[m,n]，典型后处理包含channel_scale[n]和token_scale[m]两个轴的放缩。每个group/tile必须各自偏移，source 的scale[i*N]只有在scale确实按行重复存储时才合法；一个N元素的通道scale不能随行无限前移。O8/O16/O32输出还需各自转换、舍入和饱和合同，A4W4也不保证实际吞吐翻倍。

AscendDequant是高阶接口，不等于单条固定硬件指令或O(1)工作量。已有卡记录二维(m,n,calCount)、每行对齐、标量/向量scale、输出行补齐及scratch要求；源{curVecBaseM,alignBaseN,curVecBaseN}分别代表行数、物理行尺度与有效列数的意图，需匹配实际结构。scale是否采用UINT64编码取决于具体重载，不能把浮点数普通转uint64就视为合法编码，或泛称“910B以上全部支持”。

Brcb每次读取UB源的8个元素，每个扩展成一个32字节DataBlock。它不是直接从GM读取任意per-token数组，也不会一次调用自动生成任意M×N完整行广播。需要按N宽度和目标行布局组织后续扩展/重用，并满足源容量、dstBlkStride及地址对齐；尾行不足8个源元素时也要处理实际读取跨度。广播表、反量化结果及临时区都计入UB。

省去逐行Muls调用可减少下发开销，但底层指令数仍随数据量与分块变化，20–40%延迟收益未经本次复现。SetQuantVector/Fixpipe内放缩与Vector后处理可能承担不同因子，不必一概互斥；要防止重复乘scale或漏乘。验证每个模式、行列不相同scale、尾M/N、极值、MSD空间和不同输出dtype。
