---
type: Guide
title: 多轴尾块、池化索引与阶段归属
description: 保存BatchNorm及SparseToDense切分，纠正索引单位。
tags:
- task.optimization
- topic.memory
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_11_tail_block_handling.md
  role: primary
created_at: '2026-09-11T05:55:38Z'
updated_at: '2026-09-11T05:55:38Z'
platforms: []
---

# 多轴尾块、池化索引与阶段归属

正长度N、正tileT时，L=ceil(N/T)，last=N-(L-1)*T得到1..T；N0不能照抄此尾公式，否则L-1可能负值/无符号回绕。SparseToDense分别划分default填充和sparse写入，normal/tail核各有loop/tail字段，真实defaultValueUsedCoreNum0须空路径，其他空核不读尾核字段。两阶段覆盖域不同，默认填充必须全局完成后再稀疏覆盖，否则会清掉已写值；均匀元素数不保证随机索引成本均衡。

BatchNorm channel轴A可用blockFactor=ceil(A/C)、usedCore=ceil(A/blockFactor)、tailA=A-(usedCore-1)*blockFactor；A0或C0先处理。R0按r0UbFactor循环并记录r0Tail，R1补充打包procNR0=floor(r0UbFactor/R0Align)，只在procNR0≥1时合法。来源R0Align≤r0UbFactor/2且R1>1是候选路径条件，不能推成支持任意shape；A/R0/R1各自逻辑长度、对齐padding、统计域及尾块均需匹配。

AdaptivePool每轴start=floor(o*I/O)、end=ceil((o+1)*I/O)，I%O不整除可产生重叠窗口，三轴OR作为来源重叠策略条件须I/O正且真实窗口合同一致。基础示例end用floor会漏采样，不是正确无重叠对照。重叠表示可能有多输出梯度贡献到同输入，但若输入域唯一分配并内部归并，也可不用跨核atomic；低精度FP32 workspace和最后Cast独立保证完整阶段。

PoolGrad GenkernelIndex示例展示W/H扩展，D维生成、起始值、全部Nc lane和有效kD*kH*kW容量仍需补齐。用FP32存索引后Compare，整数精确域有限，超过2^24的相邻值可能合并，不能因为原argmax在INT32范围就安全转float。源固定VL64是某路径，不是所有dtype/平台吞吐结论。

IndexConvNhwc源按idx/W得到H、idx-H*W得到W，只适用于索引本身是空间HW平面编码；若idx含HWC，则应先分出C、再除W。重组当前tile相对坐标时减curH/curW可能负，检查边界后再窄化，最后按((h*Wactual+w)*C+c)+nBase恢复正确元素位置。Vector Div的整数/浮点语义及精度要匹配，不能把实数除法自动当整商。

验证空轴、小于tile、整除/余1、多轴同时尾、跨核同目标以及default/sparse阶段竞争。来源“所有场景近完美负载均衡”“4–8倍”未附实际case/计时，不作为已验证收益。
