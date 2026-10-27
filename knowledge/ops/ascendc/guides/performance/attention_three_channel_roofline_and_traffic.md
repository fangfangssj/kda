---
type: Guide
title: Attention 的 Cube、Vector 与内存性能下界
description: 统一有效 FLOP、实际流量与阶段重叠的计量。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/roofline.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/_governance.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/shape.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/resources.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/patterns.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# Attention 的 Cube、Vector 与内存性能下界

连续稠密Attention的有效乘加FLOP可写2*m*Sk*(Dqk+Dv)，Dqk=Dv=D时为4*m*Sk*D。硬件因padding、fractal与被mask位置实际发射的FLOP应另计，不能用有效FLOP除峰值却把padding损耗忽略。Cube利用率也不只由16对齐决定。

流量应按K/V/Q/O各自dtype、布局及实际访问次数计算，另计scale、mask、稀疏索引和中间workspace。GM workspace的写读会经过相应HBM/L2路径，不能因为用于核间交接就全部归入片上流量。缓存命中可减少HBM流量，但需实测/可证模型，不能同时用L2流量和HBM带宽求强度。O_acc每tile读写2*m*Dv*accBytes*ceil(Sk/Bk)只是近似，首块不读、末块直接输出及片上驻留会改变次数。

当Sk大、K/V流量主导且Q/K/V同字节宽s、Dqk=Dv时，AI近似2*m/s。它给出容量与复用的调优目标，不是正确性下界；量化还包含scale/padding/cast成本，不能保证精确翻倍。因果裁剪可能同时减少FLOP和K/V访问，但Q/O、边界tile及metadata不同比缩放，不能声称AI严格不变。分组融合之外，量化、cache复用与算法也会改变流量。

理想完全重叠时，时间下界至少为max(F/cubeRate,HBMBytes/HBMBandwidth,VectorTime,onChipBytes/onChipBandwidth)。Vector与搬运只有串行依赖时才能直接相加；真实关键路径还包含同步、launch、load/compute粒度和串行段。峰值需绑定型号、精度、指令模式和频率来源，并非所有硬件参数都可从运行时API查询。

Roofline只能排除明显受限候选，不能证明某套Tiling全局最优。以20%利用率差、30%屏障比例或≤5%流式开销作阈值需要该工作负载测量支持。容量、流水深度、块shape和任务并行共同约束，不能仅按mEff接近理论屋顶选择配置。
