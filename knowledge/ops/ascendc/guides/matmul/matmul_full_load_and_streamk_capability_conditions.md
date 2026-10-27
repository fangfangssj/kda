---
type: Guide
title: Matmul 全载和 StreamK 能力条件
description: 明确全载的驻留范围、切K覆盖与示例筛选阈值。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/fallback-algorithm.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-variants.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-fields.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/matmul/fallback/tiling-flow.md
  role: supporting
created_at: '2026-09-11T06:32:03Z'
updated_at: '2026-09-11T06:32:03Z'
platforms: []
---

# Matmul 全载和 StreamK 能力条件

一侧全载须针对本核实际复用范围计算：A全载至少Mb*AlignK*dA，加B流式kB*Nb*dB*缓冲份数及两侧scale/bias/scratch。整矩阵M与baseM不可混用；来源M128,K4096的FP16 A已有1MiB，不能称可放512KiB L1。简单半L1阈值只是预留策略，不保证另一侧全部资源可用。

T次使用同一A理想上避免(T-1)/T的该侧重复流量，但不等于整体耗时同比减少；T5的80%节省只针对该流量项。T=N/(C*baseN)并非所有调度的复用次数，要按实际每核遍历、group/batch、物理地址与cache命中确定。带宽85%/70%、任务收益5–15%均是来源经验，不能作普遍能力判据。

全载A/B标志互斥、全载与StreamK互斥可以是特定模板限制，不是内存或算法永远无法共存。切K后每个分区也可能全驻留其局部K，双侧都能放入片上也非逻辑不可能；是否实现取决于具体Kernel和分发。

参考纯StreamK在MN tile数≤C/2且K≥max(8192,256*C)/2、K按256对齐时考虑；DP+StreamK示例还限定M/N为256倍数、tile数≥C且末波tile在(0,C/2]、K≥max(8192,128*C)/2。这些与总述K≥32768及K16384示例不一致，应标为特定候选启发，不推广所有dtype/FP4/shape。

K分区记录精确起止与最后有效长度，ceil(16384/6)=2731，来源2732只有额外对齐才可能合理且需明示。所有分区并集覆盖[0,K)、不重不漏；padding不能越读，partial结果需真实FP32等约定的归并，bias只加一次，量化scale按对应K组。workspace=C*256*256*4再加RPC只是固定base的实例，应按实际块与同时活跃partial计算。

来源切K后强制L1buffer2、L0C1、edgeSplit1、tailMain0、usedC是局部实现键，不是StreamK定义；验证K尾、不同分区数、数值顺序和全核完成协议，模型收益需完整kernel计时。
