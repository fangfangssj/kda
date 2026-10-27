---
type: Guide
title: FA 优化归因与内存证据
description: 纠正Codegen唯一根因和大输入容量估算。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: primary
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# FA 优化归因与内存证据

数学算法等价且tiling字段符合某公式，不证明剩余性能差只能来自工具链。地址布局、同步、指令数量、寄存器/缓存、调度和编译选项仍可能不同；要归因codegen需要实际编译产物与控制变量证据。相邻baseline漂移6%、候选±2–7%只说明该实验收益不清楚，不能证明编译器已消除所有Barrier或源码无优化空间。

TaskDuration减最长核耗时的611us对43us可以量化时间差，但92.9%差值不唯一等于“同步头开销”；可能含启动、调度、不同统计范围和测量聚合。aiv_scalar_ratio高也不独自证明Scalar计算，API发射次数是可进一步核验的候选。Tile32/64、任务数12–32及N-pack减少任务是可调参数，不是所有shape固定最优。

历史11case几何1.68倍偏向H1小样本，另批大H/N得到0.31–0.52倍，体现测试分布会改变汇总，不能把前者作为全族收益。序列轴M/K和并行度BH/N应分开扫描；sumkerneldevice与Host墙钟基线不可交叉算ratio。profiler失败期旧健康窗口基线与新event数据应明示不可保证同环境。

FP16 Q加O基本字节数2*B*H*Sq*D*2。B1,H16,Sq458752,D128为3.5GiB；B16,H64,Sq200000,D128为104857600000B=97.65625GiB，来源9.8GiB少10倍，尚未计KV、baseline副本和workspace，不能以64GB卡无OOM推广。固定B/H/D/Sk的Sq扫描若中途缩BH，就不再是单变量实验。

短序列S1和长KV流式适用性受有效shape/特性与完整资源影响；S1所谓“长序列458752”实际是Sq长而Sk128，不能据此说一次softmax覆盖458752列。源平均0.922既非展示行的简单比值也未给全部20case，只能保留为来源汇总，不能重算成已验证统计。

错误码507015、全零、profiling空目录和timeout都有多种可能，重试成功不排除初始化竞态，换case恢复也不证明原kernel无错误。可保存故障原始证据与受限实验条件，不能由一次7shape/5次压力/30case通过宣称所有FA变体生成完备。
