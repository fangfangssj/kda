---
type: Guide
title: 优化经验中的硬件模型与缓冲份数
description: 限定示例规格、Roofline 和 DMA 经验阈值，纠正流水份数与自动重叠解释。
tags:
- task.optimization
- topic.performance.analysis
- topic.memory
- topic.implementation.pipeline
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/hardware/ascend910b_arch.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/hardware/guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/double_buffering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/memory_coalescing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/pipeline_overlap.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/tiling_strategies.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/INDEX.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/optimization_patterns/guide.md
  role: supporting
created_at: '2026-09-11T03:58:39Z'
updated_at: '2026-09-11T03:58:39Z'
platforms: []
---

# 优化经验中的硬件模型与缓冲份数

来源目录叫a3但硬件页描述910B3并列20AIC/40AIV、UB256KB/可用192KB、L2 96MB、HBM1.2TB/s与320TFLOPS等数值，不能据目录名或示例当已验证平台规格。应记录实际芯片、配置/PlatformAscendC查询、单位和计算路径。256字节与256bit不同，向量宽度也不能直接解释为所有指令每周期吞吐。

Roofline模型P≤min(P_peak,BW*I)，I为同统计范围FLOPs/byte，拐点P_peak/BW。来源320/1.2≈267只在这两个峰值真实、单位统一且对应计算路径时成立；Cube FP16峰值不能直接用来判断Vector GELU/Softmax，I<10或>100也不是全芯片分类定律。小shape、资源争用和同步让可达性能低于简化上界。

UB应求同时活跃的每个buffer align(size*elementBytes)*份数之和，加上scratch、padding和框架实际预留。CopyIn/Compute/CopyOut三个阶段不必各占独立一份，来源ub/(buffer_num*pipe_num*sizeofT)仅是等大缓冲简化。按32B向下对齐要在字节域或以32/sizeofT元素进行，不能一概按32元素。192KiB示例FP16输入输出各双缓冲12288元素共96KiB，余量仍需覆盖其他资源。

开启双缓冲不必一律把tile减半，只有原配置占满且需等大双份时才如此；TQue模板深度还要匹配InitBuffer实际bufNum和生命周期。源码CopyIn(i)、Compute(i)、CopyOut(i)一般仍属于同一tile i，异步流水使物理执行可能交叠，并不会由EnQue/DeQue自动将后两者索引变成i-1/i-2。真正错位调度需明确预热、排空与尾块。

MTE2、Vector、MTE3可并行但共享带宽与地址依赖限制重叠。PipeBarrier具有指定作用域，不能泛称任何PipeBarrier都同步所有pipe；V→MTE3可使用合法SetFlag/WaitFlag事件或对应queue。低重叠可由极短任务、强依赖、不同阶段时长或配置问题引起，不能只由<5%诊断双缓冲失败。高MTE2/MTE3重叠也不单独证明争用；增加无用输出处理来“平衡”输入不是优化。

连续搬运可能比大量小DMA高效，UB重排可交换额外本地流量与GM请求数。但合并读取必须有合法完整span、适合容量且未引入大量无用数据，稀疏Gather不必永远先读全块。BNSD连续N轴的步幅一般含S*D而非仅D；实际stride决定tile形状。DataCopyPad需匹配具体参数重载，不能把DataCopyParams.blockLen=actualLen假定为元素长度。

来源256B/1KiB/4–64KiB及>64KiB效率区间、20–80%等收益、重叠60/40/10–30%均是候选经验；实际需要同输入/资源条件测量。NORMAL≤1ps气泡不是已确认硬件极限。Split-N、Split-D、Split-ND分别改变独立行、归约轴和二维tile的分配，轴被切开是否需合并由算子语义决定。

循环分配t=core; t<total; t+=C和连续分配[core*ceil(total/C),min(...)]都应检查C>0及溢出。来源N=100,C=32“4核各4块、28核空闲”错误：连续ceil方案是25核各4块、7核空闲；商余分配则4核各4块、28核各3块。核数选择需真实资源和成本，不是只选可整除数或无条件满核。
