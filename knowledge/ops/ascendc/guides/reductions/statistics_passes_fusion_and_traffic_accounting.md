---
type: Guide
title: 统计遍历、融合与真实搬运量
description: 保留 Welford、归约和融合方案，纠正单遍归一化与固定内存瓶颈假设。
tags:
- task.optimization
- topic.compute.reduction
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/algorithm_insights/reduction_ops.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/algorithm_insights/elementwise_fusion.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/algorithm_insights/INDEX.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-knowledge/references/a3/algorithm_insights/guide.md
  role: supporting
created_at: '2026-09-11T03:58:39Z'
updated_at: '2026-09-11T03:58:39Z'
platforms: []
---

# 统计遍历、融合与真实搬运量

Welford按count+=1、delta=x-mean、mean+=delta/count、M2+=delta*(x-mean)更新统计；总体方差为M2/count，样本修正需另定分母及空/单元素语义。它可一遍计算统计，但最终LayerNorm仍需每个x与最终mean/var：若原输入不能保留在UB，通常要重读或保存中间数据。来源“一遍Welford使整个归一化读取减半”遗漏输出阶段，也不能断言Welford天然比常规方法更不稳定。

树归约改变相加顺序，常有较短误差累积路径但不保证每组数据都更精确；ReduceSum/WholeReduceSum实际实现、临时区和返回位置按重载确认，不能从API名字断言内部唯一算法。宽轴可分块形成统计再合并，短D可多行合并以提高搬运粒度；mergedRows需同时计入输入/输出、精度提升、统计、gamma/beta和scratch的真实存活字节。

跨核归约需非重叠且完整的输入分区、每核部分结果、发布/完成和最终归并；atomic路径还要正确初始化及支持类型，不一定总比workspace归并慢或更简单。若有足够独立行，可按行分核减少跨核统计通信。在线softmax只一遍得到max/sum不等于一遍产生完整输出，见[softmax驻留与重读](softmax_storage_and_input_passes.md)。

逐元素融合常可减少中间GM写回/重读。例如GELU(x)再加bias，在不计缓存且同大小独立数组时是读x、写temp、读temp/bias、写y共5份数据；融合为读x/bias、写y共3份，流量减少40%，不自动等于时间减少40%。纯一元读写的带宽上限按至少2*sizeof(T)字节/元素，不能只用BW/sizeof(T)；bias广播和缓存还会改变实际traffic。

Elementwise不是恒为memory-bound，特殊函数、很多计算链、Scalar开销、小形状或低吞吐dtype都可能主导。相近Cube/Vector比例不证明已完美重叠，也不意味着只能同时减少两侧计算才能更快；访存、串行区、通信和调度仍可能有空间。Amdahl/roofline需真实工作量与有效峰值。

融合前后顺序必须相同：来源前例GELU(x)+bias，却在原地代码写Adds再Gelu，即GELU(x+bias)，语义不同。SwiGLU/GeGLU一般有独立gate/value支路，不能无条件简化成x*sigmoid(x)或x*GELU(x)。Axpy、ReluV2等名字不证明任意mul+add或Add+ReLU均有对应融合合同。

原地LocalTensor运算需要该重载允许别名且旧值已无消费者；从inQueue取得的buffer不能无证明直接EnQue到另一outQueue，queue ownership与完成后回收应一致。bias常驻需InitBuffer/Get及合法DataCopy，Tensor bias不能作为Adds的Scalar直接传入。常驻小参数可省重复加载，但需计占用、dtype和每行对应关系，物理容量必须以字节证明。
