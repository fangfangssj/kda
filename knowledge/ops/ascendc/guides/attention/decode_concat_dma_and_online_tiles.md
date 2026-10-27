---
type: Guide
title: Decode 拼接搬运与 online tile 归约
description: 融合 nope/rope 的物理暂存，并保存两级点积、有效域与在线 softmax 状态。
tags:
- task.optimization
- topic.compute.composite
- topic.data_movement.layout_transform
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/discovered/disc_X4.md
  role: primary
created_at: '2026-09-11T02:29:01Z'
updated_at: '2026-09-11T02:29:01Z'
platforms: []
---

# Decode 拼接搬运与 online tile 归约

当每个key的nope与rope分量分别存储时，可用合法的两组带stride搬运，将对应tile写入同一UB行的两段，形成[T,Dqk]逻辑拼接，供后续点积读取。必须分别定义源行距、目的行距、字节单位、Dnope/Drope有效长度与padding，避免覆写相邻行。拼接后的ND布局并不会自动成为Cube需要的NZ，两次相邻调用也不等于一条物理DMA。

一个KV tile搬入后，各key行与相同query做Mul，再通过块级部分和与整行合并形成score[r]。两级归约必须覆盖全部Dqk，包括非满块，并按真实物理跨度定位部分和；Mul→归约→下一Vector/Scalar消费者都有数据依赖。按tile批量搬入可将逐key搬运/同步的调用数从O(Skv)降到O(ceil(Skv/T))量级，前提是实现确实批量加载；计算部分仍可能逐行展开，不能只改循环名字即宣称完成。

跨KV tile维护FP32的max m、分母l与未归一化输出o。合入分数s时，m'=max(m,max(s))、alpha=exp(m-m')、p=exp(s-m')，l'=alpha*l+sum(p)，o'=alpha*o+pV；最终输出o/l。初始m=-inf,l=0,o=0，但空或全屏蔽tile需避免-inf-(-inf)并按语义处理。更新max后只重标l而漏掉o会改变结果，mask、scale、sink与V列顺序也必须保持。

Vector的一lane Exp替代Scalar libm只是实现候选，仍有mask设置、Vector→Scalar同步和调用成本。V预取必须计入额外缓冲及释放时刻，不能为了减少score阶段时延覆盖后续PV所需数据。算法等价不意味着浮点逐位一致，需检查溢出、特殊值及长期在线误差。

来源T=64约150KiB/192KiB、T=80更慢以及0.11–0.27到0.25–0.79的归一化性能都是特定配置，未在本批复现。验证D与Skv双尾、不同拼接长度、全masked与多tile状态，并计布局准备及最终归一化的完整代价。
