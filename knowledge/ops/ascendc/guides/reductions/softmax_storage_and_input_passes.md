---
type: Guide
title: 独立 Softmax 的驻留、重读与在线统计
description: 说明输出完整概率时的统计依赖，比较全量驻留、三遍重读与两遍在线统计的资源代价。
tags:
- task.design
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_ara_online.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_ar_full_load.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_ar_recompute.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_ar_small_r.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_ara_full_load.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_ara_recompute.md
  role: supporting
created_at: '2026-09-10T14:53:40Z'
updated_at: '2026-09-10T14:53:40Z'
platforms: []
---

# 独立 Softmax 的驻留、重读与在线统计

独立 Softmax 的输入已经位于 GM，输出是完整概率。沿归约轴 R，有限输入的稳定计算为 `m=max(x)`、`l=sum(exp(x-m))`、`y=exp(x-m)/l`。在最终统计量确定前，不能把当前 chunk 的局部归一化结果直接当作整行最终输出。

## 三种存储选择

| 组织方式 | 参考算法的输入读取 | 保存内容与代价 |
|---|---|---|
| FullLoad | 一个处理块一次搬入 | 保存整段输入或指数中间量；全部存活缓冲必须能容纳 |
| Recompute | max、exp-sum、输出各一遍 | 减少常驻输入体积，增加 GM 重读与指数重算 |
| Online max/sum | 在线统计一遍、最终输出一遍 | 保留每个独立归约组的 running max/sum，减少一遍输入重读 |

这里的次数描述参考算法的逻辑输入遍历，不是硬件总访存量，也不包括临时写回、缓存效应和其它节点。FullLoad 不等于 UB 内只做一次扫描。

## 在线统计仍需最终输出阶段

对当前非空有限 chunk，维护：

```text
m_new = max(m_old, chunk_max)
corr = exp(m_old - m_new)
l_new = l_old*corr + sum(exp(chunk - m_new))
```

旧 sum 与新 chunk 的指数必须使用同一个最大值基准。完成所有 chunk 后，再用最终 m/l 产生完整概率。空组、全屏蔽组、NaN/Inf 等需要接口单独定义，不能仅依靠 `m=-inf,l=0` 的初始化推定结果。

融合 Attention 还维护随 V 加权的输出状态，可以生成 score 后即时消费而不输出完整概率矩阵；它与独立 Softmax 的两遍读取是不同问题，见[融合 Attention 的在线状态](../attention/online_softmax_state_and_partition_merge.md)。

## 轴与缓冲预算

`[A1,R]` 的末轴归约与 `[A1,R,A0]` 的非末轴归约使用不同地址映射。短 R 可以通过布局重排让多个独立归约组沿向量方向处理，但需要计入转置与填充成本。不能只重命名 A0/R 就改变实际存储顺序。

UB 预算逐项计算输入队列、输出队列、转换结果、临时指数、归约缓存和 running 状态，并区分元素数、字节数与缓冲份数。参考各模板的固定预留量、阈值及构造函数差异属于该样例版本，不作为所有算子的通用配置。

尾块只让真实有效元素参与 max/sum，最终只写有效输出；较宽精度中间量和树形合并会影响资源与数值路径。用 R 小于、等于、大于单块容量，以及 R/A0 非整齐尾块比较结果和总耗时，再决定驻留、重读或转置是否合适。
