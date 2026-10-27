---
type: Guide
title: 索引校验、前缀计数与分片归属
description: 保存Argmax、Embedding、Scatter及插值边界的完整前提。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_07_index_boundary.md
  role: primary
created_at: '2026-09-11T05:41:33Z'
updated_at: '2026-09-11T05:41:33Z'
platforms: []
---

# 索引校验、前缀计数与分片归属

Argmax索引INT64改INT32既影响接口ABI也影响数值范围，不能只称优化。di*hi*wi须宽域计算并证明全部有效索引≤INT32_MAX；每个索引省4B只表示索引张量节省一半，非整体访存减半。转换前验证，不能先Cast截断再判断范围。

Embedding排除paddingIdx需明确负索引/哨兵如何规范化，uint64 currentId直接与负paddingIdx比较可能得到非预期语义；排除padding并没有解决其他索引越界或重复索引梯度聚合。Gather“硬件自动取模”不是缺少显式check就能推出的事实；clamp、报错、回绕和负索引归一化属于不同合同，空维度xDim-1还会下溢。

MaskedScatter用前缀末项推导所需updates时，必须明确前缀inclusive/exclusive、每项计数单位、outer/inner展开关系。xOuter0不能访问position[-1]，乘积须宽域且非负、前缀单调与最终计数合法，读取updates之前检查容量。C/C++标准assert通常只有一个参数，源两个参数片段需实际宏定义；可能被编译掉的断言不能作为生产输入校验的唯一保障。

Deformable Attention二维双线性插值有四个角，每个有效corner需0≤h<height且0≤w<width。源各分支只查部分上下界，仅在外层已约束采样坐标范围时可能充分；例如只检查hLow≥0不能排除hLow100而height10。clamp采样坐标会改变与越界corner取零的权重语义，不能随意替代。

Scatter按输出piece做相对索引，可先判断全局索引归属再转局部索引。计算-i*pieceSize时无符号乘法/取负可能回绕，强转int32无法挽回；须在足够宽的有符号/受检查域完成，再证明局部范围。每piece需从原索引重新推导，原地多次Adds可能累计偏移。当前piece真实长度而非padded容量决定合法输出。

片间不相交只排除片间写冲突，片内重复索引仍由覆盖顺序、归约或原子合同处理。PIPE_V_S一类helper只能按其实际Vector→Scalar实现使用。INT64输入UB加INT32临时UB应同时计字节，GetPhyAddr后改指针类型只是视图，不能替代数值转换或范围证明。

反向shape应分别对应value、samplingLocations和attentionWeights等原输入；源仅AppendDim第0维且含省略号，不是完整InferShape。应清理原输出shape，验证rank、各轴关系及动态维合同，不能拿首维相等作为全shape推导正确。
