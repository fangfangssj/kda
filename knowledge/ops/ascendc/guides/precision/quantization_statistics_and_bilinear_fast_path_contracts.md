---
type: Guide
title: 量化统计与双线性快速路径合同
description: 保存分层Cast和四角分支，补齐有效域、发布与尾部。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_06_multi_algo_selection.md
  role: primary
created_at: '2026-09-11T05:59:16Z'
updated_at: '2026-09-11T05:59:16Z'
platforms: []
---

# 量化统计与双线性快速路径合同

DynamicQuantUpdateScatter按FP32输入副本ReduceMax/Min、scale/offset、乘倒数加offset、RINT到INT32、deqScale1→half→低比特是实现候选。全行统计必须先完成，Vector→Scalar读取max/min的实际同步在GetValue前，单纯PIPE_V不自动涵盖Scalar。全零/常量、非有限range、scale0及INT4[-8,7]饱和要完整定义，baseline直接x/scale+7没round也没clip不是等价量化。

先INT32整数舍入并限制目标范围，half桥接才可精确传递这些小整数；任意大INT32经half可能溢出/舍入。src/dst/scratch alias、SetDeqScale状态、尾行长度和输出nibble打包均需真实API允许。更多阶段并不自动“精度最优”，应依据可用指令与同一舍入合同。

DynamicBlockQuant BF16路径升FP32、Abs、stride Max合并、WholeReduceMax，half路径Abs、128宽WholeReduceMax、Brcb再升FP32算scale。scaleCount和Brcb ceil表达式须处理不足128的尾quant block，不能先整数除128丢掉尾再ceil8。每个广播scale的physical重复布局要与后续乘法对应，不把数据block、32B block、向量repeat混同。

可变形注意力反向若0≤hLow<h-1且0≤wLow<w-1，四个corner均有效，可走ComputeGradTogether；其他走Separate逐角检查并初始化缺失角为0。h/w为0/1、坐标NaN/Inf或整数转换前越界需额外处理，unsigned h-1会回绕；只清一个wv1Local不证明其余角缓存均无旧值。

两条路径的四角权重、gradValue、gradLocations和gradAttentionWeights必须同义，边界不应改变坐标归一化/零填充合同。批量读四角可能减少搬运发起，但不等于固定4倍向量效率；重复采样目标仍需正确归并/原子与初始化完成关系。
