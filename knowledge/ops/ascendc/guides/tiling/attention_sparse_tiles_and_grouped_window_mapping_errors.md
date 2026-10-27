---
type: Guide
title: Attention 稀疏块与分组矩阵窗口映射边界
description: 保留基本块对齐和窗口化思想，指出尾窗口反向扫描的代码反例。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_02_adaptive_tiling.md
  role: primary
created_at: '2026-09-11T06:10:16Z'
updated_at: '2026-09-11T06:10:16Z'
platforms: []
---

# Attention 稀疏块与分组矩阵窗口映射边界

FlashAttention示例Softmax基本块要求M%8==0、K%64==0，这是该路径的判定，其他平台/非基本块可有不同API与padding路径。跳过S2块必须确认该query tile全部行均被mask；Causal的对角线偏移还受Q/K长度及模式影响，Band/Prefix不能直接复用一个因果公式。边界块仍需逐元素mask和有效长度。

完全mask的块可不参与在线max/sum/output状态更新，但全行没有合法key时最终输出/统计需明确，不能用未初始化状态做归一化。IBShare约束、两次BMM和Softmax scratch应共同决定S2块尺寸；“8/64对齐”不证明UB可放或最优。

GroupedMatmul ASWT的主窗口w=min(4,mTiles)、f=floor(mTiles/w)-1、尾窗口h=mTiles-f*w。h可大于4，例如mTiles10时w4、f1、h6；mTiles0/nTiles0及curBlock<count须先处理。窗口内M先推进、N后推进，奇数窗口可反向N以改变局部性；来源文本称偶数行反向而代码rowIdx%2!=0，二者冲突。

更严重的是来源在尾窗口内仍用rowIdx=floor(index/(nTiles*w))判定反向，尾窗口h>w时rowIdx可能中途增加，破坏双射。例如mTiles10,nTiles3时总30任务，直接执行该片段只覆盖24个不同tile：尾窗某些m对应n2重复、n0缺失。尾窗应使用固定窗口号f决定是否反向，而非继续按主窗宽度求rowIdx；改后需枚举所有tile验证。

这一反例来自本地纯整数映射检查，不是NPU性能或完整GroupedMatmul实现的实测失败，真实上游代码可能有未展示修正。窗口改变访问顺序不能证明存在或命中所谓L3缓存；具体缓存层级和收益需实际目标资料与测量。参照知识库已有蛇形遍历卡的完整逻辑，保留此材料独有的省略/误写警示。
