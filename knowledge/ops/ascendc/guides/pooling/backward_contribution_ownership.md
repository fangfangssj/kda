---
type: Guide
title: 池化反向的梯度贡献与 gather/scatter 归属
description: 区分按输入收集与按输出分发梯度的所有权、局部合并、跨核累加和实际性能成本。
tags:
- task.design
- topic.memory
- topic.parallelism
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/backward-implementation.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/pooling-patterns/references/grad-v2-lessons.md
  role: supporting
created_at: '2026-09-10T14:50:50Z'
updated_at: '2026-09-10T14:50:50Z'
platforms: []
---

# 池化反向的梯度贡献与 gather/scatter 归属

池化反向需要把每个输出梯度的贡献加到其对应输入位置。对平均池化，贡献由该输出窗口的除数归一化；对最大池化，应服从前向保留的 argmax 及接口规定的索引语义，不能把平均池化的均匀分配直接复用到最大池化。

## 两种遍历方向

input-driven gather 为输入位置分配唯一写入者，收集覆盖它的输出窗口贡献；output-driven scatter 遍历输出位置，把贡献发送到相关输入位置。两者应表达相同的数学求和，但访问顺序、读取重复次数、临时空间和跨核写冲突不同。

如果多个输出块落到同一输入梯度，必须有明确的合并机制。来源描述了在 UB 内先合并一个块的贡献、再处理跨块重叠的方式。采用原子累加时，目标初值、全部贡献的完成边界、写回作用范围以及最终转换均需闭合；低精度输出可以使用较宽精度 workspace 累加后转换，实际 dtype 和 API 支持按目标平台核对。原子模式本身不会初始化输出。

## 布局和实现质量决定成本

分块读、局部转置与向量累加可减少逐位置 GM 往返，但会消耗 UB 和额外搬运。naive scatter 的跨步读和逐位置原子写，与块内合并的 scatter 不是同一种性能实现。来源后续记录撤回了固定窗口体积切换阈值，因此不能仅凭“大窗口”或“有重叠”保证 scatter 更快。

输出驱动遍历可以复用正向窗口边界；输入驱动遍历需要找出覆盖该输入的输出窗口。调整循环方向时，仍须保持贡献集合、padding/除数规则和边界裁剪一致，且浮点求和顺序可能改变。

## 验证边界

用全一梯度和单点非零梯度核对覆盖次数及写入位置，覆盖重叠、边缘窗口、不同 layout、非整齐尾块与多核分配。性能比较应具有相同数学工作量，并说明是否计入布局转换、初始化、转换 pass 和框架调度。来源的具体加速比及固定阈值未在此复现，不作为通用结论。

同步按真实生产/消费流水建立；参考中将 UB 写回 GM 归为 MTE2、把生产写后消费读称作 WAR、用通用 PipeBarrier 替代所有跨流水依赖的表述不沿用。相关原则见[同步层次](../regbase/synchronization_layers.md)和[原子写作用域](../memory/dma_atomic_write_scope.md)。
