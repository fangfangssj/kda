---
type: Guide
title: 转置、寄存器循环与 Scatter 累加
description: 保存SIMD布局变换和寄存器接口，指出冲突写入并非原子归约。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_10_vectorized_copy.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_07_data_alignment.md
  role: supporting
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-09-11T06:15:28Z'
platforms: []
---

# 转置、寄存器循环与 Scatter 累加

AdaptivePoolGrad的16×8 FP32、16×16低精度及8×16辅助转置使用TransDataTo5HD地址表，每条源/目的地址按实际物理stride，repeatTimes与完整tile形状适配。源模板T却固定调用TransDataTo5HD<float>只在对应实例合法；小于基本矩形或尾部需padding/独立处理，src/dst重叠和未初始化repeat字段不可忽略。

Gather非尾轴可Transpose→Gather→ReTranspose改变访问，需额外缓冲/workspace及完成协议，两次转置成本计入收益。LayerNorm短行转置可让多行并行归约采用有利lane布局，原始按行连续的row-major输入并非天然跨步；不能照来源宣称所有LayerNorm行数据不连续。Reshape不自动完成物理转置。

DeepNorm短归约先将每elementNum块累加到tmp，再BlockReduceSum；片段for只覆盖完整elementNum，剩余列和超过repeat上限的行需补齐，并保留每行独立状态。BlockReduceSum产物不当然是全行单标量。LayerNorm Regbase的先乘n、ReduceSum、再乘nCorrectionFactor需与真实归约长度对应，尤其padding/尾块不能双重除N。

MicroAPI RegTensor/MaskReg可以减少中间UB读写，但寄存器活跃过多会有资源压力，不能由20个RegTensor推断并行20条或零开销/峰值。UpdateMask的剩余计数状态、每轮offset和16位循环上界必须完整。GroupedMX循环Max使用expRegTensor却未显示每行Load/提取，InplaceRMS循环offset未推进，APT scalarVal/scaleVal角色需核实，这些都只是不完整骨架。

PoolingGrad DataCopyGather旧y→Add grad→DataCopyScatter回y不是原子累加：同一批lane若有重复argmax，各lane读取相同旧值并覆盖，部分梯度会丢失；跨线程/核冲突也同理。必须证明每批目标唯一，或先分组归约，再用满足跨owner合同的写回/Atomic。Pack/Unpack是lane格式操作，不等于任意数值Cast；INT64 argmax重新解释uint32仍需范围和lane映射证明。

MaxPool前向嵌套h/w循环每次Gather只加载当次窗口位置的一批lane，并非单周期加载任意整个kernel。索引需随h/w推进，GT mask必须同步更新argmax，NaN与并列规则需确定；source未展示argmax更新便不能视为完整输出。DataCopyScatter的value/index不同dtype要各自物理offset，支持的mask/repeat/索引单位以接口为准。

SparseToDense SIMT VF_CALL示例确实传1024线程，grid-stride每线程仍串行处理若干索引并共享有限执行资源，不能推成接近1024倍速度。多维线性offset累加各轴index*stride需合法索引与宽乘积，输出重复目标的语义独立处理。
