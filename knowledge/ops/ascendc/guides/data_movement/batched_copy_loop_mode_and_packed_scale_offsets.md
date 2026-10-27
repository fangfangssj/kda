---
type: Guide
title: 批量搬运、LoopMode 与交错 Scale 偏移
description: 保存多维搬运和MX物理行距，补齐参数状态与反向映射。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_07_data_alignment.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_10_vectorized_copy.md
  role: supporting
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-09-11T06:15:28Z'
platforms: []
---

# 批量搬运、LoopMode 与交错 Scale 偏移

AdaptivePoolGrad可一次搬运core.ncShape个块，GM srcStride=(Do*Ho*Wo-currentSpatial)*sizeofGrad，UB dstStride0按接口隐式行距拼接；block.dohowoShape必须确实对应连续空间片段，任意D/H/W矩形不能只用元素总量代替真实stride。超blockLen或repeat范围需分组/尾，拆32B块并不自然保持原矩形的各轴间隙。

GroupedMX示例in UB gap=2*ceil(dataLen/32)-ceil(dataLen/16)，对应16位输入从32B基础行距扩到64B等特定布局；不同T需重推。scale输出每组两个uint8以交错格式写出，blockCount=ceil(blockCount/64)、blockLen=2*dataLen、src gap=2*(ceil(dataLen/32)*32-dataLen)/32、GM gap=2*(postAxis-dataLen)都绑定该格式，不能把同名dataLen当原始未量化元素数。Interleave只是lane重排，奇数group/scale尾部、postAxis≥dataLen及两端完整跨度要核验。

LoopMode将NHWC局部块的H和N循环编码为loop1/loop2：GM stride按wArgmax*c*sizeof(T1)、argmaxPlane*c*sizeof(T1)，UB stride按实际W/H与alignedC。设置OUT_TO_UB方向、发起DataCopyPad、再Reset同方向；异常/空路径或后续搬运也需状态正确，配置不应泄漏到无关操作。T1为grad还是argmax须由实际指针/dtype判断，不能照标题混用。

DataCopyEx包装器的count/blockLen/ubAligned额外1块gap可集中参数，仍需全部结构字段初始化。片段只在dst为LocalTensor时执行，GlobalTensor目标没有else路径；不能把该函数视为双向copy的完整通用实现。模板可能内联，函数包装次数不是硬件指令数量。

跨batch广播mask的1NSD路径由首batch余行、完整中间batch、末batch前缀组成；start/end是何种开闭区间须统一，end恰下一batch起点不能多拷一行。同一mask缓存复用要固定N/S/D范围，B1SD/11SD各有不同映射，不能仅判断batch相同便忽略channel切换。

Gamma预加载适用于所有所处理行共享同一gamma切片且UB区域保留到最后消费；source FastCompute循环仅++pid未展示offset推进，需按行实际偏移。若Get<T>(numColAlignF32)是元素count而非offset参数，需核对重载，不能把函数调用自动视为切到后半区。所有预加载均包含第一次GM搬运及完成同步。
