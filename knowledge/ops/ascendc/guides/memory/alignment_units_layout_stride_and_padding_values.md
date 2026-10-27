---
type: Guide
title: 对齐单位、布局行距与填充值
description: 修正以元素数冒充字节数的多处说明，保存不同类型的物理布局。
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
- resource: https://gitcode.com/cann/asc-devkit/blob/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/SIMD-API/basic_api/data_structures/GlobalTensor/SetValue.md
  role: constraint
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-10-17T18:13:11Z'
platforms: []
---

# 对齐单位、布局行距与填充值

对齐公式作用于元素数量时，结果单位仍是元素：FP16/BF16按16元素就是32B，FP32按8元素也是32B。BatchNorm文称16/8字节、LayerNorm文称FP32行16字节、NormCommon称32B或16float、DynamicQuantUpdateScatter称rowLen取整16是16B，均不能照搬。RMSQuant的F32按64元素是256B，INT4按256逻辑值通常占128B，需区分packed格式。

Dequant nAlign=ceil(N*4/64)*64/4是按FP32计算行对齐，随后XTYPE较窄时输入物理行距另算；不能由名称推成所有NPU最优64B或自动无bank conflict。ClippedSwiGLU、DeformableAttention的32B块与256B repeat是不同粒度，缓存line也不能一概等于32B。Bitmask向上取整公式仅对正2次幂alignment适用，加法需防溢出。

按AlignUp(row,B/sizeofT)分配后，计算行起点用rowAlign、有效归约count用row；源AddRMS正是ReduceSumCustom(...,numCol)，不能把padding加入分母。Modulate ubLength=ub/bufferCount/dtype/ALIGNED_SIZE*ALIGNED_SIZE中ALIGNED_SIZE作用于元素，而另一alignedD公式将ALIGNED_SIZE视为字节，必须分别核对具体常量/调用合同。五或六份buffer模型还需BF16加载到localData[ubLength]的额外后半区与FP32转换容量，不能只看逻辑类型。

GM→UB扩展copy blockLen为有效字节，GM stride为字节gap，UB stride为32B gap，且UB gap需扣掉此次搬运自身的隐式补齐。AscendQuant的(baseLen-xLen)*sizeofT/32只在恰好对应已补齐基准时才成立；不能因整数除法整齐就认为行距正确。blockCount/Len/pad/stride在窄化之前检查接口上限，uint16或uint32可表示不等于API全域可用。

LayerNorm rightPadding=rAlign-r需设定合法paddingValue，未初始化结构体不能假设默认为0。isPad=false或{true,0,0,0}不证明所有dummy lane都有指定中性值；需要按具体重载检查。Softmax input和bool mask分别按各dtype对齐，物理宽度可不同。SoftmaxGrad把到下一dataBlock的padding交DataCopyPad，剩余dupNum用逐行Duplicate，必须保证两段无缝覆盖且MASK_VALUE符合梯度乘积的语义。

BatchNorm SkipPadSubMean只对有效R0减mean，避免padding0变成-mean后污染平方和；repeat/stride字段范围与每行偏移都需匹配。源完整R0段直接传lineNum为repeat、尾段仅显示lineNum/255个满批，尚需处理lineNum超上限及余数，不能认为示例已覆盖全部大输入。后续归约仍须有效域和正确分母。

严格按物理layout验证输入编码的行列号，往返搬运后再加入计算，检查有效区与padding分别符合合同。自动连续化只代表框架可能预先复制，不证明TensorList的裸Kernel接受任意stride；ReinterpretCast不执行数值转换，对齐本身也不授予原地宽化Cast合法性。

## GM 标量写回的缓存行粒度

`GlobalTensor::SetValue` 的写回合同与 32B 数据搬运块不同：该接口先改写每个 AI Core 内部的 DCache，随后以 64B Cache Line 写出到 GM。需要立即发布到 GM 时，应按该接口说明配合 `DataCacheCleanAndInvalid`；不能仅凭元素地址不同就认定多个核互不覆盖。

对使用 `SetValue` 的多核写回，需同时核对不同核所写地址的 Cache Line 间隔与 64B 对齐，不能以“每核写不同元素”或“按32B划段”替代其地址约束。这里的64B来自 `SetValue` 的接口说明，不推成所有搬运接口的统一粒度，也不据此宣称 `DataCopyPad` 的实现采用相同机制。

## 配套 API 与常见组合

- [DataCopy](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_gm_and_ub_continuous.md)
- [DataCopyPad](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_gm_to_ub.md)
- [DataCopyPad](../../apis/simd_api/basic_api/memory_vector_compute/data_move/data_copy_pad_ub_to_gm.md)
- [copy（Unified Buffer到Global Memory的数据搬运）](../../apis/simd_api/tensor_api/reg_vector_compute/vector_compute_store/copy_ub_to_gm.md)
- [GlobalTensor::SetValue](../../apis/simd_api/basic_api/data_structures/global_tensor/set_value.md)——标量写入先落每核DCache，跨核地址所有权需按该接口的缓存行约束核对。
