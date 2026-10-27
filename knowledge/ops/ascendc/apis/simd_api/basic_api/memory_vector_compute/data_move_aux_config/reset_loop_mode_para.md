---
type: API
title: ResetLoopModePara
description: 重置loop mode参数，与SetLoopModePara搭配使用，在使能loop mode的数据搬运结束后调用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/data_move_aux_config/ResetLoopModePara.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_data_copy_intf.h
---

# ResetLoopModePara<a name="ZH-CN_TOPIC_0000002380348618"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_data_copy_intf.h"`。

重置loop mode的参数。与[SetLoopModePara](set_loop_mode_para.md)搭配使用，在使能loop mode并且设置loop mode的参数的数据搬运场景下，数据搬运结束后需要调用该函数来重置loop mode参数。

具体支持的数据通路为（以[逻辑位置TPosition](../../aux_data_structures/t_position.md)表示）：

- Global Memory -> UB
    - GM -> VECIN
- UB -> Global Memory
    - VECOUT -> GM

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void ResetLoopModePara(DataCopyMVType type)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|----------|------|
| type | 输入 | 数据搬运模式。DataCopyMVType为枚举类型，定义如下，具体参数说明请参考<a href="set_loop_mode_para.md#table1166074612214">表3</a>。<pre class="screen" codetype="Cpp">enum class DataCopyMVType : uint8_t {<br>    UB_TO_OUT = 0,<br>    OUT_TO_UB = 1,<br>};</pre> |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

与SetLoopModePara搭配使用。在使能loop mode并完成数据搬运后，必须调用该接口重置loop mode参数，否则会影响下一次对应通路的数据搬运，引发异常。

## 调用示例<a name="section1227835243314"></a>

ResetLoopModePara需与[SetLoopModePara](set_loop_mode_para.md)搭配使用，请参考[SetLoopModePara调用示例](set_loop_mode_para.md#section1227835243314)。

- [Resize 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/resize_950.md)
- [ConfusionTranspose 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/confusion_transpose_950.md)
- [LogSoftmaxBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/log_softmax_backward_950.md)
- [ApplyRotaryPosEmb 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/posembedding/apply_rotary_pos_emb_950.md)
- [UpsampleBicubic2d 950 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_950.md)
- [ReflectionPad2dBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/reflection_pad_2d_backward_950.md)
- [SoftmaxBackward 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/activation/softmax_backward_950.md)
- [GridSampler3D 950 Direct Invoke 算子知识](../../../../../operators/direct/cv/image/grid_sampler_3d_950.md)
- [ReplicationPad2d 950 ACLNN 算子知识](../../../../../operators/aclnn/math/conversion/replication_pad_2d_950.md)
- [Embedding 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/embedding_950.md)
