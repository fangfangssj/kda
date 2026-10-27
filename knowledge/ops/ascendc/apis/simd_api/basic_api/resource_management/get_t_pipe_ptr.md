---
type: API
title: GetTPipePtr
description: 获取创建TPipe对象时设置的全局唯一TPipe指针，获取后可进行TPipe相关的操作如InitBuffer。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/GetTPipePtr.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_common.h
---

# GetTPipePtr<a name="ZH-CN_TOPIC_0000001808018758"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_common.h"`。

创建[TPipe](t_pipe/index.md)对象时，对象初始化会设置全局唯一的TPipe指针。本接口用于获取该指针，获取该指针后，可进行[TPipe](t_pipe/index.md)相关的操作。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline AscendC::TPipe* GetTPipePtr()
```

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section6191129670"></a>

如下样例中，在核函数（Kernel）入口处创建TPipe对象，对象初始化会设置全局唯一的TPipe指针。在调用KernelAdd类Init函数时，无需显式传入TPipe指针，而是在函数内直接使用GetTPipePtr获取全局TPipe指针，用来做InitBuffer等操作。如需运行，请参考[GetTPipePtr样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/07_tpipe_tque/get_tpipe_ptr)实现完整的代码。

```cpp
// inQueueX、inQueueY为VECIN上的TQue，outQueueZ为VECOUT上的TQue
// 获取全局TPipe指针，使用TPipe的InitBuffer接口给TQue分配内存
GetTPipePtr()->InitBuffer(inQueueX, 2, this->tileLength * sizeof(float));
GetTPipePtr()->InitBuffer(inQueueY, 2, this->tileLength * sizeof(float));
GetTPipePtr()->InitBuffer(outQueueZ, 2, this->tileLength * sizeof(float));
```

- [DequantBias A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/quant/dequant_bias.md)
- [UpsampleBicubic2dAAGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad.md)
- [Cat 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/cat_950.md)
- [ClippedSwiglu 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/clipped_swiglu_950.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [CircularPad2d A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d.md)
- [GeGluBackward 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/ge_glu_backward_950.md)
- [FlashAttentionVarLenScore 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_var_len_score_950.md)
- [CircularPad2dBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d_backward.md)
- [GeluQuant 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/gelu_quant_950.md)

## 配套 API 与常见组合

- [FetchEventID](t_pipe/fetch_event_id.md)——FetchEventID 是 TPipe 成员接口，kernel 内经 GetTPipePtr() 调用，证据行即 GetTPipePtr()->FetchEventID 同一行代码
- [SetFlag/WaitFlag](../sync_control/intra_core_sync/set_flag_wait_flag_isasi.md)——SetFlag 消费的 event_t 由 kernel 内标准惯用法 GetTPipePtr()->FetchEventID(HardEvent::V_S) 取得，证据为同一代码序列
