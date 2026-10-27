---
type: API
title: GetSysWorkSpacePtr
description: 获取系统workspace指针。部分高阶API如Matmul需要使用系统workspace，相关接口需要传入系统workspace指针，此时可以通过该接口获取。使用系统workspace时，host侧开发者需要自行申请系统workspace的空间，其预留空间大小可以通过GetLibApiWorkSpaceSize接口获取。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/workspace/GetSysWorkSpacePtr.md
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
- include/basic_api/kernel_operator_swap_mem_intf.h
---

# GetSysWorkSpacePtr<a name="ZH-CN_TOPIC_0000001666431622"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取系统workspace指针。部分高阶API如Matmul需要使用系统workspace，相关接口需要传入系统workspace指针，此时可以通过该接口获取。使用系统workspace时，host侧开发者需要自行申请系统workspace的空间，其预留空间大小可以通过[GetLibApiWorkSpaceSize](../../../../utils_api/platform_info/platform_ascend_c/get_lib_api_work_space_size.md)接口获取。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

系统workspace指针。

## 调用示例<a name="section642mcpsimp"></a>

```
...
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化，传入系统workspace指针
// CopyIn阶段：完成从GM到LocalMemory的搬运
mm.SetTensorA(gm_a);    // 设置左矩阵A
mm.SetTensorB(gm_b);    // 设置右矩阵B
mm.SetBias(gm_bias);    // 设置Bias
// Compute阶段：完成矩阵乘计算
while (mm.Iterate()) { 
    // CopyOut阶段：完成从LocalMemory到GM的搬运
    mm.GetTensorC(gm_c); 
}
// 结束矩阵乘操作
mm.End();
```

- [grouped_matmul_add A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_add.md)
- [grouped_matmul_add 950 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul_add_950.md)
- [UpsampleBicubic2dAAGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad.md)
- [DeformableConv2d A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/conv/deformable_conv2d.md)
- [FlashAttentionScoreGrad A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/flash_attention_score_grad.md)
- [UpsampleBilinear2dAA A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa.md)
- [LSTM A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/rnn/lstm.md)
- [NsaSelectedAttention A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/attention/nsa_selected_attention.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)

## 配套 API 与常见组合

- [TPipe简介](../t_pipe/t_pipe_intro.md)——两者共同作为 REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling) 同一注册语句的必需参数，两张卡的调用示例均固定成对出现，P(a|b)=1.00
