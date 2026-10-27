---
type: API
title: GetCoreNumAic
description: 获取当前硬件平台AI Core中Cube核数。若AI Core的架构为Cube、Vector分离模式，返回Cube Core的核数；耦合模式返回AI
  Core的核数。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetCoreNumAic.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetCoreNumAic<a name="ZH-CN_TOPIC_0000002114052977"></a>

## 功能说明<a name="zh-cn_topic_0000001675181281_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台AI Core中Cube核数。若AI Core的架构为Cube、Vector分离模式，返回Cube Core的核数；耦合模式返回AI Core的核数。

## 函数原型<a name="zh-cn_topic_0000001675181281_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
uint32_t GetCoreNumAic(void) const
```

## 参数说明<a name="zh-cn_topic_0000001675181281_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001675181281_zh-cn_topic_0000001442758437_section25791320141317"></a>


Atlas训练系列产品，耦合模式，返回AI Core的核数


Atlas推理系列产品，耦合模式，返回AI Core的核数


Atlas A2系列产品，分离模式，返回Cube Core的核数


Atlas A3系列产品，分离模式，返回Cube Core的核数


Ascend 950PR&950DT系列产品，分离模式，返回Cube Core的核数


## 约束说明<a name="zh-cn_topic_0000001675181281_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001675181281_zh-cn_topic_0000001442758437_section320753512363"></a>

### 自定义算子工程

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    // ...按照aivNum切分
    context->SetSimdNumBlocks(ascendcPlatform.CalcTschNumBlocks(aivNum, aicNum, aivNum));
    return ret;
}
```

### Kernel直调

```cpp
void GetInfoFun()
{
    auto* ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return;
    }
    auto aicNum = ascendcPlatform->GetCoreNumAic();
    auto aivNum = ascendcPlatform->GetCoreNumAiv();
    // ... 根据aicNum和aivNum自行设计切分策略
}
```

- [UpsampleBicubic2dAA A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa.md)
- [AddLora A2 ACLNN 算子知识](../../../../operators/aclnn/math/math/add_lora.md)
- [DeformableConv2d A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/conv/deformable_conv2d.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [UpsampleBicubic2dAAGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad.md)
- [FusedLinearCrossEntropyLossGrad A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/matmul/fused_linear_cross_entropy_loss_grad.md)
- [SparseFlashAttention 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/sparse_flash_attention_950.md)
- [UpsampleBilinear2dAABackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_bilinear_2d_aa_backward.md)
- [FlatQuant A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/quant/flat_quant.md)
- [AlltoAllAllGatherBatchMatMul A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/allto_all_all_gather_batch_mat_mul.md)
