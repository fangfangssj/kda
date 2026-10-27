---
type: API
title: GetCoreNumAiv
description: 获取当前硬件平台AI Core中Vector核数。若AI Core的架构为Cube、Vector分离模式，返回Vector Core的核数；耦合模式返回AI
  Core的核数。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetCoreNumAiv.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetCoreNumAiv<a name="ZH-CN_TOPIC_0000002078498906"></a>

## 功能说明<a name="zh-cn_topic_0000001675101185_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台AI Core中Vector核数。若AI Core的架构为Cube、Vector分离模式，返回Vector Core的核数；耦合模式返回AI Core的核数。

## 函数原型<a name="zh-cn_topic_0000001675101185_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
uint32_t GetCoreNumAiv(void) const
```

## 参数说明<a name="zh-cn_topic_0000001675101185_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001675101185_zh-cn_topic_0000001442758437_section25791320141317"></a>


Atlas训练系列产品，耦合模式，返回AI Core的核数


Atlas推理系列产品，耦合模式，返回AI Core的核数


Atlas A2系列产品，分离模式，返回Vector Core的核数


Atlas A3系列产品，分离模式，返回Vector Core的核数


Ascend 950PR&950DT系列产品，分离模式，返回Vector Core的核数


## 约束说明<a name="zh-cn_topic_0000001675101185_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001675101185_zh-cn_topic_0000001442758437_section320753512363"></a>

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

- [Im2colBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/im2col_backward_950.md)
- [CircularPad2d A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d.md)
- [Celu 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/celu_950.md)
- [AttentionToFfn A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/attention_to_ffn.md)
- [Rasterizer A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/rasterizer.md)
- [CircularPad2dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d_backward_950.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [DistributeBarrier A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/distribute_barrier.md)
- [ThreeInterpolateBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/three_interpolate_backward.md)
- [CircularPad3d A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_3d.md)
