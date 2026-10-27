---
type: API
title: GetCoreNum
description: 获取当前硬件平台的核数。若AI Core的架构为Cube、Vector分离模式，返回Vector Core的核数；耦合模式返回AI Core的核数。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetCoreNum.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetCoreNum<a name="ZH-CN_TOPIC_0000002114097973"></a>

## 功能说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台的核数。若AI Core的架构为Cube、Vector分离模式，返回Vector Core的核数；耦合模式返回AI Core的核数。

## 函数原型<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
uint32_t GetCoreNum(void) const
```

## 参数说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section25791320141317"></a>


Atlas训练系列产品，耦合模式，返回AI Core的核数


Atlas推理系列产品，耦合模式，返回AI Core的核数


Atlas A2系列产品，分离模式，返回Vector Core的核数


Atlas A3系列产品，分离模式，返回Vector Core的核数


Ascend 950PR&950DT系列产品，分离模式，返回Vector Core的核数


## 约束说明<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001647201621_zh-cn_topic_0000001442758437_section320753512363"></a>

### 自定义算子工程

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    // ... 根据核数自行设计Tiling策略
    context->SetSimdNumBlocks(coreNum);
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
    auto coreNum = ascendcPlatform->GetCoreNum();
    // ... 根据核数自行设计切分策略
}
```

- [RopeWithSinCosCache A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/posembedding/rope_with_sin_cos_cache.md)
- [RopeWithSinCosCache 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/posembedding/rope_with_sin_cos_cache_950.md)
