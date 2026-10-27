---
type: API
title: GetVecRegLen
description: 获取当前硬件平台芯片架构Vec计算单元位宽。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetVecRegLen.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetVecRegLen<a name="ZH-CN_TOPIC_0000002556048947"></a>

## 功能说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台芯片架构Vec计算单元位宽。

## 函数原型<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
uint32_t GetVecRegLen(void) const
```

## 参数说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section25791320141317"></a>

-   对于RegBase硬件平台芯片架构，返回当前硬件平台Vector计算单元位宽。
-   对于非RegBase硬件平台芯片架构，返回0 。

## 约束说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section320753512363"></a>

### 自定义算子工程

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t vecLenB32 = ascendcPlatform.GetVecRegLen() / LAYERNORM_SIZEOF_FLOAT;
    const uint32_t vecLenB16 = ascendcPlatform.GetVecRegLen() / LAYERNORM_SIZEOF_HALF;
    // ... 使用vecLenB32和vecLenB16设计Tiling策略
    return ret;
}
```

### Kernel直调

```cpp
void GetLayerNormMaxMinTmpSize(...)
{
    auto* ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return;
    }
    const uint32_t vecLenB32 = ascendcPlatform->GetVecRegLen() / LAYERNORM_SIZEOF_FLOAT;
    const uint32_t vecLenB16 = ascendcPlatform->GetVecRegLen() / LAYERNORM_SIZEOF_HALF;
    // ... 使用vecLenB32和vecLenB16设计切分策略
}
```

- [AddRmsNormDynamicQuant 950 Direct Invoke 算子知识](../../../../operators/direct/nn/norm/add_rms_norm_dynamic_quant_950.md)
- [GroupNorm 950 Direct Invoke 算子知识](../../../../operators/direct/nn/norm/group_norm_950.md)
