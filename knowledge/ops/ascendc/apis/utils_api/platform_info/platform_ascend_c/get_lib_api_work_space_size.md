---
type: API
title: GetLibApiWorkSpaceSize
description: 获取AscendC API需要的workspace空间大小。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetLibApiWorkSpaceSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetLibApiWorkSpaceSize<a name="ZH-CN_TOPIC_0000002114097981"></a>

## 功能说明<a name="zh-cn_topic_0000001663835704_zh-cn_topic_0000001391767420_section36583473819"></a>

获取AscendC API需要的workspace空间大小。

## 函数原型<a name="zh-cn_topic_0000001663835704_zh-cn_topic_0000001391767420_section13230182415108"></a>

```
uint32_t GetLibApiWorkSpaceSize(void) const
```

## 参数说明<a name="zh-cn_topic_0000001663835704_zh-cn_topic_0000001391767420_section189014013619"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001663835704_zh-cn_topic_0000001391767420_section25791320141317"></a>

返回uint32\_t数据类型的结果，该结果代表当前系统workspace的大小，单位为字节。接口正常返回workspace大小，当返回`std::numeric_limits<uint32_t>::max()`（即`UINT32_MAX`）则接口异常。

## 约束说明<a name="zh-cn_topic_0000001663835704_zh-cn_topic_0000001391767420_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001663835704_zh-cn_topic_0000001391767420_section320753512363"></a>

### 自定义算子工程

```cpp
// 用户自定义的tiling函数
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddApiTiling tiling;
    ...
    size_t usrSize = 256; // 设置用户需要使用的workspace大小。
    // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
    auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (sysWorkspaceSize == std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize; // 设置总的workspace的数值大小，总的workspace空间由框架来申请并管理。
    ...
}
```

### Kernel直调

```cpp
uint32_t GetSysWorkspaceSize()
{
    auto* ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return std::numeric_limits<uint32_t>::max();
    }
    return ascendcPlatform->GetLibApiWorkSpaceSize();
}
```

- [Im2colBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/im2col_backward_950.md)
- [ChunkCat A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/chunk_cat.md)
- [Celu 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/celu_950.md)
- [FusedCausalConv1d 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/fused_causal_conv1d_950.md)
- [RoiPoolingWithArgMax 950 ACLNN 算子知识](../../../../operators/aclnn/cv/objdetect/roi_pooling_with_arg_max_950.md)
- [ChunkCat 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/chunk_cat_950.md)
- [ClippedSwiglu 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/clipped_swiglu_950.md)
- [NsaCompress A2 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/nsa_compress.md)
- [CircularPad3d A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_3d.md)
- [HardswishBackwardV2 A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/hardswish_backward_v2.md)
