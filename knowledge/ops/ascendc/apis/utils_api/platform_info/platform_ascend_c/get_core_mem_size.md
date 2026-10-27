---
type: API
title: GetCoreMemSize
description: 获取硬件平台存储空间的内存大小，例如L1、L0_A、L0_B、L2等，支持的存储空间类型定义如下：
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.programming_model.system_variable
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/platform_info/PlatformAscendC/GetCoreMemSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
source_headers:
- include/utils/tiling/platform/platform_ascendc.h
---

# GetCoreMemSize<a name="ZH-CN_TOPIC_0000002114052981"></a>

## 功能说明<a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_section36583473819"></a>

获取硬件平台存储空间的内存大小，例如L1、L0\_A、L0\_B、L2等，支持的存储空间类型定义如下：

```
enum class CoreMemType {
L0_A = 0, // L0A Buffer
L0_B = 1, // L0B Buffer
L0_C = 2, // L0C Buffer
L1 = 3,   // L1 Buffer
L2 = 4,   // L2 Cache
UB = 5,   // Unified Buffer（UB）
HBM = 6,  // GM
FB = 7,   // Fixpipe Buffer
BT = 8,   // BiasTable Buffer
RESERVED
};
```

## 函数原型<a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_section13230182415108"></a>

```
void GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const
```

## 参数说明<a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_section189014013619"></a>

<a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p10223674448"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p10223674448"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p645511218169"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p645511218169"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p1922337124411"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p1922337124411"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p17770136956"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p17770136956"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p17770136956"></a>memType</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p167701536957"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p167701536957"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p167701536957"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p4611154016587"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p4611154016587"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p4611154016587"></a>硬件存储空间类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_row19496552114312"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p124961352184311"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p124961352184311"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p124961352184311"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p1496852144310"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p1496852144310"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p1496852144310"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p94961752154312"><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p94961752154312"></a><a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_p94961752154312"></a>对应类型的存储空间大小，单位：字节。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001597362348_zh-cn_topic_0000001442486577_section320753512363"></a>

### 自定义算子工程

```cpp
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ub_size, l1_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
    // ...
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
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // ... 使用ubSize
}
```

- [DequantBias A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/quant/dequant_bias.md)
- [Sigmoid 950 Direct Invoke 算子知识](../../../../operators/direct/nn/activation/sigmoid_950.md)
- [Threshold 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/threshold_950.md)
- [FfnToAttention A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/ffn_to_attention.md)
- [InplaceWeightQuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/inplace_weight_quant_matmul_all_reduce_add_rms_norm.md)
- [QuantMatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/quant_matmul_all_reduce_add_rms_norm.md)
- [MhcSinkhorn 950 Direct Invoke 算子知识](../../../../operators/direct/transformer/mhc/mhc_sinkhorn_950.md)
