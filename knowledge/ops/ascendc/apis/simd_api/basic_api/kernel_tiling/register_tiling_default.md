---
type: API
title: REGISTER_TILING_DEFAULT
description: 用于在kernel侧注册用户使用标准C++语法自定义的默认TilingData结构体。 注册TilingData结构体用于告知框架侧用户使用标准C++语法来定义TilingData，同时告知框架TilingData结构体类型，用于框架做tiling数据解析。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.tiling
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/Kernel-Tiling/REGISTER_TILING_DEFAULT.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/utils/base/sys_macros.h
---

# REGISTER\_TILING\_DEFAULT<a name="ZH-CN_TOPIC_0000002123262037"></a>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

头文件路径为：`"utils/base/sys_macros.h"`。

用于在核函数（Kernel）侧注册用户使用标准C++语法自定义的默认TilingData结构体。

注册TilingData结构体用于告知框架侧用户使用标准C++语法来定义TilingData，同时告知框架TilingData结构体类型，用于框架做tiling数据解析。

## 函数原型<a name="zh-cn_topic_0000001526206862_section1630753514297"></a>

```
REGISTER_TILING_DEFAULT(TILING_STRUCT)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.32%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.46%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="p18315519741"><a name="p18315519741"></a><a name="p18315519741"></a><span>TILING_STRUCT</span></p>
</td>
<td class="cellrowborder" valign="top" width="15.32%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p320343694214"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p320343694214"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.46%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a><span>用户注册的默认自定义</span><span>TilingData结构体</span>。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

-   若TilingData结构体在命名空间内，注册时需要携带对应的命名空间作用域符。
-   暂不支持核函数（Kernel）直调工程。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::TilingData); // 用于在核函数（Kernel）侧注册用户使用标准C++语法自定义的默认TilingData结构体
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.blkDim, tilingData.totalSize, tilingData.splitTile);
    op.Process();
}
```

- [FastGelu 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/fast_gelu_950.md)
- [Im2colBackward 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/im2col_backward_950.md)
- [CircularPad2dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d_backward_950.md)
- [ClippedSwiglu 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/clipped_swiglu_950.md)
- [ChunkGatedDeltaRule A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/chunk_gated_delta_rule.md)
- [UpsampleBicubic2dAA 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_950.md)
- [ReflectionPad2dBackward 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/reflection_pad_2d_backward_950.md)
- [HardswishBackward 950 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/hardswish_backward_950.md)
- [FlashAttentionScore 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/flash_attention_score_950.md)
- [UpsampleBicubic2dAAGrad 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/upsample_bicubic_2d_aa_grad_950.md)
- [All 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/all_950.md)

## 配套 API 与常见组合

- [GET_TILING_DATA_WITH_STRUCT](get_tiling_data_with_struct.md)——注册告知框架结构体类型供解析，GET_TILING_DATA_WITH_STRUCT 按同名结构体解析，类型必须一致（证据同名同文件）
- [GET_TPL_TILING_KEY](../../../utils_api/tiling_template_programming/get_tpl_tiling_key.md)——Tiling 模板编程流程中 kernel 侧注册默认 tiling 结构体、host 侧用 GET_TPL_TILING_KEY 生成模板参数 TilingKey，缺一 TPL 不成立
- [模板参数定义](../../../utils_api/tiling_template_programming/template_params_definition.md)——模板参数组合的 tiling 结构体兜底"必须通过 REGISTER_TILING_DEFAULT 提供默认结构体"（模板参数定义卡明文）
