---
type: API
title: TILING_KEY_IS
description: 在核函数中判断本次执行时的tiling_key是否等于host侧运行时设置的某个key，从而标识tiling_key==key的一条kernel分支。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.tiling
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/Kernel-Tiling/TILING_KEY_IS.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/utils/base/sys_macros.h
---

# TILING\_KEY\_IS<a name="ZH-CN_TOPIC_0000001586995454"></a>

## 功能说明<a name="zh-cn_topic_0000001610027821_section212607105720"></a>

头文件路径为：`"utils/base/sys_macros.h"`。

在核函数（Kernel）中判断本次执行时的tiling\_key是否等于host侧运行时设置的某个key，从而标识tiling\_key==key的一条核函数（Kernel）分支。

## 函数原型<a name="zh-cn_topic_0000001610027821_section1630753514297"></a>

```
TILING_KEY_IS(key)
```

## 参数说明<a name="zh-cn_topic_0000001610027821_section129451113125413"></a>

<a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p2340183613156"><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p2340183613156"></a><a name="zh-cn_topic_0000001610027821_zh-cn_topic_0000001389783361_p2340183613156"></a>key</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001610027821_p18857658124919"><a name="zh-cn_topic_0000001610027821_p18857658124919"></a><a name="zh-cn_topic_0000001610027821_p18857658124919"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001610027821_p53562135013"><a name="zh-cn_topic_0000001610027821_p53562135013"></a><a name="zh-cn_topic_0000001610027821_p53562135013"></a>key表示某个核函数（Kernel）的分支，必须是非负整数。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001610027821_section65498832"></a>

-   TILING\_KEY\_IS运用于if和else if分支，不支持else分支，即用TILING\_KEY\_IS函数来表征N个分支，必须用N个TILING\_KEY\_IS\(key\)来分别表示。
-   暂不支持核函数（Kernel）直调工程。

## 调用示例<a name="zh-cn_topic_0000001610027821_section97001499599"></a>

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    KernelAdd op;
    op.Init(x, y, z, tilingData.numBlocks, tilingData.totalLength, tilingData.tileNum);
    // 当TilingKey为1时，执行Process1；为2时，执行Process2；为3时，执行Process3
    if (TILING_KEY_IS(1)) {
        op.Process1();
    } else if (TILING_KEY_IS(2)) {
        op.Process2();
    } else if (TILING_KEY_IS(3)) {
        op.Process3();
    }
    // 其他代码逻辑
    ...
    // 此处示例当TilingKey为3时，会执行ProcessOther
    if (TILING_KEY_IS(3)) {
        op.ProcessOther();
    }
}
```

配套的host侧tiling函数示例（伪代码）：

```
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 其他代码逻辑
    ...
    if (context->GetInputShape(0) > 10) {
        context->SetTilingKey(1);
    } else if (some condition) {
        context->SetTilingKey(2);
    } else if (some condition) {
        context->SetTilingKey(3);
    }
}
```

- [Resize 950 ACLNN 算子知识](../../../../operators/aclnn/cv/image/resize_950.md)
- [CircularPad2d A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d.md)
- [ClippedSwiglu A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/clipped_swiglu.md)
- [NsaCompressAttentionInfer A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/nsa_compress_attention_infer.md)
- [ThreeInterpolateBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/cv/image/three_interpolate_backward.md)
- [CircularPad2dBackward A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_2d_backward.md)
- [FatreluMul A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/activation/fatrelu_mul.md)
- [NsaSelectedAttention A2 ACLNN 算子知识](../../../../operators/aclnn/transformer/attention/nsa_selected_attention.md)
- [NMS 950 Direct Invoke 算子知识](../../../../operators/direct/cv/image/nms_950.md)
- [CircularPad3d A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/circular_pad_3d.md)
