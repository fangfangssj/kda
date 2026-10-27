---
type: API
title: GlobalTensor构造函数
description: GlobalTensor构造函数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/GlobalTensor/GlobalTensor_constructor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
---

# GlobalTensor构造函数<a name="ZH-CN_TOPIC_0000002167642833"></a>

## 功能说明<a name="section618mcpsimp"></a>

GlobalTensor构造函数。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline GlobalTensor<T>() {}
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.28%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.72%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>GlobalTensor的数据类型。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无。

## 约束说明<a name="section633mcpsimp"></a>

无。

## 调用示例<a name="section17531157161314"></a>

参考[调用示例](set_global_buffer.md#section17531157161314)。

- [QuantBatchMatmulInplaceAdd 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/quant_batch_matmul_inplace_add_950.md)
