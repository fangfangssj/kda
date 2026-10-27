---
type: API
title: Operator
description: 根据输入的offset偏移返回新的GlobalTensor，用于获取指定偏移量起始的子张量。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/GlobalTensor/operator.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
---

# operator\[\]<a name="ZH-CN_TOPIC_0000002132249364"></a>

## 功能说明<a name="section618mcpsimp"></a>

根据输入的offset偏移返回新的GlobalTensor。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline GlobalTensor operator[](const uint64_t offset) const
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="13.94%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.98%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.08%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p151479166327"><a name="p151479166327"></a><a name="p151479166327"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p126017529210"><a name="p126017529210"></a><a name="p126017529210"></a>偏移offset个元素。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

指定偏移量的GlobalTensor。

## 约束说明<a name="section633mcpsimp"></a>

无。

## 调用示例<a name="section17531157161314"></a>

```cpp
// operator[]使用方法, aGlobal[4]为从起始地址开始偏移量为4的新tensor
auto gmA = aGlobal[4];
// 示例结果如下：
// 输入数据(aGlobal): [1 2 3 4 5 6 ... 100]
// 输出数据(gmA ): [5 6 7 8 ... 100]
```
