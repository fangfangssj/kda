---
type: API
title: Print（数据结构）
description: 只限于CPU调试，在调试窗口中打印LocalTensor数据用于精度调试，每一行打印一个DataBlock（32字节）的数据。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/Print.md
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

# Print<a name="ZH-CN_TOPIC_0000002330927402"></a>

## 功能说明<a name="section618mcpsimp"></a>

只限于CPU调试，在调试窗口中打印LocalTensor数据用于精度调试，每一行打印一个DataBlock（32字节）的数据。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
inline void Print()
inline void Print(uint32_t len)
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p2806181517286"><a name="p2806181517286"></a><a name="p2806181517286"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p1958416205288"><a name="p1958416205288"></a><a name="p1958416205288"></a>打印元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// 示例只限于CPU调试，在调试窗口中打印LocalTensor数据用于精度调试，每一行打印一个datablock(32Bytes)的数据
AscendC::LocalTensor<int32_t> inputLocal = softmaxMaxBuf.template Get<int32_t>();
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对input_local中第i个位置进行赋值为i
}
inputLocal.Print();
// 0000: 0 1 2 3 4 5 6 7 8
// 0008: 9 10 11 12 13 14 15
```
