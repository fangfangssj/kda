---
type: API
title: SetBufferLen
description: 设置Buffer长度。当用户调用operator\[\]函数创建新LocalTensor时，建议调用该接口设置新LocalTensor长度，便于编译器对内存及同步进行自动优化。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/SetBufferLen.md
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
- include/basic_api/kernel_tensor.h
---

# SetBufferLen<a name="ZH-CN_TOPIC_0000002364805861"></a>

## 功能说明<a name="section618mcpsimp"></a>

设置Buffer的元素个数。当用户调用operator\[\]函数创建新LocalTensor时，建议调用该接口设置新LocalTensor的元素个数，便于编译器对内存及同步进行自动优化。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void SetBufferLen(uint32_t dataLen)
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p28044175211"><a name="p28044175211"></a><a name="p28044175211"></a>dataLen</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p199381338192512"><a name="p199381338192512"></a><a name="p199381338192512"></a>Buffer的元素个数。</p>
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
// 如下示例将申请的Tensor长度修改为1024个元素，float类型对应4096字节
AscendC::LocalTensor<float> tmpBuffer2 = tempBmm2Queue.AllocTensor<float>();
tmpBuffer2.SetBufferLen(1024);
```
