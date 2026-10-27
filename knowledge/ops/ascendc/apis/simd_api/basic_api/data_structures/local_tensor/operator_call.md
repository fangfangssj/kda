---
type: API
title: operator
description: 获取本LocalTensor的第offset个变量的引用。用于左值，相当于SetValue接口，用于右值，相当于GetValue接口。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/operator_call.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tensor.h
---

# operator\(\)<a name="ZH-CN_TOPIC_0000002330767546"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取本LocalTensor的第offset个变量的引用。用于左值，相当于[SetValue](set_value.md)接口，用于右值，相当于[GetValue](get_value.md)接口。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline __inout_pipe__(S) __ubuf__ PrimType& operator()(const uint32_t offset) const
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p16457182213494"><a name="p16457182213494"></a><a name="p16457182213494"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p1585316254266"><a name="p1585316254266"></a><a name="p1585316254266"></a>LocalTensor下标索引。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

返回指定索引位置的元素的PrimType类型引用。

PrimType定义如下：

```cpp
// PrimT用于从T中提取基础数据类型：T传入基础数据类型，直接返回数据类型；T传入为TensorTrait类型时萃取TensorTrait中的LiteType基础数据类型
using PrimType = PrimT<T>;
```

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// operator()使用方法,用于左值，相当于Setvalue接口，用于右值，相当于Getvalue接口
// 输入数据(inputLocal): [100 100 100 ... 100]
inputLocal(2) = 200;
// 输出数据(inputLocal): [100 100 200 ... 100]
auto x = inputLocal(2);
// x显示为200
```
