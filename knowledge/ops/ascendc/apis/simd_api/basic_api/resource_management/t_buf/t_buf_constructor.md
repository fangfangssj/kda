---
type: API
title: TBufConstructor
description: 创建TBuf对象时，初始化数据成员。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TBuf/TBuf_constructor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_tpipe.h
---

# TBuf构造函数<a name="ZH-CN_TOPIC_0000002055722337"></a>

## 功能说明<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section36583473819"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

创建TBuf对象时，初始化数据成员。

## 函数原型<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section13230182415108"></a>

```cpp
template <TPosition pos = TPosition::LCM>
__aicore__ inline TBuf();
```

## 参数说明<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section75395119104"></a>

**表1**  模板参数说明

<a name="table473143421713"></a>
<table><thead align="left"><tr id="row117317341170"><th class="cellrowborder" valign="top" width="14.760000000000002%" id="mcps1.2.3.1.1"><p id="p19731113414177"><a name="p19731113414177"></a><a name="p19731113414177"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85.24000000000001%" id="mcps1.2.3.1.2"><p id="p4731153431717"><a name="p4731153431717"></a><a name="p4731153431717"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row16731173415179"><td class="cellrowborder" valign="top" width="14.760000000000002%" headers="mcps1.2.3.1.1 "><p id="p185019592913"><a name="p185019592913"></a><a name="p185019592913"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="85.24000000000001%" headers="mcps1.2.3.1.2 "><p id="p35011591693"><a name="p35011591693"></a><a name="p35011591693"></a>TBuf所在的逻辑位置，支持的TPosition可以为VECIN、VECOUT、A1、A2、B1、B2、CO1、CO2。<span>关于TPosition的具体介绍请参考</span><a href="../../aux_data_structures/t_position.md">TPosition</a>。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001935531088_zh-cn_topic_0000001339105488_section19165124931511"></a>

无。
