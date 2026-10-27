---
type: API
title: SetQuit
description: 通过AllocMessage接口获取到消息空间地址后，发送退出消息，告知该消息队列对应的AIC无需处理该队列的消息。如下图，Queue5对应的AIV发了退出消息后，Block1将不再处理Queue5的任何消息。
  图1 消息队列退出示意图 !
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_group_mgmt_ISASI/CubeResGroupHandle/SetQuit.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
source_headers:
- include/basic_api/core_mng/roc/kernel_operator_cube_group_intf.h
---

# SetQuit<a name="ZH-CN_TOPIC_0000001996865757"></a>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

通过AllocMessage接口获取到消息空间地址后，发送退出消息，告知该消息队列对应的AIC无需处理该队列的消息。如下图，Queue5对应的AIV发了退出消息后，Block1将不再处理Queue5的任何消息。

**图1**  消息队列退出示意图<a name="fig915491163918"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/message_queue_exit.png "消息队列退出示意图")

## 函数原型<a name="section765814724715"></a>

```
__aicore__ inline void SetQuit(__gm__ CubeMsgType* msg)
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

**表1**  接口参数说明

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="9.09%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="9.01%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="81.89999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.2.4.1.1 "><p id="p153103773911"><a name="p153103773911"></a><a name="p153103773911"></a>msg</p>
</td>
<td class="cellrowborder" valign="top" width="9.01%" headers="mcps1.2.4.1.2 "><p id="p1831010712394"><a name="p1831010712394"></a><a name="p1831010712394"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="81.89999999999999%" headers="mcps1.2.4.1.3 "><p id="p203101270393"><a name="p203101270393"></a><a name="p203101270393"></a>该CubeResGroupHandle中的消息空间地址。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

无

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
handle.AssignQueue(queIdx);  
auto msgPtr = a.AllocMessage();        // 获取消息空间指针msgPtr
handle.SetQuit(msgPtr);              // 发送退出消息
```
