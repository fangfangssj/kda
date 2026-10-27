---
type: API
title: 构造函数与析构函数
description: 用于创建KfcWorkspace对象，Kfc全称为kernel function call，表示核间通信调用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_group_mgmt_ISASI/KfcWorkspace/constructor_and_destructor.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
---

# 构造函数与析构函数<a name="ZH-CN_TOPIC_0000001968645092"></a>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

用于创建KfcWorkspace对象，Kfc全称为kernel function call，表示核间通信调用。

## 函数原型<a name="section765814724715"></a>

```
class KfcWorkspace;
__aicore__ inline KfcWorkspace(GM_ADDR workspace)
__aicore__ inline ~KfcWorkspace()
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

**表1**  KfcWorkspace构造函数参数说明

<a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2340183613156"></a>workspace</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="p19741912147"><a name="p19741912147"></a><a name="p19741912147"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a><a name="zh-cn_topic_0000001526206862_zh-cn_topic_0000001389783361_p2684123934216"></a>Global Memory上的消息空间地址，用户需保证地址对齐和清零。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

KfcWorkspace对象实例。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

不能和[REGIST\_MATMUL\_OBJ](../../../adv_api/cube_compute/matmul_kernel/regist_matmul_obj.md)接口同时使用。使用资源管理API时，用户自主管理AIC和AIV的核间通信，REGIST\_MATMUL\_OBJ内部是由框架管理AIC和AIV的核间通信，同时使用可能会导致通信消息错误等异常。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
AscendC::KfcWorkspace desc(workspaceGM);
```
