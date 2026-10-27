---
type: API
title: DynamicCompileStaticFlag
description: 用于标识该算子实现是否支持入图时的静态Shape编译。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
- topic.implementation.operator_registration
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/prototype_register_management/OpAICoreConfig/DynamicCompileStaticFlag.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# DynamicCompileStaticFlag<a name="ZH-CN_TOPIC_0000002078497998"></a>

## 功能说明<a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section36583473819"></a>

用于标识该算子实现是否支持入图时的静态Shape编译。

## 函数原型<a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section13230182415108"></a>

```
OpAICoreConfig &DynamicCompileStaticFlag(bool flag)
```

## 参数说明<a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section75395119104"></a>

<a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p318615392613"></a>flag</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a><a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001575944081_p320343694214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="p460320016426"><a name="p460320016426"></a><a name="p460320016426"></a>用户开发的自定义算子，如果需要支持入图时静态Shape场景下的编译，需要配置该选项为true，否则配置为false。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section25791320141317"></a>

OpAICoreConfig类，请参考[OpAICoreConfig](index.md)。

## 约束说明<a name="zh-cn_topic_0000001575931912_zh-cn_topic_0000001576875005_zh-cn_topic_0000001525424352_section19165124931511"></a>

无

- [ForeachSigmoid 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_sigmoid_950.md)
