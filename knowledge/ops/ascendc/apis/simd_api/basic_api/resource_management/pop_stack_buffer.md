---
type: API
title: PopStackBuffer
description: 在指定position（逻辑位置）申请临时空间，空间大小为指定position的全部剩余空间。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/PopStackBuffer.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
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

# PopStackBuffer<a name="ZH-CN_TOPIC_0000001870331257"></a>

## 功能说明<a name="section618mcpsimp"></a>

在指定position（逻辑位置）申请临时空间，空间大小为指定position的全部剩余空间。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, TPosition pos>
__aicore__ inline bool PopStackBuffer(LocalTensor<T>& popLocal)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="20.61%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="79.39%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.3.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="79.39%" headers="mcps1.2.3.1.2 "><p id="p10931155464814"><a name="p10931155464814"></a><a name="p10931155464814"></a>popLocal的数据类型，支持的数据类型如下：uint8_t、int8_t、int16_t、uint16_t、int32_t、uint32_t、int64_t、uint64_t、float、half。</p>
</td>
</tr>
<tr id="row860118831216"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.3.1.1 "><p id="p1860214819121"><a name="p1860214819121"></a><a name="p1860214819121"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="79.39%" headers="mcps1.2.3.1.2 "><p id="p20602138111212"><a name="p20602138111212"></a><a name="p20602138111212"></a>需要申请临时空间的position，数据类型为<a href="../aux_data_structures/t_position.md">TPosition</a>。</p>
</td>
</tr>
</tbody>
</table>

**表2**  参数说明

<a name="table628324453115"></a>
<table><thead align="left"><tr id="row7283174414314"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p32831448310"><a name="p32831448310"></a><a name="p32831448310"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p128394414316"><a name="p128394414316"></a><a name="p128394414316"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p42831044103118"><a name="p42831044103118"></a><a name="p42831044103118"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1283164418312"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p20283154433118"><a name="p20283154433118"></a><a name="p20283154433118"></a>popLocal</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p18283204413317"><a name="p18283204413317"></a><a name="p18283204413317"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6283154420313"><a name="p6283154420313"></a><a name="p6283154420313"></a>申请临时空间对应的Tensor，Tensor大小为对应position的剩余全部空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

表示函数执行的结果，true表示成功，false表示失败。

## 约束说明<a name="section633mcpsimp"></a>

-   该接口不支持嵌套使用，比如函数A中调用了PopStackBuffer，那么调用函数A的其他函数中则不可以再调用PopStackBuffer。
-   因为当前高阶API内部实现中会使用到本接口，所以算子实现中调用了高阶API的场景，不支持调用该接口。

## 调用示例<a name="section837496171220"></a>

```
AscendC::LocalTensor<int16_t> popBuffer;
bool ret = AscendC::PopStackBuffer<int16_t, AscendC::TPosition::VECCALC>(popBuffer);
```

- [DiagFlat A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/diag_flat.md)
