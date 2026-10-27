---
type: API
title: LocalMemAllocatorConstructor
description: LocalMemAllocator构造函数，用于初始化指定物理位置的内存分配器对象。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/LocalMemAllocator/LocalMemAllocator_constructor.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-10T04:14:14Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_tensor.h
---

# LocalMemAllocator构造函数<a name="ZH-CN_TOPIC_0000002337736701"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tensor.h"`。

LocalMemAllocator构造函数。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <Hardware hard>
__aicore__ inline LocalMemAllocator<hard>::LocalMemAllocator()
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="27.839999999999996%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="72.16%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row18835145716587"><td class="cellrowborder" valign="top" width="27.839999999999996%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>hard</p>
</td>
<td class="cellrowborder" valign="top" width="72.16%" headers="mcps1.2.3.1.2 "><p id="p823866165711"><a name="p823866165711"></a><a name="p823866165711"></a><span id="ph184621011705"><a name="ph184621011705"></a><a name="ph184621011705"></a><span id="ph969812395911"><a name="ph969812395911"></a><a name="ph969812395911"></a>用于表示数据的物理位置，Hardware枚举类型，定义如下，合法位置为：Unified Buffer（UB）、L1、L0A、L0B、L0C、BIAS、FIXBUF。</span></span></p>
<a name="screen21992320439"></a><a name="screen21992320439"></a><pre class="screen" codetype="Cpp" id="screen21992320439">enum class Hardware : uint8_t { 
GM,     // Global Memory
UB,     // UB
L1,     // L1 Buffer
L0A,    // L0A Buffer
L0B,    // L0B Buffer
L0C,    // L0C Buffer
BIAS,   // BiasTable Buffer
FIXBUF, // Fixpipe Buffer
MAX };</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

同一个物理位置的LocalMemAllocator对象，在算子生命周期内只能存在1个。

## 配套 API 与常见组合

- [Alloc](alloc.md)——Alloc 是 LocalMemAllocator 的成员接口且 pos 须匹配构造时指定的 Hardware 物理位置（Alloc 卡明文），P 双向 1.00
- [InitSocState](../../tool_interface/system_init/init_soc_state.md)——构造 LocalMemAllocator 即进入静态Tensor编程，核入口必须 InitSocState，P(b|a)=1.00 全量成立
