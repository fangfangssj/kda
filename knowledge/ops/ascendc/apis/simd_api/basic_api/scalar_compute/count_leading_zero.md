---
type: API
title: CountLeadingZero
description: 统计uint64_t数值二进制表示中前导0的个数，输入为0时返回64。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.scalar
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/scalar_compute/CountLeadingZero.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_scalar_intf.h
---

# CountLeadingZero<a name="ZH-CN_TOPIC_0000001834229569"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_scalar_intf.h"`。

统计uint64\_t类型数值的二进制表示中前导0的个数（二进制表示中最高有效位到第一个'1'的'0'的数量，输入为0时返回64）。

以如下CountLeadingZero调用为例：

```cpp
int64_t leadingZeros = AscendC::CountLeadingZero(0x0100000000000020ULL); // 返回7
```

其计算结果如下图所示：

**图1** CountLeadingZero计算示意图<a name="fig1"></a>

![CountLeadingZero计算示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/CountLeadingZero.png)

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline int64_t CountLeadingZero(uint64_t valueIn)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| valueIn | 输入 | 被统计的二进制数字。 |

## 返回值说明<a name="section640mcpsimp"></a>

返回valueIn的前导0的个数。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```cpp
int64_t leadingZeros;
uint64_t valueIn = 0x000000000000FFFFULL;          // 二进制表示有48个前导0
leadingZeros = AscendC::CountLeadingZero(valueIn); // 返回48

valueIn = 0;                                       // 二进制表示全为0
leadingZeros = AscendC::CountLeadingZero(valueIn); // 返回64

valueIn = 0xFFFFFFFFFFFFFFFFULL;                   // 二进制表示全为1
leadingZeros = AscendC::CountLeadingZero(valueIn); // 返回0
```
