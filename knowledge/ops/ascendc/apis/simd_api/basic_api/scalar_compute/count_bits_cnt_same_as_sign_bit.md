---
type: API
title: CountBitsCntSameAsSignBit
description: 统计int64_t数值二进制表示中从最高数值位开始与符号位相同的连续比特位个数。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.scalar
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/scalar_compute/CountBitsCntSameAsSignBit.md
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

# CountBitsCntSameAsSignBit<a name="ZH-CN_TOPIC_0000001834069645"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_scalar_intf.h"`。

统计int64\_t类型数值的二进制表示中，从最高数值位开始与符号位相同的连续比特位的个数。

以如下CountBitsCntSameAsSignBit调用为例：

```cpp
int64_t valueOut = AscendC::CountBitsCntSameAsSignBit(0x0100000000000020ULL); // 返回6
```

其计算结果如下图所示：

**图1**  CountBitsCntSameAsSignBit计算示意图<a name="fig1"></a>

![CountBitsCntSameAsSignBit计算示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/CountBitsCntSameAsSignBit.png)

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline int64_t CountBitsCntSameAsSignBit(int64_t valueIn)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| valueIn | 输入 | 被统计的二进制数字。 |

## 返回值说明

返回从最高数值位开始与符号位相同的连续比特位的个数。

## 约束说明<a name="section633mcpsimp"></a>

当输入是-1（比特位全1）或者0（比特位全0）时，返回-1。

## 调用示例<a name="section837496171220"></a>

```cpp
int64_t valueOut;
int64_t valueIn = 0;                                    // 二进制表示全为0
valueOut = AscendC::CountBitsCntSameAsSignBit(valueIn); // 返回-1

valueIn = 0xFFFFFFFFFFFFFFFFLL;                         // -1,二进制表示全为1
valueOut = AscendC::CountBitsCntSameAsSignBit(valueIn); // 返回-1

valueIn = 0x7FFFFFFFFFFFFFFFLL;                         // 正数最大值
valueOut = AscendC::CountBitsCntSameAsSignBit(valueIn); // 返回0

valueIn = 0x8000000000000000LL;                         // 负数最小值
valueOut = AscendC::CountBitsCntSameAsSignBit(valueIn); // 返回0

valueIn = 1;
valueOut = AscendC::CountBitsCntSameAsSignBit(valueIn); // 返回62

valueIn = 0x00F0000000000000LL;
valueOut = AscendC::CountBitsCntSameAsSignBit(valueIn); // 返回7
```
