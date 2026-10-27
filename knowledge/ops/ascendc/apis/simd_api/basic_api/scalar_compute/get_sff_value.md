---
type: API
title: GetSFFValue
description: 查找uint64_t数值二进制表示中从最低有效位开始的首个指定比特值位置，未找到则返回-1。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.scalar
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/scalar_compute/GetSFFValue.md
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

# GetSFFValue<a name="ZH-CN_TOPIC_0000001787630014"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_scalar_intf.h"`。

查找uint64\_t类型数值的二进制表示中从最低有效位开始的首个指定比特值（0或1）的位置，如果没找到则返回-1。

以如下GetSFFValue调用为例：

```cpp
int64_t sffValue = AscendC::GetSFFValue<1>(0x1000000000000100ULL); // 返回8
```

其计算结果如下图所示：

**图1** 使用GetSFFValue查找从最低有效位开始首个1的位置<a name="fig1"></a>

![使用GetSFFValue查找从最低有效位开始首个1的位置](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/GetSFFValue.png "使用GetSFFValue查找从最低有效位开始首个1的位置")

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <int countValue>
__aicore__ inline int64_t GetSFFValue(uint64_t valueIn)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
| ------ | ------ |
| countValue | 指定要查找的比特值，取值为0或1。 |

**表2** 参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| valueIn | 输入 | 被查找的二进制数字。 |

## 返回值说明<a name="section640mcpsimp"></a>

返回valueIn中第一个0或1出现的位置。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```cpp
int64_t sffValue;
uint64_t valueIn = 0xFFFFFFFFFFFFFFFFULL;    // 二进制表示全为1
sffValue = AscendC::GetSFFValue<0>(valueIn); // 返回-1

valueIn = 0;                                 // 二进制表示全为0
sffValue = AscendC::GetSFFValue<1>(valueIn); // 返回-1

valueIn = 0xFF00ULL;
sffValue = AscendC::GetSFFValue<0>(valueIn); // 返回0
sffValue = AscendC::GetSFFValue<1>(valueIn); // 返回8

valueIn = 0x00FFULL;
sffValue = AscendC::GetSFFValue<0>(valueIn); // 返回8
sffValue = AscendC::GetSFFValue<1>(valueIn); // 返回0
```
