---
type: API
title: asc_set_l12l0_padding_val（废弃）
description: asc_set_l12l0_padding_val接口已废弃。请使用asc_set_l12l0b_3d_padding接口替代。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/deprecated_interface/asc_set_l12l0_padding_val_deprecated.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/c_api/cube_datamove/cube_datamove.h
---

# asc_set_l12l0_padding_val（废弃）

## 功能说明

**`asc_set_l12l0_padding_val`接口已废弃。请使用[asc_set_l12l0b_3d_padding](../cube_datamove/cube_load_aux_config/asc_set_l12l0b_3d_padding.md)接口替代。**

头文件路径为：`"c_api/cube_datamove/cube_datamove.h"`。

对PADDING_B寄存器中的值进行设置，这是一个64bit的寄存器，用于存储搬运过程中padding的值。
其中各bit含义如下：

**表1** PADDING_B寄存器比特位说明

| bit范围 |                                                                           含义                                                                           |
| :-----------|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 31:0 | 存储进行padding时的值。<br>数据位宽为32时，直接使用31:0位；<br> 数据位宽为16时，使用15:0位，31:16位被忽略； <br> 数据位宽为8时，15:8位应与7:0位相同，31:16位被忽略；数据位宽为4时，15:12，11:8，7:4和3:0位应该相同，31:16位被忽略。 |
| 33:32 | 表示所使用的padding模式，当前支持三种模式。<br>2'b00：固定padding；<br>2'b01：按通道padding；<br>2'b10/2'b11：当前保留。 |

## 函数原型

```cpp
__aicore__ inline void asc_set_l12l0_padding_val(uint64_t config)
```

## 参数说明

**表2** 参数说明

|参数名|输入/输出| 描述        |
| :------ | :---  |:----------|
|config   |输入   | 待设置的寄存器值。 |

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
uint64_t config = 0;
asc_set_l12l0_padding_val(config);
```
