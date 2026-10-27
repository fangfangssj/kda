---
type: API
title: ResetCtrlSpr
description: 对CTRL寄存器的特定比特位做重置，恢复为默认值。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.system_variable
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/special_register_access/ResetCtrlSpr_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_common_intf.h
---

# ResetCtrlSpr(ISASI)

## 功能说明

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

对CTRL寄存器（控制寄存器）的特定比特位做重置。由模板参数\[startBit, endBit\]指定的比特位重置为默认值，默认值参考[表 常用CTRL寄存器比特位说明](set_ctrl_spr_isasi.md#tab-950)。

$\text{CTRL}[\,\text{startBit} : \text{endBit}\,] \leftarrow \text{defaultCtrl}[\,\text{startBit} : \text{endBit}\,]$

## 函数原型

```cpp
template <int8_t startBit, int8_t endBit>
__aicore__ static inline void ResetCtrlSpr()
```

## 参数说明

**表1** 模板参数说明

| 参数名 | 描述 |
| --- | --- |
| startBit | 起始比特位索引。 |
| endBit | 终止比特位索引。 |

## 数据类型

本接口为寄存器访问接口，不涉及张量数据类型操作。

## 返回值说明

无

## 约束说明

仅支持CTRL[8:6]、CTRL[10:9]、CTRL[48]、CTRL[50]、CTRL[53]、CTRL[59]、CTRL[60]比特位，各比特位的说明请参考[SetCtrlSpr(ISASI)参数说明](set_ctrl_spr_isasi.md#参数说明)。

## 调用示例

如下示例中重置CTRL[48]比特位，开启饱和模式。

```cpp
AscendC::SetCtrlSpr<48, 48>(1);
...
AscendC::ResetCtrlSpr<48, 48>();
```

更多示例请参考[CtrlSpr样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/09_utils/ctrl_spr)。
