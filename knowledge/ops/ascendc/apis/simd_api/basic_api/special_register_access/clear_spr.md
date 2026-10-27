---
type: API
title: ClearSpr
description: 对指定的特殊寄存器进行清零操作。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.system_variable
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/special_register_access/ClearSpr.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# ClearSpr<a name="ZH-CN_TOPIC_0000002537397560"></a>

## 功能说明<a name="section11971757181915"></a>

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

对指定的特殊寄存器进行清零。当前支持[表SpecialPurposeReg模板参数说明](get_spr.md#table37531617424)。

## 函数原型<a name="section113251712205"></a>

```cpp
template <SpecialPurposeReg spr>
__aicore__ inline void ClearSpr()
```

## 参数说明<a name="section1116018170208"></a>

**表1** 模板参数说明

| 参数名 | 描述 |
|--------|------|
| spr | 特殊寄存器，类型为SpecialPurposeReg枚举类，具体的取值请参考[表SpecialPurposeReg模板参数说明](get_spr.md#table37531617424)。 |

## 数据类型<a name="section3853852112218"></a>

无

## 返回值说明<a name="section16895132314202"></a>

无

## 约束说明<a name="section162221734202016"></a>

本接口只能在VF函数外调用，命名空间为AscendC，函数标记符为\_\_aicore\_\_。如果需要在VF内调用，命名空间为AscendC::Reg，函数标记符为\_\_simd\_callee\_\_，具体请参考[ClearSpr](../reg_vector_compute/system_variable_access/clear_spr.md)。

## 调用示例<a name="section849174212202"></a>

无
