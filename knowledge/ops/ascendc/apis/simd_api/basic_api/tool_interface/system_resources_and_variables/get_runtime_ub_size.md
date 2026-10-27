---
type: API
title: GetRuntimeUBSize
description: 获取运行时UB空间的大小，单位为Byte。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetRuntimeUBSize.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetRuntimeUBSize

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

获取运行时Unified Buffer（UB）空间的大小，单位为Byte。开发者根据UB的大小来计算循环次数等参数值。

## 函数原型

```cpp
__aicore__ inline uint32_t GetRuntimeUBSize()
```

## 参数说明

无

## 返回值说明

运行时UB空间的大小，单位为Byte。


Ascend 950PR&950DT系列产品架构下，[SIMD与SIMT混合场景](../../../../simt_api/simd_simt_hybrid_programming_intro/index.md)中，获取到的UB大小上限为216KB，非SIMD与SIMT混合场景返回值为固定值248KB。


## 约束说明

无

## 调用示例

本调用示例通过`GetRuntimeUBSize`获取的UB空间大小，来计算tileNum的值。完整的算子样例参考：[get\_ub\_size样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/09_utils/get_ub_size)。

```cpp
// totalLength为待处理的总数据长度(元素个数)
this->totalLength = totalLength;
// GetRuntimeUBSize() / sizeof(half) -> 计算UB能容纳多少个half类型元素
// 除2 -> 预留50%的UB空间
if (totalLength > AscendC::GetRuntimeUBSize() / sizeof(half) / 2) {
    this->tileLength = AscendC::GetRuntimeUBSize() / sizeof(half) / 2;
} else { // 防止分片大小超过实际数据总量
    this->tileLength = this->totalLength;
}
// 需要迭代的分片数量
this->tileNum = this->totalLength / this->tileLength;
```
