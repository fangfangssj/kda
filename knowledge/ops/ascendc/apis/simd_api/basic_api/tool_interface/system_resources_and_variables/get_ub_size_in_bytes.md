---
type: API
title: GetUBSizeInBytes
description: 获取UB空间的大小，单位为Byte。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/tool_interface/system_resources_and_variables/GetUBSizeInBytes.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/basic_api/kernel_operator_sys_var_intf.h
---

# GetUBSizeInBytes

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

获取Unified Buffer（UB）空间的大小，单位为Byte。开发者根据UB的大小来计算循环次数等参数值。

## 函数原型

```cpp
__aicore__ inline constexpr uint32_t GetUBSizeInBytes()
```

## 参数说明

无

## 返回值说明

UB空间的大小。


Ascend 950PR&950DT系列产品架构下，返回常量248KB，单位为Byte。


## 约束说明

无

## 调用示例

本调用示例通过GetUBSizeInBytes获取的UB空间大小，来计算tileNum的值。完整的算子样例请参考：[get\_ub\_size样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/09_utils/get_ub_size)。

```cpp
// totalLength为待处理的总数据长度(元素个数)
this->totalLength = totalLength;
// GetUBSizeInBytes() / sizeof(half) -> 计算UB能容纳多少个half类型元素
// 除2 -> 预留50%的UB空间
if (totalLength > AscendC::GetUBSizeInBytes() / sizeof(half) / 2) {
    this->tileLength = AscendC::GetUBSizeInBytes() / sizeof(half) / 2;
} else { // 防止分片大小超过实际数据总量
    this->tileLength = this->totalLength;
}
// 需要迭代的分片数量
this->tileNum = this->totalLength / this->tileLength;
```

- [GatherV2 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/gather_v2_950.md)
- [Embedding 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/embedding_950.md)
- [Take 950 ACLNN 算子知识](../../../../../operators/aclnn/nn/index/take_950.md)
