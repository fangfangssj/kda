---
type: API
title: PRELOAD
description: 头文件路径为："utils/base/sys_macros.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cache_control/PRELOAD.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/utils/base/sys_macros.h
---

# PRELOAD

## 功能说明

头文件路径为：`"utils/base/sys_macros.h"`。

从当前指令所在的GM地址开始，将后续指令预加载到ICache中。

## 函数原型

```cpp
#define PRELOAD(len)
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| len | 输入 | 预取长度。 |


针对如下型号：

- Ascend 950PR&950DT系列产品


- Atlas A3系列产品


- Atlas A2系列产品


`len`的单位为2K Byte，取值应小于ICache大小除以2K。其中，AIC和AIV的ICache大小分别为32KB和16KB。



- 针对Atlas推理系列产品AI Core，`len`不控制预取长度，预取长度固定为128Byte。


## 返回值说明

无

## 约束说明

仅支持在核函数（Kernel）侧调用。

## 调用示例

```cpp
// 从当前指令地址开始预加载4个单位的指令。
PRELOAD(4);
```
