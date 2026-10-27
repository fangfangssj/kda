---
type: API
title: DestroyWithoutPipeAll
description: 释放TPipe资源，用于重复申请释放TPipe。内部未设置PIPE_ALL同步，需开发者自行保证同步流水正确。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.memory_management
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/DestroyWithoutPipeAll.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tpipe.h
---

# DestroyWithoutPipeAll<a name="ZH-CN_TOPIC_0000002532992898"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_tpipe.h"`。

释放资源。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void DestroyWithoutPipeAll()
```

## 约束说明<a name="section633mcpsimp"></a>

用于重复申请释放TPipe，创建TPipe对象后，可调用DestroyWithoutPipeAll手动释放资源。

该接口内部未设置PIPE\_ALL同步，需要开发者自行保证同步流水正确。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```cpp
AscendC::TPipe pipe; // Pipe内存管理对象
AscendC::TQue<AscendC::TPosition::VECOUT, 2> que; //输出数据Queue队列管理对象，TPosition为VECOUT
uint8_t num = 2;
uint32_t len = 128;
pipe.InitBuffer(que, num, len);
pipe.DestroyWithoutPipeAll();
```
