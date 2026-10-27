---
type: API
title: GetWorkspaceLen
description: 返回当前GroupBarrier所占用的Global Memory消息空间大小。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_group_mgmt_ISASI/GroupBarrier/GetWorkspaceLen.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
source_headers:
- include/basic_api/core_mng/roc/kernel_operator_group_barrier_intf.h
---

# GetWorkspaceLen<a name="ZH-CN_TOPIC_0000001968485296"></a>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

返回当前GroupBarrier所占用的Global Memory消息空间大小。

## 函数原型<a name="section765814724715"></a>

```
__aicore__ inline uint64_t GetWorkspaceLen()
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

无

## 返回值说明<a name="section39217325237"></a>

当前GroupBarrier所占用的Global Memory消息空间大小。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

无

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
// 6个AIV等3个AIV Arrive后再开始后续业务，总共需要6*512B地址空间，起始地址为用户指定的startAddr。
AscendC::GroupBarrier<AscendC::PipeMode::MTE3_MODE> barA(startAddr, 3, 6);
uint64_t offset = barA.GetWorkspaceLen(); // 返回barA所占用的GlobalMemory空间。
```

结果示例如下：

```
占用的GlobalMemory空间(offset)的大小为: 3072
```

