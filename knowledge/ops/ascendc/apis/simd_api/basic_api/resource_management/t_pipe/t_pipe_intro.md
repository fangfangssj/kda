---
type: API
title: TPipe简介
description: TPipe用于统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象。其主要功能包括：
tags:
- task.lookup
- topic.implementation.memory_management
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TPipe/TPipe_intro.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-10T10:12:59Z'
platforms: []
---

# TPipe简介<a name="ZH-CN_TOPIC_0000002417498765"></a>

TPipe用于统一管理Device端内存等资源，一个核函数（Kernel）必须且只能初始化一个TPipe对象。其主要功能包括：

-   **内存资源管理**：通过TPipe的InitBuffer接口，可以为TQue和TBuf分配内存，分别用于队列的内存初始化和临时变量内存的初始化。
-   **同步事件管理**：通过TPipe的AllocEventID、ReleaseEventID等接口，可以申请和释放事件ID，用于同步控制。

## 配套 API 与常见组合

- [InitBuffer](init_buffer.md)——TPipe 的内存分配通过 TPipe::InitBuffer 完成，调用对象是 TPipe（pipe.InitBuffer），与资源池上的 TBufPool::InitBuffer 不是同一接口
- [GetSysWorkSpacePtr](../workspace/get_sys_work_space_ptr.md)——两者共同作为 REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling) 同一注册语句的必需参数，两张卡的调用示例均固定成对出现，P(a|b)=1.00
- [AllocTensor](../t_que/alloc_tensor.md)——AllocTensor 的内存源是 kernel 唯一 TPipe 经 InitBuffer 绑定的队列内存（P(a|b)=1.00），缺 TPipe 则队列范式不成立
- [FreeTensor](../t_que/free_tensor.md)——FreeTensor 作用的队列内存池由 kernel 唯一 TPipe 统一管理，同一内存生命周期链（P(a|b)=1.00）
- [TQue简介](../t_que/t_que_intro.md)——TQue 使用前需通过 InitBuffer 分配内存；可由 TPipe 或已初始化的 TBufPool 分配，具体约束见对应接口。
- [DeQue](../t_que/de_que.md)——DeQue 依赖的队列内存与同步由 kernel 唯一 TPipe 统一管理（P(a|b)=1.00）
- [EnQue](../t_que/en_que.md)——EnQue 依赖的队列内存与同步由 kernel 唯一 TPipe 统一管理（P(a|b)=1.00）
