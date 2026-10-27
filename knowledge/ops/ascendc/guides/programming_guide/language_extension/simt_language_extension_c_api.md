---
type: Guide
title: SIMT语言扩展层C API
description: SIMT编程基于AI Core的硬件能力实现，当前，SIMT语言扩展层支持的C API类别如下：
tags:
- task.lookup
- topic.programming_model.language_extension
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/programming_guide/language_extension/simt_language_extension_c_api.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms: []
---

# SIMT语言扩展层C API<a name="ZH-CN_TOPIC_0000002509743873"></a>

SIMT编程基于AI Core的硬件能力实现，当前，SIMT语言扩展层支持的C API类别如下：

-   [同步与内存栅栏](../../../apis/simt_api/sync_and_memory_fence/index.md)：提供同步与内存栅栏接口，解决线程间可能存在的数据竞争以及线程的同步问题。
-   [原子操作](../../../apis/simt_api/atomic_operations/index.md)：提供对Unified Buffer（UB）或Global Memory上的数据与指定数据执行原子操作的一系列API接口。
-   [Warp函数](../../../apis/simt_api/warp_functions/index.md)：提供对单个Warp内32个线程数据进行处理的一系列API接口。
-   [数学函数](../../../apis/simt_api/math_functions/index.md)：提供处理数学运算的函数接口集合。
-   [访存函数](../../../apis/simt_api/memory_access_functions/index.md)：提供开启Cache Hints的Load/Store函数。
-   [地址空间谓词函数](../../../apis/simt_api/address_space_predicate_functions/index.md)：提供判断输入指针所属地址空间的函数接口集合。
-   [地址空间转换函数](../../../apis/simt_api/address_space_conversion_functions/index.md)：提供不同地址空间地址值与指针之间相互转换的函数接口集合。
-   [协作组](../../../apis/simt_api/cooperative_groups/index.md)：提供更细粒度的线程间协作的API接口。
-   [调测接口](../../../apis/utils_api/tuning_interface/index.md)：SIMT调试场景下使用的相关接口。
