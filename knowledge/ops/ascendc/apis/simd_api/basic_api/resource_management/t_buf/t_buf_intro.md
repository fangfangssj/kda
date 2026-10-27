---
type: API
title: TBuf简介
description: 使用Ascend C编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的TPosition逻辑位置。
tags:
- task.lookup
- topic.implementation.memory_management
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/resource_management/TBuf/TBuf_intro.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# TBuf简介<a name="ZH-CN_TOPIC_0000001645413417"></a>

使用Ascend C编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的[TPosition](../../aux_data_structures/t_position.md)逻辑位置。

TBuf继承自TQueBind父类，继承关系如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/enque-42.png)

TBuf占用的存储空间通过TPipe进行管理，您可以通过[InitBuffer](../t_pipe/init_buffer.md)接口为TBuf进行内存初始化操作，之后即可通过[Get](get.md)获取指定长度的Tensor参与计算。

使用[InitBuffer](../t_pipe/init_buffer.md)为TBuf分配内存和为队列分配内存有以下差异：

-   为TBuf分配的内存空间只能参与计算，无法执行队列的入队出队操作。
-   调用一次内存初始化接口，TPipe只会为TBuf分配一块内存，为队列可以通过参数设置申请多块内存。如果要使用多个临时变量，需要定义多个TBuf数据结构，对每个TBuf数据结构分别调用[InitBuffer](../t_pipe/init_buffer.md)接口进行内存初始化。
-   TBuf获取的Tensor无需释放。
