---
type: API
title: HCCL Context简介
description: 本章节的接口用于在kernel侧设置/获取通算融合算子每个通信域对应的context（消息区）地址。需要同步在host侧调用HcclGroup接口配置通信域名称后，才可以调用GetHcclContext获取对应的context地址。
tags:
- task.lookup
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL-Context/HCCL_Context_intro.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# HCCL Context简介

本章节的接口用于在核函数（Kernel）侧设置/获取通算融合算子每个通信域对应的context（消息区）地址。需要同步在host侧调用[HcclGroup](../../../../utils_api/prototype_register_management/op_mc2_def/hccl_group.md)接口配置通信域名称后，才可以调用[GetHcclContext](get_hccl_context.md)获取对应的context地址。

> [!NOTE]说明
>本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

