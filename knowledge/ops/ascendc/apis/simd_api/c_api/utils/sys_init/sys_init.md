---
type: API
title: 系统初始化
description: 系统初始化用于在算子开始执行时，将NPU中可能跨任务保留的控制状态恢复为C API约定的默认值。建议在Kernel入口、数据搬运和计算之前调用asc_init，避免前序任务遗留的原子操作、掩码或搬运循环配置影响当前任务。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/utils/sys_init/sys_init.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 系统初始化

系统初始化用于在算子开始执行时，将NPU中可能跨任务保留的控制状态恢复为C API约定的默认值。建议在Kernel入口、数据搬运和计算之前调用`asc_init`，避免前序任务遗留的原子操作、掩码或搬运循环配置影响当前任务。

`asc_init`初始化的是硬件控制状态，不会清零GM、L1、L0、UB或Reg中的用户数据。对于Ascend 950PR&950DT系列产品，接口会初始化原子操作、矢量掩码、CTRL状态，以及GM与UB之间搬运的循环配置等状态；具体行为以接口文档为准。

## 接口列表

- **[asc_init](asc_init.md)**
