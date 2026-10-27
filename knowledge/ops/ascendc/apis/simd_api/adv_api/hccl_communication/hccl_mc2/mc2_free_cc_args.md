---
type: API
title: Mc2FreeCcArgs
description: 释放Mc2GetCcArgs创建的MC2通信参数对象。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_MC2/Mc2FreeCcArgs.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
---

# Mc2FreeCcArgs

## 功能说明

释放[Mc2GetCcArgs](mc2_get_cc_args.md)创建的MC2通信参数对象。

## 函数原型

```cpp
extern Mc2Result __attribute__((visibility("default"))) Mc2FreeCcArgs(void* ccArgs)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccArgs | 输入 | [Mc2GetCcArgs](mc2_get_cc_args.md)返回的MC2通信参数对象指针。 |

## 返回值说明

-   0表示释放成功。
-   非0表示释放失败，例如参数为空指针。

## 约束说明

-   本接口仅释放ccArgs，不释放[Mc2AcquireCcResCtx](mc2_acquire_cc_res_ctx.md)返回的ccResCtx。
-   ccResCtx由通信域生命周期管理，不需要用户单独释放。
-   不支持重复释放同一个ccArgs。

## 调用示例

本接口的调用示例请见[HCCL MC2使用说明](hccl_mc2_usage.md)。
