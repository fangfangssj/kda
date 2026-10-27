---
type: API
title: GetStoreAtomicConfig
description: 获取当前原子操作的开启状态和数据类型配置。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.atomic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/GetStoreAtomicConfig_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_common_intf.h
---

# GetStoreAtomicConfig\(ISASI\)<a name="ZH-CN_TOPIC_0000001836926165"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

获取原子操作开启位与原子操作类型的值，详细说明见[表1](set_store_atomic_config_isasi.md#table622mcpsimp)。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline void GetStoreAtomicConfig(uint16_t& atomicType, uint16_t& atomicOp)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| atomicType | 输出 | 原子操作开启位。<br>0：无原子操作<br>1：开启原子操作，进行原子操作的数据类型为float<br>2：开启原子操作，进行原子操作的数据类型为half<br>3：开启原子操作，进行原子操作的数据类型为int16_t<br>4：开启原子操作，进行原子操作的数据类型为int32_t<br>5：开启原子操作，进行原子操作的数据类型为int8_t<br>6：开启原子操作，进行原子操作的数据类型为bfloat16_t |
| atomicOp | 输出 | 原子操作类型。<br>0：求和操作 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

此接口需要与[SetStoreAtomicConfig\(ISASI\)](set_store_atomic_config_isasi.md)配合使用，用以获取原子操作开启位与原子操作类型的值。

## 调用示例<a name="section837496171220"></a>

```cpp
AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
uint16_t type = 0;       // 原子操作开启位
uint16_t op = 0;         // 原子操作类型
AscendC::GetStoreAtomicConfig(type, op);
```
