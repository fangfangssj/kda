---
type: API
title: SetStoreAtomicConfig
description: 设置原子操作的开启位和操作类型，支持多种数据类型配置。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.atomic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/SetStoreAtomicConfig_ISASI.md
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

# SetStoreAtomicConfig\(ISASI\)<a name="ZH-CN_TOPIC_0000001790206710"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

设置原子操作开启位与原子操作类型。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <AtomicDtype type, AtomicOp op>
__aicore__ inline void SetStoreAtomicConfig()
```

## 参数说明<a name="section622mcpsimp"></a>

<a name="table622mcpsimp"></a>**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| type | 输入 | 原子操作开启位，AtomicDtype枚举类的定义如下：<br>enum class AtomicDtype {<br>    ATOMIC_NONE = 0,  // 无原子操作<br>    ATOMIC_F32,       // 开启原子操作，进行原子操作的数据类型为float<br>    ATOMIC_F16,       // 开启原子操作，进行原子操作的数据类型为half<br>    ATOMIC_S16,       // 开启原子操作，进行原子操作的数据类型为int16_t<br>    ATOMIC_S32,       // 开启原子操作，进行原子操作的数据类型为int32_t<br>    ATOMIC_S8,        // 开启原子操作，进行原子操作的数据类型为int8_t<br>    ATOMIC_BF16       // 开启原子操作，进行原子操作的数据类型为bfloat16_t<br>}; |
| op | 输入 | 原子操作类型，仅当开启原子操作时有效（即"type"为非"ATOMIC_NONE"的场景），当前仅支持求和操作。<br>enum class AtomicOp {<br>    ATOMIC_SUM = 0   // 求和操作<br>}; |

## 数据类型

支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```cpp
// 设置原子操作为求和操作，支持的数据类型为half
AscendC::SetStoreAtomicConfig<AscendC::AtomicDtype::ATOMIC_F16, AscendC::AtomicOp::ATOMIC_SUM>();
```
