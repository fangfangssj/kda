---
type: Guide
title: AscCopy：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。完成矢量寄存器之间、掩码寄存器之间以及矢量寄存器到掩码寄存器的数据搬运。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_load/asc_copy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_copy.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_copy
---

# AscCopy

完成三种寄存器搬运：矢量数据寄存器到矢量数据寄存器、掩码寄存器到掩码寄存器、矢量数据寄存器到掩码寄存器。各模式的 mask 与 part 合同不同。
函数原型：void asc_copy(vector_int8_t& dst, vector_int8_t src, vector_bool mask)
约束：针对矢量数据寄存器复制到矢量数据寄存器：dst中未被mask筛选的位置保持原值。；针对掩码寄存器复制到掩码寄存器：带mask的接口dst中未被mask筛选的位置填0。；针对矢量寄存器复制到掩码寄存器：part的值必须在编译期确定，可传入整数字面量或constexpr整数常量，不支持传入运行时变量。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 寄存器 C 调用合同与来源边界

以下 13 个声明来自固定的 950PR/950DT reference，分为同类寄存器复制和矢量到掩码复制：

```cpp
__simd_callee__ inline void asc_copy(vector_int8_t& dst, vector_int8_t src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_uint8_t& dst, vector_uint8_t src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_int16_t& dst, vector_int16_t src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_half& dst, vector_half src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_int32_t& dst, vector_int32_t src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_float& dst, vector_float src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_bool& dst, vector_bool src, vector_bool mask);
__simd_callee__ inline void asc_copy(vector_bool& dst, vector_bool src);
__simd_callee__ inline void asc_copy(vector_bool& dst, vector_uint16_t src, int16_t part);
__simd_callee__ inline void asc_copy(vector_bool& dst, vector_uint32_t src, int16_t part);
```

带 mask 的同类数据寄存器重载包含 int8/uint8、int16/uint16、half/bfloat16、int32/uint32 和 float。vector_bool 既有带 mask 也有不带 mask 的复制；不带 mask 的形式没有额外筛选参数。最后两种重载以 vector_uint16_t 或 vector_uint32_t 为源、vector_bool 为目的，part 的声明类型是 int16_t；它是从矢量中取出对应部分的偏移选择，不能用 mask 参数代替。返回 void，流水类型为 PIPE_V。

固定 reference 的通用 mask 参数段统一写“未筛选元素置零”，而本卡官方 reg_load 来源明确区分：数据寄存器复制保留未选中目的值，掩码寄存器的 masked 复制把未选中位清零。两份文档不足以证明所有版本都采用同一行为。使用时按目标 SDK 的具体重载核对；不能用通用段落覆盖分模式约束。对 vector→mask，继续要求 part 为编译期整数字面量或 constexpr，资料没有提供可推广的 part 数值范围。

这组 vector 寄存器参数与 [A2/A3 UB 指针形式](../../apis/simd_api/c_api/vector_datamove/asc_copy.md)不同。验证 masked 重载时，应先给 dst 写入可区分旧值，再检查有效及无效位置；只用全有效 mask 无法区分保留与清零。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_copy](../../apis/simd_api/c_api/reg_compute/reg_copy/asc_copy.md)
