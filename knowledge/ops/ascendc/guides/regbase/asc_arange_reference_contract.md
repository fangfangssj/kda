---
type: Guide
title: AscArange：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。以指定起始值生成递增或递减索引序列并保存到矢量数据寄存器，仅在AIV上生效。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/index_operate/asc_arange.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_arange.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_arange
- asc_arange_descend
---
# AscArange

以传入的value为起始值，生成递增/递减的索引，并将生成的索引保存在dst中，Vector Length (VL)表示矢量数据寄存器的位宽，VL_T表示该寄存器可存储的元素数量。算法逻辑表示
以int16_t数据类型，起始值value=10为例
递增索引为{10, 11, 12, 13, ... 135, 136, 137}，递减索引为{137, 136, 135, 134, ... 12, 11, 10}。

本接口仅在AIV上生效。

参数：dst(输出): 目的操作数（矢量数据寄存器）。；value(输入): 源操作数（标量），dtype须与dst一致。作为递增序列的起点，序列第0个元素等于value，后续元素按1递增。取值范围为该dtype的可表示范围。；dst(输出): 目的操作数（矢量数据寄存器）。；value(输入): 源操作数（标量），dtype须与dst一致。作为递减序列的起点，序列第0个元素等于value + VL_T - 1，后续元素按1递减。取值范围为该 dtype 的可表示范围。

固定 reference 列出的数据类型：int8_t、int16_t、half、int32_t、float，具体寄存器/标量配对见下文声明。

约束：- 本接口在非AIV上调用直接返回。
- 整型dtype（int8_t、int16_t、int32_t）结果在超出该dtype可表示范围时回绕（wrap-around），不触发异常。例如int8_t取value=127时，序列前128个元素依次为127、−128、−127、…、-2；value=−128时，序列前128个元素依次为−128、−127、…、−1。

仅 Ascend 950PR/Ascend 950DT 支持。

## 逐类型的递增与递减 C 声明

固定 950PR/950DT reference 分别给出 5 个递增和 5 个递减声明，dst 的寄存器类型与 value 的标量类型需按重载配对：

```cpp
__simd_callee__ inline void asc_arange(vector_int8_t& dst, int8_t value);
__simd_callee__ inline void asc_arange(vector_int16_t& dst, int16_t value);
__simd_callee__ inline void asc_arange(vector_half& dst, half value);
__simd_callee__ inline void asc_arange(vector_int32_t& dst, int32_t value);
__simd_callee__ inline void asc_arange(vector_float& dst, float value);
__simd_callee__ inline void asc_arange_descend(vector_int8_t& dst, int8_t value);
__simd_callee__ inline void asc_arange_descend(vector_int16_t& dst, int16_t value);
__simd_callee__ inline void asc_arange_descend(vector_half& dst, half value);
__simd_callee__ inline void asc_arange_descend(vector_int32_t& dst, int32_t value);
__simd_callee__ inline void asc_arange_descend(vector_float& dst, float value);
```

`asc_arange` 的第 i 个元素为 value+i；`asc_arange_descend` 为 value+VL_T-1-i，i 取 0..VL_T-1。递减形式的首元素不是 value，value 是该序列的最低端起始基值。两者均无 mask 参数；流水类型为 PIPE_V，返回 void。

来源原型没有给出其他整数、bfloat16 或 FP8 重载，不能从已列出的类型自行扩展。原卡的 AIV 调用范围和整型回绕约束继续保留；不能因为旧 reference 的约束栏写“无”就删去它们。浮点大值的相邻整数能否被目标 dtype 区分另受表示精度限制，本文不从索引名称推出无限精确整数序列。

实际使用应核对 SDK 声明，并分别检查所选 dtype 的首末元素、升降方向和整数边界。本卡保存文档合同，没有执行编译或设备测试；来源中未初始化的 value 示例不能作为有确定输出的验证用例。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_arange](../../apis/simd_api/c_api/reg_compute/reg_index/asc_arange.md)
