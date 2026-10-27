---
type: Guide
title: AscGather：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据索引位置index将源操作数src按元素收集到目的操作数dst中。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_load/asc_gather.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_load/asc_gather.md
  role: supporting
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_gather
---

# AscGather

根据索引位置index将源操作数src按元素收集到目的操作数dst中。 图 1  收集UB中的元素 图 2  收集矢量数据寄存器中的元素
函数原型：void asc_gather(vector_int16_t& dst, __ubuf__ int8_t* src, vector_uint16_t index, vector_bool mask)
约束：收集UB中的元素；源操作数在UB中的起始地址需要32B对齐。；对于mask筛选需要搬运的元素，对应的地址需要在UB有效范围内；对于mask未筛选的元素，对应的地址不会触发任何地址越界异常，同时dst中对应的元素将被置零。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 固定来源中的完整 C 声明

以下补充来自固定 Skill 仓归档的 950PR/950DT 文档，保留该资料列出的类型配对。它是文档声明，不是本机 SDK 编译结果。现有卡关于 UB 起点 32B 对齐、有效地址和 mask 的约束仍保留；版本升级时应同时核对真实头文件，不能用旧资料删除当前有效约束。

UB 输入路径共 17 个重载，源为 UB 指针，index 为元素索引，带 mask：

```cpp
__simd_callee__ inline void asc_gather(vector_int16_t& dst, __ubuf__ int8_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_uint16_t& dst, __ubuf__ uint8_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_int32_t& dst, __ubuf__ int32_t* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_uint32_t& dst, __ubuf__ uint32_t* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_half& dst, __ubuf__ half* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_float& dst, __ubuf__ float* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_fp8_e4m3fn_t& dst, __ubuf__ fp8_e4m3fn_t* src, vector_uint16_t index,vector_bool mask)
__simd_callee__ inline void asc_gather(vector_fp8_e5m2_t& dst, __ubuf__ fp8_e5m2_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_fp8_e8m0_t& dst, __ubuf__ fp8_e8m0_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_hifloat8_t& dst, __ubuf__ hifloat8_t* src, vector_uint16_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_int16_t& dst, __ubuf__ int16_t* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_uint16_t& dst, __ubuf__ uint16_t* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_half& dst, __ubuf__ half* src, vector_uint32_t index, vector_bool mask)
__simd_callee__ inline void asc_gather(vector_bfloat16_t& dst, __ubuf__ bfloat16_t* src, vector_uint32_t index, vector_bool mask)
```

寄存器输入路径共 12 个重载，源为矢量寄存器，不带 mask：

```cpp
__simd_callee__ inline void asc_gather(vector_int8_t& dst, vector_int8_t src, vector_uint8_t index)
__simd_callee__ inline void asc_gather(vector_uint8_t& dst, vector_uint8_t src, vector_uint8_t index)
__simd_callee__ inline void asc_gather(vector_int16_t& dst, vector_int16_t src, vector_uint16_t index)
__simd_callee__ inline void asc_gather(vector_uint16_t& dst, vector_uint16_t src, vector_uint16_t index)
__simd_callee__ inline void asc_gather(vector_int32_t& dst, vector_int32_t src, vector_uint32_t index)
__simd_callee__ inline void asc_gather(vector_uint32_t& dst, vector_uint32_t src, vector_uint32_t index)
__simd_callee__ inline void asc_gather(vector_half& dst, vector_half src, vector_uint16_t index)
__simd_callee__ inline void asc_gather(vector_bfloat16_t& dst, vector_bfloat16_t src, vector_uint16_t index)
__simd_callee__ inline void asc_gather(vector_fp8_e4m3fn_t& dst, vector_fp8_e4m3fn_t src, vector_uint8_t index)
__simd_callee__ inline void asc_gather(vector_fp8_e5m2_t& dst, vector_fp8_e5m2_t src, vector_uint8_t index)
__simd_callee__ inline void asc_gather(vector_fp8_e8m0_t& dst, vector_fp8_e8m0_t src, vector_uint8_t index)
__simd_callee__ inline void asc_gather(vector_hifloat8_t& dst, vector_hifloat8_t src, vector_uint8_t index)
```

返回值为 void，来源流水类型为 PIPE_V。dst 为输出寄存器；UB 路径 src 指向基地址，index 表示相对于它的元素位置，mask=1 才搬运，未选中 dst lane 清零。寄存器路径 src/index 均为寄存器，不应额外传入 UB 路径的 mask。不同数据宽度对应的 index 类型须逐行核对，不能根据名称自行拓展 unsigned/signed、float 或 packed 类型重载；例如来源没有列寄存器输入的 vector_float 声明，不从 UB 的 float 支持反推。

## 索引、扩展与跨接口复用

来源对 UB b8→b16 路径描述为低 8 位保留、高 8 位补零。因此 int8 位模式 -40（0xD8）进入该路径可得 16 位数 216，不是有符号扩展到 -40。不能把这条规则无差别套到表中所有 FP8/其他格式；必须确认所选类型配对的解释。已有 C++ [离散 Gather](../../apis/simd_api/basic_api/reg_vector_compute/reg_data_load/gather_discrete.md)承载同类原理，但 RegTensor 模板不是 C vector_* 的可调用声明。

寄存器输入长度固定，来源规定索引 i 超过当前寄存器元素容量 vl_length 时使用 `i % vl_length`；该取模仅描述寄存器输入路径，**不能**用来放宽 UB 输入的地址合法性。相关原理亦见 C++ [离散操作](../../apis/simd_api/basic_api/reg_vector_compute/scatter_operations/gather.md)。

950 的这两条路径与 [a2/a3 的 UB vector_compute 同名接口](../../apis/simd_api/c_api/vector_compute/vector_gather/asc_gather.md)参数及适用平台不同，不能仅按 asc_gather 名称合并。验证应分别覆盖两种输入、每个选定 dtype/index 配对、重复索引、寄存器容量边界、UB mask 选中/未选中地址和 int8 负值扩展。来源调用示例部分地址/index 未声明，不能直接当作完整可编译用例；本卡未执行 SDK 编译或设备验证。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_gather（UB 输入）](../../apis/simd_api/c_api/reg_compute/ub_gather/asc_gather.md)
- [asc_gather（寄存器输入）](../../apis/simd_api/c_api/reg_compute/reg_gather/asc_gather.md)
