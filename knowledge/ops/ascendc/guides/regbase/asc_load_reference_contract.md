---
type: Guide
title: AscLoad：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。从UB非32字节对齐的源地址搬运VL大小的数据到矢量数据寄存器。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_load/asc_load.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_load/asc_load.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
---

# AscLoad

reg计算数据搬运接口，支持从UB非32字节对齐的源地址src搬运至矢量数据寄存器，搬运量为VL。 连续搬运时，用户需手动更新src地址。
函数原型：void asc_load(vector_int8_t& dst, __ubuf__ int8_t* src)
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 固定文档中的重载类型与调用合同

Skill 仓固定版本转存的文档声明本组接口用于 Ascend 950PR/950DT；接口均为 `__simd_callee__ inline void asc_load(vector_type& dst, __ubuf__ scalar_type* src)`，不同重载的寄存器类型和源元素类型对应如下。该表补足单个 int8 原型不能表达的类型覆盖，不表示其他 SDK 版本自动具备全部重载。

| dst 的 vector_type | src 的 scalar_type |
|---|---|
| vector_int8_t | int8_t |
| vector_uint8_t | uint8_t |
| vector_int16_t | int16_t |
| vector_uint16_t | uint16_t |
| vector_int32_t | int32_t |
| vector_uint32_t | uint32_t |
| vector_int64_t | int64_t |
| vector_half | half |
| vector_float | float |
| vector_bfloat16_t | bfloat16_t |
| vector_fp8_e4m3fn_t | fp8_e4m3fn_t |
| vector_fp8_e5m2_t | fp8_e5m2_t |
| vector_fp8_e8m0_t | fp8_e8m0_t |
| vector_fp4x2_e2m1_t | fp4x2_e2m1_t |
| vector_fp4x2_e1m2_t | fp4x2_e1m2_t |
| vector_hifloat8_t | hifloat8_t |
| vector_int4x2_t | int4b_t |

src 位于 UB，dst 为输出矢量寄存器；文档给出的搬运量为 VL、流水类型为 PIPE_V、返回值为 void。允许源起点非 32 字节对齐，不等于能够越过已分配的源范围。连续调用需由调用者推进源地址，不能按 postupdate 接口使用。推进量还需按 VL 与实际元素/打包类型解释，不能从函数名推断固定逻辑元素数。

来源示例用 vector_half 与 UB half 指针调用；它不独立证明前序搬入已经完成、尾部容量足够或目标 SDK 编译成功。实际使用时核对安装头文件重载、寄存器定义、有效源范围和生产消费依赖。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_load](../../apis/simd_api/c_api/reg_compute/load/asc_load.md)
