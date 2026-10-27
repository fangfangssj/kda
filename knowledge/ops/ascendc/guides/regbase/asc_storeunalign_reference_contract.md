---
type: Guide
title: AscStoreUnalign：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。从矢量数据寄存器连续非32B对齐搬出到UB，使用uint32_t作为存储偏移量，不改变源操作数地址，每次循环需用户手动更新目的地址。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_store/asc_storeunalign.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_store/asc_storeunalign.md
  role: supporting
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_storeunalign
---

# AscStoreUnalign

reg计算数据搬运接口，适用于从矢量数据寄存器连续非32B对齐的起始地址连续搬出到UB的场景。 该接口使用uint32_t作为存储偏移量，不改变源操作数在UB上的地址，每次循环的调用都需要用户手动更新目的操作数在UB上的地址。
函数原型：void asc_storeunalign(__ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count)
约束：该接口中的dst不需要32B对齐。；调用该接口之后，需要调用asc_storeunalign_post处理非对齐搬出的尾块。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 固定来源中的完整 C 声明

以下为固定 950PR/950DT 文档列出的 17 个类型配对；它们是来源声明，本卡未验证实际 SDK 是否在所有版本提供相同集合。保留原卡的非对齐场景及 post 收尾要求，不从已列类型推导其他重载。

```cpp
__simd_callee__ inline void asc_storeunalign(__ubuf__ int8_t* dst, vector_store_unalign& src0, vector_int8_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ uint8_t* dst, vector_store_unalign& src0, vector_uint8_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ int16_t* dst, vector_store_unalign& src0, vector_int16_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ uint16_t* dst, vector_store_unalign& src0, vector_uint16_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ int32_t* dst, vector_store_unalign& src0, vector_int32_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ int64_t* dst, vector_store_unalign& src0, vector_int64_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ uint32_t* dst, vector_store_unalign& src0, vector_uint32_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ half* dst, vector_store_unalign& src0, vector_half src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ float* dst, vector_store_unalign& src0, vector_float src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ bfloat16_t* dst, vector_store_unalign& src0, vector_bfloat16_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ fp8_e4m3fn_t* dst, vector_store_unalign& src0, vector_fp8_e4m3fn_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ hifloat8_t* dst, vector_store_unalign& src0, vector_hifloat8_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ fp8_e5m2_t* dst, vector_store_unalign& src0, vector_fp8_e5m2_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ fp8_e8m0_t* dst, vector_store_unalign& src0, vector_fp8_e8m0_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ fp4x2_e2m1_t* dst, vector_store_unalign& src0, vector_fp4x2_e2m1_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ fp4x2_e1m2_t* dst, vector_store_unalign& src0, vector_fp4x2_e1m2_t src1, uint32_t count)
__simd_callee__ inline void asc_storeunalign(__ubuf__ int4b_t* dst, vector_store_unalign& src0, vector_int4x2_t src1, uint32_t count)
```

返回 void，来源流水类型为 PIPE_V。dst 为目的 UB 指针；src0 是输入输出的 `vector_store_unalign&` 状态，来源长度为 32B，用来跨调用暂存未对齐数据；src1 是输入数据寄存器；count 为本次存储搬运个数。int4b_t* 对应 vector_int4x2_t，FP4 使用其 packed 配对，不能把指针基类型与寄存器名机械等同。来源未完整列出 packed count 的逻辑元素/打包单位和所有 count 边界，使用时需核对该类型头文件合同，不自行填入 VL/sizeof(T) 或零值支持。

## 地址推进、缓存与尾部

本接口不自动推进目的 UB 地址，多次调用时调用者提供每轮合法起点。原摘要中“不改变源操作数在 UB 上的地址”是来源沿用的表述；该签名唯一 UB 指针是 dst，不能据此假设另有 UB 源地址参数，也不能把 count 混成一个自动累加地址寄存器。

同一连续数据流需保留 src0 状态，最后由 [asc_storeunalign_post](../../apis/simd_api/c_api/reg_compute/store/asc_storeunalign_post.md)把剩余尾块写出，并使用与前序相同的状态。post 的基地址与 offset 应定位到完整搬运的结束地址；手动地址路线不能与会更新指针/地址寄存器的 postupdate 变体混用。缓存与最终收尾原理复用[连续非对齐优化](../operator_practice/simd_operator_optimization/vector_compute/vf_optimization/continuous_unaligned_optimization.md)。

来源示意 uint32_t 起点 UB 字节地址 8、每轮 64 元素、两轮共 512B：首轮可先写 248B 并暂存 8B，下一轮结合缓存继续写，最终 post 完成至字节地址 520。若 post 的基指针仍是原起点，其 offset 为 128 个 uint32_t 元素；不能将 512 字节直接作为同一指针的元素偏移。该例只解释所给宽度/状态路线，不推成所有 packed dtype 的 count 合同。

原例 `asc_storeunalign(dst[i * count], ...)` 将元素表达式放入指针形参位置，与列出的原型不一致；实际代码需形成合法 UB 指针（如相应元素地址），不能照抄后宣称可编译。验证应覆盖非 32B 起点、跨多轮缓存、最终短尾、输出有效区外哨兵、状态是否复用及各类型地址推进；不能只检查主循环输出而遗漏最后 post。

## 相关

- [AscStoreUnalignPost](../../apis/simd_api/c_api/reg_compute/store/asc_storeunalign_post.md)（`related`）：调用本接口之前，需要调用asc_storeunalign接口，或者asc_storeunalign_postupdate接口，且本接口与前序接口的非对齐寄存器要保持一致。；调用该接口之后，需要调用asc_storeunalign_post处理非对齐搬出的尾块。两端互相声明前序主体搬出与尾块收尾的强制调用顺序。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_storeunalign](../../apis/simd_api/c_api/reg_compute/store/asc_storeunalign.md)
