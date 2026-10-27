---
type: Guide
title: AscStore：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。从矢量数据寄存器搬出到 UB，保存 950 来源的 33 个 asc_store 重载、显式
  count 与满向量搬运边界。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_store/asc_store.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_store/asc_store.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_store
---

# AscStore

reg计算数据搬运接口，适用于从矢量数据寄存器搬出到UB的场景，不区分是否对齐，在追求极致性能时，对齐场景推荐使用asc_storealign接口。
函数原型：void asc_store(__ubuf__  int8_t* dst, vector_int8_t src)
约束：count的取值范围为[1, VL/sizeof(dst)]，VL为矢量长度。超出该范围会导致未定义行为。
支持平台：Ascend 950PR/Ascend 950DT；不支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品, Atlas A2 训练系列产品/Atlas A2 推理系列产品, Atlas 200I/500 A2 推理产品等

## 固定来源的完整重载集合

补充来源为固定 Skill 提交中的 asc-devkit 文档副本，支持范围为 Ascend 950PR/Ascend 950DT；未据此扩大本卡平台范围。下列是独立的函数重载声明，不是可以任意实例化的通用模板，共 17 个不带 count、16 个带 count：

```cpp
__simd_callee__ inline void asc_store(__ubuf__  int8_t* dst, vector_int8_t src);
__simd_callee__ inline void asc_store(__ubuf__  uint8_t* dst, vector_uint8_t src);
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src);
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src);
__simd_callee__ inline void asc_store(__ubuf__  int4b_t* dst, vector_int4x2_t src);
__simd_callee__ inline void asc_store(__ubuf__  fp8_e8m0_t* dst, vector_fp8_e8m0_t src);
__simd_callee__ inline void asc_store(__ubuf__  fp8_e5m2_t* dst, vector_fp8_e5m2_t src);
__simd_callee__ inline void asc_store(__ubuf__  fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src);
__simd_callee__ inline void asc_store(__ubuf__  hifloat8_t* dst, vector_hifloat8_t src);
__simd_callee__ inline void asc_store(__ubuf__  int16_t* dst, vector_int16_t src);
__simd_callee__ inline void asc_store(__ubuf__  uint16_t* dst, vector_uint16_t src);
__simd_callee__ inline void asc_store(__ubuf__  half* dst, vector_half src);
__simd_callee__ inline void asc_store(__ubuf__  bfloat16_t* dst, vector_bfloat16_t src);
__simd_callee__ inline void asc_store(__ubuf__  int32_t* dst, vector_int32_t src);
__simd_callee__ inline void asc_store(__ubuf__  uint32_t* dst, vector_uint32_t src);
__simd_callee__ inline void asc_store(__ubuf__  float* dst, vector_float src);
__simd_callee__ inline void asc_store(__ubuf__  int64_t* dst, vector_int64_t src);
__simd_callee__ inline void asc_store(__ubuf__  int8_t* dst, vector_int8_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  uint8_t* dst, vector_uint8_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  int16_t* dst, vector_int16_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  uint16_t* dst, vector_uint16_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  int32_t* dst, vector_int32_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  uint32_t* dst, vector_uint32_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  float* dst, vector_float src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  int64_t* dst, vector_int64_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  bfloat16_t* dst, vector_bfloat16_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  fp8_e8m0_t* dst, vector_fp8_e8m0_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  fp8_e5m2_t* dst, vector_fp8_e5m2_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  fp8_e4m3fn_t* dst, vector_fp8_e4m3fn_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  hifloat8_t* dst, vector_hifloat8_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e2m1_t* dst, vector_fp4x2_e2m1_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  fp4x2_e1m2_t* dst, vector_fp4x2_e1m2_t src, uint32_t count);
__simd_callee__ inline void asc_store(__ubuf__  int4b_t* dst, vector_int4x2_t src, uint32_t count);
```

`dst` 为 UB 目的起始地址，`src` 为对应矢量寄存器，返回 void，来源标注流水类型为 `PIPE_V`。不带 count 的接口执行该类型的满向量搬运；带 count 的接口搬运指定元素数。调用示例中的未初始化 `vector_half src` 只展示参数形式，不是可用于验证数值正确的完整程序。

来源列出 half 的不带 count 形式，却未列 half+count；不能从相邻类型或默认形式推导这一重载存在。低比特类型也必须保留指针/寄存器配对，例如 `int4b_t*` 对 `vector_int4x2_t`，FP4 指针类型带 `x2`；不能把打包存储单元和逻辑元素数自行等同。

## count、VL 与地址边界

原卡和补充文档均写 `count ∈ [1, VL/sizeof(dst)]`，越界为未定义行为，但该处 dst 又在原型中是指针，文字没有消除“指针大小还是元素宽度”的歧义；不带 count 的注释也只笼统称 VL 长度。使用时必须核实实际版本头文件对 VL 单位、类型元素数及打包计数的定义，不应在 Host 上直接以 `sizeof(pointer)` 计算可写元素数，或把所有类型一律设成同一个 count。

通用安全边界是目的有效区间足以容纳实际写入：尾部有效元素少于满向量时，不能直接调用默认满向量形式；count=0 也不属于来源给出的合法区间。该接口不区分地址是否对齐，但这不豁免目的容量、类型及真实流水依赖。来源在追求对齐路径性能时建议 [asc_storealign](../../apis/simd_api/c_api/reg_compute/store/asc_storealign.md)，它是另一个 API，不能只替换名称而忽略其独立合同。

核验时按实际 SDK 编译每个所需签名，覆盖 count 的最小/最大合法值、尾部目的区间、各实际 dtype 以及打包格式；声明集合只证明来源记录了这些形式，本批未执行 NPU 编译或搬运正确性测试。

## 相关

- [AscStorealign](../../apis/simd_api/c_api/reg_compute/store/asc_storealign.md)（`related`）：reg计算数据搬运接口，适用于从矢量数据寄存器搬出到UB的场景，不区分是否对齐，在追求极致性能时，对齐场景推荐使用asc_storealign接口。；将矢量数据寄存器或掩码寄存器中的数据以对齐的方式搬出到Unified Buffer（UB）。。AscStore 明确在对齐且追求性能时转用 asc_storealign，AscStorealign 则给出对齐搬出到 UB 的专用行为，形成通用搬出到对齐快路径的选择关系。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_store](../../apis/simd_api/c_api/reg_compute/store/asc_store.md)
