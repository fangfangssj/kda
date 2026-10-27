---
type: Guide
title: AscReduceMin：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据mask对源操作数进行归约最小值操作，将最小值及其索引写入目的操作数。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reduce_compute/asc_reduce_min.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_reduce_min.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_reduce_min
---
# AscReduceMin

根据mask对源操作数src进行归约最小值操作，将最小值写入目的操作数dst的第0个元素，将最小值在src中的索引原始位模式写入dst的第1个元素，并将dst中的其他元素置0。如果存在多个最小值，则写入最小的索引。

参数：dst(输出): 目的操作数（矢量数据寄存器）。；src(输入): 源操作数（矢量数据寄存器）。；mask(输入): 源操作数掩码（掩码寄存器），用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。

支持数据类型：float、half、int16_t、int32_t、uint16_t、uint32_t。

约束：- 未被mask选中的元素被视为对应数据类型的最大值，浮点数类型的最大值为+inf。如果src中的所有元素均未被mask选中，则将该最大值写入dst的第0个元素，并将其余元素置0。
- 如果输入数据中存在nan，则将nan写入dst的第0个元素，并将第一个nan的索引写入dst的第1个元素。

仅 Ascend 950PR/Ascend 950DT 支持。

## 六种寄存器 C 声明

固定 950PR/950DT reference 给出以下重载，dst/src 逐项保持相同类型，mask 为 vector_bool：

```cpp
__simd_callee__ inline void asc_reduce_min(vector_uint16_t& dst, vector_uint16_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_min(vector_int16_t& dst, vector_int16_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_min(vector_half& dst, vector_half src, vector_bool mask);
__simd_callee__ inline void asc_reduce_min(vector_uint32_t& dst, vector_uint32_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_min(vector_int32_t& dst, vector_int32_t src, vector_bool mask);
__simd_callee__ inline void asc_reduce_min(vector_float& dst, vector_float src, vector_bool mask);
```

返回 void，流水为 PIPE_V。结果和源索引写入目的寄存器的布局、空 mask、同值取最小索引及 NaN 规则按上文约束处理；源 mask 的坐标不能直接当作输出坐标。索引保存的是原始位模式，浮点 dst[1] 不能用普通浮点数值转换恢复索引。

这里是单个向量寄存器内的最小值归约；[SIMT 同名函数](../../apis/simt_api/warp_functions/warp_reduce_functions/asc_reduce_min.md) 按 warp 活跃线程归约并返回标量，不具有本接口的 dst/src/vector_bool 签名。验证应分别覆盖唯一极值、相同极值、空 mask 和有效 NaN。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_reduce_min](../../apis/simd_api/c_api/reg_compute/reg_reduce/asc_reduce_min.md)
