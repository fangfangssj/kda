---
type: Guide
title: RegBase 的 GM、UB 与寄存器数据流分层
description: 区分 Kernel 搬运与 VF 寄存器计算的对象、地址空间和生命周期，说明 MemBase 迁移时需要保留和重写的边界。
tags:
- task.implementation
- topic.programming_model
- topic.data_movement
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/api/datacopy_best_practices.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/api/compute_api_membase_vs_regbase.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/pitfalls/regbase_vs_membase_confusions.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/dev-experience/regbase_programming_notes.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/regbase_development_guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-regbase-best-practice/references/dev-experience/regbase_performance_practices.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/reg_vector_compute/reg_data_load/load_align_continuous.md
  role: constraint
created_at: '2026-09-10T12:22:42Z'
updated_at: '2026-09-10T12:22:42Z'
platforms:
- '950'
---

# RegBase 的 GM、UB 与寄存器数据流分层

## 数据通路

在支持 Reg 计算的 Ascend 950 上，典型通路为 GM → UB → RegTensor → UB → GM。各段承担不同契约：

| 层次 | 输入输出 | 主要职责 |
|---|---|---|
| Host tiling | shape、dtype、切分参数 | 计算核间分工、块长、尾块与资源预算 |
| Kernel 外层 | GlobalTensor、LocalTensor、队列 | 分配 UB、调度搬运、安排 tile 与数据交接 |
| VF 入口和计算 | UB 地址、RegTensor、MaskReg | 加载寄存器、完成数学链、按有效 lane 写回 UB |

`DataCopy`/`DataCopyPad` 的 GM↔UB 形态不会直接产生 RegTensor；VF 中的 `LoadAlign`/`StoreAlign` 处理 UB↔寄存器段。两类调用的参数对象、地址空间和执行上下文不能互换。外层使用 TQue 或 LocalTensor 并不妨碍计算链采用 RegBase。

从 MemBase 迁移时，通常可以保留已正确的 Host 分工与 GM/UB 暂存设计；计算层需要重新确定寄存器装载、运算、mask 和写回。给原 API 名增加 `Reg::` 前缀不能完成迁移，同名函数可能有不同签名与 dtype 支持。

## 中间值与资源

单次 VF 内短暂使用的数学中间值可留在寄存器，以减少 UB 往返。跨 VF 调用或需要被其他阶段消费的状态应通过明确的 UB/GM 存储交接，并满足对应同步约束。不能由某个归约案例推导出所有 RegTensor 都不能跨 VF 内循环存活。

融合是否有收益取决于实际减少的搬运、转换和同步；函数数量减少不等于数据流成本下降。资源预算仍须包含输入、输出、scratch、转换缓冲和实际分配的缓冲份数。

## 边界与验证

本文是数据流设计说明，不承诺来源中的伪代码可直接编译。`__simd_vf__`、`asc_vf_call`、命名空间别名和具体重载应以目标 SDK 为准。来源提及的 CANN 9.0.0 签名不能无条件外推到其他版本或 A2/A3。

验证时分别检查 GM 输入、UB 暂存、VF 输出与最终 GM 输出，覆盖单 VL、多 VL 和尾块；确认短输入通过后，还需验证跨块状态与缓冲复用。

相关接口与案例：[LoadAlign](../../apis/simd_api/basic_api/reg_vector_compute/reg_data_load/load_align_continuous.md)、[特定 Softmax 归约的 UB 状态回退](../../runbooks/precision/regtensor_cross_chunk_state_should_persist_in_ub_slot.md)。
