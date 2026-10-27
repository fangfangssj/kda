---
type: Guide
title: ND/NZ 随路变换与块位置重排
description: 保留 Nd2Nz 参数意图和 Attention 分块写回公式，明确实际数据方向与单位。
tags:
- task.implementation
- topic.data_movement.layout_transform
- topic.data_movement.data_copy
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P24_nd2nz_format_copy.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P24_nd2nz_format_copy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P26_stride_fused_transpose.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P24.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P26.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute/cube_compute_load/data_copy_gm_to_l1_nd2_nz.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/cube_compute/cube_compute_load/data_copy_ub_to_l1_nd2_nz.md
  role: constraint
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-11T01:44:57Z'
platforms: []
---

# ND/NZ 随路变换与块位置重排

矩阵 GM→L1 的 ND→NZ 随路搬运可以省去独立格式转换阶段，但仍执行数据复制及格式处理。ND/NZ 是逻辑矩阵到物理分形布局的变化，不能仅凭标题 ND↔NZ 认定同一重载双向支持。已有接口也记录 UB→L1 的 ND→NZ，因此“所有 ND→NZ 只能 GM→L1”并不成立；L1→L0 后续加载需继续遵守自己的矩阵合同。

来源 CopyToL1Nd2Nz 示例传入 nValue、dValue、srcDValue，并令 `dstNzC0Stride=ceil(nValue/16)*16, dstNzNStride=1`。这表达一个按 N 轴对齐的目的布局选择，不是任意矩阵的通用 stride。srcDValue 要与源矩阵实际行跨度对应；参数类型、单位、支持 dtype、矩阵数量及其余结构体字段都须按实际 SDK 初始化。仅给部分成员赋值且不确认默认值，不能作为完整构造模板。

来源将 FP16/FP8 的内部分形宽度举为16/32元素；这些用于具体类型布局的候选，不能把所有轴和所有 FP8 支持都从元素位宽推出来。L1 容量要按实际 padding 后布局，而非 nValue*dValue 的有效载荷计算。只有算子真的省去了一次中间存储往返，才可计入相应带宽收益。

ScatterUpdateNZ 是来源描述的逐 token 写出组织，不是此处已确认的独立 SDK 接口，也不是 NZ→ND 逆变换。逐 token 发布可能有标量地址更新到 MTE3 的依赖和短 DMA 成本；合并写回要保留完整 token 到 NZ 目标地址映射，不能机械删除同步。

Attention 写回示例把 g 个连续 D 元素块放到目的间距 t*D 个元素的位置。设 e=sizeof(OUT_T)，每块有效字节 L=D*e，则 GM 侧额外 gap=(t-1)*D*e，目的起点为 base+j*t*D*e。需要 t>=1、乘法无溢出且每段都落在自己输出范围。UB 侧 srcStride=0 表示没有额外 gap，物理块仍可能包含接口隐式对齐；D*e 非对齐时不能误读为各 UB 起点相距恰 L。

上述重排省去独立转置阶段，仍是搬运。能否表达一个转置取决于各块内部连续性与块间仿射地址映射，不能用“只能一维”代替证明。普通 DataCopyParams 的 uint16 字段范围不是65535字节的统一 stride上限；Ext 参数也需要确认该方向、单位与真实硬件范围。列提取可按源块间距组织，但其对齐要求不能跨重载泛化。

验证源坐标编码、N/D 尾部、跨 token/group、非紧密 srcDValue、不同 dtype、填充区和 GM 哨兵。基础接口见[GM→L1 ND2NZ](../../apis/simd_api/basic_api/cube_compute_isasi/cube_compute_load/data_copy_gm_to_l1_nd2_nz.md)及[UB→L1 ND2NZ](../../apis/simd_api/basic_api/cube_compute_isasi/cube_compute_load/data_copy_ub_to_l1_nd2_nz.md)。
