---
type: Guide
title: 卷积分组合并与多核代价模型范围
description: 保存合组因子和分核候选，指出组数与输出通道混用。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/tiling/convolution/fallback/script/conv_tiling.py
  role: primary
created_at: '2026-09-11T06:37:24Z'
updated_at: '2026-09-11T06:37:24Z'
platforms: []
---

# 卷积分组合并与多核代价模型范围

分核可沿batch、线性M或独立Ho/Wo、输出通道N和group。每维候选为问题维度约数混合核数约数/部分2幂，乘积≤C；M模式最小化近似load_fmap/load_weight/load_out与Cube成本，HW模式仅最大化使用核数。二者都非全局最优保证；HW的waste=C-total在同total下完全相同，没有额外尾浪费判别，可能忽略halo重复与实际每核尾。

单核形状ceil分配并对齐N，必须按每个任务的真实起点和边界裁掉padding。M单核长度由Align(M,16)/mDim再ceil，未必仍16对齐，后续L0容量与有效长度应分开；用核数0自动回退32也不等于合法平台配置。

合组示例令ci_pg=Ci/G、co_pg=Co/G，enlarge取使ci_pg、co_pg分别对齐K0/N0所需倍率的lcm，再限G；groupOpt=ceil(G/enlarge)。还用展开权重双份UB≤UB-256判断是否启用。前提Ci和Co能按G整除、每组正，合并权重的块对角/零填充及最后不足enlarge组保持组间独立，不能只因UB可放就假定布局已实现。

普通未合组的_compute_group_params错误使用ceil(cur_co/group_dim)作为single_core_group_opt，cur_co是每组输出通道而非组总数。CPU设置G8、每组Co32、groupDim2返回16，预期按组均分应4。修正需传真实G、保持groupDim与组索引映射一致。

源码有C04小通道、UB权重转换与fmap copy相关字段和helper，但主_run_l0_l1_tiling没有传is_c04/is_dma，转换步长字段一直0；不可据字段存在就宣称已支持该优化。新增路径要同时实现布局转换、资源、事件和TilingKey匹配，并覆盖depthwise、一般分组、不整齐空间、不同dtype与尾组。
