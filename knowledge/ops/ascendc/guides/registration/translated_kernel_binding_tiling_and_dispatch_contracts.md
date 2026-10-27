---
type: Guide
title: 转换后 Kernel 的绑定、Tiling 与分发合同
description: 保存 Pybind/TORCH_LIBRARY 身份及整除假设，排除文件组织强制规则。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_host.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_cube.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_vector.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc_compute_cv.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/tool_interface/system_resources_and_variables/get_sub_block_num_isasi.md
  role: constraint
created_at: '2026-09-11T04:44:50Z'
updated_at: '2026-09-11T04:44:50Z'
platforms: []
---

# 转换后 Kernel 的绑定、Tiling 与分发合同

Host绑定负责真实输入校验、输出/workspace分配、Tiling与launch；设备阶段负责对应计算。来源规定只收输入、每个prim_func独立类/入口/extern、Scope拆文件及禁止ATen计算属于该Skill结构选择，不能解释为Ascend C只允许此组织。分支可以共用或分开，只要完整保留数学、ABI、编译变体和实际执行范围，不能留下空Device入口却把Host其他算子计作本Kernel。

PYBIND11_MODULE的模块名需与Python扩展入口匹配，任务同名目录可能遮蔽扩展。它与仅用TORCH_LIBRARY静态注册再load_library的包是两条有效加载路径，不能把其中一条当普遍唯一。来源推荐_<op>_ext与另一文档<op>包名不是同一层命名；需绑定实际导出与加载文件。

Host、Cube、Vector对baseM/N/K、物理stride、workspace slot及dtype必须一致，但Vector chunk宽度可小于或大于Cube块，只要有显式正确拼接/分段，vecBlockN不一定数值等于baseN。subBlockM=baseM/vecNum需要尾行分配；不能整数截断遗漏。Tiling字段int32 M/N/H_K、baseM/N/K/K_L1、nTiles/nTilesPerH等需检查范围与ABI。

来源nTiles=N/baseN、nTilesPerH=H_K/baseN依赖整除且baseN>0；否则漏尾或nTilesPerH=0使groupId除零。即使用ceil，跨group边界仍需单独确定每组tile及尾部，不能简单把全局by除一个错误计数。Host预计算可减少重复开销，也可在Kernel合法计算；把所有派生量写struct不是正确性必需。

workspace声明表示原设计需要暂存，转换后应保留真实消费者需求、byte数、对齐和生命周期；若证明完全消除暂存，则不必仅因原workspace_idx存在而无用分配。Byte tensor可承载原始字节，但typed偏移、allocator对齐、stream使用及异步释放需匹配。输入device、连续性/stride、属性、输出shape和多输出结构也不能只校验dtype。

通用入口的纯Cube/纯Vector例都写MIX_AIC_1_2，不代表它们必须混合启动；真实核型/分支需与编译及任务映射一致。来源混合表把GetSubBlockNum写为1+vecNum的2/3，与已有API所述“当前逻辑AI Core上AIC或者AIV数量”不符；不能把AIC+AIV总数用于AIV行切分，应按当前执行侧及配置取值并核对实际映射。

来源主文重复scan monkey-patch及L≤512精度切换，其证据边界见已抽取的参考污染卡；这不是可迁入的默认验证行为。
