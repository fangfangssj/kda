---
type: Guide
title: Flash VF 的掩码、Cast 分布与标量槽
description: 补全 Reg 实现模板的物理访问和数值合同，限定项目 API 选择与版本假设。
tags:
- task.implementation
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke-flash/skills/ops-direct-invoke-flash/references/implementation-patterns.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke-flash/skills/ops-direct-invoke-flash/references/reg-api-guide.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke-flash/skills/ops-direct-invoke-flash/references/reg-api-patterns.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-official/ops-direct-invoke-flash/skills/ops-direct-invoke-flash/references/review-prompts.md
  role: supporting
created_at: '2026-09-11T03:52:52Z'
updated_at: '2026-09-11T03:52:52Z'
platforms: []
---

# Flash VF 的掩码、Cast 分布与标量槽

来源Flash采用三层：__simd_vf__函数接收__ubuf__地址并做Reg计算；__aicore__包装器从LocalTensor取GetPhyAddr并调用VF；Kernel外壳管理TQue/TPipe和DMA。Reg-only、禁用MicroAPI/MemBase或裸asc函数但允许asc_vf_call，是来源针对3510工程的选择，不是所有Ascend C版本的硬件禁令。包装器与调用者的地址空间和参数类型仍需真实编译器支持。

以FP32计算时VL=VECTOR_REG_WIDTH/sizeof(float)，B16输入不应直接沿用B16的lane数来推进FP32寄存器。int64 count转uint32需先证明非负与范围；ceil(count/VL)可写商加余数是否非零以避免count+VL-1溢出，chunkCount不能多于有效分块。最后一轮offset<count，再计算剩余量，避免无符号下溢。来源版本UpdateMask(uint32_t&)会消耗引用变量，不能额外重复递减。

计算与Store的mask不保证无mask的LoadAlign物理安全，尾块需要有效padding或相应有界加载方案。来源单lane写回使用UpdateMask<float>(1)与非const引用签名不匹配，应准备可修改的uint32_t局部计数并按目标API核对，不能把伪代码字面量直接当可编译示例。

来源B16→FP32模式使用LoadDist::DIST_UNPACK_B16，CastTrait包含RegLayout::ZERO、SatMode::UNKNOWN、MaskMergeMode::ZEROING、RoundMode::UNKNOWN，FP32 Store为DIST_NORM_B32；反向用FP32正常加载，Cast采用NO_SAT/CAST_RINT并DIST_PACK_B32写B16。这是一组来源模式，类型、traits与分布组合须由实际Reg Cast重载验证，不代表全部B16或所有SDK均支持。中间FP32提升会改变舍入次数和UB预算。

标量归约示例为每行分配32B槽，即FP32的8元素，行i的槽偏移i*8，单有效lane保存归约结果；后续可用Reg层Div/Mul/Duplicate组合完成标量运算。但寄存器Duplicate重载、标量值取得方式和跨VF/标量读写完成关系应匹配实际API，禁止GetValue是来源风格限制，不能代替同步证明。每行槽仍必须在已初始化、足够容量的LocalTensor内。

更多RegTensor、Load/Store、MaskMerge与归约族约束见[API操作数合同](api_families_masks_and_memory_contracts.md)。本文保留可复用实现语义，来源提示词、角色分配和审查输出格式留在Skill。
