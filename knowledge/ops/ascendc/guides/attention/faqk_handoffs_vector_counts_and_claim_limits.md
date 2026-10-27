---
type: Guide
title: FAQK 交接与 Vector Count 约束
description: 保存Token复用和批量布局知识，限制器件故障泛化。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: primary
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# FAQK 交接与 Vector Count 约束

来源FAQK可配置紧密LayoutC，而FAIQK选定特化带512stride与KV堆叠；换组件需同时更新调用签名、Q装载、page能力、资源分配和构造析构事件。FAQK示意接受可变pingpongFlag和isFirst，后者表示该调用真实A重新装载，不是任何调用永远true。

连续调用复用L1A时，下一GM→L1写必须等待前次MTE1读完。来源Wait(MTE1_MTE2,prev)再Set恢复token可在特定组件内部后续仍需消费该credit时成立；需核验实际事件所有者及调用步数，不能机械插入或用“短于32用全Barrier、长128用dance”当通用阈值。跨阶段、0/1调用和出口残留均需账目一致。

count重载Add/Muls/Exp/Cast/Adds表达元素数，不是repeat次数；repeat参数重载则另有mask/stride/次数语义。ReduceMax/Sum“每64只出一个/128静默错”可能来自具体重载、block/whole归约或workspace误用，不是所有高阶Reduce固定只支持64。分段64再合并是可选保守实现，末尾不足64要有效mask，scratch256float也需实际API而非固定经验。

批量16行可减少发射与搬运，前提是S0等确实紧密；若UB行pitch=KSEG，bulk load也必须按同一物理pitch，不能一边紧排一边以KSEG逐行计算。source330次*100ns约33us，与其约25us估计不同，计数/单次成本是粗模型不能当精确因果。

来源2D store仅首块、局部小stride load混行、Duplicate后SetValue全零等观测不足以禁止所有2D搬运或Scalar/Vector混合。应检查stride单位、blockLen、地址、元素位宽和S↔V/MTE依赖；逐行1D及物理独立scratch是候选修复，PIPE_ALL通过也不证明唯一根因。

UB偏移93188与93184相差4B说明源实例存在32B对齐条件，但全零正常返回不是该条件唯一症状。“器件文档不可信”“dump慢就不是race”等绝对结论不入库；模板注释、目标头文件、最小探针和真实运行证据应一起核验。
