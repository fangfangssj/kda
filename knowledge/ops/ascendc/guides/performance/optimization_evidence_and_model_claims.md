---
type: Guide
title: 优化证据与预测结论的区分
description: 从工具设计中提取可复用的实验记录语义，剥离架构和流程。
tags:
- task.optimization
- topic.performance.analysis
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/architecture/ARCHITECTURE.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/architecture/API_AND_MIGRATION.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/references/architecture/DESIGN_PRINCIPLE_IMPLEMENTATION.md
  role: supporting
created_at: '2026-09-11T04:58:58Z'
updated_at: '2026-09-11T04:58:58Z'
platforms: []
---

# 优化证据与预测结论的区分

观察指标、根因诊断、修改方案、预期收益及验证结果是不同对象。aiv_scalar_ratio>0.4与代码存在for循环，只能支持检查Scalar开销的假设；比值可随重叠与等待变化，循环也可能承担必需的GM分块。把所有tile循环去掉改Counter不能突破UB容量、生命周期或接口计数限制。vec<0.3不证明其他路径都是balanced。

定量证据应记录来源CSV行的任务身份、指标单位、源码位置、条件与对照。只取df.iloc[0]可能选错任务/shape，固定line_5字符串不是真正溯源。确定性的阈值算法、dataclass类型标注或JSON结构校验都不能保证诊断没有错误；来源宣称数据驱动即“无幻觉”、Python类型为编译时检查过强。

已验证优化case至少关联baseline与optimized代码/制品、相同工作量的profiling、正确性结果、环境、规则变体及时间。只有源码SHA不足以绑定编译配置、输入和设备状态。Build成功、correctness通过、performance_delta>0与goal_met是不同状态，模型中success省略goal不意味着已达用户目标；仅ProfilingData字段也不能独立验证输出数值。

规则success/failure计数形成历史样本频率，零样本时0是实现约定，不是失败概率1；同一算子反复尝试并非独立样本。平均改善需注明基线、单位、权重与shape/platform分层，不能把成功样本选择偏差当普适收益。冲突规则、失败原因及未决证据保留后才能解释适用范围。

预测模型输出改善区间与置信度属于待校准预测，不是实测。仅保存成功案例训练会失去失败边界；同case或同算子族跨训练/验证集合可能泄漏。来源40–60%预期、8.1→4.5us、42成功3失败等在架构提案中是示例，未随文附原始测量，不能作为本次验证能力或生产性能。

五层系统、CLI接口、状态机、迁移周计划、LLM模板、审批与自动重试均是Skill工具设计；知识库保留上述性能实验与推断边界，不导入该设计为执行流程。
