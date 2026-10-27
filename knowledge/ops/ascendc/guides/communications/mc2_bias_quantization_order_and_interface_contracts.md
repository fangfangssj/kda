---
type: Guide
title: MC2 bias、量化次序与接口合同
description: 从需求文档抽取影响正确性的参数语义，排除追问流程。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/grill-protocol.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/quality-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/template.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/review-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow_integration.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# MC2 bias、量化次序与接口合同

融合次序决定数学：AllReduce(Ci+bias)在R个rank相同bias时得到sum(Ci)+R*bias，而AllReduce(Ci)+bias只加一次。激活、归一化和量化通常不与归约交换；dequantScale须明确per-tensor/per-token/per-channel或MX block及作用轴。形状匹配不能证明量化语义相同，per-tensor FP8基线与MX scale输入需声明差异。

每个输入输出需要逻辑shape、真实layout/stride、dtype、可选参数及广播关系；输入/累加/输出精度分开。FP8 E4M3和E5M2位模式不同，同一字节流不能硬编码按E4M3入口解释。若A/B各允许两种FP8则独立组合通常4种，Host按实际dtype分发；只有一种合同不必强造4入口。0元素、极值、Inf/NaN、最大/最小shape及非整除尾要有输出语义。

ACLNN常见两阶段为GetWorkspaceSize产生workspace尺寸和executor，再执行接口用workspace/executor/stream；这是模板提供的接口形态，不代表每个算子都含group/reduceOp/commTurn或只能ACLNN调用。通信域必须所有rank一致且生命周期覆盖通信，但“模型内全部MC2只允许一个group”是某些接口约束，不能默认全局定律。in-place/out-of-place、完成与可见性以及重复调用资源复用也独立定义。

精度要求应量化绝对/相对误差、特殊值及参考运算顺序；模板双千分之一/双万分之一是默认示例，不是所有dtype/shape的自动验收标准。确定性要覆盖跨rank和核内累加次序、AtomicAdd及量化舍入，不能只用固定seed证明。

拓扑/链路/通信域大小、实际AIC/AIV数、window和workspace预算及接口依赖版本影响实现可行性。没有读到资料只代表证据不足，不等于接口未实现；需求拷问轮次、角色审批和文档闸门属于Skill工作流，不作为知识规则。
