---
type: Guide
title: DAG 输出与 Foreach Tensor 区间合同
description: 保存调度抽象和列表游标，补齐状态输出及整数边界。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_13_special_algorithms.md
  role: primary
created_at: '2026-09-11T06:02:18Z'
updated_at: '2026-09-11T06:02:18Z'
platforms: []
---

# DAG 输出与 Foreach Tensor 区间合同

AdaGrad的Bind节点可描述grad²、accum+grad²、sqrt、lr*grad、除法及var减更新；Elems显式声明var_out/accum_out两个输出，DAGSch与ElementwiseSch依据模板与MemOptCfg调度。此图仍需匹配update_slots、epsilon/分母0、Cast和别名语义，不能因选LEVEL_2就保证最优寄存器分配/所有访存融合。手工队列也可合法优化，实际融合由生成代码和时间线确认。

FakeQuant Base集中公共字段与CopyIn/Out、Fp32/Fp16派生处理转换差异，可减少重复实现；新增BF16仍需注册、Host key、所有转换/缓冲/数值链支持，不能只加继承类。模板多态可静态绑定，不意味着全入口和通用helper无运行时代价。

Foreach每核可拥有TensorList的区间[tensorStart,tensorEnd]及首尾元素cursor。来源cursorEnd=最后元素位置而非exclusive终点，dataCount应为end-start+1；中间Tensor的end=numel-1，空Tensor须先处理，不能无符号下溢。uint16 i<=tensorEnd在end65535时++会回绕，必须限制列表范围或使用足够宽的安全循环。首尾同一个Tensor、跨多个空项和每核空区间均需覆盖。

GetTensorAddr返回地址单位须按实际描述符/typed指针，+cursorStart只有在元素指针语义正确时才是元素offset；多个输入列表和输出同Tensor身份/偏移，不能只绑定第一个输入。source未展示dataCount计算及全部加数/除数指针，不能当可运行完整ForeachAddcDiv。

ProcessPlusInLoop可按Tensor索引加载对应scalar、重绑定第二/第三输入，减少一次性载入所有scalar的容量；多个核处理同Tensor片段仍应读同一scalar。强转T可能窄化API层较宽scalar，需符合实际算术合同，非每种场景都能先转低精度。TensorList批量启动可减少逐Tensorlaunch次数，代价是metadata、非均匀负载及地址间接访问。

AutoContiguous可让支持的框架路径在调用前转换非连续输入，可能新增分配/复制，不能证明裸Kernel支持任意stride，也不能据此忽略非连续输出写回。真实收益用同一调用边界比较，不能把Host变换开销排除后声称整体更快。
