---
type: Guide
title: TBuf 常驻仍占 UB 且需要真实数据依赖
description: 保留循环不变量、状态更新和搬入转换链，纠正常驻比例与队列误解。
tags:
- task.optimization
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P81_resident_ub_tbuf_reuse.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P81.yaml
  role: supporting
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# TBuf 常驻仍占 UB 且需要真实数据依赖

小参数在同一复用域不变时，可以把DataCopy及所需Cast移到循环外，整个循环保留结果，再在最后消费者完成后回收。节省的是重复加载/转换，收益不要求至少4次；复用次数、准备成本、主tile缩小和并行机会共同决定。20% UB占比只是来源经验，不是合法性边界。

TBuf是资源管理方式，VECCALC位置的TBuf仍占用UB，不能把“大于20%就换TBuf”视为扩容方案。TQue同样不保证只要调用AllocTensor就已有搬入/发布同步；手动使用已分配句柄需要完整数据依赖。循环内分配释放可以合法，是否前移取决于寿命和队列协议，不是编译器一律禁止。

GM→UB搬入与后续Cast之间应有相应MTE2→V完成边界，仅在Cast之后加PIPE_V不能补救已提前读取的输入。不同dtype的源/目的容量和允许别名需核对；来源直接把half输入搬到float视图再原地Cast的示意不能省略真实源类型。只读常驻数据在后续迭代不被写入时，也不需要仅因再次读取而机械增加每轮屏障。

若Compute更新状态，常驻仍可能合法，但该对象已是跨迭代递推状态，必须维护正确新旧版本及最终发布，而非继续宣称每轮使用初始参数。多核各有独立UB不等于共享running_mean等GM写回无竞争，最终合并/所有权需另行保证。参数变化、head/group切换及下一次调用均要重新建立复用域。

验证零循环、多轮、类型扩宽、参数变化与共享状态最终输出，核算原始/转换/广播对象同时存活的真实字节数。扩宽视图的具体边界见[常驻参数视图](promoted_constant_views_and_reuse_domains.md)。
