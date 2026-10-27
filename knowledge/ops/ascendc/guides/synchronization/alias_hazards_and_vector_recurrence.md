---
type: Guide
title: 向量递推中的真实依赖与别名冲突
description: 用读写集合区分 RAW、WAR 和 WAW，说明额外缓冲与局部屏障的实际作用。
tags:
- task.design
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P30_dense_pipebarrier_vector.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P30_dense_pipebarrier_vector.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P30.yaml
  role: supporting
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-11T01:44:57Z'
platforms: []
---

# 向量递推中的真实依赖与别名冲突

按同一地址的先后访问定义：先写后读是 RAW，先读后写是 WAR，先写后写是 WAW。`Mul(tmp,x,w)` 后 `Add(out,y0,tmp)` 需要读取刚生产的 tmp，属于 RAW；即使 out 与 tmp 不同，这条真实依赖仍存在。来源把它标为 WAR，并说改成独立 dst/src 就可删除全部 barrier，不能成立。

增加临时缓冲能消除因复用同一存储而产生的某些 WAR/WAW，但不会消除数据本身的 RAW，也不保证原地重叠操作得到接口支持。是否需要显式同步，取决于实际 TensorBase/RegBase 执行与编译器、队列、API 已提供的合同。先列出地址范围和消费者，再确认现有保证，不能只按变量名字或指令相邻判断。

递推 `y1=x*w1+y0; y2=x*w2+y1; y3=x*w3+y2` 同时有每次 Mul→Add 和上一次 Add→下一次 Add 的依赖。来源在中间 Mul 后放一次 PIPE_V；若该模式的屏障排空此前相关 Vector 操作，它可能覆盖两条前置边，不能只看最近一条指令。独立 Mul 可以提前调度，但不能提前消费 y1/y2。

PIPE_V 的作用域仍只覆盖其规定的 Vector 流水。一个 Kernel 含 Cube 并不禁止其 Vector 子阶段使用合法的 PIPE_V；只是它不能代替 Cube 完成或 AIC/AIV 交接。多核存在也不意味着任意局部递推都要 SyncAll，只有实际共享数据与参与协议才提出跨核要求。

来源按短链<=4选择密集屏障、每条固定1–2周期、UB充裕时交替buffer必最优等都不是通用保证。可比较选择性屏障与重排后实现，但应同时记录新增字节、同步等待和完整时延。验证连续多轮递推、别名/独立区、尾长度以及不同执行模式；数值变化仍需区分同步错误与改变算术顺序。
