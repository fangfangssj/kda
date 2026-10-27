---
type: Guide
title: 高阶 Scratch 复用与 TPipe Reset 边界
description: 区分临时空间、输出和 GM 分配，按实际 API 空间需求建立阶段复用。
tags:
- task.design
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P89_ub_api_temp_buffer_sharing.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P88_multiphase_compute_phase_decomposit.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P89_ub_api_temp_buffer_sharing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P88.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P89.yaml
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_pipe/reset.md
  role: constraint
created_at: '2026-09-11T02:13:08Z'
updated_at: '2026-09-11T02:13:08Z'
platforms: []
---

# 高阶 Scratch 复用与 TPipe Reset 边界

多个高阶API或普通计算阶段的临时空间可以分时共享，前提是前阶段最后消费者与后阶段首个写者不重叠。相同起点且全部阶段互斥时，分配至少max(各阶段真实字节需求)，并满足最严格的对齐；有并发对象时按同时存活集合及布局计算，简单max单个buffer不够。不同dtype可以在不同时段建立合法视图，不要求所有阶段dtype相同，但类型转换不能由reinterpret替代。

Softmax等API的scratch应由对应版本、shape、dtype与算法的空间查询/tiling合同确定，profiling不能替代最小容量要求。源 `SoftMax(sharedBuf,input,params)` 没有区分输出和scratch实参：输出若仍要给下一阶段使用，就不能因API返回而把该值所在区域当临时区覆写。TBuf/TQue的InitBuffer重载也不同，源三参数示意不被认证为完整接口。

串行函数调用并不总等于硬件最后访问结束。复用需要覆盖实际流水/核间消费者的完成协议；PIPE_V只约束相应Vector域，不能普遍代替GM写出、跨核或Matmul异步完成。source建议无法共享时改用P85仍受同一物理寿命约束，不能靠更换策略名称绕开。

TPipe::Reset释放其管理资源并初始化eventId等状态，恢复TPipe初始状态。此前应完成相关计算、DMA、队列与事件使用，Reset后旧LocalTensor与事件身份不可沿用；另建TPipe也不会自动获得另一份物理UB。由Host分配的GM workspace容量不会因pipe->Reset自动缩减，需单独规划跨阶段GM别名并确保仍需传递的结果存续。

共享存储直接减少的是分配峰值，未必减少任何搬运指令。来源16→8次搬运或“预期减半”没有由所示地址重用推出；需要实际去掉中间读写或让更大tile减少轮数才可能产生间接流量收益。阶段拆分还增加初始化、同步和暂存成本，过细会抵消收益。

验证API scratch不足边界、不同dtype/对齐、输出仍活跃、异步排空与Reset后首个任务，再比较峰值空间和完整时延。已有[阶段资源重建](phase_boundaries_and_resource_reinitialization.md)保留跨阶段结果交接原则。
