---
type: Guide
title: 优化组合的资源与依赖合同
description: 按同一缓冲对象、同一控制域和峰值存活量判断兼容，纠正全局硬互斥。
tags:
- task.design
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/strategy-compatibility.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/SCHEMA.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/SCHEMA.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/SCHEMA.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/prompts/lingxi-partial-prompt.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_que_bind/t_que_bind_intro.md
  role: constraint
created_at: '2026-09-11T02:29:01Z'
updated_at: '2026-09-11T02:29:01Z'
platforms: []
---

# 优化组合的资源与依赖合同

两项优化能否组合，要看它们是否争夺同一资源或修改同一协议。一个buffer不能同时由互不协调的队列与手动ping-pong管理，但不同buffer可采用不同机制。双/三缓冲是具体对象的份数选择，三份不自动保证三个阶段完全重叠；TQueBind也不因绑定位置就与所有双缓冲机制互斥，需核对目标接口的depth/份数与同步合同。

缓存hint与选择性禁用若修改同一Tensor同一访问域，最终策略需一致；不同Tensor可以有不同策略。IBShare和L1预载共享容量时要明确分区/所有权，P78的TSCM属于L1，不是独立全局cache。常驻UB状态和分时区域别名也可以共存，只要寿命不冲突且最后消费者已完成。

Split-K若用不定顺序的浮点AtomicAdd通常不能承诺逐次一致，但分片写独占workspace后以固定顺序归并是另一实现，因此不能把所有Split-K与确定性硬互斥。推理也可能需要可复现输出。自适应tiling可按平台/属性选择，固定shape不等于所有选择无效；Matmul和elementwise都可能有物理尾块，不能凭族名略掉边界。

算法依赖应描述为数据/资源条件：预发射需要可安全复用的在途槽和完成通知；分核需要覆盖全部工作并处理尾轮；L1常驻需要明确合法分区。它们不强制依赖来源某一个策略ID。G维划分不一律依赖TND布局，手动Mmad不以MatmulImpl为先决条件，梯度Cube融合也不只一种高阶实现可承载。

UB峰值按同时存活对象求和：sum(align_up(物理元素数_i*sizeof(type_i),所需对齐_i)*份数_i)+scratch及固定资源。只有所有对象长度/类型/份数相同时，才可简写ubSize/BUFFER_NUM/pipeCount/sizeof(T)；尾部、广播与不同dtype应分别算。把元素数向32取整是特定元素粒度选择，不能当通用32B换算。只升一部分中间值到FP32时并非整条链翻倍；只将部分队列2份改3份时也不是全部UB加50%。

固定50%/75%“压力级别”、一律tile减半和“三缓冲+FP32几乎必极小tile”均不能代替实际容量验算。tile变小会增多调度/搬运次数，也可能改变复用和有效并行，需要测量完整算子。验证每个分支/尾轮、缓冲所有权、初始化和特殊值；未执行源策略组合性能复测，不把推荐搭配写成已验证高效组合。
