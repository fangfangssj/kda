---
type: Guide
title: 手动双缓冲、三缓冲与二维槽的游标合同
description: 区分任务游标、物理存储位置和事件所有权，补齐初始化、延迟阶段与释放闭环。
tags:
- task.implementation
- topic.memory
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P28_hardevent_producer_consumer.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P19_custom_pingpong_double_buffer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P20_triple_buffer_rotation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P21_matrix2x2_buffer_policy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P28_hardevent_producer_consumer.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P19.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P20.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P21.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P28.yaml
  role: supporting
created_at: '2026-09-11T01:44:57Z'
updated_at: '2026-09-11T01:44:57Z'
platforms: []
---

# 手动双缓冲、三缓冲与二维槽的游标合同

手写 Buffer Policy 可以分别管理分配、消费、回收及跨迭代复用，但取槽操作不建立完成关系。双缓冲 Get 翻转当前索引，GetPre 返回相反索引；在第一轮，前一槽可能尚未生产。GetReused 使用独立游标也不意味着两路访问独立物理空间：两游标仍选同一对 buffer，必须协调所有权。初值取 false/0 是一种约定，其他初值只要整套映射一致也可成立。

三阶段按逻辑任务 t 分配 Load(t)、Compute(t-1)、后处理(t-2) 时，稳态槽分别可用 t%3、(t+2)%3、(t+1)%3，但任务号和有效范围必须先判断，不能只对负任务取模。来源注释把后两者落后轮数写反。若三个游标都从 0 开始且每轮同时加一，它们取到同一槽，不会自动形成三个任务的重叠。跳过一个阶段时要按真实任务身份推进，而非为凑每轮一次调用强行移动游标。

二维 M×K 槽可用四个物理区保存两组 M、两组 K，并给 Alloc/Reuse/Free 独立计数。来源 GetBuffer 计算 `m=(m+mExtent-1)%mExtent, k=(x/mExtent)%2`。mExtent 初值为 0，使用前必须设置且与实际数组匹配；四槽实现的 mExtent 不能无界扩张。mExtent=2、初始 m=0 时序列为 (1,0),(0,0),(1,1),(0,1)，调用者必须采用这一真实顺序，不能凭“列优先”名称假设从 (0,0) 开始。buffers_ 的实体及初始化也需补齐。

TBuf 的模板位置决定实际物理存储。把所有实例都声明为 VECCALC 不会因变量名叫 l1/l0a/l0b 就获得 L1/L0A/L0B；Cube 输入输出还要满足其合法存储路径。若 TBuf::Get 返回 LocalTensor 值，不能返回对该临时值的非 const 引用；可返回句柄值或持有生命周期明确的句柄。Size 是字节数，模板实参还须为编译期常量，运行时 tileSize 不能直接用于 static constexpr。

每条数据边既有 ready，也有最后消费者结束后允许复用的 release。对源 P28 的单槽例，MTE2→V 与 V→MTE3 描述搬入后计算、计算后搬出；若下一轮 MTE2 覆写同一物理区，仅等待 MTE3_V 不能直接证明 MTE2 已被门控。输入输出分区不同时又可能有不同释放时刻。MTE1_M 用于相应 L0 输入发布给 Matrix，M_MTE1 对应 Matrix 消费后允许 MTE1 复用输入；L0C 交给 Fixpipe 则要核对 M_FIX 及反向复用关系，不能把 M_MTE1 当作 L0C 已写出。

首次生产前不要等待不存在的 ready；末尾应排空已发布的真实任务。loopCount=0 时不能无条件等待最后 CopyOut。多级前移两轮的循环必须有启动和排空分支，不能只循环原始任务数后退出。异步消费者全部完成之前不能重新分配、改写槽或释放其事件资源。

事件类型与 ID 是不同维度：EVENT_ID_X 不是另一类“软件事件”，HardEvent 也不是零开销的保证。容量、ID 范围及组件占用按具体执行模式与 SDK 处理，不能统一规定总共16个或每 buffer 恰两个。核内事件不因核号相同就跨核冲突，不应一律加 blockIdx*eventsPerCore。跨核通知另有作用域，AtomicAdd 也不能替代一般数据发布协议。

事件可以按阶段或槽分配，前提是所有尚未消费的通知不会混淆；仅看静态 Set/Wait 数量、固定常量或把 Wait 放在源码 Set 后面不够。减少 buffer 后也要重建此协议，不能只更换取模基数。纯 Vector 多阶段也可评估三槽，多种深度可用于不同存储边；收益由真实重叠和资源成本决定。

验证 0/1/2/3 个任务、尾块、跳块、超过槽深度多轮、不同 M 范围和最终输出发布。uint32 游标回绕后 `%3` 的周期会在边界断裂，换成同宽 loopIdx 仍不解决；可维护有界环形状态和足够宽的逻辑任务计数。
