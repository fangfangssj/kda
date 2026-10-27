---
type: Guide
title: Blaze 两阶段初始化与空闲路径
description: 纠正成员析构安全推断、SyncAll 默认假设和 tuple 特化重名，保留全参与者初始化。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv-sync-and-two-phase-entry.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv1-v2-adaptation-for-softmax-reduce.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-design.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-development.md
  role: supporting
created_at: '2026-09-11T03:01:48Z'
updated_at: '2026-09-11T03:01:48Z'
platforms: []
---

# Blaze 两阶段初始化与空闲路径

若基础Kernel的idle核在Epilogue.Init前return，把全参与者SyncAll放在该Init内会导致参与者不可达。可在所有所需参与者都经过的外层阶段先初始化共享workspace，完成初始化写入与同步后再进入CV1；CV1最终drain后再建立V2读取依赖。是否需要所有核、仅AIV或别的集合由真实API、调度和生产者决定，不能把无参SyncAll注释为自动“全部AIC+AIV”。

Epilogue是否为栈对象或类成员不决定析构是否安全。真正条件是本次对象是否完成事件初始化、进行了哪些生产/消费，以及析构等待是否存在对应发布。initialized_能保护未Init路径，但Init部分失败/条件退出仍需按状态处理。来源整个生命周期“set=wait+1”表述错误：预发后暂时多1，最终drain后应按协议平衡，不能留着多余信号跨期污染。

MTE3从UB读取并写GM，MTE2从GM读取并写UB；源码把它们描述为MTE3写UB或MTE2读UB不准确。MTE3→V可能保护下一Vector覆盖同一UB，MTE3→MTE2可能保护同一GM更新后的重读或UB再写，需追实际地址。局部事件、CV通道和阶段屏障是不同资源，不由一个析构动作全部完成。

初始化骨架以cubeCoreNum*R个状态元素分配给AIV，但raw GetBlockIdx与逻辑AIV rank的关系必须由具体MIX入口确定，不能仅按GetBlockNum*GetTaskRation推导稠密覆盖。每个状态元素恰好初始化一次且先于任何V1访问；分核尾部的GM起点和Copy范围也需支持。初始化UB使用后在CV1重用前需完成在途搬出。

骨架用0xFF800000、B32 store和固定128元素暂存，只适用于确认的FP32位表示及向量宽度；不能保留ComputeType模板名便认为任意dtype可用。完整向量store的物理覆盖可能大于逻辑128，需按实际VL计容量/mask。通过普通指针reinterpret读取浮点常量还需符合设备语言与别名合同，不能把示意强转视为通用构造方法。

外层tuple<V1,V2>只描述两阶段组件类型，V2可能采用外部行slice调用，也可能由Init+ReduceAll自行读多buffer；替换接口后Params、row映射与workspace必须同步。删除外层行分配不等于不再需要行覆盖证明。多个同模式GemmUniversal偏特化同时可见可能重定义或歧义，仅重命名文件/trait而不参与约束不能区分；需互斥的真实类型约束，具体编译诊断取决于声明形态。

验证active/idle、未Init与正常析构、无本地行仍承担通知的lane、首末期、不同producer数、连续launch和最终V2输出。来源的mode/flag偏移及每tile等待数量仅供核对实际库，不作为无条件插入模板。本批未声称骨架已编译或上板。
