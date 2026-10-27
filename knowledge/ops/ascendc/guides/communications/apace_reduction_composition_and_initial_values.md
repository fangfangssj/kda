---
type: Guide
title: APACE 归约组合与远端累加初值
description: APACE 归约组合与远端累加初值的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/communication.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/paradigm-mapping.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE 归约组合与远端累加初值

固定来源只为AllToAll GET/PUT和AllGather PUT给出已注册hook；ReduceScatter枚举预留、AllReduce枚举缺失是该版本状态。新collective需要语义、CRTP四hook、分发特化与tiling/布局消费者闭合，单独新增枚举或一个文件不构成可用实现。后续版本是否已提供需独立定位。

来源提出WriteReduceNbi远端原位SUM/MAX/MIN路线，以及AllToAll→本地ReduceSum→AllGather分解路线。前者需要匹配dtype/reduceOp支持、目标唯一初值、所有来源完成和输出归属；不能从任意remote reduce到任意window直接推导完整AllReduce。后者需保证接收者聚合对应同一输出坐标的各rank分量，再将互补分区收集到各rank。

独立tile槽只防不同tile重叠，不提供归约单位元。SUM新分配槽如果未清零仍把垃圾加入；MAX/MIN需合适初值或明确第一贡献覆盖。采用首轮WriteNbi覆盖后其余WriteReduce时，还必须让所有后续累加都晚于覆盖，跨来源竞争不能凭本地提交顺序解决。复用槽还要上轮consumer完成与新epoch初始化。

低比特通信需共同scale/zero-point解释或先反量化到兼容域，不能直接相加不同scale编码值而称等价原始AllReduce。requant额外误差、溢出、编码metadata与分段同步都进golden。来源“通信量减半以上”与路线演进成本低属于特定配置设想，未作为本次量测结论。

WriteValueNbi可写远端状态，AtomicFAA/CAS可建计数握手，但单个原子也不自动发布之前数据或解决epoch回绕；需实际内存顺序和资源注册保证。调用返回值、完成语义与跨rank参与者必须成套验证。
