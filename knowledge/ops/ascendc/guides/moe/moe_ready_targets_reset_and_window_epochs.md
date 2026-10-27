---
type: Guide
title: MoE ready 目标、状态复位与窗口轮次
description: MoE ready 目标、状态复位与窗口轮次的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/api-rules/index.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/reading/design-overview.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/tiling-scheme/index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/dispatch-dataflow.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/combine-dataflow.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE ready 目标、状态复位与窗口轮次

dispatch每目标expert/source槽发布完整tokenCount与ready；本source给某expert零token时，如receiver仍等待该槽，也要发布零计数完成。不同发送核的局部值要先汇总或按完整路由重算；发布者不能在别核payload未完成时仅凭自己发完就写全槽ready。

源用sum(flags)==状态槽数或K+sharedExpertNum判断齐备，只在flags严格0/1、各slot当前epoch有效、等待集合恰好所有应产生项时等价逐slotready。旧epoch、重复累计、NaN/非法flag或mask改变集合都会破坏等价；例如[0,2]总和2并未都到达。零/copy/const专家无远端发送时必须从等待集合剔除或提供等价本地ready。

payload、count/offset等元数据先完成并可见，再发布ready。source文档将“等待后立刻清状态”与“消费后复位”混写：提前清ready若被producer当可重用信号，会在consumer读payload期间覆盖；只有另有slot生命周期隔离、epoch或ack保证时，清ready可早于读取。不能用函数名ClearLocalStatus证明旧数据不再被访问。

初始状态应在尚无远端producer访问的阶段正确初始化。Host aclrtMemset与kernel Init早清可能竞态，不能随意追加；SHMEM库内部同步区自动初始化不意味着用户window自动清零。自管理状态、library内部状态和data buffer分别确认范围与owner。

四区dispatch0/1、combine0/1各自切换可隔离相邻阶段，但N+2重用仍需证明N消费者全部完成。仅每dispatch存在某个全卡同步不足以自动证明此条件，需标明barrier位置、跨kernel顺序及所有rank同epoch。故旧overview“进度基本一致”不能当完整证明，延迟rank、空batch与两次以上绕回是必要探针。
