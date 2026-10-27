---
type: Guide
title: MC2 tile数量、轮转槽与flag积压
description: 区分工作块数和在途容量，补齐tiling变化与正确性条件。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/shared/pipeline_tuning.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/operators/collective-comm.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/review-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow_integration.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# MC2 tile数量、轮转槽与flag积压

此类M切分参考无尾时headM=M/T，tileCnt=T是总工作块数，bufferSize=B是同时可用轮转槽数，二者独立。i&(B-1)仅在B正且2幂时等价i%B，并不要求T为2幂或T为B倍数。流水并发深度由槽、依赖和通信/计算资源共同决定，不能称T块就是T级同时在途。

T1仍可有LocalMatmul等内部重叠；T>1也不保证实际通信被完全掩盖。理想两阶段同成本c/d、T块时总时长约c+d+(T-1)*max(c,d)，还需增加同步/填充/排空/布局成本。增大T让消息变小、启动和flag更多，也改变Matmul调度、尾比例和缓存；相同数学不保证浮点结果逐位不变或多tile代码无错。

修改headM后应重算匹配路径的Matmul和通信tiling，核数/scale/Win偏移保持同源，并非每个字段必然数值变化。headM≥baseM或必须128倍数是具体实现限制，很多Matmul本身支持尾M，不能上升为通用硬件门槛。参考tail0路径确实需整除；其他有单尾/多尾和padding的路径按实际tail字段与独立tiling处理，不能机械拒绝所有非整除M。

flagId空间与每ID未消费事件积压是不同资源。来源3510说明16个ID、计数值0..15，不意味着T≤15；固定ID按轮Set/Wait可复用，但快producer若不受反向credit约束仍可能积压，不能仅凭各自循环代码紧邻推断全系统≤2。轮次直接作ID则需限制/轮转并避开相应SyncAll保留ID。API重载的mode、producer pipe和consumer pipe要表达真实依赖，并非两端PIPE必须字面相同。

Win预算数据B*headM*kPerRank*rankSize*d只是一个布局的数据部分，还包括scale、元数据、barrier、alignment、tail和其他驻留区。默认B4、headM512、rank≤2取512/rank≥4取128及512KB消息都是来源案例，不能直接设置跨平台强制上限；source本身也记录更大单轮稳定实例。对各合法候选重新验证T>1、槽绕回、重复launch和慢rank，再以端到端时间择优。
