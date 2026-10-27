---
type: Guide
title: MoE 样例输出顺序、有效尾与 UB 预算
description: MoE 样例输出顺序、有效尾与 UB 预算的代码事实、前提与失效边界。
tags:
- task.implementation
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/moe_dispatch.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/mte_dispatch_comm.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/include/utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/include/tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/test/test_moe_dispatch.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/scripts/verify_dispatch.py
  role: supporting
created_at: '2026-09-11T07:32:48Z'
updated_at: '2026-09-11T07:32:48Z'
platforms: []
---

# MoE 样例输出顺序、有效尾与 UB 预算

在localE=1/topK=1支持域内，SendTokens按linear route递增并以同expert prefix选slot，receiver按srcRank再slot回搬。expandX是有效token紧排，expandIdx独立存三个int32；epRecvCounts是一维按source裸count，expertTokenNums是单expert总count，不是prefix。FinalizeCounts注释提到清无效尾，但实际只写统计；输出尾零来自Host预先把整输出清零，复用未清输出会破坏脚本的全容量比较。

tokenQueue深度1逻辑大小record；三个int32路由buffer加一个float work各为AlignUp(BS*K,8)*4；epRecv输出UB AlignUp(EP,8)*4；expert输出64B；comm写状态32B、读/清各localE*EP*32B。当前localE1下逻辑合计record+16*AlignUp(BS*K,8)+4*AlignUp(EP,8)+64+64*EP+32，还需各InitBuffer实际分配粒度/框架开销，不能只拿payload估UB。

maxRecvTokens虽被读入但未用于CopyWindowToOutputs边界，localE计数数组和rank数组也没保护非法route。GetDispatchDataAddr按静态最大BS stride寻址，与按实际recv count累加紧排输出是两种坐标。多expert扩展必须重新实现expert→source→slot遍历及相应统计，不能用source所有expert计数回搬expert0区。

CeilDiv/AlignUp/FloorAlign在b=0分别返回0/a/a仅是helper行为，不能把非法除数当合法形状；a+b-1可溢出。ExpandIdxCount计算存在但不负责分配或验证GM容量。源的未使用srcAddr/inputTensor中linearIdx*h不应当成实际已执行读；真正WritePackedToken读取tokenIdx*h，topK泛化应审实际使用路径。
