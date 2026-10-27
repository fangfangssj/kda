---
type: Guide
title: MC2 流水递推与错误配平剪枝
description: 保存非均匀切分搜索，修正会排除全部非零尾块的约束。
tags:
- task.optimization
- topic.performance.analysis
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-perf-optimize/references/comm-compute/pipeline_balancing.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/mc2/pipeline_balancing_design.md
  role: supporting
created_at: '2026-09-11T06:22:14Z'
updated_at: '2026-09-11T06:22:14Z'
platforms: []
---

# MC2 流水递推与错误配平剪枝

两阶段按相同块顺序执行时，可直接递推：第一阶段完成A_i=A_(i-1)+a_i，第二阶段完成B_i=max(A_i,B_(i-1))+b_i，初始0，最终B末为总时间。通信→计算时a_i为通信、b_i为计算，计算→通信时交换。该模型假设各阶段内部串行、不同阶段可并发且无其他资源竞争，启动/网络开销应纳入a_i/b_i。均匀n块时间a+(n-1)*max(a,b)+b。

来源α=通信us/row、β=计算us/row、R=β/α。它用ρ=max(R,1/R)且tail=M%hs，再要求tail≥ρ*hs；因0<tail<hs且ρ≥1，该剪枝会排除所有非零尾块。不能把剪枝后候选叫完整可行域；“短块通信掩盖长块计算”与计算bound的角色也颠倒。计算快慢与哪端有额外local工作决定相邻块应满足的覆盖方向，需从递推/时间线导出。

例如计算→通信且计算bound，希望末短块的计算覆盖前长块通信，条件β*Mshort≥α*Mlong，即Mshort≥Mlong/R，允许短块；通信bound在前短块通信覆盖后长块计算时α*Mshort≥β*Mlong。通信→计算且有前置LocalMatmul时，初始第二阶段可用时间受local计算影响，不能用只看bound的固定前/后表覆盖所有情况。

来源均匀模型a+(n-1)*max(a,b)+max(a,b)在通信更慢时尾项错误，通算前后顺序的若干非均匀公式也混用了端点。用通用递推对0/1/2/多块、两执行顺序、无尾/有尾/Local前置分别核对，再作为候选排序代理；不将单一线性公式当实测。

候选记录真实M块列表、每块对应Matmul tiling的baseM/N/K、L1/L0与实际核数。按ceil(Mblock/baseM)*ceil(N/baseN)估可调度块数量，min(totalBlocks,C)/C只是可用核比例，不是真实MAC利用率、最后一波效率或性能硬下界。随着Mblock改变tiling可能变化，不能用一份全局baseM/N评估全部块。

TAIL_MIN128、80%/50%用核阈值、tcCap8/16、step16与Top15都是来源启发式，可能排除有收益方案；M行按16取整不是16字节对齐。候选范围无解时应保留合法基线/扩大合理候选，而不是空列表当成功。msprof实测Top候选有助比较模型偏差，但不必把必须全部搜索/CLI形式迁成知识强制流程。

比较时保持总M不变：来源示例表512*4+256=2304，而576*3+272=2000、640*3=1920，不能把这些行当同一case的性能对比。长短块独立tiling、通信粒度膨胀、状态重新初始化及cache效应都计入完整时延。
