---
type: Guide
title: APACE AllGather 三区与输出归属
description: APACE AllGather 三区与输出归属的来源实例与适用边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/put-all-gather-quant-matmul/development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/operator-anatomy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/scenarios/index.md
  role: supporting
created_at: '2026-09-11T07:21:06Z'
updated_at: '2026-09-11T07:21:06Z'
platforms: []
---

# APACE AllGather 三区与输出归属

来源PUT AllGather每rank持A_r[m,K]，收集所有A沿M拼接；B_r[K,N]在各rank可不同，所以C_r=concat_M(A_s)@B_r的shape是[R*m,N]且不同rank输出未必相同。不能把AllGather误当AllReduce或复制A2A的本地[m,N]输出分配。B和scaleB按DN实际落盘解释。

固定实现将计算逻辑分HEAD本地已就绪数据、MAIN各轮R-1个remote fragment、TAIL各rank有效尾行。每rank真实m=headTileCnt*tileM+tailM，逻辑padding后总M=R*(headTileCnt*tileM+tailCnt*paddedTailM)，paddedTailM按16对齐。HEAD和MAIN仅分配各自有效主块范围，TAIL含本地；累计真实输出仍是R*m行，不能把padding写进用户输出。

MAIN首次命中才BuildMainFragment，换轮UpdateAddrList；旧消费者完成后才能改其持有地址表，且新远端段须就绪。QmmMxBlockMmadFragment消费A/ScaleA/C视图，B/scaleB/bias普通Tensor。SetL2Cache中scaleB有fullMBlock和alignment双条件，不能只拷A2A的一个条件。

Host按聚合的逻辑总M生成mm tiling，并在来源AG使用SetOptimizeEnable(false)+SetMTailAlignEnable(true)。scheduler tail拆分需检验(endBlock+1)*mTailTile*nTailTile<=实际block数才更新；乘积也需防溢出。WINDOW_LEN宏局部定义为1再include再undef是该版本调度器配置技巧，受头文件include guard及其他定义影响，不保证任何编译顺序均生效。

三个固定数组cFragAddrs/winDataRankBase/winScaleRankBase容量8限制此kernel的R<=8；FragmentTensor通用32上限和context64上限不能覆盖这个更小界。扩R需所有数组、视图容量、计算核/通信job/标志预算一起满足，不是改一个数组即可。
