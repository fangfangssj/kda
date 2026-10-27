---
type: Guide
title: Blaze SWAT Tiling 搜索与资源边界
description: Blaze SWAT Tiling 搜索与资源边界的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_mx_tiling_swat.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_tiling_base.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_tiling_common.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/tiling/quant_matmul_tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/utils/constant.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/common/host_utils/common_utils.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/matmul_blaze.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze SWAT Tiling 搜索与资源边界

每次GetTilingData清args/platform/run状态，查询当前AIC/AIV与UB/L1/L0/L2/BT，再CalcBasicBlock→OptimizeEdgeBasicBlock→CalcTailBasicBlock→CalcPathSpecificL1→序列化。初始M/N至多256并按转置对齐，K以FP8至多128或FP4至多256再64对齐；块数不足核数时重平衡MN并受L0A半容量推K，BASEK_LIMIT4095后缩减。此算法是候选启发式，不代表全形状最优或自动证明L0B/L0C可容纳。

L0C双buffer条件baseM*baseN*4*2<=l0cSize，否则选1，但仍需检查单份是否容纳；平台core数为0会参与取模/除法，不能只查baseM/N/K非零。边缘合并在K实际字节128对齐且A非转置/B转置时尝试4行窗口平衡小尾，m/nBaseTailSplitCnt与TailMain描述重排后大小，原块数不简单消失。

尾轮按较大边优先增加mTailTile/nTailTile，在tailBlockCnt*两split<=AIC且每维上限内搜索；这与外层通信tileCnt不同。usedCoreNum在多块通常取全AIC，在单块按拆尾数取值，通信目标映射需额外验证。

L1预算同时含A/B编码字节和各scale块；GetDepthA1B1首轮按2幂增长、次轮按512/256B K粒度调整。scale复用取两边可承担窗口的min，stepK上限4，输出nBufferNum固定2。搜索的depth>=1不自动保证baseL1Size<=L1；极小资源下减法u64下溢，step=depth/2可能0并进入后续除法，scaleFactorMax也可能0。必须在序列化前验证正step、正scale窗口、整数比例和实际峰值容量。

Host m/n/k公开字段u32，stepK/nBuffer/dbL0c为u8，独立调用Tiling接口无统一CheckUint32Shape保护；当前CLI正int不能替代所有调用域检查。FP4 GetSize=(logical+1)/2，GetShape=bytes*2，不能用于每行packed尾时只对总元素除2而忽略行padding；所有shape乘积先扩展类型再检查溢出。
