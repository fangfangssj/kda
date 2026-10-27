---
type: Guide
title: GQA 任务几何与 Split-KV 并行条件
description: 区分 head 分组、查询行、物理 stride 和可用核数。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/design/shape.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/foundation/composition.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/flashattention/subfamilies/base_design.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# GQA 任务几何与 Split-KV 并行条件

GQA中G=Hq/Hkv要求Hkv>0且整除；实际计算行数mEff=curBq*curG，容量规划通常使用Bq*gMax，不必一次包含全部G。分组融合可改善K/V复用，但不是数学正确性的必需条件；curG=1或分组切分仍可计算相同输出，只是资源/性能不同。

一个明确任务网格为B*Hkv*ceil(Sq/Bq)*ceil(G/gMax)，每任务保留query/head起点和curBq/curG尾数。来源r=g*Bq+i是head外层、query内层映射，不能因文字称Bq-major就改变公式。行间使用名义Bq还是curBq作stride，必须在Q加载、mask、P/V与输出解码中一致。BSND输出元素偏移为(((b*Sq+q)*Hq+h)*Dv+d)，query行步长Hq*Dv；gOff须明确按head、行还是元素计量。

DataCopyPad参数长度若接口单位为字节，D需乘dtype字节；示例{1,D}不能跨重载复制。Dqk决定QK归约，Dv决定PV输出列，两者无需相同。MHA是G=1的数学特例，不要求源码与GQA完全同一分支。

Split-KV按K/V区间生成局部最大值、指数和与加权V，再用全局最大值重标度合并；局部区间内部仍可使用Cube和在线softmax，不是Vector专属。空/全mask区间需中性状态，不能计算-inf-(-inf)后把NaN传播给合并。

是否缺少并行应先比较原始任务数与物理可用AIC数。若usedCoreNum=min(totalTasks,aicNum)，条件totalTasks<usedCoreNum永远不成立，不能作为启用Split-KV的判据。Sq=1或Sk>2048是场景启发，已有足够任务时也可能因其他瓶颈受益，仍需计额外workspace、归约、尾片和launch开销。

Vector FA快/慢10–100倍、固定__mix__(1,2)以及统一增加AIV均无通用证据。增加AIV可能争用UB/带宽或改变同步参与者；更多AIV对应AIC:AIV数值比降低，不能把比例名称与方向混淆。
