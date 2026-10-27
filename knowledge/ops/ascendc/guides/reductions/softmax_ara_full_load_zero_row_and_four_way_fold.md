---
type: Guide
title: ARA FullLoad的零填充行与四路折叠
description: 保留8行压缩和Vector存储加载顺序，解释尾偏移哨兵。
tags:
- task.optimization
- topic.precision
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/dav310/softmax_v2_ara_full_load.template
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling.template
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-performance-best-practices/references/reduce/templates/softmax_v2_tiling_data.template
  role: supporting
created_at: '2026-09-11T07:01:45Z'
updated_at: '2026-09-11T07:01:45Z'
platforms: []
---

# ARA FullLoad的零填充行与四路折叠

ARA FullLoad每个任务处理一个A1和一片A0，R全载：GM起点=a1*R*A0+a0Tile*tileA0，行stride为真实A0；输入双buffer，输出双float buffer，xTmp含R行exp再加一行零padding，xReduce一行sum。输出的float队列在后期可写压紧的T2数据，必须保持被覆盖的临时归约结果已不再使用。

R≤2直接累加，3..4/5..8分别与缺失行的零哨兵做树式Add。R>8取严格小于R的2幂P，主/余数配对8行压到P/8行，再BinaryAddVF每级四行合一，必要时最后两行合一。binaryAddK、Last和InnerLoop由P/8推导；每级用LocalMemBar<VEC_STORE,VEC_LOAD>保障当前写出的中间值供下级读取。

尾偏移超出有效R映射到validNumInXUb=R*tileA0的额外零行，并在该VF列块开头显式清零。源码pick判断row>tailCount而不是>=，row==tailCount的算术地址已经恰好等于validNumInXUb，因此此处不能只看比较符就认定多读一行。物理额外行与清零/屏障不可删除；普通DataCopyPad isPad=false不替它初始化归约中性值。

VFShiftVector/VFReduceSum/Output把R和curTileA0转uint16，Host需要限制而非仅靠UB估计。a0基块在Host实际64/128元素，模板还含精确尾mask；资料把其概括为8/16元素是宽度对齐惯例的混用。R非2幂、R=3/5/9/17及a0尾覆盖主配对、部分配对和零行路径。
