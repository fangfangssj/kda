---
type: Guide
title: Paged FA 组件布局与 ABI 修正
description: 保存三项关键修复，补齐GQA基址和声明宏问题。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-paged-handcraft-recipe.md
  role: primary
created_at: '2026-09-11T05:34:07Z'
updated_at: '2026-09-11T05:34:07Z'
platforms: []
---

# Paged FA 组件布局与 ABI 修正

该FA实例Q/K/V/P/O为half、S/OTmp/Update为float，K ColumnMajor、V RowMajor；MLA的V消费视图不同，不能直接套用。FAIQK/FAIPV<true,false>、EpilogueOnlineSoftmax与RescaleO三GemmType参数均为所选版本，真实Tail名来源明确FAITailQK/FAITailPV，不能把导航别名FAIQKTail当已存在符号。

QK在L1分配Q一份、K两份，128*128*2*(1+2)=98304B=96KiB；PV构造从L1_QK_SIZE之后分配可避免源实例重叠，实际资源还需PV/Tail/额外区总预算。不同组件共用相同区只有生命周期互斥才合法，不能只因构造传了offset就认定整体无冲突。

AIV softmax以P物理rowStride512写，PV必须用同stride读取即LayoutP(rowNum,validCols,512)；two-arg只在其默认stride恰好一致时等价。OTmp行距为alignUp(D,16)，O完整行距为Hq*D。S/P/OTmp虽然slot元素offset形式相同，但buffer基址与dtype不同。

GQA任务已算kvHeadIdx/qHeadIdx，Q/O显式加head偏移；源NO_MASK调用却直接传gK/gV基址且无kvHead入参。除非组件另有可证明的head视图来源，必须给K/V正确kvHeadIdx*D偏移，不能只传strideKV=Hkv*D就自动选中KV头。masked片段出现gmKOffset/gmVOffset也要真实定义。Hkv1测试会掩盖此遗漏。

若GM_ADDR是宏展开为__gm__uint8_t*，写GM_ADDR q,k会让后者不带星号，独立声明或真正指针类型alias才能一致；这不是C++禁止所有逗号声明。source Step2要求逐声明、Step9却再次给逗号列表，不能照抄后者。类内显式模板特化也需目标编译器支持或改为合法特化组织。

FA Params13个GM指针加entry hardwareSync共14参，Host顺序、POD FATilingData的uint64字段对齐及float scale位模式须一致。Host先GetFATilingParam再按结果分配四段，来源步骤先malloc尺寸后填tiling顺序颠倒。四段可在一块充分对齐allocation中划分并传独立指针，不必四次malloc；禁合并分配没有数学或ABI依据。
