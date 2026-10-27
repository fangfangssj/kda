---
type: Guide
title: Paged FA 延迟 Finalize 与 Mask 限制
description: 保存排空和Tail调用，纠正任意Mask套Causal及无符号边界。
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

# Paged FA 延迟 Finalize 与 Mask 限制

普通路径4个128KV页组成最多512列stack，preLaunch2给3槽，QK/softmax当前stack生产P，PV/rescale消费两stack前。收尾需消费剩余真实stack；primary rescale不finalize，secondary仅当消费序号==totalStackSeqNum-1才除l写最终O，不能所有drain步isLast恒1。

一到多个stack、KV0、未满128/512尾、多个task复用须独立验证。若按无符号kvLoopNum-1判尾，0循环数会下溢；loop推进的slot与实际生产序号应对应，不能等待未生产P。源码AIV预置标10项实际11，AIC25项与骨架旧表18又不同，需目标组件全生命周期核验。

Mask0/1/2必须区别。源对所有maskType!=0套右下causal noSkip公式，却把MASK_SPEC解释为任意逐位mask；任意mask不能据query位置跳掉本可能可见的key，也不能假定只有对角块需要加载mask。只有明确压缩causal/band/prefix合同及正确索引才可结构性裁剪。

令qStart为query块起点、qEnd为有效行exclusive终点、delta=Sk-Sq。右下causal候选最大可见key exclusive上界为max(0,min(Sk,qEnd+delta))，逐行边界仍是k≤q+delta。delta和noSkip-qRows须在宽有符号域计算并夹紧；Sq>Sk时uint32差值与后续减法可回绕，让本应空可见区错误变成全KV或巨长区。

来源noMaskKvS/noSkipKvS正确区间应是[0,noMask)全可见候选、[noMask,noSkip)需mask、[noSkip,Sk)跳过，原文把最后区间说成对角块有误。全可见前缀可保守收缩，但不能漏有效key；masked宽度也不总等于qRows，尤其早期全/部分空行。totalStack=ceil(noMask/512)+1只有确有一个masked stack时成立。

noMaskTailS=prefix%128、maskedStart回退页、noMaskTailInteStackNum及延迟preLaunchStackNum的变步长是特定压缩布局索引。整数步长0可能需要靠stackSeqCount改变下一分支，必须证明整体循环进展和数据覆盖，不能只看kvIdx。Tail QK/PV多两个尾参、最后PVTail选择也须与这些枚举任务一一对应。

NO_MASK外层显式wait qkReady，masked epilogue可能内部wait；若外层也wait同一token会双消费，需核验实际签名和所有权。1024² half固定模板通常2MiB GM，不是任意大mask可直接无压缩复制到UB；1表示屏蔽实际上构造的是上三角指示，source“下三角mask”指可见域而非矩阵中1的区域。

该文件末尾明确Step8 mask未单独回归，标题NO_MASK全过不能为masked边界背书。有限-3e38与全mask/NaN策略、类型与scale范围另须定义，不能因带Tail组件就宣称全场景一次通过。
