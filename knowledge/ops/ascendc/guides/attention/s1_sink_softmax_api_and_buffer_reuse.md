---
type: Guide
title: S1 Sink Softmax API 与缓冲复用
description: 保存播种法和状态写回，纠正Tiling配置及行列循环混淆。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-sink-handcraft-recipe.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/activation_functions/soft_max_interface/softmax_flash_v2.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/activation_functions/soft_max_interface/softmax_flash_v2_tiling_interface.md
  role: constraint
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# S1 Sink Softmax API 与缓冲复用

分母型sink可作为初始m=sink[head]、l=1、未归一化o=0，再用SoftmaxFlashV2 update处理实际score；l1是exp(sink-m)=1，不是额外乘exp(m)。每组新query行只播种一次，跨KV chunk继续更新，不能每chunk重复加入sink。无sink可首轮使用非update或正确空状态，不能令m取任意sink占位值且忽略全mask行。

来源调用支持src/dst、max/inMax、sum/inSum同址的候选，但不等于所有tensor/scratch都可互相别名。已有API卡约束isReuseSource为预留默认false，状态末轴通常32B；source启用isReuse=true并把expMax按rows*64分配不能推成通用必需布局。应按实际API/config的广播模式和最小scratch查询确定。

来源反复强调Tiling isUpdate=false而Kernel模板true；与已有Tiling API卡要求isUpdate和isBasicBlock两侧一致冲突。不能将此经验当正确固定配方，本批未读外部归档实现，应使用相匹配的接口配置或取得该旧版本特例证据。ShapeInfo需同时表达逻辑shape、物理pitch和有效size，不是任意二维声明即可修复数值。

ProcessVec1文字称realSplitN=ceil(s2Real/列预算)，代码却以loopIdx*vecS1Base切query行，并计算s1Real-loopIdx*vecS1Base；这实际要求按行分批的loop数。若真按列拆，则需同一query的online更新及alpha接续，不能沿用这些行偏移。vecS1Tail、S读址、P写址及max/sum槽必须同一解释。

maskPse分别加载且遵循PSE外乘/内加合同，scale与mask先后不能任意换。尾列显式mask，compressed-mask快路径也要确保所有padding列已无贡献。hasSink时maskPing被复用为exp输出，需在mask最后消费者结束后才能覆写；相应V_MTE2释放移后是该共享生命周期的一部分。

来源UB命名区11+11+16+32+32+16+8+8+8+32=174KiB，仍需实际分配对齐及其他框架区。max/sum各8KiB攒多批行后写GM，copyOutLimit必须正且不越容量；Vec2从相同batch/head/query偏移读回l，不能依赖已被复用的UB旧状态。

exp输出中alpha位置应按真实合同读取，若自算exp(mOld-mNew)，旧max在update覆写前保存并保留至延迟消费者。P降精度而sum来自FP32指数是一种近似路径，不能据sum相同就证明O bitwise一致。来源“手写softmax慢430倍”仅为历史失败实现，不是API以外算法均不可用。
