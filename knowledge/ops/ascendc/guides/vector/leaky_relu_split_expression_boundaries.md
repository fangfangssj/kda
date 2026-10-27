---
type: Guide
title: LeakyRelu 分解表达式与尾块边界
description: 保存BF16升精度链，限定分段等价的特殊值范围。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/leaky_relu/leaky_relu_tiling_example_for_lowering.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/leaky_relu/leaky_relu_init_example_for_lowering.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/dsl-lowering/references/lowering_examples/leaky_relu/leaky_relu_process_example_for_lowering.md
  role: supporting
created_at: '2026-09-11T05:05:12Z'
updated_at: '2026-09-11T05:05:12Z'
platforms: []
---

# LeakyRelu 分解表达式与尾块边界

来源LeakyRelu计算pos=max(x,0)、neg=min(x,0)、out=pos+alpha*neg，BF16输入升FP32，末尾以受支持的最近偶数转换回BF16。对通常有限实数与有限alpha符合分段LeakyRelu，但NaN、signed zero、Inf以及alpha非有限时Max/Min和乘加链可能与条件选择不同，例如有限正x、alpha=Inf使0*Inf为NaN，直接正分支应保留x；特殊值按实际合同验证。

该分解不同于简单max(x,alpha*x)，后者对alpha<0、alpha>1等并非普遍等价。来源Attr negative_slope默认0.01，并未证明只允许某个alpha区间，不能靠固定默认值裁剪合法属性。中间FP32运算及最终Cast有舍入，不能把相同数学式称bit-exact。

I/O BF16队列两份共4T字节，FP32输入/输出及pos/neg四份共16T，合计20T字节，T2048为40960B=40KiB，另计对齐和运行资源。队列建立搬入/Cast/计算/降精度/写回交接，pos/neg的同pipe依赖与复用也需合法；可按生命周期优化buffer，但不能释放尚有消费者的输入。

Process对统一innerLoops始终读写tileSize，没有按myElems缩短尾块，即使pivot分核准确仍会越过当前片段并产生跨核输出覆盖。普通BF16DataCopy还需该重载长度和地址要求，GM起点不一定因tileSize2048就对齐。正确范围是每核off<myElems、cnt=min(tileSize,myElems-off)，所有Cast/Vector/写回共享cnt。

原DSL也未给load/store尾mask，不能把这种遗漏机械迁成完整Ascend C实现。tiling/init/process分别是阶段示例，前两篇保留TODO，仅三者合同一致且补齐上述边界后才具备可运行实现的基础。
