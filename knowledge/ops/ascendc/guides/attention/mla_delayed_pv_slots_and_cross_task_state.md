---
type: Guide
title: MLA 延迟 PV 槽位与跨任务状态
description: 保存全局和局部PingPong、前取与事件所有权。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-paged.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-mla-handcraft-recipe.md
  role: supporting
created_at: '2026-09-11T05:34:07Z'
updated_at: '2026-09-11T05:34:07Z'
platforms: []
---

# MLA 延迟 PV 槽位与跨任务状态

来源通用MLA先prologue QK0→softmax0而无PV，随后n1..nLoop生产QKn/softmaxn并消费PV(n-1)/rescale(n-1)，最后可预取下一有效task的首块以延续流水。该调度匹配所选组件；单块锁步也只有重新设计完整事件和资源协议后才能替换，不是数学attention只能延迟PV。

全局pp跨task保留，S元素offset=core*32768+(pp%2)*16384，P=core*65536+(pp%2)*32768，OTmp=core*131072+((pp-1)%2)*65536，OUpdate=core*65536。生产与消费对同一slot达成一致，pp0之前不能取pp-1；locPingPongIdx是组件内部另一个状态，taskPingPongFlag保护跨task统计，不能全部统一为一个奇偶变量。

lookahead要找到下一真正有效process并保存其batch/head/split/首page身份；跳过空任务不能遗漏首块，已预取的task进入主循环也不能再prologue一次。下一task softmax可能早于上一task最后rescale，m/l及alpha必须按task槽隔离，glFlag[2]真实初始化与组件消费协议一致；非null数组不是通用替代所有状态初始化。

QK从blocktable[realBatch*maxBlocks+startKV/128+n]解析latent和Rope各自stride。PV来源调用不显式传V GM，依赖专用组件保留的latent/K资源关系；换通用PV必须补齐数据来源，不能假定省掉指针后仍自动读到正确V。相邻task输入不同不得无条件重用旧L1。

来源AIC预置8+2+8+6+1=25、AIV3+7+5=15，实际预置/析构和排空顺序需匹配组件；“反序”文字与所列V_MTE2 drain4,0,1,2,3不是简单逆序，应按依赖核验。裸CrossCoreWait与Set<0x2,PIPE>配对可能由封装默认语义实现，不能简单比较模板文字数字判不配对或适用于所有版本。

资源对象只构造一次、先读取tor再构造Epilogue；源片段在声明tor前已使用它，说明仍是有省略的示例。AIV logical core=blockIdx/subBlockNum是该混合任务映射，原文“裸编号0与20都映0”并非这条公式本身推出，应核对真实编号范围。宏分流、同步基址和Host核型也需与真实编译结果匹配。
