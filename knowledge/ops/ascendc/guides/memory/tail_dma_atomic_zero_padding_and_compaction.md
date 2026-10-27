---
type: Guide
title: 尾部 DMA、原子补零与压紧
description: 保存DataCopyPad及GatherMask，补齐合法GM范围和Mask宽度。
tags:
- task.optimization
- topic.memory
- topic.implementation.tiling
- topic.implementation.synchronization
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_11_tail_block_handling.md
  role: primary
created_at: '2026-09-11T05:55:38Z'
updated_at: '2026-09-11T05:55:38Z'
platforms: []
---

# 尾部 DMA、原子补零与压紧

Foreach主块使用对齐DataCopy，尾块通过DataCopyPad按真实byte长度搬运。copyTimes须包含余数块，源算tempDataCount后调用却用dataCount变量，必须实际传本块有效count；pad=false没有承诺padding内容为0，计算mask只覆盖有效元素或显式初始化需要的中性padding。零元素、恰整除和小于一个block分别处理。

某AdaptiveAvgPool尾部使用Duplicate清无效lane、SetAtomicAdd写对齐cTailAlign再恢复。此方案仍要求整个写入byte范围都在合法GM allocation内，atomic并不使越界访问安全。若有效值按加法输出，目标初值应为0或算法已有partial；相邻值加0还涉及NaN/有符号0等语义，不能把任意已有内存视作无影响。精确长度DataCopyPad已可能正确，source声称baseline无尾处理却展示DataCopyPad，不能据标题否定。

尾mask=(1<<numPerBlock)-(1<<validDataLen)仅在0≤validLen≤numPerBlock且移位小于操作数bit宽时合法；完整字宽移位需特殊构造。mask lane类型与Duplicate重载匹配，清零的是无效源lane而非保证目标初始化。

GatherMask可将错位有效数据压紧到连续UB再输出，pattern类型float常uint32、half/BF16常uint16仍须实际重载定义。preLeftShift=numPerBlock+lastLeftShift应检查不越pattern位宽，rsvdCnt与输出长度一致，原地src/dst重叠需支持；压紧本身不定义最终DMA尾字节合法性，也不自动消除全部原子需求。

DynamicBlockQuant把列处理分为无padding连续搬、整block间隔搬和DataCopyPad显式补齐。DataCopyParams blockLen/gap通常以相应块为单位，Ext blockLen按byte、pad数量按元素，各字段范围要在窄化前验证；rowCount转uint16、colBytes转uint16及pad转uint8都可能截断。colPadExt/colPadToBlock/colClearExt分别表达物理间距与需要初始化区域，不可混用。

“全部GM访问必须32B对齐”过宽，实际重载允许有效字节尾写等不同形式；GM逻辑长度和UB物理对齐应分开。RMSGrad DataCopyExtParams以calcLen为行数、colVal*sizeof(T)为每行有效byte，dst起点calcLen*colValAlign仅在目标分区已为其保留空间时合法，padParams{true,0,0,0}不代表所有完整padded行都已清零。
