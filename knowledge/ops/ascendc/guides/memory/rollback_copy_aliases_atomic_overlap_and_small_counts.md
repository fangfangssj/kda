---
type: Guide
title: 回退搬运的别名、原子重叠与短长度
description: 保留老平台尾部搬运思路，明确回退并非天然安全。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_07_data_alignment.md
  role: primary
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-09-11T06:15:28Z'
platforms: []
---

# 回退搬运的别名、原子重叠与短长度

Foreach在Local→Local复制时向上取整到32B，只在源/目标实际buffer均容纳对齐量且多余lane可读取/覆盖时合法；不能把同样策略用于没有额外分配的GM边界。DeepNorm老路径DataCopy(...,ROUND_UP(size))可能读取或写出有效GM范围，必须有合法物理padding或另一安全tail方案；新DataCopyPad路径的宏适用性需按真实编译器/SoC核验。

InplaceAddRMS自定义尾搬运先处理完整num，再从count-blockElems复制一整块到UB头部，写到dst[count-blockElems]，用重叠最后块覆盖剩余。count<blockElems时减法下溢；count0须直接处理，不能以有tail就强行回退。回退目标起点是否被该方向/平台DataCopy允许也需证据，并非长度对齐便足够。

第一次DataCopy若仍在MTE3读取src，Scalar SetValue改写UB头部可能竞争，需完成交接；搬入方向则要在读取src已就绪后再做Scalar复制。源参数const视图不保证底层不可变，修改src会影响后续其他消费者，必须确认最后消费/专用scratch。逐元素回填还有重叠顺序问题，需像memmove一样证明读值未被提前覆盖。

TransQuant SafeDataCopy按DataCopyPad支持且sizeofT<8选择新路径，否则对齐或rollback。来源rollbackDstIdx=numAlignedBlocks-blockElems同样在总量不足一块时下溢，rollbackSrcIdx未定义于片段，不能原样运行。forAtomicAdd=true的描述要求重复区域填0，但所示代码没展示该分支；普通回退会对重叠部分重复累加，即使填0仍须合法GM地址和已初始化输出。

SparseToDense Default阶段可Duplicate一个UB批次再DataCopyPad写出，每轮tail长度应来自实际剩余，不能把defaultUbFactor全长用于所有尾块；队列Free/排空和进入Sparse阶段前的全局完成独立核对。

验证0、小于1块、恰1块、1块+1、多块尾、原子模式、别名和跨核邻接输出。禁止用“SafeDataCopy”或“避免踩踏”的函数名替代边界证明，低于一块时需平台支持的精确搬运或经验证的专用写回。
