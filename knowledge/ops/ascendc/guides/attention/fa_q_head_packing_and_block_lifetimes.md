---
type: Guide
title: FA Q Head 打包与 Block 生命周期
description: 保存GQA行映射和组件内部槽位，限制硬事件照抄。
tags:
- task.implementation
- topic.attention
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-handcraft.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/fa-kernel-recipes.md
  role: supporting
created_at: '2026-09-11T05:27:29Z'
updated_at: '2026-09-11T05:27:29Z'
platforms: []
---

# FA Q Head 打包与 Block 生命周期

FAInfer候选每task为(batch,qSBlock,qNBlock)，qS128，qN=min(max(floor(128/qLen)//2*2,1),groupSize)，groupSize=Hq/Hkv须为正整数。每组qN块数ceil(group/qN)，task数逐batch为qSBlocks*qNBlocksPerGroup*Hkv。该启发式在qLen0除零，且取min到奇数group或尾head后并不保证偶数；两AIV分头必须处理余头，不能都只取floor(qN/2)。

rowNum=validQRows*validQHeads，把BSND的[S,head,D]片段按真实stride载入为二维Q。kvHead=qNBlock/qNBlocksPerGroup，qHead=kvHead*group+(qNBlock%qNBlocksPerGroup)*qNTile；Q/O和K/V使用各自平面基址。qN1可按query行分两sub，qN>1可按head分，写回必须逆映射且覆盖奇数/尾行。

AIC/AIV对同一task使用一致的逻辑core编号、步长和batch累计。源伪码curTotalTaskNum=0后while立即++curBatch会跳过batch0，只有另有正确首batch初始化才可用；空batch连续跳过还需boIdx边界。AIV GetBlockIdx除subBlockNum与GetBlockNum语义必须匹配launch，不能两者都机械除2。

选定FAIQK的Q经GM→L1一次装载并在任务内复用，K/V用L1双缓冲再送L0，L0A/B与L0C生命周期分开。四个128KV块叠512列，L0C可能两块成对累加/写出；gCOffset=(stackIdx//2*2)*128需与输出stride和有效列一致。尾K、最后奇数KV块必须执行正确initMmad/endItr，不能只靠整4块流程。

computeQK伪码列GM→L1、L1→L0、Mmad、Fixpipe事件，但省略部分反向复用和consumer释放，不能当完整可执行协议。M_MTE1保护Mmad读完后MTE1覆盖L0，MTE1_MTE2保护MTE1读完后MTE2覆盖L1，M_FIX/FIX_M交接L0C计算与写出；源把MTE1说成L0→UB是错误解释。

来源AIV预置表2+4+4+1实际11项，自查却写10；AIC18与AIV11也只适用于那些成员组件和event池。换FAQK若构造已管理6类/槽信用，重复手动预置会破坏协议；析构行为也需核对。“多余Set导致缺token”的描述不构成完整因果，必须推演真实计数。

OnlineSoftmax把S/P、local/global max/sum、diff和scratch按向量/行广播布局分配，8192元素S槽和固定字节偏移只是特化。preLaunch延迟的RescaleO要读对应块的alpha和最终l，不能拿当前最新版覆盖旧消费者。常量表、14参数入口和模板特化语法仍需与真实类声明/Host ABI核对。
