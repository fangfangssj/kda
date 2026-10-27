---
type: Guide
title: CATLASS FA 接口映射及参考配方冲突
description: 保存长度、布局、mask 和 MLA 语义，限制固定尺寸与通路结论。
tags:
- task.design
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/flash-attention.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-design/references/kernels/flash-attention-npu-reference.md
  role: supporting
created_at: '2026-09-11T04:33:09Z'
updated_at: '2026-09-11T04:33:09Z'
platforms: []
---

# CATLASS FA 接口映射及参考配方冲突

标准前向为O=softmax(scale*QK^T+mask)*V，默认1/sqrt(Dqk)不覆盖自定义scale。来源FAInferTiling可能自行设默认scale，调用方需要使其与接口值一致。MHA/GQA映射hkv=floor(hq/(Hq/Hkv))要求合法整除。训练dropout、softmax统计输出、PSE、sink及其他字段不能仅因参考API存在就静默宣称支持；keep_prob=1是本来源前向范围。

来源内部Q/O用BSND，K/V用[numBlocks,128,Hkv,D]paged布局，公开BNSD/BSH/SBH/BSND/TND的转换需保持batch/head/序列stride。TND仅在packed序列与任务offset完全吻合时可直透，不能只按名称免转换。block_table恒等只在物理块排列、batch隔离及padding合同成立时等价连续KV；非恒等表还需边界/重复page语义。

外部actual_seq_qlen/kvlen若为累计边界，来源内核接收per-batch长度，需要差分；必须核对当前API究竟采用何种形式、是否包含起始0、非递减性、最终总数及每batch的物理基址。仅把总T除BT不能处理varlen。输入文件逻辑字节与paged物理分配大小分开，尾部K/V补0仍须mask无效score，零K产生0分而非负无穷，不能防止多余softmax权重。

来源maskType 0/1/2对应无mask/显式mask/结构causal，mask非0屏蔽的乘法score+=mask*(-3e38)只适用于对应0/1表示。sparse_mode0并不自动等于无mask，2/3的左上/右下causal在Sq=Sk时才重合；不等长需query偏移。band/prefix/varlen等转显式mask要保留pre/next token边界和实际shape/stride，不能只改枚举。

外部某压缩mask[2048,2048]与来源内核固定[1024,1024] FP16是两套协议，不能互换，更不能声明所有API都必须这两个尺寸。压缩模板的块索引、对角位置及更长序列映射必须可证，不能以一次报161001推导全局限制。全mask行有限极小值问题另见在线Softmax知识。

来源组件候选MmadAtlasA2FAIQK/FAIPV、FAITailQK/FAITailPV、EpilogueAtlasA2OnlineSoftmax/RescaleO、FAInferKernel的AIC/AIV特化和128立方tile需绑定版本。L1.K==D与固定K128限制了该具体变体，不能同时据此声称D64任意支持；D%16也只是其路径约束。PAGED=false报507015、固定true的绕过、kv128到204800及max_abs≤7.6e-6均为来源案例，不是全部A2固有限制或本次实测。

同一source既写A2 S/P/O_tmp/O_update用GM三槽交接，又画L0C→Fixpipe→UB并禁止所有S进HBM，与同组A2通路约束相矛盾。FlashAttention避免的是完整Sq*Sk矩阵物化，分块GM workspace仍可保留在线算法。跨核mode0/1/2与950 mode4必须按实际版本核验，Set/Wait对应事件和参与者一致；名字相同不够。

来源一处称BlockMmad+Epilogue+CrossCoreFlag从零拼装必死锁，另一处又给出可运行paged/MLA配方；应将结论限定为特定分流、预置、prologue/延迟PV和drain是否满足token协议。__mix__本身不导致必死锁，特定g_coreType分支/锁步调度也需真实编译与事件轨迹证明。kfc单TU直调、+4GB窗口、运行时transB、每chunk IterateAll等是后续配方的检索线索，尚不能从本导航推导完整实现。

MLA来源明确的特化合同是O=softmax(scale*(Q*c_kv+q_rope*k_rope))*V_latent，Dc512、Dr64、scale1/sqrt576、block128、q1–4，通用H16/32/64及H128/AMLA特化，MmadAtlasA2MLAQK/MLAPV。它是已吸收投影后的特定接口，不是所有MLA通用形式；验证包络kv≤16384与声称实测20万以上应分开，缺原始日志时不升级为当前支持。

精度应绑定完整公式、布局、mask与scale，stage参考还需独立验证其上游中间量。mixed tolerance、matched_ratio/max_abs只是指标，具体通过比例与NaN/Inf行为需标准；性能需同语义全流程、launch count及workspace峰值，不能拿单C1或C2时间代表FA全流程。
