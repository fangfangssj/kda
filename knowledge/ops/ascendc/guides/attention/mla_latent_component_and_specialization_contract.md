---
type: Guide
title: MLA Latent 组件与特化合同
description: 保存两段Score、V复用与模板差异，限定历史适用域。
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

# MLA Latent 组件与特化合同

该MLA接口Q[tokens,H,512]与QRope[tokens,H,64]分开，paged K latent与KRope分别宽512/64，score=scale*(Q*Klatent^T+QRope*KRope^T)，V为同一latent512维，输出[tokens,H,512]。默认scale1/sqrt576只适用于此明确维度与默认合同，不能误用sqrt512，也不能对两段score独立softmax。输入RoPE部分若已经编码，不应再次应用位置变换。

来源q1–4 decode/MTP、KV≥128且尾页可不满、H16/32/64通用、H128 TP1/AMLA、FP16/BF16是其声明范围，不是所有MLA定义。q>4仅打印告警不会实际执行接口拒绝；历史ATB只支持q1与KV16384以上未经其原验证的区间须版本限定。黄金应计算该latent域语义，不能擅自解压V再比较。

通用组件候选MmadAtlasA2MLAQK/MLAPV是该版本普通struct，FAIQK则可能是模板；BlockMmad实际GemmType参数与某伪码裸类型不同。Softmax、RescaleO、FDRescaleO参数个数也不能照另一版本类比。TP1候选MLAQKTp1Spec/MLAPVTp1Spec、AMLAPVTp1Spec及对应MLATP1/AMLATP1 epilogue只提供导航，换名字并不完成全部task布局特化。

源H128一处称K常驻L1，一处称4-block且无K常驻，不能同时采为已定架构。L1/L0静态Shape<128,128,576>不证明完整tile一次驻留每级：单份128*576*2=147456B已大于典型64KiB L0A/B，需要组件真实内部分块。L1.Q/K/P分区和stage份数也要逐项计算，K576不自动证明跨task复用。

预处理__DAV_CUBE__/__DAV_VEC__分支、FA模板特化及KFC __mix__是不同编译/调度模型。来源某组合死锁不能推出__mix__属性自身必错；匹配实际二进制核型、组件事件协议及entry才是可复用约束。单kernel、小GM轮转、专用两段QK、KV复用、split-KV是性能候选，不能宣称缺任一项就必慢一个数量级。
