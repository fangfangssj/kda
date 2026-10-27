---
type: Guide
title: FP32 升精、算子状态与不完整归约
description: 保存逐算子混合精度细节，纠正算法与类型混淆。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_01_fp32_intermediate.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/acc_05_softmax_stability.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/dtype_05_bf16_special_handling.md
  role: supporting
created_at: '2026-09-11T05:47:04Z'
updated_at: '2026-09-11T05:47:04Z'
platforms: []
---

# FP32 升精、算子状态与不完整归约

FP16/BF16存储配合FP32中间运算可减少多次舍入和FP16平方/求和溢出风险，代价是额外Cast、FP32 UB和可能的workspace流量。FP32仍可溢出，尤其BF16输入本就具有接近FP32的指数范围；不能称双精度或保证永不丢精度。输入已量化的信息也不会因升精恢复。Abs与ReduceMax只比较已表示数值，不涉及求和；BF16升精可能是API支持或后续计算需求，不能声称BF16 Max必因多次累加而失真、FP16 Max则总足够。

AdaGrad/AdamW可分别定义存储U与计算T：var、grad、accum进入FP32后平方、累计、开方、除法、参数更新，再按输出合同转换。低精度存储优化器状态仍会每step舍入，不能当成持久FP32 master状态；对照全FP32算子表示不同输入/输出合同，不是同精度下免费减少带宽。DivHighPrecision名字不能证明具体Newton实现、所有近零分母都安全或训练收敛改善；epsilon位置及分母0策略须由算子定义。

AdaptiveAvgPool3D每输出窗口长度乘积决定平均分母，非整除时不能统一固定kernel_volume。Host预计算倒数可减少Device除法，但示例使用1.0f及float乘法，不是更高于FP32的计算，也不能称精确无误差；有效窗口非空、边界与计入padding规则须一致。先在宽整数域检查体积，再按约定精度求倒数。

LayerNorm输入/输出Tfm、权重Tweight与mean/rstd输出float是独立ABI；RMSNormGrad的rstd/dgamma为FP32也要在注册、分配和Device读写三端一致。precisionMode0在FP32乘gamma后舍入，mode1先转half再乘，改变舍入位置；来源数值枚举与high_precision_mode含义仅适用其版本。

源RMSNormQuant片段用factor=1/sum(x²)直接缩放，缺少除N、epsilon、sqrt，不能替代完整RMS公式。另一AddRMS段Add写xLocalFp32，却从yLocalFp32直接ReduceSum且未平方，不能以函数名或省略段声称正确。DeepNorm段Axpy修改local_x却从local_y求mean，再算local_y²；若无后续居中或减mean²，得到二阶矩而非方差，且可能统计错张量。归一化目标必须与残差融合结果一致。

ReduceSumFP32_V2示例只处理一个elementNumPerRep，未遍历repeatTimes、尾部、初始化value或return；GetAccVal和likely本身不证明更高精度或完整大向量归约。读取ACC的位视图与同步、来源类型、有效位宽均需真实接口支持，不能用未完整函数取代ReduceSum。

Scatter add应在FP32中累计同一目标全部更新再最终舍入；对照SetValue(idx,val)是覆盖而非add，不能作精度对照。Pooling梯度多贡献到同一位置同理；FP32 GetValue+SetValue不是跨核atomic，也不解决寄存器同批重复scatter index冲突。并行归属、初始化全局完成、归约次序和最终cast阶段必须独立成立，FP32 workspace不能自动提供同步。
