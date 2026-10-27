---
type: Guide
title: Softmax 有限域稳定性与 NaN/Inf 处理
description: 区分减最大值的数学保证、在线缩放、非有限值和算子要求的特殊行为。
tags:
- task.design
- topic.precision
- topic.compute.reduction
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/A5_softmax_stability.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/A5_softmax_stability.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/A5.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# Softmax 有限域稳定性与 NaN/Inf 处理

对非空有限输入，m=max(x) 后每个 x-m<=0，因此 exp(x-m) 位于 [0,1]，至少一个对应最大值的项为 1。精确算术下归约和位于 [1,n]；如果此条件下出现大于零的偏移，先检查归约轴、最大值、索引和同步，而非把它解释成“大输入即使减 max 仍必然溢出”。低精度累加的 sum 仍可能溢出或丢精度，需与单个 Exp 的溢出区分。

FP16 最大有限值对应的自然对数约 11.09，FP32 约 88.72；BF16 的指数范围接近 FP32，不能套用 FP16 的约 11 阈值。减最大值通常解除有限输入的正向 Exp 溢出，但没有自动解决输出转换、累计误差或全部特殊值。

全负输入的最大值不能固定初始化为 0。有限域可用真实首个有效元素或正确中性值；允许 -Inf 时，-FLT_MAX 不是 -Inf 的等价替代。全 -Inf、存在 +Inf、NaN、空行、全 mask 行必须遵守具体算子的语义，不能无条件改为全零或均匀分布；`-Inf-(-Inf)` 可能直接产生 NaN，而非先得到一个可以用 sum==0 处理的正常零和。

`x==x` 在通常浮点比较语义下可区分 NaN，但不能检出 +Inf/-Inf；若编译配置改变非有限值假设还需单独确认。Compare+Select 将 NaN 清零是部分量化合同可能要求的行为，不能自动加入所有 Softmax 或训练算子。Mask 的打包宽度、有效 lane 与 Select 真假分支须按具体 API 处理。

分块在线统计必须把旧 sum 重新缩放到新最大值基准：`l_new=l_old*exp(m_old-m_new)+sum(exp(chunk-m_new))`。缩放因子下溢为零可能表示旧贡献在该精度下已经不可分辨，不应随意改为非零保底值；是否满足误差要求应验证。单纯先求 max 再求 sum 是两阶段归约，不因这两步就成为在线算法；完整概率输出还依赖最终统计量，见[驻留和重读选择](softmax_storage_and_input_passes.md)。

来源的 Sub/Mul 标量参数和泛化 Reduce 签名只表达数学意图，不能直接作为所有重载的可编译代码。验证有限大值、全负、常量、长轴、分块尾部、全 mask 和合同允许的非有限值，并比较中间 max、sum、缩放和最终输出。
