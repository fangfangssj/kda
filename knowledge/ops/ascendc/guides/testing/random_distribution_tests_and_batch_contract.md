---
type: Guide
title: 随机数分布检验的样本、参数与批量判定
description: 区分随机数分布检验与逐位比较，明确样本、分布参数、单次显著性和批量次数的一致性。
tags:
- task.testing
- topic.compute.random_sampling
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/references/random_generation.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ops-precision-standard/scripts/random_distribution_check.py
  role: implementation
created_at: '2026-09-14T04:03:29Z'
updated_at: '2026-09-14T04:03:29Z'
platforms: []
---

# 随机数分布检验的样本、参数与批量判定

随机数生成的分布检查与固定种子的逐位复现回答不同问题。来源以单样本 KS 检验比较观测数据与指定分布：uniform 使用 loc=low、scale=high-low；normal 使用 loc=mean、scale=std；exponential 使用 loc=0、scale=scale。边界合法性、有限样本、dtype 离散化和分布参数应先确认，不能把不同的参数或样本单位混在同一结果中。

需要区分单次检验的样本数 n 与批量运行次数 N。二维张量的第一维长度不等于总元素数；把多维数组直接传入现有脚本可能产生数组形式的 statistic/p_value，随后触发真假值歧义。若业务合同允许把元素作为同一分布样本，应显式选择展平；若行列代表不同分布或相关序列，应保留分组并分别检查。展平本身不证明样本独立。

来源单次函数固定 alpha=0.01，以 p>alpha 作为不拒绝门限。批量函数却另接收 alpha 和 N，没有把 alpha 下传，且 N 不必等于实际 outputs_list 长度；这会让单次判定与批量门限不一致。批量检查必须绑定实际运行数、实际单次 alpha、非法输入/失败样本和独立重复方式，不能只改变一个汇总参数便声称采用了另一套检验标准。

来源批量候选门限为 `(1-alpha)-3.0902*sqrt(alpha*(1-alpha)/N)`；alpha=0.01、N=100 时约 0.959253。这是来源的正态近似规则，不能在未核验样本独立性、有限样本适用性及真正 N 时提升为统计保证。无效分布、空列表或检验异常应明确报告，不能作为没有失败证据的通过项。

p 值超过门限表示本次检验未拒绝指定假设，不能证明分布完全相同、生成器没有偏差、序列独立或所有种子都正确。均值、标准差、最小/最大值可辅助定位，也不替代分布合同。来源把通过解释为“分布与期望一致”的绝对措辞应收窄。

验证比较器需覆盖非默认 alpha、N 与实际次数不同、二维输入、非法参数和不匹配分布。还应分别检查随机状态推进、重复调用、inplace 状态和固定种子的复现要求；这些条件不由 KS 检验单独建立。
