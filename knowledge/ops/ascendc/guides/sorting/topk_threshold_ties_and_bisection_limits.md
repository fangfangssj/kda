---
type: Guide
title: TopK 阈值、并列值与二分搜索边界
description: 保存分段排序和采样语义，限制固定迭代及 GatherMask 原地断言。
tags:
- task.implementation
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/ascendc_sort_topk_patterns.md
  role: primary
created_at: '2026-09-11T04:38:57Z'
updated_at: '2026-09-11T04:38:57Z'
platforms: []
---

# TopK 阈值、并列值与二分搜索边界

找第k大阈值与输出有序值/有序索引不同。值域二分只能帮助求阈值/选择集合，不能直接代替argsort；其成本约迭代数乘扫描量，分段Sort则有排序及归并成本。来源910B3交叉点8192、分组N≤1024优先排序是场景选择，不是通用边界。k≤2也可比较归约成本，但需tie/index合同。

二分需有限合法端点、单调计数及有效数据范围。range/2^iter小于相邻目标值gap只是有限正gap条件，ties时gap=0不能靠更多迭代区分固定k，浮点中点还可能等于端点停滞。padding哨兵-1e6会扩大range，也可能与合法更小数据冲突；最好不让无效padding参与端点/计数。epsilon提前停止只能给误差界，不保证精确kth，NaN/Inf需专门合同。

TopP中exp(x-max)可预计算并按相同归一化/筛选集合复用，改变topK后归一化集合可能改变概率，不能只因指数不依赖阈值就省略所有重算。计数可用CompareScalar、Select、ReduceSum，数量大时FP32计数精确范围和输出整型转换需核对。来源static const -1/0被编译为0属未附构建证据的编译案例，不是C++或所有AICORE的规律；-inf合法性与端点算术问题应分开。

分段Sort产生value/index结构，再MrgSort4归并候选。2048段、repeatTime=dataLen/32及tmp=2*dataLen属于来源选定API，scratch仍应按真实dtype/重载查询，padding补齐与索引还原不可省略。Concat在某910B3路径可能被封装为空，不表示所有Concat都无效。降序Sort可用有限浮点取负转升序，NaN/±0/稳定tie及整型最小值另处理；常量名THIRTY_TWO撞头文件应避免命名污染。

来源GatherMask reduceMode/vreducev2例报507035后要求dst=src0、mask独立；这些最多是该版本/调用的保守方案，不能从非对齐offset失败推导一切out-of-place禁止。需要检查src/dst/mask对齐、支持别名、有效长度和压缩计数。若目标API允许原地，可先每块压缩再拷到累计区；标量回填开销是否可忽略取决于幸存量。ASCENDC_REPORT_NOT_SUPPORT宏可能在条件为false时报错，需读宏定义，不能按名字把VERSION_V2支持条件反读。

“保留所有value≥kth”可超过k，与恰好k个位置语义不同。若目标是值阈值，局部topK归并取得kth后需重新扫描所有原值收齐并列；bf16 ties密集更易暴露问题。后续TopP按排序位置累计并scatter时还需稳定tie索引和阈值包含规则，重新排序若不稳定仍可能改变被留位置，不能只重收集值就保证一致。

p通常概率范围[0,1]，k合法范围需目标算子定义；来源k≤min(N,1024)、p>1触发官方异常均限定版本/接口，不是TopK数学上限。randn生成概率不符合这类输入合同，超范围比例并非固定40–70%。小B仅1–2核可能受固定开销限制，但是否验收通过由目标约束决定。性能比较要同语义/范围，eager多launch、融合Kernel及Task Duration/端到端应分别报告。
