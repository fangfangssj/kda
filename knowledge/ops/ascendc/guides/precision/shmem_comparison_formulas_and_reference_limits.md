---
type: Guide
title: SHMEM 误差公式、逐位比较与参考兜底边界
description: 区分逐元素、平均偏置及参考误差比，保留来源阈值的场景属性并纠正模板判定差异。
tags:
- task.testing
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-testcase-gen/references/precision-standard.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-testcase-gen/references/correctness.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-testcase-gen/references/test-structure-template.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM 误差公式、逐位比较与参考兜底边界

比较合同先区分纯 MOVE、整数计算/索引、量化、浮点计算和高精度累加，不能仅凭 collective/fused_compute_comm 标签选统一阈值；融合结果与整数路由 metadata 可以需要不同规则。输入/累加/输出 dtype、cast、归约树及操作长度决定误差传播。来源 compute_times=PE数*每PE元素数 是全工作量启发式，AllReduce 每个输出实际归约 P 个贡献，并不因 Tensor 更长而增加该元素的累加深度。

来源场景阈值（均令 atol=rtol）按 compute_times<2048/≥2048 给出：FP16 COMPUTE_FLOAT 为 2^-8/2^-7，BF16 为 2^-7/2^-6，FP32 为 2^-11/2^-10；HIGH_PRECISION 分别固定 2^-11、2^-11、2^-14；FP16 QUANT 固定 2^-8；MOVE/INTEGER 为 0。eb 阈值 FP16/BF16/FP32 分别为 2^-10/2^-7/2^-14。表格是来源验收策略，未证明为所有算子、累积顺序或平台的误差上界；未列类型也不能补猜默认值。

来源逐元素公式 |a-g|≤r*max(|g|,1)，与模板 np.allclose 的 |a-g|≤atol+rtol*|g| 不同，即使 atol=rtol=r 也不等价。例如 g=1、r=0.1，误差0.15通过后式却不通过前式。mismatch 掩码也必须采用最终真实判据，不能用“绝对误差和相对误差同时超标”替代加法阈值。

MaxAE=max|a-g|，MaxRE=max(|a-g|/max(|g|,eps))，MeanRE 对同一逐元素相对误差求均值。来源用 float64 tiny 作 eps，会使近零参考的相对误差极大甚至溢出，并非可靠的数值尺度。eb=|mean(a-g)|/max_i|g_i| 是全局归一化的有符号平均偏差，不是逐元素相对误差均值；正负误差可能抵消，全零 golden 时需明确零分母处理。空输出也不能直接 max/mean 后按 NaN 结果隐式判断。

rtol=atol=0 的数值相等仍不等于逐位相等：+0 与 -0 数值相等，NaN payload 也不能由 array_equal 保真。纯字节搬运若合同要求 bitwise，应比较同 dtype/shape 的有效字节；整数 exact 应保持整数域，int64/uint64 超过 2^53 后转 float64 可能合并不同值。误差统计提升 float64 只在可表示范围内有帮助，不能恢复先前舍入丢失的信息。

来源 Reference pass 将 SHMEM 与 Torch 分别相对 golden 的 MaxRE/MeanRE/RMSE 作比，门限为10/2/2，最后 basic_pass OR reference_pass。这是一种额外容许规则，不能默认覆盖 exact、索引或公共数值合同。Torch误差为0时须定义0/0与非零/0；两者同错、golden不足精度或布局相错都会让比值失去证明力。报告应分别列出basic与reference，不能把兜底通过写成满足原逐元素阈值。

NaN/Inf 规则由算子语义确定：有限输入正常路径出现非预期非有限值需要定位；合同允许非有限结果时应分别核对位置、符号及必要payload，不能一律拒绝或一律把同 Inf 放过。来源模板只检查 actual，golden也要审视；其代码未实现文中要求的eb和torch-output兜底，不能直接宣称完整实现该精度标准。
