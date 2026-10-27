---
type: Guide
title: 自适应容量分块与多行缓冲预算
description: 补齐按不同 dtype、固定预留、索引区和同时存活状态计算行数的具体方法。
tags:
- task.design
- topic.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P2_adaptive_tiling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P3_small_d_optimization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/P8_ub_memory_mgmt.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P2_adaptive_tiling.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P3_small_d_optimization.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/P8_ub_memory_partitioning.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P2.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P3.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/P8.yaml
  role: supporting
created_at: '2026-09-11T01:38:31Z'
updated_at: '2026-09-11T01:38:31Z'
platforms: []
---

# 自适应容量分块与多行缓冲预算

自适应 Tiling 先确定目标执行模式的实际可用 UB，再减去固定区、API scratch、索引/统计状态和其他常驻区。每行占用按各存活 Tensor 的真实行跨度、dtype、缓冲份数逐项求和；可容纳行数为 `floor((UB-fixed)/bytes_per_row)`，减法前验证 fixed<=UB，且 bytes_per_row>0。包含非线性 scratch 或分配粒度时，还需对候选行数计算真实分配量。

来源 CalMaxFormerNum 的具体组织是 `available=ubSize-reserved-idxAlignNum*4*3`，再按 `gradAlignNum*4*useGradNum` 除出完整行数。它报告 reserved=20KiB、3 份索引、FP32 useGradNum=3/FP16=4；这些是样例缓冲模型，不是所有平台固定预留。最后乘 gradAlignNum 得到的是对齐元素容量，不能把它仍称为行数。

多行小 D 的核心是一次搬入多个连续行，减少短 DMA 的发起开销和逐行控制。若每行 D=8 个 FP32 元素，行占 32 字节，一个 256 字节向量容纳 8 行，公式应为 `vector_bytes/(D*sizeof(T))`，不能把 256 字节直接除以 D。独立行归约还要保留每行的 max/sum，不能把合并批次当作一条长归约。

大 TBuf 划分多个区间时，应统一先计算字节偏移，再转换为各 LocalTensor 的元素索引。源例 x/x2 使用 T 视图而 fp32/out 使用 float 视图，offset 数字递增不证明物理区间不重叠或不越界。用不同 dtype 的相同 numBuffers 系数估计全部区间会漏算。rowsPerUB=0 代表当前一行不适配，强行改为 1 会越界，应切行内维或选择另一实现。

双缓冲只对增加了实际物理份数的区间乘 2；TBuf scratch、常驻数据不必一并翻倍。生命周期不重叠的状态可在最后消费者完成后共享同一物理区；不同偏移的同时存活区只是分区，并不会自动减少总容量。把中间状态搬到 GM 可减 UB，但增加 workspace、搬运与完成交接。

块尺寸与核数需要分别考虑。以 `ceil(total_bytes/min_bytes)` 估计核数并不能保证尾核至少 min_bytes；如 257 字节、min_bytes=256，会得到两个块但尾部很小。小输入应避免零 blockDim/division，AIC/AIV 数按实际执行类型区分。二维候选按宽高比估计后，还要取整、对齐并检查真实资源与尾块。

源文 195584 字节 SingleN、192/256/512KiB 容量、50%/80%占用分档、D<=32/512、最少 64KiB 缓冲等只是不同材料的示例或策略阈值，且平台表互有冲突；不能当硬件合同。32 字节对齐也不充分消除 bank conflict，后者取决于并发访问模式。索引预计算的成本与收益应计入同一预算，而不只计算 Cast。

验证容量下界、仅一行能容纳、混合 dtype、奇数 D、多行尾批、不同 buffer 数和模式分发。较大 tile 不必然最佳，自适应也不强制排除静态特化；比较完整运算的时间、实际内存峰值及每行数值结果。

源例还包含自适应池化的一维窗口：输出位置d对应输入起点 floor(d*Din/Dout)、终点 ceil((d+1)*Din/Dout)，终点为开区间。可在shape、轴与分块映射不变时预计算并复用，需保证Dout>0和乘法整数范围；多维分别计算各轴。索引buffer的数量/位宽/生命期须计入UB，不能只按避免除法估算净收益。
