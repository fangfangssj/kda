---
type: Guide
title: Matmul 分层 Tiling、Batch 广播与 Split-K 边界
description: 保留缓存/核/基础块模型，修正广播、分配和转置偏移公式。
tags:
- task.optimization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/matmul/ascendc-api-matmul-tiling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-tiling-design/references/matmul/patterns.md
  role: supporting
created_at: '2026-09-11T04:27:49Z'
updated_at: '2026-09-11T04:27:49Z'
platforms: []
---

# Matmul 分层 Tiling、Batch 广播与 Split-K 边界

来源将Matmul分为L2缓存块、单核块(singleCoreM/N/K)及baseM/N/K基础块；TCubeTiling示意列出约196字节和shareL1/L0C/UB、buffer份数、方向等字段，不构成跨版本ABI定义。一个基础块也不必等于一次硬件发射。950优先Blas/Blaze类路线是来源的选择建议，不能解释为其他Matmul API在该平台一概禁用。

缓存工作集应计A/B/C各自dtype字节数、bias/scratch、并发和实际缓存行为；M*N≤L2/2忽略A/B，mTileBlock*nTileBlock≤L2/2还混淆元素与字节。蛇形遍历可能重用相邻A/B块，但不能仅从形状证明L2命中或L1 bank冲突已消除。候选base块如FP16/BF16的M/N=128或256、K=32或64及FP32的M/N=64或128、K=16或32只是调优起点。

非空维度下mCount=ceil(M/singleM)、nCount=ceil(N/singleN)，末块有效M=M-(mCount-1)*singleM；空维须先返回相应合同。任务总数T大于usedCoreNum时，仅把blockIdx映射成一个tile会遗漏，其后需遍历t+=usedCoreNum或等价完整分配。ceil(T/C)与T%C定义大/小核分配时，余数0改为C仅适用于对应非空公式，必须核对边界。

来源LCM交错映射m=t%a、n=(t+floor(t/lcm(a,b)))%b，在0≤t<a*b及正a/b时可作为完整网格映射候选；需要验证覆盖唯一性及尾部L2块的实际a/b，不能把它混同蛇形。blockIdx-blockIdxStart要先保证合法范围，避免无符号下溢。GetTiling成功表示取得候选，不证明全局最优；改变singleM/N后需重新验证所有依赖字段，来源“交替减半”伪代码实际先减M再减N。增加核数也可能增加流量和尾块浪费。

Batch广播按右对齐的每个轴判断相等或1，不是取两个batch总数的LCM。[2]与[3]不能广播，即使LCM=6；[2,1]与[1,3]产生[2,3]。零维规则需按接口。仅当B在batch间共享、A未转置且物理连续、输出和bias合同兼容时，才能把A的batch合入M。iterBatch=min(batchC,C)及ceil(batchC/(iterBatch*C))并不单独给出每核batch映射，必须与实际循环共同证明覆盖。

逻辑K、Ka/Kb物理步长和原始存储shape应区分；不能为通过检查而强制Ka=Kb=K。SetOrgShape按实际ND/NZ和padding语义填写，c0依赖dtype/平台；ND尺寸对齐不自动免除ND→NZ转换，VEC转换也需要平台能力。Host到Device的阻塞Tiling拷贝是充分方式之一，合法异步方式只要满足stream依赖与生命周期也可成立。HF32需Host/Device精度模式对应；enUnitFlag提高99%或所有db固定2均未提供通用证据。

Split-K适用于需要额外并行且收益覆盖归约/流量的情况，“K≫M*N”只是一种启发。联合任务应唯一映射(outputTile,splitIndex)，blockDim可能受硬件上限限制；来源既扩展blockDim又让每核循环全部split，会重复累加。连续行主序A[M,K]不转置时k片偏移+kStart，转置存储A[K,M]时+kStart*M；B[K,N]不转置时+kStart*N，转置存储B[N,K]时+kStart。尾K、原始leading dimension和tile起点仍需另加。

GetTensorC(c,2)中的2不能脱离真实重载解释为通用原子模式。若使用原子累加需初始化C、保证参与者完成、恢复模式并只加一次bias；workspace分段本身不保证确定性，固定合并顺序才能约束浮点顺序。所有尺寸/字节计算需防溢出并处理零块，合法对齐由实际API给定，不能笼统要求所有轴16。
