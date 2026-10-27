---
type: Guide
title: Matmul Softmax 的历史指数 workspace 与行归并
description: 保留 per-core max/sum、逐 tile 历史最大值及跨核完整归一化地址。
tags:
- task.design
- topic.compute.reduction
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-design.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cube-matmul-then-vector-softmax-reduce-development.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-blaze-best-practice/references/scenarios/cube-matmul-then-vector-softmax-reduce/cv1-v2-adaptation-for-softmax-reduce.md
  role: supporting
created_at: '2026-09-11T03:01:48Z'
updated_at: '2026-09-11T03:01:48Z'
platforms: []
---

# Matmul Softmax 的历史指数 workspace 与行归并

对有限有效logits的某行某生产分区，状态(m,l)初始为空。新N tile最大值t令m′=max(m,t)，p=exp(x-m′)，l′=exp(m-m′)*l+sum(p)。空初态需分支处理，不能直接算-inf-(-inf)。每tile将p写expWorkspace时，必须同时保存该p所用的m′，记为mHistory；后续状态继续改变，不能用最终分区m代替每tile历史值。

跨分区令m*=max_c(m_c)，l*=sum_c(l_c*exp(m_c-m*))，对应tile输出为p*exp(mHistory-m*)/l*。每个有效列应恰好属于一个分区/输出tile，重复或遗漏改变归一化。无贡献分区不应因0乘NaN污染合并，全空/全masked/NaN/Inf按算子合同另处理，见[online softmax](../attention/online_softmax_state_and_partition_merge.md)。

固定FP32、C个状态分区、R逻辑行、J个可独立寻址N tile时，逻辑存储量为onlineMax/onlineSum各C*R*4字节，mHistory为R*J*4，expWorkspace为R*P*4，其中普通32B行对齐候选P=ceil(N/8)*8。各buffer起点按实际消费者对齐，总容量还需计间隙与整数溢出；不能只拼裸逻辑长度。若Scheduler重分边缘tile或改变遍历，J和历史索引应跟真实tile集合一致，而非盲用ceil(N/baseN)。

来源写[C,M]与[M,N]只足以直接表达单问题；BMM/GMM需展开正确R或增加batch/group维度，并让状态分区、mHistory、exp数据和输出共用同一row映射。V2可以只负责一部分行，但必须读取每个所属行在全部producer上的partial；来源“不能只读自己分到的行”应解释为不能只读本producer贡献，并非每个V2重复输出全部行。

V1按L0C2UB的实际SplitM接收本地行，需与Copy trait、curM/baseM、task ratio及UB slot相符。来源首版ubDB=1是实现简化，不是Softmax数学要求。V2跨核max/sum与读取mHistory、expTile两槽需要各自UB预算和就绪/复用依赖，物理P不同于有效N。

验证让最大值在后续tile/另一核出现，改变分区数、tile遍历、奇数M、非对齐N、空分区、多次启动和历史槽复用。来源固定max_abs<0.002或max_rel<0.01只是其示范门禁，不移植成统一标准；本批未运行设备实现。
