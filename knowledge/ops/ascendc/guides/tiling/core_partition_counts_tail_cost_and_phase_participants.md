---
type: Guide
title: 分核数量、尾部成本与阶段参与者
description: 区分差一份的余数分配和末核余量，保留两阶段稀疏写入合同。
tags:
- task.optimization
- topic.implementation.tiling
- topic.memory
- topic.precision
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_04_load_balance.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_10_vectorized_copy.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/code-performance-advisor/assets/rules/special_rules/R_TILING_LOAD_BALANCE/R_TILING_LOAD_BALANCE.md
  role: supporting
created_at: '2026-09-11T06:15:28Z'
updated_at: '2026-09-14T04:09:11Z'
platforms: []
---

# 分核数量、尾部成本与阶段参与者

对N个等价任务、C>0个核，q=N/C、r=N%C，核i起点i*q+min(i,r)，数量q+(i<r)，任务数量差≤1。AdaptiveAvgPool、MaxPool、Softmax/Grad、Modulate等用该模式；“大小核”指任务份数不同，并非异构核心。窗口大小、数据稀疏性、参数复用不同仍会导致时间差，N<C还可能有空核，不能保证无空闲。Modulate所示baseline本身也是同样商余数分配，不支持新增10–30%收益的对比。

另一类AddRMSCast/Gemma/DeepNorm/Dequant及DynamicQuantUpdateScatter先h=ceil(N/C)，used=ceil(N/h)，前used-1核各h，末核N-(used-1)*h。这保证非空且覆盖时仍可能差多于1，例如N10,C3得到4/4/2。来源“tail总比head少1”不成立，按实际Host参数推导，GetBlockNum必须等于used，N0须先处理。

AdamW以2432元素为一次任务，再对ceil(total/2432)循环做商余数分配，差≤1个loop并不等于差≤1元素。2432个FP32为9728B只是一份数据，所有状态/双缓冲需另算。NormCommon以nRow行基本块均衡，再由末核处理tailNRow；循环currentBlockOffset/currentParamOffset需推进，末核GlobalTensor有效范围须包含附加尾行，片段xGmSize只有完整块，不能认为已经覆盖tail。

FakeQuant按head分配可固定每channel scale/zero_point，GetNeedCoreNum=min(headNum,platformCores)在head0需空路径；circleNum的普通分支要初始化。Select临时8000B是该实现参数，不能覆盖全部芯片/shape要求。Per-head量化的ProcessLoop/ParamLoop/InputLoop须由真实维度合同确定，gmXOffset+=dim1*dim2、baseXOffset+=nLoopLen*dim2与baseSOffset+=baseN在尾param块可能不同，不能从注释直接推成通用BSND。

LinearIndex先按DATA_ALIGN任务粒度估核数，再按eachCount重算used；零indices/零maxSize须处理，max_ub=ub/max_ub*max_ub/BUFFER_NUM要求旧max_ub是有效非零粒度，不是未初始化变量。SIMT MaskedScatter以min(ceil(N/1024),C)选核数，达到C上限后每核可循环处理远超1024元素；线程数不等于1024倍速度或每核固定处理量。

SparseToDense以max(defaultCores,sparseCores)启动，两阶段分别判定active范围，但所有协议要求的参与者均需经过SyncAll，不能让闲核提前return造成等待；每个阶段实际GM写完成后再发布。源代码仅对defaultCores限到totalCores，sparseCores也须合法限制并采用循环分发。default填充与稀疏覆盖不是相互独立可随意重叠的任务；重复稀疏索引须定义覆盖优先级/拒绝或累加政策，SyncAll不解决第二阶段内部冲突。

对分配函数验证各核区间不重叠、并集为全部任务；另核验实际stride、每核尾loop、barrier参与者和输出初始化。元素均衡是正确分工手段，性能仍需同负载测量，不将原文10–100倍或无空闲当事实。

## Former/Tail 整除反例

另一份来源规则直接使用 formerLength=ceil(N/C)、formerNum=N%C、tailLength=formerLength-1，但没有给出 r=0 分支。N=12、C=3 时得到 formerNum=0，三核都取 tailLength=3，只覆盖 9 个任务，漏掉 3 个。不能因为文件将“恰好整除”列为不适用，就让实际分发中的此类输入失去覆盖。

采用前述 q/r 公式可同时覆盖整除与非整除；若保留 former/tail 字段，应在 r=0 时让全部核取 q，或使用与真实 Kernel 完全一致的另一编码。N<C、N=0、C=0 和字段窄化另设合同，先验证区间并集与唯一归属，再评估时间是否均衡。
