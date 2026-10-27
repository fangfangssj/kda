---
type: Guide
title: 分层分块、参数复用与 Foreach 余数
description: 保存核、UB、循环层的数据划分，纠正整块数量与尾块位置。
tags:
- task.optimization
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/collaborative-agent-kernel-evolution/skills/cake-evo/references/meta_prompts/strategies/perf_02_adaptive_tiling.md
  role: primary
created_at: '2026-09-11T06:10:16Z'
updated_at: '2026-09-11T06:10:16Z'
platforms: []
---

# 分层分块、参数复用与 Foreach 余数

AscendQuant按Block/UB/Loop三级安排；根据shape0、shape1的cacheline数选择单轴或双轴切核，baseN/baseLen与UB和搬运对齐共同约束。Scale/offset可以在固定参数切片范围内被多个x块复用，跨channel/不同batch参数时需重载。增加活跃核数只是候选准则，细碎tile或参数重复加载可能抵消收益。

Modulate按B/L/D选择：B*L<coreNum时估D切块字节，太小时回B；B≥coreNum优先B，部分B范围按tileL字节决定B/L，其余L。MIN_TILE_SIZE、HALF_CORE_NUM依源码值，不是硬件常量；coreNum>0、宽乘积、tileD0/尾D和真实参数广播轴均需处理。

GroupedMX按uo*groupSize任务分核心，coreData=ceil(tasks/totalCores)，used=ceil(tasks/coreData)，blockFactor=ceil(tasks/used)，末核tail=tasks-(used-1)*blockFactor。tasks0需先返回避免coreData0；每核区间、group身份、UB尾块与quant block的scale分组边界要同步，不能把maxUbCol行数和ubFactor列数混用。

Foreach tiling保存每Tensor元素数量与每核首尾Tensor/offset。uint16列表索引有上界，dataCount可64位但copyTimes使用uint32可能窄化；多个输入和标量列表需一致身份。核级可以跨Tensor均衡，UB级仍要以实际dtype转换区的maxDataCount/maxCastDataCount处理。

来源ForeachAddScalar例先q=dataCount/maxCount、r=dataCount%maxCount，却仅循环i<q并把最后整块替换成r：q=1,r>0会只处理r个而漏整块，q=0时所有r都漏。正确覆盖可处理q个完整块，再在offset=q*maxCount处理r；或使用ceil块数且每轮min(剩余,maxCount)。不能把该错误例总结成已支持任意长度。

DataCopyPad可按DataCopyExtParams字节blockLen搬运不整块长度，源isPad=false不代表为计算lane补0；有效tail与UB物理容量/对齐仍需分开。普通DataCopy的合法长度和起点由该重载要求决定，不能只因非末块就默认合法；多个输入/输出都需相同tail语义。TPipe::Destroy依对象生命周期和已完成使用，不由Regbase派生类自动保证释放时安全。

DAGSch/MemOptCfg LEVEL_2可表达依赖与内存策略，仍需检查状态输出、别名和编译结果；不是对手工队列必然更快或自动最优的证明。对完整算子比较Host选择成本、搬运、内核与输出准备，不能无测量断言Host开销可忽略。
