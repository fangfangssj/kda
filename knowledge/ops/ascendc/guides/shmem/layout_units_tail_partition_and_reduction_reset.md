---
type: Guide
title: SHMEM 布局单位、尾块地址与归约复位
description: 补入对称头部单位换算、MoE 元数据索引、peer 分工和多轮原子结果初始化约束。
tags:
- task.implementation
- topic.memory
- topic.implementation.atomic
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/code-style.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-review/references/code-review-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-dev/references/agent-execution-contract.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/design-template.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM 布局单位、尾块地址与归约复位

对称区中的数据、signal/state 与元数据可能使用不同元素宽度。来源 symm_data_elems + block_num*SYNC_FLAG_INTERVAL 错把 int32 标记槽数当 T 元素数；先求头部字节 H=block_num*interval*sizeof(int32_t)，再按实际 T/地址对齐计算 ceil(H/sizeof(T)) 及 padding，最后分配全部数据与头部。仅整除会在 H 非 T 宽度倍数时少分配，乘加本身也应在范围检查后进行。

MoE 示例 count_matrix=[src_pe][dst_pe][local_expert] 的展平索引为 (src*P+dst)*E+e；output=[local_expert][slot][hidden] 的行起点为 (e*S+slot)*H；metadata=[e][slot][3] 中三项为 src_pe/src_token/expert。每个索引、count/prefix 的累积值和容量需独立约束，64 位中间量也会溢出。来源 prefix 循环未更新 count_idx，会重复读同一格，不能仅凭注释“prefix sum”判断正确。

用 q=N//C 均分、最后核接收余数时，核 i 起点仍为 i*q，长度为 q（最后核为 N-(C-1)*q）。若把修改后的 per_core 再乘 core_idx，最后核地址会错。N<C 时前面空核的同步参与规则仍需定义；只向下整除通常先造成遗漏，不一定立即越界。更均衡的 q/r 分工见[PE 与 Tile 分工](work_partition_and_pipeline.md)。

chunk 循环要覆盖小于、等于及大于 chunk 的尾部，count 单位、UB 中转容量和源目标剩余长度分别约束。来源 190KiB/16KiB 是某些例程参数，不是通用单次传输上限或最优值。跨 PE 大消息可以从合法用户 GM 直接 put 到远端 symmetric 以减少本地 staging，但前提是引擎支持该源地址；接收者不能在相应数据完成前 get/消费。

分核并不自动带来 peer 并发：若每个核仍串行遍历 peer，链路利用可能受限。生产和消费组应有数据就绪与消费完成两条依赖，signal 的对象/期次/条件需匹配；设计指定全局 barrier 可作为一种实现，按 chunk 通知也可在证明等价后替换，不能把所有路径都必须 global barrier 当硬件规则。

reduce/allreduce/reduce-scatter 需定义每轮初态。若本轮存在普通覆盖写初始化完整目的区，再累加远端贡献，可以不额外 memset；若只有原子累加，移除每轮清零会把上轮结果继续累入。来源“仅 warmup 清零”不是通用优化，初始化成本是否计时取决于两侧比较的同一语义边界。self contribution 是否跳过取决于已计入一次，详见[贡献与写权](reduction_contribution_ownership.md)。

缓存 local_counts 可减少重复 GM 访问，前提是数组容量覆盖实际 expert 范围、初始化与计数不溢出，并计入 Device 局部存储/编译开销；栈上数组不自动等于 UB 高速缓存。循环不变量提到外面还需保证无副作用及内存内容不变，编译器已优化时收益不能重复宣称。
