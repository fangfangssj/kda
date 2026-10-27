---
type: Guide
title: SHMEM 数据面引擎与完成域
description: 补全 RMA 族、UB 生命周期、quiet/sync/barrier 差别及数据先于通知的条件。
tags:
- task.implementation
- topic.data_movement.communication.collective_communication
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/api.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/internal-api-boundary.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/code-style.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-review/references/code-review-checklist.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/hardware-architecture.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM 数据面引擎与完成域

来源 Host RMA 提供 typed put/get、bit-size 8/16/32/64/128、标量 p/g、跨步 iput/iget、按字节 putmem/getmem、NBI 及 stream 版本。typed count、bit-size count 和 byte count 的单位必须跟具体重载一致。Device GM2GM、UB2GM 及 GlobalTensor/LocalTensor 重载有独立地址要求；non_contiguous_copy_param 的 repeat、length、src_ld、dst_ld 描述二维搬运，不能只给总元素数代替跨度。

put 从本地向指定 PE 写，get 从指定 PE 读到本地。公开接口通常接收本 PE 对称地址加目标 PE，以其合同定位远端；不能先手动映射再无条件传给仍会映射的重载。裸 DataCopy 不能替代 SHMEM 公开数据面的拓扑、引擎和完成合同；本 PE 的 GM/UB 整理也需是 Ascend C 实际支持的搬运重载，不因来源写“GM↔GM”就认定所有本地空间组合合法。

MTE put/get NBI 使用相应 UB 中转、容量和 sync_id；SDMA 与 RoCE/RDMA 另有命令 scratch/注册及完成方式。来源 API 梳理列 SDMA 至少 64B、RoCE 大于 64B 的 UB 参数，而硬件摘要又说 RDMA 不需 UB、SDMA 只能 Host 发起，存在混淆：有效 payload 可直接 GM→GM，不等于提交 scratch 不需要 UB，也不能据硬件摘要删掉公开 Device SDMA 接口。aclshmemx_rdma_* 与 aclshmemx_roce_* 名称不能猜测互换。

来源说明部分 RoCE 实现对同 PE 的并发 RMA/AMO 排序有限制；应按实际引擎支持确定 sync_id、提交串行化与完成条件，不能把它当所有版本都禁止并发。UB staging/scratch 必须活到该操作不再访问为止，不能在 NBI 返回后立即覆盖。UB 地址 0、512B placeholder 或空指针是否有效只由该重载和编译模型决定，命名常量并不使它合法。

NBI 返回表明提交阶段结束，消费数据前还需完成。来源 aclshmem_quiet 覆盖全引擎，mte_quiet/sdma_quiet/udma_quiet(pe)/roce_quiet 面向相应引擎；用 MTE quiet 不能 drain 另一个引擎。单引擎专用 quiet 可减少无关等待，来源“纯 MTE 禁止全局 quiet”是选型策略，不证明更强完成在语义上必然错误。实际引擎默认配置和覆盖对象需随版本核对。

来源 fence 与 quiet 在其实现中等价，不是永久 API 定理，尤其不能将 fence 解释成保证任意收到的远端写入已完成。sync/team sync 不能自动保证所有远端 RMA 完成。Host 发起操作和 Device 发起操作的完成域也不同；Host barrier、Kernel stream 同步、库 handle_wait、Device quiet 只有满足实际接口前提才可用于对应链路，SetFlag/WaitFlag<MTE3_S> 不会自然完成所有通信引擎。

signal_op 的 SET/ADD、put_signal/put_signal_nbi 及 signal_wait_until 的 EQ/NE/GT/GE/LT/LE 用于发布和等待。独立 signal 写入不自动给任意先前数据建立远端可见顺序，ADD 也不能仅凭名称当作多写者原子累加。typed wait/test 的 all/any/some/vector 变体需一致的集合与期次。put+signal 的数据先行性质以该接口真实合同为准。

来源 Device barrier 有 MIX、FFTS 初始化和有效核形态前提，deprecated *_barrier_vec 改名不是忽略这些前提的理由；同 PE SyncAll 与跨 PE barrier 分属不同参与者集合，不能连续罗列便声称任意内核可用。原子 add 的 dtype 与 transport 支持分别核对，int8/int16/int32/half/bfloat16/float 列表是来源能力描述；异步原子读取前仍需完成，未支持的传输累加可用独立槽位和本地归约。
