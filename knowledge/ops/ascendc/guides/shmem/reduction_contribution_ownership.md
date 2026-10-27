---
type: Guide
title: SHMEM 归约中的贡献计数与目标写权
description: 区分本地初值、远端贡献、原子累加与独占归约，明确分组并行和对称缓冲复用的条件。
tags:
- task.design
- topic.implementation.atomic
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/atomic-add-pattern.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/code-patterns.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-performance-optim/references/optimization-patterns.md
  role: supporting
created_at: '2026-09-10T13:45:27Z'
updated_at: '2026-09-10T13:45:27Z'
platforms: []
---

# SHMEM 归约中的贡献计数与目标写权

归约实现应先定义每个输出分片需要哪些 PE 的贡献，以及由谁完成初始化和合并。对加法归约，每个应计入的贡献必须恰好计入一次；普通覆盖写不能自动表达多来源求和，但存在多来源也不意味着唯一实现是原子加。

来源中的通算融合实现让 AIC 将 self contribution 直接写入本地 D，AIV 再从其他 PE 的对称 workspace 拉取贡献并累加。在这套协议中，远端循环跳过 self 的前提是 self 已经正确进入 D，且其初始化完成先于其他累加。若换成清零后统一归约全部 PE，self 也可以参与同一归约路径；不能无条件沿用跳过 self 的分支。

同一个输出地址可以由受支持的原子更新处理，也可以由一个归约者独占：各生产者写入独立槽位，归约者在数据就绪后按约定顺序合并。后者需要额外存储和数据交接，却不要求多个远端写者直接原子更新同一地址。是否使用浮点原子还受 dtype、传输路径及结果顺序要求影响，参见 [DMA 原子状态范围](../memory/dma_atomic_write_scope.md)。

逐 peer 拉取、等待、累加可能形成串行链。增加并行度的一种方式是按源 PE 分工，各组先把贡献放到互不覆盖的暂存槽，再执行本地汇总；另一种方式由 sender 主动推送到 owner 上的独立槽位。前者的等分核公式需要满足核数与 PE 数的整除条件，后者仍要等待各来源数据就绪，都不能省掉合并和完成同步。

对称 workspace 具有两条独立的交接关系：生产完成后远端才能读取；所有相关远端读者完成后，本地才能复用。等待本 PE 的 AIC 完成，只证明本 PE 的生产条件，不证明其他 PE 已经写好或读完。具体用集合同步还是 signal/wait，应覆盖实际参与者和本轮槽位，参见 [PE、核与 Tile 分工](work_partition_and_pipeline.md)。

`SetAtomicAdd` 是受支持 DMA 路径的状态配置，不是通用通信引擎的批量归约 API。不能把来源中 MTE 的某个 get 实现推到 SDMA/RDMA，也不能将其固定类型表、64 元素分界、190 KB tile 或跨架构执行单元描述作为已验证限制。来源的 SDMA 伪码混用 GM/UB 计算参数，不能直接作为可编译范例。

验证应覆盖非零 self 贡献、只有 self 的情形、每个远端具有不同可辨认数值、尾块、重复迭代和不均衡 PE。分别检查贡献遗漏/重复、普通写覆盖、暂存槽冲突和过早复用；性能比较还应包含暂存与本地汇总成本。
