---
type: Guide
title: SHMEM Bootstrap、Team 与公开接口分层
description: 补充默认、MPI、UniqueID 初始化，Team 编号转换及对称分配的接口角色。
tags:
- task.design
- topic.data_movement.communication.collective_communication
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/api.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-code-gen/references/internal-api-boundary.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/op-dsl.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/design-template.md
  role: supporting
created_at: '2026-09-11T03:40:03Z'
updated_at: '2026-09-11T03:40:03Z'
platforms: []
---

# SHMEM Bootstrap、Team 与公开接口分层

来源 SHMEM 接口按 Host 与 Device 分域。aclshmem_* 为标准公开接口，aclshmemx_* 为扩展公开接口，扩展前缀本身不表示 internal；aclshmemi_*/shmemi_* 实现符号不构成公开兼容承诺。legacy examples 的可见调用不能独立证明推荐给新算子的 API 稳定性。本卡未固定 SHMEM 发行版，以下是来源的能力组织，具体签名/可用性需与安装头文件一致。

Host 先建立 ACL/设备上下文，再配置 aclshmemx_init_attr_t 并调用 aclshmemx_init_attr。来源三种 bootstrap：DEFAULT 使用控制面传递 my_pe、n_pes、ip_port、local_mem_size 与 option_attr；MPI 使用进程组传递身份；UNIQUEID 由根 PE 生成 uniqueid，经外部机制广播，再通过 aclshmemx_set_attr_uniqueid_args 配置。同一会话必须共享根信息及总规模，各 PE 身份唯一；test_set_attr 是例程 helper，不能当成所有安装自带 API。

init_status、info_get_version/info_get_name 用于状态和库身份查询；finalize 结束库生命周期，global_exit 对应全局异常退出能力。TLS 配置属于控制面连接设置，不能由它推导数据面加密或任意默认安全状态。

my_pe/n_pes 与 team_my_pe/team_n_pes 分别属于 world 和 team 编号域。team_translate_pe 与 Device team_pe_mapping 用来处理映射，不能把 team 的局部编号直接当 world PE。Host team_split_strided、team_split_2d、team_destroy、team_get_config 分别提供子团队构造、销毁及配置能力；实际可达范围和参与者需与操作的 team 一致。

malloc/calloc/align/free 管理对应对称区域；扩展 xmalloc/xcalloc/xalign/xfree 家族在来源中可区分 DEVICE_SIDE/HOST_SIDE 内存类型（具体符号写作 aclshmemx_malloc 等）。各 PE 对应分配的顺序、大小与释放必须一致，local_mem_size 也需满足一致及对齐要求。默认堆大小或默认分配对齐值不是硬件上限，需要更强对齐时用实际 allocator 合同。

aclshmem_ptr 将本 PE 的 symmetric 对象映射到指定 PE 的对应对象；RoCE 指针接口另有传输注册要求。映射失败/不可达需要处理，获得指针也不代表通信完成。拓扑、buffer 可见性和输出 ownership 共同定义算子：replicated 要求相应 PE 均得到完整结果，sharded 是各自分片，owner_only 仅指定 owner 持有结果。send/recv、reduce、ready/done 等名称不能直接决定 put/get、归约类型或完成协议。

语义中的本地计算、通信和最终合并应独立于 tile/chunk/核数调度；更改调度需保留输入输出、dtype/cast、PE 映射和可见性。DSL 的字段名、先设计分核再填 phase、固定 CATLASS 路径与审批顺序是工程编排约定，不属于运行时接口限制。
