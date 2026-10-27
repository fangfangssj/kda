---
type: Guide
title: Ascend C SHMEM 的对称内存、地址可达性与完成语义
description: 区分 PE 对称堆、远端地址映射、RMA 提交、完成和通知，说明不同通信引擎的地址约束需要独立核对。
tags:
- task.design
- topic.memory
- topic.data_movement.communication.collective_communication
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/capacity.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/shmem-ops-generator/skills/shmem-ops-design/references/implementation-boundary.md
  role: supporting
created_at: '2026-09-10T12:45:59Z'
updated_at: '2026-09-10T12:45:59Z'
platforms: []
---

# Ascend C SHMEM 的对称内存、地址可达性与完成语义

SHMEM 程序先建立 PE 身份、通信域及对称堆，再在相应 PE 间使用约定一致的对称分配。对应分配的顺序和大小影响远端对象的定位，不能把某个 rank 的任意本地地址直接当作其他 PE 上同一对象的地址。

地址映射、数据传输与完成通知是三个不同问题。获得可定位的远端指针不说明数据已传输；发起非阻塞操作不说明接收者已经可以消费；可见的通知还必须与对应数据的完成及可见性建立顺序。

## 地址与引擎边界

用户 GM、对称堆、已注册内存和 UB 并不是所有通信引擎都可互换的地址空间。来源区分 MTE、SDMA、UDMA 和 RDMA 路径；是否需要 staging，应由所用引擎、方向和注册要求决定。不能将一个路径支持用户 GM 的结论推广到另一个路径，也不能仅根据指针类型判断可达性。

## 同步作用域

| 机制 | 必须明确的范围 |
|---|---|
| 核内事件或流水同步 | 本核的指定生产/消费流水 |
| 核间交接 | 当前 Kernel 中实际参与的核及对应数据区 |
| quiet / fence 等 SHMEM 接口 | 当前版本中覆盖的操作、引擎、顺序及完成定义 |
| signal / wait 与团队同步 | 对应 PE/team、期次及发布协议 |

不应依据名称将 fence 与 quiet 视为永久等价，也不能假设单个完成接口覆盖所有引擎。来源未固定底层 SHMEM 发行版本，具体签名和完成范围须以实际安装版本为准，本卡保留为 draft。

验证应覆盖多个 PE、重复迭代、异步提交和缓冲复用，并确认消费者在通知后读取到的是本期数据；单次地址可访问不验证通信协议。
