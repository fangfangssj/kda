---
type: Guide
title: 设备 payload 常驻与 Host 索引交换
description: 为路由和重排行为保留索引编码、稳定排序及总线字节模型。
tags:
- task.optimization
- topic.data_movement
- topic.compute.indexing
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/discovered/disc_X2.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/basic_api/resource_management/t_que_bind/t_que_bind_intro.md
  role: constraint
created_at: '2026-09-11T02:29:01Z'
updated_at: '2026-09-11T02:29:01Z'
platforms: []
---

# 设备 payload 常驻与 Host 索引交换

当Host需要协调gather/scatter时，可让N×H的payload继续留在设备，仅回传N×K的索引/路由元数据，Host生成排列后将必要索引传回，最终由设备搬运payload。收益来自减少Host–Device传输，不是消除设备上真实gather读写；若排序也能合法在设备完成，应比较两条完整路径，而不能笼统断言NPU不支持int32排序。

总线字节分别按方向计数：payload=N*H*sizeof(value)，index=N*K*sizeof(index)，此外计返回排列、长度和其他元数据。H=4608、K=1在相同元素宽度下单向规模比为4608；half值和int32索引则是2304。N或K为0需要空路径，不能用比值或取模继续计算。

Host排序需保持原算子的升降序、稳定性和并列值规则。dst_to_src[dst]%N只适用于扁平ID以row+slot*N编码的约定；若ID=row*K+slot，则原行是ID/K。逆排列、重复路由、topK slot身份和输出元数据要共同还原，不能只保证payload值集合相同。

设备每行按真实H与物理对齐跨度搬运；DataCopyExt的blockLen和相应stride单位逐字段核对，UB右padding不自动使任意GM地址合法。源宽、输出宽、尾部有效字节与缓冲大小分别计算。纯搬运可用受支持的TQueBind或显式MTE2→MTE3完成协议，不要求必有两份VECIN/VECOUT独立队列；也不能把一个VECIN队列直接用于未受保护的搬出。

E2E计时包含等待索引可读、Host稳定排序、索引回传、Kernel及必要同步。来源0.17/0.093约1.828，并非1.87，约100微秒等案例未复现。验证N/K边界、重复专家、并列键、非连续payload、非对齐H和排列往返。
