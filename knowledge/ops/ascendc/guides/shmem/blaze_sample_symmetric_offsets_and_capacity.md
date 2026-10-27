---
type: Guide
title: Blaze-SHMEM 对称地址、源目标偏移与容量
description: Blaze-SHMEM 对称地址、源目标偏移与容量的来源实现、适用条件与审视结论。
tags:
- task.implementation
- topic.implementation.tiling
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_comm_udma.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/include/kernel/all_to_all_matmul_impl.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/all_to_all_matmul/src/all_to_all_matmul.cpp
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/mc2_architecture.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/comm_shmem.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/workflow_integration.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/blaze-shmem/codebase_map.md
  role: supporting
created_at: '2026-09-11T07:41:58Z'
updated_at: '2026-09-11T07:41:58Z'
platforms: []
---

# Blaze-SHMEM 对称地址、源目标偏移与容量

该UDMA用本PE对称地址作为dst并传remoteRank，GetDataAddrGm取aclshmem_ptr(shmemSpace,selfRank)。不要先换remote物理地址再传仍会翻译的重载。设R、每target输出m、局部Ka、通信行h、D=sizeof(AType)、B个ring槽，dataRow=Ka*D，scaleRow=2*ceil(Ka/64)，bufferBlock=R*h*dataRow。

发送source s到target t的第i轮：src=(t*m+i*h)*dataRow；dst=buffer(i%B)+s*h*dataRow，拷h*dataRow字节。scale一次全m行：src=t*m*scaleRow，dst=B*bufferBlock+s*m*scaleRow。每PE至少B*R*h*Ka*D+R*m*scaleRow字节；bufferBlock已含R，文档某处再乘R会重复估算。对称堆其他元数据/分配开销按真实库合同另计，aclshmem_align返回空指针亦须检查。

PutScaleToRank支持srcMOffset参数却dst未加该偏移，所以只有当前offset0/全量发送合同自然成立；改成分段不能只改参数。self发送被跳过，自己的data/scale区未填，计算必须走local GM。静态槽stride为h，实际tail若改变需明确预留上限与实际拷贝长度。

512B allocation对齐、128B cache hint门槛、32B搬运块和64元素MX分组不是同一概念，不能互换成一条“UDMA总是512B”规则。源码固定1GiB没有公式检查；更多rank/大Ka/深buffer必须检验字节乘积、总容量与指针边界，不能只因分配调用成功就认定layout可容纳。
