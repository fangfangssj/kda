---
type: Guide
title: APACE context 建链复用与窗口归属
description: APACE context 建链复用与窗口归属的来源契约、适用范围和失效边界。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/architecture.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/fundamentals/communication.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/operator-design/paradigm-mapping.md
  role: supporting
created_at: '2026-09-11T07:14:32Z'
updated_at: '2026-09-11T07:14:32Z'
platforms: []
---

# APACE context 建链复用与窗口归属

来源CommUdmaContext含rankId/rankSize、channelHandles[64]、commBufferAddrs[64]；CommUbmemContext含rank信息与barrier地址。聚合CommContext由各算子tiling头定义而非通用collective_comm_context.h，namespace也不同。入口传__gm__ context指针，tiling按值POD；不可用手工猜的字段顺序/句柄替代有效建链结果。

Host先创建HCCL域，CommChannelBuilder用相同engine创建/获取/拷贝context，数据URMA与barrier UBMEM分别建链。self channel留0，self数据地址有效。CreateDeviceContext后来源要求跨rank Host barrier才能launch，返回仅代表本rank建链阶段结束。rankSize<=定长数组容量，窗口总量用HcclGetHcclBuffer实测，不能按配置期望值代替。

ctxTag命中已存在context会直接返回并跳过填充/建链；tag需与通信域、资源布局和版本对应，复用错误tag会保留旧指针。来源engine常量4是该builder值。engine创建的devContext所有权不能凭普通aclMalloc对象类比：两份ST一份aclrtFree一份不释放，正确释放接口需以匹配HCCL版本的真实所有权合同核实，本次不认定任一ST做法通用正确。

data+scale可分段共用Win：示例每rank数据chunkBytes=m*Ka*sizeof(data)，scale从R*chunkBytes后开始，完整容量另加scale、padding及任何metadata。若barrier在独立2MiB区，数据offset0可用；共享窗口头含metadata时必须跳过真实预留量。Host分配、发送、接收、归约读取四处同源；128B仅某种共享布局例子。UB单对象512+32=544B，双对象512*2+32=1056B，之后计算buffer也计入预算。

来源2026-08-30漂移记录报告上游block/aiv_comm迁core/aiv_comm、kernel迁fusions及新增ReduceScatter UBMEM模板，属于该引用快照的观察，未作为本次拉取核验。它也报告dcci刷新的是跨launch CommContext，与staging数据事件不是一个对象。版本升级需同时匹配include、ABI、barrier和资源生命周期；不能用pin目录缺实现否定其他版本支持。
