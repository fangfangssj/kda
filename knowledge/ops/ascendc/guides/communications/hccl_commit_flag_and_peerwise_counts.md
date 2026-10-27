---
type: Guide
title: HCCL commit 参数与逐对 rank 计数
description: 纠正阻塞参数误读、AlltoAllV 总和检查和 MoE 路由边界的不足。
tags:
- task.implementation
- topic.data_movement.communication.collective_communication
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/mc2-specific.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-code-review/references/ascendc-topk.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/hccl_communication/hccl_kernel/all_gather.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/hccl_communication/hccl_kernel/allto_all_v.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/hccl_communication/hccl_kernel/commit.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/hccl_communication/hccl_kernel/wait.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/adv_api/hccl_communication/hccl_kernel/finalize.md
  role: constraint
created_at: '2026-09-11T03:21:56Z'
updated_at: '2026-09-11T03:21:56Z'
platforms: []
---

# HCCL commit 参数与逐对 rank 计数

固定知识库 AllGather 定义 template<bool commit=false>：true 表示 Prepare 时通知服务端执行，false 表示暂不通知。它不表示阻塞/非阻塞完成；来源“true 自动包含 Commit/Wait”错误。关闭自动提交的任务需按其协议 Commit，消费输出前还需 Wait 或该路径证明的完成机制；自动提交路径不能无依据再 Commit，否则可能改变重复次数。AlltoAllV 同样要按真实 commit 参数理解。

AllGather 返回的有效 handle 标识准备任务，失败值不能继续用于 Commit/Wait。Commit/Wait 的核类型、顺序、repeat 和细粒度步长须与 Prepare 一致；非细粒度提交与等待次数对应 repeat，细粒度按该 API 的总步骤规则。Finalize 终止该对象的任务服务，sync 配置与多队列先行完成条件分别核对，不能以销毁名字替代数据就绪。Hcomm 的同名 Commit 属于另一句柄协议。

AllGather 中 sendCount 以数据元素计，接收容量包含各 rank 拼接；来源调用表把 strideCount 标成 rankCnt 并以 repeatCount 替换参数名，不能照抄其示意签名。通信 dtype、归约算子、数据布局与实际 buffer 必须匹配，类型枚举支持依目标平台/版本，不将来源 FP8 等静态表当全集。

AlltoAllV 允许各 rank 不同收发数量和本地 displacement。对每对发送 i→接收 j，在对应类型合同下要满足 sendCounts_i[j]=recvCounts_j[i]，并验证各本地 offset+count 不越界及有效接收区布局。所有 rank 总和相同不充分，各数组完全相同也非必要。集合调用的域、次序与参与者要兼容，合法 rank 特化不天然错误。

MoE 的 tokenIdx 数组访问范围与读出的 expertIdx 值域是两次独立验证，仅检查 moeExpertNum<=MAX 不保护两者。还要检查 local/global expert 映射、共享专家分支、rank归属、容量、dispatch permutation 与 combine 逆映射；两侧字段同名同值只是部分条件，名称不同也不证明数学不一致。

per-tensor、per-channel、per-token、per-block、MX per-group 分别指定 scale 共享粒度，实际 shape/轴/编码须与消费者一致。高精度转低精度的舍入、饱和、Inf/NaN 由量化合同决定，饱和不是所有转换的统一修复。EP*TP、rank*stride 的类型提升和乘积范围需要端到端证明。

通信引擎的单次限制、workspace 与实际传输字节是不同量。不能从 sysWorkspaceSize=256MiB 推导 CCU 传输上限，也不能把任意原语字节数都算成count*dtypeBytes*rank。AICPU 并非资源无限；分块/更换引擎后仍需匹配对端次序、支持域和缓冲预算。来源 CCU 英文展开及固定上限未由本批核验，不传播为平台事实。
