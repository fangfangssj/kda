---
type: Guide
title: MoE 样例 context 窗口寻址与容量
description: MoE 样例 context 窗口寻址与容量的代码事实、前提与失效边界。
tags:
- task.implementation
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/include/moe_dispatch_base_compat.h
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/include/tiling_data.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/mte_dispatch_comm.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/kernel/moe_dispatch.h
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/moe_dispatch_direct_invoke_sample/test/test_moe_dispatch.cpp
  role: supporting
created_at: '2026-09-11T07:32:48Z'
updated_at: '2026-09-11T07:32:48Z'
platforms: []
---

# MoE 样例 context 窗口寻址与容量

compat在__NPU_ARCH__=3510取A5 windowsIn[rank]作为state基址、再加1MiB作为data基址；其他架构取A3本地localWindowsIn作为data、localWindowsExp作为state，远端经remoteRes[rank].nextDevicePtr解释HcclRankRelationResV2后取windows字段。这个ABI选择及固定1MiB偏移只描述来源，不能推广到未知版本；Context实际rankDim与tiling EP必须一致，kernel目前直接采用传入EP而未交叉检查。

HcclA5OpResParam含workspace/size、rankId/rankDim、winSize、64个windowsIn/Out及后续控制字段；A3结构使用大remoteRes表，不能用A5大小/布局调试解码A3。Host无条件按A5结构回拷并打印，debug输出在A3分支不可靠。HcclAllocComResourceByTiling返回设备context，直接传kernel，生命周期归HCCL；Host自己申请的device tiling仍须自行释放。

设D=sizeof(DataType)，payload=AlignUp(H*D,32)，record=payload+12，expertRegion=BS*record，sourceRegion=localE*expertRegion；目标data地址=dataBase(dst)+src*sourceRegion+e*expertRegion+slot*record。状态则stateBase(dst)+(e*EP+src)*32，两者排列轴不同。总data至少EP*localE*BS*record，state至少EP*localE*32，并验证不与保留区或其他owner重叠。payload对齐不等于record整体对齐：H16/half时record44B，后续slot非32B边界，需对应DataCopyPad重载允许而不能假称每记录32B对齐。

Host totalWinSize赋gHcclBufferSize=16，但未据此执行容量检查或证明其单位，不能当已分配16MiB；实际window大小与偏移应从匹配资源合同确认。MAX_RANK_SIZE=64、MAX_EXPERT_SIZE=8仅数组上限，合法expert范围、每source/expert slot<BS及接收数<=maxRecvTokens仍要检查。u64 tiling大量截到u32，形状乘积、字节地址和计数溢出不能靠大类型字段名避免。
