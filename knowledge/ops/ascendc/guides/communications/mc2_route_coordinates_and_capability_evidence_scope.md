---
type: Guide
title: MC2 路线坐标与能力证据范围
description: 区分通信资源分配、设备执行路径和样例验证范围。
tags:
- task.design
- topic.implementation.synchronization
- topic.implementation.tiling
- topic.performance.analysis
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/capability-declaration.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/codebase-analysis.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/classification.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/requirement-analysis/comm-path-decision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/operators/collective-comm.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow/step2-investigation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/apace/workflow/step3-design.md
  role: supporting
created_at: '2026-09-11T07:06:47Z'
updated_at: '2026-09-11T07:06:47Z'
platforms: []
---

# MC2 路线坐标与能力证据范围

MC2路线由目标arch、算子语义、调用形态、通信路径及编程抽象共同确定，不能把各列支持项任意组合。来源登记的参考路线包括3510 collective直调blaze-shmem/APACE UDMA，以及2201/3510 MoE直调MTE window；APACE MoE URMA列为planned。它表达此Skill版本的样例覆盖，不是本次新做的硬件验证。未登记/未找到/未验证与物理不支持是不同结论，不能把空表或缺样例提升为禁止所有实现。

HCCL的Host通信域/窗口分配、AscendC Hccl高阶device collective、SHMEM单边通信和APACE CollectiveComm不是同一接口。MoE直调可由Host HcclAllocComResourceByTiling获得window再显式传入；这不等于直接调用依赖框架注入GetHcclContext的注册kernel。来源一处说直调不支持HCCL windows、另一处提供window直调，区别在上下文获取及入口ABI，不能仅按名称判断冲突。

URMA/UDMA在材料中常混称，通信协议、执行引擎、库和具体DMA操作宜分别记录。HCOMM/SHMEM/HCCL的版本和后端关系不能仅从标签或include推定。CCU表同时写直调与无直调参考工程，需记录为该目录缺少已验证入口，不据此宣布所有CCU直调可能或不可能。512B对齐也是特定消息/API布局约束，须落实到地址、长度、scale或window字段，不能套全部通信。

既有代码可从实际编译npu-arch/宏确认目标、从aclshmemx_udma调用确认UDMA路径、从CollectiveComm实例和CommContext确认APACE组合、从window寻址/MTE确认数据通路。只有HcclAllocComResourceByTiling或独立CMake不足以推断底层搬运。MoE专家路由与固定collective也可能组合，按真实数据流分类。

来源禁AscendC::Matmul/Hccl高阶调用是所选直调框架的约束，不是通算融合普遍不能使用高阶API。注册ReduceScatter可作数学和golden参考，但其通信上下文、调度及入口不能直接替换成<<<>>>。每项能力需以同版本入口、依赖和实测范围为证据，本卡保持draft和未定位平台。
