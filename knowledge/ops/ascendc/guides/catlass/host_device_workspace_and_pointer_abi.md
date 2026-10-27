---
type: Guide
title: CATLASS Host Device 与 Workspace 指针 ABI
description: 保留适配器、Params、别名和系统前缀的实现知识。
tags:
- task.implementation
- topic.implementation.synchronization
- topic.precision
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/architecture/02-device-calling.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/branch-instantiation.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/quant-matmul.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/patterns/with-epilogue.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/catlass-op-develop/references/rules.md
  role: supporting
created_at: '2026-09-11T05:12:50Z'
updated_at: '2026-09-11T05:12:50Z'
platforms: []
---

# CATLASS Host Device 与 Workspace 指针 ABI

DeviceGemm是Host侧workspace与stream启动适配器，不应在Device函数体构造；Kernel{}(params)是设备functor调用，可由手写global入口或ACLNN设备封装调用。Arguments转Params必须检查真实字段及to_underlying_arguments行为，不能假设两者布局一致。

裸launch明确传入用户workspace时应保持该指针语义；框架workspace是否需要GetUserWorkspace取决于系统区、运行时初始化和实际接口，不取决于用了CATLASS这个名字。来源主体要求直传，分支表又写GetUserWorkspace，二者不能直接拼接。来源针对CANN9.1beta1/arch2201描述GetUserWorkspace可能通过KFC和约16MiB保留区获得用户基址，未在本批核验外部实现，只能作为该版本排障线索，不能推广成所有版本函数忽略入参或SetSysWorkspaceForce全局禁止。

MatmulEpilogue的Arguments转换在来源版本可能默认令X与D共址。Y=AB-X需要额外输入X和输出Y分别绑定，检查减法方向、shape/stride和允许的别名；手工构造Params是一种方式，已确认支持独立指针的适配器也可以。仅改变Host参数列表而Device Params仍别名不能生效。

__gm__地址空间必须贯穿类型转换，普通void*/uint8_t*与GM指针不能因位宽相同就任意互换。量化Params中的per-channel与per-token scale必须对应实际N/M维及stage消费方式，workspace尺寸由完整Kernel查询或等价推导确定；某字段写sizeof(float)可能表达每元素字节，不能当作总workspace仅4B。

多dtype、转置和swizzle由实际编译实例覆盖；2种dtype×4种转置×2种swizzle是16组合的上界，合法子集需按模板和布局合同确定，不能宣称必然大幅少于16。可在分支外声明共用workspace等Host变量，或用合法模板helper组织实例；变量作用域与设备资源生命周期不是一回事。
