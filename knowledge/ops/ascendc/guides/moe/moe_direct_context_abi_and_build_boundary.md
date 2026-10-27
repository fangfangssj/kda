---
type: Guide
title: MoE 直调 context ABI 与构建边界
description: MoE 直调 context ABI 与构建边界的接口、实现假设及失效边界。
tags:
- task.design
- topic.mixture_of_experts
- topic.implementation.synchronization
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/build-framework.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/sample-helper-map.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/change-routing.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/index.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-mc2-best-practice/references/foundations/ascendc-api/moe-dispatch-combine/samples/spec-template.md
  role: supporting
created_at: '2026-09-11T07:25:58Z'
updated_at: '2026-09-11T07:25:58Z'
platforms: []
---

# MoE 直调 context ABI 与构建边界

sample将Host设备/通信域/stream初始化与kernel主流程分开，Host通过HcclAllocComResourceByTiling(comm,stream,完整mc2Tiling,&mc2Context)取得资源并把地址传给kernel。源明确不是只传mc2CcTiling，也不能漏stream。注册路径可由框架注入，直调须显式创建；这也说明“HCCL window一律不能直调”不成立，关键是context来源与入口ABI。

tiling头中的HcclA3/A5OpResParam布局与底层资源ABI绑定，兼容层只解释平台字段/基础window地址；comm层在基址上叠加data/state偏移并管理ready；主kernel处理route、slot和输出。逻辑分层可避免把平台字段、业务索引和状态动作混为一处，但文件名/组织不是运行正确性的强制形式。

来源CMake有ascendc kernel共享库和Host测试可执行，链接tiling_api/platform/pthread/hccl/hccl_fwk/hcomm等，寻找compiler/tikcpp或tools/tikcpp的ascendc.cmake。文件后缀.cpp并不自动意味着普通Host编译，实际编译器和target决定设备代码处理。ABI=0是此sample的构建设置，需和实际链接CANN/HCCL版本一致，不能宣称所有HCCL永远要求此值。

__NPU_ARCH__==3510选A5结构，否则选A3是来源compat写法；未知arch默入else不证明支持。SOC_VERSION需被对应CMake/编译链消费并确认真实arch宏，仅定义一个Host字符串宏不会自动改变Device架构。__has_include正式moe_distribute_base头优先、缺失用compat能选择来源，但文件存在不保证签名/布局相容。

修改入口参数需Host声明、launch包装和kernel定义一致；资源结构不能随意重排，新增业务tiling也须核对Host/Device消费及sizeof/offset。版本升级可能同时影响API和ABI，不是只改target_link_libraries即可。测试必须绑定每rank设备映射、完整资源、真实运行结果和独立输出校验，构建成功不替代多卡可运行。
