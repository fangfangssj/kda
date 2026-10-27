---
type: Guide
title: SIMT 注册工程的限制与内存配置
description: 提取调用域、共享区与缓存配置知识，限定线程数和工程路径规则。
tags:
- task.implementation
- topic.implementation.api_layer.simt_api
- topic.memory
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/simt-engineering-guide.md
  role: primary
created_at: '2026-09-11T04:05:46Z'
updated_at: '2026-09-11T04:05:46Z'
platforms: []
---

# SIMT 注册工程的限制与内存配置

SIMT VF与自定义callee需匹配目标版本的__simt_vf__/__simt_callee__或等效公共调用域，SIMD callee不能未经兼容证明混用。LAUNCH_BOUND与VF_CALL/Dim3的实际线程几何需符合该版本上界及调用合同，source的constexpr、必须严格相等、禁止TilingData含threadNum是该工程规则，不能一概当所有SIMT语法限制。

来源1024线程、28×32bit参数上限、32–2048范围及32倍数建议均应按实际ABI/硬件确认。通过__ubuf__指针间接传参数需布局、对齐、初始化、容量与可见性，不会自动解除所有参数限制。SIMT可直接用GM_ADDR，也可使用受支持的描述/访存封装；SIMD同样不必在所有路径都用GlobalTensor。

来源ub256KiB减DCache128KiB并SetLocalMemorySize只是配置例，不是各芯片固定可用共享区。SIMD缓冲、SIMT共享、缓存和系统预留共同占用实际资源，减法需有界且API字段含义正确。跨核workspace交换的SetScheduleMode(1)或SetWorkspaceSize示例需核对真实Context接口与调度范围，不替代数据完成、可见性、参与者和ready/reuse协议。

Scalar写GM涉及DCache时需正确写回/失效及后续消费者可见性，这并非SIMT独有、SIMD外壳也可能有Scalar访问。VF内外打印入口分别匹配调用域与格式类型，见[缓存可见性](debug_context_and_cache_visibility.md)。打印时序变化不能当同步正确的证明。

将SIMT头放namespace外有助于避免改变第三方声明作用域；include是否必须arch35前缀取决于真实搜索路径，不是同名头绝对找不到。固定50行、禁cce_parallel和特定目录/入口偏好属于linter/工程选择；接口迁移与lane语义参见[SIMT API边界](api_migration_requires_signature_and_lane_semantics.md)。
