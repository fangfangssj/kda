---
type: Guide
title: Tiling 分发、模板实例与运行数据合同
description: 区分 TilingKey、TilingData、BlockDim 和 workspace，保持 Host 决策与已编译 Kernel 分支一致。
tags:
- task.design
- topic.implementation.operator_registration
- topic.implementation.tiling
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/ops/ascendc-registry-invoke-template/references/advanced-guide.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/perf_shape_spec_01.md
  role: supporting
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/utils_api/tiling_template_programming/get_tpl_tiling_key.md
  role: constraint
created_at: '2026-09-10T13:38:09Z'
updated_at: '2026-09-10T13:38:09Z'
platforms: []
---

# Tiling 分发、模板实例与运行数据合同

TilingKey 标识已编译的实现分支；TilingData 传递该次执行需要的运行参数；BlockDim 和 workspace 分别参与执行资源与工作内存配置。它们彼此关联，但不能用其中一项代替其他项。例如选中了双缓冲变体后，Host 的 tile 预算仍需与该变体真实分配匹配。

dtype 可以由构建系统注入的编译宏承载，也可以进入 Tiling 模板参数。算法模式、布局或其他特化维度应与所采用机制一致，避免重复编码、遗漏合法组合或膨胀无效变体。不能仅按声明列表长度相乘就认定最终二进制数，实际还受组合选择与构建过滤影响。

`GET_TPL_TILING_KEY` 按声明的参数顺序和位宽编码；Host 手写的语义编号不能假定等于模板编码值。采用配套声明与生成接口，并核对最终入口映射，可避免选择错实例。已有 [TPL 编码失配案例](../../runbooks/compilation/tiling_key_tpl_bit_encoding_mismatch.md)展示了同一语义分支如何因错误 key 运行到另一个实例。

按 shape 增加特化分支时，选择条件应描述真实数学或资源适用域。每个条件必须映射到实际编译的实例，未命中特化条件的合法输入仍需到达完整实现。保留默认分支不能自动证明其他 shape 性能不变：公共代码、编译选项、tiling 字段和布局仍可能受到影响。

不要把来源中的 key=0、按 1000 递增、固定核数或每分支独立类作为 CANN 通用约定。在已有编码空间中插入新值前，应确认它与 dtype、选项位和编译配置不会碰撞；模板实例可以共享实现，但各自适用条件需明确。

TilingData 的生产与解析需要相同字段含义、单位、类型宽度及布局。每条 Host 路径都要给 Kernel 实际读取的字段赋值，不能把 Host 地址直接作为 Device 可用地址；属性只在 Host 参与决策时，可只传最终参数，参见[属性传递](../programming_guide/advanced_programming/aclnn_operator_development/design_and_implementation/tiling_data_attributes.md)。

注册入口的参数约定应与该框架生成接口一致；直调入口则按自己的签名与 launcher 核对，不能把一个注册样例的参数布局推广到全部直调工程。验证包括 Host 选择值、已生成实例、Kernel 解析值、workspace 容量及空输入/边界/不支持组合。
