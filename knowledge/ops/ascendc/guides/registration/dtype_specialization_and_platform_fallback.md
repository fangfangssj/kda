---
type: Guide
title: dtype 特化、TilingKey 编码与平台 Fallback
description: 补充存储/索引/计算类型组合、分发位域一致性和原生能力不足时的实际转换依赖。
tags:
- task.design
- topic.compilation
- topic.implementation.operator_registration
status: draft
sources:
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/D1_mixed_precision.md
  role: primary
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/D2_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/D3_tilingkey_dispatch.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/cards/D5_bf16_special_handling.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/D1_mixed_precision.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/D2_template_kernel.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/D3_tilingkey_dispatch.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/playbooks/D5_bf16_special_handling.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/D1.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/D2.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/D3.yaml
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ops-perf-evolution/skills/evolution-strategies/references/preconditions/D5.yaml
  role: supporting
created_at: '2026-09-11T01:26:08Z'
updated_at: '2026-09-11T01:26:08Z'
platforms: []
---

# dtype 特化、TilingKey 编码与平台 Fallback

一套模板可以分离存储类型 U、计算类型 T；索引算子还可独立参数化 IDX_T。支持列表必须描述允许的完整组合，而不是分别声明多个 dtype 就推断其笛卡尔积全部可运行。纯拷贝路径可令 T=U 并省去数值转换；整数索引不得因计算模板默认 float 而失去精度。

OpDef 的输入/输出 DataType、Format 和 UnknownShapeFormat，Host 校验与 Tiling、Kernel 模板实例、最终构建产物必须对应。多个 dtype 本身不要求把 REQUIRED 改成 DYNAMIC；AutoContiguous 也不是增加 dtype 的必选动作，其影响应按算子输入布局合同处理。参见[注册类型组合](opdef_dtype_combinations_and_binary_config.md)。

编译期模板实例化与运行时选择实例是不同阶段。运行时 dtype 可以用于 Host 或入口分发，`if constexpr` 的条件则必须在编译期成立；不能把读出的运行时 dtype 直接当作模板参数。Host Tiling 代码不能普通调用 `__aicore__` Compute 来代替 Kernel 启动。对于依赖模板的无效分支，if constexpr 可避免实例化，但不代表所有普通 if 均禁止。

TilingKey 的数值和位域是本工程分发协议。例如若约定 `key=(outId<<8)|inId`，低 8 位就是输入 ID；half 输入 ID=1、FP32 输出 ID=0 的 key 应为 0x0001，不能与倒置的 0x0100 示例混用。检查字段范围、移位宽度、其他 shape/算法/平台位域、编码碰撞和全部可达分支。不同 dtype 是否必须占不同 key，取决于框架是否已在其他编译维度区分它们；1/2/3 不是通用固定枚举。

源文的 REGISTER_KERNEL_API、SetTilingKeyMode、GetChipName/GetChipFeature/GetXXXConfig 等还可能是项目辅助层，不能由名称推断公共 API。模板与函数分发的 2/4/8 类型阈值属于组织建议，不是性能定律；多个实例的编译与二进制成本仍存在，换成 TilingKey 不会自动消除模板膨胀。

平台适配需要分别确认 dtype 转换、计算、搬运、布局和 shape 能力。某条 L1→BT 搬运的 BF16 feature 不代表整个 Vector 支持 BF16；某算子在 910_95 上允许 (g,n)、其他分支仅允许 (1,n)/(n,) 的记录，不能推出芯片只有该 shape 能力。Kirin/310P/910B/910_93 不能未经验证共享配置，未知平台也不能靠一个“默认支持”分支获得兼容性。

BF16 存储、FP32 计算的 fallback 仍依赖 BF16→FP32、FP32→BF16 的可用实现及正确输出 ABI。若目标连这些转换也不支持，需要已核实的另一实现或明确拒绝，不能只设置 needsCast=true。片上逐 tile 转换只需对应 UB；只有采用全量 GM staging 时才按 `totalElems*4` 等实际数量申请 workspace，两者不是同一资源。每条路径要计算自己的队列份数、scratch、对齐和有效长度。

验证注册支持集合、Host key、实际编译分支和最终二进制的一致性，覆盖不支持组合、所有 dtype、平台路径和尾块。固定收益 2x、20% 才值得适配、所有 BF16 原生或 fallback 的硬件表没有从这些模板中获得独立证据。
