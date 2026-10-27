---
type: Guide
title: 比较与选择：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。寄存器侧比较与选择类接口汇总，提供eq、ge、gt、le、lt、ne等逐元素比较及select、squeeze选择能力。
tags:
- task.lookup
- topic.compute.elementwise.comparison_logic
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/compare_and_select/compare_and_select.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_squeeze.md
  role: primary
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/c_api/reg/compare_and_select/asc_squeeze_with_status_deprecated.md
  role: constraint
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/a70025d581ae45da8a93b4938b5ba19da6fa6f92/knowledge/ops/ascendc/apis/simd_api/c_api/reg/reg_store/asc_squeeze_and_storeunalign.md
  role: constraint
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_squeeze 接口选型
---

# 比较与选择

寄存器侧比较与选择接口提供逐元素比较和条件选择能力。比较运算含asc_eq/ge/gt/le/lt/ne及对应_scalar标量变体，生成掩码结果。
[asc_select](asc_select_reference_contract.md) 根据掩码从两个源中逐元素选择。[asc_squeeze](asc_squeeze_reference_contract.md) 将被选中的源元素按原顺序压紧到目的寄存器低位，尾部清零；它不是生成解压索引，也不写 AR。

[asc_squeeze_v2](asc_squeeze_deprecated_reference_contract.md) 与 [asc_squeeze_with_status](../../apis/simd_api/c_api/deprecated_interface/asc_squeeze_with_status_deprecated.md) 已废弃。只需要寄存器压紧时可使用普通 squeeze；需要筛选、AR 状态和连续非对齐搬出到 UB 时，应使用 [asc_squeeze_and_storeunalign](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze_and_storeunalign.md) 及其 init/finalize 配套接口，不能丢掉旧调用依赖的地址推进语义。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_select](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_select.md)
- [asc_squeeze](../../apis/simd_api/c_api/reg_compute/reg_permute_sel/asc_squeeze.md)
