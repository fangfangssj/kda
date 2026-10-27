---
type: Guide
title: AscDiv：固定 reference 调用约束
description: 固定 reference 的调用约束与版本边界。根据mask对源操作数src0、src1按元素求商，将结果写入目的操作数dst。
tags:
- task.lookup
- topic.programming_model.execution_unit.vector
status: stable
sources:
- resource: https://gitcode.com/qq_41667743/cannbot-knowledge/blob/81e3f36881d82c4f5d8aa1aa10c7afa8e14fd711/knowledge/ops/ascendc/apis/simd_api/c_api/reg/arithmetic_compute/asc_div.md
  role: supporting
- resource: https://gitcode.com/cann/cannbot-skills/blob/502e3ed33f1440aa01bdc5a5bd396e4bf539c9ea/plugins-community/ascendc-port-orchestrator/kb/target/ascendc/migration/reg-base-vector/reg_vector/asc_div.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-14T13:31:01Z'
platforms:
- '950'
aliases:
- asc_div
---
# AscDiv

该接口根据mask，对源操作数src0、src1进行按元素求商操作，将结果写入目的操作数dst。

参数：dst(输出): 目的操作数（矢量数据寄存器）。；src0(输入): 源操作数（矢量数据寄存器）。；src1(输入): 源操作数（矢量数据寄存器）。；mask(输入): 源操作数掩码（掩码寄存器）。用于指示在计算过程中哪些元素参与计算。对应位置为1时参与计算，为0时不参与计算。mask未筛选的元素在输出中置零。

现有官方归档卡记载的数据类型：float、half、int16_t、int32_t、uint16_t、uint32_t。下方固定 Skill 来源仅列 half/float 声明，整数范围未通过该来源复核。

约束：- mask控制源操作数是否参与计算，源操作数不参与计算的元素在输出对应位置置零。

- 注意除零错误；具体结果依所选 dtype 与目标版本合同核对。

仅 Ascend 950PR/Ascend 950DT 支持。

## 固定来源的寄存器 C 声明

固定 950PR/950DT Skill reference 明确列出以下两种浮点寄存器重载：

```cpp
__simd_callee__ inline void asc_div(vector_half& dst, vector_half src0, vector_half src1, vector_bool mask);
__simd_callee__ inline void asc_div(vector_float& dst, vector_float src0, vector_float src1, vector_bool mask);
```

dst 是输出，src0/src1 是输入寄存器；mask 选中位置计算 `dst_i = src0_i / src1_i`，未选中输出清零。返回 void，流水为 PIPE_V。注意这里的 mask=0 与 [asc_select](asc_select_reference_contract.md) 取第二输入的选择语义不同。

现有官方归档卡另记载 int16_t、int32_t、uint16_t、uint32_t。该较旧 Skill 文档未列出整数声明，既不能据此删去现有归档中的整数范围，也不能用它补造整数签名、舍入模式或除零返回值。使用整数路径时需回到对应归档/实际 SDK 取得该版本声明；两种来源的范围应分别保留。

[a2/a3 的同名接口](../../apis/simd_api/c_api/vector_compute/vector_arith/asc_div.md) 接受 UB 指针与 count，平台和参数不同，不能用它的原型或 UB 对齐条件替代这里的寄存器 API。验证需覆盖所选重载、mask 选中/未选中 lane，并按真实 SDK 的数值合同处理零除数；本卡没有编译或上板结果，来源仅提醒除零错误，未定义所有异常输入的输出值。

## 官方接口与版本核对

上述声明和约束以本卡固定 reference 为边界；其中的完整重载清单仅指该份来源，不能据此排除新版 SDK 新增的重载或数据类型。当前官方文档见下列链接，接口名称相同不代表不同版本的签名、状态寄存器或掩码语义等价。

- [asc_div](../../apis/simd_api/c_api/reg_compute/reg_arith/asc_div.md)
