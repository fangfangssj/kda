---
type: Guide
title: AIC和AIV独立运行机制
description: AIC和AIV独立运行机制，又称双主模式。在分离模式下，区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，开启双主模式能够提高Matmul计算性能。默认情况下，双主模式不开启，需要通过MatmulConfig中的enableMixDualMaster参数开启。
tags:
- task.lookup
- topic.programming_model.execution_model.simd
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/aic_aiv_independent_execution.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# AIC和AIV独立运行机制<a name="ZH-CN_TOPIC_0000002532388137"></a>

## 功能介绍<a name="zh-cn_topic_0000002299608693_section310824820358"></a>

AIC和AIV独立运行机制，又称双主模式。在[分离模式](../../../../programming_guide/advanced_programming/hardware_implementation/basic_architecture.md#li188191010204418)下，区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，开启双主模式能够提高Matmul计算性能。默认情况下，双主模式不开启，需要通过MatmulConfig中的enableMixDualMaster参数开启。

## 使用场景<a name="zh-cn_topic_0000002299608693_section118051016163613"></a>

算子中的矩阵计算和矢量计算相关代码独立运行，不依赖消息驱动时，可以开启双主模式，以提高Matmul计算性能。

## 约束说明<a name="zh-cn_topic_0000002299608693_section14160134220363"></a>

-   该功能仅支持[Norm模板](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/matmul_config.md)和[MDL模板](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/matmul_config.md)。
-   算子核函数（Kernel）的类型为MIX，同时AIC核数 : AIV核数为1:1。
-   算子核函数（Kernel）的类型为MIX，同时AIC核数 : AIV核数为1:2，且A矩阵和B矩阵同时开启[IBSHARE](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/matmul_usage.md)参数。
-   同一算子中所有Matmul对象的该参数取值必须保持一致。
-   A、B、Bias矩阵只支持从Global Memory输入。
-   获取矩阵计算结果只支持调用[IterateAll](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/iterate_all.md)接口输出到[GlobalTensor](../../../../../apis/simd_api/basic_api/data_structures/global_tensor/index.md)，即计算结果放置于Global Memory的地址，不能调用[GetTensorC](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/get_tensor_c.md)等接口获取结果。

## 调用示例<a name="zh-cn_topic_0000002299608693_section15486294368"></a>

完整的算子样例请参考[开启双主模式的算子样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_mixdualmaster)。

```
// 修改模板参数enableMixDualMaster=true，Norm模板开启双主模式，MDL模板使用GetMDLConfig接口获取模板参数。
constexpr static MatmulConfig MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, true, IterateOrder::ORDER_M, ScheduleType::OUTER_PRODUCT, false, true/*enableMixDualMaster*/);
Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;

// 常规Matmul计算
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
