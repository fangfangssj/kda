---
type: API
title: GetNormalConfig
description: 用于配置Norm模板的参数，获取自定义Norm模板。Norm模板的介绍请参考表模板特性。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/GetNormalConfig.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
source_headers:
- include/adv_api/matmul/tiling.h
---

# GetNormalConfig

## 功能说明

用于配置Norm模板的参数，获取自定义Norm模板。Norm模板的介绍请参考[表模板特性](matmul_config.md#template-features)。

## 函数原型

```
__aicore__ constexpr MatmulConfig GetNormalConfig(const bool intrinsicsLimit = false, const bool batchLoop = false, const bool isVecND2NZ = false, const BatchMode bmmMode = BatchMode::BATCH_LESS_THAN_L1, const bool isMsgReuse = true, const IterateOrder iterateOrder = IterateOrder::UNDEF, const ScheduleType scheduleType = ScheduleType::INNER_PRODUCT, const bool enUnitFlag = true, const bool enableMixDualMaster = false, const BatchOutMode bmmOutMode = BatchOutMode::SINGLE_BATCH)
```

## 参数说明

本接口的所有参数用于设置[MatmulConfig结构体](matmul_config.md#matmulconfig-params)中的参数，其中互相对应的参数的功能作用相同。

**表1**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| intrinsicsLimit | 输入 | 用于设置参数intrinsicsCheck。<br><br>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否开启循环执行数据从Global Memory到L1 Buffer的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：<br>false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不开启循环执行数据的搬入（默认值）。<br>true：当左矩阵或右矩阵在单核上内轴大于等于65535时，开启循环执行数据的搬入。 |
| batchLoop | 输入 | 用于设置参数isNBatch。<br><br>是否多Batch输入多Batch输出。仅对BatchMatmul有效，开启多Batch后，仅支持Norm模板，且需调用[IterateNBatch](iterate_n_batch.md)实现多Batch输入多Batch输出。参数取值如下：<br>false：不开启多Batch（默认值）。<br>true：开启多Batch。 |
| isVecND2NZ | 输入 | 用于设置参数enVecND2NZ。<br><br>开启通过vector指令进行ND2NZ。开启时需要设置[SetLocalWorkspace](set_local_workspace.md)。参数取值如下：<br>false：不开启通过vector指令进行ND2NZ（默认值）。<br>true：开启通过vector指令进行ND2NZ。 |
| bmmMode | 输入 | 用于设置参数batchMode。该参数用于BatchMatmul场景。<br><br>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与L1 Buffer的大小关系。参数取值如下：<br>BatchMode::BATCH_LESS_THAN_L1：多batch数据总和<L1 Buffer Size；<br>BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和>L1 Buffer Size；<br>BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和>L1 Buffer Size。 |
| isMsgReuse | 输入 | 用于设置参数enableReuse。<br><br>[SetSelfDefineData](set_self_define_data.md)函数设置的回调函数中的dataPtr是否直接传递计算数据。若未调用SetSelfDefineData设置dataPtr，该参数仅支持默认值true。参数取值如下：<br>true：直接传递计算数据，仅限单个值。<br>false：传递GM上存储的数据地址信息。 |
| iterateOrder | 输入 | 用于设置参数iterateOrder。<br><br>Matmul做矩阵运算的循环迭代顺序，与[表1](../matmul_tiling/t_cube_tiling_struct.md#tcubetiling-struct)中的iterateOrder参数含义相同。当ScheduleType参数取值为ScheduleType::OUTER_PRODUCT时，本参数生效。参数取值如下：<br><br>ORDER_M：先往M轴方向偏移再往N轴方向偏移。<br><br>ORDER_N：先往N轴方向偏移再往M轴方向偏移。<br><br>UNDEF：当前无效。<br><br>注：Norm模板的Matmul场景、MDL模板使用时，若IterateOrder取值ORDER_M，[TCubeTiling结构](../matmul_tiling/t_cube_tiling_struct.md#tcubetiling-struct)中的stepN需要大于1，IterateOrder取值ORDER_N时，TCubeTiling结构中的stepM需要大于1。MxMatmul仅支持MDL模板。 |
| scheduleType | 输入 | 用于设置参数scheduleType。<br><br>配置Matmul数据搬运模式。参数取值如下：<br>ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运；<br>ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运；设置该取值后，需要与IterateOrder参数配合使用。该配置当前只在BatchMatmul场景（开启Norm模板）或Matmul场景（开启MDL模板或Norm模板）生效。<br>若IterateOrder取值ORDER_M，则N方向循环搬运（在singleCoreN大于baseN场景可能有性能提升），即B矩阵的MTE1搬运并行；<br>若IterateOrder取值ORDER_N，则M方向循环搬运（在singleCoreM大于baseM场景可能有性能提升），即A矩阵的MTE1搬运并行；<br>不能同时开启M方向和N方向循环搬运；<br><br>注：<br>Norm模板的Batch Matmul场景或者MDL模板中，singleCoreK>baseK时，不能设置为ScheduleType::OUTER_PRODUCT取值，需使用默认模式。<br>Norm模板或MDL模板的Matmul场景，仅支持在纯Cube模式（只有矩阵计算）下配置ScheduleType::OUTER_PRODUCT。<br>MDL模板仅在调用[IterateAll](iterate_all.md)计算的场景支持配置ScheduleType::OUTER_PRODUCT。<br>仅在C矩阵输出至GM时，支持配置ScheduleType::OUTER_PRODUCT。 |
| enUnitFlag | 输入 | 用于设置参数enUnitFlag。<br><br>开启UnitFlag功能，使计算与搬运流水并行，提高性能。Norm, IBShare下默认开启，MDL下默认不开启。参数取值如下：<br>false：不开启UnitFlag功能。<br>true：开启UnitFlag功能。 |
| enableMixDualMaster | 输入 | 用于设置参数enableMixDualMaster。<br><br>是否开启MixDualMaster（双主模式）。区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，用于提升性能。该参数默认值为false，仅能在以下场景设置为true：<br>核函数（Kernel）的类型为MIX，同时AIC核数 : AIV核数为1:1。<br>核函数（Kernel）的类型为MIX，同时AIC核数 : AIV核数为1:2，且A矩阵和B矩阵同时开启[IBSHARE](matmul_usage.md#table1188045714378)参数。<br><br>注意，开启MixDualMaster场景，需要满足：<br>同一算子中所有Matmul对象的该参数取值必须保持一致。<br>A/B/Bias矩阵只支持从GM搬入。<br>获取矩阵计算结果只支持调用[IterateAll](iterate_all.md)接口输出到GlobalTensor或者LocalTensor，即计算结果放置于Global Memory或者Local Memory的地址，不能调用[GetTensorC](get_tensor_c.md)等接口获取结果。 |
| bmmOutMode | 输入 | 预留参数。 |

## 返回值说明

[MatmulConfig结构体](matmul_config.md#matmulconfig-params)。

## 约束说明

无



## 调用示例

```
// 配置Norm模板的参数，获取自定义Norm模板
constexpr MatmulConfig MM_CFG = GetNormalConfig(
    /* intrinsicsLimit   */ false,
    /* batchLoop         */ false,
    /* isVecND2NZ        */ false,
    /* bmmMode           */ BatchMode::BATCH_LESS_THAN_L1,
    /* isMsgReuse        */ true,
    /* iterateOrder      */ IterateOrder::UNDEF,
    /* scheduleType      */ ScheduleType::INNER_PRODUCT,
    /* enUnitFlag        */ true,
    /* enableMixDualMaster */ false,
    /* bmmOutMode        */ BatchOutMode::SINGLE_BATCH);
// 常规Matmul计算，最后输出使用自定义Norm模板的计算结果
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(gmBias);
}
mm.IterateAll(gm_c);
mm.End();
```

- [AlltoAllMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_matmul.md)
