---
type: API
title: SetNextTaskStart
description: 在SuperKernel子Kernel中调用，使其后指令与后续子Kernel并行执行，提升整体性能。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/inter_task_sync/SetNextTaskStart.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a3
source_headers:
- include/basic_api/kernel_operator_common_intf.h
---

# SetNextTaskStart<a name="ZH-CN_TOPIC_0000002337763421"></a>

> [!NOTE]说明 
>本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_common_intf.h"`。

算子层面支持SuperKernel特性Early-Start能力的接口，调用后在TorchAir层面：1.GE图模式下进行默认启用; 2.npugraph_ex后端通过early_start选项进行控制启用。

在[SuperKernel](../../../../../guides/programming_guide/advanced_programming/super_kernel/principles.md)的子核函数（Kernel）中调用，调用后的指令可以和后续其他的子核函数（Kernel）实现并行，提升整体性能。如[图1](#fig37581010773)所示，SuperKernel按序调用子核函数（Kernel），为保证子核函数（Kernel）之间数据互不干扰，会在子核函数（Kernel）间插入算子间同步进行保序，子核函数（Kernel）<sub>N-1</sub>调用该接口后，之后的指令会和后续子核函数（Kernel）<sub>N</sub>实现并行。

SuperKernel是一种算子的二进制融合技术，与源码融合不同，它聚焦于核函数（Kernel） \(核函数（Kernel）\)的二进制的调度方案，展开深度优化，于已编译的二进制代码基础上融合创建一个超级核函数（Kernel）（SuperKernel），以调用子函数的方式调用多个其他核函数（Kernel），也就是子核函数（Kernel）。相对于单算子下发，SuperKernel技术可以减少任务调度等待时间和调度开销，同时利用Task间隙资源进一步优化算子头开销。

**开发者需要自行保证调用此接口后的指令不会与后序算子互相干扰而导致精度问题，推荐在整个算子最后一条搬运指令后调用此接口。**

**图1**  通过SetNextTaskStart实现并行示意图<a name="fig37581010773"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/set_next_task_start_parallel_diagram.png "通过SetNextTaskStart实现并行示意图")

## 函数原型<a name="section620mcpsimp"></a>


- 该原型支持如下产品型号：

    
    Ascend 950PR&950DT系列产品 
    

    
    Atlas A3系列产品 
    

    
    Atlas A2系列产品 
    

    
    Atlas 200I/500 A2推理产品 
    

    
    Kirin X90处理器系列产品 
    

    
    Kirin 9030处理器系列产品 
    

    ```cpp
    template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX, bool FORCE = false>
    __aicore__ inline void SetNextTaskStart()
    ```



- 该原型支持如下产品型号：

    
    Atlas推理系列产品AI Core 
    

    
    Atlas推理系列产品Vector Core 
    

    
    Atlas训练系列产品 
    

    ```cpp
    template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_MTE3, bool FORCE = false>
    __aicore__ inline void SetNextTaskStart()
    ```


## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| AIV_PIPE | SetNextTaskStart之后运行的指令，如果位于AIV上的AIV_PIPE流水，可以与后序算子并行。AIV_PIPE的取值范围为PIPE_MTE2、PIPE_MTE3、PIPE_S、PIPE_V。 |
| AIC_PIPE | SetNextTaskStart之后运行的指令，如果位于AIC上的AIC_PIPE流水，可以与后序算子并行。AIC_PIPE的取值范围为PIPE_MTE1、PIPE_MTE2、PIPE_MTE3、PIPE_FIX、PIPE_M。 |
| FORCE | 离线编译时是否强制启用Early-Start能力，不影响在线编译。<br>false：离线编译时不强制启用Early-Start能力；<br>true：离线编译时启用Early-Start能力，适用于无在线编译流程的算子（如<<<>>>开发的算子）。 |

> [!NOTE]说明 
>不同的硬件架构，每一种硬件流水类型包含的具体流水会有所差异，流水类型详细介绍请参考[硬件流水类型](../intra_core_sync/intra_core_sync_overview.md#zh-cn_topic_0000002542725361_section1272612276459)。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 该接口适用于TorchAir图模式开发场景，且需在启用SuperKernel特性后方可生效。具体内容请参考[图内标定SuperKernel范围](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/features/advanced/super_kernel_scope.md)。
- 在算子运行过程中，需要保证此接口在每个核上都被调用，且每个核上仅被调用一次。
- 若子核函数（Kernel）某个TilingKey分支调用了此接口，则开发者需要保证当前算子可能会运行的所有TilingKey均调用了此接口，否则会出现因同步指令数量不匹配而卡住的现象。
- 本接口内部实现会占用核间同步flagId，若与CrossCoreSetFlag/CrossCoreWaitFlag同时使用需注意避免flagId冲突，详见[核间同步flagId占用说明](../inter_core_sync/key_features.md#inter_core_sync_flagid_usage)。

## 调用示例<a name="section837496171220"></a>

```cpp
AscendC::DataCopy(dstGlobal, dstLocal, TOTAL_LENGTH);
// 在最后一条搬运指令后调用，使之后的指令可与后续子核函数（Kernel）并行执行。
AscendC::SetNextTaskStart();
```

完整样例请参考[TaskSync样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/05_sync_control/task_sync)。

- [QuantMatmulReduceSumWeightNz A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/matmul/quant_matmul_reduce_sum_weight_nz.md)
- [GroupedMatmul A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/gmm/grouped_matmul.md)
