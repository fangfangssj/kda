---
type: API
title: DataCacheCleanAndInvalid
description: 解决Data Cache与GM一致性问题，将脏数据写回GM并将指定Cache Line标记为无效，确保下次访问从GM重新加载。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.data_movement.memory_access
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cache_control/DataCacheCleanAndInvalid.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_cache_intf.h
---

# DataCacheCleanAndInvalid<a name="ZH-CN_TOPIC_0000001787873306"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_cache_intf.h"`。

如图1所示，在AI Core内部，Scalar单元和DMA单元都可能对GM进行访问：

- DMA搬运单元读写GM，数据通过DataCopy等接口在Unified Buffer（UB）等Local Memory和GM间交互，没有Cache一致性问题。
- Scalar单元访问GM，首先会访问每个核内的Data Cache（DCache），因此存在DCache与GM的Cache一致性问题，具体原因请参考[缓存一致性](../../../../guides/programming_guide/advanced_programming/memory_model/cache_coherence.md)。

**图1**  DCache内存层次示意图<a name="zh-cn_topic_0000002530021782_fig1161014168448"></a>

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/datacache.png)

DataCacheCleanAndInvalid接口用于解决上述一致性问题，其功能可以拆解为两部分：

- Clean：将缓存中被修改过的数据（脏数据，Dirty Data）写回到GM中，避免数据丢失。
- Invalid：
    - 将指定地址范围的Cache Line标记为“无效”，使其从缓存中移除。
    - 保证下一次访问这些内存地址时，会从GM重新加载数据，而不是使用可能过期的缓存数据。

该接口的典型使用场景如下：

- 读取GM的数据，但该数据可能在外部被其余核修改，此时需要使用DataCacheCleanAndInvalid接口，直接访问GM，获取最新数据。
- 用户通过Scalar单元写GM的数据，希望立刻写出，也需要使用DataCacheCleanAndInvalid接口。


- 针对Ascend 950PR&950DT系列产品，原子操作过程中，如果希望改变后续数据的饱和模式，需要先使用DataCacheCleanAndInvalid接口将Cache Line中现存的数据立刻写出，再调用[SetCtrlSpr](../special_register_access/set_ctrl_spr_isasi.md)设置后续数据的饱和模式。


## 函数原型<a name="section620mcpsimp"></a>

- 配置dcciDst的原型

  ```cpp
  // 支持通过配置dcciDst确保DCache与GM存储的一致性。
  template <typename T, CacheLine entireType, DcciDst dcciDst>
  __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
  // 支持通过配置dcciDst确保DCache与Local Memory存储的一致性。
  template <typename T, CacheLine entireType, DcciDst dcciDst>
  __aicore__ inline void DataCacheCleanAndInvalid(const LocalTensor<T>& dst)
  ```

- 不支持配置dcciDst，仅支持保证DCache与GM的一致性。

    ```cpp
    template <typename T, CacheLine entireType>
    __aicore__ inline void DataCacheCleanAndInvalid(const GlobalTensor<T>& dst)
    ```


> [!NOTE]说明 
> 对于Atlas推理系列产品AI Core，仅支持不配置dcciDst参数的原型。



## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | dst的数据类型。 |
| entireType | 指令操作的模式：<br>SINGLE_CACHE_LINE：只刷新传入地址所在的Cache Line，**注意如果该地址非64B对齐，只会操作传入地址到64B对齐的部分。**<br>ENTIRE_DATA_CACHE：此时传入的地址无效，核内会刷新整个DCache，但是耗时较大，**性能敏感的场景慎用**。 |
| dcciDst | 表示使用该接口来保证DCache与哪一种存储保持一致性，类型为DcciDst枚举类。<br>&bull; CACHELINE_ALL：与CACHELINE_OUT效果一致。<br>&bull; CACHELINE_UB：表示通过该接口来保证DCache与UB的一致性。<br>&bull; CACHELINE_OUT：表示通过该接口来保证DCache与GM的一致性。<br>&bull; CACHELINE_ATOMIC：原子操作过程中保证DCache和GM的一致性。<br> |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| dst | 输入 | 需要刷新Cache的Tensor。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- 当`entireType`为`SINGLE_CACHE_LINE`时，`dst`生效，`dcciDst`不生效。
- 当`entireType`为`ENTIRE_DATA_CACHE`时，`dst`不生效，`dcciDst`生效。
- `dst`为`LocalTensor`类型的函数原型仅支持在AIV中调用。

## 调用示例<a name="section837496171220"></a>

- **示例1：SINGLE\_CACHE\_LINE模式（64B对齐场景）**  
  假设mmAddr\_为0x40（64B对齐），调用一次指令即可刷新整个Cache Line。

    ```cpp
    AscendC::GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
    for( int i = 0; i < 8; i++) {
        global.SetValue(i, AscendC::GetBlockIdx());
    }
    // 由于首地址64B对齐，调用DataCacheCleanAndInvalid指令后，会立刻刷新前8个数。
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
    ```

- **示例2：SINGLE\_CACHE\_LINE模式（非64B对齐场景）**  
  假设mmAddr\_为0x20（非64B对齐），需要调用两次指令才能刷新完整Cache Line。

    ```cpp
    AscendC::GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
    for( int i = 0; i < 8; i++) {
        global.SetValue(i, AscendC::GetBlockIdx());
    }
    // 由于首地址非64B对齐，调用1条指令，只会刷新起始地址至64B字节对齐的部分，即前4个数。
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
    // 需要再次调用DataCacheCleanAndInvalid指令，刷新后4个数。
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global[4]);
    ```

- **<a name="example3_multi_core"></a>示例3：SINGLE\_CACHE\_LINE模式（多核处理场景）**  
  假设mmAddr\_为0x40（64B对齐），多核操作同一条Cache Line时可能出现随机覆盖。本样例仅做示例说明，便于开发者理解使用限制，非正常使用样例。

    ```cpp
    AscendC::GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_);
    global.SetValue(AscendC::GetBlockIdx(), AscendC::GetBlockIdx());
    // 算子中多核操作虽然不在同一个地址，但在同一个Cache Line，会出现数据的随机覆盖，和通用CPU的行为不同。
    // 调用DataCacheCleanAndInvalid指令后，由于多核操作的时间不一致，最终结果存在随机性，后执行的核会覆盖前面核的结果。
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
    ```

- **示例4：ENTIRE\_DATA\_CACHE模式**  
  假设mmAddr\_为0x20（非64B对齐），刷新整个DCache。本样例仅做示例说明，便于开发者理解使用限制，非正常使用样例。

    ```cpp
    AscendC::GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer((__gm__ uint64_t*)mmAddr_ + AscendC::GetBlockIdx() * 1024);
    for( int i = 0; i < 8; i++) {
        global.SetValue(i, AscendC::GetBlockIdx());
    }
    // 刷新整个DCache，性能较差。
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(global);
    ```

- [MatmulAllReduceAddRmsNorm A2 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/matmul_all_reduce_add_rms_norm.md)
- [MaskedSelect A2/A3 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/masked_select.md)
- [ForeachNonFiniteCheckAndUnscale 950 ACLNN 算子知识](../../../../operators/aclnn/nn/foreach/foreach_non_finite_check_and_unscale_950.md)
- [grouped_matmul_add A2/A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/gmm/grouped_matmul_add.md)
- [RepeatInterleave 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/repeat_interleave_950.md)
- [AttentionToFfn A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/attention_to_ffn.md)
- [RepeatInterleaveWithDim 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/repeat_interleave_with_dim_950.md)
- [DistributeBarrier A3 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/distribute_barrier.md)
- [TopKTopPSample A2/A3 ACLNN 算子知识](../../../../operators/aclnn/nn/index/top_k_top_p_sample.md)
- [DistributeBarrier 950 ACLNN 算子知识](../../../../operators/aclnn/transformer/mc2/distribute_barrier_950.md)
- [TopKTopPSample 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/top_k_top_p_sample_950.md)
