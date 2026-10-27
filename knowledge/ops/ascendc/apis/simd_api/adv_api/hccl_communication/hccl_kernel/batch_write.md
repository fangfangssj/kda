---
type: API
title: BatchWrite
description: 集合通信BatchWrite任务下发接口，实现点对点批量数据传输，可同时将多份数据发送到不同Global Memory地址。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.data_movement.communication.collective_communication
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/BatchWrite.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- a2
- a3
source_headers:
- include/adv_api/hccl/hccl.h
---

# BatchWrite

## 功能说明

集合通信BatchWrite的任务下发接口，返回该任务的标识handleId给用户。BatchWrite实现了一种点对点通信，这是一种直接传输数据的通信模式，能够同时将多份数据发送到不同的Global Memory地址上。


对于Atlas A3系列产品，BatchWrite通信支持在相同或不同的昇腾AI Server之间进行。



对于Atlas A2系列产品，BatchWrite通信必须在不同昇腾AI Server（通常是8卡或16卡的昇腾NPU设备组成的服务器形态的统称）之间进行。


**图1**  BatchWrite示意图  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/batchwrite_diagram.png "BatchWrite示意图")

## 函数原型

```
template <bool commit = false>
__aicore__ inline HcclHandle BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum, uint16_t queueID = 0U)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| commit | 输入 | bool类型。参数取值如下：<br>true：在调用Prepare接口时，Commit同步通知服务端可以执行该通信任务。<br>false：在调用Prepare接口时，不通知服务端执行该通信任务。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| batchWriteInfo | 输入 | 通信任务信息的Global Memory地址。一组通信数据的相关信息必须按指定的格式保存，在执行通信任务时，可以同时指定多组通信任务信息，执行通信任务时批量发送数据。结构体定义如下代码所示。 |
| itemNum | 输入 | 批量任务的个数。该参数取值必须与batchWriteInfo中通信任务信息的组数一致。 |
| queueID | 输入 | 指定当前通信所在的队列ID，默认值为0。 |


```
// Atlas A3系列产品
struct BatchWriteItem {
    uint64_t type;
    uint32_t res1[5];
    uint32_t length;
    uint32_t srcAddrLow;
    uint32_t srcAddrHigh;
    uint32_t dstAddrLow;
    uint32_t dstAddrHigh;
    uint32_t res2[4];
};
```



```
// Atlas A2系列产品
struct BatchWriteItem {
    uint64_t localBuf;
    uint64_t remoteBuf;
    uint64_t count;
    uint32_t dataType;
    uint32_t remoteRankId;
};
```


## 返回值说明

返回该任务的标识handleId，handleId大于等于0。调用失败时，返回 -1。

## 约束说明

-   调用本接口前确保已调用过[InitV2](init_v2.md)和[SetCcTilingV2](set_cc_tiling_v2.md)接口。
-   若HCCL对象的[config模板参数](hccl_template_params.md#hccl-template-params)未指定下发通信任务的核，该接口只能在AIC核或者AIV核两者之一上调用。若HCCL对象的[config模板参数](hccl_template_params.md#hccl-template-params)中指定了下发通信任务的核，则该接口可以在AIC核和AIV核上同时调用，接口内部会根据指定的核的类型，只在AIC核、AIV核二者之一下发该通信任务。
-   一个通信域内，所有Prepare接口和InterHcclGroupSync接口的总调用次数不能超过63。

-   对于Atlas A2系列产品，当前接口仅支持不同AI Server间的通信，同时通信任务信息中指定的目的卡号不能是本卡号。

-   通信任务信息写入batchWriteInfo前，必须通过调用[DataCacheCleanAndInvalid](../../../basic_api/cache_control/data_cache_clean_and_invalid.md)接口，保证预期的数据成功刷新到Global Memory上。

## 调用示例


-   不同AI Server之间的点对点通信

    在Atlas A2系列产品上，假设本卡要将不同的数据分别发送到其它AI Server的2卡、3卡的指定位置，通过调用一次BatchWrite接口，实现批量点对点通信。

    ```
    struct BatchWriteItem {
        uint64_t localBuf;     // 本端发送数据的window地址
        uint64_t remoteBuf;    // 对端接收数据的window地址
        uint64_t count;        // 发送的数据个数
        uint32_t dataType;     // 发送的数据类型
        uint32_t remoteRankId; // 发送数据的目的卡号
    }; // 按接口的约定定义格式

    extern "C" __global__ __aicore__ void BatchWrite_custom(GM_ADDR inputGM, GM_ADDR workspace, GM_ADDR tilingGM)
    {
        REGISTER_TILING_DEFAULT(BatchWriteCustomTilingData); // BatchWriteCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(BatchWriteCustomTilingData, tilingData, tilingGM);
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();

        if constexpr (g_coreType == AscendC::AIV) {
            Hccl hccl;
            hccl.InitV2(contextGM, &tilingData);
            hccl.SetCcTilingV2(offsetof(BatchWriteCustomTilingData, mc2CcTiling));

            __gm__ BatchWriteItem* sendInfo = reinterpret_cast<__gm__ BatchWriteItem*>(workspace);

            // 需要提前将待发送的数据从inputGM搬运到localBuf所填的window地址上
            sendInfo->localBuf = hccl.GetWindowsOutAddr(hccl.GetRankId());
            // 对端的接收地址也要是window地址，接收端需要考虑是否搬运到输出或者workspace上
            sendInfo->remoteBuf = hccl.GetWindowsInAddr(2U);
            sendInfo->count = 16U;
            sendInfo->dataType = HcclDataType::HCCL_DATA_TYPE_FP16;
            sendInfo->remoteRankId = 2U;

            // 可以组装多个通信任务，实现批量发送
            (sendInfo + 1)->localBuf = hccl.GetWindowsOutAddr(hccl.GetRankId());
            (sendInfo + 1)->remoteBuf = hccl.GetWindowsInAddr(3U);
            (sendInfo + 1)->count = 32U;
            (sendInfo + 1)->dataType = HcclDataType::HCCL_DATA_TYPE_BFP16;
            (sendInfo + 1)->remoteRankId = 3U;

            // 确保cache中的数据已刷新到GM地址上
            GlobalTensor<int64_t> tempTensor;
            tempTensor.SetGlobalBuffer((__gm__ int64_t*)sendInfo);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tempTensor);

            auto handleId = hccl.BatchWrite<true>(sendInfo, 2U);
            // wait仅表示本端发送完毕，对端是否接收到数据需要在对端判断
            hccl.Wait(handleId);
            AscendC::SyncAll();
            hccl.Finalize();
        }
    }
    ```

    当通信数据量较大时，可以在Tiling流程中调用SetAicpuBlockDim接口来设置AI CPU的核数。算子内部将自动在多个AI CPU核中选择最优的核进行通信，以实现更优的性能。建议将可调度的AI CPU核数设置为5。

    ```
    static ge::graphStatus BatchWriteTilingFunc(gert::TilingContext* context)
    {
        // 省略无关代码
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const auto aicCoreNum = ascendcPlatform.GetCoreNumAic();
        auto coreNum = use_aiv ? aicCoreNum * 2 : aicCoreNum;
        context->SetAicpuBlockDim(5U);
        context->SetSimdNumBlocks(coreNum);
        context->SetTilingKey(1000);

        // 省略无关代码
        SdmaBatchWriteCustomTilingData* tiling = context->GetTilingData<SdmaBatchWriteCustomTilingData>();
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, HCCL_CMD_BATCH_WRITE, "BatchWrite=level0:fullmesh", 0);
        mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
        return ge::GRAPH_SUCCESS;
    }
    ```



-   <a name="batchwrite_multi_queue"></a>多个队列的点对点通信

    在Atlas A3系列产品上，假设要将一段数据分别拷贝到两个不同的Global Memory上，可以通过调用一次BatchWrite接口，实现批量点对点通信。

    ```
    struct BatchWriteItem {
        uint64_t type;
        uint32_t res1[5];
        uint32_t length;
        uint32_t srcAddrLow;
        uint32_t srcAddrHigh;
        uint32_t dstAddrLow;
        uint32_t dstAddrHigh;
        uint32_t res2[4];
    }; // 按接口的约定定义格式

    extern "C" __global__ __aicore__ void BatchWrite_custom(
        GM_ADDR inputGM, GM_ADDR outputGM1, GM_ADDR outputGM2, GM_ADDR workspace, GM_ADDR tilingGM)
    {
        REGISTER_TILING_DEFAULT(BatchWriteCustomTilingData); // BatchWriteCustomTilingData为对应算子头文件定义的结构体
        GET_TILING_DATA_WITH_STRUCT(BatchWriteCustomTilingData, tilingData, tilingGM);
        GM_ADDR contextGM = AscendC::GetHcclContext<0>();

        if constexpr (g_coreType == AscendC::AIV) {
            Hccl hccl;
            hccl.InitV2(contextGM, &tilingData);
            hccl.SetCcTilingV2(offsetof(BatchWriteCustomTilingData, mc2CcTiling));

            __gm__ BatchWriteItem* sendInfo = reinterpret_cast<__gm__ BatchWriteItem*>(inputGM);

            sendInfo->type = 0UL;
            sendInfo->length = 64U;
            sendInfo->srcAddrLow = static_cast<uint32_t>((uint64_t)(inputGM) & 0xFFFFFFFF);
            sendInfo->srcAddrHigh = static_cast<uint32_t>(((uint64_t)(inputGM) >> 32) & 0xFFFFFFFF);
            sendInfo->dstAddrLow = static_cast<uint32_t>((uint64_t)(outputGM1) & 0xFFFFFFFF);
            sendInfo->dstAddrHigh = static_cast<uint32_t>(((uint64_t)(outputGM1) >> 32) & 0xFFFFFFFF);

            // 可以组装多个通信任务，实现批量发送
            (sendInfo + 1)->type = 0UL;
            (sendInfo + 1)->length = 64U;
            (sendInfo + 1)->srcAddrLow = static_cast<uint32_t>((uint64_t)(inputGM) & 0xFFFFFFFF);
            (sendInfo + 1)->srcAddrHigh = static_cast<uint32_t>(((uint64_t)(inputGM) >> 32) & 0xFFFFFFFF);
            (sendInfo + 1)->dstAddrLow = static_cast<uint32_t>((uint64_t)(outputGM2) & 0xFFFFFFFF);
            (sendInfo + 1)->dstAddrHigh = static_cast<uint32_t>(((uint64_t)(outputGM2) >> 32) & 0xFFFFFFFF);

            // 确保cache中的数据已刷新到GM地址上
            GlobalTensor<int64_t> tempTensor;
            tempTensor.SetGlobalBuffer((__gm__ int64_t*)sendInfo);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tempTensor);

            // 分别将两次拷贝部署在队列0、队列1上
            auto handleId0 = hccl.BatchWrite<true>(sendInfo, 1U, 0U);
            auto handleId1 = hccl.BatchWrite<true>(sendInfo, 1U, 1U);

            // 在所有队列上阻塞BatchWrite通信任务，所有队列将等到通信任务全部完成后再继续往下执行，实现所有队列的同步
            const uint16_t queueNum = hccl.GetQueueNum();
            for (uint16_t i = 0U; i < queueNum; ++i) {
                hccl.QueueBarrier<ScopeType::ALL>(i);
            }

            // Finalize可以无需等待服务端的通信任务全部完成即可退出，尽早释放AIV核心资源
            hccl.Finalize<false>();
            AscendC::SyncAll();
        }
    }
    ```

    当通信数据量较大时，可以在Tiling流程中调用SetAicpuBlockDim、[SetCommBlockNum](../hccl_tiling/set_comm_block_num.md)、[SetQueueNum](../hccl_tiling/set_queue_num.md)接口，通过并发机制提升算子的性能。

    在如下示例代码中，参与BatchWrite通信的核数为24，通信队列的数量为2，总的队列数=24\*2，即48；与此同时，服务端AI CPU的核数为4，这样每个AI CPU核只需要负责编排48/4即12个通信队列上的任务即可，提升了通信效率。

    ```
    static ge::graphStatus BatchWriteTilingFunc(gert::TilingContext* context)
    {
        // 省略无关代码
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const auto aicCoreNum = ascendcPlatform.GetCoreNumAic();
        auto coreNum = use_aiv ? aicCoreNum * 2 : aicCoreNum;
        context->SetAicpuBlockDim(4U);
        context->SetSimdNumBlocks(coreNum);
        context->SetTilingKey(1000);

        // 省略无关代码
        SdmaBatchWriteCustomTilingData* tiling = context->GetTilingData<SdmaBatchWriteCustomTilingData>();
        AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupName, HCCL_CMD_BATCH_WRITE, "BatchWrite=level0:fullmesh", 0);
        mc2CcTilingConfig.SetCommBlockNum(24U);
        mc2CcTilingConfig.SetQueueNum(2U);
        mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
        return ge::GRAPH_SUCCESS;
    }
    ```

- [MoeDistributeCombine A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_combine.md)
- [MoeDistributeDispatch A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/moe_distribute_dispatch.md)
