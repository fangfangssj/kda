---
type: API
title: SetFlag/WaitFlag
description: 核内多流水间同步指令，SetFlag在源流水前序指令完成后设置硬件标志位，WaitFlag在目的流水等待标志位并清除，标注为ISASI类别，不保证跨硬件版本兼容。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.synchronization
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
- kirin_x90
- kirin_9030
source_headers:
- include/basic_api/kernel_operator_block_sync_intf.h
---

# SetFlag/WaitFlag\(ISASI\)<a name="ZH-CN_TOPIC_0000001837467889"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_block_sync_intf.h"`。

如图1所示，SetFlag/WaitFlag接口用于核内多流水间的同步：

> [!NOTE]说明
>该接口标注为ISASI类别，不能保证跨硬件版本兼容。如需保证跨硬件版本兼容的同步控制接口，请参见[TQueSync类](t_que_sync.md)。二者的区别在于，TQueSync类接口可以保证跨硬件版本兼容。

- SetFlag：当源流水的前序指令的所有读写操作都完成之后，当前指令开始执行，并将硬件中的对应标志位设置为1。SetFlag只是设置硬件中的对应标志位，并不会阻塞源流水中的下一个指令。
- WaitFlag：当目的流水执行到该指令时，如果发现硬件中对应标志位为0，目的流水的后续指令将一直被阻塞；如果发现硬件中对应标志位为1，则将硬件中对应标志位设置为0，同时目的流水的后续指令开始执行。

**图1**  SetFlag/WaitFlag接口功能示意图<a name="zh-cn_topic_0000002511125384_fig58242463299"></a>  
![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/setflag_waitflag_multi_pipeline_sync.png "SetFlag_WaitFlag_多流水同步示意图")

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| event | 输入 | 模板参数。同步事件，数据类型为HardEvent。同一核内的不同流水之间，在存在数据访问依赖时，需要根据数据访问的先后顺序，插入对应的同步事件。HardEvent用来表示对应的同步事件。<br>HardEvent命名规则为<源流水\_目标流水\>，其中源流水的指令先执行、目标流水中的指令后执行。例如MTE2\_V，代表PIPE\_MTE2为源流水，PIPE\_V为目标流水，标识从PIPE\_MTE2到PIPE\_V的同步，PIPE\_V等待PIPE\_MTE2。<br>流水类型的取值及含义请参考[硬件流水类型](intra_core_sync_overview.md#硬件流水类型)。由于硬件架构版本代际间的差异，不同硬件架构上的事件存在差异。[NPU架构2201](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md#npu-arch)下HardEvent的取值范围请参考[核内同步组合](intra_core_sync_overview.md#npu_arch_2201_intra_core_sync_combinations)。 |
| eventID | 输入 | 事件ID。数据类型为int32_t类型。eventID的取值范围与产品型号有关，具体请参考[约束说明](#section633mcpsimp)。 |

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

- SetFlag只是设置硬件标志位，不会阻塞源流水中的下一个指令。

- SetFlag和WaitFlag必须成对使用，且SetFlag和WaitFlag的参数必须完全一致（包括模板参数event和输入参数eventID）。如果不匹配，会引发timeout问题。例如，`SetFlag<HardEvent::S_MTE3>(1)`和`WaitFlag<HardEvent::MTE3_MTE1>(1)`并不匹配，因为其模板参数event不同。

- 在采用[TPipe-TQue框架编程范式](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/tpipe_tque_programming/tpipe_tque_paradigm.md)时，eventID需要通过[AllocEventID](../../resource_management/t_pipe/alloc_event_id.md)或者[FetchEventID](../../resource_management/t_pipe/fetch_event_id.md)来获取。

- 在采用[静态Tensor编程范式](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)时，事件的类型和事件ID由开发者自行管理，建议使用事件ID0-5，事件ID6用于系统内部规划（当前未使用），事件ID7用于TPipe编程中的**自动同步**功能，目前暂不建议直接使用事件ID6-7。

- eventID的取值范围如下：

    
    - Ascend 950PR&950DT系列产品，数据范围为：0-7。
    
    
    - Atlas A3系列产品，数据范围为：0-7。
    
    
    - Atlas A2系列产品，数据范围为：0-7。
    
    
    - Atlas推理系列产品AI Core，数据范围为：0-7。
    
    
    - Atlas训练系列产品，数据范围为：0-3。
    

- 相同流水、相同eventID下，连续使用SetFlag会引发未定义行为，此时再执行PipeBarrier<PIPE\_ALL\>会出现卡死现象：

    ```cpp
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);  // 第一次调用SetFlag。
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);  // 第二次调用SetFlag（相同流水、相同eventID连续使用，引发未定义行为）。
    ...
    AscendC::PipeBarrier<PIPE_ALL>();  // 触发卡死。
    ...
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    ```

## 调用示例<a name="section837496171220"></a>

```cpp
AscendC::DataCopy(src1Local, src1Global[i * tileLength], tileLength);
AscendC::DataCopy(src0Local, src0Global[i * tileLength], tileLength);

// 循环内依赖：先“DataCopy(PIPE_MTE2)写src0Local”，后“Maxs和Mins（PIPE_V）读src0Local”。
// 由于PIPE_V需要等待PIPE_MTE2，所以需要插入以下同步。
AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

AscendC::Maxs(tmpTensor1, src0Local, inputVal, tileLength);
AscendC::Mins(tmpTensor2, src0Local, inputVal, tileLength);
```

## 配套 API 与常见组合

- [TQueSync](t_que_sync.md)
- [CrossCoreSetFlag](../inter_core_sync/cross_core_set_flag_isasi.md)
- [CrossCoreWaitFlag](../inter_core_sync/cross_core_wait_flag_isasi.md)
- [TQueBind 复用中的内存收益与中间计算](../../../../../guides/memory/tquebind_reuse_and_intermediate_compute.md)
- [GetTPipePtr](../../resource_management/get_t_pipe_ptr.md)——SetFlag 消费的 event_t 由 kernel 内标准惯用法 GetTPipePtr()->FetchEventID(HardEvent::V_S) 取得，证据为同一代码序列
- [FetchEventID](../../resource_management/t_pipe/fetch_event_id.md)——FetchEventID 返回的 TEventID 直接作为 SetFlag/WaitFlag 入参，卡片示例即此链
- [InitSocState](../../tool_interface/system_init/init_soc_state.md)——自管 eventID 的 SetFlag/WaitFlag 手动同步属静态Tensor编程（SetFlag 卡约束明文）
