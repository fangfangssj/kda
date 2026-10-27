---
type: API
title: TQueSync
description: 头文件路径为："basic_api/kernel_operator_block_sync_intf.h"。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/TQueSync.md
  role: primary
created_at: '2026-09-12T03:09:26Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_operator_block_sync_intf.h
---

# TQueSync

## 功能说明

头文件路径为：`"basic_api/kernel_operator_block_sync_intf.h"`。

TQueSync类用于核内多流水间的同步。TQueSync类的模板参数用于指定源流水和目标流水，目的流水等待源流水，从而解决多流水间的数据访问依赖。TQueSync类的成员函数SetFlag/WaitFlag接口用于完成具体的同步操作，同一核内不同流水线之间的同步指令，具有数据依赖的不同流水指令之间需要插入此同步。其功能说明如下：

- SetFlag：当前序指令的所有读写操作都完成之后，当前指令开始执行，并将硬件中的对应标志位设置为1。
- WaitFlag：当执行到该指令时，如果发现对应标志位为0，该队列的后续指令将一直被阻塞；如果发现对应标志位为1，则将对应标志位设置为0，同时后续指令开始执行。

> [!NOTE]说明
> 此接口与[SetFlag/WaitFlag(ISASI)](set_flag_wait_flag_isasi.md)接口也能实现相同的功能。二者的区别在于，TQueSync类接口可以保证跨硬件版本兼容，而[SetFlag/WaitFlag(ISASI)](set_flag_wait_flag_isasi.md)中的接口标注为ISASI类别，不能保证跨硬件版本兼容。

## 函数原型

```cpp
using TEventID = int8_t;

template<pipe_t src, pipe_t dst>
class TQueSync {
public:
    __aicore__ inline void SetFlag(TEventID id)
    __aicore__ inline void WaitFlag(TEventID id)
};
```

## 参数说明

**表1**  类模板参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src | 输入 | 源流水。支持的流水参考[硬件流水类型](intra_core_sync_overview.md#硬件流水类型)。 |
| dst | 输入 | 目的流水。支持的流水参考[硬件流水类型](intra_core_sync_overview.md#硬件流水类型)。 |

**表2**  成员函数参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| id | 输入 | 事件ID，数据类型为TEventID。推荐通过[AllocEventID](../../resource_management/t_pipe/alloc_event_id.md)或者[FetchEventID](../../resource_management/t_pipe/fetch_event_id.md)来获取。<br>不同产品对id的取值范围说明请参见[id取值范围说明](#id取值范围说明)。 |

### id取值范围说明


- Ascend 950PR&950DT系列产品，取值范围是0-7。


- Atlas A3系列产品，取值范围是0-7。


- Atlas A2系列产品，取值范围是0-7。


- Atlas 200I/500 A2推理产品，取值范围是0-7。


- Atlas推理系列产品AI Core，取值范围是0-7。


- Atlas训练系列产品，取值范围是0-3。


## 返回值说明

无

## 约束说明

- `src`和`dst`必须为支持的硬件流水，且二者不能相同。

- SetFlag只是设置硬件标志位，不会阻塞源流水中的下一个指令。

- SetFlag和WaitFlag必须成对使用，且模板参数`src`、`dst`和输入参数`id`必须完全一致。如果不匹配，会引发timeout问题。

- 在采用[TPipe-TQue框架编程范式](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/tpipe_tque_programming/tpipe_tque_paradigm.md)时，`id`需要通过[AllocEventID](../../resource_management/t_pipe/alloc_event_id.md)或者[FetchEventID](../../resource_management/t_pipe/fetch_event_id.md)来获取。

- 在采用[静态Tensor编程范式](../../../../../guides/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/static_tensor_programming.md)时，事件类型和`id`由开发者自行管理，建议使用事件ID0-5，事件ID6用于系统内部规划（当前未使用），事件ID7用于TPipe编程中的**自动同步**功能，目前暂不建议直接使用事件ID6-7。

- 相同源流水、目的流水和`id`下，连续使用SetFlag会引发未定义行为，此时再执行PipeBarrier<PIPE\_ALL>会出现卡死现象：

    ```cpp
    AscendC::TQueSync<PIPE_M, PIPE_MTE1> sync;
    sync.SetFlag(0);  // 第一次调用SetFlag。
    sync.SetFlag(0);  // 第二次调用SetFlag（相同源流水、目的流水和id连续使用，引发未定义行为）。
    ...
    AscendC::PipeBarrier<PIPE_ALL>();  // 触发卡死。
    ...
    sync.WaitFlag(0);
    sync.WaitFlag(0);
    ```

## 调用示例

如DataCopy需要等待SetValue执行完成后才能执行，需要插入PIPE\_S到PIPE\_MTE3的同步。

```cpp
AscendC::GlobalTensor<half> dstGlobal;
AscendC::LocalTensor<half> dstLocal;
dstLocal.SetValue(0, 0);
AscendC::TQueSync<PIPE_S, PIPE_MTE3> sync;
sync.SetFlag(0);
sync.WaitFlag(0);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```

## 配套 API 与常见组合

- [SetFlag/WaitFlag](set_flag_wait_flag_isasi.md)
- [EnQue](../../resource_management/t_que/en_que.md)
- [DeQue](../../resource_management/t_que/de_que.md)
- [TQueBind 复用中的内存收益与中间计算](../../../../../guides/memory/tquebind_reuse_and_intermediate_compute.md)
