---
type: API
title: SetCmpMask
description: 设置CmpMask比较寄存器的值，配合不传入mask参数的Select接口使用。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.compute.elementwise.comparison_logic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/compare_and_select/SetCmpMask_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
source_headers:
- include/basic_api/kernel_operator_vec_cmpsel_intf.h
---

# SetCmpMask\(ISASI\)<a name="ZH-CN_TOPIC_0000001836926169"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_vec_cmpsel_intf.h"`。

设置比较寄存器的值，配合不传入mask参数的[Select](select.md)接口使用，根据不同的selMode传入不同的数据。

- 模式0（SELMODE::VSEL\_CMPMASK\_SPR）

    SetCmpMask中传入selMask LocalTensor。

- 模式1（SELMODE::VSEL\_TENSOR\_SCALAR\_MODE）

    SetCmpMask中传入src1 LocalTensor。

- 模式2（SELMODE::VSEL\_TENSOR\_TENSOR\_MODE）

    SetCmpMask中传入LocalTensor，LocalTensor中存放的是selMask的地址。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void SetCmpMask(const LocalTensor<T>& src)
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。 |

**表2**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src | 输入 | 类型为[LocalTensor](../../data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br>LocalTensor的起始地址需要16字节对齐。 |

## 数据类型

支持数据类型为：b8、b16、b32、b64。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

- 当selMode为模式0或模式2时：

    ```cpp
    uint32_t dataSize = 256;
    uint32_t selDataSize = 8;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueY;
    TQue<TPosition::VECIN, 1> inQueueSel;
    TQue<TPosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueSel, 1, selDataSize * sizeof(uint8_t));
    pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
    AscendC::LocalTensor<float> dst = outQueue.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sel = inQueueSel.AllocTensor<uint8_t>();
    AscendC::LocalTensor<float> src0 = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> src1 = inQueueY.AllocTensor<float>();
    uint8_t repeat = 4;
    uint32_t mask = 64;
    AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };

    // selMode为模式0（SELMODE::VSEL_CMPMASK_SPR）
    AscendC::SetCmpMask(sel);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(mask);
    AscendC::Select<float, AscendC::SELMODE::VSEL_CMPMASK_SPR>(dst, src0, src1, repeat, repeatParams);

    // selMode为模式2（SELMODE::VSEL_TENSOR_TENSOR_MODE）
    AscendC::LocalTensor<int32_t> tempBuf;
    #if defined(ASCENDC_CPU_DEBUG) && (ASCENDC_CPU_DEBUG == 1)  // cpu调试
    tempBuf.ReinterpretCast<int64_t>().SetValue(0, reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(sel.GetPhyAddr())));
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
    AscendC::SetFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventIdSToV);
    #else // npu调试
    uint32_t selAddr = static_cast<uint32_t>(reinterpret_cast<int64_t>(reinterpret_cast<__ubuf__ int64_t*>(sel.GetPhyAddr())));
    AscendC::SetVectorMask<uint32_t>(32);
    AscendC::Duplicate<uint32_t, false>(tempBuf.ReinterpretCast<uint32_t>(), selAddr, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    #endif
    AscendC::SetCmpMask<int64_t>(tempBuf.ReinterpretCast<int64_t>());
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(mask);
    AscendC::Select<float, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE>(dst, src0, src1, repeat, repeatParams);
    ```

- 当selMode为模式1时：

    ```cpp
    uint32_t dataSize = 256;
    uint32_t selDataSize = 8;
    TPipe pipe;
    TQue<TPosition::VECIN, 1> inQueueX;
    TQue<TPosition::VECIN, 1> inQueueY;
    TQue<TPosition::VECIN, 1> inQueueSel;
    TQue<TPosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueY, 1, dataSize * sizeof(float));
    pipe.InitBuffer(inQueueSel, 1, selDataSize * sizeof(uint8_t));
    pipe.InitBuffer(outQueue, 1, dataSize * sizeof(float));
    AscendC::LocalTensor<float> dst = outQueue.AllocTensor<float>();
    AscendC::LocalTensor<uint8_t> sel = inQueueSel.AllocTensor<uint8_t>();
    AscendC::LocalTensor<float> src0 = inQueueX.AllocTensor<float>();
    AscendC::LocalTensor<float> tmpScalar = inQueueY.AllocTensor<float>();

    uint8_t repeat = 4;
    uint32_t mask = 64;
    AscendC::BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };

    // selMode为模式1（SELMODE::VSEL_TENSOR_SCALAR_MODE）
    AscendC::SetVectorMask<uint32_t>(32);
    AscendC::Duplicate<float, false>(tmpScalar, static_cast<float>(1.0), AscendC::MASK_PLACEHOLDER, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetCmpMask(tmpScalar);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(mask);
    AscendC::Select(dst, sel, src0, repeat, repeatParams);
    ```
