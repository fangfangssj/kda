---
type: API
title: SetAtomicMin
description: 对后续搬运到GM的数据开启原子取小，将拷贝内容与GM已有值比较后写入最小值。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.atomic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/atomic_operations/SetAtomicMin_ISASI.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_set_atomic_intf.h
---

# SetAtomicMin\(ISASI\)<a name="ZH-CN_TOPIC_0000001637481306"></a>

## 功能说明<a name="section618mcpsimp"></a>

头文件路径为：`"basic_api/kernel_operator_set_atomic_intf.h"`。

设置后续搬运到GM的数据是否执行原子比较：将待拷贝的内容和GM已有内容进行比较，然后将最小值写入GM。SetAtomicMin接口可通过设置模板参数来设定不同的数据类型。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
template <typename T>
__aicore__ inline void SetAtomicMin() 
```

## 参数说明<a name="section622mcpsimp"></a>

**表1**  模板参数说明

| 参数名 | 描述 |
|--------|------|
| T | 设定不同的原子最小操作数据类型。 |

## 数据类型

支持的数据类型为int8_t、int16_t、half、bfloat16_t、int32_t、float。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>


- Ascend 950PR&950DT系列产品，支持的数据通路为UB/L0C Buffer->GM。



- Atlas A3系列产品，支持的数据通路为UB/L0C Buffer/L1 Buffer->GM。



- Atlas A2系列产品，支持的数据通路为UB/L0C Buffer/L1 Buffer->GM。


- 使用完后，建议通过[DisableDmaAtomic](disable_dma_atomic.md)关闭原子最小操作，以免影响后续相关功能。
- 该接口执行前不会自动清零GM数据。开发者需根据算子逻辑判断是否清零，如需清零，请在执行前手动完成。
- SetAtomicMin内部已集成与[SetAtomicType](set_atomic_type.md)相同的功能。建议调用上述接口时，通过设置模板参数显式指定原子操作的数据类型，无需额外调用SetAtomicType接口。

- 针对[NPU架构版本2201](../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)，整数类型和bfloat16_t数据类型**默认均为非饱和模式**，原子最小操作时的饱和模式无法通过[SetSaturationStrategy](../special_register_access/set_saturation_strategy.md)或[SetCtrlSpr](../special_register_access/set_ctrl_spr_isasi.md)设置，需要通过`npu-smi`命令进行设备级设置，具体设置方法请参考[SetAtomicAdd约束说明](set_atomic_add.md#npu_smi_saturation_mode)。


## 调用示例<a name="section177231425115410"></a>

```cpp
#include "kernel_operator.h"

constexpr uint32_t SIZE = 256;
__aicore__ inline void CopyIn()
{
    AscendC::LocalTensor<T> srcLocal = queueSrc.AllocTensor<T>();
    // 清空原子操作的状态
    AscendC::DisableDmaAtomic();
    AscendC::DataCopy(srcLocal, srcGlobal, SIZE);
    queueSrc.EnQue(srcLocal);
    // 核间同步
    AscendC::SyncAll();
}

__aicore__ inline void CopyOut()
{
    AscendC::LocalTensor<T> srcLocal = queueSrc.DeQue<T>();
    // 开启原子MIN模式，3个核取最小值
    AscendC::SetAtomicMin<T>();
    AscendC::DataCopy(dstGlobal, srcLocal, SIZE);
    // 关闭原子操作
    AscendC::DisableDmaAtomic();
    queueSrc.FreeTensor(srcLocal);
}

/*
每个核的输入数据为: 
Src0: [1,1,1,1,1,...,1] // 256个1
Src1: [2,2,2,2,2,...,2] // 256个2
最终输出数据: [1,1,1,1,1,...,1] // 256个1
*/
```

完整样例请参考[DataMovementWithAtomicOperations样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/03_basic_api/06_atomic/data_movement_with_atomic_operations)。
