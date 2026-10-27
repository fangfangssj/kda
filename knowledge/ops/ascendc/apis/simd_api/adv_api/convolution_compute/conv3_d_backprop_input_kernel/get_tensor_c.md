---
type: API
title: GetTensorC
description: 在完成Iterate操作后调用本接口，获取结果矩阵块，完成数据从L0C到GM的搬运。此接口与Iterate接口配合使用，用于在Iterate执行迭代计算后，获取结果矩阵。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.neighborhood.convolution
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/convolution_compute/Conv3DBackpropInput_Kernel/GetTensorC.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# GetTensorC

## 功能说明

在完成Iterate操作后调用本接口，获取结果矩阵块，完成数据从L0C到GM的搬运。此接口与[Iterate](iterate.md)接口配合使用，用于在Iterate执行迭代计算后，获取结果矩阵。

## 函数原型

```
template <bool sync = true>
__aicore__ inline void GetTensorC(const AscendC::GlobalTensor<DstT>& output, uint8_t enAtomic = 0, bool enSequentialWrite = false)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sync | 预留参数，用户无需感知。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| output | 输入 | 将计算结果搬至Global Memory的GM地址。 |
| enAtomic | 输入 | 预留参数，用户无需感知。 |
| enSequentialWrite | 输入 | 预留参数，用户无需感知。 |

## 返回值说明

无

## 约束说明

GetTensorC接口必须在Iterate后进行调用，完成卷积反向实现，调用顺序如下。

```
while (Iterate()) {
    GetTensorC();
}
```

## 调用示例

```
while (gradInput_.Iterate()) {
    gradInput_.GetTensorC(gradInputGm_[offsetC_]);
}
```
