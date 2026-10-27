---
type: API
title: Trap
description: Kernel侧调试接口，NPU模式下中断AI Core运行，CPU模式下等同于assert，用于异常场景调试。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.debugging
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/debug_interface/exception_detection/Trap.md
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
- include/basic_api/kernel_operator_sys_var_intf.h
---

# Trap<a name="ZH-CN_TOPIC_0000001834672609"></a>

## 功能说明<a name="section259105813316"></a>

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

在核函数（Kernel）侧调用，NPU模式下会中断AI Core的运行，CPU模式下等同于assert。可用于核函数（Kernel）侧异常场景的调试。

## 函数原型<a name="section2067518173415"></a>

```cpp
__aicore__ inline void Trap()
```

## 参数说明<a name="section158061867342"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

无

## 调用示例<a name="section82241477610"></a>

```cpp
AscendC::LocalTensor<half> src0Local;
AscendC::LocalTensor<half> src1Local;
AscendC::LocalTensor<half> dstLocal;
constexpr int32_t count = 512;    // 参与计算的元素个数。
if (src1Local[0] == 0) {    // 如果src1Local[0]为0，则程序终止。
    AscendC::Trap();
} else {
    AscendC::Divs(dstLocal, src0Local, src1Local[0], count);
}
```

- [GroupQuant A2/A3 ACLNN 算子知识](../../../../../operators/aclnn/nn/quant/group_quant.md)
