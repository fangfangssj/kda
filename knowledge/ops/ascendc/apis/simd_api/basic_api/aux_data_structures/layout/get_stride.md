---
type: API
title: GetStride
description: 返回描述内存访问步长的Stride对象，与Shape的维度信息一一对应。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/aux_data_structures/Layout/GetStride.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
---

# GetStride<a name="ZH-CN_TOPIC_0000002331573721"></a>

## 功能说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section36583473819"></a>

返回描述内存访问步长的Stride对象，与Shape的维度信息一一对应。

## 函数原型<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section13230182415108"></a>

```cpp
__aicore__ inline constexpr decltype(auto) GetStride() {}    
__aicore__ inline constexpr decltype(auto) GetStride() const {}
```

## 参数说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section25791320141317"></a>

描述内存访问步长的Stride对象，Stride结构类型（[Std::tuple](../../../../utils_api/cpp_stdlib/container_functions/index.md)类型的别名），定义如下：

```cpp
template <typename... Strides>
using Stride = Std::tuple<Strides...>;
```

## 约束说明<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000002042526794_zh-cn_topic_0000001602767550_zh-cn_topic_0000001600123102_section320753512363"></a>

```cpp
// 初始化Layout数据结构，获取对应数值
AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

AscendC::Layout<AscendC::Shape<int, int, int>, AscendC::Stride<int, int, int>> layoutInit(shape, stride);

int value = AscendC::Std::get<0>(layoutInit.GetStride()); // value = 1
value = AscendC::Std::get<1>(layoutInit.GetStride()); // value = 100
value = AscendC::Std::get<2>(layoutInit.GetStride()); // value = 200
```
