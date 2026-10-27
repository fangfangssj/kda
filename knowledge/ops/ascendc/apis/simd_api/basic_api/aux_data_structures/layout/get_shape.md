---
type: API
title: GetShape
description: 返回描述张量形状的Shape对象。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/aux_data_structures/Layout/GetShape.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- a2
- a3
source_headers:
- include/basic_api/kernel_operator_list_tensor_intf.h
---

# GetShape<a name="ZH-CN_TOPIC_0000002297494516"></a>

## 功能说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section36583473819"></a>

返回描述张量形状的Shape对象。

## 函数原型<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section13230182415108"></a>

```cpp
__aicore__ inline constexpr decltype(auto) GetShape() {}   
__aicore__ inline constexpr decltype(auto) GetShape() const {}
```

## 参数说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section75395119104"></a>

无

## 返回值说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section25791320141317"></a>

描述张量形状的Shape对象，Shape结构类型（[Std::tuple](../../../../utils_api/cpp_stdlib/container_functions/index.md)类型的别名），定义如下：

```cpp
template <typename... Shapes>
using Shape = Std::tuple<Shapes...>;
```

## 约束说明<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000002078447573_zh-cn_topic_0000001576806829_zh-cn_topic_0000001339187720_section320753512363"></a>

```cpp
// 初始化Layout数据结构，获取对应数值
AscendC::Shape<int,int,int> shape = AscendC::MakeShape(10, 20, 30);
AscendC::Stride<int,int,int> stride = AscendC::MakeStride(1, 100, 200);

auto layoutMake = AscendC::MakeLayout(shape, stride);

int value = AscendC::Std::get<0>(layoutMake.GetShape()); // value = 10
value = AscendC::Std::get<1>(layoutMake.GetShape()); // value = 20
value = AscendC::Std::get<2>(layoutMake.GetShape()); // value = 30
```
