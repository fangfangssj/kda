---
type: API
title: is_array
description: 在程序编译时，检测一个类型是否为数组类型，可以用于在编译时进行类型检查和条件处理。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/type_traits/is_array.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
---

# is\_array

## 功能说明

在程序编译时，检测一个类型是否为数组类型，可以用于在编译时进行类型检查和条件处理。

## 函数原型

```
template <typename Tp>
struct is_array;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tp | 需要检测的类型，包括基本类型（如int、float等）、复合类型（如指针、引用）、用户自定义类型（如类、结构体等），以及数组类型本身。 |

## 约束说明

无

## 返回值说明

is\_array的静态常量成员value用于获取返回的布尔值，is\_array<Tp\>::value取值如下：

-   true：Tp是数组类型。
-   false：Tp不是数组类型。

## 调用示例

```
// 自定义测试类型
struct MyStruct{};
using FuncType = int(int);

// Legitimate array types
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<int[5]>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<char[]>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<double[2][3]>::value);

// Array types limited by CV
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<const int[10]>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<volatile char[3]>::value);

// non-array types
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<int*>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<int>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<double>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<MyStruct>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<FuncType>::value);
AscendC::printf("AscendC::Std::is_array::value:%d\n", AscendC::Std::is_array<void>::value);
```

```
// 执行结果：
AscendC::Std::is_array::value:1
AscendC::Std::is_array::value:1
AscendC::Std::is_array::value:1
AscendC::Std::is_array::value:1
AscendC::Std::is_array::value:1
AscendC::Std::is_array::value:0
AscendC::Std::is_array::value:0
AscendC::Std::is_array::value:0
AscendC::Std::is_array::value:0
AscendC::Std::is_array::value:0
AscendC::Std::is_array::value:0
```
