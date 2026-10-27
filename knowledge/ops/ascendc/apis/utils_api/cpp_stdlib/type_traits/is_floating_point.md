---
type: API
title: is_floating_point
description: 在程序编译时，检测一个类型是否为浮点类型，可以用于在编译时进行类型检查和条件处理。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/type_traits/is_floating_point.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
---

# is\_floating\_point

## 功能说明

在程序编译时，检测一个类型是否为浮点类型，可以用于在编译时进行类型检查和条件处理。

## 函数原型

```
template <typename T>
struct is_floating_point;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| T | 需要检测的类型，包括基本数据类型、修饰类型等。 |

## 约束说明

无

## 返回值说明

is\_floating\_point的静态常量成员value用于获取返回的布尔值，is\_floating\_point<Tp\>::value取值如下：

-   true：Tp是浮点类型。
-   false：Tp不是浮点类型。

## 调用示例

```
// 自定义测试类型
struct MyStruct{};

// 返回值为浮点的函数类型
using FuncType = double(int);

// Legal floating point type
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<float>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<double>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<long double>::value);

// Floating point types limited by CV
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<const float>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<volatile double>::value);

// non-floating point type
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<int>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<bool>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<double*>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<float&>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<double[5]>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<FuncType>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<MyStruct>::value);
AscendC::printf("AscendC::Std::is_floating_point::value:%d\n", AscendC::Std::is_floating_point<void>::value);
```

```
// 执行结果：
AscendC::Std::is_floating_point::value:1
AscendC::Std::is_floating_point::value:1
AscendC::Std::is_floating_point::value:1
AscendC::Std::is_floating_point::value:1
AscendC::Std::is_floating_point::value:1
AscendC::Std::is_floating_point::value:0
AscendC::Std::is_floating_point::value:0
AscendC::Std::is_floating_point::value:0
AscendC::Std::is_floating_point::value:0
AscendC::Std::is_floating_point::value:0
AscendC::Std::is_floating_point::value:0
AscendC::Std::is_floating_point::value:0
AscendC::Std::is_floating_point::value:0
```
