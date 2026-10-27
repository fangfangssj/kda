---
type: API
title: is_reference
description: 在程序编译时，检测一个类型是否为引用类型，可以用于在编译时进行类型检查和条件处理。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/type_traits/is_reference.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
---

# is\_reference

## 功能说明

在程序编译时，检测一个类型是否为引用类型，可以用于在编译时进行类型检查和条件处理。

## 函数原型

```
template <typename Tp>
struct is_reference;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tp | 需要检测的类型，包括基本类型（如int、float等）、复合类型（如数组、指针）、用户自定义类型（如类、结构体等），以及引用类型本身。 |

## 约束说明

无

## 返回值说明

is\_reference的静态常量成员value用于获取返回的布尔值，is\_reference<Tp\>::value取值如下：

-   true：Tp是引用类型。
-   false：Tp不是引用类型。

## 调用示例

```
// 自定义测试类型
struct MyStruct{int val;};
// 函数类型
using FuncType = void(int);

// Legitimate reference type
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<int&>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<int&&>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<int(&)[5]>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<int*&>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<MyStruct&>::value);

// CV restricted reference types
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<const int&>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<volatile double&&>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<const MyStruct&>::value);

// non-reference type
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<int>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<int*>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<int[5]>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<double>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<MyStruct>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<FuncType>::value);
AscendC::printf("AscendC::Std::is_reference::value:%d\n", AscendC::Std::is_reference<void>::value);
```

```
// 执行结果：
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:1
AscendC::Std::is_reference::value:0
AscendC::Std::is_reference::value:0
AscendC::Std::is_reference::value:0
AscendC::Std::is_reference::value:0
AscendC::Std::is_reference::value:0
AscendC::Std::is_reference::value:0
AscendC::Std::is_reference::value:0
```
