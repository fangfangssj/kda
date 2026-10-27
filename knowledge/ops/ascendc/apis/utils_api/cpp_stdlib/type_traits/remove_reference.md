---
type: API
title: remove_reference
description: 在程序编译时，从给定类型中移除引用限定符，包括左值引用T&和右值引用T&&，可以用于在编译时进行类型转换。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/type_traits/remove_reference.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
---

# remove\_reference

## 功能说明

在程序编译时，从给定类型中移除引用限定符，包括左值引用T&和右值引用T&&，可以用于在编译时进行类型转换。

## 函数原型

```
template <typename Tp>
struct remove_reference;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tp | 需要处理的类型，包括基本类型（如int、float等）、复合类型（如数组、指针）、用户自定义类型（如类、结构体等），以及带有左值引用&或右值引用&&限定的类型。 |

## 约束说明

无

## 返回值说明

remove\_reference是一个结构体，其提供一个嵌套类型type，表示移除引用限定符后的类型。通过remove\_reference<Tp\>::type来访问该类型。

## 调用示例

```
// Test non-reference type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<int>::type, int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<double>::type, double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<char>::type, char>));
// Test lvalue reference type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<int&>::type, int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<double&>::type, double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<char&>::type, char>));
// Test rvalue reference type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<int&&>::type, int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<double&&>::type, double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference<char&&>::type, char>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference_t<int>, int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference_t<double>, double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference_t<int&>, int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference_t<double&>, double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference_t<int&&>, int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::remove_reference_t<double&&>, double>));
```
