---
type: API
title: add_rvalue_reference
description: 在程序编译时，为指定类型添加右值引用限定符，可以用于在编译时进行类型转换。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/type_traits/add_rvalue_reference.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
---

# add\_rvalue\_reference

## 功能说明

在程序编译时，为指定类型添加右值引用限定符，可以用于在编译时进行类型转换。

## 函数原型

```
template <typename Tp>
struct add_rvalue_reference;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tp | 需要处理的类型，包括基本类型（如int、float等）、复合类型（如数组、指针）、用户自定义类型（如类、结构体等），以及带有引用限定符的类型。 |

## 约束说明

无

## 返回值说明

add\_rvalue\_reference是一个结构体，其提供一个嵌套类型type，表示添加右值引用限定符后的类型。通过add\_rvalue\_reference<Tp\>::type来访问该类型。

## 调用示例

```
// Test basic type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<int>, int&&>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<float>, float&&>));

// Test pointer type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<int*>, int*&&>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<const int*>, const int*&&>));

// Test reference type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<int&>, int&>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<int&&>, int&&>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<const int&>, const int&>));

// Test void type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<void>, void>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<const void>, const void>));

// Test function type
using FuncType = void();
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<FuncType>, FuncType&&>));

// Test array type
using ArrayType = int[];
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<ArrayType>, ArrayType&&>));

// Test class type
class MyClass {};
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<MyClass>, MyClass&&>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_rvalue_reference_t<const MyClass>, const MyClass&&>));
```
