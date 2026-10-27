---
type: API
title: add_cv
description: 在程序编译时，为指定类型添加const和volatile限定符，可以用于在编译时进行类型转换。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/cpp_stdlib/type_traits/add_cv.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
---

# add\_cv

## 功能说明

在程序编译时，为指定类型添加const和volatile限定符，可以用于在编译时进行类型转换。

## 函数原型

```
template <typename Tp>
struct add_cv;
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 含义 |
| --- | --- |
| Tp | 需要处理的类型，包括基本类型（如int、float等）、复合类型（如数组、指针、引用）、用户自定义类型（如类、结构体等），以及带有const限定符、volatile限定符或这两个限定符的类型。 |

## 约束说明

无

## 返回值说明

add\_cv是一个结构体，其提供一个嵌套类型type，表示添加const和volatile限定符后的类型。通过add\_cv<Tp\>::type来访问该类型。

## 调用示例

```
// Test non-const and non-volatile type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<char>::type, const volatile char>));
// Test const type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const char>::type, const volatile char>));
// Test volatile type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<volatile int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<volatile double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<volatile char>::type, const volatile char>));
// Test const and volatile type
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const volatile int>::type, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const volatile double>::type, const volatile double>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv<const volatile char>::type, const volatile char>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<double>, const volatile double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const double>, const volatile double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<volatile int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<volatile double>, const volatile double>));

ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const volatile int>, const volatile int>));
ascendc_assert((AscendC::Std::is_same_v<AscendC::Std::add_cv_t<const volatile double>, const volatile double>));
```
