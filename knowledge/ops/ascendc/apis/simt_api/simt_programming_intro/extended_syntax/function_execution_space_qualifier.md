---
type: API
title: 函数执行空间限定符
description: 函数执行空间限定符（Function Execution Space Qualifier）指示函数是在Host侧执行还是在Device侧执行，以及能被调用的空间范围。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/SIMT_programming_intro/extended_syntax/function_execution_space_qualifier.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
---

# 函数执行空间限定符

函数执行空间限定符（Function Execution Space Qualifier）指示函数是在Host侧执行还是在Device侧执行，以及能被调用的空间范围。

**表1**  函数执行空间限定符

| 函数执行空间限定符 | 执行空间 | 说明 |
| --- | --- | --- |
| \_\_host\_\_ | host | 只能被Host侧函数调用，无函数执行空间限定符定义的函数，默认是host函数。 |
| \_\_global\_\_ | device | 修饰的函数是核函数（Kernel）入口，只能被Host侧函数调用，函数返回类型必须为void，不能是class、struct或者union的成员函数，不支持递归调用且对__global__函数的调用是异步的。 |
| \_\_aicore\_\_ | device | 修饰的函数只能在Device侧执行，只能被__global__函数，或者其他__aicore__函数调用。 |

函数调用关系如下图所示：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/function_call_hierarchy.png)
