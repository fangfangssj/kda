---
type: API
title: asc_prof_stop
description: SIMD场景下，设置性能数据采集信号停止，和asc_prof_start配合使用。使用msOpProf工具进行算子上板调优时，可在kernel侧代码段前后分别调用asc_prof_start和asc_prof_stop来指定需要调优的代码段范围。
tags:
- task.lookup
- topic.programming_model
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/tuning_interface/asc_prof_stop.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/utils/debug/asc_time.h
---

# asc\_prof\_stop

## 功能说明

SIMD场景下，设置性能数据采集信号停止，和asc\_prof\_start配合使用。使用msOpProf工具进行算子上板调优时，可在核函数（Kernel）侧代码段前后分别调用asc\_prof\_start和asc\_prof\_stop来指定需要调优的代码段范围。

## 函数原型

```
__aicore__ inline void asc_prof_stop()
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

```
asc_prof_stop();
```
