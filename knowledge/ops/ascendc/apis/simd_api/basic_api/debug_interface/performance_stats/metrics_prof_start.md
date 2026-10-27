---
type: API
title: MetricsProfStart
description: 设置性能数据采集信号启动，与MetricsProfStop配合使用，用于指定msOpProf调优的kernel侧代码段范围。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.debugging
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/debug_interface/performance_stats/MetricsProfStart.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
source_headers:
- include/basic_api/kernel_prof_trace_intf.h
---

# MetricsProfStart<a name="ZH-CN_TOPIC_0000002080803573"></a>

## 功能说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section259105813316"></a>

头文件路径为：`"basic_api/kernel_prof_trace_intf.h"`。

推荐使用[asc_prof_start](../../../../utils_api/tuning_interface/asc_prof_start.md)接口进行性能数据采集信号启动的设置，该接口同时适用于C语言和C++语言编程。

用于设置性能数据采集信号启动，和[MetricsProfStop](metrics_prof_stop.md)配合使用。使用msOpProf工具进行算子上板调优时，可在核函数（Kernel）侧代码段前后分别调用MetricsProfStart和MetricsProfStop来指定需要调优的代码段范围。

## 函数原型<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section2067518173415"></a>

```cpp
__aicore__ inline void MetricsProfStart()
```

## 参数说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section158061867342"></a>

无

## 返回值说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section43265506459"></a>

无

## 调用示例<a name="zh-cn_topic_0000001963799138_zh-cn_topic_0000001997078721_section82241477610"></a>

```cpp
MetricsProfStart();
... // 需要调优的核函数（Kernel）侧代码段。
MetricsProfStop();
```
