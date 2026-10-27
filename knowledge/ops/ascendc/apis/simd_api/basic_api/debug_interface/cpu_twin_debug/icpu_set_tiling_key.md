---
type: API
title: IcpuSetTilingKey
description: 指定本次CPU调测使用的tilingKey，调测时只执行核函数中该tilingKey对应的分支。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.implementation.debugging
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/debug_interface/cpu_twin_debug/ICPU_SET_TILING_KEY.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# ICPU\_SET\_TILING\_KEY<a name="ZH-CN_TOPIC_0000002044842850"></a>

## 功能说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section259105813316"></a>

头文件路径为：`"tools/cpudebug/include/kern_fwk.h"`。

用于指定本次CPU调测使用的tilingKey。调测执行时，将只执行算子核函数（Kernel）中该tilingKey对应的分支。

## 函数原型<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section2067518173415"></a>

```cpp
#define ICPU_SET_TILING_KEY(tilingKey)
```

## 参数说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section158061867342"></a>

<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_zh-cn_topic_0235751031_table33761356"></a>
**表1**  模板参数说明

| 参数名称 | 输入/输出 | 描述 |
| ------ | ------ | ------ |
| tilingKey | 输入 | 指定本次CPU调测使用的tilingKey。参数类型为uint64_t。|

## 返回值说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section794123819592"></a>

- 为了保留接口兼容，推荐[<<<>>>](../../../../../guides/programming_guide/debug_and_tuning/functional_debug/cpu_twin_debug.md)编译使用。
- tilingKey建议传入正整数，如果设置为负数或者0，将会告警并继续调测。如果传入0，将会执行tilingKey为0的分支；tilingKey传入负数，将导致未定义的行为。
- 该接口需要在[ICPU_RUN_KF](icpu_run_kf.md)前调用。

## 调用示例<a name="zh-cn_topic_0000002000279997_zh-cn_topic_0000001610028277_section82241477610"></a>

```cpp
ICPU_SET_TILING_KEY(10086);
ICPU_RUN_KF(sort_kernel0, coreNum, (uint8_t*)x, (uint8_t*)y);
```
