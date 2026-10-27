---
type: API
title: Utils API简介
description: ContextBuilder类提供一系列的API接口，支持手动构造TilingContext类来验证Tiling函数以及KernelContext类用于TilingParse函数的验证。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/Tiling_tuning/ContextBuilder/intro.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-04T01:25:11Z'
platforms: []
---

# 简介<a name="ZH-CN_TOPIC_0000002114097953"></a>

ContextBuilder类提供一系列的API接口，支持手动构造TilingContext类来验证Tiling函数以及KernelContext类用于TilingParse函数的验证。

## 调用示例<a name="zh-cn_topic_0000001819028992_zh-cn_topic_0000001389787297_section320753512363"></a>

```
// 构造KernelContext
auto kernelContextHolder = context_ascendc::ContextBuilder()
    .Inputs(...)
    .Outputs(...)
    .BuildKernelRunContext();
gert::KernelContext* tilingParseContext = kernelContextHolder->GetContext<gert::KernelContext>();

// 构造TilingContext
auto tilingContextHolder = context_ascendc::ContextBuilder()
    .SetOpNameType(...,...)
    .NodeIoNum(...)
    .IrInstanceNum(...)
    .AddInputTd(...)
    .AddOutputTd(...)
    .AddAttr(...)
    .BuildTilingContext(...);
gert::TilingContext* tilingContext = tilingContextHolder->GetContext<gert::TilingContext>();
```

