---
type: API
title: Utils API简介
description: OpTilingRegistry类属于context_ascendc命名空间，主要用于加载Tiling实现的动态库，并获取算子的Tiling函数指针以进行调试和验证。该类通常与ContextBuilder类配合使用，后者用于构造Tiling函数所需的输入参数。具体用法可参考如何进行Tiling调测。
tags:
- task.lookup
- topic.implementation.api_layer.utils_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/Utils-API/Tiling_tuning/OpTilingRegistry/intro.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms: []
---

# 简介<a name="ZH-CN_TOPIC_0000002333578885"></a>

OpTilingRegistry类属于context\_ascendc命名空间，主要用于加载Tiling实现的动态库，并获取算子的Tiling函数指针以进行调试和验证。该类通常与[ContextBuilder](../context_builder/index.md)类配合使用，后者用于构造Tiling函数所需的输入参数。具体用法可参考[如何进行Tiling调测](../../../../guides/programming_guide/appendix/common_operations/tiling_debug.md)。

## 所需的头文件和库文件<a name="section78885814919"></a>

-   需要包含的头文件

    ```
    #include "tiling/context/context_builder.h"
    ```

-   需要链接的库文件：libtiling\_api.a。

## Public成员函数<a name="section1173524710"></a>

```
[OpTilingRegistry](constructor_and_destructor.md)()
[~OpTilingRegistry](constructor_and_destructor.md)()
TilingFunc [GetTilingFunc](GetTilingFunc.md)(const char *opType) const
bool [LoadTilingLibrary](LoadTilingLibrary.md)(const char *tilingSoPath) const
```

