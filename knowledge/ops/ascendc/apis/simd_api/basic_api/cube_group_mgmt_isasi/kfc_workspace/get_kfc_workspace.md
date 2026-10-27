---
type: API
title: GetKfcWorkspace
description: 获取用于CubeResGroupHandle消息通信区的内存地址。用户使用CubeResGroupHandle接口时，需要用此接口自主管理空间地址。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_group_mgmt_ISASI/KfcWorkspace/GetKfcWorkspace.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
source_headers:
- include/basic_api/core_mng/roc/kernel_operator_cube_group_intf.h
---

# GetKfcWorkspace<a name="ZH-CN_TOPIC_0000002005165737"></a>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

获取用于CubeResGroupHandle消息通信区的内存地址。用户使用[CubeResGroupHandle](../cube_res_group_handle/cube_res_group_handle_constructor.md)接口时，需要用此接口自主管理空间地址。

## 函数原型<a name="section765814724715"></a>

```
__aicore__ inline GM_ADDR GetKfcWorkspace()
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

无

## 返回值说明<a name="section358232015810"></a>

workspace地址。

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

本接口不能和[CreateCubeResGroup](../cube_res_group_handle/create_cube_res_group.md)接口同时使用。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

```
AscendC::KfcWorkspace desc(workspaceGM);
GM_ADDR workspace = desc.GetKfcWorkspace();
```

