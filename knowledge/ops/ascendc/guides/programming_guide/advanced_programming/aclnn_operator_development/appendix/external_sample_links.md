---
type: Guide
title: 外部样例链接汇总（附录）
description: 本文是参考内容，汇总指南中引用的外部代码样例和API参考链接，便于开发者快速查找和参考。
tags:
- task.lookup
- topic.implementation
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/programming_guide/advanced_programming/aclnn_operator_development/appendix/external_sample_links.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 外部样例链接汇总

本文是参考内容，汇总指南中引用的外部代码样例和API参考链接，便于开发者快速查找和参考。

## 代码样例

### 基础算子实现

- [AddCustom算子样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/custom_op)里的AddCustom样例。
  - 说明：展示基础的完整算子实现，包括TilingData定义、Tiling函数和核函数（Kernel）代码。

### msOpGen工具生成自定义算子工程模板

- [msOpGen样例](https://gitcode.com/cann/asc-tools/tree/master/examples/03_msopgen)。
  - 说明：基于AddCustom算子演示如何使用msOpGen工具生成自定义算子工程模板，并通过ACLNN调用示例验证生成的自定义算子。

### TilingData传递属性

- [LeakyReluCustom](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/custom_op/op_host/leaky_relu_custom)里的leaky_relu_custom样例。
  - 说明：通过TilingData传递属性信息。

### 模板编程算子实现

- [核函数（Kernel）模板编程样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/custom_op/op_host/add_custom_template)。
  - 说明：展示使用核函数（Kernel）模板参数实现编译期分支控制的示例。

### 编译与集成样例

- [静态库集成和使用样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/custom_op_static_lib)。
  - 说明：展示如何将算子编译为静态库并集成到应用中。

- [aclnn单算子调用样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/02_features/99_acl_based/01_acl_invocation/aclnn_invocation)。
  - 说明：展示如何通过aclnn API调用自定义算子。

## API参考

### Tiling API

- [Tiling数据结构](../../../../../apis/utils_api/tiling_data_structure_register/index.md)。
  - 说明：Tiling数据结构参考。

- [GetLibApiWorkSpaceSize](../../../../../apis/utils_api/platform_info/platform_ascend_c/get_lib_api_work_space_size.md)。
  - 说明：获取Ascend C API预留workspace大小的方法。

### 模板编程API

- [模板参数定义](../../../../../apis/utils_api/tiling_template_programming/template_params_definition.md)。
  - 说明：ASCENDC_TPL_ARGS_DECL、ASCENDC_TPL_ARGS_SEL等模板参数宏的定义。

### 原型注册API

- [OpAICoreConfig注册接口](../../../../../apis/utils_api/prototype_register_management/register_op_aicore_config.md)。
  - 说明：OpAICoreConfig类用于多硬件平台差异化原型注册。

## 其他参考

- [《图开发》](https://gitcode.com/cann/ge/blob/master/docs/zh/user_guides/graph_dev/README.md)。
  - 说明：Ascend Graph图开发的完整指南，适用于算子入图场景。
