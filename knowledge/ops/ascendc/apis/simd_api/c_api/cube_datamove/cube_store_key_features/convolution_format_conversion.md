---
type: API
title: 卷积格式转换
description: 在卷积运算中，L0C Buffer上存放的数据格式为NC1HWC0格式，Fixpipe指令通过Nz2ND能力，支持将L0C Buffer上数据转为NHWC格式输出。转换示意图如下所示：
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_store_key_features/convolution_format_conversion.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 卷积格式转换

在卷积运算中，L0C Buffer上存放的数据格式为NC1HWC0格式，Fixpipe指令通过[Nz2ND](nz2_nd.md)能力，支持将L0C Buffer上数据转为NHWC格式输出。转换示意图如下所示：

**图1** NC1HWC0转换为NHWC格式

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_NC1HWC0_NHWC.png)


特别的，针对Ascend 950PR&950DT系列产品，还支持Fixpipe指令通过[Nz2DN](nz2_dn.md)能力，将L0C Buffer上数据转为NCHW格式输出。转换示意图如下所示：

**图2** NC1HWC0转换为NCHW格式

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_NC1HWC0_NCHW.png)


卷积输入的im2col搬入、矩阵计算与结果搬出之间的关系见[矩阵计算流程](../../cube_compute/overview/cube_compute_flow.md)中的卷积场景。C API结果搬出入口为[asc_copy_l0c2gm](../cube_compute_store/index.md)；选择当前架构版本后，再按目标布局设置格式转换参数。
