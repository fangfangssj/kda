---
type: API
title: 卷积格式转换
description: 在卷积运算中，L0C Buffer上存放的数据格式为NC1HWC0格式，Fixpipe指令通过NZ2ND能力，支持将L0C Buffer上数据转为NHWC格式输出。转换示意图如下所示：
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.execution_unit.cube
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_store_key_features/convolution_format_conversion.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 卷积格式转换

在卷积运算中，L0C Buffer上存放的数据格式为NC1HWC0格式，Fixpipe指令通过[NZ2ND](nz2_nd.md)能力，支持将L0C Buffer上数据转为NHWC格式输出。转换示意图如下所示：

**图1** NC1HWC0转换为NHWC格式<a name="zh-cn_topic_0000002515830464_fig63620361414"></a>  

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_NC1HWC0_NHWC.png)


特别的，针对Ascend 950PR&950DT系列产品，还支持Fixpipe指令通过[NZ2DN](nz2_dn.md)能力，将L0C Buffer上数据转为NCHW格式输出。转换示意图如下所示：

**图2**  NC1HWC0转换为NCHW格式<a name="fig188671630134414"></a>  

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_NC1HWC0_NCHW.png)

