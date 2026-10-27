---
type: API
title: GetUserTag
description: 获取指定Tensor块的Tag信息，用户可以根据Tag信息对Tensor进行不同操作。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
- topic.programming_model.development_mode.tensor_programming
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/basic_api/data_structures/LocalTensor/GetUserTag.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- 310p
- '910'
source_headers:
- include/basic_api/kernel_tensor.h
---

# GetUserTag<a name="ZH-CN_TOPIC_0000002364845757"></a>

## 功能说明<a name="section618mcpsimp"></a>

获取指定Tensor块的Tag信息，用户可以根据Tag信息对Tensor进行不同操作。

## 函数原型<a name="section620mcpsimp"></a>

```cpp
__aicore__ inline TTagType GetUserTag() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

指定Tensor块的Tag信息。TTagType定义如下：

```cpp
using TTagType = int32_t
```

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```cpp
// 示例8
AscendC::LocalTensor<half> tensor1 = que1.DeQue<half>();
AscendC::TTagType tag1 = tensor1.GetUserTag();
AscendC::LocalTensor<half> tensor2 = que2.DeQue<half>();
AscendC::TTagType tag2 = tensor2.GetUserTag();
AscendC::LocalTensor<half> tensor3 = que3.AllocTensor<half>();
/* 使用Tag控制条件语句执行*/
if ((tag1 <= 10) && (tag2 >= 9)) {
    AscendC::Add(tensor3, tensor1, tensor2, TILE_LENGTH); // 当tag1小于等于10，tag2大于等于9的时候，才能进行相加操作。
}
```

