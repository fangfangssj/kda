---
type: API
title: F32 Channel Split
description: F32 Channel Split是矩阵计算结果搬出过程中的通道拆分能力。当源类型和目的类型均为float、目的格式为NZ，且l0c_to_gm_trait::enable_channel_split或l0c_to_ub_trait::enable_channel_split设置为true时，硬件会将16×16的分形矩阵拆分为两个独立的16×8小Z分形矩阵。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/matrix_compute/cube_store_key_features/f32_channel_split.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-09-12T03:09:26Z'
platforms: []
source_headers:
- include/tensor_api/tensor.h
---

# F32 Channel Split

F32 Channel Split是矩阵计算结果搬出过程中的通道拆分能力。当源类型和目的类型均为`float`、目的格式为NZ，且`l0c_to_gm_trait::enable_channel_split`或`l0c_to_ub_trait::enable_channel_split`设置为`true`时，硬件会将16×16的分形矩阵拆分为两个独立的16×8小Z分形矩阵。

启用该能力后，目的NZ输出的C0按8处理，N方向大小必须为8的倍数。如下图所示，当源操作数形状为`[64, 32]`时，搬出结果会被拆分为16个独立的16×8分形矩阵。当源操作数形状为`[64, 24]`时，搬出结果会被拆分为3列16×8分形矩阵。

**图1**  F32 Channel Split示意图1

![F32 Channel Split示意图1](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_Channel_Split.png)

**图2**  F32 Channel Split示意图2

![F32 Channel Split示意图2](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/Fixpipe_Channel_Split_with_dirty.png)

## 调用形式

```cpp
#include "tensor_api/tensor.h"

using namespace asc::te;

auto l0c = make_tensor(make_mem_ptr(l0c_addr), make_frame_layout<nz_layout_ptn, 16>(m, n));
auto gm = make_tensor(make_mem_ptr(gm_addr), make_frame_layout<nz_layout_ptn, float>(m, n));

constexpr l0c_to_gm_trait l0c_to_gm_trait_value = {
    round_mode::default_round,
    false,
    true
};

struct copy_l0c_to_gm_trait_custom {
    using trait_type = l0c_to_gm_trait;
    static constexpr const trait_type value = l0c_to_gm_trait_value;
};

l0c_to_gm_params params;
auto atom = make_copy(copy_l0c_to_gm{}, copy_l0c_to_gm_trait_custom{}).with(params);
copy(atom, gm, l0c);
```

## 约束说明

- F32 Channel Split仅支持源类型和目的类型均为`float`，目的格式为NZ。
- 启用F32 Channel Split时，不能同时使能`unit_flag`，N方向大小必须为8的倍数。
