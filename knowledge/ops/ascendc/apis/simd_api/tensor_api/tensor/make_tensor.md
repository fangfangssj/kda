---
type: API
title: make_tensor
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/tensor/make_tensor.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/tensor_api/tensor.h
---

# make_tensor

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

make_tensor负责将内存指针（由[make_mem_ptr](make_mem_ptr.md)生成）与布局对象（由[make_frame_layout](../layout/make_frame_layout.md)/[make_layout](../layout/make_layout.md)生成）绑定，生成带有完整类型信息的Tensor对象。

make_tensor会根据内存指针的物理位置标记，自动推导返回global_tensor或local_tensor，当指针物理位置位于Global Memory（外部存储）时，返回global_tensor；当指针位置位于AI Core上的Local Memory（内部存储）时，返回local_tensor。

生成的Tensor对象可直接传入copy（数据搬运）和mmad（矩阵计算）等接口，编译器会根据其Engine中的物理位置标记和layout中的布局模式，自动选择底层硬件指令。

## 函数原型

```cpp
template <typename Iterator, typename... Args>
__aicore__ inline auto make_tensor(const Iterator& iter, const Args&... args)
```

## 参数说明

**表1**  参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| iter | 输入 | 迭代器，要求传入view_engine对象，用于创建Tensor。返回类型由该迭代器绑定的内存位置决定。 |
| args... | 输入 | 可变参数，当前支持传入一个或两个参数。<br>&bull;当传入一个入参时：根据传入的layout对象创建Tensor。<br>&bull;当传入两个入参时：根据传入的参数构建layout对象，并基于该layout创建Tensor。 |

## 返回值说明

- 返回`global_tensor<Engine, Layout>`或`local_tensor<Engine, Layout>`类型的张量对象，具体由engine绑定的内存位置决定。

## 调用示例

  ```cpp
  #include "tensor_api/tensor.h"

  using namespace asc::te;

  // 示例1：使用GM指针和layout创建global_tensor
  auto gm_ptr = make_mem_ptr<location::gm, half>(gm_addr);
  auto gm_layout = make_layout(make_shape(32, 32), make_stride(32, 1));
  auto gm_tensor = make_tensor(gm_ptr, gm_layout);

  // 示例2：使用Local Memory指针和layout创建local_tensor
  constexpr int tile_num = 4;
  __cbuf__ half data_ptr[tile_num]; // 初始化
  auto ptr = make_mem_ptr(data_ptr);
  auto layout = make_frame_layout<nz_layout_ptn, half>(32, 32);
  auto tensor = make_tensor(ptr, layout);

  // 示例3：使用指针和布局形状创建张量
  auto tensor2 = make_tensor(ptr, make_shape(32, 32), make_stride(32, 32));
  ```

## 配套 API 与常见组合

- [make_mem_ptr](make_mem_ptr.md)
- [make_frame_layout](../layout/make_frame_layout.md)
- [copy（Global Memory到Unified Buffer的数据搬运）](../reg_vector_compute/vector_compute_load/copy_gm_to_ub.md)
- [copy（Unified Buffer到Global Memory的数据搬运）](../reg_vector_compute/vector_compute_store/copy_ub_to_gm.md)
- [Algorithm](../algorithm.md)
