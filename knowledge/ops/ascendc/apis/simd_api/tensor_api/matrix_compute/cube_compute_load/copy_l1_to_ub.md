---
type: API
title: copy（L1 Buffer到Unified Buffer数据搬运）
description: 头文件路径为：tensor_api/tensor.h。
tags:
- task.lookup
- topic.implementation.api_layer.basic_api
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/tensor_api/matrix_compute/cube_compute_load/copy_l1_to_ub.md
  role: primary
created_at: '2026-09-03T08:52:46Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
---

# copy（L1 Buffer到Unified Buffer数据搬运）

## 功能说明

头文件路径为：`tensor_api/tensor.h`。

本接口将源张量的数据从L1 Buffer搬运到Unified Buffer（UB）。接口根据源张量和目的张量的Layout选择对应搬运实现。

接口支持完整Tensor搬运，也支持通过`dst_coord`、`src_coord`和`copy_shape`指定目的Tensor中的起始坐标、源Tensor中的起始坐标和搬运区域的形状，执行区域搬运。

## 函数原型

- 执行L1 Buffer到UB搬运。

    ```cpp
    template <typename Atom, typename DstTensor, typename SrcTensor>
    __aicore__ inline void copy(const copy_atom<Atom>& atom,
        const DstTensor& dst, const SrcTensor& src)
    ```

- 根据源张量和目的张量的存储位置自动推导搬运通路，使用[默认trait](../../algorithm.md#copy_traits)执行L1 Buffer到UB搬运。

    ```cpp
    template <typename DstTensor, typename SrcTensor>
    __aicore__ inline void copy(const DstTensor& dst, const SrcTensor& src)
    ```

- 按指定源坐标、目的坐标和搬运形状执行L1 Buffer到UB搬运。

    ```cpp
    template <typename Atom, typename DstTensor, typename SrcTensor, typename DstCoord,
        typename SrcCoord, typename CopyShape>
    __aicore__ inline void copy(const copy_atom<Atom>& atom, const DstTensor& dst,
        const SrcTensor& src, const DstCoord& dst_coord, const SrcCoord& src_coord,
        const CopyShape& copy_shape)
    ```

- 根据源张量和目的张量的存储位置自动推导搬运通路，使用默认trait按指定源坐标、目的坐标和搬运形状执行L1 Buffer到UB搬运。

    ```cpp
    template <typename DstTensor, typename SrcTensor, typename DstCoord,
        typename SrcCoord, typename CopyShape>
    __aicore__ inline void copy(const DstTensor& dst,
        const SrcTensor& src, const DstCoord& dst_coord, const SrcCoord& src_coord,
        const CopyShape& copy_shape)
    ```

- 使用默认trait构造搬运原子对象。

    ```cpp
    template <typename CopyOperation>
    __aicore__ inline constexpr auto make_copy(const CopyOperation& operation)
    ```

- 使用指定trait构造搬运原子对象。

    ```cpp
    template <typename CopyOperation, typename CopyTrait>
    __aicore__ inline constexpr auto make_copy(const CopyOperation& operation,
        const CopyTrait& trait)
    ```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| :--- | :--- |
| Atom | `copy_atom`的模板参数，由`make_copy`接口推导得到。 |
| DstTensor | 目的张量类型，需为Tensor API Tensor类型。 |
| SrcTensor | 源张量类型，需为Tensor API Tensor类型。 |
| DstCoord | 目的张量起始坐标类型，也可为`zero_coord`。 |
| SrcCoord | 源张量起始坐标类型，也可为`zero_coord`。 |
| CopyShape | 搬运区域形状的类型。 |
| CopyOperation | 搬运操作对象类型。 |
| CopyTrait | 搬运trait对象类型。 |

**表2**  `copy`接口参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| atom | 输入 | 搬运原子对象，封装特定搬运通路的Operation、Trait与可选运行时Params组合，可由`make_copy`接口构造，详见[Algorithm](../../algorithm.md#copy_atom)。 |
| dst | 输出 | 目的张量，存储位置必须为`location::ub`。 |
| src | 输入 | 源张量，存储位置必须为`location::l1`。 |
| dst_coord | 输入 | 搬运区域在目的张量中的起始坐标。 |
| src_coord | 输入 | 搬运区域在源张量中的起始坐标。 |
| copy_shape | 输入 | 搬运区域的形状，用于指定搬运区域的大小。 |

**表3**  `make_copy`接口参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| operation | 输入 | 搬运操作对象。L1 Buffer到UB搬运取`copy_l1_to_ub{}`。 |
| trait | 输入 | 搬运trait对象。L1 Buffer到UB默认取`l1_to_ub_trait_default{}`。 |

### copy_l1_to_ub说明

`copy_l1_to_ub`用于标识L1 Buffer到UB数据搬运通路，仅列出其public内容：

```cpp
struct copy_l1_to_ub {
    template <typename Trait, const Trait& trait, typename... Args>
    __aicore__ inline static void copy(const Args&... args);
};
```

`copy`静态成员函数用于接收Trait和搬运参数，并分发L1 Buffer到UB数据搬运。

### l1_to_ub_trait说明

`l1_to_ub_trait`表示该通路的静态配置。当前没有可配置成员，定义如下：

```cpp
struct l1_to_ub_trait {};
```

`l1_to_ub_trait_default`提供该通路的默认Trait类型和默认值：

```cpp
struct l1_to_ub_trait_default {
    using trait_type = l1_to_ub_trait;
    static constexpr const trait_type value = {};
};
```


## 返回值说明

`copy`无返回值。`make_copy`返回`copy_atom`对象。

## 约束说明

- `dst`必须位于UB，`src`必须位于L1 Buffer。
- 源张量和目的张量的Layout组合需要属于该通路支持的数据排布。
- 使用坐标搬运时，coord和copy_shape需要与对应张量的形状结构匹配，且搬运范围不能越界。
