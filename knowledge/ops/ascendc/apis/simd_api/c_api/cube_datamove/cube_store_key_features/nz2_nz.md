---
type: API
title: NZ2NZ
description: 矩阵计算的搬出过程中Nz2Nz的核心功能体现为，矩阵计算结果搬出前后保持Nz大分形排布；输出类型转换及Channel Split/Channel
  Merge可能改变小分形宽度。以下伪代码仅描述未启用通道拆分/合并的逐块地址关系。
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_datamove/cube_store_key_features/NZ2NZ.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# NZ2NZ

## 特性说明

矩阵计算的搬出过程中Nz2Nz的核心功能体现为，矩阵计算结果搬出前后保持Nz大分形排布；输出类型转换及Channel Split/Channel Merge可能改变小分形宽度。以下伪代码仅描述未启用通道拆分/合并的逐块地址关系。


以如下产品型号为例：

Atlas A3系列产品


Atlas A2系列产品

Nz2Nz转化过程可以参考以下伪代码（src/dst及临时地址均按字节地址计数）：

```cpp
for (j = 0; j < m_size; j++) {
    src_temp_n_addr = src + j * sizeof(L0C_element) * 16;
    dst_temp_n_addr = dst + j * 16 * sizeof(DST_element);
    for (k = 0; k < ((n_size + 15) / 16); k++) {
        src_block_addr = src_temp_n_addr + k * src_stride * sizeof(L0C_element) * 16;
        dst_block_addr = dst_temp_n_addr + k * dst_stride * 32;
    }
}
```

伪代码使用的参数具体说明如下：

- src表示源操作数搬运的起始地址。
- dst表示目的操作数搬运的起始地址。
- 源操作数中Nz矩阵每一行的起始地址为src\_temp\_n\_addr。
- 目的操作数中Nz矩阵每一行的起始地址为dst\_temp\_n\_addr。
- 在每一行中，以16个数据为粒度进行数据块切分，源操作数每一块数据块的起始地址为src\_block\_addr，目的操作数中每一数据块起始地址为dst\_block\_addr。

以下参数为Nz2Nz用户可配置参数（2201目的步长参数名为dst_stride_dst_d，本文统一记为dst_stride），含义及取值范围参见[C API搬出参数](../cube_compute_store/index.md)：

- m_size与n_size分别表示矩阵M方向和N方向上的维度。
- src_stride表示源Nz矩阵中不同Z分形列的间隔，单位为C0\_Size（C0\_Size=16\*sizeof\(T\)，T为操作数的数据类型）。
- dst_stride表示目标Nz矩阵中不同Z分形列的间隔，单位为32字节。



以Ascend 950PR&950DT系列产品为例，Nz2Nz转化过程可以参考以下伪代码（src/dst及临时地址均按字节地址计数）：

```cpp
for (j = 0; j < m_size; j++) {
    src_temp_n_addr = src + j * sizeof(L0C_element) * 16;
    dst_temp_n_addr = dst + j * 16 * sizeof(DST_element);
    for (k = 0; k < ((n_size + 15) / 16); k++) {
        src_block_addr = src_temp_n_addr + k * src_stride * sizeof(L0C_element) * 16;
        dst_block_addr = dst_temp_n_addr + k * dst_stride * sizeof(DST_element);
    }
}
```

伪代码使用的参数具体说明如下：

- src表示源操作数搬运的起始地址。
- dst表示目的操作数搬运的起始地址。
- 源操作数中Nz矩阵每一行的起始地址为src\_temp\_n\_addr。
- 目的操作数中Nz矩阵每一行的起始地址为dst\_temp\_n\_addr。
- 在每一行中，以16个数据为粒度进行数据块切分，源操作数每一块数据块的起始地址为src\_block\_addr，目的操作数中每一数据块起始地址为dst\_block\_addr。

以下参数为Nz2Nz用户可配置参数，含义及取值范围参见[C API搬出参数](../cube_compute_store/index.md)：

- m_size与n_size分别表示矩阵M方向和N方向上的维度。
- src_stride表示源Nz矩阵中不同Z分形列的间隔，单位为C0\_Size（C0\_Size=16\*sizeof\(T\)，T为操作数的数据类型）。
- dst_stride表示目标Nz矩阵中不同Z分形列的间隔，单位为element。


本特性用于保持分形输出布局的场景。各架构的搬出参数见[asc_copy_l0c2gm](../cube_compute_store/index.md)和[asc_copy_l0c2l1](../cube_compute_store/index.md)；输出类型与小分形宽度的关系见[随路量化](accompanying_quantization.md)、[F32 Channel Split](f32_channel_split.md)和[Int8 Channel Merge](int8_channel_merge.md)。

## 特性约束

- 当Nz2Nz与F32 Channel Split特性组合开启时，搬出到GM的Nz分形中的小分形维度为16\*8，确保小z分形的大小为512B。
- 对于B8数据类型Nz输出时，Channel Merge默认开启，搬出GM/L1 Buffer小分形维度为16\*32，确保小z分形的大小为512B。
- 对于B4数据类型Nz输出时，Channel Merge默认开启，小分形维度为16\*64，确保小z分形的大小为512B。
- 目标操作数地址之间不能重叠。

## 使用示例

如下图所示，在L0C Buffer中有一个Nz格式的矩阵，该矩阵N方向上维度为48，M方向上维度为24，经过Nz2Nz转换，将数据格式变为右部分所示排布。


**图1** Nz2Nz示意图（[NPU架构版本2201](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)）

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_nz2nz_c_api.png)

示意图中参数设置示例（通过L0C Buffer搬出接口搬运并去除dummy数据）和解释说明如下：

- 当M方向上的数据元素个数不是16的倍数时，源分形可能包含补齐的dummy数据，搬出按m_size选择有效行，dummy行不写入目标位置。矩阵块被定义为连续的16\*16的数据块，数据块的个数为M/16向上取整，矩阵块的长度为M\*16\*sizeof\(T\)，T是数据类型。
- n_size=48，表示源Nz矩阵中待搬运矩阵在N方向上的大小为48个元素。
- m_size=24，表示源Nz矩阵中待搬运矩阵在M方向上的大小为24个元素。
- src_stride=64，表示源Nz矩阵中待搬运矩阵相邻Z排布的起始地址偏移，即下图中第一个Z排布的起始地址与第二个Z排布的起始地址之间的间隔为64 \* C0\_Size。
- dst_stride = 40，表示目的Nz矩阵中相邻Z排布的起始地址偏移，即下图中第一个Z排布的起始地址与第二个Z排布的起始地址之间的间隔为40\*32字节。



**图2** Nz2Nz示意图（[NPU架构版本3510](../../../../../guides/programming_guide/language_extension/simd_builtin_keywords.md)）

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/fixpipe_nz2nz_c_api_950.png)

示意图中参数设置示例（通过L0C Buffer搬出接口搬运并去除dummy数据）和解释说明如下：

- 当M方向上的数据元素个数不是16的倍数时，源分形可能包含补齐的dummy数据，搬出按m_size选择有效行，dummy行不写入目标位置。矩阵块被定义为连续的16\*16的数据块，数据块的个数为M/16向上取整，矩阵块的长度为M\*16\*sizeof\(T\)，T是数据类型。
- n_size=48，表示源Nz矩阵中待搬运矩阵在N方向上的大小为48个元素。
- m_size=24，表示源Nz矩阵中待搬运矩阵在M方向上的大小为24个元素。
- src_stride=64，表示源Nz矩阵中待搬运矩阵相邻Z排布的起始地址偏移，即下图中第一个Z排布的起始地址与第二个Z排布的起始地址之间的间隔为64 \* C0\_Size。
- dst_stride = 40 * 16，表示目的Nz矩阵中相邻Z排布的起始地址偏移，即下图中第一个Z排布的起始地址与第二个Z排布的起始地址之间的间隔为40\*16个元素。

