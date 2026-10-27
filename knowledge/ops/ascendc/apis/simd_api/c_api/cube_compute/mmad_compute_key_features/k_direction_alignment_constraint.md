---
type: API
title: K方向对齐约束
description: 特性说明：
tags:
- task.lookup
- topic.implementation.api_layer.c_api
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/c_api/cube_compute/mmad_compute_key_features/k_direction_alignment_constraint.md
  role: primary
created_at: '2026-10-20T07:46:54Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# K方向对齐约束

**特性说明：**


Atlas A2系列产品/Atlas A3系列产品，asc_mmad接口支持开启此特性。



Ascend 950PR&950DT系列产品上L0A Buffer为Nz排布，非Zz排布，A矩阵转置非对齐float数据类型场景，L0A Buffer在K方向8对齐，无需开启此特性去确保计算正确性。



K方向对齐的核心功能是通过k_direction_align参数控制在使用float数据类型（其他数据类型无需关注此特性）时，L0A Buffer和L0B Buffer矩阵在K方向上的对齐方式。

当设置k_direction_align = true时，L0A Buffer/L0B Buffer中的矩阵会将K方向对齐到 `ceil\(K/16\)\*16`。

![K方向对齐示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/mmad_kdirectionalign_c_api.png)

例如，当K=44和K=36时，都会被对齐到48。在这种情况下，对于K=44，L0A Buffer/L0B Buffer中的12个分形块都会被完整读入CUBE；而对于K=36的情况，虽然对齐后也是48，但只有其中的10个分形块会被读入CUBE。

![K方向对齐到48示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/kdirectionalign_k_is36.png)

当设置 `k_direction_align = false`时，L0A Buffer/L0B Buffer中的矩阵会将K方向对齐到 `ceil\(K/8\)\*8`。

例如，当K=44时对齐到48，而K=36时对齐到40。在这种情况下，对于K=44，L0A Buffer/L0B Buffer中的12个分形块都会被完整读入CUBE；而对于K=36的情况，L0A Buffer/L0B Buffer中的10个分形块都会被读取到CUBE中。

![K方向对齐到40示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/kdirectionalign_k_is36_false.png)

1. A矩阵非转置的场景，无需开启k_direction_align。

2. A矩阵转置的场景，需要开启k_direction_align。

当输入需要进行转置操作时，例如原始矩阵A的维度为K×M，在搬入L1 Buffer后，其存储形式会变为Nz排布（本质上仍为K×M）。此时应调用具备转置能力的矩阵搬入接口。该接口的内部实现如下图所示：在K方向上会进行16的对齐处理（如左图所示）。完成转置后，数据将呈现出如右图所示的Zz分形形式，存储在L0A Buffer中。为了避免读入无效数据，可以通过开启k_direction_align参数，仅读取有效分形的数据。

![K方向对齐转置示意图](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/kdirectionalign_demo.png)


该参数在[asc_mmad](../mmad_compute/asc_mmad.md)的Atlas A3系列产品和Atlas A2系列产品对应重载中提供；A、B矩阵的分形排布可参见[关键分形格式详解](../cube_compute_fractal_intro/key_fractal_format_details.md)，转置搬入可参见[asc_copy_l12l0a_trans](../../cube_datamove/cube_compute_load/asc_copy_l12l0a_trans.md)。


**使用示例：**

```cpp
asc_mmad(c, a, b, m, k, n, asc_unit_flag_mode::DISABLE,
         true /* k_direction_align */, false, true);
```

