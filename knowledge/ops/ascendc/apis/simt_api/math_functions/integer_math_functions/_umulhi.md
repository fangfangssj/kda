---
type: API
title: __umulhi
description: 获取输入uint32类型数据x和y乘积的高32位。
tags:
- task.lookup
- topic.implementation.api_layer.simt_api
- topic.compute.elementwise.arithmetic
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMT-API/math_functions/integer_math_functions/__umulhi.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
source_headers:
- include/simt_api/device_functions.h
---

# \_\_umulhi

## 功能说明

头文件路径为：`"simt_api/device_functions.h"`。

获取输入uint32类型数据x和y乘积的高32位。

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002531050526.png)

## 函数原型

```cpp
unsigned int __umulhi(unsigned int x, unsigned int y)
```

## 参数说明

**表1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，乘数。 |
| y | 输入 | 源操作数，乘数。 |

## 返回值说明

输入uint32类型数据x和y乘积的高32位。

## 约束说明

无

## 调用示例

-   SIMT编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __global__ __launch_bounds__(1024) void KernelUmulhi(unsigned int* dst, unsigned int* x, unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __umulhi(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```cpp
    #include "simt_api/device_functions.h"

    __simt_vf__ __launch_bounds__(1024) inline void KernelUmulhi(__gm__ unsigned int* dst, __gm__ unsigned int* x, __gm__ unsigned int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __umulhi(x[idx], y[idx]);
    }
    ```

输入输出示例如下：

```text
x: 65536, 65536, 131072, 262144, 4294967295, 3, 5, 0
y: 65536, 131072, 65536, 65536, 4294967295, 3, 5, 0
result: 1, 2, 2, 4, 4294967294, 0, 0, 0
```

- [Cat 950 ACLNN 算子知识](../../../../operators/aclnn/math/conversion/cat_950.md)
- [GatherNd 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/gather_nd_950.md)
- [Eye 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/eye_950.md)
- [InplaceScatterUpdate 950 ACLNN 算子知识](../../../../operators/aclnn/nn/index/inplace_scatter_update_950.md)
- [OneHot 950 ACLNN 算子知识](../../../../operators/aclnn/math/math/one_hot_950.md)
