---
type: API
title: Silu
description: 按元素做Silu激活运算，支持half和float数据类型，不支持源操作数与目的操作数地址重叠。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.compute.elementwise.activation
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/activation_functions/Silu_interface/Silu.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms:
- '950'
- a2
- a3
- 310p
- kirin_x90
- kirin_9030
source_headers:
- include/adv_api/activation/silu.h
---

# Silu

## 功能说明

按元素做Silu运算，计算公式如下：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000002350049101.png)

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/api/figures/formulaimage_0000001729001889.png)

## 函数原型

```
template <typename T, bool isReuseSource = false>
__aicore__ inline void Silu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint32_t dataSize)
```

## 参数说明

**表1**  模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 操作数的数据类型。支持的数据类型为：half、float。 |
| isReuseSource | 是否允许修改源操作数。该参数预留，传入默认值false即可。 |

**表2**  接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstLocal | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。 |
| srcLocal | 输入 | 源操作数。<br><br>类型为[LocalTensor](../../../basic_api/data_structures/local_tensor/index.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>源操作数的数据类型需要与目的操作数保持一致。 |
| dataSize | 输入 | 实际计算数据元素个数。 |

## 返回值说明

无

## 约束说明

-   操作数地址偏移对齐要求请参见[通用说明和约束](../../../general_description_and_constraints.md)。
-   **不支持源操作数与目的操作数地址重叠。**
-   当前仅支持ND格式的输入，不支持其他格式。


-   针对Ascend 950PR&950DT系列产品，接口内部计算对Subnormal的处理方式受编译选项`--cce-ftz`控制（默认值为`true`）：
    -   配置为`false`时，计算过程中保留Subnormal，并按照其实际数值参与后续计算。
    -   配置为`true`时，启用FTZ（Flush-To-Zero）模式，计算过程中产生或参与运算的Subnormal将按0处理，可能导致计算结果与保留Subnormal时存在精度差异。


## 调用示例

```
// dstLocal: 存放Silu计算结果的Tensor
// srcLocal: 存放Silu计算输入的Tensor

// 需要参与计算的元素个数为512
AscendC::Silu(dstLocal, srcLocal, 512);
```

结果示例如下：

```
输入数据(srcLocal):
[-100.    -50.    -20.    -10.     -5.     -4.     -3.     -2.
   -1.     -0.5    -0.4    -0.3    -0.2    -0.1    -0.01    0.
    0.      0.01    0.1     0.2     0.3     0.4     0.5     1.
    2.      3.      4.      5.     10.     20.     50.    100.  ]
输出数据(dstLocal):
[ -0.          -0.          -0.00000004  -0.00045398  -0.03346425  -0.07194484  -0.14227761  -0.23840587
  -0.2689414   -0.18877034  -0.16052495  -0.12766725  -0.0900332   -0.04750208  -0.004975     0.
   0.           0.005025     0.05249792   0.10996681   0.17233276   0.23947507   0.31122968   0.7310586
   1.761594     2.8577225    3.928055     4.9665356    9.999546    20.          50.         100.        ]
```

- [FFN A2 ACLNN 算子知识](../../../../../operators/aclnn/transformer/ffn/ffn.md)
- [AlltoAllAllGatherBatchMatMul A3 ACLNN 算子知识](../../../../../operators/aclnn/transformer/mc2/allto_all_all_gather_batch_mat_mul.md)
