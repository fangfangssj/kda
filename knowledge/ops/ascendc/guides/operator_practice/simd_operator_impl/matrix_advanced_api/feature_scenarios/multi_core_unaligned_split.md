---
type: Guide
title: 多核非对齐切分
description: 多核场景，对矩阵进行切分时，若M、N、K无法整除singleCoreM 、singleCoreN、 singleCoreK时，就会出现尾块，即多核非对齐场景。如下图矩阵A、B的最后一行和最后一列的矩阵块：
tags:
- task.lookup
- topic.programming_model.execution_model.simd
status: draft
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/guide/operator_practice/simd_operator_impl/matrix_advanced_api/feature_scenarios/multi_core_unaligned_split.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-10-20T07:46:54Z'
platforms: []
---

# 多核非对齐切分<a name="ZH-CN_TOPIC_0000002500548096"></a>

## 功能介绍<a name="zh-cn_topic_0000002264134828_section310824820358"></a>

多核场景，对矩阵进行切分时，若M、N、K无法整除singleCoreM 、singleCoreN、 singleCoreK时，就会出现尾块，即多核非对齐场景。如下图矩阵A、B的最后一行和最后一列的矩阵块：

![](https://raw.gitcode.com/cann/asc-devkit/raw/639b2b5c1031188107d23d785d52b935f1b4f13b/docs/zh/guide/figures/nd2nz_new_1.png)

此时，C矩阵中的R矩阵块，依然是通过A1\*B1+A2\*B2+A3\*B3+A4\*B4累加得到的，处理A1\*B1、A2\*B2、A3\*B3、A4\*B4等尾块时，需在核函数（Kernel）侧设置尾块大小，在不改变原有tiling的情况下，调用[SetTail](../../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_tail.md)接口重新设置本次计算的singleCoreM/singleCoreN/singleCoreK，在处理尾块的时候按照设置的值也就是tailM/tailN/tailK进行搬运和计算。

## 使用场景<a name="zh-cn_topic_0000002264134828_section118051016163613"></a>

多核处理Matmul矩阵计算，存在尾块的场景。

## 约束说明<a name="zh-cn_topic_0000002264134828_section14160134220363"></a>

处理尾块调用的SetTail接口，需要在Iterate/IterateAll之前调用。

## 调用示例<a name="zh-cn_topic_0000002264134828_section15486294368"></a>

Matmul多核非对齐场景的完整样例请参考[Matmul多核非对齐切分算子样例](https://gitcode.com/cann/asc-devkit/tree/639b2b5c1031188107d23d785d52b935f1b4f13b/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_multi_core_unaligned)。该场景的关键代码示例如下。

```
// 处理尾块
int tailM = tiling.M - mCoreIndex * tiling.singleCoreM;
tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
int tailN = tiling.N - nCoreIndex * tiling.singleCoreN;
tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
// 当tailM < singleCoreM或tailN < singleCoreN时被认为需要处理尾块，此时可以调用SetTail接口进行设置
if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN) {
    matmulObj.SetTail(tailM, tailN);
}
```
