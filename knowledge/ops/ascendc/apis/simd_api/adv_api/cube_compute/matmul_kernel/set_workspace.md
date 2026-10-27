---
type: API
title: SetWorkspace
description: Iterate计算的异步场景，调用本接口申请一块临时空间来缓存计算结果，然后调用GetTensorC时会在该临时空间中获取C的矩阵分片。
  IterateNBatch计算时，调用本接口申请一块临时空间来缓存计算结果，然后根据同步或异步场景进行其它接口的调用。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetWorkspace.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- a2
- a3
- kirin_x90
---

# SetWorkspace

## 功能说明

[Iterate](iterate.md)计算的[异步场景](get_tensor_c.md#li17508136205415)，调用本接口申请一块临时空间来缓存计算结果，然后调用[GetTensorC](get_tensor_c.md)时会在该临时空间中获取C的矩阵分片。

[IterateNBatch](iterate_n_batch.md)计算时，调用本接口申请一块临时空间来缓存计算结果，然后根据[同步或异步场景](iterate_n_batch.md)进行其它接口的调用。

## 函数原型

建议用户使用GlobalTensor类型传入：

```
template <class T> __aicore__ inline void SetWorkspace(GlobalTensor<T>& addr)
```

```
template <class T> __aicore__ inline void SetWorkspace(__gm__ const T* addr, int size)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| addr | 输入 | 用户传入的GM上的workspace空间，GlobalTensor类型。 |
| addr | 输入 | 用户传入的GM上的workspace空间，GM地址类型。 |
| size | 输入 | 传入GM地址时，需要配合传入元素个数。 |

## 返回值说明

无

## 约束说明

当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](matmul_config.md#matmulconfig-params)设置为true，不支持使用该接口。

## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetWorkspace(workspaceGM);    //设置异步时使用的临时空间
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(biasGlobal);
}
mm.template Iterate<false>();
for (int i = 0; i < singleCoreM/baseM * singleCoreN/baseN; ++i) {
    mm.template GetTensorC<false>(ub_c);
}
```
