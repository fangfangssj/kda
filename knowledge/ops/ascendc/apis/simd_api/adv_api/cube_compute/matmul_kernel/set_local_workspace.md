---
type: API
title: SetLocalWorkspace
description: 设置Matmul内部额外占用的UB临时空间，由用户预留并申请LocalTensor后传入，需在Iterate前调用。
tags:
- task.lookup
- topic.implementation.api_layer.high_level_api
- topic.programming_model.execution_unit.cube
status: stable
sources:
- resource: cann-docs-raw/asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetLocalWorkspace.md
  role: primary
created_at: '2026-08-18T14:51:00Z'
updated_at: '2026-09-08T03:31:04Z'
platforms:
- '950'
- 310p
---

# SetLocalWorkspace

## 功能说明

对于某些场景Matmul内部需要额外占用Unified Buffer（UB，VECCALC）空间，如果用户希望在算子中复用这个额外占用的UB（VECCALC）空间，则该空间需要用户预留，并申请好LocalTensor，将其起始物理地址传入给Matmul。具体需要申请的UB（VECCALC）临时空间大小由tiling接口[MatmulGetTmpBufSize](../get_matmul_compute_space/matmul_get_tmp_buf_size.md)给出，满足以下几个条件之一就需要使用该接口传入UB临时空间：

-   C矩阵Position为TPosition::GM；
-   C矩阵CubeFormat为CubeFormat::ND；
-   A矩阵或者B矩阵CubeFormat为CubeFormat::ND；
-   存在Bias且Bias的Position不是VECCALC。

请在[Iterate](iterate.md)或者[IterateAll](iterate_all.md)之前调用该接口。

获取到的UB临时空间大小以字节为单位。

## 函数原型

```
__aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tmpBuffer | 输入 | 临时空间，由用户申请并管理，TPosition为VECCALC。 |

## 返回值说明

无

## 约束说明

当开启MixDualMaster（双主模式）场景时，即模板参数[enableMixDualMaster](matmul_config.md#matmulconfig-params)设置为true，不支持使用该接口。

## 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetLocalWorkspace(mmFormatUb);    //设置临时VECCALC空间
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
if (tiling.isBias) {
    mm.SetBias(biasGlobal);
}
mm.IterateAll(gm_c);
mm.End();
```
