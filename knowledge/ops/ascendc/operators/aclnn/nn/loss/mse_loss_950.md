---
type: Operator
title: "MseLoss 950 ACLNN 算子知识"
description: "说明 aclnnMseLoss 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_api/aclnn_mse_loss.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/docs/aclnnMseLoss.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_api/aclnn_mse_loss.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_api/mse_loss.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/common/level2_base_loss.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_host/mse_loss_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_host/arch35/mse_loss_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_host/arch35/mse_loss_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_kernel/arch35/mse_loss_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_kernel/arch35/mse_loss_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_kernel/arch35/mse_loss_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_kernel/mse_loss.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/third_party/opbase.cmake
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/elewise/elewise_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_tiling_data.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_tiling_key_decl.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/src/op_common/atvoss/reduce/reduce_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_sch_aux.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_sch_aux_base.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_util.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/reduce/reduce_operator.h
    role: implementation
created_at: "2026-08-25T17:21:38Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MseLoss 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMseLossGetWorkspaceSize` / `aclnnMseLoss`，公开 ACLNN；950 由 L0 selector 进入 `MseLossV1`，其算子类型为 `MseLoss` |
| 计算语义 | `none: y_i=(self_i-target_i)^2`；`sum: y=Σ_i(...)`；`mean: y=(1/N)Σ_i(...)` |
| 输入输出 | 两输入规范化成同 shape/同 dtype ND；FP16/FP32/BF16；none 输出同 shape，sum/mean 输出标量 |
| 执行类型 | 单 `AIV_VECTOR` Kernel；elementwise 使用 ATVOSS `ElementwiseSch`，归约使用 ATVOSS `ReduceSch`；`useNddma` 可令 CopyIn 走 NDDMA 子路径，AR-group 在同一 Kernel 内以 GM workspace + `SyncAll` 两阶段执行 |
| Kernel 路线 | R-001 none-FP32；R-002 none-FP16/BF16；R-003 sum-AR normal；R-004 sum-AR group；R-005 sum-A；R-006 mean-AR normal；R-007 mean-AR group；R-008 mean-A |
| 核心资源 | Elementwise 的单 TBuf 分区/互斥复用；ReduceSch 的 pre/post/result UB、16 KiB cache reserve、可选 NDDMA、16 MiB 框架 workspace reserve及 group partial slice |
| 来源闭包 | source_closed：ACLNN → L0 `MseLossV1` → 950 `MseLoss` OpDef/arch35 Tiling → pinned opbase@b8bd… Elementwise/Reduce 模板 → `mse_loss` entry/DAG → ascend950 构建接线均已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令广播后元素数为 `N`。none 的输出有效域为 N 个平方差；sum/mean 在所有维上归约，mean 的 Host `meanVar` 由通用 Reduce Tiling 生成并经 `SetVar<PromoteType,0>` 注入 DAG 的 `Muls`。FP16/BF16 的差、平方和归约均提升到 FP32；最终再按 DAG round mode 1 cast 回输出 dtype。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，rank 1~8 | FP16/FP32/BF16 | 可非连续、可广播；核心 Kernel 前变成连续同 shape |
| `target` | ND，rank 1~8 | FP16/FP32/BF16 | 与 self 做 promotion、Contiguous/Cast/BroadcastTo |
| `reduction` | 字符串 | `char *` | none/sum/mean 映射 0/1/2 |
| `out` | none 为共同 shape；sum/mean 为 0 维 | promotion/out dtype | OpDef 为 ND；核心 Host 要求 input shape/dtype 相同 |

### 入口与版本

L0 selector 的 V2 SoC 白名单不含 ascend950，因此 950 总是进入 `MseLossV1`，通过 `ADD_TO_LAUNCHER_LIST_AICORE(MseLoss,...)` 启动本目录实现。OpDef 和 CMake 均只为 `ascend950` 接线。此卡不包含 `aclnnMseLossOut`，也不包含 backward/grad 变体。

## Host/Launcher 执行流程

### 参数校验与前处理

公开层顺序与 A2/A3 相同：非空指针、rank、reduction、广播/out shape、SoC dtype/promotion 校验；Contiguous、Cast、BroadcastTo。空域在 L0 MseLoss 前处理：none 直接返回，mean 用 Fill(NaN)，sum 用 Fill(0)。非空进入 950 Tiling 后再次要求两输入 storage shape 和 dtype 完全相同，并校验输出 dtype；none 走 `TilingEle`，sum/mean 把所有轴写入 `ReduceOpInputParam.axes` 后走通用 `Tiling4ReduceOp`。

### 参数、Workspace 与生命周期

ACLNN executor 持有规范化中间 Tensor、MSE 节点和最终 Cast/ViewCopy。none 的 MseLoss 节点显式设置 16 MiB workspace reserve；归约由通用 Reduce Tiling 设置 16 MiB reserve，并在 AR-group 时追加 `C * align(outSize*promoteBytes, cacheLine)` partial 区。workspace 所有权属于调用方/本次 executor，随给定 stream 消费。POD 是 `MseLossTilingData{baseTiling,reduceTiling}`；仅当前 route 对应的一半有效。

### 启动图与失败传播

```text
aclnnMseLossGetWorkspaceSize
 -> checks -> Contiguous/Cast/BroadcastTo
 -> empty branch: return or Fill->ViewCopy
 -> l0op::MseLoss -> MseLossV1(A5)
 -> arch35 TilingForMseLoss
    none: ElewiseBaseTiling::DoTiling
    sum/mean: Tiling4ReduceOp(all axes)
 -> Cast/ViewCopy
aclnnMseLoss -> CommonOpExecutorRun(stream)
 -> [单节点] mse_loss<REDUCE_TPL_VALUE,Reduction,Dtype><<<blockDim,AIV>>>
```

Elementwise 无多节点边；Reduce AR-group 的 stage1 partial → workspace → `SyncAll` → stage2 是同一 Kernel 内部边，producer 为所有 AIV，consumer 为归并 AIV，lifetime 为 Kernel 本次执行。所有 Host/Tiling/launch 错误均沿 executor 返回；空域不启动核心 Kernel。

## Route 与参数设计

### 分核与资源计算

Elementwise：`N=shape product`，`C=min(H,ceil(N*minDtypeBits/32768))`；每核 former 块按 512 对齐，`blockNum=ceil(N/blockFormer)`。设 `Bdag=OpDag::BufferNum`、`w=maxDtypeBytes`、UB 可用量 `Ue`，`ubFormer=floor_align(Ue/(Bdag*w), 256*8/minDtypeBits)`；POD 给出 former/tail 核的 loop 与 tail。Device 分配一个 `ubFormer*w*Bdag` TBuf，并由 DAG buffer id 切片。

Reduce：固定缓存预留 16 KiB。全轴归约经维度压缩后通常形成 AR；全为单例时可形成 A。通用 Tiling 生成 `basicBlock/resultBlock/groupR/coreNum/useNddma/meanVar` 等。UB 精确符号账为：`preBuf=basicBlock*PRE_MTE2_TOTAL + basicBlock*ratio*(PRE_MTE3_TOTAL+PRE_TEMP)`，`postBuf=resultBlock*(POST_MTE2_TOTAL+POST_TEMP+POST_MTE3_TOTAL)`，另有 `resultBlock` 和 16 KiB cache。`groupR>1` 选择 group 调度和 schedule mode 1；group workspace partial 为 `C*align(outSize*promoteBytes,cacheLine)`，叠加 16 MiB reserve。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 none-FP32 | empty reduce tuple, reduction0,dtype20 / `ElementwiseSch<MseLossOp<float>>` | 950、N>0、none、FP32 | 16 位去 R-002 |
| 2 | R-002 none-FP16/BF16 | empty tuple, reduction0,dtype10/30 / `MseLossOp<DTYPE_PREDICT>` | none、FP16/BF16 | dtype 失败 |
| 3 | R-003 sum-AR normal | AR_NORMAL,reduction1,dtype20 / `ReduceSch<...,MseLossSumDag>` | sum，压缩 pattern AR，`groupR==1` | group 去 R-004；A 去 R-005 |
| 4 | R-004 sum-AR group | AR_GROUP,reduction1,dtype20 | sum，pattern AR，`groupR>1` | A 去 R-005 |
| 5 | R-005 sum-A | A,reduction1,dtype20 | sum，压缩 pattern A | Tiling 不支持则失败 |
| 6 | R-006 mean-AR normal | AR_NORMAL,reduction2,dtype20 / mean DAG | mean，pattern AR，`groupR==1` | group 去 R-007；A 去 R-008 |
| 7 | R-007 mean-AR group | AR_GROUP,reduction2,dtype20 | mean，pattern AR，`groupR>1` | A 去 R-008 |
| 8 | R-008 mean-A | A,reduction2,dtype20 | mean，压缩 pattern A | Tiling 不支持则失败 |

```text
empty -> ACLNN empty branch
nonempty -> reduction none -> FP32:R-001 / FP16,BF16:R-002
         -> reduction sum  -> compressed A:R-005 / AR(groupR==1:R-003, groupR>1:R-004)
         -> reduction mean -> compressed A:R-008 / AR(groupR==1:R-006, groupR>1:R-007)
AR CopyIn 内部: useNddma==0 -> MoveAlign/DataCopyPad; useNddma!=0 -> NDDMA MultiCopy 子路径
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | EMPTY,0,20 | 950,N>0,none,FP32 | 单 AIV | `ElementwiseSch<0,MseLossOp<float>::OpDag>` | 单 TBuf 按 DAG buffer id 复用 | former/tail block POD |
| R-002 | EMPTY,0,10/30 | 950,N>0,none,FP16/BF16 | 单 AIV | `ElementwiseSch<0,MseLossOp<DTYPE_PREDICT>::OpDag>` | Cast FP32，单 TBuf 互斥复用 | DataCopyPad 保护尾 |
| R-003 | AR_NORMAL,1,20 | sum,AR,groupR=1 | 单 AIV ReduceSch | sum DAG + normal aux | pre/post/result UB；可选 NDDMA | sliceShape/sliceNum 控制尾 |
| R-004 | AR_GROUP,1,20 | sum,AR,groupR>1 | 同一 Kernel stage1→barrier→stage2 | sum DAG + group aux | GM partial + SyncAll | group partial 对齐 cacheLine |
| R-005 | A,1,20 | sum,A pattern | 单 AIV ReduceSch | sum DAG + A scheduler | A 维切片，无 group | 通用 reducer padding=0 |
| R-006 | AR_NORMAL,2,20 | mean,AR,groupR=1 | 单 AIV ReduceSch | mean DAG + normal aux | 同 R-003，另有 Var0 | padding/尾不计入均值 |
| R-007 | AR_GROUP,2,20 | mean,AR,groupR>1 | stage1→barrier→stage2 | mean DAG + group aux | 同 R-004 | 同 R-004 |
| R-008 | A,2,20 | mean,A pattern | 单 AIV ReduceSch | mean DAG + A scheduler | 同 R-005，Var0 | reducer 空域由上层截断 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| L0 version guard | 950 不在 V2 SoC 列表 | 选择 arch35 实现 | 全部 | `MseLossV1` |
| Same shape/dtype | storage shape 和 dtype 相同 | Device DAG 无 broadcast/mixed-input dtype | 全部 | Tiling 失败；公开层预处理保证 |
| Elementwise core cap | `C=min(H,ceil(N*minBits/32768))` | 控制小问题核数 | R-001/002 | 不会超过 H |
| Elementwise UB align | `alignFactor=256*8/minBits` | Vector/DataCopy 对齐 | R-001/002 | ubFormer 向下对齐 |
| 16 KiB reduce cache | reduce UB 先保留 cache | 累加/后处理缓存 | R-003~008 | Tiling 容量不足失败 |
| groupR | `groupR>1` | 单核不能一次覆盖全部 R，需要跨核 partial | R-004/007 | `groupR==1` 去 normal |
| NDDMA | stride/维数/cacheLine/factor 条件设置 `useNddma` | 非连续多维片搬运 | AR/A 内部 | `useNddma=0` 用 MoveAlign |
| 空域 | N==0 | 通用核心不承担公开空域语义 | 全部 | ACLNN return/Fill |

### POD/ABI

`MseLossTilingData` 顺序嵌入两个标准布局结构；下面按固定 opbase commit 的声明顺序列字段。编译器实际 `sizeof/alignof/offsetof` 数值没有在源码常量中发布，因此只记录成员顺序/位宽，不推测 padding。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `MseLossTilingData` | `baseTiling` | `EleBaseTilingData` | 第 1 成员；exact offset 0 | 结构 | `TilingEle` | `ElementwiseSch` | none 有效 |
| 同上 | `reduceTiling` | `ReduceOpTilingData` | 第 2 成员；exact offset 依前结构 sizeof | 结构 | `TilingReduce` | `ReduceSch` | sum/mean 有效 |
| `EleBaseTilingData` | `dim0` | int64/64 | 第 1 字段 | 元素 | `DoTiling` | scheduler | N |
| 同上 | `coreNum` | int32/32 | 第 2 字段 | 核 | 同上 | scheduler | <=H |
| 同上 | `ubFormer` | int32/32 | 第 3 字段 | 元素 | UB tiling | TBuf size/tile | 对齐后 |
| 同上 | `blockFormer` | int64/64 | 第 4 字段 | 元素 | core split | GM offset | 512 对齐 |
| 同上 | `blockNum` | int64/64 | 第 5 字段 | 核 | core split | block guard | <=coreNum |
| 同上 | `ubLoopOfFormerBlock` | int64/64 | 第 6 字段 | 次 | tiling | former loop | floor |
| 同上 | `ubLoopOfTailBlock` | int64/64 | 第 7 字段 | 次 | tiling | tail core loop | floor |
| 同上 | `ubTailOfFormerBlock` | int64/64 | 第 8 字段 | 元素 | tiling | former tail | 可为0 |
| 同上 | `ubTailOfTailBlock` | int64/64 | 第 9 字段 | 元素 | tiling | tail tail | 可为0 |
| 同上 | `elemNum` | int64/64 | 第 10 字段 | 元素 | tiling | valid bound | N |
| 同上 | `scheMode` | uint64/64 | 第 11 字段 | enum | tiling | scheduler | 固定仓定义 |
| `ReduceOpTilingData` | `factorACntPerCore` | uint64/64 | 第 1 字段 | A 数 | Reduce tiling | loops | >=1 |
| 同上 | `factorATotalCnt` | uint64/64 | 第 2 字段 | A 数 | 同上 | loops | shape-derived |
| 同上 | `ubFactorA` | uint64/64 | 第 3 字段 | A 数 | 同上 | UB slice | capacity-derived |
| 同上 | `factorRCntPerCore` | uint64/64 | 第 4 字段 | R 数 | 同上 | loops | shape-derived |
| 同上 | `factorRTotalCnt` | uint64/64 | 第 5 字段 | R 数 | 同上 | loops | N/A product |
| 同上 | `ubFactorR` | uint64/64 | 第 6 字段 | R 数 | 同上 | UB slice | capacity-derived |
| 同上 | `groupR` | uint64/64 | 第 7 字段 | group | 同上 | normal/group branch | >=1 |
| 同上 | `outSize` | uint64/64 | 第 8 字段 | 元素 | 同上 | output/workspace | 本算子标量=1 |
| 同上 | `basicBlock` | uint64/64 | 第 9 字段 | 元素 | 同上 | pre UB | 对齐后 |
| 同上 | `resultBlock` | uint64/64 | 第 10 字段 | 元素 | 同上 | post/result UB | 对齐后 |
| 同上 | `coreNum` | int32/32 | 第 11 字段 | 核 | 同上 | scheduler | <=H |
| 同上 | `useNddma` | int32/32 | 第 12 字段 | bool/int | stride guard | CopyIn helper | 0/1 |
| 同上 | `meanVar` | float/32 | 第 13 字段 | 无量纲 | Reduce mean tiling | `SetVar` | mean 有效 |
| 同上 | `shape` | uint64[9] | 第 14 字段 | 元素 | dimension transform | view | MAX_DIM=9 |
| 同上 | `stride` | int64[9] | 第 15 字段 | 元素 stride | 同上 | input view | MAX_DIM=9 |
| 同上 | `dstStride` | int64[9] | 第 16 字段 | 元素 stride | 同上 | output view | MAX_DIM=9 |
| 同上 | `sliceNum` | uint64[9] | 第 17 字段 | slice 数 | 同上 | loop | MAX_DIM=9 |
| 同上 | `sliceShape` | uint64[9] | 第 18 字段 | 元素 | 同上 | copy/reduce | MAX_DIM=9 |
| 同上 | `sliceStride` | uint64[9] | 第 19 字段 | 元素 stride | 同上 | copy | MAX_DIM=9 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | EMPTY,0,20 | `TilingEle<float>` | `baseTiling` | `mse_loss`, reduction0 | `ElementwiseSch` + FP32 DAG | ascend950 kernel CMake |
| R-002 | EMPTY,0,10/30 | `TilingEle<half>` | `baseTiling` | reduction0 binary dtype | `ElementwiseSch` + cast DAG | 同上 |
| R-003 | AR_NORMAL,1,20 | `Tiling4ReduceOp<SumDag>` | `reduceTiling` | reduction1 | `ReduceSch` normal + SumDag | 同上 |
| R-004 | AR_GROUP,1,20 | 同上，groupR>1 | 同上+workspace | reduction1 | group aux + SumDag | 同上 |
| R-005 | A,1,20 | 同上，A key | 同上 | reduction1 | A scheduler + SumDag | 同上 |
| R-006 | AR_NORMAL,2,20 | `Tiling4ReduceOp<MeanDag>` | reduce+meanVar | reduction2 | normal + MeanDag | 同上 |
| R-007 | AR_GROUP,2,20 | 同上，groupR>1 | 同上+workspace | reduction2 | group + MeanDag | 同上 |
| R-008 | A,2,20 | 同上，A key | reduce+meanVar | reduction2 | A + MeanDag | 同上 |

## Kernel 实现

### 公共执行流程

`mse_loss` 以 `REDUCE_TPL_PARAM,Reduction,Dtype` 编译期参数路由。none 构造 `ElementwiseSch`，绑定 predict/label/y 后执行 DAG。sum/mean 构造 `ReduceSch`；mean 先把 `meanVar` 注入 Var0。公共 DAG 数据流为 `CopyIn0/1 → Cast(PromoteType) → Sub → Mul`，sum 添加 `ReduceSumOp`，mean 再添加 `Muls(meanVar)`，最后 Cast/CopyOut。执行族为 AIV_VECTOR；只有 `useNddma!=0` 时 CopyIn helper 使用 NDDMA，未出现 Cube 或独立通信 Kernel。

### Route R-001：none-FP32 Elementwise

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `mse_loss_tiling_arch35.cpp` | `TilingEle` | output FP32, none | float DAG, dtype20 |
| Router | `mse_loss.cpp` | `Reduction==0` | binary DTYPE FP32 | ElementwiseSch |

#### Device 数据流

##### 函数级调用链

```text
mse_loss -> ElementwiseSch::Init -> Process
 -> DAG CopyIn0/1 -> Sub<float> -> Mul<float> -> CopyOut
```

##### 源码等价伪码

```text
for each block/tile: lock buffer ids; DataCopyPad inputs; y=(x0-x1)^2; DataCopyPad output; release ids
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs/output | `blockFormer*blockIdx*sizeof(T)+tileOffset` | block/tile | FP32 ND | executor/MTE3 | MTE2/caller | executor | DataCopyPad tail |
| shared TBuf | VECCALC | `ubFormer*4*Bdag` B | DAG buffer slots | MTE2/Vector | Vector/MTE3 | TPipe | mutex by buffer id |

##### 分层内存布局

```mermaid
block-beta
  columns 1
  ub["VECCALC TBuf · Bdag slots · each ubFormer·4 B"]
```

##### 字节账与复用顺序

```text
UB_total=Bdag*ubFormer*4 <= elementwise available UB; workspace reserve=16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG slot[id] | CopyIn/Vector op | dependent node | next holder | `GetTensor/ReleaseTensor` mutex | DAG refcount/lifetime complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| elementwise | TBuf mutex/event | MTE2/Vector | Vector/MTE3 | slot id | FP32；DataCopyPad 处理尾，输出仅 elemNum 有效 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | `mse_loss_tiling_arch35.cpp` | `TilingEle<float>` | elementwise POD/key |
| Entry | `mse_loss.cpp` | `Reduction==0` | 选 ElementwiseSch |
| Body | `mse_loss_dag.h`, opbase `elewise_sch.h` | `MseLossOp<float>::OpDag` | DAG 和 buffer 互斥流水 |

### Route R-002：none-FP16/BF16 Elementwise

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `mse_loss_tiling_arch35.cpp` | `TilingEle` | FP16 或 BF16 | half DAG 资源模型，key dtype10/30 |
| Router | `mse_loss.cpp` | `Reduction==0` | binary DTYPE_PREDICT | typed DAG |

#### Device 数据流

##### 函数级调用链

```text
CopyIn<T> -> Cast<float,T,0> x2 -> Sub<float> -> Mul<float> -> Cast<T,float,1> -> CopyOut<T>
```

##### 源码等价伪码

```text
for tile: copy T; fp32_a=cast(a); fp32_b=cast(b); fp32_y=(a-b)^2; y=cast_round1<T>(fp32_y); copy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | block/tile offset | logical tile | FP16/BF16 | executor/Kernel | Kernel/caller | executor | DataCopyPad |
| shared TBuf | VECCALC | `Bdag*ubFormer*4` B | T/FP32 slots | DAG nodes | DAG nodes | TPipe | max dtype bytes=4 |

##### 分层内存布局

```mermaid
block-beta
  columns 1
  ub["VECCALC TBuf · Bdag slots · stride ubFormer·4 B"]
```

##### 字节账与复用顺序

```text
UB_total=Bdag*ubFormer*4; Bdag=OpDag::BufferNum compile-time constant; workspace reserve=16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| typed/FP32 DAG slots | CopyIn/Cast/Sub/Mul | dependent op | allocator next use | ReleaseTensor | 节点消费完毕 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DAG | mutex/event | Copy/Cast/Vector | successors | UB slots | FP32 中间、round1 输出、padding 不计有效域 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | `mse_loss_tiling_arch35.cpp` | `MseLossOp<half>` | 16 位共同资源 tiling |
| Entry | `mse_loss.cpp` | `DTYPE_PREDICT` | BF16/FP16 由 binary specialization 区分 |
| Body | `mse_loss_dag.h` | `MseLossOp<T>` | 两次 input cast、一次 output cast |

### Route R-003：sum-AR normal

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | pinned opbase `reduce_tiling.cpp` | key selection | pattern AR 且 groupR==1 | AR_NORMAL |
| Router | `mse_loss.cpp` | Reduction==1 | SumDag | ReduceSch normal |

#### Device 数据流

##### 函数级调用链

```text
ReduceSch -> normal ReduceSchAux -> CopyIn(MoveAlign or NDDMA)
 -> pre DAG(square FP32) -> ReduceSum -> cache/post DAG -> Cast/CopyOut scalar
```

##### 源码等价伪码

```text
for assigned A/R slices: copy according to view; square_diff_fp32; zero-pad reduction tail; bisection ReduceSum; accumulate; cast/copy out
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM slices | stride/slice POD | sliceShape | T ND view | MTE2/NDDMA | pre DAG | executor | useNddma branch |
| pre/post/result UB | scheduler partition | symbolic formulas in byte账 | FP32/T | copy/DAG/reducer | reducer/MTE3 | TPipe | LEVEL_2 reuse |
| cache | UB | 16 KiB | FP32 partial | reducer | reducer | TPipe | fixed reserve |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  pre["preBuf"] post["postBuf"] result["resultBlock"] cache["16 KiB cache"]
```

##### 字节账与复用顺序

```text
UB_total=preBuf+postBuf+resultBlock+16KiB <= U
workspace_total=16MiB reserve (group partial=0)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| pre/post DAG slots | Copy/DAG | reducer/post DAG | next slice | MemLevel2 allocator/events | current slice consumed |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| normal reduce | pipe events; no global barrier | MTE2/NDDMA/Vector | dependent stage | UB | FP32 accumulation；reduction padding=0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | `mse_loss_tiling_arch35.cpp` | `Tiling4ReduceOp<SumDag>` | all axes, sum DAG |
| Entry | `mse_loss.cpp` | Reduction1 | ReduceSch |
| Body | pinned opbase `reduce_sch.h/aux.h` | AR normal | slice/reduce/copy |

### Route R-004：sum-AR group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase `reduce_tiling.cpp` | `groupR>1` | AR needs grouped R | AR_GROUP, mode1 |
| Router | `mse_loss.cpp` | Reduction1 | SumDag | group ReduceSch |

#### Device 数据流

##### 函数级调用链

```text
stage1 each AIV: CopyIn -> square -> partial reduce -> GM workspace
 -> SyncAll -> stage2 designated AIV: load group partials -> ReduceSum -> Cast/CopyOut
```

##### 源码等价伪码

```text
partial[core]=reduce(assigned R); SyncAll; stage2=sum(partial groups); output scalar
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| UB | 同 R-003 | pre+post+result+16KiB | FP32/T | stage ops | stage ops | TPipe | level2 |
| group workspace | 16MiB 后的 per-core slice | `C*align(outSize*4,cacheLine)` | FP32 | stage1 cores | stage2 | executor | cacheLine 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  pre["preBuf"] post["postBuf"] res["result"] cache["16 KiB"]
```

##### 字节账与复用顺序

```text
UB_total=preBuf+postBuf+resultBlock+16KiB
workspace_total=16MiB+C*align(outSize*4,cacheLine)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| group partial slice | stage1 core | stage2 | Kernel end | SyncAll | slice 不重叠且全部 core 已提交 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group boundary | `SyncAll` | stage1 AIVs | stage2 | GM partial | FP32 partial；padding=0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | opbase `reduce_tiling.cpp` | group key/workspace | groupR 和 slice |
| Entry | `mse_loss.cpp` | SumDag | sum route |
| Body | opbase `reduce_sch.h` | group branch | 两阶段和 barrier |

### Route R-005：sum-A

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase reduce tiling | compressed pattern A | 无 R 维工作 | A key |
| Router | `mse_loss.cpp` | Reduction1 | SumDag | A scheduler |

#### Device 数据流

##### 函数级调用链

```text
ReduceSch<A> -> CopyIn A slice -> square DAG -> ReduceSumOp/post -> scalar CopyOut
```

##### 源码等价伪码

```text
按 A slice 处理；固定仓由通用 A pattern body保持与 ReduceSch ABI一致。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | reduce POD | basic/result block | T/FP32 | Copy/DAG | reducer/output | executor/TPipe | 通用 A pattern 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  pre["preBuf"] post["postBuf"] res["result"] cache["16 KiB"]
```

##### 字节账与复用顺序

```text
UB_total=preBuf+postBuf+resultBlock+16KiB; workspace=16MiB reserve
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG slots | Copy/Vector | successors | next slice | level2 allocator | consumer complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| A body | pipe event | copy/vector | next stage | UB | FP32 square/reduce，零 padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | opbase reduce tiling | pattern A | A key |
| Entry | `mse_loss.cpp` | Reduction1 | SumDag |
| Body | opbase ReduceSch | A instance | A schedule |

### Route R-006：mean-AR normal

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `mse_loss_tiling_arch35.cpp` + opbase | MeanDag,AR,groupR1 | all axes | AR_NORMAL reduction2 |
| Router | `mse_loss.cpp` | Reduction2 | SetVar(meanVar) | mean ReduceSch |

#### Device 数据流

##### 函数级调用链

```text
同 R-003 -> ReduceSum -> Muls(meanVar) -> Cast -> CopyOut
```

##### 源码等价伪码

```text
sum=schedule_reduce(square_diff); out=cast(sum*meanVar)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB/cache | 同 R-003 | pre+post+result+16KiB | T/FP32 | scheduler | mean DAG | executor/TPipe | useNddma subpath |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  pre["preBuf"] post["postBuf+Muls"] res["result"] cache["16 KiB"]
```

##### 字节账与复用顺序

```text
UB_total=preBuf+postBuf+resultBlock+16KiB; workspace=16MiB reserve
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| post slot | ReduceSum | Muls/Cast | next slice | level2 DAG dependency | Muls 已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| mean | Var0 + pipe | Host/reducer | Muls | scalar partial | FP32 meanVar/accumulation；padding=0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | `mse_loss_tiling_arch35.cpp` | `MeanDag` | meanVar POD |
| Entry | `mse_loss.cpp` | `SetVar` | 注入 meanVar |
| Body | `mse_loss_dag.h` | `MseLossMeanDag` | ReduceSum 后 Muls |

### Route R-007：mean-AR group

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase reduce tiling | mean, AR, groupR>1 | group key/mode1 | R-007 |
| Router | `mse_loss.cpp` | Reduction2 | MeanDag | group ReduceSch |

#### Device 数据流

##### 函数级调用链

```text
stage1 square/reduce -> workspace partial -> SyncAll -> stage2 reduce -> Muls(meanVar) -> Cast/CopyOut
```

##### 源码等价伪码

```text
partial[core]=sum(square_diff slice); barrier; out=cast(sum(partials)*meanVar)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| UB/cache | 同 R-004 | pre+post+result+16KiB | T/FP32 | stages | stages | TPipe | level2 |
| group workspace | 16MiB 后 | `C*align(outSize*4,cacheLine)` | FP32 | stage1 | stage2 | executor | cacheLine |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  pre["preBuf"] post["postBuf+Muls"] res["result"] cache["16 KiB"]
```

##### 字节账与复用顺序

```text
UB_total=preBuf+postBuf+resultBlock+16KiB
workspace=16MiB+C*align(outSize*4,cacheLine)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| group partial | stage1 | stage2 | Kernel end | SyncAll | all producers arrived |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group mean | SyncAll + Var0 | cores/Host | stage2/Muls | GM partial | FP32，padding=0，scale once in post DAG |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | opbase reduce tiling | groupR/workspace | group selection |
| Entry | `mse_loss.cpp` | Reduction2 | MeanDag + Var0 |
| Body | opbase group scheduler | stage1/stage2 | barrier protocol |

### Route R-008：mean-A

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | opbase reduce tiling | mean + compressed A | A key | R-008 |
| Router | `mse_loss.cpp` | Reduction2 | MeanDag | A scheduler |

#### Device 数据流

##### 函数级调用链

```text
ReduceSch<A> -> square DAG -> ReduceSum -> Muls(meanVar) -> Cast/CopyOut
```

##### 源码等价伪码

```text
按 A pattern 切片执行；FP32 sum 乘 Host meanVar 后输出。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB/cache | reduce POD | basic/result | T/FP32 | Copy/DAG | output | executor/TPipe | A pattern |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  pre["preBuf"] post["postBuf+Muls"] res["result"] cache["16 KiB"]
```

##### 字节账与复用顺序

```text
UB_total=preBuf+postBuf+resultBlock+16KiB; workspace=16MiB reserve
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG slots | Copy/Reduce | Muls/Cast | next slice | level2 dependency | current output committed |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| A mean | pipe + Var0 | Host/Vector | post DAG | UB | FP32 scale；零 padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host | opbase reduce tiling | A key | pattern selection |
| Entry | `mse_loss.cpp` | Reduction2 | MeanDag |
| Body | opbase ReduceSch A + `mse_loss_dag.h` | A/MeanDag | schedule + scale |

## 算子特定优化

### Trick T-001：DAG 显式 FP32 promotion 与 LEVEL_2 内存复用

#### 触发与源码依据

- 适用 Route：R-002~R-008（R-001 的 Cast 在同类型实例中等价保留/消去由模板实现决定）。
- 触发条件：输入 FP16/BF16，或 sum/mean 使用归约 DAG。
- Golden 来源：[`mse_loss_dag.h:48-78`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_kernel/arch35/mse_loss_dag.h#L48-L78)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using OpCopyIn0Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;
using OpCopyIn1Cast = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn1>;
using OpSub = Bind<Vec::Sub<PromteT>, OpCopyIn0Cast, OpCopyIn1Cast>;
using OpMul = Bind<Vec::Mul<PromteT>, OpSub, OpSub>;
using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, OpMul>;
using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
```

#### 机制与设计原因

输入在 UB 内提升一次，平方差和归约保持 FP32；只有最终输出才窄化。归约 DAG 显式 LEVEL_2 让 ATVOSS 按依赖生命周期复用 pre/post buffer，Host 资源计算与 Device DAG 使用同一个 `OpDag` 类型。

#### 迁移边界

- 适用：低精度输入、可用 FP32 Vector 算术的逐元素+归约融合。
- 前提：Tiling 必须用同一 DAG 模板计算 buffer 统计；输出 rounding 语义保持 round mode 1。
- 失效：若中间精度/输出 dtype 或 DAG 节点顺序改变，必须重新实例化 Tiling 与 key。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

Host 用 FP16 DAG 算 UB、Device 却实例化 BF16/FP32 不同生命周期时，会破坏资源闭包；把 Cast 推到每次 ReduceSum 后会改变精度和 buffer 图。

### Trick T-002：AR group 的单 Kernel 两阶段归约

#### 触发与源码依据

- 适用 Route：R-004、R-007。
- 触发条件：通用 Reduce Tiling 生成 AR_GROUP（`groupR>1`）。
- Golden 来源：[`mse_loss_tiling_key.h:54-62`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_kernel/arch35/mse_loss_tiling_key.h#L54-L62)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
ASCENDC_TPL_ARGS_SEL(
    REDUCE_TPL_KEY_SEL_AR_GROUP(),
    ASCENDC_TPL_UINT_SEL(Reduction, ASCENDC_TPL_UI_LIST, MSELOSS_SUM, MSELOSS_MEAN),
    ASCENDC_TPL_UINT_SEL(Dtype, ASCENDC_TPL_UI_LIST, MSELOSS_TPL_FP32)),
```

#### 机制与设计原因

编译 key 同时覆盖 sum/mean 的 AR-group 实例；通用 scheduler 将每核第一阶段 partial 写到 cacheLine 对齐 workspace，barrier 后第二阶段归并。它保持一个 Kernel launch，避免额外 Host 节点和中间 Tensor 生命周期。

#### 迁移边界

- 适用：通用 Reduce key 能表达、输出 partial 可按 core 切片的 AIV 全归约。
- 前提：workspace 追加区、`groupR`、schedule mode 与 `SyncAll` 协议成套迁移。
- 失效：跨核通信协议不同或 partial 超过 workspace/UB 上限时需新路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只复制 AR_GROUP key 而不追加 per-core workspace，或把该路线标成两个 Host Kernel，都会与固定源码的 ABI/启动拓扑冲突。

## 模板与实现对比

| 对比维度 | R-001/002 Elementwise | R-003/006 AR normal | R-004/007 AR group | R-005/008 A | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | none | reduce,AR,groupR1 | reduce,AR,groupR>1 | compressed A | reduction→pattern→group |
| 模板与 dtype | ElementwiseSch；dtype key10/20/30 | ReduceSch Sum/Mean；key dtype20 | 同左 group key | ReduceSch A | binary DTYPE 仍决定 T/PromoteT |
| 循环与分核 | block/tile former-tail | A/R slice loop | 两阶段 group | A slice | 通用 Tiling producer不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 单 TBuf DAG slots | pre/post/result/cache | 同左+GM partial | pre/post/result/cache | 无 Cube/通信 window |
| 内存总占用与复用 | `Bdag*ubFormer*w` | symbolic reduce UB+16KiB | 同左+group WS | symbolic reduce UB | LEVEL_2 for reduce |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Vec | MoveAlign 或 NDDMA + Vec | 同左+SyncAll | 通用 A copy + Vec | `useNddma` 为 POD 子路径 |
| 同步、精度与 tail | buffer mutex；FP32中间 | pipe；FP32 reduce；zero pad | 加全核 barrier | pipe | public empty不入核 |
| fallback/失败边界 | UB/dtype失败 | group/A 或失败 | workspace失败 | Tiling失败 | 不补造其他 key |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，走 MSELossV2 | 支持，走 MSELossV2 | 支持，走 MseLoss/arch35 |
| Host/Launcher | 手写 V2 Tiling | 同 A2 | ATVOSS Elewise/Reduce Tiling |
| Kernel/资源 | 手写 TQue/TBuf；归约 64*C user WS | 同 A2 | DAG TBuf/Reduce cache；16MiB reserve；可选 NDDMA/group WS |

## 限制与失败边界

- 公开接口限制 rank 1~8、ND、受支持 dtype 和 reduction；Host 核心还要求两输入 storage shape/dtype 相同，依赖 ACLNN 预处理实现该前置条件。
- 空 Tensor 由 ACLNN return/Fill 路径处理，不进入本卡八条核心 Route；Kernel `Process(NAN)` 参数不能替代公开空域协议。
- Reduce POD 最多记录 9 维；公开输入 rank 最大 8，通用维度转换可增加内部维，但不得超过其固定数组上限。
- none 路线预留 16 MiB workspace 即使 Device elementwise body不使用用户 partial；不得擅自把 workspace size 改为 0。
- `Bdag`、PRE/POST buffer 计数是 pinned opbase 模板常量；本卡给出源码等价公式，不猜测某次编译实例展开后的整数或 ABI padding。
- 未处理 Out/Backward；未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/mse_loss/op_api/aclnn_mse_loss.h` | ACLNN declarations | 两阶段公开 API |
| 入口声明/约束 | `loss/mse_loss/docs/aclnnMseLoss.md` | 文档约束 | dtype/shape/platform |
| 入口实现 | `loss/mse_loss/op_api/aclnn_mse_loss.cpp` | checks/workspace/run | 预处理、空域、executor |
| 入口实现 | `loss/mse_loss/op_api/mse_loss.cpp` | `MseLoss`, `MseLossV1` | 950 version selector/launcher |
| 入口实现 | `loss/common/level2_base_loss.h` | common helpers | dtype/reduction/Fill |
| 构建接线 | `loss/mse_loss/CMakeLists.txt` | ascend950 | Host/Tiling 总接线 |
| Host/Tiling/Launcher | `loss/mse_loss/op_host/mse_loss_def.cpp` | `MseLoss` OpDef | ascend950 dtype/format |
| Host/Tiling/Launcher | `loss/mse_loss/op_host/arch35/mse_loss_tiling.cpp` | `TilingForMseLoss` | Tiling 注册入口 |
| Host/Tiling/Launcher | `loss/mse_loss/op_host/arch35/mse_loss_tiling_arch35.cpp` | `MseLossTiling` | checks/ele/reduce/key/workspace |
| Kernel entry/body | `loss/mse_loss/op_kernel/arch35/mse_loss_tiling_def.h` | `MseLossTilingData` | POD composition |
| Kernel entry/body | `loss/mse_loss/op_kernel/arch35/mse_loss_tiling_key.h` | MSELOSS template selection | compiled key set |
| Kernel entry/body | `loss/mse_loss/op_kernel/arch35/mse_loss_dag.h` | elementwise/sum/mean DAG | Vector node graph |
| Kernel entry/body | `loss/mse_loss/op_kernel/mse_loss.cpp` | `mse_loss` | entry/router |
| 构建接线 | `loss/mse_loss/op_kernel/CMakeLists.txt` | ascend950 kernel target | Kernel 编译接线 |
| 构建接线 | `cmake/third_party/opbase.cmake` | `OPBASE_TAG_ID` | 固定依赖 commit b8bd… |
| Host/Tiling/Launcher | opbase `elewise_base_struct.h` | `EleBaseTilingData` | elementwise POD |
| Host/Tiling/Launcher | opbase `elewise_tiling.h` | `ElewiseBaseTiling` | DAG 资源接口 |
| Host/Tiling/Launcher | opbase `elewise_tiling.cpp` | `ComputeParams/DoTiling` | core/UB/tail 公式 |
| Kernel entry/body | opbase `elewise_sch.h` | `ElementwiseSch` | TBuf/互斥/copy loop |
| Kernel entry/body | opbase `util/dag.h` | `DAGSch/MemOptCfg` | DAG 生命周期与 buffer 统计 |
| Kernel entry/body | opbase `util/vec.h` | `CopyIn/Cast/Sub/Mul/CopyOut` | Vector 与搬运 primitive |
| Kernel entry/body | opbase `util/sync.h` | buffer lock/unlock | MTE2/V/MTE3 slot 同步 |
| Host/Tiling/Launcher | opbase `reduce_tiling_data.h` | `ReduceOpTilingData` | reduce POD |
| Host/Tiling/Launcher | opbase `reduce_tiling_key_decl.h` | AR/A keys | pattern key ABI |
| Host/Tiling/Launcher | opbase `reduce_tiling.h` | `Tiling4ReduceOp` | reduce Host API |
| Host/Tiling/Launcher | opbase `reduce_tiling.cpp` | key/resource/workspace funcs | pattern/group/NDDMA/UB |
| Kernel entry/body | opbase `reduce_sch.h` | `ReduceSch` | normal/group/A router |
| Kernel entry/body | opbase `reduce_sch_aux.h` | `ReduceSchAux` | slice/搬运/归约 body |
| Kernel entry/body | opbase `reduce_sch_aux_base.h` | reduce aux base | UB 分区、循环公共状态 |
| Host/Tiling/Launcher | opbase `reduce_util.h` | dimension/reduce helpers | shape 压缩与辅助公式 |
| Kernel entry/body | opbase `reduce_operator.h` | `ReduceSumOp` | Vector reduction primitive |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | outer POD 同时容纳 elementwise/reduce 数据 | 成员顺序、数组长度9、inactive half 不可误读 | `MseLossTilingData` |
| 调用顺序 | ACLNN规范化→V1→Tiling→单 Kernel→Cast/View | empty 必须上层截断，meanVar 必须 entry 注入 | API / `mse_loss` |
| workspace/内存 | 16MiB reserve + optional aligned group partial | none 也保留16MiB；group offset/对齐不能改 | `SetTilingData`, reduce tiling |
| 同步 | TBuf slot events；group 使用 SyncAll | stage2 不得在所有 partial 前读 | ElementwiseSch/ReduceSch |
| 精度与尾块 | FP16/BF16 提升 FP32；reduction padding=0 | round mode、meanVar、DataCopyPad/NDDMA view 保持 | DAG/reduce aux |
| 失败边界 | dtype/shape/reduction/UB/key 均有失败终点 | 不新增未编译 key，不把 useNddma 当独立 launch | Host + key selection |
