---
type: Operator
title: "BitwiseXorScalar 950 ACLNN 算子知识"
description: "说明 aclnnBitwiseXorScalar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/docs/aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_api/aclnn_bitwise_xor_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_api/bitwise_xor.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_host/arch35/bitwise_xor_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_kernel/arch35/bitwise_xor_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_kernel/bitwise_xor_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_host/bitwise_xor_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_api/not_equal.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_host/arch35/not_equal_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_kernel/arch35/not_equal_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_kernel/not_equal_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/cast_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel_aicpu/cast_aicpu.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
created_at: "2026-08-25T10:21:15Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# BitwiseXorScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnBitwiseXorScalarGetWorkspaceSize` + `aclnnBitwiseXorScalar`，公开 ACLNN 两段式接口，无版本后缀 |
| 计算语义 | `out[i] = self[i] XOR other`；BOOL×BOOL 用逻辑异或，源码等价为 `NotEqual` |
| 输入输出 | `self/out` 同 shape、ND、至多 8 维；950 的 `self/other` 支持 BOOL 与 8 种有符号/无符号整数，输出还可为浮点、BF16、DOUBLE、COMPLEX64/128 等可转换类型 |
| 执行类型 | 核心节点为 AIV_VECTOR；scalar broadcast 的公共调度为 AIV mode 202；输出或隐式类型不同可插入 AIV Cast 或 AICPU Cast。无 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001 BOOL→BOOL；R-002 BOOL 核心+AIV Cast；R-003 BOOL 核心+AICPU Cast；R-004 整数核心直出；R-005 整数核心+AIV Cast；R-006 整数核心+AICPU Cast；R-007 BOOL 提升后进入整数核心（前/后 Cast 组合） |
| 核心资源 | 核心 AIV 使用 GM、隐式 UB tensor pool、MicroAPI RegTensor、16 MiB tiling workspace 请求；AICPU Cast 使用 CPU 线程分片和 GM tensor |
| 来源闭包 | source_closed：公开 ACLNN→参数推导→`NotEqual`/`BitwiseXor`/`Cast` launcher→Tiling/POD→AIV/AICPU body 与公共 ATVOSS scalar broadcast 均已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `P = GetPromoteDType(self, other)`。950 (`IsRegBase()==true`) 调用 `CombineCategories(self.dtype, other.dtype)`：`self` 非 BOOL 时 `P=self.dtype`；`self` 为 BOOL 时调用 `PromoteType(BOOL, other.dtype)`。`other` 在 Host 侧被物化成 dtype 为 `P` 的单元素 tensor。

- `P==BOOL`：执行 `NotEqual(self, otherTensor)`，即 BOOL 逻辑异或。
- `P!=BOOL`：先将 `self` Cast 到 `P`（相同 dtype 时 `Cast` 直接返回原 tensor），再执行逐元素整数 `BitwiseXor`。
- 最后将核心结果 Cast 到 `out.dtype`（相同 dtype 时不产生 Kernel），再经 `ViewCopy` 写入可能非连续的 `out`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；≤8 维；可非连续 | BOOL、INT8/16/32/64、UINT8/16/32/64 | 与 `out` shape 完全相同；空 tensor 直接成功，不创建计算节点 |
| `other` | Host scalar，无 layout | 同上 | 必须与 `self` 满足 TensorScalar 类型推导；转换为 dtype `P` 的单元素 tensor |
| `out` | ND；≤8 维；可非连续 | BOOL、INT8/16/32/64、UINT8/16/32/64、FLOAT、FLOAT16、DOUBLE、BFLOAT16、COMPLEX64/128 | `P` 必须能转换到 `out.dtype`；shape 等于 `self` |

### 入口与版本

固定提交只公开无后缀 `aclnnBitwiseXorScalar`，没有更新的版本族。本文不包含 `aclnnBitwiseXorTensor`，也不包含 `aclnnInplaceBitwiseXorScalar`；后者虽复用 `GetWorkspaceSizeCommon`，但其输出别名语义是独立逻辑算子。公开文档明确支持 Ascend 950PR/950DT；`BitwiseXor`、`NotEqual` 和 `Cast` 的 arch35 OpDef/CMake 构建接线共同证明 950 活动实现。

## Host/Launcher 执行流程

### 参数校验与前处理

`GetWorkspaceSizeCommon` 的顺序不可交换：创建 executor → 非空指针检查 → 950 dtype 列表检查 → 计算 `P` → 校验 `P→out.dtype` 可转换 → 校验 shape 相同和 `self` 维数≤8。空输入写 `workspaceSize=0` 并立即释放 executor。非空路径依次执行 `Contiguous(self)`、`ConvertToTensor(other,P)`；然后按 `P==BOOL` 选择 `NotEqual` 或 `Cast(self,P)→BitwiseXor`；最后 `Cast(coreOut,out.dtype)→ViewCopy(out)`。任一返回空指针立即返回内部错误，后续节点不入队。

### 参数、Workspace 与生命周期

`CREATE_EXECUTOR` 拥有连续化临时 tensor、scalar tensor、各 Cast/核心输出及 launcher 列表；第一段通过 `GetWorkspaceSize()` 返回框架聚合后的 workspace，并把 executor 转交调用者。第二段 `CommonOpExecutorRun(workspace,workspaceSize,executor,stream)` 按入队顺序提交节点。`BitwiseXor`/`NotEqual` 的 ATVOSS Tiling 各请求 `16*1024*1024` 字节默认 workspace，Kernel entry 的 `workspace` 参数未读取；AIV Cast 也请求 16 MiB。多节点时框架如何复用这些保留区属于 executor 壳边界，调用者必须使用第一段返回的聚合值，不得自行相加或缩小。

### 启动图与失败传播

```text
self --Contiguous-- contiguousSelf
other --ConvertToTensor(P)-- otherTensor

P == BOOL:
  [NotEqual AIV] -> boolCore
P != BOOL:
  [Cast self->P: no-op/AIV/AICPU] -> castSelf
  [BitwiseXor AIV] -> intCore

core --[Cast core->out.dtype: no-op/AIV/AICPU]-- castOut
castOut --ViewCopy(stream ordered)-- out
```

| 边 | offset/extent/dtype/layout | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|
| `contiguousSelf→core/preCast` | 独立 GM tensor；`N` 元素；self dtype；ND | 同 executor/stream 前序节点完成后可见 | executor 持有到执行结束 | 前序失败阻断核心 |
| `otherTensor→NotEqual/BitwiseXor` | 单元素 GM tensor；dtype `P`；broadcast | 同 stream | executor | 转换失败不入队核心 |
| `preCast→BitwiseXor` | `N` 元素；dtype `P`；ND | 同 stream | executor | Cast 失败阻断 BitwiseXor |
| `core→postCast/ViewCopy` | `N` 元素；核心 dtype；ND | 同 stream | executor | 核心失败阻断后续 |
| `postCast→ViewCopy` | `N` 元素；`out.dtype`；ND | 同 stream | executor，ViewCopy 后可回收 | Cast 失败阻断写出 |

## Route 与参数设计

### 分核与资源计算

`Contiguous` 后，`self` 的所有非 1 轴共享“self 不广播、other 广播”标志，维度折叠为一维 `N`；全 1 轴被合并。因此 `InputSize+OutputSize=3≤8` 且 DAG `VarSize=0`，固定选择 mode 202 `BroadcastOneDimAdvanceSch`。对核心 DAG：

```text
alive = BitwiseXor 6；NotEqual 9
tile0 = floor((UB_bytes/alive)/512)*512 / max_dtype_bytes
若 coreNum*(512/min_dtype_bytes) > N: tile = 512/min_dtype_bytes
否则若 coreNum*tile0 > N: tile = floor((N/coreNum)/(512/max_dtype_bytes))*(512/max_dtype_bytes)
否则 tile = tile0
ubOuter = ceil(N/tile)
blockFormer = ceil(ubOuter/coreNum)，若为 1 则改为 2（核内 ping-pong）
blockNum = ceil(ubOuter/blockFormer)
```

最后一核由 Kernel 重算 `blockTail=ubOuter-(blockNum-1)*blockFormer` 和 `ubTail=N-(ubOuter-1)*tile`。每核 GM 基址为 `blockFormer*tile*blockIdx*sizeof(T)`。

### Route 选择

记 `C(s,d)` 为 `(s,d)` 命中 950 `REGBASE_AICORE_DTYPE_SUPPORT_LIST`；`s==d` 的 Cast 是 Host no-op。

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 BOOL 直出 | NotEqual mode 202 | `P==BOOL && out.dtype==BOOL` | R-002/R-003 |
| 2 | R-002 BOOL+AIV Cast | NotEqual 202 + Cast key 0 | `P==BOOL && out.dtype!=BOOL && C(BOOL,out.dtype)` | R-003 |
| 3 | R-003 BOOL+AICPU Cast | NotEqual 202 + AICPU Cast | `P==BOOL && out.dtype!=BOOL && !C(BOOL,out.dtype)`，且 Cast AICPU dtype 表支持 | 失败 |
| 4 | R-004 整数直出 | BitwiseXor 202 | `P!=BOOL && self.dtype==P && out.dtype==P` | R-005/R-006 |
| 5 | R-005 整数+AIV Cast | BitwiseXor 202 + Cast key 0 | `P!=BOOL && self.dtype==P && out.dtype!=P && C(P,out.dtype)` | R-006 |
| 6 | R-006 整数+AICPU Cast | BitwiseXor 202 + AICPU Cast | `P!=BOOL && self.dtype==P && out.dtype!=P && !C(P,out.dtype)` | 失败 |
| 7 | R-007 BOOL 提升整数 | preCast + BitwiseXor 202 + 可选 postCast | `self.dtype==BOOL && P!=BOOL`；每个 Cast 独立按同 dtype/no-op、`C`、AICPU 顺序选择 | Cast/BitwiseXor 失败 |

```text
empty(self) -> success(workspace=0)
P = CombineCategories(self.dtype, other.dtype)
if P == BOOL:
  NotEqual(mode=202)
  if out.dtype != BOOL: Cast(BOOL,out.dtype)
else:
  if self.dtype != P: Cast(self.dtype,P)
  BitwiseXor(mode=202)
  if out.dtype != P: Cast(P,out.dtype)
ViewCopy -> out
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 202 | 950、非空、BOOL×BOOL、out BOOL | 单 NotEqual AIV | `BroadcastOneDimAdvanceSch<NotEqualCompute<int8_t>::OpDag>` | 9×tile 字节隐式 UB pool；stream | mask 保护尾 tile |
| R-002 | 202→0 | R-001 核心且 `C(BOOL,out)` | NotEqual AIV→Cast AIV | NotEqual DAG；Cast policy family | 两个串行节点，各自 UB；GM 中间边 | 各节点独立 tail |
| R-003 | 202→AICPU | R-001 核心且 `!C(BOOL,out)` | NotEqual AIV→Cast AICPU | NotEqual DAG；`CastCpuKernel::CastTask` | AIV UB→GM→CPU 分片 | CPU `[start,end)` 尾天然闭合 |
| R-004 | 202 | 950、非空、`P=self.dtype!=BOOL`、out=P | 单 BitwiseXor AIV | `BroadcastOneDimAdvanceSch<BitwiseXorCompute<P>::OpDag>` | 6×tile×sizeof(P) UB；stream | MicroAPI mask |
| R-005 | 202→0 | R-004 核心且 `C(P,out)` | BitwiseXor AIV→Cast AIV | XOR DAG；Cast policy family | 两个串行 AIV 节点 | 各节点独立 tail |
| R-006 | 202→AICPU | R-004 核心且 `!C(P,out)` | BitwiseXor AIV→Cast AICPU | XOR DAG；`CastTask<P,out>` | AIV UB→GM→CPU 分片 | CPU 分片尾 |
| R-007 | Cast selector→202→Cast selector | `self=BOOL && P!=BOOL` | preCast(AIV/AICPU)→BitwiseXor AIV→postCast(no-op/AIV/AICPU) | Cast policy/CPU + XOR DAG | 串行 GM 边；executor 持有全部临时 | 任一节点失败阻断后续 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域 | `self->IsEmpty()` | 禁止一维 Tiling 的 `dimLength==0` | 全部 | Host 直接成功 | `GetWorkspaceSizeCommon` |
| 单轴 Advance | 合轴 rank=1、连续、3 个 GM 参数≤8、`VarSize=0` | 使用寄存器/隐式 UB pool 的快速调度 | 核心节点 | 条件理论上由 scalar 形态保证；Tiling 失败 | `BroadcastBaseTiling::DoTiling` |
| UB 512B 对齐 | `tileNum>0` | ping-pong pool 与 Vector 访问粒度 | R-001~R-007 核心 | `GRAPH_FAILED` | `GetOneDimUBFormerAdvance` |
| dtype 同构 | `x1.dtype==x2.dtype==y.dtype` | XOR DAG 模板只有单一 `T` | BitwiseXor | `GRAPH_FAILED` | `BitwiseXorTiling::DoOpTiling` |
| AIV Cast pair | `(src,dst)∈REGBASE_AICORE_DTYPE_SUPPORT_LIST` | arch35 policy 必须存在 | R-002/R-005/R-007 | AICPU Cast | `IsAiCoreSupport` |
| Cast SIMT | 仅 `INT64→DOUBLE` 且 AIV pair | Cast 为该对保留 32 KiB local memory | R-005/R-007 | 非该对走普通 Vector；不触发 CV | `CastTiling::IsSimt` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen` | int64/64 | sizeof=16, alignof=8, offset=0 | 元素 | `DoOneDimFormerTilingAdvance: dims.back()[0]` | `Init/Process` 计算 ubOuter/tail | `>0` |
| 同上 | `tileNum` | int32/32 | offset=8 | 元素/tile | `GetOneDimUBFormerAdvance` 后按核数修正 | `blockLen`、循环步长 | `>0`、512B 粒度 |
| 同上 | `blockNum` | int16/16 | offset=12 | AIV core | `DoOneDimFormerTilingAdvance`（从 uint64 窄化） | 尾核判断 | 必须容纳实际核数 |
| 同上 | `scalarFlag` | int16/16 | offset=14 | 输入位图 | `Σ((dims[i][0]==1)<<i)` | `CopyIn` 判断 scalar cache | 常规 self 非标量时值 2 |
| `CastTilingData` | `blockNum` | int64/64 | sizeof=96, alignof=8, offset=0 | core | `CastTiling::DoOpTiling` | Cast body 分核 | `1..AIV coreNum` |
| 同上 | `ubFormer` | int64/64 | offset=8 | 元素 | `GetUbFormer` | 每次 CopyIn/Compute/CopyOut | UB 上限 |
| 同上 | `blockFormer` | int64/64 | offset=16 | 元素/core | 8 元素对齐的 `ceil(N/core)` | GM core offset | 尾核另取余量 |
| 同上 | `ubLoopOfFormerBlock` | int64/64 | offset=24 | loop | `ceil(blockFormer/ubFormer)` | 非尾核循环 | ≥1 |
| 同上 | `ubLoopOfTailBlock` | int64/64 | offset=32 | loop | `ceil(blockTail/ubFormer)` | 尾核循环 | ≥1 |
| 同上 | `ubTailOfFormerBlock` | int64/64 | offset=40 | 元素 | Host 余数公式 | 非尾核末 loop | 有效长度 |
| 同上 | `ubTailOfTailBlock` | int64/64 | offset=48 | 元素 | Host 余数公式 | 尾核末 loop | 有效长度 |
| 同上 | `regCopyInStep` | int64/64 | offset=56 | 元素/register step | `GetUbCopyStep` | `CastMicro` load | policy 决定 |
| 同上 | `regCopyOutStep` | int64/64 | offset=64 | 元素/register step | `GetUbCopyStep` | `CastMicro` store | policy 决定 |
| 同上 | `ubFormerRegLoop` | int64/64 | offset=72 | register loop | Host 由 VL/输入 bit 数推导 | 整 tile MicroAPI 循环 | 部分 policy 为0 |
| 同上 | `ubTailOfFormerRegLoop` | int64/64 | offset=80 | register loop | Host tail 推导 | 非尾核末 tile | 同上 |
| 同上 | `ubTailOfTailRegLoop` | int64/64 | offset=88 | register loop | Host tail 推导 | 尾核末 tile | 同上 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001~R-003 | `P==BOOL`；202；Cast selector | scalar API + NotEqual Tiling + Cast selector | broadcast POD；可选 Cast POD/AICPU attrs | `not_equal`；可选 `cast`/AICPU Cast | NotEqual DAG + ATVOSS；Cast impl/CPU | not_equal/cast CMake+OpDef ascend950 |
| R-004~R-006 | `P!=BOOL,self=P`；202；Cast selector | scalar API + BitwiseXor Tiling + Cast selector | broadcast POD；可选 Cast POD/AICPU attrs | `bitwise_xor`；可选 `cast`/AICPU Cast | XOR DAG + ATVOSS；Cast impl/CPU | bitwise_xor/cast CMake+OpDef ascend950 |
| R-007 | preCast selector→202→postCast selector | 同上三类 producer | 串行各节点独立 POD | Cast→`bitwise_xor`→Cast | 全部直接依赖 | 全部 ascend950 接线 |

## Kernel 实现

### 公共执行流程

核心 `bitwise_xor`/`not_equal` entry 都构造 `BroadcastSch<schMode,OpDag>`；mode 202 解析 16B POD，调用 `InitSocState` 和 `BroadcastOneDimAdvanceSch`。scheduler 按 `blockIdx` 绑定 GM；每核循环 tile，用 `i&1` 在 ping/pong buffer IDs 间切换。非 scalar 输入执行 `DataCopyPad`，scalar 输入只在前两个 ping/pong 迭代各读 GM[0] 一次并 `Duplicate`，随后复用。DAG 按拓扑调用计算函数，最终 `DataCopyPad` 写有效 `tileLength`。

### Route R-001：BOOL NotEqual 直出

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_bitwise_xor_scalar.cpp` | `GetWorkspaceSizeCommon` | `promoteType==DT_BOOL` | `l0op::NotEqual` |
| Host | `not_equal.cpp` | `IsAiCoreSupport` | 950 BOOL 命中 | AIV launcher |
| Tiling | `not_equal_tiling_arch35.cpp` | `DoOpTiling` | BOOL 使用 `NotEqualCompute<int8_t>` | mode 202 |
| Router | `not_equal_apt.cpp` | `not_equal` | `DTYPE_X1==bool` | int8 DAG body |

#### Device 数据流

##### 函数级调用链

```text
NotEqualAiCore -> not_equal<202>
  -> BroadcastOneDimAdvanceSch::Init/Process
    -> CopyInBrc<int8_t>(self)
    -> CopyInBrc<int8_t>(other scalar cache)
    -> Duplicate<uint8_t>(1), Duplicate<uint8_t>(0)
    -> Compare<uint8_t,int8_t,5>
    -> Select<uint8_t,uint8_t,2>
    -> CopyOut<uint8_t>(boolCore)
```

##### 源码等价伪码

```text
for tile on this block (ping = tileIndex & 1):
  x = DataCopyPad(self[coreBase+tileOffset], len)
  if first use of ping: s[ping] = Duplicate(load(other[0]), len)
  one = Duplicate(1, len); zero = Duplicate(0, len)
  mask = Compare(x, s[ping], CMPMODE=5, len)
  y = Select(mask, one, zero, SELMODE=2, len)
  DataCopyPad(boolCore[coreBase+tileOffset], y, len)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| self GM | `coreBase+offset` | `len` | int8 view of BOOL/ND | DataCopyPad | Compare | caller/executor | tail按len |
| other GM | 0 | 1 | int8 scalar | GM GetValue | Duplicate | executor | 每个 ping 首次一次 |
| UB pool | `bufId*tileNum` | `9*tileNum` bytes | mixed int8/uint8 | scheduler | DAG | per core | 4 input ping/pong+3 temp+2 output ping/pong |
| output GM | `coreBase+offset` | `len` | BOOL/ND | CopyOut | postCast/ViewCopy | executor | 只写有效len |

##### 分层内存布局

```mermaid
block-beta
  columns 9
  xPing["x ping · tile B"] xPong["x pong · tile B"]
  sPing["scalar ping · tile B"] sPong["scalar pong · tile B"]
  cmp["compare · tile B"] one["one · tile B"] zero["zero · tile B"]
  yPing["out ping · tile B"] yPong["out pong · tile B"]
```

##### 字节账与复用顺序

```text
UB_total = 9 * tileNum bytes <= UB_bytes
dynamic_ub = N/A
workspace_node = 16 MiB reserved；entry 不读取 workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| scalar ping/pong | 前两个 `CopyIn` | 每轮 Compare | 同 ping 不再覆盖 | buffer ID 与顺序执行 | scalar 全程不变 |
| temp/output IDs | DAG node | 后继 node/CopyOut | 生命周期结束后的 node | `GetTensor/ReleaseTensor` | 拓扑消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | `GetTensor/ReleaseTensor` | MTE2 | Vector DAG | UB buffer ID | BOOL按int8字节比较 |
| Compute→CopyOut | 同上 | Select | MTE3 | output ID | `len` 控制末 tile |
| 跨节点 | executor stream | NotEqual | Cast/ViewCopy | GM intermediate | 无跨核归约/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/not_equal/op_host/arch35/not_equal_tiling_arch35.cpp` | `NotEqualTiling::DoOpTiling` | BOOL→int8 DAG→mode 202 |
| Kernel entry/router | `math/not_equal/op_kernel/not_equal_apt.cpp` | `not_equal<202>` | bool specialization |
| Device body/helper | `math/not_equal/op_kernel/arch35/not_equal_dag.h` | `NotEqualCompute<int8_t>` | Compare+Select |

### Route R-002：BOOL 核心后 AIV Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `cast.cpp` | `Cast` | dtype不同且 `IsAiCoreSupport(BOOL,out)` | `CastAiCore` |
| Tiling | `cast_tiling.cpp` | `GetShapeAttrsInfo/DoOpTiling` | pair 命中 `castMap` | key 0、AIV blockNum |

#### Device 数据流

##### 函数级调用链

```text
R-001 NotEqual -> boolCore(GM)
CastAiCore -> cast<0> -> GetCastPolicy<BOOL,out>::body -> castOut(GM)
ViewCopy -> out
```

##### 源码等价伪码

```text
run R-001; stream-order GM edge
for each Cast core/tile: CopyIn(boolCore) -> policy cast/pack -> CopyOut(castOut, validLen)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boolCore GM | 0 | N | BOOL/ND | NotEqual | Cast | executor | stream edge |
| Cast UB | core/tile policy | `ubFormer` 双缓冲输入/输出及可选中间 | pair-specific | CopyIn | Cast body/CopyOut | Cast core | Host公式保证UB上界 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inPing["Cast input ping"] inPong["Cast input pong"] outPing["Cast output ping"] outPong["Cast output pong"]
```

##### 字节账与复用顺序

```text
普通/Through/Micro: UB_total <= UB_bytes-128B reserve
DST_BOOL: UB_total <= UB_bytes-192B（本 Route 目的非BOOL，不使用）
workspace_node = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Cast ping/pong | CopyIn/Compute | Compute/CopyOut | 同 parity 后续 tile | pipe queue/event | 前一消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast | policy `RoundMode`/Micro load-store dist | input | output | UB/RegTensor | pair 的 `cast_struct.h` 明确 round/pack；POD 给有效尾长 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling` | pair、UB、block、tail |
| Kernel entry/router | `math/cast/op_kernel/cast_apt.cpp` | `cast<0>` | policy→body |
| Device body/helper | `math/cast/op_kernel/arch35/cast_impl.h` | `CastDirect/Through/Two/Micro` | CopyIn/Compute/CopyOut |

### Route R-003：BOOL 核心后 AICPU Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `cast.cpp` | `Cast` | dtype不同、AIV pair不支持、AICPU两端dtype支持 | `CastAiCpu` |
| CPU | `cast_aicpu.cpp` | `SetBoolMap` | BOOL→目标存在 | `CastTask<bool,D>` |

#### Device 数据流

##### 函数级调用链

```text
R-001 NotEqual -> boolCore(GM)
CastAiCpu -> CastCpuKernel::Compute -> Validate -> ParallelFor -> CastTask<bool,D>
```

##### 源码等价伪码

```text
threads partition [0,N)
for i in [start,end): out[i] = static_cast<D>(boolCore[i])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boolCore/out GM | `i` | N | BOOL→D | AIV/CPU thread | ViewCopy | executor | 无AIV UB |

##### 分层内存布局

N/A：AICPU 节点直接对 GM tensor 做 CPU 区间分片，不使用 AIV UB。

##### 字节账与复用顺序

```text
UB_total = N/A
workspace_total = AICPU task metadata 由 executor 壳管理，固定算子源码无独立 slice 公式
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM output interval | CPU worker | ViewCopy | 执行结束后 | `ParallelFor` join + stream task完成 | worker区间不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU Cast | `ParallelFor` | workers | Compute return | `[start,end)` | C++ `static_cast`; end独占自然处理尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cast/op_api/cast.cpp` | `CastAiCpu` | AICPU fallback |
| Kernel entry/router | `math/cast/op_kernel_aicpu/cast_aicpu.cpp` | `CastCpuKernel::SetMap` | dtype函数表 |
| Device body/helper | 同上 | `CastTask`/`ExecuteCast` | 区间并行 static_cast |

### Route R-004：整数 BitwiseXor 直出

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_bitwise_xor_scalar.cpp` | `GetWorkspaceSizeCommon` | `P!=BOOL` | `BitwiseXor` |
| Host | `bitwise_xor.cpp` | `IsAiCoreSupport` | 950 8 种整数 | AIV |
| Tiling | `bitwise_xor_tiling.cpp` | `DoOpTiling` | dtype同构 | 对应 `BitwiseXorCompute<T>`、202 |

#### Device 数据流

##### 函数级调用链

```text
BitwiseXorAiCore -> bitwise_xor<202>
  -> BroadcastOneDimAdvanceSch::Process
    -> CopyInBrc<T>(self), CopyInBrc<T>(other scalar cache)
    -> XorCompute<T> -> CopyOut<T>
```

##### 源码等价伪码

```text
for each tile:
  a = CopyIn(self); b = cached Duplicate(other[0])
  for j in [0,ceil(len/VL_T)):
    preg = UpdateMask<T>(remaining)
    ra=load(a+j*VL_T); rb=load(b+j*VL_T)
    store(y+j*VL_T, Xor(ra,rb,preg), preg)
  CopyOut(valid len)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| self/output GM | `coreBase+offset` | len | T/ND | CopyIn/Xor | CopyOut | executor | tail len |
| other GM | 0 | 1 | T scalar | GetValue | Duplicate | executor | ping/pong cache |
| UB pool | `bufId*tile*sizeof(T)` | `6*tile*sizeof(T)` | T | scheduler | Xor | per core | 4 input +2 output buffers |
| registers | `j*VL_T` | 3 RegTensor<T> + MaskReg | T | MicroAPI load/Xor | store | vector scope | masked tail |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  xPing["x ping · tile*sizeof(T)"] xPong["x pong"] sPing["scalar ping"] sPong["scalar pong"] yPing["out ping"] yPong["out pong"]
```

##### 字节账与复用顺序

```text
UB_total = 6 * tileNum * sizeof(T) <= UB_bytes
dynamic_ub = N/A
workspace_node = 16 MiB reserved
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output ping-pong | CopyIn/Xor | Xor/CopyOut | 同 parity 后续 tile | buffer ID + Get/Release | 前 tile消费完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Vector | `UpdateMask<T>` | count | Xor/load/store | 最后寄存器向量 | 整数按位，无 cast/round；mask阻止无效lane写出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/bitwise_xor/op_host/arch35/bitwise_xor_tiling.cpp` | `BitwiseXorTiling::DoOpTiling` | dtype→DAG→202 |
| Kernel entry/router | `math/bitwise_xor/op_kernel/bitwise_xor_apt.cpp` | `bitwise_xor<202>` | Broadcast scheduler |
| Device body/helper | `math/bitwise_xor/op_kernel/arch35/bitwise_xor_dag.h` | `XorCompute<T>` | masked MicroAPI XOR |

### Route R-005：整数核心后 AIV Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | scalar API | post `Cast` | `out.dtype!=P` | Cast selector |
| Host | `cast.cpp` | `IsAiCoreSupport` | `C(P,out)` | AIV Cast |

#### Device 数据流

##### 函数级调用链

```text
R-004 BitwiseXor -> intCore(GM) -> R-002 的 Cast AIV 公共实现 -> ViewCopy
```

##### 源码等价伪码

```text
run masked XOR; stream-order; run pair-specific Cast tile pipeline; write out dtype
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| intCore GM | 0 | N | P/ND | XOR | Cast | executor | 节点间GM边 |
| Cast UB/Reg | POD core/tile | policy公式 | P→out | Cast | CopyOut | Cast core | ping-pong |

##### 分层内存布局

与 R-002 Cast AIV 布局相同；前置 XOR 的 6-buffer pool 在其 Kernel 结束后失效，不跨 Kernel 复用。

##### 字节账与复用顺序

```text
XOR UB = 6*tileXor*sizeof(P)
Cast UB <= Cast Host 公式上界
workspace nodes each request 16 MiB; aggregate由executor返回
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM intCore | XOR | Cast | Cast完成后 | stream order | 不提前释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| XOR→Cast | stream | XOR | Cast | intCore GM | XOR精确；Cast round由policy决定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | scalar API + `cast_tiling.cpp` | postCast/CastTiling | 串行参数生产 |
| Kernel entry/router | XOR entry + `cast<0>` | 两节点 | AIV→AIV |
| Device body/helper | XOR DAG + Cast impl | policy body | 逐节点闭合 |

### Route R-006：整数核心后 AICPU Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `cast.cpp` | `Cast` | `out!=P && !C(P,out)` 且 AICPU支持 | AICPU |
| CPU | `cast_aicpu.cpp` | `SetInt*/SetUInt*Map` | pair存在 | `CastTask<P,D>` |

#### Device 数据流

##### 函数级调用链

```text
R-004 BitwiseXor -> intCore(GM) -> CastCpuKernel::ParallelFor -> CastTask<P,D> -> ViewCopy
```

##### 源码等价伪码

```text
parallel ranges: for i=start..end-1 out[i]=static_cast<D>(intCore[i])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| intCore/out GM | i | N | P→D | AIV/CPU | ViewCopy | executor | worker区间不重叠 |

##### 分层内存布局

N/A：Cast 节点为 AICPU；XOR 节点布局见 R-004。

##### 字节账与复用顺序

```text
XOR UB = 6*tile*sizeof(P); AICPU UB=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| intCore GM | XOR | CPU workers | join后 | stream + ParallelFor join | N元素有效 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| AIV→AICPU | executor ordering | XOR | CPU task | GM | CPU static_cast；区间end处理尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `cast.cpp` | `CastAiCpu` | fallback |
| Kernel entry/router | `cast_aicpu.cpp` | `SetMap` | integer source函数表 |
| Device body/helper | 同上 | `CastTask/ExecuteCast` | CPU并行转换 |

### Route R-007：BOOL 提升后进入整数核心

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | scalar API | `GetPromoteDType` | `self==BOOL` 且 `P!=BOOL` | preCast→BitwiseXor |
| Cast | `cast.cpp` | `Cast` | BOOL→P | AIV或AICPU |
| Core | `bitwise_xor.cpp` | `BitwiseXor` | P为整数 | AIV |

#### Device 数据流

##### 函数级调用链

```text
Cast(BOOL,P) [AIV/AICPU] -> castSelf(GM)
BitwiseXor<202>(castSelf, otherTensor) -> intCore(GM)
Cast(P,out) [no-op/AIV/AICPU] -> ViewCopy
```

##### 源码等价伪码

```text
cast each BOOL self element to P
xor with scalar other already materialized as P
optionally cast integer result to out dtype
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| castSelf | 0 | N | P/ND | preCast | XOR | executor | stream edge |
| intCore | 0 | N | P/ND | XOR | postCast/ViewCopy | executor | stream edge |
| node local | 各自POD | 各自公式 | pair-specific | node | node | node lifetime | 不跨Kernel alias |

##### 分层内存布局

pre/post AIV Cast 使用 R-002 布局；AICPU Cast 为 N/A；中间 XOR 使用 R-004 的 6-buffer 布局。

##### 字节账与复用顺序

```text
每个AIV Cast: UB<=其Host公式上界
XOR: UB=6*tile*sizeof(P)
AICPU Cast: UB=N/A
workspace aggregate=第一段接口返回值
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| castSelf/intCore GM | 前节点 | 后节点 | 后节点完成后 | executor stream | 失败阻断后续 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| preCast→XOR→postCast | stream/task join | 每节点 | 下一节点 | GM intermediates | Cast policy/static_cast；XOR本身无精度损失；各节点真实len |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | scalar API | `CombineCategories/GetWorkspaceSizeCommon` | BOOL混合整数分支 |
| Kernel entry/router | Cast/BitwiseXor entries | selector→202→selector | 多节点拓扑 |
| Device body/helper | Cast impl/CPU + XOR DAG | pair policy + `XorCompute<P>` | 全链实现 |

## 算子特定优化

### Trick T-001：标量只填充一次 ping/pong UB，后续 tile 复用

#### 触发与源码依据

- 适用 Route：R-001~R-007 的 `NotEqual`/`BitwiseXor` scalar 输入。
- 触发条件：mode 202 且 `scalarFlag` 对应输入位为 1。
- Golden 来源：[`broadcast_sch_one_dim_advance.h:201-215`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h#L201-L215)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bool isScalar = tilingData->scalarFlag & inputIdx;
if (isScalar) {
    if (isFirstScalarTime) {
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ inputType*>(inGm[input::Pos].GetPhyAddr(0)));
        inputType scalar = globalTensor.GetValue(0);
        GetTensor<TPosition::VECCALC>(bufId);
        Vec::Duplicate(inTensor, scalar, tileLength);
        ReleaseTensor<TPosition::VECCALC>(bufId);
    }
}
```

#### 机制与设计原因

`Process` 用 `i&1` 交替 ping/pong，并在执行过两个整 tile 后清除 `isFirstScalarTime`。因此 GM scalar 只需为两个 parity buffer 各读取和广播一次；之后所有 tile 复用已填好的 UB，不随 `N` 重复搬入标量。该机制减少 GM scalar 读取与 Duplicate 次数，同时要求 scalar 在整个 Kernel 生命周期不变。

#### 迁移边界

- 适用：只读标量广播、双缓冲 tile 循环。
- 前提：标量值全 Kernel 不变；两套 parity buffer 均在停止填充前初始化。
- 失效：标量随 tile/core 变化时必须每轮重填；普通 tensor 输入继续走 DataCopyPad。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在仅初始化 ping buffer 后就停止填充，否则第二个 tile 的 pong buffer 未定义。固定仓未提供其它可验证反例。

### Trick T-002：MicroAPI 掩码直接完成任意整数宽度 XOR 尾块

#### 触发与源码依据

- 适用 Route：R-004~R-007 的 BitwiseXor 核心。
- 触发条件：每个 tile 的 `count` 可能不是 `VL_T=GetVRegSize()/sizeof(T)` 的整数倍。
- Golden 来源：[`bitwise_xor_dag.h:36-59`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_kernel/arch35/bitwise_xor_dag.h#L36-L59)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr uint32_t VL_T = VECTOR_LENGTH / sizeof(T);
uint16_t loopTimes = CeilDiv(count, VL_T);
uint32_t sregMask = count;
for (uint16_t j = 0; j < loopTimes; j++) {
    preg = MicroAPI::UpdateMask<T>(sregMask);
    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(src1Value, inputX1Addr + VL_T * j);
    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(src2Value, inputX2Addr + VL_T * j);
    MicroAPI::Xor(resValue, src1Value, src2Value, preg);
    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
}
```

#### 机制与设计原因

`VL_T` 随 dtype 字节数调整，每次 `UpdateMask` 从剩余 `count` 产生 predicate；同一 predicate 同时约束 XOR 和 store。这样整 tile 与尾 tile 共用一个 body，不需额外 padding 参与运算，也不会将无效 lane 写回 GM。

#### 迁移边界

- 适用：逐元素、无跨 lane 依赖的整数位运算。
- 前提：load/store distribution 都是 `DIST_NORM`，地址按 `VL_T*j` 递增。
- 失效：跨 lane 算法必须另外处理无效 lane；dtype 必须有对应 MicroAPI Xor 支持。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

省略 store predicate 会把最后向量的无效 lane 写到有效输出范围之外。固定仓未提供可验证性能反例。

## 模板与实现对比

| 对比维度 | R-001 BOOL | R-004 整数 | R-002/R-005 AIV Cast | R-003/R-006 AICPU Cast | R-007 混合 | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | P=BOOL,out=BOOL | P非BOOL,out=P | 输出转换命中950 pair | pair不命中AIV但CPU支持 | self BOOL,P非BOOL | Host固定顺序 |
| 模板与 dtype | NotEqual<int8> | Xor<T> 8种整数 | pair policy | C++模板函数表 | Cast+Xor+Cast | 语义/转换能力 |
| 循环与分核 | mode202 | mode202 | Cast POD | CPU ParallelFor | 各节点独立 | 节点资源模型 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 9-buffer+Compare/Select | 6-buffer+3 RegTensor | 双缓冲+RegTensor；INT64→DOUBLE含SIMT保留 | 无AIV UB | 组合 | dtype转换复杂度 |
| 内存总占用与复用 | 9×tile B | 6×tile×sizeof(T) | Host UB公式 | GM区间 | 节点间GM | DAG活跃节点 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Compare/Select | DataCopyPad/Micro Xor | Cast/Micro pack；特例SIMT | static_cast | 组合 | pair policy |
| 同步、精度与 tail | buffer ID+len | predicate mask | pipe+round mode | ParallelFor join | stream | 精确尾长 |
| fallback/失败边界 | Cast/失败 | Cast/失败 | AICPU | 不支持则失败 | 任一节点失败 | 源码 selector |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 API 文档支持，但固定提交无 `BitwiseXor`/`NotEqual` A2 构建与 Kernel 闭包 | 公开 API 文档支持，但固定提交无 A3 构建与 Kernel 闭包 | 支持且源码闭合 |
| Host/Launcher | scalar ACLNN 源码共用；核心实现固定仓缺失 | 同左 | `IsRegBase` 类型推导与 950 selector |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | arch35 AIV mode202、可选 AIV/AICPU Cast |

## 限制与失败边界

- 任一 `self/other/out` 空指针、dtype 不支持/不可推导/不可转换、shape 不同或维数>8，第一段失败。
- 空 tensor 是 Host 成功短路，不能送入要求 `dimLength>0` 的 mode 202 Tiling。
- Cast AIV pair 必须同时存在于 L0 selector 表和 arch35 `castMap`；否则回退 AICPU。AICPU 表也不支持时返回空指针并阻断后续。
- `BroadcastOneDimTilingDataAdvance::blockNum` 从计算值写入 int16；固定源码未增加独立溢出检查，复用时不得扩大到超出 int16 可表示核数。
- executor 对多个 16 MiB 节点 workspace 请求的聚合/复用实现属于固定算子源码外的框架壳；只能使用 API 返回值。
- A2/A3 虽有公开支持声明，但固定提交 `BitwiseXor` 与 `NotEqual` 只接线 `ascend950`，故未生成无后缀卡。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/bitwise_xor/docs/aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar.md` | 产品/参数表 | 公开 API、950 支持、dtype/shape |
| 入口实现 | `math/bitwise_xor/op_api/aclnn_bitwise_xor_scalar.cpp` | `GetWorkspaceSizeCommon` | 校验、P、launch graph |
| L0 selector | `math/bitwise_xor/op_api/bitwise_xor.cpp` | `BitwiseXor/IsAiCoreSupport` | AIV/AICPU选择 |
| Host/Tiling | `math/bitwise_xor/op_host/arch35/bitwise_xor_tiling.cpp` | `BitwiseXorTiling` | dtype、202 producer |
| Kernel entry/body | `math/bitwise_xor/op_kernel/bitwise_xor_apt.cpp` | `bitwise_xor` | entry |
| Kernel entry/body | `math/bitwise_xor/op_kernel/arch35/bitwise_xor_dag.h` | `XorCompute` | MicroAPI body |
| 构建接线 | `math/bitwise_xor/op_host/bitwise_xor_def.cpp` | `BitwiseXor` OpDef | ascend950 |
| 构建接线 | `math/bitwise_xor/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950/arch35 |
| 直接依赖 | `math/not_equal/op_api/not_equal.cpp` | `NotEqual` | BOOL selector |
| 直接依赖 | `math/not_equal/op_host/arch35/not_equal_tiling_arch35.cpp` | `NotEqualTiling` | BOOL mode202 |
| 直接依赖 | `math/not_equal/op_kernel/arch35/not_equal_dag.h` | `NotEqualCompute` | Compare/Select |
| 直接依赖 | `math/not_equal/op_kernel/not_equal_apt.cpp` | `not_equal` | bool router |
| 直接依赖 | `math/cast/op_api/cast.cpp` | `Cast` | no-op/AIV/AICPU selector |
| 直接依赖 | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling` | Cast POD/workspace |
| 直接依赖 | `math/cast/op_kernel/arch35/cast_struct.h` | `castMap/GetCastPolicy` | pair policy |
| 直接依赖 | `math/cast/op_kernel/cast_apt.cpp` | `cast` | AIV policy router |
| 直接依赖 | `math/cast/op_kernel_aicpu/cast_aicpu.cpp` | `CastCpuKernel` | CPU fallback body |
| 公共 POD | `pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | `BroadcastOneDimTilingDataAdvance` | 16B ABI |
| 公共 Tiling | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling` | mode202、tile/core |
| 公共 Tiling | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse` | scalar合轴 |
| 公共 router | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch` | 202映射 |
| 公共 body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch` | GM/UB/循环/同步 |
| 公共 primitive | `pkg_inc/op_common/atvoss/util/vec.h` | `CopyInBrc/CopyOut` | DataCopyPad |
| 公共内存 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | buffer ID/活跃节点 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | mode202 16B POD、Cast 96B POD | 字段顺序/位宽/单位及 int16 blockNum | 两 POD 定义 |
| 调用顺序 | Contiguous→scalar tensor→核心→Cast→ViewCopy | BOOL/整数分支与失败短路 | `GetWorkspaceSizeCommon` |
| workspace/内存 | 核心6/9 buffer；节点16MiB请求 | 聚合值只能由executor返回；GM中间生命周期 | ATVOSS/Cast Tiling |
| 同步 | 核内 buffer ID；跨节点同 stream；CPU join | scalar ping/pong需各初始化一次 | Advance scheduler |
| 精度与尾块 | XOR predicate；Cast policy round/static_cast | 不得用无mask store；不得改变Cast round | `XorCompute/GetCastPolicy/CastTask` |
| 失败边界 | dtype/shape/UB/pair检查 | 不得把 A2/A3 实现缺失推测补齐 | API/Tiling/CMake |