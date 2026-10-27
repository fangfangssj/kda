---
type: Operator
title: "GeScalar 950 ACLNN 算子知识"
description: "说明 aclnnGeScalar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/docs/aclnnGeScalar&aclnnInplaceGeScalar.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_api/aclnn_ge_scalar.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_api/aclnn_ge_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_api/greater_equal.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_host/arch35/greater_equal_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/arch35/greater_equal_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/arch35/greater_equal_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/greater_equal_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_host/greater_equal_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_impl.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/cast_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel_aicpu/cast_aicpu.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
created_at: "2026-08-25T11:09:24Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# GeScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnGeScalarGetWorkspaceSize` + `aclnnGeScalar`，公开 ACLNN 两段式接口，无版本后缀 |
| 计算语义 | `out[i] = (self[i] >= other) ? true : false`；比较前按 950 TensorScalar 规则得到提升类型 `P`，最终 BOOL 结果可再转换到调用者声明的 `out.dtype` |
| 输入输出 | `self/out` 同 shape、ND、0～8 维、可非连续；`other` 为 Host scalar；空 `self` 直接成功 |
| 执行类型 | 核心为 AIV_VECTOR mode 202 或 AICPU；可达的前/后 Cast 为 AIV_VECTOR 或 AICPU。无 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001～R-003：AIV GreaterEqual，分别直出/AIV Cast/AICPU Cast；R-004～R-006：AICPU GreaterEqual，分别直出/AIV Cast/AICPU Cast。每条路线的可选前置 Cast 由 `self.dtype→P` 独立选择 no-op/AIV/AICPU |
| 核心资源 | AIV 核心使用 GM、隐式 LEVEL_2 UB tensor pool、512 B tile 对齐和 16 MiB tiling workspace 请求；AIV Cast 使用双缓冲 Queue/TBuf；AICPU 节点使用 GM 与 CPU 区间分片 |
| 来源闭包 | source_closed：公开 ACLNN→类型提升与 executor 节点图→GreaterEqual/Cast launcher→950 Tiling/POD→AIV/AICPU body→ATVOSS mode 202 直接依赖均闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `P` 为 Host 计算的提升类型。950 的 `CheckPromoteType` 先把浮点 scalar 的默认类别归为 `FLOAT`、复数 scalar 归为 `COMPLEX64`，再调用 `CombineCategoriesWithComplex(self.dtype, scalarDefaultDtype)`；若结果为 BOOL，又强制改为 INT8。随后 `self` 与 `other` 都转换到 `P`，核心节点产生 BOOL：

```text
P = PromoteForRegBase(self.dtype, other.dtype)
P = (P == BOOL) ? INT8 : P
core[i] = cast<P>(self[i]) >= cast<P>(other)
out[i] = cast<out.dtype>(core[i])
```

因此 BOOL×BOOL 并不是 BOOL Device 比较，而是 BOOL→INT8 后执行 INT8 GreaterEqual。本文只覆盖非原地 `aclnnGeScalar`；不合并 Tensor-Tensor，也不合并复用 `aclnnGeScalarCommon` 的 `aclnnInplaceGeScalar`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；0～8 维；可非连续 | DOUBLE、BFLOAT16、FLOAT16、FLOAT、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL | 与 `out` shape 完全相同；非空时先经 `Contiguous` |
| `other` | Host scalar，无 layout/shape | 同公开列表 | 按 950 标量默认类别与 `self` 推导 `P`，再由 executor 物化为 dtype `P` 的单元素 tensor |
| `out` | ND；0～8 维；可非连续 | DOUBLE、BFLOAT16、FLOAT16、FLOAT、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128 | shape 必须等于 `self`；BOOL 核心结果必须可 Cast 到此 dtype |
| `workspaceSize` | 标量指针 | `uint64_t*` | 第一段返回 executor 聚合的 Device workspace 大小；空输入固定写 0 |
| `executor` | Host 对象 | `aclOpExecutor**` | 持有中间 tensor、节点顺序及生命周期，交给第二段执行 |
| `stream` | Device stream | `aclrtStream` | 第二段按 executor 顺序提交所有可达节点 |

### 入口与版本

固定提交公开的最新入口是无版本后缀 `aclnnGeScalarGetWorkspaceSize/aclnnGeScalar`。公开文档明确列出 Ascend 950PR/950DT；`greater_equal_def.cpp` 为 `ascend950` 注册 `greater_equal_apt`，binary config 与 CMake 的 `ascend950→arch35` 接线一致。固定提交也公开 A2/A3 支持，但 `math/greater_equal/` 只包含 `arch35` Host/Kernel，CMake 对 A2/A3 指定的 `arch32` 源码不可见，故本任务不生成无后缀 A2/A3 卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnGeScalarCommon` 的顺序是：创建 executor；检查 `workspaceSize/self/other/out` 非空；950 上检查 `self` 属于 `REGBASE_DTYPE_SUPPORT_LIST`、`out` 属于输出列表；计算 `P` 并校验 `self→P`、`other→P`、`BOOL→out.dtype` 可转换；检查 `self/out` 均不超过 8 维且 shape 相同。任一步失败即返回对应 ACLNN 错误。

空 `self` 写 `workspaceSize=0`、释放 executor 并成功返回，不创建 Device 节点。非空路径严格按以下顺序构造：

1. `l0op::Contiguous(self)`；
2. `executor->ConvertToTensor(other, P)`；
3. `l0op::Cast(selfContiguous, P)`，同 dtype 时返回原 tensor而不入队；
4. `l0op::GreaterEqual(selfCasted, otherTensor)`；
5. `l0op::Cast(greaterEqualResult, out.dtype)`，`out.dtype==BOOL` 时不入队；
6. `l0op::ViewCopy(..., out)`。

`Contiguous`、`ConvertToTensor`、`ViewCopy` 的内部 launcher 是 ACLNN executor 框架边界；当前算子拥有的选择器、GreaterEqual/Cast 节点及其顺序已继续闭合。

### 参数、Workspace 与生命周期

第一段的 `CREATE_EXECUTOR` 拥有连续化结果、单元素 scalar tensor、前置 Cast 输出、BOOL 核心输出及后置 Cast 输出。`GetWorkspaceSize()` 返回整张节点图的聚合值并把 executor 转交调用者；第二段由 `CommonOpExecutorRun(workspace, workspaceSize, executor, stream)` 执行。

AIV GreaterEqual 的 `BroadcastBaseTiling::DoTiling` 将单节点 workspace[0] 写为 `16*1024*1024` 字节，而 `greater_equal` entry 不读取 `workspace`。AIV Cast 同样请求 16 MiB；AICPU 元数据与多节点 workspace 的合并/复用公式位于 executor 壳，固定仓不可见。工程复用必须使用第一段返回值，不能把各节点请求简单相加或擅自缩小。

### 启动图与失败传播

记 `C(s,d)` 为 `(s,d)` 命中 Cast 的 950 `REGBASE_AICORE_DTYPE_SUPPORT_LIST`，核心 AIV 类型集 `A={FLOAT16,BFLOAT16,FLOAT,INT32,INT64,INT8,UINT8,UINT64}`；Scalar API 的 `P==BOOL` 已被改为 INT8。

```text
self --Contiguous--> selfC
other --ConvertToTensor(P)--> scalarP

if selfC.dtype != P:
  C(selfC.dtype,P) ? [Cast AIV] : [Cast AICPU] -> selfP
else:
  selfP aliases selfC

if P in A: [GreaterEqual AIV mode 202] -> boolCore
else:      [GreaterEqual AICPU]         -> boolCore

if out.dtype == BOOL: boolOut aliases boolCore
else if C(BOOL,out.dtype): [Cast AIV]   -> boolOut
else:                       [Cast AICPU] -> boolOut

boolOut --ViewCopy--> out
```

| 边 | offset/extent/dtype/layout | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|
| `selfC→preCast/core` | 独立或别名 GM tensor；`N` 元素；self dtype；连续 ND | executor 同 stream | executor 持有至末节点完成 | Contiguous 失败不入队后续 |
| `scalarP→core` | GM 单元素；dtype `P`；标量广播 | 同 stream；物化先于核心 | executor | 转换失败阻断核心 |
| `preCast→core` | `N` 元素；dtype `P`；连续 ND | Cast→core stream 顺序 | executor | Cast 失败阻断核心 |
| `core→postCast/ViewCopy` | `N` 个 BOOL 字节；连续 ND | core→consumer stream 顺序 | executor，最后消费后可回收 | core 失败阻断写出 |
| `postCast→ViewCopy` | `N` 元素；`out.dtype`；连续 ND | Cast→ViewCopy stream 顺序 | executor | Cast 失败阻断 ViewCopy |

## Route 与参数设计

### 分核与资源计算

Scalar 路径的两输入在连续化后合轴为一维输出 `N`；`InputSize+OutputSize=3≤4` 且 DAG `VarSize=0`，因此核心固定选择 `SCH_MODE_ONE_DIM_ADVANCE=202`。`GreaterEqualCompute<T>` 的 LEVEL_2 DAG 最大同时存活 buffer 数为 9，`Smax=max(sizeof(T),1)`、`Smin=min(sizeof(T),1)=1`：

```text
tile0 = floor((UB_bytes / 9) / 512) * 512 / Smax
if coreNum * (512 / Smin) > N:
    tile = 512 / Smin
else if coreNum * tile0 > N:
    tile = floor((N / coreNum) / (512 / Smax)) * (512 / Smax)
else:
    tile = tile0
ubOuter = ceil(N / tile)
blockFormer = ceil(ubOuter / coreNum)
if blockFormer == 1: blockFormer = 2
blockNum = ceil(ubOuter / blockFormer)
```

Device 对最后一核重算 `blockTail=ubOuter-(blockNum-1)*blockFormer`，最后一 tile 有效长度 `ubTail=N-(ubOuter-1)*tile`。每核元素基址为 `blockFormer*tile*blockIdx`。AICPU 核心在 `N>=32768` 时按 `max(4, CPU数-kResvCpuNum)` 计算 `per_unit_size=N/min(N,maxCore)` 并行分片，否则单区间执行。

### Route 选择

前置 Cast 是每条非空 Route 的正交前缀：`pre=N` 当 `self.dtype==P`，`pre=V` 当不等且 `C(self.dtype,P)`，否则在两端 dtype 均命中 AICPU 表时 `pre=C`；失败则整条路线失败。

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 AIV 核心直出 | `pre=N/V/C → GE key 202 → post=N` | 非空，`P∈A && out.dtype==BOOL` | R-002/R-003 |
| 2 | R-002 AIV 核心+AIV 后 Cast | `pre=N/V/C → 202 → Cast key 0` | `P∈A && out.dtype!=BOOL && C(BOOL,out.dtype)` | R-003 |
| 3 | R-003 AIV 核心+AICPU 后 Cast | `pre=N/V/C → 202 → AICPU Cast` | `P∈A && out.dtype!=BOOL && !C(BOOL,out.dtype)`，两端命中 AICPU Cast 表 | 失败 |
| 4 | R-004 AICPU 核心直出 | `pre=N/V/C → AICPU GE → post=N` | `P∈{DOUBLE,INT16,UINT16,UINT32} && out.dtype==BOOL` | R-005/R-006 |
| 5 | R-005 AICPU 核心+AIV 后 Cast | `pre=N/V/C → AICPU GE → Cast key 0` | 上述 AICPU 核心类型，且 `out.dtype!=BOOL && C(BOOL,out.dtype)` | R-006 |
| 6 | R-006 AICPU 核心+AICPU 后 Cast | `pre=N/V/C → AICPU GE → AICPU Cast` | 上述 AICPU 核心类型，且 `out.dtype!=BOOL && !C(BOOL,out.dtype)`，AICPU Cast 支持 | 失败 |

```text
empty -> success(workspace=0)
P = PromoteForRegBase(self, other); if P==BOOL: P=INT8
pre = same ? N : (C(self,P) ? V : C)
core = P in A ? AIV_GE_202 : AICPU_GE
post = out==BOOL ? N : (C(BOOL,out) ? V : C)
select Route by (core, post); pre is the recorded route prefix
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | pre selector→202 | 950、非空、`P∈A`、out BOOL | 可选 preCast→单 AIV GE | `BroadcastOneDimAdvanceSch<GreaterEqualCompute<P>::OpDag>` | pre 节点 GM 边；核心 9-buffer pool；stream | core mask/count 由有效 tile 长度保护 |
| R-002 | pre→202→0 | R-001 核心，BOOL→out 命中 AIV Cast pair | 可选 preCast→AIV GE→AIV Cast | GE DAG；`GetCastPolicy<BOOL,out>` 的 Direct/Through/TwoCast | 三节点各自 UB，跨节点 GM | 各节点独立 former/tail |
| R-003 | pre→202→AICPU | R-001 核心，BOOL→out 不命中 AIV、命中 AICPU | 可选 preCast→AIV GE→AICPU Cast | GE DAG；`CastCpuKernel::CastTask<bool,D>` | AIV UB→GM→CPU 分片 | CPU `[start,end)` 自然闭合 |
| R-004 | pre→AICPU | 950、非空、`P` 为 DOUBLE/INT16/UINT16/UINT32、out BOOL | 可选 preCast→AICPU GE | `GreaterEqualCpuKernel::NoBcastCompute<P>` | GM；无 AIV UB；CPU 分片 join | `N<32768` 串行，否则并行；N=1 SAME_SHAPE |
| R-005 | pre→AICPU→0 | R-004 核心，BOOL→out 命中 AIV Cast | 可选 preCast→AICPU GE→AIV Cast | CPU GE；Cast policy body | CPU 写 GM→stream 可见→AIV UB | Cast POD 处理尾核/尾 tile |
| R-006 | pre→AICPU→AICPU | R-004 核心，BOOL→out 仅 AICPU Cast | 可选 preCast→AICPU GE→AICPU Cast | CPU GE；CPU Cast | 两次 CPU 分片，中间 BOOL GM | 两节点分别 `ParallelFor` join |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域 | `self->IsEmpty()` | mode 202 禁止 `dimLen==0` | 全部 | Host 直接成功，workspace=0 | `aclnnGeScalarCommon` |
| AIV core dtype | `P∈A` | arch35 binary/模板与 Vector Compare 可实例化集合 | R-001～R-003 | AICPU GE | `l0op::IsAiCoreSupport` |
| 输入同 dtype | `x1.dtype==x2.dtype` | 一个 `GreaterEqualCompute<T>` 模板读取两输入 | AIV core | Tiling `GRAPH_FAILED` | `GreaterEqualTiling::DoOpTiling` |
| 一维 Advance | 合轴 rank=1、输入连续、3 个 GM 参数≤4、`VarSize=0` | 使用 mode 202 的固定一维调度 | AIV core | 本 Scalar 路径前置 Contiguous 保证；异常则其它广播 mode/失败 | `BroadcastBaseTiling::DoTiling` |
| UB 512 B 对齐 | `floor((UB/9)/512)>0` | 9 个 LEVEL_2 buffer 与 ping/pong 调度 | AIV core | `GRAPH_FAILED` | `GetOneDimUBFormerAdvance` |
| blockNum 窄化 | `ceil(ubOuter/blockFormer)` 写入 `int16_t` | POD 字段只有 16 位 | AIV core | 固定源码未见显式溢出检查 | `DoOneDimFormerTilingAdvance` |
| AIV Cast pair | `(src,dst)∈REGBASE_AICORE_DTYPE_SUPPORT_LIST` | `GetCastPolicy` 必须存在 | 所有 pre/post Cast | AICPU Cast | `l0op::Cast/IsAiCoreSupport` |
| Cast 非空 | `shapeSize_>0` | AIV Cast tiling 不接受空 tensor | R-002/R-005 及 pre=V | 顶层空域已先返回 | `CastTiling::GetShapeAttrsInfo` |
| AICPU GE dtype | DOUBLE/INT16/UINT16/UINT32 等命中 switch | CPU 模板表必须存在 | R-004～R-006 | 参数错误 | `GreaterEqualCpuKernel::Compute` |
| AICPU 并行阈值 | scalar broadcast `N>=32768` | 降低小张量分片开销 | R-004～R-006 | 单线程区间 | `NoBcastCompute` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen` | `int64_t`/64 | sizeof=16，alignof=8，offset=0 | 元素 | `DoOneDimFormerTilingAdvance: dims.back()[0]` | `BroadcastOneDimAdvanceSch::Init` 计算 `ubOuter/ubTail` | `>0` |
| 同上 | `tileNum` | `int32_t`/32 | offset=8 | 元素/tile | `GetOneDimUBFormerAdvance` 后按核数修正 | `Init/Process/CalcBlockOffset` | `>0`；基于 512 B 对齐 |
| 同上 | `blockNum` | `int16_t`/16 | offset=12 | AIV core | `DoOneDimFormerTilingAdvance` 从 `uint64_t blockNum` 窄化 | `Init` 识别尾核 | 固定源码无显式窄化检查 |
| 同上 | `scalarFlag` | `int16_t`/16 | offset=14 | 输入位图 | `Σ((dims[i][0]==1)<<i)` | `CopyIn` 决定 GM copy 或 scalar cache | 常规 N>1 为 2；N=1 可为 3 |
| `CastTilingData` | `blockNum` | `int64_t`/64 | sizeof=96，alignof=8，offset=0 | core | `CastTiling::DoOpTiling` | Cast body 的尾核判断 | `1..AIV coreNum` |
| 同上 | `ubFormer` | `int64_t`/64 | offset=8 | 元素/tile | `GetUbFormer` | Queue 容量与每轮 copy/compute | policy UB 上界 |
| 同上 | `blockFormer` | `int64_t`/64 | offset=16 | 元素/core | `ceil(N/core)` 再 8 元素对齐 | GM block offset | 尾核可能更短 |
| 同上 | `ubLoopOfFormerBlock` | `int64_t`/64 | offset=24 | loop/core | `ceil(blockFormer/ubFormer)` | 非尾核 `Process` | ≥1 |
| 同上 | `ubLoopOfTailBlock` | `int64_t`/64 | offset=32 | loop/core | `ceil(blockTail/ubFormer)` | 尾核 `Process` | ≥1 |
| 同上 | `ubTailOfFormerBlock` | `int64_t`/64 | offset=40 | 元素 | Host 余数公式 | 非尾核最后 loop | 有效长度 |
| 同上 | `ubTailOfTailBlock` | `int64_t`/64 | offset=48 | 元素 | Host 余数公式 | 尾核最后 loop | 有效长度 |
| 同上 | `regCopyInStep` | `int64_t`/64 | offset=56 | 元素/register step | `GetUbCopyStep` | Micro Cast load | Direct/Through/TwoCast 可为 0 |
| 同上 | `regCopyOutStep` | `int64_t`/64 | offset=64 | 元素/register step | `GetUbCopyStep` | Micro Cast store | 同上 |
| 同上 | `ubFormerRegLoop` | `int64_t`/64 | offset=72 | register loop | Host 按 VL/bit 数推导 | Micro Cast 整 tile | 非 Micro policy 为 0 |
| 同上 | `ubTailOfFormerRegLoop` | `int64_t`/64 | offset=80 | register loop | Host tail 推导 | Micro Cast former-tail | 同上 |
| 同上 | `ubTailOfTailRegLoop` | `int64_t`/64 | offset=88 | register loop | Host tail 推导 | Micro Cast tail-core-tail | 同上 |
| AICPU GE 参数 | N/A | N/A | 无算子 POD | N/A | launcher 传 tensor descriptors | `CpuKernelContext` | AICPU task ABI 为框架边界 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | pre selector→202 | scalar API；`GreaterEqualTiling::DoOpTiling` | 可选 Cast POD；16 B broadcast POD | 可选 `cast<0>`/AICPU Cast；`greater_equal<202>` | Cast body；GE DAG；ATVOSS one-d scheduler | greater_equal/cast OpDef+CMake ascend950 |
| R-002 | pre→202→0 | 同上 + Cast Tiling | broadcast POD + post Cast POD | `greater_equal`→`cast` | GE DAG + `GetCastPolicy<BOOL,D>` | 同上 |
| R-003 | pre→202→AICPU | scalar API + GE Tiling + Cast selector | broadcast POD；AICPU attrs | `greater_equal`→AICPU Cast | GE DAG + `CastCpuKernel` | GE 950 config + Cast AICPU 注册 |
| R-004 | pre→AICPU GE | scalar API + l0 selector | 可选 Cast POD；AICPU tensor descriptors | optional Cast→AICPU GE | `GreaterEqualCpuKernel::NoBcastCompute` | `greater_equal_aicpu_def.cpp` |
| R-005 | pre→AICPU GE→0 | 同上 + Cast Tiling | Cast POD | AICPU GE→`cast<0>` | CPU GE + Cast policy | AICPU GE + Cast ascend950 |
| R-006 | pre→AICPU GE→AICPU Cast | scalar API + 两个 AICPU selector | AICPU attrs/descriptors | AICPU GE→AICPU Cast | 两个 CPU Kernel body | 两个 AICPU 注册 |

## Kernel 实现

### 公共执行流程

所有 Route 先完成可选前置 Cast。AIV Cast 的 selector、POD 与 body 和后置 AIV Cast 相同；AICPU Cast 由 `CastCpuKernel::SetMap` 选择 `(src,dst)` 的 `CastTask`。核心 AIV entry `greater_equal<schMode>` 在 BOOL 模板时改用 `int8_t` DAG，否则用 `DTYPE_X1`；本 Scalar API 的 `P==BOOL→INT8` 使 BOOL entry 分支不可达。mode 202 解析 16 B POD、调用 `InitSocState`，再执行一维 scheduler。核心 AICPU 识别 `otherTensor` 为单元素，走 `Y_ONE_ELEMENT` 的 `NoBcastCompute`。

### Route R-001：AIV GreaterEqual 直出

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/greater_equal/op_api/aclnn_ge_scalar.cpp` | `aclnnGeScalarCommon` | 非空；产生 `P`；out BOOL | optional preCast→GreaterEqual |
| Host | `math/greater_equal/op_api/greater_equal.cpp` | `IsAiCoreSupport` | 950 且 `P∈A` | AICore launcher |
| Tiling | `op_host/arch35/greater_equal_tiling_arch35.cpp` | `DoOpTiling` | 两输入同 dtype；dtype 分支实例化 DAG | key 202 |
| Router | `op_kernel/greater_equal_apt.cpp` | `greater_equal<202>` | `DTYPE_X1=P` | `BroadcastSch<202,OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
optional l0op::Cast(selfC,P) -> cast<0>/CastCpuKernel
GreaterEqual AICore launcher -> greater_equal<202>
  -> BroadcastSch<202,GreaterEqualCompute<P>::OpDag>::Process
    -> BroadcastOneDimAdvanceSch::Init/Process
      -> CopyInBrc<P>(selfP)
      -> CopyInBrc<P>(scalarP; first ping/pong only)
      -> Compare<uint8_t,P,4>
      -> Duplicate<uint8_t>(1), Duplicate<uint8_t>(0)
      -> Select<uint8_t,uint8_t,2>
      -> CopyOut<uint8_t>(boolCore)
ViewCopy(boolCore,out)
```

##### 源码等价伪码

```text
for tile i on blockIdx (ping = i & 1):
  len = (last block && last tile) ? ubTail : tileNum
  x = DataCopyPad(selfP[blockFormer*tileNum*blockIdx + offset], len)
  if scalar buffer ping has not been initialized:
      s[ping] = Duplicate(load scalarP[0], len)
  mask = Compare(x, s[ping], CMPMODE=4, count=len)
  ones = Duplicate(uint8(1), len)
  zeros = Duplicate(uint8(0), len)
  y = Select(mask, ones, zeros, SELMODE=2, count=len)
  DataCopyPad(boolCore[blockBase + offset], y, len)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| selfP GM | `blockFormer*tile*blockIdx+offset` | `len` | P/ND | caller/preCast | CopyIn | executor | tail 按 len |
| scalarP GM | 0 | 1 元素 | P/scalar | ConvertToTensor | scalar cache | executor | 每个 ping 首次读一次 |
| UB input x ping/pong | `bufId*tile*Smax` | `2*tile*Smax` | P | DataCopyPad | Compare | 当前 AIV core | 512 B tile；双缓冲 |
| UB scalar ping/pong | 同一 pool 的两个 bufId | `2*tile*Smax` | P | GetValue+Duplicate | Compare | 当前 AIV core | 初始化后不覆盖 |
| UB compare/one/zero | pool 独立/可复用 ID | `3*tile*Smax` | uint8 逻辑值 | Compare/Duplicate | Select | DAG 生命周期 | LEVEL_2 活跃期复用 |
| UB output ping/pong | pool 两个 bufId | `2*tile*Smax` | uint8 | Select | CopyOut | 当前 AIV core | 双缓冲 |
| boolCore GM | `blockBase+offset` | `len` bytes | BOOL/ND | CopyOut | ViewCopy | executor | 只写有效 len |

##### 分层内存布局

```mermaid
block-beta
  columns 9
  x0["x ping · tile*Smax"] x1["x pong · tile*Smax"]
  s0["scalar ping · tile*Smax"] s1["scalar pong · tile*Smax"]
  cmp["compare · tile*Smax"] one["one · tile*Smax"] zero["zero · tile*Smax"]
  y0["out ping · tile*Smax"] y1["out pong · tile*Smax"]
```

##### 字节账与复用顺序

```text
UB_total = 9 * tileNum * Smax bytes <= UB_bytes
dynamic_ub = N/A
workspace_core = 16 MiB requested by Tiling；entry不读取
workspace_total = executor聚合值（固定仓不可见具体跨节点复用公式）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x/output ping-pong | CopyIn/Select | Compare/CopyOut | 同 parity 的后续 tile | buffer ID 的 `GetTensor/ReleaseTensor` | 前一消费者已释放 |
| scalar ping/pong | 前两轮 CopyIn | 每轮 Compare | 本 Kernel 不再覆盖 | `isFirstScalarTime` 在 `i>1` 后关闭 | scalar 全程不变 |
| compare/one/zero ID | DAG node | Select | DAG planner 的后续非重叠节点 | LEVEL_2 buffer ID + release | 所有引用已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `GetTensor<VECIN>/ReleaseTensor` | MTE2 | Compare | x tile | `blockLen=len*sizeof(P)`，无 padding lane 写出 |
| scalar→UB | `GetTensor<VECCALC>/ReleaseTensor` | GM scalar+Duplicate | Compare | ping/pong scalar cache | P 原精度，无额外 cast |
| Compute DAG | `GetTensor<VECCALC>/ReleaseTensor` | Compare/Duplicate | Select | 中间 buffer | `CMPMODE=4`；Select 写 0/1 byte |
| UB→GM | `GetTensor<VECOUT>/ReleaseTensor` | Select | MTE3 | output tile | `len` 控制最后 tile |
| 跨节点 | executor stream | preCast/core | core/ViewCopy | GM intermediate | 无 atomic、无跨核归约 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/greater_equal/op_host/arch35/greater_equal_tiling_arch35.cpp` | `GreaterEqualTiling::DoOpTiling` | dtype→DAG→schMode→key |
| Kernel entry/router | `math/greater_equal/op_kernel/greater_equal_apt.cpp` | `greater_equal<202>` | AIV-only BroadcastSch entry |
| Device body/helper | `math/greater_equal/op_kernel/arch35/greater_equal_dag.h` | `GreaterEqualCompute<P>` | CopyIn→Compare(4)→0/1 Select→CopyOut |
| Device scheduler | `opbase/.../broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch` | core/tile 地址、scalar cache、buffer 同步与 tail |

### Route R-002：AIV GreaterEqual 后接 AIV Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_ge_scalar.cpp` | post `l0op::Cast` | `out.dtype!=BOOL` | Cast selector |
| Host | `math/cast/op_api/cast.cpp` | `IsAiCoreSupport` | `(BOOL,out.dtype)` 命中 REGBASE pair | `CastAiCore` |
| Tiling | `cast_tiling.cpp` | `GetShapeAttrsInfo/DoOpTiling` | pair 命中 `castMap` | key 0、Cast POD |
| Router | `cast_apt.cpp` | `cast<0>` | `GetCastPolicy<BOOL,D>::isValid_` | Direct/Through/TwoCast body |

#### Device 数据流

##### 函数级调用链

```text
R-001 AIV GreaterEqual -> boolCore(GM)
CastAiCore -> cast<0>
  -> BOOL→FLOAT16/INT32: CastDirect
  -> BOOL→UINT8/INT8: CastThrough
  -> BOOL→FLOAT/BF16/INT64/INT16: CastTwo
  -> castOut(GM)
ViewCopy(castOut,out)
```

##### 源码等价伪码

```text
run R-001; stream-order boolCore
for each Cast core and tile:
  CopyIn boolCore[coreBase+offset]
  apply GetCastPolicy<BOOL,D> with its exact map types and RoundMode
  CopyOut castOut[coreBase+offset] for validLen
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boolCore GM | 0 | N bytes | BOOL/ND | GE | Cast | executor | stream edge |
| Cast input Queue | `blockIdx*blockFormer+tileOffset` | depth 2 × `ubFormer*sizeof(bool)` | BOOL | DataCopyPad | policy compute | Cast TPipe | ping/pong |
| Cast output Queue | 同上 | depth 2 × `ubFormer*sizeof(D)` | D | policy compute | DataCopyPad | Cast TPipe | ping/pong |
| Cast mid TBuf | policy dependent | TwoCast 为 `ubFormer*sizeof(mid)`；Direct/Through 无 mid | mapped dtype | first Cast | second Cast | Cast Kernel | 仅 TwoCast 分配 |
| castOut GM | 0 | `N*sizeof(D)` | D/ND | Cast CopyOut | ViewCopy | executor | 有效尾长 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  in0["Cast inQ ping"] in1["Cast inQ pong"] mid["optional mid TBuf"] out0["Cast outQ ping"] out1["Cast outQ pong"]
```

##### 字节账与复用顺序

```text
Direct/Through UB = 2*ubFormer*sizeof(bool) + 2*ubFormer*sizeof(D)
TwoCast UB = 上式 + ubFormer*sizeof(mid)
Host GetUbFormer 使用 (UB-reserve)*4/(inputBits+outputBits[+midBits]) 保守约束
workspace_GE = 16 MiB；workspace_Cast = 16 MiB；聚合值由executor返回
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Cast in/out ping-pong | CopyIn/Compute | Compute/CopyOut | 同 parity 后续 tile | Queue EnQue/DeQue/FreeTensor | 上轮已消费 |
| optional mid | CastTwo 第一步 | CastTwo 第二步 | 下一 tile | 函数内顺序/pipe | 第一步结果已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GE→Cast | executor stream | GE MTE3 | Cast MTE2 | boolCore GM | 值严格为 0/1 |
| Cast Queue | EnQue/DeQue/FreeTensor | CopyIn/Compute | Compute/CopyOut | ping/pong UB | policy RoundMode 固定；tail 用 POD 有效长度 |
| Cast→ViewCopy | executor stream | Cast | ViewCopy | castOut GM | 不写超出 N 的元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling` | pair、UB、block、loop、tail |
| Kernel entry/router | `math/cast/op_kernel/cast_apt.cpp` | `cast<0>` | policy 到 body 的编译期映射 |
| Device body/helper | `math/cast/op_kernel/arch35/cast_impl.h` | `CastDirect/CastThrough/CastTwo` | 双缓冲 CopyIn/Compute/CopyOut |

### Route R-003：AIV GreaterEqual 后接 AICPU Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `math/cast/op_api/cast.cpp` | `Cast` | BOOL→D 不命中 AIV pair；两端命中 AICPU dtype 表 | `CastAiCpu` |
| AICPU | `cast_aicpu.cpp` | `SetBoolMap` | D 为 DOUBLE/UINT16/UINT32/UINT64/COMPLEX64/128 等 | `CastTask<bool,D>` |

#### Device 数据流

##### 函数级调用链

```text
R-001 AIV GreaterEqual -> boolCore(GM)
CastAiCpu -> CastCpuKernel::Compute
  -> ValidateTensors -> ValidateDataType -> SetBoolMap
  -> ExecuteCast -> ParallelFor -> CastTask<bool,D>
ViewCopy -> out
```

##### 源码等价伪码

```text
run AIV GE; stream-order boolCore
workers partition [0,N)
for i in [start,end): castOut[i] = static_cast<D>(boolCore[i])
join workers; expose castOut to ViewCopy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boolCore GM | `i` | N bytes | BOOL | AIV GE | CPU worker | executor | stream task边 |
| castOut GM | `i*sizeof(D)` | `N*sizeof(D)` | D/ND | CPU worker | ViewCopy | executor | worker 区间不重叠 |

##### 分层内存布局

N/A：AICPU Cast 直接读取/写入 GM tensor，不使用 AIV UB。

##### 字节账与复用顺序

```text
UB_total = N/A
workspace_total = AICPU task metadata与executor聚合值；固定仓无独立slice公式
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| castOut worker interval | `CastTask` | ViewCopy | task结束后 | `ParallelFor` join + stream ordering | `[start,end)` 互不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU Cast | `ParallelFor` | worker tasks | `ExecuteCast` return | GM intervals | C++ `static_cast`；end 独占处理尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cast/op_api/cast.cpp` | `IsAiCoreSupport/IsAiCpuSupport` | AIV miss 后的 AICPU fallback |
| Kernel entry/router | `math/cast/op_kernel_aicpu/cast_aicpu.cpp` | `SetBoolMap` | BOOL→D 函数表 |
| Device body/helper | 同上 | `CastTask/ExecuteCast` | 区间并行 `static_cast` |

### Route R-004：AICPU GreaterEqual 直出

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `math/greater_equal/op_api/greater_equal.cpp` | `IsAiCoreSupport` | 950 且 `P` 不在 AIV 表 | AICPU launcher |
| AICPU | `greater_equal_aicpu.cpp` | `Compute` | DOUBLE/INT16/UINT16/UINT32 命中 switch | `GreaterEqualCompute<P>` |
| Broadcast | 同上 | `GreaterEqualCompute` | input1 元素数为 1 | `NoBcastCompute`, Y_ONE_ELEMENT |

#### Device 数据流

##### 函数级调用链

```text
optional preCast -> selfP(GM)
ADD_TO_LAUNCHER_LIST_AICPU(GreaterEqual)
  -> GreaterEqualCpuKernel::Compute
    -> GreaterEqualParamCheck
    -> GreaterEqualCompute<P>
      -> NoBcastCompute<P>
        -> SpecialCompute(Y_ONE_ELEMENT, start, end)
ViewCopy(boolCore,out)
```

##### 源码等价伪码

```text
type = (N==1) ? SAME_SHAPE : Y_ONE_ELEMENT
if N >= 32768: ParallelFor([0,N), perUnit)
else: one interval [0,N)
for i in interval:
  boolCore[i] = (type==SAME_SHAPE) ? selfP[i] >= scalarP[i]
                                      : selfP[i] >= scalarP[0]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| selfP GM | `i*sizeof(P)` | `N*sizeof(P)` | P/ND | caller/preCast | CPU worker | executor | 只读 |
| scalarP GM | 0 | `sizeof(P)` | P/scalar | ConvertToTensor | all workers | executor | 只读共享 |
| boolCore GM | i | N bytes | BOOL/ND | CPU worker | ViewCopy/postCast | executor | worker 区间不重叠 |

##### 分层内存布局

N/A：AICPU GreaterEqual 使用 CPU 指针直接访问 GM tensor，不使用 AIV UB、TQue 或 TBuf。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total = AICPU task/executor边界；算子源码没有独立workspace slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| boolCore worker interval | `SpecialCompute` | ViewCopy/postCast | CPU task结束后 | `ParallelFor` join | worker区间不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU GE | serial loop/`ParallelFor` | workers | Compute return | boolCore intervals | 按 P 原生 C++ `>=`；`[start,end)` 无越界 |
| 跨节点 | executor stream | preCast/GE | GE/ViewCopy | GM intermediate | 前序失败阻断后序 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/greater_equal/op_api/greater_equal.cpp` | `GreaterEqual/IsAiCoreSupport` | AICPU fallback selector |
| Kernel entry/router | `math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp` | `Compute` | dtype switch |
| Device body/helper | 同上 | `NoBcastCompute/SpecialCompute` | scalar广播、并行阈值、区间比较 |

### Route R-005：AICPU GreaterEqual 后接 AIV Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Core | `greater_equal.cpp` | `IsAiCoreSupport` | `P` 不在 AIV GE 表 | AICPU GE |
| Post | `cast.cpp` | `IsAiCoreSupport` | BOOL→out 命中 REGBASE pair | AIV Cast |

#### Device 数据流

##### 函数级调用链

```text
R-004 AICPU GreaterEqual -> boolCore(GM)
R-002 CastAiCore -> cast<0> -> pair-specific body -> castOut(GM)
ViewCopy -> out
```

##### 源码等价伪码

```text
CPU intervals compute boolCore; join and stream-order
AIV Cast cores process boolCore in POD-sized tiles with ping/pong queues
copy valid cast tail to castOut; ViewCopy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| boolCore GM | i | N bytes | BOOL | AICPU GE | AIV Cast | executor | CPU→AIV stream edge |
| Cast UB | Cast POD core/tile | Direct/Through/TwoCast 公式 | BOOL→D | MTE2/Vector | MTE3 | Cast core | ping/pong + optional mid |
| castOut GM | i | `N*sizeof(D)` | D | AIV Cast | ViewCopy | executor | tail有效长度 |

##### 分层内存布局

与 R-002 的 AIV Cast 布局相同；AICPU GE 不占 AIV UB，两个节点之间仅通过 BOOL GM tensor 交接。

##### 字节账与复用顺序

```text
AICPU_GE UB = N/A
AIV_Cast UB = 2*ubFormer*(sizeof(bool)+sizeof(D)) + optional mid
workspace_Cast = 16 MiB；executor返回聚合值
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| boolCore GM | CPU GE | AIV Cast | Cast完成后 | CPU join + executor stream | 不提前回收 |
| Cast ping/pong | CopyIn/Compute | Compute/CopyOut | 同 parity 下一 tile | Queue 生命周期 | 前轮已释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU→AIV | executor stream | AICPU GE | AIV Cast | boolCore GM | BOOL 0/1 无精度损失 |
| AIV Cast | Queue + POD tail | Cast stages | CopyOut | Cast UB/GM | policy RoundMode；tail精确 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `greater_equal.cpp` + `cast_tiling.cpp` | selectors/`CastTiling` | CPU core→AIV post组合 |
| Kernel entry/router | `cast_apt.cpp` | `cast<0>` | Cast policy |
| Device body/helper | `greater_equal_aicpu.cpp` + `cast_impl.h` | CPU GE + Cast bodies | 两节点完整实现 |

### Route R-006：AICPU GreaterEqual 后接 AICPU Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Core | `greater_equal.cpp` | `IsAiCoreSupport` | `P` 不在 AIV GE 表 | AICPU GE |
| Post | `cast.cpp` | `Cast` | BOOL→out AIV miss、AICPU hit | AICPU Cast |

#### Device 数据流

##### 函数级调用链

```text
R-004 GreaterEqualCpuKernel -> boolCore(GM)
R-003 CastCpuKernel::Compute -> CastTask<bool,D> -> castOut(GM)
ViewCopy -> out
```

##### 源码等价伪码

```text
ParallelFor/serial GE writes boolCore; join
ParallelFor Cast partitions [0,N): castOut[i]=static_cast<D>(boolCore[i]); join
ViewCopy castOut to caller layout
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| selfP/scalarP GM | i/0 | N/1 | P | Host/preCast | CPU GE | executor | read-only |
| boolCore GM | i | N bytes | BOOL | CPU GE | CPU Cast | executor | 两任务顺序交接 |
| castOut GM | `i*sizeof(D)` | `N*sizeof(D)` | D | CPU Cast | ViewCopy | executor | worker区间不重叠 |

##### 分层内存布局

N/A：两个核心节点均为 AICPU，直接使用 GM tensor 与 CPU 区间，不使用 AIV UB。

##### 字节账与复用顺序

```text
UB_total = N/A
workspace_total = executor/AICPU task边界，固定仓不可见独立slice公式
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| boolCore | GE workers | Cast workers | Cast join后 | GE join + stream顺序 | Cast不得早于GE完成 |
| castOut intervals | Cast workers | ViewCopy | task结束后 | Cast `ParallelFor` join | 区间不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GE | serial/ParallelFor | GE workers | join | boolCore | P 原生 `>=` |
| GE→Cast | executor stream | GE task | Cast task | boolCore | BOOL 精确 0/1 |
| Cast | ParallelFor | Cast workers | join | castOut | C++ `static_cast`；end独占尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `greater_equal.cpp` + `cast.cpp` | 两个 AICPU selector | 节点类型与顺序 |
| Kernel entry/router | 两个 `op_kernel_aicpu/*.cpp` | `Compute`/dtype maps | CPU 模板选择 |
| Device body/helper | 同上 | `SpecialCompute` + `CastTask` | 比较与输出转换循环 |

## 算子特定优化

### Trick T-001：scalar 只装载两个 ping/pong 副本并跨 tile 复用

#### 触发与源码依据

- 适用 Route：R-001～R-003 的 AIV GreaterEqual 核心；R-001～R-006 中 pre=V 且 AIV Cast 本身不使用此机制。
- 触发条件：mode 202 的 `scalarFlag` 标记 `otherTensor` 为长度 1；`CopyInBrc` 当前输入被识别为 scalar。
- Golden 来源：[`broadcast_sch_one_dim_advance.h:201-225`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h#L201-L225)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int32_t inputIdx = 1 << input::Pos;
bool isScalar = tilingData->scalarFlag & inputIdx;
if (isScalar) {
    if (isFirstScalarTime) {
        globalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ inputType*>(inGm[input::Pos].GetPhyAddr(0)));
        inputType scalar = globalTensor.GetValue(0);

        GetTensor<TPosition::VECCALC>(bufId);
        if constexpr (std::is_same_v<inputType, double>) {
          Vec::Duplicate(inTensor.template ReinterpretCast<int64_t>(), *reinterpret_cast<const int64_t*>(&scalar), tileLength);
        } else {
          Vec::Duplicate(inTensor, scalar, tileLength);
        }
        ReleaseTensor<TPosition::VECCALC>(bufId);
    }
}
```

#### 机制与设计原因

`scalarFlag` 的第二输入位使 scheduler 不对每个 tile 做 `N` 元素 GM CopyIn。它只在 `isFirstScalarTime` 为真时读取 `other[0]`，分别初始化 ping 与 pong 两个 UB 副本；循环越过前两次后关闭初始化，后续 tile 复用同一 parity 的副本。这样核心每个后续 tile 只搬运 `self` 和输出，不重复读取或物化标量广播张量。

#### 迁移边界

- 适用：一维 elementwise/broadcast DAG，某输入在合轴后长度为 1 且整个 Kernel 生命周期值不变。
- 前提：Host 正确生成 `scalarFlag`；至少为实际 ping/pong parity 各初始化一次；UB buffer ID 不被其它活跃节点提前复用。
- 失效：输入不是 scalar 时必须按 core/tile 地址执行普通 CopyIn；非 mode 202 调度要使用其自身广播实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不得在仅初始化一个 parity 后关闭 `isFirstScalarTime`，否则另一个 parity 的 Compare 会读取未初始化 UB。固定仓未提供独立错误实现；此边界由 `i&1` 与 `i>1` 的配套源码直接约束。

### Trick T-002：Compare mask 立即 Select 为 0/1，直接作为 BOOL 输出

#### 触发与源码依据

- 适用 Route：R-001～R-003 的 AIV GreaterEqual 核心。
- 触发条件：所有 mode 202 AIV dtype 实例。
- Golden 来源：[`greater_equal_dag.h:24-40`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/arch35/greater_equal_dag.h#L24-L40)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using CompareRes = Bind<Vec::Compare<uint8_t, T, GREATER_EQUAL_CMP_MODE>, InputX1, InputX2>;

using ConstZero = MAKE_CONST(uint8_t, 0);
using DupZero = Bind<Vec::Duplicate<uint8_t>, ConstZero>;
using ConstOne = MAKE_CONST(uint8_t, 1);
using DupOne = Bind<Vec::Duplicate<uint8_t>, ConstOne>;
using SelectRes = Bind<Vec::Select<uint8_t, uint8_t, GREATER_EQUAL_SEL_MODE>, CompareRes, DupOne, DupZero>;

using OpCopyOut = Bind<Vec::CopyOut<uint8_t>, Placeholder::Out0<uint8_t>, SelectRes>;
```

#### 机制与设计原因

比较结果不经过额外整数 Cast Kernel：DAG 在同一 tile 内用 `Compare<uint8_t,T,4>` 产生选择掩码，以常量 1/0 的 uint8 tensor 做 `Select<...,2>`，随后直接 `CopyOut<uint8_t>`。这把比较与 BOOL 字节物化融合在一个 AIV Kernel；只有调用者要求非 BOOL `out.dtype` 时才增加后置 Cast 节点。

#### 迁移边界

- 适用：输出 ABI 用单字节 BOOL，且 Compare 的 mask 可被 Select mode 2 消费。
- 前提：输出仅允许 0/1；`count` 必须是当前有效 tile 长度；下游把 uint8 物理字节解释为 BOOL。
- 失效：非 BOOL 输出必须按 Host selector 增加 AIV/AICPU Cast，不能把 uint8 字节直接重解释为宽类型。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 BOOL core GM 直接按 `out.dtype` 宽度写入会改变 extent 与元素布局；源码明确通过独立 Cast 节点处理。固定仓未提供其它可验证反例。

## 模板与实现对比

| 对比维度 | R-001 AIV直出 | R-002/R-003 AIV后Cast | R-004 AICPU直出 | R-005/R-006 AICPU后Cast | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | `P∈A,out=BOOL` | `P∈A,out!=BOOL` | `P` 为 AIV不支持的公开类型、out BOOL | 同前且 out非BOOL | 先 core selector，再 post Cast selector |
| 模板与 dtype | `GreaterEqualCompute<P>` | 同左 + BOOL→D Cast | CPU `GreaterEqualCompute<P>` | CPU GE + BOOL→D Cast | 保留 P 比较语义 |
| 循环与分核 | mode 202 core/tile | core循环 + Cast独立循环 | `<32768` 串行，否则 CPU分片 | 两节点分别分片/分核 | 节点各自资源模型 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 9-buffer隐式 pool | 再加 Cast Queue/TBuf 或 CPU | N/A | Cast AIV有Queue；CPU无UB | 无通信/dynamic UB |
| 内存总占用与复用 | `9*tile*Smax` | GE 与 Cast UB 生命周期不重叠 | GM only | GM edge + Cast资源 | Kernel边界后UB失效 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Compare/Duplicate/Select | 再加 Cast policy或CPU loop | CPU scalar-broadcast loop | CPU loop + Cast | 本入口无SIMT/NDDMA/通信 |
| 同步、精度与 tail | buffer token；P原精度；len尾 | stream边；Cast round/tail | ParallelFor join；P原生比较 | 两节点各自join/tail | executor顺序保护GM边 |
| fallback/失败边界 | AIV dtype miss→R-004 | Cast AIV miss→R-003 | CPU dtype miss失败 | Cast CPU miss失败 | selector逐层失败关闭 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持；固定提交缺 `arch32` 实现，未建卡 | 公开文档写支持；固定提交缺 `arch32` 实现，未建卡 | 公开支持且 source_closed |
| Host/Launcher | API selector可见，AIV Host/Tiling闭包不可见 | 同左 | RegBase 类型提升；arch35 BroadcastBaseTiling；AIV/AICPU selector闭合 |
| Kernel/资源 | CMake 指向 `arch32`，固定树无对应 body/POD | 同左 | mode 202 AIV LEVEL_2 pool；AICPU fallback；Cast直接依赖闭合 |

## 限制与失败边界

- `self/other/out/workspaceSize` 任一必要指针为空、dtype/提升/Cast 不支持、`self/out` shape 不同或维数超过 8 均在第一段失败。
- 空 tensor 直接成功且 workspace 为 0；不能进入拒绝 `dimLen==0` 的 AIV Tiling 或拒绝 `shapeSize<=0` 的 AIV Cast Tiling。
- 950 AIV GreaterEqual 不支持 DOUBLE、INT16、UINT16、UINT32，这些公开类型依赖 AICPU；BF16、UINT64 则走 AIV。AICPU GreaterEqual 不实现 BF16/BOOL，但 Scalar API 的这两种可达情况分别由 AIV BF16 和 BOOL→INT8 AIV 承接。
- `BroadcastOneDimTilingDataAdvance.blockNum` 从 `uint64_t` 写入 `int16_t`，固定源码没有显式窄化保护；平台实际 core 数必须可表示。
- `Contiguous/ConvertToTensor/ViewCopy` 的内部 Kernel、executor 聚合 workspace slice、跨 task stream 实现属于框架边界，固定算子仓不可见；算子拥有的调用顺序和失败传播可见。
- A2/A3 公开性与 CMake 声明存在，但固定提交缺少被 `SUPPORT_TILING_DIR=arch32` 指向的 Host/POD/Kernel，无法达到 `source_closed`，故未生成 `ge_scalar.md`。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/greater_equal/docs/aclnnGeScalar&aclnnInplaceGeScalar.md` | `aclnnGeScalar*` | 公开平台、接口语义、dtype/shape |
| 入口声明/约束 | `math/greater_equal/op_api/aclnn_ge_scalar.h` | `aclnnGeScalarGetWorkspaceSize/aclnnGeScalar` | C ABI |
| 入口实现 | `math/greater_equal/op_api/aclnn_ge_scalar.cpp` | `CheckParams/aclnnGeScalarCommon` | 950提升、节点顺序、空域、workspace |
| Launcher | `math/greater_equal/op_api/greater_equal.cpp` | `IsAiCoreSupport/GreaterEqual` | AIV/AICPU选择 |
| Host/Tiling | `math/greater_equal/op_host/arch35/greater_equal_tiling_arch35.cpp` | `GreaterEqualTiling::DoOpTiling` | dtype→DAG→key |
| Kernel entry/body | `math/greater_equal/op_kernel/arch35/greater_equal_struct.h` | `GreaterEqual` template args | schMode key集合 |
| Kernel entry/body | `math/greater_equal/op_kernel/arch35/greater_equal_dag.h` | `GreaterEqualCompute<T>` | Compare+Select DAG |
| Kernel entry/body | `math/greater_equal/op_kernel/greater_equal_apt.cpp` | `greater_equal<schMode>` | AIV entry/router |
| Kernel entry/body | `math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp` | `GreaterEqualCpuKernel` | CPU dtype、scalar广播、并行 |
| 构建接线 | `math/greater_equal/op_host/greater_equal_def.cpp` | `ops::GreaterEqual` | ascend950 AICore config |
| 构建接线 | `math/greater_equal/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | 950 arch35与A2/A3缺口 |
| 入口实现 | `math/cast/op_api/cast.cpp` | `Cast/IsAiCoreSupport/IsAiCpuSupport` | pre/post Cast selector |
| Host/Tiling | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling` | Cast POD与资源公式 |
| Kernel entry/body | `math/cast/op_kernel/arch35/cast_struct.h` | `GetCastPolicy/castMap` | BOOL pair policy |
| Kernel entry/body | `math/cast/op_kernel/arch35/cast_impl.h` | Cast body classes | AIV CopyIn/Compute/CopyOut |
| Kernel entry/body | `math/cast/op_kernel/cast_apt.cpp` | `cast<0>` | policy router |
| Kernel entry/body | `math/cast/op_kernel_aicpu/cast_aicpu.cpp` | `CastCpuKernel` | AICPU Cast map/loop |
| Host/Tiling | `opbase/pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | `BroadcastOneDimTilingDataAdvance` | POD ABI |
| Host/Tiling | `opbase/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling` | mode 202、tile/core/workspace |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch<202>` | scheduler router |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch` | 地址、循环、scalar cache、同步、tail |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/util/vec.h` | `Compare/Select/CopyInBrc/CopyOut` | Vector primitive参数 |
| Kernel entry/body | `opbase/pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch/MemOptCfg` | LEVEL_2 buffer规划 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | mode 202 使用固定 16 B POD；Cast 使用 96 B POD | 字段顺序、位宽、单位与 `blockNum` 窄化 | 两个 TilingData 定义 |
| 调用顺序 | Contiguous→ConvertScalar→preCast→GE→postCast→ViewCopy | 任一失败必须阻断后续；不能把后 Cast 提到比较前 | `aclnnGeScalarCommon` |
| workspace/内存 | AIV GE/Cast 各自请求16 MiB；UB按真实存活节点分配 | 调用者只能使用 executor 聚合 workspace；scalar ping/pong需各初始化 | Broadcast/Cast Tiling与scheduler |
| 同步 | AIV buffer token、CPU ParallelFor join、executor stream | GM producer完成前consumer不得启动；UB不得提前复用 | scheduler/CPU body/executor调用 |
| 精度与尾块 | 在提升类型 P 上比较；AIV用len，CPU用end独占；输出Cast遵循policy | BOOL×BOOL先转INT8；不能把BOOL字节重解释为宽输出 | `CheckPromoteType`、DAG、Cast policy |
| 失败边界 | AIV miss走AICPU；Cast pair miss再查AICPU；空域Host返回 | 不支持的dtype/pair必须失败关闭，不推测新模板 | 两个 L0 selector |