---
type: Operator
title: "InplaceClampMax 950 ACLNN 算子知识"
description: "说明 aclnnInplaceClampMax 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/docs/aclnnClampMax&aclnnInplaceClampMax.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/aclnn_clamp.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/aclnn_clamp.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_api/clip_by_value.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/cast_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_impl.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T23:52:15Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceClampMax 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceClampMaxGetWorkspaceSize` + `aclnnInplaceClampMax`，公开 ACLNN 两段式接口 |
| 计算语义 | 对 `selfRef` 的每个元素原地写入 `min(selfRef[i], clipValueMax)`；内部补出同推导 dtype 的最小值作为下界 |
| 输入输出 | `selfRef`：ND、1–8 维、可非连续；950 支持 FLOAT16/FLOAT/DOUBLE/INT8/UINT8/INT16/INT32/INT64/BFLOAT16，输出与其共享最终目标内存；`clipValueMax` 为 scalar |
| 执行类型 | 空张量无 Device 节点；主计算节点为 AIV_VECTOR；INT8/UINT8/INT16 的非浮点 scalar 与 DOUBLE 路线为 AICPU_CPU；整型遇浮点 scalar 时为 `AIV Cast -> AIV Clip -> AIV Cast` |
| Kernel 路线 | R-001 空张量；R-002 ASS-AIV 同 dtype；R-003 AICPU 同 dtype；R-004～R-008 分别为 INT8/UINT8/INT16/INT32/INT64 的 FLOAT 推导多 Kernel 路线 |
| 核心资源 | AIV Clip：GM + 双缓冲 VECIN/VECOUT；Cast：GM + 双缓冲队列，INT8/UINT8 输入另有中间 TBuf；AICPU：线性地址与 CPU 线程分片；executor 持有中间 Tensor 与 workspace |
| 来源闭包 | source_closed：公开入口、类型推导、Cast/Clip selector、950 Host/Tiling/POD、AIV/AICPU body、最终 ViewCopy 顺序与构建接线均已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公开语义为 `selfRef[i] <- min(selfRef[i], max)`。实现把缺省下界规范化为推导 dtype 的 `-inf` 或整型最小值，执行 `Maxs(tmp, x, minValue)` 后执行 `Mins(y, tmp, maxValue)`。`selfRef` 不直接作为 Clip 输出：executor 先生成连续/转换后的中间 Tensor，Clip 写新 Tensor，最后 `ViewCopy(..., selfRef)` 才提交原地结果，因此复用实现时不能让主 Kernel 盲目别名输入输出。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND，1–8 维，可非连续 | FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、INT32、INT64、BFLOAT16；不支持 BOOL | 输入/输出；空 Tensor 直接返回 workspace 0；最终由 ViewCopy 写回原存储 |
| `clipValueMax` | scalar；文档沿用 1–8 维描述，但 ACL 对象类型是 `aclScalar` | 上述数值类型外还允许 BOOL | 不得为空；950 按 TensorScalar 规则与 `selfRef` 推导计算 dtype |
| `workspaceSize` | 标量输出 | `uint64_t` | 来自 executor 的完整图 workspace，不等于单个 Clip Kernel 的 workspace |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段构图，第二段在给定 stream 上按顺序执行 |

### 入口与版本

固定版本只有 `aclnnInplaceClampMax` 这一公开标量 max 原地入口；`aclnnClampMax`、Tensor max 版本和非原地版本不是本卡的版本族。公开文档明确支持 Ascend 950PR/950DT。950 `ClipByValueV2` OpDef 只注册 FLOAT16/BFLOAT16/FLOAT/INT32/INT64 AICore dtype，CMake 只为 `ascend950` 选择 `arch35` tiling；其它公开 dtype 由 L0 selector 转 AICPU。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `aclnnInplaceClampMaxGetWorkspaceSize` 记录 DFX，把 `selfRef` `const_cast` 为输出，先拒绝空 `clipValueMax`。
2. `aclnnClampCommon` 创建 executor；`CheckParams` 检查 self/out 非空、950 dtype 支持表、输入输出 shape 相同且维数不超过 8。原地入口的 out 即 selfRef。
3. 空 Tensor 设置 `workspaceSize=0` 并释放 executor，不添加 Kernel。
4. 950 `IsRegBase(curArch)` 路径以 self dtype 为 `higher`，用 `ClampPromoteType` 与 scalar 默认 dtype 推导 `promoteType`，并验证推导结果可转回 self dtype。
5. 生成下界 scalar（浮点为 `-inf`，整型为最小值），保留调用者的 max scalar；二者转换成 `promoteType` 的一元素 Tensor。
6. `Contiguous(selfRef)` 后调用 `l0op::Cast`；若 dtype 相同 Cast 直接返回输入，不添加节点。随后添加 `ClipByValueV2`。
7. 950 总会再次调用 `l0op::Cast(clipResult, selfRef.dtype)`；相同 dtype 时仍是零节点直通，否则添加 Cast 节点。最后 `ViewCopy` 把结果写回 `selfRef`。

### 参数、Workspace 与生命周期

`uniqueExecutor` 拥有 Contiguous/Cast/Clip 中间 Tensor、scalar Tensor 和 ViewCopy 节点，`GetWorkspaceSize()` 返回完整图需求，`ReleaseTo(executor)` 把所有权交给调用者。可见的节点级请求为：ASS-AIV Clip 16 MiB，AIV Cast 每节点 16 MiB，AICPU OpDef 声明 100 bytes；executor 如何对这些请求对齐、复用或叠加属于固定仓框架边界，不能把它们直接求和。Clip/ Cast entry 的 `workspace` 参数在本路线 body 内没有读取。

### 启动图与失败传播

```text
R-001: validate -> empty shortcut -> no Device node
R-002: [framework Contiguous] -> ClipByValueV2(AIV, key 2001100) -> [framework ViewCopy -> selfRef]
R-003: [framework Contiguous] -> ClipByValueV2(AICPU) -> [framework ViewCopy -> selfRef]
R-004..R-008:
  [framework Contiguous]
    -> Cast(AIV, self integer -> FLOAT)
    -> ClipByValueV2(AIV, key 2001100, FLOAT)
    -> Cast(AIV, FLOAT -> self integer)
    -> [framework ViewCopy -> selfRef]
```

各节点产出 executor 分配的 ND 中间 Tensor，extent 均为 `N` 个元素；同一 stream 的 executor 顺序保证 producer 完成后 consumer 可见，中间 Tensor 生命周期持续到其最后一个 consumer。任一分配、infer-shape 或 `ADD_TO_LAUNCHER_LIST_*` 失败返回 `nullptr`，`aclnnClampCommon` 立即返回内部错误，不再添加后续节点；第二段统一由 `CommonOpExecutorRun` 返回执行状态。

## Route 与参数设计

### 分核与资源计算

令 `N=selfRef` 元素数、`S=sizeof(promoteType)`、`C=AIV core count`、`UBytes=UB bytes`。标量上下界使 `DimensionCollapse` 得到一维 `[N],[1],[1] -> [N]`，优先级 1 的 SingleDim 模板命中且 `minDim=maxDim=1`，选择 ASS key `2001100`。

ASS 初始 `U=floor(floor((UBytes/2)/2)/128)*128/S`；随后若 `blockNum<C/2` 且 `U*S*2>8192`，把 `U` 缩到接近 `2N/C` 的 128-byte 对齐值，但不低于每个 alive buffer 分摊的 8 KiB 下界。`ubOuter=ceil(N/U)`，`blockFormer=ceil(ubOuter/C)`，`blockNum=ceil(ubOuter/blockFormer)`，最后核使用 `blockTail` 和 `ubTail`。

Cast 使用自己的 `N` 元素 tiling：`core=min(C, ceil(N*srcBits/(4 KiB*8)))`，`blockFormer=align_up(ceil(N/core),8)`，再计算 former/tail 核的 UB loop 与尾元素。Direct/Micro 的 `ubFormer=floor((UBytes-128)*4/(srcBits+dstBits))` 后按 `VL/srcBits` 对齐；TwoCast 用 `(UBytes-160)*4/(srcBits+dstBits+midBits)`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 空张量 | 无 Kernel | `selfRef->IsEmpty()` | 非空进入类型推导 |
| 2 | R-002 ASS-AIV 同 dtype | Clip `2001100` / `ClipByValueV2Ass<T>` | `promoteType==self.dtype` 且 dtype∈{F16,F32,BF16,S32,S64} | R-003 |
| 3 | R-003 AICPU 同 dtype | `ClipByValueV2CpuKernel::DoCompute<T>` | `promoteType==self.dtype` 且 dtype∈{S8,U8,S16,F64} | 推导为 FLOAT 时进入 R-004～008 |
| 4 | R-004 S8→F32→S8 | Cast id 5 -> Clip 2001100 -> Cast id 7 | self=S8 且 scalar 为任一浮点 dtype，TensorScalar 表推导 F32 | 其它 self dtype |
| 5 | R-005 U8→F32→U8 | Cast id 5 -> Clip 2001100 -> Cast id 9 | self=U8 且 scalar 为任一浮点 dtype | 其它 self dtype |
| 6 | R-006 S16→F32→S16 | Cast id 1 -> Clip 2001100 -> Cast id 7 | self=S16 且 scalar 为任一浮点 dtype | 其它 self dtype |
| 7 | R-007 S32→F32→S32 | Cast id 1 -> Clip 2001100 -> Cast id 1 | self=S32 且 scalar 为任一浮点 dtype | 其它 self dtype |
| 8 | R-008 S64→F32→S64 | Cast id 1 -> Clip 2001100 -> Cast id 1 | self=S64 且 scalar 为任一浮点 dtype | 失败 |

```text
if empty: R-001
promoteType = ClampPromoteType(self.dtype, GetScalarDefaultDtype(max.dtype))
if promoteType == self.dtype:
  if dtype in {F16,F32,BF16,S32,S64}: R-002
  else: R-003
else:  # 公开 dtype 组合中即 integer Tensor + floating scalar -> F32
  dispatch self.dtype to R-004/R-005/R-006/R-007/R-008
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | empty | 950，N=0 | 0 Kernel | Host shortcut | workspace 0 | 成功，无写入 |
| R-002 | 2001100 | 950，N>0，同 dtype且 AICore 列表 | 单 Clip AIV | `ClipByValueV2Ass<T>` | 双 VECIN/双 VECOUT；queue 同步 | last core + last UB tail |
| R-003 | AICPU selector | 950，N>0，同 dtype且不在 AICore 列表 | 单 AICPU task | `DoCompute<T>` | 直接读 GM；小任务串行，大任务 ParallelFor | `[beg,end)` 自然保护 |
| R-004 | 5/2001100/7 | S8 + 浮点 scalar | CastTwo -> ASS -> CastMicro | `CastTwo<int8,half,float>`；ASS<float>；MicroCast<float,int8> | 三个 GM 中间边；每节点独立 queue | 各节点独立 former/tail |
| R-005 | 5/2001100/9 | U8 + 浮点 scalar | CastTwo -> ASS -> CastMicro | `CastTwo<uint8,half,float>`；ASS<float>；MicroCastDeinter<float,uint8> | 同上，输出 Cast 每 reg loop 双路处理 | mask + tail reg loop |
| R-006 | 1/2001100/7 | S16 + 浮点 scalar | CastDirect -> ASS -> CastMicro | `CastDirect<int16,float>`；ASS<float>；MicroCast<float,int16> | GM 中间边 + queues | tail elements + mask |
| R-007 | 1/2001100/1 | S32 + 浮点 scalar | CastDirect -> ASS -> CastDirect | int32↔float direct，回转 `CAST_TRUNC` | 双 queues/节点 | tail elements |
| R-008 | 1/2001100/1 | S64 + 浮点 scalar | CastDirect -> ASS -> CastDirect | int64↔float direct，回转 `CAST_TRUNC` | 双 queues/节点 | tail elements |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AICore dtype | Clip self dtype in F16/F32/S32/S64/BF16 | 950 binary/OpDef 只实例化五类 dtype | R-002、R-004～008 | AICPU R-003 | `clip_by_value.cpp::IsAiCoreSupport` |
| SingleDim/ASS | collapse 后四路 rank 均为 1，且 minDim=maxDim=1 | 只需每核读取两个 scalar，不需广播 tensor | 所有 AIV Clip | 若不成立才尝试 UB-broadcast/common；本入口不会构造该形状 | `DetermineSigDimMode` |
| 128-byte UB 对齐 | `ubFormerByte` 向下对齐 128 | 保证整 32B DMA block并给双缓冲分区 | R-002、R-004～008 | U=0 则 tiling 失败 | `CalcInitialUbParams` |
| 低利用率重切 | `blockNum<C/2 && U*S*aliveNum>8192` | 缩小 tile 以增加启用 AIV 核数 | AIV Clip | 保留初始 tile | `AdjustForLowUtilization` |
| AICPU 并行阈值 | `N*sizeof(T) >= 200*1024` | 小输入避免 ParallelFor 调度；大输入保留 2 CPU | R-003 | 串行 loop | `DispatchParallel` |
| Cast pair policy | `(src,dst)` 必须存在 `castMap` 且 REGBASE 支持 | entry 编译期 policy 决定模板、round、Micro load/store | R-004～008 | Host 返回 nullptr/tiling failed | `CastTiling::GetShapeAttrsInfo` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ClipByValueSigDimTilingData` | `blockNum` | int64/64 | sizeof=72, align=8, offset=0 | core | `set_blockNum` | `Init` block guard | >0 |
| 同上 | `ubFormer` | int64/64 | offset=8 | element | `CalcInitialUbParams/AdjustForLowUtilization` | buffer capacity、full tile | 128-byte aligned |
| 同上 | `ubTail` | int64/64 | offset=16 | element | `CalcBlockParams` | last loop len | 1..ubFormer |
| 同上 | `blockFormer` | int64/64 | offset=24 | UB loop/core | `CalcBlockParams` | core GM base、non-last loops | >0 |
| 同上 | `blockTail` | int64/64 | offset=32 | UB loop | `CalcBlockParams` | last core loops | 1..blockFormer |
| 同上 | `xDim` | int64/64 | offset=40 | element | collapsed x dim | 读取但不参与地址上界计算 | N |
| 同上 | `minDim` | int64/64 | offset=48 | element | collapsed min dim | 路由证据；body 不读取 | 本路线=1 |
| 同上 | `maxDim` | int64/64 | offset=56 | element | collapsed max dim | 路由证据；body 不读取 | 本路线=1 |
| 同上 | `yDim` | int64/64 | offset=64 | element | collapsed output dim | Host 分核；body 不读取 | N |
| `CastTilingData` | `blockNum` | int64/64 | sizeof=96, align=8, offset=0 | core | Cast `DoOpTiling` | `Process` last-core test | >0 |
| 同上 | `ubFormer` | int64/64 | offset=8 | element | `GetUbFormer` | queue/TBuf capacity、loop | VL 对齐 |
| 同上 | `blockFormer` | int64/64 | offset=16 | element/core | align_up(ceil(N/core),8) | GM base | 8-element aligned |
| 同上 | `ubLoopOfFormerBlock` | int64/64 | offset=24 | loop | Host ceil-div | non-last core loop count | >=1 |
| 同上 | `ubLoopOfTailBlock` | int64/64 | offset=32 | loop | Host ceil-div | last core loop count | >=1 |
| 同上 | `ubTailOfFormerBlock` | int64/64 | offset=40 | element | Host remainder | non-last last tile | 1..ubFormer |
| 同上 | `ubTailOfTailBlock` | int64/64 | offset=48 | element | Host remainder | last core last tile | 1..ubFormer |
| 同上 | `regCopyInStep` | int64/64 | offset=56 | element/register load | `GetUbCopyStep` | Micro DataCopy post-update | Direct/Two 路线为0且不读 |
| 同上 | `regCopyOutStep` | int64/64 | offset=64 | element/register store | `GetUbCopyStep` | Micro DataCopy post-update | Direct/Two 路线为0且不读 |
| 同上 | `ubFormerRegLoop` | int64/64 | offset=72 | reg loop | Host按 VL 推导 | full Micro tile | 非 Micro=0 |
| 同上 | `ubTailOfFormerRegLoop` | int64/64 | offset=80 | reg loop | Host ceil-div | former-core tail | 非 Micro=0 |
| 同上 | `ubTailOfTailRegLoop` | int64/64 | offset=88 | reg loop | Host ceil-div | final tail | 非 Micro=0 |

AICPU task 不消费上述 POD；其任务参数序列化 ABI 由 `ADD_TO_LAUNCHER_LIST_AICPU` 框架生成，固定仓不可见。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | empty | `aclnnClampCommon` | 无 | 无 | 无 | ACLNN op_api |
| R-002 | 2001100 | `ClipByValueTilingSingleDim` | `ClipByValueSigDimTilingData` | `clip_by_value_v2` | `ClipByValueV2Ass<T>` | ClipByValueV2 ascend950 CMake/OpDef |
| R-003 | AICPU | `ClipByValueV2` selector | Tensor descriptors | AICPU registry | `Compute -> DispatchByDtype -> DoCompute` | AICPU def + shared CMake glob |
| R-004～R-008 | Cast policy + 2001100 | CastTiling + Clip SingleDim | `CastTilingData` + Clip POD | `cast` -> `clip_by_value_v2` -> `cast` | CastDirect/Two/Micro + ASS | Cast/Clip ascend950 targets |

## Kernel 实现

### 公共执行流程

所有 AIV entry 都声明 `KERNEL_TYPE_AIV_ONLY`；Clip entry 还显式在 `g_coreType==AIC` 时返回，故可达图不含 Cube/AIC 或 MIX_CV。Clip ASS 每核只读取一次 min/max scalar，随后以双缓冲 pipeline 对 x 执行 `GM->inQueue->Maxs->Mins->outQueue->GM`。Cast 节点按独立 POD 重新分核，以中间 GM Tensor 连接；executor stream 顺序是跨 Kernel 可见性边。AICPU 路线直接遍历连续 Tensor，不使用 AIV UB。

### Route R-001：空张量

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/clip_by_value_v2/op_api/aclnn_clamp.cpp` | `aclnnClampCommon` | `self->IsEmpty()` | workspace=0，释放 executor，返回成功 |

#### Device 数据流

##### 函数级调用链

```text
aclnnInplaceClampMaxGetWorkspaceSize -> aclnnClampCommon -> empty return
```

##### 源码等价伪码

```text
if N == 0: workspaceSize=0; return SUCCESS
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| selfRef GM | 0 | 0 | 原 dtype/ND | caller | 无 | caller | 不访问 |

##### 分层内存布局

N/A：无 Device 节点、无 AIV UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | N/A | N/A | Host shortcut | N=0 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| shortcut | Host branch | API | caller | 空 Tensor | 无计算 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_api/aclnn_clamp.cpp` | `aclnnClampCommon` | 空输入零 workspace |
| Kernel entry/router | N/A | N/A | 不启动 |
| Device body/helper | N/A | N/A | 不执行 |

### Route R-002：ASS-AIV 同 dtype

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_clamp.cpp` | `ClampPromoteType` | promoteType 保持 self dtype | Cast 直通 |
| L0 | `op_api/clip_by_value.cpp` | `IsAiCoreSupport` | dtype in F16/F32/S32/S64/BF16 | AICore launcher |
| Tiling | `clip_by_value_single_dim_tiling.cpp` | `DetermineSigDimMode/GetTilingKey` | collapsed rank=1且min=max=1 | key 2001100 |
| Router | `op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` | `TILING_KEY_IS(2001100)` | `ClipByValueV2Ass<DTYPE_X>` |

#### Device 数据流

##### 函数级调用链

```text
ClipByValueTilingSingleDim::PostTiling
 -> clip_by_value_v2 -> ClipByValueV2Ass<T>::Init
 -> Process -> (CopyIn -> Compute[Maxs,Mins] -> CopyOut)*
 -> framework ViewCopy -> selfRef
```

##### 源码等价伪码

```text
coreBase = blockIdx * blockFormer * ubFormer
minValue=minGM[0]; maxValue=maxGM[0]
for each full tile: load x[coreBase+off:U]; y=max(x,minValue); y=min(y,maxValue); store
last tile uses ubTail on final core, otherwise U
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `coreBase+gmOffset` | len elements | T/ND | prior node | CopyIn | executor Tensor | effective bytes len*S |
| min/max GM | 0 | 1 each | T/scalar | ScalarToTensor | Init | executor | each loaded once/core |
| inQueueX | 0 | depth 2 * U*S | T | CopyIn | Compute | TPipe | Free after Maxs |
| outQueue | separate | depth 2 * U*S | T | Compute | CopyOut | TPipe | Maxs/Mins same tensor |
| y GM | `coreBase+gmOffset` | len | T/ND | CopyOut | ViewCopy | executor | tail exact length |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueueX · depth=2 · 2*U*S bytes"]:2
  outQ["outQueue · depth=2 · 2*U*S bytes"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*U*S + 2*U*S = 4*U*S <= UBytes
dynamic_ub=N/A
workspace node request = 16 MiB; body workspace use = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| outQueue tile | Maxs then in-place Mins | CopyOut | next AllocTensor | EnQue/DeQue/FreeTensor | Mins 完成后才能 EnQue |
| queue slot | CopyIn/Compute | Compute/CopyOut | double-buffer next iteration | queue ownership | depth=2 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn->Compute | EnQue/DeQue | DMA | Vector | x tile | T 原 dtype |
| Maxs->Mins | 同一 LocalTensor 程序顺序 | Maxs | Mins | yLocal | 无升精度；len 为有效元素 |
| Compute->CopyOut | EnQue/DeQue | Vector | DMA | y tile | 最后 tile=`ubTailNum` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `DoOpTiling/PostTiling` | U、core、tail、key、16 MiB |
| Kernel entry/router | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` | AIV-only；2001100 映射 ASS |
| Device body/helper | `.../clip_by_value_v2_ass.h` | `ClipByValueV2Ass<T>` | scalar cache、双队列、Maxs/Mins、tail |

### Route R-003：AICPU 同 dtype

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| L0 | `op_api/clip_by_value.cpp` | `ClipByValueV2` | `!IsAiCoreSupport(self)` | AICPU launcher |
| AICPU | `op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `DispatchByDtype` | S8/U8/S16/F64 | `DoCompute<T>` |

#### Device 数据流

##### 函数级调用链

```text
ClipByValueV2AiCpu -> ADD_TO_LAUNCHER_LIST_AICPU
 -> ClipByValueV2CpuKernel::Compute -> GetInputAndCheck -> DispatchByDtype
 -> DoCompute<T> -> KernelScalarBound<T> -> ClampScalar<T>
 -> framework ViewCopy
```

##### 源码等价伪码

```text
mn=min[0]; mx=max[0]
if N==1: y[0]=clamp(x[0],mn,mx)
else if N*sizeof(T)<200KiB: for i in [0,N): y[i]=clamp(...)
else ParallelFor([0,N), grain): same loop on disjoint [beg,end)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | i | N | T/ND | prior/AICPU | AICPU/ViewCopy | executor | disjoint CPU shards |
| scalar locals | register/stack | 2*T | T | min/max[0] | loop | AICPU worker | preload once per task |

##### 分层内存布局

N/A：AICPU CPU 线性遍历，不使用 AIV UB/TQue。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; AICPU OpDef workspace request=100 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| y shard `[beg,end)` | worker | ViewCopy | task 完成后 | ParallelFor join/task completion | shards 不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| workers | `ParallelFor` | shards | task completion | y ranges | double/integers 原 dtype；[beg,end) 无越界 |
| float special | `std::isnan` | load | ClampScalar | NaN | DOUBLE 任一输入 NaN 返回 quiet NaN |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_api/clip_by_value.cpp` | `ClipByValueV2AiCpu` | 添加 AICPU task |
| Kernel entry/router | `op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `Compute/DispatchByDtype` | 校验并按 dtype 分发 |
| Device body/helper | 同上 | `DoCompute/DispatchParallel/KernelScalarBound` | scalar preload、阈值、CPU 分片 |

### Route R-004：INT8 浮点标量三节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_clamp.cpp` | `ClampPromoteType` | self S8 + floating scalar | F32 |
| Cast policy | `cast_struct.h` | S8->F32 / F32->S8 | id 5 / id 7 | TwoCast / MicroCast |

#### Device 数据流

##### 函数级调用链

```text
cast<S8,F32>: CastTwo<int8,half,float>::Process
 -> clip_by_value_v2: ClipByValueV2Ass<float>::Process
 -> cast<F32,S8>: CastMicro<id7,float,int8,...>::Process
 -> ViewCopy
```

##### 源码等价伪码

```text
node1 per tile: S8 GM -> S8 UB -> cast half -> cast float -> F32 GM
node2: F32 GM -> max(-inf) -> min(maxScalarF32) -> F32 GM
node3 per reg tile: load F32 -> MicroAPI Cast to int32 with SAT/ZEROING/TRUNC -> packed S8 store
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Cast1 queues+mid | coreBase+tile | `2U*1 + U*2 + 2U*4` bytes | S8/F16/F32 | CastTwo | Clip | per Cast node | mid TBuf reused/tile |
| Clip queues | separate node | `4Uc*4` bytes | F32 | Clip | Cast2 | per Clip node | double buffer |
| Cast2 queues | coreBase+tile | `2U*4+2U*1` bytes | F32/S8 | MicroCast | ViewCopy | per Cast node | register mask tail |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  c1in["Cast1 inQ 2U·S8"] c1mid["mid U·F16"] c1out["outQ 2U·F32"]:2
  clip["Clip 4Uc·F32"]:2 c2in["Cast2 inQ 2U·F32"] c2out["outQ 2U·S8"]:2
```

##### 字节账与复用顺序

```text
Cast1 UB=2U*1+U*2+2U*4; Clip UB=4Uc*4; Cast2 UB=2U*4+2U*1
workspace graph: three 16MiB node requests; executor reuse/total fixed仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Cast1 mid TBuf | first Cast | second Cast | next tile | same Compute program order | second Cast 已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast1 | queue + GM stream edge | S8->F16->F32 | Clip | F32 Tensor | two `CAST_NONE`; exact tail |
| Cast2 | Micro mask | F32 | S8 GM | final values | int32 intermediate、TRUNC/SAT；mask zeroing |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling::DoOpTiling` | 每 Cast 独立分核/tail |
| Kernel entry/router | `math/cast/op_kernel/cast_apt.cpp` | `cast` id 5/7 | policy到 Two/Micro |
| Device body/helper | `math/cast/op_kernel/arch35/cast_impl.h` | `CastTwo/CastMicro::ComputeCast` | 中间 F16 与 masked register cast |

### Route R-005：UINT8 浮点标量三节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_clamp.cpp` | `ClampPromoteType` | self U8 + floating scalar | F32 |
| Cast policy | `cast_struct.h` | U8->F32 / F32->U8 | id 5 / id 9 | TwoCast / MicroCastDeinter |

#### Device 数据流

##### 函数级调用链

```text
CastTwo<uint8,half,float> -> ClipByValueV2Ass<float>
 -> CastMicro<id9,float,uint8,...>::ComputeCastDeinter -> ViewCopy
```

##### 源码等价伪码

```text
U8 -> F16 -> F32; clamp in F32; F32 register cast/deinterleave -> U8, masked tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Cast1/Clip/Cast2 | 各节点 coreBase+tile | 与 R-004 相同字节公式 | U8/F16/F32/U8 | 三节点 | 下一节点 | executor+TPipe | id9 使用双 load/deinter tail |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  c1["CastTwo: 2U·U8 + U·F16 + 2U·F32"]
  cl["Clip: 4Uc·F32"]
  c2["Micro: 2U·F32 + 2U·U8"]
```

##### 字节账与复用顺序

```text
UB 各节点独立，跨节点只经 GM Tensor；workspace total 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| mid F16 | Cast1 stage1 | Cast1 stage2 | next tile | Compute 顺序 | stage2 已读完 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast2 | Micro mask + deinterleave | F32 regs | U8 UB | packed lanes | tailRegLoop 来自 POD |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `cast_tiling.cpp` | `GetUbCopyStep/DoOpTiling` | id9 双-register loop |
| Kernel entry/router | `cast_apt.cpp` | `CastMicro<id9>` | Micro deinter route |
| Device body/helper | `cast_impl.h` | `ComputeCastDeinter` | masked U8 output |

### Route R-006：INT16 浮点标量三节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Cast policy | `cast_struct.h` | S16->F32 / F32->S16 | id1 / id7 | Direct / MicroCast |

#### Device 数据流

##### 函数级调用链

```text
CastDirect<int16,float> -> ClipByValueV2Ass<float> -> CastMicro<id7,float,int16> -> ViewCopy
```

##### 源码等价伪码

```text
S16 DMA -> Vector Cast F32 -> clamp F32 -> Micro masked Cast S16 -> ViewCopy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Cast1 | coreBase+tile | `2U*2+2U*4` | S16/F32 | Direct | Clip | TPipe | double buffer |
| Clip | coreBase+tile | `4Uc*4` | F32 | ASS | Cast2 | TPipe | double buffer |
| Cast2 | coreBase+tile | `2U*4+2U*2` | F32/S16 | Micro | ViewCopy | TPipe | masked tail |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  c1["Direct 2U·S16 + 2U·F32"] cl["Clip 4Uc·F32"] c2["Micro 2U·F32 + 2U·S16"]
```

##### 字节账与复用顺序

```text
各节点 UB_total 如图；跨节点不别名 UB，仅 GM 中间 Tensor
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Cast queue slot | DMA/Vector | Vector/DMA | next tile | FreeTensor | consumer 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast1/2 | queues + Micro mask | S16/F32 | next node | dtype edge | 输入 CAST_NONE；输出经 int32 intermediate 并截断 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `cast_tiling.cpp` | `GetUbFormer` | src/dst bits 决定 U |
| Kernel entry/router | `cast_apt.cpp` | id1/id7 | Direct/Micro |
| Device body/helper | `cast_impl.h` | `CastDirect/ComputeCast` | queue pipeline + mask |

### Route R-007：INT32 浮点标量三节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Cast policy | `cast_struct.h` | S32->F32 / F32->S32 | id1 RINT / id1 TRUNC | Direct/Direct |

#### Device 数据流

##### 函数级调用链

```text
CastDirect<int32,float> -> ClipByValueV2Ass<float> -> CastDirect<float,int32> -> ViewCopy
```

##### 源码等价伪码

```text
for each node/core/tile: DMA in -> Vector Cast or Maxs/Mins -> DMA out; final float->int32 uses CAST_TRUNC
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| each AIV node | coreBase+tile | `4U*4` bytes | S32/F32 | node | next node | TPipe/executor | depth 2 in/out |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  c1["Cast1 4U·4B"] cl["Clip 4Uc·4B"] c2["Cast2 4U·4B"]
```

##### 字节账与复用顺序

```text
每个节点 UB_total=4*U*4 bytes（各自 U）；workspace总复用策略固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| outQueue tile | Vector | DMA | next tile | DeQue/FreeTensor | DMA 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| final Cast | `Cast<int32,float>(CAST_TRUNC)` | clamp F32 | selfRef | integer result | 超范围/NaN具体 primitive 行为固定仓不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `cast_tiling.cpp` | `DoOpTiling` | 32-bit对称 tile |
| Kernel entry/router | `cast_apt.cpp` | `CastDirect` | id1 |
| Device body/helper | `cast_impl.h` | `CastDirect::Compute` | round mode 来自 policy |

### Route R-008：INT64 浮点标量三节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Cast policy | `cast_struct.h` | S64->F32 / F32->S64 | id1 RINT / id1 TRUNC | Direct/Direct |

#### Device 数据流

##### 函数级调用链

```text
CastDirect<int64,float> -> ClipByValueV2Ass<float> -> CastDirect<float,int64> -> ViewCopy
```

##### 源码等价伪码

```text
S64 tile -> Cast F32 -> clamp F32 -> Cast(TRUNC) S64 -> final writeback
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Cast1 | coreBase+tile | `2U*8+2U*4` | S64/F32 | Direct | Clip | TPipe | depth 2 |
| Clip | coreBase+tile | `4Uc*4` | F32 | ASS | Cast2 | TPipe | depth 2 |
| Cast2 | coreBase+tile | `2U*4+2U*8` | F32/S64 | Direct | ViewCopy | TPipe | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  c1["Cast1 2U·8 + 2U·4"] cl["Clip 4Uc·4"] c2["Cast2 2U·4 + 2U·8"]
```

##### 字节账与复用顺序

```text
Cast UB_total=24U bytes/node; Clip UB_total=16Uc bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| each queue slot | DMA/Vector | Vector/DMA | next tile | queue protocol | tail len exact |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| S64->F32 | Vector Cast RINT | input | Clip | intermediate | 大整数可能失去精确位；源码明确选择 F32 |
| F32->S64 | Vector Cast TRUNC | Clip | output | final integer | exact tail；异常转换行为固定仓不可见 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `cast_tiling.cpp` | `DoOpTiling` | 64/32 bit UB 配比 |
| Kernel entry/router | `cast_apt.cpp` | `CastDirect<int64,float>` / reverse | id1 |
| Device body/helper | `cast_impl.h` | `CastDirect::Process` | former/tail core pipeline |

## 算子特定优化

### Trick T-001：ASS 标量上下界每核只读取一次

#### 触发与源码依据

- 适用 Route：R-002、R-004～R-008 的 Clip 节点
- 触发条件：collapse 后 minDim=maxDim=1，key=2001100
- Golden 来源：[`conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h:84-94`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h#L84-L94)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t gmBlockOffset = blockIdx * tilingData->blockFormer * tilingData->ubFormer;
xGm.SetGlobalBuffer((__gm__ T*)x + gmBlockOffset);
minGm.SetGlobalBuffer((__gm__ T*)min);
maxGm.SetGlobalBuffer((__gm__ T*)max);
yGm.SetGlobalBuffer((__gm__ T*)y + gmBlockOffset);

minValue = minGm.GetValue(0);
maxValue = maxGm.GetValue(0);

pipe->InitBuffer(inQueueX, bufferNum, tilingData->ubFormer * sizeof(T));
pipe->InitBuffer(outQueue, bufferNum, tilingData->ubFormer * sizeof(T));
```

#### 机制与设计原因

上下界保存为成员 scalar，后续所有 tile 的 `Maxs/Mins` 都复用，不为每个 x tile 搬入两份广播 Tensor。与通用 broadcast 路线相比，这直接消除了边界 Tensor 的重复 GM→UB 搬运和 UB buffer。

#### 迁移边界

- 适用：两个边界均为单元素且所有核可读取同一 GM scalar。
- 前提：dtype 已统一，key 必须是 ASS；广播 Tensor 不能误用此法。
- 失效：任一边界非 scalar 时必须使用 SingleDim 非 ASS、UB broadcast 或 common 路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但 router 已把非 ASS key 分给不同 body，不能强行套用本模板。

### Trick T-002：低核利用率时反向缩小 tile

#### 触发与源码依据

- 适用 Route：所有 AIV Clip 节点
- 触发条件：初算 `blockNum < coreNum/2` 且 `ubFormer*dTypeSize*aliveNum > 8192`
- Golden 来源：[`conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp:271-288`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp#L271-L288)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
void ClipByValueTilingSingleDim::AdjustForLowUtilization(int64_t& ubFormer, int64_t& ubOuter, int64_t& ubTail,
                                                         int64_t& blockFormer, int64_t& blockTail, int64_t& blockNum)
{
    int64_t aliveNum = isAss ? ClipByValueSigDim::ASS_ALIVE_NODE_NUM : ClipByValueSigDim::SIG_DIM_ALIVE_NODE_NUM;
    if (static_cast<uint64_t>(blockNum) < (coreNum / ClipByValueSigDim::HALF_CORE_NUM_DIVIDE) &&
        (ubFormer * dTypeSize * aliveNum) > ClipByValueSigDim::PER_CORE_MIN_UB_BYTE) {
        int64_t dimPerCore = yDim * ClipByValueSigDim::LOW_UTILIZATION_DIM_MULTIPLIER / coreNum;
        int64_t alignDimPerCore = ((((dimPerCore * dTypeSize) + ClipByValueSigDim::CACHELINE_SIZE - 1) / ClipByValueSigDim::CACHELINE_SIZE) * ClipByValueSigDim::CACHELINE_SIZE) / dTypeSize;
        ubFormer = (ubFormer > alignDimPerCore) ? alignDimPerCore : ubFormer;

        int64_t lowestUbFormer =
            (((ClipByValueSigDim::PER_CORE_MIN_UB_BYTE / aliveNum) / ClipByValueSigDim::CACHELINE_SIZE) * ClipByValueSigDim::CACHELINE_SIZE) / dTypeSize;
        if (ubFormer < lowestUbFormer) {
            ubFormer = lowestUbFormer;
        }
        CalcBlockParams(ubFormer, ubOuter, ubTail, blockFormer, blockTail, blockNum);
    }
}
```

#### 机制与设计原因

初始 tile 只按 UB 容量最大化可能导致启用核数不足。此分支将 tile 缩到约 `2N/C` 并保持 128-byte 对齐，同时保留下限，重新分核；改变的是调度并行度，不改变计算顺序或输出。

#### 迁移边界

- 适用：纯 elementwise、tile 间无依赖、缩 tile 不增加额外跨 tile 状态。
- 前提：每核至少保留 8 KiB alive buffer 配额和 cacheline 对齐。
- 失效：有跨 tile reduction/状态或 tile 受固定 Vector repeat 限制时需重做证明。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-003：AICPU 标量预载与阈值串并行切换

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：两个边界 scalar；`N*sizeof(T)` 与 200 KiB 比较
- Golden 来源：[`conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp:154-172`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp#L154-L172)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
const int64_t data_num = ctx.Input(0)->NumElements();
const bool is_scalar = (ctx.Input(1)->NumElements() == 1);

// Degenerate form: single-element output computed directly.
if (__builtin_expect(data_num == 1, 0)) {
    y_ptr[0] = ClampScalar<T>(x_ptr[0], mn_ptr[0], mx_ptr[0]);
    return KERNEL_STATUS_OK;
}

constexpr int64_t elem_bytes = static_cast<int64_t>(sizeof(T));

if (is_scalar) {
    // Specialized branch: min/max broadcast; preload the scalars into local registers.
    const T mn_v = mn_ptr[0];
    const T mx_v = mx_ptr[0];
    auto body = [x_ptr, y_ptr, mn_v, mx_v](int64_t b, int64_t e) {
        KernelScalarBound<T>(x_ptr, y_ptr, b, e, mn_v, mx_v);
    };
    return DispatchParallel(ctx, "real-scalar", data_num, elem_bytes, kParallelBytesThresh, body);
}
```

#### 机制与设计原因

scalar 先复制到局部值，循环不再反复解引用边界 Tensor；小于 200 KiB 直接串行，达到阈值才按 CPU 数减 2 个保留核做 `ParallelFor`，减少小任务调度开销并限制大任务占满 CPU。

#### 迁移边界

- 适用：AICPU 线性 elementwise scalar broadcast。
- 前提：shard 写区间不重叠；阈值是本实现常量，不应跨平台照搬。
- 失效：边界为逐元素 Tensor时走 `KernelElemBound`，不能预载 scalar。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 空 | R-002 ASS-AIV | R-003 AICPU | R-004～R-008 多 AIV | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | N=0最高 | 同 dtype且 AICore dtype | 同 dtype非 AICore dtype | integer + floating scalar | API shortcut与 L0 dtype selector |
| 模板与 dtype | 无 | ASS<T> | DoCompute<T> | Cast policy + ASS<float> | promoteType |
| 循环与分核 | 无 | AIV core/UB tile | CPU serial/shard | 每 Kernel 独立 AIV tiling | 各节点资源不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 | 双 in/out queue | CPU locals | Direct/Micro queues；TwoCast mid TBuf | policy identity |
| 内存总占用与复用 | 0 | 4US | 无 AIV UB | 节点级公式见各 Route | Host selector保守容量 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 无 | DataCopyPad+Maxs+Mins | CPU loop | DataCopyPad+Cast/MicroAPI | dtype pair能力 |
| 同步、精度与 tail | 无 | queue，原 dtype | ParallelFor join | stream边+queue/mask；部分升F32 | scalar promotion |
| fallback/失败边界 | 成功 | tiling/router失败 | dtype/task失败 | 任一节点失败阻断后续 | executor错误传播 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档支持，但固定仓缺少对应 AICore Host/Tiling/Kernel 接线，本卡不发布 | 同 A2 | 支持且 source_closed |
| Host/Launcher | ACLNN 公共代码可见；实现平台闭包缺失 | 同 A2 | RegBase 类型推导；AICore/AICPU selector闭合 |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | Clip/Cast AIV-only arch35 + AICPU fallback |

## 限制与失败边界

- `selfRef` 或 `clipValueMax` 为空、dtype 不支持、维数大于 8、类型推导失败或推导 dtype 不能 Cast 回 self 时第一段报错。
- AIV Clip 使用 key 2001100 的前提是本入口确实生成两个 scalar Tensor；不得把 Tensor max 变体或非 scalar broadcast 合并到本卡。
- `INT64 + floating scalar` 在 F32 中 clamp 后再转回 INT64，存在源码明确的中间精度收窄；具体越界/NaN 到整数的硬件 primitive 结果固定仓不可见。
- executor 对多个 16 MiB 节点 workspace 的全图对齐和复用算法、AICPU task 序列化 ABI、Contiguous/ViewCopy 内部 launch 细节属于固定仓框架边界。
- A2/A3 虽有公开支持声明，但此 Commit 的 `ClipByValueV2` CMake、OpDef、binary config、Host tiling与 Kernel 只闭合 ascend950；因此不生成无后缀正式卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/clip_by_value_v2/docs/aclnnClampMax&aclnnInplaceClampMax.md` | 产品表/函数原型 | 950公开支持、语义、dtype/shape |
| 入口声明/约束 | `conversion/clip_by_value_v2/op_api/aclnn_clamp.h` | `aclnnInplaceClampMax*` | C ABI |
| 入口实现 | `conversion/clip_by_value_v2/op_api/aclnn_clamp.cpp` | `aclnnInplaceClampMaxGetWorkspaceSize/aclnnClampCommon` | 校验、推导、节点顺序、原地写回 |
| L0 selector | `conversion/clip_by_value_v2/op_api/clip_by_value.cpp` | `ClipByValueV2` | AICore/AICPU分发 |
| Host/Tiling | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.h` | POD/class | ASS POD ABI |
| Host/Tiling | `conversion/clip_by_value/op_host/arch35/clip_by_value_single_dim_tiling.cpp` | `DetermineSigDimMode/DoOpTiling` | key、UB、core、tail、workspace |
| Host/Tiling | `conversion/clip_by_value_v2/op_host/arch35/clip_by_value_v2_tiling.cpp` | tiling registry | AIV core/UB平台producer |
| Kernel entry/body | `conversion/clip_by_value_v2/op_kernel/clip_by_value_v2_apt.cpp` | `clip_by_value_v2` | AIV-only router |
| Kernel entry/body | `conversion/clip_by_value_v2/op_kernel/arch35/clip_by_value_v2_ass.h` | `ClipByValueV2Ass` | scalar/queue/Maxs/Mins/tail |
| Kernel entry/body | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu.cpp` | `Compute/DoCompute` | AICPU fallback |
| 构建接线 | `conversion/clip_by_value_v2/op_kernel_aicpu/clip_by_value_v2_aicpu_def.cpp` | `ApplyMathAicpuDefaultCfg` | AICPU SO/workspace注册 |
| L0 selector | `math/cast/op_api/cast.cpp` | `Cast/IsAiCoreSupport` | zero-node同dtype与950 Cast分发 |
| Host/Tiling | `math/cast/op_host/arch35/cast_tiling.h` | `CastTilingData` | Cast POD |
| Host/Tiling | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling::DoOpTiling` | Cast分核/UB/tail/workspace |
| Kernel entry/body | `math/cast/op_kernel/arch35/cast_struct.h` | `GetCastPolicy` | 五组整数↔F32 policy |
| Kernel entry/body | `math/cast/op_kernel/cast_apt.cpp` | `cast` | AIV-only模板router |
| Kernel entry/body | `math/cast/op_kernel/arch35/cast_impl.h` | `CastDirect/CastTwo/CastMicro` | 搬运、计算、资源、mask |
| 构建接线 | `conversion/clip_by_value_v2/op_host/clip_by_value_v2_def.cpp` | `ClipByValueV2` | 950 AICore dtype/文件名 |
| 构建接线 | `conversion/clip_by_value_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950/arch35 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Clip 72-byte、Cast 96-byte int64字段序列 | 字段顺序、单位、inactive Micro字段 | 两个 TILING_DATA_DEF |
| 调用顺序 | Cast可为零节点；多节点必须 Cast-in→Clip→Cast-out→ViewCopy | 不得直接把 Clip 输出别名 selfRef | `aclnnClampCommon` |
| workspace/内存 | ASS 4US；Cast按 policy分配 queues/TBuf | depth=2必须计入；executor workspace不能简单求和 | `InitBuffer/GetWorkspaceSize` |
| 同步 | 节点内 queue，节点间 executor stream | GM producer完成前 consumer不可启动 | `CommonOpExecutorRun`/queue API |
| 精度与尾块 | scalar promotion、S64/S32回转TRUNC、各节点独立 tail | 不得省略回转 Cast 或以 padded len 计算 | `GetCastPolicy/Process` |
| 失败边界 | selector/tiling/launcher任一失败阻断后续 | 不得 fallback 到未注册 dtype或未知 key | L0 checks/router |
