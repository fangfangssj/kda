---
type: Operator
title: "InplaceLtScalar 950 ACLNN 算子知识"
description: "说明 aclnnInplaceLtScalar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/docs/aclnnLtScalar&aclnnInplaceLtScalar.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_api/aclnn_lt_scalar.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_api/aclnn_lt_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_api/less.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_host/arch35/less_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_host/less_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_kernel/less_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_kernel/arch35/less_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_kernel_aicpu/less_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/cast_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_impl.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel_aicpu/cast_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/CMakeLists.txt
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
created_at: "2026-08-25T11:16:43Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceLtScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceLtScalarGetWorkspaceSize` + `aclnnInplaceLtScalar`，公开 ACLNN 两段式无后缀入口 |
| 计算语义 | `selfRef[i] = cast<bool,selfRef.dtype>(selfRef_before[i] < other)`；原地提交由最后的 `ViewCopy` 完成，Less 核心不直接覆盖输入 |
| 输入输出 | `selfRef`：ND、0～8 维、可非连续；DOUBLE/FLOAT/FLOAT16/BFLOAT16/INT32/UINT32/INT64/UINT64/INT16/UINT16/INT8/UINT8/BOOL；`other` 为 Host scalar |
| 执行类型 | Less 核心为 `AIV_VECTOR` 或 AICPU fallback；Cast 节点为 `AIV_VECTOR` 或 AICPU fallback；无 Cube/AIC、MIX_CV、SIMT、通信节点 |
| Kernel 路线 | R-001 BOOL 快路；R-002 AIV Less+AIV Cast；R-003 AIV Less+AICPU Cast；R-004 AICPU Less+AIV Cast；R-005 AICPU Less+AICPU Cast；R-006 含输入 promote Cast 的组合路线 |
| 核心资源 | AIV Less 使用静态 UB buffer-id 池；AIV Cast 使用双缓冲 TQue/TBuf 或寄存器策略；AICPU 使用 CPU shard；executor workspace 聚合多个节点 |
| 来源闭包 | `source_closed`：公开入口、950 promote/分发、Less/Cast 的 AIV 与 AICPU body、ATVOSS POD/调度及构建接线闭合；Contiguous/ViewCopy/executor 为明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `N=selfRef.numel()`，第一段先计算 950 promote dtype `P`：scalar 的浮点默认类型先归一成 FLOAT；`CombineCategoriesWithComplex(selfRef.dtype, scalarDefaultDtype)` 决定 `P`，BOOL promote 再改为 UINT8。非空时执行：

```text
x = contiguous(selfRef_before)
xp = cast(x, P)
s  = tensor(other, P)
b[i] = (xp[i] < s[0])              # BOOL 临时结果
r[i] = cast<bool,selfRef.dtype>(b[i])
selfRef = ViewCopy(r, selfRef)
```

因此非 BOOL 输出保存数值 `0/1`，不是把输出对象改成 BOOL；输入旧值必须活到 Less 消费完成。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND；0～8 维；允许非连续 | DOUBLE、FLOAT、FLOAT16、BFLOAT16、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL | 输入/输出别名；shape 不变；空 Tensor 成功短路 |
| `other` | Host scalar，物化后为单元素 Tensor | 同公开支持集 | 必须能与 `selfRef` 推导并转换到 `P`；浮点 scalar 默认按 FLOAT 参与 950 推导 |
| `workspaceSize` | 标量输出 | `uint64_t` | executor 聚合 Contiguous、可选输入 Cast、Less、可选结果 Cast、ViewCopy 的 workspace |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段拥有建图临时量，第二段绑定 workspace 与 stream 执行 |

### 入口与版本

公开头文件只声明无版本后缀的 `aclnnInplaceLtScalar*`。实现将 `selfRef` 经 `const_cast` 同时作为 `aclnnLtScalarGetWorkspaceSize` 的输入与输出，因此共享非原地建图 body，但最终目标、生命周期和错误语义仍是独立的原地逻辑算子。公开文档明确支持 Ascend 950PR/950DT；Less 与 Cast 的 OpDef、CMake 均接到 `ascend950/arch35`。固定仓没有更新版本入口，本卡覆盖该实际公开版本。

## Host/Launcher 执行流程

### 参数校验与前处理

真实调用顺序：

1. `aclnnInplaceLtScalarGetWorkspaceSize` 令 `out=selfRef`，进入 `aclnnLtScalarGetWorkspaceSize`；RegBase 选择 V35 body。
2. 创建 executor；检查 `selfRef/other` 非空。`GetScalarDefaultDtype` 与 `CombineCategoriesWithComplex` 计算 `P`，BOOL 改为 UINT8。
3. `CheckParamsScalar` 检查输入/输出 dtype、promote 可达、三次 Cast 合法、rank≤8，以及 `selfRef` 与别名输出 shape 相同。
4. `N==0` 时写 `workspaceSize=0`、交出 executor 并成功返回，不添加 Device 节点。
5. `Contiguous(selfRef)`；`ConvertToTensor(other,P)`；`Cast(selfContiguous,P)`。同 dtype Cast 直接返回输入，不创建节点。
6. `l0op::Less` 按 `P` 选择 AICore 或 AICPU，产生 BOOL 临时 Tensor。
7. `Cast(ltOpOut,selfRef.dtype)`；BOOL 输出时同 dtype无节点，其余按 RegBase Cast pair 选择 AICore/AICPU。
8. `ViewCopy(castOut,selfRef)` 回写可能非连续的原地目标；任一步返回空指针即阻断后续建图。
9. `GetWorkspaceSize()` 返回聚合值；第二段 `CommonOpExecutorRun` 在同一 stream 上依序执行。

### 参数、Workspace 与生命周期

Less AIV 的 ATVOSS Tiling 将核心 workspace slice 声明为 `16 MiB`，entry 不读取该指针；Cast AIV Tiling同样声明 `16 MiB`。公开 `workspaceSize` 是 executor 聚合结果，不能把多节点路线简化为某一个 slice。AICPU shard 不使用 AIV UB。连续化临时、promote 临时、BOOL Less 输出及结果 Cast 输出均由 executor 持有，最后一个消费者完成后才可释放；`selfRef` 旧存储在 ViewCopy 前不得覆盖。

### 启动图与失败传播

```text
[framework] Contiguous(selfRef)
  -> [optional] Cast selfRef.dtype -> P (AIV or AICPU)
  -> Less<P> (AIV less<202> or AICPU LessCpuKernel)
  -> [optional] Cast BOOL -> selfRef.dtype (AIV or AICPU)
  -> [framework] ViewCopy(result, selfRef)
```

| 节点/边 | entry 与参数 | blockDim/POD | 交接、ordering、owner/lifetime | 失败传播 |
|---|---|---|---|---|
| 输入 Cast（可选） | `cast<0>(x,xp,workspace,tiling)` 或 `CastCpuKernel` | AIV：`CastTilingData`; AICPU：CPU shard | `xp` 为 `P/ND`，executor 所有，活到 Less 完成 | launcher/CPU 错误阻断 Less |
| Less AIV | `less<202>(xp,s,b,workspace,tiling)` | `blockNum`; `BroadcastOneDimTilingDataAdvance` | 写 BOOL 连续临时 `b`；同 executor/stream 排在输入 Cast 后 | Tiling/launcher 失败阻断结果 Cast |
| Less AICPU | `LessCpuKernel::Compute` | CPU 核数由 `GetCPUNum-kResvCpuNum` 与元素数限定 | 逐 shard 写 BOOL `b` | 参数/broadcast/ParallelFor 错误阻断后续 |
| 结果 Cast（可选） | `cast<0>(b,r,workspace,tiling)` 或 `CastCpuKernel` | 同 Cast 路线 | `r` dtype 为 `selfRef.dtype`，活到 ViewCopy | 失败阻断回写 |
| ViewCopy | 框架边界 | 固定仓不可见 | 从连续 `r/b` 写回原始 view；owner 为 executor | 第二段传播执行错误 |

## Route 与参数设计

### 分核与资源计算

Less scalar broadcast 经合轴成为一维 `N`，`InputSize+OutputSize=3<=4`、`VarSize=0`，故固定走 `schMode=202`。令 `s=max dtype bytes`、`B=OpDag::GetBufferNum<true,true>()`、`C=AIV core count`、`U=UB bytes`：

```text
tile0 = floor((U/B)/512)*512/s
tile = 512/min_dtype_bytes,                    if C*(512/min_dtype_bytes)>N
     = floor((N/C)/(512/s))*(512/s),           else if C*tile0>N
     = tile0,                                  otherwise
ubOuter = ceil(N/tile)
blockFormer = ceil(ubOuter/C); if blockFormer==1: blockFormer=2
blockNum = ceil(ubOuter/blockFormer)
```

Cast AIV 令输入/输出位宽为 `bi/bo`：`coreNum=min(platformC,ceil(N*bi/(4KiB*8)))`，`blockFormer=align_up(ceil(N/coreNum),8)`，再计算 former/tail block 的 UB loop。直接/through/two-cast模板的 `ubFormer=floor((((U-reserve)*4)/(bi+bo))/VL_elems)*VL_elems`；具体 two-cast还计入中间位宽。所有这些量以元素计，workspace 以字节计。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 BOOL 快路 | Less key 202 | 非空；`selfRef=BOOL` 且推导 `P=UINT8`（BOOL/BOOL 在源码被改为 UINT8）；输入有 BOOL→UINT8 through Cast，Less AIV，结果 BOOL→BOOL无 Cast | 其它 dtype 继续下表 |
| 2 | R-002 AIV Less+AIV 结果 Cast | Less 202；Cast key 0 | `P` 在 `{INT64,UINT64,INT32,INT8,UINT8,FLOAT,FLOAT16,BFLOAT16,BOOL,DOUBLE}`，且 BOOL→self pair 在 RegBase AICore表；无结果 Cast仅当 self BOOL | 不支持的结果 pair 到 R-003 |
| 3 | R-003 AIV Less+AICPU 结果 Cast | Less 202；AICPU Cast | Less 可走 AIV，结果 BOOL→`{UINT16,UINT32,UINT64,DOUBLE}` 不在 RegBase AICore pair | AICPU Cast 不支持则建图失败 |
| 4 | R-004 AICPU Less+AIV 结果 Cast | AICPU Less；Cast key 0 | `P=INT16`，且 `selfRef=INT16`；Less RegBase表无 INT16，BOOL→INT16 有 AIV Cast | AICPU Less失败则终止 |
| 5 | R-005 AICPU Less+AICPU 结果 Cast | 两个 AICPU node | `P in {UINT16,UINT32}` 且 `selfRef` 也为对应 dtype；两节点均不在各自 RegBase AICore表 | CPU错误终止 |
| 6 | R-006 含 promote Cast | Cast key 0或AICPU；随后 R-002～R-005 的 Less/结果节点 | `selfRef.dtype != P`：整数/BOOL输入遇浮点 scalar，或 BOOL 输入遇非 BOOL整数 scalar | 输入 Cast pair 不支持则第一段失败 |

```text
if null/dtype/promote/rank invalid -> ACLNN error
if N==0 -> success, no Device node
P = CombineCategoriesWithComplex(self.dtype, ScalarDefault(other.dtype)); BOOL -> UINT8
if self.dtype != P -> add Cast(self->P), select AIV else AICPU
select Less: P in RegBaseLessSet ? AIV less<202> : AICPU LessCpuKernel
if self.dtype != BOOL -> add Cast(BOOL->self.dtype), select AIV else AICPU
add ViewCopy -> selfRef
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 BOOL 快路 | Less 202 | 950、非空、self/other BOOL | CastThrough BOOL→UINT8 → Less AIV → ViewCopy BOOL | `CastThrough<uint8_t>`；`LessCompute<uint8_t>` | Cast TQue DB；Less静态 buffer-id | Cast/Less各自 former/tail；无 CPU fallback after launch |
| R-002 AIV/AIV | 202 + 0 | 950；P在 Less AIV集；结果 pair在 Cast AIV表 | 可选输入 AIV Cast → Less AIV → AIV Cast → ViewCopy | CastDirect/CastThrough/CastTwo；`BroadcastOneDimAdvanceSch<LessCompute<T>>` | 各节点独立 UB，GM 临时边串联 | 每节点独立 tail；selector不满足转 R-003/006 |
| R-003 AIV/CPU | 202 | 结果目标 UINT16/UINT32/UINT64/DOUBLE | 可选输入 Cast → Less AIV → CastCpuKernel → ViewCopy | Less同上；CPU `CastTask<bool,D>` | Less UB；CPU shard；stream ordering | CPU shard `[start,end)` 精确覆盖 |
| R-004 CPU/AIV | AICPU + 0 | P/self INT16 | LessCpuKernel → CastTwo BOOL→INT16 → ViewCopy | `SpecialCompute/BcastCompute<int16_t>`；`CastTwo<int8,int32,int16>` | CPU shard后接AIV DB | CPU阈值分片；Cast tail |
| R-005 CPU/CPU | AICPU | P/self UINT16或UINT32 | LessCpuKernel → CastCpuKernel → ViewCopy | CPU比较；`CastTask<bool,D>` | 无AIV UB，两个CPU任务有序 | 各ParallelFor shard |
| R-006 promote组合 | 0/AICPU + 上述 | `self!=P` | 输入 Cast → R-002～R-005余下节点 | pair对应 Cast policy或CastTask | 临时 `P` tensor增加一条GM边 | 任一前置节点失败阻断后续 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Less AIV dtype | 两输入均命中 RegBase set且同 dtype | arch35 Less binary/OpDef只实例化十种 dtype | R-001～R-003/R-006 | `LessAiCpu` | `less.cpp::IsAiCoreSupport`; `LessTiling::DoOpTiling` |
| Cast pair | pair存在 `REGBASE_AICORE_DTYPE_SUPPORT_LIST` | 只有对应 `GetCastPolicy` 才能实例化 AIV body | R-001～R-004/R-006 | `CastAiCpu`，再检查CPU支持集 | `cast.cpp::IsAiCoreSupport/Cast` |
| Less one-dim advance | 合轴rank=1、连续、I/O≤4、无Var | 使用16-byte紧凑POD和静态UB buffer-id | 所有AIV Less | 通用broadcast；本scalar链不可达 | `BroadcastBaseTiling::DoTiling` |
| 512-byte tile floor | `floor((U/B)/512)>0` | ping/pong buffer-id区域按512字节切片 | AIV Less | Tiling失败 | `GetOneDimUBFormerAdvance` |
| Cast UB reserve | `U>reserve` 且计算出的 `ubFormer>0` | 双缓冲/中间TBuf必须同时容纳 | AIV Cast | Tiling失败 | `CastTiling::GetUbFormer` |
| AICPU并行阈值 | Less：同shape≥7168、broadcast≥2048；Cast：按可用CPU直接分片 | 限制CPU任务粒度 | R-003～R-006 | 小Less串行；Cast仍ParallelFor | `LessCpuKernel::NoBcastCompute/BcastCompute`; `CastCpuKernel::ExecuteCast` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen` | int64/64 | sizeof=16, alignof=8, offset=0 | 元素 | `DoOneDimFormerTilingAdvance: N` | Less scheduler算ubOuter/tail | N>0 |
| 同上 | `tileNum` | int32/32 | offset=8 | 元素/tile | UB公式与缩核分支 | `blockLen/tile loop` | >0；无显式窄化检查 |
| 同上 | `blockNum` | int16/16 | offset=12 | AIV core | `ceil(ubOuter/blockFormer)` | 尾核判断 | 无显式int16上界检查 |
| 同上 | `scalarFlag` | int16/16 | offset=14 | 输入bit位图 | scalar输入置bit1 | x2只填充ping/pong两相 | x2始终scalar |
| `CastTilingData` | `blockNum` | int64/64 | 字段序列第1项；宏生成精确offset固定仓不可见 | core | `DoOpTiling` | Cast各body尾核判断 | >0 |
| 同上 | `ubFormer` | int64/64 | 第2项 | 元素 | `GetUbFormer` | Queue/TBuf容量与loop | >0且VL对齐 |
| 同上 | `blockFormer` | int64/64 | 第3项 | 元素/core | align-up到8 | GM block offset | 尾核可少于该值 |
| 同上 | `ubLoopOfFormerBlock` | int64/64 | 第4项 | loop | Host ceil-div | 非尾核Process | ≥1 |
| 同上 | `ubLoopOfTailBlock` | int64/64 | 第5项 | loop | Host ceil-div | 尾核Process | ≥1 |
| 同上 | `ubTailOfFormerBlock` | int64/64 | 第6项 | 元素 | Host余数归一 | 非尾核末loop | 1..ubFormer |
| 同上 | `ubTailOfTailBlock` | int64/64 | 第7项 | 元素 | Host余数归一 | 尾核末loop | 1..ubFormer |
| 同上 | `regCopyInStep` | int64/64 | 第8项 | 元素/register step | policy load-dist producer | Micro Cast；普通模板为0 | inactive时0 |
| 同上 | `regCopyOutStep` | int64/64 | 第9项 | 元素/register step | policy store-dist producer | Micro Cast；普通模板为0 | inactive时0 |
| 同上 | `ubFormerRegLoop` | int64/64 | 第10项 | register loop | Host按VL/bit宽ceil-div | Micro Cast | 普通模板0 |
| 同上 | `ubTailOfFormerRegLoop` | int64/64 | 第11项 | register loop | Host | Micro Cast former tail | 普通模板0 |
| 同上 | `ubTailOfTailRegLoop` | int64/64 | 第12项 | register loop | Host | Micro Cast tail block | 普通模板0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～R-003 | Less 202 | `LessTiling::DoOpTiling` + ATVOSS | one-dim POD | `less<202>` | Less DAG + one-dim scheduler + Vec | Less ascend950/arch35 |
| R-004～R-005 | `LessAiCpu` | L0 selector；CPU自己分片 | CPU context | `LessCpuKernel::Compute` | SpecialCompute/NoBcast/BcastCompute | AICPU注册 + Less模块 |
| R-001/R-002/R-004/R-006 Cast AIV | Cast 0 + pair policy | `CastTiling::GetShapeAttrsInfo/DoOpTiling` | `CastTilingData` | `cast<0>` | Direct/Through/Two/Micro具体policy | Cast ascend950/arch35 |
| R-003/R-005/R-006 Cast CPU | `CastAiCpu` | L0 pair selector | CPU tensor context | `CastCpuKernel::Compute` | `SetMap` + `CastTask<S,D>` + ParallelFor | AICPU注册 + Cast模块 |

## Kernel 实现

### 公共执行流程

AIV Less entry固定声明 `KERNEL_TYPE_AIV_ONLY`（由 `BroadcastSch`），BOOL物理输入映射为int8，其余以 `DTYPE_X1` 构造 DAG。schMode 202逐核执行 CopyInBrc→Compare(LT)→Duplicate(1/0)→Select→CopyOut。AIV Cast entry同样声明AIV-only，pair表在编译期选 Direct、Through、Two或Micro body。AICPU Less与Cast都在CPU context上按 `[start,end)` shard处理。所有节点通过 executor/stream顺序交接GM临时量，无跨节点原子。

### Route R-001：BOOL 快路

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/less/op_api/aclnn_lt_scalar.cpp` | V35 body | promote为BOOL后改UINT8 | BOOL输入先Cast到UINT8 |
| Cast selector | `math/cast/op_api/cast.cpp` | `Cast` | BOOL→UINT8在RegBase AICore pair | `CastThrough<uint8_t>` |
| Less selector | `math/less/op_api/less.cpp` | `IsAiCoreSupport` | UINT8在RegBase set | LessAiCore |
| Tiling/router | `less_tiling_arch35.cpp` / `less_apt.cpp` | UINT8分支 / `less<202>` | scalar一维 | Less AIV |

#### Device 数据流

##### 函数级调用链

```text
CastAiCore -> cast<0> -> CastThrough<uint8_t>::Process
  -> [GM edge] LessAiCore -> less<202>
    -> BroadcastOneDimAdvanceSch<LessCompute<uint8_t>>::Process
      -> CopyInBrc(x) -> Duplicate(scalar) -> Compare(LT) -> Duplicate(1/0) -> Select -> CopyOut
  -> [framework] ViewCopy(bool result,selfRef)
```

##### 源码等价伪码

```text
# 伪码
cast each bool x to uint8 (0/1)
for each AIV core and tile:
  xUB = copy(xPromoted[offset:offset+len])
  sUB[ping] = duplicate(scalar, len) only while initializing ping/pong
  mask = Compare(xUB,sUB,CMPMODE=0/LT,len)
  yUB = Select(mask, Duplicate(1), Duplicate(0), SELMODE=2, len)
  copy yUB[0:len] to BOOL GM
ViewCopy to original selfRef view
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| promoted GM | core/tile element offset | N bytes | UINT8 ND | CastThrough | Less CopyIn | executor至Less完成 | Cast独立DB |
| Less x local | `bufId*tile` | tile bytes | uint8 local | DataCopyPad | Compare | 当前ping/pong | 512B tile对齐 |
| scalar local | scalar bufId | tile bytes×2相 | uint8 local | GM GetValue+Duplicate | Compare | 核内 | 两相初始化后复用 |
| mask/one/zero/select local | DAG buffer ids | `B`中相应区域×tile | uint8 local | Compare/Duplicate/Select | 下游节点 | 当前tile | LEVEL_2着色复用 |
| BOOL GM | core/tile offset | len bytes | BOOL ND | CopyOut | ViewCopy | executor至回写 | 只写有效len |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inPP["Less MTE2 x/scalar · ping/pong · 2*M2*tile"]:2
  calc["Compare/one/zero/select · Temp*tile"]:2
  outPP["MTE3 bool · ping/pong"]:1
```

##### 字节账与复用顺序

```text
Less_UB_total = B_less * tileNum * 1 byte <= U
CastThrough_UB_total = 2*ubFormer*sizeof(uint8) input queue + 2*ubFormer*sizeof(uint8) output queue
dynamic_ub = N/A
workspace_total = executor聚合；每个AIV节点声明16MiB slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Less x ping/pong | CopyIn | Compare | 同parity下下tile | Get/ReleaseTensor | Compare读完后覆盖 |
| scalar ping/pong | 前两tileDuplicate | 所有同parity tile | 核结束 | `isFirstScalarTime` | scalar全Kernel不变 |
| DAG temp | Compare/Duplicate/Select | 后继节点/CopyOut | 生命周期着色后下一节点 | Get/ReleaseTensor | DAG拓扑顺序 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast输入 | TQue DB + Copy | BOOL GM | Less | promoted GM | 0/1位模式不变 |
| Less搬入 | DataCopyPad/GetTensor | GM | Compare | x/scalar local | len精确控制tail |
| 比较/物化 | Compare mode0 + Select mode2 | x/scalar | CopyOut | mask/result | 输出严格0/1；无round |
| 回写 | executor stream + ViewCopy | BOOL临时 | selfRef | 原地目标 | 核心读完旧值后写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/less/op_host/arch35/less_tiling_arch35.cpp` | `LessTiling::DoOpTiling` UINT8 | Less DAG与key |
| Kernel entry/router | `math/less/op_kernel/less_apt.cpp` | `less<202>` | dtype绑定与scheduler |
| Device body/helper | `math/less/op_kernel/arch35/less_dag.h` | `LessCompute<uint8_t>` | 比较并Select 0/1 |
| 公共scheduler | `opbase/.../broadcast_sch_one_dim_advance.h` | `Process/CopyIn/Run/CopyOut` | 分核、DB、scalar cache、tail |

### Route R-002：AIV Less + AIV 结果 Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lt_scalar.cpp` | V35 body | 非空并完成promote | 可选输入Cast→Less→结果Cast |
| Less selector | `less.cpp` | RegBase set | P在十种AIV类型 | `LessAiCore` |
| Cast selector | `cast.cpp` | RegBase pair table | BOOL→self pair命中 | `CastAiCore` |
| Cast policy | `cast_struct.h` | `GetCastPolicy` | pair决定Direct/Through/Two | `cast<0>` |

#### Device 数据流

##### 函数级调用链

```text
[optional Cast self->P] -> less<202> -> cast<0>(BOOL->D) -> ViewCopy
less: BroadcastOneDimAdvanceSch -> CopyIn/Compare/Select/CopyOut
cast: CastDirect | CastThrough | CastTwo -> Init -> Process -> CopyIn/Compute/CopyOut
```

##### 源码等价伪码

```text
# 伪码；D=selfRef.dtype
for Less tile: b = select(xP < scalarP, 1, 0)
for Cast core:
  loops = lastCore ? ubLoopTail : ubLoopFormer
  len = lastLoop ? correspondingTail : ubFormer
  bUB = DataCopyPad(bGM + coreBase + offset, len)
  dUB = CastPolicy<bool,D>(bUB, policyRoundModes, len)
  DataCopyPad(dGM + coreBase + offset, dUB, len*sizeof(D))
ViewCopy(dGM,selfRef)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Less BOOL GM | element offset | N bytes | BOOL | Less | Cast | executor | stream有序 |
| Cast input queue | local | depth2×ubFormer bytes | int8 | CopyIn | Cast policy | Cast node | FreeTensor后复用 |
| Cast mid TBuf | local | policy相关 `ubFormer*sizeof(M)` | intermediate | first Cast | second Cast | CastTwo only | Compute内复用 |
| Cast output queue | local | depth2×ubFormer×sizeof(D) | D | policy | CopyOut | Cast node | DB |
| D GM | coreBase+loop offset | N×sizeof(D) | D ND | Cast | ViewCopy | executor | 只写有效tail |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["Cast inQ · depth=2 · 2*ubFormer bytes"]:1
  mid["optional midBuf · ubFormer*sizeof(M)"]:1
  outQ["Cast outQ · depth=2 · 2*ubFormer*sizeof(D)"]:2
```

##### 字节账与复用顺序

```text
CastDirect/Through UB = 2*ubFormer*sizeof(S) + 2*ubFormer*sizeof(D)
CastTwo UB = above + ubFormer*sizeof(M)
Less UB = B_less*tile*max(sizeof(P),1)
workspace_total = executor聚合；节点间GM临时不别名
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Cast inQ slot | CopyIn | Compute | 下一DB迭代 | DeQue/FreeTensor | 消费后覆盖 |
| Cast midBuf | first Cast | second Cast | 下一tile | 同一Compute顺序 | 两次Cast不可重排 |
| Cast outQ slot | Compute | CopyOut | 下一DB迭代 | DeQue/FreeTensor | CopyOut完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Less | Get/Release buffer ids | inputs | bool GM | tile locals | Compare LT；Select生成0/1 |
| Cast | TQue EnQue/DeQue | CopyIn | Compute/CopyOut | DB queues | BOOL只含0/1，转换到D无溢出；policy round不改变0/1 |
| tail | POD former/tail | Host | Process末loop | GM有效区 | 两节点分别只处理真实len |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling::DoOpTiling` | Cast分核、UB与POD |
| Kernel entry/router | `math/cast/op_kernel/cast_apt.cpp` | `cast<0>` | pair policy到body |
| Device body/helper | `math/cast/op_kernel/arch35/cast_impl.h` | `CastDirect/Through/Two` | Queue、Cast、tail |

### Route R-003：AIV Less + AICPU 结果 Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Less selector | `less.cpp` | `IsAiCoreSupport` | P命中RegBase set | AIV Less |
| Cast selector | `cast.cpp` | `IsAiCoreSupport/IsAiCpuSupport` | BOOL→UINT16/UINT32/UINT64/DOUBLE未命中AIV、命中CPU集 | CastAiCpu |
| CPU body | `cast_aicpu.cpp` | `SetBoolMap/ExecuteCast` | pair查到 `CastTask<bool,D>` | ParallelFor |

#### Device 数据流

##### 函数级调用链

```text
less<202> -> BOOL GM -> CastCpuKernel::Compute
  -> ValidateTensors -> ValidateDataType -> SetBoolMap
  -> ExecuteCast -> ParallelFor -> CastTask<bool,D>
  -> ViewCopy
```

##### 源码等价伪码

```text
# 伪码
b = LessAIV(...)
workers = min(max(1,GetCPUNum-kResvCpuNum),N)
parallel shard [start,end): for i: d[i] = static_cast<D>(b[i])
ViewCopy(d,selfRef)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| BOOL GM | i | N bytes | BOOL | Less AIV | CPU shard | executor | stream task boundary |
| D GM | i*sizeof(D) | N*sizeof(D) | D | CastTask | ViewCopy | executor | disjoint shards |
| CPU local | scalar loop vars | shard extent | registers | CPU load | CPU store | shard | 无UB/TQue |

##### 分层内存布局

N/A：结果 Cast 在 AICPU 上逐 shard 直接读写 GM；实际替代路径是 CPU registers 与互不重叠的 `[start,end)`。

##### 字节账与复用顺序

```text
Less_UB_total = B_less*tile*max(sizeof(P),1)
Cast_dynamic_ub = N/A (AICPU)
Cast GM temporary = N*sizeof(D)
workspace_total = executor聚合
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| BOOL GM | Less | all CPU shards | Cast完成后 | executor task ordering | Cast启动前Less完成 |
| D GM shards | CastTask | ViewCopy | 回写完成后 | ParallelFor完成 + executor | shard不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 跨执行族 | executor stream/task ordering | Less AIV | Cast AICPU | BOOL GM | 必须Device可见后CPU读 |
| CPU cast | ParallelFor | BOOL GM | D GM | shards | static_cast把false/true变0/1 |
| 回写 | executor ordering | CPU Cast | ViewCopy | D GM | N个元素全覆盖 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/cast/op_api/cast.cpp` | `Cast` | AIV pair失败后CPU fallback |
| Kernel entry/router | `math/cast/op_kernel_aicpu/cast_aicpu.cpp` | `CastCpuKernel::Compute` | 校验、map、并行 |
| Device body/helper | 同上 | `CastTask<bool,D>` | 逐元素static_cast |

### Route R-004：AICPU Less + AIV 结果 Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Less selector | `less.cpp` | RegBase set | P=INT16不命中 | LessAiCpu |
| CPU body | `less_aicpu.cpp` | INT16 switch | `LessCompute<int16_t>` | CPU比较 |
| Cast selector | `cast.cpp/cast_struct.h` | BOOL→INT16 | AIV pair + TwoCast policy | CastAiCore |

#### Device 数据流

##### 函数级调用链

```text
LessCpuKernel::Compute -> LessCompute<int16_t> -> NoBcastCompute -> SpecialCompute
  -> BOOL GM -> cast<0> -> CastTwo<int8_t,int32_t,int16_t>::Process -> ViewCopy
```

##### 源码等价伪码

```text
# 伪码；scalar走Y_ONE_ELEMENT
CPU shard: b[i] = x[i] < scalar
AIV Cast tile: int8 bool -> int32 -> int16 -> GM
ViewCopy to selfRef
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| BOOL GM | i | N | BOOL | CPU Less | AIV Cast | executor | task ordering |
| Cast queues/mid | core/loop local | `2*ubFormer + ubFormer*4 + 2*ubFormer*2` bytes | int8/int32/int16 | CopyIn/Cast1 | Cast2/CopyOut | Cast node | Queue+TBuf复用 |
| INT16 GM | i*2 | 2N | INT16 | Cast | ViewCopy | executor | tail exact |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQ bool · depth=2 · 2*ubFormer"]:1
  mid["mid int32 · 4*ubFormer"]:1
  outQ["outQ int16 · depth=2 · 4*ubFormer"]:2
```

##### 字节账与复用顺序

```text
Less UB = N/A (AICPU)
Cast UB = 2*ubFormer + 4*ubFormer + 2*ubFormer*2 bytes
workspace_total = executor聚合
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| CPU BOOL GM | Less shards | Cast | Cast后 | ParallelFor完成 | shards全部完成 |
| Cast mid | first Cast | second Cast | 下一tile | Compute顺序 | 不跨tile存活 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CPU Less | ParallelFor/串行阈值 | x/scalar | BOOL GM | shard | C++ `<` |
| Cast | TQue+TwoCast | BOOL | INT16 | queues/mid | 0/1精确 |
| tail | CPU end + Cast POD | Host/runtime | stores | GM | 不越N |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/less/op_api/less.cpp` | `Less` | INT16转AICPU |
| Kernel entry/router | `less_aicpu.cpp` | `Compute` INT16 case | CPU模板实例 |
| Device body/helper | `cast_impl.h` | `CastTwo<int8,int32,int16>` | BOOL到INT16 AIV管线 |

### Route R-005：AICPU Less + AICPU 结果 Cast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Less selector | `less.cpp` | RegBase set | P=UINT16/UINT32 | LessAiCpu |
| Less CPU | `less_aicpu.cpp` | dtype switch | UINT16/UINT32实例 | CPU比较 |
| Cast selector | `cast.cpp` | BOOL→UINT16/UINT32 | AIV pair不命中，CPU支持 | CastAiCpu |

#### Device 数据流

##### 函数级调用链

```text
LessCpuKernel::Compute -> SpecialCompute<T> -> BOOL GM
  -> CastCpuKernel::Compute -> CastTask<bool,T> -> T GM
  -> ViewCopy
```

##### 源码等价伪码

```text
# 伪码
parallel shard1: b[i] = x[i] < scalar
barrier/task completion
parallel shard2: out[i] = static_cast<T>(b[i])
ViewCopy(out,selfRef)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/scalar GM | i/0 | N/1 | T | API temporaries | CPU Less | executor | read-only |
| BOOL GM | i | N | BOOL | CPU Less | CPU Cast | executor | task edge |
| T GM | i*sizeof(T) | N*sizeof(T) | T | CPU Cast | ViewCopy | executor | disjoint shards |

##### 分层内存布局

N/A：两个核心节点均为 AICPU；使用 CPU registers 和 GM 临时边，不使用 AIV UB。

##### 字节账与复用顺序

```text
UB_total = N/A
intermediate_GM = N bytes BOOL + N*sizeof(T) result
workspace_total = executor聚合；AICPU task内部额外内存固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| BOOL GM | Less CPU | Cast CPU | Cast完成 | executor task ordering | 不得并行重叠两任务 |
| result GM | Cast CPU | ViewCopy | 回写完成 | task completion | shards不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Less shards | ParallelFor | input | task完成 | BOOL GM | 每元素C++比较 |
| Cast shards | ParallelFor | BOOL | task完成 | T GM | false/true→0/1 |
| ViewCopy | executor ordering | T GM | selfRef | 原地目标 | 完整N元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `less.cpp` / `cast.cpp` | 两个L0 selector | 两次CPU fallback |
| Kernel entry/router | `less_aicpu.cpp` / `cast_aicpu.cpp` | `Compute` | dtype switch/map |
| Device body/helper | 同上 | `SpecialCompute<T>` / `CastTask<bool,T>` | 两段逐元素CPU实现 |

### Route R-006：含输入 promote Cast 的组合路线

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lt_scalar.cpp` | `GetScalarDefaultDtype/CombineCategoriesWithComplex` | `self.dtype != P` | 创建输入Cast |
| Cast selector | `cast.cpp` | pair表 | pair命中AIV否则CPU | promoted GM |
| Less/结果 selector | `less.cpp/cast.cpp` | 上述集合 | 按P与self分别选择 | R-002～R-005后半段 |

#### Device 数据流

##### 函数级调用链

```text
Cast(selfRef.dtype -> P) [cast<0> or CastCpuKernel]
  -> Less<P> [less<202> or LessCpuKernel]
  -> Cast(BOOL -> selfRef.dtype) [cast<0>, CastCpuKernel, or no-op]
  -> ViewCopy
```

##### 源码等价伪码

```text
# 伪码；每个节点使用自己的分核和tail
xp[i] = cast<P>(selfRef_before[i])
b[i]  = xp[i] < cast<P>(other)
r[i]  = selfRef.dtype==BOOL ? b[i] : cast<selfRef.dtype>(b[i])
ViewCopy(r,selfRef)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input Cast local | 各Cast POD | policy UB或CPU registers | S/P | Cast | promoted GM | node-local | 见R-002/003 |
| promoted GM | i*sizeof(P) | N*sizeof(P) | P ND | input Cast | Less | executor | Less后释放 |
| BOOL GM | i | N | BOOL | Less | result Cast/ViewCopy | executor | 不能与promoted GM别名 |
| result GM | i*sizeof(S) | N*sizeof(S) | S | result Cast | ViewCopy | executor | BOOL self时就是BOOL GM |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  castIn["input Cast UB or CPU shard"]:1
  promoted["GM promoted · N*sizeof(P)"]:1
  less["Less UB or CPU shard"]:1
  result["BOOL/result GM + optional Cast"]:1
```

##### 字节账与复用顺序

```text
workspace_total = executor聚合各AIV节点16MiB声明与框架需求
GM_temporaries = N*sizeof(P) + N bytes BOOL + optional N*sizeof(self.dtype)
UB_total按节点独立计算，不能跨Kernel相加为单核UB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| promoted GM | input Cast | Less | Less完成 | executor ordering | Cast完成后Less读取 |
| BOOL GM | Less | result Cast/ViewCopy | consumer完成 | executor ordering | 不提前覆盖原输入 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| promote | Cast policy/static_cast | selfRef | Less | promoted GM | 950类型推导决定P；浮点scalar默认FLOAT |
| compare | AIV Compare或C++ `<` | promoted values | BOOL GM | comparison | 比较发生在P精度 |
| result conversion | Cast/static_cast | BOOL | ViewCopy | result GM | 0/1到原dtype精确 |
| tail | 各POD/shard end | producer | store | 每节点GM | 每节点独立保护N |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `aclnn_lt_scalar.cpp` | V35 body | promote与节点顺序 |
| Kernel entry/router | `cast.cpp/less.cpp` | 两个L0 selector | AIV/AICPU逐节点选择 |
| Device body/helper | Cast/Less AIV与AICPU源码 | pair对应模板 | 完整多节点实现 |

## 算子特定优化

### Trick T-001：scalar只初始化ping/pong两相

#### 触发与源码依据

- 适用 Route：所有AIV Less路线
- 触发条件：schMode 202、x2 scalarFlag置位、当前核超过两个tile
- Golden 来源：[`pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h:75-92`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h#L75-L92)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
      isFirstScalarTime = true;
      for (; i < blockLoopNum - 1; i++) {
          if (isFirstScalarTime && i > 1) {
              isFirstScalarTime = false;
          }
          Run<0>(offset, tilingData->tileNum, i & 1);
```

#### 机制与设计原因

x2只在i=0、1分别从GM读标量并Duplicate到ping/pong区域，i≥2直接复用同parity缓存；x1与输出仍逐tile搬运。它减少后续tile的标量读取与Duplicate，同时保持双缓冲两相都已初始化。

#### 迁移边界

- 适用：全域scalar、两相buffer-id调度的elementwise DAG。
- 前提：scalar在Kernel期间不变，两个parity均先初始化。
- 失效：非scalar输入或不同流水深度必须逐tile搬入/初始化。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只初始化ping会令奇数tile读取未定义local数据；固定源码通过前两次循环分别初始化两相。

### Trick T-002：比较mask后显式Select成字节BOOL

#### 触发与源码依据

- 适用 Route：所有AIV Less路线
- 触发条件：进入 `LessCompute<T>`
- Golden 来源：[`math/less/op_kernel/arch35/less_dag.h:36-44`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_kernel/arch35/less_dag.h#L36-L44)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    using ResCmp = Bind<Vec::Compare<uint8_t, T, LESS_CMP_MODE>, Input0, Input1>;
    using ConstOne = MAKE_CONST(uint8_t, 1);
    using ConstZero = MAKE_CONST(uint8_t, 0);
    using DupOne = Bind<Vec::Duplicate<uint8_t>, ConstOne>;
    using DupZero = Bind<Vec::Duplicate<uint8_t>, ConstZero>;
    using ResSelect = Bind<Vec::Select<uint8_t, uint8_t, LESS_SELECT_MODE>, ResCmp, DupOne, DupZero>;
```

#### 机制与设计原因

Compare的选择mask不直接当公开BOOL存储，而是以mode2 Select在local中物化为uint8 1/0，再CopyOut。这样后继结果Cast和AICPU都消费规范的字节BOOL，避免依赖mask压缩布局。

#### 迁移边界

- 适用：公开结果要求字节BOOL，而底层Compare产生选择mask的路线。
- 前提：Select mode2的mask/两路uint8输入约定保持一致。
- 失效：若输出ABI改成bit-packed BOOL，不能复用此CopyOut布局。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

直接把Compare mask写GM会改变BOOL布局；固定源码明确先Select 1/0。

### Trick T-003：结果Cast延迟原地覆盖

#### 触发与源码依据

- 适用 Route：所有非BOOL `selfRef`
- 触发条件：Less输出BOOL而原地目标保持原dtype
- Golden 来源：[`math/less/op_api/aclnn_lt_scalar.cpp:350-360`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/less/op_api/aclnn_lt_scalar.cpp#L350-L360)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    auto ltOpOut = l0op::Less(selfCasted, otherCasted, uniqueExecutor.get());
    CHECK_RET(ltOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castOut = l0op::Cast(ltOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto view_copy_result = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
```

#### 机制与设计原因

Less先完整写BOOL临时量，Cast再生成原dtype的0/1临时量，最后ViewCopy提交。该顺序既保持公开原地dtype，又避免核心读取旧selfRef时被提前覆盖；代价是非BOOL路线存在额外临时量/节点。

#### 迁移边界

- 适用：原地比较API要求对象dtype不变。
- 前提：executor严格保持Less→Cast→ViewCopy顺序和临时量生命周期。
- 失效：独立BOOL输出可直接消费Less结果；直接原地写会破坏尚未读取的数据。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把Less BOOL输出直接别名到非BOOL selfRef会同时违反dtype和存储宽度约束。

## 模板与实现对比

| 对比维度 | R-001/R-002 AIV主路 | R-003/R-004混合路 | R-005 CPU路 | R-006 promote组合 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | pair全部命中RegBase表 | 一个核心pair未命中 | Less与结果Cast均未命中 | self dtype与P不同 | L0 selector逐节点判断 |
| 模板与 dtype | Less DAG + Cast policy | AIV模板与CPU模板串联 | CPU C++模板 | 可再增加输入Cast | P与self共同决定 |
| 循环与分核 | AIV core/tile | AIV tile + CPU shard | 两次CPU shard | 每节点独立 | 不共享POD |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 静态buffer-id；Cast Queue/TBuf | 一侧UB一侧CPU registers | CPU registers | 取决于各节点 | 无dynamic UB/通信 |
| 内存总占用与复用 | 每AIV节点独立UB | GM边连接两执行族 | GM临时 | 额外P临时 | executor owner |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Compare/Select/Cast | Vector + C++ loop | C++ loop | 组合 | 无SIMT/通信；scalar一维不走NDDMA |
| 同步、精度与 tail | buffer id/TQue + POD tail | task ordering + shard end | task ordering | 多边串联 | 比较在P精度，结果0/1 |
| fallback/失败边界 | Tiling/launch失败终止 | CPU错误终止 | CPU错误终止 | 任一节点失败阻断 | 无静默跨Route重试 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但固定ops-math Commit缺少A2 Less/Cast Host/Tiling/Kernel构建闭包，本卡不覆盖 | 公开文档写支持，但固定Commit缺少A3核心闭包，本卡不覆盖 | 文档、OpDef、CMake一致支持 |
| Host/Launcher | 非RegBase body存在，但所达旧实现不闭合 | 同左 | RegBase V35 promote；逐节点AIV/AICPU选择 |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | arch35 AIV Less/Cast + AICPU fallback |

## 限制与失败边界

- `selfRef/other` 为空返回参数错误；dtype、promote/Cast合法性或rank>8失败返回参数错误；executor创建/节点返回空指针属于内部错误。
- 空Tensor成功且workspace为0，不启动任何核心节点。
- 950公开dtype集合包含部分必须走AICPU的pair；不得只描述AIV主路，也不得把AICPU误标为Cube/AIC。
- AIV Less `tileNum:int32_t`、`blockNum:int16_t` 无显式窄化上界检查；Cast POD为int64但GM copy长度会窄化到API字段，极端shape行为不得推测。
- `Contiguous`、`ConvertToTensor`、`ViewCopy`和executor内部切片/event是允许的框架边界；其具体Kernel、额外workspace与event细节固定仓不可见。
- A2/A3只有公开支持声明，没有本固定源码集合中的source closure，因此未生成无后缀卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/less/docs/aclnnLtScalar&aclnnInplaceLtScalar.md` | `aclnnInplaceLtScalar*` | 950支持、原地语义、dtype/rank |
| 入口声明 | `math/less/op_api/aclnn_lt_scalar.h` | 同上 | 两段式原型 |
| 入口实现 | `math/less/op_api/aclnn_lt_scalar.cpp` | V35 body/inplace wrapper | promote、空域、节点顺序、回写 |
| Less分发 | `math/less/op_api/less.cpp` | `IsAiCoreSupport/Less` | AIV/AICPU选择 |
| Less Host/Tiling | `math/less/op_host/arch35/less_tiling_arch35.cpp` | `LessTiling` | dtype DAG与key |
| Less OpDef | `math/less/op_host/less_def.cpp` | `ops::Less` | 950注册 |
| Less AIV entry | `math/less/op_kernel/less_apt.cpp` | `less<schMode>` | dtype router与scheduler实例 |
| Less AIV DAG | `math/less/op_kernel/arch35/less_dag.h` | `LessCompute<T>` | Compare/Select body |
| Less AICPU | `math/less/op_kernel_aicpu/less_aicpu.cpp` | `LessCpuKernel` | CPU dtype、broadcast与shard |
| Less构建 | `math/less/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950/arch35 |
| Cast分发 | `math/cast/op_api/cast.cpp` | `Cast/IsAiCoreSupport` | 同dtype消除、pair selector |
| Cast Host/Tiling | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling` | policy、分核与UB公式 |
| Cast POD | `math/cast/op_host/arch35/cast_tiling.h` | `CastTilingData` | 12字段POD |
| Cast AIV entry | `math/cast/op_kernel/cast_apt.cpp` | `cast<0>` | policy router |
| Cast AIV body | `math/cast/op_kernel/arch35/cast_impl.h` | Cast body classes | Queue、Cast与tail |
| Cast policy | `math/cast/op_kernel/arch35/cast_struct.h` | `GetCastPolicy` | pair到模板/round映射 |
| Cast AICPU | `math/cast/op_kernel_aicpu/cast_aicpu.cpp` | `CastCpuKernel/CastTask` | CPU pair map与ParallelFor |
| Cast构建 | `math/cast/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950/arch35 |
| 公共POD | `pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | `BroadcastOneDimTilingDataAdvance` | one-dim POD与key集 |
| 公共Tiling | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling::DoTiling` | scalarFlag、tile/core/workspace |
| 公共router | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch<202>` | AIV-only与body选择 |
| 公共body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch` | 循环、搬运、同步与tail |
| Vector primitive | `pkg_inc/op_common/atvoss/util/vec.h` | Compare/Select/Copy | Vector API参数 |
| 内存策略 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | buffer-id内存策略 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | one-dim 16-byte POD与Cast 12字段POD | 字段顺序、位宽、元素单位、scalar bit不可改 | 两个TilingData结构 |
| 调用顺序 | Contiguous→可选promote Cast→Less→可选结果Cast→ViewCopy | 原地目标不能在旧值消费前覆盖 | V35 body |
| workspace/内存 | 每个AIV节点独立UB与16MiB声明；GM边由executor持有 | 不把多节点UB相加成单核UB，不提前别名 | ATVOSS/Cast Tiling |
| 同步 | buffer-id或TQue节点内同步；executor节点间ordering | scalar两相先初始化；CPU/AIV边必须任务完成 | schedulers/ParallelFor |
| 精度与尾块 | 比较在P dtype；结果严格0/1；各节点独立tail | 不在self原dtype直接比较当P不同；不省略结果Cast | promote/DAG/Cast policy |
| 失败边界 | 空域短路；pair selector与Tiling/CPU错误终止 | 不补写A2/A3，不把CPU route隐去 | API/L0/CMake |
