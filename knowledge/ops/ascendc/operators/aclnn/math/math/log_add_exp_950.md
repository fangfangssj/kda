---
type: Operator
title: "LogAddExp 950 ACLNN 算子知识"
description: "说明 aclnnLogAddExp 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/docs/aclnnLogAddExp.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_api/aclnn_logaddexp.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_api/aclnn_logaddexp.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_api/logaddexp.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_host/log_add_exp_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_host/arch35/log_add_exp_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_host/config/ascend950/log_add_exp_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_kernel/arch35/log_add_exp_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_kernel/arch35/log_add_exp_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_kernel/log_add_exp_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/CMakeLists.txt
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
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/CMakeLists.txt
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/broadcast_utils.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
created_at: "2026-08-25T11:32:27Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# LogAddExp 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnLogAddExpGetWorkspaceSize` + `aclnnLogAddExp`，公开 ACLNN 两段式接口，无版本后缀 |
| 计算语义 | broadcast 后逐元素计算 `m + ln(1 + exp(-abs(x-y)))`，其中 `m=max(x,y)`；与 `LogAddExp2` 的以 2 为底语义严格分离 |
| 输入输出 | `self/other` 为 ND、0～8 维、可非连续、支持 13 类数值/BOOL dtype；两者可 broadcast；`out` 为 FLOAT16/FLOAT/BFLOAT16 且 shape 等于 broadcast shape |
| 执行类型 | 可选前置 Cast 为 AIV_VECTOR 或 AiCPU；LogAddExp 核心为 AIV_VECTOR，部分搬入使用 NDDMA；`BroadcastSch` 明确设置 `KERNEL_TYPE_AIV_ONLY`，无 Cube/AIC/MIX_CV |
| Kernel 路线 | 核心 R-001 mode202；R-002 mode1；R-003 mode2；R-004 mode101–104/109。每条路线以前缀 `P(a,b)` 表示两输入各自 no-op/AIV Cast/AiCPU Cast 的独立拓扑变体 |
| 核心资源 | 输入/临时/输出 GM、AIV 静态 buffer-id pool 或 `TBuf<VECCALC>`、Vector/MicroAPI 寄存器、NDDMA 参数、每个 AIV 节点声明的 16 MiB workspace；AiCPU Cast 直接遍历 GM |
| 来源闭包 | source_closed：公开 ACLNN→连续化/类型转换→950 LogAddExp launcher→simplified Tiling/POD→全部连续 Broadcast scheduler→DAG/Vector body→构建接线均闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `S=broadcast(self.shape, other.shape)`，先把两输入分别转换为 `out.dtype`。对输出坐标 `i∈S`，令 `x`、`y` 为按 broadcast stride 取得的输入值：

```text
m = max(x, y)
d = guarded_sub(x, y)      # x,y 同为 +Inf 或同为 -Inf 时强制 d=0
u = exp(-abs(d))
l = stable_log1p(u)
out[i] = m + l
```

公开入口固定向 L0 传 `base=-1.0f, scale=1.0f, shift=0.0f`，所以 `useFullFormula_` 恒为 false，只能进入 `formulaType=0` 的 simplified DAG。`formulaType=1` 是同一内部 Op 的属性化 full 实现，但从 `aclnnLogAddExp` 不可达，不进入本文 Route。`LogAddExp2` 具有独立公开 API 和底数 2 语义，也不合并。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；0～8 维；允许非连续 | FLOAT16、FLOAT、DOUBLE、UINT8、INT8、UINT16、INT16、UINT32、INT32、UINT64、INT64、BOOL、BFLOAT16 | 必填；可转换为 `out.dtype`；与 `other` 可 broadcast |
| `other` | 同 `self` | 同 `self` | 必填；可转换为 `out.dtype`；与 `self` 可 broadcast |
| `out` | ND；0～8 维；允许非连续 | FLOAT16、FLOAT、BFLOAT16 | shape 必须严格等于两输入 broadcast 结果；950 文档支持 BF16 |
| `base/scale/shift` | 内部标量属性 | float | 公开入口固定为 `-1/1/0`，用户不可配置；决定 simplified route |
| `workspaceSize` | Host 标量输出 | `uint64_t*` | 第一段返回完整 Executor 的 Device workspace 字节数 |
| `executor` | Host 对象 | `aclOpExecutor**` | 保存连续化、Cast、核心与 ViewCopy 的执行图 |
| `workspace/workspaceSize/stream` | Device workspace/ACL stream | 指针、`uint64_t`、`aclrtStream` | 第二段交由 `CommonOpExecutorRun` |

### 入口与版本

固定提交公开的是无版本后缀 `aclnnLogAddExpGetWorkspaceSize/aclnnLogAddExp`，未发现更高版本入口。公开产品表明确支持 Ascend 950PR/950DT；`LogAddExp::AICore().AddConfig("ascend950")`、ascend950 binary config 以及 `SUPPORT_COMPUTE_UNIT "ascend950"/SUPPORT_TILING_DIR "arch35"` 共同证明 950 活动实现。相同目录中的 `aclnnLogAddExp2` 是另一逻辑算子。

## Host/Launcher 执行流程

### 参数校验与前处理

真实顺序如下：

1. 校验 `workspaceSize/executor`，创建唯一 Executor；任一失败立即返回。
2. `CheckParams` 依次检查三 Tensor 非空、950 输入/输出 dtype 表与输入到输出的可转换关系、两输入 rank≤8、broadcast 可行且 `out.shape` 正确；950 还要求三者 storage format 相同且非私有。公开契约只承诺 ND。
3. 任一输入为空时写 `workspaceSize=0`、移交空 Executor 并成功返回，不建立 Cast 或 LogAddExp Device 节点。
4. `Contiguous(self)` 后 `Cast(...,out.dtype)`；`other` 按相同顺序处理。相同 dtype 的 Cast 直接返回输入，不启动 Kernel。
5. `l0op::LogAddExp(selfCasted,otherCasted,-1,1,0)` 计算 broadcast shape、分配 `out.dtype` 临时 Tensor；核心 dtype 必为 FLOAT16/FLOAT/BFLOAT16，进入 AICore launcher。
6. `Cast(finalOut,out.dtype)` 因 dtype 已相同而 no-op；`ViewCopy(castOut,out)` 写入可能非连续的公开输出。
7. Executor 聚合 workspace 并转交调用者；第二段调用 `CommonOpExecutorRun` 在给定 stream 上按图执行。

前置 Cast 的精确选择记为：`N`=同 dtype no-op，`V`=AIV Cast，`C`=AiCPU Cast。对每个输入独立应用：

| 输入 dtype `s` → `out.dtype=d` | 前缀节点 |
|---|---|
| `s==d` | `N`，无 Device 节点 |
| `s==UINT64`，`d∈{FLOAT16,FLOAT,BFLOAT16}` | `C` |
| `s==DOUBLE && d==FLOAT16` | `C` |
| 其余公开且非同 dtype pair | `V`，均列在 950 `REGBASE_AICORE_DTYPE_SUPPORT_LIST` |

### 参数、Workspace 与生命周期

`uniqueExecutor` 持有连续化临时 Tensor、每个 Cast 输出、LogAddExp 临时输出和 launcher 列表；`ReleaseTo(executor)` 后生命周期转交公开 Executor，直至第二段执行结束。前置节点与核心通过独立 GM Tensor 交接；同一 Executor/stream 的提交顺序提供 producer→consumer 可见性。固定算子源码未公开 Executor 对各节点 workspace 的具体 offset、复用和 event 编排，调用者必须使用第一段返回的聚合值。

LogAddExp 的 `BroadcastBaseTiling::DoTiling` 固定令 `workspaces[0]=16*1024*1024`；核心 entry 接收 `workspace` 但不读取。AIV Cast 的 `CastTiling::GetWorkspaceSize` 也为 16 MiB；AiCPU Cast 由 `AicpuTaskSpace("Cast")` 管理。多节点 Executor 的总 workspace 不能从这些单节点声明简单相加或自行缩小。

### 启动图与失败传播

```text
aclnnLogAddExpGetWorkspaceSize
  -> Contiguous(self)  -> Cast(self,out.dtype)?  --GM edge--+
  -> Contiguous(other) -> Cast(other,out.dtype)? --GM edge--+-> LogAddExp AIV [R-001..R-004]
  -> Cast(coreOut,out.dtype)  # 恒为 no-op
  -> ViewCopy(coreOut,out)

aclnnLogAddExp
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

| 节点/边 | entry、POD 与执行族 | offset/extent/dtype/layout | ordering 与 owner/lifetime | 失败传播 |
|---|---|---|---|---|
| `Contiguous→Cast/core` | 框架节点；固定仓允许边界 | 各输入逻辑元素数；原 dtype 或 `out.dtype`；连续 ND | Executor 持有；同 stream 先于 consumer | 返回空指针即第一段失败 |
| `P=V` | `cast<0>(x,y,workspace,CastTilingData)`；AIV_VECTOR | 全输入元素；目标为 `out.dtype` | Cast 输出活到核心读完 | launcher/tiling 失败阻断核心 |
| `P=C` | `CastCpuKernel::CastTask<S,D>`；AiCPU | `[start,end)` GM 分片 | AiCPU task 完成后核心可见；Executor 持有输出 | 不支持 pair 或 task 失败阻断核心 |
| 核心 | `log_add_exp<schMode,0>(x1,x2,y,workspace,tiling)`；AIV_VECTOR，部分 route 含 NDDMA | broadcast 输出 `N` 元素；dtype=`out.dtype` | 单核心节点；输出活到 ViewCopy 完成 | Tiling/launcher 失败阻断写出 |
| `core→ViewCopy` | 同 dtype Cast no-op + 框架 ViewCopy | `N*sizeof(out.dtype)`；目标可非连续 | 同 Executor/stream；完成后临时可回收 | ViewCopy 失败返回内部错误 |

## Route 与参数设计

### 分核与资源计算

两路核心输入经过 `Contiguous/Cast` 后均连续。公共 Tiling 右对齐 shape，将 broadcast 轴的输入 stride 置 0，再合并广播模式相同的相邻轴。记合轴输出维数 `L`，元素总数 `N`，AIV 核数 `C`，UB 字节 `U`，DAG 最大/最小 dtype 字节 `smax/smin`；FLOAT16 simplified 为 `2/2`，FLOAT32 为 `4/4`，BF16 因升 FP32 为 `4/2`。令 `B_N=OpDag::GetBufferNum<true,true>()`、`B_U=OpDag::GetBufferNum<false,true>()`。

```text
# L==1，mode202
tile0_bytes = floor((U / B_N) / 512) * 512
tile0 = tile0_bytes / smax
if C*(512/smin) > N: tile = 512/smin
else if C*tile0 > N: tile = floor((N/C)/(512/smax))*(512/smax)
else: tile = tile0
ubOuter = ceil(N/tile)
blockFormer = ceil(ubOuter/C); if blockFormer==1: blockFormer=2
blockNum = ceil(ubOuter/blockFormer)

# L>1，mode1/2/101..109
maxElem0 = floor(((U*8)/(B*smax*8))/(256*8/(smin*8))) * (256*8/(smin*8))
从末轴向前累乘，选第一个超过 maxElem0 的 ubSplitAxis
ubFormer = maxElem0 / product(inner output dims)
ubOuter = ceil(outputDim[ubSplitAxis]/ubFormer)
ubTail = outputDim[ubSplitAxis]-(ubOuter-1)*ubFormer
fused = product(outputDims before ubSplitAxis)*ubOuter
blockFormer = ceil(fused/C); blockNum=ceil(fused/blockFormer)
blockTail = fused-(blockNum-1)*blockFormer
```

多轴路径 `elemNum=maxElem0`；`preferMultiCore=false`，不会进入额外缩 tile 的多核偏好循环。Host 将 `blockNum` 设为 blockDim。mode202 在 Device 重新计算 `ubTail` 与最后一核 `blockTail`；多轴 POD 直接携带这些字段。

### Route 选择

每个核心 Route 都带前缀拓扑参数 `P(a,b)`，`a/b∈{N,V,C}` 分别对应 `self/other` 的 Cast 选择；它们的谓词见 Host 前处理表，顺序固定为 self 前缀、other 前缀、核心、ViewCopy。

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001/P 单轴 Advance | `GET_TPL_TILING_KEY(202,0)` | 950；非空；两核心输入连续；合轴后 `L==1`；`InputSize+OutputSize=3≤4 && VarSize==0` | `L>1` 进入 R-004/R-002/R-003 |
| 2 | R-004/P UB Broadcast | `GET_TPL_TILING_KEY(101..104/109,0)` | 连续、多轴；`IsBigNLastBrc()`，或首个实际广播输入 dtype 为 F16/BF16 且最后输出轴按 32B 对齐 | false 进入 R-002/R-003 |
| 3 | R-002/P NDDMA≤5维 | `GET_TPL_TILING_KEY(1,0)` | 连续、多轴；UB selector false；`runningRank=L-ubSplitAxis≤5` | rank>5 进入 R-003 |
| 4 | R-003/P NDDMA 外循环 | `GET_TPL_TILING_KEY(2,0)` | 连续、多轴；UB selector false；`runningRank>5` | 无 fallback，Tiling 失败 |

```text
if self.empty || other.empty: success(workspace=0,no kernel)
a = CastSelector(self.dtype,out.dtype); b = CastSelector(other.dtype,out.dtype)
P(a,b) in program order
if collapsedRank == 1: R-001/P
else if IsUbBroadcast(): R-004/P  # rank 1..4 -> 101..104; >4 -> 109
else if runningRank <= 5: R-002/P
else: R-003/P
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001/P | `(202,0)` + `P(a,b)` | 950；非空；F16/F32/BF16 核心；连续；合轴 rank1 | 0～2 个 Cast→单 AIV 核心 | `BroadcastOneDimAdvanceSch<LogAddExpSimplified[WithCast]Compute<T>::OpDag>` | 静态 buffer-id pool；ping/pong；scalar 输入可缓存 | 最后一核/最后 tile 传真实 count；零域已 Host 短路 |
| R-002/P | `(1,0)` + `P(a,b)` | 连续多轴；UB selector false；runningRank≤5 | 0～2 Cast→单 AIV+NDDMA 核心 | `BroadcastNddmaSch<OpDag,false>` | `TBuf<VECCALC>`；`MultiCopyParams<T,5>`；position token | `ubTail`；无其它 fallback |
| R-003/P | `(2,0)` + `P(a,b)` | 同 R-002，但 runningRank>5 | 0～2 Cast→单 AIV+NDDMA 核心 | `BroadcastNddmaSch<OpDag,true>` | 软件外轴循环+内层 5D NDDMA；同一 pool 复用 | 外循环完整覆盖高维；末 tile 用 `ubTail` |
| R-004/P | `(101..104/109,0)` + `P(a,b)` | 连续多轴；UB selector true | 0～2 Cast→单 AIV 核心 | `BroadcastUbSch<OpDag,rank>`；109 的 rank=-1 | compact CopyIn 与 UB Broadcast 展开区分别占 buffer-id；position token | rank1..4 静态，109 动态；只写有效输出 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 公开 simplified | `base==-1 && scale==1 && shift==0` | 避免四个 scalar 与 full DAG；key 的 formulaType 固定 0 | 全部 | 内部 full route，但公开 ACLNN 不可达 | `aclnn_logaddexp.cpp::LOG_*`; `LogAddExpTiling::GetShapeAttrsInfo` |
| 空域 | `self->IsEmpty() \|\| other->IsEmpty()` | 避免 `dimLength==0` 与无意义节点 | 全部之前 | 成功、workspace=0、无 Device 节点 | `aclnnLogAddExpGetWorkspaceSize` |
| dtype 同构 | 核心 `x1.dtype==x2.dtype==y.dtype` | 单模板 `DTYPE_X1`；BF16 专门升 FP32 | 全部核心 | `CheckDtype` 失败；正常 ACLNN 由前置 Cast 保证 | `LogAddExpTiling::CheckDtype` |
| AIV-only | `BroadcastSch::Process` 固定设置 | DAG 只含 Vector/MicroAPI 与搬运；不启动 Cube/AIC | R-001～R-004 | N/A | `broadcast_sch.h::BroadcastSch::Process` |
| UB 容量 | `B*elemNum*smax<=U` 且 256B repeat 对齐 | pool 必须容纳全部存活节点 | 全部核心 | Tiling `GRAPH_FAILED` | `BroadcastGetMaxElemNum`; `DoBrodcastTiling` |
| one-dim 512B | `floor((U/B_N)/512)>0` | 静态 pool 的每 buffer tile 按 512B 取整 | R-001 | 失败，不退到多轴 route | `GetOneDimUBFormerAdvance` |
| 小数据缩核 | `C*(512/smin)>N` 或 `C*tile0>N` | 降 tile/核启动数；`blockFormer==1` 时合并为每核2 tile | R-001 | 保持 full tile | `DoOneDimFormerTilingAdvance` |
| UB Broadcast | 大 NLast，或 F16/BF16 且 `lastDim%(32/sizeof(T))==0` | 减少大非尾广播的 NDDMA dcache 压力，或利用 16-bit 对齐广播 | R-004 | R-002/R-003 | `IsBigNLastBrc`; `IsUbBroadcast` |
| NDDMA 维数 | `runningRank≤5` | `MultiCopyParams<T,5>` 的硬上限 | R-002/R-003 | `>5` 软件外循环 | `SetSchMode`; `BroadcastSetNddmaConfig*` |
| 前置 Cast selector | `(s,d)` 在 REGBASE pair 表 | 只有已实例化 policy 可进入 AIV | `P=V` | 支持的其余 pair 进入 AiCPU | `l0op::IsAiCoreSupport`; `Cast` |
| AiCPU Cast | `UINT64→F16/F32/BF16` 或 `DOUBLE→F16` | 950 AIV policy 表无该 pair，AiCPU map 有对应模板 | `P=C` | map 缺失则第一段失败 | `CastAiCpu`; `CastCpuKernel::SetMap` |

### POD/ABI

`BroadcastBaseTilingData<OpDag>` 的实例含 `InputSize=2`、`CopyBrcSize=2`、`VecBrcSize=0`；零长模板数组的最终 `sizeof/offsetof` 未在固定仓物化，故只对普通结构给出确定 offset，对模板结构保留源码字段顺序。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen` | int64/64 | sizeof=16, alignof=8, offset=0 | 元素 | `DoOneDimFormerTilingAdvance: dims.back()[0]` | `Init` 重算 ub/core tail | `>0` |
| 同上 | `tileNum` | int32/32 | offset=8 | 元素/tile | UB/`B_N`/512B 公式后按核数缩减 | `blockLen`、循环步长 | `>0`；无显式 int32 上界检查 |
| 同上 | `blockNum` | int16/16 | offset=12 | AIV core | `ceil(ubOuter/blockFormer)` 并 `SetBlockDim` | 尾核判断 | 无显式 int16 上界检查 |
| 同上 | `scalarFlag` | int16/16 | offset=14 | 输入位图 | 输入 i 合轴 dim=1 时置 `1<<i` | `CopyIn` 选择 `Duplicate` 或线性搬入 | bit0=self，bit1=other |
| `BroadcastBaseTilingData<OpDag>` | `scheMode` | int32/32 | 模板实例最终 offset 固定仓未物化；字段首位 | key/mode | `SetSchMode` | router 已按 key 实例化，body保留诊断 | 1/2/101..104/109 |
| 同上 | `shapeLen` | int32/32 | 同上，源码顺序第2 | 轴 | `DoDimensionCollapse` | scheduler 轴循环 | `1..8` |
| 同上 | `ubSplitAxis` | int32/32 | 同上 | 轴索引 | `GetBlockSplitFactor` | offset/NDDMA/UB BRC | `<shapeLen` |
| 同上 | `ubFormer` | int32/32 | 同上 | 元素/切分轴 | `maxElem/product(innerDims)` | 整 tile 范围 | `>0` |
| 同上 | `ubTail` | int32/32 | 同上 | 元素/切分轴 | 尾块差值 | 最后 tile | `1..ubFormer` |
| 同上 | `ubOuter` | int64/64 | 同上 | tile | `ceil(dim/ubFormer)` | core/tile 循环 | `>0` |
| 同上 | `blockFormer` | int64/64 | 同上 | tile/core | `ceil(fused/C)` | 非尾核 loop | `>0` |
| 同上 | `blockTail` | int64/64 | 同上 | tile | `fused-(blockNum-1)*blockFormer` | 尾核 loop | `1..blockFormer` |
| 同上 | `dimProductBeforeUbInner` | int64/64 | 同上 | fused tile | Host=`fused` | `BroadcastGetAxesIndices` | 正数 |
| 同上 | `elemNum` | int64/64 | 同上 | 元素/buffer | `BroadcastGetMaxElemNum` | `InitBuffer` pool 容量 | `B*elemNum*smax<=U` |
| 同上 | `blockNum` | int64/64 | 同上 | AIV core | Host 分核并 `SetBlockDim` | `GetBlockNum`/尾核 | `1..C` |
| 同上 | `outputDims[8]` | int64[8] | 同上 | 元素/轴 | 合轴输出 dims | 轴坐标与 NDDMA loopSize | inactive 轴不读 |
| 同上 | `outputStrides[8]` | int64[8] | 同上 | 元素 stride | 连续输出 stride | GM/UB offset | 末轴通常1 |
| 同上 | `inputDims[2][2]` | int64[2][2] | 同上 | former/tail 元素 | `SetInputParams` | 非 CopyInBrc 输入；本 DAG inactive | CopyBrcSize=2，正常不读 |
| 同上 | `inputBrcDims[2][8]` | int64[2][8] | 同上 | 元素/轴 | `SetInputNddmaParams` | NDDMA/UB BRC shape | 分别对应 x1/x2 |
| 同上 | `inputStrides[2][8]` | int64[2][8] | 同上 | 元素 stride | 合轴输入 stride | 普通 CopyIn；本 DAG inactive | CopyBrc 输入跳过 |
| 同上 | `inputBrcStrides[2][8]` | int64[2][8] | 同上 | 元素 stride | `SetInputNddmaParams` | `BroadcastNddma*` | broadcast 轴为0 |
| 同上 | `scalarData[16]` | char[16] | 结构尾部 | 字节 | simplified route 未调用 `SetScalar`，保持0 | simplified DAG 无 Var，不读 | full route 才使用 |
| `CastTilingData` | `blockNum` | int64/64 | sizeof=96, alignof=8, offset=0 | core | `CastTiling::DoOpTiling` | Cast core loop | `1..AIV coreNum` |
| 同上 | `ubFormer` | int64/64 | offset=8 | 元素/tile | `GetUbFormer` | CopyIn/MicroAPI/CopyOut | UB 上限 |
| 同上 | `blockFormer` | int64/64 | offset=16 | 元素/core | 8 元素对齐分核 | GM core offset | 尾核单独余量 |
| 同上 | `ubLoopOfFormerBlock` | int64/64 | offset=24 | loop | `ceil(blockFormer/ubFormer)` | 非尾核循环 | `>=1` |
| 同上 | `ubLoopOfTailBlock` | int64/64 | offset=32 | loop | `ceil(blockTail/ubFormer)` | 尾核循环 | `>=1` |
| 同上 | `ubTailOfFormerBlock` | int64/64 | offset=40 | 元素 | Host 余数公式 | 非尾核末 tile | 有效 count |
| 同上 | `ubTailOfTailBlock` | int64/64 | offset=48 | 元素 | Host 余数公式 | 尾核末 tile | 有效 count |
| 同上 | `regCopyInStep` | int64/64 | offset=56 | 元素/register step | policy 的 load distribution | `CastMicro` load | pair-specific |
| 同上 | `regCopyOutStep` | int64/64 | offset=64 | 元素/register step | policy 的 store distribution | `CastMicro` store | pair-specific |
| 同上 | `ubFormerRegLoop` | int64/64 | offset=72 | register loop | VL/输入 bit 数公式 | 整 tile | 部分 policy 为0 |
| 同上 | `ubTailOfFormerRegLoop` | int64/64 | offset=80 | register loop | Host tail 公式 | 非尾核末 tile | 同上 |
| 同上 | `ubTailOfTailRegLoop` | int64/64 | offset=88 | register loop | Host tail 公式 | 尾核末 tile | 同上 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/P | `(202,0)` + Cast selector | `LogAddExpTiling::DoSimplifiedTiling`→`BroadcastBaseTiling::DoTiling` | OneDimAdvance + 可选 Cast POD | `log_add_exp<202,0>`；可选 `cast<0>`/AiCPU Cast | `BroadcastOneDimAdvanceSch`; simplified DAG; `vec.h`; Cast body | `log_add_exp/CMakeLists.txt`; `cast/CMakeLists.txt` |
| R-002/P | `(1,0)` + Cast selector | 同上；`SetSchMode(false)` | Base POD + 可选 Cast POD | `log_add_exp<1,0>` | `BroadcastNddmaSch<false>`; `broadcast_utils.h`; DAG | 同上 |
| R-003/P | `(2,0)` + Cast selector | 同上；runningRank>5 | Base POD + 可选 Cast POD | `log_add_exp<2,0>` | `BroadcastNddmaSch<true>`; outer-loop helper; DAG | 同上 |
| R-004/P | `(101..104/109,0)` + Cast selector | `IsUbBroadcast`→`SetSchMode(true)` | Base POD + 可选 Cast POD | `log_add_exp<101..109,0>` | `BroadcastUbSch<rank>`; DAG; `vec.h` | 同上 |

## Kernel 实现

### 公共执行流程

所有核心 Route 的 entry 参数顺序都是 `(x1,x2,y,workspace,tiling)`；`workspace` 不寻址。`formulaType=0` 后，FP16/F32 选择 `LogAddExpSimplifiedCompute<T>::OpDag`，BF16 选择 `LogAddExpSimplifiedWithCastCompute<bfloat16_t>::OpDag`。公共 DAG 顺序为：两路 `CopyInBrc`→`max/min/sub`→同号无穷保护→`abs`→乘 `-1`→`exp`→稳定 `log1p`→与 `max` 相加→`CopyOut`。

前缀 `P(a,b)` 在每个核心 Route 前独立展开：`V` 的 `cast<0>` 设置 `KERNEL_TYPE_AIV_ONLY`，根据 dtype pair 的 policy 选择 Direct/Micro 模板并按 Cast POD 分核；`C` 的 `CastTask<S,D>` 对不重叠 `[start,end)` 逐元素 `static_cast`。两输入前缀输出均为连续 GM，核心完成后同 stream 执行 ViewCopy。

### Route R-001：单轴 Advance

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_logaddexp.cpp` | `aclnnLogAddExpGetWorkspaceSize` | 非空；两输入转换为 `out.dtype` | L0 LogAddExp |
| Tiling | `broadcast_tiling.h` | `BroadcastBaseTiling::DoTiling` | `dims.back().size()==1 && inputAllContiguous && 3<=4 && VarSize==0` | mode202 |
| Tiling | `log_add_exp_tiling_arch35.cpp` | `DoSimplifiedTiling` | `useFullFormula_==false` | key `(schMode,0)` |
| Router | `broadcast_sch.h` | `BroadcastSch::Process` | `schMode==202` | `BroadcastOneDimAdvanceSch` |

#### Device 数据流

##### 函数级调用链

```text
optional cast<0>/CastTask for x1 and x2
  -> log_add_exp<202,0>
    -> BroadcastSch<202, SimplifiedDag>::Process
      -> BroadcastOneDimAdvanceSch::Init/Process
        -> CopyIn(x1), CopyIn(x2)
        -> InfGuardedSub -> Abs -> Muls(-1) -> Exp
        -> StableLog1p -> Add(max,log1p)
        -> CopyOut(y)
  -> ViewCopy(y,out)
```

##### 源码等价伪码

```text
# 伪码
coreBase = blockFormer * tileNum * blockIdx
loops = lastCore ? blockTail : blockFormer
for t in [0,loops):
    valid = (lastCore && t==loops-1) ? ubTail : tileNum
    ping = t & 1
    x = scalarFlag.bit0 ? Duplicate(x1[0],valid) : CopyIn(x1[coreBase+t*tileNum],valid)
    y = scalarFlag.bit1 ? Duplicate(x2[0],valid) : CopyIn(x2[coreBase+t*tileNum],valid)
    hi=max(x,y); lo=min(x,y); d=x-y
    d = select(lo==+Inf,0,d); d=select(hi==-Inf,0,d)
    u=exp(-abs(d)); a=1+u; mid=a-1
    z=select(a!=1, log(a)*(u/mid), u)
    z=select(a!=+Inf,z,+Inf)
    CopyOut(out[coreBase+t*tileNum], hi+z, valid)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2 GM | `(coreBase+t*tileNum)*sizeof(T)`；scalar 时0 | `valid` 元素或1 scalar | T/ND连续 | 前缀节点/调用者 | `CopyIn` | 调用者/Executor | scalar 首两迭代缓存；普通线性 |
| 静态 local pool | `bufId*tileNum*smax` | `B_N*tileNum*smax` | F16/F32；BF16 中间 FP32 | scheduler/DAG | DAG后继/CopyOut | core；tile ping/pong | `GetTensor/ReleaseTensor`；LEVEL_2 复用 |
| y GM | `(coreBase+t*tileNum)*sizeof(T)` | `valid*sizeof(T)` | T/ND连续 | `CopyOut` | ViewCopy | Executor 临时 | 仅提交 valid |
| core workspace | 固定仓未公开切片 offset | 16 MiB 声明 | bytes | Executor | entry 未读 | Executor | 不得当 UB 使用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  ping["static pool ping IDs · B_N×tile×smax"]:1
  pong["static pool pong IDs · DAG按lifetime复用"]:1
  regs["Vector/MicroAPI registers · valid lanes"]:1
```

##### 字节账与复用顺序

```text
UB_total = B_N * tileNum * smax
tileNum = floor((U/B_N)/512)*512/smax（随后可能缩小）
UB_total <= U
dynamic_ub = N/A
workspace_core = 16 MiB declared, body未读取
workspace_total = Executor聚合值，固定仓未公开各节点offset
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `bufferIds[ping][pos]` | CopyIn/Vector op | DAG 后继 | 同 parity 的下一 tile | `GetTensor/ReleaseTensor` | 所有引用消费后才 Release |
| `Guard::OpSub` 结果区 | `Sub`/两次 `Select` | `Abs` | 后继允许复用时 | DAG LEVEL_2 lifetime | 两次无穷修正完成 |
| `StableLog1p` 中间区 | Adds/Div/Log/Mul/Select | 最终 Add | 当前 tile 结束 | DAG buffer-id 复用 | `a==1/+Inf` mask 已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→local | `GetTensor<VECIN>`/`ReleaseTensor` | `Vec::CopyIn` | 第一个 Vector op | local input ID | `valid` 控制尾 tile；scalar 用 Duplicate |
| DAG 节点 | `GetTensor<VECCALC>`/`ReleaseTensor` | 前驱 op | 后继 op | LEVEL_2 pool 子区 | 所有 Vector API 接收相同 `valid count` |
| local→GM | `GetTensor<VECOUT>`/`ReleaseTensor` | 最终 Add | `Vec::CopyOut` | 输出 ID | 只写 `valid` 元素 |
| FP16 | 原 dtype Vector | CopyIn | CopyOut | 数值管线 | 全流程 FP16，无内部升精度 |
| BF16 | `Cast<float,bf16,CAST_NONE>`；结果 `CAST_RINT` | 两输入 | FP32 DAG→BF16 | 精度 | 计算升 FP32，最终 RINT 降回 BF16 |
| F32 | 原 dtype Vector | CopyIn | CopyOut | 数值管线 | 全流程 FP32 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling.h` | `DoOneDimOpTilingAdvance` | tile/core/scalarFlag/POD |
| Kernel entry/router | `log_add_exp_apt.cpp` | `log_add_exp<202,0>` | dtype DAG 选择与 Process |
| Device body/helper | `broadcast_sch_one_dim_advance.h` | `Init/Process/Run/CopyIn/CopyOut` | core/tile、静态 pool、tail |
| Device body/helper | `log_add_exp_dag.h` | `InfGuardedSub/StableLog1p/Simplified*` | 完整数值管线 |

### Route R-002：连续 NDDMA（运行维数不超过 5）

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_tiling.h` | `IsUbBroadcast` | multi-axis 且返回 false | NDDMA |
| Tiling | 同上 | `SetSchMode` | `shapeLen-ubSplitAxis<=5` | mode1 |
| Router | `broadcast_sch.h` | `schMode==1` | `BroadcastNddmaSch<OpDag,false>` | 执行 R-002 |

#### Device 数据流

##### 函数级调用链

```text
log_add_exp<1,0> -> BroadcastNddmaSch<SimplifiedDag,false>
  -> BroadcastGetAxesIndices
  -> CopyInBrc(x1,x2) via DataCopy<T,5> or aligned DataCopyPad
  -> same simplified DAG
  -> CopyOut(y)
```

##### 源码等价伪码

```text
# 伪码
decode core start into axesIndices
for ubLoopIdx in core tile range:
    split = lastGlobalTile ? ubTail : ubFormer
    for each input:
        gmOffset=sum(axes*inputStride)+axisIndex*inputStride[splitAxis]*ubFormer
        if outputStride[splitAxis] != inputStride[splitAxis]:
            build 5D loopSize/srcStride/dstStride; DataCopy<T,5>(GM->pool)
        else: DataCopyPad(blockCount=1, blockLen=split*inputStride[axis]*sizeof(T))
    run guarded stable DAG on split*outputStride[axis] valid elements
    CopyOut to output GM; increment axesIndices
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| broadcast input GM window | `BroadcastGetGmOffset(axes,inputBrcStride,axis,ubFormer)` | 5D logical tile | T/合轴 ND | Executor | NDDMA/DataCopyPad | tile | stride0 轴重复读取源值 |
| `TBuf<VECCALC>` pool | `bufId*elemNum*smax` | `B_N*elemNum*smax` | local linear | NDDMA/DAG | DAG/CopyOut | core；tile | `InitBuffer` 单物理 pool |
| output GM | output stride offset | `split*outputStride[axis]` 元素 | T/ND | CopyOut | ViewCopy | Executor | tail只写有效区 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  nddma["NDDMA CopyInBrc IDs"]:1
  calc["DAG intermediates · LEVEL_2 reuse"]:1
  out["CopyOut ID"]:1
```

##### 字节账与复用顺序

```text
UB_total = B_N * elemNum * smax <= U
dynamic_ub = N/A
workspace_core = 16 MiB declared; entry/body未读取
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| NDDMA input ID | `DataCopy<T,5>` | DAG first users | 下一 tile | scheduler position token + DAG Release | 当前 tile 所有引用已结束 |
| DAG pool IDs | 各 Vector op | 后继 op | 本 tile 后继或下一 tile | `Get/ReleaseTensor` | LEVEL_2 lifetime 分析 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA→Vector | position token/pool ownership | MultiCopy/DataCopyPad | DAG | input ID | `ubTail` 产生真实 `split` |
| Vector→GM | pool ownership | final Add | CopyOut | output ID | FP16/BF16/F32 规则同 R-001 |
| 轴推进 | `BroadcastUpdateAxesIndices` | 当前 tile | 下一 tile offset | GM window | 完整覆盖 fused core range |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling.h` | `DoBroadcastOpTiling/SetSchMode` | mode1、Base POD、blockDim |
| Kernel entry/router | `broadcast_sch.h` | `BroadcastSch<1>` | NDDMA without outer loop |
| Device body/helper | `broadcast_sch_nddma.h` | `BroadcastNddmaSch::Init/Process/Run` | pool、tile、DAG 调用 |
| Device body/helper | `broadcast_utils.h` | `BroadcastNddmaWithoutLoop` | 5D MultiCopy 参数与 aligned fallback |

### Route R-003：连续 NDDMA 外循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_tiling.h` | `SetSchMode` | UB selector false 且 `shapeLen-ubSplitAxis>5` | mode2 |
| Router | `broadcast_sch.h` | `schMode==2` | `BroadcastNddmaSch<OpDag,true>` | 执行 R-003 |

#### Device 数据流

##### 函数级调用链

```text
log_add_exp<2,0> -> BroadcastNddmaSch<SimplifiedDag,true>
  -> software outer-axis loop
    -> inner DataCopy<T,5>
    -> simplified DAG -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
gmBase = BroadcastGetGmOffset(...)
nddmaProduct = product(outputDims[splitAxis+1 : shapeLen-5]) * split
for outer in [0,nddmaProduct):
    update nddmaIndices with carry
    ubOff=sum(nddmaIndices*outputStride)
    gmOff=sum(nddmaIndices*inputStride)
    DataCopy<T,5>(pool[ubOff], input[gmBase+gmOff], innerParams)
run the same DAG and CopyOut for the completed logical tile
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| outer GM windows | outer coordinate·input stride | inner≤5D window | T/ND | offset helper | inner NDDMA | iteration | 高维不物化 |
| local pool | `bufId*elemNum*smax` | `B_N*elemNum*smax` | local | NDDMA/DAG | DAG/CopyOut | tile | 与 R-002 相同 pool |
| outer coordinates | register arrays | 最多3个抛出轴索引 | int64 | software loop | offset helper | current tile | 不占 GM workspace |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  coords["outer coordinates · registers"]:1
  pool["inner 5D NDDMA + DAG pool"]:1
```

##### 字节账与复用顺序

```text
UB_total = B_N * elemNum * smax <= U
outer coordinate state = register/stack-like local scalar state
workspace_core = 16 MiB declared, body未读
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inner pool slice | 每次 inner NDDMA | DAG | 下一 outer/inner iteration | 当前逻辑 tile NDDMA 完成后进入 DAG | ubOff 不重叠或前值已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| outer→inner | program order | coordinate update | MultiCopy | GM/UB offset | carry 覆盖高维组合 |
| NDDMA→DAG | scheduler token | inner copies | first Vector op | complete tile | 末 tile `split=ubTail` |
| DAG | 同 R-001 | inputs | output | valid lanes | dtype 精度同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling.h` | `SetSchMode` | runningRank>5 选 mode2 |
| Kernel entry/router | `broadcast_sch.h` | `BroadcastSch<2>` | withLoop=true |
| Device body/helper | `broadcast_utils.h` | `BroadcastNddmaWithLoop` | 外轴软件循环+5D NDDMA |

### Route R-004：连续 UB Broadcast

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_tiling.h` | `IsBigNLastBrc/IsUbBroadcast` | 大 NLast，或 F16/BF16 尾轴32B对齐 | UB broadcast |
| Tiling | 同上 | `SetSchMode(true)` | runningRank 1..4/大于4 | 101..104/109 |
| Router | `broadcast_sch.h` | modes101..109 | `BroadcastUbSch<rank>` | 执行 R-004 |

#### Device 数据流

##### 函数级调用链

```text
log_add_exp<101..104/109,0> -> BroadcastUbSch<SimplifiedDag,rank>
  -> CopyInAndVecBrc(x1,x2)
    -> DataCopyPad compact source -> AscendC Broadcast to expanded local tile
  -> simplified DAG -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
for each core tile:
    derive input compact shape/stride and output tile shape from Base POD
    if a broadcast input can be cached across current axis, retain its compact source/result
    else CopyIn only unique source region
    Broadcast(compact, expanded, srcShape, dstShape, static rank 1..4 or runtime rank)
    run guarded stable DAG over valid expanded output lanes
    CopyOut only ubFormer/ubTail logical region
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| compact source GM/UB | axes·inputBrcStride | 非广播 unique region | T/compact | DataCopyPad | UB Broadcast | tile/cache | 避免重复搬入广播副本 |
| expanded local input | buffer ID·`elemNum*smax` | full logical tile | T/local ND | Broadcast | DAG | tile | 与 compact 区不同 live ID |
| DAG/output pool | buffer ID·`elemNum*smax` | `B_U*elemNum*smax` 总池内子区 | F16/F32 | DAG | CopyOut | tile | LEVEL_2 reuse |
| y GM | output stride offset | valid tile | T/ND | CopyOut | ViewCopy | Executor | padding lane 不提交 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  compact["compact CopyIn regions"]:1
  expanded["expanded broadcast regions"]:1
  dag["DAG mids/output · reused IDs"]:1
```

##### 字节账与复用顺序

```text
UB_total = B_U * elemNum * smax <= U
mode109仅把 rank 参数改为运行时，不增加 GM workspace
workspace_core = 16 MiB declared, body未读
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| compact/cache ID | DataCopyPad | Broadcast | cache 失效或下一需要重载的 tile | scheduler 条件+position token | stride/broadcast 轴保持源值不变 |
| expanded ID | Broadcast | DAG | 下一 tile | DAG 输入引用释放 | 当前 tile 消费完成 |
| DAG IDs | Vector ops | 后继/CopyOut | lifetime 结束 | `Get/ReleaseTensor` | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compact→expanded | position token | DataCopyPad | Broadcast | compact ID | 源 shape 只覆盖 unique 值 |
| expanded→DAG | position token | Broadcast | first Vector op | expanded ID | valid shape 来自 POD |
| DAG→GM | pool ownership | final Add | CopyOut | output ID | `ubTail` 防止 pad lane 写出 |
| dtype | 同 R-001 | inputs | output | 数值路径 | FP16 native；BF16升FP32；F32 native |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling.h` | `IsUbBroadcast/SetSchMode` | route 谓词与 101..109 |
| Kernel entry/router | `broadcast_sch.h` | `BroadcastSch<101..109>` | 静态/动态 rank scheduler |
| Device body/helper | `broadcast_sch_ub_broadcast.h` | `CopyInAndVecBrc/Run` | compact 搬入、UB 展开、cache/tail |
| Device body/helper | `log_add_exp_dag.h` | simplified DAG | 广播后完整计算 |

## 算子特定优化

### Trick T-001：稳定 `log1p` 避免小量被 `1+x` 舍入吞掉

#### 触发与源码依据

- 适用 Route：R-001～R-004 的全部 dtype/topology 变体
- 触发条件：计算 `u=exp(-abs(x-y))` 后无条件进入 `StableLog1p`，mask 决定普通、小量与 `+Inf` 分支
- Golden 来源：[`math/log_add_exp/op_kernel/arch35/log_add_exp_dag.h:62-77`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_kernel/arch35/log_add_exp_dag.h#L62-L77)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using OpMid = Bind<Vec::Adds<CT>, OpAddOne, ConstNegOne>;
using OpRatio = Bind<Vec::Div<CT>, In, OpMid>;
using OpMul = Bind<Vec::Mul<CT>, OpLog, OpRatio>;
using FixSmall = Bind<Vec::Select<uint8_t, CT, SEL_TT_MODE>, MaskNotOne, OpMul, In>;
```

#### 机制与设计原因

普通 `ln(1+u)` 在 `u` 很小时可能先把 `1+u` 舍入成 1。源码同时构造 `mid=(1+u)-1` 与 `ln(1+u)*(u/mid)`；当 `1+u==1` 时直接返回 `u`，否则使用校正乘积。另一个 mask 在 `1+u==+Inf` 时强制输出 `+Inf`，避免校正比值破坏边界。所有 Vector API 的 `count` 都是当前 tile 的真实有效元素数。

#### 迁移边界

- 适用：`u>=0` 且目标是 `log1p(u)` 的逐元素 Vector DAG
- 前提：Compare/Select、Div、Log 与 Mul 的 dtype/精度路径可用；无效 lane 不写出
- 失效：不能把该公式直接用于 `u<=-1` 或需要复数域的 `log1p`
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓可验证的被规避做法是只执行 `Log(Adds(u,1))`：当 `Adds` 结果等于 1 时会丢失小量；源码以 `FixSmall` 返回 `u`。

### Trick T-002：同号无穷差值保护

#### 触发与源码依据

- 适用 Route：R-001～R-004
- 触发条件：`x1/x2` 同为 `+Inf` 或同为 `-Inf`
- Golden 来源：[`math/log_add_exp/op_kernel/arch35/log_add_exp_dag.h:46-59`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_kernel/arch35/log_add_exp_dag.h#L46-L59)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using MaskBothPos = Bind<Vec::Compare<uint8_t, CT, CMP_EQ_MODE>, OpMin, ConstPosInf>;
using SubFixPos = Bind<Vec::Select<uint8_t, CT, SEL_TT_MODE>, MaskBothPos, DupZero, OpSub>;
using MaskBothNeg = Bind<Vec::Compare<uint8_t, CT, CMP_EQ_MODE>, OpMax, ConstNegInf>;
using SubFixed = Bind<Vec::Select<uint8_t, CT, SEL_TT_MODE>, MaskBothNeg, DupZero, SubFixPos>;
```

#### 机制与设计原因

直接 `x1-x2` 对 `Inf-Inf` 或 `-Inf-(-Inf)` 产生 NaN，随后 `abs/exp/log` 会污染结果。源码用 `min==+Inf` 检出双正无穷，用 `max==-Inf` 检出双负无穷，并把差值改为 0；后续得到 `max+ln(2)`，最终仍为正确符号的无穷。该 guard 只覆盖两类明确边界，普通 NaN 仍按 Vector primitive 传播。

#### 迁移边界

- 适用：使用稳定式 `max(x,y)+f(abs(x-y))` 且数学上同号无穷结果应保持无穷的算子
- 前提：输入为浮点；Compare/Select 对 ±Inf 语义与源码一致
- 失效：不可泛化为把所有 NaN 差值清零；非同号/NaN 输入没有该保证
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓给出真实反例：不加 guard 的 `Inf-Inf` 会产生 NaN，破坏最终 `+Inf`；`-Inf-(-Inf)` 同理。

### Trick T-003：公开入口专门化为 simplified DAG，BF16 单独升精度

#### 触发与源码依据

- 适用 Route：全部；BF16 使用 WithCast 子族
- 触发条件：公开入口固定 `base=-1,scale=1,shift=0`；核心 dtype 为 BF16 时启用 FP32 中间
- Golden 来源：[`math/log_add_exp/op_kernel/log_add_exp_apt.cpp:24-48`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/log_add_exp/op_kernel/log_add_exp_apt.cpp#L24-L48)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
    using OpDag = LogAddExpOp::LogAddExpSimplifiedWithCastCompute<DTYPE_X1>::OpDag;
    BroadcastSch<schMode, OpDag> sch(tiling);
```

#### 机制与设计原因

Host 的 `formulaType=0` 让编译期 router 删除 base/scale/shift 的 full 运算和四个 scalar 读取；BF16 又在 simplified DAG 内先 `CAST_NONE` 到 FP32，完成 guard、Exp、Log 与校正后以 `CAST_RINT` 降回 BF16。FP16 则选择原 dtype simplified DAG，不能误写成同样升 FP32。

#### 迁移边界

- 适用：属性默认值构成高频固定公式、且编译期 key 能删除通用路径的逐元素算子
- 前提：Host key 与 Kernel `formulaType` 位布局一致；BF16 的 FP32 UB 容量按 `smax=4` 计算
- 失效：任何非默认 base/scale/shift 必须进入 full DAG；该公开 API 不提供这种入口
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 BF16 的 UB 账按 2 字节中间计算会低估资源；源码明确实例化 FP32 中间。固定仓未提供公开 ACLNN 触发 full formula 的可验证用例。

## 模板与实现对比

| 对比维度 | R-001 单轴 | R-002 NDDMA≤5 | R-003 NDDMA外循环 | R-004 UB Broadcast | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | 合轴 rank1，最高优先 | 多轴、UB false、runningRank≤5 | 多轴、UB false、runningRank>5 | 多轴、UB true，先于 NDDMA rank 判断 | shape collapse→broadcast介质→运行维数 |
| 模板与 dtype | `OneDimAdvanceSch`；F16/F32/BF16三 DAG | `NddmaSch<false>`；同 DAG | `NddmaSch<true>`；同 DAG | `UbSch<1..4/-1>`；同 DAG | 核心公式完全共享 |
| 循环与分核 | core→tile；核内 ping/pong | core→ub tile→≤5D copy | core→ub tile→outer→5D copy | core→tile→compact/expand | 5D NDDMA 表达能力 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 静态 buffer IDs；寄存器；无 TBuf | 单 TBuf pool+NDDMA | 单 TBuf pool+outer坐标 | 单 TBuf pool+Broadcast shape state | 均无 TQue、dynamic UB、通信 window |
| 内存总占用与复用 | `B_N*tile*smax` | `B_N*elemNum*smax` | 同 R-002 | `B_U*elemNum*smax` | DAG LEVEL_2 lifetime 分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Duplicate + Vector | DataCopy<T,5>/Pad + Vector | 多次5D DataCopy + Vector | compact Copy + Broadcast + Vector | 无 SIMT/通信 |
| 同步、精度与 tail | buffer token；真实 count | NDDMA token；ubTail | outer program order；ubTail | Copy→Broadcast token；ubTail | BF16升FP32；FP16 native；F32 native |
| fallback/失败边界 | tile为0失败 | 无 Kernel fallback | 无 Kernel fallback | selector false回 NDDMA | 不存在 Cube/AIC fallback |

前缀 `P(a,b)` 与四个核心 scheduler 正交：`N` 不增加节点，`V` 增加 AIV Cast，`C` 增加 AiCPU Cast；两输入的组合改变 launch graph 节点数和 GM 临时边，但不改变核心 Tiling 与 simplified DAG。

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档支持，但固定 `ops-math` 提交内未公开对应 Host/Tiling/Kernel 源码闭包 | 同 A2 | 公开支持且 source_closed |
| Host/Launcher | ACLNN/L0 launcher 可见；后续实现固定仓不可见 | 同 A2 | arch35 `LogAddExpTiling` + ATVOSS Broadcast |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | AIV-only；mode202/1/2/101..109；NDDMA/UB pool；16 MiB 节点 workspace 声明 |

## 限制与失败边界

- 本卡只覆盖 `aclnnLogAddExp`；不覆盖 `aclnnLogAddExp2`，也不发布内部 full-formula 属性路径。
- 公开接口只承诺 ND。950 实现的 `CheckFormat` 对相同的非私有 storage format 更宽松，但未据此扩大发布契约。
- 任一输入为空时成功且 `workspaceSize=0`；非空核心要求合轴元素数、UB、coreNum、tile 与 blockNum 均有效。
- `BroadcastOneDimTilingDataAdvance::tileNum/blockNum` 从更宽计算结果窄化为 int32/int16，固定源码没有显式溢出 guard。
- FP16 simplified 核心不升 FP32；BF16 升 FP32 后 RINT；调用方不能假设三种输出 dtype 具有相同中间精度。
- 前置 `UINT64→F16/F32/BF16` 与 `DOUBLE→F16` 使用 AiCPU Cast；固定源码只证明逐元素 `static_cast` 与任务分片，不证明和 AIV pair 的所有极值舍入完全一致。
- Executor 的多节点 workspace offset、别名复用、内部 event 及 ViewCopy/Contiguous 具体 Kernel 属框架边界，固定算子源码不可见；必须使用公开第一段返回值。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/log_add_exp/docs/aclnnLogAddExp.md` | 产品表、公式、参数/错误表 | 950 公开性、dtype/shape/layout、接口语义 |
| 入口声明/约束 | `math/log_add_exp/op_api/aclnn_logaddexp.h` | 两段式声明 | C ABI 与参数顺序 |
| 入口实现 | `math/log_add_exp/op_api/aclnn_logaddexp.cpp` | `CheckParams/aclnnLogAddExpGetWorkspaceSize` | 校验、空域、Contiguous/Cast、固定属性、Executor |
| 入口实现 | `math/log_add_exp/op_api/logaddexp.cpp` | `LogAddExp/LogAddExpAiCore` | broadcast 输出、AICore launcher |
| Host/Tiling/Launcher | `math/log_add_exp/op_host/log_add_exp_def.cpp` | `LogAddExp` OpDef | 950 config、dtype、属性、kernel 文件 |
| Host/Tiling/Launcher | `math/log_add_exp/op_host/arch35/log_add_exp_tiling_arch35.cpp` | `LogAddExpTiling` | simplified/full selector、dtype DAG、key、compile info |
| Host/Tiling/Launcher | `math/log_add_exp/op_host/config/ascend950/log_add_exp_binary.json` | `op_list` | 950 F16/F32/BF16 binary 实例 |
| Kernel entry/body | `math/log_add_exp/op_kernel/arch35/log_add_exp_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode + formulaType key ABI |
| Kernel entry/body | `math/log_add_exp/op_kernel/arch35/log_add_exp_dag.h` | `InfGuardedSub/StableLog1p/Simplified*` | 完整数值 DAG 与精度路径 |
| Kernel entry/body | `math/log_add_exp/op_kernel/log_add_exp_apt.cpp` | `log_add_exp` | entry、dtype/formula router |
| 构建接线 | `math/log_add_exp/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | 仅 ascend950/arch35 自定义实现 |
| 入口实现 | `math/cast/op_api/cast.cpp` | `Cast/IsAiCoreSupport/CastAiCpu` | 前置 Cast selector 与 launch |
| Host/Tiling/Launcher | `math/cast/op_host/arch35/cast_tiling.cpp` | `CastTiling::DoOpTiling` | Cast POD、block/tile/workspace |
| Kernel entry/body | `math/cast/op_kernel/arch35/cast_struct.h` | `CastTilingData/GetCastPolicy` | Cast ABI 与 policy |
| Kernel entry/body | `math/cast/op_kernel/arch35/cast_impl.h` | `CastDirect/CastMicro` | AIV Copy/Compute/CopyOut |
| Kernel entry/body | `math/cast/op_kernel/cast_apt.cpp` | `cast<0>` | AIV-only router |
| Kernel entry/body | `math/cast/op_kernel_aicpu/cast_aicpu.cpp` | `CastTask/SetMap` | AiCPU 可达 pair 与逐元素 body |
| 构建接线 | `math/cast/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | 950 Cast 构建 |
| Host/Tiling/Launcher | `pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | Broadcast POD/mode 宏 | POD 字段与合法 modes |
| Host/Tiling/Launcher | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling` | selector、tile/core、workspace、POD producer |
| Host/Tiling/Launcher | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse/DoBrodcastTiling` | 合轴、stride0、UB/分核公式 |
| Host/Tiling/Launcher | `src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp` | `BroadcastGetMaxElemNum` | pool 容量与 repeat 对齐公式 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch::Process` | AIV-only 与 mode router |
| Kernel entry/body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch` | mode202 core/tile/pool/tail |
| Kernel entry/body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h` | `BroadcastNddmaSch` | mode1/2 pool 与调用链 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h` | `BroadcastUbSch` | mode101..109 compact/expand/tail |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/broadcast_utils.h` | `BroadcastNddma*` | 5D NDDMA 参数与外循环 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | LEVEL_2 buffer lifetime/IDs |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/vec.h` | `CopyInBrc/Max/Sub/Exp/Log/Select/CopyOut` | Vector primitive 实参语义 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | OneDimAdvance 或 Base POD；key=`(schMode,formulaType)` | 字段位宽/顺序、formulaType=0、scalarData inactive 条件 | `log_add_exp_struct.h`; `broadcast_base_struct.h` |
| 调用顺序 | `Contiguous→Cast?→LogAddExp→ViewCopy` | 两输入必须先转成同一 `out.dtype`；前缀失败阻断核心 | `aclnnLogAddExpGetWorkspaceSize` |
| workspace/内存 | 核心/AIV Cast 各声明16 MiB；UB pool 按 `B*smax` | 不能把节点 workspace 当 UB，不能自行相加替代 Executor 聚合值 | `BroadcastBaseTiling::DoTiling`; `CastTiling::GetWorkspaceSize` |
| 同步 | 同 stream GM 边；节点内 buffer ownership/token | NDDMA/Broadcast 消费前不得覆盖 pool ID | `BroadcastOneDimAdvanceSch`; `BroadcastNddmaSch`; `BroadcastUbSch` |
| 精度与尾块 | FP16 native、BF16 FP32+RINT、FP32 native；所有 op 使用真实 count | 不得把 BF16 pool 按2B记账；不写 padding lane | `LogAddExpSimplified*`; scheduler tail |
| 失败边界 | 空域成功短路；dtype/shape/format/UB/key 失败无核心 fallback | 不得用 dormant full route 或 LogAddExp2 补失败 | `CheckParams`; `LogAddExpTiling`; `BroadcastSch` |
