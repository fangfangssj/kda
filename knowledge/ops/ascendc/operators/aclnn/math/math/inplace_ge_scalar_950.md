---
type: Operator
title: "InplaceGeScalar 950 ACLNN 算子知识"
description: "说明 aclnnInplaceGeScalar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
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
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/view_copy/op_api/view_copy.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_host/greater_equal_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_host/arch35/greater_equal_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_host/config/ascend950/greater_equal_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/greater_equal_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/arch35/greater_equal_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/arch35/greater_equal_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel_aicpu/greater_equal_aicpu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
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
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/node.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
created_at: "2026-08-25T11:06:45Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceGeScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceGeScalarGetWorkspaceSize` + `aclnnInplaceGeScalar`，无版本后缀，公开 ACLNN 两段式入口 |
| 计算语义 | 对原 `selfRef` 的每个元素比较 `selfRef[i] >= other`，将 BOOL 结果转换回 `selfRef.dtype` 后原地写回；数值 dtype 的结果为 0/1 |
| 输入输出 | `selfRef` 为 ND、0～8 维、可非连续；950 支持 DOUBLE/BFLOAT16/FLOAT16/FLOAT/INT32/UINT32/INT64/UINT64/INT16/UINT16/INT8/UINT8/BOOL；`other` 为同一公开集合的 scalar |
| 执行类型 | R-001 核心 `GreaterEqual` 节点为 `AIV_VECTOR`；R-002 为 AiCPU scalar-broadcast；前后 `Contiguous/Cast/ViewCopy/executor` 是直接依赖边界；无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001：promote dtype 命中 950 AiCore 集，`schMode=202`；R-002：其余可执行 promote dtype 走 AiCPU `Y_ONE_ELEMENT`；空 Tensor 在 Host 成功短路 |
| 核心资源 | R-001 使用 GM、9 个统一长度的静态 AIV buffer-id UB 子区与 Vector 寄存器；R-002 使用 AiCPU 地址空间和可选 CPU 分片；executor 聚合整链 workspace |
| 来源闭包 | `source_closed`：公开原地入口、950 分发、AIV Tiling/POD/router/body、AiCPU fallback、直接依赖边界与 950 构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `N=numel(selfRef)`，第一段接口保留原输入值并构造执行图；非空域对 `0 <= i < N` 执行：

```text
p = A5 类型推导(selfRef.dtype, other.dtype)
cmp[i] = cast(selfRef_before[i], p) >= cast(other, p)  # BOOL
selfRef_after[i] = cast(cmp[i], selfRef.dtype)         # BOOL 或数值 0/1
```

原地语义不是 `greater_equal` 核心 entry 直接把 `x1` 当 `y`：核心先写 executor 临时 BOOL Tensor，随后 Cast 到 `selfRef.dtype`，最后 `ViewCopy(..., selfRef)` 提交。因而不得与非原地 `aclnnGeScalar` 合并，也不得把 Tensor 变体的广播语义并入本卡。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND；0～8 维；允许非连续 | DOUBLE、BFLOAT16、FLOAT16、FLOAT、INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL | 同一公开对象既是输入又是输出；非空时先读取旧值，最后回写 0/1 或 BOOL |
| `other` | Host scalar；物化后为 0 维/单元素 Tensor | 与 `selfRef` 相同公开集合 | 必须与 `selfRef` 满足 950 类型推导；转换到 `promoteType` 后参与比较 |
| `workspaceSize` | 标量输出 | `uint64_t` | 空 Tensor 为 0；非空时由 executor 汇总连续化、Cast、GreaterEqual 与 ViewCopy 任务需求 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并转移所有权，第二段与调用者给定 workspace/stream 一起执行 |

### 入口与版本

公开头文件只声明无版本后缀的 `aclnnInplaceGeScalar*`。实现调用 `aclnnGeScalarCommon(selfRef, other, selfRef, ...)`，复用非原地公共建图逻辑，但输出别名、回写目标和接口错误语义仍属于独立的原地逻辑算子。公开文档明确支持 Ascend 950PR/950DT；`greater_equal_def.cpp`、Ascend 950 binary config、`scripts/kernel/binary_config/ascendc_config.json` 与 `CMakeLists.txt` 均接到 950 `greater_equal_apt/arch35` 实现。

## Host/Launcher 执行流程

### 参数校验与前处理

真实顺序如下：

1. `CREATE_EXECUTOR()`，失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`；随后要求 `workspaceSize != nullptr`。
2. `CheckNotNull(selfRef, other, selfRef)`；空指针返回 `ACLNN_ERR_PARAM_NULLPTR`。
3. RegBase 使用 `REGBASE_DTYPE_SUPPORT_LIST` 校验 `selfRef`；`CheckPromoteType` 先把 scalar dtype 归一化为默认 category，再调用 `CombineCategoriesWithComplex(selfRef.dtype, scalarDefaultDtype)`；BOOL promote 被改成 INT8。
4. 校验两端能转换到 `promoteType`，并校验 BOOL 比较结果能转换回 `selfRef.dtype`；失败返回 `ACLNN_ERR_PARAM_INVALID`。
5. `CheckShape(selfRef,selfRef)` 仍执行两次最大 8 维和 shape 相等校验；原地别名使 shape 相等恒成立。
6. `selfRef->IsEmpty()` 时写 `workspaceSize=0`、释放 executor 所有权并成功返回，不创建计算节点。
7. `l0op::Contiguous(selfRef)` 产生连续核心输入；`ConvertToTensor(other,promoteType)` 物化 scalar；`l0op::Cast` 把连续输入变为 `promoteType`，同 dtype 时直接返回原 Tensor。
8. `l0op::GreaterEqual(selfCasted,otherTensor)` 按 `promoteType` 选择 AiCore 或 AiCPU，产生连续 BOOL 临时 Tensor。
9. `l0op::Cast(greaterEqualResult,selfRef.dtype)` 把 BOOL 变为公开原地 dtype；`l0op::ViewCopy(...,selfRef)` 按原 view/stride 回写。
10. `GetWorkspaceSize()` 聚合整图需求；第二段 `CommonOpExecutorRun` 在同一 stream 上执行。

### 参数、Workspace 与生命周期

`selfRef` 的旧值必须存活到连续化/核心读取完成；scalar Tensor、可选输入 Cast 临时量、BOOL 核心输出、输出 Cast 临时量均由 executor 持有到其最后消费者。`ViewCopy` 完成前不可释放或覆盖这些临时量。

R-001 的 `BroadcastBaseTiling::DoTiling` 将核心 `GreaterEqual` 的 `workspaces[0]` 设为 `DEFAULT_TILING_WORKSPACE_SIZE = 16*1024*1024` 字节，虽然 `greater_equal` entry 的 `workspace` 参数未被 body 读取。R-002 的 AiCPU task space 和 Cast/ViewCopy 直接依赖可能另有需求；公开 `workspaceSize` 只能取 `uniqueExecutor->GetWorkspaceSize()` 的聚合结果，固定源码没有把它化简为单一常量。

### 启动图与失败传播

```text
aclnnInplaceGeScalarGetWorkspaceSize
  -> [framework] Contiguous(selfRef)
  -> [executor] ConvertToTensor(other, promoteType)
  -> [optional direct dependency] Cast(selfRef_contiguous -> promoteType)
  -> R-001 GreaterEqual AiCore: greater_equal<202>(x1, scalar, boolTmp, workspace, tiling)
     or R-002 GreaterEqual AiCPU: GreaterEqualCpuKernel(Y_ONE_ELEMENT)
  -> [direct dependency] Cast(boolTmp -> selfRef.dtype)
  -> [framework] ViewCopy(castTmp, selfRef)
aclnnInplaceGeScalar
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

| 节点/边 | entry 与参数 | blockDim/POD | 交接、ordering、owner/lifetime | 失败传播 |
|---|---|---|---|---|
| 连续化/标量物化 | `Contiguous`、`ConvertToTensor` | 框架内部 | executor 拥有；同一执行图中先于 Cast/比较 | 返回空指针即第一段失败 |
| 可选输入 Cast | `Cast(selfContiguous,promoteType)` | 同 dtype时无节点；否则由 Cast L0 选择 AiCore/AiCPU | 输出连续 Tensor，活到比较节点结束 | Cast 不支持或 launcher 失败则阻断比较 |
| R-001 核心 | `greater_equal<202>(x1,x2,y,workspace,tiling)` | `blockDim=blockNum`；16-byte `BroadcastOneDimTilingDataAdvance` | 读两个 promote dtype Tensor，写 BOOL GM；executor owner | Tiling/launcher 失败则无输出，后续不执行 |
| R-002 核心 | `GreaterEqualCpuKernel::Compute` | 无 AscendC POD；CPU context 输入/输出 | scalar 为 input1；直接写 BOOL 临时 Tensor | 参数/广播/ParallelFor 失败向 executor 传播 |
| 比较→输出 Cast | BOOL Tensor，extent=`N` bytes | Cast 自有参数 | 同一 executor/stream 顺序；Cast 输出活到 ViewCopy | 返回空指针阻断回写 |
| 输出 Cast→原地回写 | dtype=`selfRef.dtype`，shape 与 selfRef 相同 | ViewCopy 自有 shape/stride tensors | `ViewCopy` 最后消费，完成后 `selfRef` 新值可见 | ViewCopy 建图/执行错误由第二段返回 |

## Route 与参数设计

### 分核与资源计算

R-001 中两个输入和内部输出都连续；x2 为单元素 Tensor，scalar broadcast 在所有原始轴一致，公共 `DimensionCollapse` 将其折叠为长度 `N` 的一维域。DAG 的 `InputSize+OutputSize=3<=4` 且 `VarSize=0`，因此选择 `DoOneDimOpTilingAdvance` 与 `schMode=202`。

记 `s=max(sizeof(promoteType),sizeof(uint8_t))=sizeof(promoteType)`、950 AIV 核数为 `C`、UB 字节数为 `U`。该 DAG 在 `MemLevel::LEVEL_2` 下有 2 路输入 ping/pong（4 槽）、1 路输出 ping/pong（2 槽）、Compare/DupZero/DupOne 三个 temp 槽，故 `B=9`：

```text
tile0 = floor((U / B) / 512) * 512 / s
if C * (512 / s) > N:
    tile = 512 / s
else if C * tile0 > N:
    tile = floor((N / C) / (512 / s)) * (512 / s)
else:
    tile = tile0
ubOuter = ceil(N / tile)
blockFormer0 = ceil(ubOuter / C)
blockFormer = 2 if blockFormer0 == 1 else blockFormer0
blockNum = ceil(ubOuter / blockFormer)
```

Device 重算 `ubTail=N-(ubOuter-1)*tile` 与 `blockTail=ubOuter-(blockNum-1)*blockFormer`。最后一核执行 `blockTail` 个 tile，且最后 tile 仅处理 `ubTail` 个元素；其他核执行 `blockFormer` 个整 tile。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 AIV one-dim scalar | `GET_TPL_TILING_KEY(202)`；`GreaterEqualCompute<T>::OpDag` | 非空、950，且 ACLNN 可达 `promoteType` 属于 `{FLOAT,FLOAT16,BFLOAT16,INT32,INT64,UINT64,INT8,UINT8}`；L0 虽支持 BOOL，API 已先把 BOOL promote 改为 INT8，entry 的 BOOL 特化只保留图入口通用性 | 不在 AiCore 集时进入 R-002 |
| 2 | R-002 AiCPU scalar | `GreaterEqualCpuKernel<T>` | 非空、950，且 `promoteType` 不在 R-001 集但被 API/AiCPU 支持；当前闭合集为 `{DOUBLE,INT16,UINT16,UINT32}` | dtype/promotion/Cast 不成立则第一段失败 |

```text
if selfRef/other/workspaceSize is null -> parameter error
if selfRef dtype unsupported, promote undefined, cast invalid, or rank > 8 -> parameter error
if N == 0 -> success, workspaceSize=0, no compute node
promoteType = A5 CombineCategoriesWithComplex(selfRef.dtype, scalarDefaultDtype)
if promoteType == BOOL -> promoteType = INT8
if promoteType in A5 GreaterEqual AiCore list -> R-001, schMode=202
else -> R-002 AiCPU
after either route -> Cast(BOOL,selfRef.dtype) -> ViewCopy(...,selfRef)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 AIV one-dim scalar | key `202` | 950；非空；promote dtype 命中 AiCore list；连续 x1 + scalar x2 | 单个算子拥有核心 `greater_equal<202>`，前后可有 Cast/框架任务 | Host：`GreaterEqualCompute<int64_t/int32_t/half/float/uint8_t/int8_t/uint64_t>`；Device：`GreaterEqualCompute<DTYPE_X1>`，BOOL 特化 int8 | 9 个等长 UB slot；input/output ping-pong；Mutex Get/Release；无 Atomic | 最后一核/最后 tile 用 `blockTail/ubTail`；Tiling 失败无核心 fallback |
| R-002 AiCPU scalar | L0 `else` + `Y_ONE_ELEMENT` | 950；非空；promote dtype 为 DOUBLE/INT16/UINT16/UINT32 | 单个 `GreaterEqual` AiCPU task；前后可有 Cast/框架任务 | `GreaterEqualCpuKernel::NoBcastCompute<T>`→`SpecialCompute(Y_ONE_ELEMENT)` | 无 AIV UB/POD；CPU 直接读输入地址，`N>=32768` 时 `ParallelFor` | 每 shard 为 `[start,end)`；ParallelFor 失败返回错误，无下一核心 Route |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 AiCore dtype | `IsRegBase && CheckType(promoteType,REGBASE_DTYPE_SUPPORT_LIST)` in L0；本 ACLNN 的 BOOL 已规范化为 INT8 | binary config 中九种输入 dtype 有 AIV body，其中 ACLNN 可达八种 | R-001 | R-002 AiCPU | `greater_equal.cpp::IsAiCoreSupport` |
| one-dim advance | 合轴 rank=1、输入连续、I/O≤4、`VarSize<=0` | 使用紧凑 POD 与静态 buffer-id 双缓冲 | R-001 | 公共 broadcast 其它 mode；本 scalar ACLNN 链不可达 | `BroadcastBaseTiling::DoTiling` |
| UB 512-byte floor | `floor((U/B)/512)>0` | 每 slot 的统一 `blockLen=tile*s` 按 512 字节切分 | R-001 | Tiling `GRAPH_FAILED` | `GetOneDimUBFormerAdvance` |
| 小数据缩核/缩 tile | `C*(512/s)>N` 或 `C*tile0>N` | 避免为不足 cache line/满 tile 的数据保留过多核和 UB | R-001 | 保留 `tile0` | `DoOneDimFormerTilingAdvance` |
| 每核至少两 tile | 初算 `blockFormer==1` 时改为 2 | 减少启动核数并形成 ping/pong 两相 | R-001 | `blockFormer>1` 原样使用 | `DoOneDimFormerTilingAdvance` |
| AiCPU scalar fast path | `input1_elements_nums==1` | 避免构造通用 `Bcast` 索引对象 | R-002 | 非 scalar 才进入 `BcastCompute`；本入口不可达 | `GreaterEqualCpuKernel::GreaterEqualCompute` |
| AiCPU 并行阈值 | `N>=32*1024` | 大的一维 scalar 比较按 CPU 核分片 | R-002 | 小于阈值串行 `SpecialCompute(0,N)` | `GreaterEqualCpuKernel::NoBcastCompute` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen` | `int64_t`/64 | `sizeof=16, alignof=8, offsetof=0` | 元素 | `DoOneDimFormerTilingAdvance: dimLen=N` | `Init` 计算 `ubOuter/ubTail` | `N>0`；空域在 API 短路 |
| 同上 | `tileNum` | `int32_t`/32 | `offsetof=8` | 元素/tile | UB 公式后按小数据条件缩减 | `Init/Process/CalcBlockOffset` | 必须大于 0；无显式 int32 上界检查 |
| 同上 | `blockNum` | `int16_t`/16 | `offsetof=12` | AIV core | `ceil(ubOuter/blockFormer)` | 尾核判断与 `blockTail` 重算 | 无显式 int16 窄化检查 |
| 同上 | `scalarFlag` | `int16_t`/16 | `offsetof=14` | 输入位图 | 合轴长度为 1 的输入 i 设置 `1<<i` | `CopyIn` 选择 GM DataCopy 或 scalar load+Duplicate | x2 必置 bit1；N=1 时 x1 也置 bit0 |
| AiCPU context | N/A | N/A | 无 AscendC POD | N/A | launcher 传两个 Tensor 与一个 BOOL 输出 | `CpuKernelContext::Input/Output` | R-002 不读取 tiling key/POD |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 AIV one-dim scalar | dtype selector；key 202 | `GreaterEqualTiling::DoOpTiling` + `BroadcastBaseTiling::DoOneDimOpTilingAdvance` | `BroadcastOneDimTilingDataAdvance` | `greater_equal<202>`→`BroadcastSch<202,...>` | `GreaterEqualCompute<T>::OpDag`；`BroadcastOneDimAdvanceSch`；`Vec::Compare/Duplicate/Select/CopyOut` | OpDef `ascend950`、binary JSON、CMake `ascend950/arch35`、ascendc config |
| R-002 AiCPU scalar | `!IsAiCoreSupport(promoteType)`；`Y_ONE_ELEMENT` | `l0op::GreaterEqual` 添加 AiCPU task | Tensor shape/dtype/context，无 POD | `REGISTER_CPU_KERNEL("GreaterEqual",...)` | `Compute`→`GreaterEqualCompute`→`NoBcastCompute`→`SpecialCompute` | `greater_equal_aicpu_def.cpp` + module CMake |

## Kernel 实现

### 公共执行流程

两条 Route 共享 API 前处理与结果提交：连续化、按 950 规则生成 `promoteType`、物化 scalar、可选输入 Cast，核心产生 BOOL Tensor，再 Cast 到 `selfRef.dtype` 并 `ViewCopy` 回写。R-001 的 entry 参数顺序固定为 `x1,x2,y,workspace,tiling`；R-002 由 AiCPU context 取得两个输入和一个输出。核心均只实现比较，不直接覆盖 `selfRef`。

### Route R-001：AIV one-dim scalar

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/greater_equal/op_api/aclnn_ge_scalar.cpp` | `CheckPromoteType` | RegBase：`CombineCategoriesWithComplex`；BOOL→INT8；结果须在 RegBase support list | 得到合法 `promoteType` |
| L0 selector | `math/greater_equal/op_api/greater_equal.cpp` | `IsAiCoreSupport/GreaterEqual` | RegBase 且 dtype 属九种 AiCore set | `ADD_TO_LAUNCHER_LIST_AICORE(GreaterEqual,...)` |
| Tiling | `math/greater_equal/op_host/arch35/greater_equal_tiling_arch35.cpp` | `GreaterEqualTiling::DoOpTiling` | x1/x2 dtype 必须相同；按 dtype 实例化 DAG | 公共 Broadcast Tiling |
| ATVOSS | `opbase/.../broadcast_tiling.h` | `BroadcastBaseTiling::DoTiling` | 连续且合轴 rank=1；3 个 I/O；无 Var | `DoOneDimOpTilingAdvance`，schMode 202 |
| Router | `opbase/.../broadcast_sch.h` | `BroadcastSch<202>::Process` | 编译期 `schMode==202` | AIV-only `BroadcastOneDimAdvanceSch` |

#### Device 数据流

##### 函数级调用链

```text
GreaterEqual AiCore -> greater_equal<202>(x1, x2, y, workspace, tiling)
  -> BroadcastSch<202, GreaterEqualCompute<T>::OpDag>::Process
    -> BroadcastOneDimAdvanceSch::Init(x1, x2, y)
    -> BroadcastOneDimAdvanceSch::Process
      -> Run: CopyInBrc<T>(x1)
      -> Run: CopyInBrc<T>(x2 scalar)
      -> Vec::Compare<uint8_t,T,CMPMODE=4>
      -> Vec::Duplicate<uint8_t>(0)
      -> Vec::Duplicate<uint8_t>(1)
      -> Vec::Select<uint8_t,uint8_t,SELMODE=2>(mask,one,zero)
      -> Vec::CopyOut<uint8_t>(y)
  -> [executor ordering] Cast(y -> selfRef.dtype) -> ViewCopy(..., selfRef)
```

##### 源码等价伪码

```text
# 伪码；T=promoteType，s=sizeof(T)
base = blockIdx * blockFormer * tile
loops = lastCore ? blockTail : blockFormer
for i in [0, loops):
    len = (lastCore && i==loops-1) ? ubTail : tile
    ping = i & 1
    if x1_is_scalar: x1UB[ping] = Duplicate<T>(x1GM[0], len)  # 仅 N=1
    else: x1UB[ping] = DataCopyPad(x1GM + base + i*tile, len*s bytes)
    if i < 2: x2UB[ping] = Duplicate<T>(x2GM[0], len)
    maskUB = Compare<T, GE>(x1UB[ping], x2UB[ping], len)
    zeroUB = Duplicate<uint8_t>(0, len)
    oneUB = Duplicate<uint8_t>(1, len)
    yUB = Select<uint8_t>(maskUB, oneUB, zeroUB, len)
    DataCopyPad(yGM + base + i*tile, yUB, len bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1 GM | `(blockIdx*blockFormer*tile+i*tile)*s` 字节 | `len*s` 有效字节 | T/ND contiguous | Contiguous/Cast | CopyIn | executor 临时或 self contiguous | tail 精确长度 |
| x2 GM | 0 | `s` 字节 | T/scalar Tensor | ConvertToTensor | scalar load | executor，覆盖核心全程 | 每核读值并广播两相 |
| x1 MTE2 ping/pong | `bufId*tile*s` | 每槽 `tile*s` | T/local | DataCopyPad/Duplicate | Compare | 当前 parity | 两槽，512-byte tile 对齐 |
| x2 MTE2 ping/pong | `bufId*tile*s` | 每槽 `tile*s` | T/local | 前两次 Duplicate | Compare | 核内全循环 | 两槽持久复用 |
| Compare temp | `bufId*tile*s` | 统一槽容量 `tile*s`，有效数据 `len` bytes | uint8 mask/local | Compare | Select | 当前 tile | LEVEL_2 独立 temp 槽 |
| zero temp | `bufId*tile*s` | 统一槽容量 `tile*s` | uint8/local | Duplicate(0) | Select | 当前 tile | LEVEL_2 独立 temp 槽 |
| one temp | `bufId*tile*s` | 统一槽容量 `tile*s` | uint8/local | Duplicate(1) | Select | 当前 tile | LEVEL_2 独立 temp 槽 |
| y MTE3 ping/pong | `bufId*tile*s` | 每槽统一容量 `tile*s`，有效 `len` bytes | uint8/local | Select | CopyOut | 当前 parity | 两槽；输出物理有效量小于槽容量 |
| y GM | `blockIdx*blockFormer*tile+i*tile` 字节 | `len` bytes | BOOL/ND contiguous | CopyOut | output Cast | executor 临时 | 不提交无效 lane |

##### 分层内存布局

```mermaid
block-beta
  columns 9
  x1p["x1 ping · tile*s"]:1
  x1q["x1 pong · tile*s"]:1
  x2p["x2 scalar ping · tile*s"]:1
  x2q["x2 scalar pong · tile*s"]:1
  cmp["compare · tile*s"]:1
  z["zero · tile*s"]:1
  o["one · tile*s"]:1
  yp["y ping · tile*s"]:1
  yq["y pong · tile*s"]:1
```

##### 字节账与复用顺序

```text
MTE2 = 2 inputs * 2 phases = 4 slots
TEMP = Compare + DupZero + DupOne = 3 slots
MTE3 = 1 output * 2 phases = 2 slots
B = 9
UB_total = 9 * tileNum * sizeof(promoteType)
tileNum initial = floor((UB_bytes/9)/512)*512/sizeof(promoteType)
UB_total <= UB_bytes
dynamic_ub = N/A（AIV 静态 buffer-id LocalTensor）
workspace_total = executor 聚合；GreaterEqual 核心 tiling slice 声明 16*1024*1024 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1 ping/pong | CopyIn | Compare | 同 parity 的下下个 tile | `GetTensor(VECIN)`/`ReleaseTensor`，计算侧 Get/Release | Compare 消费前不覆盖 |
| x2 scalar ping/pong | i=0/1 的 scalar load+Duplicate | 所有同 parity tile 的 Compare | 核结束 | 两相分别初始化，后续不写 | scalar 在核心执行中不变 |
| compare/zero/one temp | 各 Vector producer | Select | 下一 tile | `GetTensor/ReleaseTensor(VECCALC)` | Select 取完三个输入后才释放 |
| y ping/pong | Select | CopyOut | 同 parity 的下下个 tile | VECCALC→VECOUT buffer mutex | CopyOut 完成前不覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPadExtParams{1,len*s,0,0,0}` 或 `GetValue(0)+Duplicate` | x1/x2 GM | Compare 前 Get | 两路 input slot | scalar 按 promote dtype 预转换；尾块只搬 `len` |
| 比较 | `AscendC::Compare(...,CMPMODE(4),len)` | x1/x2 local | Select | mask temp | 直接按 T 比较，无内部升/降精度 |
| BOOL 物化 | `Duplicate(0/1)` + `Select(...,SELMODE(2),len)` | compare mask | CopyOut | uint8 result | True/False 显式物化为 1/0；无 round |
| UB→GM | `DataCopyPad` | result local | output Cast | BOOL GM | 仅写 `len` bytes，尾 lane 不提交 |
| 核内同步 | `Mutex::Lock/Unlock<PIPE_MTE2/PIPE_V/PIPE_MTE3>` | 各 pipeline producer | 下一消费者/复用者 | buffer-id 子区 | parity 与生命周期着色共同防止提前覆盖 |
| 跨节点 | executor/stream 顺序 | GreaterEqual BOOL y | Cast→ViewCopy | 原地目标 | 核心读旧 `selfRef` 完毕后才回写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/greater_equal/op_host/arch35/greater_equal_tiling_arch35.cpp` | `GreaterEqualTiling::DoOpTiling` | 同 dtype guard、dtype→DAG、key producer |
| Kernel entry/router | `math/greater_equal/op_kernel/greater_equal_apt.cpp` | `greater_equal<202>` | BOOL 映射 int8，其余按 DTYPE_X1 进入 BroadcastSch |
| Device DAG | `math/greater_equal/op_kernel/arch35/greater_equal_dag.h` | `GreaterEqualCompute<T>::OpDag` | CopyInBrc→Compare→0/1 Select→CopyOut，LEVEL_2 |
| 公共 body | `opbase/.../broadcast_sch_one_dim_advance.h` | `Init/Process/CopyIn/Run/CopyOut` | 分核、地址、scalar 两相缓存、buffer mutex 与 tail |
| Vector primitive | `opbase/.../util/vec.h` | `Compare/Duplicate/Select/CopyInBrc/CopyOut` | 对应 AscendC API 与 count/字节参数 |

### Route R-002：AiCPU scalar fallback

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/greater_equal/op_api/aclnn_ge_scalar.cpp` | `CheckPromoteType` | 合法 RegBase promote dtype，但不必属于 AiCore set | 进入 L0 GreaterEqual |
| L0 selector | `math/greater_equal/op_api/greater_equal.cpp` | `GreaterEqual` | `IsAiCoreSupport(self)==false` | `ADD_TO_LAUNCHER_LIST_AICPU(GreaterEqual,...)` |
| AiCPU router | `math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp` | `GreaterEqualCpuKernel::Compute` | dtype switch 命中 DOUBLE/INT16/UINT16/UINT32 等 | `GreaterEqualCompute<T>` |
| shape selector | 同上 | `GreaterEqualCompute` | input1 elements=1 | `NoBcastCompute`，type=`Y_ONE_ELEMENT` |
| CPU scheduler | 同上 | `NoBcastCompute` | `N>=32*1024` | ParallelFor；否则串行 |

#### Device 数据流

##### 函数级调用链

```text
GreaterEqual AiCPU task
  -> GreaterEqualCpuKernel::Compute(ctx)
    -> GreaterEqualParamCheck(ctx)
    -> GreaterEqualCompute<T>(ctx)
      -> NoBcastCompute<T>(ctx)
        -> SpecialCompute<T>(Y_ONE_ELEMENT,start,end,x1,x2,y)
          -> y[i] = x1[i] >= *x2
  -> [executor ordering] Cast(y -> selfRef.dtype) -> ViewCopy(..., selfRef)
```

##### 源码等价伪码

```text
# 伪码
assert inputs==2 && outputs==1
assert dtype(x1)==dtype(x2)
T = dtype(x1)
N = numel(y)
type = Y_ONE_ELEMENT  # numel(x2)==1
if N >= 32768:
    workers = max(4, GetCPUNum(ctx)-reserved)
    grain = N / min(N, workers)
    ParallelFor(N, grain, lambda(start,end):
        for i in [start,end): y[i] = x1[i] >= x2[0])
else:
    for i in [0,N): y[i] = x1[i] >= x2[0]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1 memory | `i*sizeof(T)` | `N*sizeof(T)` | T/contiguous | Contiguous/input Cast | CPU loop | executor | shard 只读 |
| x2 memory | 0 | `sizeof(T)` | T/scalar | ConvertToTensor | 每个 CPU iteration | executor | 所有 shard 共享只读 |
| y memory | `i` | `N` bytes | BOOL/contiguous | CPU assignment | output Cast | executor | shard 区间互不重叠 |
| CPU shard | `[start,end)` | `grain` 附近，末 shard 收尾 | index range | ParallelFor | SpecialCompute | task runtime | 无跨 shard reduction/atomic |

##### 分层内存布局

N/A：该 Route 是 AiCPU 实现，不分配 AIV UB、TQue/TBuf、dynamic UB 或 NDDMA buffer；实际介质是 CpuKernelContext 暴露的输入/输出地址和 ParallelFor shard。

##### 字节账与复用顺序

```text
UB_total = N/A（AiCPU）
dynamic_ub = N/A（AiCPU）
workspace_total = executor/AiCPU task space 聚合，固定 GreaterEqual AiCPU body 未公开独立字节公式
output bytes = N * sizeof(bool)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| y 的 shard `[start,end)` | 对应 ParallelFor worker | 后续 output Cast | Cast 完成后由 executor 回收 | `ParallelFor` 返回前所有 shard 完成 | shard 区间不重叠；错误返回阻断 Cast |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar 比较 | C++ `operator>=` on T | x1/x2 memory | bool store | 单个元素 | 按 promote dtype T 精确比较；无二次 cast |
| CPU 分片 | `CpuKernelUtils::ParallelFor` | workers | 函数返回 | y 的互斥区间 | `[start,end)` 自带尾区间，无 padding |
| 小域 | 串行 `SpecialCompute(0,N)` | 当前 task | output Cast | 全部 y | 无线程同步 |
| 跨节点 | executor/stream task ordering | AiCPU y | Cast/ViewCopy | 原地目标 | AiCPU 完成后才提交回写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host selector | `math/greater_equal/op_api/greater_equal.cpp` | `IsAiCoreSupport/GreaterEqual` | 非 AiCore dtype 添加 AiCPU task |
| Kernel registration | `math/greater_equal/op_kernel_aicpu/greater_equal_aicpu_def.cpp` | `ops::GreaterEqual` | AiCPU dtype 注册与 BOOL 输出 |
| Device body | `math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp` | `Compute/GreaterEqualCompute/NoBcastCompute/SpecialCompute` | dtype switch、scalar fast path、阈值、shard 与比较循环 |

## 算子特定优化

### Trick T-001：scalar 仅初始化 ping/pong 两相

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：`schMode=202`，`scalarFlag` 标记 x2，当前核有超过两次 tile 使用
- Golden 来源：[`pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h:75-92`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h#L75-L92)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
  __aicore__ inline void Process()
  {
      uint64_t offset = 0;
      uint64_t i = 0;
      isFirstScalarTime = true;
      for (; i < blockLoopNum - 1; i++) {
          if (isFirstScalarTime && i > 1) {
              isFirstScalarTime = false;
          }
          Run<0>(offset, tilingData->tileNum, i & 1);
```

#### 机制与设计原因

`CopyIn` 对 runtime scalar 只在 `isFirstScalarTime` 为真时执行 `GetValue(0)` 与 `Duplicate`。i=0、1 分别填满 ping/pong x2 子区，i≥2 复用同 parity 的值；x1 与 y 仍正常轮转。该机制减少后续 tile 对同一 scalar 的 GM 读取与广播指令，并依赖 scalar 在整个核心生命周期不变。

#### 迁移边界

- 适用：一路输入为真正全域 scalar、采用两相 buffer-id 的逐元素算子。
- 前提：两个 parity 都在首次复用前初始化；scalar Tensor 生命周期覆盖 Kernel。
- 失效：普通 Tensor/broadcast pattern 必须按 tile 搬入；若流水深度不是 2，必须初始化所有相位。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只初始化 ping 而复用 pong 会读取未定义 local 数据；固定实现通过 i=0、1 两次初始化避免该错误。

### Trick T-002：Compare mask 后直接选择 byte 0/1

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：所有 AIV dtype 与有效 tile 长度 `len`
- Golden 来源：[`math/greater_equal/op_kernel/arch35/greater_equal_dag.h:29-37`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel/arch35/greater_equal_dag.h#L29-L37)

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

比较结果直接作为 `Select` mask，在 UB 内从 byte 常量 1/0 选出最终 BOOL 表示，再直接连接 CopyOut；没有额外 Cast 或算术归一化节点。`CMPMODE=4` 与 `SELMODE=2` 是编译期常量，`count=len` 来自当前 tile，有效尾块不会把无效 lane 写回 GM。

#### 迁移边界

- 适用：输出要求未打包 byte BOOL 的比较类算子。
- 前提：目标 BOOL ABI 为 uint8 表示；Compare/Select mode 与语义一致；输出容量按 byte 计。
- 失效：bit-packed mask 输出、三值逻辑或需要保留原值的 merge 语义不能复用该 DAG。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-003：AiCPU 单元素输入绕过通用广播索引

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`numel(input1)==1`，这是 Scalar 入口物化后的固定条件
- Golden 来源：[`math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp:175-192`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp#L175-L192)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
uint32_t GreaterEqualCpuKernel::GreaterEqualCompute(const CpuKernelContext& ctx)
{
    Tensor* input0_tensor = ctx.Input(0);
    auto input0_shape = input0_tensor->GetTensorShape()->GetDimSizes();
    int64_t input0_elements_nums = input0_tensor->NumElements();

    Tensor* input1_tensor = ctx.Input(1);
    auto input1_shape = input1_tensor->GetTensorShape()->GetDimSizes();
    int64_t input1_elements_nums = input1_tensor->NumElements();

    const bool is_need_call_bcast =
        !((input0_shape == input1_shape) || (input0_elements_nums == 1) || (input1_elements_nums == 1));
```

#### 机制与设计原因

scalar x2 令 `is_need_call_bcast=false`，随后 `NoBcastCompute` 将分支类型设为 `Y_ONE_ELEMENT`，内层直接读取 `*input2`，避免每个元素调用通用 `Bcast::GetBroadcastYIndex`。当 N 达到 32768 时同一直接寻址循环再按 CPU shard 并行。

#### 迁移边界

- 适用：AiCPU 二元 elementwise 的任一路恰有一个元素。
- 前提：输出 shape 已在 L0 正确广播推导；scalar 地址在所有 shard 完成前有效。
- 失效：两个多元素、shape 不同的 Tensor 必须构造合法 `Bcast`；不能只凭某一轴长度为 1 使用该路径。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把“某轴为 1”误当作“整个输入只有一个元素”会绕过必要的广播索引；源码判断使用的是 `NumElements()==1`。

## 模板与实现对比

| 对比维度 | R-001 AIV one-dim scalar | R-002 AiCPU scalar | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | promote dtype 在 RegBase AiCore set | 合法但不在该 set | L0 `IsAiCoreSupport` 先 AiCore、否则 AiCPU |
| 模板与 dtype | `GreaterEqualCompute<T>`；Host BF16 用同字节数 half 资源 DAG，Device 用实际 DTYPE_X1 | `GreaterEqualCpuKernel<T>`，当前 fallback 为 DOUBLE/INT16/UINT16/UINT32 | binary 与 CPU switch 的实际支持集 |
| 循环与分核 | AIV block→tile；最后核/尾 tile | CPU `[start,end)` shard 或串行 | 平台执行介质不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 TQue/TBuf/dynamic UB；9 个静态 buffer-id slot + Vector register | 无 AIV 资源；CpuKernelContext 地址与 worker shard | R-001 是 AIV-only，R-002 是 AiCPU |
| 内存总占用与复用 | `9*tile*s`；scalar 双相驻留 | 输出 N bytes；task workspace 固定公式不可见 | AIV 用 UB 上界，AiCPU 直接访问 task tensors |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad、Duplicate、Compare、Select、DataCopyPad | C++ loop、ParallelFor | scalar 一维不走 NDDMA/SIMT/通信 |
| 同步、精度与 tail | buffer mutex；按 promote dtype 比较；`ubTail` | ParallelFor join；按 T 比较；end-exclusive shard | 均不做内部精度提升 |
| fallback/失败边界 | Tiling/launcher 失败无二次核心 fallback | 参数/ParallelFor 失败终止 | route 在 L0 建图时已固定 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但固定 Commit 的 `CMakeLists.txt` 指向不存在的 `arch32`，Host/Tiling/Kernel 源码闭包缺失；不生成卡 | 公开文档写支持；固定 Commit 同样缺少 `arch32` 实体且 binary config 只有 Ascend 950，无法闭合；不生成卡 | 公开文档、OpDef、binary config、Kernel 与 CMake 一致闭合 |
| Host/Launcher | 非 RegBase promote 分支可见，但其 AiCore 实体不可见 | 同左 | RegBase promote；AiCore/AiCPU 两条核心路线 |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | arch35 AIV schMode 202 + AiCPU fallback |

## 限制与失败边界

- `selfRef`、`other` 或 `workspaceSize` 为空时失败；dtype、类型推导、BOOL 结果回转或 rank>8 失败返回参数错误。
- 空 Tensor 成功短路且 `workspaceSize=0`；R-001 的 `dimLen` 因此始终大于 0。
- `tileNum:int32_t`、`blockNum:int16_t` 的 producer 没有显式窄化上界检查；极端 shape/平台值的溢出行为不得推测。
- R-001 的 Host BF16 资源计算实例化 `GreaterEqualCompute<half>`，Device body 按实际 `DTYPE_X1` 实例化；二者都为 2-byte 类型，不能据此推断 BF16 被数值转换成 FP16。
- `Contiguous`、`ConvertToTensor`、Cast 与 ViewCopy 的内部 Kernel/POD 属直接依赖边界；卡中闭合其可见选择、顺序、owner、结果 dtype 与失败传播，不把它们误称为 `GreaterEqual` 核心 body。
- A2/A3 的公开支持声明与固定源码闭包不一致：`arch32` 实体缺失，故未创建无后缀卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/greater_equal/docs/aclnnGeScalar&aclnnInplaceGeScalar.md` | `aclnnInplaceGeScalar*` | 950 支持、原地语义、参数与错误 |
| 入口声明 | `math/greater_equal/op_api/aclnn_ge_scalar.h` | 同上 | 两段式原型 |
| 入口实现 | `math/greater_equal/op_api/aclnn_ge_scalar.cpp` | `CheckPromoteType/aclnnGeScalarCommon/aclnnInplaceGeScalar*` | 校验、950 promote、空域、节点顺序、workspace 与执行 |
| L0 分发 | `math/greater_equal/op_api/greater_equal.cpp` | `IsAiCoreSupport/GreaterEqual` | AiCore/AiCPU selector 与输出分配 |
| Cast 依赖 | `math/cast/op_api/cast.cpp` | `Cast/IsAiCoreSupport/IsAiCpuSupport` | 同 dtype消除、AiCore/AiCPU Cast 边界 |
| ViewCopy 依赖 | `conversion/view_copy/op_api/view_copy.cpp` | `ViewCopy/IsAiCoreSupport` | RegBase 回写选择、shape/stride 参数物化 |
| Host/Tiling | `math/greater_equal/op_host/arch35/greater_equal_tiling_arch35.cpp` | `GreaterEqualTiling::DoOpTiling` | dtype guard、DAG 与 key producer |
| OpDef | `math/greater_equal/op_host/greater_equal_def.cpp` | `ops::GreaterEqual` | Ascend 950 dtype/ND 与 apt 接线 |
| binary config | `math/greater_equal/op_host/config/ascend950/greater_equal_binary.json` | `GreaterEqual_*` | 950 AiCore binary dtype 规格 |
| Kernel entry | `math/greater_equal/op_kernel/greater_equal_apt.cpp` | `greater_equal<schMode>` | entry 参数、BOOL 特化与 router |
| Kernel DAG | `math/greater_equal/op_kernel/arch35/greater_equal_dag.h` | `GreaterEqualCompute<T>` | Compare/Select 数据流与 LEVEL_2 |
| Kernel key | `math/greater_equal/op_kernel/arch35/greater_equal_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 模板 key |
| AiCPU body | `math/greater_equal/op_kernel_aicpu/greater_equal_aicpu.cpp` | `GreaterEqualCpuKernel` | fallback dtype、scalar 路径、ParallelFor 与循环 |
| AiCPU registration | `math/greater_equal/op_kernel_aicpu/greater_equal_aicpu_def.cpp` | `ops::GreaterEqual` | AiCPU 输入/输出 dtype 注册 |
| 构建接线 | `math/greater_equal/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | 950 arch35 与 A2/A3 arch32 缺口 |
| 构建配置 | `scripts/kernel/binary_config/ascendc_config.json` | `GreaterEqual` | 950 AscendC 编译接线 |
| POD | `pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | `BroadcastOneDimTilingDataAdvance` | 16-byte POD 与字段顺序 |
| 公共 Tiling | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling::DoTiling/DoOneDimOpTilingAdvance` | 合轴分流、tile/core、workspace 与 POD producer |
| 合轴依赖 | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse/DoDimensionCollapse` | scalar broadcast 折叠与 stride |
| 公共 router | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch<202>` | AIV-only one-dim scheduler |
| 公共 body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch` | UB、循环、地址、同步、scalar 缓存与 tail |
| Vector primitive | `pkg_inc/op_common/atvoss/util/vec.h` | `Compare/Duplicate/Select/CopyInBrc/CopyOut` | AscendC API 参数 |
| 内存策略 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch::GetBufferNum/GetBufferIds` | LEVEL_2 buffer 数和 id |
| 内存策略 | `pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo::GetBufferNumLevel2` | MTE2/MTE3 ping-pong 与 temp 公式 |
| 同步 | `pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor/ReleaseTensor` | PIPE mutex 映射 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 复用 16-byte one-dim advance POD | 字段顺序、位宽、元素单位与 scalarFlag bit 位不能改 | `BroadcastOneDimTilingDataAdvance` |
| 调用顺序 | Contiguous→scalar Tensor→可选 Cast→GreaterEqual→Cast→ViewCopy | 原地目标不能在核心读取旧值前覆盖；任一失败阻断后续 | `aclnnGeScalarCommon` |
| workspace/内存 | R-001 `9*tile*s` UB；核心声明 16 MiB workspace | 512-byte floor、9 槽、executor owner/lifetime | `GetOneDimUBFormerAdvance`; `GetBufferNumLevel2` |
| 同步 | AIV buffer mutex、scalar 两相缓存；AiCPU ParallelFor join | 两个 parity 均须初始化；CPU shard 不得重叠 | `BroadcastOneDimAdvanceSch::Process`; `NoBcastCompute` |
| 精度与尾块 | promote dtype 比较；AIV 选择 byte 0/1；AiCPU bool store | 不得按 self 原 dtype比较混合 scalar；tail 仅提交有效元素 | `CheckPromoteType`; `GreaterEqualCompute` |
| 失败边界 | 空域 Host 短路；dtype selector固定 AiCore/AiCPU；A2/A3 不发布 | 不得补造 arch32、Tensor 或非原地路线 | `CheckParams`; `IsAiCoreSupport`; `CMakeLists.txt` |