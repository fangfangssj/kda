---
type: Operator
title: "InplaceBitwiseOrScalar 950 ACLNN 算子知识"
description: "说明 aclnnInplaceBitwiseOrScalar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/docs/aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_api/aclnn_bitwise_or_scalar.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_api/aclnn_bitwise_or_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_api/bitwise_or.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/op_api/logical_or.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_host/arch35/bitwise_or_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/op_host/arch35/logical_or_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_host/bitwise_or_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/op_host/logical_or_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_host/config/ascend950/bitwise_or_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/op_host/config/ascend950/logical_or_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_kernel/bitwise_or_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_kernel/arch35/bitwise_or_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_kernel/arch35/bitwise_or_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/op_kernel/logical_or_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/op_kernel/arch35/logical_or_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/op_kernel/arch35/logical_or_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/logical_or/CMakeLists.txt
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
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/include/nnopbase/opdev/data_type_utils.h
    role: implementation
created_at: "2026-08-25T10:20:38Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceBitwiseOrScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceBitwiseOrScalarGetWorkspaceSize` + `aclnnInplaceBitwiseOrScalar`，无版本后缀，公开 ACLNN 两段式入口 |
| 计算语义 | 对 `selfRef` 的每个元素执行 `selfRef[i] = selfRef[i] \| other`；BOOL 时是逻辑或；原地语义由最后的 `ViewCopy(..., selfRef)` 提交，不是核心 Kernel 直接覆盖输入 GM |
| 输入输出 | `selfRef` 为 ND、0～8 维、可非连续；950 支持 BOOL/INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64；`other` 为同一支持集中的 Host scalar |
| 执行类型 | R-001、R-002 的算子拥有核心节点均为 `AIV_VECTOR`；`BroadcastSch<202,...>` 显式声明 `KERNEL_TYPE_AIV_ONLY`；`Contiguous/ViewCopy/executor` 是框架边界 |
| Kernel 路线 | R-001 BOOL：`LogicalOr`；R-002 整数：`BitwiseOr`；空 Tensor 在 Host 成功短路且不启动核心 Kernel |
| 核心资源 | GM 输入/临时输出、AIV UB 静态 buffer-id 池、寄存器/Vector `Or`、核心节点声明的 16 MiB workspace；无 TQue/TBuf、无 dynamic UB、无跨核心 Atomic |
| 来源闭包 | `source_closed`：公开入口、950 分发、两个 Host/Tiling、one-dim POD、router/body、ATVOSS 直接依赖与 950 构建接线闭合；框架 `Contiguous/ViewCopy/executor` 按允许边界标注 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `N = selfRef` 的逻辑元素数，非空域满足 `0 <= i < N`：

```text
整数：selfRef_after[i] = selfRef_before[i] bitwise_or cast(other, dtype(selfRef))
BOOL：selfRef_after[i] = selfRef_before[i] logical_or bool(other)
```

950 的 `GetPromoteDType` 使用 `CombineCategories`：`selfRef` 非 BOOL 时直接选择 `selfRef.dtype`，所以 `Cast(contiguousSelf, promoteType)` 和结果到 `selfRef.dtype` 的 `Cast` 都是同 dtype 返回原 Tensor；`selfRef` 为 BOOL 时，只有能得到 BOOL promote dtype 的 BOOL scalar 可通过原地输出检查，进入 `LogicalOr`。固定依赖中的 `CanCast` 明确拒绝非 BOOL 到 BOOL，因此不能把 BOOL 输入与整数 scalar 误列为可执行 Route。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND；维数不超过 8；允许非连续 | BOOL、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64 | 输入/输出共用公开对象；空 Tensor 成功短路；非空时先读取旧值，最后由 `ViewCopy` 写回 |
| `other` | Host scalar；转换后形成标量 Tensor | 同 `selfRef` 支持集 | 必须与 `selfRef` 可推导；整数 Route 转换为 `selfRef.dtype`，BOOL Route 为 BOOL |
| `workspaceSize` | 标量输出 | `uint64_t` | executor 聚合整条流程所需 workspace；核心 `LogicalOr/BitwiseOr` Tiling 各声明 16 MiB，框架边界可能追加需求 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并交给调用者，第二段绑定 `workspace` 与 `stream` 执行 |

### 入口与版本

公开头文件只声明无版本后缀的 `aclnnInplaceBitwiseOrScalar*`，实现复用 `GetWorkspaceSizeCommon(selfRef, other, selfRef, ...)`，但其输出别名、最终 `ViewCopy` 目标和错误语义均保持原地逻辑算子身份，不能与非原地 `aclnnBitwiseOrScalar` 合并。公开文档明确支持 Ascend 950PR/950DT；`math/bitwise_or/CMakeLists.txt` 仅将 `BitwiseOr` 接到 `ascend950/arch35`，BOOL 依赖 `LogicalOr` 也具有 `ascend950/arch35` 构建接线，因此本卡只覆盖 950。

## Host/Launcher 执行流程

### 参数校验与前处理

真实顺序如下：

1. `CREATE_EXECUTOR()`；失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
2. `CheckNotNull(selfRef, other, selfRef)`；空指针返回 `ACLNN_ERR_PARAM_NULLPTR`。
3. 根据 `IsRegBase()` 选择包含九种 950 dtype 的支持表；`GetPromoteDType` 计算 promote dtype；`CanCast(promoteType, selfRef.dtype)` 失败则返回 `ACLNN_ERR_PARAM_INVALID`。
4. `CheckShape(selfRef, selfRef)` 的 shape 相等恒成立，仍执行最大 8 维检查。
5. `selfRef->IsEmpty()` 时写 `workspaceSize=0`、释放 executor 所有权并成功返回，不创建核心计算节点。
6. `l0op::Contiguous(selfRef)` 得到核心输入；这是允许的框架边界。
7. `executor->ConvertToTensor(other, promoteType)` 将 Host scalar 物化为标量 Tensor。
8. promote dtype 为 BOOL 时调用 `l0op::LogicalOr`；否则同 dtype `Cast` 返回 `contiguousSelf`，再调用 `l0op::BitwiseOr`。
9. 核心结果到 `selfRef.dtype` 的 `Cast` 在两个可达 Route 中也是同 dtype返回；随后 `l0op::ViewCopy(castOut, selfRef)` 提交原地结果。
10. `uniqueExecutor->GetWorkspaceSize()` 产生公开 workspace 大小；第二段 `CommonOpExecutorRun` 在给定 stream 上按 executor 顺序执行，任一建图节点返回空指针都会阻断后续节点。

### 参数、Workspace 与生命周期

`selfRef` 的旧值必须活到核心 `LogicalOr/BitwiseOr` 消费完毕；核心输出是 executor 分配的临时 Tensor，必须活到 `ViewCopy` 完成。对于非连续 `selfRef`，连续化临时量与回写任务的具体切片由框架拥有，固定算子源码不可见；其 owner 是 executor，生命周期覆盖第二段执行。两个核心 Host Tiling 都通过 ATVOSS `BroadcastBaseTiling::DoTiling` 将 `workspaces[0]` 设为 `16 * 1024 * 1024` 字节，尽管核心 entry 的 `workspace` 参数未在 body 中读取；公开 `workspaceSize` 是 executor 对核心与可能的框架任务的聚合值，不能无条件简化成 16 MiB。

### 启动图与失败传播

```text
aclnnInplaceBitwiseOrScalarGetWorkspaceSize
  -> [framework] Contiguous(selfRef) + ConvertToTensor(other)
  -> R-001: LogicalOrAiCore -> logical_or<202>(x1, x2, tmp, workspace, tiling)
     或
     R-002: BitwiseOrAiCore -> bitwise_or<202>(x1, x2, tmp, workspace, tiling)
  -> [framework] ViewCopy(tmp, selfRef)
aclnnInplaceBitwiseOrScalar
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

| 节点/边 | entry 与参数 | blockDim/POD | 交接、ordering、owner/lifetime | 失败传播 |
|---|---|---|---|---|
| 连续化/标量物化 | 框架 `Contiguous`、`ConvertToTensor` | 固定仓不可见 | executor 拥有临时量；同一 executor/stream 顺序先于核心 | 任一返回空指针即第一段失败 |
| R-001 核心 | `logical_or<202>(x1,x2,y,workspace,tiling)` | `blockNum`；`BroadcastOneDimTilingDataAdvance` | 读连续 `selfRef` 与 scalar Tensor，写连续临时 `y` | launcher 失败返回空指针，阻断回写 |
| R-002 核心 | `bitwise_or<202>(x1,x2,y,workspace,tiling)` | 同上 | 同上 | 同上 |
| 核心→回写 | `y`，extent=`N*sizeof(T)`，ND 连续 | N/A | `ViewCopy` 在核心之后消费；executor 持有到回写完成 | `ViewCopy` 返回空指针则第一段失败 |
| 回写→`selfRef` | 框架 ViewCopy，目标可能非连续 | 固定仓不可见 | 同 stream/executor 顺序保证先读旧输入再写目标；完成后 `selfRef` 可见 | 第二段返回执行错误 |

## Route 与参数设计

### 分核与资源计算

连续输入与 scalar 的广播模式在各原始轴一致，`DimensionCollapse` 将其折叠为长度 `N` 的一维输出；`InputSize + OutputSize = 3 <= 4` 且 `VarSize=0`，所以 `DoOneDimOpTilingAdvance` 选择 `schMode=202`。

记 `s=sizeof(T)`，`B=OpDag::GetBufferNum<true,true>()`，`C=GetCoreNumAiv()`，`U=UB bytes`：

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

`tileNum` 写入 `int32_t`，`blockNum` 写入 `int16_t`；源码未提供显式窄化上界检查。Device 重新得到 `ubTail=N-(ubOuter-1)*tile`、`blockTail=ubOuter-(blockNum-1)*blockFormer`。最后一核处理 `blockTail` 个 tile 且最后 tile 有效长度为 `ubTail`，其他核处理 `blockFormer` 个整 tile。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 BOOL 逻辑或 | `GET_TPL_TILING_KEY(202)`；`LogicalOrCompute<uint8_t>::OpDag` | 非空；950；`promoteType==BOOL`，结合原地 `CanCast` 等价于 `selfRef=BOOL && other=BOOL` | 非 BOOL promote 进入 R-002；BOOL 输入与非 BOOL scalar 在参数检查失败 |
| 2 | R-002 整数按位或 | `GET_TPL_TILING_KEY(202)`；`BitwiseOrCompute<T>::OpDag` | 非空；950；`selfRef` 为八种整数 dtype；`CombineCategories` 令 `promoteType=selfRef.dtype` | dtype 不支持或推导失败则参数错误 |

```text
if selfRef/other is null -> ACLNN_ERR_PARAM_NULLPTR
if dtype unsupported or promote undefined or !CanCast(promote, selfRef.dtype) -> ACLNN_ERR_PARAM_INVALID
if rank(selfRef) > 8 -> ACLNN_ERR_PARAM_INVALID
if N == 0 -> success, workspaceSize=0, no core kernel
if promoteType == BOOL -> R-001 LogicalOr<BOOL>, schMode=202
else -> R-002 BitwiseOr<promoteType>, schMode=202
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 BOOL 逻辑或 | `202` | 950；非空；BOOL/BOOL；连续化后 scalar broadcast | 单个算子拥有核心 `logical_or<202>`；前后可有框架任务 | `BroadcastOneDimAdvanceSch<LogicalOrCompute<uint8_t>::OpDag>`；`Vec::Or<uint8_t>` | 静态 AIV buffer-id 池，输入/输出 ping-pong，scalar 两份缓存；无 Atomic | 最后一核/最后 tile 用 `blockTail/ubTail`；失败无 fallback |
| R-002 整数按位或 | `202` | 950；非空；`selfRef` 为 INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64 | 单个算子拥有核心 `bitwise_or<202>`；前后可有框架任务 | `BroadcastOneDimAdvanceSch<BitwiseOrCompute<T>::OpDag>`；`Vec::Or<T>` | 与 R-001 同构，`s` 随 dtype 为 1/2/4/8 | 同上；不支持 dtype 失败，不走 AICPU |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AIV-only | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` | 调度只使用 Vector/GM↔UB 路径，无 Cube/AIC 节点 | 两条 Route | 无其它执行族 fallback | `broadcast_sch.h::BroadcastSch::Process` |
| one-dim advance | 合轴后 rank=1、输入连续、I/O 数≤4、无 Var | 使用紧凑 16-byte POD 和静态 buffer-id 双缓冲 | 两条 Route | 通用 broadcast modes；本 ACLNN scalar 链不可达 | `BroadcastBaseTiling::DoTiling` |
| UB 512-byte floor | `floor((U/B)/512)>0` | 每个 buffer 的 tile 字节按 512 对齐并支持 ping/pong | 两条 Route | Tiling `GRAPH_FAILED` | `GetOneDimUBFormerAdvance` |
| 小数据缩核 | `C*(512/s)>N` 或 `C*tile0>N` | 避免为不足一条 cache line 的域保留全核/full tile | 两条 Route | 使用原 `tile0` | `DoOneDimFormerTilingAdvance` |
| 至少两 tile/核 | 初算 `blockFormer==1` 时改为 `2` | 减少核启动并使 ping/pong 两相都有稳定缓存 | 两条 Route | `blockFormer>1` 保持原值 | `DoOneDimFormerTilingAdvance` |
| 原地 BOOL 回转 | `CanCast(promoteType, BOOL)` | 固定依赖禁止非 BOOL 结果转换到 BOOL | R-001 | 参数错误，不进入 Cast/AICPU | `CheckDtypeValid`; `CanCast` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen` | `int64_t`/64 | `sizeof=16, alignof=8, offsetof=0` | 元素 | `DoOneDimFormerTilingAdvance: dimLen=N` | `Init` 计算 `ubOuter/ubTail` | `N>0`；空域已 Host 短路 |
| 同上 | `tileNum` | `int32_t`/32 | `offsetof=8` | 元素/tile | 先由 UB 公式产生，再按小数据条件缩减 | `Init/Process/CalcBlockOffset` | 必须大于 0；源码无 `int32_t` 上界检查 |
| 同上 | `blockNum` | `int16_t`/16 | `offsetof=12` | AIV core | `ceil(ubOuter/blockFormer)` | 判断尾核并计算 `blockTail` | 源码无 `int16_t` 上界检查 |
| 同上 | `scalarFlag` | `int16_t`/16 | `offsetof=14` | 输入位图 | 输入 i 的合轴长度为 1 时置 `1<<i`；本算子 x2 置 bit1，N=1 时 x1 也置 bit0 | `CopyIn` 判断 GM 连续搬入或读 scalar+Duplicate | x2 必为 scalar；值通常 2，N=1 时为 3 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 BOOL 逻辑或 | `promoteType==BOOL`; key 202 | `logical_or_tiling_arch35.cpp::LogicalOrTiling::DoOpTiling` + ATVOSS `DoOneDimOpTilingAdvance` | `BroadcastOneDimTilingDataAdvance` | `logical_or_apt.cpp::logical_or<202>` | `LogicalOrCompute<uint8_t>::OpDag`; `BroadcastOneDimAdvanceSch`; `Vec::Or` | `math/logical_or/CMakeLists.txt` 的 `ascend950/arch35` |
| R-002 整数按位或 | `promoteType!=BOOL`; key 202 | `bitwise_or_tiling.cpp::BitwiseOrTiling::DoOpTiling` + 同一 ATVOSS producer | 同上 | `bitwise_or_apt.cpp::bitwise_or<202>` | `BitwiseOrCompute<T>::OpDag`; `BroadcastOneDimAdvanceSch`; `Vec::Or` | `math/bitwise_or/CMakeLists.txt` 的 `ascend950/arch35` |

## Kernel 实现

### 公共执行流程

两个核心 entry 都把 `schMode=202` 和 dtype DAG 交给 `BroadcastSch`；router 解析 `BroadcastOneDimTilingDataAdvance*`、初始化 SoC 状态、构造 `BroadcastOneDimAdvanceSch`，再按核循环执行 `CopyInBrc(x1)`、`CopyInBrc(x2)`、`Or`、`CopyOut`。one-dim advance 不创建 `TPipe/TQue/TBuf`，而是用 `LocalTensor(TPosition, bufferId*blockLen, blockLen)` 和 `GetTensor/ReleaseTensor` 管理静态 UB 子区。核心 entry 保留 `workspace` ABI 参数但 body 不读取它。

### Route R-001：BOOL 逻辑或

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/bitwise_or/op_api/aclnn_bitwise_or_scalar.cpp` | `GetPromoteDType/CheckDtypeValid` | RegBase；BOOL 与 BOOL promote 为 BOOL；非 BOOL→BOOL 被 `CanCast` 拒绝 | `promoteType==BOOL` |
| Launcher | `math/bitwise_or/op_api/aclnn_bitwise_or_scalar.cpp` | `GetWorkspaceSizeCommon` | `if (promoteType == DT_BOOL)` | `l0op::LogicalOr` |
| L0 selector | `math/logical_or/op_api/logical_or.cpp` | `IsAiCoreSupport/LogicalOr` | 950 RegBase 且 BOOL 在 AICORE list | `LogicalOrAiCore` |
| Tiling | `math/logical_or/op_host/arch35/logical_or_tiling_arch35.cpp` | `LogicalOrTiling::DoOpTiling` | x1/x2 dtype 相同且 BOOL | `LogicalOrCompute<uint8_t>` |
| ATVOSS | `opbase/.../broadcast_tiling.h` | `BroadcastBaseTiling::DoTiling` | 合轴 rank=1、连续、I/O≤4、VarSize=0 | schMode 202 |

#### Device 数据流

##### 函数级调用链

```text
LogicalOrAiCore -> logical_or<202>(x1, x2, y, workspace, tiling)
  -> BroadcastSch<202, LogicalOrCompute<uint8_t>::OpDag>::Process
    -> BroadcastOneDimAdvanceSch::Init(x1, x2, y)
    -> BroadcastOneDimAdvanceSch::Process
      -> Run: CopyInBrc(x1) -> CopyInBrc(x2) -> Vec::Or<uint8_t> -> CopyOut<uint8_t>
  -> [executor ordering] ViewCopy(y, selfRef)
```

##### 源码等价伪码

```text
# 伪码；s=1，scalarFlag bit1 标记 x2
tile = tiling.tileNum
base = blockIdx * blockFormer * tile
loops = (blockIdx == blockNum-1) ? blockTail : blockFormer
lastLen = (blockIdx == blockNum-1) ? ubTail : tile
for i in [0, loops):
    len = (i == loops-1) ? lastLen : tile
    ping = i & 1
    x1UB[ping] = DataCopyPad(x1GM + base + i*tile, len)
    if i < 2:
        scalar = x2GM[0]
        x2UB[ping] = Duplicate(scalar, len)
    yUB = Or(x1UB[ping], x2UB[ping], len)
    DataCopyPad(yGM + base + i*tile, yUB, len)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1 GM | `(blockIdx*blockFormer*tile+i*tile)` 元素 | `len` 有效元素 | uint8/BOOL，ND contiguous | framework Contiguous | `CopyIn` | executor 临时或原存储，活到核心完成 | tail 仅搬 `len` |
| x2 GM | 0 | 1 元素 | uint8/BOOL scalar Tensor | ConvertToTensor | scalar load | executor，活到核心完成 | 每核各读；核内仅填 ping/pong 两份 |
| MTE2 buffer-id 区 | `bufId*tile` | 每区 `tile` 字节 | uint8 local | DataCopyPad/Duplicate | `Vec::Or` | 当前 ping/pong 相位 | `GetTensor/ReleaseTensor` |
| calc buffer-id 区 | `bufId*tile` | `tile` 字节 | uint8 local | `Vec::Or` | CopyOut | 当前 tile | DAG LEVEL_2 生命周期复用 |
| MTE3 buffer-id 区 | `bufId*tile` | 每区 `tile` 字节 | uint8 local | Or 结果 | DataCopyPad out | 当前 ping/pong 相位 | release 后可覆盖 |
| y GM | 同 x1 全局元素偏移 | `len` 有效元素 | BOOL ND contiguous | CopyOut | ViewCopy | executor 临时，活到回写完成 | 不写 padding lane |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inPP["MTE2 input regions · ping/pong · 2*M2*tile bytes"]:2
  calc["LEVEL_2 temp · Tmp*tile bytes"]:1
  outPP["MTE3 output regions · ping/pong · 2*M3*tile bytes"]:2
```

##### 字节账与复用顺序

```text
B = LogicalOrCompute<uint8_t>::OpDag::GetBufferNum<true,true>()
UB_total = B * tileNum * 1 byte
tileNum initial = floor((UB_bytes/B)/512)*512
UB_total <= UB_bytes by construction
dynamic_ub = N/A（AIV 静态 buffer-id LocalTensor）
workspace_total = executor 聚合；LogicalOr 核心 tiling slice 声明 16*1024*1024 bytes，entry 不读取
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1 ping/pong | `CopyIn` | `Or` | 同 parity 的下下次 tile | `GetTensor(VECIN)`→`ReleaseTensor`，计算侧再 Get/Release | 每相位在消费结束前不覆盖 |
| x2 scalar ping/pong | `Duplicate`（i=0/1） | 每个同 parity tile 的 `Or` | 核结束 | 两相分别初始化；之后不再写 | scalar 对所有元素不变 |
| calc/output region | `Or` | `CopyOut` | 下一可用相位 | `ReleaseTensor(VECCALC/VECOUT)` | CopyOut 完成后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` 或 scalar load+`Duplicate` | GM | `Or` 前 buffer Get | x1/x2 local region | BOOL 以 uint8 存储；实际 `len` 控制尾块 |
| 计算 | `AscendC::Or(dst,src1,src2,count)` | 两输入 local | CopyOut | calc region | 无 cast/舍入；`count=len` 屏蔽尾部无效 lane |
| UB→GM | `DataCopyPad` | Or 结果 | ViewCopy | 连续临时 y | 只提交 `len` 个元素 |
| 跨节点 | executor/stream 顺序 | core y | ViewCopy | 原地目标 `selfRef` | 先完整读旧输入，后回写，避免别名提前覆盖 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/logical_or/op_host/arch35/logical_or_tiling_arch35.cpp` | `LogicalOrTiling::DoOpTiling` | BOOL 实例化 `LogicalOrCompute<uint8_t>` 并取 schMode |
| Kernel entry/router | `math/logical_or/op_kernel/logical_or_apt.cpp` | `logical_or<202>` | BOOL 映射到 uint8 DAG 与 BroadcastSch |
| Device body/helper | `opbase/.../broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch::Process/Run/CopyIn/CopyOut` | 分核循环、scalar 双相缓存、地址与 tail |
| Vector primitive | `opbase/.../util/vec.h` | `Vec::Or<uint8_t>` | 调用 `AscendC::Or(..., count)` |

### Route R-002：整数按位或

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/bitwise_or/op_api/aclnn_bitwise_or_scalar.cpp` | `CombineCategories/GetPromoteDType` | RegBase 且 `selfRef.dtype != BOOL` | `promoteType=selfRef.dtype` |
| Launcher | 同上 | `GetWorkspaceSizeCommon` | `promoteType != BOOL`；两个 Cast 同 dtype返回输入 | `l0op::BitwiseOr` |
| L0 selector | `math/bitwise_or/op_api/bitwise_or.cpp` | `IsAiCoreSupport/BitwiseOr` | RegBase 的八种整数全部在 AICORE list | `BitwiseOrAiCore`，无 AICPU fallback 可达 |
| Tiling | `math/bitwise_or/op_host/arch35/bitwise_or_tiling.cpp` | `BitwiseOrTiling::DoOpTiling` | x1/x2/y dtype 相同且为八种整数之一 | 对应 `BitwiseOrCompute<T>` |
| ATVOSS | `opbase/.../broadcast_tiling.h` | `BroadcastBaseTiling::DoTiling` | 合轴 rank=1、连续、I/O≤4、VarSize=0 | schMode 202 |

#### Device 数据流

##### 函数级调用链

```text
BitwiseOrAiCore -> bitwise_or<202>(x1, x2, y, workspace, tiling)
  -> BroadcastSch<202, BitwiseOrCompute<T>::OpDag>::Process
    -> BroadcastOneDimAdvanceSch::Init(x1, x2, y)
    -> BroadcastOneDimAdvanceSch::Process
      -> Run: CopyInBrc<T>(x1) -> CopyInBrc<T>(x2) -> Vec::Or<T> -> CopyOut<T>
  -> [executor ordering] ViewCopy(y, selfRef)
```

##### 源码等价伪码

```text
# 伪码；T=selfRef.dtype，s=sizeof(T)
baseElem = blockIdx * blockFormer * tile
loops = lastCore ? blockTail : blockFormer
for i in [0, loops):
    len = (lastCore && i==loops-1) ? ubTail : tile
    ping = i & 1
    x1UB[ping] = DataCopyPad<T>(x1GM + baseElem + i*tile, len*s bytes)
    if i < 2:
        scalar = x2GM[0]
        x2UB[ping] = Duplicate<T>(scalar, len)
    yUB = Or<T>(x1UB[ping], x2UB[ping], len)
    DataCopyPad<T>(yGM + baseElem + i*tile, yUB, len*s bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1 GM | `(blockIdx*blockFormer*tile+i*tile)*s` 字节 | `len*s` 字节 | T/ND contiguous | Contiguous | CopyIn | executor 输入临时 | tail 精确字节数 |
| x2 GM | 0 | `s` 字节 | T/scalar Tensor | ConvertToTensor | scalar load | executor | 每核读取一次值、核内双相复用 |
| input buffer-id 区 | `bufId*tile*s` | 每区 `tile*s` | T/local | CopyIn/Duplicate | Or | 当前相位 | 512-byte tile 对齐；ping/pong |
| calc/output buffer-id 区 | `bufId*tile*s` | 每区 `tile*s` | T/local | Or | CopyOut | 当前 tile | DAG LEVEL_2 生命周期着色复用 |
| y GM | 与 x1 对应元素偏移 | `len*s` | T/ND contiguous | CopyOut | ViewCopy | executor 临时 | 仅有效 tail |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inPP["MTE2 input regions · ping/pong · 2*M2*tile*s"]:2
  calc["LEVEL_2 temp · Tmp*tile*s"]:1
  outPP["MTE3 output regions · ping/pong · 2*M3*tile*s"]:2
```

##### 字节账与复用顺序

```text
B = BitwiseOrCompute<T>::OpDag::GetBufferNum<true,true>()
UB_total = B * tileNum * sizeof(T)
tileNum initial = floor((UB_bytes/B)/512)*512/sizeof(T)
UB_total <= UB_bytes by construction
dynamic_ub = N/A（AIV 静态 buffer-id LocalTensor）
workspace_total = executor 聚合；BitwiseOr 核心 tiling slice 声明 16*1024*1024 bytes，entry 不读取
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1 ping/pong | CopyIn | Or | 同 parity 的下下次 tile | buffer-id Get/Release | 计算消费完成后覆盖 |
| x2 scalar ping/pong | i=0/1 的 Duplicate | 所有同 parity tile | 核结束 | scalar 不变且两个 parity 均已初始化 | 不得在后续迭代改 scalar |
| Or 结果区 | Or | CopyOut | 生命周期着色后的下一使用者 | VECCALC/VECOUT Release | CopyOut 完成后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | `DataCopyPadExtParams{1,len*s,0,0,0}` | x1 GM | Or | input buffer | 无 padding 写入；精确 `len*s` |
| scalar broadcast | `GetValue(0)` + `Duplicate(dst,scalar,len)` | x2 GM | Or | 两个 scalar buffer 相位 | 按 T 位模式，无转换 Kernel |
| 计算 | `AscendC::Or(...,len)` | input buffers | CopyOut | result buffer | 整数逐位精确；无溢出、cast、round |
| CopyOut | `DataCopyPad` | result buffer | ViewCopy | y GM | 最后核最后 tile 使用 `ubTail` |
| 跨节点 | executor/stream 顺序 | y GM | ViewCopy | `selfRef` | 核心完成后才覆盖公开输入输出对象 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/bitwise_or/op_host/arch35/bitwise_or_tiling.cpp` | `BitwiseOrTiling::DoOpTiling` | 八种 dtype 分支、同 dtype guard、key producer |
| Kernel entry/router | `math/bitwise_or/op_kernel/bitwise_or_apt.cpp` | `bitwise_or<202>` | `DTYPE_X1` 绑定 DAG 并进入 BroadcastSch |
| Device body/helper | `math/bitwise_or/op_kernel/arch35/bitwise_or_dag.h` | `BitwiseOrCompute<T>::OpDag` | CopyInBrc→Or→CopyOut、LEVEL_2 内存策略 |
| 公共 scheduler | `opbase/.../broadcast_sch_one_dim_advance.h` | `Init/Process/Run` | 地址、循环、ping/pong、scalar 缓存和尾块 |

## 算子特定优化

### Trick T-001：RegBase 原地 dtype 固定消除 Cast Kernel

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：950 `IsRegBase()` 且 `selfRef.dtype != BOOL`
- Golden 来源：[`math/bitwise_or/op_api/aclnn_bitwise_or_scalar.cpp:55-60`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_or/op_api/aclnn_bitwise_or_scalar.cpp#L55-L60)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
static DataType GetPromoteDType(const aclTensor *self, const aclScalar *other) {
  auto promoteType = DataType::DT_UNDEFINED;
  if (IsRegBase()) {
    promoteType = CombineCategories(self->GetDataType(), other->GetDataType());
    return promoteType;
  }
```

#### 机制与设计原因

`CombineCategories` 对非 BOOL 的 higher 直接返回 `selfRef.dtype`；随后两次 `l0op::Cast` 都命中 `self->GetDataType()==dstDtype` 的无任务返回。这样整数原地路线只物化 scalar Tensor并启动一次 `BitwiseOr` 核心，不产生前后 Cast 中间张量与 Kernel。机制改变的是 launch 数与中间物化，不代表已测性能收益。

#### 迁移边界

- 适用：输出必须回写到与输入同 dtype 的原地整数 elementwise 算子。
- 前提：950 RegBase；scalar 能按 API 规则转换成 `selfRef.dtype`；核心算子支持该 dtype。
- 失效：BOOL 走 R-001；非 950 的 promote 规则不同，不能沿用；不支持或不能回转的 dtype 在参数检查失败。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把非 950 的“INT64 scalar 按数值缩到 INT32”规则套到 950，会错误插入 Cast 路线；固定 950 源码明确采用 `CombineCategories`。

### Trick T-002：scalar 只初始化 ping/pong 两相

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：schMode 202，`scalarFlag` 标记 x2，当前核 tile 循环超过两次
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

`CopyIn` 对 runtime scalar 只在 `isFirstScalarTime` 为真时执行 GM `GetValue(0)` 与 `Duplicate`。i=0、1 分别填充 ping/pong 子区，i≥2 时沿用同 parity 的 scalar 区；x1 和输出仍正常轮转。该机制减少每个后续 tile 的 scalar GM 读取和 UB Duplicate，同时要求 scalar 在整个核心生命周期不变。

#### 迁移边界

- 适用：逐元素二元算子的一路输入是真正全域 scalar，且采用两相 buffer-id 调度。
- 前提：两个 parity 在复用前均初始化；scalar Tensor 生命周期覆盖 Kernel；没有 tile 内更新 scalar。
- 失效：非 scalar 输入必须每 tile CopyIn；超过两相的流水深度必须初始化所有相位；不同 broadcast pattern 不能直接套用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若仅初始化 ping 而未初始化 pong，则奇数 tile 会读取未定义 local 数据；源码通过 i=0、1 两次初始化避免该问题。

## 模板与实现对比

| 对比维度 | R-001 BOOL 逻辑或 | R-002 整数按位或 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | BOOL/BOOL，先由 API 判 `promoteType==BOOL` | 非 BOOL `selfRef`，后续 else | BOOL 语义要求逻辑或，整数要求逐位或 |
| 模板与 dtype | `LogicalOrCompute<uint8_t>` | `BitwiseOrCompute<T>`，T 为八种整数 | BOOL 物理存储按 uint8；整数保留位宽 |
| 循环与分核 | schMode 202，同一 one-dim advance | 同左 | scalar broadcast pattern 与 shape 相同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/dynamic UB；uint8 LocalTensor+Vector register | 无 Queue/TBuf/dynamic UB；T LocalTensor+Vector register | 静态 buffer-id 调度 |
| 内存总占用与复用 | `B*tile`；scalar 双相缓存 | `B*tile*sizeof(T)`；scalar 双相缓存 | Host 以 MaxDtypeBytes 反推 tile |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad、Duplicate、Or、DataCopyPad | 同 API，T 不同 | 一维连续场景不走 NDDMA/SIMT/通信 |
| 同步、精度与 tail | Get/Release；无 cast；`ubTail` | Get/Release；逐位精确；`ubTail` | 同一 scheduler |
| fallback/失败边界 | 非 BOOL scalar 不能回转 BOOL，参数失败 | dtype 不支持/tiling 失败，无 AICPU fallback | 公开 950 可达集合闭合 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但固定 `ops-math` Commit 缺少 A2 Host/Tiling/Kernel 构建闭包，本卡不覆盖 | 公开文档写支持，但固定 Commit 缺少 A3 Host/Tiling/Kernel 构建闭包，本卡不覆盖 | 公开文档、OpDef、binary config 与 CMake 一致支持 |
| Host/Launcher | 非 RegBase promote 分支存在，但所达实现不在固定范围闭合 | 同左 | RegBase `CombineCategories`；BOOL/整数两路线 |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | arch35 schMode 202、AIV-only、ATVOSS 静态 UB buffer-id |

## 限制与失败边界

- `selfRef`/`other` 为空返回 `ACLNN_ERR_PARAM_NULLPTR`；dtype、推导、回转或 rank>8 失败返回 `ACLNN_ERR_PARAM_INVALID`。
- 空 Tensor 不启动核心 Kernel且公开 `workspaceSize=0`；非空一维 POD 的 `dimLen` 必须大于 0。
- 950 原地 BOOL 与非 BOOL scalar 不能作为 BOOL 输出闭合，不能误写成 Cast+BitwiseOr Route。
- `tileNum:int32_t`、`blockNum:int16_t` 的窄化没有固定源码内显式上界检查；极端平台/shape 的溢出行为不得推测。
- `Contiguous`、`ConvertToTensor`、`ViewCopy` 与 executor 的内部 Kernel、额外 workspace 切片和 event 细节是固定源码允许的框架边界；卡中只陈述算子可见的顺序、owner 和失败传播。
- A2/A3 只有公开支持声明而没有此固定源码集合中的 source closure，因此未生成无后缀卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/bitwise_or/docs/aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar.md` | `aclnnInplaceBitwiseOrScalar*` | 950 支持、参数、原地语义和错误 |
| 入口声明 | `math/bitwise_or/op_api/aclnn_bitwise_or_scalar.h` | 同上 | 函数原型、950 dtype 与计算图 |
| 入口实现 | `math/bitwise_or/op_api/aclnn_bitwise_or_scalar.cpp` | `GetWorkspaceSizeCommon` | 校验、promote、空域、分支、回写和 executor |
| L0 分发 | `math/bitwise_or/op_api/bitwise_or.cpp` | `IsAiCoreSupport/BitwiseOr` | 950 八整数到 AiCore |
| L0 分发 | `math/logical_or/op_api/logical_or.cpp` | `IsAiCoreSupport/LogicalOr` | 950 BOOL 到 AiCore |
| Host/Tiling | `math/bitwise_or/op_host/arch35/bitwise_or_tiling.cpp` | `BitwiseOrTiling::DoOpTiling` | dtype guard、DAG 实例与 key |
| Host/Tiling | `math/logical_or/op_host/arch35/logical_or_tiling_arch35.cpp` | `LogicalOrTiling::DoOpTiling` | BOOL DAG 与 key |
| OpDef | `math/bitwise_or/op_host/bitwise_or_def.cpp` | `ops::BitwiseOr` | 950 dtype/ND 注册与 `bitwise_or_apt` 接线 |
| OpDef | `math/logical_or/op_host/logical_or_def.cpp` | `ops::LogicalOr` | 950 BOOL 注册与 `logical_or_apt` 接线 |
| binary config | `math/bitwise_or/op_host/config/ascend950/bitwise_or_binary.json` | `BitwiseOr_*` | 八种 950 整数 binary 规格 |
| binary config | `math/logical_or/op_host/config/ascend950/logical_or_binary.json` | `LogicalOr_BOOL` | BOOL 950 binary 规格 |
| Kernel entry | `math/bitwise_or/op_kernel/bitwise_or_apt.cpp` | `bitwise_or<schMode>` | BitwiseOr router |
| Kernel DAG | `math/bitwise_or/op_kernel/arch35/bitwise_or_dag.h` | `BitwiseOrCompute<T>` | CopyInBrc→Or→CopyOut |
| Kernel key | `math/bitwise_or/op_kernel/arch35/bitwise_or_struct.h` | `ASCENDC_TPL_ARGS_DECL(BitwiseOr,...)` | schMode 可编译 key 集 |
| Kernel entry | `math/logical_or/op_kernel/logical_or_apt.cpp` | `logical_or<schMode>` | BOOL router |
| Kernel DAG | `math/logical_or/op_kernel/arch35/logical_or_dag.h` | `LogicalOrCompute<uint8_t>` | BOOL CopyInBrc→Or→CopyOut |
| Kernel key | `math/logical_or/op_kernel/arch35/logical_or_struct.h` | `ASCENDC_TPL_ARGS_DECL(LogicalOr,...)` | schMode 可编译 key 集 |
| 构建接线 | `math/bitwise_or/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | BitwiseOr 950/arch35 |
| 构建接线 | `math/logical_or/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | LogicalOr 950/arch35 |
| POD | `pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | `BroadcastOneDimTilingDataAdvance` | 16-byte POD 字段与 key 集 |
| 公共 Tiling | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling::DoTiling` | 合轴分流、tile/core、workspace 和 POD producer |
| 公共 Tiling | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse` | scalar broadcast 合轴、stride 与失败边界 |
| 公共 router | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch<202>` | AIV-only 与 one-dim scheduler |
| 公共 body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim_advance.h` | `BroadcastOneDimAdvanceSch` | UB、循环、搬运、同步和 tail |
| Vector primitive | `pkg_inc/op_common/atvoss/util/vec.h` | `CopyInBrc/Or/CopyOut` | DataCopyPad 与 AscendC::Or 参数 |
| 内存策略 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch::GetBufferNum/GetBufferIds` | LEVEL_2 buffer 数与 id 生成 |
| 内存策略 | `pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo::GetBufferNumLevel2` | MTE2/MTE3 ping-pong 与 temp 字节乘数 |
| dtype 规则 | `include/nnopbase/opdev/data_type_utils.h` | `CanCast/PromoteType` | 原地 BOOL 回转失败与 promote lookup |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 复用 16-byte one-dim advance POD | 字段顺序/位宽/元素单位、scalarFlag bit 位不能改 | `BroadcastOneDimTilingDataAdvance` |
| 调用顺序 | Contiguous→scalar Tensor→核心→ViewCopy | 原地目标不能在核心读完旧值前覆盖 | `GetWorkspaceSizeCommon` |
| workspace/内存 | `B*tile*s` 静态 UB；核心 16 MiB workspace 声明 | 512-byte floor、buffer-id 乘数、executor owner | `GetOneDimUBFormerAdvance`; `DoTiling` |
| 同步 | ping/pong Get/Release 与 executor stream ordering | scalar 两相都必须先初始化；结果消费前不能覆盖 | `BroadcastOneDimAdvanceSch::Process/Run` |
| 精度与尾块 | dtype 原样逐位 Or；最后核最后 tile 用 `ubTail` | BOOL 用 uint8 Or；不得插入数值算术或错误 cast | `Vec::Or`; `BroadcastOneDimAdvanceSch::Init` |
| 失败边界 | 空域 Host 短路；dtype/rank/Tiling/launcher 失败关闭 | 不得把 AICPU、A2/A3 或 Tensor 变体补入 | `CheckParams`; `IsAiCoreSupport`; CMake |
