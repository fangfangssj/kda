---
type: Operator
title: "BitwiseXorTensor 950 ACLNN 算子知识"
description: "说明 aclnnBitwiseXorTensor 在 950 上的公开语义、Host 图、BitwiseXor/NotEqual broadcast Route 及可达 Cast 节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/docs/aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_api/aclnn_bitwise_xor_tensor.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_api/aclnn_bitwise_xor_tensor.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_api/bitwise_xor.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_host/bitwise_xor_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_host/arch35/bitwise_xor_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_kernel/arch35/bitwise_xor_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_kernel/bitwise_xor_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/op_host/config/ascend950/bitwise_xor_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/bitwise_xor/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_api/not_equal.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_host/not_equal_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_host/arch35/not_equal_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_kernel/arch35/not_equal_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_kernel/not_equal_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/op_host/config/ascend950/not_equal_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/not_equal/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_api/cast.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/cast_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/arch35/cast_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/cast_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel/arch35/cast_impl.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_kernel_aicpu/cast_aicpu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/op_host/config/ascend950/cast_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/cast/CMakeLists.txt
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/include/nnopbase/opdev/data_type_utils.h
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
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_nddma.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_ub_broadcast.h
    role: implementation
created_at: "2026-08-25T10:23:59Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# BitwiseXorTensor 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnBitwiseXorTensorGetWorkspaceSize` + `aclnnBitwiseXorTensor`，无版本后缀。 |
| 语义隔离 | 仅非原地 Tensor-Tensor 逻辑算子；不包含 Scalar 入口，也不包含同文件中的 `aclnnInplaceBitwiseXorTensor`。 |
| 核心语义 | 整数按提升后的定宽整数逐位 XOR；两输入均为 BOOL 时以 `NotEqual` 实现逻辑 XOR。两输入先 broadcast。 |
| 950 输入 | BOOL、INT8/16/32/64、UINT8/16/32/64；ND，rank≤8，可非连续。 |
| 核心执行族 | BOOL 核心是 NotEqual AIV；整数核心是 BitwiseXor AIV；broadcast 可走单轴线性、NDDMA 或 UB Broadcast。无 Cube/AIC/MIX_CV。 |
| 可选节点 | 输入提升 Cast 仅在整数分支出现；结果 Cast 由 out dtype 决定，可为 identity、AIV（INT64→DOUBLE 含 SIMT guard）或 AICPU。 |
| 来源闭包 | `source_closed`：ACLNN→参数校验/类型提升→Contiguous→NotEqual 或 Cast+BitwiseXor→结果 Cast→ViewCopy；两核心的 Tiling/POD/router/body/build 与 Cast AIV/AICPU 直接依赖均闭合。 |
| 验证状态 | 仅完成固定提交源码闭包与静态一致性检查；按任务约束未运行 validator、编译、测试或 review。 |

## 接口与语义

### 计算语义

令 `S` 是 `self` 与 `other` 的 broadcast shape，`p=PromoteType(dtype(self), dtype(other))`。对每个 `i∈S`：

```text
if p == BOOL:
    core[i] = (self[broadcast(i)] != other[broadcast(i)])
else:
    a = cast(self[broadcast(i)], p)
    b = cast(other[broadcast(i)], p)
    core[i] = bitwise_xor(a, b)           # p 位宽内逐位 XOR
out[i] = cast(core[i], dtype(out))
```

`PromoteType` 对相同 dtype 直接返回自身。950 公开输入中的不同 dtype 仅在 lookup 表给出非 `UNDEFINED` 时成立：BOOL 与受支持的有符号整数/UINT8 提升为对方整数，较窄有符号整数向较宽者提升，INT8+UINT8 提升为 INT16；UINT16/UINT32/UINT64 与不同 dtype 的表项为 `UNDEFINED`。这不是按 C++ 通常算术转换规则猜测。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束与语义 |
|---|---|---|---|
| `self` | ND，rank≤8，可非连续 | BOOL、INT8/16/32/64、UINT8/16/32/64 | 必填；与 `other` 可 broadcast。 |
| `other` | 同上 | 同上 | 必填；不同 dtype 必须能由固定 `PromoteType` 表提升。 |
| `out` | ND，rank≤8，可非连续 | BOOL、INT8/16/32/64、UINT8/16/32/64、FLOAT/FLOAT16/DOUBLE/BFLOAT16、COMPLEX64/128 | shape 必须严格等于 broadcast shape；`CanCast(promoteType,out.dtype)` 必须为真。故整数结果不能写 BOOL，虽 BOOL 出现在总输出类型表中。 |
| `workspaceSize` | Host 标量 | `uint64_t*` | 第一段返回完整 Executor workspace 字节数。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段移交完整 launch 图。 |
| `workspace/workspaceSize/stream` | Device 区域与 ACL stream | 指针/整数/stream | 第二段传给 `CommonOpExecutorRun`。 |

无用户属性。空 Tensor 在 Host 提前成功返回，不创建 Device 核心节点。

### 入口与版本

固定提交仅公开无版本后缀的 `aclnnBitwiseXorTensorGetWorkspaceSize/aclnnBitwiseXorTensor`，所以本卡把它视为 950 的实际最新公开入口。文档产品表、`IsRegBase()` 的 950 dtype 表、`BitwiseXor::AICore().AddConfig("ascend950")`、ascend950 binary JSON 与 `SUPPORT_COMPUTE_UNIT "ascend950"` 共同确认平台接线。

同实现文件中的 `aclnnInplaceBitwiseXorTensor*` 复用 `GetWorkspaceSizeCommon`，并将 `selfRef` 同时作为输入与输出，但原地别名语义是独立逻辑算子，本卡不合并；Scalar 变体也不在本入口调用链内。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CREATE_EXECUTOR` 创建 Executor，失败返回内部错误。
2. `CheckNotNull` 检查三 Tensor；`CheckDtypeValid` 依次验证 950 输入 dtype、`PromoteType!=UNDEFINED`、`CanCast(promoteType,out.dtype)`；`CheckShape` 检查三者 rank≤8、输入可 broadcast、out shape 精确匹配。
3. 任一输入为空则写 `workspaceSize=0`，移交空执行器并返回成功。
4. `l0op::Contiguous` 分别连续化两输入。其具体 Kernel/POD 是 ACLNN 框架边界，但返回空指针会立即失败。
5. `promoteType==BOOL` 时调用 `l0op::NotEqual`；950 BOOL 命中其 AICore 表，分配 BOOL 临时输出并加入 NotEqual launcher。
6. 否则两输入分别调用 `l0op::Cast` 转为 `promoteType`；同 dtype时返回原 Tensor。再调用 `l0op::BitwiseXor`，950 八种整数均命中 AICore 表，分配 broadcast 临时输出并加入 BitwiseXor launcher。
7. 核心结果 `Cast` 到 out dtype；identity 不建节点，REGBASE pair 表命中则建 AIV Cast，其余只要两端均在 AICPU 表就建 AICPU Cast，否则失败。最后 `ViewCopy` 写入可能非连续的 out。
8. 第一段聚合 workspace 并 `ReleaseTo(executor)`；第二段只调用 `CommonOpExecutorRun`。

### 参数、Workspace 与生命周期

调用者持有输入、out、第二段 workspace 与 stream。Executor 持有连续化 Tensor、输入 Cast 临时 Tensor、核心临时输出、结果 Cast 临时 Tensor及各 launcher；第一段移交后存活到第二段执行结束。

公共 `BroadcastBaseTiling::DoTiling` 为 NotEqual/BitwiseXor 核心写入 16 MiB workspace 声明，但两个 kernel entry 都不读取 `workspace`。Cast AIV 同样声明 16 MiB 且 body 不寻址该参数；Cast AICPU 由 task space 管理。完整 `uniqueExecutor->GetWorkspaceSize()` 还可能包含 Contiguous/ViewCopy 物化区，固定仓不公开框架如何合并 workspace slice，故不能把核心声明当作 API 唯一需求。

### 启动图与失败传播

```text
aclnnBitwiseXorTensorGetWorkspaceSize
  -> Contiguous(self) -> Contiguous(other)
  -> if promoteType == BOOL:
       NotEqual[BOOL broadcast AIV]
     else:
       Cast(self -> promoteType)?
       Cast(other -> promoteType)?
       BitwiseXor[promoteType broadcast AIV]
  -> Cast(coreOut -> out.dtype)? [AIV/SIMT or AICPU]
  -> ViewCopy(castOut,out)

aclnnBitwiseXorTensor
  -> CommonOpExecutorRun(workspace,workspaceSize,executor,stream)
```

`?` 表示 identity Cast 不物化。所有 `CHECK_RET`/launcher add 失败都阻断后继图构建；第二段在同一 stream 按 Executor 拓扑提交，运行错误由 `CommonOpExecutorRun` 返回。核心间无本算子自定义 workspace edge。

## Route 与参数设计

### 分核与资源计算

NotEqual 与 BitwiseXor 都实例化公共 `BroadcastBaseTiling<OpDag>`。两输入右对齐至最多 8 维，合并 broadcast 标志相同的相邻轴；broadcast 输入 stride 写 0。记合轴输出维度为 `D[0..L-1]`、AIV 核数为 `C`、每核 UB 字节为 `U`、DAG 最大/最小 dtype 位宽为 `bmax/bmin`、对应调度编译期 buffer 数为 `B`。

单轴 Advance（mode 202）按 512 B 对齐计算 `tileNum`，再令 `ubOuter=ceil(N/tileNum)`、`blockFormer=ceil(ubOuter/C)`；若 `blockFormer==1` 强制改 2，最后 `blockNum=ceil(ubOuter/blockFormer)`。多轴先计算：

```text
alignFactor = 256*8/bmin
maxElem = floor(((U-extraBytes)*8/(B*bmax))/alignFactor)*alignFactor
ubSplitAxis = 从内向外首个使维度累乘超过 maxElem 的轴
ubFormer = floor(maxElem/product(D[ubSplitAxis+1:]))
ubOuter = ceil(D[ubSplitAxis]/ubFormer)
ubTail = D[ubSplitAxis]-(ubOuter-1)*ubFormer
fused = product(D[:ubSplitAxis])*ubOuter
blockFormer = ceil(fused/C)
blockNum = ceil(fused/blockFormer)
blockTail = fused-(blockNum-1)*blockFormer
```

BitwiseXor DAG 三个 Tensor dtype 相同，`bmax=bmin=8/16/32/64`；NotEqual BOOL specialization 输入按 int8_t、输出 uint8_t，另有 Compare mask和两个常量/Select中间量，buffer 数由 `OpDag::GetBufferNum<cache,ubBrc>()` 编译期产生，固定仓未输出实例化后的整数，故公式保留该精确 producer 而不猜数值。

### Route 选择

| Route | 入口选择 | Device family/selector | 精确条件 | 下一跳或失败 |
|---|---|---|---|---|
| R-001 Empty | Host | 无 Device | `self.IsEmpty() \|\| other.IsEmpty()` | workspaceSize=0，成功。 |
| R-002 BOOL core | Host→NotEqual | AIV；内部 mode 202/1/2/101–104/109 | `PromoteType==BOOL`，即两输入均 BOOL | NotEqual失败则整个 API失败。 |
| R-003 Integer core | Host→输入 Cast?→BitwiseXor | AIV；内部 mode同上 | `PromoteType` 为八种 950 整数 | 输入 Cast或BitwiseXor失败则失败。 |
| R-004 Cast identity | Cast selector | 无 Device | `src.dtype==dst.dtype` | 直接返回输入 Tensor。 |
| R-005 Cast AIV/SIMT | Cast selector | `KERNEL_TYPE_AIV_ONLY`；pair policy；INT64→DOUBLE 触发 SIMT local-memory guard | `src!=dst` 且 pair 在 `REGBASE_AICORE_DTYPE_SUPPORT_LIST` | AICore launcher失败则失败。 |
| R-006 Cast AICPU | Cast selector | AICPU `CastCpuKernel` | pair未命中AIV，但两端都在 AICPU dtype表；本 API 可达的复杂输出及部分 unsigned/double pair在此 | pair不受AICPU支持则 Cast 返回空。 |

R-002/R-003 内部 broadcast selector 完全相同：合轴 `L==1` 且 DAG 参数总数≤4、无 Var 时 mode202；多轴若 `IsUbBroadcast` 真则按 `runningRank=L-ubSplitAxis` 取 101/102/103/104，超过4取109；否则 `runningRank>5` 取 mode2，余者 mode1。`IsUbBroadcast` 在大非尾轴广播或 8/16-bit 尾轴满足 32 B 对齐等条件下偏向 UB Broadcast；无 broadcast 一定走 NDDMA选择。

### 模板/实现执行矩阵

| Route | key/selector | 节点拓扑 | 实际模板/body | 内存/同步 | tail/fallback |
|---|---|---|---|---|---|
| R-001 | Host empty guard | 无节点 | Executor空图 | 无Device内存 | 无fallback。 |
| R-002 | BOOL + broadcast mode | `NotEqual` | `BroadcastSch<mode,NotEqualCompute<int8_t>::OpDag>`：CopyInBrc×2→Compare NE→Select 1/0→CopyOut BOOL | AIV；mode1/2可用NDDMA；mode101+先UB broadcast | 真实 count 传入 DAG，尾 lane屏蔽/有效长度搬出。 |
| R-003 | integer + broadcast mode | 0/1/2个输入Cast→`BitwiseXor` | `BroadcastSch<mode,BitwiseXorCompute<T>::OpDag>`：CopyInBrc×2→MicroAPI Xor→CopyOut | AIV；NDDMA/UB路径同公共 scheduler | `UpdateMask`保护最后寄存器批。 |
| R-004 | `src==dst` | 别名返回 | `l0op::Cast` no-op | 无额外buffer或同步 | N/A。 |
| R-005 | REGBASE pair membership | 单 Cast AIV | `cast<policy id>`→Direct/Through/Two/Micro系列；INT64→DOUBLE为Direct且预留32KiB SIMT local memory | GM→双缓冲UB→寄存器/Vector或SIMT→GM | 最后一核/最后UB循环使用四个tail字段。 |
| R-006 | AICPU dtype membership | 单 Cast AICPU task | `CastCpuKernel::Compute`→Validate→`ParallelFor`→`CastTask<S,D>` | CPU直接读写Tensor内存；框架任务同步 | 元素数取输入/输出较小者；线程分块无Device tail mask。 |

### 隐藏性能 Guard

| Guard | 条件 | 原因/影响 | 不满足时 |
|---|---|---|---|
| 类型提升表 | `PromoteType!=UNDEFINED` | 决定BOOL与整数图，也决定输入Cast数量；UINT16/32/64异型组合被拒绝 | 161002。 |
| 输出可转换 | `CanCast(promote,out)` | 禁止整数→BOOL；允许整数/BOOL→浮点、complex | 161002。 |
| 950 XOR AICore | 八种整数全在 `ASCEND950_AICORE_DTYPE_SUPPORT_LIST` | R-003核心不会回退AICPU | launcher失败。 |
| 单轴 fast path | 合轴rank1、参数≤4、无Var | 线性tile避免通用轴索引 | 进入多轴selector。 |
| UB/NDDMA选择 | 大非尾broadcast或8/16-bit尾轴32B对齐优先UB；其余NDDMA | 匹配broadcast搬运成本和NDDMA 5D能力 | mode1/2或101+互转。 |
| NDDMA运行rank | `runningRank<=5` | `MultiCopyParams` 当前实例的最大 rank 模板实参为 5 | 大于5走mode2软件外循环。 |
| Cast SIMT | policy为Direct且INT64→DOUBLE | Tiling从UB扣32KiB并设置local memory | 其它pair走普通AIV policy或AICPU。 |
| XOR寄存器tail | `loop=ceil(count/(VRegBytes/sizeof(T)))` | 最轮 `UpdateMask<T>`避免无效lane写出 | N/A。 |

### POD/ABI

`BroadcastOneDimTilingDataAdvance` 的定义可确定为自然布局16字节、align 8；公共 `BroadcastBaseTilingData<OpDag>` 含模板长度数组，固定仓未导出具体实例化的 `sizeof/offsetof`，不猜测。Cast POD为十二个连续 `int64_t` 字段，按定义顺序自然布局96字节、align 8。

| 结构 | 字段/布局 | Host producer | Device consumer | 单位/边界 |
|---|---|---|---|---|
| `BroadcastOneDimTilingDataAdvance` | `dimLen@0:int64`、`tileNum@8:int32`、`blockNum@12:int16`、`scalarFlag@14:int16` | `DoOneDimFormerTilingAdvance` | `BroadcastOneDimAdvanceSch` | 元素、元素/tile、核数、输入scalar位图。 |
| `BroadcastBaseTilingData<OpDag>` | `scheMode,shapeLen,ubSplitAxis,ubFormer,ubTail:int32`，随后 `ubOuter,blockFormer,blockTail,dimProductBeforeUbInner,elemNum,blockNum:int64` | `DoBroadcastOpTiling` | NDDMA/UB scheduler | mode、轴、元素、tile、核数。 |
| 同上 | `outputDims[8],outputStrides[8]` | shape collapse/stride推导 | GM offset、局部shape | 仅前shapeLen有效。 |
| 同上 | `inputBrcDims[2][8],inputBrcStrides[2][8]`；BitwiseXor/NotEqual两个输入均CopyInBrc | `SetInputNddmaParams` | NDDMA/UB Broadcast CopyIn | broadcast轴stride=0。其它普通/VecBrc数组在这两个DAG不活跃。 |
| 同上 | `scalarData[16]` | 零初始化；DAG无Var | 不读 | inactive。 |
| `CastTilingData` | 12×`int64_t`，offset依次0..88 | `CastTiling::DoOpTiling/PostTiling` | Cast各policy | `blockNum,ubFormer,blockFormer,ubLoopOfFormerBlock,ubLoopOfTailBlock,ubTailOfFormerBlock,ubTailOfTailBlock,regCopyInStep,regCopyOutStep,ubFormerRegLoop,ubTailOfFormerRegLoop,ubTailOfTailRegLoop`。 |

### Route 源码闭包

| Route | Host/Tiling producer | POD/参数 | entry/router | body/直接依赖 | build |
|---|---|---|---|---|---|
| R-001 | `GetWorkspaceSizeCommon` empty guard | workspaceSize=0 | 无 | 无 | ACLNN API模块。 |
| R-002 | `l0op::NotEqual`→`NotEqualTiling`→`BroadcastBaseTiling` | Advance或Base POD | `not_equal<mode>`→`BroadcastSch` | NotEqual DAG + 公共 OneDim/NDDMA/UB scheduler | NotEqual ascend950 arch35定义/binary/CMake。 |
| R-003 | `l0op::BitwiseXor`→`BitwiseXorTiling`→`BroadcastBaseTiling` | 同上 | `bitwise_xor<mode>`→`BroadcastSch` | Xor DAG + 公共scheduler | `bitwise_xor_def.cpp`、binary JSON、CMake仅ascend950。 |
| R-004 | `l0op::Cast`同dtype早返 | 无 | 无 | 无 | Cast API。 |
| R-005 | `IsAiCoreSupport`→`CastTiling` | `CastTilingData` | `cast<id>` | `cast_impl.h`各policy；INT64→DOUBLE `IsSimt` | Cast ascend950/arch35 CMake与binary。 |
| R-006 | `IsAiCpuSupport`→`CastAiCpu` | attrs+Tensor meta，无AIV POD | AICPU task `Cast` | `CastCpuKernel`/`CastTask` | Cast AICPU定义/源码模块。 |

## Kernel 实现

### 公共执行流程

公开图先用通用 Contiguous 建立连续输入。R-002/R-003的核心都是单个 broadcast AIV Kernel；`BroadcastSch::Process` 明确设置 `KERNEL_TYPE_AIV_ONLY`，再按 mode 构造 OneDim Advance、NDDMA或UB Broadcast scheduler。scheduler把输入搬进编译期 DAG tensor pool，执行 DAG，再把有效区写回GM。可选 Cast节点在核心之前/之后按 R-004/005/006独立物化；ViewCopy是框架边界。

### Route R-001：空 Tensor Host 短路

#### 适用条件

##### 选择证据

`GetWorkspaceSizeCommon` 在校验之后、Contiguous之前检查任一输入 `IsEmpty()`；命中即 `workspaceSize=0` 并移交 Executor。

#### Device 数据流

##### 函数级调用链

```text
GetWorkspaceSizeCommon -> empty guard -> ReleaseTo(executor) -> ACLNN_SUCCESS
```

##### 源码等价伪码

```text
if self.empty or other.empty: return success(workspace=0, launches=[])
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| GM/UB/workspace | 无核心分配、无搬运、无Device节点。 |

##### 分层内存布局

无Device布局。

##### 字节账与复用顺序

API返回0字节；无复用。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项目 | 行为 |
|---|---|
| stream/数值/tail | 不提交任务，不产生数值，不存在tail。 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 证据 |
|---|---|---|
| `aclnn_bitwise_xor_tensor.cpp` | `GetWorkspaceSizeCommon` | 空输入短路位于连续化和核心选择之前。 |

### Route R-002：BOOL→NotEqual AIV Broadcast

#### 适用条件

##### 选择证据

| 层 | 符号 | 条件/结果 |
|---|---|---|
| API | `CheckDtypeValid/GetWorkspaceSizeCommon` | 两输入BOOL→promoteType BOOL→调用NotEqual。 |
| l0 | `NotEqual::IsAiCoreSupport` | 950 BOOL在ASCEND950表→AICore launcher。 |
| Tiling | `NotEqualTiling::DoOpTiling` | BOOL实例化 `NotEqualCompute<int8_t>`，公共broadcast返回值向上传播。 |
| router | `not_equal<mode>` | BOOL再次特化int8 DAG；`BroadcastSch`设AIV_ONLY。 |

#### Device 数据流

##### 函数级调用链

```text
not_equal<mode>(x1,x2,y,workspace,tiling)
  -> BroadcastSch<mode,NotEqualCompute<int8_t>::OpDag>::Process
    -> scheduler CopyInBrc(x1), CopyInBrc(x2)
    -> Duplicate<uint8_t>(1), Duplicate<uint8_t>(0)
    -> Compare<uint8_t,int8_t,CMP_MODE=5>
    -> Select<uint8_t,uint8_t,SELECT_MODE=2>
    -> CopyOut<uint8_t>(y)
```

##### 源码等价伪码

```text
for each core tile:
  a = broadcast_copyin_bool_as_int8(x1)
  b = broadcast_copyin_bool_as_int8(x2)
  mask = compare_not_equal(a,b)
  local_y = select(mask, uint8(1), uint8(0))
  copyout_valid_count(y,local_y)
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | extent/地址 | producer→consumer | 生命周期/对齐 |
|---|---|---|---|
| x1/x2 GM | broadcast stride 0轴复用；其它轴由坐标×stride | Contiguous→CopyInBrc | Executor到Kernel结束。 |
| tensor pool | `elemNum*maxDtypeBytes*B_NE` | CopyIn/常量/Compare/Select→CopyOut | DAG LEVEL_2按buffer ID复用，tile间ping-pong。 |
| NDDMA | mode1用≤5D参数；mode2外层软件循环+5D参数 | GM→UB broadcast布局 | stride相同可退化连续搬入。 |
| UB Broadcast | 先CopyIn源片再 `Broadcast<T,R>` 扩展 | GM→源子区→展开子区 | mode101–109；stride0输入可跨tile cache。 |
| y GM | 有效BOOL元素区 | CopyOut→后续Cast/ViewCopy | 只写真实count。 |

##### 分层内存布局

```text
GM x1/x2 -> [CopyIn/NDDMA source] -> [broadcast local tensors]
          -> [Compare mask + constants + Select result] -> GM bool coreOut
```

##### 字节账与复用顺序

公共Tiling用 `B_NE=OpDag::GetBufferNum<cache,ubBrc>()` 做精确容量生产者，UB上界为 `B_NE*elemNum*1B`（另加scheduler按实现定义的额外区）；常量、mask和Select结果由LEVEL_2生命周期分析复用。固定仓未输出 `B_NE` 实例值，因此不伪造数字。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项目 | 实现 |
|---|---|
| 队列/复用同步 | DAG `GetTensor/ReleaseTensor` 与buffer ID控制生产消费；scheduler按ping-pong索引复用。 |
| 数值 | BOOL按字节读入，NE后显式选择0/1；无浮点精度问题。 |
| tail | OneDim与多轴均把真实count/局部shape传给DAG；CopyOut只写有效元素。 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 证据 |
|---|---|---|
| `not_equal.cpp` | `IsAiCoreSupport/NotEqualAiCore` | 950 BOOL固定加入AICore。 |
| `not_equal_tiling_arch35.cpp` | `NotEqualTiling::DoOpTiling` | BOOL映射int8 DAG并检查公共Tiling返回值。 |
| `not_equal_dag.h` | `NotEqualCompute` | NE Compare+Select 1/0完整body。 |
| `not_equal_apt.cpp` | `not_equal` | BOOL entry特化和公共router。 |

### Route R-003：整数 BitwiseXor AIV Broadcast

#### 适用条件

##### 选择证据

| 层 | 符号 | 条件/结果 |
|---|---|---|
| API | `GetWorkspaceSizeCommon` | promoteType非BOOL；0/1/2个输入Cast后调用BitwiseXor。 |
| l0 | `BitwiseXor::IsAiCoreSupport` | 950八种整数全部命中ASCEND950表。 |
| Tiling | `BitwiseXorTiling::DoOpTiling` | 三Tensor dtype必须相同且为八种整数，实例化对应Xor DAG。 |
| router | `bitwise_xor<mode>` | `BroadcastSch`执行AIV-only scheduler。 |

#### Device 数据流

##### 函数级调用链

```text
bitwise_xor<mode>(x1,x2,y,workspace,tiling)
  -> BroadcastSch<mode,BitwiseXorCompute<T>::OpDag>::Process
    -> CopyInBrc<T>(x1), CopyInBrc<T>(x2)
    -> XorCompute<T>
       -> UpdateMask<T> -> DataCopy RegTensor×2
       -> MicroAPI::Xor -> masked DataCopy store
    -> CopyOut<T>(y)
```

##### 源码等价伪码

```text
for each scheduled tile(count):
  local_a = broadcast_copyin<T>(x1)
  local_b = broadcast_copyin<T>(x2)
  VL = GetVRegSize()/sizeof(T)
  remaining = count
  for j in 0..ceil(count/VL)-1:
    pred = UpdateMask<T>(remaining)
    reg_a = load(local_a+j*VL); reg_b = load(local_b+j*VL)
    store_masked(local_y+j*VL, xor(reg_a,reg_b,pred), pred)
  copyout_valid_count(y,local_y)
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | extent/地址 | producer→consumer | 生命周期/对齐 |
|---|---|---|---|
| x1/x2 GM | collapsed坐标与inputBrcStrides；broadcast轴stride0 | 输入Cast/Contiguous→CopyInBrc | 核心期间只读。 |
| tensor pool | `B_XOR*elemNum*sizeof(T)` | 两输入→Xor结果→CopyOut | LEVEL_2、buffer ID、ping-pong。 |
| NDDMA | mode1≤5D；mode2高维外循环 | GM→broadcast后UB | NDDMA route仍属于VV。 |
| UB Broadcast | 源片+展开片 | GM→Vec::CopyIn→Broadcast | mode101–109；可cache stride0输入。 |
| RegTensor | 每批 `VL=VRegBytes/sizeof(T)` lanes | UB→Xor→UB | 最后批pred屏蔽。 |
| y GM | `count*sizeof(T)`有效区 | CopyOut→结果Cast | 无workspace寻址。 |

##### 分层内存布局

```text
GM promoted x1/x2 -> scheduler UB pool -> RegTensor<T> x1/x2
                                      -> MicroAPI::Xor -> UB result -> GM coreOut
```

##### 字节账与复用顺序

NDDMA/OneDim容量由 `B_XOR=OpDag::GetBufferNum<true,true>()`，UB Broadcast由对应 cache/ubBrc 模板参数的 `GetBufferNum` 精确生成；公共公式确保 `B_XOR*elemNum*sizeof(T)<=U-extra` 并按 `256B/sizeof(T)` 元素对齐。两个输入活到Xor完成，结果随后被CopyOut，LEVEL_2允许生命周期不重叠的ID复用。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项目 | 实现 |
|---|---|
| 同步 | scheduler/DAG的tensor acquire-release和ping-pong保证搬入、计算、搬出次序；无跨核归约。 |
| 精度 | 直接对T位模式XOR，无舍入；输入提升Cast和输出Cast按各自policy决定数值转换。 |
| tail | `UpdateMask<T>(sregMask)`逐轮更新，masked store不写寄存器尾lane；GM只搬真实count。 |
| 64位 | VL按sizeof(T)缩小；同一代码体支持INT64/UINT64。 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 证据 |
|---|---|---|
| `bitwise_xor.cpp` | `ASCEND950_AICORE_DTYPE_SUPPORT_LIST` | 八种整数不走BitwiseXor AICPU。 |
| `bitwise_xor_tiling.cpp` | `BitwiseXorTiling::DoOpTiling` | dtype检查与八个DAG实例。 |
| `bitwise_xor_dag.h` | `XorCompute/BitwiseXorCompute` | 完整MicroAPI XOR与DAG边。 |
| `bitwise_xor_apt.cpp` | `bitwise_xor` | entry到BroadcastSch。 |

### Route R-004：Cast identity

#### 适用条件

##### 选择证据

`l0op::Cast` 首句比较源/目标dtype，相等直接返回原指针；输入预Cast和输出Cast都可命中。

#### Device 数据流

##### 函数级调用链

```text
l0op::Cast(self,dst) -> if self.dtype==dst: return self
```

##### 源码等价伪码

```text
cast_node = alias(input)
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| Tensor/GM/UB | 复用输入对象，不分配输出、不启动Kernel、不搬运。 |

##### 分层内存布局

输入Tensor别名直接连接下一节点。

##### 字节账与复用顺序

新增0字节。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项目 | 行为 |
|---|---|
| 同步/精度/tail | 无节点；位值不变。 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 证据 |
|---|---|---|
| `cast/op_api/cast.cpp` | `l0op::Cast` | 同dtype早返。 |

### Route R-005：Cast AIV（含 INT64→DOUBLE SIMT guard）

#### 适用条件

##### 选择证据

不同dtype pair命中 `REGBASE_AICORE_DTYPE_SUPPORT_LIST`。输入提升可达的整数pair均在该表；输出转换的部分pair命中。`castMap`再选择 Direct/Through/Two/Micro等policy。仅 Direct INT64→DOUBLE使 `IsSimt()`为真；它仍从设置了 `KERNEL_TYPE_AIV_ONLY` 的Cast entry启动，SIMT属于VV执行域。

#### Device 数据流

##### 函数级调用链

```text
l0op::Cast -> CastAiCore -> cast<policy id>
  -> GET_TILING_DATA -> KERNEL_TYPE_AIV_ONLY
  -> CastDirect / CastThrough / CastTwo / CastMicro 等实现
  -> CopyIn GM->UB -> Cast/Copy/MicroAPI [-> mid UB] -> CopyOut UB->GM
```

##### 源码等价伪码

```text
coreBase = blockIdx*blockFormer
loops,tail = select(lastCore, tailFields, formerFields)
for each UB tile:
  xLocal = DataCopyPad(src,count)
  yLocal = policy(xLocal, optional_mid, round_modes, reg_copy_modes)
  DataCopyPad(dst,yLocal,count)
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 容量 | owner/流程 | 复用/对齐 |
|---|---|---|---|
| input queue | `2*ubFormer*sizeof(S)` | Cast policy；GM→UB | 双缓冲。 |
| output queue | `2*ubFormer*sizeof(D)` | UB→GM | 双缓冲。 |
| mid buffer | TwoCast为 `ubFormer*sizeof(M)`；Micro按policy寄存器/局部布局 | 两段转换之间 | 非TwoCast不活跃。 |
| SIMT reserve | INT64→DOUBLE从U扣32KiB并 `SetLocalMemorySize` | Direct policy | 其它pair为0。 |
| workspace | Tiling声明16MiB，body未寻址 | Executor | 不等同API总workspace。 |

##### 分层内存布局

```text
GM S -> input DB -> [vector/micro or SIMT conversion] -> optional mid -> output DB -> GM D
```

##### 字节账与复用顺序

Direct/Through/Micro的 `ubFormer=floor((((U-reserve)-alignReserve)*4)/(srcBits+dstBits)/alignInput)*alignInput`；TwoCast分母再加 `midBits`。`alignInput=2048/srcBits`。队列按2份双缓冲分配，TwoCast另占一份mid；tail循环沿用同一UB区。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项目 | 实现 |
|---|---|
| 同步 | `TQue` 的 `Alloc/EnQue/DeQue/FreeTensor` 串联双缓冲。 |
| 数值 | round mode和中间dtype来自编译期 `GetCastPolicy<src,dst>`；不能统一假设截断。 |
| tail | `ubTailOfFormerBlock/ubTailOfTailBlock`决定最后搬运字节和Cast count。 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 证据 |
|---|---|---|
| `cast.cpp` | `IsAiCoreSupport/CastAiCore` | 950 pair selector。 |
| `cast_tiling.cpp` | `IsSimt/GetUbFormer` | INT64→DOUBLE唯一SIMT guard和32KiB reserve。 |
| `cast_apt.cpp` | `cast<id>` | AIV-only entry与policy dispatch。 |
| `cast_impl.h` | `CastDirect/CastTwo/CastMicro` | GM-UB-寄存器/中间buffer实现。 |

### Route R-006：Cast AICPU

#### 适用条件

##### 选择证据

源目标不同、pair未命中950 AICore表，但两端都在 `AICPU_DTYPE_SUPPORT_LIST`。公开out中的COMPLEX64/128以及部分DOUBLE/UINT16/32/64转换可到此；整数→BOOL已在API `CanCast` 处提前拒绝。

#### Device 数据流

##### 函数级调用链

```text
l0op::Cast -> IsAiCoreSupport(false) -> IsAiCpuSupport(true)
  -> ADD_TO_LAUNCHER_LIST_AICPU(Cast)
  -> CastCpuKernel::Compute
     -> ValidateTensors -> ValidateDataType -> SetMap
     -> ExecuteCast -> CpuKernelUtils::ParallelFor
     -> CastTask<S,D>(start,end)
```

##### 源码等价伪码

```text
n = min(input_bytes/sizeof(S), output_bytes/sizeof(D))
cores = min(n, max(1, GetCPUNum()-reservedCpu))
parallel_for(n, grain=n/cores): out[i] = cast_task<S,D>(in[i])
```

#### 内存与数据搬运

##### 资源与搬运表

| 资源 | 行为 |
|---|---|
| input/output Tensor memory | AICPU线程按不重叠 `[start,end)` 直接读写；无AIV UB/POD。 |
| task attrs | 非complex源传 `dst_type`；complex源路径传 `SrcT/DstT/Truncate=false`，本API核心源不是complex。 |

##### 分层内存布局

```text
Tensor input memory -> AICPU parallel CastTask -> Tensor output memory
```

##### 字节账与复用顺序

处理元素数由较小的输入/输出容量确定；线程数上限为可用CPU数与元素数的较小者。固定仓未公开AICPU task-space自身workspace字节，标记为框架边界。

#### 同步、精度与尾块

##### 同步与数值边界表

| 项目 | 实现 |
|---|---|
| 同步 | `ParallelFor`分配不重叠区间；Executor保证它与前后节点的依赖顺序。 |
| 数值 | `CastTask<S,D>`表选择具体转换；complex构造与普通标量转换按模板实现。 |
| tail | 最后CPU区间自然覆盖到n，无向量mask。 |

#### 关键实现

##### 符号证据表

| 文件 | 符号 | 证据 |
|---|---|---|
| `cast/op_api/cast.cpp` | `IsAiCpuSupport/CastAiCpu` | AICPU fallback门与attrs。 |
| `cast_aicpu.cpp` | `SetMap/ValidateDataType/ExecuteCast` | dtype映射、元素数、分核和ParallelFor。 |

## 算子特定优化

### Trick 1：BOOL XOR 复用 NotEqual，而非整数按位 Kernel

#### trigger / 为什么成立 / 收益 / 代价 / 正确性边界

trigger是 `promoteType==BOOL`。BOOL逻辑 XOR恰等价于不相等，Host直接复用NotEqual的Compare+Select DAG，避免为BOOL增加BitwiseXor实例。边界是只有两输入均BOOL才会提升到BOOL；BOOL与整数会进入整数XOR且BOOL先Cast。

### Trick 2：公共broadcast selector在NDDMA与UB展开之间切换

#### trigger / 为什么成立 / 收益 / 代价 / 正确性边界

大非尾轴广播或8/16-bit尾轴32B对齐时用UB Broadcast，其余多轴用NDDMA；运行rank>5时NDDMA增加软件外循环。这样把NDDMA 5D限制和UB容量显式纳入Tiling。两路径必须使用同一collapsed dims/strides和真实tail shape，才保持相同broadcast语义。

### Trick 3：同dtype Cast完全消除

#### trigger / 为什么成立 / 收益 / 代价 / 正确性边界

输入已经是promoteType或核心结果已经等于out dtype时，`l0op::Cast`返回原Tensor，不创建launcher和临时GM。该优化只在dtype严格相等时成立，不依据位宽相等做别名。

## 模板与实现对比

| 对比项 | BitwiseXor整数核心 | BOOL NotEqual核心 | Cast AIV | Cast AICPU |
|---|---|---|---|---|
| 入口 | `bitwise_xor<mode>` | `not_equal<mode>` | `cast<id>` | `CastCpuKernel::Compute` |
| 主模板 | BroadcastSch + Xor DAG | BroadcastSch + Compare/Select DAG | policy id Direct/Through/Two/Micro | dtype函数表 |
| 执行族 | AIV/NDDMA（VV） | AIV/NDDMA（VV） | AIV，特定pair含SIMT（VV） | AICPU |
| 计算 | 位模式XOR | BOOL NE→0/1 | 数值类型转换 | CPU类型转换 |
| Tiling/POD | 公共Advance/Base POD | 同族但DAG buffer数不同 | 96B Cast POD | 无AIV POD |
| tail | MicroAPI pred + scheduler真实count | scheduler真实count | Cast POD tail fields | ParallelFor区间尾 |

不要仅按函数名把本算子视为单一XOR Kernel：BOOL路径、可选Cast以及Cast的AIV/AICPU selector都是公开API可达的实际实现。

## 平台差异

| 平台 | 公开情况 | 固定提交实现证据 | 本卡处理 |
|---|---|---|---|
| 950/950 | 文档支持全部9种输入；额外支持UINT16/32/64及complex输出 | BitwiseXor仅ascend950 AICore配置、arch35 Tiling/binary/CMake；NotEqual/Cast 950直接依赖闭合 | 发布本 `_950` 卡。 |
| A2/A3 | 文档声明支持，输入不含UINT16/32/64，输出也不含unsigned16+与complex | `BitwiseXor` l0对部分dtype可走AICPU，但 `math/bitwise_xor/` 固定目录无AICPU body；当前BitwiseXor CMake/OpDef仅ascend950，无法闭合A2/A3全部公开dtype | 不创建无后缀卡，不用950实现外推。 |

## 限制与失败边界

- `self/other/out` 任一空指针返回161001；dtype、提升、cast、broadcast、out shape或rank约束失败返回161002。
- shape rank最大8；out shape必须精确等于broadcast结果，不能只要求元素数相同。
- UINT16/UINT32/UINT64 与不同输入dtype在固定提升表中通常为UNDEFINED；相同dtype仍支持。
- 非BOOL promote结果不能Cast到BOOL；文档的总输出dtype表不表示每种输入都能产生每种输出。
- BitwiseXor Tiling调用公共 `DoTiling()` 后未检查其返回值再取mode；这是固定源码事实。公共Tiling内部资源失败如何被外层运行时最终处理，固定仓没有更多证据，不能宣称安全fallback。
- 三个核心entry的workspace参数均不在body中寻址，但API workspace还受通用节点影响，不能传小于第一段返回值的区域。
- Contiguous、ViewCopy、Executor workspace合并/event细节属于框架边界；不推测其Kernel/POD。
- 本卡无真机、编译、数值或性能结论；所有事实限定在 ops-math 提交 `165d41f02a43a62324b434cb6bda7c533b14da3e` 与直接依赖 opbase 提交 `d574f5571391390f8b619c6a134b65c819842318`。

## 源码索引

### 链路索引

| 链路层 | 路径/符号 |
|---|---|
| 公共文档/头 | `math/bitwise_xor/docs/aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor.md`；`op_api/aclnn_bitwise_xor_tensor.h` |
| ACLNN Host | `op_api/aclnn_bitwise_xor_tensor.cpp::CheckDtypeValid/CheckShape/GetWorkspaceSizeCommon/aclnnBitwiseXorTensor*` |
| 类型提升 | opbase `include/nnopbase/opdev/data_type_utils.h::kPromoteTypesLookup/PromoteType/CanCast` |
| XOR l0/Host | `op_api/bitwise_xor.cpp`；`op_host/bitwise_xor_def.cpp`；`op_host/arch35/bitwise_xor_tiling.cpp` |
| XOR Kernel | `op_kernel/bitwise_xor_apt.cpp`；`op_kernel/arch35/bitwise_xor_dag.h::XorCompute` |
| BOOL依赖 | `math/not_equal/op_api/not_equal.cpp`；`op_host/arch35/not_equal_tiling_arch35.cpp`；`op_kernel/arch35/not_equal_dag.h`；`op_kernel/not_equal_apt.cpp` |
| Cast依赖 | `math/cast/op_api/cast.cpp`；`op_host/arch35/cast_tiling.{h,cpp}`；`op_kernel/cast_apt.cpp`；`op_kernel/arch35/cast_impl.h`；`op_kernel_aicpu/cast_aicpu.cpp` |
| Broadcast依赖 | opbase `broadcast_base_struct.h`、`broadcast_tiling.h/.cpp`、`broadcast_sch.h`、`broadcast_sch_one_dim_advance.h`、`broadcast_sch_nddma.h`、`broadcast_sch_ub_broadcast.h` |
| 构建接线 | `math/bitwise_xor/CMakeLists.txt`、`op_host/config/ascend950/bitwise_xor_binary.json`；NotEqual/Cast对应ascend950定义与配置 |

### 实现复用检查

复用 NotEqual、Cast 和 ATvoss broadcast 是源码直接调用/包含关系，不是相似实现推断。BOOL只复用NotEqual，不并入独立 NotEqual 逻辑算子语义；Cast只作为本公开图的可达直接依赖展开；Scalar与Inplace入口均没有纳入。未发现任何可达 Cube/AIC/MIX_CV body，NDDMA与INT64→DOUBLE的SIMT均按VV Skill归类。