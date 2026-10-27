---
type: Operator
title: "InplaceTriu 950 ACLNN 算子知识"
description: "说明 aclnnInplaceTriu 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/docs/aclnnTriu&aclnnInplaceTriu.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_api/aclnn_triu.h"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_api/aclnn_triu.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_api/triu.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triu_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triu_tiling.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/triu_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/triangulator_base.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_all.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_tiny_shape.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_medium_shape.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_row_and_col.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/triu_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/config/ascend950/triu_binary.json"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/CMakeLists.txt"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json"
    role: implementation
created_at: "2026-08-25T09:52:04Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceTriu 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceTriuGetWorkspaceSize` + `aclnnInplaceTriu`，无版本后缀的公开两段式 ACLNN 接口 |
| 计算语义 | 原地更新最后二维：若列号 `j >= i + diagonal` 则保留 `selfRef[..., i, j]`，否则写零 |
| 输入输出 | 同一个 `selfRef` 既是输入又是输出；公开约束为 2～8 维、非私有格式；950 支持 1/2/4/8-byte dtype，包括仅 950 开放的 `COMPLEX32/COMPLEX64` |
| 执行类型 | 算子拥有的 `triu` 节点为 `AIV_VECTOR`（`KERNEL_TYPE_AIV_ONLY`）；`Contiguous`/`ViewCopy` 是 ACLNN 框架边界，不含算子自有 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001 全零、R-002 全拷贝、R-003 tiny 矩阵、R-004 medium 整平面、R-005 normal 二维分块；每条 Route 的算子核心均为单 AIV Kernel |
| 核心资源 | GM 输入/临时输出、AIV UB、`TBuf/TQue/RegTensor`；Tiling 为核心节点声明 16 MiB user workspace，但 Kernel 仅检查其非空，不读写该区 |
| 来源闭包 | `source_closed`：公开 Inplace ACLNN、参数校验、executor 图、950 Tiling/POD、20 个 key router、五个 body、950 OpDef/binary/CMake 接线均已闭合；框架 `Contiguous/ViewCopy/CommonOpExecutorRun` 作为明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入最后两维为 `row × col`，前导维展平后的矩阵数为 `F`。对每个前导索引、行 `i`、列 `j` 和属性 `d = diagonal`：

```text
selfRef_after[..., i, j] = selfRef_before[..., i, j],  j >= i + d
selfRef_after[..., i, j] = 0,                          j <  i + d
```

接口的“原地”语义发生在 ACLNN 图末端：核心 `Triu` 先分配连续临时结果，随后 `ViewCopy` 把结果写回 `selfRef` 的原存储，因此核心 Kernel 不会边读边覆盖调用者输入。公开文档声明该接口默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | 公开范围 2～8 维；ND、NCHW、NHWC、HWCN、NDHWC、NCDHW 等非私有 storage format；核心 Tiling 使用 origin shape 最后两维 | 950：`UINT8/INT8/BOOL`，`UINT16/INT16/FLOAT16/BFLOAT16/COMPLEX32`，`UINT32/INT32/FLOAT/COMPLEX64`，`UINT64/INT64/DOUBLE` | 非空；同时作为输入和最终输出；非连续布局由 `Contiguous` 与 `ViewCopy` 框架节点处理 |
| `diagonal` | 标量属性 | Host `int64_t` | `d >= col` 时全零，`d <= 1-row` 时全拷贝，其余按逐行边界处理 |
| `workspaceSize` | Host 输出标量 | `uint64_t*` | 必须非空；由 executor 汇总 `Contiguous`、临时输出、核心节点 16 MiB user workspace 和 `ViewCopy` 所需空间，固定仓未展开框架汇总公式 |
| `executor` | Host 输出句柄 | `aclOpExecutor**` | 必须非空；第一段构造，第二段消费 |
| `workspace/workspaceSize/executor/stream` | 第二段执行参数 | runtime ABI | 交给 `CommonOpExecutorRun`，在指定 stream 上执行第一段记录的节点序列 |

### 入口与版本

固定版本只公开无后缀的 `aclnnInplaceTriuGetWorkspaceSize`/`aclnnInplaceTriu`，不存在更高版本号接口。第一段把同一 `selfRef` 同时传给共享 helper `doTriuGetWorkspaceSize(selfRef, diagonal, selfRef, ...)`；这只是实现复用，不改变 InplaceTriu 的独立原地语义。公开文档同时声明 A2、A3、950 支持；本卡只发布具有完整 `ascend950` Host/Kernel/构建接线的 950。固定快照没有 A2/A3 活动后端接线，故不据文档声明外推 A2/A3 实现。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize`、`executor`。
2. `CheckParams(selfRef, selfRef)` 检查指针、当前平台 dtype、私有 storage format 与 rank。因为输入输出是同一对象，dtype/shape/format 相等检查天然成立；非空张量仍要求 rank 至少 2，rank 上限为 8。
3. `selfRef->IsEmpty()` 时直接返回空 executor 的 workspace 大小，不添加 Device 节点。
4. 非空时调用框架 `l0op::Contiguous` 得到连续输入；其已连续时是否别名复用、非连续时所需 workspace 和具体 Device body 在固定仓不可见。
5. `l0op::Triu` 以输入 storage shape/dtype/format 分配临时 `out`，通过 `ADD_TO_LAUNCHER_LIST_AICORE(Triu, AI_CORE, ...)` 加入核心节点。
6. `l0op::ViewCopy(triuResult, selfRef, executor)` 把连续临时结果提交回调用者 storage；具体 stride copy body 属框架边界。
7. 任一节点构造返回空指针即返回 `ACLNN_ERR_INNER_NULLPTR`；成功后由 `GetWorkspaceSize` 返回总空间并移交 executor。

### 参数、Workspace 与生命周期

`selfRef` 在整个 executor 生命周期内由调用者持有。`Contiguous` 结果和 `Triu` 临时输出由 executor 分配并至少存活到其下游消费完成；`ViewCopy` 完成后调用者 storage 才包含最终原地结果。`TriangulatorTiling::PostTiling` 为核心节点设置 `userWorkspaceSize[0] = 16,777,216` bytes；`triangulatorEntry` 通过 `GetUserWorkspace` 取得地址并只做 null guard，所有五个 body 都不读写它。ACLNN 返回的总 `workspaceSize` 是 executor 级汇总值，固定仓没有 `Contiguous/ViewCopy` 的 allocator 和切片公式，不能把总值误写为恒定 16 MiB。

### 启动图与失败传播

```text
selfRef (caller-owned, possibly strided)
  -> N0 l0op::Contiguous [framework boundary, same executor/stream]
  -> edge E0: contiguous temporary, original dtype/shape
  -> N1 Triu/triu [AIV_VECTOR, blockDim=usedCoreNum, one POD selected by key]
  -> edge E1: contiguous Triu temporary, original dtype/shape
  -> N2 l0op::ViewCopy [framework boundary, same executor/stream]
  -> selfRef storage (caller-owned final state)
```

| 节点/边 | Host 添加函数 / entry | 参数与空间 | ordering、owner/lifetime、失败传播 |
|---|---|---|---|
| N0 | `doTriuGetWorkspaceSize` → `l0op::Contiguous` | `selfRef` → 连续 tensor；框架 workspace 切片固定仓不可见 | executor 记录顺序；失败返回 null 并阻断 N1/N2；临时 tensor 由 executor 持有 |
| E0 | `selfContiguous` | shape/dtype 不变；layout 转连续 | N0 producer、N1 consumer；至少活到 N1 读完 |
| N1 | `l0op::Triu` → `ADD_TO_LAUNCHER_LIST_AICORE` → `triu` | `x,y,workspace,tiling`；`blockDim=usedCoreNum`；核心 user workspace 请求 16 MiB | 与同 executor 前后节点按 stream 顺序执行；Tiling/launch 失败阻断后续；临时 `y` 由 executor 持有 |
| E1 | `triuResult` | 连续 GM，extent=`F*row*col*D` bytes | N1 producer、N2 consumer；N2 完成前不可回收或覆盖 |
| N2 | `doTriuGetWorkspaceSize` → `l0op::ViewCopy` | E1 → `selfRef` storage；框架 workspace 切片固定仓不可见 | executor stream 顺序保证核心输出可见；失败返回 null；完成后 `selfRef` 可观察 |

第二段 `aclnnInplaceTriu` 只调用 `CommonOpExecutorRun`；框架的逐节点 stream/event 实现与 cleanup 细节在固定仓不可见。算子自有核心是单 Kernel，无跨核心 Kernel workspace 数据边。

## Route 与参数设计

### 分核与资源计算

设 `D = dtypeBytes ∈ {1,2,4,8}`、`R=row`、`C=col`、`F=∏shape[0:-2]`、`N=F*R*C`、`A=availableAIVCoreNum`、`U=availableUBSize-8192`、`V=GetVRegSize(context)`。`TilingPrepare4Trilu` 从平台读取 AIV 核数和 UB；任一值不大于零即失败。

Route 判定完成后，key 恒为 `10*mode + D`。R-001/R-002 的 `baseBlockNum=N`，以不少于约 1024 bytes 的连续段分核：若 `N*D<1024` 则单核；否则 `normal=max(ceil(N/A), floor(1024/D))`、`used=ceil(N/normal)`、`tail=N-normal*(used-1)`。R-003/R-004 的 base block 是前导维 group；R-005 的 base block 是 `F*ceil(R/rowInner)*ceil(C/colInner)`。后三者先取 `used=min(baseBlockNum,A)`，再令 `q=floor(baseBlockNum/used)`、`r=baseBlockNum%used`；Device 让前 `r` 个 core 各处理 `q+1` block，其余处理 `q` block。

缓冲区大小均以字节计并向下 32-byte 对齐：R-001/R-002 为 `B=floor_align(U/2,32)`；R-003/R-004 为 `B=floor_align(U/4,32)`；R-005 为 `B=floor_align(U/8,32)`。`8192` bytes 预留量不进入这些 `B`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 全零 | `1/2/4/8`；`SplitAll<int{8,16,32,64}_t, OUTPUT_ZERO_MODE>` | `d >= C` | R-002 |
| 2 | R-002 全拷贝 | `11/12/14/18`；`SplitAll<carrier, OUTPUT_INPUT_MODE>` | 前一条件不成立且 `d <= 1-R` | R-003 |
| 3 | R-003 tiny | `31/32/34/38`；`SplitTinyShape<carrier>` | 前两条件不成立且 `R*C*D <= V` | R-004 |
| 4 | R-004 medium | `41/42/44/48`；`SplitMediumShape<carrier,false>` | 前三条件不成立且 `R*ceil_align(C*D,V)*4 <= U` | R-005 |
| 5 | R-005 normal | `21/22/24/28`；`SplitRowAndCol<carrier,false>` | 上述条件全部不成立 | Tiling 资源/序列化失败 |

```text
if d >= C:
    mode = 0; key = D
else if d <= 1 - R:
    mode = 1; key = 10 + D
else if R*C*D <= V:
    mode = 3; key = 30 + D
else if R*ceil_align(C*D, V)*4 <= U:
    mode = 4; key = 40 + D
else:
    mode = 2; key = 20 + D
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 全零 | `1,2,4,8` | 950，`D` 对应 key，`d>=C` | 单 `triu` AIV 节点 | `SplitAll<T,0>::Process` | 1 个 `TBuf`，预先 `Duplicate(0)`，V→MTE3 event | 尾 loop 用有效 `tailLoopBlockNum`；无下一 Device fallback |
| R-002 全拷贝 | `11,12,14,18` | 950，`d<=1-R` | 单 `triu` AIV 节点 | `SplitAll<T,1>::Process` | 2 个 `TBuf` ping-pong，MTE2↔MTE3 event | 尾 loop 用有效长度；无下一 Device fallback |
| R-003 tiny | `31,32,34,38` | 其余对角线且 `R*C*D<=V` | 单 `triu` AIV 节点 | `SplitTinyShape<T>::Process` | 2-depth in/out queues + 32-byte mask buffer；Host mask | `highTail` 处理最后 group；DMA 以 VReg stride pad |
| R-004 medium | `41,42,44,48` | tiny 不成立且整平面 4-buffer 能装 UB | 单 `triu` AIV 节点 | `SplitMediumShape<T,false>::Process` | 2-depth in/out queues；MicroAPI 行掩码 | `highTail`；整行 DMA pad；资源不满足已在 Host 转 R-005 |
| R-005 normal | `21,22,24,28` | 其余合法非空输入 | 单 `triu` AIV 节点 | `SplitRowAndCol<T,false>::Process` | 3 组 2-depth queue + zero TBuf；block 区域分类 | `rowTail/colTail` 与 DMA 有效长度；未知 key 静默不调用 body |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台资源 | `A>0 && availableUBSize>0`，随后 `U=UB-8192` | 分核和所有 UB 公式依赖正核数/UB；8 KiB 给框架/额外资源留余量 | 全部 | `GRAPH_FAILED` | `triu_tiling.cpp::TilingPrepareTrilForAscendC`；`triangulator_tiling.cpp::GetPlatformInfo` |
| dtype 宽度 | `GetSizeByDataType(xDtype)!=0` | key 个位和 carrier 模板只覆盖 1/2/4/8 bytes | 全部 | `GRAPH_FAILED` 或 ACLNN dtype error | `Tiling4Trilu`；`ComputeTilingKey`；`triangulatorEntry` |
| tiny mask 容量 | `R*C*D<=V` | 一个矩阵须放入一个 VReg；Host 的 `highMask[4]` 最多 256 bits，再按 D 扩展 | R-003 | R-004 | `ComputeTilingMode`；`ComputeMask`；`SplitTinyShape::DoVectorCopy` |
| medium 四缓冲 | `R*ceil_align(C*D,V)*4<=U` | in/out 各 depth=2，整 `highInner` 个完整平面装入 UB | R-004 | R-005 | `ComputeTilingMode`；`ComputeMediumInfo` |
| normal 小列 | `C<=floor_align(V,32)/D` | `colInner` 对齐到 32-byte block，单行可进入 VReg | R-005 | 继续 normal 的下一子分支 | `ComputeNormalInfo` |
| normal 小行 | 前项不成立且 `R<=128` | 固定整行数、沿列切片，避免小行场景再次切行 | R-005 | normal 通用子分支 | `ComputeNormalInfo` |
| AIV-only | entry 固定 `KERNEL_TYPE_AIV_ONLY` | 所有 body 只使用 GM/UB/Vector MicroAPI；无 Cube/AIC/MIX_CV | 全部 | N/A | `triu_apt.cpp::triu` |
| user workspace null | `GetUserWorkspace(workspace)!=nullptr` | entry 在 router 前做 guard；body 本身不消费该空间 | 全部 | Kernel 直接 return，无输出 | `triangulator_base.h::triangulatorEntry` |
| raw tiling capacity | 当前 Route POD `GetDataSize() <= raw capacity` | 防止序列化越界 | 全部 | `GRAPH_FAILED` | `TriangulatorTiling::PostTiling` |

### POD/ABI

所有声明字段都是 `int64_t`/`uint64_t`。固定仓只含 `BEGIN_TILING_DATA_DEF` 宏调用，不含 `register/tilingdata_base.h` 的宏展开，因此下表可闭合字段声明顺序和 64-bit 宽度，但 `sizeof/alignof/offsetof` 必须标为固定仓不可见，不能按普通 C struct 猜测。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `TriangulatorTilingData` | `bufferSize` | `int64_t/64` | 第 1 字段；offset/sizeof/alignof 固定仓不可见 | bytes | R-001/R-002：`floor_align(U/2,32)` | `SplitAll::ParseTilingData`；R-001 乘 2，R-002 原值 | 32-byte 对齐 |
| 同上 | `usedCoreNum` | `int64_t/64` | 第 2 字段；其余固定仓不可见 | AIV core | `OutputZeroAndInputTiling` | `SplitAll::Init/Process` | `blockIdx>=used` return |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第 3 字段；其余固定仓不可见 | elements | 同上 | `SplitAll::Init` | 非尾核覆盖量 |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第 4 字段；其余固定仓不可见 | elements | 同上 | `SplitAll::Init` | 最后一核覆盖量 |
| `TriangulatorNormalTilingData` | `bufferSize` | `int64_t/64` | 第 1 字段；宏布局固定仓不可见 | bytes | `floor_align(U/8,32)` | `SplitRowAndCol::Init` | 每个 queue slot/TBuf 容量 |
| 同上 | `row` | `int64_t/64` | 第 2 字段；固定仓不可见 | elements | `params->row` | `ParseTilingData` | 最后倒数二维 |
| 同上 | `col` | `int64_t/64` | 第 3 字段；固定仓不可见 | elements | `params->col` | `ParseTilingData/ComputeBlockInfo` | 最后一维 |
| 同上 | `diagOffset` | `int64_t/64` | 第 4 字段；固定仓不可见 | columns | `params->diagonal` | `ComputeBlockInfo` | 可正可负 |
| 同上 | `usedCoreNum` | `int64_t/64` | 第 5 字段；固定仓不可见 | core | `min(baseBlockNum,A)` | `Init/Process` | `blockIdx` guard |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第 6 字段；固定仓不可见 | base blocks/core | `baseBlockNum/used` | `Process` | 余数之后的 core |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第 7 字段；固定仓不可见 | remainder cores | `baseBlockNum%used` | `Process` | 前若干 core 多处理 1 block；字段名不是尾核元素数 |
| 同上 | `rowInner` | `int64_t/64` | 第 8 字段；固定仓不可见 | rows/block | `ComputeNormalInfo` | `ParseTilingData/ComputeBlockInfo` | 末块 `rowTail` 派生 |
| 同上 | `colInner` | `int64_t/64` | 第 9 字段；固定仓不可见 | columns/block | `ComputeNormalInfo` | `ParseTilingData/ComputeBlockInfo` | 末块 `colTail` 派生 |
| `TriangulatorTinyTilingData` | `bufferSize` | `int64_t/64` | 第 1 字段；宏布局固定仓不可见 | bytes | `floor_align(U/4,32)` | `SplitTinyShape::InitBuffer` | 每 queue slot |
| 同上 | `planeArea` | `int64_t/64` | 第 2 字段；固定仓不可见 | elements/matrix | `R*C` | DMA blockLen、GM offset | `planeArea*D<=V` |
| 同上 | `usedCoreNum` | `int64_t/64` | 第 3 字段；固定仓不可见 | core | `min(highOuter,A)` | `Init/Process` | block guard |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第 4 字段；固定仓不可见 | groups/core | `highOuter/used` | `Process` | 非余数 core |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第 5 字段；固定仓不可见 | remainder cores | `highOuter%used` | `Process` | 前若干 core 多 1 group |
| 同上 | `highInner` | `int64_t/64` | 第 6 字段；固定仓不可见 | matrices/group | `ComputeTinyInfo` | `ComputeBlockInfo` | 正常 group |
| 同上 | `highOuter` | `int64_t/64` | 第 7 字段；固定仓不可见 | groups | `ceil(F/highInner)` | `ComputeBlockInfo` | 最后一组判定 |
| 同上 | `highTail` | `int64_t/64` | 第 8 字段；固定仓不可见 | matrices | `F%highInner`，零则 `highInner` | `ComputeBlockInfo` | 最后一组有效量 |
| 同上 | `highMask[4]` | `uint64_t[4]/256` | 第 9 字段；数组内部连续性由宏实现，offset/sizeof 固定仓不可见 | one bit/matrix element | `ComputeMask` 按行列置保留位 | `Init` 写 32-byte mask UB；`DoVectorCopy` 扩展到 dtype lane | 只在 R-003 活跃 |
| `TriangulatorMediumTilingData` | `bufferSize` | `int64_t/64` | 第 1 字段；宏布局固定仓不可见 | bytes | `floor_align(U/4,32)` | in/out queue slot | 每队列 depth=2 |
| 同上 | `row` | `int64_t/64` | 第 2 字段；固定仓不可见 | rows | `params->row` | DMA blockCount、vector loops | 非空 |
| 同上 | `col` | `int64_t/64` | 第 3 字段；固定仓不可见 | elements/row | `params->col` | DMA blockLen、regCount | 非空 |
| 同上 | `copyCols` | `int64_t/64` | 第 4 字段；固定仓不可见 | elements | `(headRaw>0?1:-headRaw+1)-1`（Triu） | 中间行 `UpdateMask(copyCols+i)` | R-004 |
| 同上 | `usedCoreNum` | `int64_t/64` | 第 5 字段；固定仓不可见 | core | `min(highOuter,A)` | `Init/Process` | block guard |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第 6 字段；固定仓不可见 | groups/core | `highOuter/used` | `Process` | 非余数 core |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第 7 字段；固定仓不可见 | remainder cores | `highOuter%used` | `Process` | 前若干 core 多 1 group |
| 同上 | `highInner` | `int64_t/64` | 第 8 字段；固定仓不可见 | matrices/group | `ComputeMediumInfo` | `planeArea=R*C*highInner` | 正常 group |
| 同上 | `highOuter` | `int64_t/64` | 第 9 字段；固定仓不可见 | groups | `ceil(F/highInner)` | 最后一组判定 | base block 数 |
| 同上 | `highTail` | `int64_t/64` | 第 10 字段；固定仓不可见 | matrices | `F%highInner`，零则 `highInner` | `ComputeBlockInfo` | 最后一组有效量 |
| 同上 | `headRows` | `int64_t/64` | 第 11 字段；固定仓不可见 | rows | `max(-d,0)` | 全保留/全零行循环 | R-004 |
| 同上 | `midRows` | `int64_t/64` | 第 12 字段；固定仓不可见 | rows | `R-headRows-tailRows` | 掩码行循环 | R-004 |
| 同上 | `tailRows` | `int64_t/64` | 第 13 字段；固定仓不可见 | rows | `max(R-C-(-d),0)` | 全保留/全零行循环 | R-004 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 全零 | `1/2/4/8` | `ComputeTilingMode/OutputZeroAndInputTiling` | `TriangulatorTilingData` | `triu` → `triangulatorEntry<false>` | `SplitAll<T,OUTPUT_ZERO_MODE>` | `triu_def.cpp` + `triu_binary.json` + `CMakeLists.txt` |
| R-002 全拷贝 | `11/12/14/18` | 同上 | `TriangulatorTilingData` | 同上 | `SplitAll<T,OUTPUT_INPUT_MODE>` | 同上 |
| R-003 tiny | `31/32/34/38` | `ComputeTinyInfo/ComputeMask` | `TriangulatorTinyTilingData` | 同上 | `SplitTinyShape<T>` | 同上 |
| R-004 medium | `41/42/44/48` | `ComputeMediumInfo` | `TriangulatorMediumTilingData` | 同上 | `SplitMediumShape<T,false>` | 同上 |
| R-005 normal | `21/22/24/28` | `ComputeNormalInfo` | `TriangulatorNormalTilingData` | 同上 | `SplitRowAndCol<T,false>` | 同上 |

## Kernel 实现

### 公共执行流程

所有非空 Route 共用 `l0op::Triu` 添加的单核心节点。`triu(x,y,workspace,tiling)` 声明 AIV-only 后调用 `triangulatorEntry<false>`；这里 `false` 是 Device `IS_LOWER=false`，表示上三角，不是 Host `operatorType_=true` 的反值错误。router 先取 user workspace 并做 null guard，再根据完整 key 解析相应 POD、在栈上创建 `TPipe` 和 Route 类、调用 `Init`/`Process`。Dtype binary 采用 `DtypeByte` 匹配，1/2/4/8-byte 实际类型以等宽有符号整数 carrier 搬运；无算术 cast，保留输入 bit pattern。router 没有 unknown-key `else` 报错，未知 key 会直接结束而不写 `y`。

核心节点无跨 Kernel workspace 边；每个 core 只写自己分配的数据域。Queue 的 `EnQue/DeQue/FreeTensor` 提供 MTE2→V→MTE3 的数据生命周期，显式 event 用于非 Queue 的零 buffer或双 TBuf 搬运。

### Route R-001：全零

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/triu/op_api/aclnn_triu.cpp` | `aclnnInplaceTriuGetWorkspaceSize` | `doTriuGetWorkspaceSize(selfRef,d,selfRef,...)` | 添加共享 Triu 核心图 |
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | `operatorType_ && d>=col` | mode 0 |
| Tiling | 同上 | `OutputZeroAndInputTiling/ComputeTilingKey` | 按 `N,D,A` 分连续元素段；`key=D` | `1/2/4/8` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<false>` | `TILING_KEY_IS(1/2/4/8)` | `SplitAll<T,0>`；未知 key 无输出 |

#### Device 数据流

##### 函数级调用链

```text
doTriuGetWorkspaceSize -> l0op::Triu -> ADD_TO_LAUNCHER_LIST_AICORE
  -> triu(x, y, workspace, tiling)
    -> triangulatorEntry<false>
      -> invokeTemplateSplitAll<T, OUTPUT_ZERO_MODE>
        -> SplitAll::Init
        -> SplitAll::Process
          -> Duplicate(zero)
          -> ProcessPerLoop -> DataCopyPad(UB zero -> y GM)
  -> ViewCopy(triuResult -> selfRef)  # framework boundary
```

##### 源码等价伪码

```text
# 源码等价伪码
B_host = floor_align((UBraw-8192)/2, 32)
B_device = 2*B_host
segment = blockIdx*normalCoreProcessNum
count = tailCoreProcessNum if last_core else normalCoreProcessNum
loops = ceil(count/(B_device/D))
zero(ub[0:B_device/D]); wait V->MTE3
for loop in loops:
    valid = B_device/D except last loop
    DataCopyPad(y[segment + loop*(B_device/D)], ub_zero, valid*D bytes)
PipeBarrier(MTE3)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `y` GM | `blockIdx*normal + loop*normalLoop` elements | 当前 loop `valid*D` bytes | carrier T，连续 | `DataCopyPad` | 下游 ViewCopy | executor 临时输出 | core 段不重叠；尾部精确 byte length |
| `ubBuf1_` | UB allocator | `2*B_host` bytes | `T` | `Duplicate(0)` | 所有 MTE3 loop | 当前 Kernel/core | 单 buffer；全 loop 只读复用 |
| user workspace | framework slice | 16 MiB 请求 | opaque | 未写 | 仅 entry null guard | executor | 不参与数据流 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  zeroBuf["ubBuf1 · depth=N/A TBuf · 2*floor_align(U/2,32) bytes"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*B_host
UB_total <= U
dynamic_ub = N/A
Triu_node_workspace = 16,777,216 bytes（body 未使用）
ACLNN_workspace_total = executor->GetWorkspaceSize()（框架汇总公式固定仓不可见）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `ubBuf1_` 全区 | `Duplicate` 一次 | 每个 `ProcessPerLoop` 的 MTE3 | 不再覆盖 | `V_MTE3` event 在首次 copy 前完成；末尾 `PIPE_MTE3` barrier | 所有 loop 只读零值，直到最终 DMA 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 置零→输出 | `SetFlag/WaitFlag<HardEvent::V_MTE3>` | `Duplicate` | 第一次 `DataCopyPad` 前 | zero TBuf | `T(0)` 按 carrier 宽度生成全零 bit pattern |
| 多次输出 | `PipeBarrier<PIPE_MTE3>` | 各 loop DMA | Kernel 结束 | GM 写完成 | 最后 loop 用 `tailLoopBlockNum*D` bytes，不写段外 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode/DoTiling` | `d>=C` 优先选择 mode 0，按元素连续分核 |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<false>` | keys `1/2/4/8` 映射等宽 `SplitAll<T,0>` |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_all.h` | `SplitAll::Process/ProcessPerLoop` | UB 一次清零，按有效 byte length 反复写 GM |

### Route R-002：全拷贝

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | `d<C && d<=1-row` | mode 1 |
| Tiling | 同上 | `OutputZeroAndInputTiling/ComputeTilingKey` | 同 R-001 分段，`key=10+D` | `11/12/14/18` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<false>` | key 命中 | `SplitAll<T,OUTPUT_INPUT_MODE>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Triu -> triu -> triangulatorEntry<false>
  -> invokeTemplateSplitAll<T, OUTPUT_INPUT_MODE>
    -> SplitAll::Init(two TBuf)
    -> SplitAll::Process
      -> ping/pong ProcessPerLoop
        -> DataCopyPad(x GM -> selected TBuf)
        -> DataCopyPad(selected TBuf -> y GM)
-> ViewCopy(y -> selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码
B = floor_align(U/2,32); capacity=B/D
for loop in ceil(coreCount/capacity):
    buf,event = ping ? (buf1,event1) : (buf2,event2)
    wait MTE3->MTE2(event)
    DataCopyPad(buf, x[coreBase+loop*capacity], valid*D bytes)
    signal+wait MTE2->MTE3(event)
    DataCopyPad(y[coreBase+loop*capacity], buf, valid*D bytes)
    signal MTE3->MTE2(event)
wait both MTE3->MTE2 events before return
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `x/y` GM | `blockIdx*normal + loop*capacity` | `valid*D` bytes | carrier T，连续 | x caller temp / MTE3 | MTE2 / ViewCopy | executor | core/loop 段不重叠 |
| `ubBuf1_` | allocator | `B` bytes | T | 偶数轮 MTE2 | 偶数轮 MTE3 | Kernel/core | 与 `eventFirst` 绑定 |
| `ubBuf2_` | allocator | `B` bytes | T | 奇数轮 MTE2 | 奇数轮 MTE3 | Kernel/core | 与 `eventSecond` 绑定 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  ping["ubBuf1 · TBuf · B bytes"]:1
  pong["ubBuf2 · TBuf · B bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = B + B = 2*floor_align(U/2,32) <= U
dynamic_ub = N/A
Triu_node_workspace = 16,777,216 bytes（body 未使用）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ping/pong TBuf | 当前轮 MTE2 | 当前轮 MTE3 | 同奇偶性的后两轮 MTE2 | `MTE3_MTE2` event 必须先被 wait | 相同 buffer 的前一轮 GM 输出完成后才可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入→输出 | `MTE2_MTE3` event | input DMA | output DMA | 当前 TBuf 有效数据 | 纯 byte-width copy，无 cast/round |
| 输出→复用 | `MTE3_MTE2` event | output DMA | 后续 input DMA | ping/pong TBuf | 最后一轮 copyParams 使用 `tailLoopBlockNum*D` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 极负 diagonal 选择全拷贝 |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | keys `11/12/14/18` | 选择 `SplitAll<T,1>` |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_all.h` | `ProcessPerLoop` | 两个 TBuf 与两组 event 交替搬运 |

### Route R-003：tiny 矩阵

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 非全零/全拷贝且 `R*C*D<=V` | mode 3 |
| Tiling | 同上 | `ComputeTinyInfo/ComputeMask` | `B=floor_align(U/4,32)`；按前导矩阵 group；保留位条件 `c>=r+d` | tiny POD，key `30+D` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | keys `31/32/34/38` | DtypeByte key | `SplitTinyShape<T>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Triu -> triu -> triangulatorEntry<false>
  -> invokeTemplateSplitTinyShape<T>
    -> SplitTinyShape::Init(mask TBuf + depth-2 in/out queues)
    -> Process
      -> ComputeBlockInfo -> CopyIn
      -> DoVectorCopy(Host mask -> MaskReg -> zero then masked source lanes)
      -> CopyOut
-> ViewCopy(y -> selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码
mask256 = Host bits: bit(i*C+j)=1 iff j>=i+d
for assigned group g:
    h = highTail if g is last else highInner
    GM base = g*highInner*R*C
    DMA h blocks, each blockLen=R*C*D, dstStride=V/32-ceil(blockLen/32)
    expand mask by D: byte mask unchanged; 2-byte DIST_US; 4/8-byte repeated MaskUnPack
    for each matrix in group:
        load one VReg from padded UB
        write zero to all VReg lanes
        overwrite mask-selected lanes with original carrier bits
    DMA h valid matrices back, reversing padded stride
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM group | `g*highInner*R*C` elements | `highActual*R*C*D` bytes | consecutive matrices | x / CopyOut | CopyIn / ViewCopy | executor | tail group uses `highTail` |
| `inQueue_` | queue slots | depth `2`, each `B` bytes | each matrix padded to V bytes | `CopyIn` | `DoVectorCopy` | Kernel/core | EnQue/DeQue/FreeTensor |
| `outQueue_` | queue slots | depth `2`, each `B` bytes | same padded layout | vector body | `CopyOut` | Kernel/core | double buffered |
| `maskBuf` | UB TBuf | 32 bytes | four `uint64_t` | `Init` copies POD mask | MicroAPI mask load | Kernel/core | read-only after Init |
| RegTensor/MaskReg | registers | one source/dest/zero carrier VReg + masks；物理 bytes 固定仓不可见 | T lanes | MicroAPI | MicroAPI | VEC scope | 每矩阵复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  mask["maskBuf · 32 bytes"]:1
  inQ["inQueue · depth=2 · 2*B bytes"]:2
  outQ["outQueue · depth=2 · 2*B bytes"]:2
```

##### 字节账与复用顺序

```text
B = floor_align(U/4,32)
UB_total = 2*B + 2*B + 32 = 4*B + 32
4*B <= U；额外 32 bytes 由 Host 预留的 8192-byte 区间覆盖，故总量 <= availableUBSize
dynamic_ub = N/A
workspace_total(core node) = 16 MiB user workspace（body 未使用）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | `CopyIn` | `DoVectorCopy` | `FreeTensor` 后后续 group | queue EnQue/DeQue/FreeTensor | vector 读完才释放 |
| outQueue slot | `DoVectorCopy` | `CopyOut` | `FreeTensor` 后后续 group | queue EnQue/DeQue/FreeTensor | MTE3 完成后释放 |
| maskBuf | `Init` SetValue | 所有 group | 无覆盖 | 初始化后只读 | POD mask 在 Kernel 生命周期内稳定 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→V | inQueue | `CopyIn` | `DeQue` | padded input slot | `blockLen=planeArea*D`，无效 VReg lane 不来自 GM |
| mask compute | MaskReg + `DataCopy/MaskUnPack` | Host `highMask` | masked register store | 每个 dtype lane | 纯 bit selection，无数值 cast；先全零再覆盖保留 lane |
| V→MTE3 | outQueue | vector body | `DeQue` in `CopyOut` | output slot | 只回写 `planeArea*D` 有效 bytes；last group=`highTail` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTinyInfo/ComputeMask` | Host 预计算二维保留 mask 并按 F 分 group |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitTinyShape<T>` | tiny POD 到 body |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_tiny_shape.h` | `Init/DoVectorCopy` | mask 按 D 扩展，一 VReg 处理一矩阵 |

### Route R-004：medium 整平面

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 前三 Route 不命中且 `R*ceil_align(C*D,V)*4<=U` | mode 4 |
| Tiling | 同上 | `ComputeMediumInfo` | `B=floor_align(U/4,32)`，求 `highInner/highOuter/highTail` 与 head/mid/tail rows | medium POD，key `40+D` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | keys `41/42/44/48` | `IS_LOWER=false` | `SplitMediumShape<T,false>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Triu -> triu -> triangulatorEntry<false>
  -> invokeTemplateSplitMediumShape<T,false>
    -> SplitMediumShape::Init(depth-2 in/out queues, DMA descriptors)
    -> Process
      -> ComputeBlockInfo -> CopyIn(full matrix groups)
      -> DoVectorCopy(head/mid/tail row loops)
      -> CopyOut
-> ViewCopy(y -> selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码
alignCount = (C <= V/D) ? 32/D : V/D
rowPad = ceil_align(C, alignCount)
for group g assigned to core:
    h = last ? highTail : highInner
    CopyIn blockCount=h*R, blockLen=C*D, srcStride=0,
           dstStride=rowPad*D/32-ceil(C*D/32)
    for each of h matrices:
        emit headRows full rows (Triu: original)
        for i in [0,midRows):
            copyEleNum=copyCols+i
            for each VReg chunk: load source; Select(zero,source,UpdateMask(copyEleNum)); store
        emit tailRows full rows (Triu: zero)
    CopyOut reverses UB row padding with blockCount=h*R
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM group | `g*(R*C*highInner)` elements | `highActual*R*C*D` | dense matrices | x / MTE3 | MTE2 / ViewCopy | executor | last group `highTail` |
| `inQueue_` | allocator | depth 2 × `B` bytes | rows padded to `rowPad` | `CopyIn` | `DoVectorCopy` | Kernel/core | queue lifecycle |
| `outQueue_` | allocator | depth 2 × `B` bytes | same | vector body | `CopyOut` | Kernel/core | queue lifecycle |
| RegTensor/MaskReg | registers | zero/src/dst + masks；物理 bytes 固定仓不可见 | carrier T | vector load/duplicate | select/store | VEC scope | 行/矩阵循环复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 2*B bytes"]:2
  outQ["outQueue · depth=2 · 2*B bytes"]:2
```

##### 字节账与复用顺序

```text
B = floor_align(U/4,32)
UB_total = 2*B + 2*B = 4*B <= U
dynamic_ub = N/A
workspace_total(core node) = 16 MiB user workspace（body 未使用）
=> highInner=floor((B/D)/(R*ceil_align(C,tmp)))，至少由 Route guard 保证整平面可容纳
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | group `CopyIn` | `DoVectorCopy` | `FreeTensor` 后 | EnQue/DeQue/FreeTensor | pipeline 先预取 next group，但 depth=2 隔离当前 group |
| outQueue slot | `DoVectorCopy` | group `CopyOut` | `FreeTensor` 后 | EnQue/DeQue/FreeTensor | 输出 DMA 完成后才能复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→V→MTE3 | depth-2 queues | CopyIn / vector | DeQue / CopyOut | group UB slot | carrier bit select，无 cast/round |
| row tail | `DataCopyPad` stride | GM input | GM output | 每行有效 C elements | UB padding lane不回写；last high group用 `highTail` |
| diagonal | `UpdateMask(copyCols+i)` + `Select` | Host row partition fields | vector output | 中间行边界 | Triu 路线以 `IS_LOWER=false` 选择零/源；固定源码按该 mask count 执行 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeMediumInfo` | 整平面 UB guard、前导维 group、三段行数 |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitMediumShape<T,false>` | keys 到上三角模板 |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_medium_shape.h` | `CopyIn/DoVectorCopy/CopyOut` | 整 group DMA、按行 MicroAPI 选择、反向去 padding |

### Route R-005：normal 二维分块

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 全零/全拷贝/tiny/medium 均不命中 | mode 2 |
| Tiling | 同上 | `ComputeNormalInfo` | `B=floor_align(U/8,32)`；依次执行 small-col、small-row、normal 子分支 | `rowInner/colInner`，key `20+D` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | keys `21/22/24/28` | `IS_LOWER=false` | `SplitRowAndCol<T,false>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Triu -> triu -> triangulatorEntry<false>
  -> invokeTemplateSplitRowAndCol<T,false>
    -> SplitRowAndCol::Init(three depth-2 queues + zero TBuf)
    -> Process
      -> ComputeBlockInfo -> CopyIn
      -> DoCopyVector(only ON_LINE blocks)
      -> CopyOut(all-zero / all-copy / ON_LINE)
-> ViewCopy(y -> selfRef)
```

##### 源码等价伪码

```text
# 源码等价伪码
for assigned baseBlock b:
    matrix = b/(rowOuter*colOuter)
    rowTile = (b%baseBlocksPerMatrix)/colOuter
    colTile = b%colOuter
    actual rows/cols = inner or tail
    head = colStart-(rowStart+d)
    tail = actualRows-actualCols-head
    position = upper_right if head>=actualRows
               lower_left if tail>=actualRows
               on_line otherwise
    GM offset = matrix*R*C + rowStart*C + colStart
    if Triu lower_left:
        skip CopyIn; CopyOut from pre-zeroed zeroBuf
    else if Triu upper_right:
        CopyIn to inOutQueue; CopyOut same bits without vector compute
    else:
        CopyIn to inQueue
        split rows into head/mid/tail; for Triu keep upper-right rows, zero lower-left rows
        middle row i uses UpdateMask(copyCols+i) and Select(zero,source,...)
        enqueue outQueue and CopyOut
    DMA uses blockCount=actualRows, blockLen=actualCols*D,
    GM stride=(C-actualCols)*D and UB stride based on colInner padding
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM tile | `matrix*R*C + rowTile*rowInner*C + colTile*colInner` | `rowActual × colActual × D` valid bytes with GM row stride | 2D dense source window | x / CopyOut | CopyIn / ViewCopy | executor | tiles write disjoint logical regions |
| `inQueue_` | allocator | depth 2 × `B` | ON_LINE input tile | CopyIn | vector body | Kernel/core | only ON_LINE allocated |
| `outQueue_` | allocator | depth 2 × `B` | ON_LINE output tile | vector body | CopyOut | Kernel/core | only ON_LINE allocated |
| `inOutQueue_` | allocator | depth 2 × `B` | all-copy tile | CopyIn | CopyOut | Kernel/core | bypasses vector materialization |
| `zeroBuf_` | allocator | `B` | all-zero tile source | one `Duplicate` | lower-left CopyOut | Kernel/core | read-only after V→MTE3 event |
| RegTensor/MaskReg | registers | zero/src/dst + row/select masks；物理 bytes 固定仓不可见 | carrier T | MicroAPI | MicroAPI | VEC scope | ON_LINE only |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  inQ["inQueue · depth=2 · 2*B"]:2
  outQ["outQueue · depth=2 · 2*B"]:2
  ioQ["inOutQueue · depth=2 · 2*B"]:2
  zero["zeroBuf · B"]:1
```

##### 字节账与复用顺序

```text
B = floor_align(U/8,32)
UB_total = 2*B + 2*B + 2*B + B = 7*B
7*B <= U；Host 按 8*B 预算，保留至少一个 B 级余量（另有全局 8192-byte reserve）
dynamic_ub = N/A
workspace_total(core node) = 16 MiB user workspace（body 未使用）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | ON_LINE CopyIn | `DoCopyVector` | FreeTensor 后 | queue 生命周期 | 预取 next block 由 depth=2 隔离 |
| outQueue slot | `DoCopyVector` | ON_LINE CopyOut | FreeTensor 后 | queue 生命周期 | MTE3 完成后释放 |
| inOutQueue slot | all-copy CopyIn | all-copy CopyOut | FreeTensor 后 | queue 生命周期 | 不经 VEC，仍保持 MTE2→MTE3 次序 |
| zeroBuf | `Duplicate` once | 任意 all-zero block CopyOut | 无覆盖 | `V_MTE3` flag/wait 在 block 循环前完成 | Kernel 期间只读 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| zero 初始化 | `FetchEventID(V_MTE3)` + flag/wait | `Duplicate` | block CopyOut | zeroBuf | carrier 全零 |
| tile pipeline | depth-2 queues | CopyIn/vector | vector/CopyOut | 三种 tile 队列 | 纯 bit copy/select；无浮点计算 |
| DMA tail | `DataCopyExtParams` | `rowActual/colActual` | GM/UB | 最后一行/列 tile | blockLen 只含有效列；row/col tail 不写邻接 tile |
| VReg tail | `rowMask=UpdateMask(regCount)` | Tiling `colInner`/dtype | register store | 每个 VReg chunk | UB padding由 rowMask 屏蔽，GM 输出再按 `colActual` 截断 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeNormalInfo` | 三分支计算 row/col inner、outer、tail 与 base blocks |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitRowAndCol<T,false>` | normal keys 到上三角 body |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_row_and_col.h` | `ComputeBlockInfo/CopyIn/DoCopyVector/CopyOut` | tile 区域分类、跳过无用输入、直接 copy、对角 tile 掩码 |

## 算子特定优化

### Trick T-001：按 dtype 字节宽度复用 bitwise Kernel

#### 触发与源码依据

- 适用 Route：R-001～R-005。
- 触发条件：950 binary config 以 `DtypeByte` 匹配，Tiling key 个位为 `D∈{1,2,4,8}`。
- Golden 来源：[`conversion/triu/op_kernel/arch35/triangulator_base.h:98-117`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/triangulator_base.h#L98-L117)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// 个位：位宽  1 int8_t    2 int16_t    4 int32_t   8  int64_t
// 十位：模板  0 output_zero  1 output_input    2 output_normal    3 tiny_shape 4 split_outer
if (TILING_KEY_IS(OUTPUT_ZERO_INT8)) {
    invokeTemplateSplitAll<int8_t, OUTPUT_ZERO_MODE>(x, y, tiling);
} else if (TILING_KEY_IS(OUTPUT_ZERO_INT16)) {
    invokeTemplateSplitAll<int16_t, OUTPUT_ZERO_MODE>(x, y, tiling);
} else if (TILING_KEY_IS(OUTPUT_ZERO_INT32)) {
    invokeTemplateSplitAll<int32_t, OUTPUT_ZERO_MODE>(x, y, tiling);
} else if (TILING_KEY_IS(OUTPUT_ZERO_INT64)) {
    invokeTemplateSplitAll<int64_t, OUTPUT_ZERO_MODE>(x, y, tiling);
}
```

#### 机制与设计原因

Triu 只需复制原 bit pattern 或写全零，不依赖数值解释。binary config 对输入输出启用 `DtypeByte`，Host 用 `dtypeBytes` 生成 key，router 用等宽整数实例完成所有 DMA、zero 和 register select。因此 FLOAT/BFLOAT/COMPLEX/整数/BOOL 可共享 1/2/4/8-byte body，减少按语义 dtype 复制的模板数量；该结论仅是源码级实现复用，不声称实测性能百分比。

#### 迁移边界

- 适用：只做 bitwise copy/zero/select、且全零 bit pattern等于数值零的算子。
- 前提：输入输出等宽同 dtype；binary dtype-byte 匹配、Host key 与 router carrier 宽度一致。
- 失效：需要数值比较、类型转换、NaN 语义或非全零编码时，必须按真实 dtype 拆模板。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但 router 对未知字节宽度没有 fallback，迁移时不得只扩 API dtype 而不同时扩 binary/key/router。

### Trick T-002：Host 对角线极值直接退化为纯搬运

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：Triu 的 `d>=C` 或 `d<=1-R`。
- Golden 来源：[`conversion/triu/op_host/arch35/triangulator_tiling.cpp:198-216`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.cpp#L198-L216)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if ((!operatorType_ && baseInfoOp.diagOffset <= -baseInfoOp.row) ||
    (operatorType_ && baseInfoOp.diagOffset >= baseInfoOp.col)) {
    splitCoreOp.tilingMode = OUTPUT_ZERO_MODE;
} else if (
    (!operatorType_ && baseInfoOp.diagOffset >= baseInfoOp.col - 1) ||
    (operatorType_ && baseInfoOp.diagOffset <= 1 - baseInfoOp.row)) {
    splitCoreOp.tilingMode = OUTPUT_INPUT_MODE;
}
```

#### 机制与设计原因

Host 在 shape 已知时证明整张量全零或全保留，把一般二维 mask 计算替换成连续 1D 分核。R-001 不搬入 x，只复用预置零 UB；R-002 用两个 TBuf ping-pong 做纯 GM→UB→GM copy。`OutputZeroAndInputTiling` 还令每核正常段不少于约 `1024/D` elements，避免把大张量切成过小搬运段。

#### 迁移边界

- 适用：属性极值可让逐元素谓词恒真或恒假的结构化 mask 算子。
- 前提：边界公式必须基于最后两维的真实 `R/C`，且空域已在 ACLNN 层提前结束。
- 失效：`1-R < d < C` 时必须进入 tiny/medium/normal，不能用全拷贝/全零覆盖部分区域。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时若交换 `>=C` 与 `<=1-R` 的 Triu 边界，会在包含对角线的临界值产生整张量错误。

### Trick T-003：normal Route 按 tile 与对角线位置避免无效 CopyIn/Vector

#### 触发与源码依据

- 适用 Route：R-005。
- 触发条件：normal 二维 tile 被 `ComputeBlockInfo` 判为全零 lower-left、全拷贝 upper-right 或跨线 ON_LINE。
- Golden 来源：[`conversion/triu/op_kernel/arch35/split_row_and_col.h:199-220`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_row_and_col.h#L199-L220)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (info.position == UPPER_RIGHT && IS_LOWER || info.position == LOWER_LEFT && !IS_LOWER) {
    return;
}

if (info.position == ON_LINE) {
    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam, copyInPadParam);
    inQueue_.EnQue(xLocal);
} else {
    LocalTensor<T> xLocal = inOutQueue_.AllocTensor<T>();
    AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam, copyInPadParam);
    inOutQueue_.EnQue(xLocal);
}
```

#### 机制与设计原因

对 Triu (`IS_LOWER=false`)，全零的 lower-left tile 直接跳过 GM input DMA，CopyOut 从一次性清零的 `zeroBuf_` 取数据；全保留的 upper-right tile 使用绑定的 `inOutQueue_` 从 CopyIn 直达 CopyOut；只有跨对角线 tile 才进入 MicroAPI `Select`。这同时减少不必要的 GM 读取与 Vector 指令，并通过三个独立 queue 保持 pipeline 生命周期。

#### 迁移边界

- 适用：规则边界能把二维 tile 分类为恒零、恒拷贝和边界 tile 的 mask/triangular 算子。
- 前提：`gmOffset/headRows/tailRows` 与 tile tail 完整闭合；全零 buffer 的 V→MTE3 同步必须先完成。
- 失效：非线性或数据依赖 mask 无法由 tile 坐标证明恒值，必须每 tile 计算真实 mask。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若在 lower-left tile 仍从输入 CopyIn，语义虽可保持但丢失该源码可证明的搬运削减，且不得错误复用 `inOutQueue_` 代替零源。

## 模板与实现对比

| 对比维度 | R-001 全零 | R-002 全拷贝 | R-003 tiny | R-004 medium | R-005 normal | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | `d>=C`，最高 | `d<=1-R`，第二 | 矩阵 bytes≤V | 整平面 4-buffer≤U | 最终 fallback | 先判语义退化，再按 VReg/UB 容量扩大处理粒度 |
| 模板与 dtype | `SplitAll<T,0>` | `SplitAll<T,1>` | `SplitTinyShape<T>` | `SplitMediumShape<T,false>` | `SplitRowAndCol<T,false>` | 全部按 1/2/4/8-byte carrier |
| 循环与分核 | 连续元素 loop | 连续元素 ping-pong | 前导矩阵 group | 前导矩阵 group × 行 | matrix×rowTile×colTile | 容量越大，单 block 覆盖粒度越大 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 1 TBuf | 2 TBuf | in/out depth2 + mask + regs | in/out depth2 + regs | 三组 depth2 + zero + regs | 全部 dynamic UB=N/A，无通信 window |
| 内存总占用与复用 | `2B` | `2B` | `4B+32` | `4B` | `7B`，Host 按 `8B` 预算 | `B` 由 U 的 1/2、1/4、1/8 分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Duplicate + MTE3 | MTE2/MTE3 | DataCopyPad + mask MicroAPI | 整平面 DataCopyPad + Select | 2D DataCopyPad + tile Select/bypass | 均为 AIV；无 SIMT/NDDMA/通信节点 |
| 同步、精度与 tail | V→MTE3 event，loop tail | 双 event，loop tail | queue，highTail | queue，highTail/row pad | queue+event，row/col tail | bitwise，无 cast/round；有效 byte/mask 控 tail |
| fallback/失败边界 | 无 | 无 | 资源不满足在 Host 转 R-004 | 资源不满足在 Host 转 R-005 | unknown key 无输出；Tiling capacity失败 | Device router 不进行跨 Route fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档声明支持；固定 commit 没有 `ascend910b` 活动构建/OpDef/binary 闭包，故不生成正式卡 | 公开文档声明支持；固定 commit 没有 `ascend910_93` 活动构建/OpDef/binary 闭包，故不生成正式卡 | 公开文档支持，`AddConfig("ascend950")`、binary config、CMake 与 AIV body 闭合 |
| Host/Launcher | ACLNN API 校验分支可见，但 A2 后端 Host/Tiling producer 固定仓不可见 | ACLNN API 校验分支可见，但 A3 后端 Host/Tiling producer 固定仓不可见 | `arch35` Tiling，AIV core/UB/VReg 驱动五 Route |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | `triu_apt` AIV-only；五 Route；16 MiB 核心 user workspace 声明 |

## 限制与失败边界

- 公开 Inplace 接口限定 2～8 维和非私有 storage format；代码对 empty tensor 在 rank 下限检查上更宽松并直接返回，但本卡不把内部宽松行为扩成公开 0-D 支持。
- 950 dtype 由 ACLNN `IsRegBase()` 列表和 OpDef/binary 的 1/2/4/8-byte 匹配共同限制；扩 dtype 必须同时闭合 API、OpDef、binary 和 router。
- `F*R*C`、`R*C*D` 等均在 Host `int64_t` 中相乘；固定仓未见显式乘法溢出检查。超大 shape 的溢出行为不可推测。
- `TriangulatorTilingData` 等结构的宏展开、精确 `sizeof/alignof/offsetof` 固定仓不可见；只能复用声明字段顺序、类型与 producer/consumer，不能手写假定 packing。
- `Contiguous`、`ViewCopy`、executor workspace 汇总、stream/event 与 cleanup 的框架实现固定仓不可见；它们是公开 ACLNN 到算子核心的明确框架边界。
- router 对 user workspace null 或未知 key 都直接 return，未在 Device 侧产生显式错误码；Host 必须保证 `PostTiling` 与 workspace 分配成功。
- A2/A3 虽有公开产品支持声明，但固定 commit 只活动接线 `ascend950` 与 `mc62`，缺少 A2/A3 核心闭包；不能用 `mc62` 推测为 A2/A3。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/triu/docs/aclnnTriu&aclnnInplaceTriu.md` | `aclnnInplaceTriu*` 文档 | 公开性、950 支持、原地语义、dtype/rank、确定性 |
| 入口声明/约束 | `conversion/triu/op_api/aclnn_triu.h` | `aclnnInplaceTriuGetWorkspaceSize/aclnnInplaceTriu` | C ABI 与两段式参数 |
| 入口实现 | `conversion/triu/op_api/aclnn_triu.cpp` | `CheckParams/doTriuGetWorkspaceSize/aclnnInplaceTriu*` | 校验、empty、Contiguous→Triu→ViewCopy、executor |
| Launcher | `conversion/triu/op_api/triu.cpp` | `l0op::Triu` | 临时输出分配和 AICORE launcher 添加 |
| Host/Tiling | `conversion/triu/op_host/arch35/triu_tiling.cpp` | `TilingPrepare4Trilu/Tiling4Trilu` | AIV core/UB producer、shape/dtype/diagonal producer |
| Host/Tiling | `conversion/triu/op_host/arch35/triu_tiling.h` | `Tiling4Trilu` | `F/R/C/D` 推导与外层调用 |
| POD | `conversion/triu/op_host/arch35/triangulator_tiling.h` | 四个 `Triangulator*TilingData` | 字段声明、类型、注册 key |
| Host/Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode/*Info/PostTiling` | 决策树、资源公式、POD producer、blockDim/workspace/key |
| Kernel entry | `conversion/triu/op_kernel/triu_apt.cpp` | `triu` | `KERNEL_TYPE_AIV_ONLY` 与入口参数 |
| Kernel router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry/invokeTemplate*` | 20 keys、POD 解析、模板映射、workspace guard |
| Kernel body | `conversion/triu/op_kernel/arch35/split_all.h` | `SplitAll` | 全零/全拷贝 body、TBuf/event/tail |
| Kernel body | `conversion/triu/op_kernel/arch35/split_tiny_shape.h` | `SplitTinyShape` | Host mask、VReg lane、group DMA |
| Kernel body | `conversion/triu/op_kernel/arch35/split_medium_shape.h` | `SplitMediumShape` | 整平面 queue、行掩码、high tail |
| Kernel body | `conversion/triu/op_kernel/arch35/split_row_and_col.h` | `SplitRowAndCol` | 2D tile、区域分类、三 queue、row/col tail |
| 构建接线 | `conversion/triu/op_host/triu_def.cpp` | `ops::Triu` | 950 dtype/format、`ascend950`、`triu_apt` |
| 构建接线 | `conversion/triu/op_host/config/ascend950/triu_binary.json` | `Triu` binary entries | 950 四种 dtype-byte binary |
| 构建接线 | `conversion/triu/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | 活动 `ascend950/mc62` 与 `arch35` Tiling |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `Triu` | `ascend950/mc62` AIV binary 编译配置 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 四种 Route POD 的字段类型、声明顺序和逐字段语义 | 宏 packing 固定仓不可见；不能假设 offset；key 必须选择匹配 POD | `triangulator_tiling.h`；`SaveToTilingData` |
| 调用顺序 | `Contiguous → Triu temporary → ViewCopy(selfRef)` | 不得让核心 y 与 selfRef 直接别名而破坏读后写；任一 null 阻断后续 | `doTriuGetWorkspaceSize`；`l0op::Triu` |
| workspace/内存 | 核心声明 16 MiB，五 body 实际只使用 GM/UB | user workspace null guard、UB reserve、queue depth 和 B 公式不得省略 | `PostTiling`；`triangulatorEntry`；各 `InitBuffer` |
| 同步 | TBuf route 用显式 event，queue route用 EnQue/DeQue/FreeTensor | 零 buffer在 MTE3 前完成；ping-pong 覆盖前等待前一 MTE3 | `SplitAll::ProcessPerLoop`；各 Route `Process` |
| 精度与尾块 | DtypeByte carrier 做 bitwise copy/zero/select；high/row/col/loop tail 均显式 | carrier 宽度必须等于 D；DMA 只提交有效 bytes；Triu 使用 `IS_LOWER=false` | `ComputeTilingKey`；`triangulatorEntry`；各 CopyOut |
| 失败边界 | ACLNN 参数错、Tiling 资源/容量错、launcher null 均有 Host 终点 | Device unknown key/user workspace null 是静默 return，不能视为成功输出 | `CheckParams`；`PostTiling`；`triangulatorEntry` |