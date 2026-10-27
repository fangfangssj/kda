---
type: Operator
title: "Expand 950 ACLNN 算子知识"
description: "说明 aclnnExpand 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/docs/aclnnExpand.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/op_api/aclnn_expand.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/op_api/expand.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/op_host/expand_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/op_host/arch35/expand_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/op_kernel/expand_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_kernel/arch35/broadcast_to.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_kernel/arch35/broadcast_to_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_dma.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_copypad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_tailAxis.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_ub.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/op_host/config/ascend950/expand_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T10:55:57Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Expand 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnExpandGetWorkspaceSize` + `aclnnExpand`，固定仓唯一公开版本，入口类型 ACLNN |
| 计算语义 | 把 `self` 右对齐到 `size`，对输入维为 1 的轴复制数据；公开 ACLNN 的有效调用使用具体目标维值 |
| 输入输出 | `self/out` 为 0–8 维 ND，支持 FLOAT16/FLOAT/UINT8/INT8/INT32/INT64/BOOL/BF16；`size` 是 `aclIntArray`，内部转 INT64 Tensor |
| 执行类型 | 空 Tensor 为零 Device 节点；0 维标量直达框架 `ViewCopy`；普通路线为单 AIV Kernel，key 11000/11004 使用 NDDMA，其余 key 使用 AIV_VECTOR/MicroAPI |
| Kernel 路线 | R-001 空 Tensor，R-002 0D 标量直拷，R-003 NDDMA-4D，R-004 UB 广播，R-005 尾轴大 A，R-006 尾轴大 B，R-007 FULL-NDDMA-5D，R-008 尾轴小 A |
| 核心资源 | GM + `TQue/TQueBind` + UB + RegTensor；普通 Expand 节点申请 16 MiB system workspace；NDDMA 路线使用 `MultiCopyParams` |
| 来源闭包 | `source_closed`：公开 ACLNN→参数校验/空域与标量分流→l0 `Expand`→950 OpDef/Tiling/POD→六 key router→共享 BroadcastTo body/helper→binary/CMake 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输出秩为 `n=size.Size()`，输入秩为 `r`，先在输入 shape 左侧补 `n-r` 个 1。公开 ACLNN 的有效目标 shape 使用具体维值；输入维非 1 时目标必须等于输入维。输出坐标到输入坐标的映射为：输入维为 1 时取 0，否则取同一坐标。数据仅复制，不做数值运算或 dtype 转换。

内部图/Tiling 的 `ApplyBroadcastRules` 含 `outDim==-1` 时继承输入维的分支，但公开 ACLNN `CheckShape` 和 l0 `ExpandInfershape` 在临时写入继承值后又无条件写回原始 `-1`，导致合法实体 `out` 无法通过最终 shape 相等检查。因此本卡不把 `-1` 声明为公开 ACLNN 支持语义。

公开 ACLNN 的 `CheckShape` 还规定 `out.viewShape` 必须等于推导结果。普通计算前，Tiling 删除输出长度为 1 的轴，并把相邻、同为“不广播 A”或“广播 B”的轴相乘合并；这只改变 Device 索引表达，不改变逻辑 ND 顺序。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 0–8 维 / ND | FLOAT16/FLOAT/UINT8/INT8/INT32/INT64/BOOL/BF16 | 必选；可为非连续 Tensor；950 支持 BF16 |
| `size` | `aclIntArray` | API 为 INT；l0 固定转换成 INT64 Tensor | 元素数不得小于 `self` 秩；公开文档只要求与输入满足广播关系，固定 ACLNN 实现不能闭合 `-1` 继承语义 |
| `out` | 推导 shape / ND | 与 `self` 相同 | 必选；可为非连续 Tensor；最大 8 维 |
| `workspaceSize` | 标量输出 | `uint64_t` | 第一段返回 executor 全流程 workspace；空 Tensor 明确返回 0 |
| `workspace` | Device 字节区 | 无 dtype | 调用者持有至第二段 stream 执行完成；普通 Expand 核心节点的 system workspace 需求为 16 MiB |

### 入口与版本

固定仓只公开 `aclnnExpand` 两段式接口，无同语义 V2/V3。文档明确支持 Ascend 950PR/950DT；`Expand` OpDef 只为 `ascend950` 注册 `expand_apt` AICore 文件，950 binary config 列出 8 种 `x/y` dtype 与 INT32/INT64 shape 的 16 个实例，平台事实一致。

公开文档也列出 A2/A3，但固定提交的 `Expand` OpDef 和 binary config 没有 `ascend910b/ascend910_93` AICore 接线，且本仓的 AICPU body 不是 VV 执行族；因此这些平台不能在本卡中声明源码闭包，也不由本文件覆盖。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnExpandGetWorkspaceSize` 依次执行：

1. 检查 `workspaceSize/executor` 公共输出参数，以及 `self/size/out` 非空。
2. 读取当前 `NpuArch`；950 `DAV_3510` 允许 8 种 dtype，并要求 `self/out` dtype 相同。
3. 构造 `sizeShape`。若 `selfShape==sizeShape==outShape` 立即通过；否则要求 `sizeRank>=selfRank`，右对齐逐轴检查输入维等于目标或输入维为 1；代码对 `-1` 的临时继承赋值随后被原值覆盖，最终仍要求 `outShape==expectShape`，所以公开有效调用应给出具体目标维。
4. 要求 `self/out` 秩不超过 8；storage format 为 FRACTAL_NZ 只告警，不在此处拒绝。
5. 若 `self.IsEmpty() || out.IsEmpty()`，返回 workspace 0 和空计算 executor，不创建任何 Device 节点。
6. 非空且 `selfRank==0 && size.Size()==0` 时只加入 `l0op::ViewCopy(self,out)`。
7. 其余情况加入 `Contiguous(self)`、`l0op::Expand(selfContiguous,size)`、`ViewCopy(expandOut,out)`；`Expand` 再检查连续性、推 shape、分配中间输出、把 `aclIntArray` 转成 INT64 `shapeTensor`。950 已接受的 dtype 全部命中 `ExpandAiCore`。

### 参数、Workspace 与生命周期

API 第一段创建 `aclOpExecutor`，中间连续输入、INT64 shape Tensor、Expand 输出及最终 ViewCopy 的 workspace/临时 Tensor 都由 executor 管理；其精确合并 offset 和总量公式属于固定仓不可见的框架实现。核心 `Expand` Tiling 固定写 `currentWorkspace[0]=16*1024*1024`，Kernel 入口拒绝空 workspace，然后 `SetSysWorkspace(workspace)`；该空间由 system/NDDMA primitive 使用，算子 body 不自行切 slice。

普通路由的 `TilingPrepare4Expand` 生产 `coreNum=GetCoreNumAiv()`、`ubSize=GetCoreMemSize(UB)`、`clSize=GetCacheLineSize()`、`blockSize=GetUbBlockSize()`、`vRegSize=GetVRegSize()`，任一非正即失败。Tiling 根据这些量和 dtype 字节数生成 POD、`blockDim=usedCoreCnt*dFactor`。executor 转交给调用者后，在第二段 `CommonOpExecutorRun(workspace,workspaceSize,executor,stream)` 上按队列顺序提交；调用者须保持 workspace、输入输出和 executor 有效到 stream 完成。

### 启动图与失败传播

```text
R-001: aclnnExpandGetWorkspaceSize -> empty check -> executor（无节点，workspaceSize=0）

R-002: aclnnExpandGetWorkspaceSize -> ViewCopy(self -> out)（框架节点）

R-003..R-008:
  Contiguous(self -> selfContiguous)             # 框架边界，可能物化连续副本
    -> ExpandAiCore / K-001 expand
         inputs: selfContiguous, INT64 shapeTensor
         output: expandOut
         workspace: system workspace 16 MiB
         blockDim: usedCoreCnt*dFactor
         POD: BroadcastToTilingData
    -> ViewCopy(expandOut -> out)                 # 框架边界，支持非连续 out
```

所有节点由同一 executor 在调用者 stream 上保持顺序；核心输出 `expandOut` 在 ViewCopy 完成前不可释放。任何校验、连续化、shape Tensor 转换、Tiling 或 launcher 添加失败都会返回错误并阻断后续节点。核心 entry 对未知 key 没有 fallback，不写输出；workspace 为空时直接返回。框架节点的内部 Kernel、workspace offset 和事件实现固定仓不可见。

## Route 与参数设计

### 分核与资源计算

Tiling 先把归一化/合并后的轴标成 A（输入输出维相同）或 B（输入维 1、输出维扩张）。选择一个连续后缀 U：`uAxisCnt` 为 U 含有的维数，`uLpUnit` 为首个 U 轴一次处理长度，`uInOffset/uOutOffset` 是该轴后续输入/输出元素乘积；U 前的 A/B 轴分别压成 `aAxisLen/bAxisLen`，并保留逐轴 size 与 trailing stride。

令 `S=sizeof(T)`、`B=blockSize`、`CL=cacheLine`、`U=ubSize`：

```text
raw_max = floor(U/B)*B/S                 # 元素
minTensorSize = CL/S                     # 元素
ubGate = raw_max/4
tensorSize 根据尾轴、广播类型和核利用率计算，再反复 /2 直到 <= maxTensorSize
uLpCnt = ceil(ntcULen/uLpUnit)
blockDim = usedCoreCnt*dFactor
```

`CalcAxisWeight(lp)=ceil(lp/ceil(lp/coreNum))`，若整除还加 `coreNum`；A、B、U 三者最大权重决定 `blockAxis`。普通核处理 `ntc*Len`，最后一核处理 `tc*Len`。核数不足时 `doubleMode=1` 把 B 循环按 `dFactor` 二次分核，`doubleMode=2` 把 U 循环二次分核；Device 用 `realBlockIdx%usedCoreCnt` 定主核、`realBlockIdx/usedCoreCnt` 定二次分片。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 空 Tensor | API Host shortcut | `self.IsEmpty() \|\| out.IsEmpty()` | 检查 0D 标量 |
| 2 | R-002 0D 标量直拷 | framework `ViewCopy` | 非空且 `selfDimNum==0 && size.Size()==0` | 普通 Expand Tiling |
| 3 | R-005 尾轴大 A | 11002 / `BrcToDataCopyPad<T>` | `lastDim>=tensorSize && !lastIsB`，随后 `uAxisCnt!=5` | 检查 11003 |
| 4 | R-006 尾轴大 B | 11003 / `BrcToWithTailAxis<T>` | 前项未中，`lastDim>=tensorSize && lastIsB && ubSizeCheck`，随后 `uAxisCnt!=5` | 检查 UB 条件 |
| 5 | R-004 UB 广播 | 11001 / `BroadcastToUb<T,false>` | 前两项未中；`((dimNum==1&&!lastIsB)\|\|(lastDim<=tensorSize&&((!lastIsB&&!isDMABrcA)\|\|(lastIsB&&lastDim>=8))))&&ubSizeCheck`；随后不满足 11004/11005 覆盖 | R-003 |
| 6 | R-003 NDDMA-4D | 11000 / `BrcToWithNDDMA<T,TD,4>` | 上述三类未中，且计算 U 后 `uAxisCnt!=5` | 无 Host fallback |
| 7 | R-007 FULL-NDDMA-5D | 11004 / `BrcToWithNDDMA<T,TD,5>` | 初选任意 key 后，`uAxisCnt==5` 无条件覆盖 | 无 Host fallback |
| 8 | R-008 尾轴小 A | 11005 / `BroadcastToUb<T,true>` | 初选 11001、`!lastIsB`、`uAxisCnt>1`、`out[d-2]*ceilAlign(lastDim*S,B)<=vRegSize`，且未先被 11004 覆盖 | 保留 11001 |

```text
if empty(self) or empty(out): R-001
else if rank(self)==0 and sizeRank==0: R-002
else:
  if lastDim>=tensorSize and !lastIsB: key=11002
  else if lastDim>=tensorSize and lastIsB and ubSizeCheck: key=11003
  else if UB_BRC_predicate and ubSizeCheck: key=11001
  else: key=11000
  build U suffix
  if uAxisCnt==5: key=11004
  else if key==11001 and last-A small-tail predicate: key=11005
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 空 Tensor | Host shortcut | 950、合法 dtype/shape，任一 Tensor 空 | 零 Device 节点 | API return | workspace 0 | 不写 Device；成功返回 |
| R-002 0D 标量直拷 | Host shortcut | 非空 0D self、size 为空、0D out | 单 framework ViewCopy | 固定仓框架边界 | workspace 由 executor 决定 | 无 Expand key |
| R-003 NDDMA-4D | 11000 | 普通非空，初选 fallback，U 维数 <5 | K-001 `expand` | `BrcToWithNDDMA<DTYPE_X_,BroadcastToTilingData,4>` | 1×UB Queue；NDDMA→MTE3，显式 event | `uLeft` 重设 loopSize[0]/blockLen |
| R-004 UB 广播 | 11001 | UB predicate，U 维数 <5，未命中小尾 A | K-001 | `BroadcastToUb<T,TD,false>` | 双 Queue；MicroAPI；Queue + PIPE_V | `uLeft` 调整首 U 轴与 DMA 长度 |
| R-005 尾轴大 A | 11002 | 尾轴不广播且 `lastDim>=tensorSize`，U 维数 <5 | K-001 | `BrcToDataCopyPad<T,TD>` | 绑定 Queue，纯 DataCopyPad | `uLeft` 独立 blockLen |
| R-006 尾轴大 B | 11003 | 尾轴广播、足够 UB、`lastDim>=tensorSize`，U 维数 <5 | K-001 | `BrcToWithTailAxis<T,TD>` | 输入/输出 Queue + RegTensor Duplicate | A 块尾与 U 块尾均单独提交 |
| R-007 FULL-NDDMA-5D | 11004 | `uAxisCnt==5` | K-001 | `BrcToWithNDDMA<T,TD,5>` | 与 R-003 相同，五层 loopInfo | `uLeft` 覆盖第 0 维 |
| R-008 尾轴小 A | 11005 | UB 初选、末轴 A、U>1、末两轴对齐字节不超 VReg | K-001 | `BroadcastToUb<T,TD,true>` | 双 Queue；DIST_BLK 或 Gather | 末轴不齐 32 B 时分 blockCount |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 非空核心 | `x/y` 任一维为 0 时 API 已短路；Tiling 自身拒绝空 | 避免零除、零核与无效 DMA | 全部核心 Route | R-001 或 Tiling 失败 | `aclnnExpandGetWorkspaceSize`、`IsEmptyTensor` |
| 最大秩 | 输出秩 `<=8` | POD 的 A/B 参数数组仅容纳 8 轴 | 全部 | 参数/Tiling 失败 | `CheckMaxDimension`、`GetShapeInfo` |
| UB 四分之一 | `tensorSize <= floor(U/B)*B/S/4` | 最坏两路 Queue×双 buffer 需要 `4*tensorSize*S` | R-004/R-006/R-008 | R-003，或 R-005 | `CalcTilingKey::ubSizeCheck` |
| NDDMA 最大维 | `uAxisCnt==5` | 默认 NDDMA 模板只有 4 维参数 | R-003/R-007 | 5 维改 R-007 | `GetUAxisInfo`、`UpdateTilingKey` |
| 小尾 A VReg | `out[d-2]*ceilAlign(lastDim*S,B)<=vRegSize` | 整个末两轴广播片必须能由 RegTensor 路线覆盖 | R-008 | R-004 | `UpdateTilingKey` |
| 8 元素尾轴门槛 | B 尾轴要求 `lastDim>=8` 才走普通 UB；短 A 尾形状令 `isDMABrcA` | 控制 MicroAPI 广播与 NDDMA 的适用形态 | R-003/R-004 | 互相切换 | `CalcTensorSize`、`CalcTilingKey` |
| 16 MiB workspace | `context->GetWorkspaceSizes(1)[0]=16MiB` 且 entry `workspace!=nullptr` | `SetSysWorkspace` 为 system/NDDMA primitive 建立工作区 | R-003..R-008 | 空指针时 Device 直接返回 | `WriteTilingData`、`broadcast_to_impl` |
| 64 位重解释 | `sizeof(T)==8` 时以两个 32 位 lane interleave/copy | MicroAPI 广播原语按 32 位 lane 操作 | R-004/R-006/R-008 | 同 Route 特化 | `DTYPE_X_`、`VFBroadcastOneElemLBB64` |

### POD/ABI

`BroadcastToTilingData` 由注册宏生成序列化结构；固定仓可见字段顺序和源码类型，但宏展开后的 `sizeof/alignof/offsetof` 不可见。数组字段逐元素同类型、连续序列化。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastToTilingData` | `tilingKey` | int64_t/64 | offset 固定仓不可见 | key | `CalcTilingKey/UpdateTilingKey` | router 宏；POD 内副本仅诊断 | 11000–11005 |
| 同上 | `dFactor` | int64_t/64 | 同上 | 二次分核因子 | `SetBlockCnt` | `CalcAxesLoopInfo` | `>=1` |
| 同上 | `doubleMode` | uint8_t/8 | 同上 | 枚举 | `SetBlockCnt` | `CalcAxesLoopInfo` | 0/1/2 |
| 同上 | `uAxisCnt` | uint8_t/8 | 同上 | 轴数 | `GetUAxisInfo` | `CalcInnerShape`；router key 更新 | 1–5 |
| 同上 | `bufferCnt` | uint8_t/8 | 同上 | 每 Queue buffer 数 | `CalcDBMode` | UB/tail `InitBuffer` | 1 或 2 |
| 同上 | `blockAxis` | uint8_t/8 | 同上 | A/B/U 枚举 | `GetMCTilingInfo` | base offset/loop | 0/1/2 |
| 同上 | `tensorSize` | uint32_t/32 | 同上 | T 元素 | `CalcTensorSize` 后窄化 | 全部 `InitBuffer`/tile | Host 未显式检查 uint32_t 窄化；上限受 UB/MAX_TENSOR_SIZE |
| 同上 | `usedCoreCnt` | int64_t/64 | 同上 | 主核数 | `GetMCTilingInfo` | `blockIdx%usedCoreCnt` | 必须非 0 |
| 同上 | `ntcALen` | int64_t/64 | 同上 | A 循环 | `GetMCTilingInfo` | 非尾主核 A loop | 正值 |
| 同上 | `tcALen` | int64_t/64 | 同上 | A 循环 | 同上 | 尾主核 A loop | 正值 |
| 同上 | `ntcBLen` | int64_t/64 | 同上 | B 循环 | 同上 | 非尾主核 B loop | 正值 |
| 同上 | `tcBLen` | int64_t/64 | 同上 | B 循环 | 同上 | 尾主核 B loop | 正值 |
| 同上 | `ntcULen` | int64_t/64 | 同上 | 首 U 轴元素 | 同上 | 非尾主核 U loop | 正值 |
| 同上 | `tcULen` | int64_t/64 | 同上 | 首 U 轴元素 | 同上 | 尾主核 U loop | 正值 |
| 同上 | `aLpUnit` | int64_t/64 | 同上 | A 元素/批 | `min(ntcALen,tensorSize)` | R-006 A 批 | `>=1` |
| 同上 | `uLpUnit` | int64_t/64 | 同上 | 首 U 轴元素/tile | `GetUAxisInfo` | 全部 tile 与地址式 | `>=1` |
| 同上 | `uInOffset` | int64_t/64 | 同上 | 输入 T 元素 stride | trailing input product | GM input offset | `isUNotB=0` 时乘零 |
| 同上 | `uOutOffset` | int64_t/64 | 同上 | 输出 T 元素 stride | trailing output product | GM output offset | 正值 |
| 同上 | `isUNotB` | int32_t/32 | 同上 | 布尔 | `!abInfo[uAxis]` | 输入偏移与 NDDMA copy cache | 0/1 |
| 同上 | `isLastDimB` | int32_t/32 | 同上 | 布尔 | `abInfo[last]` | UB body 分支 | 0/1 |
| 同上 | `aAxesNum` | int32_t/32 | 同上 | 轴数 | `GetABAxesParams` | A offset loop | 0–8 |
| 同上 | `bAxesNum` | int32_t/32 | 同上 | 轴数 | 同上 | B offset loop | 0–8 |
| 同上 | `xSrcStride[5]` | uint64_t×5 | 同上 | T 元素 | A 轴 trailing input product；B 轴 0 | NDDMA `loopSrcStride` | inactive 槽初值 0 |
| 同上 | `xDstStride[5]` | uint32_t×5 | 同上 | T 元素 | trailing output product 后窄化 | NDDMA `loopDstStride` | 固定仓无显式 uint32 溢出检查 |
| 同上 | `xSize[5]` | uint32_t×5 | 同上 | T 元素 | 第 0 槽 `uLpUnit`，后续槽 output dim | NDDMA loopSize/UB innerShape | 固定仓无显式 uint32 溢出检查 |
| 同上 | `aAxesParams[24]` | int64_t×24 | 同上 | 每轴 `[size,inStride,outStride]`，T 元素 | `GetABAxesParams` | A 输入/输出 offset | inactive 槽 0 |
| 同上 | `bAxesParams[16]` | int64_t×16 | 同上 | 每轴 `[size,outStride]`，T 元素 | 同上 | B 输出 offset | inactive 槽 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | Host empty | `op_api/aclnn_expand.cpp::aclnnExpandGetWorkspaceSize` | 无 | 无 | 无 | ACLNN API |
| R-002 | Host scalar | 同上 | ViewCopy 参数 | 框架边界 | 框架边界 | ACLNN API |
| R-003 | 11000 | `expand_tiling_arch35.cpp` + `broadcast_to_tiling_base.cpp` | `BroadcastToTilingData` | `expand_apt.cpp::expand`→`broadcast_to_impl` | `broadcast_to_with_dma.h`，`maxDim=4` | `expand_def.cpp`、950 binary、CMake |
| R-004 | 11001 | 同上 | 同上 | 同上 | `broadcast_to_with_ub.h::BroadcastToUb<...,false>` | 同上 |
| R-005 | 11002 | 同上 | 同上 | 同上 | `broadcast_to_with_copypad.h::BrcToDataCopyPad` | 同上 |
| R-006 | 11003 | 同上 | 同上 | 同上 | `broadcast_to_with_tailAxis.h::BrcToWithTailAxis` | 同上 |
| R-007 | 11004 | 同上 | 同上 | 同上 | `broadcast_to_with_dma.h`，`maxDim=5` | 同上 |
| R-008 | 11005 | 同上 | 同上 | 同上 | `broadcast_to_with_ub.h::BroadcastToUb<...,true>` | 同上 |

## Kernel 实现

### 公共执行流程

普通 Route 共享单核心节点 `expand(x,shape,y,workspace,tiling)`；`shape` 仅参与 Host 值依赖和接口 ABI，Device body 不读取它。entry 先要求 workspace 非空并 `SetSysWorkspace`，把 1/2/4/8 字节 dtype 分别重解释为同宽无符号传输类型，声明 `KERNEL_TYPE_AIV_ONLY`，解析 POD，再按 key 实例化 body。

所有 body 用 `BrcToBase` 计算 A/B/U 循环。A 输入 offset 对最后一个 A 参数做线性乘法，A 输出和 B 输出则逆序做 `%axisSize` 与 trailing stride 的混合基数展开。主核尾块通过 `ntc*/tc*` 切换；二次分核通过 `doubleMode/dFactor` 缩小 B 或 U 的 `[begin,end)`。除 R-001/R-002 外没有多 Kernel workspace 边。

### Route R-001：空 Tensor

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/expand/op_api/aclnn_expand.cpp` | `aclnnExpandGetWorkspaceSize` | `self->IsEmpty() \|\| out->IsEmpty()` | `workspaceSize=0`，释放 executor，成功返回 |

#### Device 数据流

##### 函数级调用链

```text
aclnnExpandGetWorkspaceSize -> empty shortcut -> empty executor
```

##### 源码等价伪码

```text
if self.empty or out.empty:
  workspaceSize = 0
  return SUCCESS  # 不 enqueue Contiguous/Expand/ViewCopy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| workspace | N/A | 0 B | N/A | API | 无 | 调用者无需分配 | 无 |

##### 分层内存布局

N/A：没有 AIV、NDDMA 或 Device buffer。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | 无 | 无 | 无 | API 直接返回 | 参数校验已通过 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Host shortcut | 直接返回 | API | 调用者 | executor | 空输出不写；无精度变换 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/expand/op_api/aclnn_expand.cpp` | `aclnnExpandGetWorkspaceSize` | 空 Tensor 在 l0 调用前返回 |
| Kernel entry/router | N/A | N/A | 不启动 Kernel |
| Device body/helper | N/A | N/A | 不访问输入输出 |

### Route R-002：0D 标量直拷

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/expand/op_api/aclnn_expand.cpp` | `aclnnExpandGetWorkspaceSize` | 非空且 `selfDimNum==0 && size->Size()==0` | `l0op::ViewCopy(self,out)` |

#### Device 数据流

##### 函数级调用链

```text
aclnnExpandGetWorkspaceSize -> l0op::ViewCopy(self,out) -> framework executor
```

##### 源码等价伪码

```text
if rank(self)==0 and sizeRank==0:
  enqueue ViewCopy(self -> out)
  return executor.workspaceSize
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| self→out | 框架决定 | 1 个 T 元素 | 原 dtype/0D | ViewCopy | out | executor/stream 完成 | 固定仓不可见 |

##### 分层内存布局

N/A：没有 Expand AIV Kernel；ViewCopy 内部资源为框架边界。

##### 字节账与复用顺序

```text
Expand_UB_total = 0
dynamic_ub = N/A
workspace_total = 固定仓不可见（ViewCopy executor 生成）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ViewCopy 临时区 | 固定仓不可见 | out 提交 | stream 完成后 | executor stream 顺序 | self/out 生命周期覆盖执行期 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| ViewCopy | framework stream ordering | self | out | 1 个标量 | 位复制，无 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/expand/op_api/aclnn_expand.cpp` | `aclnnExpandGetWorkspaceSize` | 0D/空 size 专门绕过 Expand |
| Kernel entry/router | 固定仓不可见 | `l0op::ViewCopy` | 框架边界 |
| Device body/helper | 固定仓不可见 | ViewCopy body | 不属于 Expand 核心实现 |

### Route R-003：NDDMA-4D

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_to_tiling_base.cpp` | `CalcTilingKey` | 11002/11003/11001 谓词均不成立 | 初选 11000 |
| Tiling | 同上 | `UpdateTilingKey` | `uAxisCnt!=5` | 保持 11000 |
| Router | `broadcast_to.h` | `broadcast_to_impl` | `TILING_KEY_IS(11000)` | `BrcToWithNDDMA<T,TD,4>` |

#### Device 数据流

##### 函数级调用链

```text
ExpandAiCore -> expand -> broadcast_to_impl
  -> BrcToWithNDDMA<T,TD,4>::Init
  -> Process -> SetDMAParams -> CalcOutSize
     -> CopyDataInWithDMA(DataCopy<T,4>) -> MTE2_MTE3
     -> CopyDataOut(DataCopyPad) -> MTE3_MTE2
```

##### 源码等价伪码

```text
copyLpInfo[d] = {srcStride=xSrcStride[d], dstStride=xDstStride[d], size=xSize[d]}
for a in this_core_A:
  copySwitch=1
  for u in this_subcore_U_full_tiles:
    loopSize[0]=uLpUnit
    if copySwitch: NDDMA GM[xOffset] -> UB; wait MTE2_MTE3
    for b in this_subcore_B: DataCopyPad UB -> GM[yOffset+bOffset]
    wait MTE3_MTE2
    copySwitch *= isUNotB
  if uLeft: repeat with loopSize[0]=uLeft and exact byte length
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `Ain+inBlock+u*uLpUnit*uInOffset*isUNotB` | NDDMA 4D `xSize` | T/ND | caller | NDDMA | caller | B 轴 srcStride=0 |
| `VECIN que` | 0 | `tensorSize*S` B | T/packed NDDMA result | NDDMA | DataCopyPad | TPipe/整个 Process | depth 1，buffer 1 |
| output GM | `Aout+outBlock+u*uLpUnit*uOutOffset+Boffset` | `uLen*S*prod(xSize[1..3])` B/每 B | T/ND | DataCopyPad | ViewCopy | expandOut | tail 精确 blockLen |
| system workspace | framework offset | 16 MiB | bytes | executor | `SetSysWorkspace`/primitive | Expand 节点 | 不由 body 切分 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  nddmaQ["VECIN que · depth=1 · 1×tensorSize×S B"]:2
```

##### 字节账与复用顺序

```text
UB_total = tensorSize*S
UB_total <= floor(ubSize/blockSize)*blockSize
dynamic_ub = N/A
workspace_total(core node) = 16*1024*1024 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| NDDMA Queue buffer | `CopyDataInWithDMA` | 所有 B 输出 `CopyDataOut` | 下一 U tile | `MTE3_MTE2` event | 所有 B 副本提交后才覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | typed 4D NDDMA + `MTE2_MTE3` | MTE2 | MTE3 copyout | Queue buffer | 位复制；B 维 stride 0 实现广播 |
| UB→GM | `DataCopyPad` + `MTE3_MTE2` | MTE3 | 下一次 NDDMA | Queue buffer | `uLeft` 改 loopSize[0] 与 blockLen |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.cpp` | `CalcTilingKey/GetDMAAxesParams` | fallback key 与 4D stride/size |
| Kernel entry/router | `conversion/broadcast_to/op_kernel/arch35/broadcast_to.h` | `broadcast_to_impl` | 11000→4D NDDMA |
| Device body/helper | `conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_dma.h` | `BrcToWithNDDMA<T,TD,4>::Process` | NDDMA、B 复制、tail、event |

### Route R-004：UB 广播

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_to_tiling_base.cpp` | `CalcTilingKey` | UB_BRC 完整谓词且 `ubSizeCheck` | 初选 11001 |
| Tiling | 同上 | `UpdateTilingKey` | `uAxisCnt!=5` 且小尾 A 谓词不成立 | 保持 11001 |
| Router | `broadcast_to.h` | `broadcast_to_impl` | key 11001 | `BroadcastToUb<T,TD,false>` |

#### Device 数据流

##### 函数级调用链

```text
expand -> broadcast_to_impl -> BroadcastToUb<T,TD,false>::Init/Process
  -> BrcUIsB or BrcUNotB
  -> CopyDataIn -> BroadcastUb
     -> VFInnerBroadcastToB or VFInnerBroadcastToA
  -> CopyDataOut
```

##### 源码等价伪码

```text
innerShape[5-uAxisCnt..4] = xSize[0..uAxisCnt-1]
for each A:
  if U head is B:
    load one input U slice once
    MicroAPI broadcast B pattern into output Queue
    reuse result for all U tiles and all B offsets
  else:
    for each full U tile: load -> A/B MicroAPI reshape -> copy all B outputs
    if uLeft: shrink innerShape head and DMA lengths, then repeat
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | base formulas | logical `uLen*uInOffset` | T/ND | caller | DataCopyPad | caller | 不齐末轴用 blockCount |
| `inQueue` | 独立 UB | `bufferCnt*tensorSize*S` | T/innerShape | MTE2 | MicroAPI | TPipe/tile | depth=bufferCnt |
| `outQueue` | 独立 UB | 同上 | T/广播 innerShape | MicroAPI | MTE3 | TPipe/tile | depth=bufferCnt |
| RegTensor | 寄存器 | VReg 固定仓硬件量 | T/RT | MicroAPI load/gather | MicroAPI store | VEC scope | 64 位拆成两 32 位 lane |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue · depth=bufferCnt · bufferCnt×tensorSize×S B"]:1
  outQ["outQueue · depth=bufferCnt · bufferCnt×tensorSize×S B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*bufferCnt*tensorSize*S
bufferCnt=1 when ntcALen==1 and (U is B or uLpUnit>=ntcULen), else 2
UB_total <= 4*tensorSize*S <= aligned_ubSize
workspace_total(core node) = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input Queue slot | CopyDataIn | `BroadcastUb` DeQue | 下一 tile Alloc | `FreeTensor` | MicroAPI 完成后释放 |
| output Queue slot | MicroAPI | 所有 B copyout | 下一 tile Alloc | DeQue/FreeTensor | B/U 复用全部结束 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→V | TQue EnQue/DeQue | MTE2 | Vector | input Queue | DataCopyPad，不做 cast |
| Vector 内部 | `PipeBarrier<PIPE_V>` | 首轮 B broadcast | 外层 B 复制 | out Queue | mask 按 VL/tail 更新 |
| V→GM | outQueue EnQue/DeQue | Vector | MTE3 | output Queue | 末轴非 32 B 对齐时 blockCount 分段 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_to_tiling_base.cpp` | `CalcTilingKey/CalcDBMode` | UB key 与 bufferCnt |
| Kernel entry/router | `broadcast_to.h` | `BroadcastToUb<T,TD,false>` | 11001 实例 |
| Device body/helper | `broadcast_to_with_ub.h` | `BrcUIsB/BrcUNotB/BroadcastUb` | GM/UB、MicroAPI、复用、tail |

### Route R-005：尾轴大 A

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_to_tiling_base.cpp` | `CalcTilingKey` | `lastDim>=tensorSize && !abInfo[last]` | 11002 |
| Router | `broadcast_to.h` | `broadcast_to_impl` | key 11002 | `BrcToDataCopyPad<T,TD>` |

#### Device 数据流

##### 函数级调用链

```text
expand -> broadcast_to_impl -> BrcToDataCopyPad::Init/Process
  -> CopyDataIn(DataCopyPad + EnQue)
  -> CopyDataOut(DeQue + repeated DataCopyPad + FreeTensor)
```

##### 源码等价伪码

```text
for a in this_core_A:
  for u full tile:
    copy contiguous uLpUnit*uOutOffset bytes GM->bound Queue
    for b in this_subcore_B: copy same Queue bytes -> output+bOffset
  if uLeft: repeat with exact shorter blockLen
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `TQueBind VECIN/VECOUT` | 共享 UB | `2*tensorSize*S` | T/contiguous | DataCopyPad | DataCopyPad | TPipe/tile | depth 1、bufferNum 2 |
| GM input/output | 公共 A/B/U 公式 | `uLen*uOutOffset*S` | T/ND | caller/Queue | Queue/ViewCopy | caller/executor | tail 精确字节 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  boundQ["TQueBind · depth=1 · 2×tensorSize×S B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*tensorSize*S
dynamic_ub = N/A
workspace_total(core node) = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bound Queue slot | CopyDataIn | 所有 B CopyDataOut | 下一 U tile | EnQue/DeQue/FreeTensor | 全部 B 副本结束 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→GM | bound Queue protocol | MTE2 | MTE3 | tile | 位复制；`uLeft` 独立 blockLen |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_to_tiling_base.cpp` | `CalcTilingKey` | 11002 最高优先级 |
| Kernel entry/router | `broadcast_to.h` | `BrcToDataCopyPad<T,TD>` | 纯搬运实例 |
| Device body/helper | `broadcast_to_with_copypad.h` | `Process/CopyDataIn/CopyDataOut` | 绑定 Queue 与 B 复用 |

### Route R-006：尾轴大 B

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_to_tiling_base.cpp` | `CalcTilingKey` | `lastDim>=tensorSize && lastIsB && ubSizeCheck` | 11003 |
| Router | `broadcast_to.h` | `broadcast_to_impl` | key 11003 | `BrcToWithTailAxis<T,TD>` |

#### Device 数据流

##### 函数级调用链

```text
expand -> BrcToWithTailAxis::Init/Process
  -> CopyDataIn(A scalars)
  -> ProcessBrc -> BroadcastUb -> VFBrcTo(GatherMask + Duplicate)
  -> CopyDataOut(repeat B/U tiles)
```

##### 源码等价伪码

```text
for each A batch of aLpUnit (tail aLpLeft separately):
  load A scalars
  for each scalar in 32B groups and tail:
    gather scalar bytes from RegTensor; Duplicate across tensorSize
    for b and U tiles: copy duplicated UB span to output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inQueue | UB | `bufferCnt*tensorSize*S` | T/A scalars | MTE2 | RegTensor | TPipe/A batch | bufferCnt 1 iff `aAxesNum==0` |
| outQueue | UB | 同上 | duplicated T | Vector | MTE3 | TPipe/scalar | 同上 |
| RegTensor | VReg | `VL_CNT=VRegSize/S` lanes | T | load/gather | Duplicate/store | VEC scope | mask protects tail |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue · bufferCnt×tensorSize×S B"]:1
  outQ["outQueue · bufferCnt×tensorSize×S B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*bufferCnt*tensorSize*S
bufferCnt = (aAxesNum==0 ? 1 : 2)
workspace_total(core node) = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input Queue | CopyDataIn | 所有 batch scalar broadcast | 下一 A batch | input FreeTensor | scalar均已消费 |
| output Queue | VFBrcTo | 所有 B/U copyout | 下一 scalar | output FreeTensor | 所有输出副本提交 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→V→GM | 两个 TQue | MTE2/Vector | Vector/MTE3 | A batch/scalar | 位级 Gather+Duplicate；A tail、U tail均保护 |
| VEC | `UpdateMask` | brcCnt | RegTensor store | 最后 VL | 只写剩余 lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_to_tiling_base.cpp` | `CalcTilingKey/CalcDBMode` | 11003 与 bufferCnt |
| Kernel entry/router | `broadcast_to.h` | `BrcToWithTailAxis<T,TD>` | key 映射 |
| Device body/helper | `broadcast_to_with_tailAxis.h` | `ProcessBrc/VFBrcTo/CopyDataOut` | 标量收集、广播、双 tail |

### Route R-007：FULL-NDDMA-5D

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_to_tiling_base.cpp` | `UpdateTilingKey` | `uAxisCnt==5` | 覆盖为 11004 |
| Router | `broadcast_to.h` | `broadcast_to_impl` | key 11004 | `BrcToWithNDDMA<T,TD,5>` |

#### Device 数据流

##### 函数级调用链

```text
expand -> broadcast_to_impl -> BrcToWithNDDMA<T,TD,5>::Init/Process
  -> typed five-dimensional DataCopy -> DataCopyPad outputs
```

##### 源码等价伪码

```text
与 R-003 相同，但 SetDMAParams 填满 xSrcStride/xDstStride/xSize 的 5 个维度；
第 0 维按 uLpUnit/uLeft 动态修改，其余 4 维保持 Tiling 生成的后缀布局。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM→UB | 公共公式 | 5D `xSize`/stride | T/ND | typed NDDMA | Queue | tile | B 维 srcStride 0 |
| Queue | 0 | `tensorSize*S` | T/5D materialized suffix | NDDMA | MTE3 | Process | depth 1 |
| UB→output GM | 公共公式 | `uLen*S*prod(xSize[1..4])` | T/ND | DataCopyPad | ViewCopy | tile | 精确 tail |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  nddmaQ["VECIN que · 1×tensorSize×S B · 5D MultiCopy"]:2
```

##### 字节账与复用顺序

```text
UB_total = tensorSize*S
dynamic_ub = N/A
workspace_total(core node) = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| NDDMA Queue | five-dimensional DataCopy | all B copyouts | 下一 U tile | MTE2_MTE3 + MTE3_MTE2 | 同 R-003 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 5D NDDMA→MTE3 | hard events | MTE2 | MTE3/next MTE2 | Queue | 位复制；首维 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_to_tiling_base.cpp` | `GetUAxisInfo/UpdateTilingKey` | 五维覆盖优先于初选 key |
| Kernel entry/router | `broadcast_to.h` | `BrcToWithNDDMA<T,TD,5>` | 11004→5D 模板 |
| Device body/helper | `broadcast_to_with_dma.h` | `SetDMAParams/Process` | 5 槽参数、tail 与同步 |

### Route R-008：尾轴小 A

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `broadcast_to_tiling_base.cpp` | `UpdateTilingKey` | 11001、末轴 A、U>1、末两轴对齐字节 `<=vRegSize` | 11005 |
| Router | `broadcast_to.h` | `broadcast_to_impl` | key 11005 | `BroadcastToUb<T,TD,true>` |

#### Device 数据流

##### 函数级调用链

```text
expand -> BroadcastToUb<T,TD,true>::Process
  -> CopyDataIn -> BroadcastUb
     -> VFInnerBrcLastDimLEBlock (DIST_BLK)
        or VFInnerBrcLastDimGTBlock (GenGatherIdx + Gather)
  -> CopyDataOut
```

##### 源码等价伪码

```text
if innerLastDim <= 32/S:
  block-distributed load each axis2 row; copy to all axis1 positions
else:
  gatherIdx = arange % alignedLastDim
  gather repeated last-dim data, then copy to all axis1 positions
Queue/DMA loop and uLeft handling沿用 BroadcastToUb
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output Queue | 独立 UB | 各 `bufferCnt*tensorSize*S` | T/inner suffix | MTE2/Vector | Vector/MTE3 | tile | 同 R-004 |
| gatherIdx | RegTensor<int32_t> | VReg lanes | lane index | `GenGatherIdx` | Gather | VEC scope | `%axis4BA` |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["inQueue · bufferCnt×tensorSize×S B"]:1
  outQ["outQueue · bufferCnt×tensorSize×S B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*bufferCnt*tensorSize*S
bufferCnt = 2  # key 已更新为 11005，CalcDBMode 的 11001 单 buffer 条件不再命中
UB_total <= aligned_ubSize
workspace_total(core node) = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyDataIn/Vector | Vector/CopyDataOut | 下一 tile | EnQue/DeQue/FreeTensor | 同 R-004 |
| gatherIdx RegTensor | GenGatherIdx | Gather | 下一 VEC scope | VEC program order | VReg predicate已满足 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue pipeline | TQue | MTE2/Vector | Vector/MTE3 | tile | 位复制 |
| small last axis | mask + DIST_BLK/Gather | RegTensor load | store | aligned last dim | mask 限定 `lastASize`；DMA 不齐 32 B 分块 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_to_tiling_base.cpp` | `UpdateTilingKey` | VReg 门槛与 11005 |
| Kernel entry/router | `broadcast_to.h` | `BroadcastToUb<T,TD,true>` | bool 模板特化 |
| Device body/helper | `broadcast_to_with_ub.h` | `VFInnerBrcLastDimLEBlock/GTBlock` | DIST_BLK/Gather 两分支 |

## 算子特定优化

### Trick T-001：U 为广播轴时一次 NDDMA 装载复用全部 U/B 输出

#### 触发与源码依据

- 适用 Route：R-003、R-007
- 触发条件：`isUNotB==0`，即选中的 U 首轴是 B 轴
- Golden 来源：[`conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_dma.h:88-94`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_dma.h#L88-L94)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (copySwitch > 0) {
    DataCopy<T, maxDim, copyCfg>(tensor, inGM[gmInOffset], mCopyParams);
    this->InsertSync(HardEvent::MTE2_MTE3);
    copySwitch = copySwitch * uint8_t(tdPtr_->isUNotB);
}
```

#### 机制与设计原因

每个 A 循环先把 `copySwitch` 置 1。U 为 B 时输入地址不随 U 变化，首次 NDDMA 后 `copySwitch` 变 0，后续 U tile 与 B 轴副本直接复用同一 UB 物化结果，避免重复 GM→UB；U 为 A 时保持 1，每 tile 正常重载。`MTE2_MTE3` 保证首次数据对输出搬运可见。

#### 迁移边界

- 适用：广播轴在 NDDMA U 首维、多个输出 tile 读取同一输入片。
- 前提：输入地址确实不随 U 变化，所有 copyout 完成前 UB 不覆盖。
- 失效：U 是 A 轴时每 tile 必须重新 NDDMA；非 NDDMA Route 使用各自 Queue 复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若在 `isUNotB==1` 时也关闭 `copySwitch`，后续 U tile 会复制首 tile 的错误数据；这是源码布尔门控直接证明的反例。

### Trick T-002：连续同类广播轴合并后再构造 NDDMA/Vector 后缀

#### 触发与源码依据

- 适用 Route：R-003–R-008
- 触发条件：相邻轴的 `inShape[i]!=outShape[i]` 标志相同
- Golden 来源：[`math/expand/op_host/arch35/expand_tiling_arch35.cpp:142-155`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/expand/op_host/arch35/expand_tiling_arch35.cpp#L142-L155)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (size_t oIdx = 1; oIdx < dimNum; oIdx++) {
    if (abInfo[oIdx] == abInfo[mIdx]) {
        inShape[mIdx] *= inShape[oIdx];
        outShape[mIdx] *= outShape[oIdx];
    } else {
        mIdx += 1;
        inShape[mIdx] = inShape[oIdx];
        outShape[mIdx] = outShape[oIdx];
        abInfo[mIdx] = abInfo[oIdx];
    }
}
```

#### 机制与设计原因

合并保持 A/B 状态切换点，却减少 Device 需要展开的轴数；这直接降低 `uAxisCnt`、A/B 参数项和 NDDMA loop 层数，并允许更大的连续 U tile。只合并广播属性相同的相邻轴，因而不会把 srcStride=0 的 B 轴与正常递增的 A 轴混为一谈。

#### 迁移边界

- 适用：纯广播/复制且连续 layout 的相邻轴折叠。
- 前提：轴相邻、A/B 属性相同、shape 乘积与后续 offset 类型不溢出。
- 失效：A/B 交替边界必须保留；固定仓对 shape 连乘溢出没有显式检查。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

跨 A/B 边界合并会丢失“输入 stride 正常/为零”的差异，无法重建正确广播地址。

### Trick T-003：末两轴可装入 VReg 时覆盖通用 UB 路线

#### 触发与源码依据

- 适用 Route：R-008
- 触发条件：初选 11001、末轴 A、U 轴数大于 1，且 `out[d-2]*ceilAlign(lastDim*S,blockSize)<=vRegSize`
- Golden 来源：[`conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.cpp:309-320`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.cpp#L309-L320)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tilingKey_ == TILING_MODE_UB_BRC && isLastDimB_ == 0 && uAxisCnt_ > 1) {
    auto dimNum = outShapePtr_->GetDimNum();
    auto lastDimAlign = Ops::Base::CeilAlign(outShapePtr_->GetDim(dimNum - 1) * dtypeSize_, blockSize_);
    if (outShapePtr_->GetDim(dimNum - nTwo) * lastDimAlign <= vlSize_) {
        tilingKey_ = TILING_MODE_LAST_DIM_SMALL_A;
        return;
    }
}
```

#### 机制与设计原因

Host 用真实 dtype 字节数和 block 对齐后的末轴计算末两轴字节量；能装进 VReg 时，Device 改用 `DIST_BLK` 或预生成 `%axis4BA` gather 索引的专门模板，减少通用多层地址生成。条件在 Host 固化为 key，Device 不承担资源可行性猜测。

#### 迁移边界

- 适用：末轴不广播、倒数第二轴存在扩张、末两轴工作集不超过 VReg。
- 前提：使用同一 `blockSize/vRegSize` 口径并保留 dtype 对齐。
- 失效：条件不成立时保持 R-004；U 恰为 5 维时先改 R-007。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只按元素数、不乘 dtype 字节数或不做 block 对齐，会错误放行超出 VReg 的 shape；源码的字节表达式给出了可验证反例。

## 模板与实现对比

| 对比维度 | R-001/R-002 Host shortcut | R-003/R-007 NDDMA | R-004/R-008 UB Vector | R-005/R-006 尾轴专用 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | 空域最先，0D 次之 | 通用 fallback；5D 强制覆盖 | UB 可容纳；小尾 A 再特化 | 大尾 A/B 高优先级 | 先语义捷径，再资源/尾轴 |
| 模板与 dtype | framework | `BrcToWithNDDMA<T,TD,4/5>` | `BroadcastToUb<T,TD,false/true>` | `BrcToDataCopyPad`/`BrcToWithTailAxis` | dtype 均按位宽重解释 |
| 循环与分核 | 无核心循环 | A→U→B | A→U→B + inner VEC | A/U/B 或 A batch→scalar→B/U | 公共 A/B/U 分核 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 固定仓不可见/无 | 1×Queue | 双 Queue + RegTensor | bound Queue 或双 Queue+RegTensor | 无 dynamic UB/通信 |
| 内存总占用与复用 | Expand 为 0 | `tensorSize*S` | `2*bufferCnt*tensorSize*S` | `2*tensorSize*S` 或同双 Queue | ubSizeCheck 覆盖最坏 4 倍 |
| 搬运/Vector/SIMT/NDDMA/通信 API | ViewCopy | typed 4D/5D DataCopy + DataCopyPad | MicroAPI broadcast/gather | DataCopyPad 或 Gather+Duplicate | 无 SIMT/通信/Cube |
| 同步、精度与 tail | framework | hard event；U tail | Queue/PIPE_V；mask/U tail | Queue；A/U/VL tail | 全部位复制、无 cast |
| fallback/失败边界 | 成功捷径 | 未知 key 无输出 | 超 UB 转 NDDMA | 不满足条件转后续 Route | workspace 空时 entry 返回 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但固定源码缺对应 AICore 注册/config，未建卡 | 同左 | 公开支持且 source_closed |
| Host/Launcher | ACLNN API 可见；对应实现接线不闭合 | 同左 | `DAV_3510` 接受 BF16；AICore `expand_apt` |
| Kernel/资源 | 固定仓可见 AICPU body不属于本 VV 卡；实际公开路线闭包不足 | 同左 | 六 key AIV/NDDMA，共享 BroadcastTo body，16 MiB system workspace |

## 限制与失败边界

- `self/out` 必须同 dtype，输出秩最大 8，`sizeRank>=selfRank`；非 1 输入维不能扩成不同大小。API 支持非连续输入输出，但通过框架 `Contiguous/ViewCopy` 物化。
- 内部图/Tiling 存在 `-1` 继承分支，但公开 ACLNN/l0 shape 推导会把继承结果覆盖回 `-1`；固定源码无法证明公开入口可接受该值，因此不得依赖它。
- API 与 Tiling 对目标维 1 的处理存在接口层差异：API `CheckShape` 要求非 1 输入维与 `size` 对应维相等，而图侧 Tiling 的 `ApplyBroadcastRules` 会把目标 1 改回输入维；公开 ACLNN 先执行更严格的 API 检查。
- 空 Tensor 在 ACLNN Host 直接成功返回且不写 Device；直接绕过 ACLNN 调 Tiling 时，`IsEmptyTensor` 会失败。这是入口层语义差异。
- `xDstStride/xSize` 写入 uint32_t，`tensorSize` 写入 uint32_t；固定仓没有公开 shape 连乘或窄化溢出检查。POD 宏展开后的精确 `sizeof/alignof/offsetof` 固定仓不可见。
- 普通核心节点明确需要 16 MiB system workspace；ACLNN 全 executor 合并 Contiguous/Expand/ViewCopy 后的总 workspace offset、对齐与框架同步事件固定仓不可见。
- entry 在 workspace 为空或未知 key 时静默返回，不提供 Device 错误码；正确性依赖 Host producer 和 executor。
- A2/A3 虽有公开支持声明，但本固定仓没有满足 VV `source_closed` 的对应 AICore 平台接线，因此未生成 `expand.md`。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/expand/docs/aclnnExpand.md` | `aclnnExpandGetWorkspaceSize/aclnnExpand` | 平台、dtype、shape、错误语义 |
| 入口实现 | `math/expand/op_api/aclnn_expand.cpp` | `CheckParams/aclnnExpandGetWorkspaceSize/aclnnExpand` | 校验、空域、0D、executor 图 |
| 入口实现 | `math/expand/op_api/expand.cpp` | `Expand/ExpandAiCore/IsAiCoreSupport` | 连续性、shape Tensor、AICore 路由 |
| Host/Tiling/Launcher | `math/expand/op_host/expand_def.cpp` | `ops::Expand` | 950 OpDef 和 `expand_apt` 注册 |
| Host/Tiling/Launcher | `math/expand/op_host/arch35/expand_tiling_arch35.cpp` | `GetShapeInfo/Tiling4Expand/TilingPrepare4Expand` | shape 归一化与硬件参数 |
| Host/Tiling/Launcher | `conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.h` | `BroadcastToTilingData/BroadcastToTilingAscendC` | POD 和 Tiling 状态 |
| Host/Tiling/Launcher | `conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.cpp` | `CalcTilingData/WriteTilingData` | key、tile、分核、workspace |
| Kernel entry/body | `math/expand/op_kernel/expand_apt.cpp` | `expand` | Expand entry 委托共享 body |
| Kernel entry/body | `conversion/broadcast_to/op_kernel/arch35/broadcast_to.h` | `broadcast_to_impl` | workspace、dtype、六 key router |
| Kernel entry/body | `conversion/broadcast_to/op_kernel/arch35/broadcast_to_base.h` | `BrcToBase` | A/B/U 地址和二次分核 |
| Kernel entry/body | `conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_dma.h` | `BrcToWithNDDMA` | 4D/5D NDDMA |
| Kernel entry/body | `conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_copypad.h` | `BrcToDataCopyPad` | 大尾 A 纯搬运 |
| Kernel entry/body | `conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_tailAxis.h` | `BrcToWithTailAxis` | 大尾 B Vector 广播 |
| Kernel entry/body | `conversion/broadcast_to/op_kernel/arch35/broadcast_to_with_ub.h` | `BroadcastToUb` | 通用/小尾 UB MicroAPI |
| 构建接线 | `math/expand/op_host/config/ascend950/expand_binary.json` | 16 个 binary 条目 | 950 dtype×shape dtype 实例 |
| 构建接线 | `math/expand/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/add_all_modules_sources` | 950 构建与 broadcast_to 依赖 |
| 构建接线 | `conversion/broadcast_to/CMakeLists.txt` | `SUPPORT_TILING_DIR/add_all_modules_sources` | 共享 body 的 arch35 构建接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 共享 `BroadcastToTilingData` 可供 Expand/BroadcastTo 复用 | 字段顺序、位宽、数组长度、元素/字节单位必须一致 | `BEGIN_TILING_DATA_DEF(BroadcastToTilingData)` |
| 调用顺序 | normal path 可复用 Contiguous→Expand→ViewCopy | empty 与 0D shortcut 必须先于 normal path | `aclnnExpandGetWorkspaceSize` |
| workspace/内存 | 核心固定 16 MiB system workspace；各 Route UB 公式可复用 | entry 先 `SetSysWorkspace`；Queue bufferCnt 不得漏乘 | `WriteTilingData`、各 `InitBuffer` |
| 同步 | NDDMA 的双 hard-event 与 Queue 生命周期可复用 | MTE3 未结束前不得覆盖单 NDDMA buffer | `InsertSync`、`CopyDataInWithDMA/CopyDataOut` |
| 精度与尾块 | 全 dtype 按位复制；统一用 `uLeft`、mask、A tail | 64 位必须保留双 32 位 lane；DMA 长度是字节 | `DTYPE_X_`、各 `Process` |
| 失败边界 | Host 校验/资源失败阻断 launch | 不得给未知 key 或 null workspace 编造 fallback | `CheckParams`、`broadcast_to_impl` |

- [SetSysWorkSpace](../../../../apis/simd_api/basic_api/resource_management/workspace/set_sys_work_space.md)
