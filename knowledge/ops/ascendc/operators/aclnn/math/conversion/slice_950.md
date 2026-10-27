---
type: Operator
title: "Slice 950 ACLNN 算子知识"
description: "说明 aclnnSlice 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/slice/docs/aclnnSlice.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/slice/op_api/aclnn_slice.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/slice/op_api/aclnn_slice.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_v3/op_api/strided_slice_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_v3/op_host/strided_slice_v3_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_v3/op_host/arch35/strided_slice_v3_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_v3/op_kernel/strided_slice_v3_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_last_dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_two_dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_nddma.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_nddma_last_dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_gather.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_v3/op_host/config/ascend950/strided_slice_v3_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_v3/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/tests/ut/op_api/stub/opdev/platform.h
    role: implementation
created_at: "2026-08-25T00:00:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Slice 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 公开入口 | 两段式 `aclnnSliceGetWorkspaceSize` / `aclnnSlice`；本卡不包含 `aclnnSliceV2` |
| 核心链路 | `aclnnSlice` → `l0op::StridedSliceV3` → `StridedSliceV3` arch35 Tiling → `strided_slice_v3` router → 共用 `strided_slice` arch35 body |
| 计算语义 | 仅沿一个 `dim` 取半开区间 `self[..., start:end:step, ...]`；`step>0`，结果是纯数据搬运，不改变元素值 |
| 950 dtype | FLOAT、FLOAT16、BFLOAT16、INT64、INT8、INT32、UINT8、BOOL；固定实现的 SoC gate 为 `ASCEND910B..ASCEND910E`，固定仓枚举中 `ASCEND950` 位于该闭区间内 |
| ACLNN 可达 Route | R-001 key 100 MoveAlign；R-002 key 101 MoveAlign-last；R-003 key 102 NDDMA；R-004 key 103 NDDMA-last；R-005 key 150 2D MoveAlign；R-006 key 200 SIMT；R-007 key 300 MoveAlign-gather |
| 执行单元 | MoveAlign 与 gather 为 AIV vector 路线；102/103 为 NDDMA；200 为 SIMT。均属于 VV；不存在公开可达 Cube/AIC/MIX_CV 节点 |
| Workspace | 核心 StridedSliceV3 Tiling 固定申报 16 MiB；Device entry 接收但 body 不读取该地址。executor 还可能包含框架 Contiguous/ViewCopy 临时区，固定仓不公开其内部切分 |
| 来源闭包 | `source_closed`：公开 ACLNN、校验与规范化、L0 launcher、950 OpDef/binary/CMake、Host Tiling/POD、router 和 7 条可达 body 已闭合 |
| 验证状态 | 按任务要求未运行 validator、测试、编译、真机或第二轮 review |

## 接口与语义

### 计算公式与索引规范化

设 `d=self.rank`，选中轴长度为 `N=self.shape[dim]`。入口先把负轴变为 `dim+=d`；负 `start/end` 各加一次 `N`，随后把 `start` 截到 `[0,N]`，把 `end` 截到 `[start,N]`。固定实现最终使用：

```text
out.shape[j] = self.shape[j]                         (j != dim)
out.shape[dim] = (end - start + step - 1) / step
out[c0,...,cd-1] = self[c0,...,start+c_dim*step,...,cd-1]
```

这里是整数除法，区间为半开区间。`step<=0` 在公开入口被拒绝；因此公共 Slice 不会进入 router 的负步长 key。文档给出的分段表达式与 `CalculateValuesSliceUpperForSlice`、`CalculateValuesSliceLowerForSlice` 的两阶段实现一致。

### 参数、shape 与失败条件

| 参数 | 类型/shape | 固定实现约束 | Host/Device 去向 |
|---|---|---|---|
| `self` | ND tensor，rank 1–8 | 可非连续；dtype 为 FLOAT/FLOAT16/BFLOAT16/INT64/INT8/INT32/UINT8/BOOL | 先经 `Contiguous`，再作为 StridedSliceV3 的 `x` |
| `dim` | int64 标量 | `[-rank,rank-1]`，负值归一化 | L0 变成单元素 INT64 `axes` tensor |
| `start` | int64 标量 | 负值加 N，再截到 `[0,N]` | 单元素 INT64 `begin` tensor |
| `end` | int64 标量 | 负值加 N，再截到 `[start,N]` | 单元素 INT64 `end` tensor |
| `step` | int64 标量 | 必须 `>0` | 单元素 INT64 `strides` tensor |
| `out` | ND tensor，rank 1–8 | dtype 与 self 相同，view shape 必须精确匹配推导结果；可非连续 | `ViewCopy(sliceOut,out)` 写回用户 view |
| `workspaceSize/executor` | Host 输出 | 非空地址；由 executor 汇总 | 供第二段调用 |
| `workspace/stream` | Device 执行参数 | 第二段传给 `CommonOpExecutorRun` | executor 内节点按同一 stream 调度 |

空输入或空输出在第一段直接返回 `workspaceSize=0` 并交出 executor，不添加 StridedSliceV3 节点。因此核心 key 200 中的 empty no-op 是底层能力，但不是本公开入口非空主链的必经节点。

### 950 dtype 与二进制匹配

`DTYPE_SUPPORT_LIST` 列出 FLOAT/FLOAT16/INT64/INT8/INT32/UINT8/BOOL/BF16。`CheckSocVersionIsSupportBf16` 接受枚举闭区间 `ASCEND910B..ASCEND910E`；固定仓 `SocVersion` 定义中的顺序为 `ASCEND910B, ASCEND910_93, ASCEND950, ASCEND910E`，因此 Ascend 950 命中该 gate。`strided_slice_v3_binary.json` 为 x/y 编译 1/2/4/8 字节匹配，并以 `DtypeByte` 覆盖 BOOL/UINT8/INT8、FLOAT16/BFLOAT16、FLOAT/INT32、INT64 的位宽；L0 生成的索引 tensor 是 INT64，对应 binary 中的 INT64 index 变体。Device dispatch 对 1/4/8 字节显式映射为 `int8/int32/int64`，2 字节保留 `DTYPE_X`，所以 FLOAT16 与 BFLOAT16 都保留各自 `DTYPE_X`；这些路线只复制位模式，不做数值 cast。

## Host/Launcher 执行流程

### ACLNN 与 L0 调用链

```text
aclnnSliceGetWorkspaceSize(self, dim, start, end, step, out)
  -> CREATE_EXECUTOR
  -> CheckParams
       -> tensor 非空指针
       -> self/out dtype 相同且受支持；A5 BF16 命中 SoC 枚举闭区间
       -> rank 1..8、dim 合法、step>0
       -> start/end/dim 规范化并核对 out shape
  -> if self.IsEmpty || out.IsEmpty: workspace=0, return
  -> l0op::Contiguous(self)
  -> 再次规范化 dim/start/end
  -> l0op::StridedSliceV3(selfContiguous,start,end,dim,step)
       -> 4 个标量转单元素 INT64 tensor
       -> INFER_SHAPE(StridedSliceV3)
       -> ADD_TO_LAUNCHER_LIST_AICORE
  -> l0op::ViewCopy(sliceOut,out)
  -> executor.GetWorkspaceSize

aclnnSlice(workspace, workspaceSize, executor, stream)
  -> CommonOpExecutorRun
```

任一检查失败、executor 创建失败、Contiguous/L0/ViewCopy 返回空都会立刻返回错误，不会继续添加后续节点。Contiguous、ViewCopy、CommonOpExecutorRun 的内部 kernel、事件、临时 tensor offset 和释放细节位于 opdev 框架边界；固定算子仓只给出调用点，不能推测补齐。

### StridedSliceV3 Host 参数形成

`TilingPrepareForStridedSliceV3AscendC` 从 `PlatformAscendC` 读取 AIV core 数、UB 大小和 cache-line 大小，三者必须大于 0。Tiling 将单轴 begin/end/axes/strides 扩成全 rank 数组：未选轴为 `begin=0`、`end=shape`、`stride=1`，选中轴使用 ACLNN 已规范化的标量。`MakePerformanceParams` 还会合并相邻的“完整、stride=1”维度；所以逻辑 rank 可达 8，但 route 选择使用的是性能合并后的有效 rank。

`TilingForStridedSliceV3` 实际调用共用 `StrideSliceTiling::DoTiling`。V3 文件中的另一套 `SetTilingMode`（mode 1–18）在固定调用链中未被调用，不能把它当成本入口 router 的 selector；真正 selector 是共用 arch35 Tiling 写出的 key 100/101/102/103/150/200/300。

### 启动、Workspace 与生命周期

核心节点只有一个 AICore launcher：

```text
strided_slice_v3<<<realCoreNum>>>(
    x, begin, end, axes, strides, y, workspace, tiling)
```

router 只消费 `x/y/tiling`；`begin/end/axes/strides` 的值已固化到 Tiling POD，`workspace` 也未被各 body 读取。`RunStrideSliceTiling` 仍把 `workspace[0]` 设为 `16777216` 字节，这一 ABI 申报不能因 body 未读而从调用侧删除。blockDim 与 `realCoreNum` 一致；executor、workspace 和输入输出必须保持有效，直至 stream 上执行完成。

## Route 与参数设计

### 分核、UB 与执行族选择

令 `B=sizeof(T)`、`C=cacheLineSize`、`U=ubSize`、`O=product(outputShape)`。Tiling 内 `totalOutputSize=O*B`。一般分核以每核约 1 KiB 输出为目标：

```text
rightDiv = min(coreNum, ceil(totalOutputBytes / 1024))
blkFactor = ceil(outputShape[blkIndex] / rightDiv)
realCoreNum = ceil(outputShape[blkIndex] / blkFactor)
              * product(outputShape[0:blkIndex])
```

普通 MoveAlign/NDDMA 的可用元素数为 `(U-8192)/B/2`，其中 8192 字节是保留区，除以 2 对应 depth-2 Queue。二维特化采用约 4 KiB 输出分核目标。gather 因输入覆盖跨度约为输出的 `lastStride` 倍，改为：

```text
ubOutBase    = floor((U-8192) / 2 / (lastStride+1), C)
ubSizeInput  = ubOutBase * lastStride
ubSizeOutput = ubOutBase
```

小 dtype gather 还会把输入 slot 限在约 65536 字节内，以适配窄索引。

### Route 选择表

| Route | key / body | 精确 Host 条件 | 资源与 fallback |
|---|---|---|---|
| R-001 | 100 / `StridedSliceMoveAlign` | 正步长；命中 MoveAlign；UB 不切末轴；排除 rank=2 的专用情形；非 gather | 最多四维 compact MoveAlign；POD primitive 字段超宽时向更内轴重算 |
| R-002 | 101 / `StridedSliceMoveAlignLastDim` | MoveAlign 且 `ubIndex=rank-1`、`useGather=false` | 末轴线性切块；分核轴与 UB 轴同/异两套尾块公式 |
| R-003 | 102 / `StridedSliceNDDMA` | 未命中 MoveAlign；NDDMA 条件成立；`ubIndex<rank-1` | 最多五维 `MultiCopy`；密度不足且非小域/溢出才落到 SIMT |
| R-004 | 103 / `StridedSliceNDDMALastDim` | 同一 NDDMA 条件，且 `ubIndex=rank-1` | 一维 `MultiCopy`，源 stride 来自切片步长 |
| R-005 | 150 / `StridedSliceMoveAlignTwoDim` | rank=2、MoveAlign、`ubIndex=rank-2`、非 gather | 二维行搬运专用 POD 与尾核公式 |
| R-006 | 200 / `StridedSliceSIMT` | 未命中 MoveAlign/NDDMA，shape 乘积不超 UINT32；底层 empty 也用该 key | 直接 GM grid-stride；正常 Slice 空 tensor 已在 ACLNN 提前返回 |
| R-007 | 300 / `StridedSliceMoveAlignGather` | 末轴 stride 2–16、覆盖输入段至少一条 cache line、输出总字节>1024 | MoveAlign 载入跨度、MicroAPI gather、reorder、紧凑写回 |

正向 selector 的关键优先级为：

```text
if lastStride==1 and lastOut*B>=64:
    MoveAlign
else if 2<=lastStride<=16 and coveredSpanBytes>=C and totalOutputBytes>1024:
    MoveAlignGather
else if validElementsPerCacheLine*B>=64
     or shapeProductExceedsUint32
     or outputElements<=1024:
    NDDMA
else:
    SIMT
```

`SetTilingMode` 再把最多 4 个活跃搬运维映射到 MoveAlign、5 个映射到 NDDMA，其余映射到 SIMT。shape 乘积溢出 UINT32 时在正向 selector 中已经强制进 NDDMA，所以 key 201 的 big-shape SIMT 在公开正步长 Slice 下不可达。key 301/302/303 只从负 stride 分支产生，而 `aclnnSlice` 在 L0 前拒绝 `step<=0`，同样不是本卡 Route。

### POD/ABI 闭包

所有 POD 通过 Tiling 宏声明，固定仓没有宏展开后的 `sizeof/alignof/offsetof`，因此只能确认字段顺序、类型和 producer/consumer，不能猜 padding。

| POD | 关键字段 | Host producer | Device consumer |
|---|---|---|---|
| `StridedSliceBaseTilingData` | `blkFactor/blkTailFactor` int64；`ubInLoopSteps/ubOutLoopSteps` int64；`ubSize/ubFactor/ubTailFactor/ubTailTailFactor` int32；`realCoreNum/inputDims/blkIndex/ubIndex` int16；`outputShape/begin/strides/inputSteps/rowsOffsetSteps[8]` int64 | block/UB split、性能维合并、后缀乘积与地址步长函数 | `StridedSliceBase` 解析分核、地址、主尾 tile |
| `StridedSliceMATilingData` | base + `MoveAlignParams` | `SetMovAlignV2TilingMode` | key 100 compact CopyIn/CopyOut |
| `StridedSliceMALastDimTilingData` | base | 同上 | key 101 线性末轴 body |
| `StridedSliceNDDMATilingData` | base + `nddmaTotalNum` int64 + `ubSizeInput` int32 + 5 维 `loopSize/srcStride/dstStride` int64 数组 | `SetNddmaTilingMode` | key 102 `MultiCopy<T,5>` |
| `StridedSliceNDLastDimTilingData` | 与 NDDMA 专用字段对应的一维有效参数 | 同上 | key 103 `MultiCopy<T,1>` |
| `StridedSliceMATwoDimTilingData` | 二维 shape/stride/切分字段 + MoveAlign 参数 | `SetTwoDimTilingInfo` | key 150 二维专用 body |
| `StridedSliceSIMTTilingData` | `isEmpty` uint16、`inputDims` int16、`begin/strides/outputShapeProd/inputShapeProd[8]` int64 | `SetSIMTTilingMode` | key 200 展平/反展平索引 |
| `StridedSliceMAGatherTilingData` | base + MoveAlign params + `ubSizeInput` int32 | gather UB 与 copy 参数生成 | key 300 输入跨度、索引和输出 slot |

`MoveAlignParams` 内含 `blockCount/loop1Size/loop2Size` 等 uint16，以及 `blockLen/srcStride/dstStride/loop*SrcStride` 等 uint32 和部分 uint16 目的步长。`ResetMovAlignV2Para` 在值超出 primitive 位宽时向内移动 UB 切轴并重算，而不是截断。

## 逐 Route Device 实现

### R-001：key 100 MoveAlign compact

```text
router(100)
  -> StridedSliceMoveAlignProcess<T,U>
  -> Init: ParseBaseTilingDataV2 + TQueBind<VECIN,VECOUT,1>
           InitBuffer(vecQue, 2, ubSize/2)
  -> ProcessPerBlock
       -> CopyIn: DataCopyPad<Compact>(GM -> LocalTensor)
       -> EnQue / DeQue
       -> CopyOut: DataCopyPad(LocalTensor -> GM)
       -> FreeTensor
```

`MoveAlignParams` 表达 block count/length 与至多两层 loop stride。每核按 base POD 计算输入 `begin + row/tile stride` 和紧凑输出 offset。主 tile 使用 `ubFactor`，最后 UB tile 用 `ubTailFactor`；尾核且 block/UB 同轴时还用 `ubTailTailFactor`。没有 Vector 算术或 cast，队列承担 MTE2→MTE3 的依赖与双缓冲槽生命周期。

### R-002：key 101 MoveAlign last dimension

该 body 同样分配 depth-2 绑定队列，每槽 `ubSize/2`。当 `blkIndex==ubIndex==lastDim` 时，多核可切同一行，输入/输出线性 offset 分别按源 stride 与输出紧凑长度推进；否则按外层行遍历。CopyIn/CopyOut 都是线性 `DataCopyPad`，主核、尾核、主 tile 与尾 tile 分开修正长度，保证不把 padding 写成有效输出。

### R-003：key 102 NDDMA multi-dimensional

```text
router(102) -> StridedSliceNDDMAProcess<T,U>
  -> InitBuffer(queue, 2, ubSize/2)
  -> 对每个 core/row/tile:
       MultiCopy<T,5>(GM -> UB, loopSize, srcStride, dstStride)
       Queue EnQue/DeQue
       DataCopyPad(UB -> compact GM output)
```

Host 把活跃维反向填入最多 5 维 NDDMA 参数；未用维的 loopSize 为 1、stride 为 0。源 stride 含输入后缀步长和切片 stride，目的 stride 表达紧凑 UB。尾 tile 只改变最外活跃 loop 的有效次数，Device 不重新推导 shape。NDDMA 数据搬运后不做值计算。

### R-004：key 103 NDDMA last dimension

该 route 把多维配置退化为 `MultiCopy<T,1>`。源步长是末轴切片 `step`，目的步长为 1。若 block 和 UB 都切末轴，输入 offset 必须额外乘源 stride；输出始终紧凑递增。depth-2 Queue 和尾块生命周期与 R-003 相同。

### R-005：key 150 two-dimensional MoveAlign

二维专用 body 直接以：

```text
inputOffset  = begin[1] + (begin[0] + row0*stride[0]) * inputSteps[1]
outputOffset = row0 * outputShape[1]
```

定位行，再执行 compact 二维 CopyIn/CopyOut。分核与 UB 都围绕第 0 轴，尾核最后一段由 `ubTailTailFactor` 给出。资源仍是 depth-2 绑定 Queue，无额外 TBuf、dynamic UB、通信或 atomic。

### R-006：key 200 SIMT

```text
router(200) -> StridedSliceSIMTProcess<T,U>
  -> if isEmpty: return
  -> rank 1..8 specialization
  -> for idx = blockIdx*threads + threadIdx; idx < outputSize;
       idx += gridSize:
       coord = unflatten(idx, outputShapeProd, magic, shift)
       inIdx = sum((begin[j] + coord[j]*strides[j]) * inputShapeProd[j])
       y[idx] = x[inIdx]
```

rank 1–7 使用 2048 threads，rank 8 使用 1024 threads。普通 SIMT 的输出乘积和坐标使用 uint32 路径，Host 通过 overflow guard 保证其边界。线程只读写唯一元素，不分配 AIV UB，不使用 Queue、barrier 或 atomic；grid-stride 的 `idx<outputSize` 自然保护尾部。这里同样是位精确复制。

### R-007：key 300 MoveAlign gather

```text
router(300) -> StridedSliceMoveAlignGatherProcess<T,U>
  -> InitBuffer(inQue,  2, ubSizeInput)
  -> InitBuffer(outQue, 2, ubSizeOutput)
  -> InitBuffer(idxBuf, vlCnt*sizeof(T))
  -> GenGatherIndex: idx[lane] = lane * lastStride
  -> per core/tile:
       MoveAlign/DataCopyPad covered GM span -> inQue
       MicroAPI::DataCopyGather(inUB, idx, mask) -> registers
       Reorder4Output -> outQue
       DataCopyPad outQue -> compact GM y
```

等差索引只生成一次，跨 tile 复用；每行或每 VL 通过寄存器偏移调整。末 VL 用 mask 限制有效 lane。8-bit 路线借用 16-bit register view 并做 `DIST_PACK_B16`，只是重新紧凑排列字节，不是数值类型转换。输入与输出 Queue 独立双缓冲：inQue 的 enqueue/dequeue/free 保护 MTE2→Vector，outQue 保护 Vector→MTE3；index TBuf 在 kernel lifetime 内只读。总显式 UB 为 `2*ubSizeInput + 2*ubSizeOutput + idxBuf`，Host 的 8192 字节 reserve 覆盖索引及辅助开销。

## 同步、精度与资源总表

| Route | 显式 UB | 同步/依赖 | 数值语义 | tail |
|---|---|---|---|---|
| 100/101/150 | depth-2 绑定 Queue，总 payload 为 `ubSize` | EnQue/DeQue/Free | 原 dtype 位模式搬运 | 主尾核 + UB 主尾 tile，必要时 `ubTailTailFactor` |
| 102/103 | depth-2 Queue，总 payload 为 `ubSize` | NDDMA 产出后 Queue 交给 MTE3 | 原 dtype 位模式搬运 | Host 修改有效 loopSize/长度 |
| 200 | 0 | 每线程程序序；无跨线程同步 | 单次 GM load/store | `idx<outputSize` |
| 300 | `2*in + 2*out + idxBuf` | in/out Queue；Vector register 顺序 | gather/reorder 后位模式不变 | mask + 重算覆盖跨度/输出长度 |

所有可达 Route 都没有 Cube/AIC/MIX_CV、跨核通信窗口、global atomic、reduction 或动态 workspace 消费。节点申报的 16 MiB workspace 与显式 UB 是不同资源：前者属于 GM workspace ABI，后者由 kernel `TPipe` 管理。

## 算子特定优化

### T-001：先按 cache-line 有效密度选搬运引擎

Tiling 不把非连续 slice 一律降成 SIMT。连续末轴输出达到 64 字节时优先 MoveAlign；稀疏但一个 cache line 内仍能产生至少 64 字节有效输出时使用 NDDMA；stride 2–16、覆盖跨度足够且输出大于 1 KiB 时走 gather；只把剩余低密度且非小域场景交给 SIMT。这个选择同时考虑 `sizeof(T)`，所以相同 shape/stride 可能因 dtype 字节宽度而改变 Route。

迁移该优化时必须保留优先级：小输出和 uint32 shape overflow 明确强制 NDDMA，不能仅依据“稀疏”选择 SIMT；MoveAlign primitive 位宽超界也必须重新切轴，不能静默窄化。

### T-002：gather 的 UB 按 stride 比例分配

末轴步长为 `s` 时，一个紧凑输出 tile 需要约 `s` 倍输入跨度，所以输入 slot 与输出 slot 不是等分，而是 `s:1`。公式还同时计入双缓冲与 cache-line 对齐，并对窄 gather index 限长。这样 Host 的输入覆盖区、Device 的等差索引范围和 Queue extent 一致。

### T-003：性能维合并减少 active dimensions

V3 Host 把连续完整取值、stride=1 的相邻维合并后再交给共用 Tiling。单轴 Slice 大部分未切维都是完整维，因此有效 rank 往往远小于逻辑 rank；这直接扩大 MoveAlign/NDDMA 能覆盖的 shape 范围。复用时不能在合并中跨越真实切片轴，且必须同步更新 output/input 后缀乘积。

## 平台与版本边界

| 项目 | A2/A3 | 950 |
|---|---|---|
| 固定 commit 发布闭包 | `conversion/slice/CMakeLists.txt` 声称旧平台支持，但固定树内缺少对应 arch32 Host/Kernel 源与可验证构建闭包；本任务保持缺失结论，不生成无后缀卡 | `aclnnSlice` 公开入口可达 `StridedSliceV3`；OpDef `AddConfig("ascend950")`、ascend950 binary JSON、arch35 Host/Kernel 与 CMake 依赖闭合 |
| 核心实现 | 缺失，不能从 950 反推 | 共用 arch35 MoveAlign/NDDMA/SIMT/gather |
| BF16 | 不在本卡结论范围 | 支持；公开文档、ACLNN dtype 列表、SoC 枚举闭区间与 2-byte `DtypeByte` binary 匹配闭合 |

直接 `conversion/slice/op_host/arch35` 与 `op_kernel/arch35/slice_*` 是另一条 Slice 算子实现；其 CMake 标为 `ACLNNTYPE aclnn_exclude` 且未被 `aclnnSlice` 的 L0 调用链引用。本卡只记录公开 `aclnnSlice → StridedSliceV3`，不把 direct Slice 的 key、POD 或 body 混入。

## 限制与失败边界

- 仅覆盖公开 `aclnnSlice`，不含 `aclnnSliceV2`，也不含 direct `Slice` 实现。
- `step` 必须正；key 301/302/303 的负步长 body 对本入口不可达。key 201 同样不可达，因为正向 shape overflow 会先强制选择 NDDMA。
- 输入 rank 必须为 1–8，`dim` 合法，out dtype/shape 必须精确一致。空 tensor 在 ACLNN 层短路。
- 950 BF16 依赖 `SocVersion` 枚举顺序：固定仓中 `ASCEND950` 位于实现检查的 `ASCEND910B..ASCEND910E` 闭区间内；迁移时不能把该比较误读为只接受名称含 910 的平台。
- Host 的部分 shape/stride 乘积使用 int64；固定仓未提供所有有符号溢出证明。普通 SIMT 的 uint32 边界有显式 guard，MoveAlign primitive 的 uint16/uint32 字段有退化逻辑。
- POD 宏展开布局、opdev 框架内部 workspace offset/event/cleanup 固定仓不可见，均不得推测。
- router 对未知 key 没有恢复性 fallback；key、POD 与 body 必须成对复用。
- 核心节点固定申报 16 MiB workspace，即使 Device body 未读也不能由调用者擅自省略。
- 按任务要求未运行 validator、测试、编译、真机、数值或性能验证。

## 源码索引

| 链路阶段 | 文件 | 关键符号 | 支撑事实 |
|---|---|---|---|
| 公开契约 | `conversion/slice/docs/aclnnSlice.md` | `aclnnSliceGetWorkspaceSize/aclnnSlice` | 950 平台、参数、shape 公式、正步长契约 |
| ACLNN 声明 | `conversion/slice/op_api/aclnn_slice.h` | 两段式函数原型 | ABI 入口 |
| ACLNN 实现 | `conversion/slice/op_api/aclnn_slice.cpp` | `CheckDtypeValid`, `CheckShape`, `CalculateValuesSlice*`, 两段式入口 | dtype/SoC gate、规范化、空域、Contiguous、StridedSliceV3、ViewCopy |
| L0 launcher | `conversion/strided_slice_v3/op_api/strided_slice_v3.cpp` | `StridedSliceV3`, `StridedSliceV3AiCore` | 标量→INT64 tensor、infer shape、launcher |
| 950 注册 | `conversion/strided_slice_v3/op_host/strided_slice_v3_def.cpp` | `StridedSliceV3` OpDef | ascend950 AICore config |
| 950 binary | `conversion/strided_slice_v3/op_host/config/ascend950/strided_slice_v3_binary.json` | `DtypeByte` configs | x/y 与 INT32/INT64 index binary 匹配 |
| 平台枚举证据 | `tests/ut/op_api/stub/opdev/platform.h` | `op::SocVersion` | 固定仓可见枚举顺序把 `ASCEND950` 放在 BF16 gate 闭区间内 |
| V3 Host | `conversion/strided_slice_v3/op_host/arch35/strided_slice_v3_tiling_arch35.cpp` | `TilingPrepareForStridedSliceV3AscendC`, `TilingForStridedSliceV3`, `MakePerformanceParams` | 平台资源、全 rank 参数、维合并、共用 Tiling 调用 |
| POD | `conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.h` | `BEGIN_TILING_DATA_DEF` 结构 | key-specific 字段与类型 |
| selector/Tiling | `conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.cpp` | `Init`, `CalMaxSplitDim`, `Set*TilingMode`, `RunStrideSliceTiling` | route、分核、UB、workspace、POD producer |
| router | `conversion/strided_slice_v3/op_kernel/strided_slice_v3_apt.cpp` | `strided_slice_v3` | 11 key 分发；本入口 7 key 可达 |
| dtype dispatch | `conversion/strided_slice/op_kernel/arch35/strided_slice.h` | `StridedSlice*Process` | 按 x 字节宽度实例化 body |
| 公共 helper | `conversion/strided_slice/op_kernel/arch35/strided_slice_base.h` | `StridedSliceBase` | POD 解析、core/tile 地址与 tail |
| R-001 | `conversion/strided_slice/op_kernel/arch35/strided_slice_move_align.h` | `StridedSliceMoveAlign` | compact MoveAlign |
| R-002 | `conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_last_dim.h` | `StridedSliceMoveAlignLastDim` | last-dim MoveAlign |
| R-003 | `conversion/strided_slice/op_kernel/arch35/strided_slice_nddma.h` | `StridedSliceNDDMA` | 5D NDDMA |
| R-004 | `conversion/strided_slice/op_kernel/arch35/strided_slice_nddma_last_dim.h` | `StridedSliceNDDMALastDim` | 1D NDDMA |
| R-005 | `conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_two_dim.h` | `StridedSliceMoveAlignTwoDim` | 2D 专用 MoveAlign |
| R-006 | `conversion/strided_slice/op_kernel/arch35/strided_slice_simt.h` | `StridedSliceSIMT`, `StridedSliceDim1..8Compute` | uint32 SIMT 直接 GM 路线 |
| R-007 | `conversion/strided_slice/op_kernel/arch35/strided_slice_move_align_gather.h` | `StridedSliceMoveAlignGather` | gather index、双 Queue、reorder |
| 构建 | `conversion/strided_slice_v3/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `TILING_DIR`, dependency | ascend950/mc62、arch35、复用 strided_slice |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 |
|---|---|---|
| 调用顺序 | `Contiguous → StridedSliceV3 → ViewCopy` | 空 tensor 短路、错误传播和 executor/stream 保序 |
| Route ABI | 7 个可达 key 分别绑定各自 POD/body | 不把 201、301–303 宣称为 ACLNN 可达；不混入 direct Slice key |
| dtype | 核心按 1/2/4/8 字节位搬运；950 BF16 同时命中 ACLNN SoC gate 与 2-byte binary | 保留入口 dtype/SoC 校验；不能仅靠字节匹配扩展未公开 dtype |
| UB | 普通 route depth-2；gather 按 stride 比例分配两组 depth-2 Queue | Queue depth、8192-byte reserve、gather index 上界 |
| 同步 | Queue 保护异步搬运 slot；SIMT 唯一 idx 直接读写 | Free 前必须完成消费；SIMT 保留尾界判断 |
| workspace | 核心节点申报 16 MiB、body 未读 | ABI 申报仍需保留；框架内部切分未知 |
| 精度 | 无数值 cast，输出保持输入位模式 | B8 gather 的 B16 view/pack 只是存储重排 |
| tail | Host POD 给出主尾核和 UB 尾；SIMT 用 idx guard | block/UB 同轴时保留 `ubTailTailFactor` |