---
type: Operator
title: "Lerp 950 ACLNN 算子知识"
description: "说明 aclnnLerp 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/docs/aclnnLerp&aclnnInplaceLerp.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_api/aclnn_lerp_tensor.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_api/aclnn_lerp_tensor.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_api/lerp.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_host/arch35/lerp_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_host/arch35/lerp_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/lerp_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f16_nddma_with_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f32_nddma_without_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f32_nddma_with_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_bf16_nddma_without_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_bf16_nddma_with_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_host/lerp_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_host/config/ascend950/lerp_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/CMakeLists.txt
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling_base.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/broadcast_utils.h
    role: implementation
created_at: "2026-08-25T11:07:40Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Lerp 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnLerpGetWorkspaceSize` / `aclnnLerp`，Tensor weight、非原地、无版本后缀公开 ACLNN 两段式接口 |
| 计算语义 | 广播后逐元素计算 `out=start+weight*(end-start)`；Device 按 `weight<0.5` 选择等价的 start-anchored 或 end-anchored 公式 |
| 输入输出 | `self/end/weight/out` 均为 ND，公开契约 dtype 全部相同且为 FP16/FP32/BF16，shape 三输入可广播到 out |
| 执行类型 | 核心节点为 `NDDMA + AIV_VECTOR`：AIV-only entry 用 NDDMA/连续 DMA 搬入，再由 MicroAPI RegTensor 计算；可选 `Contiguous/ViewCopy` 为框架节点 |
| Kernel 路线 | R-001/R-002 FP16 无/有 NDDMA 外循环；R-003/R-004 FP32 无/有 NDDMA 外循环；R-005/R-006 BF16 无/有 NDDMA 外循环 |
| 核心资源 | GM、4 个 depth-2 TQue、RegTensor/MaskReg、固定 32 B Lerp workspace；低精度 Route 用 FP32 寄存器中间值 |
| 来源闭包 | `source_closed`：公开 Tensor-weight ACLNN→校验/连续化→L0 launcher→950 Broadcast Tiling/POD→六个 ACLNN 可达 key→AIV-only NDDMA+Vector body→ascend950 构建接线闭合；框架 `Contiguous/ViewCopy/CommonOpExecutorRun` 为明确外部边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

三路输入先按尾轴对齐广播到 `outShape`，逻辑结果为 `out_i=start_i+weight_i*(end_i-start_i)`。六条核心 Route 的 Vector body 都先算 `delta=end-start`，再逐 lane 比较 `weight<0.5`：真分支算 `start+weight*delta`，假分支算 `end+(weight-1)*delta`，两式实数域等价。FP16/BF16 先解包并转 FP32，完成比较、减法、融合乘加和选择后以 `NO_SAT + CAST_RINT` 降回原 dtype；FP32 不转换。

本卡只覆盖 Tensor weight 的非原地 `aclnnLerp`。同文件中的 `aclnnInplaceLerp`、标量 weight 的 `aclnnLerps/aclnnInplaceLerps` 是不同入口/语义边界，不并入 Route。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，公开维数 0..8，可非连续 | FP16/FP32/BF16 | 公式中的 start；950 `IsRegBase` 使 BF16 进入支持表 |
| `end` | ND，公开维数 0..8，可非连续 | 与 `self` 相同 | 与 `self/weight` 可广播 |
| `weight` | ND，公开维数 0..8，可非连续 | 与 `self` 相同 | 逐元素权重；API 不限制到 `[0,1]` |
| `out` | ND，公开维数 0..8，可非连续 | 与三输入相同 | view shape 必须恰为三输入广播结果；连续临时结果经 `ViewCopy` 写入该 tensor |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段返回整个 executor 图的 workspace 与 owner；必须传非空 |
| `workspace/executor/stream` | Device 地址/Host handle/stream | `void*` / handle | 第二段交给 `CommonOpExecutorRun`，按同一 stream 执行第一段构建的节点 |

### 入口与版本

固定提交只有无版本后缀 `aclnnLerpGetWorkspaceSize/aclnnLerp`，是 950 实际最新公开 Tensor-weight 版本。公开产品表标记 Ascend 950PR/950DT、A2、A3 及更早产品支持；本固定实现的 `CMakeLists.txt` 却只设置 `SUPPORT_COMPUTE_UNIT "ascend950"` 和 `SUPPORT_TILING_DIR "arch35"`，`Lerp::OpDef` 也只添加 `ascend950` config。本卡据此只发布 950；A2/A3 需另一套固定 Host/Tiling/Kernel 证据才能建无后缀卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnLerpGetWorkspaceSize` 先检查 `workspaceSize/executor`，创建 executor，再按如下顺序执行：

1. `CheckNotNull` 拒绝任一 `self/end/weight/out == nullptr`。
2. `CheckDtypeValid` 在 950 取 `{FP16, FP32, BF16}`，并要求四个 tensor dtype 全相同。因此 Tiling/二进制中 `(BF16,BF16,FP32,BF16)` 与 `(FP16,FP16,FP32,FP16)` 两族对本 ACLNN 不可达。
3. `CheckShape` 先广播 `self,end`，再与 `weight` 广播，并要求结果等于 `out` view shape。只要有 tensor 非连续，就对三输入逐个执行最大 8 维校验；全部连续而 `self` 超过 8 维时源码只告警，但公开契约仍为 0..8 维。
4. `self->IsEmpty()` 时直接成功，不添加 `Contiguous/Lerp/ViewCopy` 节点；其余输入是否为空不单独触发该终点。
5. 三输入依次调用框架 `l0op::Contiguous`；非 ND storage format 只记录告警。随后 `l0op::Lerp` 分配连续广播输出并添加一个 AI Core launcher，最后 `l0op::ViewCopy` 把临时结果提交到 `out`。

每一步失败都立即返回对应 ACLNN 错误；第二段仅运行已经构建成功的 executor。

### 参数、Workspace 与生命周期

核心 Lerp Tiling 固定声明 32 B workspace，entry 将 `workspace` 传给各 Route `Init`，但六个 ACLNN 可达 body 均不读取它。整个 ACLNN 返回值是 `uniqueExecutor->GetWorkspaceSize()`，还包含按输入连续性可能插入的三个 `Contiguous` 与最终 `ViewCopy` 所需空间；这些框架节点的切片公式、内部 POD 和 owner 细节在固定算子仓不可见。executor 从 `uniqueExecutor` 释放给调用者，在第二段执行完成前必须保持有效；用户 workspace 至少为第一段返回的大小，stream 由第二段传入。

核心节点 POD 来自 `LerpTilingData`。平台 producer 读取 AIV 核数和 UB 字节数；Broadcast Tiling 合轴、选 UB 切分和核切分，写 key、blockDim、32 B workspace 与 POD。L0 分配的连续临时 out 由 executor 持有，`ViewCopy` 消费后其生命周期结束；公开 `out` 由调用者持有。

### 启动图与失败传播

```text
aclnnLerpGetWorkspaceSize
  -> [可选] Contiguous(self)   # 框架边界
  -> [可选] Contiguous(end)    # 框架边界
  -> [可选] Contiguous(weight) # 框架边界
  -> l0op::Lerp
       -> ADD_TO_LAUNCHER_LIST_AICORE(Lerp, AI_CORE, start,end,weight -> tempOut)
            -> TilingForLerp / LerpTilingData / blockDim
            -> lerp(start,end,weight,tempOut,workspace,tiling)  # 单个 AIV-only 核心节点
  -> ViewCopy(tempOut, out)     # 框架边界
aclnnLerp(workspace,size,executor,stream)
  -> CommonOpExecutorRun(...,stream)
```

核心节点无跨 Kernel workspace 边；三路 GM 输入与连续 `tempOut` 均为 ND。executor 保证节点按构图顺序在传入 stream 上执行；任一前处理或 launcher 添加失败会阻断后续节点。`lerp` entry 未命中任何已知 key 时直接返回且不写输出；正常 Host 只产生后文六个 key。空 `self` 终点不启动任何节点。

## Route 与参数设计

### 分核与资源计算

Broadcast 公共层先把标量 shape 变为 `[1]`，再按三输入在相邻轴上的广播 pattern 合轴：输入广播轴 stride 写 0，输出为连续 stride。合轴后 `shapeLen<=8`，否则 Tiling 失败。令 `U` 为平台 UB 字节：

```text
alignFactor = 256 bytes * 8 / minDtypeBits(=1) = 2048 elements
FP16/BF16: elemNum = floor((U*8/128)/2048)*2048 = floor((U/16)/2048)*2048
FP32:      elemNum = floor((U*8/256)/2048)*2048 = floor((U/32)/2048)*2048
```

从输出末轴向前累乘，首次超过 `elemNum` 的轴为 `ubSplitAxis`；`ubFormer=elemNum/末端内轴乘积`（一维时直接为 `elemNum`），`ubOuter=ceil(outputDim[axis]/ubFormer)`，`ubTail=outputDim-(ubOuter-1)*ubFormer`。`fusedProduct=ubOuter*prod(outputDims[0:ubSplitAxis])` 是 UB tile 总数；`blockFormer=ceil(fusedProduct/coreNum)`，`blockNum=ceil(fusedProduct/blockFormer)`，`blockTail=fusedProduct-(blockNum-1)*blockFormer`。最后一核跑 `blockTail` 个 tile，其余核跑 `blockFormer` 个。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 NDDMA 无外循环 | `100000001000100` | 四 tensor 全 FP16，且 `shapeLen-ubSplitAxis<=5` | 同 dtype 且大于 5 去 R-002；dtype 不同继续 |
| 2 | R-002 FP16 NDDMA 有外循环 | `100000001001100` | 四 tensor 全 FP16，且 `shapeLen-ubSplitAxis>5` | dtype 不同继续 |
| 3 | R-003 FP32 NDDMA 无外循环 | `200000001000100` | 四 tensor 全 FP32，且 `shapeLen-ubSplitAxis<=5` | 同 dtype 且大于 5 去 R-004；dtype 不同继续 |
| 4 | R-004 FP32 NDDMA 有外循环 | `200000001001100` | 四 tensor 全 FP32，且 `shapeLen-ubSplitAxis>5` | dtype 不同继续 |
| 5 | R-005 BF16 NDDMA 无外循环 | `300000001000100` | 四 tensor 全 BF16，且 `shapeLen-ubSplitAxis<=5` | 同 dtype 且大于 5 去 R-006；dtype 不同失败 |
| 6 | R-006 BF16 NDDMA 有外循环 | `300000001001100` | 四 tensor 全 BF16，且 `shapeLen-ubSplitAxis>5` | 不满足则 Tiling/API 失败 |

```text
API: dtype(self)==dtype(end)==dtype(weight)==dtype(out) and dtype in {FP16,FP32,BF16}
  -> opKey = FP16 ? 1 : FP32 ? 2 : BF16 ? 3 : invalid
  -> BroadcastTiling: axisInsideUB = shapeLen - ubSplitAxis
       axisInsideUB <= 5 -> scheduleKey=100
       axisInsideUB >  5 -> scheduleKey=1100
  -> innerKey = 1*1,000,000 + scheduleKey
  -> tilingKey = opKey*100,000,000,000,000 + innerKey
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `100000001000100` | 950、全 FP16、in-UB 轴数 `<=5` | 单 `lerp`，NDDMA/连续 DMA→AIV Vector→DMA | `LerpF16NddmaWithoutLoops` | 4×depth-2 FP16 Queue；queue producer/consumer 同步 | POD 的 block/UB tail；未知 key 无写出 |
| R-002 | `100000001001100` | 950、全 FP16、in-UB 轴数 `>5` | 同上，NDDMA helper 外层枚举超出 5D 的轴 | `LerpF16NddmaWithLoops` | 同 R-001 | 同 R-001 |
| R-003 | `200000001000100` | 950、全 FP32、in-UB 轴数 `<=5` | 单 `lerp` | `LerpF32NddmaWithoutLoops` | 4×depth-2 FP32 Queue；无 cast | 同上 |
| R-004 | `200000001001100` | 950、全 FP32、in-UB 轴数 `>5` | 单 `lerp`，NDDMA 外层枚举 | `LerpF32NddmaWithLoops` | 同 R-003 | 同上 |
| R-005 | `300000001000100` | 950、全 BF16、in-UB 轴数 `<=5` | 单 `lerp` | `LerpBf16NddmaWithoutLoops` | 4×depth-2 BF16 Queue；FP32 RegTensor 中间值 | 同上 |
| R-006 | `300000001001100` | 950、全 BF16、in-UB 轴数 `>5` | 单 `lerp`，NDDMA 外层枚举 | `LerpBf16NddmaWithLoops` | 同 R-005 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 合轴维数 | `shapeLen<=8` | POD 数组固定 8 项，NDDMA helper 也按 8 维数组编译 | 全部 | Tiling `GRAPH_FAILED` | opbase `DoDimensionCollapse` |
| NDDMA 维数 | `shapeLen-ubSplitAxis<=5` | 单次 `MultiCopyParams<T,5>` 只能表达 5 个 in-UB 轴 | R-001/R-003/R-005 | 有外循环 Route | `BroadcastGetScheduleKey` |
| NDDMA 外循环 | `shapeLen-ubSplitAxis>5` | 枚举被 5D NDDMA 丢出的前置 in-UB 轴 | R-002/R-004/R-006 | 无外循环 Route | `BroadcastNddmaWithLoop` |
| UB 可用量 | `elemNum>0`；FP16/BF16 divisor 128，FP32 divisor 256；2048 元素对齐 | 4 个 depth-2 Queue 必须整体装入 UB，并让 Vector repeat 对齐 | 全部 | Tiling 失败 | `GetComputeMap`, `BroadcastGetMaxElemNum` |
| 连续搬运快捷路 | `outputStride[split]==inputStride[split]` | 无广播扩展时一次 `DataCopyPad` 比 5D NDDMA 配置更直接 | 全部 | 否则 MultiCopy NDDMA | `BroadcastNddmaWithoutLoop/WithLoop` |
| 广播 tile 缓存 | `inputStride[split]==0` 且不是每个 depth-2 槽的首次/新 outer 周期前两 tile | 同一个广播切片可保留在两个 ping/pong Queue 槽 | 全部 | 条件命中才重新 CopyIn | 各 body `CopyIn0/1/2` |
| Vector tail | `size=ubSplitSize*outputStride[split]`，每轮 `UpdateMask(size)` | 最后寄存器只提交有效 lane | 全部 | 无标量 fallback | 各 body `Compute3` |
| 平台/核型 | `ascend950` config；entry 遇 `g_coreType==AIC` 返回，任务声明 `AIV_ONLY` | 实现只有 Vector/NDDMA 数据通路 | 全部 | 非 950 无本固定实现；AIC 无写出 | `lerp`, `Lerp::OpDef`, `CMakeLists.txt` |

### POD/ABI

`LerpTilingData` 由注册宏生成；字段声明顺序和类型可见，但宏展开后的 `sizeof/alignof/offsetof` 在固定源码中不可见，故不推测偏移。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LerpTilingData` | `blockFormer` | `int64_t/64` | 固定仓不可见 | UB tile/普通核 | `DoOpTiling:set_blockFormer` | `Process` 每核循环与起始 tile | 正值 |
| 同上 | `ubFormer` | `int64_t/64` | 固定仓不可见 | split-axis 元素 | `set_ubFormer` | tile size、GM offset、NDDMA | 尾 tile 用 `ubTail` |
| 同上 | `ubOuter` | `int64_t/64` | 固定仓不可见 | tile 个数 | `set_ubOuter` | axes 更新、广播缓存周期 | `ceil(dim/ubFormer)` |
| 同上 | `ubTail` | `int64_t/64` | 固定仓不可见 | split-axis 元素 | `set_ubTail` | `Process` 最后 tile | 1..`ubFormer` |
| 同上 | `blockTail` | `int64_t/64` | 固定仓不可见 | UB tile/尾核 | `set_blockTail` | 最后一核 `ubLoopNum` | 1..`blockFormer` |
| 同上 | `shapeLen` | `int64_t/64` | 固定仓不可见 | 合轴后维数 | `set_shapeLen` | NDDMA config/外循环 | 1..8 |
| 同上 | `ubSplitAxis` | `int64_t/64` | 固定仓不可见 | 轴下标 | `set_ubSplitAxis` | 所有 offset/stride/shape 计算 | 0..`shapeLen-1` |
| 同上 | `dimProductBeforeUbInner` | `int64_t/64` | 固定仓不可见 | UB tile 总数 | `set_dimProductBeforeUbInner` | `BroadcastGetAxesIndices` | 等于 `fusedProduct` |
| 同上 | `elemNum` | `int64_t/64` | 固定仓不可见 | 元素/Queue 槽 | `set_elemNum` | `InitBuffer` 容量 | 2048 元素对齐，必须非零 |
| 同上 | `input0Dims[8]` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `set_input0Dims`，来源 `dims[0]` | 六个可达 body 不读取 | 未用项由 Host 成员零初始化 |
| 同上 | `input1Dims[8]` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `set_input1Dims` | 六个可达 body 不读取 | 同上 |
| 同上 | `input2Dims[8]` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `set_input2Dims` | 六个可达 body 不读取 | 同上 |
| 同上 | `outputDims[8]` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `set_outputDims`，来源 `dims[3]` | axes 进位、NDDMA loop size | 有效前 `shapeLen` 项 |
| 同上 | `input0Strides[8]` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `set_input0Strides` | start CopyIn/GM offset | 广播轴为 0 |
| 同上 | `input1Strides[8]` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `set_input1Strides` | end CopyIn/GM offset | 广播轴为 0 |
| 同上 | `input2Strides[8]` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `set_input2Strides` | weight CopyIn/GM offset | 广播轴为 0 |
| 同上 | `outputStrides[8]` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `set_outputStrides` | UB size、NDDMA dst stride、CopyOut offset | 连续输出 stride |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `100000001000100` | `lerp_tiling_arch35.cpp:LerpTiling::DoOpTiling` + opbase Broadcast Tiling | `LerpTilingData` | `lerp_apt.cpp:lerp` | `LerpF16NddmaWithoutLoops`; opbase `BroadcastNddmaWithoutLoop` | `CMakeLists.txt`, `lerp_def.cpp`, `lerp_binary.json` |
| R-002 | `100000001001100` | 同上 | 同上 | 同上 | `LerpF16NddmaWithLoops`; `BroadcastNddmaWithLoop` | 同上 |
| R-003 | `200000001000100` | 同上 | 同上 | 同上 | `LerpF32NddmaWithoutLoops`; `BroadcastNddmaWithoutLoop` | 同上 |
| R-004 | `200000001001100` | 同上 | 同上 | 同上 | `LerpF32NddmaWithLoops`; `BroadcastNddmaWithLoop` | 同上 |
| R-005 | `300000001000100` | 同上 | 同上 | 同上 | `LerpBf16NddmaWithoutLoops`; `BroadcastNddmaWithoutLoop` | 同上 |
| R-006 | `300000001001100` | 同上 | 同上 | 同上 | `LerpBf16NddmaWithLoops`; `BroadcastNddmaWithLoop` | 同上 |

## Kernel 实现

### 公共执行流程

`lerp` entry 先让 AIC 直接返回，并声明 `KERNEL_TYPE_AIV_ONLY`；随后解析同一 `LerpTilingData`，按 key 构造唯一 Route 类。每个类持有调用者传入的 `TPipe`，把 start/end/weight/out 绑定为 GM tensor，workspace 参数不消费。

每核从线性 tile 起点 `blockFormer*blockIdx` 解码 `axesIndices`。每 tile 依次为 weight、end、start 分配 VECIN Queue 槽并用 NDDMA/连续 DMA 广播搬入；Vector 阶段 DeQue 三输入、Alloc 输出，按 `VECTOR_REG_WIDTH/4` 个 FP32 lane 分段执行双公式和 Select；之后释放输入、EnQue 输出，CopyOut 以单 block `DataCopyPad` 写连续 GM。四个 Queue 的 depth 都是 2；Queue 协议提供搬运→计算→写回及槽复用顺序。

### Route R-001：FP16 NDDMA 无外循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_lerp_tensor.cpp` | `CheckDtypeValid` | 四 tensor dtype 相同且 950 支持 FP16 | `l0op::Lerp` |
| Tiling | `op_host/arch35/lerp_tiling_arch35.cpp` | `GetOpKey/GenerateTilingKey` | 全 FP16→opKey 1 | key 高位 1 |
| 公共 Tiling | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | `shapeLen-ubSplitAxis<=5` | schedule 100 |
| Router | `op_kernel/lerp_apt.cpp` | `lerp` | key `100000001000100` | `LerpF16NddmaWithoutLoops` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Lerp -> ADD_TO_LAUNCHER_LIST_AICORE -> lerp(..., LerpTilingData)
  -> LerpF16NddmaWithoutLoops::Init
  -> Process
     -> CopyIn0(weight) -> CopyIn1(end) -> CopyIn2(start)
     -> Compute3(FP16 unpack -> FP32 dual lerp -> RINT FP16)
     -> CopyOut4
```

##### 源码等价伪码

```text
tiles = lastCore ? blockTail : blockFormer
axes = decode(blockFormer*blockIdx, outputDims, splitAxis, totalTiles)
for tile in [0, tiles):
  if tile>0: axes = increment(axes, outputDims, splitAxis, ubOuter)
  n = axes[splitAxis]==ubOuter-1 ? ubTail : ubFormer
  for input in [weight,end,start]:
    if inputStride[splitAxis]!=0 or tile<=1 or globalTile%ubOuter<=1:
      if outputStride[splitAxis]!=inputStride[splitAxis]: MultiCopy5D_without_loop(input -> UB)
      else: DataCopyPad(input[gmOffset] -> UB, n*inputStride[splitAxis]*2 bytes)
  valid = n*outputStride[splitAxis]
  for regChunk: mask=UpdateMask(valid); w,e,s=FP32(load_fp16); d=e-s
                a=s+w*d; b=e+(w-1)*d; y=select(w<0.5,a,b); store_rint_fp16(y)
  DataCopyPad(UB -> y[outputOffset], valid*2 bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs | `BroadcastGetGmOffset(axes,inputStride,splitAxis,ubFormer)` | 广播后当前 tile 有效域 | FP16/ND | caller/Contiguous | CopyIn0/1/2 | caller/executor 至核心节点结束 | stride 0 表示广播 |
| `queIn0/1/2` | Queue 槽 | 每槽 `elemNum*2 B` | FP16 | NDDMA/DataCopyPad | `Compute3` | TPipe，Alloc→EnQue→DeQue→Free | depth 2；广播 tile 可复用槽内容 |
| RegTensor | 每 chunk `i*VECTOR_REG_WIDTH/4` 元素 | 固定寄存器宽，tail mask | FP16+FP32 | MicroAPI load/cast | 算术/Select/store | 单 Vector loop | 无跨迭代存活 |
| `queOut0` | Queue 槽 | 每槽 `elemNum*2 B` | FP16 | `Compute3` | `CopyOut4` | TPipe，Alloc→EnQue→DeQue→Free | depth 2 |
| GM out | `BroadcastGetGmOffset(axes,outputStride,...)` | `valid*2 B` | FP16/连续 ND | CopyOut | ViewCopy/out | executor 临时 | 单 block DataCopyPad |
| workspace | entry 参数 | 32 B Tiling 声明 | bytes | Host | body 不读取 | executor | 未复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inW["queIn0 weight · depth=2 · 2*elemNum*2 B"]:1
  inE["queIn1 end · depth=2 · 2*elemNum*2 B"]:1
  inS["queIn2 start · depth=2 · 2*elemNum*2 B"]:1
  out["queOut0 y · depth=2 · 2*elemNum*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4 queues * depth2 * elemNum * 2 B = 16*elemNum B <= platform UB
dynamic_ub = N/A
workspace_total(core Lerp) = 32 B; body reads 0 B
=> elemNum=floor((UB/16)/2048)*2048
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三个输入 Queue 的 ping/pong 槽 | CopyIn0/1/2 | `Compute3` | 两次 tile 后同槽 | Queue EnQue/DeQue/Free；depth 2 | split stride 0 时每个 outer 周期前两 tile刷新，之后复用相同广播值 |
| 输出 Queue 槽 | `Compute3` | `CopyOut4` | Free 后下一 tile | EnQue/DeQue/Free | CopyOut 完成前不可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | TQue | NDDMA/DMA | DeQue | 三输入槽 | NDDMA 5D；尾 tile 用 `ubTail` |
| Compute | `UpdateMask<float>` | size counter | 所有 MicroAPI | 寄存器 lane | FP16→FP32；无效 lane ZEROING |
| 公式选择 | `CompareScalar LT 0.5` + `MulAddDst` + `Select` | weight/delta | output reg | 数值路径 | `w<0.5` 真，否则 end-anchored |
| 降精度 | `Cast NO_SAT/ZEROING/CAST_RINT` | FP32 result | packed FP16 store | 输出 lane | 仅一次末端降精度 |
| Compute→CopyOut | TQue | EnQue | DeQue | 输出槽 | `blockLen=valid*2 B` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/lerp_tiling_arch35.cpp` | `GetOpKey`, `DoOpTiling` | FP16 key、elemNum、block/tile/POD |
| Kernel entry/router | `op_kernel/lerp_apt.cpp` | `lerp` key `100000001000100` | AIV-only 到类映射 |
| Device body/helper | `op_kernel/arch35/lerp_f16_nddma_without_loops.h` | `Process/CopyIn*/Compute3/CopyOut4` | 5D NDDMA、FP32 双公式、tail mask |

### Route R-002：FP16 NDDMA 有外循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `aclnn_lerp_tensor.cpp`, `lerp_tiling_arch35.cpp` | `CheckDtypeValid/GetOpKey` | 全 FP16 | opKey 1 |
| 公共 Tiling | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | `shapeLen-ubSplitAxis>5` | schedule 1100 |
| Router | `lerp_apt.cpp` | `lerp` | `100000001001100` | `LerpF16NddmaWithLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpF16NddmaWithLoops::Init -> Process
  -> CopyIn0/1/2 -> BroadcastNddmaWithLoop -> MultiCopy<T,5> loop
  -> Compute3 -> CopyOut4
```

##### 源码等价伪码

```text
for each core tile (block/tail logic同R-001):
  for each input:
    base = BroadcastGetGmOffset(...)
    if broadcast expansion:
      params5D = last five in-UB axes
      outerCount = product(outputDims[splitAxis+1 : shapeLen-5]) * ubSplitSize
      for j in [0,outerCount): MultiCopy5D(input[base+gmOff[j]] -> UB[ubOff[j]])
    else: one DataCopyPad
  execute same FP32 dual-form Vector pipeline and FP16 RINT store as R-001
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM↔UB | base plus `BroadcastGetNddmaOffset` | 5D params × `outerCount` | FP16 | `BroadcastNddmaWithLoop` | Compute3 | 当前 CopyIn | 超出 5D 的轴显式循环；连续时单 DMA |
| 4×TQue | 同 R-001 | 每槽 `elemNum*2 B` | FP16 | CopyIn/Compute | Compute/CopyOut | TPipe | depth 2 |
| RegTensor/GM out/workspace | 同 R-001 | 同 R-001 | FP32 中间/FP16 out | Compute3 | CopyOut/ViewCopy | 当前 chunk/executor | tail mask |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inW["weight · depth=2 · 2*elemNum*2 B"]:1
  inE["end · depth=2 · 2*elemNum*2 B"]:1
  inS["start · depth=2 · 2*elemNum*2 B"]:1
  out["out · depth=2 · 2*elemNum*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16*elemNum B<=UB; dynamic_ub=N/A; core workspace=32 B
NDDMA额外外循环只复用同一输入 Queue 槽，不增加 UB slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 输入/输出 Queue 槽 | CopyIn/Compute | Compute/CopyOut | depth-2 周期 | Queue 协议 | 与 R-001 相同；外循环必须在 EnQue 前完成整 tile |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA 外循环 | 顺序 `MultiCopy<T,5>` 调用 | 各 outer slice | Queue EnQue | 同一 UB tile 的不同 offset | offset 由 input/output stride 分别计算 |
| Vector/CopyOut | Queue+mask | CopyIn/Compute | Compute/CopyOut | tile | 精度和 tail 与 R-001 相同 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | 大于 5 个 in-UB 轴选 1100 |
| Kernel entry/router | `op_kernel/lerp_apt.cpp` | key `100000001001100` | 映射 with-loops body |
| Device body/helper | `lerp_f16_nddma_with_loops.h`, opbase `broadcast_utils.h` | `CopyIn*`, `BroadcastNddmaWithLoop` | 5D 参数加显式外循环 |

### Route R-003：FP32 NDDMA 无外循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `aclnn_lerp_tensor.cpp`, `lerp_tiling_arch35.cpp` | dtype checks/`GetOpKey` | 全 FP32 | opKey 2 |
| 公共 Tiling | opbase `BroadcastGetScheduleKey` | axisInsideUB | `<=5` | schedule 100 |
| Router | `lerp_apt.cpp` | key branch | `200000001000100` | `LerpF32NddmaWithoutLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpF32NddmaWithoutLoops::Init -> Process
  -> CopyIn0(weight)/CopyIn1(end)/CopyIn2(start) [without-loop NDDMA]
  -> Compute3(FP32 load/dual lerp/select/FP32 store) -> CopyOut4
```

##### 源码等价伪码

```text
tile/core/address/CopyIn同R-001，但dtype为float、每元素4 B
d=end-start
a=start+weight*d
b=end+(weight-1)*d
out=select(weight<0.5,a,b)  # 全程FP32，无Cast
tail=UpdateMask(valid); DataCopyPad blockLen=valid*4 B
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs/out | 同 R-001 | tile `valid` | FP32/ND | caller/Kernel | Kernel/ViewCopy | executor | 4 B/元素 |
| 4×TQue | Queue 槽 | 每槽 `elemNum*4 B` | FP32 | CopyIn/Compute | Compute/CopyOut | TPipe | depth 2 |
| RegTensor | chunk | Vector width/4 lanes | FP32 | load | arithmetic/store | loop | 无 cast |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inW["weight · depth=2 · 2*elemNum*4 B"]:1
  inE["end · depth=2 · 2*elemNum*4 B"]:1
  inS["start · depth=2 · 2*elemNum*4 B"]:1
  out["out · depth=2 · 2*elemNum*4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=4*2*elemNum*4 B=32*elemNum B<=UB
elemNum=floor((UB/32)/2048)*2048; dynamic_ub=N/A; core workspace=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| depth-2 Queue 槽 | CopyIn/Compute | Compute/CopyOut | 两 tile 后 | EnQue/DeQue/Free | 同 R-001 广播缓存条件 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→Vector→DMA | TQue | CopyIn/Compute | Compute/CopyOut | Queue 槽 | 全程 FP32；`UpdateMask` 保护 tail |
| 公式 | FP32 `Sub/MulAddDst/Select` | 三输入 | FP32 out | lane | 阈值 0.5；无 round cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp` | `GetComputeMap(OP_KEY_2)` | divisor 256、opKey 2 |
| Kernel entry/router | `lerp_apt.cpp` | key `200000001000100` | FP32 without-loop 类 |
| Device body/helper | `lerp_f32_nddma_without_loops.h` | `Compute3` | FP32 原生双公式，无 Cast |

### Route R-004：FP32 NDDMA 有外循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `aclnn_lerp_tensor.cpp`, `lerp_tiling_arch35.cpp` | dtype checks/`GetOpKey` | 全 FP32 | opKey 2 |
| 公共 Tiling | opbase `BroadcastGetScheduleKey` | `shapeLen-ubSplitAxis>5` | schedule 1100 | with-loop |
| Router | `lerp_apt.cpp` | key branch | `200000001001100` | `LerpF32NddmaWithLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpF32NddmaWithLoops::Init -> Process
  -> CopyIn* -> BroadcastNddmaWithLoop<float,float> -> Compute3 -> CopyOut4
```

##### 源码等价伪码

```text
按R-002枚举超出5D的in-UB轴并填满FP32 UB tile
按R-003在FP32寄存器执行weight阈值双公式
以valid*4 B单block搬出，尾lane由UpdateMask保护
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| NDDMA GM↔UB | base + per-outer gm/ub offset | last 5 dims × outerCount | FP32 | helper | Compute3 | CopyIn | 同 R-002 |
| 4×TQue/RegTensor/GM out | 同 R-003 | 同 R-003 | FP32 | CopyIn/Compute | Compute/CopyOut | TPipe/executor | depth 2/tail mask |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inW["weight · 2*elemNum*4 B"]:1
  inE["end · 2*elemNum*4 B"]:1
  inS["start · 2*elemNum*4 B"]:1
  out["out · 2*elemNum*4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=32*elemNum B<=UB; dynamic_ub=N/A; workspace=32 B
with-loop NDDMA不增加常驻buffer
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue 槽 | CopyIn/Compute | Compute/CopyOut | depth-2 周期 | Queue | 外循环完成后才 EnQue |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA | 顺序 MultiCopy + Queue | GM slices | Compute | UB tile | 5D tail由loop参数，Vector tail由mask |
| Compute | FP32 MicroAPI | Queue | CopyOut | output | 与 R-003 相同 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | schedule 1100 |
| Kernel entry/router | `lerp_apt.cpp` | key `200000001001100` | FP32 with-loop 类 |
| Device body/helper | `lerp_f32_nddma_with_loops.h`, `broadcast_utils.h` | `Compute3`, `BroadcastNddmaWithLoop` | 外循环搬运+FP32计算 |

### Route R-005：BF16 NDDMA 无外循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lerp_tensor.cpp` | `GetDtypeSupportList` | `IsRegBase(npuArch)` 支持 BF16，四 tensor 同 dtype | 进入 Lerp |
| Tiling | `lerp_tiling_arch35.cpp` | `GetOpKey` | 全 BF16 | opKey 3 |
| 公共 Tiling/Router | opbase key helper；`lerp_apt.cpp` | axisInsideUB/key | `<=5`, `300000001000100` | `LerpBf16NddmaWithoutLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpBf16NddmaWithoutLoops::Init -> Process
  -> CopyIn* [BF16 without-loop NDDMA]
  -> Compute3(BF16 unpack -> FP32 dual lerp -> RINT BF16)
  -> CopyOut4
```

##### 源码等价伪码

```text
搬运/地址/核循环同R-001，元素类型bfloat16_t
w,e,s = Cast<float>(DIST_UNPACK_B16 load)
d=e-s; a=s+w*d; b=e+(w-1)*d; r=Select(w<0.5,a,b)
store DIST_PACK_B32(Cast<bfloat16_t,CAST_RINT,NO_SAT>(r))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/TQue | 同 R-001 | 每槽 `elemNum*2 B` | BF16/ND | CopyIn | Compute | caller/TPipe | depth 2 |
| RegTensor | chunk | Vector width/4 FP32 lanes | BF16 load、FP32 intermediates | load/cast | math/cast | Vector loop | RINT末端降精度 |
| GM out/workspace | 同 R-001 | `valid*2 B` / 32 B | BF16/bytes | CopyOut/Host | ViewCopy/body不读workspace | executor | tail mask |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inW["weight · depth=2 · 2*elemNum*2 B"]:1
  inE["end · depth=2 · 2*elemNum*2 B"]:1
  inS["start · depth=2 · 2*elemNum*2 B"]:1
  out["out · depth=2 · 2*elemNum*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16*elemNum B<=UB; elemNum=floor((UB/16)/2048)*2048
dynamic_ub=N/A; core workspace=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue ping/pong | CopyIn/Compute | Compute/CopyOut | 两 tile 后 | Queue | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy/Compute | Queue + mask | NDDMA | Vector/CopyOut | UB/GM | tail同R-001 |
| 精度 | BF16 unpack→FP32→BF16 RINT NO_SAT | input regs | packed output | lane | 中间算术全 FP32 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `aclnn_lerp_tensor.cpp`, `lerp_tiling_arch35.cpp` | `GetDtypeSupportList/GetOpKey/GetComputeMap` | 950 BF16 可达、divisor 128 |
| Kernel entry/router | `lerp_apt.cpp` | key `300000001000100` | BF16 without-loop 类 |
| Device body/helper | `lerp_bf16_nddma_without_loops.h` | `Compute3/castTrait*` | FP32 中间与 BF16 RINT |

### Route R-006：BF16 NDDMA 有外循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `aclnn_lerp_tensor.cpp`, `lerp_tiling_arch35.cpp` | BF16 support/`GetOpKey` | 全 BF16 | opKey 3 |
| 公共 Tiling | opbase `BroadcastGetScheduleKey` | `shapeLen-ubSplitAxis>5` | schedule 1100 | with-loop |
| Router | `lerp_apt.cpp` | key branch | `300000001001100` | `LerpBf16NddmaWithLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpBf16NddmaWithLoops::Init -> Process
  -> CopyIn* -> BroadcastNddmaWithLoop<bfloat16_t,bfloat16_t>
  -> Compute3(FP32 intermediate) -> CopyOut4
```

##### 源码等价伪码

```text
按R-002以outerCount次5D MultiCopy组成完整BF16 UB tile
按R-005执行BF16->FP32双公式->RINT BF16
按POD尾块和Vector mask仅写有效元素
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| NDDMA GM↔UB | base+outer offsets | 5D params×outerCount | BF16 | with-loop helper | Compute | CopyIn | 不增加额外 UB |
| 4×TQue/RegTensor/out | 同 R-005 | 同 R-005 | BF16+FP32 | CopyIn/Compute | Compute/CopyOut | TPipe | depth 2/mask |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inW["weight · 2*elemNum*2 B"]:1
  inE["end · 2*elemNum*2 B"]:1
  inS["start · 2*elemNum*2 B"]:1
  out["out · 2*elemNum*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16*elemNum B<=UB; dynamic_ub=N/A; workspace=32 B
outer NDDMA slices share the same allocated Queue slot
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue 槽 | MultiCopy/Compute | Compute/CopyOut | depth-2 周期 | Queue | 所有 outer slice 完成后 EnQue |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA/Queue | 顺序 MultiCopy + EnQue | GM | Compute | 完整 tile | 5D params与outer offset共同覆盖有效域 |
| Vector/CopyOut | mask+Queue | Compute | CopyOut | output | 与 R-005 相同 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | 大维选 with-loop |
| Kernel entry/router | `lerp_apt.cpp` | key `300000001001100` | BF16 with-loop 类 |
| Device body/helper | `lerp_bf16_nddma_with_loops.h`, `broadcast_utils.h` | `Compute3`, `BroadcastNddmaWithLoop` | BF16外循环搬运、FP32中间计算 |

## 算子特定优化

### Trick T-001：按 weight 阈值切换锚点的双公式

#### 触发与源码依据

- 适用 Route：R-001..R-006
- 触发条件：每个有效 lane 的 `weight<0.5` 选 start-anchored，否则选 end-anchored
- Golden 来源：[`math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h:159-176`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h#L159-L176)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::CompareScalar<float, AscendC::CMPMODE::LT>(preg2, vreg1, static_cast<float>(0.5),
                                                              preg0);
AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg6, vreg3, vreg5, preg0);
AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg1, vreg6, preg0);
AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg8, vreg1, vreg7, preg0);
AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg3, vreg8, vreg6, preg0);
AscendC::MicroAPI::Select<float>(vreg9, vreg5, vreg3, preg2);
```

#### 机制与设计原因

`vreg1=weight`、`vreg3=end`、`vreg5=start`，`vreg6=end-start`，`vreg7=1`。第一条融合乘加写出 `start+weight*delta`；第二条先算 `weight-1`，再以 end 为累加目标写出 `end+(weight-1)*delta`。阈值选择使接近 0 的权重锚定 start、接近 1 的权重锚定 end；源码可证明它改变浮点运算顺序并保留末端单次 cast，固定仓未给出误差上界或真机收益。

#### 迁移边界

- 适用：广播逐元素线性插值，目标硬件支持对应 Compare/Select/MulAddDst 路径。
- 前提：两分支使用同一 FP32 `delta`，低精度输入须先升到 FP32；tail mask 必须覆盖全部分支指令。
- 失效：若目标语义规定不同舍入顺序、需要复数或整数插值，不能直接复用；本实现没有其它数值 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时不得把 `Select` 条件反向却不交换两个结果，也不得在 FP16/BF16 中间步骤提前降精度。

### Trick T-002：用 depth-2 Queue 缓存 split-axis 广播 tile

#### 触发与源码依据

- 适用 Route：R-001..R-006 的任一输入
- 触发条件：该输入 `inputStride[ubSplitAxis]==0`；仅各 depth-2 槽首次及新 `ubOuter` 周期前两 tile 执行实际 CopyIn
- Golden 来源：[`math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h:76-88`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h#L76-L88)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bufferIn0_ = queIn0_.AllocTensor<half>();
if ((tilingDataPtr_->input2Strides[tilingDataPtr_->ubSplitAxis] != 0) ||
    (ubLoopIdx <= 1 ||
     (AscendC::GetBlockIdx() * tilingDataPtr_->blockFormer + ubLoopIdx) % tilingDataPtr_->ubOuter <= 1)) {
    Ops::Base::BroadcastNddmaWithoutLoop(inputGmWeight_, bufferIn0_, tilingDataPtr_->outputDims,
                              tilingDataPtr_->outputStrides, tilingDataPtr_->input2Strides, axesIndices,
                              tilingDataPtr_->ubSplitAxis, tilingDataPtr_->shapeLen, ubSplitSize,
                              tilingDataPtr_->ubFormer);
}
queIn0_.EnQue<half>(bufferIn0_);
```

#### 机制与设计原因

split-axis stride 为 0 表示沿该轴广播，同一 `ubOuter` 周期各 tile 读取相同 GM slice。Queue 深度为 2，代码让 ping/pong 两个物理槽在最初两 tile 或新周期前两 tile各填一次，之后 `AllocTensor` 取得的槽保留旧广播值并直接 EnQue，减少重复 GM→UB/NDDMA 搬运。条件中的全局 tile 模 `ubOuter` 保证跨 core/tile 编号仍在周期边界刷新。

#### 迁移边界

- 适用：输入在切分轴 stride 为 0、Queue 槽内容在 Free 后仍可被下一次 Alloc 复用的同型广播流水。
- 前提：Queue depth 固定为 2；每个 outer 周期广播源相同；consumer 在覆盖前完成读取。
- 失效：非广播 stride 每 tile 都必须搬运；改变 Queue depth 或 slot 分配策略时必须重推刷新次数。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码中的真实非适用路径是 `inputStride[splitAxis]!=0`，此时条件无条件执行 CopyIn；若仍跳过，会复用前一 tile 的不同地址数据并产生错误输出。

### Trick T-003：以 5D NDDMA 能力为界拆分 schedule

#### 触发与源码依据

- 适用 Route：R-001/R-003/R-005 对比 R-002/R-004/R-006
- 触发条件：`shapeLen-ubSplitAxis<=5` 直接一组 5D 参数；大于 5 时 Host 选 with-loop body
- Golden 来源：[`src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp:31-37`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp#L31-L37)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t BroadcastGetScheduleKey(uint32_t axisInsideUB)
{
    if (axisInsideUB <= MAX_NDDMA_DIM) {
        return SCHEDULE_KEY_1;
    }
    return SCHEDULE_KEY_2;
}
```

#### 机制与设计原因

Host 用真实 in-UB 轴数生成不同 key，Device 无需在单一 body 内动态判断 NDDMA 维数。without-loop 直接构造 `MultiCopyParams<T,5>`；with-loop 固定最后 5 轴为 NDDMA 参数，并显式枚举更外侧轴的 GM/UB offset。这样扩展到最多 8 个合轴维度而不改变 5D NDDMA ABI。

#### 迁移边界

- 适用：硬件多维搬运维数固定、但逻辑广播 rank 更高的算子。
- 前提：Host key 与两个 entry body一一对应，POD 中 shape/stride 数组覆盖全部逻辑轴。
- 失效：目标硬件最大 NDDMA 维数或维度顺序改变时必须同步修改 schedule guard 与 helper。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码的真实错误终点是 Host 产生未在 entry 注册的 key时 entry直接返回；不得只改 Host schedule key而不增加匹配 body。

## 模板与实现对比

| 对比维度 | R-001/R-002 FP16 | R-003/R-004 FP32 | R-005/R-006 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | opKey 1；轴数决定 100/1100 | opKey 2；轴数决定 100/1100 | opKey 3；轴数决定 100/1100 | API 同 dtype + Host NDDMA 维数 |
| 模板与 dtype | `half` GM/UB，FP32 regs | `float` GM/UB/regs | `bfloat16_t` GM/UB，FP32 regs | 公开 dtype |
| 循环与分核 | 两个 schedule 仅 NDDMA 内部是否外循环不同 | 同左 | 同左 | `shapeLen-ubSplitAxis` 与 5 比较 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4×depth-2 TQue；无 TBuf/dynamic UB/window | 同结构，Queue 槽字节翻倍 | 同 FP16 容量 | dtype 字节与 MicroAPI load/store distribution |
| 内存总占用与复用 | `16*elemNum B`，广播槽缓存 | `32*elemNum B`，广播槽缓存 | `16*elemNum B`，广播槽缓存 | divisor 128/256 保证不超 UB |
| 搬运/Vector/SIMT/NDDMA/通信 API | 5D MultiCopy或DataCopyPad；UNPACK/PACK | 同搬运；NORM B32 load/store | 5D MultiCopy；UNPACK/PACK | 无 SIMT/通信 |
| 同步、精度与 tail | Queue；FP32中间；RINT；mask | Queue；全FP32；mask | Queue；FP32中间；RINT；mask | dtype与最后一 tile |
| fallback/失败边界 | 无/有外循环互为选择，不是运行时 fallback | 同左 | 同左 | 未知 key无输出；Tiling失败不 launch |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档标支持；固定 `math/lerp/` 无 A2 Host/Tiling/Kernel 接线，未建卡 | 公开文档标支持；固定实现闭包缺失，未建卡 | 公开支持且固定实现闭合 |
| Host/Launcher | 固定仓不可见 | 固定仓不可见 | `arch35/LerpTiling` + opbase Broadcast Tiling，六个 ACLNN 可达 key |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | `lerp_apt` AIV-only；NDDMA + MicroAPI Vector；4×depth-2 TQue |

## 限制与失败边界

- 本卡严格排除 `aclnnInplaceLerp` 和 `aclnnLerps/aclnnInplaceLerps`。虽然它们最终可能调用同一个 L0 `Lerp`，其输出 owner、原地约束或标量物化/额外 Cast 图不同。
- 公开契约要求 ND、0..8 维、四 tensor dtype 相同。API 对全连续 `self` 超 8 维只告警、对非 ND 只告警，但不应据此扩展公开契约；Tiling 合轴结果仍必须不超过 8。
- 仅 `self->IsEmpty()` 触发 API 的空图成功终点；该路径不添加 ViewCopy，固定源码没有进一步说明调用者 out 的既有内容。
- Tiling 支持的 mixed-weight opKey 4/5 与 binary config 条目被 `CheckDtypeValid` 阻断，因此不是 `aclnnLerp` Route；不能用它们宣称公开 API 支持 weight FP32 与低精度 start/end 混用。
- 核心 Tiling 声明 32 B workspace但 body不读；整个 ACLNN workspace 还可能包括框架 `Contiguous/ViewCopy`。框架内部精确切片、事件与 POD 在固定算子仓不可见。
- `LerpTilingData` 宏展开的 `sizeof/alignof/offsetof` 固定仓不可见；只可依赖字段声明顺序、类型和同一注册/解析宏，不得手写推测 ABI。
- 公共 Broadcast 直接依赖来自固定 `opbase@d574f5571391390f8b619c6a134b65c819842318`；AscendC primitive 的硬件内部实现是外部边界。
- entry 的未知 key 和 AIC 分支都直接返回不写输出；正常 950 Host 通过 key/核型约束避免这两个终点。本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/lerp/docs/aclnnLerp&aclnnInplaceLerp.md` | 产品表、参数表 | 950公开支持、dtype/shape/layout/错误语义 |
| 入口声明/约束 | `math/lerp/op_api/aclnn_lerp_tensor.h` | `aclnnLerp*` | 两段式函数原型与计算图 |
| 入口实现 | `math/lerp/op_api/aclnn_lerp_tensor.cpp` | `CheckParams/CalculateResult/aclnnLerp*` | 校验、空域、连续化、ViewCopy、executor |
| 入口实现 | `math/lerp/op_api/lerp.cpp` | `l0op::Lerp` | 广播输出分配和单 AI Core launcher |
| Host/Tiling/Launcher | `math/lerp/op_host/arch35/lerp_tiling_arch35.h` | `LerpTilingData/LerpTiling` | POD 字段与 Tiling 类 |
| Host/Tiling/Launcher | `math/lerp/op_host/arch35/lerp_tiling_arch35.cpp` | `GetOpKey/GetComputeMap/DoOpTiling/PostTiling` | key、UB参数、POD、blockDim、32 B workspace |
| Host/Tiling/Launcher | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling_base.h` | `BroadcastTilingData/Params` | 公共参数含义与常量 |
| Host/Tiling/Launcher | `src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey/GetMaxElemNum` | NDDMA 维数分流和 elemNum |
| Host/Tiling/Launcher | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse/GetBlockSplitFactor/DoBrodcastTiling` | 合轴、stride、tile/core 公式 |
| Kernel entry/body | `math/lerp/op_kernel/lerp_apt.cpp` | `lerp` | 十 key router；ACLNN 可达六 key；AIV-only |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h` | `LerpF16NddmaWithoutLoops` | R-001 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f16_nddma_with_loops.h` | `LerpF16NddmaWithLoops` | R-002 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f32_nddma_without_loops.h` | `LerpF32NddmaWithoutLoops` | R-003 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f32_nddma_with_loops.h` | `LerpF32NddmaWithLoops` | R-004 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_bf16_nddma_without_loops.h` | `LerpBf16NddmaWithoutLoops` | R-005 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_bf16_nddma_with_loops.h` | `LerpBf16NddmaWithLoops` | R-006 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/broadcast_utils.h` | `BroadcastNddmaWithoutLoop/WithLoop` | NDDMA 参数、地址、连续 DMA fallback |
| 构建接线 | `math/lerp/op_host/lerp_def.cpp` | `Lerp::Lerp` | ascend950 config 和 `lerp_apt` 文件名 |
| 构建接线 | `math/lerp/op_host/config/ascend950/lerp_binary.json` | `op_list` | 950 binary dtype组合 |
| 构建接线 | `math/lerp/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | 仅 ascend950/arch35 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 9 个 int64 标量+8个 int64[8] 数组的字段顺序 | 宏生成 ABI、字段类型/顺序、数组上限 8；offset 不可猜 | `LerpTilingData` |
| 调用顺序 | 三路 CopyIn→Compute→CopyOut；API Contiguous→Lerp→ViewCopy | weight/end/start Queue 与 input2/1/0 stride 映射不可交换 | `CalculateResult`, 各 body `Process` |
| workspace/内存 | 核心 32 B workspace未读；4×depth-2 Queue | depth 必须计入 UB；ACLNN总 workspace 必须用第一段返回值 | `GetWorkspaceSize`, 各 body `Init` |
| 同步 | Queue EnQue/DeQue/Free 管理四级流水 | 广播缓存依赖 depth=2及槽内容保留，不能提前覆盖 | `CopyIn*/Compute3/CopyOut4` |
| 精度与尾块 | 低精度升 FP32、阈值双公式、末端 RINT；FP32原生；UpdateMask tail | `weight<0.5` 与 Select输入顺序、`NO_SAT/CAST_RINT`、`ubTail` | 各 body `Compute3/CopyOut4` |
| 失败边界 | dtype/shape/API错误阻断 launch；轴数/UB失败返回；未知 key无写出 | mixed dtype key不属于本ACLNN；AIC不可执行 | `CheckParams`, `DoDimensionCollapse`, `lerp` |