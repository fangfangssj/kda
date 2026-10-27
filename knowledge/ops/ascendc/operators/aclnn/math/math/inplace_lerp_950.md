---
type: Operator
title: "InplaceLerp 950 ACLNN 算子知识"
description: "说明 aclnnInplaceLerp 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
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
created_at: "2026-08-25T11:09:23Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceLerp 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceLerpGetWorkspaceSize` / `aclnnInplaceLerp`，Tensor weight 语义的公开 ACLNN 两段式接口 |
| 计算语义 | `selfRef <- selfRef + weight * (end - selfRef)`，逐元素 broadcast，最终结果写回 `selfRef` |
| 输入输出 | `selfRef/end/weight` 为 ND，公开 ACLNN 要求三者同 dtype，FP16/FP32/BF16，0..8 维，broadcast 结果 shape 必须等于 `selfRef` |
| 执行类型 | 单个 AIV-only Device 节点：CopyIn 用 NDDMA/连续 DMA，Compute 用 AIV MicroAPI Vector；无 Cube/AIC/MIX_CV 计算节点 |
| Kernel 路线 | R-001/R-002 FP16，R-003/R-004 FP32，R-005/R-006 BF16；每种 dtype 按 UB 内轴数 `<=5` / `>5` 分为 NDDMA without-loops / with-loops |
| 核心资源 | GM；4 个 depth-2 `TQue`；FP32 `RegTensor`；950 UB；Lerp Tiling 固定 32 B system workspace（body 不读） |
| 来源闭包 | `source_closed`：公开 ACLNN→executor 前后处理→`l0op::Lerp`→950 Broadcast Tiling/POD→单个 NDDMA+AIV Vector body→`ascend950` 构建接线已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 broadcast 输出域中的每个位置 `i`，语义为 `y_i=start_i+weight_i*(end_i-start_i)`，且 `y` 通过 `ViewCopy` 提交回 `selfRef`。Device 为稳定性按 `weight<0.5` 分支：真分支算 `start+weight*(end-start)`，假分支算 `end+(weight-1)*(end-start)`，两式数学等价。FP16/BF16 输入先升为 FP32 计算，最后以 `CAST_RINT` 降回原 dtype；FP32 全程不 cast。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND，0..8 维，支持非连续 Tensor | FP16/FP32/BF16 | 输入 start 与原地输出 owner；与 `end/weight` broadcast 后 shape 必须仍等于自身 view shape |
| `end` | ND，0..8 维，支持非连续 Tensor | 与 `selfRef` 相同 | 公开接口不做 dtype 提升，不同 dtype 直接失败 |
| `weight` | ND，0..8 维，支持非连续 Tensor | 与 `selfRef` 相同 | 本卡仅 Tensor weight，不包含 scalar `aclnnInplaceLerps` |
| `workspaceSize` / `executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段由 executor 返回全图 workspace 与执行图 |
| `workspace` / `stream` | Device workspace / ACL stream | `void*` / `aclrtStream` | 第二段交给 `CommonOpExecutorRun` 异步执行 |

### 入口与版本

固定仓公开的 Tensor-weight 原地入口只有无版本后缀的 `aclnnInplaceLerp`，未见 V2 或更高公开版本。文档明确支持 Ascend 950PR/950DT；`Lerp` OpDef 只注册 `ascend950`，`opFile.value=lerp_apt`，目录 CMake 也只设 `SUPPORT_COMPUTE_UNIT "ascend950"` 与 `SUPPORT_TILING_DIR "arch35"`。Host/Kernel 还实现了 FP16/BF16 start/end 配 FP32 weight 的内部 key，但 `CheckDtypeValid` 要求 weight 与 `selfRef` 同 dtype，因而这些 key 不是本 ACLNN 入口的可达 Route。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnInplaceLerpGetWorkspaceSize` 先检查 `workspaceSize/executor`，创建 executor，然后以 `out=selfRef` 调用 `CalculateResult`。`CheckParams` 按 null→dtype→shape 顺序失败关闭：dtype 依当前架构支持 FP16/FP32/BF16 且三输入与输出必须同 dtype；shape 先合并 `selfRef/end`，再与 `weight` broadcast，结果必须等于 `selfRef` view shape。`selfRef->IsEmpty()` 时直接成功，不加 Lerp 节点。

非空时按 `selfRef`、`end`、`weight` 顺序调用框架 `l0op::Contiguous`，再将三个连续 Tensor 交给 `l0op::Lerp`，最后 `l0op::ViewCopy(lerpResult,selfRef,executor)` 写回原 owner。因此“原地”是 ACLNN 可见语义，Lerp Device body 仍以独立 `start` 与 `y` GM 指针计算；Contiguous/ViewCopy 的内部 kernel 是允许的框架边界。

### 参数、Workspace 与生命周期

`l0op::Lerp` 为 broadcast shape 分配内部连续输出，通过 `ADD_TO_LAUNCHER_LIST_AICORE(Lerp, op::AI_CORE, ...)` 加入 executor。第一段最终以 `uniqueExecutor->GetWorkspaceSize()` 返回包含可选 Contiguous/ViewCopy 在内的全图 workspace，固定仓中无法从算子代码化简为常数。Lerp Tiling 对自有节点固定写 `workspaces[0]=32` B，但十个 body 都不读 `workspace`。executor、用户 workspace、三个输入与 `selfRef` 必须覆盖 `CommonOpExecutorRun` 在指定 stream 上的异步执行期；框架精确释放时点固定仓不可见。

### 启动图与失败传播

executor 图为 `Contiguous(selfRef/end/weight) -> Lerp -> ViewCopy(selfRef)`，可选框架节点在同一 executor/stream 中建立顺序。算子自有 launch graph 只有一个 Device 节点：`l0op::Lerp -> lerp(start,end,weight,y,workspace,tiling)`，`blockDim=blockNum`，每核内先 NDDMA/`DataCopyPad` 将三输入展开到 UB，再用 MicroAPI 计算并搬回 `y`。entry 对 `g_coreType==AIC` 立即返回，并声明 `KERNEL_TYPE_AIV_ONLY`，不存在 Cube body。API 校验、Contiguous、L0 加图、BroadcastTiling、raw POD 保存或 ViewCopy 任一失败都阻断后续；未创建算子自有 event 或跨 Device 节点 workspace 边。

## Route 与参数设计

### 分核与资源计算

Host 先对三输入的 storage shape 用 `EnsureNotScalar`（scalar 转 `{1}`），用 `DimensionCollapse` 对相同 broadcast pattern 的相邻轴合轴，并将 broadcast 输入轴 stride 设 0。设平台 UB 字节数为 `U`，`E=elemNum`：FP16/BF16 使用 `bufferDivisor=128`，得 `E=align_down(U/16,2048)`；FP32 使用 `bufferDivisor=256`，得 `E=align_down(U/32,2048)`。对输出合轴 shape 从内向外累乘，首次超过 `E` 的轴成为 `ubSplitAxis`，`ubFormer=E/product(inner axes)`，`ubOuter=ceil(dim[axis]/ubFormer)`，`ubTail=dim[axis]-(ubOuter-1)*ubFormer`。

`fusedProduct=ubOuter*product(dims before split axis)`；Lerp 未设 `preferMultiCore`，因此直接使用 `blockFormer=ceil(fusedProduct/coreNum)`、`blockNum=ceil(fusedProduct/blockFormer)`、`blockTail=fusedProduct-(blockNum-1)*blockFormer`。普通核循环 `blockFormer` 次，尾核循环 `blockTail` 次。`shapeLen-ubSplitAxis<=5` 选 without-loops NDDMA，否则选 with-loops。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 NDDMA without-loops | `100000001000100` / `LerpF16NddmaWithoutLoops` | 四个 Lerp tensor 均 FP16，且合轴后 `shapeLen-ubSplitAxis<=5` | 内轴数 `>5` 转 R-002；dtype 转后续 key |
| 2 | R-002 FP16 NDDMA with-loops | `100000001001100` / `LerpF16NddmaWithLoops` | 四个 Lerp tensor 均 FP16，且合轴后 `shapeLen-ubSplitAxis>5` | dtype 转后续 key |
| 3 | R-003 FP32 NDDMA without-loops | `200000001000100` / `LerpF32NddmaWithoutLoops` | 四个 Lerp tensor 均 FP32，且合轴后 `shapeLen-ubSplitAxis<=5` | 内轴数 `>5` 转 R-004；dtype 转后续 key |
| 4 | R-004 FP32 NDDMA with-loops | `200000001001100` / `LerpF32NddmaWithLoops` | 四个 Lerp tensor 均 FP32，且合轴后 `shapeLen-ubSplitAxis>5` | dtype 转后续 key |
| 5 | R-005 BF16 NDDMA without-loops | `300000001000100` / `LerpBf16NddmaWithoutLoops` | 四个 Lerp tensor 均 BF16，且合轴后 `shapeLen-ubSplitAxis<=5` | 内轴数 `>5` 转 R-006；其他 dtype 失败 |
| 6 | R-006 BF16 NDDMA with-loops | `300000001001100` / `LerpBf16NddmaWithLoops` | 四个 Lerp tensor 均 BF16，且合轴后 `shapeLen-ubSplitAxis>5` | 其他 dtype 失败 |

```text
aclnnInplaceLerp CheckDtypeValid: selfRef == end == weight == out
  FP16 -> opKey=1
  FP32 -> opKey=2
  BF16 -> opKey=3
  other/mismatch -> ACLNN_ERR_PARAM_INVALID
BroadcastTiling:
  computeKey=1
  scheduleKey = (shapeLen-ubSplitAxis <= 5) ? 100 : 1100
  innerKey = 1*1,000,000 + scheduleKey
  finalKey = opKey*100,000,000,000,000 + innerKey
router exact-key match -> R-001..R-006; unknown key -> no output and return
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `100000001000100` | 950，同 dtype FP16，内轴 `<=5` | 单 AIV node：5D NDDMA/continuous DMA→Vector→GM | `LerpF16NddmaWithoutLoops::{Init,Process,CopyIn0/1/2,Compute3,CopyOut4}` | 4×depth2 FP16 queue，FP32 registers，queue 交接 | `ubTail` + Vector mask；无其他 fallback |
| R-002 | `100000001001100` | 950，同 dtype FP16，内轴 `>5` | 单 AIV node：外循环分解5D NDDMA→Vector→GM | `LerpF16NddmaWithLoops::{Init,Process,CopyIn0/1/2,Compute3,CopyOut4}` | 同 R-001，NDDMA copy 多一层外循环 | 外循环指数 + `ubTail` + mask |
| R-003 | `200000001000100` | 950，同 dtype FP32，内轴 `<=5` | 单 AIV node：5D NDDMA/continuous DMA→Vector→GM | `LerpF32NddmaWithoutLoops::{Init,Process,CopyIn0/1/2,Compute3,CopyOut4}` | 4×depth2 FP32 queue，FP32 registers | `ubTail` + Vector mask |
| R-004 | `200000001001100` | 950，同 dtype FP32，内轴 `>5` | 单 AIV node：外循环分解5D NDDMA→Vector→GM | `LerpF32NddmaWithLoops::{Init,Process,CopyIn0/1/2,Compute3,CopyOut4}` | 同 R-003，NDDMA copy 多一层外循环 | 外循环指数 + `ubTail` + mask |
| R-005 | `300000001000100` | 950，同 dtype BF16，内轴 `<=5` | 单 AIV node：5D NDDMA/continuous DMA→Vector→GM | `LerpBf16NddmaWithoutLoops::{Init,Process,CopyIn0/1/2,Compute3,CopyOut4}` | 4×depth2 BF16 queue，FP32 registers，RINT cast | `ubTail` + Vector mask |
| R-006 | `300000001001100` | 950，同 dtype BF16，内轴 `>5` | 单 AIV node：外循环分解5D NDDMA→Vector→GM | `LerpBf16NddmaWithLoops::{Init,Process,CopyIn0/1/2,Compute3,CopyOut4}` | 同 R-005，NDDMA copy 多一层外循环 | 外循环指数 + `ubTail` + mask |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| dtype key | API 先要求三输入同 dtype；Tiling `GetOpKey` 再识别精确四元组 | queue dtype、load distribution 和 cast pipeline 不同 | 全部 | API 报 `ACLNN_ERR_PARAM_INVALID`；Tiling 层 invalid key 失败 | `CheckDtypeValid`, `LerpTiling::GetOpKey` |
| UB 有效元素上限 | `E=align_down(U*8/bufferDivisor,2048)`，`E>0` | 4 个 depth-2 queue 必须同时容纳一个 tile | 全部 | `maxElemNum==0` 时 Tiling 失败 | `BroadcastGetMaxElemNum`, `DoBrodcastTiling` |
| NDDMA 维数 | `shapeLen-ubSplitAxis<=5` | 950 `DataCopy<T,5>` 可一次描述所有 UB 内轴 | R-001/R-003/R-005 | `>5` 转对应 with-loops Route | `BroadcastGetScheduleKey` |
| 外循环 NDDMA | `shapeLen-ubSplitAxis>5`，且合轴后 `shapeLen<=8` | 5D descriptor 外最多丢出 3 轴，用 `nddmaIndices[3]` 遍历 | R-002/R-004/R-006 | `shapeLen>8` 时 BroadcastTiling 失败 | `BroadcastNddmaWithLoop`, `DoDimensionCollapse` |
| broadcast DB cache | `inputStride[ubSplitAxis]==0` 且不是 DB 首两槽/新 outer sweep 首两次 | 两个 queue slot 可交替复用相同 broadcast slice | 全部 | stride 非 0 或 cache 边界时重新 CopyIn | 各 body `CopyIn0/1/2` |
| 空 Tensor | `selfRef->IsEmpty()` | 无输出元素，避免产生零 block/tile | 全部之前 | ACLNN 直接成功，不进入 Route | `CalculateResult` |

### POD/ABI

`LerpTilingData` 声明为 9 个 `int64_t` 标量加 8 个 `int64_t[8]` 数组，字段有效 payload 合计 584 B。`BEGIN_TILING_DATA_DEF/TILING_DATA_FIELD_DEF` 的宏展开不在固定 `ops-math`/`opbase` 源码内，因而 C++ 对象的 `sizeof/alignof/offsetof` 不作推测；Host 与 Device 都依同一注册结构的序列化顺序。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LerpTilingData` | `blockFormer` | `int64_t`/64 | 第 1 个声明字段；精确 offset 不可见 | UB loop/core | `DoOpTiling <- BroadcastTilingData.blockFormer` | `Process`、cache guard、起始坐标 | 普通核 loop 数；尾核不读作 loop 数 |
| `LerpTilingData` | `ubFormer` | `int64_t`/64 | 第 2 个 | split-axis elements | `DoOpTiling` | `Process`、NDDMA helper、GM offset | 整 tile split 因子 |
| `LerpTilingData` | `ubOuter` | `int64_t`/64 | 第 3 个 | loop positions | `DoOpTiling` | `Process`、cache guard | 必须大于 0 |
| `LerpTilingData` | `ubTail` | `int64_t`/64 | 第 4 个 | split-axis elements | `DoOpTiling` | `Process` 选尾 tile | 1..`ubFormer` |
| `LerpTilingData` | `blockTail` | `int64_t`/64 | 第 5 个 | UB loops | `DoOpTiling` | 最后一核 `Process` | 尾核 loop 数 |
| `LerpTilingData` | `shapeLen` | `int64_t`/64 | 第 6 个 | collapsed axes | `DoOpTiling` | NDDMA helper | 1..8；超过 8 失败 |
| `LerpTilingData` | `ubSplitAxis` | `int64_t`/64 | 第 7 个 | axis index | `DoOpTiling` | 所有 `Process/CopyIn/Compute/CopyOut` | 0..`shapeLen-1` |
| `LerpTilingData` | `dimProductBeforeUbInner` | `int64_t`/64 | 第 8 个 | loop positions | `DoOpTiling` | `BroadcastGetAxesIndices` | 等于 `fusedProduct` |
| `LerpTilingData` | `elemNum` | `int64_t`/64 | 第 9 个 | elements/queue slot | `DoOpTiling <- maxElemNum` | `InitBuffer` 容量 | 2048-element 对齐，不得为 0 |
| `LerpTilingData` | `input0Dims` | `int64_t[8]` | 第 10 个，64 B payload | elements/axis | `dims[0]` | 当前 6 Route body 未读 | inactive consumer |
| `LerpTilingData` | `input1Dims` | `int64_t[8]` | 第 11 个，64 B payload | elements/axis | `dims[1]` | 当前 body 未读 | inactive consumer |
| `LerpTilingData` | `input2Dims` | `int64_t[8]` | 第 12 个，64 B payload | elements/axis | `dims[2]` | 当前 body 未读 | inactive consumer |
| `LerpTilingData` | `outputDims` | `int64_t[8]` | 第 13 个，64 B payload | elements/axis | `dims[3]` | 坐标更新与 NDDMA descriptor | `shapeLen` 后未写槽保持 Host 零初值 |
| `LerpTilingData` | `input0Strides` | `int64_t[8]` | 第 14 个，64 B payload | elements | `strides[0]` | start `CopyIn2` | broadcast 轴为 0 |
| `LerpTilingData` | `input1Strides` | `int64_t[8]` | 第 15 个，64 B payload | elements | `strides[1]` | end `CopyIn1` | broadcast 轴为 0 |
| `LerpTilingData` | `input2Strides` | `int64_t[8]` | 第 16 个，64 B payload | elements | `strides[2]` | weight `CopyIn0` | broadcast 轴为 0 |
| `LerpTilingData` | `outputStrides` | `int64_t[8]` | 第 17 个，64 B payload | elements | `strides[3]` | 所有 CopyIn、Compute、CopyOut | 连续 collapsed output stride |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `100000001000100` | `lerp_tiling_arch35.cpp::GetOpKey/DoOpTiling` + opbase `BroadcastTiling` | `LerpTilingData` | `lerp_apt.cpp::lerp` | `LerpF16NddmaWithoutLoops` + `broadcast_utils.h` | `lerp_def.cpp`, `ascend950/lerp_binary.json`, `math/lerp/CMakeLists.txt` |
| R-002 | `100000001001100` | 同上，schedule key 1100 | 同上 | `lerp_apt.cpp::lerp` | `LerpF16NddmaWithLoops` + `broadcast_utils.h` | 同上 |
| R-003 | `200000001000100` | 同上，opKey 2 | 同上 | `lerp_apt.cpp::lerp` | `LerpF32NddmaWithoutLoops` + `broadcast_utils.h` | 同上 |
| R-004 | `200000001001100` | 同上，opKey 2/schedule 1100 | 同上 | `lerp_apt.cpp::lerp` | `LerpF32NddmaWithLoops` + `broadcast_utils.h` | 同上 |
| R-005 | `300000001000100` | 同上，opKey 3 | 同上 | `lerp_apt.cpp::lerp` | `LerpBf16NddmaWithoutLoops` + `broadcast_utils.h` | 同上 |
| R-006 | `300000001001100` | 同上，opKey 3/schedule 1100 | 同上 | `lerp_apt.cpp::lerp` | `LerpBf16NddmaWithLoops` + `broadcast_utils.h` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析同一 `LerpTilingData`，用完整 key 串行路由到 6 个 ACLNN 可达 body；另 4 个 mixed-weight body 被 API dtype guard 截断。每个 body 绑定 GM，初始化 weight/end/start/output 四个 depth-2 queue，然后按 core 的 `blockFormer/blockTail` 遍历 UB tile。每个 tile 严格执行 `CopyIn(weight) -> CopyIn(end) -> CopyIn(start) -> Compute -> CopyOut`。

CopyIn 在切分轴 stride 不同时用 5D NDDMA 将 broadcast 展开到连续 UB；stride 与输出相同时改用单 block `DataCopyPad`。Compute 的有效元素数为 `n=ubSplitSize*outputStrides[ubSplitAxis]`，以 `VECTOR_REG_WIDTH/4` 个 FP32 lane 为一次 MicroAPI 循环，`UpdateMask(n)` 保护末次 tail。CopyOut 以 `gmOffset=sum(indices[j]*outStride[j])+indices[split]*outStride[split]*ubFormer` 找到 GM，只写 `n*sizeof(out_dtype)` B。TQue `EnQue/DeQue/FreeTensor` 建立 NDDMA→Vector→DMA 顺序，无显式 barrier/event/atomic。

### Route R-001：FP16 NDDMA without-loops

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_lerp_tensor.cpp` | `CheckDtypeValid` | `selfRef=end=weight=out=FP16` | 进入 Lerp |
| Tiling | `op_host/arch35/lerp_tiling_arch35.cpp` | `GetOpKey`, `BroadcastGetScheduleKey` | opKey=1，`shapeLen-ubSplitAxis<=5` | key `100000001000100` |
| Router | `op_kernel/lerp_apt.cpp` | `lerp` | `TILING_KEY_IS(100000001000100)` | `LerpF16NddmaWithoutLoops` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Lerp -> lerp(start,end,weight,y,workspace,tiling)
  -> LerpF16NddmaWithoutLoops::Init
  -> Process
    -> CopyIn0(weight, BroadcastNddmaWithoutLoop<DataCopy<half,5>>/DataCopyPad)
    -> CopyIn1(end, same)
    -> CopyIn2(start, same)
    -> Compute3(FP16 unpack -> FP32 stable lerp -> CAST_RINT FP16 pack)
    -> CopyOut4(DataCopyPad -> y)
```

##### 源码等价伪码

```text
loops = last_core ? blockTail : blockFormer
indices = decode(blockFormer*blockIdx)
for tile in [0, loops):
  update indices when tile>0
  s = (indices[split] == ubOuter-1) ? ubTail : ubFormer
  NDDMA_without_loop_expand(weight/end/start, s, indices) unless DB broadcast cache hits
  for r in [0, ceil(s*outStride[split]/fp32_lanes)):
    mask = tail_mask()
    w,e,a = fp32(load_fp16(weight,end,start))
    d = e-a
    left = a + w*d
    right = e + (w-1)*d
    store_fp16_rint(select(w<0.5, left, right), mask)
  copy exactly s*outStride[split]*2 bytes to y[gmOffset]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM start/end/weight | input broadcast strides + tile indices | logical input extent | FP16/ND | caller/Contiguous | CopyIn2/1/0 | executor stream | stride 0 表示 broadcast |
| `queIn0/1/2` | queue slot | depth2 × `E*2` B each | FP16 expanded contiguous | NDDMA/DMA | Compute3 | per core | 两槽交替；stride0 可 cache |
| FP32 RegTensor | vector iteration | 编译期寄存器数，字节占用不可见 | FP32 lanes | load/cast | Sub/MulAddDst/Select | one vector iteration | 中间值原地改写 |
| `queOut0` | queue slot | depth2 × `E*2` B | FP16 contiguous | Compute3 | CopyOut4 | per core | Vector→DMA queue |
| GM y | `BroadcastGetGmOffset(...)` | `s*outStride[split]*2` B | FP16/ND | CopyOut4 | ViewCopy/selfRef | executor stream | 只写有效 tail |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["queIn0 weight · depth=2 · 2·E·2 B"]:1
  in1["queIn1 end · depth=2 · 2·E·2 B"]:1
  in2["queIn2 start · depth=2 · 2·E·2 B"]:1
  out0["queOut0 y · depth=2 · 2·E·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4 * 2 * E * sizeof(half) = 16E bytes <= U
dynamic_ub = N/A
workspace_total(Lerp node) = 32 bytes; Device body does not read it
=> E = align_down(U/16, 2048)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input queue slot 0/1 | CopyIn0/1/2 | Compute3 `DeQue` | 同 parity 后续 tile | `EnQue/DeQue/FreeTensor` | depth=2；无消费前不覆盖 |
| stride-0 cached slot | 每个 outer sweep 首两次 CopyIn | 后续交替 DeQue | 下一 sweep 首两次 | queue parity + guard | split-axis input stride=0，源 slice 未变 |
| output queue slot | Compute3 | CopyOut4 | 同 parity 后续 tile | `EnQue/DeQue/FreeTensor` | CopyOut 完成后才覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | TQue enqueue/dequeue | NDDMA/DMA | Compute3 | 三个 input slots | NDDMA 展开；末 tile 使用 `ubTail` |
| Compute | Vector mask | FP16 unpack/cast | FP32 arithmetic | register lanes | `UpdateMask(size)` 屏蔽末次无效 lane |
| Compute→CopyOut | TQue enqueue/dequeue | Compute3 | CopyOut4 | output slot | `CAST_RINT`, `NO_SAT`，只存 mask 内 lane |
| CopyOut | `DataCopyPad`, blockCount=1 | output UB | y GM | 当前 tile | blockLen 为有效字节数 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/lerp_tiling_arch35.cpp` | `GetComputeMap`, `DoOpTiling` | FP16 divisor=128，产生 block/tile/POD/key |
| Kernel entry/router | `op_kernel/lerp_apt.cpp` | `lerp`, key `100000001000100` | AIV-only 路由到 R-001 |
| Device body/helper | `op_kernel/arch35/lerp_f16_nddma_without_loops.h` | `LerpF16NddmaWithoutLoops` | 5D CopyIn、FP32 stable branch、RINT CopyOut |

### Route R-002：FP16 NDDMA with-loops

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_lerp_tensor.cpp` | `CheckDtypeValid` | 同 dtype FP16 | 进入 Lerp |
| Tiling | `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | `shapeLen-ubSplitAxis>5` | schedule 1100，完整 key `100000001001100` |
| Router | `op_kernel/lerp_apt.cpp` | `lerp` | 精确 key match | `LerpF16NddmaWithLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpF16NddmaWithLoops::Init -> Process
  -> CopyIn0/1/2 -> BroadcastNddmaWithLoop
     -> BroadcastSetNddmaConfigWithLoop -> looped DataCopy<half,5>
  -> Compute3 -> CopyOut4
```

##### 源码等价伪码

```text
for each core tile:
  s = full_or_tail_split_size
  for each input requiring refresh:
    if outStride[split] != inStride[split]:
      desc = last 5 axes of output/input strides
      count = s * product(outputDims[split+1 : shapeLen-5])
      for q in [0,count): DataCopy<T,5>(UB[ubOffset(q)], GM[gmOffset+inOffset(q)], desc)
    else: DataCopyPad(one contiguous block)
  run the same FP16->FP32 stable lerp and masked FP16 store as R-001
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs→UB | base `BroadcastGetGmOffset` + thrown-axis input offset | 每次 5D descriptor 覆盖的 slice，总有效 `n` 元素 | FP16 broadcast→contiguous | looped NDDMA | Compute3 | tile | 最多 3 个 thrown-axis index |
| 4 depth-2 queues | 同 R-001 | 总计 16E B | FP16 | CopyIn/Compute | Compute/CopyOut | per core | DB cache 与 queue parity不变 |
| y GM | output offset 公式 | `n*2` B | FP16 | CopyOut4 | ViewCopy | stream | tail exact bytes |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weightQ depth2 · 4E B"]:1
  in1["endQ depth2 · 4E B"]:1
  in2["startQ depth2 · 4E B"]:1
  out0["outQ depth2 · 4E B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 16E <= U; dynamic_ub=N/A; Lerp workspace=32 B
NDDMA descriptor = 5 axes; outer iteration count = ubSplitSize * product(thrown inner axes)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output DB slots | CopyIn/Compute | Compute/CopyOut | same parity later tile | queue API | 与 R-001 相同 |
| NDDMA output sub-slices | each outer `DataCopy<T,5>` | Compute3 after CopyIn returns/enqueues | next refreshed tile | synchronous CopyIn stage + queue | UB offsets from output strides do not overlap incorrectly |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thrown axes | serial outer loop + `DataCopy<T,5>` | GM | input queue enqueue | each UB sub-slice | loop count covers dimensions outside 5D descriptor |
| Vector | queue + mask | expanded FP16 UB | output queue | valid lanes | FP32 arithmetic, RINT FP16 tail |
| output | `DataCopyPad` | output queue | GM | exact tile bytes | `ubTail` controls last tile |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | 内轴数 >5 生成 1100 |
| Kernel entry/router | `op_kernel/lerp_apt.cpp` | key `100000001001100` | 路由 FP16 with-loops |
| Device body/helper | `lerp_f16_nddma_with_loops.h`, opbase `broadcast_utils.h` | `LerpF16NddmaWithLoops`, `BroadcastNddmaWithLoop` | 外循环拆 5D NDDMA，Vector pipeline 完整 |

### Route R-003：FP32 NDDMA without-loops

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lerp_tensor.cpp` | `CheckDtypeValid` | 同 dtype FP32 | Lerp |
| Tiling | `lerp_tiling_arch35.cpp` | `GetOpKey/GetComputeMap` | opKey=2, divisor=256，内轴<=5 | key `200000001000100` |
| Router | `lerp_apt.cpp` | `lerp` | exact key | `LerpF32NddmaWithoutLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpF32NddmaWithoutLoops::Init -> Process
  -> CopyIn0/1/2(BroadcastNddmaWithoutLoop<float>)
  -> Compute3(FP32 load/Sub/MulAddDst/Select/store)
  -> CopyOut4
```

##### 源码等价伪码

```text
for each tile:
  expand w,end,start to three FP32 UB queues with one 5D descriptor or contiguous DMA
  for masked FP32 vector chunk:
    d=end-start
    y=select(w<0.5, start+w*d, end+(w-1)*d)
  copy n*4 valid bytes to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| three input queues | queue slot | each depth2×`E*4` B | FP32 | NDDMA/DMA | Compute3 | per core | stride0 DB cache |
| output queue | queue slot | depth2×`E*4` B | FP32 | Compute3 | CopyOut4 | per core | exact tail store |
| registers | FP32 lanes | fixed source does not expose byte count | FP32 | loads | arithmetic/store | vector iteration | no cast buffers |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weightQ depth2 · 8E B"]:1
  in1["endQ depth2 · 8E B"]:1
  in2["startQ depth2 · 8E B"]:1
  out0["outQ depth2 · 8E B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4*2*E*4 = 32E <= U
E = align_down(U/32,2048); dynamic_ub=N/A; Lerp workspace=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input DB slots | CopyIn | Compute3 | same parity tile | EnQue/DeQue/FreeTensor | depth2 |
| FP32 register destinations | load/Sub/MulAddDst | next arithmetic/store | next vector iteration | lexical vector scope | no cross-iteration dependency |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | queue | NDDMA/DMA | Compute3 | FP32 input slots | no conversion |
| Compute | FP32 MicroAPI + mask | input registers | output store | valid lanes | stable two-form branch; no cast |
| CopyOut | queue + DataCopyPad | output slot | y GM | exact bytes | `n*4` B |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp` | opKey 2 / divisor 256 | FP32 UB 上限与 key |
| Kernel entry/router | `lerp_apt.cpp` | key `200000001000100` | R-003 |
| Device body/helper | `lerp_f32_nddma_without_loops.h` | `LerpF32NddmaWithoutLoops` | no-loop NDDMA + FP32 stable branch |

### Route R-004：FP32 NDDMA with-loops

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lerp_tensor.cpp` | `CheckDtypeValid` | 同 dtype FP32 | Lerp |
| Tiling | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | opKey=2，内轴>5 | key `200000001001100` |
| Router | `lerp_apt.cpp` | `lerp` | exact key | `LerpF32NddmaWithLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpF32NddmaWithLoops::Init -> Process
  -> CopyIn0/1/2 -> BroadcastNddmaWithLoop<float> -> repeated DataCopy<float,5>
  -> Compute3(FP32) -> CopyOut4
```

##### 源码等价伪码

```text
for each tile and each input:
  enumerate at most three axes outside 5D descriptor
  issue one 5D NDDMA for each enumerated coordinate, writing its UB output-stride offset
after all three queues are ready, run masked FP32 stable lerp and copy n*4 bytes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| NDDMA sub-slices | thrown-axis output/input stride offsets | total `n` FP32 elements/input | FP32 | looped 5D copies | Compute3 | current queue slot | offsets derive from POD strides |
| four queues | queue slots | 32E B total | FP32 | CopyIn/Compute | Compute/CopyOut | per core | depth2/cache |
| GM output | output gmOffset | `n*4` B | FP32 | CopyOut4 | ViewCopy | stream | tail exact |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weightQ 8E B"]:1
  in1["endQ 8E B"]:1
  in2["startQ 8E B"]:1
  out0["outQ 8E B"]:1
```

##### 字节账与复用顺序

```text
UB_total=32E<=U; E=align_down(U/32,2048)
dynamic_ub=N/A; workspace=32 B; NDDMA outer count=s*product(thrown axes)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DB queue slots | looped CopyIn | Compute3 | later same-parity tile | queue protocol | all NDDMA subcopies finish before EnQue |
| broadcast cached slots | sweep first two CopyIns | later tiles | next sweep | guard + parity | source stride at split axis is zero |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| looped NDDMA | serial loop | GM | EnQue | complete expanded tile | thrown-axis tail is exact loop bound |
| Vector | mask | queues | output queue | lanes | FP32 throughout |
| CopyOut | DataCopyPad | output queue | GM | tile | blockLen=`n*4` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | opbase `broadcast_tiling.cpp` | `DoBrodcastTiling` | UB/core/tail 参数 |
| Kernel entry/router | `lerp_apt.cpp` | key `200000001001100` | R-004 |
| Device body/helper | `lerp_f32_nddma_with_loops.h`, `broadcast_utils.h` | `LerpF32NddmaWithLoops`, `BroadcastNddmaWithLoop` | FP32 looped NDDMA + Vector |

### Route R-005：BF16 NDDMA without-loops

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lerp_tensor.cpp` | `GetDtypeSupportList/CheckDtypeValid` | 950 `IsRegBase` 支持 BF16，三输入同 dtype | Lerp |
| Tiling | `lerp_tiling_arch35.cpp` | `GetOpKey/GetComputeMap` | opKey=3，divisor=128，内轴<=5 | key `300000001000100` |
| Router | `lerp_apt.cpp` | `lerp` | exact key | `LerpBf16NddmaWithoutLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpBf16NddmaWithoutLoops::Init -> Process
  -> three BroadcastNddmaWithoutLoop<bfloat16_t>
  -> Compute3(BF16 unpack/cast -> FP32 arithmetic -> RINT BF16 pack)
  -> CopyOut4
```

##### 源码等价伪码

```text
for each tile:
  expand three BF16 inputs by 5D NDDMA or contiguous DMA
  for each masked FP32-lane chunk:
    w,e,a = cast_fp32(unpack_bf16(...))
    y32 = select(w<0.5, a+w*(e-a), e+(w-1)*(e-a))
    pack_bf16(cast_rint(y32))
  copy n*2 bytes to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input queues | slot | each depth2×`E*2` B | BF16 | NDDMA/DMA | Compute3 | per core | broadcast cache |
| registers | vector chunk | byte count not exposed | FP32/BF16 register layouts | unpack/cast | arithmetic/pack | iteration | ZEROING mask |
| output queue/GM | slot / gmOffset | depth2×`E*2` B / `n*2` valid B | BF16 | Compute/CopyOut | GM/ViewCopy | tile/stream | RINT, no saturation |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weightQ depth2 · 4E B"]:1
  in1["endQ depth2 · 4E B"]:1
  in2["startQ depth2 · 4E B"]:1
  out0["outQ depth2 · 4E B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16E<=U; E=align_down(U/16,2048)
dynamic_ub=N/A; Lerp workspace=32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input DB/cache slots | CopyIn | Compute3 | parity/sweep dependent | queue + refresh guard | 同 R-001 |
| FP32 intermediate registers | casts/arithmetic | Select/cast | next vector iteration | vector scope | mask limits valid lanes |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load | `DIST_UNPACK_B16` + Cast | BF16 UB | FP32 math | active lanes | cast trait round UNKNOWN for widening |
| math | ZEROING Sub/MulAddDst/Select | FP32 regs | narrow cast | active lanes | stable branch |
| store | `CAST_RINT`, `DIST_PACK_B32` | FP32 result | BF16 queue | active lanes | NO_SAT, masked tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp` | opKey 3 / divisor 128 | BF16 route params |
| Kernel entry/router | `lerp_apt.cpp` | key `300000001000100` | R-005 |
| Device body/helper | `lerp_bf16_nddma_without_loops.h` | `LerpBf16NddmaWithoutLoops` | BF16→FP32→BF16 no-loop body |

### Route R-006：BF16 NDDMA with-loops

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lerp_tensor.cpp` | `GetDtypeSupportList/CheckDtypeValid` | 950 同 dtype BF16 | Lerp |
| Tiling | opbase `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | opKey=3，内轴>5 | key `300000001001100` |
| Router | `lerp_apt.cpp` | `lerp` | exact key | `LerpBf16NddmaWithLoops` |

#### Device 数据流

##### 函数级调用链

```text
lerp -> LerpBf16NddmaWithLoops::Init -> Process
  -> CopyIn0/1/2 -> BroadcastNddmaWithLoop<bfloat16_t> -> repeated DataCopy<bfloat16_t,5>
  -> Compute3(BF16->FP32 stable lerp->BF16) -> CopyOut4
```

##### 源码等价伪码

```text
for each tile:
  enumerate dimensions outside last five NDDMA axes for each refreshed input
  fill a contiguous BF16 UB tile with repeated 5D copies
  run the same widened, masked and RINT-narrowed compute as R-005
  write only valid tail bytes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| looped NDDMA slices | gm base + input thrown-axis offset; UB output offset | total `n` BF16/input | BF16 broadcast→contiguous | repeated 5D DataCopy | Compute3 | queue slot | at most 3 thrown dims |
| four queues | slot | 16E B total | BF16 | CopyIn/Compute | Compute/CopyOut | per core | depth2/cache |
| output | gmOffset | `n*2` B | BF16 | CopyOut4 | ViewCopy | stream | exact tail |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weightQ 4E B"]:1
  in1["endQ 4E B"]:1
  in2["startQ 4E B"]:1
  out0["outQ 4E B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16E<=U; E=align_down(U/16,2048)
dynamic_ub=N/A; workspace=32 B; NDDMA outer count=s*product(thrown axes)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue DB slots | looped CopyIn/Compute | Compute/CopyOut | later same-parity tile | queue protocol | all subcopies complete before EnQue |
| cached broadcast slots | sweep first two writes | later DeQue reads | next sweep first two | guard/parity | split stride zero |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA loop | serial repeated `DataCopy<BF16,5>` | GM | EnQue | expanded tile | exact coordinate bounds |
| Vector | queue + mask | BF16 UB | output queue | active lanes | widen FP32, stable branch, RINT narrow |
| output | DataCopyPad | output queue | GM | tile bytes | `ubTail` and `n*2` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp`, opbase `broadcast_tiling_base.cpp` | opKey 3 + schedule 1100 | BF16 with-loops key |
| Kernel entry/router | `lerp_apt.cpp` | key `300000001001100` | R-006 |
| Device body/helper | `lerp_bf16_nddma_with_loops.h`, `broadcast_utils.h` | `LerpBf16NddmaWithLoops`, `BroadcastNddmaWithLoop` | looped BF16 NDDMA + widened Vector |

## 算子特定优化

### Trick T-001：以 weight=0.5 分界的等价双公式

#### 触发与源码依据

- 适用 Route：R-001..R-006
- 触发条件：每个有效 Vector lane，以 FP32 `weight<0.5` predicate 选择结果
- Golden 来源：[`math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h:167-173`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h#L167-L173)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg6, vreg3, vreg5, preg0);
                AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg1, vreg6,
                                                                                               preg0);
                AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg8, vreg1, vreg7, preg0);
                AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg3, vreg8, vreg6,
                                                                                               preg0);
                AscendC::MicroAPI::Select<float>(vreg9, vreg5, vreg3, preg2);
```

#### 机制与设计原因

`vreg6=end-start`，第一个 `MulAddDst` 得 `start+weight*diff`；`vreg8=weight-1`，第二个得 `end+(weight-1)*diff`。`Select` 在 weight 较小时取前式，否则取后式，使接近 1 的 weight 不必把小修正量先加到 start。两路都在 FP32 register 中完成，没有额外 UB 中间 Tensor；mask 来自当前 tile 有效 lane。

#### 迁移边界

- 适用：线性插值、输出 dtype 允许以 FP32 中间值计算的逐元素管线
- 前提：必须保留 `weight<0.5` 的 predicate、相同 `diff`、MicroAPI mask 与 FP16/BF16 最终 `CAST_RINT`
- 失效：若改变 NaN/比较/舍入语义，不能仅凭代数等价替换，需重新做数值验证
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：用 depth-2 queue 缓存 split-axis broadcast slice

#### 触发与源码依据

- 适用 Route：R-001..R-006
- 触发条件：对任一输入，`inputStride[ubSplitAxis]==0`，并且当前迭代不是本 core/outer sweep 需初始化两个 queue parity 槽的位置
- Golden 来源：[`math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h:79-88`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h#L79-L88)

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
```

#### 机制与设计原因

split-axis stride 0 表明沿该轴的连续 tile 使用同一 GM slice。queue depth 为 2，所以每个复用周期的前两次分别填充奇偶槽，后续迭代仍做 Alloc/EnQue 生命周期，却跳过 GM→UB NDDMA，直接消费槽内保留的 broadcast 数据。这会源码可证地减少重复搬运，但不改变 UB 容量。

#### 迁移边界

- 适用：broadcast 轴 stride 为 0 且使用双缓冲 queue 的逐 tile 算子
- 前提：必须精确保留 queue depth=2、global tile 对 `ubOuter` 的边界判定、EnQue/DeQue/FreeTensor 顺序
- 失效：stride 非 0、queue depth 改变、outer sweep 导致源 slice 改变时必须重新搬运
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在 `inputStride[ubSplitAxis]!=0` 时跳过 CopyIn；源码对该条件始终重新搬运，否则会复用上一 tile 的不同数据。

### Trick T-003：以5D NDDMA 描述能力自动切换单发与外循环分解

#### 触发与源码依据

- 适用 Route：R-001/R-003/R-005 用 without-loops；R-002/R-004/R-006 用 with-loops
- 触发条件：`shapeLen-ubSplitAxis<=5` 与 `>5`
- Golden 来源：[`math/lerp/op_kernel/lerp_apt.cpp:28-35`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/lerp_apt.cpp#L28-L35)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// start is float16, end is float16, weight is float16, y is float16, max dims in ub is 5 and nddma does not need loops
#define LERP_F16_NDDMA_WITHOUT_LOOPS_TILING_KEY 100000001000100
// start is float16, end is float16, weight is float16, y is float16, max dims in ub is 8 and nddma needs loops
#define LERP_F16_NDDMA_WITH_LOOPS_TILING_KEY 100000001001100
// start is float32, end is float32, weight is float32, y is float32, max dims in ub is 5 and nddma does not need loops
#define LERP_F32_NDDMA_WITHOUT_LOOPS_TILING_KEY 200000001000100
// start is float32, end is float32, weight is float32, y is float32, max dims in ub is 8 and nddma needs loops
#define LERP_F32_NDDMA_WITH_LOOPS_TILING_KEY 200000001001100
```

#### 机制与设计原因

Host 将 UB 切分轴到最内轴的轴数编进 schedule key。不超过 5 轴时，`BroadcastSetNddmaConfigWithoutLoop` 用 size=1 补齐5D descriptor，每输入每 tile 一次 NDDMA；6..8 轴时，`BroadcastNddmaWithLoop` 把 descriptor 之外最多 3 轴转为显式坐标循环，每次仍提交一个 5D copy。这样无需为低 rank 付出外循环，又不丢失 8 维 broadcast 能力。

#### 迁移边界

- 适用：可合轴且最终 rank<=8 的 broadcast CopyIn
- 前提：Host schedule key 必须与 Device helper 一一对应，POD dims/strides/split axis 必须保持元素单位
- 失效：合轴后 rank>8 在 Host Tiling 失败；未知 key 在 router 不写输出
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能用 without-loops helper 处理 UB 内轴数超过 5 的 Route；固定源码对这类 key 明确路由到 with-loops body。

## 模板与实现对比

| 对比维度 | R-001/R-002 FP16 | R-003/R-004 FP32 | R-005/R-006 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | opKey1；按内轴<=5/>5 | opKey2；按内轴<=5/>5 | opKey3；按内轴<=5/>5 | dtype 先编 opKey，schedule 再编 inner key |
| 模板与 dtype | half GM/UB，FP32 register | FP32 GM/UB/register | BF16 GM/UB，FP32 register | FP16/BF16 升精度，FP32 直算 |
| 循环与分核 | 普通/尾核 tile loops；成对 Route 差在 NDDMA 外循环 | 同左 | 同左 | 5D NDDMA 能力决定 schedule |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4 depth2 half TQue；无 TBuf/dynamic UB/window | 4 depth2 FP32 TQue | 4 depth2 BF16 TQue | 一 tile 三输入+一输出同时存活 |
| 内存总占用与复用 | 16E B，stride0 DB cache | 32E B，stride0 DB cache | 16E B，stride0 DB cache | divisor 精确匹配 4×depth2 queue bytes |
| 搬运/Vector/NDDMA API | B16 unpack/pack，5D NDDMA | FP32 normal load/store，5D NDDMA | B16 unpack/pack，5D NDDMA | rank<=5 单 descriptor，rank>5 外循环 |
| 同步、精度与 tail | queue；FP32 math；RINT；mask | queue；FP32 math；mask | queue；FP32 math；RINT；mask | 所有 Route 共用 stable formula 与 exact-byte CopyOut |
| fallback/失败边界 | unknown key no output；rank>8 Tiling 失败 | 同左 | 同左 | 无 SIMT/Cube fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档列为支持，但固定 `math/lerp` 源码无 A2 Host/Kernel/构建接线，不在本卡闭合 | 公开文档列为支持，但固定 `math/lerp` 源码无 A3 Host/Kernel/构建接线，不发布无后缀卡 | 公开文档、OpDef、binary config、CMake、arch35 Tiling/Kernel 一致支持 |
| Host/Launcher | 固定范围不可见 | 固定范围不可见 | BroadcastTiling，GetCoreNumAiv/UB，6 个 ACLNN 可达 key |
| Kernel/资源 | 固定范围不可见 | 固定范围不可见 | AIV-only，NDDMA CopyIn + MicroAPI Vector，4 depth-2 queues |

## 限制与失败边界

- 本卡仅覆盖 `aclnnInplaceLerp` Tensor weight 语义；不合并非原地 `aclnnLerp`、scalar-weight `aclnnInplaceLerps/aclnnLerps` 或 mixed-weight 内部 Route。
- API 要求 `selfRef/end/weight` 非 null、dtype 完全相同、可 broadcast，且 broadcast shape 等于 `selfRef`；否则返回 161001/161002 对应错误。
- 公开文档约束为 0..8 维。API 对全连续 Tensor 的 `selfRef` rank>8 只记 warning，而 BroadcastTiling 只在合轴后 `shapeLen>8` 才失败；不应将这一实现宽松当作公开承诺。
- `LerpTilingData` 序列化宏的精确 `sizeof/alignof/offsetof`、executor 全图 workspace 的闭式总字节数、Contiguous/ViewCopy 的内部 kernel 与 executor 精确释放时点在固定仓不可见。
- router 对 unknown key 直接 return 而不写 `y`；Host Tiling 应保证只产生登记 key。Device `workspace` 指针未读，不得由此推断 executor 总 workspace 为 0。
- A3 文档支持与固定 `math/lerp` 实现接线不闭合；按任务边界不借用 `experimental/math/lerp`，因而未创建 `inplace_lerp.md`。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/lerp/docs/aclnnLerp&aclnnInplaceLerp.md` | `aclnnInplaceLerp*` 文档 | 950 公开支持、Tensor weight 语义、shape/dtype/错误 |
| 入口声明/约束 | `math/lerp/op_api/aclnn_lerp_tensor.h` | `aclnnInplaceLerpGetWorkspaceSize/aclnnInplaceLerp` | C ABI 原型 |
| 入口实现 | `math/lerp/op_api/aclnn_lerp_tensor.cpp` | `CheckParams`, `CalculateResult`, `aclnnInplaceLerp*` | 校验、Contiguous、Lerp、ViewCopy、executor |
| 入口实现 | `math/lerp/op_api/lerp.cpp` | `l0op::Lerp` | broadcast 输出分配与 AICore launcher 加图 |
| Host/Tiling/Launcher | `math/lerp/op_host/arch35/lerp_tiling_arch35.h` | `LerpTilingData`, `LerpTiling` | POD 字段、compile info、Tiling class |
| Host/Tiling/Launcher | `math/lerp/op_host/arch35/lerp_tiling_arch35.cpp` | `GetOpKey`, `GetComputeMap`, `DoOpTiling`, `PostTiling` | dtype key、UB divisor、POD/block/workspace |
| Kernel entry/body | `math/lerp/op_kernel/lerp_apt.cpp` | `lerp` | AIV-only 入口与 10 key router，其中 6 个 ACLNN 可达 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f16_nddma_without_loops.h` | `LerpF16NddmaWithoutLoops` | R-001 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f16_nddma_with_loops.h` | `LerpF16NddmaWithLoops` | R-002 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f32_nddma_without_loops.h` | `LerpF32NddmaWithoutLoops` | R-003 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f32_nddma_with_loops.h` | `LerpF32NddmaWithLoops` | R-004 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_bf16_nddma_without_loops.h` | `LerpBf16NddmaWithoutLoops` | R-005 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_bf16_nddma_with_loops.h` | `LerpBf16NddmaWithLoops` | R-006 |
| Host/Tiling 直接依赖 | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling_base.h` | `BroadcastTilingParams/Data`, constants | broadcast Tiling 结构与 key 偏移 |
| Host/Tiling 直接依赖 | `src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp` | `BroadcastGetMaxElemNum/GetScheduleKey` | E 上限与 NDDMA schedule key |
| Host/Tiling 直接依赖 | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse`, `DoBrodcastTiling` | 合轴、stride、UB/core/tail 公式 |
| Kernel 直接依赖 | `pkg_inc/op_common/atvoss/util/broadcast_utils.h` | `BroadcastNddmaWithoutLoop/WithLoop` | 5D descriptor、外循环、offset 与连续 DMA |
| 构建接线 | `math/lerp/op_host/lerp_def.cpp` | `Lerp::Lerp` | `ascend950`, `lerp_apt`, dtype tuples |
| 构建接线 | `math/lerp/op_host/config/ascend950/lerp_binary.json` | `op_list` | 950 五个 binary dtype 组合 |
| 构建接线 | `math/lerp/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | 仅 `ascend950` + `arch35` |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 9 scalars + 8×`int64_t[8]`，Host 序列化后 Device 直读 | 字段顺序、int64 位宽、dims/strides 元素单位；不得臆造 offset | `LerpTilingData`, `GET_TILING_DATA` |
| 调用顺序 | Contiguous→Lerp→ViewCopy；Kernel 内 weight/end/start CopyIn→Compute→CopyOut | 原地语义必须最终提交到 `selfRef`；三输入必须在 Compute 前 ready | `CalculateResult`, `Process` |
| workspace/内存 | Lerp node 32 B workspace；4 个 depth2 queue；body 不读 workspace | executor 总 workspace 不等同 32 B；depth 必须计入 UB 总账 | `GetWorkspaceSize`, `Init` |
| 同步 | TQue 交接每个 tile，单 Device node 无 event/atomic | 不得在 `DeQue` 前覆盖 input slot，不得在 CopyOut 前释放 output | `CopyIn*`, `Compute3`, `CopyOut4` |
| 精度与尾块 | FP16/BF16 升 FP32，0.5 分支，RINT 降精度；`ubTail` + register mask + exact byte CopyOut | 不得只保留单公式或删除 mask/RINT；无效 lane 不得写 GM | 各 body `Compute3/CopyOut4` |
| 失败边界 | API dtype/shape/null 失败，Tiling rank/UB/core/key 失败，unknown Device key 无输出 | mixed-weight key 不得当作 `aclnnInplaceLerp` 可达 Route；A3 不得仅凭文档生成卡 | `CheckParams`, `GetOpKey`, `DoBrodcastTiling`, `lerp` |