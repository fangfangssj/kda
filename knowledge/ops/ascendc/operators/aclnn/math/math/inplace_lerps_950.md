---
type: Operator
title: "InplaceLerps 950 ACLNN 算子知识"
description: "说明 aclnnInplaceLerps 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/docs/aclnnLerps&aclnnInplaceLerps.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_api/aclnn_lerp_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_api/lerp.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_host/lerp_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_host/arch35/lerp_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_host/arch35/lerp_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/lerp_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f32_nddma_without_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f32_nddma_with_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_dtype_comb_0_nddma_without_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_dtype_comb_0_nddma_with_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_dtype_comb_1_nddma_without_loops.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_dtype_comb_1_nddma_with_loops.h
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
created_at: "2026-08-25T11:10:07Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceLerps 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceLerpsGetWorkspaceSize` + `aclnnInplaceLerps`，公开 ACLNN 两段式接口，无版本后缀；固定仓未发现更高版本入口。 |
| 计算语义 | 原地更新 `selfRef_i`。令 `d=end_i-selfRef_i`：`weight<0.5` 时算 `selfRef_i+weight*d`，否则算 `end_i+(weight-1)*d`；两式数学上均为 `selfRef_i+weight*(end_i-selfRef_i)`。 |
| 输入输出 | `selfRef`、`end` 为 ND，公开支持 FP16/FP32/BF16、rank 0–8、可非连续；二者 dtype 相同且广播结果必须等于 `selfRef` shape。`weight` 是 Host `aclScalar`，在 950 被物化为 FP32 标量 Tensor。输出写回 `selfRef`。 |
| 执行类型 | 算子拥有的核心节点是单 Kernel `AIV_VECTOR + NDDMA CopyIn`；entry 对 AIC 立即返回并声明 `KERNEL_TYPE_AIV_ONLY`。`Contiguous/Cast/ViewCopy` 是 ACLNN 框架边界，不作为本算子的 Cube 节点。 |
| Kernel 路线 | R-001/R-002：FP32；R-003/R-004：BF16 + FP32 weight；R-005/R-006：FP16 + FP32 weight。每个 dtype 族按 UB 内运行维数 `<=5`/`>5` 选择 NDDMA 无循环/有循环 body。 |
| 核心资源 | 三个 `VECIN` 双缓冲 Queue、一个 `VECOUT` 双缓冲 Queue、MicroAPI 寄存器、NDDMA/`DataCopyPad`；Lerp tiling 请求 32 B workspace，但 Kernel body 不读取该指针。ACLNN executor 总 workspace 另由框架汇总。 |
| 来源闭包 | `source_closed`：公开入口→复合 executor→Lerp launcher→950 OpDef/Tiling/POD→六个 Host 可达 key→AIV-only entry→六个 NDDMA/Vector body→`opbase` 广播切分与 NDDMA helper→950 构建接线已闭合；框架 `Contiguous/Cast/ViewCopy` 按明确边界标注。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对广播后的每个输出位置 `i`，读取原 `selfRef_i`、`end_i` 和同一个标量 `weight`。Device 实现先在 FP32 中求 `d=end_i-selfRef_i`，再按 FP32 `weight<0.5` 分支：

```text
candidate_lo = selfRef_i + weight * d
candidate_hi = end_i + (weight - 1.0) * d
selfRef_i    = select(weight < 0.5, candidate_lo, candidate_hi)
```

FP32 路线直接保存 FP32；FP16/BF16 路线先把 `selfRef/end` 升为 FP32，完成上述运算后以 `NO_SAT + CAST_RINT` 降回原 dtype。Kernel 先写连续临时 `y`，executor 最后以 `ViewCopy` 写回调用者的 `selfRef` view，所以可支持非连续原地目标。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND，公开 rank 0–8，可非连续 | FP16/FP32/BF16 | 输入 start，同时是原地输出；与 `end` dtype 相同；`broadcast(selfRef,end)` 必须等于 `selfRef.shape`。空 Tensor 在 ACLNN 层直接成功返回。 |
| `end` | ND，公开 rank 0–8，可非连续 | 与 `selfRef` 相同 | 与 `selfRef` 可广播；由 `Contiguous` 生成 Lerp 节点使用的连续输入。 |
| `weight` | Host scalar；Lerp 内部 Tensor shape 为 scalar，Tiling 以 `[1]` 处理 | 调用侧类型只要求可转为 `selfRef`；950 内部固定 FP32 | `InferScalarTensorDtype` 对 `IsRegBase` 返回 `DT_FLOAT`，`ConvertToTensor` 负责物化。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | 返回整个 executor 所需 Device workspace；数值由框架 `GetWorkspaceSize()` 汇总，固定仓未给出闭式公式。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并转移所有权，第二段在给定 stream 上执行。 |
| `workspace` | Device 字节区 | opaque | 由调用者按第一段返回值分配；Lerp 节点自身 tiling slice 为 32 B。 |
| `stream` | ACL runtime stream | `aclrtStream` | `CommonOpExecutorRun` 在该 stream 提交 executor 中的任务。 |

### 入口与版本

公开文档直接声明 `aclnnInplaceLerpsGetWorkspaceSize`/`aclnnInplaceLerps`，950 产品表为支持；固定 Commit 中没有带版本后缀的更新入口。`aclnnInplaceLerpsGetWorkspaceSize` 调用共享 `CalculateResult(selfRef,end,weight,selfRef,executor)`，这是原地 API 对公共计算图的复用，不等同于发布非原地 `aclnnLerps` 知识：本卡只保留 `selfRef` 同时作为输入/最终写回目标的语义与 950 标量 weight 可达路线。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize`、`executor`，`CheckNotNull` 校验 `selfRef/end/weight`（公共函数还校验作为 `out` 传入的同一 `selfRef`）。
2. `CheckDtypeValid` 在 950（`IsRegBase`）允许 `selfRef/end` 为 FP16、FP32、BF16，要求二者相同；作为 `out` 的 `selfRef` 也落在支持表内。
3. `BroadcastInferShape(selfRef.viewShape,end.viewShape)` 必须成功且结果等于 `selfRef.viewShape`。公开约束 rank 0–8；连续输入若超过 8 维，API 实现只告警，但这属于公开范围外，且 Tiling 仍要求合轴后不超过 8 维。
4. `selfRef->IsEmpty()` 时直接返回成功，不创建 Lerp 节点，也没有输出写任务。
5. 依次执行框架边界 `Contiguous(selfRef)`、`Contiguous(end)`；非 ND storage format 只告警。随后 `ConvertToTensor(weight, DT_FLOAT)`。
6. `l0op::Lerp` 再校验三输入广播，按广播 shape 分配连续临时输出，并通过 `ADD_TO_LAUNCHER_LIST_AICORE(Lerp, op::AI_CORE, ...)` 加入核心节点。
7. `Cast(lerpResult,selfRef.dtype)` 与 `ViewCopy(...,selfRef)` 构造后处理。这里 Cast 的目标 dtype 与 Lerp 输出 dtype 相同；是否消除为别名是框架实现细节，固定仓不可见。`ViewCopy` 完成真正的原地 view 写回。

### 参数、Workspace 与生命周期

第一段持有 `uniqueExecutor`，成功后以 `GetWorkspaceSize()` 得到整个复合图 workspace，再 `ReleaseTo(executor)` 转交调用者。临时连续 self/end、FP32 标量 Tensor、Lerp 连续结果以及 Cast 结果的分配和复用由 executor 管理，固定仓只公开构图顺序，不公开每个框架节点的 slice offset/extent。

Lerp Tiling 独立写 `workspaceSize_=32` 并在 `PostTiling` 写入 `workspaces[0]`；六个 body 虽接收 `workspace`，但 `Init/Process` 均未绑定或读取它。因此 32 B 只应视为该节点的 ABI 保留 workspace，不能替代第一段返回的 executor 总 workspace。

### 启动图与失败传播

```text
aclnnInplaceLerpsGetWorkspaceSize
  -> optional Contiguous(selfRef)        # 框架边界
  -> optional Contiguous(end)            # 框架边界
  -> ConvertToTensor(weight, FP32)        # 框架边界
  -> Lerp(start_contig,end_contig,weight_tensor -> y_contig)
       Host: TilingForLerp / blockDim=blockNum / POD=LerpTilingData / workspace=32 B
       Device: lerp -> one of R-001...R-006, AIV-only
  -> Cast(y_contig,selfRef.dtype)         # 同 dtype，框架边界
  -> ViewCopy(... -> selfRef view)        # 框架边界，完成原地提交
aclnnInplaceLerps(workspace,executor,stream)
  -> CommonOpExecutorRun                  # 按 executor/stream 顺序执行
```

| 节点/边 | producer / consumer | 交接介质与范围 | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| 连续化边 | `Contiguous` → `l0op::Lerp` | executor 临时 Tensor，shape/dtype 与对应输入相同 | executor 列表与同一 stream | executor 持有至消费者结束 | 任一返回空指针，第一段返回 `ACLNN_ERR_INNER_NULLPTR`，不提交后续节点。 |
| scalar 边 | `ConvertToTensor` → `Lerp` | FP32、shape scalar；Tiling 转为 `[1]` | executor 依赖 | executor 持有 | 转换失败终止构图。 |
| 核心边 | `Lerp` → `Cast` | 连续 GM `y`，广播后的 `selfRef.shape` | launcher/executor stream 顺序 | executor 临时输出 | launcher 添加失败使 `Lerp` 返回空，阻断后处理。 |
| 原地提交边 | `Cast` → `ViewCopy` → `selfRef` | 调用者目标 view；精确框架 workspace offset 固定仓不可见 | executor 在传入 stream 上排序 | `selfRef` 由调用者拥有，第二段完成前不得释放/覆盖 | Cast/ViewCopy 构图失败使第一段失败；第二段返回 `CommonOpExecutorRun` 状态。 |

## Route 与参数设计

### 分核与资源计算

Tiling 读取 AIV core 数 `C` 与 UB 字节数 `U`。三个输入 shape 与输出 shape 先把 scalar 变成 `[1]`、右对齐、按每轴广播模式合轴，并生成元素 stride；广播轴 stride 为 0。合轴后维数 `shapeLen` 必须 `<=8`。

`BroadcastGetComputeKey()` 固定为 1。950 标量入口可达的 `maxElemNum` 为：

```text
alignFactor = 256 B * 8 / 1 bit = 2048 elements
FP32:  floor((U * 8 / 256) / 2048) * 2048 = floor((U / 32) / 2048) * 2048
mixed: floor((U * 8 / 160) / 2048) * 2048 = floor((U / 20) / 2048) * 2048
```

从输出合轴维度最内侧向外累计 `innerProduct`；首次超过 `maxElemNum` 的轴为 `ubSplitAxis`。若全部放入 UB，则去掉首轴后作为内层积。随后：

```text
ubFormer = (shapeLen == 1) ? maxElemNum : floor(maxElemNum / innerProduct)
ubOuter  = ceil(outputDims[ubSplitAxis] / ubFormer)
ubTail   = outputDims[ubSplitAxis] - (ubOuter - 1) * ubFormer
fusedProduct = ubOuter * product(outputDims[0 : ubSplitAxis])
blockFormer  = ceil(fusedProduct / C)
blockNum     = ceil(fusedProduct / blockFormer)
blockTail    = fusedProduct - (blockNum - 1) * blockFormer
elemNum      = maxElemNum
```

本算子未设置 `preferMultiCore`，所以公共模板中的二次缩小 tile 分支不可达。`shapeLen-ubSplitAxis<=5` 产生 schedule key 100，否则 1100；最终 key 是 `opKey*10^14 + 1*10^6 + scheduleKey`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 NDDMA 无循环 | `200000001000100` / `LerpF32NddmaWithoutLoops` | `selfRef=end=y=FP32`，950 scalar Tensor 为 FP32，且 `shapeLen-ubSplitAxis<=5` | 同 dtype 且运行维数 `>5` → R-002；dtype 不同继续匹配。 |
| 2 | R-002 FP32 NDDMA 有循环 | `200000001001100` / `LerpF32NddmaWithLoops` | `selfRef=end=y=FP32` 且 `shapeLen-ubSplitAxis>5` | 无 fallback；未知 key entry 静默返回。 |
| 3 | R-003 BF16/FP32 weight NDDMA 无循环 | `400000001000100` / `LerpDtypeComb0NddmaWithoutLoops` | `selfRef=end=y=BF16`、950 weight Tensor=FP32，且运行维数 `<=5` | 运行维数 `>5` → R-004。 |
| 4 | R-004 BF16/FP32 weight NDDMA 有循环 | `400000001001100` / `LerpDtypeComb0NddmaWithLoops` | 同上且运行维数 `>5` | 无 fallback。 |
| 5 | R-005 FP16/FP32 weight NDDMA 无循环 | `500000001000100` / `LerpDtypeComb1NddmaWithoutLoops` | `selfRef=end=y=FP16`、950 weight Tensor=FP32，且运行维数 `<=5` | 运行维数 `>5` → R-006。 |
| 6 | R-006 FP16/FP32 weight NDDMA 有循环 | `500000001001100` / `LerpDtypeComb1NddmaWithLoops` | 同上且运行维数 `>5` | 无 fallback。 |

```text
if API/shape/dtype check fails: ACLNN error
if selfRef is empty: success, no Lerp route
weightTensorDtype = IsRegBase(A5) ? FP32 : selfRef.dtype
opKey = {FP32:2, BF16+FP32:4, FP16+FP32:5}; otherwise Tiling failure
collapse broadcast dimensions; require shapeLen <= 8
axisInsideUb = shapeLen - ubSplitAxis
scheduleKey = (axisInsideUb <= 5) ? 100 : 1100
tilingKey = opKey * 100000000000000 + 1000000 + scheduleKey
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | `200000001000100` | 950；四张量 FP32；合轴后 UB 内维数 1–5 | 单 Lerp AIV 节点 | `LerpF32NddmaWithoutLoops::{Init,Process,CopyIn0/1/2,Compute3,CopyOut4}` | 4×双缓冲 FP32 Queue；单次 5D NDDMA 或连续 `DataCopyPad`；Queue 同步 | `ubTail`、最后核 `blockTail`、MicroAPI mask；未知 key 无输出。 |
| R-002 | `200000001001100` | 同 dtype；UB 内维数 6–8 | 单 Lerp AIV 节点 | `LerpF32NddmaWithLoops` | 同 UB；外层枚举被 5D NDDMA 丢出的维度 | 同上。 |
| R-003 | `400000001000100` | 950；start/end/y BF16，weight FP32；UB 内维数 1–5 | 单 Lerp AIV 节点 | `LerpDtypeComb0NddmaWithoutLoops` | FP32 weight Queue + 3 个 BF16 Queue，均双缓冲；FP32 寄存器计算 | BF16 `CAST_RINT`，其余同 R-001。 |
| R-004 | `400000001001100` | 同 dtype；UB 内维数 6–8 | 单 Lerp AIV 节点 | `LerpDtypeComb0NddmaWithLoops` | mixed UB + NDDMA 外循环 | 同上。 |
| R-005 | `500000001000100` | 950；start/end/y FP16，weight FP32；UB 内维数 1–5 | 单 Lerp AIV 节点 | `LerpDtypeComb1NddmaWithoutLoops` | FP32 weight Queue + 3 个 FP16 Queue，均双缓冲；FP32 寄存器计算 | FP16 `CAST_RINT`，其余同 R-001。 |
| R-006 | `500000001001100` | 同 dtype；UB 内维数 6–8 | 单 Lerp AIV 节点 | `LerpDtypeComb1NddmaWithLoops` | mixed UB + NDDMA 外循环 | 同上。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| dtype 组合 | 950 scalar 强制 FP32；`GetOpKey` 仅 2/4/5 可达 | entry body 的 GM/UB 类型静态固定 | 全部 | 其它组合 Tiling `GRAPH_FAILED` | `InferScalarTensorDtype`、`LerpTiling::GetOpKey` |
| UB 容量 | `U>0` 且对 32 B/elem 或 20 B/elem 预算后向下对齐 2048 elem，结果非 0 | 四个 depth=2 Queue 必须同时容纳 tile | 全部 | `BroadcastTiling` 失败 | `BroadcastGetMaxElemNum`、`DoBrodcastTiling` |
| NDDMA 维数 | `shapeLen-ubSplitAxis<=5` | 一次 `DataCopy<T,5>` 可表达全部 UB 内轴 | R-001/003/005 | `>5` 选择有循环 Route | `BroadcastGetScheduleKey` |
| NDDMA 外循环 | `shapeLen-ubSplitAxis>5`，最大 8 | 固定 5D descriptor，额外最多 3 维由标量循环枚举 | R-002/004/006 | 合轴后 `shapeLen>8` 失败 | `BroadcastNddmaWithLoop` |
| 连续搬运 fast path | `outputStrides[ubSplitAxis]==inputStrides[ubSplitAxis]` | 无需广播 descriptor，单块 `DataCopyPad` 即可 | 全部每个输入独立 | 不等时使用 NDDMA multi-copy | `BroadcastNddmaWithoutLoop/WithLoop` |
| 广播缓存复用 | split 轴 input stride 为 0，且不是双缓冲首两槽/新 `ubOuter` 周期 | 广播 tile 不随该轴变化，可复用该 Queue 物理槽 | 全部每个输入独立 | stride 非 0 时每 tile 重载 | 各 body `CopyIn0/1/2` |
| Vector mask | `size=ubSplitSize*outputStride[splitAxis]` 窄化为 `uint32_t`，`vfLoopNum` 窄化为 `uint16_t` | MicroAPI 以 FP32 lane 数和寄存器循环驱动 tail | 全部 | 固定源码无显式溢出 fallback；UB 上界间接限制 | 各 body `Compute3` |
| 空域 | `selfRef->IsEmpty()` | 避免公共广播 Tiling 接收零维长度 | 全部 | ACLNN 成功且不 launch | `CalculateResult` |

### POD/ABI

`LerpTilingData` 的字段定义全部可见，但生成宏 `BEGIN_TILING_DATA_DEF/TILING_DATA_FIELD_DEF` 的展开和序列化 ABI 不在固定 `ops-math/opbase` 源码中，因此 `sizeof/alignof/offsetof` 均标为固定仓不可见，不能按自然布局臆算。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LerpTilingData` | `blockFormer` | `int64_t`/64 | 固定仓不可见 | UB tile 次数/普通核 | `DoOpTiling <- broadcastTilingData.blockFormer` | `Process` 的普通核循环数、block 起始序号 | `>=1`；空域已在 API 截断。 |
| 同上 | `ubFormer` | `int64_t`/64 | 固定仓不可见 | split-axis 元素 | `GetBlockSplitFactor` | full tile 大小、GM offset 步长、NDDMA 参数 | `floor(maxElemNum/innerProduct)`。 |
| 同上 | `ubOuter` | `int64_t`/64 | 固定仓不可见 | split-axis tile 数 | `ceil(dim/ubFormer)` | axes 进位与广播缓存周期 | `>=1`。 |
| 同上 | `ubTail` | `int64_t`/64 | 固定仓不可见 | split-axis 元素 | `dim-(ubOuter-1)*ubFormer` | 最后 split tile 有效大小 | `1..ubFormer`。 |
| 同上 | `blockTail` | `int64_t`/64 | 固定仓不可见 | UB tile 次数/尾核 | `fusedProduct-(blockNum-1)*blockFormer` | 最后 block 循环数 | `1..blockFormer`。 |
| 同上 | `shapeLen` | `int64_t`/64 | 固定仓不可见 | 合轴后维数 | `DoDimensionCollapse` | NDDMA descriptor/外循环范围 | `1..8`；scalar 先转 `[1]`。 |
| 同上 | `ubSplitAxis` | `int64_t`/64 | 固定仓不可见 | 轴下标 | `GetBlockSplitFactor` | 全部地址、loop、mask 公式 | `[0,shapeLen)`。 |
| 同上 | `dimProductBeforeUbInner` | `int64_t`/64 | 固定仓不可见 | 融合 UB tile 个数 | `fusedProduct` | `BroadcastGetAxesIndices` 的除法基数 | 名称易误导，实际赋值为 `fusedProduct`。 |
| 同上 | `elemNum` | `int64_t`/64 | 固定仓不可见 | 元素/Queue 槽 | `maxElemNum` | `InitBuffer` 单槽容量 | FP32 路线约 `U/32`，mixed 约 `U/20`，均 2048 elem 对齐。 |
| 同上 | `input0Dims` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `dims[0]` | 六条可达 body 不读取 | inactive 仍被序列化。 |
| 同上 | `input1Dims` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `dims[1]` | 六条可达 body 不读取 | inactive。 |
| 同上 | `input2Dims` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `dims[2]` | 六条可达 body 不读取 | scalar weight 合轴事实通过 stride 消费。 |
| 同上 | `outputDims` | `int64_t[8]` | 固定仓不可见 | 元素/轴 | `dims[3]` | axes 更新、NDDMA loopSize | 仅前 `shapeLen` 项有效。 |
| 同上 | `input0Strides` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `strides[0]` | start CopyIn、GM offset、broadcast 判断 | 广播轴为 0。 |
| 同上 | `input1Strides` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `strides[1]` | end CopyIn | 广播轴为 0。 |
| 同上 | `input2Strides` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `strides[2]` | weight CopyIn/缓存复用 | scalar weight 的有效广播轴为 0。 |
| 同上 | `outputStrides` | `int64_t[8]` | 固定仓不可见 | 元素 stride | `strides[3]` | UB extent、dst stride、GM output offset | 连续输出 stride。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `200000001000100` | `lerp_tiling_arch35.cpp::GetOpKey/GetComputeMap/DoOpTiling` | `LerpTilingData` | `lerp_apt.cpp::lerp` | `LerpF32NddmaWithoutLoops` + `broadcast_utils.h` | `lerp/CMakeLists.txt` ascend950/arch35 |
| R-002 | `200000001001100` | 同上 | 同上 | 同上 | `LerpF32NddmaWithLoops` + helper | 同上 |
| R-003 | `400000001000100` | 同上 | 同上 | 同上 | `LerpDtypeComb0NddmaWithoutLoops` + helper | 同上 |
| R-004 | `400000001001100` | 同上 | 同上 | 同上 | `LerpDtypeComb0NddmaWithLoops` + helper | 同上 |
| R-005 | `500000001000100` | 同上 | 同上 | 同上 | `LerpDtypeComb1NddmaWithoutLoops` + helper | 同上 |
| R-006 | `500000001001100` | 同上 | 同上 | 同上 | `LerpDtypeComb1NddmaWithLoops` + helper | 同上 |

## Kernel 实现

### 公共执行流程

`lerp(start,end,weight,y,workspace,tiling)` 首先在 `g_coreType==AIC` 时返回，随后声明 AIV-only、解析 POD 并按精确 key 构造一个 body；未知 key 直接返回，不写 `y`。所有可达 body 共享：绑定四个 GM Tensor；按 `elemNum` 初始化四个 depth=2 Queue；每核根据 `blockFormer/blockTail` 遍历 UB tile；分别广播搬入 weight、end、start；在 FP32 lane 中计算两种等价 lerp 形式并按 `weight<0.5` 选择；Queue 输出经单 block `DataCopyPad` 写连续 `y`。`workspace` 未消费。

### Route R-001：FP32 NDDMA 无循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `math/lerp/op_api/aclnn_lerp_scalar.cpp` | `InferScalarTensorDtype` | 950 `IsRegBase(npuArch)` | weight Tensor=FP32。 |
| Tiling | `math/lerp/op_host/arch35/lerp_tiling_arch35.cpp` | `GetOpKey` | start/end/weight/y 全 FP32 | `opKey=2`。 |
| Tiling helper | `opbase/.../broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | `shapeLen-ubSplitAxis<=5` | schedule 100。 |
| Router | `math/lerp/op_kernel/lerp_apt.cpp` | `lerp` | key `200000001000100` | `LerpF32NddmaWithoutLoops`。 |

#### Device 数据流

##### 函数级调用链

```text
TilingForLerp -> LerpTiling::DoTiling -> BroadcastTiling -> PostTiling(blockNum,key,POD,32 B)
  -> lerp(start,end,weight,y,workspace,tiling)
    -> LerpF32NddmaWithoutLoops::Init
    -> Process
      -> CopyIn0(weight) -> CopyIn1(end) -> CopyIn2(start)
      -> Compute3(FP32 piecewise lerp)
      -> CopyOut4(y)
  -> executor Cast/ViewCopy boundary -> selfRef
```

##### 源码等价伪码

```text
loopNum = lastBlock ? blockTail : blockFormer
axes = decode(blockFormer * blockIdx, outputDims, splitAxis, fusedProduct)
for u in [0, loopNum):
  if u != 0: increment axes through splitAxis with radix ubOuter
  split = (axes[splitAxis] == ubOuter-1) ? ubTail : ubFormer
  for input in [weight,end,start]:
    if inputStride[splitAxis] != 0 || u <= 1 || globalTile % ubOuter <= 1:
      if outputStride[splitAxis] != inputStride[splitAxis]: NDDMA_5D_without_loop(...)
      else: DataCopyPad(blockCount=1, blockLen=split*inputStride[splitAxis]*4)
  valid = split * outputStride[splitAxis]
  for regBase in ceil(valid/(VECTOR_REG_WIDTH/4)):
    mask = UpdateMask<float>(valid)
    w,endv,startv = load_fp32(mask)
    delta = endv-startv
    lo = startv + w*delta
    hi = endv + (w-1)*delta
    store(select(w<0.5,lo,hi), mask)
  gm = sum(axes[j]*outStride[j],j<splitAxis) + axes[splitAxis]*outStride[splitAxis]*ubFormer
  DataCopyPad(y[gm], outQ, blockCount=1, blockLen=valid*4)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM start/end/weight | `BroadcastGetGmOffset(axes,inputStride,splitAxis,ubFormer)` | 当前广播 tile | FP32/ND | caller/executor | `CopyIn2/1/0` | caller或executor 至 Kernel 完成 | NDDMA 5D；stride 相等走单块。 |
| `queIn0/1/2` | Queue 槽 | 每槽 `elemNum*4` B | FP32 | CopyIn | Compute3 | `TPipe`，Alloc→EnQue→DeQue→Free | depth=2；stride=0 时隔槽复用广播数据。 |
| `queOut0` | Queue 槽 | 每槽 `elemNum*4` B | FP32 | Compute3 | CopyOut4 | `TPipe`，Alloc→EnQue→DeQue→Free | depth=2。 |
| RegTensor/MaskReg | VEC scope | 固定仓未给寄存器字节数 | FP32 | MicroAPI load/compute | store | 单次 `Compute3` scope | tail 由 `UpdateMask<float>`。 |
| GM y | `BroadcastGetGmOffset(axes,outStride,...)` | `valid*4` B | FP32/连续 ND | CopyOut4 | Cast/ViewCopy 边界 | executor 临时 | 单块 `DataCopyPad`。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["queIn0 weight · depth=2 · 2×elemNum×4 B"]:1
  in1["queIn1 end · depth=2 · 2×elemNum×4 B"]:1
  in2["queIn2 start · depth=2 · 2×elemNum×4 B"]:1
  out0["queOut0 y · depth=2 · 2×elemNum×4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4 queues * 2 slots * elemNum * 4 B = 32 * elemNum B
elemNum = align_down(U / 32, 2048 elements)
UB_total <= U
dynamic_ub = N/A；AIV 使用静态 TPipe Queue
workspace_total = 32 B（Lerp 节点 ABI；body 未读取）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三个 input Queue 的两个槽 | CopyIn | Compute3 DeQue | 后续同奇偶 tile | EnQue/DeQue；消费后 FreeTensor | stride 非 0 时每轮重写；stride=0 时条件保证周期首两个槽先初始化。 |
| output Queue 两槽 | Compute3 | CopyOut4 | 后续 tile | EnQue/DeQue，CopyOut 后 FreeTensor | CopyOut 完成前不覆盖。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | TQue EnQue/DeQue | 三个 CopyIn | Compute3 | 三个 UB tile | NDDMA padding lane 不参与，Vector mask 只覆盖 `valid`。 |
| Compute→CopyOut | TQue EnQue/DeQue | Compute3 | CopyOut4 | output UB tile | 全程 FP32；`weight<0.5` 选择计算顺序。 |
| 核/tile tail | `blockTail/ubTail/UpdateMask` | Tiling/POD | Process/MicroAPI | 最后核、最后 split、最后寄存器 | 仅写 `valid` 元素。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp` | `GetComputeMap/DoOpTiling` | FP32 divisor 256，生成 key/POD/blockNum。 |
| Kernel entry/router | `lerp_apt.cpp` | `lerp` key `200000001000100` | AIV-only 到 FP32 无循环 body。 |
| Device body/helper | `lerp_f32_nddma_without_loops.h`、`broadcast_utils.h` | `Process/CopyIn*/Compute3/CopyOut4`、`BroadcastNddmaWithoutLoop` | 5D NDDMA、双表达式 FP32 lerp、masked tail。 |

### Route R-002：FP32 NDDMA 有循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `aclnn_lerp_scalar.cpp`、`lerp_tiling_arch35.cpp` | `InferScalarTensorDtype/GetOpKey` | 四张量 FP32 | `opKey=2`。 |
| Tiling helper | `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | `shapeLen-ubSplitAxis>5` | schedule 1100。 |
| Router | `lerp_apt.cpp` | `lerp` | `200000001001100` | `LerpF32NddmaWithLoops`。 |

#### Device 数据流

##### 函数级调用链

```text
TilingForLerp -> BroadcastTiling -> lerp
  -> LerpF32NddmaWithLoops::Init -> Process
    -> CopyIn0/1/2 -> BroadcastNddmaWithLoop
      -> BroadcastSetNddmaConfigWithLoop + outer-index loop + DataCopy<T,5>
    -> Compute3 -> CopyOut4
  -> Cast/ViewCopy boundary -> selfRef
```

##### 源码等价伪码

```text
Process/core/tile/compute/copyout 与 R-001 相同
对每个需要广播的 input tile:
  descriptor = last 5 running dims(outputDims,inputStrides,outputStrides)
  outerCount = product(outputDims[splitAxis+1 : shapeLen-5]) * ubSplitSize
  nddmaIndices[3] = 0
  for k in [0,outerCount):
    if k != 0: increment thrown-dimension indices
    ubOff = dot(indices, outputStrides[splitAxis : shapeLen-5])
    gmOff = dot(indices, inputStrides[splitAxis : shapeLen-5])
    DataCopy<T,5>(ub[ubOff], gm[base+gmOff], descriptor)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM→UB | base GM offset + thrown-dim `gmOff`；UB `ubOff` | 每次固定 5D descriptor，合计当前 tile | FP32/broadcast ND | `BroadcastNddmaWithLoop` | Compute3 | 当前 Queue 槽 | 最多 3 个外层维度显式循环。 |
| 4 个 Queue | 同 R-001 | 各 `2*elemNum*4` B | FP32 | CopyIn/Compute | Compute/CopyOut | TPipe | 布局不因 NDDMA 外循环改变。 |
| GM y | 同 R-001 | `valid*4` B | FP32 | CopyOut4 | ViewCopy | executor | 连续单块。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weight Q · 2×elemNum×4 B"]:1
  in1["end Q · 2×elemNum×4 B"]:1
  in2["start Q · 2×elemNum×4 B"]:1
  out0["y Q · 2×elemNum×4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 32 * elemNum B <= U
dynamic_ub = N/A
workspace_total = 32 B；NDDMA 外循环不增加 UB/workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output Queue 两槽 | CopyIn/Compute | Compute/CopyOut | 下一同奇偶 tile | Queue EnQue/DeQue/FreeTensor | 同 R-001；一个 tile 的全部 NDDMA 外循环完成后才 EnQue。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA 外循环 | 同一函数顺序 + Queue EnQue | 多次 5D DataCopy | Compute DeQue | 完整 input tile | 外层 count 覆盖被 5D descriptor 丢出的 1–3 维。 |
| Vector/CopyOut | 与 R-001 相同 | Compute3 | CopyOut4 | 有效 FP32 元素 | `ubTail` + mask。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp`、`broadcast_tiling_base.cpp` | `DoOpTiling/BroadcastGetScheduleKey` | UB 内维数大于 5 生成 1100。 |
| Kernel entry/router | `lerp_apt.cpp` | key `200000001001100` | 映射有循环 body。 |
| Device body/helper | `lerp_f32_nddma_with_loops.h`、`broadcast_utils.h` | `BroadcastNddmaWithLoop` | 外层索引循环分解为多次 5D NDDMA。 |

### Route R-003：BF16 + FP32 weight NDDMA 无循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lerp_scalar.cpp` | `InferScalarTensorDtype` | 950 + selfRef BF16 | weight Tensor FP32。 |
| Tiling | `lerp_tiling_arch35.cpp` | `GetOpKey` | BF16/BF16/FP32/BF16 | `opKey=4`。 |
| Helper/Router | `broadcast_tiling_base.cpp`、`lerp_apt.cpp` | schedule/key branch | UB 内维数 `<=5`，key `400000001000100` | mixed BF16 无循环 body。 |

#### Device 数据流

##### 函数级调用链

```text
TilingForLerp -> lerp
  -> LerpDtypeComb0NddmaWithoutLoops::Init -> Process
    -> CopyIn0(FP32 weight) -> CopyIn1(BF16 end) -> CopyIn2(BF16 start)
    -> Compute3(BF16 unpack -> FP32 lerp -> BF16 RINT pack)
    -> CopyOut4(BF16 y)
  -> Cast/ViewCopy boundary -> selfRef
```

##### 源码等价伪码

```text
core/tile/NDDMA/offset 同 R-001，元素字节按各输入 dtype
w = load_fp32(weightQ)
endf = cast_fp32(unpack_bf16(endQ), ZEROING)
startf = cast_fp32(unpack_bf16(startQ), ZEROING)
delta = endf-startf
lo = startf+w*delta
hi = endf+(w-1)*delta
res = select(w<0.5,lo,hi)
outbf16 = cast_bf16(res, NO_SAT, CAST_RINT, ZEROING)
pack_b32(outQ, outbf16, mask)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `queIn0` | Queue 槽 | `2*elemNum*4` B | FP32 weight | CopyIn0 | Compute3 | TPipe | scalar broadcast通常 stride 0，可隔槽复用。 |
| `queIn1/queIn2` | Queue 槽 | 各 `2*elemNum*2` B | BF16 end/start | CopyIn1/2 | Compute3 | TPipe | NDDMA/单块 fast path。 |
| `queOut0` | Queue 槽 | `2*elemNum*2` B | BF16 y | Compute3 | CopyOut4 | TPipe | depth=2。 |
| RegTensor | VEC scope | 固定仓不可见 | BF16 load + FP32 compute | Compute3 | BF16 pack | 单寄存器迭代 | FP32 累计，RINT 降精度。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weight FP32 Q · 2×elemNum×4 B"]:1
  in1["end BF16 Q · 2×elemNum×2 B"]:1
  in2["start BF16 Q · 2×elemNum×2 B"]:1
  out0["y BF16 Q · 2×elemNum×2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 8*elemNum + 4*elemNum + 4*elemNum + 4*elemNum = 20*elemNum B
elemNum = align_down(U / 20, 2048 elements)
UB_total <= U
dynamic_ub = N/A
workspace_total = 32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input Queue 两槽 | CopyIn0/1/2 | Compute3 | 后续 tile | Queue 协议 + FreeTensor | 广播缓存条件与 R-001 相同。 |
| output Queue 两槽 | Compute3 | CopyOut4 | 后续 tile | CopyOut 后 FreeTensor | BF16 pack 完成后才 EnQue。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入升精度 | `DIST_UNPACK_B16` + `Cast<float,bfloat16_t>` | BF16 Queue | FP32 arithmetic | start/end lanes | `MaskMergeMode::ZEROING`；tail mask。 |
| 输出降精度 | `Cast<bfloat16_t,float>` + `DIST_PACK_B32` | FP32 result | BF16 out Queue | result lanes | `NO_SAT + CAST_RINT`。 |
| Queue/tail | EnQue/DeQue + `ubTail`/mask | CopyIn/Compute | Compute/CopyOut | tile 生命周期 | 只提交有效 BF16 字节。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp` | `GetOpKey/GetComputeMap` | opKey4，bufferDivisor 160。 |
| Kernel entry/router | `lerp_apt.cpp` | key `400000001000100` | BF16 mixed 无循环 body。 |
| Device body/helper | `lerp_dtype_comb_0_nddma_without_loops.h` | `Compute3/castTrait0/castTrait1` | BF16→FP32→BF16 RINT。 |

### Route R-004：BF16 + FP32 weight NDDMA 有循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `aclnn_lerp_scalar.cpp`、`lerp_tiling_arch35.cpp` | scalar dtype + `GetOpKey` | BF16/BF16/FP32/BF16 | opKey4。 |
| Helper | `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | UB 内维数 `>5` | 1100。 |
| Router | `lerp_apt.cpp` | key `400000001001100` | 精确匹配 | `LerpDtypeComb0NddmaWithLoops`。 |

#### Device 数据流

##### 函数级调用链

```text
TilingForLerp -> lerp -> LerpDtypeComb0NddmaWithLoops::Process
  -> three BroadcastNddmaWithLoop CopyIn nodes inside one Kernel
  -> BF16 unpack / FP32 piecewise lerp / BF16 RINT pack
  -> BF16 DataCopyPad y -> ViewCopy boundary -> selfRef
```

##### 源码等价伪码

```text
对每个 input：若 split stride 不同，枚举被固定 5D NDDMA 丢出的维度，累加 gmOff/ubOff 后多次 DataCopy<T,5>
所有外循环填满当前 Queue 槽后 EnQue
Compute 与 R-003 完全同序：BF16 unpack/cast -> FP32 两候选 -> select -> RINT BF16 pack
CopyOut 使用 valid*sizeof(bfloat16_t)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| NDDMA GM→UB | base + thrown-dim dot(stride) | 当前 tile，多次 5D | FP32/BF16 | CopyIn0/1/2 | Compute3 | 当前 Queue 槽 | 外循环不额外分配中间 buffer。 |
| 四 Queue | 同 R-003 | 合计 `20*elemNum` B | mixed | CopyIn/Compute | Compute/CopyOut | TPipe | depth=2。 |
| GM y | output GM offset | `valid*2` B | BF16 | CopyOut4 | ViewCopy | executor | 连续单块。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weight FP32 · 2×elemNum×4 B"]:1
  in1["end BF16 · 2×elemNum×2 B"]:1
  in2["start BF16 · 2×elemNum×2 B"]:1
  out0["y BF16 · 2×elemNum×2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 20 * elemNum B <= U
dynamic_ub = N/A
workspace_total = 32 B；NDDMA 外循环只复用同一 Queue 槽
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue 双槽 | 多次 NDDMA / Compute | Compute / CopyOut | 后续 tile | 填完才 EnQue，消费后 FreeTensor | thrown-dim offset 必须覆盖互不重叠 UB 子域。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA 汇集 | 函数内顺序 + EnQue | 外层多次 copy | Compute DeQue | 完整 mixed input tile | descriptor 尾部由 dims/strides 表达。 |
| Vector | 与 R-003 相同 | Compute3 | CopyOut4 | FP32 中间/BF16 输出 | RINT + mask。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling_base.cpp` | schedule 1100 | 运行维数大于 5。 |
| Kernel entry/router | `lerp_apt.cpp` | key `400000001001100` | mixed BF16 有循环 body。 |
| Device body/helper | `lerp_dtype_comb_0_nddma_with_loops.h`、`broadcast_utils.h` | `Process/BroadcastNddmaWithLoop` | 无额外 UB 的 5D 分段网络。 |

### Route R-005：FP16 + FP32 weight NDDMA 无循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_lerp_scalar.cpp` | `InferScalarTensorDtype` | 950 + FP16 selfRef | weight FP32。 |
| Tiling | `lerp_tiling_arch35.cpp` | `GetOpKey` | FP16/FP16/FP32/FP16 | opKey5。 |
| Helper/Router | `broadcast_tiling_base.cpp`、`lerp_apt.cpp` | schedule/key | UB 内维数 `<=5`，key `500000001000100` | FP16 mixed 无循环 body。 |

#### Device 数据流

##### 函数级调用链

```text
TilingForLerp -> lerp -> LerpDtypeComb1NddmaWithoutLoops::Process
  -> CopyIn0(FP32 weight), CopyIn1(FP16 end), CopyIn2(FP16 start)
  -> Compute3(FP16 unpack -> FP32 piecewise lerp -> FP16 RINT pack)
  -> CopyOut4(FP16 y) -> ViewCopy boundary -> selfRef
```

##### 源码等价伪码

```text
地址、tile、5D NDDMA 与 R-003 相同
w = load_fp32(weight)
endf,startf = cast_fp32(unpack_fp16(end,start))
res = select(w<0.5, startf+w*(endf-startf), endf+(w-1)*(endf-startf))
outfp16 = cast_half(res, NO_SAT, CAST_RINT, ZEROING)
pack_b32(outQ,outfp16,mask); DataCopyPad(valid*2 B)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `queIn0` | Queue slot | `2*elemNum*4` B | FP32 weight | CopyIn0 | Compute3 | TPipe | scalar broadcast缓存。 |
| `queIn1/2` | Queue slot | 各 `2*elemNum*2` B | FP16 | CopyIn1/2 | Compute3 | TPipe | 5D NDDMA 或单块。 |
| `queOut0` | Queue slot | `2*elemNum*2` B | FP16 | Compute3 | CopyOut4 | TPipe | RINT pack。 |
| GM y | output offset | `valid*2` B | FP16 | CopyOut4 | ViewCopy | executor | 只写有效 tail。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weight FP32 · 2×elemNum×4 B"]:1
  in1["end FP16 · 2×elemNum×2 B"]:1
  in2["start FP16 · 2×elemNum×2 B"]:1
  out0["y FP16 · 2×elemNum×2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 20 * elemNum B
elemNum = align_down(U / 20, 2048 elements)
UB_total <= U
dynamic_ub = N/A
workspace_total = 32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output Queue 两槽 | CopyIn/Compute | Compute/CopyOut | 后续 tile | EnQue/DeQue/FreeTensor | 与 R-003 相同。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| FP16→FP32 | unpack + Cast | input Queue | arithmetic | start/end | ZEROING mask。 |
| FP32→FP16 | `NO_SAT + CAST_RINT` + pack | selected result | out Queue | output | mask 保证尾 lane 不写。 |
| Queue/GM | Queue + DataCopyPad | Compute3 | CopyOut4 | tile | `blockTail/ubTail/valid`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `lerp_tiling_arch35.cpp` | opKey5/divisor160 | mixed FP16 tile 预算。 |
| Kernel entry/router | `lerp_apt.cpp` | key `500000001000100` | FP16 mixed 无循环 body。 |
| Device body/helper | `lerp_dtype_comb_1_nddma_without_loops.h` | `Compute3/castTrait*` | FP16 输入升精度与 RINT 输出。 |

### Route R-006：FP16 + FP32 weight NDDMA 有循环

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | `aclnn_lerp_scalar.cpp`、`lerp_tiling_arch35.cpp` | scalar dtype/`GetOpKey` | FP16/FP16/FP32/FP16 | opKey5。 |
| Helper | `broadcast_tiling_base.cpp` | schedule | UB 内维数 `>5` | 1100。 |
| Router | `lerp_apt.cpp` | key `500000001001100` | 精确 key | `LerpDtypeComb1NddmaWithLoops`。 |

#### Device 数据流

##### 函数级调用链

```text
TilingForLerp -> lerp -> LerpDtypeComb1NddmaWithLoops::Process
  -> three BroadcastNddmaWithLoop CopyIn paths
  -> FP16 unpack / FP32 piecewise lerp / FP16 RINT pack
  -> FP16 CopyOut -> ViewCopy boundary -> selfRef
```

##### 源码等价伪码

```text
每 input tile：在固定 5D NDDMA descriptor 外枚举最多 3 维，按 input/output stride 求 GM/UB offset
Queue EnQue 后执行与 R-005 相同的 FP32 两候选计算和 FP16 RINT pack
最后用 gmOffset 与 valid*2 B 写连续 y；executor 再写回 selfRef view
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| NDDMA GM→UB | base + `gmOff` / `ubOff` | 多次 5D 合成一个 tile | FP32 weight、FP16 data | CopyIn0/1/2 | Compute3 | 当前 Queue slot | 无中间 workspace。 |
| 四 Queue | 同 R-005 | 合计 `20*elemNum` B | mixed | CopyIn/Compute | Compute/CopyOut | TPipe | depth=2。 |
| GM y | output offset | `valid*2` B | FP16 | CopyOut4 | ViewCopy | executor | 连续单块。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["weight FP32 · 2×elemNum×4 B"]:1
  in1["end FP16 · 2×elemNum×2 B"]:1
  in2["start FP16 · 2×elemNum×2 B"]:1
  out0["y FP16 · 2×elemNum×2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 20 * elemNum B <= U
dynamic_ub = N/A
workspace_total = 32 B；外循环不改变 owner/lifetime
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue 双槽 | 完整 NDDMA 外循环/Compute | Compute/CopyOut | 后续 tile | EnQue/DeQue/FreeTensor | 同一 slot 的外循环写完后才发布。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| NDDMA→Vector | Queue 发布 | 多次 5D copy | Compute3 | input tile | thrown dims 全覆盖后发布。 |
| Vector→GM | Queue + DataCopyPad | FP32计算/FP16 RINT | CopyOut4 | output tile | mask、`ubTail`、`blockTail`。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `broadcast_tiling_base.cpp` | `BroadcastGetScheduleKey` | 运行维数 >5。 |
| Kernel entry/router | `lerp_apt.cpp` | key `500000001001100` | FP16 mixed 有循环 body。 |
| Device body/helper | `lerp_dtype_comb_1_nddma_with_loops.h`、`broadcast_utils.h` | `BroadcastNddmaWithLoop/Compute3` | 5D 分段搬运 + FP32 计算 + FP16 RINT。 |

## 算子特定优化

### Trick T-001：950 标量统一 FP32 物化并命中 mixed-dtype body

#### 触发与源码依据

- 适用 Route：R-001–R-006。
- 触发条件：当前 NPU `IsRegBase(npuArch)`，即本卡 950 平台。
- Golden 来源：[`math/lerp/op_api/aclnn_lerp_scalar.cpp:128-134`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_api/aclnn_lerp_scalar.cpp#L128-L134)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
static inline op::DataType InferScalarTensorDtype(const aclTensor* self) {
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  if (IsRegBase(npuArch)) {
    return op::DataType::DT_FLOAT;
  }
  return self->GetDataType();
}
```

#### 机制与设计原因

950 不随 FP16/BF16 `selfRef` 降低 scalar Tensor dtype，而是保留 FP32 weight。Host 因此只会生成 opKey 2/4/5，mixed body 直接从 FP32 Queue 读取 weight，start/end 才做升精度，避免先把 Host scalar 量化到 16 bit 再参与 FP32 插值。

#### 迁移边界

- 适用：标量参数需在 RegBase Vector Kernel 中与低精度 Tensor 混合计算。
- 前提：OpDef、Tiling key 和 Device body都显式支持 `(BF16|FP16, FP32 scalar)` 组合。
- 失效：非 RegBase 平台返回 `self.dtype`；不能复用本卡 950 mixed key。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从 ABI 约束看，直接把 950 FP16/BF16 scalar Tensor 接到 all-16-bit key 会与公开入口的 Host producer 不一致。

### Trick T-002：按 weight 阈值选择两种代数等价计算顺序

#### 触发与源码依据

- 适用 Route：R-001–R-006。
- 触发条件：逐 FP32 lane 比较 `weight<0.5`。
- Golden 来源：[`math/lerp/op_kernel/arch35/lerp_dtype_comb_1_nddma_without_loops.h:166-175`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_dtype_comb_1_nddma_without_loops.h#L166-L175)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg5, vreg2, vreg4, preg0);
AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg4, vreg0, vreg5, preg0);
AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg7, vreg0, vreg6, preg0);
AscendC::MicroAPI::MulAddDst<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vreg2, vreg7, vreg5, preg0);
AscendC::MicroAPI::Select<float>(vreg8, vreg4, vreg2, preg97);
```

#### 机制与设计原因

`vreg4` 初始为 start，生成 `start+weight*(end-start)`；`vreg2` 初始为 end，生成 `end+(weight-1)*(end-start)`。两式实数等价，但浮点加法锚点和乘数范围不同；源码按 0.5 阈值确定计算顺序，明确改变舍入路径，并用 `Select` 避免标量控制流分叉。所有算术和比较都受同一个 FP32 mask 保护。

#### 迁移边界

- 适用：插值/仿射组合中两端点等价表达式可按权重区域选择。
- 前提：两候选均满足目标 API 的数值语义，且额外寄存器和 Vector 指令预算可接受。
- 失效：若目标算子要求固定逐操作舍入顺序，不能替换为此分段形式。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得把 `weight<0.5` 改成 Host 全 Tensor 分支，因为本实现支持 weight Tensor 的广播通路，比较语义是逐 lane。

### Trick T-003：split 轴广播输入按双缓冲槽缓存

#### 触发与源码依据

- 适用 Route：R-001–R-006 的任一 input；标量 weight 必然具有广播 stride 0。
- 触发条件：`inputStride[ubSplitAxis]==0`，且当前 tile 不处于 Queue 两槽初始化或新 `ubOuter` 周期的前两槽。
- Golden 来源：[`math/lerp/op_kernel/arch35/lerp_f32_nddma_without_loops.h:79-88`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/lerp/op_kernel/arch35/lerp_f32_nddma_without_loops.h#L79-L88)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
bufferIn0_ = queIn0_.AllocTensor<float>();
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

Queue depth 为 2。广播输入沿 split 轴不变时，先填充两个物理槽，之后同奇偶槽再次 `AllocTensor` 时沿用其旧内容；跨到新的外层广播坐标周期时再初始化前两槽。该条件直接省去标量 weight 及其它 split-axis 广播输入的重复 GM→UB NDDMA，不减少 Queue 容量。

#### 迁移边界

- 适用：输入沿 tile 推进轴 stride 为 0，且双缓冲槽与迭代奇偶稳定对应。
- 前提：Queue depth、`ubOuter` 周期和 global tile 序号公式保持一致；外层坐标变化时必须重载两槽。
- 失效：stride 非 0 时每 tile 都重载；修改 depth 时必须重写初始化窗口。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若只在整个 block 的第一次迭代装载一个槽，下一次 `AllocTensor` 可能取得未初始化的另一个双缓冲槽；源码明确初始化两个槽并在周期边界重复这一过程。

### Trick T-004：以固定 5D NDDMA descriptor 加最多三维外循环闭合 rank 8

#### 触发与源码依据

- 适用 Route：R-002、R-004、R-006。
- 触发条件：`shapeLen-ubSplitAxis>5`；合轴后总 rank 仍 `<=8`。
- Golden 来源：[`opbase/pkg_inc/op_common/atvoss/util/broadcast_utils.h:197-216`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/broadcast_utils.h#L197-L216)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t nddmaIndices[NDDMA_THROW_DIMS] = {0};
int64_t nddmaProduct = BroadcastFuseAxes(outputDims, ubSplitAxis + 1, shapeLen - NDDMA_MAX_DIMS) * ubSplitSize;
for (int64_t i = 0; i < nddmaProduct; i++) {
  if (i != 0) {
    BroadcastUpdateNddmaAxesIndices(nddmaIndices, outputDims, ubSplitAxis, shapeLen - NDDMA_MAX_DIMS - 1 - ubSplitAxis);
  }
  int64_t nddmaUbOffset = BroadcastGetNddmaOffset(nddmaIndices, outputStrides, ubSplitAxis, shapeLen - NDDMA_MAX_DIMS);
  int64_t nddmaGmOffset = BroadcastGetNddmaOffset(nddmaIndices, inputStrides, ubSplitAxis, shapeLen - NDDMA_MAX_DIMS);
```

#### 机制与设计原因

硬件 multi-copy descriptor 固定表达 5 维；公共 helper 把最内 5 维写入 descriptor，把额外 1–3 维变成显式索引与 GM/UB offset。这样复用同一 Queue 槽和同一计算 body，不需 workspace 中间转置，也不把高 rank 广播退回其他执行族。

#### 迁移边界

- 适用：合轴 rank 至多 8、核心数据变换可由 5D NDDMA覆盖内层。
- 前提：`NDDMA_THROW_DIMS=3`、offset 为元素单位、input/output stride 正确表达广播。
- 失效：合轴后 rank >8 时 Host Tiling 失败；不能继续扩大标量索引数组。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；把 `blockLen/srcStride/dstStride` 当字节写入 multi-copy descriptor会破坏本 helper 的元素 stride 约定。

## 模板与实现对比

| 对比维度 | R-001 FP32 no-loop | R-002 FP32 loop | R-003/R-005 mixed no-loop | R-004/R-006 mixed loop | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | opKey2，运行维数<=5 | opKey2，>5 | opKey4/5，<=5 | opKey4/5，>5 | dtype 先定 opKey，UB 内维数再定 schedule。 |
| 模板与 dtype | 全 FP32 | 全 FP32 | FP32 weight + BF16/FP16 data | 同 mixed | 950 scalar 固定 FP32。 |
| 循环与分核 | 每核 UB tile loop；一次 5D NDDMA | 再加 NDDMA thrown-dim loop | 同 R-001 | 同 R-002 | 固定 NDDMA 最大 5 维。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4 个 FP32 depth2 Queue；无 TBuf/dynUB | 同左 | 1 FP32 + 3×16-bit depth2 Queue | 同左 | `bufferDivisor` 精确覆盖 Queue 字节。 |
| 内存总占用与复用 | `32*elemNum` B | 同左 | `20*elemNum` B | 同左 | 广播输入均可按双槽缓存。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 单次 `DataCopy<T,5>` 或 `DataCopyPad`；MicroAPI FP32 | 多次 `DataCopy<T,5>` | mixed NDDMA + unpack/cast/pack | mixed + 多次 NDDMA | 无 SIMT、通信、AIC。 |
| 同步、精度与 tail | Queue；全 FP32；mask | 同左 | Queue；FP32 compute，RINT 16-bit；mask | 同左 | `ubTail/blockTail/UpdateMask`。 |
| fallback/失败边界 | >5 转 loop Route | 无；未知 key静默返回 | >5 转 loop Route | 无 | dtype/UB/rank 失败在 Tiling 终止。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档列支持，但固定 Commit 无 A2 活动实现闭包，本卡不覆盖 | 公开文档列支持，但固定 Commit 无 A3 Host/Tiling/Kernel/构建接线，本卡不覆盖 | 公开文档支持，且 `ascend950 + arch35` 源码与构建闭合 |
| Host/Launcher | 固定仓不可见 | 固定仓不可见 | scalar 强制 FP32；`opKey=2/4/5`；公共 BroadcastTiling |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | AIV-only，NDDMA+MicroAPI；20/32 B per elem UB 预算 |

## 限制与失败边界

- 公开 API rank 为 0–8；连续输入大于 8 维时 API 代码只告警，但属于公开范围外，且 Host Tiling 对合轴后的 `shapeLen>8` 明确失败。
- `selfRef/end` 必须同 dtype且广播输出必须正好等于 `selfRef.shape`；原地接口不能扩大 `selfRef`。
- 950 `weight` 必须能由框架转换为 FP32 scalar Tensor；转换失败返回内部空指针错误。
- `BroadcastTiling` 要求 `coreNum>0`、`ubSize>0`、对齐后 `maxElemNum>0`。FP32/mixed body的 UB 预算分别为 32/20 B 每元素。
- POD 生成宏的 `sizeof/alignof/offsetof` 在固定 `ops-math@165d41...` 与直接依赖 `opbase@d574...` 中不可见；不可据自然布局重建二进制 ABI。
- `size` 到 `uint32_t`、`vfLoopNum` 到 `uint16_t` 的窄化没有独立 guard；固定源码仅通过 UB tile 上界间接约束。
- Lerp tiling 声明 32 B workspace而 body 不使用；ACLNN executor 总 workspace及 `Contiguous/Cast/ViewCopy` 的内部 slice/Kernel 路由属于框架边界，固定仓不可见。
- 未知 TilingKey、AIC core 路径均在 entry 直接返回且不写输出；正常 Host 必须保证只 launch 六个可达 key之一。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/lerp/docs/aclnnLerps&aclnnInplaceLerps.md` | `aclnnInplaceLerps*` | 950 支持、公开 shape/dtype/错误语义。 |
| 入口实现 | `math/lerp/op_api/aclnn_lerp_scalar.cpp` | `CheckParams/InferScalarTensorDtype/CalculateResult/aclnnInplaceLerps*` | 原地构图、950 scalar FP32、workspace/executor。 |
| Launcher | `math/lerp/op_api/lerp.cpp` | `l0op::Lerp` | 输出分配、AICORE launcher 添加。 |
| OpDef | `math/lerp/op_host/lerp_def.cpp` | `ops::Lerp` | 五组 dtype 配置、ascend950 注册、`lerp_apt`。 |
| Host/Tiling/POD | `math/lerp/op_host/arch35/lerp_tiling_arch35.h` | `LerpTilingData/LerpTiling` | POD 字段、编译信息、Tiling 生命周期。 |
| Host/Tiling | `math/lerp/op_host/arch35/lerp_tiling_arch35.cpp` | `GetOpKey/GetComputeMap/DoOpTiling/PostTiling` | opKey、UB divisor、key/block/workspace。 |
| Kernel entry/body | `math/lerp/op_kernel/lerp_apt.cpp` | `lerp` | AIV-only、十 key router；本入口可达六 key。 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f32_nddma_without_loops.h` | `LerpF32NddmaWithoutLoops` | R-001。 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_f32_nddma_with_loops.h` | `LerpF32NddmaWithLoops` | R-002。 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_dtype_comb_0_nddma_without_loops.h` | `LerpDtypeComb0NddmaWithoutLoops` | R-003。 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_dtype_comb_0_nddma_with_loops.h` | `LerpDtypeComb0NddmaWithLoops` | R-004。 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_dtype_comb_1_nddma_without_loops.h` | `LerpDtypeComb1NddmaWithoutLoops` | R-005。 |
| Kernel entry/body | `math/lerp/op_kernel/arch35/lerp_dtype_comb_1_nddma_with_loops.h` | `LerpDtypeComb1NddmaWithLoops` | R-006。 |
| 构建接线 | `math/lerp/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | 仅 ascend950 + arch35 活动接线。 |
| Host/Tiling helper | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling_base.h` | `BroadcastTilingParams/BroadcastTilingData` | 公共参数结构、key offset、UB 与 rank 常量。 |
| Host/Tiling helper | `src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp` | `BroadcastGetComputeKey/GetScheduleKey/GetMaxElemNum` | compute/schedule key与 tile UB 上界。 |
| Host/Tiling helper | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse/GetBlockSplitFactor/DoBrodcastTiling/BroadcastTiling` | 合轴、stride、分核、tail。 |
| Device helper | `pkg_inc/op_common/atvoss/util/broadcast_utils.h` | `BroadcastNddmaWithoutLoop/WithLoop` | NDDMA descriptor、外循环、offset、连续 fast path。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 广播公共切分的 9 个标量字段、4 组 dims、4 组 strides | 字段顺序、64-bit 类型、数组长 8、元素单位；宏布局不可猜测 | `LerpTilingData` |
| 调用顺序 | Contiguous→scalar FP32→Lerp→Cast→ViewCopy | ViewCopy 必须最后写回 `selfRef`；空 Tensor提前成功 | `CalculateResult` |
| workspace/内存 | Lerp 32 B；FP32/mixed UB 为 `32/20*elemNum` | depth=2 必须计入；executor 总 workspace不能用 32 B 替代 | `GetWorkspaceSize`、各 `Init` |
| 同步 | TQue EnQue/DeQue/FreeTensor；executor stream ordering | 广播缓存必须先初始化双槽；外循环填完后才 EnQue | `CopyIn*/Compute3/CopyOut4` |
| 精度与尾块 | FP32 两候选+阈值 Select；低精度 RINT；`blockTail/ubTail/mask` | 不得合并成未经证明的单表达式或改变 cast/round 顺序 | 各 `Compute3/Process` |
| 失败边界 | dtype/shape/UB/rank 在 API/Tiling 失败；未知 key无输出 | 不得把 A2/A3 文档支持外推成该固定源码的实现闭包 | `CheckParams/DoBrodcastTiling/lerp` |