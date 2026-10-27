---
type: Operator
title: "MoeFinalizeRoutingGrad A2/A3 ACLNN 算子知识"
description: "说明 aclnnMoeFinalizeRoutingV2Grad 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐 Route Vector 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/README.md
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/op_api/aclnn_moe_finalize_routing_v2_grad.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/op_api/aclnn_moe_finalize_routing_v2_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/op_api/moe_finalize_routing_v2_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_def.cpp
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_without_scale_not_cut_h.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_without_scale_cut_h.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_with_scale_not_cut_h.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_with_scale_cut_h.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/config/ascend910b/moe_finalize_routing_v2_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/config/ascend910_93/moe_finalize_routing_v2_grad_binary.json
    role: implementation
created_at: "2026-08-24T00:00:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# MoeFinalizeRoutingGrad A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize` / `aclnnMoeFinalizeRoutingV2Grad`；公开 V2 Grad 两段式 ACLNN，CANN 9.1.0 |
| 计算语义 | 以 `expandedRowIdx` 把 `gradY` 反散射为 `gradExpandedX`；存在 `scales` 时再计算 `gradScales` |
| 输入输出 | `gradY[R,H]`、`expandedRowIdx[R*K]`、四个可选输入，输出 ND `gradExpandedX` 与 `gradScales` |
| 执行类型 | 所有 A2/A3 Route 均为单节点 `AIV_VECTOR`；entry 指定 `KERNEL_TYPE_MIX_AIV_1_0`，AIV:AIC=`1:0` |
| Kernel 路线 | R-001/R-002：无 scale 的整 H/切 H；R-003～R-006：scale × bias 无/有 × 整 H/切 H |
| 核心资源 | GM、depth=1 `TQue`、标量状态与硬事件；无独立 `TBuf`、SIMT dynamic UB、NDDMA/通信 buffer；Kernel 不使用 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 launcher→Tiling/POD→单 AIV router→六条可达 body→A2/A3 config/CMake 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `R=batch*sequence`、`H=hidden`、`K=topK`，计算项 `i∈[0,R*K)`，token 行 `r=floor(i/K)`，散射行 `p=expandedRowIdx[i]`。无 `scales` 时 `K=1`，有效 `p` 上 `gradExpandedX[p,j]=gradY[r,j]`。存在 `scales` 时，`gradExpandedX[p,j]=gradY[r,j]*scales[i]`；无 bias 时 `gradScales[i]=Σ_j expandedX[p,j]*gradY[r,j]`，有 bias 时为 `Σ_j (expandedX[p,j]+bias[expertIdx[i],j])*gradY[r,j]`。Kernel 对 `p<0` 或 `p>=expandedXDim0` 不写 `gradExpandedX`；无 bias 路线把该项 `gradScales` 写 0，有 bias 路线仍计算 `Σ_j bias[expertIdx[i],j]*gradY[r,j]`。`dropPadMode=1` 时先把整个 `gradExpandedX` 有效域清零，再反散射。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradY` | `[R,H]` / ND | FP16/BF16/FP32 | 必选、2D；输出梯度来源 |
| `expandedRowIdx` | `[R*K]` / ND | INT32 | 必选、1D；无 scale 时 K=1；公开约束要求除 `-1` 外无重复索引 |
| `expandedXOptional` | dropless `[R*K,H]` 或 `[activeNum,H]`；drop `[E,C,H]` | 与 `gradY` 相同 | `scales` 存在时必选；参与 `gradScales` |
| `scalesOptional` | `[R,K]` / ND | FP16/BF16/FP32 | A2/A3 必须与 `gradY` 同 dtype；存在性选择有 scale Route |
| `expertIdxOptional` | `[R,K]` / ND | INT32 | bias 存在时必选，公开值域 `[0,E-1]` |
| `biasOptional` | `[E,H]` / ND | 与 `gradY` 相同 | 存在性选择 bias Route |
| `dropPadMode` | scalar | INT64 | 默认 0，只允许 0/1 |
| `activeNum` | scalar | INT64 | 默认 0；dropless 且 `0<activeNum<R*K` 时决定 X 输出行数 |
| `expertNum` | scalar | INT64 | 默认 0；drop 时必须大于 0，存在 bias 时等于 E |
| `expertCapacity` | scalar | INT64 | 默认 0；drop 时必须大于 0 |
| `gradExpandedXOut` | 与推导 X 域一致 / ND | 与 `gradY` 相同 | 必选；ACLNN 内部连续临时结果经 `ViewCopy` 提交 |
| `gradScalesOut` | `[R,K]` / ND | 与 `scales` 相同 | 必选；仅 `scales` 存在时有定义 |

### 入口与版本

固定 Commit 中同一反散射梯度语义仅公开 `MoeFinalizeRoutingV2Grad`，没有更高版本 ACLNN；因此它是 A2/A3 的实际最新公开入口。API 文档与 README 同时公开支持 A2、A3 和 Ascend 950；本文件是 Prompt 唯一指定的无后缀卡，按知识库规则只承载 A2/A3 的 Membase 实现。950 使用 `ascend950`/`moe_finalize_routing_v2_grad_apt` RegBase 路线，应进入独立 `_950` 卡，本次未生成。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段先用 `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`，再由 `CheckNotNull` 检查两个必选输入和两个输出。`CheckDtypeValid` 校验公开 dtype，并约束 `expandedX/bias/gradExpandedX` 与 `gradY` 同 dtype；A2/A3 的 scales 同 dtype约束在 Tiling `CheckOptionalInputDtype` 补齐。每个非空输入依次进入 `l0op::Contiguous`。L0 按用户输出 shape/dtype 分配两个连续 ND 临时 Tensor；若二者不同时为空，则把六输入、两输出、四属性加入 AICore launcher。最后两个 `ViewCopy` 将连续临时结果提交到用户输出。

Tiling 顺序是 `GetPlatformInfo→GetShapeAttrsInfo→DoOpTiling→GetWorkspaceSize→PostTiling`。它读取必选/可选 Tensor 与属性，检查维数、shape、dtype 和可选依赖；`CalcExpandedXDimInfo` 将 X 域归一为行数 `X=expandedXDim0`，再计算分核、Route 与 POD。任何空指针、shape/dtype、属性、平台资源或 launcher 失败均返回对应 ACLNN/`GRAPH_FAILED`，不进入 Device。

### 参数、Workspace 与生命周期

`TilingPrepare4MoeFinalizeRoutingV2Grad` 从平台产生 `aivNum=N` 与 `ubSize=U`；`CalcBaseInfo` 用它们产生初始化域和计算域的核切分。Tiling 固定声明 `workspace[0]=16*1024*1024` 字节，ACLNN 返回的是 executor 整图的 `GetWorkspaceSize()`，还可能包括 `Contiguous/ViewCopy` 的框架空间；固定仓未公开该合计公式。Kernel ABI 保留 `workspace` 指针，但六条 A2/A3 Route 均不解引用它。

`uniqueExecutor.ReleaseTo(executor)` 后 executor 与其临时 Tensor/launcher 图交给调用方；第二段 `CommonOpExecutorRun(workspace,workspaceSize,executor,stream)` 在用户 stream 执行。两个 `ViewCopy` 位于同一 executor 图的 AICore 节点之后，具体框架内部事件实现固定仓不可见，但图节点失败会使第一段构图或第二段执行返回失败。

### 启动图与失败传播

```text
aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize
  -> Contiguous(每个存在的输入)
  -> l0op::MoeFinalizeRoutingV2Grad
    -> ADD_TO_LAUNCHER_LIST_AICORE(MoeFinalizeRoutingV2Grad)
      -> Tiling4MoeFinalizeRoutingV2Grad
      -> moe_finalize_routing_v2_grad<<<blockDim, stream>>>
  -> ViewCopy(gradExpandedX临时量 -> 用户输出)
  -> ViewCopy(gradScales临时量 -> 用户输出)
aclnnMoeFinalizeRoutingV2Grad
  -> CommonOpExecutorRun
```

Device 启动图只有一个 Kernel 节点。其 `blockDim=max(initOutNeedCoreNum,computeNeedCoreNum)`，输入/输出直接为 launcher 管理的 GM Tensor，POD 为 `MoeFinalizeRoutingV2GradTilingData`，workspace slice 对 Device 为 0 B。`dropPadMode=1` 时 Host 设置 `scheduleMode=1`，以满足 Kernel 内 `SyncAll` 的全核同时启动前提。未知 key 在 entry 直接 `return`，不会写输出；launcher/Tiling 失败则不启动 Kernel，Device 失败由 executor/stream 运行边界传播。

## Route 与参数设计

### 分核与资源计算

设平台 AIV 数 `N`，输出初始化行数 `X=expandedXDim0`，计算项数 `M=expandedRowIdx.shape[0]`。对域 `D∈{X,M}`：`each=floor(D/N)`、`mod=D%N`、`need=(each==0?mod:N)`；核 `b<mod` 处理 `each+1` 项，其余活动核处理 `each` 项，超出 `need` 的核处理空区间。

设 `U=ubSize`、`s=sizeof(T1)`、`B=32`。无 scale 时 `P=floor((U-B)/B)*B/s`。有 scale 时每个 H 元素预算为 `4(gradY FP32)+4(expandedX FP32)+s(gradExpandedX)+(bias?4:0)`，固定先预留 row/scale/expert/gradScale 四个 32 B 块，再计算 `P=floor((U-4B)/(8+s+4*bias))*s/B*B/s`；最后一步使 `P*s` 为 32 B 整数倍。`H<=P` 选择整 H，否则 `hiddenInnerLoops=floor(H/P)`、`hiddenLastPart=H%P`。`P<=0` 直接 `GRAPH_FAILED`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 无 scale/整 H | 10001 / Membase | `!IsRegbaseSocVersion && scales==null && H<=P` | `H>P`→R-002 |
| 2 | R-002 无 scale/切 H | 10002 / Membase | `!IsRegbaseSocVersion && scales==null && H>P && P>0` | `P<=0`→失败 |
| 3 | R-003 scale/无 bias/整 H | 20001 / Membase | `!IsRegbaseSocVersion && scales!=null && bias==null && H<=P` | bias 存在→R-004；`H>P`→R-005 |
| 4 | R-004 scale/bias/整 H | 20001 / Membase | `!IsRegbaseSocVersion && scales!=null && bias!=null && H<=P` | `H>P`→R-006 |
| 5 | R-005 scale/无 bias/切 H | 20002 / Membase | `!IsRegbaseSocVersion && scales!=null && bias==null && H>P && P>0` | bias 存在→R-006 |
| 6 | R-006 scale/bias/切 H | 20002 / Membase | `!IsRegbaseSocVersion && scales!=null && bias!=null && H>P && P>0` | `P<=0`→失败 |

```text
Membase::IsCapable = !IsRegbaseSocVersion(context)
if scales == null:
    P = floor((U-32)/32)*32/s
    key = (H <= P ? 10001 : 10002)
else:
    bytesPerH = 8+s+(bias ? 4 : 0)
    P = floor((U-128)/bytesPerH)*s/32*32/s
    key = (H <= P ? 20001 : 20002)
if P <= 0: GRAPH_FAILED
entry 对 20001/20002 再以 bias GM 指针是否为空选择实际 body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 10001 | A2/A3；无 scale；`H<=P` | 单 AIV 节点 | `WithoutScaleNotCutH<T1,int32>` | `(P*s+32)B` UB；drop 时 `SyncAll` | 整行 DMA；无效 row 跳过 |
| R-002 | 10002 | A2/A3；无 scale；`H>P>0` | 单 AIV 节点 | `WithoutScaleCutH<T1,int32>` | 同 R-001，跨 H tile 复用 | `floor(H/P)` 满块加 `H%P` 尾块 |
| R-003 | 20001 + null bias | A2/A3；有 scale；无 bias；`H<=P` | 单 AIV 节点 | `WithScaleNotCutH<T1,int32>::SubProcessWithOutBias` | `P*(8+s)+96` B 实配；Host 保守留 128 B；硬事件 | 无效 row 的 gradScale=0 |
| R-004 | 20001 + nonnull bias | A2/A3；有 scale+bias；`H<=P` | 单 AIV 节点 | `WithScaleNotCutH<T1,int32>::SubProcessWithBias` | `P*(12+s)+128` B；硬事件 | 无效 row 仍归约 bias×gradY |
| R-005 | 20002 + null bias | A2/A3；有 scale；无 bias；`H>P` | 单 AIV 节点 | `WithScaleCutH<T1,int32>::SubProcessWithOutBias` | 同 R-003，`gradScalesSum_` 跨 tile 累加 | 末 tile=`H%P`，随后一次写 gradScale |
| R-006 | 20002 + nonnull bias | A2/A3；有 scale+bias；`H>P` | 单 AIV 节点 | `WithScaleCutH<T1,int32>::SubProcessWithBias` | 同 R-004，跨 tile 累加 | 同 R-005；无效 row 走 bias-only |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台族 | `!IsRegbaseSocVersion(context_)` | 本卡 entry/body 是 Membase AIV 实现 | 全部 | RegBase 由 950 模板处理，不进入本卡 | `MoeFinalizeRoutingV2GradMembase::IsCapable` |
| 平台资源 | `aivNum>0 && ubSize>0` | 产生 blockDim 与 P 上界 | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` / `TilingPrepare4...` |
| UB tile | `hiddenPrePart>0` | 每个 Route 至少容纳一个 H 元素及固定标量块 | 全部 | `GRAPH_FAILED` | `CalcTilingKey` |
| 32 B 对齐 | `P*s` 经公式下对齐到 32 B | `InitBuffer` 容量与 DMA block 对齐 | 全部 | 选择更小 P；P≤0 失败 | `CalcTilingKey*` |
| drop 同步 | `dropPadMode==1` 时 `SetScheduleMode(1)` | 清零阶段后所有核必须到达 `SyncAll` 才能 scatter | 全部 | 非 drop 不清零/不同步 | `DoOpTiling`、各 `Process` |
| row 有效域 | `0<=expandedRowIdx<expandedXDim0` | 防止 scatter 与 expandedX 读取越界 | 全部 | 不写 gradX；R-003/R-005 写 0 gradScale，bias Route 走 bias-only | `CopyOut` / `ComputeWith*Bias` |
| Reduce repeat | 64 元素/repeat，单次最多 255 repeat | `BlockReduceSum` repeat 字段上限 | R-003～R-006 | 拆为 `maxRptNum` 段、余 repeat 和余元素 | `ReduceCalc` |
| 空输出构图 | `!(grad_expanded_x->IsEmpty() && grad_scales->IsEmpty())` | 两输出同时为空时无需启动 AICore | 全部 | 跳过 launcher，仍返回临时 Tensor | `l0op::MoeFinalizeRoutingV2Grad` |

### POD/ABI

固定仓可见 14 个 `int64_t` 字段及序列化顺序，但 `TILING_DATA_FIELD_DEF` 的宏展开由仓外框架提供，因此 `sizeof/alignof/offsetof` 不能从固定仓逐字确认。以下 ABI 布局列保留声明顺序，不用自然对齐假设补齐缺失。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `MoeFinalizeRoutingV2GradTilingData` | `initOutNeedCoreNum` | int64/64 | 序号0；具体 offset/总大小固定仓不可见 | 核 | `CalcBaseInfo` | `GetInitOutBatches` | `[0,N]` |
| 同上 | `initOutEachCoreBatchNum` | int64/64 | 序号1；同上 | 行/核 | `X/N` | `GetInitOutBatches` | 非负 |
| 同上 | `initOutModCoreNum` | int64/64 | 序号2；同上 | 核 | `X%N` | `GetInitOutBatches` | `[0,N)` |
| 同上 | `computeNeedCoreNum` | int64/64 | 序号3；同上 | 核 | `CalcBaseInfo` | `GetSubProcessBatches` | `[0,N]` |
| 同上 | `computeEachCoreBatchNum` | int64/64 | 序号4；同上 | 项/核 | `M/N` | `GetSubProcessBatches` | 非负 |
| 同上 | `computeModCoreNum` | int64/64 | 序号5；同上 | 核 | `M%N` | `GetSubProcessBatches` | `[0,N)` |
| 同上 | `dropPadMode` | int64/64 | 序号6；同上 | 枚举 | `GetAttrInfo` | 各 `Process` | 0/1 |
| 同上 | `topK` | int64/64 | 序号7；同上 | 元素 | `scales.dim1`，无 scale 为1 | `batchIdx/topK` | scale Route 要求正数 |
| 同上 | `hidden` | int64/64 | 序号8；同上 | 元素 | `gradY.dim1` | 地址与 DMA count | 固定仓未显式检查 `H>0` |
| 同上 | `expandedXDim0` | int64/64 | 序号9；同上 | 行 | `CalcExpandedXDimInfo` | row guard/GM 地址 | dropless=`activeNum`或M；drop=`E*C` |
| 同上 | `hiddenPrePart` | int64/64 | 序号10；同上 | 元素 | `CalcTilingKey*` | `InitBuffer`/tile | 必须>0 |
| 同上 | `hiddenInnerLoops` | int64/64 | 序号11；同上 | 次 | `H/P` | cut-H loop | 整 H Route 为0 |
| 同上 | `hiddenLastPart` | int64/64 | 序号12；同上 | 元素 | `H%P` | cut-H tail | `[0,P)` |
| 同上 | `tilingKey` | int64/64 | 序号13；同上 | key | `CalcTilingKey*` | Device 通过框架 key 宏路由；字段仅日志/兼容 | 10001/10002/20001/20002 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10001 | Membase `CalcTilingKey` | 基础14字段 | `moe_finalize_routing_v2_grad` key10001 | `without_scale_not_cut_h.h` + base | host CMake + 910b/910_93 config |
| R-002 | 10002 | 同上 | 加 loop/tail | 同 entry key10002 | `without_scale_cut_h.h` + base | 同上 |
| R-003 | 20001/null bias | `CalcTilingKeyWithScales` | scale、P、H | entry key20001 后指针分支 | `with_scale_not_cut_h.h::SubProcessWithOutBias` + base | 同上 |
| R-004 | 20001/non-null bias | 同上 | 加 bias/expert 资源 | entry key20001 后指针分支 | `with_scale_not_cut_h.h::SubProcessWithBias` + base | 同上 |
| R-005 | 20002/null bias | 同上 | 加 loop/tail | entry key20002 后指针分支 | `with_scale_cut_h.h::SubProcessWithOutBias` + base | 同上 |
| R-006 | 20002/non-null bias | 同上 | 加 bias/expert/loop/tail | entry key20002 后指针分支 | `with_scale_cut_h.h::SubProcessWithBias` + base | 同上 |

## Kernel 实现

### 公共执行流程

单 entry 按 key 构造四个 AIV 类，有 scale 类在 `Init` 中以 bias GM 指针设置 `hasBias`，在 `SubProcess` 中选择实际 body。每个核先按 POD 求连续 item 区间；drop Route 执行分片 `InitOutput`，然后全核 `SyncAll`，再处理 item。无 scale Route只搬运 `gradY` 与 row index；有 scale Route以 FP32 中间值计算 gradX 与 gradScale，最后按 T1 降精度。节点间 workspace/GM 边不适用，因为启动图只有一个 Device 节点。

### Route R-001：无 scale、整 H 直接反散射

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKey` | `!scales && hidden<=hiddenPrePart` | key10001 |
| Router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | `moe_finalize_routing_v2_grad` | `TILING_KEY_IS(10001)` | `WithoutScaleNotCutH<T1,T2>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> WithoutScaleNotCutH::Init -> Base::SubInit
      -> Process -> [drop: Base::InitOutNotCutH -> SyncAll]
      -> SubProcess -> CopyIn -> CopyOut -> Base::Mte3ToMte2
```

##### 源码等价伪码

```text
core = GetBlockIdx(); derive [start,end) from compute* fields
if drop: zero assigned rows [initStart,initEnd) in gradExpandedX; SyncAll()
for i in [start,end):
  DMA gradY[i*H : i*H+H] and row=expandedRowIdx[i] into depth-1 queues
  if 0 <= row < X: DMA queue gradY -> gradExpandedX[row*H : row*H+H]; MTE3_MTE2 fence
  free both queue slots
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `gradY`→`gradExpandedX` | src=`i*H`；dst=`row*H` | `H*s` B | T1/ND | `CopyIn` | `CopyOut` | 当前 item | row guard 后写 |
| VECIN `gradYInQueue` | UB子区0 | `P*s` B | T1 | MTE2 | MTE3 | `Alloc→EnQue→DeQue→Free` | depth=1 |
| VECIN `expandedRowIdxInQueue` | gradY子区后 | 32 B | INT32 | MTE2 | scalar `GetValue(0)` | 当前 item | depth=1，仅4B有效 |
| workspace | N/A | Device 0 B | byte | Host ABI | 无 consumer | executor期 | 未解引用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  gy["gradYInQueue · depth=1 · P*s B"]:2
  row["rowIdxInQueue · depth=1 · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = P*s + 32
P = floor((U-32)/32)*32/s
UB_total <= U
dynamic_ub = N/A
workspace_total(Device) = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 两个 depth-1 Queue slot | `CopyIn` | `CopyOut` | 下一 item 的 `AllocTensor` | `DeQue/FreeTensor`；GM写后 `MTE3_MTE2` | 前一写已完成才复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| drop 清零 | `scheduleMode=1` + `SyncAll` | 各核 `InitOutput` | scatter | 整个 gradX GM | 原 dtype 写0；整 H |
| scatter | Queue + `HardEvent::MTE3_MTE2` | GM写 | 下一 item MTE2/Queue复用 | 输出与UB槽 | 不计算、不 cast；DMA有效长度 H |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKey` | P 与 key10001 |
| Kernel entry/router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | entry key10001 | AIV-only模板实例 |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_without_scale_not_cut_h.h` | `Process/CopyIn/CopyOut` | 整行搬运与row guard |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_base.h` | `GetSubProcessBatches/InitOutNotCutH` | 分核、清零、事件 |

### Route R-002：无 scale、切 H 反散射

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKey` | `!scales && hidden>hiddenPrePart>0` | key10002 |
| Router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | entry | key10002 | `WithoutScaleCutH<T1,T2>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> WithoutScaleCutH::Init -> Base::SubInit
      -> Process -> [drop: Base::InitOutCutH -> SyncAll]
      -> SubProcess -> for H tile: CopyIn -> CopyOut -> Mte3ToMte2
```

##### 源码等价伪码

```text
for i in assigned items:
  for t in [0,hiddenInnerLoops): copyNum=P; copy gradY[i*H+t*P]; read row; guarded write gradX[row*H+t*P]
  if hiddenLastPart!=0: repeat with t=hiddenInnerLoops, copyNum=hiddenLastPart
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM tile | src=`i*H+t*P`; dst=`row*H+t*P` | `copyNum*s` B | T1/ND | `CopyIn` | `CopyOut` | 当前 tile | 尾块用 `H%P` |
| VECIN queues | gradY子区+row子区 | `P*s + 32` B | T1+INT32 | MTE2 | MTE3/scalar | 当前 tile | depth=1，逐 tile复用 |
| workspace | N/A | Device 0 B | byte | Host ABI | 无 | executor期 | 未解引用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  gy["gradYInQueue · depth=1 · P*s B"]:2
  row["rowIdxInQueue · depth=1 · 32 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = P*s + 32 <= U
dynamic_ub = N/A
workspace_total(Device) = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| gradY/row Queue | 当前 tile `CopyIn` | 当前 tile `CopyOut` | 下一 H tile | `FreeTensor` + `MTE3_MTE2` | `copyNum<=P` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| drop 清零 | `InitOutCutH` + `SyncAll` | 分片清零 | scatter | gradX GM | 满 tile+可选尾 tile |
| tile写回 | Queue + `MTE3_MTE2` | 当前tile GM写 | 下一tile复用 | depth-1槽 | `hiddenLastPart==0` 时无额外尾循环 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKey` | loop/tail/key10002 |
| Kernel entry/router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | entry key10002 | 切H模板实例 |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_without_scale_cut_h.h` | `SubProcess/CopyIn/CopyOut` | tile地址与尾块 |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_base.h` | `InitOutCutH` | 分tile清零 |

### Route R-003：有 scale、无 bias、整 H

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKeyWithScales` | scales存在、bias不存在、`H<=P` | key20001 |
| Router/body | | entry + `WithScaleNotCutH::SubProcess` | `bias==nullptr` / `hasBias=false` | `SubProcessWithOutBias` |

#### Device 数据流

##### 函数级调用链

```text
entry -> WithScaleNotCutH::Init -> Base::SubInit
      -> Process -> [drop: InitOutNotCutH -> SyncAll]
      -> SubProcessWithOutBias
         -> Base::CopyInWithOutBias -> ComputeWithOutBias
            -> ComputeGradExpandedX -> CopyOutGradExpandedX
            -> DataCopyPad(expandedX) -> WaitMte2ToVec -> Mul -> ReduceCalc
         -> CopyOutGradScales
```

##### 源码等价伪码

```text
for i in assigned items: r=i/topK; load gradY[r,:], row[i], scale[i]
  if row valid:
    gradX[row,:] = cast_T1(cast_fp32(gradY)*scale)
    load expandedX[row,:]; product_fp32=expandedX*gradY; gradScalesSum += Reduce(product_fp32)
  write gradScales[i]=cast_T1(gradScalesSum); reset sum=0
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | gradY=`(i/K)*H`; X/gradX=`row*H`; scalar=`i` | H*s + scalar | ND | MTE2/Vector | Vector/MTE3 | 当前item | row guard |
| VECIN gradY | UB | `P*4` B | FP32工作区；T1原值 | `CopyIn`/Cast | Muls/Mul | 当前item | FP16/BF16原值放后半区 |
| VECIN expandedX | UB | `P*4` B | FP32临时/expandedX | Muls后被GM加载覆盖 | Reduce | 当前item | 生命周期复用 |
| row+scale queues | UB | 32+32 B | INT32+T1 | MTE2 | scalar | CopyIn内 | depth=1，读取后即Free |
| VECOUT gradX+gradScale | UB | `P*s+32` B | T1 | Vector | MTE3 | 当前item | depth=1 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gy["gradYIn · P*4 B"]:1
  x["expandedXIn/temp · P*4 B"]:1
  gx["gradXOut · P*s B"]:1
  scalar["row+scale+gradScale · 96 B"]:1
```

##### 字节账与复用顺序

```text
UB_actual = P*(8+s) + 96
Host selector reserves = P*(8+s) + 128 <= U
dynamic_ub = N/A; workspace_total(Device)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| expandedXIn Queue | `ComputeGradExpandedX` 的 Muls临时值 | gradX cast/写出 | `DataCopyPad(expandedX)` | `CopyOutGradExpandedX` 后再加载；`WaitMte2ToVec` | gradX先消费临时值 |
| gradY Queue前/后半 | GM原T1写后半；Cast写前半 | FP32计算 | 下一item | Queue生命周期 | 仅FP16/BF16启用 offset=P |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| X加载 | `MTE2_V`硬事件 | DataCopyPad | Cast/Mul | expandedXIn | 整H |
| gradX写回 | Queue+`MTE3_MTE2` | Cast/Muls | 后续复用 | gradXOut/GM | FP16/BF16先升FP32，`CAST_RINT`降精度 |
| gradScale | `BlockReduceSum`+`ReduceSum`，`V_MTE3` | FP32 product | 标量写GM | gradScale槽 | FP32累加，最终T1 `CAST_RINT` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKeyWithScales` | 无bias P公式/key20001 |
| Kernel entry/router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | entry key20001 | WithScaleNotCutH |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_with_scale_not_cut_h.h` | `SubProcessWithOutBias` | 整H调用顺序 |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_base.h` | `CopyInWithOutBias/ComputeWithOutBias/ReduceCalc` | 地址、复用、精度、归约 |

### Route R-004：有 scale、有 bias、整 H

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `tiling_membase.cpp` | `CalcTilingKeyWithScales` | scales+bias存在、`H<=P`，bytesPerH=`12+s` | key20001 |
| Router/body | | entry + `WithScaleNotCutH::SubProcess` | `bias!=nullptr`→`hasBias=true` | `SubProcessWithBias` |

#### Device 数据流

##### 函数级调用链

```text
entry -> WithScaleNotCutH::Init -> Process -> SubProcessWithBias
      -> CopyInWithBias -> ComputeWithBias
         -> [valid row: ComputeGradExpandedX -> CopyOut; load expandedX]
         -> Add(bias,expandedX) or bias-only -> Mul(gradY) -> ReduceCalc
      -> CopyOutGradScales
```

##### 源码等价伪码

```text
load gradY[r,:], row[i], scale[i], expert[i], bias[expert[i],:]
if row valid: write scaled gradX; load X[row,:]; reduce (X+bias)*gradY
else: do not write gradX; reduce bias*gradY
write one gradScale and reset accumulator
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM bias | `expertIdx[i]*H` | `H*s` B | T1/ND | MTE2 | Add/Mul | 当前item | expertIdx须满足公开值域 |
| gradY/expandedX/bias Queue | 各独立UB子区 | `P*4` B 各 | FP32工作区 | MTE2/Cast | Vector | 当前item | FP16/BF16后半存原值 |
| gradX Queue | UB | `P*s` B | T1 | Vector | MTE3 | 当前item | depth=1 |
| row/scale/expert/gradScale | UB | 4×32 B | 混合 | MTE2/Vector | scalar/MTE3 | 当前item | depth=1 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gy["gradYIn · P*4 B"]:1
  x["expandedXIn/temp · P*4 B"]:1
  bias["biasIn · P*4 B"]:1
  gx["gradXOut · P*s B"]:1
  scalar["row+scale+expert+gradScale · 128 B"]:4
```

##### 字节账与复用顺序

```text
UB_total = P*(12+s) + 128 <= U
dynamic_ub=N/A; workspace_total(Device)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| expandedXIn临时区 | gradX Muls temp | gradX降精度/写出 | valid row的X加载 | gradX先DeQue并MTE3提交 | 仅valid row覆盖 |
| biasIn | bias GM/Cast | Add或bias-only Mul | 下一item | `DeQue/FreeTensor` | invalid row也必须保留bias数据 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→Vector | Queue与`MTE2_V` | gradY/bias/X DMA | Cast/Add/Mul | 三个输入子区 | 整H |
| Vector→GM | `V_MTE3`、`MTE3_MTE2` | gradX/gradScale | GM提交/下一item | 输出槽 | FP32 Add/Mul/Reduce，T1 `CAST_RINT` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKeyWithScales` | bias计入P预算 |
| Kernel entry/router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | entry key20001 | 指针分支 |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_with_scale_not_cut_h.h` | `SubProcessWithBias` | 整H body |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_base.h` | `CopyInWithBias/ComputeWithBias` | bias地址、valid/invalid分支 |

### Route R-005：有 scale、无 bias、切 H

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `tiling_membase.cpp` | `CalcTilingKeyWithScales` | scale存在、bias不存在、`H>P>0` | key20002 |
| Router/body | | entry + `WithScaleCutH::SubProcess` | `hasBias=false` | `SubProcessWithOutBias` |

#### Device 数据流

##### 函数级调用链

```text
entry -> WithScaleCutH::Init -> Process -> SubProcessWithOutBias
      -> for each H tile: CopyInWithOutBias -> ComputeWithOutBias -> ReduceCalc
      -> after all tiles: CopyOutGradScales
```

##### 源码等价伪码

```text
for item i: gradScalesSum=0
  for each full tile and optional tail: compute/write scaled gradX tile; if row valid load X tile and accumulate FP32 dot
  write gradScales[i] once; reset gradScalesSum
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM H tile | base同R-003，附加 `innerOffset=t*P` | `copyNum*s` B | T1/ND | MTE2/Vector | Vector/MTE3 | 当前tile | tail=`H%P` |
| UB queues | 同R-003 | `P*(8+s)+96` B | 混合 | CopyIn/Vector | Vector/CopyOut | 当前tile | 每tile复用 |
| scalar accumulator | 成员 `gradScalesSum_` | FP32 scalar | float | `ReduceCalc` | `CopyOutGradScales` | 跨当前item全部tile | 写出后清0 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gy["gradYIn · P*4 B"]:1
  x["expandedXIn/temp · P*4 B"]:1
  gx["gradXOut · P*s B"]:1
  scalar["row+scale+gradScale · 96 B"]:1
```

##### 字节账与复用顺序

```text
UB_actual=P*(8+s)+96; Host保守预算=P*(8+s)+128<=U
workspace_total(Device)=0; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 全部depth-1 Queue | 当前tile | 当前tile计算/写回 | 下一tile | Queue Free + 硬事件 | `copyNum<=P` |
| `gradScalesSum_` | 每tile `ReduceCalc` | item末 `CopyOutGradScales` | 下一item置0 | 顺序标量依赖 | 尾tile完成后才写出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile管线 | Queue+MTE2_V+MTE3_MTE2 | 当前tile DMA/Vector | 下一tile | depth-1槽 | 每tile FP32计算 |
| 跨tile归约 | C++ FP32标量累加 | 每tile `ReduceCalc` | item末写出 | `gradScalesSum_` | 满tile加可选 `H%P`；最终一次T1降精度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKey` | key20002、loop/tail |
| Kernel entry/router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | entry key20002 | WithScaleCutH |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_with_scale_cut_h.h` | `SubProcessWithOutBias` | tile循环/末次写gradScale |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_base.h` | `ReduceCalc/CopyOutGradScales` | 跨tile累加与清0 |

### Route R-006：有 scale、有 bias、切 H

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `tiling_membase.cpp` | `CalcTilingKeyWithScales` | scale+bias存在、`H>P>0` | key20002 |
| Router/body | | entry + `WithScaleCutH::SubProcess` | `hasBias=true` | `SubProcessWithBias` |

#### Device 数据流

##### 函数级调用链

```text
entry -> WithScaleCutH::Init -> Process -> SubProcessWithBias
      -> for H tile: CopyInWithBias -> ComputeWithBias -> ReduceCalc
      -> CopyOutGradScales(item)
```

##### 源码等价伪码

```text
for item i: load row/scale/expert per tile; load gradY and bias tile
  valid row: write scaled gradX tile; load X tile; accumulate (X+bias)*gradY
  invalid row: accumulate bias*gradY only
after full tiles and tail: write gradScale once and reset accumulator
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM gradY/X/bias/gradX | 行base + `innerOffset` | `copyNum*s` B | T1/ND | MTE2/Vector | Vector/MTE3 | 当前tile | row guard仅保护X/gradX |
| UB vector queues | gradY+X+bias+gradX | `P*(12+s)` B | FP32/T1 | MTE2/Vector | Vector/MTE3 | 当前tile | depth=1 |
| UB scalar queues | row+scale+expert+gradScale | 128 B | 混合 | MTE2/Vector | scalar/MTE3 | 当前tile/item | depth=1 |
| FP32 accumulator | 寄存/标量状态 | 固定仓不公开寄存器字节 | float | `ReduceCalc` | item末写回 | 跨tile | 写回后置0 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gy["gradYIn · P*4 B"]:1
  x["expandedXIn/temp · P*4 B"]:1
  bias["biasIn · P*4 B"]:1
  gx["gradXOut · P*s B"]:1
  scalar["row+scale+expert+gradScale · 128 B"]:4
```

##### 字节账与复用顺序

```text
UB_total=P*(12+s)+128<=U
workspace_total(Device)=0; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Vector Queue子区 | 当前tile DMA/Vector | 当前tile Add/Mul/Reduce/MTE3 | 下一tile | Queue释放与硬事件 | full/tail都不超过P |
| FP32 accumulator | 每tile归约 | item末gradScale输出 | 下一item清0 | 程序顺序 | invalid row仍累加bias-only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| drop初始化 | `InitOutCutH`+`SyncAll` | 各核清零 | scatter | gradX GM | 分tile清零 |
| tile计算 | Queue/硬事件 | DMA/Vector | 下一tile | 所有depth-1槽 | FP16/BF16升FP32，Add/Mul/Reduce后最终`CAST_RINT`；tail=`H%P` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `CalcTilingKeyWithScales` | bias P预算/key20002 |
| Kernel entry/router | `op_kernel/moe_finalize_routing_v2_grad.cpp` | entry key20002 | 指针分支 |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_with_scale_cut_h.h` | `SubProcessWithBias` | full/tail tile顺序 |
| Device body/helper | `op_kernel/moe_finalize_routing_v2_grad_base.h` | `CopyInWithBias/ComputeWithBias/ReduceCalc` | bias-only fallback、精度与事件 |

## 算子特定优化

### Trick T-001：按 UB 容量选择整 H/切 H Route

#### 触发与源码依据

- 适用 Route：R-001、R-002；R-003～R-006 使用同一整 H/切 H 选择框架，但采用含 scale/bias 的独立字节公式
- 触发条件：Membase 且无 scale；以平台 `ubSize` 扣除 row Queue 后计算 `hiddenPrePart`，再比较 `H<=P`
- Golden 来源：[`moe_finalize_routing_v2_grad_tiling_membase.cpp:54-64`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp#L54-L64)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (isScalesExist_) {
    CalcTilingKeyWithScales();
} else {
    int64_t expandedRowIdxUbSize = BYTE_BLOCK;
    hiddenPrePart_ = (aicoreParams_.ubSize - expandedRowIdxUbSize) / BYTE_BLOCK * BYTE_BLOCK / gradYTypeByteSize_;
    if (hidden_ <= hiddenPrePart_) {
        tilingKey_ = TILING_KEY_WITHOUT_SCALE_NOT_CUT_H;
    } else {
        tilingKey_ = TILING_KEY_WITHOUT_SCALE_CUT_H;
    }
}
```

#### 机制与设计原因

无 scale 路线只为 gradY 容量加一个 32 B row Queue，先把 P 下对齐到 dtype 对应的 32 B 元素倍数；`H<=P` 走单次整行 DMA，否则以同一物理 Queue 循环处理 H tile。这样小 H 不引入 tile 循环，大 H 又不会超过 `InitBuffer(P*s)`。有 scale Route 在同一 selector 文件中采用更严格的 FP32工作区与可选bias预算，Route矩阵中的实际字节账不能与这里的无scale公式混用。

#### 迁移边界

- 适用：存在整轴/切轴实现，Host 能精确获得 UB 与固定标量 Queue 容量的 AIV 算子。
- 前提：UB 字节数、dtype 字节数和 32 B 对齐规则可由 Host 确定。
- 失效：`P<=0` 必须失败；不能让 Device 自行访问超过分配容量的 H。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓给出的真实失败终点是 `hiddenPrePart_<=0 -> GRAPH_FAILED`；不得把它改成 P=0 的切 H 循环，否则会产生除零或零步长。

### Trick T-002：FP16/BF16 原值放 Queue 后半区，前半区原地承载 FP32 结果

#### 触发与源码依据

- 适用 Route：R-003～R-006
- 触发条件：`T1` 为 `half` 或 `bfloat16_t`，`orgF16OBf16UbOffset_=P`
- Golden 来源：[`moe_finalize_routing_v2_grad_base.h:272-290`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_base.h#L272-L290)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::ComputeGradExpandedX(
    LocalTensor<T1>& gradYUb, LocalTensor<T1>& tempUb, LocalTensor<T1>& gradExpandedXUb, int64_t innerLoopCopyNum)
{
    auto gradYUbF32 = gradYUb.template ReinterpretCast<float>();
    auto tempUbF32 = tempUb.template ReinterpretCast<float>();

    if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
        Cast(gradYUbF32, gradYUb[orgF16OBf16UbOffset_], RoundMode::CAST_NONE, innerLoopCopyNum);
        if constexpr (IsSameType<T1, half>::value) {
            Muls(tempUbF32, gradYUbF32, (float)scale_, innerLoopCopyNum);
        } else {
            Muls(tempUbF32, gradYUbF32, ToFloat(scale_), innerLoopCopyNum);
        }
        Cast(gradExpandedXUb, tempUbF32, RoundMode::CAST_RINT, innerLoopCopyNum);
    } else {
        Muls(gradExpandedXUb, gradYUb, (float)scale_, innerLoopCopyNum);
    }

    gradExpandedXOutQueue_.template EnQue(gradExpandedXUb);
}
```

#### 机制与设计原因

gradY、expandedX、bias Queue 都按 `P*sizeof(float)` 分配。对 16 bit 输入，GM 数据写入以 T1 元素计的偏移 P，即物理后半区；随后 `ReinterpretCast<float>` 把 FP32结果写入前半区。这样同一 Queue 同时完成“原值暂存→升精度工作区”的生命周期复用，不再另分 FP32 TBuf。`expandedXInQueue` 还先作为 gradX 的 Muls 临时区，gradX写出后才由 expandedX GM 数据覆盖。

#### 迁移边界

- 适用：16 bit 输入需要 FP32 Vector 计算，且输入消费顺序允许前后半区原地转换。
- 前提：分配容量为 P×4 B、源偏移为P个T1元素、Cast重叠语义与消费顺序保持一致。
- 失效：FP32 输入直接走原 dtype分支；若原值仍被后续步骤需要，则不能提前覆盖。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供独立错误实现；从当前地址协议可验证，若仍从 Queue 起始处装入 16 bit 原值再向同一位置写 FP32，会破坏尚未消费的源区。

### Trick T-003：分段 BlockReduceSum 规避 255-repeat 上限并跨 H tile 只写一次标量

#### 触发与源码依据

- 适用 Route：R-003～R-006；切 H 时收益机制跨 tile 延续
- 触发条件：`innerLoopCopyNum` 任意正长度，按64元素/repeat；repeat超过255时分段
- Golden 来源：[`moe_finalize_routing_v2_grad_base.h:293-325`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_base.h#L293-L325)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T1, typename T2>
__aicore__ inline void MoeFinalizeRoutingV2GradBase<T1, T2>::ReduceCalc(
    LocalTensor<float>& inputUb, LocalTensor<float>& tempUb, int64_t innerLoopCopyNum)
{
    int32_t repeat = innerLoopCopyNum / MASK_FLOAT_ONE_REPEAT;
    uint32_t modNum = innerLoopCopyNum % MASK_FLOAT_ONE_REPEAT;
    uint32_t maxRptNum = repeat / MAX_REPEAT;
    uint32_t modRptNum = repeat % MAX_REPEAT;
    uint32_t offset = 0;

    if (maxRptNum != 0) {
        for (uint32_t rptIdx = 0; rptIdx < maxRptNum; rptIdx++) {
            offset = rptIdx * MAX_REPEAT * MASK_FLOAT_ONE_REPEAT;
            BlockReduceSum<float>(
                tempUb[rptIdx * MAX_REPEAT * BLOCK_NUM_ONE_RPT], inputUb[offset], MAX_REPEAT, MASK_FLOAT_ONE_REPEAT, 1,
                1, STRIDE);
        }
    }

    if (modRptNum != 0) {
        offset = maxRptNum * MAX_REPEAT * MASK_FLOAT_ONE_REPEAT;
        BlockReduceSum<float>(
            tempUb[maxRptNum * MAX_REPEAT * BLOCK_NUM_ONE_RPT], inputUb[offset], modRptNum, MASK_FLOAT_ONE_REPEAT, 1, 1,
            STRIDE);
    }

    if (modNum != 0) {
        offset = repeat * MASK_FLOAT_ONE_REPEAT;
        BlockReduceSum<float>(tempUb[repeat * BLOCK_NUM_ONE_RPT], inputUb[offset], 1, modNum, 1, 1, STRIDE);
    }

    ReduceSum(inputUb, tempUb, inputUb, (innerLoopCopyNum + NUM_FLOAT_ONE_BLOCK - 1) / NUM_FLOAT_ONE_BLOCK);
    gradScalesSum_ += inputUb.GetValue(0);
}
```

#### 机制与设计原因

每个 `BlockReduceSum` 处理 64 FP32元素/repeat，满段固定255 repeats；之后处理余 repeats 与不足64元素的尾段，再用 `ReduceSum` 收口，结果累加到 `gradScalesSum_`。切 H Route让该 FP32标量跨 tile 存活，等一整个 item 的 full/tail tile 完成后才向 GM 写一次，避免每 tile 物化 partial gradScale 或申请 workspace。

#### 迁移边界

- 适用：长向量点积/归约，单条 Vector repeat字段有限且最终只需一个标量。
- 前提：temp Queue容量能容纳分块结果；跨 tile累计顺序满足精度要求。
- 失效：跨核共同归约不能直接复用当前核内标量协议，需要额外原子或中转与同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码中真实边界是 `MAX_REPEAT=255`；把任意长向量直接塞入一次 `BlockReduceSum` 会超出 repeat字段表达范围。

## 模板与实现对比

| 对比维度 | R-001 | R-002 | R-003 | R-004 | R-005 | R-006 | 设计选择依据 |
|---|---|---|---|---|---|---|---|
| 命中条件/优先级 | 无scale整H | 无scale切H | scale无bias整H | scale+bias整H | scale无bias切H | scale+bias切H | 先看scale，再由H/P定key，body内看bias |
| 模板与 dtype | 搬运T1 | 搬运T1 | T1↔FP32 | T1↔FP32+bias | 同R-003 | 同R-004 | 有scale才需要乘法/归约 |
| 循环与分核 | item循环 | item×H tile | item循环 | item循环 | item×H tile | item×H tile | M/X按AIV均分，H按P切分 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2 Queue；无TBuf | 同左 | 5 Queue族+FP32标量 | 8 Queue+FP32标量 | 同R-003 | 同R-004 | 所有Route均无dynamic UB/通信window |
| 内存总占用与复用 | `P*s+32` | 同左逐tile复用 | `P*(8+s)+96`实配 | `P*(12+s)+128` | 同R-003 | 同R-004 | 可选bias决定每元素4B与expert 32B |
| 搬运/Vector API | DataCopyPad | 分tileDataCopyPad | Cast/Muls/Mul/Reduce | 加Add与bias DMA | 同R-003跨tile | 同R-004跨tile | 仅有效row读取X/写gradX |
| 同步、精度与 tail | drop SyncAll；原dtype | 加H尾块 | FP32累加/T1写回 | 同左，invalid bias-only | 跨tile FP32累加+尾块 | 同左 | 硬事件保护depth-1复用 |
| fallback/失败边界 | invalid row跳过 | 同左 | invalid写0 gradScale | invalid仍算bias gradScale | 同R-003 | 同R-004 | 未知key返回；P≤0失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | README/API公开支持；本卡覆盖 | README/API公开支持；本卡覆盖 | README/API公开支持；本次目标未包含 `_950` 卡 |
| Host/Launcher | `ascend910b` Membase config | `ascend910_93` Membase config | `ascend950` RegBase selector、不同dtype组合与 `_apt` entry |
| Kernel/资源 | key10001/10002/20001/20002，AIV-only | 与A2共用同一Membase body | key扩展到20011/20021/20012/20022；不应混入本卡 |

## 限制与失败边界

- A2/A3 的 `scales`、`expandedX`、`bias`、`gradExpandedX` 与 `gradY` 必须同 dtype；950允许 scales dtype独立，但不属于本卡。
- Tiling检查shape/dtype与可选依赖，不扫描 `expandedRowIdx`/`expertIdx` 的GM值；索引范围与去重是调用方公开前提。Device只对 row 做边界保护，bias Route会直接用 `expertIdx` 寻址。
- `hiddenPrePart<=0`、AIV数/UB不可用、shape/dtype/属性不符均 `GRAPH_FAILED`；未知 key 的 Kernel直接返回且不写输出。
- `dropPadMode=1` 依赖 `SetScheduleMode(1)+SyncAll` 完成清零→scatter顺序；dropless 不预清零，正确性依赖公开的索引覆盖/不重复约束。
- `H>0` 未在固定 Tiling 中被显式检查；空 Tensor 情况仅 L0 在两个临时输出都为空时跳过 launcher。不要把未覆盖的零维行为推断为已支持。
- Tiling声明16 MiB算子workspace，但 A2/A3 Kernel不解引用；ACLNN executor总workspace及框架内部stream/event细节固定仓不可见。
- POD字段声明顺序可见，宏展开后的 `sizeof/alignof/offsetof` 固定仓不可见；不得把自然布局推测当作ABI事实。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口约束 | `moe/moe_finalize_routing_v2_grad/README.md` | 产品支持表/公式 | A2/A3/950公开支持与逻辑语义 |
| 入口约束 | `moe/moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad.md` | 两段式原型/参数表 | 公开API、shape/dtype/错误语义 |
| 入口声明 | `moe/moe_finalize_routing_v2_grad/op_host/op_api/aclnn_moe_finalize_routing_v2_grad.h` | 两个ACLNN符号 | C ABI参数顺序 |
| 入口实现 | `moe/moe_finalize_routing_v2_grad/op_host/op_api/aclnn_moe_finalize_routing_v2_grad.cpp` | `CheckParams`/两段式函数 | Contiguous、L0、ViewCopy、executor |
| L0 Launcher | `moe/moe_finalize_routing_v2_grad/op_host/op_api/moe_finalize_routing_v2_grad.cpp` | `l0op::MoeFinalizeRoutingV2Grad` | 临时输出、空输出guard、AICore节点 |
| 平台注册 | `moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_def.cpp` | `MoeFinalizeRoutingV2Grad` OpDef | 910b/910_93 Membase与950 RegBase |
| shape推导 | `moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_infershape.cpp` | `MoeFinalizeRoutingV2GradInferShape` | 两输出shape/dtype |
| POD | `moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_tiling.h` | `MoeFinalizeRoutingV2GradTilingData` | 字段顺序与Tiling类 |
| Host/Tiling | `moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_tiling.cpp` | `CheckParams/CalcBaseInfo/PostTiling` | 校验、分核、workspace、blockDim |
| Host selector | `moe/moe_finalize_routing_v2_grad/op_host/moe_finalize_routing_v2_grad_tiling_membase.cpp` | `MoeFinalizeRoutingV2GradMembase` | A2/A3 key与UB公式 |
| Kernel router | `moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad.cpp` | `moe_finalize_routing_v2_grad` | AIV task type、四key映射 |
| Kernel公共体 | `moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_base.h` | `MoeFinalizeRoutingV2GradBase` | 分核、搬运、Vector、归约、事件 |
| Kernel body | `moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_without_scale_not_cut_h.h` | `WithoutScaleNotCutH` | R-001 |
| Kernel body | `moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_without_scale_cut_h.h` | `WithoutScaleCutH` | R-002 |
| Kernel body | `moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_with_scale_not_cut_h.h` | `WithScaleNotCutH` | R-003/R-004 |
| Kernel body | `moe/moe_finalize_routing_v2_grad/op_kernel/moe_finalize_routing_v2_grad_with_scale_cut_h.h` | `WithScaleCutH` | R-005/R-006 |
| 构建接线 | `moe/moe_finalize_routing_v2_grad/op_host/CMakeLists.txt` | `add_modules_sources` | Host模块与950特定编译选项 |
| binary config | `moe/moe_finalize_routing_v2_grad/op_host/config/ascend910b/moe_finalize_routing_v2_grad_binary.json` | `op_list` | A2三种同dtype二进制实例 |
| binary config | `moe/moe_finalize_routing_v2_grad/op_host/config/ascend910_93/moe_finalize_routing_v2_grad_binary.json` | `op_list` | A3三种同dtype二进制实例 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 14字段生产/消费关系 | 字段声明顺序、int64位宽、元素/核单位 | `MoeFinalizeRoutingV2GradTilingData` / `PostTiling` |
| 调用顺序 | Contiguous→AICore→ViewCopy；drop清零→SyncAll→scatter | 不得把Host书写顺序替代框架图或删去全核同步 | 两段式ACLNN、各`Process` |
| workspace/内存 | Kernel workspace为0消费；UB按Route公式 | depth=1仍计完整容量；FP16/BF16前后半区重叠协议 | `GetWorkspaceSize`、各`InitBuffer` |
| 同步 | Queue与四类硬事件可复用 | MTE2/V/MTE3依赖和drop `scheduleMode=1` | `Mte3ToMte2/WaitMte2ToVec/MTE3ToVSync/VToMTE3Sync` |
| 精度与尾块 | FP16/BF16升FP32，最终`CAST_RINT`；切H仅末尾一次写gradScale | 不得逐tile提前降精度或漏掉 `H%P` | `ComputeGradExpandedX/ReduceCalc/CopyOutGradScales` |
| 失败边界 | row guard、P/平台资源失败、unknown key return | expertIdx值域仍是调用方前提；950不能套用Membase卡 | `CheckParams/CalcTilingKey/entry` |
