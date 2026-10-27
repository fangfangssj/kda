---
type: Operator
title: "CircularPad3d A2/A3 ACLNN 算子知识"
description: "说明 aclnnCircularPad3dGetWorkspaceSize/aclnnCircularPad3d 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/docs/aclnnCircularPad3d.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/op_api/aclnn_circular_pad3d.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/op_api/aclnn_circular_pad3d.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3/op_api/padv3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3/op_host/pad_v3_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad_2d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad_3d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/config/ascend910b/circular_pad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/config/ascend910_93/circular_pad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-24T23:51:11Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# CircularPad3d A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCircularPad3dGetWorkspaceSize` / `aclnnCircularPad3d`，CANN 9.1.0 公开 ACLNN 两段式接口。 |
| 计算语义 | 对最后三维 `(D,H,W)` 做循环填充；正 padding 从维度另一端循环复制，负 padding 先裁去对应端。 |
| 输入输出 | `self/out` 为 4D 或 5D、ND 语义，dtype 为 INT8/FP16/BF16/FP32/INT32；`padding` 为 6 个 INT64，顺序 `(left,right,top,bottom,front,back)`。 |
| 执行类型 | 核心单 Kernel `AIV_VECTOR` 数据搬运实现；只用 AIV 核、`TQueBind`、`DataCopyPad` 与 MTE 事件，无 Cube/AIC、MIX_CV、SIMT 或数值 Vector 算术节点。 |
| Kernel 路线 | R-001/R-002：无深度 padding 的 small-shape，分别有/无左右正 padding；R-003/R-004：无深度 padding 的 big-shape；R-005/R-006：有深度 padding 的 small-shape；R-007/R-008：有深度 padding 的 big-shape。 |
| 核心资源 | 每核固定 192 KiB UB：双缓冲 `TQueBind<VECIN,VECOUT,1>`，每槽 96 KiB；user workspace 保存完整中间行或左右窄条，另有 runtime system-workspace 前缀。 |
| 来源闭包 | `source_closed`：公开 ACLNN→Contiguous/PadV3/ViewCopy→CircularPad launcher→A2/A3 Tiling/POD→20 个可达 key→全部 2D/3D DMA body/helper→binary/CMake 接线已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设输入最后三维为 `(D,H,W)`，padding 为 `(l,r,t,b,f,k)`，输出后三维为 `(D+f+k,H+t+b,W+l+r)`。对每个输出坐标，先减去对应前侧 padding，再按输入维长做循环映射；负 padding 等价于先裁去输入端点，再对剩余区间循环补齐。源码不执行任何数值运算，只搬运原始 dtype 元素，因此有效输出位模式保持不变。

当 `f=k=0` 时，Tiling 把每个深度页视为独立 2D 任务，entry 复用 `CircularPad2D<T>`；只要 `f!=0 || k!=0`（包括负裁剪），就按完整 volume 分核并使用 `CircularPad3D<T>`，最后从已写好的 `y` 中循环复制 front/back 页。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 4D 或 5D；公开格式 ND，API 允许非私有 view format 并要求与 `out` 一致 | INT8/FP16/BF16/FP32/INT32 | 必选；非连续时先 `Contiguous`；4D 空 tensor 一律报错，5D 仅允许 batch 维为 0 且其余输入维非 0 的空路径。 |
| `padding` | 长度 6 的 Host `aclIntArray` | INT64 | 顺序 `(left,right,top,bottom,front,back)`；正值按公开契约严格小于对应输入维；负值由 Tiling 支持裁剪，但不能裁空对应维。 |
| 内部 `paddingsTensor` | 长度 `2*rank`，与输入 rank 成对连续 | INT64 | `GetPaddingTensor` 在 leading 维补 `(0,0)`，尾部依次放 `(front,back),(top,bottom),(left,right)`；供 PadV3 infer-shape 与 CircularPad Tiling 读取。 |
| 内部 `constantValueTensor` | 标量 | 与 `self` 相同 | 由标量 0 转换而来；`mode="circular"` 时 `CircularPadAiCore` 不把它传给 CircularPad Kernel，因此不被核心实现消费。 |
| `out` | 与 `self` 同 rank/view format；后三维按 padding 改变，leading 维不变 | 与 `self` 相同 | 必选，可非连续；核心先写内部连续 `pad3dResult`，再 `ViewCopy` 到用户输出。 |
| `workspaceSize/executor` | Host 输出 | UINT64 / executor handle | 第一段返回整个 executor 图所需空间和生命周期对象。 |
| `workspace/stream` | Device 地址 / runtime stream | 不适用 | 第二段交给 `CommonOpExecutorRun`；输入、输出与 workspace 必须存活到 stream 中执行完成。 |

公开 API 还限制 `out` 最后一维：INT8 为 `(0,98176)`，FP16/BF16 为 `(0,9088)`，INT32/FP32 为 `(0,24544)`。同仓 README 给出的 INT8/FP16/BF16 上界不同；正式调用应遵循更具体的 `aclnnCircularPad3d.md`。

### 入口与版本

固定仓只有无版本后缀的 `aclnnCircularPad3dGetWorkspaceSize/aclnnCircularPad3d`，没有 V2/V3，因此这是 A2/A3 的最新公开版本。公开产品表对 Atlas A2、A3 标记支持，对 Ascend 950PR/950DT 明确标记不支持；OpDef 与 binary config 只接入 `ascend910b/ascend910_93`（另有不属于本卡的 Kirin），全局 AscendC config 也没有 `ascend950`。因此本文只发布 A2/A3 卡，不发布 950 卡。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段先检查 `workspaceSize/executor` 公共输出指针，再检查 `self/padding/out` 非空。`CheckDtypeValid` 要求输入输出属于五种公开 dtype 且相同；`CheckFormat` 拒绝 private storage format并要求 view format 一致；`CheckShape` 要求 rank 为 4 或 5、输入输出同 rank、padding 长度为 6，并逐维核对输出 shape。

若 `self` 或 `out` 为空，函数先置 `workspaceSize=0`：4D 直接返回 `ACLNN_ERR_PARAM_INVALID`；5D 输入的第 1～4 维任一为 0 也报错，否则释放空计算 executor 并成功返回。非空路径将 `self` 连续化，按完整 rank 构造常量 paddings tensor 和同 dtype 的 0 标量，然后调用 `l0op::PadV3(...,"circular",true,...)`。PadV3 infer-shape 要求 paddings 数量恰为 `2*rank`，按连续 pair 推导输出；`mode=="circular"` 无条件选择 `CircularPadAiCore`，不会落到 PadV3 AICPU 或 950 PadV3 body。核心结果最终经 `ViewCopy` 写入用户 `out`。

### 参数、Workspace 与生命周期

`TilingPrepare4CircularPad` 从平台取得 `GetCoreNumAiv()`、UB 大小和 `GetLibApiWorkSpaceSize()`；`GetShapeAttrsInfo` 从内部 paddings 常量及 storage shape 产生 D/H/W、padding 和任务数。核心 workspace 为：

```text
core_user_workspace_bytes = totalTasks * workspaceLen * sizeof(float)
total_core_workspace_bytes = core_user_workspace_bytes + sysWorkspaceSize
```

这里固定乘 4 字节，即对 INT8/FP16/BF16 保守超配；Kernel 经 `GetUserWorkspace(workspace)` 跳过 system-workspace 前缀，再按实际 `T` 解释 user 区。`workspaceLen` 单位是元素，每核只绑定 `[workspaceLen*startIdx, workspaceLen*(startIdx+localTaskCount))`，各核不重叠。API 返回的最终 `uniqueExecutor->GetWorkspaceSize()` 还可能含 Contiguous/ViewCopy 的框架临时空间；其聚合布局与精确 system/user 偏移属于公共 executor/runtime，固定仓不可见。

executor 持有内部 paddings、标量、连续输入/输出及 launch 图；调用者持有原输入、用户输出和第二段 workspace，生命周期均覆盖同一 stream 上的异步执行。

### 启动图与失败传播

```text
aclnnCircularPad3dGetWorkspaceSize
  -> optional Contiguous(self)
  -> GetPaddingTensor + ConvertToTensor(0)
  -> PadV3 infer-shape
  -> CircularPadAiCore / ADD_TO_LAUNCHER_LIST_AICORE(CircularPad)
     -> Tiling4CircularPad
     -> one circular_pad(blockDim=useCoreNum, POD=CircularPadCommonTilingData)
        -> R-001 ... R-008 (AIV_VECTOR DMA body)
  -> ViewCopy(internal pad3dResult -> out)
aclnnCircularPad3d
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

| 节点/边 | producer→consumer | 参数或交接介质 | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| F-001 | 调用者→Contiguous→CircularPad | `self` 原 view→连续 ND GM | executor 同 stream | 原 tensor 由调用者持有；连续副本由 executor 持有 | `Contiguous==nullptr` 阻断后续。 |
| K-001 | `Tiling4CircularPad`→`circular_pad` | `x,paddings,y,workspace,tiling`；`blockDim=useCoreNum`；单个 POD | launch 与 executor 同 stream | runtime/executor 持有 tiling 与 workspace 到完成 | Tiling 任一步 `GRAPH_FAILED` 阻断 launch；未知 key 在 entry 静默返回且不写输出。 |
| W-001 | runtime→K-001 | `[system prefix][user workspace]`；user extent 为 `4*totalTasks*workspaceLen` B | 与 K-001 launch 绑定；核内靠 queue/MTE event | runtime/executor；K-001 完成后可复用 | workspace/executor 运行错误由 `CommonOpExecutorRun` 返回。 |
| F-002 | K-001→ViewCopy→用户 `out` | 内部连续 `y` GM→用户 view | executor 同 stream，K-001 在 ViewCopy 前完成 | 内部 y 由 executor，out 由调用者持有 | K-001 或 ViewCopy 失败阻断/返回错误；框架具体错误映射固定仓不可见。 |

核心只有一个算子 Device 节点，没有跨 Kernel workspace 边。Contiguous/ViewCopy 是公共框架节点，源码允许作为明确框架边界；CircularPad 自有的 selector、POD、entry/body、workspace 与同步均已闭合。

## Route 与参数设计

### 分核与资源计算

令 `t=sizeof(T)`，`A(x)=ceil(x*t/32)*32/t`，`pX=max(X,0)`，`nX=min(X,0)`：

```text
inOutputH = inputH + nTop + nBottom
inOutputW = inputW + nLeft + nRight
inOutputL = inputL + nFront + nBack
inputWAlign = A(inputW)
outputWAlign = A(outputW)
inOutputWAlign = A(inOutputW)
leftAlign = A(pLeft), rightAlign = A(pRight)
totalTasks = product(storage_shape[0 : rank-2])   # 页数，含 D
```

Host 先要求 `inputW < ubSize/(2*t)`，再以严格条件 `inOutputH*outputWAlign < ubSize/(2*t)` 选择 small-shape。small 的 `workspaceLen=inOutputH*(leftAlign+inOutputWAlign+rightAlign)`；big 的左右对齐宽在零 padding 时至少保留 `32/t`，若对齐宽超过 `inputW` 则改用实际正 padding，随后 `workspaceLen=inOutputH*(leftAlign+rightAlign)`。

无深度 padding时：`perCoreTaskNum=totalTasks/coreNum`，`tailTaskNum=totalTasks%coreNum`，前 `tailTaskNum` 核各多 1 页。存在深度 padding时令 `volumeCount=totalTasks/inputL`，`perCoreTaskNum=(volumeCount/coreNum)*inputL`、`tailTaskNum=(volumeCount%coreNum)*inputL`，前 `tailTaskNum/inputL` 核各多一个完整 volume，避免 front/back 页跨核依赖。`useCoreNum` 是实际非空核数。

### Route 选择

dtype digit 为 INT8→1、FP16→2、BF16→3、FP32→4、INT32→5。key 公式为 `(dPad?300:200)+10*dtypeDigit+shapeType`，其中 small=`shapeType=1`、big=`2`。同一个 key 在 Device `Init` 后再按 `hasLR=(left>0||right>0)` 分成两条实际搬运路线。

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 2D-small-LR | `211/221/231/241/251`；`CircularPad2D<T>` | `front==0 && back==0` 且 small 条件成立，且 `left>0 \|\| right>0` | `hasLR=false` 到 R-002；small 失败到 R-003/R-004。 |
| 2 | R-002 2D-small-direct | 同 R-001 keys；`CircularPad2D<T>` | `front==0 && back==0` 且 small，且 `left<=0 && right<=0` | small 失败到 R-004。 |
| 3 | R-003 2D-big-LR | `212/222/232/242/252`；`CircularPad2D<T>` | `front==0 && back==0` 且 small 条件不成立，且 `left>0 \|\| right>0` | `hasLR=false` 到 R-004。 |
| 4 | R-004 2D-big-direct | 同 R-003 keys；`CircularPad2D<T>` | `front==0 && back==0` 且 big，且 `left<=0 && right<=0` | 无 fallback；校验失败则 Tiling 失败。 |
| 5 | R-005 3D-small-LR | `311/321/331/341/351`；`CircularPad3D<T>` | `(front!=0 \|\| back!=0)` 且 small，且 `left>0 \|\| right>0` | `hasLR=false` 到 R-006；small 失败到 R-007/R-008。 |
| 6 | R-006 3D-small-direct | 同 R-005 keys；`CircularPad3D<T>` | `(front!=0 \|\| back!=0)` 且 small，且 `left<=0 && right<=0` | small 失败到 R-008。 |
| 7 | R-007 3D-big-LR | `312/322/332/342/352`；`CircularPad3D<T>` | `(front!=0 \|\| back!=0)` 且 big，且 `left>0 \|\| right>0` | `hasLR=false` 到 R-008。 |
| 8 | R-008 3D-big-direct | 同 R-007 keys；`CircularPad3D<T>` | `(front!=0 \|\| back!=0)` 且 big，且 `left<=0 && right<=0` | 无 fallback；校验失败则 Tiling 失败。 |

```text
CheckDtype -> CalculateParams -> CheckInput
  -> dPad = (front != 0 || back != 0)
  -> small = inOutputH * outputWAlign < ubSize / 2 / t
  -> key = (dPad ? 300 : 200) + 10*dtypeDigit + (small ? 1 : 2)
  -> entry selects CircularPad2D<T> or CircularPad3D<T>
  -> Init sets hasLR = (left > 0 || right > 0)
  -> HASLR=true/false body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 2x1 + `hasLR` | A2/A3；五 dtype；D pad=0；small；LR 正 pad | 单 K-001 | `CircularPad2D<T>::ProcessSmallShape`→`PadLeftAndRightSmallShape<true>` | 完整中间行 workspace；MTE3→MTE2 后读回 | 前若干核多 1 页；H/DMA tail 由 blockCount。 |
| R-002 | 2x1 + `!hasLR` | 同上但 LR 均非正 | 单 K-001 | `PadLeftAndRightSmallShape<false>`→直接 x→y | workspace 不被 body 使用；完整页 UB | 前若干核多 1 页。 |
| R-003 | 2x2 + `hasLR` | D pad=0；big；LR 正 pad | 单 K-001 | `ProcessBigShape`→`PadLeftAndRightBigShape<true>` | 窄条 workspace；分行 tile；阶段间 MTE event | `rowsNum` 整 tile + H tail。 |
| R-004 | 2x2 + `!hasLR` | D pad=0；big；LR 均非正 | 单 K-001 | big direct x→y | workspace 不被 body 使用；分行 tile | H tail。 |
| R-005 | 3x1 + `hasLR` | D pad非零；small；LR 正 pad | 单 K-001 | `CircularPad3D<T>::ProcessSmallShape` | 完整行 workspace；中心页后 y→y front/back | 整 volume 分核；front/back 按 96 KiB 分块。 |
| R-006 | 3x1 + `!hasLR` | D pad非零；small；LR 均非正 | 单 K-001 | small direct + `PadFrontAndBack` | workspace 不被中心页使用；front/back 从 y 读回 | volume tail。 |
| R-007 | 3x2 + `hasLR` | D pad非零；big；LR 正 pad | 单 K-001 | big narrow-strip + `PadFrontAndBack` | 窄条 workspace；多次 MTE3→MTE2 | 行 tail + front/back byte tail。 |
| R-008 | 3x2 + `!hasLR` | D pad非零；big；LR 均非正 | 单 K-001 | big direct + `PadFrontAndBack` | workspace 不被中心页使用；y→y 深度循环复制 | 同 R-007。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 单行 UB | `inputW < ubSize/(2*t)` | 一个 queue 槽仅 96 KiB，至少要容纳一条输入行 | 全部 | `GRAPH_FAILED`，无更小 tile fallback | `CircularPadTiling::CheckInput` |
| 整页 UB | `inOutputH*outputWAlign < ubSize/(2*t)` | small 一次把整个有效 H×对齐 W 放进单槽 | R-001/2/5/6 | 否则 big 路线 | `CircularPadTiling::DoOpTiling` |
| 左右正 pad | `left>0 \|\| right>0` | 需要从另一端循环复制，必须生成 workspace 中间列 | 奇数 LR Route | 否则 direct Route | `CircularPad<T>::Init` / `hasLR` |
| 深度依赖 | `front!=0 \|\| back!=0` | front/back 需读取同一 volume 已完成的 y 页 | R-005～8 | 为 0 时按独立页走 2D Route | `CalculateParams` / `DivCore` |
| padding 有效域 | 四组正负组合检查 | 防止源 offset 越界或裁空维度 | 全部 | `GRAPH_FAILED` | `CheckLeftAndRight/CheckTopAndBottom/CheckFrontAndBack` |
| DMA 窄化 | `inOutputH/top/bottom` 写入 `uint16_t blockCount`；宽/stride 写入 `uint32_t` | `DataCopyExtParams` 字段位宽 | 全部 | 固定 Host 无显式溢出 fallback；须遵守公开 shape 上界 | `CalculateLeftAndRightParams` / `CopyToOutSmallShapeOnePage` |
| BF16 编译 guard | A2/A3 不命中 `__NPU_ARCH__==3003/3113` 排除条件 | Kirin 编译不实例化 BF16，A2/A3 保留 | BF16 keys `231/232/331/332` | 被排除平台 unknown-key return | `circular_pad` router |
| 未知 key | 不属于 20 个 key | entry 无通用实现 | 全部 | 直接 `return`，输出不写 | `circular_pad` |

### POD/ABI

`CircularPadCommonTilingData` 由 15 个 `int64_t` 字段按声明顺序注册，Host `SaveToBuffer`，Device `GET_TILING_DATA` 后同名读取。宏 `TILING_DATA_FIELD_DEF` 的定义不在固定仓，故精确 `sizeof/alignof/offsetof` 不可见；不能凭普通 C++ struct 推测 padding。所有字段均由 `SetTilingData` 无条件赋值，没有未初始化 inactive 字段。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CircularPadCommonTilingData` | `inputH` | `int64_t/64` | 声明序 0；精确 offset/总大小固定仓不可见 | 元素 | `set_inputH(inputH)` | `GetTiling`→`inputLen_`/行地址 | `>0` 非空路径 |
| 同上 | `inputW` | `int64_t/64` | 声明序 1；offset 不可见 | 元素 | `set_inputW(inputW)` | 对齐、stride、offset | `<ubSize/(2*t)` |
| 同上 | `outputH` | `int64_t/64` | 声明序 2；offset 不可见 | 元素 | `set_outputH(outputH)` | `outputLen_`、top/bottom | `inputH+top+bottom` |
| 同上 | `outputW` | `int64_t/64` | 声明序 3；offset 不可见 | 元素 | `set_outputW(outputW)` | 输出行长与 DMA blockLen | `inputW+left+right` |
| 同上 | `left` | `int64_t/64` | 声明序 4；offset 不可见 | 元素 | `set_left(left)` | `hasLR`、裁剪/循环 offset | 正/负组合经检查 |
| 同上 | `right` | `int64_t/64` | 声明序 5；offset 不可见 | 元素 | `set_right(right)` | 同上 | 同上 |
| 同上 | `top` | `int64_t/64` | 声明序 6；offset 不可见 | 元素 | `set_top(top)` | 行裁剪与 top copy | 正/负组合经检查 |
| 同上 | `bottom` | `int64_t/64` | 声明序 7；offset 不可见 | 元素 | `set_bottom(bottom)` | 行裁剪与 bottom copy | 同上 |
| 同上 | `front` | `int64_t/64` | 声明序 8；offset 不可见 | 页 | `set_front(front)` | 仅 `CircularPad3D::Init3D/PadFrontAndBack` | 2D keys 写入但不读 |
| 同上 | `back` | `int64_t/64` | 声明序 9；offset 不可见 | 页 | `set_back(back)` | 仅 3D body | 2D keys inactive |
| 同上 | `inputL` | `int64_t/64` | 声明序 10；offset 不可见 | 页/volume | `set_inputL(inputL)` | 3D 分页与 volume 核分配 | 2D body 不读 |
| 同上 | `outputL` | `int64_t/64` | 声明序 11；offset 不可见 | 页/volume | `set_outputL(outputL)` | 3D y offset/stride | 2D body 不读 |
| 同上 | `perCoreTaskNum` | `int64_t/64` | 声明序 12；offset 不可见 | 页 | `DivCore`→`set_perCoreTaskNum` | Init 按 blockId 修正后控制外层循环 | 3D 时必为 `inputL` 的倍数 |
| 同上 | `tailTaskNum` | `int64_t/64` | 声明序 13；offset 不可见 | 页 | `DivCore`→`set_tailTaskNum` | 决定前部多任务核和 startIdx | 3D 时为整 volume 页数 |
| 同上 | `workspaceLen` | `int64_t/64` | 声明序 14；offset 不可见 | `T` 元素/页 | shape Route→`set_workspaceLen` | workspace GM page stride/extent | direct Route 分配但 body 可不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/2 | 2x1 + `hasLR` | `CircularPadTiling::DoOpTiling/SetTilingKey` | `CircularPadCommonTilingData` | `circular_pad`→`CircularPad2D<T>` | `circular_pad_2d.h` + `circular_pad.h/common.h` | A2/A3 JSON + AscendC config |
| R-003/4 | 2x2 + `hasLR` | 同上 | 同上 | 同上，`ProcessBigShape` | 同上 | 同上 |
| R-005/6 | 3x1 + `hasLR` | 同上，`dPad=true` | 同上，含 L/front/back | `circular_pad`→`CircularPad3D<T>` | `circular_pad_3d.h` + common body | 同上 |
| R-007/8 | 3x2 + `hasLR` | 同上 | 同上 | 同上，`ProcessBigShape` | 同上 | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD、建立 `TPipe`，用 `GetUserWorkspace` 取得 user 区，并按 key 实例化 `T∈{int8_t,half,bfloat16_t,float,int32_t}` 的 2D/3D small/big body。所有 body 共享：`InitCommon` 计算 32B 对齐、正负 padding 和行/页长度；`InitBuffer(queBind_,2,96 KiB)` 建立 VECIN→VECOUT 绑定双缓冲；每次 `AllocTensor→DataCopyPad(GM→UB)→EnQue→DeQue→DataCopyPad(UB→GM)→FreeTensor`。没有 Cast、算术 Vector、Atomic 或跨核归约。

small 将完整有效页放入一个 UB 槽；big 的 `CopyLines/CopyOnePage` 以 `rowsNum=floor(96KiB/(A(loopW)*t))` 分行。左右正 pad 先从 `x` 两端生成 workspace，中心和 top/bottom 再写 `y`；深度 pad 最后从当前核已完成的 `y` volume 中复制 front/back。`MTE3ToMTE2Sync` 的 `SetFlag/WaitFlag<MTE3_MTE2>` 保证 GM 写完成后才能作为后续 GM→UB 输入。

### Route R-001：2D-small-LR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `circular_pad_common_tiling.cpp` | `CalculateParams/SetTilingKey` | `front==0 && back==0` | key 百位 2 |
| Tiling | `circular_pad_tiling.cpp` | `DoOpTiling` | `inOutputH*outputWAlign < ubSize/2/t` | key 末位 1 |
| Router | `circular_pad.cpp` | `circular_pad` | 2x1 key | `CircularPad2D<T>::ProcessSmallShape` |
| Device | `circular_pad.h` | `Init` | `left>0 \|\| right>0` | `PadLeftAndRightSmallShape<true>` |

#### Device 数据流

##### 函数级调用链

```text
CircularPadAiCore -> circular_pad(2x1)
  -> CircularPad2D<T>::Init2D -> CircularPad<T>::Init -> InitCommon/InitBuffer
  -> ProcessSmallShape
    -> PadLeftAndRightSmallShape<true> (x -> UB -> workspace right/interior; workspace -> UB -> workspace left)
    -> MTE3ToMTE2Sync
    -> CopyToOutSmallShape<true> -> CopyToOutSmallShapeOnePage (workspace -> UB -> y center/top/bottom)
```

##### 源码等价伪码

```text
for local page i:
  copy cropped x[inOutputH,inOutputW] into one UB slot
  write aligned interior rows to workspace; if right>0, barrier then write right cyclic slice
sync workspace writes -> reads
if left>0: for each page load rightmost source slice from workspace and write left cyclic slice
sync
for each local page:
  load full padded-width effective rows from workspace into UB
  write y center rows; conditionally write top rows from UB tail and bottom rows from UB head
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `inputLen*startIdx + page*inputLen - nTop*inputW - nLeft` | `inOutputH*inOutputW` 有效元素 | T/row-major | caller/Contiguous | LR materialize | caller/executor | row stride inputW |
| workspace GM | `workspaceLen*startIdx + page*workspaceLen` | `workspaceLen=inOutputH*(leftAlign+inOutputWAlign+rightAlign)` | T/aligned rows | LR stage | output stage | runtime，K-001 内有效 | 32B 列区；Host 按 4B/元素超配 |
| `queBind_` | 单槽复用 | `2*96KiB=192KiB` | T/local | MTE2 | MTE3 | `TPipe`/Kernel | depth=2；每次 Free 后覆盖 |
| y GM | `outputLen*startIdx + page*outputLen` | `outputH*outputW` | T/row-major | output stage | ViewCopy | executor | 仅有效元素写回 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB"]:1
  q1["queBind slot1 · 96 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = depth(2) * (192 KiB / 2) = 192 KiB
small live bytes <= inOutputH * outputWAlign * t < ubSize/2 <= 96 KiB on A2/A3
dynamic_ub = N/A
workspace_total = sysWorkspaceSize + 4 * totalTasks * workspaceLen
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | `DataCopyPad` MTE2 | MTE3 workspace/y write | `FreeTensor` 后下一页 | EnQue/DeQue/FreeTensor | 单次有效页不超过 96 KiB |
| workspace page | LR materialize | output stage | K-001 完成后 | `MTE3ToMTE2Sync` | 各核 page 范围不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| x→UB→workspace | queue + `PIPE_MTE3` barrier | MTE2/MTE3 | 下一次 workspace 写 | 同一 local/GM 行 | 位拷贝；blockLen/stride 以字节传入 |
| workspace 写→读 | `MTE3_MTE2 SetFlag/WaitFlag` | MTE3 | 后续 MTE2 | workspace page | H 由 `uint16_t blockCount` 表示 |
| UB→y | queue | MTE2 | MTE3 | queue slot | top/bottom 只写有效行，无算术精度变化 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/circular_pad/op_host/circular_pad_tiling.cpp` | `DoOpTiling` | small key 与完整行 workspace 公式 |
| Kernel entry/router | `conversion/circular_pad/op_kernel/circular_pad.cpp` | keys 211/221/231/241/251 | dtype→`CircularPad2D<T>::ProcessSmallShape` |
| Device body/helper | `conversion/circular_pad/op_kernel/circular_pad.h` | `PadLeftAndRightSmallShape<true>` / `CopyToOutSmallShapeOnePage` | 完整行中间化、左右/top/bottom 循环复制 |

### Route R-002：2D-small-direct

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `circular_pad_tiling.cpp` / `circular_pad.cpp` | `DoOpTiling` / 2x1 keys | D pad=0 且 small | `CircularPad2D<T>` |
| Device | `circular_pad.h` | `Init/ProcessSmallShape` | `left<=0 && right<=0` | `PadLeftAndRightSmallShape<false>`，从 x 直接输出 |

#### Device 数据流

##### 函数级调用链

```text
circular_pad(2x1) -> Init2D -> ProcessSmallShape
  -> PadLeftAndRightSmallShape<false> (compile-time no-op)
  -> MTE3ToMTE2Sync
  -> CopyToOutSmallShape<false> -> CopyToOutSmallShapeOnePage(x -> UB -> y)
```

##### 源码等价伪码

```text
for each local page:
  src = x + page*inputLen - nTop*inputW - nLeft
  load inOutputH rows, each outputW bytes wide after crop, into aligned UB rows
  write center y rows; if top/bottom positive, reuse UB tail/head rows for cyclic fill
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `inputLen*startIdx+i*inputLen-nTop*inputW-nLeft` | cropped page | T/row-major | caller | UB | caller | negative LR only crops |
| `queBind_` | N/A | 2×96 KiB | T/local | x MTE2 | y MTE3 | Kernel | full-page reuse |
| workspace GM | allocated slice | Host formula | byte region | none | none | runtime | inactive Route，不读写 |
| y GM | `outputLen*(startIdx+i)` | output page | T/row-major | UB | ViewCopy | executor | valid rows only |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB"]:1
  q1["queBind slot1 · 96 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB; one live page < 96 KiB
dynamic_ub = N/A
workspace_total = sysWorkspaceSize + 4*totalTasks*workspaceLen  # body user bytes=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | x→UB | UB→y | Free 后下一页 | queue lifecycle | small guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| direct copy | queue | MTE2 | MTE3 | UB slot | 原 dtype 位拷贝；blockCount 行尾 |
| top/bottom | 同一 UB local | cropped page load | additional MTE3 writes | UB head/tail | padding 行循环复用，无 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_common_tiling.cpp` | `SetTilingKey` | D pad=0 产生 2xx |
| Kernel entry/router | `circular_pad.cpp` | 2x1 keys | small 2D body |
| Device body/helper | `circular_pad_2d.h` | `CopyToOutSmallShape<false>` | 绕过 workspace，直接 x→UB→y |

### Route R-003：2D-big-LR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `circular_pad_tiling.cpp` | `DoOpTiling` | D pad=0；整页不满足 1 槽 | key 2x2、窄条 workspace |
| Device | `circular_pad.h` | `Init/ProcessBigShape` | `left>0 \|\| right>0` | big `<true>` |

#### Device 数据流

##### 函数级调用链

```text
circular_pad(2x2) -> Init2D -> ProcessBigShape
  -> PadLeftAndRightBigShape<true> -> CopyLines(x -> UB -> workspace left/right strips)
  -> MTE3ToMTE2Sync
  -> CopyToOutBigShape -> CopyToOutBigShapeOnePage
     -> CopyWSToOutOnce -> sync -> CopyInToOutOnce -> sync -> PadTopAndBottomOnce
```

##### 源码等价伪码

```text
for left and right strip independently:
  rowsNum = floor(96KiB / alignedStripBytes)
  for each page: copy full row tiles plus tail rows from x edge to narrow workspace
sync
for each page:
  workspace strips -> y left/right
  sync; cropped x center -> y hole
  sync; y source rows -> y top/bottom
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| workspace strips | `page*workspaceLen + {0,leftAlign}` | `inOutputH*(leftAlign+rightAlign)` T | two packed strips | `CopyLines` | `CopyWSToOutOnce` | runtime | 32B strip width |
| x center | `page*inputLen+leftAlign-pLeft-nLeft-nTop*inputW` | `inOutputH*holeW` | T/strided rows | caller | `CopyInToOutOnce` | caller | `holeW=inOutputW-(leftAlign-pLeft)-(rightAlign-pRight)` |
| queue | N/A | 2×96 KiB | T/local | GM MTE2 | GM MTE3 | Kernel | `rowsNum` tiles |
| y | per page | outputLen | T/row-major | strip/center/top-bottom stages | ViewCopy | executor | stages write disjoint regions except intentional y readback |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB · row tile"]:1
  q1["queBind slot1 · 96 KiB · row tile"]:1
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB
rowsNum = floor(96KiB / (A(loopW)*t)) >= 1
workspace_total = sysWorkspaceSize + 4*totalTasks*inOutputH*(leftAlign+rightAlign)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | each row tile MTE2 | MTE3 | Free 后下一 tile | queue lifecycle | inputW single-row guard |
| workspace strips | `CopyLines` | `CopyWSToOutOnce` | Kernel 后 | MTE3→MTE2 event | per-core pages disjoint |
| y center rows | center MTE3 | top/bottom MTE2 | ViewCopy 后 | MTE3→MTE2 event | source rows已完整 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| strip→workspace→y | MTE3_MTE2 | strip writer | strip reader | workspace | `loopH%rowsNum` tail |
| y center→top/bottom | MTE3_MTE2 | center writer | y reader | y page | raw dtype copy |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_tiling.cpp` | `DoOpTiling` | big narrow workspace 公式 |
| Kernel entry/router | `circular_pad.cpp` | 212/222/232/242/252 | big 2D dtype实例 |
| Device body/helper | `circular_pad.h` | `PadLeftAndRightBigShape<true>/CopyLines/CopyToOutBigShapeOnePage` | 窄条与行 tile 管线 |

### Route R-004：2D-big-direct

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `circular_pad_tiling.cpp` / `circular_pad.cpp` | big 2x2 | D pad=0，big | 2D big body |
| Device | `circular_pad.h` | `hasLR` | `left<=0 && right<=0` | `<false>` no-op，跳过 strips |

#### Device 数据流

##### 函数级调用链

```text
circular_pad(2x2) -> ProcessBigShape
  -> PadLeftAndRightBigShape<false>
  -> sync -> CopyToOutBigShapeOnePage
     -> CopyWSToOutOnce(no-op) -> sync
     -> CopyInToOutOnce(x -> UB row tiles -> y) -> sync
     -> PadTopAndBottomOnce(y -> UB -> y)
```

##### 源码等价伪码

```text
for each local page:
  tile cropped x rows by 96KiB and write y center
  after MTE3->MTE2 sync, tile source y rows and fill top/bottom
workspace is bound but never read or written
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x | cropped row offsets | `inOutputH*inOutputW` | T | caller | queue | caller | tiled rows |
| queue | N/A | 2×96 KiB | T | x/y MTE2 | y MTE3 | Kernel | reused by center and top/bottom |
| workspace | allocated | Host big formula | bytes | none | none | runtime | inactive |
| y | page offset | outputLen | T | center then top/bottom | ViewCopy | executor | event protects readback |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB"]:1
  q1["queBind slot1 · 96 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB; dynamic_ub=N/A
workspace_total = sysWorkspaceSize + 4*totalTasks*workspaceLen; body user consumption=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | x/y MTE2 | y MTE3 | Free 后 | queue | rowsNum>=1 |
| y center | center writer | top/bottom reader | ViewCopy | explicit event | source row不与未完成写冲突 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| center | queue | x MTE2 | y MTE3 | row tile | loop+tail 行，位拷贝 |
| top/bottom | MTE3_MTE2 | center y | y MTE2 | y page | `top/bottom` 行 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_tiling.cpp` | big branch | key/workspace |
| Kernel entry/router | `circular_pad.cpp` | 2x2 | body映射 |
| Device body/helper | `circular_pad.h` / `circular_pad_2d.h` | `<false>` / `CopyToOutBigShape` | 直接中心行与 y 自读回 |

### Route R-005：3D-small-LR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `circular_pad_common_tiling.cpp` | `CalculateParams/DivCore` | `front!=0 \|\| back!=0` | key 3x1，整 volume 分核 |
| Device | `circular_pad.h` | `hasLR` | LR 正 pad | 3D small `<true>` |

#### Device 数据流

##### 函数级调用链

```text
circular_pad(3x1) -> CircularPad3D<T>::Init3D
  -> ProcessSmallShape -> LR full-row workspace -> center/top/bottom y
  -> MTE3ToMTE2Sync -> PadFrontAndBack -> CopyGmToGm(y -> UB -> y)
```

##### 源码等价伪码

```text
assign each core whole volumes (page count multiple of inputL)
for each volume and surviving input depth page:
  perform R-001 2D small LR pipeline into y page pFront+page
sync all center-page writes of this core
if front>0: copy last front source pages of y to y front
if back>0: copy first back source pages of y to y back
each page group is chunked into 96KiB blocks plus byte tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x volume | `inputLen*startIdx` | `inputLen*localPages` | T/DHW pages | caller | center stage | caller | whole-volume ownership |
| workspace | `workspaceLen*startIdx` | `workspaceLen*localPages` | T/aligned row pages | LR stage | center stage | runtime | same as R-001 per page |
| y volume | `outputLen*(startIdx*outputL/inputL)` | `outputLen*(localPages*outputL/inputL)` | T/DHW pages | center then front/back | ViewCopy | executor | current core exclusive |
| queue | N/A | 2×96 KiB | T | x/workspace/y MTE2 | workspace/y MTE3 | Kernel | all stages reuse |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB · 2D/front-back chunk"]:1
  q1["queBind slot1 · 96 KiB · 2D/front-back chunk"]:1
```

##### 字节账与复用顺序

```text
UB_total=192 KiB; small live page<96 KiB
front/back loop = floor(outputLen*pages*t / 96KiB), tail bytes = mod(...)
workspace_total=sysWorkspaceSize+4*totalTasks*workspaceLen
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue | 2D or depth stage MTE2 | MTE3 | Free 后 | queue | stage sequential |
| y center pages | 2D stage | `PadFrontAndBack` | ViewCopy 后 | MTE3_MTE2 event | full volume assigned to one core |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 2D stages | queue/events | x/workspace | y | page | 同 R-001 |
| depth stage | MTE3_MTE2 before call + queue | completed y center | front/back reader/writer | volume | `CopyGmToGm` 96KiB loop + byte tail；位拷贝 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_common_tiling.cpp` | `DivCore` | volume 粒度分核 |
| Kernel entry/router | `circular_pad.cpp` | 311/321/331/341/351 | 3D small实例 |
| Device body/helper | `circular_pad_3d.h` / `circular_pad.h` | `ProcessSmallShape/PadFrontAndBack/CopyGmToGm` | 2D页完成后深度循环复制 |

### Route R-006：3D-small-direct

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `circular_pad_common_tiling.cpp` / `circular_pad.cpp` | dPad + 3x1 | D pad非零、small | `CircularPad3D<T>` |
| Device | `circular_pad.h` | `hasLR` | LR 均非正 | direct center + depth copy |

#### Device 数据流

##### 函数级调用链

```text
circular_pad(3x1) -> Init3D -> ProcessSmallShape
  -> PadLeftAndRightSmallShape<false>
  -> CopyToOutSmallShape<false>(x -> UB -> y center/top/bottom)
  -> sync -> PadFrontAndBack(y -> UB -> y)
```

##### 源码等价伪码

```text
for whole volumes owned by core:
  for each surviving D page: direct cropped x page -> UB -> y center/top/bottom
  sync; copy cyclic front/back pages inside y in 96KiB chunks
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y volumes | 同 R-005 | local whole volumes | T/DHW | caller/Kernel | Kernel/ViewCopy | core exclusive | center direct；depth y→y |
| queue | N/A | 2×96 KiB | T | GM reads | GM writes | Kernel | 2D full page + depth chunks |
| workspace | allocated | Host small formula | bytes | none | none | runtime | inactive |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB"]:1
  q1["queBind slot1 · 96 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total=192 KiB; dynamic_ub=N/A
workspace_total=sysWorkspaceSize+4*totalTasks*workspaceLen; body user consumption=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | center/depth MTE2 | MTE3 | Free 后 | queue | small/page或96KiB chunk上界 |
| y center | center | depth stage | ViewCopy | event | whole volume per core |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| center→depth | MTE3_MTE2 | y center writes | y depth reads | volume | raw copy |
| depth tail | queue | y source | y destination | final <96KiB bytes | `paramsFront.blockLen=tail` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_common_tiling.cpp` | dPad/DivCore | 3D key和volume任务 |
| Kernel entry/router | `circular_pad.cpp` | 3x1 | 3D small |
| Device body/helper | `circular_pad_3d.h` | `CopyToOutSmallShape<false>/PadFrontAndBack` | workspace-free center 与深度复制 |

### Route R-007：3D-big-LR

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `circular_pad_tiling.cpp` | `DoOpTiling` | dPad，big，LR正 pad | key 3x2 |
| Router/Device | `circular_pad.cpp` / `circular_pad_3d.h` | 3x2 / `ProcessBigShape` | `hasLR=true` | narrow strips + depth copy |

#### Device 数据流

##### 函数级调用链

```text
circular_pad(3x2) -> Init3D -> ProcessBigShape
  -> PadLeftAndRightBigShape<true>/CopyLines
  -> sync -> CopyToOutBigShape -> per-page CopyToOutBigShapeOnePage
  -> sync -> PadFrontAndBack/CopyGmToGm
```

##### 源码等价伪码

```text
for each whole volume:
  materialize left/right strips of every surviving D page in narrow workspace using row tiles
  for each page: strips->y; sync; x center->y; sync; y center rows->top/bottom
  sync; completed y pages->front/back in 96KiB chunks
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| workspace strips | per input page | `workspaceLen=inOutputH*(leftAlign+rightAlign)` | T/two strips | x stage | y LR stage | runtime | 32B widths |
| x/y volume | Init3D formulas | whole local volume | T/DHW | caller/Kernel | Kernel/ViewCopy | core exclusive | page and depth tiles |
| queue | N/A | 2×96KiB | T | all GM reads | all GM writes | Kernel | sequential stage reuse |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB · row/depth tile"]:1
  q1["queBind slot1 · 96 KiB · row/depth tile"]:1
```

##### 字节账与复用顺序

```text
UB_total=192KiB
workspace_total=sysWorkspaceSize+4*totalTasks*inOutputH*(leftAlign+rightAlign)
row tail=loopH%rowsNum; depth tail=(outputLen*pages*t)%96KiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue | current row/depth tile | GM writer | Free 后 | queue | one aligned row<=96KiB |
| workspace | LR strip stage | y LR stage | Kernel后 | event | unique page range |
| y center | page stages | top/bottom then depth | ViewCopy | per-page and pre-depth events | whole volume core ownership |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| strips/center/top-bottom | repeated MTE3_MTE2 | previous GM writer | next GM reader | workspace/y page | row tail，位拷贝 |
| page→depth | MTE3_MTE2 | all page writes | `PadFrontAndBack` | y volume | 96KiB byte tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_tiling.cpp` | big workspace branch | strip容量 |
| Kernel entry/router | `circular_pad.cpp` | 312/322/332/342/352 | 3D big dtype实例 |
| Device body/helper | `circular_pad_3d.h` + `circular_pad.h` | `ProcessBigShape/CopyLines/PadFrontAndBack` | 完整多阶段搬运网络 |

### Route R-008：3D-big-direct

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `circular_pad_tiling.cpp` / `circular_pad.cpp` | dPad + big 3x2 | LR均非正 | 3D big direct |
| Device | `circular_pad.h` | `PadLeftAndRightBigShape<false>` | compile-time no-op | x center→y→depth |

#### Device 数据流

##### 函数级调用链

```text
circular_pad(3x2) -> Init3D -> ProcessBigShape
  -> PadLeftAndRightBigShape<false>
  -> CopyToOutBigShape -> CopyInToOutOnce/PadTopAndBottomOnce
  -> sync -> PadFrontAndBack
```

##### 源码等价伪码

```text
for each owned whole volume and surviving depth page:
  tiled cropped x rows -> y center; sync; y rows -> y top/bottom
sync; y source pages -> front/back pages in 96KiB chunks
workspace remains inactive
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y | Init3D formulas | whole volume | T/DHW | caller/Kernel | Kernel/ViewCopy | core exclusive | row/depth tiles |
| queue | N/A | 2×96KiB | T | GM MTE2 | GM MTE3 | Kernel | all stages reuse |
| workspace | allocated | Host big formula | bytes | none | none | runtime | inactive |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBind slot0 · 96 KiB"]:1
  q1["queBind slot1 · 96 KiB"]:1
```

##### 字节账与复用顺序

```text
UB_total=192KiB; dynamic_ub=N/A
workspace_total=sysWorkspaceSize+4*totalTasks*workspaceLen; body user consumption=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue | current GM load | current GM store | Free 后 | queue | row guard |
| y center/pages | center/top-bottom | later y readback | ViewCopy | explicit events | per-core whole volume |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| x→y→y rows | queue + event | previous y writer | top/bottom reader | page | row tile tail |
| y→y depth | queue + pre-event | center pages | front/back writer | volume | depth byte tail；无 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `circular_pad_common_tiling.cpp` / `circular_pad_tiling.cpp` | dPad + big | volume任务与key |
| Kernel entry/router | `circular_pad.cpp` | 3x2 | 3D big |
| Device body/helper | `circular_pad_3d.h` / `circular_pad.h` | `<false>/CopyToOutBigShape/PadFrontAndBack` | workspace-free 中心与深度复制 |

## 算子特定优化

### Trick T-001：整页与窄条 workspace 双路线

#### 触发与源码依据

- 适用 Route：R-001～R-008；small 命中 R-001/2/5/6，否则 R-003/4/7/8。
- 触发条件：`inOutputH*outputWAlign < ubSize/2/tSize`。
- Golden 来源：[`conversion/circular_pad/op_host/circular_pad_tiling.cpp:174-184`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_tiling.cpp#L174-L184)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (inOutputH * outputWAlign < (int64_t)ubSize / BUFFER_NUM / tSize) {
    shapeType = DIM_1;
    workspaceLen = inOutputH * (leftAlign + inOutputWAlign + rightAlign);
} else {
    shapeType = DIM_2;
    leftAlign = left > 0 ? leftAlign : ALIGN / tSize;
    rightAlign = right > 0 ? rightAlign : ALIGN / tSize;
    leftAlign = leftAlign > inputW ? left : leftAlign;
    rightAlign = rightAlign > inputW ? right : rightAlign;
    workspaceLen = inOutputH * (leftAlign + rightAlign);
}
```

#### 机制与设计原因

small 用一个 96 KiB 槽承载完整有效页，能以二维 `DataCopyPad` 合并多行；一旦整页放不下，big 只物化左右循环条带，并按 `rowsNum=floor(96KiB/alignedStripBytes)` 分块搬运，避免为宽大页面在 workspace 物化中心区域。`shapeType` 同时进入 key，Device 不必在 hot loop 中再判断整页容量。

#### 迁移边界

- 适用：行式 padding/layout transform，且中间态可以拆成窄边界与直接中心区。
- 前提：单条输入行必须放入 96 KiB；DMA stride/blockLen 位宽与 32B 对齐成立。
- 失效：small 条件不满足必须走 big；单行也不满足则 Tiling 直接失败，没有更细列 tile。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的真实失败边界是仍强行让超 96 KiB 的输入行进入实现：`CheckInput` 会返回 `GRAPH_FAILED`；固定仓没有列切分 fallback。

### Trick T-002：深度 padding 时以完整 volume 分核

#### 触发与源码依据

- 适用 Route：R-005～R-008。
- 触发条件：`front!=0 || back!=0`。
- Golden 来源：[`conversion/circular_pad/op_host/circular_pad_common_tiling.cpp:141-152`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.cpp#L141-L152)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (dPad) {
    int64_t batchNum = totalTasks / inputL;
    perCoreTaskNum = (batchNum / coreNum) * inputL;
    tailTaskNum = (batchNum % coreNum) * inputL;
    useCoreNum = perCoreTaskNum > 0 ? coreNum : tailTaskNum / inputL;
} else {
    perCoreTaskNum = totalTasks / coreNum;
    tailTaskNum = totalTasks % coreNum;
    useCoreNum = perCoreTaskNum > 0 ? coreNum : tailTaskNum;
}
```

#### 机制与设计原因

front/back 复制需要读取同一输出 volume 中已经完成的中心页。以 `inputL` 页为最小任务粒度，使一个 volume 的中心、H/W padding 和 D padding 都由同一 AIV 核完成；这样只需核内 MTE3→MTE2 事件，不需要跨核 barrier、Atomic 或额外全局同步。`tailTaskNum` 也保持 `inputL` 的整数倍。

#### 迁移边界

- 适用：输出边界依赖同一 group/volume 已生成的内部页。
- 前提：group 内所有页可由一个核串行处理，且不同 group 的输出范围不重叠。
- 失效：若把单个 volume 的页拆到多核，现有 `PadFrontAndBack` 前只有核内 event，不能保证跨核可见性，必须改成多 Kernel 或显式全局同步设计。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的可验证错误边界是沿用无 D padding 的逐页分核：这会破坏 `PadFrontAndBack` 所依赖的同核完整 y volume 生命周期；源码通过 `dPad` 分支明确避免该做法。

### Trick T-003：front/back 为零时复用逐页 2D body

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：`front==0 && back==0`，即 `dPad=false`。
- Golden 来源：[`conversion/circular_pad/op_host/circular_pad_common_tiling.cpp:155-161`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.cpp#L155-L161)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (dPad) {
    tilingKey_ = TYPE_3D + DATA_TYPE * dataType + shapeType;
} else {
    tilingKey_ = TYPE_2D + DATA_TYPE * dataType + shapeType;
}
```

#### 机制与设计原因

3D API 的 D padding 为零时，各 D 页互不依赖。Host 直接产生 2xx key，entry 用 `CircularPad2D<T>` 将 `totalTasks` 个页面独立分配到更多 AIV 核，省去 3D volume 循环与无效 front/back 阶段，同时复用完全相同的 H/W DMA body。这是逻辑 CircularPad3d 的可达实现复用，不是另一个 ACLNN 算子调用。

#### 迁移边界

- 适用：高维算子的外层维无数据依赖，可展平成独立低维页面。
- 前提：front/back 都严格为 0；负 front/back 也属于依赖路线，不能复用 2D key。
- 失效：任一 D padding 非零必须使用 3xx key和完整 volume 分核。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若仅以“正 D padding”为条件，负 D cropping 会错误进入 2D Route；源码使用 `front!=0 || back!=0`，保证负裁剪也维持 3D 页索引关系。

## 模板与实现对比

| 对比维度 | R-001/2 small 2D | R-003/4 big 2D | R-005/6 small 3D | R-007/8 big 3D | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | D pad=0，整页<半 UB | D pad=0，整页不满足 | D pad非零，整页<半 UB | D pad非零，整页不满足 | 先 D 依赖，再 UB 容量；LR在 Device 内分支 |
| 模板与 dtype | `CircularPad2D<T>` 五 dtype | 同左 | `CircularPad3D<T>` 五 dtype | 同左 | dtype只改变 t、对齐和 key 十位 |
| 循环与分核 | 独立页；整页 DMA | 独立页；rowsNum 行 tile | 完整 volume；整页 DMA + depth chunks | 完整 volume；行 tile + depth chunks | 避免跨核 D 依赖 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2×96 KiB `TQueBind` | 同左 | 同左 | 同左 | 无 TBuf/dynamic UB/通信 |
| 内存总占用与复用 | 192 KiB；LR时完整行 workspace | 192 KiB；LR时窄条 workspace | small workspace + y depth readback | narrow workspace + y depth readback | HASLR=false 时 user workspace 不消费但 Host仍分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad` 2D | `DataCopyPad` tiled | 同左 + `CopyGmToGm` | 同左 | 无数值 Vector/SIMT/NDDMA/通信 |
| 同步、精度与 tail | queue/MTE event；位拷贝 | 行 tail | page + depth byte tail | 行 + depth tail | 所有 dtype 不 cast |
| fallback/失败边界 | 超整页到 big | 单行超 UB失败 | 超整页到 big | 单行超 UB失败 | unknown key直接返回 |

每列内部的 LR/Direct 差异：LR 路线将正左右 padding 物化到 workspace；Direct 路线只有负或零 LR，跳过 workspace 并从裁剪后的 x 直接写 y。top/bottom 和 front/back 的正值分别以条件阶段实现，不改变 key。

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` config | 公开支持；`ascend910_93` config | 公开文档明确不支持，不发布卡 |
| Host/Launcher | 共用 `CircularPadAiCore`、同一 Tiling/POD、五 dtype | 同 A2 | 尽管 `pad_v3` 另有 arch35 实现，`aclnnCircularPad3d` 的公开产品表为 ×，不能据目录存在发布 |
| Kernel/资源 | 同一 20 key entry；AIV 核；固定 192 KiB queue | 同 A2 | 本入口无公开可达 CircularPad binary/config |

## 限制与失败边界

- 公开契约要求正 padding 严格小于对应输入维；Tiling 的部分正值检查实际允许等于输入维，这是文档与实现边界差异，调用者不得依赖较宽松实现。
- `inputW >= ubSize/(2*t)` 直接 Tiling 失败；没有列切分 fallback。small/big 只解决整页是否放入单槽。
- small Route 把 H/top/bottom 窄化为 `uint16_t blockCount`，多处 byte length/stride 窄化为 `uint32_t`；固定 Host 没有逐字段溢出检查，应同时遵守公开 shape 上界和平台 DMA 约束。
- `workspaceSize` 对核心 user 区按每元素 4B计算，而 Device 按实际 T 访问；移植时不能把它误改成 `sizeof(T)` 后又保留原 runtime/system 前缀协议。
- 空路径不启动 Kernel：4D 空输入报错；5D 只有 batch 可为 0。`out->IsEmpty()` 也进入这段逻辑，固定源码没有进一步核验输出空维是否只来自 batch。
- `paddings` GM 参数传入 `circular_pad` 但 body 不读；真实 padding 仅来自 POD。未知 key 静默返回，无 Device 错误码或 fallback。
- `CircularPadCommonTilingData` 宏展开的精确 `sizeof/alignof/offsetof`、`GetUserWorkspace` 的 system/user 精确偏移、executor 聚合 Contiguous/ViewCopy workspace 的细节固定仓不可见。
- 950 公开不支持，因此 `circular_pad_3d_950.md` 不应创建；Kirin 路线不属于目标平台。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/circular_pad/docs/aclnnCircularPad3d.md` | 产品表、原型、约束 | A2/A3支持、950不支持、dtype/rank/width契约 |
| 入口声明/约束 | `conversion/circular_pad/op_host/op_api/aclnn_circular_pad3d.h` | 两段接口声明 | ABI 参数与公开语义 |
| 入口实现 | `conversion/circular_pad/op_host/op_api/aclnn_circular_pad3d.cpp` | `CheckParams/GetPaddingTensor/aclnnCircularPad3d*` | 校验、空路径、Contiguous、PadV3、ViewCopy、executor |
| Level0 selector | `conversion/pad_v3/op_api/padv3.cpp` | `PadV3/CircularPadAiCore` | circular mode无条件添加 CircularPad AICore launcher |
| shape infer | `conversion/pad_v3/op_host/pad_v3_infershape.cpp` | `PadV3Infershape/InferShape4PadV3` | 完整 rank padding pair 与输出 shape |
| 平台注册 | `conversion/circular_pad/op_host/circular_pad_def.cpp` | `CircularPad::CircularPad` | dtype、ND 与 A2/A3 config |
| POD | `conversion/circular_pad/op_host/circular_pad_common_tiling.h` | `CircularPadCommonTilingData` | 字段声明顺序与 compile info |
| Host公共Tiling | `conversion/circular_pad/op_host/circular_pad_common_tiling.cpp` | `GetShapeAttrsInfo/DivCore/SetTilingKey/SetTilingData` | 形状、分核、key、workspace |
| Host Tiling | `conversion/circular_pad/op_host/circular_pad_tiling.cpp` | `DoOpTiling/Tiling4CircularPad/TilingPrepare4CircularPad` | guard、small/big、AIV核、注册 |
| Kernel entry/body | `conversion/circular_pad/op_kernel/circular_pad.cpp` | `circular_pad` | 20 key router与unknown key终点 |
| Kernel helper | `conversion/circular_pad/op_kernel/circular_pad_common.h` | `InitCommon/MTE3ToMTE2Sync` | 对齐、公共字段与事件 |
| Kernel body | `conversion/circular_pad/op_kernel/circular_pad.h` | `CircularPad<T>` 全部方法 | queue、small/big、DMA参数、workspace/top-bottom |
| Kernel body | `conversion/circular_pad/op_kernel/circular_pad_2d.h` | `CircularPad2D<T>` | D pad为零时的可达逐页复用 body |
| Kernel body | `conversion/circular_pad/op_kernel/circular_pad_3d.h` | `CircularPad3D<T>` | volume分核、D循环复制 |
| binary config | `conversion/circular_pad/op_host/config/ascend910b/circular_pad_binary.json` | `CircularPad` op_list | A2五 dtype binary |
| binary config | `conversion/circular_pad/op_host/config/ascend910_93/circular_pad_binary.json` | `CircularPad` op_list | A3五 dtype binary |
| 构建接线 | `conversion/circular_pad/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN/Host模块收集 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `CircularPad` | AscendC compute units与auto_sync |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15 个 INT64 字段统一服务2D/3D、small/big | 字段声明/序列化顺序、元素单位、2D inactive 的 L/front/back | `CircularPadCommonTilingData/SetTilingData/GetTiling` |
| 调用顺序 | circular mode复用 PadV3 infer，再只启动一个 CircularPad核心 | Contiguous→infer/launcher→核心→ViewCopy；3D 页写完后才能 D padding | `aclnnCircularPad3dGetWorkspaceSize/PadV3/Process*` |
| workspace/内存 | small完整行、big窄条；direct 路线可跳过 body workspace | system prefix、Host固定4B超配、per-core offset/extent、2×96KiB queue | `GetWorkspaceSize/Init2D/Init3D/InitBuffer` |
| 同步 | queue生命周期与 MTE3→MTE2 event可复用 | workspace/y 从写转读前不可删除事件；volume不能跨核拆分 | `MTE3ToMTE2Sync/CircularPad3D::ProcessSmallShape/CircularPad3D::ProcessBigShape` |
| 精度与尾块 | 所有 dtype 位拷贝，无 cast；统一行/depth tile | 32B align、`uint16_t/uint32_t` DMA字段、row/depth tail有效长度 | `CopyLines/CopyOnePage/CopyGmToGm` |
| 失败边界 | small不适用自动到big | 单行超 UB、padding越界、unknown key无 fallback；950公开不支持 | `CheckInput/Check* / circular_pad` |