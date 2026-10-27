---
type: Operator
title: "CircularPad3dBackward A2/A3 ACLNN 算子知识"
description: "说明 aclnnCircularPad3dBackward 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/docs/aclnnCircularPad3dBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_api/aclnn_circular_pad3d_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_api/aclnn_circular_pad3d_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_api/padv3grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/pad_v3_grad/op_host/pad_v3_grad_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_host/circular_pad_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/circular_pad_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad_2d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad_3d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad/op_kernel/circular_pad_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/circular_pad_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/config/ascend910b/circular_pad_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_host/config/ascend910_93/circular_pad_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T23:49:49Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# CircularPad3dBackward A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCircularPad3dBackwardGetWorkspaceSize` + `aclnnCircularPad3dBackward`，固定仓中唯一公开、无版本后缀的 ACLNN 版本 |
| 计算语义 | 对 3D circular pad 做反向散加：正向中映射到同一 `self[D,H,W]` 位置的 `gradOutput` 元素在 `gradInput` 中求和；负 padding 表示裁剪 |
| 输入输出 | `gradOutput/self/gradInput` 同 dtype、同 rank（4D 或 5D），`self.shape == gradInput.shape`；dtype 为 FP16/BF16/FP32；padding 为 6 个 INT64，顺序 left/right/top/bottom/front/back |
| 执行类型 | ACLNN executor 中有框架 `Contiguous`/`ViewCopy` 边界；核心节点为单个 `CircularPadGrad` AIV_VECTOR Kernel，不含 Cube/AIC 或 MIX_CV |
| Kernel 路线 | R-001～R-004：front/back 均为 0 的 slice 路线；R-005～R-008：真实 depth circular/crop 路线；各自再按 small/big 与 FP32/16-bit cast 分开 |
| 核心资源 | FP32 路线：一条 depth=2 的 192 KiB AIV UB queue；FP16/BF16 路线：depth=2 的 32 KiB T1 queue 与 depth=2 的 64 KiB FP32 queue；GM FP32 workspace 加系统 workspace |
| 来源闭包 | source_closed：公开 ACLNN→L0 circular selector→CircularPadGrad Host/Tiling/POD→arch32 router→全部 2D/3D、small/big、dtype body→A2/A3 config/CMake 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `p=(l,r,t,b,f,k)`，`gradOutput` 最后三维为 `(Di,Hi,Wi)`，`self/gradInput` 最后三维为 `(Do,Ho,Wo)`，接口校验要求 `Di=Do+f+k`、`Hi=Ho+t+b`、`Wi=Wo+l+r`。对正向 circular 映射 `M_p(d,h,w)`，反向结果为

```text
gradInput[u,v,w] = Σ gradOutput[d,h,x]
                     over every (d,h,x) whose M_p(d,h,x)=(u,v,w)
```

Kernel 不用 atomic 写最终 `gradInput`，而是先在每个 task 独占的 FP32 workspace 页面中依次合并 top/bottom、left/right、front/back 重叠，最后搬到输出。任一 padding 为负时，输出先清零，随后只提交保留下来的有效区域。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | 4D/5D；公开文档为 ND，API 实现接受非 private 且 view format 与另外两张量相同 | FP16/BF16/FP32 | rank 与 `self/gradInput` 相同；最后三维等于 `self` 最后三维加两侧 padding；非连续时先 `Contiguous` |
| `self` | 4D/5D | 与 `gradOutput` 相同 | 只提供正向输入的 shape/dtype/format；Kernel 不读取其值；shape 等于 `gradInput` |
| `padding` | 长度 6 | INT64 host array | `[left,right,top,bottom,front,back]`；每值必须小于对应 `self` 维长；Host 还施加两侧正 padding 合计/单侧上界 |
| `gradInput` | 与 `self` 相同；可非连续 | 与 `self` 相同 | 核心 Kernel 产生连续临时结果，再由 `ViewCopy` 写入用户 view |
| `workspaceSize/executor` | 标量/句柄 | `uint64_t*`/`aclOpExecutor**` | 第一段返回完整 executor 图及其总 workspace |
| `workspace/workspaceSize/executor/stream` | 第二段执行参数 | runtime 类型 | `CommonOpExecutorRun` 在给定 stream 上执行第一段封装的顺序图 |

### 入口与版本

固定 Commit 只公开无后缀版本 `aclnnCircularPad3dBackwardGetWorkspaceSize`/`aclnnCircularPad3dBackward`，因此它是 A2/A3 实际支持的最新公开版本。专用 API 文档明确支持 Atlas A2、A3；`CircularPadGrad` OpDef、`ascend910b`/`ascend910_93` binary config 与 CMake 的 arch32 选择一致。L0 `PadV3Grad` 在 `mode=="circular" && !IsRegBase()` 时添加 `CircularPadGrad` AICore launcher；A2/A3 正是此分支。

## Host/Launcher 执行流程

### 参数校验与前处理

`CommonPad3dBackward` 顺序执行：创建 executor；检查四个业务指针、三张量 dtype 支持且一致、非 private format 且 view format 一致、`self==gradInput` shape、rank 4～5、padding 长度 6、各 padding 小于对应 self 维、`gradOutput` 最后三维 shape 关系。空 tensor 时 4D 必然报错；5D 只允许 batch 维为 0 且其余维非 0，并返回 workspace 0 的空 executor。非空时对 `gradOutput` 和 `self` 做 `Contiguous`，但后续核心只消费连续 `gradOutput`；`self` 的连续结果没有作为核心 Kernel 输入。

`GetPaddingTensor` 将 6 个 padding 扩展到 `2*rank`：高维未填充轴写零，最后三轴按 `[front,back,top,bottom,left,right]` 排布；`PadV3Grad` infer-shape 逐轴执行 `out_dim=x_dim-pad_front-pad_end`。L0 的 circular selector 在 A2/A3 添加单个 `CircularPadGrad(x,paddings)->y`，然后 ACLNN 图添加 `ViewCopy(y,gradInput)`。

### 参数、Workspace 与生命周期

Tiling 从 `gradOutput` 得到 `inputL/H/W`，从核心连续输出得到 `outputL/H/W`，从常量 paddings tensor 读取三组 padding。`workspaceLen` 单位为 FP32 元素：small 为 `inputH*(inputWAlign+16)`，big 为 `(inputH-pTop-pBottom)*(inputWAlign+16)`，其中 `inputWAlign=align_up(inputW*sizeof(T),32)/sizeof(T)`。核心用户 workspace 为 `totalTasks*workspaceLen*4` 字节，框架再加 `sysWorkspaceSize`；Device 通过 `GetUserWorkspace(workspace)` 跳过系统区。每个 task 拥有互不重叠的 `workspaceLen` slice，Kernel 结束前有效；executor 拥有 Contiguous 临时量、padding tensor、核心输出和 ViewCopy 图，第二段完成后才可释放。

### 启动图与失败传播

```text
GetWorkspaceSize phase
  Contiguous(gradOutput)                         # framework boundary
  Contiguous(self)                               # shape/lifetime boundary, value not consumed by core
  GetPaddingTensor -> PadV3Grad infer shape
  CircularPadGradAiCore -> add CircularPadGrad(x,paddings)->tmp
  ViewCopy(tmp, gradInput)                       # framework boundary

Run phase on one stream
  framework preprocessing nodes, if needed
  circular_pad_grad(x, paddings, tmp, user_workspace, tiling)  # one AIV node
  ViewCopy(tmp, gradInput)
```

核心节点 `blockDim=useCoreNum`，POD 为 `CircularPadCommonTilingData`，输入是连续 `gradOutput` 与 padding tensor（Device body 不读取后者），输出是连续临时 `y`，workspace slice 从系统 workspace 后开始。所有节点由同一 executor/stream 排序；任一 `Contiguous`、infer-shape、launcher add 或 `ViewCopy` 返回空指针都会阻断 executor 构造，第二段 `CommonOpExecutorRun` 返回运行状态。框架内部 Contiguous/ViewCopy 的 Kernel、临时 workspace 分片和精确 event 在固定算子仓不可见。

## Route 与参数设计

### 分核与资源计算

设 `totalTasks=Π gradOutput.shape[0:rank-2]`。若 `front!=0 || back!=0`，令 `batchNum=totalTasks/inputL`，则 `perCoreTaskNum=floor(batchNum/coreNum)*inputL`、`tailTaskNum=(batchNum%coreNum)*inputL`，整批 depth pages 不跨核；否则按独立 `(H,W)` page 分核：`perCoreTaskNum=floor(totalTasks/coreNum)`、`tailTaskNum=totalTasks%coreNum`。`useCoreNum` 是有任务的 core 数。

FP32：`usedUBSize=ubSize/2`。FP16/BF16：`usedUBSize=ubSize/2/3`。若 `inputH*inputWAlign < usedUBSize/sizeof(T)` 选 small，否则 big。Host 还要求 `inputW < usedUBSize/sizeof(T)`，因此每行至少能放入相应 UB buffer。Kernel 固定 `UB_SIZE=192 KiB`，A2/A3 Host 资源选择与此实现配套。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 depth-flat FP32 small | 211 / `CircularPadGrad2D<float,float,false>` | `front==0 && back==0 && dtype=FP32 && inputH*inputWAlign < ubSize/8` | R-002 或 dtype 路线 |
| 2 | R-002 depth-flat FP32 big | 212 / 同模板 | `front==0 && back==0 && dtype=FP32` 且 small 谓词为假 | 输入 guard 失败或 Kernel |
| 3 | R-003 depth-flat 16-bit small | 221/231 / `half` 或 `bfloat16_t`, FP32, true | `front==0 && back==0 && dtype∈{FP16,BF16} && inputH*inputWAlign < ubSize/12` | R-004 |
| 4 | R-004 depth-flat 16-bit big | 222/232 / 同 dtype 模板 | 同上且 small 谓词为假 | 输入 guard 失败或 Kernel |
| 5 | R-005 3D FP32 small | 311 / `CircularPadGrad3D<float,float,false>` | `(front!=0 \|\| back!=0) && dtype=FP32 && small` | R-006 |
| 6 | R-006 3D FP32 big | 312 / 同模板 | `(front!=0 \|\| back!=0) && dtype=FP32 && !small` | 输入 guard 失败或 Kernel |
| 7 | R-007 3D 16-bit small | 321/331 / `half` 或 `bfloat16_t`, FP32, true | `(front!=0 \|\| back!=0) && dtype∈{FP16,BF16} && small` | R-008 |
| 8 | R-008 3D 16-bit big | 322/332 / 同 dtype 模板 | `(front!=0 \|\| back!=0) && dtype∈{FP16,BF16} && !small` | 输入 guard 失败或 Kernel |

```text
CheckDtype -> CalculateParams
if inputW >= usedUBSize/sizeof(T): fail
shapeType = (inputH*inputWAlign < usedUBSize/sizeof(T)) ? 1 : 2
dimensionBase = (front!=0 || back!=0) ? 300 : 200
dtypeDigit = FP32:1, FP16:2, BF16:3
key = dimensionBase + 10*dtypeDigit + shapeType
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 211 | A2/A3, FP32, depth-flat, small | 单 AIV | 2D float `ProcessSmallShape` | FP32 queue；H/W atomic fold | page tail core |
| R-002 | 212 | A2/A3, FP32, depth-flat, big | 单 AIV | 2D float `ProcessBigShape` | FP32 queue；按 rows 分块 | row tail + page tail |
| R-003 | 221/231 | A2/A3, FP16/BF16, depth-flat, small | 单 AIV | 2D cast `ProcessSmallShape` | T1↔FP32 queues；FP32 workspace | cast tail + page tail |
| R-004 | 222/232 | A2/A3, FP16/BF16, depth-flat, big | 单 AIV | 2D cast `ProcessBigShape` | rows 分块、FP32 atomic | row/cast/page tail |
| R-005 | 311 | A2/A3, FP32, depth-active, small | 单 AIV | 3D float `ProcessSmallShape` | H/W fold 后 D fold | batch-group tail |
| R-006 | 312 | A2/A3, FP32, depth-active, big | 单 AIV | 3D float `ProcessBigShape` | rows 分块 + D fold | row/batch tail |
| R-007 | 321/331 | A2/A3, FP16/BF16, depth-active, small | 单 AIV | 3D cast `ProcessSmallShape` | FP32 accumulation + D fold | cast/batch tail |
| R-008 | 322/332 | A2/A3, FP16/BF16, depth-active, big | 单 AIV | 3D cast `ProcessBigShape` | rows 分块、FP32 workspace | row/cast/batch tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 单行 UB | `inputW < usedUBSize/tSize` | big route 的 `rowsNum=T1UBSize/align(lineW)` 必须非零 | 全部 | Tiling 失败 | `CircularPadGradTiling::CheckInput` |
| small 严格小于 | `inputH*inputWAlign < usedUBSize/tSize` | small 一次把整页搬入一块 queue buffer | R-001/3/5/7 | 对应 big route | `DoOpTiling` |
| padding 重叠上界 | 双正侧合计 `<= 2*dim/3`；单正侧 `<= dim/2` | fold 源/目的区需保持合法、有限重叠 | 全部 | Tiling 失败 | `CheckLeftAndRight/CheckTopAndBottom/CheckFrontAndBack` |
| 输出 shape | `outputH=inputH-top-bottom` 等 | workspace 与最终搬运地址均依赖该恒等式 | 全部 | Tiling 失败 | `CheckInput` |
| DataCopy 窄化 | `inputH/inOutputH/rowsNum` 被转成 `uint16_t`，blockLen 被转成 `uint32_t` | AscendC `DataCopyExtParams` ABI | 全部 | 固定仓没有额外 fallback；受 API/shape 与 UB guard 限制 | `Calculate*Parms`, `CopyGMToGMLines*` |
| 负 padding 初始化 | 任一最后三维 padding `<0` | 最终输出可能有未覆盖位置 | 全部 | 先 `SetGMtoZero` | `Init2D/Init3D` |

### POD/ABI

`CircularPadCommonTilingData` 由仓外 `tilingdata_base.h` 宏生成；固定仓能证明字段声明顺序和每字段 `int64_t/64-bit`，但宏展开后的 `sizeof/alignof/offsetof` 不可见，因此不臆测数值 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| CircularPadCommonTilingData | inputH | int64_t/64 | 第1字段；精确 offset 固定仓不可见 | element rows | `GetShapeAttrsInfo`→`set_inputH` | `InitCommon`；H 循环 | >0 |
| 同上 | inputW | int64_t/64 | 第2字段；offset 不可见 | elements/row | `set_inputW` | alignment、DMA blockLen | `< usedUB/tSize` |
| 同上 | outputH | int64_t/64 | 第3字段；offset 不可见 | rows | `set_outputH` | `outputLen` | `inputH-top-bottom` |
| 同上 | outputW | int64_t/64 | 第4字段；offset 不可见 | elements/row | `set_outputW` | output stride/len | `inputW-left-right` |
| 同上 | left | int64_t/64 | 第5字段；offset 不可见 | elements | padding tensor | W fold/address | signed |
| 同上 | right | int64_t/64 | 第6字段；offset 不可见 | elements | padding tensor | W fold/address | signed |
| 同上 | top | int64_t/64 | 第7字段；offset 不可见 | rows | padding tensor | H fold/address | signed |
| 同上 | bottom | int64_t/64 | 第8字段；offset 不可见 | rows | padding tensor | H fold/address | signed |
| 同上 | front | int64_t/64 | 第9字段；offset 不可见 | pages | padding tensor | `Init3D/AddFrontAndBack`; 2D inactive | signed；也决定 key 2xx/3xx |
| 同上 | back | int64_t/64 | 第10字段；offset 不可见 | pages | padding tensor | `Init3D/AddFrontAndBack`; 2D inactive | signed |
| 同上 | inputL | int64_t/64 | 第11字段；offset 不可见 | pages/batch | gradOutput shape | 3D core mapping；2D inactive | depth-active 时非零 |
| 同上 | outputL | int64_t/64 | 第12字段；offset 不可见 | pages/batch | output shape | 3D output base；2D inactive | `inputL-front-back` |
| 同上 | perCoreTaskNum | int64_t/64 | 第13字段；offset 不可见 | pages | `DivCore` | core GM extent/loops | 3D 时为 inputL 的倍数 |
| 同上 | tailTaskNum | int64_t/64 | 第14字段；offset 不可见 | pages | `DivCore` | former/tail core mapping | 3D 时为 inputL 的倍数 |
| 同上 | workspaceLen | int64_t/64 | 第15字段；offset 不可见 | FP32 elements/task | `DoOpTiling` | workspace page stride | small/big 公式不同 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～R-004 | 211/212/221/222/231/232 | arch32 `DoOpTiling/SetTilingKey` | CircularPadCommonTilingData | `circular_pad_grad` 2D branches | `CircularPadGrad2D` + base/common | CMake arch32 + A2/A3 binary json |
| R-005～R-008 | 311/312/321/322/331/332 | 同上，`dPad=true` | 同 POD，含 L/front/back | `circular_pad_grad` 3D branches | `CircularPadGrad3D` + base/common | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 只有一个 Device 节点。entry 取 POD、建立 `TPipe`，按 key 实例化 2D/3D 与 dtype 模板。各 body 先把主体 H×W 区写入 task 独占 FP32 workspace，再以 `SetAtomicAdd<T2>()` 折叠正 padding 造成的重叠；`MTE3_MTE2` event 保证 workspace 写回在下一轮读前可见。FP16/BF16 入 workspace 前 `CAST_NONE` 到 FP32，最终以 `CAST_ROUND` 降精度。small 将整页放入 UB；big 用 `rowsNum` 循环和 row tail。3D body 在 H/W 后再折叠 front/back。未知 key 直接 return，不写输出；Host 只产生列出的 12 个 key。

### Route R-001：depth-flat FP32 small

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `circular_pad_grad_tiling.cpp` | `DoOpTiling` | FP32，`front==back==0`，整页小于 UB half | key 211 |
| Router | `circular_pad_grad.cpp` | key 211 branch | `CircularPadGrad2D<float,float,false>` | `ProcessSmallShape` |

#### Device 数据流

##### 函数级调用链

```text
circular_pad_grad -> Init2D -> Init -> InitCommon
  -> AddTopAndBottomSmallShapeNoCast -> MTE3ToMTE2Sync
  -> AddLeftAndRightSmallShape -> MTE3ToMTE2Sync
  -> CopyToOutSmallShape -> CopyToOutSmallShapeOnePage
```

##### 源码等价伪码

```text
for task in core_pages:
  load full input HxalignedW to FP32 UB
  copy interior H rows to workspace; atomic-add wrapped top/bottom rows
  load wrapped left/right columns; atomic-add them to workspace
  copy valid interior rectangle workspace[Align+pLeft] -> y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xGM | `inputLen*startIdx` | `inputLen*perCoreTaskNum` | FP32 ND | API | page loader | caller | row 32B pad |
| queBindT2 | 0 | depth 2 × 98,304 B | FP32 | MTE2 | MTE3 | TPipe | ping-pong queue |
| workspaceGM | `workspaceLen*startIdx` | `workspaceLen*perCoreTaskNum` FP32 | padded rows | H/W fold | final copy | executor | task-disjoint |
| yGM | `outputLen*startIdx` | `outputLen*perCoreTaskNum` | FP32 ND | final copy | ViewCopy | executor temp | valid rectangle only |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBindT2 slot0 · 98,304 B"]:1
  q1["queBindT2 slot1 · 98,304 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * (192 KiB / 2) = 192 KiB
workspace_total = totalTasks * workspaceLen * 4 + sysWorkspaceSize
dynamic_ub = N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queBindT2 slot | DataCopyPad | workspace/y DataCopyPad | next queue allocation | EnQue/DeQue/FreeTensor | consumer 完成后释放 |
| workspace task slice | H fold | W fold/final copy | Kernel 结束后 | MTE3→MTE2 event | task slices 不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| H→W→out | MTE3_MTE2 SetFlag/WaitFlag | workspace store | 下一次 workspace load | FP32 task slice | 无 cast；tail core 多 1 page |
| wrap fold | AtomicAdd FP32 GM | top/bottom/left/right store | 后续 stage | workspace overlaps | 正 padding才执行 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp` | `DoOpTiling` | small/FP32 资源条件 |
| Kernel entry/router | `conversion/circular_pad_grad/op_kernel/circular_pad_grad.cpp` | key 211 | 映射到 2D float small |
| Device body/helper | `conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h` | `Add*SmallShape*` | 搬运、atomic、tail |

### Route R-002：depth-flat FP32 big

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `circular_pad_grad_tiling.cpp` | `DoOpTiling` | R-001 small 谓词为假但单行 guard 成立 | key 212 |
| Router | `circular_pad_grad.cpp` | key 212 | 2D float | `ProcessBigShape` |

#### Device 数据流

##### 函数级调用链

```text
circular_pad_grad -> Init2D -> ProcessBigShape
  -> AddTopAndBottomBigShape -> CopyGMToGMLines -> sync
  -> AddLeftAndRightBigShape -> AddLeftAndRightLines -> sync
  -> CopyToOutBigShapeBig -> CopyGMToGMLinesOnce
```

##### 源码等价伪码

```text
rowsNum = 98,304 / align32(inputW*4)
for task: for full row chunk and row tail: x -> UB -> workspace
atomic-add top/bottom; then chunk left/right columns and atomic-add
for output page: chunk inOutputH rows workspace -> y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| queBindT2 | 0 | depth 2 × 98,304 B | FP32 row chunks | x/workspace load | workspace/y store | TPipe | rowsNum + tail |
| workspace | task×`inOutputH*(inputWAlign+16)` | per-task extent | FP32 | chunk stores | folds/final | executor | 8 FP32 guard elems each side |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBindT2 slot0 · 98,304 B"]:1
  q1["queBindT2 slot1 · 98,304 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB
workspace_total = totalTasks * inOutputH * (inputWAlign+16) * 4 + sysWorkspaceSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | row chunk load | row chunk store | next chunk | FreeTensor | `rowsNum>0` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| chunk loop | queue + MTE event | each chunk store | fold/final load | workspace | `tail=lines%rowsNum` |
| fold | FP32 AtomicAdd | wrap chunks | final copy | overlap | no cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp` | `DoOpTiling` | shapeType=2 |
| Kernel entry/router | `conversion/circular_pad_grad/op_kernel/circular_pad_grad.cpp` | key 212 | big float body |
| Device body/helper | `conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h` | `CopyGMToGMLines*` | rowsNum/loop/tail |

### Route R-003：depth-flat FP16/BF16 small

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch32 tiling | `CheckDtype/DoOpTiling` | dtype mode 2/3，depth-flat，small | 221/231 |
| Router | kernel entry | key 221/231 | half/bfloat16_t, float, true | 2D small cast |

#### Device 数据流

##### 函数级调用链

```text
Init2D -> AddTopAndBottomSmallShapeCast(Cast T1->FP32)
 -> sync -> AddLeftAndRightSmallShape(FP32 atomic)
 -> sync -> CopyToOutSmallShapeOnePage(Cast FP32->T1)
```

##### 源码等价伪码

```text
load full T1 page; CAST_NONE to FP32
store/fold H and W in FP32 workspace
load valid FP32 rectangle; CAST_ROUND to FP16 or BF16; store y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| queBindT1 | 0 | depth 2 × 32,768 B | FP16/BF16 | MTE2/Cast out | Cast/MTE3 | TPipe | T1 path |
| queBindT2 | after T1 region | depth 2 × 65,536 B | FP32 | Cast/workspace load | atomic/store/Cast | TPipe | FP32 accumulation |
| workspace | task slice | `workspaceLen*4` | FP32 | cast/fold | final cast | executor | dtype-independent |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  t10["T1 slot0 · 32 KiB"]:1
  t11["T1 slot1 · 32 KiB"]:1
  t20["FP32 slot0 · 64 KiB"]:2
  t21["FP32 slot1 · 64 KiB"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*32 KiB + 2*64 KiB = 192 KiB
workspace_total = totalTasks * workspaceLen * 4 + sysWorkspaceSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| T1/T2 queue slots | load/Cast | Cast/store | next task | typed EnQue/DeQue/FreeTensor | each consumer releases |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| ingress | `Cast(...CAST_NONE)` | T1 input | FP32 workspace | values | FP32 fold |
| egress | `Cast(...CAST_ROUND)` | FP32 workspace | T1 output | output | cast count includes aligned row/stride span |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch32 tiling | dtype modes 2/3 | same 2-byte UB predicate |
| Kernel entry/router | kernel entry | 221/231 | half/BF16 cast templates |
| Device body/helper | base body | `AddTopAndBottomSmallShapeCast`, `CopyGMToGMOnceCastFromFP32` | FP32 accumulation order |

### Route R-004：depth-flat FP16/BF16 big

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch32 tiling | `DoOpTiling` | 16-bit、depth-flat、!small | 222/232 |
| Router | kernel entry | key 222/232 | cast template | 2D big |

#### Device 数据流

##### 函数级调用链

```text
Init2D -> ProcessBigShape -> chunked T1->FP32 H fold
 -> chunked FP32 W fold -> chunked FP32->T1 output
```

##### 源码等价伪码

```text
for task and row chunk: load T1; cast FP32; store workspace
fold top/bottom and left/right with FP32 atomic adds
for row chunk/tail: load workspace; cast_round T1; store y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| T1 queues | 0 | 2×32 KiB | FP16/BF16 chunks | GM/Cast | Cast/GM | TPipe | row tail |
| FP32 queues | 64 KiB region | 2×64 KiB | FP32 chunks | Cast/GM | GM/Cast | TPipe | row tail |
| workspace | task stride | `inOutputH*(inputWAlign+16)*4` | FP32 | chunks/folds | output cast | executor | task-disjoint |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  t10["T1 slot0 · 32 KiB"]:1
  t11["T1 slot1 · 32 KiB"]:1
  t20["FP32 slot0 · 64 KiB"]:2
  t21["FP32 slot1 · 64 KiB"]:2
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB
workspace_total = totalTasks * inOutputH * (inputWAlign+16) * 4 + sysWorkspaceSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| T1/T2 slots | row chunk | Cast/GM | next chunk | FreeTensor | `rowsNum>0` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| row chunks | queues/events | input/fold | next stage | UB/workspace | full chunks + row tail |
| precision | CAST_NONE/CAST_ROUND | T1↔FP32 | workspace/output | accumulation | one final down-cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch32 tiling | shapeType 2 modes 2/3 | keys 222/232 |
| Kernel entry/router | kernel entry | 222/232 | big cast body |
| Device body/helper | base body | `CopyGMToGMLines*` | cast chunk loops/tails |

### Route R-005：3D FP32 small

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | common tiling | `CalculateParams/SetTilingKey` | front/back 任一非零、FP32、small | 311 |
| Router | kernel entry | key 311 | 3D float | `ProcessSmallShape` |

#### Device 数据流

##### 函数级调用链

```text
Init3D -> H fold -> sync -> W fold -> sync
 -> AddFrontAndBack(CopyGmToGm with FP32 atomic) -> sync
 -> per batch/page CopyToOutSmallShapeOnePage
```

##### 源码等价伪码

```text
assign whole depth groups to one core
perform small H/W fold for every input depth page
if front>0: atomic-add first front pages to wrapped rear pages
if back>0: atomic-add last back pages to wrapped front pages
copy pages [pFront,inputL-pBack) to mapped output depth
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| core x slice | `inputLen*startIdx` | whole depth groups | FP32 | caller | H fold | caller | group-disjoint |
| queue | 0 | 2×98,304 B | FP32 | GM loads | GM stores | TPipe | reused across H/W/D |
| workspace | `workspaceLen*startIdx` | whole depth groups | FP32 padded pages | H/W fold | D fold/output | executor | batch group isolated |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBindT2 slot0 · 98,304 B"]:1
  q1["queBindT2 slot1 · 98,304 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB
workspace_total = totalTasks * workspaceLen * 4 + sysWorkspaceSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| workspace depth page | H/W fold | D fold/final output | Kernel end | MTE3→MTE2 between stages | all pages of batch remain same core |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| W→D | MTE3_MTE2 | W fold | D reads | workspace pages | FP32 |
| D fold | FP32 AtomicAdd | front/back source pages | final output | wrapped page | former cores receive one extra depth group |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | common tiling | `DivCore/SetTilingKey` | group-preserving key 311 |
| Kernel entry/router | kernel entry | 311 | 3D float small |
| Device body/helper | `circular_pad_grad_3d.h` | `AddFrontAndBack/getNewBatchOffset` | D atomic fold/mapping |

### Route R-006：3D FP32 big

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch32/common | depth-active, FP32, !small | key 312 | big body |
| Router | kernel entry | key 312 | 3D float | `ProcessBigShape` |

#### Device 数据流

##### 函数级调用链

```text
Init3D -> chunked H/W fold -> AddFrontAndBack
 -> CopyToOutBigShapeBig -> CopyToOutBigShapeOnePage -> CopyGMToGMLinesOnce
```

##### 源码等价伪码

```text
same depth-group mapping as R-005
H/W operate by rowsNum chunks and row tail
D wraps full workspace pages by UB-sized chunks and byte tail
final output also chunks rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| queue | 0 | 2×98,304 B | FP32 chunks | H/W/D GM loads | GM stores | TPipe | row/page chunks |
| workspace | depth page stride | `inOutputH*(inputWAlign+16)` FP32/page | padded page | H/W | D/output | executor | group-disjoint |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  q0["queBindT2 slot0 · 98,304 B"]:1
  q1["queBindT2 slot1 · 98,304 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB
workspace_total = totalTasks * workspaceLen * 4 + sysWorkspaceSize
D_chunk_loop = floor(workspaceLen*pages*4 / 98,304), byte_tail = remainder
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queue slot | H/W/D chunk load | corresponding store | next chunk | queue FreeTensor | byte tail blockLen corrected |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| D chunks | AtomicAdd + MTE event | wrapped page chunks | output copy | workspace | FP32; byte tail handled |
| H/out rows | queue loop | row chunks | next stage | UB/workspace | row tail handled |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch32 tiling | shapeType=2 | workspaceLen big |
| Kernel entry/router | kernel entry | 312 | 3D float big |
| Device body/helper | base + 3D | `CopyGMToGMLines*`, `CopyGmToGm` | two tail kinds |

### Route R-007：3D FP16/BF16 small

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch32/common | depth-active、mode 2/3、small | 321/331 | 16-bit 3D small |
| Router | kernel entry | 321/331 | `half/bfloat16_t,float,true` | `ProcessSmallShape` |

#### Device 数据流

##### 函数级调用链

```text
Init3D -> T1->FP32 H fold -> FP32 W fold -> FP32 D fold
 -> per output page FP32->T1 CopyToOutSmallShapeOnePage
```

##### 源码等价伪码

```text
for each depth group: cast each full page to FP32 once
perform all circular additions in FP32 workspace across H, W and D
cast each valid output page once to FP16/BF16
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| T1 queues | 0 | 2×32 KiB | FP16/BF16 | input/final cast | Cast/GM | TPipe | depth pages |
| FP32 queues | 64 KiB region | 2×64 KiB | FP32 | cast/workspace load | fold/cast | TPipe | H/W/D shared |
| workspace | whole depth groups | FP32 padded pages | FP32 | cast/H/W/D | final cast | executor | group-disjoint |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  t10["T1 slot0 · 32 KiB"]:1
  t11["T1 slot1 · 32 KiB"]:1
  t20["FP32 slot0 · 64 KiB"]:2
  t21["FP32 slot1 · 64 KiB"]:2
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB
workspace_total = totalTasks * workspaceLen * 4 + sysWorkspaceSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 workspace page | input cast/H/W | D fold/final cast | end | MTE event | one core owns full depth group |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all folds | FP32 AtomicAdd | H/W/D wraps | final cast | workspace | one down-cast after all sums |
| core tail | depth group distribution | host DivCore | Init3D mapping | GM slices | group not split |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | common tiling | dPad/DivCore | keys 321/331 |
| Kernel entry/router | entry | 321/331 | 3D cast templates |
| Device body/helper | base + 3D | cast helpers/AddFrontAndBack | FP32 full-pipeline accumulation |

### Route R-008：3D FP16/BF16 big

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | arch32/common | depth-active、16-bit、!small | 322/332 | 3D big cast |
| Router | entry | 322/332 | 3D cast template | `ProcessBigShape` |

#### Device 数据流

##### 函数级调用链

```text
Init3D -> chunked T1->FP32 H fold -> chunked FP32 W fold
 -> chunked FP32 D fold -> chunked FP32->T1 output
```

##### 源码等价伪码

```text
for each core-owned depth group and each row chunk:
  cast input to FP32 workspace; fold H/W
for front/back page chunks: atomic-add FP32 workspace
for each retained depth page and row chunk: cast_round to T1 output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| T1 queue | 0 | 2×32 KiB | FP16/BF16 chunks | GM/Cast | Cast/GM | TPipe | row tail |
| FP32 queue | 64 KiB region | 2×64 KiB | FP32 chunks | Cast/GM | fold/Cast | TPipe | row/page tail |
| workspace | depth page/task | big workspaceLen FP32 | padded | H/W/D | output | executor | same core owns group |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  t10["T1 slot0 · 32 KiB"]:1
  t11["T1 slot1 · 32 KiB"]:1
  t20["FP32 slot0 · 64 KiB"]:2
  t21["FP32 slot1 · 64 KiB"]:2
```

##### 字节账与复用顺序

```text
UB_total = 192 KiB
workspace_total = totalTasks * inOutputH * (inputWAlign+16) * 4 + sysWorkspaceSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| queues | row/page chunk load/Cast | GM/Cast store | next chunk | FreeTensor + stage events | row/page tail blockLen exact |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| stage order | MTE3_MTE2 | each fold | next fold/output | workspace | H row tail + D byte tail |
| precision | CAST_NONE, FP32 AtomicAdd, CAST_ROUND | input | output | sum | only final down-cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch32/common | mode 2/3 shapeType2 dPad | 322/332 |
| Kernel entry/router | entry | 322/332 | 3D big cast |
| Device body/helper | base + 3D | chunk/cast/D fold helpers | complete big pipeline |

## 算子特定优化

### Trick T-001：16-bit 输入只在边界转换，全部 circular 合并驻留 FP32 workspace

#### 触发与源码依据

- 适用 Route：R-003、R-004、R-007、R-008
- 触发条件：dtype 为 FP16 或 BF16，router 实例化 `ISCAST=true`
- Golden 来源：[`conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h:206-240`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h#L206-L240)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void AddTopAndBottomSmallShapeCast(sDataCopyExtParams& params)
{
    auto blockCount = params.paramsOut.blockCount;
    for (int64_t i = 0; i < perCoreTaskNum_; i++) {
        auto inLocal = queBindT1_.AllocTensor<T1>();
        DataCopyPad(inLocal, xGM_[i * inputLen_], params.paramsIn, padParmsT1);
        queBindT1_.EnQue<QuePosition::GM, QuePosition::VECIN, T1>(inLocal);
        inLocal = queBindT1_.DeQue<QuePosition::GM, QuePosition::VECIN, T1>();
        auto inLocalT2 = queBindT2_.AllocTensor<T2>();
        Cast(inLocalT2, inLocal, RoundMode::CAST_NONE, inputH_ * inputWAlign_);
        queBindT2_.EnQue<QuePosition::VECOUT, QuePosition::GM, T2>(inLocalT2);
```

#### 机制与设计原因

T1 queue 只承担 16-bit GM 边界，T2 queue/workspace 始终为 FP32。H/W/D 的所有 `AtomicAdd<T2>` 因此在 FP32 完成，最终输出路径才 `CAST_ROUND` 回 T1；这既避免每个 fold 后反复降精度，也使 FP16 与 BF16 共用相同的 FP32 workspace ABI。代价是 queue 按 `2×32 KiB + 2×64 KiB` 划分，Host small guard必须按更小 T1 slot 计算。

#### 迁移边界

- 适用：多个搬运/散加阶段共享同一中间张量的 16-bit backward 算子。
- 前提：A2/A3、FP16/BF16、FP32 workspace 与 AtomicAdd 可用，且 UB 能容纳双 queue。
- 失效：FP32 使用 R-001/2/5/6 的无 cast 模板；若 UB small 条件不成立切 big Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不能把中间 workspace 改成 T1，否则会同时改变 atomic dtype、workspace 字节公式和数值舍入顺序。

### Trick T-002：按完整 depth group 分核，消除跨核 circular-D 竞争

#### 触发与源码依据

- 适用 Route：R-005～R-008
- 触发条件：`front!=0 || back!=0`
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

3D Route 的最小调度单元不是一页，而是同一 batch 的完整 `inputL` 页。于是 front/back 的 source 与 wrapped destination 总在同一核拥有的 workspace slice 内，D 维合并不需要跨核同步；`tailTaskNum` 也保持 `inputL` 的整数倍。depth-flat Route 则退化为逐页均分，提高可用核数。

#### 迁移边界

- 适用：跨页变换需要在 D 维首尾互相读取/累加的单 Kernel 算子。
- 前提：workspace 按 page 连续、一个 group 能由单核串行处理、Host 能把 blockDim 降到 batch group 数。
- 失效：front/back 都为 0 时使用 R-001～R-004 的逐页分核。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若仍按单页 round-robin 分核，`AddFrontAndBack` 会访问别核拥有的 page slice，当前源码的仅核内 MTE event 无法提供跨核 ordering。

## 模板与实现对比

| 对比维度 | R-001/2 FP32 depth-flat | R-003/4 16-bit depth-flat | R-005/6 FP32 3D | R-007/8 16-bit 3D | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | front/back=0 | 同左，16-bit | front/back 非全0 | 同左，16-bit | dPad 决定 2xx/3xx；shapeType 决定 small/big |
| 模板与 dtype | float,float,false | half/BF16,float,true | 3D float | 3D cast | 16-bit 全程 FP32 累加 |
| 循环与分核 | page | page | depth group | depth group | D fold不跨核 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2×96 KiB | 2×32+2×64 KiB | 同 FP32 | 同 cast | 无 dynamic UB/通信 window；寄存器字节固定仓不可见 |
| 内存总占用与复用 | 192 KiB | 192 KiB | 192 KiB + D pages | 192 KiB + FP32 D pages | workspace 均按 FP32 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Atomic | +Cast | +D chunk copy | +Cast+D copy | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | MTE event，FP32 | 最终 cast | +D byte tail | row/D/cast tail | stage 间显式 MTE3→MTE2 |
| fallback/失败边界 | big 或 tiling失败 | big 或失败 | big 或失败 | big 或失败 | 无 AiCPU fallback；unknown key return |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 | 支持 | 未发布：固定仓公开 README 与专用 API 文档冲突 |
| Host/Launcher | ascend910b，arch32 | ascend910_93，arch32 | arch35/RegBase 实现信号存在，但不能越过公开冲突 |
| Kernel/资源 | 共用 arch32 entry/body 与 192 KiB UB 常量 | 同左 | 本卡不描述；未生成 `_950` 卡 |

## 限制与失败边界

- 公开接口要求 rank 4/5、padding 长度 6、三张量 dtype 和 view format 一致、`self.shape==gradInput.shape`，且 gradOutput 最后三维严格满足 padding shape 关系。
- API 只检查每个 padding `< self_dim`，Tiling 进一步限制正 padding 的两侧合计或单侧比例；两层任一失败都不 launch。
- A2/A3 Tiling 对最后一维有 UB guard；公开 API 文档另给 FP16/BF16 `(0,16362)`、FP32 `(0,24544)`。应遵守公开约束而不是只依赖更靠后的 Tiling guard。
- `DataCopyExtParams` 中多处 `uint16_t/uint32_t` 窄化没有独立 overflow fallback；固定仓只证明现有 API、padding、UB guard，不能外推更大 shape。
- 负 padding 时 Device 先清零输出；若移植时删除该步骤，未被有效 rectangle 覆盖的区域会保留旧值。
- ACLNN executor 内框架 Contiguous/ViewCopy 的具体 Kernel、workspace offset、event 和实现源码不在固定算子仓；它们按 Skill 允许的框架边界处理。
- Tiling 宏定义来自仓外头文件，POD 的精确 `sizeof/alignof/offsetof` 固定仓不可见。
- 950 的专用 API 文档写支持，而同目录公开 README 明确写不支持；依照平台冲突 gate 未创建/覆盖 950 正式卡。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/circular_pad_grad/README.md` | 产品支持情况 | A2/A3 支持与 950 公开冲突证据 |
| 入口声明/约束 | `conversion/circular_pad_grad/docs/aclnnCircularPad3dBackward.md` | public support/constraints | A2/A3 支持、接口约束 |
| 入口声明/约束 | `conversion/circular_pad_grad/op_api/aclnn_circular_pad3d_backward.h` | 两段式声明 | 参数 ABI |
| 入口实现 | `conversion/circular_pad_grad/op_api/aclnn_circular_pad3d_backward.cpp` | `CommonPad3dBackward` | 校验、preprocess、executor graph |
| Launcher selector | `conversion/pad_v3_grad/op_api/padv3grad.cpp` | `PadV3Grad`, `CircularPadGradAiCore` | circular 到核心 op |
| shape producer | `conversion/pad_v3_grad/op_host/pad_v3_grad_infershape.cpp` | `PadV3GradInfershape` | 输出 shape |
| Host/Tiling | `conversion/circular_pad_grad/op_host/arch32/circular_pad_grad_tiling.cpp` | `DoOpTiling` | dtype、shapeType、guard |
| POD/Tiling common | `conversion/circular_pad/op_host/circular_pad_common_tiling.h` | `CircularPadCommonTilingData` | POD 字段 |
| POD/Tiling common | `conversion/circular_pad/op_host/circular_pad_common_tiling.cpp` | `DivCore/SetTilingKey/SetTilingData` | key、core、workspace |
| Kernel entry/body | `conversion/circular_pad_grad/op_kernel/circular_pad_grad.cpp` | `circular_pad_grad` | 12 key router |
| Kernel entry/body | `conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad.h` | `CircularPadGrad` | H/W fold、Cast、copy tails |
| Kernel entry/body | `conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad_2d.h` | `CircularPadGrad2D` | depth-flat internal route |
| Kernel entry/body | `conversion/circular_pad_grad/op_kernel/arch32/circular_pad_grad_3d.h` | `CircularPadGrad3D` | D fold/group mapping |
| Kernel helper | `conversion/circular_pad/op_kernel/circular_pad_common.h` | `InitCommon/MTE3ToMTE2Sync` | UB 常量、event |
| 平台注册 | `conversion/circular_pad_grad/op_host/circular_pad_grad_def.cpp` | `CircularPadGrad` | dtype/config |
| 构建接线 | `conversion/circular_pad_grad/op_host/config/ascend910b/circular_pad_grad_binary.json` | CircularPadGrad binaries | A2 dtype binaries |
| 构建接线 | `conversion/circular_pad_grad/op_host/config/ascend910_93/circular_pad_grad_binary.json` | CircularPadGrad binaries | A3 dtype binaries |
| 构建接线 | `conversion/circular_pad_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | A2/A3→arch32 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15 个 int64 字段、Host/Device 共用类 | 字段顺序、单位、front/back 对 key 的作用 | `CircularPadCommonTilingData` |
| 调用顺序 | H fold→W fold→D fold→输出 | 每阶段间 MTE3→MTE2；3D group 不跨核 | `Process*Shape` |
| workspace/内存 | 每 task 一个 FP32 padded page，系统区在前 | `workspaceLen` 单位 FP32 元素、task slice 不重叠 | `GetWorkspaceSize`, `Init2D/3D` |
| 同步 | queue 生命周期、MTE event、GM AtomicAdd | AtomicNone 恢复；写后读必须 wait | `MTE3ToMTE2Sync`, `Add*` |
| 精度与尾块 | 16-bit→FP32→一次 down-cast；row/page/core tails | `CAST_NONE/CAST_ROUND` 顺序和 tail blockLen | cast/copy helpers |
| 失败边界 | Host/Tiling guard 失败即不 launch；unknown key return | 不添加未经源码证明的 fallback | `CheckInput`, router `else` |