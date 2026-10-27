---
type: Operator
title: "Celu 950 ACLNN 算子知识"
description: "说明 aclnnCelu 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/docs/aclnnCelu&aclnnInplaceCelu.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_api/aclnn_celu.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_api/aclnn_celu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_api/celu_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_host/celu_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/celu_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T14:57:58Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Celu 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCeluGetWorkspaceSize` + `aclnnCelu`，公开两段式 ACLNN；本文不包含 `aclnnInplaceCelu`。 |
| 计算语义 | 对每个输入元素计算 `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`。 |
| 输入输出 | `self`/`out` 为 0～8 维 ND Tensor，公开支持 FLOAT/FLOAT16/BFLOAT16，shape 相同；输入输出均允许非连续。 |
| 执行类型 | 算子自有节点 `CeluV2` 为单 Kernel `AIV_VECTOR`；ACLNN 外围的 `Contiguous`、`Cast`、`ViewCopy` 是框架前后处理边界。 |
| Kernel 路线 | R-001：`ELEMENTWISE_TPL_SCH_MODE_0` 单一逐元素 Vector 路线。 |
| 核心资源 | GM 输入/输出；双缓冲 VECIN/VECOUT `TQue`；两个 FP32 `TBuf`；系统 workspace `512 + GetLibApiWorkSpaceSize()` 字节。 |
| 来源闭包 | `source_closed`：公开 ACLNN、参数检查、executor 编排、CeluV2 launcher、950 Tiling/POD、Kernel entry/body 与 950 构建接线闭合；框架通用节点按框架边界标注。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入逻辑元素总数为 `N`，对 `i in [0, N)`：

```text
z_i = x_i / alpha
neg_i = min(0, alpha * (exp(z_i) - 1))
pos_i = max(0, x_i)
y_i = pos_i + neg_i
```

公开 ACLNN 在第一段接口中拒绝 `alpha == 0` 和 NaN（比较结果既不大于 0 也不小于 0）；正负有限非零值及正负无穷可越过该层检查。950 Tiling 还有更严格的 `fabs(alpha) >= 1e-7` 条件。空 Tensor 不启动 `CeluV2`，直接返回 workspace 0。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，0～8 维；允许非连续 | FLOAT、FLOAT16、BFLOAT16 | 必填；逐元素公式的 `x`。950/Regbase 走包含 BFLOAT16 的支持表。 |
| `alpha` | Host scalar | 可转换为 FLOAT | 必填 ACL scalar；传给 L0 前经 `ToFloat()`；不能为 0 或 NaN，且 950 Tiling 拒绝绝对值小于 `1e-7`。 |
| `out` | 与 `self` shape 相同；允许非连续 | FLOAT、FLOAT16、BFLOAT16 | 必填；dtype 必须能由 `self` dtype 转换；最终由 `Cast` 和 `ViewCopy` 写入。 |
| `workspaceSize` | 标量指针 | `uint64_t*` | 第一段接口输出整个 executor 图需要的 Device workspace 字节数。 |
| `executor` | 不适用 | `aclOpExecutor**` | 第一段接口输出并拥有已编排计算图；交给第二段接口执行。 |
| `workspace` | Device 地址 | `void*` | 第二段接口输入；大小必须使用同一次第一段调用返回值。 |
| `stream` | ACL stream | `aclrtStream` | 第二段接口在该 stream 上运行 executor。 |

### 入口与版本

公开入口是 `aclnnCeluGetWorkspaceSize` 与 `aclnnCelu`，API 文档明确 950（Ascend 950PR/950DT）支持。第一段调用 `GetWorkspaceSizeCommon` 创建 executor、完成检查和图编排；第二段只调用 `CommonOpExecutorRun`。L0 节点是 `l0op::CeluV2`，它通过 `ADD_TO_LAUNCHER_LIST_AICORE(CeluV2, ...)` 注册算子自有 Kernel。

固定仓没有另一个更高版本号的公开 `aclnnCelu`。同文件中的 `aclnnInplaceCelu` 共享 `GetWorkspaceSizeCommon`，但它是原地语义变体，不属于本文。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnCeluGetWorkspaceSize` 的顺序是：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize` 与 `executor` 输出指针。
2. `CheckParams` 依次检查 `self/alpha/out` 非空、950 dtype、`alpha` 能否转换为 FLOAT、`self -> out` dtype 可转换、`self` rank 不超过 8、两端 shape 相同、`alpha` 既大于 0 或小于 0。
3. `self->IsEmpty()` 时把 `workspaceSize` 置 0、释放 executor 所有权并成功返回；后续节点都不加入图。
4. 非空输入调用框架 `l0op::Contiguous(self)`，产生连续输入或复用框架可接受的连续表示。
5. 调用 `l0op::CeluV2(contiguousSelf, alpha->ToFloat())`；L0 按输入 view shape/dtype 分配 ND 中间输出并加入 AICore launcher list。
6. 调用框架 `l0op::Cast(celuV2Out, out->GetDataType())`。
7. 调用框架 `l0op::ViewCopy(castOut, out)`，处理可能非连续的目标 view。
8. `uniqueExecutor->GetWorkspaceSize()` 汇总整个图 workspace，再把 executor 所有权交给调用者。

任一步返回空 Tensor 或 launcher 注册失败都会立即返回对应内部错误，不继续添加后续节点。

### 参数、Workspace 与生命周期

`self`、中间 Tensor 与 `out` 都由 executor 图持有到第二段完成。`CeluV2TilingFunc` 为算子节点设置一个 workspace slice：

```text
celu_node_workspace = 512 + PlatformAscendC::GetLibApiWorkSpaceSize()
```

Kernel entry 虽接收 `workspace`，`celu_v2` body 没有解引用它；该 slice 是系统/库 API 预留。整个 ACLNN 图还可能因 `Contiguous`、`Cast`、`ViewCopy` 产生 workspace，其精确 slice/offset 由通用 executor 实现决定，固定算子仓不可见；公开边界只保证 `GetWorkspaceSize()` 返回总量。第二段 `CommonOpExecutorRun` 使用调用者提供的同等大小 workspace 与 stream。

### 启动图与失败传播

非空路线的 executor 编排如下；框架节点可能按连续性或 dtype 等价性退化为 view/复用，具体消除规则属于固定仓不可见的框架边界：

```text
self
  -> F0 l0op::Contiguous
  -> K0 l0op::CeluV2 / celu_v2<schMode=0> [AIV_VECTOR]
  -> F1 l0op::Cast(to out dtype)
  -> F2 l0op::ViewCopy(to out view)
  -> out
```

| 节点 | Host 添加函数 / entry | 网格与参数 | 输入输出 / workspace | ordering、owner 与失败传播 |
|---|---|---|---|---|
| F0 | `GetWorkspaceSizeCommon -> l0op::Contiguous` | 框架决定 | `self -> contiguousSelf`；slice 固定仓不可见 | 同一 executor/stream 的前驱；空返回阻断 K0。 |
| K0 | `l0op::CeluV2 -> ADD_TO_LAUNCHER_LIST_AICORE` / `celu_v2<0>` | `blockDim=finalCoreNum`；POD=`CeluV2TilingData` | 连续 GM x -> GM y；节点 workspace=`512+libApiWorkspace`，body 不读取 | executor 按依赖在 F0 后启动；注册或运行失败阻断 F1/F2。 |
| F1 | `GetWorkspaceSizeCommon -> l0op::Cast` | 框架决定 | K0 中间输出 -> `out` dtype 中间；slice 固定仓不可见 | 同一 executor/stream 依赖 K0；空返回阻断 F2。 |
| F2 | `GetWorkspaceSizeCommon -> l0op::ViewCopy` | 框架决定 | Cast 中间 -> 用户 `out` view；slice 固定仓不可见 | 图尾节点；空返回使第一段失败，执行错误由 `CommonOpExecutorRun` 返回。 |

空 Tensor 路线没有 Device 节点、workspace 为 0。固定仓没有显式 event；跨节点可见性由 executor 在单个调用 stream 上维护，具体 event/任务协议不可见。

## Route 与参数设计

### 分核与资源计算

对非空输入，Tiling 先读取 AIV 核数 `C = GetCoreNumAiv()` 与 UB 大小 `U`；二者为 0 都失败。令 `s = sizeof(dtype)`、`N = GetStorageShape().GetShapeSize()`：

```text
inputLengthBytes = N * s
blocksTotal = ceil(inputLengthBytes / 32)
finalCoreNum = min(C, blocksTotal)                  # 至少 1
q = floor(blocksTotal / finalCoreNum)
r = blocksTotal mod finalCoreNum                   # tailBlockNum

ubDataNumber = 4, s <= 2                           # 2 个 TQue×depth2 + 两个 FP32 TBuf 的折算
             = 3, s == 4
tileBlockNum = max(1, floor(floor(U / 32 / 2) / ubDataNumber))
tileDataNum = max(1, tileBlockNum * 32 / s)

smallCoreDataNum = q * 32 / s
bigCoreDataNum = (q + 1) * 32 / s
finalSmallTileNum = ceil(q / tileBlockNum)
finalBigTileNum = ceil((q + 1) / tileBlockNum)
smallTailDataNum = smallCoreDataNum - tileDataNum * floor(q / tileBlockNum)
bigTailDataNum = bigCoreDataNum - tileDataNum * floor((q + 1) / tileBlockNum)
```

若尾元素表达式不大于 0，Host 把对应 `TailDataNum` 设为完整 `tileDataNum`。前 `r` 个 block 是大核，每核覆盖 `q+1` 个 32-byte block；其余是小核，每核覆盖 `q` 个 block。Kernel 的 GM 起点对大核为 `bigCoreDataNum * blockIdx`；小核再减去已跨过的大小核差额。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 950 通用逐元素路线 | `GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0)`；`celu_v2<schMode=0>` | ACLNN 非空；dtype 为 FLOAT/FLOAT16/BFLOAT16；AIV core/UB 非零；`fabs(alpha) >= 1e-7`；type length 非零。无其他 Host 可达 key。 | 任一检查失败则 Tiling/ACLNN 返回失败；无 fallback。 |

```text
aclnnCeluGetWorkspaceSize
  if 参数非法: 返回 ACLNN 错误
  if self.empty: workspace=0，成功，无 Kernel
  else:
    编排 Contiguous -> CeluV2 -> Cast -> ViewCopy
    CeluV2TilingFunc
      if AIV核数==0 or UB==0 or dtype非法 or |alpha|<1e-7 or typeLength==0: GRAPH_FAILED
      else: key=schMode0, blockDim=finalCoreNum -> R-001
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 950 通用逐元素路线 | `schMode=0` | 950；非空 ND；FLOAT/FLOAT16/BFLOAT16；资源与 alpha guard 全通过 | 单个算子自有 K0；外围框架 F0→K0→F1→F2 | `celu_v2<0>` -> `NsCeluV2::CeluV2<DTYPE_X>::Init/Process` | 双缓冲输入/输出 Queue、两个 FP32 TBuf；Vector pipe barrier；无跨核共享/atomic | 核域和 tile 都按 32-byte block；最后 tile 只搬入/搬出 `processDataNum`，计算仍跑满 `tileDataNum`；无 Device fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AIV/UB 可用 | `coreNum != 0 && ubSize != 0` | 必须产生有效 blockDim 并容纳 Queue/TBuf | R-001 | `GRAPH_FAILED` | `celu_v2_tiling_arch35.cpp::GetPlatformInfo` |
| dtype | `{DT_FLOAT, DT_FLOAT16, DT_BF16}` | 决定元素宽度、DTYPE_X 和 Cast 分支 | R-001 | `GRAPH_FAILED` | `GetShapeAttrsInfo` |
| alpha | `fabs(alpha) >= 1e-7f` | 避免 Tiling 计算 `1/alpha` 的近零输入 | R-001 | `GRAPH_FAILED`，无 fallback | `CeluV2TilingFunc` |
| 32-byte tile | `tileDataNum = tileBlockNum*32/s` | GM 搬运与每核域以 32-byte block 切分 | R-001 | `tileBlockNum` 至少 1 | `CeluV2TilingFunc` |
| UB 容量 | `tileBlockNum=floor((U/32/2)/ubDataNumber)` | 总分配恰按两组 depth-2 Queue 加两块 FP32 TBuf预算 | R-001 | 缩小 tile，无第二 Route | `CeluV2TilingFunc`、`CeluV2::Init` |
| blockDim 上限 | `finalCoreNum=min(AIV cores, ceil(N*s/32))` | 避免启动超过有效 32-byte block 数的核 | R-001 | 缩减 blockDim | `CeluV2TilingFunc` |
| 尾 tile | `i == tileNum-1` | 只提交 Host 计算的 core-tail 长度 | R-001 | `processDataNum=tailDataNum` | `CeluV2::Process` |
| 平台接线 | `SUPPORT_COMPUTE_UNIT=ascend950`、OpDef `AddConfig("ascend950")` | 当前实现只发布 arch35 body | R-001 | A2/A3 固定仓缺少本实现接线 | `CMakeLists.txt`、`celu_v2_def.cpp::CeluV2` |

### POD/ABI

`CeluV2TilingData` 在固定源码中是无显式 packing 的标准顺序结构；按当前字段类型，其 `alignof=8`、`sizeof=72`。Host 先整结构清零，再写所有非空路线字段；Device 通过 `GET_TILING_DATA_WITH_STRUCT` 读取同一结构。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CeluV2TilingData` | `smallCoreDataNum` | `int64_t`/64 | offset 0 | 元素 | `smallCoreDataNum = uint32(q*32/s)` 后转 int64 | `Init` 窄化到 `uint32_t coreDataNum` | 小核每核 block-rounded 容量。 |
| `CeluV2TilingData` | `bigCoreDataNum` | `int64_t`/64 | offset 8 | 元素 | `bigCoreDataNum = uint32((q+1)*32/s)` 后转 int64 | `Init` 计算大核容量和所有 GM 起点 | 大小核差一组 32-byte block。 |
| `CeluV2TilingData` | `finalBigTileNum` | `int64_t`/64 | offset 16 | tile | `ceil((q+1)/tileBlockNum)` | 大核 `Init -> tileNum` | 非空且大核存在时至少 1。 |
| `CeluV2TilingData` | `finalSmallTileNum` | `int64_t`/64 | offset 24 | tile | `ceil(q/tileBlockNum)` | 小核 `Init -> tileNum` | `q>=1`，至少 1。 |
| `CeluV2TilingData` | `tileDataNum` | `int64_t`/64 | offset 32 | 元素/tile | `tileBlockNum*32/s`，最小 1 | `InitBuffer` 容量、tile GM offset、Vector count | Device 窄化为 `uint32_t`。 |
| `CeluV2TilingData` | `smallTailDataNum` | `int64_t`/64 | offset 40 | 元素 | 小核余数；整除时为 `tileDataNum` | 小核最后一次 `processDataNum` | 只影响 CopyIn/CopyOut，不缩短 Vector count。 |
| `CeluV2TilingData` | `bigTailDataNum` | `int64_t`/64 | offset 48 | 元素 | 大核余数；整除时为 `tileDataNum` | 大核最后一次 `processDataNum` | 只影响 CopyIn/CopyOut，不缩短 Vector count。 |
| `CeluV2TilingData` | `tailBlockNum` | `int64_t`/64 | offset 56 | core | `blocksTotal % finalCoreNum` | `Init` 判断 `blockIdx < tailBlockNum` | 表示大核数量。 |
| `CeluV2TilingData` | `alpha` | `float`/32 | offset 64 | 标量 | attrs[0]，缺失时 1.0；ACLNN 始终显式传 `ToFloat()` | `Compute` 的第二个 `Muls` | 950 Tiling 拒绝绝对值小于 `1e-7`。 |
| `CeluV2TilingData` | `invAlpha` | `float`/32 | offset 68 | 标量 | `1.0f / alpha` | `Compute` 的第一个 `Muls` | 与 `alpha` 同时有效。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 950 通用逐元素路线 | `schMode=0` | `op_api/celu_v2.cpp::l0op::CeluV2`；`op_host/arch35/celu_v2_tiling_arch35.cpp::CeluV2TilingFunc` | `CeluV2TilingData` 全 10 字段 | `op_kernel/celu_v2_apt.cpp::celu_v2<0>` | `op_kernel/arch35/celu_v2.h::CeluV2<T>::Init/Process/CopyIn/Compute/CopyOut` | `CMakeLists.txt` 的 ascend950/arch35；`celu_v2_def.cpp` 的 ascend950 config 与 `celu_v2_apt` |

## Kernel 实现

### 公共执行流程

只有 R-001。entry `celu_v2<schMode>` 解析 POD，实例化 `CeluV2<DTYPE_X>`，依次调用 `Init` 和 `Process`；`schMode` 不进入 body 分支，Host 仅产生值 0。每个 AIV block 独占一段 GM，没有跨核共享、workspace 交接或 atomic。

`Init` 按 `tailBlockNum` 选择大小核参数，绑定 GM 子区并分配四个 UB 区。`Process` 串行迭代本核 tiles；每次由 VECIN Queue 搬入、Vector pipe 计算、VECOUT Queue 搬出。Queue 的 `EnQue/DeQue/FreeTensor` 管生命周期；算术依赖间由 `PipeBarrier<PIPE_V>` 保护。

### Route R-001：950 通用逐元素路线

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/celu_v2/op_api/aclnn_celu.cpp` | `CheckParams` / `GetWorkspaceSizeCommon` | 非空、950 支持 dtype、shape 相等、rank<=8、alpha 有序且非零；空 Tensor提前成功 | 非空进入 L0 `CeluV2`。 |
| Launcher | `activation/celu_v2/op_api/celu_v2.cpp` | `l0op::CeluV2` | 按 `self` view shape/dtype 分配 ND 输出，`ADD_TO_LAUNCHER_LIST_AICORE` 成功 | 添加 K0；失败返回 nullptr。 |
| Tiling | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | 资源非零、dtype 支持、`N>0`、`fabs(alpha)>=1e-7` | key=`schMode0`，blockDim=`finalCoreNum`。 |
| Router | `activation/celu_v2/op_kernel/celu_v2_apt.cpp` | `celu_v2<schMode>` | 编译实例 `schMode=0`，`DTYPE_X` 来自输入 dtype | `CeluV2<DTYPE_X>::Init -> Process`。 |

#### Device 数据流

##### 函数级调用链

```text
GetWorkspaceSizeCommon
  -> l0op::Contiguous(self)                         # 框架边界
  -> l0op::CeluV2(contiguousSelf, alpha)
    -> ADD_TO_LAUNCHER_LIST_AICORE(CeluV2, x, y, alpha)
      -> CeluV2TilingFunc(context)
      -> celu_v2<schMode=0>(x, y, workspace, tiling)
        -> GET_TILING_DATA_WITH_STRUCT(CeluV2TilingData)
        -> CeluV2<DTYPE_X>::Init(x, y, &tilingData)
        -> CeluV2<DTYPE_X>::Process()
          -> CopyIn(i)
          -> Compute(i)
          -> CopyOut(i)
  -> l0op::Cast(celuV2Out, out.dtype)                # 框架边界
  -> l0op::ViewCopy(castOut, out)                    # 框架边界
```

##### 源码等价伪码

```text
# 以下为固定源码等价伪码
block = GetBlockIdx()
if block < tailBlockNum:
    coreCount = bigCoreDataNum
    loops = finalBigTileNum
    tailCount = bigTailDataNum
    gmBase = bigCoreDataNum * block
else:
    coreCount = smallCoreDataNum
    loops = finalSmallTileNum
    tailCount = smallTailDataNum
    gmBase = bigCoreDataNum * block
             - (bigCoreDataNum-smallCoreDataNum)*(block-tailBlockNum)

bind xGM = x[gmBase : gmBase+coreCount]
bind yGM = y[gmBase : gmBase+coreCount]
allocate inputQueueX(depth=2, tileDataNum*T)
allocate outputQueueY(depth=2, tileDataNum*T)
allocate tmp1(tileDataNum*fp32)
allocate tmp2(tileDataNum*fp32)

for i in 0 .. loops-1:
    valid = tailCount if i == loops-1 else tileDataNum
    xLocal = inputQueueX.alloc()
    DataCopy(xLocal, xGM[i*tileDataNum], valid)
    inputQueueX.enqueue(xLocal)

    xLocal = inputQueueX.dequeue()
    yLocal = outputQueueY.alloc()
    tmp2 = cast_fp32(xLocal[0:tileDataNum])           # float 使用 Adds(...,0)
    barrier(V)
    tmp1 = tmp2 * invAlpha
    barrier(V)
    tmp1 = exp(tmp1)
    barrier(V)
    tmp1 = (tmp1 - 1) * alpha
    barrier(V)
    tmp1 = min(tmp1, 0)
    tmp2 = max(tmp2, 0)
    barrier(V)
    tmp1 = tmp2 + tmp1
    yLocal = cast_T_round(tmp1)                      # float 使用 Adds(...,0)
    outputQueueY.enqueue(yLocal)
    inputQueueX.free(xLocal)

    yLocal = outputQueueY.dequeue()
    DataCopy(yGM[i*tileDataNum], yLocal, valid)
    outputQueueY.free(yLocal)
```

尾 tile 的 Vector API 仍以 `tileDataNum` 为 count；只有 GM 搬入/搬出用 `valid=processDataNum`。因此无效 lane 的算术结果不会被提交到输出，但 Queue 中未由本次 CopyIn 覆盖的 lane 内容不应被解释为有效值。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` 子区 | 大核 `B*bigCoreDataNum`；小核为该值减大小核差额 | `coreDataNum` 元素 | T/ND 连续 | F0 / 用户输入 | `CopyIn` | executor；K0 只读 | 核域按 32-byte block 分配。 |
| `inputQueueX` | UB 由 `TPipe` 分配 | depth 2 × `tileDataNum*sizeof(T)` | T | `CopyIn` | `Compute` | K0 block；Alloc→EnQue→DeQue→Free | 双缓冲；每 tile 循环复用。 |
| `tmpBuffer1` | 独立 VECCALC 子区 | `tileDataNum*4` 字节 | FP32 | `Muls/Exp/Adds/Mins/Add` | 后续 Vector 指令 / cast | K0 block 全程 | 同一物理区原地执行 Exp、Adds、Muls、Mins、Add。 |
| `tmpBuffer2` | 独立 VECCALC 子区 | `tileDataNum*4` 字节 | FP32 | 输入 Cast 或 Adds(0)，随后 Maxs | Muls（读）、最终 Add | K0 block 全程 | 同区由原始 FP32 输入覆盖为正半轴。 |
| `outputQueueY` | UB 由 `TPipe` 分配 | depth 2 × `tileDataNum*sizeof(T)` | T | `Compute` | `CopyOut` | K0 block；Alloc→EnQue→DeQue→Free | 双缓冲；每 tile 循环复用。 |
| GM `y` 子区 | 与 x 同一核/tile offset | `coreDataNum` 元素，最终公开有效域为逻辑输出 shape | T/ND 连续中间 | `CopyOut` | F1 Cast | executor | 仅 `processDataNum` 从每 tile 写回。 |
| K0 workspace | executor 分配 slice；offset 固定仓不可见 | `512 + libApiWorkspace` | 字节 | Tiling/executor | 系统库边界；body 不读取 | executor 到 K0 完成 | 非算法中间量。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["inputQueueX · depth=2 · 2×tileDataNum×sizeof(T)"]:2
  tmp1["tmpBuffer1 · tileDataNum×4B"]:1
  tmp2["tmpBuffer2 · tileDataNum×4B"]:1
  outQ["outputQueueY · depth=2 · 2×tileDataNum×sizeof(T)"]:2
```

##### 字节账与复用顺序

```text
inputQueueX = 2 * tileDataNum * sizeof(T)
outputQueueY = 2 * tileDataNum * sizeof(T)
tmpBuffer1 = tileDataNum * 4
tmpBuffer2 = tileDataNum * 4
UB_total = tileDataNum * (4*sizeof(T) + 8)

sizeof(T)=2: UB_total = 16*tileDataNum，Host 用 ubDataNumber=4
sizeof(T)=4: UB_total = 24*tileDataNum，Host 用 ubDataNumber=3
tileDataNum = floor(floor(U/32/2)/ubDataNumber) * 32/sizeof(T)
=> 忽略整数向下取整余量时 UB_total <= U

dynamic_ub = N/A（AIV TPipe 静态 UB）
workspace_total(K0) = 512 + GetLibApiWorkSpaceSize()
workspace_total(ACLNN) = executor.GetWorkspaceSize()，框架节点分片公式固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inputQueueX` 当前 slot | `CopyIn` | `Compute` | depth-2 Queue 后续 tile | EnQue→DeQue→FreeTensor | Free 后才能复用；`CopyOut` 不再读取输入。 |
| `tmpBuffer1` | `Muls(invAlpha)` | `Exp` 后连续原地链 | 下一 tile 的首个 Muls | 每个依赖段的 `PipeBarrier<PIPE_V>` | barrier 前不得覆盖前一 Vector 结果。 |
| `tmpBuffer2` | Cast/Adds 输入 | `Muls` 读后由 `Maxs` 原地覆盖 | 下一 tile 的输入转换 | Muls 后 barrier；Maxs 与 Mins 后 barrier | 最终 Add 前 tmp2 必须保留正半轴结果。 |
| `outputQueueY` 当前 slot | `Compute` | `CopyOut` | depth-2 Queue 后续 tile | EnQue→DeQue→FreeTensor | CopyOut 完成并 Free 后复用。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | VECIN Queue EnQue/DeQue | `CopyIn` | `Compute` | 当前 x slot | DataCopy count=`processDataNum`。 |
| 输入升精度 | `Cast(...CAST_NONE)` 或 float `Adds(0)` + `PipeBarrier<PIPE_V>` | 输入 x | `Muls(invAlpha)` | `tmp2` | FP16/BF16 先升到 FP32；float 保持 FP32。 |
| 指数链 | 各 Vector API 之间的 V barrier | `Muls`、`Exp`、`Adds/Muls` | 后继 API | `tmp1` | 全部以 FP32、count=`tileDataNum` 执行。`Adds(-1)` 与 `Muls(alpha)` 之间无显式 barrier，固定源码依赖同一 Vector pipe 顺序。 |
| 正负支路 | `Mins`、`Maxs` 后 V barrier | tmp1 负支路、tmp2 正支路 | `Add` | 两个 FP32 TBuf | 两支都以 FP32 合并。 |
| 输出降精度 | `Cast(...CAST_ROUND)`；float `Adds(0)` | FP32 和 | VECOUT Queue | `yLocal` | FP16/BF16 使用 `CAST_ROUND`；FLOAT 不转换。 |
| UB→GM | VECOUT Queue EnQue/DeQue | `Compute` | `CopyOut` | 当前 y slot | 仅写 `processDataNum`，尾外 lane 不提交。 |
| 跨核 | N/A | 各核 | 各核 | 独占 GM 子区 | 无共享、atomic、event 或 barrier。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | 产生 blockDim、tile/core/tail、workspace、alpha/invAlpha 与 key 0。 |
| Kernel entry/router | `activation/celu_v2/op_kernel/celu_v2_apt.cpp` | `celu_v2<schMode=0>` | 解析 POD，按 `DTYPE_X` 实例化唯一 body。 |
| Device body/helper | `activation/celu_v2/op_kernel/arch35/celu_v2.h` | `CeluV2<T>::Init/Process/CopyIn/Compute/CopyOut` | GM 地址、四区 UB、Vector 指令顺序、Queue 生命周期与尾搬运。 |

## 算子特定优化

### Trick T-001：低精度输入统一升到 FP32 完成指数链

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：输入 `T` 为 FLOAT16 或 BFLOAT16；FLOAT 路线用同样的 FP32 TBuf，但以 `Adds(..., 0.0f)` 搬入计算区。
- Golden 来源：[`activation/celu_v2/op_kernel/arch35/celu_v2.h:112-132`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2.h#L112-L132)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (!std::is_same_v<T, float>) {
    AscendC::Cast(tmpTensor2, xLocal, RoundMode::CAST_NONE, this->tileDataNum);
    AscendC::PipeBarrier<PIPE_V>();
} else {
    AscendC::Adds(tmpTensor2, xLocal, 0.0f, this->tileDataNum);
    AscendC::PipeBarrier<PIPE_V>();
}
AscendC::Muls(tmpTensor1, tmpTensor2, invAlpha, this->tileDataNum);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Exp(tmpTensor1, tmpTensor1, this->tileDataNum);
```

#### 机制与设计原因

`tmpTensor1/tmpTensor2` 固定为 FP32。低精度输入先 `CAST_NONE` 到 `tmpTensor2`，再由 `Muls(invAlpha) -> Exp -> Adds(-1) -> Muls(alpha) -> Mins` 计算负半轴；正半轴直接在保存原始 FP32 输入的 `tmpTensor2` 上 `Maxs(0)`。这避免用 FP16/BF16 执行对误差敏感的除法等价缩放和指数链；最终只在输出边界以 `CAST_ROUND` 降回 T。两个 FP32 TBuf 同时也让正、负分支复用一次输入升精度结果。

#### 迁移边界

- 适用：需要 `max(x,0)+min(alpha*(exp(x/alpha)-1),0)` 且低精度输入可用两块 FP32 tile buffer 的逐元素激活。
- 前提：950 Vector Cast/Exp 可用；UB 必须容纳两组 depth-2 T Queue 与两个 FP32 TBuf；Host 必须用同一 `tileDataNum` 预算。
- 失效：UB 预算或 dtype guard 不满足时固定实现没有备用 Route，只能 Tiling 失败；不能删掉 FP32 buffer 后仍声称数值顺序等价。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从现有依赖可确定，若在 `Cast/Exp/Muls/Add` 的 producer-consumer 之间提前覆盖 `tmpTensor1/tmpTensor2`，会破坏正负支路合并。

### Trick T-002：Host 按真实四区 UB 字节账生成满容量 tile

#### 触发与源码依据

- 适用 Route：R-001 全 dtype。
- 触发条件：`ubSize>0` 且 type length 为 2 或 4 字节。
- Golden 来源：[`activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp:121-133`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp#L121-L133)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t inputBytes = static_cast<uint64_t>(typeLength);
uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;

uint32_t ubDataNumber = (inputBytes <= 2ULL) ? 4U : 3U;
uint64_t tmp = (ubSize / BLOCK_SIZE / BUFFER_NUM);
uint32_t tileBlockNum = 1U;
if (tmp > 0) {
    uint64_t tb = tmp / ubDataNumber;
    tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
}

uint32_t tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * BLOCK_SIZE) / inputBytes);
```

#### 机制与设计原因

`BUFFER_NUM=2` 先计入输入、输出 Queue 的双缓冲；`ubDataNumber` 再把两个 FP32 TBuf 折算到 32-byte block 预算中。2-byte dtype 时总字节为 `16*tileDataNum`，4-byte dtype 时为 `24*tileDataNum`，恰与 `U/32/2/4` 和 `U/32/2/3` 的分母对应。tile 因而同时保持 32-byte 对齐并由可用 UB 直接决定，不需要 Device 运行时再判断 UB。

#### 迁移边界

- 适用：Queue 深度、TBuf 数量和 dtype 宽度与本实现相同的单输入单输出 Vector pipeline。
- 前提：必须把 depth=2 和两块 FP32 buffer 全部计入；`tileDataNum` 仍需能安全窄化到 `uint32_t`。
- 失效：增加 buffer、提高 Queue depth 或改变中间 dtype 后必须重算 `ubDataNumber`；固定实现没有小 UB fallback，最小 tile 被钳到一个 32-byte block。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但 `CeluV2::Init` 的四次 `InitBuffer` 明确证明，若只按逻辑输入/输出两块 buffer 估算 UB，会漏算双缓冲倍数与两个 FP32 TBuf。

## 模板与实现对比

| 对比维度 | R-001 950 通用逐元素路线 | R-002 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 非空且所有 API/Tiling guard 通过；唯一 key 0 | N/A：无第二 Host 可达 Route | `schMode1` 只在 key 声明列表出现，Host 从不选择。 |
| 模板与 dtype | `CeluV2<DTYPE_X>`，FLOAT/FLOAT16/BFLOAT16 | N/A | 非 FLOAT 通过 `if constexpr` 升/降精度。 |
| 循环与分核 | 32-byte block 均分核；tile 循环 | N/A | 避免跨核共享。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth-2 VECIN/VECOUT + 2×FP32 TBuf；无 dynamic UB/通信 | N/A | 与 Host `ubDataNumber` 一致。 |
| 内存总占用与复用 | `tileDataNum*(4*sizeof(T)+8)`；tmp1 原地链、tmp2 保存并覆盖正支路 | N/A | 压缩中间物化到两个 FP32 tile。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy + Cast/Adds/Muls/Exp/Mins/Maxs/Add；无 SIMT/NDDMA/通信 | N/A | 纯 AIV_VECTOR 单 Kernel。 |
| 同步、精度与 tail | Queue + V barrier；FP32 中间；尾 tile 搬运缩短、计算满 tile | N/A | 无效 lane 不写回。 |
| fallback/失败边界 | 无 fallback；资源、dtype、near-zero alpha 失败 | N/A | 单实现族。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 API 文档声明支持，但 `activation/celu_v2` 固定仓没有 A2 实现闭包 | 公开 API 文档声明支持，但 `activation/celu_v2` 固定仓没有 A3 实现闭包 | 公开文档支持，且 ascend950/arch35 Host、Tiling、Kernel、构建接线闭合 |
| Host/Launcher | ACLNN 前端源码共用；实际 A2 Host/Tiling/Kernels 固定仓不可见 | ACLNN 前端源码共用；实际 A3 Host/Tiling/Kernels 固定仓不可见 | `CeluV2TilingFunc` + `ADD_TO_LAUNCHER_LIST_AICORE` |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | AIV Vector，双缓冲 Queue + 2×FP32 TBuf |

## 限制与失败边界

- 本卡只覆盖非原地 `aclnnCelu`；`aclnnInplaceCelu` 虽复用公共 helper，仍是独立语义入口，未并入本卡。
- API 层只检查 `alpha` 非 0/非 NaN，950 Tiling 进一步拒绝 `fabs(alpha)<1e-7`；因此极小但非零的 alpha 会通过第一层参数函数并在 Tiling 阶段失败。
- 空 Tensor 由 ACLNN 第一段直接成功返回且 workspace 为 0；Tiling 中 `totalIdx<=0` 的 key-0/`blockDim=1` 分支对本公开入口不可达。
- 分核域由 `ceil(N*sizeof(T)/32)` 构造，POD 没有单独保存逻辑总元素数。最后物理 block 的 padding/分配保证属于 Tensor/AscendC 框架边界，固定算子仓不可见；调用方只应读取 `out` 逻辑 shape 内元素。
- Tiling 的 `uint64_t -> uint32_t` tile/core 元素数窄化没有显式溢出 guard；真实 950 UB/core 上限下是否触发不由本卡推测。
- `workspaceSize` 的 ACLNN 总公式、通用 `Contiguous/Cast/ViewCopy` 的实际 Kernel 拓扑、workspace slice offset 与任务同步协议在固定算子仓不可见；`CeluV2` 自有 slice 公式和单 Kernel body 已闭合。
- A2/A3 虽有公开支持声明，但当前目录 CMake 明确只设 `ascend950`，OpDef 也只注册 `ascend950`，且只有 `arch35` Tiling/Kernel；所以未生成无后缀 A2/A3 卡。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/celu_v2/docs/aclnnCelu&aclnnInplaceCelu.md` | `aclnnCeluGetWorkspaceSize`、`aclnnCelu` | 公开平台、接口、dtype/shape/alpha 与错误约束。 |
| 入口声明/约束 | `activation/celu_v2/op_api/aclnn_celu.h` | 两段式 ACLNN 原型 | 公开 C ABI 与计算图。 |
| 入口实现 | `activation/celu_v2/op_api/aclnn_celu.cpp` | `CheckParams`、`GetWorkspaceSizeCommon`、`aclnnCelu` | 校验、空域、框架前后处理、executor/workspace。 |
| Host/Launcher | `activation/celu_v2/op_api/celu_v2.cpp` | `l0op::CeluV2` | 输出分配与 AICore launcher 添加。 |
| Host/Launcher | `activation/celu_v2/op_host/celu_v2_def.cpp` | `ops::CeluV2` | dtype/format、ascend950 config、kernel 文件名。 |
| Host/Tiling/Launcher | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | 资源、分核、tile、workspace、POD 与 key。 |
| POD/ABI | `activation/celu_v2/op_kernel/arch35/celu_v2_tiling_data.h` | `CeluV2TilingData` | Host/Device 参数布局。 |
| POD/ABI | `activation/celu_v2/op_kernel/arch35/celu_v2_tiling_key.h` | `ELEMENTWISE_TPL_SCH_MODE_0/1`、`ASCENDC_TPL_ARGS_DECL` | key 候选声明；Host 实际只产生 mode 0。 |
| Kernel entry/body | `activation/celu_v2/op_kernel/celu_v2_apt.cpp` | `celu_v2<schMode>` | POD 解析与 body 实例化。 |
| Kernel entry/body | `activation/celu_v2/op_kernel/arch35/celu_v2.h` | `CeluV2<T>` | GM/UB、循环、Vector、同步、精度与 tail。 |
| 构建接线 | `activation/celu_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR` | ascend950 + arch35 活动接线。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 8 个 int64 元素/计数字段后跟 2 个 float 标量 | 字段顺序、offset、Host 先清零、Device 窄化与单位 | `CeluV2TilingData`、`CeluV2TilingFunc`、`CeluV2::Init` |
| 调用顺序 | `Contiguous -> CeluV2 -> Cast -> ViewCopy`，K0 内 `CopyIn -> Compute -> CopyOut` | 空域提前返回；任一空返回/launcher 失败阻断后继 | `GetWorkspaceSizeCommon`、`CeluV2::Process` |
| workspace/内存 | K0 系统 workspace；UB 四区字节账与 tmp 原地复用 | depth=2、两个 FP32 TBuf、32-byte tile、owner/lifetime | `GetWorkspaceSize`、`CeluV2::Init` |
| 同步 | Queue 生命周期 + Vector pipe barriers；跨节点 executor stream ordering | 不能在 DeQue/Free 或 barrier 前覆盖 producer 数据 | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | 低精度升 FP32、FP32 指数链、`CAST_ROUND` 降精度；尾搬运缩短 | Vector count 保持 tileDataNum，无效 lane 不得写回 | `CeluV2::Compute`、`CeluV2::Process` |
| 失败边界 | API 参数错误、AIV/UB/dtype/typeLength/near-zero alpha Tiling 失败，无 fallback | 不能把 `schMode1` 当成 Host 可达；A2/A3 不得凭文档补实现 | `CheckParams`、`CeluV2TilingFunc`、`celu_v2_tiling_key.h` |
