---
type: Operator
title: "Hardswish 950 ACLNN 算子知识"
description: "说明 aclnnHardswish 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/docs/aclnnHardswish&aclnnInplaceHardswish.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/op_api/aclnn_hardswish.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/op_api/aclnn_hardswish.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/op_api/hardswish.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/hard_swish_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/hard_swish_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/config/ascend950/hard_swish_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_kernel/arch35/hard_swish_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_kernel/arch35/hard_swish_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_kernel/hard_swish_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_kernel/arch35/hard_swish.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T19:07:04Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Hardswish 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnHardswishGetWorkspaceSize` + `aclnnHardswish`，公开两段式 ACLNN；本文不包含原地变体 `aclnnInplaceHardswish`。 |
| 计算语义 | 逐元素 `y = max(x,-3) * clamp(x+3,0,6) / 6`；对有限数与公开分段式等价，`max(x,-3)` 还避免 `x=-inf` 时的 `-inf*0 -> NaN`。 |
| 输入输出 | `self`/`out` 为同 shape ND Tensor，公开文档给出 0～8 维、FLOAT/FLOAT16/BFLOAT16，两者均可非连续；`self -> out` dtype 必须可转换。 |
| 执行类型 | 算子自有 `HardSwish` 节点为单 Kernel `AIV_VECTOR`；ACLNN 外围 `Contiguous`/`Cast`/`ViewCopy` 是框架前后处理边界。 |
| Kernel 路线 | R-001 FP32 原生计算；R-002 FP16 升 FP32 计算；R-003 BF16 升 FP32 计算。 |
| 核心资源 | GM 输入/输出；depth=2 的 VECIN/VECOUT `TQue`；FP16/BF16 路线的 FP32 中间 Queue；算子节点 workspace=0。 |
| 来源闭包 | `source_closed`：公开 ACLNN、executor 编排、L0 launcher、950 Tiling/POD、三个 key 的 entry/body 与 ascend950 构建接线闭合；框架通用节点按边界标注。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公开语义对每个逻辑元素 `x_i` 定义：

```text
y_i = x_i,                    x_i >= 3
y_i = 0,                      x_i <= -3
y_i = x_i * (x_i + 3) / 6,    otherwise
```

950 Device body 的等价执行顺序是：

```text
t = max(min(x + 3, 6), 0)
x_safe = max(x, -3)
y = (x_safe * t) * (1/6)
```

有限 `x` 上两式等价。额外的 `x_safe` 防止 `x=-inf` 时原始乘法产生 `-inf*0=NaN`；`x<=-3` 时 `t=0`，所以结果仍为 0。固定仓未证明 Vector `Mins/Maxs` 对 NaN 的完整传播契约，不在此推测。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，0～8 维；可非连续 | FLOAT、FLOAT16、BFLOAT16 | 必填；公开输入 `x`。非空时先经 `Contiguous`。 |
| `out` | 与 `self` shape 相同；可非连续 | FLOAT、FLOAT16、BFLOAT16 | 必填；`self` dtype 必须可转换为 `out` dtype；最终由 `Cast` + `ViewCopy` 写入。 |
| `workspaceSize` | Host 标量指针 | `uint64_t*` | 第一段输出整个 executor 图的 Device workspace 大小。 |
| `executor` | 不适用 | `aclOpExecutor**` | 第一段输出并持有已编排计算图。 |
| `workspace` | Device 地址 | `void*` | 第二段输入；必须与同一 executor 返回的 `workspaceSize` 匹配。 |
| `stream` | ACL stream | `aclrtStream` | `CommonOpExecutorRun` 在该 stream 上执行编排图。 |

### 入口与版本

实际公开入口是 `aclnnHardswishGetWorkspaceSize` 与 `aclnnHardswish`，文档明确列出 Ascend 950PR/950DT、A2 和 A3 支持。本卡只发布 950：固定 Commit 中 `HardSwish` 的 OpDef、`SUPPORT_COMPUTE_UNIT`、Tiling 目录、binary config 与 Kernel body 只对 `ascend950/arch35` 活动接线，因而只有 950 达到 `source_closed`。

同文件的 `aclnnInplaceHardswish` 是原地语义变体，不纳入本逻辑算子。`experimental/activation/hard_swish_v2` 定义另一个 `aclnnHardSwishV2`/`HardSwishV2`，根 CMake 默认 `ENABLE_EXPERIMENTAL=OFF`，且它不是 `aclnnHardswish` 的 L0 下游，因此不作为 A2/A3 正式实现候选。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnHardswishGetWorkspaceSize` 按以下顺序执行：

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize` 与 `executor` 输出指针，然后 `CREATE_EXECUTOR`。
2. `CheckParams` 先检查 `self/out` 非空，再按 SoC 选择 dtype 支持表，校验 `self -> out` 可转换，最后检查 shape 相等。
3. `self->IsEmpty()` 时直接返回 `workspaceSize=0`，交付 executor，不编排 Device 节点。
4. 非空输入先调用框架 `l0op::Contiguous(self)`。
5. 仅在 SoC 不在 `[ASCEND910B, ASCEND910E]` 且输入为 BF16 时增加 BF16→FP16 的前置 `Cast`；950 正常路径不命中该分支。
6. `l0op::HardSwish(selfCast)` 按输入 storage shape/dtype/format 分配中间输出，通过 `ADD_TO_LAUNCHER_LIST_AICORE(HardSwish, ...)` 加入算子自有节点。
7. 无条件编排 `l0op::Cast(hardswishOpOut, out.dtype)`，再编排 `l0op::ViewCopy(castOut, out)`。等 dtype 或连续 view 时框架是否消除节点，固定算子仓不可见。
8. `GetWorkspaceSize()` 汇总整图 workspace，将 executor 所有权交给调用者。

任一中间 Tensor 为空或 launcher 注册失败均立即返回，不再编排后续节点。

### 参数、Workspace 与生命周期

`self`、中间 Tensor 和 `out` 由 executor 图持有至第二段执行完成。`HardSwishTilingFunc::GetWorkspaceSize` 将算子自有 K0 节点的唯一 slice 设为 `WS_SYS_SIZE=0`；Kernel entry 仍保留 `workspace` 形参，body 不读取它。

整个 ACLNN 图的 `workspaceSize` 还受 `Contiguous`/`Cast`/`ViewCopy` 框架节点影响，这些节点的 slice offset/extent 与消除规则不在固定算子源码内；因此不能把公开入口的整图 workspace 推测为 0。第二段 `CommonOpExecutorRun` 消费同一 executor、workspace 和 stream。

### 启动图与失败传播

非空 950 路径：

```text
self
  -> F0 Contiguous                                      # 框架边界
  -> K0 HardSwish / hard_swish<schMode> [AIV_VECTOR]   # 算子自有
  -> F1 Cast(to out.dtype)                              # 框架边界
  -> F2 ViewCopy(to out view)                           # 框架边界
  -> out
```

| 节点 | Host 添加函数 / entry | 网格与参数 | 输入输出 / workspace | ordering、owner 与失败传播 |
|---|---|---|---|---|
| F0 | `aclnnHardswishGetWorkspaceSize -> l0op::Contiguous` | 框架决定 | `self -> selfContiguous`；slice 固定仓不可见 | 同一 executor/stream 的前驱；空返回阻断 K0。 |
| K0 | `l0op::HardSwish -> ADD_TO_LAUNCHER_LIST_AICORE` / `hard_swish<0\|1\|2>` | `blockDim=usedCoreNum`；POD=`HardSwishTilingData` | 连续 GM x→同 dtype GM y；节点 workspace=0 | executor 依赖 F0 后启动；Tiling/launch 失败阻断 F1/F2。 |
| F1 | `aclnnHardswishGetWorkspaceSize -> l0op::Cast` | 框架决定 | K0 中间输出→`out.dtype` 中间；slice 不可见 | 依赖 K0；空返回阻断 F2。 |
| F2 | `aclnnHardswishGetWorkspaceSize -> l0op::ViewCopy` | 框架决定 | Cast 中间→用户 `out` view；slice 不可见 | 图尾节点；空返回使第一段失败，运行错误由 `CommonOpExecutorRun` 返回。 |

空 Tensor 路径没有 Device 节点且 workspace=0。固定仓中没有 K0 跨核共享、atomic 或显式 event；跨框架节点可见性由 executor 在传入 stream 上维护，具体任务/event 协议不可见。

## Route 与参数设计

### 分核与资源计算

记非空展平元素数为 `N`（rank-0 标量由 `EnsureNotScalar` 视为 shape `{1}`），AIV 核数为 `C`，UB 字节数为 `U`，`Q=GetUbBlockSize(context)` 的数值作为 `FloorAlign` 量子：

```text
blockFactor B = ceil(N / C)              # 元素/核
usedCoreNum C' = ceil(N / B)             # blockDim

R-001 FP32: ubFactor T = floor_align(floor((U / 4) / 6), Q)
R-002 FP16: ubFactor T = floor_align(floor((U / 4) / 8), Q)
R-003 BF16: ubFactor T = floor_align(floor((U / 4) / 8), Q)
```

固定仓可见代码直接把 `Q` 传给以元素计的 `FloorAlign`，但公共 helper 对 `Q` 的物理单位契约不在本算子文件中，因此不擅自将其写成固定字节数。`C==0`、`U==0` 或 `T<=0` 均 `GRAPH_FAILED`。

对 block `b=GetBlockIdx()`：

```text
remainder = N - B*b
blockLength = min(remainder, B)
GM base = B*b
loopCount = ceil(blockLength / T)
currentNum(i) = T,                       i < loopCount-1
              = blockLength - T*i,       i == loopCount-1
```

前 `C'-1` 核通常处理 `B` 个元素，末核通过 `remainder` 缩短；不使用 former/tail core POD。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原生路线 | `HARDSWISH_TPL_SCH_MODE_FP32=0`；`HardSwish<float>` | 950；ACLNN 非空；K0 dtype=`DT_FLOAT`；AIV core/UB 非零且 `ubFactor>0` | dtype 不同则继续下一分支；资源失败无 fallback。 |
| 2 | R-002 FP16 升精度路线 | `HARDSWISH_TPL_SCH_MODE_FP16=1`；`HardSwishFp16` | 950；ACLNN 非空；K0 dtype=`DT_FLOAT16`；AIV core/UB 非零且 `ubFactor>0` | dtype 不同则继续下一分支；资源失败无 fallback。 |
| 3 | R-003 BF16 升精度路线 | `HARDSWISH_TPL_SCH_MODE_BF16=2`；`HardSwishBf16` | 950；ACLNN 非空；K0 dtype=`DT_BF16`；AIV core/UB 非零且 `ubFactor>0` | 其他 dtype 在 API/Tiling 失败；无 fallback。 |

```text
aclnnHardswishGetWorkspaceSize
  if 参数非法: 返回 ACLNN 错误
  if self.empty: workspace=0，成功，无 Kernel
  else: 编排 Contiguous -> HardSwish -> Cast -> ViewCopy
        HardSwishTilingFunc
          if AIV核数==0 or UB==0 or dtype非法: GRAPH_FAILED
          compute blockFactor/usedCoreNum
          if dtype==FLOAT:   key=0, T=floor_align((U/4)/6,Q) -> R-001
          else if dtype==FP16: key=1, T=floor_align((U/4)/8,Q) -> R-002
          else if dtype==BF16: key=2, T=floor_align((U/4)/8,Q) -> R-003
          if T<=0: GRAPH_FAILED
          else blockDim=usedCoreNum
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | key 0 | 950；非空 ND；K0 FLOAT；资源 guard 通过 | 单个自有 K0；外围 F0→K0→F1→F2 | `hard_swish<0>` → `HardSwish<float>::Init/Process` | 3 个 depth-2 Queue；FP32 原地 clamp 输入；Queue 管生命周期 | 末核与末tile用 `currentNum`；无 fallback。 |
| R-002 FP16 | key 1 | 950；非空 ND；K0 FP16；资源 guard 通过 | 同上 | `hard_swish<1>` → `HardSwishFp16::Init/Process` | 4 个 depth-2 Queue；FP16→FP32，FP32 中间原地复用，`CAST_ROUND`→FP16 | 同上。 |
| R-003 BF16 | key 2 | 950；非空 ND；K0 BF16；资源 guard 通过 | 同上 | `hard_swish<2>` → `HardSwishBf16::Init/Process` | 4 个 depth-2 Queue；BF16→FP32，FP32 中间原地复用，`CAST_ROUND`→BF16 | 同上。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域 | API `self->IsEmpty()` | 避免 `N=0` 进入 `ceil(N/blockFactor)` 和无效 Kernel | 全部 | workspace=0，成功，不启动 K0 | `aclnn_hardswish.cpp::aclnnHardswishGetWorkspaceSize` |
| AIV/UB 可用 | `coreNum != 0 && ubSize != 0` | 必须产生有效 blockDim 并分配 Queue | 全部 | `GRAPH_FAILED` | `hard_swish_tiling.cpp::GetPlatformInfo` |
| dtype | `{DT_FLOAT,DT_FLOAT16,DT_BF16}` | 决定 key、元素宽度与升精度 body | 全部 | API/Tiling 失败 | `CheckDtypeValid`；`GetShapeAttrsInfo` |
| UB 容量 | FP32 按 `6*4*T`、FP16/BF16 按 `8*4*T` 保守预算 | 覆盖 depth=2 Queue；低精度路线需 FP32 中间量 | 全部 | 缩小 T；若对齐后 `T<=0` 则失败 | `HardSwishTilingFunc`；各 `Init` |
| UB 对齐 | `T=FloorAlign(...,GetUbBlockSize(context))` | 给 DMA/Vector tile 一致对齐量子 | 全部 | `T<=0` 失败，无第二 Route | `HardSwishTilingFunc` |
| 末核 | `remainder=N-B*blockIdx`，`blockLength=min(remainder,B)` | blockDim 可不整除 N | 全部 | 末核缩短有效域 | 各 Route `Init` |
| 末 tile | `i==loopCount-1` | 只读写剩余元素，不要求 GM 长度对齐 | 全部 | `currentNum=blockLength-T*i` | 各 Route `Process/CopyIn/CopyOut` |
| 平台接线 | `SUPPORT_COMPUTE_UNIT=ascend950`，OpDef `AddConfig("ascend950")` | 当前 body/tiling 只发布 arch35 | 全部 | A2/A3 无 `source_closed` 正式卡 | `CMakeLists.txt`；`ops::HardSwish::HardSwish` |

### POD/ABI

`HardSwishTilingData` 为三个连续 `int64_t` 字段，无显式 packing；按字段定义 `alignof=8`、`sizeof=24`、offset 依次为 0/8/16。Host 先整结构清零，再对非空 Route 写入全部字段；entry 通过 `GET_TILING_DATA_WITH_STRUCT` 解析同一结构。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `HardSwishTilingData` | `totalNum` | `int64_t`/64 | offset 0 | 元素 | `GetShapeAttrsInfo: EnsureNotScalar(storageShape).GetShapeSize()` | 各 `Init`：计算本核 remainder | API 空 Tensor 不进 K0；rank-0 记为 1。 |
| `HardSwishTilingData` | `blockFactor` | `int64_t`/64 | offset 8 | 元素/核 | `CeilDiv(totalNum, coreNum)` | 各 `Init`：GM base、blockLength | 仅非空有效；末核可小于该值。 |
| `HardSwishTilingData` | `ubFactor` | `int64_t`/64 | offset 16 | 元素/tile | dtype 分支的 `FloorAlign(FloorDiv((U/4),6\|8),Q)` | 各 `Init/Process/CopyIn/CopyOut`：Queue 容量、loop 与 GM offset | `<=0` 时 Tiling 失败；不进 Device。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | `schMode=0` | `l0op::HardSwish`；`HardSwishTilingFunc` FLOAT 分支 | `HardSwishTilingData` 全 3 字段 | `hard_swish_apt.cpp::hard_swish<0>` | `hard_swish.h::HardSwish<float>::Init/Process/CopyIn/Compute/CopyOut` | `CMakeLists.txt` ascend950/arch35；OpDef `hard_swish_apt`；binary config `HardSwish` |
| R-002 FP16 | `schMode=1` | 同上，FLOAT16 分支 | 同上 | `hard_swish_apt.cpp::hard_swish<1>` | `hard_swish.h::HardSwishFp16::*` | 同上 |
| R-003 BF16 | `schMode=2` | 同上，BF16 分支 | 同上 | `hard_swish_apt.cpp::hard_swish<2>` | `hard_swish.h::HardSwishBf16::*` | 同上 |

## Kernel 实现

### 公共执行流程

`hard_swish<schMode>` 解析 POD，在编译期按 0/1/2 分别实例化 `HardSwish<float>`、`HardSwishFp16`、`HardSwishBf16`，各自调用 `Init -> Process`。三条 Route 均是单 AIV Kernel；每个 block 独占 `[B*b, B*b+blockLength)` GM 域，无跨核共享、workspace 交接或 atomic。

`Process` 串行遍历本核 tile，每次按 `CopyIn -> Compute -> CopyOut` 执行。Queue 通过 `AllocTensor/EnQue/DeQue/FreeTensor` 管理生命周期；固定 body 没有显式 `PipeBarrier`/event。虽然 Queue depth=2，源码未把不同迭代拆成显式预取/计算交叠调度，不将未实测的重叠效果作为结论。

### Route R-001：FP32 原生计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/hard_swish/op_host/op_api/aclnn_hardswish.cpp` | `CheckParams` / `aclnnHardswishGetWorkspaceSize` | 参数有效且 `self` 非空 | 编排 L0 `HardSwish`。 |
| Launcher | `activation/hard_swish/op_host/op_api/hardswish.cpp` | `l0op::HardSwish` | 按 self storage metadata 分配同 dtype 输出，launcher 注册成功 | 添加 K0。 |
| Tiling | `activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp` | `HardSwishTilingFunc` | `dataType==DT_FLOAT`，资源非零，`ubFactor>0` | key 0，`blockDim=usedCoreNum`。 |
| Router | `activation/hard_swish/op_kernel/hard_swish_apt.cpp` | `hard_swish<schMode>` | `schMode==0` | `HardSwish<float>::Init -> Process`。 |

#### Device 数据流

##### 函数级调用链

```text
aclnnHardswishGetWorkspaceSize
  -> Contiguous(self)                                  # 框架边界
  -> l0op::HardSwish(selfContiguous)
    -> ADD_TO_LAUNCHER_LIST_AICORE(HardSwish, x, y)
      -> HardSwishTilingFunc
      -> hard_swish<0>(x, y, workspace, tiling)
        -> HardSwish<float>::Init
        -> HardSwish<float>::Process
          -> CopyIn(i,currentNum)
          -> Compute(currentNum)
          -> CopyOut(i,currentNum)
  -> Cast(..., out.dtype) -> ViewCopy(..., out)         # 框架边界
```

##### 源码等价伪码

```text
# 伪码；b=blockIdx, T=ubFactor
len = min(totalNum - blockFactor*b, blockFactor)
xGM = x + blockFactor*b; yGM = y + blockFactor*b
for i in [0, ceil(len/T)):
    n = (last_iter ? len - T*i : T)
    xLocal = DataCopyPad(xGM + i*T, blockCount=1, blockLen=n*4, strides=0)
    tmp = xLocal + 3                  # Adds, count=n
    tmp = min(tmp, 6)                 # Mins, count=n, in-place
    tmp = max(tmp, 0)                 # Maxs, count=n, in-place
    xLocal = max(xLocal, -3)          # Maxs, count=n, in-place
    yLocal = xLocal * tmp              # Mul, count=n
    yLocal = yLocal * (1/6)           # Muls, count=n, in-place
    DataCopyPad(yGM + i*T, yLocal, blockCount=1, blockLen=n*4, strides=0)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `B*b + T*i` | 本核 `blockLength`；当次 `n` | FP32/ND 展平 | F0 | `CopyIn` | executor/调用者；K0 全程有效 | 末 tile 用 `DataCopyPad` |
| `inputQueueX` | UB 由 `TPipe` 分配 | `2*T*4` bytes | FP32 | `CopyIn` | `Compute` | `Alloc -> EnQue -> DeQue -> Free` | depth=2；`Compute` 原地 `Maxs` |
| `tmpQueue` | UB 由 `TPipe` 分配 | `2*T*4` bytes | FP32 | `Compute::Adds` | `Mins/Maxs/Mul` | 单 tile Compute 内 | depth=2；Adds后原地覆盖 |
| `outputQueueY` | UB 由 `TPipe` 分配 | `2*T*4` bytes | FP32 | `Mul/Muls` | `CopyOut` | `Alloc -> EnQue -> DeQue -> Free` | depth=2；Muls 原地 |
| GM y | `B*b + T*i` | 本核 `blockLength`；当次 `n` | FP32/ND 展平 | `CopyOut` | F1 Cast | executor 中间 Tensor | 仅写 `n` 元素 |
| workspace | 不读取 | 0 bytes | N/A | Host Tiling | N/A | K0 节点 | 形参保留，body 不用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["inputQueueX · VECIN · depth=2 · 2*T*4 B"]:1
  tmpQ["tmpQueue · VECIN · depth=2 · 2*T*4 B"]:1
  outQ["outputQueueY · VECOUT · depth=2 · 2*T*4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*T*4 + 2*T*4 + 2*T*4 = 24*T bytes
Host budget: 6*T*4 = 24*T <= U （再受 FloorAlign 缩小）
dynamic_ub = N/A
K0 workspace_total = 0
=> T > 0 才可启动 R-001
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inputQueueX` 当前 slot | `CopyIn` | `Compute` 全部算术，其中 `Maxs` 原地改写 | `FreeTensor` 后的后续 tile | `EnQue -> DeQue -> FreeTensor` | `Mul` 完成前不得释放。 |
| `tmpQueue` 当前 slot | `Adds` | `Mins -> Maxs -> Mul` | `FreeTensor` 后的后续 tile | `FreeTensor(tmpLocal)` | `Mul` 完成前不得覆盖。 |
| `outputQueueY` 当前 slot | `Mul -> Muls` | `CopyOut` | `FreeTensor` 后的后续 tile | `EnQue -> DeQue -> FreeTensor` | GM 写完后才可释放。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + Queue | GM x | `inputQueueX.DeQue` | 当前 x slot | `blockLen=n*4`，无效域不参与。 |
| Vector | 数据依赖，无显式 barrier | `Adds/Mins/Maxs` | 后续 `Mul/Muls` | x/tmp/y LocalTensor | 全程 FP32；所有 count=`n`；`x` 先 clamp 到 -3。 |
| UB→GM | Queue + `DataCopyPad` | `Muls`/EnQue | `CopyOut`/DeQue | y slot 与 GM 有效域 | `blockLen=n*4`；末 tile 只写有效元素。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp` | `HardSwishTilingFunc` FLOAT 分支 | 产生 B/T/blockDim/key0/workspace0。 |
| Kernel entry/router | `activation/hard_swish/op_kernel/hard_swish_apt.cpp` | `hard_swish<0>` | key0 映射 `HardSwish<float>`。 |
| Device body/helper | `activation/hard_swish/op_kernel/arch35/hard_swish.h` | `HardSwish<float>::Init/CopyIn/Compute/CopyOut/Process` | 分核、Queue、六步 Vector 公式和尾 tile 闭合。 |

### Route R-002：FP16 升 FP32 计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/hard_swish/op_host/op_api/aclnn_hardswish.cpp` | `CheckParams` / `aclnnHardswishGetWorkspaceSize` | FP16 可用，非空，shape/转换校验通过 | 编排 L0 `HardSwish`。 |
| Launcher | `activation/hard_swish/op_host/op_api/hardswish.cpp` | `l0op::HardSwish` | 产生同 FP16 的 K0 输出 | 添加 K0。 |
| Tiling | `activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp` | `HardSwishTilingFunc` | `dataType==DT_FLOAT16`，资源 guard 通过 | key1，T 按 `8*float` 预算。 |
| Router | `activation/hard_swish/op_kernel/hard_swish_apt.cpp` | `hard_swish<1>` | `schMode==1` | `HardSwishFp16::Init -> Process`。 |

#### Device 数据流

##### 函数级调用链

```text
l0op::HardSwish(FP16 x)
  -> HardSwishTilingFunc -> key=1
  -> hard_swish<1>
    -> HardSwishFp16::Init
    -> HardSwishFp16::Process
      -> CopyIn -> Compute[Cast up, FP32 HardSwish, Cast down] -> CopyOut
  -> framework Cast(to out.dtype) -> ViewCopy
```

##### 源码等价伪码

```text
# 伪码；地址/分核/loop 与 R-001 相同，低精度 DMA 每元素 2 bytes
x16 = DataCopyPad(xGM + i*T, blockLen=n*2)
x32 = Cast<float>(x16, CAST_NONE, n)
tmp32 = max(min(x32 + 3.0, 6.0), 0.0)
x32 = max(x32, -3.0)
x32 = x32 * tmp32                 # 原地复用 x32 Queue
x32 = x32 * (1.0/6.0)
y16 = Cast<half>(x32, CAST_ROUND, n)
DataCopyPad(yGM + i*T, y16, blockLen=n*2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `B*b+T*i` | 当次 n | FP16/ND 展平 | F0 / `CopyOut` | `CopyIn` / F1 | executor | DMA `blockLen=n*2` |
| `inputFp16Queue` | UB/TPipe | `2*T*2` B | FP16 | `CopyIn` | up-Cast | Queue slot | depth=2 |
| `inputFp32Queue` | UB/TPipe | `2*T*4` B | FP32 | up-Cast | FP32 算术/down-Cast | Compute 内至 down-Cast 完成 | `Maxs/Mul/Muls` 原地复用 |
| `tmpQueue` | UB/TPipe | `2*T*4` B | FP32 | `Adds` | `Mins/Maxs/Mul` | Compute 内 | 原地 clamp |
| `outputFp16Queue` | UB/TPipe | `2*T*2` B | FP16 | down-Cast | `CopyOut` | Queue slot | depth=2 |
| workspace | 不读取 | 0 B | N/A | Host | N/A | K0 | N/A |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in16["inputFp16Queue · d2 · 2*T*2 B"]:1
  in32["inputFp32Queue · d2 · 2*T*4 B"]:1
  tmp32["tmpQueue FP32 · d2 · 2*T*4 B"]:1
  out16["outputFp16Queue · d2 · 2*T*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*T*2 + 2*T*4 + 2*T*4 + 2*T*2 = 24*T bytes
Host conservative budget = 8*T*4 = 32*T <= U
dynamic_ub = N/A
K0 workspace_total = 0
=> 可见实际分配为保守预算的 3/4，T>0 时启动
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inputFp16Queue` slot | `CopyIn` | up-Cast | `FreeTensor` 后 | Queue 协议 | up-Cast 完成前不释放。 |
| `inputFp32Queue` slot | up-Cast | clamp、Mul/Muls、down-Cast | down-Cast 后 `FreeTensor` | 函数顺序 + `FreeTensor` | y16 生成前不覆盖。 |
| `tmpQueue` slot | Adds | Mins/Maxs/Mul | Mul 后 `FreeTensor` | `FreeTensor` | Mul 完成前保留。 |
| `outputFp16Queue` slot | down-Cast | CopyOut | GM 写完后 | EnQue/DeQue/Free | CopyOut 完成前保留。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + Queue | GM FP16 | up-Cast | input slot | `n*2` bytes，只搬有效尾部。 |
| 升精度 | `Cast(...CAST_NONE,n)` | FP16 x | FP32 计算 | x32 slot | 转 FP32 后才执行 Adds/Mins/Maxs/Mul/Muls。 |
| 降精度 | `Cast(...CAST_ROUND,n)` | FP32 y | FP16 y | output slot | 每 tile 结束一次舍入。 |
| UB→GM | Queue + `DataCopyPad` | y16 | GM y | output slot | `n*2` bytes，末 tile 有效保护。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp` | `HardSwishTilingFunc` FLOAT16 分支 | key1，按 8 个 FP32 份额预算 T。 |
| Kernel entry/router | `activation/hard_swish/op_kernel/hard_swish_apt.cpp` | `hard_swish<1>` | key1 映射 `HardSwishFp16`。 |
| Device body/helper | `activation/hard_swish/op_kernel/arch35/hard_swish.h` | `HardSwishFp16::*` | 上/下 Cast、FP32 公式、Queue 复用与 tail 闭合。 |

### Route R-003：BF16 升 FP32 计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/hard_swish/op_host/op_api/aclnn_hardswish.cpp` | `CheckSocVersionGe910B` / `CheckDtypeValid` | 950 属公开支持平台，BF16 使用包含 BF16 的支持表；输入非空 | 950 不做 BF16→FP16 前置 Cast，编排 BF16 K0。 |
| Launcher | `activation/hard_swish/op_host/op_api/hardswish.cpp` | `l0op::HardSwish` | 产生同 BF16 的 K0 输出 | 添加 K0。 |
| Tiling | `activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp` | `HardSwishTilingFunc` | `dataType==DT_BF16`，资源 guard 通过 | key2，T 按 `8*float` 预算。 |
| Router | `activation/hard_swish/op_kernel/hard_swish_apt.cpp` | `hard_swish<2>` | `schMode==2` | `HardSwishBf16::Init -> Process`。 |

#### Device 数据流

##### 函数级调用链

```text
l0op::HardSwish(BF16 x)
  -> HardSwishTilingFunc -> key=2
  -> hard_swish<2>
    -> HardSwishBf16::Init
    -> HardSwishBf16::Process
      -> CopyIn -> Compute[Cast up, FP32 HardSwish, Cast down] -> CopyOut
  -> framework Cast(to out.dtype) -> ViewCopy
```

##### 源码等价伪码

```text
# 伪码；与 R-002 的唯一数据类型差异是 BF16 输入/输出
xb = DataCopyPad(xGM + i*T, blockLen=n*2)
x32 = Cast<float>(xb, CAST_NONE, n)
tmp32 = max(min(x32 + 3.0, 6.0), 0.0)
x32 = max(x32, -3.0)
x32 = (x32 * tmp32) * (1.0/6.0)
yb = Cast<bfloat16_t>(x32, CAST_ROUND, n)
DataCopyPad(yGM + i*T, yb, blockLen=n*2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `B*b+T*i` | 当次 n | BF16/ND 展平 | F0 / `CopyOut` | `CopyIn` / F1 | executor | DMA `blockLen=n*2` |
| `inputBf16Queue` | UB/TPipe | `2*T*2` B | BF16 | `CopyIn` | up-Cast | Queue slot | depth=2 |
| `inputFp32Queue` | UB/TPipe | `2*T*4` B | FP32 | up-Cast | FP32 算术/down-Cast | Compute 内至 down-Cast | 原地 `Maxs/Mul/Muls` |
| `tmpQueue` | UB/TPipe | `2*T*4` B | FP32 | Adds | Mins/Maxs/Mul | Compute 内 | 原地 clamp |
| `outputBf16Queue` | UB/TPipe | `2*T*2` B | BF16 | down-Cast | CopyOut | Queue slot | depth=2 |
| workspace | 不读取 | 0 B | N/A | Host | N/A | K0 | N/A |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inb["inputBf16Queue · d2 · 2*T*2 B"]:1
  in32["inputFp32Queue · d2 · 2*T*4 B"]:1
  tmp32["tmpQueue FP32 · d2 · 2*T*4 B"]:1
  outb["outputBf16Queue · d2 · 2*T*2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*T*2 + 2*T*4 + 2*T*4 + 2*T*2 = 24*T bytes
Host conservative budget = 8*T*4 = 32*T <= U
dynamic_ub = N/A
K0 workspace_total = 0
=> 可见实际分配为保守预算的 3/4，T>0 时启动
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inputBf16Queue` slot | CopyIn | up-Cast | Free 后 | Queue 协议 | up-Cast 完成前不释放。 |
| `inputFp32Queue` slot | up-Cast | clamp、Mul/Muls、down-Cast | down-Cast 后 Free | 函数顺序 + Free | yb 生成前不覆盖。 |
| `tmpQueue` slot | Adds | Mins/Maxs/Mul | Mul 后 Free | `FreeTensor` | Mul 完成前保留。 |
| `outputBf16Queue` slot | down-Cast | CopyOut | GM 写完后 | EnQue/DeQue/Free | CopyOut 完成前保留。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + Queue | GM BF16 | up-Cast | input slot | `n*2` bytes，只搬有效尾部。 |
| 升精度 | `Cast(...CAST_NONE,n)` | BF16 x | FP32 计算 | x32 slot | 算术全在 FP32 执行。 |
| 降精度 | `Cast(...CAST_ROUND,n)` | FP32 y | BF16 y | output slot | 每 tile 结束一次舍入。 |
| UB→GM | Queue + `DataCopyPad` | yb | GM y | output slot | `n*2` bytes，末 tile 有效保护。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp` | `HardSwishTilingFunc` BF16 分支 | key2，按 8 个 FP32 份额预算 T。 |
| Kernel entry/router | `activation/hard_swish/op_kernel/hard_swish_apt.cpp` | `hard_swish<2>` | key2 映射 `HardSwishBf16`。 |
| Device body/helper | `activation/hard_swish/op_kernel/arch35/hard_swish.h` | `HardSwishBf16::*` | BF16↔FP32 Cast、FP32 公式、Queue 复用与 tail 闭合。 |

## 算子特定优化

### Trick T-001：乘法前 clamp x 消除负无穷边界 NaN

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：所有非空 tile；对 `x<=-3` 尤其是 `x=-inf` 决定结果边界。
- Golden 来源：[`activation/hard_swish/op_kernel/arch35/hard_swish.h:114-126`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_kernel/arch35/hard_swish.h#L114-L126)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// Step 1: tmpLocal = x + 3
Adds(tmpLocal, xLocal, static_cast<T>(3.0f), currentNum);
// Step 2: tmpLocal = min(tmpLocal, 6)
Mins(tmpLocal, tmpLocal, static_cast<T>(6.0f), currentNum);
// Step 3: tmpLocal = max(tmpLocal, 0) => clamp(x+3, 0, 6)
Maxs(tmpLocal, tmpLocal, static_cast<T>(0.0f), currentNum);
// Step 4 (ISSUE-001 fix): clamp x to >= -3, so -inf becomes -3.
//   When x <= -3, clamp result is 0, so max(x,-3)*0 = -3*0 = 0 (not NaN).
Maxs(xLocal, xLocal, static_cast<T>(-3.0f), currentNum);
// Step 5: yLocal = max(x,-3) * clamp(x+3, 0, 6)
Mul(yLocal, xLocal, tmpLocal, currentNum);
// Step 6: yLocal = yLocal * (1/6)
Muls(yLocal, yLocal, static_cast<T>(1.0f / 6.0f), currentNum);
```

#### 机制与设计原因

`currentNum` 来自当前 tile 的有效元素数，同时作为每个 scalar Vector API 的 count。先得到 `t=clamp(x+3,0,6)`，再把乘数 x 原地改为 `max(x,-3)`。对有限数，当 x<-3 时 t=0，改写 x 不改变结果；对 `-inf`，它把未定形 `-inf*0` 改成 `-3*0`。这是精度/特殊值稳定机制，且原地 `Maxs` 不需新 UB 缓冲区。R-002/R-003 在 FP32 中执行同样顺序。

#### 迁移边界

- 适用：可写成 `x*clamp(...)` 且 clamp 的零区域与 x 安全截断区域一致的分段激活。
- 前提：截断后有限 x 上数学语义不变；Vector `Maxs` 在乘法前完成。
- 失效：若零区域不覆盖被截断的 x，该改写会改变有限数结果，必须另设特殊值处理。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码注释给出了可验证反例：直接计算 `x*clamp(x+3,0,6)` 会在 `x=-inf` 形成 `-inf*0=NaN`，违反分段语义的零输出。

### Trick T-002：FP16/BF16 升 FP32 计算并原地复用中间 Queue

#### 触发与源码依据

- 适用 Route：R-002、R-003。
- 触发条件：Tiling dtype 为 FLOAT16 或 BF16，分别选 key1/key2。
- Golden 来源：[`activation/hard_swish/op_kernel/arch35/hard_swish.h:223-240`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish/op_kernel/arch35/hard_swish.h#L223-L240)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// fp16 -> fp32
Cast(xFp32Local, xFp16Local, RoundMode::CAST_NONE, currentNum);

// HardSwish in fp32
Adds(tmpLocal, xFp32Local, 3.0f, currentNum);
Mins(tmpLocal, tmpLocal, 6.0f, currentNum);
Maxs(tmpLocal, tmpLocal, 0.0f, currentNum);
// ISSUE-001 fix: clamp x to >= -3 to avoid -inf * 0 = NaN
Maxs(xFp32Local, xFp32Local, -3.0f, currentNum);
Mul(xFp32Local, xFp32Local, tmpLocal, currentNum);
Muls(xFp32Local, xFp32Local, 1.0f / 6.0f, currentNum);

// fp32 -> fp16
Cast(yFp16Local, xFp32Local, RoundMode::CAST_ROUND, currentNum);

outputFp16Queue.EnQue<half>(yFp16Local);
tmpQueue.FreeTensor(tmpLocal);
inputFp32Queue.FreeTensor(xFp32Local);
```

#### 机制与设计原因

低精度 GM 输入先以 `CAST_NONE` 提升到 FP32，六步 HardSwish 管线全在 FP32 执行，最后用 `CAST_ROUND` 一次降回目标低精度。`xFp32Local` 同时承载升精度 x、clamp 后 x 和乘除结果，避免再分配一块 FP32 result Queue。参数 `currentNum` 由 tile tail 生产，确保 Cast 与算术只处理有效 lane。BF16 body 在行 332～350 使用同样管线，仅低精度类型/队列名不同。

#### 迁移边界

- 适用：对中间舍入敏感、但输入输出仍是 FP16/BF16 的逐元素链。
- 前提：UB 能容纳两个 depth-2 FP32 Queue 与两个低精度 Queue；中间 x 在下游不再需原值后才能原地复用。
- 失效：UB 对齐后 `ubFactor<=0` 则 Host 直接失败，固定仓没有低精度直算 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证的低精度直算反例或误差数据；不能据此声称具体精度提升幅度。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16→FP32 | R-003 BF16→FP32 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | `DT_FLOAT`，key0 | `DT_FLOAT16`，key1 | `DT_BF16`，key2 | Tiling 互斥 dtype if/else 顺序。 |
| 模板与 dtype | `HardSwish<float>` | 非模板类 `HardSwishFp16` | 非模板类 `HardSwishBf16` | router 的 `if constexpr(schMode)`。 |
| 循环与分核 | 均为 `B*b`、`ceil(blockLength/T)` | 同左 | 同左 | 共用 POD 与等价 `Init/Process` 结构。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 个 depth-2 Queue | 4 个 depth-2 Queue，含 2 个 FP32 Queue | 同 R-002，低精度类型为 BF16 | 无 TBuf/通信/dynamic UB；寄存器占用不可见。 |
| 内存总占用与复用 | `24*T` B，Host 亦预算 `24*T` | `24*T` B，Host 保守预算 `32*T` | 同 R-002 | 低精度 Queue 按 FP32 份额保守估算，FP32 x 原地承载结果。 |
| 搬运/Vector API | 4-byte `DataCopyPad`；Adds/Mins/Maxs/Mul/Muls | 2-byte DMA + up/down Cast + FP32 算术 | 同 R-002，BF16 Cast | 输入宽度与中间精度不同。 |
| 同步、精度与 tail | Queue；全 FP32；`n` 控制尾 tile | Queue；`CAST_NONE`升、`CAST_ROUND`降；`n` tail | 同 R-002 | 固定 body 无显式 barrier，无 padding lane 算术。 |
| fallback/失败边界 | 资源/对齐失败即 `GRAPH_FAILED` | 同左 | 同左 | 没有第二 Tiling 或 Device fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 `aclnnHardswish` 文档列为支持，但固定仓无活动 HardSwish Host/Tiling/Kernel 接线，不达 `source_closed` | 同 A2 | 公开支持且 ascend950/arch35 全链接线，`source_closed` |
| Host/Launcher | ACLNN 前置编排可见，但平台自有 `HardSwish` body 缺失 | 同 A2 | `HardSwishTilingFunc`；FP32/FP16/BF16 三 key；`GetCoreNumAiv` |
| Kernel/资源 | 不发布推测 | 不发布推测 | `hard_swish_apt.cpp` + arch35 三 body；depth-2 Queue；workspace0 |

## 限制与失败边界

- 公开 API 约束 ND、0～8 维、FLOAT/FLOAT16/BFLOAT16、同 shape、dtype 可转换；实现中 `CheckParams` 显式检查空指针、dtype/转换和 shape，rank/format 的通用检查边界在框架宏与 OpDef 内。
- 空 Tensor 是成功的无 Kernel 路径；rank-0 非空标量在 Tiling 中按 1 元素处理。
- `N`、B、T 和 GM offset 均用 `int64_t`；`GetShapeSize()` 乘法溢出契约属框架边界，固定算子文件未加额外溢出检查。
- `FloorAlign` 数值对齐后 `ubFactor<=0` 直接失败；没有小 UB fallback。`GetUbBlockSize` 的物理单位契约在公共 helper 边界，本卡保留原始表达式。
- FP16/BF16 路线的 Host UB 预算为 `32*T` bytes，可见 Queue 实际合计 `24*T` bytes；不将保守余量推测为可供其它未分配缓冲使用的空间。
- Vector primitive 的硬件 repeat/mask 拆分由 scalar count 形式 API 内部完成，固定算子 body 只可证明 count=`currentNum`，不推测内部 repeat stride。
- A2/A3 虽在公开 API 产品表中支持，但固定 Commit 缺少与 `aclnnHardswish -> l0op::HardSwish` 相匹配的活动 A2/A3 Host/Tiling/Kernel 全链；因此未生成 `hardswish.md`。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/hard_swish/docs/aclnnHardswish&aclnnInplaceHardswish.md` | 产品支持表；`aclnnHardswish*` 原型 | 公开性、平台、语义、dtype/shape/layout。 |
| 入口声明/约束 | `activation/hard_swish/op_host/op_api/aclnn_hardswish.h` | `aclnnHardswishGetWorkspaceSize`；`aclnnHardswish` | 两段式 ABI 与框架路径说明。 |
| 入口实现 | `activation/hard_swish/op_host/op_api/aclnn_hardswish.cpp` | `CheckParams`；`aclnnHardswishGetWorkspaceSize`；`aclnnHardswish` | 校验、空域、前后处理、executor/workspace/失败传播。 |
| Launcher | `activation/hard_swish/op_host/op_api/hardswish.cpp` | `l0op::HardSwish` | 中间输出分配与 AICore launcher 注册。 |
| OpDef/平台 | `activation/hard_swish/op_host/hard_swish_def.cpp` | `ops::HardSwish::HardSwish` | 950 dtype/format、ascend950 config、`hard_swish_apt`。 |
| shape 推导 | `activation/hard_swish/op_host/hard_swish_infershape.cpp` | `InferShape4HardSwish` | 输出 shape 精确复制输入 shape，空 context 指针失败。 |
| Host/Tiling | `activation/hard_swish/op_host/arch35/hard_swish_tiling.cpp` | `GetPlatformInfo`；`GetShapeAttrsInfo`；`HardSwishTilingFunc` | 分核、UB 公式、三 key、workspace0 和 guard。 |
| binary 元数据 | `activation/hard_swish/op_host/config/ascend950/hard_swish_binary.json` | `op_type=HardSwish` 的三个 `op_list` | ascend950 上 BF16/FP16/FP32 的 ND 输入输出 binary 实例。 |
| POD | `activation/hard_swish/op_kernel/arch35/hard_swish_tiling_data.h` | `HardSwishTilingData` | 三字段 ABI。 |
| Tiling key | `activation/hard_swish/op_kernel/arch35/hard_swish_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL(HardSwish)` | schMode 0/1/2 合法实例。 |
| Kernel entry | `activation/hard_swish/op_kernel/hard_swish_apt.cpp` | `hard_swish<schMode>` | key→FP32/FP16/BF16 body 映射。 |
| Kernel body | `activation/hard_swish/op_kernel/arch35/hard_swish.h` | `HardSwish<T>`；`HardSwishFp16`；`HardSwishBf16` | Queue、字节账、DMA、Vector 顺序、Cast、复用、tail。 |
| 构建接线 | `activation/hard_swish/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`；`SUPPORT_TILING_DIR` | 只活动 ascend950/arch35。 |
| Host target 接线 | `activation/hard_swish/op_host/CMakeLists.txt` | `add_modules_sources(... OPTYPE hard_swish ... COMPUTE_UNIT ... TILING_DIR ...)` | 将 ACLNN/Host 源码与 ascend950/arch35 参数加入活动 target。 |
| binary 接线 | `scripts/kernel/binary_config/ascendc_config.json` | `name=HardSwish` | ascend950 binary 与 SIMD 编译选项。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三 route 共用 24-byte POD | `totalNum/blockFactor/ubFactor` 的顺序、64 位与元素单位 | `HardSwishTilingData` |
| 调用顺序 | API 图 F0→K0→F1→F2；K0 `Init->Process->CopyIn/Compute/CopyOut` | 空域必须在编排前返回；Cast/ViewCopy 失败必须阻断 | `aclnnHardswishGetWorkspaceSize`；各 `Process` |
| workspace/内存 | K0 workspace=0；FP32 `24*T` B；FP16/BF16 实际 `24*T` B、Host 预算 `32*T` B | Queue depth=2 必须计入；整图 workspace 不可由 K0 的 0 替代 | `GetWorkspaceSize`；各 `InitBuffer` |
| 同步 | Queue slot 用 EnQue/DeQue/Free 维护 | 输出 GM 写完前不释放 output slot；不虚构 barrier | 各 `CopyIn/Compute/CopyOut` |
| 精度与尾块 | x 先 clamp 到 -3；FP16/BF16 升 FP32、`CAST_ROUND` 降精度；`currentNum` 控制全管线 | 不得恢复 `-inf*0`；不得让无效 tail lane 进入算术或 GM 写 | 各 `Compute/Process`；`DataCopyPad` |
| 失败边界 | 空 Tensor 成功短路；资源/dtype/`T<=0` 失败 | 固定实现没有可以自动代替的 fallback Route | `aclnnHardswishGetWorkspaceSize`；`HardSwishTilingFunc` |