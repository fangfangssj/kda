---
type: Operator
title: "Swish A2 ACLNN 算子知识"
description: "说明 aclnnSwish 在 A2 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/docs/aclnnSwish.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_host/op_api/aclnn_swish.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_host/op_api/aclnn_swish.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_host/op_api/swish.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_host/swish_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_host/swish_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_kernel/swish_v2_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_kernel/swish_v2_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_kernel/swish_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_kernel/swish_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:59:58Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2]
cann_versions: ["9.1.0"]
---

# Swish A2 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnSwishGetWorkspaceSize` + `aclnnSwish`，公开两段式 ACLNN；A2 采用当前仓中语义相同而实现版本更新的内部 OpType `SwishV2`。 |
| 计算语义 | 逐元素计算 `y = x / (1 + exp(-scale * x))`；`betaOptional == nullptr` 时 `scale=1.0f`。 |
| 输入输出 | `self/out` 为 ND、0～8 维、shape 与 dtype 相同，支持空 Tensor 和非连续 Tensor；A2 支持 FP32/FP16/BF16。 |
| 执行类型 | 核心语义节点为单 Kernel `swish_v2`，`AIV_VECTOR`；ACLNN 外围 `Contiguous`、`ViewCopy` 为框架边界；无可达 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP32 原生计算；R-002 FP16/BF16 升到 FP32 计算后 `CAST_RINT` 回写。 |
| 核心资源 | GM 输入/输出；绑定 VECIN/VECOUT 的双块 `TQueBind`；R-001 两个同 dtype `TBuf`，R-002 三个 FP32 `TBuf`；核心 Kernel 仅请求平台系统 workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN→executor 编排→`SwishV2` launcher→A2 OpDef→Tiling/完整 POD→dtype 模板→entry/body→experimental CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对逻辑元素 `i in [0,N)`：

```text
scale = betaOptional == nullptr ? 1.0f : betaOptional->ToFloat()
y[i] = x[i] / (1 + exp(-scale * x[i]))
```

Kernel 将任意 0～8 维 ND Tensor 展平成长度 `N` 的连续一维域。FP16/BF16 路线先把 `x[i]` 转为 FP32，整个乘、取负、`Exp`、加一、倒数和乘原输入过程均使用 FP32，最后以 `RoundMode::CAST_RINT` 转回原 dtype。本文不包含语义不同的 Silu、Backward 或 inplace 入口。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，0～8 维 | FP32/FP16/BF16 | 不可为 null；允许空 Tensor、标量和非连续 Tensor；与 `out` shape/dtype 相同。 |
| `betaOptional` | Host scalar | 可转换为 FLOAT | 可为 null；null 时取 `1.0f`，否则用 `ToFloat()` 生成 `scale`。 |
| `out` | 与 `self` 相同 | 与 `self` 相同 | 不可为 null；允许非连续 Tensor，最终通过 `ViewCopy` 回写。 |
| `workspaceSize` | Host 标量 | `uint64_t*` | 第一段接口返回整个 executor 所需 Device workspace，而非仅核心 Kernel 的 workspace。 |
| `executor` | Host handle | `aclOpExecutor**` | 第一段创建并移交所有权，第二段在指定 stream 执行。 |

### 入口与版本

公开入口原型位于 `experimental/activation/swish_v2/op_host/op_api/aclnn_swish.h`，实现直接调用内部 `l0op::SwishV2`，不是旧 `Swish` body 的别名。该候选相较同仓 `experimental/activation/swish/` 使用显式 `SwishV2` OpType、独立 POD 和完整 dtype router，因此作为 A2 当前实际可达的最新实现。公开文档列出 A2，`SwishV2` OpDef 的活动 AICore 注册为 `ascend910b`；没有把仅文档列出的 A3/950 误并入本卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnSwishGetWorkspaceSize` 的顺序是：

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize/executor`；`CREATE_EXECUTOR` 创建 executor。
2. `CheckParams` 依次检查 `self/out` 非空，dtype 均在 FP32/FP16/BF16 且相同，`betaOptional` 可转换到 FP32，rank 不超过 8 且 shape 相同。
3. 任一 Tensor 为空时直接返回 `workspaceSize=0`，executor 中不加入核心 Kernel。
4. `l0op::Contiguous` 将输入变为连续 Tensor；这是通用框架边界。
5. 将可选 scalar 归一为 FP32 `scale`；`l0op::SwishV2` 分配连续中间输出并把一个 AICore 节点加入 launcher list。
6. `l0op::ViewCopy` 把连续中间结果写入用户 `out` 的 view/stride；任一步返回 null 都终止第一段。
7. 读取 executor 聚合 workspace，移交 executor；第二段 `CommonOpExecutorRun` 在用户 stream 上按 executor 顺序执行并传播状态码。

### 参数、Workspace 与生命周期

`l0op::SwishV2` 以 `self->GetViewShape()`、输入 dtype 和 ND format 分配 `swishOut`，其生命周期由 executor 持有到第二段完成。Tiling 的 `GetWorkspaceSize` 把核心 `SwishV2` 节点的唯一 workspace 项设为 `PlatformAscendC::GetLibApiWorkSpaceSize()`；`swish_v2` entry 接收 `workspace` 但 body 不解引用它。ACLNN 返回的 `workspaceSize` 还会聚合可能存在的 `Contiguous/ViewCopy` 框架节点和中间 Tensor，固定仓不展开 executor 的通用切片公式，调用者必须使用第一段原值且保持 workspace 到第二段 stream 完成。

### 启动图与失败传播

```text
aclnnSwishGetWorkspaceSize
  -> [需要时] framework Contiguous(self -> selfContiguous)
  -> SwishV2 AICore node(selfContiguous, scale -> swishOut)
       entry: swish_v2<D_T_X>(x, y, workspace, tiling)
       blockDim: usedCoreNum
       POD: SwishV2TilingData
  -> framework ViewCopy(swishOut -> out)
aclnnSwish(workspace, workspaceSize, executor, stream)
  -> CommonOpExecutorRun（同一 executor/stream 顺序）
```

核心节点是单 Kernel，无算子自有跨 Kernel workspace 边。`selfContiguous` 与 `swishOut` 均由 executor 分配并在后继节点消费完之前存活；launcher 添加失败或任一通用节点返回 null 时第一段不返回可执行的成功路径，第二段的执行失败由 `CommonOpExecutorRun` 原样返回。

## Route 与参数设计

### 分核与资源计算

设 `N=EnsureNotScalar(storageShape).GetShapeSize()`，`s=sizeof(dtype)`，A2 AIV 核数为 `C`，可用 UB 字节为 `U`：

```text
totalBytes = N * s
targetCoreNum = clamp(ceil(totalBytes / 16384), 1, C)
cacheLineElements = max(1, 512 / s)
candidate = ceil(N / targetCoreNum)
formerLength = align_up(candidate, cacheLineElements)
usedCoreNum = max(1, ceil(N / formerLength))
formerNum = usedCoreNum - 1
tailLength = N - formerNum * formerLength
alignElements = max(1, 32 / s)
tileLength = floor((U / 16) / alignElements) * alignElements
blockDim = usedCoreNum
```

`tileLength<=0` 时回退为 `alignElements`；正常平台必须由 `U` 保证实际 UB 分配可容纳它。前 `formerNum` 个核各处理 `formerLength` 元素，最后一个核处理 `tailLength`。每核 `tileNum=ceil(blockLength/tileLength)`，前 `tileNum-1` 个 tile 为整块，最后一块为 `blockLength-(tileNum-1)*tileLength`。512 B 分核边界避免相邻核起点落在同一缓存线；32 B tile 对齐满足常规搬运粒度，末 tile 由 `DataCopyPad` 处理非对齐长度。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原生 | `D_T_X=C_DT_FLOAT` → `swish_v2<float>` | ACLNN dtype 检查通过且 `self.dtype=DT_FLOAT` | dtype 为 FP16/BF16 时进入 R-002；其它 dtype 第一段失败 |
| 2 | R-002 16-bit 升精度 | `D_T_X=C_DT_FLOAT16`/`C_DT_BF16` → `swish_v2<half/bfloat16_t>` | ACLNN dtype 检查通过且 dtype 分别为 FP16/BF16 | 其它 dtype 第一段失败 |

```text
if self/out null -> ACLNN_ERR_PARAM_NULLPTR
if dtype unsupported/mismatch, beta不可转FP32, rank>8, shape mismatch -> ACLNN_ERR_PARAM_INVALID
if self/out empty -> success, workspace=0, no SwishV2 launch
template dtype == FLOAT -> R-001 KernelSwishV2<float>
template dtype == FLOAT16 or BF16 -> R-002 KernelSwishV2Upcast<T>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 | `ASCENDC_TPL_DATATYPE_SEL(D_T_X,C_DT_FLOAT)` | A2/ascend910b，非空，FP32，rank≤8，shape 相同 | 单 `swish_v2<float>` AIV 节点 | `KernelSwishV2<float>::Compute` | 双块绑定 Queue + 2 个 FP32 TBuf；Queue 保序；UB 内原地覆盖 `xLocal` | `DataCopyPad` + `curTileLength` 控制尾块；无 Device fallback |
| R-002 16-bit 升精度 | `...C_DT_FLOAT16` / `...C_DT_BF16` | A2/ascend910b，非空，FP16/BF16，rank≤8，shape 相同 | 单 `swish_v2<T>` AIV 节点 | `KernelSwishV2Upcast<T>::Compute` | 双块绑定 Queue + 3 个 FP32 TBuf；FP32 计算、RINT 回写 | 同上；不支持 dtype 在 ACLNN/tiling 失败 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台注册 | `AICore().AddConfig("ascend910b",...)` | 只有该活动 config 把 `SwishV2` 接到 A2 AICore | 全部 | 其它代际不由本卡实现保证 | `swish_v2_def.cpp::SwishV2` |
| 最小每核工作量 | `targetCoreNum=ceil(totalBytes/16KiB)` 后限制到 `[1,C]` | 避免按所有 AIV 核切出过小工作块 | 全部 | 自动减少 blockDim，不失败 | `FillNormalTiling` |
| 512 B 分核对齐 | `formerLength=align_up(ceil(N/targetCoreNum),512/s)` | 核边界按 cache-line 元素数对齐 | 全部 | 无另一 Route | `FillNormalTiling` |
| UB 上界 | `tileLength=align_down(U/16,32/s)` | 两路线的真实 UB 总量均为 `16*tileLength` 字节 | 全部 | 结果≤0 时置 `32/s`；平台 UB 过小无独立错误保护 | `FillNormalTiling`、`InitExtraBuffers` |
| 空域 | ACLNN `self->IsEmpty() \|\| out->IsEmpty()` | 避免 `N=0` 的无意义 Device launch | 全部 | 成功返回、workspace 0 | `aclnnSwishGetWorkspaceSize` |
| dtype router | `sizeof(D_T_X)==sizeof(float)` | FP32 直接走 native body；16-bit 必须升精度 | R-001/R-002 | 编译期二选一 | `swish_v2` entry |

### POD/ABI

`SwishV2TilingData` 是顺序字段 POD；固定仓没有 `static_assert(sizeof/offsetof)`。按源码字段类型的自然布局，字段 offset 为 0/8/16/24/32，整体通常为 40 B、8 B 对齐；复用时必须以实际 CANN Host/Device ABI 为准，不能自行重排或压缩。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `SwishV2TilingData` | `formerNum` | `int64_t`/64 | offset 0 | core 数 | `usedCoreNum-1` | `InitBlockGm` 判断当前核与计算尾核 offset | `>=0`；正常非空路径 `<blockDim` |
| `SwishV2TilingData` | `formerLength` | `int64_t`/64 | offset 8 | 元素 | `align_up(ceil(N/targetCoreNum),512/s)` | 前核 extent、所有核 GM 基址步长 | 512 B 对齐对应元素数 |
| `SwishV2TilingData` | `tailLength` | `int64_t`/64 | offset 16 | 元素 | `N-formerNum*formerLength` | 最后核 GM extent | 正常非空路径 `>0` |
| `SwishV2TilingData` | `tileLength` | `int64_t`/64 | offset 24 | 元素 | `align_down(U/16,32/s)` | Queue/TBuf 容量、tile loop 步长 | 正常应 `>0` 且按 32 B 元素数对齐 |
| `SwishV2TilingData` | `scale` | `float`/32 | offset 32 | 无量纲 | 属性指针或默认 `1.0f` | `InitBlockGm` 保存，Compute 的 `Muls` 使用 | 不过滤 NaN/Inf |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 | `D_T_X=C_DT_FLOAT` | `swish_v2_tiling.cpp::SwishV2TilingFunc/FillNormalTiling` | `SwishV2TilingData` 全字段 | `swish_v2.cpp::swish_v2<float>` | `swish_v2.h::KernelSwishV2<float>` + `KernelSwishV2Base` + `ProcessTiles` | `swish_v2_def.cpp` ascend910b；两级 CMake |
| R-002 16-bit 升精度 | `D_T_X=C_DT_FLOAT16/C_DT_BF16` | 同上 | 同上 | `swish_v2.cpp::swish_v2<half/bfloat16_t>` | `swish_v2.h::KernelSwishV2Upcast<T>` + base + `ProcessTiles` | 同上 |

## Kernel 实现

### 公共执行流程

两条 Route 共享单 Kernel launch graph。entry 解析 `SwishV2TilingData`，再以 `sizeof(D_T_X)==sizeof(float)` 编译期选择 native 或 upcast 类。基类 `Init` 根据 `GetBlockIdx()` 绑定本核 GM 子区，设置 `scale_/tileLength_`，分配绑定 VECIN/VECOUT 的两个 Queue buffer；派生类再分配计算 TBuf。`ProcessTiles` 严格执行 `CopyIn -> Compute -> CopyOut`，整 tile 后单独执行尾 tile。Queue 的 `EnQue/DeQue/FreeTensor` 建立 GM→VEC、VEC→GM 阶段顺序；无跨核共享写、atomic 或算子自有 workspace 交接。

### Route R-001：FP32 原生计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `experimental/activation/swish_v2/op_host/op_api/aclnn_swish.cpp` | `CheckDtypeValid` | dtype in `{FLOAT,FLOAT16,BF16}` 且输入输出相同 | FP32 继续 |
| Launcher | `.../op_host/op_api/swish.cpp` | `l0op::SwishV2` | `ADD_TO_LAUNCHER_LIST_AICORE(SwishV2,...)` | 单 AICore 节点 |
| Tiling | `.../op_host/swish_v2_tiling.cpp` | `SetTemplateParam` | `dTypeX=DT_FLOAT` | 选择 FLOAT 模板 |
| Router | `.../op_kernel/swish_v2.cpp` | `swish_v2<D_T_X>` | `sizeof(D_T_X)==sizeof(float)` | `KernelSwishV2<float>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::SwishV2 -> ADD_TO_LAUNCHER_LIST_AICORE(SwishV2)
  -> swish_v2<float>(x,y,workspace,tiling)
    -> KernelSwishV2<float>::Init
      -> KernelSwishV2Base::InitBlockGm
      -> InitBuffer(inOutQueue_,2,tileLength*4)
      -> KernelSwishV2::InitExtraBuffers
    -> KernelSwishV2Base::Process
      -> ProcessTiles
        -> CopyIn -> KernelSwishV2::Compute -> CopyOut
```

##### 源码等价伪码

```text
# 源码等价伪码
blockLength = blockIdx < formerNum ? formerLength : tailLength
gmBase = blockIdx < formerNum ? formerLength*blockIdx : formerLength*formerNum
for tile in [0, ceil(blockLength/tileLength)):
    len = tile is last ? blockLength-(tileNum-1)*tileLength : tileLength
    xLocal = DataCopyPad(xGm[gmBase + tile*tileLength], len*4 bytes)
    xInput = xLocal                         # Adds(...,0) 保存原输入
    xLocal = xLocal * scale
    xLocal = xLocal * -1
    xLocal = exp(xLocal)
    xLocal = xLocal + 1
    oneLocal = 1
    xLocal = oneLocal / xLocal
    xLocal = xInput * xLocal
    DataCopyPad(yGm[gmBase + tile*tileLength], xLocal, len*4 bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `xGm_` | `base + progress*tileLength` | 本核 `blockLength` | FP32/flat ND | ACLNN input/Contiguous | `CopyIn` | 用户/executor；launch 全程 | 核基址 512 B 对齐，尾 DMA 可不对齐 |
| `inOutQueue_` | Queue 管理 | `2*tileLength*4 B` | FP32 | `CopyIn` | `Compute` 后同一 tensor 给 `CopyOut` | `AllocTensor` 至 `FreeTensor` | `BUFFER_NUM=2`；同一 LocalTensor 原地转为输出 |
| `tmpBufInput_` | 独立 TBuf | `tileLength*4 B` | FP32 | `Adds(xInput,xLocal,0)` | 最后 `Mul` | 单 Compute | 保存原输入，下一 tile 覆盖 |
| `tmpBufOne_` | 独立 TBuf | `tileLength*4 B` | FP32 | `Duplicate(1)` | `Div` | 单 Compute | 下一 tile 覆盖 |
| GM `yGm_` | 同 GM x 的 tile offset | 有效 `len` 元素 | FP32/flat ND | `CopyOut` | 后继 ViewCopy | executor 中间 Tensor | 只写有效字节 |
| workspace | entry 参数 | `GetLibApiWorkSpaceSize()` | opaque | runtime | AscendC 库边界 | launch | 算子 body 不读写 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  ioQ["inOutQueue · 2 blocks · 2×tileLength×4 B"]:2
  input["tmpBufInput · tileLength×4 B"]:1
  one["tmpBufOne · tileLength×4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*tileLength*4 + tileLength*4 + tileLength*4
         = 16*tileLength bytes
tileLength <= floor((U/16)/(32/4))*(32/4)
UB_total <= U
dynamic_ub = N/A（AIV TPipe 静态分配）
workspace_total(core node) = GetLibApiWorkSpaceSize()
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inOutQueue_` 当前 tensor | `CopyIn` | `Compute`，随后 `CopyOut` | `FreeTensor` 后下一 tile | GM→VEC `EnQue/DeQue`，VEC→GM `EnQue/DeQue` | Compute 入队输出前不释放；CopyOut 后才 Free |
| `xLocal` | GM CopyIn；Compute 连续原地写 | 下一 Vector primitive / CopyOut | 当前指令完成后原地覆盖 | 同一 Vector pipeline 与 Queue 阶段 | 原输入先复制到 `tmpBufInput_` |
| `tmpBufInput_` | `Adds(...,0)` | 最终 `Mul` | 下一 tile Compute | 单 Compute 程序序 | 最终乘前不得覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + Queue | `CopyIn` | Compute `DeQue` | 当前输入 tile | `blockLen=len*4`，末 tile 仅 `len` 有效 |
| 计算 | 同一 AIV pipeline | `Adds/Muls/Exp/Adds/Duplicate/Div/Mul` | 后续 primitive | `xLocal/tmpBuf*` | 全 FP32；Vector API count=`len` |
| UB→GM | Queue + `DataCopyPad` | Compute `EnQue` | CopyOut `DeQue` | 当前输出 tile | 只提交 `len*4` 字节 |
| 跨核 | 不需要 | 每核独占 GM slice | N/A | 输出域 | `formerNum/tailLength` 保证无重叠 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/swish_v2/op_host/swish_v2_tiling.cpp` | `FillNormalTiling` | 分核、tile、blockDim、dtype 模板参数 |
| Kernel entry/router | `experimental/activation/swish_v2/op_kernel/swish_v2.cpp` | `swish_v2<float>` | `sizeof(float)` 分派 native 类 |
| Device body/helper | `experimental/activation/swish_v2/op_kernel/swish_v2.h` | `KernelSwishV2<float>::Compute` / `ProcessTiles` | 原地 Vector 管线、Queue 生命周期、tail |

### Route R-002：FP16/BF16 升精度计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `experimental/activation/swish_v2/op_host/op_api/aclnn_swish.cpp` | `CheckDtypeValid` | dtype 为 FP16 或 BF16，输入输出相同 | 继续；A2 OpDef 同时注册两者 |
| Tiling | `.../op_host/swish_v2_tiling.cpp` | `SetTemplateParam` | `dTypeX=DT_FLOAT16/DT_BF16` | 对应模板 key |
| Router | `.../op_kernel/swish_v2.cpp` | `swish_v2<D_T_X>` | `sizeof(D_T_X)!=sizeof(float)` | `KernelSwishV2Upcast<T>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::SwishV2 -> ADD_TO_LAUNCHER_LIST_AICORE(SwishV2)
  -> swish_v2<half|bfloat16_t>
    -> KernelSwishV2Upcast<T>::Init
      -> base InitBlockGm + InitBuffer(inOutQueue_,2,tileLength*2)
      -> InitExtraBuffers(3×tileLength×FP32)
    -> ProcessTiles
      -> CopyIn -> KernelSwishV2Upcast<T>::Compute -> CopyOut
```

##### 源码等价伪码

```text
# 源码等价伪码；分核、地址和 tile/tail 与 R-001 相同
xLocal<T> = DataCopyPad(xGm[offset], len*2 bytes)
xFp32 = Cast<T->float>(xLocal, CAST_NONE, len)
xInputFp32 = xFp32
xFp32 = xFp32 * scale
xFp32 = xFp32 * -1.0f
xFp32 = exp(xFp32) + 1.0f
oneFp32 = 1.0f
xFp32 = oneFp32 / xFp32
xFp32 = xInputFp32 * xFp32
xLocal<T> = Cast<float->T>(xFp32, CAST_RINT, len)
DataCopyPad(yGm[offset], xLocal, len*2 bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `coreBase+progress*tileLength` | `blockLength` / 有效 `len` | FP16 或 BF16 / flat ND | framework / CopyOut | CopyIn / ViewCopy | launch | 核边界 512 B；尾搬运 len×2 B |
| `inOutQueue_` | Queue 管理 | `2*tileLength*2 B` | T | CopyIn；末尾 Cast 回写 | Compute/CopyOut | Alloc→Free | 输入与输出复用同一 tensor |
| `tmpBufXFp32_` | 独立 TBuf | `tileLength*4 B` | FP32 | 首次 Cast，随后原地管线 | 每个后续 Vector primitive | 单 Compute | 原地覆盖 |
| `tmpBufInputFp32_` | 独立 TBuf | `tileLength*4 B` | FP32 | `Adds(...,0)` | 最终 Mul | 单 Compute | 保存升精度原输入 |
| `tmpBufOneFp32_` | 独立 TBuf | `tileLength*4 B` | FP32 | Duplicate | Div | 单 Compute | 下一 tile 覆盖 |
| workspace | entry 参数 | 库系统 workspace | opaque | runtime | 库边界 | launch | body 不读写 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  ioQ["inOutQueue · 2 blocks · 2×tileLength×2 B"]:2
  x32["xFp32 · tileLength×4 B"]:1
  input32["inputFp32 · tileLength×4 B"]:1
  one32["oneFp32 · tileLength×4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*tileLength*2 + 3*tileLength*4
         = 16*tileLength bytes
UB_total <= U
dynamic_ub = N/A（AIV TPipe 静态分配）
workspace_total(core node) = GetLibApiWorkSpaceSize()
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inOutQueue_` 当前 tensor | CopyIn；末尾降精度 Cast | 首次 Cast；CopyOut | Free 后下一 tile | 两次 Queue EnQue/DeQue | FP32 计算完成前不回写 |
| `tmpBufXFp32_` | 升精度 Cast | 连续 Vector primitives；最终降精度 Cast | primitive 后原地覆盖/下一 tile | 同一 pipeline 程序序 | 原值先复制至 inputFp32 |
| `tmpBufInputFp32_` | `Adds(...,0)` | 最终 Mul | 下一 tile | 单 Compute 程序序 | 最终 Mul 前保持 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入 | `DataCopyPad` + Queue | GM | Compute DeQue | T 输入 tile | blockLen=`len*2`，padding lane 不进入 count |
| 升精度 | `Cast(...CAST_NONE,len)` | `xLocal` | FP32 管线 | `xFp32` | FP16/BF16→FP32 |
| 计算 | FP32 Vector API | Cast/各前序 primitive | 下一 primitive | 三个 FP32 TBuf | count=`len`；尾 lane 不参与 |
| 降精度 | `Cast(...CAST_RINT,len)` | FP32 result | Queue output | `xLocal<T>` | FP32→FP16/BF16，RINT |
| 搬出 | Queue + `DataCopyPad` | `xLocal<T>` | GM | 输出有效域 | 仅 `len*2` 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/swish_v2/op_host/swish_v2_tiling.cpp` | `FillNormalTiling` | 同一保守 16 B/元素 UB 预算覆盖 16-bit upcast 路线 |
| Kernel entry/router | `experimental/activation/swish_v2/op_kernel/swish_v2.cpp` | `swish_v2<half/bfloat16_t>` | `sizeof(T)!=4` 选择 upcast body |
| Device body/helper | `experimental/activation/swish_v2/op_kernel/swish_v2.h` | `KernelSwishV2Upcast<T>::Compute` | FP32 中间、RINT 回写、Queue/tail 生命周期 |

## 算子特定优化

### Trick T-001：统一 16 B/元素 UB 预算精确覆盖两种 body

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：正常非空 Tiling；`tileLength=align_down(ubSize/16,32/sizeof(dtype))`。
- Golden 来源：[`experimental/activation/swish_v2/op_host/swish_v2_tiling.cpp:131-136`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_host/swish_v2_tiling.cpp#L131-L136)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    int64_t maxTileElements = static_cast<int64_t>(ubSize) / BUFFER_COEFFICIENT;
    int64_t alignElements = std::max<int64_t>(1, 32 / dtypeSize);
    tiling->tileLength = (maxTileElements / alignElements) * alignElements;
    if (tiling->tileLength <= 0) {
        tiling->tileLength = alignElements;
    }
```

#### 机制与设计原因

R-001 每元素需要 Queue 双块 `2×4 B` 加两个 FP32 TBuf `2×4 B`，合计 16 B；R-002 每元素需要 Queue 双块 `2×2 B` 加三个 FP32 TBuf `3×4 B`，同样合计 16 B。Host 用同一个 `BUFFER_COEFFICIENT=16` 直接把 UB 字节预算变成最大元素数，再按 32 B DMA 对应元素数向下对齐，既共享 Tiling 公式又精确覆盖最重的真实分配，不需要按 dtype 维护两套 UB guard。

#### 迁移边界

- 适用：具有 FP32 native 与 16-bit upcast 两类 body，且两者逐元素 UB 字节恰好相同的逐元素算子。
- 前提：Queue buffer 数、TBuf 数量和 dtype 与本实现一致；UB 由 `GetCoreMemSize(UB)` 给出，tile 按 32 B 对齐。
- 失效：新增 scratch、改变双缓冲数或引入更宽中间 dtype 时必须重新计算系数；不能继续沿用 16。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：Queue tensor 原地承载分母、倒数和最终输出

#### 触发与源码依据

- 适用 Route：R-001；R-002 同样复用 Queue tensor 作为降精度输出载体。
- 触发条件：当前 tile 已从 Queue DeQue，原输入已复制到 `tmpBufInput_`。
- Golden 来源：[`experimental/activation/swish_v2/op_kernel/swish_v2.h:120-134`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/swish_v2/op_kernel/swish_v2.h#L120-L134)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
        LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        LocalTensor<T> xInput = tmpBufInput_.Get<T>();
        LocalTensor<T> oneLocal = tmpBufOne_.Get<T>();

        T scaleValue = static_cast<T>(this->scale_);
        Adds(xInput, xLocal, static_cast<T>(0), curTileLength);
        Muls(xLocal, xLocal, scaleValue, curTileLength);
        Muls(xLocal, xLocal, static_cast<T>(-1), curTileLength);
        Exp(xLocal, xLocal, curTileLength);
        Adds(xLocal, xLocal, static_cast<T>(1), curTileLength);
        Duplicate(oneLocal, static_cast<T>(1), curTileLength);
        Div(xLocal, oneLocal, xLocal, curTileLength);
        Mul(xLocal, xInput, xLocal, curTileLength);

        this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
```

#### 机制与设计原因

输入 Queue tensor 在保存一份原输入后，被连续原地改写为 `scale*x`、`-scale*x`、指数、分母、倒数和最终输出，并直接作为 VECOUT 入队。这样没有为每个数学中间量物化独立 UB Tensor；唯一必须长期保留的是最终乘法所需原输入和 Div 的常量一。关键 Vector API 的 `curTileLength` 来自尾 tile 计算，保证原地覆盖只涉及有效元素。

#### 迁移边界

- 适用：单消费者的逐元素链，原输入只在链末再使用一次。
- 前提：先把原输入复制到独立 TBuf；Queue 支持 VECIN/VECOUT bind；所有原地 primitive 对 dtype 和 count 合法。
- 失效：若后续还需要指数前的多个版本或存在并行消费者，必须增加 buffer 或重排生命周期。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 原生 | R-002 FP16/BF16 升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | dtype=FP32 | dtype=FP16/BF16 | 编译期 `sizeof(D_T_X)` |
| 模板与 dtype | `KernelSwishV2<float>` | `KernelSwishV2Upcast<T>` | 16-bit 显式 FP32 中间 |
| 循环与分核 | 相同 `ProcessTiles` | 相同 | Tiling 与地址公式共享 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2×FP32 Queue + 2×FP32 TBuf | 2×T Queue + 3×FP32 TBuf | 两者均 16 B/元素；无 dynamic UB/通信 |
| 内存总占用与复用 | `16*tileLength`，Queue 原地 | `16*tileLength`，FP32 scratch 原地、Queue 回写 | 统一 Host UB 系数 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad；Adds/Muls/Exp/Div/Mul | 另加双向 Cast | 均为 AIV_VECTOR，无其它执行族 |
| 同步、精度与 tail | Queue；全 FP32；count=tail | Queue；FP32 计算、RINT 回写；count=tail | 精度由 dtype route 决定 |
| fallback/失败边界 | 无 Device fallback | 无 Device fallback | 非支持 dtype 在 ACLNN/Tiling 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本卡实现公开支持并以 `ascend910b` 活动注册 | `swish_v2` 文档声称支持，但固定仓无 `ascend910_93`/A3 注册或 config，未发布 | `swish_v2` 文档声称支持但无 950 注册；稳定 `activation/swish` 有 950 接线，但依赖的 atvoss POD/scheduler 源码不在固定仓，未发布 |
| Host/Launcher | `aclnnSwish`→`SwishV2` | 固定仓无法闭合实际活动实现 | 稳定入口→`Swish`，但 kernel scheduler 闭包缺失 |
| Kernel/资源 | 本文两条 AIV Route | 固定仓不可见 | 稳定 entry 为 AIV-only，完整调度/UB/POD 不可见 |

## 限制与失败边界

- ACLNN 拒绝 null Tensor、dtype 不支持或不匹配、`betaOptional` 不可转 FP32、rank>8、shape 不同；空 Tensor 成功短路且不启动核心 Kernel。
- Tiling 的 `totalLength*dtypeSize`、分核公式使用 `int64_t`，固定仓未见显式乘法溢出检查；Device DMA `blockLen` 窄化为 `uint32_t`，正常 tile 受 UB 上界限制，但没有独立静态断言。
- `scale` 不做有限值检查；NaN/Inf 的数值行为由 FP32 primitive 决定，固定仓未声明额外语义。
- POD 没有 `static_assert` 固化 `sizeof/alignof/offsetof`；迁移必须保持 Host/Device 相同编译 ABI。
- A3 缺少活动 `ascend910_93` 构建/注册闭包；950 稳定实现的 `EleBaseTilingData32B`、`ElewiseBaseTiling::DoTiling32B`、`ElementwiseSchWithScalar` 定义来自固定仓未包含的 `atvoss` 头，无法完成逐字段 POD、真实 UB/循环和同步闭包，因此未生成 `swish_950.md`。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `experimental/activation/swish_v2/docs/aclnnSwish.md` | `aclnnSwish` | 公开产品、参数、空/非连续 Tensor 约束 |
| 入口声明/约束 | `experimental/activation/swish_v2/op_host/op_api/aclnn_swish.h` | 两段式原型 | C ABI 参数 |
| 入口实现 | `experimental/activation/swish_v2/op_host/op_api/aclnn_swish.cpp` | `CheckParams`、`aclnnSwishGetWorkspaceSize` | 校验、Contiguous、空域、scale、ViewCopy、executor |
| 入口实现 | `experimental/activation/swish_v2/op_host/op_api/swish.cpp` | `l0op::SwishV2` | 中间输出分配、AICore launcher |
| Host/Tiling/Launcher | `experimental/activation/swish_v2/op_host/swish_v2_def.cpp` | `ops::SwishV2` | dtype/format/attr 与 ascend910b 注册 |
| Host/Tiling/Launcher | `experimental/activation/swish_v2/op_host/swish_v2_tiling.cpp` | `SwishV2TilingFunc`、`FillNormalTiling` | shape、分核、tile、workspace、POD producer |
| Host/Tiling/Launcher | `experimental/activation/swish_v2/op_kernel/swish_v2_tiling_data.h` | `SwishV2TilingData` | 完整 POD ABI |
| Host/Tiling/Launcher | `experimental/activation/swish_v2/op_kernel/swish_v2_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个 dtype 模板选择 |
| Kernel entry/body | `experimental/activation/swish_v2/op_kernel/swish_v2.cpp` | `swish_v2<D_T_X>` | entry 与 native/upcast router |
| Kernel entry/body | `experimental/activation/swish_v2/op_kernel/swish_v2.h` | `ProcessTiles`、`KernelSwishV2Base`、两个派生类 | GM/UB、Vector 管线、同步、精度、tail |
| 构建接线 | `experimental/activation/swish_v2/op_host/CMakeLists.txt` | `add_modules_sources(...swish_v2...)` | Host 模块与 op type 接线 |
| 构建接线 | `experimental/activation/swish_v2/CMakeLists.txt` | 子目录枚举 | experimental 活动构建入口 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 五字段足以还原分核、tile 和 scale | 字段顺序、int64/float 位宽、元素单位 | `SwishV2TilingData` |
| 调用顺序 | `Contiguous→SwishV2→ViewCopy`；每 tile `CopyIn→Compute→CopyOut` | 空域短路和失败必须阻断后续 | `aclnnSwishGetWorkspaceSize`、`ProcessTiles` |
| workspace/内存 | 每 Route 恰为 16 B/元素 UB 预算 | Queue buffer 数、TBuf dtype/数量、32 B 对齐 | `FillNormalTiling`、`Init/InitExtraBuffers` |
| 同步 | Queue bind 在 GM/VECIN/VECOUT/GM 间传递同一 tensor | EnQue/DeQue/FreeTensor 次序不可改变 | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | FP32 native；16-bit FP32 计算后 RINT；尾块按有效 count | Cast 顺序、`curTileLength`、末 tile 单独处理 | `KernelSwishV2*::Compute`、`ProcessTiles` |
| 失败边界 | 非法参数在第一段失败；空 Tensor 无 Kernel；无 Device fallback | 不得把 A3/950 文档声明当成活动实现闭包 | `CheckParams`、`swish_v2_def.cpp` |
