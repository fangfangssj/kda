---
type: Operator
title: "Sigmoid A2 ACLNN 算子知识"
description: "说明 aclnnSigmoid 在 A2 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/docs/aclnnSigmoid&aclnnInplaceSigmoid.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_host/op_api/aclnn_sigmoid.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_host/op_api/sigmoid.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_host/sigmoid_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_host/sigmoid_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_kernel/sigmoid_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_kernel/sigmoid_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_kernel/sigmoid.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_kernel/sigmoid.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/sigmoid/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/sigmoid/docs/aclnnSigmoid&aclnnInplaceSigmoid.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/sigmoid/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/sigmoid/op_host/arch35/sigmoid_tiling_arch35.cpp
    role: implementation
created_at: "2026-08-25T18:56:41Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2]
cann_versions: ["9.1.0"]
---

# Sigmoid A2 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnSigmoidGetWorkspaceSize` + `aclnnSigmoid`，公开 ACLNN 两段式接口；采用活动 experimental 候选 `sigmoid_v2`。 |
| 计算语义 | 逐元素 `y = 1 / (1 + exp(-x))`。 |
| 输入输出 | `self/out` shape、dtype 相同，ND，rank 0–8；`FLOAT`、`FLOAT16`、`BFLOAT16`；支持非连续 Tensor 和空 Tensor。 |
| 执行类型 | 非空输入为单节点 `AIV_VECTOR`；空输入在 ACLNN Host 侧直接结束，不启动 Device Kernel。 |
| Kernel 路线 | R-001 FP32 原类型计算；R-002 FP16/BF16 升精度计算。 |
| 核心资源 | GM 输入/输出；一个绑定 VECIN/VECOUT 的双 buffer 区；R-001 一个 FP32 `TBuf`，R-002 两个 FP32 `TBuf`；Kernel 不读写传入 workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN→`l0op::Sigmoid`→Tiling/POD→dtype 模板 selector→`sigmoid` entry→两条实际 body→experimental CMake/`ascend910b` 注册闭合；`Contiguous/ViewCopy` 与生成 launcher 为明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `self` 的每个有效元素独立计算：

```text
t0 = -self[i]
t1 = exp(t0)
t2 = 1 + t1
out[i] = 1 / t2
```

FP32 路线全程以 FP32 计算；FP16/BF16 路线先升为 FP32，完成上述序列后以 `CAST_RINT` 降回输入 dtype。输出有效域与输入 shape 完全相同，没有跨元素依赖。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 与 `out` 相同；ND；rank 0–8 | `FLOAT`、`FLOAT16`、`BFLOAT16` | 支持空 Tensor 与非连续 Tensor；非空时先转连续。 |
| `out` | 与 `self` 相同；ND；rank 0–8 | 必须与 `self` 相同 | 可非连续；最终通过框架 `ViewCopy` 提交。 |
| `workspaceSize` | 标量 Host 输出 | `uint64_t` | 空输入固定写 0；非空为 executor 汇总的框架任务 workspace。 |
| `executor` | Host 对象 | `aclOpExecutor **` | 第一段创建并移交，第二段消费。 |
| `workspace` | Device 地址 | 字节区 | 由调用者按第一段返回值分配；Sigmoid Kernel entry 接收但不使用。 |
| `stream` | ACL runtime stream | — | `CommonOpExecutorRun` 在该 stream 执行 executor 中的任务。 |

### 入口与版本

公开入口是 `aclnnSigmoidGetWorkspaceSize`/`aclnnSigmoid`，本文不包含 `aclnnInplaceSigmoid` 或 `GluBackward`。固定仓同时存在 experimental `sigmoid` 与 `sigmoid_v2`；本卡采用目录版本名为 v2、可经 `ENABLE_EXPERIMENTAL` + `--ops=sigmoid_v2` 分流进入构建、且公开声明 A2 支持的实现。其 OpDef 仅注册 `ascend910b`，因此本卡平台限定为 A2。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnSigmoidGetWorkspaceSize` 先用 `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`，再依次执行：

1. `CheckNotNull`：`self/out` 均非空指针；失败返回 `ACLNN_ERR_PARAM_NULLPTR`。
2. `CheckDtypeValid`：两端均属于 `{FLOAT, FLOAT16, BFLOAT16}` 且 dtype 相同；失败返回 `ACLNN_ERR_PARAM_INVALID`。
3. `CheckShape`：shape 相同，且两端 rank 均不大于 8；失败返回 `ACLNN_ERR_PARAM_INVALID`。
4. 若 `self->IsEmpty()`，写 `workspaceSize=0`、移交空 executor 并成功返回，不建立 Sigmoid/Copy task。
5. 非空时依次加入 `l0op::Contiguous(self)`、`l0op::Sigmoid(selfContiguous)`、`l0op::ViewCopy(sigmoidOpOut, out)`；任一结果为空均终止。

`Contiguous/ViewCopy` 是仓外框架 Kernel 边界；它们处理非连续输入和输出。算子拥有的 `l0op::Sigmoid` 只分配同 shape/dtype 的 ND 临时输出并用 `ADD_TO_LAUNCHER_LIST_AICORE` 添加 `Sigmoid`。

### 参数、Workspace 与生命周期

第一段 `CREATE_EXECUTOR` 创建 owner；成功后 `ReleaseTo(executor)` 把生命周期交给调用者。非空输入的 `workspaceSize` 是 `uniqueExecutor->GetWorkspaceSize()`，包含框架 `Contiguous/ViewCopy` 及 Sigmoid tiling 声明的 system library workspace；后者由 `PlatformAscendC::GetLibApiWorkSpaceSize()` 写入 `context->GetWorkspaceSizes(1)[0]`。固定仓未展开 executor 的 workspace 聚合/切片细节，这是允许的框架边界。

第二段 `CommonOpExecutorRun(workspace, workspaceSize, executor, stream)` 按 executor 顺序在调用者 stream 上执行。`sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)` 不读取 `workspace`，所有算子计算中间量位于每个 AIV core 的 UB。

### 启动图与失败传播

```text
aclnnSigmoidGetWorkspaceSize
  -> [framework] Contiguous(self)                 # 仅非连续时可能产生搬运任务
  -> l0op::Sigmoid
       -> ADD_TO_LAUNCHER_LIST_AICORE(Sigmoid)
          -> Tiling: SigmoidTilingFunc
          -> Device: sigmoid<D_T_X>(x, y_tmp, workspace, tiling)
  -> [framework] ViewCopy(y_tmp, out)

aclnnSigmoid(workspace, size, executor, stream)
  -> CommonOpExecutorRun(..., stream)
```

核心语义节点只有一个 AIV Kernel。其输入为连续临时 `selfContiguous`，输出为 executor 分配的 `sigmoidOut`；框架随后把该临时张量写入用户 `out`。三节点按同一 executor/stream 顺序提交；任一 Host 建图步骤失败即不返回可执行 executor，运行阶段的错误由 `CommonOpExecutorRun` 传播。固定仓未展开框架临时张量的具体 workspace offset 和释放点。

## Route 与参数设计

### 分核与资源计算

设 `N` 为非标量 storage shape 元素积；标量 shape 被 `EnsureNotScalar` 转为 `{1}`。设 `S=sizeof(dtype)`，`C=GetCoreNumAiv()`，`U=UB bytes`：

```text
totalBytes = N * S
targetCoreNum = min(C, max(1, ceil(totalBytes / 16384)))
cacheLineElements = max(1, 512 / S)
perCoreRaw = ceil(N / targetCoreNum)
formerLength = align_up(perCoreRaw, cacheLineElements)
usedCoreNum = max(1, ceil(N / formerLength))
formerNum = usedCoreNum - 1
tailLength = N - formerNum * formerLength
alignElements = max(1, 32 / S)
tileLength = floor((U / 12) / alignElements) * alignElements
blockDim = usedCoreNum
```

`tileLength<=0` 时回退为 `alignElements`；正常硬件 UB 下该分支不命中。前 `formerNum` 个 core 各处理 `formerLength`，最后一个 core 处理 `tailLength`。每 core 的 `tileNum=ceil(blockLength/tileLength)`；末 tile 长度为 `blockLength-(tileNum-1)*tileLength`。`12 bytes/element` 同时精确覆盖两条 Route 的 UB 分配。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原类型 | `D_T_X=C_DT_FLOAT` | A2 且 ACLNN 校验后 `self/out=FLOAT`；dtype selector 生成 float 实例。 | dtype 为 16 位时进入 R-002；其它 dtype 在 API 校验失败。 |
| 2 | R-002 16 位升精度 | `D_T_X=C_DT_FLOAT16` 或 `C_DT_BF16` | A2 且 ACLNN 校验后 `self/out=FLOAT16` 或 `BFLOAT16`。两个 selector 实例的 body、UB、同步和 tail 等价。 | 无后续 Device fallback；其它 dtype 在 API 校验失败。 |

```text
if dtype == FLOAT:
    sigmoid<float> -> KernelSigmoid<float>                 # R-001
else if dtype == FLOAT16:
    sigmoid<half> -> KernelSigmoidUpcast<half>             # R-002
else if dtype == BFLOAT16:
    sigmoid<bfloat16_t> -> KernelSigmoidUpcast<bfloat16_t> # R-002
else:
    ACLNN_ERR_PARAM_INVALID
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原类型 | `D_T_X=C_DT_FLOAT` | A2；非空；rank≤8；输入输出 shape/dtype 相同且为 FP32 | 单 AIV 节点 `sigmoid<float>` | `KernelSigmoid<float>::ComputeBase` | 绑定队列双 buffer + 一个 FP32 ones `TBuf`；Queue 排序 | `DataCopyPad` + 有效 `curTileLength`；无 Device fallback |
| R-002 16 位升精度 | `D_T_X=C_DT_FLOAT16/C_DT_BF16` | A2；非空；rank≤8；输入输出 shape/dtype 相同且为 FP16/BF16 | 单 AIV 节点 `sigmoid<T>` | `KernelSigmoidUpcast<T>::ComputeBase` | 绑定队列双 buffer + 两个 FP32 `TBuf`；Queue 排序 | 同上；无 Device fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域 | API `self->IsEmpty()` | 避免 `N=0` 时启动无工作 Kernel | 全部 | Host 成功返回，`workspaceSize=0` | `op_host/op_api/aclnn_sigmoid.cpp:ExecSigmoidGetWorkspaceSize` |
| AIV core/UB 有效 | `coreNum>0 && ubSize>0` | 分核与 tile 公式的除数和上界 | 全部 | Tiling `GRAPH_FAILED` | `op_host/sigmoid_tiling.cpp:GetPlatformInfo` |
| 每核最少 16 KiB | `targetCoreNum=ceil(totalBytes/16384)` 后钳位 | 避免对小张量启用过多 core | 全部 | 不是 fallback；直接减少 `usedCoreNum` | `op_host/sigmoid_tiling.cpp:FillNormalTiling` |
| 512 B core 边界 | `formerLength` 按 `512/S` 元素向上对齐 | 保持前核 GM 切片为 512 B 边界 | 全部 | 最后核吸收真实 tail | `op_host/sigmoid_tiling.cpp:FillNormalTiling` |
| 32 B tile 对齐 | `tileLength` 按 `32/S` 元素向下对齐 | 满足常规 DMA block 对齐，尾 tile 交给 pad API | 全部 | 极小 UB 时置为一个 `alignElements` | `op_host/sigmoid_tiling.cpp:FillNormalTiling` |
| dtype 模板闭集 | selector 仅 FLOAT/FLOAT16/BF16 | entry 只有原类型与升精度两种实例 | R-001/R-002 | API 参数错误或 Tiling dtype 错误 | `op_kernel/sigmoid_tiling_key.h:ASCENDC_TPL_ARGS_DECL/SEL` |
| DMA `blockLen` 窄化 | `curTileLength*S` 转 `uint32_t` | `DataCopyExtParams::blockLen` 为 32 位字节数 | 全部 | 固定仓未提供单独 fallback；tile 受 UB 上界限制，实际远小于 2^32 B | `op_kernel/sigmoid.h:CopyInBase/CopyOutBase` |

### POD/ABI

`SigmoidTilingData` 是四个连续 `int64_t` 的标准布局结构：`sizeof=32`、`alignof=8`，各字段 offset 分别为 0/8/16/24。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `SigmoidTilingData` | `formerNum` | `int64_t`/64 | offset 0；struct size 32/align 8 | core 数 | `usedCoreNum - 1` | `InitBase`: 判断 former/tail core，并计算 tail base | 非空 ACLNN 路线为 `[0, usedCoreNum-1]` |
| `SigmoidTilingData` | `formerLength` | `int64_t`/64 | offset 8 | 元素/core | `align_up(ceil(N/targetCoreNum), 512/S)` | `InitBase`: former block extent 与 `formerLength*blockIdx` | 可能大于均分值；最后核不读取为 extent |
| `SigmoidTilingData` | `tailLength` | `int64_t`/64 | offset 16 | 元素 | `N-formerNum*formerLength` | `InitBase`: 最后 core extent | 非空路线大于 0 |
| `SigmoidTilingData` | `tileLength` | `int64_t`/64 | offset 24 | 元素/tile | `align_down(U/12, 32/S)`；非正时取 `32/S` | `InitBase` 分配 UB；`ProcessBase` 计算 loop/offset | 必须大于 0；正常值受 UB 限制 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 原类型 | `D_T_X=C_DT_FLOAT` | `op_host/sigmoid_tiling.cpp:SigmoidTilingFunc/FillNormalTiling` | `SigmoidTilingData` 四字段 | `op_kernel/sigmoid.cpp:sigmoid<float>` | `op_kernel/sigmoid.h:KernelSigmoidBase<float,...>/KernelSigmoid<float>` | `op_host/CMakeLists.txt:add_modules_sources`；`sigmoid_def.cpp:AddConfig("ascend910b")` |
| R-002 16 位升精度 | `D_T_X=C_DT_FLOAT16/C_DT_BF16` | 同上 | 同上 | `op_kernel/sigmoid.cpp:sigmoid<half/bfloat16_t>` | `op_kernel/sigmoid.h:KernelSigmoidBase<T,...>/KernelSigmoidUpcast<T>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 用 `GET_TILING_DATA_WITH_STRUCT` 解析 32 B POD。编译期 `sizeof(D_T_X)==sizeof(float)` 选择 R-001，否则选择 R-002。两条路线共享 CRTP 基类：按 `GetBlockIdx()` 绑定本 core 的 GM 切片，按 POD `tileLength` 初始化绑定队列；每 tile 执行 `CopyInBase -> ComputeBase -> CopyOutBase`。`DataCopyPad` 的 `blockCount=1`、`blockLen=curTileLength*S`、两侧 stride=0；末 tile 只搬有效字节。

### Route R-001：FP32 原类型计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `experimental/activation/sigmoid_v2/op_host/op_api/aclnn_sigmoid.cpp` | `CheckDtypeValid` | dtype 属闭集且 `self==out dtype` | 允许进入 launcher |
| Tiling | `experimental/activation/sigmoid_v2/op_host/sigmoid_tiling.cpp` | `GetShapeAttrsInfo` | `dataType==DT_FLOAT` | selector 写入 FP32 dtype 参数 |
| Selector | `experimental/activation/sigmoid_v2/op_kernel/sigmoid_tiling_key.h` | `ASCENDC_TPL_SEL` | `D_T_X=C_DT_FLOAT` | 实例化 `sigmoid<float>` |
| Router | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.cpp` | `sigmoid<D_T_X>` | `sizeof(D_T_X)==sizeof(float)` | `KernelSigmoid<float>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Sigmoid -> ADD_TO_LAUNCHER_LIST_AICORE(Sigmoid)
  -> sigmoid<float>(x, y, workspace, tiling)
    -> KernelSigmoid<float>::Init
      -> KernelSigmoidBase::InitBase
    -> KernelSigmoidBase::Process
      -> ProcessBase
        -> CopyInBase
        -> KernelSigmoid<float>::ComputeBase
        -> CopyOutBase
```

##### 源码等价伪码

```text
# 伪码
if blockIdx < formerNum:
    blockLength = formerLength
    base = formerLength * blockIdx
else:
    blockLength = tailLength
    base = formerLength * formerNum
bind xGm=x+base, yGm=y+base

for tile in [0, ceil(blockLength/tileLength)):
    len = tileLength except final tile uses exact tail
    xLocal = alloc bound queue slot
    DataCopyPad(xLocal, xGm[tile*tileLength], bytes=len*4)
    enqueue GM->VECIN; dequeue
    xLocal = -xLocal
    xLocal = exp(xLocal)
    xLocal = xLocal + 1
    oneLocal = 1
    xLocal = oneLocal / xLocal
    enqueue VECOUT->GM; dequeue
    DataCopyPad(yGm[tile*tileLength], xLocal, bytes=len*4)
    free slot
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` core slice | `base` 如上 | `blockLength` 元素 | FP32/ND flat | framework contiguous output | `CopyInBase` | executor 临时；Kernel 只读 | former base 512 B 对齐；tail 精确 |
| GM `y` core slice | 同一 `base` | `blockLength` 元素 | FP32/ND flat | `CopyOutBase` | framework `ViewCopy` | executor 临时，ViewCopy 后可释放 | core 间不重叠 |
| `inOutQueue_` | tile-local 0 | `2 * tileLength * 4` B | FP32 | `CopyInBase`，随后 Compute 原地覆写 | `ComputeBase/CopyOutBase` | `TPipe`，tile 完成后 `FreeTensor` | `TQueBind<VECIN,VECOUT,1>`；`InitBuffer(...,2,...)` 双块 |
| `tmpBufOne_` | 独立 TBuf | `tileLength * 4` B | FP32 | `Duplicate(1)` | `Div` numerator | Kernel/core 全生命周期 | 每 tile 覆盖 |
| workspace | entry 第 3 参数 | Kernel 侧 0 B | — | Host/framework | Kernel 不消费 | 调用者/executor | N/A |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  io0["inOutQueue block 0 · tileLength×4 B"]:1
  io1["inOutQueue block 1 · tileLength×4 B"]:1
  one["tmpBufOne · tileLength×4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * tileLength * 4 + tileLength * 4 = 12 * tileLength bytes
tileLength <= floor((UB_size / 12) / (32/4)) * (32/4)
dynamic_ub = N/A
workspace_total_for_sigmoid_kernel = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bound queue 当前 slot | `CopyInBase` | `ComputeBase` 原地更新，继而 `CopyOutBase` | slot `FreeTensor` 后的后续 tile | GM→VECIN EnQue/DeQue，VECOUT→GM EnQue/DeQue，最后 Free | 不能在 GM copy-out 完成前 Free/重分配 |
| `tmpBufOne_` | 每 tile `Duplicate` | 同 tile `Div` | 下一 tile `Duplicate` | 同一 Compute 调用内顺序 | `Div` 完成前不可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `TQueBind` EnQue/DeQue | `DataCopyPad` | `ComputeBase` dequeue | 当前 input slot | 精确 `curTileLength*4` B，pad API 处理非整 block tail |
| Vector | 同一函数顺序 | Muls→Exp→Adds→Duplicate→Div | 下一 primitive | xLocal/oneLocal | 全 FP32；只对 `curTileLength` 个元素执行 |
| UB→GM | `TQueBind` EnQue/DeQue | `ComputeBase` enqueue | `CopyOutBase` dequeue | 原地结果 slot | 只写有效 tail；无 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/sigmoid_v2/op_host/sigmoid_tiling.cpp` | `FillNormalTiling` | 16 KiB/core、512 B core 对齐、12 B/element UB 与 POD 写入 |
| Kernel entry/router | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.cpp` | `sigmoid<float>` | `sizeof(float)==sizeof(float)` 选择原类型 body |
| Device body/helper | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.h` | `KernelSigmoidBase`、`KernelSigmoid<float>::ComputeBase` | core/tile 地址、绑定队列、原地 FP32 计算和 tail DMA |

### Route R-002：FP16/BF16 升精度计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `experimental/activation/sigmoid_v2/op_host/op_api/aclnn_sigmoid.cpp` | `CheckDtypeValid` | dtype 为 FP16/BF16 且输入输出相同 | 允许进入 launcher |
| Tiling | `experimental/activation/sigmoid_v2/op_host/sigmoid_tiling.cpp` | `GetShapeAttrsInfo` | `dataType==DT_FLOAT16 \|\| DT_BF16` | selector 写对应 16 位 dtype |
| Selector | `experimental/activation/sigmoid_v2/op_kernel/sigmoid_tiling_key.h` | `ASCENDC_TPL_SEL` | `D_T_X=C_DT_FLOAT16/C_DT_BF16` | 实例化对应 `sigmoid<T>` |
| Router | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.cpp` | `sigmoid<D_T_X>` | `sizeof(D_T_X)!=sizeof(float)` | `KernelSigmoidUpcast<T>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Sigmoid -> ADD_TO_LAUNCHER_LIST_AICORE(Sigmoid)
  -> sigmoid<half|bfloat16_t>(x, y, workspace, tiling)
    -> KernelSigmoidUpcast<T>::Init
      -> KernelSigmoidBase::InitBase
    -> KernelSigmoidBase::Process
      -> ProcessBase
        -> CopyInBase
        -> KernelSigmoidUpcast<T>::ComputeBase
        -> CopyOutBase
```

##### 源码等价伪码

```text
# 伪码；core/base/tile 循环与 R-001 相同，S=2
xLocal<T> = DataCopyPad(xGm, bytes=len*2)
xFp32 = Cast<T -> float>(xLocal, CAST_NONE, len)
xFp32 = -xFp32
xFp32 = exp(xFp32)
xFp32 = xFp32 + 1
tmpFp32 = 1
xFp32 = tmpFp32 / xFp32
xLocal = Cast<float -> T>(xFp32, CAST_RINT, len)
DataCopyPad(yGm, xLocal, bytes=len*2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x/y` core slice | 与 R-001 相同 | `blockLength` 元素 | FP16 或 BF16/ND flat | framework / `CopyOutBase` | `CopyInBase` / ViewCopy | executor 临时 | former base 512 B 对齐，tail 精确 |
| `inOutQueue_` | tile-local 0 | `2 * tileLength * 2` B | FP16/BF16 | CopyIn；最终 cast 回写同一 slot | Cast source / CopyOut | `TPipe`，tile 完成后 Free | 输入消费后作为输出复用 |
| `tmpBufXFp32_` | 独立 TBuf | `tileLength * 4` B | FP32 | 首次 Cast 与后续原地 Vector | 最终 Cast source | Kernel/core 全生命周期 | 全计算链原地复用 |
| `tmpBufFp32_` | 独立 TBuf | `tileLength * 4` B | FP32 | `Duplicate(1)` | `Div` numerator | Kernel/core 全生命周期 | 每 tile 覆盖 |
| workspace | entry 第 3 参数 | Kernel 侧 0 B | — | Host/framework | Kernel 不消费 | 调用者/executor | N/A |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  io["inOutQueue · 2×tileLength×2 B"]:2
  x32["tmpBufXFp32 · tileLength×4 B"]:1
  one32["tmpBufFp32 · tileLength×4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * tileLength * 2 + tileLength * 4 + tileLength * 4
         = 12 * tileLength bytes
tileLength <= floor((UB_size / 12) / (32/2)) * (32/2)
dynamic_ub = N/A
workspace_total_for_sigmoid_kernel = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bound queue 当前 slot | `CopyInBase` | 首次 Cast；最终 Cast 回写；`CopyOutBase` | Free 后下一 tile | 两次 queue transition + Free | 首次 Cast 已消费输入后才可回写；GM copy-out 完成后才可 Free |
| `tmpBufXFp32_` | Cast | Muls/Exp/Adds/Div 原地更新，最终 Cast 读取 | 下一 tile Cast | 同一 Compute 顺序 | 最终 Cast 完成前不可覆盖 |
| `tmpBufFp32_` | Duplicate | Div | 下一 tile Duplicate | 同一 Compute 顺序 | Div 完成前不可覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + bound Queue | CopyIn | Compute dequeue | 16 位 input slot | 有效字节 `len*2` |
| 升精度 | `Cast(..., CAST_NONE)` | xLocal | FP32 Muls | `tmpBufXFp32_` | FP16/BF16→FP32 |
| FP32 计算 | Muls→Exp→Adds→Duplicate→Div | 前一 primitive | 后一 primitive | 两个 FP32 TBuf | mask 长度由 `curTileLength` 隐式传入 API |
| 降精度 | `Cast(..., CAST_RINT)` | FP32 result | queue enqueue | 原 16 位 slot | FP32→FP16/BF16；round mode 精确保留为源码枚举 `CAST_RINT` |
| UB→GM | bound Queue + `DataCopyPad` | Compute enqueue | CopyOut dequeue | 16 位 result slot | 仅写有效 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/sigmoid_v2/op_host/sigmoid_tiling.cpp` | `FillNormalTiling` | 两种 16 位 dtype 共享 12 B/element UB 预算 |
| Kernel entry/router | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.cpp` | `sigmoid<half/bfloat16_t>` | `sizeof(T)!=4` 选择 upcast body |
| Device body/helper | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.h` | `KernelSigmoidUpcast<T>::ComputeBase` | 升精度、FP32 Sigmoid、`CAST_RINT` 降精度与 buffer 复用 |

## 算子特定优化

### Trick T-001：绑定输入/输出队列并在升精度路线回写原输入 slot

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：`D_T_X` 为 FP16 或 BF16，entry 选择 `KernelSigmoidUpcast<T>`。
- Golden 来源：[`experimental/activation/sigmoid_v2/op_kernel/sigmoid.h:147-159`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/sigmoid_v2/op_kernel/sigmoid.h#L147-L159)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
LocalTensor<T> xLocal = this->inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
LocalTensor<float> xFp32 = tmpBufXFp32_.Get<float>();
LocalTensor<float> tmpLocal = tmpBufFp32_.Get<float>();

Cast(xFp32, xLocal, RoundMode::CAST_NONE, curTileLength);
Muls(xFp32, xFp32, -1.0f, curTileLength);
Exp(xFp32, xFp32, curTileLength);
Adds(xFp32, xFp32, 1.0f, curTileLength);
Duplicate(tmpLocal, 1.0f, curTileLength);
Div(xFp32, tmpLocal, xFp32, curTileLength);
Cast(xLocal, xFp32, RoundMode::CAST_RINT, curTileLength);

this->inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(xLocal);
```

#### 机制与设计原因

输入 16 位 slot 在第一次 Cast 后已不再承载活跃输入，最终降精度结果直接写回同一 `xLocal`，并通过 `TQueBind` 从 VECIN 状态转为 VECOUT 状态。这样无需另建输出 Queue；两块 FP32 TBuf 分别承载原地计算链和常量 1。Host 用统一 `12 bytes/element` 同时覆盖 `2×2 B` 绑定队列与 `2×4 B` FP32 TBuf，tile 上界与真实分配闭合。

#### 迁移边界

- 适用：输入在转换后不再读取、输出 dtype/shape 与输入相同的逐元素升精度算子。
- 前提：A2 AIV；FP16/BF16；Queue 状态转换与 Cast/Vector/CopyOut 严格按源码顺序；UB 预算至少为 `12*tileLength`。
- 失效：FP32 使用 R-001 的单 FP32 TBuf；输入后续仍需保留或输入输出大小不同时不能直接复用该 slot。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从生命周期证据可确定，在首次 Cast 消费前覆盖 `xLocal`，或在 CopyOut 完成前释放/复用 slot，都会破坏当前实现依赖的数据顺序。

## 模板与实现对比

| 对比维度 | R-001 FP32 原类型 | R-002 16 位升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | dtype=FP32 | dtype=FP16/BF16 | 编译期 dtype selector + `sizeof(D_T_X)` router |
| 模板与 dtype | `KernelSigmoid<float>` | `KernelSigmoidUpcast<half/bfloat16_t>` | 16 位先升精度 |
| 循环与分核 | 同一 16 KiB/core、512 B core 对齐、32 B tile 对齐 | 同左 | dtype size 进入元素对齐换算 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 绑定队列 2×FP32 + 1×FP32 TBuf | 绑定队列 2×16 位 + 2×FP32 TBuf | 两路线均为 12 B/element；无 dynamic UB/通信 |
| 内存总占用与复用 | xLocal 原地 Vector；ones 每 tile 覆盖 | FP32 结果原地；结果回写 xLocal | 避免独立输出 Queue |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad；Muls/Exp/Adds/Duplicate/Div | 同左 + 两次 Cast | 均为 AIV Vector，无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue；FP32；有效长度 tail | Queue；FP32 计算，CAST_RINT；有效长度 tail | DataCopyPad 覆盖非对齐尾块 |
| fallback/失败边界 | 无 Device fallback | 无 Device fallback | 非支持 dtype 在 ACLNN/Tiling 失败；空输入 Host 结束 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；`sigmoid_v2` 文档 + `ascend910b` OpDef + experimental 构建闭合 | 不发布：experimental 与 stable ACLNN 文档声称支持，但 `sigmoid_v2` 仅注册 `ascend910b`，且 stable README 明确 A3 不支持，证据冲突 | 不发布：stable README/CMake 明确 Ascend 950 与 `arch35`，但固定仓缺少直接依赖 `ElewiseTiling` 的实现，未达 `source_closed` |
| Host/Launcher | 自包含 `SigmoidTilingFunc` | 无对应活动平台注册可闭合 | `activation/sigmoid/op_host/arch35/SigmoidTiling`，依赖仓外 `op_common/atvoss/elewise/ElewiseTiling` |
| Kernel/资源 | 本卡两条 AIV Route | 不生成平台卡 | stable MicroAPI 三 dtype body 可见，但 tile/core/POD producer 公式未闭合 |

## 限制与失败边界

- 仅覆盖非原地公开入口 `aclnnSigmoid`；不覆盖 `aclnnInplaceSigmoid`、Grad 或 `GluBackward`。
- 仅 A2 发布。A3 存在公开文档与平台注册/稳定 README 冲突，不能据文档单独发布。
- 950 的 stable Host 直接调用固定仓不可见的 `Ops::NN::Optiling::ElewiseTiling`；`innerKey`、`blockNum`、八个 POD 字段的具体公式和 UB 上界无法追溯，因此未生成 `sigmoid_950.md`。
- API 的 `N*S`、core/tile 公式使用 `int64_t`，固定仓未见显式乘法溢出检查；DMA `blockLen` 窄化为 `uint32_t`，但单 tile 由 UB 限制。
- `FillNormalTiling` 在 `typeLength==0` 时仅从 `void` 函数返回；该情况已由前置 dtype 闭集排除。
- executor workspace 的框架切片、`Contiguous/ViewCopy` 实体和生成 launcher bridge 为固定仓外框架边界；没有将其推测为 0。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `experimental/activation/sigmoid_v2/docs/aclnnSigmoid&aclnnInplaceSigmoid.md` | `aclnnSigmoid*` 章节 | A2 公开支持、dtype/shape/rank/空域/非连续约束 |
| 入口实现 | `experimental/activation/sigmoid_v2/op_host/op_api/aclnn_sigmoid.cpp` | `CheckParams`、`ExecSigmoidGetWorkspaceSize`、`aclnnSigmoid` | 校验、空域、Contiguous/Sigmoid/ViewCopy、executor |
| Launcher | `experimental/activation/sigmoid_v2/op_host/op_api/sigmoid.cpp` | `l0op::Sigmoid` | 临时输出与 AICore launcher 添加 |
| 平台注册 | `experimental/activation/sigmoid_v2/op_host/sigmoid_def.cpp` | `ops::Sigmoid` | 三 dtype ND 与 `ascend910b` 注册 |
| Host/Tiling | `experimental/activation/sigmoid_v2/op_host/sigmoid_tiling.cpp` | `SigmoidTilingFunc`、`FillNormalTiling` | 平台资源、workspace、分核、tile、selector |
| POD | `experimental/activation/sigmoid_v2/op_kernel/sigmoid_tiling_data.h` | `SigmoidTilingData` | 四字段 ABI |
| Selector | `experimental/activation/sigmoid_v2/op_kernel/sigmoid_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三 dtype 模板实例 |
| Kernel entry | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.cpp` | `sigmoid<D_T_X>` | dtype router |
| Kernel body | `experimental/activation/sigmoid_v2/op_kernel/sigmoid.h` | `KernelSigmoidBase`、`KernelSigmoid`、`KernelSigmoidUpcast` | core/tile、DMA、Vector、UB、同步、精度、tail |
| 构建接线 | `experimental/activation/sigmoid_v2/op_host/CMakeLists.txt` | `add_modules_sources` | `OPTYPE sigmoid` experimental 活动模块 |
| 构建总开关 | `CMakeLists.txt` | `ENABLE_EXPERIMENTAL` | experimental 模块由显式构建开关启用 |
| 构建目录分流 | `cmake/func.cmake` | `add_category_subdirectory` | 开关为真时扫描 `experimental/<category>`，并按目录名选择 `sigmoid_v2` |
| 平台审计 | `activation/sigmoid/README.md` | 产品支持表 | stable 声明 950/A2 支持、A3 不支持 |
| 平台审计 | `activation/sigmoid/docs/aclnnSigmoid&aclnnInplaceSigmoid.md` | 产品支持表 | stable ACLNN 文档声明 A3 支持，形成公开证据冲突 |
| 平台审计 | `activation/sigmoid/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | stable 活动实现仅 `ascend950/mc62 -> arch35` |
| 950 缺口 | `activation/sigmoid/op_host/arch35/sigmoid_tiling_arch35.cpp` | `SigmoidTiling::DoOpTiling` | 直接调用固定仓不可见 `ElewiseTiling`，证明 950 闭包缺口 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 四个连续 `int64_t` 可直接复用 | 字段顺序、32 B size、元素单位不可改变 | `SigmoidTilingData` |
| 调用顺序 | `Contiguous -> Sigmoid -> ViewCopy`；Kernel 内 `CopyIn -> Compute -> CopyOut` | 空输入不得启动 Kernel；失败不得继续建图 | `ExecSigmoidGetWorkspaceSize`、`ProcessBase` |
| workspace/内存 | 两路线统一 12 B/element UB 预算 | `InitBuffer` 的双块必须计入；Kernel workspace 与 executor workspace 不可混同 | `FillNormalTiling`、各 `Init` |
| 同步 | 绑定 Queue 的 GM→VECIN→VECOUT→GM 状态流可复用 | EnQue/DeQue/Free 次序不可改变 | `CopyInBase/CopyOutBase/ComputeBase` |
| 精度与尾块 | FP32 原计算；16 位升 FP32 后 `CAST_RINT`；DataCopyPad 有效 tail | 不能删除两次 Cast 或以分配容量代替 `curTileLength` | 两个 `ComputeBase`、`ProcessBase` |
| 失败边界 | 参数错误、core/UB 无效、Tiling 空指针均有显式失败 | 不得为不支持 dtype/A3 构造 fallback | `CheckParams`、`GetPlatformInfo`、`SigmoidTilingFunc` |