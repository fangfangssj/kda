---
type: Operator
title: "MishBackward A2/A3 ACLNN 算子知识"
description: "说明 aclnnMishBackward 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/docs/aclnnMishBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_host/op_api/aclnn_mish_backward_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_host/op_api/mish_grad_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_host/mish_grad_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_host/mish_grad_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish_grad/docs/aclnnMishBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish_grad/op_host/arch35/mish_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish_grad/op_kernel/mish_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/gen_ops_info.cmake
    role: implementation
created_at: "2026-08-25T18:52:15Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# MishBackward A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN `aclnnMishBackwardGetWorkspaceSize` / `aclnnMishBackward`；本卡采用活动实验构建中的最新 `MishGradV2` body，对外符号不带 `V2`。 |
| 计算语义 | `gradInput = gradOutput * (t + self * (1 - t²) / (1 + exp(-self)))`，其中 `t = tanh(relu(self) + ln(1 + exp(-abs(self))))`。 |
| 输入输出 | `gradOutput`、`self`、`gradInput` 均为 ND、同 shape、同 dtype；A2/A3 支持 `FLOAT32/FLOAT16/BFLOAT16`，维数不超过 8；ACLNN 支持空 Tensor 和非连续 Tensor。 |
| 执行类型 | 单 Kernel；唯一 Device 节点为 `AIV_VECTOR`。`FLOAT32` 使用原生 FP32 body，`FLOAT16/BFLOAT16` 使用 FP32 计算的 cast body。 |
| Kernel 路线 | R-001 `FP32_DIRECT`；R-002 `FP16_BF16_CAST`。空 Tensor 在 ACLNN Host 侧直接成功返回，不启动 Kernel。 |
| 核心资源 | GM 输入/输出；双缓冲 `VECIN/VECOUT` Queue；按 Route 条件分配的 FP32 `TBuf`；Kernel 不读 `workspace`，Host workspace 由 executor 和平台系统库 workspace 共同管理。 |
| 来源闭包 | `source_closed`：公开 ACLNN → `l0op::MishGradV2` → `MishGradV2TilingFunc`/`MishGradV2TilingData` → dtype 模板选择 → `mish_grad_v2` → 两个实际 body 全部闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `g = gradOutput`、`x = self`，先以稳定 softplus 形式重算正向中间量：

```text
sp = relu(x) + ln(1 + exp(-abs(x)))
t  = tanh(sp)
gradInput = g * (t + x * (1 - t * t) / (1 + exp(-x)))
```

公式逐元素成立，输出有效域严格等于输入 shape。公开 ACLNN 不接收 `tanhx`；`l0op::MishGradV2(..., nullptr, ...)` 使本卡所有可达 Route 都在 Kernel 内重算 `t`。源码中的可选 `tanhx` 分支属于内部 Host 可达能力，但不属于此公开入口的 Route。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | ND，0–8 维 | `FLOAT32/FLOAT16/BFLOAT16` | 上游梯度；必须与 `self`、`gradInput` shape 和 dtype 完全一致；支持空 Tensor 与非连续 Tensor。 |
| `self` | ND，0–8 维 | 同 `gradOutput` | Mish 正向输入；不支持 broadcast 或隐式类型提升。 |
| `gradInput` | ND，0–8 维 | 同 `gradOutput` | 输出；可为非连续 Tensor，由 `ViewCopy` 写回。 |
| `workspaceSize` | 标量 Host 输出 | `uint64_t` | 第一段接口从 executor 读取；空 Tensor 固定为 0。 |
| `executor` | Host 对象 | `aclOpExecutor*` | 第一段接口构造并移交所有权，第二段接口消费其任务图。 |
| `stream` | Device stream | `aclrtStream` | 第二段接口在该 stream 上运行任务图。 |

### 入口与版本

固定 Commit 中存在旧候选 `experimental/activation/mish_grad/` 与新候选 `experimental/activation/mish_grad_v2/`。本卡采用后者：其 README 明确说明目录后缀仅区分新实现，对外仍导出 `aclnnMishBackward`，并给出 `build.sh --experimental --ops=mish_grad_v2` 的活动构建入口；CMake 将 `op_host` 以 `OPTYPE mish_grad_v2` 接入。公开文档明确列出 A2、A3 支持。旧候选仅公开列出 A2，且使用旧的 broadcast/type-promotion 语义，未混入本版本族。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecMishBackwardV2GetWorkspaceSize` 的顺序为：

1. `CREATE_EXECUTOR()`，失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
2. `CheckNotNull` 检查三个 Tensor 指针；`CheckDtypeValid` 按平台支持表检查 dtype 并要求三者相同；`CheckShape` 检查最大 8 维并要求三者 shape 相同。
3. `grad` 或 `x` 为空时，将 `workspaceSize=0`、移交 executor 并成功返回；此时无 Device 节点。
4. 非空时依次添加 `Contiguous(grad)`、`Contiguous(x)`；这是框架连续化边界。
5. 调用 `l0op::MishGradV2(gradContiguous, xContiguous, nullptr, executor)`。内部按 `grad` 的 storage shape/dtype/format 分配临时输出并添加一个 AICore launcher。
6. 添加 `ViewCopy(result, xGrad)`，支持非连续输出。
7. 从 executor 取 workspace 大小并移交 executor。第二段接口以 `CommonOpExecutorRun` 在用户 stream 上执行任务图。

任一步返回空指针或 launcher 添加失败都会立即返回错误，不再追加后续节点。

### 参数、Workspace 与生命周期

`MishGradV2TilingFunc` 从 `PlatformAscendC` 读取 AIV 核数和 UB 字节数，并把平台 `GetLibApiWorkSpaceSize()` 写入该 Kernel 节点的 workspace size。Kernel entry 的 `workspace` 参数未被读取。ACLNN 返回的总 `workspaceSize = uniqueExecutor->GetWorkspaceSize()` 还包含 `Contiguous`/`ViewCopy` 等框架任务所需空间；其精确聚合与切片布局位于固定仓外的 executor 框架边界，不能由本算子源码给出常数公式。

第一段接口创建的 executor 持有连续化临时 Tensor、`MishGradV2` 临时输出、launcher 参数与任务顺序，`ReleaseTo(executor)` 后由调用者传给第二段接口。输出 `gradInput` 和用户 workspace 由调用者持有，至少须覆盖 `aclnnMishBackward` 异步提交期间；完成时间由调用者对同一 stream 同步确定。

### 启动图与失败传播

```text
aclnnMishBackwardGetWorkspaceSize
  -> [必要时] framework Contiguous(gradOutput)
  -> [必要时] framework Contiguous(self)
  -> l0op::MishGradV2
       -> ADD_TO_LAUNCHER_LIST_AICORE(MishGradV2)
          -> mish_grad_v2<D_T_X>(grad, x, nullptr, tmpOut, workspace, tiling)
  -> framework ViewCopy(tmpOut, gradInput)

aclnnMishBackward
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

| 节点 | Host 添加函数 | entry/执行族 | blockDim/POD | 输入→输出 | workspace/ordering/失败 |
|---|---|---|---|---|---|
| N0 | `l0op::Contiguous` | 框架边界 | executor 内部 | 用户 `gradOutput` → 连续 `grad` | 与后续任务在 executor/同一 stream 顺序执行；返回空则第一段失败。 |
| N1 | `l0op::Contiguous` | 框架边界 | executor 内部 | 用户 `self` → 连续 `x` | 同上。 |
| N2 | `ADD_TO_LAUNCHER_LIST_AICORE(MishGradV2, ...)` | `mish_grad_v2<D_T_X>` / `AIV_VECTOR` | `usedCoreNum`；`MishGradV2TilingData` | 连续 `grad,x` → 临时连续 `x_grad` | 节点系统 workspace=`GetLibApiWorkSpaceSize()`，entry 不读 workspace；launcher 添加失败则第一段失败。 |
| N3 | `l0op::ViewCopy` | 框架边界 | executor 内部 | 临时连续结果 → 用户 `gradInput` | 消费 N2 完整结果；返回空则第一段失败。 |

N2→N3 的交接对象是 executor 分配的、shape/dtype/format 与 `grad` storage 描述一致的临时 Tensor；offset 为 0，extent 为 `N * sizeof(dtype)`。固定源码没有显式 event，顺序由同一 executor 在传入 stream 上的任务序列保证。`CommonOpExecutorRun` 的返回状态传播任一运行失败。空 Tensor 路径不创建 N0–N3。

## Route 与参数设计

### 分核与资源计算

对非空 Tensor，令：

```text
N = gradStorageShape.GetShapeSize()                         # 元素
C = max(AIVCoreNum, 1)                                     # core
s = sizeof(dtype)                                          # 字节/元素
A512 = max(1, 512 / s)                                     # 元素
L0 = ceil(N / C)                                           # 元素/core
L  = ceil(L0 / A512) * A512                                # formerLength，512B 对齐
U  = max(1, ceil(N / L))                                   # usedCoreNum=blockDim
formerNum    = U - 1
formerLength = L
tailLength   = N - (U - 1) * L
A32 = max(1, 32 / s)                                       # 元素
tileLength = floor((UB_bytes / 36) / A32) * A32            # 元素
if tileLength <= 0: tileLength = A32
```

block `b < formerNum` 处理 `[b*formerLength, (b+1)*formerLength)`；最后一个 block 处理从 `formerNum*formerLength` 开始的 `tailLength`。每核 `tileNum=ceil(blockLength/tileLength)`。最后 tile 的 `validLength` 可能非 32B 对齐，`alignedLength=ceil(validLength/A32)*A32`；GM→UB 用零填充，UB→GM 只写 `validLength*s` 字节。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 `FP32_DIRECT` | `D_T_X=C_DT_FLOAT` → `mish_grad_v2<float>` | API 校验通过、非空且 input 0 dtype 为 `FLOAT32`。`ASCENDC_TPL_SEL_PARAM(context, dTypeX)` 直接以 input dtype 选择。 | dtype 为 FP16/BF16 时进入 R-002；其它 dtype 在 API 或 Tiling 失败。 |
| 2 | R-002 `FP16_BF16_CAST` | `D_T_X=C_DT_FLOAT16/C_DT_BF16` → `mish_grad_v2<half/bfloat16_t>` | API 校验通过、非空且 input 0 dtype 为 `FLOAT16` 或 `BFLOAT16`。两 key 的 Queue/TBuf、计算和尾块行为相同，仅模板类型不同。 | 其它 dtype 失败，无 Device fallback。 |

```text
if any required tensor pointer is null: ACLNN_ERR_PARAM_NULLPTR
if dtype/shape/rank invalid: ACLNN_ERR_PARAM_INVALID
if grad.IsEmpty() || x.IsEmpty(): success, workspaceSize=0, no kernel
switch grad.dtype:
  FLOAT32          -> R-001 mish_grad_v2<float>
  FLOAT16/BFLOAT16 -> R-002 mish_grad_v2<T>, T∈{half,bfloat16_t}
  otherwise        -> API/Tiling failure
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 `FP32_DIRECT` | `C_DT_FLOAT` | A2/A3；三个 Tensor 同 shape、同 FP32、rank≤8、非空 | 单节点 N2 `AIV_VECTOR` | `mish_grad_v2<float>` → `KernelMishGradFloat::{Init,Process,CopyIn,Compute}` | grad/x/out 三个 depth=2 Queue；`tmpBuf0/tmpBuf1`；Queue producer-consumer 同步 | 32B 零填充输入，按有效字节输出；无 fallback。 |
| R-002 `FP16_BF16_CAST` | `C_DT_FLOAT16/C_DT_BF16` | A2/A3；三个 Tensor 同 shape、同 FP16 或 BF16、rank≤8、非空 | 单节点 N2 `AIV_VECTOR` | `mish_grad_v2<T>` → `KernelMishGradCast<T>` | 三个 depth=2 Queue；五个固定 FP32 TBuf 加条件 `tmpBuf1`；FP32 计算后 round cast | 32B 零填充输入，按有效字节输出；无 fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台资源有效 | `coreNum>0 && ubSize>0` | 分核和 `tileLength` 均依赖平台资源。 | 两 Route | `GRAPH_FAILED` | `mish_grad_v2_tiling.cpp::GetPlatformInfo` |
| 512B 分核粒度 | `formerLength=align_up(ceil(N/C),512/s)` | 除最后核外，各核起点保持 cache-line 粒度。 | 两 Route | 固定公式，无替代 Route。 | `FillTilingData` |
| UB 保守系数 | `tileLength=floor((ubSize/36)/A32)*A32` | R-002 无 `tanhx` 时正好需要 `36*tileLength` 字节；R-001 需要 `32*tileLength`。 | 两 Route | `tileLength<=0` 时退为一个 32B 对齐块；源码未再次证明该异常资源下可装入 UB。 | `FillTilingData` |
| 同 shape/dtype | grad/x/tanhx（若有）storage shape 与 dtype 相同 | Kernel 无 broadcast/转换地址逻辑。公开 ACLNN 固定无 `tanhx`。 | 两 Route | API 或 Tiling 失败。 | `CheckShape`、`GetShapeAttrsInfo` |
| 32B DMA tail | `alignedLength=align_up(validLength,32/s)` | `DataCopyPad` 以 32B lane 补零，Vector 计算覆盖对齐长度。 | 两 Route | 输出只提交有效字节，无额外 Route。 | `AlignUp`、`CopyIn`、`CopyOutTile` |
| dtype 模板集 | `FLOAT32/FLOAT16/BFLOAT16` | entry 只有三个显式模板实例。 | 两 Route | API/Tiling 失败；生成模板机制的数值 key 固定仓不可见。 | `mish_grad_v2_tiling_key.h::ASCENDC_TPL_SEL` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `MishGradV2TilingData` | `formerNum` | `int64_t/64` | 声明序第 1；固定仓无 `static_assert(sizeof/offsetof)`，精确 offset 不外推 | core | `usedCoreNum-1` | 两 body 的 `Init`：`blockIdx<formerNum` 与 tail 起点 | 非空时≥0；公开空域不 launch。 |
| `MishGradV2TilingData` | `formerLength` | `int64_t/64` | 声明序第 2；精确 offset 固定仓不可见 | 元素/core | `totalLengthCoreAlign` | `Init` 的 GM offset、former core extent 和 tail 起点 | 512B 对齐的元素数。 |
| `MishGradV2TilingData` | `tailLength` | `int64_t/64` | 声明序第 3；精确 offset 固定仓不可见 | 元素 | `N-formerNum*formerLength` | 最后 core 的 GM extent | 非空路径 `1..formerLength`。 |
| `MishGradV2TilingData` | `tileLength` | `int64_t/64` | 声明序第 4；精确 offset 固定仓不可见 | 元素/tile | UB/36 后按 32B 对齐 | `InitBuffer` 容量、循环次数和每 tile 地址步长 | 正数；乘 dtype 字节数传给 Queue/TBuf。 |
| `MishGradV2TilingData` | `haveTanhx` | `int64_t/64` | 声明序第 5；精确 offset 固定仓不可见 | 布尔值 0/1 | input 2 是否存在 | 两 body 的条件 Queue/TBuf 和计算分支 | 公开 ACLNN 固定传 `nullptr`，故可达值仅 0。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 `FP32_DIRECT` | `D_T_X=C_DT_FLOAT` | `mish_grad_v2_tiling.cpp::MishGradV2TilingFunc/FillTilingData` | `MishGradV2TilingData`；`blockDim=usedCoreNum` | `mish_grad_v2.cpp::mish_grad_v2<float>` | `mish_grad_v2.h::KernelMishGradFloat`、`ComputeTanhxCommon`、`ComputeCoreCommon` | `op_host/CMakeLists.txt`：`OPTYPE mish_grad_v2`、`ACLNNTYPE aclnn_exclude`、`DISABLE_IN_OPP TRUE` |
| R-002 `FP16_BF16_CAST` | `D_T_X=C_DT_FLOAT16/C_DT_BF16` | 同 R-001 | 同 R-001 | `mish_grad_v2<half/bfloat16_t>` | `KernelMishGradCast<T>` 与两个 common helper | 同 R-001 |

## Kernel 实现

### 公共执行流程

entry 使用 `GET_TILING_DATA_WITH_STRUCT` 解包 POD，并由编译期 `D_T_X` 选择 `KernelMishGradFloat` 或 `KernelMishGradCast<T>`。每个 block 在 `Init` 中绑定自身连续 GM 片段，分配 Queue/TBuf；`Process` 串行遍历本核 tiles。每 tile 都执行 `CopyIn → Compute → CopyOutTile`。Kernel 为单节点，无跨 Kernel workspace 边、event、atomic、SIMT、NDDMA 或 Cube/MIX_CV 节点。

两条 Route 共享 Mish 导数顺序：重算 `tanhx`，构造 `1-tanhx²`，除以 `1+exp(-x)`，乘 `x`，加 `tanhx`，最后乘 `grad`。区别仅在 GM/Queue dtype 与是否显式升降精度。

### Route R-001：FP32_DIRECT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_mish_backward_v2.cpp` | `CheckParams` / `ExecMishBackwardV2GetWorkspaceSize` | 非空、三 Tensor 同 shape/dtype、rank≤8，dtype=FP32 | 添加 `MishGradV2`；空域提前成功。 |
| Host/Tiling | `op_host/mish_grad_v2_tiling.cpp` | `GetShapeAttrsInfo` / `MishGradV2TilingFunc` | `dataType=DT_FLOAT`，POD/`blockDim` 计算成功 | `ASCENDC_TPL_SEL_PARAM(..., DT_FLOAT)`。 |
| Router | `op_kernel/mish_grad_v2.cpp` | `mish_grad_v2<float>` | `std::is_same_v<D_T_X,float>` | `KernelMishGradFloat`。 |

#### Device 数据流

##### 函数级调用链

```text
l0op::MishGradV2 -> ADD_TO_LAUNCHER_LIST_AICORE
  -> mish_grad_v2<float>(grad, x, nullptr, x_grad, workspace, tiling)
    -> KernelMishGradFloat::Init
    -> KernelMishGradFloat::Process
      -> CopyIn
      -> Compute
        -> ComputeTanhxCommon
        -> ComputeCoreCommon
      -> CopyOutTile<float>
```

##### 源码等价伪码

```text
# 伪码
if blockIdx < formerNum:
    blockLength = formerLength
    base = blockIdx * formerLength
else:
    blockLength = tailLength
    base = formerNum * formerLength

for i in [0, ceil(blockLength/tileLength)):
    valid = min(tileLength, blockLength - i*tileLength)
    aligned = align_up(valid, 8)                  # FP32: 32B/4B
    gradQ <- DataCopyPad(GM_grad[base+i*tileLength], valid, zero to aligned)
    xQ    <- DataCopyPad(GM_x[base+i*tileLength], valid, zero to aligned)

    xLocal = DeQue(xQ); gradLocal = DeQue(gradQ)
    t = relu(xLocal) + ln(1 + exp(-abs(xLocal)))
    t = tanh(t)
    y = t*t
    y = 1-y
    tmp = 1 + exp(-xLocal)
    y = y/tmp
    y = y*xLocal + t
    y = y*gradLocal
    EnQue(outQ, y)
    FreeTensor(gradLocal, xLocal)

    DataCopyPad(GM_y[base+i*tileLength], DeQue(outQ), valid*4 bytes)
    FreeTensor(y)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `grad` | `base+i*tileLength` | `validLength` | FP32/ND contiguous | N0/用户 | `CopyIn` | executor/用户；整次 launch 有效 | tail 允许非对齐。 |
| GM `x` | 同上 | 同上 | FP32/ND contiguous | N1/用户 | `CopyIn` | 同上 | 同上。 |
| `inQueueGrad_` | Queue 管理 | `depth=2 * tileLength * 4B` | FP32 | `CopyIn` | `Compute` | Kernel block 生命周期 | 32B 对齐；Alloc/EnQue/DeQue/Free。 |
| `inQueueX_` | Queue 管理 | `depth=2 * tileLength * 4B` | FP32 | `CopyIn` | `Compute` | 同上 | 同上。 |
| `outQueueY_` | Queue 管理 | `depth=2 * tileLength * 4B` | FP32 | `Compute` | `CopyOutTile` | 同上 | `yLocal` 在重算 `t` 时兼作中间 buffer，随后原位成为输出。 |
| `tmpBuf0_` | TBuf 管理 | `tileLength * 4B` | FP32 | `ComputeTanhxCommon`/`ComputeCoreCommon` | 后续 Vector op | 同上 | 先保存 `ln`，后覆盖为 `1+exp(-x)`。 |
| `tmpBuf1_` | TBuf 管理 | `tileLength * 4B` | FP32 | `ComputeTanhxCommon` | `ComputeCoreCommon` | 同上 | 保存重算的 `tanhx`。 |
| GM `y` | `base+i*tileLength` | 仅 `validLength` | FP32/ND contiguous | `CopyOutTile` | N3 `ViewCopy` | executor 临时 Tensor，N3 后可释放 | 只写有效字节。 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  gradQ["inQueueGrad · depth=2 · 8*tileLength B"]:1
  xQ["inQueueX · depth=2 · 8*tileLength B"]:1
  outQ["outQueueY · depth=2 · 8*tileLength B"]:1
  tmp0["tmpBuf0 · 4*tileLength B"]:1
  tmp1["tmpBuf1 · 4*tileLength B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*t*4 + 2*t*4 + 2*t*4 + t*4 + t*4 = 32*t bytes
t = tileLength <= floor(UB_bytes/36) after 32B alignment
UB_total <= (32/36)*UB_bytes, modulo downward alignment
dynamic_ub = N/A（AIV Queue/TBuf 路径）
workspace_total = executor framework total；本 Kernel 仅申报平台 GetLibApiWorkSpaceSize() 且 entry 不读 workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `outQueueY_` 当前 tensor | `ComputeTanhxCommon` 先写 `exp(-abs(x))` 中间，`ComputeCoreCommon` 后写最终 `y` | `CopyOutTile` | `FreeTensor` 后下一 tile Alloc | `EnQue`→`DeQue`；输出后 `FreeTensor` | 中间值在最终 `EnQue` 前已完成消费。 |
| `tmpBuf0_` | 先写 `ln(1+exp(-abs(x)))`，后写 `1+exp(-x)` | 各紧随的 `Add/Div` | 下一 Vector 阶段/下一 tile | 同一 `Compute` 指令顺序 | 后写发生在前值已被 `Add` 消费后。 |
| Queue depth slot | `CopyIn`/`Compute` | `Compute`/`CopyOutTile` | `FreeTensor` 后 | Queue EnQue/DeQue/Free 协议 | 最多 depth=2 个在途 tensor。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VECIN | `AllocTensor/DataCopyPad/EnQue` | DMA | `DeQue` | grad/x Queue slot | 尾部零填充到 8 个 FP32 元素对齐。 |
| VECIN→计算 | Queue `DeQue/FreeTensor` | `CopyIn` | `Compute` | 当前 tile 输入 | 全流程 FP32；补零 lane 被计算但不写回。 |
| 计算→VECOUT | `AllocTensor/EnQue` | `Compute` | `CopyOutTile::DeQue` | 当前输出 slot | softplus 用 `exp(-abs(x))`；核心含 `exp(-x)` 和 `Div`，源码无额外异常值 guard。 |
| VECOUT→GM | `DataCopyPad` + `FreeTensor` | `CopyOutTile` | N3/下一复用 | GM 有效区 | `blockLen=validLength*4`，不会提交 padding lane。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/mish_grad_v2/op_host/mish_grad_v2_tiling.cpp` | `FillTilingData` | 512B 分核、UB/36 tile、former/tail 与 blockDim。 |
| Kernel entry/router | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.cpp` | `mish_grad_v2<float>` | FP32 映射到 `KernelMishGradFloat`。 |
| Device body/helper | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.h` | `KernelMishGradFloat`、`ComputeTanhxCommon`、`ComputeCoreCommon` | Queue、循环、稳定 softplus、导数 Vector 顺序和有效字节 CopyOut。 |

### Route R-002：FP16_BF16_CAST

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_mish_backward_v2.cpp` | `GetDtypeSupportList` / `CheckParams` | A2/A3 公开支持，三 Tensor 同 FP16 或同 BF16、同 shape、rank≤8、非空 | 添加 `MishGradV2`；空域提前成功。 |
| Host/Tiling | `op_host/mish_grad_v2_tiling.cpp` | `MishGradV2TilingFunc` | `dataType=DT_FLOAT16` 或 `DT_BF16` | 选择对应 `D_T_X` 模板。 |
| Router | `op_kernel/mish_grad_v2.cpp` | `mish_grad_v2<half/bfloat16_t>` | 非 `float` 编译分支 | `KernelMishGradCast<D_T_X>`。 |

#### Device 数据流

##### 函数级调用链

```text
l0op::MishGradV2 -> ADD_TO_LAUNCHER_LIST_AICORE
  -> mish_grad_v2<half|bfloat16_t>(grad, x, nullptr, x_grad, workspace, tiling)
    -> KernelMishGradCast<T>::Init
    -> KernelMishGradCast<T>::Process
      -> CopyIn
      -> Compute
        -> Cast(grad/x, T -> FP32, CAST_NONE)
        -> ComputeTanhxCommon
        -> ComputeCoreCommon
        -> Cast(y, FP32 -> T, CAST_ROUND)
      -> CopyOutTile<T>
```

##### 源码等价伪码

```text
# 伪码；分核/base 与 R-001 相同
for each tile:
    valid = min(tileLength, remaining)
    aligned = align_up(valid, 16)                 # 32B/2B
    gradQ<T> <- DataCopyPad(GM_grad, valid, zero to aligned)
    xQ<T>    <- DataCopyPad(GM_x, valid, zero to aligned)

    gradFp32 = Cast(DeQue(gradQ), CAST_NONE)
    xFp32    = Cast(DeQue(xQ), CAST_NONE)
    tanhxFp32 = tanh(relu(xFp32) + ln(1 + exp(-abs(xFp32))))
    yFp32 = gradFp32 * (tanhxFp32 + xFp32*(1-tanhxFp32²)/(1+exp(-xFp32)))
    yLocal<T> = Cast(yFp32, CAST_ROUND)
    EnQue(outQ, yLocal)
    FreeTensor(input Queue tensors)
    DataCopyPad(GM_y, DeQue(outQ), valid*2 bytes)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `grad/x/y` | `base+i*tileLength` | 输入/output 分别 `validLength` | FP16 或 BF16/ND contiguous | executor/Kernel | Kernel/N3 | launch/N3 生命周期 | tail 以有效字节搬运。 |
| `inQueueGrad_` | Queue 管理 | `depth=2 * tileLength * 2B` | T | `CopyIn` | `Compute` Cast | block 生命周期 | 32B 对齐。 |
| `inQueueX_` | Queue 管理 | `depth=2 * tileLength * 2B` | T | `CopyIn` | `Compute` Cast | 同上 | 同上。 |
| `outQueueY_` | Queue 管理 | `depth=2 * tileLength * 2B` | T | FP32→T Cast | `CopyOutTile` | 同上 | `CAST_ROUND` 后入队。 |
| `gradBufFp32_` | TBuf | `tileLength*4B` | FP32 | Cast grad | core multiply | block 生命周期 | 下一 tile 覆盖。 |
| `xBufFp32_` | TBuf | `tileLength*4B` | FP32 | Cast x | tanhx/core | 同上 | 下一 tile 覆盖。 |
| `tanhxBufFp32_` | TBuf | `tileLength*4B` | FP32 | `ComputeTanhxCommon` | `ComputeCoreCommon` | 同上 | 公开 ACLNN 固定重算。 |
| `tmpBuf0_` | TBuf | `tileLength*4B` | FP32 | tanhx helper/core helper | 后续 Vector op | 同上 | 先 `ln`，后 `1+exp(-x)`。 |
| `tmpBuf1_` | TBuf，条件分配 | `tileLength*4B` | FP32 | tanhx helper | tanhx helper | 同上 | `haveTanhx=0` 时存在，保存 `exp(-abs(x))` 路径中间。 |
| `yBufFp32_` | TBuf | `tileLength*4B` | FP32 | core helper | FP32→T Cast | 同上 | 最终 FP32 结果。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  gradQ["inQueueGrad · depth=2 · 4*t B"]:1
  xQ["inQueueX · depth=2 · 4*t B"]:1
  outQ["outQueueY · depth=2 · 4*t B"]:1
  grad32["gradFp32 · 4*t B"]:1
  x32["xFp32 · 4*t B"]:1
  tanh32["tanhxFp32 · 4*t B"]:1
  tmp0["tmpBuf0 · 4*t B"]:1
  tmp1["tmpBuf1 · 4*t B"]:1
  y32["yFp32 · 4*t B"]:1
```

##### 字节账与复用顺序

```text
t = tileLength
Queue = 3 * depth(2) * t * sizeof(T=2) = 12*t bytes
FP32 TBuf = 6 * t * 4 = 24*t bytes   # grad,x,tanhx,tmp0,tmp1,y
UB_total = 36*t bytes
tileLength = floor((UB_bytes/36)/16)*16
UB_total <= UB_bytes, modulo downward alignment
dynamic_ub = N/A（AIV Queue/TBuf 路径）
workspace_total = executor framework total；Kernel workspace 指针未读
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `tmpBuf1_` | `Abs/Muls/Exp/Adds` 产生 `1+exp(-abs(x))` | `Ln(tmpBuf0_, tmpBuf1_)` | 下一 tile | 同一 `Compute` 指令顺序 | `Ln` 完成前不得覆盖。 |
| `tmpBuf0_` | 先保存 `ln(...)`，后保存 `1+exp(-x)` | `Add`，随后 `Div` | 下一 tile | 同一 `Compute` 顺序 | 前一值先被消费。 |
| FP32 staging TBuf | Cast/helper | 下一 helper/Cast | 下一 tile | 单 tile 串行 `Compute` | `outQueueY_` 入队前 FP32→T Cast 完成。 |
| Queue slot | `CopyIn`/Cast | Cast/CopyOut | `FreeTensor` 后 | Queue 协议 | depth=2 容量计入 UB。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VECIN | `DataCopyPad` + Queue | DMA | `DeQue` | T 输入 slot | 尾部补零到 16 个 T 元素。 |
| 升精度 | `Cast(..., CAST_NONE)` | T Queue | FP32 helper | grad/x FP32 TBuf | FP16/BF16 先升到 FP32，padding 零也被转换。 |
| FP32 计算 | Vector `Relu/Abs/Exp/Ln/Tanh/Mul/Adds/Muls/Div/Add` | FP32 TBuf | 下一个 Vector op | 对齐 tile | 全导数在 FP32 完成；源码没有 NaN/Inf 特判。 |
| 降精度 | `Cast(..., CAST_ROUND)` | `yBufFp32_` | `outQueueY_` | 有效+padding lane | 最终一次 round cast。 |
| VECOUT→GM | `DataCopyPad` | output Queue | GM | 有效输出 | 仅写 `validLength*2` 字节，padding lane 不外泄。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `experimental/activation/mish_grad_v2/op_host/mish_grad_v2_tiling.cpp` | `FillTilingData` | `UB/36` 正好覆盖公开无 `tanhx` cast Route 的 36*t 字节。 |
| Kernel entry/router | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.cpp` | `mish_grad_v2<half/bfloat16_t>` | 非 FP32 模板映射到 `KernelMishGradCast<T>`。 |
| Device body/helper | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.h` | `KernelMishGradCast<T>::Init/Process/CopyIn/Compute` | 条件 TBuf、双缓冲 Queue、FP32 升降精度、tail 和生命周期。 |

## 算子特定优化

### Trick T-001：用绝对值形式重算 softplus 中间量

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：公开 `aclnnMishBackward` 固定把 `tanhx=nullptr` 传给 `MishGradV2`，POD `haveTanhx=0`，两 body 调用 `ComputeTanhxCommon`。
- Golden 来源：[`experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.h:64-71`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.h#L64-L71)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    Relu(tanhxLocal, xLocal, alignedLength);
    Abs(tmp0, xLocal, alignedLength);
    Muls(tmp0, tmp0, -1.0f, alignedLength);
    Exp(tmp0, tmp0, alignedLength);
    Adds(tmp0, tmp0, 1.0f, alignedLength);
    Ln(tmp1, tmp0, alignedLength);
    Add(tanhxLocal, tanhxLocal, tmp1, alignedLength);
    Tanh(tanhxLocal, tanhxLocal, alignedLength);
```

#### 机制与设计原因

这段顺序实现 `tanh(relu(x)+ln(1+exp(-abs(x))))`。`Exp` 的输入恒为 `-abs(x)≤0`，避免在重算 softplus 时直接形成正大数 `exp(x)`；随后 `Relu(x)` 恢复 softplus 的正半轴主项。R-001 复用 output Queue tensor 和 `tmpBuf0_` 承载两个中间量，R-002 复用 `tmpBuf1_`/`tmpBuf0_`，不额外物化 GM 中间结果。`alignedLength` 来自 32B tail 对齐，padding lane 虽参与计算，但 CopyOut 只提交有效 lane。

#### 迁移边界

- 适用：需要从原输入重算 `tanh(softplus(x))` 的逐元素 Mish/MishGrad 实现。
- 前提：FP32 或先升到 FP32；具有 `Relu/Abs/Exp/Ln/Tanh`；保持源码顺序和临时 buffer 生命周期。
- 失效：若调用方已提供可信 `tanhx`，内部 Host 路径会跳过重算并分配输入 Queue；该路径不是公开 ACLNN 可达 Route，不能直接套用本卡字节账。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得据此声称任意数学等价重排都有相同溢出、舍入或 UB 行为。

## 模板与实现对比

| 对比维度 | R-001 FP32_DIRECT | R-002 FP16_BF16_CAST | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | dtype=FP32 | dtype=FP16/BF16 | input dtype 直接选择模板，无运行时 fallback。 |
| 模板与 dtype | `mish_grad_v2<float>` | `mish_grad_v2<half/bfloat16_t>` | 两个低精度 key 合并，因为 body/资源/同步/tail 相同。 |
| 循环与分核 | 512B former core + tail core；tile 32B 对齐 | 相同 | `s` 改变对齐元素数，不改变算法。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 个 depth=2 FP32 Queue + 2 FP32 TBuf；寄存器占用固定仓不可见 | 3 个 depth=2 T Queue + 6 FP32 TBuf；寄存器占用固定仓不可见 | cast Route 需要显式 FP32 staging；均无 dynamic UB/通信。 |
| 内存总占用与复用 | `32*t` B；output Queue 和 tmp0 复用 | `36*t` B；tmp0 跨阶段复用 | Host 用最坏 `UB/36` 同时约束两 Route。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | FP32 `DataCopyPad` + Vector 公式 | T `DataCopyPad` + Cast + FP32 Vector 公式 | 均仅 AIV Vector。 |
| 同步、精度与 tail | Queue 同步，全 FP32；8 元素对齐 | Queue 同步，T→FP32→T；16 元素对齐 | 都只写有效字节。 |
| fallback/失败边界 | 非 FP32 去 R-002 | 非 FP16/BF16 失败 | 空域 Host 成功短路；无 Device fallback。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 `MishGradV2` 文档支持；本卡闭合 | 公开 `MishGradV2` 文档支持；本卡闭合 | 公开支持，但本固定仓内稳定实现的 ATVOSS 调度依赖不闭合，未纳入本卡 |
| Host/Launcher | 共用 `MishGradV2` API/Tiling；BF16 公开支持 | 共用；BF16 公开支持 | 稳定目录使用 `MishGrad` + `arch35` `ElewiseBaseTiling`，不同版本族 |
| Kernel/资源 | `mish_grad_v2`，资源由运行平台 AIV core/UB 生产 | 同 A2 | 稳定 `mish_grad_apt` + `ElementwiseSch16B`；固定仓缺少 ATVOSS 实体，无法给出完整字节账 |

## 限制与失败边界

- 只覆盖公开 `aclnnMishBackward` 可达的 `haveTanhx=0` 路径；内部三输入 `MishGradV2` 路径不是本卡 Route。
- 不支持 broadcast 和隐式类型提升；三个 Tensor shape/dtype 必须完全一致，rank≤8。
- `GetShapeSize()`、分核乘加、`validLength*sizeof(T)` 到 `uint32_t blockLen` 的窄化没有算子内显式溢出检查；安全范围由 ACLNN shape/平台资源的上游约束承担，固定仓不能补出额外上限。
- `tileLength<=0` 时 Host 强制一个 32B 对齐块，但没有再次校验异常小 UB 是否容纳；正常支持平台要求 `ubSize>0`，更强下界固定仓不可见。
- API executor workspace 的精确切片、框架 `Contiguous/ViewCopy` 实体和生成模板 key 的数值编码属于固定仓外框架边界；算子自有 selector、POD、entry 与 body 已闭合。
- 本实现仅通过 `ENABLE_EXPERIMENTAL` 路径构建，CMake 标记 `DISABLE_IN_OPP TRUE`；不能把它描述为默认内置 OPP 实现。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 版本/构建说明 | `experimental/activation/mish_grad_v2/README.md` | `MishGradV2` 说明、Example/Tests 命令 | 新实现身份、对外 ACLNN 符号与活动 `--experimental --ops=mish_grad_v2` 构建入口。 |
| 入口声明/约束 | `experimental/activation/mish_grad_v2/docs/aclnnMishBackward.md` | `aclnnMishBackward*` 文档 | A2/A3 公开支持、shape/dtype/空域/非连续约束和公式。 |
| 入口实现 | `experimental/activation/mish_grad_v2/op_host/op_api/aclnn_mish_backward_v2.cpp` | `CheckParams`、`ExecMishBackwardV2GetWorkspaceSize`、`aclnnMishBackward` | 两段式 API、连续化、空域、内部调用、ViewCopy、executor。 |
| 入口实现 | `experimental/activation/mish_grad_v2/op_host/op_api/mish_grad_v2.cpp` | `l0op::MishGradV2` | 临时输出分配、`tanhx=nullptr`、AICore launcher。 |
| Host 注册 | `experimental/activation/mish_grad_v2/op_host/mish_grad_v2_def.cpp` | `MishGradV2` | 输入输出 schema、AutoContiguous、AICore config。 |
| Host/Tiling/Launcher | `experimental/activation/mish_grad_v2/op_host/mish_grad_v2_tiling.cpp` | `GetPlatformInfo`、`FillTilingData`、`MishGradV2TilingFunc` | 平台资源、分核、tile、workspace、模板 selector。 |
| POD | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2_tiling_data.h` | `MishGradV2TilingData` | 五个 Host/Device ABI 字段声明。 |
| 模板选择 | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个 dtype 模板实例。 |
| Kernel entry | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.cpp` | `mish_grad_v2<D_T_X>` | POD 解包与 FP32/cast body 路由。 |
| Kernel body | `experimental/activation/mish_grad_v2/op_kernel/mish_grad_v2.h` | `KernelMishGradFloat`、`KernelMishGradCast`、common helpers | GM/UB、Queue/TBuf、Vector 顺序、精度、同步和 tail。 |
| 构建接线 | `experimental/activation/mish_grad_v2/op_host/CMakeLists.txt` | `add_modules_sources` | 活动实验构建 target 与 `DISABLE_IN_OPP` 边界。 |
| 950 公开约束 | `activation/mish_grad/docs/aclnnMishBackward.md` | 产品支持表 | 950 公开支持。 |
| 950 构建接线 | `activation/mish_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR` | 稳定实现只接 `ascend950/arch35`。 |
| 950 Host/Tiling | `activation/mish_grad/op_host/arch35/mish_grad_tiling_arch35.cpp` | `MishGradTiling::RunTiling` | 委托固定仓外 `ElewiseBaseTiling::DoTiling`。 |
| 950 Kernel entry | `activation/mish_grad/op_kernel/mish_grad_apt.cpp` | `mish_grad` | 委托固定仓外 `ElementwiseSch16B` 调度器。 |
| 950 公共依赖接线 | `cmake/gen_ops_info.cmake` | `atvoss_src_copy` | ATVOSS 从 `${OPBASE_SOURCE_PATH}` 复制，证明实体不属于当前固定仓。 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | former/tail 分核与 tile/tanhx 标志 | 字段声明顺序、64 位宽、元素单位；公开入口 `haveTanhx=0` | `MishGradV2TilingData` |
| 调用顺序 | Contiguous→MishGradV2→ViewCopy；Kernel 内 CopyIn→Compute→CopyOut | 空域必须在添加任务前短路；N2 结果先于 N3 | `ExecMishBackwardV2GetWorkspaceSize`、`Process` |
| workspace/内存 | R-001 `32*t`、R-002 `36*t`，Host `t≤UB/36` | Queue depth=2、条件 `tmpBuf1`、有效字节 CopyOut、owner/lifetime | `FillTilingData`、两个 `Init` |
| 同步 | Queue EnQue/DeQue/Free 和 executor stream 顺序 | 不得提前覆盖 Queue/TBuf；固定源码无跨核通信 | `CopyIn`、`Compute`、`CopyOutTile` |
| 精度与尾块 | 稳定 softplus；低精度 Route 在 FP32 完成核心计算 | `CAST_NONE`/`CAST_ROUND` 顺序、32B padding、只写 validLength | `ComputeTanhxCommon`、`KernelMishGradCast::Compute` |
| 失败边界 | 无效参数/Tiling/launcher 失败；空域成功无 launch | 不得新增 broadcast、内部 tanhx Route 或 Device fallback | `CheckParams`、`MishGradV2TilingFunc` |
