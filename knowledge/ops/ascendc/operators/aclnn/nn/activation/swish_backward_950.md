---
type: Operator
title: "SwishBackward 950 ACLNN 算子知识"
description: "说明 aclnnSwishBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/docs/aclnnSwishBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_api/aclnn_swish_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_api/aclnn_swish_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_api/swish_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_host/swish_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_host/config/ascend950/swish_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_kernel/swish_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_kernel/arch35/swish_grad_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_kernel/arch35/swish_grad_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_kernel/arch35/swish_grad_tilingdata.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/third_party/opbase.cmake
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/include/nnopbase/opdev/platform.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/elewise/elewise_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/util/node.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
created_at: "2026-08-25T19:01:50Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# SwishBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnSwishBackwardGetWorkspaceSize` + `aclnnSwishBackward`，公开 ACLNN 两段式接口；固定仓未见 V2 入口 |
| 计算语义 | `gradInput = gradOutput * sigmoid(beta*self) * (1 + beta*self*(1-sigmoid(beta*self)))` |
| 输入输出 | `gradOutput/self/gradInput` 同 shape、同 dtype、ND；dtype 为 FP16/BF16/FP32；`betaOptional` 可转 FP32，空指针取 `1.0f` |
| 执行类型 | 单个 `AIV_VECTOR` 节点；`swish_grad` 显式声明 `KERNEL_TYPE_AIV_ONLY`，无 Cube/AIC、SIMT、NDDMA 或通信节点 |
| Kernel 路线 | R-001 FP32 原生寄存器计算；R-002 FP16/BF16 升 FP32 计算后 RINT 降精度 |
| 核心资源 | 两路 GM 输入、一路 GM 输出；单个 `TBuf<VECCALC>` 划分为 6 个 ping/pong 子块；MicroAPI RegTensor；算子 tiling 声明 16 MiB workspace，但 Kernel 不读 workspace |
| 来源闭包 | `source_closed`：ACLNN 校验/连续化 → `SwishGrad` launcher → 950 Tiling/POD/key → AIV entry → ElementwiseSch → SwishGrad MicroAPI body；共享调度依赖由 `ops-nn` 锁定的 `opbase@b8bd9e6600eb4fc11d59586bc1712cbc9d87c3ed` 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `g=gradOutput`、`x=self`、`z=beta*x`、`s=1/(1+exp(-z))`，逐元素输出：

```text
gradInput = g * s * (1 + z * (1 - s))
```

Kernel 的等价顺序是 `z=beta*x`、`s=sigmoid(z)`、`t=z*(1-s)`、`d=t*s+s`、`out=d*g`。`SwishGrad` 内部 IR 仍保留第三输入 `y`，ACLNN launcher 将 `self` 同时传给 `x` 与 `y`，但 950 Kernel 的 `ElementwiseSch::Init(grad, x, grad_x)` 不绑定也不读取 `y`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | 与 `self/gradInput` 相同，ND，公开维数 0–8 | FP16/BF16/FP32 | 可非连续；空 Tensor 直接成功返回且不启动算子 |
| `self` | 与 `gradOutput/gradInput` 相同，ND | 与 `gradInput` 相同 | Swish 正向输入 `x`；可非连续 |
| `betaOptional` | Host scalar | 可转换为 FP32 | 空指针取 `1.0f`，否则 `ToFloat()`；950 的 `SocVersion::ASCEND950` 是 `ASCEND910_95` 别名，落在 ACLNN BF16 允许区间 `[ASCEND910B, ASCEND910E]` |
| `gradInput` | 与两输入相同，ND | 与两输入相同 | 可非连续；由 `ViewCopy` 写回用户 view |

### 入口与版本

公开文档与头文件只给出 `aclnnSwishBackward` 两段式入口，没有同语义 V2。第一段构造 executor 并返回总 workspace，第二段把 workspace 与 stream 交给 `CommonOpExecutorRun`。950 公开支持由接口文档的 Ascend 950 行、`SwishGrad::AICore().AddConfig("ascend950", ...)`、`ascend950/swish_grad_binary.json`、`SUPPORT_COMPUTE_UNIT="ascend950"` 四层共同确认。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize/executor` 输出指针，`CREATE_EXECUTOR` 创建 executor。
2. `CheckNotNull` 要求 `gradOutput/self/gradInput` 非空；`CheckDtypeValid` 限定 FP16/BF16/FP32、三 Tensor dtype 相同、scalar 可转 FP32，并执行 BF16 SoC gate；`CheckShapeValid` 要求三 Tensor shape 相同且维数不超过 `MAX_SUPPORT_DIMS_NUMS`。
3. 任一输入为空时设置 `workspaceSize=0`、移交 executor 并返回，不创建 `SwishGrad` 节点。
4. 非空时分别经框架 `l0op::Contiguous` 获得连续 `gradOutput/self`。源码保留“大于 8 维 reshape 成 1D”的分支，但它位于前述最大维数校验之后，对本入口不可达。
5. `betaOptional==nullptr` 时 `scale=1.0f`，否则 `scale=betaOptional->ToFloat()`。
6. `l0op::SwishGrad` 分配临时输出，并以 `OP_INPUT(gradOutput, self, self)` 添加一个 AICore 节点；最后由框架 `ViewCopy` 将临时结果写入可能非连续的 `gradInput`。

### 参数、Workspace 与生命周期

`SwishGradTiling::SetTilingData` 为该 AICore 节点写 `currentWorkspace[0]=16777216` 字节。`swish_grad` entry 接收 `workspace` 但不解引用，故这 16 MiB 是 Host 声明的保留量而非本 Kernel 的中间数据。ACLNN 返回的总 workspace 还可能包含 `Contiguous/ViewCopy` 的框架切片；其精确 offset/extent 由 executor 生成层管理，固定源码未展开。用户在第二段调用期间拥有 workspace；executor 拥有连续化临时 Tensor 与 `SwishGrad` 临时输出，直至同一执行流中的 `ViewCopy` 消费完成。

### 启动图与失败传播

```text
aclnnSwishBackwardGetWorkspaceSize
  -> [framework] Contiguous(gradOutput)
  -> [framework] Contiguous(self)
  -> SwishGradAiCore: ADD_TO_LAUNCHER_LIST_AICORE
       entry swish_grad<schMode=1,dType>
       blockDim = baseTiling.blockNum
       inputs = contiguous_grad, contiguous_self, contiguous_self(dummy y)
       output = executor-owned temporary grad_x
       workspace declaration = 16 MiB; body unused
  -> [framework] ViewCopy(temporary grad_x, user gradInput)
aclnnSwishBackward
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

框架节点按 executor 记录顺序提交到传入 stream；`SwishGrad` 成功产出临时结果后 `ViewCopy` 才可消费。任一 `Contiguous/SwishGrad/ViewCopy` 构造失败，第一段立即返回对应错误而不移交不完整流程；第二段的执行错误由 `CommonOpExecutorRun` 返回。框架节点内部 kernel、workspace slice 与事件细节属于固定仓可接受的 executor/Contiguous/ViewCopy 边界。

## Route 与参数设计

### 分核与资源计算

令 `N=product(output storage shape)`、`Cplat=GetCoreNumAiv()`、`U=平台 UB 字节数`、`b=sizeof(T)`。`SwishGradDAG<T>::OpDag` 在 `MemLevel::LEVEL_2` 下有两路非 scalar 输入与一路直连 CopyOut 的结果，得到 `BufferNum=2*2+1*2=6`，`MaxDtypeBytes=MinDtypeBytes=b`。共享 Tiling 依次计算：

```text
C = min(Cplat, ceil(N*b / 4096))
blockFormer = align_up(ceil(N/C), 512)                 # 元素
blockNum = ceil(N / blockFormer)
blockTail = N - (blockNum-1)*blockFormer
alignFactor = 256 / b                                 # 元素
ubFormer = floor((U/(6*b)) / alignFactor)*alignFactor # 元素
loopFormer = ceil(blockFormer/ubFormer)
loopTail = ceil(blockTail/ubFormer)
tailFormer = blockFormer-(loopFormer-1)*ubFormer
tailTail = blockTail-(loopTail-1)*ubFormer
```

FP16/BF16 的 `alignFactor=128` 元素，FP32 为 64 元素。`blockNum` 既写入 POD 又作为 launch `blockDim`；最后一核使用 tail block 参数，其余核使用 former 参数。Tiling 拒绝零维长度、零 AIV core、UB 不足或 `ubFormer==0`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原生 | `GET_TPL_TILING_KEY(1, TPL_FP32)`；`ElementwiseSch<1, SwishGradDAG<float>::OpDag>` | ACLNN 三 Tensor 均 FP32；Tiling `outputDtype==DT_FLOAT` | 其它支持 dtype 继续匹配 R-002；其它 dtype 失败 |
| 2 | R-002 FP16/BF16 升精度 | `GET_TPL_TILING_KEY(1, TPL_FP16)` 或 `GET_TPL_TILING_KEY(1, TPL_BF16)` | 三 Tensor 均 FP16，或均 BF16 且 ACLNN BF16 SoC gate 成立；Tiling 按 output dtype 选对应 `T` | 其它 dtype 失败 |

```text
if outputDtype == FP16: dType=TPL_FP16, R-002
else if outputDtype == BF16: dType=TPL_BF16, R-002
else if outputDtype == FP32: dType=TPL_FP32, R-001
else: GRAPH_FAILED
scheMode = ElewiseBaseTiling::DEFAULT_SCHEMODE = 1
key = GET_TPL_TILING_KEY(scheMode, dType)
```

`swish_grad_struct.h` 声明 `schMode` 可取 0/1，但活动 Host producer 恒写 1，因此 `schMode=0` 是固定入口不可达实例，不构成 Route。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 | `(1,TPL_FP32)` | 950；FP32；同 shape、同 dtype；非空后进入 Tiling | 单 AIV `swish_grad` | `SwishGradDAG<float>` → `SwishGradCustom<float>` | 6 个 FP32 tile 子块；MTE2/V/MTE3 mutex；全程 FP32 RegTensor | MicroAPI mask 保护尾 lane；最后核/最后 tile 使用独立 tail 长度；无 fallback |
| R-002 FP16/BF16 升精度 | `(1,TPL_FP16)` / `(1,TPL_BF16)` | 950；相应 dtype；BF16 还需 ACLNN SoC gate；同 shape、同 dtype；非空 | 单 AIV `swish_grad` | `SwishGradDAG<half/bfloat16_t>` → `SwishGradCustom<T>` | 6 个 16-bit tile 子块；unpack/load 后 FP32 寄存器计算，RINT 后 pack/store | 以 FP32 lane mask 保护尾部；无 dtype fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域短路 | `gradOutput->IsEmpty() \|\| self->IsEmpty()` | `ElewiseTiling` 明确拒绝 shape dim 为 0 | 全部 | workspace=0，成功且不启动 Kernel | `aclnnSwishBackwardGetWorkspaceSize`; `ElewiseTiling` |
| 每核最小粒度 | `C=min(Cplat, ceil(N*b/4096))` | 避免把少于 4 KiB 数据继续拆核 | 全部 | 自动减小 `blockNum` | `ElewiseTiling` |
| UB 容量 | `ubFormer=floor(U/(6*b)/(256/b))*(256/b)` 且非 0 | 单 `TBuf` 必须容纳 6 个 tile 子区并按 256 B repeat 对齐 | 全部 | Tiling `GRAPH_FAILED` | `GetMaxElemNum`; `ElewiseBaseTiling::DoTiling`; `ElementwiseSch::Init` |
| BF16 API gate | `ASCEND910B <= SocVersion <= ASCEND910E`；950 枚举为 `ASCEND910_95` | ACLNN 在 launch 前限制 BF16 平台 | R-002 BF16 | `ACLNN_ERR_PARAM_INVALID` | `CheckSocVersionIsSupportBf16`; `SocVersion` |
| POD 窄化 | `ElewiseTilingData.ubFormer(int64)` 赋给 `EleBaseTilingData.ubFormer(int32)` | Device `blockLen` 也是 `int`；源码无显式溢出检查 | 全部 | 固定源码没有 fallback | `AdaptEleBaseTilingData`; `ElementwiseSch::Init` |
| key 可达性 | `scheMode=1` | 只编译/选择活动调度模式；模式 0 无 Host producer | 全部 | 未知/不可达 key 的生成 router 行为固定仓不可见 | `GetScheMode`; `SetTilingData`; `ASCENDC_TPL_SEL` |

### POD/ABI

`EleBaseTilingData` 是标准布局字段序列；按固定依赖使用的 LP64 ABI，其 `sizeof=80`、`alignof=8`。外层 `SwishGradTilingData` 的 `scale` 位于 offset 80，尾部补齐后 `sizeof=88`、`alignof=8`；源码没有额外 packing/static_assert。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `EleBaseTilingData` | `dim0` | int64/64 | off 0 | 元素 | `ElewiseTiling: product(shape)` | 本 Kernel scheduler 不读取 | shape 乘法无显式溢出检查 |
| `EleBaseTilingData` | `coreNum` | int32/32 | off 8 | AIV core | `AdaptEleBaseTilingData(static_cast<int32_t>(coreNum))` | 本 Kernel scheduler 不读取 | Host 已限制到平台 AIV 数 |
| `EleBaseTilingData` | `ubFormer` | int32/32 | off 12 | 元素/tile | `GetMaxElemNum` 后隐式窄化 | `Init` 的 blockLen/UB；`Process` full tile | 必须非 0 且 256 B 对齐 |
| `EleBaseTilingData` | `blockFormer` | int64/64 | off 16 | 元素/core | `align_up(ceil(N/C),512)` | `CalcBlockOffset` | 最后一核也以 former stride 定址 |
| `EleBaseTilingData` | `blockNum` | int64/64 | off 24 | core/block | `ceil(N/blockFormer)` | `Process` 判定最后一核；Host `SetBlockDim` | `blockNum>=1` |
| `EleBaseTilingData` | `ubLoopOfFormerBlock` | int64/64 | off 32 | tile loop | `ceil(blockFormer/ubFormer)` | 非最后核 `Process.loopNum` | 至少 1 |
| `EleBaseTilingData` | `ubLoopOfTailBlock` | int64/64 | off 40 | tile loop | `ceil(blockTail/ubFormer)` | 最后一核 `Process.loopNum` | 至少 1 |
| `EleBaseTilingData` | `ubTailOfFormerBlock` | int64/64 | off 48 | 元素 | `blockFormer-(loops-1)*ubFormer` | 非最后核最后 tile 长度 | `1..ubFormer` |
| `EleBaseTilingData` | `ubTailOfTailBlock` | int64/64 | off 56 | 元素 | `blockTail-(loops-1)*ubFormer` | 最后一核最后 tile 长度 | `1..ubFormer` |
| `EleBaseTilingData` | `elemNum` | int64/64 | off 64 | 元素 | `maxElemNum` | 本 Kernel scheduler 不读取 | 等于未窄化的 UB tile 上限 |
| `EleBaseTilingData` | `scheMode` | uint64/64 | off 72 | 枚举 | `GetScheMode()` 恒 1 | Host 生成模板 key；Device body不读取 | 模式 0 不可达 |
| `SwishGradTilingData` | `scale` | float/32 | off 80 | 标量 beta | `attrs[0]`，空属性取 1.0f | entry `SetVar<float,0>` → `SwishGradCustom(scale1)` | NaN/Inf/数值范围无额外校验 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 | `(1,TPL_FP32)` | `swish_grad_tiling_arch35.cpp::RunTiling/SetTilingData`；opbase `ElewiseBaseTiling/ElewiseTiling` | `SwishGradTilingData{EleBaseTilingData,scale}` | `swish_grad_apt.cpp::swish_grad<1,TPL_FP32>` | `SwishGradCustom<float>`；opbase `ElementwiseSch`, `Vec::CopyIn/CopyOut`, sync helpers | `activation/swish_grad/CMakeLists.txt`; `swish_grad_def.cpp`; `ascend950/swish_grad_binary.json` |
| R-002 FP16/BF16 升精度 | `(1,TPL_FP16/BF16)` | 同上，dtype 分支实例化 `SwishGradDAG<half/bfloat16_t>` | 同一 POD，tile 对齐为 128 元素 | `swish_grad<1,TPL_FP16/BF16>` | `SwishGradCustom<half/bfloat16_t>` 与相同 scheduler/helper | 同上 |

## Kernel 实现

### 公共执行流程

entry 用 `GET_TILING_DATA` 解析 POD，构造一个栈上 `TPipe` 和 `ElementwiseSch<schMode,OpDag>`，将 `scale` 写入 `Var<float,0>`，绑定 `grad/x/grad_x` 后执行。每核 GM 起点为 `blockFormer*blockIdx*sizeof(T)`；`Process` 以 `ubFormer` 为 full tile，最后一次使用对应 former/tail core 的尾长。每个 tile 的 DAG 顺序固定为 `CopyIn grad → CopyIn x → SwishGradCustom → CopyOut`。调度器以同一 `TBuf` 内的 ping/pong buffer ID 和 MTE2/V/MTE3 mutex 保证搬运、计算、搬出的所有权转移。

### Route R-001：FP32 原生寄存器计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/swish_grad/op_api/aclnn_swish_backward.cpp` | `CheckDtypeValid` | 三 Tensor dtype 相同且在 FP16/FP32/BF16 列表 | FP32 通过 |
| Tiling | `activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp` | `SwishGradTiling::RunTiling` | `outputDtype == ge::DT_FLOAT` | `dType=TPL_FP32`; `DoTiling<SwishGradDAG<float>::OpDag>` |
| Router | `activation/swish_grad/op_kernel/swish_grad_apt.cpp` | `swish_grad<schMode,dType>` | `dType == TPL_FP32` | `ElementwiseSch<schMode,SwishGradDAG<float>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
SwishGradAiCore -> swish_grad<1,TPL_FP32>(grad,x,y,grad_x,workspace,tiling)
  -> SetVar<float,0>(scale)
  -> ElementwiseSch::Init(grad,x,grad_x)
  -> ElementwiseSch::Process
    -> Run: CopyIn<float>(grad)
    -> Run: CopyIn<float>(x)
    -> RunNormalOp: SwishGradCustom<float>
    -> Run: CopyOut<float>(grad_x)
```

##### 源码等价伪码

```text
# 伪码；每个 AIV core
base = blockFormer * blockIdx
loops, tail = (blockIdx == blockNum-1) ? tailBlockParams : formerBlockParams
for tile in [0, loops):
    count = (tile == loops-1) ? tail : ubFormer
    off = tile * ubFormer
    DMA grad[base+off : +count] -> gradBuf[pingpong]
    DMA x[base+off : +count]    -> xBuf[pingpong]
    for regChunk in ceil(count/(VECTOR_REG_WIDTH/4)):
        mask = remaining FP32 lanes
        z = beta * x
        s = 1 / (1 + exp(-z))
        d = (z * (1-s)) * s + s
        out = d * grad
        masked_store out -> outBuf[pingpong]
    DMA outBuf[pingpong, 0:count] -> grad_x[base+off : +count]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `grad` | `(blockFormer*blockIdx + tile*ubFormer)*4` B | `count*4` B | FP32/ND flattened | caller/Contiguous | `Vec::CopyIn<float>` | caller/framework temp；整次 launch | 单 `DataCopyPad`, blockCount=1, blockLen=count*4 B, stride=0 |
| GM `x` | 同上 | `count*4` B | FP32/ND flattened | caller/Contiguous | `Vec::CopyIn<float>` | 同上 | 同上 |
| UB grad ping/pong | logical buffer id selected by `GetBufId` | 每子块 `ubFormer*4` B | FP32 | MTE2 | Vector | `TBuf`/tile phase | 双缓冲；PIPE_MTE2→PIPE_V mutex |
| UB x ping/pong | 同上 | 每子块 `ubFormer*4` B | FP32 | MTE2 | Vector | 同上 | 双缓冲 |
| UB out ping/pong | 同上 | 每子块 `ubFormer*4` B | FP32 | Vector | MTE3 | 同上 | 双缓冲；PIPE_V→PIPE_MTE3 mutex |
| RegTensor | `loopIdx*(VECTOR_REG_WIDTH/4)` 元素 | 5 个主要 FP32 RegTensor + 常量寄存器，物理字节固定仓未公开 | FP32 registers | MicroAPI load/ops | MicroAPI store | 单寄存器循环 | `vregInput` 原地变为 z，`vregOutput` 原地串联导数 |
| GM `grad_x` | 与输入同元素 offset | `count*4` B | FP32/ND flattened | `Vec::CopyOut` | ViewCopy/user | executor temp至 ViewCopy | 单 `DataCopyPad`, blockCount=1, blockLen=count*4 B, stride=0 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  gp["grad ping · ubFormer×4 B"] xp["x ping · ubFormer×4 B"] op["out ping · ubFormer×4 B"]
  gg["grad pong · ubFormer×4 B"] xg["x pong · ubFormer×4 B"] og["out pong · ubFormer×4 B"]
```

##### 字节账与复用顺序

```text
blockLen = ubFormer * 4
UB_total = 6 * blockLen = 24 * ubFormer bytes
UB_total <= U
ubFormer is aligned down to 64 FP32 elements = 256 bytes
dynamic_ub = N/A
workspace_total (SwishGrad node declaration) = 16 MiB; Kernel body usage = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| grad/x ping 或 pong | `CopyIn` MTE2 | `SwishGradCustom` PIPE_V | 同 parity 的下一 tile | `GetTensor/ReleaseTensor` 对 PIPE_MTE2、PIPE_V 加解 mutex | 计算消费后才允许同 id 再搬入 |
| out ping 或 pong | `SwishGradCustom` PIPE_V | `CopyOut` MTE3 | 同 parity 的下一 tile | PIPE_V、PIPE_MTE3 mutex | MTE3 完成后才允许 Vector 覆盖 |
| `vregInput` | load x 后 `Muls` | sigmoid 与导数阶段 | 下一寄存器循环 | `__VEC_SCOPE__` 顺序 | mask 仅覆盖有效 lane |
| `vregOutput` | `1-s` | `Mul`、`FusedMulDstAdd`、最终乘 grad | 下一寄存器循环 | 同一寄存器依赖链 | 不在导数消费前覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `Mutex::Lock/Unlock<PIPE_MTE2>` | MTE2 CopyIn | PIPE_V `GetTensor` | grad/x 子块 | `count` 精确字节搬运，无额外 padding |
| Vector | `Mutex::Lock/Unlock<PIPE_V>` | CopyIn | custom/CopyOut | 输入与输出子块 | FP32；`UpdateMask<float>(count)` 屏蔽最后寄存器无效 lane |
| UB→GM | `Mutex::Lock/Unlock<PIPE_MTE3>` | custom | CopyOut完成/下次覆盖 | out 子块 | 仅搬 `count*4` B |
| tile/core tail | `Process` 的 former/tail 分支 | Tiling POD | 当前 core/tile | GM 有效域 | 最后一核及每核最后 tile 均使用显式尾长 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp` | `RunTiling`, `SetTilingData` | FP32 模板、scale、key、blockDim、16 MiB workspace |
| Kernel entry/router | `activation/swish_grad/op_kernel/swish_grad_apt.cpp` | `swish_grad<1,TPL_FP32>` | AIV-only；绑定 grad/x/output，忽略 y/workspace |
| Device body/helper | `activation/swish_grad/op_kernel/arch35/swish_grad_dag.h` | `SwishGradCustom<float>` | FP32 MicroAPI sigmoid/导数/grad 管线与 mask |
| Scheduler | `pkg_inc/op_common/atvoss/elewise/elewise_sch.h` | `ElementwiseSch::Init/Process/Run` | 6 子块池、分核地址、tile 循环与 DAG 调度 |

### Route R-002：FP16/BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `activation/swish_grad/op_api/aclnn_swish_backward.cpp` | `CheckDtypeValid` | FP16 直接通过；BF16 还需 `CheckSocVersionIsSupportBf16()` | 相应 16-bit route 或参数错误 |
| Tiling | `activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp` | `RunTiling` | `outputDtype==DT_FLOAT16` / `DT_BF16` | `TPL_FP16` / `TPL_BF16` 与对应 DAG |
| Router | `activation/swish_grad/op_kernel/swish_grad_apt.cpp` | `swish_grad<schMode,dType>` | `dType==TPL_FP16` / `TPL_BF16` | `ElementwiseSch<1,SwishGradDAG<T>::OpDag>` |

#### Device 数据流

##### 函数级调用链

```text
SwishGradAiCore -> swish_grad<1,TPL_FP16 or TPL_BF16>
  -> ElementwiseSch::Init/Process
    -> CopyIn<T>(grad), CopyIn<T>(x)
    -> SwishGradCustom<T>
       -> DIST_UNPACK_B16 loads -> Cast<T->float,ZEROING>
       -> FP32 sigmoid and derivative
       -> Cast<float->T,NO_SAT,ZEROING,CAST_RINT>
       -> DIST_PACK_B32 store
    -> CopyOut<T>(grad_x)
```

##### 源码等价伪码

```text
# 分核/tile 与 R-001 相同，T=half 或 bfloat16
load packed grad/x from UB with DIST_UNPACK_B16
cast both to FP32 registers under remaining-lane mask
z = FP32(beta) * FP32(x)
s = 1 / (1 + exp(-z))
d = (z*(1-s))*s + s
out32 = d * FP32(grad)
out16 = cast_RINT_no_sat(out32)
masked DIST_PACK_B32 store out16 to UB
copy exactly count*sizeof(T) bytes to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM grad/x | `(blockFormer*blockIdx + tile*ubFormer)*2` B | 每路 `count*2` B | FP16 或 BF16/flattened ND | caller/Contiguous | CopyIn | caller/framework | 单块 `DataCopyPad`, stride 0 |
| UB grad/x ping/pong | logical buffer id | 每子块 `ubFormer*2` B | T | MTE2 | unpack load | `TBuf`/tile phase | 各两份双缓冲 |
| FP32 RegTensor | register chunk | 固定仓不公开物理字节 | FP32 | unpack + Cast | 全部算术 | 单寄存器循环 | 输入和所有中间值升精度 |
| UB out ping/pong | logical buffer id | 每子块 `ubFormer*2` B | T | RINT cast + pack | CopyOut | `TBuf`/tile phase | 两份双缓冲 |
| GM grad_x | 与输入相同元素 offset | `count*2` B | T/flattened ND | CopyOut | ViewCopy/user | executor temp | 不写尾部无效 lane |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  gp["grad ping · ubFormer×2 B"] xp["x ping · ubFormer×2 B"] op["out ping · ubFormer×2 B"]
  gg["grad pong · ubFormer×2 B"] xg["x pong · ubFormer×2 B"] og["out pong · ubFormer×2 B"]
```

##### 字节账与复用顺序

```text
blockLen = ubFormer * 2
UB_total = 6 * blockLen = 12 * ubFormer bytes
UB_total <= U
ubFormer is aligned down to 128 16-bit elements = 256 bytes
dynamic_ub = N/A
workspace_total (SwishGrad node declaration) = 16 MiB; Kernel body usage = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| grad/x ping-pong | CopyIn | unpack load/custom | 同 parity 下一 tile | MTE2→V mutex | 当前 Vector 消费完成 |
| out ping-pong | pack store/custom | CopyOut | 同 parity 下一 tile | V→MTE3 mutex | 当前 CopyOut 完成 |
| FP32 input/output registers | Cast/算术 | 后续算术/RINT Cast | 下一寄存器循环 | 程序顺序 | `UpdateMask<float>` 保护尾 lane |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | MTE2 mutex + `DataCopyPad` | GM | Vector | 16-bit input 子块 | 仅 `count*2` B |
| unpack/cast | `DIST_UNPACK_B16`, `Cast<float,T>` | UB | FP32 math | valid lanes | Cast trait ZEROING，mask 类型为 FP32 |
| math | FP32 MicroAPI | Cast | downcast | RegTensor | β、exp、div、FMA、grad 乘法均 FP32 |
| downcast/pack | `Cast<T,float>` + `CAST_RINT` + `DIST_PACK_B32` | FP32 result | UB/MTE3 | output lanes | NO_SAT；尾 mask 禁止无效 lane 写出 |
| CopyOut | MTE3 mutex | UB | GM | output 子块 | 精确有效字节数 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp` | `RunTiling` | FP16/BF16 key、POD 与 16-bit tile 计算 |
| Kernel entry/router | `activation/swish_grad/op_kernel/swish_grad_apt.cpp` | `swish_grad<1,TPL_FP16/BF16>` | dtype 到 DAG 实例映射 |
| Device body/helper | `activation/swish_grad/op_kernel/arch35/swish_grad_dag.h` | `SwishGradCustom<half/bfloat16_t>` | unpack、FP32 计算、RINT、pack 与 mask |
| Platform guard | `include/nnopbase/opdev/platform.h` | `SocVersion` | 950=`ASCEND910_95`，证明 BF16 ACLNN 区间 gate 可达 |

## 算子特定优化

### Trick T-001：寄存器内融合 sigmoid 导数链

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：任意非空、通过 ACLNN/Tiling 校验的 tile。
- Golden 来源：[`activation/swish_grad/op_kernel/arch35/swish_grad_dag.h:68-77`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_kernel/arch35/swish_grad_dag.h#L68-L77)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
MicroAPI::Muls(vregInput, vregInput, scale1, mask);
MicroAPI::Neg(vregInputMid, vregInput, mask);
MicroAPI::Exp(vregInputMid, vregInputMid, mask);
MicroAPI::Adds(vregInputMid, vregInputMid, (float)1.0, mask);
MicroAPI::Div(vregInputMid, vregValue1, vregInputMid, mask);

MicroAPI::Sub(vregOutput, vregValue1, vregInputMid, mask);
MicroAPI::Mul(vregOutput, vregOutput, vregInput, mask);
MicroAPI::FusedMulDstAdd(vregOutput, vregInputMid, vregInputMid, mask);
MicroAPI::Mul(vregOutput, vregOutput, vregInput2, mask);
```

#### 机制与设计原因

`vregInput` 原地从 `x` 变为 `z=beta*x`，`vregInputMid` 原地从 `-z` 变为 `s`，`vregOutput` 从 `1-s` 变为 `z(1-s)`，再由 `FusedMulDstAdd(dst,s,s)` 得到 `dst*s+s`。导数中间量不物化到 UB/GM，且 FMA 合并乘加；最后才乘 `grad`。mask 来自剩余有效元素，所有指令共享同一有效 lane 集。

#### 迁移边界

- 适用：逐元素激活导数可在寄存器中形成短依赖链的场景。
- 前提：950 AIV MicroAPI 支持所用 Exp/Div/FusedMulDstAdd，且输入已由 scheduler 搬入 UB。
- 失效：需要跨元素归约、跨核通信或更大中间状态时，不能照搬此单寄存器链。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：16-bit 输入统一升 FP32 后一次性回写

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：FP16 或 BF16 输入。
- Golden 来源：[`activation/swish_grad/op_kernel/arch35/swish_grad_dag.h:94-115`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/swish_grad/op_kernel/arch35/swish_grad_dag.h#L94-L115)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
    vregInput16, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
    vregInput162, (__ubuf__ T*)(src2Addr + loopIdx * vlSize));
MicroAPI::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
MicroAPI::Cast<float, T, castTrait0>(vregInput2, vregInput162, mask);

MicroAPI::Muls(vregInput, vregInput, scale1, mask);
MicroAPI::Neg(vregInputMid, vregInput, mask);
MicroAPI::Exp(vregInputMid, vregInputMid, mask);
MicroAPI::Adds(vregInputMid, vregInputMid, (float)1.0, mask);
MicroAPI::Div(vregInputMid, vregValue1, vregInputMid, mask);

MicroAPI::Sub(vregOutput, vregValue1, vregInputMid, mask);
MicroAPI::Mul(vregOutput, vregOutput, vregInput, mask);
MicroAPI::FusedMulDstAdd(vregOutput, vregInputMid, vregInputMid, mask);
MicroAPI::Mul(vregOutput, vregOutput, vregInput2, mask);

MicroAPI::Cast<T, float, castTrait1>(vregOutput16, vregOutput, mask);
MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(
    (__ubuf__ T*)(dstAddr + loopIdx * vlSize), vregOutput16, mask);
```

#### 机制与设计原因

两个 16-bit 输入只在 load 后各转换一次，完整 sigmoid/导数/上游梯度乘法都保持 FP32，最终以 `NO_SAT + CAST_RINT + ZEROING` 一次降回目标 dtype，并用 pack store 写 UB。这避免在非线性链的中间步骤反复 16-bit 舍入；`vl=VECTOR_REG_WIDTH/sizeof(float)` 与 `UpdateMask<float>` 让 FP16/BF16 共用相同 FP32 寄存器迭代结构。

#### 迁移边界

- 适用：输入/输出为 FP16/BF16、内部包含 exp/div/FMA 的逐元素算子。
- 前提：UB 仍按 16-bit tile 存储，寄存器链可容纳 FP32 中间值，并接受源码指定的 RINT、NO_SAT 语义。
- 失效：要求其它舍入/饱和模式或逐步与 16-bit 参考严格一致时必须重审精度顺序。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 原生 | R-002 FP16/BF16 升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | dtype=FP32 | dtype=FP16/BF16；BF16 有 SoC gate | Host dtype 决策树互斥 |
| 模板与 dtype | `SwishGradDAG<float>` | `SwishGradDAG<half/bfloat16_t>` | binary config 三个 dtype 实例 |
| 循环与分核 | 4 KiB/core，512 元素 block 对齐，64 元素 tile 对齐 | 同分核，128 元素 tile 对齐 | alignFactor=256/sizeof(T) |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 6×FP32 子块；FP32 RegTensor；无 dynamic UB/通信 | 6×16-bit 子块；16-bit load/store + FP32 RegTensor | 相同 Level-2 DAG buffer 图 |
| 内存总占用与复用 | `24*ubFormer` B | `12*ubFormer` B | 两者均不超过平台 U |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + FP32 MicroAPI | DataCopyPad + unpack/cast/FP32 MicroAPI/pack | 均为纯 AIV_VECTOR |
| 同步、精度与 tail | MTE2/V/MTE3 mutex；FP32；mask | 相同同步；升 FP32、RINT 降精度；mask | dtype 精度路线不同 |
| fallback/失败边界 | 无 fallback | 无 fallback；BF16 gate 失败即 API 错误 | 不支持 dtype/Tiling 失败不降级 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档列为支持，但正式目录无对应活动 binary/Host/Kernel 接线，未形成 `source_closed` | 同 A2 | 公开文档、OpDef、binary config、CMake 与 Kernel 全链路支持 |
| Host/Launcher | 未发布本卡；experimental 候选由 `ENABLE_EXPERIMENTAL` 互斥选择，非正式目录活动依赖 | 同 A2 | `arch35` + `ElewiseBaseTiling`，AIV core 数/UB 来自平台 |
| Kernel/资源 | 固定正式目录不可见 | 固定正式目录不可见 | AIV-only，6 子块 TBuf + MicroAPI RegTensor |

## 限制与失败边界

- `gradOutput/self/gradInput` 必须非空指针、shape/dtype 相同，dtype 仅 FP16/BF16/FP32，维数不超过 8；空 Tensor 是无 Kernel 的成功路径。
- Tiling 将 shape 连乘到 `int64_t dim0`，未显式检查乘法溢出；`ubFormer` 从 int64 隐式窄化到 int32，固定源码无 fallback。
- `betaOptional` 只校验可转换 FP32，不额外限制 NaN/Inf/范围；exp/div 的特殊值语义由 950 MicroAPI 决定。
- 算子声明 16 MiB workspace 但 Kernel 不读取；不得据此把这块空间改作未同步的跨节点缓存。
- 未知模板 key 的生成 router 终点由构建框架生成，固定源码不可见；活动 Host 只产生 `scheMode=1` 的三个 dtype key。
- A2/A3 未生成正式卡：公开支持声明与正式源码实现接线之间缺少 Host/Tiling/Kernel/binary 闭包；不得用互斥的 experimental 候选补成正式活动路线。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/swish_grad/docs/aclnnSwishBackward.md` | `aclnnSwishBackward*` | 公开性、平台、接口、dtype/shape/空 Tensor/确定性 |
| 入口声明/约束 | `activation/swish_grad/op_api/aclnn_swish_backward.h` | 两段式函数声明 | C ABI 与参数顺序 |
| 入口实现 | `activation/swish_grad/op_api/aclnn_swish_backward.cpp` | `CheckParams`, `aclnnSwishBackwardGetWorkspaceSize`, `aclnnSwishBackward` | 校验、连续化、空域、scale、executor、ViewCopy |
| Launcher | `activation/swish_grad/op_api/swish_grad.cpp` | `SwishGradAiCore`, `SwishGrad` | 三输入映射、临时输出、AICore 节点添加 |
| 平台注册 | `activation/swish_grad/op_host/swish_grad_def.cpp` | `ops::SwishGrad` | ascend950 OpDef 与 kernel 文件名 |
| Host/Tiling | `activation/swish_grad/op_host/arch35/swish_grad_tiling_arch35.cpp` | `SwishGradTiling::*`, `Tiling4SwishGrad` | dtype/shape/attr、POD、key、blockDim、workspace |
| POD | `activation/swish_grad/op_kernel/arch35/swish_grad_tilingdata.h` | `SwishGradTilingData` | 外层 POD 与 scale |
| key/router | `activation/swish_grad/op_kernel/arch35/swish_grad_struct.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode/dtype 模板域 |
| Kernel entry | `activation/swish_grad/op_kernel/swish_grad_apt.cpp` | `swish_grad` | AIV-only entry、dtype 实例与 scheduler 绑定 |
| Kernel body | `activation/swish_grad/op_kernel/arch35/swish_grad_dag.h` | `SwishGradCustom`, `SwishGradDAG` | MicroAPI 数学、精度、mask 与 DAG |
| 构建接线 | `activation/swish_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `add_modules_sources` | 仅 ascend950/arch35 活动接线 |
| binary 接线 | `activation/swish_grad/op_host/config/ascend950/swish_grad_binary.json` | `op_list` | 三 dtype 950 binary 配置 |
| 固定依赖 | `cmake/third_party/opbase.cmake` | `OPBASE_TAG_ID` | 锁定 opbase Commit |
| 平台枚举 | `include/nnopbase/opdev/platform.h` | `SocVersion` | 950 BF16 SoC gate 的枚举顺序 |
| Tiling ABI | `pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h` | `EleBaseTilingData` | POD 字段顺序与位宽 |
| Tiling 模板 | `pkg_inc/op_common/atvoss/elewise/elewise_tiling.h` | `ElewiseBaseTiling::DoTiling` | DAG buffer 参数与平台 shape 输入 |
| Tiling 算法 | `src/op_common/atvoss/elewise/elewise_tiling.cpp` | `ElewiseTiling`, `GetMaxElemNum` | core/block/tile/tail 公式 |
| Scheduler | `pkg_inc/op_common/atvoss/elewise/elewise_sch.h` | `ElementwiseSch` | TBuf、GM offset、tile 循环与 DAG 执行 |
| Buffer 元信息 | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | BufferNum、buffer ID 与 Level-2 选择 |
| Buffer 计数 | `pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo::GetBufferNumLevel2` | 6 子块推导 |
| Copy helper | `pkg_inc/op_common/atvoss/util/vec.h` | `Vec::CopyIn`, `Vec::CopyOut` | DataCopyPad 参数与有效字节数 |
| 同步 helper | `pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor`, `ReleaseTensor` | MTE2/V/MTE3 mutex 协议 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 复用 `EleBaseTilingData + float scale` | 字段顺序/位宽/单位、scale offset 与 `scheMode=1` producer 不变 | `SwishGradTilingData`; `EleBaseTilingData` |
| 调用顺序 | Contiguous 两输入 → SwishGrad → ViewCopy | 第三 IR 输入是 self 的 dummy 复用；Kernel 只绑定 grad/x/out | `aclnnSwishBackwardGetWorkspaceSize`; `SwishGradAiCore`; `swish_grad` |
| workspace/内存 | 6 子块 ping/pong池与 16 MiB Host 声明 | `ubFormer` 公式、buffer parity、owner/lifetime 不变 | `ElementwiseSch::Init`; `SetTilingData` |
| 同步 | 按 buffer id 复用 MTE2/V/MTE3 mutex | 不得在 release 前覆盖同 parity 子块 | `GetTensor`; `ReleaseTensor` |
| 精度与尾块 | FP32 原生；16-bit 升 FP32 后 RINT；双层 tail | cast trait、mask dtype、block/tile 地址公式不可改变 | `SwishGradCustom`; `ElementwiseSch::Process` |
| 失败边界 | 空域短路、dtype/shape/SoC/UB/key 失败 | 不得添加未有证据的 fallback | `CheckParams`; `ElewiseTiling`; `RunTiling` |
