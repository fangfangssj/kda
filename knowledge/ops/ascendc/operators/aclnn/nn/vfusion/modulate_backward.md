---
type: Operator
title: "ModulateBackward A2/A3 ACLNN 算子知识"
description: "说明 aclnnModulateBackward 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/docs/aclnnModulateBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_host/op_api/aclnn_modulate_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_host/op_api/modulate_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_host/modulate_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_host/modulate_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_kernel/modulate_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_host/modulate_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_host/config/ascend910b/modulate_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_host/config/ascend910_93/modulate_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T18:38:18Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ModulateBackward A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnModulateBackwardGetWorkspaceSize` / `aclnnModulateBackward`，固定仓唯一无版本后缀公开 ACLNN 版本 |
| 计算语义 | 对 `[B,L,D]` 的 `grad_output`：`grad_input=grad_output*scale[:,None,:]`（无 scale 时源码复制 `grad_output`）；可选地沿 `L` 归约得到 `grad_scale=sum(input*grad_output)`、`grad_shift=sum(grad_output)` |
| 输入输出 | 必选 `grad_output,input,grad_input`；`scale,shift,grad_scale,grad_shift` 声明为可选；公开 dtype 为 FP16/BF16/FP32、ND，算子节点统一以 FP32 执行 |
| 执行类型 | ACLNN executor 含框架 `Contiguous/Cast/ViewCopy` 边界；算子自有节点 `ModulateGrad` 为单 Kernel `AIV_VECTOR`，无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 无 scale/shift；R-002 scale；R-003 shift；R-004 scale+shift；四者均为 key 0、同一 `ModulateGradKernel<float>`，由 POD presence flag 分流 |
| 核心资源 | GM；每 AIV 核 9 个 depth=1 的 TQue；算子节点 workspace 仅声明 LibApi workspace、Kernel body 不读取；ACLNN executor 另拥有框架临时张量 |
| 来源闭包 | `source_closed`：公开 ACLNN→executor→`ModulateGrad` launcher→Tiling/POD→key 0 entry→`ModulateGradKernel<float>`→A2/A3 binary/config 已闭合；框架 Cast/Contiguous/ViewCopy 按框架边界标注 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `g=grad_output`、`x=input`，shape 均按公开语义为 `[B,L,D]`，`s=scale`、`t=shift` 为 `[B,D]`。Device body 对每个 `(b,l,d)` 执行：存在 `s` 时 `grad_input=g*s`，不存在时复制 `g`；存在 `s` 时 `grad_scale[b,d]=sum_l(x*g)`；存在 `t` 时 `grad_shift[b,d]=sum_l(g)`。`shift` 的数值不参与反向计算，仅其 presence 决定是否产生 `grad_shift`。归约在每个 `(b,D 分片)` 所属 AIV 核内按 L tile 顺序用 FP32 累加，无跨核原子。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `grad_output` | `[B,L,D]` / ND | FP16/BF16/FP32 | 必选；ACLNN 先连续化并转 FP32 |
| `input` | `[B,L,D]` / ND | FP16/BF16/FP32 | 必选；文档要求与 `grad_output` 同 shape，但 `CheckParams` 未显式比较两者 shape |
| `scale` | `[B,D]` / ND | FP16/BF16/FP32 | 声明可选；presence 产生 `grad_input` 乘法及 `grad_scale` |
| `shift` | `[B,D]` / ND | FP16/BF16/FP32 | 声明可选；数值不读，presence 产生 `grad_shift` |
| `grad_input` | `[B,L,D]` / ND | FP16/BF16/FP32 | 必选；框架将 FP32 Kernel 结果 cast/view-copy 到目标 |
| `grad_scale` | `[B,D]` / ND | FP16/BF16/FP32 | 可选输出；只有 `scale` 存在时算子内部才分配对应结果 |
| `grad_shift` | `[B,D]` / ND | FP16/BF16/FP32 | 可选输出；只有 `shift` 存在时算子内部才分配对应结果 |

### 入口与版本

固定 Commit 只有无后缀两段式 API。公开文档明确支持 Atlas A2、A3；`ModulateGrad` OpDef、binary config 和全局 AscendC config 对应 `ascend910b` 与 `ascend910_93`。未发现 950/`ascend950` 公开文档、OpDef 或 binary 接线。本卡覆盖 A2/A3 的同一最新公开实现，不补造 950 版本。

## Host/Launcher 执行流程

### 参数校验与前处理

`GetWorkspaceSize` 依次检查公共输出指针和 executor、创建 executor、调用 `CheckParams`。必选非空检查只覆盖 `grad_output/input/grad_input`；各 present tensor dtype 必须在 FP16/BF16/FP32；`grad_output/input` 必须恰为 3 维；present `scale/shift` 必须为 2 维且分别满足 `[B,D]`。实现对非 ND 只 `OP_LOGW`，不返回失败。空 `grad_output/input/grad_input` 直接返回 workspace 0 和空 executor。

非空路径依次将 `grad_output`、`input` 连续化并 cast 到 FP32；present `scale/shift` 同样处理；`l0op::ModulateGrad` 分配 FP32 中间输出并登记单个 AICore launcher；之后将结果 cast 并 ViewCopy 到用户输出。`grad_scale`、`grad_shift` 的输出 cast 目标 dtype 在源码中都取 `grad_input->GetDataType()`，不是各自输出 tensor dtype。

### 参数、Workspace 与生命周期

第一段由 executor 持有连续化、cast、算子中间输出及 ViewCopy 节点，`GetWorkspaceSize()` 返回这些框架节点与算子节点的合计需求；固定仓没有展开 executor 的 slice offset。`TilingFunc` 为算子节点令 `workspace[0]=PlatformAscendC::GetLibApiWorkSpaceSize()`，但 `modulate_grad` entry 的 `workspace` 参数未被 body 读取。第二段 `CommonOpExecutorRun` 在调用者给定 stream 上按 executor 顺序执行；任一第一段中间节点返回空即返回错误且不登记后续节点。

### 启动图与失败传播

```text
grad_output -> Contiguous -> Cast(FP32) --+
input       -> Contiguous -> Cast(FP32) --+--> ModulateGrad(AIV, blockDim=usedCores, key=0)
scale?      -> Contiguous -> Cast(FP32) --+      -> FP32 grad_input/grad_scale?/grad_shift?
shift?      -> Contiguous -> Cast(FP32) --+         -> Cast(dst dtype) -> ViewCopy(user output)
```

框架边界节点与 `ModulateGrad` 由同一 executor、同一 stream 排序，临时 tensor 由 executor 持有至执行完成。算子节点没有跨 Kernel edge。Tiling 失败或 launcher 执行失败由框架阻断后续 cast/view-copy；框架内部 event 和各临时 slice 的精确 offset 在固定仓不可见。

## Route 与参数设计

### 分核与资源计算

令 `C=GetCoreNumAiv()`，先取 `N=min(C,B*D)`。若 `B < floor(N/2)`，按 D 分核：`coresPerB=floor(N/B)`、`usedCores=B*coresPerB`，每个 batch 的前 `D%coresPerB` 核处理 `floor(D/coresPerB)+1` 列，其余核处理 `floor(D/coresPerB)` 列。否则按 B 分核：`usedCores=min(N,B)`，前 `B%usedCores` 核处理 `floor(B/usedCores)+1` 个 batch，其余核处理 `floor(B/usedCores)` 个 batch。

Device 固定 `MAX_TILE_LENGTH=3072` 个 FP32 元素。每核 D 分片 `Dc` 不超过 3072 时，`row=align_up(4*Dc,32)` 字节、`currentD=row/4`、`maxRows=floor(3072/currentD)`，按 L tile；`Dc>3072` 时按 D 切成至多 3072 元素的块，并令每个 L tile 只有一行。空 tensor 在 ACLNN 层提前返回，避免 `GenerateTilingData` 的除数为零路径。Host 读取平台 UB 大小到 compile info，但 selector 未消费该值。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 直通 | key 0 / `ModulateGradKernel<float>` | `has_scale==0 && has_shift==0` | 其余 presence 组合 |
| 2 | R-002 Scale | key 0 / 同上 | `has_scale==1 && has_shift==0` | 其余 presence 组合 |
| 3 | R-003 Shift | key 0 / 同上 | `has_scale==0 && has_shift==1` | 其余 presence 组合 |
| 4 | R-004 Scale+Shift | key 0 / 同上 | `has_scale==1 && has_shift==1` | 无；未知 key 没有显式失败终点 |

```text
TilingFunc: SetTilingKey(0)
  optional scale shape present -> has_scale=1 else 0
  optional shift shape present -> has_shift=1 else 0
entry: GET_TILING_DATA; TILING_KEY_IS(0) 的分支体为空；随后无条件实例化 ModulateGradKernel<float>
  (0,0)->R-001; (1,0)->R-002; (0,1)->R-003; (1,1)->R-004
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 0 / flags 00 | 非空 `[B,L,D]`，无 scale/shift，A2/A3；ACLNN 统一转 FP32 | 单 AIV Kernel | `ModulateGradKernel<float>::ComputeGradients` 早退复制 | 仍分配 9 个队列并无条件搬入 input/grad_output/scale | D/L 双层 tail；无专用 fallback |
| R-002 | 0 / flags 10 | 同上且 scale `[B,D]` present | 单 AIV Kernel | FP32 `Mul` 产生 grad_input 与 `x*g`，逐 L 累加 grad_scale | Queue + `PIPE_ALL/PIPE_V` | 同上 |
| R-003 | 0 / flags 01 | 同上且 shift `[B,D]` present | 单 AIV Kernel | 复制 grad_input，逐 L `Add` grad_shift | Queue + `PIPE_ALL` | 同上；scale GM 仍被无条件读取 |
| R-004 | 0 / flags 11 | scale、shift 均 present | 单 AIV Kernel | R-002 与 R-003 的计算合并 | Queue + `PIPE_ALL/PIPE_V` | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 分核轴 | `B < floor(min(C,B*D)/2)` | B 小时用多个核覆盖 D，否则按 batch 均分 | 全部 | 另一分核分支 | `GenerateTilingData` |
| D overflow | `totalD_ > 3072` | 单行超过 12 KiB tile 容量 | 全部 | D 分块、每 tile 一行 | `InitTileParams` |
| DMA 行对齐 | `alignedRowBytes=align_up(4*d_size,32)` | `DataCopyPad` 行按 32 B 补零 | 全部 | 尾行 padding，不失败 | `InitTileParams/UpdatetileParamsDsize` |
| L tail | `L % maxRowsPerTile_ != 0` | 最后 tile 只搬有效行 | 全部 | `currentL_=lastTileLengthL_` | `CalculateOffsets` |
| 空域 | 任一必选数据/output empty | 避免 Device 分核除零和无效 launch | 全部 | workspace 0、无 Kernel | `aclnnModulateBackwardGetWorkspaceSize` |
| UB 选择缺口 | 无 Host 谓词；compile info 的 `ubSizePlatForm` 未使用 | body 固定分配 `3*len+6*row` | 全部 | 固定仓没有 fallback | `TilingPrepareForModulateGrad/InitPipeBuffers` |

### POD/ABI

`ModulateGradTiling` 由 `BEGIN_TILING_DATA_DEF` 在 `#pragma pack(push,8)` 下生成。固定仓不含宏展开，因此可证明字段顺序、源码位宽和 producer/consumer，但不能可靠给出生成类型的 `sizeof/alignof/offsetof`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ModulateGradTiling` | `B` | uint64/64 | pack(8)，序号0；offset 固定仓不可见 | 元素维 | `set_B(input.shape[0])` | `InitTilingParams` | 非空路径应 `B>0` |
| 同上 | `L` | uint64/64 | 序号1 | 元素维 | `set_L(shape[1])` | tile 数、GM offset | 乘法溢出未检查 |
| 同上 | `D` | uint64/64 | 序号2 | 元素维 | `set_D(shape[2])` | 分核、stride、offset | Device 多处窄化为 uint32 |
| 同上 | `dataType` | uint32/32 | 序号3 | 枚举值 | 默认 2 | 不读取 | inactive |
| 同上 | `block_dim` | uint32/32 | 序号4 | AIV 核 | `GetCoreNumAiv()` | 不读取 | 实际 launch 用 `usedCores` |
| 同上 | `dataTypeSize` | uint32/32 | 序号5 | 字节 | 默认 4 | 不读取 | Kernel 固定 float |
| 同上 | `splitB` | uint32/32 | 序号6 | bool flag | `B < floor(N/2)` | `splitByD_` | 名称与语义相反：1 表示按 D 分 |
| 同上 | `coresPerB` | uint32/32 | 序号7 | 核/batch | 分核公式 | core→batch/D 分片 | 按 B 分时为 1 |
| 同上 | `usedCores` | uint32/32 | 序号8 | 核 | 分核公式并传 `SetBlockDim` | `cores_` | 需非零 |
| 同上 | `formerNum` | uint32/32 | 序号9 | 核/组 | remainder | former 分支 | 可为0 |
| 同上 | `formerLength` | uint32/32 | 序号10 | D 元素或 batch 数 | quotient+1；整除 B 分支时0 | 分片长度/起点 | 单位随 splitB 变化 |
| 同上 | `tailNum` | uint32/32 | 序号11 | 核/组 | 剩余核数 | 不读取 | inactive |
| 同上 | `tailLength` | uint32/32 | 序号12 | D 元素或 batch 数 | quotient | 分片长度/起点 | 单位随 splitB 变化 |
| 同上 | `has_scale` | uint32/32 | 序号13 | bool flag | optional input shape 是否存在 | 计算/输出分支 | 与实际 GM 指针必须一致 |
| 同上 | `has_shift` | uint32/32 | 序号14 | bool flag | optional input shape 是否存在 | 归约/输出分支 | shift 数值不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key0/00 | `TilingFunc/GenerateTilingData` | `ModulateGradTiling` | `modulate_grad` | `ModulateGradKernel<float>` copy branch | A2/A3 JSON + `ascendc_config.json` |
| R-002 | key0/10 | 同上 | 同上 | 同上 | scale/mul/reduce branch | 同上 |
| R-003 | key0/01 | 同上 | 同上 | 同上 | shift/reduce branch | 同上 |
| R-004 | key0/11 | 同上 | 同上 | 同上 | combined branch | 同上 |

## Kernel 实现

### 公共执行流程

entry 忽略 `Shift` 与 `workspace`，按参数顺序把 `Input,grad_Output,grad_input,Scale,grad_scale,grad_shift` 交给 `Init`。每核由 `InitCoreSpecificParams` 确定 batch/D 区间，`InitTileParams` 确定 D 与 L tile，再为 9 个队列分配 UB。`Process` 按 batch tile→D tile→L tile 串行执行 `CopyIn→Compute→CopyOut`，阶段间使用 `PipeBarrier<PIPE_ALL>`；Vector 乘加之间另用 `PIPE_V`。

### Route R-001：无 Scale/Shift 直通

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_host/op_api/aclnn_modulate_backward.cpp` | `GetWorkspaceSize` | scale/shift 未提供 | casted 指针保持 null |
| Tiling | `op_host/modulate_grad_tiling.cpp` | `TilingFunc` | optional shape 均 null | flags 00、key0 |
| Router | `op_kernel/modulate_grad.cpp` | `modulate_grad` | 无实际 key guard | `ModulateGradKernel<float>` |
| Body | 同上 | `ComputeGradients` | `!has_scale_ && !has_shift_` | DataCopy 后早退 |

#### Device 数据流

##### 函数级调用链

```text
l0op::ModulateGrad -> modulate_grad(key0, blockDim=usedCores)
  -> InitGlobalTensors/InitTilingParams/InitCoreSpecificParams/InitTileParams/InitPipeBuffers
  -> Process -> CopyInInput + CopyInGradOutput + CopyInScale
             -> ComputeGradients(copy grad_output -> grad_input)
             -> CopyOutGradInput
```

##### 源码等价伪码

```text
for local batch; for D chunk; for L tile:
  off=(b*L+l0)*D+d_start+d_chunk; copy-pad input and grad_output; unconditionally copy scale row
  grad_input_ub[0:currentL*currentD] = grad_output_ub[...]
  copy valid currentL rows × d_size to grad_input GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/grad_output GM | `(b*L+l0)*D+d0` | `currentL` 行×`d_size` | FP32 ND | ACLNN Cast | CopyIn | executor/调用期 | 行 stride D，UB 补至32B |
| inputQ/gradOutputQ/gradInputQ | UB allocator | 各 `len=currentL*row` B | FP32 | CopyIn/Compute | Compute/CopyOut | 单 tile | depth1，DeQue 后 Free |
| scaleQ | UB allocator | `row` B | FP32 | `CopyInScale` | 此 Route 不读 | 单 tile | 仍无条件 DMA |
| 六个归约队列 | UB allocator | 各 `row` B | FP32 | 本 Route 不分配 tensor | 无 | pipe lifetime | buffer 仍预留 |
| grad_input GM | 同输入 offset | 有效 `currentL*d_size` | FP32 ND | CopyOut | output Cast | executor | 去除 UB padding |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["inputQ · depth=1 · len B"] gradOutQ["gradOutputQ · depth=1 · len B"] scaleQ["scaleQ · depth=1 · row B"]
  gradInQ["gradInputQ · depth=1 · len B"] accumQ["gradShiftLocalQ+gradScaleLocalQ · 2·row B"] tileQ["tileShift+tileScale+currentScale · 3·row B"]
```

##### 字节账与复用顺序

```text
UB_total = 3*len + 6*row <= 3*12288 + 6*12288 = 110592 B
dynamic_ub = N/A
workspace_total(operator node) = GetLibApiWorkSpaceSize(); Kernel body consumption = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/gradOutput/scale queues | CopyIn | Compute | 下一 tile | DeQue+FreeTensor、PIPE_ALL | depth1 必须串行 |
| gradInputQ | Compute | CopyOut | 下一 tile | EnQue/DeQue/FreeTensor | CopyOut 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Compute | Queue + PIPE_ALL | DataCopyPad | DeQue | 三路输入 UB | FP32；padding=0 |
| Compute→CopyOut | Queue + PIPE_ALL | DataCopy | DeQue | gradInputQ | 复制 `currentL*currentD`，GM 仅写 `d_size` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_grad_tiling.cpp` | `TilingFunc` | flags/key/blockDim/POD |
| Kernel entry/router | `op_kernel/modulate_grad.cpp` | `modulate_grad` | 固定 float 实例 |
| Device body/helper | 同上 | `ComputeGradients/CopyOutGradInput` | 直通与有效行写回 |

### Route R-002：Scale 梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_host/op_api/aclnn_modulate_backward.cpp` | `GetWorkspaceSize` | scale present、shift absent | scale cast FP32 |
| Tiling | `op_host/modulate_grad_tiling.cpp` | `TilingFunc` | optional flags 10 | R-002/key0 |
| Body | `op_kernel/modulate_grad.cpp` | `ComputeGradients` | `has_scale_` | grad_input 与 grad_scale |

#### Device 数据流

##### 函数级调用链

```text
modulate_grad -> Process -> CopyIn(input,g,scale)
  -> ComputeGradients -> Mul(g,scale)->grad_input; Mul(input,g)->current; Add tile; Add persistent grad_scale
  -> CopyOutGradInput; last L tile -> CopyOutScaleShift(grad_scale)
```

##### 源码等价伪码

```text
on first L tile: persistent_grad_scale=0; persistent_grad_shift=0
tile_scale=0
for l in current tile:
  grad_input[l]=g[l]*scale
  current=input[l]*g[l]; tile_scale+=current
persistent_grad_scale+=tile_scale
after last L tile: write persistent_grad_scale[b,d_chunk]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三输入 Q / gradInputQ | 同 R-001 | `3*len+row` B | FP32 | DMA/Vector | Vector/DMA | tile | depth1 |
| tileGradScale/currentScale | UB | 各 row B | FP32 | Duplicate/Mul/Add | persistent Add | tile | FreeTensor 后复用 |
| gradScaleLocalQ | UB | row B | FP32 | first tile zero，逐 tile Add | last tile CopyOut | 整个 D chunk | EnQue/DeQue 跨 tile |
| grad_scale GM | `b*D+d0` | `d_size` | FP32 `[B,D]` | last tile DMA | output Cast | executor | 每 `(b,d)` 单核 owner |

##### 分层内存布局

与 R-001 的 9 队列布局相同；R-002 实际激活 scale、tileGradScale、currenttileGradScale、gradScaleLocal 队列，gradShiftLocal 仍初始化以保持队列协议。

##### 字节账与复用顺序

```text
UB_total = 3*len + 6*row <= 110592 B; workspace 与 R-001 相同
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| currenttileGradScaleQ | Mul(input,g) | Add(tileScale,...) | 下一 L 行 | PIPE_V | Add 完成后覆盖 |
| gradScaleLocalQ | tile Add | 下一 tile/CopyOut | 下一 D chunk | EnQue/DeQue、最后 Free | 一个核独占 `(b,d_chunk)` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 两次 Mul→Add | PIPE_V | Vector Mul | Vector Add | current/tile accumulator | 全 FP32，按 L 顺序累加 |
| tile→persistent | PIPE_V | tile Add | persistent Add | gradScaleLocal | D tail 用 `d_size` mask长度 |
| last tile→GM | PIPE_ALL + Queue | EnQue | DeQue/DMA | persistent result | 仅最后 L tile写出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_grad_tiling.cpp` | `TilingFunc` | has_scale=1 |
| Kernel entry/router | `op_kernel/modulate_grad.cpp` | `modulate_grad` | float body |
| Device body/helper | 同上 | `ComputeGradients/CopyOutScaleShift` | FP32 乘法与分 tile 归约 |

### Route R-003：Shift 梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_host/op_api/aclnn_modulate_backward.cpp` | `GetWorkspaceSize` | scale absent、shift present | shift cast FP32 |
| Tiling | `op_host/modulate_grad_tiling.cpp` | `TilingFunc` | flags 01 | R-003/key0 |
| Body | `op_kernel/modulate_grad.cpp` | `ComputeGradients` | `has_shift_ && !has_scale_` | grad_input copy + grad_shift |

#### Device 数据流

##### 函数级调用链

```text
modulate_grad -> CopyIn(input,g,scale) -> ComputeGradients
  -> Add(tileShift,g row); DataCopy(g row,grad_input row); Add(persistentShift,tileShift)
  -> CopyOutGradInput; last L tile -> CopyOutScaleShift(grad_shift)
```

##### 源码等价伪码

```text
persistent_shift=0
for each L tile: tile_shift=0
  for l: tile_shift+=g[l]; grad_input[l]=g[l]
  persistent_shift+=tile_shift
last tile: write grad_shift[b,d_chunk]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gradOutputQ/gradInputQ | 同 R-001 | 各 len B | FP32 | DMA/copy | Add/DMA | tile | depth1 |
| tileGradShiftQ/gradShiftLocalQ | UB | 各 row B | FP32 | Duplicate/Add | persistent Add/GM | tile/chunk | Queue ordering |
| grad_shift GM | `b*D+d0` | d_size | FP32 `[B,D]` | last tile | output Cast | executor | 单核 owner |
| scale GM/Q | `b*D+d0` | row B | FP32 | 无条件 CopyIn | 不读取 | tile | scale null 时存在源码可达风险 |

##### 分层内存布局

与 R-001 相同；R-003 激活 tileGradShift 与 gradShiftLocal。其余归约 buffer 仍由 `InitPipeBuffers` 预留。

##### 字节账与复用顺序

```text
UB_total = 3*len + 6*row <= 110592 B; workspace 与 R-001 相同
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tileGradShiftQ | per-row Add | persistent Add | 下一 tile | PIPE_ALL/FreeTensor | tile Add 已消费 |
| gradShiftLocalQ | persistent Add | 下一 tile/GM | 下一 D chunk | EnQue/DeQue/Free | 单核独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| row reduction | 顺序 Vector Add + PIPE_ALL 外围 | gradOutputQ | tile accumulator | d_size lanes | FP32；padding lane不参与 |
| output | Queue + DMA | persistentShift | GM | `[B,D]` chunk | last L tile写出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_grad_tiling.cpp` | `TilingFunc` | has_shift=1 |
| Kernel entry/router | `op_kernel/modulate_grad.cpp` | `modulate_grad` | shift GM 参数不传入 Init |
| Device body/helper | 同上 | `ComputeGradients` | grad_shift 只依赖 g |

### Route R-004：Scale 与 Shift 联合梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_host/op_api/aclnn_modulate_backward.cpp` | `GetWorkspaceSize` | scale/shift 均 present | 两者连续化/cast |
| Tiling | `op_host/modulate_grad_tiling.cpp` | `TilingFunc` | flags 11 | R-004/key0 |
| Body | `op_kernel/modulate_grad.cpp` | `ComputeGradients` | 两个 flag 均真 | 合并两种归约 |

#### Device 数据流

##### 函数级调用链

```text
modulate_grad -> CopyIn(input,g,scale) -> ComputeGradients
  -> per row: shift+=g; grad_input=g*scale; current=x*g; scale+=current
  -> persistent scale/shift accumulators -> last tile双输出
```

##### 源码等价伪码

```text
for each (b,D chunk): persistent_scale=0; persistent_shift=0
  for each L tile: tile_scale=0; tile_shift=0
    for l: tile_shift+=g; grad_input=g*scale; tile_scale+=x*g
    persistent_scale+=tile_scale; persistent_shift+=tile_shift
  write grad_input tiles; on last tile write both reduced outputs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/g/scale Q | 同 R-001 | `2*len+row` B | FP32 | DataCopyPad | Mul/Add | tile | depth1 |
| gradInputQ | UB | len B | FP32 | Mul | CopyOut | tile | depth1 |
| 两组 tile/persistent + current | UB | `5*row` B | FP32 | Duplicate/Mul/Add | Add/GM | tile/chunk | Queue/PIPE_V |
| 三路 GM output | 对应输入 offset / `b*D+d0` | 有效 tile/chunk | FP32 | DMA | ACLNN casts | executor | 无跨核重叠 |

##### 分层内存布局

与公共 9 队列图相同，所有队列均激活，是最大计算路径。

##### 字节账与复用顺序

```text
UB_total = 3*len + 6*row <= 110592 B
workspace_total(operator node) = GetLibApiWorkSpaceSize(); body 不读 workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| currentScale | Mul(x,g) | Add(tileScale,current) | 下一行 | PIPE_V | Vector Add 完成 |
| tile accumulators | per-row Adds | persistent Adds | 下一 tile | PIPE_V/FreeTensor | persistent 已更新 |
| persistent accumulators | per-tile Adds | last tile DMA | 下一 D chunk | Queue + FreeTensor | last write完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | DataCopyPad + PIPE_ALL | GM | Vector | 输入 tile | 每行补0，计算 mask为 d_size |
| Vector chain | PIPE_V | Mul/Add | 后续 Add | current/tile/persistent | FP32，固定 L 顺序 |
| UB→GM | Queue + PIPE_ALL | Vector | DMA | 三输出 | D/L 尾块只写有效字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_grad_tiling.cpp` | `GenerateTilingData/TilingFunc` | 分核、flags、key0 |
| Kernel entry/router | `op_kernel/modulate_grad.cpp` | `modulate_grad` | 单 float AIV body |
| Device body/helper | 同上 | `Process/ComputeGradients/CopyOutScaleShift` | 双归约、同步、tail |

## 算子特定优化

### Trick T-001：单核拥有 `(batch,D分片)` 并在 UB 跨 L tile 累加

#### 触发与源码依据

- 适用 Route：R-002、R-003、R-004
- 触发条件：对应可选输出 present；同一核依次处理其 L tiles
- Golden 来源：[`modulate_grad.cpp:364-372`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate_grad/op_kernel/modulate_grad.cpp#L364-372)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tileLIndex == 0) {
    computegradShiftLocal_ = gradShiftLocalQueue_.AllocTensor<float>();
    computegradScaleLocal_ = gradScaleLocalQueue_.AllocTensor<float>();
    Duplicate<T>(computegradShiftLocal_, static_cast<T>(0), d_size);
    Duplicate<T>(computegradScaleLocal_, static_cast<T>(0), d_size);
    gradShiftLocalQueue_.EnQue(computegradShiftLocal_);
    gradScaleLocalQueue_.EnQue(computegradScaleLocal_);}
computegradShiftLocal_ = gradShiftLocalQueue_.DeQue<float>();
computegradScaleLocal_ = gradScaleLocalQueue_.DeQue<float>();
```

#### 机制与设计原因

分核保证每个 `(b,d chunk)` 只有一个 AIV 核 owner。persistent accumulator 只在首个 L tile 清零，随后以 Queue 形式跨 tile 传递，最后一次才写 GM；因此不需要 GM 中间和跨核 Atomic。tile accumulator 与 `currenttileGradScale` 每 tile/每行复用，Vector `Mul→Add` 之间由 `PIPE_V` 建立可见性。

#### 迁移边界

- 适用：沿连续轴外的 L 维归约，且可将每个输出 D 分片独占给一个核。
- 前提：FP32 accumulator、D chunk/UB 公式与单核 owner 均成立；last-tile 写回协议不变。
- 失效：若同一 `(b,d)` 被多核处理，必须增加跨核归约/Atomic 或第二 Kernel；固定实现没有该 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从源码协议可证明，若每个 L tile 都重新清零 persistent accumulator，会只保留最后 tile，但这不是固定仓中的实际反例代码。

## 模板与实现对比

| 对比维度 | R-001 | R-002 | R-003 | R-004 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | flags00 | flags10 | flags01 | flags11 | mutually exclusive presence flags |
| 模板与 dtype | `Kernel<float>` | 同 | 同 | 同 | ACLNN 统一 FP32 |
| 循环与分核 | 公共 B/D/L | 同 | 同 | 同 | 单核 owner 避免跨核归约 |
| Queue/TBuf | 9 Q，部分不消费 | 激活 scale/scale accum | 激活 shift accum | 全激活 | 初始化布局固定 |
| 内存总占用与复用 | `3len+6row` | 同 | 同 | 同 | 无条件 InitBuffer |
| 搬运/Vector API | copy | Mul/Mul/Add | Add/copy | 两者合并 | presence 语义 |
| 同步、精度与 tail | PIPE_ALL/FP32 | 加 PIPE_V | PIPE_ALL | PIPE_ALL+PIPE_V | D/L tail 公共 |
| fallback/失败边界 | 无 key fallback | 同 | 同 | 同 | unknown key 仍执行 float body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` | 公开支持；`ascend910_93` | 公开文档未列出，且无 `ascend950` 接线 |
| Host/Launcher | 与 A3 共用源码和 binary schema | 与 A2 共用 | 不生成卡 |
| Kernel/资源 | 同一 key0 float AIV body | 同一 body | 固定仓无实现闭包 |

## 限制与失败边界

- 文档将 `scale/shift` 声明为可选，但 `CheckMaxDimension` 和 `CheckDimension` 在 presence guard 前直接调用二者的方法；Device `CopyInScale` 也无条件从 `Scale` GM 读取。因此含 null optional 指针的 R-001/R-003 在固定源码存在空指针/无效 GM 访问风险；本次未运行验证，不能把它推断为可安全执行。
- `CheckParams` 未比较 `grad_output` 与 `input` shape，非 ND 仅告警；公开约束仍要求两者同 shape、ND。
- public dtype 检查没有强制所有输入/输出同 dtype；`grad_scale/grad_shift` 的结果 cast 使用 `grad_input` dtype。调用者应遵守文档的配对 dtype/shape，固定源码没有更完整的保护。
- `D/L/B` 为 uint64 POD，但 Device 大量 tile、stride 与分核变量为 uint32，且 Host 未检查窄化、乘法溢出、`DataCopyExtParams.blockCount` 的 uint16 上限；超大 shape 没有源码可见 fallback。
- `TILING_KEY_IS(0)` 分支为空，unknown key 没有拒绝路径；当前 Host 始终产生 key0。
- 固定仓未展开框架 Contiguous/Cast/ViewCopy 的 Kernel、workspace slice 与内部同步；这些按 ACLNN executor 框架边界处理。算子自有 Host/Tiling/POD/Kernel/构建链已闭合。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `vfusion/modulate_grad/docs/aclnnModulateBackward.md` | 两段式原型、产品表 | 公开语义与 A2/A3 |
| 入口实现 | `vfusion/modulate_grad/op_host/op_api/aclnn_modulate_backward.cpp` | `CheckParams/GetWorkspaceSize/aclnnModulateBackward` | 校验、executor 图、失败传播 |
| Launcher | `vfusion/modulate_grad/op_host/op_api/modulate_grad.cpp` | `l0op::ModulateGrad` | 输出分配与单 AICore node |
| POD | `vfusion/modulate_grad/op_host/modulate_grad_tiling.h` | `ModulateGradTiling` | ABI 字段 |
| Host/Tiling | `vfusion/modulate_grad/op_host/modulate_grad_tiling.cpp` | `GenerateTilingData/TilingFunc` | 分核、flags、workspace、key |
| Kernel entry/body | `vfusion/modulate_grad/op_kernel/modulate_grad.cpp` | `modulate_grad/ModulateGradKernel<float>` | AIV 全实现 |
| 平台注册 | `vfusion/modulate_grad/op_host/modulate_grad_def.cpp` | `ModulateGrad::ModulateGrad` | dtype/format 与 910b/910_93 |
| A2 binary | `vfusion/modulate_grad/op_host/config/ascend910b/modulate_grad_binary.json` | `ModulateGrad` entries | 三 dtype binary schema |
| A3 binary | `vfusion/modulate_grad/op_host/config/ascend910_93/modulate_grad_binary.json` | `ModulateGrad` entries | 三 dtype binary schema |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `ModulateGrad` | compute units 与 auto_sync |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 单 key、presence flags、分核参数 | 字段顺序/位宽；inactive 字段仍序列化 | `ModulateGradTiling` |
| 调用顺序 | executor 前处理→单 Kernel→输出 cast/copy | FP32 转换与 stream 顺序 | `GetWorkspaceSize/CommonOpExecutorRun` |
| workspace/内存 | 9 个 depth1 Q、`3len+6row` | 32B row padding、persistent accumulator lifetime | `InitPipeBuffers` |
| 同步 | Queue + PIPE_ALL/PIPE_V | Mul/Add 与覆盖顺序 | `Process/ComputeGradients` |
| 精度与尾块 | FP32 Kernel、D/L 双 tail | d_size 有效 lane、last L tile写归约 | `CalculateOffsets/CopyOutScaleShift` |
| 失败边界 | empty 快速返回、无 key fallback | optional null 风险、窄化缺口 | `CheckMaxDimension/CopyInScale/modulate_grad` |