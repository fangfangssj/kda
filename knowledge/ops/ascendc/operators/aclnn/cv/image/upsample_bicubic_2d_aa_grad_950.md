---
type: Operator
title: "UpsampleBicubic2dAAGrad 950 ACLNN 算子知识"
description: "说明 aclnnUpsampleBicubic2dAAGrad* 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/docs/aclnnUpsampleBicubic2dAAGrad.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/op_api/aclnn_upsample_bicubic2d_aa_grad.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/op_api/aclnn_upsample_bicubic2d_aa_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/op_api/upsample_bicubic2d_aa_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/upsample_bicubic2d_aa_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_data_copy.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_simt_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_simt_base_deter.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/arch35/upsample_bicubic2d_aa_grad_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/config/ascend950/upsample_bicubic2d_aa_grad_binary.json
    role: implementation
created_at: "2026-08-24T20:51:21Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# UpsampleBicubic2dAAGrad 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnUpsampleBicubic2dAAGradGetWorkspaceSize` / `aclnnUpsampleBicubic2dAAGrad`，无版本后缀。 |
| 计算语义 | 双三次抗锯齿插值反向；确定性路线逐输出 gather，默认非确定性路线逐输入 scatter+atomic。 |
| 输入输出 | 四维 NCHW/ND；FP16、FP32、BF16；`outputSize=[gradOutput.H,W]`，`inputSize=[N,C,gradInput.H,W]`。 |
| 执行类型 | R-001 为 AIV_VECTOR 数据搬运；R-002/R-003 为 SIMT。950 全部 Host 可达 Device 路线均不含 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | R-001 恒等 DataCopy；R-002 确定性 SIMT gather；R-003 非确定性 SIMT atomic scatter；后两者各有 32/64 位索引实例。 |
| 核心资源 | DataCopy 使用 GM 和双缓冲 `TQueBind` UB；SIMT 直接访问 GM、仅用线程局部标量与 atomic，无 dynamic UB；Host 固定预留 16 MiB workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN、RegBase selector/POD、全部可达模板 key、AIV copy、确定性/非确定性 SIMT body 及 950 构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `gradOutput` 为 `[N,C,Hin,Win]`，`gradInput` 为 `[N,C,Hout,Wout]`。每轴尺度为 `alignCorners ? (out-1)/(in-1) : (originalScale>0 ? originalScale : out/in)`；`invScale=(scale>=1 ? 1/scale : 1)`，抗锯齿支撑域 `support=(scale>=1 ? 2*scale : 2)`。三次核采用 `a=-0.5`，H/W 权重分别按当前支撑域归一化。

确定性实现让每个输出元素只有一个线程：遍历所有影响该输出的 `gradOutput`，以 FP32 标量累加后写回。非确定性实现让每个输入梯度线程枚举目标支撑域，把每项贡献转成输出 dtype 后执行 `asc_atomic_add`，原子加的到达顺序不固定。空间 shape 和两尺度均精确恒等时直接逐字节搬运。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | 4D NCHW；ND 按 NCHW 解释 | FP16/FP32/BF16 | 非空；支持非连续输入，ACLNN 先添加 `Contiguous`。 |
| `outputSize` | 长度 2 | INT64 | 必须匹配 `gradOutput` 的 H/W，元素大于 0。 |
| `inputSize` | 长度 4 | INT64 | 必须匹配输出 `out` 的 N/C/H/W，元素大于 0。 |
| `alignCorners` | 标量 | bool | 角点对齐开关，默认 false。 |
| `scalesH/scalesW` | 标量 | ACLNN double，L0 attr float | 不得为负；正值在 ACLNN 端先以 double 求倒数，再窄化为 float。 |
| `out` | 4D NCHW/ND | 与 `gradOutput` 相同 | 支持非连续输出；内部临时结果经 `ViewCopy` 提交。 |

公开文档还要求每维 `<=2^20`、`N*C*Hin<2^31`，以及 `(Hin*Win+Hout*Wout+Hin*Wout)*N*C*sizeof(dtype)<60 GiB`。950 默认非确定性，可由运行时上下文确定性开关选择确定性计算。

### 入口与版本

固定仓只公开无后缀两段式接口。README/API 产品表明确支持 Ascend 950PR/950DT；OpDef 的 `ascend950` RegBase config 指向 `upsample_bicubic2d_aa_grad_apt`，950 binary JSON 覆盖 FP16/FP32/BF16，未发现平台支持冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParams` 依次检查 `gradOutput/inputSize/out` 非空、dtype 支持且输入输出一致、scale 非负、输入输出均四维、属性数组长度、NCHW 时 N/C 一致，以及正 shape 与 ND/NCHW 格式。950 因 `IsRegBase()` 不执行 A2/A3 的 `scale>=0.02` 与单维 `INT32_MAX` 检查。空 Tensor 在公共检查后返回 workspace 0。

非空路径依次添加 `Contiguous(gradOutput)`、对 `out` 的 `Contiguous`（其返回张量未成为核心 Kernel 参数）、把正 scale 的倒数由 double 窄化成 float、添加一个 `UpsampleBicubic2dAAGrad` AICore launcher，最后用 `ViewCopy` 把临时输出提交到调用者 `out`。

### 参数、Workspace 与生命周期

L0 分配与 `out` 同 view shape/dtype/format 的临时张量。RegBase Tiling 对所有非空路线设置 `workspaces[0]=16 MiB`；950 entry 虽调用 `GetUserWorkspace(workspace)`，但 copy/SIMT body 均不读取该地址，所以这是 executor 持有到 stream 完成的固定预留，而不是源码可见的算法中间张量。POD by-value 随 launch 复制；临时输出由 executor 持有并在核心 Kernel 完成后由同 stream 的 `ViewCopy` 消费。

### 启动图与失败传播

```text
aclnn...GetWorkspaceSize
  -> Contiguous(gradOutput)
  -> ADD_TO_LAUNCHER_LIST_AICORE(UpsampleBicubic2dAAGrad)
       -> common tiling detects RegBase
       -> Tiling4UpsampleBicubic2dAAGradRegbase: key/POD/blockDim/workspace
       -> one upsample_bicubic2d_aa_grad<schId,isInt32,isDeterministic>
            schId=0: AIV DataCopy
            schId=1,deter=1: SIMT gather
            schId=1,deter=0: output init -> SyncAll -> SIMT atomic scatter
  -> ViewCopy(tempOut,out)
aclnn... -> CommonOpExecutorRun(workspace,workspaceSize,executor,stream)
```

Tiling shape、dtype、format、attr 或平台资源检查失败返回 `GRAPH_FAILED`，Kernel 不启动。L0 添加 launcher 失败会让第一段返回内部错误，后续 `ViewCopy` 不会加入 executor。每条路线只有一个核心语义 Kernel，无跨 Kernel workspace 边；最终临时输出到 `ViewCopy` 的可见性由 executor 同 stream 顺序保证。

## Route 与参数设计

### 分核与资源计算

平台 producer 为 `coreNum=GetCoreNumAiv()`、`ubSize=GetCoreMemSize(UB)`。恒等条件是 `outH==inH && outW==inW && scaleH==1.0f && scaleW==1.0f`；它按输出总元素分核。若 `outSize<=coreNum*(128/dtypeSize)`，先取 `realCoreNum=outSize/(128/dtypeSize)`，再保证至少 1。每核基础长度 `outSize/realCoreNum`，前 `outSize%realCoreNum` 核各多 1 元素。

Copy 的 `oneBlockNum=ubBlockSize/dtypeSize`，`ubFactor=floor((ubSize-oneBlockNum*dtypeSize)/(2*dtypeSize)/oneBlockNum)*oneBlockNum`，单位为元素并按 UB block 对齐。确定性 SIMT 按 `outNCHW` 分核，非确定性 SIMT 按 `inNCHW` 分核，并另外把输出初始化区间均分。输入和输出总元素均 `<=INT32_MAX` 时选 32 位索引，否则选 64 位；对应每核 `asc_vf_call` 的线程数分别为 2048 和 512，dynamic UB 为 0。`blockDim=realCoreNum`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 恒等 DataCopy | `(schId=0,isInt32=0,isDeterministic=1)` | H/W shape 相同且 `scaleH==1.0f && scaleW==1.0f`；最高优先 | 非恒等进入 SIMT |
| 2 | R-002 确定性 SIMT | `(1,1,1)` 或 `(1,0,1)` | 非恒等，`context->GetDeterministic()!=0`；总元素决定索引宽度 | 未开启确定性进入 R-003 |
| 3 | R-003 非确定性 SIMT | `(1,1,0)` 或 `(1,0,0)` | 非恒等，`context->GetDeterministic()==0` | 非法输入在 Tiling 失败 |

```text
if exact_identity:
  key=(0,0,1)
else if deterministic:
  key=(1, inputSize<=INT32_MAX && outputSize<=INT32_MAX, 1)
else:
  key=(1, inputSize<=INT32_MAX && outputSize<=INT32_MAX, 0)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 DataCopy | 0,0,1 | 950；三 dtype；四维合法且精确恒等 | 单 Kernel AIV_VECTOR | `Bicubic2dAAGradDataCopy<DTYPE_GRAD_OUTPUT>` | GM→depth-2 UB queue→GM | 每核余数和最后 UB tile |
| R-002 deterministic | 1,{1/0},1 | 950；非恒等；确定性；总元素决定 32/64 位 | 单 Kernel SIMT | `Bicubic2dAAGradSimt<T,uint32_t,int32_t,1,1>` 或 `<T,uint64_t,int64_t,1,1>` → `SimtDeterCompute` | 直接 GM gather；每输出唯一写者；无 dynamic UB/atomic | thread-stride、support clamp；无 fallback |
| R-003 non-deterministic | 1,{1/0},0 | 950；非恒等；确定性关闭 | 单 Kernel SIMT | 对应索引模板 → `SimtCompute` | 先 GM 清零、`SyncAll`，再 scatter `asc_atomic_add` | thread-stride、零权重跳过；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 恒等 fast path | shape 相等且两 scale 精确 `==1.0f` | 直接搬运避免权重计算 | R-001 | R-002/R-003 | `CalTilingData` |
| 索引宽度 | 输入、输出总元素都 `<=INT32_MAX` | 32 位 magic division 与地址安全 | R-002/R-003 | 64 位模板 | `CheckInputParams` |
| SIMT threads | 32 位 2048；64 位 512 | 分别匹配 `LAUNCH_BOUND` | R-002/R-003 | 编译期另一实例 | `THREAD_NUM_B32/B64` |
| Copy UB | 上述 `ubFactor` 且按 `oneBlockNum` 对齐 | depth 2 queue 必须落入平台 UB | R-001 | 固定仓无 `ubFactor==0` fallback | `ComputeDataCopy` |
| atomic 前清零 | non-deter route | atomic add 的初值必须为 0 | R-003 | `InitOutput` 后 `SyncAll` | `Bicubic2dAAGradSimt::Process` |

### POD/ABI

`UpsampleBicubic2dAAGradRegBaseTilingData` 是自然 8 字节对齐的标准布局结构：`sizeof=112`、`alignof=8`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `UpsampleBicubic2dAAGradRegBaseTilingData` | `blkProcessNum` | int64/64 | offset 0 | 元素/core | `CalTilingData` | copy/SIMT core span | 非负 |
| 同上 | `lenN` | int64/64 | 8 | 元素 | input N | SIMT index 参数 | 正 |
| 同上 | `lenC` | int64/64 | 16 | 元素 | input C | SIMT index 参数 | 正 |
| 同上 | `inH` | int64/64 | 24 | 元素 | gradOutput H | SIMT source axis | 正 |
| 同上 | `inW` | int64/64 | 32 | 元素 | gradOutput W | SIMT source axis | 正 |
| 同上 | `outH` | int64/64 | 40 | 元素 | gradInput H | SIMT destination axis | 正 |
| 同上 | `outW` | int64/64 | 48 | 元素 | gradInput W | SIMT destination axis | 正 |
| 同上 | `alignCorners` | int64/64 | 56 | bool-as-int | attr | Device body 不读取 | inactive |
| 同上 | `ubFactor` | int32/32 | 64 | 元素 | `ComputeDataCopy` | copy loop/tile | SIMT inactive |
| 同上 | `tailBlockNum` | int32/32 | 68 | 元素余数 | route 分核 | core span | `<realCoreNum` |
| 同上 | `scaleH` | float/32 | 72 | 坐标比 | `ComputeScalesSupportValues` | SIMT | copy inactive |
| 同上 | `scaleW` | float/32 | 76 | 坐标比 | 同上 | SIMT | copy inactive |
| 同上 | `invScaleH` | float/32 | 80 | 权重尺度 | `scaleH>=1?1/scaleH:1` | SIMT | 正 |
| 同上 | `invScaleW` | float/32 | 84 | 权重尺度 | 同上 | SIMT | 正 |
| 同上 | `supportH` | float/32 | 88 | 坐标 | `scaleH>=1?2*scaleH:2` | SIMT loop bound | `>=2` |
| 同上 | `supportW` | float/32 | 92 | 坐标 | 同上 | SIMT loop bound | `>=2` |
| 同上 | `initHeadCoreNum` | int32/32 | 96 | core | non-deter `outNCHW%realCoreNum` | output init split | 其它路线 0 |
| 同上 | ABI padding | 32 bits | 100 | padding | `memset_s` 清零 | 无 | 保持 0 |
| 同上 | `perCoreInitEle` | int64/64 | 104 | 元素/core | non-deter `outNCHW/realCoreNum` | output init | 其它路线 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 0,0,1 | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp::CalTilingData/ComputeDataCopy` | RegBase POD | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp::upsample_bicubic2d_aa_grad` | `Bicubic2dAAGradDataCopy` | ascend950 OpDef/binary JSON |
| R-002 | 1,{1/0},1 | `CalTilingData` deterministic branch | 同上 | 同上 | `Bicubic2dAAGradSimt` + `SimtDeterCompute` + cubic helpers | 同上 |
| R-003 | 1,{1/0},0 | `CalTilingData` non-deterministic branch | 同上 | 同上 | `Bicubic2dAAGradSimt` + `SimtCompute` + cubic helpers | 同上 |

## Kernel 实现

### 公共执行流程

模板 entry 参数依次为 `x,y,workspace,tiling`，声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 并解析统一 POD。`schId==0` 实例化 copy；否则 `isInt32` 选择 `(uint32_t,int32_t)` 或 `(uint64_t,int64_t)`，`isDeterministic` 选择 gather 或 atomic scatter callee。所有路线直接读写 GM；SIMT 的 `asc_vf_call(dim3(THREAD_NUM_*),...)` 在每个 AIV block 内启动固定线程数，未提供 dynamic UB 参数。

### Route R-001：恒等 DataCopy

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_upsample_bicubic2d_aa_grad.cpp` | `CheckParams` | 公共合法输入 | 进入 launcher |
| Tiling | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp` | `CalTilingData` | H/W shape 相同且 scaleH/W `==1` | key 0,0,1 |
| Router | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp` | `if constexpr(schId==0)` | schId 0 | `Bicubic2dAAGradDataCopy<T>` |

#### Device 数据流

##### 函数级调用链

```text
upsample_bicubic2d_aa_grad<0,0,1>
  -> Bicubic2dAAGradDataCopy<T>::Init
  -> Process
     -> CopyIn: AllocTensor<uint8_t> -> DataCopyPad(GM, blockCount=1, blockLen=valid*sizeof(T)) -> EnQue
     -> CopyOut: DeQue -> DataCopyPad(GM, blockCount=1, blockLen=valid*sizeof(T)) -> FreeTensor
```

##### 源码等价伪码

```text
# 源码等价伪码
coreLen = floor(outSize/realCoreNum) + (blockIdx < remainder)
coreOffset = blockIdx*floor(outSize/realCoreNum) + min(blockIdx,remainder)
for loopIdx = 0; loopIdx < coreLen; loopIdx += ubFactor:
  valid = min(ubFactor, coreLen-loopIdx)
  DataCopyPad(input[coreOffset+loopIdx], valid*sizeof(T)) GM->UB
  DataCopyPad(output[coreOffset+loopIdx], valid*sizeof(T)) UB->GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `totalOffset_+loopIdx` 元素 | `valid` 元素 | T/NCHW linear | caller | `CopyIn` | caller | 仅搬有效字节 |
| `dataQue_` | UB queue slot | depth 2 × `ubFactor*sizeof(T)` | uint8 payload | `CopyIn` | `CopyOut` | TPipe/Kernel | `ubFactor` 按 UB block 对齐 |
| GM output | 同 input offset | `valid` 元素 | T/NCHW linear | `CopyOut` | caller/ViewCopy | executor temp tensor | last tile 不越界 |
| workspace | N/A：body 未读 | Host 固定 16 MiB | N/A | N/A | N/A | executor 至 stream 完成 | 不参与算法 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  slot0["dataQue slot 0 · ubFactor*sizeof(T) bytes"]:1
  slot1["dataQue slot 1 · ubFactor*sizeof(T) bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2 * ubFactor * sizeof(T)
UB_total <= ubSize - oneBlockNum*sizeof(T)  # Host 公式保留一个 UB block
dynamic_ub = N/A
workspace_total = 16 MiB reserved; 0 bytes consumed by body
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQue_` slot | `CopyIn/EnQue` | `DeQue/CopyOut` | 后续 loop tile | `FreeTensor` | 同一 slot 释放后才能复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB→GM | EnQue/DeQue/Free queue protocol | MTE2 | MTE3 | UB slot | byte-preserving，无 cast；`valid=min(...)` 处理尾 tile |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp` | `ComputeDataCopy` | core split、tail、UB factor |
| Kernel entry/router | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp` | `<0,0,1>` | copy 编译期分支 |
| Device body/helper | `op_kernel/arch35/upsample_bicubic2d_aa_grad_data_copy.h` | `Init/Process/CopyIn/CopyOut` | depth-2 queue、DMA 参数和 tail |

### Route R-002：确定性 SIMT gather

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp` | `CalTilingData` | 非恒等且 `baseTiling_.isDeterministic` | key 1,isInt32,1；按输出域分核 |
| Router | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp` | `isInt32/isDeterministic` | deter=1 | 32/64 位 `Bicubic2dAAGradSimt` |

#### Device 数据流

##### 函数级调用链

```text
entry -> Bicubic2dAAGradSimt<T,indexU,indexS,1,1>::Init/Process
  -> GetUintDivMagicAndShift(outW/outH)
  -> asc_vf_call<calleeDeterInt32>(dim3(2048)) 或 calleeDeterInt64(dim3(512))
  -> SimtDeterCompute
     -> CalculateBounds -> CalculateWeight -> CubicFilterAA
     -> FP32 value accumulation -> outputGm[yGmIdx]=value
```

##### 源码等价伪码

```text
# 源码等价伪码
for localIdx = threadIdx.x; localIdx < coreOutputCount; localIdx += blockDim.x:
  y = coreOutputStart + localIdx
  decode Batch,H,W with precomputed magic division
  derive clipped gradOutput bounds from scale/support
  value = 0.0f
  for h,w in bounds:
    weightH/W = cubic_weight / total_axis_weight
    value += float(input[Batch,h,w]) * weightH * weightW
  output[y] = T(value)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `Batch*Hin*Win+h*Win+w` | clipped support rectangle | T/NCHW | caller | SIMT threads | caller | read-only |
| thread-local | indices、bounds、weights、FP32 `value` | scalar lifetime | int32/64 + FP32 | callee | same thread | registers/compiler | exact register bytes不可见 |
| GM output | `blkStartOffset+localIdx` | one T element per iteration | T/NCHW | unique thread | ViewCopy | executor temp | no atomic/conflict |
| UB/shared/dynamic UB/workspace | N/A：SIMT directly GM | dynamic UB 0；workspace reserved 16 MiB | N/A | N/A | N/A | executor workspace unused | no local allocation |

##### 分层内存布局

N/A：该 SIMT Route 不调用 `InitBuffer`，不使用 shared/dynamic UB；真实替代路径是 GM 输入 → 线程局部标量/寄存器 → GM 输出。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = 0
workspace_total = 16 MiB reserved; body use 0
register bytes = compiler allocation，固定仓不可见；lifetime limited to one thread iteration
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `value/weights` | current thread iteration | current output store | next strided iteration | program order | 每输出唯一线程，无跨线程别名 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| output gather | unique-writer program order | SIMT thread | Kernel completion/ViewCopy | one output element | 输入转 FP32并 FP32 累加，最终 store 窄化到 T |
| tail | `localIdx<blkProcessNum` + clipped bounds | thread/core split | loop exit | last core/thread/support | 不写越界元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp` | deterministic branch | output-domain blockDim/core span |
| Kernel entry/router | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp` | `<1,isInt32,1>` | 32/64 位模板选择 |
| Device body/helper | `op_kernel/arch35/upsample_bicubic2d_aa_grad_simt_base_deter.h` | `SimtDeterCompute` | bounds、normalized weights、unique store |

### Route R-003：非确定性 SIMT atomic scatter

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp` | `CalTilingData` | 非恒等且 deterministic=0 | key 1,isInt32,0；按输入域分核 |
| Router | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp` | `isDeterministic` | 0 | `calleeInt32/64` |

#### Device 数据流

##### 函数级调用链

```text
Bicubic2dAAGradSimt<T,indexU,indexS,1,0>::Process
  -> InitOutput(output per-core slice, 0) -> SyncAll
  -> GetUintDivMagicAndShift(inW/inH)
  -> asc_vf_call<calleeInt32>(2048) 或 calleeInt64(512)
  -> SimtCompute -> CalculateBoundariesAndWeights/CubicFilterAA
  -> asc_atomic_add(outputGm+outputIdx, T(value))
```

##### 源码等价伪码

```text
# 源码等价伪码
parallel_zero(output assigned slice)
SyncAll()
for localIdx = threadIdx.x; localIdx < coreInputCount; localIdx += blockDim.x:
  x = coreInputStart + localIdx
  decode NC,H,W
  compute normalized H/W destination support
  for h,w in support:
    contribution = T(float(input[x]) * weightH * weightW)
    atomic_add(output[NC,h,w], contribution)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM output init | `initCoreOffset` | `perCoreInitEle + head` | T/NCHW | `InitOutput` | atomic phase | executor temp | 各核区间不重叠 |
| GM input | `blkStartOffset+localIdx` | one T element | T/NCHW | caller | SIMT thread | caller | read-only |
| thread-local | bounds、weights、FP32 contribution | scalars | int32/64 + FP32 | thread | same thread | registers/compiler | next iteration reuse |
| GM output atomic | `NC*Hout*Wout+h*Wout+w` | one T element/add | T/NCHW | multiple threads | Kernel result/ViewCopy | executor temp | atomic resolves collision |
| UB/shared/dynamic UB/workspace | N/A：SIMT directly GM | 0 local bytes；16 MiB workspace unused | N/A | N/A | N/A | executor | no allocation |

##### 分层内存布局

N/A：该 SIMT Route 无 `InitBuffer`、shared 或 dynamic UB；真实路径为 GM 输出分区清零 → `SyncAll` → GM 输入/线程局部权重 → GM atomic 输出。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = 0
workspace_total = 16 MiB reserved; body use 0
register bytes = compiler allocation，固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM output | `InitOutput` | atomic writers | 不覆盖，只原子累加 | `SyncAll` | 全输出已清零后才允许 atomic |
| thread-local scalars | current source iteration | atomic contribution | next strided source | program order | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| init→scatter | `SyncAll` | all cores output zeroing | all atomic writers | full GM output | first add observes zero |
| accumulation | `asc_atomic_add` | source threads | GM result | colliding output element | 每项在 FP32 计算后先 cast T 再 atomic；顺序非确定 |
| tail | thread-stride guard、clipped support、zero-weight skip | threads | loop exit | input/output tail | 不越界、不写零贡献 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp` | non-deterministic branch | input split、output init split |
| Kernel entry/router | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp` | `<1,isInt32,0>` | atomic callee selection |
| Device body/helper | `op_kernel/arch35/upsample_bicubic2d_aa_grad_simt_base.h` | `SimtCompute` | source scatter、normalization、atomic |

## 算子特定优化

### Trick T-001：确定性 gather 与非确定性 atomic scatter 的编译期双路线

#### 触发与源码依据

- 适用 Route：R-002/R-003
- 触发条件：`context_->GetDeterministic()` 写入 key 的 `isDeterministic` 维度。
- Golden 来源：[`op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp:132-155`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp#L132-L155)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
} else if (baseTiling_.isDeterministic) {
    baseTiling_.schId = CONST_1;
    int64_t allNum = outNCHW;
    baseTiling_.blkProcessNum = allNum / static_cast<int64_t>(baseTiling_.realCoreNum);
} else {
    baseTiling_.schId = CONST_1;
    int64_t allNum = inNCHW;
    baseTiling_.blkProcessNum = allNum / static_cast<int64_t>(baseTiling_.realCoreNum);
}
```

#### 机制与设计原因

确定性路线以输出为任务域，单线程闭合一个输出的全部贡献；非确定性路线以输入为任务域，源点贡献并行 scatter，以 atomic 解决冲突。selector 把算法选择编译进 key，Device 主循环没有运行时确定性分支。线程数、blockDim 和 `blkProcessNum` 分别来自索引模板常量、平台 AIV 核数和当前任务域总元素。

#### 迁移边界

- 适用：反向插值、scatter-add 等可在逆向 gather 与正向 scatter 间转换的算子。
- 前提：gather 可枚举逆支撑域；scatter 输出 dtype 支持 atomic add。
- 失效：要求严格确定性时必须用 R-002；不支持 atomic 的 dtype 也必须改为 gather/归约。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

公开文档明确 950 默认非确定性，不能把 R-003 的 atomic 到达顺序描述为确定性结果。

### Trick T-002：按总元素范围切换索引位宽与 SIMT 线程数

#### 触发与源码依据

- 适用 Route：R-002/R-003
- 触发条件：输入、输出总元素均不超过 `INT32_MAX` 时 `isInt32=1`。
- Golden 来源：[`op_kernel/upsample_bicubic2d_aa_grad_apt.cpp:42-49`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_bicubic2d_aa_grad/op_kernel/upsample_bicubic2d_aa_grad_apt.cpp#L42-L49)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr(isInt32 == 1) {
    Bicubic2dAAGradSimt<DTYPE_GRAD_OUTPUT, uint32_t, int32_t, schId, isDeterministic> op;
    op.Init(x, y, &tilingData);
    op.Process();
} else {
    Bicubic2dAAGradSimt<DTYPE_GRAD_OUTPUT, uint64_t, int64_t, schId, isDeterministic> op;
```

#### 机制与设计原因

32 位模板使用 2048 threads 和 32 位 magic division，64 位模板使用 512 threads 和 64 位地址/除法。Host 用完整输入、输出元素数而非单一维度证明 32 位安全，从而在小范围使用更窄索引，同时保留大 Tensor 的正确性。

#### 迁移边界

- 适用：线性地址均受输入/输出总元素上界约束的 SIMT 算子。
- 前提：全部地址乘积确实不超过所选位宽。
- 失效：任一中间地址可能超过 `INT32_MAX` 时必须使用 64 位模板。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能只检查单轴维度后选择 32 位；固定源码同时检查输入与输出总元素。

## 模板与实现对比

| 对比维度 | R-001 DataCopy | R-002 deterministic | R-003 non-deterministic | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 精确恒等，最高 | 非恒等+确定性 | 非恒等+确定性关闭 | shape/scale 与上下文开关 |
| 模板与 dtype | `DataCopy<T>`；三 dtype | `Simt<T,32/64,1,1>`；三 dtype | `Simt<T,32/64,1,0>`；三 dtype | dtype binary + key specialization |
| 循环与分核 | 输出元素/core，UB tile loop | 输出元素/core，thread stride | 输入元素/core，thread stride；另分输出初始化 | 避免/管理写冲突 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth-2 `TQueBind` | 线程局部标量；dynamic UB 0 | 同；dynamic UB 0 | copy 才需搬运队列 |
| 内存总占用与复用 | `2*ubFactor*sizeof(T)`；queue slot 复用 | UB 0；寄存器字节不可见 | UB 0；GM output 初始化后 atomic | 路线数据流差异 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad` | `asc_vf_call` + gather | `InitOutput` + `asc_vf_call` + `asc_atomic_add` | 恒等/确定性选择 |
| 同步、精度与 tail | queue；字节保持；valid tail | unique writer；FP32 累加后 cast；thread tail | `SyncAll`+atomic；贡献先 cast T；thread tail | 数值与冲突语义 |
| fallback/失败边界 | 非恒等转 SIMT | 关闭确定性转 R-003 | 无算法 fallback | 非法输入统一 Tiling 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀 CV 卡 | 公开支持，见无后缀 CV 卡 | 公开支持，本卡 |
| Host/Launcher | 非 RegBase Tiling，dtype key | 同 | RegBase 决策树，三维模板 key |
| Kernel/资源 | Vector+Cube 单 Kernel，GM intermediate/radio | 同 | AIV DataCopy 或 SIMT；无 Cube/MIX_CV |

## 限制与失败边界

- 950 Tiling 拒绝空 input/output、非四维、dtype/format 不匹配与属性 shape 不匹配；ACLNN 空 Tensor shortcut 在进入 Tiling 前返回。
- 恒等 fast path 使用精确 float `==1.0f`，不是 A2/A3 的 `1e-6` 近似判断。
- 非确定性路线将每项 FP32 contribution 先转输出 dtype 再 atomic；确定性路线先 FP32 累加再最终窄化，舍入顺序不同。
- `ubFactor` 没有显式 `==0` 检查；若平台 UB 不足，copy 循环步长无仓内 fallback。
- 950 body 不消费固定 16 MiB workspace，不可据此假定存在算法中间张量。
- SIMT 寄存器实际字节数和 `asc_vf_call` 底层调度属于编译器/运行库边界；源码明确未申请 shared/dynamic UB。
- `CheckNotNull` 未显式检查 `outputSize`，随后 `CheckShape` 会解引用它；这是固定源码可见的空指针失败边界。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `docs/aclnnUpsampleBicubic2dAAGrad.md` | 产品表、函数原型、确定性说明 | 950 公开支持与接口约束 |
| 入口声明/约束 | `op_host/op_api/aclnn_upsample_bicubic2d_aa_grad.h` | 两段式 ACLNN | ABI 原型 |
| 入口实现 | `op_host/op_api/aclnn_upsample_bicubic2d_aa_grad.cpp` | `CheckParams`、两段入口 | 检查、前处理、executor |
| 入口实现 | `op_host/op_api/upsample_bicubic2d_aa_grad.cpp` | `l0op::UpsampleBicubic2dAAGrad` | 单 launcher 节点 |
| Host/Tiling/Launcher | `op_host/upsample_bicubic2d_aa_grad_tiling.cpp` | `RunBigKernelTiling` | RegBase 分发 |
| Host/Tiling/Launcher | `op_host/upsample_bicubic2d_aa_grad_tiling_arch35.cpp` | RegBase Tiling class | selector、POD、分核、workspace |
| Kernel entry/body | `op_kernel/upsample_bicubic2d_aa_grad_apt.cpp` | 模板 entry | AIV_ONLY 与 key router |
| Kernel entry/body | `op_kernel/arch35/upsample_bicubic2d_aa_grad_tiling_data.h` | RegBase POD | ABI |
| Kernel entry/body | `op_kernel/arch35/upsample_bicubic2d_aa_grad_tiling_key.h` | key declarations | 合法模板集合 |
| Kernel entry/body | `op_kernel/arch35/upsample_bicubic2d_aa_grad_data_copy.h` | DataCopy class | AIV 搬运路线 |
| Kernel entry/body | `op_kernel/arch35/upsample_bicubic2d_aa_grad_simt.h` | SIMT wrapper | core/thread 启动与同步 |
| Kernel entry/body | `op_kernel/arch35/upsample_bicubic2d_aa_grad_simt_base.h` | `SimtCompute` | atomic scatter |
| Kernel entry/body | `op_kernel/arch35/upsample_bicubic2d_aa_grad_simt_base_deter.h` | `SimtDeterCompute` | deterministic gather |
| Kernel entry/body | `op_kernel/arch35/upsample_bicubic2d_aa_grad_common.h` | `CubicFilterAA` | 三次权重 helper |
| 构建接线 | `op_host/upsample_bicubic2d_aa_grad_def.cpp` | ascend950 config | `_apt` 平台接线 |
| 构建接线 | `op_host/config/ascend950/upsample_bicubic2d_aa_grad_binary.json` | `op_list` | 三 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 112-byte RegBase POD 与三维 key | field offset、索引位宽、inactive 字段清零 | tiling data/key headers |
| 调用顺序 | 单核心 Kernel 后 ViewCopy | non-deter 必须 init→`SyncAll`→atomic | ACLNN executor、SIMT `Process` |
| workspace/内存 | 固定 16 MiB；copy depth-2 UB | 950 body不读 workspace；queue 容量公式 | `DoTiling`、DataCopy `Init` |
| 同步 | queue 或全核 barrier+atomic | 不能删除 non-deter `SyncAll` | DataCopy/SIMT `Process` |
| 精度与尾块 | deterministic FP32 gather；non-deter T atomic | 两种窄化顺序、core/thread/support tail | SIMT base headers |
| 失败边界 | Tiling 失败终止，无 Device fallback | 不把 A2/A3 scale guard误加到 950 | `CheckInputParams/CheckInputShapeAndAttr` |