---
type: Operator
title: "LogitGrad 950 ACLNN 算子知识"
description: "说明 aclnnLogitGrad 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/docs/aclnnLogitGrad.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_graph/logit_grad_proto.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/logit_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_host/config/ascend950/logit_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
created_at: "2026-08-25T17:14:22Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# LogitGrad 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnLogitGradGetWorkspaceSize` / `aclnnLogitGrad`；固定仓无 V2。 |
| 计算语义 | FP32 内部计算 `q=dy/(x*(1-x))`；`eps<0` 仅对 `x∈[0,1]` 选择 q，否则 NaN；`eps>=0` 仅对 `x∈[eps,1-eps]` 选择 q，否则 0。 |
| 输入输出 | `x/dy/out` 为同 shape、同 dtype 的 ND；0D～8D、空 Tensor，BF16/FP16/FP32；公开支持非连续 Tensor。 |
| 执行类型 | 单 `AIV_VECTOR` 节点；`key=1/2/3` 到 `LogitGradND<half/float/bfloat16_t>`；无 Cube/AIC 或 `MIX_CV`。 |
| Kernel 路线 | R-001 FP16 升精度；R-002 FP32 原生；R-003 BF16 升精度。 |
| 核心资源 | GM；每核 192 KiB `VECCALC TBuf`，两个 96 KiB ping-pong 半区；无 TQue；Kernel 不读取 workspace。 |
| 来源闭包 | `source_closed`：公开 950 证据、OpDef/生成入口接线、Tiling/POD、router/body 和 `ascend950` binary/CMake 全闭合；生成 ACLNN bridge 是明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

```text
denom_i = x_i * (1 - x_i)
q_i = dy_i / denom_i
eps < 0  : dx_i = (0 <= x_i <= 1)       ? q_i : NaN
eps >= 0 : dx_i = (eps <= x_i <= 1-eps) ? q_i : 0
```

三条 Route 均先完成 FP32 除法再做双边 compare/And/Select；无效域并不是跳过除法，而是以 Select 覆盖。FP16/BF16 在两端 Cast。公开文档另称 `eps>1` 输出 NaN、`eps==1` 输出 Inf，但可见 body 在 `eps>=0` 时 false 标量恒为 0 且要求两个比较同时成立；这是固定文档与实现的边界冲突，本文不推测消解。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 0D～8D/ND | BF16、FP16、FP32 | 支持空/非连续 Tensor；OpDef `AutoContiguous()`。 |
| `dy` | 与 x 相同/ND | 与 x 相同 | 上游梯度，支持空/非连续 Tensor。 |
| `eps` | 标量 | API `double`；OpDef/Tiling/POD `float` | 默认和建议 `-1.0`；生成 bridge 的具体窄化语句固定仓不可见。 |
| `out(dx)` | 与 x 相同/ND | 与 x 相同 | 输出梯度；OpDef `AutoContiguous()`。 |

### 入口与版本

公开唯一版本是两段式 `aclnnLogitGradGetWorkspaceSize` 与 `aclnnLogitGrad`。固定仓没有这两个符号的手写实现；`op_host/CMakeLists.txt` 的 `ACLNNTYPE aclnn` 以及 `cmake/func.cmake::add_modules_sources` 把 `logit_grad_def.cpp` 接入 ACLNN OpDef 目标，因此 API inner bridge、executor 与连续化壳属于生成框架边界。README/API 产品表明确支持 Ascend 950PR/950DT；OpDef `AddConfig("ascend950")`、950 binary JSON 的三个 dtype 实例、Kernel CMake 的独立 `COMPUTE_UNITS ascend950` 形成一致证据，目录名不是唯一依据。

## Host/Launcher 执行流程

### 参数校验与前处理

公开契约要求 `x/dy/out` 非空，dtype 属于三种支持类型，且 dtype/format/shape 相同；空 Tensor 合法。文档给出空指针 `161001` 和非法 dtype `161002`，但生成 bridge 的精确检查顺序固定仓不可见。三路 OpDef 均为 ND + `AutoContiguous()`；非连续 Tensor 的 Contiguous/ViewCopy 节点、临时 workspace 与错误传播固定仓不可见。

`InferShape4LogitGrad` 把 x shape 复制给 dx，`InferDataType4LogitGrad` 把 x dtype 复制给 dx。`RunBigKernelTiling` 显式检查 input tensor、attrs、input desc，失败即 `GRAPH_FAILED`；不支持 dtype 同样失败。

### 参数、Workspace 与生命周期

设 `N=originShape.GetShapeSize()`、`C=PlatformAscendC(...).GetCoreNumAiv()`：

```text
needCoreNum = min(max(ceil(N/8192),1), C)
workspace[0] = 16*1024*1024 bytes
blockDim = needCoreNum
POD = {elementNum=N, eps=float attribute, needCoreNum}
```

调用者持有第一段返回的 Device workspace，executor 仅供第二段调用一次。可见 Tiling 固定写 16 MiB；entry 却将 `userWs=nullptr`，body 的 workspace 形参未读取，所以没有 workspace→Kernel 数据边。框架连续化可能追加的 workspace 结构固定仓不可见。

### 启动图与失败传播

```text
aclnnLogitGradGetWorkspaceSize                    # generated boundary
  -> LogitGrad OpDef / AutoContiguous
  -> InferShape/InferDataType -> TilingLogitGradTiling
  -> executor
aclnnLogitGrad(...,stream)
  -> K-001 logit_grad<<<needCoreNum,stream>>>(x,dy,dx,workspace,tiling)
     -> one LogitGradND<T>::Process()
```

| 节点/边 | entry/介质 | 参数与范围 | ordering、owner/lifetime 与失败传播 |
|---|---|---|---|
| Host→K-001 | `logit_grad` | 五个 GM 地址；POD；blockDim | 用户 stream 上单节点提交；launcher 符号固定仓不可见；第一段失败阻断第二段。 |
| 输入/输出 | Tensor GM | 每路 `N*sizeof(T)` | x/dy 只读、dx 写出；单 Kernel 无跨节点边。 |
| workspace | caller-owned GM | visible tiling slice 16 MiB | 生存期覆盖调用；K-001 不解引用。 |

## Route 与参数设计

### 分核与资源计算

Host 以 8192 元素为 tile，核数为 `ceil(N/8192)` 与 950 AIV 核数的较小值，空域强制 1。Device 计算 `totalTimes=ceil(N/8192)`、`base=totalTimes/needCoreNum`、`loopRemain=totalTimes%needCoreNum`；前 `loopRemain` 核取 `base+1` tile，其余取 `base` tile。后组起点通过 `base*blockIdx*8192 + N%(8192*needCoreNum)` 补偿 former tiles。`remain=N%8192` 由 `lastCoreNum=(loopRemain==0 ? needCoreNum-1 : loopRemain-1)` 的最后循环处理；空域 loopNum=0。

每核 `InitBuffer` 固定 192 KiB，两个半区各 96 KiB；每半区是 32 KiB x FP32 区、32 KiB dy FP32 区和 32 KiB scratch/mask alias 区。Host 未查询 UB 或提供小 UB fallback。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | 1 / `LogitGradND<half>` | `dtype==DT_FLOAT16` | FP32 检查。 |
| 2 | R-002 FP32 | 2 / `LogitGradND<float>` | 前项未命中且 `dtype==DT_FLOAT` | BF16 检查。 |
| 3 | R-003 BF16 | 3 / `LogitGradND<bfloat16_t>` | 前两项未命中且 `dtype==DT_BF16` | `GRAPH_FAILED`。 |

```text
FLOAT16 -> key 1
else FLOAT -> key 2
else BF16 -> key 3
else GRAPH_FAILED
entry (__CCE_AICORE__==220 || ==310): key 1/2/3 -> half/float/bfloat16_t
unknown key -> no body/no output
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1 | FP16、ND、0D～8D、950 config | 单 K-001 | `LogitGradND<half>` 五方法 | raw FP16→FP32；192 KiB；EVENT0/1 | count tail，无 fallback。 |
| R-002 | 2 | FP32、同上 | 单 K-001 | `LogitGradND<float>` | 直接 FP32；同一 UB/event | 同上。 |
| R-003 | 3 | BF16、同上 | 单 K-001 | `LogitGradND<bfloat16_t>` | BF16→FP32，输出 RINT；同一 UB/event | 同上。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| tile/UB | tile=8192；`InitBuffer=192KiB` | 两个 96 KiB 半区 | 全部 | 无更小路线 | `GetNeedCoreNum`、`Init` |
| compare | `rounded=ceil(count/64)*64` | FP32 compare repeat 对齐 | 全部 | mask 处理 tail | `ComputeStepTwo` |
| DMA 窄化 | `uint32_t(count*sizeof(T))` | `blockLen` 位宽；最大 32768 B | 全部 | tile 保证 | CopyIn/CopyOut |
| architecture | CCE AICore 220 或 310 | body 编译 guard | 全部 | 其它编译值为空 entry | `logit_grad` |
| empty | `N=0→core=1,totalTimes=0` | 保持非零 blockDim | 全部 | 正常空输出 | Tiling/Process |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LogitGradTilingData` | `elementNum` | int64/64 | 宏生成 sizeof/align/offset 固定仓不可见；声明第 1 | 元素 | `set_elementNum(GetShapeSize())` | Init/Process 分 tile | 0 合法；shape 乘积溢出策略不可见。 |
| `LogitGradTilingData` | `eps` | float/32 | 宏生成布局不可见；声明第 2 | 标量 | `set_eps(*(attrs->GetFloat(0)))` | Init 的 bounds/false value | API double 窄化代码不可见。 |
| `LogitGradTilingData` | `needCoreNum` | uint64/64 | 宏生成布局不可见；声明第 3 | AIV core | min/max/ceil 公式 | Init/Process 分核 | 平台核数为 0 时无额外 guard。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 1 | `RunBigKernelTiling` | `LogitGradTilingData` | `logit_grad→LogitGradND<half>` | `logit_grad.h` 五方法 | OpDef/CMake/config `ascend950` FP16。 |
| R-002 | 2 | 同上 | 同上 | `→LogitGradND<float>` | 同上 | 950 config FP32。 |
| R-003 | 3 | 同上 | 同上 | `→LogitGradND<bfloat16_t>` | 同上 | 950 config BF16。 |

## Kernel 实现

### 公共执行流程

entry 解析 POD、忽略 workspace，再按 key 实例化 `LogitGradND<T>`。`Init` 绑定 GM、将 `eps<0` 归一为 `[0,1] + false NaN`，否则为 `[eps,1-eps] + false 0`，并申请 TBuf。`Process` 对本核 tile 交替使用两个 UB 半区，实际调用顺序固定为 CopyIn/Cast、四步 quotient、双 compare+And+Select、Cast/CopyOut。无归约、Atomic、通信、NDDMA 或第二 Kernel。

### Route R-001：FP16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Host | `logit_grad_def.cpp` | `LogitGrad` | FP16/ND/950 registered | Tiling。 |
| Tiling | `logit_grad_tiling.cpp` | `RunBigKernelTiling` | dtype FP16 | key 1。 |
| Router | `logit_grad.cpp` | `logit_grad` | key 1 + arch guard | `LogitGradND<half>`。 |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> logit_grad -> LogitGradND<half>::Init/Process
  -> CopyInAndCast -> ComputeStepOne -> ComputeStepTwo -> CastAndCopyOut
```

##### 源码等价伪码

```text
set EVENT0/1 MTE3_MTE2 ready
for each assigned tile:
  half=i&1; count=tail?remain:8192; off=coreStart+i*8192
  wait MTE3_MTE2(event)
  DMA x/dy FP16(count*2B) -> halfBase+16KiB/+48KiB
  MTE2_V; Cast both -> FP32 at +0/+32KiB (CAST_NONE)
  scratch=-x; scratch+=1; scratch*=x; dy/=scratch
  rounded=ceil(count/64)*64; m0=x>=lo; m1=x<=hi; m0&=m1
  x=Select(m0,dy,selectValue,count)
  Cast x FP32->FP16 CAST_NONE; V_MTE3; DMA count*2B -> dx
  signal MTE3_MTE2(event)
wait both events
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `coreStart+i*8192` | count*2 B each | FP16/ND | caller/K-001 | K-001/caller | call | one block/stride 0。 |
| TBuf | 0 | 192 KiB | bytes | InitBuffer | body | per core | two halves。 |
| half x/dy | +0/+32KiB, raw tails +16/+48KiB | 32 KiB each | FP32/FP16 alias | DMA/Cast | Vector/MTE3 | tile | expansion/shrink reuse。 |
| half scratch/masks | +64KiB；mask2 +80KiB | 32 KiB physical | FP32/u8/u16 alias | StepOne/Two | Div/Select | tile | phase reuse。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  h0x["h0 x/raw alias 32KiB"] h0d["h0 dy/raw alias 32KiB"] h0s["h0 scratch/mask 32KiB"]
  h1x["h1 x/raw alias 32KiB"] h1d["h1 dy/raw alias 32KiB"] h1s["h1 scratch/mask 32KiB"]
```

##### 字节账与复用顺序

```text
UB_total=2*(32+32+32)KiB=192KiB <= A5 available UB（Host 未显式检查）
dynamic_ub=N/A
workspace_total=16MiB declared；visible body consumes 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| half | MTE2/Vector | MTE3 | two iterations later | per-half MTE3_MTE2 | CopyOut finished。 |
| scratch→masks | StepOne | Div | StepTwo | PIPE_V | denominator consumed。 |
| x FP32→FP16 | Cast/Vector | compare/Select | output Cast | PIPE_V | Select finished。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| half reuse | EVENT0/1 MTE3_MTE2 | MTE3 | next MTE2 | 96 KiB half | tail unchanged。 |
| input | MTE2_V | DMA | Cast | raw FP16 | count*2 B。 |
| compute | PIPE_V | each vector op | next op | alias/data deps | FP32；compare rounded 64。 |
| output | V_MTE3 | Cast | DMA | FP16 view | CAST_NONE；only count output。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | `RunBigKernelTiling` | key/core/POD/workspace。 |
| Kernel entry/router | `loss/logit_grad/op_kernel/logit_grad.cpp` | `LogitGradND<half>` | key 1。 |
| Device body/helper | `loss/logit_grad/op_kernel/logit_grad.h` | five methods | FP16/FP32/event/tail pipeline。 |

### Route R-002：FP32 原生

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Host | `logit_grad_def.cpp` | `LogitGrad` | FP32/ND/950 | Tiling。 |
| Tiling | `logit_grad_tiling.cpp` | `RunBigKernelTiling` | dtype FP32 | key 2。 |
| Router | `logit_grad.cpp` | `logit_grad` | key 2 | `LogitGradND<float>`。 |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> logit_grad -> LogitGradND<float>::Init/Process
  -> CopyInAndCast(no Cast) -> StepOne -> StepTwo -> CastAndCopyOut(no Cast)
```

##### 源码等价伪码

```text
for tile with same core/event logic:
  DMA x/dy count*4B directly to FP32 +0/+32KiB; MTE2_V
  tmp=-x; tmp+=1; tmp*=x; dy/=tmp
  masks compare rounded-to-64; Select only count
  V_MTE3; DMA FP32 count*4B to dx; signal half ready
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | tile offset | count*4 B | FP32/ND | caller/K-001 | K-001/caller | call | one block。 |
| x/dy per half | +0/+32KiB | 32 KiB each | FP32 | DMA | Vector | tile | no Cast raw view。 |
| scratch/masks | +64/+80KiB | 32 KiB alias | FP32/u8/u16 | Steps | Div/Select | tile | phase reuse。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  h0x["h0 x FP32 32KiB"] h0d["h0 dy FP32 32KiB"] h0s["h0 scratch/mask 32KiB"]
  h1x["h1 x FP32 32KiB"] h1d["h1 dy FP32 32KiB"] h1s["h1 scratch/mask 32KiB"]
```

##### 字节账与复用顺序

```text
UB_total=192KiB; dynamic_ub=N/A; workspace_total=16MiB declared/0 body-consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dy | DMA | Div | next same-half DMA | event/barrier | output dependency complete。 |
| scratch/masks | StepOne | Div | StepTwo | PIPE_V | phase order。 |
| x | DMA | arithmetic/compare | Select in place | PIPE_V | compare complete。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| pipeline | three hard-event directions | DMA/V | next stage | half/data | same as R-001。 |
| numeric | PIPE_V | FP32 ops | next op | deps | no Cast；compare rounded, output count。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | `RunBigKernelTiling` | key 2。 |
| Kernel entry/router | `loss/logit_grad/op_kernel/logit_grad.cpp` | `LogitGradND<float>` | key 2。 |
| Device body/helper | `loss/logit_grad/op_kernel/logit_grad.h` | CopyIn/Out | compile-time dtype branches omit Cast。 |

### Route R-003：BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Host | `logit_grad_def.cpp` | `LogitGrad` | BF16/ND/950 | Tiling。 |
| Tiling | `logit_grad_tiling.cpp` | `RunBigKernelTiling` | dtype BF16 | key 3。 |
| Router | `logit_grad.cpp` | `logit_grad` | key 3 | `LogitGradND<bfloat16_t>`。 |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> logit_grad -> LogitGradND<bfloat16_t>::Init/Process
  -> CopyInAndCast(BF16->FP32) -> StepOne -> StepTwo -> CastAndCopyOut(FP32->BF16)
```

##### 源码等价伪码

```text
DMA BF16 raw tails; Cast CAST_NONE to FP32
q=dy/(x*(1-x)); masks=(x>=lo)&&(x<=hi); Select q/false scalar
Cast FP32->BF16 CAST_RINT; DMA only count*2B
# core/event/tail/offset equal R-001
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | tile offset | count*2 B | BF16/ND | caller/K-001 | K-001/caller | call | one block。 |
| half x/dy | +0/+32KiB, raw +16/+48KiB | 32 KiB each | FP32/BF16 alias | DMA/Cast | Vector/MTE3 | tile | same physical regions。 |
| scratch/masks | +64/+80KiB | 32 KiB alias | FP32/u8/u16 | Steps | Div/Select | tile | phase reuse。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  h0x["h0 x/BF16 alias 32KiB"] h0d["h0 dy/BF16 alias 32KiB"] h0s["h0 scratch/mask 32KiB"]
  h1x["h1 x/BF16 alias 32KiB"] h1d["h1 dy/BF16 alias 32KiB"] h1s["h1 scratch/mask 32KiB"]
```

##### 字节账与复用顺序

```text
UB_total=192KiB; dynamic_ub=N/A; workspace_total=16MiB declared/0 body-consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| BF16 raw tail | MTE2 | input Cast | expanded FP32/output Cast | MTE2_V/PIPE_V | Cast completed。 |
| scratch/masks | StepOne | Div | StepTwo | PIPE_V | denominator consumed。 |
| half | pipeline | MTE3 | later MTE2 | per-half event | CopyOut completed。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| input | MTE2_V/PIPE_V | BF16 DMA/Cast | FP32 ops | raw/expanded alias | CAST_NONE。 |
| output | PIPE_V/V_MTE3 | Select/RINT Cast | DMA | FP32/BF16 alias | CAST_RINT；count tail。 |
| half reuse | MTE3_MTE2 | MTE3 | later MTE2 | 96 KiB | EVENT0/1。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | `RunBigKernelTiling` | key 3。 |
| Kernel entry/router | `loss/logit_grad/op_kernel/logit_grad.cpp` | `LogitGradND<bfloat16_t>` | key 3。 |
| Device body/helper | `loss/logit_grad/op_kernel/logit_grad.h` | CopyIn/Out | BF16 input CAST_NONE/output CAST_RINT。 |

## 算子特定优化

### Trick T-001：EVENT0/1 绑定双半区进行显式流水复用

#### 触发与源码依据

- 适用 Route：全部。
- 触发条件：tile 循环；`pingPongFlag` 交替选择 96 KiB half 与 event。
- Golden 来源：[`loss/logit_grad/op_kernel/logit_grad.h:134-153`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.h#L134-L153)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
pingPongFlag = 0;
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
for (int64_t i = 0; i < loopNum; i++) {
    int64_t localOffset = i * PP_ELEMENT_NUM;

    // 最后一轮的最后一个核处理余数
    if (remain > 0 && i == loopNum - 1 && blockIdx == lastCoreNum) {
        calNum = remain;
    }
    eventId = pingPongFlag ? EVENT_ID1 : EVENT_ID0;
    CopyInAndCast(eachCoreStartOffset + localOffset, calNum);

    ComputeStepOne(calNum);

    ComputeStepTwo(calNum);

    CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);

    pingPongFlag = 1 - pingPongFlag;
}
```

#### 机制与设计原因

event 与半区用同一 flag 选择；CopyOut 发出 `MTE3_MTE2`，两次迭代后同半区 CopyIn 先等待该 flag。配合 CMake `AUTO_SYNC false`，显式同步既防止覆盖未完成输出，也允许另一半区继续流水；物理 UB 固定为 2×96 KiB。

#### 迁移边界

- 适用：无跨 tile 状态的 AIV 流水。
- 前提：950 可提供 192 KiB UB，event/half 一一对应，所有显式 barrier 保留。
- 失效：tile/半区布局或跨 tile 依赖变化时必须重做同步与字节账；无 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从依赖图可见，省略 `MTE3_MTE2` 会失去“输出完成后才覆盖半区”的证明。

### Trick T-002：半精度 raw view、FP32 view 与 mask 分阶段共享同一 TBuf

#### 触发与源码依据

- 适用 Route：R-001、R-003。
- 触发条件：`T` 为 half/BF16。
- Golden 来源：[`loss/logit_grad/op_kernel/logit_grad.h:169-193`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit_grad/op_kernel/logit_grad.h#L169-L193)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
    int64_t elementByte = PP_ELEMENT_NUM * sizeof(T);
    x1Tmp = pingPongFlag ? tmpTensor[elementByte + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                           tmpTensor[elementByte].ReinterpretCast<T>();
    x2Tmp = pingPongFlag ?
                tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float) + MAX_UB_SIZE / 2].ReinterpretCast<T>() :
                tmpTensor[elementByte + PP_ELEMENT_NUM * sizeof(float)].ReinterpretCast<T>();
    DataCopyPad(x1Tmp, inputGm[inputOffset], dataCopyParams, padParams);
    DataCopyPad(x2Tmp, dyGm[inputOffset], dataCopyParams, padParams);

} else {
    DataCopyPad(x1Tensor, inputGm[inputOffset], dataCopyParams, padParams);
    DataCopyPad(x2Tensor, dyGm[inputOffset], dataCopyParams, padParams);
}
```

#### 机制与设计原因

16 KiB raw 半精度输入位于 32 KiB FP32 目标区尾部，Cast 后直接复用同一物理区；32 KiB denominator scratch 被随后 compare masks 覆盖。逻辑 views 不能重复计入 UB，这让两路输入、scratch、masks 都留在每个 96 KiB 半区内。

#### 迁移边界

- 适用：输入窄、计算宽且阶段严格串行的逐元素 AIV 算子。
- 前提：8192 tile、相同 dtype bytes/offset、重叠 Cast 能力与 barriers。
- 失效：FP32 不用 raw tail；改变 tile/dtype/Cast 方向必须重算。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不能把每个 LocalTensor view 当独立分配相加，也不能在 Cast/Div 完成前覆盖 alias 区。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key1 | key2 | key3 | dtype if/else。 |
| 模板与 dtype | half | float | bfloat16_t | router。 |
| 循环与分核 | 8192/tile | 同 | 同 | dtype-independent。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 192KiB/raw alias | 192KiB/direct FP32 | 192KiB/raw alias | 无 TQue/dynamic/通信；寄存器固定仓不可见。 |
| 内存总占用与复用 | raw/FP32/mask | FP32/mask | raw/FP32/mask | 均 192KiB。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DMA+Cast+Vector | DMA+Vector | DMA+Cast+Vector | AIV only。 |
| 同步、精度与 tail | FP32，out CAST_NONE | FP32 | FP32，out CAST_RINT | events/barriers/count。 |
| fallback/失败边界 | none | none | none | unsupported dtype Host fail。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，由无后缀卡覆盖 | 公开支持，由无后缀卡覆盖 | 公开支持；本卡范围。 |
| Host/Launcher | 共用同一源 | 共用同一源 | 同一 OpDef/Infer/Tiling；核数动态取 950 AIV core。 |
| Kernel/资源 | 同一 body | 同一 body | CMake 独立 ascend950 条目；同一 192 KiB body，架构 guard 包含 310。 |

## 限制与失败边界

- 生成 ACLNN bridge/executor/Contiguous/ViewCopy 与其 workspace slice 固定仓不可见。
- API `double eps` 到 float attr/POD 的窄化实现固定仓不可见。
- 文档的 `eps>1/eps==1` 说明与可见 Kernel mask 行为冲突，未做数值验证。
- 固定 192 KiB UB、无资源 selector/fallback；16 MiB workspace 被声明但 Kernel 不读。
- raw tiling buffer 等部分子对象无显式空检查；unknown key/非 220/310 编译值 entry 无 body。
- origin shape 乘积溢出策略固定仓不可见；单次 DMA 窄化由 8192 tile 保证不超过 32768 B。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/logit_grad/docs/aclnnLogitGrad.md` | two ACLNN symbols | API/950/约束。 |
| 入口声明/约束 | `loss/logit_grad/README.md` | LogitGrad | 语义/950。 |
| 图入口 | `loss/logit_grad/op_graph/logit_grad_proto.h` | `REG_OP` | 默认 attr/IR。 |
| OpDef | `loss/logit_grad/op_host/logit_grad_def.cpp` | `LogitGrad` | dtype/AutoContiguous/ascend950。 |
| Infer | `loss/logit_grad/op_host/logit_grad_infershape.cpp` | InferShape/Type | dx shape/dtype。 |
| POD | `loss/logit_grad/op_host/logit_grad_tiling.h` | TilingData | fields。 |
| Tiling | `loss/logit_grad/op_host/logit_grad_tiling.cpp` | RunBigKernelTiling | routes/resources。 |
| Entry | `loss/logit_grad/op_kernel/logit_grad.cpp` | logit_grad | router。 |
| Body | `loss/logit_grad/op_kernel/logit_grad.h` | `LogitGradND<T>` | all device details。 |
| Host build | `loss/logit_grad/op_host/CMakeLists.txt` | add_modules_sources | ACLNN OpDef。 |
| Kernel build | `loss/logit_grad/op_kernel/CMakeLists.txt` | add_kernel_sources | ascend950/AUTO_SYNC。 |
| 950 config | `loss/logit_grad/op_host/config/ascend950/logit_grad_binary.json` | op_list | three dtype binaries。 |
| Build dependency | `cmake/func.cmake` | add_modules_sources/add_kernel_sources | target selection。 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | three field producers/consumers | order/types/units；macro layout不可猜 | TilingData。 |
| 调用顺序 | single Kernel, five methods | key/template/tile order | entry/body。 |
| workspace/内存 | 16MiB declared unused；192KiB UB | offsets/alias/lifetimes | Tiling/Init。 |
| 同步 | half-event mapping | three hard-event directions + PIPE_V | Process/helpers。 |
| 精度与尾块 | FP32 internal；rounded compare/count output | FP16 NONE/BF16 RINT/tail-core formulas | ComputeStepTwo/CopyOut。 |
| 失败边界 | Host dtype failure/empty safe | no unknown-key or low-UB fallback | Tiling/entry/Process。 |
