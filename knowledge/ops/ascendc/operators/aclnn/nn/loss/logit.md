---
type: Operator
title: "Logit A2/A3 ACLNN 算子知识"
description: "说明 aclnnLogit 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/docs/aclnnLogit.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/logit_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/logit_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/logit_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/config/ascend910b/logit_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_host/config/ascend910_93/logit_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:13:26Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# Logit A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnLogitGetWorkspaceSize` + `aclnnLogit`，公开 ACLNN、无 V2 版本族 |
| 计算语义 | `eps >= 0` 时先把非 NaN 的 `x` 截断到 `[eps, 1-eps]`，再算 `ln(z/(1-z))`；`eps < 0` 时直接使用 `x` |
| 输入输出 | ND，0～8 维；FP16、FP32、BF16；输出 shape/dtype/layout 与输入相同，支持空 Tensor 与非连续 Tensor |
| 执行类型 | 单 Kernel：`logit`，`AIV_VECTOR`；无 Cube/AIC、SIMT、NDDMA 或通信节点 |
| Kernel 路线 | R-001 FP16、R-002 FP32、R-003 BF16，分别由 key 1/2/3 选择 |
| 核心资源 | 输入/输出 GM；单个 192 KiB `TBuf<VECCALC>`，物理分成两个 96 KiB ping-pong 半区；声明 16 MiB workspace 但 Kernel 不读写 |
| 来源闭包 | `source_closed`：公开 API/OpDef → Tiling/POD → key router → `LogitND<T>` body → A2/A3 binary/CMake 接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对每个元素，若 `eps >= 0`，源码依次执行上界和下界选择，并保留 NaN：`z = min(max(x, eps), 1-eps)`；否则 `z=x`。随后以 FP32 执行 `t=1-z`、`q=z/t`、`y=ln(q)`。公开约束说明 `x=0/1` 分别产生 `-Inf/+Inf`，区间外产生 NaN；Kernel 的 `x==x` mask 使输入 NaN 最终仍为 NaN。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `input` | ND，0～8 维，可为空、可非连续 | BF16/FP16/FP32 | 概率输入；OpDef 使用 `AutoContiguous()`，生成的连续化实现属于固定仓不可见的框架边界 |
| `eps` | 标量属性 | ACLNN 为 `double`，OpDef/Tiling 为 `float` | 默认/建议值 `-1.0`；传给 `Logit` 属性时窄化为 FP32；非负时启用截断 |
| `out` | 与 `input` 相同 | 与 `input` 相同 | 每个有效元素写一次；空 Tensor 不发生 GM 搬运 |

### 入口与版本

固定仓只公开 `aclnnLogitGetWorkspaceSize`/`aclnnLogit`，未出现更新的公开版本。第一段完成校验并生成 executor/workspace，第二段在调用者 stream 上执行。`logit_def.cpp::Logit` 以 `ACLNNTYPE aclnn` 接入生成式 ACLNN；生成的 C 声明、executor 聚合和 `Contiguous` 实体未落在固定仓，按框架边界处理，算子拥有的 OpDef、Tiling、POD、Kernel 与构建接线均已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

公开接口先检查 `input/out` 非空和输入 dtype；文档要求输出 shape/dtype/format 与输入一致。OpDef 把输入和输出限定为 BF16/FP16/FP32、ND，并对二者启用 `AutoContiguous()`。`TilingLogitTiling` 构造 `LogitTiling`，`RunBigKernelTiling` 再依次检查输入 Tensor、attrs、输入描述非空，读取 FP32 `eps`、dtype 和 origin shape；不支持 dtype 返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

`N=inputShape.GetShapeSize()`；`needCoreNum=max(1,min(platformAivCoreNum,ceil(N/8192)))`。Tiling 固定写 `currentWorkspace[0]=16*1024*1024` 字节，保存 POD 并令 `blockDim=needCoreNum`。公开第一段返回的 executor/workspace 聚合细节固定仓不可见；调用者持有 workspace 与 executor，第二段在指定 stream 上提交。Kernel 把传入 workspace 丢弃为 `userWs=nullptr`，所以该 16 MiB 区域没有算子 Kernel producer/consumer。

### 启动图与失败传播

单节点图：executor 在 stream 上启动 `logit(input, output, workspace, tiling)`；节点 grid 为 `blockDim=needCoreNum`，POD 为 `LogitTilingData`，输入/输出是原 Tensor 的连续 GM 视图，无跨 Kernel 边。Tiling 任一检查失败会阻止启动；router 未命中 key 时函数直接结束且无输出写入。stream 提交、异常传播和生成式 executor 的确切函数名固定仓不可见。

## Route 与参数设计

### 分核与资源计算

逻辑 tile 固定 8192 元素。`totalTimes=ceil(N/8192)`，基础循环数为 `floor(totalTimes/C)`，前 `totalTimes%C` 个核多一轮。最后一个有额外轮次的核处理 `N%8192` 尾块；偏移公式把该尾块插在连续 GM 区间内。每个 block 分配固定 192 KiB UB，不依 dtype/shape 缩小；空域仍启动一个 block，但其循环数为 0。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 升精度 | key 1；`LogitND<half>` | `dataType==DT_FLOAT16` | 继续检查 FP32 |
| 2 | R-002 FP32 原生 | key 2；`LogitND<float>` | 前项未命中且 `dataType==DT_FLOAT` | 继续检查 BF16 |
| 3 | R-003 BF16 升精度 | key 3；`LogitND<bfloat16_t>` | 前两项未命中且 `dataType==DT_BF16`；router 编译 guard 排除 `__NPU_ARCH__==3003` | Host 失败；router unknown key 无写出 |

```text
if dtype == FP16: key=1
else if dtype == FP32: key=2
else if dtype == BF16: key=3
else: GRAPH_FAILED
key 1 -> LogitND<half>; key 2 -> LogitND<float>; key 3 -> LogitND<bfloat16_t>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | FP16、ND、A2/A3 | 单 `logit` | `LogitND<half>::Process` | GM→FP16 staging→FP32；双半区事件复用；FP16 `CAST_NONE` 写回 | `DataCopyPad` 精确字节尾块 |
| R-002 FP32 | 2 | FP32、ND、A2/A3 | 单 `logit` | `LogitND<float>::Process` | GM 直接到 FP32 工作区；无输入/输出 cast | 同上 |
| R-003 BF16 | 3 | BF16、ND、A2/A3 且 key 3 body 被编译 | 单 `logit` | `LogitND<bfloat16_t>::Process` | GM→BF16 staging→FP32；BF16 `CAST_RINT` 写回 | guard 不成立或 unknown key 时无写出 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 每核最大基础域 | `ceil(N/8192)` 决定核数并截断到 AIV 核数 | 单轮 UB 容量固定 8192 元素 | 全部 | 增加每核循环，不切 Route | `LogitTiling::GetNeedCoreNum` |
| 空域 | `N==0` 时 `needCoreNum=1` | 避免 blockDim 为 0；Device `loopNum=0` | 全部 | 单 block 无 GM 访问 | `GetNeedCoreNum`、`Process` |
| Compare 粒度 | `tmpDataCount=ceil(dataCount/64)*64` | FP32 Compare 以 64 元素 repeat 补齐 | 全部且 `eps>=0` | Select 仍只提交 `dataCount` | `ComputeStepOne` |
| BF16 编译 guard | 非 `__NPU_ARCH__==3003` | key 3 实例受条件编译 | R-003 | router 无分支、无写出 | `logit` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `LogitTilingData` | `elementNum` | `int64_t`/64 | 声明顺序第 1；`sizeof/alignof/offsetof` 由仓外 tiling 宏生成，固定仓不可见 | 元素 | `set_elementNum(inputShapeSize)` | `Init`→`Process` 的 tile/尾块计算 | shape 乘积与溢出检查固定仓不可见 |
| `LogitTilingData` | `eps` | `float`/32 | 声明顺序第 2；精确 offset 固定仓不可见 | 无量纲 | `set_eps(*(attrs->GetFloat(0)))` | `Init`；控制截断并生成 `hi=1-eps` | ACLNN double 到 float 的转换边界不可见 |
| `LogitTilingData` | `needCoreNum` | `uint64_t`/64 | 声明顺序第 3；精确 offset 固定仓不可见 | AIV core | `set_needCoreNum(GetNeedCoreNum(...))` | `Process` 分核和循环数 | 至少 1，不超过平台 AIV 核数 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | `logit_tiling.cpp::RunBigKernelTiling` | `LogitTilingData` | `logit.cpp::logit` | `logit.h::LogitND<half>` | `ascend910b/ascend910_93` config + kernel CMake |
| R-002 FP32 | 2 | 同上 | 同上 | 同上 | `LogitND<float>` | 同上 |
| R-003 BF16 | 3 | 同上 | 同上 | 同上 | `LogitND<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD，按 key 实例化 `LogitND<T>`。`Init` 绑定 GM、读取三个字段、取得 block index 并分配 192 KiB TBuf。`Process` 计算本核连续区间，在两个 96 KiB 半区间交替执行 `CopyInAndCast`、可选 clamp、`1-x`/除法/对数、`CastAndCopyOut`。单 Kernel 不使用 workspace，也没有跨节点同步。

### Route R-001：FP16 升精度 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | `dataType==DT_FLOAT16` | key 1、blockDim=`needCoreNum` |
| Router | `loss/logit/op_kernel/logit.cpp` | `logit` | `TILING_KEY_IS(1)` | `LogitND<half>` |

#### Device 数据流

##### 函数级调用链

```text
logit(input, output, unused workspace, LogitTilingData)
  -> LogitND<half>::Init
  -> LogitND<half>::Process
    -> CopyInAndCast -> ComputeStepOne(if eps>=0) -> ComputeStepTwo -> CastAndCopyOut
```

##### 源码等价伪码

```text
C=max(1,min(aivCores,ceil(N/8192))); 按 tile 数给前余核多分一轮
for 本核每个 tile: n=min(8192, 本轮真实剩余)
  wait MTE3->MTE2(当前半区); GM[inputOffset:inputOffset+n] -> FP16 staging
  MTE2->V; cast FP16 -> FP32 x
  if eps>=0: 以 ceil(n/64)*64 生成 NaN/下界/上界 mask，仅对 n 元素 Select
  d=1-x; x=x/d; x=ln(x)
  cast FP32 -> FP16; V->MTE3; UB -> GM[outputOffset:outputOffset+n]
  set MTE3->MTE2(当前半区); 切换 ping/pong
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `coreStart+i*8192` | `dataCount` | FP16 ND | caller/CopyOut | CopyIn/caller | 调用者持有 | `DataCopyPad`, `blockLen=dataCount*2` 字节 |
| ping/pong 半区 | `flag*98304` | 98304 B | raw UB | `TPipe` | 全阶段 | Kernel block | 两半区由 event 交替复用 |
| `x1Tmp`/`x1TensorFp32` | 半区+16384 / 半区+0 | 16384 B / 32768 B | FP16 / FP32 | MTE2/Cast | Cast/Compute | 当前 tile | staging 位于 FP32 目的区上半部，物理重叠 |
| `x2TensorFp32`、masks | +32768、+65536、+81920 | 32768 B；mask 有效至少 `ceil(n/8)` B | FP32/bit mask | Vector | Vector | 当前 tile | mask3 复用 x2 区；mask1/2 独立 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  pingX1["ping x1/FP16 staging · 32 KiB"]:1
  pingX2["ping x2/mask3 · 32 KiB"]:1
  pingMask["ping mask1/2 + spare · 32 KiB"]:1
  pong["pong · 同构 96 KiB"]:3
```

##### 字节账与复用顺序

```text
UB_total = TBuf(MAX_UB_SIZE) = 192*1024 = 196608 B
每半区 = 98304 B；tile = 8192 元素
dynamic_ub = N/A；workspace_total = 16 MiB（Kernel 不消费）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 当前 96 KiB 半区 | MTE2/Vector | Vector/MTE3 | 同 parity 的下个 tile | `MTE3_MTE2` event 0/1 | 覆盖前必须 WaitFlag |
| x2 区 | `Muls/Adds` 或 Compare mask3 | `Div`/`Select` | 同 tile 后续阶段 | `PIPE_V` barrier | 按源码调用顺序复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | `MTE3_MTE2`、`MTE2_V` | 上轮 MTE3/本轮 MTE2 | 新 CopyIn/Vector | 当前半区 | `DataCopyPad` 只搬 `n*2` B |
| Compute | `PipeBarrier<PIPE_V>` | Cast/每个 Vector op | 后继 Vector op | FP32 x/x2/mask | Compare 补齐 64，Select/算术只处理 n |
| CopyOut | `V_MTE3` | FP16 Cast | MTE3 | x1 区 | `CAST_NONE`，只写 n 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit/op_host/logit_tiling.cpp` | `GetNeedCoreNum`/`RunBigKernelTiling` | 8192 元素 tile、key 1、POD/blockDim/workspace |
| Kernel entry/router | `loss/logit/op_kernel/logit.cpp` | `logit`/`LogitND<half>` | key 到模板实例 |
| Device body/helper | `loss/logit/op_kernel/logit.h` | `Process`、四个 helper | 地址、cast、clamp、logit、事件和尾块 |

### Route R-002：FP32 原生 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | `dataType==DT_FLOAT` | key 2 |
| Router | `loss/logit/op_kernel/logit.cpp` | `logit` | `TILING_KEY_IS(2)` | `LogitND<float>` |

#### Device 数据流

##### 函数级调用链

```text
logit -> LogitND<float>::Init -> Process
  -> CopyInAndCast(实际无 Cast) -> 可选 ComputeStepOne -> ComputeStepTwo -> CastAndCopyOut(实际无 Cast)
```

##### 源码等价伪码

```text
分核/循环/事件与 R-001 相同；每轮直接搬 n*4 B 到 x1 FP32 区；
可选 clamp 后依次 Muls(-1)、Adds(1)、Div、Ln；直接把 FP32 x1 搬回 GM。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `coreStart+i*8192` | n | FP32 ND | caller/CopyOut | CopyIn/caller | caller | `blockLen=n*4` B |
| x1/x2 | 半区+0/+32768 | 各 32768 B | FP32 | MTE2/Vector | Vector/MTE3 | 当前 tile | 无 staging；x2 兼作 mask3 |
| mask1/mask2 | 半区+65536/+81920 | 各至少 `ceil(n/8)` B | bit mask | CompareScalar | Select | 当前 tile | 共享同一 96 KiB 半区 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  pingX1["ping x1 · 32 KiB"]:1
  pingX2["ping x2/mask3 · 32 KiB"]:1
  pingMask["ping masks + spare · 32 KiB"]:1
  pong["pong · 同构 96 KiB"]:3
```

##### 字节账与复用顺序

```text
UB_total=196608 B；每半区 98304 B；workspace_total=16 MiB（Kernel 不消费）；dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ping/pong 半区 | MTE2/Vector | Vector/MTE3 | 同 parity 下轮 | event 0/1 | MTE3 完成后才允许 MTE2 |
| x1 | MTE2/Select/Div/Ln | 后继 Vector/MTE3 | 当前 tile 内原地写 | `PIPE_V` barrier | 每步 barrier 保序 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入/搬出 | MTE event | MTE2/V | V/MTE3 | 半区/x1 | FP32 无 cast，精确 n 元素 |
| 算术 | `PIPE_V` barrier | 每个 Vector op | 下一 op | x1/x2/mask | 全程 FP32，除零/负数按 primitive 语义传播 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | key 2 |
| Kernel entry/router | `loss/logit/op_kernel/logit.cpp` | `LogitND<float>` | FP32 实例 |
| Device body/helper | `loss/logit/op_kernel/logit.h` | `CopyInAndCast`/`CastAndCopyOut` | `if constexpr` 跳过两次 cast |

### Route R-003：BF16 升精度 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | `dataType==DT_BF16` | key 3 |
| Router | `loss/logit/op_kernel/logit.cpp` | `logit` | key 3 且编译 guard 有效 | `LogitND<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
logit -> LogitND<bfloat16_t>::Init -> Process
  -> CopyInAndCast(BF16->FP32) -> 可选 clamp -> FP32 logit -> CastAndCopyOut(FP32->BF16)
```

##### 源码等价伪码

```text
与 R-001 共用分核、地址和 event；输入 BF16 staging 后 CAST_NONE 到 FP32；
所有 Select/Muls/Adds/Div/Ln 以 FP32 完成；最终 CAST_RINT 到 BF16 并搬出 n*2 B。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `coreStart+i*8192` | n | BF16 ND | caller/CopyOut | CopyIn/caller | caller | `blockLen=n*2` B |
| staging/FP32/masks | 与 R-001 相同 | 每半区 96 KiB | BF16/FP32/bit | MTE2/Vector | Vector/MTE3 | 当前 tile | staging 与 FP32 目的区物理重叠 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  pingX1["ping FP32 x1/BF16 staging · 32 KiB"]:1
  pingX2["ping FP32 x2/mask3 · 32 KiB"]:1
  pingMask["ping mask1/2 + spare · 32 KiB"]:1
  pong["pong · 同构 96 KiB"]:3
```

##### 字节账与复用顺序

```text
UB_total=196608 B；每半区 98304 B；workspace_total=16 MiB（Kernel 不消费）；dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| staging/FP32 x1 | MTE2/Cast | Cast/Vector/MTE3 | 当前半区下轮 | MTE 与 V event/barrier | 保持源码 offset 与操作顺序 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入 | MTE2→V + barrier | BF16 CopyIn/Cast | FP32 compute | x1 | `CAST_NONE`，n 元素 |
| 输出 | V→MTE3 | FP32 `CAST_RINT` | CopyOut | x1 | round-to-nearest integer mode，n 元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | key 3 |
| Kernel entry/router | `loss/logit/op_kernel/logit.cpp` | `LogitND<bfloat16_t>` | guard 内 BF16 实例 |
| Device body/helper | `loss/logit/op_kernel/logit.h` | cast/compute helpers | FP32 升精度与 `CAST_RINT` 写回 |

## 算子特定优化

### Trick T-001：双半区事件复用覆盖固定 192 KiB UB

#### 触发与源码依据

- 适用 Route：R-001～R-003
- 触发条件：每个非空 tile；`pingPongFlag` 在 0/1 间交替
- Golden 来源：[`loss/logit/op_kernel/logit.h:118-139`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.h#L118-L139)

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

    if (eps >= 0) {
        ComputeStepOne(calNum);
    }
    ComputeStepTwo(calNum);
    CastAndCopyOut(eachCoreStartOffset + localOffset, calNum);
    pingPongFlag = 1 - pingPongFlag;
}
WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
```

#### 机制与设计原因

两套 event 分别保护两个 96 KiB 半区；下一次复用同一半区前等待上一轮 MTE3 完成，避免输出尚未离开 UB 时被 MTE2 覆盖。固定一块 TBuf 内切分避免为每阶段建立独立 Queue，并允许相邻 tile 使用不同半区。

#### 迁移边界

- 适用：单输入单输出、tile 间无依赖的逐元素 Vector 流水。
- 前提：两半区 offset、MTE3→MTE2 event 配对和每 tile 最大 8192 元素不变。
- 失效：加入额外中间量或延长 MTE3 生命周期时必须重算 96 KiB 半区并重建事件图。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP16/BF16 统一升至 FP32，BF16 单独受控舍入

#### 触发与源码依据

- 适用 Route：R-001、R-003
- 触发条件：模板 `T` 为 `half` 或 `bfloat16_t`
- Golden 来源：[`loss/logit/op_kernel/logit.h:168-173`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/logit/op_kernel/logit.h#L168-L173)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
x1TensorFp32 = x1Tensor.template ReinterpretCast<float>();
x2TensorFp32 = x2Tensor.template ReinterpretCast<float>();
if (std::is_same_v<T, bfloat16_t> || std::is_same_v<T, half>) {
    Cast(x1TensorFp32, x1Tmp, RoundMode::CAST_NONE, dataCount);
    PipeBarrier<PIPE_V>();
}
```

#### 机制与设计原因

除法和对数对精度敏感，两个窄 dtype 只在 GM 边界存在，核心算术统一在 FP32；写回时 FP16 使用 `CAST_NONE`，BF16 使用 `CAST_RINT`。模板条件同时消除 FP32 Route 的冗余 cast。

#### 迁移边界

- 适用：需要 FP32 中间精度的逐元素非线性。
- 前提：保留 staging/FP32 物理布局、barrier 和各 dtype 的输出 round mode。
- 失效：直接以窄 dtype 计算或交换 BF16/FP16 round mode 会改变数值路径。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 1，首检 | key 2，次检 | key 3，末检且有 guard | Host dtype 决策树 |
| 模板与 dtype | `LogitND<half>` | `LogitND<float>` | `LogitND<bfloat16_t>` | 同一 body 的编译期分支 |
| 循环与分核 | 8192/tile | 同左 | 同左 | dtype 不改变 tile |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 192 KiB TBuf | 同左 | 同左 | 无 Queue/dynamic UB/通信 |
| 内存总占用与复用 | FP16 staging+FP32 | FP32 直接使用 | BF16 staging+FP32 | 两个 96 KiB 半区 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+Cast | DataCopyPad | DataCopyPad+Cast | 仅 AIV Vector |
| 同步、精度与 tail | FP32 算术，FP16 `CAST_NONE` | 全 FP32 | FP32 算术，BF16 `CAST_RINT` | n 字节尾搬运；Compare 补 64 |
| fallback/失败边界 | Host 非支持 dtype 失败 | 同左 | guard/unknown key 无写出 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 | 支持 | 支持，见独立 `_950` 卡 |
| Host/Launcher | `ascend910b` config；共用 Tiling | `ascend910_93` config；共用 Tiling | `ascend950` config；同一 Tiling |
| Kernel/资源 | 同一 `LogitND<T>`，CMake 目标 `ascend910b` | 同一 body，目标 `ascend910_93` | 同一 body，目标 `ascend950`；外层支持 `__CCE_AICORE__==310` |

## 限制与失败边界

- 生成式 ACLNN 的实际 C 实现、executor 内部、连续化 workspace 聚合与 stream 提交符号固定仓不可见；OpDef 明确了 `AutoContiguous()` 边界。
- tiling 宏生成后的 `sizeof/alignof/offsetof(LogitTilingData)`、AscendC primitive 的寄存器占用和 staging/目的区重叠契约固定仓不可见。
- Host 对 shape 乘积溢出、`GetInputShape`/workspace 指针的额外保护固定仓不可见；源码直接把 shape size 保存为 `int64_t`。
- 未知 key 或被编译 guard 排除的 key 没有显式错误返回，只是不执行 body；Kernel 声明的 workspace 参数不被使用。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/logit/docs/aclnnLogit.md` | `aclnnLogitGetWorkspaceSize`/`aclnnLogit` | 公开接口、平台、dtype/shape 与错误语义 |
| 入口实现 | `loss/logit/op_host/logit_def.cpp` | `ops::Logit`/`OP_ADD` | ACLNN 生成输入、属性、连续化和平台注册 |
| Host/Tiling/Launcher | `loss/logit/op_host/logit_tiling.h` | `LogitTilingData` | POD 字段与顺序 |
| Host/Tiling/Launcher | `loss/logit/op_host/logit_tiling.cpp` | `RunBigKernelTiling` | key、分核、workspace、POD、blockDim |
| Kernel entry/body | `loss/logit/op_kernel/logit.cpp` | `logit` | router 与模板实例 |
| Kernel entry/body | `loss/logit/op_kernel/logit.h` | `LogitND<T>` | 全部 Device 数据流、内存和同步 |
| 构建接线 | `loss/logit/op_host/config/ascend910b/logit_binary.json` | `op_list` | A2 三 dtype binary |
| 构建接线 | `loss/logit/op_host/config/ascend910_93/logit_binary.json` | `op_list` | A3 三 dtype binary |
| 构建接线 | `loss/logit/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 Kernel compute units |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三字段足以复用分核与语义 | 字段顺序、类型、eps 的 FP32 窄化 | `LogitTilingData` |
| 调用顺序 | CopyIn→可选 clamp→logit→CopyOut | clamp 必须在 `1-x` 前，NaN mask 必须保留 | `Process`/`ComputeStepOne` |
| workspace/内存 | 192 KiB TBuf、8192 元素 tile | 两个 96 KiB offset 和 staging/mask alias | `Init`/`CopyInAndCast` |
| 同步 | 两套 MTE event + Vector barrier | 同半区 MTE3 完成前不得覆盖 | `Process`/helpers |
| 精度与尾块 | 窄 dtype 升 FP32，BF16 RINT | Compare 补 64、有效算术与搬运只用 n | `ComputeStepOne`/`CastAndCopyOut` |
| 失败边界 | dtype 在 Host 失败 | unknown key 不写输出；不得假设 fallback | `RunBigKernelTiling`/`logit` |