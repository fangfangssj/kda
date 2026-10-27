---
type: Operator
title: "CrossEntropyLoss A2/A3 ACLNN 算子知识"
description: "说明 aclnnCrossEntropyLoss 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/docs/aclnnCrossEntropyLoss.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/cross_entropy_loss_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/cross_entropy_loss_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/cross_entropy_loss_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/cross_entropy_loss.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/cross_entropy_loss_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/cross_entropy_loss.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/cross_entropy_loss_fp32.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/config/ascend910b/cross_entropy_loss_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/config/ascend910_93/cross_entropy_loss_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:14:13Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# CrossEntropyLoss A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnCrossEntropyLossGetWorkspaceSize` + `aclnnCrossEntropyLoss`，公开两段式 ACLNN；固定仓仅公开原型，inner bridge/executor 是框架生成边界。 |
| 计算语义 | 对每行 logits 做稳定 `log_softmax`，按 class-index target、可选 FP32 weight、`ignoreIndex` 和 label smoothing 产生 loss；同时输出完整 `logProbOut`。公开 zloss 参数/输出当前不生效。 |
| 输入输出 | `input/logProbOut:[N,C]`，`target:[N]`，可选 `weight:[C]`，loss 为 `[N]` 或 `[1]`；A2/A3 实际 Kernel 的 target 仅 INT64。 |
| 执行类型 | 单 Kernel `cross_entropy_loss`，`AIV_VECTOR`；entry 为 `KERNEL_TYPE_MIX_AIV_1_0`，没有 Cube/AIC/MIX_CV 可达节点。 |
| Kernel 路线 | R-001：BF16/FP16 升 FP32 计算后降精度；R-002：FP32 原生计算/输出。 |
| 核心资源 | GM、一个 depth=1 VECIN `TQue`、一个分区 `TBuf`、标量寄存器；归约模式用 user workspace 的 3×64 个 FP32 槽及 `SyncAll<true>()`。 |
| 来源闭包 | `source_closed`：公开 ACLNN→OpDef/框架生成边界→A2/A3 Tiling/POD→dtype key router→基类与 FP32 specialization→binary config/CMake 闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对第 `n` 行令 `m_n=max_c x[n,c]`、`s_n=sum_c exp(x[n,c]-m_n)`、`logProb[n,c]=x[n,c]-m_n-log(s_n)`。若 `target[n]==ignoreIndex`，该样本的 target loss、weight 和 smoothing loss均置零；否则 `w_n=weight[target[n]]`，没有 weight 时为 1。无 smoothing 时 `l_n=-w_n*logProb[n,target[n]]`；有 smoothing 时源码组合为 `(1-labelSmoothing)*l_n-(labelSmoothing/C)*sum_c(weight[c]*logProb[n,c])`，无 weight 时省去逐类 weight。`none` 逐样本输出，`sum` 求和，`mean` 除以有效 `w_n` 总和。`zlossOut/lseForZlossOut` 指针进入 entry 但 body 不绑定、不写入。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `input` | `[N,C]`，ND | FP16/BF16/FP32 | `C>0`；非连续输入由 ACLNN/OpDef `AutoContiguous` 框架处理。 |
| `target` | `[N]`，ND | 公开文档 INT32/INT64；A2/A3 Host 实际仅 INT64 | 值为 `[0,C)` 或等于 `ignoreIndex`；Device 用 `GetValue` 读取。 |
| `weightOptional` | 可选 `[C]`，ND | FP32 | 缺省等价全 1；存在时 shape 必须匹配 C。 |
| `reductionOptional` | Host 字符串 | — | `none/mean/sum`，映射 0/1/2；缺省 `mean`。 |
| `ignoreIndex` | Host scalar | INT64 | 缺省 `-100`。 |
| `labelSmoothing` | Host scalar | double→float | `[0,1]`；Device 以 FP32 计算。 |
| `lseSquareScaleForZloss` / `returnZloss` | Host scalar | double/bool | 公开接口要求传入，但文档明确当前不支持，Tiling/Kernel 均不读取。 |
| `lossOut` | `none:[N]`，否则 `[1]` | 同 input | FP16/BF16 路线最终 `CAST_RINT`；FP32 直接写。 |
| `logProbOut` | `[N,C]`，ND | 同 input | 每行稳定 log-softmax。 |
| `zlossOut` / `lseForZlossOut` | 文档要求非空 | 同 input | 当前不参与计算，固定 body 不写。 |

### 入口与版本

固定 Commit 只有一个公开 `aclnnCrossEntropyLoss` 版本。产品表明确支持 Atlas A2/A3，OpDef 为 `ascend910b` 与 `ascend910_93` 注册 AICore 配置，两份 binary config 各提供 BF16/FP32/FP16 key 1000/1001/1002 实例。公开文档虽然列 target INT32/INT64，但非 RegBase OpDef、`CheckInputDtype` 和 Device `GlobalTensor<int64_t>` 三层证据均只接入 INT64；因此本文按实际实现收窄，不把 950 INT32 支持外推到 A2/A3。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口负责必需指针、dtype、shape、reduction 与 label smoothing 错误；固定仓未包含自动生成的 ACLNN C++ bridge，其参数连续化和 executor 壳按框架边界标注。算子 Host 的 `Tiling4CrossEntropyLoss` 先用 `IsRegbaseSocVersion` 排除 950，随后按 `GetTilingAttr`→`CheckInputDtype`→`CheckInputShape`→`GetTilingData`：dtype key 由 input 产生；reduction 非法或 smoothing 越界失败；input 限 FP16/BF16/FP32、target 限 INT64、weight 限 FP32；target 第 0 维必须等于 N，可选 weight 第 0 维必须等于 C。shape rank 的 `[N,C]/[N]/[C]` 检查位于 inference/ACLNN 框架层。

### 参数、Workspace 与生命周期

`CoresSplitTiling` 取 `GetCoreNumAiv()`：`N<coreNum` 时一行一核；否则余数个 former core 处理 `ceil(N/coreNum)` 行，其余处理 `floor(N/coreNum)` 行。`UbSplitTiling` 读取平台 UB，预留逐核行标量和 4096-byte reduce work 区，再按 dtype 和 smoothing-weight 条件求 `inputUbSize`。Host 始终申请 `16 MiB + 3*64*4 = 16 MiB + 768` bytes；entry 的 `GetUserWorkspace` 跳过系统区，Device 把前三个 64-float 区分别作为 loss、weight、smoothing 的跨核部分和。executor 拥有 workspace，直到第二段在指定 stream 完成。

### 启动图与失败传播

单节点图为 `cross_entropy_loss(input,target,weight,loss,log_prob,zloss,lse_for_zloss,workspace,tiling)`，`blockDim=frontCoreNum+tailCoreNum`，POD=`CrossEntropyLossTilingData`。key 1/2/3 分别进入 BF16/FP16/FP32 body；未知 key 不执行任何 body。单 Kernel 直接写 `logProbOut` 与 loss；没有跨 Kernel 边。`none` 无跨核同步；`mean/sum` 每核先写 user workspace，`SyncAll<true>()` 后仅核 0 归约并写标量 loss。任一 Host gate 失败阻断 launch。

## Route 与参数设计

### 分核与资源计算

令 `F=frontBatchNum`、`U=平台 UB bytes`、`C=targetNum`。公共预留：`L=align32(4F)`，`reserve=3L+32+32+4096`。FP32 路线令 `needWeight=(smoothing>0 && weight存在)`，`inputBytes=floor512((U-reserve)/(1+needWeight))`，再截到 `align32(4C)`，`inputUbSize=inputBytes/4`。16-bit 路线令 `inputBytes=floor512((U-reserve)/(1+1+2+2*needWeight))`，截到 `align32(2C)`，`castTmpBufByte=2*inputBytes`、`probOutBufByte=inputBytes`、`weight4SmoothingBufByte=needWeight?2*inputBytes:0`。C 方向以 `inputUbSize` 切 `ubLoopNum` 个整 tile 加 `ubTailNum`，每 tile 再以 8192 个 FP32 元素切 Vector reduce 主循环和尾段。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 16-bit 升精度 | key 1 `CrossEntropyLoss<bfloat16_t>::Process`；key 2 `CrossEntropyLoss<half>::Process` | input dtype BF16 或 FP16；同 size、POD 和控制流，模板类型决定 cast 语义 | R-002 |
| 2 | R-002 FP32 原生 | key 3 `CrossEntropyLoss<float>::ProcessFp32` | input dtype FP32 | 非支持 dtype 在 Host 失败 |

```text
Tiling4CrossEntropyLoss(non-RegBase)
  -> input BF16: key=1 -> Process<bfloat16_t>
  -> input FP16: key=2 -> Process<half>
  -> input FP32: key=3 -> ProcessFp32
  -> reduction none: direct per-row loss
  -> reduction mean/sum: per-core partial workspace -> SyncAll -> core0 final loss
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 16-bit 升精度 | 1/2 | A2/A3；input BF16/FP16；target INT64；weight absent/FP32 | 单 AIV Kernel | `CrossEntropyLoss<OriT>::Process` + base helpers | GM→OriT queue→FP32 calc；logProb/loss `CAST_RINT`；归约用 workspace+`SyncAll<true>` | `DataCopyPad` 处理非 32-byte 对齐；unknown key 无 body |
| R-002 FP32 原生 | 3 | A2/A3；input FP32；其余同上 | 单 AIV Kernel | `ProcessFp32` 与 FP32 `GetLogProbOut/GetNoneLoss/Calc*` specialization | inputLocal 直接参与 FP32 计算，logProb/loss 不做降精度 cast | 同 R-001 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AIV 分核 | `usedCore=min(N,GetCoreNumAiv())` | 每核处理完整行，避免跨核 C 归约 | 全部 | N=0 产生 blockDim 0，固定仓无单独 empty body | `CoresSplitTiling` |
| UB 512B 下取整 | `floor512((U-reserve)/factor)` | 使 tile 容纳 queue/cast/output/weight 区 | 全部 | 固定仓未显式检查算得 0 的失败分支 | `UbSplitTiling` |
| 8192 元素 reduce 分段 | `len>=8192` | Vector repeat 上限按 128×64 FP32 元素拆分 | 全部 | 小段进入 `Reduce*Small` | `SplitByRepeat/GetReduceSum/GetReduceMax` |
| 全核同步 | `reduction!=none` | 标量 loss 需聚合每核部分和 | 全部归约变体 | `none` 直接逐核写 loss | `GetTilingAttr/GetMeanLoss/GetSumLoss` |
| weight smoothing 缓冲 | `labelSmoothing>0 && weight存在` | smoothing 项需逐 C 搬入 weight | 全部 | 不分配 `weight4SmoothingBuf` | `UbSplitTiling/GetSmoothingLoss` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 的宏定义在固定仓外；字段顺序和 Host/Device 同名读取闭合，但生产头未显式给出 `sizeof/alignof/offsetof`。按字段自然 C 布局可得到 160 bytes，仍应以 CANN 生成 ABI 为准。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CrossEntropyLossTilingData` | `targetNum` | uint64/64 | 推导 offset 0 | 元素 C | input dim1 | 行地址与 C tile | C>0 由公开契约 |
| 同上 | `frontCoreNum` | uint64/64 | 推导 offset 8 | core | former core 数 | `InitGlobalTensor` | 与 tail 相加为 used core |
| 同上 | `frontBatchNum` | uint64/64 | 推导 offset 16 | row/core | `ceil(N/core)` | former core batch | — |
| 同上 | `tailCoreNum` | uint64/64 | 推导 offset 24 | core | tail core 数 | 核分支 | 可为 0 |
| 同上 | `tailBatchNum` | uint64/64 | 推导 offset 32 | row/core | `floor(N/core)` | tail core batch | 可为 0 |
| 同上 | `inputUbSize` | uint64/64 | 推导 offset 40 | input 元素 | `inputBytes/sizeof(input)` | CopyIn tile | 必须可容纳至少一段；固定仓未独立校验 0 |
| 同上 | `castTmpBufByte` | uint64/64 | 推导 offset 48 | byte | FP32 为 0；16-bit 为 `2*inputBytes` | `InitUB` 分区 | FP32 inactive |
| 同上 | `lnTmpBufSize` | uint64/64 | 推导 offset 56 | FP32 元素 | `align32(4F)/4` | per-row target logProb | — |
| 同上 | `weightTmpBufSize` | uint64/64 | 推导 offset 64 | FP32 元素 | 同上 | per-row effective weight | — |
| 同上 | `weight4SmoothingBufSize` | uint64/64 | 推导 offset 72 | FP32 元素 | 条件 `inputBytes` 对应 FP32 容量 | per-C weight tile | inactive 为 0 |
| 同上 | `totalTmpBufByte` | uint64/64 | 推导 offset 80 | byte | reserve+条件区 | `InitBuffer(calcBuf)` | 不含 input queue |
| 同上 | `ubLoopNum` | uint64/64 | 推导 offset 88 | loop | C/inputUbSize 整 tile 数 | 三遍 C 扫描 | — |
| 同上 | `ubTailNum` | uint64/64 | 推导 offset 96 | 元素 | C 余数或 C | tail Copy/Compute | — |
| 同上 | `vecLoopNum` | uint64/64 | 推导 offset 104 | loop | input tile/8192 | 整 tile reduce | — |
| 同上 | `vecTailNum` | uint64/64 | 推导 offset 112 | 元素 | input tile%8192 | 整 tile尾 reduce | — |
| 同上 | `tailVecLoopNum` | uint64/64 | 推导 offset 120 | loop | tail/8192 | 尾 tile reduce | — |
| 同上 | `tailVecTailNum` | uint64/64 | 推导 offset 128 | 元素 | tail%8192 | 最终 mask | — |
| 同上 | `reduction` | uint64/64 | 推导 offset 136 | enum | none/mean/sum→0/1/2 | loss 分支 | 非法 Host 失败 |
| 同上 | `ignoreIndex` | int64/64 | 推导 offset 144 | class index | 属性/default -100 | target 比较 | 可不在 `[0,C)` |
| 同上 | `labelSmoothing` | float/32 | 推导 offset 152 | 比例 | double 属性窄化为 float | loss 组合 | `[0,1]` |
| 同上 | `defaultWeight` | uint32/32 | 推导 offset 156 | bool | optional weight absent→1 | weight分支 | 0/1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1/2 | `cross_entropy_loss_tiling.cpp::Tiling4CrossEntropyLoss` | `cross_entropy_loss_tiling.h::CrossEntropyLossTilingData` | `op_kernel/cross_entropy_loss.cpp::cross_entropy_loss` | `CrossEntropyLoss<OriT>::Process`、`CrossEntropyLossBase` | A2/A3 binary JSON + `op_kernel/CMakeLists.txt` |
| R-002 | key 3 | 同上 | 同上 | 同上 | `ProcessFp32` 及 FP32 specializations | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD 和 user workspace，按 key 构造模板。每核把行区间映射到连续 GM；每行三遍扫描 C：第一遍逐 tile 求 row max，第二遍求 `sum(exp(x-max))`，第三遍重读 logits、减 `max+log(sum)` 并写完整 logProb，同时 gather target logProb、可选计算 smoothing 加权和。随后先乘 target weight，再按 reduction 输出。所有指数、对数、累加、weight 和 smoothing 均为 FP32；16-bit 只在 GM 边界转换。

### Route R-001：BF16/FP16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/cross_entropy_loss_tiling.cpp` | `GetTilingAttr` | BF16→1，FP16→2 | 写 key |
| Router | `op_kernel/cross_entropy_loss.cpp` | `cross_entropy_loss` | `TILING_KEY_IS(1/2)` | `CrossEntropyLoss<bfloat16_t/half>::Process` |

#### Device 数据流

##### 函数级调用链

```text
cross_entropy_loss -> InitTiling -> InitGlobalTensor -> InitUB -> Process
  -> per row: GetRowMax -> CopyIn/Cast -> GetReduceMax
  -> GetExpSum -> CopyIn/Cast -> Adds -> Exp -> GetReduceSum
  -> GetLogProbOut -> CopyIn/Cast -> Adds -> CopyOut(CAST_RINT)
       -> optional GetSmoothingLoss(CopyWeightIn, Mul, ReduceSum)
  -> Mul(target-logProb, effective weight)
  -> none:GetNoneLoss | sum:GetSumLoss | mean:GetMeanLoss
```

##### 源码等价伪码

```text
for row in this core:
  rowMax = max_over_C_fp32(reload logits tiles)
  expSum = sum_over_C_fp32(exp(reload(logits)-rowMax))
  lse = rowMax + log(expSum)
  target = targetGm[globalRow]; ignored = target==ignoreIndex
  w = ignored?0:(defaultWeight?1:weight[target])
  for C tile: logp=fp32(reload(logits))-lse; write cast_rint(logp); accumulate smoothing
  ln[row] = ignored?0:logp[target]; weight[row]=w
ln *= weight
if none: combine smoothing and write row losses
else: write per-core partial(s) -> SyncAll -> core0 reduce -> cast_rint scalar loss
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `coreOffset + row*C + c` | C/row | OriT ND | caller | `CopyIn` | caller | tail `DataCopyPad` |
| `inQueue` | UB queue | `inputUbSize*2` bytes, depth1 | BF16/FP16 | MTE2 | Cast | TPipe/kernel | 每遍/每 tile 重用 |
| `calcBuf.castTmpBuf` | 0 | `castTmpBufByte=2*inputBytes` | FP32 | Cast | vector math | TBuf/kernel | 与 input tile容量对应 |
| `probOutBuf` | `castTmpBufByte` | `inputBytes` | OriT | Cast | MTE3 | TBuf/kernel | 每 tile覆盖 |
| smoothing weight区 | 后续 offset | 条件 `2*inputBytes` | FP32 | CopyWeightIn | Mul | TBuf/kernel | 无 smoothing-weight 时 0 |
| row scalar区 | calcBuf tail | `3*align32(4F)+64` bytes | FP32 | row compute | reduction/output | TBuf/kernel | ln/weight/smooth+2 scalar blocks |
| reduce work区 | calcBuf tail | 4096 bytes | FP32 | reduce helper | reduce helper | TBuf/kernel | 不与活跃标量重叠 |
| user workspace | `0/64/128 + coreIndex` | 192 FP32=768 bytes | FP32 | each core | core0 | executor/stream | `SyncAll<true>` 前不可覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue depth=1 · inputUbSize*2 B"]:1
  cast["castTmpBuf · castTmpBufByte"]:1
  prob["probOutBuf · inputBytes"]:1
  weightC["weight4Smooth · conditional 2*inputBytes"]:1
  rows["ln/weight/smooth · 3*align32(4F)"]:2
  scalars["reduceCalc+reduceRes · 64 B"]:1
  work["reduce work · 4096 B"]:1
```

##### 字节账与复用顺序

```text
queue = inputBytes
calcBuf = 3*align32(4F)+64+4096+2*inputBytes+inputBytes+(needWeight?2*inputBytes:0)
UB_total = queue + calcBuf <= U（由 Host 除数模型与 512B 下取整保证）
dynamic_ub = N/A
workspace_total = 16 MiB system + 768 B user
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input queue/cast | `CopyIn/Cast` | 当前 max/exp/logp 阶段 | 下一 tile/下一遍 | MTE2↔V events + PIPE_V barrier | 当前 Vector 消费完成 |
| probOut | `Cast` | `DataCopyPad` | 下一 tile | V↔MTE3 event | GM 写出完成 |
| workspace 3 slices | each core | core0 final reduce | kernel结束 | `SyncAll<true>` | reduction 非 none |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB→V | MTE2_V/V_MTE2 | DataCopy/Cast | vector/下一搬运 | input/cast区 | tail按有效 `len` |
| logProb UB→GM | V_MTE3/MTE3_V | Cast | DataCopy/下一复用 | probOut | `CAST_RINT` 回 OriT |
| row reduce | PIPE_V + Whole/BlockReduce | vector stages | 后续 scalar | reduce work | 8192/4096/64 分层 tail |
| 跨核 loss | `SyncAll<true>` | each core workspace | core0 | 3 slices | 全 FP32 累加后最终降精度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/cross_entropy_loss_tiling.cpp` | `CoresSplitTiling/UbSplitTiling/VecCalcTiling` | 核、UB、循环和 workspace |
| Kernel entry/router | `op_kernel/cross_entropy_loss.cpp` | key 1/2 | BF16/FP16 实例 |
| Device body/helper | `op_kernel/cross_entropy_loss.h`、`cross_entropy_loss_base.h` | `Process/GetRowMax/GetExpSum/GetLogProbOut/GetReduce*` | 三遍稳定 log-softmax、搬运、同步与 tail |

### Route R-002：FP32 原生

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/cross_entropy_loss_tiling.cpp` | `GetTilingAttr` | input `DT_FLOAT` | key 3 |
| Router | `op_kernel/cross_entropy_loss.cpp` | `TILING_KEY_IS(3)` | key 3 | `ProcessFp32` |

#### Device 数据流

##### 函数级调用链

```text
cross_entropy_loss -> CrossEntropyLoss<float>::ProcessFp32
  -> shared GetRowMax/GetExpSum
  -> specialized GetLogProbOut(CopyOutNoCast)
  -> shared smoothing/reduction
  -> specialized GetNoneLoss/CalcMeanLoss/CalcSumLoss
```

##### 源码等价伪码

```text
同 R-001 三遍算法；inputLocal 自身即 FP32 计算区，不做 input cast；
logProb 通过 CopyOutNoCast 直接写 FP32；none/scalar loss 直接写 FP32。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inQueue` | UB queue | `inputBytes` | FP32 | GM | vector+MTE3 | TPipe/kernel | 同一 LocalTensor 原地 Adds/Exp/LogProb |
| calcBuf | 0 | `reserve+(needWeight?inputBytes:0)` | FP32 | helpers | helpers | TPipe/kernel | cast/prob区为0，公共 row/work区保留 |
| workspace | 同 R-001 | 768 user bytes | FP32 | core partial | core0 | executor | SyncAll |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue depth=1 · inputBytes"]:2
  weightC["weight4Smooth · conditional inputBytes"]:1
  rows["row scalars · 3*align32(4F)+64"]:1
  work["reduce work · 4096 B"]:4
```

##### 字节账与复用顺序

```text
UB_total = inputBytes + reserve + (needWeight?inputBytes:0) <= U
dynamic_ub = N/A
workspace_total = 16 MiB + 768 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inputLocal | CopyIn | max/exp/logProb | 下一 tile/遍 | events、PIPE_V、MTE3_MTE2 in specialization | 当前 GM 输出完成后才能重载 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 同区重载 | MTE3_MTE2 | logProb CopyOut | 下一 CopyIn | inputLocal | 防止输出尚未完成即覆盖 |
| 计算/归约 | 同 R-001 | Vector FP32 | scalar/core0 | calc/workspace | 全程 FP32，无最终 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/cross_entropy_loss_tiling.cpp` | `UbSplitTiling` FP32 branch | 无 cast/prob 临时区 |
| Kernel entry/router | `op_kernel/cross_entropy_loss.cpp` | key 3 | 调用 `ProcessFp32` |
| Device body/helper | `op_kernel/cross_entropy_loss_fp32.h` | FP32 specializations | 原生 FP32 logProb/loss 写回及 MTE3_MTE2 保护 |

## 算子特定优化

### Trick T-001：稳定 log-sum-exp 与 FP32 中间计算

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：所有非空有效行。
- Golden 来源：[`op_kernel/cross_entropy_loss.h:211-215`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_kernel/cross_entropy_loss.h#L211-L215)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
this->CopyIn(inputBuf, offset, uint32_t(this->inputUbSize));
offset += this->inputUbSize;
AscendC::Adds(inputBuf, inputBuf, -rowMax, this->inputUbSize);
PipeBarrier<PIPE_V>();
AscendC::Exp(inputBuf, inputBuf, this->inputUbSize);
PipeBarrier<PIPE_V>();
```

#### 机制与设计原因

先跨整行求 max，再逐 tile 原地减 max 后 Exp，避免直接 `exp(x)` 的上溢；所有输入类型都在 FP32 上做 Exp、Log、归约和 loss 组合。tile 长度来自 POD，tail 只对有效 `len` 发指令。

#### 迁移边界

- 适用：沿长轴实现 softmax/log-softmax 与交叉熵。
- 前提：必须在求 Exp 前获得整行 max，并保证后续所有 tile 使用同一 max。
- 失效：若只能局部求 max，需额外实现跨 tile max/sum 重标定，不能直接套用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从数据依赖可证明，省略整行 max 会改变数值稳定性。

### Trick T-002：按 dtype 消除不必要的 cast/output 缓冲

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：input 为 FP32。
- Golden 来源：[`op_host/cross_entropy_loss_tiling.cpp:170-178`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/cross_entropy_loss/op_host/cross_entropy_loss_tiling.cpp#L170-L178)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (inputDtype == ge::DT_FLOAT) {
    weightBuf = needWeightBuf ? NUM_1 : 0; // labelSmoothing计算需要搬入weight[C]
    inputUbByte = ((maxUbSize - reserveUb) / (DOUBLE_BUFFER + weightBuf)) / BLOCK_512 * BLOCK_512;
    oneBatchByte = CeilDiv(targetNum * NUM_4, BLOCK_32) * BLOCK_32;
    inputUbByte = inputUbByte >= oneBatchByte ? oneBatchByte : inputUbByte;
    castTmpBufByte = 0;
    probOutBufByte = 0;
    weight4SmoothingBufByte = needWeightBuf ? inputUbByte : 0;
    inputUbSize = GetDiv(inputUbByte, NUM_4);
```

#### 机制与设计原因

FP32 直接让 input queue 同时承担计算与输出源，省去 FP16/BF16 所需的 FP32 cast 区和 OriT output 区；Host 将释放出的 UB 给更大的 input tile。FP32 specialization 用 `CopyOutNoCast` 并增加 MTE3→MTE2 依赖，保证同一物理区安全复用。

#### 迁移边界

- 适用：输入、计算与输出 dtype 全相同且支持原地 Vector 变换的路线。
- 前提：输出 DMA 完成前不能重载下一 tile。
- 失效：16-bit 输入必须保留 FP32 计算区与降精度输出区。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若去掉 FP32 路线的 MTE3_MTE2 依赖，同区重载会与尚未完成的 GM 写出冲突。

## 模板与实现对比

| 对比维度 | R-001 16-bit | R-002 FP32 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | key 1/2 | key 3 | input dtype |
| 模板与 dtype | `Process<OriT>` | `ProcessFp32` specialization | 是否需要升/降精度 |
| 循环与分核 | 相同 N/core、C/tile | 相同 | POD 公共 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | queue+cast+prob | queue 原地 | FP32 省 buffer |
| 内存总占用与复用 | 分离输入/FP32/输出 | 输入区原地复用 | dtype size与依赖不同 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+Cast+Vector | DataCopyPad+Vector | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | FP32算后 CAST_RINT | 全 FP32，增加 MTE3_MTE2 | 两者 tail均按有效 len |
| fallback/失败边界 | unknown key无 body | 同左 | Host dtype gate |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 公开支持但见独立 `_950` 卡 |
| Host/Launcher | `ascend910b`，公共非 RegBase Tiling | `ascend910_93`，同 Host/POD | RegBase selector、tuple key与不同 POD |
| Kernel/资源 | 公共 A2/A3 body/binary | 公共 A2/A3 body/binary | empty/full-load/big-C MicroAPI body，target 支持 INT32/INT64 |

## 限制与失败边界

- A2/A3 实际实现的 target 仅 INT64，公开文档中的 INT32 不能据此认为已闭合支持。
- `zlossOut/lseForZlossOut` 当前不写；`lseSquareScaleForZloss/returnZloss` 不生效。
- Host 的 UB 公式没有单独检查 `inputUbSize==0`；极端资源不足行为固定仓不可见。
- 自动 ACLNN bridge、Contiguous 与 executor workspace 汇总实现为框架生成边界；算子拥有的 Tiling、user workspace 和 Kernel 均已闭合。
- 未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/cross_entropy_loss/docs/aclnnCrossEntropyLoss.md` | ACLNN prototypes | 公开语义、平台与限制 |
| 入口实现 | `loss/cross_entropy_loss/op_host/cross_entropy_loss_def.cpp` | `CrossEntropyLoss` OpDef | A2/A3 注册、dtype、AutoContiguous |
| Host/Tiling/Launcher | `loss/cross_entropy_loss/op_host/cross_entropy_loss_tiling.h` | `CrossEntropyLossTilingData` | POD 字段 |
| Host/Tiling/Launcher | `loss/cross_entropy_loss/op_host/cross_entropy_loss_tiling.cpp` | `Tiling4CrossEntropyLoss` | gates、key、核/UB/workspace |
| Kernel entry/body | `loss/cross_entropy_loss/op_kernel/cross_entropy_loss.cpp` | `cross_entropy_loss` | key router |
| Kernel entry/body | `loss/cross_entropy_loss/op_kernel/cross_entropy_loss_base.h` | `CrossEntropyLossBase` | GM/UB、搬运、reduce |
| Kernel entry/body | `loss/cross_entropy_loss/op_kernel/cross_entropy_loss.h` | `CrossEntropyLoss::Process` | 16-bit/common algorithm |
| Kernel entry/body | `loss/cross_entropy_loss/op_kernel/cross_entropy_loss_fp32.h` | FP32 specializations | FP32 route |
| 构建接线 | `loss/cross_entropy_loss/op_host/config/ascend910b/cross_entropy_loss_binary.json` | BF16/FP32/FP16 binaries | A2 binary 接线 |
| 构建接线 | `loss/cross_entropy_loss/op_host/config/ascend910_93/cross_entropy_loss_binary.json` | BF16/FP32/FP16 binaries | A3 binary 接线 |
| 构建接线 | `loss/cross_entropy_loss/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn)` | ACLNN Host模块接线 |
| 构建接线 | `loss/cross_entropy_loss/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 compute units |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 18×uint64 + int64 + float + uint32 字段生产/消费闭合 | 字段顺序、元素/字节单位 | `CrossEntropyLossTilingData/InitTiling` |
| 调用顺序 | max→exp-sum→logProb→loss | 三遍均需相同行 max/lse | `Process` |
| workspace/内存 | 3×64 FP32 user slices | slice offset与 SyncAll 前生命周期 | `GetTilingData/GetLnWeightSum` |
| 同步 | MTE/V events、PIPE_V、SyncAll | FP32 同区复用必须保留 MTE3_MTE2 | `CopyIn/CopyOut/GetMeanLoss` |
| 精度与尾块 | FP32中间、16-bit CAST_RINT、DataCopyPad | 有效 len、8192/4096 reduce split | `GetReduce*/cross_entropy_loss_fp32.h` |
| 失败边界 | Host gate先于 launch | target INT64、未知 key无输出、zloss不写 | `CheckInputDtype/cross_entropy_loss` |