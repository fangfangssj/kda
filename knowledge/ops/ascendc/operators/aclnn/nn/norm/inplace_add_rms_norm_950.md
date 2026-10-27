---
type: Operator
title: "InplaceAddRmsNorm 950 ACLNN 算子知识"
description: "说明 aclnnInplaceAddRmsNorm 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/inplace_add_rms_norm/docs/aclnnInplaceAddRmsNorm.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/inplace_add_rms_norm/op_host/inplace_add_rms_norm_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/inplace_add_rms_norm/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_host/add_rms_norm_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_host/add_rms_norm_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_host/add_rms_norm_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/inplace_add_rms_norm/op_kernel/inplace_add_rms_norm_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_split_d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/rms_norm/op_kernel/rms_norm_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/rms_norm/op_kernel/arch35/rms_norm_regbase_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/norm_common/op_kernel/reduce_common_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/inplace_add_rms_norm/op_host/config/ascend950/inplace_add_rms_norm_binary.json
    role: implementation
created_at: "2026-08-25T17:56:36Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceAddRmsNorm 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceAddRmsNormGetWorkspaceSize` + `aclnnInplaceAddRmsNorm`，公开 ACLNN 两段式接口，无更新版本 |
| 计算语义 | `x=x1+x2`；每个归一化行 `rstd=1/sqrt(mean(x^2)+epsilon)`；`y=x*rstd*gamma` |
| 输入输出 | ND；`x1/x2/gamma/y/x` 同 dtype（FP16/FP32/BF16），`rstd` FP32；归一化域为 `gamma` 对应的尾部维度 |
| 执行类型 | 单个 `AIV_VECTOR` Kernel；950 entry 显式 `KERNEL_TYPE_AIV_ONLY`，无 Cube/AIC/MIX_CV 可达节点 |
| Kernel 路线 | R-001 行全载 regbase（key 1000）；R-002 D 轴切分 regbase（key 2000） |
| 核心资源 | GM、双缓冲 TQue、FP32 TBuf/RegTensor；Host workspace 固定 `16 MiB + 256 B`，Kernel 未读 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN 声明（生成 bridge 为框架边界）→InplaceAddRmsNorm OpDef/Tiling/POD→AIV-only entry→两套 body/helper→950 config 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=x1Ref+x2Ref`，`D=product(gamma.shape)`，`N=product(x1Ref.shape[0:xRank-gammaRank])`。每行计算 `rstd=1/sqrt(sum(x_i^2)/D+epsilon)`，原地写回 `x1Ref_i=x_i*rstd*gamma_i` 与 `x2Ref_i=x_i`，并输出 FP32 `rstdOut`。950 两条 Route 都产生 y/rstd/x 三路 Device 输出；公开 bridge 把 y/x 分别绑定回两个 mutable ref。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1Ref` | ND，1–8 维 | FP16/FP32/BF16 | 必选输入/输出；不支持空或非连续；返回归一化结果 |
| `x2Ref` | 与 x1Ref 相同 | 与 x1Ref 相同 | 必选输入/输出；返回加法结果 |
| `gamma` | x1Ref 尾部若干维 | 与 x1Ref 相同 | 必选、非空、连续；缩放权重 |
| `epsilon` | 标量 | double→float | 文档建议 `1e-6`；Tiling 要求 `>=0` |
| `rstdOut` | 前导维同 x1Ref、归一化维为 1 | FP32 | 必选、非空、连续 |
| `workspaceSize/executor` | 标量/句柄 | uint64/opaque | 第一段输出 |
| `workspace/stream` | Device 地址/stream | opaque | 第二段输入 |

### 入口与版本

固定仓最新且唯一公开版本是 `aclnnInplaceAddRmsNorm`。文档明确支持 Ascend 950PR/950DT；`InplaceAddRmsNorm` OpDef 的 `ascend950` 配置把 `opFile.value` 指向 `inplace_add_rms_norm_apt`，950 binary config 覆盖 FP16/FP32/BF16。公开 ACLNN bridge 由构建系统生成而无手写源文件，本卡把它标为框架边界；非原地、Cast、Quant 等语义变体不在本卡。

## Host/Launcher 执行流程

### 参数校验与前处理

文档规定必选指针为空返回 `ACLNN_ERR_PARAM_NULLPTR`，不支持 dtype 返回 `ACLNN_ERR_PARAM_INVALID`；四个 tensor 均不支持空或非连续。共享 `Tiling4AddRmsNorm` 检查 rank、x1/x2 shape、gamma 尾维、rstd shape 与 epsilon，再由 `IsRegbaseSocVersion` 转入 950 selector。OpDef 的 `x1,x2,gamma -> x1,rstd,x2` 次序锁定原地别名。生成 bridge 的逐语句实现固定仓不可见，因此不补写 `Contiguous/ViewCopy`。

### 参数、Workspace 与生命周期

Tiling 生成 `numCol=gamma.shape_size`、`numRow=product(x1.shape[0:xRank-gammaRank])`，并设置 `workspace[0]=256+16*1024*1024` B；第一段 executor 返回总大小，调用者保持 workspace/executor 至第二段 stream 完成。950 Inplace entry 仍接收 workspace 参数，但 router 不向两个类的 `Init` 传递，body 也不读，因此该空间不是 Route 间数据边。

### 启动图与失败传播

单 Kernel 图：生成 ACLNN executor → `InplaceAddRmsNorm` Host/Tiling → `inplace_add_rms_norm(x1,x2,gamma,y,rstd,x,workspace,tiling)`。950 entry 显式设 `KERNEL_TYPE_AIV_ONLY`，按 key 1000/2000 解析不同 POD；y/x 分别写回 x1Ref/x2Ref。grid 为 `useCoreNum=ceil(numRow/blockFactor)`，无节点间 workspace。Tiling/launch 失败直接传播；未知 key 无 fallback，不写输出。

## Route 与参数设计

### 分核与资源计算

`ubSize=平台 UB-1024`。`numColAlign=ceil(numCol*sizeof(T)/ubBlockSize)*ubBlockSize/sizeof(T)`。每核最多 `blockFactor=ceil(numRow/numCore)` 行，尾核 `rowWork=numRow-(blockNum-1)*blockFactor`。Host 计算 `binAddQuotient` 为小于 `numColAlign` 的最高 2 次幂（恰为 2 次幂时减半），并据 VReg/UB 算出 `rowFactor`。若全载公式得到 `rowFactor>=1`，取 R-001；否则改成 512 B 对齐列宽，按 `ComputeTotalBufSize` 倍增 `ubFactor` 并进入 R-002。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 行全载 | 1000 / `KernelAddRmsNormRegBase<DTYPE_X1>` | `tmpSize=ubSize-1024-numColAlign*sizeof(T)>0` 且 `numColAlign<=vlfp32^2*4`，并且所得 `rowFactor>=1` | R-002 |
| 2 | R-002 D 轴切分 | 2000 / `KernelAddRmsNormRegBaseSplitD<DTYPE_X1>` | R-001 的 `rowFactor<1` | 无；Tiling 已校验 dtype/epsilon |

```text
IsRegbaseSocVersion(A5)
  -> compute rowFactor for full-row UB load
  -> rowFactor >= 1 ? key 1000 : key 2000
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1000 | 950，FP16/FP32/BF16，完整行满足 Host UB 公式 | 单 AIV node | `KernelAddRmsNormRegBase<DTYPE_X1>::Process` | 行批量双缓冲，MicroAPI 寄存器计算，queue ordering | `curRows` 与 `UpdateMask` 保护尾行/尾列 |
| R-002 | 2000 | 950，FP16/FP32/BF16，R-001 不成立 | 单 AIV node | `KernelAddRmsNormRegBaseSplitD<DTYPE_X1>::Process` | 列 tile 双缓冲，多级归约 TBuf，pipe barrier | `calColNum`、512 B 补零和三级 tail 分支 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 全载 UB | `rowFactor>=1` 的精确分母含四类双缓冲队列、FP32 区、rstd 与 binary-add 区 | R-001 同时驻留多行 x1/x2/y/x 与 FP32 sum | R-001 | R-002 | `TilingAddRmsNormRegbase` |
| 二分归约上限 | `numColAlign<=vlfp32*vlfp32*4` | binary-add scratch 的层级上限 | R-001 | R-002 | `TilingAddRmsNormRegbase` |
| 512 B tile | R-002 `numColAlign=ceil(bytes/512)*512/sizeof(T)` | `ComputeFormerImplV2` 以 512 B 对齐块归约 | R-002 | 无 | `TilingAddRmsNormRegbase`, `ComputeFormerHandle` |
| 非负 epsilon | `epsilon>=0` | 避免非法倒平方根输入参数 | 全部 | `GRAPH_FAILED` | `TilingAddRmsNormRegbase` |
| AIV-only | entry 执行 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` | 禁止 Cube task | 全部 | N/A | `inplace_add_rms_norm` in `inplace_add_rms_norm_apt.cpp` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AddRMSNormRegbaseRFullLoadTilingData` | `numRow` | uint64/64 | 宏生成布局，固定仓未展开 sizeof/offsetof | 行 | `set_numRow(numRow)` | `Init` 分核 | shape 连乘为 uint64 |
| 同上 | `numCol` | uint64/64 | 同上 | 元素/行 | `set_numCol` | 搬运有效列 | gamma size |
| 同上 | `numColAlign` | uint64/64 | 同上 | 元素/行 | UB block 对齐 | UB stride | `>=numCol` |
| 同上 | `blockFactor` | uint64/64 | 同上 | 行/核 | `ceil(numRow/numCore)` | GM base | 尾核另算 |
| 同上 | `rowFactor` | uint64/64 | 同上 | 行/tile | UB 公式并截到 blockFactor | 行循环/分配 | `>=1` |
| 同上 | `binAddQuotient` | uint64/64 | 同上 | 元素 | 最高 2 次幂分界 | square reduce helper | 不大于对齐列 |
| 同上 | `epsilon` | float/32 | 同上 | 标量 | ACLNN double→L0 float→tiling | rstd helper | `>=0` |
| 同上 | `avgFactor` | float/32 | 同上 | 1/元素 | `1.0f/numCol` | rstd helper | numCol 0 时 0 |
| `AddRMSNormRegbaseTilingData` | `numRow` | uint32/32 | 宏生成布局，固定仓未展开 sizeof/offsetof | 行 | `set_numRow` | split Init | 950 split POD 窄化为 32 位 |
| 同上 | `numCol` | uint32/32 | 同上 | 元素/行 | `set_numCol` | 循环/地址 | 同上 |
| 同上 | `numColAlign` | uint32/32 | 同上 | 元素/行 | 512 B 对齐 | 当前 body 不读 | inactive |
| 同上 | `blockFactor` | uint32/32 | 同上 | 行/核 | Host | GM base | `>=1` |
| 同上 | `rowFactor` | uint32/32 | 同上 | 行/tile | 固定 64 | 行循环/rstd queue | 64 |
| 同上 | `ubFactor` | uint32/32 | 同上 | 元素/tile | UB 倍增循环 | 列循环 | 2 的幂 |
| 同上 | `epsilon` | float/32 | 同上 | 标量 | Host | `ComputeRstd` | `>=0` |
| 同上 | `avgFactor` | float/32 | 同上 | 1/元素 | Host | `ComputeRstd` | `1/numCol` |
| 同上 | `ubLoop` | uint32/32 | 同上 | tile 数 | 最大 2 次幂循环数 | master/tail 拆分 | `>=1` |
| 同上 | `colBuferLength` | uint32/32 | 同上 | 元素 | `ubFactor` | queue 容量 | 等于 ubFactor |
| 同上 | `multiNNum` | uint32/32 | 同上 | 行 | 0 | 当前 body 不读 | inactive |
| 同上 | `isNddma` | uint32/32 | 同上 | 布尔 | `numCol<512` 时 1 | 当前 950 split body 不读 | inactive；不据此声称 NDDMA |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 1000 | `add_rms_norm_tiling_arch35.cpp::TilingAddRmsNormRegbase` | `AddRMSNormRegbaseRFullLoadTilingData` | `inplace_add_rms_norm_apt.cpp::inplace_add_rms_norm` | `KernelAddRmsNormRegBase` + `norm_common::CalculateSquareReduceSum/ComputeRstdNewtonRaphson` | InplaceAddRmsNorm OpDef `ascend950` + binary config |
| R-002 | 2000 | 同上 | `AddRMSNormRegbaseTilingData` | 同上 | `KernelAddRmsNormRegBaseSplitD` + AddRmsNorm/rms_norm/norm_common reduce helpers | 同上 |

## Kernel 实现

### 公共执行流程

每个 AIV core 接管连续 `blockFactor` 行，尾核按剩余行缩短。两条路线都按 `CopyIn x1/x2 → FP32 add/square/reduce → rstd → gamma scaling → CopyOut y/rstd/x` 执行；TQue 的 EnQue/DeQue/FreeTensor 与 Vector `PipeBarrier`/MicroAPI `LocalMemBar` 建立 UB 阶段顺序。没有跨 Kernel workspace 或 atomic。

### Route R-001：行全载 regbase

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/add_rms_norm_tiling.cpp` | `Tiling4AddRmsNorm` | `IsRegbaseSocVersion(context)` | 950 selector |
| Tiling | `op_host/add_rms_norm_tiling_arch35.cpp` | `TilingAddRmsNormRegbase` | `rowFactor>=1` | key 1000 |
| Router | `op_kernel/inplace_add_rms_norm_apt.cpp` | `inplace_add_rms_norm` | `TILING_KEY_IS(1000)` | `KernelAddRmsNormRegBase<DTYPE_X1>` |

#### Device 数据流

##### 函数级调用链

```text
inplace_add_rms_norm(key=1000)
  -> KernelAddRmsNormRegBase::Init
  -> Process -> CopyInGamma
    -> Compute -> CopyInXMutiMoveAlign -> CalculateXAdd -> CopyOutX
               -> CalculateSquareReduceSum -> ComputeRstdNewtonRaphson
               -> CopyOutRstd -> CalculateY -> CopyOutY
```

##### 源码等价伪码

```text
for rowTile in ceil(rowWork/rowFactor):
  curRows=min(rowFactor, remaining)
  DMA x1/x2[curRows,numCol] into UB with row stride numColAlign
  masked MicroAPI: fp32X=float(x1)+float(x2); xOut=cast_T(fp32X)
  reduce each fp32X row's squares with binAddQuotient; rstd=NR_rsqrt(sum*avg+epsilon)
  DMA rstd[curRows]
  for rows folded by 2 and VL_FP32 column chunks:
    y=cast_T(fp32X*broadcast(rstd)*float(gamma)), masked at numCol
  DMA xOut/y valid numCol columns only
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x1/x2/y/x | `core*blockFactor*numCol + tile*rowFactor*numCol` | `curRows*numCol` | T, row-major ND | caller/kernel | copy/consumer | caller/executor | UB stride numColAlign |
| x1/x2 queues | 0 | each depth2 × `rowFactor*numColAlign*sizeof(T)` | T | DMA | CalculateXAdd | TPipe/tile | double buffer |
| gamma queue | 0 | `numColAlign*sizeof(T)` | T | DMA once | CalculateY | Process | depth1 |
| y/x queues | 0 | each depth2 × `rowFactor*numColAlign*sizeof(T)` | T | vector | DMA | TPipe/tile | double buffer |
| rstd queue | 0 | depth2 × `align(rowFactor,VL_FP32)*4` | FP32 | rstd helper | DMA/Y | tile | broadcast reuse |
| xFp32 | 0 | `rowFactor*numColAlign*4` | FP32 | CalculateXAdd | reduce/Y | TBuf/tile | retained through Y |
| reduce/binary scratch | 0 | `align(rowFactor,VL)*4 + binaryAddBufLen` | FP32 | reduce helper | reduce helper | TBuf/tile | LocalMemBar governed |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq1["x1Q d2 · 2RF·CA·sT"]:1
  xq2["x2Q d2 · 2RF·CA·sT"]:1
  yq["yQ d2 · 2RF·CA·sT"]:1
  xo["xQ d2 · 2RF·CA·sT"]:1
  gamma["gammaQ d1 · CA·sT"]:1
  rstd["rstdQ d2 · 2·align(RF,VL)·4"]:1
  fp32["xFp32 · RF·CA·4"]:1
  red["reduce + binary-add scratch"]:1
```

##### 字节账与复用顺序

```text
UB_total = 8*RF*CA*sizeof(T) + CA*sizeof(T)
         + 3*align(RF,VL_FP32)*4 + RF*CA*4 + binaryAddBufLen
binaryAddBufLen=align(ceil(binAddQuotient/VL_FP32),BLK_B32)*4*RF
workspace_total=16*1024*1024+256 B (framework/system; body does not read it)
dynamic_ub=N/A
=> Host rowFactor denominator is the conservative admission guard.
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xFp32 | `CalculateXAdd` | reduce + `CalculateY` | next row tile | functions complete before loop continues | y 必须消费后覆盖 |
| rstd queue slot | rstd helper | DMA then `CalculateY` | next tile | DeQue/FreeTensor | CopyOut 不释放，Y 后释放 |
| input queue slots | DMA | CalculateXAdd | next AllocTensor | DeQue/FreeTensor | double-buffer queue protocol |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→vector | TQue EnQue/DeQue | DataCopyPad | CalculateXAdd | x/gamma UB | row padding不作为有效输出 |
| add/reduce | MicroAPI masks + helper barriers | x registers | reduction | FP32 scratch | `UpdateMask` 限制有效列 |
| reduce→Y | queue + function order | rstd helper | CalculateY | rstd/xFp32 | FP32 rstd；FP16/BF16 最终 store cast |
| tail | `curRows`, `reduceNum=numCol` | Host POD | all loops | GM | 尾核/尾行/尾列均缩短 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_rms_norm_tiling_arch35.cpp` | `TilingAddRmsNormRegbase` | 全载 UB 公式、key 1000、POD |
| Kernel entry/router | `op_kernel/inplace_add_rms_norm_apt.cpp` | `inplace_add_rms_norm`, key 1000 | AIV-only 与模板映射 |
| Device body/helper | `op_kernel/arch35/add_rms_norm_regbase.h` | `KernelAddRmsNormRegBase::{Process,CalculateXAdd,CalculateY}` | row folding、mask、DMA 与 UB 生命周期 |
| Device body/helper | `op_kernel/arch35/add_rms_norm_regbase_common.h` | reduction helpers | MicroAPI FP32 归约同步 |
| Device body/helper | `norm/norm_common/op_kernel/reduce_common_regbase.h` | `CalculateSquareReduceSum`, `ComputeRstdNewtonRaphson<true,true>` | 按 VL/2VL/二分支路平方归约；先乘 avg，再 epsilon 与 Newton-Raphson 倒平方根 |

### Route R-002：D 轴切分 regbase

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/add_rms_norm_tiling.cpp` | `Tiling4AddRmsNorm` | regbase SoC | 950 selector |
| Tiling | `op_host/add_rms_norm_tiling_arch35.cpp` | `TilingAddRmsNormRegbase` | full-load `rowFactor<1` | key 2000 |
| Router | `op_kernel/inplace_add_rms_norm_apt.cpp` | `inplace_add_rms_norm` | `TILING_KEY_IS(2000)` | split-D body |

#### Device 数据流

##### 函数级调用链

```text
inplace_add_rms_norm(key=2000)
  -> KernelAddRmsNormRegBaseSplitD::Process
    -> SubProcess
      -> ComputeFormer -> ComputeFormerHandle -> ComputeFormerImplV2
                        -> ComputeSum/ComputeMultiLevelReduce/ComputeMultiLevelRstd
      -> ComputeRstd
      -> ComputeLatter -> CopyInGamma/CopyInX -> ComputeLatterY -> CopyOutY/X
      -> CopyOutRstd
```

##### 源码等价伪码

```text
for rowBatch of up to 64 rows:
  rstd=0
  for each row:
    decompose numCol into masterLoop, paired tailLoop, tail around ubFactor
    load x1/x2 tiles; zero-pad to 32/512 B; fp32 add+square
    fold partial sums through level1/2/3 buffers into row rstd accumulator
  rstd = 1/sqrt(rstd*avgFactor + epsilon)
  for colTile in ceil(numCol/ubFactor):
    load gamma tile
    for row: load x1/x2, fp32 add, y=x*rstd*gamma, materialize x, write valid calColNum
  write calRowNum rstd values
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/gamma/output queues | `row*numCol+tile*ubFactor` | five T queues each depth2 × `ubFactor*sizeof(T)` | T | DMA/vector | reduce/Y/DMA | TPipe/tile | tail padded |
| rstd queue | row batch | depth2 × `64*4` | FP32 | ComputeRstd | Y/DMA | row batch | reused across col tiles |
| xFp32 pair | 0 | conditional 2 × `ubFactor*4` | FP32 | cast/add | ComputeLatterY | tile | absent for FP32 |
| level buffers | 0 | four × `ONCE_VECTOR_SIZE*4` + `V_LENGTH*4` | FP32 | partial reduce | next level | row | barriers in helpers |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  x1["x1Q d2 · 2UF·sT"]:1
  x2["x2Q d2 · 2UF·sT"]:1
  g["gammaQ d2 · 2UF·sT"]:1
  out["yQ+xQ · 4UF·sT"]:1
  rs["rstdQ d2 · 2·64·4"]:1
  cast["cast buf · dtype!=FP32 ? 2UF·4 : 0"]:1
  levels["work+L1+L2+L3 · 4·ONCE·4"]:1
  temp["temp · VL·4"]:1
```

##### 字节账与复用顺序

```text
UB_total = 10*ubFactor*sizeof(T) + 2*64*4
         + (T!=float ? 2*ubFactor*4 : 0)
         + 4*ONCE_VECTOR_SIZE*4 + V_LENGTH*4
workspace_total=16*1024*1024+256 B; dynamic_ub=N/A
=> ubFactor is doubled only while ComputeTotalBufSize(2,T,sizeof(T),2*ubFactor,true)<ubSize.
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| level1/2/3 | partial reduce | multi-level reducer | next row/tile | `ComputeMultiLevelReduce` + barriers | each level consumed before compaction |
| rstdLocal | ComputeRstd | every ComputeLatter + CopyOut | next row batch | CopyOut after all col tiles | must remain live through Y |
| x1 local | DMA/add | `ComputeLatterY` (FP32 direct) | queue free | PipeBarrier + FreeTensor | output computed before free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load padding | DataCopyPad/DataCopyImpl | GM | reduce | input UB | 32/512 B 右侧补零 |
| multi-level reduce | PipeBarrier/LocalMemBar | partial sum | next level | L1/L2/L3 | FP32 sum |
| normalize | `ComputeRstd` | row sums | latter pass | rstdLocal | epsilon/avg FP32 |
| output | queue ordering | ComputeLatterY | DMA | y/x queues | only `calColNum` copied to GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_rms_norm_tiling_arch35.cpp` | `TilingAddRmsNormRegbase` | split POD、UB tile、key 2000 |
| Kernel entry/router | `op_kernel/inplace_add_rms_norm_apt.cpp` | key 2000 branch | dtype-specialized AIV body |
| Device body/helper | `op_kernel/arch35/add_rms_norm_regbase_split_d.h` | `ComputeFormer`, `ComputeLatter` | 三阶段 tail、二遍扫描、输出 |
| Device body/helper | `op_kernel/arch35/add_rms_norm_regbase_common.h` | `ComputeFormerImplV2`, `ComputeLatterY` | masked FP32 add/reduce/scale/cast |
| Device body/helper | `norm/rms_norm/op_kernel/arch35/rms_norm_regbase_common.h` | `ComputeMultiLevelReduce`, `ComputeSum`, `ComputeRstd` | 每 256 个 partial sum 向高层折叠，并执行 `1/sqrt(sum*avg+epsilon)` |
| Device body/helper | `norm/norm_common/op_kernel/reduce_common_regbase.h` | `ComputeMultiLevelRstd` | 末级 partial sum 选择与合并 |

## 算子特定优化

### Trick T-001：全载路线按两行折叠并复用一次 gamma

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：Host 全载 `rowFactor>=1`，Device `curRows>=2` 时进入成对行循环。
- Golden 来源：[`op_kernel/arch35/add_rms_norm_regbase.h:203-219`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h#L203-L219)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint16_t i = 0; i < loopRowsFold; ++i) {
    uint32_t sregCount = reduceNum;
    DataCopy<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * i);
    DataCopy<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (2 * i + 1));
    for (uint16_t r = 0; r < loopCols; ++r) {
        uint32_t offset1 = (2 * i) * numColAlign + r * VL_FP32;
        uint32_t offset2 = (2 * i + 1) * numColAlign + r * VL_FP32;
        MaskReg regCurLoop = UpdateMask<float>(sregCount);
        LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset1);
        LoadRegForDtype<float>(xFp32Tmp, x2Reg, regCurLoop, offset2);
        Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
        Mul(mul1UnrollReg, x2Reg, rstd2Reg, regCurLoop);
        LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
        Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
        Mul(mul2UnrollReg, mul1UnrollReg, gammaReg, regCurLoop);
```

#### 机制与设计原因

同一列块只装载一次 gamma 寄存器，同时对两行广播各自 rstd 并并行完成两组乘法，减少 gamma 的重复寄存器装载；`UpdateMask` 由剩余有效列驱动，保留尾列安全。该结论只说明源码级搬运/调度变化，不声称实测收益。

#### 迁移边界

- 适用：同一权重跨多行的逐行归一化。
- 前提：整行驻留 UB、两行共享 gamma、FP32 中间值尚存活。
- 失效：行不能全载时使用 R-002；奇数尾行走单行循环。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：Split-D 用分级在线归约避免整行 FP32 物化

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：R-001 UB guard 失败；列域按 `ubFactor` 分块。
- Golden 来源：[`op_kernel/arch35/add_rms_norm_regbase_split_d.h:175-216`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_split_d.h#L175-L216)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint32_t repeat = 0; repeat < tailLoop; repeat++) {
    ComputeFormerHandle(tempLocal, offset, 0, ubFactor, ubFactor);
    offset += ubFactor;
    ComputeFormerHandle(tempLocal, offset, 1, ubFactor, ubFactor);
    offset += ubFactor;
    ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
    level1 += 1;
    ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
}
```

#### 机制与设计原因

每两个列 tile 先合并到 `level1`，再在线压缩到更高层，因而只保留固定大小的 work/L1/L2/L3 scratch，而不在 UB 中物化整行 FP32 平方数组。tail 分为 `<=ubFactor/2` 与 `>ubFactor/2`，配合补零保证归约只计有效元素。

#### 迁移边界

- 适用：归一化轴超过整行 UB 容量的 reduce-then-scale 算子。
- 前提：可重读 GM 做第二遍 Y/x 输出，且 FP32 分级和顺序满足精度要求。
- 失效：整行可载时 R-001 更直接；不能重读输入时需额外持久化介质。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 行全载 | R-002 Split-D | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `rowFactor>=1` | 否则 | Host UB admission |
| 模板与 dtype | `KernelAddRmsNormRegBase<DTYPE_X1>`，三 dtype binary | `KernelAddRmsNormRegBaseSplitD<DTYPE_X1>` | 950 binary dtype 专门化 |
| 循环与分核 | core→row tile→两行折叠→VL | core→row batch→row→col tile→第二遍 col tile | 是否能保留整行 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 多行双缓冲 + xFp32 + binary scratch | tile 双缓冲 + 分级 scratch | 无 dynamic UB/通信 |
| 内存总占用与复用 | 与 RF×CA 成正比 | 与 UF 成正比，固定 levels | Host 选择约束 |
| 搬运/Vector/SIMT/NDDMA/通信 API | strided DataCopyPad + MicroAPI | DataCopyImpl + Vector/MicroAPI | 均为 AIV_VECTOR；未使用 SIMT/NDDMA/通信 |
| 同步、精度与 tail | mask、queue、FP32 reduce | padding、barrier、多级 FP32 reduce | 两者最终 cast 到 T |
| fallback/失败边界 | guard 失败→R-002 | unknown key/非法参数失败 | 无第三路线 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；见无后缀卡 | 公开支持；见无后缀卡 | 公开支持，本卡 |
| Host/Launcher | 通用 5-route selector | 同 A2 | regbase selector，只产 1000/2000 |
| Kernel/资源 | 常规 AscendC Vector classes | 同 A2 | AIV-only arch35 MicroAPI、双缓冲与分级归约 |

## 限制与失败边界

- x1Ref/x2Ref/gamma/rstdOut 均必须非空、连续；rank、shape 和 dtype 必须符合公开表，epsilon 必须非负。
- 950 split POD 把 numRow/numCol 写入 uint32 字段；固定仓没有显式溢出检查，超范围不得推断。
- `isNddma` 虽由 Host 写入，但当前 Split-D body 不读取；R-001 的 `MultiCopyParams` 成员也未使用，所以两条可达节点都分类为 AIV_VECTOR，不是 NDDMA。
- 公开 bridge 的 executor/alias 对象布局固定仓不可见；OpDef 次序、原地文档语义及 y/x/rstd Device 写回链路已经闭合。
- unknown key、负 epsilon 或不支持 dtype 没有第三条 Device fallback。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/inplace_add_rms_norm/docs/aclnnInplaceAddRmsNorm.md` | `aclnnInplaceAddRmsNorm*` | 产品、参数、原地语义 |
| 入口实现 | `norm/inplace_add_rms_norm/op_host/inplace_add_rms_norm_def.cpp` | `ops::InplaceAddRmsNorm` | 输入输出次序、dtype、950 apt 映射 |
| 入口实现 | `norm/inplace_add_rms_norm/op_host/CMakeLists.txt` | `ACLNNTYPE aclnn`, shared dependencies | 生成 bridge 与 Host 复用接线 |
| Host/Tiling/Launcher | `norm/add_rms_norm/op_host/add_rms_norm_tiling.cpp` | `Tiling4AddRmsNorm` | Inplace 注册、公共校验/regbase 分流 |
| Host/Tiling/Launcher | `norm/add_rms_norm/op_host/add_rms_norm_tiling_arch35.cpp` | `TilingAddRmsNormRegbase` | 950 key/POD/UB/workspace |
| Host/Tiling/Launcher | `norm/add_rms_norm/op_host/add_rms_norm_tiling.h` | POD 与 Inplace 注册 | 字段顺序/位宽 |
| Kernel entry/body | `norm/inplace_add_rms_norm/op_kernel/inplace_add_rms_norm_apt.cpp` | `inplace_add_rms_norm` | AIV-only router |
| Kernel entry/body | `norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase.h` | `KernelAddRmsNormRegBase` | R-001 |
| Kernel entry/body | `norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_split_d.h` | `KernelAddRmsNormRegBaseSplitD` | R-002 |
| Kernel entry/body | `norm/add_rms_norm/op_kernel/arch35/add_rms_norm_regbase_common.h` | reduction/Y helpers | MicroAPI 依赖 |
| Kernel entry/body | `norm/rms_norm/op_kernel/rms_norm_base.h` | `DataCopyImpl`, constants/traits | DMA、queue 常量 |
| Kernel entry/body | `norm/rms_norm/op_kernel/arch35/rms_norm_regbase_common.h` | `ComputeMultiLevelReduce/ComputeSum/ComputeRstd` | Split-D 分级归约 |
| Kernel entry/body | `norm/norm_common/op_kernel/reduce_common_regbase.h` | square-reduce/rstd helpers | R-001 二分归约/NR 与 R-002 merge |
| 构建接线 | `norm/inplace_add_rms_norm/op_host/config/ascend950/inplace_add_rms_norm_binary.json` | `InplaceAddRmsNorm_fp16/fp32/bf16` | 950 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 两套 key/POD 对应两 body | 字段顺序、64/32 位差异、inactive 字段 | `add_rms_norm_tiling.h` |
| 调用顺序 | add→reduce→rstd→scale，单 AIV node | R-002 必须二遍读，R-001 rstd 在 Y 后释放 | 两个 `Process` |
| workspace/内存 | workspace 非节点间数据边 | queue depth 必须计入；950 body 不读 workspace | Tiling + InitBuffer |
| 同步 | queue/barrier/LocalMemBar | 不得提前覆盖 FP32/rstd/level scratch | body/helpers |
| 精度与尾块 | FP32 add/reduce/rstd，mask/pad tail | 有效 GM 只写 numCol/calColNum | Calculate*/Compute* |
| 失败边界 | full-load 自动落 split | unknown key、负 epsilon、窄化边界不可补推 | Tiling/router |