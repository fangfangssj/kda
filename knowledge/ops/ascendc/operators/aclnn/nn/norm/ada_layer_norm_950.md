---
type: Operator
title: "AdaLayerNorm 950 ACLNN 算子知识"
description: "说明 aclnnAdaLayerNormV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/op_api/aclnn_ada_layer_norm_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/op_api/ada_layer_norm_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/ada_layer_norm_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_host/ada_layer_norm_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_kernel/ada_layer_norm_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_full_load.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_welford.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/config/ascend950/ada_layer_norm_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm_v2/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:42:35Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# AdaLayerNorm 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAdaLayerNormV2GetWorkspaceSize` / `aclnnAdaLayerNormV2`，公开 ACLNN V2；950 仅 V2 公开支持，旧 `aclnnAdaLayerNorm` 的支持表明确 950 不支持。 |
| 计算语义 | 沿 x 最后一维 H 做 LayerNorm，融合可选 weight/bias 与 `out=norm*(1+scale)+shift`，返回 out/mean/rstd。 |
| 输入输出 | ND；x/scale/shift/out/mean/rstd 同为 FP32/FP16/BF16；weight/bias 可为 x dtype 或 FP32。 |
| 执行类型 | 单节点 `AIV_VECTOR`，entry 为 `KERNEL_TYPE_AIV_ONLY`；使用 AscendC MicroAPI/高阶 LayerNorm、Welford、Normalize 原语，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 key11 FullLoad同dtype参数；R-002 key12 FullLoad FP32/缺省参数；R-003 key21 Welford同dtype参数；R-004 key22 Welford FP32/缺省参数。 |
| 核心资源 | FullLoad：TQue/TBuf/MicroAPI，`229376+tmpBufferSize` B；Welford：双缓冲队列与复用，`196608+tmpBufferSize` B；节点请求16 MiB workspace但V2 body不读。 |
| 来源闭包 | `source_closed`：公开 V2 ACLNN→L0→950 OpDef/config→公共 Tiling/POD→apt entry→FullLoad/Welford/MicroAPI helper→构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对每个逻辑行 r：`mean_r=Σx/H`，`var_r=Σ(x-mean)^2/H`，`rstd_r=1/sqrt(var+epsilon)`；`norm=(x-mean)*rstd` 后按存在性应用 `weight`、`bias`，再应用按 B 广播的 `scale+1`、`shift`。`mean/rstd` shape 为 x 最后一维替换成 1。950 FullLoad 用 LayerNorm 原语整行/多行归约；Welford 用分片 update/finalize 后逐片 Normalize，数学语义一致。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| x | `[B…,S,H]`，ND，2–8维 | FP32/FP16/BF16 | 必选、非空；B/S/H>0。 |
| scale/shift | `[B…,H]` 或 `[B…,1,H]`，ND | 与 x 相同 | 必选；分别作为自适应缩放与偏移。 |
| weightOptional/biasOptional | `[H]`，ND | x dtype 或 FP32 | 可选；二者同时存在时 dtype 相同。 |
| epsilon | scalar | Host double→Device float | 文档默认1e-5；源码未校验范围。 |
| out | 同 x | 同 x | 必选。 |
| meanOutOptional/rstdOutOptional | `[B…,S,1]`，ND | 同 x | 用户可选；内部算子总是分配并产出。 |

### 入口与版本

V2 README/API 支持表明确 Ascend 950PR/950DT 支持；旧版支持表明确 950 不支持。OpDef 为 ascend950 设置独立 `OpAICoreConfig` 并将 `opFile.value` 指向 `ada_layer_norm_v2_apt`，ascend950 binary config 包含三种公开 x dtype。因此 950 只发布 V2。Quant 等不同输出语义不归入本卡。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段创建 executor，按 `CheckNotNull→CheckDtypeValid→CheckShape` 校验；B/S/H、scale/shift 两种 shape、可选 `[H]` 参数及三个输出 shape/dtype均在此闭合。x/scale/shift 与存在的 weight/bias 经 Contiguous；L0 分配 out/mean/rstd 并添加 `AdaLayerNormV2` AICore 节点；随后 ViewCopy 到用户 out 和存在的 mean/rstd。第二段 `CommonOpExecutorRun` 使用调用者 stream。任何检查、Contiguous、launcher 或 ViewCopy 失败都立即返回，阻断后续节点。

### 参数、Workspace 与生命周期

Tiling 对 V2 节点写 workspace[0]=16 MiB。950 `ada_layer_norm_v2_apt` 接收 workspace 参数但 `GmAddr` 不保存且 `InitV2` 不读取，因此它只是 executor 的节点保留 slice；第一段返回的总 workspace 还可能包含框架 Contiguous/ViewCopy，细分不可见。executor 和框架中间 tensor 的生命周期覆盖第二段执行，所有节点在同一 executor/stream 序列中。

### 启动图与失败传播

```text
aclnnAdaLayerNormV2GetWorkspaceSize
  -> [Contiguous framework nodes]
  -> AdaLayerNormV2 single AIV node
       Host: Tiling4AdaLayerNormV2Tiling(opCode=12)
       entry: ada_layer_norm_v2_apt::ada_layer_norm_v2
       blockDim: needCoreNum; POD: AdaLayerNormTilingData
       GM: x,scale,shift,weight,bias -> out,mean,rstd
  -> ViewCopy(out) -> optional ViewCopy(rstd/mean)
phase2: CommonOpExecutorRun on stream
```

不存在核心 Kernel 间 workspace/GM 边；queue/event 只在单核内部。launcher 添加失败返回空 tuple，ACLNN 返回内部空指针错误，不建立 ViewCopy。

## Route 与参数设计

### 分核与资源计算

`rows=B*S`；`singleCoreNum=rows/coreNumAiv`，`tailNum=rows%coreNumAiv`，`blockDim=singleCoreNum>0?coreNumAiv:tailNum`，前 tailNum 个 core 多一行。950 `isRegBase=true`：

- H≤3072：`rowNum=floor(6144/Hc32)`，其中 `Hc32=ceil(H/32)*32`，FullLoad 多行。
- 3072<H≤6144：rowNum=1，FullLoad 单行。
- H>6144：`batch=ceil(H/5120)`，`sliceSize=ceil(ceil(H/batch)/32)*32`，Welford 分片；`sliceCount=H/sliceSize`、`tailSize=H%sliceSize`。
- FullLoad Host 依次求 `GetLayerNormMaxMinTmpSize` 与 `GetLayerNormNDTilingInfo`；Welford Host 将 Update、Finalize、Normalize 三个 max tmp size 相加写 `tmpBufferSize`。这些外部 API 的内部公式不在固定仓，但 producer/consumer 已闭合。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FullLoad同dtype | 11 / `FullLoad<X,X,X,12>` | 950、H≤6144、任一存在的weight/bias非FP32 | 否则按dtype到R-002 |
| 2 | R-002 FullLoad FP32/缺省 | 12 / `FullLoad<X,float,X,12>` | 950、H≤6144、存在参数均FP32或均缺省 | H>6144到R-004 |
| 3 | R-003 Welford同dtype | 21 / `Welford<X,X,X,12>` | 950、H>6144、任一存在参数非FP32 | 否则R-004 |
| 4 | R-004 Welford FP32/缺省 | 22 / `Welford<X,float,X,12>` | 950、H>6144、存在参数均FP32或均缺省 | 未知key无body |

```text
isRegBase=true
  -> sizeDigit = (H>6144 ? 2 : 1)
  -> dtypeDigit = (opCode==12 && isWeightFloat ? 2 : 1)
  -> key = 10*sizeDigit + dtypeDigit
  -> 11/12 FullLoad; 21/22 Welford
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 11 | H≤6144；X=FP16/BF16；参数同X | 单AIV | `AdaLayerNormFullLoad<X,X,X,12>` | 双缓冲x/out；MicroAPI LayerNorm+Adaption | H mask；row/core tail |
| R-002 | 12 | H≤6144；X任一；参数FP32/缺省 | 单AIV | `AdaLayerNormFullLoad<X,float,X,12>` | 同上，weight/bias FP32直读 | 同上 |
| R-003 | 21 | H>6144；X=FP16/BF16；参数同X | 单AIV | `AdaLayerNormWelford<X,X,X,12>` | x/scale/shift/out双缓冲；Welford+Normalize | slice tail/core tail |
| R-004 | 22 | H>6144；X任一；参数FP32/缺省 | 单AIV | `AdaLayerNormWelford<X,float,X,12>` | 同上，参数FP32直读 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| FullLoad/Welford阈值 | H≤6144 / H>6144 | FullLoad固定6144元素工作域；大H用5120分片Welford | 全部 | 对应另一实现族 | `SplitCore/GetTilingKey` |
| 多行阈值 | H≤3072 | FullLoad可在6144工作域放多行 | R-001/R-002 | rowNum=1 | `SplitCore`; `FastProcess` |
| dtype digit | `isWeightFloat` | U模板决定weight/bias GM读取与cast | 四Route | 同尺寸另一key | `Init/GetTilingKey` |
| MicroAPI tmp | `tmpBufferSize>0` | 外部 LayerNorm/Welford/Normalize primitive临时区 | 全部 | 为0不分配tmpBuf | `DoLayerNormTiling`; `InitV2` |
| queue批上限 | batchCount+rowNum>1024 | mean/rstd queue各1024 float | 全部 | flush并重新Alloc | `FastProcess/SliceProcess` |
| UB上界 | 无显式比较 | FullLoad/Welford固定分配再加tmp | 全部 | 固定仓无fallback | `TilingPrepareTiling/InitV2` |
| 空域 | B,S,H>0 | 避免除零/零block | 全部 | ACLNN 161002 | `CheckShape` |

### POD/ABI

宏实现与 `LayerNormSeparateTiling` 类型来自仓外 `tiling/tiling_api.h`，固定仓只闭合字段顺序、源码类型及 producer/consumer，不能给出可信 `sizeof/alignof/offsetof`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| AdaLayerNormTilingData | batchSize | int64/64 | 第1字段；offset不可见 | batch | B乘积 | 950 body不读 | inactive但已赋值 |
| 同上 | seqLen | int64/64 | 第2字段 | 行/B | x[-2] | row→batch/scaleOffset | >0 |
| 同上 | hiddenDim | int64/64 | 第3字段 | 元素 | x[-1] | 全部循环/地址 | Device FullLoad窄化uint32 |
| 同上 | epsilon | float/32 | 第4字段 | 数值 | attr float | LayerNorm/Normalize | double已窄化 |
| 同上 | aveFactor | float/32 | 第5字段 | 1/元素 | float(1/H) | 950 body不读 | inactive |
| 同上 | hasWeight | int32/32 | 第6字段 | flag | optional非空 | primitive config/CopyIn | 0/1 |
| 同上 | hasBias | int32/32 | 第7字段 | flag | optional非空 | primitive config/CopyIn | 0/1 |
| 同上 | hasSmooth | int32/32 | 第8字段 | flag | V2固定0 | Quant-only路径 | inactive=0 |
| 同上 | singleCoreNum | int64/64 | 第9字段 | 行/core | rows/coreNum | rowStart/end | 可0 |
| 同上 | tailNum | int64/64 | 第10字段 | core数 | rows%coreNum | core tail | <coreNum |
| 同上 | sliceSize | int64/64 | 第11字段 | 元素 | SplitCore | Welford slice；FullLoad不读 | Welford 32对齐 |
| 同上 | rowNum | int64/64 | 第12字段 | 行/批 | 6144/Hc32或1 | FullLoad | ≥1 |
| 同上 | tmpBufferSize | int64/64 | 第13字段 | byte | 三类tmp API max值 | conditional InitBuffer | 未显式UB比较 |
| 同上 | layerNormTiling | LayerNormSeparateTiling | 第14字段；内部布局不可见 | API POD | `GetLayerNormNDTilingInfo` (H≤6144) | FullLoad LayerNorm | Welford不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 11 | `AdaLayerNormTiling` | 公共POD+LayerNorm tiling | apt entry key11 | FullLoad/common/impl | ascend950 OpDef+binary+V2 CMake |
| R-002 | 12 | 同上 | 同上 | apt entry key12 | FullLoad/common/impl | 同上 |
| R-003 | 21 | 同上 | 公共POD+三段tmpSize | apt entry key21 | Welford/common/impl | 同上 |
| R-004 | 22 | 同上 | 同上 | apt entry key22 | Welford/common/impl | 同上 |

## Kernel 实现

### 公共执行流程

apt entry 绑定 5 输入/3 输出，解析 POD，按 key 构造 FullLoad 或 Welford，均调用 `InitV2→Process`。各 core 连续分行。所有输入在需要时 GM→Queue/TBuf，FP16/BF16 提升到 FP32；LayerNorm/Welford/Normalize 与 Adaption 都在 Vector/MicroAPI 上完成；scale 在寄存器中 `+1`，用 `FusedMulDstAdd` 合并 `norm*scalePlusOne+shift`；最后按 X dtype写 out/mean/rstd。单 Kernel 内使用 queue EnQue/DeQue/FreeTensor、PipeBarrier 和必要的 MTE event，不存在跨Kernel边。

### Route R-001：FullLoad 同 dtype参数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `ada_layer_norm_tiling.cpp` | `GetTilingKey` | isRegBase，H≤6144，`isWeightFloat=false` | 11 |
| Router | `ada_layer_norm_v2_apt.cpp` | `ada_layer_norm_v2` | key11，X=half/bfloat16 | `FullLoad<X,X,X,12>` |

#### Device 数据流

##### 函数级调用链

```text
launcher -> apt ada_layer_norm_v2 key11
  -> FullLoad<X,X,X,12>::InitV2 -> Process -> CopyInOtherData -> FastProcess
     -> CopyInScaleShift -> CopyInX -> ProcessLayerNorm(LayerNorm primitive)
     -> Adaption(AdaptionVF -> Double/SingleComputeAdaption)
     -> BaseCopyOut -> CopyMeanRstdOut
```

##### 源码等价伪码

```text
preload optional weight/bias X->fp32
for group <= rowNum:
  for each covered batch: DMA one scale/shift row, UB Copy it repeatNum times
  DMA group x with right padding to Hceil; cast X->fp32 in xQueue storage
  LayerNorm(fp32, optional gamma/beta config, epsilon, tiling)
  for row pairs and tail row, masked registers:
    scale=load+1; out=FusedMulDstAdd(norm,scale,shift); cast/store X
  DMA valid H per row; flush mean/rstd at 1024-row bound or core tail
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xQueue | queue slot | 2*6144*4=49152 B | T/fp32 reinterpret | CopyInX | ProcessLayerNorm | pipe | depth2；非FP32输入放slot后半再cast |
| scale/shift Buf | replicated row offsets | 2*6144*4=49152 B | T | CopyInScaleShift | AdaptionVF | pipe | batch内UB复制 |
| weight/bias Buf | 0 | 2*6144*4=49152 B | fp32 | CopyInOtherData | LayerNorm | pipe | 缺省仍分配但不读 |
| normBuf | row*Hceil | 6144*4=24576 B | fp32 | LayerNorm | Adaption | pipe | group有效域 |
| tmpBuf | 0 | tmpBufferSize | bytes/API私有 | LayerNorm | LayerNorm | pipe | >0才分配 |
| outQueue | queue slot | 2*6144*4=49152 B | X | Adaption | BaseCopyOut | pipe | depth2 |
| mean/rstd Queue | batch idx | 2*1024*4=8192 B | fp32/X reinterpret | LayerNorm | CopyMeanRstdOut | pipe | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["xQueue d2 · 49152 B"]:1
  params["scale/shift/weight/bias · 98304 B"]:1
  normtmp["norm 24576 B + tmpBufferSize"]:1
  outstats["outQueue 49152 B + stats 8192 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 49152 + 4*24576 + 24576 + tmpBufferSize + 49152 + 2*4096
         = 229376 + tmpBufferSize B
dynamic_ub = N/A
workspace_total(node)=16 MiB（V2 body不读）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xQueue slot | CopyInX | LayerNorm | next Alloc | DeQue/FreeTensor | depth2协议 |
| outQueue slot | Adaption | BaseCopyOut | next Alloc | EnQue/DeQue/FreeTensor | producer已EnQue |
| scale/shift replicated rows | CopyInScaleShift | Adaption | batch区间改变 | PIPE events/barrier | repeatNum/rowNum不越6144域 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue流 | EnQue/DeQue/FreeTensor | MTE2/vector | vector/MTE3 | x/out slot | depth2 |
| LayerNorm | PipeBarrier V | primitive | Adaption | norm/mean/rstd | FP32统计与输出暂存 |
| Adaption | UpdateMask + FusedMulDstAdd | norm/scale/shift | store | 寄存器/UB | 每列tail mask；X CAST_RINT |
| DMA | DataCopyPad | GM/UB | queue consumer | H/Hceil | 多行右pad，仅GM写有效H |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp` | `SplitCore/GetTilingKey/DoLayerNormTiling` | key11、rowNum、tmp/API tiling |
| Kernel entry/router | `norm/ada_layer_norm_v2/op_kernel/ada_layer_norm_v2_apt.cpp` | key11 | FullLoad同dtype映射/AIV |
| Device body/helper | `.../arch35/ada_layer_norm_full_load.h` | `FullLoad::FastProcess/ProcessLayerNorm/Adaption` | queue、primitive、输出 |
| Device body/helper | `.../arch35/ada_layer_norm_common.h` | `AdaptionVF/DoubleComputeAdaption` | mask、FMA、cast |

### Route R-002：FullLoad FP32或缺省参数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `ada_layer_norm_tiling.cpp` | `Init/GetTilingKey` | H≤6144且`isWeightFloat=true` | key12 |
| Router | `ada_layer_norm_v2_apt.cpp` | key12 | `DTYPE_X`任一公开dtype | `FullLoad<X,float,X,12>` |

#### Device 数据流

##### 函数级调用链

```text
launcher -> apt entry key12 -> FullLoad<X,float,X,12>::InitV2 -> Process
  -> 与R-001相同；weight/bias按float GM读，缺省则由hasWeight/hasBias跳过
```

##### 源码等价伪码

```text
执行R-001全流程；CopyInOtherData直接DMA FP32参数；LayerNorm按四种hasWeight/hasBias配置选择模板。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| weight/bias GM | h | H each | FP32 | caller | CopyInOtherData | caller | 缺省不读 |
| FullLoad UB | 同R-001 | 229376+tmpBufferSize B | mixed | InitV2 | body | pipe | 与key11相同物理布局 |
| out/stats GM | row*H / row | 逻辑输出 | X | CopyOut | user | executor | 有效tail |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["xQueue d2 · 49152 B"]:1
  params["4 parameter TBuf · 98304 B"]:1
  normtmp["norm 24576 B + tmp"]:1
  outstats["out/stats queues · 57344 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 229376 + tmpBufferSize B
dynamic_ub=N/A; workspace node=16 MiB unused by body
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x/out queue slots | 同R-001 | 同R-001 | next group | queue protocol | depth2 |
| scale/shift | one GM row + UB Copy | Adaption | next batch interval | PIPE barriers | H≤6144，group域≤6144 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 参数 | DataCopyPad/PIPE | FP32 GM | LayerNorm | weight/bias TBuf | 无参数cast |
| 主体 | 同R-001 | queue/primitive | Adaption/CopyOut | x/norm/out | FP32计算，X输出tail mask |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `ada_layer_norm_tiling.cpp` | `GetTilingKey` | key12 |
| Kernel entry/router | `ada_layer_norm_v2_apt.cpp` | `FullLoad<DTYPE_X,float,DTYPE_X,12>` | dtype映射 |
| Device body/helper | `ada_layer_norm_impl.h` | `FullLoad::CopyInOtherData` | FP32/缺省参数 |

### Route R-003：Welford 同 dtype参数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `ada_layer_norm_tiling.cpp` | `GetTilingKey` | H>6144，`isWeightFloat=false` | key21 |
| Router | `ada_layer_norm_v2_apt.cpp` | key21 | X=half/bfloat16 | `Welford<X,X,X,12>` |

#### Device 数据流

##### 函数级调用链

```text
launcher -> apt entry key21 -> Welford<X,X,X,12>::InitV2 -> Process -> SliceProcess
  -> ComputeMeanVar -> WelfordInitialize -> repeated WelfordUpdate -> WelfordFinalize
  -> ComputeAdaLayerNorm per slice -> CopyInX/Other -> ProcessNormalize
     -> CopyInScaleShift -> Adaption(WelfordAdaptionVF) -> BaseCopyOut
  -> CopyMeanRstdOut
```

##### 源码等价伪码

```text
for row:
  allocate meanTmp from scaleQueue and varTmp from shiftQueue; zero sliceSize
  for full slices and tail: DMA x; WelfordUpdate(nRec=1/count)
  WelfordFinalize(rRec=1/H, head/tail lengths) -> mean,var; free temp queue slots
  for slices and tail:
    DMA x and optional X weight/bias; cast to fp32
    Normalize(x,mean,var,epsilon,optional params) -> norm/rstd
    DMA scale/shift; masked registers compute norm*(scale+1)+shift; cast/store X
    DMA valid slice to GM
  batch mean/rstd up to 1024 rows then DMA
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xQueue | slot | 2*5120*4=40960 B | T/fp32 reinterpret | CopyInX/mean scan | Welford/Normalize | pipe | depth2 |
| scaleQueue | slot | 40960 B | fp32/U/T | WelfordInitialize或CopyIn | finalize/Normalize/Adaption | pipe | meanTmp→weight→scale错峰 |
| shiftQueue | slot | 40960 B | fp32/U/T | 同上 | 同上 | pipe | varTmp→bias→shift错峰 |
| varBuf | batch idx | 1024*4=4096 B | fp32 | Finalize | Normalize | pipe | 每行 |
| normBuf | 0 | 5120*4=20480 B | fp32 | Normalize | Adaption | pipe | 每slice覆盖 |
| tmpBuf | 0 | tmpBufferSize | bytes/API私有 | Welford/Normalize | 同 primitive | pipe | conditional |
| outQueue | slot | 2*5120*4=40960 B | X | Adaption | BaseCopyOut | pipe | depth2 |
| mean/rstd Queue | batch | 8192 B | fp32/X reinterpret | Finalize/Normalize | CopyOut | pipe | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["xQueue d2 · 40960 B"]:1
  ssq["scaleQueue+shiftQueue d2 · 81920 B"]:1
  work["var 4096 + norm 20480 + tmp"]:1
  outstats["outQueue 40960 + stats 8192 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*(2*5120*4) + 1024*4 + 5120*4 + tmpBufferSize
         + 2*5120*4 + 2*1024*4
         = 196608 + tmpBufferSize B
dynamic_ub=N/A; workspace node=16 MiB（V2不读）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| scaleQueue slot | WelfordInitialize(meanTmp) | Finalize | weight/scale CopyIn | `FreeTensor(meanTmp)` | finalize完成后 |
| shiftQueue slot | WelfordInitialize(varTmp) | Finalize | bias/shift CopyIn | `FreeTensor(varTmp)` | 同上 |
| normBuf | Normalize | WelfordAdaptionVF | next slice Normalize | PipeBarrier | Adaption完成 |
| outQueue | Adaption | BaseCopyOut | next slice | EnQue/DeQue/FreeTensor | depth2 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Welford | Queue+PipeBarrier | DMA/update | finalize | x/meanTmp/varTmp | `abComputeLength`区分tail；FP32状态 |
| Normalize | Queue+PipeBarrier | x/params/stats | Adaption | norm/rstd | FP32；四种gamma/beta config |
| Adaption | UpdateMask/MicroAPI | norm/scale/shift | queue EnQue | registers/out UB | `dataCount`递减mask；CAST_RINT |
| stats output | queue protocol | mean/rstd FP32 | MTE3 | stats slot | 非FP32先CAST_RINT，batch tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `ada_layer_norm_tiling.cpp` | `SplitCore/DoLayerNormTiling/GetTilingKey` | 5120切片、三tmp、key21 |
| Kernel entry/router | `ada_layer_norm_v2_apt.cpp` | key21 | Welford同dtype实例 |
| Device body/helper | `ada_layer_norm_welford.h` | `SliceProcess/ComputeMeanVar/ComputeAdaLayerNorm` | launch内阶段图、queue/tail |
| Device body/helper | `ada_layer_norm_impl.h` | `WelfordInitialize/Update/Finalize/ProcessNormalize` | primitive参数与复用 |

### Route R-004：Welford FP32或缺省参数

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `ada_layer_norm_tiling.cpp` | `GetTilingKey` | H>6144，`isWeightFloat=true` | key22 |
| Router | `ada_layer_norm_v2_apt.cpp` | key22 | X任一公开dtype | `Welford<X,float,X,12>` |

#### Device 数据流

##### 函数级调用链

```text
launcher -> apt entry key22 -> Welford<X,float,X,12>::InitV2 -> Process
  -> 与R-003相同Welford/Normalize/Adaption图；参数按FP32或缺省处理
```

##### 源码等价伪码

```text
执行R-003；hasWeight/hasBias为真时DMA FP32参数到scale/shift Queue，Normalize直接取float；
缺省时选noGamma/noBeta配置并跳过对应GM搬运。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| weight/bias GM | h | H each when present | FP32 | caller | CopyInOtherData | caller | 缺省不读 |
| Welford UB | 同R-003 | 196608+tmpBufferSize B | queues/fp32 | InitV2 | body | pipe | 参数dtype不改变分配 |
| GM x/scale/shift/out/stats | row/slice公式 | 逻辑张量 | X | caller/body | body/user | caller/executor | slice tail |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xq["xQueue d2 · 40960 B"]:1
  ssq["scale+shift queues · 81920 B"]:1
  work["var+norm+tmp · 24576+tmp B"]:1
  outstats["out+stats · 49152 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=196608+tmpBufferSize B
dynamic_ub=N/A; workspace node=16 MiB unused by V2 body
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| scale/shift queues | mean/var临时→FP32参数→scale/shift | Finalize/Normalize/Adaption | 分阶段 | Queue FreeTensor | 保持源码顺序 |
| norm/out | Normalize/Adaption | Adaption/CopyOut | next slice | PipeBarrier/queue | 当前slice已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 参数 | DataCopyPad+queue | FP32 GM | Normalize | gamma/beta slots | 无参数cast；缺省跳过 |
| 主体 | 同R-003 | Welford/Normalize | Adaption/CopyOut | stats/norm/out | FP32状态、X CAST_RINT、slice tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `ada_layer_norm_tiling.cpp` | `Init/GetTilingKey` | key22 |
| Kernel entry/router | `ada_layer_norm_v2_apt.cpp` | `Welford<DTYPE_X,float,DTYPE_X,12>` | dtype模板 |
| Device body/helper | `ada_layer_norm_impl.h` | `Welford::CopyInOtherData/ProcessNormalize` | FP32/缺省参数与primitive配置 |

## 算子特定优化

### Trick T-001：一份广播参数只做一次 GM DMA，再在 UB 展开多行

#### 触发与源码依据

- 适用 Route：R-001、R-002 FullLoad。
- 触发条件：一个 batch 的多个连续 S 行落入当前 `range`，`repeatNum>1`。
- Golden 来源：[`norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h:169-181`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h#L169-L181)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
DataCopyPad(scaleLocal[localOffset], scaleGm[offset], copyInParams, padParams);
DataCopyPad(shiftLocal[localOffset], shiftGm[offset], copyInParams, padParams);
PIPE_MTE2_V();
for (int i = 1;i < repeatNum;i ++) {
    Copy(scaleLocal[localOffset + i * hiddenDimCeil], scaleLocal[localOffset], hiddenDim);
    Copy(shiftLocal[localOffset + i * hiddenDimCeil], shiftLocal[localOffset], hiddenDim);
}
PipeBarrier<PIPE_V>();
```

#### 机制与设计原因

scale/shift 按 B 广播。源码对每个 batch 只从 GM 读取一行，然后使用 Vector `Copy` 在 UB 复制到该批覆盖的 S 行，避免 `repeatNum` 次 GM DMA；`hiddenDimCeil` 保持每行 UB 间距，实际只复制有效 H。前置 MTE2→V 和后置 V barrier 保证 DMA完成且复制结果在 Adaption 前可见。

#### 迁移边界

- 适用：广播参数在连续行组内相同且 UB 可容纳展开后的行。
- 前提：`rowNum*hiddenDimCeil≤6144`，batch边界的首尾 repeatNum 按源码计算。
- 失效：H>6144 的 Welford 路逐片读取；不能照搬整行UB展开。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：Welford 的 scale/shift 双队列跨阶段复用

#### 触发与源码依据

- 适用 Route：R-003、R-004。
- 触发条件：先完成整行 Welford finalize，再进入逐片 Normalize/Adaption。
- Golden 来源：[`norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h:24-72`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h#L24-L72)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
meanTmpLocal = scaleQueue.AllocTensor<float>();
varTmpLocal = shiftQueue.AllocTensor<float>();
WelfordFinalize<true>(meanLocal[batchCount], varLocal[batchCount], meanTmpLocal, varTmpLocal, tmpLocal, para);
PipeBarrier<PIPE_V>();
scaleQueue.FreeTensor(meanTmpLocal);
shiftQueue.FreeTensor(varTmpLocal);
```

#### 机制与设计原因

统计阶段借用 scaleQueue/shiftQueue slot 保存分片 mean/variance 状态；Finalize 完成后显式释放，同一物理队列才用于逐片 weight/scale 与 bias/shift 输入。错峰复用省去两块各5120 float的独立统计缓冲，同时 queue 生命周期给出可覆盖点。

#### 迁移边界

- 适用：统计阶段和参数阶段严格串行，且队列元素容量/类型重解释兼容。
- 前提：Finalize 与 PipeBarrier 完成后才 Free；后续 CopyIn 必须重新 Alloc/EnQue。
- 失效：若要重叠统计与参数预取，需要独立 buffer 或更深队列，不能继续物理复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FullLoad X参数 | R-002 FullLoad FP32 | R-003 Welford X参数 | R-004 Welford FP32 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | H≤6144，非FP32参数 | H≤6144，FP32/缺省 | H>6144，非FP32参数 | H>6144，FP32/缺省 | H阈值+isWeightFloat |
| 模板与 dtype | FullLoad<X,X,X> | FullLoad<X,float,X> | Welford<X,X,X> | Welford<X,float,X> | GM参数dtype |
| 循环与分核 | rowNum多/单行 | 同 | 5120近似分片 | 同 | UB与归约规模 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | x/out d2+TBuf | 同 | x/scale/shift/out d2 | 同 | 无dynamic UB/通信 |
| 内存总占用与复用 | 229376+tmp | 同 | 196608+tmp；队列跨阶段复用 | 同 | body族 |
| 搬运/Vector/SIMT/NDDMA/通信 API | LayerNorm+FMA MicroAPI | 同 | Welford+Normalize+FMA | 同 | 全部AIV Vector |
| 同步、精度与 tail | queue/barrier，H mask | 同 | queue/barrier，slice tail | 同 | FP32计算，X输出 |
| fallback/失败边界 | 未知key无body | 同 | 同 | 同 | Host正常只产四key |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2公开支持 | V2公开支持 | V2公开支持；旧版明确不支持 |
| Host/Launcher | 非RegBase，key1/2 | 同 | RegBase，key11/12/21/22且生产primitive tmp tiling |
| Kernel/资源 | `AdaLayerNormND`固定184320 B | 同 | arch35 FullLoad/Welford，MicroAPI与动态tmpBufferSize |

## 限制与失败边界

- ACLNN 拒绝空必选指针、非法dtype、x维数不在2–8、B/S/H≤0及shape不匹配；epsilon 无源码范围检查。
- FullLoad 将 `hiddenDim` 从 POD int64 窄化为 uint32；多个 DMA `blockLen` 也窄化为 uint32，Host 未见显式超界检查。
- `tmpBufferSize` 由仓外 AscendC tiling API 产生，其内部公式和 `LayerNormSeparateTiling` ABI不可见；固定仓可以闭合调用参数和消费点，不能展开 primitive 内部循环。
- Host 获取 UB 大小但不比较 `229376+tmpBufferSize` 或 `196608+tmpBufferSize`；固定仓无 UB不足 fallback。
- V2 节点申请16MiB workspace但Device不读；executor总 workspace 的框架节点切片不可见。
- router 未知 key/不匹配编译dtype没有显式错误body；正常 selector 只产11/12/21/22。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/ada_layer_norm/README.md` | 产品支持表 | 旧版明确不支持950，支撑版本族选择 |
| 入口声明/约束 | `norm/ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md` | 支持表/原型 | V2公开950支持与接口语义 |
| 入口实现 | `norm/ada_layer_norm_v2/op_host/op_api/aclnn_ada_layer_norm_v2.cpp` | `CheckParams/AdaLayerNormV2Calculate/aclnn*` | ACLNN校验与executor流程 |
| 入口实现 | `norm/ada_layer_norm_v2/op_host/op_api/ada_layer_norm_v2.cpp` | `l0op::AdaLayerNormV2` | launcher和内部输出 |
| 平台注册 | `norm/ada_layer_norm_v2/op_host/ada_layer_norm_v2_def.cpp` | `config_950` | ascend950 apt opFile配置 |
| Host/Tiling/Launcher | `norm/ada_layer_norm/op_host/ada_layer_norm_tiling.h` | `AdaLayerNormTilingData` | POD字段 |
| Host/Tiling/Launcher | `norm/ada_layer_norm/op_host/ada_layer_norm_tiling.cpp` | `AdaLayerNormTiling` | 四key、分核、tmp、workspace |
| Kernel entry/body | `norm/ada_layer_norm_v2/op_kernel/ada_layer_norm_v2_apt.cpp` | `ada_layer_norm_v2` | AIV router四模板 |
| Kernel entry/body | `norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_common.h` | `AdaptionVF`等 | MicroAPI mask/FMA/cast |
| Kernel entry/body | `norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_full_load.h` | `AdaLayerNormFullLoad` | FullLoad资源/流程 |
| Kernel entry/body | `norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_welford.h` | `AdaLayerNormWelford` | Welford资源/阶段/tail |
| Kernel entry/body | `norm/ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h` | FullLoad/Welford helper | DMA、primitive参数、队列复用 |
| 构建接线 | `norm/ada_layer_norm_v2/op_host/config/ascend950/ada_layer_norm_v2_binary.json` | `AdaLayerNormV2 op_list` | 950三dtype二进制配置 |
| 构建接线 | `norm/ada_layer_norm_v2/op_host/CMakeLists.txt` | `add_modules_sources` | V2模块依赖公共实现 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 公共14字段POD与四key | 字段顺序/位宽、nested tiling只供FullLoad | `AdaLayerNormTilingData/GetTilingKey` |
| 调用顺序 | InitV2→FullLoad或Welford Process | Welford finalize先于queue复用；Normalize先于Adaption | body方法 |
| workspace/内存 | FullLoad/Welford字节公式 | depth2必须计入；tmp为byte；workspace不读 | 各`InitV2` |
| 同步 | Queue协议、PipeBarrier、MTE event | FreeTensor后才覆盖；DMA/vector可见性 | impl/common helpers |
| 精度与尾块 | FP32统计/Normalize，mask/cast回X | Welford head/tail参数、H/Hceil、CAST_RINT | primitive参数/AdaptionVF |
| 失败边界 | ACLNN先校验，Host只产四key | 不可把未知key补成fallback；UB不足无源码路径 | CheckParams/router |
