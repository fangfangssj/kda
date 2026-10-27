---
type: Operator
title: "DequantBias A2/A3 ACLNN 算子知识"
description: "说明 aclnnDequantBias 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/docs/aclnnDequantBias.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/dequant_bias_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/dequant_bias_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/dequant_bias_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/dequant_bias_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_kernel/dequant_bias.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_kernel/dequant_bias_impl.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_kernel/dequant_bias_multi.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/config/ascend910b/dequant_bias_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/config/ascend910_93/dequant_bias_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:22:44Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# DequantBias A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnDequantBiasGetWorkspaceSize` + `aclnnDequantBias`，公开 ACLNN 两段式接口；固定仓未出现更高版本符号。 |
| 计算语义 | 无 bias：`y=x*weight_scale*(activate_scale if present else 1)`；INT32 bias：`y=(x+bias)*weight_scale*(activate_scale if present else 1)`；浮点 bias：`y=x*weight_scale*(activate_scale if present else 1)+bias`。 |
| 输入输出 | `x[M,N]` INT32、`weightScale[N]` FP32/BF16、可选 `activateScale[M]` FP32、可选 `bias[N]` INT32/FP32/FP16/BF16；`out[M,N]` FP16/BF16，均为 ND。 |
| 执行类型 | 单个 `AIV_VECTOR` Kernel；没有可达 Cube/AIC、MIX_CV、SIMT、NDDMA 或通信节点。 |
| Kernel 路线 | R-001～R-005：`N<=8192` 的无 bias/INT32/FP32/FP16/BF16 bias；R-006～R-010：`N>8192` 的对应分列实现。 |
| 核心资源 | GM 输入输出；每核 `pingBuf_`/`pongBuf_`、`wsBuf_`、条件 `biasBuf_`/`asBuf_`；事件 `EVENT_ID6/7`；固定 16 MiB workspace 由 Host 预留但 Kernel 不读取。 |
| 来源闭包 | `source_closed`：公开 ACLNN 契约经 ACLNN 生成框架边界进入 `DequantBias`，算子拥有的 OpDef、校验、Tiling/POD、5 个 key、单 Kernel router、两套完整 Vector body、配置和构建接线均闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对每个 `0<=m<M, 0<=n<N`，先读取 `x[m,n]` 与按列广播的 `weightScale[n]`。当 bias 缺失时，计算 `float(x)*float(weightScale)`；当 bias 为 INT32 时先在 INT32 域做 `x+bias[n]`，再转 FP32 并乘 `weightScale`；当 bias 为 FP32/FP16/BF16 时，先把输入转 FP32并乘权重。若 `activateScale` 存在，再按行乘 `activateScale[m]`；浮点 bias 最后以 FP32 加入。结果以 `RoundMode::CAST_RINT` 转为输出 FP16/BF16，只提交每行前 `N` 个元素。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[M,N]` / ND | INT32 | 必选；Host 要求秩 2。公开约束要求 `M,N` 为正整数且 `M<=25000`。 |
| `weightScale` | `[N]` / ND | FP32、BF16 | 必选；长度等于 `N`。FP32 对应 FP16 输出，BF16 对应 BF16 输出。 |
| `activateScaleOptional` | `[M]` / ND | FP32 | 可选；非空时长度等于 `M`，按行广播。缺失或 shape size 为 0 时不执行该乘法。 |
| `biasOptional` | `[N]` / ND | INT32、FP32、FP16、BF16 | 可选；非空时长度等于 `N`。INT32 在反量化前相加，浮点 bias 在缩放后相加。 |
| `outputDtype` | 标量属性 | int64 | 必须为 1（FP16）或 27（BF16）；输出类型推导也只识别这两个值。 |
| `out` | `[M,N]` / ND | FP16、BF16 | 必选；Host 要求 shape 与 `x` 两维完全相同。 |

### 入口与版本

公开文档声明两段式 `aclnnDequantBiasGetWorkspaceSize`/`aclnnDequantBias`。固定仓中没有手写 ACLNN inner 实现；`op_host/CMakeLists.txt` 以 `ACLNNTYPE aclnn` 接入生成框架，因此参数封装、executor 壳和实际 Kernel launch 视为明确的框架边界，不推测其内部符号。仓内只存在这一公开版本族。README 和 ACLNN 文档均明确支持 Atlas A2/A3，并明确不支持 Ascend 950PR/950DT；A2/A3 分别有 `ascend910b`、`ascend910_93` binary config，二者配置集合一致。

## Host/Launcher 执行流程

### 参数校验与前处理

ACLNN 第一段检查必选指针、公开 dtype/shape 约束并建立 executor。进入算子 Tiling 后，`TilingBaseClass::DoTiling` 严格依次调用 `GetShapeAttrsInfo -> GetPlatformInfo -> IsCapable -> DoOpTiling -> DoLibApiTiling -> GetWorkspaceSize -> PostTiling -> GetTilingKey`；任一步非成功即返回并阻断 launch。`GetShapeAttrsInfo` 检查输出属性、各输入秩、广播维、输出 shape、可选输入 dtype，随后将 `M/N/nAlign/asExist` 写入 POD。非连续 Tensor 在 ACLNN 文档中标为不支持，因此本算子没有可见的 `Contiguous/ViewCopy` 前处理。

### 参数、Workspace 与生命周期

平台对象产生 `aivNum=GetCoreNumAiv()` 与 `ubSize=GetCoreMemSize(UB)`。`nAlign=ceil(N*4/64)*16` 个 FP32/INT32 元素。Host 预留 `16*1024*1024` 字节 workspace，并把 `blockDim` 设为平台全部 AIV 核数；Device 通过 `needCoreNum` 让多余 block 立即返回。该 workspace 由 ACLNN 调用者在第一段取得大小、在第二段传入并由 executor 生命周期管理；Kernel entry 虽接收 `workspace`，body 不读取它。

### 启动图与失败传播

```text
aclnnDequantBiasGetWorkspaceSize(...)
  -> ACLNN 生成 executor（固定仓框架边界）
  -> DequantBias/TilingForDequantBias
     -> 校验、POD、TilingKey、blockDim=aivNum、workspace=16 MiB
aclnnDequantBias(workspace, workspaceSize, executor, stream)
  -> 单节点 K-001 dequant_bias<<<aivNum, stream>>>
     -> key/N router -> DequantBiasImpl 或 DequantBiasMultiImpl
     -> GM y
```

K-001 的输入为四个输入 GM 地址、输出 GM 地址和 `DequantBiasTilingData`；无 Kernel 间 workspace 边。Tiling 失败会阻断 launch；executor/stream 的框架级错误传播细节固定仓不可见。Host 只产生 10100/10110/10111/10112/10113；Device 对未知 key 没有 `else` 报错而直接不写输出，因此复用时不得绕过 Host key 生成。

## Route 与参数设计

### 分核与资源计算

令 `A=aivNum`。Host 计算 `perCoreRow=ceil(M/A)`、`needCoreNum=ceil(M/perCoreRow)`、`tailCoreRow=M-(needCoreNum-1)*perCoreRow`。每核第 `b` 个 block 从 `b*perCoreRow` 行开始，尾核使用 `tailCoreRow`，其余核使用 `perCoreRow`；launch 仍发出 `A` 个 block。

`baseIn=2*nAlign*4` 字节（ping+pong），`ws=nAlign*4`，`as=asExist?align32(perCoreRow*4):0`，`bias=biasExist?ws:0`。当 `N>8192` 时把 `baseIn` 固定为 `2*8192*4=65536`、`ws=32768`、`bias=32768`。要求 `baseIn+ws+as+bias < UB` 且 `M<=25000`。随后 `rowsPerLoop=min((UB-ws-as-bias)/baseIn, coreRows)`，loop 与 tail 由向上取整产生；`inBufferSize=baseIn*rowsPerLoop`。大 N Device body 固定一行一循环，因此其可复用前提是当前支持平台计算出的 `rowsPerLoop==1`；源码未显式 assert 该条件。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 小 N 无 bias | 10100 / `DequantBiasImpl<...,false>` | `biasExist==0 && N<=8192`；`asExist` 不改变 key | R-006（大 N） |
| 2 | R-002 小 N INT32 bias | 10110 / `DequantBiasImpl<...,int32_t,...,true>` | `biasExist==1 && biasDtype==INT32 && N<=8192` | R-007 |
| 3 | R-003 小 N FP32 bias | 10111 / `...<...,float,...,true>` | `biasExist==1 && biasDtype==FP32 && N<=8192` | R-008 |
| 4 | R-004 小 N FP16 bias | 10112 / `...<...,half,...,true>` | `biasExist==1 && biasDtype==FP16 && N<=8192` | R-009 |
| 5 | R-005 小 N BF16 bias | 10113 / `...<...,bfloat16_t,...,true>` | `biasExist==1 && biasDtype==BF16 && N<=8192` | R-010 |
| 6 | R-006 大 N 无 bias | 10100 / `DequantBiasMultiImpl<...,false>` | `biasExist==0 && N>8192` | 不支持/校验失败 |
| 7 | R-007 大 N INT32 bias | 10110 / `...Multi<...,int32_t,...,true>` | `biasExist==1 && biasDtype==INT32 && N>8192` | 不支持/校验失败 |
| 8 | R-008 大 N FP32 bias | 10111 / `...Multi<...,float,...,true>` | `biasExist==1 && biasDtype==FP32 && N>8192` | 不支持/校验失败 |
| 9 | R-009 大 N FP16 bias | 10112 / `...Multi<...,half,...,true>` | `biasExist==1 && biasDtype==FP16 && N>8192` | 不支持/校验失败 |
| 10 | R-010 大 N BF16 bias | 10113 / `...Multi<...,bfloat16_t,...,true>` | `biasExist==1 && biasDtype==BF16 && N>8192` | 不支持/校验失败 |

```text
base key = 10000 + 100
if bias absent: key=10100
else if bias INT32: key=10110
else if bias FP32: key=10111
else if bias FP16: key=10112
else if bias BF16: key=10113
else: Host dtype validation fails
router then chooses DequantBiasImpl when N<=8192, otherwise DequantBiasMultiImpl
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 10100 | A2/A3，`N<=8192`，bias 空 | 单 K-001 AIV | `DequantBiasImpl<X,W,B,Y,false>`；X=INT32，`(W,Y)=(FP32,FP16)/(BF16,BF16)` | ping/pong+ws+条件 as；MTE2/V/MTE3 事件 | `nAlign` 计算，CopyOut 仅 N |
| R-002 | 10110 | 同上，bias INT32 | 单 K-001 | `DequantBiasImpl<X,W,int32_t,Y,true>` | 加 biasBuf；先 INT32 Add | 同上 |
| R-003 | 10111 | 同上，bias FP32 | 单 K-001 | `DequantBiasImpl<X,W,float,Y,true>` | biasBuf 直接装 FP32；缩放后 Add | 同上 |
| R-004 | 10112 | 同上，bias FP16 | 单 K-001 | `DequantBiasImpl<X,W,half,Y,true>` | biasBuf 后半暂存 FP16、前半 FP32 | 同上 |
| R-005 | 10113 | 同上，bias BF16 | 单 K-001 | `DequantBiasImpl<X,W,bfloat16_t,Y,true>` | biasBuf 后半暂存 BF16、前半 FP32 | 同上 |
| R-006 | 10100+`N>8192` | A2/A3，大 N，bias 空 | 单 K-001，body 内按 8192 列切分 | `DequantBiasMultiImpl<X,W,B,Y,false>` | 固定 32 KiB ping/pong 各半，列片更新 ws | 末列 `N%8192` 或 8192 |
| R-007 | 10110+`N>8192` | 同上，bias INT32 | 同上 | `...Multi<X,W,int32_t,Y,true>` | 每列片更新 INT32 bias/ws | 同上 |
| R-008 | 10111+`N>8192` | 同上，bias FP32 | 同上 | `...Multi<X,W,float,Y,true>` | 每列片更新 FP32 bias/ws | 同上 |
| R-009 | 10112+`N>8192` | 同上，bias FP16 | 同上 | `...Multi<X,W,half,Y,true>` | 每列片临时 FP16→FP32 | 同上 |
| R-010 | 10113+`N>8192` | 同上，bias BF16 | 同上 | `...Multi<X,W,bfloat16_t,Y,true>` | 每列片临时 BF16→FP32 | 同上 |

每条 Route 的 `W=BF16,Y=BF16` 编译实例还会在 `wsBuf_` 后半暂存 BF16 权重并 Cast 到前半 FP32；`W=FP32,Y=FP16` 直接 DMA 到前半。该编译期子路径改变 buffer 内部布局，但不改变总容量、Host selector 或 launch graph，故在各 Route 内逐项保留而不伪造新的 Host Route。

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台 | README 支持 A2/A3；OpDef/config 为 `ascend910b/ascend910_93` | 只构建这些平台 binary | 全部 | 950 明确不发布 | README；`DequantBias::DequantBias` |
| UB 总量 | `baseIn+ws+as+bias < ubSize` | `InitBuffer` 总和必须小于平台 UB | 全部 | `GRAPH_FAILED` | `TilingGradCompute` |
| M 上限 | `M<=25000` | 与 UB 检查同一失败分支 | 全部 | `GRAPH_FAILED` | `TilingGradCompute` |
| 小/大 N | `N<=8192` | 小 N 整行驻留；大 N 以 8192 列片驻留 | R-001～R-010 | 另一 body | `dequant_bias` router |
| 行 DMA padding | `(nAlign-N)*sizeof(INT32)>=32` 决定 `dstStride=1` | 每行对齐到 64 B；额外完整 32 B block 时跨过一 block | R-001～R-005 | `dstStride=0` | `ParseBaseTilingData`、`CopyIn` |
| BF16 bias 编译 guard | key 10113 且非 `__NPU_ARCH__==3003/3113` | Kirin 路线不实例化 BF16；A2/A3 不命中该 guard | R-005/R-010 | guard 平台无输出 | `dequant_bias` |
| 未知 key | 仅 10100/10110/10111/10112/10113 | router 无 default body | 全部 | Kernel 返回且不写 `y` | `dequant_bias` |
| 大 N 单行前提 | 支持平台上 `rowsPerLoop==1` | Multi body把 `curCoreLoopRow_` 固定为 1，却消费 Host 的 `perCoreLoops` | R-006～R-010 | 固定仓无 fallback/assert | `TilingGradCompute`、`DequantBiasMultiImpl::ParseBaseTilingData` |

### POD/ABI

`DequantBiasTilingData` 按声明顺序含 17 个 `uint32_t`。固定仓未包含 `tilingdata_base.h` 宏展开，故精确 `sizeof/alignof/offsetof` 不可见；Host `SaveToBuffer` 与 Device `GET_TILING_DATA` 使用同一注册结构，字段顺序不得改变。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `DequantBiasTilingData` | `M` | uint32/32 | 第 1 字段；精确 offset 不可见 | 行 | `set_M(M)` | 仅 Host 后续分核 | 正整数，`<=25000` |
| 同上 | `N` | uint32/32 | 第 2 字段 | 元素/行 | `set_N(N)` | router、两 body | 正整数 |
| 同上 | `nAlign` | uint32/32 | 第 3 字段 | FP32/INT32 元素 | `ceil(N*4/64)*16` | 小 N body、padding 判断 | 64 B 对齐 |
| 同上 | `asExist` | uint32/32 | 第 4 字段 | 布尔 0/1 | 可选 shape 非空 | 两 body | 0 时不读 activateScale |
| 同上 | `needCoreNum` | uint32/32 | 第 5 字段 | AIV block | `ceil(M/perCoreRow)` | block early-return | `<=aivNum` |
| 同上 | `perCoreRow` | uint32/32 | 第 6 字段 | 行 | `ceil(M/aivNum)` | 非尾核、GM offset | 主核行数 |
| 同上 | `tailCoreRow` | uint32/32 | 第 7 字段 | 行 | `M-(needCoreNum-1)*perCoreRow` | 尾核 | 最后一有效核 |
| 同上 | `inBufferSize` | uint32/32 | 第 8 字段 | 字节 | `baseIn*perCoreLoopRow` | 小 N `InitBuffer`；Multi 忽略 | ping+pong 总量 |
| 同上 | `wsBufferSize` | uint32/32 | 第 9 字段 | 字节 | `nAlign*4` 或 32768 | `wsBuf_` | 包含 BF16 临时半区 |
| 同上 | `asBufferSize` | uint32/32 | 第 10 字段 | 字节 | `align32(perCoreRow*4)` 或 0 | `asBuf_` | 仅 `asExist!=0` |
| 同上 | `biasBufferSize` | uint32/32 | 第 11 字段 | 字节 | bias 存在时等于 ws，否则 0 | `biasBuf_` | 仅 `IFBIAS` 实例 |
| 同上 | `perCoreLoopRow` | uint32/32 | 第 12 字段 | 行/循环 | UB 可容纳行数 | 小 N 非尾核 | Multi 固定改写为 1 |
| 同上 | `perCoreTailLoopRow` | uint32/32 | 第 13 字段 | 行 | 主核最后 loop 行数 | 小 N 非尾核 | Multi 固定改写为 1 |
| 同上 | `perCoreLoops` | uint32/32 | 第 14 字段 | loop | `ceil(perCoreRow/perCoreLoopRow)` | 非尾核 | 大 N 仍被 Multi 使用 |
| 同上 | `tailCoreLoopRow` | uint32/32 | 第 15 字段 | 行/循环 | 尾核 UB 可容纳行数 | 小 N 尾核 | Multi 固定改写为 1 |
| 同上 | `tailCoreTailLoopRow` | uint32/32 | 第 16 字段 | 行 | 尾核最后 loop 行数 | 小 N 尾核 | Multi 固定改写为 1 |
| 同上 | `tailCoreLoops` | uint32/32 | 第 17 字段 | loop | `ceil(tailCoreRow/tailCoreLoopRow)` | 尾核 | 大 N 仍被 Multi 使用 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～R-005 | 10100/10110/10111/10112/10113 + `N<=8192` | `dequant_bias_tiling.cpp::DequantBiasTiling` | `DequantBiasTilingData` 全字段 | `dequant_bias.cpp::dequant_bias` | `dequant_bias_impl.h::DequantBiasImpl` 全方法 | A2/A3 binary JSON + `ascendc_config.json` |
| R-006～R-010 | 同 key + `N>8192` | 同上 | 同上 | 同上 | `dequant_bias_multi.h::DequantBiasMultiImpl` 全方法 | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 只有 K-001。entry 解析 POD，以 key 选择 bias 模板，再以 `N<=8192` 选择整行或分列 body。有效 block 绑定 GM，分配 TBuf；无效 block 在 `Init` 与 `Process` 双重 early-return。每个有效核先搬入本核共享的 scale/bias，随后使用 ping/pong 在 `MTE2 -> V -> MTE3` 间以硬事件串行保护同一子区。计算均在输入 buffer 上原地 Reinterpret/Cast/Vector 运算，最终把同一 buffer 解释为 `YTYPE` 搬出；没有跨 Kernel 边和 workspace consumer。

### Route R-001：小 N 无 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `dequant_bias_tiling.cpp` | `GetTilingKey` | `biasExist==0` | 10100 |
| Router | `dequant_bias.cpp` | `dequant_bias` | 10100 且 `N<=8192` | `DequantBiasImpl<...,false>` |

#### Device 数据流

##### 函数级调用链

```text
dequant_bias -> DequantBiasImpl::Init -> Process
  -> ScaleBiasCopyIn -> DataCopyInWeightScale -> [DataCopy activateScale]
  -> per-loop CopyIn -> ComputeDequantWithoutBias -> CopyOut
```

##### 源码等价伪码

```text
for tile in coreRows/rowsPerLoop:
  wait previous MTE3; DMA x[globalRow,0:N] -> ping/pong[nAlign]
  cast INT32->FP32; for row: fp32 *= wsFP32
  if asExist: fp32 *= asLocal[rowInCore]
  cast_rint FP32->Y; DMA first N values -> y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `(block*perCoreRow+tileRow)*N` | `rows*N` | INT32 / Y ND | caller/CopyOut | CopyIn/caller | 调用期 | 每行只提交 N |
| ping/pong | 0 | 各 `inBufferSize/2` | INT32→FP32→Y | CopyIn/Compute | Compute/CopyOut | Kernel | 两块交替，原地复用 |
| wsBuf | 0 | `wsBufferSize` | FP32；BF16 输入时后半临时 | weight DMA/Cast | Mul | Kernel | BF16 临时区与 FP32 结果分半 |
| asBuf | 0 | `asBufferSize` | FP32 | activate DMA | Muls | Kernel | 可选、每核一次 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  ping["pingBuf · inBufferSize/2"]:1
  pong["pongBuf · inBufferSize/2"]:1
  ws["wsBuf · wsBufferSize"]:1
  as["asBuf · asBufferSize if present"]:1
```

##### 字节账与复用顺序

```text
UB_total = inBufferSize + wsBufferSize + (asExist ? asBufferSize : 0)
UB_total < platform UB
dynamic_ub = N/A
workspace_total = 16 MiB reserved; Kernel-used workspace = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ping/pong | CopyIn | Compute、CopyOut | 同奇偶性的下两轮 | `MTE3_MTE2` flag | CopyOut 完成后才重载 |
| xLocal | INT32 DMA/Cast | Mul/Muls/Cast | CopyOut 后 | V/MTE3 barrier | 只输出 N，padding lane 不外泄 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→V | `MTE2_V` flag | DataCopyPad | Compute | ping/pong | nAlign lane，尾 lane不输出 |
| V→CopyOut | `V_MTE3` flag | Cast_rint | DataCopyPad | ping/pong | FP32 中间，Y 输出 |
| buffer reuse | `MTE3_MTE2` flag | CopyOut | 下一 CopyIn | ping/pong | 最后一 loop 用 tailRows |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `TilingSplitCore/TilingGradCompute/GetTilingKey` | 分核、UB、10100 |
| Kernel entry/router | `dequant_bias.cpp` | `dequant_bias` | key+N 到模板 |
| Device body/helper | `dequant_bias_impl.h` | `Process/ComputeDequantWithoutBias` | ping/pong、FP32 缩放、tail |

### Route R-002：小 N INT32 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `dequant_bias_tiling.cpp` | `GetTilingKey` | bias 存在且 dtype 非三种浮点，即经校验后的 INT32 | 10110 |
| Router | `dequant_bias.cpp` | `dequant_bias` | 10110、`N<=8192` | `DequantBiasImpl<...,int32_t,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
Init -> Process -> ScaleBiasCopyIn(INT32 bias)
  -> CopyIn -> ComputeDequantWithBiasInt32(Add INT32, Cast, Mul, optional Muls, Cast_rint) -> CopyOut
```

##### 源码等价伪码

```text
load bias[0:N] once/core; for each tile: load x; for each row xInt += biasInt;
cast to FP32; multiply ws; optionally multiply row scale; cast_rint to Y; store N.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf | 0 | `biasBufferSize=nAlign*4` | INT32 | one DMA/core | INT32 Add | Kernel | padding lane不输出 |
| 其余 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | Kernel | ping/pong |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  ping["ping · inBufferSize/2"]
  pong["pong · inBufferSize/2"]
  ws["ws · wsBufferSize"]
  bias["bias · biasBufferSize"]
  as["as · optional"]
```

##### 字节账与复用顺序

```text
UB_total = inBufferSize + wsBufferSize + biasBufferSize + optional asBufferSize < UB
workspace_total = 16 MiB reserved; Kernel-used = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| xLocal | CopyIn / INT32 Add / Cast | 后继 Vector 与 CopyOut | ping/pong 下一同奇偶轮 | barriers+flags | INT32 加法必须先于 Cast |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| bias→x | `PIPE_ALL` 后 `PIPE_V` | bias DMA、INT32 Add | Cast | bias/xLocal | 加法保留 INT32 语义 |
| 其余 | 同 R-001 | 同 R-001 | 同 R-001 | ping/pong | FP32 缩放、Cast_rint |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey` | 10110 |
| Kernel entry/router | `dequant_bias.cpp` | 10110 分支 | INT32 bias 实例 |
| Device body/helper | `dequant_bias_impl.h` | `ScaleBiasCopyIn/ComputeDequantWithBiasInt32` | bias DMA 与运算顺序 |

### Route R-003：小 N FP32 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | `GetTilingKey` / `dequant_bias` | bias FP32，key 10111，`N<=8192` | `DequantBiasImpl<...,float,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
ScaleBiasCopyIn -> DataCopyInBiasFloat(FP32 direct DMA)
-> CopyIn -> ComputeDequantWithBiasFloat(Cast, Mul, optional Muls, Add, Cast_rint) -> CopyOut
```

##### 源码等价伪码

```text
load FP32 bias once/core; for each tile cast x; multiply ws and optional row scale;
add FP32 bias per row; cast_rint and store first N.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf | 0 | `nAlign*4` | FP32 | direct DMA | Add | Kernel | 单区，无转换临时区 |
| ping/pong/ws/as | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | Kernel | 同 R-001 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  ping["ping"]
  pong["pong"]
  ws["ws FP32"]
  bias["bias FP32"]
  as["as optional"]
```

##### 字节账与复用顺序

```text
UB_total = inBufferSize + wsBufferSize + biasBufferSize + optional asBufferSize < UB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| biasBuf | bias DMA | 每行 Add | Kernel 结束 | initial `PIPE_ALL` | 所有 tile 共用同一列 bias |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scale→bias | `PIPE_V` | Mul/Muls | Add | xLocal FP32 | bias 在缩放之后；nAlign lane |
| 输出 | `CAST_RINT` + V_MTE3 | Add/Cast | DMA | xLocal | 仅 N 输出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey` | 10111 |
| Kernel entry/router | `dequant_bias.cpp` | 10111 分支 | FP32 bias 实例 |
| Device body/helper | `dequant_bias_impl.h` | `DataCopyInBiasFloat/ComputeDequantWithBiasFloat` | 直接 DMA 与后加 bias |

### Route R-004：小 N FP16 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | `GetTilingKey` / `dequant_bias` | bias FP16，10112，`N<=8192` | `DequantBiasImpl<...,half,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
DataCopyInBiasFloat: DMA FP16 -> biasBuf second half -> Cast to FP32 first half
-> ComputeDequantWithBiasFloat -> CopyOut
```

##### 源码等价伪码

```text
tmpHalf=biasBuf[offset=biasBufferSize/2]; DMA N FP16; barrier; cast tmpHalf->biasFp32;
then reuse the FP32-bias computation order from R-003.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf first half | 0 | `biasBufferSize/2` bytes | FP32 result | Cast | Add | Kernel | 与后半不重叠 |
| biasBuf second half | `biasBufferSize/2` | `biasBufferSize/2` bytes | FP16 temp | DMA | Cast | 初始化阶段 | Cast 后不再读取 |
| 其余 | 同 R-003 | 同 R-003 | 同 R-003 | 同 R-003 | 同 R-003 | Kernel | 同 R-003 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  ping["ping"]
  pong["pong"]
  ws["ws"]
  biasFp32["bias first 1/2 · FP32"]
  biasTmp["bias second 1/2 · FP16"]
  as["as optional"]
```

##### 字节账与复用顺序

```text
UB_total 与 R-003 相同；biasBufferSize 已同时覆盖 FP32 N 元素与 FP16 临时 N 元素的分半布局。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bias second half | DMA | Cast | 不再覆盖 | 动态 `MTE2_V` event | N 个 FP16 可容纳于半区 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| bias conversion | fetched `MTE2_V` event + `PIPE_V` | DMA | Cast/Add | biasBuf 两半 | FP16→FP32 `CAST_NONE` |
| 后续 | 同 R-003 | 同 R-003 | 同 R-003 | xLocal | FP32 Add、Cast_rint |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey` | 10112 |
| Kernel entry/router | `dequant_bias.cpp` | 10112 分支 | half bias |
| Device body/helper | `dequant_bias_impl.h` | `DataCopyInBiasFloat` | 分半暂存与升精度 |

### Route R-005：小 N BF16 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | `GetTilingKey` / `dequant_bias` | bias BF16，10113，`N<=8192`，A2/A3 编译 guard 有效 | `DequantBiasImpl<...,bfloat16_t,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
同 R-004，但临时输入为 BF16；Cast 到 FP32 后进入 ComputeDequantWithBiasFloat。
```

##### 源码等价伪码

```text
DMA BF16 bias to second half; wait MTE2_V; cast BF16->FP32 first half;
scale x in FP32, then add converted bias, cast_rint to Y.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf | 与 R-004 相同 | `nAlign*4` | 前半 FP32、后半 BF16 | DMA/Cast | Add | Kernel | 分半 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  biasFp32["biasBuf first half · FP32"]
  biasTmp["biasBuf second half · BF16"]
```

##### 字节账与复用顺序

```text
UB_total = R-003 总量；BF16 临时区与 FP32 结果共同装入 biasBufferSize。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| biasBuf 两半 | DMA/Cast | Add | Kernel 结束 | MTE2_V + PIPE_V | A2/A3 实例未被 arch guard 排除 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| bias conversion | 同 R-004 | DMA | Cast/Add | biasBuf | BF16→FP32 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey` | 10113 |
| Kernel entry/router | `dequant_bias.cpp` | 10113 + arch guard | A2/A3 BF16 bias |
| Device body/helper | `dequant_bias_impl.h` | `DataCopyInBiasFloat/ComputeDequantWithBiasFloat` | BF16 升精度后加 |

### Route R-006：大 N 无 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | 10100 / `N>8192` | bias 空，大 N | `DequantBiasMultiImpl<...,false>` |

#### Device 数据流

##### 函数级调用链

```text
Init -> Process -> DataCopyInActivateScale
-> for colPart: DataCopyInWeightScale -> for row: CopyIn -> ComputeDequantWithoutBias -> CopyOut
```

##### 源码等价伪码

```text
for col=0..ceil(N/8192)-1:
  nProcess = last ? N%8192 : 8192 (exact multiple remains 8192)
  load/cast weightScale[col*8192 : +nProcess]
  for each core row: load x[row,colSlice]; cast/mul/[Muls]; cast_rint; store nProcess
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM slice | `row*N+col*8192` | `nProcess` | INT32/Y | caller/body | body/caller | 调用期 | 末列精确长度 |
| ping/pong | 0 | 各 32768 B | INT32→FP32→Y | CopyIn/Compute | CopyOut | Kernel | 行间交替 |
| wsBuf | 0 | 32768 B | FP32/可含 BF16 temp | 每列片更新 | Mul | Kernel | 跨行复用 |
| asBuf | 本核行起点 | `align32(coreRows*4)` | FP32 | 一次 DMA/core | Muls | Kernel | 跨列复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  ping["ping · 32768 B"]
  pong["pong · 32768 B"]
  ws["ws · 32768 B"]
  as["as · optional"]
```

##### 字节账与复用顺序

```text
UB_total = 65536 + 32768 + optional asBufferSize
workspace_total = 16 MiB reserved; Kernel-used = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| wsBuf | current-column DMA/Cast | all rows in column | next column | `PIPE_ALL` around update | 完成当前列所有行后更新 |
| ping/pong | row CopyIn | row Compute/CopyOut | same parity later row | hard events | CopyOut 先完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| column update | `PIPE_ALL` | prior row output | bias/ws DMA | shared buffers | nProcess 64 B 对齐仅用于 Vector |
| row pipeline | MTE3_MTE2/MTE2_V/V_MTE3 | 各阶段 | 下一阶段 | ping/pong | 末列只 CopyOut nProcess |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `TilingGradCompute` | 8192 固定片容量 |
| Kernel entry/router | `dequant_bias.cpp` | 10100 大 N 分支 | Multi body |
| Device body/helper | `dequant_bias_multi.h` | `Process/ComputeDequantWithoutBias` | 列片×行循环 |

### Route R-007：大 N INT32 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | 10110 / `N>8192` | INT32 bias | `DequantBiasMultiImpl<...,int32_t,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
每列片 DataCopyInBias(INT32)+DataCopyInWeightScale
-> 每行 CopyIn -> ComputeDequantWithBiasInt32 -> CopyOut
```

##### 源码等价伪码

```text
for each 8192-column part load biasInt and ws; for each row add INT32 bias,
then cast FP32, multiply scales, cast_rint, store exact nProcess.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf | 0 | 32768 B | INT32 | 每列片 DMA | Add | Kernel | 跨当前列所有行复用 |
| 其余 | 同 R-006 | 同 R-006 | 同 R-006 | 同 R-006 | 同 R-006 | Kernel | 同 R-006 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  ping["ping 32 KiB"]
  pong["pong 32 KiB"]
  ws["ws 32 KiB"]
  bias["bias 32 KiB"]
  as["as optional"]
```

##### 字节账与复用顺序

```text
UB_total = 65536 + 32768 + 32768 + optional asBufferSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| bias/ws | column DMA | every row Compute | next column | outer `PIPE_ALL` | 所有行消费完再换片 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| INT32 Add | PIPE_V | x/bias DMA | Cast | xLocal | 先加再反量化 |
| column tail | `nAlign=align64(nProcess*4)/4` | router loop | Vector/CopyOut | last slice | CopyOut 只 nProcess |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey/TilingGradCompute` | 10110、8192 |
| Kernel entry/router | `dequant_bias.cpp` | 10110 Multi 分支 | INT32 bias Multi |
| Device body/helper | `dequant_bias_multi.h` | `DataCopyInBias/ComputeDequantWithBiasInt32` | 分片 bias、运算顺序 |

### Route R-008：大 N FP32 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | 10111 / `N>8192` | FP32 bias | `DequantBiasMultiImpl<...,float,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
每列片 DataCopyInBiasFloat(FP32 direct)+weight -> 每行 ComputeDequantWithBiasFloat
```

##### 源码等价伪码

```text
load current FP32 bias/ws slice; for each row: cast x, multiply ws and optional as,
add bias, cast_rint, store current slice.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf | 0 | 32768 B | FP32 | current-column DMA | Add | Kernel | 跨当前列行复用 |
| 其余 | 同 R-006 | 同 R-006 | 同 R-006 | 同 R-006 | 同 R-006 | Kernel | 同 R-006 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xbuf["ping+pong · 64 KiB"]:2
  ws["ws · 32 KiB"]
  bias["bias · 32 KiB"]
```

##### 字节账与复用顺序

```text
UB_total = 131072 + optional asBufferSize
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| biasBuf | current column DMA | all current-column rows | next column | PIPE_ALL | Add 在缩放后 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| FP32 pipeline | PIPE_V | Cast/Mul/Muls/Add | Cast_rint | xLocal | 浮点 bias 最后加 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey` | 10111 |
| Kernel entry/router | `dequant_bias.cpp` | 10111 Multi | FP32 bias |
| Device body/helper | `dequant_bias_multi.h` | `DataCopyInBiasFloat/ComputeDequantWithBiasFloat` | 每列直接 DMA、后加 |

### Route R-009：大 N FP16 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | 10112 / `N>8192` | FP16 bias | `DequantBiasMultiImpl<...,half,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
每列片 FP16 DMA 到 biasBuf 后半 -> Cast 到前半 FP32 -> 每行浮点 bias body
```

##### 源码等价伪码

```text
for col: DMA half bias to offset 16 KiB; wait; cast to FP32 at offset 0;
then process every row as R-008 before moving to next col.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf first/second half | 0 / 16384 B | 各 16384 B | FP32 result / FP16 temp | Cast / DMA | Add / Cast | 当前列 | 每列覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  fp32["bias first half · FP32"]
  fp16["bias second half · FP16 temp"]
```

##### 字节账与复用顺序

```text
UB_total = 131072 + optional asBufferSize；bias 两半合计仍为 32768 B。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| biasBuf | DMA/Cast | current-column rows | next column | MTE2_V + outer PIPE_ALL | nProcess<=8192 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| bias升精度 | fetched MTE2_V | FP16 DMA | Cast | biasBuf | FP16→FP32；末列仅 nProcess |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey` | 10112 |
| Kernel entry/router | `dequant_bias.cpp` | 10112 Multi | FP16 bias |
| Device body/helper | `dequant_bias_multi.h` | `DataCopyInBiasFloat` | 每列分半 conversion |

### Route R-010：大 N BF16 bias

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Router | `dequant_bias_tiling.cpp` / `dequant_bias.cpp` | 10113 / `N>8192` | BF16 bias，A2/A3 | `DequantBiasMultiImpl<...,bfloat16_t,...,true>` |

#### Device 数据流

##### 函数级调用链

```text
同 R-009，临时 dtype 为 BF16，随后复用 FP32 bias 计算管线。
```

##### 源码等价伪码

```text
for col: DMA BF16 bias -> second half; cast BF16->FP32 first half;
for each row: x*ws*[as]+biasFp32; cast_rint; exact tail store.
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| biasBuf first/second half | 0 / 16384 B | 各 16384 B | FP32 / BF16 | Cast / DMA | Add / Cast | 当前列 | 每列覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  fp32["bias first half · FP32"]
  bf16["bias second half · BF16 temp"]
```

##### 字节账与复用顺序

```text
UB_total = R-009；dynamic_ub=N/A；workspace Kernel-used=0。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| biasBuf | DMA/Cast | current column | next column | MTE2_V/PIPE_ALL | A2/A3 guard 有效 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16升精度 | MTE2_V + PIPE_V | DMA/Cast | Add | biasBuf | BF16→FP32→Y |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `dequant_bias_tiling.cpp` | `GetTilingKey` | 10113 |
| Kernel entry/router | `dequant_bias.cpp` | 10113 Multi + guard | A2/A3 BF16 bias |
| Device body/helper | `dequant_bias_multi.h` | `DataCopyInBiasFloat/ComputeDequantWithBiasFloat` | BF16 升精度与末列 tail |

## 算子特定优化

### Trick T-001：UB 感知的多行批处理与 ping/pong 原地流水

#### 触发与源码依据

- 适用 Route：R-001～R-005。
- 触发条件：`N<=8192` 且 `baseIn+ws+as+bias<UB`；Host 用剩余 UB 计算每轮行数。
- Golden 来源：[`quant/dequant_bias/op_host/dequant_bias_tiling.cpp:428-442`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_host/dequant_bias_tiling.cpp#L428-L442)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t totalBuffSize = inBuffSize + weightScaleBuffSize + asBuffSize + biasBuffSize;
if (totalBuffSize >= aicoreParams_.ubSize || M > SIZE_25000) {
    OP_LOGE(context_->GetNodeName(), "input x size is too large.");
    return ge::GRAPH_FAILED;
}

uint32_t releaseUbSizeForX = aicoreParams_.ubSize - (weightScaleBuffSize + asBuffSize + biasBuffSize);
perCoreLoopRow = std::min(releaseUbSizeForX / inBuffSize, perCoreRow);
tailCoreLoopRow = std::min(releaseUbSizeForX / inBuffSize, tailCoreRow);
perCoreLoops = Ops::Base::CeilDiv(perCoreRow, perCoreLoopRow);
perCoreTailLoopRow = perCoreRow - perCoreLoopRow * (perCoreLoops - 1);
tailCoreLoops = Ops::Base::CeilDiv(tailCoreRow, tailCoreLoopRow);
tailCoreTailLoopRow = tailCoreRow - tailCoreLoopRow * (tailCoreLoops - 1);

inBuffSize *= perCoreLoopRow;
```

#### 机制与设计原因

固定 scale/bias/as buffer 后，把剩余 UB 尽量分配给一轮多行 x；`inBufferSize` 本身已含双缓冲因子 2，Device 再均分成 ping/pong。每个 tile 的 INT32 输入、FP32 中间和低精度输出都在同一物理 x buffer 原地解释，省去独立 FP32/输出物化；硬事件确保下一次 MTE2 覆盖发生在 MTE3 输出结束后。

#### 迁移边界

- 适用：逐行独立、按列广播且可在同一 buffer 原地升/降精度的 Vector 算子。
- 前提：64 B 行对齐、所有固定 buffer 计入 UB、ping/pong 深度计入两次、tailRows 单独下发。
- 失效：`N>8192` 必须转 R-006～R-010 的 8192 列片 body；UB 不足则 Host 失败。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓可验证的错误边界是忽略 `bias/as/ws` 后再用全部 UB 估算 x，会使 `InitBuffer` 总和超过 UB；源码以总量检查和剩余量公式规避。固定仓未提供真机性能反例。

### Trick T-002：大 N 以 8192 列片复用 scale/bias

#### 触发与源码依据

- 适用 Route：R-006～R-010。
- 触发条件：`N>8192`。
- Golden 来源：[`quant/dequant_bias/op_kernel/dequant_bias_multi.h:156-166`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_kernel/dequant_bias_multi.h#L156-L166)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t startRowOffset = blockIdx_ * mainCoreRow_;
for (int64_t colIdx = 0; colIdx < (N_ + PART_LEN - 1) / PART_LEN; colIdx++) {
    int64_t nFullParts = N_ / PART_LEN;
    int64_t nProcess = colIdx == nFullParts ? N_ % PART_LEN : PART_LEN;
    int64_t weightOffset = colIdx * PART_LEN;

    // update weight_scale and bias
    PipeBarrier<PIPE_ALL>();
    DataCopyInBias(weightOffset, nProcess);
    DataCopyInWeightScale(weightOffset, nProcess);
    PipeBarrier<PIPE_ALL>();
```

#### 机制与设计原因

整行不能驻留时，`PART_LEN=8192` 把 x、weightScale 和 bias 同步切成列片。每列只装载一次 scale/bias，然后跨本核所有行复用；activateScale 按行只在外层开始时装载一次并跨全部列片复用。末列用 `nProcess` 控制 DMA 和有效输出，Vector 长度则按 64 B 对齐。

#### 迁移边界

- 适用：二维逐元素主体、按列参数可跨行复用的大 N 算子。
- 前提：列片之间无归约依赖；输出可按 slice 独立提交；当前平台 `rowsPerLoop==1`。
- 失效：小 N 使用整行多行 body；若未来平台使 Host 算出 `rowsPerLoop>1`，必须同步修改 Multi loop，而不能直接复用当前代码。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓显示 Multi body把 `curCoreLoopRow_` 固定为 1，却继续读取 Host 的 loop 次数；在未证明 `rowsPerLoop==1` 的新平台直接迁移会漏行。固定仓未提供对应 fallback。

### Trick T-003：同一 scale/bias TBuf 的分半升精度

#### 触发与源码依据

- 适用 Route：所有 `WSTYPE=BF16` 子实例，以及 R-004/R-005/R-009/R-010。
- 触发条件：输入 scale 或 bias 不是 FP32。
- Golden 来源：[`quant/dequant_bias/op_kernel/dequant_bias_impl.h:329-347`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_bias/op_kernel/dequant_bias_impl.h#L329-L347)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
wsLocal_ = wsBuf_.Get<float>();
int64_t wsTypeNum = 0;
DataCopyExtParams wsParams {1, static_cast<uint32_t>(N_ * sizeof(WSTYPE)), 0, 0, 0};
if constexpr(!IsSameType<WSTYPE, float>::value) {
    DataCopyPadExtParams<WSTYPE> padParams {false, 0, 0, 0};
    wsTypeNum = wsBufferSize_ / sizeof(WSTYPE) / 2;
    LocalTensor<WSTYPE> tmpBuf = wsBuf_.GetWithOffset<WSTYPE>(wsTypeNum, wsBufferSize_ / 2);
    DataCopyPad(tmpBuf, weightScaleGM_, wsParams, padParams);
    event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2V);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
    Cast(wsLocal_, tmpBuf, RoundMode::CAST_NONE, N_);
    PipeBarrier<PIPE_V>();
} else {
    DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
    DataCopyPad(wsLocal_, weightScaleGM_, wsParams, padParams);
}
```

#### 机制与设计原因

Host 始终按 FP32 结果大小分配 `nAlign*4` 字节。低精度输入只需一半字节，因此暂存在后半区并 Cast 到前半区，避免额外 TBuf。事件保证 DMA 完成后才读临时区，`PIPE_V` 保证后续 Mul/Add 看到升精度结果。bias helper 使用相同布局。

#### 迁移边界

- 适用：源 dtype 2 B、计算 dtype 4 B，且结果和临时输入总字节恰好不超过单个 FP32 buffer 的场景。
- 前提：两半 offset/元素数按字节计算，Cast 前不得覆盖后半，后续只读前半。
- 失效：源 dtype 与目标 dtype 比例不同或需要同时保留完整源/结果时，应增加容量或独立 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码中的 `MTE2_V` event 是必要依赖；删除等待会让 Vector Cast 读取仍在 DMA 的临时区。固定仓未提供可验证的数值反例。

## 模板与实现对比

| 对比维度 | R-001～R-005 小 N | R-006～R-010 大 N | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | key 决定 bias，`N<=8192` | 同 key，`N>8192` | entry 先 key 后 N |
| 模板与 dtype | `DequantBiasImpl<X,W,B,Y,IFBIAS>` | `DequantBiasMultiImpl<X,W,B,Y,IFBIAS>` | binary config 决定 W/Y/B 编译 dtype |
| 循环与分核 | 核→多行 tile→行 | 核→8192 列片→单行 loop | 大 N 控制常驻列宽 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | TBuf ping/pong/ws/可选 bias/as；无 Queue、dynamic UB、通信 | 同资源类别，ping/pong/ws/bias 固定片容量 | AIV Vector 单 Kernel |
| 内存总占用与复用 | `inBufferSize+ws+bias+as`，x 原地、ping/pong | 64 KiB x + 32 KiB ws + 可选 32 KiB bias/as；列间覆盖 | Host UB guard |
| 搬运/Vector API | 多行 `DataCopyPad`，Cast/Mul/Muls/Add/Cast | 单行列片同 API | 末列精确 DMA |
| 同步、精度与 tail | hard events，FP32 中间，tailRows | 同步外加列间 PIPE_ALL，tailCols | bias dtype 决定 Add 前后 |
| fallback/失败边界 | UB/M/shape/dtype 失败 | 同上，另依赖 rowsPerLoop==1 | 无其它 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；公开文档 + `ascend910b` config | 支持；公开文档 + `ascend910_93` config | 公开 README/API 明确不支持 Ascend 950PR/950DT |
| Host/Launcher | 共用 `DequantBiasTiling` | 共用 | 不发布、不生成 `_950` 卡 |
| Kernel/资源 | 共用 Vector entry/body 和 8 组 binary dtype 配置 | 与 A2 配置集合一致 | `ascendc_config.json` 无 `ascend950`，OpDef 无 950 config |

## 限制与失败边界

- 公开参数表一处写“支持空 Tensor”，但公开约束又要求 `M,N` 为正；算子 Tiling 对 `N=0` 的 `inBuffSize=0` 会进入除零风险路径。固定仓看不到 ACLNN 生成层是否短路空 Tensor，因此实现复用按 `M,N>0`，空 Tensor 行为标记为框架边界缺失，不推测。
- `M`/`N` 从有符号 shape 窄化为 `uint32_t`，固定算子代码未显式检查 `N` 上限和窄化；公开契约只给出 `M<=25000`。调用者必须遵守实际 Tensor/tiling ABI 可表示范围。
- Host 将 `blockDim` 设为全部 AIV 数，Device 依赖 `blockIdx>=needCoreNum` early-return；不得把 `needCoreNum` 与 launch blockDim 混同。
- workspace 固定为 16 MiB，但 Kernel 不用；不得把它计作 UB 或跨 Kernel中间结果。
- Vector 操作覆盖 64 B 对齐后的 `nAlign`，padding lane 的输入/scale可能未初始化，但 CopyOut 仅写 N；迁移时必须保持无效 lane 不影响有效 lane的逐元素性质。
- 未知 TilingKey 和被编译 guard 排除的 10113 分支不会写输出且无 Device 错误码；只允许通过 Host 产生 key。
- 950 公开不支持，`dequant_bias_950.md` 未生成。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/dequant_bias/docs/aclnnDequantBias.md` | `aclnnDequantBiasGetWorkspaceSize/aclnnDequantBias` | 两段式 API、参数、错误、A2/A3/950 支持 |
| 入口声明/约束 | `quant/dequant_bias/README.md` | 产品支持表、公式 | 平台和三种 bias 语义 |
| 入口实现/构建 | `quant/dequant_bias/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn)` | ACLNN 生成边界接线 |
| OpDef | `quant/dequant_bias/op_host/dequant_bias_def.cpp` | `ops::DequantBias` | dtype 组合、A2/A3 config |
| shape/dtype | `quant/dequant_bias/op_host/dequant_bias_infershape.cpp` | `InferShape4DequantBias/InferDataType4DequantBias` | 输出 shape/type |
| POD | `quant/dequant_bias/op_host/dequant_bias_tiling.h` | `DequantBiasTilingData` | 17 字段 ABI |
| Host/Tiling | `quant/dequant_bias/op_host/dequant_bias_tiling.cpp` | `DequantBiasTiling` 全方法 | 校验、分核、UB、key、workspace |
| Host 公共基类 | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | Tiling 调用顺序与失败传播 |
| Kernel entry/body | `quant/dequant_bias/op_kernel/dequant_bias.cpp` | `dequant_bias` | key/N router、模板实例、guard |
| Kernel entry/body | `quant/dequant_bias/op_kernel/dequant_bias_impl.h` | `DequantBiasImpl` | 小 N 完整 Vector body |
| Kernel entry/body | `quant/dequant_bias/op_kernel/dequant_bias_multi.h` | `DequantBiasMultiImpl` | 大 N 完整列片 body |
| 构建接线 | `quant/dequant_bias/op_host/config/ascend910b/dequant_bias_binary.json` | `op_list` | A2 八组 dtype binary |
| 构建接线 | `quant/dequant_bias/op_host/config/ascend910_93/dequant_bias_binary.json` | `op_list` | A3 八组 dtype binary |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `DequantBias` entry | compute units、auto_sync |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 17 个 uint32 producer/consumer 已闭合 | 字段顺序、字节单位、inactive 字段 | `DequantBiasTilingData` |
| 调用顺序 | scale/bias/as 先装载，再 x tile；大 N 先列后行 | INT32 bias 在 Cast 前，浮点 bias 在缩放后 | 两个 `ComputeDequant*` 族 |
| workspace/内存 | Kernel 不用 16 MiB workspace；UB 按真实 TBuf 求和 | double buffer 必须乘 2；低精度临时区按半区 offset | `GetWorkspaceSize/Init` |
| 同步 | ping/pong 三阶段硬事件、低精度 DMA→Cast event、列间 PIPE_ALL | 覆盖前必须等待消费者 | `Process/DataCopyInWeightScale/DataCopyInBiasFloat` |
| 精度与尾块 | FP32 缩放、`CAST_RINT` 输出；只输出 N/nProcess | bias 顺序、64 B nAlign、末行/末列有效长度 | `Compute*`、`CopyOut` |
| 失败边界 | shape/dtype/UB/M/key/platform边界明确 | 不绕过 Host；不向 950 发布；Multi 需单行前提 | `CheckParamsValidity*`、router |

- [FetchEventID](../../../../apis/simd_api/basic_api/resource_management/t_pipe/fetch_event_id.md)
- [GET_TILING_DATA](../../../../apis/simd_api/basic_api/kernel_tiling/get_tiling_data.md)
- [GetCoreMemSize](../../../../apis/utils_api/platform_info/platform_ascend_c/get_core_mem_size.md)
- [GetTPipePtr](../../../../apis/simd_api/basic_api/resource_management/get_t_pipe_ptr.md)
- [GetWithOffset](../../../../apis/simd_api/basic_api/resource_management/t_buf/get_with_offset.md)
