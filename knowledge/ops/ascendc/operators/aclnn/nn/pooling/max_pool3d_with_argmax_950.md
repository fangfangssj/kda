---
type: Operator
title: "MaxPool3dWithArgmax 950 ACLNN 算子知识"
description: "说明 aclnnMaxPool3dWithArgmax 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/docs/aclnnMaxPool3dWithArgmax.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool3d_with_argmax.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_api/max_pool3d_with_argmax_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_tiling_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_gather_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_tiling_big_kernel_regbase.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_simt_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_gather.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_big_kernel_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_host/config/ascend950/max_pool3d_with_argmax_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:21:03Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MaxPool3dWithArgmax 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMaxPool3dWithArgmaxGetWorkspaceSize` + `aclnnMaxPool3dWithArgmax`；底层 `MaxPool3DWithArgmaxV2` 带 950 `dtype` 属性。 |
| 计算语义 | 每个输出点对 dilation 3D 窗口取最大值，同时输出单通道输入 DHW 线性索引；NaN 更新顺序由各 Route 明确实现。 |
| 输入输出 | FP16/BF16/FP32；NCDHW 或 NDHWC；indices 为 INT32 或 INT64；4D 输入由 ACLNN 扩为 5D。 |
| 执行类型 | 单 Kernel。R-001/R-002/R-003 为 AIV_VECTOR；R-004～R-007 为 SIMT。没有 AIC/Cube/MIX_CV 节点。 |
| Kernel 路线 | Gather 无 pad/有 pad；RegBase 大 W 窗口；SIMT 的 NCDHW/NDHWC × 32/64 位处理索引四族。 |
| 核心资源 | Gather 使用 depth=2 Queue、padding 专用 TBuf与寄存器；BigKernel 使用 depth=2 Queue + 两个输出 TBuf；SIMT 直接 GM load/store、无 AIV UB；均声明 16 MiB system workspace。 |
| 来源闭包 | `source_closed`：公开 API、950 OpDef/config、priority selector、三个 POD、单 entry router、Gather/BigKernel/SIMT body 与构建接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对输出 `(n,c,od,oh,ow)`，起点为 `(od*sD-pD,oh*sH-pH,ow*sW-pW)`；窗口采样 `(startD+kd*dD,startH+kh*dH,startW+kw*dW)` 中落在输入域的值。`out` 是最大值，`indices=d*H*W+h*W+w`，NDHWC 也输出不含 channel 的 DHW 坐标。输出 shape 使用 floor/ceil pooling 公式，ceil 模式丢弃完全位于右 padding 的最后窗口。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 4D 或 5D，NCDHW/NDHWC/ND storage | FP16/BF16/FP32 | 可非连续；非空核心计算。950 不施加 A2/A3 的 DHW INT32 上限。 |
| `kernelSize` | 长度 1 或 3 | INT64 数组 | 每项 >0；长度 1 广播。 |
| `stride` | 长度 0/1/3 | INT64 数组 | 空数组取 kernel；每项 >0。 |
| `padding` | 长度 1/3 | INT64 数组 | `0<=p_i<=floor(k_i/2)`；无效采样视作负无穷。 |
| `dilation` | 长度 1/3 | INT64 数组 | 每项 >0；950 支持非 1，并由 SIMT 承接。 |
| `ceilMode` | 标量 | BOOL | 输出形状取整。 |
| `out` | shape 与推导一致，format 与 self 一致 | 与 self 相同 | ViewCopy 写回用户 tensor。 |
| `indices` | shape/format 与 out 一致 | INT32/INT64 | L0 将 dtype 编为属性 3/9。 |

### 入口与版本

固定仓公开文档明确支持 Ascend 950PR/950DT；OpDef `ascend950` 配置与 binary JSON 为 FP16/BF16/FP32 × INT32/INT64 的 ND FormatAgnostic 组合。API 用 `AclnnUtil::IsRegbase` 开启 950 dtype/format/index 能力。固定仓无更新的同语义公开 API 版本；2D 与 backward/grad 变体不属于本卡。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段创建 executor，校验 RegBase 平台、非空指针、dtype、输入/输出同 storage format、rank 4/5 以及属性长度和值。空 tensor 返回 workspace 0。非空输入先 `Contiguous`；4D 先 `UnsqueezeNd(0)`，再 ReFormat；L0 扩展 1/0 项属性为 D/H/W 三项，计算临时输出 shape，分配临时 `out/indices`，并追加 `MaxPool3DWithArgmaxV2` AICore 节点。RegBase launcher 额外传 `dtype=3/9`。结果 `Squeeze/ReFormat` 后 `ViewCopy` 到用户输出。第二段在给定 stream 调 `CommonOpExecutorRun`。

### 参数、Workspace 与生命周期

算子 Tiling 三族都写一个 16 MiB system workspace；核心 950 body 未读写 user workspace 参数。ACLNN 总 workspace 还聚合 Contiguous/ReFormat/ViewCopy 框架节点，具体 slice 为 executor 外部边界。临时 tensor/属性数组由 executor 持有到第二段完成；核心 Kernel 是同 stream 上单节点，无 Kernel 间 workspace 边。

### 启动图与失败传播

```text
ACLNN phase 1
  -> framework Contiguous / optional Unsqueeze / ReFormat
  -> MaxPool3DWithArgmaxV2 launcher(dtype attr=3 or 9)
     -> priority 0 Gather
     -> priority 1 BigKernelRegbase
     -> priority 2 SIMT (IsCapable=true fallback)
     -> max_pool3d_with_argmax_v2 single Device node
  -> framework ReFormat / optional Squeeze / ViewCopy
ACLNN phase 2 -> CommonOpExecutorRun(workspace, stream)
```

每个 Route 仅有一个 Device 节点，输入/输出为临时 GM tensor；stream 顺序使后处理只在核心节点成功后执行。priority 模板 shape/参数硬错误返回 `GRAPH_FAILED` 会停止，不会 fallthrough；只有 `GRAPH_PARAM_INVALID`/`IsCapable=false` 才尝试下一模板。

## Route 与参数设计

### 分核与资源计算

Registry 按 priority 0、1、2 升序。Gather 搜索 `highAxis(N*C),Dout,Hout,Wout` 内块，按 `NC→D→H→W` 依次尝试，二分最大满足 `UB_total<=availableUb` 且外块数至少 64 的 factor；`totalBaseBlockNum=outerNC*outerD*outerH*outerW`，每 core 处理 `ceil(total/coreNum)` 个，尾 core 处理余数。BigKernel 的 `totalIdx=N*C*Dout*Hout*Wout` 均分至 AIV core。SIMT 令 `threads=min(outputCount,256)`、`blockNums=min(ceil(outputCount/threads),AIVCoreNum)`；Device 实际 `asc_vf_call(dim3(256))`，多出的 thread 由 `index<count` guard 保护。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 Gather 无 padding | priority 0，400001 | RegBase、NCDHW、dilation=(1,1,1)、`wKernel*inputBytes<128`、最小 tile `totalBufferSize <= availableUb/(16*dRate*hRate*wRate)`，且 `isPad=0` | priority 1 |
| 1 | R-002 Gather padding | priority 0，400002 | 同 R-001，但任一显式 pad 非零，或 ceilMode 在无显式 pad 时使末端输入覆盖不等式成立，令 `isPad=1` | priority 1 |
| 2 | R-003 BigKernelRegbase | priority 1，611110 | RegBase、NCDHW、dilation=1、`maxCount>1024`、`wKernel*dtypeSize>=128` | priority 2 |
| 3 | R-004 SIMT NCDHW-32 | priority 2，600001 | RegBase、data_format=NCDHW、`outputCount<=INT32_MAX`；前两模板未命中 | tiling 硬错误 |
| 3 | R-005 SIMT NDHWC-32 | priority 2，600002 | RegBase、data_format=NDHWC、`outputCount<=INT32_MAX`；前两模板因 format 不命中 | tiling 硬错误 |
| 3 | R-006 SIMT NCDHW-64 | priority 2，600011 | RegBase、NCDHW、`outputCount>INT32_MAX` | tiling 硬错误 |
| 3 | R-007 SIMT NDHWC-64 | priority 2，600012 | RegBase、NDHWC、`outputCount>INT32_MAX` | tiling 硬错误 |

```text
if Gather.IsCapable(): key = isPad ? 400002 : 400001
else if BigKernelRegbase.IsCapable(): key = 611110
else:
  key = format==NCDHW
          ? (outputCount<=INT32_MAX ? 600001 : 600011)
          : (outputCount<=INT32_MAX ? 600002 : 600012)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 400001 | Gather 条件且无需 pad | 单 AIV | `GatherKernel<DTYPE_X,DTYPE_ARGMAX,PAD_DISABLE>` | depth2 input/max/index Queue；无 calc TBuf | 四轴 tail；失败→R-003/4/6 |
| R-002 | 400002 | Gather 条件且需 pad | 单 AIV | `GatherKernel<...,PAD_ENABLE>` | R-001 + 单份 input calc TBuf，先填 -inf | 四轴 tail；失败→后续 |
| R-003 | 611110 | NCDHW dilation1 且宽窗口字节>=128 | 单 AIV | `BigKernelRegbase<DTYPE_X,float,DTYPE_ARGMAX>` | depth2 input Queue + 1024 输出 TBuf；MTE/Vector event | kernel D/H/W 分段；失败→SIMT |
| R-004 | 600001 | NCDHW，count<=INT32 | 单 SIMT | `Simt<DTYPE_X,DTYPE_ARGMAX,NCDHW,false>` | GM direct，uint32 magic division | grid-stride tail |
| R-005 | 600002 | NDHWC，count<=INT32 | 单 SIMT | `Simt<...,NDHWC,false>` | GM direct，uint32 magic division | grid-stride tail |
| R-006 | 600011 | NCDHW，count>INT32 | 单 SIMT | `Simt<...,NCDHW,true>` | GM direct，uint64 processing/index math | grid-stride tail |
| R-007 | 600012 | NDHWC，count>INT32 | 单 SIMT | `Simt<...,NDHWC,true>` | GM direct，uint64 processing/index math | grid-stride tail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Gather small W | `wKernel*inputBytes < 128` | Gather 以寄存器索引加载小宽窗；128 B 为 cache-line 门槛 | R-001/2 | R-003 或 SIMT | `GatherTiling::IsCapable` |
| Gather minimum output rate | `totalBufferSize <= UB/(16*dRate*hRate*wRate)` | 保证最小 tile 且考虑 stride/kernel 稀疏率 | R-001/2 | 后续 priority | 同上 |
| Gather UB | no-pad `2I+2M+2A+2048<=UB`; pad `3I+2M+2A+2048<=UB` | Queue 双缓冲、pad calc TBuf 与 2 KiB helper 保守预算 | R-001/2 | 缩 factor或后续 | `DoBufferCalculate` |
| BigKernel maxCount | `floor_align((UB-12KiB)/2,vRegSize)/dtypeSize >1024` | 两个输入槽和最坏 FP32 value + INT64 index 输出 TBuf | R-003 | SIMT | `BigKernelRegbaseTiling::IsCapable` |
| SIMT width | `outputCount <=/> INT32_MAX` | 反解输出坐标与 loop index 选择 uint32/uint64 magic division | R-004～7 | 对应 64 位族 | `SIMT::GetTilingKey/Compute` |
| format | Gather/BigKernel 只 NCDHW | 两 body 的 GM 地址式按 NCDHW | R-001～3 | NDHWC 总是 SIMT | `GetShapeAttrsInfo/IsCapable` |

### POD/ABI

三个 POD 均只含连续 `int64_t`，因此对该定义 `alignof=8`，字段按声明顺序每项 8 B：Gather `sizeof=296`，BigKernel `sizeof=192`，SIMT `sizeof=184`。以下 `offset` 以字节计。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| Gather | `dInput` | int64/64 | off 0 | 元素 | input D | ScalarCompute/CopyIn | >0 |
| Gather | `hInput` | int64/64 | off 8 | 元素 | input H | 地址 stride | >0 |
| Gather | `wInput` | int64/64 | off 16 | 元素 | input W | CopyIn/index restore | >0 |
| Gather | `dOutput` | int64/64 | off 24 | 元素 | output D | outer/tail与GM offset | >0 |
| Gather | `hOutput` | int64/64 | off 32 | 元素 | output H | 同上 | >0 |
| Gather | `wOutput` | int64/64 | off 40 | 元素 | output W | 同上 | >0 |
| Gather | `dKernel` | int64/64 | off 48 | 元素 | kD | ProcessW | dilation=1 |
| Gather | `hKernel` | int64/64 | off 56 | 元素 | kH | ProcessW | dilation=1 |
| Gather | `wKernel` | int64/64 | off 64 | 元素 | kW | ProcessW | `bytes<128` |
| Gather | `dStride` | int64/64 | off 72 | 元素 | sD | input window/offset | >0 |
| Gather | `hStride` | int64/64 | off 80 | 元素 | sH | 同上 | >0 |
| Gather | `wStride` | int64/64 | off 88 | 元素 | sW | gather index | >0 |
| Gather | `padFront` | int64/64 | off 96 | 元素 | pD | padding/index base | >=0 |
| Gather | `padLeft` | int64/64 | off 104 | 元素 | pW | padding/index base | >=0 |
| Gather | `padTop` | int64/64 | off 112 | 元素 | pH | padding/index base | >=0 |
| Gather | `highAxisInner` | int64/64 | off 120 | N*C 元素 | UB search | ScalarCompute | tail field配对 |
| Gather | `highAxisTail` | int64/64 | off 128 | 元素 | DoUBTiling | tail tile | 整除时=inner |
| Gather | `highAxisOuter` | int64/64 | off 136 | 次数 | DoUBTiling | block decode | >=1 |
| Gather | `dOutputInner` | int64/64 | off 144 | 元素 | UB search | ScalarCompute | >=1 |
| Gather | `dOutputTail` | int64/64 | off 152 | 元素 | DoUBTiling | tail | 同上 |
| Gather | `dOutputOuter` | int64/64 | off 160 | 次数 | DoUBTiling | block decode | >=1 |
| Gather | `hOutputInner` | int64/64 | off 168 | 元素 | UB search | ScalarCompute | >=1 |
| Gather | `hOutputTail` | int64/64 | off 176 | 元素 | DoUBTiling | tail | >=1 |
| Gather | `hOutputOuter` | int64/64 | off 184 | 次数 | DoUBTiling | block decode | >=1 |
| Gather | `wOutputInner` | int64/64 | off 192 | 元素 | UB search | Vector batching | >=1 |
| Gather | `wOutputTail` | int64/64 | off 200 | 元素 | DoUBTiling | mask/tail | >=1 |
| Gather | `wOutputOuter` | int64/64 | off 208 | 次数 | DoUBTiling | block decode | >=1 |
| Gather | `normalCoreProcessNum` | int64/64 | off 216 | blocks/core | DoBlockTiling | Process loop | >=1 |
| Gather | `tailCoreProcessNum` | int64/64 | off 224 | blocks | DoBlockTiling | last core | >=1 |
| Gather | `usedCoreNum` | int64/64 | off 232 | core | DoBlockTiling | Init guard | blockDim同值 |
| Gather | `inputBufferSize` | int64/64 | off 240 | 字节/槽 | DoBufferCalculate | InitBuffer | block 对齐 |
| Gather | `maxValueBufferSize` | int64/64 | off 248 | 字节/槽 | DoBufferCalculate | InitBuffer | block 对齐 |
| Gather | `argmaxBufferSize` | int64/64 | off 256 | 字节/槽 | DoBufferCalculate | InitBuffer | index dtype决定 |
| Gather | `isPad` | int64/64 | off 264 | bool-like | InitializationVars | 仅通过模板参数体现 | key必须一致 |
| Gather | `dDilation` | int64/64 | off 272 | 元素 | attrs | ProcessW | 此 Route=1 |
| Gather | `hDilation` | int64/64 | off 280 | 元素 | attrs | ProcessW | 此 Route=1 |
| Gather | `wDilation` | int64/64 | off 288 | 元素 | attrs | ProcessW | 此 Route=1 |
| BigKernel | `dInDim` | int64/64 | off 0 | 元素 | input D | Init/地址式 | >0 |
| BigKernel | `hInDim` | int64/64 | off 8 | 元素 | input H | Init/地址式 | >0 |
| BigKernel | `wInDim` | int64/64 | off 16 | 元素 | input W | Init/地址式 | >0 |
| BigKernel | `dOutDim` | int64/64 | off 24 | 元素 | output D | 输出反解 | >0 |
| BigKernel | `hOutDim` | int64/64 | off 32 | 元素 | output H | 输出反解 | >0 |
| BigKernel | `wOutDim` | int64/64 | off 40 | 元素 | output W | 输出反解 | >0 |
| BigKernel | `kD` | int64/64 | off 48 | 元素 | kernel D | 窗口/分段 | >0 |
| BigKernel | `kH` | int64/64 | off 56 | 元素 | kernel H | 窗口/分段 | >0 |
| BigKernel | `kW` | int64/64 | off 64 | 元素 | kernel W | 窗口/分段 | bytes>=128 |
| BigKernel | `sD` | int64/64 | off 72 | 元素 | stride D | origin | >0 |
| BigKernel | `sH` | int64/64 | off 80 | 元素 | stride H | origin | >0 |
| BigKernel | `sW` | int64/64 | off 88 | 元素 | stride W | origin | >0 |
| BigKernel | `pD` | int64/64 | off 96 | 元素 | pad D | clip | >=0 |
| BigKernel | `pH` | int64/64 | off 104 | 元素 | pad H | clip | >=0 |
| BigKernel | `pW` | int64/64 | off 112 | 元素 | pad W | clip | >=0 |
| BigKernel | `dD` | int64/64 | off 120 | 元素 | dilation D | Route证据；body连续搬运 | 必须=1 |
| BigKernel | `dH` | int64/64 | off 128 | 元素 | dilation H | Route证据；body连续搬运 | 必须=1 |
| BigKernel | `dW` | int64/64 | off 136 | 元素 | dilation W | Route证据；body连续搬运 | 必须=1 |
| BigKernel | `blockFactor` | int64/64 | off 144 | 输出点/core | DoUBTiling | Init | former/tail |
| BigKernel | `blockTail` | int64/64 | off 152 | core | DoUBTiling | Init | 前若干核多1 |
| BigKernel | `totalIdx` | int64/64 | off 160 | 输出点 | DoUBTiling | 诊断/边界 | N*C*Dout*Hout*Wout |
| BigKernel | `coreNums` | int64/64 | off 168 | core | DoUBTiling | blockDim依据 | <=AIV core |
| BigKernel | `maxCount` | int64/64 | off 176 | 元素/槽 | UB式 | Init/split | >1024 |
| BigKernel | `isSigOut` | int64/64 | off 184 | bool-like | all output DHW==1 | CalcRealKernelSize | 0/1 |
| SIMT | `threadNums` | int64/64 | off 0 | thread | `min(count,256)` | 固定仓 Device 不读取 | 记录字段，实际 dim3=256 |
| SIMT | `blockNums` | int64/64 | off 8 | block | Host blockNum | Device `blockNum_` 来自 GetBlockNum | 应等于 blockDim |
| SIMT | `nDim` | int64/64 | off 16 | 元素 | normalized N | Compute/address | >0 |
| SIMT | `cDim` | int64/64 | off 24 | 元素 | normalized C | Compute/address | >0 |
| SIMT | `dInDim` | int64/64 | off 32 | 元素 | input D | Compute/address | >0 |
| SIMT | `hInDim` | int64/64 | off 40 | 元素 | input H | Compute/address | >0 |
| SIMT | `wInDim` | int64/64 | off 48 | 元素 | input W | Compute/address | layout 由模板决定 |
| SIMT | `dOutDim` | int64/64 | off 56 | 元素 | output D | totalSize/反解 | >0 |
| SIMT | `hOutDim` | int64/64 | off 64 | 元素 | output H | totalSize/反解 | >0 |
| SIMT | `wOutDim` | int64/64 | off 72 | 元素 | output W | totalSize/反解 | >0 |
| SIMT | `kSizeD` | int64/64 | off 80 | 元素 | kernel D | 三重循环 | >0 |
| SIMT | `kSizeH` | int64/64 | off 88 | 元素 | kernel H | 三重循环 | >0 |
| SIMT | `kSizeW` | int64/64 | off 96 | 元素 | kernel W | 三重循环 | >0 |
| SIMT | `stridesD` | int64/64 | off 104 | 元素 | stride D | window start | >0 |
| SIMT | `stridesH` | int64/64 | off 112 | 元素 | stride H | window start | >0 |
| SIMT | `stridesW` | int64/64 | off 120 | 元素 | stride W | window start | >0 |
| SIMT | `padD` | int64/64 | off 128 | 元素 | pad D | clipping | <=kernel/2 |
| SIMT | `padH` | int64/64 | off 136 | 元素 | pad H | clipping | <=kernel/2 |
| SIMT | `padW` | int64/64 | off 144 | 元素 | pad W | clipping | <=kernel/2 |
| SIMT | `dilationD` | int64/64 | off 152 | 元素 | dilation D | window loop step | >0 |
| SIMT | `dilationH` | int64/64 | off 160 | 元素 | dilation H | window loop step | >0 |
| SIMT | `dilationW` | int64/64 | off 168 | 元素 | dilation W | window loop step | >0 |
| SIMT | `ceilMode` | int64/64 | off 176 | bool-like | attrs | 固定仓 Device 不读取 | output shape已由L0决定 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/2 | 400001/2 | `...gather_tiling.cpp` | Gather POD | entry key 400001/2 | `arch35/...gather.h` | ascend950 binary + module CMake |
| R-003 | 611110 | `...tiling_big_kernel_regbase.cpp` | BigKernel POD | entry key 611110 | `arch35/...big_kernel_regbase.h` | 同上 |
| R-004～7 | 600001/2/11/12 | `...simt_tiling.cpp` | SIMT POD | entry four SIMT keys | `arch35/...simt.h` | 同上 |

## Kernel 实现

### 公共执行流程

entry 在 950 编译 guard 下声明 `KERNEL_TYPE_AIV_ONLY`，依 key 构造 Gather/RegBase AIV 或 `asc_vf_call` SIMT body。所有 Route 均直接读取 `x`、写 `y/indices`，没有跨 Kernel workspace 边、Atomic 或跨核共享；每个输出点只归一个 core/thread。未知 key 无 body，Host priority 2 对合法 format 总能生成四个 SIMT key之一。

### Route R-001：Gather 无 padding

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/...gather_tiling.cpp` | `IsCapable/GetTilingKey` | NCDHW、dilation=1、W bytes<128、UB rate gate、isPad=0 | 400001 |
| Router | `op_kernel/max_pool3d_with_argmax_v2.cpp` | key 400001 | `PAD_DISABLE` | Gather body |

#### Device 数据流

##### 函数级调用链

```text
entry -> GatherKernel<PAD_DISABLE>::Init -> Process
  -> ScalarCompute -> CopyIn -> Compute
  -> SingleRowGather/MultiRowGather/MultiDepGather/MultiNcGather
  -> ProcessW(DataCopyGather+Compare+Select+Max) -> ConvertIndex -> CopyOut
```

##### 源码等价伪码

```text
for assigned baseBlock: decode NC/D/H/W outer and real tail
DMA rectangular input tile to depth2 input Queue
choose register batching level by output tile volume vs index VL
for each kd,kh,kw: gather lanes; update on (value>max) OR isnan(value)
convert aligned UB index to original d*H*W+h*W+w
DMA contiguous real output tile to GM y and argmax
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input Queue | decoded input offset | `2*inputBufferSize` | T1/NCDHW | CopyIn | Compute | depth2 | W block aligned |
| max Queue | output tile | `2*maxValueBufferSize` | T1 | ProcessW | CopyOut | depth2 | real extent写出 |
| argmax Queue | output tile | `2*argmaxBufferSize` | T2 | ConvertIndex | CopyOut | depth2 | INT32/64 |
| registers | lane gather indices/values/masks | VReg | mixed | ProcessW | Select/Max | per VEC scope | tail mask=`repeatElem` |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["inputQue d=2 · 2I"]:1
  maxQ["maxValueQue d=2 · 2M"]:1
  argQ["argmaxQue d=2 · 2A"]:1
```

##### 字节账与复用顺序

```text
actual_UB = 2I + 2M + 2A
Host_budget = actual_UB + 2048 <= availableUb
workspace_total(core-owned)=0; dynamic_ub=N/A
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slots | CopyIn/Compute | Compute/CopyOut | pipeline next tile | EnQue/DeQue/FreeTensor | depth2 生命周期 |
| register max/index | ProcessW | UB store | next batch | VEC scope | mask excludes tail lane |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→compute | Queue | CopyIn | DeQue | input tile | dtype 原精度 |
| gather reduce | register masks | candidate | Select/Max | max+index | NaN 强制更新；顺序使后遇 NaN 获胜 |
| tail | UpdateMask/DataCopyUnAlign | scalar tail fields | store/CopyOut | W/更高轴 tail | 仅 real extent |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/max_pool3d_with_argmax_v2_gather_tiling.cpp` | `SearchBestTiling/DoBufferCalculate` | factor、UB、core |
| Kernel entry/router | `op_kernel/max_pool3d_with_argmax_v2.cpp` | 400001 | PAD_DISABLE |
| Device body/helper | `op_kernel/arch35/max_pool3d_with_argmax_v2_gather.h` | `ProcessW/*Gather` | gather、最大值、索引、tail |

### Route R-002：Gather 显式负无穷 padding

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `...gather_tiling.cpp` | `InitializationVars/GetTilingKey` | Gather capable 且 `isPad=1` | 400002 |
| Router | `...max_pool3d_with_argmax_v2.cpp` | 400002 | PAD_ENABLE | pad Gather |

#### Device 数据流

##### 函数级调用链

```text
entry -> Gather<PAD_ENABLE>::Init -> CopyIn(valid rectangle)
  -> DupAndCopyToCalcBuffer(-inf fill + valid placement)
  -> same four register Gather families -> CopyOut
```

##### 源码等价伪码

```text
compute per-side overlap with input
DMA only valid rows into input Queue
fill calc TBuf full padded tile with -inf
copy valid subvolume at front/top/left offsets
run same gather; clamp converted negative padded coordinates to zero only for index arithmetic
write max/index for real outputs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input Queue | valid clipped GM rectangle | `2I` | T1 | DataCopyPad | pad copy | depth2 | no invalid GM read |
| calc TBuf | padded tile origin | `I` | T1 | `DupBufferNegInf` + CopyToCalc | Gather | per Kernel | invalid region=-inf |
| output Queues | 同 R-001 | `2M+2A` | T1/T2 | Compute | CopyOut | depth2 | 同 R-001 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inputQue d=2 · 2I"]:1
  calc["pad calc TBuf · I"]:1
  maxQ["maxQueue d=2 · 2M"]:1
  argQ["argmaxQueue d=2 · 2A"]:1
```

##### 字节账与复用顺序

```text
actual_UB = 3I + 2M + 2A
Host_budget = actual_UB + 2048 <= availableUb
workspace_total(core-owned)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| calc TBuf | -inf fill + valid copy | Gather | next tile | Vector scope then Queue order | valid copy完成后计算 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| padding | Vector DataCopy + LocalMem order | Dup/Copy | Gather | calc TBuf | -inf 不应胜过有效有限值 |
| index | Compare LT + Select zero | converted coordinate | linear index | padding lane | max 若来自有效值则输出真实坐标 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...gather_tiling.cpp` | `InitializationVars/DoBufferCalculate` | isPad 与额外 I 预算 |
| Kernel entry/router | `...max_pool3d_with_argmax_v2.cpp` | 400002 | PAD_ENABLE |
| Device body/helper | `...arch35/max_pool3d_with_argmax_v2_gather.h` | `DupAndCopyToCalcBuffer/ConvertIndexWithoutPadAlign` | -inf padding与索引还原 |

### Route R-003：RegBase 大宽窗分段归约

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `...tiling_big_kernel_regbase.cpp` | `IsCapable` | NCDHW、dilation1、maxCount>1024、kW*dtypeSize>=128 | 611110 |
| Router | `...max_pool3d_with_argmax_v2.cpp` | 611110 | `BigKernelRegbase<DTYPE_X,float,DTYPE_ARGMAX>` | AIV body |

#### Device 数据流

##### 函数级调用链

```text
entry -> BigKernelRegbase::Init -> Process -> BaseCompute
  -> CalcRealKernelSize
  -> NoSplitKernelProcess or SplitKernelProcess(D then H then W)
  -> CopyInCube[名称如此，实际为 AIV MultiCopy DataCopy]
  -> ComputeSingleBlock -> CalcRealIndex/MergeMaxAndIndex -> CopyOut
```

##### 源码等价伪码

```text
partition output points among cores
for output:
  clip real window
  if volume<=maxCount: one multi-dimensional DMA
  else split D if H*W fits, else split H if W fits, else split W
  vector reduce each segment; update on greater value or NaN
  merge segment result and map compact index back through input H/W
  batch up to 1024 results in output TBuf; V_MTE3 then GM copy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input Queue | clipped window/chunk | `2*maxCount*sizeof(T1)` | T1 compact DHW | MultiCopy DataCopy | ComputeSingleBlock | depth2 | D/H/W strides明确 |
| max output TBuf | localCurIdx | `1024*sizeof(T1)` | T1 | reduction | CopyOut | chunks of 1024 outputs | MTE3 后复用 |
| index output TBuf | localCurIdx | `1024*sizeof(TINDEX)` | INT32/64 | index restore | CopyOut | 同上 | worst 8 B reserved |
| registers | reduce value/index/masks | VReg | float/int | ComputeSingleBlock | StoreOneNum | per segment | padded lane=-inf,index=-1 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inputQue d=2 · 2×maxCount×sizeof(T1)"]:2
  max["max output · 1024×sizeof(T1)"]:1
  idx["index output · 1024×sizeof(TINDEX)"]:1
```

##### 字节账与复用顺序

```text
maxCount = floor_align((UB-12KiB)/2, vRegSize)/sizeof(T1)
actual_UB = 2*maxCount*sizeof(T1) + 1024*sizeof(T1) + 1024*sizeof(TINDEX) <= UB
workspace_total(core-owned)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| output TBuf batch | StoreOneNum | CopyOut | localCurIdx 回到0 | MTE3_V/V_MTE3 events | GM batch copy完成 |
| input slot | MultiCopy | Compute | next segment | Queue FreeTensor | segment归约完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| output reuse | `MTE3_V`/`V_MTE3` | GM copy/Vector store | Init/CopyOut | 1024 TBuf | partial batch按 localCurIdx+1 字节数 |
| segment merge | `MergeMaxAndIndex` + LocalMemBar | segment result | next segment | max/index | NaN与greater mask |
| dtype | MicroAPI load/cast helpers | T1 | float T2 reduce | value | FP16/BF16升至float；输出转T1 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling_big_kernel_regbase.cpp` | `IsCapable/DoUBTiling` | maxCount/分核/key |
| Kernel entry/router | `...max_pool3d_with_argmax_v2.cpp` | 611110 | dtype模板 |
| Device body/helper | `...arch35/max_pool3d_with_argmax_v2_big_kernel_regbase.h` | `SplitKernelProcess/ComputeSingleBlock` | 分段、MultiCopy、归约、输出批处理 |

### Route R-004：SIMT NCDHW 32 位处理索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `...simt_tiling.cpp` | `GetTilingKey` | NCDHW且count<=INT32_MAX | 600001 |
| Router | `...max_pool3d_with_argmax_v2.cpp` | 600001 | `<NCDHW,false>` | SIMT NCDHW |

#### Device 数据流

##### 函数级调用链

```text
entry -> Simt<NCDHW,false>::Compute -> asc_vf_call<MaxPool3DNcdhw<...,int32_t,uint32_t>>(256 threads)
```

##### 源码等价伪码

```text
for index=blockIdx*256+threadIdx; index<count; index+=blockNum*256:
  magic-div index -> pw,ph,pd,nxc
  clip dilated window; max=-inf
  scan d/h/w; update when val>max OR isnan(val)
  y[index]=max; indices[index]=d*H*W+h*W+w
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `nxc*DHW + idxOffset` | scalar per sample | T/NCDHW | thread loads | scalar max | caller | direct GM |
| registers | max/index/start/end/magic | per thread | T/int32/uint32 | thread | GM store | thread | 无共享/UB |

##### 分层内存布局

N/A：SIMT 直接 GM 标量访问，仅用线程寄存器；未分配 AIV Queue/TBuf 或 dynamic UB。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=0; workspace_total(core-owned)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread registers | 当前输出迭代 | GM store | grid-stride下一 index | 单线程程序序 | index<count |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread loop | 无 barrier/atomic | 单线程 | 同线程 store | 唯一输出点 | `index<count` tail；原 dtype比较 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...simt_tiling.cpp` | `DoOpTiling/GetTilingKey` | grid与32位key |
| Kernel entry/router | `...max_pool3d_with_argmax_v2.cpp` | 600001 | NCDHW,false |
| Device body/helper | `...arch35/max_pool3d_with_argmax_v2_simt.h` | `MaxPool3DNcdhw` | 线程映射、窗口循环、tail |

### Route R-005：SIMT NDHWC 32 位处理索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `...simt_tiling.cpp` | `GetTilingKey` | NDHWC且count<=INT32_MAX | 600002 |
| Router | `...max_pool3d_with_argmax_v2.cpp` | 600002 | `<NDHWC,false>` | SIMT NDHWC |

#### Device 数据流

##### 函数级调用链

```text
entry -> Simt<NDHWC,false>::Compute -> asc_vf_call<MaxPool3DNdhwc<...,int32_t,uint32_t>>
```

##### 源码等价伪码

```text
magic-div linear output -> c,pw,ph,pd,n
base=x+n*D*H*W*C+c; sample x[base+(d*H*W+h*W+w)*C]
same clipped max/NaN loop; output linear GM is NDHWC
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `(n*DHW+dhw)*C+c` | scalar/sample | T/NDHWC | thread | compare | caller | channel stride |
| register | 同 R-004 + c/n | per thread | mixed | thread | store | thread | 无共享 |

##### 分层内存布局

N/A：SIMT GM direct + registers。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=0; workspace_total(core-owned)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| registers | 当前 index | GM store | 下一 index | thread order | count guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | thread | thread | unique output | channel参与地址但不进入indices |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...simt_tiling.cpp` | `GetShapeAttrsInfo/GetTilingKey` | NDHWC轴位与key |
| Kernel entry/router | `...max_pool3d_with_argmax_v2.cpp` | 600002 | NDHWC,false |
| Device body/helper | `...arch35/max_pool3d_with_argmax_v2_simt.h` | `MaxPool3DNdhwc` | channel-stride地址 |

### Route R-006：SIMT NCDHW 64 位处理索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `...simt_tiling.cpp` | `GetTilingKey` | NCDHW且count>INT32_MAX | 600011 |
| Router | `...max_pool3d_with_argmax_v2.cpp` | 600011 | `<NCDHW,true>` | uint64 processing |

#### Device 数据流

##### 函数级调用链

```text
entry -> Simt<NCDHW,true>::Compute -> MaxPool3DNcdhw<...,int64_t,uint64_t>
```

##### 源码等价伪码

与 R-004 相同，但 grid-stride index、magic division、坐标/窗口 index 使用 uint64/int64，避免输出线性域超过 INT32 时的窄化。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/register | 同 R-004 | scalar | T + 64位处理索引 | thread | store | thread | 无 UB |

##### 分层内存布局

N/A：SIMT direct GM。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=0; workspace_total(core-owned)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| registers | thread | GM | next index | program order | 64位算术 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 64位 grid | uint64 magic division | index | coordinate decode | 大输出域 | index<count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...simt_tiling.cpp` | `GetTilingKey` | count阈值 |
| Kernel entry/router | `...max_pool3d_with_argmax_v2.cpp` | 600011 | NCDHW,true |
| Device body/helper | `...simt.h` | `Compute/MaxPool3DNcdhw<int64_t,uint64_t>` | 64位地址反解 |

### Route R-007：SIMT NDHWC 64 位处理索引

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `...simt_tiling.cpp` | `GetTilingKey` | NDHWC且count>INT32_MAX | 600012 |
| Router | `...max_pool3d_with_argmax_v2.cpp` | 600012 | `<NDHWC,true>` | uint64 NDHWC |

#### Device 数据流

##### 函数级调用链

```text
entry -> Simt<NDHWC,true>::Compute -> MaxPool3DNdhwc<...,int64_t,uint64_t>
```

##### 源码等价伪码

R-005 的 channel-last 地址式与 R-006 的 64 位输出反解组合；每线程仍独占一个输出点并用 `index<count` 保护尾线程。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/register | NDHWC channel stride | scalar | T + 64位处理索引 | thread | store | thread | 无 UB |

##### 分层内存布局

N/A：SIMT direct GM。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=0; workspace_total(core-owned)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| registers | thread | GM | next index | program order | 64位算术与count guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 无 barrier/atomic | thread | thread | unique NDHWC output | 原 dtype比较，NaN后遇优先 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...simt_tiling.cpp` | `GetShapeAttrsInfo/GetTilingKey` | NDHWC + 64位key |
| Kernel entry/router | `...max_pool3d_with_argmax_v2.cpp` | 600012 | NDHWC,true |
| Device body/helper | `...simt.h` | `MaxPool3DNdhwc<int64_t,uint64_t>` | 64位channel-last地址 |

## 算子特定优化

### Trick T-001：小宽窗用寄存器 Gather，padding 仅物化有效 tile 的补齐副本

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：NCDHW、dilation=1、`wKernel*inputBytes<128` 且最小 tile 满足带稀疏率的 UB gate。
- Golden 来源：[`max_pool3d_with_argmax_v2_gather.h:516-525`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_gather.h#L516-L525)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
                MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
                MicroAPI::Compare<T1, CMPMODE::NE>(neMask, calcReg, calcReg, gatherMask);
                MicroAPI::MaskOr(gtMask, gtMask, neMask, gatherMask);
                if constexpr (sizeof(int32_t) / sizeof(T1) == 1) {
                    MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, gtMask);
                } else {
                    MicroAPI::MaskUnPack(tmpMask, gtMask);
                    MicroAPI::Select(maxIndexReg, indexWithOffset, maxIndexReg, tmpMask);
                }
                MicroAPI::Max(maxReg, maxReg, calcReg, gatherMask);
```

#### 机制与设计原因

每个寄存器批次先生成输出 lane 的基础地址，再对 kD/kH/kW 加相对 offset 做 Gather，最大值和索引同一 mask 更新，避免为窗口展开一个 UB 矩阵。padding Route 只多一个 I 大小 calc TBuf：先负无穷填充再嵌入有效输入，避免非法 GM 访问。

#### 迁移边界

- 适用：小连续尾维窗口、地址可向量生成的滑窗。
- 前提：Gather index 位宽、VReg lane、padding负无穷和 NaN 更新语义保持一致。
- 失效：W 窗口字节达到 128、dilation非1或UB rate gate失败时转 R-003/SIMT。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

padding 路线若直接从负地址/越界地址 Gather 会非法读 GM；源码明确物化负无穷 calc TBuf。

### Trick T-002：SIMT 用 magic division 反解线性输出并按规模切换 32/64 位

#### 触发与源码依据

- 适用 Route：R-004～R-007。
- 触发条件：前两 AIV 模板未命中；按 format 与 `outputCount` 阈值选 key。
- Golden 来源：[`max_pool3d_with_argmax_v2_simt.h:216-221`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_simt.h#L216-L221)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    if constexpr (Format_T == 0 && !useINT64Index) {
        uint32_t m_[3] = {1, 1, 1};
        uint32_t shift_[3] = {1, 1, 1};
        GetUintDivMagicAndShift(m_[DIM_0], shift_[DIM_0], static_cast<uint32_t>(outputWidth));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<uint32_t>(outputHeight));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<uint32_t>(outputDi));
```

#### 机制与设计原因

线程只持有一个输出点，无 UB staging；编译期 format 消除 NCDHW/NDHWC 地址分支，magic/shift 代替循环中的普通除法参数生成。小域保持 32 位处理，超过 INT32 才扩大到 64 位，避免无条件使用更宽算术。

#### 迁移边界

- 适用：每输出点独立、窗口扫描成本主导、可用线性输出域反解坐标的 fallback。
- 前提：Host key阈值与 Device `PROCESS_T/FASTDIV_T` 模板必须匹配。
- 失效：输出点需要跨线程/跨核归约时不能直接复用此无同步结构。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

对 `outputCount>INT32_MAX` 仍选 600001/600002 会在输出线性 index/坐标反解中窄化；源码提供 600011/600012 真实 fallback。

## 模板与实现对比

| 对比维度 | R-001/2 Gather | R-003 BigKernel | R-004～7 SIMT | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | priority0，小W、dilation1、NCDHW、UB gate | priority1，大W、dilation1、NCDHW | priority2 无条件 RegBase fallback | 小窗 Gather→大窗分段→通用线程 |
| 模板与 dtype | T1 + INT32/64 + pad模板 | T1,float,TINDEX | T1,TINDEX,format,处理位宽 | key与编译期特化 |
| 循环与分核 | base tile/core | 输出点/core + window segment | 输出点/thread grid stride | 并行粒度逐渐细化 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth2 Queue，pad多I TBuf | depth2 Queue + 2 TBuf | registers only | 无通信/dynamic UB |
| 内存总占用与复用 | Host搜索至UB内 | 由UB反推maxCount | N/A | 资源决定选择 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + DataCopyGather | MultiCopy + MicroAPI reduce | scalar GM + asc_vf_call | 均非Cube |
| 同步、精度与 tail | Queue/VEC，四轴tail | event/LocalMemBar，segment/core tail | 无barrier，index<count | 独占输出无需Atomic |
| fallback/失败边界 | 后续priority | SIMT | tiling硬错误 | SIMT闭合合法输入 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | DAV_2201，dilation1、NCDHW、INT32 | 同左 | RegBase，NCDHW/NDHWC、INT32/64、dilation>0 |
| Kernel/资源 | NoExpand/BigKernel AIV | 同左 | Gather/BigKernelRegBase AIV + SIMT fallback |

## 限制与失败边界

- 950 接受 INT64 indices、NDHWC、dilation>1；这些能力不应回迁到 A2/A3 卡。
- `threadNums` 与 `ceilMode` 写入 SIMT POD，但固定 Device body不读取：实际线程维固定 256，输出 shape 已在 L0 决定。
- Gather Host 固定把 `coreUsedForBestPerformance` 覆盖为 64；这只是搜索目标，最终 `usedCoreNum` 仍受实际外块数和平台 core 数计算。
- BigKernel helper 名为 `CopyInCube`，但源码实际调用 `DataCopy<T1,THREE>` 且 entry声明 AIV-only；这不是 Cube/AIC 证据。
- 框架 Contiguous/ReFormat/ViewCopy 的内部 Kernel 与 workspace slice 在本固定算子源码中不可见；核心算子自身不消费 user workspace。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `pooling/max_pool3d_with_argmax_v2/docs/aclnnMaxPool3dWithArgmax.md` | 产品支持表 | 950公开能力 |
| 入口实现 | `pooling/max_pool3d_with_argmax_v2/op_api/aclnn_max_pool3d_with_argmax.cpp` | `aclnnMaxPool3dWithArgmax*` | 校验、前后处理 |
| 入口实现 | `pooling/max_pool3d_with_argmax_v2/op_api/max_pool3d_with_argmax_v2.cpp` | `MaxPool3DWithArgmaxV2NcdhwAiCore` | dtype属性/launcher |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `DoTilingImpl` | priority语义 |
| Host/Tiling/Launcher | `pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_tiling_struct.h` | 三个 POD | ABI |
| Host/Tiling/Launcher | `pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_gather_tiling.cpp` | `GatherTiling` | R-001/2 |
| Host/Tiling/Launcher | `pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_tiling_big_kernel_regbase.cpp` | `BigKernelRegbaseTiling` | R-003 |
| Host/Tiling/Launcher | `pooling/max_pool3d_with_argmax_v2/op_host/max_pool3d_with_argmax_v2_simt_tiling.cpp` | `TilingSIMT` | R-004～7 |
| Kernel entry/body | `pooling/max_pool3d_with_argmax_v2/op_kernel/max_pool3d_with_argmax_v2.cpp` | entry/router | 950所有key |
| Kernel entry/body | `pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_gather.h` | GatherKernel | R-001/2 body |
| Kernel entry/body | `pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_big_kernel_regbase.h` | BigKernelRegbase | R-003 body |
| Kernel entry/body | `pooling/max_pool3d_with_argmax_v2/op_kernel/arch35/max_pool3d_with_argmax_v2_simt.h` | MaxPool3DNcdhw/Ndhwc | R-004～7 body |
| 构建接线 | `pooling/max_pool3d_with_argmax_v2/op_host/config/ascend950/max_pool3d_with_argmax_v2_binary.json` | Op list | 950 dtype/layout组合 |
| 构建接线 | `pooling/max_pool3d_with_argmax_v2/CMakeLists.txt` | `add_modules_sources` | module接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三个纯int64 POD与逐字段producer | 字段顺序、8B offset、inactive字段 | tiling_struct + SetTilingData |
| 调用顺序 | API框架节点→单核心Kernel→ViewCopy | executor stream与失败传播 | phase1/phase2入口 |
| workspace/内存 | Gather/BigKernel字节式，SIMT无UB | Queue depth、pad额外I、12KiB保留 | DoBufferCalculate/InitBuffer |
| 同步 | Gather Queue、BigKernel event、SIMT线程独占 | producer/consumer方向，SIMT无Atomic | Device各 body |
| 精度与尾块 | Gather原dtype、BigKernel float处理、SIMT原dtype | NaN后遇更新、-inf pad、处理位宽key | ProcessW/ComputeSingleBlock/CycleUpdate |
| 失败边界 | priority0→1→2 | format/dilation/W字节/UB/outputCount谓词 | 三个 IsCapable/GetTilingKey |