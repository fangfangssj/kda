---
type: Operator
title: "ScaledMaskedSoftmax A2/A3 ACLNN 算子知识"
description: "说明 aclnnScaledMaskedSoftmax 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/docs/aclnnScaledMaskedSoftmax.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_host/op_api/aclnn_scaled_masked_softmax.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_host/config/ascend910b/scaled_masked_softmax_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_host/config/ascend910_93/scaled_masked_softmax_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:40:32Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ScaledMaskedSoftmax A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 API `aclnnScaledMaskedSoftmaxGetWorkspaceSize` / `aclnnScaledMaskedSoftmax`；内部算子为 `ScaledMaskedSoftmaxV2`，固定仓未出现更高公开版本。 |
| 计算语义 | 对每个 `[B,N,S1]` 行先缩放 `x`，由布尔 `mask` 在缩放值与 `-10000` 间选择，再沿末轴执行 Softmax。公开文档写作 `Softmax((scale*x)*mask)`；固定仓未包含 `SelectWithBytesMask` primitive 的布尔选边定义，因此不推测 true/false 分别选择哪一侧。 |
| 输入输出 | `x/y`: `[B,N,S1,S2]`、ND、FP32/FP16/BF16；`mask`: BOOL、ND，末两维等于 `x`，前两维分别相等或为 1。A2/A3 要求 `0 < S2 <= 4096`。 |
| 执行类型 | 单 Kernel：`scaled_masked_softmax_v2`，执行族为 `AIV_VECTOR`；无可达 Cube/AIC、MIX_CV、SIMT、NDDMA 或通信节点。 |
| Kernel 路线 | R-001 FP32（key 0）、R-002 FP16（key 1）、R-003 BF16（key 2）。 |
| 核心资源 | GM `x/mask/y`；depth=1 的 `inQueueX/inQueueMask/outQueueY`；FP32 中间 `bufX`；32 KiB/条件 64 KiB 的 `bufShare`；系统 workspace 仅为 `GetLibApiWorkSpaceSize()`。 |
| 来源闭包 | `source_closed`：公开 ACLNN 校验与两段式入口，经生成式 inner bridge 框架边界到 `ScaledMaskedSoftmaxV2` OpDef/Tiling/POD，再到单 AIV Kernel router、三种模板 body、binary config 与构建接线。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令一行的末轴长度为 `W=S2`，`m_i` 为 BOOL mask。公开文档给出的公式为 `Softmax((scale*x)*mask)`；Device body 可进一步证明它先缩放，再由 `SelectWithBytesMask` 在缩放值和 `-10000` 间选择，最后做 Softmax：

```text
z_i = float(scale) * x_i
u_i = SelectWithBytesMask(z_i, -10000.0, m_i)
y_i = exp(u_i - max_j(u_j)) / sum_j exp(u_j - max_j(u_j))
```

固定仓没有该公共 primitive 的内部定义，故本卡不补写 `m_i=true/false` 到两侧输入的映射。

Softmax 的有效列数是 `width=W`，UB 行距是 `padLineNum=A`；padding lane 只用于对齐，`SoftMaxShapeInfo` 以 `srcM=W` 排除它们。FP16 在缩放和 mask 后升为 FP32，BF16 在缩放前升为 FP32，二者均以 FP32 做 Softmax 并用 `CAST_RINT` 降回输出；FP32 全程为 FP32。`mask` 可在 batch/channel 维广播，但 height/width 必须完全相等。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[B,N,S1,S2]` / ND | FP32、FP16、BF16 | 必须为 4 维；A2/A3 上 `0 < S2 <= 4096`；允许非连续输入，OpDef `AutoContiguous()` 形成框架连续化边界。 |
| `mask` | `[Mb,Mn,S1,S2]` / ND | BOOL | 必须为 4 维；`Mb==B \|\| Mb==1`，`Mn==N \|\| Mn==1`；末两维与 `x` 相等；公开 API 路径要求非空，允许非连续并由 `AutoContiguous()` 处理。 |
| `scale` | 标量 | API 为 `double`；内部 attr/POD 为 FP32 | 公开入口把它交给生成式 inner bridge；固定仓可见的 OpDef attr 是 `Float`，Tiling 用 `GetAttrPointer<float>`，因此 Device 实际消费 FP32 `scale`。 |
| `fixTriuMask` | 标量 | BOOL | 仅支持 `false`；`true` 在公开第一段接口直接返回 `ACLNN_ERR_PARAM_INVALID`，不生成核内上三角 mask。 |
| `y` | 与 `x` 相同 / ND | 与 `x` 相同 | 公开实现检查 dtype 相同；文档要求 shape 相同。输出自身不支持非连续。 |

### 入口与版本

公开版本族只有无后缀 `aclnnScaledMaskedSoftmax`。第一段接口先执行 `CheckParams` 和 `fixTriuMask` 检查，再调用生成式框架符号 `aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize(..., false, ...)`；第二段直接转发到 `aclnnInnerScaledMaskedSoftmaxV2`。inner bridge 源文件不在固定仓，按生成式 ACLNN executor 框架边界处理；算子拥有的 OpDef、Tiling、POD、Kernel 与构建接线均可见。

公开 API 文档产品表明确列出 Atlas A2 与 A3；OpDef 的 `ascend910b`、`ascend910_93` 配置、两个 binary config 和 Kernel CMake 同时闭合这两个平台。内部名的 `V2` 不是另一个公开数学语义版本。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口按以下顺序执行：

1. `CheckNotNull` 检查 `x/mask/y`；尽管 OpDef 将 `mask` 声明为 OPTIONAL，该公开 API 路径无条件要求非空。
2. `CheckDtypeValid` 要求 `x/y` 属于 FP32、FP16、BF16，`mask` 为 BOOL，且 `x/y` dtype 相同。
3. `CheckTensorDim` 要求三者均为 4 维。
4. `CheckShape` 要求 mask 的 batch/channel 各自与 `x` 相等或为 1、height/width 相等，并在非 RegBase（A2/A3）上要求 `0 < width <= 4096`。
5. `fixedTriuMask==true` 直接失败；否则将常量 `false` 交给 inner bridge。
6. OpDef 对 `x/mask` 声明 `AutoContiguous()`；实际连续化的 executor 节点与临时 workspace 由生成式框架拥有，固定仓不可见。

`y` shape 与 `x` 相同在公开文档中是接口约束，`ScaledMaskedSoftmaxV2InferShape` 也令内部输出 shape 等于 `x`；公开 C++ `CheckParams` 本身没有显式比较 `x/y` shape，这一检查属于框架/inner bridge 边界。

### 参数、Workspace 与生命周期

`InitPlatformInfo` 取得 `AIV` 核数 `C=GetCoreNumAiv()`，将其同时写入 `blockDim` 与 POD `coreNum`。它把 `GetLibApiWorkSpaceSize()` 写入 `workspaceSizes[0]`；算子 Kernel 的 `workspace` 形参未读取，因此没有算子自有的节点间 slice。第一段 API 返回由 executor 汇总后的 workspace 大小；调用者持有 workspace，直到第二段在同一 executor/stream 上完成。

Tiling 从 storage shape 读取 `B,N,H,W` 与 mask shape，从 attr 读取 FP32 `scale`，生产对齐宽度、广播步长、分核/循环字段和 `SoftMaxTiling`。`SaveToBuffer` 序列化 POD。第二段 API 把调用者提供的 stream 交给 executor；固定仓未暴露生成式 launcher 的具体 launch 调用与错误码映射。

### 启动图与失败传播

```text
aclnnScaledMaskedSoftmaxGetWorkspaceSize
  -> CheckParams + fixedTriuMask guard
  -> aclnnInnerScaledMaskedSoftmaxV2GetWorkspaceSize  [生成式 executor 边界]
     -> ScaledMaskedSoftmaxV2 Tiling / POD / workspaceSize[0]

aclnnScaledMaskedSoftmax(workspace, workspaceSize, executor, stream)
  -> aclnnInnerScaledMaskedSoftmaxV2                   [生成式 launcher 边界]
     -> scaled_masked_softmax_v2<<<blockDim=C, stream>>>
        -> key 0/1/2 的 ScaledMaskedSoftmaxV2<T>::Init/Process
     -> GM y
```

唯一 Device 节点读取 GM `x/mask`、写 GM `y`；`workspace` 指针不被 body 消费。不存在 Kernel 间边。公开校验失败会在 executor 建立前返回；Tiling 的 `SetUbSplitInfo()` 失败返回 `GRAPH_FAILED`。需注意 `TilingScaledMaskedSoftmaxV2` 调用 `tilingObj.Init()` 后没有检查其返回值，随后仍调用 `DoTiling()`：这是固定源码可见的失败传播缺口，复用实现时不得把 `Init()` 失败假定为已被该 wrapper 阻断。

## Route 与参数设计

### 分核与资源计算

定义 `s=sizeof(T)`（FP32 为 4，FP16/BF16 为 2）、`A=ceil(W/(32/s))*(32/s)`、`M=ceil(W/32)*32`、`R=B*N*H`、`C=GetCoreNumAiv()`、`U=GetCoreMemSize(UB)`。A2/A3 的 Host 预留 Softmax 临时区 `S_host=32768` bytes，并计算：

```text
bytes_per_line = 2*A*s + A*4 + M
L = floor((U - 32768) / bytes_per_line)
q = floor(R / C)
h = R mod C
head rows = q + 1, for blockIdx < h
tail rows = q,     for blockIdx >= h
iter = ceil(rows / L)
last_rows = rows - (iter - 1)*L
```

`L==0` 或 `C==0` 时 `SetUbSplitInfo()` 失败。所有 block 都按最大 `L` 和 `A/M` 分配 UB；每核 GM 起点为：

```text
coreOffset = min(blockIdx,h)*(q+1)*W + max(blockIdx-h,0)*q*W
startLineOffset = coreOffset / W
```

当 `R<C` 时，`h=R`，后续 tail core 的 `q=0`、`iterTailCore=0`，其 `Process` 不进入循环；`lineLastTailIter` 的无符号表达式虽会形成无效值，但不被这些零循环 core 消费。`R==0` 也沿同一路径令全部 core 零循环，固定源码没有把空外层维判为错误。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 | key 0 / `ScaledMaskedSoftmaxV2<float>` | `xDataType` 既非 `DT_FLOAT16` 也非 `DT_BF16`；此前 `InitInputDtype` 只允许 FP32/FP16/BF16，所以可达时即 FP32。 | key 不是 0 时检查 key 1。 |
| 2 | R-002 FP16 | key 1 / `ScaledMaskedSoftmaxV2<half>` | `xDataType == DT_FLOAT16`。 | key 不是 1 时检查 key 2。 |
| 3 | R-003 BF16 | key 2 / `ScaledMaskedSoftmaxV2<bfloat16_t>` | `xDataType == DT_BF16`；A2/A3 编译不命中仅针对 `__NPU_ARCH__==3003/3113` 的排除宏。 | 其它 key 无 body、无输出。 |

```text
if x dtype == FP16: key = 1
else if x dtype == BF16: key = 2
else: key = 0  # 可达 dtype 仅剩 FP32

router: key 0 -> float; else key 1 -> half; else key 2 -> bfloat16_t; else -> no-op
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | 0 | A2/A3；`x/y=FP32`、`mask=BOOL`；4-D；广播合法；`0<W<=4096`；`fixTriuMask=false`；`C>0,L>0` | 单 `scaled_masked_softmax_v2` AIV_VECTOR 节点 | `ScaledMaskedSoftmaxV2<float>`；FP32 scale/select/Softmax，Softmax 直接写 output queue | depth=1 三队列；FP32 `bufX`；共享 tmp；逐迭代 queue 同步 | 行尾 DMA padding；Softmax 有效宽度 W；未知 key 无输出 |
| R-002 FP16 | 1 | 同上，`x/y=FP16` | 同一单节点 | `ScaledMaskedSoftmaxV2<half>`；FP16 Muls/Select→FP32 Cast→FP32 Softmax→RINT Cast | 队列容量按 2-byte T，FP32 `bufX` 原地 Softmax | 同上 |
| R-003 BF16 | 2 | 同上，`x/y=BF16` | 同一单节点 | `ScaledMaskedSoftmaxV2<bfloat16_t>`；先 Cast FP32→Muls/Select→FP32 Softmax→RINT Cast | 与 FP16 相同的队列字节数；FP32 中间 | 同上；A2/A3 不受 Kirin BF16 编译排除 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 末轴范围 | A2/A3：`0 < width <= 4096` | `DataCopyExtParams.blockLen`、Softmax 行宽及 UB 上界 | 全部 | API/Tiling 失败 | `CheckShape`；`InitInputShape` |
| dtype | `getDataTypeSize()!=0` 且公开 dtype 检查通过 | 决定 `s`、padding、key 与模板 | 全部 | API 或 Tiling 失败 | `CheckDtypeValid`；`InitInputDtype`；`SetTilingKey` |
| AIV 核数 | `GetCoreNumAiv()>0` | blockDim 与行分区必须非零 | 全部 | `InitPlatformInfo` 失败；wrapper 未显式传播该返回 | `InitPlatformInfo`；`TilingScaledMaskedSoftmaxV2` |
| 每迭代行数 | `(U-32768)/bytes_per_line > 0` | 三队列、FP32 中间和 Softmax tmp 必须容入 UB | 全部 | `SetUbSplitInfo` 返回 false，`DoTiling` 失败 | `SetUbSplitInfo` |
| X 对齐 | `A=align_up(W,32/s)` | X/Y DMA 每行有效 `W*s` bytes，UB 行距需 32-byte 对齐 | 全部 | 无独立 fallback；通过 padding 实现 | `SetPaddingInfo`；`CopyXIn`；`CopyOut` |
| mask 对齐 | `M=align_up(W,32)` | BOOL 每元素 1 byte，mask UB 行距按 32 byte 对齐 | 全部 | 无独立 fallback；通过 padding 实现 | `SetPaddingInfo`；`MoveMaskLines` |
| 广播合法性 | `(B==Mb\|\|Mb==1)&&(N==Mn\|\|Mn==1)&&H==Mh&&W==Mw` | mask GM offset 只支持前两维 0/1 step 广播 | 全部 | API/Tiling 失败 | `CheckShape`；`InitInputShape`；`SetMaskAdapt` |
| Softmax tmp | Host 固定预留 32 KiB；Device 在 `A*L>32768` 时分配 64 KiB | `SelectWithBytesMask` 与 `SoftMax` 共用 `bufShare` | 全部 | 固定仓无显式 fallback；若条件可达，Host 字节账未证明额外 32 KiB | `SetUbSplitInfo`；`ScaledMaskedSoftmaxV2::Init` |
| BF16 编译 guard | key 2 且架构不是 3003/3113 | 这两个 Kirin 架构不实例化 BF16 body | R-003 | A2/A3 可达；被排除平台无输出 body | `scaled_masked_softmax_v2` |
| tail core/iter | `blockIdx<h` 取 head，否则 tail；最后迭代用 `lineLast*Iter` | 避免越过最后有效行 | 全部 | 零行 tail core 的 loop 为 0 | `Init`；`Process` |
| DMA blockCount 窄化 | Device 将 `linePerIter`/分段 `lines` 转为 `uint16_t` | `DataCopyExtParams.blockCount` 字段为 16 位 | 全部 | 固定 Host 无显式 `L<=65535` guard 或 fallback | `CopyXIn`；`MoveMaskLines`；`CopyOut` |
| 空外层域 | `R=B*N*H==0` | 无有效行；所有 tail core 的 `iterTailCore=0` | 全部 | 成功产生零循环 Kernel；无输出写入 | `SetUbSplitInfo`；`Process` |

### POD/ABI

`ScaledMaskedSoftmaxV2TilingData` 由注册宏按下列声明顺序序列化；固定仓不包含 `tilingdata_base.h` 宏展开和 `SoftMaxTiling` 定义，因此整体 `sizeof/alignof` 与逐字段 `offsetof` 不可见，不能推测。除最后的嵌套结构外，字段源类型和位宽如下。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ScaledMaskedSoftmaxV2TilingData` | `coreNum` | `uint64_t`/64 | 声明序第 1；offset 不可见 | core | `InitPlatformInfo: GetCoreNumAiv()` | `Process: blockIdx>=coreNum` | 必须 >0 |
| 同上 | `batch` | `uint64_t`/64 | 第 2；offset 不可见 | 元素维度 | `SetInputShape: batch` | `Init: xBatch` | API 要求 4-D |
| 同上 | `channel` | `uint64_t`/64 | 第 3；offset 不可见 | 元素维度 | `SetInputShape: channel` | `Init: xChannel` | 同上 |
| 同上 | `height` | `uint64_t`/64 | 第 4；offset 不可见 | 行/通道 | `SetInputShape: height` | `Init: xHeight/linePerBatch` | 与 mask 相等 |
| 同上 | `width` | `uint64_t`/64 | 第 5；offset 不可见 | 元素/行 | `SetInputShape: width` | `Init/Copy/SoftmaxX` | `(0,4096]` |
| 同上 | `maskBatch` | `uint64_t`/64 | 第 6；offset 不可见 | 元素维度 | `SetInputShape: maskBatch` | 未直接读取；已折叠到 `nStep` | 相等或 1 |
| 同上 | `maskChannel` | `uint64_t`/64 | 第 7；offset 不可见 | 元素维度 | `SetInputShape: maskChannel` | `Init: maskBatchOffset` | 相等或 1 |
| 同上 | `maskHeight` | `uint64_t`/64 | 第 8；offset 不可见 | 行/通道 | `SetInputShape: height` | 未直接读取 | 必须等于 height |
| 同上 | `maskWidth` | `uint64_t`/64 | 第 9；offset 不可见 | 元素/行 | `SetInputShape: width` | `MoveMaskLines` 的 blockLen/offset | 必须等于 width |
| 同上 | `scale` | `float`/32 | 第 10；offset 不可见 | 数值标量 | `InitAttr: *GetAttrPointer<float>(0)` | `Init/Compute` | API double 经框架窄化到 FP32 |
| 同上 | `maskMode` | `uint64_t`/64 | 第 11；offset 不可见 | bitmask | `SetMaskAdapt`：batch 广播置 bit 1，channel 广播置 bit 2 | Device 未读取 | 仅诊断冗余字段 |
| 同上 | `paddingNum` | `uint64_t`/64 | 第 12；offset 不可见 | T 元素 | `SetPaddingInfo: A-W` | `CopyXIn` padding | `<32/s`，传入 uint8_t |
| 同上 | `padLineNum` | `uint64_t`/64 | 第 13；offset 不可见 | T/FP32 元素/行 | `SetPaddingInfo: A` | UB 容量、shape、elePerIter | 32-byte 对齐 |
| 同上 | `alignedMaskPadding` | `uint64_t`/64 | 第 14；offset 不可见 | BOOL 元素/bytes | `SetPaddingInfo: M-W` | `MoveMaskLines` padding | `<32`，传入 uint8_t |
| 同上 | `alignedMaskWidth` | `uint64_t`/64 | 第 15；offset 不可见 | BOOL 元素/bytes/行 | `SetPaddingInfo: M` | mask queue/shape/行距 | 32-byte 对齐 |
| 同上 | `nStep` | `uint64_t`/64 | 第 16；offset 不可见 | 0/1 | `SetMaskAdapt` | `Init: maskBatchOffset` | batch 广播为 0 |
| 同上 | `cStep` | `uint64_t`/64 | 第 17；offset 不可见 | 0/1 | `SetMaskAdapt` | `Init: maskChannelOffset` | channel 广播为 0 |
| 同上 | `headCoreNum` | `uint64_t`/64 | 第 18；offset 不可见 | core | `SetUbSplitInfo: R%C` | `Init/Process` | `[0,C)` |
| 同上 | `lineHeadCore` | `uint64_t`/64 | 第 19；offset 不可见 | 行/core | `SetUbSplitInfo: floor(R/C)+1` | `Init: coreOffset` | 仅 head core 有效 |
| 同上 | `iterHeadCore` | `uint64_t`/64 | 第 20；offset 不可见 | iteration | `SetUbSplitInfo: ceil(lineHeadCore/L)` | `Process` | head core 消费 |
| 同上 | `lineHeadIter` | `uint64_t`/64 | 第 21；offset 不可见 | 行/iteration | `SetUbSplitInfo: L` | 所有 core 的 UB 容量与常规迭代 | 必须 >0 |
| 同上 | `lineLastHeadIter` | `uint64_t`/64 | 第 22；offset 不可见 | 行 | `SetUbSplitInfo` 余数公式 | `Init/Process` | head 最后迭代 |
| 同上 | `lineTailCore` | `uint64_t`/64 | 第 23；offset 不可见 | 行/core | `SetUbSplitInfo: floor(R/C)` | `Init: coreOffset` | 可为 0 |
| 同上 | `iterTailCore` | `uint64_t`/64 | 第 24；offset 不可见 | iteration | `SetUbSplitInfo: ceil(lineTailCore/L)` | `Process` | 可为 0 |
| 同上 | `lineTailIter` | `uint64_t`/64 | 第 25；offset 不可见 | 行/iteration | `SetUbSplitInfo: L` | Device 未直接读取 | 冗余于 `lineHeadIter` |
| 同上 | `lineLastTailIter` | `uint64_t`/64 | 第 26；offset 不可见 | 行 | `SetUbSplitInfo` 余数公式 | `Init/Process` | `iterTailCore==0` 时值无效且不消费 |
| 同上 | `softmaxTilingData` | `SoftMaxTiling`/定义不可见 | 第 27；嵌套布局不可见 | 高阶 API 参数 | `SoftMaxTilingFunc([L,A],4,min(maxTmp,32768),...)` | `SoftmaxX -> SoftMax<float,...>` | 固定仓外公共高阶 API ABI |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | key 0 | `scaled_masked_softmax_v2_tiling.cpp: SetTilingKey/DoTiling` | `ScaledMaskedSoftmaxV2TilingData`，`s=4` | `scaled_masked_softmax_v2.cpp: scaled_masked_softmax_v2` → `<float>` | `scaled_masked_softmax_v2.h: Init/Process/CopyXIn/CopyMaskIn/Compute/SoftmaxX/CopyOut` | `ascend910b`、`ascend910_93` binary JSON；`op_kernel/CMakeLists.txt` |
| R-002 FP16 | key 1 | 同上，FP16 分支 | 同一 POD，`s=2` | 同一 entry → `<half>` | 同一模板的 half 编译分支 | 同上 |
| R-003 BF16 | key 2 | 同上，BF16 分支 | 同一 POD，`s=2` | 同一 entry → `<bfloat16_t>` | 同一模板的 bfloat16 编译分支 | 同上 |

## Kernel 实现

### 公共执行流程

三条 Route 共用一个单 Kernel 启动图。entry 用 `GET_TILING_DATA` 解包 POD，以 key 选择 `T` 后顺序调用 `Init`、`Process`。`Init` 计算本核的连续行区间，绑定 GM，并按最大迭代行数 `L` 初始化全部 UB buffer。`Process` 按本核 head/tail 行数循环：先把 mask 与 x 搬入、再缩放/选择/Softmax、最后只把每行有效 `W` 个元素写回。

Mask 搬运不是逐行盲目读取：`CalcCurPos/CalcEndPos` 把当前连续逻辑行区间拆成 batch/channel 段，`MoveMaskLines` 对每个段发一次多 block `DataCopyPad`；`nStep/cStep` 为 0 的广播维使 GM offset 固定，从而复用同一 mask 区域。单节点无节点间 workspace、event 或 atomic；队列的 `EnQue/DeQue/FreeTensor` 管理 VECIN/VECOUT 生命周期，高阶 Select/SoftMax 共用 `bufShare`。

### Route R-001：FP32 直接输出

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnScaledMaskedSoftmax.md` | 产品支持/参数表 | A2/A3；FP32；4-D；广播；`0<W<=4096` | 进入公开第一段接口 |
| API | `op_host/op_api/aclnn_scaled_masked_softmax.cpp` | `CheckParams` / `aclnnScaledMaskedSoftmaxGetWorkspaceSize` | null/dtype/dim/shape 成功且 `fixedTriuMask==false` | inner V2 executor |
| Tiling | `op_host/scaled_masked_softmax_v2_tiling.cpp` | `SetTilingKey` | dtype 不是 FP16/BF16，且先前仅允许三 dtype | key 0 |
| Router | `op_kernel/scaled_masked_softmax_v2.cpp` | `scaled_masked_softmax_v2` | `TILING_KEY_IS(0)` | `ScaledMaskedSoftmaxV2<float>` |

#### Device 数据流

##### 函数级调用链

```text
ScaledMaskedSoftmaxV2 Tiling -> key 0, blockDim=C
  -> scaled_masked_softmax_v2(x, mask, y, workspace_unused, tiling)
    -> ScaledMaskedSoftmaxV2<float>::Init
    -> Process
      -> CopyIn -> CopyMaskIn/MoveMaskLines -> CopyXIn
      -> Compute -> Muls<float> -> SelectWithBytesMask<float> -> SoftmaxX -> SoftMax<float>
      -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；一核处理连续 rows，最后 iter 缩短
for idx in [0, iter):
  lines = (idx == iter-1) ? last_rows : L
  mask_ub = DMA_PAD(mask segments selected by nStep/cStep, rows=lines, row_bytes=W, row_stride=M)
  x_ub    = DMA_PAD(x + coreOffset + idx*W*L, rows=lines, row_bytes=4W, row_stride=A)
  Muls(x_ub, x_ub, float(scale), A*lines)
  SelectWithBytesMask(bufX_fp32, x_ub, -10000.0f, mask_ub,
                      firstAxis=lines, srcLastAxis=A, maskLastAxis=M, shared=bufShare)
  SoftMax<float>(y_queue_fp32, bufX_fp32, bufShare,
                 shape={srcM=lines, srcK=A, oriM=lines, oriK=W})
  DMA(y + coreOffset + idx*W*L, y_queue, rows=lines, row_bytes=4W)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `coreOffset + idx*W*L` | `lines*W` 有效元素 | FP32，连续行 | caller/框架 | `CopyXIn` | caller；调用期间有效 | DMA 每行 `4W` bytes，UB pad 到 A |
| GM mask | `b*nStep*Mn*H*W + n*cStep*H*W + h*W` | 分段的 `lines*W` BOOL | BOOL，广播逻辑视图 | caller/框架 | `MoveMaskLines` | caller；调用期间有效 | 每段一次多行 DMA，UB pad 到 M |
| `inQueueX` | UB 子区 | `A*L*4` bytes，depth 1 | FP32 行距 A | `CopyXIn` | `Compute` | Alloc→EnQue→DeQue→Free | 32-byte 行对齐 |
| `inQueueMask` | UB 子区 | `M*L` bytes，depth 1 | BOOL 行距 M | `CopyMaskIn` | `Compute` | 同上 | 32-byte 行对齐 |
| `bufX` | UB 子区 | `A*L*4` bytes | FP32 | Select | SoftMax | 整个 Kernel，逐 iter 覆盖 | FP32 中间 |
| `bufShare` | UB 子区 | `Sdev=32768 if A*L<=32768 else 65536` bytes | uint8 tmp | Select/SoftMax 内部 | 同一高阶 API | 整个 Kernel；两 API 串行复用 | 不保存跨阶段结果 |
| `outQueueY` | UB 子区 | `A*L*4` bytes，depth 1 | FP32 行距 A | SoftMax | `CopyOut` | Alloc→EnQue→DeQue→Free | 仅写 GM 的 W 列 |
| GM y | `coreOffset + idx*W*L` | `lines*W` 有效元素 | FP32，连续行 | `CopyOut` | caller | caller 持有 | 每行 `4W` bytes |
| workspace | N/A | `GetLibApiWorkSpaceSize()` 系统区 | 框架私有 | runtime | 公共库/launcher | executor 调用期 | Kernel 形参未读取 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inX["inQueueX · d=1 · A·L·4 B"]:1
  inM["inQueueMask · d=1 · M·L B"]:1
  calc["bufX · A·L·4 B"]:1
  share["bufShare · Sdev B"]:1
  outY["outQueueY · d=1 · A·L·4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = A*L*4 + M*L + A*L*4 + Sdev + A*L*4
         = 12*A*L + M*L + Sdev
Host L = floor((U-32768)/(12*A+M))
若 Sdev=32768，则 UB_total <= U；Sdev=65536 分支在固定 Host 公式中无单独上界证明。
dynamic_ub = N/A
workspace_total = GetLibApiWorkSpaceSize()；算子自有 slice = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueueX` | `CopyXIn` | `Compute` | 下个 idx 的 Alloc | `EnQue→DeQue→FreeTensor` | depth 1，必须先 Free |
| `inQueueMask` | `CopyMaskIn` | `Compute` | 下个 idx | 同上 | Select 完成后才 Free |
| `bufX` | Select | SoftMax | 下个 idx Select | Process 顺序；SoftMax 调用返回后覆盖 | 不跨迭代保存 |
| `bufShare` | Select/SoftMax 内部 | 同一调用 | 后续高阶 API/下个 idx | 源码顺序调用 | 不得并行交叠两个使用者 |
| `outQueueY` | SoftMax | `CopyOut` | 下个 idx | `EnQue→DeQue→FreeTensor` | DMA 提交后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VECIN | `DataCopyPad` + queue | DMA | `DeQue` | x/mask queue | x pad `A-W`、mask pad `M-W`；padding 值 0 |
| 缩放 | `Muls` 原地 | x queue | Select | `xTensor` | FP32 scale；处理 `A*lines`，padding 也计算但后续排除 |
| mask | `SelectWithBytesMask` | Muls/mask DMA | SoftMax | `bufX`/shared | 标量侧为 `-10000.0f`；shape 显式区分 A 与 M；布尔选边语义为公共 primitive 边界 |
| Softmax | `SoftMax<float,false,false>` | Select | output queue | 行内 FP32 结果 | `oriK=W` 排除 pad；稳定细节由高阶 API tiling/body边界提供 |
| VECOUT→GM | queue + `DataCopyPad` | SoftMax | GM y | output queue | 仅每行 `W` 个元素写回 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scaled_masked_softmax_v2_tiling.cpp` | `SetPaddingInfo/SetUbSplitInfo/SetSoftmaxTiling/SetTilingKey` | A/M/L、分核、key 0 与 SoftMaxTiling |
| Kernel entry/router | `op_kernel/scaled_masked_softmax_v2.cpp` | `scaled_masked_softmax_v2` / `<float>` | key 0 实例化 FP32 body |
| Device body/helper | `op_kernel/scaled_masked_softmax_v2.h` | `Init/Process/Compute/SoftmaxX` | 队列、FP32 Select、直接输出 SoftMax、tail |

### Route R-002：FP16 先低精度选择再升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_scaled_masked_softmax.cpp` | `CheckDtypeValid` | `x/y=DT_FLOAT16`、mask BOOL，其余校验成功 | inner V2 executor |
| Tiling | `op_host/scaled_masked_softmax_v2_tiling.cpp` | `SetTilingKey` | `xDataType == ge::DT_FLOAT16` | key 1 |
| Router | `op_kernel/scaled_masked_softmax_v2.cpp` | `scaled_masked_softmax_v2` | `TILING_KEY_IS(1)` | `ScaledMaskedSoftmaxV2<half>` |

#### Device 数据流

##### 函数级调用链

```text
scaled_masked_softmax_v2 key 1
  -> ScaledMaskedSoftmaxV2<half>::Init/Process
    -> CopyMaskIn + CopyXIn
    -> Compute: Muls<half> -> SelectWithBytesMask<half> -> Cast half→float
               -> SoftMax<float> in-place bufX -> Cast float→half RINT
    -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
for each iter of this core:
  DMA_PAD mask and FP16 x for `lines` rows
  Muls(x_half, x_half, half(scale), A*lines)
  SelectWithBytesMask(x_half, x_half, half(-10000), mask, shared, {lines,A,M})
  Cast(bufX_fp32, x_half, CAST_NONE, A*lines)
  SoftMax<float>(bufX_fp32, bufX_fp32, shared, shape={lines,A,lines,W})
  Cast(y_half, bufX_fp32, CAST_RINT, A*lines)
  DMA only W half elements per row to GM y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | 与 R-001 相同 | 每 iter `lines*W` | FP16 | caller / CopyOut | CopyXIn / caller | 调用期 | 每行 `2W` bytes，UB 行距 A |
| GM mask | 与 R-001 相同 | 每 iter `lines*W` | BOOL | caller | MoveMaskLines | 调用期 | UB 行距 M |
| `inQueueX` | UB 子区 | `2*A*L` bytes，depth 1 | FP16 | CopyXIn | Muls/Select/Cast | 单 iter | 原地 Muls 与 Select |
| `inQueueMask` | UB 子区 | `M*L` bytes，depth 1 | BOOL | CopyMaskIn | Select | 单 iter | 32-byte 行距 |
| `bufX` | UB 子区 | `4*A*L` bytes | FP32 | Cast | in-place SoftMax | Kernel 期，逐 iter 覆盖 | SoftMax src/dst alias |
| `bufShare` | UB 子区 | `Sdev` bytes | uint8 | Select/SoftMax | 同一 API | Kernel 期 | 串行复用 |
| `outQueueY` | UB 子区 | `2*A*L` bytes，depth 1 | FP16 | RINT Cast | CopyOut | 单 iter | 只提交 W 列 |
| workspace | N/A | 系统 workspace | 框架私有 | runtime | 公共库 | executor 调用期 | Kernel 不读 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inX["inQueueX · d=1 · 2A·L B"]:1
  inM["inQueueMask · d=1 · M·L B"]:1
  calc["bufX FP32 · 4A·L B"]:1
  share["bufShare · Sdev B"]:1
  outY["outQueueY · d=1 · 2A·L B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2A*L + M*L + 4A*L + Sdev + 2A*L
         = 8A*L + M*L + Sdev
Host L = floor((U-32768)/(8A+M))
若 Sdev=32768，则 UB_total <= U。
dynamic_ub = N/A
workspace_total = GetLibApiWorkSpaceSize()；算子自有 slice = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueueX` | CopyXIn | Muls→Select→Cast | 下个 iter | Cast 后 `FreeTensor` | 三步必须完成后释放 |
| `bufX` | half→float Cast | SoftMax src/dst | 下个 iter Cast | 同一函数顺序 | SoftMax 明确允许这里的同 tensor src/dst 调用形态 |
| `bufShare` | Select/SoftMax | 同一 API | 后一 API/iter | 顺序调用 | 不跨调用保留 |
| `outQueueY` | float→half Cast | CopyOut | 下个 iter | queue 生命周期 | CopyOut 后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | DataCopyPad/queue | GM | Compute DeQue | x/mask | A/M padding |
| scale+mask | 原地 Muls/Select | x/mask | Cast | x queue | scale 和 `-10000` 先转换为 half，故这两步为 FP16 |
| 升精度 | `Cast(...CAST_NONE)` | half x queue | SoftMax | FP32 bufX | `A*lines` 元素 |
| Softmax | FP32 in-place高阶 API | Cast | 降精度 Cast | bufX/shared | 有效列 W，padding 不输出 |
| 降精度/CopyOut | `CAST_RINT` + queue/DMA | FP32 SoftMax | GM y | out queue | round-to-nearest integer mode；只写 W 列 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scaled_masked_softmax_v2_tiling.cpp` | `getDataTypeSize/SetTilingKey` | FP16 size=2、key 1 |
| Kernel entry/router | `op_kernel/scaled_masked_softmax_v2.cpp` | key 1 / `<half>` | half 模板实例 |
| Device body/helper | `op_kernel/scaled_masked_softmax_v2.h` | `Compute<half>` | 低精度 Muls/Select、FP32 SoftMax、RINT 输出 |

### Route R-003：BF16 先升精度再缩放选择

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_scaled_masked_softmax.cpp` | `CheckDtypeValid` | `x/y=DT_BF16`、mask BOOL，其余校验成功 | inner V2 executor |
| Tiling | `op_host/scaled_masked_softmax_v2_tiling.cpp` | `SetTilingKey` | `xDataType == ge::DT_BF16` | key 2 |
| Router | `op_kernel/scaled_masked_softmax_v2.cpp` | `scaled_masked_softmax_v2` | key 2，且 A2/A3 不等于宏排除的 3003/3113 | `<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
scaled_masked_softmax_v2 key 2
  -> ScaledMaskedSoftmaxV2<bfloat16_t>::Init/Process
    -> CopyMaskIn + CopyXIn
    -> Compute: Cast bf16→float -> Muls<float> -> SelectWithBytesMask<float>
               -> SoftMax<float> in-place bufX -> Cast float→bf16 RINT
    -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
for each iter of this core:
  DMA_PAD mask and BF16 x
  Cast(bufX_fp32, x_bf16, CAST_NONE, A*lines)
  Muls(bufX_fp32, bufX_fp32, float(scale), A*lines)
  SelectWithBytesMask(bufX_fp32, bufX_fp32, -10000.0f, mask, shared, {lines,A,M})
  SoftMax<float>(bufX_fp32, bufX_fp32, shared, shape={lines,A,lines,W})
  Cast(y_bf16, bufX_fp32, CAST_RINT, A*lines)
  DMA W BF16 elements per row to GM y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | 与 R-001 相同 | 每 iter `lines*W` | BF16 | caller / CopyOut | CopyXIn / caller | 调用期 | 每行 `2W` bytes |
| GM mask | 与 R-001 相同 | 每 iter `lines*W` | BOOL | caller | MoveMaskLines | 调用期 | 行距 M |
| `inQueueX` | UB 子区 | `2*A*L` bytes，depth 1 | BF16 | CopyXIn | 首个 Cast | 单 iter | Cast 后可释放 |
| `inQueueMask` | UB 子区 | `M*L` bytes，depth 1 | BOOL | CopyMaskIn | Select | 单 iter | 32-byte 行距 |
| `bufX` | UB 子区 | `4*A*L` bytes | FP32 | Cast→Muls→Select→SoftMax | 下一步/降精度 | Kernel 期 | 四步原地链 |
| `bufShare` | UB 子区 | `Sdev` bytes | uint8 | Select/SoftMax | 同一 API | Kernel 期 | 串行复用 |
| `outQueueY` | UB 子区 | `2*A*L` bytes，depth 1 | BF16 | RINT Cast | CopyOut | 单 iter | 只提交 W 列 |
| workspace | N/A | 系统 workspace | 框架私有 | runtime | 公共库 | executor 调用期 | Kernel 不读 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inX["inQueueX · d=1 · 2A·L B"]:1
  inM["inQueueMask · d=1 · M·L B"]:1
  calc["bufX FP32 · 4A·L B"]:1
  share["bufShare · Sdev B"]:1
  outY["outQueueY · d=1 · 2A·L B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 8A*L + M*L + Sdev
Host L = floor((U-32768)/(8A+M))
若 Sdev=32768，则 UB_total <= U。
dynamic_ub = N/A
workspace_total = GetLibApiWorkSpaceSize()；算子自有 slice = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueueX` | CopyXIn | BF16→FP32 Cast | 下个 iter | Compute 尾部 Free | Cast 完成前不可覆盖 |
| `bufX` | Cast | 原地 Muls→Select→SoftMax | 下个 iter Cast | 单函数调用顺序 | 每一步完成后才进入下一步 |
| `bufShare` | Select/SoftMax | 同一 API | 后一 API/iter | 顺序调用 | 不并行复用 |
| `outQueueY` | FP32→BF16 Cast | CopyOut | 下个 iter | queue 生命周期 | DMA 后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | DataCopyPad/queue | GM | Compute | x/mask | A/M padding |
| 升精度 | `Cast(...CAST_NONE)` | BF16 queue | Muls | bufX | 在缩放前完成 |
| scale+mask | FP32 原地 Muls/Select | Cast/mask | SoftMax | bufX/shared | scale 与 mask 常量保持 FP32 |
| Softmax | FP32 in-place | Select | output Cast | bufX | `oriK=W` |
| 降精度/CopyOut | `CAST_RINT` + queue/DMA | SoftMax | GM y | output queue | 只写 W 列 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/scaled_masked_softmax_v2_tiling.cpp` | `getDataTypeSize/SetTilingKey` | BF16 size=2、key 2 |
| Kernel entry/router | `op_kernel/scaled_masked_softmax_v2.cpp` | key 2 / `<bfloat16_t>` | A2/A3 可编译 BF16 body |
| Device body/helper | `op_kernel/scaled_masked_softmax_v2.h` | `Compute<bfloat16_t>` | 先升 FP32，再缩放/mask/Softmax，RINT 输出 |

## 算子特定优化

### Trick T-001：按广播段合并 mask DMA

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：一个 iter 的连续逻辑行可能跨 channel/batch；`nStep/cStep` 分别由对应 mask 维是否广播决定。
- Golden 来源：[`op_kernel/scaled_masked_softmax_v2.h:147-183`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.h#L147-L183)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
MaskOffset maskOffset = {curBatch, curChannel, curLineInChannel};
uint64_t lineCnt = 0u;
if (curBatch == endBatch) {
    if (curChannel == endChannel) {
        MoveMaskLines(maskTensor, lineCnt, maskOffset, linePerIter);
    } else {
        MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight - curLineInChannel);
        maskOffset.NextChannel(this->xChannel);
        for (uint64_t i = curChannel + 1; i < endChannel; ++i) {
            MoveMaskLines(maskTensor, lineCnt, maskOffset, this->xHeight);
            maskOffset.NextChannel(this->xChannel);
        }
        MoveMaskLines(maskTensor, lineCnt, maskOffset, endLineInChannel);
    }
```

#### 机制与设计原因

代码先定位 iter 的起止 batch/channel/height，再按同一 channel 内连续行发起一次 `DataCopyPad`。`MoveMaskLines` 的 `blockCount=lines`、`blockLen=W*sizeof(bool)`，所以一段可覆盖多行；广播维由 `maskBatchOffset=nStep*Mn*H*W`、`maskChannelOffset=cStep*H*W` 变为零步长地址映射，无需在 GM 物化广播后的完整 mask。该机制减少 DMA 调用粒度并避免广播张量展开；跨 channel/batch 时仍按真实非连续边界拆段。

#### 迁移边界

- 适用：末两维相同、仅外层维广播，且迭代域是连续逻辑行的 masked 算子。
- 前提：mask 每行连续，`DataCopyPad` 的 blockCount/blockLen 能表示段长和行宽；广播维必须通过零 offset 映射而非错误地连续递增。
- 失效：若 height/width 也广播、mask layout 非连续且未先框架连续化，需重新设计 offset/stride；本算子无对应 fallback Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从源码约束可确定，不能把跨 channel/batch 的整个 iter 无条件当成一段连续 mask GM 区域。

### Trick T-002：dtype 专化控制升精度时机与中间物化

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：编译期模板 `T` 分别为 float、half、bfloat16_t。
- Golden 来源：[`op_kernel/scaled_masked_softmax_v2.h:232-256`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.h#L232-L256)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (std::is_same<T, bfloat16_t>::value) {
    Cast(scaledMaskedX, xTensor, RoundMode::CAST_NONE, this->elePerIter);
    Muls(scaledMaskedX, scaledMaskedX, static_cast<float>(scale), this->elePerIter);
    AscendC::SelectWithBytesMask(
        scaledMaskedX, scaledMaskedX, MASK_VAL, maskTensor, sharedBuffer, selectShapeInfo);
} else {
    Muls(xTensor, xTensor, static_cast<T>(scale), this->elePerIter);
    if constexpr (std::is_same<T, half>::value) {
        AscendC::SelectWithBytesMask(
            xTensor, xTensor, static_cast<T>(MASK_VAL), maskTensor, sharedBuffer, selectShapeInfo);
        Cast(scaledMaskedX, xTensor, RoundMode::CAST_NONE, this->elePerIter);
    } else {
        AscendC::SelectWithBytesMask(
            scaledMaskedX, xTensor, MASK_VAL, maskTensor, sharedBuffer, selectShapeInfo);
    }
}
```

#### 机制与设计原因

BF16 在所有算术前进入 FP32，避免 BF16 缩放与 mask 常量量化；FP16 先在输入队列原地完成 Muls/Select，再只物化一个 FP32 Softmax buffer；FP32 Select 直接产出 `bufX`，Softmax 又直接写 `outQueueY`，无需额外 FP32 输出中间区。关键元素数来自 `elePerIter=A*lines`，shape 参数来自 POD 的 A/M 与当前 `lines`。

#### 迁移边界

- 适用：输入 dtype 决定精度/带宽取舍，且 Softmax 固定要求 FP32 中间的融合实现。
- 前提：必须保持 FP16 的“低精度 scale/mask 后 Cast”和 BF16 的“先 Cast 后 FP32 scale/mask”顺序；输出仍以 `CAST_RINT` 降精度。
- 失效：若数值规范要求 FP16 也在 FP32 缩放，不能复用 R-002 的顺序，应采用类似 R-003 的链路并重新计算性能边界。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得仅因 FP16/BF16 都是 2-byte dtype 就合并两条 Route，它们的精度顺序不同。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 0，剩余合法 dtype | key 1，显式 FP16 | key 2，显式 BF16 | `SetTilingKey` 与 router 顺序 |
| 模板与 dtype | `<float>` | `<half>` | `<bfloat16_t>` | binary config 三种实例 |
| 循环与分核 | 共享 R/C/L 公式，`s=4` 影响 A/L | 共享公式，`s=2` | 同 FP16 的 A/L 字节公式 | Host 按 dtype size 计算 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | FP32 in/out queue；FP32 bufX；无显式 RegTensor | 2-byte in/out queue；FP32 bufX | 同 FP16 | 均无 dynamic UB/通信 window |
| 内存总占用与复用 | `(12A+M)L+Sdev`；SoftMax 可直写 output queue | `(8A+M)L+Sdev`；x queue 原地 scale/select、bufX 原地 SoftMax | `(8A+M)L+Sdev`；bufX 连续原地链 | Queue depth 均为 1 |
| 搬运/Vector/SIMT/NDDMA/通信 API | FP32 DMA/Muls/Select/SoftMax | FP16 DMA/Muls/Select + Cast + FP32 SoftMax | BF16 DMA + Cast + FP32 Muls/Select/SoftMax | 均为 AIV_VECTOR 单节点 |
| 同步、精度与 tail | 全 FP32；oriK=W | FP16 scale/mask→FP32 SoftMax→RINT | FP32 scale/mask/SoftMax→RINT | tail 都用最后迭代行数与 W 有效列 |
| fallback/失败边界 | unknown key no-op | 同左 | Kirin 3003/3113 编译排除；A2/A3 可达 | 无额外 fallback Kernel |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档支持；`ascend910b` config 闭合 | 公开 ACLNN 文档支持；`ascend910_93` config 闭合 | 当前卡不覆盖；固定公开 ACLNN 产品表未列 950，故不发布 `_950` 卡 |
| Host/Launcher | 共用非 RegBase `W<=4096`、32 KiB Host Softmax tmp 预留 | 同 A2 | 实现存在 RegBase `W<=8192`/64 KiB 线索，但公开 API 可达性不闭合 |
| Kernel/资源 | 共用单 AIV Vector Kernel 与三 dtype Route | 同 A2 | 有 config/CMake 信号，不足以单独证明公开支持 |

## 限制与失败边界

- 仅覆盖 forward `ScaledMaskedSoftmax`；不包含 Backward/Grad、原地、量化或其它 vfusion 算子。
- 公开 API 要求 `mask` 非空且 `fixTriuMask=false`；内部 OpDef 的 OPTIONAL mask 不意味着该 ACLNN 可省略 mask。
- A2/A3 的末轴限制为 `(0,4096]`；仅 batch/channel 支持大小 1 广播，height/width 不广播。
- `scale` 从 API double 到内部 FP32 attr/POD 的转换发生在固定仓不可见的生成式 bridge 边界；Device 不保留 double 精度。
- `TilingScaledMaskedSoftmaxV2` 未检查 `Init()` 的返回值；固定源码不能证明 Init 失败必然阻断 `DoTiling()`。
- Host 以 32 KiB 计算 `L`，Device 在 `A*L>32768` 时把 `bufShare` 改为 64 KiB；固定仓没有对应额外 Host guard，故只能在实际 `Sdev=32768` 时从公式直接证明 `UB_total<=U`。
- `B*N*H`、各 GM offset 与 UB 乘积使用 `uint64_t` 且没有显式乘法溢出检查；`linePerIter`/mask 分段行数又窄化到 `uint16_t` DMA `blockCount`，固定 Host 没有显式 `<=65535` guard。
- `scale` 没有有限值/范围检查；其 NaN/Inf 行为与 Vector/SoftMax primitive 一致，固定仓未另行规定。
- 未知 tiling key 无错误分支且不写输出；正常 Host 只产生 0/1/2。
- `SoftMaxTiling` ABI、`SoftMax` 与 `SelectWithBytesMask` primitive 内部为固定仓外公共库边界；算子传入的 shape、tmp、dtype、顺序与有效宽度已经闭合。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `vfusion/scaled_masked_softmax_v2/docs/aclnnScaledMaskedSoftmax.md` | `aclnnScaledMaskedSoftmax*`、产品支持表 | 公开性、A2/A3、接口约束与两段式调用 |
| 入口实现 | `vfusion/scaled_masked_softmax_v2/op_host/op_api/aclnn_scaled_masked_softmax.cpp` | `CheckParams`、两个公开入口 | 校验、RegBase width 分支、inner bridge |
| Host/注册 | `vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_def.cpp` | `ScaledMaskedSoftmaxV2`、`OP_ADD` | dtype/layout、AutoContiguous、A2/A3/950 实现配置 |
| Host/Tiling | `vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_tiling.h` | `ScaledMaskedSoftmaxV2TilingData` | POD 字段顺序与类型 |
| Host/Tiling | `vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_tiling.cpp` | `ScaledMaskedSoftmaxV2Tiling::*` | shape/attr/platform、A/M/L、分核、key、workspace |
| Kernel entry/body | `vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.cpp` | `scaled_masked_softmax_v2` | key 到三种模板实例 |
| Kernel entry/body | `vfusion/scaled_masked_softmax_v2/op_kernel/scaled_masked_softmax_v2.h` | `ScaledMaskedSoftmaxV2<T>` | 完整 Device 循环、DMA、Vector、UB、广播和 tail |
| 构建接线 | `vfusion/scaled_masked_softmax_v2/op_host/config/ascend910b/scaled_masked_softmax_v2_binary.json` | `ScaledMaskedSoftmaxV2_*` | A2 三 dtype binary |
| 构建接线 | `vfusion/scaled_masked_softmax_v2/op_host/config/ascend910_93/scaled_masked_softmax_v2_binary.json` | `ScaledMaskedSoftmaxV2_*` | A3 三 dtype binary |
| 构建接线 | `vfusion/scaled_masked_softmax_v2/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn_inner)` | Host 源与生成式 inner ACLNN 接线 |
| 构建接线 | `vfusion/scaled_masked_softmax_v2/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 Kernel 编译单元 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | A/M、广播 step、head/tail rows 与 SoftMaxTiling 的生产/消费关系 | 字段声明顺序、64/32 位宽、元素/byte单位；整体 offset 不可臆测 | `ScaledMaskedSoftmaxV2TilingData` |
| 调用顺序 | 单 Kernel 内 CopyMask/CopyX→scale→mask→SoftMax→CopyOut | FP16 与 BF16 的 Cast 时机不同 | `Process/Compute` |
| workspace/内存 | 三 depth-1 queue + FP32 bufX + shared tmp | 容量用 A/M/L 而非 W；depth 与 Sdev 分支不可遗漏 | `Init/SetUbSplitInfo` |
| 同步 | queue 生命周期与串行 `bufShare` 复用 | Free/覆盖必须晚于消费；不能并行使用 shared tmp | `CopyXIn/CopyMaskIn/Compute/CopyOut` |
| 精度与尾块 | FP32 Softmax、RINT 降精度、`oriK=W` | mask 常量 `-10000`、有效宽度与最后 iter 行数 | `Compute/SoftmaxX/Process` |
| 失败边界 | API/shape/UB guards 与 unknown-key no-op | 不得把 `Init()` 失败传播或 950 公开支持补齐 | `CheckParams/SetUbSplitInfo/TilingScaledMaskedSoftmaxV2/scaled_masked_softmax_v2` |

- [SoftMax/SimpleSoftMax Tiling](../../../../apis/simd_api/adv_api/activation_functions/soft_max_interface/soft_max_simple_soft_max_tiling.md)
