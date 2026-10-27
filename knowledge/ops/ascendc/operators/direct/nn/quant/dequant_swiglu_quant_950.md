---
type: Operator
title: "DequantSwigluQuant 950 Direct Invoke 算子知识"
description: "说明 cann_bench::dequant_swiglu_quant 在 950 上的接口语义、Launcher 参数生成、Route 选择、单 Kernel 启动图与逐实例实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/README.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant_dynamic_not_full.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant_dynamic_not_full_vf.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_host/config/ascend950/dequant_swiglu_quant_binary.json"
    role: implementation
created_at: "2026-08-24T19:34:12Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# DequantSwigluQuant 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | PyTorch schema `cann_bench::dequant_swiglu_quant`，`PrivateUse1` 实现 `dequant_swiglu_quant_npu`；其后按 dtype 调用 `launch_dequant_swiglu_quant_{int32,half,bfloat16}`。入口类型为 Direct Invoke。 |
| 计算语义 | 对 `x[M,2H]` 做可选 INT32 反量化，选择左右半部形成 `Swish(act) * gate`，可选逐列乘 `quant_scale`，再按行动态量化为 `int8 y[M,H]` 并输出 `float32 scale[M]`。 |
| 输入输出 | `x`: contiguous ND、INT32/FP16/BF16；可选 scale 均为 contiguous FP32；`y`: INT8；`scale`: FP32。 |
| 执行类型 | 所有 Route 均为单 Kernel `AIV_VECTOR`；entry 以 `g_coreType != AIV` 直接返回，不含 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | 重要性顺序：P0 R-002、P1 R-001、P2 R-004、P3 R-006、P4 R-005、P5 R-003、P6 R-008、P7 R-007、P8 R-010、P9 R-012、P10 R-011、P11 R-009。R-001～R-006 为 FL（full-load，`H <= 5120`），R-007～R-012 为 NF（dynamic-not-full，`H > 5120`）；Route ID 保持 Host 叶节点身份。 |
| 核心资源 | GM、AIV UB、`TQue`、`TBuf`、MicroAPI 寄存器；NF 额外使用 `usedCoreNum * H * sizeof(float)` 的用户 workspace。 |
| 来源闭包 | `direct_implementation_closed`：schema/校验/分配 → PrivateUse1 → 三个 named launcher → Host tiling → 12 个 entry → FL/NF body/helper → 构建接线闭合。转换交付件是执行事实主来源；官方固定 Commit 仅作 950 支持、arch35 Route 与 body 相关映射，不把转换入口伪称为官方入口。 |

## 接口与语义

### 计算语义

令 `M=x.size(0)`、`2H=x.size(1)`。对第 `r` 行、第 `k` 列：

```text
INT32: d[r,k] = float(x[r,k]) * weight_scale[0,k] * activation_scale[r]
FP16/BF16: d[r,k] = float(x[r,k])

activate_left == true : act=d[:,0:H], gate=d[:,H:2H]
activate_left == false: act=d[:,H:2H], gate=d[:,0:H]

z = act / (1 + exp(-act)) * gate
v = quant_scale 存在 ? z * quant_scale[0,:] : z
scale[r] = max(max_k(abs(v[r,k])) / 127, 1e-12)
y[r,k] = saturating_int8_cast(v[r,k] / scale[r])
```

本直调固定 `quantMode=1`、`dstType=2`、`swiGluMode=0`、无 bias/group_index/quant_offset。INT8 cast 的真实顺序是 FP32→INT16（`CAST_RINT`、`NO_SAT`）→FP16（`CAST_ROUND`）→INT8（`CAST_TRUNC`、`SAT`），不能简化为任意语言默认取整。对有限输入，行 scale 的构造使理想商落在 `[-127,127]`，因此第一段不会依赖 INT16 饱和；其后 INT16 整数可被 FP16 精确表示，最后一段的饱和仍是舍入误差与异常值的安全边界。

### Scale、单组语义与半部选择

交付接口没有 group 参数，但 Device body 来自可分组的通用模板。此处 Host 通过模板类型和 POD 把它收缩为 `G=1`，因此不能把通用 body 中的 `[G,*]` 地址公式误当成公开能力。

| 对象 | 通用 body 地址语义 | 本直调固定语义 | 与 `activate_left` 的关系 |
|---|---|---|---|
| `x` | row-major `[M,2H]` | 每行两个连续 H 半部；不做转置或广播 | `actRight=activate_left?0:1`；0 取左半为 act，1 取右半为 act，另一半为 gate |
| `weight_scale` | `groupIdx*2H + halfOffset + k` | INT32 路线固定 `groupIdx=0`，所以是逐输入列 `[1,2H]`；act/gate 与 `x` 使用相同 halfOffset | 切换激活半部时，scale 半部必须同步切换 |
| `activation_scale` | `activation_scale[row]` | INT32 路线逐行标量 `[M]`，同时广播乘 act 与 gate；不是逐组或逐列 scale | 与左右选择无关 |
| `quant_scale` | `groupIdx*H+k` | 固定 `groupIdx=0`，是逐输出列 `[1,H]`；乘法而非除法 | 作用于 `Swish(act)*gate` 之后，不再区分半部 |
| `scale` | 每个输出 row 一个动态 scale | `max(maxabs(post_qscale_value)/127,1e-12)`；不是输入 dequant scale 的回传 | 左右选择改变数值，因而可改变行 scale |

模板实例把 `TGroup=bool`，`hasGroupIndex_` 在编译期为 false；`Process()` 直接调用 `ProcessSingleGroup(0,M,0)`。POD 中 `inGroupNum/groupIndexMode/speGroupType/isSpecialCoreCut` 虽保留 ABI 位置，但在 12 条可达 Route 中不参与寻址。`quantIsOne` 也不是本动态量化路径的标量广播开关：Q/NQ 已由不同模板实例裁剪，Q 路线始终读取 H 个 FP32 元素。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[M,2H]`，strided contiguous | INT32/FP16/BF16 | `1 <= M <= 65536`，`1 <= H <= 8192`；必须与可选输入同设备。 |
| `weight_scale` | `[1,2H]`，contiguous | FP32 | INT32 必选；FP16/BF16 必须缺省。 |
| `activation_scale` | `[M]`，contiguous | FP32 | INT32 必选；FP16/BF16 必须缺省。 |
| `quant_scale` | `[1,H]`，contiguous | FP32 | 可选；存在时先逐列乘入 `z`，再求动态量化 scale。 |
| `activate_left` | scalar | bool | 默认 `false`；Host 写 `actRight = activate_left ? 0 : 1`。 |
| `y` | `[M,H]`，contiguous | INT8 | 由 Device 完整写出。 |
| `scale` | `[M]`，contiguous | FP32 | 每行一个动态量化 scale，下限固定为 `1e-12`。 |

Host 只校验 scale 的 shape/dtype/layout/device，不校验正值或有限性。零或负 `quant_scale` 按普通乘法参与计算，最终 max 使用绝对值；`weight_scale`、`activation_scale`、`quant_scale` 中的 NaN/Inf 以及非线性溢出的传播由 950 MicroAPI 的 `Exp/Div/Abs/Max/ReduceMax` 决定，源码没有定义额外清洗或有限值 fallback。全零且有限的行得到 `scale=1e-12`、`y=0`。

### 入口与版本

`TORCH_LIBRARY_FRAGMENT` 注册 schema，Meta 与 PrivateUse1 共用 `CheckInputs` 和 `AllocateOutputs`；Meta 只做形状/类型推导。PrivateUse1 取得当前 NPU stream，通过 `OpCommand::RunOpApi("DequantSwigluQuantDirect", directCall)` 执行 named direct call。INT32、FP16、BF16 分别进入三个 Host launcher；这些符号是转换交付件入口，不是官方 ACLNN/Host 注册入口。

官方固定 Commit 的 README 与 `ascend950` binary config 证明相关 `DequantSwigluQuant` 实现支持 Ascend 950；官方 `dequant_swiglu_quant_apt.cpp` 的 arch35 router 和转换 body 构成相关映射。转换构建在 `NPU_ARCH=ascend950` 时选择 `--npu-arch=dav-3510`。本卡版本为 CANN 9.1.0。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckInputs` 检查二维、strided contiguous、dtype、`M/2H` 范围与可选 scale 的设备/dtype/shape。
2. INT32 强制两个 dequant scale 存在；FP16/BF16 强制二者缺省。`quant_scale` 独立可选。
3. `AllocateOutputs` 创建 `y[M,H]:int8` 与 `scale[M]:float32`。
4. `dequant_swiglu_quant_workspace_bytes` 调用与真实 launch 相同的 `CalcDirectTiling`；请求为零时仍分配 32B 占位 tensor，非零时按请求字节分配。
5. 取输入/输出/workspace GM 指针与当前 stream；按 `dtypeTag` 进入三个 named launcher。未知 dtype tag 在 `ParseDType` 抛出；平台查询、AIV/UB 资源或 tiling 因子非法也抛出。

### Host 整数边界与重复规划契约

公开 PyTorch 入口先完成范围校验，再窄化或相乘；在该调用链内可建立以下上界：

| Host 表达式 | 上界/类型 | 闭合理由 |
|---|---|---|
| `inDimy=2*H` | `<=16384`，写入 int64 POD | `CheckInputs` 先验证偶数尾轴 `[2,16384]` |
| `blockDim=usedCore` | `<=M<=65536`，窄化 uint32 | FL/NF 都有 `usedCore<=M` |
| `workspace=usedCore*H*4` | `<=65536*8192*4=2147483648` bytes | NF 仅在已校验的 M/H 域计算，结果可由 uint64 与 tensor 的 int64 长度表示 |
| `row*2H`、`row*H` | 分别小于 `65536*16384`、`65536*8192` elements | GM offset 使用 int64/size_t，当前 shape 域内不溢出 |
| `AlignUp(H,{8,16,32})` | `<=8224` elements | `H<=8192`，加法与乘法均远小于 int64 上界 |

源码的 `DivCeilHost/AlignUpHost` 和 workspace 乘法本身不是通用 checked-arithmetic 库；安全性依赖上述 Plugin 前置域。若独立调用 C ABI launcher/workspace API，必须在边界外补回 `rows<=65536`、`hidden<=8192` 与 `checked_mul`，不能把当前静态转换当成任意 uint64 输入都安全。

workspace 规划和 launch 各调用一次 `CalcDirectTiling`：第一次只返回字节数，第二次重新查询 AIV/UB、重建 POD 和 blockDim。两次必须使用同一组 `rows/hidden/dtypeTag/hasQuantScale/activateLeft`；调用侧按此固定参数，并在同一 `directCall` 前完成 workspace 分配。若资源查询失败或第二次规划需要的空间与先前请求不一致，源码没有扩容/重试路径。

### 参数、Workspace 与生命周期

设 `C=GetCoreNumAiv()`、`U=UB bytes`、`A32(n)=ceil(n/32)*32`、`A16(n)=ceil(n/16)*16`、`A8(n)=ceil(n/8)*8`。

- FL：`H <= 5120`，workspace 请求为 0；插件分配的 32B 只作非空占位，entry 不接收 workspace。
- NF：`H > 5120`，`usedCore=min(C,M)`，workspace 为 `usedCore*H*4` 字节。Device 以 `blockIdx*H` 为每核私有 slice；同一核处理下一行前，前一行的两遍处理已经完成，可安全覆盖。
- NF entry 直接把用户 workspace 地址映射为 `xTempGm_`，没有 registered runtime 的 system-workspace 前缀。workspace tensor 是 `dequant_swiglu_quant_npu` 的局部对象；`directCall` 使用并按值捕获的是从它导出的 `workspacePtr`，而返回的 `(y,scale)` 不持有该 tensor。源码没有显式 `recordStream`、completion event 或 host wait。因此 per-core 地址/容量及核内覆盖时刻由 Device 源码闭合，而 workspace storage 是否延续到 Kernel 完成依赖 `RunOpApi` 与 NPU allocator 的同流延迟回收契约，Direct 文件本身没有完成 owner 证明。
- 所有 Route 只有一个 Device launch，没有跨 Kernel workspace 边。

### 启动图与失败传播

```text
cann_bench::dequant_swiglu_quant (PrivateUse1)
  -> CheckInputs -> AllocateOutputs -> workspace_bytes
     -> CalcDirectTiling #1 (只返回 workspaceBytes)
  -> allocate workspace (0B 请求映射为 32B 占位)
  -> launch_dequant_swiglu_quant_{int32|half|bfloat16}
     -> CalcDirectTiling #2 (重建 POD/blockDim/Route)
     -> {FL/NF}_{dtype}_{q/nq}<<<blockDim, nullptr, stream>>>
        -> AIV class::Init -> Process
        -> GM y[M,H] + GM scale[M]
```

| 节点 | Host 启动函数 | entry | blockDim/POD | 输入输出/workspace | ordering 与失败 |
|---|---|---|---|---|---|
| K-001 | `LaunchInt32/LaunchHalf/LaunchBfloat16` | 12 个 `dequant_swiglu_quant_{fl,nf}_*` 之一 | FL `min(C,ceil(M/Fx))`；NF `min(C,M)`；POD 为 `DequantSwigluQuantV35BaseTilingData` 值传递 | 输入 GM → `y/scale` GM；仅 NF 使用 per-core workspace | 当前 NPU stream 单 launch；Host 校验/tiling 抛出会阻断 launch。entry 在非 AIV core 返回，不另行 fallback。 |

`directCall` 在执行 `<<<...>>>` 后固定返回 0，不做 stream synchronize 或显式 launch-status 查询。因此 shape、平台查询和 tiling 异常是同步 Host 失败；Kernel enqueue 之后的设备错误保持 current-stream 异步语义，只能由后续 runtime 同步点传播。输出 tensor 立即返回但其 GM 写入受同一 stream 顺序保护；调用方不得在无 stream 依赖的上下文中提前复用其存储。

## Route 与参数设计

### 分核与资源计算

FL 定义 `XF32=A8(H)`、`XX=A8(H)`（INT32）或 `A16(H)`（FP16/BF16）、`Y8=A32(H)`、`sX=4/2`。Host 先保留 `1024 + 12*XF32` 字节，再计算：

```text
Dfl = 2*(2*XX*sX + 8*4) + 2*Y8 + 8*4 + XF32*4
Fx  = min(M, floor((U - 1024 - 12*XF32) / Dfl))
usedCore = min(C, ceil(M/Fx))
```

这里 `XF32` 与 `XX` 是两套不同的行对齐：FP32 tmp/scale 使用 `XF32=A8(H)`，16-bit x 使用 `XX=A16(H)`。Host 的容量式按 `Fx*XF32` 预留 tmp；但 FL body 的当前源码在生成、归约和 cast 时用 `i*xTypeUbAlignB32_`（即 `i*XX`）寻址 tmp。于是 FP16/BF16 且 `A16(H)>A8(H)` 时，只有 `xDimPerLoop=1` 能由现有分配直接证明安全；多行 resident tile 需要的末地址为 `(xDimPerLoop-1)*XX+H`，可能超过 `Fx*XF32`。源码未提供相等性 guard 或额外 padding，这一对齐一致性是实现边界，复用时应统一 tmp 的“分配容量、Host denominator、Device stride”三者，不能沿用卡内较早的单一 `XF32` 描述掩盖差异。

NF 令 `reserve=1024+32`。INT32 的保守 `split=50`，FP16/BF16 为 `split=26`：

```text
L = floor((U-reserve)/split/32)*32
if L > H: L=A32(H)
loops=ceil(H/L), tail=H-(loops-1)*L
usedCore=min(C,M), workspace=usedCore*H*4
```

`L<=0`、`Fx<1`、`C==0` 或 `U<=1056` 均为 Host 失败。输入范围保证 `M,H` 可窄化到 POD 的 int64；`blockDim` 由不超过平台 AIV 核数的 `usedCore` 窄化到 uint32。

FL 分核还与 inScaleQueue 生命周期耦合。令 `B=ceil(M/usedCore)` 为普通 core 的连续行数：仅当 `ceil(M/Fx)<=C`、AIV 核数上限没有截断 `usedCore` 时，才有 `B<=Fx` 和 `ubDimxLoop=ceil(B/Fx)=1`。若 `ceil(M/Fx)>C`，则 `usedCore=C`、`B` 可能大于 `Fx`，同一 core 会执行多个 row batch。当前 body 把 weight/quant scale tensor 放在 batch 循环外只分配/DeQue 一次，却在每个 batch 末 `FreeTensor(inScaleLocal)`；多 batch 下第二次及后续迭代继续使用已释放 local，并重复释放同一 Queue 对象，源码没有给出安全证明。复用时应把释放移到循环后，或每个 batch 成对重新分配/搬入；不能把“FL”自动等同于“每 core 恒单 batch”。

### Route 选择

P0→P11 是本卡的实现知识重要性顺序，不是 Host 判断序号。Host 固定先按 dtype 进入 named launcher，再判断 FL/NF，最后按 quant_scale 有无短路；真实控制流保留在最后一列。

| 重要性 | Route | Route 特征 | 适用域与实现前提 | Host 顺序、短路与 fallback |
|---|---|---|---|---|
| P0 | R-002 FL-I32-Q | INT32 双 scale 反量化；整行常驻；逐列 qscale 参与行归约与量化 | `dtypeTag=0 && H<=5120 && hasQuantScale`；FL 行因子可成立 | H0 进入 INT32 launcher；H1 `notFull=false`；H2 `hasQuantScale=true` 后命中并结束。`H>5120` 先转 R-008；无 Device fallback |
| P1 | R-001 FL-I32-NQ | INT32 双 scale 反量化；整行常驻；无逐列 qscale | `dtypeTag=0 && H<=5120 && !hasQuantScale`；FL 行因子可成立 | H0 INT32；H1 FL；H2 qscale 判断为 false 后命中。`H>5120` 转 R-007；无 Device fallback |
| P2 | R-004 FL-F16-Q | FP16 升 FP32；整行常驻；逐列 qscale 乘入后归约 | `dtypeTag=1 && H<=5120 && hasQuantScale`；无需 dequant scales | H0 FP16；H1 FL；H2 qscale 为 true 后命中。`H>5120` 转 R-010；无 Device fallback |
| P3 | R-006 FL-BF16-Q | BF16 升 FP32；整行常驻；逐列 qscale 乘入后归约 | `dtypeTag=2 && H<=5120 && hasQuantScale`；无需 dequant scales | H0 BF16；H1 FL；H2 qscale 为 true 后命中。`H>5120` 转 R-012；无 Device fallback |
| P4 | R-005 FL-BF16-NQ | BF16 升 FP32；整行常驻；无逐列 qscale | `dtypeTag=2 && H<=5120 && !hasQuantScale`；FL 行因子可成立 | H0 BF16；H1 FL；H2 qscale 为 false 后命中。`H>5120` 转 R-011；无 Device fallback |
| P5 | R-003 FL-F16-NQ | FP16 升 FP32；整行常驻；无逐列 qscale | `dtypeTag=1 && H<=5120 && !hasQuantScale`；FL 行因子可成立 | H0 FP16；H1 FL；H2 qscale 为 false 后命中。`H>5120` 转 R-009；无 Device fallback |
| P6 | R-008 NF-I32-Q | INT32 双 scale 反量化；H 分片两遍；逐列 qscale；per-core workspace | `dtypeTag=0 && H>5120 && hasQuantScale`；NF 尾轴因子可成立 | H0 INT32；H1 `notFull=true`；H2 qscale 为 true 后命中并结束。Host 资源失败即异常，无 fallback |
| P7 | R-007 NF-I32-NQ | INT32 双 scale 反量化；H 分片两遍；无逐列 qscale；per-core workspace | `dtypeTag=0 && H>5120 && !hasQuantScale`；NF 尾轴因子可成立 | H0 INT32；H1 NF；H2 qscale 为 false 后命中。Host 资源失败即异常，无 fallback |
| P8 | R-010 NF-F16-Q | FP16 升 FP32；H 分片两遍；逐列 qscale；per-core workspace | `dtypeTag=1 && H>5120 && hasQuantScale`；NF 尾轴因子可成立 | H0 FP16；H1 NF；H2 qscale 为 true 后命中。Host 资源失败即异常，无 fallback |
| P9 | R-012 NF-BF16-Q | BF16 升 FP32；H 分片两遍；逐列 qscale；per-core workspace | `dtypeTag=2 && H>5120 && hasQuantScale`；NF 尾轴因子可成立 | H0 BF16；H1 NF；H2 qscale 为 true 后命中。Host 资源失败即异常，无 fallback |
| P10 | R-011 NF-BF16-NQ | BF16 升 FP32；H 分片两遍；无逐列 qscale；per-core workspace | `dtypeTag=2 && H>5120 && !hasQuantScale`；NF 尾轴因子可成立 | H0 BF16；H1 NF；H2 qscale 为 false 后命中。Host 资源失败即异常，无 fallback |
| P11 | R-009 NF-F16-NQ | FP16 升 FP32；H 分片两遍；无逐列 qscale；per-core workspace | `dtypeTag=1 && H>5120 && !hasQuantScale`；NF 尾轴因子可成立 | H0 FP16；H1 NF；H2 qscale 为 false 后命中。Host 资源失败即异常，无 fallback |

```text
ParseDType(dtypeTag)
  -> H > 5120 ? NF : FL
  -> hasQuantScale ? Q : NQ
  -> dtype-specific named entry
```

### Route 退化与失败关系

| 变化 | Route 关系 | 是否为 runtime fallback |
|---|---|---|
| `H` 穿过 5120 | 同 dtype/Q 标志在 FL 与 NF 之间切换；拓扑从整行 UB 变成分片两遍 + workspace | 否；这是 Host 主选择器，FL 资源失败不会再试 NF |
| `quant_scale` 有/无 | Q/NQ 使用不同 entry 与模板实例，分别编译进/裁掉 qscale Queue 消费和乘法 | 否；空指针只允许进入 NQ entry，Q entry 不做判空降级 |
| dtype 改变 | INT32 路线启用逐列 weight scale 和逐行 activation scale；FP16/BF16 只做升 FP32 | 否；三者由 Plugin 分发，不在 Device 中互转兜底 |
| UB/AIV/tiling 因子非法 | `CalcDirectTiling` 抛出 | 无其他 Device、scalar 或 registered-op fallback |
| entry 落在非 AIV core | 当前 core 立即返回 | 只是 core-type guard，不是替代计算路径 |

### 性能与资源 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| FL/NF 边界 | `H > 5120` 才 NF | FL 需整行 `2H` 与结果/scale 同驻 UB | 全部 | 另一实现族 | `CalcDirectTiling`, `FULL_LOAD_H_MAX` |
| AIV/UB | `C>0 && U>1056` | entry 只执行 AIV；tiling 至少保留 1024B 与 32B | 全部 | Host 异常 | `CalcDirectTiling` |
| FL 行因子 | `Fx>=1` | 全载一批行的 Queue/TBuf 必须落入 UB | P0 R-002、P1 R-001、P2 R-004、P3 R-006、P4 R-005、P5 R-003 | Host 异常 | `CalcDirectTiling` |
| NF 尾轴因子 | `L>0` 且 32 元素对齐 | INT8/FP32 搬运与保守 UB 切分 | P6 R-008、P7 R-007、P8 R-010、P9 R-012、P10 R-011、P11 R-009 | Host 异常 | `CalcDirectTiling` |
| 输入边界 | `1<=M<=65536, 1<=H<=8192` | POD、地址计算与实现覆盖域 | 全部 | `TORCH_CHECK`/Host 异常 | `CheckInputs`, `CalcDirectTiling` |
| dtype/scale 配对 | INT32 必有 dequant scales；16-bit 必无 | 模板编译期裁剪不同 dequant 分支 | 全部 | `TORCH_CHECK` | `CheckInputs` |
| AIV guard | `g_coreType==AIV` | 防止同 entry 在非 Vector core 执行 | 全部 | 当前 core 返回 | 12 个 kernel entry |

### POD/ABI

`DequantSwigluQuantV35BaseTilingData` 是 standard-layout、trivial；21 个 `int64_t` 后接 3 个 `float`。按 950 Host/Device 共用的自然对齐 ABI，`alignof=8`、`sizeof=184`；下表 offset 为字节。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| POD | `inDimx` | int64/64 | off 0 | row | `rows` | FL/NF 行循环 | `[1,65536]` |
| POD | `inDimy` | int64/64 | off 8 | element | `2*hidden` | GM 行 stride | `[2,16384]` |
| POD | `outDimy` | int64/64 | off 16 | element | `hidden` | y/workspace stride | `[1,8192]` |
| POD | `UbFactorDimx` | int64/64 | off 24 | row | FL `Fx`；NF 1 | FL 批行数 | `>=1` |
| POD | `UbFactorDimy` | int64/64 | off 32 | element | FL `H`；NF `L` | UB tile | `>=1` |
| POD | `usedCoreNum` | int64/64 | off 40 | AIV core | `usedCore` | NF block guard/workspace extent | `<=C` |
| POD | `maxCoreNum` | int64/64 | off 48 | AIV core | FL `usedCore`；NF `C` | 分核步长 | `>=usedCore` |
| POD | `inGroupNum` | int64/64 | off 56 | group | 0 | 模板无 group，不读 | inactive |
| POD | `quantMode` | int64/64 | off 64 | enum | 1 | selected body 不读 | 固定动态量化 |
| POD | `actRight` | int64/64 | off 72 | bool-as-int | `activateLeft?0:1` | act/gate offset | 0/1 |
| POD | `dstType` | int64/64 | off 80 | enum | 2 | output 已由 int8 模板固定 | 固定 int8 |
| POD | `roundMode` | int64/64 | off 88 | enum | 0 | FL/NF cast helper；int8 分支不据此分流 | 固定 0 |
| POD | `activateDim` | int64/64 | off 96 | axis | 1 | selected body 不读 | 二维尾轴 |
| POD | `loopTimesPerRow` | int64/64 | off 104 | loop | FL 0；NF `ceil(H/L)` | NF tile 循环 | NF `>=1` |
| POD | `tailPerRow` | int64/64 | off 112 | element | FL 0；NF `H-(loops-1)L` | NF 尾 tile | `(0,L]` |
| POD | `swiGluMode` | int64/64 | off 120 | enum | 0 | FL/NF 选择 contiguous halves | 固定 0 |
| POD | `biasMode` | int64/64 | off 128 | enum | 0 | bool bias 模板不读 | inactive |
| POD | `groupIndexMode` | int64/64 | off 136 | enum | 0 | bool group 模板不读 | inactive |
| POD | `quantIsOne` | int64/64 | off 144 | bool-as-int | qscale 有则 0，否则 1 | 当前动态 Q/NQ 模板不读 | 仅保留 ABI |
| POD | `speGroupType` | int64/64 | off 152 | enum | 0 | 无 group，不读 | inactive |
| POD | `isSpecialCoreCut` | int64/64 | off 160 | bool-as-int | 0 | 无 group，不读 | inactive |
| POD | `clampLimit` | float/32 | off 168 | value | 7.0 | `swiGluMode=0` 不读 | inactive |
| POD | `gluAlpha` | float/32 | off 172 | value | 1.702 | `swiGluMode=0` 不读 | inactive |
| POD | `gluBias` | float/32 | off 176 | value | 1.0 | `swiGluMode=0` 不读 | inactive；尾部 4B padding |

### Route 源码闭包

| 重要性 / Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| P0 R-002、P1 R-001、P2 R-004、P3 R-006、P4 R-005、P5 R-003 | dtype + `H<=5120` + q flag | `dequant_swiglu_quant_plugin.cpp::{CheckInputs,directCall}`；`dequant_swiglu_quant_kernel.cpp::{CalcDirectTiling,Launch*}` | `DequantSwigluQuantV35BaseTilingData` | `dequant_swiglu_quant_fl_*` | `DequantSwigluQuantBase::{Init,Process,ProcessSingleGroup,ProcessSingleGroupPerCore}` + common MicroAPI helpers | op CMake → top-level `all_kernels_obj/all_plugins_obj/_C` |
| P6 R-008、P7 R-007、P8 R-010、P9 R-012、P10 R-011、P11 R-009 | dtype + `H>5120` + q flag | 同上 | 同上 + workspace | `dequant_swiglu_quant_nf_*` | `DynamicNotFull::{Init,Process,ProcessSingleRow,...}` + `DequantAndSwiGlu/CalculateReduceMax/QuantAndCast` | 同上 |

## Kernel 实现

### 公共执行流程

每个 entry 首先做 AIV guard，创建栈上 `TPipe`，按 named entry 选择唯一模板实例，再以相同 POD 调用 `Init/Process`。两族都在 FP32 中完成 dequant、Swish、乘法、绝对值归约与除 scale；最后沿固定三段 cast 生成 INT8。Q Route 在归约前把逐列 `quant_scale` 乘入 FP32 中间值，且后续量化复用这个已缩放值。

FL 在一个 UB resident tile 内完成 `CopyIn → dequant/Swiglu → optional qscale → ReduceMax/scale → quant/cast → CopyOut`。NF 仍是单 Kernel，但对一行分两遍：第一遍逐 H tile 算 FP32 值、更新全行 scale 并写 per-core GM workspace；第二遍逐 tile 回读，用最终 scale 量化写 y。

### 融合顺序与不可交换项

```text
INT32: x_i32 --RINT cast--> x_f32 --*w[k]-->*a[row]--+
FP16/BF16: x_b16 ---------cast----------------------+
                                                       -> swish(act_f32) * gate_f32
                                                       -> [* qscale[k]]
                                                       -> abs/max/127 -> clamp_min(1e-12)
                                                       -> divide -> RINT int16 -> fp16 -> SAT int8
```

- `weight_scale` 在左右两半上各自逐列乘，`activation_scale` 对同一 row 的 act/gate 共用；二者必须在 Swish 之前完成，因为 Swish 非线性不允许把 scale 移到其后。
- `quant_scale` 必须在行 max 之前乘入，并将乘后值原地保留。若只影响 max 或只在最终除法前乘一次，会破坏 `scale` 与被量化值来自同一 `v` 的不变量。
- 对有限 `v`，`scale>=maxabs(v)/127`；因此理论上 `abs(v/scale)<=127`。最后 cast 的 `SAT` 是边缘舍入保护，不是用来弥补错误 scale 的常规截断策略。
- INT32→FP32 在 `|x|>2^24` 时可能失去整数低位；源码选择“先 cast，再乘两个 FP32 scale”，没有 FP64 或分段高精度反量化路径。
- `DIV_MODE` 启用高精度除法模式；Exp、乘法、归约也都保持 FP32。源码不做 NaN/Inf 特判，因此只对有限输入建立上述幅值不变量。

### UB 流水、同步与对象生命周期

| 家族 | 搬运/计算对象 | 源码同步 | 可安全覆盖时刻 | 实际流水边界 |
|---|---|---|---|---|
| FL | x/y Queue 各初始化 2 个 slot；inScale、scale 各 1 个 slot；tmp 为 TBuf | `AllocTensor→EnQue→DeQue→FreeTensor`；tmp 内阶段按 Vector 程序序 | x 在 Swiglu 后释放；scale 在量化完成后释放；y 在 MTE3 CopyOut 后释放 | row batch 顺序执行，源码没有展示跨 batch 的稳定态重叠；core 上多 batch 时存在 inScaleLocal 提前/重复释放边界 |
| NF 第一遍 | x/activation/y Queue 预留双 slot，w/q/scale 单 slot，tmp 单 TBuf | `S_MTE2`、`MTE2_V`、`V_MTE2`、`V_MTE3` HardEvent | tmp 只有在写 workspace 完成后才可被下一 tile 的 MTE2/Vector 复用 | 每 tile 显式 set/wait，重点是确定性生命周期，不应宣称已形成无等待搬运/计算重叠 |
| NF 两遍边 | per-core GM workspace、scaleLocal | `CopyOutScale` 后 `MTE3_MTE2` | 第二遍 MTE2 只能在第一遍整行 workspace 与 scale 输出的 MTE3 完成后开始 | 没有跨 core 同步；`blockIdx*H` 私有 slice 消除核间依赖 |
| NF 第二遍 | workspace→tmp→y | `MTE2_V`、`V_MTE2` + y Queue | tmp 被 QuantAndCast 读完、y 被 MTE3 搬出后进入下一 tile | 同一 core 必须完成当前 row 两遍后才可覆盖 slice 处理下一 row |

NF 的 `Process()` 以 `row=blockIdx; row<M; row+=maxCoreNum` 轮询；POD 中 NF `maxCoreNum=C` 而 launch `blockDim=min(C,M)`。因此 `M<C` 时每个已启动 core 至多一行，`M>=C` 时所有 C 个 core 以步长 C 分行；workspace 仍按已启动的 `usedCore` 分配，且 `blockIdx<usedCore` guard 保证 slice 不越界。

### 12 个 active 叶的完整实现差分

当前公开合法域的笛卡尔积是 3 种 dtype × 2 种 H family × 2 种 qscale 状态，12 个组合都由生产 Host selector 到达，均为 `active`，没有 `conditional` 叶。下表按知识重要性 P0→P11 排列，Host 的真实判断仍是 dtype→FL/NF→Q/NQ。

通用 body 中的 group_index/bias、`swiGluMode!=0`、静态量化/quant_offset 以及 FP8/FP4/HiFloat8 输出分支属于 `dormant` 能力，不是本卡的额外 Route：Plugin schema 不接收这些输入/属性，POD 固定 inactive 值，12 个 entry 也只实例化 bool group/bias 与 INT8 输出。重新启用任一分支都需要同时扩展 schema/校验/输出分配、Host POD producer、named entry 模板和 UB 预算，不能只改某个 POD 字段。

缩写：`w=weight_scale`、`a=activation_scale`、`q=quant_scale`、`s=scale`。FL 固定分配 `XA[2],IS[1],Y[2],S[1],TMP`：XA 是 x+可选 a 的 Queue，IS 是 w/q 共用 Queue；NF 固定分配 `X[2],W[1],A[2],Q[1],Y[2],S[1],TMP`。方括号是 InitBuffer slot 数；表内“active”只表示实际搬入/消费，模板裁掉的 Queue 容量仍计入 UB。

| 重要性 / Route | 精确谓词与 Host 叶 | entry / Device 模板 | active GM → 输出 | Queue active 差分 | workspace | 输入与 qscale 精度差分；共同输出 cast |
|---|---|---|---|---|---|---|
| P0 R-002 FL-I32-Q | `dtypeTag=0 && H<=5120 && hasQuantScale`；`LaunchInt32` | `fl_i32_q`；`Base<float,float,bool,bool,int32_t,int8_t>` | `x,w,a,q → y,s` | XA:`x+a`；IS:`w+q`；Y/S/TMP active | 无 | INT32 RINT→FP32，`*w[k]*a[row]`；Swiglu 后 `*q[k]` 再 max；共同三段 INT8 cast |
| P1 R-001 FL-I32-NQ | `dtypeTag=0 && H<=5120 && !hasQuantScale`；`LaunchInt32` | `fl_i32_nq`；`Base<float,bool,bool,bool,int32_t,int8_t>` | `x,w,a → y,s`；q 为 null | XA:`x+a`；IS:`w`；Y/S/TMP active | 无 | 同 R-002 的 dequant；无 q 乘法；共同三段 INT8 cast |
| P2 R-004 FL-F16-Q | `dtypeTag=1 && H<=5120 && hasQuantScale`；`LaunchHalf` | `fl_half_q`；`Base<bool,float,bool,bool,half,int8_t>` | `x,q → y,s`；w/a 为 null | XA:`x`；IS:`q`；Y/S/TMP active | 无 | `DIST_UNPACK_B16` FP16→FP32；`*q[k]` 再 max；共同三段 INT8 cast |
| P3 R-006 FL-BF16-Q | `dtypeTag=2 && H<=5120 && hasQuantScale`；`LaunchBfloat16` | `fl_bfloat16_q`；`Base<bool,float,bool,bool,bfloat16_t,int8_t>` | `x,q → y,s`；w/a 为 null | XA:`x`；IS:`q`；Y/S/TMP active | 无 | `DIST_UNPACK_B16` BF16→FP32；`*q[k]` 再 max；共同三段 INT8 cast |
| P4 R-005 FL-BF16-NQ | `dtypeTag=2 && H<=5120 && !hasQuantScale`；`LaunchBfloat16` | `fl_bfloat16_nq`；`Base<bool,bool,bool,bool,bfloat16_t,int8_t>` | `x → y,s`；w/a/q 为 null | XA:`x`；IS 无 active 输入；Y/S/TMP active | 无 | BF16→FP32；无 q 乘法；共同三段 INT8 cast |
| P5 R-003 FL-F16-NQ | `dtypeTag=1 && H<=5120 && !hasQuantScale`；`LaunchHalf` | `fl_half_nq`；`Base<bool,bool,bool,bool,half,int8_t>` | `x → y,s`；w/a/q 为 null | XA:`x`；IS 无 active 输入；Y/S/TMP active | 无 | FP16→FP32；无 q 乘法；共同三段 INT8 cast |
| P6 R-008 NF-I32-Q | `dtypeTag=0 && H>5120 && hasQuantScale`；`LaunchInt32` | `nf_i32_q`；`DynamicNotFull<float,float,bool,bool,int32_t,int8_t>` | `x,w,a,q → y,s` | X/W/A/Q/Y/S/TMP 全 active | `usedCore*H*4` | INT32→FP32、`*w*a`；第一遍 `*q` 后写 workspace；共同三段 INT8 cast |
| P7 R-007 NF-I32-NQ | `dtypeTag=0 && H>5120 && !hasQuantScale`；`LaunchInt32` | `nf_i32_nq`；`DynamicNotFull<float,bool,bool,bool,int32_t,int8_t>` | `x,w,a → y,s`；q 为 null | X/W/A/Y/S/TMP active；Q inactive | `usedCore*H*4` | INT32→FP32、`*w*a`；无 q 乘法；共同三段 INT8 cast |
| P8 R-010 NF-F16-Q | `dtypeTag=1 && H>5120 && hasQuantScale`；`LaunchHalf` | `nf_half_q`；`DynamicNotFull<bool,float,bool,bool,half,int8_t>` | `x,q → y,s`；w/a 为 null | X/Q/Y/S/TMP active；W/A inactive | `usedCore*H*4` | FP16→FP32；第一遍 `*q` 后写 workspace；共同三段 INT8 cast |
| P9 R-012 NF-BF16-Q | `dtypeTag=2 && H>5120 && hasQuantScale`；`LaunchBfloat16` | `nf_bfloat16_q`；`DynamicNotFull<bool,float,bool,bool,bfloat16_t,int8_t>` | `x,q → y,s`；w/a 为 null | X/Q/Y/S/TMP active；W/A inactive | `usedCore*H*4` | BF16→FP32；第一遍 `*q` 后写 workspace；共同三段 INT8 cast |
| P10 R-011 NF-BF16-NQ | `dtypeTag=2 && H>5120 && !hasQuantScale`；`LaunchBfloat16` | `nf_bfloat16_nq`；`DynamicNotFull<bool,bool,bool,bool,bfloat16_t,int8_t>` | `x → y,s`；w/a/q 为 null | X/Y/S/TMP active；W/A/Q inactive | `usedCore*H*4` | BF16→FP32；无 q 乘法；共同三段 INT8 cast |
| P11 R-009 NF-F16-NQ | `dtypeTag=1 && H>5120 && !hasQuantScale`；`LaunchHalf` | `nf_half_nq`；`DynamicNotFull<bool,bool,bool,bool,half,int8_t>` | `x → y,s`；w/a/q 为 null | X/Y/S/TMP active；W/A/Q inactive | `usedCore*H*4` | FP16→FP32；无 q 乘法；共同三段 INT8 cast |

共同输出 cast 是 `FP32 quotient → INT16(CAST_RINT,NO_SAT) → FP16(CAST_ROUND) → INT8(CAST_TRUNC,SAT)`。Q/NQ 差异由第二个模板实参 `float/bool` 编译期裁剪；INT32/16-bit 的 dequant 差异由第一个与第五个模板实参裁剪。任何叶在其资源规划失败后都不会跳到表中另一叶。

### FL body family：整行 resident、无 workspace

适用 P0～P5。Host/POD 固定关系是：

```text
Fx = min(M, floor((U - 1024 - 12*A8(H)) / Dfl))
usedCore = min(C, ceil(M/Fx))
POD = {
  inDimx=M, inDimy=2H, outDimy=H,
  UbFactorDimx=Fx, UbFactorDimy=H,
  usedCoreNum=usedCore, maxCoreNum=usedCore,
  loopTimesPerRow=0, tailPerRow=0,
  actRight=activateLeft ? 0 : 1
}
launch fl_{dtype}_{q|nq}<<<usedCore, nullptr, stream>>>
```

#### 核、行与 GM 地址

`ProcessSingleGroup(0,M,0)` 先计算 `B=ceil(M/usedCore)`、`realCore=ceil(M/B)`。仅 `blockIdx<realCore` 执行；普通 core 取连续 `B` 行，最后一个 core 取剩余行，再以最多 `Fx` 行为一个 batch。对 batch 起点 `r0`：

```text
actOff  = actRight * H
gateOff = H - actOff
xAct GM  = x[(r0+i)*2H + actOff : +H]
xGate GM = x[(r0+i)*2H + gateOff : +H]
wAct/wGate = w[actOff:+H], w[gateOff:+H]          # 仅 INT32
aRow = a[r0+i]                                     # 仅 INT32
qCol = q[0:H]                                      # 仅 Q
y GM = y[(r0+i)*H : +H], s GM = s[r0+i]
```

weight/q scale 由每个 core 搬入一次；x 与逐行 activation scale 按 batch 搬入。公开模板无 group/bias，因此所有 `groupIdx` 均为 0，左右半部与对应 weight scale 使用相同 offset。

#### UB 物理布局、流水与尾块

| 区域 | 初始化容量 | active producer→consumer | 尾块/释放 |
|---|---|---|---|
| XA Queue | 2 slots；每 slot `2*Fx*XX*sizeof(X)+Fx*8*4` | MTE2 搬 x；INT32 另搬 a → Vector | DataCopyPad 只搬 H；Swiglu 后释放当前 x tensor |
| IS Queue | 1 slot；`3*A8(H)*4` | INT32 的 w 占前 2 段；Q 的 q 占第 3 段 → Vector | 每个 VF 用 H full/tail mask；生命周期风险见下 |
| TMP TBuf | `Fx*A8(H)*4` | dequant/Swiglu → 可选 q 原地乘 → max/quant | Vector 程序序；16-bit stride 风险见下 |
| S Queue | 1 slot；`Fx*8*4` | max/127/floor → MTE3 与 broadcast quant | 每行只写 1 个有效 FP32；量化结束后释放 |
| Y Queue | 2 slots；每 slot `Fx*A32(H)` bytes | cast → MTE3 | mask 只生成 H 个值，DataCopyPad 只写 H bytes |

这里 `XX=A8(H)`（INT32）或 `A16(H)`（FP16/BF16）。x/y 的 2 slots 是物理容量；源码在每个 batch 内立即 EnQue/DeQue，未证明跨 batch 的搬运—计算稳态重叠。FL 不使用 L1/L0、SIMT、NDDMA 或通信窗口。

尾 VF 的有效宽度是 `H-V*(ceil(H/V)-1)`，Swiglu、q 乘法、Abs/Max、除法与 cast 都只对 mask 内 lane 生效；GM CopyOut 的 blockLen 也是精确 H，而不是对齐容量。

#### 关键叶 P0 R-002 的可落地伪码

R-002 同时覆盖 INT32 双 scale、左右选择和 qscale，是 FL 的超集叶；其余五叶由模板裁掉相应阶段：

```cpp
if (g_coreType != AIV) return;
Base<float, float, bool, bool, int32_t, int8_t> op(&pipe);
op.Init(x, w, a, nullptr, q, nullptr, nullptr, y, s, &pod);
copy(wAct, wGate, q);                     // 每 core 一次的 IS local
for (each row batch owned by blockIdx) {
    copy(xAct, xGate, aRow);
    tmp = swish(float(xAct)*wAct*aRow) * (float(xGate)*wGate*aRow);
    tmp *= q;                             // 原地，必须早于 max
    sRow = max(reduce_max(abs(tmp))/127.0f, 1.0e-12f);
    yRow = int8_cast_chain(tmp / sRow);
    copy_exact_H(yRow); copy_one(sRow);
}
```

#### FL 两项源码未闭合边界

1. **A8/A16 tmp 一致性**：Host/TBuf 按 `Fx*A8(H)` 个 FP32 分配 tmp，但 FP16/BF16 body 以 `i*A16(H)` 作为 tmp 行 stride。当 `A16(H)>A8(H)` 且 batch 行数大于 1 时，需要的末地址 `(rows-1)*A16(H)+H` 可能越过分配容量；源码没有 guard 或 padding 证明。
2. **多 batch inScaleLocal 生命周期**：只有 `ceil(M/Fx)<=C` 时每 core 才恒为一个 batch。若 AIV 核上限截断 usedCore，body 会多 batch；IS tensor 在循环外只 Alloc/DeQue 一次，却在每个 batch 末 Free，同一 local 后续继续使用并重复释放，源码没有闭合该域。

这两项是当前 body 的实现边界，不是新增 Route；资源或生命周期不满足时也没有转 NF/scalar fallback。

### NF body family：H 分片、单 Kernel 两遍

适用 P6～P11。Host 先按 dtype 取 `split=50`（INT32）或 `26`（FP16/BF16），再生成：

```text
L = floor((U-1056)/split/32)*32
loops = ceil(H/L)
tail = H - (loops-1)*L
usedCore = min(C,M)
POD = {
  UbFactorDimx=1, UbFactorDimy=L,
  usedCoreNum=usedCore, maxCoreNum=C,
  loopTimesPerRow=loops, tailPerRow=tail,
  actRight=activateLeft ? 0 : 1
}
workspaceBytes = usedCore*H*4
launch nf_{dtype}_{q|nq}<<<usedCore, nullptr, stream>>>
```

`L` 是 32 元素对齐的普通 tile；最后一 tile 用 `tail∈(0,L]`。INT32 的实际已分配 UB 账为 `34L+96`，16-bit 为 `26L+96`；Host 对 INT32 保留更保守的 50-way 上界，不能仅按实际账放大 L。

#### 核、workspace 与两遍伪码

NF 以 `row=blockIdx; row<M; row+=C` 轮询，每个已启动 core 独占 `workspace[blockIdx*H:(blockIdx+1)*H]`。以超集叶 P6 R-008 为例：

```cpp
if (g_coreType != AIV) return;
DynamicNotFull<float,float,bool,bool,int32_t,int8_t> op(&pipe);
for (row = blockIdx; row < M; row += C) {
    sLocal = 0;
    // pass 1: 先得到最终全行 scale，并保存“已乘 q”的同一物理值
    for (t = 0; t < loops; ++t) {
        n = (t == loops-1 ? tail : L);
        off = t*L;
        actOff  = actRight ? H+off : off;
        gateOff = actRight ? off : H+off;
        MTE2_copy(x halves, w halves, a[row], q[off:off+n]);
        wait(MTE2_V);
        tmp = swish(float(xAct)*wAct*a[row]) * (float(xGate)*wGate*a[row]);
        tmp *= q[off:off+n];
        sLocal = max(sLocal, maxabs(tmp)/127.0f, 1.0e-12f);
        wait(V_MTE3);
        MTE3_copy(workspace[blockIdx*H+off], tmp, n);
    }
    MTE3_copy(s[row], sLocal, 1);
    wait(MTE3_MTE2);                      // 第一遍 GM 对第二遍可见
    // pass 2: 只用最终 scale 量化
    for (t = 0; t < loops; ++t) {
        n = (t == loops-1 ? tail : L);
        MTE2_copy(tmp, workspace[blockIdx*H+t*L], n);
        wait(MTE2_V);
        yTile = int8_cast_chain(tmp / sLocal);
        MTE3_copy(y[row*H+t*L], yTile, n);
    }
}
```

NQ 叶移除 Q Queue 的搬入与 `tmp*=q`；FP16/BF16 叶再移除 W/A Queue 消费并把输入 load/cast 替换为 `DIST_UNPACK_B16→FP32`。地址、两遍顺序、workspace 和输出 cast 不变。

#### NF UB、事件和 owner 边界

| 区域 | 固定容量 | active 差分 | 同步/尾块 |
|---|---|---|---|
| X Queue | 2 slots，`2L*sizeof(X)`/slot | 全叶 active | tail 先按 X dtype 对齐到 UB，VF/GM 只消费 n |
| W Queue | 1 slot，`2L*4` | 仅 I32 active | 与 X 同一 MTE2_V 边 |
| A Queue | 2 slots，`8*4`/slot | 仅 I32 active | 搬 1 个 FP32，寄存器广播 |
| Q Queue | 1 slot，`L*4` | 仅 Q active | q 乘法用 full/tail mask并原地写 tmp |
| TMP TBuf | `L*4` | 全叶 active | `V_MTE2` 防下一搬入过早覆盖，`V_MTE3` 保护写 workspace |
| S Queue | 1 slot，`8*4` | 全叶 active | 跨 tile 更新；`MTE3_MTE2` 位于两遍边 |
| Y Queue | 2 slots，`L` bytes/slot | 全叶 active | cast 后精确搬 n bytes |
| GM workspace | 每 core H 个 FP32 | 全 NF 叶 active | `blockIdx*H+t*L`；当前 row 第二遍完成后才能覆盖 |

无跨 core 通信；workspace 私有 slice 消除了核间同步。第一遍的 `S_MTE2/MTE2_V/V_MTE2/V_MTE3` 和两遍间 `MTE3_MTE2` 是源码显式 HardEvent，不能仅用“Queue 自动同步”代替。NF 同样不使用 L1/L0、SIMT、NDDMA 或通信窗口。

Device 地址、容量和“同一 core 当前 row 两遍完成后可覆盖 slice”由源码闭合；workspace tensor 的完成期 owner 不由这张表推出。它是 Plugin 局部 tensor，lambda 只使用其 raw GM pointer，返回输出也不持有它；是否在异步 Kernel 完成前延迟回收属于 `RunOpApi`/NPU allocator 外部契约。

### 两个 body family 的不变量与 fallback

| 不变量 | FL | NF |
|---|---|---|
| 语义 | dequant/cast→Swish×gate→可选 q→全行 max/127/floor→同值量化 | 完全相同；仅用 workspace 两遍解决全行 max 依赖 |
| half selection | x 与 w 使用同一 `actRight` 半部；a 按 row | 相同，tile offset 额外加 `t*L` |
| block mapping | 连续行块，再按 Fx batch | blockIdx 起始、步长 C 的 row 轮询 |
| tail | H 的 VF mask + 精确 GM blockLen | 最后一 tile n=tail，再叠加 VF tail mask |
| workspace | entry 签名不接收；Plugin 32B 仅占位 | per-core H×FP32；完成期 owner 属框架边界 |
| 失败出边 | `Fx<1`、平台资源非法即 Host 异常；不尝试 NF | `L<=0`、workspace 分配/设备执行失败无 scalar fallback |

## 算子特定优化

### Trick T-001：整行常驻与分段两遍的 5120 边界

#### 触发与源码依据

- 适用 Route：P0 R-002、P1 R-001、P2 R-004、P3 R-006、P4 R-005、P5 R-003、P6 R-008、P7 R-007、P8 R-010、P9 R-012、P10 R-011、P11 R-009。
- 触发条件：`H<=5120` 选 FL，否则选 NF。
- 转换片段定位：`op_kernel/dequant_swiglu_quant_kernel.cpp:329-383`；相关官方源码：[`dequant_swiglu_quant_apt.cpp`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_apt.cpp)

**转换源码（主事实源；与官方映射边界分开）**：

```cpp
result.notFull = hidden > FULL_LOAD_H_MAX;
if (result.notFull) {
    // compute L, loops, tail and per-core workspace
}
```

#### 机制与设计原因

FL 把整行 H 的中间结果留在 UB，从而不物化 GM 中间值；当 H 超出固定边界，NF 把 H 切成 32 元素对齐 tile，并以 per-core workspace 保留第一遍结果，确保全行 scale 最终确定后才量化。官方固定 Commit 的 arch35 Host tiling也使用相同的 `H>5120` 分界，属于相关映射；direct named entry 与 1e-12 floor 是转换实现事实。

#### 迁移边界

- 适用：需要按整行 max 动态量化、且 H 可能超过单次 UB 常驻能力的融合算子。
- 前提：AIV、可查询 UB/core、NF 有 `coreNum*H*4` 用户 workspace。
- 失效：若 workspace 不可用或 L 为 0，不能退化为本实现未提供的 scalar fallback。

#### 错误做法

不能在第一遍的某个 tile 得到局部 max 后立即量化该 tile；后续 tile 可能抬高全行 scale，使同一行输出使用不一致的除数。第二遍必须等待最终 scale 与整行 workspace 对 MTE2 可见。

### Trick T-002：quant_scale 在 FP32 中间值上原地乘入

#### 触发与源码依据

- 适用 Route：P0 R-002、P2 R-004、P3 R-006、P6 R-008、P8 R-010、P9 R-012。
- 触发条件：`quant_scale` 存在，shape 为 `[1,H]`。
- 转换片段定位：`dequant_swiglu_quant_dynamic_not_full_vf.h:464-501`；相关官方源码：[`dequant_swiglu_quant_dynamic_not_full_vf.h`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant_dynamic_not_full_vf.h)

**转换源码（主事实源）**：

```cpp
if constexpr (HasQuantScale) {
    AscendC::MicroAPI::Mul(vreg0, vreg0, vreg1, maskTail);
    AscendC::MicroAPI::DataCopy(tmpXAddr, vreg0, maskTail);
}
```

#### 机制与设计原因

乘法直接覆盖 FP32 tmp，不增加第二个 H 大小的计算 buffer；随后 max 归约与量化读相同物理值，避免重新乘 qscale。FL 对应分支也在 `tmpXLocal` 原地覆盖。mask 来源于当前 H/tile 的 full/tail 元素数。

#### 迁移边界

- 适用：scale 变换必须同时影响动态量化幅值与最终被量化值。
- 前提：tmp 在归约和 cast 完成前不被别的阶段覆盖；qscale 为逐列 FP32。
- 失效：若 qscale 语义是除法、标量广播或不应影响动态 scale，不能复用此顺序。

#### 错误做法

不能先对未乘 q 的 tmp 求 max、再只在最终 cast 前乘 q；这样输出 scale 不再描述真正被量化的值。也不能把本接口的逐列乘法替换成通用 helper 中其他模式的除法语义。

### Trick T-003：NF workspace 按 core 而不是按 row 分配

#### 触发与源码依据

- 适用 Route：P6 R-008、P7 R-007、P8 R-010、P9 R-012、P10 R-011、P11 R-009。
- 触发条件：`H>5120`。
- 转换片段定位：`dequant_swiglu_quant_dynamic_not_full.h:305-390`；相关官方源码：[`dequant_swiglu_quant_dynamic_not_full.h`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant_dynamic_not_full.h)

**转换源码（主事实源）**：

```cpp
for (size_t rowIndex = blockIdx_; rowIndex < row_; rowIndex += coreNum_) {
    ProcessSingleRow(0, rowIndex);
}
uint64_t gmOffset = blockIdx_ * yLength_ + loop * blockLength_;
```

#### 机制与设计原因

每个 AIV core 串行处理自己负责的行，因此只需一个 `H` 长度的私有 slice；一行第二遍完成后即可被该 core 下一行覆盖。workspace 从 `M*H*4` 收缩到 `min(C,M)*H*4`，同时 blockIdx 隔离消除跨核写冲突。

#### 迁移边界

- 适用：同一 core 上的行处理串行、每行两遍闭合的实现。
- 前提：第二遍结束前不得调度同 core 的下一行覆盖 slice。
- 失效：若同一 core 并发多行或跨 Kernel 异步消费 workspace，必须扩大 slice 或增加更强生命周期协议。

#### 错误做法

不能在只分配 `usedCore*H` 元素时按 `rowIndex*H` 寻址 workspace；当一核轮询多行会越界。也不能在当前行第二遍结束前让下一行覆盖 `blockIdx*H` slice。

## 模板与实现对比

| 对比维度 | P0～P5 FL：R-002、R-001、R-004、R-006、R-005、R-003 | P6～P11 NF：R-008、R-007、R-010、R-012、R-011、R-009 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `H<=5120` | `H>5120` | 固定边界先决定拓扑，再按 dtype/q flag 选 named entry |
| 模板与 dtype | `DequantSwigluQuantBase<...>`；3 dtype × Q/NQ | `DynamicNotFull<...>`；3 dtype × Q/NQ | 编译期裁剪 dequant 与 qscale 分支 |
| 循环与分核 | 以 `Fx` 行为 UB tile，`usedCore=ceil(M/Fx)` 上限 C | `usedCore=min(C,M)`；每 core 轮询行，每行 H 分 tile 两遍 | 整行是否可常驻 UB |
| Queue/TBuf/寄存器/dynamic UB/通信 window | x/inScale/y/scale Queue + tmp TBuf + MicroAPI regs | x/w/a/q/y/scale Queue + tmp TBuf + regs | 无 dynamic UB、NDDMA 或通信 window |
| 内存总占用与复用 | 无 workspace；tmp 在同 tile 内复用 | `usedCore*H*4` workspace；per-core 逐行复用 | 全行 scale 依赖 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + MicroAPI cast/exp/div/mul/reduce | 同左 + workspace MTE2/MTE3 | 两者都是 AIV_VECTOR，非 SIMT/NDDMA/通信 |
| 同步、精度与 tail | TQue；FP32；VF mask | TQue + HardEvent；FP32；H tail tile + VF mask | NF 需要两遍 GM 可见性 |
| fallback/失败边界 | `Fx<1` 失败 | `L<=0` 失败 | 无 scalar 或其他 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方相关算子 README 标为支持；本 direct 交付未建立 A2 卡 | 官方相关算子 README 标为支持；本 direct 交付未建立 A3 卡 | 本卡目标；官方 README/config 与 direct `ascend950→dav-3510` 构建映射共同支撑 |
| Host/Launcher | 本卡不提取 | 本卡不提取 | named launcher + `PlatformAscendCManager` 动态查询 AIV/UB |
| Kernel/资源 | 本卡不提取 | 本卡不提取 | arch35 AIV_VECTOR FL/NF body；无 Cube/MIX_CV |

## 限制与失败边界

- 只覆盖二维 contiguous `x[M,2H]`、动态 INT8 输出、无 bias/group/quant_offset、`swiGluMode=0`；不是官方 ACLNN 全接口能力的替代说明。
- `M`、`H`、可选 scale 的 shape/dtype/device 不满足约束时在 Host 失败；unknown dtype、平台/UB 查询异常、`Fx/L` 为零也没有 Device fallback。
- entry 的非 AIV core 直接返回；所有可达计算节点均为 AIV_VECTOR，未发现 Cube/AIC 或 MIX_CV。
- FL 的无 qscale/16-bit 路线仍保留固定 inScale/activation-scale UB 容量，这是实际分配，不应按逻辑活跃输入扣除。
- FL 16-bit 路线的 tmp 分配/Host 字节账按 `A8(H)`，Device 多行寻址却按 `A16(H)`；`A16(H)>A8(H)` 且 resident row 数大于 1 时，源码层面未闭合容量证明。
- FL 在 AIV 核数截断 `usedCore`、单 core 需要多个 `Fx` row batch 时，inScaleLocal 在循环外只分配一次、循环内却每次释放；当前生命周期同样缺少多 batch 安全证明。
- NF Host 对 INT32 使用比实际 Queue 字节账更保守的 `split=50`；实现复用时不得只按 `34L+96` 放大 L 而绕过 Host 上界。
- Direct workspace 去除了官方 registered runtime 的 system-workspace 前缀；迁移回注册式入口时必须恢复对应 `GetUserWorkspace` 语义。
- 转换实现相对官方相关 body 增加 scale 的 `clamp_min(1e-12)`；该事实不得表述成官方固定 Commit 的原生语义。
- Host 不检查 scale 数值有限性，directCall 也不在 enqueue 后同步查询设备状态；异常值传播和异步设备错误均没有备用计算路径。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_launch.h` | 三个 launcher + workspace API | direct ABI |
| 入口实现 | `csrc/ops/dequant_swiglu_quant/op_plugin/dequant_swiglu_quant_plugin.cpp` | schema/Meta/PrivateUse1 | 校验、分配、stream、dtype dispatch |
| Host/Tiling/Launcher | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_kernel.cpp` | `CalcDirectTiling`, `Launch*`, 12 entries | Route、POD、blockDim、workspace |
| POD | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_tilingdata.h` | `DequantSwigluQuantV35BaseTilingData` | ABI 字段 |
| Kernel entry/body | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_full_load.h` | `DequantSwigluQuantBase` | FL body |
| Kernel helper | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_common.h` | cast traits、`DIV_MODE`、公共 MicroAPI helpers | FL/NF 公共精度与 Vector 参数 |
| SDK 依赖边界 | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_local_deps.h` | `basic_api/kernel_vec_intf.h` | 稳定 SDK include 闭包 |
| Kernel entry/body | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_dynamic_not_full.h` | `DynamicNotFull` | NF pipeline/workspace/sync |
| Kernel helper | `csrc/ops/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_dynamic_not_full_vf.h` | `DequantAndSwiGlu`, `CalculateReduceMax`, `QuantAndCast` | NF MicroAPI |
| 构建接线 | `csrc/ops/dequant_swiglu_quant/CMakeLists.txt` | `register_direct_launch_op` | op 源文件注册 |
| 构建接线 | `CMakeLists.txt` | `NPU_ARCH`, `all_kernels_obj`, `_C` | 950 编译映射与共享库 |
| 官方相关映射 | `quant/dequant_swiglu_quant/README.md` | 产品支持表 | Ascend 950 支持声明 |
| 官方相关映射 | `quant/dequant_swiglu_quant/op_host/dequant_swiglu_quant_tiling_arch35.cpp` | `DoOpTiling/DoOpTilingNotFull` | 5120 边界、UB/workspace 对应关系 |
| 官方相关映射 | `quant/dequant_swiglu_quant/op_kernel/dequant_swiglu_quant_apt.cpp` | `dequant_swiglu_quant` | AIV guard 与 arch35 router |
| 官方相关映射 | `quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant.h` | `DequantSwigluQuantBase` | FL body 相关映射 |
| 官方相关映射 | `quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant_dynamic_not_full.h` | `DynamicNotFull` | NF body 相关映射 |
| 官方相关映射 | `quant/dequant_swiglu_quant/op_kernel/arch35/dequant_swiglu_quant_dynamic_not_full_vf.h` | VF helpers | MicroAPI 相关映射 |
| 官方相关映射 | `quant/dequant_swiglu_quant/op_host/config/ascend950/dequant_swiglu_quant_binary.json` | `DequantSwigluQuant` binary entries | 950 注册映射 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 值传递 plain POD | 字段顺序、64/32 位宽、184B 布局；inactive 字段仍初始化 | `DequantSwigluQuantV35BaseTilingData`, `CalcDirectTiling` |
| 调用顺序 | schema→PrivateUse1→launcher→entry→body | dtype 与 q flag 必须选择匹配模板 | `directCall`, `Launch*` |
| workspace/内存 | FL 0；NF per-core H×FP32 | NF offset=`blockIdx*H`、extent=`usedCore*H*4`、两遍完成前不可覆盖；完成期 owner 依赖 `RunOpApi`/allocator 外部契约 | `SetGlobalTensor`, `ProcessSingleRow` |
| 同步 | FL Queue；NF Queue + HardEvent | NF `MTE3_MTE2` 必须位于两遍之间 | `ProcessSingleRow` |
| 精度与尾块 | FP32 compute/reduce，scale floor，固定 INT8 cast 链 | qscale 乘入必须早于 max；tail mask 只覆盖有效元素 | `CalculateReduceMax`, `QuantAndCast`, FL Vector scope |
| 失败边界 | Host 资源/shape/dtype 失败 | 不得虚构 scalar fallback；非 AIV entry 返回 | `CheckInputs`, `CalcDirectTiling`, entry guard |