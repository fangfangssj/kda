---
type: Operator
title: "QuantMatmul 950 Direct Invoke 算子知识"
description: "说明 cann_bench::quant_matmul 在 950 上的 shape/量化契约、Host POD 与 checked arithmetic、CATLASS/官方 OTF/raw-safe Cube 路由、分层数据流、尾块、精度和 fallback。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_api/quant_matmul_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_cube_basic_api_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_basic_api_cmct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/matmul/common/cmct/block/block_mmad_a8w8_fixpipe_quant.h
    role: implementation
created_at: "2026-08-24T19:34:50Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# QuantMatmul 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::quant_matmul`（Torch schema）→`quant_matmul_npu`→`launch_quant_matmul` |
| 数学主干 | contiguous ND `INT8×INT8`，以 INT32 累加；INT32 bias 在反量化前加入，浮点 bias 在反量化链末加入 |
| 输出 | contiguous ND FP16/BF16，shape 为右对齐广播 batch + `[M,N]` |
| 可达 Route | P0 raw Cube + ordered AIV（含 official raw producer 与 safe raw fallback）；P1 CATLASS fused per-channel；P2 official scalar OTF |
| 不可达编译体 | Basic、packed-scale producer/consumer、MIX_CV entry 仍可能参与编译，但当前 launcher 没有到它们的分发边 |
| 核心内存 | GM；raw Route 的 `[batchC,M,N]` INT32 workspace；Cube 的 L1/L0A/L0B/L0C；CATLASS/AIV epilogue 的 UB |
| 平台 | Ascend 950 / dav-3510；Direct Invoke，不是 ACLNN 调用 |

### 来源与 Direct 适配边界

- 来源范围：Frontmatter 仅固定官方 QuantBatchMatmulV3 相关实现，该 Commit 的本地文件已恢复并核对。Direct 交付快照及其 CATLASS 文件不在这些官方 sources 中；下文保留交付实现的文件定位，复用前仍需取得相应快照并固定内容哈希，不能用官方文件身份核对替代 Direct 实现复核。
- 官方原生事实：固定版本 QuantBatchMatmulV3 arch35 源码支撑 `MatMulASWKernel`、ASW batch/tail 调度、FP19 Fixpipe scalar 与 tiling POD；128×256×256 / 128×256×128 ping-pong MMAD 和 typed dequant epilogue 则属于交付记录中的 CATLASS 实现，不据本卡官方 sources 声称其已核验。
- Direct Invoke 适配：Torch schema、输入校验、100 B Host POD、workspace 计算、current-stream launcher、Route selector 与 CATLASS Params 组装是本入口的 Host 边界，不等同于官方 ACLNN 或生产 registry 分发。
- Device 派生路线：`SafeRawMatmul`、无 padding INT32 workspace 和 `SupplementalEpilogue` 是为公开 shape/有序后处理契约补充的 Device 路线；它们复用官方 Matmul/AscendC primitive，但不是官方原生 entry。

## 语义、shape、layout 与 transpose 契约

### 数学次序

令 `b` 为输出的展平 batch，`bA/bB` 为右对齐广播后对应的输入 batch，`biasBatch` 为 0 或 `b`：

```text
acc[b,m,n] = Σ(k) int32(x1[bA,m,k]) * int32(x2[bB,k,n])
if bias.dtype == INT32:
    acc += bias[biasBatch,n]

v = fp32(acc) * fp32(scale[scale_count == 1 ? 0 : n])
if offset:          v += offset[scale_count == 1 ? 0 : n]
if pertoken_scale:  v *= pertoken_scale[m]
if floating_bias:   v += fp32(bias[biasBatch,n])
y = cast_rint<FP16/BF16>(v)
```

`pertoken_scale` 没有 batch 维，对每个 batch 重用同一 `[M]` 向量。bias 的 batch 规则不是任意广播：只能共享一份，或按展平顺序与 `batchC` 一一对应。

### 输入输出契约

| 参数 | shape | dtype | 必要约束 |
|---|---|---|---|
| `x1` | rank 2..6，`[...,M,K]` | INT8 | strided + contiguous；`M,K>=1` 且各自不超过 `INT32_MAX` |
| `x2` | rank 2..6，`[...,K,N]` | INT8 | 与 `x1` 同设备、contiguous；K 必须相等；`N>=1` 且不超过 `INT32_MAX` |
| `scale` | `[1]` 或 `[N]` | FP32/BF16 | contiguous；BF16 scale 只允许 BF16 输出 |
| `offset` | 可选 `[1]` 或 `[N]` | FP32 | 元素数必须与 scale 相同，即两者同为 scalar 或同为 per-channel |
| `pertoken_scale` | 可选 `[M]` | FP32 | contiguous；跨 batch 复用 |
| `bias` | `[N]`、`[1,N]` 或 `[...,1,N]` | INT32/FP32/FP16/BF16 | leading product 只能为 1 或 `batchC`；最后一维必须为 N |
| `output_dtype` | 可选字符串 | `float16`/`bfloat16` | 默认 `float16` |
| `y` | broadcast batch + `[M,N]` | FP16/BF16 | 新分配 contiguous ND Tensor |

### batch 广播和展平

`x1/x2` 最多有四个 batch 维。Host 将两者 batch 维右对齐，缺失轴补 1；每轴要求 `a==b || a==1 || b==1`，输出轴取 `max(a,b)`。Device safe raw 用输出坐标逐轴还原输入坐标：输入轴为 1 时坐标固定为 0，否则取输出坐标。

```text
out_coord = unravel(flat_batch, batch_c_dims[4])
in_coord[i] = input_dims[i] == 1 ? 0 : out_coord[i]
flat_input = ravel(in_coord, input_dims[4])
```

这套映射是 shape 语义的一部分，不能用 `flat_batch % batch_a` 或仅比较 batch product 替代；不同四维 shape 可以有相同 product，却有不同广播结果。

### transpose 与物理 layout

公开接口没有 `transpose_x1/transpose_x2` 属性。全部可达实例都固定：

```text
x1: RowMajor/ND [M,K], aTrans=false
x2: RowMajor/ND [K,N], bTrans=false
y : RowMajor/ND [M,N]
```

Host 不调用 `transpose`、`TransData` 或 `Contiguous`。一个普通转置 view 通常不连续，会直接校验失败；即使调用方把转置结果重新 contiguous，它也只会被当作新的逻辑 `[M,K]` 或 `[K,N]` Tensor，不会触发转置 kernel。CATLASS、official OTF 和 safe raw 的 layout/stride 必须一起保持非转置，不能只改 `SetTensorA/B(..., false)` 中的布尔值。

## Host、checked arithmetic 与生命周期

### Host 流程

```text
CheckDenseNpuTensor / dtype / rank / K match
 -> right-aligned batch broadcast
 -> scale, offset, pertoken, bias, output dtype checks
 -> fill QuantMatmulHostParams
 -> allocate y
 -> quant_matmul_workspace_bytes(params)
 -> allocate uint8 workspace (at least 32 B)
 -> capture pointers + params by value
 -> current NPU stream
 -> launch_quant_matmul
```

Host 不读取 tensor 数值，也没有 CPU/ATen 数学 fallback。

### checked arithmetic

| 位置 | 检查 | 目的 |
|---|---|---|
| 单维 | batch 维及 M/N/K 必须为正且 `<=INT32_MAX` | 安全写入 uint32 Device ABI，同时避免有符号 shape 越界 |
| batch product | `value <= UINT32_MAX/factor` | `batch_a/b/c` 可写入 uint32 |
| output numel | product `<=INT64_MAX/sizeof(uint16_t)` | `torch::empty` 的有符号 allocator 范围 |
| raw workspace | 依次 checked-mul `batchC*M*N*4` 于 uint64 | 防止 workspace 字节数静默回绕 |
| allocator 边界 | `workspace_bytes<=INT64_MAX` | 转为 Tensor shape 前保持可表示 |
| tile/grid | M/N tile product和 batch tile product用 uint64 checked-mul | 防止 block grid 公式回绕 |

`M<=4096、K<=16384、N<=65535、batchC<=256` 是 fast envelope，不是公开 shape 上限。超出 envelope 的合法输入会选择 safe raw；但仍可能因为 workspace、L1、UB 或 allocator 资源不足而在 Host 阶段失败。

### Host POD

`QuantMatmulHostParams` 固定为 25 个 `uint32_t`，`sizeof=100`、`alignof=4`，launcher 先复制 `*params`，Device 再按值接收。

| offset | 字段 | 含义 |
|---:|---|---|
| 0/4/8 | `m,n,k` | 逻辑矩阵维度，单位为元素 |
| 12/16/20 | `batch_a,batch_b,batch_c` | 展平 batch product |
| 24/40/56 | `batch_a_dims[4]`,`batch_b_dims[4]`,`batch_c_dims[4]` | 右对齐四轴 batch shape |
| 72 | `scale_count` | 1 或 N |
| 76/80/88 | `scale_dtype`,`bias_dtype`,`output_dtype` | 0=FP32，1=FP16，3=INT32，27=BF16，`0xffffffff`=无 bias |
| 84 | `bias_batches` | 1 或 batchC |
| 92/96 | `has_offset`,`has_pertoken` | 0/1 presence flag |

不要把 C++ `bool` 放进该 POD，也不要重排 dtype 与 presence 字段；AIV epilogue 按固定 offset 读取整份结构。

### workspace 与 stream 生命周期

```text
CATLASS fused          : logical 32 B placeholder
official scalar OTF    : logical 32 B placeholder
official/safe raw+AIV  : batchC * M * N * sizeof(int32_t)
```

raw workspace 从 offset 0 开始，严格是无行 padding 的 ND `[batchC,M,N]`。raw AIC 先写 workspace，AIV 后读；两个 launch 在同一 current stream，顺序构成交接。workspace Tensor 由框架 allocator 持有，源码没有显式 event、`recordStream` 或手工 free，异步存活依赖 `RunOpApi` 与 NPU allocator 的 stream 语义。

## Route 选择与退化关系

### Route 重要性

下表按知识复用价值排序；Host 顺序另列，不能把文档重要性改写成 selector 顺序。

| 重要性 | Route | 适用特征 | 价值 | 边界/退化 | Host 顺序 |
|---:|---|---|---|---|---:|
| P0 | R-001 official raw Cube + AIV | fast envelope、无 batch broadcast、无 offset、`M%16=N%32=K%32=0`，且未命中 fused/scalar | 以官方 ASW Cube 物化 INT32，再统一承接 per-channel、pertoken、bias 等有序后处理 | 不满足 raw producer envelope 时退到 R-002；没有 device-error 后的运行时重试 | 3a |
| P0 | R-002 safe raw Cube + AIV | 其余未命中前两条的合法输入；尤其 batch broadcast、offset、任一对齐尾块、超 fast envelope | 独立 tile ownership、64-bit GM offset、显式广播；是完整 shape/后处理域的选择期 fallback | 需要 O(batchMN) workspace；资源失败直接报错 | 3b |
| P1 | R-003 CATLASS fused per-channel | 单 batch；`M>=128`、`N%256=0`、`K%256=0`；scale `[N]`；无 bias/offset/pertoken | 单 kernel 内 AIC MMAD + AIV FP32 dequant，避免 raw GM 中间面 | 任一谓词不满足继续 scalar 或 raw，不转成 packed-scale | 1 |
| P2 | R-004 official scalar OTF | fast envelope；A/B batch shape都与 C 完全相同；scale `[1]`；无 bias/offset/pertoken | L0C→Fixpipe→GM，workspace 最小 | batch 广播或附加后处理转 raw；scalar scale 使用 FP19 Fixpipe 表示 | 2 |

### 精确 selector

```text
catlass = fast_envelope
    && batch_a == batch_b == batch_c == 1
    && M >= 128 && N % 256 == 0 && K % 256 == 0
    && scale_count == N
    && bias == none && !offset && !pertoken

scalar_otf = fast_envelope
    && batch_a_dims == batch_c_dims
    && batch_b_dims == batch_c_dims
    && scale_count == 1
    && bias == none && !offset && !pertoken

if catlass: launch R-003; return
if scalar_otf: launch R-004; return

use_safe_raw = !fast_envelope
    || has_batch_broadcast || offset
    || K % 32 != 0 || N % 32 != 0 || M % 16 != 0

launch (use_safe_raw ? R-002 : R-001) raw producer
launch ordered AIV epilogue
```

`fast_envelope` 定义为 `M<=4096 && K<=16384 && N<=65535 && batchC<=256`。scale/output dtype 组合已经在入口校验，因此 CATLASS 的三个模板实例只需覆盖 FP32→FP16、FP32→BF16 与 BF16→BF16。

### Active/conditional Route 闭包矩阵

下表只列 launcher 可达的 active/conditional Route；详细伪码和资源公式仍以后的对应章节为准。

| Route | selector / workspace | POD、entry 与 template | L1/L0/UB/GM 数据流 | tail、precision 与 fallback |
|---|---|---|---|---|
| R-003 CATLASS fused | `IsCatlassAligned`；32 B placeholder | `KernelT::Params`；`quant_matmul_catlass_{f32_f16,f32_bf16,bf16_bf16}`；`KernelMatmulDequant<BlockMmadTla<MmadPingpong<...>>,BlockEpilogueDequant,BlockSchedulerAswt>` | GM ND→L1 A/B 双缓冲→L0A/B 双缓冲→L0C INT32→共享 UB；scale GM→UB→FP32 AIV→双缓冲 UB→GM | 无余数型 N/K tail，M tail及可选 M/N split；typed scale 保留到 FP32 向量乘，RINT 输出；selector miss 依次检查 R-004、raw |
| R-004 scalar OTF | `IsScalarOnTheFly`；32 B placeholder | by-value `MixTiling`；`quant_matmul_otf_{f32_f16,f32_bf16,bf16_bf16}`；`RunOfficialOnTheFly<ScaleT,OutT>`→`MatMulASWKernel<...,FusedOpType::NONE>` | GM ND→Matmul L1/L0A/L0B→L0C INT32→Fixpipe→GM；无独立 UB epilogue | ASW `SetSingleShape` 承接 M/N/K tail；scale 截为 FP19，Fixpipe cast；selector miss 进入 raw，不运行时重试 |
| R-001 official raw + AIV | `!UseSafeRaw` 且前两条 miss；`batchC*M*N*4` | producer: by-value `MixTiling`，`quant_matmul_otf_raw_i32`→`RunOfficialOnTheFly<float,int32_t>`；consumer: by-value `QuantMatmulHostParams`，九个 `quant_matmul_epi_*`→`SupplementalEpilogue::Kernel<ScaleT,BiasT,OutT>` | producer GM ND→L1/L0→L0C INT32→GM raw workspace；consumer raw/scale/offset/pertoken/bias GM→48 KiB UB→FP32 vector→GM | producer 限 16/32 对齐域，AIV 处理 2048-N tail且只写 count；完整 FP32 ordered chain；producer envelope miss 在 launch 前改选 R-002 |
| R-002 safe raw + AIV | `UseSafeRaw` 且前两条 miss；`batchC*M*N*4` | producer: by-value `QuantMatmulHostParams+TCubeTiling`，`quant_matmul_safe_raw_i32`→`SafeRawMatmul::Kernel`→`MatmulImpl`；consumer 与 R-001 相同 | 每个 AIC 独占 `(batch,mTile,nTile)`，GM ND→L1/L0→L0C INT32→GM raw；随后同一 48 KiB AIV epilogue | actual M/N、Matmul K tail、四轴 broadcast与64-bit GM offset；FP32 ordered chain；合法域的最终选择期 fallback，资源/Device 失败不再退化 |

九个 AIV entry 覆盖完整合法 dtype 条件域：FP32 scale→FP16、FP32 scale→BF16、BF16 scale→BF16 各按 bias BF16、bias FP16、`{FP32,INT32,none}` 分三支。第三支的 `BiasT=float` 同时承载 FP32 bias；`p.bias_dtype` 决定实际执行浮点加法还是跳过 none/INT32，INT32 bias 已在 raw Cube 中注入。

### Dormant 编译体（与 active Route 分离）

| 符号组 | 状态与边界 |
|---|---|
| `quant_matmul_basic_*`、`quant_matmul_basic_raw_i32` | Basic/CMCT body 可编译，但 `launch_quant_matmul` 没有调用边，不消费当前 selector 的 `BasicTiling` |
| `quant_matmul_pack_scale_*`、`quant_matmul_pack_scale_redundant_*`、`quant_matmul_otf_qparam_*`、`quant_matmul_otf_packed_*` | packed-scale producer/consumer 保留为编译体；workspace selector 不生成 packed ABI，launcher 不分发 |
| `quant_matmul_mix_*` | MIX_CV body 保留，但 registry graph schedule 未在 Direct launcher 重建，因此不可达 |

### Route 间不变量

- A/B/C 始终是非转置 ND RowMajor；任何 Route 都不得私自改变逻辑 K 轴。
- INT32 bias 始终在反量化前；浮点 bias 始终在 `scale→offset→pertoken` 之后。
- raw producer 不得提前量化或 cast；AIV 只消费无 padding `[batchC,M,N]` INT32。
- CATLASS/scalar 的 32 B workspace 不能与 raw workspace 公式混用；workspace selector 必须和 launch selector使用同一谓词。
- fallback 是 Host 选择期的路线退化，不是前一 kernel 失败后的重试；异步 device 错误不会切换 Route。
- Route 在实数语义上等价，但不承诺 bitwise 一致：scalar Fixpipe 截断 scale 到 FP19，CATLASS/raw AIV 使用 FP32 运算链。

## Tiling 与 launch POD

### official OTF / raw 的 Mix tiling

```text
baseM = min(align_up(M,16), 256)
baseN = min(align_up(N,32), 256)
baseK = min(align_up(K,32), 128)
MNblocks = ceil(M/baseM) * ceil(N/baseN)
usedCoreNum = min(AIC, MNblocks)
```

先从 L1 总量扣除 bias 与 per-channel scale 预留：

```text
biasBytes  = bias ? baseN * sizeof(bias_dtype) : 0
scaleBytes = scale_count == 1 ? 0 : baseN * sizeof(scale_dtype)
leftL1 = L1 - biasBytes - scaleBytes
stepKa = floor((leftL1 / 2 / (baseM*baseK)) / 2)
stepKb = floor((leftL1 / 2 / (baseN*baseK)) / 2)
stepK* = min/adjust to K loops, then made mutually divisible
depthA1 = 2*stepKa; depthB1 = 2*stepKb
```

`dbL0A=dbL0B=2`。`dbL0C` 仅在 `2*baseM*baseN*4<=L0C` 且 bias/scale guard 允许时取 2，否则取 1。`adaptiveSlidingWin` 的 M/N tail 标志和 split count 均初始化为 1；未使用字段保持零初始化。

`MixTiling` 按值传入 kernel：

| 子结构 | Host 写入 | Device 消费 |
|---|---|---|
| `params` | batch product/四轴 dims、`isPerTensor`、`isPertoken`、bias dtype/batch flag | ASW batch offset、scale 模式和 epilogue 元数据 |
| `matmulTiling` (`TCubeTiling`) | M/N/K、base、singleCore、step/depth、usedCore、bias、double buffer | `MatmulImpl`、ASW scheduler、safe raw tile shape |
| `tileL2cacheTiling` | 零初始化 | 当前可达 body 不依赖 |
| `adaptiveSlidingWin` | tail/split 六字段 | ASW block scheduler |

`TCubeTiling` 是 CANN SDK ABI，跨版本不得凭字段同名假设二进制兼容；需要与编译使用的 9.1.0 header 一起固定。

### CATLASS Params

CATLASS 不消费 `MixTiling`。Host 由模板 `ToUnderlyingArguments` 生成 by-value `Kernel::Params`：

```text
problemShape = {M,N,K}
mmad = {x1,x2,y}
epilogue = {
  y, x2Scale=scale, x1Scale=null, bias=null,
  baseM=128, baseN=256,
  x1QuantMode=DEFAULT, x2QuantMode=PERCHANNEL,
  isBiasEpilogue=false
}
```

该 Params 含模板类型和嵌套枚举，不应手写 byte offset；正确做法是由同一 CATLASS 版本的 `Arguments→Params` 生成。

### block/core 映射

| Route | blockDim | job 映射 |
|---|---|---|
| R-003 CATLASS | `min(AIC, ceil(M/128)*ceil(N/256))` | ASWT scheduler 以 128×256 L1 tile 在 M/N 网格蛇形遍历，空闲核可分割尾 tile |
| R-004 scalar OTF | `usedCoreNum` | official ASW 每核多轮处理 M/N tile；batch 在 kernel 内循环 |
| R-001 official raw | `usedCoreNum` | 与 scalar OTF 相同，输出改为 INT32 workspace |
| R-002 safe raw | `min(AIC, batchC*ceil(M/baseM)*ceil(N/baseN))` | grid-stride job 解码为 `(batch,mTile,nTile)`，每个 tile 只有一个 AIC owner |
| raw 后 AIV | `min(AIV, batchC*M*ceil(N/2048))` | grid-stride job 解码为 `(flatRow,nTile)` |

## Kernel 数据流

### R-003 CATLASS fused per-channel

固定 tile：L1=`128×256×256`，L0=`128×256×128`；A/B 的 L1 与 L0 都是两级 ping-pong，L0C 为一份 128×256 INT32 tile。

```text
GM A/B ND
 -> MTE2, L1 A/B ping-pong
 -> MTE1, L0A/L0B ping-pong
 -> MMAD, L0C INT32
 -> shared UB INT32 tile
 -> AIV: GM scale[N] -> UB, int32->fp32, mul, cast_rint
 -> output UB double buffer -> MTE3 -> GM y ND
```

资源账：

```text
L1 A = 2 * 128 * 256 * 1 B = 64 KiB
L1 B = 2 * 256 * 256 * 1 B = 128 KiB
L0A  = 2 * 128 * 128 * 1 B = 32 KiB
L0B  = 2 * 128 * 256 * 1 B = 64 KiB
L0C  = 1 * 128 * 256 * 4 B = 128 KiB

mVec = ceil(128 / taskRatio)
UB int32 share = mVec * 256 * 4
UB scale       = 256 * sizeof(scaleT)
UB out double  = 2 * ceil(mVec/4) * 256 * sizeof(outT)
```

AIC/AIV 通过 CrossCore flag 交接 UB；AIC 在复用 UB 前等待 AIV，AIV 在 dequant 前等待 AIC。AIV 内又以 `V_MTE2/MTE2_V` 保护 scale、以两组 `V_MTE3/MTE3_V` 保护输出双缓冲。

CATLASS 要求 N/K 已是完整 256 倍数，因此没有余数型 N/K tail；M 末块可小于 128。scheduler 将最后一个 tile 收缩为实际 M，并在空闲核较多时进一步沿 M/N split；AIV 按 task ratio 分半行，零行 sub-block 直接返回。GM 写回使用 `blockLen=actualN*sizeof(outT)`、`dstStride=(N-actualN)*sizeof(outT)`。

### R-004 official scalar OTF

```text
GM A/B ND -> official Matmul L1/L0 -> L0C INT32
scalar scale GM -> high-19-bit Fixpipe quant scalar
L0C -> Fixpipe dequant/cast -> GM y ND
```

FP32 scale 的位模式与 `0xffffe000`；BF16 scale 左移到 FP32 高 16 位后再走相同 scalar 表示。`SetSingleShape(actualM,actualN,K)` 在每个 ASW tile 前设置 tail，零 M/N tile跳过。K 的内部循环和 tail 由 `MatmulImpl` 消费 `TCubeTiling`；输出 stride 为逻辑 N。该 Route无 bias、offset、pertoken，也不物化 UB epilogue。

### R-001 official raw producer

模板把 `yType` 改为 INT32。official body 因而跳过 `SetQuantScalar/SetQuantVector`，只做：

```text
GM A/B ND -> L1/L0 -> MMAD INT32 (+ optional INT32 bias)
 -> GM workspace[batch,m,n] INT32
```

`MakeMixTiling(raw=true)` 只允许 INT32 bias 设置 `isBias`；浮点 bias 不进入 Cube。ASW 负责 batch 和 M/N tile，alignment guard 把不完整的 16/32 物理尾块留给 safe raw。

### R-002 safe raw producer

safe raw 使用 `MatmulImpl<GM ND int8, GM ND int8, GM ND int32>`。每核 grid-stride 取得独立 tile，先用四轴广播函数求 `batch_a/b`，再使用 64-bit 元素 offset：

```text
A = (batchA*M + m0) * K
B = batchB*K*N + n0
C = (batchC*M + m0) * N + n0
actualM = min(baseM, M-m0)
actualN = min(baseN, N-n0)
SetSingleShape(actualM, actualN, K)
SetTensorA(..., false); SetTensorB(..., false)
optional SetBias(INT32 bias[biasBatch*N+n0])
IterateAll(C, 0); End()
```

该 ownership 不需要跨 AIC barrier，也不依赖 registry 的混合核调度元数据。M/N/K 非对齐、batch 广播、offset 和超 fast envelope 都选择此 producer；它是 shape-safe fallback，不是低精度 fallback。

### raw Route 的 ordered AIV epilogue

AIV tile 固定 `kTileN=2048`，一块 48 KiB UB 被划为六个 2048×4 B 逻辑区，并按 barrier 顺序别名复用：acc、FP32 value、FP32 operand、scale source、bias source、output。

```text
for job = blockIdx; job < batchC*M*ceil(N/2048); job += blockNum:
    row, n0 = decode(job)
    count = min(2048, N-n0)
    aligned = align_up(count, 8)
    DataCopyPad INT32 GM -> UB
    cast_rint INT32 -> FP32
    mul scalar/per-channel scale
    add scalar/per-channel offset if present
    mul pertoken[row % M] if present
    add shared/batched floating bias if present
    cast_rint FP32 -> OutT
    DataCopyPad exactly count elements -> GM
```

所有向量计算使用 `aligned` lane，输入 tail 由 `DataCopyPad` 补齐；GM 输出的 `blockLen` 只含 `count*sizeof(OutT)`，不会写越界的对齐 lane。AIC→AIV 只依赖同 stream launch order；AIV 内使用 `PIPE_ALL/PIPE_V` barrier 保护 GM copy、cast、mul/add 和 UB 别名生命周期。

## 精度、溢出与尾块总表

| Route | accumulator | scale 计算 | 后处理顺序 | cast | tail |
|---|---|---|---|---|---|
| CATLASS | INT32 L0C→UB | typed FP32/BF16 scale 转 FP32后向量乘 | 仅 scale | FP32→OutT RINT | M tail；N/K 由 selector排除 |
| scalar OTF | INT32 L0C | FP19 Fixpipe scalar | 仅 scale | Fixpipe FP16/BF16 | ASW M/N/K tail |
| official raw+AIV | INT32 GM workspace | FP32 vector；BF16先扩为 FP32 | scale→offset→pertoken→float bias | OutT RINT | producer只收 16/32 对齐域；AIV N=2048 tail |
| safe raw+AIV | INT32 GM workspace | 同上 | 同上 | 同上 | actual M/N；Matmul K tail；AIV count tail |

Host 没有证明 `Σ int8*int8 + bias` 必然落在 INT32 范围，也没有声明溢出饱和语义。fast envelope 的 K 上界降低风险但不替代数值范围证明；超大合法 K 走 safe raw 时尤其不能把“INT32 中间面”误写成无限精度整数。

## 失败、fallback 与不可达边界

- null params、platform 为空、AIC/AIV 为零、L1/UB 不足、无效 L1 step、checked-mul 溢出或 allocator 范围越界都在 Host 侧抛异常；不会改走另一个数学实现。
- kernel launch 语句没有逐节点同步状态检查，`direct_call` 提交后返回 0；异步 device 错误由 runtime/stream 传播，不触发 retry。
- `UseSafeRaw` 只在 launch 前选择 producer；official raw 已提交后不会自动重放 safe raw。
- Basic、packed qparam、redundant pack 与 MIX_CV 的 dormant 符号组见前述独立表；不得把这些 body、tile 或 workspace 公式写成可达 Route。
- CATLASS entry 的 AIC/AIV task type 依赖该版本编译器对混合 body 的推导；跨编译器版本复用时要重新核对 task ratio 与 CrossCore flag 配对。
- official `MatmulImpl` 内部物理 buffer 和 Fixpipe 指令属于 CANN SDK 边界；`TCubeTiling` 与 SDK 版本必须成套。
- 本 Direct Invoke 只接线 950/950；A2/A3 没有对应 launcher 或资源参数。

## 实现检查清单

| 检查点 | 必须保持的事实 |
|---|---|
| shape/layout | 右对齐四轴广播；A/B/C contiguous ND RowMajor；三条实现族均 `trans=false` |
| selector | CATLASS→scalar OTF→raw；raw 内再选 official/safe；workspace 使用同一前两条谓词 |
| POD | HostParams 100 B；MixTiling 与 9.1.0 `TCubeTiling` 同 ABI；CATLASS Params 由模板生成 |
| GM offset | safe raw 和 AIV 使用 64-bit 元素 offset；raw workspace 无 row padding |
| Cube dataflow | GM→L1→L0A/L0B→L0C；INT32 bias只在 Cube 累加前注入 |
| epilogue | 浮点链严格 `scale→offset→pertoken→floating-bias→cast` |
| tail | safe raw actual M/N + K Matmul tail；CATLASS 无余数型 N/K tail；AIV 只写 count |
| sync | CATLASS CrossCore + HardEvent；raw 跨 kernel 同 stream；AIV UB barrier |
| fallback | 选择期 fallback，无 device-error retry，无 CPU/ACLNN 数学 fallback |

## 源码定位

| 阶段 | 文件 | 关键符号 |
|---|---|---|
| Torch/Host | `op_plugin/quant_matmul_plugin.cpp` | `CheckInputs`、`CheckedProduct`、`RightAlignFour`、`quant_matmul_npu` |
| Host ABI | `op_kernel/quant_matmul_launch.h` | `QuantMatmulHostParams` |
| selector/tiling | `op_kernel/quant_matmul_kernel.cpp` | `IsCatlassAligned`、`IsScalarOnTheFly`、`UseSafeRaw`、`MakeMixTiling`、`launch_quant_matmul` |
| official OTF/raw | `official/arch35/qbmm_cube_on_the_fly.h` | `MatMulASWKernel`、`SetMMParaAndCompute` |
| safe raw | `op_kernel/quant_matmul_kernel.cpp` | `SafeRawMatmul::BroadcastBatchOffset`、`SafeRawMatmul::Kernel` |
| ordered epilogue | 同上 | `SupplementalEpilogue::Kernel::ComputeTile` |
| CATLASS kernel | `catlass/gemm/kernel/matmul_full_dequant.hpp` | `KernelMatmulDequant` |
| CATLASS Cube | `catlass/gemm/block/block_mmad_pingpong_tla.hpp` | `BlockMmadTla<MmadPingpong<...>>` |
| CATLASS vector | `catlass/epilogue/block/block_epilogue_dequant.hpp` | `BlockEpilogueDequant::VFDoDequant` |

- [SetQuantVector](../../../../apis/simd_api/adv_api/cube_compute/matmul_kernel/set_quant_vector.md)
