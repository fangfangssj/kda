---
type: Operator
title: "MlaProlog 950 Direct Invoke 算子知识"
description: "说明 cann_bench::mla_prolog 的动态 shape 契约、四输出投影/归一化/旋转数据流、Host 有界工作区编排，以及 950 纯 AIV streaming Kernel 的 launch 与精度边界。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/mla_prolog/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/mla_prolog/op_kernel/mla_prolog.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/mla_prolog/op_kernel/mla_prolog_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/mla_prolog/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T19:33:32Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MlaProlog 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 实际契约 |
|---|---|
| Torch 入口 | `cann_bench::mla_prolog`；Meta 与 PrivateUse1 共用 shape/dtype/layout 校验。 |
| 当前可达 Route | R-001：Host 以至多 4 个 token、至多 16 个 head 分块，按当前 stream 串联多个 `__vector__` Kernel。 |
| 四个输出 | `query:[B,S,N,Hckv]`、`query_rope:[B,S,N,Dr]`、`c_kv:[B,S,Hckv]`、`k_rope:[B,S,Dr]`，均为连续 BF16 ND。 |
| 维度来源 | `He/Hcq/D/Hckv/Dr` 均由输入 shape 推导，`N` 来自 `n_heads` 并与 packed 权重互相约束；不是固定模型枚举。 |
| 精度 | BF16 输入和权重；投影、RMSNorm 与 RoPE 主计算为 FP32；FP32 激活参与的两级投影采用三项 BF16-representable 分解和 Kahan 累加；最终输出 RINT 到 BF16。 |
| 工作区 | 三个 256B 对齐的 FP32 tile：`raw`、`cq`、`packed_q`；容量只随内层维度增长，不随 `B*S` 线性增长。 |
| 执行单元 | 当前可达 Route 只使用 AIV/UB，不使用 Cube、L1、L0A/L0B/L0C，也没有 MIX_CV 内部握手。 |
| 量化 | 无量化 Route；没有 scale/zero-point/dequant 参数，所有非 BF16 tensor 在 Host 被拒绝。 |

本卡把 direct 源码的可达路径与官方映射分开表达。官方固定 Commit 支撑 MlaProlog 语义及相关 split-N 结构；本文的 shape-generic streaming 编排、固定 tile 和精度分解属于 Direct Invoke 实现契约，不能反向归因于官方实现。

## 语义、Shape、Dtype 与 Layout

### 符号与计算图

令：

```text
T   = B * S
He  = token_x.shape[2]
Hcq = w_dq.shape[1]
N   = n_heads = w_uk.shape[0]
D   = w_uk.shape[1]
Hckv= w_uk.shape[2]
Dr  = rope_sin.shape[2]
P   = D + Dr
U   = N * P
V   = Hckv + Dr
```

将前两维展平为 token 维后，逻辑语义为：

```text
cq_raw       = X @ W_dq                              # [T,Hcq], FP32
cq            = RMSNorm(cq_raw, gamma_cq, eps_cq)   # [T,Hcq], FP32
packed_q      = cq @ W_uq_qr                         # [T,N,P], FP32
qc, qr        = packed_q[..., :D], packed_q[..., D:]
query         = qc @ W_uk                            # [T,N,Hckv], BF16
query_rope    = RoPE_half(qr, cos, sin)              # [T,N,Dr], BF16

packed_kv     = X @ W_dkv_kr                         # [T,V], FP32
ckv_raw, kr   = packed_kv[..., :Hckv], packed_kv[..., Hckv:]
c_kv          = RMSNorm(ckv_raw, gamma_ckv, eps_ckv)# [T,Hckv], BF16
k_rope        = RoPE_half(kr, cos, sin)              # [T,Dr], BF16
```

`RoPE_half` 与 `torch.chunk(x, 2)` 的奇偶维语义一致。定义 `L1=ceil(Dr/2)`、`L2=floor(Dr/2)`：

```text
rotate_half(x) = concat(-x[L1:Dr], x[0:L1])
RoPE_half(x)   = x * cos + rotate_half(x) * sin
```

因此 `Dr` 可以为奇数，但必须至少为 2。奇数时输出前 `L2` 项读取后半区并取负，后 `L1` 项读取前半区；不能用固定 `Dr/2` 同长两半的实现替代。

### 输入、输出与关系约束

| 参数 | shape | dtype/layout | 关系与语义 |
|---|---|---|---|
| `token_x` | `[B,S,He]` | BF16、strided、contiguous ND | `B>0,S>0,He>0`。 |
| `w_dq` | `[He,Hcq]` | 同上 | 生成 query latent。 |
| `w_uq_qr` | `[Hcq,N*(D+Dr)]` | 同上 | 第二维按全局 head 顺序打包，每个 head 连续存放 `[D,Dr]`。 |
| `w_uk` | `[N,D,Hckv]` | 同上 | 每个 head 独立执行 `D→Hckv`。 |
| `w_dkv_kr` | `[He,Hckv+Dr]` | 同上 | 前 `Hckv` 列为 compressed KV，后 `Dr` 列为 K rope。 |
| `rmsnorm_gamma_cq` | `[Hcq]` | 同上 | query latent RMSNorm gamma。 |
| `rmsnorm_gamma_ckv` | `[Hckv]` | 同上 | compressed KV RMSNorm gamma。 |
| `rope_sin`,`rope_cos` | `[B,S,Dr]` | 同上 | 两者 shape 完全相同，并与 token 域一致。 |
| `n_heads` | scalar int64 | Host 属性 | 正数且可放入 `u32`；必须等于 `w_uk.shape[0]`。 |
| `rmsnorm_epsilon_*` | scalar double→float | Host 属性 | 窄化前必须 finite 且 `>0`。 |
| `query` | `[B,S,N,Hckv]` | BF16 contiguous ND | BMM 直接按 token/head 子区间写最终输出。 |
| `query_rope` | `[B,S,N,Dr]` | 同上 | RoPE 直接按 token/head 子区间写最终输出。 |
| `c_kv` | `[B,S,Hckv]` | 同上 | RMSNorm 直接写最终输出。 |
| `k_rope` | `[B,S,Dr]` | 同上 | 单 head RoPE 直接写最终输出。 |

所有九个 tensor 必须 defined、在同一 device、rank 精确匹配、BF16、strided 且 contiguous。实现不做 `contiguous()`、transpose、ND↔NZ、广播、量化或 cache-index 适配。

### 动态范围不是模型枚举

Host 不把 `He` 限定为 6144/7168，也不把 `Hcq/N/D/Hckv/Dr` 固定为 1536/128/128/512/64。接受范围由以下 ABI 与容量条件共同限定：

- `He/Hcq/N/D/Hckv/Dr/P/U/V` 必须为正且相关 Kernel 维度可放入 `u32`；`Dr>=2`。
- `T=B*S`、四个输出元素数及每个 tensor 的 BF16 字节数必须在 Host 使用的 `int64` tensor 索引范围内。
- packed 宽度 `U=N*(D+Dr)`、`V=Hckv+Dr`，以及输出行宽 `N*Hckv`、`N*Dr` 必须满足相应 `u32` Kernel ABI。
- 单行 DMA 字节跨度必须不超过 `UINT32_MAX`；全局元素 offset 保持 `u64`。
- 动态接受不等于任意大 shape 都可执行：工作区分配、设备内存与固定 UB tile 仍是实际资源边界。

## Host 契约与 Checked Arithmetic

### 校验与分配顺序

```text
Torch schema
  -> CheckMlaInputs
       tensor dtype/layout/rank/device
       positive dimensions and epsilon
       packed shape equalities
       checked element/byte/row-stride arithmetic
  -> AllocateMlaOutputs
  -> getCurrentNPUStream
  -> MakeGenericWorkspace
  -> allocate UInt8 workspace + recordStream(workspace)
  -> RunOpApi("MlaPrologBoundedStreaming", callback)
  -> callback 在同一 stream 上提交全部 primitive kernels
  -> 返回四个 output tensors
```

Meta 路径执行同一 `CheckMlaInputs` 并分配相同 shape 的 meta 输出，所以抽象 shape 契约与设备入口一致。

### Checked arithmetic 闭环

| 量 | 计算 | Host 防护 | Device 用途 |
|---|---|---|---|
| `P` | `D+Dr` | `CheckedAddU64`，结果≤`u32` | packed head stride。 |
| `U` | `N*P` | `CheckedMulU64`，结果≤`u32` | `w_uq_qr` 行宽、B offset。 |
| `V` | `Hckv+Dr` | `CheckedAddU64`，结果≤`u32` | KV/K rope packed 行宽。 |
| `T` | `B*S` | `CheckedMulU64`，结果≤`INT64_MAX` | token tile 循环与全局 offset。 |
| 输出元素 | `T*N*Hckv` 等 | 逐级 checked multiply，结果≤`INT64_MAX` | output 分配与写区间。 |
| tensor bytes | `numel*2` | checked multiply，结果≤`INT64_MAX` | 总 live-byte 规划。 |
| workspace bytes | tile 元素×4、offset 相加/align | checked multiply/add，结果≤`INT64_MAX` | UInt8 workspace 分配。 |
| DMA 行跨度 | 行元素×元素字节 | checked multiply，结果≤`UINT32_MAX` | `DataCopyExtParams` 的 byte stride。 |

`AlignWorkspace(x)` 先检查 `x+255`，再执行 256B 向上对齐。总 tensor bytes 与 workspace bytes还会做一次 checked add；该值用于证明 Host 侧总字节规划可表示，不作为额外内存分配。

Host 的 DMA 行检查还收紧了 Device 中的 `u32` 循环算术：`He*2`、`Hcq*4`、`U*2`、`V*4`、`N*Hckv*2`、`N*Dr*2`、`min(N,16)*P*4` 均不得超过 `UINT32_MAX`。因此 active tile 内的 `n+15`、`Hckv+191`、`k_begin+128`、`begin+512/256` 以及 `head_count*P` 不会在已接受域中越过 `u32`；全 tensor 定位则继续使用 `u64` offset。这里的证明依赖 Torch Host 校验，不能外推给绕过 `CheckMlaInputs` 的裸 launcher 调用。

需要保留的边界：epsilon 在 double 域检查后被窄化为 float，源码没有对窄化结果再次检查。极端正数可能变为 `0` 或 `inf`；若接口要承诺 FP32 epsilon 仍 finite 且正，应在 cast 后补校验，不能把现有 double 校验视为等价保证。

## Route 设计与 Fallback

### Route 重要性

| 重要性 | Route | 可达性 | 适用特征 | 价值与退化关系 |
|---:|---|---|---|---|
| P0 | R-001 bounded streaming | 所有通过 Torch Host 校验的调用 | 任意满足动态契约的 `B/S/He/Hcq/N/D/Hckv/Dr`；4-token、16-head 固定分块 | 唯一实际 Route；工作区有界，代价是多个纯 AIV Kernel 串行和 Vector 实现长点积。 |
| P1 | R-002 fused MIX_CV | 源码级不可达 | 代码中仍保留固定内层维度的单 Kernel launcher 与 by-value tiling 结构 | `kUseMeasuredFastRoute=false` 使 `UseMeasuredFastRoute()` 恒假；它不是 R-001 的运行时 fallback，也不能仅靠 shape 命中。 |

### Production 可达性证明

`UseMeasuredFastRoute(info)` 的返回式以编译期常量 `kUseMeasuredFastRoute` 开头：

```text
UseMeasuredFastRoute(info)
  = false
    && Hcq==1536 && D==128 && Hckv==512 && Dr==64 && N==128
    && U==24576 && V==576
    && He in {6144,7168}
    && B<=UINT32_MAX && S<=UINT32_MAX && T<=UINT32_MAX
  = false
```

这是 C++ 语义上的恒假，不依赖编译器是否删除 dead branch。production 选择器因此可化简为：

```cpp
if (false) {
    launch_mla_prolog_bfloat16_nd_half(...);
    return outputs;
}
run_bounded_streaming(...);  // 对每个 CheckMlaInputs 成功的调用生效
```

所以当前 production 域满足：

```text
CheckMlaInputs succeeds  =>  selected_route = R-001
CheckMlaInputs fails     =>  no kernel launch
```

不存在某个合法 shape、epsilon、平台 core 数或输入值会把当前调用切到 R-002。`epsilon` 不参与 `UseMeasuredFastRoute`，tensor 内容也不参与任何 Host Route 判断。

### Dormant MIX_CV 的重新启用条件

R-002 只有同时满足以下条件才可能重新可达：

1. 把源码常量 `kUseMeasuredFastRoute` 改为 `true` 并重新构建扩展；现有二进制没有环境变量、POD 字段或 runtime key 可改写它。
2. 调用先通过公共 Torch Host 校验。
3. 内层维度精确满足 `Hcq=1536,D=128,Hckv=512,Dr=64,N=128,U=24576,V=576`。
4. `He∈{6144,7168}`，且 `B`、`S`、`T=B*S` 均可放入 `u32`。

若未来仅把常量改成 `true`，上述谓词命中才发射 `launch_mla_prolog_bfloat16_nd_half`；其他仍合法的动态 shape 会继续落到 R-001。因此在当前源码中 R-002 是 dormant code，而在显式重新启用后的源码中 R-001 才同时承担未命中 fixed-shape 谓词的 conditional fallback。

不能从 R-002 保留的 M16/M32/M64/M128 entry、`MlaPrologTilingData`、L1/L0 或 MIX_CV 同步推导当前 active 路径。这些事实只能用于重新启用后的单独资源与精度审计。

### 失败与退化

- shape/dtype/layout/device、packed 关系、overflow 或 DMA stride 不满足时，Host 直接 `TORCH_CHECK`，没有转换型 fallback。
- R-001 内部没有算法 fallback；每个 primitive 固定使用 AIV 实现。
- `VectorBlocks()` 在平台对象为空或报告 `AIV=0` 时把 core 数退化为 1，再发射一个 block。这只是 block 数保底，不证明无可用设备时 Kernel 一定能运行。
- callback 对每次 launch 不取同步错误码并固定返回 0；异步 launch/device 错误依赖 stream/runtime 传播。
- 当前入口不接受空 token 域，因而没有 empty-output 快速返回。

### Active Route 条件分支清单

| 条件 | 结果 | 是否另一个 Route/fallback |
|---|---|---|
| `T-token_base<4` | `token_count` 缩短为 1..3 | 否，R-001 token tail。 |
| `N-head_base<16` | `head_count` 缩短为 1..15 | 否，R-001 head tail。 |
| `work_items<AIV` | block 数缩为 work item 数 | 否，只改变分核。 |
| platform pointer 为空或 `AIV=0` | `VectorBlocks` 使用 1 block | 单核 launch 保底，不切换算法。 |
| `TA=BF16` / `TA=FP32` | 单项乘积 / 三项 BF16-representable 分解 | 编译期 primitive 特化，节点由 Host 数据流固定。 |
| `TC/TOut=FP32` / `BF16` | FP32 workspace 写出 / correction fold 或 RINT 输出 | 编译期输出特化，不由 shape 临时选择。 |
| RoPE phase 0 / 1 | `x*cos-x2*sin` / `x*cos+x1*sin` | 同一 Kernel 的 split-half 两阶段。 |
| `n_begin>=n_end \|\| m==0 \|\| k==0` | `GemmCore` 防御性 no-op | 合法 Host tile 的维度均为正且 block 数不超过 work items，不构成合法域 fallback。 |

这张表覆盖 production selector 之后的条件路径：没有遗漏一个会对合法 tensor 改用 CPU、Cube、量化、不同 workspace 或另一套精度算法的 conditional fallback。

### R-001 完整闭包

| 闭包面 | 关键源码 | 已闭合契约 |
|---|---|---|
| Host 接受域 | `CheckTensor`、`CheckU32Dimension`、`CheckDmaRow`、`CheckMlaInputs` | 九输入 BF16 contiguous ND、同 device、动态 packed 关系、正维度/epsilon、checked elements/bytes/stride。 |
| selector | `kUseMeasuredFastRoute`、`UseMeasuredFastRoute`、`mla_prolog_npu` | 常量为 false；所有 Host 合法调用只到 R-001。 |
| output/stream | `AllocateMlaOutputs`、`getCurrentNPUStream`、`RunOpApi` | 四个 BF16 ND 输出；全部 primitive 按同一 stream 顺序提交。 |
| workspace | `MakeGenericWorkspace`、`AlignWorkspace`、`ByteOffset` | `raw/cq/packed_q` 三个 FP32 slice、256B 对齐、98,304B 典型内层维实例、跨节点覆盖时点。 |
| launch ABI | 六类 `launch_mla_stream_*`、`VectorBlocks` | 无 POD；GM 指针、u32 dim/stride、u64 element offset、f32 epsilon；block 上限为 AIV 数。 |
| Device 投影 | `GemmCore`、`GemmKernel`、`BmmF32Bf16Kernel` | 4×128×192 UB tile，GM ND→UB→GM，AIV Vector 长点积，无 Cube/L1/L0。 |
| Device 后处理 | `RmsNormF32Kernel`、`RopeF32Bf16Kernel` | 512 RMS chunk、256 RoPE chunk、奇偶 Dr split-half、FP32 主计算和 BF16 输出。 |
| UB | 各 Kernel 的 `InitBuffer` | GEMM 最大声明合计 166,912B；RMS 最大 11,264B；RoPE 7,680B；无容量自适配。 |
| tail/mask | `MinU32`、各 tile loop、`DataCopyExtParams` | token/head/K/N/RMS/RoPE tail 均以有效 count 限界，无独立 mask tensor，不写 padding。 |
| precision | `RoundFloatToBf16Float`、Kahan 更新、`CAST_RINT` | BF16 A 单项、FP32 A 三项分解；projection FP32 写出，终端 BMM fold correction 后 RINT。 |
| 生命周期/同步 | TQue、`PIPE_V` barrier、`V_S/S_V` event、stream order | UB tile 内队列闭环；跨 Kernel 只经 GM workspace，同 stream 防止提前覆盖。 |
| multi-output ownership | Host token/head offset 与各 Device writer loop | query/query_rope 按 token-head 分片，c_kv/k_rope 按 token 分片，每个元素唯一 writer。 |

## Workspace、生命周期与多输出所有权

### 工作区公式

定义：

```text
Mt = 4
Ht = min(N, 16)
raw_width = max(Hcq, Hckv + Dr)

raw_bytes = Mt * raw_width * 4
cq_bytes  = Mt * Hcq * 4
qr_bytes  = Mt * Ht * (D + Dr) * 4

raw_offset = 0
cq_offset  = align256(raw_bytes)
qr_offset  = align256(cq_offset + cq_bytes)
W          = align256(qr_offset + qr_bytes)
```

`raw` 先装 `X@W_dq`，稍后复用为 `X@W_dkv_kr`；`cq` 保存归一化后的 query latent；`qr` 实际保存当前 head tile 的 `[Qc,Qr]` packed FP32 结果。名称 `qr_ptr` 不表示只含旋转部分。

以 `Hcq=1536,N=128,D=128,Hckv=512,Dr=64` 为例：`raw=24576B`、`cq=24576B`、`packed_q=49152B`，总计 `98304B`。该数值只是公式实例，不是 Host 的 shape 枚举。

### 每个 token tile 的 launch 图

令 `Q=ceil(N/16)`。每个 token tile 提交 `5+3Q` 个 Kernel：

```text
token_x[t:t+m]
  -> GEMM BF16×BF16→FP32 (W_dq)             -> raw
  -> RMSNorm FP32→FP32                      -> cq
  -> for head tile h:
       GEMM FP32×BF16→FP32 (W_uq_qr slice)  -> packed_q
       ├─ BMM FP32×BF16→BF16 (Qc,W_uk)      -> query[t,h]
       └─ RoPE FP32→BF16 (Qr,sin,cos)       -> query_rope[t,h]
  -> GEMM BF16×BF16→FP32 (W_dkv_kr)         -> raw (reuse)
     ├─ RMSNorm FP32→BF16 (Ckv prefix)      -> c_kv[t]
     └─ RoPE FP32→BF16 (Kr suffix)          -> k_rope[t]
```

全部 Kernel 提交到同一 current stream。由 stream 顺序形成跨 Kernel 的 producer→consumer 和 workspace 复用关系；源码没有跨 Kernel event，也不允许把这些 launch 无依赖地分发到不同 stream。

### Workspace 生命周期

| slice | producer | consumer | 可覆盖时点 |
|---|---|---|---|
| `raw` | query down-projection | query RMSNorm | RMSNorm 完成后；当前实现等所有 head tile 结束才用 KV projection 覆盖。 |
| `cq` | query RMSNorm | 每个 head tile 的 UQ/QR projection | 当前 token 的所有 head tile 完成后。 |
| `packed_q` | 当前 head tile 的 UQ/QR projection | 同 head tile 的 Query BMM 与 Query RoPE | 两个 consumer 均完成后，下一 head tile 才覆盖。 |
| `raw`（复用） | KV/K rope projection | Ckv RMSNorm 与 K RoPE | 两个 consumer 均完成后，下一 token tile 才覆盖。 |

workspace 使用 UInt8 tensor 承载并显式 `recordStream`；FP32 slice 依靠 256B 对齐满足地址对齐。其余输入/输出生命周期由框架和当前 stream 语义承担，源码没有为每个输入指针单独调用 `recordStream`。

### 四输出写域

| 输出 | writer 分区 | 防重叠关系 |
|---|---|---|
| `query` | `(token tile, global head, Hckv 的 192 列 tile)` | Host 先把基址推进到 `token_base*N*Hckv`；BMM 再加 `global_head*Hckv+n_tile`。每个 task 只拥有一个 head/列片。 |
| `query_rope` | `(token tile, global head)` | `output_offset=token_base*N*Dr`，再加 `(head_base+local_head)*Dr`。 |
| `c_kv` | `(token tile, Hckv chunk)` | RMSNorm 的 output offset 为 `token_base*Hckv`，每 row 独占一个 token。 |
| `k_rope` | `(token tile, Dr chunk)` | RoPE 以 `heads=1` 写 `token_base*Dr` 起始的 token 行。 |

token tail、head tail 和列 tail 都缩短实际写长度，不会把 padding 写进相邻输出。

## Active Launch ABI：无 POD 的标量契约

### 关键结论

R-001 不生成或传递 tiling POD。每个 C launcher 直接把 GM 指针、`u32` 维度/stride、`u64` 元素 offset、FP32 epsilon 作为 Kernel 参数传值。旧的 `MlaPrologTilingData` 只属于不可达的 R-002；将其字段表当作当前 Host/Device 契约会产生错误实现。

### Primitive ABI

| Launcher | 输入/输出 | 标量参数 | block 数 |
|---|---|---|---|
| `launch_mla_stream_gemm_bf16_f32` | BF16 A、BF16 B、FP32 C | `m,k,n,lda,ldb,ldc:u32`；三个 element offset:`u64` | `min(ceil(n/16), AIV)`，至少 1。 |
| `launch_mla_stream_gemm_f32_f32` | FP32 A、BF16 B、FP32 C | 同上 | 同上。 |
| `launch_mla_stream_bmm_f32_bf16` | packed FP32 A、BF16 `w_uk`、BF16 query | `m,D,Hckv,head_count,head_base,packed_head_stride,output_row_stride:u32` | `min(head_count*ceil(Hckv/192), AIV)`。 |
| `launch_mla_stream_rmsnorm_f32_f32` | FP32 input、BF16 gamma、FP32 output | `rows,width,input/output_row_stride:u32`；offset:`u64`；epsilon:`f32` | `min(rows,AIV)`。 |
| `launch_mla_stream_rmsnorm_f32_bf16` | FP32 input、BF16 gamma、BF16 output | 同上 | 同上。 |
| `launch_mla_stream_rope_f32_bf16` | FP32 input、BF16 cos/sin、BF16 output | row/head/stride/column 字段:`u32`；input/coefficient/output offset:`u64` | `min(rows*heads,AIV)`。 |

所有 offset 的单位都是元素，不是字节；只有 `ByteOffset()` 推进 output/workspace 原始指针时使用字节。Host 已检查所有会进入 DMA descriptor 的行跨度，Device 不再做越界校验。

## Kernel 分核、GM/UB 数据流与资源

### 固定 tile

| 常量 | 值 | 用途 |
|---|---:|---|
| `kTokenTile/kGemmM` | 4 | Host token tile，也是 GEMM M tile 上限。 |
| `kHeadTile` | 16 | Host head tile。 |
| `kGemmK` | 128 | 所有 GEMM reduction chunk。 |
| `kGemmN` | 192 | GEMM/BMM 输出列 tile。 |
| `kVectorChunk` | 512 | RMSNorm reduce/apply chunk。 |
| `kRopeChunk` | 256 | RoPE 每个半区 phase 的 chunk。 |

### AIV block 映射

`GemmKernel` 先把 N 轴表示为 16 列 unit，再将连续 unit 区间均分给 blocks。每个 block 在自己的 `[n_begin,n_end)` 中按 192 列迭代；同一 block 内再按 M=4、K=128 遍历。

`BmmF32Bf16Kernel` 把任务定义为：

```text
task = local_head * ceil(Hckv/192) + n_tile
```

blocks 以 `task=block, block+blocks, ...` 循环。每个 task 计算一个 head 的一个 Hckv 列片，A 指向当前 head tile 内的 Qc 前缀，B 指向全局 head 的 `w_uk`，C 指向全局 head 输出子区间。

`RmsNormF32Kernel` 以 `row=block, block+blocks, ...` 分 token 行；`RopeF32Bf16Kernel` 以展平的 `(row,head)` work item 循环。Host 的 `rows<=4`，Query RoPE 的 `heads<=16`，K RoPE 的 `heads=1`。

### GM→UB→GM

当前 Route 的矩阵运算并非 Cube Matmul：

```text
GM ND A/B
  -> DataCopyPad 到 depth=1 的 VECIN queues
  -> Cast 到 FP32 UB
  -> Muls + Add/Sub 在 UB 完成长点积
  -> FP32 workspace 或 RINT BF16 VECOUT queue
  -> DataCopyPad 回 GM ND
```

没有 ND→NZ、L1 复用、L0C accumulator 或 Fixpipe。权重每个 K/N tile 从 GM 搬到 UB；FP32 A 的 UQ/QR 与 Query BMM 会为每个 reduction scalar 形成三项分解。这个选择换取统一 shape 契约和明确精度顺序，但长点积吞吐不能按 Cube 路线估算。

### 声明的 UB 缓冲

| Kernel | 主要 buffer | 声明字节合计的上界 |
|---|---|---:|
| `GemmCore<float,float>` | A/B/C queues；A/B float；accumulator/correction/term/next | 166912 B |
| `GemmCore<bfloat16_t,float>` | 同上，A queue 为 BF16 | 165888 B |
| `GemmCore<float,bfloat16_t>` | 同上，C queue 为 BF16 | 165376 B |
| `RmsNormF32Kernel<float>` | input/gamma/output queues；square/gamma_float/result | 11264 B |
| `RmsNormF32Kernel<bfloat16_t>` | 同上，output queue 为 BF16 | 10240 B |
| `RopeF32Bf16Kernel` | first/second/coefficient/output queues；coefficient/result/temp | 7680 B |

这些数值是 `InitBuffer` 声明项的求和，不含编译器/运行时附加开销。R-001 不读取平台 UB 大小，也没有 tile 缩小 fallback；移植到不同 UB 容量时必须重新审计，不能只沿用 AIV core 数。

### Cube/Vector 协同边界

- R-001 的 Kernel 均标记 `__vector__`，只有 AIV block；不存在 AIC producer、AIV consumer 或 MIX_CV 的跨子块同步。
- R-002 的保留源码是单 MIX_CV 结构，包含 Cube Matmul、Vector RMS/RoPE、L1/L0 与 by-value tiling；但 Host 常量关闭了这条路径。
- 因而当前实现的真实协同发生在“Host 串联多个 AIV Kernel”和“同 stream 的 GM workspace 交接”，不是 Cube/Vector 同 Kernel 协同。

## 精度、量化与旋转细节

### 投影累加

`GemmCore<TA,TC>` 对每个输出 tile 维护 `accumulator` 与 `correction`，逐 reduction scalar 做 Kahan 形式的固定顺序累加：

```text
term = scale_part * weight - correction
next = accumulator + term
correction = (next - accumulator) - term
accumulator = next
```

当 `TA=BF16` 时，`scale_part_count=1`。当 `TA=FP32` 时，标量 `a` 被分解为三个可由 BF16 舍入表示的 FP32 数：

```text
p0 = round_to_bf16_float(a)
r0 = a - p0
p1 = round_to_bf16_float(r0)
p2 = round_to_bf16_float(r0 - p1)
```

然后依次累加 `p0*w`、`p1*w`、`p2*w`。这用于 `cq@W_uq_qr` 与 `qc@W_uk`，不是 tensor 量化：没有生成 BF16 临时矩阵，也没有 scale/zero-point。权重本身始终是 BF16。

FP32 输出的两级 projection 直接写 `accumulator`；终端 Query BMM 在 cast 前写 `accumulator-correction`，再以 `CAST_RINT` 转 BF16。这个差异是源码确定的舍入顺序，不能把所有 GEMM 统一改成相同 terminal fold。

### RMSNorm

每行先按 512 元素 chunk 读取 FP32 输入并计算平方，再用标量 Kahan 顺序归约：

```text
inv_rms = 1 / sqrt(kahan_sum(x*x) / width + epsilon)
y = x * inv_rms * float(gamma_bf16)
```

Query latent 输出保持 FP32，供后续 FP32×BF16 projection；Ckv 输出才 RINT 到 BF16。归约过程中通过 `V_S`/`S_V` hard event 包围 `GetValue` 标量访问。

### RoPE

RoPE 同时搬入 FP32 原值、FP32 旋转配对值和 BF16 cos/sin，在 UB 中把系数 cast 为 FP32：

```text
前 L2 项: result = x*cos - x[L1:]*sin
后 L1 项: result = x*cos + x[:L1]*sin
```

每个 phase 再按 256 元素分块，最终 RINT 到 BF16。系数 offset 是 `token_base*Dr`，所以同一 token 的所有 query heads 共享该 token 的 sin/cos 行；K rope 以单 head 复用相同公式。

### 量化边界

- schema 和 `CheckTensor` 只接受 BF16 tensor；INT8/FP8/HIF8 或混合 dtype 没有可达入口。
- ABI 没有 quant/dequant/smooth scale、zero point 或 quant mode 字段。
- FP32→三项 BF16-representable 数值分解是乘法精度技术，不改变张量 dtype，也不能替代真正的量化契约。

## Tail、Mask 与同步

### Tail 闭环

| 尾部 | 处理方式 | 不变量 |
|---|---|---|
| token tail | Host `token_count=min(4,T-token_base)` | 只提交 1..4 行；全局 offset 用 `u64`。 |
| head tail | `head_count=min(16,N-head_base)` | packed workspace 行宽缩为 `head_count*P`，全局输出仍按 N-stride。 |
| K tail | `k_count=min(128,K-k_begin)` | DMA 只搬有效列，循环只归约 `k_count`。 |
| N/Hckv tail | `n_count=min(192,n_end-n_tile)` | Vector op 与 DMA byte count 均用 `n_count`。 |
| RMS width tail | `count=min(512,width-begin)` | reduce 和 apply 两遍均使用同一有效 count。 |
| RoPE tail | 每个 half phase 内 `count=min(256,phase_end-begin)` | 不跨越奇数 Dr 的 split 边界。 |

源码不构造单独的 mask tensor；有效元素数直接进入 Vector API 的 count 和 `DataCopyExtParams.blockLen`。局部行 stride 仍按固定最大 tile 留空间，padding 不参与 reduction，也不写回 GM。

### Kernel 内同步

- GEMM 的 A/B/C 使用 depth=1 queue 的 `Alloc→EnQue→DeQue→Free` 生命周期。
- 对 accumulator、term、next、correction 的相依 Vector 指令之间使用 `PipeBarrier<PIPE_V>()`。
- GEMM 在 Vector buffer 与标量 `GetValue` 循环之间使用 `V_S`/`S_V` hard event。
- RMSNorm 的平方 Vector 结果转入标量 Kahan，以及标量 `inv_rms` 返回 Vector 路径，也使用相同 hard event。
- RoPE 通过 queue 生命周期和顺序 Vector 指令形成 tile 内依赖；没有 AIC/AIV event。

跨 Kernel 不共享 UB。唯一的跨 Kernel 中间态位于 GM workspace，并由同 stream 的提交顺序保证可见性与覆盖安全。

## 实现骨架

### Host 编排最小骨架

```cpp
info = CheckMlaInputs(...);
outputs = AllocateMlaOutputs(info);
layout = MakeGenericWorkspace(info);       // raw/cq/packed_q, align256

for (uint64_t t = 0; t < T; t += 4) {
    uint32_t mt = min<uint64_t>(4, T - t);
    gemm_bf16_f32(X, W_dq, raw, mt, He, Hcq, t * He, ...);
    rms_f32_f32(raw, gamma_cq, cq, mt, Hcq, ...);

    for (uint32_t h = 0; h < N; h += 16) {
        uint32_t ht = min(16, N - h);
        gemm_f32_f32(cq, W_uq_qr + h*P, packed_q, mt, Hcq, ht*P, ...);
        bmm_f32_bf16(packed_q, W_uk, query + t*N*Hckv, mt, ht, h, ...);
        rope_f32_bf16(packed_q + D, sin, cos, query_rope, mt, ht, h, t, ...);
    }

    gemm_bf16_f32(X + t*He, W_dkv_kr, raw, mt, He, V, ...);
    rms_f32_bf16(raw, gamma_ckv, c_kv + t*Hckv, mt, Hckv, ...);
    rope_f32_bf16(raw + Hckv, sin, cos, k_rope + t*Dr, mt, 1, t, ...);
}
```

该骨架强调参数关系和生命周期，不表示可以省略 checked arithmetic、stride 校验、stream 记录或 output head/token offset。

### GEMM 最小骨架

```cpp
for (m0 : tiles<4>(m))
  for (n0 : block_owned_tiles<192>(n)) {
    acc = 0; correction = 0;
    for (k0 : tiles<128>(k)) {
      copy_pad(A[m0,k0], B[k0,n0]);
      cast_to_fp32();
      for (row)
        for (reduction)
          for (part : bf16_parts_if_A_is_fp32)
            kahan_add(acc, correction, part * B[reduction,:]);
    }
    store_fp32_or_rint_bf16(valid_m, valid_n);
  }
```

## 不变量、限制与待闭合边界

### Route 间不变量

即使将来启用保留的 fused Route，下列外部契约也必须一致：

- 四输出 shape、head packing 顺序和 split-half RoPE 语义不变。
- 输入权重仍按 contiguous ND 的逻辑行主序解释；任何 ND→NZ 只能是内部搬运。
- `query` 使用 Qc 前 `D` 项，`query_rope` 使用每 head 后 `Dr` 项；KV projection 同理。
- RMSNorm reduction 维分别为 `Hcq` 与 `Hckv`，epsilon 不交叉。
- 同一输出元素只有一个 writer，token/head tail 不写 padding。
- 最终四输出均为 BF16；若改变中间累加顺序，必须重新界定数值契约。

### 明确限制

- 当前 Route 的多个 Kernel 是串行数据流，不是可自由重排的独立算子集合。
- 固定 UB tile 没有运行时容量探测或更小 tile fallback。
- `VectorBlocks` 只读取 AIV core 数；AIC 数量不参与 R-001 launch。
- 所有 tensor 数值本身不做 finite 检查；NaN/Inf 的传播由具体 Vector 指令决定。
- Host 只检查地址跨度与整数可表示性，不知道调用方传入裸 C ABI 指针的真实 allocation 大小；Torch 入口依赖 tensor shape/storage 契约。
- 不存在 non-contiguous、量化、空 token 或 CPU fallback。
- callback 固定返回 0，不在 Host 同步等待全部 Kernel 完成。

### 仍未知或不可外推

- 官方固定 Commit 的公开实现不能证明本卡 streaming Kernel 的 tile、Kahan、三项分解或动态 shape 范围；这些只能作为 Direct Invoke 源码事实使用。
- 源码没有给出 R-001 在不同 950 core/UB 配置上的动态资源适配规则；仅能确认 launch block 上限取平台 AIV core 数。
- 源码没有为 epsilon double→float 窄化后的极端值建立完整契约。
- R-002 虽保留 Cube/Vector、POD 与 workspace 代码，但不可达；不能把它当作当前性能、资源或 fallback 事实。

## 源码闭包索引

| 链路 | Direct 源码符号 | 本卡提炼事实 |
|---|---|---|
| schema/Meta/NPU | `TORCH_LIBRARY_FRAGMENT`、`CheckMlaInputs`、`mla_prolog_meta`、`mla_prolog_npu` | 动态 shape 关系、四输出、Host checked arithmetic、Route 常量。 |
| workspace | `GenericWorkspaceLayout`、`MakeGenericWorkspace`、`ByteOffset` | 三 slice 公式、256B 对齐与复用周期。 |
| C ABI | `mla_prolog_launch.h` 的六类 `launch_mla_stream_*` | 无 POD 的 pointer/u32/u64/f32 参数边界。 |
| GEMM/BMM | `GemmCore`、`GemmKernel`、`BmmF32Bf16Kernel` | 4×128×192 tile、AIV 分核、三项分解、Kahan 与输出 offset。 |
| RMSNorm | `RmsNormF32Kernel<TOut>` | 512 chunk、FP32 Kahan reduction、FP32/BF16 双输出类型。 |
| RoPE | `RopeF32Bf16Kernel` | 奇偶 `Dr` 的 split-half、256 chunk 与 token/head 写域。 |
| launch | `VectorBlocks` 与各 extern C launcher | AIV block 上限、同 stream 多 Kernel 提交。 |
| 不可达边界 | `kUseMeasuredFastRoute`、`UseMeasuredFastRoute`、`launch_mla_prolog_bfloat16_nd_half` | 保留 fused MIX_CV 路线不参与当前选择。 |
| 构建接线 | 算子 `CMakeLists.txt::register_direct_launch_op` | kernel、launch header 与 plugin 合并进 Direct Invoke 扩展。 |

官方 `README.md`、`mla_prolog.cpp`、`mla_prolog_tiling_data.h` 与 arch35 split-N header 用于对齐算子语义、公开接口背景和相关架构概念；Direct-specific 的 Host selector、streaming primitive 与精度顺序以上表源码闭包为准。