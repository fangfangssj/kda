---
type: Operator
title: "EngramGateFusion 950 Direct Invoke 算子知识"
description: "说明派生补充实现 cann_bench::engram_gate_fusion 在 950 上的多 dtype 权重适配、双层 Route 分发、SIMT 通用闭环与 strict-fast AIV 向量路线。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_causal_conv1d/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_causal_conv1d/op_kernel/arch35/fused_causal_conv1d_cut_bh.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/fused_causal_conv1d/op_kernel/arch35/vf/fused_causal_conv1d_128dim.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/masked_causal_conv1d/op_kernel/arch35/masked_causal_conv1d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/attention/masked_causal_conv1d/op_kernel/arch35/vf/masked_conv1d_vf.h
    role: implementation
created_at: "2026-08-25T19:32:13Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# EngramGateFusion 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| Direct 入口 | Python facade → `torch.ops.cann_bench.engram_gate_fusion` → PrivateUse1 `engram_gate_fusion_npu` → `launch_engram_gate_fusion_bfloat16` |
| 计算语义 | 双 RMSNorm 产生缩放点积 gate；gate 调制共享 value 后再次 RMSNorm；随后执行 depthwise dilated causal convolution、SiLU、gated residual，并生成连续 state |
| dtype | activation、可选 state、output 均为 BF16；四个 weight 可分别为 FP32/BF16/FP16/FP64，非“四个连续 FP32”组合先在 NPU 转成连续 FP32 compute image |
| layout | 通用路线消费 activation/state 的实际 stride，也可打包 strided weight；strict-fast 只接收连续 activation/state 与直接或已打包的连续 FP32 weight |
| 执行图 | 可选 `pack_weights` 后，二选一：strict-fast 单 AIV kernel；或 `row_stats` → `emit` 两个 AIV SIMT kernel。没有 AIC/Cube、L1/L0、通信或 CPU 数值节点 |
| Route 顺序 | P0 R-006 universal；P1 R-005 strict K4 batch8；P2 R-003 strict K4 four-row；P3 R-004 strict generic；P4 R-001 strict K4 scale；P5 R-002 strict K8。重要性顺序不改写真实 selector 顺序 |
| 不可达实现 | D-001 independent-tile strict stream、D-002 SIMT materialized pair、D-003 strict 四行 norm helper；均保留源码但生产 launcher 不产生入口条件 |
| 来源闭包 | `direct_implementation_closed`：schema/Meta/校验 → stride 与 weight view → workspace/POD → outer selector → pack/strict/universal launch → Device body → 构建注册。官方固定源码只提供相关 causal-conv、state、搬运和寄存器约定，不提供同构 EngramGateFusion |

## 来源与转换关系

固定 CANN 9.1.0 `ops-transformer` 源码中的 `fused_causal_conv1d` 与
`masked_causal_conv1d` 支撑 950 causal timeline、state 拼接、DataCopyPad、寄存器卷积和
event 用法等相关事实，但没有本文的双 RMSNorm gate、gated-value RMSNorm、residual、
多 dtype weight pack、Torch schema 或同构 Host/Kernel 链路。

本文的 schema、Host 校验、POD、launcher、SIMT 通用路线及 strict-fast 路线均来自
Direct Invoke 交付源码，是 Device 派生补充实现；不能据此推出官方公开 API、官方同名
算子或其他平台支持。Frontmatter 中固定 HTTP 文件只证明相关官方原生机制，派生源码
事实则由本卡“源码索引”所列生产文件闭合。

## 接口与数学语义

### Schema 与张量契约

```text
engram_gate_fusion(
  keys, hidden_states, value,
  norm1_weight, norm2_weight, conv_norm_weight, conv_weight,
  conv_state=None, hc_mult=4, hidden_size=1024,
  kernel_size=4, dilation=3, norm_eps=1e-5
) -> (output, conv_state_out)
```

| 参数 | shape | dtype/layout | Host 约束与语义 |
|---|---|---|---|
| `keys` | `[B,L,HC,D]` | BF16；通用路线允许非连续 | `B,L,HC,D>0`；作为设备锚点 |
| `hidden_states` | 同 `keys` | BF16；可非连续 | shape、device 与 `keys` 相同 |
| `value` | `[B,L,D]` | BF16；可非连续 | 同 device；每个 HC 共享同一 value row |
| `norm1_weight` | `[HC,D]` | FP32/BF16/FP16/FP64；可非连续 | stride0/stride1 写入 weight view |
| `norm2_weight` | `[HC,D]` | 同上 | 同 device |
| `conv_norm_weight` | `[HC,D]` | 同上 | 同 device |
| `conv_weight` | `[HC*D,1,K]` | 同上 | stride0/stride2 写入 rank-3 view；中间维固定为 1 |
| `conv_state` | 可选 `[B,HC*D,S]` | BF16；通用路线允许非连续 | `S=(K-1)*dilation`；同 device |
| `hc_mult` | scalar int64 | 默认 4 | 必须等于 `HC` |
| `hidden_size` | scalar int64 | 默认 1024 | 必须等于 `D` |
| `kernel_size` | scalar int64 | 默认 4 | `K>=1` |
| `dilation` | scalar int64 | 默认 3 | `dilation>=1` |
| `norm_eps` | double→FP32 POD | 默认 `1e-5` | 源码没有范围/有限性检查；所有 Device 路线消费窄化后的 FP32 |
| `output` | `[B,L,HC,D]` | 新分配连续 BF16 | 逻辑索引与 `keys` 相同，物理输出连续 |
| `conv_state_out` | `[B,HC*D,S]` | 新分配连续 BF16 | `S=0` 时为空末维，Device 不进入 state 分支 |

Host 用逐级 `checked_mul` 证明 `HC*D`、`S`、row/output/state element count 的
int64 可寻址性。除 strict-fast admission 外没有 `B/L/HC/D/K/dilation` 的固定上限；
容量较大、混合 dtype、strided tensor 或超出 fast envelope 不是拒绝条件，而是转入 R-006。

### 七阶段 FP32 语义

令 `S=(K-1)*dilation`，`rms(x)=sqrt(sum_d(x_d*x_d)/D+eps)`：

```text
k = (BF16(keys[b,l,h,:]) / rms(keys[b,l,h,:])) * FP32(norm1[h,:])
q = (BF16(hidden[b,l,h,:]) / rms(hidden[b,l,h,:])) * FP32(norm2[h,:])
raw = sum_d(k_d*q_d) / sqrt(FP32(D))
g = sigmoid(sign(raw) * sqrt(max(abs(raw), 1e-6)))
r = rms(FP32(BF16(value[b,l,:])) * g)
u = ((FP32(BF16(value[b,l,:])) * g) / r) * FP32(convNorm[h,:])
z[d] = +0
for tap in [0,K):
  sourceTime = l - (K-1-tap)*dilation
  source = u[b,sourceTime,h,d]                     if sourceTime >= 0
           BF16(state[b,h*D+d,S+sourceTime])       if sourceTime < 0 and state exists
           +0                                      otherwise
  z[d] += FP32(convWeight[h*D+d,0,tap]) * FP32(source)
output[b,l,h,d] = BF16_RNE(silu(z[d]) + FP32(BF16(value[b,l,d]))*g)
```

`conv_state_out[b,h*D+d,s]` 取 `sourceTime=L-S+s` 对应的 `u`；若时间仍为负，
则取旧 state 的 `S+sourceTime`，无 state 时写 `+0`。不能把双 RMS 除法移出点积，
也不能把 `sum((value*g)^2)` 重关联为 `g^2*sum(value^2)`；两种改写会改变
FP32 rounding、overflow、NaN/Inf 和 signed-zero 行为。

## Host、Workspace 与 POD

### Host 前处理和 weight image

`engram_gate_fusion_npu` 的顺序为：设备 guard → `check_contract` → Meta 分配输出 →
构造 `EngramGateFusionParams` → 计算 outer selector → 分配三个 workspace → 获取当前
NPU stream → launcher。没有 `.contiguous()`、Host cast、框架 numerical composite 或
CPU fallback。

四个 weight 各有 `EngramWeightView{count,dstOffset,stride0,stride1,stride2,dtype,rank}`。
只有四者全为连续 FP32 时 `packWeights=0` 并直接把原 GM 指针交给消费者；其余组合
统一 `packWeights=1`，先把四段按 `norm1 | norm2 | convNorm | conv` 拼入 FP32
`weightWorkspace`。因此“其中三个连续 FP32、一个其他表示”也会重写全部四段。

```text
normCount = HC*D
convCount = HC*D*K
packedWeightCount = 3*normCount + convCount
dstOffset = [0, normCount, 2*normCount, 3*normCount]
```

`PackWeightsVF` 按 `core*1024+threadIdx` 起始、`blockDim*1024` 步长遍历目标元素；
rank-2 view 用 `(row,col)` 的 stride0/stride1，conv view 用 stride0/stride2。
BF16/FP16 通过 SIMT conversion 转 FP32；FP64 由 `DoubleBitsToFloat` 按二进制位执行
RNE、Inf/NaN、normal/subnormal/overflow/underflow 映射。

### Workspace 账与生命周期

令 `R=B*L*HC`、`O=R*D`：

| Workspace | shape / bytes | producer | consumer | 分支生命周期 |
|---|---|---|---|---|
| `statsWorkspace` | FP32 `[2,R]`，`8R` bytes | strict 只写 gate；R-006 `row_stats` 写 gate+rms | strict conv 或 R-006 `emit` | 每次调用都分配；同 stream 最后 consumer 完成后释放 |
| `weightWorkspace` | FP32 `[packedWeightCount]` 或空，`4*(3HC*D+HC*D*K)` bytes | `pack_weights` | strict 或 R-006 | 仅 `packWeights=1` 有容量；kernel 间由 stream 顺序保护 |
| `normalizedWorkspace` | FP32 `[O]` 或空，`4O` bytes | strict normalize | strict convolution/state writer | 仅 `useMaterializedFast=1` 分配；R-001 前缀另承载 row RMS 标量 |

R-006 不物化 `u`，只使用 `statsWorkspace`；strict-fast 总是拥有完整
`normalizedWorkspace`，即使 R-001 只为大多数 row 写 scalar RMS。workspace 都由临时
Torch tensor 持有，不出接口；launcher 中异步 kernel 的同 stream 消费发生在 tensor
析构所受 allocator/stream 语义之前。

### 通用 POD

`EngramGateFusionParams` 是 launcher 与 SIMT kernel 的按值契约：

| 字段组 | 类型 | Host producer | Device consumer / 单位 |
|---|---|---|---|
| `batch,length,hc,dim,channels,kernelSize,dilation,stateLength` | 8×int64 | shape/attrs/checked products | 64-bit index、loop、state relation；元素或 token |
| `rowCount,outputCount,stateCount,packedWeightCount` | 4×uint64 | 前三者由 checked int64 product 转换；packed count由四段 view 累加 | grid-stride 上界、workspace 边界；元素 |
| `keysStride,hiddenStride` | `EngramStrides4` | tensor stride | `Offset4`，单位为 element |
| `valueStride,stateStride` | `EngramStrides3` | tensor stride 或零 | `Offset3`，单位为 element |
| 四个 `*View` | `EngramWeightView` | count/dst offset/原 stride/dtype/rank | pack source address 与 FP32 destination segment |
| `normEps,sqrtDim` | 2×float | double 窄化；`sqrt(float(D))` | row statistics |
| `hasState,packWeights,useMaterializedFast` | 3×uint32 | optional/weight/outer selector | state、pack、strict-vs-universal 分支 |

按普通 LP64/8-byte struct alignment 推导，`EngramGateFusionParams` 为 424 bytes；
`EngramGateFusionLaunchConfig` 的 `params` 预期从 offset 8 开始，总计 432 bytes。源码没有
`static_assert(sizeof/offsetof)`，跨编译器或改字段时必须重新锁定，不能把该推导当成
显式 ABI 保证。

### strict-fast POD 与 tiling

`EngramStrictFastTiling` 保持 `14×uint32 + 3×float`：

```text
[batch,length,hc,dim,alignedDim,kernelSize,dilation,stateLength,
 normTileLength,convTileLength,normTaskCount,convTaskCount,
 hasState,reserved,normEps,invDim,sqrtDim]
```

普通 4-byte 布局下字段 offset 为 `0,4,...,64`，size 68；config 的
`blockDim + tiling` 预期 size 72，同样没有静态 ABI assertion。`alignedDim=ceil(D/64)*64`。

```text
C = B*HC
targetTilesPerSequence = max(1, ceil(2*P/C))
normTile = clamp(ceil(L/targetTilesPerSequence), 1, 16)
convTile = clamp(ceil(L/targetTilesPerSequence), 1, 4)
if no_state && K==4 && 1<L<32: normTile=max(normTile,2); convTile=max(convTile,2)
if strict K8 specialized: normTile=max(normTile,8); convTile=8
if strict K4 batch8:     normTile=max(normTile,8); convTile=8
normTaskCount=C*ceil(L/normTile); convTaskCount=C*ceil(L/convTile)
blockDim = (L==1) ? clamp(C,1,P) : P
```

其中 `P=max(1,GetCoreNumAiv())`。outer config 的 `blockDim` 也是 `P`；进入 strict 后
重新计算 strict config，故 `L=1` 可把 launch block 数缩到 `min(C,P)`。

## Route 总表与真实 selector

### 分类和重要性

P0 首先保证整个 Host 合法域有纯 NPU 终点；P1/P2 是最具体的 K4 向量主干；P3 是
strict envelope 的安全终点；P4/P5 是较窄但可复用的长序列 K4 与 K8 专门化。
排序用于阅读与迁移，不改变 H0→H6 的真实判断顺序。

| 重要性 | Route | 状态 | 作用域 | 主要价值 | fallback 出边 |
|---:|---|---|---|---|---|
| P0 | R-006 universal row-stats + causal emit | active，outer 最终 fallback | 所有通过 Host 契约且 `useMaterializedFast=0` 的输入 | 跨 dtype/stride/大 shape 完整语义闭环，小 workspace | 无 Device fallback；Host 契约失败才终止 |
| P1 | R-005 strict K4 batch8 | conditional active | strict envelope 内，无 state K4、`32<=L<1024`、中宽 D，且 batch8 capacity guard | 八行 staging、两次四行寄存器卷积 | selector false 继续 R-003/R-004；outer false→R-006 |
| P2 | R-003 strict K4 four-row | conditional active | strict envelope 内同类 K4，但未命中 batch8 | 四行 unique timeline 主干 | Device guard false→R-004；outer false→R-006 |
| P3 | R-004 strict bounded generic | active，strict 最终 fallback | strict envelope 内未命中 R-001/R-002/R-005/R-003 | state、K1..16、宽 D、长 dilation 的有界终点 | outer false→R-006；本 route 内按资源 guard 换 helper |
| P4 | R-001 strict K4 scale reconstruction | conditional active | strict envelope 内无 state 长序列 K4 | 仅发布 gate/RMS 与末 S 行，卷积按需重建 | H2 false 继续 H3；outer false→R-006 |
| P5 | R-002 strict K8 unique timeline | conditional active | strict envelope 内无 state K8、`L>=32`、中宽 D | 八 tap、最多八输出行寄存器卷积 | H3 false 继续 H4；outer false→R-006 |
| D0 | D-001 independent-tile stream | dormant | `tiling.reserved==1` | 去全核 barrier 的 halo 重算草案 | outer launcher 强制 `reserved=0`，生产不可达 |
| D1 | D-002 SIMT materialized pair | dormant | 已定义两个 kernel entry | warp-row materialize + scalar emit | launcher 无调用点，生产不可达 |
| D2 | D-003 strict four-row norm helper | dormant | strict normalize helper | 四 row 共享 weight load | caller 条件恒为 `false && ...`；生产只用 2-row/1-row |

### Outer admission H0/H1

```text
packWeights = !(all four weights are contiguous FP32)

directF32 = !packWeights
packedBF16 = all four weights are contiguous BF16
bf16FastEnvelope = B<=8 && L<=2048 && HC<=8 && 256<=D<=1024 &&
                   K in {4,8} && dilation<=3
fastWeightRepresentation = directF32 || (packedBF16 && bf16FastEnvelope)
fastInputsContiguous = keys && hidden && value && (state absent || state contiguous)
fastCapacityDomain = B<=256 && L<=4096 && HC<=16 &&
                     64<=D<=2048 && K<=16 && dilation<=16
materializedCapacity = B*L*HC*D <= (512 MiB / sizeof(float))

useMaterializedFast = fastWeightRepresentation && fastInputsContiguous &&
                      fastCapacityDomain && materializedCapacity

H0: if packWeights: launch pack_weights and replace four compute pointers
H1: if !useMaterializedFast: R-006; else enter strict selector H2
```

该 admission 只控制实现选择。特别是 FP16/FP64、混合 dtype、strided weights、
strided activation/state、BF16 fast envelope 外或 materialized 容量外均进入 R-006，
不会因 fast 条件不满足而报错。

### strict selector H2..H6

```text
H2 if no_state && K==4 && dilation<=3 && L>=1024 &&
      D>=768 && alignedD<=1024:
     R-001
H3 else if no_state && K==8 && dilation<=3 && L>=32 &&
           D>=768 && alignedD<=1024:
     R-002
H4 else if no_state && K==4 && dilation<=3 && 32<=L<1024 &&
           D>=768 && alignedD<=1024 &&
           (B*HC>=16 || D%64!=0 || HC>=8):
     R-005
H5 else launch EngramStrictFastKernel<false,false>
H6      Device useFusedConv4 = no_state && K==4 && dilation<=3 &&
                                L>=32 && D>=768 && alignedD<=1024
        if useFusedConv4: R-003 else R-004
```

R-001→R-002→R-005→ordinary 的判断与短路发生在 Host launcher；R-003/R-004 的
二选一发生在 ordinary Device body。`reserved` 在进入 strict launcher 前被无条件写 0，
所以 D-001 不参与上述 selector。

### 启动拓扑

```text
engram_gate_fusion_npu
  -> calc_engram_gate_fusion_launch_config(params)
  -> launch_engram_gate_fusion_bfloat16(current_stream)
       [packWeights] pack_weights<P,1024 threads>
       if useMaterializedFast:
         calc_engram_strict_fast_launch_config
         reserved=0
         exactly one strict kernel: R-001/R-002/R-005/R-003/R-004
       else:
         row_stats<P,1024 threads>
         emit<P,1024 threads>                         # R-006
```

多 kernel 边只依赖同一 current stream 的 launch 顺序；strict kernel 内部再用硬事件和
`SyncAll<true>`。launcher 与 kernel entry 为 `void`，`OpCommand::RunOpApi` lambda 固定
返回 0；异步 Device 错误如何映射为框架错误不在该派生源码中展开。

## P0 / R-006：universal row-stats + causal emit

### 选择与链路闭包

| 层级 | 关系 |
|---|---|
| 触发谓词 | Host 契约通过且 H1 `useMaterializedFast==0` |
| Host producer | `check_contract`、weight views、actual strides、count、`packWeights`、outer admission |
| POD consumer | `EngramGateFusionParams` 同时传给 pack、row_stats、emit |
| launch 节点 | 可选 `engram_gate_fusion_pack_weights` → `engram_gate_fusion_row_stats` → `engram_gate_fusion_emit` |
| Device body | `PackWeightsVF`；`RowStatsVF/BlockReduceSum`；`EmitVF/NormalizedValue` |
| 核任务 | row_stats：core 以 `row=core+n*P` 取 `[B,L,HC]` row，1024 threads 协作 D reduction；emit：`P*1024` threads 对 `O+stateCount` 元素 grid-stride |
| GM/workspace | actual-stride activations/state；直接或 packed FP32 weights；`stats[2,R]`；无 normalized GM |
| UB/L1/L0 | row_stats 只用 UB 地址 0 的 32-warp reduction scratch；emit 每线程 scalar；无 TBuf、L1、L0 |
| 同步/lifetime | block 内 `__sync_workitems` 保护两级 reduction；kernel 间用同 stream 顺序保护 packed weights 与 stats |
| tail/mask | D loop 为 `d=tid+n*1024`；emit linear loop 以 `<total` 截尾；K、S、非连续 stride 都由 scalar 64-bit 地址处理，无 padding lane |
| 精度 | 四个协作式 FP32 reduction（key²、hidden²、dot、gated-value²）；gate/rms 写 stats；每个 output 对每个 tap 按源码顺序重建 `u` 并从 `+0` 累加；最终 `__float2bfloat16_rn` |
| fallback/边界 | 是全合法域终点，无后续 Device Route；只有 dtype/shape/device/正值/int64 地址契约失败才在 launch 前终止 |

### 可落地伪码

```text
if packWeights:
  parallel_for i in [0, packedWeightCount):
    view = segment_of(i)
    src = view.rank==2 ? row*stride0+col*stride1
                       : row*stride0+tap*stride2
    packed[i] = FP32_RNE(load_as_declared_dtype(src))

for row = core; row < B*L*HC; row += P:
  all 1024 threads reduce key^2 and hidden^2 over strided D
  reduce ((key/keyRms)*w1) * ((hidden/hiddenRms)*w2)
  g = sigmoid(sign(raw)*sqrt(max(abs(raw),1e-6)))
  reduce (value*g)^2
  thread0 writes stats[row]=g; stats[R+row]=valueRms

for linear = core*1024+thread; linear < O+stateCount; linear += P*1024:
  if output element:
    for tap: source = reconstruct_from(value,stats,convNorm) or state or 0
    write BF16_RNE(silu(sum(+0, weight[tap]*source)) + value*g)
  else:
    choose normalized tail / old state / 0 and write contiguous stateOut
```

## P1 / R-005：strict K4 batch8

### 选择与链路闭包

| 层级 | 关系 |
|---|---|
| 触发谓词 | H1 strict；H2/H3 false；无 state，K4，`dilation<=3`，`32<=L<1024`，`D>=768`，`alignedD<=1024`，且 `B*HC>=16 \|\| D%64!=0 \|\| HC>=8` |
| selector/key/template | Host `useEightRowK4` → `engram_strict_fast_k4_batch8_kernel_bfloat16` → `EngramStrictFastKernel<false,false,true>` |
| Host/POD | strict config 把 `normTile>=8`、`convTile=8`，并生成 task counts、blockDim |
| Device body | `NormalizeAllRows` → barrier → `WriteStateOutput` → `ConvolveAllRows/ConvolveTile` → 两次 `Conv4Rows4TapsUniqueVF` |
| 核任务 | `(B,HC,length-tile)`；可整除时每 sequence 固定多 lane，否则 task 以 `blockIdx+n*P` 轮转；conv tail 为 `1..8` |
| GM/UB | 完整 FP32 normalized GM + gate GM；7 个 VECCALC TBuf；work 为 K4 coefficient、`S+8` unique timeline、8-row residual/output scratch |
| L1/L0 | 不使用 |
| 同步/lifetime | MTE2/V/MTE3/S 六类 event；`L>1` 用 `SyncAll<true>` 发布 normalized/gate；同一 block 的 UB 在 normalize/conv 两相复用 |
| tail/mask | D pad 到 64；首个/第二个四行 group 分开寄存器卷积；`currentLength<=4` 时不执行第二组；只向 GM 写有效 D |
| 精度 | 每四行每 tap 从 `+0` FP32 accumulator 起；SiLU、FP32 residual Add 后 `CAST_RINT` |
| fallback/边界 | batch8 guard false且 K4 fused guard true→R-003；K4 guard false→R-004；outer H1 false→R-006 |

### 可落地伪码

```text
materialize every u row to normalized GM; publish gate; all_blocks_barrier()
for conv tile up to 8 rows:
  gather 4 coefficient rows once
  stage unique timeline of S+8 FP32 rows, prefix zero because no state
  Conv4Rows4TapsUniqueVF(rows 0..3)
  if currentLength>4: Conv4Rows4TapsUniqueVF(rows 4..7)
  batch SiLU + value*gate + BF16_RNE; write only currentLength*D
```

## P2 / R-003：strict K4 four-row

### 选择与链路闭包

| 层级 | 关系 |
|---|---|
| 触发谓词 | H1 strict；H2/H3/H4 false；ordinary Device guard `!state && K==4 && dilation<=3 && L>=32 && D>=768 && alignedD<=1024` |
| selector/key/template | ordinary entry `engram_strict_fast_kernel_bfloat16` → `<false,false>` → `useFusedConv4` |
| Host/POD | 默认 strict tiling，conv tile `1..4`；短 K4 规则不适用，因为本 route `L>=32` |
| Device body | materialized normalize → barrier → K4 unique timeline → `Conv4Rows4TapsUniqueVF` → output |
| 核任务 | sequence-lane pinning 或 round-robin task；每 task 最多 4 output rows |
| GM/UB | 完整 normalized/gate；K4 packed weight；`S+4` timeline；7 TBuf，无 L1/L0 |
| 同步/lifetime | 与 R-005 相同；coefficient/source/accumulator/residual 在 work TBuf 按 phase alias |
| tail/mask | causal prefix填 `+0`；length tail `1..4`；aligned D padding lane不写 GM |
| 精度 | 四 tap 按 0→3 顺序、`+0` seed；post-conv FP32，末端 BF16 RINT |
| fallback/边界 | Device guard false→R-004；outer H1 false→R-006 |

### 可落地伪码

```text
for owned tile<=4:
  coefficient[0..3,D] = gather(packedConvWeight)
  timeline[0:S+currentLength,D] = 0
  fill nonnegative source times from normalized GM
  for each 64-lane slice and row<4:
    acc=+0; for tap=0..3: acc += timeline[row+tap*dilation]*coefficient[tap]
  apply SiLU + residual and write valid rows/D
```

R-001/R-003/R-005 中，当 `UsesPersistentTapWeights()` 为真（无 state K4、`dilation<=3`、`L>=127`、
`D>=768`、`alignedD<=1024`、`HC>=4`、`B*HC<=16`），packed K4 weight 先 gather 成
tap-major coefficient rows，再复制回 `paddedBuf` 跨 sequence tile 复用；否则每 tile 从
packed channel-major weight gather coefficient。

## P3 / R-004：strict bounded generic

### 选择与链路闭包

| 层级 | 关系 |
|---|---|
| 触发谓词 | H1 strict 且未命中 R-001/R-002/R-005，ordinary Device `useFusedConv4==false` |
| selector/key/template | `engram_strict_fast_kernel_bfloat16` → `EngramStrictFastKernel<false,false>` generic branch |
| Host/POD | 默认 tiling；无 state K4 且 `1<L<32` 时 norm/conv tile 至少 2；其余 conv tile `<=4` |
| Device body | `NormalizeAllRows`/`ProcessSingleToken`；`WriteStateSequence*`；generic `ConvolveTile` 的 tap×row 循环 |
| 核任务 | L=1 每 `(B,HC)` sequence 由唯一 core 顺序完成；L>1 为 sequence-tile pinning/round-robin，state writer有独立唯一 owner |
| GM/UB | 完整 normalized/gate；7 TBuf；generic work 至少 10 FP32 rows；packed或32B-strided weight staging；packed/strided state；whole/tiled state slab |
| L1/L0 | 不使用 |
| 同步/lifetime | L=1 无全核 barrier；L>1 normalize 后 `SyncAll<true>`；stateOut 与 conv 在 barrier 后可由不同 owner 并行，因二者写不同 GM |
| tail/mask | `currentLength<=4`；K loop `0..K-1`；D pad；state channel tile 处理 `currentDim`；K=1 时 S=0 跳过 state writer |
| 精度 | source/weight FP32 `Mul` 后逐 tap `Add` 到 `+0` accumulator；state BF16 cast FP32；末端 RINT |
| fallback/边界 | strict 内最终终点；各 capacity guard false只切换 helper，不退出；outer H1 false→R-006 |

### 有界 helper selector

| Guard | true 路径 | false 路径 |
|---|---|---|
| `CanPackConvWeights: K<=8` | 整段 `[D,K]` 入 UB 后 Gather | 每 tap 以 32B stride `DataCopyPad+Gather` |
| `CanUsePackedState: state && K==4 && S<=16 && alignedD<=1024` | 整个 `[D,S]` state slab 缓存 | 每个 state row strided load |
| `CanPackStateOutput: S!=0 && alignedD*S*2 <= 48KiB`（`alignedD<=1024`），否则 `S!=0 && alignedD*S*2<=32KiB` | whole-slab writer | `StatePackChannels` 分 channel tile；`S=0` 时 state writer 在 selector 前返回 |
| `CanBatchStateRows: L>1 && (S<=10 \|\| kFusedK8) && CanPackStateOutput` | 批量搬尾 normalized rows并 Scatter | row-wise/tiled writer |
| `D%64==0` | 多 row 2-D DMA | 逐 row pad/copy |

### 可落地伪码

```text
if L==1:
  owner(sequence): normalize row; optionally cache state; write new state; convolve row
else:
  materialize all u; all_blocks_barrier()
  unique state owner builds [D,S] from old-state suffix + normalized suffix
  each conv owner:
    for tap in [0,K):
      coefficient = packed_gather(tap) or strided_load_gather(tap)
      for row<currentLength:
        source = normalized / packed-state / strided-state / zero
        acc[row] += source*coefficient
    postprocess and write currentLength*D
```

state writer 先以时间主序读取 row，再 Scatter 成 channel-major `[currentDim,S]`，最后
连续 MTE3 写 `stateOut`；不能把 normalized row-major 尾部直接当 `[D,S]` view。

## P4 / R-001：strict K4 scale reconstruction

### 选择与链路闭包

| 层级 | 关系 |
|---|---|
| 触发谓词 | H1 strict 且 `!state && K==4 && dilation<=3 && L>=1024 && D>=768 && alignedD<=1024` |
| selector/key/template | H2 `useScaleRecompute` → `engram_strict_fast_scale_kernel_bfloat16` → `<true,false>` |
| Host/POD | norm batch capacity为 8；conv tile仍 `1..4`；`S<=9` |
| Device body | `ComputeNormStats*` 发布 gate/RMS，仅末 S rows materialize；barrier；`ReconstructValueNormRowsVF` + K4 unique conv |
| 核任务 | normalize sequence tile与 conv sequence tile分配同公共规则；每个 conv tile按需拉 `S+currentLength` value/scalars |
| GM/UB | gate写 stats workspace前半；RMS写 normalized workspace的 row-scalar prefix；末 S 行仍按正常 row offset物化；UB另 staging BF16 value、gate/RMS与 convNorm |
| L1/L0 | 不使用 |
| 同步/lifetime | scalar 必须以 MTE3 发布；`SyncAll<true>` 后任意 conv owner才可重建；normalized workspace中 scalar prefix与末 S row区域由地址条件避免冲突消费 |
| tail/mask | source timeline `S+4`；负前缀为 0；validCount只覆盖非负 sourceTime；D pad到64 |
| 精度 | 严格按 `(value*gate)/rms*convNorm` 重建，不能预合并 `gate/rms`；K4 +0 seed与输出 RINT不变 |
| fallback/边界 | H2 false继续 R-002/R-005/ordinary；outer H1 false→R-006 |

### 可落地伪码

```text
for each row:
  compute gate and gatedRms
  MTE3 stats[row] = gate
  MTE3 normalizedWorkspace[rowScalarOffset] = gatedRms
  if row+S>=L: materialize full u row for state output
all_blocks_barrier()
for conv tile<=4:
  load BF16 value timeline + FP32 gate/rms scalars + convNorm
  reconstruct each valid source row as (value*gate)/rms*convNorm
  run K4 unique-timeline convolution; postprocess/output
```

## P5 / R-002：strict K8 unique timeline

### 选择与链路闭包

| 层级 | 关系 |
|---|---|
| 触发谓词 | H1 strict；H2 false；`!state && K==8 && dilation<=3 && L>=32 && D>=768 && alignedD<=1024` |
| selector/key/template | H3 `useFusedK8` → `engram_strict_fast_k8_kernel_bfloat16` → `<false,true>` |
| Host/POD | norm tile至少 8、conv tile 8；`S=7*dilation<=21` |
| Device body | 完整 materialize → barrier → 8 coefficient rows + `S+8` timeline → 最多两次 `Conv4Rows8TapsUniqueVF` |
| 核任务 | 每 conv task最多 8 rows；sequence pinning或round-robin |
| GM/UB | 完整 normalized/gate；K8 packed weights为 `32*alignedD` bytes；work包含 8 coefficient rows与 unique timeline；无 L1/L0 |
| 同步/lifetime | 公共六类 event与全核 barrier；K8 state output可别名到较大的 work slab，但本 route无输入 state |
| tail/mask | 第二组仅 `currentLength>4`；最后 tile `1..8`；D padding不出 GM |
| 精度 | 8 tap按索引顺序从 `+0` 累加，post-conv FP32，BF16 RINT |
| fallback/边界 | H3 false继续 R-005/ordinary；outer H1 false→R-006 |

### 可落地伪码

```text
materialize all u; all_blocks_barrier()
for tile<=8:
  gather w0..w7; stage zero-prefixed unique timeline[S+8,D]
  conv rows 0..3 with eight FP32 coefficients
  if tail has rows 4..7: conv second four-row window
  batch SiLU/residual/RINT and write only valid rows
```

## Dormant 实现边界

### D-001：independent-tile strict stream

`launch_engram_strict_fast_bfloat16` 只有 `tiling.reserved==1` 才发射
`engram_strict_stream_kernel_bfloat16`，但 outer launcher 在每次调用前都执行
`fastConfig.tiling.reserved=0`。因此 `ProcessIndependentTiles` 的“每 tile 重算 causal halo、
不做 `SyncAll`”路径在生产控制流不可达。重新启用至少需要 Host producer 不再强制清零、
重新证明 overlap publication、tile/state owner、halo 重算精度与 UB 上限；不能仅改分支常量。

### D-002：SIMT materialized pair

`engram_gate_fusion_fast_row_materialize` 与
`engram_gate_fusion_fast_emit_materialized` 有完整 entry/body，前者以每 warp一 row写完整
normalized GM，后者逐 output/state scalar消费；但 launcher 从未调用这两个 entry。
当前 `useMaterializedFast` 的实际消费者是 strict-fast launcher。要启用必须新增成对 launch、
在同 stream 建立 producer→consumer边，并重新核算与 strict 的 selector互斥关系。

### D-003：strict 四行 norm helper

`ComputeNormStats4RowsVF` 与 `ComputeNormStats4RowsExplicitVF` 都保留定义；实际 caller 的
四行循环写成 `for (; false && groupedRows+4<=rowCount; ...)`，所以 active normalize
只以 `ComputeNormStats2RowsVF` 成对处理，再由 `ComputeNormStatsVF` 处理单行尾块。
重新启用需移除常量 false 并证明四行寄存器对象不发生别名；`NormBatchRows()==8` 只表示
八行 DMA/UB staging，不表示 active arithmetic helper一次计算八行。

## 跨 Route 资源、同步与精度边界

### strict UB 公式

令 `A=alignedDim`、`N=NormBatchRows()`、`Cpack=StatePackChannels()`：

```text
inputRows = max(3N, 6, (R-001 ? S+4 : 0))
convWorkRows = R-002 ? K+S+8
             : fusedK4 ? (R-005 ? 20 : 8)+S+(R-001 ? 1 : 0)
             : 10
workRows = max(convWorkRows, N+3)
paddedElementBytes = (hasState || K>8) ? 32 : 4K
statePackedBytes = aligned32(2*Cpack*S), except K8/S0 minimum allocation 32
stateIndexBytes = S ? 4A : 32
reduceBytes = 256N
UB = 2A*inputRows + 4A*workRows + A*paddedElementBytes + 4A
     + statePackedBytes + stateIndexBytes + reduceBytes
```

所有这些缓冲是 `VECCALC TBuf`，没有 TQue、L1 或 L0。Route guard、K<=16、
`alignedD<=2048` strict envelope与 state channel tiling共同约束 256KiB UB 设计上限；
R-006 则不采用这套静态 UB row slab。

### strict task owner 与同步

Normalize/conv task 都先编码 `(sequenceIdx,tileIdx)`，`sequenceIdx=batch*HC+hc`：

```text
if C<=P && P%C==0:
  lanes=P/C; sequence=blockIdx%C; lane=blockIdx/C
  tile = lane + n*lanes
else:
  task = blockIdx + n*P
```

`L>1` 时每个 launched block 都参与 `SyncAll<true>`，即使没有 normalize task；该边保护
normalized/gate producer与任意 conv/state consumer。`L=1` 由同一 sequence owner顺序执行
normalize→state→conv，绕过全核 barrier。MTE2↔V、V↔MTE3、V↔S 的六组 hard event
保护 UB alias与搬运完成；R-006 的 kernel间边由 current stream 顺序提供。

### state、尾块和 mask

- strict 的 D 对齐是 64 FP32 lanes；copy-in先清零/pad，reduction使用 all-lanes mask，
  copy-out只写有效 D。R-006 用 loop bound而非 padding mask。
- strict norm staging `N=3/4/8`，active arithmetic以 2 rows + 1-row tail；conv tile为
  `1..4` 或 R-002/R-005 的 `1..8`。
- state source始终是“旧 state后缀 + 当前 normalized后缀”；无 state的负时间为 `+0`。
  `S=0` 时所有 state pack/index/write分支跳过。
- R-006 output/state用统一 linear范围 `<outputCount+stateCount` 截尾；所有地址及 count
  都保持 uint64/int64，避免把大域压到 uint32。strict 只在明确 bounded envelope内把
  shape写入 uint32 tiling。

### 数值边界

- 四种 weight表示最终都进入 FP32 compute contract；Direct FP32不发生额外转换，其他
  表示先 pack。FP64 converter显式使用 RNE并处理 Inf/NaN和范围边界。
- strict register reduction、strict Vector API 与 R-006 SIMT reduction树不同；卡片只保证
  每条路线各自保持源码规定的阶段和 tap顺序，不把不同 reduction树宣称为逐 bit相同。
- gate 使用 `abs → clamp_min(1e-6) → sqrt → sign → sigmoid`；NaN不应被改写成普通零。
- convolution从 canonical `+0` 开始；从首个乘积直接初始化会改变全零卷积的 signed zero。
- residual在 FP32相加，最后才 BF16 RNE/RINT；state中的 normalized row也在写出时转 BF16。

## Fallback 图与失败边界

```text
Host contract failure ---------------------------------> launch前终止
Host contract success
  -> optional NPU FP32 weight pack
  -> strict admission false ---------------------------> R-006 (合法域终点)
  -> strict admission true
       -> long no-state K4 ----------------------------> R-001
       -> no-state K8 specialized ---------------------> R-002
       -> no-state K4 batch8 --------------------------> R-005
       -> ordinary K4 Device guard true ---------------> R-003
       -> ordinary guard false ------------------------> R-004 (strict终点)
```

- activation/state必须 BF16；weight只接受四种真实浮点 dtype；device、rank、shape、正属性
  或 int64乘法证明失败均在 launcher前停止。
- `norm_eps` 没有正值或 finite guard；迁移实现若新增该 guard会改变当前合法域。
- strict要求连续 activation/state，但这是 admission，不是公共契约；R-006按 stride消费。
- `blockDim` 查询为空/零时回退为 1；这不是数值 fallback。
- Device entry和launcher不返回状态码；lambda固定返回0，源码不展示异步错误转换契约。

## 实现复用检查

| 复用点 | 必须一并迁移 | 不能遗漏的出边 |
|---|---|---|
| 多 dtype weights | dtype code、完整 weight view、FP64 RNE converter、四段 dstOffset、same-stream pack edge | 任一非连续FP32组合都打包；只有全连续FP32直读 |
| outer selector | BF16同构连续 envelope、activation/state连续性、bounded fast domain、512MiB materialized guard | false必须进R-006，不得转成拒绝 |
| strict selector | H2 scale→H3 K8→H4 K4 batch8→ordinary H6 | 重要性排序不能改变短路顺序 |
| universal | row-level collaborative stats、element-level emit、actual stride、64-bit address | 不得误分配full normalized workspace |
| strict memory | 7 TBuf公式、state budget、scalar alias、whole/tiled writer | capacity guard false切helper或R-006 |
| 同步 | current-stream kernel edge、strict hard events、`SyncAll<true>`、L=1 owner | 所有blocks必须参加全核barrier |
| 精度 | double→float、gate顺序、gated RMS、+0 tap accumulation、末端BF16 RNE | 不得跨阶段重关联 |
| dormant代码 | reserved强制0、无caller kernels、`false &&` helper | 不能因源码存在而列为active Route |

## 源码索引

| 链路阶段 | 生产文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| Python facade | `cann_bench/__init__.py` | `engram_gate_fusion` | 显式 Python 到 torch.ops 入口 |
| schema/Meta/Host | `op_plugin/engram_gate_fusion_plugin.cpp` | `check_contract`、`make_weight_view`、`engram_gate_fusion_meta/npu` | dtype/shape/device、stride、count、workspace、outer selector |
| POD/launcher ABI | `op_kernel/engram_gate_fusion_launch.h` | 两组 Params/Tiling/LaunchConfig | 通用和 strict 的按值契约 |
| universal/outer launch | `op_kernel/engram_gate_fusion_kernel.cpp` | `PackWeightsVF`、`RowStatsVF`、`EmitVF`、`launch_engram_gate_fusion_bfloat16` | pack、SIMT fallback、strict admission与启动图 |
| strict-fast | `op_kernel/strict_fast_impl.h` | `EngramStrictFastKernel`、`calc/launch_engram_strict_fast_*`、`Conv4Rows*TapsUniqueVF` | tiling、五条 active strict Route、UB、event、state与 dormant分支 |
| 构建接线 | `csrc/ops/engram_gate_fusion/CMakeLists.txt` | `register_direct_launch_op` | 一个 kernel translation unit和一个 plugin translation unit |
| 官方相关接口 | Frontmatter 固定 `fused_causal_conv1d/README.md` | FusedCausalConv1d | causal-conv/state相关背景，非同构入口 |
| 官方相关 950 实现 | Frontmatter 其余四个固定文件 | causal state、VF、DataCopyPad、event symbols | 搬运/寄存器/同步参考，不能证明派生 Host/Route |

构建时 `engram_gate_fusion_kernel.cpp` 在 translation unit 末尾 include
`strict_fast_impl.h`，所以 universal与 strict entry都进入同一个 Bisheng AIV对象；plugin由
G++编译并与 kernel对象、extension module一起链接。该构建事实不改变“官方相关参考”与
“派生补充实现”的证据边界。
