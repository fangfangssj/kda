---
type: Operator
title: "ApplyRotaryPosEmb 950 Direct Invoke 算子知识"
description: "说明 cann_bench::apply_rotary_pos_emb 在 950 上的 RoPE 语义、stride/broadcast 契约、Host checked arithmetic、五级 Route、POD/launch 图、UB/GM 资源包络以及尾块、精度和 fallback。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/apply_rotary_pos_emb_def.cpp"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_kernel/apply_rotary_pos_emb_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/posembedding/apply_rotary_pos_emb/op_host/config/ascend950/apply_rotary_pos_emb_binary.json"
    role: implementation
created_at: "2026-08-24T19:31:12Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ApplyRotaryPosEmb 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 契约 |
|---|---|
| 入口 | `cann_bench::apply_rotary_pos_emb` → `ApplyRopeNpu` → fast/dense/generic C launcher；Direct Invoke，不是 ACLNN。 |
| 数学 | 对 `query`、`key` 分别计算 `y=x*cos+rotate(x)*sin`；`half` 配对前后半维，`interleaved` 配对相邻偶奇维。 |
| 数据 | `query/key` 是同 shape 的 4-D strided tensor；`layout=0` 为 `[B,S,N,D]`，`layout=1` 为 `[B,N,S,D]`。`cos/sin` 是紧凑半长系数，可按 B、S 广播。 |
| dtype | 四输入同为 FP32、FP16 或 BF16；两个输出与输入 dtype/shape 相同，但由 `torch::empty(sizes)` 新建为 dense contiguous。 |
| Route | P0 fast 分三条：BSND-half 紧凑系数、BSND-interleaved 展开系数、BNSD 展开系数；P1 dense 为 stride-aware 行向量；P2 generic 为 stride-aware 逐元素兜底。 |
| 执行单元 | 所有 entry 都声明 `AIV_ONLY`，device body 在 AIC 上早退；无 Cube、L1、L0 或通信路径。 |
| 临时资源 | 只有两条“展开系数”fast Route 分配一个 `[2,S,D]` GM 临时张量；其它 Route 不分配 tensor-sized 中间量。调用者 workspace 为 0。 |

## 语义、shape、dtype 与 layout

### RoPE 旋转语义

令 `h=D/2`，紧凑系数索引为 `j∈[0,h)`。对每个 `(b,s,n)` 行和 `d∈[0,D)`：

```text
half:
  j    = d                    if d < h else d-h
  peer = d+h                  if d < h else d-h
  sign = -1                   if d < h else +1

interleaved:
  j    = floor(d/2)
  peer = d+1                  if d is even else d-1
  sign = -1                   if d is even else +1

y[...,d] = x[...,d] * cos[broadcast(b),broadcast(s),j]
         + sign * x[...,peer] * sin[broadcast(b),broadcast(s),j]
```

等价地，`half` 把 `[x0..x(h-1), xh..x(D-1)]` 旋转成 `[-xh..-x(D-1), x0..x(h-1)]`；`interleaved` 把每个 `(x2j,x2j+1)` 旋转成 `(-x2j+1,x2j)`。mode 字符串只允许 `"half"/"interleaved"`，Host 分别编码为 `1/2`。继承 helper 中的 quarter 分支不由本 Direct schema 选择。

### 输入、广播与输出

| 参数 | shape | stride/layout 契约 | dtype/结果 |
|---|---|---|---|
| `query`,`key` | layout 0：`[B,S,N,D]`；layout 1：`[B,N,S,D]`；两者 shape 必须完全相同 | 必须是 strided layout、非负 storage offset、各维非负 element stride；不要求输入 contiguous | 三种支持 dtype 之一且相同；分别写 `queryOut/keyOut` |
| rank-2 `cos`,`sin` | `[Cs,D/2]`，`Cs∈{1,S}` | `Cb=1`；`Cs=1` 时沿 S 广播，否则逐位置取行 | 与 q/k 同 dtype、shape 相同 |
| rank-3 `cos`,`sin` | `[Cb,Cs,D/2]`，`Cb∈{1,B}`、`Cs∈{1,S}` | 维长为 1 的轴广播；一般 Route 按独立 cos/sin stride 取址 | 与 q/k 同 dtype、shape 相同 |
| 输出 | 与 q/k sizes 相同 | fresh dense contiguous；若输入是 transpose/narrow/expand view，输出保留逻辑轴次序但不保留输入 stride/storage offset | 与各自输入同 dtype |

公共 shape 域是 `1<=B<=8192`、`1<=S<=1048576`、`1<=N<=128`、`2<=D<=2048` 且 `D%2==0`。由于 q/k shape 相同，当前闭包中 `qN==kN==N`；POD 仍保留 `QN/KN` 两字段，复用时不能因此交换字段。

## Host checked arithmetic 与资源前检

`ValidateApplyRope` 同时被 Meta 与 NPU 入口调用；NPU 路径在 `DeviceGuard`、输出分配、stream 获取和任何 launch 之前完成全部检查。Host 的安全顺序应保持为：

```text
validate enum/rank/dtype/device/shape/broadcast
  -> checked element counts and reachable spans
  -> route-independent resource envelope
  -> build logical strides
  -> allocate dense outputs
  -> choose fast, else dense, else generic
  -> acquire stream and launch
```

### Checked arithmetic 闭环

| 检查量 | 实现关系式 | 拒绝条件/原因 |
|---|---|---|
| q/k 元素数 | `E=B*S*N*D`，用 `__int128` 乘法后转 `int64` | 维为负、乘积超过 `INT64_MAX`，或与 tensor `numel()` 不一致 |
| 系数元素数 | `Ec=Cb*Cs*(D/2)`，同样用 checked product | 与任一 `cos/sin.numel()` 不一致 |
| 逻辑 pair | `P=E/2` | `E` 非正/非偶数或 `P>2^25` |
| fresh 输出 | `Bout=(Eq+Ek)*elementSize`，用 `__int128` | 溢出或超过 512 MiB |
| 可达 element offset | `M=sum_i((size_i-1)*stride_i)`，逐项在 `__int128` 累加 | stride 为负或 `M>INT64_MAX` |
| storage 范围 | `storageOffset+M` | storage offset 为负或和超过 `INT64_MAX` |
| 单输入可达 span | `(M+1)*elementSize` | 溢出或超过 1 GiB |
| 四输入 span 总和 | `sum(span(q),span(k),span(cos),span(sin))` | 超过 2 GiB |

这些检查证明后续 signed-64 stride/offset 计算有 Host 前提，但不替代框架自身的 storage 边界不变量。一般复用契约是：先验证再分配，所有乘加先在更宽类型完成，只有通过范围检查后才窄化；不得先用 `int64_t` 乘出结果再检查。

### Route-independent 与 Route-specific 包络

| 包络 | 上限/条件 | 作用 |
|---|---|---|
| q 数据元素 | `E<=64*1024*1024`；由 `P<=2^25` 同时蕴含 | fast 谓词的显式数据上限，也是通用入口总工作上限的一部分 |
| fast 展开 GM | `2*S*D*elementSize<=4 MiB` | 限制 `[2,S,D]` 中间张量；BSND-half 紧凑 Route 不展开，故不受此项限制 |
| fast 平台资源 | `AIV cores>0`、`UB>0`、`plan.blockDim<=cores` | 无有效 fast plan 时继续 dense/generic，不提交非法 grid |
| fast 展开节点 | `cores>=56` 且 `UB>=192 KiB` | 只约束需要 K1 展开的 fast Route |
| dense 行搬运 | `D*elementSize` 是完整 32B block，且各 x/cos/sin 的 D stride 都为 1 | 不满足即退到逐元素 generic |

## Route 选择与退化关系

### Host 决策树

先定义：

```text
exactSD = cos/sin rank-2 and shape == [S,D/2]
dense4  = q,k,cos,sin all contiguous
fastShape = D in {64,128} and q/k same N and E>0
fastWorkspace = (layout==0 && mode==half)
             or (2*S*D*elementSize <= 4 MiB)
```

当 `dense4 && exactSD && fastShape && fastWorkspace` 时构造并严格复核 `FastPlan`；只有 `IsApplyRopeFastPlanValid` 为真才进入 fast。否则构造 `DensePlan`；`DensePlan.blockDim>0` 才进入 dense；其余合法输入进入 generic。fast 内部再按 `layout/mode` 分发，顺序不可交换成模糊的 layout-only 选择。

| 重要性 | 可达性 | Route | 精确命中条件 | 节点拓扑 | 不命中时 |
|---|---|---|---|---|---|
| P0 | active | R-001 Compact BSND-half | 公共校验通过；`dense4 && exactSD && 0<E<=64Mi && D∈{64,128} && layout=0 && mode=1`；`MakeFastPlan` 与 validator 均成功 | 单 K：compact `[S,D/2]` → `ApplyRotaryPosEmbCompactAB<T>` | fast selector 不成立或 plan 无效时构造 dense；dense 无效再 generic |
| P0 | active | R-002 Expanded BSND-interleaved | R-001 的 fast 公共条件，但 `layout=0 && mode=2`，且 `2*S*D*e<=4MiB`、平台允许 56-block K1 | K1 `[S,D/2]→[S,D]` → K2 `ApplyRotaryPosEmbBroadcastAB<T>` | 同上 |
| P0 | active | R-003 Expanded BNSD | fast 公共条件；`layout=1 && mode∈{1,2}`；展开 GM 与 56-block 资源包络成立 | K1 `[S,D/2]→[S,D]` → K2 `ApplyRotaryPosEmbABAAndBA<T,true>` | 同上 |
| P1 | active（条件 fallback） | R-004 Dense row-vector | `!useFast`；`D*e%32==0`；q/k 与 cos/sin 的 D stride 均为 1；DensePlan UB/core 检查成功 | 单 K：一个 row 内复用紧凑系数并依次处理 q/k | `DensePlan.blockDim==0` 退 generic |
| P2 | active（最终 fallback） | R-005 Generic stride-aware | 公共校验通过且 `!useFast && !useDense` | q、k 各一个 SIMT/VF grid-stride Kernel | 最终合法域；无 device 运行时重试 |

五条 Route 都由生产 Host selector 可达；R-004/R-005 的“条件”仅表示它们要求前序 Route 未命中，不是不可达或假设性实现。fast→dense→generic 是实现能力逐级放宽，不是出错后的重放：Kernel 一旦提交，源码没有捕获异步失败再切换 Route 的机制。

## P0 Fast Routes

### FastPlan 生成与复核

公共量：`split=(mode==half?2:1)`，`a=32/elementSize`，`dAlign=ceil_align(D/split,a)*split`，`totalHeads=2*N`。Host 从平台读取 AIV core 数 `C` 与 UB 字节 `U`。

BSND plan：

```text
BS             = B*S
blockFactorBS  = ceil_div(BS,C)
blockNumBS     = ceil_div(BS,blockFactorBS)
blockTailBS    = BS-(blockNumBS-1)*blockFactorBS
rowsInUb       = floor_align(U/4,32)/(dAlign*elementSize)
ubLoopN        = ceil_div(2N,rowsInUb-1)
ubFactorN      = min(ceil_div(2N,ubLoopN),4095/split)
ubTailN        = 2N-(ubLoopN-1)*ubFactorN
ubFactorBS     = min(blockFactorBS,max(1,rowsInUb/(2N+1)))
```

复核器不只检查正数，还重新证明分核覆盖、N 尾覆盖、`uint16` DMA blockCount、`uint32/int32` 中间计数以及实际 UB 字节数。若截断 `ubFactorN` 后不再满足 `ubLoopN==ceil_div(2N,ubFactorN)`，plan 会被拒绝，而不是带着自相矛盾的 tail 进入 device。

BNSD plan 先比较 B-first 与 S-first 两种分核并选择 `blockNumB*blockNumS` 较大者；整除时优先把整轴分给 C 个 core。随后：

```text
alignedDBytes = ceil_align(D*elementSize/split,32)*split
capacity      = U/(4*alignedDBytes)
ubFactorS     = min(blockFactorS,capacity/2,4095/split)
ubFactorB     = 1
ubFactorQN    = ubFactorKN = min(N,capacity/ubFactorS-1)  # capacity 足够时
blockDim      = blockNumB*blockNumS
```

实际代码按顺序缩减 `capacity` 并保守提升 B/N tile；复核器再次检查所有 loop/tail 恒等式、16/32 位搬运计数与
`4*ubB*ubS*ubN*dAlign*elementSize + 4*ubS*dAlign*elementSize <= U`。

### R-001：Compact BSND-half

| 闭包项 | R-001 实现契约 |
|---|---|
| selector/Host producer | `ApplyRopeNpu` 先证明四输入 contiguous、rank-2 系数严格为 `[S,D/2]`、`D∈{64,128}`，再由 `MakeApplyRopeFastPlan(...,layout=0,mode=1)` 生产 plan，并以 `IsApplyRopeFastPlanValid` 二次复核。 |
| POD/key/template | 无独立整数 tiling key，selector 本身就是 `useFast/layout/mode`；launcher 按值接收 `ApplyRotaryPosEmbFastPlan`，消费 `plan.blockDim/plan.ab`；核心字段是 `dSplitCoef=2`、BS block/factor/tail、`ubFactorBS/ubFactorN/ubTailN`；模板为 `ApplyRotaryPosEmbCompactAB<T>`。 |
| launch/entry/body | `LaunchApplyRopeFast{Float,Half,BFloat16}` 的首个 `if(layout==0&&mode==1)` 直接启动 `ApplyRopeCompact*Kernel`；entry 为 AIV-only，进入 `RunCompactAB<T>` → `Init/Process` → `CompactHalfVF`。无 K1。 |
| core/index | `blockIdx` 映射连续 BS 分片；q/k 用 `(globalBS*N+n)*D+d`，系数用 `(globalBS%S)*(D/2)+j`；一个 block 内按 BS tile 和拼接的 Q/K head tile 循环。 |
| GM/UB/L1/L0/workspace | GM 直接读 q/k 与 compact cos/sin、写 dense qOut/kOut；四路 TQue，各分配 2 个 buffer；无中间 GM、L1/L0、通信或 caller workspace。 |
| 同步/tail/precision | Queue 的 EnQue/DeQue/FreeTensor 保护 DMA→VF→DMA；BS 尾、S-wrap、N 尾和 halfD mask 分别裁剪；FP16/BF16 升 FP32，两个 Mul 后 Add/Sub，再 RINT 降回。 |
| fallback | 只在 launch 前因 fast predicate/plan validator 未命中而退 R-004/R-005；entry 提交后没有运行时 fallback。 |

这是唯一完全不物化全长系数的 fast Route。每个 block 负责一个连续 BS 区间；block 内 `globalBS=blockIdx*blockFactorBS+processedBS`，数据基址为：

```text
qBase = (globalBS*N)*D
kBase = (globalBS*N)*D
coef  = (globalBS % S)*(D/2)
```

`currBS` 同时被 `ubFactorBS`、当前 block 尾和 `S-globalBS%S` 截断，因此一次系数 DMA 永不跨越共享 `[S,D/2]` 表的 S 尾；跨 batch 时开始下一 tile。旧式的“B>1 一律 ubFactorBS=1”不是当前不变量，当前边界由 device 端 sequence-wrap clipping 精确保护。

每行系数在 UB 中保持 `D/2`，`CompactHalfVF` 直接加载成 FP32 RegTensor，分别计算前后半输出。`D=64 && T=float` 有 gather/select 专用向量体；其它 fast 组合走按 `VL_FLOAT32_SIZE` 分段的通用 compact VF。`currBS>1` 时 q/k 可在同一 input/output queue slot 中合并处理；复核器要求此时所有 `2N` head 可由一个 N loop 覆盖，避免合并缓冲越界。`currBS==1` 时按拼接的 `[Q heads,K heads]` 做 N 尾 tile，并在跨 Q/K 边界时拆分 GM copy。

资源字节账：

```text
coefficientAlign = ceil_align(D/2,32/elementSize)
UB = 4*ubFactorBS*ubFactorN*dAlign*elementSize
   + 4*ubFactorBS*coefficientAlign*elementSize
```

前一项是 input/output 两路 depth-2 queue，后一项是 cos/sin 两路 depth-2 queue。

### R-002：Expanded BSND-interleaved

| 闭包项 | R-002 实现契约 |
|---|---|
| selector/Host producer | `ApplyRopeNpu` 的 fast 公共谓词成立，且 `layout=0,mode=2`、`2*S*D*e<=4MiB`；fast planner/validator 还要求展开节点可用 56 AIV blocks 与至少 192KiB UB。 |
| POD/key/template | 无独立整数 tiling key；K1 使用标量 `elements=S*D,D,mode=2`；K2 按值消费 `FastPlan.ab`，其中 `dSplitCoef=1`、BS/UB/tail 字段由 Host 生产；K2 模板为 `ApplyRotaryPosEmbBroadcastAB<T>`。 |
| launch/entry/body | launcher 先启动 `ExpandApplyRope*Kernel` → `RunExpandApplyRopeCoefficients<T>`，同 stream 再启动 `ApplyRopeBroadcast*Kernel` → `RunBroadcastAB<T>` → `InterleaveModeVF`。两 entry 均 AIV-only。 |
| core/index | K1 以 `physicalBlock/2` 映射 28 个逻辑分区，按 row tile 展开；K2 与 R-001 相同按 BS 分片，系数索引改为 `(globalBS%S)*D+d`。 |
| GM/UB/L1/L0/workspace | K1 读 compact 系数，写一个 `[2,S,D]` 临时 tensor 的两 plane；K1 单 TBuf。K2 四路 depth-2 Queue；无 L1/L0/通信/caller workspace。 |
| 同步/tail/precision | K1 三个 `PIPE_ALL` barrier；K1→K2 只有同 stream 提交顺序；K2 用 Queue。K1 row 尾与展开 mask、K2 BS/S-wrap/N/VL 尾独立处理；K1 只重排，K2 FP32 分离乘加。 |
| fallback | 展开 GM、core/UB、fast plan 任一条件失败时，launch 前退 R-004/R-005；已提交 K1 后不会改走其它 Route。临时 Tensor owner 条件见“异步 owner 边界”。 |

K1 用固定 56-block AIV grid 把 `[S,D/2]` 紧凑系数扩成 `[S,D]`：interleaved 模式把每个系数复制到相邻两 lane。`GetBlockIdx()/2` 和 `GetBlockNum()/2` 把 56 个物理 block 映射为 28 个逻辑分区；相邻物理 block 可能对相同 GM 行执行幂等写。迁移时只能在“同值、无副作用、输出不重叠到其它行”的前提下保留这种成对映射，不能推广为一般并发写模式。

K1 每次处理 `tileRows=floor(192KiB/(3*D*elementSize))` 行，单 TBuf 被切为两路半长输入与两路全长输出，总计：

```text
K1_UB = 3*tileRows*D*elementSize <= 192 KiB
GM_intermediate = 2*S*D*elementSize <= 4 MiB
```

K2 与 R-001 使用相同 BS 分核和 sequence-wrap clipping，但从全长 `[S,D]` 系数表读取，调用 `InterleaveModeVF`。其 UB 为
`4*ubBS*ubN*dAlign*e + 4*ubBS*dAlign*e`。

```text
# R-002 决定性路径
expandRow[s,d] = compact[s,d/2]
for globalBS in block_partition(B*S), clipped before S wrap:
  coeffRow = globalBS % S
  for nk in tiled(concat(Q_heads,K_heads)):
    out[globalBS,nk,:] = InterleaveVF(in[globalBS,nk,:],
                                     expandedCos[coeffRow,:],
                                     expandedSin[coeffRow,:])
```

### R-003：Expanded BNSD

| 闭包项 | R-003 实现契约 |
|---|---|
| selector/Host producer | fast 公共谓词成立且 `layout=1,mode∈{1,2}`；rank-2 `[S,D/2]` 系数严格共享 B，展开 GM/56-block 条件成立；Host 生产 B×S FastPlan 并复核。 |
| POD/key/template | 无独立整数 tiling key；K1 标量同 R-002；K2 消费 `FastPlan.aba` 的 `blockNum/FactorB/S`、B/S/QN/KN loop/factor/tail、`rotaryMode/realDim`；模板固定 `ApplyRotaryPosEmbABAAndBA<T,true>`。 |
| launch/entry/body | `ExpandApplyRope*Kernel` 后，同 stream 启动 `ApplyRopeABA*Kernel` → `RunOfficialABA<T>` → `ProcessInSLoop/ProcessInSBLoop/ProcessInSBNLoop`。 |
| core/index | `bIdx=blockIdx%blockNumB,sIdx=blockIdx/blockNumB`；数据地址 `((b*N+n)*S+s)*D+d`；模板 `true` 令系数地址为 `s*D+d`，不加 b offset。 |
| GM/UB/L1/L0/workspace | K1 `[2,S,D]` 临时 GM；K2 qIn/qOut 与 cos/sin 四路 depth-2 Queue，UB 账由 `4*ubB*ubS*ubN*dAlign*e+4*ubS*dAlign*e` 约束；无 L1/L0/通信/caller workspace。 |
| 同步/tail/precision | K1 barrier 与同流 K1→K2 ordering；K2 LoopMode 每次 set 后 reset，Queue slot 在全部消费者完成后释放。B/S/N 尾由 loop length 裁剪，half/two-VL mask 处理 D 尾；FP32 分离乘加。 |
| fallback | planner/validator 或展开资源条件失败时在 launch 前退 R-004/R-005；K1 已提交后无 fallback。临时 owner 仍是外部边界。 |

K1 同 R-002；half 模式把前半系数复制到前后半，interleaved 模式复制到偶奇 lane。K2 grid 为二维逻辑映射：

```text
bIdx = blockIdx % blockNumB
sIdx = blockIdx / blockNumB
bRange = [bIdx*blockFactorB, clipped tail)
sRange = [sIdx*blockFactorS, clipped tail)
```

`ApplyRotaryPosEmbABAAndBA<T,true>` 的循环次序是 S tile → B tile → Q 的 N tile → K 的 N tile。模板参数 `true` 表示全长系数只按 S 取行并在 B 上广播；一个 S 系数 tile 必须等当前 B 范围的所有 q/k head 消费完才释放。Q/K 的 BNSD GM 地址为
`b*N*S*D+n*S*D+s*D`，LoopMode DMA 设置后必须在每次搬运结束复位。

```text
for sTile in blockS step ubFactorS:
  coeff = DMA(expanded[sTile,:])       # B broadcast
  for bTile in blockB step ubFactorB:
    for nTile in Q then K step ubFactorQN:
      x = LoopModeDMA(B,N,S,D tile)
      y = Batch{Half|Interleave}VF(x,coeff)
      LoopModeDMA(y -> dense output)
  free(coeff)                          # after every B/Q/K consumer
```

## P1 Dense row-vector Route

| 闭包项 | R-004 实现契约 |
|---|---|
| selector/Host producer | 仅在 `!useFast` 时调用 `MakeApplyRopeDensePlan`；要求 `D` 偶数、`D*e%32==0`、q/k/cos/sin 的 D stride 均为 1、UB/core 有效；`blockDim>0` 即选择。 |
| POD/key/template | 无独立整数 tiling key，`DensePlan.blockDim>0` 是 key 等价谓词；Host 按值生产 `ApplyRotaryPosEmbDensePlan`：shape/layout/mode/broadcast、`dAlign/coefficientAlign` 与 query/key 两份 stride POD；device 模板为 `ApplyRotaryPosEmbDenseGeneric<T>`。 |
| launch/entry/body | `LaunchApplyRopeDense{Float,Half,BFloat16}` 单次启动 `ApplyRopeDense*Kernel` → `Init/Process/ProcessRow/ProcessTensor`。entry AIV-only。 |
| core/index | `row=blockIdx+k*blockDim`；按 layout 解码 `(b,s,n)`；input 以各自 stride 取址，输出以 `row*D` dense 写；系数用 `Cb/Cs` 做 B/S 广播。 |
| GM/UB/L1/L0/workspace | 无 tensor-sized 中间 GM；depth-1 input/output/cos/sin Queue，interleaved 另有两个全长系数行 TBuf；无 L1/L0/通信/caller workspace。 |
| 同步/tail/precision | Queue 保护每行 copy/compute/copy；grid-stride 无 block 尾，D 与系数尾由 mask/对齐区处理，只写真实 D。half 用 compact VF，interleaved 行内展开；均为 FP32 分离乘加。 |
| fallback | DensePlan 任一条件失败即在 launch 前选择 R-005；dense Kernel 提交后不重试 generic。 |

R-004 覆盖 fast 不接受的广播形态、外层非连续 view、一般偶数 D 等情况，但要求每个 D 行连续：`xD=cosD=sinD=1`，并要求 `D*elementSize` 是 32B 的整数倍。q/k 可以拥有不同的 B/S/N stride；cos/sin 也各自保留独立 B/S stride。输出始终按 dense `row*D` 写。

```text
totalRows       = B*S*N
blockDim        = min(totalRows,AIV cores,56), at least 1
dAlign          = ceil_align(D,32/elementSize)
coefficientAlign= ceil_align(D/2,32/elementSize)
```

每个 block 以 `row=blockIdx; row<totalRows; row+=blockDim` 遍历。row 按 layout 解码为 `(b,s,n)`；系数索引分别用 `Cb==1?0:b` 与 `Cs==1?0:s` 做广播。一次把紧凑 cos/sin 行搬入 UB，随后依次处理同位置的 q 行和 k 行，因此系数可在两次计算间复用。

half 直接调用 `CompactHalfVF`；interleaved 先在一个可复用 TBuf 中把单行系数扩成全长，再调用 `InterleaveModeVF`。资源上界只依赖 D：

```text
Dense_UB = 2*dAlign*elementSize
         + 2*coefficientAlign*elementSize
         + (mode==interleaved ? 2*dAlign*elementSize : 0)
```

input/output/cos/sin queue 均为 depth 1；interleaved 的最后一项是 cos/sin 全长 TBuf。`D<=2048` 仍需以运行时 UB 查询和上述公式为准，不能仅凭 shape 上限假设可容纳。

```text
for row = blockIdx; row < B*S*N; row += blockDim:
  b,s,n = decode(row,layout)
  cRow = (Cb==1 ? 0:b)*cosBStride + (Cs==1 ? 0:s)*cosSStride
  coeff = DMA(compact[cRow,0:D/2])
  if mode==interleaved: coeff = expand_one_row_in_UB(coeff)
  process(query[strided(b,s,n,:)] -> queryOut[row,:], coeff)
  process(key[strided(b,s,n,:)]   -> keyOut[row,:],   coeff)
```

## P2 Generic stride-aware Route

| 闭包项 | R-005 实现契约 |
|---|---|
| selector/Host producer | 公共 Host 校验通过，且 fast 未选、`DensePlan.blockDim==0`；`MakeGenericStrides` 为 q/k 分别生产数据 stride，并复制独立 cos/sin stride。 |
| POD/key/template | 无独立整数 tiling key，前述 selector 全未命中即为 key；每次 launcher 按值接收 B/S/N/D/layout/mode/Cb/Cs/elements 与一个 `ApplyRotaryPosEmbGenericStrides`；模板实例为 `ApplyRopeGenericVF<T>`。 |
| launch/entry/body | `LaunchGenericPair` 依次为 q、k 调用 dtype launcher；各启动一个 `ApplyRopeGeneric*Kernel`，entry → `RunApplyRopeGeneric<T>` → `asc_vf_call<...,1024 threads>`。 |
| core/index | Host grid `min(56,ceil(E/1024))`；AIV block 先取连续 logical interval，SIMT thread 再 grid-stride。每元素解码 `(b,s,n,d)`，通过 signed-64 stride 计算 x/peer/cos/sin index。 |
| GM/UB/L1/L0/workspace | 直接 GM load/store，不建 TQue/TBuf、L1/L0、中间 GM、通信或 caller workspace；q 与 k 是两个同流节点。 |
| 同步/tail/precision | block interval 与 `localElements` 裁剪尾，线程只写一个有效 element；同流保证 q 节点先于 k 节点提交但二者无数据依赖。FP16/BF16 用 float 标量；FP32 finite product 额外补 `fmaf` residual。 |
| fallback | 完整合法域的最终受控 fallback；提交失败没有进一步 device Route。 |

R-005 是公共合法域的最终 fallback，覆盖 D 行非连续、行字节不足 32B、任意合法 B/S 广播以及其它 fast/dense 退化。它不分配 UB queue 或全长中间张量，q 与 k 各启动一次：

```text
blocks = min(56,ceil_div(elements,1024)), at least 1
per AIV block: contiguous logical element interval
inside block: asc_vf_call with 1024 SIMT threads and grid stride
```

每个逻辑 output index 先按 dense layout 解出 `(b,s,n,d)`，再通过 POD 中的 signed-64 element strides 计算 input、peer、cos、sin GM index；所以 transpose、narrow、expand 和非零 storage offset view 不需要 materialize。输出 index 本身始终是 dense 线性 index。Host 的 reachable-offset 前检和非负 stride 契约是此 Route 使用 64 位索引的必要前提。

```text
index -> (b,s,n,d)
j,peer,sign = decode_rotary_pair(d,mode,D)
xBase = b*xB + s*xS + n*xN
cIdx  = (Cb==1?0:b)*cosB + (Cs==1?0:s)*cosS + j*cosD
sIdx  = (Cb==1?0:b)*sinB + (Cs==1?0:s)*sinS + j*sinD
y[index] = x[xBase+d*xD]*cos[cIdx]
         + sign*x[xBase+peer*xD]*sin[sIdx]
```

### Dormant 实现分支

| 分支 | 不可达原因 | 重新启用条件 |
|---|---|---|
| quarter/partial RoPE helper | Host 只接受 mode 1/2，所有 plan 固定 `isPartialRope=false` | 扩展 schema/shape 语义，补 Host producer、系数 shape、POD validator 和对应精度检查；不能只放开 device `else` |
| `ApplyRotaryPosEmbAB<T>` 官方全长 AB 类 | 当前 router 不 include/实例化该类；layout 0 分别选择 compact 或 broadcast 类 | 新增明确 selector、entry 和与系数 B/S 布局一致的 GM 生产链 |
| `ExpandApplyRopeCoefficientsVF` SIMT helper | 有函数定义，但实际 Expand entry 调用的是 `RunExpandApplyRopeCoefficients<T>` 的 UB/VF body | 新增受控 entry/launcher，并重新证明 grid、索引、精度与 owner 生命周期 |

## POD、ABI 与 launch 图

### POD 字段组

| POD | 字段组 | producer → consumer | 单位与边界 |
|---|---|---|---|
| `ApplyRotaryPosEmbRegbaseABTilingData` | `B,CosB,S,D,QN,KN,dAlign,dSplitCoef`；BS block/factor/tail；UB N/BS factor/tail；mode/realDim/partial | `MakeApplyRopeFastPlan` → compact/broadcast AB | 全部 `int64_t`，末尾 bool；Direct 中 `CosB=1,isPartialRope=false` |
| `ApplyRotaryPosEmbRegbaseTilingData` | shape；B/S 分核；B/S/QN/KN 的 loop/factor/tail；mode/realDim/partial | fast planner → ABA body | element/block 计数均为 `int64_t`；body 的部分 DMA 字段会窄化，须由 validator 证明范围 |
| `ApplyRotaryPosEmbFastPlan` | `blockDim + ab + aba` | Host 按值传 launcher，launcher 只选择匹配 layout 的子 POD | LP64 推导 size 352；不得让 Host/device 字段顺序漂移 |
| `ApplyRotaryPosEmbGenericStrides` | `xB,xS,xN,xD,cosB,cosS,cosD,sinB,sinS,sinD` | Host 从 tensor stride 生成 → dense/generic indexer | 均为 signed-64 element stride，不是 byte stride；LP64 推导 size 80 |
| `ApplyRotaryPosEmbDensePlan` | grid/shape/layout/mode/broadcast/alignment + query/key 两份 stride POD | dense planner → 单 dense Kernel | LP64 推导 size 256 |

当前头文件没有 `static_assert(sizeof/offsetof)`；移植时应在共享 ABI 头中为上述推导尺寸和关键 offset 增加 Host/device 双侧静态断言。不要使用编译器相关的隐式 packing，也不要把 element stride 误乘两次 `sizeof(T)`。

### 节点与生命周期

```text
ApplyRopeNpu
  ├─ R-001: CompactAB_T<<<fast.blockDim,stream>>>(q,k,cosHalf,sinHalf,...,fast.ab)
  ├─ R-002: Expand_T<<<56,stream>>>(cosHalf,sinHalf,plane0,plane1,S*D,D,mode)
  │          -> BroadcastAB_T<<<fast.blockDim,same stream>>>(...,plane0,plane1,...,fast.ab)
  ├─ R-003: Expand_T<<<56,stream>>>(...)
  │          -> ABA_T<<<fast.blockDim,same stream>>>(...,plane0,plane1,...,fast.aba)
  ├─ R-004: Dense_T<<<dense.blockDim,stream>>>(q,k,cosHalf,sinHalf,...,dense)
  └─ R-005: Generic_T(q)<<<blocks,stream>>>(...,qStrides)
             -> Generic_T(k)<<<blocks,same stream>>>(...,kStrides)
```

K1 TBuf 用 `PIPE_ALL` barrier 隔开 GM→UB、VF、UB→GM；各 K2 用 `TQue` 的 Alloc/EnQue/DeQue/FreeTensor 建立 tile 生命周期。launcher 形参中的 workspace 不被 body 使用，调用者 workspace 为 0。R-001/R-004/R-005 没有局部 tensor-sized 中间 owner；R-002/R-003 另有下述 Host 异步边界。

#### R-002/R-003 异步 owner 边界：当前源码未闭合

`expandedCoefficients` 是 `ApplyRopeNpu` 的局部 `[2,S,D]` Tensor，两 plane 地址相差 `S*D*elementSize`。源码的最小所有权关系是：

```text
owner = empty([2,S,D])
expandedCos, expandedSin = raw_data_ptr(owner)
aclCall [=] uses expandedCos/expandedSin raw pointers
RunOpApi(aclCall)                 # callback submits K1 then K2
ApplyRopeNpu returns qOut,kOut    # owner itself is not returned
```

lambda body没有引用 `owner` Tensor，仅使用事先提取的 raw pointer；C++ 默认捕获不会因此把未被使用的 Tensor owner 保存在 closure 中。可由 Direct 源码证明的只有：K1 与 K2 向同一 stream 依次提交，所以在内存仍有效时 K2 不会早于 K1 读取。它**不能**证明 `RunOpApi` 返回、局部 Tensor 析构后 allocator 不会在 K2 完成前复用该 storage；`RunOpApi` 是否延迟持有 callback，以及框架 allocator 是否自动为这种未返回的局部 Tensor 记录 stream，均是当前 Direct 文件之外的框架契约。故本卡不把 R-002/R-003 的 owner lifetime 标记为源码已闭合。

安全集成至少选择并验证一种保活协议：

- 在 owner 析构前调用目标框架的 `record_stream(currentStream)` 或等价 caching-allocator stream-record API，使该 storage 直到 K2 所在 stream 到达记录点后才可复用；具体 C++ API 必须按所用 torch_npu/allocator 版本确认。
- 把 Tensor owner 显式放入由 stream event、future 或 completion callback 持有的 completion context，并只在 K2 completion event 后释放。若 `RunOpApi` 只是同步执行 callback 来“入队后立即返回”，仅把 owner 捕获到这个短生命周期 lambda 仍不够；必须证明 closure 本身被保留到 device 完成，或另建 completion owner。
- 若 K1/K2 改到不同 stream，除 owner 保活外还必须记录 K1 event、让 K2 stream wait；event ordering 与 allocator lifetime 是两个独立条件，缺一不可。

在上述框架语义或显式协议未闭合时，安全选择是禁用需要 `[2,S,D]` 临时 Tensor 的 R-002/R-003，退到无需该 owner 的 R-004/R-005；不能仅凭“同 stream”保留这两条 Route。

## 索引、尾块与搬运契约

| Route | 分核尾 | tile 尾 | D/lane 尾 | 写出边界 |
|---|---|---|---|---|
| R-001/R-002 | 最后 block 用 `blockTailBS` | `currBS=min(剩余,ubFactorBS,S-globalBS%S)`；N 最后一轮用 `ubTailN` | compact half 用 `halfD` mask；interleaved 按 VL 和剩余 lane mask | Q-only、K-only、跨 Q/K 三种区间分别 copy |
| R-003 | 最后 bIdx/sIdx 以真实 B/S 余数缩短 | B/S/N 三层分别取 `min(factor,remaining)` | Batch VF 以 half 或 two-VL mask | LoopMode 只搬实际 b/s/n length，之后 reset |
| R-004 | grid-stride row，无空 block | 每次严格一行；系数广播索引独立计算 | half/interleaved helper mask；系数 expansion 最后一段按 pairCount mask | DataCopy 仅 `D*elementSize`，输出 dense row |
| R-005 | 每 block 用 ceil 区间并裁剪 `localElements` | 线程以 1024 stride 覆盖区间 | 每线程只负责一个有效 d，不读 padding lane | 直接写一个有效 dense output element |

一般 DMA 契约：Host 先证明 blockCount、blockLen、stride 的窄化范围；device 端尾长必须从总量减去已处理量计算，不能假设整除；对广播系数，任何连续多行 DMA 都不得穿过 S 表尾。

## 精度路径

| Route/阶段 | 计算 dtype | 运算次序 | cast/特殊值 |
|---|---|---|---|
| K1 系数展开 | FP32 RegTensor 暂存，写回原 T | 仅 duplicate/interleave，不做 RoPE 乘加 | FP16/BF16 load 升 FP32、store 以 `NO_SAT + CAST_RINT` 回写；系数值只重排 |
| fast/dense Vector | FP32 RegTensor | 分别 `Mul(x,c)`、`Mul(peer,s)`，再 `Add/Sub`；不应误改为单 FMA | FP16/BF16 升 FP32，结果 RINT 降回；FP32 直接 load/store |
| generic FP16/BF16 | 标量 FP32 | 两个 float product 后 add/sub | 最后 `static_cast<T>` |
| generic FP32 | 标量 FP32 | 先物化两个 product；若二者 finite，再用 `fmaf(a,b,-product)` 恢复各乘积残差并按符号补入 | Inf/NaN 跳过残差修正，保留普通 IEEE 传播 |

Route 切换可能改变 FP32 最后几位，因为 generic 具有有限值残差修正，而 Vector 路径是分离乘加。实现或比较器必须按所选 Route 的明确数值序列建立契约，不能把所有 Route 笼统描述成“FP32 累加完全相同”。两种 mode 的系数索引、peer 索引和 sign 必须同步切换；只换旋转索引而不换系数展开会产生系统性语义错误。

## Fallback、失败与一般安全契约

- 无效 enum、rank、dtype、device、shape、broadcast、负 stride、算术溢出或资源总量在 Host 前检阶段同步拒绝；不得让非法 metadata 到达 Route planner。
- fast plan 的平台、UB、grid、tail 或 DMA 计数复核失败是正常的 Route 退化条件，继续 dense/generic；dense 的行连续/32B/UB 条件失败继续 generic。
- generic 只承诺非负 stride；需要支持负 stride 的上层必须先 materialize 或设计独立的有符号基址协议，不能直接移除 Host guard。
- C launcher 返回 `void`，`RunOpApi` 回调固定返回 0；现有闭包没有 Kernel 提交后的状态回传或运行时 Route 重试。需要强失败传播的集成应在 launcher 层增加平台错误检查，而不是假装 fallback 已发生。
- 所有输出是新分配 dense tensor，不允许把输入输出别名当作既有能力。若改为原地，必须重新证明 peer 读取不会被更早 lane 写覆盖。
- fast 展开临时 Tensor 的 owner 保活未由当前 Direct 文件证明，必须按“异步 owner 边界”验证 `record_stream`/completion owner，否则在 launch 前禁用 R-002/R-003；Queue slot 和 TBuf 的 device 内生命周期则由 barrier/Queue 闭合。
- 所有 entry 是 AIV-only；AIC 早退不是计算 fallback。grid 必须与平台 AIV 数及成对 block 约定一致。
- `ApplyRotaryPosEmbFastPlan`、`DensePlan` 与 stride POD 均按值过 ABI；字段、位宽、对齐或 bool 尾填充变化必须同时更新 Host/device 并以静态断言闭合。

## 实现闭包索引

| 层级 | 文件/符号 | 支撑事实 |
|---|---|---|
| Schema/Meta/NPU | `apply_rotary_pos_emb_plugin.cpp`：`ValidateApplyRope`、`ApplyRopeMeta`、`ApplyRopeNpu` | checked arithmetic、输出、Route 顺序、stream/GM 地址 |
| ABI | `apply_rotary_pos_emb_tiling.h`、`apply_rotary_pos_emb_launch.h` | Fast/Dense/Generic POD 与 C launcher |
| Planner/router | `apply_rotary_pos_emb_kernel.cpp`：`MakeApplyRopeFastPlan`、`IsApplyRopeFastPlanValid`、`MakeApplyRopeDensePlan`、launch macros | 五 Route 分发、资源验证、grid、K1→K2 图 |
| Compact fast | `apply_rotary_pos_emb_compact_ab.h`、`apply_rotary_pos_emb_compact_vf.h` | R-001 半长系数直算、S-wrap clipping、D64 float 特化 |
| Expanded BSND | `apply_rotary_pos_emb_broadcast_ab.h` | R-002 全长系数 BSND body |
| Expanded BNSD | `apply_rotary_pos_emb_official_aba.h` | R-003 B/S/N LoopMode 与 B 广播 |
| Dense | `apply_rotary_pos_emb_dense_generic.h` | R-004 单行 UB、stride-aware outer axes、q/k 复用 |
| Generic | `apply_rotary_pos_emb_kernel.cpp`：`ApplyRopeGenericVF` | R-005 逐元素 signed-64 索引与 FP32 residual 路径 |
| 公共数值/搬运 | `apply_rotary_pos_emb_official_common.h`、`op_kernel/load_store_utils.h` | RegTensor mask、mode VF、FP16/BF16 cast、DataCopy helper |
| 构建 | 算子 `CMakeLists.txt` 的 `register_direct_launch_op` | kernel/plugin/launch header 接线 |

官方固定版本链接用于确认相关逻辑算子与 950 注册范围；Direct schema、五级 Route、资源 guard 和 stride-aware fallback 由本卡所述 Direct 实现闭包定义，不能把两者误当作逐行同一接口。
