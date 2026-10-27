---
type: Operator
title: "MoeReRouting 950 Direct Invoke 算子知识"
description: "说明 MoeReRouting 在 950 上从 Torch Host、Direct selector、普通 POD 与 current-stream launcher 到 RE、fast-RE、R 三条 AIV Route 的完整参数、分核、GM/UB、同步、尾块、精度和 fallback 关系。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/README.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/moe_re_routing_def.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/moe_re_routing_tiling.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/moe_re_routing_tiling_base.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/moe_re_routing_re_tiling.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/moe_re_routing_re_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/moe_re_routing_r_tiling.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/moe_re_routing_r_tiling.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_kernel/moe_re_routing_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_kernel/arch35/moe_re_routing_re_regbase.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_kernel/arch35/moe_re_routing_r_regbase.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/moe/moe_re_routing/op_host/config/ascend950/moe_re_routing_binary.json"
    role: implementation
created_at: "2026-08-25T19:38:59Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MoeReRouting 950 Direct Invoke 算子知识

## 范围与来源边界

本卡描述 950/Ascend 950 上的 Direct Invoke 子集：

```text
cann_bench::moe_re_routing_npu
  -> calc_moe_re_routing_tiling
  -> launch_moe_re_routing(current NPU stream)
  -> one of 36 explicit AIV global entries
  -> RE | fast-RE | R
```

固定官方源码直接支撑算子语义、950 OpDef、R/RE selector、POD 字段、registered tiling、950 router 和 R/RE Device body。Direct 转换关系必须分开理解：

| 层次 | 与固定官方源码的关系 | 不应扩大成的结论 |
|---|---|---|
| 官方原生 | `MoeReRoutingReTilingData`、`MoeReRoutingRTilingData` 字段顺序；R/RE 分核公式；R/RE body；950 dtype 配置 | 不等于 Direct ABI、Torch 注册或 current-stream launcher。 |
| Direct 适配 | Torch schema/meta/NPU wrapper、普通 C++ POD、`calc_*`、36 个显式 entry、`launch_*`、`register_direct_launch_op` | 固定官方仓没有这些同名文件或符号，不能称为官方原生实现。 |
| Device 派生补充 | `MoeReRoutingFastReRegbase`、`useReRoute=2`、无 scale 输出清零 | fast-RE 的矩形 `Process` 和清零逻辑不是官方 R/RE 文件的逐字内容。 |

Direct 中的官方 R/RE 头文件仅替换三个 include 为本地依赖闭包；类体其余内容保持一致。fast-RE 以官方 R 为基线：`Init`、UB 分支、copy/index/expert-sum helper 一致，只把 POD 类型换成 RE POD并改写 `Process` 为 rank×expert 矩形遍历。

## 语义、接口与布局契约

### 重排映射

令 `C[r,e] = expert_token_num_per_rank[r,e]`，`r in [0,N)`，`e in [0,E)`。输入按 rank-major、rank 内 expert-major 分段；输出按 expert-major、expert 内 rank-major 分段：

```text
S(r,e) = sum_{rr<r, ee in [0,E)} C[rr,ee] + sum_{ee<e} C[r,ee]
D(r,e) = sum_{ee<e, rr in [0,N)} C[rr,ee] + sum_{rr<r} C[rr,e]
c = C[r,e]

permute_tokens[D(r,e):D(r,e)+c, :]
    = tokens[S(r,e):S(r,e)+c, :]
expert_token_num[e] = sum_r C[r,e]
```

有 scale 时，对 `[S,S+c)` 到 `[D,D+c)` 做相同置换。Direct 固定 `idx_type=0`，因此：

```text
permute_token_idx[D(r,e)+k] = S(r,e)+k, 0 <= k < c
```

这是一张“输出位置取哪个输入 token”的 gather 表。官方 950 body 也保留 `idx_type=1` 的 scatter 写法，但 Direct Host拒绝该属性。

### Direct ABI 子集

| 参数 | Direct shape / dtype | Host 处理 | Device 解释 |
|---|---|---|---|
| `tokens` | `[A,H]`; fp16/bf16/int8 | 要求 2D 且 `A,H>0`；不做 contiguous 转换 | dense ND；每 token 为 `H` 个 `T` 元素。 |
| `expert_token_num_per_rank` | `[N,E]`; int32/int64 | 要求 2D 且 `N,E>0`；不读取元素值 | dense row-major `C[r*E+e]`。 |
| `per_token_scales` | optional `[A]`; fp32 | 有值时检查 shape/dtype/device；不做 contiguous 转换 | `scaleSize=1`，随 token 同区间搬运。 |
| `expert_token_num_type` | int64，默认 1 | 仅接受 1 | 输出为每 expert count，不是 cumsum。 |
| `idx_type` | int64，默认 0 | 仅接受 0 | gather index。 |
| `permute_tokens` | `[A,H]`; 同 `tokens` | `empty_like(tokens)` | 写 `D*H` 起始的连续 token 段。 |
| `permute_per_token_scales` | `[A]`; fp32 | 总是分配 | 有 scale 时重排；无 scale 时由同一 Kernel 分片写 0。 |
| `permute_token_idx` | `[A]`; int32 | 分配连续一维输出 | `CreateVecIndex` 生成。 |
| `expert_token_num` | `[E]`; 同 counts | 分配连续一维输出 | 每列以 `TIndex` 累加。 |

官方 950 注册还包含 FP8/HIFLOAT8/FP4、fp32或FLOAT8_E8M0 scale、多维 scale 以及 `idx_type=1`。这些均不在本 Direct wrapper 的 dtype/shape/属性矩阵中；不能只因官方 binary config 存在就从 Direct 入口调用。

### 值域前提

Host 只检查元数据，调用方仍须同时保证：

- `C[r,e] >= 0`，且 `sum(C)=A`；否则 S/D 不再覆盖 `[0,A)` 的互斥分段，可能重叠或越过 GM。
- 官方文档约束 `0 < H < 16384`；Direct Host只检查 `H>0`，没有实现上界检查。
- `tokens`、counts、scale 都必须是 Device 按 dense contiguous 解释的内存。官方 OpDef有 `AutoContiguous`，Direct wrapper 不调用 `contiguous()`；`empty_like` 的输出内存格式也不能被默认为任意 stride 可用。
- 每个 gather index `S+k` 必须落入 int32；精确条件是 `A-1 <= INT32_MAX`。每列 count 和输入 count 还必须能由 `TIndex` 表示。
- 输入必须同一 device；PrivateUse1 分派提供 NPU 入口，wrapper 显式比较 counts/scale 与 tokens 的 device。

## Host、selector 与 launcher 闭环

### Host 调用序列

```text
moe_re_routing_npu(tokens, counts, scales?, expert_type, idx_type):
  OptionalDeviceGuard(tokens.device)
  require counts.device == tokens.device
  if scales: require scales.device == tokens.device
  outputs = moe_re_routing_meta(...)
  tokenType = fp16 ? 0 : bf16 ? 1 : 2
  indexType = counts.i32 ? 0 : 1
  params = zero-initialized MoeReRoutingLaunchParams
  calc_tiling(A,H,N,E,sizeof(TIndex),hasScale,idx_type,&params)
  stream = getCurrentNPUStream().stream(false)
  RunOpApi("MoeReRoutingDirect", lambda:
    launch(inputs, outputs, A, tokenType, indexType, hasScale, &params, stream)
    return 0)
  return outputs
```

`launchParams` 在 Host 栈上生存到 `RunOpApi` 的 lambda 执行完成；launcher立即读取它，global entry再按值接收选中的 R 或 RE POD。没有 Host tensor-value 计算，也没有把 counts 搬回 Host。

### selector 的完备分区

记 `P=GetCoreNumAiv()`。实际判断顺序是：

```cpp
if (N < 5 && E != 1)       useReRoute = 1;  // R-001 RE
else if (2 * N <= P && E != 1)
                              useReRoute = 2;  // R-002 fast-RE
else                         useReRoute = 0;  // R-003 R
```

在 `N>0,E>0,P>0` 下可化为互斥且完备的集合：

| selector 顺序 | Route | 精确域 | 下一跳 |
|---:|---|---|---|
| 1 | R-001 official RE | `E>1 && N<5` | 命中即停止；否则检查 R-002。 |
| 2 | R-002 derived fast-RE | `E>1 && N>=5 && 2N<=P` | 命中即停止；否则 R-003。 |
| 3 | R-003 official R | `E==1 \|\| (E>1 && N>=5 && 2N>P)` | 最终 fallback，无下一 Route。 |

Route 的知识复用优先级与 selector 顺序分开：

| 复用优先级 | Route | 排序理由 |
|---:|---|---|
| P0 | R-002 fast-RE | Direct 独有：保留 R helper，却以 RE POD暴露二维 cell 并行；迁移时最容易误归为官方 RE 或官方 R。 |
| P1 | R-003 R | 覆盖 `E==1` 与无法暴露足够二维 block 的兜底域；边界最宽。 |
| P2 | R-001 RE | 官方小 rank 专用二维 Route；selector 条件最窄且先短路。 |

这里的 P0/P1/P2 是实现复用与审查顺序，不表示 launcher 顺序，也不表达任何时延结论。

### 实例分派与 fallback

每条 Route 有 `3 token dtype × 2 count dtype × 2 scale state = 12` 个显式 global entry，共 36 个。实例维度不是额外算法 Route：

```text
T     = half | bfloat16_t | int8_t
I     = int32_t | int64_t
S     = false | true
Scale = float
```

`LAUNCH_ROUTE` 按 `(tokenType,indexType,hasScale)` 选择实例。最后一个裸 `else` 会启动 `int8_i64_scale`，所以 launcher自身不是 fail-closed；安全性依赖 wrapper先把枚举限制到 `tokenType in {0,1,2}`、`indexType in {0,1}`、`hasScale in {0,1}`。`useReRoute` 同样只有 `0/1/2` producer；launcher把任何非零且非1的值都当 fast-RE。

### 单 Kernel 启动图

```text
launch_moe_re_routing
  -> choose RE POD or R POD
  -> entry<<<blockDim, nullptr, current_stream>>>(..., pod_by_value, A)
     -> if g_coreType == AIC: return
     -> TPipe pipe
     -> construct RouteClass<T,I,float,S>(&pipe,&pod)
     -> if !S: ZeroUnusedScaleOutput(outputScale,A,pod.coreNum)
     -> RouteClass::Init(seven GM pointers)
     -> RouteClass::Process()
```

所有可达 entry 都是 `AIV_VECTOR` 工作；AIC实例由入口 guard 返回。没有 Cube body、MIX_CV 协作、第二个 Kernel、event 链或跨 Kernel依赖。

## POD、tiling 与 ABI

### 普通 POD 布局

三个 struct 的字段全为 `int64_t`，转换实现依赖标准 8 字节布局：

| struct | 大小/对齐 | 字段 offset（byte） |
|---|---|---|
| `MoeReRoutingReTilingData` | 96 / 8 | `tokenSize`0, `scaleSize`8, `rankNum`16, `expertNum`24, `coreNum`32, `blockNumR`40, `blockFactorR`48, `blockNumE`56, `blockFactorE`64, `ubFactor`72, `idxType`80, `tokenSizeOrigin`88 |
| `MoeReRoutingRTilingData` | 72 / 8 | `tokenSize`0, `scaleSize`8, `rankNum`16, `expertNum`24, `coreNum`32, `blockFactor`40, `ubFactor`48, `idxType`56, `tokenSizeOrigin`64 |
| `MoeReRoutingLaunchParams` | 184 / 8 | `blockDim`0, `useReRoute`8, `re`16, `r`112 |

复用时应保留相当于以下 ABI 防漂移断言：

```cpp
static_assert(sizeof(MoeReRoutingReTilingData) == 96);
static_assert(sizeof(MoeReRoutingRTilingData) == 72);
static_assert(sizeof(MoeReRoutingLaunchParams) == 184);
static_assert(offsetof(MoeReRoutingLaunchParams, re) == 16);
static_assert(offsetof(MoeReRoutingLaunchParams, r) == 112);
```

官方 tiling macro的 R/RE 字段名、类型和顺序与前两个普通 POD一致；`MoeReRoutingLaunchParams` 与 `useReRoute=2` 是 Direct 新增 ABI。

### 字段 producer/consumer

| 字段 | Direct producer | Device consumer | 直接边界 |
|---|---|---|---|
| `blockDim` | 选中 Route 的 block 数 | launch 配置 | 必须 `>0`。 |
| `useReRoute` | selector 写 0/1/2 | launcher | 2 只属于派生 fast-RE。 |
| `tokenSize` | `H` | GM token地址、copy长度、UB分支 | Direct `T` 无压缩。 |
| `scaleSize` | `hasScale ? 1 : 0` | scale地址、copy长度 | `S=false` 编译期不绑定 scale GM。 |
| `rankNum`,`expertNum` | `N,E` | 分核、offset扫描、expert sum | 均为正。 |
| `coreNum` | 实际 `blockDim` | rank/expert轮转、无-scale分片 | 不是物理 `P`；必须与 launch block 数一致。 |
| RE `blockNumR/E` | 二维分核公式 | `divmod(blockIdx,blockNumR)` | 乘积等于 `coreNum`。 |
| RE `blockFactorR/E` | 二维分核公式 | cell矩形和末块裁剪 | 最后块可小于 factor。 |
| R `blockFactor` | rank分核公式 | Direct R body不读取 | ABI保留字段，不能据此假设连续rank块。 |
| `ubFactor` | UB字节公式 | queue容量和 full/overload判断 | R/fast 的 overload helper又把它用作元素 factor，迁移时必须重审单位。 |
| `idxType` | Direct 固定0 | index head/offset方向 | body虽有1分支，Direct不可达。 |
| `tokenSizeOrigin` | Direct 固定0 | int8模板的FP4压缩地址分支 | Direct int8不走2:1压缩。 |

### 分核公式

令 `ceil(a,b)=ceil_div(a,b)`，`align32(x)=ceil_align(x,32)`，`Ibytes=sizeof(TIndex)`，`U=UB bytes`：

```text
indexReserve = 256 * 2 * Ibytes

R-001 / R-002:
  tmpR         = ceil(N,P)
  blockNumR    = ceil(N,tmpR)
  blockFactorR = ceil(N,blockNumR)
  rawBlockNumE = min(floor(P/blockNumR),E)
  blockFactorE = ceil(E,rawBlockNumE)
  blockNumE    = ceil(E,blockFactorE)
  blockDim     = blockNumR * blockNumE
  usable       = (U-indexReserve)/2 - (R-002 ? 32 : 0)
  ubFactor     = align32(usable)

R-003:
  tmp          = ceil(N,P)
  blockNum     = ceil(N,tmp)
  blockFactor  = ceil(N,blockNum)
  blockDim     = blockNum
  usable       = (U-indexReserve)/2 - 32
  ubFactor     = align32(usable)
```

Direct Meta拒绝 `A==0`，所以源码中 `tokensNum==0 -> blockNum(R)=1` 的保护在本入口不可达。公式还隐含 `P>0`、`U>indexReserve`，R/fast还要求减32后仍为正；Direct tiler不返回错误码，也不检查这些平台前提。

`align32` 是向上对齐。R/fast先减32，为最多31字节的向上舍入留 gap；RE没有该 gap，依赖实际平台上 `(U-indexReserve)/2` 的对齐几何。不要把这组公式无条件移植到任意 UB 大小。

### workspace 差异

| 链路 | Host 声明 | Kernel 使用 |
|---|---|---|
| 官方 registered R/RE | `MoeReRoutingTilingBase::RESERVE_WORKSPACE_BYTE = 16 MiB`，`PostTiling` 写 `workspaces[0]` | 官方 global entry带 `workspace` 参数，但 R/RE body不读取它。 |
| Direct 转换 | `launch_moe_re_routing` 没有 workspace 参数，plugin也不申请或传入 workspace | 36 个 entry均无 workspace形参；GM↔UB直接 relay。 |

因此“Device body不使用 workspace”与“官方 registered Host预留 workspace”可以同时成立；不能把 Direct 的零 workspace ABI反推为官方 Host也没有预留。

## 公共 Device 数据路径

### GM ownership 与路由写域

| GM | 读者/写者 | 地址公式 | 并行安全条件 |
|---|---|---|---|
| input tokens | Route body读 | Direct为 `(S+k)*H` 个 `T` 元素 | 输入只读。 |
| input counts | 所有核标量读 | `r*E+e` | 输入只读；offset扫描可能重复。 |
| input scales | `S=true` body读 | `S+k` 个 fp32 | 输入只读。 |
| output tokens | cell owner写 | `(D+k)*H` 个 `T` 元素 | 合法 counts使所有 `(r,e)` 的 D区间互斥。 |
| output scales | `S=true` cell owner写；`S=false`各核清零 | `D+k`；或按 A均分 | 两种模板编译期互斥。 |
| output index | cell owner写 | gather为 `D+k` | 与 output token使用同一 D区间。 |
| output expert count | `e = blockIdx + q*coreNum` 的核写 | `e` | 每个 expert恰有一个写者。 |

没有 atomic。合法 counts提供 token/scale/index 的不重叠写域；`CalcExpertTokenNum` 的步长分派提供 expert输出的单写者关系。

### UB 队列与字节账

每条 Route都只初始化两个 queue：

```text
queBind_   : TQueBind<VECIN,VECOUT,1>, depth=2
idxOutQue_ : TQue<VECOUT,1>,           depth=2
```

设：

```text
q(T) = ceil_align(ubFactor, 32/sizeof(T))
qi(I)= ceil_align(256*sizeof(I), 32/sizeof(I))
allocated queue bytes = 2*q(T) + 2*qi(I)
```

`idxOutQue_` 同时承载 int32 index和 `TIndex` expert count。`I=int64` 时按 2048 byte/slot预留，但每个256项 index块实际只写1024 byte；这不是另一个 buffer。没有 `TBuf`、dynamic UB、workspace scratch、NDDMA buffer或通信 window。

`depth=2` 说明资源上有两个 slot；实际 helper在每个 tile内立即执行 `Alloc -> copy/produce -> EnQue -> DeQue -> copy -> Free`。源码没有独立的 ping/pong循环，因而只能据此主张 queue生命周期闭合，不能仅凭 depth=2声称已经发生跨 tile交叠。

### 同步

| 场景 | primitive | 保证范围 | 不保证 |
|---|---|---|---|
| GM→UB→GM relay | `EnQue/DeQue/FreeTensor` | 同一核、同一 queue slot的生产消费和复用顺序 | 跨核同步。 |
| index/expert sum | `EnQue/DeQue/FreeTensor` | index queue单生产者/单消费者生命周期 | 全局顺序。 |
| 无-scale清零 | `InitOutput` 后 `PipeBarrier<PIPE_ALL>` | 发出清零的当前 AIV 核内，清零先于该核后续 body | 不是 block间 barrier；`start>=A` 的核不进入该分支。 |
| Kernel整体 | current stream上的单次 launch | 同一 stream内遵循框架 enqueue顺序 | Host不读取 Device完成码。 |

### 精度与类型

- token和scale只经 `DataCopyPad` 转发，没有 Cast、量化、反量化或浮点算术；有效元素保持位模式。
- S/D offset保存在 `int64_t` 标量中，读取的 `TIndex` counts参与加法。负 counts、总和不一致或 int64 offset溢出都没有 Device保护。
- `expert_token_num[e]` 在 `TIndex` 中逐项相加，不升精度；int32列和可能溢出。
- index总是 int32；`CreateVecIndex` 的 head显式窄化为 `int32_t`。
- `tokenSizeOrigin>0` 时 int8模板可按FP4 2:1物理地址换算，但 Direct producer固定写0，所以本子集没有压缩地址语义。

### 公共尾块

| 尾块 | 裁剪方式 |
|---|---|
| rank/expert分核 | RE/fast取 `min(start+factor,N/E)`；官方 RE等价地在最后 block以取模缩小 `rNum/eNum`。 |
| R rank轮转 | 每核 rank数为 `N/coreNum`，前 `N%coreNum` 个核多处理一个；实际 rank为 `loop*coreNum+blockIdx`。 |
| UB full-load | 末批 `currFactor = total % factor`，整除时保持完整 factor。 |
| UB overload | 末子段 `sub = logical_size % ubFactor`，整除时保持完整 ubFactor。 |
| index | 每块最多256项，末块 `rows%256`。 |
| expert sum | `e += coreNum`，自然覆盖非整除尾部。 |
| 无-scale清零 | `perCore=ceil(A/coreNum)`；末核取 `min(perCore,A-start)`，超出 A 的核不写。 |

## Route R-002（P0）：派生 fast-RE

### 代码关系与选择

| 环节 | 关系 |
|---|---|
| selector | 第一分支未命中后，`2*N<=P && E!=1` 写 `useReRoute=2`。 |
| tiling/POD | 复用 RE 的二维分核公式与 `MoeReRoutingReTilingData`；UB可用量额外减32。 |
| entry | `moe_re_routing_fast_re_<T>_<I>_<scale-state>` 共12个。 |
| body | `MoeReRoutingFastReRegbase<T,I,float,S>`；除 `Process` 与 POD类型外，helper与官方 R机械一致。 |
| fallback | guard失败进入 R-003；launcher没有从已选fast-RE回退到别的 Kernel。 |

### 核任务与路由伪码

```text
rb = blockIdx % blockNumR
eb = blockIdx / blockNumR
rStart = rb*blockFactorR; rEnd = min(rStart+blockFactorR,N)
eStart = eb*blockFactorE; eEnd = min(eStart+blockFactorE,E)

for r in [rStart,rEnd):
  src = sum_{rr<r, all e} C[rr,e]
  dst = 0
  for e in [0,eStart):
    src += C[r,e]
    dst += sum_{rr=0..N-1} C[rr,e]
  for e in [eStart,eEnd):
    c = C[r,e]
    dst += sum_{rr<r} C[rr,e]
    if c != 0:
      R-style ProcessTokenScale(c) at src,dst
    src += c
    dst += sum_{rr>=r} C[rr,e]

for e=blockIdx; e<E; e+=coreNum:
  expertOut[e] = sum_r C[r,e]
```

关键关系是 `dst` 在每个 expert结束时增长到“截至该 expert的完整列前缀”，`src` 则只在当前 rank内递增；它避免官方 RE 对每个非空 cell 重算完整 `CalDstOffSet`，src 前缀仍按 rank 重建。

### payload 子路

fast-RE完全复用 R 的 payload分支：

```text
tokSclBytes = align32(H*sizeof(T)) + (S ? align32(1*sizeof(float)) : 0)
if ubFactor < tokSclBytes:
  CopyOutIndex(c,src,dst)
  for each token:
    split token and scale through queBind_
else:
  factor = floor(ubFactor/tokSclBytes)
  for each factor-sized token batch:
    CopyIn(token batch + optional scale batch in one slot)
    CopyOutIndex(batch,src,dst)
    CopyOut(token batch + optional scale batch)
```

full-load时 scale子区起点是 `align32(batch*H*sizeof(T))`，不是 `batch*align32(H*sizeof(T))`；`tokSclBytes` 的逐 token估算因此是保守容量因子。overload helper把 `ubFactor` 直接作为 token/scale元素分段因子，而 Host把它从 UB字节数生成；这是原 R helper继承的单位耦合，扩展 Direct shape或平台前必须重新证明，不可机械泛化。

## Route R-003（P1）：官方 rank-only R

### 代码关系与选择

| 环节 | 关系 |
|---|---|
| 官方 selector | `MoeReRoutingRTiling::IsCapable` 接受 RegBase；注册优先级11000，在RE未命中后兜底。 |
| Direct selector | `useReRoute=0` 的最终 `else`。 |
| tiling/POD | R公式与 `MoeReRoutingRTilingData`；Direct保留 `blockFactor` 字段，但 body实际按轮转rank读取。 |
| entry/body | 12个 `moe_re_routing_r_*`；`MoeReRoutingRRegbase<T,I,float,S>`。 |
| 官方 key关系 | 无scale 210000、fp32 scale 210100；Direct不携带 tiling key，改为显式entry。 |

### 核任务与路由伪码

```text
rankLoops = floor(N/coreNum) + (blockIdx < N%coreNum ? 1 : 0)
for l in [0,rankLoops):
  r = l*coreNum + blockIdx
  src = sum_{rr<r, all e} C[rr,e]
  dst = 0
  for e in [0,E):
    c = C[r,e]
    dst += sum_{rr<r} C[rr,e]
    ProcessTokenScale(c) at src,dst
    src += c
    dst += sum_{rr>=r} C[rr,e]

for e=blockIdx; e<E; e+=coreNum:
  expertOut[e] = sum_r C[r,e]
```

每个核负责 `r=blockIdx, blockIdx+coreNum, ...`，并非 `[blockIdx*blockFactor,(blockIdx+1)*blockFactor)`。因此不能用 POD里的 `blockFactor` 推导 Device rank ownership。

### GM/UB与尾块

R使用合并 token+scale的 bound slot，payload full/overload伪码与 R-002完全相同。zero count没有显式 `continue`，但 `CopyOutIndex(0)`、UB循环和DMA循环都执行0次。每个 rank的 `src/dst` 从零重建，所以核间不共享 offset状态，也不需要跨核同步。

R是 Host最终 fallback，不含运行时二级兜底：UB过小、非法counts或枚举越界不会自动换到 RE。

## Route R-001（P2）：官方 rank×expert RE

### 代码关系与选择

| 环节 | 关系 |
|---|---|
| 官方/Direct selector | RegBase且 `N<5 && E!=1`；Direct写 `useReRoute=1`。 |
| tiling/POD | 官方RE二维公式与 `MoeReRoutingReTilingData`；RE UB公式不减32。 |
| entry/body | 12个 `moe_re_routing_re_*`；`MoeReRoutingReRegbase<T,I,float,S>`。 |
| 官方 key关系 | 无scale 200000、fp32 scale 200100；Direct改为显式entry。 |
| fallback | selector未命中才检查fast-RE，再到R；已启动RE后没有运行时fallback。 |

### 核任务与路由伪码

```text
rb = blockIdx % blockNumR
eb = blockIdx / blockNumR
rStart = rb*blockFactorR
eStart = eb*blockFactorE
rCount = (last-r-block && N%blockFactorR!=0) ? N%blockFactorR : blockFactorR
eCount = (last-e-block && E%blockFactorE!=0) ? E%blockFactorE : blockFactorE

for r in [rStart,rStart+rCount):
  src = S(r,eStart)                         // CalSrcOffSet once per rank
  for e in [eStart,eStart+eCount):
    c = C[r,e]
    if c == 0: continue
    dst = D(r,e)                            // CalDstOffSet per nonempty cell
    ProcessTokenScale(c,H,isScale=false)
    if S: ProcessTokenScale(c,1,isScale=true)
    src += c

for e=blockIdx; e<E; e+=coreNum:
  expertOut[e] = sum_r C[r,e]
```

这里 token与scale分别经过 bound queue，不像 R/fast在 full-load时把两者拼进一个 slot。

### payload 子路

RE先把 `ubFactor` 从字节换算到对应 payload的元素容量：

```text
capacity(X) = ceil_align(ubFactor/sizeof(X), 32/sizeof(X))

relay(c, width, X, isScale):
  u = capacity(X)
  if u < width:
    if token: CopyOutIndex(c,src,dst) once
    for each token:
      for each width-segment of at most u elements:
        CopyIn(segment); CopyOut(segment)
  else:
    factor = floor(u/width)
    for each factor-sized token batch:
      CopyIn(batch)
      if token: CopyOutIndex(batch,src,dst)
      CopyOut(batch)
```

`width=H,X=T` 用于 token；`width=1,X=float` 用于Direct scale。末 token batch、单 token内末分段以及256项 index块都独立裁尾。

## 无-scale输出、构建与生命周期

### 无-scale占位输出

官方 R/RE body在 `hasScales=false` 时不绑定、不写 `permute_per_token_scales`。Direct schema却始终返回 `[A]` fp32，因此每个 no-scale entry在 `InitAndProcess` 前执行：

```text
perCore = ceil(A,coreNum)
start = blockIdx*perCore
if start < A:
  count = min(perCore,A-start)
  InitOutput<float>(outScale[start],count,0.0f)
  PipeBarrier<PIPE_ALL>()
```

各核区间互斥。barrier只约束当前核的流水；这里不需要跨核barrier，因为后续 `S=false` body不会再读写该输出。

### 构建接线

`CMakeLists.txt` 只把 `moe_re_routing_kernel.cpp` 与 `moe_re_routing_plugin.cpp` 交给 `register_direct_launch_op`，include根为 `op_kernel`。本地依赖头提供 `platform::GetUbBlockSize()==32` 和 `CeilDiv/FloorDiv/CeilAlign`，使官方 R/RE body脱离原 registered-op依赖。构建接线不会自动补充非法 shape、counts值域或枚举检查。

### 所有权与生命周期

- 四个输出由 Torch allocator创建；输入输出 GM所有权始终属于框架。
- `MoeReRoutingLaunchParams` 属于 Host调用栈，选中 POD按值进入 Kernel参数区。
- `TPipe`、两个 queue及其 LocalTensor只在一个 AIV block生命周期内存在；每个 slot在 `FreeTensor` 后复用。
- 当前 stream由 wrapper取得并原样传给 launcher；没有私建 stream、event或通信域。
- Direct ABI没有显式 workspace；临时 Device存储仅为每核静态 queue。

## 失败关闭与迁移检查表

| 边界 | 当前行为 | 复用要求 |
|---|---|---|
| `A,H,N,E` 非正 | Meta拒绝 | 不绕过Meta直接调用tiler。 |
| `sum(C)!=A` 或负 counts | Host不读值；Device可产生重叠/OOB | 上游证明counts契约，或在独立前处理阶段检查。 |
| 非contiguous输入 | Host不整理stride | 入口前转dense contiguous，且确认 `empty_like` 输出布局。 |
| `H>=16384` | Direct未实现官方文档上界 | 上游限制，不把当前UB分支当作无限H支持。 |
| index超int32 | `CreateVecIndex` head窄化 | 保证 `A-1<=INT32_MAX`。 |
| int32 expert列和溢出 | `TIndex` 原类型累加 | 保证每列和可表示，或设计新输出dtype。 |
| `P<=0` / UB不足 | tiler可能除零或生成无效factor | 平台查询必须成功；新平台重做UB不等式。 |
| `2*N` 溢出 | selector使用有符号int64乘法 | 维度上限应使乘法安全，或改写为无溢出比较。 |
| 非法 token/index/scale枚举 | launcher裸 `else` 落到int8/i64/scale | 只能经已校验wrapper进入，或把launcher改成fail-closed。 |
| 非法 `useReRoute` | 非0且非1落到fast-RE | selector与launcher必须共享受控枚举。 |
| tiling/launch状态 | `calc_*` 与 `launch_*` 都返回 `void`，`RunOpApi` lambda无条件返回0 | `TORCH_CHECK`只覆盖launch前元数据；异步运行错误由框架/stream后续机制呈现，不能从lambda返回值推断成功。 |
| R/fast overload单位 | byte来源的`ubFactor`被helper当元素factor | 扩形状/换平台前按每个 `sizeof(T)` 重新推导。 |
| 官方 workspace差异 | registered Host预留16 MiB，Direct不传 | 不混用两套launcher/POD/workspace契约。 |
| `idx_type=1`、FP8/HIF8/FP4、多维scale | 官方body/配置部分支持，Direct ABI拒绝 | 视为新的Host+POD+实例矩阵扩展，不只改一个枚举。 |

## 源码关系索引

| 阶段 | 固定官方符号 | Direct 对应关系 |
|---|---|---|
| API/平台 | README；`ops::MoeReRouting` | Torch schema/meta缩窄dtype、shape、属性；950仍为目标平台。 |
| 平台参数 | `TilingPrepare4MoeReRouting` / `GetPlatformInfo` | `PlatformAscendCManager::GetCoreNumAiv/GetCoreMemSize(UB)`。 |
| selector | `MoeReRoutingReTiling::IsCapable`、`MoeReRoutingRTiling::IsCapable` | 保留RE首选与R兜底，在中间插入fast-RE guard。 |
| POD | `MoeReRoutingReTilingData`、`MoeReRoutingRTilingData` macros | 普通C++ struct同字段顺序；外包 `LaunchParams`。 |
| tiling | R/RE `SplitCore/SplitUb/PostTiling` | Host函数复写公式；Direct A>0、scaleSize 0/1、tokenSizeOrigin 0。 |
| router | `moe_re_routing_apt.cpp` keys 200000/200100/210000/210100 | 36个显式entry，不传tiling key。 |
| RE body | `MoeReRoutingReRegbase` | 除include闭包外保留；12实例。 |
| R body | `MoeReRoutingRRegbase` | 除include闭包外保留；12实例。 |
| fast-RE | 官方固定提交无该类 | R helper + RE POD + 派生矩形 `Process`；12实例。 |
| no-scale输出 | 官方 `hasScales=false` 不写scale输出 | entry前置分片清零并做核内`PIPE_ALL` barrier。 |
| workspace | official tiling写16 MiB，body不读 | Direct launcher移除workspace参数。 |
| 构建 | OpDef/binary config接 `moe_re_routing_apt` | `register_direct_launch_op(kernel,plugin,op_kernel)`。 |
