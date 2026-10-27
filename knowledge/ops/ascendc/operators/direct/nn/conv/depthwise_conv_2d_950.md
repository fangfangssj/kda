---
type: Operator
title: "DepthwiseConv2D 950 Direct Invoke 算子知识"
description: "说明 cann_bench::depthwise_conv_2d 在 950 上的接口语义、Direct launcher 参数生成、十条 active Route、单 Kernel 启动图与逐路线实现闭包。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/conv/conv2d_v2/op_host/config/ascend950/conv2d_v2_binary.json
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/conv/conv2d_v2/op_kernel/conv2d_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/conv/conv2d_v2/op_kernel/arch35/conv2d_v2_group.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_kernel/avg_pool_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_kernel/arch35/avg_pool_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel_pad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_host/config/ascend950/avg_pool.json
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T19:32:50Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# DepthwiseConv2D 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::depthwise_conv_2d` → `depthwise_conv_2d_npu`，Direct Invoke；冻结四字符串载荷 SHA-256 为 `5eeb06a130fa6c6f2b40761b9fa39ae76965c1029e932dd1f470f7f72f3e3a2a`。 |
| 计算语义 | NCHW depthwise cross-correlation：每个输出通道只使用同编号输入通道，`groups=C`，FP32 累加后写回输入 dtype。 |
| 输入输出 | `x[N,C,H,W]`、`weight[C,Kh,Kw]`、`bias[C]` → `y[N,C,OH,OW]`；FP16/FP32/BF16，连续 NCHW。 |
| 执行类型 | 所有可达 entry 都声明 `KERNEL_TYPE_AIV_ONLY`；R-001～R-007 为 AIV_VECTOR/MicroAPI，R-008～R-010 为 AIV 上的 SIMT；无 Cube/AIC/MIX_CV 可达节点。 |
| Kernel 路线 | 十条 production terminal Route 均为 `active`，没有独立 `conditional` Route；重要性顺序为 P0/R-001、P1/R-002、P2/R-003、P3/R-006、P4/R-005、P5/R-009、P6/R-007、P7/R-010、P8/R-004、P9/R-008。源码中的 cached/spatial/fixed-plane4/fused 节点为 dormant，不计入十条生产 Route。 |
| 核心资源 | 56 AIV blocks；Vector 路线使用 `TBuf/TQue` 与寄存器，SIMT 路线直接访问 GM 并用线程私有 FP32 累加器；无 workspace。 |
| 来源闭包 | `direct_implementation_closed`：schema/meta → Host selector → C launch ABI → 单个 AIV-only entry → 实际 Vector/SIMT body/helper → 构建注册均闭合。官方固定源码仅是转换相关设计来源；Route、ABI 与资源公式均来自冻结交付件。 |

## 接口与语义

### 计算语义

令 `OH=floor((H+2*padH-dilationH*(Kh-1)-1)/strideH)+1`，`OW` 同理。对 `0<=n<N, 0<=c<C`：

```text
y[n,c,oh,ow] = bias[c]
  + Σ(kh=0..Kh-1) Σ(kw=0..Kw-1)
      pad_zero(x[n,c,oh*strideH-padH+kh*dilationH,
                     ow*strideW-padW+kw*dilationW]) * weight[c,kh,kw]
```

这是不翻转卷积核的 cross-correlation。交付件的 SIMT fallback 即使越界也令 `xValue=0` 后继续执行乘法，因而保留 `0*Inf/NaN` 的 IEEE 传播；各 Vector 路线也先显式零填充再按固定 tap 顺序执行 FP32 `Axpy`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[N,C,H,W]` / contiguous NCHW | FP16/FP32/BF16 | 4-D；与其余张量同 dtype。 |
| `weight` | `[C,Kh,Kw]` / contiguous | 同 `x` | `weight.size(0)=groups=C`，空间维与 `kernelSize` 一致。 |
| `bias` | `[C]` / contiguous | 同 `x` | 必选；`bias.size(0)=C`。 |
| `kernelSize` | 长度 2 | int64 Host 属性 | 等于 `[Kh,Kw]`。 |
| `stride` | 长度 2 | int64 Host 属性 | 两项均大于 0。 |
| `padding` | 长度 2 | int64 Host 属性 | 两项均非负且随后窄化为 `int32_t`。 |
| `dilation` | 长度 2 | int64 Host 属性 | 两项均大于 0。 |
| `groups` | scalar | int64 Host 属性 | 必须等于 `C`。 |
| `y` | `[N,C,OH,OW]` / contiguous NCHW | 同 `x` | `OH>0 && OW>0`，元素数不超过 `UINT32_MAX`。 |

### 入口与版本

入口是交付件注册的 Torch `PrivateUse1` Direct Invoke，而不是 ACLNN，也不复用官方 Conv2DV2 launcher。固定官方仓没有同名 `depthwise_conv_2d` arch35 实现；转换来源由两部分组成：官方 AvgPool arch35 提供 NCHW SIMT grid-stride、padding/UB/gather 组织，官方 Conv2DV2 提供 950 NCHW、三 dtype 与 group-convolution 语义参照。后者的固定入口在 NCHW filter 下是 `KERNEL_TYPE_MIX_AIC_1_2` 且要求 workspace，因此未进入本交付件可达图；这不改变本卡的 VV 分类。

## Host/Launcher 执行流程

### 参数校验与前处理

`depthwise_conv_2d_npu` 先设置输入 device guard，再调用 `depthwise_conv_2d_meta` 顺序检查 rank、属性长度、dtype 一致性与白名单、`groups=C`、weight/bias channel、kernel shape、正 stride/dilation、非负 padding 和正输出空间。随后要求三个输入连续、输出元素数可安全窄化到 `uint32_t`，创建同 device/dtype 的连续输出，取当前 NPU stream，并把维度/属性窄化为 Direct ABI scalar。

### 参数、Workspace 与生命周期

Host 固定 `blocks=56`。`x/weight/bias/y` 的 data pointer、所有 scalar 与当前 stream 被值捕获进同步构造的 launch lambda；`OpCommand::RunOpApi("DepthwiseConv2DDirect", launch)` 调用恰好一个 launcher。workspace size 为 0，没有中间 GM slice。输入和输出由 Torch tensor 持有，stream 上 kernel 完成前必须保持有效；lambda 固定返回 0，交付件未在 lambda 内读取底层 launch 状态。

### 启动图与失败传播

```text
cann_bench::depthwise_conv_2d
  -> depthwise_conv_2d_meta / contiguous + uint32 guards
  -> depthwise_conv_2d_npu::launch selector
  -> exactly one launch_depthwise_conv_2d_* (blocks=56, current stream)
  -> exactly one depthwise_conv_2d_* entry [KERNEL_TYPE_AIV_ONLY]
  -> Vector class::Process / MicroAPI body
     or dtype-specific Simt::VF_CALL(Dim3(512))
  -> y GM
```

单 Kernel 图没有跨节点 workspace 边、event 或 owner 交接。Host 校验失败在 launch 前抛出；selector 每个 dtype 都有最终 fallback。Device 侧只有显式空分核/越界 return/continue，没有 unknown key。lambda 的固定 `0` 意味着错误传播依赖 `RunOpApi`/runtime 边界，源码未提供额外 cleanup。

## Route 与参数设计

### 分核与资源计算

记 `O=OH*OW`、`P=N*C`、`K=Kh*Kw`、`V=(strideW==1)`、`L=(K>1 && O>=512 && P>=64)`。所有 launcher 传 `blockDim=56`。Vector 路线按 block 连续分 plane：`planesPerBlock=ceil(P/56)`；gather 路线按 `plane=block+q*56`；SIMT scalar 路线以全局线程 `block*512+thread` 对输出元素做 grid-stride；plane4 路线每线程处理同一行最多四个相邻输出，并在工作不足 512 groups 时让一个 block 并行多个 plane。

Vector tiled 路线令 `rowPitch=align_up(OW,32/sizeof(T))`。普通 tile 为 8192 元素；packed K3 的 B16/FP32 tile 分别为 8192/4096 元素；packed K5 为 8192 元素。`rowsPerTile=max(1,min(OH,tileElements/rowPitch))`，最后 tile 用 `tileRows=min(rowsPerTile,OH-ohBegin)`。

### Cube/Vector 与核任务映射

| Route family | launch 数与 task type | 56 block 的工作所有权 | block 内执行 | 不存在的层级 |
|---|---|---|---|---|
| R-001/R-002/R-003/R-004/R-005 | 一个 `depthwise_conv_2d_vec_*`，`KERNEL_TYPE_AIV_ONLY` | `planesPerBlock=ceil(P/56)`，每 block 取一个连续 `[planeBegin,planeEnd)` | AIV Vector/MicroAPI，plane 内再按 `ohBegin` 分 tile；空 block 提前 return | 无 AIC/Cube、L1、L0A/L0B/L0C、workspace 或跨核归约 |
| R-006/R-007 | 一个 gather entry，`KERNEL_TYPE_AIV_ONLY` | `plane=block+q*56` 的 block-stride plane | AIV Vector/MicroAPI；每个 block 先在私有 UB 建一次 index 表，再串行处理其 plane | 同上；不存在 Cube→Vector 交接 |
| R-008/R-009 | 一个 plane4 entry，`KERNEL_TYPE_AIV_ONLY` | block-stride plane group；`groupsPerPlane<512` 时以 `planeLane` 在同一 block 打包多个 plane | 每个 AIV block 发 `VF_CALL(Dim3(512))`，每线程拥有一个四输出 group | 无显式 TQue/TBuf、共享归约、atomic、AIC/Cube 或 workspace |
| R-010 | 一个 dtype base entry，`KERNEL_TYPE_AIV_ONLY` | 逻辑线程以 `56*512` 为全局步长覆盖 `totalOutput`；仅当 `totalOutput<=4,294,938,624` 时能证明 `uint32_t linear` 的末次加法不回绕 | 每个 SIMT thread 独占一个 `linear` 输出并串行遍历 tap | 同上；没有第二 kernel 或失败重试；超安全上界不是正确性 fallback |

`KERNEL_TYPE_AIV_ONLY` 约束的是 entry 的核任务类型；R-008～R-010 虽在 AIV entry 内调用 SIMT body，也不能据此归类为 AIC/Cube 路线。所有 active Route 都只有一个 runtime kernel 节点。

### Route 选择

重要性只表示阅读、排障与复用时的关注顺序，不改变 Host selector 的实际判断顺序，也不是运行时分值。排序先考虑复杂特化的生产价值与迁移耦合，再考虑 selector 的前置优先级和 fallback 覆盖价值；Route ID 保持原有实现身份。这里的 `active/conditional/dormant` 描述生产调用链可达性：依赖合法 shape 谓词不等于 `conditional`，只要公开 schema 的某组合法输入能由当前 Host 直接命中，就是 `active`。

#### 生产状态与合法输入域完备划分

除前述 `O/P/K/V/L` 外，记：

```text
K3 = (Kh==3 && Kw==3)
K5 = (Kh==5 && Kw==5)
F  = (V && O>=512 && P>=64)                       // FP32 vector outer gate
B  = (V && K>1 && O>=512 && P>=64 && OW>=16)     // BF16 vector outer gate
Q  = (K==1 && P>=64)                              // BF16 fixed-K1 outer gate
```

下表给出经过 dtype 短路后的**完整终端谓词**。同一 dtype 内各行互斥，合起来覆盖通过 Host 校验的全部输入；每条都有 production Host producer、launcher、entry 与 body，因此十条均为 `active`。当前实现没有依赖 capability 探测、外部前置图或编译开关才能成功的 `conditional` terminal Route。

| 状态 | Route | 完整终端谓词 | selector 终点与未命中出边 |
|---|---|---|---|
| active | R-001 | `(FP16 && V && K3) \|\| (BF16 && B && K3)` | 进入 vector entry 后由 Device `usePackedK3` 终止；FP16/BF16 内层未命中分别去 R-004。 |
| active | R-002 | `FP32 && F && K3 && O<=8192` | Device packed-K3 终止；`O>8192` 或非 K3 在同一 entry 去 R-005。 |
| active | R-003 | `BF16 && B && !K3 && K5` | Device K5 分支终止；非 K3/K5 去 R-004。 |
| active | R-006 | `FP16 && L && !V && K3` | FP16 Host 第一分支终止；非 K3 继续到 R-008。 |
| active | R-005 | `FP32 && F && !(K3 && O<=8192)` | FP32 vector entry 的 Device fallback；该 entry 内终止。 |
| active | R-009 | `BF16 && Q` | BF16 Host 第一分支终止；`!Q` 继续检查 B。 |
| active | R-007 | `FP32 && !F && L && K5 && strideH==2 && strideW==2` | plane4 family 内的 K5 gather 终止；未命中去 R-008。由于 `strideW==2`，此式中的 `!F` 是显式保留的上游短路事实。 |
| active | R-010 | `(FP16 && !V && !L) \|\| (FP32 && !F && !L) \|\| (BF16 && !Q && !B && (!L \|\| OW<48))` | 各 dtype 最终 `else`，无下一条计算 Route；只有 `totalOutput<=UINT32_MAX-28672+1=4,294,938,624` 时 grid-stride 加法可证明不回绕。Host 仍接受到 `UINT32_MAX`，所以该 Route 的安全域不能覆盖整个 Host 合法域。 |
| active | R-004 | `(FP16 && V && !K3) \|\| (BF16 && B && !K3 && !K5)` | B16 vector entry 的 Device fallback；该 entry 内终止。 |
| active | R-008 | `(FP16 && L && !V && !K3) \|\| (FP32 && !F && L && !(K5 && strideH==2 && strideW==2)) \|\| (BF16 && !B && L && OW>=48)` | 各 dtype 的通用 plane4 终点；只有 Host 未命中该式时才继续到 dormant 判断或 R-010。 |

空 `N` 或空 `C` 没有被 Meta 拒绝：此时 `totalOutput=0`、`P=0`。FP16 且 `V` 仍可能进入 vector entry，但 `planeBegin>=planeEnd` 立即返回；其余 dtype 会落到无工作量的 R-010。这个空域行为属于现有 active Route 的空任务边界，不另拆 Route。

#### 重要性特征

| 重要性 | 状态 | Route | 主要特征 | 执行族/资源重心 | 复用关注点 |
|---|---|---|---|---|---|
| P0 | active | R-001 | FP16/BF16、strideW=1、3×3 packed | AIV Vector；9 patch TBuf + 寄存器 FP32 累加 | B16 主 packed 路线，先核对 dtype 外层门槛与 Device K3 二级判断 |
| P1 | active | R-002 | FP32、strideW=1、3×3、O≤8192 | AIV Vector；4096-element slots | FP32 packed 路线，O 上界决定是否短路到 general |
| P2 | active | R-003 | BF16、strideW=1、5×5 packed | AIV Vector；5 patch slots 分组复用 + FP32 acc | K5 五组归约与 buffer 复用必须成对迁移 |
| P3 | active | R-006 | FP16、非 strideW=1、3×3 gather | AIV Vector；padded plane + index + output | Host 在 FP16 分支最先判断，命中即短路 |
| P4 | active | R-005 | FP32 通用 tiled Vector | AIV Vector；acc 直接 MTE3 | 承接 FP32 Vector 入口内未命中 packed K3 的形状 |
| P5 | active | R-009 | BF16、K=1 fixed plane4 | AIV-SIMT；GM direct + 固定 1 tap | BF16 分支第一优先，先于全部 Vector 条件 |
| P6 | active | R-007 | FP32、5×5、strideH=strideW=2 gather | AIV Vector；FP32 padded plane/index/output | 仅在 FP32 Vector 入口未命中后进入 plane4 family 再判断 |
| P7 | active | R-010 | 三 dtype 通用 scalar fallback | AIV-SIMT；GM direct + 单输出 FP32 累加器 | selector 终端兜底；复用必须先满足 u32 grid-stride 安全上界 |
| P8 | active | R-004 | FP16/BF16 通用 tiled Vector | AIV Vector；queue + acc/term TBuf | 承接 B16 Vector 入口内非 K3/K5 分支 |
| P9 | active | R-008 | 三 dtype 四输出 plane4 | AIV-SIMT；GM direct + 4 个 FP32 累加器 | 位于各 dtype 不同层级，必须按 Host 顺序分别理解 |

#### Host selector 与短路

| 重要性 | Route | Host 顺序 | Key/模板 | 精确选择条件、短路与 fallback |
|---|---|---|---|---|
| P0 | R-001 | FP16 #4 Device #1；BF16 #2 Device #1 | `half/bfloat16_t` | FP16 在 vector 入口中以 K3 命中；BF16 在 vector 入口中以 K3 命中；命中后不再检查该 dtype 后项。 |
| P1 | R-002 | FP32 #2 Device #1 | `float` | `V&&O>=512&&P>=64` 进入 vector，随后 `K3&&O<=8192` 命中 packed；否则在同一入口落到 R-005。 |
| P2 | R-003 | BF16 #2 Device #2 | `bfloat16_t` | BF16 vector 入口中 K3 未命中后，以 K5 命中；否则在同一入口落到 R-004。 |
| P3 | R-006 | FP16 #1 | `half,3` | `L&&!V&&Kh==3&&Kw==3`；命中即短路，未命中继续 FP16 #2。 |
| P4 | R-005 | FP32 #2 Device fallback | `float` | 已进入 FP32 vector，但 R-002 的内层条件不成立；该入口内终止，不再检查 FP32 后项。 |
| P5 | R-009 | BF16 #1 | `bfloat16_t,1` | `K==1&&P>=64`；命中即短路，未命中继续 BF16 #2。 |
| P6 | R-007 | FP32 #3 Device #1 | `float,5` | FP32 #2 未命中后，以 `L` 进入 plane4 family；其中 `K5&&strideH==2&&strideW==2` 命中 gather，否则落到 R-008。 |
| P7 | R-010 | FP16 #5；FP32 #5；BF16 #5 | `T` | 各 dtype 的最终 `else`；前序全部未命中时终止选择。 |
| P8 | R-004 | FP16 #4 Device fallback；BF16 #2 Device fallback | `half/bfloat16_t` | 已进入相应 B16 vector，但 packed K3/K5 未命中；该入口内终止，不再检查该 dtype 后项。 |
| P9 | R-008 | FP16 #2；FP32 #3 Device fallback；BF16 #3 | `T,FixedKernel=0` | FP16：`L&&!V` 且 R-006 未命中；FP32：以 `L` 进入 family 且 R-007 未命中；BF16：`L&&OW>=48`。命中后短路，否则继续后项。 |

Host 中还有三个显式关闭的判断位置：FP16 #3、FP32 #1/#4、BF16 #4；对应 `useCachedRoute=false` 或 `useFp32SpatialRoute=false`，因此不会产生可达 Route，也不会改变上表的短路关系。

```text
dtype=FP16:
  #1 R-006 if L&&!V&&K3
  #2 R-008 if L&&!V
  #3 cached route disabled
  #4 Vector entry if V: Device R-001 if K3, else R-004
  #5 R-010 fallback

dtype=FP32:
  #1 spatial route disabled
  #2 Vector entry if V&&O>=512&&P>=64:
       Device R-002 if K3&&O<=8192, else R-005
  #3 plane4 family if L:
       Device R-007 if K5&&strideH==2&&strideW==2, else R-008
  #4 cached route disabled
  #5 R-010 fallback

dtype=BF16:
  #1 R-009 if K==1&&P>=64
  #2 Vector entry if V&&K>1&&O>=512&&P>=64&&OW>=16:
       Device R-001 if K3, R-003 if K5, else R-004
  #3 R-008 if L&&OW>=48
  #4 cached route disabled
  #5 R-010 fallback
```

#### Dormant 源码与重新启用条件

下列节点随 translation unit 编译或保留在同一源码中，但当前 `depthwise_conv_2d_npu` 不能把合法调用送到它们。它们不计入 production Route，也不是 R-010 的失败重试；只有补齐表中的 producer、容量与语义证明后才能重新编号为 Route。

| dormant family | ABI/entry 可见性与已存在闭包 | 生产链不可达原因 | 重新启用的最低条件 |
|---|---|---|---|
| cached FP16/FP32/BF16 | ABI 可见：launch header 有三项 C 声明，translation unit 有三项 C launcher 和三项 global entry；闭包为 `launch_depthwise_conv_2d_cached_*` 循环发多批 kernel → `depthwise_conv_2d_cached_*` → `DepthwiseConv2dCached<T>` → `DepthwiseConv2dPlaneUbSimt<T>` | 三个 dtype 分支共享的 `useCachedRoute` 被常量赋为 `false`；所有更早分支未命中也会跳过它并进入 R-010 | 增加可达 Host 谓词；证明 `IH*IW*sizeof(T)<=32 KiB`、common ABI/`planeOffset` 位宽、单批 `currentBlocks` 与多 kernel 同流顺序；重新核对 UB-SIMT 的 padding/精度语义和相对 R-010 的价值。 |
| FP32 spatial reuse | ABI 可见：有 C 声明、C launcher 与 global `depthwise_conv_2d_spatial_fp32` entry；闭包到 `DepthwiseConv2dSpatialFp32` | FP32 第一判断的 `useFp32SpatialRoute` 被常量赋为 `false`；plugin 计算的 `inputRowPitch/outputRowPitch` 只服务这个关闭的判断且不进入 ABI | 增加只接受 `strideH=strideW=1` 的 Host 谓词；证明 padded input `<=8192` FP32 元素、acc `<=8192` 元素、repeat stride/次数和 DMA 的 u8/u16 字段均可表达；否则不能调用忽略 stride 参数的 body。 |
| fixed plane4 FP16-K3 / FP32-K5 | ABI 可见：两项都有 C 声明、C launcher 与 global entry；FP16 闭包到 `DepthwiseConv2dPlane4Simt<half,3>`，FP32 launcher 调用源码拼写为 `depthwise_conv_2_d_plane4_fp32_k5` 的 entry 并闭包到 `DepthwiseConv2dPlane4Simt<float,5>` | production selector 对相同特征分别调用 R-006/R-007 gather launcher，从不调用这两个 fixed-plane4 launcher | 显式改变两个 Host 特化分支并保留 K3/K5、dtype 与 stride 条件；核对 entry 符号拼写、512-thread ownership、精度顺序和无 UB 的 GM 访存代价。仅存在 launcher 不能构成 fallback。 |
| fused K×K Vector template | 仅源码潜伏：只有 `DepthwiseConv2dFused<T,Kernel>`、`DepthwiseConv2dFusedVf<T,Kernel>` 与递归 helper；没有 global entry、C launcher、launch header ABI 声明或 Host selector 实例 | 缺少 entry/launcher/Host 三层生产接线，因而不是 ABI 可见 dormant family | 新增明确的 `T/Kernel` entry、launcher、header 声明与 Host predicate；证明 padded input `<=20000` 元素、padded output `<=17000` 元素、`OW` 不超过 body 实际处理的两个 FP32 vector、stride 固定为 1，以及 DMA/event/rounding 边界。 |

`DepthwiseConv2dPlaneUbSimt` 是 cached family 的 Device 叶子，不另算 dormant Route；`inputRowPitch/outputRowPitch` 两个 plugin 局部量是 spatial 判断的 inactive 计算，也不构成 Route。cached/spatial/fixed-plane4 保留 ABI 可见符号，fused template 则仅影响源码维护、没有 ABI 身份；当前知识消费不得把任一 family 写入可达图。

### 模板/实现执行矩阵

| 重要性 | 状态 | Route | Host 顺序 | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|---|---|---|
| P0 | active | R-001 | FP16 #4 Device #1；BF16 #2 Device #1 | Host + inner K3 | 见完备划分表 | 单 AIV Vector | `DepthwiseConv2dVector<T>::ProcessPackedK3` → `DepthwisePackedK3Vf<T>` | 9 patch + output TBuf；MTE2/V/MTE3 event | mask 与二维 DMA tail |
| P1 | active | R-002 | FP32 #2 Device #1 | FP32 inner K3 | 见完备划分表 | 单 AIV Vector | 同上，`T=float` | 4096-element slots；总 UB 同 R-001 | `O>8192` 去 R-005 |
| P2 | active | R-003 | BF16 #2 Device #2 | BF16 inner K5 | 见完备划分表 | 单 AIV Vector | `ProcessPackedK5` → 五次 `DepthwisePackedK5GroupVf` | 5 patch slots 循环复用 + FP32 acc | group 0/4 控制 init/store |
| P3 | active | R-006 | FP16 #1 | FP16 K3 | 见完备划分表 | 单 AIV Vector | `DepthwiseConv2dStride2Gather<half,3>` | raw+FP32 padded plane/index/output | mask=`remaining` |
| P4 | active | R-005 | FP32 #2 Device fallback | FP32 general | 见完备划分表 | 单 AIV Vector | general body，`T=float` | 无 outputQueue；acc 直接 MTE3 | row/tap/valid rectangle tail |
| P5 | active | R-009 | BF16 #1 | BF16 K1 | 见完备划分表 | 单 AIV→SIMT | `DepthwiseConv2dPlane4Simt<bfloat16_t,1>` | GM direct；固定展开 1 tap | 每个 store 有 `ow<OW` guard |
| P6 | active | R-007 | FP32 #3 Device #1 | FP32 K5 gather | 见完备划分表 | 单 AIV Vector | `DepthwiseConv2dStride2Gather<float,5>` | FP32 padded plane/index/output | mask=`remaining` |
| P7 | active | R-010 | 三 dtype #5 | fallback | dtype 最终 else；另需满足 u32 步进安全上界 | 单 AIV→SIMT | `DepthwiseConv2dNchwSimt<T>` | GM direct；512-thread grid-stride | `linear<totalOutput` 只保护比较；加法仍需防回绕 |
| P8 | active | R-004 | FP16 #4 Device fallback；BF16 #2 Device fallback | B16 general | 见完备划分表 | 单 AIV Vector | `DepthwiseConv2dVector<T>::Process` general body | input/output queue + acc/term TBuf | row/tap/valid rectangle tail |
| P9 | active | R-008 | FP16 #2；FP32 #3 Device fallback；BF16 #3 | plane4 | 见完备划分表 | 单 AIV→SIMT | `DepthwiseConv2dPlane4Simt<T,0>` | GM direct；512 threads；4 FP32 accumulators | 每个 store 有 `ow<OW` guard |

### Selector 与资源 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AIV-only | 每个可达 entry 调用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` | 阻止调度到 Cube/AIC | 全部 | 无 CV fallback | `depthwise_conv_2d_*` entries |
| 大 plane/并行度 | `O>=512 && P>=64` | 避免复杂 plane/Vector 路线落在小工作量 | R-001～R-009 的相应分支 | R-010 或 dtype 后续项 | `depthwise_conv_2d_npu` |
| FP16 Vector | `strideW==1` | `BuildTapDesc` 的行内 patch 是连续 `OW` 宽 | R-001/R-004 | 非 1 去 R-006/R-008/R-010 | `useVectorRoute` |
| BF16 Vector | `V&&K>1&&O>=512&&P>=64&&OW>=16` | 摊薄 BF16 cast/patch 管线 | R-001/R-003/R-004 | R-008/R-010 | `useBf16VectorRoute` |
| FP32 packed K3 | `K3 && O<=8192`（在 FP32 Vector 内） | FP32 packed tile 只有 4096 元素，限制 plane 过大时的 patch 行为 | R-002 | R-005 | `usePackedK3` |
| plane4 | `K>1&&O>=512&&P>=64` | 每 thread 共享一次 weight 给四个相邻输出 | R-008 | R-010 | `usePlane4Route` |
| gather specialization | FP16 K3 或 FP32 K5 且 Host 条件命中 | 模板数组固定为 9/25 tap | R-006/R-007 | R-008 | Host nested `if` + entry template |
| tile row pitch | `rowPitch=align_up(OW,32/sizeof(T))` | 二维 DMA block 对齐与 row padding | R-001～R-005 | 无 runtime fallback | `DepthwiseConv2dVector` |
| dynamic UB bounds | Host 未显式校验各 `InitBuffer` 总量和字段窄化 | 超大 shape 可能超过 UB 或 DMA 参数位宽 | R-001～R-007 | 无 fallback；属于失败边界 | 各 `Init` |
| dead routes | `useCachedRoute=false`、`useFp32SpatialRoute=false` | 冻结 selector 不可达 | 不属于正式 Route | N/A | `depthwise_conv_2d_npu` |

### POD/ABI

本实现没有 tiling POD struct、tiling key、workspace 描述符或 runtime tiler；`O/P/K/V/L/F/B/Q` 只在 Host launch lambda 中即时计算，不序列化到 GM。ABI 是位置固定的 C scalar 参数。所有 shape/属性 producer 都是 `depthwise_conv_2d_npu`，consumer 是相应 launcher、entry 与 `Init`/SIMT body；common ABI 与 vector ABI 不能互换，因为 common arg4 是 `totalOutput`，vector device arg4 是 `batch` 且完全省略 `strideW`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `DEPTHWISE_LAUNCH_ARGS` | `x` | `GM_ADDR` | arg0 | GM pointer | `x.data_ptr()` | GM tensor/input load | tensor lifetime |
| 同上 | `weight` | `GM_ADDR` | arg1 | GM pointer | `weight.data_ptr()` | per-channel tap load | 同 dtype |
| 同上 | `bias` | `GM_ADDR` | arg2 | GM pointer | `bias.data_ptr()` | FP32 accumulator seed | 长度 C |
| 同上 | `y` | `GM_ADDR` | arg3 | GM pointer | `y.data_ptr()` | output store | output owner=Torch |
| 同上 | `blocks` | `uint32_t` | launcher-only arg4 | block | 常量 56 | `<<<blocks>>>` | 固定 950 快照 |
| 同上 | `totalOutput` | `uint32_t` | device arg4 | element | `uint32_t(y.numel())` | loop bound/derive `P` | `<=UINT32_MAX` |
| 同上 | `channels` | `uint32_t` | device arg5 | element/channel | `x.size(1)` | `plane%C` | Host 未逐字段检查 uint32 |
| 同上 | `inputH` | `uint32_t` | device arg6 | element | `x.size(2)` | address/padding | 同上 |
| 同上 | `inputW` | `uint32_t` | device arg7 | element | `x.size(3)` | address/row pitch | 同上 |
| 同上 | `outputH` | `uint32_t` | device arg8 | element | `y.size(2)` | plane/tile loop | 正数 |
| 同上 | `outputW` | `uint32_t` | device arg9 | element | `y.size(3)` | row/tail loop | 正数 |
| 同上 | `kernelH` | `uint32_t` | device arg10 | tap | `kernelSize[0]` | tap loop/template guard | 与 weight 一致 |
| 同上 | `kernelW` | `uint32_t` | device arg11 | tap | `kernelSize[1]` | tap loop/template guard | 与 weight 一致 |
| 同上 | `strideH` | `uint32_t` | device arg12 | input element/output step | `stride[0]` | input origin/row DMA stride | `>0` |
| 同上 | `strideW` | `uint32_t` | device arg13 | input element/output step | `stride[1]` | SIMT/gather address | `>0` |
| 同上 | `padH` | `int32_t` | device arg14 | input element | `int32_t(padding[0])` | input origin/padded plane | 非负但可能窄化 |
| 同上 | `padW` | `int32_t` | device arg15 | input element | `int32_t(padding[1])` | input origin/row pitch | 非负但可能窄化 |
| 同上 | `dilationH` | `uint32_t` | device arg16 | input element/tap | `dilation[0]` | tap offset | `>0` |
| 同上 | `dilationW` | `uint32_t` | device arg17 | input element/tap | `dilation[1]` | tap offset | `>0` |
| launcher | `stream` | `void*` | launcher final arg | stream handle | current NPU stream | kernel launch configuration 的 stream 位 | current-device lifetime |
| `DEPTHWISE_VECTOR_LAUNCH_ARGS` | `batch` | `uint32_t` | device arg4 | batch | `uint32_t(x.size(0))` | `P=batch*channels` | 乘法溢出未显式审计 |
| 同上 | `channels..dilationW` | 同上 | device arg5..16 | 同上 | 同上 | `DepthwiseConv2dVector::Init` | 不传 `strideW`；Host 已保证 1 |

### Route 源码闭包

| 重要性 | 状态 | Route | Host 顺序 | C launcher → Kernel entry | POD/参数 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|---|
| P0 | active | R-001 | FP16 #4 Device #1；BF16 #2 Device #1 | `launch_depthwise_conv_2d_vec_fp16/bf16` → `depthwise_conv_2d_vec_fp16/bf16` | vector scalar ABI | `DepthwiseConv2dVector::ProcessPackedK3`、`DepthwisePackedK3Vf` | operator `CMakeLists.txt::register_direct_launch_op` |
| P1 | active | R-002 | FP32 #2 Device #1 | `launch_depthwise_conv_2d_vec_fp32` → `depthwise_conv_2d_vec_fp32` | vector scalar ABI | 同上，`T=float` | 同上 |
| P2 | active | R-003 | BF16 #2 Device #2 | `launch_depthwise_conv_2d_vec_bf16` → `depthwise_conv_2d_vec_bf16` | vector scalar ABI | `ProcessPackedK5`、`DepthwisePackedK5GroupVf` | 同上 |
| P3 | active | R-006 | FP16 #1 | `launch_depthwise_conv_2d_stride2_gather_fp16_k3` → 同名 entry | common scalar ABI | `DepthwiseConv2dStride2Gather<half,3>`、`DepthwiseStride2GatherVf` | 同上 |
| P4 | active | R-005 | FP32 #2 Device fallback | `launch_depthwise_conv_2d_vec_fp32` → `depthwise_conv_2d_vec_fp32` | vector scalar ABI | `DepthwiseConv2dVector::Process/BuildTapDesc/CopyTapIn`，FP32 direct MTE3 | 同上 |
| P5 | active | R-009 | BF16 #1 | `launch_depthwise_conv_2d_plane4_bf16_k1` → 同名 entry | common scalar ABI | `DepthwiseConv2dPlane4Simt<bfloat16_t,1>` | 同上 |
| P6 | active | R-007 | FP32 #3 Device #1 | `launch_depthwise_conv_2d_stride2_gather_fp32_k5` → 同名 entry | common scalar ABI | `DepthwiseConv2dStride2Gather<float,5>`、同 helper | 同上 |
| P7 | active | R-010 | 三 dtype #5 | `launch_depthwise_conv_2d_{fp16\|fp32\|bf16}` → `depthwise_conv_2d_{fp16\|fp32\|bf16}` | common scalar ABI；Host 上界宽于 Device 安全域 | `DepthwiseConv2dNchwSimt<T>` | 同上 |
| P8 | active | R-004 | FP16 #4 Device fallback；BF16 #2 Device fallback | `launch_depthwise_conv_2d_vec_fp16/bf16` → `depthwise_conv_2d_vec_fp16/bf16` | vector scalar ABI | `DepthwiseConv2dVector::Process/BuildTapDesc/CopyTapIn` | 同上 |
| P9 | active | R-008 | FP16 #2；FP32 #3 Device fallback；BF16 #3 | `launch_depthwise_conv_2d_plane4_{fp16\|fp32\|bf16}` → `depthwise_conv_2d_plane4_{fp16\|fp32\|bf16}` | common scalar ABI | `DepthwiseConv2dPlane4Simt<T,0>`、`DepthwiseConv2dPlane4AccumulateOne` | 同上 |

十行的 Host/Tiling producer 都是 `op_plugin/depthwise_conv_2d_plugin.cpp::depthwise_conv_2d_npu`；它既生成 positional scalar，也完成 Route 短路。该表把 launcher 与 entry 分开列出，避免把 C ABI wrapper 当成 Device router。

## Kernel 实现

### 公共执行流程

所有 Route 都是单 Kernel、当前 stream、无 workspace。AIV entry 先声明 `KERNEL_TYPE_AIV_ONLY`。Vector entry 构造 `TPipe` 和模板类，`Init` 绑定 GM/分配 UB，`Process` 按 plane/tile 搬运并执行 MicroAPI；SIMT entry 直接以 `Dim3(512)` 发起 dtype-specific `VF_CALL`。所有路线都用 FP32 累加；Vector 的 B16 输出显式使用 `CAST_RINT`，SIMT 的 FP16/BF16 输出使用源码中的 `static_cast<T>`，固定源码未进一步暴露其 rounding mode；FP32 保持 FP32。padding lane 被清零但仍参加乘法，tail lane 由 mask 或 store guard 屏蔽。

### Route R-001（P0）：B16 packed-K3 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_plugin/depthwise_conv_2d_plugin.cpp` | dtype branch | FP16 `V`；BF16 `useBf16VectorRoute` | vector launcher |
| Device router | `op_kernel/depthwise_conv_2d_kernel.cpp` | `DepthwiseConv2dVector::Process` | `Kh==3&&Kw==3&&(!float\|\|O<=8192)` | `ProcessPackedK3` |

#### Device 数据流

##### 函数级调用链

```text
launch_*_vec -> depthwise_conv_2d_vec_{fp16|bf16}
  -> DepthwiseConv2dVector<T>::Init/Process
  -> ProcessPackedK3 -> nine DataCopyPad patches
  -> DepthwisePackedK3Vf -> recursive nine Axpy -> DepthwiseStoreAlign
  -> DataCopyPad(outputLocal -> y)
```

##### 源码等价伪码

```text
for plane in block-owned contiguous range:
  load bias and 9 weights as fp32
  for ohBegin by rowsPerTile:
    zero each of 9 patch slots; DMA valid rectangles with horizontal padding
    wait MTE2->V
    for each vector mask: acc=bias; for tap=0..8: acc += fp32(patch[tap])*w[tap]
    cast/store masked result to packedOutput; wait V->MTE3; copy valid rows to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inputPatchBuffer` | `tap*8192` | `9*8192*2=147456 B` | B16, 9 planar slots | MTE2 | VF recursive Axpy | TPipe/kernel | per tile overwrite |
| `packedOutputBuffer` | 0 | `8192*2=16384 B` | B16 padded rows | VF | MTE3 | TPipe/kernel | per tile overwrite |
| registers | vector offset | one FP32 vector acc + value | FP32 | Duplicate/Cast | Axpy/store | VF iteration | mask tail |

##### 分层内存布局

```mermaid
block-beta
  columns 10
  patches["9 patch slots · 147456 B"]:9
  output["output · 16384 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 147456 + 16384 = 163840 B
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 9 patch slots | zero + MTE2 | VF | next tile | V_MTE2/MTE2_V | VF 完成后才重写 |
| output slot | VF | MTE3 | next tile | V_MTE3/MTE3_V | MTE3 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| clear→DMA | `PIPE_V` + `V_MTE2` | Vector clear | MTE2 | patches | padding=0 |
| DMA→VF | `MTE2_V` | MTE2 | VF | patches | B16→FP32 zeroing cast |
| VF→DMA | `V_MTE3/MTE3_V` | VF | MTE3/next tile | output | `UpdateMask(remaining)`, `CAST_RINT` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_plugin/depthwise_conv_2d_plugin.cpp` | `depthwise_conv_2d_npu` | dtype/shape selector、56 blocks |
| Kernel entry/router | `op_kernel/depthwise_conv_2d_kernel.cpp` | `depthwise_conv_2d_vec_fp16/bf16` | AIV-only 到 `DepthwiseConv2dVector<T>` |
| Device body/helper | 同上 | `ProcessPackedK3/DepthwisePackedK3Vf` | 9 patch、寄存器驻留归约、mask store |


### Route R-002（P1）：FP32 packed-K3 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | `useFp32VectorRoute` | `V&&O>=512&&P>=64` | FP32 vector |
| Device | kernel | `usePackedK3` | `Kh=Kw=3&&O<=8192` | packed K3；否则 R-005 |

#### Device 数据流

##### 函数级调用链

```text
launch_vec_fp32 -> entry -> DepthwiseConv2dVector<float>::ProcessPackedK3
  -> 9 patch DMA -> DepthwisePackedK3Vf<float> -> y
```

##### 源码等价伪码

```text
与 R-001 同序；tileElements=4096，patch/value/output 均 FP32，无 dtype cast。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| patches | `tap*4096` | `9*4096*4=147456 B` | FP32 | MTE2 | VF | kernel | per tile |
| output | 0 | `4096*4=16384 B` | FP32 | VF | MTE3 | kernel | per tile |

##### 分层内存布局

```mermaid
block-beta
  columns 10
  patches["9 FP32 slots · 147456 B"]:9
  output["FP32 output · 16384 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=163840 B; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| patch/output | MTE2/VF | VF/MTE3 | next tile | hard events | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile pipeline | `PIPE_V` + hard events | V/MTE2/V | next stage | UB | FP32；mask + 2-D output tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | `useFp32VectorRoute` | 外层 predicate |
| Kernel entry/router | kernel | `depthwise_conv_2d_vec_fp32` | `T=float` |
| Device body/helper | kernel | `ProcessPackedK3` | 4096-element FP32 specialization |


### Route R-003（P2）：BF16 packed-K5 分组 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | `useBf16VectorRoute` | `V&&K>1&&O>=512&&P>=64&&OW>=16` | BF16 vector |
| Device | kernel | `Process` | `Kh=Kw=5` | `ProcessPackedK5` |

#### Device 数据流

##### 函数级调用链

```text
launch_vec_bf16 -> DepthwiseConv2dVector<bfloat16_t>::ProcessPackedK5
  -> for group=0..4: DMA five tap slots -> DepthwisePackedK5GroupVf<First,Last>
  -> group 0 seeds bias; groups 0..3 store FP32 acc; group 4 casts to output -> MTE3
```

##### 源码等价伪码

```text
for each tile:
  for group in 0..4:
    clear/copy taps [group*5 .. group*5+4] into same five slots
    acc = bias if group=0 else packedAcc
    perform five BF16->FP32 Axpy in tap order
    store packedAcc unless group=4, then CAST_RINT to BF16 output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| five patches | `lane*8192` | `5*8192*2=81920 B` | BF16 | MTE2 | VF | tile | reused by five groups |
| output | 0 | `8192*2=16384 B` | BF16 | last VF | MTE3 | tile | final group only |
| accumulator | 0 | `8192*4=32768 B` | FP32 | group 0..3 | next group | tile | explicit cross-group state |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  patches["5 BF16 slots · 81920 B"]:5
  acc["FP32 acc · 32768 B"]:1
  output["BF16 output · 16384 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=81920+32768+16384=131072 B; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| five patch slots | group DMA | same group VF | next group DMA | MTE2_V then VF completion | groups serial |
| FP32 acc | groups 0..3 | next group | next group writeback | single VF call order | group order fixed 0→4 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| group load | V_MTE2/MTE2_V | clear/MTE2 | VF | patch slots | mask on `computeElements` |
| group state | program order | VF group g | VF group g+1 | FP32 acc | BF16 inputs, FP32 ordered sum |
| final output | V_MTE3/MTE3_V | group 4 | MTE3 | output | `CAST_RINT` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | BF16 branch | vector eligibility |
| Kernel entry/router | kernel | `depthwise_conv_2d_vec_bf16` | AIV-only |
| Device body/helper | kernel | `ProcessPackedK5`、`DepthwisePackedK5GroupVf` | 25 taps 分五组并复用 UB |


### Route R-006（P3）：FP16 stride/gather K3 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | FP16 nested selector | `L&&!V&&Kh=Kw=3` | K3 gather entry |

#### Device 数据流

##### 函数级调用链

```text
DepthwiseConv2dStride2Gather<half,3>::Process
  -> build baseIndex[linear]=oh*strideH*inputPitch+ow*strideW
  -> zero padded raw plane -> MTE2 input -> Cast raw to FP32
  -> DepthwiseStride2GatherVf<half,3> -> 9 Gather/Axpy -> B16 output -> MTE3
```

##### 源码等价伪码

```text
index[o]=oh*sH*inputPitch+ow*sW; DSB(UB)
for plane=block; plane<P; plane+=56:
  raw=0; DMA x into padded raw; cast whole padded raw to fp32
  for vector chunk: acc=bias; for tap: gatherIndex=index+kh*dH*pitch+kw*dW;
                    acc += Gather(fp32Input,gatherIndex)*weight[tap]
  CAST_RINT to FP16 output; DMA O elements to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| raw padded input | `padH*pitch` | `2*PI` bytes | FP16, `pitch=align_up(IW+2padW,16)` | V+MTE2 | Cast | plane | overwrite next plane |
| FP32 input | 0 | `4*PI` bytes | FP32 padded | Cast | Gather | plane | same plane |
| base index | `o` | `4*O` bytes | uint32 | scalar SetValue | VF | kernel | initialized once |
| output | 0 | `2*O` bytes | FP16 | VF | MTE3 | plane | overwrite next plane |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  raw["raw FP16 · 2PI"]:1
  fp32["input FP32 · 4PI"]:1
  index["index u32 · 4O"]:1
  output["output FP16 · 2O"]:1
```

##### 字节账与复用顺序

```text
PI=(IH+2*padH)*align_up(IW+2*padW,16)
UB_total=6*PI+6*O bytes; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| baseIndex | scalar loop | VF | never in kernel | DSB(UB) | O entries fit UB |
| input/output | V/MTE2/VF | Cast/Gather/MTE3 | next plane | hard events | serial plane loop |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| index | `DataSyncBarrier<UB>` | scalar writes | VF | index table | exact uint32 index |
| input/output | PIPE_V + hard events | V/MTE2/VF | next stage | plane buffers | FP16→FP32; mask `remaining`; RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | FP16 selector | large non-strideW1 K3 |
| Kernel entry/router | kernel | `depthwise_conv_2d_stride2_gather_fp16_k3` | `DepthwiseConv2dStride2Gather<half,3>` |
| Device body/helper | kernel | `Process/DepthwiseStride2GatherVf` | padded plane、index、Gather/Axpy |


### Route R-005（P4）：FP32 通用 tiled Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | `useFp32VectorRoute` | `V&&O>=512&&P>=64` | vector entry |
| Device | kernel | `usePackedK3` false | 非 R-002 | general FP32 |

#### Device 数据流

##### 函数级调用链

```text
DepthwiseConv2dVector<float>::Process
  -> tap DMA -> Adds(term,input,0) -> Axpy(acc,term,w)
  -> hard-event protected DataCopyPad(acc -> y)
```

##### 源码等价伪码

```text
与 R-004 相同，但输入无需 Cast，输出 acc TBuf 直接由 MTE3 写 GM。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputQueue | 0 | `8192*4=32768 B` | FP32 | MTE2 | Adds | per tap | depth1 |
| acc/term | 0 | each `32768 B` | FP32 | V | V/MTE3 | tile | term per tap |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["VECIN d=1 · 32768 B"]:1
  acc["acc · 32768 B"]:1
  term["term · 32768 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=98304 B; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| acc | Duplicate/Axpy | next Axpy/MTE3 | next tile | PIPE_V + V_MTE3/MTE3_V | MTE3 done before reuse |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| output | V_MTE3/MTE3_V | FP32 acc | DMA/next tile | acc | 2-D DMA writes only valid OW columns |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | FP32 selector | large stride-1 route |
| Kernel entry/router | kernel | `depthwise_conv_2d_vec_fp32` | AIV-only |
| Device body/helper | kernel | general `Process` FP32 branch | direct accumulator MTE3 |


### Route R-009（P5）：BF16 K1 fixed plane4 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | `usePackedK1Route` | BF16 `K==1&&P>=64`，BF16 分支第一项 | fixed K1 entry |

#### Device 数据流

##### 函数级调用链

```text
launch_plane4_bf16_k1 -> entry
  -> VF_CALL<DepthwiseConv2dPlane4Simt<bfloat16_t,1>>(512)
  -> compile-time one-tap AccumulateFixed -> four guarded stores
```

##### 源码等价伪码

```text
同 R-008 的 plane/thread 映射，但 effectiveKernelH/W=1，递归在编译期止于唯一 tap。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM + registers | 同 R-008 | 4 FP32 accumulators | BF16/FP32 | SIMT | SIMT/store | thread | no UB |

##### 分层内存布局

N/A：SIMT registers + GM direct。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| registers | thread | thread | next group | program order | unique output group |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| fixed tap | compile-time recursion | load | store | registers | BF16→FP32 one multiply-add→BF16；four-store tail guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | BF16 first branch | K1 priority |
| Kernel entry/router | kernel | `depthwise_conv_2d_plane4_bf16_k1` | fixed template 1 |
| Device body/helper | kernel | `DepthwiseConv2dPlane4AccumulateFixed<bfloat16_t,1,0>` | one-tap compile-time path |


### Route R-007（P6）：FP32 stride/gather K5 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | FP32 nested selector | `!F&&L&&K5&&strideH=strideW=2` | K5 gather entry |

#### Device 数据流

##### 函数级调用链

```text
DepthwiseConv2dStride2Gather<float,5>::Process
  -> base index -> padded FP32 plane -> DepthwiseStride2GatherVf<float,5>
  -> 25 Gather/Axpy -> FP32 output -> MTE3
```

##### 源码等价伪码

```text
同 R-006，但无 raw/cast buffer，Kernel=5，25 taps，input/output 都是 FP32。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| padded input | `padH*pitch` | `4*PI` | FP32, pitch align 8 | V+MTE2 | Gather | plane | per plane |
| index/output | 0 | each `4*O` | uint32/FP32 | scalar/VF | VF/MTE3 | kernel/plane | serial |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  input["FP32 input · 4PI"]:1
  index["u32 index · 4O"]:1
  output["FP32 output · 4O"]:1
```

##### 字节账与复用顺序

```text
PI=(IH+2*padH)*align_up(IW+2*padW,8)
UB_total=4*PI+8*O bytes; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output | V/MTE2/VF | Gather/MTE3 | next plane | hard events | dynamic allocations fit UB |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| whole route | DSB + hard events | index/V/MTE2/VF | VF/MTE3/next plane | UB | FP32; mask tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | FP32 nested K5-s2 | selector priority |
| Kernel entry/router | kernel | `depthwise_conv_2d_stride2_gather_fp32_k5` | AIV-only `DepthwiseConv2dStride2Gather<float,5>` |
| Device body/helper | kernel | Gather body | one padded plane, 25 tap recursion |


### Route R-010（P7）：通用 scalar SIMT fallback

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | each dtype final `else` | 未命中此前 route | dtype scalar entry；终点 |

Host 谓词只决定 selector 可达性，不证明整个可达域数值安全。Device 的 `linear` 与 `globalStride` 都是 `uint32_t`，其中 `globalStride=56*512=28672`。为保证每个已执行迭代末尾的 `linear += globalStride` 不回绕，需要：

```text
totalOutput <= UINT32_MAX - globalStride + 1
             = 4,294,967,295 - 28,672 + 1
             = 4,294,938,624
```

当前 Host 只检查 `totalOutput<=UINT32_MAX`，因此 `4,294,938,625..4,294,967,295` 仍是 selector 可达的合法 Host 域，却可能在 Device 末次步进时回绕并重访较小下标。复用 R-010 必须增加上述显式 guard，或把 `linear/globalThread/globalStride` 与循环加法全部提升到 64 位；`linear<totalOutput` 本身不能防止加法回绕。

#### Device 数据流

##### 函数级调用链

```text
launch_depthwise_conv_2d_{fp16|fp32|bf16}
  -> dtype entry [56 blocks]
  -> VF_CALL<DepthwiseConv2dNchwSimt<T>>(512)
  -> per-thread grid-stride scalar stencil -> y
```

##### 源码等价伪码

```text
globalThread=block*512+thread; globalStride=56*512
for linear=globalThread; linear<totalOutput; linear+=globalStride:
  decode nc,oh,ow,c; acc=fp32(bias[c])
  for kh,kw in row-major order:
    xValue = in-bounds ? fp32(x[inputOffset]) : 0
    acc += xValue*fp32(weight[c,kh,kw])
  y[linear]=T(acc)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs | decoded NCHW offsets | one scalar per load | T | Host | SIMT | Torch | direct |
| thread register | one output | FP32 acc/x/weight | FP32 | SIMT | SIMT/store | thread | loop local |
| GM output | `linear` | one T | NCHW | thread | caller | Torch | unique index |

##### 分层内存布局

N/A：无显式 AIV UB/dynamic UB；SIMT registers + GM direct。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread acc | same thread | final store | next linear | program order | 仅在 u32 步进安全上界内，global grid-stride 才持续给出唯一 `linear` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | no barrier/atomic | thread | same thread | unique output | 安全上界内 FP32 accumulate；`linear<totalOutput` + u32 加法不回绕；2-D bounds；padding zero participates in multiply |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | dtype final `else` | complete fallback |
| Kernel entry/router | kernel | base dtype entries | AIV-only + 512 threads |
| Device body/helper | kernel | `DepthwiseConv2dNchwSimt<T>` | grid-stride、NCHW decode、IEEE padding语义 |


### Route R-004（P8）：B16 通用 tiled Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | FP16/BF16 dtype branch | 上表 Vector 条件 | vector entry |
| Device | kernel | `Process` | 非 K3，且 BF16 非 K5 | general body |

#### Device 数据流

##### 函数级调用链

```text
DepthwiseConv2dVector<T>::Process
  -> per tap: Duplicate(term=0) -> BuildTapDesc -> CopyTapIn -> Cast to FP32 term
  -> Axpy(acc,term,weight) -> CAST_RINT outputQueue -> DataCopyPad(y)
```

##### 源码等价伪码

```text
for plane/tile: acc=bias
  for tap in row-major K:
    term=0; derive valid rows/width and GM/local offsets
    if nonempty: 2-D DMA+pad into inputQueue; cast valid rectangle to FP32 term
    acc += term*fp32(weight[tap])
  cast acc to B16 outputQueue; DMA only OW bytes for each valid row
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inputQueue depth1 | local 0 | `8192*2=16384 B` | B16 valid rectangle | MTE2 | Cast | per tap | FreeTensor after cast |
| outputQueue depth1 | local 0 | `16384 B` | B16 padded rows | Cast | MTE3 | per tile | Free after DMA |
| acc/term TBuf | each local 0 | `2*8192*4=65536 B` | FP32 | V | V | per tile/tap | term reused each tap |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["VECIN d=1 · 16384 B"]:1
  acc["acc · 32768 B"]:1
  term["term · 32768 B"]:1
  outQ["VECOUT d=1 · 16384 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16384+16384+32768+32768=98304 B; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inputQueue | MTE2 | Cast | next tap | EnQue/DeQue/FreeTensor | one tap at a time |
| term | zero/Cast | Axpy | next tap | PIPE_V barriers | Axpy completes before zero |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| queue | TQue lifecycle | MTE2/Cast | Cast/MTE3 | B16 tiles | invalid rectangle remains zero |
| vector | `PipeBarrier<PIPE_V>` | Duplicate/Cast/Axpy | next op | acc/term | row-major FP32 accumulate; RINT output |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | dtype selector | B16 Vector route |
| Kernel entry/router | kernel | vector B16 entries | AIV-only templates |
| Device body/helper | kernel | `Process/BuildTapDesc/CopyTapIn` | valid rectangle、queue、FP32 term/acc |


### Route R-008（P9）：四输出 plane4 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | plugin | `usePlane4Route` branches | 上表三 dtype 条件且未命中特化 | dtype plane4 entry |

#### Device 数据流

##### 函数级调用链

```text
plane4 launcher -> dtype entry [56 blocks]
  -> Simt::VF_CALL<DepthwiseConv2dPlane4Simt<T,0>>(512 threads)
  -> DepthwiseConv2dPlane4AccumulateOne per kh/kw -> guarded four GM stores
```

##### 源码等价伪码

```text
derive groupsPerRow=ceil(OW/4), groupsPerPlane=OH*groupsPerRow
pack multiple small planes into one 512-thread block when groupsPerPlane<512
for assigned plane/group: initialize acc0..3=bias
  for kh,kw: load weight once; conditionally load four x positions; acc[j]+=xj*w
  store acc0 and guarded acc1..3
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/weight/bias | per plane/channel/tap | logical tensors | T | Host | SIMT loads | Torch | direct GM |
| thread registers | group | 4 FP32 acc + 4 values + scalar weight | FP32 | SIMT | SIMT | thread/group | no cross-thread sharing |
| GM y | `plane*O+oh*OW+ow0+j` | guarded 1 element | T | SIMT | consumer | Torch | no atomic/conflict |

##### 分层内存布局

N/A：该 Route 不分配 AIV `TQue/TBuf` 或 dynamic UB；实际替代路径是 SIMT thread-private registers + GM direct load/store。

##### 字节账与复用顺序

```text
UB_total=0 explicit bytes; dynamic_ub=0; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread registers | same thread | same thread | next group | program order | each group uniquely owned |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | N/A barrier/atomic | thread | same thread | unique outputs | FP32 row-major sum；`ow+j<OW` guard；padding zero still multiplies weight |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | plugin | plane4 branches | shape/dtype priority |
| Kernel entry/router | kernel | three plane4 entries | AIV-only + 512 SIMT threads |
| Device body/helper | kernel | `DepthwiseConv2dPlane4Simt<T,0>` | four-output sharing、multi-plane packing、tail guards |

## 算子特定优化

### Trick T-001：复用官方 NCHW SIMT grid-stride 调度并替换归约语义

#### 触发与源码依据

- 适用 Route：R-010；R-008/R-009 继承相同的 block/thread 基础。
- 触发条件：复杂 Vector/plane4 predicate 均不成立，或进入 plane4 的 SIMT 路线。
- 固定官方源码依据：[`pooling/avg_pool/op_kernel/arch35/avg_pool_simt.h:133-145`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_kernel/arch35/avg_pool_simt.h#L133-L145)

**官方源码节选（固定 Commit；省略号表示未展示的前后文）**：

```cpp
// ...
TYPE_T outSize = SimtTilingData->nDim * SimtTilingData->cDim * hOutDim * wOutDim;
for (DIV_T i = blockIdx.x * blockDim.x + threadIdx.x; i < outSize;
    i += gridDim.x * blockDim.x) {
    // ...
```

#### 机制与设计原因

官方实现把 NCHW 输出线性域映射到全局 SIMT thread 并以全 grid 步长扫尾。冻结交付件保持这个调度骨架、512 threads 和边界解码思想，但把平均归约替换成 `bias[c]+Σx*w`，并保留越界零值参与乘法。这样不需要 workspace、跨线程同步或 atomic；只有满足 `totalOutput<=4,294,938,624` 或改用 64 位循环索引时，才可证明 grid-stride tail 不回绕。

#### 迁移边界

- 适用：每个输出可由单线程独立完成的 NCHW stencil。
- 前提：`totalOutput<=4,294,938,624`，输出索引和地址乘法不溢出 Direct ABI 位宽；每线程寄存器可容纳标量累加。更大输出必须改用 64 位 grid-stride 或在 Host 拒绝。
- 失效：需要跨输出复用时应选择 R-001～R-009；需要跨线程归约则不能直接复用。

#### 错误做法

固定仓未提供可验证反例；交付件明确避免“padding 越界直接跳过乘法”，因为该改写会改变 `0*Inf/NaN` 语义。

### Trick T-002：预生成 UB base index，再以 tap offset 驱动 Gather

#### 触发与源码依据

- 适用 Route：R-006、R-007。
- 触发条件：Host 命中 FP16 K3 非 strideW1 大 plane，或 FP32 K5 stride2 大 plane。
- 固定官方源码依据：[`pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel.h:289-303`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel.h#L289-L303)

**官方源码节选（固定 Commit；省略号表示后续分支未展示）**：

```cpp
if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
    GenGatherIndexSingleRow<U>(wStride, indexLocal);
} else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
    GenGatherIndexMultiRow<U>(wFactorOut, wIn, hStride, wStride, indexLocal);
} else if constexpr (GATHER_MODE == GATHER_MULTI_BATCH) {
    // ...
```

#### 机制与设计原因

相关官方 AvgPool 源码按 stride/shape 生成不同 gather index。交付件将其 Direct 派生为每个输出位置一次性生成 `baseIndex=oh*strideH*inputRowPitch+ow*strideW`，每个 tap 只做 `baseIndex+tapOffset`，然后 `MicroAPI::Gather`。一个 padded plane 每 plane 只搬一次，输出向量在 9/25 次 Axpy 中驻留寄存器，避免逐 tap 重复 GM patch 搬运。

#### 迁移边界

- 适用：输入 plane 与 index/output 同时可放入 UB，且窗口地址可表达为 base+tapOffset。
- 前提：`PI/O` 满足 R-006/R-007 字节账，padding/stride/dilation 乘法不溢出 uint32。
- 失效：UB 不足或非固定 K3/K5 时改用 R-008/R-010 或设计显式 tile gather。

#### 错误做法

固定仓未提供可验证反例；交付件没有 Host UB-capacity fallback，因此把该路线泛化到任意大 plane 是不安全的。

### Trick T-003：按真实实现族条件分配 UB，寄存器融合完整 tap 序列

#### 触发与源码依据

- 适用 Route：R-001～R-005，尤其 R-001～R-003。
- 触发条件：K3/K5 inner selector 命中；否则使用 queue+acc+term 的通用布局。
- 固定官方源码依据：[`pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel_pad.h:103-109`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel_pad.h#L103-L109)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
pipe_->InitBuffer(tmpBuf_, tilingData_->inUbSize * sizeof(T));
pipe_->InitBuffer(indexBuf_, tilingData_->indiceUbSize);
```

#### 机制与设计原因

官方 padded NCHW 路线让 UB 容量由具体实现所需的 input/output/tmp/index 决定。交付件进一步把 `InitBuffer` 放进 K3/K5/general 三分支：K3 一次物化 9 个 patch，整段归约保持 FP32 register accumulator；BF16 K5 只保留 5 个 patch slot 并跨五组复用一个 FP32 acc TBuf；general 才分配 input/output queue 与 acc/term。该分配与 body 一一对应，避免 dormant buffer 同时占 UB。

#### 迁移边界

- 适用：Host/Device selector 完全一致且模板 K 固定的 stencil。
- 前提：分支条件在 `Init` 与 `Process` 必须相同；buffer 字节数满足平台 UB。
- 失效：K 或 dtype 改变时必须重算全部 buffer，不能只改 body。

#### 错误做法

固定仓未提供可验证反例；若 `Init` 选 general 布局而 `Process` 进入 packed body，会读取未分配 TBuf，这是迁移时必须禁止的状态。

## 模板与实现对比

| 重要性 | Route | 命中位置 | 模板与 dtype | 循环与分核 | Queue/TBuf/寄存器 | 搬运/计算形态 | tail/fallback |
|---|---|---|---|---|---|---|---|
| P0 | R-001 | FP16 #4 Device #1；BF16 #2 Device #1 | packed K3，FP16/BF16 | block→planes→tile→9 taps | 9 patch + out + register acc | 9 DMA + MicroAPI Axpy | mask；未命中内层去 R-004 |
| P1 | R-002 | FP32 #2 Device #1 | packed K3，FP32 | block→planes→tile→9 taps | 9 patch + register acc | 9 DMA + MicroAPI Axpy | mask；未命中内层去 R-005 |
| P2 | R-003 | BF16 #2 Device #2 | packed K5，BF16 | block→planes→tile→5 groups×5 taps | 5 patch + acc + out | slot 循环复用 + MicroAPI Axpy | group 0/4 控制；未命中去 R-004 |
| P3 | R-006 | FP16 #1 | gather `half,3` | block-stride planes→O vectors→9 taps | padded plane + index + out | whole-plane DMA + Gather | mask；未命中继续 FP16 #2 |
| P4 | R-005 | FP32 #2 Device fallback | general Vector，FP32 | block→planes→tile→K taps | queue + acc + term | per-tap DMA/Cast/Axpy | rectangle tail；入口内终止 |
| P5 | R-009 | BF16 #1 | plane4 `bfloat16_t,1` | block/thread→plane/group→1 tap | registers only | scalar GM SIMT×4 | store guard；未命中继续 BF16 #2 |
| P6 | R-007 | FP32 #3 Device #1 | gather `float,5` | block-stride planes→O vectors→25 taps | padded plane + index + out | whole-plane DMA + Gather | mask；未命中内层去 R-008 |
| P7 | R-010 | 三 dtype #5 | scalar SIMT `T` | global thread→linear→K taps | registers only | scalar GM SIMT×1 | linear guard + u32 步进上界；终端 fallback |
| P8 | R-004 | FP16 #4/BF16 #2 Device fallback | general Vector，B16 | block→planes→tile→K taps | queue + acc + term | per-tap DMA/Cast/Axpy | rectangle tail；入口内终止 |
| P9 | R-008 | FP16 #2；FP32 #3 Device fallback；BF16 #3 | plane4 `T,0` | block/thread→plane/group→K taps | 4 register accumulators | scalar GM SIMT×4 | store guard；未命中继续该 dtype 后项 |
## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 本 Direct 交付件固定源码不可见 | 本 Direct 交付件固定源码不可见 | 支持；目标为 Ascend 950，相关官方 `avg_pool` 构建声明 `ascend950`，Conv2DV2 有 ascend950 binary config。 |
| Host/Launcher | 未提供 | 未提供 | Direct selector 固定 56 AIV blocks、当前 stream、0 workspace。 |
| Kernel/资源 | 未提供 | 未提供 | 所有可达 entry `KERNEL_TYPE_AIV_ONLY`；Vector/MicroAPI 或 AIV-SIMT。 |

## 限制与失败边界

- 只支持连续 NCHW FP16/FP32/BF16，bias 必选，`groups=C`，weight 形状必须为 `[C,Kh,Kw]`。
- Host 只检查 `y.numel()<=UINT32_MAX`；R-010 的 `uint32_t linear += 28672` 还要求更紧的 `totalOutput<=UINT32_MAX-28672+1=4,294,938,624`。当前 `4,294,938,625..UINT32_MAX` 是 Host 可接受但 Device 可能回绕的域；复用时必须增加 Host guard 或将 grid-stride 索引提升到 64 位。`N/C/H/W/K/stride/dilation` 到 `uint32_t`、padding 到 `int32_t` 以及其余 Device 地址乘法也没有独立溢出 guard。
- R-001～R-007 没有 Host UB/单 slot 容量检查。R-001/R-003/R-004 要求 `align_up(OW,16)<=8192`；R-002/R-005 要求 `align_up(OW,8)<=4096/8192`（分别对应 packed/general slot）。当 `rowPitch>tileElements` 时，源码把 `rowsPerTile` 从 0 强制为 1，却仍令 `computeElements=rowPitch`，会越过已分配 slot，不能把 `max(1,...)` 当作安全 fallback。
- gather 还要求 R-006 的 `6*PI+6*O`、R-007 的 `4*PI+8*O` 连同 TPipe 元数据可放入可用 UB；`PI` 的 pitch 分别按 16/8 元素对齐。Host 没有 capacity fallback，分配或 DMA 失败不会转去 R-008/R-010。
- `DataCopyExtParams` 中 row count 窄化为 `uint16_t`，左右 padding 窄化为 `uint8_t`；超界 shape 没有 fallback。
- FP16 的 `useVectorRoute` 只看 `strideW==1`；BF16/FP32 还受大 plane guard。改变 selector 时必须同时审计 `BuildTapDesc`、UB 与 tail。
- `useCachedRoute=false`、`useFp32SpatialRoute=false`，相关 launcher/body 是 dormant 源码，不是 Host 可达实现事实；`DepthwiseConv2dFused` 与 fixed FP16 K3/FP32 K5 plane4 entries 同样未被冻结 Host selector 调用。
- 官方 Conv2DV2 相关源码是语义/平台参照，不是本 Direct body：它要求 workspace，并在 NCHW filter 路线选择 MIX AIC；不得误写为本卡可达节点或直接复用闭包。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conv/conv2d_v2/op_host/config/ascend950/conv2d_v2_binary.json` | Conv2DV2 NCHW FP16/BF16/FP32 entries | 相关官方 950 conv 语义、dtype/layout 来源；非 Direct body。 |
| 入口实现 | `conv/conv2d_v2/op_kernel/conv2d_v2.cpp` | `conv2dv2` | group 分支、workspace 前置与 AIC/MIX 边界。 |
| Device helper | `conv/conv2d_v2/op_kernel/arch35/conv2d_v2_group.h` | `GroupConv2d` | 相关官方 grouped-conv 依赖复杂度，说明未移植边界。 |
| Kernel router | `pooling/avg_pool/op_kernel/avg_pool_apt.cpp` | `avg_pool` | AIV-only arch35 router、NCHW SIMT/small-kernel lineage。 |
| SIMT body | `pooling/avg_pool/op_kernel/arch35/avg_pool_simt.h` | `AvgPoolNcSimtCompute` | 512-thread grid-stride NCHW stencil 来源。 |
| Vector body | `pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel.h` | `GenGatherIndex/Compute*` | NCHW UB/gather/tail 来源。 |
| Padded Vector body | `pooling/avg_pool/op_kernel/arch35/avg_pool_nchw_small_kernel_pad.h` | `Init/BaseCompute` | padding、条件 buffer、former/tail core 来源。 |
| 平台 config | `pooling/avg_pool/op_host/config/ascend950/avg_pool.json` | FP16/BF16/FP32 op entries | 相关官方 950 三 dtype 来源。 |
| 构建接线 | `pooling/avg_pool/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | `ascend950` 与 `arch35` 构建接线。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 两套 positional scalar ABI，无 tiling POD/workspace | 参数顺序、位宽；Vector ABI 不含 `strideW` | `DEPTHWISE_*_LAUNCH_ARGS`（交付件） |
| 调用顺序 | 校验→selector→单 launcher→单 AIV entry→body | 56 blocks、当前 stream、恰好一节点 | `depthwise_conv_2d_npu`（交付件） |
| workspace/内存 | workspace=0；各 Route 字节账如上 | dynamic/fixed UB 上界、depth、row pitch、owner | 各 `InitBuffer`（交付件） |
| 同步 | queue、PIPE_V、hard event、UB DSB 或 SIMT program order | producer→consumer 顺序不能删除 | 各 `Process`（交付件） |
| 精度与尾块 | FP32 row-major accumulate；Vector B16 为 `CAST_RINT`，SIMT B16 为 `static_cast<T>`；mask/store bounds | padding zero仍参与乘法；tap 顺序固定；不得假设 SIMT 隐含 rounding mode | VF/SIMT helpers（交付件） |
| 失败边界 | Host validation 与 dtype final fallback 完整 | 不可把 dead cached/spatial/fused body当可达 Route | selector + entry graph（交付件） |

- [Axpy](../../../../apis/simd_api/basic_api/reg_vector_compute/composite_compute/axpy.md)
- [DataSyncBarrier](../../../../apis/simd_api/basic_api/sync_control/intra_core_sync/data_sync_barrier_isasi.md)
