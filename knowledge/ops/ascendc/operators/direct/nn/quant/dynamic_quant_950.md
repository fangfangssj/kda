---
type: Operator
title: "DynamicQuant 950 Direct Invoke 算子知识"
description: "说明 cann_bench::dynamic_quant / dynamic_quant_npu 在 950 上的接口语义、Direct Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/op_host/config/ascend950/dynamic_quant_binary.json
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/op_host/dynamic_quant_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/op_kernel/arch35/dynamic_quant_arch35_tilingdata.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/op_kernel/dynamic_quant_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_full_load.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_quant/op_kernel/arch35/gelu_quant_base_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T19:31:39Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# DynamicQuant 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | PyTorch 注册名 `cann_bench::dynamic_quant(Tensor x)`；NPU 实现 `dynamic_quant_npu`；Direct launcher 为 `launch_dynamic_quant_half` / `launch_dynamic_quant_bfloat16`。 |
| 计算语义 | 对最后一维逐行做对称动态量化：`amax=max(1e-12, max(abs(fp32(x))))`，`scale=amax/127`，`y=RINT(fp16_odd(fp32(x)/scale))` 后写 `int8`。 |
| 输入输出 | 连续 ND `x`，rank 2..8，`float16`/`bfloat16`；`y` 与 `x` 同 shape、`int8`；`scale` 为 `x.shape[:-1]`、`float32`。 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；无 Cube/AIC、SIMT、NDDMA、通信或 Hybrid 节点。 |
| Kernel 路线 | 生产可达仅 P0：R-001 `cast2-full-load-db2`，包含 fp16 与 bf16 两个同构模板实例；源码还保留 3 个 dormant 家族。 |
| 核心资源 | GM `x/y/scale`；每 AIV core 的 depth=2 `VECIN inQueue`、`VECOUT outQueue/scaleQueue`；MicroAPI 寄存器；无算子 workspace。 |
| 来源闭包 | `direct_implementation_closed`：已从 Direct 注册/校验、平台查询、tiling、dtype launcher、Kernel entry、公共基类、适配 body、队列与构建接线闭合活动调用链。 |
| 证据状态 | 官方固定提交与 Direct 交付源码已闭合到活动调用链；特殊值指令语义与 launch 同步报错协议保持边界说明。 |

> 证据边界：Frontmatter 仅列固定提交的官方相关源码。正文凡称“官方相关实现”，指 `ops-nn` DynamicQuant/GeluQuant；凡称“交付件 Direct 事实”，指本卡审计的独立 Direct 实现。交付件不是官方 `aclnnDynamicQuant` 入口，也不能把官方全功能算子的可选 smooth、group、offset、int4/fp8 路线外推给它。

## 接口与语义

### 计算语义

令 `H=x.shape[-1]`，`N=x.numel()/H`，把所有前导维展平成 `N` 行。对第 `r` 行：

```text
v[r,c] = fp32(x[r,c])
amax[r] = max(1e-12, max_c(abs(v[r,c])))
scale[r] = amax[r] / 127
q32[r,c] = v[r,c] / scale[r]
y[r,c] = int8_rint(fp16_odd(q32[r,c]))
```

官方相关 README 给出的基础语义是 `scaleOut=row_max(abs(x))/dtypeMax`、`yOut=round(x/scaleOut)`；交付件 Direct 固定 `dtypeMax=127`、无 smooth、对称 int8，并把归约初值改为 `1e-12`。因此全零行得到非零 `scale=1e-12/127`，量化结果为 0。两段转换 `fp32 -> fp16(CAST_ODD) -> int8(CAST_RINT)` 是交付件适配事实，不能误称为官方 DynamicQuant full-load 原 body；官方相邻 GeluQuant 提供了相同转换组合的可迁移依据。

对有限值，按 `amax/127` 缩放后理论绝对值不超过 127，因此 body 不另发 `Clamp`。NaN/Inf 的具体转换结果受 MicroAPI 指令语义约束；固定交付源码没有对所有特殊值定义独立软件语义，本卡不外推。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 连续 ND，rank 2..8；最后一维 `H>0` | `float16`、`bfloat16` | 唯一输入；所有前导维展平为行。Direct wrapper 显式要求 contiguous。 |
| `y` | 与 `x` 同 shape/ND | `int8` | wrapper 分配；每行按上述 scale 量化，仅写逻辑 `H` 个元素。 |
| `scale` | `x.shape[:-1]`/ND | `float32` | 每行一个缩放值，线性长度为 `N`。 |
| 属性 | N/A | N/A | Direct schema 不暴露 `dst_type`、smooth、group、offset、per-channel、per-tensor、int4 或 fp8 属性。 |

### 入口与版本

入口类型是 **Direct Invoke**。`TORCH_LIBRARY_FRAGMENT` 声明 `cann_bench::dynamic_quant`，Meta 与 PrivateUse1 分别绑定 `dynamic_quant_meta`、`dynamic_quant_npu`；后者通过当前 NPU stream 调用 dtype launcher。它不是公开 ACLNN/API 卡，也没有 executor/workspace-size 查询。

官方相关 DynamicQuant README 明确列出 Ascend 950PR/950DT 支持；Ascend 950 binary config 同时含 fp16→int8 与 bf16→int8 组合。交付件 Direct 固定 CANN 9.1.0、950/Ascend 950（dav-3510），只收窄到无 smooth 的 per-token 对称 int8 子域。

## Host/Launcher 执行流程

### 参数校验与前处理

交付件 Direct 的顺序如下：

1. `dynamic_quant_meta` 检查 `2 <= rank <= 8`、`H>0`、dtype 为 fp16/bf16、`x.is_contiguous()`。
2. 分配 `y=empty_like(x, int8)`，删除 shape 最后一维后分配 fp32 `scale`。
3. `dynamic_quant_npu` 在 `x.device()` 的 `OptionalDeviceGuard` 下计算 `rowLen=H`、`rowNum=x.numel()/H`。
4. `calc_dynamic_quant_tiling` 再要求 `0 < rowNum,rowLen <= UINT32_MAX`；零前导维会在这里失败，而不是形成空 Tensor Kernel。
5. 查询平台 UB 字节数与 AIV core 数；空平台对象、`UB<=1024` 或 core 数为 0 均抛异常。
6. 取得当前 NPU stream，按输入 dtype 调用 half/bfloat16 launcher；lambda 交给 `OpCommand::RunOpApi("DynamicQuantDirect", ...)`。

没有 Contiguous/ViewCopy：非连续输入直接拒绝。Host 不读取 Tensor 数据值，也不执行量化数学。

### 参数、Workspace 与生命周期

记平台 AIV core 数为 `V`，UB 字节数为 `U`，`A=align_up(H,16)`，`B=align_up(H,32)`。下式是数学意图；源码中 `rowPerHeadCore` 具体写成 32-bit 的 `(rows+V-1U)/V`，其近上界溢出风险见“限制与失败边界”：

```text
headCoreNum    = N % V
rowPerHeadCore = ceil_div(N,V)
rowPerTailCore = floor(N/V)
coreNum        = max(min(V,N),1)
maxUseUbSize   = U - 1024
calcSize       = 11*A
calcDbSize     = 22*A
```

若 `calcSize > U-1024`，Direct 实现拒绝，不进入官方的大 shape 路线。否则 `useDoubleBuffer` 在两个资源分支都写 1：`calcDbSize>U-1024` 时先以 `floor((U-1024)/(11A))` 给出保守行数；否则以 `floor((U-1024)/(22A))` 给出保守行数。只有 `max(rowPerHeadCore,rowPerTailCore)*H >= 512*1024` 时，才从 255 向下搜索满足真实队列字节账的最大行数。最终每次 UB 行数还取 `min(ubAvail,255,本核行数)`。

交付件没有给 Device Kernel 传 workspace，`workspace_total=0`。`x` 由调用方拥有，`y/scale` 由 wrapper 创建并在返回 tuple 后由 PyTorch Tensor 生命周期管理；Kernel 在当前 stream 上异步消费/写入。框架 `OpCommand` 内部 bookkeeping 属于框架边界，不等同于算子 workspace。

### 启动图与失败传播

```text
cann_bench::dynamic_quant
  -> dynamic_quant_npu
    -> dynamic_quant_meta
    -> calc_dynamic_quant_tiling(N,H)
    -> dtype == fp16
         ? launch_dynamic_quant_half
         : launch_dynamic_quant_bfloat16
    -> dynamic_quant_{half|bfloat16}_cast2_db2_entry<<<coreNum,nullptr,current_stream>>>
         -> DynamicQuantRegbaseFullLoadCast2<xDtype,int8_t,false,2,true>
```

| 节点 | Host 启动函数 | Kernel entry | grid/blockDim | POD | 输入/输出 | workspace slice |
|---|---|---|---|---|---|---|
| K-001 | `launch_dynamic_quant_half` / `launch_dynamic_quant_bfloat16` | `dynamic_quant_half_cast2_db2_entry` / `dynamic_quant_bfloat16_cast2_db2_entry` | `coreNum` 个 AIV block；线程级 grid 不适用 | `DynamicQuantTilingDataArch35` 按值 | GM `x`→GM `y,scale` | N/A，0 B |

单 Kernel 无跨节点边。当前 stream 提供 Host 提交顺序；Device 内由 TQue 建立 MTE2→V→MTE3 依赖。tiling/平台检查抛异常时不 launch；launcher 本身无显式返回码检查，lambda 固定返回 0，因此固定源码没有进一步的 Kernel launch 错误传播协议。

## Route 与参数设计

### 分核与资源计算

设 block 索引为 `b`。若 `b<headCoreNum`，本核 `R_b=rowPerHeadCore`、`baseRow=b*R_b`；否则 `R_b=rowPerTailCore`、`baseRow=headCoreNum*rowPerHeadCore+(b-headCoreNum)*rowPerTailCore`。每核用其 `M_b=multiRowNum{Head|Tail}Core` 分块：`loopCnt=R_b/M_b`，`remainRow=R_b%M_b`。活动 block 中 `M_b>0`；当 `N<V` 时 tail 字段可为 0，但不会 launch tail block。

每个正常循环处理 `M_b` 行，尾循环处理 `remainRow` 行。GM 基址均由 `baseRow` 推导；x/y 行跨度为 `H`，scale 行跨度为 1。UB 为最大 `M_b` 预分配，尾循环复用相同 buffer，只缩短 `DataCopyPad.blockCount` 与有效输出长度。

### 生产可达 Route 重要性（高到低）

| 重要性 | Route | 状态 | Route 特征 | 适用范围 | 实现前提 |
|---:|---|---|---|---|---|
| P0 | R-001 `cast2-full-load-db2` | active | 单 AIV full-load；每核按行切分；三条 depth=2 Queue；fp32 行归约与除法后执行 `fp32 -> fp16(CAST_ODD) -> int8(CAST_RINT)` | 连续 ND、rank 2..8、fp16/bf16 输入、对称 per-token int8；最后一维整行可装入 UB 的 950 子域 | `0<N,H<=UINT32_MAX`；平台 AIV/UB 信息有效；`11*align16(H)<=U-1024`；Host producer 生成 `useDoubleBuffer=1`；POD、Queue depth 与 UB 字节账一致 |

P0 表示实现与复用时的优先级，不表示 Host 的判断先后。当前生产入口没有 conditional Route，只有 active 的 R-001；fp16 与 bf16 是该 Route 的两个互斥模板实例，不拆分为新 Route。源码中另有三个 dormant 家族，见下表；它们不参与 active Route 排序。

#### Dormant Route 清单与重新启用条件

| Route | 源码 selector/关键代码关系 | 生产链不可达原因 | 重新启用条件 |
|---|---|---|---|
| R-D01 `cast2-full-load-db1` | `launch_dynamic_quant_{half\|bfloat16}` 的 `config.useDoubleBuffer == 0` 分支，实例化 `DynamicQuantRegbaseFullLoadCast2<xDtype,int8_t,false,1,true>` | 唯一 Host producer `calc_dynamic_quant_tiling` 在 `calcDbSize>maxUseUbSize` 与其 `else` 分支中都写 `useDoubleBuffer=1`；注册 wrapper 没有第二个 config producer | 修改并证明 Host UB selector，使其在 DB1 字节账成立时产出 0；同时保持 dtype、POD、行分核和 current-stream launcher 链不变 |
| R-D02 `official-full-load-db2` | 两个 `dynamic_quant_{half\|bfloat16}_db2_entry` 实例化 `DynamicQuantRegbaseFullLoad<xDtype,int8_t,false,2,true>` | entry 虽参与编译，但两个 launcher 的非零分支只引用 `*_cast2_db2_entry`，注册表也没有直达节点 | 显式增加 DB2 launcher/selector 出边，并先处理原 body 的 scale 初值和三段 cast 与 Direct 语义差异；不能只凭符号存在恢复 |
| R-D03 `official-full-load-db1` | 两个 `dynamic_quant_{half\|bfloat16}_db1_entry` 实例化 `DynamicQuantRegbaseFullLoad<xDtype,int8_t,false,1,true>` | entry 虽参与编译，但两个 launcher 的零值分支只引用 `*_cast2_db1_entry`；活动 producer 还不会产出零 | 同时增加 DB1 producer 与 launcher 出边，重做 depth=1 UB 字节账，并处理原 body 与 Direct 的数值语义差异 |

```text
# dormant 代码关系；不是当前调用图的运行时 fallback
if producer_can_emit(useDoubleBuffer=0):
    dtype_launcher -> *_cast2_db1_entry -> DynamicQuantRegbaseFullLoadCast2<...,1,true>

if launcher_is_rewired_to_official_body:
    *_db2_entry -> DynamicQuantRegbaseFullLoad<...,2,true>  # R-D02
    *_db1_entry -> DynamicQuantRegbaseFullLoad<...,1,true>  # R-D03
```

### Host selector 顺序、短路与 fallback

以下顺序是实际 Host 执行顺序，独立于上面的重要性排序：

1. `dynamic_quant_meta` 依次约束 rank、`H`、dtype 与 contiguous，并分配 `y/scale`；任一条件失败即短路，不进入 tiling。
2. `calc_dynamic_quant_tiling` 检查 uint32 行域、平台对象、AIV/UB 数据与单行 full-load 上界；任一条件失败即抛异常，不启动 Kernel。
3. `dynamic_quant_npu` 先按 dtype 选择 half 或 bfloat16 launcher；前置校验已排除其它 dtype，因此此处两个分支互斥且完备。
4. 选定的 launcher 再判断 `config.useDoubleBuffer != 0`：非零进入 cast2 DB2 entry，零进入同 dtype 的 cast2 DB1 entry。活动 producer 在两个 UB 资源分支中都写 1，所以注册 wrapper 流程只到达 R-001 DB2；DB1 归入 R-D01 dormant，不是 R-001 失败后的运行时 fallback。
5. 官方原样 `DynamicQuantRegbaseFullLoad` 的 fp16/bf16 DB1/DB2 entry 同样编译在翻译单元内，但 launcher 不选择它们；单行 full-load 上界失败时也没有 large-shape Route 接管。

```text
validate rank/H/dtype/contiguous
  -> validate uint32 row domain and AIV/UB
  -> if 11*align16(H) > U-1024: fail
  -> calculate M_head/M_tail; useDoubleBuffer = 1
  -> dtype selector: half launcher | bfloat16 launcher
  -> DB selector: nonzero -> cast2 DB2 | zero -> dormant R-D01（仅替代 producer 可启用）
```

R-001 的 fallback 出边是封闭的：rank/H/dtype/contiguous、uint32 域、平台信息或单行 UB guard 任一失败都以 Host 异常结束；Kernel 启动后也没有转入 R-D01/R-D02/R-D03 的 Device 分支。R-D01 只是 launcher 中保留的零值分支，R-D02/R-D03 只有已编译符号，三者都不是 active Route 的失败接管路径。

### Route 执行矩阵（P0→低优先级）

| 重要性 | Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---:|---|---|---|---|---|---|---|
| P0 | R-001 `cast2-full-load-db2` | dtype selector + `useDoubleBuffer=1` | fp16/bf16、rank 2..8、contiguous、非空行域、uint32 N/H、950 AIV/UB 有效、单行满足 11A 上界；可靠复用还须满足下文窄整数边界 | 单节点 K-001 `AIV_VECTOR` | `DynamicQuantRegbaseFullLoadCast2<xDtype,int8_t,false,2,true>::Init/Process/LoopProcess/CopyIn/Compute/ComputeVF/ComputeScaleVF/ComputeYVF/CopyOut` | 三个 depth=2 TQue；Queue 同步；无 workspace | 行尾由 pad+mask+有效字节 CopyOut；注册 wrapper 无 Route fallback；R-D01 只有替代 producer 产出 0 后才可达 |

### 隐式性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 单行 full-load | `11*A <= U-1024` | 保留官方通用 full-load 单行上界；Direct 不携带切 H POD | R-001 | Host 抛异常 | 交付件 `calc_dynamic_quant_tiling`；官方相关 `DynamicQuantRegbaseTiling::CalculateTilingData` |
| DB 保守行数 | `22*A <= U-1024` 取 `floor((U-1024)/(22A))`，否则取 `floor((U-1024)/(11A))` | 官方通用估算包含可选分支，Direct 实例实际只有 x/y/scale 三队列 | R-001 | 仍为 R-001，降低每次行数 | `calc_dynamic_quant_tiling` |
| 大工作量精确扩行 | `max(R_head,R_tail)*H >= 512*1024` | 仅在足够大工作量时按真实 depth=2 队列字节从 255 回退搜索 | R-001 | 阈值不满足则保留官方保守行数 | `targetDbBytes` / `kExpandedRowsMinElementsPerCore` |
| 每 UB 行数 | `M<=255` | `indexRow` 为 `uint16_t`，DMA `blockCount` 也是 `uint16_t`；255 是更紧上限 | R-001 | clamp 到 255 | `kMaxRowsPerUb`、`ComputeVF` |
| 输入对齐 | `A=align16(H)` | B16 每行 32 B 对齐；`DataCopyPad` 补 `A-H` 个零 | R-001 | 同 Route pad | `DynamicQuantBase::InitParams`、`CopyIn` |
| 输出对齐 | `B=align32(H)` | int8 本地行按 32 B 容量分配；GM 只提交 H B | R-001 | 同 Route 有效字节 CopyOut | `InitParams`、`CopyOut` |
| scale 对齐 | `S=align8(M)` | fp32 scale Queue 容量按 32 B 对齐 | R-001 | 同 Route | `InitParams`、`InitAndSetBuffer` |
| dtype/platform | fp16/bf16；950/dav-3510 AIV | entry 仅实例化 B16 输入与 AIV-only body | R-001 | Host dtype 拒绝或平台数据失败 | `dynamic_quant_meta`、两个 cast2 DB2 entry |

### POD/ABI

`DynamicQuantTilingDataArch35` 的交付件头文件与固定官方版本字节一致。自然 ABI 下 `alignof=8`、`sizeof=248`；26 个 `uint32_t` 占前 104 B，17 个 `int64_t` 占后续 136 B，末尾 float 后补齐。`DynamicQuantLaunchConfig` 为 Host-only 包装：`tiling` offset 0/248 B，`useDoubleBuffer` offset 248/32-bit，`sizeof=256`；Device 只收到内部 tiling POD。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `DynamicQuantLaunchConfig` | `tiling` | struct/1984-bit | offset 0，size 248 | POD | `config{}` 后逐字段赋值 | launch 时按值传入 entry | 仅活动字段见下 |
| `DynamicQuantLaunchConfig` | `useDoubleBuffer` | `uint32_t`/32 | offset 248 | bool selector | 两个资源分支都赋 1 | dtype launcher 选择 DB2 | 实际入口恒非零 |
| `DynamicQuantTilingDataArch35` | `coreNum` | `uint32_t`/32 | offset 0 | AIV core | `max(min(V,N),1)` | `Process` 越界保护；launch grid | `1..min(V,N)` |
| 同上 | `rowLen` | `uint32_t`/32 | offset 4 | element/row | `H` | 对齐、DMA、VF 循环 | `H>0` 且单行 UB 可容纳 |
| 同上 | `headCoreNum` | `uint32_t`/32 | offset 8 | core | `N%V` | head/tail core 分支 | 可为 0 |
| 同上 | `rowPerHeadCore` | `uint32_t`/32 | offset 12 | row/core | `(rows+V-1U)/V` | head core 行数/GM 基址 | 活动 head core 时 >0；加法没有上溢检查 |
| 同上 | `rowPerTailCore` | `uint32_t`/32 | offset 16 | row/core | `floor(N/V)` | tail core 行数/GM 基址 | `N<V` 时为 0 且无 tail block |
| 同上 | `multiRowNumHeadCore` | `uint32_t`/32 | offset 20 | row/UB loop | `min(ubAvail,255,rowPerHeadCore)` | head `loopCnt/remainRow`、队列容量 | 活动 head core 时 >0 |
| 同上 | `multiRowNumTailCore` | `uint32_t`/32 | offset 24 | row/UB loop | `min(ubAvail,255,rowPerTailCore)` | tail `loopCnt/remainRow`、队列容量 | 无 tail block 时可为 0 |
| 同上 | `innerLoopEle` | `uint32_t`/32 | offset 28 | element | value-init 0 | `ParseTilingData` 复制；full-load 不读 | inactive |
| 同上 | `innerLoopTimes` | `uint32_t`/32 | offset 32 | loop | value-init 0 | 仅大 shape body 使用 | inactive |
| 同上 | `innerLoopTail` | `uint32_t`/32 | offset 36 | element | value-init 0 | 仅大 shape body 使用 | inactive |
| 同上 | `groupNum` | `uint32_t`/32 | offset 40 | group | value-init 0 | `ParseTilingData` 复制；模板无 group | inactive |
| 同上 | `alignGroupNum` | `uint32_t`/32 | offset 44 | group | value-init 0 | 公共基类复制；本 body 不读 | inactive |
| 同上 | `hasSmooth` | `uint32_t`/32 | offset 48 | bool | value-init 0 | 模板参数 `false` 已编译期裁剪 | inactive |
| 同上 | `unused` | `uint32_t`/32 | offset 52 | N/A | value-init 0 | 无 | reserved |
| 同上 | `ubSize` | `uint32_t`/32 | offset 56 | byte | value-init 0 | 本 body 无 | inactive；Host 使用独立 `uint64_t U` |
| 同上 | `sizeH` | `uint32_t`/32 | offset 60 | element | value-init 0 | 本 body 无 | inactive |
| 同上 | `sizeX` | `uint32_t`/32 | offset 64 | element | value-init 0 | 本 body 无 | inactive |
| 同上 | `sizeZOut` | `uint32_t`/32 | offset 68 | element | value-init 0 | 本 body 无 | inactive |
| 同上 | `sizeCopyRow` | `uint32_t`/32 | offset 72 | row | value-init 0 | 本 body 无 | inactive |
| 同上 | `numCopyRow` | `uint32_t`/32 | offset 76 | row | value-init 0 | 本 body 无 | inactive |
| 同上 | `numHeadCore` | `uint32_t`/32 | offset 80 | core | value-init 0 | 本 body 无 | inactive |
| 同上 | `numTailCore` | `uint32_t`/32 | offset 84 | core | value-init 0 | 本 body 无 | inactive |
| 同上 | `numHeadTimes` | `uint32_t`/32 | offset 88 | loop | value-init 0 | 本 body 无 | inactive |
| 同上 | `numTailTimes` | `uint32_t`/32 | offset 92 | loop | value-init 0 | 本 body 无 | inactive |
| 同上 | `numLastTailRow` | `uint32_t`/32 | offset 96 | row | value-init 0 | 本 body 无 | inactive |
| 同上 | `alignType` | `uint32_t`/32 | offset 100 | enum | value-init 0 | 本 body 无 | inactive |
| 同上 | `totalBatchLen` | `int64_t`/64 | offset 104 | batch | value-init 0 | per-channel body only | inactive |
| 同上 | `mLen` | `int64_t`/64 | offset 112 | element | value-init 0 | per-channel body only | inactive |
| 同上 | `mBlockSize` | `int64_t`/64 | offset 120 | element | value-init 0 | per-channel body only | inactive |
| 同上 | `mTailBlockSize` | `int64_t`/64 | offset 128 | element | value-init 0 | per-channel body only | inactive |
| 同上 | `mBlockNum` | `int64_t`/64 | offset 136 | block | value-init 0 | per-channel body only | inactive |
| 同上 | `nLen` | `int64_t`/64 | offset 144 | element | value-init 0 | per-channel body only | inactive |
| 同上 | `nBlockSize` | `int64_t`/64 | offset 152 | element | value-init 0 | per-channel body only | inactive |
| 同上 | `nTailBlockSize` | `int64_t`/64 | offset 160 | element | value-init 0 | per-channel body only | inactive |
| 同上 | `nBlockNum` | `int64_t`/64 | offset 168 | block | value-init 0 | per-channel body only | inactive |
| 同上 | `nBaseSize` | `int64_t`/64 | offset 176 | element | value-init 0 | per-channel body only | inactive |
| 同上 | `nBaseLoopNum` | `int64_t`/64 | offset 184 | loop | value-init 0 | per-channel body only | inactive |
| 同上 | `blockPerHead` | `int64_t`/64 | offset 192 | block/core | value-init 0 | per-channel body only | inactive |
| 同上 | `blockPerTail` | `int64_t`/64 | offset 200 | block/core | value-init 0 | per-channel body only | inactive |
| 同上 | `totalBlockNum` | `int64_t`/64 | offset 208 | block | value-init 0 | per-channel body only | inactive |
| 同上 | `batchBlockSize` | `int64_t`/64 | offset 216 | batch | value-init 0 | per-channel body only | inactive |
| 同上 | `batchTailBlockSize` | `int64_t`/64 | offset 224 | batch | value-init 0 | per-channel body only | inactive |
| 同上 | `batchBlockNum` | `int64_t`/64 | offset 232 | block | value-init 0 | per-channel body only | inactive |
| 同上 | `dstTypeMax` | `float`/32 | offset 240；struct 尾补齐到 248 | scalar | `32768.0f` | int8 specialization 使用常量 `1/127`，不读此值 | 对活动 Route inactive |

### Route 源码闭包（P0→低优先级）

| 重要性 | Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---:|---|---|---|---|---|---|---|
| P0 | R-001 `cast2-full-load-db2` | wrapper dtype + `useDoubleBuffer=1` | 交付件 `dynamic_quant_meta`、`dynamic_quant_npu`、`calc_dynamic_quant_tiling`；官方相关 `DynamicQuantRegbaseTiling::CalculateCoreNum/CalculateTilingData` | `DynamicQuantLaunchConfig`、`DynamicQuantTilingDataArch35` | 交付件两个 `*_cast2_db2_entry`；官方相关 router `dynamic_quant<useDb,quantMode,...>` | 交付件 `DynamicQuantRegbaseFullLoadCast2` + 官方字节一致 `DynamicQuantBase`；官方相关 `DynamicQuantRegbaseFullLoad` | 交付件 `register_direct_launch_op`；官方相关顶层 DynamicQuant CMake 子目录接线 |

## Kernel 实现

### 公共执行流程

两个 dtype 实例共享单节点图与同一个模板 body：entry 标记 `KERNEL_TYPE_AIV_ONLY`，栈上创建 `TPipe`，实例化 `DynamicQuantRegbaseFullLoadCast2<xDtype,int8_t,false,2,true>`，以 `smooth=nullptr, offset=nullptr, workspace=nullptr` 调 `Init`，再调 `Process`。

`Init` 复制活动 tiling 字段，按 block 选择 head/tail 行域，计算 `A/B/S`、循环与尾行，绑定 GM，创建三个 depth=2 Queue，并设置 `maxValue=1/127`。`Process` 对完整 UB 批次和尾批次顺序执行 `CopyIn -> Compute -> CopyOut`。`ComputeVF` 对批内每行先算 scale，再复用寄存器广播 scale 计算 y；scale 不落 GM 后重读。

### P0 · Route R-001：cast2-full-load-db2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Direct API | 交付件 `op_plugin/dynamic_quant_plugin.cpp` | `dynamic_quant_meta` | rank 2..8、`H>0`、fp16/bf16、contiguous | 通过后分配 y/scale；否则 TORCH_CHECK 失败 |
| Direct Host | 交付件 `op_kernel/dynamic_quant_kernel.cpp` | `calc_dynamic_quant_tiling` | uint32 行域、平台有效、`11*A<=U-1024` | 生成 core/M/POD；否则抛异常 |
| Launcher | 同上 | `launch_dynamic_quant_half/bfloat16` | `config.useDoubleBuffer != 0`；实际 producer 恒写 1 | 两个 cast2 DB2 entry 之一 |
| Router/entry | 同上 | `dynamic_quant_{half\|bfloat16}_cast2_db2_entry` | 模板 `xDtype` 随 Host dtype，`y=int8,false,2,true` | R-001 body |
| 官方相关 | `quant/dynamic_quant/op_kernel/dynamic_quant_apt.cpp` | `dynamic_quant<...>` | `quantMode==TPL_COMMON_FULL_LOAD` | 证明官方 full-load 对应类与 AIV-only 执行族；不是 Direct entry |

#### Device 数据流

##### 函数级调用链

```text
dynamic_quant_npu
  -> calc_dynamic_quant_tiling(N,H)
  -> launch_dynamic_quant_{half|bfloat16}(x,y,scale,config,current_stream)
    -> dynamic_quant_{half|bfloat16}_cast2_db2_entry(x,y,scale,tiling)
      -> DynamicQuantRegbaseFullLoadCast2<xDtype,int8_t,false,2,true>::Init
        -> ParseTilingData -> InitParams -> InitAndSetBuffer -> SetMaxValue
      -> Process
        -> LoopProcess(rows,loop)
          -> CopyIn
          -> Compute -> ComputeVF
             -> ComputeScaleVF -> DataCopyInputVF
             -> ComputeYVF     -> DataCopyInputVF
          -> CopyOut
```

##### 源码等价伪码

```text
PSEUDOCODE: A=align16(H), B=align32(H), S=align8(M)
b = GetBlockIdx()
if b >= coreNum: return
(baseRow,R,M) = head_or_tail_partition(b)
bind xGm=x+baseRow*H, yGm=y+baseRow*H, scaleGm=scale+baseRow
init depth2 queues: in[M*A,b16], out[M*B,i8], scale[S,fp32]

for loop in [0, R/M), then optional tail rows=R%M:
    inLocal = Alloc(inQueue)
    DataCopyPad(inLocal, xGm + loop*M*H,
                blockCount=rows, blockLen=H*2 bytes,
                srcStride=0, dstStride=0, rightPadding=A-H zeros)
    EnQue(inQueue)

    xLocal = DeQue(inQueue)
    yLocal = Alloc(outQueue); sLocal = Alloc(scaleQueue)
    for row in [0, rows):
        maxReg = 1e-12
        for j in [0, ceil(A/VL)):
            mask = UpdateMask(remaining=A-j*VL)
            v = CastFp32(LoadUnpackB16(xLocal[row*A+j*VL]), ZEROING, mask)
            maxReg = Max(Abs(v,mask), maxReg, ALL)
        amax = ReduceMax(maxReg, ALL)
        s = amax * (1/127)
        StoreUnaligned(sLocal+row, s)
        scaleReg = Duplicate(s)

        for j in [0, ceil(A/VL)):
            mask = UpdateMask(remaining=A-j*VL)
            v = CastFp32(LoadUnpackB16(xLocal[row*A+j*VL]), ZEROING, mask)
            q32 = Div(v, scaleReg, mask)
            q16 = Cast(q32, F32_TO_F16, NO_SAT, ZEROING, CAST_ODD, mask)
            q8  = Cast(q16, F16_TO_I8, NO_SAT, ZEROING, CAST_RINT, mask)
            StorePack4B32(yLocal+row*B+j*VL, q8, mask)
    EnQue(outQueue,yLocal); EnQue(scaleQueue,sLocal); Free(inQueue,xLocal)

    yLocal=DeQue(outQueue); sLocal=DeQue(scaleQueue)
    DataCopyPad(yGm+loop*M*H, yLocal,
                blockCount=rows, blockLen=H bytes, strides=0)
    DataCopyPad(scaleGm+loop*M, sLocal,
                blockCount=1, blockLen=rows*4 bytes, strides=0)
    Free(yLocal); Free(sLocal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `(baseRow+loop*M)*H` | 当前批 `rows*H`；本核共 `R*H` | xDtype/row-major ND | caller | `CopyIn` | caller；覆盖整个 async launch | contiguous；行无 GM padding |
| UB `inQueue` | slot base + `row*A` | depth 2 × `M*A*2` B | fp16/bf16 padded rows | `CopyIn` | `ComputeVF` | TPipe；Alloc→EnQue→DeQue→Free | A=16 元素对齐；尾 lane 填 0 |
| MicroAPI registers | `j*VL` lane | 固定寄存器集合；源码未暴露字节总数 | fp32/xDtype/fp16/int8 | `DataCopyInputVF`/scale reduction | scale/y pipeline | 单次 `__VEC_SCOPE__` | scale 寄存器直接广播给 y，不落 UB 中间区 |
| UB `outQueue` | slot base + `row*B` | depth 2 × `M*B` B | int8 padded rows | `ComputeYVF` | `CopyOut` | TPipe；Alloc→EnQue→DeQue→Free | B=32 元素对齐；GM 只写 H B |
| UB `scaleQueue` | slot base + `row` | depth 2 × `S*4` B | fp32 linear | `ComputeScaleVF` | `CopyOut` | TPipe；Alloc→EnQue→DeQue→Free | S=align8(M) |
| GM `y` | `(baseRow+loop*M)*H` | 当前批 `rows*H` B | int8/原 shape | `CopyOut` | caller downstream | returned Tensor | 无效 UB padding 不写 GM |
| GM `scale` | `baseRow+loop*M` | 当前批 `rows*4` B | fp32/前导 shape | `CopyOut` | caller downstream | returned Tensor | 用 unaligned store 聚合到 scale Queue 后一次 DMA |
| workspace | N/A | 0 B | N/A | N/A | N/A | N/A | 无跨节点中间区 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  reserve["Host 保留 · 1024 B（不由 InitBuffer 分配）"]:1
  inQ["inQueue · VECIN · depth=2 · 4*M*A B"]:2
  outQ["outQueue · VECOUT · depth=2 · 2*M*B B"]:1
  scaleQ["scaleQueue · VECOUT · depth=2 · 8*S B"]:1
```

`smoothQueue`、`offsetQueue` 被 `hasSmooth=false` 与 `isSymmetrical=true` 编译期裁剪；公共基类的 `tempScale/tempCastUb/fp32_buf_` 在该 full-load Init 链未调用 `InitBaseBuffer`，不计入活动 UB。

##### 字节账与复用顺序

```text
A = align_up(H,16) elements
B = align_up(H,32) elements
S = align_up(M,8) rows
inQueue    = 2 * M*A * sizeof(B16) = 4*M*A bytes
outQueue   = 2 * M*B * sizeof(int8) = 2*M*B bytes
scaleQueue = 2 * S   * sizeof(float)= 8*S bytes
UB_total   = 4*M*A + 2*M*B + 8*S
UB_total <= U - 1024                 # 精确扩行分支直接检查；保守分支由 11A/22A 上界覆盖
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inQueue` slot | `CopyIn/DataCopyPad` | `Compute` DeQue 后两次 VF 扫描 | `FreeTensor(xLocal)` 后的后续 Alloc | EnQue→DeQue + FreeTensor | 两次扫描都结束后才 Free |
| `outQueue` slot | `ComputeYVF` | `CopyOut/DataCopyPad` | CopyOut Free 后 | EnQue→DeQue + FreeTensor | MTE3 消费由 Queue 协议保护 |
| `scaleQueue` slot | `ComputeScaleVF` | `CopyOut` | CopyOut Free 后 | EnQue→DeQue + FreeTensor | unaligned register store `Post` 完成后 EnQue |
| `vregDupScale` | 当前行 `ComputeScaleVF` | 紧随其后的 `ComputeYVF` | 下一行 scale 归约 | 同一 `__VEC_SCOPE__` 程序序 | 每行必须先 scale 后 y |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `AllocTensor/DataCopyPad/EnQue` | MTE2 CopyIn | `DeQue` in `Compute` | input Queue slot | 每行只读 H 个 B16，补 A-H 个 0 |
| 归约 | MicroAPI mask + `Max/ReduceMax` | B16 unpack/cast | scale multiply | 当前行寄存器 | B16→fp32；ZEROING mask；epsilon 初值防零 scale |
| scale→y | 同一 V scope 程序序 | `ComputeScaleVF` | `ComputeYVF` | `vregDupScale` | fp32 除法；scale 不经 GM 往返 |
| y 转换 | 两次 MicroAPI `Cast` | fp32 quotient | packed int8 store | 当前 VF lane | `CAST_ODD` 后 `CAST_RINT`，均 NO_SAT/ZEROING |
| UB→GM | `EnQue/DeQue/DataCopyPad/FreeTensor` | VECOUT Queue | MTE3 CopyOut | y/scale Queue slot | y 仅写每行 H B；scale 仅写 rows×4 B |
| 批尾 | `remainRow` | `Process` | `LoopProcess(remainRow,loopCnt)` | 最后不足 M 行 | buffer 容量仍按 M，DMA blockCount/scale 长度按实际 rows |
| 行尾 | `UpdateMask` + GM 有效字节 | VF loop | packed store/CopyOut | 最后 VF 与 UB padding | padding 0 不改变 maxabs；GM 不提交 padding |
| 跨 Kernel | N/A | 单节点 | N/A | N/A | 当前 stream 仅有一次 Device launch |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Direct Host producer | 交付件 `op_plugin/dynamic_quant_plugin.cpp`、`op_kernel/dynamic_quant_kernel.cpp` | `dynamic_quant_npu`、`calc_dynamic_quant_tiling` | 校验、展平、平台查询、分核、精确队列行数和 dtype launch |
| 官方 Host 相关 | `quant/dynamic_quant/op_host/dynamic_quant_tiling_arch35.cpp` | `CalculateCoreNum`、`CalculateTilingData` | 官方相同的按最后维成行、core 均分、1024 B 保留与 7/11/14/22 倍资源估算 |
| Direct entry/router | 交付件 `op_kernel/dynamic_quant_kernel.cpp` | 两个 `*_cast2_db2_entry` | AIV-only；模板 `<xDtype,int8_t,false,2,true>` |
| 官方 entry 相关 | `quant/dynamic_quant/op_kernel/dynamic_quant_apt.cpp` | `dynamic_quant<...>` | `TPL_COMMON_FULL_LOAD` 到官方 full-load 类映射 |
| 公共 Device helper | `quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_base.h` | `ParseTilingData`、`InitParams` | POD 消费、head/tail 循环、A/B/S 对齐 |
| Direct Device body | 交付件 `op_kernel/dynamic_quant_regbase_full_load_cast2.h` | `InitAndSetBuffer/CopyIn/ComputeScaleVF/ComputeYVF/CopyOut` | depth2 三 Queue、epsilon seed、两段 cast、有效输出 tail |
| 官方 body 相关 | `quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_full_load.h` | `DynamicQuantRegbaseFullLoad` | 原始 full-load 数据流、Queue、MicroAPI 归约与搬运基线 |

## 算子特定优化

### Trick T-001：以模板 depth 固化 Queue 布局并裁剪可选缓冲

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：`hasSmooth=false`、`isSymmetrical=true`、`useBufferNum=2`、输出 int8。
- Golden 来源：[`dynamic_quant_regbase_full_load.h:115-125`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_full_load.h#L115-L125)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (isSymmetrical == false) {
    pPipe->InitBuffer(offsetQueue, useBufferNum, sizeFloatLen * sizeof(float));
}
pPipe->InitBuffer(inQueue, useBufferNum, lenMultiRow * sizeof(xDtype));
pPipe->InitBuffer(outQueue, useBufferNum, outLen * sizeof(yCopyDtype));
pPipe->InitBuffer(scaleQueue, useBufferNum, sizeFloatLen * sizeof(float));
```

#### 机制与设计原因

官方相关 full-load 用编译期布尔值消除 offset/smooth Queue，并让 `useBufferNum` 同时决定三个活动 Queue 的 depth。交付件 Direct 固定无 smooth、对称、DB2，因此 UB 只包含 `2×(B16 input + int8 output + fp32 per-row scale)`；这使 Host 能用真实 `4MA+2MB+8S` 做扩行，而无需为不可达可选分支付费。depth=2 给 MTE2/V/MTE3 提供两个轮换 slot；是否形成实际性能增益仍需真机测量。

#### 迁移边界

- 适用：语义能在编译期固定可选输入/输出，并且 Queue 生命周期仍为 CopyIn→Compute→CopyOut 的 AIV full-load Kernel。
- 前提：Host 的 `M/A/B/S` 与 Device `InitBuffer` 完全一致，depth 必须乘入 UB 总量。
- 失效：加入 smooth、非对称 offset、其它 y dtype 或不同 Queue 后，必须重算字节账；不能沿用当前精确扩行公式。

#### 错误做法

把逻辑有效长度 `M*H` 当作物理队列容量、漏乘 depth=2，或仍声称未分配的 smooth/offset Queue 占 UB，都会破坏 selector 与 Device 分配的一致性。固定官方源码没有给出独立反例实现。

### Trick T-002：scale 留在寄存器并采用官方相邻量化的两段 RINT 转换

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：输出 `int8`，量化商为 fp32，要求最终 `CAST_RINT`；交付件采用 `fp32 -> fp16(CAST_ODD) -> int8(CAST_RINT)`。
- Golden 来源：[`gelu_quant_base_regbase.h:280-285`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/gelu_quant/op_kernel/arch35/gelu_quant_base_regbase.h#L280-L285)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::Cast<half, float, castTraitF32ToF16>(vregHalf, vregInput, preg0);
if constexpr (roundMode == AscendC::RoundMode::CAST_RINT) {
    AscendC::MicroAPI::Cast<dstType, half, castTraitF16ToI8Rint>(vregY, vregHalf, preg0);
}
```

#### 机制与设计原因

官方相邻 GeluQuant 证明 950 regbase 路径存在 `CAST_ODD` 的 fp32→fp16 trait 与 `CAST_RINT` 的 fp16→int8 trait。交付件把这一组合迁移到 DynamicQuant 的量化商，并把刚算出的 `vregScale` Duplicate 到 `vregDupScale`，紧接着完成该行除法和转换；因此 scale 只为最终输出写入 scaleQueue，不需要先写 GM 再回读。相较官方 DynamicQuant 原 body 的 fp32→int16→fp16→int8 三段转换，交付件少一个寄存器转换阶段；这里只陈述指令链差异，不声称实测加速。

#### 迁移边界

- 适用：950 regbase 上 fp32 中间值量化到 int8，且数值契约允许 fp16 中间舍入。
- 前提：两条 CastTrait 必须保持 `NO_SAT + ZEROING`，第一段 `CAST_ODD`、第二段 `CAST_RINT`；mask 与 quotient 使用同一有效 lane。
- 失效：若契约要求直接 fp32→int8 的不同精确舍入、显式饱和或 bit-exact 复刻官方 DynamicQuant 三段链，必须替换该实现并重新验证。

#### 错误做法

只把第二段改成 `CAST_RINT`、却保留不匹配的第一段 round/mask，或忽略 fp16 中间舍入对边界值的影响，不能视为等价迁移。固定仓未提供可验证的性能反例。

## 模板与实现对比

| 对比维度 | R-001 fp16 实例 | R-001 bf16 实例 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | dtype==fp16 | dtype==bf16 | wrapper 已排除其它 dtype；两者同优先级、互斥 |
| 模板与 dtype | `<half,int8_t,false,2,true>` | `<bfloat16_t,int8_t,false,2,true>` | 仅 xDtype 不同 |
| 循环与分核 | N 行均分 AIV core，M 行一批 | 同左 | 由共同 POD 生产 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 三个 depth2 Queue；无活动 TBuf/dynamic UB/window | 同左 | B16 输入宽度相同，UB 公式一致 |
| 内存总占用与复用 | `4MA+2MB+8S`；scale register→y | 同左 | fp16/bf16 都为 2 B |
| 搬运/Vector/SIMT/NDDMA/通信 API | B16 unpack→fp32；MicroAPI reduce/div/cast；DataCopyPad | 同左，B16 cast 源类型为 bf16 | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue；fp32 scale/除法；ODD→RINT；pad/mask | 同左 | 同一 body 的类型实例化 |
| fallback/失败边界 | 无大 shape fallback | 无大 shape fallback | 单行 11A guard 失败即 Host 异常 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方相关 DynamicQuant README 支持；本 Direct 交付未闭合 | 官方相关 DynamicQuant README 支持；本 Direct 交付未闭合 | 官方相关 README 与 ascend950 binary config 支持；本卡 Direct 目标平台 |
| Host/Launcher | Direct 未提供目标证据 | Direct 未提供目标证据 | CANN 9.1.0 查询 AIV core/UB，当前 stream 单 Kernel launch |
| Kernel/资源 | Direct 未提供目标证据 | Direct 未提供目标证据 | dav-3510 regbase，AIV-only，depth2 三 Queue，无 workspace |

## 限制与失败边界

- Direct 只接受连续 fp16/bf16、rank 2..8、最后维非零；零前导维会导致 `rowNum==0` 并在 tiling 失败。
- `rowNum`、`rowLen` 只在入口检查为不超过 `UINT32_MAX`，但 Host 的 `(rows+V-1U)/V` 在 32-bit 无符号域计算，未检查 `rows+V-1` 上溢。可靠复用须额外满足 `N<=UINT32_MAX-(V-1)`，或把该式提升到 64-bit 后再窄化。
- Device 的 `baseRow` 是 `int32_t`，`lenHead/lenTail/lenMultiRow/lenGMMultiRow/outLen` 是 `uint32_t`，`loopCnt` 还由 `int32_t` 循环变量遍历；固定源码未显式检查行偏移、元素乘积和循环计数的窄化/溢出。可靠子域必须保证每核 GM offset、对齐后 Queue 元素数与循环数分别落入对应类型。
- `rowLen*sizeof(xDtype)` 和输出 DMA blockLen 使用 32-bit 字段；950 实际 full-load UB guard 会给出远小于该上限的 H，但移植到异常大 UB 平台时仍需重审窄化。
- 单行不满足 `11*align16(H)<=U-1024` 时直接失败；Direct 没有接入官方 large-shape、多核切 H、空 Tensor、MoE 或 per-channel Route。
- `useDoubleBuffer` 的实际 producer 恒写 1，DB1 entry 不在入口可达图中；不得将保留代码误当运行时 fallback。
- Direct 不暴露 smooth/group/offset/dst_type，也不支持 int4/fp8；官方相关源码存在这些能力不代表交付件支持。
- 两段 `CAST_ODD -> CAST_RINT` 与官方 DynamicQuant full-load 原三段转换不同；要求 bit-exact 官方 body 时不能直接复用。
- 特殊值的最终 int8 行为未由 Direct wrapper 单独定义；不得在无指令级验证时补写 NaN/Inf 软件语义。
- 固定源码没有显式 Kernel launch 返回码；`RunOpApi` lambda 固定返回 0。需要同步错误报告的集成必须另行补齐。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `quant/dynamic_quant/README.md` | 产品支持、公式、参数表 | 官方相关 950 支持与 DynamicQuant 基础语义 |
| 平台组合 | `quant/dynamic_quant/op_host/config/ascend950/dynamic_quant_binary.json` | fp16/bf16→int8 entries | 官方相关 950 dtype/layout 组合 |
| Host/Tiling | `quant/dynamic_quant/op_host/dynamic_quant_tiling_arch35.cpp` | `CalculateCoreNum`、`CalculateTilingData`、`RunFusionKernelTiling` | 官方相关分行、分核、UB 常量、Route 与 workspace 基线 |
| POD | `quant/dynamic_quant/op_kernel/arch35/dynamic_quant_arch35_tilingdata.h` | `DynamicQuantTilingDataArch35` | Direct 复用的 ABI 字段顺序与类型 |
| Kernel entry | `quant/dynamic_quant/op_kernel/dynamic_quant_apt.cpp` | `dynamic_quant<...>` | 官方相关 common full-load 到 AIV body 映射 |
| Device helper | `quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_base.h` | `ParseTilingData`、`InitParams` | Direct 字节一致公共 helper 的 POD/对齐/循环事实 |
| Kernel body | `quant/dynamic_quant/op_kernel/arch35/dynamic_quant_regbase_full_load.h` | `DynamicQuantRegbaseFullLoad` | Direct 适配所依据的官方 full-load Queue 与计算管线 |
| 转换机制 | `activation/gelu_quant/op_kernel/arch35/gelu_quant_base_regbase.h` | CastTrait、int8 cast 分支 | Direct 两段 ODD→RINT 的官方相邻机制依据 |
| 构建接线 | `quant/dynamic_quant/CMakeLists.txt` | `add_subdirectory` | 官方相关算子子树构建入口；Direct 自身由独立注册宏接线 |

每行只列 Frontmatter 中存在的固定提交官方文件。交付件 Direct 文件仅以 repo-relative 名称出现在正文，用于明确独立实现事实，不被伪造为官方 blob 来源。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 复用 `DynamicQuantTilingDataArch35` 字段顺序；活动字段仅 core/row/M | 248 B 布局、字段位宽、按值传参、inactive 字段置零 | `DynamicQuantTilingDataArch35`、`ParseTilingData` |
| 调用顺序 | Meta→tiling→dtype launcher→单 AIV entry→Init→Process | scale 必须先于同一行 y；不得把不可达 DB1 当 fallback | `dynamic_quant_npu`、`ComputeVF` |
| workspace/内存 | 三个 depth2 Queue，精确式 `4MA+2MB+8S` | 对齐 A/B/S、depth、1024 B 保留、owner/lifetime 不变 | `InitParams`、`InitAndSetBuffer` |
| 同步 | TQue EnQue/DeQue/Free 管理 MTE2/V/MTE3 | input 两次 VF 读取结束前不得 Free；输出 DMA 完成前不得覆盖 slot | `CopyIn`、`Compute`、`CopyOut` |
| 精度与尾块 | fp32 max/scale/div；epsilon seed；ODD→RINT；GM 仅写 H | mask ZEROING、scale=amax/127、padding 0、有效字节数不可改 | `ComputeScaleVF`、`ComputeYVF` |
| 失败边界 | Direct full-load 子域可独立复用 | rank/dtype/contiguous/uint32/UB guard；无 large-shape fallback | `dynamic_quant_meta`、`calc_dynamic_quant_tiling` |