---
type: Operator
title: "ForeachNorm 950 Direct Invoke 算子知识"
description: "说明 cann_bench::foreach_norm 在 950 上的 Direct Host 分组、scalar/dtype Route 选择、离散 ABI、单 Kernel 两阶段 AIV 归约及其资源与数值边界。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_host/foreach_norm_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_reduce_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_reduce_regbase_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_reduce_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/arch35/foreach_norm.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/arch35/foreach_norm_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T19:26:55Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachNorm 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::foreach_norm(Tensor[] x, float scalar) -> Tensor[]`；Meta 为 `foreach_norm_meta`，PrivateUse1 为 `foreach_norm_npu`。 |
| 计算语义 | 对列表中每个 Tensor 独立求 p-范数，每项输出一个同 dtype、同 device 的 0-D Tensor。 |
| 有效输入 | 非空 TensorList；所有项同 device、同 FP32/FP16/BF16 dtype 且 contiguous；单个 Tensor 可以为空。 |
| 生产 Route | 36 个 `active` 叶：3 dtype × 12 scalar mode；PrivateUse1 对合法输入必达其中一个叶。没有 `conditional` Route。 |
| dormant 实现 | 未被任何 Host launcher 调用的 runtime-mode helper，以及无 Host launcher 的三个独立 finalize entry。 |
| 启动拓扑 | Host 每四个 Tensor 分一组；每组在 current stream 启动一个 `AIV_VECTOR` Kernel；单 Kernel 内完成 partial 生成、全核同步与最终归约。 |
| 核心资源 | 每核两个 126464 B 静态输入槽和 32 B partial/output 区；每次逻辑调用持有 320×FP32=1280 B GM workspace。 |
| 证据边界 | 固定官方源码支撑 950/arch35 的接口、RegBase 两阶段机制与构建映射；schema、C ABI、分组、静态局部地址、扩展 mode 和 hard-event 协议是 Direct 转换/派生实现事实。 |

## 来源与转换关系

### 官方原生事实

固定 Commit 的 README、OpDef 与 Kernel CMake 支撑 Ascend 950、FP16/FP32/BF16 和 arch35 构建范围；官方 tiling/POD 与 RegBase body 支撑“按核产生 FP32 partial，经 `SyncAll` 后由输出核做第二阶段归约”的机制。官方 950 entry 对 scalar 的公开分流比本卡 Direct ABI 更窄，因此不能反向证明这里的全部 mode。

### Direct Invoke 适配

交付实现新增 `TORCH_LIBRARY_FRAGMENT`/Meta/PrivateUse1、current-stream 获取、每四项分组、按值参数 ABI、三个 dtype launcher 和 dtype×mode 符号矩阵。它不经过官方 OpDef、tiling registry、ListTensorDesc 或 ACLNN executor；Host 自己生成 `n/meta/numBlocks`，并用一个 FP32 Tensor 承载 workspace 生命周期。

### Device 派生补充路线

Device 保留官方“两阶段归约”思想，但改为固定静态 local 地址、显式 hard event、L2 allocation bypass、tensor-major packed partial 和同 Kernel Stage-2；同时把 mode 扩展为 `ZERO/ONE/TWO/POS_INF/NEG_INF/GENERAL/THREE/ONE_HALF/FOUR/FIVE/TWO_HALF/NEG_ONE`。这些是转换源码的可复核事实，不是官方原生 ABI 或逐字实现。

## 接口与数学语义

令列表第 `t` 项有 `n_t` 个元素，`a_i=abs(float32(x_t[i]))`。Host 以 C++ `double` 接收 scalar；Device 只接收收窄后的 `float p`。

| Host mode | 精确 Host 谓词 | Stage-1 元素变换与合并 | Stage-2 最终变换 |
|---|---|---|---|
| `POS_INF` | `isinf(scalar) && scalar>0` | `max(a_i)` | 恒等 |
| `NEG_INF` | `isinf(scalar) && scalar<=0` | `min(FLT_MAX, a_0, ..., a_{n-1})` | 恒等；B16 输出再做 `CAST_RINT` |
| `ZERO` | 非 Inf 且 `scalar==0` | `sum(x_i!=0)` | 恒等 |
| `ONE` | `scalar==1` | `sum(a_i)` | 恒等 |
| `TWO` | `scalar==2` | `sum(float32(x_i)^2)` | `sqrt(sum)` |
| `THREE` | `scalar==3` | `sum(a_i^3)` | `exp(ln(sum)/3)` |
| `ONE_HALF` | `scalar==1.5` | `sum(a_i*sqrt(a_i))` | `exp(ln(sum)/1.5)` |
| `FOUR` | `scalar==4` | `sum((x_i*x_i)^2)` | `sqrt(sqrt(sum))` |
| `FIVE` | `scalar==5` | `sum(a_i^5)` | `exp(ln(sum)/5)` |
| `TWO_HALF` | `scalar==2.5` | `sum(a_i^2*sqrt(a_i))` | `exp(ln(sum)/2.5)` |
| `NEG_ONE` | `scalar==-1` | `sum(1/a_i)` | `exp(-ln(sum))` |
| `GENERAL` | 上述谓词均未命中 | `sum(exp(p*ln(a_i)))` | `exp(ln(sum)/p)` |

NaN scalar 不命中任一等值谓词，进入 `GENERAL`。`+0.0` 与 `-0.0` 都进入 `ZERO`。有限 double 可能在收窄后成为 FP32 的 0 或 Inf，但 mode 仍由收窄前的 double 决定；因此 selector 与 Device 数学不能合并成一次 FP32 判断。

空 Tensor 不产生 partial，由分配给该输出的核直接写 0。`NEG_INF` 的 accumulator、无效 lane 和 Stage-2 neutral 都是有限 `FLT_MAX`，所以其真实结果是 `min(FLT_MAX, abs(x_i)...)`：FP32 输入全部为 ±Inf 时输出 `FLT_MAX`，不是 Inf；FP16/BF16 叶先得到该 FP32 结果，再以 `CAST_RINT` 降精度，最终 B16 表示遵循 cast 语义。其它路线没有 epsilon、有限性修正或零值保护；`NEG_ONE`、负阶 `GENERAL` 及超越函数边界按对应 Device 原语传播。

| 参数 | shape/layout | dtype | 约束与生命周期 |
|---|---|---|---|
| `x` | 非空 TensorList；每项 contiguous ND；单项可 `numel==0` | 列表统一 FP32/FP16/BF16 | 全部同 device；输入 Tensor 必须覆盖异步执行期。 |
| `scalar` | schema 属性，无默认值 | C++ double→Device FP32 | Host double 决定 mode，FP32 值仅供 Device 的通用幂/根。 |
| `y` | 每项 `torch::empty({}, tensor.options())` | 与对应输入相同 | 新分配，不与输入别名；每项仅一个 Stage-2 writer。 |
| `workspace` | 320 个连续元素 | FP32 | 每次逻辑调用唯一 owner；所有 group 在同一 stream 顺序复用。 |

## Host、selector 与 launch 闭包

### 校验和 producer 顺序

```text
reject empty list
  -> guard(x[0].device)
  -> validate dtype in {FP32,FP16,BF16}
  -> validate every item: same device, same dtype, contiguous
  -> allocate one scalar output per input
  -> allocate 320-element FP32 workspace
  -> scalar(double) -> mode
  -> split list into groups G<=4 and produce n/meta/numBlocks
  -> get current NPU stream
  -> RunOpApi callback: for each group, dtype launcher on that stream
```

Host 不读取输入数值。校验失败发生在任何 launch 之前。三个 launcher 返回 `void`，callback 在顺序发出所有 group 后返回 0；源码没有逐次把 Device launch 状态转换为 callback 错误码，也没有框架或 CPU 自动 fallback。

### scalar 与 dtype selector

```cpp
if (isinf(s)) mode = s > 0 ? POS_INF : NEG_INF;
else if (s == 0) mode = ZERO;
else if (s == 1) mode = ONE;
else if (s == 2) mode = TWO;
else if (s == 3) mode = THREE;
else if (s == 1.5) mode = ONE_HALF;
else if (s == 4) mode = FOUR;
else if (s == 5) mode = FIVE;
else if (s == 2.5) mode = TWO_HALF;
else if (s == -1) mode = NEG_ONE;
else mode = GENERAL;

dtype == FP32 ? launch_foreach_norm_float(...)
: dtype == FP16 ? launch_foreach_norm_half(...)
: launch_foreach_norm_bfloat16(...); // BF16 已由前置校验证明
```

launcher 的 mode branch 选择具体 symbol；未知 mode 也落到 `general`。每个 entry 内 `(void)mode`，实际算术由 symbol tag 对应的 `RunKernelMode<T,MODE>` 在编译期固定，不能依赖 entry 的 runtime `mode` 自行纠错。

### 分组、分核与 meta

对一个 group，令 `G∈[1,4]`、元素字节数 `s`、`E=sum(n_j)`、平台 AIV 核数 `C`：

```text
elementsPerCore = 4096 / s
b0 = clamp(ceil(E/elementsPerCore), 1, C)
B  = max(b0, G)                              // launch numBlocks
L  = ceil(E/B)                               // E>0；E==0 时 Host 不计算 meta
```

对非空 Tensor `j`，Host 在逻辑拼接坐标上生产：

```text
first_j = floor(prefix_j/L)
last_j  = floor((prefix_j+n_j-1)/L)
count_j = last_j-first_j+1
meta_j  = first_j | (count_j<<8) | (middleStart_j<<16)
middleStart_{j+1} = middleStart_j + count_j
```

空 Tensor 的 `meta` 保持 0。`ForeachNormLaunchGroup group{}` 使 inactive 槽的指针、长度和 meta 为零。`B>=G` 是 Stage-2 能为每个输出分配唯一 `blockIdx` 的必要条件；950 还需 `G<=C<=80`，否则全核同步、固定 workspace 和 8-bit block 字段不再闭合。

### 按值 ABI

Direct entry 没有序列化 POD；以下 ordinal 是 kernel 形参 ABI。Host 声明、launcher 与 entry 必须保持同序同位宽。

| ordinal | 字段 | 类型 | producer | Device consumer | 边界 |
|---:|---|---|---|---|---|
| 0～3 | `x0..x3` | `GM_ADDR` | group 输入地址 | `Init`→`x_[]` | inactive 为 null |
| 4～7 | `y0..y3` | `GM_ADDR` | 标量输出地址 | `OutputZero/FinishTensor` | inactive 为 null |
| 8～11 | `n0..n3` | int64 | `numel()` | 拼接区间与 Tensor 交集 | inactive 为 0 |
| 12 | `tensorCount` | int32 | `G` | segment loop、Stage-2 writer | 1～4 |
| 13～16 | `meta0..meta3` | uint32 | 8/8/16-bit packing | partial 位置与 Stage-2 slice | first/count<256 |
| 17 | `workspace` | `GM_ADDR` | 320-FP32 Tensor | Stage-1 写、Stage-2 读 | group 间顺序复用 |
| 18 | `mode` | int32 | scalar selector | entry 丢弃 | symbol tag 才是模板真值 |
| 19 | `p` | float | `static_cast<float>(scalar)` | 元素幂与最终根 | 已收窄 |

`numBlocks` 与 `stream` 只属于 Host launcher：前者成为 grid 并由 Device `GetBlockNum()` 恢复，后者接 current stream。workspace Tensor 被 callback 按值捕获；各 group 同 stream 顺序发出，所以前一 Kernel 完成 Stage-2 后，后一 Kernel 才能覆盖相同地址。

### 启动图

```text
cann_bench::foreach_norm / PrivateUse1
  -> validate + outputs + one 1280-B workspace
  -> scalar selector -> mode
  -> for group g=0..ceil(N/4)-1 on current stream
       -> dtype launcher -> dtype_tag_kernel<<<B,nullptr,stream>>>
          -> Stage-1: block flattened span -> per-Tensor FP32 partial
          -> MTE3 copy -> tensor-major packed workspace
          -> SyncAll<true,{PIPE_MTE3,PIPE_MTE2}>
          -> Stage-2: blockIdx<G -> reduce one Tensor slice -> y
```

列表长度大于四只增加同 Route 的 group 节点，不生成新 Route。Host 校验、静态资源或 metadata 前提失败都没有跨 Route 退路；只有 scalar selector 的最后一条边进入 `GENERAL`。

## Route 分类与重要性顺序

排序按“官方核心语义与低成本专门化→宽 scalar 覆盖→边界语义→补充代数专门化→不可达保留代码”。同一 mode 下的三个 dtype 叶由输入互斥选择，不人为声明 dtype 间优先级。

| 优先级 | family | 状态 | 覆盖与价值 | selector/fallback |
|---:|---|---|---|---|
| P0 | ONE/TWO | active | 官方 950 核心 mode；无超越函数或只需最终 sqrt | 精确等值命中；不命中继续 Host 链 |
| P0 | GENERAL | active terminal fallback | 覆盖全部未列举 scalar，包括 NaN | Host 链终点；无后继 Route |
| P1 | ZERO/POS_INF/NEG_INF | active boundary | 分别固定计数、以 0 为 neutral 的最大值、以有限 `FLT_MAX` 为 neutral 的最小值语义 | Inf 最先短路，ZERO 次之；不命中继续 |
| P2 | THREE/ONE_HALF/FOUR/FIVE/TWO_HALF/NEG_ONE | active specialized | 用乘法、sqrt 或 reciprocal 代替通用元素幂 | 精确等值命中；不命中继续到 GENERAL |
| P3 | runtime-mode/finalize 保留族 | dormant | 不在生产调用图中 | 无自动进入边；只可按复活条件重新接线 |

本实现没有 `conditional` Route：三个 Host-visible dtype launcher 都由 PrivateUse1 生产使用，36 个 mode symbol 都可由合法 scalar 达到；dormant 符号则完全没有 Host producer。

### 36 个 active 叶

下表的路径缩写均闭合为：`foreach_norm_npu` 生产 group/metadata→`launch_foreach_norm_<dtype>`→`foreach_norm_<dtype>_<tag>_kernel`→`RunKernelMode<T,MODE>`→`KernelForeachNorm<T,MODE>::Process`。`A(mode)` 指下一节的可落地 mode 伪码；所有叶共享后续的 GM/UB/workspace、同步与尾块闭包。

| Route | 状态/优先级 | 精确触发谓词 | launch→entry/template | mode delta 与 fallback |
|---|---|---|---|---|
| R-002 FP32 ONE | active/P0 | dtype=FP32，`scalar==1` | `float`→`float_one`→`<float,ONE>` | `A(ONE)`；不命中继续 TWO |
| R-014 FP16 ONE | active/P0 | dtype=FP16，`scalar==1` | `half`→`half_one`→`<half,ONE>` | `A(ONE)`；末端 RINT FP16 |
| R-026 BF16 ONE | active/P0 | dtype=BF16，`scalar==1` | `bfloat16`→`bfloat16_one`→`<bfloat16_t,ONE>` | `A(ONE)`；末端 RINT BF16 |
| R-003 FP32 TWO | active/P0 | dtype=FP32，`scalar==2` | `float`→`float_two`→`<float,TWO>` | `A(TWO)`；不命中继续 THREE |
| R-015 FP16 TWO | active/P0 | dtype=FP16，`scalar==2` | `half`→`half_two`→`<half,TWO>` | B16 `MulAddDst`；sqrt 后 RINT FP16 |
| R-027 BF16 TWO | active/P0 | dtype=BF16，`scalar==2` | `bfloat16`→`bfloat16_two`→`<bfloat16_t,TWO>` | B16 `MulAddDst`；sqrt 后 RINT BF16 |
| R-006 FP32 GENERAL | active/P0 | FP32；非 Inf 且不等于九个专门常量 | `float`→`float_general`→`<float,GENERAL>` | `A(GENERAL)`；terminal，无后继 |
| R-018 FP16 GENERAL | active/P0 | FP16；同一补集谓词 | `half`→`half_general`→`<half,GENERAL>` | FP32 超越函数，末端 RINT FP16；terminal |
| R-030 BF16 GENERAL | active/P0 | BF16；同一补集谓词 | `bfloat16`→`bfloat16_general`→`<bfloat16_t,GENERAL>` | FP32 超越函数，末端 RINT BF16；terminal |
| R-001 FP32 ZERO | active/P1 | dtype=FP32，非 Inf 且 `scalar==0` | `float`→`float_zero`→`<float,ZERO>` | `A(ZERO)`；不命中继续 ONE |
| R-013 FP16 ZERO | active/P1 | dtype=FP16，非 Inf 且 `scalar==0` | `half`→`half_zero`→`<half,ZERO>` | FP32 count 后 RINT FP16 |
| R-025 BF16 ZERO | active/P1 | dtype=BF16，非 Inf 且 `scalar==0` | `bfloat16`→`bfloat16_zero`→`<bfloat16_t,ZERO>` | FP32 count 后 RINT BF16 |
| R-004 FP32 POS_INF | active/P1 | dtype=FP32，`isinf(scalar)&&scalar>0` | `float`→`float_pos_inf`→`<float,POS_INF>` | `A(POS_INF)`；Inf selector 已短路 |
| R-016 FP16 POS_INF | active/P1 | dtype=FP16，同上 | `half`→`half_pos_inf`→`<half,POS_INF>` | FP32 max 后 RINT FP16 |
| R-028 BF16 POS_INF | active/P1 | dtype=BF16，同上 | `bfloat16`→`bfloat16_pos_inf`→`<bfloat16_t,POS_INF>` | FP32 max 后 RINT BF16 |
| R-005 FP32 NEG_INF | active/P1 | dtype=FP32，`isinf(scalar)&&scalar<=0` | `float`→`float_neg_inf`→`<float,NEG_INF>` | `A(NEG_INF)`；Inf selector 已短路 |
| R-017 FP16 NEG_INF | active/P1 | dtype=FP16，同上 | `half`→`half_neg_inf`→`<half,NEG_INF>` | FP32 `min(FLT_MAX,abs)` 后 RINT FP16 |
| R-029 BF16 NEG_INF | active/P1 | dtype=BF16，同上 | `bfloat16`→`bfloat16_neg_inf`→`<bfloat16_t,NEG_INF>` | FP32 `min(FLT_MAX,abs)` 后 RINT BF16 |
| R-007 FP32 THREE | active/P2 | dtype=FP32，`scalar==3` | `float`→`float_three`→`<float,THREE>` | `A(THREE)`；不命中继续 ONE_HALF |
| R-019 FP16 THREE | active/P2 | dtype=FP16，`scalar==3` | `half`→`half_three`→`<half,THREE>` | FP32 cube/root，末端 RINT FP16 |
| R-031 BF16 THREE | active/P2 | dtype=BF16，`scalar==3` | `bfloat16`→`bfloat16_three`→`<bfloat16_t,THREE>` | FP32 cube/root，末端 RINT BF16 |
| R-008 FP32 ONE_HALF | active/P2 | dtype=FP32，`scalar==1.5` | `float`→`float_one_half`→`<float,ONE_HALF>` | `A(ONE_HALF)`；不命中继续 FOUR |
| R-020 FP16 ONE_HALF | active/P2 | dtype=FP16，`scalar==1.5` | `half`→`half_one_half`→`<half,ONE_HALF>` | FP32 sqrt/root，末端 RINT FP16 |
| R-032 BF16 ONE_HALF | active/P2 | dtype=BF16，`scalar==1.5` | `bfloat16`→`bfloat16_one_half`→`<bfloat16_t,ONE_HALF>` | FP32 sqrt/root，末端 RINT BF16 |
| R-009 FP32 FOUR | active/P2 | dtype=FP32，`scalar==4` | `float`→`float_four`→`<float,FOUR>` | `A(FOUR)`；不命中继续 FIVE |
| R-021 FP16 FOUR | active/P2 | dtype=FP16，`scalar==4` | `half`→`half_four`→`<half,FOUR>` | 双 FP32 sqrt，末端 RINT FP16 |
| R-033 BF16 FOUR | active/P2 | dtype=BF16，`scalar==4` | `bfloat16`→`bfloat16_four`→`<bfloat16_t,FOUR>` | 双 FP32 sqrt，末端 RINT BF16 |
| R-010 FP32 FIVE | active/P2 | dtype=FP32，`scalar==5` | `float`→`float_five`→`<float,FIVE>` | `A(FIVE)`；不命中继续 TWO_HALF |
| R-022 FP16 FIVE | active/P2 | dtype=FP16，`scalar==5` | `half`→`half_five`→`<half,FIVE>` | FP32 fifth-root，末端 RINT FP16 |
| R-034 BF16 FIVE | active/P2 | dtype=BF16，`scalar==5` | `bfloat16`→`bfloat16_five`→`<bfloat16_t,FIVE>` | FP32 fifth-root，末端 RINT BF16 |
| R-011 FP32 TWO_HALF | active/P2 | dtype=FP32，`scalar==2.5` | `float`→`float_two_half`→`<float,TWO_HALF>` | `A(TWO_HALF)`；不命中继续 NEG_ONE |
| R-023 FP16 TWO_HALF | active/P2 | dtype=FP16，`scalar==2.5` | `half`→`half_two_half`→`<half,TWO_HALF>` | FP32 sqrt/root，末端 RINT FP16 |
| R-035 BF16 TWO_HALF | active/P2 | dtype=BF16，`scalar==2.5` | `bfloat16`→`bfloat16_two_half`→`<bfloat16_t,TWO_HALF>` | FP32 sqrt/root，末端 RINT BF16 |
| R-012 FP32 NEG_ONE | active/P2 | dtype=FP32，`scalar==-1` | `float`→`float_neg_one`→`<float,NEG_ONE>` | `A(NEG_ONE)`；不命中进入 GENERAL |
| R-024 FP16 NEG_ONE | active/P2 | dtype=FP16，`scalar==-1` | `half`→`half_neg_one`→`<half,NEG_ONE>` | FP32 reciprocal/root，末端 RINT FP16 |
| R-036 BF16 NEG_ONE | active/P2 | dtype=BF16，`scalar==-1` | `bfloat16`→`bfloat16_neg_one`→`<bfloat16_t,NEG_ONE>` | FP32 reciprocal/root，末端 RINT BF16 |

### mode 关键伪码

每个 active 叶至少落到下列一个编译期算术体；`reduce` 在 Stage-1 和 Stage-2 分别使用同类 `sum/max/min`，`root` 只在 Stage-2 执行。

```cpp
A(ZERO):     acc += select(x != 0, 1.0f, 0.0f);
A(ONE):      acc += abs(x);
A(TWO):      acc += x*x;                       root = sqrt(acc);
A(POS_INF):  acc  = max(acc, abs(x));
A(NEG_INF):  acc=FLT_MAX; acc=min(acc, select(valid,abs(x),FLT_MAX));
A(THREE):    a=abs(x); acc += (a*a)*a;          root = exp(log(acc)/3.0f);
A(ONE_HALF): a=abs(x); acc += a*sqrt(a);        root = exp(log(acc)/1.5f);
A(FOUR):     q=x*x; acc += q*q;                 root = sqrt(sqrt(acc));
A(FIVE):     a=abs(x); acc += (a*a)*(a*a)*a;    root = exp(log(acc)/5.0f);
A(TWO_HALF): a=abs(x); acc += (a*a)*sqrt(a);    root = exp(log(acc)/2.5f);
A(NEG_ONE):  acc += 1.0f/abs(x);                root = exp(-log(acc));
A(GENERAL):  acc += exp(p*log(abs(x)));          root = exp(log(acc)/p);
```

FP32 `TWO` 显式 `Mul+Add`，B16 `TWO` 使用 FP32 `MulAddDst`。其它 B16 叶也先经 `DIST_UNPACK_B16` 和 cast 得到 FP32 RegTensor；输入 dtype 只改变 load/store，mode 决定中间算术。

### dormant Route

| Route | 状态 | 生产链不可达原因 | 重新启用的最小条件 |
|---|---|---|---|
| D-001 runtime-mode `RunKernel<T>` | dormant | 没有 global entry、Host launcher 或调用者；其调用 `KernelForeachNorm::Init` 时也没有提供当前 active body 必需的四个 meta 参数 | 先补齐与 active ABI 一致的 metadata，再新增明确 entry/launcher/Host selector；定义它与 36 个专门 symbol 的优先关系并复核 binary 体积 |
| D-002 `*_finalize_kernel`/`KernelForeachNormFinalize` | dormant | 三个 global entry 参与编译，但没有 Host-visible launcher；该 body 保留一个 `TPipe` 和两个 depth=1 `TQue`，input/output Queue 容量分别为 512 B 与 32 B；active Stage-1 采用变长 tensor-major packed slice，而该 body 按 `tensor*partialBlockNum` 读取固定 stride | 新增 stage1-only producer、与 finalize 一致的 workspace 布局和容量、C ABI launcher、current-stream 第二 launch、对象生命周期与错误传播；补齐空 Tensor 语义，并证明 `partialBlockNum<=128` 的 input Queue 容量 |

Dormant 族不是 active Route 的错误退路，也不能因符号存在就视为 conditional。当前生产行为只有同 Kernel Stage-2；重新接入 D-002 会改变 launch 图和 workspace 契约。

## Device 执行闭包

### 核与任务映射

每个 block 处理逻辑拼接区间：

```text
begin = blockIdx * L
end   = min(E, begin+L)
for tensor j in [0,G):
    tensor_range = [prefix_j, prefix_j+n_j)
    overlap = intersection([begin,end), tensor_range)
    if overlap nonempty:
        ProcessSegment(j, overlap.begin-prefix_j, overlap.length,
                       packedOffset=touchedTensorCount)
```

一个 block 最多触达四个相邻 Tensor。每个触达 Tensor 先在 local partial 的连续槽生成一个 FP32 值；随后从首个触达 Tensor 的 `middleStart + blockIdx-firstBlock` 开始，一次 MTE3 copy 写入 `touchedTensorCount` 个连续 FP32。这个公式依赖 Host 的 tensor-major `middleStart` 与相邻 Tensor 在边界 block 中的 partial 恰好连续，不能只传 first/count 而省略 start。

`SyncAll` 后，仅 `blockIdx<G` 的前 G 个 block 进入 Stage-2；block `j` 读取 `workspace[middleStart_j : middleStart_j+count_j]`，做同 mode 合并、最终 root 和 dtype store。一个输出只有一个 writer，不使用 atomic。

### 分层存储与数据流

| 层级/对象 | offset 与 extent | producer→consumer | owner/lifetime | 对齐、复用与边界 |
|---|---|---|---|---|
| input GM | 每个 Tensor 独立 base；segment 内 `offset+tile*tileElements` | caller→MTE2 `DataCopyPad` | 输入 Tensor 覆盖 launch | `CACHE_MODE_DISABLE`；不把不同 Tensor 当物理连续 |
| ping/pong local | addr 0 / 126464；各 126464 B | MTE2→Vector | 当前 block | 两基址 512 B 对齐；按 `inputSlot&1` 轮换 |
| partial/output local | addr 252928；32 B | Vector partial→MTE3；Stage-2 Vector→MTE3 output | 当前 block/当前阶段 | Stage-1 后同址复用为 T 输出；容量覆盖最多四个 FP32 partial |
| workspace GM | `middleStart_j + blockIdx-firstBlock_j` | Stage-1 MTE3→Stage-2 MTE2 | 调用内 FP32 Tensor | 分配 320 FP32；group 间只允许同 stream 顺序覆盖 |
| output GM | `y_j[0]` | Stage-2 MTE3→caller | 输出 Tensor | 每 Tensor 唯一 writer；无输入别名 |

```text
UB_static/core = 126464 + 126464 + 32 = 252960 bytes
workspace/call = 4 * 80 * sizeof(float) = 1280 bytes
tileElements   = 126464 / sizeof(T)     // FP32 31616；B16 63232
```

36 个 active 叶不使用动态 `TPipe/TQue`、L1、L0、Cube、AIC、NDDMA、通信 buffer 或额外 workspace；D-002 dormant finalize 是例外，保留一个 `TPipe`、input/output 两个 depth=1 `TQue` 以及 512 B/32 B Queue。Host 没有查询并校验 active 叶的 252960 B 静态 local 需求，因此它是目标平台前提，而不是可回退条件。

### 同步和生命周期

| 顺序边 | primitive | 保护对象 | 下一次覆盖条件 |
|---|---|---|---|
| slot 初始可写 | `SetFlag<V_MTE2>(EVENT_ID0/1)` | ping/pong | 对应 `WaitFlag<V_MTE2>` 取得 token |
| GM→slot→Vector | `DataCopyPad`，`Set/Wait<MTE2_V>` | 当前 tile 的 DMA 完成 | Vector 开始消费后才进入算术 |
| Vector→slot free | `SetFlag<V_MTE2>` | 当前 slot 的读取完成 | 后续同 slot MTE2 必须先 wait |
| local partial→workspace | `Set/Wait<V_MTE3>` 后 `DataCopyPad` | partial 的 Vector producer | 发出当前 block 的 MTE3 写 |
| Stage-1→Stage-2 | `SyncAll<true,{PIPE_MTE3,PIPE_MTE2}>` | 所有 block 的 FP32 workspace partial | 全部写对 Stage-2 MTE2 可见后才读 |
| workspace→Vector | `DataCopyPad`，`Set/Wait<MTE2_V>` | 当前 Tensor 的 partial slice | Stage-2 reduce 可开始 |
| Vector→output GM | `Set/Wait<V_MTE3>` 后 `DataCopyPad`，末端 `PipeBarrier<PIPE_ALL>` | 一个 T 标量 | Kernel 结束前完成 pipe ordering |
| group→下一 group | current stream 顺序 | 共享 workspace 与输出 owner | 前一 Kernel 完成后才覆盖 workspace |

hard event 只同步单 block 内的 DMA/Vector 资源；跨 block 的唯一同步点是 Stage-1 与 Stage-2 之间的 `SyncAll`。删掉这条边会让 Stage-2 读到尚未可见的 partial；把 group 改到并发 stream 则会造成 workspace 互相覆盖。

### 尾块与 mask

- segment 按 `current=min(remaining,tileElements)` 以真实 `current*sizeof(T)` 字节搬入；最后 tile 不要求 32 B 整除。
- `ComputeTile` 每次处理两个 64-lane FP32 repeat；完整部分用 `pairs=floor(current/128)`，剩余部分分别为 A/B 建 `UpdateMask<float>`。B16 也按 FP32 lane 归约。
- `ZERO` 用 `CompareNE+Select` 把无效 lane 变为 0；`NEG_INF` 显式把无效 lane Select 为 `FLT_MAX`。其它 mode 依赖 masked MicroAPI 只让有效 lane参与与 neutral 合并，不能改成无 mask 的整向量归约。
- Stage-2 以 `ceil(partialCount/64)` 个 repeat 读取，`UpdateMask<float>(remaining)` 生成最后一段有效 mask，再用 neutral 清理无效 lane。
- `E==0` 时 `B=G`、没有 Stage-1 partial；前 G 个 block 分别走 `OutputZero`。混合空/非空 Tensor 时空项不占 partial slice，但仍占一个 Stage-2 writer block。

### 精度顺序

所有 mode 的 accumulator、partial、workspace 和 Stage-2 reduce 都是 FP32。FP16/BF16 输入先提升到 FP32，最终 `StoreFromFloat` 以 `CAST_RINT` 降回原 dtype；FP32 直接 store。ZERO 的计数也先保存在 FP32，随后可能降为 B16，因此大计数不保证整数精确表示。

Stage-1 使用双 accumulator 合并 A/B repeat；跨 tile 通过 local 标量继续累积，跨 block 再在 Stage-2 合并。这个树形顺序改变浮点加法结合次序，不能承诺与串行实现或另一分核数 bitwise 相同。`FOUR` 的最终顺序固定为两次 FP32 sqrt；其它根路线固定为 `ln→muls(1/p)→exp`，不可在没有重新分析边界时换成另一幂函数。

## 资源 Guard、fallback 与未闭合边界

| Guard | 已有条件或证明 | 不满足时行为 | 影响范围 |
|---|---|---|---|
| 构建目标 | `ascend950 -> dav-3510`，Kernel 为 `__vector__ __schedmode__(1)` | 构建配置失败；无其它平台 Route | 全部 active/dormant 符号 |
| TensorList | 非空；统一 device/dtype；contiguous | `TORCH_CHECK`，不 launch | 36 active 叶 |
| group | Host 保证 `1<=G<=4` | 超过四项切下一 group，不是 Device fallback | 全部 active 叶 |
| AIV 核数 | 闭包要求 `G<=C<=80` | 源码无显式 guard；超界时 SyncAll、workspace/meta 证明失效 | 全部 active 叶 |
| 静态 local | 每核至少 252960 B 可用 VECCALC | Host 不查询；无小 tile fallback | 全部 active 叶 |
| metadata | first/count 各 8 bit，start 16 bit；在 `B<=80,G<=4` 下容量成立 | 无 checked pack；超界会截断 | 全部 active 叶 |
| workspace | 320 FP32；Device view 为 `4*B` | `B>80` 时 view/分配不一致；无重分配 | 全部 active 叶 |
| int64 算术 | `total/prefix/blockLength` 使用 int64 | `sum(n_j)` 无 checked-add；溢出后无恢复 | 全部 active 叶 |
| scalar 收窄 | mode 用 double，Device p 用 float | 无 FP64 Device Route；极端有限值可能改变 FP32 类别 | GENERAL 与根路线最敏感 |
| 数值域 | 不加 epsilon、有限性或 overflow 修正 | 结果按 FP32 原语传播；无数值 fallback | NEG_ONE/GENERAL 与超越函数路线 |

生产 fallback 图只有两类边：专门 scalar 谓词未命中时继续 Host selector，最终到 `GENERAL`；不支持的 dtype/layout/device、资源上界和 launch 前提失败时终止。不存在 active→dormant、AIV→CPU、Direct→ACLNN 或单 Kernel→独立 finalize 的自动边。

## 可复用实现要点

### 同 launch 的 tensor-major 两阶段归约

适用于输出数少、partial 运算可结合且 `blockDim>=outputCount` 的列表归约：Host 先给每个 Tensor 分配 packed slice，Stage-1 每个 block 写一到四个相邻 partial，`SyncAll` 后前 G 个 block 各完成一个输出。迁移时必须同时重推 workspace 容量、metadata 位宽、全核同步合法性和唯一 writer 映射；只复制 Device 循环而不复制 Host `middleStart` producer 会破坏地址闭包。

### 固定双槽与双 accumulator

单遍流式输入可用两个对齐 local 槽配合独立 hard event，将 MTE2 资源生命周期与 Vector 读取绑定；A/B 两个 accumulator 缩短同一寄存器的串行依赖。复用前要证明静态 local 容量、event pipe 语义和 tile byte 上界。更小 local 或不同事件协议的平台应重新 tiling，而不是沿用固定地址。

### Host 预选编译期 mode

对少量高价值 scalar，Host selector→launcher branch→tagged entry 可让热循环只保留目标算术。映射必须原子维护：Host enum、launcher branch、entry tag 与 `MODE` 模板值任一错位都会静默执行错误数学；entry 中被忽略的 runtime `mode` 无法补救。

## 平台与失败边界

| 项目 | A2/A3 | 950 |
|---|---|---|
| 官方范围 | 固定 README 说明相关产品支持，但不属于本卡 Direct 构建闭包 | 固定 README、OpDef 与 CMake 支撑 950/arch35 |
| Direct 构建 | 交付构建未提供对应目标接线 | `ascend950 -> dav-3510` |
| Direct 资源 | 未推导 | 静态 252960 B local、最多 80 AIV 的 workspace/meta 前提 |

- 列表可以超过四项，但所有 group 必须保持 current stream 顺序和 workspace owner；并发化需要每个并发 group 独立 workspace slice。
- scalar 的官方公开边界不能替代 Direct 的 12-mode 契约；反之，Direct 扩展也不能声明为官方接口能力。
- dormant helper 和 finalize entry 仅说明源码中保留了未接线实现，不证明可作为现成 fallback。
- 本卡完成源码静态闭包；未在本次改写中执行编译、设备数值或性能运行验证。

## 源码关系索引

Frontmatter 仅保留固定 Commit 的 HTTP 来源；下表区分官方锚点与 Direct 转换单元，不生成本机定位符。

| 链路阶段 | 官方固定来源中的锚点 | Direct 转换关键符号 | 支撑关系 |
|---|---|---|---|
| 接口/平台 | README、`ForeachNorm::ForeachNorm` | schema、Meta、PrivateUse1 | 官方 dtype/platform 范围→Direct dispatcher 适配 |
| Host selector | RegBase capability/tiling | `foreach_norm_npu` | 官方机制边界→Direct mode/group producer |
| metadata | `ForeachReduceTilingDataRegbase` | `ForeachNormLaunchGroup`、`meta` packing | 官方 partial 思路→Direct 离散 ABI |
| launch | arch35 `foreach_norm` entry | 三个 dtype launcher、36 个 tagged entry | 官方 model template→Direct dtype×mode 符号矩阵 |
| Device body | `ForeachNormNDRegbase` | `KernelForeachNorm<T,MODE>` | 官方两阶段 RegBase→Direct 静态 local/hard-event 派生 |
| 构建 | operator Kernel CMake | `register_direct_launch_op` | arch35/950 映射→Direct kernel/plugin 编译接线 |

复用检查应以 `Host producer→按值 ABI→launcher symbol→RunKernelMode→ProcessSegment/ComputeTile→workspace SyncAll→FinishTensor` 为最短闭包。任何只到宏 entry、只到公共模板名或只到官方 RegBase body 的描述，都不足以证明本卡 Direct Route。