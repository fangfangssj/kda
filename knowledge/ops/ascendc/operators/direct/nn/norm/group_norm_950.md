---
type: Operator
title: "GroupNorm 950 Direct Invoke 算子知识"
description: "说明 cann_bench::group_norm、calc_group_norm_tiling 与 launch_group_norm_direct 在 950 上的接口语义、Launcher 参数生成、七条 active/conditional Route、dormant 源码分支与单/多 Kernel 实现闭包。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm/docs/aclnnGroupNorm.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm/op_host/op_api/aclnn_group_norm.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_host/group_norm_silu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_host/group_norm_silu_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_host/group_norm_silu_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/group_norm_silu_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_welford.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_welford_generalized.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_two_pass.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_two_pass_generalized.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_host/config/ascend950/group_norm_silu_binary.json
    role: implementation
created_at: "2026-08-25T19:31:46Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# GroupNorm 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::group_norm` → `group_norm_npu` → `calc_group_norm_tiling` → `launch_group_norm_direct`；`Direct Invoke` |
| 计算语义 | 对每个 `(n,g)` 的 `R=(C/G)·HW` 个元素求 `mean` 与 `rstd=(variance+epsilon)^(-1/2)`，base 输出 `y=(x-mean)·rstd·gamma+beta`；补充 Route 按指定浮点顺序物化 `scale=rstd·gamma`、`bias=-mean·scale+beta`，再算 `y=x·scale+bias`。两种形式在 Inf/NaN 域不保证浮点等价，repair 正依赖该求值顺序 |
| 输入输出 | 连续 `x:(N,C,...)`、`gamma:(C)`、`beta:(C)`；同 dtype `float16/float32/bfloat16`；公开输出仅 `y:shape(x)` |
| 执行类型 | P0/R-002、P1/R-004、P4/R-005、P6/R-007 为单 `AIV_VECTOR`；P2/R-003、P5/R-006 为 `AIV_VECTOR→SIMT/InitGlobalMemory`；P3/R-001 为两个顺序 SIMT 节点。无可达 Cube/AIC 或 MIX_CV 节点。 |
| Kernel 路线 | P0 R-002 active Welford perf；P1 R-004 active two-pass perf；P2 R-003 conditional Welford+BF16 修复；P3 R-001 active FP32 大组并行；P4 R-005 active Welford generalized；P5 R-006 conditional generalized+修复；P6 R-007 active two-pass generalized |
| Dormant source | `GroupNormSiluEmpty`、`activateSilu=true` 与 nullable gamma/beta 分支没有完整生产边；它们不计入七条可达 Route，重新启用条件见后文 |
| 核心资源 | 官方适配 body：GM、整块 `TBuf` UB、MicroAPI `RegTensor` 与硬事件；补充路线：GM partials、1 KiB 每块静态 scratch 或线程寄存器；无显式 workspace |
| 来源闭包 | `direct_implementation_closed`：Torch Direct 入口、校验、POD、资源查询、7 条 active/conditional Route、全部 entry/body/helper、跨 Kernel GM 边、dormant 排除证据与构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `D=C/G`，把 channel 后的所有维度展平为 `HW`，每组元素数 `R=D·HW`。对 `q=n·G+g`：

```text
mean[q] = Σ_{d,h} x[n,gD+d,h] / R
var[q]  = Σ_{d,h} (x[n,gD+d,h]-mean[q])² / R
rstd[q] = 1 / sqrt(var[q] + epsilon)
y[n,gD+d,h] = (x[n,gD+d,h]-mean[q]) * rstd[q] * gamma[gD+d] + beta[gD+d]
```

四个官方适配 AIV body 设置 `activateSilu=0`，因此复用的是 `GroupNormSilu` 的 GroupNorm 分支。FP16/BF16 load 后在 MicroAPI 寄存器升为 FP32，统计量与归一化在 FP32 计算，最终以 `CAST_RINT` 回到输入 dtype；内部 `mean/rstd` 在 Direct 同 dtype路线也回写输入 dtype。

R-001 用 `sum(x)` 和 `sum(x²)` 合成 `variance=max(sumSq/R-mean²,0)`；R-003/R-006 对大 BF16 域追加修复：坏组填 BF16 quiet NaN，其他组只对 `gamma/beta` 非有限的 channel 按 `scale=rstd·gamma`、`bias=-mean·scale+beta` 重算。后两者是转换交付新增语义，不是官方源码 Route。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | contiguous `(N,C,...)`，rank `>=2` | FP16/FP32/BF16 | `C>0`；非空时 `N>0` 且 `HW>0` 由 tiling 检查；按 NCHW/ND 密集次序解释 |
| `gamma` | contiguous `(C)` | 与 `x` 相同 | 必选、一维、同设备；逐 channel scale |
| `beta` | contiguous `(C)` | 与 `x` 相同 | 必选、一维、同设备；逐 channel bias |
| `num_groups` | Host 标量 | `int64` | `>0` 且 `C % G == 0` |
| `epsilon` | Host 标量 | `double→float` | Direct 包装未检查正值/有限值；tiling 前窄化为 FP32 |
| `y` | `empty_like(x)` | 与 `x` 相同 | 唯一公开返回值；空输入直接返回，未启动 Device |
| `mean` | 通常 `(N·G)`；R-001 分配 2048 元素 | 与 `x` 相同 | 私有 GM 临时；R-001 作为两片 FP32 partials 复用 |
| `rstd` | `(N·G)` | 与 `x` 相同 | 私有 GM 临时；R-001 不使用 |

### 入口与版本

`TORCH_LIBRARY_FRAGMENT` 注册 `cann_bench::group_norm(Tensor x, Tensor gamma, Tensor beta, int num_groups, float epsilon) -> Tensor`，Meta 实现分配 `empty_like`，PrivateUse1 绑定 `group_norm_npu`。`OpCommand::RunOpApi("GroupNormDirect", launch)` 内的 lambda 直接把当前 stream 交给 `launch_group_norm_direct`，没有调用 ACLNN 数学算子。

本卡以最终交付中的 `csrc/ops/group_norm` 为 Direct 事实源。官方固定仓 `ops-nn@955f8b33985b6d72989ebff341bb9c2c564a51bc` 只负责转换映射：950 上公开 GroupNorm 由 `aclnn_group_norm.cpp` 调用 `GroupNormSilu(..., activate_silu=false)`；交付机械保留 184-byte RegBase POD、四个 key 及四个 arch35 body，并用运行时平台查询、by-value POD 与 typed entry 替换官方注册/opaque tiling/workspace 壳。Torch 包装、R-001 和 BF16 修复节点是交付新增，不能反推为官方接口或官方性能 Route。官方 README、OpDef 的 `ascend950` config 与 binary config共同证明被映射 body 的 950/950 来源；本 Direct 载荷目标版本为 CANN 9.1.0。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `group_norm_meta` 检查 rank、`G>0`、`C>0`、`C%G==0`、`gamma/beta==(C)`、三者同 dtype 且 dtype 属于 FP16/FP32/BF16，然后分配 `y`。
2. `group_norm_npu` 用 `OptionalDeviceGuard` 切换设备；空输入直接返回 `y`。非空输入要求三 tensor contiguous，且 affine 与 `x` 同设备。
3. 产生 `dtypeCode∈{0,1,2}`，取 `N=x.size(0)`、`C=x.size(1)`、`HW=numel/(N·C)`，调用 `calc_group_norm_tiling`；失败由 `TORCH_CHECK` 终止。
4. 通常分配同 dtype `mean/rstd` 各 `N·G` 元素。若 FP32、`N·G<=16` 且 `R>2^18`，`mean` 扩为 2048 元素，承载 `2·totalBlocks·64<=2048` 个 partials。注意这里的 `16` 是插件常量，而 launcher 用运行时 `reserved=A` 判断；在目标 950 的 `A=16` 上二者同域，迁移到 `A>16` 的平台前必须改成同一资源谓词。
5. 获取当前 NPU stream，按值捕获指针、POD 与 stream，在 `RunOpApi` 中执行 Direct launcher。

Direct 边界比官方 ACLNN 窄：不接受 optional affine、mixed affine dtype、非连续 tensor；空输入只返回未写 `y`，不复用官方 empty kernel 的 mean/rstd 语义。相反，Direct 未检查官方 ACLNN 的 `epsilon>0` 和 rank 上限。

### 参数、Workspace 与生命周期

平台查询产生 `UB=GetCoreMemSize(UB)`、`A=GetCoreNumAiv()`、`VLbytes=GetVecRegLen()`；950 映射的观测规格是 `UB=253952 B`、`A=16`、`VLbytes=256`。`totalGroups=N·G`：

```text
numPerCore = ceil(totalGroups/A)
realCoreNum = ceil(totalGroups/numPerCore)
numLastCore = totalGroups - numPerCore*(realCoreNum-1)
blockDim = realCoreNum
reserved = A
```

无显式 workspace；官方 body 的 `workspace` 参数被传 `nullptr` 且 body 不消费。`mean/rstd` 是插件分配的 GM 临时，不计 workspace。R-001 中 `mean` 在第一节点写 partials、第二节点只读；R-003/R-006 中 base 节点写 `mean/rstd/y`，repair 节点在同一 stream 上读取并可能覆盖 `y`。源码未显式记录 allocator stream 生命周期，完成期回收属于 `OpCommand/PyTorch` 框架边界。

### 启动图与失败传播

| 优先级 | Route | Host 顺序 | Host 启动函数 | Device 节点 | grid/threads | GM 边、ordering 与 owner |
|---|---|---|---|---|---|---|
| P0 | R-002 | H5→H6，H7=false | `launch_group_norm_direct` | key 1100 typed AIV entry | `blockDim=realCoreNum`；MicroAPI Vector | 单节点，`x/gamma/beta→y,mean,rstd` |
| P1 | R-004 | H2→H6 | 同上 | key 1110 typed AIV entry | 同上 | 单节点，无跨节点边 |
| P2 | R-003 | H5→H6→H7=true | 同上 | key 1100 AIV → repair | base `blockDim`；repair 每块 2048 threads | `y/mean/rstd` 为同 stream GM 边，repair 可覆盖 `y` |
| P3 | R-001 | H1，命中即 return | 同上 | `parallel_reduce_f32` → `parallel_normalize_f32` | `totalBlocks=(N·G)·floor(A/(N·G))`；各 2048 threads | `mean[0:2·totalBlocks·64)` partials；同 stream 顺序 |
| P4 | R-005 | H5→H6，H7=false | 同上 | key 1120 typed AIV entry | `blockDim=realCoreNum` | 单节点，affine 在 UB 按 tile 搬入 |
| P5 | R-006 | H5→H6→H7=true | 同上 | key 1120 AIV → repair | base `blockDim`；repair 每块 2048 threads | base 的 `y/mean/rstd` 供 repair 读取 |
| P6 | R-007 | H4→H6 | 同上 | key 1130 typed AIV entry | `blockDim=realCoreNum` | 单节点，无跨节点边 |

`calc_group_norm_tiling` 对参数、平台对象或资源为零返回 `-1`，插件会抛错并阻断 launch。Launcher 与 Kernel launch 均为 `void`，lambda 无条件返回 0，源码没有显式 enqueue 错误回传；多节点的第二节点不会由 Host 返回码条件性阻断，这是 Direct 失败传播边界。

## Route 与参数设计

### 分核与资源计算

记 dtype 字节 `s=4(FP32)` 或 `2(FP16/BF16)`，`alignB(v)=ceil(v/32)·32`，`alignE(v)=ceil(v/(32/s))·(32/s)`，`R=D·HW`，`M=min(2048,max(numPerCore,numLastCore))`。公共保留：

```text
gammaNddma = betaNddma = HW<=32 ? alignB(alignE(D*HW)*s)*G : alignB(C*s)
gammaOrdinary = betaOrdinary = alignB(C*s)
meanSize = rstdSize = alignB(M*4)
meanExtra = rstdExtra = (s!=4 ? alignB(M*s) : 0)
dichotomyExtra = alignB((dichotomyPower/(VLbytes/4))*4)
maxReduceCount = floor((UB-other)/(2*2*s))
```

`dichotomyPower` 取不超过 `R` 的最高 2 次幂（`R` 本身为 2 次幂时减半）；`dichotomyK/LastNum` 描述回刷 FP32 VL 后的二分折叠。Two-pass 的 `processSize=floor(maxReduceCount/alignE(R))·alignE(R)`。Welford 另由剩余 UB 求 32B 对齐的 `parallelN`，并产生 `loopNum/loopTail/innerLoopNum/innerLoopTail`：能装至少一行 `alignE(HW)` 时沿 D 分 tile，否则沿 HW 分 tile。

### Route 选择

下表按合法域覆盖、复用价值、性能价值与安全兜底价值从 P0 到 P6 排列；P 顺序只规定阅读和复用优先级，“Host 顺序”保留真实判断次序、短路点与 fallback。五条 `active` Route 都有独立生产 selector；两条 `conditional` Route 是 Welford base 后按 dtype/规模追加的语义修复图，不是独立 tiling key。

| 优先级 | Route | 状态 | 路线特征 | 适用域 | 实现前提 | Host 顺序、短路与 fallback | Key/模板 |
|---:|---|---|---|---|---|---|---|
| P0 | R-002 Welford perf | active fallback | 全 C affine 常驻，R 分段 Welford，D/HW 双缓冲归一化 | 非 R-001；不能 full-load；`C<=4096`；不追加 BF16 修复 | `parallelN>0`；全 C affine 与 stats 尾区可驻 UB；typed `<T,T>` | H5 选 1100→H6 启动；H7=false 后返回 | 1100 / `GroupNormSiluWelford<T,T>` |
| P1 | R-004 two-pass perf | active fast | R 全载后双遍统计；`HW<=32` 可用 NDDMA 展开 affine | 非 R-001；H2 的 strict full-load 成立 | `maxReduceCount>R`；X/Y 双缓冲、统计区、dichotomy 与 affine 同驻 UB | H2 成功即短路；失败后 H3 只重复同式、不能翻转结果，继续 H4 | 1110 / `GroupNormSiluTwoPass<T,T>` |
| P2 | R-003 Welford perf + BF16 修复 | conditional semantic | P0 base 后追加组级非有限值修复 | BF16、`C<=4096`、`R>2^20`，且 base key=1100 | base 必须先写 BF16 `y/mean/rstd`；repair 与 base 同 stream | H5 选1100→H6 base→H7 命中 repair；repair 后结束 | 1100 + `RepairNonFiniteBf16` |
| P3 | R-001 FP32 large-group parallel | active override | 每组拆 part；先写 warp partial，再合并并归一化 | FP32、`N·G<=A`、`R>2^18`；目标 950 上 `A=16` | `parts=floor(A/(N·G))>=1`；插件 scratch 扩容还要求 `N·G<=16`；两片 partial 总 extent不超过 2048 FP32 | H1 命中后双节点并立即 return；未命中→H2 | supplemental SIMT selector |
| P4 | R-005 Welford generalized | active capacity fallback | R 分段 Welford，gamma/beta 按 D 或 HW tile 搬入 | 非 R-001；不能 full-load；`C>4096`；不追加 BF16 修复 | Host 必须求得正 `parallelN/processSize`；affine tile 与 phase2 双缓冲可共驻 | H2 失败、H3 保持失败、H4 失败→H5 选1120→H6；H7=false 后返回 | 1120 / `GroupNormSiluWelfordGeneralized<T,T>` |
| P5 | R-006 Welford generalized + BF16 修复 | conditional semantic | P4 base 后追加与 P2 相同的修复节点 | BF16、`C>4096`、`R>2^20`，且 base key=1120 | P4 全部前提；base/repair 的 GM edge 与同 stream ordering 必须保留 | H5 选1120→H6 base→H7 命中 repair；repair 后结束 | 1120 + `RepairNonFiniteBf16` |
| P6 | R-007 two-pass generalized | active fast | affine 仅保留 D 大小切片，使 full R 可驻 UB | 非 R-001；首轮不全载；`C>4096 \|\| R<=VLbytes/4`；split 后 `newMaxReduceCount>R` | X/Y 双缓冲、stats、dichotomy 与两片 D affine 同驻；逐组 affine 事件完整 | H4 split-affine 成功即短路→H6；失败→H5 Welford fallback | 1130 / `GroupNormSiluTwoPassGeneralized<T,T>` |

```text
H0 plugin: if x.numel()==0 -> Host return, no Device Route
H1 launcher: if FP32 && totalGroups<=A && R>2^18 -> R-001; launch two nodes; return
H2 tiler: compute NDDMA-affine UB account
    if maxReduceCount>R -> key1110 / R-004; short-circuit tiling selection
H3 tiler: if HW<=32, recompute the same maxReduceCount from unchanged ubRemain
    H2 was false, so this identical predicate remains false; continue H4
H4 tiler: switch to ordinary affine account
    if C>4096 || R<=VLbytes/4:
        recompute with D-sized affine slices
        if newMaxReduceCount>R -> key1130 / R-007; short-circuit
H5 tiler fallback:
    if C>4096 -> key1120 / R-005 or R-006
    else      -> key1100 / R-002 or R-003
H6 launcher: launch dtype+key typed AIV base node
H7 launcher post-step:
    if BF16 && R>2^20 -> append repair; key1100 maps R-003, key1120 maps R-006
    else return the base Route
```

### 模板/实现执行矩阵

| 优先级 | Route | Host 顺序 | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---:|---|---|---|---|---|---|---|---|
| P0 | R-002 | H5→H6，H7=false | 1100 | partial load，`C<=4096`，非大 BF16 repair 域 | 单 AIV | `GroupNormSiluWelford<T,T>` | phase1/phase2 同一整 UB TBuf；ping-pong hard events | Welford non-align 三分支；D/HW tail |
| P1 | R-004 | H2→H6 | 1110 | H2 strict full-load | 单 AIV | `GroupNormSiluTwoPass<T,T>` | X/Y 双缓冲；小 HW 可 NDDMA 复制 affine | group/tile tail；失败经冗余 H3 继续 H4 |
| P2 | R-003 | H5→H6→H7 | 1100 + repair | R-002 base 域且 BF16、`R>2^20` | AIV → repair AIV/SIMT | Welford + `RepairNonFiniteBf16` | `y/mean/rstd` GM 边，同 stream | 坏组全 NaN；其余只重算异常 affine channel |
| P3 | R-001 | H1，return | supplemental FP32 | FP32，`N·G<=A`，`R>2^18`，950 | SIMT reduce → SIMT normalize | `ReduceLargeGroupF32`、`NormalizeLargeGroupF32` | GM partials；normalize 1 KiB scratch；stream 边 | channel/thread grid-stride tail；未命中去H2 |
| P4 | R-005 | H5→H6，H7=false | 1120 | partial load、`C>4096`，非大 BF16 repair 域 | 单 AIV | `GroupNormSiluWelfordGeneralized<T,T>` | affine 按 D/HW tile 搬入 | Welford + tile tail |
| P5 | R-006 | H5→H6→H7 | 1120 + repair | R-005 base 域且 BF16、`R>2^20` | AIV → repair AIV/SIMT | generalized Welford + repair | base affine 为切片；跨节点 GM edge | 同 P2 repair |
| P6 | R-007 | H4→H6 | 1130 | affine 切片后 full-load | 单 AIV | `GroupNormSiluTwoPassGeneralized<T,T>` | 每组搬 affine；X/Y 双缓冲 | tile tail；H4失败去H5 |

`H7` 的源码判断本身不检查 tiling key；但 `R>2^20` 时，仅 X/Y 双缓冲就远超 950 的 253952-byte UB，因而 key 1110/1130 的 strict full-load 谓词不可能成立。由此生产可达的 repair base 只会是 key 1100 或 1120；若迁移到 UB 规模显著不同的平台，必须重新枚举 H7 与全部四个 key 的组合，不能沿用这一 950 可达性结论。

### 关键分发与性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台资源 | `UB>0 && A>0 && VLbytes>4` | 分核、FP32 VL 与 UB 公式必需 | 全部 | `calc_group_norm_tiling=-1` | `group_norm_kernel.cpp::calc_group_norm_tiling` |
| strict full-load | `maxReduceCount>R`，不是 `>=` | X/Y 两路各 double-buffer 并保留统计/affine 区 | R-004/R-007 | Welford | `calc_group_norm_tiling`、官方 `SetTilingKey4Regbase` |
| NDDMA fold | `HW<=32` | 将 `(G,D)` affine 广播成 `(G,D,HW)` UB layout，便于 fold VF | R-004 | ordinary affine copy | `ProcessGammaAndBeta`、`CopyGammaAndBeta2UBByNDDMA` |
| generalized affine | `C>4096 \|\| R<=VLbytes/4` | 只驻留 `D` 或当前 rows 的 affine，释放 UB | R-005/R-007 | perf Welford | `SetWelfordGeneralizedTiling` |
| Welford parallel | `parallelN` 以 `32/s` 对齐且 UB 字节账成立 | `x ping/pong + tmpMean/tmpVar FP32 + dichotomy` 同驻 | R-002/R-003/R-005/R-006 | tiling 继续减小 `parallelN` | `SetWelfordParallelN` |
| supplemental FP32 | launcher：`N·G<=A && R>2^18`；插件扩容：`N·G<=16 && R>2^18` | 每组分配至少一 part；目标 `A=16` 时 `partials<=2048 FP32` | R-001 | key route | `group_norm_npu`、`launch_group_norm_direct` |
| BF16 repair | `R>2^20` | 限定非有限值求值次序修复的扫描域 | R-003/R-006 | 单 base 节点 | `launch_group_norm_direct` |
| dtype/entry | code 0/1/2 | typed entry 与 `<T,T>` ABI | 全部 | 参数校验失败；Launcher else 按 BF16，故必须先校验 | `group_norm_meta`、`LAUNCH_GROUP_NORM_ROUTE` |
| 空域 | `x.numel()==0` 在 tiling 前返回 | 避免 `HW=0`、`R=0` 下 `clz(0)` 与零 tile | Host terminal | 无 Device | `group_norm_npu` |

### POD/ABI

`GroupNormSiluRegbaseTilingData` 为 `sizeof=184, alignof=8`；`epsilon` 后有 4-byte padding。`GroupNormLaunchParams` 为 `sizeof=192, alignof=8`。以下均按交付自然布局闭合：

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| tiling | `numGroups` | `int64/64` | offset 0 | group | `numGroups` | group/channel 地址 | `>0` |
| tiling | `hwNum` | `int64/64` | offset 8 | 元素/channel | `numel/(N·C)` | DMA、D/HW loop | 非空时 `>0` |
| tiling | `elemNum` | `int64/64` | offset 16 | 元素/group | `shapeD*hwNum` | 核地址、归约、repair | 乘法未做溢出检查 |
| tiling | `shapeN` | `int64/64` | offset 24 | batch | `N` | supplemental totalGroups | 非空时 `>0` |
| tiling | `shapeC` | `int64/64` | offset 32 | channel | `C` | affine容量/large-channel | `>0` |
| tiling | `shapeD` | `int64/64` | offset 40 | channel/group | `C/G` | channel loop/地址 | 整除 |
| tiling | `realCoreNum` | `int64/64` | offset 48 | core | `ceil(totalGroups/numPerCore)` | body 不读；兼容字段 | blockDim 同值 |
| tiling | `numPerCore` | `int64/64` | offset 56 | group/core | `ceil(totalGroups/A)` | 非末核 work、GM base | 至多 totalGroups |
| tiling | `numLastCore` | `int64/64` | offset 64 | group | 余数式 | 末核 work | 正数 |
| tiling | `processSize` | `int64/64` | offset 72 | dtype 元素 | Two-pass/Welford tiling | UB tile、loop | 32B 元素对齐 |
| tiling | `loopNum` | `int64/64` | offset 80 | loop | Welford tiling | D 或 channel loop | Two-pass inactive/零 |
| tiling | `loopTail` | `int64/64` | offset 88 | 对齐元素 | Welford tiling | 最后 D tile rows | Two-pass inactive |
| tiling | `innerLoopNum` | `int64/64` | offset 96 | loop | Welford tiling | D/HW 分支 | Two-pass inactive |
| tiling | `innerLoopTail` | `int64/64` | offset 104 | 元素 | Welford HW split | 最后 HW tile | `innerLoopNum=1` 时 inactive |
| tiling | `tilingKey` | `int64/64` | offset 112 | key | 1100/1110/1120/1130 | Launcher router | body 不读 |
| tiling | `epsilon` | `float/32` | offset 120 | 标量 | `static_cast<float>` | rstd helper/SIMT normalize | 未校验正值/finite；124–127 padding |
| tiling | `activateSilu` | `int64/64` | offset 128 | bool | 常量 0 | normalization compile/runtime selector | Direct 固定 0 |
| tiling | `parallelN` | `int64/64` | offset 136 | 元素 | Welford tiling | phase1 segment | Two-pass inactive |
| tiling | `ubSize` | `int64/64` | offset 144 | byte | 平台 UB | `InitBuffer(innerBuf,ubSize)` | 必须适配目标平台 |
| tiling | `dichotomyAddPower` | `int64/64` | offset 152 | FP32 元素 | dichotomy producer | reduce helper | `R>0` |
| tiling | `dichotomyAddK` | `int64/64` | offset 160 | fold level | dichotomy producer | `DichotomyAdd` outer loop | 可为 0 |
| tiling | `dichotomyAddLastNum` | `int64/64` | offset 168 | FP32 lane | dichotomy producer | final mask | `<=VLbytes/4` |
| tiling | `powerOfTwoForReduce` | `int64/64` | offset 176 | 元素 | next power of two | Two-pass scale correction | Welford inactive |
| launch | `tiling` | POD/184B | offset 0 | ABI | `calc_group_norm_tiling` | typed entry by value | 全字段零初始化后写 |
| launch | `blockDim` | `uint32/32` | offset 184 | core | `realCoreNum` | launch grid/repair stride | 非零 |
| launch | `reserved` | `uint32/32` | offset 188 | AIV core | `A` | R-001 selector/parts | base routes inactive |

### Route 源码闭包

| 优先级 | Route | Host 顺序 | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---:|---|---|---|---|---|---|---|---|
| P0 | R-002 | H5→H6 | 1100 | `calc_group_norm_tiling` | `GroupNormLaunchParams` | `group_norm_1100_{f16,f32,bf16}` | `GroupNormSiluWelford<T,T>`、base helpers | `csrc/ops/group_norm/CMakeLists.txt::register_direct_launch_op` |
| P1 | R-004 | H2→H6 | 1110 | 同上 | 同上 | `group_norm_1110_*` | `GroupNormSiluTwoPass<T,T>`、base helpers | 同上 |
| P2 | R-003 | H5→H6→H7 | 1100+repair | 同上 + launcher post-step | 同上 | `group_norm_1100_bf16` → repair entry | Welford、base helpers、repair body | 同上 |
| P3 | R-001 | H1 | FP32 predicate | `launch_group_norm_direct` | 逐值标量；`mean` partials | `group_norm_parallel_{reduce,normalize}_f32` | `ReduceLargeGroupF32`、`NormalizeLargeGroupF32` | 同上 |
| P4 | R-005 | H5→H6 | 1120 | `calc_group_norm_tiling` | `GroupNormLaunchParams` | `group_norm_1120_*` | `GroupNormSiluWelfordGeneralized<T,T>`、base helpers | 同上 |
| P5 | R-006 | H5→H6→H7 | 1120+repair | 同上 + launcher post-step | 同上 | `group_norm_1120_bf16` → repair entry | generalized Welford、base helpers、repair | 同上 |
| P6 | R-007 | H4→H6 | 1130 | `calc_group_norm_tiling` | `GroupNormLaunchParams` | `group_norm_1130_*` | `GroupNormSiluTwoPassGeneralized<T,T>`、base helpers | 同上 |

### Dormant 源码与 Route 内条件分支

| 类别 | 已存在源码 | 生产链不可达证明 | 重新启用条件 |
|---|---|---|---|
| dormant empty Route | `group_norm_silu_regbase_empty_tensor.h::GroupNormSiluEmpty<T>` 可把 mean/rstd 置零，且 `group_norm_kernel.cpp` 包含该头文件 | `group_norm_npu` 在 `x.numel()==0` 时先行返回；`calc_group_norm_tiling` 又拒绝 `shapeN<=0 \|\| hwNum<=0`；typed entry 宏没有实例化 Empty，launcher 四个 key 也没有 Empty 分支 | 先定义空输入对公开 `y` 与私有 mean/rstd 的可观察语义，再同时增加 Host selector、合法空域 POD、typed entry 和 launcher；还需证明零 grid、Queue 容量和 `shapeN==0` 的终止条件，不能只删除 Host early-return |
| dormant SiLU mode | 公共 helper 与四个 class 都保留 `activateSilu=true` 的 normalize-and-SiLU body | `calc_group_norm_tiling` 无条件写 `td.activateSilu=0`，GroupNorm schema 也没有激活属性；没有任何 Host producer 能写 1 | 应作为不同算子语义重新定义接口/输出与 Host 属性，补 selector 和数值闭包；不得在 GroupNorm Route 内直接翻转 POD 位 |
| dormant nullable affine | 四个 class 的 `hasGamma/hasBeta`、nullable GM 与模板分支可省略 gamma 或 beta | Torch schema、Meta 与 NPU wrapper 都把 gamma/beta 定义为必选 Tensor，并在取指针前检查 shape/dtype/device；当前 launcher从不传 null | 修改 schema 为 optional，重做 Meta/Host 校验和 null pointer ABI，并让 Host UB 账按存在性扣除 affine；随后逐 Route 复核 NDDMA、generalized tile 和 repair 对空指针的分支 |
| dormant H3 promotion edge | `calc_group_norm_tiling` 在 `!reduceFullLoad && HW<=32` 时再次计算 `maxReduceCount` 并保留一次 key 1110 判断 | H3 使用的 `ubRemain`、dtypeSize、常量与 H2 完全相同，且只有 H2 谓词为 false 才进入；因此 H3 的成功边在当前源码中不可达，不是 R-004 的第二个 producer | 若未来要把它变成有效 retry，必须先改变 UB 账或 affine layout输入，再以新值重算并闭合对应 Device layout；仅保留相同算式不能形成新 Route |
| active Route 内 Device 条件 | R-003/R-006 的 `badGroup` 与异常 affine channel 判断 | H7 已生产并启动 repair，但 fill、重算或保持原 `y` 由 Device 数据决定；它不是第三条 repair Route | 保持同 stream base→repair GM 边、BF16 位级 exponent 判断和 `scale/bias` 顺序；扩展扫描域时须重新证明内部非有限 X 的检测充分性 |

Empty、SiLU 与 nullable-affine 源码不参与七条 active/conditional Route 的 POD、内存或同步验收。未知 key 落入 1130 是 launcher 的 ABI `else` 行为，不构成第八条安全 fallback；生产调用者只能传 `calc_group_norm_tiling` 产生的四个合法 key。

## Kernel 实现

### 公共执行流程

四个 key 的 typed entry 都先在 AIC core 返回，声明 `KERNEL_TYPE_AIV_ONLY`，实例化 `<half,half>`、`<float,float>` 或 `<bfloat16_t,bfloat16_t>`，依次调用 `Init(..., workspace=nullptr, &td)` 与 `Process()`。公共 helper 用 `DataCopyPad` 在 GM/UB 间按 `blockCount` 行搬运、右侧补零到 32B；MicroAPI 以 `UpdateMask` 屏蔽尾 lane。FP16/BF16 用 UNPACK/两路 even-odd cast 升 FP32，输出 `CAST_RINT` 后 PACK。归一化严格执行 `Sub→Mul(rstd)→Mul(gamma)→Add(beta)`。

所有官方适配节点只分配一个大小等于 `tiling.ubSize` 的 `TBuf innerBuf`，再用 `LocalTensor` 切片；没有 `TQue`。Two-pass 同一 tile先求 mean/rstd 再归一化；Welford phase1 分段更新并二分 finalize，phase2 复用 TBuf 下部做 X/Y ping-pong。`MTE2_V`、`V_MTE3`、`V_MTE2`、`MTE3_V` 事件保护前向数据与反向覆盖，跨 group batch 用 `MTE3_MTE2` 防止 mean/rstd 输出和下次复用冲突。

### Route R-002（P0）：Welford perf 单节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_kernel/group_norm_kernel.cpp` | `calc_group_norm_tiling` | 不能 full-load，`C<=4096` | key 1100 |
| Launcher | 同上 | `launch_group_norm_direct` | key 1100，且不满足 FP32 supplemental | dtype `group_norm_1100_*` |
| Entry/body | 同上 + `arch35/...welford.h` | typed entry；`GroupNormSiluWelford<T,T>` | BF16 `R<=2^20` 或非 BF16 | 单 AIV node |

#### Device 数据流

##### 函数级调用链

```text
group_norm_1100_<dtype> -> Welford::Init -> Process
  -> per group CalMeanAndRstdByWelford
     -> CopyX2UB -> VFWelfordParallelUpdate -> VFWelfordParallelFinalize
  -> ProcessGammaAndBeta once -> NormalizeAndSwishByChannel/ByHW
     -> CopyX2UB -> VFNormalizeAndSwishAlign(activateSilu=false) -> CopySilu2Gm
  -> ProcessMeanAndRstd -> GM
```

##### 源码等价伪码

```text
for assigned group q:
  for segment k of R by parallelN, ping/pong:
    xseg=GM[q*R+k*parallelN]; update FP32 lane-wise Welford(meanVec,M2Vec,count)
  finalize aligned or one of 3 non-aligned count-weighted folds
  rstd=high_precision_rsqrt(var+eps)
  for D/HW tiles, ping/pong:
    load x; broadcast mean/rstd and gamma/beta; y=(x-mean)*rstd*gamma+beta
    masked store valid tail
write same-dtype mean/rstd for q
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `innerBuf` | UB 0 | `ubSize` bytes | raw TBuf | `InitBuffer` | all phases | Kernel/core | 整块分配 |
| phase1 X | 0 / `parallelNAlign` | `2*parallelNAlign*s` | T ping/pong | MTE2 | Welford update | phase1 | 32B align |
| tmp mean/M2 | phase1 X 后 | 各 `alignE(parallelNAlign*4/s)*s` | FP32 | update | finalize | phase1 | in-place update |
| dichotomy | tmp M2 后 | Host `dichotomyExtra` | FP32 | ReduceSum回刷 | `DichotomyAdd` | finalize | 原地二分覆盖 |
| phase2 X/Y | UB 0 / `2*processSize*s` | 各 `2*processSize*s` | T ping/pong | MTE2/Vector | Vector/MTE3 | phase2 | 复用 phase1 下部 |
| mean/rstd + cast | UB 尾部 | 各 `alignB(M*4)`，B16另各`alignB(M*s)` | FP32/T | finalize/cast | normalize/MTE3 | batch | 尾部固定区 |
| gamma/beta | mean区后 | 各 `alignB(C*s)` | T | 首组 MTE2 | all groups/core | Process期间 | 全载一次 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  low["phase1 X ping/pong；phase2 复用为 X/Y ping/pong"]:3
  mid["tmpMean + tmpM2 + dichotomy"]:1
  stat["mean/rstd (+B16 cast)"]:1
  affine["gamma/beta 全 C"]:1
```

##### 字节账与复用顺序

```text
UB_total = ubSize（一个 TBuf）；所有切片和别名必须落在该上界
phase1_required = 2*parallelNAlign*s + 2*alignB(parallelNAlign*4) + dichotomyExtra
phase2_required = 4*processSize*s + mean/rstd/cast + 2*alignB(C*s)
dynamic_ub = N/A; workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| phase1/phase2 UB 下部 | Welford CopyX | update/finalize | normalize CopyX/Y | phase1函数返回及 `V_MTE2` fence | 两阶段不并发 |
| ping/pong X/Y | MTE2/Vector | Vector/MTE3 | 同 parity 的 k+2 | 对应正反 hard event | 覆盖前消费完成 |
| tmp mean/M2 | first segment | later update/finalize | 下一 group | `MTE3_MTE2` group序 | group statistics已输出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| segment load | `MTE2_V`/`V_MTE2` | DMA/Vector | update/next DMA | phase1 ping/pong | `welfordLoopTail` mask |
| finalize | `LocalMemBar` | Store to dichotomy | next fold load | dichotomy UB | aligned/3 non-align count分支 |
| normalize | four hard-event pairs | MTE2/Vector/MTE3 | next stage | phase2 ping/pong | `loopTail/innerLoopTail` |
| cast | FP32 + `CAST_RINT` | RegTensor | UB/GM | y,mean,rstd | B16 even/odd pack；padding不写 GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling producer | `op_kernel/group_norm_kernel.cpp` | `SetWelfordPerfTiling` | parallelN 与 D/HW tile |
| Kernel entry | 同上 | `group_norm_1100_{f16,f32,bf16}` | key+dtype 到 `<T,T>` |
| Device body | `arch35/group_norm_silu_regbase_welford.h`、`...base.h` | `CalMeanAndRstdByWelford`、`VFWelfordParallelFinalize`、normalize helpers | 分段 Welford、二分规约、双缓冲、tail |

### Route R-004（P1）：Two-pass perf 单节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_kernel/group_norm_kernel.cpp` | `calc_group_norm_tiling` | 首轮 `maxReduceCount>R` | key 1110 |
| Router | 同上 | `launch_group_norm_direct` | key 1110 | typed 1110 entry |
| Body | `arch35/...two_pass.h` | `GroupNormSiluTwoPass<T,T>` | full-load R | 单 AIV node |

#### Device 数据流

##### 函数级调用链

```text
group_norm_1110_<dtype> -> TwoPass::Init -> Process
  -> ProcessGammaAndBeta(ordinary or NDDMA)
  -> CopyX2UB -> CalMeanAndRstd(two passes/dichotomy)
  -> VFNormalizeAndSwishUnAlign/Fold(false) -> CopySilu2Gm
  -> ProcessMeanAndRstd
```

##### 源码等价伪码

```text
load affine once (HW<=32: NDDMA broadcast fold layout)
for group batches up to onceNumPerCore=processSize/alignE(R), ping/pong:
  load full aligned R per group
  pass1 mean via scaled ReduceSum+dichotomy
  pass2 squared deviation via ReduceSum+dichotomy; high-precision rstd
  normalize with unaligned VF or folded multi-group VF
  store only valid R; then store mean/rstd
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| X/Y ping-pong | 0 / `2*processSize*s` | 各 `2*processSize*s` | T | MTE2/Vector | reduce/normalize/MTE3 | tile | parity复用 |
| stats | X+Y 后 | mean/rstd各`alignB(M*4)` | FP32 | reduce | normalize/output | group batch | B16有额外cast区 |
| dichotomy | stats 后 | `dichotomyExtra` | FP32 | partial ReduceSum | fold | each statistic | 原地覆盖 |
| affine ordinary | tail | 各`alignB(C*s)` | T | first batch MTE2 | normalize | Kernel/core | 只装一次 |
| affine NDDMA | tail | 各`G*alignB(alignE(R)*s)` | `(G,D,HW)` broadcast | NDDMA | Fold VF | Kernel/core | `HW<=32` |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  x["X ping/pong · 2·processSize·s"]:2
  y["Y ping/pong · 2·processSize·s"]:2
  stat["mean/rstd+dichotomy"]:1
  affine["gamma/beta ordinary 或 NDDMA fold"]:1
```

##### 字节账与复用顺序

```text
UB_total=ubSize（整块 TBuf）
visible_required=4*processSize*s + mean/rstd/cast + dichotomyExtra + gamma/beta
dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y parity | CopyX/normalize | reduce/MTE3 | tile i+2 | MTE2_V,V_MTE3,V_MTE2,MTE3_V | depth=2 |
| dichotomy | mean reduction | final mean | variance partials | `LocalMemBar`/顺序 VF | mean已写独立 stats |
| affine | first `ProcessGammaAndBeta` | all normalize | Kernel结束 | MTE2_V fence | Direct gamma/beta必有 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| affine | MTE2_V | DMA/NDDMA | Vector | gamma/beta UB | NDDMA三维 stride复制 |
| tile pipe | hard events | MTE2/V/MTE3 | next stage | X/Y ping-pong | last `numPerCoreProcess` |
| reduce | mask+dichotomy | FP32 registers/UB | stats | padded X | `R<64` special VF，否则二分 |
| output | UnAlign/Fold | Vector | MTE3 | valid y | B16 even/odd，GM blockLen只含有效元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling producer | `op_kernel/group_norm_kernel.cpp` | `SetTwoPassTiling` | full-load batch与二分字段 |
| Kernel entry | 同上 | `group_norm_1110_*` | dtype typed body |
| Device body | `arch35/group_norm_silu_regbase_two_pass.h`、`...base.h` | `CalNormalize`、`ProcessGammaAndBeta`、`CalMeanAndRstd` | NDDMA fold、双遍规约、双缓冲、tail |

### Route R-003（P2）：Welford perf + BF16 修复

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Base selector | `op_kernel/group_norm_kernel.cpp` | `calc_group_norm_tiling` | key 1100、BF16 | 先执行 R-002 base body |
| Post selector | 同上 | `launch_group_norm_direct` | `dtype==BF16 && elemNum>2^20` | 追加 `group_norm_repair_nonfinite_bf16` |
| Repair body | 同上 | `RepairNonFiniteBf16` | 每组 Device 数据判断 | fill 或 selective recompute |

#### Device 数据流

##### 函数级调用链

```text
group_norm_1100_bf16 -> Welford body -> GM y/mean/rstd
  -> same-stream group_norm_repair_nonfinite_bf16
     -> badGroup ? InitGlobalMemory(qNaN) : asc_vf_call<RepairNonFiniteBf16>(2048)
```

##### 源码等价伪码

```text
run complete R-002
for group=blockIdx; group<totalGroups; group+=blockDim:
  bad = nonfinite(mean[group] or rstd[group] or x[first] or x[last])
  if bad: fill y[group*R:(group+1)*R] with BF16 0x7fc0
  else in 2048 threads:
    for each channel whose gamma or beta exponent is all-ones:
      scale=fp32(rstd)*fp32(gamma); bias=-fp32(mean)*scale+fp32(beta)
      recompute thread-strided spatial and cast BF16
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| base UB | 同 R-002 | `ubSize` | TBuf | base | base | first node | 不跨节点 |
| GM stats edge | `group` | `N*G` each | BF16 | base | repair | plugin | same stream |
| GM Y edge | `group*R` | `N*C*HW` | BF16 | base | repair/caller | caller | repair可原位覆盖 |
| repair registers | thread private | scale,bias,x | FP32 | repair | repair | one channel/spatial | 无显式 UB |

##### 分层内存布局

R-003 第一节点 UB 同 R-002；repair 节点不分配 AIV TBuf，实际路径为 GM bit检查 → `InitGlobalMemory`，或 GM → SIMT FP32 registers → GM。

##### 字节账与复用顺序

```text
node1 UB_total=ubSize; node2 UB_total=0 explicit bytes
dynamic_ub=N/A; workspace_total=0
cross-node GM=edge(y,mean,rstd), no extra allocation
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM Y group | base | repair判断后的 caller | repair fill/recompute或保持 | 同 stream | 每 group 由一个 outer block 处理 |
| mean/rstd | base | repair | op完成回收 | 同 stream | BF16同 dtype解释 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 跨节点 | stream ordering | Welford | repair | y/mean/rstd | 无 Host sync |
| 坏组 | BF16 exponent mask + `InitGlobalMemory` | group owner block | caller | whole group | 固定 qNaN pattern 0x7fc0 |
| 好组异常 affine | SIMT grid-stride | 2048 threads | GM | channel spatial | scale/bias顺序FP32，末端 BF16 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/router | `op_kernel/group_norm_kernel.cpp` | `launch_group_norm_direct` | base 后按 BF16/R 阈值追加节点 |
| Kernel entry | 同上 | `group_norm_repair_nonfinite_bf16` | group-stride、坏组 fill 与 VF 调用 |
| Device body | 同上 | `IsNonFiniteBf16`、`RepairNonFiniteBf16` | exponent 判断、selective scale/bias 重算 |

### Route R-001（P3）：FP32 大组双 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Launcher | `op_kernel/group_norm_kernel.cpp` | `launch_group_norm_direct` | `dtypeCode==FLOAT32 && totalGroups<=reserved && elemNum>2^18` | 两个 supplemental entry，提前 return |
| Entry | 同上 | `group_norm_parallel_reduce_f32` | grid=`totalBlocks`，VF threads=2048 | `ReduceLargeGroupF32` |
| Entry | 同上 | `group_norm_parallel_normalize_f32` | 同 grid，1 KiB TBuf，VF threads=2048 | `NormalizeLargeGroupF32` |

#### Device 数据流

##### 函数级调用链

```text
launch_group_norm_direct
  -> parallel_reduce_f32<<<totalBlocks>>> -> ReduceLargeGroupF32
  -> same-stream GM partials edge
  -> parallel_normalize_f32<<<totalBlocks>>> -> NormalizeLargeGroupF32
```

##### 源码等价伪码

```text
parts=floor(A/(N*G)); block=(group,part); 64 warps/block
for channel=part; channel<D; channel+=parts:
  each thread scans spatial+=2048; accumulate fp32 sum,sumSq
warp-reduce; lane0 writes partials[block,warp] and second plane
next kernel: every block of group reads all parts*64 partials
warp-reduce -> scratch[warp]; barrier; warp0 combines 64 entries
lane0 writes scratch mean,rstd; barrier
for channel in this part's channel-stride assignment:
  scale=rstd*gamma[channel]
  bias=-mean*scale+beta[channel]
  for thread-strided spatial:
    y=x*scale+bias
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM partial sum | `logicalBlock*64+warp` | `totalBlocks*64` | FP32 | reduce lane0 | normalize all blocks/group | plugin `mean`，两节点之间 | 连续第一平面 |
| GM partial sq | `totalBlocks*64+partial` | 同上 | FP32 | reduce lane0 | normalize | 同上 | 第二平面；总计不超 2048 |
| UB scratch | `[0:64),[64:128),[128],[129]` | 1024 B allocation | FP32 | normalize warps/lane0 | normalize threads | 每 block，Kernel 内 | barrier 前后复用其值 |
| GM Y | `group*R+channelInGroup*HW+spatial` | `N*C*HW` | FP32 dense | normalize threads | caller | caller | part/channel 唯一写 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  sum["warp sums · 64×4 B"]:1
  sq["warp square sums · 64×4 B"]:1
  stat["mean+rstd · 2×4 B"]:1
  spare["1 KiB allocation 的其余空间"]:1
```

##### 字节账与复用顺序

```text
UB_total = 1024 B（normalize node）；reduce node = 0 B 显式 UB
dynamic_ub = N/A（静态 TBuf + SIMT VF）
workspace_total = 0
GM temporary = 2*totalBlocks*64*4 <= 8192 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `mean` 临时 | reduce node | normalize node | op 完成后 allocator 回收 | 同 stream launch 顺序 | `totalBlocks<=16` |
| scratch `[0:128)` | 各 warp lane0 | warp0 | Kernel 结束 | 第一次 `asc_syncthreads` | 每 warp 唯一槽 |
| scratch stats | warp0 lane0 | 全 threads | Kernel 结束 | 第二次 `asc_syncthreads` | index 128/129 不与 partial槽重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 跨节点 | current stream ordering | reduce | normalize | GM partials | 无显式 event |
| block reduce | `WarpReduceAddSync` + `asc_syncthreads` | threads/warps | warp0/all threads | scratch | FP32 sum/sumSq；不足循环自然跳过 |
| variance | clamp | warp0 lane0 | all threads | mean/rstd | `max(sumSq/R-mean²,0)`，`rsqrtf` |
| output | grid-stride | each thread | GM | unique channel/spatial | channel 按 part stride，spatial按 2048 stride |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host producer | `op_plugin/group_norm_plugin.cpp` | `group_norm_npu` | R-001 扩展 mean 到 2048 FP32 元素 |
| Kernel router | `op_kernel/group_norm_kernel.cpp` | `launch_group_norm_direct` | 优先于官方 key 路由的两节点图 |
| Device body | 同上 | `ReduceLargeGroupF32`、`NormalizeLargeGroupF32` | per-warp partial、block scratch 合并与 scale/bias 输出 |

### Route R-005（P4）：Welford generalized 单节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_kernel/group_norm_kernel.cpp` | `calc_group_norm_tiling` | 不能 full-load 且 `C>4096` | key 1120 |
| Router | 同上 | `launch_group_norm_direct` | key 1120，非 repair 域 | typed 1120 entry |
| Body | `arch35/...welford_generalized.h` | `GroupNormSiluWelfordGeneralized<T,T>` | affine切片 | 单 AIV node |

#### Device 数据流

##### 函数级调用链

```text
group_norm_1120_<dtype> -> WelfordGeneralized::Init -> Process
  -> segmented Welford update/finalize
  -> per D/HW tile CopyGammaAndBeta2UB
  -> CopyX2UB -> VFNormalizeAndSwishAlign(false) -> CopySilu2Gm
  -> ProcessMeanAndRstd
```

##### 源码等价伪码

```text
for assigned group: compute FP32 Welford statistics exactly as R-002
if at least one aligned HW row fits:
  for D tiles: load rows gamma/beta and corresponding X; normalize rows
else:
  for each D channel: load one gamma/beta; for HW tiles normalize one row
write y and same-dtype mean/rstd with masked tails
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| phase1 regions | 同 R-002 | 同 R-002 | T/FP32 | Welford | finalize | phase1 | 同样别名 |
| phase2 X/Y | UB 0 / `2*processSize*s` | 各`2*processSize*s` | T | MTE2/Vector | Vector/MTE3 | phase2 | ping/pong |
| affine tile | UB tail | 各 `baseSize*s`，`baseSize=rows?alignE(rows):32/s` | T | per tile MTE2 | normalize | current tile | 不驻留全 C |
| stats/cast | UB tail前 | 同 R-002 | FP32/T | finalize | normalize/output | group batch | 固定尾区 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  low["phase1 X；phase2 复用为 X/Y ping-pong"]:3
  mid["tmpMean/tmpM2/dichotomy"]:1
  stat["mean/rstd (+cast)"]:1
  affine["gamma/beta 当前 tile"]:1
```

##### 字节账与复用顺序

```text
UB_total=ubSize
phase2_required=4*processSize*s + stats/cast + 2*baseSize*s
Host 对 count 递减，直到 affine tile + double-buffer X/Y <= ubRemain
dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| affine tile | MTE2 | current normalize | next D tile | 独立 `V_MTE2→MTE2_V` | 当前 Vector完成 |
| phase1/phase2 lower UB | Welford | finalize | normalize | phase order fence | 不并发 |
| X/Y parity | MTE2/Vector | Vector/MTE3 | tile+2 | hard events | depth 2 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Welford | 同 R-002 helper | phase1 | finalize | FP32 stats | aligned/三 non-align分支 |
| affine/X | MTE2_V + V_MTE2 | DMA | Vector/next DMA | tile buffers | rows或单channel tail |
| Y | V_MTE3 + MTE3_V | Vector/MTE3 | next Vector | ping/pong Y | `loopTail/innerLoopTail` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling producer | `op_kernel/group_norm_kernel.cpp` | `SetWelfordGeneralizedTiling` | affine参与的 count递减与 HW fallback |
| Kernel entry | 同上 | `group_norm_1120_*` | typed generalized body |
| Device body | `arch35/group_norm_silu_regbase_welford_generalized.h` | `NormalizeAndSwishByChannel/ByHW` | per-tile affine，Welford复用与tail |

### Route R-006（P5）：Welford generalized + BF16 修复

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Base selector | `op_kernel/group_norm_kernel.cpp` | `calc_group_norm_tiling` | key 1120、BF16 | R-005 base |
| Post selector | 同上 | `launch_group_norm_direct` | `elemNum>2^20` | append repair |
| Repair | 同上 | `group_norm_repair_nonfinite_bf16` | Device badGroup/affine tests | fill/recompute |

#### Device 数据流

##### 函数级调用链

```text
group_norm_1120_bf16 -> WelfordGeneralized body -> GM y/mean/rstd
  -> same-stream repair_nonfinite_bf16 -> InitGlobalMemory or RepairNonFiniteBf16
```

##### 源码等价伪码

```text
run complete R-005 generalized Welford
then execute exactly R-003 repair algorithm over every group
the repair reads original full gamma/beta GM, independent of base affine tiling
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| base UB | 同 R-005 | `ubSize` | TBuf | base | base | first node | generalized affine切片 |
| GM edge | y全域、mean/rstd各N·G | 原 tensor extent | BF16 | base | repair | plugin/caller | same stream |
| repair registers | per thread | FP32 scale/bias/x | register | repair | repair | iteration | 无额外 UB |

##### 分层内存布局

第一节点同 R-005；第二节点 N/A AIV UB，使用 GM bit读取、`InitGlobalMemory` 或 SIMT 寄存器重算。

##### 字节账与复用顺序

```text
node1 UB_total=ubSize; node2 explicit UB=0
workspace_total=0; GM edge不新增 tensor
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| y group | generalized base | repair/caller | conditional in-place repair | same stream | group outer loop唯一 |
| stats | generalized base | repair | op completion | same stream | 同 dtype BF16 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| base | R-005 hard events | DMA/Vector | next stage | UB slices | generalized tails |
| cross node | stream ordering | base | repair | GM y/stats | no explicit event |
| repair | R-003 mechanisms | group block/threads | GM | whole group/channel | qNaN or FP32 scale/bias |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling/base | `op_kernel/group_norm_kernel.cpp`、`arch35/...welford_generalized.h` | key1120 body | large-C partial load |
| Router/entry | `op_kernel/group_norm_kernel.cpp` | `launch_group_norm_direct`、repair entry | conditional second node |
| Device helper | 同上 | `RepairNonFiniteBf16` | group fill/selective recompute |

### Route R-007（P6）：Two-pass generalized 单节点

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_kernel/group_norm_kernel.cpp` | `calc_group_norm_tiling` | 首轮不全载；split affine 后 `newMaxReduceCount>R` | key 1130 |
| Router | 同上 | `launch_group_norm_direct` | key else terminal（合法 tiling 仅1130） | typed 1130 entry |
| Body | `arch35/...two_pass_generalized.h` | `GroupNormSiluTwoPassGeneralized<T,T>` | full R + per-group affine | 单 AIV node |

#### Device 数据流

##### 函数级调用链

```text
group_norm_1130_<dtype> -> TwoPassGeneralized::Init -> Process
  -> CopyX2UB -> CalMeanAndRstd
  -> for each group: CopyGammaAndBeta2UB(D)
  -> VFNormalizeAndSwishUnAlign(false) -> CopySilu2Gm
  -> ProcessMeanAndRstd
```

##### 源码等价伪码

```text
for group batches, ping/pong full aligned R:
  load X; two-pass FP32 mean/variance+dichotomy; compute rstd
  serialize groups in batch:
    load only this group's D gamma/beta; fence
    normalize all HW positions and write Y
  write same-dtype mean/rstd
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| X/Y | same as R-004 | each `2*processSize*s` | T ping/pong | MTE2/Vector | reduce/normalize/MTE3 | batch | depth 2 |
| stats+dichotomy | X/Y 后 | 同 R-004 | FP32/T | reduce | normalize/output | batch | sequential reuse |
| affine slice | UB tail | gamma/beta各`alignB(D*s)` | T | per group MTE2 | current group VF | current group | V_MTE2防提前覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  x["X ping/pong"]:2
  y["Y ping/pong"]:2
  stat["mean/rstd+dichotomy"]:1
  affine["当前 group 的 gamma/beta"]:1
```

##### 字节账与复用顺序

```text
UB_total=ubSize
required=4*processSize*s + stats/cast + dichotomyExtra + 2*alignB(D*s)
dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| affine slice | group i MTE2 | group i normalize | group i+1 MTE2 | V_MTE2/MTE2_V | 串行 group loop |
| X/Y parity | MTE2/Vector | Vector/MTE3 | tile+2 | ping/pong events | depth 2 |
| dichotomy | mean | mean stats | variance | local barrier/顺序 | 独立 stats已写 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| full R | MTE2_V/V_MTE2 | X DMA/Vector | reduce/next load | X parity | right padding与mask |
| affine serial | V_MTE2/MTE2_V | normalize/MTE2 | next group/current VF | gamma/beta slice | D有效元素 |
| output | V_MTE3/MTE3_V | Vector/MTE3 | next parity | Y | GM只写 `numPerCoreProcess*R` |
| numeric | same base helpers | FP32 regs | T store | stats/y | high-precision rstd + CAST_RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling producer | `op_kernel/group_norm_kernel.cpp` | split-affine branch、`SetTwoPassTiling` | key1130与processSize |
| Kernel entry | 同上 | `group_norm_1130_*` | dtype到 generalized two-pass |
| Device body | `arch35/group_norm_silu_regbase_two_pass_generalized.h`、`...base.h` | `CalNormalize`、`NormalizeAndSwish` | full R、逐group affine、事件与tail |

## 算子特定优化

### Trick T-001：分段 Welford 与二分回刷闭合大 R

#### 触发与源码依据

- 适用 Route：R-002、R-003、R-005、R-006
- 触发条件：full-load 谓词失败；Host 产生 `parallelN`、`dichotomyAddPower/K/LastNum`。
- 固定官方来源：[`group_norm_silu_regbase_welford.h:122-129`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_welford.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
CopyX2UB<T1>(xGm[xGmOffset + curNumPerCore * elemNum + i * parallelN], xPhase1Tensor[xPhase1Offset], 1,
             welfordLen);
count = count + 1;
float scale = static_cast<float>(1.0) / static_cast<float>(count);
VFWelfordParallelUpdate<T1>(x1Local, tmpMeanLocal, tmpVarLocal, i, welfordLen, scale);
```

#### 机制与设计原因

`parallelN` 把不能全载的 R 分成 UB 可容纳的 segment；每个 FP32 lane 以 `1/count` 更新 mean/M2，最后将 lane 结果按真实 segment count 加权并通过 dichotomy buffer 折叠。这样把 GM 流量维持为逐段一次读取，同时避免全 R 常驻 UB；非整除尾段由三种 count-weighted finalize 分支保持统计权重。

#### 迁移边界

- 适用：大 reduction 的 normalization，且可为 mean/M2 与二分临时区保留 FP32 UB。
- 前提：`parallelN` 32B 对齐、ping/pong 事件完整、尾段真实长度参与 finalize。
- 失效：R 可全载时应转 Two-pass；若资源不足以产生正 `parallelN`，Direct tiler没有安全 Device fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的非对齐 finalize 明确区分 `tailSize>=power`、`tailSize<=roundUp(reminder)` 与其余场景；把三者合并为等权 lane 平均会丢失尾 segment 的真实 count。

### Trick T-002：小 HW 用 NDDMA 预展开 affine，折叠多个 group 的 VF

#### 触发与源码依据

- 适用 Route：R-004
- 触发条件：`HW<=32`；Direct 中 gamma/beta 均存在，`isFold=true`。
- 固定官方来源：[`group_norm_silu_regbase_two_pass.h:235-245`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_two_pass.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (isFold) {
    CopyGammaAndBeta2UBByNDDMA<T2>(gammaGm, betaGm, gammaTensor, betaTensor, numGroups, shapeD, hwNum,
                                   elemNumAlign, hasGamma, hasBeta);
} else {
    CopyGammaAndBeta2UB<T2>(gammaGm, betaGm, gammaTensor, betaTensor, 1, shapeC, hasGamma, hasBeta);
}
```

#### 机制与设计原因

三维 NDDMA 的最后一维源 stride 为 0、目标 stride 为 1，把每个 channel 的 gamma/beta 广播到 `HW` 连续位置；Fold VF 随后可把跨 group 的连续区间合并，减少逐 group VF 启动与寄存器广播准备。代价是 affine UB 从 `C` 扩为 `G·alignE(D·HW)`，因此 Host 把它计入 full-load guard。

#### 迁移边界

- 适用：小空间维、affine广播可换取连续 Fold VF 的 normalization。
- 前提：NDDMA 三维 loop/stride 与 `(G,D,HW)` 布局一致，UB 计入展开后容量。
- 失效：`HW>32` 使用 ordinary affine copy；generalized Route 按 D tile 搬运，不做全局展开。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能只按 `C*s` 预留 UB 后仍启用 NDDMA 展开；Device 的目标 stride 是 `elemNumAlign`，容量不足会越过 affine 子区。

### Trick T-003：高精度 reciprocal-sqrt 修正与异常选择

#### 触发与源码依据

- 适用 Route：R-002–R-007 的官方适配 base 节点
- 触发条件：mean/variance 归约完成，调用 `CalRstdByHighPrecision(var,rstd,epsilon)`。
- 固定官方来源：[`group_norm_silu_regbase_base.h:200-217`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_base.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Adds(var, var, epsilon, pregMerge);
Div(r, one, var, pregMerge);
Sqrt(y, r, pregMerge);
Muls(t, var, SCALAR1, pregMerge);
Mul(t, t, y, pregMerge);
Mula(t1, t, y, pregMerge);
Mul(rstd, y, t1, pregMerge);
```

#### 机制与设计原因

先算 `sqrt(1/x)`，再用 `1.5-0.5*x*y*y` 做一次 Newton 型修正，后续还构造 residual 做补偿，并用比较/选择处理极端值。整个标量统计路径保持 FP32，B16 只在最终输出阶段转换，因此减少低精度 rstd 对所有 group 元素的放大误差。

#### 迁移边界

- 适用：variance 非负且 epsilon 使分母处于 helper 定义域的 normalization。
- 前提：保留指令顺序、FP32 中间值和异常 `Select`；不得代数重排。
- 失效：Direct 未检查负 epsilon；分母非法时只能接受 helper 的异常传播，不能宣称有限输出。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证的低精度 reciprocal-sqrt 替代实现；不能据此编造性能或误差收益百分比。

## 模板与实现对比

| 优先级 | Route | 命中条件与 Host 顺序 | 模板与 dtype | 循环与分核 | UB/GM 与复用 | 搬运/计算 | 同步、精度与 tail | fallback/失败边界 |
|---:|---|---|---|---|---|---|---|---|
| P0 | R-002 | partial、`C<=4096`；H5→H6 | Welford `<T,T>` | group/core，R segment+D/HW tile | phase1/phase2 lower UB alias；全 C affine | DataCopyPad、Welford/dichotomy | hard events；FP32；三 non-align分支 | 大 BF16转P2；launch错误不显式回传 |
| P1 | R-004 | strict full-load；H2→H6 | TwoPass `<T,T>` | 多 group batch，full R | X/Y DB+全 affine | DataCopyPad；小 HW NDDMA；two-pass | hard events；FP32；mask/pad | H2 失败经冗余 H3 继续H4 |
| P2 | R-003 | P0 base+大 BF16；H5→H6→H7 | Welford BF16 + repair | base group/core；repair group/thread | base UB + 跨节点 GM y/stats | base Welford；坏组fill或scale/bias重算 | stream序；B16/FP32；spatial stride tail | repair后结束 |
| P3 | R-001 | FP32少组大R；H1命中return | FP32 SIMT | group×part block | 两片 GM partials；1 KiB scratch | direct GM、warp reduce | stream+barrier；FP32 sumSq；grid stride | 未命中去H2 |
| P4 | R-005 | partial、`C>4096`；H5→H6 | WelfordGeneralized `<T,T>` | group/core，R segment+affine tile | phase alias+tile affine | DataCopyPad、Welford | hard events；FP32；D/HW tile tail | 大 BF16转P5 |
| P5 | R-006 | P4 base+大 BF16；H5→H6→H7 | generalized Welford BF16 + repair | 同P4后接repair | tile affine + 跨节点 GM edge | 同P4后接fill/recompute | stream序；B16/FP32 | repair后结束 |
| P6 | R-007 | split-affine full-load；H4→H6 | TwoPassGeneralized `<T,T>` | 多 group batch，full R，逐group affine | X/Y DB+group affine | DataCopyPad、two-pass | hard events；FP32；group/tile tail | H4失败去H5 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方 GroupNorm/GroupNormSilu 文档支持；本 Direct 交付未目标化 | 官方文档支持；本 Direct 交付未目标化 | 本卡目标 `950`；官方 README/OpDef/config 支持 |
| Host/Launcher | 本卡固定交付不可见 | 本卡固定交付不可见 | 运行时查询 UB/AIV/VL；Direct typed launch；CANN 9.1.0 |
| Kernel/资源 | 不得复用本卡 950 常量 | 不得复用本卡 950 常量 | DAV-3510 arch35 body；16 AIV、253952 B UB、256 B VL 为转换映射规格 |

## 限制与失败边界

- Direct 输入只覆盖 mandatory、同 dtype、连续、同设备 gamma/beta；不等同于官方 ACLNN 的 optional/mixed-dtype/contiguous 转换语义。
- 空 `x` 在插件层直接返回 `empty_like`，不写 mean/rstd，也不启动官方 empty body；公开只返回 y，空内容无可观察写值保证。
- `epsilon` 未校验正值或有限值；Host shape/元素数乘法及若干 `int64→uint32/int32/uint16` 参数窄化未做统一溢出检查。
- R-001 的 `sumSq/R-mean²` 对严重 cancellation 的域没有源码级数值保证；只可按交付公式复用。其 launcher 用运行时 `A`，而插件只在 `N·G<=16` 时扩容 partial buffer；目标 950 由 `A=16` 闭合，任何不同 AIV 数的平台都必须统一这两个谓词后才能复用。
- BF16 repair 只在 `R>2^20` 启用，坏组判定还依赖 mean/rstd 与 X 首尾；更小 R 或内部非有限但边界有限的 adversarial 域不在转换闭包声明内。
- 合法 tiling key 只有 1100/1110/1120/1130；Launcher 的最终 `else` 映射 1130，没有 unknown-key 报错。调用者不得绕过 `calc_group_norm_tiling` 注入任意 POD。
- 多 Kernel 依赖同一 stream 的入队顺序；Launcher 为 `void`，没有显式检查 enqueue/Device 错误。allocator 完成期生命周期由框架边界保证，交付源码未展开。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 公开语义/平台 | `norm/group_norm/docs/aclnnGroupNorm.md` | `aclnnGroupNorm` 文档 | GroupNorm 公式域、950支持映射 |
| 稳定实现转发 | `norm/group_norm/op_host/op_api/aclnn_group_norm.cpp` | `aclnnGroupNormGetWorkspaceSize` | RegBase 调 `GroupNormSilu(...,false)` |
| 映射平台 | `norm/group_norm_silu/README.md` | 产品支持表 | Ascend 950支持 |
| OpDef | `norm/group_norm_silu/op_host/group_norm_silu_def.cpp` | `GroupNormSilu` | ascend950→`group_norm_silu_apt` 与 dtype |
| POD | `norm/group_norm_silu/op_host/group_norm_silu_tiling.h` | `GroupNormSiluRegbaseTilingData` | 字段顺序与 key |
| Host/Tiling | `norm/group_norm_silu/op_host/group_norm_silu_tiling_arch35.cpp` | `SetTilingKey4Regbase`、`Tiling4GroupNormSiluRegBase` | key、UB、blockDim、workspace的转换来源 |
| Kernel router | `norm/group_norm_silu/op_kernel/group_norm_silu_apt.cpp` | `group_norm_silu` | key→四个 arch35 body |
| 公共 helper | `norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_base.h` | Welford/two-pass/normalize/DMA helpers | FP32、mask、NDDMA、tail |
| Welford perf | `norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_welford.h` | `GroupNormSiluWelford` | R-002/R-003 base |
| Welford generalized | `norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_welford_generalized.h` | `GroupNormSiluWelfordGeneralized` | R-005/R-006 base |
| Two-pass perf | `norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_two_pass.h` | `GroupNormSiluTwoPass` | R-004 |
| Two-pass generalized | `norm/group_norm_silu/op_kernel/arch35/group_norm_silu_regbase_two_pass_generalized.h` | `GroupNormSiluTwoPassGeneralized` | R-007 |
| 构建接线 | `norm/group_norm_silu/op_host/config/ascend950/group_norm_silu_binary.json` | `GroupNormSilu` variants | 950 binary dtype/activate_silu组合 |

Frontmatter 仅收录固定官方 HTTP(S) blob URL。Direct 包装、typed entry、R-001 与 repair 的事实来自最终交付相对文件 `csrc/ops/group_norm/op_plugin/group_norm_plugin.cpp`、`op_kernel/group_norm_{launch,tilingdata}.h`、`op_kernel/group_norm_kernel.cpp` 和 `CMakeLists.txt`；它们没有被伪造为官方 URL，也不用于反推官方接口。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 184B tiling + 8B launch tail | 字段顺序、epsilon padding、by-value typed entry | `GroupNormSiluRegbaseTilingData`、`GroupNormLaunchParams` |
| 调用顺序 | plugin→tiling→launcher；base后可append repair | R-001必须提前return；repair必须在base后同stream | `group_norm_npu`、`launch_group_norm_direct` |
| workspace/内存 | 无显式 workspace；mean可作partials | R-001 partial双平面<=2048；base整 UB TBuf切片不得重叠活跃区 | `NormalizeLargeGroupF32`、四个 `InitInnerBuffer` |
| 同步 | AIV hard events；SIMT barrier；跨节点stream序 | ping/pong正反事件、两次barrier、producer→consumer顺序 | `CalNormalize`、`NormalizeLargeGroupF32` |
| 精度与尾块 | FP32统计与normalize、B16 CAST_RINT、Welford三种非对齐 finalize | 不代数重排 rstd/scale-bias；mask用真实有效长 | `CalRstdByHighPrecision`、`VFWelfordParallelFinalizeNonAlign` |
| 失败边界 | tiling `-1` 可回传；空输入Host终止 | unknown key不可外部注入；launch错误当前不显式回传 | `calc_group_norm_tiling`、`group_norm_npu` |