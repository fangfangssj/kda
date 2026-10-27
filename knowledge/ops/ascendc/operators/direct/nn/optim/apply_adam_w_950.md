---
type: Operator
title: "ApplyAdamW 950 Direct Invoke 算子知识"
description: "说明 cann_bench::apply_adam_w 在 950 上的接口语义、Direct Launcher 参数生成、Route 选择、单 Kernel 启动图与 AIV 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w/op_host/arch35/apply_adam_w_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w/op_kernel/arch35/apply_adam_w.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w/op_kernel/arch35/apply_adam_w_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T19:29:35Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ApplyAdamW 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::apply_adam_w` 的 `PrivateUse1` 实现 `apply_adam_w_npu`，经 `launch_apply_adam_w_direct` 选择 12 个显式 entry 之一 |
| 计算语义 | 由输入 `m/v` 计算 bias-corrected moment，只写新张量 `y`；不回写 `var/m/v`；`maximize` 改变最终更新方向 |
| 输入输出 | `var/grad/m/v` 同 shape、同 dtype、连续 ND；rank 1～8；dtype 为 FP32/FP16/BF16；输出 `y` 与 `var` 同 shape/dtype |
| 执行类型 | 所有 R-001～R-004 均为单节点 `AIV_VECTOR`；无可达 Cube/AIC、MIX_CV、SIMT、NDDMA 或通信节点 |
| Kernel 路线 | 四条均为 `active`，无 `conditional`/`dormant` Route。按重要性为 R-002 有 decay/minimize、R-001 无 decay/minimize、R-004 有 decay/maximize、R-003 无 decay/maximize；每条 Route 各有 FP32、FP16、BF16 entry |
| 核心资源 | 5 路 GM、一个由 ATVOSS 划成 4×2 个输入 ping/pong 子区和 1×2 个输出 ping/pong 子区的 `TBuf`、MicroAPI 寄存器；workspace 为 0 |
| 来源闭包 | `direct_implementation_closed`：转换交付件已闭合 schema→Meta/PrivateUse1→Host tiling→Direct launcher→12 个 entry→`ElementwiseSch`→寄存器 body；官方固定 Commit 仅支撑 950、ATVOSS、AIV 与低精度转换机制，Direct 输出语义和 ABI 明确归于转换交付件 |
| 证据边界 | 固定来源闭合源码链路；本卡不主张编译、真机、数值或性能结论。 |

## 接口与语义

### 计算语义

对展平后的每个元素 `i`，交付件先在每个 AIV core 上用 FP32 标量计算：

```text
c1 = 1 / (1 - beta1^step)
c2 = 1 / (1 - beta2^step)
m_hat[i] = (beta1*m[i] + (1-beta1)*grad[i]) * c1
v_hat[i] = (beta2*v[i] + (1-beta2)*grad[i]*grad[i]) * c2
u[i] = m_hat[i] / (sqrt(v_hat[i]) + epsilon)
if weight_decay != 0: u[i] += weight_decay * var[i]
y[i] = var[i] + (maximize ? lr : -lr) * u[i]
```

FP16/BF16 输入在寄存器中提升到 FP32 完成上述流水，最后以 `CAST_RINT` 降回 `T`；FP32 不做 cast。公式中的 `m_hat/v_hat` 只是寄存器中间值，不写回 `m/v`。

转换边界必须保留：官方固定源码的 `ApplyAdamW` 接收 beta-power 张量、可选 AMSGrad，并原位输出 `var/m/v`；其 `maximize` 在 moment 计算前反转 `grad`。本文 Direct 交付件改为标量 `beta1/beta2/step`、只产生 `y`，并把 `maximize` 施加于完整最终更新方向。因此官方源码只证明可迁移的 950 ATVOSS/AIV 机制，不证明本文目标公式等同于官方公开算子。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `var` | rank 1～8，非空时连续 ND | FP32/FP16/BF16 | 参数基值和 device guard 锚点；不被修改 |
| `grad` | 与 `var` 完全相同，非空时连续 ND | 与 `var` 相同 | 当前梯度；不被修改；实现未显式校验 device 等于 `var.device()` |
| `m` | 与 `var` 完全相同，非空时连续 ND | 与 `var` 相同 | 旧一阶 moment；不被修改；实现未显式校验 device 等于 `var.device()` |
| `v` | 与 `var` 完全相同，非空时连续 ND | 与 `var` 相同 | 旧二阶 moment；不被修改；实现未显式校验 device 等于 `var.device()` |
| `lr` | Host 标量 | schema 为 double，Device 为 float | 学习率；launcher 显式窄化为 FP32 |
| `beta1` | Host 标量 | double→float | 一阶衰减系数 |
| `beta2` | Host 标量 | double→float | 二阶衰减系数 |
| `weight_decay` | Host 标量 | double→float | `double != 0.0` 决定是否实例化 decay body |
| `epsilon` | Host 标量 | double→float | 默认 `1e-8`；加到 `sqrt(v_hat)` 后 |
| `step` | Host 标量 | int64 | 默认 1；Device 用整数快速幂求 `beta^step` |
| `maximize` | Host 标量 | bool | 默认 false；true 时 `signed_lr=+lr` |
| `y` | 与 `var` 相同，连续 ND | 与 `var` 相同 | `empty_like(var)` 新分配，只写有效元素 |

受支持执行域要求四个输入位于 `var` 所在的同一 NPU device。源码只安装 `var.device()` guard，并校验 shape/dtype/contiguous；它不比较 `grad/m/v` 的 device。因此“同 device”是调用方必须满足、但 Host 未证明的安全前提，不能把混合 device 输入归入任何 Route。空域在读取其余输入地址前返回，仍不把混合 device 扩展为受支持语义。

### 入口与版本

转换交付件通过 `TORCH_LIBRARY_FRAGMENT(cann_bench)` 声明 schema，Meta 入口 `apply_adam_w_meta` 校验元数据并创建 `empty_like`，`TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1)` 绑定 `apply_adam_w_npu`。它在当前 NPU stream 上直接启动 AscendC kernel，不是 ACLNN 两段式 workspace/executor 入口，也没有 OPP Host/Tiling registry POD。

目标为 CANN 9.1.0、DAV-3510/Ascend 950。固定官方 Commit 的 README 明确列出 Ascend 950PR/950DT 支持，根 `CMakeLists.txt` 又把 `SUPPORT_COMPUTE_UNIT` 限定为 `ascend950`、Tiling 目录限定为 `arch35`；Direct 交付件据此转换 950 实现机制。

## Host/Launcher 执行流程

### 参数校验与前处理

真实顺序为：安装 `var.device()` guard→调用 Meta 检查四个 Tensor shape/dtype 完全相同、rank 1～8、dtype 属于 FP32/FP16/BF16→分配 `y=empty_like(var)`→读取 `total_length=var.numel()`。若元素数为 0，立即返回 `y`；非空时再逐一要求 `var/grad/m/v.is_contiguous()`。这条链没有 `grad/m/v.device()==var.device()` 检查；合法调用必须由上层保证同一 NPU device。

随后由 dtype 生成 `dtype_code`（0/1/2），调用 `calc_apply_adam_w_tiling_params`，取得 `num_blocks/block_former/ub_former`，抓取当前 stream 和五个 GM 地址，并以 `weight_decay != 0.0` 生成 `has_weight_decay`。所有 double 属性在调用 launcher 时才静态转换为 float。

### 参数、Workspace 与生命周期

设 `N=total_length`、`S=sizeof(T)`、`C=max(GetCoreNumAiv(),1)`、`U=GetCoreMemSize(UB)`、`A=256/S` 个元素：

```text
B0 = max(min(C, ceil(N/2048)), 1)
block_former = ceil_align(ceil(N/B0), A)
num_blocks = ceil(N/block_former)
buffer_num = TargetAdamWDag<T,...>::OpDag::BufferNum = 10
max_dtype_bytes = S
ub_former = floor((U/(10*S))/A) * A
if ub_former < A: ub_former = A
```

`y` 及输入由 PyTorch Tensor 持有；lambda 捕获 GM 地址、标量、tiling 和 stream，在 `OpCommand::RunOpApi("ApplyAdamWDirect", launch)` 内提交。没有用户 workspace、registry tiling buffer或跨 Kernel 中间区，`workspace_total=0`。`TPipe` 与其单个大 `TBuf` 只在一次 kernel 生命周期内存在。

### 启动图与失败传播

```text
cann_bench::apply_adam_w / PrivateUse1
  -> apply_adam_w_npu
    -> calc_apply_adam_w_tiling_params(N,dtype_code)
    -> launch_apply_adam_w_direct(...,num_blocks,block_former,ub_former,...,stream)
      -> exactly one apply_adam_w_{f32|f16|bf16}_{wd0|wd1}_{min|max}<<<num_blocks,...,stream>>>
        -> RunTargetAdamW<T,HasWeightDecay,Maximize>
          -> ElementwiseSch<1,TargetAdamWDag<...>::OpDag>::Process
```

这是单 Kernel 图，没有节点间 workspace 边。launcher 的嵌套 `if` 精确选择一个 entry；所有 launch 共用当前 stream。Direct launch 函数返回 `void`，`RunOpApi` lambda 无条件返回 0，因此交付件没有可见的 kernel launch 返回码检查、异步错误收集或 cleanup 分支；Meta/contiguous 失败会在 launch 前抛出。

## Route 与参数设计

### 分核与资源计算

先按至少 2048 元素/核估算 block 数，再把每个 former block 向上对齐到 256B DMA 粒度，最后重新计算实际 `num_blocks`，所以最后一个 block 的元素数为 `tail_block=N-block_former*(num_blocks-1)`。每个 core 的 GM 基址为 `blockIdx*block_former`；非尾核处理 `block_former`，尾核处理 `tail_block`。

每核又按 `ub_former` 切 tile：former/tail block 的循环数分别为 `ceil(block_length/ub_former)`，末 tile 分别由 `ubTailOfFormerBlock/ubTailOfTailBlock` 给出。ATVOSS `Process` 用 `i&1` 在两套 input/output 子区间切换；tile 内每个 MicroAPI repeat 固定覆盖 `256/sizeof(float)=64` 个 FP32 lane。raw UB→Reg `DataCopy` 读取完整寄存器跨度，安全性来自完整 UB 子区；`UpdateMask<float>(remaining)` 从低精度 promote cast/FP32 算术延续到 masked store，防止最后不足 64 lane 的值参与或发布。

### Route 选择

#### Route 特征表（重要性降序）

| 重要性 | Route | 可达性 | Key/模板 | 精确选择条件 | 计算特征 | Host 顺序、短路与 fallback |
|---:|---|---|---|---|---|---|
| P0 | R-002 有 decay/minimize | active | `T,true,false` | `has_weight_decay==true && maximize==false` | 覆盖完整 optimizer update：FP32 update 中保留 decay `Axpy`，最终 `signed_lr=-lr` | 空域已在 router 前短路；dtype 按 0→1→`else`，每个 dtype 内布尔树先进入 wd1、再由 `maximize=false` 命中第二个叶子 |
| P1 | R-001 无 decay/minimize | active | `T,false,false` | `has_weight_decay==false && maximize==false` | 编译期删除 decay `Axpy`，避免在无 decay 语义下执行 `0*var`；最终 `signed_lr=-lr` | 空域已短路；wd1 子树不命中后进入 wd0，`maximize=false` 命中第四个叶子 |
| P2 | R-004 有 decay/maximize | active | `T,true,true` | `has_weight_decay==true && maximize==true` | 完整 decay update，最终 `signed_lr=+lr` | 空域已短路；每个 dtype 布尔树的第一个叶子 |
| P3 | R-003 无 decay/maximize | active | `T,false,true` | `has_weight_decay==false && maximize==true` | 编译期删除 decay `Axpy`，最终 `signed_lr=+lr` | 空域已短路；wd1 子树不命中后进入 wd0，命中第三个叶子 |

四条 Route 对合法 shape/dtype 的覆盖宽度相同；重要性先按常规 minimize 方向，再在同一方向中优先保留算术更完整的 wd1 body、随后保留为无 decay 提供数值隔离的 wd0 specialization，最后放置两条 maximize 镜像路线。P0～P3 只表示维护与复用优先级，不改写 launcher 的真实控制流。四条 Route 构成合法非空输入域上的完备互斥分区：源码没有失败后再尝试另一 body 的 `conditional` 路线，也没有编译但被生产 selector 永久关闭的 `dormant` entry。空域返回、Host 拒绝和非法裸 C ABI 的误落都不是新 Route。

Host 的 dtype `else` 在合法 Plugin 域内只承接 BF16；若绕过 Plugin 直接调用 C ABI，非 0/1 的码也会误落 BF16 entry。这是 fail-open 边界，不是受支持的 BF16 fallback。

```text
if dtype_code == 0: T=float
else if dtype_code == 1: T=half
else: T=bfloat16_t

if has_weight_decay:
  launch Maximize ? entry<T,true,true> : entry<T,true,false>
else:
  launch Maximize ? entry<T,false,true> : entry<T,false,false>
```

### Production 可达性与 fallback 出边

Plugin 先把 Torch API 输入收窄到 `dtype_code∈{0,1,2}`，并在非空域证明四个 Tensor 同 shape、同 dtype、连续；同一 NPU device 是未被 Host 检查的共同安全前提。launcher 随后只读 `dtype_code/has_weight_decay/maximize`。每条命中后恰好发射一个 entry，Device 不再选择算法，也没有跨 Route fallback。

| Route | 生产可达谓词 | selector/key/template consumer | 不命中时的真实下一跳 | 命中后的出边 |
|---|---|---|---|---|
| R-002 | `N>0 && dtype∈{FP32,FP16,BF16} && weight_decay!=0.0 && !maximize` | dtype 树→wd1→min；`{f32,f16,bf16}_wd1_min`；`<T,true,false>` | `maximize=true` 转同 dtype 的 R-004；`weight_decay==0.0` 转 wd0 子树 | 单 entry→单 `AIV_VECTOR` body；无 Device fallback |
| R-001 | `N>0 && dtype∈{FP32,FP16,BF16} && weight_decay==0.0 && !maximize` | dtype 树→wd0→min；`{f32,f16,bf16}_wd0_min`；`<T,false,false>` | `maximize=true` 转同 dtype 的 R-003；wd1 已由 R-002/R-004 承接 | 单 entry→单 body；无 Device fallback |
| R-004 | `N>0 && dtype∈{FP32,FP16,BF16} && weight_decay!=0.0 && maximize` | dtype 树→wd1→max；`{f32,f16,bf16}_wd1_max`；`<T,true,true>` | `maximize=false` 转同 dtype 的 R-002；`weight_decay==0.0` 转 wd0 子树 | 单 entry→单 body；无 Device fallback |
| R-003 | `N>0 && dtype∈{FP32,FP16,BF16} && weight_decay==0.0 && maximize` | dtype 树→wd0→max；`{f32,f16,bf16}_wd0_max`；`<T,false,true>` | `maximize=false` 转同 dtype 的 R-001；wd1 已由 R-002/R-004 承接 | 单 entry→单 body；无 Device fallback |

决策树外只有三类边界：Meta/contiguous 检查在 launch 前抛错；`N==0` 直接返回空 `y`；合法非空输入提交 kernel 后，lambda 固定返回 0，源码不同步读取 Device 状态。非法裸 C ABI dtype 会误落 BF16，不能把它记作兼容或精度 fallback。

### 逐 Route 实现闭包

下表把共同的 scheduler 不变量显式展开到每条 Route；“本地 tiling”指 kernel 内按 direct scalar 参数构造的 `EleBaseTilingData`，不是 Host 序列化 POD。

| 重要性/Route | Host producer→POD/consumer→launch | Device entry/body 与核/任务映射 | GM/片上存储/workspace | 同步、生命周期与 tail | 优化器精度顺序与 fallback |
|---:|---|---|---|---|---|
| P0 R-002 | `apply_adam_w_npu` 生成合法 `dtype_code`、`has_weight_decay=true`、`maximize=false`、三元 tiling 和 FP32 标量→direct scalar ABI→`RunTargetAdamW` 构造本地 tiling→`*_wd1_min<<<num_blocks,...,stream>>>` | `RunTargetAdamW<T,true,false>`→ATVOSS→`CalcTargetAdamW`；`blockIdx` 映射一个连续 former/tail block，tile 顺序映射该核任务 | 4 路 GM 读+1 路 GM 写；单 `TBuf` 划成 10 个 `ub_former*S` 子区；寄存器中间值；L1/L0/workspace 均无分配或访问 | 当前 stream 保序；`Get/ReleaseTensor` 约束 VECIN→VECCALC→VECOUT；无跨核同步；kernel/`TPipe` 结束即释放；core tail→tile tail→64-lane live mask | double 属性先窄化 FP32；power/系数、moment、sqrt/add/div、decay Axpy、负 lr Axpy 均 FP32；低精度仅边界 cast，最后 RINT；无 Device fallback |
| P1 R-001 | `apply_adam_w_npu` 生成合法 `dtype_code`、`has_weight_decay=false`、`maximize=false`、三元 tiling 和 FP32 标量→direct scalar ABI→Device 本地 tiling→`*_wd0_min<<<num_blocks,...,stream>>>` | `RunTargetAdamW<T,false,false>`→ATVOSS→`CalcTargetAdamW`；每个 `blockIdx` 处理自己的连续 former/tail block，块内顺序切 tile | 4 路 GM 读+1 路 GM 写；单 `TBuf` 的 10 个子区和寄存器中间值；无 L1/L0/workspace；`if constexpr` 删除 decay 但不改变 UB 账 | 当前 stream；VECIN→VECCALC→VECOUT Get/Release；无跨核同步；kernel 末释放 `TPipe`；core/tile/64-lane 三级 tail | FP32 power/系数→moment→sqrt/add/div→负 lr Axpy，跳过 decay 以避免 `0*var`；低精度仅边界 cast/RINT；无 Device fallback |
| P2 R-004 | `apply_adam_w_npu` 生成合法 `dtype_code`、`has_weight_decay=true`、`maximize=true`、三元 tiling 和 FP32 标量→direct scalar ABI→Device 本地 tiling→`*_wd1_max<<<num_blocks,...,stream>>>` | `RunTargetAdamW<T,true,true>`→ATVOSS→`CalcTargetAdamW`；`blockIdx` 连续分块，块内 former/tail tile 顺序执行 | 4 路 GM 读+1 路 GM 写；单 `TBuf` 的 10 个子区和寄存器中间值；decay 不增加 UB；无 L1/L0/workspace | 当前 stream 与 Get/Release producer-consumer 协议；无跨核同步；资源限于单 kernel；core/tile/64-lane 三级 tail | FP32 power→moment→sqrt/add/div→decay Axpy→正 lr Axpy；maximize 不在 moment 前翻转 grad；低精度最终 RINT；无 Device fallback |
| P3 R-003 | `apply_adam_w_npu` 生成合法 `dtype_code`、`has_weight_decay=false`、`maximize=true`、三元 tiling 和 FP32 标量→direct scalar ABI→Device 本地 tiling→`*_wd0_max<<<num_blocks,...,stream>>>` | `RunTargetAdamW<T,false,true>`→ATVOSS→`CalcTargetAdamW`；`blockIdx` 连续分块，块内 former/tail tile 顺序执行 | 4 路 GM 读+1 路 GM 写；单 `TBuf` 的 10 个子区和寄存器中间值；无 decay 指令、L1、L0 或 workspace | 当前 stream 与 VECIN/VECCALC/VECOUT Get/Release；无跨核同步；kernel 末释放资源；core/tile/64-lane 三级 tail | FP32 power/系数→moment→sqrt/add/div→正 lr Axpy，跳过 decay；FP16/BF16 仅边界 promote/RINT；无 Device fallback |

四条 Route 是 source mapping 冻结的语义路线；每条保留三个精确 dtype entry。FP16/BF16 共享 unpack→FP32→pack 流水，FP32 entry 删除这两端转换；这项 dtype 子路径差异在矩阵、内存和精度表中逐项列出，不把不同 entry 隐去。

### 模板/实现执行矩阵

| 重要性 | Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---:|---|---|---|---|---|---|---|
| P0 | R-002 | `*_wd1_min` | dtype_code 0/1/2；非空连续 ND；`weight_decay!=0.0`；`maximize=false`；950 | 单 AIV Kernel | `RunTargetAdamW<float/half/bfloat16_t,true,false>` | 10 个 ATVOSS 子区；多 `Axpy(update,var,weight_decay)`；`signed_lr=-lr` | core/tile/64-lane mask；无 Device fallback |
| P1 | R-001 | `*_wd0_min` | 同上但 `weight_decay==0.0` | 单 AIV Kernel | `RunTargetAdamW<T,false,false>` | 同上，编译期删除 decay Axpy；`signed_lr=-lr` | 同上 |
| P2 | R-004 | `*_wd1_max` | `weight_decay!=0.0 && maximize=true` | 单 AIV Kernel | `RunTargetAdamW<T,true,true>` | decay Axpy；`signed_lr=+lr` | 同上 |
| P3 | R-003 | `*_wd0_max` | `weight_decay==0.0 && maximize=true` | 单 AIV Kernel | `RunTargetAdamW<T,false,true>` | 无 decay；`signed_lr=+lr` | 同上 |

### 隐式性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 非空短路 | `total_length==0` | 避免 `num_blocks` 与 Device loop 的零域 | 全部 | Host 直接返回空 `y` | 转换件 `apply_adam_w_npu` |
| 连续布局 | 四输入 `is_contiguous()` | scheduler 只用展平线性地址，无 stride POD | 全部 | Host 抛错 | `apply_adam_w_npu` |
| 每核最少工作估算 | `ceil(N/2048)` 与 `min(C,...)` | 避免对小张量启动全部 AIV core | 全部 | 至少 1 block | `CalcTiling<T>` |
| 256B 对齐 | `block_former` 与 `ub_former` 对齐 `256/S` 元素 | former block/tile 满足 DMA 粒度，尾部交给 `DataCopyPad` 与 mask | 全部 | 强制最小 `A` | `CalcTiling<T>` |
| UB 上界 | `ub_former=floor(U/(BufferNum*MaxDtypeBytes))` 后向下对齐 | 10 个等长子区共同放入一个 `TBuf` | 全部 | 若小于 A 则强制 A；源码无额外失败检查 | `CalcTiling<T>`、`ElementwiseSch::Init` |
| decay 编译期消除 | `has_weight_decay=(weight_decay!=0.0)` | `if constexpr` 完全删除 decay `Axpy` | R-002/R-001/R-004/R-003 | 依 Host 顺序选择对应 wd0/wd1 entry | launcher、`CalcTargetAdamW` |
| repeat 位宽 | `repeat_times=ceil(count/64)` 后循环变量/界限转 `uint16_t` | MicroAPI 寄存器循环 | 全部 | 无 runtime fallback；需由 UB tile 上界保证不截断 | `CalcTargetAdamW` |
| dtype default | launcher 中 dtype_code 非 0/1 进入 BF16 分支 | Direct ABI router 没有 default error | 全部 | 被解释为 BF16 | `launch_apply_adam_w_direct` |

### POD/ABI

Direct entry 不接收序列化 tiling POD；下表的 “direct scalar ABI” 是独立 kernel 参数，故无结构 `offsetof/sizeof`。entry 在 Device 上构造自然对齐的 `EleBaseTilingData`：`sizeof=80`、`alignof=8`，字段 offset 如表。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | producer | consumer | 边界 |
|---|---|---|---|---|---|---|---|
| direct scalar ABI | `var` | GM_ADDR/指针宽度 | 独立参数 | GM 地址 | `var.data_ptr()` | scheduler `In0` | 连续 N 元素 |
| 同上 | `grad` | GM_ADDR | 独立参数 | GM 地址 | `grad.data_ptr()` | `In1` | 同上 |
| 同上 | `m` | GM_ADDR | 独立参数 | GM 地址 | `m.data_ptr()` | `In2` | 同上 |
| 同上 | `v` | GM_ADDR | 独立参数 | GM 地址 | `v.data_ptr()` | `In3` | 同上 |
| 同上 | `y` | GM_ADDR | 独立参数 | GM 地址 | `empty_like(var).data_ptr()` | `Out0` | 仅此地址被写 |
| 同上 | `total_length` | int64/64 | 独立参数 | 元素 | `var.numel()` | `RunTargetAdamW` | 非空 launch |
| 同上 | `num_blocks` | int64/64 | 独立参数 | AIV core | `CalcTiling` | tiling `coreNum/blockNum` | `coreNum` 窄化 int32 |
| 同上 | `block_former` | int64/64 | 独立参数 | 元素/非尾核 | `CalcTiling` | tiling `blockFormer` | 256B 对齐 |
| 同上 | `ub_former` | int32/32 | 独立参数 | 元素/tile | Host 从 int64 窄化 | tiling `ubFormer` | 无显式溢出检查 |
| 同上 | `lr` | float/32 | 独立参数 | 数值 | `static_cast<float>(lr)` | Var 0、`signed_lr` | double 精度不保留 |
| 同上 | `beta1` | float/32 | 独立参数 | 数值 | Host cast | `DeviceIntegerPower`/系数 | 无范围校验 |
| 同上 | `beta2` | float/32 | 独立参数 | 数值 | Host cast | 同上 | 无范围校验 |
| 同上 | `weight_decay` | float/32 | 独立参数 | 数值 | Host cast | Var 5；wd1 body | wd0 body 不读计算路径 |
| 同上 | `epsilon` | float/32 | 独立参数 | 数值 | Host cast | Var 6、`Adds` | 无非负校验 |
| 同上 | `step` | int64/64 | 独立参数 | 次数 | schema 属性 | `DeviceIntegerPower` | 允许 0/负数，未校验 |
| launcher-only control | `dtype_code` | int32/32 | 独立参数，不进入 kernel 参数表 | selector 枚举 | Host 从 `var.scalar_type()` 生成 0/1/2 | `launch_apply_adam_w_direct` 外层 dtype 树 | 非 0/1 会误落 BF16；正常 Host 只产生 0/1/2 |
| 同上 | `has_weight_decay` | C++ `bool` | 独立参数，不序列化 | selector 布尔 | Host 以 double `weight_decay!=0.0` 生成 | launcher wd0/wd1 树→模板常量 | 与窄化后 Device float 可能不一致 |
| 同上 | `maximize` | C++ `bool` | 独立参数，不序列化 | selector 布尔 | schema 属性 | launcher min/max 树→模板常量 | 只决定最终 `signed_lr` 符号 |
| launch config | `stream` | `void*`/指针宽度 | 独立参数，不进入 kernel 参数表 | runtime stream | `getCurrentNPUStream().stream(false)` | `entry<<<num_blocks,nullptr,stream>>>` | callback 不同步、不取回异步状态 |
| `EleBaseTilingData` | `dim0` | int64/64 | off 0 | 元素 | Device `RunTargetAdamW←total_length` | 元数据；scheduler 主路径不用于地址 | 非空 |
| 同上 | `coreNum` | int32/32 | off 8 | core | Device `RunTargetAdamW←(int32)num_blocks` | 元数据 | 窄化未检查 |
| 同上 | `ubFormer` | int32/32 | off 12 | 元素/tile | Device `RunTargetAdamW←ub_former` | scheduler `Init/Process` | 需为正 |
| 同上 | `blockFormer` | int64/64 | off 16 | 元素/非尾核 | Device `RunTargetAdamW←block_former` | scheduler GM core offset | 尾核另算 |
| 同上 | `blockNum` | int64/64 | off 24 | core | Device `RunTargetAdamW←num_blocks` | scheduler 尾核判断 | 与 launch grid 相同 |
| 同上 | `ubLoopOfFormerBlock` | int64/64 | off 32 | tile loop | Device `RunTargetAdamW` 计算 `ceil(block_former/ub_former)` | scheduler 非尾核循环 | 至少 1 |
| 同上 | `ubLoopOfTailBlock` | int64/64 | off 40 | tile loop | Device `RunTargetAdamW` 计算 `ceil(tail_block/ub_former)` | scheduler 尾核循环 | 至少 1 |
| 同上 | `ubTailOfFormerBlock` | int64/64 | off 48 | 元素 | Device 计算 remainder，整除时为 `ub_former` | scheduler former 最后 tile | 非零 |
| 同上 | `ubTailOfTailBlock` | int64/64 | off 56 | 元素 | Device 计算 tail remainder | scheduler tail 最后 tile | 非零 |
| 同上 | `elemNum` | int64/64 | off 64 | 元素 | Device `RunTargetAdamW←total_length` | 元数据 | 与 dim0 相同 |
| 同上 | `scheMode` | uint64/64 | off 72 | 枚举 | Device 常量 1 | scheduler 模板固定为 1，字段不参与分派 | 固定值 |

### Route 源码闭包

| 重要性 | Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---:|---|---|---|---|---|---|---|
| P0 | R-002 | `{f32,f16,bf16}_wd1_min` | 转换件 `apply_adam_w_npu/CalcTiling` | direct scalar ABI→本地 `EleBaseTilingData` | `launch_apply_adam_w_direct`→三个 wd1_min entry | `RunTargetAdamW<T,true,false>`→`CalcTargetAdamW`→ATVOSS | `apply_adam_w/CMakeLists.txt::register_direct_launch_op` |
| P1 | R-001 | `{f32,f16,bf16}_wd0_min` | 同上 | 同上 | 三个 wd0_min entry | `RunTargetAdamW<T,false,false>` | 同上 |
| P2 | R-004 | `{f32,f16,bf16}_wd1_max` | 同上 | 同上 | 三个 wd1_max entry | `RunTargetAdamW<T,true,true>` | 同上 |
| P3 | R-003 | `{f32,f16,bf16}_wd0_max` | 同上 | 同上 | 三个 wd0_max entry | `RunTargetAdamW<T,false,true>` | 同上 |

## Kernel 实现

### 公共执行流程

12 个 entry 只改变 `T/HasWeightDecay/Maximize` 模板常量，均启动一个 AIV kernel。`RunTargetAdamW` 先构造本地 tiling，再以指数平方法各求一次 `beta1^step/beta2^step`，形成四个 bias-correction 系数，写入 scheduler 的 Var 1～4；lr/weight_decay/epsilon 写 Var 0/5/6。

ATVOSS 对每 tile 按 `CopyIn(var,grad,m,v)→CalcTargetAdamW→CopyOut(y)` 的 DAG 顺序执行。四个输入各用 depth-2（ping/pong）等长子区；计算结果直接占用 output 的 depth-2 子区，没有额外 temp tensor。每个 tile 内四路 T 值加载寄存器，低精度先 unpack/cast 到 FP32，然后依次做 moment、sqrt/div、可选 decay 与最终 Axpy，最后只写 `y`。

### Route R-002（P0）：有 decay / minimize

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | 转换件 `apply_adam_w_plugin.cpp` | `apply_adam_w_npu` | `weight_decay != 0.0 && !maximize` | `has_weight_decay=true` |
| Launcher | 转换件 `apply_adam_w_kernel.cpp` | `launch_apply_adam_w_direct` | dtype 分支内 `has_weight_decay && !maximize` | `*_wd1_min` |
| Entry | 同文件 | `DEFINE_TARGET_KERNEL` | `<T,true,false>` | R-002 body |

#### Device 数据流

##### 函数级调用链

```text
apply_adam_w_{dtype}_wd1_min -> RunTargetAdamW<T,true,false>
  -> scheduler.Process -> CopyIn×4 -> CalcTargetAdamW
    -> moment/denom -> Axpy(update,var,weight_decay) -> Axpy(var,update,-lr)
  -> CopyOut(y)
```

##### 源码等价伪码

```text
沿用公共 core/tile/repeat、load、moment 和 denom；
update = mHat/(sqrt(vHat)+epsilon)
update += weight_decay*var
y = var - lr*update
按 T 的 FP32 或 promote/RINT 路径写出有效 tail。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM 与 10 个 UB 子区 | 公共 core/tile GM offset | `5*2*ub_former*S` | T/flat ND | CopyIn/Calc | Calc/CopyOut | Tensor/TPipe | ping/pong、256B tile 对齐 |
| `reg_update` | 寄存器 | 64 FP32 lane | float | Div | decay Axpy→final Axpy | 单 repeat | 原地加入 `weight_decay*reg_var` |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  varQ["var · depth=2 · 2*ub_former*S"]:1
  gradQ["grad · depth=2 · 2*ub_former*S"]:1
  mQ["m · depth=2 · 2*ub_former*S"]:1
  vQ["v · depth=2 · 2*ub_former*S"]:1
  yQ["y · depth=2 · 2*ub_former*S"]:1
```

##### 字节账与复用顺序

```text
UB_total = 10*ub_former*S <= U
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/y ping-pong | 公共 CopyIn/Calc | 公共 Calc/CopyOut | 下个同 parity tile | ATVOSS Get/ReleaseTensor | producer-consumer 已完成 |
| `reg_update` | `Div` | decay Axpy、final Axpy | 下一 repeat | Vector 程序顺序 | `reg_var` 在最终 Axpy 前保持原值 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/调度 | 公共 `DataCopyPad` 与 Get/ReleaseTensor 协议 | GM/Calc | Calc/GM | 10 个子区 | GM 搬运只提交 `tileLength*S`；raw Reg load 可读满 UB span，live mask 从 cast/算术延续到写回 |
| decay | `Axpy(reg_update,reg_var,weight_decay,mask)` | Div 与原始 var | final Axpy | update RegTensor | FP32，tail mask 一致 |
| 降精度/写回 | 公共 RINT/pack 与 `DataCopyPad` 路径 | final Axpy | GM y | y 子区 | FP16/BF16 仅最终一次 RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | 转换件 `apply_adam_w_plugin.cpp` | `weight_decay != 0.0` | 选择 wd1 |
| Kernel entry/router | 转换件 `apply_adam_w_kernel.cpp` | `*_wd1_min` | `<T,true,false>` |
| Device body/helper | 同文件 | `CalcTargetAdamW<T,true,false>` | decay 与 minimize 两次 Axpy 顺序 |

### Route R-001（P1）：无 decay / minimize

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | 转换件 `op_plugin/apply_adam_w_plugin.cpp` | `apply_adam_w_npu` | `weight_decay==0.0 && maximize==false` | `has_weight_decay=false` |
| Launcher | 转换件 `op_kernel/apply_adam_w_kernel.cpp` | `launch_apply_adam_w_direct` | dtype 外层分支后 `!has_weight_decay && !maximize` | `apply_adam_w_{dtype}_wd0_min` |
| Entry | 同文件 | `DEFINE_TARGET_KERNEL` | `<T,false,false>` | R-001 body |

#### Device 数据流

##### 函数级调用链

```text
apply_adam_w_{f32|f16|bf16}_wd0_min(args)
  -> RunTargetAdamW<T,false,false>
    -> DeviceIntegerPower × 2 -> SetVar × 7 -> scheduler.Init
    -> ElementwiseSch::Process
      -> CopyIn(var,grad,m,v) -> CalcTargetAdamW<T,false,false> -> CopyOut(y)
```

##### 源码等价伪码

```text
for tile in this core's [blockIdx*block_former, block_length):
  DMA var/grad/m/v GM -> four ping-or-pong UB regions
  for 64-lane repeat in valid tile:
    mask = UpdateMask<float>(remaining)
    raw-load four full register spans from the allocated UB regions
    if T is FP16/BF16: unpack, then cast to FP32 under mask
    mReg = beta1*c1*mReg + (1-beta1)*c1*gradReg
    vReg = beta2*c2*vReg + ((1-beta2)*c2*gradReg)*gradReg
    update = mReg / (sqrt(vReg)+epsilon)
    varReg += (-lr) * update
    store varReg to y region; low precision uses RINT cast + pack
  DMA only tileLength valid y elements UB -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs | `blockIdx*block_former + tileOffset` | `tileLength` 各 1 路 | T/flat ND | Tensor | CopyIn | 调用方 Tensor | `DataCopyPad`, blockCount=1, blockLen=`tileLength*S` B |
| GM y | 同上 | `tileLength` | T/flat ND | CopyOut | Tensor consumer | 输出 Tensor | 只写有效元素 |
| var ping/pong | scheduler buffer id | 每区 `ub_former*S` B | T | CopyIn | Calc | TPipe/相邻 tile | 两区交替 |
| grad ping/pong | 同上 | 每区 `ub_former*S` B | T | CopyIn | Calc | 同上 | 两区交替 |
| m ping/pong | 同上 | 每区 `ub_former*S` B | T | CopyIn | Calc | 同上 | 两区交替 |
| v ping/pong | 同上 | 每区 `ub_former*S` B | T | CopyIn | Calc | 同上 | 两区交替 |
| y ping/pong | 同上 | 每区 `ub_former*S` B | T | Calc | CopyOut | 同上 | 计算结果直接作为 MTE3 源 |
| RegTensor | 寄存器 | 4 输入+3 中间/结果 FP32，并有低精度 pack 临时 | register layout | raw load/Vector API | store | 单 repeat | raw load 读完整 UB span；mask 保护有效算术 lane 与发布 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  varQ["var · depth=2 · 2*ub_former*S"]:1
  gradQ["grad · depth=2 · 2*ub_former*S"]:1
  mQ["m · depth=2 · 2*ub_former*S"]:1
  vQ["v · depth=2 · 2*ub_former*S"]:1
  yQ["y · depth=2 · 2*ub_former*S"]:1
```

##### 字节账与复用顺序

```text
BufferNum = 4 inputs*2 + 1 output*2 = 10
blockLen = ub_former*S
UB_total = 10*ub_former*S <= U  # 正常 U 足以容纳至少一个 256B 对齐 tile
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每个 input ping/pong | `Vec::CopyIn` | `CalcTargetAdamW` | 下一个同 parity tile | `GetTensor/ReleaseTensor` 的 VECIN→VECCALC 协议 | calc 已释放该 buffer id |
| y ping/pong | `CalcTargetAdamW` | `Vec::CopyOut` | 下一个同 parity tile | VECCALC→VECOUT 协议 | MTE3 已释放该 buffer id |
| `reg_m/reg_v/reg_var` | load 后原地 Vector 更新 | 后续同 repeat API/store | 下一 repeat | `__VEC_SCOPE__` 程序顺序 | mask 一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + scheduler `Get/ReleaseTensor<VECIN>` | GM | VECCALC | 四个输入子区 | blockLen 为有效字节；无效 padding 不被 mask store |
| UB→Reg | `Reg::DataCopy`; 低精度 `UnPack/Cast` | input UB | FP32 算术 | 当前 repeat | raw load 固定满跨度；低精度 promote cast 受 live mask；FP32 直载后仅有效 lane 参与算术 |
| 计算 | MicroAPI 程序顺序 | FP32 RegTensor | store | moment/update | `UpdateMask<float>(remaining)` 保护 64-lane 尾部 |
| Reg→UB→GM | 低精度 `Cast(CAST_RINT)/Pack/MaskPack`；`DataCopyPad` | FP32 result | GM y | output 子区 | FP32 直写；只提交 tileLength |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | 转换件 `apply_adam_w_plugin.cpp`/`apply_adam_w_kernel.cpp` | `apply_adam_w_npu`、`CalcTiling<T>` | 校验、空域、分核、UB tile |
| Kernel entry/router | 转换件 `apply_adam_w_kernel.cpp` | `*_wd0_min`、`launch_apply_adam_w_direct` | dtype×布尔到单 entry |
| Device body/helper | 同文件 | `RunTargetAdamW<T,false,false>`、`CalcTargetAdamW`、`LoadAsFloat/StoreFromFloat` | 系数、寄存器公式、mask/tail |

### Route R-004（P2）：有 decay / maximize

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | 转换件 `apply_adam_w_plugin.cpp` | `apply_adam_w_npu` | `weight_decay!=0.0 && maximize` | wd1/max flags |
| Launcher | 转换件 `apply_adam_w_kernel.cpp` | `launch_apply_adam_w_direct` | `has_weight_decay && maximize` | `*_wd1_max` |
| Entry | 同文件 | `DEFINE_TARGET_KERNEL` | `<T,true,true>` | R-004 body |

#### Device 数据流

##### 函数级调用链

```text
apply_adam_w_{dtype}_wd1_max -> RunTargetAdamW<T,true,true>
  -> scheduler.Process -> CopyIn×4 -> CalcTargetAdamW
    -> moment/denom -> decay Axpy -> Axpy(var,update,+lr)
  -> CopyOut(y)
```

##### 源码等价伪码

```text
沿用 R-002 的 decay pipeline，但 signed_lr=+lr：
update = mHat/(sqrt(vHat)+epsilon) + weight_decay*var
y = var + lr*update
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | 同 R-002 | `10*ub_former*S` | T/flat ND | scheduler/body | body/GM | Tensor/TPipe | ping/pong |
| `reg_update/reg_var` | register | 各 64 FP32 lane | float | Div/load | decay+positive Axpy/store | 单 repeat | 先读原 var 进 update，再原地改 var |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  varQ["var · depth=2 · 2*ub_former*S"]:1
  gradQ["grad · depth=2 · 2*ub_former*S"]:1
  mQ["m · depth=2 · 2*ub_former*S"]:1
  vQ["v · depth=2 · 2*ub_former*S"]:1
  yQ["y · depth=2 · 2*ub_former*S"]:1
```

##### 字节账与复用顺序

```text
UB_total = 10*ub_former*S <= U
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/y ping-pong | 同 R-001 | 同 R-001 | 下个同 parity tile | ATVOSS Get/ReleaseTensor | 同 R-001 |
| register values | load/Div | decay Axpy→positive Axpy→store | 下一 repeat | Vector 程序顺序 | 除 raw load 外，cast/算术/masked store 使用同一 live mask |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/寄存器 | 同 R-001 | 同 R-001 | 同 R-001 | input/output | 同 R-001 |
| decay+maximize | 两个顺序 `Axpy` | Div、var、weight_decay、lr | store | `reg_update/reg_var` | FP32；同 mask；仅最终一次低精度 RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | 转换件 `apply_adam_w_plugin.cpp` | `has_weight_decay/maximize` | 两个 true flag |
| Kernel entry/router | 转换件 `apply_adam_w_kernel.cpp` | `*_wd1_max` | `<T,true,true>` |
| Device body/helper | 同文件 | `CalcTargetAdamW<T,true,true>` | decay 后正向最终更新 |

### Route R-003（P3）：无 decay / maximize

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | 转换件 `apply_adam_w_plugin.cpp` | `apply_adam_w_npu` | `weight_decay==0.0 && maximize` | 两个布尔传 launcher |
| Launcher | 转换件 `apply_adam_w_kernel.cpp` | `launch_apply_adam_w_direct` | `!has_weight_decay && maximize` | `*_wd0_max` |
| Entry | 同文件 | `DEFINE_TARGET_KERNEL` | `<T,false,true>` | R-003 body |

#### Device 数据流

##### 函数级调用链

```text
apply_adam_w_{dtype}_wd0_max -> RunTargetAdamW<T,false,true>
  -> scheduler.Process -> CopyIn×4 -> CalcTargetAdamW
    -> moment/denom -> Axpy(var,update,+lr)
  -> CopyOut(y)
```

##### 源码等价伪码

```text
沿用 R-001 的全部搬运、moment 和 mask；不执行 decay Axpy；
signed_lr = +lr
y = var + lr*(mHat/(sqrt(vHat)+epsilon))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | 同 R-001 | `10*ub_former*S` | T/flat ND | scheduler | body/GM | Tensor/TPipe | ping/pong |
| `signed_lr` | Device scalar | 1 FP32 | float | 模板常量 `Maximize=true` 选择 `lr` | final Axpy | kernel | 不占 UB 子区 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  varQ["var · depth=2 · 2*ub_former*S"]:1
  gradQ["grad · depth=2 · 2*ub_former*S"]:1
  mQ["m · depth=2 · 2*ub_former*S"]:1
  vQ["v · depth=2 · 2*ub_former*S"]:1
  yQ["y · depth=2 · 2*ub_former*S"]:1
```

##### 字节账与复用顺序

```text
UB_total = 10*ub_former*S <= U
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/y ping-pong | 同 R-001 | 同 R-001 | 下个同 parity tile | ATVOSS Get/ReleaseTensor | 同 R-001 |
| `reg_var` | load | final positive Axpy→store | 下一 repeat | Vector 程序顺序 | decay 分支编译期删除 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/寄存器 | 同 R-001 | 同 R-001 | 同 R-001 | input/output | 同 R-001 |
| maximize | `signed_lr=lr` + final `Axpy` | update | store | `reg_var` | 只改变最终符号，不反转 grad；mask 同前 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | 转换件 `apply_adam_w_plugin.cpp` | `maximize` | 直接传 bool |
| Kernel entry/router | 转换件 `apply_adam_w_kernel.cpp` | `*_wd0_max` | `<T,false,true>` |
| Device body/helper | 同文件 | `CalcTargetAdamW<T,false,true>` | 正 `signed_lr`，无 decay |

## 算子特定优化

### Trick T-001：低精度只在边界转换、主体统一 FP32 寄存器流水

#### 触发与源码依据

- 适用 Route：按重要性依次为 R-002、R-001、R-004、R-003 的 FP16/BF16 entry。
- 触发条件：模板 `T` 非 float；每个 tile/repeat 均使用相同有效 lane mask。
- 固定官方源码依据：[`optim/apply_adam_w/op_kernel/arch35/apply_adam_w_dag.h:44-57`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w/op_kernel/arch35/apply_adam_w_dag.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T, typename U = float>
__aicore__ inline void LoadOneTensor(MicroAPI::RegTensor<U>& dst, __local_mem__ T*& input, MicroAPI::MaskReg& pregUp,
                                     int32_t oneRepeat)
{
    MicroAPI::RegTensor<T> regTmp;
    MicroAPI::RegTensor<T> regCopyIn;
    if constexpr (IsSameType<U, float>::value && !IsSameType<T, U>::value) {
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regCopyIn, input, (int32_t)oneRepeat);
        MicroAPI::UnPack((RegTensor<int32_t>&)regTmp, (RegTensor<int16_t>&)regCopyIn);
        MicroAPI::Cast<U, T, castTrait0>(dst, regTmp, pregUp);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dst, input, (int32_t)oneRepeat);
    }
}
```

#### 机制与设计原因

Direct 交付件保留这一 950 idiom：FP16/BF16 从已分配的完整 UB 子区以 packed T 满跨度搬入寄存器，`UnPack` 后在 `ZEROING` live mask 下转换为 FP32；moment、平方、`sqrt`、除法、decay 与最终 Axpy 全部保持 FP32，最后才以 `NO_SAT + CAST_RINT` 降回 T 并 `Pack/MaskPack`。这样不会在长算术链的每一步反复降精度，也不需要为 FP32 中间值物化额外 UB tensor；关键实参 `oneRepeat=256/4=64` 来自 950 256B vector register，mask 来自当前 tile 的 `remaining`。注意 raw `DataCopy` 本身没有 mask 参数：它只能读取已分配 UB，GM tail 的有效字节仍由 scheduler 的 `DataCopyPad(tileLength)` 限定。

#### 迁移边界

- 适用：950 上输入/输出同为 FP16 或 BF16、主体适合 FP32 register elementwise 流水的单输出算子。
- 前提：UB 中 T 数据连续且子区至少覆盖一个完整寄存器读取跨度；每个 repeat 的 live mask 覆盖低精度 promote cast、全部算术与最终 masked store；最后 cast 的舍入要求允许 `CAST_RINT`。
- 失效：FP32 entry 应直载直存；需要保持中间状态为低精度、不同 round/saturation，或需要跨 tile 中间张量时必须另建实现。
- 证据边界：本卡仅陈述固定来源可证明的接口、Route 与实现事实，不主张编译、真机、数值或性能结论。

#### 错误做法

固定仓可验证的边界是：低精度路径若跳过 `UnPack/Pack/MaskPack`，寄存器 lane 布局和尾 mask 都不再等价；固定仓未提供其它可验证反例。

## 模板与实现对比

| 对比维度 | P0 · R-002 有 decay/min | P1 · R-001 无 decay/min | P2 · R-004 有 decay/max | P3 · R-003 无 decay/max | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | wd!=0,false | wd=0,false | wd!=0,true | wd=0,true | Host 双布尔 specialization |
| 模板与 dtype | `<T,true,false>` | `<T,false,false>` | `<T,true,true>` | `<T,false,true>` | 各有 float/half/bfloat16_t entry |
| 循环与分核 | 相同 | 相同 | 相同 | 相同 | 只依赖 N、AIV core、UB、T 大小 |
| Queue/TBuf/寄存器 | 10 子区；多 decay Axpy | 同 UB；无 decay 指令 | 同 R-002；正 lr | 同 R-001；正 lr | 布尔分支不增加 UB |
| 内存总占用与复用 | `10*ub_former*S` | 同 | 同 | 同 | LEVEL_2 直接 output buffer |
| 搬运/Vector API | 4 load、moment、sqrt/div、decay Axpy、final Axpy | 编译期删除 decay Axpy | R-002 但正向 final Axpy | R-001 但正向 final Axpy | 编译期删除不需要的 decay |
| 同步、精度与 tail | 共同 ATVOSS 协议；T 决定 cast | 同 | 同 | 同 | Route 不改变 tile/mask |
| fallback/失败边界 | 无 Device fallback | 无 | 无 | 无 | Host 校验后精确 entry |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方同名 README 声明支持，但本文 Direct 交付件未提取 | 官方同名 README 声明支持，但本文 Direct 交付件未提取 | 本卡目标；官方 README 与 `SUPPORT_COMPUTE_UNIT=ascend950` 闭合 |
| Host/Launcher | 固定交付件不可见 | 固定交付件不可见 | `PlatformAscendCManager` 查询 AIV core/UB，Direct 选择 12 entries |
| Kernel/资源 | 不外推 | 不外推 | DAV-3510 AIV、256B vReg、ATVOSS LEVEL_2、无 workspace |

## 限制与失败边界

- 仅支持四输入 shape/dtype 完全相同、rank 1～8、FP32/FP16/BF16；非空输入还必须全部连续。空域在 contiguous 检查之前直接返回。
- 非空执行要求四输入位于 `var` 所在的同一 NPU device；Host 没有显式比较 `grad/m/v.device()`，混合 device 可能把不可由当前 stream 安全访问的地址传入 kernel，不属于受支持 Route。
- 只写 `y`，不更新 `var/m/v`，无 AMSGrad、`max_grad_norm` 或 beta-power Tensor；不得用官方公开 ApplyAdamW 的原位状态语义替换本文 Direct 语义。
- `lr/beta1/beta2/weight_decay/epsilon` 从 double 窄化到 float；没有有限值或范围检查。`step=0`、`beta1==1`、`beta2==1` 可令 bias-correction 分母为 0；负 step 虽由整数快速幂实现支持，但其优化器有效域未被校验。
- `weight_decay != 0.0` 在 double 精度决定 Route，Device 收到的是 float；极小非零 double 可能在窄化后变 0，但仍走 wd1 body。NaN 也走 wd1。
- Direct router 把所有非 0/1 的 `dtype_code` 当 BF16；正常 plugin 已先限制 dtype，但外部直接调用 launcher 不具备独立防护。
- UB 公式在 `ub_former<A` 时强制一个 256B tile，没有显式复核 `10*A*S<=U`；950 正常 UB 满足，但不能把该兜底外推到任意平台。`num_blocks/coreNum` 与 `ub_former` 存在 int32 窄化，源码无溢出检查。
- launcher/lambda 不检查异步 kernel 返回码。固定源码不含性能数字，本卡不主张编译、真机、数值或性能结论。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `optim/apply_adam_w/README.md` | 产品支持表、公式与参数表 | 950 支持及官方同名语义转换边界 |
| Host/Tiling | `optim/apply_adam_w/op_host/arch35/apply_adam_w_tiling.cpp` | `DoElewiseTiling/SetTilingData/TilingPrepareForApplyAdamW` | 官方 950 AIV/UB elementwise tiling 机制来源 |
| Kernel entry | `optim/apply_adam_w/op_kernel/arch35/apply_adam_w.cpp` | `apply_adam_w<schMode,amsgrad,dType>` | AIV-only、ATVOSS scheduler 与官方 ABI 差异 |
| Kernel body | `optim/apply_adam_w/op_kernel/arch35/apply_adam_w_dag.h` | `LoadOneTensor/StoreOneTensor/ApplyAdamWDAG` | 低精度 FP32 寄存器转换与 LEVEL_2 DAG 来源 |
| 构建接线 | `optim/apply_adam_w/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950/arch35 固定接线 |

每行文件都对应 Frontmatter 中的固定 Commit HTTP(S) blob URL。Direct 交付件的 plugin、launcher、entry 与 fused body 是本文技术事实主源，但不是官方仓文件，故不伪造为 Frontmatter 官方 source；上表官方源码只用于转换来源、平台和机制审计。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | direct scalar 参数后在 Device 构造 80B `EleBaseTilingData` | 参数顺序、dtype_code、布尔 specialization、字段位宽/单位 | `launch_apply_adam_w_direct/RunTargetAdamW` |
| 调用顺序 | Meta→tiling→单 entry→系数→Init→Process | 空域先返回；四个系数必须在每核 scheduler 前形成 | `apply_adam_w_npu/RunTargetAdamW` |
| workspace/内存 | workspace 0；10 个等长 ATVOSS 子区 | `UB_total=10*ub_former*S`、256B 对齐、ping/pong lifetime | `CalcTiling/ElementwiseSch::Init` |
| 同步 | scheduler 的 VECIN/VECCALC/VECOUT Get/ReleaseTensor 协议 | producer 完成前不得让同 buffer id 被同 parity tile 覆盖 | `ElementwiseSch::CopyIn/RunNormalOp/CopyOut` |
| 精度与尾块 | FP16/BF16 边界 cast、主体 FP32、最终 RINT；64-lane mask | raw load 只能读已分配 UB span；`UpdateMask` 必须贯穿 promote cast/compute/masked store；GM 只写有效 `tileLength` | `LoadAsFloat/CalcTargetAdamW/StoreFromFloat` |
| 失败边界 | plugin 拒绝 shape/dtype/rank/contiguous 错误 | 不得把 invalid dtype launcher default、无 launch 检查当作安全 fallback | `apply_adam_w_meta/apply_adam_w_npu/launch_apply_adam_w_direct` |