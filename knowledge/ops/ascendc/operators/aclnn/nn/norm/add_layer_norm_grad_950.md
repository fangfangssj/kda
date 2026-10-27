---
type: Operator
title: "AddLayerNormGrad 950 ACLNN 算子知识"
description: "说明 aclnnAddLayerNormGrad 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐路线实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/docs/aclnnAddLayerNormGrad.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_host/add_layer_norm_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_host/add_layer_norm_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_host/add_layer_norm_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad_cut_n.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad_cut_d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_kernel/add_layer_norm_determinstic_compute.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad_utils.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_host/config/ascend950/add_layer_norm_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T17:43:20Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# AddLayerNormGrad 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnAddLayerNormGradGetWorkspaceSize` + `aclnnAddLayerNormGrad`，公开 ACLNN 两段式接口，无其它公开版本 |
| 计算语义 | 令 `x=x1+x2`；输出 `dx=rstd*(dy*gamma-mean(dy*gamma)-xhat*mean(dy*gamma*xhat))+dsumOptional`、`dgamma=sum(dy*xhat)`、`dbeta=sum(dy)` |
| 输入输出 | `dy/x1/x2/gamma/[dsum]/dx` 为 FP32/FP16/BF16、ND；`mean/rstd/dgamma/dbeta` 为 FP32；归一化轴是 `gamma` 对应的后缀维 |
| 执行类型 | 单 Kernel `add_layer_norm_grad`，全部路线均为 `AIV_VECTOR`；确定性模式在同一 Kernel 内追加 workspace 归并阶段 |
| Kernel 路线 | R-001～R-006 为 Cut-N，R-007～R-012 为 Cut-D；每组按 FP32/FP16/BF16 和 dsum 无/有拆分 |
| 核心资源 | GM、单深度 `TQue`、FP32 `TBuf`、原子加、全核同步；确定性模式另用逐核 GM workspace 与双缓冲归并队列 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档和 ACLNN 构建声明经 `AddLayerNormGrad` OpDef、`Tiling4AddLayerNormGrad`、POD、router、两类 body/helper 到 950 binary/build 接线闭合；生成式 ACLNN inner bridge 属框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

将 `dy/x1/x2` 与输出 `dx` 展平为 `[N,D]`，其中 `D=prod(gamma.shape)`，`N=prod(dy.shape[:-rank(gamma)])`，`xhat=(x1+x2-mean)*rstd`。逐行计算：

```text
g = dy * gamma
dx = rstd * (g - sum(g)/D - xhat*sum(g*xhat)/D) + (dsum if supplied else 0)
dgamma[d] = sum_n(dy[n,d] * xhat[n,d])
dbeta[d]  = sum_n(dy[n,d])
```

Kernel 实际把 FP16/BF16 路线提升到 FP32 完成上述运算；FP16 `dx` 以 `CAST_NONE` 降精度，BF16 以 `CAST_RINT` 降精度。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `dy` | 1～8 维 ND、非空 | FP32/FP16/BF16 | 主梯度；与 `x1/x2/dx` shape、dtype 相同 |
| `x1`, `x2` | 同 `dy` | 同 `dy` | 正向 Add 两输入 |
| `rstd`, `mean` | rank 同 `dy`；前 `rank(dy)-rank(gamma)` 维同 `dy`，其余为 1 | FP32 | 每个 N 行一个标量 |
| `gamma` | 1～8 维 ND；等于 `dy` 后缀 shape | 同 `dy` | LayerNorm 缩放参数 |
| `dsumOptional` | 若存在则同 `dy` | 同 `dy` | 可选附加梯度；nullptr 选择无 dsum key |
| `dxOut` | 同 `dy` | 同 `dy` | `x1+x2` 的梯度 |
| `dgammaOut`, `dbetaOut` | 同 `gamma` | FP32 | 沿 N 归约结果 |

### 入口与版本

公开文档只声明无版本后缀的 `aclnnAddLayerNormGrad`；第一段返回 workspace 与 executor，第二段在调用者提供的 stream 上执行。`op_host/CMakeLists.txt` 以 `OPTYPE add_layer_norm_grad ACLNNTYPE aclnn` 接入生成式 ACLNN 框架；固定仓未公开生成后的 inner bridge 源码，连续化、executor 封装和 stream 提交按框架边界处理，但算子自有 OpDef/Tiling/POD/Kernel 链路均可见。README 与 API 文档明确支持 Ascend 950PR/950DT，OpDef 注册 `ascend950`，对应 binary config 和全局 `ascendc_config.json` 均接线，无发布冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

OpDef 对七个输入启用 `AutoContiguous`。公开第一段接口检查必选指针和 dtype；Tiling 再检查：`x1/gamma` rank 均为 1～8，shape 不含 0，`dy/x1/x2/dx` 全等，`mean/rstd` 全等且 rank 同 `dy`，`gamma/dgamma/dbeta` shape 等于 `dy` 后缀，`mean/rstd` 的归一化后缀维全为 1。`CheckAndSetDtype` 仅接受 FP32/FP16/BF16。任一失败返回 `GRAPH_FAILED`，不启动 Kernel。

### 参数、Workspace 与生命周期

`SetSocInfo` 读取 AIV 核数 `Cmax` 与 UB 容量 `UBmax`。`N,D` 的 shape 乘积先窄化存入 `uint32_t`，源码未做乘法溢出检查。Host 固定上报 `16 MiB` 系统 workspace + `20 MiB` 用户 workspace；确定性开启时再加 `numCore*roundUp(D,8)*4*2` 字节。Kernel 通过 `GetUserWorkspace` 取得用户区；非确定性计算不使用逐核归约区，确定性计算在用户区起始位置依次放每核 gamma/beta FP32 切片。workspace/executor 由第一段创建并保持到第二段 stream 提交结束。

### 启动图与失败传播

```text
aclnnAddLayerNormGradGetWorkspaceSize
  -> generated ACLNN framework boundary / AddLayerNormGrad OpDef
  -> Tiling4AddLayerNormGrad -> AddLayerNormGradTilingData + key + blockDim=Cmax
aclnnAddLayerNormGrad(workspace, executor, stream)
  -> add_layer_norm_grad<<<Cmax, stream>>>
       -> CutNProcess 或 CutDProcess
       -> [isDeterministicKey] SyncAll -> pipe.Reset -> FinalProcessDeterministic
  -> dx / dgamma / dbeta
```

只有一个 Device 节点。`numCore<=Cmax` 个核处理数据，但发射 `Cmax` 个 block；所有 block 参与初始化和 `SyncAll`，仅 `blockIdx<numCore` 进入主计算。非确定性时各有效核直接原子累加输出；确定性时每核先写独占 workspace 两个切片，全核同步后按 64 列任务做固定顺序归并。Kernel launch 失败由 ACLNN executor 返回并阻断该次执行；固定仓没有额外 cleanup 节点。

## Route 与参数设计

### 分核与资源计算

记 `C=maxAivCore`，`numCore=ceil(N/ceil(N/C))`，普通核行数 `Ln=ceil(N/numCore)`，尾核 `Lt=N-Ln*(numCore-1)`。Host 先保留 `512+512+256=1280` UB 字节，`U=UBmax-1280`。

- FP32 Cut-N：`free=U/4-4*roundUp(D,8)`，`Navail=floor(free/(roundUp(D,8)*(5+hasDsum)+16))`。
- FP16/BF16 Cut-N：按源码常量采用保守 coexist 公式：先扣 `3*roundUp(D,8)*4 + roundUp(D,16)*2`，分母为 `2*roundUp(1,8)*4 + (4+hasDsum)*roundUp(D,8)*4 + (4+hasDsum)*roundUp(D,16)*2`。
- 若 `Navail<1` 或有符号 `free<0`，转 Cut-D；FP32 `dInner=roundDown((U/4-16)/(4+5+hasDsum),8)`，低精度 `dInner=roundDown(U/((4+hasDsum+3)*4+(4+hasDsum+1)*2),16)`。
- `dOuter=ceil(D/dInner)`，尾段 `Dtail=D-dInner*(dOuter-1)`。Cut-N 每次载入 `min(本核剩余行,Navail)` 行；Cut-D 每行对 D 做两遍分段扫描。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 Cut-N FP32 | 10 / `KernelAddLayerNormGrad<float,10>` | dtype FP32，Cut-N 容量成立，dsum 缺失 | 同 dtype 的 R-007 或失败 |
| 2 | R-002 Cut-N FP32+dsum | 11 / `<float,11>` | FP32，Cut-N 容量成立，dsum 存在 | R-008 或失败 |
| 3 | R-003 Cut-N FP16 | 20 / `<half,20>` | FP16，Cut-N 容量成立，dsum 缺失 | R-009 或失败 |
| 4 | R-004 Cut-N FP16+dsum | 21 / `<half,21>` | FP16，Cut-N 容量成立，dsum 存在 | R-010 或失败 |
| 5 | R-005 Cut-N BF16 | 30 / `<bfloat16_t,30>` | BF16，Cut-N 容量成立，dsum 缺失 | R-011 或失败 |
| 6 | R-006 Cut-N BF16+dsum | 31 / `<bfloat16_t,31>` | BF16，Cut-N 容量成立，dsum 存在 | R-012 或失败 |
| 7 | R-007 Cut-D FP32 | 40 / `KernelAddLayerNormGradLarge<float,40>` | FP32，Cut-N 容量不成立，dsum 缺失 | 无；非法 dtype/shape 失败 |
| 8 | R-008 Cut-D FP32+dsum | 41 / `<float,41>` | FP32，Cut-N 容量不成立，dsum 存在 | 失败 |
| 9 | R-009 Cut-D FP16 | 50 / `<half,50>` | FP16，Cut-N 容量不成立，dsum 缺失 | 失败 |
| 10 | R-010 Cut-D FP16+dsum | 51 / `<half,51>` | FP16，Cut-N 容量不成立，dsum 存在 | 失败 |
| 11 | R-011 Cut-D BF16 | 60 / `<bfloat16_t,60>` | BF16，Cut-N 容量不成立，dsum 缺失 | 失败 |
| 12 | R-012 Cut-D BF16+dsum | 61 / `<bfloat16_t,61>` | BF16，Cut-N 容量不成立，dsum 存在 | 失败 |

```text
validate -> dtypeKey(1/2/3) -> compute Cut-N capacity
  -> fits: key=dtypeKey*10
  -> not fits: key=dtypeKey*10+30
  -> dsum exists: key += 1
  -> deterministic 只写 POD 字段，不改变 key
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001/R-002 | 10/11 | FP32、Cut-N fit、950、dsum 无/有 | 单 AIV Kernel | `KernelAddLayerNormGrad<float,key>::CutNProcess/MainCompute` | 行 tile；可选 dsum queue；输出原子加 | N 尾 tile + 非对齐 D 安全搬运；否则 40/41 |
| R-003/R-004 | 20/21 | FP16、Cut-N fit、dsum 无/有 | 同上 | `<half,key>` | FP32 TBuf 升精度，`CAST_NONE` 输出 | 否则 50/51 |
| R-005/R-006 | 30/31 | BF16、Cut-N fit、dsum 无/有 | 同上 | `<bfloat16_t,key>` | FP32 TBuf，`CAST_RINT` 输出 | 否则 60/61 |
| R-007/R-008 | 40/41 | FP32、Cut-N 不 fit、dsum 无/有 | 单 AIV Kernel、每行两遍 D | `KernelAddLayerNormGradLarge<float,key>` | 第一遍原子累计 dgamma/dbeta 并累积标量，第二遍写 dx | Dtail；无其它 fallback |
| R-009/R-010 | 50/51 | FP16、Cut-D、dsum 无/有 | 同上 | `<half,key>` | FP32 计算 TBuf + FP16 queues | `CAST_NONE`、Dtail |
| R-011/R-012 | 60/61 | BF16、Cut-D、dsum 无/有 | 同上 | `<bfloat16_t,key>` | FP32 计算 TBuf + BF16 queues | `CAST_RINT`、Dtail |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Cut-N UB fit | 上述 `Navail>=1 && free>=0` | 至少容纳一行输入/输出、gamma/归约结果和保留区 | R-001～R-006 | Cut-D | `AddLayerNormGradTilingImpl` |
| 32B 对齐 | FP32 `roundUp(*,8)`；FP16/BF16 `roundUp(*,16)` | DMA block 与 Queue 容量 | 全部 | `DataCopyPad` 或回退尾块搬运 | `CalNInUb`, `SafeDataCopy`, `DataCopyCustom` |
| BF16 编译 | `__CCE_AICORE__==220 \|\| 310` | router 只在该 guard 下实例化 BF16 | R-005/006/011/012 | 未知 key 无输出 | `add_layer_norm_grad` |
| 有效核 | `blockIdx<numCore` | launch 使用 `Cmax` 而数据只分给 `numCore` | 全部 | 空闲核只参与同步/归并调度 | `InitVar` |
| 确定性 | `isDeterministicKey!=0` | 避免跨核直接原子归约顺序变化 | 全部 | 0 时直接 atomic 到输出 | `CutNProcess`, `CutDProcess` |
| 归并 stride | 非 220/310 且 stride 超 `uint16_t` 才逐行；950 编译支路使用扩展搬运 | DMA stride 字段上限 | 确定性阶段 | `copyInByRow`（当前公开 950 路线不走） | `AddLayerNormGradDeterminsticCompute::copyIn` |

### POD/ABI

宏生成结构的字段顺序如下；固定仓未包含 `tilingdata_base.h` 的宏展开，因此 `sizeof/alignof/offsetof` 不可见。除单独标注外均为 `uint32_t/32-bit`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AddLayerNormGradTilingData` | `numCore` | u32 | 宏布局不可见 | core | `numCore` | 有效核、确定性行数 | `1..Cmax` |
| 同上 | `numLastDim` | u32 | 同上 | element | `D` | 全部计算/归约 | 非零；shape 乘积可能窄化 |
| 同上 | `numFirstDim` | u32 | 同上 | row | `N` | 小 D 清零、范围 | 非零 shape 乘积 |
| 同上 | `nInOneCoreLength` | u32 | 同上 | row/core | `ceil(N/numCore)` | 普通核基址/循环 | 全部 route |
| 同上 | `nInOneCoreLengthTail` | u32 | 同上 | row | 尾核余数 | 尾核循环 | 全部 route |
| 同上 | `ndInOneCoreLength` | u32 | 同上 | element/core | `Ln*D` | Cut-N GM 基址 | Cut-D 不读 |
| 同上 | `nAvailInUb` | u32 | 同上 | row | UB 公式 | Cut-D CopyIn blockCount 实参为 1；Cut-N 基值 | 至少 1 |
| 同上 | `dInnerLength` | u32 | 同上 | element | `D` 或 Cut-D tile | Cut-D D 循环/UB | 非零依赖 UB |
| 同上 | `dInnerLengthTail` | u32 | 同上 | element | `D-dInner*(dOuter-1)` | Cut-D 尾段 | Cut-N 不读 |
| 同上 | `dOuterLength` | u32 | 同上 | loop | `ceil(D/dInner)` | Cut-D 两遍循环 | Cut-N 不读 |
| 同上 | `nInOneCoreNorm` | u32 | 同上 | row | `Ln` | Cut-N 普通核 | Cut-D 不读 |
| 同上 | `gmOneCoreElemXYNorm` | u32 | 同上 | element | `Ln*D` | Cut-N GM extent | Cut-D 不读 |
| 同上 | `nAvailInUbNorm` | u32 | 同上 | row/tile | `min(Ln,Navail)` | Cut-N 普通核 | Cut-D 不读 |
| 同上 | `nMiddleCountNorm` | u32 | 同上 | loop | `ceil(Ln/nAvailNorm)` | Cut-N 普通核 | 同上 |
| 同上 | `ndRoundUpDtypeNorm` | u32 | 同上 | byte | `nAvailNorm*roundUpDtypeBytes` | Cut-N input/output queue | 同上 |
| 同上 | `n1RoundUpFloatNorm` | u32 | 同上 | byte | `nAvailNorm*32` | mean/rstd queue | 同上 |
| 同上 | `nInUbTotalNormTail` | u32 | 同上 | row | 最后 N tile 行数 | Cut-N tail | 同上 |
| 同上 | `nInOneCoreTail` | u32 | 同上 | row | `Lt` | Cut-N 尾核 | Cut-D 不读 |
| 同上 | `gmOneCoreElemXYTail` | u32 | 同上 | element | `Lt*D` | Cut-N 尾核 extent | 同上 |
| 同上 | `nAvailInUbTail` | u32 | 同上 | row/tile | `min(Lt,Navail)` | Cut-N 尾核 | 同上 |
| 同上 | `nMiddleCountTail` | u32 | 同上 | loop | `ceil(Lt/nAvailTail)` | Cut-N 尾核 | 同上 |
| 同上 | `ndRoundUpDtypeTail` | u32 | 同上 | byte | `nAvailTail*roundUpDtypeBytes` | Cut-N tail queues | 同上 |
| 同上 | `n1RoundUpFloatTail` | u32 | 同上 | byte | `nAvailTail*32` | tail mean/rstd queues | 同上 |
| 同上 | `nInUbTotalTailTail` | u32 | 同上 | row | 最后 tail-core tile | Cut-N | producer 使用 `nAvailNorm`，须原样复用 |
| 同上 | `dyPadRight` | u32 | 同上 | element | `roundUp(D,blockElems)-D` | Cut-N `DataCopyPad` | 非 220/310 支路不读 |
| 同上 | `rstdPadRight` | u32 | 同上 | FP32 element | `8-1=7` | mean/rstd `DataCopyPad` | 同上 |
| 同上 | `roundUpNumLastDim` | u32 | 同上 | element | dtype block 对齐 D | Cut-N UB 行 stride | 全部 dtype |
| 同上 | `roundUpNumLastDimDtype` | u32 | 同上 | byte | `roundUp(D)*sizeof(T)` | Gamma queue/queue公式 | 名称是 byte |
| 同上 | `roundUp1Dtype` | u32 | 同上 | FP32 element | `roundUp(1,8)=8` | Cut-N mean/rstd stride | 名称不是 byte |
| 同上 | `roundUpNumLastDimFloat` | u64/64-bit | 同上 | byte | `roundUp(D,8)*4` | FP32 queues/workspace | 64 位 |
| 同上 | `isDeterministicKey` | u32 | 同上 | bool flag | `context->GetDeterministic()` | workspace/归并分支 | 不改变 tiling key |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～R-006 | 10/11/20/21/30/31 | `add_layer_norm_grad_tiling.cpp::Tiling4AddLayerNormGrad` | `AddLayerNormGradTilingData` | `add_layer_norm_grad.cpp::add_layer_norm_grad` | `add_layer_norm_grad_cut_n.h::KernelAddLayerNormGrad` + utils + deterministic helper | `ascend950/...binary.json`, `ascendc_config.json` |
| R-007～R-012 | 40/41/50/51/60/61 | 同上 | 同上 | 同上 | `add_layer_norm_grad_cut_d.h::KernelAddLayerNormGradLarge` + utils + deterministic helper | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD 后按 key 实例化模板；未知 key 直接结束。每个 block 先由 block 0 清零 `dgamma/dbeta`，再全核同步；有效核绑定自己的 N 行。非确定性路线把局部 FP32 `dgamma/dbeta` 原子累加到输出。确定性路线把每核结果写到 `[core,2,roundUp(D,8)]` workspace，`SyncAll` 后重置 pipe，并由 `FinalProcessDeterministic` 把列按 64 分块、行按 180 分块归并到最终输出。所有计算只使用 Vector primitive，不含 Cube/AIC/MIX_CV。

### Route R-001：Cut-N FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/add_layer_norm_grad_tiling.cpp` | `AddLayerNormGradTilingImpl/GetTilingKey` | FP32，`nAvailInUb>=1 && ndAvailUb>=0`，无 dsum | key 10 |
| Router | `op_kernel/add_layer_norm_grad.cpp` | `add_layer_norm_grad` | `TILING_KEY_IS(10)` | `<float,10>::CutNProcess` |

#### Device 数据流

##### 函数级调用链

```text
add_layer_norm_grad -> KernelAddLayerNormGrad<float,10>::Init
 -> CopyInGamma -> for N tile: CopyIn -> PrecisionCompute -> MainCompute -> CopyOut
 -> [deterministic] SyncAll -> FinalProcessDeterministic
```

##### 源码等价伪码

```text
for tile rows: load dy,x1,x2,mean,rstd; for row: x=x1+x2; g=dy*gamma;
  xhat=(x-mean)*rstd; accumulate dgamma=dy*xhat, dbeta=dy;
  dx=rstd*g-xhat*sum(g*xhat)/D-sum(rstd*g)/D;
store dx; atomic-add local dgamma/dbeta to output or per-core workspace
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs/dx | `core*Ln*D + tile*Navail*D` | 有效 `rows*D` | FP32 `[N,D]` | caller/CopyOut | CopyIn/caller | 调用期 | D 以 8 元素对齐进 UB |
| input/output queues | UB | `(4*rows*Da + 2*rows*8 + 3*Da)*4` | FP32 | `InitBuffer` | Vector/MTE | 每 tile | depth=1；`Da=roundUp(D,8)` |
| deterministic workspace | `core*2*Da + {0,Da}` | `2*numCore*Da*4` bytes | FP32 | CopyOut | final reducer | Kernel 内两阶段 | 每核独占 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dy/x1/x2 · depth=1 · 3*rows*Da*4"]:2
  scalarQ["mean/rstd · depth=1 · 2*rows*8*4"]:1
  gammaQ["gamma · depth=1 · Da*4"]:1
  outQ["dx/dgamma/dbeta · depth=1 · (rows*Da+2*Da)*4"]:4
```

##### 字节账与复用顺序

```text
UB_explicit = (4*rows*Da + 2*rows*8 + 3*Da)*4
Host additionally reserves 1280 bytes and uses a conservative coexist formula
dynamic_ub = N/A
workspace_total = 36 MiB + deterministic*(2*numCore*Da*4)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| x1/x2/dy queues | CopyIn | `MainCompute` 原地复用为 x、g、临时项 | `FreeTensor` 后下个 tile | Queue + `PipeBarrier` | depth=1 串行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE/Vector | Queue、`PipeBarrier`、MTE2_S/S_V | CopyIn/标量读取 | MainCompute | UB | FP32；ReduceSum 后标量等待 |
| 输出归约 | AtomicAdd 或 workspace+`SyncAll` | 各有效核 | 输出/最终归并 | dgamma/dbeta | N tail 只处理有效行；D tail 用 pad/安全回退 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_layer_norm_grad_tiling.cpp` | `Tiling4AddLayerNormGrad` | key、blockDim、workspace |
| Kernel entry/router | `op_kernel/add_layer_norm_grad.cpp` | `<float,10>` | key 10 映射 Cut-N |
| Device body/helper | `op_kernel/add_layer_norm_grad_cut_n.h` | `CutNProcess/MainCompute/CopyOut` | 行 tile、原地 FP32、原子/确定性归约 |

### Route R-002：Cut-N FP32 + dsum

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `op_host/add_layer_norm_grad_tiling.cpp`, `op_kernel/add_layer_norm_grad.cpp` | `GetTilingKey`, `<float,11>` | R-001 条件且 optional input shape 非空，key `10+1` | 读取 dsum；否则 R-001 |

#### Device 数据流

##### 函数级调用链

```text
<float,11>::Init -> dSumQue -> CutNProcess -> MainCompute -> Add(dx,dx,dsum) -> CopyOut
```

##### 源码等价伪码

```text
执行 R-001 全流程；每行最终提交前 dx += dsum。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dsum GM/Queue | 同 dy | `rows*Da*4` | FP32 | caller/CopyIn | final Add | tile | depth=1 |

##### 分层内存布局

R-001 布局增加 `dSumQue=rows*Da*4`。

##### 字节账与复用顺序

```text
UB_explicit = R-001 + rows*Da*4; workspace 同 R-001
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dSumQue | CopyIn | final Add | FreeTensor 后 | Queue | key%10==1 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dsum Add | `Add` + `PipeBarrier<PIPE_V>` | MainCompute | CopyOut | dx UB | FP32，有效 D |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_layer_norm_grad_tiling.cpp` | `BasicCheck/GetTilingKey` | optional shape 使 key+1 |
| Kernel entry/router | `op_kernel/add_layer_norm_grad.cpp` | `<float,11>` | dsum specialization |
| Device body/helper | `op_kernel/add_layer_norm_grad_cut_n.h` | `HAS_ADDITIONAL_INPUT` | 条件分配/搬运/相加 |

### Route R-003：Cut-N FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `CheckAndSetDtype`, `<half,20>` | FP16、Cut-N fit、无 dsum | key 20 |

#### Device 数据流

##### 函数级调用链

```text
<half,20>::CutNProcess -> Cast inputs/gamma FP16->FP32 -> MainCompute -> Cast dx FP32->FP16(CAST_NONE)
```

##### 源码等价伪码

```text
按 R-001 分行；所有算术和归约在复用的 FP32 D 缓冲上完成，最后仅 dx 降精度。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| T queues | 行地址同 R-001 | `4*rows*roundUp(D,16)*2 + roundUp(D,16)*2` | FP16 | CopyIn | Cast/CopyOut | tile | 32B |
| FP32 queues/TBuf | UB | `(2+5)*Da32*4` | FP32 | Cast/compute | compute/output atomic | 逐行复用 | `Da32=roundUp(D,8)` |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  tQ["FP16 input/output queues · depth=1"]:1
  fp32["5 FP32 TBuf + dgamma/dbeta queues"]:1
  scalar["mean/rstd queues"]:1
```

##### 字节账与复用顺序

```text
UB_explicit=(4*rows+1)*Da16*2 + 7*Da32*4 + 2*rows*8*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 5 FP32 TBuf | per-row Cast/MainCompute | same row | next row | `PipeBarrier<PIPE_V>` | 顺序处理行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast | `CAST_NONE` | FP16 queues | FP32 compute / dx queue | 数值 | D 有效元素，padding 不归约 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `AddLayerNormGradTilingImpl` | 低精度 UB 公式 |
| Kernel entry/router | kernel cpp | `<half,20>` | FP16 Cut-N |
| Device body/helper | cut_n.h | `PrecisionCompute` | FP32 升精度、CAST_NONE |

### Route R-004：Cut-N FP16 + dsum

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `GetTilingKey`, `<half,21>` | R-003 且 dsum 存在 | key 21 |

#### Device 数据流

##### 函数级调用链

```text
<half,21>::CutNProcess -> Cast dsum/inputs -> FP32 MainCompute+dsum -> CAST_NONE dx
```

##### 源码等价伪码

```text
R-003 + 每行 dsum FP16->FP32 后累加。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dsum Queue/TBuf | 行地址 | `rows*Da16*2 + Da32*4` | FP16/FP32 | CopyIn/Cast | final Add | tile/row | 条件分配 |

##### 分层内存布局

R-003 布局增加 dsum queue 与一个 FP32 TBuf。

##### 字节账与复用顺序

```text
UB_explicit = R-003 + rows*Da16*2 + Da32*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dSumFp32Buf | Cast | final Add | next row | PIPE_V barrier | key 21 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dsum | Cast+Add | dsum queue | dx cast | dx | FP32 累加后 FP16 输出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `ComputeCoexistFactor` | dsum 增加 coexist factor |
| Kernel entry/router | kernel cpp | `<half,21>` | key 21 |
| Device body/helper | cut_n.h | `HAS_ADDITIONAL_INPUT` | 条件 TQue/TBuf |

### Route R-005：Cut-N BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `CheckAndSetDtype`, `<bfloat16_t,30>` | BF16、Cut-N fit、无 dsum、编译 guard 220/310 | key 30 |

#### Device 数据流

##### 函数级调用链

```text
<bfloat16_t,30>::CutNProcess -> BF16->FP32 -> MainCompute -> CAST_RINT dx
```

##### 源码等价伪码

```text
与 R-003 相同，但最终 BF16 使用 round-to-nearest-integer 模式。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| BF16/FP32 UB | 同 R-003 | 同 R-003 | BF16 + FP32 | CopyIn/Cast | compute/CopyOut | tile | 32B |

##### 分层内存布局

同 R-003，T queues 的元素类型改为 BF16。

##### 字节账与复用顺序

```text
UB_explicit = R-003（字节相同）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 TBuf | BF16 Cast | MainCompute | next row | PIPE_V | 仅 guard 支持平台 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输出 Cast | `CAST_RINT` | FP32 dx | BF16 dXQue | 舍入 | 有效 D |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `CheckAndSetDtype` | dtypeKey=3 |
| Kernel entry/router | kernel cpp | `<bfloat16_t,30>` | guarded BF16 Cut-N |
| Device body/helper | cut_n.h | `PrecisionCompute` | FP32 + RINT |

### Route R-006：Cut-N BF16 + dsum

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `GetTilingKey`, `<bfloat16_t,31>` | R-005 且 dsum 存在 | key 31 |

#### Device 数据流

##### 函数级调用链

```text
<bfloat16_t,31>::CutNProcess -> BF16 inputs/dsum Cast -> FP32 compute+dsum -> RINT
```

##### 源码等价伪码

```text
R-005 + FP32 dsum 累加。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dsum Queue/TBuf | 行地址 | `rows*Da16*2 + Da32*4` | BF16/FP32 | CopyIn/Cast | Add | tile/row | 条件分配 |

##### 分层内存布局

R-005 增加 dsum queue/TBuf。

##### 字节账与复用顺序

```text
UB_explicit = R-005 + rows*Da16*2 + Da32*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dSumFp32Buf | Cast | Add | next row | PIPE_V | key 31 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dsum/output | FP32 Add + RINT | dsum | BF16 dx | 数值 | D tail masked by count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `GetTilingKey` | 30+1 |
| Kernel entry/router | kernel cpp | `<bfloat16_t,31>` | BF16 dsum |
| Device body/helper | cut_n.h | `HAS_ADDITIONAL_INPUT` | conditional storage/compute |

### Route R-007：Cut-D FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `AddLayerNormGradTilingImpl`, `<float,40>` | FP32、Cut-N 不 fit、无 dsum | key 40 |

#### Device 数据流

##### 函数级调用链

```text
<float,40>::CutDProcess
 -> per row first D loop: CopyIn -> MainComputeFirstPart -> CopyOutBetaGamma
 -> second D loop: CopyIn -> MainComputeSecondPart -> CopyOutX
 -> [deterministic] final reducer
```

##### 源码等价伪码

```text
for row:
  tmpVar=tmpMean=0
  for D tile: compute partial dgamma/dbeta and ReduceSum partials; atomic/store gradient outputs
  for D tile: reload inputs; use completed tmpVar/tmpMean to compute and store dx
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM tile | `core*Ln*D+row*D+dTile*dInner` | `Dcur` | FP32 | caller | two CopyIn passes | row | inputs intentionally reload |
| UB queues/TBuf | UB | `7*Da*4 + 4*8*4` | FP32 | InitBuffer | two passes | row/tile | `Da=roundUp(dInner,8)` |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dy/x1/x2/gamma · 4*Da*4"]:2
  outQ["dx/dgamma/dbeta · 3*Da*4"]:1
  scalar["mean/rstd/tmpMean/tmpVar · 4*8*4"]:1
```

##### 字节账与复用顺序

```text
UB_explicit = 7*Da*4 + 128; Host selector uses conservative 9*Da*4 + scalar reserve
workspace 同 R-001
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| tmpMean/tmpVar | each first-pass ReduceSum | second pass | next row after all D tiles | PIPE_V + loop order | first pass complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| two-pass | V_S/S_V/V_MTE3 + barriers | first-pass reductions | second pass | scalar TBuf/output | `Dcur=Dtail` on last tile |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `AddLayerNormGradTilingImpl` | dInner/dOuter |
| Kernel entry/router | kernel cpp | `<float,40>` | Cut-D |
| Device body/helper | cut_d.h | `CutDProcess/MainComputeFirstPart/SecondPart` | 两遍 D |

### Route R-008：Cut-D FP32 + dsum

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `GetTilingKey`, `<float,41>` | R-007 且 dsum 存在 | key 41 |

#### Device 数据流

##### 函数级调用链

```text
<float,41>::CutDProcess -> first pass(no dsum) -> second pass CopyIn(dsum=true) -> Add dx+dsum
```

##### 源码等价伪码

```text
R-007；dsum 仅在第二遍 D 扫描搬入并累加。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dsum queue | D tile 地址 | `Da*4` | FP32 | second CopyIn | final Add | second pass tile | first pass不分配 tensor |

##### 分层内存布局

R-007 增加 `dSumQue=Da*4`。

##### 字节账与复用顺序

```text
UB_explicit=R-007+Da*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dSumQue | second CopyIn | second compute | next D tile | Queue | has_dsum=true only second pass |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dsum | Queue+Add | second CopyIn | CopyOutX | dx | Dcur count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `GetTilingKey` | 40+1 |
| Kernel entry/router | kernel cpp | `<float,41>` | key41 |
| Device body/helper | cut_d.h | `CopyIn(...,true)` | second pass dsum |

### Route R-009：Cut-D FP16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `CheckAndSetDtype`, `<half,50>` | FP16、Cut-D、无 dsum | key 50 |

#### Device 数据流

##### 函数级调用链

```text
<half,50>::CutDProcess -> each pass Cast inputs -> FP32 first/second part -> CAST_NONE dx
```

##### 源码等价伪码

```text
R-007 两遍结构；每个 D tile 升到 FP32，第二遍结果降为 FP16。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| FP16 queues | D tile | `5*Da16*2` | FP16 | CopyIn | Cast/CopyOut | tile | depth1 |
| FP32 UB | UB | `7*Da32*4+128` | FP32 | Cast/reduce | compute | tile/row | 5 TBuf+2 output queues+scalars |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  q16["5 FP16 queues · 5*Da16*2"]:1
  fp32["5 TBuf + 2 output queues · 7*Da32*4"]:1
  scalar["4 scalar blocks · 128"]:1
```

##### 字节账与复用顺序

```text
UB_explicit=5*Da16*2+7*Da32*4+128
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 TBuf | Cast/first or second compute | current pass | next pass/tile | PIPE_V | no overlap |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| precision | Cast NONE | FP16 | FP32 math / FP16 dx | numeric | Dtail uses elem count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `AddLayerNormGradTilingImpl` | low precision dInner |
| Kernel entry/router | kernel cpp | `<half,50>` | key50 |
| Device body/helper | cut_d.h | `CutDProcess` | FP32 two-pass |

### Route R-010：Cut-D FP16 + dsum

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `GetTilingKey`, `<half,51>` | R-009 且 dsum 存在 | key 51 |

#### Device 数据流

##### 函数级调用链

```text
<half,51>::CutDProcess -> second pass dsum FP16->FP32 -> Add -> CAST_NONE
```

##### 源码等价伪码

```text
R-009 + second-pass dsum。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dsum Queue/TBuf | D tile | `Da16*2+Da32*4` | FP16/FP32 | CopyIn/Cast | Add | second pass | conditional |

##### 分层内存布局

R-009 增加一对 dsum Queue/TBuf。

##### 字节账与复用顺序

```text
UB_explicit=R-009+Da16*2+Da32*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dSumFp32Buf | Cast | final Add | next tile | PIPE_V | second pass |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dsum | FP32 Add | dsum Cast | dx Cast | numeric | Dcur |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `ComputeCoexistFactor` | optional resource |
| Kernel entry/router | kernel cpp | `<half,51>` | key51 |
| Device body/helper | cut_d.h | `HAS_ADDITIONAL_INPUT` | dsum branch |

### Route R-011：Cut-D BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `CheckAndSetDtype`, `<bfloat16_t,60>` | BF16、Cut-D、无 dsum、guard 220/310 | key60 |

#### Device 数据流

##### 函数级调用链

```text
<bfloat16_t,60>::CutDProcess -> BF16->FP32 two-pass -> CAST_RINT dx
```

##### 源码等价伪码

```text
R-009 字节与两遍结构相同，输出舍入改为 RINT。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| BF16/FP32 UB | D tile | 同 R-009 | BF16/FP32 | CopyIn/Cast | compute/CopyOut | tile | 32B |

##### 分层内存布局

同 R-009，FP16 queues 改为 BF16。

##### 字节账与复用顺序

```text
UB_explicit=R-009
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 TBuf | BF16 Cast | compute | next tile | PIPE_V | compiler guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| output | CAST_RINT | FP32 dx | BF16 | numeric | Dtail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `CheckAndSetDtype` | dtypeKey3 |
| Kernel entry/router | kernel cpp | `<bfloat16_t,60>` | guarded key60 |
| Device body/helper | cut_d.h | `CutDProcess` | FP32 + RINT |

### Route R-012：Cut-D BF16 + dsum

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | tiling cpp / kernel cpp | `GetTilingKey`, `<bfloat16_t,61>` | R-011 且 dsum 存在 | key61 |

#### Device 数据流

##### 函数级调用链

```text
<bfloat16_t,61>::CutDProcess -> second pass dsum Cast -> FP32 Add -> RINT
```

##### 源码等价伪码

```text
R-011 + second-pass dsum。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dsum Queue/TBuf | D tile | `Da16*2+Da32*4` | BF16/FP32 | CopyIn/Cast | Add | second pass | conditional |

##### 分层内存布局

R-011 增加 dsum Queue/TBuf。

##### 字节账与复用顺序

```text
UB_explicit=R-011+Da16*2+Da32*4
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dsum FP32 | Cast | Add | next D tile | PIPE_V | key61 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dsum/output | FP32 Add + RINT | dsum | BF16 dx | numeric | Dcur |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `GetTilingKey` | 60+1 |
| Kernel entry/router | kernel cpp | `<bfloat16_t,61>` | key61 |
| Device body/helper | cut_d.h | `HAS_ADDITIONAL_INPUT` | conditional dsum |

## 算子特定优化

### Trick T-001：UB 容量驱动的 Cut-N/Cut-D 双算法

#### 触发与源码依据

- 适用 Route：R-001～R-012
- 触发条件：Cut-N coexist 公式不能容纳至少一行时切换 Cut-D。
- Golden 来源：[`op_host/add_layer_norm_grad_tiling.cpp:141-178`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_host/add_layer_norm_grad_tiling.cpp#L141-L178)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tilingStruct.nAvailInUb < 1 || ndAvailUb < 0) {
    cutDPath = true;
    tilingStruct.nAvailInUb = 1;
    ndAvailUb = actualAvailUb / sizeof(float) - ROUND_UP(1, BLOCK_NUMBER) * COEXIST_TENSOR_INPUT32_D_1;
    tilingStruct.dInnerLength = ndAvailUb / (COEXIST_TENSOR_INPUT32_ONLY_D + coexistTensorInput32WithN);
    tilingStruct.dInnerLength = ROUND_DOWN(tilingStruct.dInnerLength, BLOCK_NUMBER);
}
```

#### 机制与设计原因

常见 D 可整行驻留时 Cut-N 一次载入多行并复用 gamma；D 太大时改为 Cut-D，两遍扫描换取有界 UB。`dInnerLength` 按 dtype、可选 dsum 和 32B 对齐推导，避免用固定 tile 造成 UB 超限。

#### 迁移边界

- 适用：含逐行归约且 D 可能跨越 UB 的融合反向算子。
- 前提：两遍读取可接受，且第一遍标量归约完成后能安全进入第二遍。
- 失效：`dInnerLength==0` 的极小 UB 平台固定仓没有额外 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：逐核私有 workspace 的确定性归并

#### 触发与源码依据

- 适用 Route：全部，`isDeterministicKey!=0`
- 触发条件：运行上下文开启确定性。
- Golden 来源：[`op_kernel/add_layer_norm_grad_cut_n.h:213-225`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad_cut_n.h#L213-L225)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if(isDeterministicKey) {
    SyncAll();
    pipe.Reset();
    AddLayerNormGradDeterminsticCompute op1;
    op1.initBuffer(pipe, dGammaGm, dBetaGm, workspaceGMOri, CONSTANT_TWO);
    op1.FinalProcessDeterministic(roundUpNumLastDimFloatLen, selfTiling.numCore, selfTiling.numLastDim);
}
```

#### 机制与设计原因

主阶段不让不同核以不定顺序竞争最终 `dgamma/dbeta`，而是写每核独占的两个对齐切片；同步后按列 tile 分配唯一归并者，行方向按固定顺序相加。归并 UB 使用 180×64 输入双缓冲和 64 元素输出双缓冲，总计 `185344` 字节。

#### 迁移边界

- 适用：跨核 FP32 归约需要确定重现的算子。
- 前提：workspace 至少 `2*numCore*roundUp(D,8)*4` 字节且存在全核同步。
- 失效：关闭确定性时直接原子累加输出，不支付归并 workspace。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001～R-006 Cut-N | R-007～R-012 Cut-D | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 至少一行可驻留 UB | Cut-N fit 失败 | Host 容量公式 |
| 模板与 dtype | `KernelAddLayerNormGrad<T,key>` | `KernelAddLayerNormGradLarge<T,key>` | dtype 决定 Cast/对齐；key 个位决定 dsum |
| 循环与分核 | 每核 N 行，N tile，逐行一次计算 | 每核 N 行，每行两遍 D tile | 大 D 限制 UB |
| Queue/TBuf | 多行 TQue；低精度逐行 FP32 TBuf | 单 D tile TQue/TBuf + 两个归约标量 TBuf | 驻留粒度不同 |
| 内存总占用与复用 | gamma 跨本核所有行复用 | 输入为求 dx 再搬一次 | 容量换搬运 |
| 搬运/Vector API | DataCopyPad、Add/Mul/Muls/Adds/ReduceSum | 相同 primitive，两遍调用 | 均为 AIV_VECTOR |
| 同步、精度与 tail | N tail；FP16 NONE/BF16 RINT | Dtail；相同精度策略 | 有效 count 防止 padding 入归约 |
| fallback/失败边界 | 不 fit 转 Cut-D | 无第三算法 | dInner=0 未显式防护 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；详见无后缀卡 | 公开支持；详见无后缀卡 | 公开支持；`ascend950` OpDef、binary config、kernel config 三重接线 |
| Host/Launcher | 与 A3/950 共用源码 | 与 A2/950 共用源码 | 共用 Tiling/OpDef，平台查询实际 AIV 核数和 UB |
| Kernel/资源 | 同源 router/body | 同源 router/body | 同源 router/body；BF16 guard 覆盖 `__CCE_AICORE__==220 \|\| 310`，产品到宏值的映射由固定仓外工具链提供 |

## 限制与失败边界

- 不支持空 shape、rank 0 或 rank>8；shape/dtype 关系不符、dtype 非 FP32/FP16/BF16 时 Tiling 失败。
- `N/D`、每核元素数及多项积存入 u32，源码没有显式溢出检查；超大 shape 不得据此推断安全。
- Host 以 `maxUbSize-1280` 做无下界检查的无符号减法，且 Cut-D 后没有显式 `dInnerLength>0` guard；极小 UB 行为固定仓未闭合为可用 fallback。
- 非 220/310 平台的 `SafeDataCopy` 小于 32B 尾块源码注释明确仍可能踩踏；950 公开构建的编译宏数值映射由外部工具链提供，固定仓不可见。
- ACLNN 生成式 inner bridge、实际 runtime 错误码到 Host Tiling 失败的转换属于框架边界；本次未执行任何运行验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `norm/add_layer_norm_grad/docs/aclnnAddLayerNormGrad.md` | 两段式函数原型、产品表 | 公开 API、平台、参数 |
| 入口实现 | `norm/add_layer_norm_grad/op_host/add_layer_norm_grad_def.cpp` | `ops::AddLayerNormGrad`, `OP_ADD` | OpDef、AutoContiguous、平台注册 |
| Host/Tiling | `norm/add_layer_norm_grad/op_host/add_layer_norm_grad_tiling.h` | `AddLayerNormGradTilingData` | POD 字段 |
| Host/Tiling | `norm/add_layer_norm_grad/op_host/add_layer_norm_grad_tiling.cpp` | `Tiling4AddLayerNormGrad` | 校验、selector、workspace |
| Kernel entry/body | `norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad.cpp` | `add_layer_norm_grad` | key router |
| Kernel entry/body | `norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad_cut_n.h` | `KernelAddLayerNormGrad` | Cut-N body |
| Kernel entry/body | `norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad_cut_d.h` | `KernelAddLayerNormGradLarge` | Cut-D body |
| Kernel helper | `norm/add_layer_norm_grad/op_kernel/add_layer_norm_determinstic_compute.h` | `FinalProcessDeterministic` | 确定性归并 |
| Kernel helper | `norm/add_layer_norm_grad/op_kernel/add_layer_norm_grad_utils.h` | `SafeDataCopy`, `ReduceSumCustom` | 尾块、归约、初始化 |
| 构建接线 | `norm/add_layer_norm_grad/op_host/config/ascend950/add_layer_norm_grad_binary.json` | `AddLayerNormGrad` op_list | 950 FP32/FP16/BF16 binary |
| 构建接线 | `norm/add_layer_norm_grad/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn)` | ACLNN Host 接入 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `AddLayerNormGrad` | A2/A3/950 kernel 编译单元 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 31 字段按声明顺序序列化 | 类型、单位，尤其 byte/element 混合 | `AddLayerNormGradTilingData` |
| 调用顺序 | 初始化→同步→主计算→可选确定性归并 | 所有 launch block 必须参加全核同步 | `Init`, `CutN/CutDProcess` |
| workspace/内存 | 非确定性直接输出，确定性逐核双切片 | offset=`(2*core+channel)*Da`、owner/lifetime | `InitWorkspaceGmBuffer` |
| 同步 | Queue/barrier + 原子或 SyncAll | 输出清零必须先于跨核累加 | `InitOuputGmBuffer`, `SyncAll` |
| 精度与尾块 | 低精度升 FP32；FP16 NONE、BF16 RINT | 有效 D count、32B 对齐和尾块路径 | `PrecisionCompute`, `CopyOutX` |
| 失败边界 | Cut-N 失败仅降到 Cut-D | 不得虚构第三 fallback 或未知 key 行为 | `AddLayerNormGradGetTilingKey`, router |
