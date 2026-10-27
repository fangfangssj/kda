---
type: Operator
title: "ApplyFusedEmaAdam 950 ACLNN 算子知识"
description: "说明 aclnnApplyFusedEmaAdam 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/docs/aclnnApplyFusedEmaAdam.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_host/apply_fused_ema_adam_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_host/apply_fused_ema_adam_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_host/apply_fused_ema_adam_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/apply_fused_ema_adam.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/apply_fused_ema_adam_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/apply_fused_ema_adam_f32.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/apply_fused_ema_adam_f_bf16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_host/config/ascend950/apply_fused_ema_adam_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:11:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ApplyFusedEmaAdam 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnApplyFusedEmaAdamGetWorkspaceSize` + `aclnnApplyFusedEmaAdam`，公开 ACLNN 两段式接口；固定仓只有这一版本族。 |
| 计算语义 | 融合 Adam/AdamW 参数更新与 EMA；原语义状态 `var/m/v/s` 经单次 Kernel 分别写到四个 ref 输出。 |
| 输入输出 | `grad,var,m,v,s` 同 shape、同 dtype、ND，dtype 为 FP32/FP16/BF16；`step` 为 shape `[1]` 的 INT64；公开 ACLNN 文档约束主张量 rank 1–8。 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；所有公开可达节点都不含 Cube/AIC 或 `MIX_CV`。 |
| Kernel 路线 | R-001 FP32 AdamW；R-002 FP32 L2；R-003 FP16/BF16 AdamW；R-004 FP16/BF16 L2。 |
| 核心资源 | GM 输入/四路 GM 输出，AIV UB 中 depth=1 的 `VECIN/VECOUT`、FP32 cast TBuf（低精度路线）、两个 32 B pow TBuf；算子不消费 user workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN 契约→自动生成 ACLNN/连续化框架边界→`ApplyFusedEmaAdam` OpDef/Tiling/POD→单 Kernel router→全部 dtype/mode body→950 binary/CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `g=grad`、`p=var`，当 `biasCorrection` 为 false 时 `c1=c2=1`，否则 `c1=1-beta1^step`、`c2=1-beta2^step`。公开语义为：

```text
mode=0: g' = g + weightDecay*p       # L2
mode=1: g' = g                       # AdamW
mOut = beta1*m + (1-beta1)*g'
vOut = beta2*v + (1-beta2)*g'^2
u = (mOut/c1) / (sqrt(vOut/c2)+eps)
mode=1: u = u + weightDecay*p
varOut = p - lr*u
sOut = emaDecay*s + (1-emaDecay)*varOut
```

四个有效输出是 `varOut/mOut/vOut/sOut`。Kernel 对 `mode==1` 走 AdamW；其它值走 L2 body，公开接口语义只定义 0 和 1。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `grad` | rank 1–8，ND | FP32/FP16/BF16 | 梯度；支持非连续 ACLNN tensor，框架执行连续化。 |
| `varRef,mRef,vRef,sRef` | 与 `grad` 相同，ND | 与 `grad` 相同 | 输入/输出引用；Kernel ABI 中分别表现为原状态输入与独立输出地址。 |
| `step` | `[1]`，ND | INT64 | 每个 AIV core 读取元素 0，转为 FP32 后用于 `Power`。 |
| `lr` | scalar | ACLNN `double`→Op attr/POD FP32 | 推荐/默认 `1e-3`；公开文档给出 0–1。 |
| `emaDecay` | scalar | `double`→FP32 | 默认 `0.9999`。 |
| `beta1,beta2` | scalar | `double`→FP32 | 默认 `0.9/0.999`。 |
| `eps` | scalar | `double`→FP32 | 默认 `1e-8`。 |
| `mode` | scalar | INT64→POD UINT64 | 公开定义 0=L2、1=AdamW，默认 1。 |
| `biasCorrection` | scalar | BOOL→POD UINT64 | 默认 true。 |
| `weightDecay` | scalar | `double`→FP32 | 默认 0。 |

### 入口与版本

公开文档声明唯一两段式入口：第一段返回 workspace 和 executor，第二段把两者及 stream 提交执行。固定 Commit 未提供手写 ACLNN inner 文件；`op_host/CMakeLists.txt` 以 `ACLNNTYPE aclnn` 接入生成框架，OpDef 的六个输入均标记 `AutoContiguous()`，因此 ACLNN 参数封装、连续化及 ref 回写属于明确的生成框架边界。固定仓中没有 `V2` 或其它更新版本，实际实现落到 `ApplyFusedEmaAdam`。

公开 README 列出 Ascend 950PR/950DT 支持；OpDef 配置 `ascend950`，binary config 和 Kernel CMake 均有活动接线，无公开“不支持”冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段检查 tensor 空指针、dtype/format，并接收非连续 tensor；生成框架按 `AutoContiguous()` 准备连续输入。
2. `CheckInputDtype` 要求五个主 tensor dtype 相同，只接受 FP32/FP16/BF16，并要求 `step` 为 INT64；依次选择 key 102/101/100。
3. `CheckInputShape` 比较五个主 tensor 的 storage shape；不相同返回 `GRAPH_FAILED`。
4. `GetTilingAttr` 逐项读取八个 attr；任一 attr 指针为空即失败。
5. `TilingCompute` 查询平台 AIV 核数与 UB；取核数失败即失败。固定 Host Tiling 未额外验证 `step` shape、attr 数值范围或空张量。

### 参数、Workspace 与生命周期

设 `N=GetShapeSize(v)`、`P=GetCoreNumAiv()`、`S=sizeof(dtype)`、`U=UB bytes`：

```text
frontCoreNum = (N % P != 0) ? N % P : P
tailCoreNum  = (N <= P) ? 0 : P-frontCoreNum
blockDim     = frontCoreNum+tailCoreNum
coreCalcNum  = ceil(N/P)
fixed_pow    = 2*32 B
bytesPerElem = FP32 ? (mode==1 ? 32 : 36) : (mode==1 ? 48 : 54)
alignElems   = FP32 ? 8 : 16
coreCalcMax  = floor((U-64)/bytesPerElem/alignElems)*alignElems
loopNum      = coreCalcMax<coreCalcNum ? ceil(coreCalcNum/coreCalcMax) : 1
frontExtra   = loopNum==1 ? coreCalcNum : coreCalcNum-coreCalcMax*(loopNum-1)
tailExtra    = frontExtra-1
```

`Tiling4ApplyFusedEmaAdam` 写出 POD、`blockDim`，并把 workspace 0 设为 `GetLibApiWorkSpaceSize()`。entry 调用 `GetUserWorkspace(workspace)` 但结果未消费，所以无算子自有 workspace slice；ACLNN executor/workspace 与 stream 由两段式框架持有到第二段提交完成。

### 启动图与失败传播

```text
aclnnApplyFusedEmaAdamGetWorkspaceSize
  -> generated ACLNN executor / AutoContiguous boundary
  -> Tiling4ApplyFusedEmaAdam (key, blockDim, macro-defined POD bytes, lib workspace)
aclnnApplyFusedEmaAdam(workspace, workspaceSize, executor, stream)
  -> stream launch apply_fused_ema_adam<<<blockDim>>>
     inputs grad,var,m,v,s,step; outputs var_ref,m_ref,v_ref,s_ref
```

POD 的确切序列化字节数由仓外 `tilingdata_base.h` 宏生成，不能写死；上图不把它推测为某个尺寸。只有一个 Device 节点，没有 Kernel 间边。Tiling 任一步失败即不生成可启动 executor；第二段返回 launch 状态。输出由该 stream 上的单 Kernel 写入，owner 是调用者提供的 ref tensor；生成框架内部连续化临时量的 offset/extent 固定仓不可见。

## Route 与参数设计

### 分核与资源计算

余数核在前：`blockIdx<frontCoreNum` 的 base 为 `coreCalcNum*blockIdx`，每核逻辑长度 `coreCalcNum`；其余核 base 为 `coreCalcNum*frontCoreNum+(blockIdx-frontCoreNum)*(coreCalcNum-1)`，长度少 1。每核按 `coreCalcMax` 分 tile，前 `loopNum-1` 个 tile 满载，末 tile 分别使用 `frontCalcExtra` 或 `tailCalcExtra`。`coreCalcMax` 同时满足 32 B 元素对齐与真实 UB 字节账；下文内存公式用 `C=coreCalcMax`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 AdamW | 102 / `FusedEmaAdamF32<float>` | 五主输入同 dtype=FP32，`mode==1` | dtype 不符继续；mode 不符 R-002 |
| 2 | R-002 FP32 L2 | 102 / `FusedEmaAdamF32<float>` | 五主输入同 dtype=FP32，`mode!=1`；公开有效值为 0 | dtype 不符继续 |
| 3 | R-003 FP16/BF16 AdamW | 101→`half`，100→`bfloat16_t` | 五主输入同 dtype=FP16 或 BF16，`mode==1` | mode 不符 R-004 |
| 4 | R-004 FP16/BF16 L2 | 同 R-003 | 五主输入同 dtype=FP16 或 BF16，`mode!=1`；公开有效值为 0 | 不支持 dtype 在 Host 失败 |

```text
validate same dtype + step INT64
if dtype==FP32: key=102 -> if mode==1 R-001 else R-002
else if dtype==FP16: key=101 -> if mode==1 R-003 else R-004
else if dtype==BF16: key=100 -> if mode==1 R-003 else R-004
else GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 102 + `mode==1` | 同 shape FP32、step INT64、950 | 单 AIV Kernel | `FusedEmaAdamF32<float>::Compute` | `32*C+64` B UB；grad/var slot 复用；显式 MTE/V event | `DataCopyPad`; 末 tile `frontExtra/tailExtra` |
| R-002 | 102 + `mode!=1` | 同上，公开 mode=0 | 单 AIV Kernel | 同模板 L2 分支 | `36*C+64` B UB；独立 grad slot | 同上 |
| R-003 | 100/101 + `mode==1` | 同 shape BF16/FP16、step INT64、950 | 单 AIV Kernel | `FusedEmaAdamF16<bfloat16_t/half>::Compute` | `48*C+64` B UB；FP32 cast TBuf；slot 复用 | FP32 算术，`CAST_RINT` 输出，pad tail |
| R-004 | 100/101 + `mode!=1` | 同上，公开 mode=0 | 单 AIV Kernel | 同模板 L2 分支 | `54*C+64` B UB；独立 grad slot | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| UB tile | `coreCalcMax=floor((U-64)/bytesPerElem/A)*A` | 精确容纳 Queue/TBuf，A=8/16 元素即 32 B | 全部 | 无显式 fallback；若 `coreCalcMax==0`，后续除法无保护 | `TilingCompute` |
| mode UB | `mode==1` 时 `coreOnesize` 减一组输入元素字节 | grad 生命周期结束后 slot 0 才装 var | R-001/R-003 | `mode!=1` 分配独立 var+grad | `TilingCompute`, `Init` |
| dtype key | FP32→102，FP16→101，BF16→100 | router 只实例化三种模板 | 全部 | 其它 dtype `GRAPH_FAILED`；未知 key entry 无 body | `CheckInputDtype`, `apply_fused_ema_adam` |
| DMA 长度窄化 | `dataCount*sizeof(T)` 转 `uint16_t` | `DataCopyParams.blockLen` 为 16 位字节数 | 全部 | 固定 Host 未显式检查；实际 tile 由 UB 上界间接限制 | `Compute` |
| 空域 | 固定 Host 无 `N>0` guard | `N=0` 会令 `coreCalcNum/frontExtra=0` 且 `tailExtra` 无符号下溢 | 全部 | 固定仓未提供专用 empty route | `TilingCompute`, `Process` |

### POD/ABI

结构为 `ApplyFusedEmaAdamTilingData`。字段顺序和 Host/Device 类型闭合；`BEGIN_TILING_DATA_DEF` 的实际 C++ 布局实现位于固定仓外，故 `sizeof/alignof/offsetof` 统一标为不可见，不能按自然对齐推测。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| ApplyFusedEmaAdamTilingData | `lr` | float/32 | 固定仓不可见 | 标量 | `GetTilingAttr:set_lr((float)*attrLr)` | `InitData`; `Muls(update,lr)` | ACLNN double 窄化 |
| 同上 | `beta1` | float/32 | 同上 | 标量 | `set_beta1` | m 更新与 `ScalarPow` | 公开 0–1 |
| 同上 | `beta2` | float/32 | 同上 | 标量 | `set_beta2` | v 更新与 `ScalarPow` | 公开 0–1 |
| 同上 | `eps` | float/32 | 同上 | 标量 | `set_eps` | `Adds(sqrt(v),eps)` | 未做 Host 范围检查 |
| 同上 | `emaDecay` | float/32 | 同上 | 标量 | `set_emaDecay` | s 更新 | 公开 0–1 |
| 同上 | `weightDecay` | float/32 | 同上 | 标量 | `set_weightDecay` | L2/AdamW 分支 | ACLNN double 窄化 |
| 同上 | `mode` | uint64/64 | 同上 | 枚举 | `set_mode((uint64_t)*attrMode)` | `Init/Compute` | 1 特判，其它走 L2 |
| 同上 | `biasCorrection` | uint64/64 | 同上 | bool 0/1 | bool 转 0/1 | `Init` 是否算 pow | 仅等于 1 开启 |
| 同上 | `frontCoreNum` | uint64/64 | 同上 | core | `N%P` 或 P | base/末 tile 分支 | N=0 特殊边界 |
| 同上 | `tailCoreNum` | uint64/64 | 同上 | core | `N<=P?0:P-front` | 仅通过 blockDim 生效 | body 不直接读取 |
| 同上 | `coreCalcNum` | uint64/64 | 同上 | element/core | `ceil(N/P)` | core base | N=0 为 0 |
| 同上 | `loopNum` | uint64/64 | 同上 | tile/core | `ceil(coreCalcNum/coreCalcMax)` 或 1 | `Process` | `coreCalcMax=0` 无保护 |
| 同上 | `coreCalcMax` | uint64/64 | 同上 | element/tile | UB 公式并按 8/16 对齐 | buffer 分配、offset、满 tile | 32 B 对齐 |
| 同上 | `frontCalcExtra` | uint64/64 | 同上 | element | 末 tile余量 | 前核末 `Compute` | 可为 0 |
| 同上 | `tailCalcExtra` | uint64/64 | 同上 | element | `frontExtra-1` | 尾核末 `Compute` | 0 时跳过；N=0 下溢但前核路径不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/R-002 | 102 | `apply_fused_ema_adam_tiling.cpp::CheckInputDtype/TilingCompute` | `ApplyFusedEmaAdamTilingData` | `apply_fused_ema_adam.cpp::apply_fused_ema_adam` | `FusedEmaAdamF32<float>` + `FusedEmaAdamBase<float>` | config `ApplyFusedEmaAdam_2`; Kernel CMake 950 unit |
| R-003/R-004 | 101/100 | 同上 | 同上 | 同上 | `FusedEmaAdamF16<half/bfloat16_t>` + base | config `_1/_0`; Kernel CMake 950 unit |

## Kernel 实现

### 公共执行流程

entry 解析同一 POD，并按 key 构造一个模板对象；未知 key 不执行任何 body。`InitData` 缓存 attr 与分核字段，`Init` 绑定当前 core 的十路 GM 地址、按 mode/dtype 分配 UB、读取 `step[0]` 并可选算两次 FP32 `Power`。`Process` 遍历满 tile 后处理一块前/尾余量。每个 `Compute` 以 `DataCopyPad` 逐路装入，按公式用 Vector primitive 计算，在 m、v、var、s 各阶段后直接写对应 GM 输出；没有多 Kernel workspace 边。

### Route R-001：FP32 AdamW

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/apply_fused_ema_adam_tiling.cpp` | `CheckInputDtype` | `gradDtype==DT_FLOAT` | key 102 |
| Tiling | 同上 | `TilingCompute` | `mode==1` 减去一个 FP32 输入 slot | `bytesPerElem=32` |
| Router | `op_kernel/apply_fused_ema_adam.cpp` | `apply_fused_ema_adam` | `TILING_KEY_IS(102)` | `FusedEmaAdamF32<float>` |

#### Device 数据流

##### 函数级调用链

```text
apply_fused_ema_adam -> FusedEmaAdamF32<float>::Init
  -> InitData -> optional ScalarPow(beta1/beta2)
  -> Process -> Compute(tile)
     -> GM grad/m/v/var/s -> inQue -> FP32 Vector pipeline -> outQue -> GM m/v/var/s refs
```

##### 源码等价伪码

```text
slot0=grad; m'=beta1*m+(1-beta1)*grad; write mRef
v'=beta2*v+(1-beta2)*grad*grad; write vRef
u=(m'/c1)/(sqrt(v'/c2)+eps)
overwrite slot0 with var; u+=weightDecay*var
var'=var-lr*u; write varRef
s'=emaDecay*s+(1-emaDecay)*var'; write sRef
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM each tensor | `blockOffset+tile*C` | `dataCount` | FP32/ND | caller / prior state | Kernel | caller; launch lifetime | tail padded copy |
| `inQue` | slots 0..3 | `4*C*4` B | FP32 | `DataCopyPad` | Vector | pipe; one tile | depth1；slot0 grad→var |
| `outQue` | slots var/m/v/s=0/1/2/3 | `4*C*4` B | FP32 | Vector | GM copy | pipe; one tile | intermediates overwritten by stage |
| `powTBuf1/2` | separate | 32 B each | FP32 | Duplicate/Power | scalar read | Kernel | fixed 64 B |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["VECIN depth=1 · 16*C B · grad/var alias"]:2
  outQ["VECOUT depth=1 · 16*C B"]:1
  pow["2 pow TBuf · 64 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 16*C + 16*C + 64 = 32*C+64 B <= U
dynamic_ub = N/A；AIV 静态 TPipe buffer
workspace_total = 0 B operator-owned；仅框架 lib workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| in slot0 | grad DMA | m/v 计算 | var DMA | grad 最后消费后 `PipeVM2/M2V` | `mode==1` |
| out slots | 各 Vector stage | 对应 GM copy/后续算术 | 下一 stage/tile | `PipeVM3` 后 copy；tile 尾 FreeTensor | 单 tile 顺序执行 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→Vector | `MTE2_V` flag | DataCopyPad | Vector op | in slot | FP32；`dataCount` 有效元素 |
| Vector→GM | `V_MTE3` flag | Vector op | DataCopyPad out | out slot | FP32；tail 用实际字节数 |
| pow→scalar | `V_S` flag | `Power` | `GetValue(0)` | pow result | step INT64 先转 FP32 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_fused_ema_adam_tiling.cpp` | `TilingCompute` | mode1 的 32 B/elem 与 tile |
| Kernel entry/router | `op_kernel/apply_fused_ema_adam.cpp` | key 102 branch | 实例化 FP32 body |
| Device body/helper | `op_kernel/apply_fused_ema_adam_f32.h` | `Init/Compute/Process` | slot alias、FP32 公式、tail |

### Route R-002：FP32 L2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/apply_fused_ema_adam_tiling.cpp` | `CheckInputDtype` | FP32 | key 102 |
| Tiling/Device | tiling + `apply_fused_ema_adam_f32.h` | `TilingCompute/Compute` | `mode!=1`；公开 mode=0 | 独立 var/grad slot；先做 `grad+=wd*var` |
| Router | `op_kernel/apply_fused_ema_adam.cpp` | key 102 | 同 R-001 | FP32 body |

#### Device 数据流

##### 函数级调用链

```text
apply_fused_ema_adam -> FusedEmaAdamF32<float>::Init -> Process -> Compute
  -> load var+grad and form L2 grad -> m/v -> denominator -> var -> s -> four GM refs
```

##### 源码等价伪码

```text
slot0=var; slot4=grad; grad=grad+weightDecay*var
m',v',u as common formula; var'=var-lr*u; s'=emaDecay*s+(1-emaDecay)*var'
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | common core/tile formula | dataCount | FP32/ND | caller | Kernel | caller | DataCopyPad tail |
| `inQue` | var/m/v/s/grad slots 0..4 | `20*C` B | FP32 | DMA | Vector | tile | no grad/var alias |
| `outQue` | 4 slots | `16*C` B | FP32 | Vector | GM | tile | stage reuse |
| pow TBuf | separate | 64 B | FP32 | Power | scalar | Kernel | fixed |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["VECIN depth=1 · 20*C B"]:2
  outQ["VECOUT depth=1 · 16*C B"]:1
  pow["pow TBuf · 64 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 20*C+16*C+64 = 36*C+64 B <= U
dynamic_ub = N/A
workspace_total = 0 B operator-owned
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| out m slot | `Muls(wd,var)` 临时 | L2 Add | mOut | 顺序 Vector 指令 | mode!=1 |
| queues | tile stages | GM / downstream stage | next tile | events + FreeTensor | 当前 tile 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load/compute/store | `MTE2_V`,`V_MTE3` | DMA/Vector | Vector/DMA | queue slots | 全 FP32；实际 `dataCount` |
| tile tail | branch | `Process` | `Compute` | GM valid range | 尾核 extra=0 时跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_fused_ema_adam_tiling.cpp` | `TilingCompute` | mode0 36 B/elem |
| Kernel entry/router | `op_kernel/apply_fused_ema_adam.cpp` | key 102 | FP32 template |
| Device body/helper | `op_kernel/apply_fused_ema_adam_f32.h` | `Compute` lines 125–185 | L2 grad 与四状态输出 |

### Route R-003：FP16/BF16 AdamW

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | tiling cpp | `CheckInputDtype` | FP16→101；BF16→100 | reduced route |
| Tiling | tiling cpp | `TilingCompute` | dtype size 2 且 mode1 | 48 B/elem，align 16 |
| Router | kernel cpp | entry branches | 101/100 | `FusedEmaAdamF16<half/bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> FusedEmaAdamF16<T>::Init -> optional ScalarPow -> Process -> Compute
  -> GM T -> inQue T -> Cast FP32 inCastBuf -> FP32 math outCastBuf
  -> CAST_RINT outQue T -> GM refs
```

##### 源码等价伪码

```text
gradT -> gradF; compute mOutF/vOutF; cast each to T only for GM output
continue bias correction with unrounded mOutF/vOutF
after grad lifetime ends, overwrite T/FP32 slot0 with var
compute varOutF and sOutF; CAST_RINT only at four output commits
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inQue/outQue` | 4 slots each | `8*C` B each | T | GM/Cast | Cast/GM | tile | depth1；in slot0 grad→var |
| `inCastBuf` | 4 slots | `16*C` B | FP32 | Cast | math | tile | slot0 reuse |
| `outCastBuf` | 4 slots | `16*C` B | FP32 | math | cast/downstream | tile | stage reuse |
| pow TBuf | separate | 64 B | FP32 | Power | scalar | Kernel | fixed |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inout["T queues · 8*C + 8*C B"]:2
  casts["FP32 cast TBuf · 16*C + 16*C B"]:1
  pow["pow · 64 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 8*C+8*C+16*C+16*C+64 = 48*C+64 B <= U
dynamic_ub = N/A
workspace_total = 0 B operator-owned
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/cast slot0 | grad DMA+Cast | m/v | var DMA+Cast | `PipeVM2/M2V` and program order | AdamW only |
| FP32 m/v output slots | m/v math | correction math + output Cast | later stages | Vector order; GM copy fenced | do not replace with rounded T |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load cast | `MTE2_V` | DataCopyPad | Cast | T input | `CAST_NONE` T→FP32 |
| output | `CAST_RINT` + `V_MTE3` | FP32 math | GM copy | T output | 四状态降精度；tail actual count |
| correction | FP32 local | m/v math | Div/Sqrt | optimizer update | 使用未降精度 local |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `CheckInputDtype/TilingCompute` | keys 与 48 B/elem |
| Kernel entry/router | kernel cpp | keys 100/101 | BF16/FP16 instances |
| Device body/helper | `op_kernel/apply_fused_ema_adam_f_bf16.h` | `Init/Compute/Process` | FP32 compute、RINT output、slot alias |

### Route R-004：FP16/BF16 L2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | tiling cpp | `CheckInputDtype` | FP16/BF16 | key 101/100 |
| Tiling/Device | tiling + f_bf16 header | mode!=1；公开 mode0 | 54 B/elem；L2 grad | R-004 |
| Router | kernel cpp | key branches | 101/100 | shared low-precision body |

#### Device 数据流

##### 函数级调用链

```text
entry -> FusedEmaAdamF16<T>::Init -> Process -> Compute
  -> separately load/cast var and grad -> FP32 L2 grad -> common FP32 optimizer+EMA -> RINT outputs
```

##### 源码等价伪码

```text
varF=Cast(var); gradF=Cast(grad)+weightDecay*varF
mOutF,vOutF,varOutF,sOutF = FP32 fused pipeline
write CAST_RINT(mOutF,vOutF,varOutF,sOutF)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| T queues | 5 input +4 output slots | `10*C+8*C` B | FP16/BF16 | GM/Cast | Cast/GM | tile | depth1 |
| FP32 cast TBuf | 5 input +4 output slots | `20*C+16*C` B | FP32 | Cast/math | math/Cast | tile | stage reuse |
| pow TBuf | separate | 64 B | FP32 | Power | scalar | Kernel | fixed |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  tq["T queues · 18*C B"]:2
  fq["FP32 TBuf · 36*C B"]:1
  pow["pow · 64 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 10*C+8*C+20*C+16*C+64 = 54*C+64 B <= U
dynamic_ub = N/A
workspace_total = 0 B operator-owned
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| outCast m slot | weight-decay temp | L2 Add | mOutF | Vector program order | mode!=1 |
| all tile buffers | DMA/Vector | downstream/GM | next tile | events + `FreeTensor` | current tile complete |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| L2 prep | Cast/Muls/Add | var+grad DMA | m/v | FP32 gradF | invalid padded lanes excluded by count |
| outputs | RINT + MTE3 | FP32 stages | GM refs | four states | only valid `dataCount` copied |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | tiling cpp | `TilingCompute` | 54 B/elem and align16 |
| Kernel entry/router | kernel cpp | keys 100/101 | low precision template |
| Device body/helper | f_bf16 header | `Compute` lines 122–192 | L2、FP32 算术、RINT 输出 |

## 算子特定优化

### Trick T-001：AdamW 分支复用 grad/var slot

#### 触发与源码依据

- 适用 Route：R-001、R-003
- 触发条件：`mode==1`
- Golden 来源（FP32）：[`apply_fused_ema_adam_f32.h:92-102`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/apply_fused_ema_adam_f32.h#L92-L102)
- Golden 来源（FP16/BF16）：[`apply_fused_ema_adam_f_bf16.h:84-97`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/apply_fused_ema_adam_f_bf16.h#L84-L97)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (this->mode == 1) {
    INPUT_NUM -= 1;
}
pipe->InitBuffer(inQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * INPUT_NUM);
pipe->InitBuffer(outQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * OUTPUT_NUM);

varOffset = this->coreCalcMax * INDEX_VAR;
mOffset = this->coreCalcMax * INDEX_M;
vOffset = this->coreCalcMax * INDEX_V;
sOffset = this->coreCalcMax * INDEX_S;
gradOffset = this->mode == 0 ? this->coreCalcMax * INDEX_GRAD : varOffset;
```

低精度模板在同一谓词下同时消除 T queue 与 FP32 cast TBuf 的一个输入 slot：

```cpp
if (this->mode == 1) {
    INPUT_NUM -= 1;
}
pipe->InitBuffer(inQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * INPUT_NUM);
pipe->InitBuffer(outQue, BUFFER_NUM, this->coreCalcMax * sizeof(T) * OUTPUT_NUM);

pipe->InitBuffer(inCastBuf, this->coreCalcMax * sizeof(float) * INPUT_NUM);
pipe->InitBuffer(outCastBuf, this->coreCalcMax * sizeof(float) * OUTPUT_NUM);

varOffset = this->coreCalcMax * INDEX_VAR;
mOffset = this->coreCalcMax * INDEX_M;
vOffset = this->coreCalcMax * INDEX_V;
sOffset = this->coreCalcMax * INDEX_S;
gradOffset = this->mode == 0 ? this->coreCalcMax * INDEX_GRAD : varOffset;
```

#### 机制与设计原因

AdamW 不在开头同时需要 `var` 与 `grad`：先在 slot0 保存 grad 完成 m/v，再加载 var 覆盖 slot0。Host 同步把每元素预算减少一组 `dtype`（低精度还减少对应 FP32 cast slot），使 UB 账与 Device 条件分配一致；这减少物理 UB 容量而不改变 GM 搬运次数。

#### 迁移边界

- 适用：两个输入生命周期不重叠的逐元素融合 Kernel。
- 前提：Host tile 预算与 Device alias 谓词完全一致，且覆盖发生在 grad 最后一次消费之后。
- 失效：L2 需同时使用 var 与 grad，必须走 R-002/R-004 独立 slot。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的可验证反例是把该 alias 用到 mode=0：L2 首阶段同时读取 var 和 grad，会覆盖输入；源码因此为 mode=0 保留第 5 个 slot。

### Trick T-002：低精度存储、FP32 融合计算与延迟降精度

#### 触发与源码依据

- 适用 Route：R-003、R-004
- 触发条件：dtype 为 FP16 或 BF16
- Golden 来源：[`apply_fused_ema_adam_f_bf16.h:142-146`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_fused_ema_adam/op_kernel/apply_fused_ema_adam_f_bf16.h#L142-L146)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Add(outLocalC[mOffset], outLocalC[mOffset], inLocalC[mOffset], dataCount);
Cast(outLocal[mOffset], outLocalC[mOffset], RoundMode::CAST_RINT, dataCount);
this->PipeVM3();
DataCopyPad(gmMRef[offset], outLocal[mOffset], copyParams);
Muls(inLocalC[mOffset], outLocalC[mOffset], 1 / this->beta1Correction, dataCount);
```

#### 机制与设计原因

GM/Queue 保持 16-bit，进入 `inCastBuf` 后所有 Adam/EMA 算术用 FP32。虽然 `mOut` 先 RINT 写 GM，后续偏差校正明确继续读取未舍入的 `outLocalC[mOffset]`；v 路径同理。这样避免在融合管线内部反复低精度舍入，同时只在四个状态提交处降精度。

#### 迁移边界

- 适用：低精度状态存储、但融合内部需要稳定除法/平方根/累计的更新算子。
- 前提：UB 能容纳输入/输出 FP32 cast TBuf；输出舍入模式保持 `CAST_RINT`。
- 失效：若直接以已舍入 `outLocal` 继续校正，会改变固定源码的数值顺序。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

将 `Muls(..., 1/betaCorrection)` 的源换成低精度 `outLocal` 是固定源码可验证的不等价改写，会提前引入一次舍入。

## 模板与实现对比

| 对比维度 | R-001 FP32 AdamW | R-002 FP32 L2 | R-003 FP16/BF16 AdamW | R-004 FP16/BF16 L2 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key102, mode1 | key102, mode非1 | key100/101, mode1 | key100/101, mode非1 | dtype 先定 key，mode 定资源/公式分支 |
| 模板与 dtype | F32<float> | 同 | F16<BF16/half> | 同 | 16-bit 共用 FP32 cast body |
| 循环与分核 | 前/尾核，align8 | 同左 | 前/尾核，align16 | 同左 | 对齐元素数均对应 32 B |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2 Queue+pow | 同 | 2 Queue+4 TBuf | 同 | 无 dynamic UB/通信 |
| 内存总占用与复用 | 32C+64，alias | 36C+64 | 48C+64，alias | 54C+64 | mode1 消除一组输入 slot |
| 搬运/Vector/SIMT/NDDMA/通信 API | FP32 DataCopyPad+Vector | 同+L2 prep | Cast+FP32 Vector | 同+L2 prep | 全部 AIV Vector |
| 同步、精度与 tail | events，FP32 | 同 | events，RINT commit | 同 | DataCopyPad actual count |
| fallback/失败边界 | 无 Device fallback | mode非1自然分支 | 无 Device fallback | mode非1自然分支 | 非法 dtype Host 失败，未知 key no-op |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；见无后缀 A2/A3 卡 | 公开支持；见无后缀 A2/A3 卡 | 公开支持；`ascend950` 接线，当前卡覆盖 |
| Host/Launcher | 共用 `Tiling4ApplyFusedEmaAdam`，运行时查询 AIV 核数/UB | 同 | 同一源码，平台查询值可不同 |
| Kernel/资源 | 共用三 key 单 AIV Vector body | 同 | 同一 body，binary config 独立为 ascend950 |

## 限制与失败边界

- 固定 Host 仅校验主 tensor storage shape/dtype 与 `step` dtype；ACLNN 文档声明的 rank、`step=[1]`、attr 范围和 mode 0/1 在可见 Tiling 中没有对应检查，不能假设由 Kernel补验。
- `step` 从 INT64 转 FP32后做 `Power`；大 step 的整数精确性受该显式转换约束。`biasCorrection` 为 1 时若校正因子为 0，固定源码没有除零 fallback。
- `N=0` 与 `coreCalcMax=0` 均无专用保护；不得把固定仓未证明的 empty 行为写成受支持路线。
- `DataCopyParams.blockLen` 接收窄化到 UINT16 的字节数；Host 依赖平台 UB 间接约束，未写显式范围检查。
- 生成 ACLNN bridge、连续化临时 workspace 的精确 slice/回写细节不可见；算子自有 Kernel 不使用 user workspace。未知 tiling key 在 entry 中无 else 错误回报。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `README.md` | 产品支持/公式 | A2/A3/950 支持、算子语义 |
| 入口声明/约束 | `docs/aclnnApplyFusedEmaAdam.md` | 两段式函数原型 | ACLNN 参数、shape/dtype、错误语义 |
| 入口实现边界 | `op_host/apply_fused_ema_adam_def.cpp` | `ApplyFusedEmaAdam` | AutoContiguous、attr、平台配置 |
| Host/Tiling | `op_host/apply_fused_ema_adam_tiling.h` | `ApplyFusedEmaAdamTilingData` | key 常量与 POD 字段 |
| Host/Tiling | `op_host/apply_fused_ema_adam_tiling.cpp` | `GetTilingAttr/CheckInputDtype/CheckInputShape/TilingCompute/Tiling4ApplyFusedEmaAdam` | 校验、分发、资源、workspace |
| Kernel entry/body | `op_kernel/apply_fused_ema_adam.cpp` | `apply_fused_ema_adam` | key router |
| Kernel entry/body | `op_kernel/apply_fused_ema_adam_base.h` | `InitData/ScalarPow/Pipe*` | 公共参数、pow、event |
| Kernel entry/body | `op_kernel/apply_fused_ema_adam_f32.h` | `FusedEmaAdamF32` | FP32 body |
| Kernel entry/body | `op_kernel/apply_fused_ema_adam_f_bf16.h` | `FusedEmaAdamF16` | FP16/BF16 body |
| 构建接线 | `op_host/config/ascend950/apply_fused_ema_adam_binary.json` | `ApplyFusedEmaAdam_0/1/2` | 950 dtype binaries |
| 构建接线 | `op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend910b/910_93/950 活动目标 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15 字段 Host/Device 共用 | 字段顺序、FP32 attr 窄化、UINT64 mode/计数 | `ApplyFusedEmaAdamTilingData`, `InitData` |
| 调用顺序 | 单 Kernel 内 m→v→var→s | m/v 输出与未舍入 local 的消费先后 | 两个 `Compute` |
| workspace/内存 | mode/dtype 四套线性 UB 公式 | pow 64 B、depth1、mode1 slot alias | `TilingCompute`, `InitBuffer` |
| 同步 | MTE2↔V、V→MTE3、V→S event | 覆盖或 GM copy 前的等待关系 | `PipeVM2/PipeM2V/PipeVM3/ScalarPow` |
| 精度与尾块 | 低精度 FP32 compute、RINT output；实际 count tail | 不可提前以 T 值继续校正；front/tail base 公式 | `FusedEmaAdamF16::Compute/Process` |
| 失败边界 | dtype/shape/attr/core 获取失败即终止 | 不得补造 empty、非法 mode 或 unknown key fallback | Host checks + router |