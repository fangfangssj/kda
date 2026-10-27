---
type: Operator
title: "MseLoss A2/A3 ACLNN 算子知识"
description: "说明 aclnnMseLoss 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_api/aclnn_mse_loss.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/docs/aclnnMseLoss.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_api/aclnn_mse_loss.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss/op_api/mse_loss.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/common/level2_base_loss.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_host/mse_loss_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_host/mse_loss_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_host/mse_loss_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_kernel/mse_loss_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_kernel/mse_loss_v2_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_kernel/mse_loss_v2_none.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_kernel/mse_loss_v2_sum.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_kernel/mse_loss_v2_mean.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_host/config/ascend910b/mse_loss_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_host/config/ascend910_93/mse_loss_v2_binary.json
    role: implementation
created_at: "2026-08-25T17:21:38Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# MseLoss A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMseLossGetWorkspaceSize` / `aclnnMseLoss`，公开 ACLNN；非空 ND 调用在 A2/A3 经 L0 selector 进入 `MSELossV2` |
| 计算语义 | `none: y_i=(self_i-target_i)^2`；`sum: y=Σ_i(...)`；`mean: y=(1/N)Σ_i(...)` |
| 输入输出 | 两输入先提升到共同 dtype、连续化、广播到同 shape；公开接口为 ND，rank 1~8；FP16/FP32/BF16；`none` 输出广播 shape，其余输出 0 维标量 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；`sum/mean` 在 Kernel 内以 GM workspace + `SyncAll` 完成跨 AIV 归并 |
| Kernel 路线 | R-001 none-FP32；R-002 none-FP16/BF16；R-003 sum-FP32；R-004 sum-FP16/BF16；R-005 mean-FP32；R-006 mean-FP16/BF16 |
| 核心资源 | GM 输入/输出、两路 VECIN TQue、VECOUT TQue 或 FP32 cast TBuf；归约路线另有 32 B 上传区、`32*C` B 下载区与 `64*C` B 用户 workspace |
| 来源闭包 | source_closed：公开 ACLNN → 输入规范化/空域处理 → L0 `MseLoss` selector → `MSELossV2` Host/Tiling → `mse_loss_v2` entry → none/sum/mean body → A2/A3 构建接线均已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令广播后的共同 shape 元素数为 `N`，逻辑索引为 `i`。`reduction="none"` 逐元素输出平方差；`"sum"` 对全部元素求和；`"mean"` 再乘 Tiling 生成的 FP32 `scale=1/N`。有效输出域只含逻辑 `N` 个元素；Device 为满足 32 B 搬运可能访问对齐尾块，不能把对齐 padding 当成有效输出。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，rank 1~8 | FP16/FP32/BF16 | 可非连续；与 `target` 可广播；进入核心 Kernel 前已连续化并广播 |
| `target` | ND，rank 1~8 | FP16/FP32/BF16 | 与 `self` 做 dtype promotion；两输入最终 dtype/shape 相同 |
| `reduction` | 标量字符串 | `char *` | 仅 `none`、`mean`、`sum`；内部编码分别形成 key 十位 1、3、2 |
| `out` | `none` 为广播 shape；其余为 0 维 | promotion 后可再 cast 到调用方 out dtype | `none` 要求 out shape 等于广播 shape；归约要求 out dim 数为 0 |

### 入口与版本

公开入口由 `aclnn_mse_loss.h/.cpp` 实现。A2/A3 对非空公共 ND 域，经 Contiguous/Cast/BroadcastTo 后两个输入 view shape、format 相同，`l0op::MseLoss` 的 SoC/shape/format 谓词命中 `MSELossV2`。`MseLossV1` 在固定仓仅声明 `ascend950` 配置，不是本卡 A2/A3 的闭合实现。`aclnnMseLossOut` 与 backward 语义均不属于本逻辑算子。

## Host/Launcher 执行流程

### 参数校验与前处理

顺序为：检查 `self/target/out` 非空；限制两输入 rank 不超过 8；检查 reduction；推导广播 shape 并校验 out shape；按当前 SoC 校验 dtype/promotion。随后分别执行 `Contiguous`、必要的 `Cast`，再对 shape 不同者 `BroadcastTo`。空 Tensor 在进入 L0 selector 前截断：`none` 直接成功，`mean` 通过 `Fill(NAN)` 写 out，`sum` 通过 `Fill(0)` 写 out。

### 参数、Workspace 与生命周期

`aclnnMseLossGetWorkspaceSize` 创建 `aclOpExecutor`，把规范化算子和 MSE 节点加入 executor，并返回 executor 统计的 workspace。非空归约核心节点申请 `systemWorkspace + 32*C + 32*C`：前一用户区供 `SyncAll`，后一用户区每核保存 8 个 FP32；none 核心节点为 0。调用方持有 workspace，`aclnnMseLoss` 在给定 stream 上以 `CommonOpExecutorRun` 消费 executor；任何添加节点、Tiling 或运行失败均返回 ACLNN 错误。

### 启动图与失败传播

```text
aclnnMseLossGetWorkspaceSize
  -> CheckParams -> PromoteType -> Contiguous -> Cast -> BroadcastTo
  -> empty? none:return / mean:Fill(NaN)->ViewCopy / sum:Fill(0)->ViewCopy
  -> l0op::MseLoss -> selector(A2/A3,same shape,same ND) -> MSELossV2
  -> Cast(loss,out dtype) -> ViewCopy(out)
aclnnMseLoss(workspace,executor,stream)
  -> CommonOpExecutorRun
     -> [单节点] mse_loss_v2<<<blockDim=C, AIV, stream>>>(input,target,loss,workspace,POD)
```

输入规范化节点的输出由 executor 管理，直到其最后消费者完成；MSE 输出经 Cast/ViewCopy 交付 `out`。核心归约 workspace 仅在本次 executor/stream 执行期有效。空域分支不启动 `mse_loss_v2`，其 Fill 失败直接向 ACLNN 返回。

## Route 与参数设计

### 分核与资源计算

设 `e=8(FP32)` 或 `16(FP16/BF16)`，`B=floor(N/e)`、`r=N mod e`、硬件 AIV 数 `H`。Tiling 令 `C=max(1, ceil(B/128))` 当 `B<128H`，否则 `C=H`；`b=floor(B/C)`，余块 `q=B mod C`，普通核长度 `b*e`，最后核多 `q*e`。UB 以 32 B block 计：none-FP32 每 queue block 数为 `floor((U/32)/(3*d))`；none-16 位为 `floor((U/32)/(3*d+4))`；归约先扣 `32*(C+1)` B，再以 `2*d` 或 `2*d+4` 为分母。先用 `d=1`，若完整 epoch 数大于 1 切换 `d=2` 并重算。`tileLength=queueBlocks*e`，普通核和尾核分别生成 epoch/tail。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 none-FP32 | 11 / `MSELossV2None<float>` | A2/A3、非空、共同 ND shape、dtype FP32、reduction none | dtype 16 位去 R-002；其他 reduction 继续 |
| 2 | R-002 none-FP16/BF16 | 12/13 / `MSELossV2None<T>` | 同上，dtype FP16/BF16 | 不支持 dtype 失败 |
| 3 | R-003 sum-FP32 | 21 / `MSELossV2Sum<float>` | dtype FP32、sum | 16 位去 R-004 |
| 4 | R-004 sum-FP16/BF16 | 22/23 / `MSELossV2Sum<T>` | dtype FP16/BF16、sum | 不支持 dtype 失败 |
| 5 | R-005 mean-FP32 | 31 / `MSELossV2Mean<float>` | dtype FP32、mean | 16 位去 R-006 |
| 6 | R-006 mean-FP16/BF16 | 32/33 / `MSELossV2Mean<T>` | dtype FP16/BF16、mean | 不支持 reduction/dtype 失败 |

```text
empty -> ACLNN empty branch（不进入 R）
nonempty -> A2/A3 && same shape/format && ND -> MSELossV2
  reduction none -> FP32:R-001; FP16/BF16:R-002
  reduction sum  -> FP32:R-003; FP16/BF16:R-004
  reduction mean -> FP32:R-005; FP16/BF16:R-006
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 11 | A2/A3，N>0，FP32，none | 单 AIV Kernel | `MSELossV2None<float>::Process` | 双输入/单输出 TQue，1/2 buffer | 尾部按 8 元素向上对齐 |
| R-002 | 12/13 | A2/A3，N>0，FP16/BF16，none | 单 AIV Kernel | `MSELossV2None<half/bfloat16_t>` | 两输入 TQue、两 FP32 cast TBuf、输出 TQue | 尾部按 16 元素向上对齐 |
| R-003 | 21 | A2/A3，N>0，FP32，sum | 单 AIV Kernel、核内两级归约 | `MSELossV2Sum<float>` | FP32 partial workspace；`SyncAll` | 最后核读额外 8 元素对齐块 |
| R-004 | 22/23 | A2/A3，N>0，FP16/BF16，sum | 同 R-003 | `MSELossV2Sum<T>` | cast 后 FP32 累加；最终 RINT | 最后核搬运长度额外增加 16 个输入 T 元素，计算仍取逻辑长度 |
| R-005 | 31 | A2/A3，N>0，FP32，mean | 单 AIV Kernel、核内两级归约 | `MSELossV2Mean<float>` 继承 sum | partial 先乘 `1/N`；`SyncAll` | 同 FP32 sum |
| R-006 | 32/33 | A2/A3，N>0，FP16/BF16，mean | 同 R-005 | `MSELossV2Mean<T>` | FP32 算术，最终 RINT | 同 16 位 sum |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| V2 selector | A2/A3 且 shape/format 相同且 ND/NCL | V2 body没有广播逻辑 | 全部 | L0 V1；但固定仓 A2/A3 无该实现配置，因此不构成本卡支持路线 | `l0op::MseLoss` |
| 128 blocks/core | `B < 128H` 时缩减 C | 避免每核工作量过小 | 全部 | 使用全部 H 核 |
| 自适应 buffer | 以单 buffer 算得 `epochs>1` | 多 epoch 才为流水重叠支付双缓冲 UB | 全部 | `bufferNum=1` |
| 空域 | `N==0` | V2 `CoreSplit`拒绝零长度 | 全部 | ACLNN empty Fill/return |
| 归约 workspace | reduction != none | 每核需 32 B partial 且 `SyncAll` 需 32 B/核 | R-003~006 | none workspace=0 |

### POD/ABI

固定仓通过 `BEGIN_TILING_DATA_DEF` 生成 ABI，宏展开后的 `sizeof/alignof/offsetof` 不在限定源码中；字段顺序和声明位宽如下，不臆测 padding。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `MSELossV2TilingData` | `coreNum` | uint64/64 | 第 1 字段；offset 固定仓不可见 | AIV 核 | `CoreSplit` | sum 下载范围/资源 | >=1 |
| 同上 | `tailElems` | uint64/64 | 第 2 字段 | 元素 | `CoreSplit` | 最后核尾块 | `<e` |
| 同上 | `bufferNum` | uint64/64 | 第 3 字段 | buffer 个数 | `UBSplit` | `InitBuffer` depth | 1 或 2 |
| 同上 | `scale` | float/32 | 第 4 字段 | 无量纲 | `CoreSplit: 1/N` | mean | N>0 |
| 同上 | `epochs` | uint64/64 | 第 5 字段 | tile 次数 | `TileSplit` | 普通核循环 | floor |
| 同上 | `epochsForLastCore` | uint64/64 | 第 6 字段 | tile 次数 | `TileSplit` | 最后核循环 | floor |
| 同上 | `coreLength` | uint64/64 | 第 7 字段 | 元素 | `CoreSplit` | GM byte offset | `b*e` |
| 同上 | `tileLength` | uint64/64 | 第 8 字段 | 元素 | `UBSplit` | queue/copy/vector | 32 B 对齐 |
| 同上 | `tailTileLength` | uint64/64 | 第 9 字段 | 元素 | `TileSplit` | 普通核尾轮 | 可为 0 |
| 同上 | `tailTileLengthForLastCore` | uint64/64 | 第 10 字段 | 元素 | `TileSplit` | 最后核尾轮 | 可为 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 11 | `GetTilingAttr/CoreSplit/UBSplit/TileSplit` | `MSELossV2TilingData` | `mse_loss_v2`, key 11 | `MSELossV2Base<float>` + `MSELossV2None<float>` | `mse_loss_v2/op_host/CMakeLists.txt` |
| R-002 | 12/13 | 同上 | 同上 | entry keys 12/13 | generic base + none | 同上 |
| R-003 | 21 | 同上；workspace producer | 同上 | entry key 21 | float base + sum | 同上 |
| R-004 | 22/23 | 同上 | 同上 | entry keys 22/23 | generic base + sum | 同上 |
| R-005 | 31 | 同上，`scale=1/N` | 同上 | entry key 31 | float base + mean→sum | 同上 |
| R-006 | 32/33 | 同上 | 同上 | entry keys 32/33 | generic base + mean→sum | 同上 |

## Kernel 实现

### 公共执行流程

`mse_loss_v2` 读取 POD 和扣除系统区后的 user workspace，以编译期 tiling key 实例化唯一 body。每核 GM 起点为 `coreLength*sizeof(T)*blockIdx`；最后核替换为专属 epoch/tail 并附加 `tailElems`。所有路线执行 `CopyIn → FP32 Sub → FP32 Mul`；none 直接 cast/copy out，sum/mean 先 tile 内二分 Add，再核间归约。执行族始终是 AIV_VECTOR，不含 Cube、SIMT 或 NDDMA 节点。

### Route R-001：none-FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | `CheckInputDtype/GetTilingAttr` | FP32 + none | key 11 |
| Router | `mse_loss_v2.cpp` | `TILING_KEY_IS(11)` | key 11 | `MSELossV2None<float>` |

#### Device 数据流

##### 函数级调用链

```text
mse_loss_v2 -> MSELossV2None<float>::Init -> Process
  -> MSELossV2Base<float>::CopyIn -> Sub -> Mul -> CopyOut
```

##### 源码等价伪码

```text
for full tile: copy self,target; target=self-target; out=target*target; copy out
if tail: aligned=ceil((tailTile+tailElems)/8)*8; repeat same; logical valid=N
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM inputs | `coreLength*blockIdx*4` B | core/tail length | FP32 ND | executor | CopyIn | caller/executor | 32 B tile |
| VECIN q0/q1 | 0 / 独立 queue | 各 `d*tileLength*4` B | FP32 | MTE2 | Vector | TPipe | depth d |
| VECOUT | 独立 queue | `d*tileLength*4` B | FP32 | Vector | MTE3 | TPipe | depth d |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  q0["inQ0 · d·L·4 B"] q1["inQ1 · d·L·4 B"] qo["outQ · d·L·4 B"]
```

##### 字节账与复用顺序

```text
UB_total = 3*d*tileLength*4 <= U; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| q0/q1 slot | CopyIn | Compute | 下一 tile | DeQue/FreeTensor | Vector 已消费 |
| outQ slot | Compute | CopyOut | 下一 tile | EnQue/DeQue/FreeTensor | MTE3 已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile pipe | TQue | MTE2/Vector | Vector/MTE3 | UB slot | 全 FP32；尾块 8 元素对齐，逻辑域 N |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | `UBSplit/TileSplit` | 生成 d、L、epoch/tail |
| Entry | `mse_loss_v2.cpp` | key 11 | 路由 float none |
| Body | `mse_loss_v2_none.h` | `MSELossV2None<float>` | Sub/Mul/CopyOut |

### Route R-002：none-FP16/BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | dtype key | FP16/BF16 + none | 12/13 |
| Router | `mse_loss_v2.cpp` | keys 12/13 | A2/A3 | generic none body |

#### Device 数据流

##### 函数级调用链

```text
mse_loss_v2 -> MSELossV2None<T> -> CopyIn+Cast(FP32) -> Sub -> Mul -> Cast(RINT,T) -> CopyOut
```

##### 源码等价伪码

```text
for tile: copy T inputs; cast both to fp32; square difference; cast RINT to T; copy
tail copy/vector length=ceil((tailTile+tailElems)/16)*16; valid=N
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input queues | core byte offset | `2*d*L*2` B | FP16/BF16 | MTE2 | Cast | TPipe | depth d |
| castBuf0/1 | UB | `2*L*4` B | FP32 | Cast | Sub/Mul | TPipe | depth 1 |
| output queue | UB | `d*L*2` B | T | Cast | MTE3 | TPipe | depth d |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  q0["in0 d·L·2"] q1["in1 d·L·2"] c0["cast0 L·4"] c1["cast1 L·4"] qo["out d·L·2"]
```

##### 字节账与复用顺序

```text
UB_total=(3*d+4)*L*2 <= U; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| castBuf0/1 | Cast | Sub/Mul | 下一 tile | 同一顺序 Process | 当前 Vector 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| cast pipeline | TQue + TBuf | MTE2/Cast | Vector/MTE3 | 输入/输出 slot | 算术 FP32，输出 RINT；尾块 16 元素对齐 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | `UBSplit` | 分母增加 4 个 32 B 等价 cast buffer |
| Entry | `mse_loss_v2.cpp` | keys 12/13 | FP16/BF16 实例 |
| Body | `mse_loss_v2_base.h`, `mse_loss_v2_none.h` | generic templates | FP32 中间、RINT 输出 |

### Route R-003：sum-FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | FP32 + sum | key 21、schedule mode 1 | R-003 |
| Router | `mse_loss_v2.cpp` | key 21 | 编译期命中 | `MSELossV2Sum<float>` |

#### Device 数据流

##### 函数级调用链

```text
CopyIn -> Sub/Mul -> ReduceSumBisect(tile) -> Add(core partial)
-> DataCopy(workspace[core]) -> SyncAll -> core0 ReduceSum(C*8) -> DataCopyPad(out scalar)
```

##### 源码等价伪码

```text
acc[8]=0; for tile: square_diff_fp32; binary_add_to_<=8; acc+=partial
ws[C+core][0:8]=acc; SyncAll; if core==0: out=ReduceSum(ws[C:C+C][0:8])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input queues | core offset | `2*d*L*4` B | FP32 | MTE2 | Vector | TPipe | depth d |
| uploadQue/outLocal | UB | 32 B | FP32[8] | core reduce | MTE3 | TPipe | 固定 32 B |
| downloadQue | UB | `32*C` B | FP32[C,8] | MTE2 | core0 reduce | TPipe | 一次跨核下载 |
| user workspace | `32*C + 32*core` | `32*C` partial 区；另有 `32*C` sync 区 | FP32 | 各核 | core0 | executor | 每核 32 B |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  ins["inQ0/1 · 2dL4 B"]:2 up["upload · 32 B"] down["download · 32C B"]
```

##### 字节账与复用顺序

```text
UB_total=2*d*L*4 + 32*(C+1) <= U
workspace_user=32*C(sync)+32*C(partials); total adds system workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| partial[core] | 当前 core | core0 | Kernel 结束后 | SyncAll | 每核唯一 32 B slice |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 跨核 | `SyncAll` | 所有核 partial | core0 GM load | workspace partial | FP32；尾核额外对齐读 8 元素，计算只用逻辑长度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | `GetTilingData` | 64*C 用户 workspace |
| Entry | `mse_loss_v2.cpp` | key 21 | float sum |
| Body | `mse_loss_v2_sum.h` | `ReduceSumBisect/CopyOut` | 两级归约和 barrier |

### Route R-004：sum-FP16/BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | 16 位 + sum | keys 22/23 | R-004 |
| Router | `mse_loss_v2.cpp` | 22/23 | dtype specialization | generic sum |

#### Device 数据流

##### 函数级调用链

```text
CopyIn(T)->Cast FP32->Sub/Mul->ReduceSumBisect->workspace->SyncAll->core0 ReduceSum->Cast RINT(T)->DataCopyPad scalar
```

##### 源码等价伪码

```text
与 R-003 相同，但两输入进入 UB 后先转 FP32，最终标量才 RINT 转回 FP16/BF16。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input+cast | core offset/UB | `2*d*L*2 + 2*L*4` B | T→FP32 | MTE2/Cast | Vector | TPipe | d + 两 TBuf |
| upload/download | UB | `32+32*C` B | FP32/T | Vector/MTE | core0 | TPipe | 固定块 |
| user workspace | 同 R-003 | `64*C` B | FP32 | cores | core0 | executor | 32 B/核 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  ins["inputs 2dL2"] casts["casts 2L4"]:2 up["32 B"] down["32C B"]
```

##### 字节账与复用顺序

```text
UB_total=(2*d*L*2)+(2*L*4)+32*(C+1) <= U; workspace_user=64*C
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| cast TBuf | Cast | square/reduce | 下一 tile | 顺序 Vector 指令 | 当前 tile 完成 |
| partial | each core | core0 | Kernel 结束 | SyncAll | 唯一 slice |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 归约 | TQue + SyncAll | MTE2/cores | Vector/core0 | UB/GM | FP32 累加，最终 RINT；逻辑尾长度排除 padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | keys 22/23 | 16 位资源公式 |
| Entry | `mse_loss_v2.cpp` | keys 22/23 | generic sum |
| Body | `mse_loss_v2_sum.h` | `MSELossV2Sum<T>` | FP32 partial、RINT scalar |

### Route R-005：mean-FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | FP32 + mean | key 31、`scale=1/N` | R-005 |
| Router | `mse_loss_v2.cpp` | key 31 | float | mean body |

#### Device 数据流

##### 函数级调用链

```text
MSELossV2Mean<float>::Process -> inherited Sum pipeline -> ReduceSumBisect(tile,scale) -> workspace -> SyncAll -> final sum
```

##### 源码等价伪码

```text
for tile: partial=binary_sum(square_diff); partial*=scale; coreAcc+=partial
cross_core_sum(coreAcc) -> out
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 全部资源 | 同 R-003 | `2*d*L*4 + 32*(C+1)` UB；`64*C` user WS | FP32 | inherited sum | inherited sum | TPipe/executor | 32 B |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  in["inputs · 2dL4 B"] acc["upload · 32 B"] down["download · 32C B"]
```

##### 字节账与复用顺序

```text
UB_total=2*d*L*4+32*(C+1); workspace_user=64*C
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| partial | cores | core0 | end | SyncAll | 同 R-003 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| mean | `Muls(scale)` + SyncAll | tile/core | core acc/core0 | FP32 partial | 每 tile 先乘 FP32 `1/N`；尾规则同 R-003 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | `CoreSplit` | FP32 scale |
| Entry | `mse_loss_v2.cpp` | key 31 | float mean |
| Body | `mse_loss_v2_mean.h` | `MSELossV2Mean<float>` | 继承 sum，只改 scale |

### Route R-006：mean-FP16/BF16

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | FP16/BF16 + mean | keys 32/33 | R-006 |
| Router | `mse_loss_v2.cpp` | keys 32/33 | A2/A3 | generic mean |

#### Device 数据流

##### 函数级调用链

```text
MSELossV2Mean<T> -> generic Sum pipeline with scale -> FP32 cross-core sum -> Cast RINT(T) -> scalar out
```

##### 源码等价伪码

```text
cast inputs to fp32; per-tile square/reduce/scale; cross-core sum; cast once to output T
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 全部资源 | 同 R-004 | `(2*d*L*2)+(2*L*4)+32*(C+1)` UB；`64*C` user WS | T/FP32 | inherited sum | inherited sum | TPipe/executor | 32 B |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in["inputs 2dL2"] cast["casts 2L4"] acc["32 B"] down["32C B"]
```

##### 字节账与复用顺序

```text
UB_total=(2*d*L*2)+(2*L*4)+32*(C+1); workspace_user=64*C
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| cast/partial | Cast/cores | Vector/core0 | tile/end | queue + SyncAll | 同 R-004 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| mean | TQue/Muls/SyncAll | MTE2/tiles/cores | Vector/core0 | UB/WS | FP32 运算，最终 RINT；尾规则同 R-004 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Tiling | `mse_loss_v2_tiling.cpp` | key 32/33 | 16 位 mean |
| Entry | `mse_loss_v2.cpp` | keys 32/33 | generic mean |
| Body | `mse_loss_v2_mean.h` | `MSELossV2Mean<T>` | scale + inherited reduction |

## 算子特定优化

### Trick T-001：按 epoch 数自适应单双缓冲

#### 触发与源码依据

- 适用 Route：R-001~R-006
- 触发条件：先以 `bufferNum=1` 计算容量；仅当 `floor(blockPerCore/blockPerQue)>1` 时改为 2 并重算。
- Golden 来源：[`mse_loss_v2_tiling.cpp:180-189`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_host/mse_loss_v2_tiling.cpp#L180-L189)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint8_t bufferNum = 1u;
CalcBlockPerQue(bufferNum);
uint32_t epochs = Ops::Base::FloorDiv(optiling::blockPerCore, optiling::blockPerQue);
if (epochs > 1u) {
    bufferNum = 2u; // double buffer
    CalcBlockPerQue(bufferNum);
}
```

#### 机制与设计原因

只有多 tile 核才能用 ping-pong 覆盖 MTE/Vector；单 tile 时保留单 buffer，把 UB 让给更大 tile。切换 buffer 数后必须重算 `blockPerQue`，否则 Host 容量与 Device `InitBuffer` 不一致。

#### 迁移边界

- 适用：固定深度 TQue、可按 tile 顺序流水的 VV 算子。
- 前提：UB 公式包含全部 queue/TBuf，且 Device depth 直接取 POD `bufferNum`。
- 失效：存在跨 tile 常驻数据或异步生命周期跨越两个以上 tile 时需重做生命周期分析。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定使用双缓冲会缩小单 tile 容量；只改 `bufferNum` 不重算 tile 会造成资源公式与实际分配不闭合。

### Trick T-002：两级二分归约与每核独占 partial

#### 触发与源码依据

- 适用 Route：R-003~R-006
- 触发条件：reduction 为 sum/mean。
- Golden 来源：[`mse_loss_v2_sum.h:77-106`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/mse_loss_v2/op_kernel/mse_loss_v2_sum.h#L77-L106)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
while (tileLength > this->FLOATS_PER_BLOCK) {
    offset = (tileLength + 15u) >> this->ALIGN_SHIFT << this->OFFSET_SHIFT;
    AscendC::Add<float>(srcLocal, srcLocal, srcLocal[offset], tileLength - offset);
    tileLength = offset;
}
AscendC::DataCopy(this->usrGlobal, this->outLocal, this->FLOATS_PER_BLOCK);
AscendC::SyncAll();
```

#### 机制与设计原因

tile 内以就地二分 Add 收敛至 8 个 FP32，核内累加后写独占 32 B；`SyncAll` 后仅 core0 下载所有 partial 并最终归约，避免 GM atomic。mean 复用同一结构，只把 tile partial 乘以 `1/N`。

#### 迁移边界

- 适用：标量全归约、AIV 数较小且 `32*C` 可装入 UB 的场景。
- 前提：workspace 为 `SyncAll` 和 partial 各预留 `32*C` B；core0 下载区容量也为 `32*C`。
- 失效：输出非标量、partial 大于单核 UB 或平台无相同 `SyncAll` 协议时应换调度。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

去掉 barrier 就让 core0 可能读取尚未提交的 partial；把 partial 起点误写成 user workspace 开头会覆盖 `SyncAll` 保留区。

## 模板与实现对比

| 对比维度 | R-001 none-FP32 | R-002 none-16bit | R-003/005 reduce-FP32 | R-004/006 reduce-16bit | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | none+FP32 | none+FP16/BF16 | sum/mean+FP32 | sum/mean+FP16/BF16 | key 十位是 reduction，个位是 dtype |
| 模板与 dtype | float 特化 | generic T | float 特化 sum/mean | generic T sum/mean | 16 位统一用 FP32 中间 |
| 循环与分核 | 每核逐 tile | 同左 | tile→core→global | 同左 | reduction 需要第二级 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 queues | 3 queues+2 cast TBuf | 2 in queues+upload/download | 再加 2 cast TBuf | 固定仓无 dynamic UB/通信 window |
| 内存总占用与复用 | `3dL4` | `(3d+4)L2` | `2dL4+32(C+1)` | `2dL2+2L4+32(C+1)` | Tiling 与 InitBuffer 对应 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy/Sub/Mul | +Cast | +Add/ReduceSum/SyncAll | +Cast RINT | 全为 AIV_VECTOR |
| 同步、精度与 tail | queue；FP32；8 对齐 | queue；FP32中间；16 对齐 | queue+barrier；FP32 | 同左+最终 RINT | padding 不进入逻辑归约长度 |
| fallback/失败边界 | 空域不入核 | 同左 | workspace/Tiling 失败 | 同左 | 不猜测 V1 A2/A3 支持 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，V2 | 支持，V2 | 支持，但走 V1/arch35，见 `_950` 卡 |
| Host/Launcher | selector 命中 `ASCEND910B` | selector 命中 `ASCEND910_93` | selector 不命中 V2 |
| Kernel/资源 | V2 手写 TQue/TBuf + workspace | 与 A2 同源实现 | ATVOSS Elementwise/ReduceSch，不复用 V2 POD |

## 限制与失败边界

- 公开接口只接受 reduction `none/mean/sum`、rank 1~8 和受支持 dtype；广播或 out shape 校验失败即返回。
- V2 Tiling 明确拒绝 `N==0`；公开空域已在上层以 return/Fill 闭合，不能直接调用 V2 模拟该路径。
- 对齐尾搬运的物理 GM 访问范围大于逻辑 N；固定仓未给出调用方分配边界证明，本卡只声明逻辑有效域，不把 padding 作为有效结果。
- POD 宏展开后的精确 `sizeof/alignof/offsetof` 在限定固定仓不可见；字段顺序和位宽可追溯，padding 未推测。
- 未处理 `aclnnMseLossOut` 或 backward/grad 变体；未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/mse_loss/op_api/aclnn_mse_loss.h` | ACLNN declarations | 两阶段 API |
| 入口声明/约束 | `loss/mse_loss/docs/aclnnMseLoss.md` | API 文档 | shape/dtype/platform |
| 入口实现 | `loss/mse_loss/op_api/aclnn_mse_loss.cpp` | `CheckParams`, workspace/API funcs | 校验、预处理、空域、executor |
| 入口实现 | `loss/mse_loss/op_api/mse_loss.cpp` | `l0op::MseLoss` | V2/V1 selector |
| 入口实现 | `loss/common/level2_base_loss.h` | dtype/reduction/Fill helpers | 公共校验与空域 Fill |
| Host/Tiling/Launcher | `loss/mse_loss_v2/op_host/mse_loss_v2_def.cpp` | `MSELossV2` OpDef | A2/A3 dtype/format |
| Host/Tiling/Launcher | `loss/mse_loss_v2/op_host/mse_loss_v2_tiling.h` | `MSELossV2TilingData` | POD 字段 |
| Host/Tiling/Launcher | `loss/mse_loss_v2/op_host/mse_loss_v2_tiling.cpp` | `TilingForMSELossV2` | key/分核/UB/workspace |
| Kernel entry/body | `loss/mse_loss_v2/op_kernel/mse_loss_v2.cpp` | `mse_loss_v2` | key router |
| Kernel entry/body | `loss/mse_loss_v2/op_kernel/mse_loss_v2_base.h` | `MSELossV2Base` | GM offset/queues/cast |
| Kernel entry/body | `loss/mse_loss_v2/op_kernel/mse_loss_v2_none.h` | `MSELossV2None` | elementwise body/tail |
| Kernel entry/body | `loss/mse_loss_v2/op_kernel/mse_loss_v2_sum.h` | `MSELossV2Sum` | 二级归约/workspace/barrier |
| Kernel entry/body | `loss/mse_loss_v2/op_kernel/mse_loss_v2_mean.h` | `MSELossV2Mean` | scale/继承 sum |
| 构建接线 | `loss/mse_loss_v2/op_host/CMakeLists.txt` | `add_modules_sources` | V2 Host 模块接线 |
| 构建接线 | `loss/mse_loss_v2/CMakeLists.txt` | `add_subdirectory` | V2 子目录总接线 |
| 构建接线 | `loss/mse_loss_v2/op_host/config/ascend910b/mse_loss_v2_binary.json` | `MSELossV2` binary list | A2 三种 dtype binary 配置 |
| 构建接线 | `loss/mse_loss_v2/op_host/config/ascend910_93/mse_loss_v2_binary.json` | `MSELossV2` binary list | A3 三种 dtype binary 配置 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 十字段 POD 可整体复用 | 顺序、位宽、元素单位；不能猜 padding | `MSELossV2TilingData` |
| 调用顺序 | 规范化→selector→单 Kernel→cast/view | 空域必须在 selector 前截断 | `aclnnMseLossGetWorkspaceSize` |
| workspace/内存 | reduction 的双 `32*C` slice | partial 从 `32*C+32*core` 开始 | `GetTilingData`, `MSELossV2Sum::Init` |
| 同步 | 每核写 partial 后 `SyncAll` | core0 不得提前读 | `MSELossV2Sum::CopyOut` |
| 精度与尾块 | 16 位输入以 FP32 算术，最终 RINT | tail compute 使用逻辑长度，搬运按块对齐 | base/none/sum templates |
| 失败边界 | 空域、dtype、shape、reduction 有明确终点 | 不把固定仓无配置的 V1 当 A2/A3 fallback | API checks / L0 selector |
