---
type: Operator
title: "InplaceCelu 950 ACLNN 算子知识"
description: "说明 aclnnInplaceCelu 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/docs/aclnnCelu&aclnnInplaceCelu.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_api/aclnn_celu.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_api/aclnn_celu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_api/celu_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_host/celu_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/celu_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T19:30:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceCelu 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnInplaceCeluGetWorkspaceSize` / `aclnnInplaceCelu`；本文不包含非原地 `aclnnCelu`。 |
| 计算语义 | 对 `selfRef` 每个逻辑元素计算 `max(0,x)+min(0,alpha*(exp(x/alpha)-1))`，最终写回同一个用户 Tensor。 |
| 输入输出 | `selfRef` 为 0～8 维 ND Tensor，950 支持 FLOAT/FLOAT16/BFLOAT16，允许非连续；输入和最终输出是同一对象。 |
| 执行类型 | 算子自有 `CeluV2` 节点为单 Kernel `AIV_VECTOR`；外围 `Contiguous`、`Cast`、`ViewCopy` 是框架边界。 |
| Kernel 路线 | R-001：FLOAT 原生 FP32；R-002：FLOAT16/BFLOAT16 升到 FP32 计算后降精度。两条均使用 `schMode=0`。 |
| 核心资源 | GM 输入/中间输出；depth-2 VECIN/VECOUT `TQue`；两个 FP32 `TBuf`；节点系统 workspace=`512+libApiWorkspace`。 |
| 来源闭包 | `source_closed`：公开原地 ACLNN、检查与 executor 编排、`CeluV2` launcher、950 Tiling/POD、entry/body 和 ascend950 构建接线闭合；通用框架节点明确作为边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令调用前 `selfRef[i]=x_i`，逻辑元素总数为 `N`。成功执行后：

```text
z_i = x_i / alpha
neg_i = min(0, alpha * (exp(z_i) - 1))
pos_i = max(0, x_i)
selfRef[i] = pos_i + neg_i,  i in [0,N)
```

“原地”发生在 ACLNN 图尾：实现先把可能非连续的 `selfRef` 连续化，`CeluV2` 写一个 executor 分配的中间 Tensor，再经 `Cast` 和 `ViewCopy` 写回 `selfRef`。因此 Kernel 的 `x`、`y` 并非同一 GM 指针，不能把 Device body 误写成原址覆盖。

API 的 `CheckAttributeValue` 以 `alpha>0 || alpha<0` 判断有效性，所以精确 0、`-0` 和 NaN 被拒绝；无穷值可通过这层。950 Tiling 另外要求 `fabs(alpha)>=1e-7f`。空 Tensor 在 ACLNN 第一段直接返回，workspace 为 0，不加入任何 Device 节点。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND，0～8 维；允许非连续 | FLOAT、FLOAT16、BFLOAT16 | 必填、读写；950/Regbase 选择三 dtype 支持表。最终输出仍写入该对象。 |
| `alpha` | Host scalar | 可转换为 FLOAT | 必填；L0 接收 `alpha->ToFloat()`；API 拒绝 0/NaN，950 Tiling 拒绝绝对值小于 `1e-7`。 |
| `workspaceSize` | Host 标量指针 | `uint64_t*` | 第一段返回整个 executor 图所需 Device workspace 字节数。 |
| `executor` | Host 对象指针 | `aclOpExecutor**` | 第一段返回已编排图并移交所有权。 |
| `workspace` | Device 地址 | `void*` | 第二段输入；必须与同次第一段返回的大小相配。 |
| `stream` | ACL stream | `aclrtStream` | `CommonOpExecutorRun` 在此 stream 上执行。 |

### 入口与版本

固定 Commit 的公开原型位于 `activation/celu_v2/op_api/aclnn_celu.h`，产品文档明确支持 Ascend 950PR/950DT。`aclnnInplaceCeluGetWorkspaceSize` 把同一 `selfRef` 同时传给 `GetWorkspaceSizeCommon` 的输入、输出位置；第二段 `aclnnInplaceCelu` 调用 `CommonOpExecutorRun`。

内部实际算子版本是 `CeluV2`：`l0op::CeluV2` 通过 `ADD_TO_LAUNCHER_LIST_AICORE` 添加任务；稳定目录的 CMake 只接线 `ascend950` 与 `arch35`，OpDef 的 `AddConfig("ascend950")` 指向 `celu_v2_apt`。不同逻辑语义的公开 `aclnnCelu` 以及 `activation/celu/` 不并入本文。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段的严格顺序如下：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize`、`executor` 输出指针。
2. `CREATE_EXECUTOR` 创建唯一 executor，失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
3. `CheckParams(selfRef,alpha,selfRef)` 依次执行非空、dtype、shape/rank、alpha 检查。由于输入输出是同一 Tensor，shape 相等和 dtype 可转换检查仍被调用但自然满足。
4. `selfRef->IsEmpty()` 时设置 `*workspaceSize=0`，移交空图 executor 并成功返回。
5. `l0op::Contiguous(selfRef)` 得到连续输入，失败阻断后续。
6. `l0op::CeluV2(contiguousSelf,alpha->ToFloat())` 按输入 view shape/dtype 分配连续 ND 中间输出并添加 AICore 节点。
7. `l0op::Cast(celuV2Out,selfRef->GetDataType())` 产生目标 dtype 表示；是否消除为复用属于框架边界。
8. `l0op::ViewCopy(castOut,selfRef)` 把结果写回可能非连续的原用户 view。
9. 读取 executor 总 workspace，移交 executor 所有权。

任何中间 Tensor 为空或 launcher 注册失败都立即返回内部错误；不会继续编排图尾。

### 参数、Workspace 与生命周期

`selfRef` 由调用者拥有；连续输入、`CeluV2` 中间输出、Cast 表示和任务依赖由 executor 持有到第二段结束。`CeluV2TilingFunc` 为自有节点写一个 workspace slice：

```text
celu_node_workspace = 512 + PlatformAscendC::GetLibApiWorkSpaceSize()
```

`celu_v2` entry 接收 `workspace` 但 body 不解引用，因而该 slice 是系统/库 API 预留。整个 ACLNN 图的 workspace 还可能包含框架 `Contiguous/Cast/ViewCopy` 需求；固定算子仓看不到它们的 offset、extent 与复用策略，只能以 `executor->GetWorkspaceSize()` 为权威总量。第二段把调用者 workspace、大小和 stream 一并交给 `CommonOpExecutorRun`。

### 启动图与失败传播

```text
selfRef(read)
  -> F0 Contiguous
  -> K0 CeluV2 / celu_v2<schMode=0> [AIV_VECTOR]
  -> F1 Cast(to selfRef dtype)
  -> F2 ViewCopy(to selfRef view)
  -> selfRef(write)
```

| 节点 | Host 添加函数 / entry | 网格与 POD | 输入、输出与 workspace | ordering、owner/lifetime、失败传播 |
|---|---|---|---|---|
| F0 | `GetWorkspaceSizeCommon -> l0op::Contiguous` | 框架决定 | 用户 `selfRef` -> 连续表示；slice 不可见 | executor 持有；失败阻断 K0。 |
| K0 | `l0op::CeluV2 -> ADD_TO_LAUNCHER_LIST_AICORE` / `celu_v2<0>` | `blockDim=finalCoreNum`；`CeluV2TilingData` | 连续 GM x -> 中间 GM y；节点 workspace=`512+libApiWorkspace` | 同一 executor/stream 依赖 F0；注册/运行失败阻断 F1/F2。 |
| F1 | `l0op::Cast` | 框架决定 | K0 中间 -> selfRef dtype 表示 | 同一图依赖 K0；失败阻断 F2。 |
| F2 | `l0op::ViewCopy` | 框架决定 | Cast 表示 -> 用户 `selfRef` view | 最终原地提交；第一段空返回或第二段执行失败向 ACLNN 返回。 |

固定仓没有算子自有 event 或跨 stream 协议；依赖可见性由 executor 在调用 stream 上保证。空 Tensor 路线无 K0/F0/F1/F2，workspace 为 0。

## Route 与参数设计

### 分核与资源计算

令 `U=UB bytes`、`C=GetCoreNumAiv()`、`s=sizeof(T)`、`N=storage element count`：

```text
B = ceil(N*s / 32)                           # 32-byte block 数
C' = max(1, min(C,B))                        # finalCoreNum
q = floor(B/C')
r = B mod C'                                 # 前 r 核为 big core

ubDataNumber = 4, s<=2                       # 两组 depth-2 Queue + 两块 FP32 TBuf
             = 3, s==4
tileBlockNum = max(1, floor(floor(U/32/2)/ubDataNumber))
tileDataNum = max(1, tileBlockNum*32/s)

smallCoreDataNum = q*32/s
bigCoreDataNum = (q+1)*32/s
finalSmallTileNum = ceil(q/tileBlockNum)
finalBigTileNum = ceil((q+1)/tileBlockNum)
```

尾长度是 `coreDataNum - tileDataNum*floor(coreBlocks/tileBlockNum)`；若结果不大于 0，置为完整 `tileDataNum`。第 `b<r` 核的 GM 起点为 `b*bigCoreDataNum`；其余核为 `b*bigCoreDataNum-(bigCoreDataNum-smallCoreDataNum)*(b-r)`，即紧接前 `r` 个大核。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原生 | key=`ELEMENTWISE_TPL_SCH_MODE_0`；`CeluV2<float>` | 950、ACLNN 非空、FLOAT；AIV/UB 非零、`fabs(alpha)>=1e-7`、type length 非零 | 条件失败即 ACLNN/Tiling 失败；无 fallback。 |
| 2 | R-002 16-bit 升精度 | 同一 key；`CeluV2<half>` 或 `CeluV2<bfloat16_t>` | 950、ACLNN 非空、FLOAT16/BFLOAT16；其余资源 guard 同 R-001 | 条件失败即 ACLNN/Tiling 失败；无 fallback。 |

```text
aclnnInplaceCeluGetWorkspaceSize
  if 参数非法: ACLNN 错误
  if selfRef.empty: workspace=0，成功，无 Device 节点
  else: Contiguous -> CeluV2 -> Cast -> ViewCopy(selfRef)
    CeluV2TilingFunc
      if resource/dtype/alpha/typeLength guard 失败: GRAPH_FAILED
      else key=schMode0, blockDim=C'
        FLOAT -> R-001
        FLOAT16/BFLOAT16 -> R-002
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 | `schMode=0`；编译 dtype FLOAT | 950；非空 ND；所有 guard 通过 | 单个自有 K0，外围 F0→K0→F1→F2 | `celu_v2<0>` -> `CeluV2<float>::Init/Process/Compute` | depth-2 in/out Queue + 两个 FP32 TBuf；FP32 Vector 算术 | 最后 tile Copy 取 `tailDataNum`，Vector 仍按满 tile；无 fallback。 |
| R-002 16-bit 升精度 | `schMode=0`；编译 dtype FLOAT16/BFLOAT16 | 950；非空 ND；所有 guard 通过 | 同上 | `CeluV2<T>`，`T=half/bfloat16_t`；`if constexpr` 走 Cast 分支 | 输入升 FP32，所有公式在 FP32，`CAST_ROUND` 回 T；相同 Queue/Barrier 图 | 同 R-001。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台 | CMake `ascend950`、OpDef `AddConfig("ascend950")` | 选择 arch35 Tiling 与 `celu_v2_apt` | 全部 | 不发布非 950 路线 | `activation/celu_v2/CMakeLists.txt`、`ops::CeluV2` |
| AIV/UB | `coreNum!=0 && ubSize!=0` | 需要有效网格和本地 buffer | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| dtype | `{FLOAT,FLOAT16,BFLOAT16}` | 决定 `DTYPE_X`、元素宽度和精度分支 | R-001/R-002 | `GRAPH_FAILED` | `GetShapeAttrsInfo` |
| alpha | `fabs(alpha)>=1e-7f` | Host 在写 POD 前计算 `1/alpha` | 全部 | `GRAPH_FAILED`，无 fallback | `CeluV2TilingFunc` |
| 32-byte 域 | `B=ceil(N*s/32)` | 分核、tile 和 DataCopy 都以完整 block 对齐 | 全部 | 无替代路线 | `CeluV2TilingFunc` |
| UB 预算 | `ubDataNumber=4(s<=2),3(s=4)` | 对应实际 `4*N*s+8*N` 字节分配 | 全部 | 缩小 tile，最小一 block | `CeluV2TilingFunc`、`CeluV2::Init` |
| blockDim | `C'=min(AIV cores,B)` | 避免比 32-byte block 更多的核 | 全部 | 降低 blockDim | `CeluV2TilingFunc` |
| core/tile tail | 前 `r` 核大；`i==tileNum-1` 切换 Copy 长度 | 覆盖 block-rounded GM 域 | 全部 | 无 Device fallback | `CeluV2::Init/Process` |

### POD/ABI

`CeluV2TilingData` 无显式 packing；由字段顺序和类型可得 `alignof=8`、`sizeof=72`。Host 先清零整结构，非空路线再覆盖全部字段；entry 通过 `GET_TILING_DATA_WITH_STRUCT` 读取同一类型。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `CeluV2TilingData` | `smallCoreDataNum` | int64/64 | offset 0 | 元素/core | `uint32(q*32/s)` 转 int64 | `Init -> coreDataNum` | Device 窄化 uint32。 |
| 同上 | `bigCoreDataNum` | int64/64 | offset 8 | 元素/core | `uint32((q+1)*32/s)` 转 int64 | `Init` 容量和 GM offset | 大小核差一 block。 |
| 同上 | `finalBigTileNum` | int64/64 | offset 16 | tile/core | `ceil((q+1)/tileBlockNum)` | 大核 `tileNum` | Device 窄化 uint32。 |
| 同上 | `finalSmallTileNum` | int64/64 | offset 24 | tile/core | `ceil(q/tileBlockNum)` | 小核 `tileNum` | 非空时 `q>=1`。 |
| 同上 | `tileDataNum` | int64/64 | offset 32 | 元素/tile | `tileBlockNum*32/s` | buffer 容量、offset、Vector count | 分配长度；Device 窄化 uint32。 |
| 同上 | `smallTailDataNum` | int64/64 | offset 40 | 元素 | 小核最后 tile，有整除则满 tile | `Process -> processDataNum` | 只缩 Copy，不缩 Vector。 |
| 同上 | `bigTailDataNum` | int64/64 | offset 48 | 元素 | 大核最后 tile，有整除则满 tile | 同上 | 只缩 Copy，不缩 Vector。 |
| 同上 | `tailBlockNum` | int64/64 | offset 56 | core | `B mod C'` | `Init` 判断大核 | 实际含义是大核数量。 |
| 同上 | `alpha` | float/32 | offset 64 | 标量 | attrs[0]，缺失默认 1.0 | `Compute` 负支缩放 | ACLNN 显式提供 ToFloat。 |
| 同上 | `invAlpha` | float/32 | offset 68 | 标量 | `1.0f/alpha` | `Compute` 指数输入缩放 | 受近零 guard。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 | `schMode=0`、FLOAT bin specialization | `l0op::CeluV2`；`CeluV2TilingFunc` | `CeluV2TilingData` 10 字段 | `celu_v2<0>` / `CeluV2<DTYPE_X>` | `CeluV2<float>::Init/Process/CopyIn/Compute/CopyOut` | CMake ascend950/arch35；OpDef `celu_v2_apt` |
| R-002 16-bit 升精度 | `schMode=0`、FLOAT16/BFLOAT16 specialization | 同上 | 同一 POD | 同一 entry，`DTYPE_X=half/bfloat16_t` | `CeluV2<T>::Compute` 非 float constexpr 分支 | 同上 |

## Kernel 实现

### 公共执行流程

K0 的 entry 参数顺序为 `(x,y,workspace,tiling)`。它注册并解析 `CeluV2TilingData`，构造 `CeluV2<DTYPE_X>`，依次调用 `Init`、`Process`；`workspace` 未使用。`Init` 按 `GetBlockIdx()` 选 big/small core，绑定各自 GM 子域并初始化四块逻辑本地资源。`Process` 串行执行每个 tile 的 `CopyIn -> Compute -> CopyOut`；Queue 的 `EnQue/DeQue/FreeTensor` 建立搬运与 Vector 阶段生命周期。

### Route R-001：FP32 原生路线

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `activation/celu_v2/op_api/aclnn_celu.cpp` | `GetDtypeSupportList`、`CheckParams` | 950/Regbase 支持 FLOAT；非空、rank<=8、alpha 合法 | 编排公共图。 |
| L0 | `activation/celu_v2/op_api/celu_v2.cpp` | `l0op::CeluV2` | 输入 dtype/shape 分配同 dtype ND 输出 | 添加 AICore K0。 |
| Tiling | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | dtype FLOAT、资源与 alpha guard 通过 | key0、blockDim=C'。 |
| Router | `activation/celu_v2/op_kernel/celu_v2_apt.cpp` | `celu_v2<0>` | `DTYPE_X=float` | `CeluV2<float>`。 |

#### Device 数据流

##### 函数级调用链

```text
GetWorkspaceSizeCommon
  -> Contiguous(selfRef)
  -> l0op::CeluV2 -> celu_v2<0>(x,y,workspace,tiling)
    -> CeluV2<float>::Init
    -> CeluV2<float>::Process
      -> CopyIn -> Compute -> CopyOut
  -> Cast -> ViewCopy(selfRef)
```

##### 源码等价伪码

```text
# 伪码
b = GetBlockIdx()
if b < tailBlockNum:
  coreN,loops,lastN = bigCoreDataNum,finalBigTileNum,bigTailDataNum
  base = bigCoreDataNum*b
else:
  coreN,loops,lastN = smallCoreDataNum,finalSmallTileNum,smallTailDataNum
  base = bigCoreDataNum*b-(bigCoreDataNum-smallCoreDataNum)*(b-tailBlockNum)
bind xGM=x+base, yGM=y+base
for i in [0,loops):
  copyN = lastN if i==loops-1 else tileDataNum
  xLocal = DataCopy(xGM+i*tileDataNum, copyN); enqueue
  tmp2 = xLocal + 0.0                           # tileDataNum FP32 lanes
  barrier
  tmp1 = tmp2 * invAlpha; barrier
  tmp1 = exp(tmp1); barrier
  tmp1 = (tmp1-1.0)*alpha; barrier
  tmp1 = min(tmp1,0); tmp2=max(tmp2,0); barrier
  yLocal = tmp2+tmp1; enqueue
  DataCopy(yGM+i*tileDataNum,yLocal,copyN); free
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x | `base+i*tileN` | coreN；每次 copyN | FP32/ND contiguous | F0 | CopyIn | executor input until K0 done | core/tile 均 32B block。 |
| `inputQueueX` | Queue slot | depth2 × tileN | FP32 | CopyIn | Compute | TPipe；FreeTensor 后复用 | `2*tileN*4` bytes。 |
| `tmpBuffer1` | 独立 TBuf | tileN | FP32 | Muls/Exp/Adds/Muls | Mins/Add | TPipe，跨 Vector 步骤原地覆盖 | `4*tileN` bytes。 |
| `tmpBuffer2` | 独立 TBuf | tileN | FP32 | Adds(x,0) | Maxs/Add | TPipe；每 tile 覆盖 | `4*tileN` bytes。 |
| `outputQueueY` | Queue slot | depth2 × tileN | FP32 | Compute | CopyOut | TPipe；CopyOut 后释放 | `2*tileN*4` bytes。 |
| GM y | `base+i*tileN` | coreN；每次 copyN | FP32/ND contiguous | CopyOut | F1 | executor intermediate through F1 | 不与用户 selfRef 直接 alias。 |
| node workspace | executor slice | `512+libApiWorkspace` | bytes | Tiling/executor | 系统边界 | K0 launch 生命周期 | body 未读取。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["inputQueueX · depth=2 · 2×tileN×4 B"]:2
  t1["tmpBuffer1 · tileN×4 B"]:1
  t2["tmpBuffer2 · tileN×4 B"]:1
  outQ["outputQueueY · depth=2 · 2×tileN×4 B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*tileN*4 + 2*tileN*4 + tileN*4 + tileN*4 = 24*tileN bytes
tileN = tileBlockNum*32/4
tileBlockNum = max(1,floor(floor(U/32/2)/3))
=> UB_total <= U（当平台 U 足以容纳最小一 block；Tiling 只检查 U!=0）
dynamic_ub = N/A（AIV TPipe 静态分配）
workspace_total(node) = 512+libApiWorkspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input Queue slot | CopyIn | Compute `DeQue` | 后续 tile CopyIn | EnQue/DeQue/FreeTensor | 释放后才复用。 |
| tmp1 | Muls→Exp→Adds/Muls→Mins | 最终 Add | 下一 tile Muls | 每个依赖阶段之间 `PIPE_V` barrier | barrier 前不得覆盖。 |
| tmp2 | Adds(x,0)→Maxs | 最终 Add | 下一 tile Adds | `PIPE_V` barrier | Add 读完前不得覆盖。 |
| output Queue slot | Compute | CopyOut | 后续 tile Compute | EnQue/DeQue/FreeTensor | CopyOut 释放后复用。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue EnQue/DeQue | CopyIn | Compute | xLocal | Copy 仅 copyN；分配仍 tileN。 |
| 正支保留 | `Adds(...,0)` + `PIPE_V` | tmp2 | Muls/Maxs | tmp2 | FP32，无 Cast。 |
| 指数链 | Muls/Exp/Adds/Muls 间 barrier | 前一 Vector 指令 | 下一指令 | tmp1 原地结果 | 顺序严格为 x/alpha→exp→减1→乘alpha。 |
| 合并 | Mins、Maxs、barrier、Add | 两支 | Add | tmp1/tmp2 | 全 FP32。 |
| UB→GM | Queue + DataCopy | Compute | CopyOut | yLocal | 只输出 copyN；满 tile 的无效 lanes 不提交。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | FP32 UB 预算、分核、POD、key0/blockDim。 |
| Kernel entry/router | `activation/celu_v2/op_kernel/celu_v2_apt.cpp` | `celu_v2<0>` | 解析 POD 并实例化 `CeluV2<DTYPE_X>`。 |
| Device body/helper | `activation/celu_v2/op_kernel/arch35/celu_v2.h` | `CeluV2<float>::Init/Process/Compute` | FP32 Queue/TBuf、完整 Vector 顺序、tail Copy。 |

### Route R-002：FLOAT16/BFLOAT16 升精度路线

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `activation/celu_v2/op_api/aclnn_celu.cpp` | `GetDtypeSupportList` | 950/Regbase 支持 FP16/BF16 | 编排公共图。 |
| Host | `activation/celu_v2/op_host/celu_v2_def.cpp` | `ops::CeluV2` | ascend950 config 含三 dtype | 选择对应 `DTYPE_X`。 |
| Tiling | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | `inputBytes<=2 -> ubDataNumber=4` | key0、blockDim=C'。 |
| Body | `activation/celu_v2/op_kernel/arch35/celu_v2.h` | `Compute<T>` | `!is_same_v<T,float>` | Cast 到 FP32，最终 `CAST_ROUND`。 |

#### Device 数据流

##### 函数级调用链

```text
GetWorkspaceSizeCommon -> Contiguous
  -> celu_v2<0> -> CeluV2<half|bfloat16_t>::Init/Process
    -> CopyIn<T> -> Compute<T>(Cast→FP32 CELU→Cast) -> CopyOut<T>
  -> Cast -> ViewCopy(selfRef)
```

##### 源码等价伪码

```text
# 分核/地址/loop 与 R-001 相同
xLocal<T> = DataCopy(xGM+offset,copyN)
tmp2<float> = Cast<T,float>(xLocal,tileN); barrier
tmp1 = tmp2*invAlpha; barrier
tmp1 = exp(tmp1); barrier
tmp1 = (tmp1-1.0)*alpha; barrier
tmp1 = min(tmp1,0); tmp2=max(tmp2,0); barrier
tmp1 = tmp2+tmp1; barrier
yLocal<T> = Cast<float,T>(tmp1,CAST_ROUND,tileN)
DataCopy(yGM+offset,yLocal,copyN)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | 同 R-001 | coreN/copyN | T/ND contiguous | F0/CopyOut | CopyIn/F1 | executor | 32B block-rounded。 |
| input Queue | slot | depth2×tileN | T | CopyIn | Cast | TPipe | `2*tileN*2` B。 |
| tmp1 | TBuf | tileN | FP32 | CELU 负支/合并 | final Cast | TPipe | `4*tileN` B，原地链。 |
| tmp2 | TBuf | tileN | FP32 | input Cast/Maxs | Add | TPipe | `4*tileN` B。 |
| output Queue | slot | depth2×tileN | T | final Cast | CopyOut | TPipe | `2*tileN*2` B。 |
| node workspace | slice | `512+libApiWorkspace` | bytes | Tiling | 系统边界 | launch | body 未读取。 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["inputQueueX · depth=2 · 2×tileN×2 B"]:2
  t1["tmpBuffer1 · tileN×4 B"]:1
  t2["tmpBuffer2 · tileN×4 B"]:1
  outQ["outputQueueY · depth=2 · 2×tileN×2 B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*tileN*2 + 2*tileN*2 + tileN*4 + tileN*4 = 16*tileN bytes
tileN = tileBlockNum*32/2
tileBlockNum = max(1,floor(floor(U/32/2)/4))
=> UB_total <= U（同样受最小一 block 的未显式容量边界）
dynamic_ub = N/A
workspace_total(node) = 512+libApiWorkspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output Queue slots | CopyIn/Compute | Compute/CopyOut | 后续 tile | Queue lifecycle | FreeTensor 后覆盖。 |
| tmp2 FP32 | input Cast，随后 Maxs 原地 | 最终 Add | 下一 tile Cast | 各依赖点 `PIPE_V` barrier | Add 前保持正支。 |
| tmp1 FP32 | Muls/Exp/Adds/Muls/Mins/Add | final Cast | 下一 tile Muls | `PIPE_V` barrier | final Cast 前保持合并结果。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `Cast(...,CAST_NONE)` + barrier | xLocal<T> | FP32 CELU 链 | tmp2 | FP16/BF16 先转 FP32。 |
| CELU | 与 R-001 相同的 barrier 链 | tmp1/tmp2 | 下一阶段 | 两个 FP32 TBuf | alpha/invAlpha 都为 FP32。 |
| 降精度 | barrier + `Cast(...,CAST_ROUND)` | FP32 Add | output Queue | yLocal<T> | 指定 ROUND；具体 BF16/FP16硬件舍入细节属于 primitive。 |
| tail | copyN + 满 tile Vector | CopyIn/Compute | CopyOut | 尾 tile | 无效 lane 不写回 GM。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | 16-bit UB 折算、分核、POD、key。 |
| Kernel entry/router | `activation/celu_v2/op_kernel/celu_v2_apt.cpp` | `celu_v2<0>` | dtype specialization 到 `CeluV2<T>`。 |
| Device body/helper | `activation/celu_v2/op_kernel/arch35/celu_v2.h` | `CeluV2<half/bfloat16_t>::Compute` | 双 Cast、FP32 CELU、barrier 与尾 Copy。 |

## 算子特定优化

### Trick T-001：16-bit 路线以两个 FP32 TBuf 保持公式精度

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：`T` 不是 `float`。
- Golden 来源：[`activation/celu_v2/op_kernel/arch35/celu_v2.h:112-132`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_kernel/arch35/celu_v2.h#L112-L132)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (!std::is_same_v<T, float>) {
    AscendC::Cast(tmpTensor2, xLocal, RoundMode::CAST_NONE, this->tileDataNum);
    AscendC::PipeBarrier<PIPE_V>();
}
```

#### 机制与设计原因

输入只搬运一次 T 数据，进入 UB 后转为 FP32；指数、减法、两次乘法、min/max 和最终加法都使用 FP32 TBuf，最后一次性 `CAST_ROUND` 回写。这改变了精度顺序，也避免为公式各阶段反复 GM 物化。Host 以 4 个折算数据区预算 16-bit tile，恰覆盖两组 depth-2 T Queue 与两个 FP32 TBuf。

#### 迁移边界

- 适用：含指数且 16-bit 输入对中间误差敏感的逐元素 AIV 算子。
- 前提：950、FP16/BF16、UB 可容纳 `16*tileN` 字节并保留相同 barrier 顺序。
- 失效：FP32 输入使用 R-001；若不能提供两个 FP32 TBuf，必须重新设计 tile/精度，不能直接套用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但不得删去 final Cast 前的 barrier，因为源码把它作为 Add 结果到 Cast 的显式依赖。

### Trick T-002：32-byte block 大小核切分配合 dtype-aware UB 上界

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：所有非空输入。
- Golden 来源：[`activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp:124-142`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp#L124-L142)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t ubDataNumber = (inputBytes <= 2ULL) ? 4U : 3U;
uint64_t tmp = (ubSize / BLOCK_SIZE / BUFFER_NUM);
```

#### 机制与设计原因

Host 先把全域向上取整为 32-byte blocks，再令核数不超过 block 数；余数只让前 `r` 核多一个 block，因此各核负载差最多 32 字节。tile 预算按 dtype 分支：FP32 用 3、16-bit 用 4，再结合 `BUFFER_NUM=2`，与 Device 的真实 Queue/TBuf 字节账匹配。它同时避免启动空核并使每个常规 tile 的 GM 长度天然 32-byte 对齐。

#### 迁移边界

- 适用：连续逐元素、每元素 2/4 字节、相同四缓冲布局的 AIV 算子。
- 前提：分核域与 DataCopy 都以 32 字节为单位，POD/Device 使用完全相同的大小核地址公式。
- 失效：新增 TBuf、改变 Queue depth 或 dtype 宽度时必须重算 `ubDataNumber`；不能只复制常量。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 原生 | R-002 16-bit 升精度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | FLOAT | FLOAT16/BFLOAT16 | 编译 dtype 与 `if constexpr`。 |
| 模板与 dtype | `CeluV2<float>` | `CeluV2<half/bfloat16_t>` | 同 entry、不同实例。 |
| 循环与分核 | 32B block、big/small core、tile loop | 相同 | 只由 `s` 改变元素计数。 |
| Queue/TBuf | Queue FP32；两个 FP32 TBuf | Queue T；两个 FP32 TBuf | 16-bit 中间升精度。 |
| 内存总占用与复用 | `24*tileN` B | `16*tileN` B | Host 分别用 3/4 折算区。 |
| 搬运/Vector API | DataCopy、Adds/Muls/Exp/Mins/Maxs/Add | 加入首尾 Cast | 精度分支。 |
| 同步、精度与 tail | 全 FP32；tail 只缩 Copy | FP32 公式 + CAST_ROUND；同 tail | Vector count 均为满 tile。 |
| fallback/失败边界 | 无 | 无 | 资源/dtype/alpha失败直接终止。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档列支持，但稳定 `activation/celu_v2` 构建未给出 A2 Host/Kernel 接线；本文不发布 | 公开文档列支持，但稳定目录未给出 A3 Host/Kernel 接线；本文不发布 | 支持，`source_closed` |
| Host/Launcher | 固定稳定链路不可见 | 固定稳定链路不可见 | 稳定 `CeluV2`，ascend950/arch35 |
| Kernel/资源 | 固定稳定链路不可见 | 固定稳定链路不可见 | FP32 TBuf、dtype-aware UB、key0 |

## 限制与失败边界

- API 拒绝 null、rank>8、非法 dtype、不可转 FLOAT 的 alpha、0/NaN alpha；950 Tiling还拒绝 `fabs(alpha)<1e-7`，因此非零极小 alpha 会在 API 后于 Tiling 失败。
- executor 图的框架节点 workspace 切片、事件和消除规则不在固定仓中；只可依赖总 `workspaceSize` 和同 stream 图依赖。
- Tiling 按 `ceil(N*s/32)` 生成 core 域，最后一核可能覆盖逻辑数据之后的 block padding；固定仓未显示用户 Tensor 的底层 padding 保证，不能据此承诺越界区域语义。
- Host 将 core/tile 数先计算为 `uint32_t` 再存入 int64 POD，Device 又窄化为 uint32；极大 Tensor 的乘法、截断上界没有额外显式 guard。
- Tiling 只检查 `ubSize!=0`，而 `tileBlockNum` 最小强制为 1；若平台 UB 小于实际最小 buffer 总量，源码没有第二个容量错误检查。950公开平台的实际 UB 值未硬编码在本仓。
- Kernel 最后 tile 的 Copy 使用 `processDataNum`，但 Vector API 始终处理 `tileDataNum`；未搬入的 UB lanes 不会被 CopyOut 提交，但其 Vector 输入值不由本算子显式初始化。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/celu_v2/docs/aclnnCelu&aclnnInplaceCelu.md` | `aclnnInplaceCelu*` | 950公开支持、原地语义、dtype/rank/空域约束。 |
| 入口声明/约束 | `activation/celu_v2/op_api/aclnn_celu.h` | 两段式原型 | C ABI 与公开计算图。 |
| 入口实现 | `activation/celu_v2/op_api/aclnn_celu.cpp` | `CheckParams`、`GetWorkspaceSizeCommon`、`aclnnInplaceCelu*` | 检查、空域、公共图、写回 selfRef。 |
| 入口实现 | `activation/celu_v2/op_api/celu_v2.cpp` | `l0op::CeluV2` | 中间输出分配与 AICore launcher。 |
| Host/Tiling/Launcher | `activation/celu_v2/op_host/celu_v2_def.cpp` | `ops::CeluV2` | ascend950 dtype/config 与 `celu_v2_apt`。 |
| Host/Tiling/Launcher | `activation/celu_v2/op_host/arch35/celu_v2_tiling_arch35.cpp` | `CeluV2TilingFunc` | guard、workspace、分核/tile/POD/key。 |
| Host/Tiling/Launcher | `activation/celu_v2/op_kernel/arch35/celu_v2_tiling_data.h` | `CeluV2TilingData` | POD ABI。 |
| Host/Tiling/Launcher | `activation/celu_v2/op_kernel/arch35/celu_v2_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode0/1 声明；Host 实际只达 key0。 |
| Kernel entry/body | `activation/celu_v2/op_kernel/celu_v2_apt.cpp` | `celu_v2<schMode>` | entry、POD解析与实例化。 |
| Kernel entry/body | `activation/celu_v2/op_kernel/arch35/celu_v2.h` | `CeluV2<T>::Init/Process/Compute` | GM/UB、Vector、同步、精度、tail。 |
| 构建接线 | `activation/celu_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR` | ascend950/arch35 活动接线。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 72B 十字段结构与 big/small core 模型 | 字段顺序、int64/float 位宽、元素/核/tile单位 | `CeluV2TilingData` |
| 调用顺序 | Contiguous→CeluV2→Cast→ViewCopy(selfRef) | 原地语义必须在图尾写回，不能令 K0直接 alias 用户非连续 view | `GetWorkspaceSizeCommon` |
| workspace/内存 | 节点预留 `512+libApiWorkspace`；真实 UB 字节账 | Queue depth2、两个 FP32 TBuf、dtype-aware tile预算 | `GetWorkspaceSize`、`CeluV2::Init` |
| 同步 | Queue 生命周期和 `PIPE_V` barriers | 原地 TBuf 链每个依赖点及 final Cast 前 barrier | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | 16-bit 升 FP32、`CAST_ROUND`；尾部只缩 Copy | Vector满 tile与Copy有效长度的差异不可抹平 | `CeluV2::Compute/Process` |
| 失败边界 | 无 fallback；空域 ACLNN短路 | alpha近零、资源、dtype和类型长度失败必须传播 | `CheckAttributeValue`、`CeluV2TilingFunc` |
