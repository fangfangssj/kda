---
type: Operator
title: "HardshrinkBackward 950 ACLNN 算子知识"
description: "说明 aclnnHardshrinkBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/docs/aclnnHardshrinkBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_api/aclnn_hardshrink_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_api/aclnn_hardshrink_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_api/hardshrink_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_host/hard_shrink_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_host/arch35/hard_shrink_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_kernel/hard_shrink_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:46:11Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# HardshrinkBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnHardshrinkBackwardGetWorkspaceSize` + `aclnnHardshrinkBackward`，公开 ACLNN 两段式接口；固定仓未出现更高版本入口。 |
| 计算语义 | 令 `lambda=max(float(lambd),0)`；逐元素在 `abs(self)<=lambda` 时输出 0，否则传递 `gradOutput`。这与有限实数域上的 `self>lambda \|\| self<-lambda` 等价。 |
| 输入输出 | `gradOutput` 与 `self` 可 broadcast，输出 shape 必须等于 broadcast 结果；ND、0–8 维、支持非连续与空 Tensor；950 核心 dtype 为 `FLOAT16/FLOAT/BFLOAT16`。 |
| 执行类型 | 核心语义节点是单个 `AIV_VECTOR` Kernel；executor 外围可包含框架 `Contiguous/BroadcastTo/Cast/ViewCopy` 节点。无可达 Cube/AIC 或 `MIX_CV` body。 |
| Kernel 路线 | R-001/R-002 FP16 单/双缓冲；R-003/R-004 FP32 单/双缓冲；R-005/R-006 BF16→FP32 单/双缓冲。 |
| 核心资源 | 三路 GM；3 个 native dtype `TQue`；bit mask `TBuf`；BF16 另有 2 个 FP32 `TBuf`；核心 Kernel workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN、L0 launcher、950 OpDef、arch35 Tiling/POD/key、Kernel entry/body 与 CMake 接线均在固定 Commit 闭合；外围通用张量转换是明确标注的框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设广播并类型提升后的输入为 `g=gradOutput[i]`、`x=self[i]`，API 与 Tiling 都把负阈值钳为 0：

```text
lambda = lambd < 0 ? 0 : lambd
mask   = abs(x) <= lambda
gradInput[i] = mask ? 0 : g
```

有限实数输入下，这等价于公开公式在 `x>lambda` 或 `x<-lambda` 时传递梯度，边界 `x=±lambda` 输出 0。Kernel 实际使用 `abs(x)<=lambda` 的补集，因此 `x` 为 NaN 时比较为假并传递 `g`；`lambda` 为 NaN 时所有比较也为假。固定源码没有拒绝 NaN/Inf 属性。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | ND，0–8 维；可非连续；与 `self` 可 broadcast | 950：`FLOAT16/FLOAT/BFLOAT16` | 上游梯度。空 Tensor 触发 API 成功快返。公开文档要求与另外两 Tensor dtype 一致；实现还执行 `PromoteType`。 |
| `self` | ND，0–8 维；可非连续；与 `gradOutput` 可 broadcast | 同上 | 前向输入；广播后逐元素决定是否阻断梯度。 |
| `lambd` | `aclScalar` | 公开文档为 `FLOAT` | 非空；`ToFloat()` 后小于 0 则取 0。源码未单独校验 scalar dtype 或有限性。 |
| `gradInput` | ND，0–8 维；可非连续；shape 必须等于两输入 broadcast 结果 | 同上 | 输出；实现要求提升 dtype 能转换到其 dtype，最终经 `Cast` 和 `ViewCopy` 写入用户视图。 |

### 入口与版本

公开头文件只声明这一组两段式入口。第一段构造 executor、返回整条执行流程所需 workspace；第二段把该 executor 提交到调用者 stream。950 由 `IsRegbase(curArch)` 进入包含 BF16 的 dtype 支持表。

公开文档同时列 A2、A3 与 950 支持，但固定源码的 `HardShrinkGrad` OpDef 只注册 `ascend950`，CMake 只选择 `SUPPORT_COMPUTE_UNIT="ascend950"` 与 `arch35` Tiling。固定仓内没有与目标公开入口活动接线的 A2/A3 Host/Tiling/Kernel 链，因此本卡只覆盖最新且源码闭合的 950；不把 `experimental/` 下另一入口族的代码当作 `aclnnHardshrinkBackward` 的活动实现。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段接口按源码顺序执行：

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize/executor`，随后创建 `aclOpExecutor`。
2. `CheckParams` 依次检查四个业务指针非空、三 Tensor dtype 属于当前平台列表、维数不超过 8；ND 不额外检查 format。
3. `CheckPromoteType` 对两输入做类型提升，要求结果仍在支持表中且可转换到输出 dtype。
4. 若 `gradOutput` 或 `self` 为空，返回 `workspaceSize=0` 和不含核心节点的 executor；此路径发生在 broadcast shape 校验前。
5. 非空时验证两输入可 broadcast，且推导 shape 与 `gradInput` shape 相等。
6. 先对 `self`、再对 `gradOutput` 建立 `Contiguous`；原始 view shape 不同时，两者都 `BroadcastTo` 到共同 shape。
7. 两输入各自 `Cast(promoteType)`；`lambd->ToFloat()` 后负值钳零；添加 `HardShrinkGrad` AICore launcher。
8. 核心结果 `Cast(gradInput dtype)`，再 `ViewCopy` 到用户输出。任一框架节点返回空都立即失败。

### 参数、Workspace 与生命周期

`l0op::HardShrinkGrad` 从两输入 view shape 再做一次 `BroadcastInferShape`，按 `self` 的提升后 dtype 与 ND format 分配中间输出，并以 `ADD_TO_LAUNCHER_LIST_AICORE(HardShrinkGrad, OP_INPUT(gradOutput,self), OP_OUTPUT(result), OP_ATTR(lambd))` 添加核心节点。

核心 Tiling 把唯一 workspace slice 写为 0，entry 的 `workspace` 参数未读取。ACLNN 返回的总 `workspaceSize=executor->GetWorkspaceSize()` 仍可能包含 `Contiguous/BroadcastTo/Cast/ViewCopy` 与中间 Tensor 的框架需求；其分片公式在固定仓不可见。调用者必须按第一段返回值分配，并让 workspace、executor、输入输出一直有效到第二段 stream 工作完成。

### 启动图与失败传播

```text
self       -> Contiguous --[view shape differs]--> BroadcastTo --+-> Cast(promoteType) --+
                                                                  |                     |
gradOutput -> Contiguous --[view shape differs]--> BroadcastTo --+-> Cast(promoteType) --+-> HardShrinkGrad[AIV_VECTOR]
                                                                                              -> Cast(gradInput dtype)
                                                                                              -> ViewCopy(gradInput)
```

| 节点/边 | producer → consumer | 介质与范围 | ordering / owner / lifetime | 失败传播 |
|---|---|---|---|---|
| 规范化 | 用户 Tensor → `Contiguous` | executor 中间 Tensor，逻辑 shape 不变 | executor 持有；同一执行流程 | 返回空即第一段内部错误 |
| 广播（条件节点） | `Contiguous` → `BroadcastTo` | 两个共同 broadcast shape Tensor | 仅原始 view shape 不同时添加；executor 持有 | 任一路返回空即终止 |
| 类型提升 | 规范化/广播结果 → `Cast(promoteType)` | 同 shape、提升 dtype | 核心节点前添加 | 返回空则不添加后续核心节点 |
| 核心节点 | 两个提升后 Tensor → `hard_shrink_grad` | 线性 GM 输入/输出，`totalNum` 元素；核心 workspace 0 | launcher list 在第二段由同一 stream 提交 | 添加失败返回空；运行失败由 executor 返回 |
| 输出适配 | 核心输出 → `Cast` → `ViewCopy` | executor 中间 Tensor → 用户视图 | executor/stream 顺序；完成前均有效 | 任一返回空，第一段立即失败 |

核心语义只有一个 Device 节点；没有 Kernel 间 workspace 边、event、atomic 或跨核同步。

## Route 与参数设计

### 分核与资源计算

令 `N` 为提升并广播后的线性元素数，`C` 为平台 AIV 核数，`S=sizeof(T)`，`D=(N>1024?2:1)`：

```text
blockFactor = ceil(N / C)                         # 元素/core
blockDim    = ceil(N / blockFactor)
blockLength(b) = min(blockFactor, N-b*blockFactor)

computeAlignment = FP16 ? 256/2 : 256/4           # FP16 128 元素；FP32/BF16 64 元素
A = max(GetUbBlockSize(context), computeAlignment) # 源码直接使用的数值对齐因子
availableUb = ubSize - 16*1024                     # 字节
B = BF16 ? (3*D*2 + 2*4) : (3*D*S)               # selector 的每元素字节预算
rawU = floor(availableUb / B)
U = floor_align(rawU, A)
if N < U: U = ceil(N/A)*A
```

POD 写入 `ubFactor=U`。每核循环 `ceil(blockLength/U)` 次，GM base 为 `b*blockFactor+i*U`，最后一次 `currentNum=blockLength-i*U`。源码没有 former/tail core 独立字段；最后 block 由 remainder 公式收口。标量 storage shape 被 `EnsureNotScalar` 视为 `{1}`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 0 | 空 Tensor 快返 | 无核心 key | API 校验与类型提升后，`gradOutput->IsEmpty() \|\| self->IsEmpty()` | `workspaceSize=0` 成功返回，不运行 Tiling/Kernel |
| 1 | R-001 FP16 单缓冲 | `(C_DT_FLOAT16,0)` | 950、非空、核心 dtype `FLOAT16` 且 `N<=1024` | 同 dtype 且 `N>1024` → R-002；其它 dtype 继续 |
| 2 | R-002 FP16 双缓冲 | `(C_DT_FLOAT16,1)` | 950、非空、核心 dtype `FLOAT16` 且 `N>1024` | 不满足 dtype → 后续 Route |
| 3 | R-003 FP32 单缓冲 | `(C_DT_FLOAT,0)` | 950、非空、核心 dtype `FLOAT` 且 `N<=1024` | 同 dtype 且 `N>1024` → R-004；其它 dtype 继续 |
| 4 | R-004 FP32 双缓冲 | `(C_DT_FLOAT,1)` | 950、非空、核心 dtype `FLOAT` 且 `N>1024` | 不满足 dtype → 后续 Route |
| 5 | R-005 BF16 升精度单缓冲 | `(C_DT_BF16,0)` | 950、非空、核心 dtype `BFLOAT16` 且 `N<=1024` | 同 dtype 且 `N>1024` → R-006；其它 dtype 失败 |
| 6 | R-006 BF16 升精度双缓冲 | `(C_DT_BF16,1)` | 950、非空、核心 dtype `BFLOAT16` 且 `N>1024` | 其它 dtype 被 ACLNN/Tiling 拒绝 |

```text
ACLNN invalid/null/dim/broadcast/promote -> error
      empty input                       -> success, no core launch
      nonempty                          -> framework normalize/broadcast/cast
Tiling dtype ∈ {FP16, FP32, BF16}
       BUFFER_MODE = (N > 1024 ? 1 : 0)
       -> ASCENDC_TPL_SEL_PARAM(dtype, BUFFER_MODE)
Kernel hard_shrink_grad<D_T_X, BUFFER_MODE>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | FP16,0 | 950；非空；FP16；`N<=1024`；平台 core/UB 有效 | 单 `hard_shrink_grad<half,0>` AIV_VECTOR | `HardShrinkGrad<half,0>`；native FP16 `Abs/Compares/Select` | 3×depth-1 FP16 Queue + mask；Queue 同步 | core/tile remainder；无 Device fallback |
| R-002 | FP16,1 | 同上但 `N>1024` | 单 `hard_shrink_grad<half,1>` | `HardShrinkGrad<half,1>` | 3×depth-2 FP16 Queue + mask | 同上 |
| R-003 | FP32,0 | 950；非空；FP32；`N<=1024`；资源有效 | 单 `hard_shrink_grad<float,0>` | `HardShrinkGrad<float,0>`；native FP32 | 3×depth-1 FP32 Queue + mask | 同上 |
| R-004 | FP32,1 | 同上但 `N>1024` | 单 `hard_shrink_grad<float,1>` | `HardShrinkGrad<float,1>` | 3×depth-2 FP32 Queue + mask | 同上 |
| R-005 | BF16,0 | 950；非空；BF16；`N<=1024`；资源有效 | 单 `hard_shrink_grad<bfloat16_t,0>` | BF16→FP32，FP32 比较/选择，`CAST_RINT` 回 BF16 | 3×depth-1 BF16 Queue + 2×FP32 TBuf + mask | 同上 |
| R-006 | BF16,1 | 同上但 `N>1024` | 单 `hard_shrink_grad<bfloat16_t,1>` | 同 R-005，depth=2 | 3×depth-2 BF16 Queue + 2×FP32 TBuf + mask | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台资源 | `GetCoreNumAiv()!=0 && ubSize!=0` | 分核与 UB tile 必须有有效平台资源 | 全部 | Tiling `GRAPH_FAILED` | `GetPlatformInfo` |
| 双缓冲阈值 | `N>1024` | 只改变三个 Queue 的 depth 和 UB 每元素预算 | 全部 | `N<=1024` 选单缓冲 | `MIN_SPLIT_THRESHOLD`、`HardShrinkGradTilingFunc` |
| dtype 对齐 | FP16 `128` 元素；FP32/BF16 `64` 元素；再与 `GetUbBlockSize` 取大者 | Vector 按 256B 计算粒度分配完整 tile | 全部 | 无替代 Route；公式直接决定 `U` | `HardShrinkGradTilingFunc` |
| 16KiB 预留 | `availableUb=ubSize-16384` | 注释分别归因于 TPipe/Queue 管理和 `Select` 模式 1 | 全部 | 无显式 `availableUb>0` 检查 | 同上 |
| mask 容量 | `M=max(32,align_up(U/8,32))` | `Compares` 每元素 1 bit，至少 32B | 全部 | 无独立 fallback | `HardShrinkGrad::Init` |
| DMA `uint16_t` 窄化 | `blockLen=static_cast<uint16_t>(currentNum*sizeof(T))` | `DataCopyPad` 单 block 字节长度字段为 16 位 | 全部 | 固定源码没有显式 `<=65535` guard | `CopyIn`、`CopyOut` |
| 空域 | API 两输入任一 `IsEmpty()` | 避免 `blockFactor=0` 后继续分核 | 全部之前 | 成功快返 | `aclnnHardshrinkBackwardGetWorkspaceSize` |
| 完整 tile 计算 | `computeNum=U`，不使用形参 `currentNum` | Vector 每次处理对齐容量；只在 CopyOut 截取有效元素 | 全部 | 无 tail 专用 Route | `Compute`、`Process` |

### POD/ABI

`HardShrinkGradTilingData` 在 Host 与 Device 共用同一头文件。源码给出字段顺序，但没有 `sizeof/alignof/offsetof` 静态断言，因此具体 ABI offset 不推测。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `HardShrinkGradTilingData` | `totalNum` | `int64_t`/64 | 第 1 字段；精确 offset/整体 sizeof 固定仓不可见 | 元素 | `tiling->totalNum=totalIdx` | `Init` 计算 core remainder | ACLNN 非空路径为正；内部直接调用空域没有本地安全闭包 |
| 同上 | `blockFactor` | `int64_t`/64 | 第 2 字段；精确 offset 固定仓不可见 | 元素/core | `CeilDiv(totalIdx,coreNum)` | `Init` 的 core 长度与 GM base | 非空为正 |
| 同上 | `ubFactor` | `int64_t`/64 | 第 3 字段；精确 offset 固定仓不可见 | 元素/tile | UB 预算、下对齐及小输入上对齐 | `InitBuffer` 容量、tile loop、GM tile offset、Vector `computeNum` | 固定源码无显式 `>0` 校验 |
| 同上 | `lambd` | `float`/32 | 第 4 字段；精确 offset 固定仓不可见 | 输入数值单位 | API 先钳负值；Tiling 再执行 `(lambd<0)?0:lambd` | `Compares(...,lambd_,CMPMODE::LE,U)` | NaN/Inf 未拒绝；注释写“取绝对值”，实际表达式仅钳负值 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | FP16,0 | `hard_shrink_grad_tiling.cpp::HardShrinkGradTilingFunc` | `HardShrinkGradTilingData` | `hard_shrink_grad_apt.cpp::hard_shrink_grad<half,0>` | `hard_shrink_grad.h::HardShrinkGrad<half,0>` | OpDef + CMake `ascend950/arch35` |
| R-002 | FP16,1 | 同上 | 同上 | `hard_shrink_grad<half,1>` | `HardShrinkGrad<half,1>` | 同上 |
| R-003 | FP32,0 | 同上 | 同上 | `hard_shrink_grad<float,0>` | `HardShrinkGrad<float,0>` | 同上 |
| R-004 | FP32,1 | 同上 | 同上 | `hard_shrink_grad<float,1>` | `HardShrinkGrad<float,1>` | 同上 |
| R-005 | BF16,0 | 同上 | 同上 | `hard_shrink_grad<bfloat16_t,0>` | `HardShrinkGrad<bfloat16_t,0>` | 同上 |
| R-006 | BF16,1 | 同上 | 同上 | `hard_shrink_grad<bfloat16_t,1>` | `HardShrinkGrad<bfloat16_t,1>` | 同上 |

## Kernel 实现

### 公共执行流程

六条 Route 都是单 Kernel。编译期 entry `hard_shrink_grad<D_T_X,BUFFER_MODE>` 解析同一 POD，实例化对应 `HardShrinkGrad` 并调用 `Init→Process`。`Init` 以 `blockFactor*blockIdx` 绑定三路连续 GM；`Process` 对每个 tile 串行执行 `CopyIn→Compute→CopyOut`。

两路 GM→UB `DataCopyPad` 均使用 `blockCount=1`、`blockLen=currentNum*sizeof(T)` 字节、`srcStride=dstStride=0`，padding 配置为 `{false,0,0,0}`。UB→GM 也为单 block、零 stride，只写 `currentNum` 有效元素。Queue 的 `Alloc/EnQue/DeQue/FreeTensor` 建立 DMA 与 Vector 阶段的生产消费关系；单 Kernel 内无 event、atomic、跨核通信、dynamic UB 或 NDDMA。

### Route R-001：FP16 单缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardshrink_backward.cpp` | `GetDtypeSupportList` / `CheckPromoteType` | 950 支持 FP16，提升结果为 FP16 | 构造 FP16 核心输入 |
| Tiling | `op_host/arch35/hard_shrink_grad_tiling.cpp` | `HardShrinkGradTilingFunc` | `dataType==DT_FLOAT16 && totalIdx<=1024` | `(D_T_X=C_DT_FLOAT16,BUFFER_MODE=0)` |
| Entry | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<D_T_X,BUFFER_MODE>` | 编译期模板实参 `half,0` | `HardShrinkGrad<half,0>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(HardShrinkGrad)
  -> hard_shrink_grad<half,0>
    -> HardShrinkGrad<half,0>::Init
    -> Process -> CopyIn -> Compute(native FP16) -> CopyOut
```

##### 源码等价伪码

```text
# 源码等价伪码
b=GetBlockIdx(); L=min(blockFactor,totalNum-b*blockFactor); bind GM at b*blockFactor
for i in [0,ceil(L/U)):
  n=(last ? L-i*U : U); copy g,x 的 n 个 FP16 元素到 depth-1 Queue
  abs(x[0:U]) in-place; mask=abs(x[0:U])<=FP16(lambd)
  fill x[0:U]=0; out[0:U]=select(mask,x_zero,g)
  copy out[0:n] to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `grad/self/out` | `b*blockFactor+i*U` 元素 | core `L`，tile 有效 `n` | FP16/线性 ND | executor/CopyOut | CopyIn/用户 | executor；stream 完成前 | core 区间不重叠 |
| `inputQueueGrad_` | Queue slot | `U*2` B | FP16 | CopyIn | Compute | `TPipe`，Compute 后释放 | depth=1 |
| `inputQueueSelf_` | Queue slot | `U*2` B | FP16 | CopyIn | Compute | 同上 | depth=1；先 Abs 后覆零 |
| `outputQueue_` | Queue slot | `U*2` B | FP16 | Compute | CopyOut | CopyOut 后释放 | depth=1 |
| `maskBuf_` | 独立 TBuf | `M=max(32,align_up(U/8,32))` B | uint8 bit mask | `Compares` | `Select` | core 全生命周期 | 32B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gQ["grad VECIN · depth=1 · U*2 B"]:1
  xQ["self VECIN · depth=1 · U*2 B"]:1
  outQ["out VECOUT · depth=1 · U*2 B"]:1
  mask["mask VECCALC · M B"]:1
```

##### 字节账与复用顺序

```text
UB_explicit = 3*U*2 + M
selector_budget = 3*U*2 <= ubSize-16384
dynamic_ub = N/A
workspace_total(core node) = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inputQueueSelf_` slot | CopyIn → `Abs` | `Compares` | `Duplicate(0)`，再由下一 tile CopyIn | Queue DeQue；Compute 末尾 Free | Select 完成前不得释放 |
| `maskBuf_` | `Compares` | 紧随其后的 `Select` | 下一 tile `Compares` | 同一串行 Compute | Select 已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VEC | Queue `EnQue/DeQue` | CopyIn | Compute | 两个输入 slot | 搬运 `n`，不填充其余 lane |
| 计算 | `Abs/Compares/Select` | Compute | output `EnQue` | `U` 个 FP16 lane | lambda 转 FP16；完整 U 计算，尾部无效 lane 不写回 |
| VEC→GM | output `EnQue/DeQue/Free` | Compute | CopyOut | 输出 slot | 只写 `n` 个有效元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_shrink_grad_tiling.cpp` | `HardShrinkGradTilingFunc` | FP16 对齐、单缓冲 key、block/U/POD |
| Kernel entry/router | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<half,0>` | POD 解析并实例化 body |
| Device body/helper | `op_kernel/arch35/hard_shrink_grad.h` | `HardShrinkGrad<half,0>::Init/Process/Compute` | native FP16 比较选择、Queue 与 tail |

### Route R-002：FP16 双缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardshrink_backward.cpp` | `CheckPromoteType` | 提升结果 FP16 | FP16 核心输入 |
| Tiling | `op_host/arch35/hard_shrink_grad_tiling.cpp` | `HardShrinkGradTilingFunc` | `dataType==DT_FLOAT16 && totalIdx>1024` | `(C_DT_FLOAT16,1)`；预算 `B=12` B/元素 |
| Entry | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<half,1>` | 编译期 `BUFFER_NUM=2` | 双缓冲 body |

#### Device 数据流

##### 函数级调用链

```text
HardShrinkGrad launcher -> hard_shrink_grad<half,1>
  -> HardShrinkGrad<half,1>::Init
  -> Process -> CopyIn -> Compute(native FP16) -> CopyOut
```

##### 源码等价伪码

```text
# 与 R-001 相同的 core/tile/地址/FP16 计算；差异是三条 Queue 各有 2 个 U 元素 slot。
for tile: copy n -> abs/compare/select over U -> copy n; Queue 协议决定 slot 可复用时点。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三路 GM | `b*blockFactor+i*U` | `n` 有效 | FP16/ND | executor/CopyOut | CopyIn/用户 | stream 完成前 | tail 精确 |
| 三个 Queue | 各自 slot | 每 Queue `2*U*2` B | FP16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth=2 |
| `maskBuf_` | 独立 TBuf | `M` B | bit mask | `Compares` | `Select` | core 全生命周期 | 32B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gQ["grad · depth=2 · 2U*2 B"]:1
  xQ["self · depth=2 · 2U*2 B"]:1
  outQ["out · depth=2 · 2U*2 B"]:1
  mask["mask · M B"]:1
```

##### 字节账与复用顺序

```text
UB_explicit = 3*2*U*2 + M
selector_budget = 12U <= ubSize-16384
dynamic_ub=N/A; workspace_total(core node)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slot 0/1 | CopyIn/Compute | Compute/CopyOut | 对应 slot 被 Free 后的后续 tile | `EnQue/DeQue/FreeTensor` | 不覆盖未消费 slot |
| `self` slot / mask | `Abs→Duplicate` / `Compares` | `Compares→Select` | slot 释放 / 下一 tile | 串行调用和 Queue 生命周期 | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue pipeline | depth=2 Queue | CopyIn/Compute | DeQue/Free | 两套 slot | 实际循环仍源码顺序调用；Queue 提供复用协议 |
| FP16 计算 | native Vector | self/grad Queue | output Queue | U lanes | lambda FP16；只提交 n lanes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_shrink_grad_tiling.cpp` | `totalIdx>MIN_SPLIT_THRESHOLD` | 选择 depth=2 并按 12 B/元素定 U |
| Kernel entry/router | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<half,1>` | 模板实例 |
| Device body/helper | `op_kernel/arch35/hard_shrink_grad.h` | `BUFFER_NUM=2`、`Process` | 双 Queue 容量与共同 body |

### Route R-003：FP32 单缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardshrink_backward.cpp` | `CheckPromoteType` | 提升结果 FP32 | FP32 核心输入 |
| Tiling | `op_host/arch35/hard_shrink_grad_tiling.cpp` | `HardShrinkGradTilingFunc` | `DT_FLOAT && N<=1024` | `(C_DT_FLOAT,0)`；`A>=64`，`B=12` B/元素 |
| Entry | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<float,0>` | 编译期实例 | native FP32 body |

#### Device 数据流

##### 函数级调用链

```text
HardShrinkGrad launcher -> hard_shrink_grad<float,0>
  -> Init -> Process -> CopyIn -> Compute(Abs/Compares/Select FP32) -> CopyOut
```

##### 源码等价伪码

```text
b/L/U/n 与公共公式一致；每 tile 搬 n 个 FP32。
abs(x[0:U]) in-place; mask=abs(x)<=lambd(float); x[0:U]=0;
out[0:U]=select(mask,x_zero,g); 仅 out[0:n] 写 GM。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三路 GM | `b*blockFactor+i*U` | `n` 有效 | FP32/ND | executor/CopyOut | CopyIn/用户 | stream 完成前 | 不重叠 |
| 三个 Queue | 各自 slot | 每 Queue `U*4` B | FP32 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth=1 |
| mask | 独立 | `M` B | bit mask | `Compares` | `Select` | core 生命周期 | 32B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gQ["grad · depth=1 · U*4 B"]:1
  xQ["self · depth=1 · U*4 B"]:1
  outQ["out · depth=1 · U*4 B"]:1
  mask["mask · M B"]:1
```

##### 字节账与复用顺序

```text
UB_explicit = 3*U*4 + M
selector_budget = 12U <= ubSize-16384
dynamic_ub=N/A; workspace_total(core node)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `self` Queue slot | CopyIn/Abs | Compares | Duplicate 0，然后下一 tile | DeQue/Free | Select 前保留零 Tensor |
| mask | Compares | Select | 下一 tile | 串行 Compute | 当前 Select 已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM/Vector | Queue | CopyIn/Compute | Compute/CopyOut | FP32 slots | n 搬运，U 计算，n 写回 |
| 计算 | FP32 `Abs/Compares/Select` | self/grad | out Queue | U lanes | 无 Cast；边界用 `LE` 归零 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_shrink_grad_tiling.cpp` | FP32 branch | 4B dtype、64 元素计算对齐、单缓冲 |
| Kernel entry/router | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<float,0>` | entry 实例 |
| Device body/helper | `op_kernel/arch35/hard_shrink_grad.h` | `HardShrinkGrad<float,0>::Compute` | FP32 原生路径 |

### Route R-004：FP32 双缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardshrink_backward.cpp` | `CheckPromoteType` | 提升结果 FP32 | FP32 核心输入 |
| Tiling | `op_host/arch35/hard_shrink_grad_tiling.cpp` | `HardShrinkGradTilingFunc` | `DT_FLOAT && N>1024` | `(C_DT_FLOAT,1)`；`B=24` B/元素 |
| Entry | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<float,1>` | `BUFFER_NUM=2` | FP32 双缓冲 body |

#### Device 数据流

##### 函数级调用链

```text
HardShrinkGrad launcher -> hard_shrink_grad<float,1>
  -> Init -> Process -> CopyIn -> Compute(native FP32) -> CopyOut
```

##### 源码等价伪码

```text
# 计算、地址和 tail 与 R-003 相同；三个 Queue 的 slot 数由 1 改为 2。
for tile: copy n -> abs/compare/select over U -> copy n。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三路 GM | 公共公式 | `n` | FP32/ND | executor/CopyOut | CopyIn/用户 | stream 完成前 | tail 精确 |
| 三个 Queue | 各两个 slot | 每 Queue `2*U*4` B | FP32 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth=2 |
| mask | 独立 | `M` B | bit mask | Compares | Select | core 生命周期 | 32B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gQ["grad · depth=2 · 2U*4 B"]:1
  xQ["self · depth=2 · 2U*4 B"]:1
  outQ["out · depth=2 · 2U*4 B"]:1
  mask["mask · M B"]:1
```

##### 字节账与复用顺序

```text
UB_explicit = 3*2*U*4 + M
selector_budget = 24U <= ubSize-16384
dynamic_ub=N/A; workspace_total(core node)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue 双 slot | CopyIn/Compute | Compute/CopyOut | Free 后的后续 tile | Queue 协议 | 未消费 slot 不覆盖 |
| self/mask | Abs/Compares | Compares/Select | Duplicate/下一 tile | 串行调用 | 同 R-003 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 双缓冲 Queue | depth=2 | 各阶段 producer | DeQue/Free | 两组 FP32 slot | U 计算，n 写回 |
| 数值 | FP32 native | self/grad | out | 比较与选择 | 无降精度；NaN 比较假 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_shrink_grad_tiling.cpp` | `useDoubleBuffer`、FP32 branch | depth=2、24 B/元素 |
| Kernel entry/router | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<float,1>` | entry 实例 |
| Device body/helper | `op_kernel/arch35/hard_shrink_grad.h` | `HardShrinkGrad<float,1>` | FP32 双缓冲实现 |

### Route R-005：BF16 升精度单缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardshrink_backward.cpp` | `GetDtypeSupportList` | 950 `IsRegbase` 支持 BF16 | BF16 核心输入 |
| Tiling | `op_host/arch35/hard_shrink_grad_tiling.cpp` | BF16 branch | `DT_BF16 && N<=1024` | `(C_DT_BF16,0)`；`A>=64`，`B=14` B/元素 |
| Entry | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<bfloat16_t,0>` | `CAST_TO_FP32=true` | BF16 升精度 body |

#### Device 数据流

##### 函数级调用链

```text
HardShrinkGrad launcher -> hard_shrink_grad<bfloat16_t,0>
  -> Init -> Process -> CopyIn
  -> Compute(Cast BF16→FP32 -> Abs -> Compares -> Duplicate -> Select -> Cast FP32→BF16)
  -> CopyOut
```

##### 源码等价伪码

```text
b/L/U/n 与公共公式一致；copy n 个 BF16。
selfF[0:U]=cast_none(self); gradF[0:U]=cast_none(grad)
abs(selfF) in-place; mask=abs(selfF)<=lambd(float); selfF=0
gradF=select(mask,selfF_zero,gradF) in-place
outBF16=cast_rint(gradF); 仅 out[0:n] 写 GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三路 GM | 公共公式 | `n` | BF16/ND | executor/CopyOut | CopyIn/用户 | stream 完成前 | tail 精确 |
| 三个 Queue | 各一个 slot | 每 Queue `U*2` B | BF16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth=1 |
| `selfFloatBuf_` | 独立 TBuf | `U*4` B | FP32 | Cast/Abs/Duplicate | Compares/Select | core 生命周期 | 自身原地覆写 |
| `gradFloatBuf_` | 独立 TBuf | `U*4` B | FP32 | Cast | Select/Cast out | core 生命周期 | Select 原地写回 |
| mask | 独立 | `M` B | bit mask | Compares | Select | core 生命周期 | 32B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  gQ["grad · d1 · U*2 B"]:1
  xQ["self · d1 · U*2 B"]:1
  outQ["out · d1 · U*2 B"]:1
  fbuf["2 FP32 TBuf · 2U*4 B"]:1
  mask["mask · M B"]:1
```

##### 字节账与复用顺序

```text
UB_explicit = 3*U*2 + 2*U*4 + M = 14U+M
selector_budget = 14U <= ubSize-16384
dynamic_ub=N/A; workspace_total(core node)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `selfFloatBuf_` | Cast→Abs | Compares | Duplicate 0，为 Select 的 zero Tensor | 串行 Compute | Compares 已完成 |
| `gradFloatBuf_` | Cast | Select 输入 | Select 原地输出，再由 Cast 读取 | 同一指令序列 | 降精度 Cast 前不可覆盖 |
| mask | Compares | Select | 下一 tile | 串行 Compute | Select 已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入升精度 | `Cast(...CAST_NONE,U)` | BF16 Queue | FP32 运算 | 两个 FP32 TBuf | 完整 U lanes |
| 比较选择 | FP32 `Abs/Compares/Select` | FP32 buffers | 降精度 Cast | mask/gradF | lambda 保持 FP32 |
| 输出降精度 | `Cast(...CAST_RINT,U)` | gradF | output Queue | BF16 输出 slot | 完整 U 计算，只写 n |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_shrink_grad_tiling.cpp` | BF16 buffer branch | 64 元素对齐、14 B/元素 |
| Kernel entry/router | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<bfloat16_t,0>` | entry 实例 |
| Device body/helper | `op_kernel/arch35/hard_shrink_grad.h` | `CAST_TO_FP32` branch | 两次升精度、FP32 比较选择、`CAST_RINT` |

### Route R-006：BF16 升精度双缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardshrink_backward.cpp` | `GetDtypeSupportList` | 950 BF16 | BF16 核心输入 |
| Tiling | `op_host/arch35/hard_shrink_grad_tiling.cpp` | BF16 branch | `DT_BF16 && N>1024` | `(C_DT_BF16,1)`；`B=20` B/元素 |
| Entry | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<bfloat16_t,1>` | `BUFFER_NUM=2`、FP32 TBuf 仍单份 | BF16 双缓冲 body |

#### Device 数据流

##### 函数级调用链

```text
HardShrinkGrad launcher -> hard_shrink_grad<bfloat16_t,1>
  -> Init -> Process -> CopyIn -> Compute(BF16→FP32→BF16) -> CopyOut
```

##### 源码等价伪码

```text
# 地址、计算、tail 与 R-005 相同；三个 BF16 Queue 为 depth=2，两个 FP32 TBuf 和 mask 仍各一份。
for tile: copy n -> cast/abs/compare/select/cast over U -> copy n。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三路 GM | 公共公式 | `n` | BF16/ND | executor/CopyOut | CopyIn/用户 | stream 完成前 | tail 精确 |
| 三个 Queue | 各两个 slot | 每 Queue `2*U*2` B | BF16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth=2 |
| 两个 FP32 TBuf | 独立 | 合计 `2*U*4` B | FP32 | Cast/Vector | Vector/Cast | core 生命周期 | 不双缓冲；按 tile 串行复用 |
| mask | 独立 | `M` B | bit mask | Compares | Select | core 生命周期 | 32B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  gQ["grad · d2 · 2U*2 B"]:1
  xQ["self · d2 · 2U*2 B"]:1
  outQ["out · d2 · 2U*2 B"]:1
  fbuf["2 FP32 TBuf · 2U*4 B"]:1
  mask["mask · M B"]:1
```

##### 字节账与复用顺序

```text
UB_explicit = 3*2*U*2 + 2*U*4 + M = 20U+M
selector_budget = 20U <= ubSize-16384
dynamic_ub=N/A; workspace_total(core node)=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| BF16 Queue 双 slot | CopyIn/Compute | Compute/CopyOut | Free 后后续 tile | Queue 协议 | 未消费 slot 不覆盖 |
| FP32 TBuf/mask | 当前 tile Cast/Compares | 当前 tile Vector/Cast | 下一 tile | Process 串行调用 | 当前 tile CopyOut 前计算已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16 Queue | depth=2 | CopyIn/Compute | DeQue/Free | BF16 slots | n 搬运 |
| FP32 compute | `CAST_NONE` + Vector | BF16 inputs | `CAST_RINT` | 单份 FP32 TBuf/mask | U 计算；NaN 比较假 |
| CopyOut | Queue | BF16 Cast | GM | output slots | 只写 n |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_shrink_grad_tiling.cpp` | BF16 + `useDoubleBuffer` | 20 B/元素预算 |
| Kernel entry/router | `op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad<bfloat16_t,1>` | entry 实例 |
| Device body/helper | `op_kernel/arch35/hard_shrink_grad.h` | `HardShrinkGrad<bfloat16_t,1>` | BF16 Queue 双缓冲、FP32 TBuf 单份 |

## 算子特定优化

### Trick T-001：复用 self 中间区构造 Select 的零输入

#### 触发与源码依据

- 适用 Route：R-001～R-006。
- 触发条件：完成 `abs(self)` 与 `Compares` 后，原 self 数值不再需要。
- Golden 来源：[`op_kernel/arch35/hard_shrink_grad.h:174-184`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad.h#L174-L184)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
// Step 2: Abs in-place on selfFloat
Abs(selfFloatLocal, selfFloatLocal, computeNum);

// Step 3: Compares |x| <= lambd -> bitmask (mask=1 means dead zone)
Compares(maskLocal, selfFloatLocal, lambd_, CMPMODE::LE, computeNum);

// Step 4: Select - mask=1 (dead zone) picks 0, mask=0 picks gradFloat
// Fill selfFloatLocal with 0 (reuse as zero tensor, no longer needed)
Duplicate(selfFloatLocal, 0.0f, computeNum);
Select(gradFloatLocal, maskLocal, selfFloatLocal, gradFloatLocal,
       SELMODE::VSEL_TENSOR_TENSOR_MODE, computeNum);
```

#### 机制与设计原因

BF16 Route 把 `selfFloatBuf_` 依次作为 `abs(self)`、比较源和全零 Select 输入；`gradFloatBuf_` 同时作为 Select 的输入与输出。native FP16/FP32 Route 对 `selfLocal` 做同样的 `Abs→Compares→Duplicate(0)` 复用，再直接把结果写到 output Queue。比较完成后旧 self 已无消费者，因此无需另分配 zero Tensor；BF16 还避免额外结果 TBuf。

#### 迁移边界

- 适用：逐元素谓词计算后不再需要条件输入、且 Select 支持该原地关系的 Vector 实现。
- 前提：必须保持 `Compares` 在 `Duplicate` 之前，且 mask 在 `Select` 前有效。
- 失效：后续步骤仍需原始/绝对值 self，或目标平台不支持 Select 的该别名模式时，需独立缓冲。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从现有依赖可确定，若提前 `Duplicate` 会破坏比较输入。

### Trick T-002：按全局元素数切换 Queue depth，并重算 UB tile

#### 触发与源码依据

- 适用 Route：`N<=1024` 进入 R-001/R-003/R-005；`N>1024` 进入 R-002/R-004/R-006。
- 触发条件：`useDoubleBuffer = totalIdx > MIN_SPLIT_THRESHOLD`。
- Golden 来源：[`op_host/arch35/hard_shrink_grad_tiling.cpp:195-211`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_host/arch35/hard_shrink_grad_tiling.cpp#L195-L211)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t bufferBytes;
int64_t numTQueBuffers = (useDoubleBuffer != 0) ? 2 : 1;
if (dataType == ge::DT_BF16) {
    // bf16: 3 queues * BUFFER_NUM * 2B + 2 float TBuf * 4B
    bufferBytes = 3 * numTQueBuffers * typeSize + 2 * static_cast<int64_t>(sizeof(float));
} else {
    // fp16/fp32: 3 queues * BUFFER_NUM * typeSize
    bufferBytes = 3 * numTQueBuffers * typeSize;
}
int64_t rawUbFactor = FloorDiv(availableUb, bufferBytes);
int64_t alignFactor = (ubBlockSize > computeAlignment) ? ubBlockSize : computeAlignment;
int64_t ubFactorAligned = FloorAlign(rawUbFactor, alignFactor);
// Cap ubFactor by totalIdx to avoid allocating more than necessary.
// For small data, ubFactor = ceil(totalIdx / alignFactor) * alignFactor.
if (totalIdx < ubFactorAligned) {
    int64_t needed = CeilDiv(totalIdx, alignFactor) * alignFactor;
    ubFactorAligned = needed;
}
```

#### 机制与设计原因

Host 不只切换模板位；它把三条 Queue 的 depth 精确乘入每元素预算，再据此重算 `ubFactor`。小域使用单缓冲减少 Queue 占用，大域选择双缓冲协议；BF16 的两块 FP32 TBuf 不随 depth 翻倍，因 `Process` 在 Compute 内串行复用它们。

#### 迁移边界

- 适用：CopyIn/Compute/CopyOut Queue 流水且可依据任务规模选择 depth 的逐元素 Kernel。
- 前提：Host 预算与 Device `BUFFER_NUM` 必须来自同一 selector 位；所有 Queue 都使用相同 depth。
- 失效：阈值或某个 Queue depth 改变时必须同步修改预算；否则 `U` 与真实 UB 分配不一致。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码结构明确排除了“只改模板 depth 而不重算 `bufferBytes`”的做法。

### Trick T-003：统一对齐 U 计算，尾部只在 DMA 提交处收口

#### 触发与源码依据

- 适用 Route：R-001～R-006 的每个 tile，尤其最后一 tile `currentNum<U`。
- 触发条件：Tiling 将 `ubFactor` 对齐；`Compute` 固定 `computeNum=ubLength_`，`CopyOut` 使用真实 `currentNum`。
- Golden 来源：[`op_kernel/arch35/hard_shrink_grad.h:215-220`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad.h#L215-L220)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
for (int64_t i = 0; i < loopCount; i++) {
    int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
    CopyIn(i, currentNum);
    Compute(currentNum);
    CopyOut(i, currentNum);
}
```

#### 机制与设计原因

`Compute` 不分尾块 mask，而统一对 U 个 lane 执行 Vector 序列；GM 输入输出搬运长度仍由 `currentNum*sizeof(T)` 精确限制。这样避免 tail 专用 Route 与 Vector 参数分支。无效 UB lane 可能保留旧值并参与局部计算，但 `CopyOut` 不把它们提交到 GM。

#### 迁移边界

- 适用：逐 lane 独立、无跨 lane reduction/scan/atomic，且无效 lane 结果不会影响有效 lane。
- 前提：本地缓冲按完整 U 分配；输出 DMA 必须严格使用 `currentNum`。
- 失效：存在跨 lane 数据依赖、异常值会影响全局状态、或输出整块 DMA 时，必须显式填充或 mask tail。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；对本实现而言，把输出 `blockLen` 改成 `U*sizeof(T)` 会越过有效 GM 范围。

## 模板与实现对比

| 对比维度 | R-001/R-002 FP16 | R-003/R-004 FP32 | R-005/R-006 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | FP16；`N<=1024` d1，否则 d2 | FP32；同阈值 | BF16；同阈值 | dtype selector × `N>1024` |
| 模板与 dtype | `half,0/1` | `float,0/1` | `bfloat16_t,0/1` | key header 只枚举这六种 |
| 循环与分核 | 共同 `ceil(N/C)` core 切分与 U tile | 同左 | 同左 | 仅 U 因预算不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3×FP16 Queue + mask | 3×FP32 Queue + mask | 3×BF16 Queue + 2×FP32 TBuf + mask | 无显式 RegTensor/dynamic UB/通信 |
| 内存总占用与复用 | `(6或12)U+M` | `(12或24)U+M` | `(14或20)U+M` | depth 和升精度缓冲决定 |
| 搬运/Vector/SIMT/NDDMA/通信 API | FP16 DataCopyPad + native Vector | FP32 DataCopyPad + native Vector | BF16 DataCopyPad + Cast/FP32 Vector/Cast | 仅 AIV_VECTOR |
| 同步、精度与 tail | Queue；lambda FP16；U 计算/n 写回 | Queue；FP32；U/n | Queue；`CAST_NONE/CAST_RINT`；U/n | 无 event/atomic |
| fallback/失败边界 | 无 Device fallback | 无 Device fallback | 无 Device fallback | 非支持 dtype 在 API/Tiling 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档标为支持；固定仓无目标入口的活动实现闭包，未生成无后缀卡 | 同 A2 | 公开支持，且 `ascend950/arch35` 全链路闭合 |
| Host/Launcher | API wrapper 可见，但 A2 构建/Tiling/POD/Kernel 接线缺失 | 同左 | `IsRegbase` dtype 表；`HardShrinkGrad` launcher + arch35 Tiling |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | 六个 AIV_VECTOR 模板组合；三 Queue + mask，BF16 加两 FP32 TBuf |

## 限制与失败边界

- A2/A3 只有公开支持声明和共享 API wrapper，固定 Commit 没有对应活动 OpDef/config、Tiling 与 Kernel 构建闭包；不得从 950 或 `experimental/` 实现外推。
- 公开文档要求三 Tensor dtype 一致；API 实现同时包含 `PromoteType` 与输出 Cast。跨 dtype 组合是否属于公开承诺存在文档/实现宽严差异，复用时应遵守公开文档。
- Tiling 只比较三 Tensor 的 storage 元素数是否相等，不比较逐维 shape；ACLNN 已在外围完成 broadcast 并产生等 shape 连续核心输入。绕过 ACLNN 直接调用内部 Op 时不能依赖该外围保证。
- `availableUb=ubSize-16384` 没有显式正值 guard；`ubFactor` 也没有显式 `>0` guard。平台资源需满足公式隐含前提。
- selector 的 `bufferBytes` 不单列 `maskBuf_`，而是另减 16KiB 固定预留；固定源码没有给出“显式 Queue/TBuf 总和 + 固定系统开销”的最终断言，不能把注释当作任意 UB 配置上的形式证明。
- `DataCopyPad.blockLen` 把 `currentNum*sizeof(T)` 窄化为 `uint16_t`，Tiling 没有显式 65535B 上限检查；950 实际 UB 配置是否天然保证该上限，固定仓公共平台实现不可见。
- Vector 计算覆盖完整 U，输入 DMA 不填充尾部无效 lane；安全性依赖逐 lane 独立且 CopyOut 仅写 `currentNum`。不得迁移到跨 lane 算法。
- `self` 或 `lambd` 为 NaN 时，`LE` 比较为假并传递梯度，这与直接按公开两个严格不等式解释 NaN 的结果不同；源码未拒绝非有限值。
- executor 总 workspace 的外围分片、通用 `Contiguous/BroadcastTo/Cast/ViewCopy` Kernel body 与内部同步实现属于固定仓框架边界；必须使用第一段返回值和同一 executor/stream，不得自行假设总 workspace 为 0。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/hard_shrink_grad/docs/aclnnHardshrinkBackward.md` | 产品表、公式、两段式原型 | 公开性、A2/A3/950 声明、接口约束 |
| 入口声明/约束 | `activation/hard_shrink_grad/op_api/aclnn_hardshrink_backward.h` | 两个 ACLNN 声明 | ABI 参数顺序 |
| 入口实现 | `activation/hard_shrink_grad/op_api/aclnn_hardshrink_backward.cpp` | `CheckParams`、`BroadcastTo`、两段式函数 | 校验、空域、框架节点图、lambda 钳位、workspace/executor |
| L0 launcher | `activation/hard_shrink_grad/op_api/hardshrink_grad.cpp` | `l0op::HardShrinkGrad` | 中间输出分配和核心 launcher |
| OpDef/平台 | `activation/hard_shrink_grad/op_host/hard_shrink_grad_def.cpp` | `ops::HardShrinkGrad` | dtype/ND、`ascend950` 注册、apt 文件名 |
| Host/Tiling | `activation/hard_shrink_grad/op_host/arch35/hard_shrink_grad_tiling.cpp` | `HardShrinkGradTilingFunc` | 平台资源、分核、UB、key、workspace |
| POD | `activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad_tiling_data.h` | `HardShrinkGradTilingData` | 四字段 Host/Device ABI |
| 模板选择 | `activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 六个 dtype×buffer 实例 |
| Kernel entry | `activation/hard_shrink_grad/op_kernel/hard_shrink_grad_apt.cpp` | `hard_shrink_grad` | POD 解析与模板 body 调用 |
| Kernel body | `activation/hard_shrink_grad/op_kernel/arch35/hard_shrink_grad.h` | `HardShrinkGrad::Init/CopyIn/Compute/CopyOut/Process` | GM/UB、Vector、同步、精度、tail |
| 构建接线 | `activation/hard_shrink_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR` | 仅 `ascend950/arch35` 活动构建 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 四字段共同驱动所有 Route | 字段顺序、`int64/int64/int64/float`、元素单位、lambda 已钳负值 | `HardShrinkGradTilingData` |
| 调用顺序 | `Contiguous→可选 BroadcastTo→Cast→HardShrinkGrad→Cast→ViewCopy`；Kernel `CopyIn→Compute→CopyOut` | 广播与提升必须先于核心；输出 Cast/ViewCopy 后置 | `aclnnHardshrinkBackwardGetWorkspaceSize`、`Process` |
| workspace/内存 | 核心 workspace 0；U 由 dtype/depth 预算；BF16 两块 FP32 TBuf | 总 executor workspace 不等于核心 workspace；mask、depth 和 16KiB 预留不可漏 | `GetWorkspaceSize`、`HardShrinkGradTilingFunc`、`Init` |
| 同步 | Queue 管理 DMA/Vector producer-consumer；核心无跨核同步 | slot 只能在 `FreeTensor` 后覆盖；mask 在 Select 前有效 | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | FP16/FP32 native；BF16 FP32 比较后 `CAST_RINT`；U 计算/n 写回 | `CMPMODE::LE`、Select 输入次序、round mode、CopyOut 有效长度不可变 | `Compute`、`CopyOut` |
| 失败边界 | 空域 API 快返；资源、shape、dtype、POD 获取失败阻断 launch | 不得绕过 A2/A3 缺失闭包、`uint16_t` DMA 与 UB 隐含前提 | `CheckParams`、`HardShrinkGradTilingFunc` |

- [ASCENDC_TPL_SEL_PARAM](../../../../apis/utils_api/tiling_template_programming/ascendc_tpl_sel_param.md)
