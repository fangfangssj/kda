---
type: Operator
title: "NLLLossBackward 950 ACLNN 算子知识"
description: "说明 aclnnNLLLossBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/docs/aclnnNLLLossBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_api/aclnn_nll_loss_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_api/aclnn_nll_loss_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_api/nll_loss_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_api/aclnn_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/nll_loss_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/config/ascend950/nll_loss_grad_binary.json
    role: implementation
created_at: "2026-08-25T17:17:42Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# NLLLossBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnNLLLossBackwardGetWorkspaceSize` + `aclnnNLLLossBackward`，公开 ACLNN 两段式接口；固定仓未出现更高版本后缀入口。 |
| 计算语义 | 先将 `out` 全域置零；对每个未被忽略的样本 `n`，仅在类别 `target[n]` 处写 `-gradOutput[n] * weight[target[n]]`（`none`）、`-(gradOutput[0] / totalWeight[0]) * weight[target[n]]`（`mean`）或 `-gradOutput[0] * weight[target[n]]`（`sum`）。 |
| 输入输出 | ACLNN 仅接受 `self/out` 为 `[C]` 或 `[N,C]`、ND；值类型 FLOAT/FLOAT16/BFLOAT16，`target` 为标量或 `[N]` 且类型 INT32/INT64/UINT8；`weight=[C]`、`totalWeight` 单元素。 |
| 执行类型 | 核心节点为单 Kernel 非 Cube Hybrid：AIV `InitOutput` 清零 + SIMT `asc_vf_call` 稀疏写回；无 AIC/Cube、无 MIX_CV。ACLNN 的 Contiguous/Cast/Unsqueeze/Squeeze/ViewCopy 是框架边界。 |
| Kernel 路线 | R-001 `none`、R-002 `mean`、R-003 `sum`；三者均为 key `0`，共享 `KernelNLLLossGrad<DTYPE_X, DTYPE_TARGET>`，由 `reductionMode` 在 Device 内分支。 |
| 核心资源 | GM 输入/输出；每核三个请求容量为 1 个 FP32 的 `TBuf<VECCALC>`；SIMT 线程数 `min(platformMaxThread,1024)`；核心 Tiling workspace 固定 16 MiB，local-memory envelope 为平台 UB 减 32 KiB。无 `TQue`、无 Atomic。 |
| 来源闭包 | `source_closed`：公开 ACLNN、950 API selector、Host/Tiling、POD、Kernel entry、Hybrid body、公共 Tiling 调用顺序及 950 构建/二进制配置均闭合；框架通用变换按允许边界标注。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 ACLNN 输入 `self` 的逻辑形状为 `[C]` 或 `[N,C]`。一维情形在核心算子前被视为 `[1,C]`。令 `t_n=target[n]`，`w_c=weight[c]`，输出初值全部为零。若 `t_n==ignoreIndex`，该行保持零；否则 Device 端要求 `0<=t_n<C`，并写：

```text
reduction = none: out[n,t_n] = -cast_T(float(gradOutput[n]) * float(w_t_n))
reduction = mean: out[n,t_n] = -cast_T((float(gradOutput[0]) / float(totalWeight[0])) * float(w_t_n))
reduction = sum : out[n,t_n] = -cast_T(float(gradOutput[0]) * float(w_t_n))
```

除目标类别外不执行乘法写回，而由此前的整输出清零给出零梯度。`self` 的数值在核心 Kernel 中不读取；它只提供输出 shape、dtype 和二进制模板类型。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | `none` 且 `[N,C]` 时为 `[N]`；其余为标量或单元素一维；ND，可非连续 | 与 `self` 相同，FLOAT/FLOAT16/BFLOAT16 | `none` 逐样本读取；`mean/sum` 只使用元素 0。 |
| `self` | `[C]` 或 `[N,C]`，ND，可非连续 | FLOAT/FLOAT16/BFLOAT16 | 与 `out` shape、dtype 相同；数值不参与公式。空 tensor 在 ACLNN 第一段直接成功返回且 workspace 为 0。 |
| `target` | 标量或 `[N]`，ND，可非连续 | INT32/INT64/UINT8 | `self=[C]` 时须为标量；其他非 `ignoreIndex` 值由 Device `TargetCheck` 要求落在 `[0,C)`。 |
| `weight` | `[C]`，ND，可非连续 | 与 `self` 相同 | 按目标类别随机读取。 |
| `reduction` | Host 标量 | `int64_t` | ACLNN 接受 0/1/2，依次转为 `none/mean/sum`，再由 Tiling 编为 0/1/2。 |
| `ignoreIndex` | Host 标量 | `int64_t` | 命中时跳过目标位置写回；内部 OpDef 默认值为 -100，公开 ACLNN 显式传入。 |
| `totalWeight` | 单元素，ND | 与 `self` 相同 | `mean` 的分母。公开文档称其他 reduction 不处理，但可见 Kernel 对三个 reduction 都加载它并计算一次除法；仅 `mean` 使用商。 |
| `out` | 与 `self` 相同，ND，可非连续 | 与 `self` 相同 | 核心临时输出连续，最终经框架 `ViewCopy` 写入用户 view。 |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段返回完整 executor 图的 workspace 和所有权。 |
| `workspace/executor/stream` | 第二段输入 | 指针/handle | `CommonOpExecutorRun` 在给定 stream 执行第一段构造的图。 |

### 入口与版本

公开文档和头文件只声明无版本后缀的 `aclnnNLLLossBackwardGetWorkspaceSize/aclnnNLLLossBackward`。文档产品表明确支持 Ascend 950PR/950DT；`GetDtypeSupportListBySocVersion` 将 `DAV_3510` 接口 dtype 设为 FLOAT/FLOAT16/BFLOAT16，`AclnnUtil::IsRegbase` 也精确以 `DAV_3510` 判定 950。Level-0 selector 的 950 AICore 列表覆盖上述三种 dtype，因此 950 的非空合法输入均进入 `NLLLossGradAiCore`，不会进入 AICPU fallback。

本卡只覆盖 `aclnnNLLLossBackward` 的一维/二维语义，不把独立公开入口 `aclnnNLLLoss2dBackward` 或四维内部 body 合并进来。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnNLLLossBackwardGetWorkspaceSize` 的顺序为：创建 `aclOpExecutor`；检查六个 tensor 指针；要求 `weight/gradOutput/totalWeight` 与 `self` 同 dtype、`out` 可由 `self` dtype 转换且 shape 相同；按 950 dtype 表校验 `self`，按 INT32/INT64/UINT8 校验 `target`；校验 `reduction` 在 `[0,2]`；再校验 rank、batch、`weight=[C]`、`totalWeight` 单元素和 `gradOutput` shape。任何检查失败立即返回对应错误。

空 `self` 在构图前返回 `workspaceSize=0`。非空时 950 的 `IsRegbase()==true`，所以 `promoteType=self.dtype`，且 `target` 不强制转 INT32。随后按源码顺序将 `gradOutput/self/weight/target/totalWeight` 变连续并经过同 dtype Cast；若原 `self=[C]`，对核心输入在轴 0 `UnsqueezeNd` 为 `[1,C]`。核心结果随后按需 `SqueezeNd`、Cast 到 `out.dtype`，最后 `ViewCopy` 到用户输出。上述通用算子可能被 executor 消除或发射独立任务，其实现和消除规则不在固定算子源码内，作为框架边界。

### 参数、Workspace 与生命周期

第一段中的 `uniqueExecutor` 持有所有临时 tensor 与任务描述，最终由 `ReleaseTo(executor)` 转移给调用者；第二段用同一 `executor`、用户按返回值分配的 workspace 和 stream 执行。完整 executor workspace 为 `uniqueExecutor->GetWorkspaceSize()`，包含框架变换与核心节点，固定算子源码未公开其聚合公式。

核心 `NLLLossGrad` Tiling 单独将 `workspace[0]` 设为 `16*1024*1024` 字节。Kernel 要求系统 workspace 和 `GetUserWorkspace(workspace)` 均非空，调用 `SetSysWorkspace` 后并不再切片或读写 `userWS`；这 16 MiB 是 AscendC tools/system workspace 契约，不是算子中间结果。Tiling 还以 `SetLocalMemorySize(platformUB-32KiB)` 给出 SIMT local-memory envelope；显式 `TBuf` 由每个 Kernel block 的 `TPipe` 持有，Kernel 结束时失效。

### 启动图与失败传播

```text
ACLNN GetWorkspaceSize
  -> [框架边界] Contiguous/Cast/可选 Unsqueeze
  -> NLLLossGradAiCore 添加核心节点
     -> nll_loss_grad(x, y_grad, target, weight, total_weight, x_grad, workspace, tiling)
  -> [框架边界] 可选 Squeeze/Cast/ViewCopy
ACLNN run -> CommonOpExecutorRun(..., stream)
```

| 节点/边 | producer / consumer | 参数或介质 | ordering、owner/lifetime 与失败传播 |
|---|---|---|---|
| 前处理节点 | ACLNN API / `NLLLossGradAiCore` | 连续的 `gradOutput,self,target,weight,totalWeight`；一维 `self` 变 `[1,C]` | 同一 executor 图；任一返回空指针即停止构图并返回内部错误。临时 tensor 归 executor。 |
| 核心节点 | `NLLLossGradAiCore` / `nll_loss_grad` | `OP_INPUT(self,gradOutput,target,weight,totalWeight)`，输出连续 `x_grad`，POD 为 `NLLLossGradSimtTilingData`，`blockDim=max(zeroCoreCount,computeCoreCount)` | 单个非 Cube Hybrid Kernel；由 executor 在同一 stream 排序。workspace 由调用者分配、第二段期间有效。 |
| 后处理节点 | 核心 `x_grad` / 用户 `out` | 可选 reshape/cast 后 `ViewCopy` | 核心或任一后处理失败由 `CommonOpExecutorRun` 返回；固定算子源码未公开框架任务内部 workspace slice。 |

核心 Kernel 内先完成分核清零，再由每个计算 block 的线程 0 把标量装入本核 UB；所有 block 到达第二个 `SyncAll` 后才执行稀疏 SIMT 写回。没有跨 Kernel workspace 边。

## Route 与参数设计

### 分核与资源计算

ACLNN 一维输入已提升为 `[1,C]`，所以核心 Host 侧始终按二维 `[N,C]`：`batchNum=N`、`classNum=C`、`totalOutElements=N*C`。空 tensor 已在 ACLNN 层截断，因此核心路径的 `totalOutElements>0`。

```text
zeroCoreCount = min(N*C, platformAivCoreNum)
blockPerCore  = floor((N*C) / zeroCoreCount)
blockTailCore = N*C - blockPerCore*zeroCoreCount
usedThread    = min(platformSimtMaxThread, 1024)
computeCoreCount = min(ceil(N / usedThread), platformCoreNum)
blockDim      = max(zeroCoreCount, computeCoreCount)
localMemorySize = platformUBBytes - 32*1024
coreWorkspace = 16*1024*1024 bytes
```

清零 block `b<zeroCoreCount` 处理连续区间：若 `b<blockTailCore`，长度为 `blockPerCore+1`、起点为 `b*blockPerCore+b`；否则长度为 `blockPerCore`、起点为 `b*blockPerCore+blockTailCore`。计算 block 内 SIMT 线程覆盖 `i=b*usedThread+threadIdx.x`，后续步长为 `blockDim*usedThread`，直到 `i>=N`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 no-reduction | key `0`; `KernelNLLLossGrad<T,F>` | ACLNN `reduction==0` → 字符串 `none` → POD `reductionMode=0`；非空且通过校验 | `reductionMode!=0` 继续 Device 分支 |
| 2 | R-002 mean | key `0`; `KernelNLLLossGrad<T,F>` | ACLNN `reduction==1` → 字符串 `mean` → POD `reductionMode=1` | `reductionMode!=1` 继续 Device 分支 |
| 3 | R-003 sum | key `0`; `KernelNLLLossGrad<T,F>` | ACLNN `reduction==2` → 字符串 `sum` → POD `reductionMode=2` | 其他值已在 ACLNN/Host 失败 |

```text
self empty? -> workspace=0、无核心节点
否则 A5(DAV_3510) 且 self dtype in {FP32,FP16,BF16} -> NLLLossGradAiCore
  Tiling: rank/dtype/target/reduction 合法? -> key 最终强制为 0
  Device:
    reductionMode==0 -> R-001
    else reductionMode==1 -> R-002
    else reductionMode==2 -> R-003
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 no-reduction | `0`；binary config 的 9 个 `(T,F,none)` 实例 | 950；`T∈{float,half,bfloat16}`；`F∈{int32,int64,uint8}`；核心 shape `[N,C]`；mode 0 | 单 Kernel：AIV 全域清零 → SIMT 每样本稀疏写 | `KernelNLLLossGrad<T,F>::Process` → `SimtComputeNoReduce2d<T,F>` | 三个 FP32 标量 TBuf 仍分配；`SyncAll` 后直接从 GM 逐样本读 `y_grad` | 线程 grid-stride 覆盖 N；ignore 跳过；非法 target `ASSERT`；无 fallback |
| R-002 mean | `0`；9 个 `(T,F,mean)` 实例 | 同上，mode 1 | 单 Kernel：AIV 清零 → 每核预载标量/FP32 除法 → SIMT 稀疏写 | `SimtComputeReduceMean2d<T,F>` | 每计算核线程 0 读两个标量；`tmpOut=singleGrad/localTotalWeight`；barrier 后共享给本核线程 | 同上；`totalWeight==0` 的可靠检查无法从固定源码闭合，无 fallback |
| R-003 sum | `0`；9 个 `(T,F,sum)` 实例 | 同上，mode 2 | 单 Kernel：AIV 清零 → 每核预载标量 → SIMT 稀疏写 | `SimtComputeReduceSum2d<T,F>` | 使用 `singleGrad`，虽仍计算 `tmpOut` 但结果不消费 | 同上；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| ACLNN 空域短路 | `self->IsEmpty()` | 避免 Tiling 中 `zeroCoreCount=0` 后除零，也无需输出写回 | 全部 | workspace 0，直接成功 | `op_api/aclnn_nll_loss_backward.cpp::aclnnNLLLossBackwardGetWorkspaceSize` |
| 950 AICore dtype | `DAV_3510` 且 `self.dtype∈{FP32,FP16,BF16}` | 950 binary config 仅实例化这些值类型 | 全部 | API 非法 dtype 先失败；合法 950 dtype 不走 AICPU | `op_api/nll_loss_grad.cpp::GetAiCoreDtypeSupportListBySocVersion/IsAiCoreSupport` |
| 输入 rank | ACLNN `self.rank∈{1,2}`；rank1 先 Unsqueeze | 本卡只使 2D SIMT body 可达，避免合并独立 2d API 的 4D 路线 | 全部 | ACLNN 参数错误 | `op_api/aclnn_nll_loss_backward.cpp::CheckShape` |
| SIMT 线程上限 | `usedThread=min(GetSimtMaxThreadNum(context),1024)` | `LAUNCH_BOUND(1024)` 且 Host/Device 使用同一线程数计算覆盖域 | 全部 | 无其他模板；平台返回值被截到 1024 | `op_host/arch35/nll_loss_grad_tiling_arch35.cpp::NLLLossGradParse/PostTiling` |
| local-memory 预留 | `localMemorySize=platformUB-32768` | 为 DCache 固定保留 32 KiB；三个显式 TBuf 在剩余 local memory 内 | 全部 | 固定仓无 fallback；若平台 UB 获取失败则 Tiling 失败 | `NLLLossGradSimtTiling::GetShapeAttrsInfo/PostTiling` |
| workspace 指针 | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | entry 在设置系统 workspace 后才进入 body | 全部 | Kernel 直接返回、无输出提交 | `op_kernel/nll_loss_grad.cpp::nll_loss_grad` |
| target 有效域 | `target[i]==ignoreIdx \|\| 0<=target[i]<C` | 稀疏 GM 地址使用 `target` 作为类别偏移 | 全部 | ignore 跳过；其他越界触发 Device `ASSERT`，无 fallback | `op_kernel/arch35/nll_loss_grad.h::TargetCheck` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 的序列化宏实现来自仓外工具链；固定仓可见字段声明顺序和标量位宽，但精确 `sizeof/alignof/offsetof` 不可见。下表的 ABI 布局列因此不推测 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `NLLLossGradSimtTilingData` | `batchNum` | `uint64_t`/64 | 声明序号 1；精确 offset/总 sizeof 固定仓不可见 | 元素行数 N | `ProcessShapeInfo`: rank1 写 1，rank2 写 dim0 | `Init` → `batchNum_`；SIMT 循环上界 | ACLNN 核心非空；rank1 经 API 后实际为 rank2 |
| 同上 | `classNum` | `uint64_t`/64 | 序号 2；精确 offset 不可见 | 类别数 C | `ProcessShapeInfo`: rank1 dim0 或 rank2 dim1 | 地址公式、`TargetCheck` | 核心路径要求输出元素数非零 |
| 同上 | `usedThread` | `uint64_t`/64 | 序号 3；精确 offset 不可见 | thread/block | `PostTiling`: `min(maxThread_,1024)` | `Init` 窄化到 `uint32_t usedThread_`；所有 `asc_vf_call(dim3{...})` | 上界 1024 使窄化安全 |
| 同上 | `height` | `uint64_t`/64 | 序号 4；精确 offset 不可见 | 元素 | 仅内部 rank4 在 `ProcessShapeInfo` 赋值 | `Init`；仅 4D body 地址计算 | 对本 ACLNN 的 xDims=2 inactive；其序列化值未由可见 producer 明确写入 |
| 同上 | `width` | `uint64_t`/64 | 序号 5；精确 offset 不可见 | 元素 | 仅内部 rank4 赋值 | 同上 | 对本 ACLNN inactive；序列化值不可见 |
| 同上 | `veryImportantProcessCoreNums` | `uint64_t`/64 | 序号 6；精确 offset 不可见 | block/core | `PostTiling`: `min(ceil(N/usedThread),coreNum_)` | `Init`: `isComputeCore_` | `coreNum_` 来自 AIV core 数 |
| 同上 | `notVeryImportantProcessCoreNums` | `uint64_t`/64 | 序号 7；精确 offset 不可见 | block/core | `DoOpTiling`: `min(N*C,aivCoreNum_)`，`PostTiling` 写入 | `Init`: `isMemsetCore_` | 非空 ACLNN 路径至少为 1 |
| 同上 | `blockPerCore` | `uint64_t`/64 | 序号 8；精确 offset 不可见 | 输出元素/core | `DoOpTiling`: `floor(N*C/zeroCoreCount)` | `ProcessMemset` | Host 成员为 `int64_t`，保存时转 64 位无符号；核心非空避免负/零除数 |
| 同上 | `blockTailCore` | `uint64_t`/64 | 序号 9；精确 offset 不可见 | core 数/余数元素 | `DoOpTiling`: `N*C-blockPerCore*zeroCoreCount` | `ProcessMemset` former/tail 分支 | 范围 `[0,zeroCoreCount)` |
| 同上 | `xDims` | `uint32_t`/32 | 序号 10；精确 offset 不可见 | rank | `ProcessShapeInfo`: 输入 storage rank | `Process`: 1/2 走 2D，4 走 4D | 本 ACLNN 经 reshape 后核心可达值为 2 |
| 同上 | `reductionMode` | `uint32_t`/32 | 序号 11；精确 offset 不可见 | 枚举 | `ProcessAttributesInfo`: none/mean/sum→0/1/2 | `Process` Device 分支 | 其他字符串 Tiling 失败 |
| 同上 | `ignoreIdx` | `int64_t`/64 | 序号 12；精确 offset 不可见 | 类别标签值 | `ProcessAttributesInfo`: attr 1 原值 | 所有 SIMT compute 的忽略判断 | 可在 `[0,C)` 内或外；命中优先于有效域断言 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 no-reduction | key 0；950 AICore；mode 0 | `NLLLossGrad`、`NLLLossGradSimtTiling::{ProcessShapeInfo,DoOpTiling,PostTiling}` | `NLLLossGradSimtTilingData`；`reductionMode=0` | `op_kernel/nll_loss_grad.cpp::nll_loss_grad` → `COMMON_OP_IMPL` | `KernelNLLLossGrad::{Init,ProcessMemset,Process}`、`SimtComputeNoReduce2d` | 算子/Kernel CMake 均仅 `ascend950`；binary config 9 个 none dtype 组合 |
| R-002 mean | key 0；mode 1 | 同上；`ProcessAttributesInfo` 写 1 | 同一 POD；`totalWeight`/`singleGrad`/`tmpOut` active | 同上 | `ComputeSetValue`、FP32 除法、`SimtComputeReduceMean2d` | 9 个 mean dtype 组合 |
| R-003 sum | key 0；mode 2 | 同上；`ProcessAttributesInfo` 写 2 | 同一 POD；`singleGrad` active，`tmpOut` 计算但不消费 | 同上 | `ComputeSetValue`、`SimtComputeReduceSum2d` | 9 个 sum dtype 组合 |

## Kernel 实现

### 公共执行流程

核心 launch graph 只有一个 Device 节点。entry 先验证 system/user workspace，解析 POD，声明任务为 `KERNEL_TYPE_MIX_AIV_1_0`，再以 binary config 注入的 `DTYPE_X/DTYPE_TARGET` 实例化 `KernelNLLLossGrad<T,F>`。该 task 标记只含 AIV 比例，body 中没有 Cube API；真实管线是 AIV `InitOutput` 加 SIMT VF，因此仍由 VV Skill 收录。

所有 Route 都先把 `N*C` 个输出按 block 连续分片清零，随后为每核分配三个 FP32 标量 TBuf。计算 block 用 `ComputeSetValue` 的线程 0加载 `y_grad[0]` 和 `total_weight[0]`，全体到达 `SyncAll`，再算 `tmpOut=singleGrad/localTotalWeight` 并用 `PipeBarrier<PIPE_ALL>` 使 UB 标量可供后续 SIMT 调用读取。最后按 `reductionMode` 选择对应函数；每个线程处理一个或多个样本，只写目标类别地址。

### Route R-001：no-reduction

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `loss/nll_loss_grad/op_api/aclnn_nll_loss_backward.cpp` | `CheckReduction/GetReductionStr` | `reduction==0` | attr 字符串 `none` |
| Level-0 | `loss/nll_loss_grad/op_api/nll_loss_grad.cpp` | `NLLLossGrad` | 950 三种 API dtype 都满足 `IsAiCoreSupport(self)` | 添加 `NLLLossGrad` AICore 节点 |
| Tiling | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo/GenerateTilingKey` | `strcmp(reduction,"none")==0`；dtype/target 合法后 key 被覆写为 0 | POD mode 0，key 0 |
| Router | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | workspace 与 user workspace 非空 | `KernelNLLLossGrad<DTYPE_X,DTYPE_TARGET>` |
| Device | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `Process` | `xDims_∈{1,2} && reductionMode_==NONE_MODE` | `SimtComputeNoReduce2d` |

#### Device 数据流

##### 函数级调用链

```text
NLLLossGradAiCore -> nll_loss_grad(..., POD)
  -> KernelNLLLossGrad<T,F>::Init
  -> KernelNLLLossGrad<T,F>::Process
    -> ProcessMemset -> InitOutput<T>(本核连续输出片, 0)
    -> InitBuffer(3 个 FP32 标量 TBuf)
    -> asc_vf_call<ComputeSetValue<T,F>>(usedThread)  # 结果仅保留公共流程一致性
    -> SyncAll -> tmpOut=singleGrad/localTotalWeight -> PipeBarrier
    -> asc_vf_call<SimtComputeNoReduce2d<T,F>>(usedThread)
```

##### 源码等价伪码

```text
# 伪码；b=GetBlockIdx(), B=GetBlockNum(), U=usedThread
if b < zeroCoreCount:
    len = blockPerCore + (b < blockTailCore ? 1 : 0)
    off = b*blockPerCore + min(b, blockTailCore)
    output[off : off+len] = 0

allocate localTotalWeight[1], singleGrad[1], tmpOut[1] as FP32 UB
if b < computeCoreCount and tid == 0:
    localTotalWeight[0] = float(totalWeight[0])
    singleGrad[0] = float(yGrad[0])
SyncAll()
if b >= computeCoreCount: return
tmpOut[0] = singleGrad[0] / localTotalWeight[0]  # 本 Route 不消费
PipeBarrier(PIPE_ALL)
for i = b*U + tid; i < N; i += B*U:
    c = int64(target[i])
    if c == ignoreIdx: continue
    assert 0 <= c < C
    output[i*C+c] = -T(float(yGrad[i]) * float(weight[c]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x/self` | base 0 | `N*C` 元素 | T/ND | ACLNN contiguous/cast | 仅 `Init` 绑定，body 不读 | executor 输入临时 | 无搬运到 UB |
| GM `y_grad` | `i`（公共预载另读 0） | `N`（或 N=1） | T/ND | ACLNN | `ComputeSetValue`、`SimtComputeNoReduce2d` | executor 任务期间 | SIMT 标量随机读 |
| GM `target` | `i` | N | F/ND | ACLNN | 每个 SIMT 线程 | 同上 | 无 UB 物化 |
| GM `weight` | `target[i]` | C | T/ND | ACLNN | 每个非忽略线程 | 同上 | 按类别随机读 |
| GM `total_weight` | 0 | 1 | T/ND | ACLNN | 公共 `ComputeSetValue` | 同上 | 本 Route 商不消费，但仍读取 |
| GM `output` 清零片 | `b*blockPerCore+min(b,tail)` | `blockPerCore+(b<tail)` | T/ND | `InitOutput` | 后续稀疏写/最终输出 | 核心临时 tensor | block 间无重叠 |
| GM `output` 目标点 | `i*C+target[i]` | 每样本 1 元素 | T/ND | SIMT 线程 | 后处理 | 同上 | 不同 i 属于不同行，无 Atomic |
| UB `localTotalWeightBuf_` | 本核 TBuf | 请求 `1*sizeof(float)` | FP32 | thread 0 | 公共除法 | `TPipe`/Kernel block | 无别名证据 |
| UB `singleGradBuf_` | 本核 TBuf | 请求 `1*sizeof(float)` | FP32 | thread 0 | 公共除法 | 同上 | 无别名证据 |
| UB `tmpOutBuf_` | 本核 TBuf | 请求 `1*sizeof(float)` | FP32 | scalar 除法 | 本 Route不消费 | 同上 | 无别名证据 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  totalW["localTotalWeightBuf · TBuf VECCALC · 请求 4 B"]:1
  grad["singleGradBuf · TBuf VECCALC · 请求 4 B"]:1
  tmp["tmpOutBuf · TBuf VECCALC · 请求 4 B"]:1
```

##### 字节账与复用顺序

```text
explicit_UB_request = 1*4 + 1*4 + 1*4 = 12 bytes/block
physical_aligned_UB = 固定仓不可见（TPipe 对 4-byte InitBuffer 的物理对齐规则在外部 AscendC 工具链）
local_memory_envelope = platformUBBytes - 32*1024
dynamic_ub = local_memory_envelope（由 SetLocalMemorySize 设置）
core_workspace = 16*1024*1024 bytes
executor_workspace_total = 固定仓不可见（由 executor 聚合框架节点）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `singleGrad_[0]` | `ComputeSetValue` thread 0 | scalar `tmpOut` 计算 | Kernel 结束，无覆盖 | `SyncAll` 后读取 | 仅计算 block 读取 |
| `localTotalWeight_[0]` | `ComputeSetValue` thread 0 | scalar `tmpOut` 计算 | Kernel 结束 | `SyncAll` | 同上 |
| `tmpOut_[0]` | scalar 除法 | 本 Route不消费 | Kernel 结束 | `PipeBarrier` 后分支 | 不作为 R-001 结果来源 |
| GM output 零值 | `InitOutput` | 未命中目标位置的最终值 | 命中位置由 SIMT 覆盖 | 所有 block 的第二个 `SyncAll` 位于稀疏写前 | 清零分片覆盖全域且互不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全域清零 | `InitOutput` + `SyncAll` | memset cores | 所有 compute cores 在第二个 `SyncAll` 后写回 | GM output | former/tail core 精确覆盖 `N*C`；无 padding 写 |
| 标量预载 | `asc_vf_call<ComputeSetValue>` + `SyncAll` | 每核 thread 0 | 本核 scalar/SIMT 阶段 | 三个本核 UB 标量 | T→FP32；本 Route的商不影响输出 |
| 稀疏计算 | SIMT grid-stride | 每线程 GM load/multiply | 直接 GM store | 单行目标位置 | T 输入升为 FP32 相乘，乘积先 cast T 再取负；ignore 跳过，循环条件保护 tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessShapeInfo/DoOpTiling/PostTiling` | 生成 N/C、清零与计算核数、threads、blockDim、workspace 和 local memory |
| Kernel entry/router | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` / `KernelNLLLossGrad<DTYPE_X,DTYPE_TARGET>` | workspace guard、POD 解析和唯一模板实例 |
| Device body/helper | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `ProcessMemset/Process/SimtComputeNoReduce2d` | 分片清零、同步、样本线程映射、稀疏地址和 FP32 乘法 |

### Route R-002：mean

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `loss/nll_loss_grad/op_api/aclnn_nll_loss_backward.cpp` | `GetReductionStr` | `reduction==1` | attr `mean` |
| Level-0 | `loss/nll_loss_grad/op_api/nll_loss_grad.cpp` | `NLLLossGrad` | 950 AICore dtype 谓词 | `NLLLossGradAiCore` |
| Tiling | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo` | `strcmp(reduction,"mean")==0` | POD mode 1，key 0 |
| Router | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | workspace guards | 公共模板 |
| Device | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `Process` | xDims 1/2 且 mode 1 | `SimtComputeReduceMean2d` |

#### Device 数据流

##### 函数级调用链

```text
NLLLossGradAiCore -> nll_loss_grad -> KernelNLLLossGrad<T,F>::Init/Process
  -> ProcessMemset -> InitOutput<T>
  -> ComputeSetValue: thread0 loads yGrad[0], totalWeight[0] to FP32 UB
  -> SyncAll
  -> tmpOut[0] = singleGrad[0] / localTotalWeight[0]
  -> PipeBarrier<PIPE_ALL>
  -> asc_vf_call<SimtComputeReduceMean2d<T,F>>
```

##### 源码等价伪码

```text
# 清零和分核同 R-001
if compute block and tid==0:
    singleGrad[0] = float(yGrad[0])
    localTotalWeight[0] = float(totalWeight[0])
SyncAll()
if not compute block: return
tmpOut[0] = singleGrad[0] / localTotalWeight[0]
PipeBarrier(PIPE_ALL)
for i = b*U + tid; i < N; i += B*U:
    c = int64(target[i])
    if c == ignoreIdx: continue
    assert 0 <= c < C
    output[i*C+c] = -T(tmpOut[0] * float(weight[c]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `y_grad` / `total_weight` | 各自 0 | 各 1 元素 | T/ND | ACLNN | 每个计算 block 的 `ComputeSetValue` thread 0 | executor 输入 | 每 block 各读一次 |
| GM `target` / `weight` | `i` / `target[i]` | N / C | F / T | ACLNN | SIMT 线程 | executor 输入 | 随机类别读，无 UB tile |
| GM output | 清零片；目标 `i*C+c` | `N*C` | T/ND | `InitOutput`、SIMT | 后处理 | 核心临时 | 零值被目标位置原地覆盖 |
| UB `singleGrad` | 本核标量 | 4 B 请求 | FP32 | thread 0 cast | FP32 除法 | block | `SyncAll` 后读 |
| UB `localTotalWeight` | 本核标量 | 4 B 请求 | FP32 | thread 0 cast | FP32 除法 | block | 同上 |
| UB `tmpOut` | 本核标量 | 4 B 请求 | FP32 | 除法 | 全部本核 SIMT 线程 | block | `PipeBarrier` 后读 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  totalW["localTotalWeight · FP32 · 请求 4 B"]:1
  grad["singleGrad · FP32 · 请求 4 B"]:1
  tmp["tmpOut=grad/totalWeight · FP32 · 请求 4 B"]:1
```

##### 字节账与复用顺序

```text
explicit_UB_request = 12 bytes/block；物理对齐后占用固定仓不可见
dynamic_ub = platformUBBytes - 32768
core_workspace = 16777216 bytes
workspace_total = executor 聚合值，固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `singleGrad[0]` | thread 0 | FP32 除法 | 不覆盖 | `SyncAll` | 每 block 独立 UB |
| `localTotalWeight[0]` | thread 0 | FP32 除法 | 不覆盖 | `SyncAll` | 同上 |
| `tmpOut[0]` | scalar stage | `SimtComputeReduceMean2d` threads | 不覆盖 | `PipeBarrier<PIPE_ALL>` | 进入 SIMT 前完成写入 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零→稀疏写 | `SyncAll` | `InitOutput` | SIMT compute | GM output | 与 R-001 相同 |
| 标量加载→除法 | `SyncAll` | thread 0 | scalar 除法 | `singleGrad/localTotalWeight` | 两个 T 值先转 FP32再除；固定源码未提供明确可闭合的零分母保护 |
| 除法→SIMT | `PipeBarrier<PIPE_ALL>` | scalar `tmpOut` store | SIMT VF | `tmpOut` | FP32 商与 FP32 weight 相乘，再 cast T、取负；grid-stride 尾部以 `i<N` 保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo/PostTiling` | mode 1、线程与 blockDim |
| Kernel entry/router | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | 唯一 key-agnostic 模板 entry |
| Device body/helper | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `ComputeSetValue/Process/SimtComputeReduceMean2d` | 每核标量加载、FP32 除法、barrier 与目标位置写回 |

### Route R-003：sum

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `loss/nll_loss_grad/op_api/aclnn_nll_loss_backward.cpp` | `GetReductionStr` | `reduction==2`（合法范围内的 else） | attr `sum` |
| Level-0 | `loss/nll_loss_grad/op_api/nll_loss_grad.cpp` | `NLLLossGrad` | 950 AICore dtype 谓词 | `NLLLossGradAiCore` |
| Tiling | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo` | `strcmp(reduction,"sum")==0` | POD mode 2，key 0 |
| Router | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | workspace guards | 公共模板 |
| Device | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `Process` | xDims 1/2 且 mode 2 | `SimtComputeReduceSum2d` |

#### Device 数据流

##### 函数级调用链

```text
NLLLossGradAiCore -> nll_loss_grad -> KernelNLLLossGrad<T,F>::Init/Process
  -> ProcessMemset -> InitOutput<T>
  -> ComputeSetValue -> SyncAll
  -> tmpOut=singleGrad/localTotalWeight -> PipeBarrier  # 商不消费
  -> asc_vf_call<SimtComputeReduceSum2d<T,F>>
```

##### 源码等价伪码

```text
# 清零、标量加载和无效 tmpOut 除法同公共流程
for i = b*U + tid; i < N; i += B*U:
    c = int64(target[i])
    if c == ignoreIdx: continue
    assert 0 <= c < C
    output[i*C+c] = -T(singleGrad[0] * float(weight[c]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `y_grad` | 0 | 1 | T/ND | ACLNN | 每核 thread 0 | executor 输入 | 转 FP32 后本核复用 |
| GM `total_weight` | 0 | 1 | T/ND | ACLNN | 公共除法 | executor 输入 | 商不消费但依然读取 |
| GM `target/weight/output` | `i` / `target[i]` / `i*C+target[i]` | N / C / N*C | F/T/T | ACLNN、清零 | SIMT/后处理 | executor/core output | 无 Atomic；目标点覆盖零值 |
| 三个 UB TBuf | 本核 | 各请求 4 B | FP32 | thread 0/scalar | sum 使用 `singleGrad` | block | `tmpOut` inactive 但仍分配和写入 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  totalW["localTotalWeight · 请求 4 B · 仅无效商"]:1
  grad["singleGrad · 请求 4 B · sum active"]:1
  tmp["tmpOut · 请求 4 B · 不消费"]:1
```

##### 字节账与复用顺序

```text
explicit_UB_request = 12 bytes/block；物理对齐固定仓不可见
dynamic_ub = platformUBBytes - 32768
core_workspace = 16777216 bytes
workspace_total = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `singleGrad[0]` | thread 0 | `SimtComputeReduceSum2d` | 不覆盖 | `SyncAll`、随后 `PipeBarrier` | block 私有，所有线程只读 |
| `localTotalWeight/tmpOut` | thread 0 / scalar stage | 输出不消费 | 不覆盖 | Kernel 结束释放 | 不能据此删除 buffer，除非同时修改公共 `Process` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零→稀疏写 | `SyncAll` | memset cores | SIMT | GM output | 分片 former/tail + `i<N` |
| scalar→SIMT | `SyncAll` + `PipeBarrier` | thread 0 / scalar stage | `SimtComputeReduceSum2d` | `singleGrad` | `yGrad[0]` 转 FP32，乘 FP32 weight，cast T 后取负 |
| ignore/target | 条件 + `ASSERT` | target load | GM store | 地址合法性 | ignore 不写；其他 target 必须 `[0,C)` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo/PostTiling` | mode 2 与资源参数 |
| Kernel entry/router | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | 950 模板 entry |
| Device body/helper | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `ComputeSetValue/SimtComputeReduceSum2d` | 标量复用、FP32 乘法、SIMT grid-stride 与稀疏 store |

## 算子特定优化

### Trick T-001：全域分片清零后只计算目标类别

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：非空输出；`blockId<notVeryImportantProcessCoreNums` 的 block 负责互不重叠的连续清零片，随后 SIMT 只覆盖每个非忽略样本的目标类别。
- Golden 来源：[`loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h:256-269`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h#L256-L269)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if(this->isMemsetCore_){
    uint64_t initCoreNum = this->blockPerCore_;
    uint64_t initOffset = initCoreNum * this->blockId_;
    if(this->blockId_ < this->blockTailCore_){
        initCoreNum += 1;
        initOffset += this->blockId_;
    }else {
        initOffset += this->blockTailCore_;
    }
    InitOutput<T>(outputGm_[initOffset], initCoreNum, static_cast<T>(0));
    SyncAll();
}
```

#### 机制与设计原因

梯度矩阵每行最多只有目标类别一个非零元素。实现没有为全部 `N*C` 元素执行 target 比较和乘法：AIV `InitOutput` 以均衡连续片一次性建立零背景，SIMT 再以样本为并行域，仅对 `i*C+target[i]` 计算一次 FP32 乘法并覆盖。这样源码可证明乘法与 target/weight 随机访问量从按矩阵元素遍历变为按样本遍历，同时仍保证未命中位置为零。former/tail 公式保证所有清零片首尾相接；第二个全体 `SyncAll` 位于稀疏写之前，保护清零值不会在后写之后覆盖目标点。

#### 迁移边界

- 适用：输出大部分为确定零、每个逻辑行只有少量散点非零的反向算子。
- 前提：散点地址彼此无写冲突；能先完整初始化输出；有跨 block 的清零→写回 ordering；target 必须先处理 ignore 并校验范围。
- 失效：若每行有多个或动态数量非零点，需要扩展 SIMT 写集合；若多个线程可能写同一地址，必须增加冲突规约/Atomic；若无法全局同步，需拆为有序的两个 Kernel。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能在稀疏写之后再无序清零同一 GM 区域，否则目标梯度会被覆盖。固定仓未提供另一条无全局 ordering 的可验证 fallback。

## 模板与实现对比

| 对比维度 | R-001 no-reduction | R-002 mean | R-003 sum | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | mode 0 | mode 1 | mode 2 | ACLNN 0/1/2 顺序映射；非法值先失败 |
| 模板与 dtype | 共享 `KernelNLLLossGrad<T,F>`；9 个 dtype 组合 | 同左 | 同左 | binary config 将 T×F×reduction 展开成 27 个 950 binary，但 Device key 均为 0 |
| 循环与分核 | SIMT 遍历 N；清零遍历 N*C | 同左 | 同左 | reduction 不改变 shape 或线程映射 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 3 个 FP32 TBuf；无 TQue/通信 | 同左，三个都 active | `singleGrad` active，另两个仅公共无效商 | 公共 `Process` 未做条件分配 |
| 内存总占用与复用 | 显式请求 12 B/block；直接 GM 稀疏写 | 同左 | 同左 | 物理对齐固定仓不可见 |
| 搬运/Vector/SIMT/NDDMA/通信 API | AIV `InitOutput` + SIMT 逐样本读取 `yGrad[i]` | AIV + SIMT；每核复用 FP32 商 | AIV + SIMT；每核复用 FP32 grad | 无 NDDMA/通信/Cube |
| 同步、精度与 tail | `SyncAll`、FP32 乘、cast T | 加 FP32 除法与 `PipeBarrier` | 公共流程也执行除法；sum 只消费 grad | 清零 former/tail；SIMT `i<N` tail |
| fallback/失败边界 | ignore 跳过；target 越界 ASSERT | 同左；零分母边界不闭合 | 同 R-001 | 无第二模板或 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档声明支持，但固定仓无该平台 Host/Tiling/Kernel 与可达 AICPU body，未达到 `source_closed` | 同 A2 | 公开文档支持；OpDef、CMake、binary config 与 Kernel 均显式接线 `ascend950`，`source_closed` |
| Host/Launcher | API dispatcher 可见 DAV_2201 dtype/AICore 选择，但被选 body 不可见 | 同左 | `DAV_3510` 为 Regbase；三种值 dtype 均走 AICore |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | 单 Kernel AIV+SIMT Hybrid；16 MiB core workspace；UB-32 KiB local-memory envelope；3 个 FP32 TBuf |

## 限制与失败边界

- ACLNN 层只接受 `self` rank 1/2；独立 `aclnnNLLLoss2dBackward` 和内部 rank4 body 不属于本卡。
- `self`、`weight`、`gradOutput`、`totalWeight` 必须同 dtype，`out` 同 shape/dtype；`weight` 必须 `[C]`，`totalWeight` 必须单元素。
- 空 `self` 直接返回且不启动 Kernel。固定核心 Tiling 若绕过 ACLNN 直接接收零元素，会使 `FloorDiv(totalOutElements,zeroCoreCount)` 的除数为零；本入口以前置短路避免。
- 公开文档说非 mean 不处理 `totalWeight`，但固定 950 body 对三个 Route 都读取它并执行 `singleGrad/localTotalWeight`；只有 mean 使用结果。工程复用不能据文档在 none/sum 省略该 GM 输入或传空指针。
- mean helper 中的 `ASSERT` 表达式引用名为 `localTotalWeight`，该标识符的定义在固定仓算子源码中不可见；因此不能把它当作已闭合的零分母防护。可见执行语句本身会做 FP32 除法。
- POD 序列化宏、`TPipe` 的物理对齐、`GetSimtMaxThreadNum` 的工具链实现，以及 ACLNN executor 对通用 Contiguous/Cast/ViewCopy 的具体 Kernel、workspace 切片与优化消除规则均为固定仓不可见边界。
- Kernel entry 在 workspace 或 user workspace 为空时直接返回，没有写错误码；调用者必须使用第一段返回的 executor/workspace 契约。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/nll_loss_grad/docs/aclnnNLLLossBackward.md` | 产品支持表、函数原型、参数表 | 950 公开支持、接口语义与约束 |
| 入口声明/约束 | `loss/nll_loss_grad/op_api/aclnn_nll_loss_backward.h` | 两段式 ACLNN 声明 | ABI 参数顺序 |
| 入口实现 | `loss/nll_loss_grad/op_api/aclnn_nll_loss_backward.cpp` | `CheckParams/aclnnNLLLossBackwardGetWorkspaceSize/aclnnNLLLossBackward` | 校验、空域、框架变换、executor 和 stream |
| 入口实现 | `loss/nll_loss_grad/op_api/nll_loss_grad.cpp` | `NLLLossGrad/IsAiCoreSupport/NLLLossGradAiCore` | 950 selector 与核心 launcher |
| 入口实现依赖 | `common/inc/op_api/aclnn_util.h` | `AclnnUtil::IsRegbase` | DAV_3510 的 Regbase 判定 |
| Host 定义 | `loss/nll_loss_grad/op_host/nll_loss_grad_def.cpp` | `ops::NLLLossGrad` | dtype tuple、attr、ascend950 注册 |
| Host/Tiling | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.h` | `NLLLossGradSimtTilingData` | POD 字段及类型 |
| Host/Tiling | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `NLLLossGradSimtTiling` 全部 override | shape、分核、workspace、local memory、key |
| Host/Tiling 依赖 | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | override 的真实调用/失败顺序 |
| Kernel entry/body | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | workspace guard、POD 解析、模板实例 |
| Kernel entry/body | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `KernelNLLLossGrad` 与 SIMT helper | 清零、UB、同步、线程、地址、精度与 tail |
| 构建接线 | `loss/nll_loss_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | Host 只接 ascend950/arch35 |
| 构建接线 | `loss/nll_loss_grad/op_kernel/CMakeLists.txt` | `add_kernel_sources` | Kernel 只编译 ascend950 |
| 构建接线 | `loss/nll_loss_grad/op_host/config/ascend950/nll_loss_grad_binary.json` | `op_list` | 3×3×3 共 27 个 dtype/reduction binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 单 POD 传 N/C、threads、两组 core 分片、rank/mode/ignore | 字段声明顺序与位宽、`usedThread` 64→32 位消费、inactive height/width 不可擅自读取 | `NLLLossGradSimtTilingData`、`KernelNLLLossGrad::Init` |
| 调用顺序 | ACLNN 前处理→核心→后处理；核心清零→标量预载→SIMT 写回 | 一维 self 必须先变 `[1,C]`；清零必须先于稀疏 store | `aclnnNLLLossBackwardGetWorkspaceSize`、`Process` |
| workspace/内存 | 核心 workspace 16 MiB；local memory 为 UB-32 KiB；3 个 FP32 TBuf | entry 需要 system/user workspace 非空；物理 UB 对齐不可按 12 B 假定 | `GetWorkspaceSize/PostTiling/nll_loss_grad/Process` |
| 同步 | `SyncAll` 保护清零和 thread0 标量；`PipeBarrier` 保护 tmpOut | 去掉 ordering 会造成零覆盖目标值或 UB 可见性问题 | `ProcessMemset/Process` |
| 精度与尾块 | FP16/BF16/FP32 均转 FP32 做乘/除，最后 cast T；清零 former/tail，SIMT 用 `i<N` | cast/取负顺序、ignore 优先、target 地址公式不可变 | 三个 `SimtCompute*2d`、`TargetCheck` |
| 失败边界 | ACLNN 参数错误/空域、Tiling rank/dtype/reduction 错误、workspace guard、target ASSERT | 不得把 A2/A3 缺失实现或 mean 零分母保护推测补齐 | `CheckParams/ProcessShapeInfo/GenerateTilingKey/nll_loss_grad/TargetCheck` |
