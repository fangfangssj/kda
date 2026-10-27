---
type: Operator
title: "NLLLoss2dBackward 950 ACLNN 算子知识"
description: "说明 aclnnNLLLoss2dBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/docs/aclnnNLLLoss2dBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_api/aclnn_nll_loss2d_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_api/aclnn_nll_loss2d_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_api/nll_loss_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/nll_loss_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_host/config/ascend950/nll_loss_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T17:15:57Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# NLLLoss2dBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnNLLLoss2dBackwardGetWorkspaceSize` / `aclnnNLLLoss2dBackward`，固定仓未出现更高版本后缀 |
| 计算语义 | 对 `self[N,C,H,W]` 求 NLLLoss 的输入梯度；每个 `(n,h,w)` 只在 `target[n,h,w]` 指定的类别位置写负梯度，其余类别为 0，`ignoreIndex` 位置全为 0 |
| 输入输出 | `self/gradOutput/weight/totalWeight/out` 为 FLOAT/FLOAT16/BFLOAT16 且同 dtype；`target` 为 INT32/INT64/UINT8；ND；`out.shape=self.shape` |
| 执行类型 | 算子自有核心为单 Kernel `SIMT`：AIV block 内用 `asc_vf_call` 启动最多 1024 个 SIMT 线程；无 Cube/AIC 或 `MIX_CV` 节点 |
| Kernel 路线 | R-001 `none`，R-002 `mean`，R-003 `sum`；27 个 binary descriptor 归入三条 reduction Route，运行时 TilingKey 均为 0 |
| 核心资源 | GM 输入/输出；每核三个 4 B `TBuf<VECCALC>`；SIMT 线程；Tiling 为核心节点请求 16 MiB workspace，并把 local memory 设为平台 UB−32 KiB |
| 来源闭包 | `source_closed`：公开 ACLNN→950 L0 AICore selector→arch35 Host/Tiling/POD→单 SIMT entry/body→ascend950 binary/CMake 已闭合；同 dtype `Cast` 调用、`Contiguous/ViewCopy` 与 executor 聚合是明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `p=n*H*W+h*W+w`、`t=target[p]`，输出先整体清零。非忽略位置的唯一非零候选为：

```text
reduction = none: out[n,t,h,w] = -float(gradOutput[p]) * float(weight[t])
reduction = mean: out[n,t,h,w] = -(float(gradOutput[0]) / float(totalWeight[0])) * float(weight[t])
reduction = sum : out[n,t,h,w] = -float(gradOutput[0]) * float(weight[t])
target[p] == ignoreIndex: 本位置所有 C 个输出保持 0
```

乘法或除法在源码中显式转为 FP32，最后 `static_cast<T>` 写回 `self` dtype。`self` 的数值从不参与计算：它只提供输出 shape、dtype、format 和类别数。非忽略 target 在 Device 由 `TargetCheck` 断言满足 `0 <= t < C`；ACLNN Host 只文档化该范围，不扫描 target 值。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | reduction=none 时 `[N,H,W]`；否则 rank≤1 且 numel=1；ND，可非连续 | FLOAT/FLOAT16/BFLOAT16，与 `self` 相同 | 上游 loss 梯度；none 逐位置读取，mean/sum 只读元素 0 |
| `self` | `[N,C,H,W]`，ND，可非连续 | FLOAT/FLOAT16/BFLOAT16 | 950 `DAV_3510` 支持三种 dtype；值不读，仅 shape/dtype/format 决定结果描述与 Kernel 模板 |
| `target` | `[N,H,W]`，ND，可非连续 | INT32/INT64/UINT8 | 每项应在 `[0,C-1]` 或等于 `ignoreIndex` |
| `weight` | numel=`C`，ND，可非连续 | 与 `self` 相同 | 类别缩放因子，Kernel 按 `target` 间接索引 |
| `reduction` | Host 标量 | INT64 | 0/1/2 分别映射字符串 `none/mean/sum`；越界返回 `ACLNN_ERR_PARAM_INVALID` |
| `ignoreIndex` | Host 标量 | INT64 | 命中时跳过稀疏写，依赖预清零保持 0 |
| `totalWeight` | numel=1，ND | 与 `self` 相同 | mean 路线作除数；文档称其它 reduction 不处理，但当前 Kernel 仍无条件加载并计算一次未消费的商 |
| `out` | 与 `self` 完全同 view shape；ND，可非连续 | 与 `self` 相同 | 用户输出，由连续核心结果经 `ViewCopy` 提交 |
| `workspaceSize` | Host 输出标量 | `uint64_t` | executor 聚合整图需求；核心 NLLLossGrad 节点明确请求 16 MiB |
| `executor` | Host 输出句柄 | `aclOpExecutor**` | 持有框架节点、中间 Tensor 和启动顺序，第一段转交给调用者 |

### 入口与版本

固定 Commit 公开的最新入口是无版本后缀的两段式 ACLNN。公开产品表明确支持 Ascend 950PR/950DT，也同时列出 A2/A3；本卡仅覆盖 950。第一段建图并返回 workspace 与 executor，第二段调用 `CommonOpExecutorRun(workspace,workspaceSize,executor,stream)`。

950 由 `IsRegbase()` 对 `DAV_3510` 返回 true：API 不执行 A2/A3 的 NHWC 转置/二维 reshape，直接把四维 `self` 交给 `NLLLossGrad`。`GetAiCoreDtypeSupportListBySocVersion()` 对 `DAV_3510` 返回 FLOAT/BFLOAT16/FLOAT16，故本接口允许的三种 `self` dtype 均选 `NLLLossGradAiCore`，不会触发核心 AICPU fallback。Grad 与前向、普通 NLLLossBackward 均是独立语义，本卡不合并。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckNotNull` 依次拒绝空 `gradOutput/self/target/weight/out/totalWeight`。
2. `CheckDtypeValid` 在 950 接受 `self` 的 FLOAT/FLOAT16/BFLOAT16，要求 `gradOutput/weight/out/totalWeight` 与 `self` 同 dtype，并将 `target` 限为 INT64/UINT8/INT32。
3. `CheckShape`：none 要求 `gradOutput.viewShape==target.viewShape`；mean/sum 要求 `gradOutput.rank<=1 && numel==1`；`self.rank=4`、`target.rank=3`、`self[0,2,3]==target[0,1,2]`、`weight.numel=C`、`out.shape=self.shape`、`totalWeight.numel=1`。
4. `CheckReduction` 最后检查 `0<=reduction<=2`。注意 shape 检查在 reduction 范围检查之前；非法 reduction 会先按“非 none”规则检查 gradOutput。
5. 创建 executor；若 `self->IsEmpty()`，直接返回 `workspaceSize=0`，不添加核心或写回节点，固定源码未定义调用者 `out` 既有内容如何处理。
6. 950 令 `promoteType=self.dtype`。按顺序请求 `Contiguous(gradOutput)`→同 dtype `Cast`、`Contiguous(self)`→同 dtype `Cast`、`Contiguous(target)`、`Contiguous(weight)`→同 dtype `Cast`、`totalWeight` 同 dtype `Cast`。
7. RegBase 分支请求 target 到其自身 dtype 的 `Cast`，将上述 Tensor、`GetReductionStr(reduction)` 与 `ignoreIndex` 送入 `l0op::NLLLossGrad`。
8. 核心结果请求到 `out.dtype` 的同 dtype `Cast`，再 `ViewCopy` 到可能非连续的用户 `out`。这些通用 helper 在固定 `ops-nn` 仓只暴露声明/测试 stub；本卡不外推其生产实现、物理 workspace 或是否物化节点，只闭合其顺序、dtype 恒等关系和失败短路。

每个 helper 返回空指针都立即以 `ACLNN_ERR_INNER_NULLPTR` 停止建图；executor 创建失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。

### 参数、Workspace 与生命周期

`l0op::NLLLossGrad` 先按 `self` 的 view shape、dtype、storage format 由 executor 分配连续中间 `gradInput`，再提交：

```text
ADD_TO_LAUNCHER_LIST_AICORE(
  NLLLossGrad,
  OP_INPUT(self, gradOutput, target, weight, totalWeight),
  OP_OUTPUT(gradInput),
  OP_ATTR(reductionString, ignoreIndex))
```

核心 Tiling 的 `GetWorkspaceSize` 固定设置 `workspace[0]=16*1024*1024` B；entry 要求总 workspace 非空并通过 `SetSysWorkspace/GetUserWorkspace` 得到 user 区，但 body 不读写 `userWS`。ACLNN 返回值是 `uniqueExecutor->GetWorkspaceSize()` 的整图聚合值，通用连续化/写回与 allocator 的 slice offset、alignment、复用策略固定仓不可见。executor 持有所有中间 Tensor，从第一段 `ReleaseTo(executor)` 起持续到第二段执行完成；用户持有输入、输出、workspace 和 stream。

### 启动图与失败传播

非空 950 核心 launch graph：

```text
ACLNN GetWorkspaceSize
  -> F0..Fn Contiguous / 同 dtype Cast                 # 通用框架边界
  -> K0 NLLLossGradAiCore
       entry: nll_loss_grad(x,y_grad,target,weight,total_weight,x_grad,workspace,tiling)
       blockDim = max(memsetCoreNum, computeCoreNum)
       threadsPerBlock = usedThread <= 1024
       localMemorySize = platformUB - 32768 B
       user workspace request = 16777216 B
  -> Fm 同 dtype Cast / ViewCopy(gradInput,out)        # 通用框架边界
  -> CommonOpExecutorRun(...,stream)
```

K0 是唯一算子自有 Device 节点。框架中间 Tensor 是 GM 边，producer/consumer 由 executor 同一 stream 的任务顺序保证；具体物理 offset 和 event 实现不可见。K0 launcher 添加失败返回空并阻断后续写回；Tiling、entry 的 workspace/null 检查或 Device 失败由 executor/stream 返回路径传播，固定仓未展开清理细节。

## Route 与参数设计

### 分核与资源计算

对本 2d backward，`x=self` 恒为 rank 4：

```text
N=batchNum, C=classNum, H=height, W=width
O=N*C*H*W                         # 输出元素
P=N*H*W                           # target/位置数
A=平台 AIV core 数
T=min(platform maxThread, 1024)   # usedThread，线程/块
M=min(O,A)                        # notVeryImportantProcessCoreNums，清零核数
q=floor(O/M)                      # blockPerCore，元素/核
r=O-q*M                           # blockTailCore，前 r 核多 1 元素
K=min(ceil(P/T),A)                # veryImportantProcessCoreNums，稀疏写核数
blockDim=max(M,K)
localMemorySize=platformUB-32768 B
```

空 `self` 已在 ACLNN 前置返回，因此进入 Tiling 时 `O>0`、`M>0`。清零核 `b` 的范围为：`b<r` 时 `offset=b*q+b,count=q+1`；否则 `offset=b*q+r,count=q`。计算线程的扁平位置为 `i=b*T+threadIdx.x+k*blockDim*T`，直到 `i>=P`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 none | binary attr `none`；TilingKey=0；`KernelNLLLossGrad<DTYPE_X,DTYPE_TARGET>` | 950、非空、公开 dtype/shape 检查通过、`reduction==0`；binary config 对 3×3 个 `T/F` 组合分别接线 | reduction=1 去 R-002；=2 去 R-003；其它 Host 报错 |
| 2 | R-002 mean | binary attr `mean`；TilingKey=0；同模板 | 同上且 `reduction==1` | reduction=2 去 R-003；其它 Host 报错 |
| 3 | R-003 sum | binary attr `sum`；TilingKey=0；同模板 | 同上且 `reduction==2` | 其它 Host 报错 |

```text
if self.empty: 返回 workspace=0，无 Route
else if platform arch != DAV_3510: 不属于本卡
else if reduction == 0: R-001
else if reduction == 1: R-002
else if reduction == 2: R-003
else: ACLNN_ERR_PARAM_INVALID

每条 Route 内：binary config 由 (T∈{bf16,float32,float16},
F∈{int32,int64,uint8}, reduction) 选择 27 个专用 binary 之一；TilingKey 最终恒为 0。
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 none | attr=none，key=0 | 950；`self` T=FP32/BF16/FP16；target F=I32/I64/U8；四维非空；grad shape=target | 单 K0 SIMT | `KernelNLLLossGrad<T,F>::Process`→`SimtComputeNoReduce4d<T,F>` | 全输出先并行清零；3×4 B TBuf；`SyncAll`、`PipeBarrier`；逐位置读取 grad | 线程 grid-stride tail；ignore 留零；target 越界 ASSERT；无 fallback |
| R-002 mean | attr=mean，key=0 | 同上；grad numel=1 | 单 K0 SIMT | 同 Process→`SimtComputeReduceMean4d<T,F>` | 每核 thread0 加载 scalar grad/totalWeight，FP32 除法；随后稀疏写 | grid-stride tail；totalWeight 应非零；无 fallback |
| R-003 sum | attr=sum，key=0 | 同上；grad numel=1 | 单 K0 SIMT | 同 Process→`SimtComputeReduceSum4d<T,F>` | 每核 thread0 加载 scalar grad；仍分配并计算未使用 mean 商 | grid-stride tail；ignore 留零；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 非空前置 | `self->IsEmpty()` | Tiling 的 `M=min(O,A)` 后执行 `FloorDiv(O,M)`，必须避免 M=0 | 全部 | ACLNN 返回 workspace=0，不启动 | `aclnnNLLLoss2dBackwardGetWorkspaceSize` |
| 950 AICore dtype | arch=`DAV_3510` 且 T∈{FP32,BF16,FP16} | 固定仓只为 ascend950 构建 SIMT Kernel | 全部 | L0 其它平台可能 AICPU/缺失实现，不属于本卡 | `GetAiCoreDtypeSupportListBySocVersion` |
| 线程上限 | `T=min(compile_info.max_thread,1024)` | `LAUNCH_BOUND(1024)` 与 `asc_vf_call(dim3{usedThread})` | 全部 | 无备用 Route；平台 producer 必须给正值 | `PostTiling`、`THREAD_DIM` |
| DCache 预留 | `localMemorySize=UB-32768` | 为 SIMT local memory 从平台 UB 扣除固定 32 KiB | 全部 | UB≤32 KiB 时无 Host fallback，固定源码未加显式下溢 guard | `NLLLossGradSimtTiling::PostTiling` |
| target 范围 | `target==ignoreIdx \|\| 0<=target<C` | 防止 weight/output GM 越界 | 全部 | 非 ignore 越界触发 Device `ASSERT` | `TargetCheck` |
| mean 除数 | 语义要求 `totalWeight[0]!=0` | mean 先做 FP32 除法 | R-002 | 固定 ACLNN 不校验；Device 源码含非零 ASSERT 文本，但其标识符与函数形参不一致，见限制 | `SimtComputeReduceMean4d` |
| workspace 指针 | entry 中 `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | entry 先设置系统 workspace；虽 body 不用 user 区，仍硬检查 | 全部 | 直接 return，无输出 | `nll_loss_grad` |

### POD/ABI

宏 `BEGIN_TILING_DATA_DEF/END_TILING_DATA_DEF` 生成序列化结构；字段声明顺序和源码位宽可见，但宏展开后的精确 `sizeof/alignof/offsetof` 不在固定仓中，故不推测 padding。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `NLLLossGradSimtTilingData` | `batchNum` | uint64_t/64 | 第1字段；精确 offset 固定仓不可见 | batch | rank4: `x.shape[0]` | `Init`→`productOfNHW`、函数参数 | >0（非空路径） |
| 同上 | `classNum` | uint64_t/64 | 第2字段 | 类别 | `x.shape[1]` | target 检查、weight/output 地址 | >0（非空路径） |
| 同上 | `usedThread` | uint64_t/64 | 第3字段 | 线程/块 | `min(maxThread,1024)` | 所有 `asc_vf_call(dim3{usedThread_})` | ≤1024；正值由平台 producer 前提保证 |
| 同上 | `height` | uint64_t/64 | 第4字段 | 元素维 | rank4 `x.shape[2]` | `productOfNHW/HW`、4d helper | Tiling 显式要求 >0 |
| 同上 | `width` | uint64_t/64 | 第5字段 | 元素维 | rank4 `x.shape[3]` | 同上 | Tiling 显式要求 >0 |
| 同上 | `veryImportantProcessCoreNums` | uint64_t/64 | 第6字段 | core | `min(ceil(NHW/usedThread),coreNum)` | `isComputeCore_` | 计算核数 |
| 同上 | `notVeryImportantProcessCoreNums` | uint64_t/64 | 第7字段 | core | `min(O,aivCoreNum)` | `isMemsetCore_` | 清零核数 |
| 同上 | `blockPerCore` | uint64_t/64 | 第8字段 | 输出元素/core | `floor(O,memsetCoreNum)` | `ProcessMemset` | q |
| 同上 | `blockTailCore` | uint64_t/64 | 第9字段 | core 数 | `O-q*memsetCoreNum` | `ProcessMemset` | `[0,memsetCoreNum)` |
| 同上 | `xDims` | uint32_t/32 | 第10字段 | rank | `x.GetDimNum()` | 选择 1/2d 或 4d helper | 本 API 恒 4 |
| 同上 | `reductionMode` | uint32_t/32 | 第11字段 | enum | none/mean/sum→0/1/2 | 三路 helper 分支 | 其它值 Tiling 失败 |
| 同上 | `ignoreIdx` | int64_t/64 | 第12字段 | label 值 | attrs[1] | 每线程 ignore 比较 | 可在类别范围外 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 none | attr none；key 0 | `nll_loss_grad.cpp::NLLLossGrad`；`nll_loss_grad_tiling_arch35.cpp::NLLLossGradTiling` | `NLLLossGradSimtTilingData`，mode=0 | `op_kernel/nll_loss_grad.cpp::nll_loss_grad`，`KernelNLLLossGrad<DTYPE_X,DTYPE_TARGET>` | `ProcessMemset`、`ComputeSetValue`、`SimtComputeNoReduce4d` | ascend950 binary JSON 的 9 个 none 条目；两级 CMake |
| R-002 mean | attr mean；key 0 | 同上 | 同 POD，mode=1 | 同 entry/template | `ProcessMemset`、`ComputeSetValue`、`SimtComputeReduceMean4d` | JSON 的 9 个 mean 条目；两级 CMake |
| R-003 sum | attr sum；key 0 | 同上 | 同 POD，mode=2 | 同 entry/template | `ProcessMemset`、`ComputeSetValue`、`SimtComputeReduceSum4d` | JSON 的 9 个 sum 条目；两级 CMake |

## Kernel 实现

### 公共执行流程

所有 Route 都是单 Kernel。entry 先拒绝空 workspace，设置系统 workspace，再拒绝空 user workspace；解析唯一 POD 后声明 `KERNEL_TYPE_MIX_AIV_1_0`（1 个 AIV、0 个 AIC），直接实例化 `KernelNLLLossGrad<DTYPE_X,DTYPE_TARGET>`，无 runtime key switch。

`Init` 将六个 GM 指针绑定为 `inputXGm_/yGrad_/target_/weight_/total_weight_/outputGm_`；`inputXGm_` 后续不读。`Process` 先由全部清零核分段 `InitOutput`，经 `SyncAll` 后每个计算核用 thread0 把 scalar grad 与 totalWeight 转成 FP32 放入本核 UB。第二次 `SyncAll` 后，非计算核退出；计算核无条件形成 `tmpOut=singleGrad/localTotalWeight`，`PipeBarrier<PIPE_ALL>` 后按 reduction 进入对应 4d SIMT helper。

### Route R-001：none 逐位置梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_nll_loss2d_backward.cpp` | `GetReductionStr` | `reduction==0` | attr=`"none"` |
| L0 | `op_api/nll_loss_grad.cpp` | `NLLLossGrad` | arch DAV_3510 的 T 在 AICore list | `NLLLossGradAiCore` |
| Tiling | `op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo` | `strcmp(reduction,"none")==0` | `reductionMode=0` |
| Router | `op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | binary 已按 T/F 特化；key 固定0 | `KernelNLLLossGrad<T,F>` |
| Body | `op_kernel/arch35/nll_loss_grad.h` | `Process` | `xDims==4 && reductionMode==NONE_MODE` | `SimtComputeNoReduce4d` |

#### Device 数据流

##### 函数级调用链

```text
NLLLossGradAiCore -> nll_loss_grad(...)
  -> KernelNLLLossGrad<T,F>::Init
  -> KernelNLLLossGrad<T,F>::Process
    -> ProcessMemset -> InitOutput(out slice,0) -> SyncAll
    -> InitBuffer(3 × float scalar)
    -> asc_vf_call<ComputeSetValue<T,F>>(T threads) -> SyncAll
    -> tmpOut=singleGrad/localTotalWeight -> PipeBarrier<PIPE_ALL>  # 结果未消费
    -> asc_vf_call<SimtComputeNoReduce4d<T,F>>(T threads)
      -> target load -> ignore/range check -> grad/weight FP32 multiply -> selected GM store
  -> executor framework edge -> ViewCopy(user out)
```

##### 源码等价伪码

```text
# 伪码；b=blockIdx，t=threadIdx，B=blockDim，T=usedThread
parallel each block b:
  zero_count  = q + (b < r)
  zero_offset = b*q + min(b,r)
  InitOutput(out[zero_offset : zero_offset+zero_count], 0)
  SyncAll()

  UB.singleGrad = float(y_grad[0])       # 仅 thread 0 写；none 路线实际不用
  UB.totalWeight = float(total_weight[0])# 仅 thread 0 写；none 路线实际不用
  SyncAll()
  if b >= computeCoreNum: return
  UB.tmpOut = UB.singleGrad / UB.totalWeight  # 未消费
  PipeBarrier(ALL)

  for i = b*T+t; i < N*H*W; i += B*T:
    label = target[i]
    if label == ignoreIndex: continue
    assert 0 <= label < C
    n = i / (H*W); rem = i % (H*W)
    out[n*C*H*W + label*H*W + rem] = T(-float(y_grad[i])*float(weight[label]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `target` | `i` | P elements | F, `[N,H,W]` contiguous core input | framework Contiguous | SIMT thread | executor/input，K0期间 | 逐元素 load |
| GM `y_grad` | `i` | P elements | T, `[N,H,W]` | framework | SIMT thread | executor/input | 逐元素 load；元素0另被无效 scalar 路径读 |
| GM `weight` | `label` | C elements | T, `[C]` | caller/framework | SIMT thread | input | 间接随机 load |
| GM `out` | `n*C*H*W+label*H*W+rem` | O elements | T, `[N,C,H,W]` | InitOutput + SIMT | ViewCopy | executor 中间到 K0结束 | 清零与稀疏写分阶段 |
| `localTotalWeightBuf_` | 本核 UB | 4 B 请求 | FP32 scalar | ComputeSetValue | 未消费商的除法 | K0每核 Process | 独立 TBuf |
| `singleGradBuf_` | 本核 UB | 4 B 请求 | FP32 scalar | ComputeSetValue | 未消费商的除法 | 同上 | 独立 TBuf |
| `tmpOutBuf_` | 本核 UB | 4 B 请求 | FP32 scalar | 标量除法 | none 不读 | 同上 | 独立 TBuf |
| user workspace | `GetUserWorkspace(workspace)` | 16 MiB 请求 | bytes | runtime | body不读 | executor/K0 | entry只检查非空 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  tw["localTotalWeightBuf · TBuf VECCALC · 4 B"]:1
  sg["singleGradBuf · TBuf VECCALC · 4 B"]:1
  to["tmpOutBuf · TBuf VECCALC · 4 B"]:1
```

##### 字节账与复用顺序

```text
UB_requested_by_body = 4 + 4 + 4 = 12 B/core
UB physical alignment = 固定仓不可见的 TPipe 实现
localMemorySize = platformUB - 32768 B
dynamic_ub = localMemorySize（SIMT launch local memory 上限）
workspace_total(core node) = 16777216 B
=> 12 B 请求必须落在扣除 DCache 后的 local memory；Host 未写额外容量 fallback
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 三个 TBuf | 各自独立，无别名 | scalar setup/未消费商 | 本 Kernel 无下一次覆盖 | `SyncAll`、`PipeBarrier` 后只读；Process结束释放 | 不得把三个 4 B 区误合并；源码无 alias 证据 |
| GM out | `InitOutput` 清零 | 后续稀疏 store / ViewCopy | selected 元素由 SIMT 覆盖 | 第一个 `SyncAll` 在稀疏写前 | blockDim 必须等于参与清零同步的核集合；正 shape 下 K≤M |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输出初始化 | `InitOutput` + `SyncAll` | 每个清零 core | 后续稀疏写 | 完整 O 元素 | q/r 分段覆盖全部输出，无重叠 |
| scalar 装载 | thread0 + `SyncAll` | `ComputeSetValue` | 标量除法 | 本核三个 UB scalar | T→FP32；none 结果不消费 |
| 标量到 SIMT | `PipeBarrier<PIPE_ALL>` | `tmpOut` 写 | SIMT helper | `tmpOut` | none 不读该值 |
| 稀疏计算 | grid-stride SIMT | 每线程 | GM store | 唯一 `(n,label,h,w)` | FP32乘后 cast T；`i<P` 是 tail guard；ignore跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `DoOpTiling`、`PostTiling` | q/r 清零分核、P/T 计算核和 blockDim |
| Kernel entry/router | `op_kernel/nll_loss_grad.cpp` | `nll_loss_grad`、`COMMON_OP_IMPL` | key无分支；T/F由 binary宏特化 |
| Device body/helper | `op_kernel/arch35/nll_loss_grad.h` | `ProcessMemset`、`SimtComputeNoReduce4d<T,F>` | 全零后按 label 稀疏写，线程 grid-stride tail |

### Route R-002：mean 标量归一化梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_nll_loss2d_backward.cpp` | `GetReductionStr` | `reduction==1` | attr=`"mean"` |
| L0 | `op_api/nll_loss_grad.cpp` | `NLLLossGrad` | DAV_3510，T在三 dtype AICore list | `NLLLossGradAiCore` |
| Tiling | `op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo` | attr mean | `reductionMode=1` |
| Router | `op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | binary T/F/mean descriptor | `KernelNLLLossGrad<T,F>` |
| Body | `op_kernel/arch35/nll_loss_grad.h` | `Process` | `xDims==4 && mode==MEAN_MODE` | `SimtComputeReduceMean4d` |

#### Device 数据流

##### 函数级调用链

```text
NLLLossGradAiCore -> nll_loss_grad -> KernelNLLLossGrad<T,F>::Init/Process
  -> ProcessMemset -> SyncAll
  -> InitBuffer(three FP32 scalars)
  -> ComputeSetValue: y_grad[0],total_weight[0] -> UB FP32 -> SyncAll
  -> tmpOut=singleGrad/localTotalWeight -> PipeBarrier
  -> SimtComputeReduceMean4d<T,F>
     -> target/ignore/range -> tmpOut*weight[label] -> selected out
  -> framework ViewCopy
```

##### 源码等价伪码

```text
# 伪码；清零分区同 R-001
zero_full_output(); SyncAll()
thread0: singleGrad=float(y_grad[0]); total=float(total_weight[0])
SyncAll()
if b >= K: return
tmpOut = singleGrad / total
PipeBarrier(ALL)
for i=b*T+t; i<P; i+=B*T:
  label=target[i]
  if label==ignoreIndex: continue
  assert 0<=label<C
  n=i/(H*W); rem=i%(H*W)
  out[n*C*H*W+label*H*W+rem] = T(-tmpOut*float(weight[label]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `y_grad` | 0 | 1 element | T scalar | caller/framework | ComputeSetValue | input | 每个计算核各加载一次 |
| GM `total_weight` | 0 | 1 element | T scalar | caller | ComputeSetValue | input | 每个计算核各加载一次 |
| GM `target/weight/out` | 同 R-001 | P/C/O | F/T/T | framework/清零 | mean SIMT | executor | selected output唯一写 |
| `localTotalWeightBuf_` | 本核 UB | 4 B | FP32 | ComputeSetValue | division | Process | 独立 |
| `singleGradBuf_` | 本核 UB | 4 B | FP32 | ComputeSetValue | division | Process | 独立 |
| `tmpOutBuf_` | 本核 UB | 4 B | FP32 | division | all SIMT threads in block | Process | block内广播复用 |
| user workspace | runtime user区 | 16 MiB | bytes | runtime | body不读 | K0 | 只检查 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  tw["totalWeight FP32 · 4 B"]:1
  sg["grad FP32 · 4 B"]:1
  to["grad/totalWeight FP32 · 4 B"]:1
```

##### 字节账与复用顺序

```text
UB_requested_by_body = 12 B/core
dynamic_ub = platformUB - 32768 B
workspace_total(core node) = 16777216 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `tmpOutBuf_[0]` | AIV scalar division | 同 block 的 mean SIMT threads | 无 | `PipeBarrier<PIPE_ALL>` | totalWeight非零且 barrier在SIMT调用前 |
| GM out | 清零 cores | mean SIMT | selected位置覆盖 | `SyncAll` | 每个 i 唯一映射，线程间无 atomic需求 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零 | InitOutput/SyncAll | M cores | K compute cores | GM out | q/r tail |
| scalar广播 | thread0/SyncAll | ComputeSetValue | AIV标量除法 | 两个 UB scalar | T→FP32 |
| 商发布 | PipeBarrier ALL | AIV scalar store | mean SIMT | tmpOut | FP32 divide；源码意图断言非零 |
| 稀疏写 | grid-stride | SIMT threads | GM | selected outputs | FP32乘，cast T；i<P；ignore保持零 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo`、`PostTiling` | mean→1，thread/core参数 |
| Kernel entry/router | `op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | 单模板启动 |
| Device body/helper | `op_kernel/arch35/nll_loss_grad.h` | `ComputeSetValue`、`SimtComputeReduceMean4d<T,F>` | 每核一次FP32归一化并在线程间复用 |

### Route R-003：sum 标量梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_nll_loss2d_backward.cpp` | `GetReductionStr` | `reduction==2` | attr=`"sum"` |
| L0 | `op_api/nll_loss_grad.cpp` | `NLLLossGrad` | DAV_3510 AICore dtype支持 | AICore launcher |
| Tiling | `op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo` | attr sum | `reductionMode=2` |
| Router | `op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | T/F/sum binary | `KernelNLLLossGrad<T,F>` |
| Body | `op_kernel/arch35/nll_loss_grad.h` | `Process` | `xDims==4 && mode==SUM_MODE` | `SimtComputeReduceSum4d` |

#### Device 数据流

##### 函数级调用链

```text
NLLLossGradAiCore -> nll_loss_grad -> KernelNLLLossGrad<T,F>::Init/Process
  -> ProcessMemset -> SyncAll
  -> ComputeSetValue(y_grad[0],total_weight[0]) -> SyncAll
  -> unused tmpOut division -> PipeBarrier
  -> SimtComputeReduceSum4d<T,F>
     -> target/ignore/range -> singleGrad*weight[label] -> selected out
  -> framework ViewCopy
```

##### 源码等价伪码

```text
# 伪码
zero_full_output(); SyncAll()
thread0: singleGrad=float(y_grad[0]); total=float(total_weight[0])
SyncAll()
if b>=K: return
tmpOut=singleGrad/total                 # 计算但 sum helper不读取
PipeBarrier(ALL)
for i=b*T+t; i<P; i+=B*T:
  label=target[i]
  if label==ignoreIndex: continue
  assert 0<=label<C
  out[address4d(i,label)] = T(-singleGrad*float(weight[label]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM scalar grad | 0 | 1×sizeof(T) | T | caller | ComputeSetValue | input | 每计算核重复载入 |
| GM target/weight/out | 同 R-001 | P/C/O | F/T/T | caller/清零 | sum SIMT | input/executor | label间接读、selected写 |
| three scalar TBuf | 本核 UB | 各4 B | FP32 | setup/division | sum只消费 singleGrad | Process | total/tmpOut虽无效仍分配 |
| user workspace | user区 | 16 MiB | bytes | runtime | body不读 | K0 | entry非空guard |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  tw["totalWeight FP32 · 4 B · 未用于结果"]:1
  sg["singleGrad FP32 · 4 B"]:1
  to["tmpOut FP32 · 4 B · 未用于结果"]:1
```

##### 字节账与复用顺序

```text
UB_requested_by_body = 12 B/core（其中8 B不影响sum结果）
dynamic_ub = platformUB - 32768 B
workspace_total(core node) = 16777216 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `singleGradBuf_[0]` | ComputeSetValue thread0 | sum SIMT threads | 无 | SyncAll，随后 PipeBarrier | T→FP32一次后只读 |
| `localTotalWeight/tmpOut` | setup/division | sum结果不消费 | 无 | Process结束释放 | 不得据此删减ABI或源码分配；迁移实现可另行特化 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零 | InitOutput/SyncAll | M cores | sum SIMT | GM out | q/r覆盖O |
| scalar | thread0/SyncAll | grad load | SIMT | singleGrad | T→FP32 |
| 未消费商 | PipeBarrier | scalar divide | 后续SIMT顺序 | tmpOut | 即使 totalWeight不活跃仍执行除法 |
| sum写 | grid-stride | threads | GM out | selected位置 | FP32乘后cast；i<P；ignore跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `ProcessAttributesInfo` | sum→2 |
| Kernel entry/router | `op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | 单SIMT模板 |
| Device body/helper | `op_kernel/arch35/nll_loss_grad.h` | `ComputeSetValue`、`SimtComputeReduceSum4d<T,F>` | scalar grad复用，full-zero后稀疏写 |

## 算子特定优化

### Trick T-001：扁平 NHW 线程映射配合全零后单类别稀疏写

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：950 2d backward 的 `xDims==4`，每个 `target[i]` 唯一决定一个类别输出。
- Golden 来源：[`op_kernel/arch35/nll_loss_grad.h:178-198`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h#L178-L198)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void SimtComputeNoReduce4d(uint64_t classNum_, uint64_t batchNum_, uint64_t height_, uint64_t width_,
                                                                                __gm__ F* target_, int64_t ignoreIdx_,
                                                                                __gm__ T* outputGm_, __gm__ T* yGrad_,
                                                                                __gm__ T* weight_, uint32_t blockId_,
                                                                                uint32_t blockNums_, uint64_t productOfNHW_, uint64_t productOfHW_)
{
    int64_t localClassNum = classNum_;
    for (uint64_t i = blockId_ * blockDim.x + threadIdx.x; i < productOfNHW_;
         i = i + blockNums_ * blockDim.x) {
        int64_t chosenClass = target_[i];
        if (chosenClass == ignoreIdx_) {
            continue;
        }
        auto n = i / productOfHW_;
        auto rem = i % productOfHW_;
        TargetCheck(chosenClass, localClassNum);

        outputGm_[n * localClassNum * productOfHW_ + chosenClass * productOfHW_ + rem] =
            -static_cast<T>(static_cast<float>(yGrad_[i]) * static_cast<float>(weight_[chosenClass]));
```

#### 机制与设计原因

输出梯度在类别轴上是一热稀疏的。实现不为每个 `N*C*H*W` 元素做 label 比较，而先用 `ProcessMemset` 线性清零完整输出，再把 `N*H*W` 个空间位置展平给 `usedThread` 个线程做 grid-stride 循环。每线程只加载一个 target、一个 grad 和被选中的一个 weight，并只覆盖一个类别位置；ignore 直接跳过，依赖清零阶段得到正确 0。`blockId*T+threadIdx` 与步长 `blockNums*T` 都来自 POD 的 core/thread producer；4d 地址用 `n=i/(H*W)` 与 `rem=i%(H*W)` 恢复 NCHW。

#### 迁移边界

- 适用：每个外部位置只激活一个类别槽的 one-hot/scatter 型梯度。
- 前提：能在稀疏写前可靠清零完整输出；每个位置只有一个线程写；label 范围受保护；输出布局是 NCHW。
- 失效：若一个位置可对应多个类别、需要累加冲突、输出不能预清零或 layout 不同，必须改变地址与同步方案，不能直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若移除 `ProcessMemset` 或让稀疏写早于全局清零完成，未选类别和 ignore 位置会保留旧 GM 内容；固定源码以 `InitOutput` 后 `SyncAll` 给出了可验证反例边界。

### Trick T-002：每核一次 FP32 scalar 归一化并在 1024 个线程间复用

#### 触发与源码依据

- 适用 Route：R-002；R-003复用 scalar grad 但不消费归一化商。
- 触发条件：reduction=mean，`gradOutput` 与 `totalWeight` 都只有一个元素。
- Golden 来源：[`op_kernel/arch35/nll_loss_grad.h:243-251`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h#L243-L251)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T, typename F>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void ComputeSetValue(
    uint64_t classNum_, uint64_t batchNum_, uint32_t blockId_, uint32_t blockNums_, __ubuf__ float* singleGrad_,
    __gm__ T* outputGm_, __gm__ T* yGrad_, __ubuf__ float* localTotalWeight_, __gm__ T* total_weight_)
{
    if (threadIdx.x == 0) {
        localTotalWeight_[0] = static_cast<float>(total_weight_[0]);
        singleGrad_[0] = static_cast<float>(yGrad_[0]);
    }
}
```

#### 机制与设计原因

每个计算 block 只让 thread0 从 GM 加载两个 scalar 并升到 FP32，`SyncAll` 后 AIV 只做一次除法写 `tmpOut`，再通过 `PipeBarrier` 供该 block 的所有 SIMT 线程读取。相比每个空间线程重复读取 `gradOutput[0]/totalWeight[0]` 并做除法，这改变了 GM scalar 搬运次数和除法次数；源码只证明机制，不给出实测收益。

#### 迁移边界

- 适用：block 内所有线程共享相同 scalar 缩放的逐元素/稀疏写计算。
- 前提：每个 block 有独立 UB scalar，thread0写后有可见性同步，除数非零，FP32中间顺序可接受。
- 失效：scalar 随线程变化、跨 block 共享写入、没有 block 内同步或要求不同精度顺序时不可直接迁移。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

在 `SyncAll`/`PipeBarrier` 前让其它线程读取 UB scalar 会破坏 producer→consumer 可见性；固定源码未提供该错误写法的独立实现。

## 模板与实现对比

| 对比维度 | R-001 none | R-002 mean | R-003 sum | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | reduction=0 | reduction=1 | reduction=2 | ACLNN先验证0..2，binary attr与POD mode一致 |
| 模板与 dtype | 9个T/F组合，同一模板 | 同左 | 同左 | binary config静态特化，runtime key恒0 |
| 循环与分核 | P个grad位置grid-stride | P个target位置grid-stride | 同mean | 清零按O，计算按P；线程上限1024 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 三个4 B TBuf；无TQue | 同左，tmpOut有效 | 同左，total/tmpOut不影响结果 | 统一Process简化模板分支，但sum/none有冗余scalar路径 |
| 内存总占用与复用 | 12 B请求；完整GM清零 | 12 B；tmpOut block内广播 | 12 B；singleGrad广播 | 物理对齐固定仓不可见 |
| 搬运/Vector/SIMT/NDDMA/通信 API | InitOutput + SIMT target/grad/weight | InitOutput + scalar setup + SIMT target/weight | 同mean但不读total结果 | 无NDDMA/通信/Cube |
| 同步、精度与 tail | 两次SyncAll+Barrier；FP32乘；i<P | FP32除法+乘；同tail | FP32乘；仍执行未消费除法 | ignore依赖清零；target越界ASSERT |
| fallback/失败边界 | 无Device fallback | totalWeight非零前提 | 无Device fallback | 不支持在Host/Tiling/entry失败，不切换Route |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开API文档支持，但固定仓无该平台完整核心闭包，本卡不覆盖 | 同A2 | 公开支持且固定仓 source_closed |
| Host/Launcher | 非RegBase API会把FP16提升为FP32并做NHWC transpose/二维reshape；固定仓无A2核心实现/构建 | 同类DAV_2201分流；固定仓无A3 Host/Kernel构建 | RegBase保留NCHW；三种T均AICore；单SIMT Kernel |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | arch35 SIMT；3×4 B TBuf；16 MiB user workspace请求 |

## 限制与失败边界

- 仅 950、ND、`self[N,C,H,W]`、`target[N,H,W]`；公开 dtype 组合和 shape/同 dtype 约束见接口表。Host 不检查 target 元素值，非 ignore 越界由 Device `ASSERT` 终止。
- 空 `self` 在第一段返回 workspace=0且不建写回节点；固定源码没有写明 `out` 清零或保持语义。
- mean 依赖 `totalWeight[0]!=0`，但 ACLNN 不做值校验。`SimtComputeReduceMean2d/4d` 的源码断言写作 `ASSERT(localTotalWeight != 0...)`，而 helper 形参是 `tmpOut_`、没有名为 `localTotalWeight` 的局部标识符；固定源码无法闭合该标识符如何解析，本次未编译验证。
- none/sum 中 `totalWeight` 文档语义不活跃，但公共 `Process` 仍读取它并计算一次未消费的 FP32 商。不得据文档把该 GM 访问或三个 TBuf 从当前实现中删除。
- Tiling 对 H/W≤0 显式失败；N/C 的正值依赖 ACLNN 非空路径。`O=N*C*H*W`、`P=N*H*W` 使用 `uint64_t/int64_t` 混合中间量，固定源码未加乘法溢出检查。
- `SetLocalMemorySize(ubSizePlatForm-32768)` 未显式检查 UB 大于 32 KiB；950平台事实满足与否需运行环境保证。
- 核心固定请求16 MiB workspace并要求user workspace非空，但body不消费；保留原因、系统区/user区精确offset及executor总workspace聚合固定仓不可见。
- 同 dtype `Cast`、`Contiguous/ViewCopy` 的生产 selector、Kernel、物理 workspace、event与释放点不在固定 `ops-nn` 实现中；卡中仅陈述API源码可见的调用顺序、恒等dtype请求、owner与失败短路，未将其伪装成算子自有Route。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/nll_loss_grad/docs/aclnnNLLLoss2dBackward.md` | 产品支持表、函数原型 | A2/A3/950公开性、dtype/shape/错误语义 |
| 入口声明/约束 | `loss/nll_loss_grad/op_api/aclnn_nll_loss2d_backward.h` | 两段式声明 | ABI参数顺序 |
| 入口实现 | `loss/nll_loss_grad/op_api/aclnn_nll_loss2d_backward.cpp` | `CheckParams`、`aclnnNLLLoss2dBackwardGetWorkspaceSize` | 校验、950建图、空域、framework边界 |
| Host/Launcher | `loss/nll_loss_grad/op_api/nll_loss_grad.cpp` | `NLLLossGrad`、`NLLLossGradAiCore` | arch/dtype selector、launcher输入顺序 |
| Host/Launcher | `loss/nll_loss_grad/op_host/nll_loss_grad_def.cpp` | `NLLLossGrad::NLLLossGrad` | 950 OpDef dtype/config |
| Host/Tiling/Launcher | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.h` | `NLLLossGradSimtTilingData` | POD字段和Tiling类 |
| Host/Tiling/Launcher | `loss/nll_loss_grad/op_host/arch35/nll_loss_grad_tiling_arch35.cpp` | `NLLLossGradSimtTiling` | shape/attr/core/thread/workspace/local memory producer |
| Kernel entry/body | `loss/nll_loss_grad/op_kernel/nll_loss_grad.cpp` | `nll_loss_grad` | workspace guard、POD解析、模板实例 |
| Kernel entry/body | `loss/nll_loss_grad/op_kernel/arch35/nll_loss_grad.h` | `KernelNLLLossGrad<T,F>`及SIMT helpers | 清零、UB、同步、精度、tail和地址 |
| 构建接线 | `loss/nll_loss_grad/op_host/config/ascend950/nll_loss_grad_binary.json` | 27个`op_list`条目 | T/F/reduction binary特化 |
| 构建接线 | `loss/nll_loss_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | Host仅接ascend950/arch35 |
| 构建接线 | `loss/nll_loss_grad/op_kernel/CMakeLists.txt` | `add_kernel_sources` | Kernel接入ascend950 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 12字段producer/consumer映射、key恒0 | 字段顺序/位宽、mode枚举、单位；宏生成offset不得猜测 | `NLLLossGradSimtTilingData` |
| 调用顺序 | 全输出清零→scalar装载→Route SIMT→ViewCopy | 稀疏写不能早于清零同步；框架失败必须短路 | `Process`、ACLNN GetWorkspaceSize |
| workspace/内存 | 三个4 B TBuf、16 MiB核心workspace、UB−32 KiB local memory | 不能按逻辑12 B删除entry workspace guard；物理alignment未知 | `GetWorkspaceSize`、`PostTiling`、`InitBuffer` |
| 同步 | `SyncAll`保护清零与scalar，`PipeBarrier<PIPE_ALL>`发布tmpOut | producer/consumer顺序和参与核集合不变 | `ProcessMemset`、`Process` |
| 精度与尾块 | scalar/乘除FP32、最终cast T；q/r清零和grid-stride `i<P` | ignore留零、NCHW地址、target范围断言、mean除数前提 | 三个4d SIMT helper |
| 失败边界 | Host校验、Tiling shape/dtype失败、workspace null return、target ASSERT | 不得把A2/A3外部体或框架helper补造成已闭合950自有实现 | `CheckParams`、`GenerateTilingKey`、`nll_loss_grad`、`TargetCheck` |