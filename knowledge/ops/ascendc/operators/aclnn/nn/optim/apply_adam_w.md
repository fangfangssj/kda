---
type: Operator
title: "ApplyAdamW A2/A3 ACLNN 算子知识"
description: "说明 aclnnApplyAdamWV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐 Route AIV 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/docs/aclnnApplyAdamWV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/op_api/aclnn_apply_adam_w_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/op_api/aclnn_apply_adam_w_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/op_api/apply_adam_w_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/apply_adam_w_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/apply_adam_w_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/apply_adam_w_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_b16.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_fp.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_mix_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/config/ascend910b/apply_adam_w_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_host/config/ascend910_93/apply_adam_w_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:11:22Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ApplyAdamW A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnApplyAdamWV2GetWorkspaceSize` + `aclnnApplyAdamWV2`，公开 ACLNN 两段式 V2 接口。 |
| 计算语义 | 原地更新 `var/m/v`；`amsgrad=true` 时还原地更新 `maxGradNorm`；`maximize` 可先反转梯度。V2 从标量 `step` 在 Device 侧生成 bias correction。 |
| 输入输出 | `var/m/v/grad/maxGradNorm` 为同 shape ND Tensor；公开表列 F16/BF16/F32，`step` 为单元素 F32/I64；实现还允许 `var/m/v=F32`、`grad/maxGradNorm=F16/BF16` 的混合路线。 |
| 执行类型 | 所有 R-001～R-010 均为单 Kernel `AIV_VECTOR`；无 Cube/AIC/MIX_CV。ACLNN 的通用 Contiguous/Cast/ViewCopy 是框架边界。 |
| Kernel 路线 | R-001～R-004：同 dtype BF16/FP16；R-005/R-006：同 dtype FP32；R-007～R-010：FP32 状态与 FP16/BF16 梯度混合；每类再按 `step` F32/I64 分开。 |
| 核心资源 | GM、双缓冲 `TQue<VECIN/VECOUT>`、计算 `TBuf`、Vector 临时结果、两块 32 B `Power` 缓冲；核心声明 16 MiB workspace 但 body 不寻址。 |
| 来源闭包 | `source_closed`：公开 ACLNN→参数检查/连续化/可选 Cast→ApplyAdamWV2 launcher→A2/A3 Tiling/POD→10 key router→全部三类 Kernel body→binary/CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `s=float(step)+1`，`g=maximize ? -grad : grad`，逐元素执行：

```text
m' = beta1*m + (1-beta1)*g
v' = beta2*v + (1-beta2)*g*g
h  = amsgrad ? max(maxGradNorm, v') : v'
bc1 = 1 - beta1^s
bc2 = 1 - beta2^s
denom = sqrt(h) / sqrt(bc2) + eps
var' = (1-lr*weightDecay)*var - (lr/bc1)*m'/denom
maxGradNorm' = h                         # 仅 amsgrad=true
```

更新写回原 `var/m/v/maxGradNorm`。`step` 只读；源码未检查其值大于零，也未检查 `beta1/beta2` 导致 bias correction 为零或 `v/h` 为负的数值域。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `varRef` | ND；公开文档 rank 1～8；允许非连续 | F16/BF16/F32 | 输入/输出；空 Tensor 直接成功返回。 |
| `mRef` | 与 `varRef` 相同 | API 接受 F16/BF16/F32 | 输入/输出；公开契约要求同 dtype，API 实现会把其它合法组合统一 Cast 到 F32。 |
| `vRef` | 与 `varRef` 相同 | API 接受 F16/BF16/F32 | 输入/输出；源码不扫描负值。 |
| `maxGradNormOptionalRef` | 存在时与 `varRef` 相同 | API 接受 F16/BF16/F32 | `amsgrad=true` 时必需；输入/输出。公开表要求同 `varRef`，实现混合路线要求其 dtype 与 `grad` 一致。 |
| `grad` | 与 `varRef` 相同 | F16/BF16/F32 | 只读；特殊公开可达实现为状态 F32、梯度 F16/BF16。 |
| `step` | 标量或单元素 ND；API 先做 Contiguous | F32/I64 | 只读；Device 读取元素 0、转 F32 后加 1。 |
| `lr/beta1/beta2/weightDecay/eps` | Host 标量 | float | 直接作为算子属性写入 POD。 |
| `amsgrad/maximize` | Host 标量 | bool | 分别控制最大二阶矩更新与梯度取反。 |
| `workspaceSize/executor` | Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段聚合完整 Executor workspace并转移执行图所有权。 |
| `workspace/workspaceSize/stream` | Device workspace/ACL stream | 指针/`uint64_t`/`aclrtStream` | 第二段交给 `CommonOpExecutorRun`。 |

### 入口与版本

固定提交同时存在旧 `aclnnApplyAdamW` 与 V2。公开 `aclnnApplyAdamWV2.md` 明确支持 A2、A3 与 950，且 V2 是该版本族在 A2/A3 实际支持的最新公开入口；A2/A3 因此只提取 V2。V2 与旧入口的输入语义不同：旧入口传入 beta power Tensor，V2 传 `step` 并在 Device 侧求幂，不能把两者合并为同一实现版本。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 创建 `OpExecutor`；`CheckNotNull` 要求 `var/m/v/grad/step` 非空，并在 `amsgrad=true` 时要求 `maxGradNorm` 非空。
2. `GetDtypeSupportListBySocVersion` 对 DAV_2201/DAV_3510 允许 F32/F16/BF16，`step` 只允许 F32/I64；`CheckShape` 要求状态、梯度、可选最大二阶矩与 `var` 同 shape且 `step.Numel()==1`。
3. `var` 为空时返回 `workspaceSize=0`，不加入 Device 节点。
4. 依次对 `var/m/v/maxGradNorm/grad/step` 调用 `Contiguous`。若五个状态/梯度 Tensor既非全同 dtype，也非“`var/m/v=F32` 且 `grad/maxGradNorm=F16或BF16`”，则把它们全部 Cast 到 F32。
5. `l0op::ApplyAdamWV2` 通过 `ADD_TO_LAUNCHER_LIST_AICORE` 加入单个核心节点。若做过归一化 Cast，核心输出再分别 Cast 回各引用原 dtype；随后依次 `ViewCopy` 写回 `var/m/v/maxGradNorm`。
6. 任一中间 Tensor 或 launcher 创建失败立即返回错误并阻断后续节点；成功时 `GetWorkspaceSize` 后 `ReleaseTo(executor)`。

### 参数、Workspace 与生命周期

调用者拥有引用 Tensor、workspace 与 stream。第一段的 `uniqueExecutor` 暂时拥有所有 Contiguous/Cast 临时 Tensor和 launcher，`ReleaseTo` 后交给调用者持有的 `executor`；第二段在给定 stream 上执行。核心 Tiling 固定写 `16,777,216` B workspace，entry 仅以 `GetUserWorkspace(workspace)` 判空，三个真实 body 都不读写 `userWS`；ACLNN 总 workspace 还可能包含通用 Contiguous/Cast/ViewCopy 节点，其切片 offset 与内部事件由框架管理，固定仓不可见。

### 启动图与失败传播

```text
aclnnApplyAdamWV2GetWorkspaceSize
  -> Contiguous(var,m,v,max?,grad,step)
  -> [必要时] Cast(var,m,v,max?,grad -> F32)
  -> ApplyAdamWV2 AICore node
       entry apply_adam_w_v2(var,m,v,grad,step,max,workspace,tiling)
       blockDim = usedCoreNum
  -> [必要时] Cast(var,m,v,max? -> 各引用原 dtype)
  -> ViewCopy(var) -> ViewCopy(m) -> ViewCopy(v) -> [ViewCopy(max)]
aclnnApplyAdamWV2
  -> CommonOpExecutorRun(..., stream)
```

核心是单 Kernel，无跨核心节点 workspace edge。前后通用节点通过同一 Executor/stream 保持依赖顺序；任一 producer 创建或执行失败会阻断后继，临时 Tensor 至其最后 consumer 完成前由 Executor 持有。

## Route 与参数设计

### 分核与资源计算

记总元素数 `N`、AIV 核数 `C`、固定 tile `P=2432`：

```text
L = ceil(N/P)                         # 全部逻辑 tile 数
q = floor(L/C), r = L mod C
usedCoreNum = (q>0 ? C : r)
if r==0: handleExtra=usedCoreNum; loopPerCore=q-1
else:    handleExtra=r;           loopPerCore=q
numLastLoop = (N mod P)==0 ? P : (N mod P)
blockDim = usedCoreNum
tile(core,index) 起点 = (core + usedCoreNum*index)*P
```

`core < handleExtra-1` 先执行 `loopPerCore+1` 个整 tile；其它核执行 `loopPerCore` 个；`core==handleExtra-1` 再执行一个 `numLastLoop` 尾 tile。Host 虽取得 `ubSize`，但 `DoTiling` 不用它做 `P=2432` 的容量 guard。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 BF16/F32-step | 101 / `B16<bfloat16_t,float>` | 所有状态、grad、存在的max均BF16；step F32 | 下一 dtype 组合 |
| 2 | R-002 BF16/I64-step | 102 / `B16<bfloat16_t,int64_t>` | 同上；step I64 | 下一 dtype 组合 |
| 3 | R-003 FP16/F32-step | 103 / `B16<half,float>` | 所有状态、grad、存在的max均FP16；step F32 | 下一 dtype 组合 |
| 4 | R-004 FP16/I64-step | 104 / `B16<half,int64_t>` | 同上；step I64 | 下一 dtype 组合 |
| 5 | R-005 FP32/F32-step | 105 / `Fp<float,float>` | 所有状态、grad、存在的max均FP32；step F32 | 下一 dtype 组合 |
| 6 | R-006 FP32/I64-step | 106 / `Fp<float,int64_t>` | 同上；step I64 | 下一 dtype 组合 |
| 7 | R-007 FP32+FP16/F32-step | 107 / `MixType<float,half,float>` | var/m/v FP32；grad及存在的max FP16；step F32 | 下一 dtype 组合 |
| 8 | R-008 FP32+FP16/I64-step | 108 / `MixType<float,half,int64_t>` | 同上；step I64 | 下一 dtype 组合 |
| 9 | R-009 FP32+BF16/F32-step | 109 / `MixType<float,bfloat16_t,float>` | var/m/v FP32；grad及存在的max BF16；step F32 | 下一 dtype 组合 |
| 10 | R-010 FP32+BF16/I64-step | 110 / `MixType<float,bfloat16_t,int64_t>` | 同上；step I64 | 未命中不调用 body |

```text
if any non-step dtype differs:
  grad==F16  -> step==F32 ? 107 : 108
  grad==BF16 -> step==F32 ? 109 : 110
else:
  dtype==BF16 -> step==F32 ? 101 : 102
  dtype==F16  -> step==F32 ? 103 : 104
  dtype==F32  -> step==F32 ? 105 : 106
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 101 | A2/A3，同形BF16，step F32 | 单AIV | `ApplyAdamWV2B16<bfloat16_t,float>` | 16位GM、FP32计算TBuf、BF16 `CAST_ROUND` | `numLastLoop` |
| R-002 | 102 | 同上，step I64 | 单AIV | `ApplyAdamWV2B16<bfloat16_t,int64_t>` | 同R-001；step先I64→F32 | 同上 |
| R-003 | 103 | A2/A3，同形FP16，step F32 | 单AIV | `ApplyAdamWV2B16<half,float>` | 16位GM、FP32计算TBuf、FP16 `CAST_RINT` | 同上 |
| R-004 | 104 | 同上，step I64 | 单AIV | `ApplyAdamWV2B16<half,int64_t>` | 同R-003；step先I64→F32 | 同上 |
| R-005 | 105 | A2/A3，同形FP32，step F32 | 单AIV | `ApplyAdamWV2Fp<float,float>` | FP32 queue内原地复用，无宽度Cast | 同上 |
| R-006 | 106 | 同上，step I64 | 单AIV | `ApplyAdamWV2Fp<float,int64_t>` | 同R-005；step先I64→F32 | 同上 |
| R-007 | 107 | A2/A3，状态FP32、梯度/max FP16，step F32 | 单AIV | `ApplyAdamWV2MixType<float,half,float>` | 分 dtype queue；梯度升FP32；max降FP16 RINT | 同上 |
| R-008 | 108 | 同上，step I64 | 单AIV | `ApplyAdamWV2MixType<float,half,int64_t>` | 同R-007；step先I64→F32 | 同上 |
| R-009 | 109 | A2/A3，状态FP32、梯度/max BF16，step F32 | 单AIV | `ApplyAdamWV2MixType<float,bfloat16_t,float>` | 分 dtype queue；梯度升FP32；max降BF16 ROUND | 同上 |
| R-010 | 110 | 同上，step I64 | 单AIV | `ApplyAdamWV2MixType<float,bfloat16_t,int64_t>` | 同R-009；step先I64→F32 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空域 | `varRef->IsEmpty()` | 避免 `L=0` 导致 `usedCoreNum=0` | 全部之前 | 成功、无 Device 节点 | `aclnnApplyAdamWV2GetWorkspaceSize` |
| 混合 dtype 白名单 | `var=m=v=F32 && grad∈{F16,BF16} && (max为空或max=grad)` | 只有107～110具备分 dtype body | R-007～R-010 | 其它合法组合先全Cast F32→R-005/006 |
| 固定 tile | `P=2432`，不读取 `compileInfo.ubSize` | 三类 body静态分配分别为175168 B或184896 B | 全部 | 固定仓无更小tile fallback |
| workspace非空 | `GetUserWorkspace(workspace)!=nullptr` | entry即使不寻址userWS也以此为启动前提 | 全部 | 直接return、无写回 |
| AMSGrad可选输入 | `amsgrad==1 => max desc != nullptr` | body会CopyIn/CopyOut第4状态 | 全部 | Tiling/API失败 |
| step类型 | `F32 \|\| I64` | `GlobalTensor<U>::GetValue(0)` 后转F32 | 全部 | API/Tiling失败 |
| 尾DMA | `dataCount<=2432`，字节数窄化为uint32 | `DataCopyPad`只搬有效尾元素 | 全部 | 无额外fallback |
| 未知key | router仅有101～110，无最终else | 防止实例化不支持组合 | 全部 | kernel静默不写输出 |

### POD/ABI

`TILING_DATA_FIELD_DEF` 的生成宏定义不在固定仓内，故最终 `sizeof/alignof/offsetof` 不可见；下表保留源码声明顺序。所有字段由 `GetTilingData` 写入同名成员。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ApplyAdamWV2TilingData` | `totalCoreNum` | int64/64 | 第1字段；offset不可见 | AIV核 | platform `GetCoreNumAiv()` | body不读 | >0；inactive |
| 同上 | `handleExtraLoopCoreNum` | int64/64 | 第2字段 | 核数 | `r`或`usedCoreNum` | `Process`整tile/尾tile边界 | 1..used |
| 同上 | `usedCoreNum` | int64/64 | 第3字段 | 核数 | `q>0?C:r` | GM跨loop步长、block guard | >0 |
| 同上 | `numPerLoop` | int64/64 | 第4字段 | 元素 | 常量2432 | InitBuffer、GM地址、整tilecount | 2432 |
| 同上 | `loopNumPerCore` | int64/64 | 第5字段 | loop/核 | q或q-1 | `Process`循环 | ≥0 |
| 同上 | `numLastLoop` | int64/64 | 第6字段 | 元素 | `N%2432`或2432 | 尾Copy/Compute | 1..2432 |
| 同上 | `isBfloat16` | int64/64 | 第7字段 | bool标志 | `grad.dtype==BF16` | B16/Mix downcast round；FP不读 | 0/1 |
| 同上 | `lr` | float/32 | 第8字段 | 标量 | attr0 | `Init` stepSize/decay | 未做数值域校验 |
| 同上 | `beta1` | float/32 | 第9字段 | 标量 | attr1 | `ScalarPow`、m更新 | 未做数值域校验 |
| 同上 | `beta2` | float/32 | 第10字段 | 标量 | attr2 | `ScalarPow`、v更新 | 未做数值域校验 |
| 同上 | `weightDecay` | float/32 | 第11字段 | 标量 | attr3 | `1-lr*weightDecay` | 未做数值域校验 |
| 同上 | `eps` | float/32 | 第12字段 | 标量 | attr4 | denom加数 | 未做正值校验 |
| 同上 | `amsgrad` | int64/64 | 第13字段 | bool标志 | attr5 bool→0/1 | `ParseTilingData` | 0/1 |
| 同上 | `maximize` | int64/64 | 第14字段 | bool标志 | attr6 bool→0/1 | `ParseTilingData` | 0/1 |
| 同上 | `tilingKey` | int64/64 | 第15字段 | key | `GetTilingKey` | body不读；router读launch key | 101..110；POD副本inactive |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/R-002 | 101/102 | `GetTilingKey`,`DoTiling` | `ApplyAdamWV2TilingData` | `apply_adam_w_v2` 101/102 | `ApplyAdamWV2B16<bfloat16_t,U>` | ascend910b/910_93 binary + kernel CMake |
| R-003/R-004 | 103/104 | 同上 | 同上 | 同entry 103/104 | `ApplyAdamWV2B16<half,U>` | 同上 |
| R-005/R-006 | 105/106 | 同上 | 同上 | 同entry 105/106 | `ApplyAdamWV2Fp<float,U>` | 同上 |
| R-007/R-008 | 107/108 | 同上 | 同上 | 同entry 107/108 | `ApplyAdamWV2MixType<float,half,Z>` | 同上 |
| R-009/R-010 | 109/110 | 同上 | 同上 | 同entry 109/110 | `ApplyAdamWV2MixType<float,bfloat16_t,Z>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 先取得 user workspace并判空，再按 launch key实例化模板。每个模板在 `Init` 中读取单元素 `step`、绑定本核 GM基址、初始化 Queue/TBuf、每核只计算一次两个 bias correction；`Process` 按 `(blockIdx+usedCoreNum*loop)*2432` 遍历 tile。每个 tile严格执行 `CopyIn→Compute→CopyOut`，输入/输出 Queue提供搬运与Vector阶段生命周期，Vector步骤之间显式 `PipeBarrier<PIPE_V>`；`ScalarPow` 用 `V_S` event等待Vector结果后由标量核读取 lane0。

### Route R-001：BF16 状态 + F32 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey` | 同dtype BF16且step F32 | key101 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | `apply_adam_w_v2` | `TILING_KEY_IS(101)` | `B16<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
apply_adam_w_v2 -> B16<bfloat16_t,float>::Init -> ScalarPow(beta1/beta2)
  -> Process -> CopyIn -> Compute(BF16->FP32, AdamW, FP32->BF16 ROUND) -> CopyOut
```

##### 源码等价伪码

```text
for assigned tile: DMA 4/5 BF16 arrays -> cast FP32 -> optional -grad -> m',v',h,denom,var'
  -> cast var/m/v/(max) BF16 CAST_ROUND -> DMA valid dataCount back
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM states | `(core+used*loop)*2432` | `dataCount` | BF16 ND-flat | CopyIn | Compute/CopyOut | caller | tail按有效字节 |
| `inQueue` | 5个连续子区 | `2*5*2432*2=48640B` | BF16 | CopyIn | Compute | TPipe | depth2 |
| `outQueue` | 4个连续子区 | `2*4*2432*2=38912B` | BF16 | Compute | CopyOut | TPipe | depth2 |
| `inCast/outCast` | 同序子区 | `5*2432*4 + 4*2432*4=87552B` | FP32 | Cast/Vector | Vector/Cast | TPipe | 多阶段原地复用 |
| pow buffers | 各offset0 | `32+32B` | FP32 | Duplicate/Power | scalar read | TPipe | 每核初始化期 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inQueue d2 · 48640B"]:1
  outQ["outQueue d2 · 38912B"]:1
  inCast["inCast · 48640B"]:1
  outCast["outCast · 38912B"]:1
  pow["pow1+pow2 · 64B"]:1
```

##### 字节账与复用顺序

```text
UB_total=48640+38912+48640+38912+64=175168B; dynamic_ub=N/A; workspace_total=16MiB声明、body零切片
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `inCast[var]` | var Cast | decay/后续临时值 | denom、div结果 | PIPE_V barrier | 单tile串行 |
| `inCast[grad]` | grad Cast | m/v更新 | grad平方 | PIPE_V barrier | m更新已消费原grad |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA→Vector | TQue EnQue/DeQue | CopyIn | Compute | inQueue tile | BF16有效count |
| Vector链 | `PipeBarrier<PIPE_V>` | 每条Vector op | 下一条op | 原地TBuf | 全程FP32 |
| Power→Scalar | V_S event | `Power` | `GetValue(0)` | pow lane0 | step F32+1 |
| Vector→DMA | TQue EnQue/DeQue | Compute | CopyOut | outQueue tile | BF16 ROUND；tail=dataCount |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey/DoTiling` | key101、2432 tile、blockDim |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key101 branch | BF16/F32-step实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_b16.h` | `Init/ScalarPow/CopyIn/Compute/CopyOut/Process` | FP32升精度与BF16 round |

### Route R-002：BF16 状态 + I64 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey` | 同dtype BF16且step I64 | key102 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key102 | `B16<bfloat16_t,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> B16<bfloat16_t,int64_t>::Init(I64 step->F32) -> ScalarPow -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
stepF=float(stepI64)+1; 其余逐tile BF16->FP32 AdamW->BF16 ROUND，与R-001相同
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | `(core+used*loop)*2432` | GM `dataCount`；UB 175168B | BF16；step I64 | CopyIn/Init | Compute/ScalarPow | caller/TPipe | 同R-001 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inQ d2 48640B"] outQ["outQ d2 38912B"] inCast["inCast 48640B"] outCast["outCast 38912B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=175168B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| B16 cast区 | Cast/Vector | 下一Vector | 同tile后续Vector | PIPE_V | I64 step已在Init转标量F32 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | Queue+PIPE_V+V_S | DMA/Vector | 后继阶段 | tile/pow | I64先窄化F32；BF16 ROUND；有效tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_adam_w_v2_tiling.cpp` | key102 | I64 step分支 |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key102 | 模板实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_b16.h` | B16 methods | `GetValue<int64_t>`后`static_cast<float>` |

### Route R-003：FP16 状态 + F32 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey` | 同dtype FP16且step F32 | key103 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key103 | `B16<half,float>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> B16<half,float>::Init -> ScalarPow -> Process -> CopyIn -> Compute(FP32) -> CopyOut
```

##### 源码等价伪码

```text
逐tile搬FP16，升FP32完成全部AdamW，var/m/v/(max)用CAST_RINT降FP16后写回
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | `(core+used*loop)*2432` | GM有效count；UB 175168B | FP16/FP32 | CopyIn/Cast | Vector/CopyOut | caller/TPipe | 同R-001布局 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inQ d2 48640B"] outQ["outQ d2 38912B"] inCast["inCast 48640B"] outCast["outCast 38912B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=175168B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| cast区 | FP16 Cast | AdamW Vector | denom/div等 | PIPE_V | 原值最后消费后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | Queue+PIPE_V+V_S | DMA/Vector | 后继阶段 | tile/pow | FP32算、FP16 RINT、有效tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_adam_w_v2_tiling.cpp` | key103 | FP16/F32-step |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key103 | B16 half实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_b16.h` | `Compute` | FP32计算、FP16 `CAST_RINT` |

### Route R-004：FP16 状态 + I64 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey` | 同dtype FP16且step I64 | key104 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key104 | `B16<half,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> B16<half,int64_t>::Init(I64->F32) -> ScalarPow -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
stepF=float(stepI64)+1；逐tile FP16->FP32 AdamW->FP16 RINT
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | 公共地址式 | GM有效count；UB 175168B | FP16；step I64 | CopyIn/Init | Compute | caller/TPipe | depth2+cast区 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inQ d2 48640B"] outQ["outQ d2 38912B"] inCast["inCast 48640B"] outCast["outCast 38912B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=175168B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| cast区 | Cast/Vector | 后继Vector | 同tile后续 | PIPE_V | 同R-003 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | Queue+PIPE_V+V_S | DMA/Vector | 后继阶段 | tile/pow | I64→F32；FP16 RINT；tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_adam_w_v2_tiling.cpp` | key104 | 分支 |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key104 | 实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_b16.h` | B16 methods | I64 scalar+FP16输出 |

### Route R-005：FP32 状态 + F32 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey` | 同dtype FP32且step F32 | key105 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key105 | `Fp<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> Fp<float,float>::Init -> ScalarPow -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
逐tile搬4/5个FP32数组；queue子区原地轮换完成m',v',denom,var'；写3/4个FP32数组
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inQueue` | 5子区 | `2*5*2432*4=97280B` | FP32 | CopyIn | Compute | TPipe | depth2 |
| `outQueue` | 4子区 | `2*4*2432*4=77824B` | FP32 | Compute | CopyOut | TPipe | depth2 |
| pow | 0 | 64B | FP32 | ScalarPow | scalar | TPipe | 初始化期 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["inQueue d2 · 97280B"] outQ["outQueue d2 · 77824B"] pow["pow · 64B"]
```

##### 字节账与复用顺序

```text
UB_total=97280+77824+64=175168B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input var子区 | CopyIn | decay | v临时、denom、div | PIPE_V | 旧var已消费 |
| output max子区 | m/v临时 | 后继Vector或CopyOut(max) | 下一阶段 | PIPE_V | amsgrad决定最终语义 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/Vector/DMA | Queue+PIPE_V | 每阶段 | 后继 | queue子区 | FP32全程；tail有效count |
| Power→Scalar | V_S | Power | GetValue | lane0 | step F32 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_adam_w_v2_tiling.cpp` | key105 | FP32/F32-step |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key105 | Fp实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_fp.h` | all methods | FP32原地临时复用 |

### Route R-006：FP32 状态 + I64 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey` | 同dtype FP32且step I64 | key106 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key106 | `Fp<float,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> Fp<float,int64_t>::Init(I64->F32) -> ScalarPow -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
stepF=float(stepI64)+1；其余FP32逐tile流程与R-005相同
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/queues/pow | 公共地址式 | 175168B UB | FP32；step I64 | CopyIn/Init | Compute | caller/TPipe | 同R-005 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["inQ d2 97280B"] outQ["outQ d2 77824B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=175168B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output子区 | DMA/Vector | 后继Vector/DMA | 同tile下一阶段 | PIPE_V+Queue | 同R-005 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | Queue+PIPE_V+V_S | DMA/Vector | 后继 | tile/pow | I64可能窄化F32；tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/apply_adam_w_v2_tiling.cpp` | key106 | 分支 |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key106 | 实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_fp.h` | Fp methods | I64 step转F32 |

### Route R-007：FP32 状态 + FP16 梯度 + F32 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_apply_adam_w_v2.cpp` | `IsNeedDtypeCast` | 状态F32、grad/max FP16 | 不做整图Cast |
| Tiling/Router | Tiling与entry | key107 | grad FP16、step F32 | `MixType<float,half,float>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> MixType<float,half,float>::Init -> ScalarPow -> Process -> CopyIn(T/U) -> Compute -> CopyOut(T/U)
```

##### 源码等价伪码

```text
搬var/m/v FP32与grad/(max) FP16；grad升FP32；算m',v',var'；ams时max升FP32比较后RINT降FP16
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| inT/inU | 3/2子区 | `58368+19456B` | FP32/FP16 | CopyIn | Compute | TPipe | 各depth2 |
| outT/outU | 3/1子区 | `58368+9728B` | FP32/FP16 | Compute | CopyOut | TPipe | 各depth2 |
| result1/result2 | offset0 | `19456+19456B` | FP32 | Cast/Vector | Vector | TPipe | 2*2432元素各 |
| pow | offset0 | 64B | FP32 | Power | scalar | TPipe | 初始化期 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inT["inT d2 58368B"] inU["inU d2 19456B"] outT["outT d2 58368B"] outU["outU d2 9728B"] tmp["tmp1+2 38912B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=58368+19456+58368+9728+19456+19456+64=184896B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| result1 | grad Cast | m/v更新 | grad平方/scale | PIPE_V | 原grad末次消费后覆盖 |
| dataLocalT[var] | CopyIn | decay | denom/div | PIPE_V | 旧var已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dtype桥 | Cast NONE/RINT | FP16 grad/max | FP32计算/FP16 max | tmp/outU | 状态结果保持FP32 |
| pipeline | Queue+PIPE_V+V_S | DMA/Vector | 后继 | tile/pow | tail按各dtype字节数 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | API+Tiling | `IsNeedDtypeCast/GetTilingKey` | 混合路线公开可达 |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key107 | Mix实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_mix_dtype.h` | all methods | 分dtype queue和Cast |

### Route R-008：FP32 状态 + FP16 梯度 + I64 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | API+Tiling | mixed whitelist/`GetTilingKey` | 状态F32、grad/max FP16、step I64 | key108 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key108 | `MixType<float,half,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> MixType<float,half,int64_t>::Init(I64->F32) -> ScalarPow -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
stepF=float(stepI64)+1；分dtype逐tile处理与R-007相同
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | 公共地址式 | 184896B UB | 状态FP32、grad/max FP16、step I64 | CopyIn/Init | Compute | caller/TPipe | 同R-007 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inT["inT 58368B"] inU["inU 19456B"] outT["outT 58368B"] outU["outU 9728B"] tmp["tmp 38912B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=184896B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| result/dataLocal | Cast/Vector | 后继Vector | 同tile后续 | PIPE_V | 同R-007 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | Queue+PIPE_V+V_S | DMA/Vector | 后继 | tile/pow | I64→F32；max FP16 RINT；tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | API+Tiling | mixed/key108 | 分支 |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key108 | 实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_mix_dtype.h` | Mix methods | I64 step与FP16梯度 |

### Route R-009：FP32 状态 + BF16 梯度 + F32 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | API+Tiling | mixed whitelist/`GetTilingKey` | 状态F32、grad/max BF16、step F32 | key109 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key109 | `MixType<float,bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> MixType<float,bfloat16_t,float>::Init -> ScalarPow -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
搬FP32状态与BF16 grad/max；升FP32计算；ams结果CAST_ROUND回BF16；状态保持FP32
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | 公共地址式 | 184896B UB | FP32/BF16 | CopyIn | Compute/CopyOut | caller/TPipe | 同R-007 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inT["inT 58368B"] inU["inU 19456B"] outT["outT 58368B"] outU["outU 9728B"] tmp["tmp 38912B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=184896B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| result1/result2 | Cast/Max | m/v/denom | 同tile后续 | PIPE_V | 同R-007 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dtype桥 | Cast NONE/ROUND | BF16 | FP32/BF16 | tmp/outU | BF16最大二阶矩ROUND |
| pipeline | Queue+PIPE_V+V_S | DMA/Vector | 后继 | tile/pow | 有效tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | API+Tiling | mixed/key109 | 分支 |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key109 | 实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_mix_dtype.h` | Mix methods | BF16 round |

### Route R-010：FP32 状态 + BF16 梯度 + I64 step

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/Tiling | API+Tiling | mixed whitelist/`GetTilingKey` | 状态F32、grad/max BF16、step I64 | key110 |
| Router | `op_kernel/apply_adam_w_v2.cpp` | entry | key110 | `MixType<float,bfloat16_t,int64_t>` |

#### Device 数据流

##### 函数级调用链

```text
entry -> MixType<float,bfloat16_t,int64_t>::Init(I64->F32) -> ScalarPow -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
stepF=float(stepI64)+1；FP32状态+BF16梯度逐tile流程与R-009相同
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/UB | 公共地址式 | 184896B UB | FP32/BF16；step I64 | CopyIn/Init | Compute | caller/TPipe | 同R-009 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inT["inT 58368B"] inU["inU 19456B"] outT["outT 58368B"] outU["outU 9728B"] tmp["tmp 38912B"] pow["pow 64B"]
```

##### 字节账与复用顺序

```text
UB_total=184896B; dynamic_ub=N/A; workspace core slice=0B used
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| result/dataLocal | Cast/Vector | 后继 | 同tile后续 | PIPE_V | 同R-009 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流程 | Queue+PIPE_V+V_S | DMA/Vector | 后继 | tile/pow | I64→F32；BF16 ROUND；tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | API+Tiling | mixed/key110 | 最终分支 |
| Kernel entry/router | `op_kernel/apply_adam_w_v2.cpp` | key110 | 实例 |
| Device body/helper | `op_kernel/apply_adam_w_v2_mix_dtype.h` | Mix methods | I64+BF16路径 |

## 算子特定优化

### Trick T-001：每核只做一次 bias correction 幂计算

#### 触发与源码依据

- 适用 Route：R-001～R-010。
- 触发条件：每次 Kernel 初始化，任意非空输入。
- Golden 来源：[`op_kernel/apply_adam_w_v2_b16.h:167-180`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_b16.h#L167-L180)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
LocalTensor<float> baseLocal = powTempBuf1_.Get<float>();
LocalTensor<float> outLocal = powTempBuf2_.Get<float>();
PipeBarrier<PIPE_V>();
Duplicate(baseLocal, x, BLOCK_SIZE_FOR_FLOAT32);
PipeBarrier<PIPE_V>();
Power<float, false>(outLocal, baseLocal, y, BLOCK_SIZE_FOR_FLOAT32);
event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(eventIdVToS);
WaitFlag<HardEvent::V_S>(eventIdVToS);
float result = outLocal.GetValue(0);
```

#### 机制与设计原因

`step` 和 beta 对所有元素相同，`Init` 用两块各32 B的buffer生成8个相同lane，`Power` 后用V→S event只取lane0；两个幂值随后转成每核标量 `stepSize` 与 `biasCorrection2Sqrt`，所有tile复用，避免在逐元素主循环重复求幂。

#### 迁移边界

- 适用：标量超参数控制的大批量逐元素优化器。
- 前提：每个Kernel节点内step/beta不变，具备V→S可见性协议。
- 失效：超参数逐元素变化时必须放回Vector主循环。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若省略 `SetFlag/WaitFlag<V_S>` 就在标量侧读取 `outLocal[0]`，源码协议不能保证Power结果可见；固定仓未提供运行时反例。

### Trick T-002：16位状态全程升到FP32计算后按dtype选择舍入

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：状态 dtype 为 FP16 或 BF16。
- Golden 来源：[`op_kernel/apply_adam_w_v2_b16.h:277-285`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_b16.h#L277-L285)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (isBfloat16_) {
    Cast(dataOutLocal[varOffset_], outCastLocal[varOffset_], RoundMode::CAST_ROUND, dataCount);
    Cast(dataOutLocal[expAvgOffset_], outCastLocal[expAvgOffset_], RoundMode::CAST_ROUND, dataCount);
    Cast(dataOutLocal[expAvgSqOffset_], outCastLocal[expAvgSqOffset_], RoundMode::CAST_ROUND, dataCount);
} else {
    Cast(dataOutLocal[varOffset_], outCastLocal[varOffset_], RoundMode::CAST_RINT, dataCount);
    Cast(dataOutLocal[expAvgOffset_], outCastLocal[expAvgOffset_], RoundMode::CAST_RINT, dataCount);
    Cast(dataOutLocal[expAvgSqOffset_], outCastLocal[expAvgSqOffset_], RoundMode::CAST_RINT, dataCount);
}
```

#### 机制与设计原因

输入一次升到FP32，动量、平方、开方、除法和权重衰减都不在16位中间结果上连锁舍入；最终根据 `isBfloat16` 选择 BF16 `CAST_ROUND` 或 FP16 `CAST_RINT`。代价是显式cast TBuf，但三类布局仍在固定tile下给出可证明UB上界。

#### 迁移边界

- 适用：需要多步非线性与累积的F16/BF16状态更新。
- 前提：UB能容纳输入/输出16位Queue和两组FP32计算区。
- 失效：UB不足时必须减tile或改成分阶段实现；固定Host没有该fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把所有中间量保持16位会改变固定源码的精度顺序；固定仓未提供数值误差case。

### Trick T-003：按最后消费点原地覆盖UB子区

#### 触发与源码依据

- 适用 Route：R-005/R-006；B16与Mix body也使用同类生命周期复用。
- 触发条件：单tile内严格串行Vector管线。
- Golden 来源：[`op_kernel/apply_adam_w_v2_fp.h:247-253`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_fp.h#L247-L253)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
PipeBarrier<PIPE_V>();
Div(dataLocal[expAvgOffset_], dataOutLocal[expAvgOffset_], dataLocal[varOffset_], dataCount);
PipeBarrier<PIPE_V>();
Muls(dataLocal[varOffset_], dataLocal[expAvgOffset_], stepSize_, dataCount);
PipeBarrier<PIPE_V>();
Sub(dataOutLocal[varOffset_], dataOutLocal[varOffset_], dataLocal[varOffset_], dataCount);
```

#### 机制与设计原因

旧 `var` 在权重衰减结果写入 `dataOutLocal[var]` 后已结束生命周期，因此其输入子区依次承载sqrt/denom和缩放后的更新量；旧m子区又承载除法结果。显式PIPE_V barrier约束覆盖顺序，避免为denom与update另开TBuf。

#### 迁移边界

- 适用：单流水、消费者顺序静态可知的逐元素链。
- 前提：覆盖前旧值已完成最后消费，并保持同一Vector pipe barrier。
- 失效：若并行消费者仍读旧值，必须拆buffer或增加正确同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

提前覆盖 `dataLocal[var]` 会破坏权重衰减或denom；源码通过操作顺序和barrier给出真实反例边界。

## 模板与实现对比

| 对比维度 | R-001～R-004 B16 | R-005/R-006 FP32 | R-007～R-010 Mix | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 全状态/grad同F16或BF16 | 全同FP32或ACLNN归一化到FP32 | 状态FP32、grad/max F16/BF16 | `GetTilingKey`顺序 |
| 模板与 dtype | `B16<T,U>` | `Fp<float,U>` | `MixType<float,U,Z>` | GM宽度与step类型 |
| 循环与分核 | 固定2432、相同core循环 | 同左 | 同左 | Host统一Tiling |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 双Queue+2 cast TBuf+pow | 双Queue+pow | 4 Queue+2 result TBuf+pow | dtype桥数量 |
| 内存总占用与复用 | 175168B | 175168B | 184896B | Mix需要同时保存两种宽度 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+Cast+Vector | DataCopyPad+Vector | 分dtype DataCopyPad+Cast+Vector | 均无SIMT/NDDMA/通信 |
| 同步、精度与 tail | FP32算，BF16 ROUND/FP16 RINT | FP32全程 | FP32算，max降16位 | Queue、PIPE_V、V_S；有效count尾块 |
| fallback/失败边界 | UB无容量fallback | 同左 | 非白名单混合由ACLNN先Cast FP32 | 未知key静默无写回 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2公开支持；本卡闭合 | V2公开支持；本卡闭合 | V2公开支持，但固定ops-nn仓缺ATVOSS scheduler源码，未生成950卡 |
| Host/Launcher | 非Regbase `DoTiling`，ascend910b config | 非Regbase `DoTiling`，ascend910_93 config | Regbase `ApplyAdamWV2RegbaseTiling` |
| Kernel/资源 | 10 key自定义AIV body | 同A2 | DAG+`ElementwiseSch`；tile/UB/同步固定仓不可见 |

## 限制与失败边界

- 公共文档要求 `var/m/v/grad` dtype一致，但ACLNN实现和OpDef额外接通状态FP32、梯度/最大二阶矩FP16或BF16的107～110路线；这是固定源码可达能力，调用方若只依赖公开表应按同dtype使用。
- API代码没有显式rank 1～8检查、step>0检查、`v>=0`扫描，也没有beta/eps数值域检查；非法数值可能在Power/Sqrt/Div中传播NaN/Inf。
- `step` I64先转F32，绝对值较大时精度会窄化；所有地址/元素数由shape乘积进入size_t/int64，固定实现未给出显式溢出检查。
- Host固定tile 2432且不使用已读取的UB大小做容量选择；所需175168B或184896B是移植时不可改变的资源前提，固定仓没有小UB fallback。
- entry在user workspace为空时静默return；未知key也无错误分支，均可能导致引用Tensor未更新。
- ACLNN前后通用Contiguous/Cast/ViewCopy的内部kernel、workspace切片与事件属于框架边界；本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `optim/apply_adam_w_v2/docs/aclnnApplyAdamWV2.md` | 产品表/函数原型 | A2/A3公开支持与接口约束 |
| 入口声明/约束 | `optim/apply_adam_w_v2/op_host/op_api/aclnn_apply_adam_w_v2.h` | 两段式声明 | ABI顺序 |
| 入口实现 | `optim/apply_adam_w_v2/op_host/op_api/aclnn_apply_adam_w_v2.cpp` | `CheckParams/IsNeedDtypeCast/aclnnApplyAdamWV2*` | 校验、前后处理、执行器 |
| 入口实现 | `optim/apply_adam_w_v2/op_host/op_api/apply_adam_w_v2.cpp` | `l0op::ApplyAdamWV2` | AICore launcher节点 |
| Host注册 | `optim/apply_adam_w_v2/op_host/apply_adam_w_v2_def.cpp` | `ApplyAdamWV2` | dtype组合与平台注册 |
| Host/Tiling/POD | `optim/apply_adam_w_v2/op_host/apply_adam_w_v2_tiling.h` | key常量/`ApplyAdamWV2TilingData` | POD字段和10 key |
| Host/Tiling/POD | `optim/apply_adam_w_v2/op_host/apply_adam_w_v2_tiling.cpp` | `GetTilingKey/DoTiling/Tiling4ApplyAdamWV2` | route、tile、block、workspace |
| Kernel entry/body | `optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2.cpp` | `apply_adam_w_v2` | 10 key router |
| Kernel entry/body | `optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_base.h` | 常量 | queue深度、子区序号 |
| Kernel entry/body | `optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_b16.h` | `ApplyAdamWV2B16` | F16/BF16升精度body |
| Kernel entry/body | `optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_fp.h` | `ApplyAdamWV2Fp` | FP32 body |
| Kernel entry/body | `optim/apply_adam_w_v2/op_kernel/apply_adam_w_v2_mix_dtype.h` | `ApplyAdamWV2MixType` | 混合dtype body |
| 构建接线 | `optim/apply_adam_w_v2/op_host/config/ascend910b/apply_adam_w_v2_binary.json` | `ApplyAdamWV2` entries | A2 binary组合 |
| 构建接线 | `optim/apply_adam_w_v2/op_host/config/ascend910_93/apply_adam_w_v2_binary.json` | `ApplyAdamWV2` entries | A3 binary组合 |
| 构建接线 | `optim/apply_adam_w_v2/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend910b/910_93活动构建 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15字段同序producer/consumer | 类型、顺序、单位；最终offset固定仓不可见 | `ApplyAdamWV2TilingData/GetTilingData` |
| 调用顺序 | `Contiguous→可选Cast→Apply→可选Cast→ViewCopy` | 引用写回顺序与失败短路 | `aclnnApplyAdamWV2GetWorkspaceSize` |
| workspace/内存 | 核心声明16MiB但不寻址；三类UB账固定 | userWS判空、tile2432、depth2 | `Tiling4ApplyAdamWV2/Init` |
| 同步 | Queue、PIPE_V、V_S event | Power结果标量读取前必须V_S等待 | `ScalarPow/Compute` |
| 精度与尾块 | 16位升FP32；BF16 ROUND、FP16 RINT；I64 step转F32 | Cast顺序与`numLastLoop`有效count | B16/Mix `Compute`, all `Process` |
| 失败边界 | 空Tensor成功无kernel；未知key/userWS空静默不写 | 不得把静默return解释为成功更新 | API entry/kernel entry |

- [Power（数学计算）](../../../../apis/simd_api/adv_api/math_compute/power_interface/power.md)
