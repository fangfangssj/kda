---
type: Operator
title: "ChamferDistanceBackward A2/A3 ACLNN 算子知识"
description: "说明 aclnnChamferDistanceBackward 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/docs/aclnnChamferDistanceBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_host/op_api/aclnn_chamfer_distance_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_host/op_api/chamfer_distance_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_host/chamfer_distance_grad_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_host/chamfer_distance_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_host/chamfer_distance_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_host/config/ascend910b/chamfer_distance_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_host/config/ascend910_93/chamfer_distance_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T17:13:44Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ChamferDistanceBackward A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式接口 `aclnnChamferDistanceBackwardGetWorkspaceSize` / `aclnnChamferDistanceBackward`；算子自有 Host op 为 `ChamferDistanceGrad`。 |
| 计算语义 | 根据 `idx1/idx2` 指定的最近邻关系，把两侧距离梯度变成点坐标梯度；每一侧都包含本点连续写贡献和另一点集反向散射贡献。 |
| 输入输出 | `xyz1/xyz2: [B,N,2]`，`idx1/idx2/gradDist1/gradDist2: [B,N]`，输出 `gradXyz1/gradXyz2: [B,N,2]`；ND；公开 ACLNN 接受 FP16/FP32，索引为 INT32，核心 op 实际接收 FP32。 |
| 执行类型 | 算子自有 Device 节点为 `AIV_VECTOR`；单 Kernel。ACLNN 的 `Contiguous/Cast/ViewCopy` 属于框架前后处理边界。 |
| Kernel 路线 | R-001：ACLNN 可达 FP32 核心路线，TilingKey 1，`ChamferDistanceGrad<float>`。源码中的 Key 2 / `half` body 被 ACLNN 的前置 FP32 Cast 与 FP32 binary config 截断，不是正式 Route。 |
| 核心资源 | GM 输入/输出；6 个 depth=1 的 `TQue`；无算子业务 workspace 中间量，Tiling 固定申报 16 MiB 系统 workspace；输出以 `InitOutput + SyncAll + AtomicAdd` 汇聚。 |
| 来源闭包 | `source_closed`：公开 ACLNN → FP32 归一化 → `ChamferDistanceGrad` Host/Tiling/POD → `chamfer_distance_grad` entry → `ChamferDistanceGrad<float>` body → A2/A3 构建接线闭合；外部框架 Cast/Contiguous/ViewCopy 明确作为框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `p[b,i]=xyz1[b,i,:]`、`q[b,j]=xyz2[b,j,:]`，`j=idx1[b,i]`、`i=idx2[b,j]`。核心 Kernel 实际累加：

```text
gradXyz1[b,i] += 2 * gradDist1[b,i] * (p[b,i] - q[b,idx1[b,i]])
gradXyz2[b,idx1[b,i]] -= 2 * gradDist1[b,i] * (p[b,i] - q[b,idx1[b,i]])

gradXyz2[b,j] += 2 * gradDist2[b,j] * (q[b,j] - p[b,idx2[b,j]])
gradXyz1[b,idx2[b,j]] -= 2 * gradDist2[b,j] * (q[b,j] - p[b,idx2[b,j]])
```

输出先整体清零；随后所有连续贡献与按最近邻索引散射的贡献均用 `AtomicAdd` 合并。多个点命中同一索引时加法次序不固定，因此公开文档将默认实现标为非确定性。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `xyz1` | `[B,N,2]`, ND | FP16/FP32 | 点集 1。核心 Tiling 只从该输入读取 `B,N`。 |
| `xyz2` | 实现要求 `[B,N,2]`, ND | FP16/FP32 | 点集 2。API 表中写作 `[B,M,2]`，但 proto、GM extent 和地址公式均使用与 `xyz1` 相同的 `N`，故固定实现要求 `M=N`。 |
| `idx1` | `[B,N]`, ND | INT32 | `xyz1` 到 `xyz2` 的最近邻索引；固定 ACLNN 参数检查不做值域校验，调用方必须保证 `0 <= idx1 < N`。 |
| `idx2` | `[B,N]`, ND | INT32 | `xyz2` 到 `xyz1` 的最近邻索引；固定 ACLNN 参数检查不做值域校验，调用方必须保证 `0 <= idx2 < N`。 |
| `gradDist1` | `[B,N]`, ND | FP16/FP32 | 第一侧距离梯度。 |
| `gradDist2` | `[B,N]`, ND | FP16/FP32 | 第二侧距离梯度。 |
| `gradXyz1` | `[B,N,2]`, ND | FP16/FP32 | 点集 1 梯度，写入用户提供 Tensor/View。 |
| `gradXyz2` | `[B,N,2]`, ND | FP16/FP32 | 点集 2 梯度，写入用户提供 Tensor/View。 |

公开层只检查非空指针和每个 Tensor 的 dtype 是否属于支持集合；shape/rank、输入间 shape 一致性、浮点 dtype 一致性和索引值域没有在 `CheckParams` 中校验。内部核心 op 只注册 FP32 binary，且两个输出都按 `xyz1` 的原始 dtype 回 Cast，因此复用时应保持所有浮点输入输出 dtype 一致。

### 入口与版本

固定 Commit 只有 `aclnnChamferDistanceBackward` 这一公开版本族。第一段接口创建 executor、做参数检查、处理空 Tensor、将六个输入连续化，把四个浮点输入统一 Cast 到 FP32，再登记一个 `ChamferDistanceGrad` AICore 节点；核心输出 Cast 回 `xyz1` 的原始 dtype并通过 `ViewCopy` 写入用户输出。第二段接口调用 `CommonOpExecutorRun` 在给定 stream 上执行整个 executor。

README 与 ACLNN 文档公开支持 Atlas A2、A3，明确不支持 Ascend 950PR/950DT；OpDef、binary config、Kernel CMake 和全局 `ascendc_config.json` 也只接线 `ascend910b` 与 `ascend910_93`。因此本卡覆盖 A2/A3，不存在 950 公开版本。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序如下：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`，`CheckNotNull` 检查 8 个 Tensor 指针。
2. `CheckDtypeValid` 要求 `xyz* / gradDist* / gradXyz*` 各自为 FP16 或 FP32，`idx*` 为 INT32。
3. 六个输入任一 `IsEmpty()` 时直接返回 `workspaceSize=0`，不登记核心或输出拷贝节点。
4. 对六个输入依次调用 `l0op::Contiguous`；空返回转为 `ACLNN_ERR_INNER_NULLPTR`。
5. 记录 `curType=xyz1Contiguous->GetDataType()`；将 `xyz1/xyz2/gradDist1/gradDist2` Cast 为 FP32，索引保持 INT32。
6. `l0op::ChamferDistanceGrad` 按 `xyz1/xyz2` view shape 分配两个 FP32 中间输出，并通过 `ADD_TO_LAUNCHER_LIST_AICORE` 登记核心节点。
7. 两个核心输出均 Cast 到 `curType`，再分别 `ViewCopy` 到 `gradXyz1/gradXyz2`。
8. executor 汇总 workspace，第一段返回；第二段将 caller-owned workspace、大小、executor 和 stream 交给 `CommonOpExecutorRun`。

`Contiguous/Cast/ViewCopy` 的具体 Kernel、workspace 切片和 no-op/alias 条件不在该固定算子源码中；它们是框架边界。它们不承担 Chamfer 反向的核心数学语义。

### 参数、Workspace 与生命周期

核心 Tiling 的 compile info 由 `TilingPrepareForChamferDistanceGrad` 生成：`totalCoreNum=GetCoreNumAiv()`，`ubSizePlatForm=GetCoreMemSize(UB)`；任一值非正则 Tiling Prepare 失败。业务 Tiling 将可用 UB 定义为 `ub_size=ubSizePlatForm-1024`，并固定申报 `currentWorkspace[0]=16*1024*1024` 字节。

核心 Kernel 参数中的 `workspace` 未被 body 读取，所以这 16 MiB 是系统 workspace 申报而非本算子的业务中间结果。公开第一段返回的 `workspaceSize=executor->GetWorkspaceSize()` 还可能包含框架 Contiguous/Cast/ViewCopy 的需求；其分片 offset 在固定算子源码中不可见。workspace 由调用方在第二段执行前分配并保持有效，executor 持有中间 Tensor/节点描述直到 `CommonOpExecutorRun` 完成入队；GM 输入、输出和 workspace 至少保持到 stream 上相关任务结束。

### 启动图与失败传播

```text
public inputs
  -> framework Contiguous(x6)
  -> framework Cast floating inputs -> FP32 (x4)
  -> node C0: ChamferDistanceGrad / chamfer_distance_grad
       blockDim = core_used
       POD = ChamferDistanceGradTilingData
       inputs = FP32 xyz1, xyz2, gradDist1, gradDist2 + INT32 idx1, idx2
       outputs = FP32 intermediate gradXyz1, gradXyz2
       workspace = 16 MiB system slice, body unused
  -> framework Cast outputs -> curType (x2)
  -> framework ViewCopy -> user gradXyz1, gradXyz2
```

所有节点由同一个 executor 按同一 stream 的依赖顺序登记和运行；每一步空返回或 `ADD_TO_LAUNCHER_LIST_AICORE` 失败都会立即终止第一段，后续节点不再登记。核心 C0 内先由 block 0 清零两份输出，再 `SyncAll`，随后所有 block 原子累加；C0 结束后框架输出 Cast/ViewCopy 才消费其中间输出。固定算子源码没有暴露 executor 对框架节点的 workspace offset、event 或清理实现。

## Route 与参数设计

### 分核与资源计算

令总点任务 `Q=B*N`，平台 AIV 核数 `C=totalCoreNum`：

```text
task_per_core  = ceil_div(Q, C) = (Q - 1) / C + 1
core_used      = ceil_div(Q, task_per_core)
task_tail_core = Q - (core_used - 1) * task_per_core
blockDim       = core_used

start_task_id(block) = task_per_core * blockIdx
cur_core_task_num    = blockIdx == core_used-1 ? task_tail_core : task_per_core
```

ACLNN 在任一输入为空时提前返回，因而正式入口不会把 `Q=0` 送入上述含 `Q-1` 的无符号公式。每个 block 的任务区间连续，chunk 可跨 batch 边界；单点循环以 `(globalTask/num, globalTask%num)` 恢复 batch 和点索引。

令 `U=ubSizePlatForm-1024` 字节、`P=per_ub_size` 元素：

```text
P = floor(U / 6 / sizeof(float) / 32) * 32
  = floor(U / 768) * 32 elements
```

公开核心固定为 FP32，因此 `t_per_block=32/4=8` 个元素。每次处理 `min(P, remaining)` 个点；完整 chunk 用 `P`，尾 chunk 用余数。`P` 对齐到 32 个元素，比一次 32-byte DMA 对齐更保守，同时保证 Muls 的向量分段和 UB 六区预算。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原子散射 | Key 1 / `ChamferDistanceGrad<float>` | 公开 ACLNN 的非空路径总把四个浮点输入 Cast 为 `DT_FLOAT`；Tiling 读取输入 0 dtype 为 `DT_FLOAT` 后设置 Key 1。A2/A3 binary config 仅登记 float32。 | 非法 dtype 在 ACLNN 校验失败；空输入直接成功返回；Key 2 只存在于 Device router，ACLNN 不可达。 |

```text
if any public input is empty:
    return success, workspaceSize=0
else:
    cast xyz1/xyz2/gradDist1/gradDist2 to FP32
    if core input dtype == DT_FLOAT:
        tilingKey = 1 -> R-001
    else if core input dtype == DT_FLOAT16:
        tilingKey = 2 -> dormant ChamferDistanceGrad<half>
    else:
        tiling source defaults key 1, but no public ACLNN path reaches it
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原子散射 | Key 1 | ACLNN 非空；公开 dtype 校验通过；浮点输入经 Cast 后为 FP32；`[B,N,2] / [B,N]` 等长约束；A2/A3 | 单个自有核心节点 C0；外围框架前后处理不展开为算子 Route | `chamfer_distance_grad` → `ChamferDistanceGrad<float>::process` → d1、d2 两阶段 | 6 个 depth=1 Queue；block 0 清零后全核同步；连续贡献与索引散射均 AtomicAdd | 每核尾任务、每 chunk 尾点；尾 chunk 将 `d_local` 对齐区补零；非法/空域在 ACLNN 前置终止。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 非空域 | ACLNN 六个输入均非空 | 避免 `Q-1` 无符号下溢及零 blockDim | R-001 | 直接成功，workspace 0，不启动 Kernel | `aclnnChamferDistanceBackwardGetWorkspaceSize` |
| AIV 核数 | `GetCoreNumAiv() > 0` | 分核公式以核数作除数 | R-001 | Tiling Prepare 返回失败 | `TilingPrepareForChamferDistanceGrad` |
| UB 容量 | `GetCoreMemSize(UB) > 0`；隐含需使 `P>0` | `P` 是 chunk 除数并决定 6 个 Queue 容量 | R-001 | 非正 UB 显式失败；`0 < UB <= 1791` 导致 `P=0` 的情况源码未设 fallback | `TilingPrepareForChamferDistanceGrad`, constructor |
| 32 元素对齐 | `P=floor((UB-1024)/768)*32` | 约束 Queue 分区并简化向量/DMA tail | R-001 | 缩小 P，无替代 Route | `ChamferDistanceGrad` constructor |
| Muls repeat 上限 | `max_repeat=255`, `mask=256/sizeof(T)=64` | 单次连续 Muls repeat 不超过 255 | R-001 | 外层按 `255*64` 元素切分，余 repeat 与 mask tail 单独处理 | `muls_template` |
| 等长点集 | 实际地址公式统一使用 `num=xyz1.shape[1]` | `xyz2/idx2/gradDist2/gradXyz2` 的 GM extent 与 batch stride均按 N | R-001 | 无运行时 fallback；不满足属于调用边界 | constructor, `main_cal_func_d2` |
| 索引值域 | `0 <= idx1,idx2 < N`，由调用方保证 | 每点直接形成 GM 坐标地址 | R-001 | 无运行时校验/fallback，越界行为不受支持 | `main_cal_func_d1`, `main_cal_func_d2` |
| 平台 | `ascend910b` 或 `ascend910_93` | 只有这两个 binary/config/build target | R-001 | 950 不发布 | OpDef、Kernel CMake、`ascendc_config.json` |

### POD/ABI

`ChamferDistanceGradTilingData` 由仓外 `tilingdata_base.h` 宏生成；固定仓能确认字段声明顺序、源码类型和 Host/Device 读写，但不能确认宏展开后的 `sizeof/alignof/offsetof`，因此不推测具体 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ChamferDistanceGradTilingData` | `batch_size` | `uint64_t` / 64 bit | 第 1 字段；具体 offset/整体布局固定仓不可见 | batch 数 | `Init`: `xyz1.storageShape.dim(0)`；`RunKernelTiling`: `set_batch_size` | constructor 计算 `B*N` 与 GM extent | 非空公开路径应大于 0 |
| 同上 | `num` | `uint64_t` / 64 bit | 第 2 字段；具体 offset 固定仓不可见 | 点/批 | `Init`: `xyz1.storageShape.dim(1)` | constructor、batch/point 地址恢复 | 实现要求两点集同 N |
| 同上 | `ub_size` | `uint64_t` / 64 bit | 第 3 字段；具体 offset 固定仓不可见 | 字节 | `Init`: `ubSizePlatForm - 1024` | constructor 计算 `P` | 未验证减法下溢；平台值须足够大 |
| 同上 | `task_per_core` | `uint64_t` / 64 bit | 第 4 字段；具体 offset 固定仓不可见 | 点任务/核 | `Init`: `ceil_div(B*N,totalCoreNum)` | constructor 生成起始 task 与常规核任务量 | 转入 `start_task_id` 保持 64 bit |
| 同上 | `core_used` | `uint64_t` / 64 bit | 第 5 字段；具体 offset 固定仓不可见 | 核 | `Init`: `ceil_div(B*N,task_per_core)`；同时设为 blockDim | constructor 判断最后一核 | Device 与 `GetBlockNum()` 应一致 |
| 同上 | `task_tail_core` | `uint64_t` / 64 bit | 第 6 字段；具体 offset 固定仓不可见 | 点任务 | `Init`: `B*N-(core_used-1)*task_per_core` | constructor 覆盖最后一核任务量 | 仅最后 block 使用 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 原子散射 | DT_FLOAT → Key 1 | `op_host/chamfer_distance_grad_tiling.cpp`: `Init`, `RunKernelTiling`, `TilingPrepareForChamferDistanceGrad` | `op_host/chamfer_distance_grad_tiling.h`: `ChamferDistanceGradTilingData` | `op_kernel/chamfer_distance_grad.cpp`: `chamfer_distance_grad`, Key 1 | `op_kernel/chamfer_distance_grad.h`: constructor, `process`, `main_cal_func_d1/d2`, `muls_template` | `op_host/chamfer_distance_grad_def.cpp`; 两份平台 binary json；`op_kernel/CMakeLists.txt`; `scripts/kernel/binary_config/ascendc_config.json` |

## Kernel 实现

### 公共执行流程

唯一正式 Route 的核心节点先绑定 8 路 GM Tensor，根据 blockIdx 切连续点任务，block 0 将两个完整输出清零并执行全核 `SyncAll`。每核一次性申请六个 Queue buffer，先运行 d1 阶段，再运行 d2 阶段：每阶段把本侧坐标、梯度和索引成块搬入 UB，将梯度原地乘 2；逐点随机读取另一侧最近邻坐标，形成二维差分梯度；本侧连续贡献保存在 `d_local` 后成块原子写回，另一侧贡献取负后逐点原子散射。两个阶段复用同一组 UB buffer，最后统一释放。

### Route R-001：FP32 原子散射

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `loss/chamfer_distance_grad/op_host/op_api/aclnn_chamfer_distance_backward.cpp` | `aclnnChamferDistanceBackwardGetWorkspaceSize` | 非空后对四个浮点输入调用 `Cast(..., DT_FLOAT, ...)` | 生成 FP32 核心输入 |
| Host API | `loss/chamfer_distance_grad/op_host/op_api/chamfer_distance_backward.cpp` | `l0op::ChamferDistanceGrad` | `ADD_TO_LAUNCHER_LIST_AICORE(ChamferDistanceGrad, ...)` | 登记核心节点 C0 |
| Tiling | `loss/chamfer_distance_grad/op_host/chamfer_distance_grad_tiling.cpp` | `SetTilingKeyMode` | `dType_str == ge::DT_FLOAT` | `SetTilingKey(1)` |
| Router | `loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.cpp` | `chamfer_distance_grad` | `TILING_KEY_IS(1)` | 实例化 `ChamferDistanceGrad<float>` 并调用 `process()` |

#### Device 数据流

##### 函数级调用链

```text
l0op::ChamferDistanceGrad -> ADD_TO_LAUNCHER_LIST_AICORE
  -> chamfer_distance_grad(..., workspace, tiling_data)
    -> ChamferDistanceGrad<float>::constructor
      -> InitOutput(grad_xyz1_gm, 2*B*N, 0) [block 0]
      -> InitOutput(grad_xyz2_gm, 2*B*N, 0) [block 0]
      -> SyncAll()
      -> InitBuffer(six queues)
    -> process
      -> main_fuc_of_mode_zero(start_task_id)
        -> init_ub_mode_zero
        -> compute_mode_zero
          -> main_cal_func_d1 (chunk loop + tail)
            -> muls_template -> per-point nearest lookup -> atomic scatter grad_xyz2
          -> atomic contiguous add grad_xyz1
          -> compute_mode_zero_with_d2 (chunk loop + tail)
            -> main_cal_func_d2 -> per-point nearest lookup -> atomic scatter grad_xyz1
          -> atomic contiguous add grad_xyz2
          -> FreeTensor(six buffers)
```

##### 源码等价伪码

```text
# 伪码；T=FP32，P=per_ub_size，global range is [start, start+count)
if blockIdx == 0:
    gradXyz1[0:2*B*N] = 0
    gradXyz2[0:2*B*N] = 0
SyncAll()

allocate xyzLocal[2P], gradLocal[P], idLocal[P], dLocal[2P]
allocate scatterLocal[8], pointLocal[8]

for side in [d1, d2]:
    for each chunk [task, task+len), len <= P:
        batch0 = task / N
        point0 = task % N
        copy contiguous side.xyz[batch0,point0 : +len] -> xyzLocal[0:2len]
        copy contiguous side.grad[batch0,point0 : +len] -> gradLocal[0:len]
        copy contiguous side.idx [batch0,point0 : +len] -> idLocal[0:len]
        gradLocal *= 2                         # in place, vector Muls

        for k in [0,len):
            global = task + k
            batch = global / N
            nearest = idLocal[k]
            copy other.xyz[batch,nearest,:] -> pointLocal[0:2]
            delta = (xyzLocal[2k:2k+2] - pointLocal[0:2]) * gradLocal[k]
            dLocal[2k:2k+2] = delta
            AtomicAdd(other.grad[batch,nearest,:], -delta)

        if tail chunk:
            zero dLocal[2len : align_up(2len, 8)]
        AtomicAdd(side.grad[batch0,point0 : +len,:], dLocal[0:2len])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `xyz1/xyz2` | base 0；chunk `2*(batch*N+point)`；nearest `2*(batch*N+idx)` | 各 `2*B*N` 元素 | FP32, `[B,N,2]` ND | ACLNN Cast | d1/d2 chunk DMA 与随机点 DMA | executor/调用方；C0 全程只读 | chunk `DataCopyPad`；随机点每次 8 B 有效数据 |
| GM `gradDist1/2` | `batch*N+point` | 各 `B*N` 元素 | FP32, `[B,N]` ND | ACLNN Cast | d1/d2 chunk DMA | executor；C0 全程只读 | 每 chunk 有效 `len*4` B |
| GM `idx1/2` | `batch*N+point` | 各 `B*N` 元素 | INT32, `[B,N]` ND | public input/Contiguous | d1/d2 chunk DMA | 调用方/executor；C0 全程只读 | 每 chunk有效 `len*4` B |
| GM `gradXyz1/2` 中间输出 | 连续贡献 `2*(batch*N+point)`；散射 `2*(batch*N+idx)` | 各 `2*B*N` 元素 | FP32, `[B,N,2]` ND | block 0 初始化，所有 block AtomicAdd | 后置 Cast | executor-owned；C0 后有效 | 输出全局共享；AtomicAdd 解决写冲突 |
| `in_queue_xyz` | Queue local base | `2P` FP32 = `8P` B | FP32, 点坐标交错 | chunk `DataCopyPad` | 每点标量读取 | `TPipe`; d1 与 d2 复用，结束释放 | VECIN, depth 1 |
| `in_queue_grad_dist` | Queue local base | `P` FP32 = `4P` B | FP32 | chunk DMA 后 `Muls` 原地写 | 每点标量读取 | d1/d2 复用 | VECIN, depth 1，原地覆盖 |
| `in_queue_id` | Queue local base | `P` INT32 = `4P` B | INT32 | chunk DMA | 最近邻地址生成 | d1/d2 复用 | VECIN, depth 1 |
| `out_queue_d` | Queue local base | `2P` FP32 = `8P` B | FP32 | 每点 `SetValue` | 连续 Atomic `DataCopyPad` | d1/d2 复用 | VECOUT, depth 1；尾对齐区补零 |
| `out_queue_xy_zero_block` | Queue local base | `t_per_block=8` FP32 = 32 B | FP32 | 每点写前两个元素 | 散射 Atomic `DataCopyPad` | 每点复用 | VECOUT, depth 1；其余 lane 初始 Duplicate 为 0 |
| `out_queue_block` | Queue local base | `t_per_block=8` FP32 = 32 B | FP32 | nearest 点 `DataCopyPad` | 标量读取前两个元素 | 每点复用 | VECOUT, depth 1 |
| core workspace | Kernel 参数 `workspace` | Tiling 申报 16 MiB | byte/system | caller/executor | body 不读取 | stream 任务生命周期 | 无业务 slice/alias |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  xyzQ["in_queue_xyz · VECIN · depth=1 · 8P B"]:2
  gradQ["in_queue_grad_dist · VECIN · depth=1 · 4P B"]:1
  idQ["in_queue_id · VECIN · depth=1 · 4P B"]:1
  dQ["out_queue_d · VECOUT · depth=1 · 8P B"]:2
  scatterQ["out_queue_xy_zero_block · VECOUT · depth=1 · 32 B"]:3
  pointQ["out_queue_block · VECOUT · depth=1 · 32 B"]:3
```

##### 字节账与复用顺序

```text
P = floor((UB_platform - 1024) / 768) * 32 elements
UB_total = 8P + 4P + 4P + 8P + 32 + 32
         = 24P + 64 bytes
24P <= UB_platform - 1024
therefore UB_total <= UB_platform - 960
dynamic_ub = N/A（AIV Queue 静态 TPipe 分配）
workspace_total(core declaration) = 16 * 1024 * 1024 bytes
workspace_used_by_body = 0 bytes
=> P > 0 是循环安全的隐含平台条件
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `in_queue_grad_dist` | d1/d2 chunk DMA，随后 `muls_template` 原地乘 2 | 同 chunk 每点标量计算 | 下一 chunk或从 d1 切换 d2 | MTE2→V、V→S hard event；阶段串行 | 当前 chunk 的所有点读完后才覆盖 |
| `out_queue_d` | 每点 `SetValue` | chunk 末 Atomic `DataCopyPad` | 下一 chunk；d1 结束后供 d2 使用 | V→MTE3、MTE3→S hard event | 写回完成后才能覆盖；尾部先补零 |
| `out_queue_xy_zero_block` | 每点负贡献写入前两 lane | 索引散射 Atomic copy | 下一点 | V→MTE3、MTE3→S | 前一次原子搬运完成后覆盖 |
| `out_queue_block` | 每点 nearest 坐标 DMA | 标量读取前两 lane | 下一点 | MTE2→V，d1 另有 MTE3→MTE2 | 当前点坐标读取完成后覆盖 |
| 全部六 Queue | d1 阶段 | d1 | d2 阶段 | d1 chunk及写回循环全部完成后函数顺序进入 d2 | 单核串行复用；Kernel 末统一 FreeTensor |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输出初始化 | `InitOutput`, `SyncAll` | block 0 | 所有 block 的首次 AtomicAdd | 两份完整 GM 输出 | 保证原子累加从 0 开始 |
| chunk CopyIn | `DataCopyPad`, `PipeBarrier<PIPE_MTE2>`, MTE2→V | GM DMA | `Muls` 与标量读取 | xyz/grad/id LocalTensor | 有效字节分别 `2len*4`, `len*4`, `len*4`；padding 不进入 `k<len` 标量循环 |
| 梯度缩放 | `Muls`, `PipeBarrier<PIPE_V>` | `gradLocal` | 标量读 `g` | 同一 UB buffer | FP32 原地乘 2；mask 64，repeat最多255，余 repeat/tail 分开 |
| nearest 读取 | MTE2→V；d1 还使用 MTE3→MTE2 | 单点 8 B DMA | 标量 `GetValue(0/1)` | `pointLocal` | 索引无 Device 边界检查 |
| 标量计算 | S↔V hard events | 标量 `(x-other)*g` | LocalTensor `SetValue` | `dLocal/scatterLocal` | 核心全程 FP32；FP16 public 路径只在核心前后 Cast |
| 散射写回 | `SetAtomicAdd`, `DataCopyPad`, `SetAtomicNone`, MTE3→S | `scatterLocal[0:2]` | GM 输出 | 共享 nearest 位置 | 多 block/多点冲突，默认非确定性 |
| 连续写回 | 同样使用 AtomicAdd | `dLocal[0:2len]` | GM 输出 | 当前连续任务段 | 核间任务段不重叠，但与另一阶段散射可能命中同位置 |
| 尾 chunk | `align_up(2len,t_per_block)` 后 `SetValue(...,0)` | 标量补零 | Atomic DataCopyPad | `dLocal` 最后一个 32 B block | 仅搬运 `2len*sizeof(T)` 有效字节；补零避免对齐 lane 残值参与块级原子行为 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/chamfer_distance_grad/op_host/chamfer_distance_grad_tiling.cpp` | `ChamferDistanceGradTiling::Init/RunKernelTiling` | 从 xyz1 shape、AIV core 与平台 UB 生成分核、POD、Key 和 16 MiB workspace |
| Kernel entry/router | `loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.cpp` | `chamfer_distance_grad`, Key 1 | Key 1 映射到 `ChamferDistanceGrad<float>`；未知 Key 无 body/错误返回 |
| Device body/helper | `loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h` | constructor, `compute_mode_zero`, `main_cal_func_d1/d2`, `muls_template` | 输出清零同步、六区 UB、两阶段连续加与索引散射、Muls repeat 切分、尾部补零 |

## 算子特定优化

### Trick T-001：连续贡献与索引散射直接汇入最终输出

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：所有非空任务；输出已由 block 0 清零并完成 `SyncAll`。
- Golden 来源：[`op_kernel/chamfer_distance_grad.h:367-383`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h#L367-L383)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
PipeSync<AscendC::HardEvent::V_MTE3>();
SetAtomicAdd<T>();
DataCopyPad(
    grad_xyz2_gm[cur_batch * this->num * doub + cur_ind * doub], xy_zero_block_local,
    {1, static_cast<uint32_t>(doub * sizeof(T)), 0, 0, 0});
SetAtomicNone();
```

#### 机制与设计原因

每个点的另一侧贡献只占两个标量，直接对最终 GM 坐标执行 AtomicAdd；本侧连续 `dLocal` 也直接原子成块写最终输出。这样不需要按输入点物化散射临时数组，也不需要第二个归约 Kernel。原子目标由 `batch*N+idx` 产生，长度固定为 `2*sizeof(T)`；`SyncAll` 确保清零先于任一累加。代价是冲突加法次序不固定。

#### 迁移边界

- 适用：输出为加法归约、索引可能冲突且目标 dtype 支持 AtomicAdd 的 AIV 算子。
- 前提：输出初始化与所有累加之间有全核可见同步；索引合法；原子目标和传输长度正确。
- 失效：需要严格固定加法顺序、目标 dtype 不支持原子加或无法安全全局清零时，必须改用显式分段归约/确定性实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

在 `SyncAll` 前由其他 block 开始累加会与 block 0 清零竞争并丢失贡献；固定源码以 `InitOutput` 后的 `SyncAll` 给出可验证反例边界。

### Trick T-002：梯度在输入 Queue 内原地倍乘并按 255 repeat 分段

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：每个 d1/d2 chunk 完成 CopyIn 后，`calCount=align_up(len,t_per_block)`。
- Golden 来源：[`op_kernel/chamfer_distance_grad.h:398-425`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h#L398-L425)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int32_t mask = unit / sizeof(T);
int32_t repeats = calCount / mask;
int32_t loop = repeats / max_repeat;
Muls(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, mask, repeats_tail, {1, 1, 8, 8});
```

#### 机制与设计原因

`dstLocal` 与 `srcLocal` 都是 `grad_dist_local`，因此倍乘 2 不另占 UB；FP32 下 mask 为 `256/4=64` 元素，完整段用 `repeat=255` 和 `{blockStride=1, repeatStride=8}` 覆盖连续数据，再处理余 repeat 与 mask tail。参数分别来自类型宽度、硬编码 Vector 单元 256 B、最大 repeat 255 和当前对齐 chunk 长度。

#### 迁移边界

- 适用：后续不再需要未缩放输入，且算子可接受原地 Vector 写回的逐元素缩放。
- 前提：MTE2→V 同步完成；buffer 容量覆盖 `calCount`；mask/repeat stride 与 dtype 匹配。
- 失效：原输入还会被其他消费者读取或输入、输出别名不被 primitive 支持时，必须使用独立计算 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不按 255 repeat 上限切分会超出固定源码采用的 Vector API 参数边界；固定仓未提供另一条 fallback Route。

### Trick T-003：六分区保守切块与跨 d1/d2 生命周期复用

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：`ub_size=platformUB-1024`，构造任意非空 core。
- Golden 来源：[`op_kernel/chamfer_distance_grad.h:77-85`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h#L77-L85)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t ub_num = 6;
uint64_t per_ub_size = tiling_data->ub_size / ub_num / sizeof(float) / block_bytes * block_bytes;
pipe.InitBuffer(in_queue_xyz, 1, doub * per_ub_size * sizeof(T));
pipe.InitBuffer(out_queue_d, 1, doub * per_ub_size * sizeof(T));
```

#### 机制与设计原因

Host 先保留 1024 B，Device 再按 6 个 FP32 等价区计算 P 并向下对齐到 32 元素。两个双宽坐标/梯度区各占两份预算，grad 与 id 各占一份，两个单 block buffer只额外占 64 B；因此 FP32 总量严格落在平台 UB 内。d1 完成后同一物理 Queue 直接处理 d2，避免为两个方向各复制一套 UB。

#### 迁移边界

- 适用：两阶段串行且 buffer shape/dtype 可统一的双向梯度或对称算子。
- 前提：前一阶段所有 DMA/Vector/Atomic写回完成后才覆盖；P 必须非零；字节账按真实 depth 计算。
- 失效：两阶段需要流水重叠、dtype/容量不同或存在跨阶段未消费数据时，必须拆分 buffer 或增加同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

按六个 `P*sizeof(float)` 直接估算会漏掉 xyz 与 d 的双宽容量；正确账目是 `24P+64` B。固定仓没有 UB 不足时的替代 Route。

## 模板与实现对比

| 对比维度 | R-001 FP32 原子散射 | 设计选择依据 |
|---|---|---|
| 命中条件/优先级 | ACLNN 非空路径统一 Cast 到 FP32，Key 1 唯一可达 | Key 2 只在 router 存在，不被公开入口与 float32 binary config选中 |
| 模板与 dtype | `ChamferDistanceGrad<float>` | 内部统一 FP32，FP16 仅在 ACLNN 边界转换 |
| 循环与分核 | `B*N` 连续任务均分 AIV 核，每核按 P chunk，d1 后 d2 | 同一任务映射同时支持连续写和逐点 scatter |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 六个 depth=1 Queue；标量临时变量；无 dynamic UB、通信 window | 以 Queue 承载 DMA/Vector/Atomic 流程 |
| 内存总占用与复用 | `24P+64` B；d1/d2 完全复用 | Host 预留 1024 B，Device 六分区保守切块 |
| 搬运/Vector/SIMT/NDDMA/通信 API | `DataCopyPad`, `Muls`, `InitOutput`, `AtomicAdd`；无 SIMT/NDDMA/通信 | 连续输入成块搬运，nearest 坐标逐点随机搬运 |
| 同步、精度与 tail | HardEvent/PipeBarrier/SyncAll；FP32 核心；尾 `dLocal` 补零 | 保护跨 pipe 可见性和原子写回的块尾 |
| fallback/失败边界 | 空输入提前成功；非法 dtype失败；shape/索引/UB 隐含边界无 fallback | 固定源码只有单一公开核心实现族 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；`ascend910b` | 支持；`ascend910_93` | 不支持；公开文档明确标 `×`，且无 OpDef/binary/CMake 接线 |
| Host/Launcher | 与 A3 共用同一 Host/Tiling；平台提供实际 AIV 核数和 UB | 与 A2 共用同一 Host/Tiling；平台提供实际 AIV 核数和 UB | 无公开 Route |
| Kernel/资源 | 同一 `ChamferDistanceGrad<float>`；P 随平台 UB、blockDim 随 AIV 核数 | 同一 body，资源公式相同 | 无 Kernel 构建目标；不生成 `_950` 卡 |

## 限制与失败边界

- ACLNN 文档将 `xyz2` 写为 `[B,M,2]`，但 proto 规定与 `xyz1` 同 shape，Tiling/Kernel 也仅保存一个 `N` 并把所有第二侧 extent/stride写成 N；本固定实现只闭合 `M=N`。
- 第一段不校验 rank、shape 一致性、浮点输入输出 dtype 一致性或 idx 值域。越界索引、不同 N、不同浮点 dtype 或不匹配输出 shape 均无源码 fallback。
- `TilingChamferDistanceGrad` 调用 `tilingObject.Init()` 后没有检查其返回值便继续 `RunKernelTiling()`；Prepare 阶段能拒绝无效 core/UB，但其它 Init 失败传播在固定源码中不完整。
- Tiling 的 `Q-1` 与 `ubSizePlatForm-1024` 使用无符号算术；公开 ACLNN 空输入绕过 `Q=0`，但 Host/Tiling 直接调用路径仍需调用方保证非空且平台 UB 足够令 P 非零。
- Key 2 / `ChamferDistanceGrad<half>` 是源码中的 Device 分支，但公开 ACLNN 前置 Cast、OpDef 与两份 binary config 都使它不可达，不能作为 ACLNN FP16 Route 复用。
- Kernel unknown key 分支静默不执行 body，没有显式错误或输出初始化；正式 Host 只产生 Key 1。
- 输出归约依赖 AtomicAdd，默认非确定性。文档称可通过 `aclrtCtxSetSysParamOpt` 开启确定性，但其替代调度/实现不在固定算子源码中，不能据此推导确定性 Kernel。
- 核心 Tiling 固定申报 16 MiB workspace，body 不读 `workspace`；ACLNN 总 workspace 还包含框架前后处理，具体 offset、对齐、别名和清理由外部 executor 管理，固定仓不可见。
- 950 被公开文档明确标为不支持，且无 `ascend950` 构建接线；不得由同源 Kernel 文件推测支持。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/chamfer_distance_grad/docs/aclnnChamferDistanceBackward.md` | `aclnnChamferDistanceBackward*` 文档 | 公开接口、dtype/shape、平台支持、默认非确定性 |
| 入口实现 | `loss/chamfer_distance_grad/op_host/op_api/aclnn_chamfer_distance_backward.cpp` | `CheckParams`, `aclnnChamferDistanceBackwardGetWorkspaceSize`, `aclnnChamferDistanceBackward` | 校验、空域、Contiguous/Cast/ViewCopy、executor workspace与运行 |
| 入口实现 | `loss/chamfer_distance_grad/op_host/op_api/chamfer_distance_backward.cpp` | `l0op::ChamferDistanceGrad` | 中间输出分配与 AICore launcher登记 |
| Host/Tiling/Launcher | `loss/chamfer_distance_grad/op_host/chamfer_distance_grad_tiling.h` | `ChamferDistanceGradTilingData`, `ChamferDistanceGradCompileInfo` | POD 字段顺序和 compile info |
| Host/Tiling/Launcher | `loss/chamfer_distance_grad/op_host/chamfer_distance_grad_tiling.cpp` | `Init`, `RunKernelTiling`, `TilingPrepareForChamferDistanceGrad` | Key、分核、UB、workspace、schedule mode |
| Kernel entry/body | `loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.cpp` | `chamfer_distance_grad` | Key 1/2 router与模板实例 |
| Kernel entry/body | `loss/chamfer_distance_grad/op_kernel/chamfer_distance_grad.h` | `ChamferDistanceGrad<T>` 全部方法 | GM/UB、循环、DMA、Vector、Atomic、同步与尾块 |
| 构建接线 | `loss/chamfer_distance_grad/op_host/chamfer_distance_grad_def.cpp` | `ChamferDistanceGrad` OpDef | FP32 核心 dtype和 A2/A3平台注册 |
| 构建接线 | `loss/chamfer_distance_grad/op_host/config/ascend910b/chamfer_distance_grad_binary.json` | `ChamferDistanceGrad` binary entry | A2 FP32 binary输入输出 |
| 构建接线 | `loss/chamfer_distance_grad/op_host/config/ascend910_93/chamfer_distance_grad_binary.json` | `ChamferDistanceGrad` binary entry | A3 FP32 binary输入输出 |
| 构建接线 | `loss/chamfer_distance_grad/op_kernel/CMakeLists.txt` | `add_kernel_sources` | 仅 `ascend910b/ascend910_93` 构建 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `ChamferDistanceGrad` entry | 全局 compute units 与 auto_sync |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六个 uint64 字段按声明顺序由 Host 保存、Device 读取 | 不可猜测宏展开 offset；`ub_size` 是字节，其余任务字段是元素/核数 | `ChamferDistanceGradTilingData` |
| 调用顺序 | 空域检查→连续化/FP32化→核心→输出回 Cast/ViewCopy；核心内部 d1→d2 | 不得交换输出清零/SyncAll/Atomic，也不得在 d1 未写回时覆盖 Queue | ACLNN GetWorkspaceSize, `compute_mode_zero` |
| workspace/内存 | 核心 UB 为 `24P+64` B；业务 body 不使用 workspace | Queue depth=1；P 按 `sizeof(float)` 与 32 元素对齐；16 MiB 系统 workspace申报不可当作业务中间量 | constructor, `RunKernelTiling` |
| 同步 | `SyncAll` 保护初始化；HardEvent/PipeBarrier 保护 MTE2/V/MTE3/S | AtomicAdd 前后与 buffer 下一次覆盖的依赖不能删除 | constructor, `PipeSync`, d1/d2 helpers |
| 精度与尾块 | 公开核心 FP32；grad 原地乘 2；尾 dLocal 补零 | FP16 只在 ACLNN 边界转换；Muls mask/repeat/stride与 `t_per_block` 不得错配 | ACLNN Cast, `muls_template`, tail branches |
| 失败边界 | 空输入成功短路；非法 dtype失败；950不支持 | 不得把 dormant Key 2、未知 Key、shape/idx 未校验或文档确定性开关推成已闭合实现 | `CheckParams`, router, platform docs/config |
