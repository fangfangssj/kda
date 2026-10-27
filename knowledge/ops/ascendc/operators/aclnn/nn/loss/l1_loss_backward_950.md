---
type: Operator
title: "L1LossBackward 950 ACLNN 算子知识"
description: "说明 aclnnL1LossBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/docs/aclnnL1LossBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_api/aclnn_l1_loss_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_api/aclnn_l1_loss_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_api/l1_loss_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_host/arch35/l1_loss_grad_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_kernel/arch35/l1_loss_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_kernel/arch35/l1_loss_grad_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_kernel/l1_loss_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_host/l1_loss_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_host/config/ascend950/l1_loss_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/broadcast/broadcast_tiling.cpp
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/node.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/buffer.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/nnopbase/composite_op/aclnn_engine/op_executor.cpp
    role: implementation
created_at: "2026-08-25T17:16:52Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# L1LossBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnL1LossBackwardGetWorkspaceSize` + `aclnnL1LossBackward`，公开两段式 ACLNN，无版本后缀。 |
| 计算语义 | `gradInput = gradOutput * sign(self-target) * scale`；`scale=1/N` 当 `reduction=mean`，否则为 1；相等位置的 `sign` 为 0。 |
| 输入输出 | 三输入广播到共同 ND shape，公开维数 1–8；核心 dtype 为 FP16/BF16/FP32，输出可由 ACLNN 尾部 Cast 到 `gradInput` dtype。 |
| 执行类型 | 核心节点 `L1LossGrad` 为单 Kernel `AIV_VECTOR`；ACLNN 的 Contiguous/BroadcastTo/Cast/ViewCopy 是框架前后处理节点。 |
| Kernel 路线 | R-001 `schMode=201,inputGradsIsScalar=0` 张量梯度；R-002 `schMode=201,inputGradsIsScalar=1` 标量/单元素梯度。 |
| 核心资源 | GM 输入输出；单个 `TBuf<VECCALC>` 切成 DAG 编译期 buffer-id 池；128 B 对齐 tile；16 MiB 系统 workspace 声明。 |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 launcher→950 Host/Tiling/POD→模板 key→AIV entry→BroadcastOneDimSch→L1LossGrad DAG/Vector primitive→950 构建接线均闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `B` 为 `gradOutput`、`self`、`target` 的广播后 shape，`N=∏B[i]`。核心逐元素计算：

```text
s =  1, self > target
     0, self == target
    -1, self < target
scale = 1/N  (reduction == 1 / "mean")
        1    (reduction == 0 / "none" 或 2 / "sum")
gradInput = cast_to_output(cast_fp32(gradOutput) * scale * s)
```

`reduction=sum` 的 L1 前向导数与 `none` 一样不缩放；`mean` 的 `N` 来自传给核心 `L1LossGrad` 的 `label` storage shape。ACLNN 已把 `target` 广播到 `B`，因此这里等于广播输出元素数。FP16/BF16 路线将梯度转为 FP32、在 FP32 中做缩放和乘法，再用 `RoundMode=1` 降回输入类型；比较本身在原输入 dtype 上执行。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | ND，1–8 维，可非连续；与另两输入可广播 | FP16/BF16/FP32，经互推导后落入 950 支持集 | 上游梯度；950 RegBase 且 view shape 为 scalar 时不预先 BroadcastTo，由核心标量路线复用。 |
| `self` | ND，1–8 维，可非连续 | FP16/BF16/FP32 | 预测值；先 Contiguous，再 BroadcastTo 到 `B`，再按 promote dtype Cast。 |
| `target` | ND，1–8 维，可非连续 | FP16/BF16/FP32 | 标签；处理顺序同 `self`。 |
| `reduction` | Host 标量 | `int64_t` | 仅 0/1/2，L0 映射为 `"none"/"mean"/"sum"`；只有 `mean` 产生 `1/N`。 |
| `gradInput` | 必须等于 `B`，ND，可非连续 | 可由 promote dtype Cast 到该 dtype | ViewCopy 提交最终结果；空输入时 ACLNN 返回成功且 workspace 为 0，不建立核心计算。 |

### 入口与版本

公开文档和头文件只定义无后缀 `aclnnL1LossBackward` 两段式版本。第一段创建 executor、校验并建图，第二段将用户 workspace 与 stream 交给 `CommonOpExecutorRun`。逻辑语义是 L1 loss 的 backward，不能与 `L1Loss` 前向、`SmoothL1LossBackward` 或其他 Grad 合并。

950 公开支持由文档的 Ascend 950PR/950DT 行、`OpDef::AICore().AddConfig("ascend950")`、ascend950 binary config、`SUPPORT_COMPUTE_UNIT "ascend950"` 和 kernel CMake 的 `COMPUTE_UNITS ascend950` 共同支撑。该固定源码没有更高版本号或同平台新入口。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序如下：

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize/executor`，`CheckParams` 校验四个 tensor 非空、`reduction∈[0,2]`、三输入 promote 可行、promote 结果可 Cast 到输出、维数不超过 8、广播 shape 与 `gradInput` 相同。
2. 任一计算输入为空时令 `workspaceSize=0`，释放 executor 给调用者并成功返回；此分支不加入 L1LossGrad kernel。
3. 对三输入分别调用 `l0op::Contiguous`。
4. 计算共同 `broadcastShape2=B`。950 RegBase 的 scalar `gradOutput` 保留标量；其余 `gradOutput` 以及 `self/target` 在 shape 不等于 `B` 时调用 `BroadcastTo`。
5. 求三输入 promote dtype，并按需对每路调用 `Cast`。
6. `l0op::L1LossGrad` 分配 shape `B` 的中间输出，并通过 `ADD_TO_LAUNCHER_LIST_AICORE` 加入核心节点，属性从数值 reduction 映射到字符串。
7. 按需把核心结果 Cast 到 `gradInput` dtype，再用 `ViewCopy` 写入可能非连续的用户输出。

任一步返回空指针都会立即返回相应 ACLNN 错误，不再加入后续节点。`CheckFormat` 只对 `gradOutput` 的 FRACTAL_NZ 发出精度警告；公开接口声明 ND，核心 OpDef 也只注册 ND。

### 参数、Workspace 与生命周期

`uniqueExecutor` 拥有前后处理临时 tensor、核心中间输出与 launcher；第一段末尾用 `GetWorkspaceSize()` 通过图的 live-range allocator 计算总 workspace，再 `ReleaseTo(executor)` 转移所有权。具体 Contiguous/BroadcastTo/Cast/ViewCopy 临时切片 offset 由通用 executor 图分配器决定，固定算子源码未写死，不能把它们与核心的 16 MiB 声明简单相加。

核心 Broadcast tiling 始终把其 workspace 条目设为 `16*1024*1024` 字节。entry 要求 `workspace` 和 `GetUserWorkspace(workspace)` 非空，调用 `SetSysWorkspace`，但 L1 DAG 本身不读写 `userWS`。第二段校验“workspaceSize>0 时 workspace 不可空”，更新 tensor 地址、绑定 stream，然后按拓扑序逐 launcher 执行；首个失败立即中止后续节点。

### 启动图与失败传播

```text
可选 Contiguous(gradOutput/self/target)
  -> 可选 BroadcastTo(gradOutput/self/target)
  -> 可选 Cast(三输入到 promote dtype)
  -> L1LossGrad [单 AIV_VECTOR kernel l1_loss_grad<201, scalarFlag>]
  -> 可选 Cast(核心结果到 gradInput dtype)
  -> ViewCopy(到用户 gradInput)
```

核心节点的 `blockDim=BroadcastOneDimTilingData.blockNum`，POD 为同一结构，输入/输出均为连续 GM tensor。所有节点由一个 executor 在调用者传入的同一 stream 上按排序后的图顺序启动，GM/workspace 临时 tensor 的 producer→consumer 可见性由该 stream 顺序保证；任一 launcher 非成功返回会 `break`，所以不会消费失败节点的输出。中间 tensor 归 executor/workspace 图所有，执行结束或 executor 销毁后失效。

## Route 与参数设计

### 分核与资源计算

ACLNN 前处理后，三路核心输入除保留的标量 `gradOutput` 外均为连续且具有共同 shape `B`。`DimensionCollapse` 按每轴广播 flag 合并相邻轴，因此输出压成单轴 `dimLength=N`；scalar 输入压成长度 1。DAG 有 3 个输入、1 个输出且 `VarSize=1`，不满足 Advance 模板的 `VarSize<=0`，故固定选择 `DoOneDimOpTiling`/mode 201。

Host template registry 以升序 `std::map` 遍历 priority；本算子只注册 `COMMON_TILING_PRIORITY=0` 的 `L1LossGradTiling`，且 `IsCapable()` 恒为 true。因此 dtype/shape/资源校验一旦通过就由该唯一模板产出 key，不存在较低优先级 fallback。

设 Host 的切片除数 `A=OpDag::GetBufferNum<true,true>()`、Device 实际分配 slot 数 `B=OpDag::BufferNum=OpDag::GetBufferNum<true,false>()`、`S=OpDag::MaxDtypeBytes`、平台可用 UB 为 `U`：

```text
ubFormerBytes = floor((U / A) / 128) * 128
ubFormer = ubFormerBytes / S
ubOuter = ceil(N / ubFormer)
ubTail = (N % ubFormer == 0) ? ubFormer : N % ubFormer
blockFormer = ceil(ubOuter / aivCoreNum)
blockTail = (ubOuter % blockFormer == 0) ? blockFormer : ubOuter % blockFormer
blockNum = ceil(ubOuter / blockFormer)
```

若初算 `blockNum<aivCoreNum/2` 且 `ubFormer*S*A>8 KiB`，Host 尝试以约半数核重新缩小 `ubFormer`：每核元素量向上对齐 128 B，并设每核下限为按 `A` 分摊的 8 KiB；随后重算上述分核字段。所有长度单位为元素，`ubFormerBytes`/8 KiB/128 B 是字节。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 张量梯度 | `GET_TPL_TILING_KEY(201,0)` | `CheckGradsIsScalar` 归一化后的 grads storage shape 不是 `[1]`；ACLNN 已连续化并广播，合轴长度为 1；dtype 为 FP16/BF16/FP32。 | grads 为单元素时进入 R-002；dtype/shape/reduction 校验失败则终止。 |
| 2 | R-002 标量梯度 | `GET_TPL_TILING_KEY(201,1)` | grads storage shape 经 `EnsureNotScalar` 后恰为 `[1]`；其余条件同 R-001。 | 不存在第三条 Host 可达核心路线；资源或 POD 生成失败返回 GRAPH_FAILED。 |

```text
CheckGradsIsScalar(): EnsureNotScalar(grads.storageShape)==[1] ? inputGradsIsScalar=1 : 0
CaluateReduceElts(): reduction=="mean" ? reduceElts=逐维除以 label.storageShape[i] : 1
DoTiling(dtype): select L1LossGrad[Scalar]Cast<half|bfloat16_t|float>::OpDag
  -> BroadcastBaseTiling(..., KERNEL_TYPE_NDDMA).DoTiling()
  -> contiguous + collapsed rank==1 + VarSize==1 => schMode=201
  -> SetScalar(reduceElts)
  -> key=GET_TPL_TILING_KEY(201,inputGradsIsScalar)
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 张量梯度 | `(201,0)` | 950；核心三输入同 dtype FP16/BF16/FP32；grads 非单元素；连续共同 shape | 单节点 `l1_loss_grad<201,0>` | `L1LossGradCast<DTYPE_PREDICT,float>::OpDag`→`BroadcastOneDimSch` | 单 TBuf liveness pool；MTE2/V/MTE3 Mutex；tile ping/pong | 尾核用 `blockTail`，最后 tile 用 `ubTail`；失败无 fallback。 |
| R-002 标量梯度 | `(201,1)` | 950；同 dtype；grads 单元素；predict/label/output 为共同 shape | 单节点 `l1_loss_grad<201,1>` | `L1LossGradScalarCast<DTYPE_PREDICT,float>::OpDag`→同 scheduler | grad 用 Duplicate；标量 tile 前两次填充 ping/pong 后复用；其余同 R-001 | 同 R-001。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 标量保留 | `IsRegbase() && gradOutput.viewShape.IsScalar()` | 避免 ACLNN 预先物化整张广播梯度，交给核心 Duplicate | R-002 | 非标量或非 RegBase 走 BroadcastTo 后 R-001 | `aclnn_l1_loss_backward.cpp::aclnnL1LossBackwardGetWorkspaceSize` |
| mode 201 | collapsed rank=1、全部连续且 `OpDag::VarSize>0` | mode 202 POD 不携带 DAG Var；mode 201 的 16 B `scalarData` 传 scale | 两 Route | 合轴/连续条件失败会进入公共其他模板，但该情况经公开 ACLNN 前处理不可达 | `BroadcastBaseTiling::DoTiling` |
| UB 存活数 | `aliveNum=A=GetBufferNum<true,true>()>0` 且 floor 对齐后 `ubFormer>0` | Host 用 cache-broadcast 编译期存活数切 tile；Device 以 `B=BufferNum` 分配单 TBuf | 两 Route | GRAPH_FAILED | `GetOneDimUBFormer`、`BroadcastOneDimSch::Init` |
| 128 B tile | `floor((U/A)/128)*128` | DMA tile 与 pool 子区按 128 B 对齐 | 两 Route | 对齐后为 0 则 GRAPH_FAILED | `GetOneDimUBFormer` |
| dtype 一致 | grads/predict/label/y 完全同 dtype，且为 FP16/BF16/FP32 | entry 只按 `DTYPE_PREDICT` 实例化一个 DAG 类型 | 两 Route | GRAPH_FAILED/ACLNN 参数错误 | `L1LossGradTiling::GetShapeAttrsInfo` |
| 非空 | API 任一输入为空提前成功；Tiling 对 label 的零维长度报错 | `mean` 不能除以 0，Device 循环要求 `N>0` | 两 Route | 不启动核心或 GRAPH_FAILED | `CaluateReduceElts`、ACLNN 空 tensor 分支 |

### POD/ABI

`BroadcastOneDimTilingData` 按固定字段定义与自然 8 字节对齐为 56 字节；`ubSplitAxis` 在 one-dim producer 中没有赋值，scheduler 只将它用于日志，不参与地址或循环。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `BroadcastOneDimTilingData` | `scalarFlag` | `int32_t`/32 | offset 0 | 输入位图 | `GetOneDimScalarFlag`：对长度 1 的输入置 `1<<inputIndex` | `CopyIn` 判断从 GM 搬 tile 还是读一个值后 Duplicate | 位0/1/2对应 grads/predict/label。 |
| 同上 | `ubSplitAxis` | `int32_t`/32 | offset 4 | 轴 | one-dim producer 未赋值 | 仅 `RUN_LOG` 读取 | inactive；不得依赖其值。 |
| 同上 | `ubFormer` | `int32_t`/32 | offset 8 | 元素 | `GetOneDimUBFormer`，可能被半核重切分修正 | pool slot 长度、整 tile 长度、核起点 | 必须大于 0；容量计算先用 `int64_t` 后写入 int32。 |
| 同上 | `ubTail` | `int32_t`/32 | offset 12 | 元素 | `N%ubFormer`，整除时写 `ubFormer` | 最后核最后 tile 有效长度 | 1..ubFormer。 |
| 同上 | `blockNum` | `int32_t`/32 | offset 16 | AIV block | `ceil(ubOuter/blockFormer)` | 最后核判断 | 不超过可用 AIV 核数。 |
| 同上 | `blockFormer` | `int64_t`/64 | offset 24（20–23 padding） | tile/核 | `ceil(ubOuter/coreNum)`，重切分后重算 | `CalcBlockOffset`、普通核循环数 | 正数。 |
| 同上 | `blockTail` | `int64_t`/64 | offset 32 | tile | `ubOuter%(blockFormer)`，整除时取 blockFormer | 尾核循环数 | 1..blockFormer。 |
| 同上 | `scalarData` | `char[16]`/128 | offset 40，结构 sizeof 56/alignof 8 | 原始字节 | `SetScalar(float reduceElts)` 写 offset 0..3 | `BroadcastOneDimSch::SetScalar<0>` 按 `Var<float,0>` 读取 | 仅前 4 字节活动；其余保持结构原值。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 张量梯度 | `(201,0)` | `l1_loss_grad_tiling.cpp::DoTiling/GetTilingKey` + `broadcast_tiling.h::DoOneDimOpTiling` | `BroadcastOneDimTilingData` | `l1_loss_grad<201,0>`→`BroadcastSch<201>` | `BroadcastOneDimSch` + `L1LossGradCast` + `Vec::*` | `l1_loss_grad/CMakeLists.txt`、OpDef、ascend950 binary config |
| R-002 标量梯度 | `(201,1)` | 同上 + `CheckGradsIsScalar` | 同 POD，`scalarFlag` 含 grads 位 | `l1_loss_grad<201,1>`→`BroadcastSch<201>` | `BroadcastOneDimSch` + `L1LossGradScalarCast` + `Vec::*` | 同 R-001 |

## Kernel 实现

### 公共执行流程

核心是一个 AIV-only Device 节点。entry 先拒绝空 workspace/user workspace，设置系统 workspace，然后按编译期 `inputGradsIsScalar` 选择 DAG；`BroadcastSch<201>` 解析 POD，建立 `TPipe` 和 `BroadcastOneDimSch`。scheduler 初始化 GM 句柄、从 `scalarData` 恢复 FP32 scale、按 liveness `BufferNum` 初始化单 TBuf，逐核逐 tile 深度优先执行 DAG。

公共 DAG 次序为：predict/label 搬入→`>` 与 `<` 两次比较→Select 得到 `+1/0/-1`→梯度转 FP32→乘 scale→乘 sign→按需降精度→CopyOut。`MemOptCfg<LEVEL_2>` 让编译期 buffer 分配器按节点活跃区间复用 pool；每个 tile 的 ping/pong 由 `i&1` 选择，具体 buffer-id 表由 `GetBufferIds<true,true>()` 生成。

### Route R-001：张量梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_l1_loss_backward.cpp` | `aclnnL1LossBackwardGetWorkspaceSize` | 非 scalar gradOutput 会 BroadcastTo 到 `B` | 连续共同 shape 进入 L0。 |
| Tiling | `op_host/arch35/l1_loss_grad_tiling.cpp` | `CheckGradsIsScalar` | normalized storage shape 不为 `[1]` | `inputGradsIsScalar_=0`。 |
| Tiling | 同上 | `DoTiling` | dtype 三分支构造 `L1LossGradCast<U>::OpDag` | common Broadcast tiling。 |
| Router | `op_kernel/l1_loss_grad.cpp` | `l1_loss_grad<201,0>` | `if constexpr(inputGradsIsScalar==1)` 为假 | `L1LossGradCast<DTYPE_PREDICT,float>`。 |

#### Device 数据流

##### 函数级调用链

```text
L1LossGradAiCore -> l1_loss_grad<201,0>(grads,predict,label,y,workspace,tiling)
  -> BroadcastSch<201,L1LossGradCast<U,float>::OpDag>::Process
    -> BroadcastOneDimSch::Init(TPipe, grads,predict,label,y)
    -> BroadcastOneDimSch::Process
      -> Run(tile)
        -> CopyInBrc(grads/predict/label)
        -> Cast<float,U>(grads)
        -> Compare<U,GT/LT>(predict,label)
        -> Duplicate<float>(+1/-1)
        -> Select<float>(sign)
        -> Muls<float>(grads,reduceElts)
        -> Mul<float>(sign,scaledGrads)
        -> Cast<U,float,1>
        -> CopyOut<U>(y)
```

##### 源码等价伪码

```text
# 伪码；N=dimLen，T=ubFormer
coreBase = blockFormer * T * blockIdx
loops = (blockIdx == blockNum-1) ? blockTail : blockFormer
lastLen = (blockIdx == blockNum-1) ? ubTail : T
for i in [0, loops):
    len = (i == loops-1) ? lastLen : T
    pingpong = i & 1
    offset = i*T
    g = DataCopyPad(grads[coreBase+offset : +len])
    p = DataCopyPad(predict[coreBase+offset : +len])
    t = DataCopyPad(label[coreBase+offset : +len])
    signPos = Select(p>t, 1.0f, 0.0f)
    sign = Select(p<t, -1.0f, signPos)
    gf = Cast<float>(g, roundMode=0)
    scaled = Muls(gf, POD.scalarData<float>[0])
    outF = Mul(sign, scaled)
    out = Cast<U>(outF, roundMode=1)
    DataCopyPad(y[coreBase+offset : +len], out)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| grads/predict/label GM | `(blockFormer*ubFormer*blockIdx + i*ubFormer)*sizeof(U)` | 当前 tile `len*sizeof(U)` | U/ND contiguous | ACLNN 前处理 | `CopyIn` | 用户或 executor 中间 tensor，核心期间有效 | `DataCopyPad` 单 block，尾长为真实字节。 |
| y GM | 同一元素 offset | `len*sizeof(U)` | U/ND contiguous | `CopyOut` | 后 Cast/ViewCopy | executor 中间 tensor | 只写有效 len。 |
| `buf` TBuf pool | slot `bufId * ubFormer * MaxDtypeBytes` | `BufferNum*ubFormer*MaxDtypeBytes` | uint8 pool，按节点 reinterpret | `InitBuffer`/各 DAG 节点 | 后继节点/MTE3 | 单 Kernel 单核；每 tile 使用一组编译期 slot | `LEVEL_2` liveness 复用；ping/pong id 表。 |
| scale | POD offset 40 | 4 B | FP32 scalar | Host `SetScalar` | `Muls<float>` | 整个 kernel | 寄存器/标量容器，不占 tile slot。 |
| workspace | 框架系统前缀 + user 区 | 16 MiB 核心声明 | bytes | executor | entry 仅校验/设置系统 workspace | launch 期间 | DAG 不读 userWS；不能当 UB。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  pool0["TBuf pool slot 0 · ubFormer×MaxDtypeBytes"]:1
  pool1["slot 1 · 同容量"]:1
  poolN["... liveness/ping-pong slots"]:1
  scalar["Var<float> scale · 4 B scalar"]:1
```

##### 字节账与复用顺序

```text
A = L1LossGradCast<U,float>::OpDag::GetBufferNum<true,true>()       # Host tile divisor
B = L1LossGradCast<U,float>::OpDag::BufferNum                      # Device allocated slots
slot_bytes = ubFormer * MaxDtypeBytes
UB_total = B * slot_bytes
ubFormer = floor((platformUB/A)/128)*128/MaxDtypeBytes
required_bound = B * floor((platformUB/A)/128)*128 <= platformUB
dynamic_ub = N/A（AIV TBuf 路线）
workspace_total(core declaration) = 16*1024*1024 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| DAG slot `bufId` | CopyIn/计算节点 | 后继计算或 CopyOut | 最后消费者释放后由后续节点/下一 tile 使用 | `GetTensor/ReleaseTensor` 映射到 MTE2/V/MTE3 Mutex | 必须使用官方 `GetBufferIds`，不能手工改 slot。 |
| ping/pong slot 组 | tile i | tile i 的 DAG | tile i+2 | `i&1` 与 pipe Mutex | Host `ubFormer` 容量必须覆盖每 slot。 |
| sign 中间 | 两次 Compare+Select | 最终 Mul | liveness 结束后 | DAG `LEVEL_2` 活跃区间 | equality lane 必须保留第一 Select 的 0。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `Mutex::Lock/Unlock<PIPE_MTE2>` | DataCopyPad | V 节点 `GetTensor<VECCALC>` | pool slot | copyParams `{1,len*sizeof(U),0,0,0}`，无 padding 写入。 |
| V→V | `PIPE_V` Mutex + DAG liveness | 每个 Vector 节点 | 后继节点 | 同一/复用 slot | Compare 在 U；梯度和最终乘法在 FP32。 |
| UB→GM | `PIPE_MTE3` Mutex | final Cast/Mul | DataCopyPad CopyOut | 输出 slot | 只提交 len；FP16/BF16 用 roundMode 1。 |
| tail | `blockTail/ubTail` | Host POD | 尾核/尾 tile | GM 有效域 | 整除时 tail 字段取 full 值，避免 0 长最后循环。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/l1_loss_grad/op_host/arch35/l1_loss_grad_tiling.cpp` | `DoTiling/CaluateReduceElts` | dtype DAG、scale 与 `(schMode,scalarFlag)` key。 |
| Kernel entry/router | `loss/l1_loss_grad/op_kernel/l1_loss_grad.cpp` | `l1_loss_grad<201,0>` | 张量 DAG 与 scheduler 实例。 |
| Device body/helper | `loss/l1_loss_grad/op_kernel/arch35/l1_loss_grad_dag.h`、`broadcast_sch_one_dim.h` | `L1LossGradCast::OpDag`、`BroadcastOneDimSch::Run` | 完整比较/选择/缩放/乘法、地址、tile 与 pool 同步。 |

### Route R-002：标量梯度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_l1_loss_backward.cpp` | `aclnnL1LossBackwardGetWorkspaceSize` | 950 RegBase 且 gradOutput view scalar 时跳过 BroadcastTo | 核心仍收到单元素 grads。 |
| Tiling | `op_host/arch35/l1_loss_grad_tiling.cpp` | `CheckGradsIsScalar` | normalized storage shape 为 `[1]` | `inputGradsIsScalar_=1`。 |
| Tiling | 同上 | `DoTiling` | dtype 分支构造 `L1LossGradScalarCast<U>::OpDag` | mode 201 + scalar key。 |
| Router | `op_kernel/l1_loss_grad.cpp` | `l1_loss_grad<201,1>` | constexpr 分支为真 | ScalarCast DAG。 |

#### Device 数据流

##### 函数级调用链

```text
L1LossGradAiCore -> l1_loss_grad<201,1>
  -> BroadcastSch<201,L1LossGradScalarCast<U,float>::OpDag>::Process
    -> BroadcastOneDimSch::Init/Process
      -> CopyIn(predict,label)
      -> Compare/Select 生成 sign
      -> CopyInBrc(grads scalar) 的 scheduler scalar 分支
         -> GM GetValue(0) -> Vec::Duplicate<U>(tile)
      -> Cast<float> -> Muls(scale) -> Mul(sign) -> Cast<U> -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；分核、尾块同 R-001
if first use of this ping/pong scalar slot:
    gradScalar = grads[0]
    Duplicate(gradTile, gradScalar, len)
# scheduler 在前两次 tile 分别填充 ping/pong；i>1 后 isFirstScalarTime=false，复用既有 slot
for each tile:
    sign = (predict>label) ? 1 : ((predict<label) ? -1 : 0)
    outF = sign * (Cast<float>(gradTile) * reduceElts)
    CopyOut(Cast<U,roundMode=1>(outF), len)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| grads GM | 固定 0 | 1 元素 | U scalar/[1] | 用户/Contiguous | GM `GetValue(0)` | 核心期间 | 每个核读取一次标量值；不按 coreBase 偏移。 |
| predict/label/y GM | 同 R-001 | tile len | U/ND | ACLNN/核心 | Compare/CopyOut | 同 R-001 | 同 R-001。 |
| grad scalar UB slot | 对应 CopyIn 节点的 ping/pong `bufId` | 每 slot `ubFormer*MaxDtypeBytes` | U tile | `Vec::Duplicate` | Cast | 首两次 tile 填充后跨 tile 复用 | 避免从 GM 搬运整张广播梯度。 |
| 其他 TBuf/scale/workspace | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001 | 同 R-001。 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gradPing["grad scalar duplicate · ping · ubFormer×MaxDtypeBytes"]:1
  gradPong["grad scalar duplicate · pong · 同容量"]:1
  live["其余 DAG liveness slots"]:1
  scalar["reduceElts · FP32 4 B"]:1
```

##### 字节账与复用顺序

```text
A = L1LossGradScalarCast<U,float>::OpDag::GetBufferNum<true,true>()
B = L1LossGradScalarCast<U,float>::OpDag::BufferNum
UB_total = B * ubFormer * MaxDtypeBytes
required_bound = B * floor((platformUB/A)/128)*128 <= platformUB
dynamic_ub = N/A
workspace_total(core declaration) = 16*1024*1024 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| grad ping slot | 首个偶数 tile Duplicate | 所有偶数 tile 的 Cast | kernel 结束前不重填 | `isFirstScalarTime` 与 buffer-id 固定映射 | grad 值在 kernel 生命周期内不变。 |
| grad pong slot | 首个奇数 tile Duplicate | 所有奇数 tile 的 Cast | kernel 结束前不重填 | 同上 | 至少存在第二 tile 时才初始化并使用。 |
| 其余 slots | 同 R-001 | 同 R-001 | 同 R-001 | MTE2/V/MTE3 Mutex | 同 R-001。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scalar GM→UB | `GetValue` + `PIPE_V` Mutex + Duplicate | grads[0] | Cast | grad ping/pong slot | Duplicate 的 count 为当前首次 tile len；后续同 slot只在同奇偶 tile容量内读取有效 len。 |
| 计算/输出 | 同 R-001 | Compare/Select/Muls/Mul/Cast | CopyOut | liveness slots | FP32 缩放与乘法，tail len 贯穿全部 primitive。 |
| 核间 | 无跨核共享/Atomic | 每核独立输入区 | 独立输出区 | GM 区间 | coreBase 由 blockIdx 唯一划分，只有 grads[0] 被只读共享。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `loss/l1_loss_grad/op_host/arch35/l1_loss_grad_tiling.cpp` | `CheckGradsIsScalar/DoTiling` | 单元素识别与 ScalarCast DAG。 |
| Kernel entry/router | `loss/l1_loss_grad/op_kernel/l1_loss_grad.cpp` | `l1_loss_grad<201,1>` | constexpr 标量分支。 |
| Device body/helper | `l1_loss_grad_dag.h`、`broadcast_sch_one_dim.h` | `L1LossGradScalarCast`、`CopyIn` scalar branch | GM 标量只读、Duplicate、ping/pong 缓存复用。 |

## 算子特定优化

### Trick T-001：950 标量上游梯度不物化广播张量

#### 触发与源码依据

- 适用 Route：R-002
- 触发条件：`IsRegbase()` 且 `gradOutput->GetViewShape().IsScalar()`；核心 Tiling 将归一化后的 `[1]` 标记为 scalar。
- Golden 来源：[`loss/l1_loss_grad/op_api/aclnn_l1_loss_backward.cpp:210-215`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_api/aclnn_l1_loss_backward.cpp#L210-L215)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
auto gradOutputBroadcast = gradOutputContiguous;
if (!Ops::NN::AclnnUtil::IsRegbase() ||
    !gradOutput->GetViewShape().IsScalar()) {
    // 判断gradOutput是否需要进行broadcast或者promote
    gradOutputBroadcast = BroadcastTensor(gradOutputContiguous, broadcastShape2, uniqueExecutor.get());
```

#### 机制与设计原因

ACLNN 不建立 `BroadcastTo(gradOutput→B)` 临时张量，核心 scheduler 从 `grads[0]` 读取一个 U 值，用 `Duplicate(dst,scalar,len)` 填充两个可交替使用的 tile slot。它把广播成本从 GM 中间张量的写入与再次读回，变为每核一次标量读取和片上 Duplicate；`isFirstScalarTime` 在 ping/pong 两组均填充后阻止后续 tile 重复 Duplicate。

#### 迁移边界

- 适用：只读标量在全输出域广播、且每核能安全读同一 GM 标量的逐元素 DAG。
- 前提：950 RegBase；Host key、scheduler `scalarFlag`、DAG 的 `Placeholder::ScalarAttr<true>` 三者必须一致。
- 失效：非 scalar grad 必须走 R-001；不能把多元素张量误标 scalar，否则所有位置都会错误复用元素 0。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的实际失效去向是非 scalar 时调用 `BroadcastTensor` 并走 R-001；绕过 shape 判定强制标量化会改变数学结果。

### Trick T-002：比较在输入 dtype，缩放与乘法在 FP32

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：核心 dtype 为 FP16/BF16/FP32；模板第二类型固定为 `float`。
- Golden 来源：[`loss/l1_loss_grad/op_kernel/arch35/l1_loss_grad_dag.h:41-55`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/loss/l1_loss_grad/op_kernel/arch35/l1_loss_grad_dag.h#L41-L55)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using OpCopyGradsCast = Bind<Vec::Cast<T, U, 0>, OpCopyGrads>;

using OpCompareGT = Bind<Vec::Compare<uint8_t, U, COMPARE_MODE_GT>, OpCopyPredict, OpCopyLabel>;
using OpOneTensor = Bind<Vec::Duplicate<T>, ConstValueOne>;
using OpSelect = Bind<Vec::Select<uint8_t, T, SELECT_MODE_SCALAR>, OpCompareGT, OpOneTensor, ConstValueZero>;

using OpGrads = Bind<Vec::Muls<T>, OpCopyGradsCast, Placeholder::Var<T, 0>>;

using OpResCast = Bind<Vec::Mul<T>, OpSign, OpGrads>;
using OpRes = Bind<Vec::Cast<U, T, CAST_MODE_RINT>, OpResCast>;
```

#### 机制与设计原因

sign 只依赖 `self` 与 `target` 在接口计算 dtype 上的顺序关系，避免为比较额外物化两路 FP32 tensor；而可能带来舍入的 `gradOutput*scale*sign` 在 FP32 完成。`Cast<T,U,0>` 的 `T=float` 来自 DAG 模板，scale 来自 POD `Var<float,0>`，最终 `Cast<U,float,1>` 明确使用 roundMode 1。FP32 输入的同 dtype Cast 可由 DAG 元编程识别为 no-op。

#### 迁移边界

- 适用：输出只需要输入 dtype 顺序关系，但缩放乘法希望用 FP32 的逐元素 backward。
- 前提：比较 dtype 的 NaN/相等语义与目标算子一致；最终 Cast roundMode 必须保持官方值 1。
- 失效：若目标语义要求先 promote 后比较，或需要更高精度比较，不能复用该顺序。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得把“先比较后升精度”泛化到顺序关系会因 dtype 转换改变的其他算子。

## 模板与实现对比

| 对比维度 | R-001 张量梯度 | R-002 标量梯度 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | grads normalized shape 非 `[1]` | grads normalized shape 为 `[1]` | `CheckGradsIsScalar` 唯一分流。 |
| 模板与 dtype | `L1LossGradCast<U,float>` | `L1LossGradScalarCast<U,float>` | 后者把 In0 标成 ScalarAttr。 |
| 循环与分核 | mode 201，按共同 N 切 tile/core | 完全相同 | 输出域相同。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 单 LEVEL_2 TBuf pool；FP32 scale | 同 pool，多出长期复用的 grad ping/pong tile语义 | 无 TQue、dynamic UB、通信 window。 |
| 内存总占用与复用 | `A_tensor*ubFormer*MaxDtypeBytes` | `A_scalar*ubFormer*MaxDtypeBytes` | A 由各自 DAG liveness 编译期计算。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 三路 DataCopyPad | 两路 DataCopyPad + grads GetValue/Duplicate | 均为 AIV_VECTOR；mode201 不使用 NDDMA。 |
| 同步、精度与 tail | MTE2/V/MTE3 Mutex；FP32 缩放；blockTail/ubTail | 同，grad slot跨 tile 复用 | tail 参数逐 primitive 传递。 |
| fallback/失败边界 | 无 Device fallback | 无 Device fallback | Host 校验或 Tiling 失败即终止。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档声称支持，但固定源码缺少 A2 活动 Host/Tiling/Kernel/构建闭包，本卡不发布其实现 | 同 A2 | 公开支持且 source_closed |
| Host/Launcher | 固定仓不可见 | 固定仓不可见 | arch35 `L1LossGradTiling` + ACLNN RegBase scalar fast path |
| Kernel/资源 | 固定仓不可见 | 固定仓不可见 | mode201 单 AIV kernel、DAG LEVEL_2 TBuf pool、16 MiB workspace 声明 |

## 限制与失败边界

- 公开 API 允许三路输入互相 promote，但核心 `L1LossGrad` 要求三路与输出同 dtype；ACLNN 必须保留前置 Cast，不能直接把混合 dtype 送入核心。
- `reduction` 仅 0/1/2；Host 内部字符串仅 `none/mean/sum`。`mean` 的任何 label 维为 0 都使 Tiling 失败；公开 ACLNN 空输入更早返回成功且不启动核心。
- POD 的 `ubSplitAxis` 在 one-dim producer 未初始化且 Device 仅用于日志；复用 ABI 时不得把它当有效轴。
- API 级完整 workspace 包含可选框架节点临时 tensor，具体 offset/alias 由 executor allocator 决定；固定算子源码没有给出一条对所有输入布局成立的数值公式。
- DAG 的 Host `GetBufferNum<true,true>()`、Device `BufferNum=GetBufferNum<true,false>()` 与具体 buffer-id 数组没有在固定仓输出实例化后的数值；字节账保留二者的官方编译期表达式，复用时必须验证 `B*floor((U/A)/128)*128<=U`，不能把 A、B强行视为同一常量或手写猜测 slot 数。
- A2/A3 的 README/API 支持声明与本 Commit 活动构建接线不闭合：根 CMake、OpDef、binary config、Kernel CMake只给出 ascend950，因此未创建无后缀正式卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `loss/l1_loss_grad/docs/aclnnL1LossBackward.md` | 产品表、函数原型 | 950公开支持、参数与错误语义。 |
| 入口声明/约束 | `loss/l1_loss_grad/op_api/aclnn_l1_loss_backward.h` | 两段式声明 | C ABI 与参数顺序。 |
| 入口实现 | `loss/l1_loss_grad/op_api/aclnn_l1_loss_backward.cpp` | `CheckParams/aclnnL1LossBackwardGetWorkspaceSize` | 前处理、标量保留、图节点顺序。 |
| 入口实现 | `loss/l1_loss_grad/op_api/l1_loss_grad.cpp` | `L1LossGrad/L1LossGradAiCore` | L0 dtype gate、launcher 和 attr 映射。 |
| Host/Tiling/Launcher | `loss/l1_loss_grad/op_host/arch35/l1_loss_grad_tiling.cpp` | `L1LossGradTiling::*` | scalar/scale/dtype/key producer。 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | Host阶段顺序与最终 SetTilingKey。 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority升序选择和失败终点。 |
| Host/Tiling/Launcher | `pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h` | `BroadcastBaseTiling` | 合轴 route、UB/core/POD/workspace。 |
| Host/Tiling/Launcher | `src/op_common/atvoss/broadcast/broadcast_tiling.cpp` | `DimensionCollapse` | 相邻轴合并与广播 stride。 |
| POD/ABI | `loss/l1_loss_grad/op_kernel/arch35/l1_loss_grad_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL` | schMode/scalar 模板 key。 |
| POD/ABI | `pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h` | `BroadcastOneDimTilingData` | POD 字段与 mode 列表。 |
| Kernel entry/body | `loss/l1_loss_grad/op_kernel/l1_loss_grad.cpp` | `l1_loss_grad` | workspace gate 与两 DAG router。 |
| Kernel entry/body | `loss/l1_loss_grad/op_kernel/arch35/l1_loss_grad_dag.h` | `L1LossGradCast/ScalarCast` | 逐节点数学 DAG。 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch.h` | `BroadcastSch<201>` | AIV-only scheduler 路由。 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/broadcast/broadcast_sch_one_dim.h` | `BroadcastOneDimSch` | pool、循环、地址、scalar、tail。 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch/GetBufferIds` | liveness 与 buffer-id 生成。 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/node.h` | `DagNodeInfo` | DAG holder、Var 与存活节点统计。 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/buffer.h` | `GenerateBufferIdOrder` | LEVEL_2 slot 分配、释放与 ping/pong id。 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/vec.h` | `Vec::*` | DataCopyPad、Compare、Select、Cast、Muls、Mul。 |
| Kernel entry/body | `pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor/ReleaseTensor` | MTE2/V/MTE3 Mutex 协议。 |
| Launcher/runtime | `src/nnopbase/composite_op/aclnn_engine/op_executor.cpp` | `GetWorkspaceSize/Run/CommonOpExecutorRun` | workspace 分配、stream 顺序与失败传播。 |
| 构建接线 | `loss/l1_loss_grad/op_host/l1_loss_grad_def.cpp` | `L1LossGrad` OpDef | ascend950 注册与 dtype/format。 |
| 构建接线 | `loss/l1_loss_grad/op_host/config/ascend950/l1_loss_grad_binary.json` | `op_list` | 三 dtype 950 binary。 |
| 构建接线 | `loss/l1_loss_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | 仅 ascend950 + arch35。 |
| 构建接线 | `loss/l1_loss_grad/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 Kernel 与 AIV 编译选项。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | mode201 的 56 B one-dim POD 与 float scale | 字段顺序/位宽/padding、scalarData offset、inactive ubSplitAxis | `BroadcastOneDimTilingData` |
| 调用顺序 | ACLNN连续化→广播→promote Cast→核心→输出 Cast→ViewCopy | RegBase scalar 特判必须在 grad BroadcastTo 前 | `aclnnL1LossBackwardGetWorkspaceSize` |
| workspace/内存 | 16 MiB核心声明；TBuf=`B*ubFormer*MaxDtypeBytes` | workspace不是UB；Host A与Device B必须分别由 DAG liveness计算并满足上界 | `BroadcastBaseTiling::DoTiling`、`BroadcastOneDimSch::Init` |
| 同步 | 单 stream节点顺序；pool slot使用 MTE2/V/MTE3 Mutex | Get/Release配对与 buffer-id 不得改序 | `aclOpExecutor::Run`、`sync.h` |
| 精度与尾块 | 输入 dtype 比较，FP32缩放/乘法，roundMode1降精度 | equality=0；整除 tail写 full；所有 primitive用真实 len | `L1LossGradCast`、`DoOneDimFormerTiling` |
| 失败边界 | 校验失败/Tiling失败/launcher失败均阻断后续节点 | 不存在 AiCPU 或 Device fallback | `CheckParams`、`L1LossGrad`、`aclOpExecutor::Run` |
