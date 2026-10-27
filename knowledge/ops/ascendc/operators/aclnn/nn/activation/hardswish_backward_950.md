---
type: Operator
title: "HardswishBackward 950 ACLNN 算子知识"
description: "说明 aclnnHardswishBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/docs/aclnnHardswishBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_host/op_api/aclnn_hardswish_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_host/op_api/aclnn_hardswish_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_host/op_api/hard_swish_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_host/arch35/hard_swish_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_kernel/hard_swish_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_host/hard_swish_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_host/config/ascend950/hard_swish_grad_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:01:43Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# HardswishBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnHardswishBackwardGetWorkspaceSize` + `aclnnHardswishBackward`；当前固定仓未提供 V2。 |
| 计算语义 | `out[i] = gradOutput[i] * g(self[i])`，其中 `g(x)=0 (x<-3)`、`x/3+0.5 (-3<=x<=3)`、`1 (x>3)`。 |
| 输入输出 | `gradOutput`、`self`、`out` 同 shape、同 dtype；公开 950 支持 `FLOAT32/FLOAT16/BFLOAT16`、ND，输入和输出可为非连续 Tensor。 |
| 执行类型 | 算子拥有的 Device 节点为单个 `AIV_VECTOR` Kernel；`Contiguous` 与 `ViewCopy` 是 ACLNN 框架边界。未发现 Cube/AIC 计算或 `MIX_CV` 节点。 |
| Kernel 路线 | R-001 FP32 单缓冲；R-002 FP32 双缓冲；R-003 FP16/BF16 单缓冲升精度；R-004 FP16/BF16 双缓冲升精度。 |
| 核心资源 | 两个 VECIN Queue、一个 VECOUT Queue、一个或两个 FP32/同型计算 TBuf、一个位掩码 TBuf；算子 Kernel workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN→Level-0 `HardSwishGrad` launcher→arch35 Tiling/POD→模板 entry→`HardSwishGrad<T, BUFFER_MODE>` body→Ascend950 OpDef/binary/CMake 接线闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对扁平化后的每个有效元素 `i`：

```text
x = self[i]
d = 0,                 x < -3
    x / 3 + 0.5,       -3 <= x <= 3
    1,                 x > 3
out[i] = gradOutput[i] * d
```

两个端点由实际 Kernel 的 `GE(-3)` 与 `LE(3)` 掩码保留在线性分支中，和公开文档一致；因此 `x=-3` 时 `d=-0.5`，`x=3` 时 `d=1.5`。Kernel 文件头注释把端点写到外侧分支，但真正执行的 `Compares/Select` 顺序以及公开 ACLNN 文档均采用上述闭区间，复用时应以 body 为准。

张量只按存储元素总数一维切分，Kernel 不解释原始维度或 layout；有效输出范围恰为 `totalNum` 个元素。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | 与 `self/out` 相同；公开契约为 ND，最大维数检查为 8 | 950：`FLOAT32/FLOAT16/BFLOAT16` | 必选、可非连续；表示正向输出梯度。Level-2 先转连续。 |
| `self` | 与 `gradOutput/out` 相同；公开文档给出 rank 0–8 | 与 `gradOutput/out` 相同 | 必选、可非连续、支持空 Tensor；表示正向输入。标量在 Tiling 内按 shape `{1}` 处理。 |
| `out` | 与 `self` 相同；公开契约为 ND | 与输入相同 | 必选、可非连续；Level-0 临时输出通过 `ViewCopy` 写回。 |
| `workspaceSize` | N/A | `uint64_t*` | 第一段输出；是整个 executor 的 workspace，包含框架 `Contiguous/ViewCopy` 可能需要的空间。 |
| `executor` | N/A | `aclOpExecutor**` | 第一段构造并转移所有权，第二段消费其节点列表。 |
| `workspace`、`stream` | N/A | Device 指针、`aclrtStream` | 第二段传给 `CommonOpExecutorRun`；算子本体 Tiling 的 workspace slice 为 0。 |

`CheckParams` 对三个 Tensor 做非空检查，对输入 dtype 支持范围、三者 dtype 一致、输入 shape 相等、输出与 `self` shape 相等做检查。`CheckShapeValid` 对两个输入执行最大维数检查；输出通过 shape 相等约束间接受限。

### 入口与版本

固定 Commit `955f8b33985b6d72989ebff341bb9c2c564a51bc` 中公开入口只有 `aclnnHardswishBackward` 两段式版本，未出现 V2。第一段构造 `Contiguous(self)`、`Contiguous(gradOutput)`、`l0op::HardSwishGrad`、`ViewCopy`，第二段调用 `CommonOpExecutorRun`。

950 公开性由接口文档的 Ascend 950PR/950DT 支持表确认；实现侧由 `HardSwishGrad` 的 Ascend950 OpDef、`ascend950/hard_swish_grad_binary.json`、`arch35` Tiling/Kernel 与 CMake 共同闭合。公开头文件注释还列出 NCHW/NHWC，而正式文档和 OpDef 均以 ND 为公开/注册格式，本卡不把注释中的更宽范围提升为正式契约。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnHardswishBackwardGetWorkspaceSize` 的源码顺序为：

1. `OP_CHECK_COMM_INPUT` 检查 `workspaceSize/executor`，`CREATE_EXECUTOR` 创建临时 executor。
2. `CheckParams` 检查三个 Tensor 非空、dtype 支持与一致性、shape 一致性、输入最大 rank。
3. 任一输入为空时直接令 `workspaceSize=0`，转移 executor 后成功返回；该公开 ACLNN 路径不添加 `HardSwishGrad` Kernel。
4. 依次调用 `l0op::Contiguous(self)` 和 `l0op::Contiguous(gradOutput)`；任一步返回空指针即报内部错误。
5. 调用 `l0op::HardSwishGrad(gradOutputContiguous, selfContiguous)`；Level-0 按 `self` 的 view shape/dtype 分配临时输出并执行 `ADD_TO_LAUNCHER_LIST_AICORE`。
6. 调用 `l0op::ViewCopy` 把临时输出提交到用户 `out`；随后读取 executor 总 workspace 并转移所有权。

`Contiguous/ViewCopy` 的内部 selector、额外 Kernel 与 workspace 公式不在当前算子目录中，属于 Skill 允许的 ACLNN 框架边界；算子拥有的 Level-0、Tiling、POD 与 Kernel 均已继续闭合。

### 参数、Workspace 与生命周期

对非空 Tensor，Tiling 从 `grad_output` 与 `self` 的 storage shape 得到元素数 `N`，要求两者元素数相同，并从输入 0 描述符读取 dtype。标量 storage shape 先归一化为 `{1}`。

```text
executor_workspace = framework_contiguous_slices
                   + HardSwishGrad_slice(0 bytes)
                   + framework_view_copy_slice
```

外层 ACLNN 只公开 `uniqueExecutor->GetWorkspaceSize()` 的总值；框架切片的 offset/extent 在固定范围内不可见。`HardSwishGradTilingFunc` 明确将 `currentWorkspace[0]` 写为 `WS_SYS_SIZE=0`，Kernel 的 `workspace` 形参未读取。executor 在第一段由 `uniqueExecutor` 创建并经 `ReleaseTo` 转给调用者；第二段在调用者提供的 workspace 与 stream 上运行完整节点列表。

### 启动图与失败传播

```text
aclnnHardswishBackwardGetWorkspaceSize
  -> Contiguous(self)                 [框架边界，可能产生 executor workspace]
  -> Contiguous(gradOutput)           [框架边界，可能产生 executor workspace]
  -> l0op::HardSwishGrad
       -> ADD_TO_LAUNCHER_LIST_AICORE(HardSwishGrad)
       -> hard_swish_grad<D_T_X, BUFFER_MODE>
          [单 AIV_VECTOR 节点，blockDim=usedCoreNum，POD=HardSwishGradTilingData，workspace=0]
  -> ViewCopy(tempGradInput, out)      [框架边界，可能产生 executor workspace]

aclnnHardswishBackward
  -> CommonOpExecutorRun(workspace, workspaceSize, executor, stream)
```

| 节点/边 | 输入→输出 | 参数/交接介质 | ordering 与 owner/lifetime | 失败传播 |
|---|---|---|---|---|
| `Contiguous(self)` | 用户 `self`→连续 Tensor | executor 管理的 Tensor/workspace，具体 slice 固定范围不可见 | 在 executor 节点列表中先于算子节点；executor 持有临时 Tensor | 空返回立即终止第一段 |
| `Contiguous(gradOutput)` | 用户 `gradOutput`→连续 Tensor | 同上 | 源码添加顺序位于 `self` 后、算子前 | 空返回立即终止第一段 |
| `HardSwishGrad` | 两个连续输入→Level-0 临时输出 | `HardSwishGradTilingData`；`blockDim=ceil(N/ceil(N/C))`；算子 workspace 0 | 临时输出由 executor `AllocTensor` 持有，直到后续 `ViewCopy` 消费 | launcher 添加失败返回空并阻断 `ViewCopy` |
| `ViewCopy` | 临时输出→用户 `out` | executor 框架边界 | 同一 executor/第二段 stream 上位于算子节点之后；完成前不得释放 out/workspace/executor | 空返回令第一段失败 |

固定范围没有暴露框架节点内部 event 或 workspace offset；第二段把同一个 stream 交给 `CommonOpExecutorRun`，算子节点间顺序由 executor 框架保证。算子自身是单 Kernel，不存在算子拥有的跨 Kernel workspace 边。

## Route 与参数设计

### 分核与资源计算

令 `N=totalNum`，平台 AIV 核数为 `C`，平台 UB 字节数为 `B`，元素字节数 `s=4(FP32)` 或 `2(FP16/BF16)`：

```text
blockFactor = ceil_div(N, C)                   # 元素/核
usedCoreNum = ceil_div(N, blockFactor)         # blockDim
coreOffset(b) = b * blockFactor                # 元素
coreLength(b) = min(blockFactor, N-coreOffset) # 元素

double = (N > 1024) ? 1 : 0
bufferNum = FP32 ? (double ? 8 : 5)
                 : (double ? 11 : 8)
alignUnit = max(GetUbBlockSize(context), 256/sizeof(float))
ubFactor = floor_align(floor((B/s)/bufferNum), alignUnit) # 元素/UB 迭代
loopCount = ceil_div(coreLength, ubFactor)
```

`blockFactor` 使每个启动 block 覆盖一段连续元素；最后一核用 `remainderLength` 截短。每核最后一轮的 `currentNum=coreLength-ubFactor*i`。`usedCoreNum` 仅启动实际需要的 AIV 核。

Host 把 `ubFactor` 按 `alignUnit` 向下对齐，并通过保守的 `bufferNum` 把 Queue、计算 TBuf 与掩码余量纳入容量预算。源码只检查平台 `C/B` 非零，没有单独检查计算后的 `ubFactor>0`；这依赖 Ascend950 实际 UB 足够大。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 单缓冲 | `(D_T_X=C_DT_FLOAT, BUFFER_MODE=0)` | `dtype=DT_FLOAT && N<=1024 && N>0` | dtype/阈值不符时匹配后续 tuple；非法 dtype 在 Tiling 失败 |
| 2 | R-002 FP32 双缓冲 | `(C_DT_FLOAT, 1)` | `dtype=DT_FLOAT && N>1024` | 同上 |
| 3 | R-003 FP16/BF16 单缓冲升精度 | `(C_DT_FLOAT16,0)` 或 `(C_DT_BF16,0)` | `dtype in {DT_FLOAT16,DT_BF16} && N<=1024 && N>0` | 同上 |
| 4 | R-004 FP16/BF16 双缓冲升精度 | `(C_DT_FLOAT16,1)` 或 `(C_DT_BF16,1)` | `dtype in {DT_FLOAT16,DT_BF16} && N>1024` | 同上 |

```text
if N == 0:
    public ACLNN returns before launcher creation
else:
    require dtype in {FLOAT, FLOAT16, BF16}
    BUFFER_MODE = (N > 1024)
    D_T_X = dtype(input0)
    dispatch hard_swish_grad<D_T_X, BUFFER_MODE>
```

Tiling 中仍保留 `N==0` 的 tuple `(dtype,0)` 与 `blockDim=1`，但公开 ACLNN 已在 Level-2 空 Tensor分支提前返回，因此它不是本入口的 Device Route。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 单缓冲 | `FLOAT,0` | 950、非空、同 shape/同 dtype、FP32、`N<=1024` | 单 `AIV_VECTOR` | `HardSwishGrad<float,0>::Process` | Queue depth=1；同型 derivative TBuf；Queue 协议 | 核尾与 tile 尾均截短；无 Device fallback |
| R-002 FP32 双缓冲 | `FLOAT,1` | 同上但 `N>1024` | 单 `AIV_VECTOR` | `HardSwishGrad<float,1>::Process` | Queue depth=2；同型 derivative TBuf | 同上 |
| R-003 FP16/BF16 单缓冲 | `FLOAT16,0` / `BF16,0` | 950、非空、同 shape/同 dtype、FP16/BF16、`N<=1024` | 单 `AIV_VECTOR` | `HardSwishGrad<half,0>` / `<bfloat16_t,0>` | Queue depth=1；两个 FP32 TBuf；cast→FP32→RINT cast；Queue 协议 | 同上 |
| R-004 FP16/BF16 双缓冲 | `FLOAT16,1` / `BF16,1` | 同上但 `N>1024` | 单 `AIV_VECTOR` | `HardSwishGrad<half,1>` / `<bfloat16_t,1>` | Queue depth=2；两个 FP32 TBuf；同上 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 全局双缓冲阈值 | `N > 1024` | 把三个 Queue 的 depth 从 1 改为 2，同时用更大的 `bufferNum` 缩小 `ubFactor` | R-001↔R-002，R-003↔R-004 | 选择单缓冲或双缓冲 tuple | `HardSwishGradTilingFunc` |
| 计算 lane 对齐 | `alignedNum=ceil_align(currentNum,64)` | 所有支持 dtype 的 `ComputeType` 都为 FP32，`Compares/Select` 按 256 字节粒度执行 | 全部 | 不存在替代 Route；仅把无效 lane 算入 UB，不写回 GM | `HardSwishGrad::Compute` |
| UB 容量预算 | `ubFactor=floor_align(floor((B/s)/bufferNum),alignUnit)` | Queue depth、FP32 中间量与 bit mask 必须同时驻留 | 全部 | 无运行时 fallback；Tiling 失败仅覆盖平台信息异常，未显式保护 `ubFactor=0` | `HardSwishGradTilingFunc` |
| dtype 升精度 | `T=half \|\| T=bfloat16_t` | 线性导数和乘法在 FP32 完成，结果 `CAST_RINT` 回原 dtype | R-003/R-004 | FP32 走 R-001/R-002 直接计算 | `NEED_CAST`、`ComputeType`、`Compute` |
| 空域 | Level-2：任一输入 `IsEmpty()` | 避免建立 launcher；输出为空无需 Device 写入 | 无 Kernel Route | `workspaceSize=0` 成功返回 | `aclnnHardswishBackwardGetWorkspaceSize` |
| 平台资源 | `GetCoreNumAiv()!=0 && UB!=0` | 分核与 UB 推导的除数 | 全部 | Tiling `GRAPH_FAILED` | `GetPlatformInfo` |

### POD/ABI

`HardSwishGradTilingData` 只含三个连续 `int64_t`；按声明的标准布局，`sizeof=24`、`alignof=8`，字段 offset 依次为 0/8/16。Host 在写字段前对整结构清零，Device 以同名结构解析。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `HardSwishGradTilingData` | `totalNum` | `int64_t/64` | `sizeof=24, alignof=8, offsetof=0` | 元素 | `totalIdx=shapeGradOutput.GetShapeSize(); tiling->totalNum=totalIdx` | `Init` 计算当前核 `remainderLength` | 非空 ACLNN 路径为正；Tiling 空域写 0 |
| 同上 | `blockFactor` | `int64_t/64` | `offsetof=8` | 元素/核 | `CeilDiv(totalIdx, coreNum)` | `Init` 计算核 offset 与 `blockLength_` | 非空时至少 1；空域写 0 |
| 同上 | `ubFactor` | `int64_t/64` | `offsetof=16` | 元素/轮 | 由 UB、dtype `typeSize`、`bufferNum`、`alignUnit` 向下对齐得到 | `Init` 写 `ubLength_`；`Process` 计算 loop 和 GM offset | 源码无显式正数检查；依赖 950 UB 配置 |

模板 selector 另由 `ASCENDC_TPL_SEL_PARAM(context,dTypeX,useDoubleBuffer)` 产生，不属于上述 POD。其数值编码由 AscendC 模板宏生成，固定目录未展开数值 key；Host 与 entry 以同一个 `HardSwishGrad` 模板声明闭合为 `(D_T_X,BUFFER_MODE)` tuple，不存在手写 router switch。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `FLOAT,0` | `hard_swish_grad_tiling_arch35.cpp::HardSwishGradTilingFunc` | `HardSwishGradTilingData` | `hard_swish_grad_apt.cpp::hard_swish_grad<float,0>` | `hard_swish_grad.h::HardSwishGrad<float,0>` | Ascend950 OpDef、binary JSON、root/op_host CMake |
| R-002 | `FLOAT,1` | 同上 | 同上 | `hard_swish_grad<float,1>` | `HardSwishGrad<float,1>` | 同上 |
| R-003 | `FLOAT16,0` / `BF16,0` | 同上 | 同上 | `hard_swish_grad<half,0>` / `<bfloat16_t,0>` | 对应 `HardSwishGrad<T,0>` | 同上 |
| R-004 | `FLOAT16,1` / `BF16,1` | 同上 | 同上 | `hard_swish_grad<half,1>` / `<bfloat16_t,1>` | 对应 `HardSwishGrad<T,1>` | 同上 |

## Kernel 实现

### 公共执行流程

所有非空 Route 共用单 Kernel launch graph。模板 entry 解析 `HardSwishGradTilingData`，构造 `HardSwishGrad<D_T_X,BUFFER_MODE>`，顺序调用 `Init` 与 `Process`。`Init` 将第 `b` 核绑定到三个 GM Tensor 的 `[b*blockFactor, b*blockFactor+blockLength)`，并按 `ubFactor` 分配 Queue/TBuf。

每轮严格执行 `CopyIn(i,currentNum) → Compute(currentNum) → CopyOut(i,currentNum)`：两个 GM 输入分别通过 `DataCopyPad` 入 VECIN Queue；Vector 流水计算导数和乘法；结果进入 VECOUT Queue，再只按 `currentNum*sizeof(T)` 写回 GM。算子本体无 workspace、跨核归约、Atomic、通信或 Cube API。

### Route R-001：FP32 单缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_hardswish_backward.cpp` | `CheckParams` / GetWorkspaceSize | 非空、同 shape、同 FP32 dtype | 添加 `HardSwishGrad` launcher |
| Tiling | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | `dataType==DT_FLOAT`; `useDoubleBuffer=(N>1024)=0` | selector `(FLOAT,0)` |
| Router | `op_kernel/hard_swish_grad_apt.cpp` | `hard_swish_grad<D_T_X,BUFFER_MODE>` | `D_T_X=float, BUFFER_MODE=0` | `HardSwishGrad<float,0>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::HardSwishGrad -> ADD_TO_LAUNCHER_LIST_AICORE
  -> hard_swish_grad<float,0>(grad_output,self,grad_input,workspace,tiling)
    -> HardSwishGrad<float,0>::Init
    -> HardSwishGrad<float,0>::Process
      -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
b = GetBlockIdx()
len = min(blockFactor, totalNum-b*blockFactor)
bind each GM at offset b*blockFactor, extent len
for i in [0, ceil(len/ubFactor)):
    n = min(ubFactor, len-i*ubFactor)
    copy n FP32 grad/self elements into depth-1 queues
    a = ceil_align(n, 64)
    deriv[0:a] = self[0:a]/3 + 0.5
    deriv = select(self>=-3, deriv, 0)
    deriv = select(self<=3, deriv, 1)
    result[0:a] = grad * deriv
    copy only n FP32 results to GM offset i*ubFactor
```

#### 内存与数据搬运

##### 资源与搬运表

令 `U=ubFactor`，`M=align_up(ceil(U/8),256)` 字节。

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `gradOutputGM` | `b*blockFactor+i*U` | 当前核 `len`，本轮 `n` | FP32/扁平 ND | 用户输入 | `CopyIn` | 用户/整次执行 | 有效搬运仅 `4n` 字节 |
| `selfGM` | 同上 | 同上 | FP32/扁平 ND | 用户输入 | `CopyIn` | 用户/整次执行 | 同上 |
| `gradOutputQueue` | Queue 子区 | `U*4` | FP32 | `CopyIn` | `Compute` | `TPipe`/单轮 Alloc→Free | VECIN，depth=1 |
| `selfQueue` | Queue 子区 | `U*4` | FP32 | `CopyIn` | `Compute` | 同上 | VECIN，depth=1 |
| `gradInputQueue` | Queue 子区 | `U*4` | FP32 | `Compute` | `CopyOut` | 单轮 Alloc→EnQue→DeQue→Free | VECOUT，depth=1 |
| `tmpBuf` | 独立 VECCALC 子区 | `U*4` | FP32 derivative | `Muls/Adds/Select` | `Mul` | `TPipe`/Kernel | 同一 buffer 原地更新导数 |
| `maskBuf` | 独立 VECCALC 子区 | `M` | bit mask/256B 对齐 | 两次 `Compares` | 各自后续 `Select` | `TPipe`/Kernel | 第二次比较覆盖第一次 mask |
| `gradInputGM` | `b*blockFactor+i*U` | 本轮仅 `n` | FP32/扁平 ND | `CopyOut` | 用户 | 用户/执行后 | `DataCopyPad` 写 `4n` 字节 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  go["gradOutputQueue · depth=1 · 4U B"]:1
  sx["selfQueue · depth=1 · 4U B"]:1
  out["gradInputQueue · depth=1 · 4U B"]:1
  tmp["tmpBuf · 4U B"]:1
  mask["maskBuf · M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*(1*U*4) + U*4 + M = 16U + M
Host nominal budget = 5*U*4 = 20U bytes, U 按 alignUnit 向下对齐
dynamic_ub = N/A（AIV Queue/TBuf 路径）
workspace_total = 0（HardSwishGrad 节点）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `tmpBuf` | `Muls/Adds` | 两次 `Select`、最终 `Mul` | 下一 tile 的 `Muls` | 单轮顺序执行，结果出队后进入下一轮 | 最终 `Mul` 完成后才覆盖 |
| `maskBuf` | 第一次 `Compares` | 第一次 `Select` | 第二次 `Compares` | 源码依赖顺序 | 第一次 `Select` 已消费 |
| `maskBuf` | 第二次 `Compares` | 第二次 `Select` | 下一 tile | 源码依赖顺序 | 第二次 `Select` 已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `AllocTensor/DataCopyPad/EnQue` | `CopyIn` | `Compute::DeQue` | 两路输入 Queue | 搬运 `n*4` 字节；Queue 容量 U |
| Vector | 顺序 Vector 数据依赖 | `Muls/Adds/Compares` | `Select/Mul` | derivative 与 mask | 全 FP32；计算 lane 数 `ceil_align(n,64)` |
| UB→GM | `EnQue/DeQue/DataCopyPad/FreeTensor` | `Compute` | `CopyOut` | 输出 Queue | 仅写回 n 个元素，aligned tail 的无效 lane 不落 GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | FP32、阈值、分核、U 与 selector |
| Kernel entry/router | `op_kernel/hard_swish_grad_apt.cpp` | `hard_swish_grad<float,0>` | tuple 到 body |
| Device body/helper | `op_kernel/arch35/hard_swish_grad.h` | `HardSwishGrad<float,0>::Init/CopyIn/Compute/CopyOut/Process` | Queue、Vector、tail 与 GM 地址 |

### Route R-002：FP32 双缓冲

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_hardswish_backward.cpp` | GetWorkspaceSize | 合法非空 FP32 | 添加 launcher |
| Tiling | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | `DT_FLOAT && N>1024` | `(FLOAT,1)`，`bufferNum=8` |
| Router | `op_kernel/hard_swish_grad_apt.cpp` | 模板 entry | `float,1` | `HardSwishGrad<float,1>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::HardSwishGrad
  -> hard_swish_grad<float,1>
    -> HardSwishGrad<float,1>::Init
    -> Process -> (CopyIn -> Compute -> CopyOut) * loopCount
```

##### 源码等价伪码

```text
# 伪码；地址、公式与 R-001 相同
BUFFER_NUM = 2
U = floor_align(floor((UB_bytes/4)/8), alignUnit)
for each local tile:
    CopyIn(n); Compute(ceil_align(n,64)); CopyOut(n)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三个 GM Tensor | 同 R-001 | 每核 `len`、每轮 `n` | FP32/扁平 ND | 用户/Kernel | Kernel/用户 | 整次执行 | 有效搬运 `4n` 字节/张量 |
| 两个 VECIN Queue | Queue 子区 | 各 `2*U*4` | FP32 | `CopyIn` | `Compute` | `TPipe` | depth=2 |
| VECOUT Queue | Queue 子区 | `2*U*4` | FP32 | `Compute` | `CopyOut` | `TPipe` | depth=2 |
| `tmpBuf` | 独立子区 | `U*4` | FP32 | Vector | Vector | Kernel | 导数原地更新 |
| `maskBuf` | 独立子区 | `M` | bit mask | `Compares` | `Select` | Kernel | 两次覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  go["gradOutputQueue · depth=2 · 8U B"]:1
  sx["selfQueue · depth=2 · 8U B"]:1
  out["gradInputQueue · depth=2 · 8U B"]:1
  tmp["tmpBuf · 4U B"]:1
  mask["maskBuf · M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*(2*U*4) + U*4 + M = 28U + M
Host nominal budget = 8*U*4 = 32U bytes
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slot | 当前 tile `CopyIn/Compute` | 当前 tile `DeQue/CopyOut` | 后续 tile 分配 | Queue `EnQue/DeQue/FreeTensor` | slot 已 Free 才可复用 |
| `tmpBuf/maskBuf` | 当前 tile Vector | 当前 tile 后续 Vector | 下一 tile | 源码循环顺序 | 当前 tile 的最终 `Mul` 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue 流水 | depth=2 Queue 协议 | `CopyIn/Compute` | `DeQue/FreeTensor` | 两套 Queue slot | FP32，tail 同 R-001 |
| Vector | 数据依赖 | `Muls/Compares` | `Select/Mul` | TBuf | 64 lane 对齐 |
| CopyOut | VECOUT Queue | `Compute` | `CopyOut` | result | 只写 n |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | `N>1024`、`bufferNum=8` |
| Kernel entry/router | `op_kernel/hard_swish_grad_apt.cpp` | `hard_swish_grad<float,1>` | 双缓冲实例 |
| Device body/helper | `op_kernel/arch35/hard_swish_grad.h` | `HardSwishGrad<float,1>` | depth=2，其他计算同 FP32 路线 |

### Route R-003：FP16/BF16 单缓冲升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_hardswish_backward.cpp` | `CheckDtypeValid` | 950 支持 FP16/BF16 且三者 dtype 一致 | 添加 launcher |
| Tiling | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | `dtype!=DT_FLOAT && N<=1024` | `(FLOAT16,0)` / `(BF16,0)`，`bufferNum=8` |
| Router | `op_kernel/hard_swish_grad_apt.cpp` | 模板 entry | `half,0` / `bfloat16_t,0` | 对应升精度 body |

#### Device 数据流

##### 函数级调用链

```text
hard_swish_grad<T,0>, T in {half,bfloat16_t}
  -> HardSwishGrad<T,0>::Init
  -> Process
    -> CopyIn<T>
    -> Compute
       Cast(self -> selfFp32)
       -> FP32 derivative via Muls/Adds/Compares/Select
       -> Cast(gradOut -> reused selfFp32)
       -> Mul FP32 -> Cast_RINT(result<T>)
    -> CopyOut<T>
```

##### 源码等价伪码

```text
# 伪码
load n values of T; a=ceil_align(n,64)
selfFp32[0:a] = cast_fp32(selfLocal)
derivFp32 = selfFp32/3 + 0.5
derivFp32 = select(selfFp32>=-3, derivFp32, 0)
derivFp32 = select(selfFp32<=3, derivFp32, 1)
selfFp32[0:a] = cast_fp32(gradOutLocal)  # 覆盖已完成读取的 selfFp32
derivFp32 *= selfFp32
result<T> = cast_rint(derivFp32)
store only n values
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三个 GM Tensor | `b*blockFactor+i*U` | 本轮 n | FP16 或 BF16 | 用户/Kernel | Kernel/用户 | 整次执行 | 每路有效搬运 `2n` 字节 |
| 两个 VECIN Queue | Queue 子区 | 各 `U*2` | T | `CopyIn` | `Compute` | 单轮 | depth=1 |
| VECOUT Queue | Queue 子区 | `U*2` | T | 末次 Cast | `CopyOut` | 单轮 | depth=1 |
| `tmpBuf` | VECCALC | `U*4` | FP32 | self Cast，随后 grad Cast | derivative 构造/最终 Mul | Kernel | 同一物理区分阶段复用 |
| `tmpBuf2` | VECCALC | `U*4` | FP32 derivative | Muls/Adds/Select/Mul | Cast_RINT | Kernel | 原地更新 |
| `maskBuf` | VECCALC | `M` | bit mask | `Compares` | `Select` | Kernel | 256B 对齐、两次覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  go["gradOutputQueue · depth=1 · 2U B"]:1
  sx["selfQueue · depth=1 · 2U B"]:1
  out["gradInputQueue · depth=1 · 2U B"]:1
  tmp["tmpBuf · 4U B"]:1
  deriv["tmpBuf2 · 4U B"]:1
  mask["maskBuf · M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*(1*U*2) + 2*(U*4) + M = 14U + M
Host nominal budget = 8*U*2 = 16U bytes
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `tmpBuf/selfFp32` | `Cast(self)` | 两次 `Compares` 与 `Muls` | `Cast(gradOut)` | 源码把覆盖放在第二次 `Select` 后 | 所有 selfFp32 读取已结束 |
| `tmpBuf2/derivFp32` | `Muls/Adds/Select` | 下一步 Select/Mul/Cast | 原地写或下一 tile | Vector 数据依赖 | 当前 consumer 完成 |
| `maskBuf` | 第一次/第二次 `Compares` | 对应 `Select` | 下一次比较 | 顺序依赖 | 对应 Select 已读 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 输入 | Queue | `CopyIn<T>` | `DeQue<T>` | 原 dtype 输入 | `n` 元素入队 |
| 升精度 | `Cast(...CAST_NONE)` | self/grad T | FP32 Vector | `tmpBuf` | a=64 元素对齐 |
| 计算 | FP32 Vector 数据依赖 | derivative/grad FP32 | `Cast_RINT` | `tmpBuf/tmpBuf2/maskBuf` | 阈值和常量均 FP32 |
| 降精度 | `Cast(...CAST_RINT)` | FP32 result | VECOUT Queue | result T | 只将 n 个有效元素写 GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | 非 FP32、单缓冲、`bufferNum=8` |
| Kernel entry/router | `op_kernel/hard_swish_grad_apt.cpp` | `hard_swish_grad<half/bfloat16_t,0>` | dtype tuple |
| Device body/helper | `op_kernel/arch35/hard_swish_grad.h` | `HardSwishGrad<T,0>::Compute` | FP32 升降精度与 TBuf 复用 |

### Route R-004：FP16/BF16 双缓冲升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_hardswish_backward.cpp` | GetWorkspaceSize | 合法非空 FP16/BF16 | 添加 launcher |
| Tiling | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | `dtype!=DT_FLOAT && N>1024` | `(FLOAT16,1)` / `(BF16,1)`，`bufferNum=11` |
| Router | `op_kernel/hard_swish_grad_apt.cpp` | 模板 entry | `half,1` / `bfloat16_t,1` | 对应双缓冲升精度 body |

#### Device 数据流

##### 函数级调用链

```text
hard_swish_grad<T,1>
  -> HardSwishGrad<T,1>::Init [Queue depth=2]
  -> Process -> CopyIn<T> -> Compute<FP32> -> CopyOut<T>
```

##### 源码等价伪码

```text
# 伪码；计算与 R-003 相同
BUFFER_NUM=2
U=floor_align(floor((UB_bytes/2)/11),alignUnit)
for each tile: load T -> cast/compute FP32 -> cast_rint T -> store valid n
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三个 GM Tensor | 同 R-003 | 本轮 n | FP16/BF16 | 用户/Kernel | Kernel/用户 | 整次执行 | `2n` 字节 |
| 两个 VECIN Queue | Queue 子区 | 各 `2*U*2` | T | CopyIn | Compute | TPipe | depth=2 |
| VECOUT Queue | Queue 子区 | `2*U*2` | T | Compute | CopyOut | TPipe | depth=2 |
| `tmpBuf/tmpBuf2` | 两个独立子区 | 各 `U*4` | FP32 | Cast/Vector | Vector/Cast | Kernel | tmpBuf 分阶段复用 |
| `maskBuf` | 独立子区 | M | bit mask | Compare | Select | Kernel | 256B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  go["gradOutputQueue · depth=2 · 4U B"]:1
  sx["selfQueue · depth=2 · 4U B"]:1
  out["gradInputQueue · depth=2 · 4U B"]:1
  tmp["tmpBuf · 4U B"]:1
  deriv["tmpBuf2 · 4U B"]:1
  mask["maskBuf · M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 3*(2*U*2) + 2*(U*4) + M = 20U + M
Host nominal budget = 11*U*2 = 22U bytes
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slot | 当前 tile | 当前 tile consumer | 后续 tile | `EnQue/DeQue/FreeTensor` | slot Free |
| `tmpBuf` | self Cast | derivative reads | grad Cast | 源码顺序 | 第二次阈值比较已完成 |
| `tmpBuf2/maskBuf` | Vector | 后续 Vector | 原地/下一 tile | 数据依赖 | consumer 已完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue | depth=2 协议 | CopyIn/Compute | DeQue/Free | Queue slots | n 元素有效 |
| FP32 管线 | Cast/Vector/Cast_RINT | 输入 T | 输出 T | 两个 FP32 TBuf | a=ceil_align(n,64) |
| 输出 | VECOUT Queue/DataCopyPad | result T | GM | 有效结果 | 只写 n，padding lane 丢弃 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | `N>1024`、非 FP32、`bufferNum=11` |
| Kernel entry/router | `op_kernel/hard_swish_grad_apt.cpp` | `hard_swish_grad<half/bfloat16_t,1>` | 双缓冲升精度实例 |
| Device body/helper | `op_kernel/arch35/hard_swish_grad.h` | `HardSwishGrad<T,1>` | depth=2、FP32 中间计算、tail |

## 算子特定优化

### Trick T-001：FP16/BF16 的 FP32 中间量复用

#### 触发与源码依据

- 适用 Route：R-003、R-004
- 触发条件：`std::is_same_v<T,half> || std::is_same_v<T,bfloat16_t>`
- Golden 来源：[`op_kernel/arch35/hard_swish_grad.h:166-171`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad.h#L166-L171)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Select(derivFp32, maskLocal, derivFp32, 1.0f,
       SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedNum);
Cast(selfFp32, gradOutLocal, RoundMode::CAST_NONE, alignedNum);
Mul(derivFp32, selfFp32, derivFp32, alignedNum);
Cast(resultLocal, derivFp32, RoundMode::CAST_RINT, alignedNum);
```

#### 机制与设计原因

`tmpBuf` 先承载 `selfFp32`。完成第二次 `Compares` 后，后续导数 `Select` 只读取 mask 与 `derivFp32`，不再读取 `selfFp32`；于是源码把同一 `tmpBuf` 覆盖为 `gradOutput` 的 FP32 表示。这样升精度路线只需两个 `U*4` FP32 TBuf，而非为 self、gradOutput、derivative 分配三个。最终乘法仍在 FP32 完成，再以 `CAST_RINT` 降回原 dtype。

#### 迁移边界

- 适用：逐元素反向算子中，一个升精度输入的最后消费点早于另一个输入升精度时。
- 前提：覆盖前必须完成 `selfFp32` 的全部读取；两个阶段容量、dtype 与对齐相同。
- 失效：若后续计算仍读取原 `selfFp32`，必须保留独立 buffer 或重新安排生命周期。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

在第二次 `Compares(selfFp32,3)` 完成前覆盖 `tmpBuf` 会改变上界掩码；这是源码读写顺序直接给出的反例。

### Trick T-002：按全局规模切换 Queue 深度并重算 UB tile

#### 触发与源码依据

- 适用 Route：全部；`N<=1024` 选 depth 1，`N>1024` 选 depth 2
- 触发条件：`useDoubleBuffer = totalIdx > MIN_SPLIT_THRESHOLD`
- Golden 来源：[`op_host/arch35/hard_swish_grad_tiling_arch35.cpp:146-159`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_host/arch35/hard_swish_grad_tiling_arch35.cpp#L146-L159)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t useDoubleBuffer = totalIdx > MIN_SPLIT_THRESHOLD;
int64_t bufferNum;
if (dataType == ge::DT_FLOAT) {
    bufferNum = (useDoubleBuffer != 0) ? SIZE_8 : SIZE_5;
} else {
    bufferNum = (useDoubleBuffer != 0) ? SIZE_11 : SIZE_8;
}
```

#### 机制与设计原因

Host 不只改变模板参数，还同步改变 UB 分母：FP32 采用 5/8，FP16/BF16 采用 8/11。这个选择同时决定三个 Queue 的 `BUFFER_NUM` 与每轮 `ubFactor`，避免把 depth 2 套到按 depth 1 估算的 tile 上。低规模保留更大单缓冲 tile，高规模启用两套 Queue slot；源码未提供性能百分比。

#### 迁移边界

- 适用：Queue depth 是编译期参数且 Host 能按规模选择模板的逐元素 AIV 算子。
- 前提：Host 的容量分母必须覆盖 dtype 特有的 Queue/TBuf/mask 组合，并和 Device `BUFFER_NUM` 一致。
- 失效：若 Kernel queue depth 或中间 buffer 数改变，必须同步更新 `bufferNum`，不能只改阈值。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓给出的可验证错误边界是 Host `bufferNum` 与 Device `BUFFER_NUM` 不一致会破坏 UB 预算；固定仓未提供运行时 fallback。

### Trick T-003：对齐计算、按有效长度搬出

#### 触发与源码依据

- 适用 Route：全部，尤其每核最后一个非 64 元素倍数的 tile
- 触发条件：`currentNum % 64 != 0`
- Golden 来源：[`op_kernel/arch35/hard_swish_grad.h:143-145`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad.h#L143-L145)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
constexpr int64_t ALIGN_ELEM =
    256 / static_cast<int64_t>(sizeof(ComputeType));
int64_t alignedNum =
    (currentNum + ALIGN_ELEM - 1) / ALIGN_ELEM * ALIGN_ELEM;
```

#### 机制与设计原因

所有路线的 `ComputeType` 都是 FP32，因此对齐粒度恒为 64 元素。Vector `Cast/Muls/Adds/Compares/Select/Mul` 处理 `alignedNum`，但 `CopyOut` 的 `blockLen` 只设为 `currentNum*sizeof(T)`。这样尾 tile 不需要单独的标量分支，无效 lane 始终留在已分配 UB 内且不会写入 GM。

#### 迁移边界

- 适用：Vector API 需要固定字节对齐、输出允许按有效长度 DMA 的逐元素算法。
- 前提：Queue/TBuf 容量至少为向上对齐后的 lane 数；本实现由 U 的 Host 对齐与 `currentNum<=U` 保证。
- 失效：若无效 lane 会参与归约、Atomic 或跨 tile 状态，必须显式清零或 mask，不能只截短 CopyOut。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 `alignedNum` 直接用于 GM CopyOut 会越过有效输出范围；源码以 `currentNum` 明确避免该行为。

## 模板与实现对比

| 对比维度 | R-001 FP32/SB | R-002 FP32/DB | R-003 FP16/BF16/SB | R-004 FP16/BF16/DB | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | `N<=1024` | `N>1024` | `N<=1024` | `N>1024` | dtype tuple + 全局阈值 |
| 模板与 dtype | `<float,0>` | `<float,1>` | `<half/bf16,0>` | `<half/bf16,1>` | `D_T_X,BUFFER_MODE` |
| 循环与分核 | 公共 block/tile 公式 | 同，U 分母更大 | 同 | 同，U 分母最大 | UB 驻留资源不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth1 + 1 FP32 TBuf + mask | depth2 + 1 FP32 TBuf + mask | depth1 + 2 FP32 TBuf + mask | depth2 + 2 FP32 TBuf + mask | dtype 精度与 Queue 深度 |
| 内存总占用与复用 | `16U+M` | `28U+M` | `14U+M`，tmpBuf 跨语义复用 | `20U+M`，同复用 | Host 分母分别 5/8/8/11 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad + FP32 Vector | 同 | DataCopyPad + Cast/FP32 Vector/Cast | 同 | 全部 AIV_VECTOR，无其它执行族 |
| 同步、精度与 tail | Queue；FP32；64 lane | depth2 Queue；FP32 | Queue；FP32 中间；RINT | depth2；FP32 中间；RINT | CopyOut 仅有效 n |
| fallback/失败边界 | 无 Device fallback | 无 | 无 | 无 | 非法输入/API/Tiling 失败；空域 ACLNN 早退 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 表标为支持，但固定扫描范围没有 A2 OpDef、binary、Tiling、Kernel 与活动 Host 接线，未达到 `source_closed` | 同 A2 | 公开支持，且 `ascend950 + arch35` 全链路 `source_closed` |
| Host/Launcher | ACLNN Level-2 写有 910/910B dtype 分表，但 `GetDtypeSupportListV2` 的平台映射与 A2 Host 实现不在固定范围，不能补齐 | 同 A2 | Level-2 共用；Level-0 添加 `HardSwishGrad` AICore launcher；arch35 Tiling |
| Kernel/资源 | 固定范围不可见，不生成无后缀卡 | 固定范围不可见，不生成无后缀卡 | `hard_swish_grad<D_T_X,BUFFER_MODE>`；AIV Vector Queue/TBuf；workspace 0 |

## 限制与失败边界

- 公开 ACLNN 对空 Tensor 成功早退、workspace 为 0且不启动算子 Kernel；Tiling 内的空域分支不是该入口的 Device Route。
- `CheckShapeValid` 比较两个输入和输出 shape，但 arch35 Tiling 只再次比较两个输入的元素总数；复用 direct Host 时不能丢掉 ACLNN 的完整 shape 校验。
- Tiling 从 `int64_t totalIdx` 推导 `blockFactor/ubFactor`，最终把 `usedCoreNum` 窄化为 `uint32_t`；固定源码没有显式溢出检查。
- 平台 core/UB 为 0时 Tiling 失败；计算后的 `ubFactor` 没有显式正数检查，依赖 Ascend950 平台资源。
- `alignedNum` 范围内的无效 lane 会参与逐元素 Vector 计算，但只复制 `currentNum` 个结果；该策略不能直接迁移到会聚合无效 lane 的算法。
- 框架 `Contiguous/ViewCopy` 的内部 Kernel、workspace slice、event 与 identity fast path 不在固定目录；它们按 ACLNN 框架边界记录，算子拥有的 workspace 明确为 0。
- 公开头注释声称 ND/NCHW/NHWC，正式文档和 OpDef 注册为 ND；本卡只承诺 ND。
- A2/A3 虽有公开支持声明，但指定固定范围未提供相应全链路实现，未生成 `hardswish_backward.md`，不得用 950 arch35 body 推测补齐。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/hard_swish_grad/docs/aclnnHardswishBackward.md` | `aclnnHardswishBackward*` | 公开性、产品、dtype/shape、两段式入口 |
| 入口声明/约束 | `activation/hard_swish_grad/op_host/op_api/aclnn_hardswish_backward.h` | 两个 `ACLNN_API` 声明 | 参数顺序与框架路径 |
| 入口实现 | `activation/hard_swish_grad/op_host/op_api/aclnn_hardswish_backward.cpp` | `CheckParams`、两个 ACLNN 函数 | 校验、空域、Contiguous/HardSwishGrad/ViewCopy、executor |
| Host/Launcher | `activation/hard_swish_grad/op_host/op_api/hard_swish_grad.cpp` | `l0op::HardSwishGrad` | 临时输出与 launcher 添加 |
| Host/Tiling/Launcher | `activation/hard_swish_grad/op_host/arch35/hard_swish_grad_tiling_arch35.cpp` | `HardSwishGradTilingFunc` | 分核、UB、workspace、selector |
| POD/ABI | `activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad_tiling_data.h` | `HardSwishGradTilingData` | 三字段 ABI |
| Selector | `activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | dtype×buffer mode 实例空间 |
| Kernel entry/body | `activation/hard_swish_grad/op_kernel/hard_swish_grad_apt.cpp` | `hard_swish_grad<D_T_X,BUFFER_MODE>` | 模板 entry 与 POD 解析 |
| Kernel entry/body | `activation/hard_swish_grad/op_kernel/arch35/hard_swish_grad.h` | `HardSwishGrad<T,BUFFER_MODE>` | GM/UB、Vector、同步、精度、tail |
| 构建接线 | `activation/hard_swish_grad/op_host/hard_swish_grad_def.cpp` | `HardSwishGrad` OpDef | Ascend950 注册与 `hard_swish_grad_apt` |
| 构建接线 | `activation/hard_swish_grad/op_host/config/ascend950/hard_swish_grad_binary.json` | `HardSwishGrad_fp32/fp16/bf16` | 三 dtype binary 配置 |
| 构建接线 | `activation/hard_swish_grad/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | Ascend950/arch35 构建选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 三个 `int64_t` 描述总量、每核量、每 tile 量 | 字段顺序、64 位、元素单位；Host/Device 同结构 | `HardSwishGradTilingData` |
| 调用顺序 | Level-2 前处理→Level-0 launcher→ViewCopy；Kernel `Init→Process` | 两输入次序是 gradOutput/self；输出第三参数；tiling 最后 | ACLNN GetWorkspaceSize、模板 entry |
| workspace/内存 | 算子 workspace 0；UB 按 dtype/Queue 深度预算 | Queue depth 与 Host `bufferNum` 必须同步；mask 256B 对齐 | `HardSwishGradTilingFunc`、`Init` |
| 同步 | Queue EnQue/DeQue/Free；Vector 顺序依赖 | tmpBuf 覆盖必须晚于 selfFp32 最后读取 | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | FP16/BF16 在 FP32 计算并 `CAST_RINT`；64 lane 对齐后仅写有效 n | 阈值闭区间、cast 顺序、CopyOut 长度不可改 | `Compute`、`CopyOut` |
| 失败边界 | API 参数失败、空域成功早退、Tiling 平台/shape/dtype失败 | 不得把空域变成 Kernel 路线；无 Device fallback | `CheckParams`、`HardSwishGradTilingFunc` |
