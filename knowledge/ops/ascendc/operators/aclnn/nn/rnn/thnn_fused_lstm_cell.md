---
type: Operator
title: "ThnnFusedLstmCell A2/A3 ACLNN 算子知识"
description: "说明 aclnnThnnFusedLstmCell 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐 Route AIV 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/docs/aclnnThnnFusedLstmCell.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_api/aclnn_thnn_fused_lstm_cell.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_api/thnn_fused_lstm_cell.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_host/thnn_fused_lstm_cell_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_host/thnn_fused_lstm_cell_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_host/thnn_fused_lstm_cell_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/gen_ops_info.cmake
    role: implementation
created_at: "2026-08-25T18:36:16Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ThnnFusedLstmCell A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 最新且唯一公开两段式 `aclnnThnnFusedLstmCellGetWorkspaceSize` / `aclnnThnnFusedLstmCell` |
| 计算语义 | 接收已完成矩阵乘的四门张量，叠加 hidden gates 与两份 bias，计算 `i/f/o=sigmoid`、`g=tanh`、`cy=f*cx+i*g`、`hy=o*tanh(cy)` |
| 输入输出 | 输入 `inputGates/hiddenGates:[B,4H]`、`cx:[B,H]`、可选 bias `[4H]`；输出 `hy/cy:[B,H]`、`storage:[B,4H]`；ND、FP16/FP32 且全体同 dtype |
| 执行类型 | 主计算为单 Kernel `AIV_VECTOR`；ACLNN 的 Contiguous、ZerosLike 与 ViewCopy 是框架辅助节点；无 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001 FP32 原生 Vector；R-002 FP16 搬入后升至 FP32 计算、RINT 降精度输出 |
| 核心资源 | GM；一个 `TQueBind<VECIN,VECOUT,2>`，每个 slot 为 `17*taskSize` 个 FP32；FP16 视图原位别名；系统 workspace 由平台库给出，主 Kernel 不读写 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN → 校验/连续化/可选零 bias → level-0 launcher → OpDef/Tiling/POD → 单 AIV entry → FP16/FP32 模板 body → 模块收集、平台检查、Kernel copy/编译接线 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `inputGates`、`hiddenGates` 每行按 Device 实际 offset 切为 `[i,f,g,o]` 四段，每段长度为 `H`，两份 bias 同样切分。对每个 `(b,h)`：

```text
zq = inputGates_q + hiddenGates_q + inputBias_q + hiddenBias_q
i = sigmoid(zi), f = sigmoid(zf), g = tanh(zg), o = sigmoid(zo)
cy = f * cx + i * g
hy = o * tanh(cy)
storage = concat(i, f, g, o)
```

公开 README/ACLNN 文档把 split/concat 次序写成 `[i,g,f,o]`，但固定实现 `FillTaskInfo` 明确令 `fOffset=H+colOffset`、`gOffset=2H+colOffset`，`ProcessCopyOut` 也写回同一 offset，因此复用 Device 实现时必须遵守源码的 `[i,f,g,o]` 物理布局；本文不推测文档与实现差异的上层兼容方式。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `inputGates` | `[B,4H]`，ND，可非连续 | FP16/FP32 | 必选；为空时 ACLNN 直接返回空执行器路径，workspace 为 0 |
| `hiddenGates` | `[B,4H]`，ND，可非连续 | 与 `inputGates` 相同 | 必选；四门的 hidden 侧矩阵乘结果 |
| `cx` | `[B,H]`，ND，可非连续 | 与 `inputGates` 相同 | 必选；上一时刻 cell state，并由其 shape 定义 `B,H` |
| `inputBiasOptional` | `[4H]`，ND，可非连续 | 与 `inputGates` 相同 | 可选；两份 bias 必须同时存在或同时为空；为空时各由 `ZerosLike` 物化为零 |
| `hiddenBiasOptional` | `[4H]`，ND，可非连续 | 与 `inputGates` 相同 | 可选；存在性与 `inputBiasOptional` 绑定 |
| `hyOut` | `[B,H]`，ND | 与 `inputGates` 相同 | 必选；通过临时 base output 与 ViewCopy 写入 |
| `cyOut` | `[B,H]`，ND | 与 `inputGates` 相同 | 必选；通过临时 base output 与 ViewCopy 写入 |
| `storageOut` | `[B,4H]`，ND | 与 `inputGates` 相同 | 必选；保存四门激活供反向使用；固定 Device 物理次序 `[i,f,g,o]` |

### 入口与版本

固定仓只公开这一组无版本后缀接口；不存在同语义的 V2/V3，因此它就是 A2/A3 实际支持的最新公开版本。`aclnnThnnFusedLstmCellGetWorkspaceSize` 组装 executor，`aclnnThnnFusedLstmCell` 通过 `CommonOpExecutorRun` 在给定 stream 上执行。README 与 ACLNN 文档均明确支持 Atlas A2、A3，明确不支持 Ascend 950；OpDef 对 `ascend910b`、`ascend910_93` 注册同一配置。构建侧在没有 binary JSON 时由 `get_op_type_and_validate` 调用 `check_op_supported` 检查 OpDef 的 `.AddConfig("<compute_unit>")`；因此 A2/A3 可达，950 会被判为不支持，与公开表一致。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParamsValid` 的真实顺序是：必选指针及 bias 成对存在性 → 若 `inputGates.IsEmpty()` 则立即成功 → 所有 tensor rank → 以 `cx` 得到 `B,H` 并校验 shape 关系 → FP16/FP32 与全体同 dtype。空输入终点发生在 rank/shape/dtype 校验之前，随后第一段接口把 `workspaceSize=0` 并释放 executor，不添加主 Kernel。

非空路径先对三项必选输入和已存在的两份 bias 调用 `l0op::Contiguous`。bias 均为空时，executor 分配 `[4H]` 临时 tensor，并分别调用两次 `ZerosLike` 生成零 bias。随后分配三个连续 base output，调用 `l0op::ThnnFusedLstmCell` 添加 AICore launcher，最后用三次 `ViewCopy` 把 base output 提交到用户输出。Contiguous、ZerosLike、ViewCopy 与 executor 调度属于框架边界；算子自有 selector、POD、entry 与 body 已继续闭合。

### 参数、Workspace 与生命周期

第一段接口返回 `uniqueExecutor->GetWorkspaceSize()`，该值包含框架为 Contiguous、可选 ZerosLike、base outputs、ViewCopy 和 launcher 所需的总量；固定仓未公开 executor 的逐 slice offset。主算子 Tiling 将 `workspace[0]` 设置为 `PlatformAscendC::GetLibApiWorkSpaceSize()`，注释明确“未使用用户workspace”；Device entry 虽接收 `workspace`，但不传给 `Init` 且 body 不读写它。

临时连续输入、零 bias 和 base outputs 均由 executor 所有，第二段接口执行完成前有效。用户提供 workspace 与 executor；`CommonOpExecutorRun` 使用用户 stream 保证 executor 内节点按其记录顺序执行，任一前处理、主 launcher 或 ViewCopy 添加失败都会在第一段构图时返回错误，第二段只执行成功构建的图。

### 启动图与失败传播

```text
非空 ACLNN
  -> [framework] Contiguous(inputGates, hiddenGates, cx, optional biases)
  -> [optional framework] ZerosLike([4H]) x 2
  -> node K0: ThnnFusedLstmCell / thnn_fused_lstm_cell
       blockDim = aivNum, POD = ThnnFusedLstmCellTilingData
       inputs = 5 个连续 GM tensor, outputs = 3 个 base output
       workspace = 系统 workspace 参数（body 未使用）
  -> [framework] ViewCopy(hy, cy, storage) x 3
```

K0 是唯一算子自有 Device 节点，执行族为 `AIV_VECTOR`。同一 stream/executor 建立前处理→K0→ViewCopy 的 ordering；中间 tensor 由 executor 持有至执行结束。K0 tiling 获取不到有效 AIV core 数时返回 `GRAPH_FAILED`，不会启动 Kernel；level-0 `ADD_TO_LAUNCHER_LIST_AICORE` 失败映射为 `ACLNN_ERR_INNER_NULLPTR`。空输入不启动任何自有 Kernel。

## Route 与参数设计

### 分核与资源计算

平台提供 `UB=ubByteSize` 与 `C=aivNum`。Host 以 FP32 为统一计算粒度：

```text
taskSize = floor((floor(uint32(UB)/2) - 8192) / 17 / 256) * 64  # FP32 元素
col = ceil(H / taskSize)                                         # 每行任务块数
tailSize = (uint32(H) % taskSize != 0) ? uint32(H)%taskSize : taskSize
taskTotal = B * col
taskSingle = ceil(taskTotal / C)
blockDim = C
```

`taskSize` 始终是 64 个 FP32（256 B）的整数倍。每核获得连续的 `taskSingle` 个 task；`taskIdx/C` 不用于地址计算，实际映射为 `taskIdxStart=blockIdx*taskSingle`，循环中超出 `taskTotal` 立即返回。task 再映射到 `rowIdx=taskIdx/col` 与 `colIdx=taskIdx%col`。没有 former/tail core POD；最后一个有效 core 和最后一列分别由 `taskTotal` guard、`tailSize` 处理。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 原生 Vector | 无 TilingKey；`ThnnFusedLstmCell<float>` | 非空、全部 dtype 为 FP32；kernel build 的 `DTYPE_HY=float` | dtype 为 FP16 时进入 R-002；其它 dtype 在 ACLNN 校验失败 |
| 2 | R-002 FP16 升精度 Vector | 无 TilingKey；`ThnnFusedLstmCell<half>` | 非空、全部 dtype 为 FP16；kernel build 的 `DTYPE_HY=half` | 其它 dtype 在 ACLNN 校验失败 |

```text
inputGates.empty -> workspace=0，结束
else dtype == FP32 -> R-001: ThnnFusedLstmCell<float>
else dtype == FP16 -> R-002: ThnnFusedLstmCell<half>
else -> ACLNN_ERR_PARAM_INVALID
```

这里没有运行时 TilingKey/router 分支；`DTYPE_HY` 由算子 dtype 编译实例选择模板。R-001 与 R-002 的 POD 相同，但搬入位置、cast、输出子区与精度顺序不同，不能合并为同一 Route。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 Vector | `DTYPE_HY=float` | 非空；八个 tensor 均 FP32；shape 为 `[B,4H]/[B,H]/[4H]`；A2/A3 | 单节点 K0 `thnn_fused_lstm_cell` | `ThnnFusedLstmCell<float>::ProcessOneTask` → `ProcessCopyIn/ProcessCompute/ProcessCopyOut` | GM 直接搬入 17 个 FP32 子区；Queue 在 MTE2↔V 与 V↔MTE3 交接；逐 Vector 依赖用 `PIPE_V` barrier | 每 task 的 `x=taskSize/tailSize`；超额 core/task 返回；无其它 fallback |
| R-002 FP16 升精度 Vector | `DTYPE_HY=half` | 非空；八个 tensor 均 FP16；shape 同上；A2/A3 | 单节点 K0 `thnn_fused_lstm_cell` | `ThnnFusedLstmCell<half>::ProcessOneTask` → `ProcessCopyInHalf/ProcessCastIn/ProcessCompute/ProcessCopyOutHalf` | FP16 搬到每个 FP32 子区后半，原位 cast 到前半 FP32；输出 cast 到已失效子区的 FP16 前半；其余同步同 R-001 | 同 R-001；不满足 dtype 无 Device fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| AIV core 有效 | `aivNum != 0` | blockDim 与 task 分配均依赖 AIV 数 | R-001/R-002 | Tiling `GRAPH_FAILED` | `op_host/thnn_fused_lstm_cell_tiling.cpp::GetInfo` |
| UB 统一上界 | `taskSize=floor((UB/2-8192)/17/256)*64` | Queue depth=2，每 slot 同时容纳 17 个 FP32 子区，并为库/管线留出每半 UB 的 8 KiB 预算 | R-001/R-002 | 固定源码没有显式 `taskSize==0` fallback；平台 UB 必须使其非零 | `FillTilingInfo`; `ThnnFusedLstmCell::Init` |
| 256 B task 对齐 | `taskSize % 64 == 0` | 高精度 tanh 的 CompareScalar mask 把 `x` 向上扩到 64 个 FP32 的整数倍，且需留在已分配子区内 | R-001/R-002 | Host 公式天然满足 | `FillTilingInfo`; `TanhPartialHighPrecision` |
| 非空短路 | `inputGates.IsEmpty()` | 避免 `H=0` 时 `ceil(H/taskSize)` 后的 `col-1` 与无效 GM 访问 | R-001/R-002 | ACLNN 返回 workspace 0，不启动 K0 | `CheckParamsValid`; `aclnnThnnFusedLstmCellGetWorkspaceSize` |
| 尾列 | `colIdx==col-1` | GM 只允许处理该行剩余 `tailSize` 元素 | R-001/R-002 | `x=taskSize` 用于非尾列 | `ThnnFusedLstmCell::FillTaskInfo` |
| dtype 编译特化 | `DTYPE_HY` 为 `float` 或 `half` | FP16 route 需要升/降精度和不同 UB 子区；FP32 不需要 cast | R-001/R-002 | API 拒绝其它 dtype | kernel entry 与 `ProcessOneTask` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 的宏定义位于仓外 CANN Host 头；固定仓可见字段顺序和 getter/setter，但 `sizeof/alignof/offsetof` 展开不可见。Host 用 `SaveToBuffer/GetDataSize` 序列化，Device 用 `GET_TILING_DATA` 解析同一注册类。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ThnnFusedLstmCellTilingData` | `B` | `uint64_t`/64 | 第 1 字段；精确 offset/总 sizeof 固定仓不可见 | 行/batch | `set_B(uint64(cxShape[0]))` | GM extent、`taskTotal` 地址映射 | 非空路径应为正；int64 shape 转 uint64 未另做溢出检查 |
| 同上 | `H` | `uint64_t`/64 | 第 2 字段；精确 offset 固定仓不可见 | 元素/行 | `set_H(uint64(cxShape[1]))` | gate/bias/common offset 与 GM extent | API shape 关系要求 gate 最后一维 `4H` |
| 同上 | `BH` | `uint64_t`/64 | 第 3 字段；精确 offset 固定仓不可见 | 元素 | `set_BH(B*H)` | GM tensor extent | `B*H` 固定源码未显式检查 uint64 溢出 |
| 同上 | `col` | `uint64_t`/64 | 第 4 字段；精确 offset 固定仓不可见 | task/行 | `set_col(ceil(H/taskSize))` | `taskIdx` 到 row/col 映射 | K0 只在非空路径启动 |
| 同上 | `taskTotal` | `uint64_t`/64 | 第 5 字段；精确 offset 固定仓不可见 | task | `set_taskTotal(B*col)` | 超额 task 终止条件 | 乘法无显式溢出检查 |
| 同上 | `taskSingle` | `uint64_t`/64 | 第 6 字段；精确 offset 固定仓不可见 | task/core | `set_taskSingle(ceil(taskTotal/aivNum))` | 每核起点与循环上界 | `aivNum==0` 已失败 |
| 同上 | `taskSize` | `uint32_t`/32 | 第 7 字段；精确 offset 固定仓不可见 | FP32 元素/task | UB 公式后 `set_taskSize` | UB slot 容量、列 offset、非尾 `x`、FP16 alias offset | 固定仓未显式处理 0；应为 64 的倍数 |
| 同上 | `tailSize` | `uint32_t`/32 | 第 8 字段；精确 offset/总 sizeof 固定仓不可见 | 元素 | `set_tailSize(uint32(H)%taskSize ?: taskSize)` | 尾 task 的 `info.x` | `H` 先窄化为 uint32 再取模，超大 H 行为不得外推 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 Vector | `DTYPE_HY=float` | `ThnnFusedLstmCellTilingFunc/FillTilingInfo` | `ThnnFusedLstmCellTilingData` 八字段 | `op_kernel/thnn_fused_lstm_cell.cpp::thnn_fused_lstm_cell` → `ThnnFusedLstmCell<float>` | `op_kernel/thnn_fused_lstm_cell.h::ProcessCopyIn/ProcessCompute/ProcessCopyOut/TanhPartialHighPrecision` | 算子/类别 CMake 收集；`add_modules_sources` 记录 `COMPILED_OP_DIRS`；`gen_ops_info` 以 OpDef `ascend910b/ascend910_93` 判定支持并复制/编译 `op_kernel` |
| R-002 FP16 升精度 Vector | `DTYPE_HY=half` | 同 R-001 | 同 R-001 | 同一 entry → `ThnnFusedLstmCell<half>` | `ProcessCopyInHalf/ProcessCastIn/ProcessCompute/ProcessCopyOutHalf/TanhPartialHighPrecision` | 同 R-001 |

## Kernel 实现

### 公共执行流程

entry 依次 `GET_TILING_DATA`、构造 `ThnnFusedLstmCell<DTYPE_HY>`、`Init`、`Process`。`Init` 绑定五个输入和三个输出 GM extent，只初始化一个 depth=2 的双向 Queue；两个 slot 每个都可容纳 `17*taskSize` 个 FP32。每个 core 顺序处理自己的 task，单 task 一次分配整 slot、完成搬入→计算→搬出后释放。四门分别占四组四个子区，`cx` 占第 17 子区；计算阶段会就地覆盖已经消费的输入子区。

所有 gate 的加法顺序固定为 `input + hidden`，再加 `inputBias`，最后加 `hiddenBias`。两处 tanh 都先调用 AscendC `Tanh`，随后对 `abs(x)<=0.55` 的 lane 用奇次多项式结果覆盖。Queue 的 EnQue/DeQue 建立 MTE2→V 与 V→MTE3 阶段可见性；同一 Vector pipe 上的真实读写依赖由每步后的 `PipeBarrier<PIPE_V>()` 保护。

### Route R-001：FP32 原生 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_thnn_fused_lstm_cell.cpp` | `CheckDtypes` | `inputGates` 属于 `{DT_FLOAT,DT_FLOAT16}` 且其余存在 tensor dtype 全等 | FP32 合法，进入 level-0；否则参数错误 |
| Host/Tiling | `op_host/thnn_fused_lstm_cell_tiling.cpp` | `ThnnFusedLstmCellTilingFunc` | `aivNum!=0`，无 TilingKey 分支 | 写公共 POD，`blockDim=aivNum` |
| Router | `op_kernel/thnn_fused_lstm_cell.cpp` | `thnn_fused_lstm_cell` | 编译期 `DTYPE_HY=float` | 实例化 `ThnnFusedLstmCell<float>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::ThnnFusedLstmCell -> ADD_TO_LAUNCHER_LIST_AICORE
  -> thnn_fused_lstm_cell(..., workspace, tiling)
    -> ThnnFusedLstmCell<float>::Init
    -> Process
      -> FillTaskInfo
      -> ProcessOneTask
        -> ProcessAlloc -> ProcessCopyIn
        -> ProcessI -> ProcessG -> ProcessF -> ProcessO
        -> ProcessCy -> ProcessCTanh -> ProcessHy
        -> ProcessCopyOut -> que.FreeTensor
```

##### 源码等价伪码

```text
start = blockIdx * taskSingle
for taskIdx in [start, start + taskSingle):
  if taskIdx >= taskTotal: return
  row = taskIdx / col; tile = taskIdx % col
  x = (tile == col-1) ? tailSize : taskSize
  common = row*H + tile*taskSize
  gateBase = 4*row*H + tile*taskSize
  offsets(i,f,g,o) = gateBase + (0,H,2H,3H)
  alloc one float slot [17, taskSize]
  DataCopyPad 16 gate/bias slices + cx slice, each x*sizeof(float)
  queue MTE2 -> V
  i = sigmoid(xi+xhi+bi_i+bh_i)                 # BUF1 in-place
  g = corrected_tanh(xg+xhg+bi_g+bh_g)          # BUF6 result
  f = sigmoid(xf+xhf+bi_f+bh_f)                 # BUF9 in-place
  o = sigmoid(xo+xho+bi_o+bh_o)                 # BUF13 in-place
  cy = f*cx + i*g                               # BUF3
  hy = o*corrected_tanh(cy)                     # BUF15
  queue V -> MTE3
  copy i/g/f/o to their physical offsets; copy hy/cy to common
  free slot
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| gate inputs GM | `4*row*H + q*H + colIdx*taskSize`，`q={0,1,2,3}` 对应 i/f/g/o | 每路 `x`，全 tensor `4BH` | FP32，ND | ACLNN contiguous/base input | `ProcessCopyIn` | executor/user，K0 全程 | DataCopyPad 支持尾字节 |
| bias GM | `q*H + colIdx*taskSize` | 每路 `x`，全 tensor `4H` | FP32，ND | 用户或 ZerosLike | `ProcessCopyIn` | executor/user，K0 全程 | 每 batch 重读同一 bias slice |
| `cx` GM | `row*H+colIdx*taskSize` | `x`，全 tensor `BH` | FP32，ND | 用户 contiguous | `ProcessCopyIn` | K0 全程 | 与 hy/cy 地址公式相同但对象不同 |
| Queue slot | `slotBase + k*taskSize*4`，`k=0..16` | `17*taskSize*4` B/slot，depth=2 | FP32 子区数组 | `ProcessCopyIn` | Vector pipeline/`ProcessCopyOut` | 单 task alloc→free | `taskSize*4` 为 256 B 倍数；大量子区就地覆盖 |
| output GM | gate offset 或 common offset | storage `4BH`，hy/cy 各 `BH` | FP32，ND | `ProcessCopyOut` | ViewCopy/用户 | K0 后有效 | 每次只写 `x` 个有效元素 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  slot0["Queue slot 0 · 17 × taskSize × 4 B"]:2
  slot1["Queue slot 1 · 17 × taskSize × 4 B"]:2
  gates["BUF1..BUF16 · 四门输入/偏置，随后原位复用"]:3
  cxbuf["BUF17 · cx，后续可复用"]:1
```

##### 字节账与复用顺序

```text
queue_bytes = depth 2 * 17 * taskSize * sizeof(float)
UB_total = 2 * 17 * taskSize * 4 B
taskSize = floor((UB/2 - 8192)/17/256)*64
=> 17*taskSize*4 <= UB/2 - 8192（考虑整除向下取整）
=> UB_total <= UB - 16384
dynamic_ub = N/A（AIV Queue 静态容量）
workspace_total(main body) = 0；Tiling workspace[0] = GetLibApiWorkSpaceSize()
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| BUF1/5/9/13 | gate CopyIn | 各门 Add/激活 | BUF1/5/9/13 原位写和；其中 BUF5 最终不保存 g | 每条依赖后 `PIPE_V` barrier | 单 task 内严格顺序 |
| BUF2/3/4 等输入子区 | CopyIn | 当前门加法或后续 `cx` | 消费后成为 g/cy/tanh 临时 | `PIPE_V` barrier；下一 task 前 Free/Alloc | 不跨 task 保留 UB 值 |
| BUF6 | hidden-g 输入 | G 加法先读，之后写 builtin/corrected tanh | 作为 g 保留到 cy 与 CopyOut | barriers 分隔 Add/Tanh/Select/Mul | 覆盖发生在 hidden-g 已消费后 |
| BUF3 | input-bias-i 输入 | I 加法读取；随后写 `f*cx` 与最终 cy | CTanh 读取后可释放 | barriers | storage i 已在 BUF1，不依赖 BUF3 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + Queue EnQue/DeQue | MTE2 | Vector `ProcessI` | 整个 float slot | `blockCount=1, blockLen=x*4 B, stride=0`；只处理 x |
| gate 计算 | `PipeBarrier<PIPE_V>` | 每次 Add/Sigmoid/Tanh/Select | 下一次读写同子区的 Vector op | 原位 gate/temp 子区 | 全 FP32；顺序不可交换 |
| tanh 修正 | `CompareScalar` + `Select` | builtin tanh 与多项式 | 后续 cy/hy | abs(x)<=0.55 lane | mask 计数上取整到 64 FP32，Select 只提交 x lane |
| UB→GM | Queue EnQue/DeQue + `DataCopyPad` | Vector `ProcessHy` | MTE3 CopyOut | 整个 float slot | `blockLen=x*4 B`，不写 tail 之外 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/thnn_fused_lstm_cell_tiling.cpp` | `FillTilingInfo/ThnnFusedLstmCellTilingFunc` | 由 UB/AIV/H/B 推导 task、POD、blockDim、系统 workspace |
| Kernel entry/router | `op_kernel/thnn_fused_lstm_cell.cpp` | `thnn_fused_lstm_cell` / `ThnnFusedLstmCell<float>` | `DTYPE_HY` 直接决定模板，无 TilingKey |
| Device body/helper | `op_kernel/thnn_fused_lstm_cell.h` | `ProcessCopyIn/ProcessCompute/ProcessCopyOut/TanhPartialHighPrecision` | 17 路搬入、四门激活、cell/hidden 更新、尾长搬出 |

### Route R-002：FP16 升精度 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_thnn_fused_lstm_cell.cpp` | `CheckDtypes` | 全部存在 tensor dtype 等于 FP16 | 进入 level-0；混合 dtype 失败 |
| Host/Tiling | `op_host/thnn_fused_lstm_cell_tiling.cpp` | `FillTilingInfo` | taskSize 仍按 FP32 资源推导 | POD 与 R-001 相同 |
| Router | `op_kernel/thnn_fused_lstm_cell.cpp` | `thnn_fused_lstm_cell` | 编译期 `DTYPE_HY=half` | 实例化 `ThnnFusedLstmCell<half>`，命中 constexpr half 分支 |

#### Device 数据流

##### 函数级调用链

```text
l0op::ThnnFusedLstmCell -> ADD_TO_LAUNCHER_LIST_AICORE
  -> thnn_fused_lstm_cell
    -> ThnnFusedLstmCell<half>::Init
    -> Process -> FillTaskInfo -> ProcessOneTask
      -> ProcessAlloc -> ProcessCopyInHalf -> ProcessCastIn
      -> ProcessI/G/F/O/Cy/CTanh/Hy（FP32）
      -> ProcessCopyOutHalf -> que.FreeTensor
```

##### 源码等价伪码

```text
按 R-001 计算 task 与 GM offset
alloc [17, taskSize] float slot，并为每个子区建立 alias half[2*taskSize]
for each of 17 inputs:
  copy x 个 half 到 alias[taskSize : taskSize+x]       # 每个 FP32 子区后半
for k=0..16:
  cast float[k,0:x] <- half_alias[k,taskSize:taskSize+x], CAST_NONE
按 FP32 执行全部 gate、cy、hy 计算
cast i/g/f/o/cy/hy 到已消费子区的 half_alias[0:x]，CAST_RINT
queue V -> MTE3
copy six 个 half 结果各 x 个元素到对应 GM offset
free slot
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 输入 GM | 与 R-001 相同 | gate `4BH`、bias `4H`、cx `BH` | FP16，ND | ACLNN | `ProcessCopyInHalf` | K0 全程 | 每路 `blockLen=x*2 B` |
| 每个 FP32 子区后半 | `ubBase+k*taskSize*4 + taskSize*2` | `taskSize` 个 half | FP16 alias | DataCopyPad | `ProcessCastIn` | Cast 完成后失效 | 与同子区 FP32 前半同一物理区但不重叠 |
| 每个 FP32 子区前部 | `ubBase+k*taskSize*4` | `taskSize` 个 float | FP32 | Cast/Vector op | FP32 pipeline | 单 task | Cast source 后半→destination 全 FP32 的写入区间会覆盖 source 尾端；AscendC Cast 的该用法由固定源码明确采用 |
| FP16 输出别名 | 被选中的 BUF2/7/10/14/16/17 前 `taskSize` half | 每项 `x` half | FP16 alias | `Cast(..., CAST_RINT)` | `ProcessCopyOutHalf` | 对应原 FP32 输入已失效后到 CopyOut | 输出直接落到待搬出子区，无额外 output buffer |
| 输出 GM | gate/common offset | storage `4BH`，hy/cy `BH` | FP16，ND | `ProcessCopyOutHalf` | ViewCopy/用户 | K0 后有效 | 每路 `blockLen=x*2 B` |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  slot0["Queue slot 0 · 17 × taskSize × 4 B"]:2
  slot1["Queue slot 1 · 17 × taskSize × 4 B"]:2
  fp32["每子区全长 · taskSize float · 计算视图"]:2
  halfin["同子区后半 · taskSize half · 搬入/cast source"]:1
  halfout["复用子区前半 · taskSize half · cast output"]:1
```

##### 字节账与复用顺序

```text
queue_bytes = 2 * 17 * taskSize * 4 B
额外 FP16 alias bytes = 0（ReinterpretCast，不分配）
UB_total = 2 * 17 * taskSize * 4 B <= UB - 16384
dynamic_ub = N/A
workspace_total(main body) = 0；系统 workspace 同 R-001
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 所有 BUF 的 FP16 后半 | `ProcessCopyInHalf` | `ProcessCastIn` | Cast 写入同一子区 FP32 视图 | 17 次 Cast 后统一 `PIPE_V` barrier | Cast 源 offset 固定为 `taskSize` half |
| BUF2 前半 | `ProcessI` 的 RINT Cast | `ProcessCopyOutHalf` 写 storage-i | 下一 task 分配后输入覆盖 | Vector barrier 后继续；最终 Queue V→MTE3 | hidden-i 已在 I 的第一 Add 消费完 |
| BUF7/10/14 前半 | G/F/O 的 RINT Cast | storage-g/f/o CopyOut | 下一 task | 每次 Cast 后 `PIPE_V` barrier | 对应 bias/temp 输入已失效 |
| BUF17 前半 | Cy 的 RINT Cast | cy CopyOut | 下一 task | Cast 后 barrier | cx 已参与 `f*cx`，无需保留 |
| BUF16 前半 | Hy 的 RINT Cast | hy CopyOut | 下一 task | Queue V→MTE3 | hidden-o 输入已消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| FP16 GM→UB | DataCopyPad + Queue | MTE2 | `ProcessCastIn` | 每子区后半 FP16 | `blockLen=x*2 B` |
| 升精度 | `Cast(..., CAST_NONE)` | FP16 alias | FP32 gate pipeline | 17 个 FP32 子区 | 每项只 cast x；统一 V barrier 后计算 |
| 主计算 | FP32 Add/Mul/Sigmoid/Tanh/Select + barriers | FP32 子区 | 后续依赖 | gate/cy/hy | 中间值不先量化到 FP16；storage/cy/hy 才单独降精度 |
| 降精度 | `Cast(..., CAST_RINT)` | FP32 i/g/f/o/cy/hy | FP16 CopyOut | 六个复用 alias | round-to-nearest-integer 模式由源码固定；每项 x lane |
| FP16 UB→GM | Queue + DataCopyPad | Vector Cast | MTE3 | FP16 输出 alias | 不写尾块以外元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/thnn_fused_lstm_cell_tiling.cpp` | `FillTilingInfo` | FP16 仍按 17 个 FP32 子区确定 taskSize |
| Kernel entry/router | `op_kernel/thnn_fused_lstm_cell.cpp` | `ThnnFusedLstmCell<half>` | `DTYPE_HY=half` 命中 constexpr half 路线 |
| Device body/helper | `op_kernel/thnn_fused_lstm_cell.h` | `ProcessCopyInHalf/ProcessCastIn/ProcessCopyOutHalf` | 后半搬入、FP32 全流程、前半复用输出 |

## 算子特定优化

### Trick T-001：FP16 在同一 FP32 子区内完成搬入、升精度与输出复用

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：编译模板 `dtype=half`，每个 UB 子区容量为 `taskSize*sizeof(float)`。
- Golden 来源：[`rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.h:221-258`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.h#L221-L258)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void ProcessCopyInHalf()
// 搬入buf的后半以备CAST
{
    // igates -> ub
    CopyIn(ubsHalf[BUF1][tilingData.taskSize], igatesGM[info.iOffset]);
    CopyIn(ubsHalf[BUF5][tilingData.taskSize], igatesGM[info.gOffset]);
    CopyIn(ubsHalf[BUF9][tilingData.taskSize], igatesGM[info.fOffset]);
    CopyIn(ubsHalf[BUF13][tilingData.taskSize], igatesGM[info.oOffset]);
    // hgates -> ub
    CopyIn(ubsHalf[BUF2][tilingData.taskSize], hgatesGM[info.iOffset]);
    CopyIn(ubsHalf[BUF6][tilingData.taskSize], hgatesGM[info.gOffset]);
    CopyIn(ubsHalf[BUF10][tilingData.taskSize], hgatesGM[info.fOffset]);
    CopyIn(ubsHalf[BUF14][tilingData.taskSize], hgatesGM[info.oOffset]);
    // b_ih -> ub
    CopyIn(ubsHalf[BUF3][tilingData.taskSize], bIhGM[info.biOffset]);
    CopyIn(ubsHalf[BUF7][tilingData.taskSize], bIhGM[info.bgOffset]);
    CopyIn(ubsHalf[BUF11][tilingData.taskSize], bIhGM[info.bfOffset]);
    CopyIn(ubsHalf[BUF15][tilingData.taskSize], bIhGM[info.boOffset]);
    // b_hh -> ub
    CopyIn(ubsHalf[BUF4][tilingData.taskSize], bHhGM[info.biOffset]);
    CopyIn(ubsHalf[BUF8][tilingData.taskSize], bHhGM[info.bgOffset]);
    CopyIn(ubsHalf[BUF12][tilingData.taskSize], bHhGM[info.bfOffset]);
    CopyIn(ubsHalf[BUF16][tilingData.taskSize], bHhGM[info.boOffset]);
    // c0 -> ub
    CopyIn(ubsHalf[BUF17][tilingData.taskSize], c0GM[info.commonOffset]);

    que.EnQue<TPosition::GM, TPosition::VECIN, float>(ubTotal);
    ubTotal = que.DeQue<TPosition::GM, TPosition::VECIN, float>();
}

__aicore__ inline void ProcessCastIn()
{
    // fp16 -> fp32
    for (uint32_t i = 0; i < bufCnt; i++) {
        Cast(ubs[i], ubsHalf[i][tilingData.taskSize], RoundMode::CAST_NONE, info.x);
    }

    PipeBarrier<PIPE_V>();
}
```

#### 机制与设计原因

每个逻辑子区物理容量是 `taskSize*4 B`。源码先把 `taskSize` 个 half 放到 offset `taskSize`（即字节 offset `2*taskSize`）的后半区，再把它们 cast 到同一子区从 offset 0 开始的 `taskSize` 个 float。这样无需为 17 路输入分别再分配独立 FP16 Queue。输入消费后，BUF2/7/10/14/16/17 的前半又承接六路 RINT FP16 输出，因此输出也不额外物化。关键前提是单 task 内的严格生命周期与 `PIPE_V`/Queue ordering。

#### 迁移边界

- 适用：大量 FP16 输入需要统一 FP32 计算，且输入、计算、输出生命周期能严格分阶段的逐元素融合。
- 前提：每个物理子区为 FP32 容量；half source 放在后半；Cast 支持源码所用的同子区布局；输出目标子区对应的旧输入已消费。
- 失效：FP32 输入使用 R-001；若输入与输出生命周期重叠、dtype 宽度关系变化或 Cast 不支持该布局，必须分配独立 buffer。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把 FP16 source 放在子区前半再原位扩成 FP32，否则目的写入范围会覆盖尚未读取的 source；固定仓未提供除此生命周期约束外的可验证反例。

### Trick T-002：小幅值 tanh 用多项式结果替换通用 Tanh

#### 触发与源码依据

- 适用 Route：R-001、R-002 的 g gate 与 `tanh(cy)`。
- 触发条件：每个有效 lane 的 `abs(x)<=0.55`；其余 lane 保留通用 `Tanh` 结果。
- Golden 来源：[`rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.h:74-103`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.h#L74-L103)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Mul(temp1Tensor, inputTensor, inputTensor, calcSizeAlign);
PipeBarrier<PIPE_V>();
Muls(temp2Tensor, temp1Tensor, 0.016090461204f, calcSizeAlign);  // Tanh多项式系数
PipeBarrier<PIPE_V>();
Adds(temp2Tensor, temp2Tensor, -0.052421370438f, calcSizeAlign);  // Tanh多项式系数
PipeBarrier<PIPE_V>();
Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
PipeBarrier<PIPE_V>();
Adds(temp2Tensor, temp2Tensor, 0.133147126779f, calcSizeAlign);  // Tanh多项式系数
PipeBarrier<PIPE_V>();
Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
PipeBarrier<PIPE_V>();
Adds(temp2Tensor, temp2Tensor, -0.333324134737f, calcSizeAlign);  // Tanh多项式系数
PipeBarrier<PIPE_V>();
Mul(temp2Tensor, temp2Tensor, temp1Tensor, calcSizeAlign);
PipeBarrier<PIPE_V>();
Adds(temp2Tensor, temp2Tensor, 0.999999873294f, calcSizeAlign);  // Tanh多项式系数
PipeBarrier<PIPE_V>();
Mul(temp2Tensor, temp2Tensor, inputTensor, calcSizeAlign);
PipeBarrier<PIPE_V>();
// temp1Tensor: abs(x)
Abs(temp1Tensor, inputTensor, calcSizeAlign);
PipeBarrier<PIPE_V>();
// tanhMaskTensor: mask for abs(x) <= 0.55, inplace
auto tanhMaskTensor = temp1Tensor.template ReinterpretCast<uint8_t>();
CompareScalar(tanhMaskTensor, temp1Tensor, 0.55f, CMPMODE::LE, Ceil(calcSizeAlign, cnt256B) * cnt256B);  // Tanh多项式实现范围
PipeBarrier<PIPE_V>();
// pick higher result into tanhLowTensor
Select(tanhLowTensor, tanhMaskTensor, temp2Tensor, tanhLowTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, calcSizeAlign);
```

#### 机制与设计原因

Helper 用 Horner 形式在 FP32 上计算 `x*(c0+x²*(c1+x²*(c2+x²*(c3+x²*c4))))`，复用两个临时子区；`CompareScalar` 以 `ceil(x/64)*64` 个 FP32 lane 生成 mask，`Select` 只对 `x` 个有效 lane 合并。小幅值 lane 取多项式，其他 lane 保留先前通用 `Tanh` 结果。这是源码可证明的精度路径选择；固定仓没有真机误差或性能数据。

#### 迁移边界

- 适用：已有通用 tanh 结果、允许为 FP32 多项式准备两个临时子区、并希望对固定阈值内 lane 采用替代近似的 Vector 流程。
- 前提：阈值、五个系数、FP32 运算顺序、每步 `PIPE_V` barrier、mask 向上取整和 `Select` 有效长度完全保持。
- 失效：不能把多项式外推到 `abs(x)>0.55`；这些 lane 必须保留通用 `Tanh`。taskSize 的 256 B 对齐必须容纳 mask 的向上取整访问。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

把 `Select` 长度也扩到向上取整后的 mask 长度会写入无效 tail lane；固定实现仅扩 mask 生成长度，结果选择仍严格使用 `calcSizeAlign=info.x`。

## 模板与实现对比

| 对比维度 | R-001 FP32 原生 Vector | R-002 FP16 升精度 Vector | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 全体 FP32 | 全体 FP16 | ACLNN 同 dtype 校验；编译期 `DTYPE_HY` |
| 模板与 dtype | `ThnnFusedLstmCell<float>` | `ThnnFusedLstmCell<half>`，内部 FP32 | 输入输出 dtype 决定 constexpr 路径 |
| 循环与分核 | `aivNum` cores、连续 task、相同尾块 | 同左 | POD 完全共用 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth=2 Queue，17 个 FP32 子区/slot；无单独 TBuf、dynamic UB、通信 window | 同一物理容量并增加零开销 half alias | 统一以 FP32 计算资源定 tile |
| 内存总占用与复用 | `2*17*taskSize*4 B`；FP32 子区就地覆盖 | 同总量；FP16 后半搬入、前半输出复用 | FP16 不额外增加 UB |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad、Add/Mul/Sigmoid/Tanh/Compare/Select | 加入 Cast(CAST_NONE/RINT)；其余同左 | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | 全 FP32；Queue + PIPE_V；`x` 控制尾块 | FP16→FP32→FP16；同步/tail 同左 | R-002 保持中间 FP32 |
| fallback/失败边界 | 其它 dtype API 失败；AIV=0 tiling 失败 | 同左 | 无 Device fallback 与未知 key |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 公开 README 与 ACLNN 文档明确不支持 Ascend 950 |
| Host/Launcher | OpDef `ascend910b`，共用 ACLNN/Tiling；构建检查通过 AddConfig | OpDef `ascend910_93`，共用 ACLNN/Tiling；构建检查通过 AddConfig | 无 `ascend950` AddConfig，构建检查判不支持；不生成正式卡 |
| Kernel/资源 | 共用单 AIV body；UB/AIV 数在运行时从平台读取 | 共用单 AIV body；UB/AIV 数在运行时从平台读取 | 固定仓没有公开可达 Host/Kernel 闭包 |

## 限制与失败边界

- 非空路径要求 rank、shape 与 dtype 严格匹配；bias 必须同时存在或同时为空。空 `inputGates` 在 rank/shape/dtype 校验之前短路，固定实现不会继续验证其它 tensor 的这些属性。
- 固定实现的门物理次序为 `[i,f,g,o]`，与公开文档的 `[i,g,f,o]` 文字存在冲突；复用 Kernel 地址公式时不得按文档重排。
- `B*H`、`B*col` 没有显式 uint64 溢出检查；`tailSize` 用 `static_cast<uint32_t>(H)%taskSize`，超出 uint32 的 H 不得依据本文外推。
- Tiling 未提供 `taskSize==0` fallback，依赖 A2/A3 平台 UB 足以满足公式；`aivNum==0` 明确失败。
- ACLNN 返回的总 workspace 由 executor 计算，固定仓未公开框架辅助节点的逐 slice offset；主 Kernel 自身不使用 workspace。
- FP16 Cast 的同子区别名是固定源码行为；底层 AscendC primitive 的微架构实现属于框架边界。
- 950 被公开产品表明确标为不支持，因此未创建 `_950` 卡。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `rnn/thnn_fused_lstm_cell/docs/aclnnThnnFusedLstmCell.md` | 接口原型、产品表、参数表 | A2/A3 公开支持、950 不支持、shape/dtype 与两段式 API |
| 入口声明/约束 | `rnn/thnn_fused_lstm_cell/README.md` | 产品支持、公式 | 逻辑语义与平台交叉证据 |
| 入口实现 | `rnn/thnn_fused_lstm_cell/op_api/aclnn_thnn_fused_lstm_cell.cpp` | `CheckParamsValid/HandleParams/RunBaseOp/ViewCopyResults` | 校验、空域、连续化、零 bias、executor 图与 workspace |
| 入口实现 | `rnn/thnn_fused_lstm_cell/op_api/thnn_fused_lstm_cell.cpp` | `l0op::ThnnFusedLstmCell` | 单个 AICore launcher 添加与失败映射 |
| Host 定义 | `rnn/thnn_fused_lstm_cell/op_host/thnn_fused_lstm_cell_def.cpp` | `ops::ThnnFusedLstmCell` | dtype/format 与 ascend910b/ascend910_93 注册 |
| Host/Tiling/POD | `rnn/thnn_fused_lstm_cell/op_host/thnn_fused_lstm_cell_tiling.h` | `ThnnFusedLstmCellTilingData` | 八字段 POD 声明与注册 |
| Host/Tiling/Launcher | `rnn/thnn_fused_lstm_cell/op_host/thnn_fused_lstm_cell_tiling.cpp` | `GetInfo/FillTilingInfo/ThnnFusedLstmCellTilingFunc` | UB/task/core/workspace 参数生产 |
| Kernel entry | `rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.cpp` | `thnn_fused_lstm_cell` | POD 解析与模板实例化 |
| Kernel body | `rnn/thnn_fused_lstm_cell/op_kernel/thnn_fused_lstm_cell.h` | `ThnnFusedLstmCell<T>` 全部方法 | task 映射、UB、搬运、Vector、同步、精度、tail 与复用 |
| 构建接线 | `rnn/thnn_fused_lstm_cell/CMakeLists.txt` | `add_modules_sources` | `OPTYPE thnn_fused_lstm_cell` 模块接线，ACLNN 使用仓内独立公开实现 |
| 构建接线 | `rnn/CMakeLists.txt` | category 子目录扫描 | 活动构建递归加入算子根 CMake |
| 构建接线 | `cmake/func.cmake` | `add_modules_sources/add_tiling_sources` | 收集 op_api、OpDef、Tiling，并把算子根写入 `COMPILED_OP_DIRS` |
| 构建接线 | `cmake/gen_ops_info.cmake` | `get_op_type_and_validate/check_op_supported/kernel_src_copy/gen_ops_info_and_python` | 无 binary JSON 时按 OpDef AddConfig 判平台，复制 op_kernel，并接入实现生成与二进制编译 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 八字段共同驱动两种 dtype route | 字段顺序/位宽、元素单位与 Host/Device 同一注册类；精确 offset 不可猜测 | `ThnnFusedLstmCellTilingData`; `SetTilingData`; `GET_TILING_DATA` |
| 调用顺序 | 校验→连续化/零 bias→K0→ViewCopy | 空域短路位置、bias 成对规则、K0 前后节点顺序 | `aclnnThnnFusedLstmCellGetWorkspaceSize` |
| workspace/内存 | 主 body workspace 为 0；Queue 总量按统一 FP32 tile | depth=2 必须计入；17 子区容量和 FP16 alias offset 不可改 | `FillTilingInfo`; `ThnnFusedLstmCell::Init` |
| 同步 | Queue 负责 MTE↔V，barrier 负责 V 内依赖 | 原位覆盖前必须消费旧值；CopyOut 前必须完成 V→MTE3 交接 | `ProcessCopyIn*`; `ProcessCompute`; `ProcessHy` |
| 精度与尾块 | FP16 全程升 FP32 计算；两处 tanh 小值修正；`x` 控制有效范围 | Cast round mode、tanh 系数/阈值/顺序、mask 与 Select 长度、门物理 offset | `ProcessCastIn`; `TanhPartialHighPrecision`; `ProcessCopyOutHalf` |
| 失败边界 | AIV=0 tiling 失败；其它 dtype/shape API 失败；空输入不启动 | 不添加未实现的 fallback；不宣称 950 支持 | `GetInfo`; `CheckParamsValid`; `check_op_supported`; 公开产品支持表 |