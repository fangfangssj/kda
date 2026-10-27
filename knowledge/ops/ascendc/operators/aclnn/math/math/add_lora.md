---
type: Operator
title: "AddLora A2 ACLNN 算子知识"
description: "说明 aclnnAddLora 在 A2 上的接口语义、Host/Launcher 参数生成、Route 选择与 Kernel 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/docs/aclnnAddLora.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_host/add_lora_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_host/add_lora_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_host/add_lora_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_host/add_lora_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_kernel/add_lora.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_kernel/add_lora_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_kernel/add_lora_normal_core.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_kernel/add_lora_single_core.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_host/config/ascend910b/add_lora_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T10:05:01Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2]
cann_versions: ["9.1.0"]
---

# AddLora A2 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | CANN 9.1.0 公开两段式 API `aclnnAddLoraGetWorkspaceSize` / `aclnnAddLora`；固定仓只公开原型与自动生成接线，生成的 inner bridge 是框架边界。 |
| 计算语义 | 公开语义是按 `indices` 分组后计算 `Z1=X_i·W_A`（`weightAOptional` 缺省时跳过）、`Z2=Z1·W_B`，再将 `scale·Z2` 加到 `y[:, yOffset:yOffset+ySliceSize]`。固定 Device 源码实际对 `y` 的 `yOffset:yOffset+H2` 做 FP16 原子加，未写 `y_out`，详见限制。 |
| 输入输出 | `y:[B,H3]`、`x:[B,H1]`、`weightB:[W,L,H2,R]`、可选 `weightA:[W,L,R,H1]`、`indices:[B]`、`out:[B,H3]`；A2 均为 ND，数据为 FP16，`indices` 为 INT32。 |
| 执行类型 | 单 Kernel `MIX_CV`：一个 `add_lora` launch 以 `KERNEL_TYPE_MIX_AIC_1_2` 运行；AIC 完成两级手写 Cube `Mmad`，两个 AIV 子核完成分组搬运、FP16 scale 与原子累加。 |
| Kernel 路线 | R-001：`100000`/大 batch/有 `weightA`；R-002：`100000`/大 batch/无 `weightA`；R-003：`100001`/小 batch/有 `weightA`；R-004：`100001`/小 batch/无 `weightA`。 |
| 核心资源 | 用户 workspace 中的重排 `x`、MM1、MM2 和三个索引数组；AIC 的 L1、L0A/L0B/L0C；AIV 的双缓冲搬运 Queue 与 192 KiB UB；跨族 flag 2/8。 |
| 来源闭包 | `source_closed`：公开 API 约束、自动 ACLNN 接线、OpDef、Tiling/POD、A2 两个 key、四条运行分支、MIX_CV entry/body、同步与构建接线均已追到固定 Commit；生成 inner bridge 作为明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `indices[b]=w` 的样本属于 LoRA 权重组 `w`。公开接口描述的有效组计算为：

```text
weightA 存在：Z1[b, :] = x[b, :] · weightA[w, layerIdx, :, :]^T
weightA 缺省：Z1[b, :] = x[b, 0:R]
Z2[b, :] = Z1[b, :] · weightB[w, layerIdx, :, :]^T
out[b, yOffset:yOffset+ySliceSize] = y[...] + scale * Z2[b, :]
```

固定 A2 Kernel 先把同组行压紧到 workspace，以 FP16 输入、FP32 L0C 累加执行 MM1/MM2，再由 `Fixpipe(F322F16)` 降到 FP16；AIV 将 `scale` 转成 half 做原地 `Muls`，最后用 `SetAtomicAdd<half>` 累加。负索引被映射到哨兵组 `W` 并跳过计算。源码未校验非负索引是否小于 `W`。

必须区分公开契约与本 Commit 的真实提交点：`CopyOutRes` 写的是输入 `yGm_`，不是绑定到 `y_out` 的 `youtputGm_`；并且循环长度来自 `H2`，不是 `y_slice_size`。因此复用实现时不能假定独立 `out` 已由 Kernel 填充，也不能把 `y_slice_size<H2` 当作已在 Device 端裁剪。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `y` | `[B,H3]` / ND | FP16 | 必选，非空；`H3<=131072` 且 16 对齐；是公开加数，也是固定 Kernel 的实际原子写目标。 |
| `x` | `[B,H1]` / ND | FP16 | 必选，非空；与 `y` 的 `B` 相同，`H1<=131072` 且 16 对齐。无 `weightA` 时 Kernel 仅将前 `R` 列送入 MM2，Host 未检查 `H1==R`。 |
| `weightB` | `[W,L,H2,R]` / A2 为 ND | FP16 | 必选，非空；`W<=32`、`L<=32`、`H2<=H3`，`H2<=131072` 且 16 对齐；`R` 16 对齐，有 `weightA` 时还要求 `R<=128`。 |
| `indices` | `[B]` / ND | INT32 | 必选，非空；负值由 Device 当作跳过更新的哨兵；固定 Host/公开文档均未给出 `0<=indices<W` 的检查。 |
| `weightAOptional` | `[W,L,R,H1]` / A2 为 ND | FP16 | 可选；存在时前两维匹配 `weightB`，第三维为 `R`，第四维为 `H1`，并令 `addLoraFlag=1`。 |
| `layerIdx` | 标量 | API INT64，OpDef/tiling 最终为 32-bit | 默认 0；转换后必须 `<L`。生成层的 INT64→OpDef attr 转换细节固定仓不可见。 |
| `scale` | 标量 | API double，POD float | 默认 `1e-3`；Device 再转成 half 后做 `Muls`。double→float 的生成层转换固定仓不可见。 |
| `yOffset` | 标量 | API INT64，POD uint32 | 默认 0；必须非负且 `<H3`，并要求 `yOffset+H2<=H3`。 |
| `ySliceSize` | 标量 | API INT64，POD uint32 | 默认 -1，Host 归一化为 `H2`；其它值须非负且 `<=H2`。A2 Device 读取 POD 到成员后不再使用。 |
| `out` | `[B,H3]` / ND | FP16 | 公开必选输出，shape/dtype 与 `y` 相同；固定 A2 Device 为其绑定 `youtputGm_`，但没有后续写入。 |

### 入口与版本

固定 Commit 只公开 `aclnnAddLora` 这一版本族，没有 `V2` 或更新入口。第一段 API 完成参数检查、workspace 计算并返回 executor，第二段以 caller 提供的 workspace 和 stream 执行。`op_host/CMakeLists.txt` 的 `add_modules_sources(OPTYPE add_lora ACLNNTYPE aclnn)` 证明 ACLNN 自动生成接线；生成的 API C++ body/header 不在固定仓，普通非连续 Tensor 连续化及 executor 内部生命周期作为框架边界。

公开产品表明确 A2 支持、A3 与 Ascend 950 不支持；OpDef 注册 `ascend910b`，binary config 与全仓 `ascendc_config.json` 也包含 `ascend910b`。本卡只覆盖 A2/ascend910b 的非 `__CCE_AICORE__==200` 分支，不纳入 310P 的 `200000/200001/200010` 实现。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开 ACLNN 第一段检查必选 `y/x/weightB/indices/out` 非空、dtype/shape/layout；固定生成 body 不可见，文档声明非连续 Tensor 可用。
2. OpDef 把 A2 数据限定为 FP16/INT32、ND，并声明四个默认属性；`InferShape4AddLora` 将输入 0（`y`）的两维复制给输出。
3. `GetShapeAttrsInfo` 取得五个输入、输出和属性；`DoOpTiling` 依次调用 dtype、必选 shape、可选 shape、属性和上限检查，任一步失败返回 `GRAPH_FAILED`。
4. `y/x` 必须二维且 batch 相等，`weightB` 必须四维，`indices` 必须一维且长度为 `B`；存在 `weightA` 时检查 `[W,L,R,H1]` 的维度值。固定 Tiling 未显式检查 `weightA` 的 dim count。
5. Host 将 int64 shape/属性窄化到 uint32；校验 `layerIdx<L`、`yOffset<H3`、`ySliceSize<=H2`、`yOffset+H2<=H3`、`W/L/H1/H2/H3/R` 上限与 16 对齐。
6. `AddLoraTilingFunc` 设置 `BATCH_MODE=1` 后调用基类 `DoTiling`；`PostTiling` 保存 POD、设置 `blockDim=C` 和 key。第二段 API 在给定 stream 上提交唯一 `add_lora` Kernel；生成 executor 的 launch/释放细节固定仓不可见。

### 参数、Workspace 与生命周期

记 `C=GetCoreNumAic()`、`B16=ceil(B/16)*16`。Host 返回的总 workspace 为：

```text
system_workspace = 16 MiB
user_workspace_reserved = 2*B*H1 + 2*B16*R + 2*B16*H2 + 3*4*B16 + 1 MiB
workspace_total = system_workspace + user_workspace_reserved
```

Kernel 通过 `GetUserWorkspace(workspace)` 取得用户区，并实际切片：

```text
xIndice = [0, 2*B*H1)
tmp/MM1 = [2*B*H1, 2*B*H1 + 2*B16*R)
mm2 = [..., + 2*B16*H2)
mOffsetBatch = [..., + 4*B)
countIndiceBatch = [..., + 4*B)
IndicebymOffset = [..., + 4*B)
```

Host 为三个 uint32 数组按 `B16` 而 Device 只按 `B` 推进 offset；额外 1 MiB 是保留余量。caller 持有总 workspace，至少保持到第二段 stream 工作完成。Kernel 若总 workspace 或 `GetUserWorkspace` 为空直接返回且不写结果。

### 启动图与失败传播

```text
aclnnAddLoraGetWorkspaceSize(...)
  -> generated ACLNN/OpDef boundary
  -> AddLoraTilingFunc -> AddLoraTiling::DoTiling/PostTiling
  -> key 100000 or 100001, blockDim=C
aclnnAddLora(workspace, executor, stream)
  -> add_lora(..., workspace, tiling)  [单 Kernel, MIX_AIC_1_2]
     AIV: 计算分组表/可选重排 x -> flag 2
     AIC: 等待 flag 2（需要时）-> MM1 -> MM2 -> workspace mm2 -> flag 8
     AIV: 等待 flag 8 -> scale -> atomic add 到 y
```

这是单 launch 内的 MIX 任务图，不是多个 Kernel。`blockDim=C` 表示 AIC 逻辑核数；task type 固定 1 AIC:2 AIV，AIV 在分组搬运阶段使用物理 `GetBlockIdx()` 覆盖 `2C` 份任务，进入逻辑计算后用 `GetBlockIdx()/2` 与 AIC 配对。未知 key 没有 `else` 错误分支，Kernel 静默结束；workspace 空指针也静默返回。ACLNN launch 返回到错误码的生成层映射固定仓不可见。

## Route 与参数设计

### 分核与资源计算

Host 从平台取 `C=GetCoreNumAic()`，虽然读取 UB 大小，但没有把 UB 值用于 route/resource guard。A2 固定：

```text
maxCore = 2*C
taskNumPerCore = floor(B/(2*C))                 # AIV 重排 x
H2PerCore = floor(ceil(H2/16)/(2*C))*16        # A2 Device 不消费
usedCoreNum = C                                 # blockDim
key = 100001 if B<=C else 100000
```

`100000` 路线按 M 维把 B 行均衡分给 C 个 AIC，`mSplit=128`。`nSplit=floor(16384/(R*256))*256`，最小 128；若 `nSplit*mToProcess>16384`，大 batch 路线降为 128。H1、H2、R 的 K/N 循环均用 `DivCeil` 产生尾块。

`100001` 路线中，MM1 按 batch 行分给前 B 个 AIC；MM2 按 16 元素 N-block 分给 C 个 AIC。设 `NB=ceil(H2/16)`，源码先取 `ceil(NB/C)`，对 `coreId>=NB%C` 再减 1；当 `NB%C==0` 时该式会让每核少一个 16 元素块，属于固定实现边界。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 大 batch 双 MM | `100000` / `AddLoraNormalCoreBatchKernel` | A2（`isAscend310P_=false`）且 `B>C`，`weightAOptional!=nullptr`，`addLoraFlag=1` | `B<=C` 去 R-003；缺省 `weightA` 去 R-002。 |
| 2 | R-002 大 batch 单 MM | `100000` / `AddLoraNormalCoreBatchKernel` | A2 且 `B>C`，`weightAOptional==nullptr`，`addLoraFlag=0` | `B<=C` 去 R-004。 |
| 3 | R-003 小 batch 双 MM | `100001` / `AddLoraSingleCoreBatchKernel` | A2 且 `B<=C`，`weightAOptional!=nullptr`，`addLoraFlag=1` | 缺省 `weightA` 去 R-004。 |
| 4 | R-004 小 batch 单 MM | `100001` / `AddLoraSingleCoreBatchKernel` | A2 且 `B<=C`，`weightAOptional==nullptr`，`addLoraFlag=0` | 参数不满足则 Tiling 失败。 |

```text
参数/shape/dtype/attr 检查失败 -> GRAPH_FAILED
否则：
  socVersionFlag = 1                         # A2/ascend910b
  if B <= C:
    key = 100001
    if weightA exists -> R-003 else -> R-004
  else:
    key = 100000
    if weightA exists -> R-001 else -> R-002
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行族与 kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 大 batch 双 MM | `100000`, `addLoraFlag=1` | A2、全 FP16/ND、`B>C`、存在 WA、已通过 Host 约束 | 单 `add_lora` MIX_CV；AIV 重排，AIC MM1+MM2，AIV epilogue | `AddLoraNormalCoreBatchKernel::{Process,ComputeWeightA,ComputeWeightB,MoveOutMM}` + base helper | x Queue 后 `TPipe::Reset`；L1/L0；tmp/mm2；flag 2/8 | M 每核均分、M tile 128，H1/H2/R `DivCeil`；空组与负索引哨兵跳过。 |
| R-002 大 batch 单 MM | `100000`, `addLoraFlag=0` | 同上但 WA 缺省 | 同一 MIX entry；AIV 重排，AIC 直接 X·WB，AIV epilogue | normal body，跳过 `ComputeWeightA`，`CopyInA2` 从 `xIndice` 读 | tmp slice 不消费；flag 2/8 | R 以 128 分块；Host 未要求 H1=R。 |
| R-003 小 batch 双 MM | `100001`, `addLoraFlag=1` | A2、`B<=C`、存在 WA | 单 MIX entry；AIC 按行 MM1、AIC barrier、按 N MM2、AIV epilogue | `AddLoraSingleCoreBatchKernel::{Process,ComputeWeightAMM,ComputeWeightBMM}` | 不分配 x Queue；tmp/mm2；AIC-only flag 1，AIC→AIV flag 8 | 无 batch 行的 AIC 参加 barrier；N 每核分块公式存在整除边界。 |
| R-004 小 batch 单 MM | `100001`, `addLoraFlag=0` | 同上但 WA 缺省 | AIV 重排，AIC 等待后按 N 做 MM2，AIV epilogue | single body，跳过 MM1 | x Queue 后 Reset；flag 2/8 | R 以 128 分块；N 空核直接返回。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 大/小 batch 分界 | `B<=C` | 小 batch 用 M=1 的 MM1 与按 N 分核；大 batch 用按 M 分核 | 全部 | 选择另一 key | `AddLoraTiling::GetTilingKey` |
| Cube 16 对齐 | `H1%16=H2%16=H3%16=R%16=0` | ND→NZ、LoadData、Mmad 与 Fixpipe 以 16×16 fractal 为基本块 | 全部 | `GRAPH_FAILED` | `CheckInputLimits` |
| R 上限 | 仅 `addLoraFlag && R>128` 失败 | WA 路线 L0 tile 与 MM1 N=R；无 WA 路线未执行该上限检查 | R-001/R-003 | 失败；无 WA 继续 | `CheckInputLimits` |
| N tile | `nSplit=max(128,floor(16384/(R*256))*256)`；大 batch 还检查 `nSplit*mToProcess` | L0C 有效 view 为 16384 个 float | 全部；额外收缩仅 R-001/R-002 | 大 batch 降到 128；小 batch 无同类容量 guard | `Process` 两实现 |
| 大 stride 搬运 | `H1>65535` | `Nd2NzParams.srcDValue` 位宽/stride 边界 | WA 的 MM1 CopyIn | 改逐 16 列 `DataCopyL1Nz` | `CopyIn`、`CopyInA` |
| AIV Queue 容量 | Host 只读取 UB 大小，不验证 `2*H1*2` | `xMoveBuf_` depth=2、每槽 `2*H1` 字节 | R-001/R-002/R-004 | 固定仓无 fallback | `GetPlatformInfo`、`InitDataMoveBuffer` |
| 输出尾段 | `nInner%16` | AIV GM 写需要处理非 16 元素尾块 | 全部 | 用 `DataCopyPad`；当前公开 H2 16 对齐时仅内部分块尾段可能触发 | `CopyOutRes` |
| workspace 指针 | `workspace!=nullptr && GetUserWorkspace(workspace)!=nullptr` | 所有中间结果与索引表依赖用户 workspace | 全部 | Device 静默返回 | `add_lora` |

### POD/ABI

`AddLoraTilingData` 的 15 个声明字段全为 32 位。固定仓缺少 `TILING_DATA_FIELD_DEF` 宏展开和生成的 `kernel_tiling.h`，故不能可靠给出 `sizeof/alignof/offsetof`；`PostTiling` 只证明最终序列化大小必须为 8 字节倍数。下表保持声明顺序，不臆造 padding。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `AddLoraTilingData` | `usedCoreNum` | uint32/32 | 声明序号 0；精确 offset/总 size 固定仓不可见 | AIC core | `usedCoreNum=C` | `PostTiling` 取值设置 `blockDim`；A2 body 不加载成员 | `C>0` |
| 同上 | `layer` | uint32/32 | 序号 1；精确布局不可见 | 层数 | `L_` | `Init`→WA/WB GM offset | `L<=32` |
| 同上 | `batch` | uint32/32 | 序号 2；精确布局不可见 | 行 | `B` | 分核、workspace 地址、循环 | 公开非空；固定 Host 未使用 `MAX_BATCH_SIZE` 检查 |
| 同上 | `H1` | uint32/32 | 序号 3；精确布局不可见 | FP16 元素/行 | `x.shape[1]` | x/WA stride、MM1 K、x Queue | `<=131072`、16 对齐 |
| 同上 | `H2` | uint32/32 | 序号 4；精确布局不可见 | FP16 元素/行 | `weightB.shape[2]` | WB stride、MM2 N、输出长度 | `<=131072`、16 对齐、`<=H3` |
| 同上 | `R` | uint32/32 | 序号 5；精确布局不可见 | FP16 元素 | `weightB.shape[3]` | WA/WB stride、MM1 N、MM2 K、nSplit | 16 对齐；WA 存在时 `<=128` |
| 同上 | `wBatch` | uint32/32 | 序号 6；精确布局不可见 | 权重组 | `weightB.shape[0]` | 分组数组、哨兵 W、权重 offset | `<=32` |
| 同上 | `layer_idx` | uint32/32 | 序号 7；精确布局不可见 | 层索引 | int64 attr 窄化后赋值 | WA/WB layer offset | `<L` |
| 同上 | `scale` | float/32 | 序号 8；精确布局不可见 | 标量 | float attr | `CopyOutRes` 转 half 后 `Muls` | 无范围检查 |
| 同上 | `y_offset` | uint32/32 | 序号 9；精确布局不可见 | FP16 元素 | 非负 int64 attr 窄化 | `CopyOutRes::dstOffset` | `<H3` 且 `+H2<=H3` |
| 同上 | `y_slice_size` | uint32/32 | 序号 10；精确布局不可见 | FP16 元素 | -1→H2，否则 attr | `Init` 加载后无 A2 consumer | `<=H2`，Device 不据此裁剪 |
| 同上 | `taskNumPerCore` | uint32/32 | 序号 11；精确布局不可见 | 行/AIV | `floor(B/(2C))` | `QueryDataMove` | 余数由物理 AIV core id 分配 |
| 同上 | `H2PerCore` | uint32/32 | 序号 12；精确布局不可见 | FP16 元素 | `floor(ceil(H2/16)/(2C))*16` | A2 body 无 consumer | 310P 路线字段；A2 inactive |
| 同上 | `addLoraFlag` | uint32/32 | 序号 13；精确布局不可见 | 布尔 0/1 | `weightAOptional!=nullptr` | 四条 Route 的 MM1/输入/同步分支 | 仅 0/1 |
| 同上 | `y_column` | uint32/32 | 序号 14；精确布局不可见 | FP16 元素/行 | `H3` | y/y_out extent 与 `dstOffset` stride | `<=131072`、16 对齐 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 大 batch 双 MM | `100000`, WA present | `add_lora_tiling.cpp::GetTilingKey/DoOpTiling` | `AddLoraTilingData`, `addLoraFlag=1` | `add_lora.cpp::add_lora`→`AddLoraNormalCoreBatchKernel` | `add_lora_normal_core.h` + `add_lora_base.h` | `op_host/CMakeLists.txt`; ascend910b binary/config |
| R-002 大 batch 单 MM | `100000`, WA null | 同上 | 同一 POD，`addLoraFlag=0` | 同上 | normal body + base `CopyInA2/ComputeMM2` | 同上 |
| R-003 小 batch 双 MM | `100001`, WA present | 同上 | 同一 POD，`addLoraFlag=1` | `add_lora`→`AddLoraSingleCoreBatchKernel` | `add_lora_single_core.h` + base | 同上 |
| R-004 小 batch 单 MM | `100001`, WA null | 同上 | 同一 POD，`addLoraFlag=0` | 同上 | single body + base | 同上 |

## Kernel 实现

### 公共执行流程

`add_lora` 的参数顺序为 `y,x,weightB,indices,weightA,y_out,workspace,tiling`。entry 先验证 workspace，解析 POD，在 A2 编译分支声明 `KERNEL_TYPE_MIX_AIC_1_2`，按 key 构造 normal 或 single class。两个 class 共用 `AddLoraKernelBase` 的 GM 切片、手写 ND/NZ 搬运、L1/L0 初始化、`Mmad`、Fixpipe、AIV scale/AtomicAdd 及同步 helper。

每个 MIX task 都先执行 `ComputeBatchIndice`：负索引映射为 W，生成每行的组内序号 `countIndiceBatch`、分组前缀和 `sumIndiceCount`、压紧行到原行映射 `mOffsetBatch` 及压紧位置所属组 `IndicebymOffset`。AIC/AIV 均调用该函数并向同一 workspace 写相同元数据；固定源码没有在此函数外加跨族同步，后续的 Query/计算协议承担所需可见性。

公共 Cube 数据流是 FP16 GM→L1 ND/NZ→L0A/L0B→FP32 L0C `Mmad`→`Fixpipe(F322F16)`→workspace。公共 AIV epilogue 从 `mm2Gm_` 搬到 UB，half `Muls(scale)`，再以 half AtomicAdd 更新 `yGm_`。`youtputGm_` 只绑定，不消费。

### Route R-001：大 batch 双 MM

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnAddLora.md` | `aclnnAddLoraGetWorkspaceSize` | A2 支持，WA 可选，输入满足公开约束 | 进入 AddLora Host |
| Host | `op_host/add_lora_tiling.cpp` | `GetTilingKey` | `batchSize_>coreNum_`, 非 310P | `100000` |
| Tiling | 同上 | `CheckOptionalInputTensorShape` | `weightATensor!=nullptr` | `addLoraFlag=1` |
| Router | `op_kernel/add_lora.cpp` | `add_lora` | `TILING_KEY_IS(100000)` | `AddLoraNormalCoreBatchKernel::Process` |

#### Device 数据流

##### 函数级调用链

```text
add_lora -> AddLoraNormalCoreBatchKernel::Init
  -> ComputeBatchIndice
  -> AIV: InitDataMoveBuffer -> QueryDataMove -> TPipe::Reset -> InitVectorBuffer
          -> VectorNotifyCube(flag 0 then flag 2)
  -> AIC: InitMatmulBuffer -> CubeWaitVector(flag 2)
          -> per group/per M tile: ComputeWeightA
             -> CopyIn -> SplitA/SplitB -> Compute(Mmad) -> CopyL0C2A1(tmp)
          -> ComputeWeightB
             -> CopyIn2(tmp) -> SplitA -> CopyWbB -> ComputeMM2 -> CopyL0C2GM(mm2)
          -> CubeNotifyVector(flag 8)
  -> AIV: VectorWaitCube(flag 8) -> MoveOutMM -> CopyOutRes -> AtomicAdd(y)
```

##### 源码等价伪码

```text
# 伪码；保持源码循环与地址含义
build_group_metadata(indices)
AIV physical_id in [0,2C):
  copy its balanced rows x[b,0:H1] -> xIndice[group_prefix + local_rank, :]
  all AIV sync; set cross flag 2
AIC logical_id in [0,C):
  wait flag 2
  [m0,m0+mCount) = balanced B/C rows
  for each overlapped valid group w:
    for each M tile mt (<=128):
      choose nSplit; for k0 in 0:H1:nSplit:
        ND xIndice[mt,k0] and WA[w,layer,R,k0] -> L1 -> L0A/L0B
        L0C = init_or_accumulate Mmad(FP16, FP16, FP32)
      Fixpipe F32->F16 L0C -> tmp[mt,R]
      for n0 in 0:H2:nSplit:
        for r0 in 0:R:16:
          tmp tile and WB[w,layer,n0,r0] -> L1/L0
          L0C = init_or_accumulate Mmad
        Fixpipe F32->F16 -> mm2[mt,n0]
  set cross flag 8
AIV paired by logical_id=physical_id/2:
  wait flag 8; split rows by subBlockIdx
  for each row and <=12288-element N tile:
    UB = mm2[row,n]; UB = half(UB * half(scale)); AtomicAdd y[original_row,yOffset+n]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x/y/WA/WB` | 按 POD 行/层/组 stride | 逻辑 tensor 全域 | FP16 ND | caller | AIV/AIC | caller，stream 完成前有效 | H1/H2/H3/R 16 对齐 |
| workspace `xIndice` | 0 | `2*B*H1` bytes | FP16 ND、按组压紧 | `QueryDataMove` | MM1 `CopyIn` | caller workspace | flag 2 后可读 |
| workspace `tmp` | `2*B*H1` | `2*B16*R` bytes | FP16，Fixpipe 默认 fractal/后续块式读取 | MM1 `CopyL0C2A1` | MM2 `CopyIn2` | 当前 launch | FIX→MTE2 event |
| workspace `mm2` | 前两 slice 之后 | `2*B16*H2` bytes | FP16 ND | `CopyL0C2GM` | AIV `CopyOutRes` | 当前 launch | flag 8 后可读 |
| metadata | mm2 后连续三段 | `3*4*B` 实际字节 | uint32 ND | `ComputeBatchIndice` | 分组/回写地址 | 当前 launch | Host 预留按 B16 |
| A1/B1 | 各自 offset 0/32768 bytes 的 ping/pong view | 各 `512KiB/2=256KiB` 分配；每 view 16384 half | FP16 NZ | MTE2 | L0 LoadData | AIC TPipe | 双 view；总 L1 分配 512 KiB |
| L0A/L0B | 每侧 view offset 0/32768 | 各 64 KiB 分配 | FP16 ZZ/ZN | LoadData | Mmad | AIC TPipe | ping/pong |
| L0C | offset 0 | 128 KiB 分配；有效 view 16384 float=64 KiB | FP32 fractal | Mmad | Fixpipe | AIC TPipe | K loop 累加复用 |
| AIV x Queue | 两槽 | `2 * H1 * 2 = 4H1` bytes | FP16 | GM→UB | UB→workspace | AIV；Query 后 Reset | depth 2 |
| AIV `ubBuf_` | 0..192 KiB | 192 KiB | FP16 | mm2 GM | Muls/AtomicAdd | Reset 后重新分配 | `mmResUb[2]` 实际参与；其余 view 已分配但未读 |

##### 分层内存布局

```mermaid
flowchart LR
  X["GM x / weightA"] --> L1A["L1 A1/B1 · 2×256 KiB"]
  L1A --> L0A["L0A 64 KiB + L0B 64 KiB"]
  L0A --> MM1["Mmad MM1 · FP32 L0C"]
  MM1 --> TMP["workspace tmp · FP16"]
  TMP --> L1B["L1 A1; GM weightB→L1 B1"]
  L1B --> MM2["Mmad MM2 · FP32 L0C"]
  MM2 --> WS2["workspace mm2 · FP16 ND"]
  WS2 --> UB["AIV UB · scale"]
  UB --> Y["GM y · FP16 AtomicAdd"]
```

```mermaid
block-beta
  columns 8
  mm0["mmResUb0 · 24 KiB"]:1
  mm1["mmResUb1 · 24 KiB"]:1
  y0["yUb0 · 24 KiB · inactive"]:1
  y1["yUb1 · 24 KiB · inactive"]:1
  yo0["yOffsetUb0 · 48 KiB · inactive"]:2
  yo1["yOffsetUb1 · 48 KiB · inactive"]:2
```

##### 字节账、交接边与复用顺序

```text
workspace_total = 16MiB + 2*B*H1 + 2*B16*R + 2*B16*H2 + 12*B16 + 1MiB
L1_total = 256KiB(A1) + 256KiB(B1) = 512KiB
L0_total = 64KiB(L0A) + 64KiB(L0B) + 128KiB(L0C allocation) = 256KiB
AIV_preprocess_UB = 2 * H1 * 2 = 4H1 bytes
AIV_epilogue_UB = 192KiB
AIV_peak = max(4H1, 192KiB) because TPipe::Reset separates phases
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| x→xIndice | AIV `QueryDataMove` | `groupOffset*H1`, H1 FP16/row | ND | AIC MM1 | AIV all-sync flag 0；AIV→AIC flag 2 | workspace | AIC 必须先 wait 2 |
| L0C→tmp | AIC MM1 | `mProcessOffset*R`，M×R | FP16 fractal block stream | AIC MM2 | `M_FIX`、`FIX_MTE2`、PIPE_FIX | workspace | 同一 AIC 在读前完成 Fixpipe |
| L0C→mm2 | AIC MM2 | `mOffset*H2+nOffset`，M×N | FP16 ND | paired AIV | AIC→AIV flag 8 | workspace | 所有当前逻辑核 Cube tile 完成后通知 |
| mmRes ping/pong | `CopyOutRes` MTE2 | 两个 24 KiB view | FP16 | Muls/MTE3 | MTE2_V、V_MTE3、MTE3_MTE2 | AIV UB | 下一 tile 覆盖前完成 GM write |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| x 重排完成 | mode0 flag 0 + mode2 flag 2 | 2C 个 AIV | AIC `CubeWaitVector(2)` | xIndice workspace | 行按 2C 均分，负索引进哨兵组 |
| L1/L0/Mmad | HardEvent + PipeBarrier | MTE2/MTE1/M | 下一 pipeline | ping/pong L1/L0 | H1、R 用 `DivCeil`；首 K tile init L0C，其后累加 FP32 |
| MM1 写 tmp | `M_FIX`、`FIX_MTE2` | Mmad/Fixpipe | `CopyIn2` | tmp 当前 M×R | F32→F16；R 以 16 分片 |
| MM2→AIV | mode2 flag 8 | AIC | paired AIV wait | mm2 当前逻辑核结果 | F32→F16 后交接 |
| scale/add | MTE2_V、V_MTE3、AtomicAdd | AIV Muls | GM y | 当前行/N tile | scale 先 float→half；nInner 非16倍用 DataCopyPad；原子加为 FP16 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_lora_tiling.cpp` | `DoOpTiling/GetTilingKey/GetWorkspaceSize` | 产生 100000、POD、C block 和 workspace |
| Kernel entry/router | `op_kernel/add_lora.cpp` | `add_lora` / `KERNEL_TYPE_MIX_AIC_1_2` | 100000 映射 normal body |
| Device body/helper | `op_kernel/add_lora_normal_core.h` | `Process/ComputeWeightA/ComputeWeightB/MoveOutMM` | 大 batch 两级 MM 与跨族图 |
| Device body/helper | `op_kernel/add_lora_base.h` | `CopyIn/SplitA/SplitB/Compute/ComputeMM2/CopyOutRes` | 搬运、Mmad、Fixpipe、scale/AtomicAdd |

### Route R-002：大 batch单 MM

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnAddLora.md` | `weightAOptional` | `nullptr` 允许，语义为跳过首个矩阵乘 | Host 将 WA 记为缺省 |
| Host | `op_host/add_lora_tiling.cpp` | `GetTilingKey` | `B>C`, A2 | `100000` |
| Tiling | 同上 | `CheckOptionalInputTensorShape` | `weightATensor==nullptr` | `addLoraFlag=0` |
| Router | `op_kernel/add_lora.cpp` | `add_lora` | key 100000 | normal body 的无 WA 分支 |

#### Device 数据流

##### 函数级调用链

```text
add_lora -> AddLoraNormalCoreBatchKernel::Process
  -> ComputeBatchIndice
  -> AIV QueryDataMove(x -> xIndice) -> notify flag 2
  -> AIC wait flag 2 -> skip ComputeWeightA
       -> ComputeWeightB -> CopyInA2(xIndice) -> SplitA
       -> CopyWbB -> ComputeMM2 -> CopyL0C2GM -> notify flag 8
  -> AIV wait flag 8 -> MoveOutMM -> CopyOutRes -> AtomicAdd(y)
```

##### 源码等价伪码

```text
# 伪码
group x rows into xIndice; AIC waits flag 2
for each AIC M domain, valid weight group, M tile, H2 N tile:
  for r0 in 0:R:128:
    A = xIndice[M, r0:r0+RInner]       # 源行 stride 仍是 H1
    B = weightB[w,layer,N,r0:RInner]
    L0C = init_or_accumulate Mmad(A,B)
  Fixpipe F32->F16 -> mm2[M,N]
notify AIV; AIV half-scale and AtomicAdd to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xIndice workspace | 0 | `2*B*H1` | FP16 ND | AIV | AIC `CopyInA2` | launch | 读取每行前 R 个元素，stride H1 |
| tmp workspace | xIndice 后 | Host 仍保留 `2*B16*R` | 未使用 | N/A | N/A | launch | 不因 WA 缺省缩小 workspace |
| mm2/metadata | 与 R-001 相同 | 同 R-001 | FP16/uint32 | AIC/metadata | AIV/control | launch | flag 8 |
| L1/L0 | 同 R-001 | L1 512 KiB；L0 256 KiB 分配 | FP16/FP32 | Copy/Load/Mmad | Fixpipe | AIC | MM1 阶段不执行但缓冲仍初始化 |
| AIV UB | x Queue 后 Reset，再 192 KiB | peak `max(4H1,192KiB)` | FP16 | x/mm2 | workspace/y | AIV | 两阶段不并存 |

##### 分层内存布局

```mermaid
flowchart LR
  X["GM x"] --> XQ["AIV x Queue"]
  XQ --> XI["workspace xIndice"]
  XI --> L1["L1 A1; weightB→L1 B1"]
  L1 --> L0["L0A/L0B→Mmad→FP32 L0C"]
  L0 --> MM2["workspace mm2 · FP16"]
  MM2 --> UB["AIV UB · scale"]
  UB --> Y["GM y · AtomicAdd"]
```

##### 字节账、交接边与复用顺序

```text
workspace/L1/L0/AIV allocations = R-001
tmp_reserved = 2*B16*R bytes but inactive
R_split = 128; MM2 K = R; source x row stride = H1
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| xIndice | AIV QueryDataMove | B×H1 | FP16 ND | AIC MM2 | flag 2 | workspace | 语义正确复用要求 x 的前 R 列就是 MM2 输入；Host 未强制 H1=R |
| mm2 | AIC | B16×H2 reserved | FP16 ND | AIV | flag 8 | workspace | 每组/每 M tile写完后全路线通知 |
| UB ping/pong | AIV | 2×24 KiB active views | FP16 | AtomicAdd | pipeline events | AIV | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 重排 | flag 0/2 | AIV | AIC | xIndice | batch tail 按 2C 分配 |
| 单 MM | MTE/Load/Mmad events | AIC | Fixpipe | L1/L0 | R 按128、H2按 nSplit；FP32 累加→FP16 |
| epilogue | flag 8 + AtomicAdd | AIC/AIV | y GM | mm2/y | half scale；H2 内部尾 tile保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_lora_tiling.cpp` | `CheckOptionalInputTensorShape/GetTilingKey` | WA null→flag0，B>C→100000 |
| Kernel entry/router | `op_kernel/add_lora.cpp` | `add_lora` | normal 类 |
| Device body/helper | `op_kernel/add_lora_normal_core.h` | `Process/ComputeWeightB` | 跳过 MM1，R split 改128 |
| Device body/helper | `op_kernel/add_lora_base.h` | `CopyInA2/CopyWbB/ComputeMM2` | 从 xIndice 直接形成 MM2 A operand |

### Route R-003：小 batch双 MM

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnAddLora.md` | `weightAOptional` | 非空并满足 `[W,L,R,H1]` | 双 MM |
| Host | `op_host/add_lora_tiling.cpp` | `GetTilingKey` | `B<=C`, A2 | `100001` |
| Tiling | 同上 | `CheckOptionalInputTensorShape` | WA present | `addLoraFlag=1` |
| Router | `op_kernel/add_lora.cpp` | `add_lora` | key 100001 | `AddLoraSingleCoreBatchKernel` |

#### Device 数据流

##### 函数级调用链

```text
add_lora -> AddLoraSingleCoreBatchKernel::Process
  -> ComputeBatchIndice
  -> AIV: InitVectorBuffer; coreId=GetBlockIdx()/2
  -> AIC: InitMatmulBuffer -> ComputeWeightAMM
       -> CopyInA(x direct) -> SplitA/SplitB -> Compute -> CopyL0CGM(tmp)
       -> SyncAicOnly(flag 1)
  -> all AIC: partition H2 by N blocks -> ComputeWeightBMM
       -> CopyInA2(tmp) -> SplitA -> CopyWbB -> ComputeMM2 -> CopyL0C2GM
       -> per nonempty weight group CubeNotifyVector(flag 8)
  -> paired AIV: per group wait flag 8 -> CopyOutRes -> AtomicAdd(y)
```

##### 源码等价伪码

```text
# 伪码
each AIC core c<B computes row c MM1 directly from x[c] and WA[indices[c]]
negative or sentinel-W row skips MM1
all C AIC synchronize on flag 1 so every tmp row is visible
partition H2 16-element blocks over C AIC cores
for each valid weight group w:
  each AIC computes all group rows for its N slice, R split by16
  write mm2[group_rows, nOffset:nOffset+nIncore]
  signal paired AIV with flag 8
  two AIV subblocks divide group rows, wait, scale, AtomicAdd to original y rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xIndice | workspace offset 0 | Host 保留 `2*B*H1` | inactive | N/A | N/A | launch | WA-present small route 不重排 x |
| tmp | `2*B*H1` | `2*B16*R` | FP16 ND via NZ2ND Fixpipe | per-row AIC MM1 | all AIC MM2 | workspace | AIC-only barrier flag1 |
| mm2 | tmp 后 | `2*B16*H2` | FP16 ND | N-owner AIC | paired AIV | workspace | 每 weight group flag8 |
| L1/L0 | 与公共配置相同 | L1 512 KiB，L0 256 KiB 分配 | FP16/FP32 | AIC | AIC/Fixpipe | TPipe | MM1/2 复用同一 buffer |
| AIV UB | 192 KiB | 192 KiB | FP16 | mm2 | Muls/AtomicAdd | AIV | 无 x Queue 阶段 |

##### 分层内存布局

```mermaid
flowchart LR
  X["GM x + weightA"] --> AIC1["per-row AIC MM1"]
  AIC1 --> TMP["workspace tmp · FP16 ND"]
  TMP --> BAR["AIC-only flag 1"]
  BAR --> AIC2["N-sharded AIC MM2"]
  AIC2 --> MM2["workspace mm2"]
  MM2 --> FLAG["AIC→AIV flag 8 / group"]
  FLAG --> UB["paired AIV UB"]
  UB --> Y["GM y AtomicAdd"]
```

##### 字节账、交接边与复用顺序

```text
workspace_total = 公共 Host 公式
L1_total = 512KiB; L0_total allocation = 256KiB
AIV_UB = 192KiB
x_queue = 0 in this route
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| per-row tmp | first B AIC | grouped row×R | FP16 ND | every N-owner AIC | `SyncAicOnly<PIPE_FIX>(flag1)` | workspace | 空/负索引行不被有效组消费 |
| per-group mm2 N slice | each AIC | group rows × `[nOffset,nOffset+nIncore)` | FP16 ND | two paired AIV | flag8 per weight group | workspace | 相同 flag 按 group 顺序 set/wait；不同组占不同 M 行 |
| L1/L0 reuse | MM1 then MM2 | fixed buffers | FP16/FP32 | next phase | HardEvent + AIC barrier | TPipe | barrier 后 MM2 才跨核读 tmp |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MM1 全局完成 | mode0 AIC-only flag1 | 每个 AIC，包括 m=0/无效组核 | 所有 AIC | tmp 全部有效组行 | MM1 FP32 累加、F32→F16 |
| MM2 分 N | 16-element block partition | C AIC | Fixpipe | H2 slices | `nIncore<=0` 的核直接 return；整除 NB%C==0 有少一块边界 |
| group 交接 | mode2 flag8 | 每个 N-owner AIC | 两个 paired AIV | 当前组的 N slice | R 以16累加；F32→F16 |
| scale/add | AIV events + AtomicAdd | AIV | y | 原始 row/N slice | half scale，FP16 atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_lora_tiling.cpp` | `GetTilingKey` | B<=C→100001 |
| Kernel entry/router | `op_kernel/add_lora.cpp` | `add_lora` | single 类 |
| Device body/helper | `op_kernel/add_lora_single_core.h` | `Process/ComputeWeightAMM/ComputeWeightBMM` | MM1 按 M、barrier、MM2 按 N |
| Device body/helper | `op_kernel/add_lora_base.h` | 同步 helper、MM/CopyOut helper | flag1/8、workspace 与 epilogue |

### Route R-004：小 batch单 MM

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnAddLora.md` | `weightAOptional` | null | 跳过 MM1 |
| Host | `op_host/add_lora_tiling.cpp` | `GetTilingKey` | `B<=C`, A2 | `100001` |
| Tiling | 同上 | `CheckOptionalInputTensorShape` | WA null | `addLoraFlag=0` |
| Router | `op_kernel/add_lora.cpp` | `add_lora` | 100001 | single body 无 WA 分支 |

#### Device 数据流

##### 函数级调用链

```text
add_lora -> AddLoraSingleCoreBatchKernel::Process
  -> ComputeBatchIndice
  -> AIV InitDataMoveBuffer -> QueryDataMove -> Reset -> InitVectorBuffer -> notify flag2
  -> AIC InitMatmulBuffer -> wait flag2
  -> partition H2 by N -> ComputeWeightBMM
       -> CopyInA2(xIndice) -> SplitA -> CopyWbB -> ComputeMM2 -> mm2 -> flag8
  -> AIV wait flag8 -> CopyOutRes -> AtomicAdd(y)
```

##### 源码等价伪码

```text
# 伪码
AIV reorder x rows by weight group, then flag2
each AIC waits and owns one N slice
for each valid group and R chunk of 128:
  A = grouped x[:,r:r+RInner] with row stride H1
  B = weightB[group,layer,Nslice,r:r+RInner]
  accumulate FP32 Mmad; Fixpipe to mm2
flag8; paired AIV split M rows and AtomicAdd half(scale*mm2) to y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| xIndice | 0 | `2*B*H1` | FP16 ND | AIV | all AIC N owners | workspace | flag2 |
| tmp | xIndice 后 | `2*B16*R` reserved | inactive | N/A | N/A | workspace | 不缩小 |
| mm2/metadata | 同公共切片 | `2*B16*H2` + arrays | FP16/uint32 | AIC/metadata | AIV/control | workspace | flag8 |
| L1/L0 | 公共固定分配 | L1 512 KiB、L0 256 KiB | FP16/FP32 | AIC | Fixpipe | TPipe | MM1不执行 |
| AIV UB | 先 `4H1`，Reset 后192 KiB | peak max | FP16 | x/mm2 | workspace/y | AIV | phase reuse |

##### 分层内存布局

```mermaid
flowchart LR
  X["GM x"] --> Q["AIV Queue depth2"]
  Q --> XI["workspace xIndice"]
  XI --> FLAG2["flag2"]
  FLAG2 --> AIC["N-sharded AIC MM2"]
  AIC --> MM2["workspace mm2"]
  MM2 --> FLAG8["flag8"]
  FLAG8 --> AIV["AIV scale"]
  AIV --> Y["GM y AtomicAdd"]
```

##### 字节账、交接边与复用顺序

```text
workspace_total = 公共 Host 公式
AIV_peak = max(4H1,192KiB)
L1/L0 = 公共固定分配
tmp slice inactive but reserved
```

| 边/复用子区 | producer | offset/extent | dtype/layout | consumer | ordering/sync | owner/lifetime | 安全条件 |
|---|---|---|---|---|---|---|---|
| xIndice | 2C AIV | B×H1 | FP16 ND | C AIC | flag2 | workspace | x 前 R 列作为 MM2 K 域 |
| mm2 group/N | each AIC | group M×N slice | FP16 ND | paired AIV | reused flag8 per group | workspace | wait 后才读；组间 M 地址不重叠 |
| x Queue→UB reuse | AIV | `4H1` then192 KiB | FP16 | preprocess/epilogue | `TPipe::Reset` | AIV | 两阶段严格分隔 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| x ready | flag0/2 | AIV | AIC | xIndice | 2C task tail |
| MM2 | HardEvent/PipeBarrier | AIC | Fixpipe | L1/L0 | R split128；FP32→FP16 |
| group ready | flag8 | AIC | paired AIV | mm2 | N 分核整除边界同 R-003 |
| output | MTE/V events + AtomicAdd | AIV | y | row/N | half scale；非16尾 DataCopyPad |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/add_lora_tiling.cpp` | `GetTilingKey/DoOpTiling` | 100001/flag0 |
| Kernel entry/router | `op_kernel/add_lora.cpp` | `add_lora` | single body |
| Device body/helper | `op_kernel/add_lora_single_core.h` | `Process/ComputeWeightBMM` | AIV预处理→按N MM2→AIV后处理 |
| Device body/helper | `op_kernel/add_lora_base.h` | `QueryDataMove/CopyInA2/CopyOutRes` | workspace交接与原子加 |

## 算子特定优化

### Trick T-001：按 batch 规模切换 M 分核与 N 分核

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：`batchSize_<=coreNum_` 选 single，否则选 normal；A2 soc offset 固定 100000。
- Golden 来源：[`math/add_lora/op_host/add_lora_tiling.cpp:355-372`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_host/add_lora_tiling.cpp#L355-L372)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t AddLoraTiling::GetTilingKey() const
{
    uint64_t tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_DEFAULT_SCENE);
    if (batchSize_ <= static_cast<uint32_t>(coreNum_)) {
        tilingKey = static_cast<uint64_t>(TilingKeyInfo::KEY_SPARSE_SCENE);
    }

    uint32_t socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_910);
    if (isAscend310P_) {
        socVersionFlag = static_cast<uint32_t>(SocVersionKey::KEY_SOC_VERSION_310);
```

#### 机制与设计原因

大 batch 时 `normal` 让每个 AIC 持有连续 M 行并在本核完成两级 MM，减少小 M 下的空核；`B<=C` 时 `single` 把 MM1 按单行分发，再用 AIC-only barrier 将 tmp 交给按 H2/N 分核的 MM2，使更多 AIC 能参与宽输出。key 只区分 batch 分核骨架，`addLoraFlag` 在同一 body 内选择是否执行 MM1。

#### 迁移边界

- 适用：M 很小但 N 较宽、且中间 M×R 可落 workspace 的双 GEMM/可选首 GEMM。
- 前提：A2 MIX 1:2、FP16 输入、16 对齐、相同分组 metadata 与 flag 协议。
- 失效：平台不是 ascend910b、shape 检查失败，或 small route 的 N 分块/资源边界不满足时必须重新设计 selector，不能直接沿用 key。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要只切 key 而复用同一分核公式：normal 的 MM2 按 M 域，single 的 MM2 按 N 域，并额外依赖 AIC-only barrier。固定仓未提供可验证的性能反例。

### Trick T-002：MIX_CV workspace 交接与 flag 协议

#### 触发与源码依据

- 适用 Route：R-001～R-004；R-003 额外使用 AIC-only flag1。
- 触发条件：entry 选择 `KERNEL_TYPE_MIX_AIC_1_2`；AIV 与 AIC 共享 POD/workspace。
- Golden 来源：[`math/add_lora/op_kernel/add_lora_normal_core.h:88-153`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_kernel/add_lora_normal_core.h#L88-L153)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    if ASCEND_IS_AIV {
        InitDataMoveBuffer();
        QueryDataMove();
        pipe_->Reset();
        InitVectorBuffer();
        VectorNotifyCube<PIPE_MTE3>(SYNC_AIV_ONLY_ALL_FLAG, SYNC_AIV_AIC_FLAG);
    }
    if ASCEND_IS_AIC {
        InitMatmulBuffer();
        CubeWaitVector(SYNC_AIV_AIC_FLAG);
    }
```

#### 机制与设计原因

AIV 先完成不规则按组重排并通过 mode0 全 AIV flag0 收口，再用 mode2 flag2 唤醒 AIC；AIC 的两级 MM 写入 tmp/mm2 后以 PIPE_FIX 上的 flag8 唤醒配对 AIV。`TPipe::Reset` 让 x 搬运 Queue 与 192 KiB epilogue UB 分时占用。该设计把分组搬运、Cube 计算与 Vector scale/add 放在单 launch 中，但仍通过 GM workspace 明确交接。

#### 迁移边界

- 适用：1 AIC:2 AIV、AIV 负责 gather/epilogue、AIC 负责 GEMM 的融合任务。
- 前提：flag 0/2/8 的参与核集合、PIPE、set/wait 顺序和 `coreId/2` 配对完全一致。
- 失效：改变 AIC:AIV 比例、拆成多 Kernel 或让 workspace slice 重叠时必须重建同步与生命周期。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把 Host/源码书写顺序当作跨族可见性；删除 flag2 会让 AIC 提前读取 xIndice，删除 flag8 会让 AIV 提前读取 mm2。

### Trick T-003：FP32 Cube 累加后直接 half scale 原子加到 y

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：每个 MM2 tile 已由 Fixpipe 写成 FP16 `mm2Gm_`，AIV 获得 flag8。
- Golden 来源：[`math/add_lora/op_kernel/add_lora_base.h:643-659`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/add_lora/op_kernel/add_lora_base.h#L643-L659)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
            DataCopy(mmResUb[pingPongFlag], mm2Gm_[resOffset], CeilCubeBlock(nInner) * CUBE_BLOCK);
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            Muls(
                mmResUb[pingPongFlag], mmResUb[pingPongFlag], static_cast<half>(scale),
                CeilCubeBlock(nInner) * CUBE_BLOCK);
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);
            SetAtomicAdd<half>();
```

#### 机制与设计原因

MMAD 在 FP32 L0C 累加，Fixpipe 一次性转 FP16；AIV 在同一 UB view 原地乘 half scale，并通过 FP16 AtomicAdd 直接合入 y，避免单独搬入 y 和物化一个 `y+delta` 中间张量。代价是 scale 在 half 精度生效，且真实目标是输入 y 而非公开 out。

#### 迁移边界

- 适用：输出语义确实允许原地更新、并接受 FP16 scale/atomic 舍入的 epilogue。
- 前提：目标 GM 支持 half AtomicAdd，N tile 不重叠或允许原子冲突，y 生命周期覆盖 stream。
- 失效：独立 out、需要 FP32 scale/加法、或不允许修改输入时必须改为显式读 y、计算并写 out。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

直接把 `yGm_` 改名为 `y_out` 不足以保持公开语义：还必须复制未更新区、处理 `ySliceSize`，并确认 out/y 是否别名。固定仓的公开样例为 out 分配独立地址，构成需要显式处理的边界。

## 模板与实现对比

| 对比维度 | R-001 大 batch双 MM | R-002 大 batch单 MM | R-003 小 batch双 MM | R-004 小 batch单 MM | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | `B>C`, WA present | `B>C`, WA null | `B<=C`, WA present | `B<=C`, WA null | batch 决定 key；WA presence 决定运行子图 |
| 执行族/节点图 | AIV gather→AIC MM1/MM2→AIV epilogue | AIV gather→AIC MM2→AIV | AIC MM1→AIC barrier→AIC MM2→AIV | AIV gather→AIC MM2→AIV | 全部单 Kernel MIX 1:2 |
| M/N/K、tile 与 scheduler | M 分核，M tile128，K/N nSplit | 同左，R split128 | MM1 M=1；MM2 N分核，R split16 | N分核，R split128 | 小 M 用 N 并行，大 M 用 M 并行 |
| workspace/L1/L0/UB | xIndice/tmp/mm2；peak max(4H1,192KiB) | tmp inactive | xIndice inactive；AIV 192KiB | tmp inactive；peak max | Host workspace 不按子图缩小；AIC buffer固定 |
| AIC/AIV 或 kernel 交接 | flag2/8 | flag2/8 | AIC flag1 + flag8 | flag2/8 | WA-present single 需要 tmp 跨 AIC 可见 |
| 同步、精度与 tail | FP32 MM→FP16→half scale/atomic；M/H1/H2 tail | 同左，R128 | 同精度；N-block 整除边界 | 同左 | 同一 epilogue，分核 tail 不同 |
| fallback/失败边界 | key mismatch静默；参数失败 | 另有 H1/R 语义边界 | N整除边界 | N整除+H1/R边界 | 固定仓无额外安全 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；本卡覆盖 ascend910b | 公开文档明确不支持 | 公开文档明确 Ascend 950PR/950DT 不支持 |
| Host/Launcher | `socVersionFlag=1`；key 100000/100001；blockDim=AIC core count | 无公开 Host/config | 无公开 Host/config |
| Kernel/资源 | `MIX_AIC_1_2`，normal/single 两 body | 不发布 | 不发布 `_950` 卡 |

## 限制与失败边界

- 固定 A2 Device 的 `CopyOutRes` 只写输入 `yGm_`，从未写绑定公开输出的 `youtputGm_`；公开样例为 y/out 分配独立地址。固定源码未给出 alias/copy 补偿，因此独立 out 的最终值无法由本仓 Kernel 证明。
- `y_slice_size` 被 Host 校验并序列化，A2 `Init` 也读取，但输出循环固定覆盖 `H2`；`ySliceSize<H2` 的公开语义与 Device 行为不闭合。
- 无 WA 时 Host 不检查 `H1==R`，Device MM2 只消费每行前 R 个 x 元素、行 stride 仍为 H1；调用方需避免把任意 H1/R 组合误认为已受校验保护。
- `indices<0` 被当作跳过组，非负值没有 `<W` 检查；越界索引会访问固定大小计数数组或权重 offset，固定仓无 fallback。
- `MAX_BATCH_SIZE=65536` 已声明但 `CheckInputLimits` 未使用；shape/offset 多处窄化为 uint32，极大 B 的乘法、workspace offset 和循环安全性固定仓未证明。
- Host 读取平台 UB 大小但不据此限制 `xMoveBuf_`；有 gather 的路线分配 `4H1` 字节双缓冲，`H1` 上限 131072 时可达 512 KiB，固定仓没有资源 fallback。
- small route 的 N 分核公式在 `ceil(H2/16)%C==0` 时对所有核执行 `nBlockNum-1`，源码无法证明完整覆盖 H2；小 route 也没有 normal route 的 `nSplit*m` L0C 容量收缩 guard。
- `weightA` 可选 shape 检查读取四个维度但未显式验证 dim count；`scale` 不做有限性/范围检查。unknown key、空 workspace 均在 Device 静默返回。
- ACLNN 生成 body、非连续 Tensor 的 Contiguous/ViewCopy、executor launch 错误传播、Tiling 宏展开后的精确 ABI offset/size，以及 AscendC primitive 的库内实现均为固定仓不可见边界。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/add_lora/docs/aclnnAddLora.md` | `aclnnAddLoraGetWorkspaceSize/aclnnAddLora` | API、产品、参数、公式和错误码 |
| 入口实现 | `math/add_lora/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn)` | 自动 ACLNN 生成接线边界 |
| Host 注册 | `math/add_lora/op_host/add_lora_def.cpp` | `AddLora::AddLora` | 输入/输出/属性及 ascend910b 注册 |
| shape 推导 | `math/add_lora/op_host/add_lora_infershape.cpp` | `InferShape4AddLora` | 输出复制 y shape |
| POD 声明 | `math/add_lora/op_host/add_lora_tiling.h` | `AddLoraTilingData` | 字段顺序、常量和 Host 状态 |
| Host/Tiling/Launcher | `math/add_lora/op_host/add_lora_tiling.cpp` | `AddLoraTiling::*` | 检查、key、workspace、blockDim、POD producer |
| Kernel entry/body | `math/add_lora/op_kernel/add_lora.cpp` | `add_lora` | A2 key router 与 MIX 1:2 task type |
| Kernel entry/body | `math/add_lora/op_kernel/add_lora_base.h` | `AddLoraKernelBase::*` | workspace、L1/L0/UB、Mmad、同步、AtomicAdd |
| Kernel entry/body | `math/add_lora/op_kernel/add_lora_normal_core.h` | `AddLoraNormalCoreBatchKernel::*` | 100000 大 batch body |
| Kernel entry/body | `math/add_lora/op_kernel/add_lora_single_core.h` | `AddLoraSingleCoreBatchKernel::*` | 100001 小 batch body |
| 构建接线 | `math/add_lora/op_host/config/ascend910b/add_lora_binary.json` | `AddLora` binary item | ascend910b 输入/属性/输出 binary 配置 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `AddLora` | compute_units 为 ascend310p/ascend910b、auto_sync |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 15 个32位字段的声明顺序与 producer/consumer | 不臆造宏展开 offset；`y_slice_size/H2PerCore` 在 A2 inactive | `AddLoraTilingData`、`Init` |
| 调用顺序 | metadata→可选 gather/MM1→MM2→AIV epilogue | normal/single 与 WA presence 的阶段不能互换 | 两个 `Process` |
| workspace/内存 | xIndice/tmp/mm2/三数组切片及 L1/L0/UB固定分配 | offset、B16容量、TPipe Reset 生命周期 | `GetWorkspaceSize/GlobalTensorInit/Init*Buffer` |
| 同步 | flag0/2/8，small+WA 的 flag1，pipeline HardEvent | AIC:AIV 1:2、coreId/2、PIPE 和每组 flag 轮次 | sync helper、两个 `Process` |
| 精度与尾块 | FP16输入→FP32 Mmad→F322F16→half scale→FP16 atomic | cast/scale/add顺序、16 fractal、DivCeil tail | `Compute/ComputeMM2/CopyL0C2GM/CopyOutRes` |
| 失败边界 | Host检查失败、空 workspace、unknown key、空核/空组 | 不把静默返回补成成功输出；保留 y_out/ySlice/N分核冲突 | `DoOpTiling/add_lora/Process/CopyOutRes` |
