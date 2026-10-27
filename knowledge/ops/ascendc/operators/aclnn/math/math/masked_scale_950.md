---
type: Operator
title: "MaskedScale 950 ACLNN 算子知识"
description: "说明 aclnnMaskedScale 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/docs/aclnnMaskedScale.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_api/aclnn_masked_scale.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_api/masked_scale.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_host/arch35/masked_scale_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_kernel/arch35/masked_scale_tiling_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_kernel/arch35/masked_scale_dag.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_kernel/masked_scale_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_host/masked_scale_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_host/config/ascend950/masked_scale_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/CMakeLists.txt
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_sch.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/dag.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/vec.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/util/sync.h
    role: implementation
  - resource: https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/src/op_common/atvoss/elewise/elewise_tiling.cpp
    role: implementation
created_at: "2026-08-25T11:33:42Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MaskedScale 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMaskedScaleGetWorkspaceSize` + `aclnnMaskedScale`，公开 ACLNN 两段式接口；固定仓未定义其它版本。 |
| 计算语义 | 对同 shape 的 `self`、`mask` 逐元素计算 `y[i] = self[i] * float(mask[i]) * scale`；数学阶段统一在 FP32 完成。 |
| 输入输出 | `self/y`：同 shape、同 dtype，`FLOAT16/BFLOAT16/FLOAT`；`mask`：同 shape，`UINT8/INT8/FLOAT16/FLOAT`；ND，公开 rank 上限 8。 |
| 执行类型 | 非空核心节点为单 `AIV_VECTOR` Kernel；ACLNN 还可排入框架 `Contiguous/ViewCopy` 节点。所有核心 key 都由 `KERNEL_TYPE_AIV_ONLY` entry 执行，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 低精度 x + 8-bit mask；R-002 FP32 x + 8-bit mask；R-003 低精度 x + FP16 mask；R-004 FP32 x + FP16 mask；R-005 低精度 x + FP32 mask；R-006 FP32 x + FP32 mask。 |
| 核心资源 | 输入/输出 GM、ATVOSS LEVEL_2 单 `TBuf` tensor pool、编译期 buffer IDs、PIPE mutex；每个核心节点声明 16 MiB system workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 launcher→950 Host/Tiling/POD→12-key AIV entry→两个 DAG→ATVOSS Tiling/调度/Vector primitive→binary/OpDef/CMake 已闭合；`Contiguous/ViewCopy` 与 executor 是显式框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `N=product(self.shape)`。对 `0<=i<N`，先把 `self[i]` 与 `mask[i]` 转为 FP32，执行 `p=float(self[i])*float(mask[i])`、`q=p*scale`，再以 `RoundMode=1` 的 Cast 写回 `self` dtype。`UINT8/INT8 mask` 先精确转为 FP16，再转 FP32；其值域均可由 FP16 精确表示。`FLOAT16 mask` 直接转 FP32，`FLOAT mask` 的同类型 Cast 被 DAG 的 CastNoOp 规则消除。`self=FLOAT` 时输入和输出同类型 Cast同样消除。输出只覆盖前 `N` 个逻辑元素；空 Tensor 在 ACLNN Host 直接成功，不启动 Device。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND；与 `mask/y` shape 完全一致；rank≤8；允许非连续 | `FLOAT16/BFLOAT16/FLOAT` | 被乘数；非连续时先由 executor 排入 `Contiguous`。 |
| `mask` | ND；与 `self/y` shape 完全一致 | `UINT8/INT8/FLOAT16/FLOAT` | 逐元素乘数。公开文档表未承诺非连续 mask；实现仍调用 `Contiguous(mask)`，工程复用不得据此扩大公开契约。 |
| `scale` | Host 标量 | `float` | 必填，无默认值；原样写入 POD，Device `Muls<float>` 每 tile 读取同一标量。 |
| `y` | ND；shape、dtype、storage format 与 `self` 一致；允许非连续 | `FLOAT16/BFLOAT16/FLOAT` | 临时连续结果经 `ViewCopy` 提交到用户 Tensor。 |
| `workspaceSize/executor` | Host 输出 | `uint64_t` / `aclOpExecutor**` | 空域为0；否则由 executor 汇总框架临时 Tensor 与核心节点 workspace。 |
| `workspace/stream` | Device byte buffer / runtime stream | N/A | 第二段交给 `CommonOpExecutorRun`；workspace 地址、大小与第一段 executor 必须配套。 |

### 入口与版本

固定仓只公开 `aclnnMaskedScale` 两段式版本。README/API 文档明确 Ascend 950PR/950DT 支持，而 A2、A3 均为“不支持”；OpDef 只注册 `ascend950`，binary config 只提供 `ascend950` 的12个 dtype组合。因此本卡只覆盖 950，不能由顶层 CMake 中遗留的 `arch32` 映射推导 A2/A3 公开支持。950 L2 dtype 检查仅在 `IsRegBase()` 时返回支持表，非 RegBase 平台返回空表。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段依次创建 executor，执行 `CheckNotNull`、`self.dtype==y.dtype`、950 dtype支持表、`self/mask/y` shape相等、`self/mask rank<=8`、`self.storageFormat==y.storageFormat`。`self->IsEmpty()` 时写 `workspaceSize=0` 并返回。非空时依次建立 `Contiguous(self)`、`Contiguous(mask)`，调用 `l0op::MaskedScale` 分配连续临时输出并以 `ADD_TO_LAUNCHER_LIST_AICORE` 加入核心节点，最后 `ViewCopy(result,y)`。任一步返回空指针或 launcher 添加失败即终止第一段，后续节点不再建立。

### 参数、Workspace 与生命周期

`selfContiguous`、`maskContiguous`、核心输出均由 `uniqueExecutor` 持有，直到第二段相应 consumer 完成。核心 Tiling 固定写 `workspace[0]=16*1024*1024` bytes；entry 要求 `workspace!=nullptr`，调用 `SetSysWorkspace` 后还要求 `GetUserWorkspace(workspace)!=nullptr`，但数学 DAG 不读取 `userWS`。ACLNN 返回的总 workspace 由 `GetWorkspaceSize()` 汇总，固定算子源码未公开框架临时 Tensor、核心 system workspace 的最终 offset 和复用算法，不能把 16 MiB等同于所有输入形态的 ACLNN 总量。

### 启动图与失败传播

```text
self -> [Contiguous framework node] --\
                                      MaskedScale AIV -> [ViewCopy framework node] -> y
mask -> [Contiguous framework node] --/
```

| 节点/边 | Host 添加函数 | entry/grid/POD | 介质与 extent | ordering、owner/lifetime、失败 |
|---|---|---|---|---|
| N0/N1 | `l0op::Contiguous` | 框架边界 | 连续 GM Tensor，各 `N*sizeof(dtype)` | executor 持有；任一空指针阻断 N2。连续输入可由框架消除实际 copy。 |
| N2 | `l0op::MaskedScale` | `masked_scale`; `blockDim=baseTiling.blockNum`; `MaskedScaleTilingData` | 两个连续输入→连续临时 y；核心 workspace声明16 MiB | AIV_ONLY；未知 key/null workspace直接无输出；launcher添加失败阻断 N3。 |
| N3 | `l0op::ViewCopy` | 框架边界 | N2 临时 Tensor→用户 y，`N*sizeof(Tx)` | 与 N2 同 executor/stream 依赖顺序；失败使第一段返回内部空指针错误。 |

固定算子源码未显式暴露 framework node 的 event，但 producer/consumer Tensor依赖由同一 executor 建图，第二段用同一 `stream` 执行。N0/N1/N2 的结果在最后 consumer 完成前不可覆盖或释放。

## Route 与参数设计

### 分核与资源计算

令 `U` 为 950 每核 UB 字节，`P` 为 AIV 核数，`bmin=8*MinDtypeBytes`，`B=OpDag::BufferNum`。公共 Tiling 合轴为 `N`，计算：

```text
C = min(P, ceil(N*bmin/32768))
blockFormer = align_up(ceil(N/C), 512)                 # 元素
blockNum = ceil(N/blockFormer)
blockTail = N-(blockNum-1)*blockFormer
alignFactor = 2048/bmin                                # 256 B repeat对应的元素数
ubFormer = align_down(floor(U/(B*4)), alignFactor)     # MaxDtypeBytes=4
formerLoops = ceil(blockFormer/ubFormer)
tailLoops = ceil(blockTail/ubFormer)
```

所有路线 `MaxDtypeBytes=4`。LEVEL_2 buffer planner为两个输入各分配 ping/pong MTE2 区、一个输出分配 ping/pong MTE3 区，再按峰值存活的 FP32中间量分配 temp区：R-001/R-003 同时存活两路 cast结果，`B=8`；R-002/R-004/R-005/R-006 峰值只有一个 temp，`B=7`。因此 UB 总量分别为 `32*ubFormer` 或 `28*ubFormer` bytes。`ubFormer==0`、平台 core数为0、shape含0或同 shape检查失败均使 Tiling失败；ACLNN空域提前返回避免零维到达 Tiling。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 低精度 x + 8-bit mask | 11/12/21/22，`MaskCastTwo<Tx,Tmask>` | `x∈{F16,BF16}` 且 `mask∈{U8,I8}`；key为 x码1/2 + mask码10/20 | R-002或后续 dtype分支 |
| 2 | R-002 FP32 x + 8-bit mask | 13/23，`MaskCastTwo<float,Tmask>` | `x=F32` 且 `mask∈{U8,I8}` | R-003 |
| 3 | R-003 低精度 x + FP16 mask | 31/32，`MaskCastOne<Tx,half>` | `x∈{F16,BF16}` 且 `mask=F16` | R-004 |
| 4 | R-004 FP32 x + FP16 mask | 33，`MaskCastOne<float,half>` | `x=F32, mask=F16` | R-005 |
| 5 | R-005 低精度 x + FP32 mask | 41/42，`MaskCastOne<Tx,float>` | `x∈{F16,BF16}, mask=F32` | R-006 |
| 6 | R-006 FP32 x + FP32 mask | 43，`MaskCastOne<float,float>` | `x=F32, mask=F32` | dtype非法→失败 |

```text
if x not in {F16,BF16,F32}: fail
if mask not in {U8,I8,F16,F32}: fail
key = {F16:1,BF16:2,F32:3}[x] + {U8:10,I8:20,F16:30,F32:40}[mask]
DoTiling(key-specific DAG); if failed: fail
set scale, key, blockDim=blockNum, workspace=16MiB
entry dispatches the same key; otherwise returns without output
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 11/12/21/22 | 950、非空、同 shape；低精度 x、8-bit mask | 单 AIV | `MaskCastTwo<half/bfloat16_t,uint8_t/int8_t>` | B=8；两路 FP32 cast值同时存活；PIPE mutex | former/tail字段；无fallback |
| R-002 | 13/23 | 950、F32 x、8-bit mask | 单 AIV | `MaskCastTwo<float,uint8_t/int8_t>` | x CastNoOp；B=7 | 同上 |
| R-003 | 31/32 | 950、低精度 x、F16 mask | 单 AIV | `MaskCastOne<half/bfloat16_t,half>` | 两个FP32 cast值；B=8 | 同上 |
| R-004 | 33 | 950、F32 x、F16 mask | 单 AIV | `MaskCastOne<float,half>` | x/final CastNoOp；B=7 | 同上 |
| R-005 | 41/42 | 950、低精度 x、F32 mask | 单 AIV | `MaskCastOne<half/bfloat16_t,float>` | mask CastNoOp；B=7 | 同上 |
| R-006 | 43 | 950、F32/F32 | 单 AIV | `MaskCastOne<float,float>` | 输入/final CastNoOp；Mul temp一块，B=7 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台公开性 | 文档支持950且 `IsRegBase()` | API dtype表只在 RegBase返回 | 全部 | 参数非法；A2/A3不建卡 | `GetXDtypeSupportList/GetMaskDtypeSupportList` |
| 空域 | `self->IsEmpty()` | 公共 Tiling拒绝 shape维含0 | 全部 | workspace=0，无Kernel | `aclnnMaskedScaleGetWorkspaceSize` |
| exact shape | L2三 Tensor shape相同；Tiling `IsSameElewiseShape`成功 | 本 DAG无广播地址生成 | 全部 | `ACLNN_ERR_PARAM_INVALID`/`GRAPH_FAILED` | `CheckShape`; `GetShapeInfo` |
| 4 KiB/核 | `C=ceil(N*bmin/32768)` 后受 P 限制 | 防止初始分核域过小 | 全部 | 自动减少blockDim | `ElewiseTiling` |
| UB tile | `U>=0` 且 `ubFormer=align_down(U/(B*4),2048/bmin)>0` | tensor pool必须容纳 B 个最大4B slice | 全部 | `GRAPH_FAILED` | `GetMaxElemNum`; `ElementwiseSch::Init` |
| key一致性 | Host key属于11/12/13/21/22/23/31/32/33/41/42/43 | entry只有这些分支 | 全部 | Tiling失败或entry无写出 | `Tiling4DoTilingDag`; `masked_scale` |
| workspace指针 | `workspace!=nullptr && GetUserWorkspace()!=nullptr` | entry在建 `TPipe` 前硬返回 | 全部 | 无输出 | `masked_scale` |
| CastNoOp | 源/目标 dtype相同 | DAG编译期移除冗余 Cast，改变B与tile上界 | R-002/4/5/6 | 改dtype后必须重算DAG | `GetRealArgs`; `Vec::IsCastNoOp` |

### POD/ABI

`EleBaseTilingData` 采用自然 C++ ABI：`sizeof=80, alignof=8`；外层 `MaskedScaleTilingData` 的 `scale` 位于 offset 80，尾部补齐后 `sizeof=88, alignof=8`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `MaskedScaleTilingData.baseTiling` | `dim0` | `int64_t/64` | offset 0 | 元素 | `ElewiseTiling: product(shape)` | 元数据；Process按分核字段覆盖 | 非空；乘法溢出检查固定仓不可见 |
| 同上 | `coreNum` | `int32_t/32` | offset 8 | AIV核 | `min(P,ceil(N*bmin/32768))` | 元数据 | `[1,P]`，写入时窄化为int32 |
| 同上 | `ubFormer` | `int32_t/32` | offset 12 | 元素/tile | `GetMaxElemNum` | `InitBuffer/blockLen/Process` | `>0`，由int64窄化 |
| 同上 | `blockFormer` | `int64_t/64` | offset 16 | 元素/核 | `align_up(ceil(N/C),512)` | `CalcBlockOffset` | 最后一核只读blockTail有效域 |
| 同上 | `blockNum` | `int64_t/64` | offset 24 | 启动核 | `ceil(N/blockFormer)` | last-block判断；Host `SetBlockDim` | `1..C` |
| 同上 | `ubLoopOfFormerBlock` | `int64_t/64` | offset 32 | loop | `ceil(blockFormer/ubFormer)` | 非尾核 `Process` | ≥1 |
| 同上 | `ubLoopOfTailBlock` | `int64_t/64` | offset 40 | loop | `ceil(blockTail/ubFormer)` | 尾核 `Process` | ≥1 |
| 同上 | `ubTailOfFormerBlock` | `int64_t/64` | offset 48 | 元素 | `blockFormer-(loops-1)*ubFormer` | 非尾核末tile | `1..ubFormer` |
| 同上 | `ubTailOfTailBlock` | `int64_t/64` | offset 56 | 元素 | `blockTail-(loops-1)*ubFormer` | 尾核末tile | `1..ubFormer` |
| 同上 | `elemNum` | `int64_t/64` | offset 64 | 元素 | `maxElemNum` | 当前 scheduler未直接读取 | =ubFormer，保留ABI |
| 同上 | `scheMode` | `uint64_t/64` | offset 72 | 调度模式 | `GetScheMode()=1` | 当前 entry模板固定 `ElementwiseSch<0>`，未读取 | inactive但必须保持布局 |
| `MaskedScaleTilingData` | `scale` | `float/32` | offset 80；结构size88 | 数值 | `*GetAttrPointer<float>(0)` | entry `SetVar<float,0>`→`Muls<float>` | attr指针空检查固定仓不可见 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 11/12/21/22 | `masked_scale_tiling_arch35.cpp::Tiling4MaskedScale/Tiling4DoTilingDag` | `MaskedScaleTilingData` | `masked_scale_apt.cpp::masked_scale` | `MaskedScaleMaskCastTwo` + `ElementwiseSch/Vec` | OpDef 950 + binary 12项 + CMake arch35 |
| R-002 | 13/23 | 同上 | 同上 | 同上 | `MaskCastTwo<float,U8/I8>` | 同上 |
| R-003 | 31/32 | 同上 | 同上 | 同上 | `MaskCastOne<half/bfloat16_t,half>` | 同上 |
| R-004 | 33 | 同上 | 同上 | 同上 | `MaskCastOne<float,half>` | 同上 |
| R-005 | 41/42 | 同上 | 同上 | 同上 | `MaskCastOne<half/bfloat16_t,float>` | 同上 |
| R-006 | 43 | 同上 | 同上 | 同上 | `MaskCastOne<float,float>` | 同上 |

## Kernel 实现

### 公共执行流程

entry先验证 system/user workspace、声明 `KERNEL_TYPE_AIV_ONLY`、解析 POD并按 key实例化 `ElementwiseSch<0,OpDag>`。`Init` 将每路 GM基址偏移到 `blockFormer*blockIdx*sizeof(dtype)`，一次分配 `B*ubFormer*4` 字节 `TBuf` tensor pool。`Process` 对完整 tile 调 `Run(offset,ubFormer,i&1)`，最后一次用对应 former/tail长度。每个 `Run` 按 DAG拓扑执行 CopyIn、必要 Cast、FP32 Mul、FP32 Muls、必要 RINT Cast、CopyOut。`GetTensor/ReleaseTensor` 映射到 `Mutex::Lock/Unlock<PIPE_MTE2/PIPE_V/PIPE_MTE3>`，编译期 buffer id与 tile奇偶共同保证物理子区复用安全。

### Route R-001：低精度 x + 8-bit mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_masked_scale.cpp` | `CheckDtypeValid` | x∈F16/BF16，mask∈U8/I8 | 允许进入L0 |
| Tiling | `op_host/arch35/masked_scale_tiling_arch35.cpp` | `Tiling4DoTilingDag` | key 11/12/21/22 | `MaskCastTwo<half,U8/I8>`做资源生产；BF16与half同宽 |
| Router | `op_kernel/masked_scale_apt.cpp` | `masked_scale` | `TILING_KEY_IS(11/12/21/22)` | body使用真实 half/bfloat16_t |

#### Device 数据流

##### 函数级调用链

```text
l0op::MaskedScale -> masked_scale<key 11/12/21/22>
  -> ElementwiseSch::Init/Process/Run
    -> CopyIn<Tx>(x) + CopyIn<Tmask>(mask)
    -> Cast<float,Tx,0>(x) + Cast<half,Tmask,0> -> Cast<float,half,0>(mask)
    -> Mul<float> -> Muls<float>(scale) -> Cast<Tx,float,1>
    -> CopyOut<Tx>(y)
```

##### 源码等价伪码

```text
baseX=b*blockFormer; baseM=b*blockFormer; baseY=b*blockFormer
for tile i in this block:
  len = (last tile ? route-specific tail : ubFormer)
  ux = DataCopyPad(x[baseX+off : +len])
  um = DataCopyPad(mask[baseM+off : +len])
  fx = CastFP32(ux); hm = CastFP16(um); fm = CastFP32(hm)
  fy = CastTx_RINT((fx * fm) * scale)
  DataCopyPad(y[baseY+off : +len], fy)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/mask GM | `b*blockFormer+off` | `len` | Tx/Tmask ND连续 | caller/framework | CopyIn | caller/executor | tail只搬有效字节 |
| tensorPool | `bufId*ubFormer*4` | `8*ubFormer*4` | byte slices | `InitBuffer` | DAG nodes | Kernel/TPipe | 4个MTE2 ping/pong+2 MTE3+2 temp |
| y GM | 同逻辑offset | `len` | Tx ND连续 | CopyOut | ViewCopy | executor | 不写padding lane |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xPP["x MTE2 ping/pong · 2×4u B"]:1
  mPP["mask MTE2 ping/pong · 2×4u B"]:1
  tmp["FP32 temp · 2×4u B"]:1
  yPP["y MTE3 ping/pong · 2×4u B"]:1
```

##### 字节账与复用顺序

```text
u=ubFormer; UB_total=8*u*4=32u <= U; dynamic_ub=N/A
workspace core reserve=16MiB; ACLNN workspace_total由executor汇总
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| temp0/temp1 | 两路Cast/Mul/Muls/Cast3 | 后继Vector或CopyOut | 输入最后消费后/下一tile同parity | PIPE_V mutex + liveness buffer id | 不得绕过 `PostArgs/ReleaseTensor` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad`, MTE2 mutex | GM | Cast | x/mask slice | blockCount=1, blockLen=`len*sizeof(T)` bytes，stride=0，无pad |
| compute | PIPE_V mutex | Cast | Mul/Muls/Cast | temp slices | 8-bit→FP16→FP32；其值域无舍入；数学FP32 |
| UB→GM | `DataCopyPad`, MTE3 mutex | final Cast | GM y | output slice | `RoundMode=1`; 只提交len元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `math/masked_scale/op_host/arch35/masked_scale_tiling_arch35.cpp` | `Tiling4DoTilingDag` | keys与资源DAG |
| Kernel entry/router | `math/masked_scale/op_kernel/masked_scale_apt.cpp` | `masked_scale` 11/12/21/22 | BF16真实body与scale绑定 |
| Device body/helper | `math/masked_scale/op_kernel/arch35/masked_scale_dag.h` | `MaskedScaleMaskCastTwo` | 双Cast mask与FP32乘链 |

### Route R-002：FP32 x + 8-bit mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `masked_scale_tiling_arch35.cpp` / `masked_scale_apt.cpp` | key 13/23 | x=F32, mask=U8/I8 | `MaskCastTwo<float,Tmask>` |

#### Device 数据流

##### 函数级调用链

```text
masked_scale<13/23> -> CopyIn<float/U8|I8> -> mask Cast half->float
  -> Mul<float> -> Muls<float> -> CopyOut<float>
```

##### 源码等价伪码

```text
for each valid tile: y = x_float * float(CastHalf(mask8)) * scale; copy len floats
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/tensorPool | `b*blockFormer+off` / `bufId*u*4` | len / `7*u*4` | F32+8bit→F32 | CopyIn/DAG | CopyOut | Kernel | x/final CastNoOp，1 temp |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inPP["2 inputs MTE2 · 4 slices"]:1
  tmp["mask/Mul temp · 1 slice"]:1
  outPP["output MTE3 · 2 slices"]:1
```

##### 字节账与复用顺序

```text
UB_total=7*u*4=28u<=U; workspace core reserve=16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| temp | mask Cast/Mul | next op | liveness结束 | PIPE_V mutex | key13/23的CastNoOp集合不变 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| all | 同公共PIPE协议 | CopyIn/compute | compute/CopyOut | pool slices | x/y已FP32；mask精确升FP32；len保护tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `masked_scale_tiling_arch35.cpp` | keys13/23 | float资源模板 |
| Kernel entry/router | `masked_scale_apt.cpp` | keys13/23 | float body |
| Device body/helper | `masked_scale_dag.h` + opbase `dag.h` | `MaskCastTwo<float,Tmask>` | 同类型Cast消除、B=7 |

### Route R-003：低精度 x + FP16 mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `masked_scale_tiling_arch35.cpp` / `masked_scale_apt.cpp` | key31/32 | x=F16/BF16, mask=F16 | `MaskCastOne<Tx,half>` |

#### Device 数据流

##### 函数级调用链

```text
CopyIn<Tx/half> -> Cast both to FP32 -> Mul -> Muls(scale) -> Cast Tx,RINT -> CopyOut
```

##### 源码等价伪码

```text
for each valid tile: yTx = RINT_Tx(float(xTx)*float(maskHalf)*scale)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tensorPool | `bufId*u*4` | `8*u*4` | max F32 | scheduler | DAG | Kernel | 两个cast值同时存活 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  xPP["x · 2 slices"]:1
  mPP["mask · 2 slices"]:1
  tmp["FP32 temp · 2 slices"]:1
  yPP["y · 2 slices"]:1
```

##### 字节账与复用顺序

```text
UB_total=32u<=U; workspace core reserve=16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| two temp slices | x/mask Cast | Mul | Mul之后 | PIPE_V liveness | 两输入均在Mul前保持有效 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | Cast→Mul→Muls→Cast | inputs | y | two FP32 temps | F16/BF16只在边界舍入；尾len直传所有Vector API |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `masked_scale_tiling_arch35.cpp` | key31/32 | BF16用同宽half模板计算资源 |
| Kernel entry/router | `masked_scale_apt.cpp` | key31/32 | body区分half/bfloat16_t |
| Device body/helper | `masked_scale_dag.h` | `MaskCastOne<Tx,half>` | 单级mask Cast |

### Route R-004：FP32 x + FP16 mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `masked_scale_tiling_arch35.cpp` / `masked_scale_apt.cpp` | key33 | x=F32, mask=F16 | `MaskCastOne<float,half>` |

#### Device 数据流

##### 函数级调用链

```text
CopyIn<float/half> -> Cast mask to FP32 -> Mul -> Muls -> CopyOut<float>
```

##### 源码等价伪码

```text
for each valid tile: y = x * float(maskHalf) * scale
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tensorPool | `bufId*u*4` | `28u` | max F32 | scheduler | DAG | Kernel | B=7 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inPP["inputs · 4 slices"]:1
  tmp["FP32 temp · 1 slice"]:1
  outPP["output · 2 slices"]:1
```

##### 字节账与复用顺序

```text
UB_total=28u<=U; workspace core reserve=16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| temp | mask Cast/Mul | next node | last use后 | PIPE_V mutex | key33 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute/copy | 公共mutex | mask Cast | FP32 Mul/GM | temp/output | 无输出降精度；tail len |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `masked_scale_tiling_arch35.cpp` | key33 | B=7资源DAG |
| Kernel entry/router | `masked_scale_apt.cpp` | key33 | float/half实例 |
| Device body/helper | `masked_scale_dag.h` | `MaskCastOne<float,half>` | x/final CastNoOp |

### Route R-005：低精度 x + FP32 mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `masked_scale_tiling_arch35.cpp` / `masked_scale_apt.cpp` | key41/42 | x=F16/BF16, mask=F32 | `MaskCastOne<Tx,float>` |

#### Device 数据流

##### 函数级调用链

```text
CopyIn<Tx/float> -> Cast x to FP32 -> Mul -> Muls -> Cast Tx,RINT -> CopyOut
```

##### 源码等价伪码

```text
for each valid tile: yTx = RINT_Tx(float(xTx)*maskFloat*scale)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tensorPool | `bufId*u*4` | `28u` | max F32 | scheduler | DAG | Kernel | mask CastNoOp，B=7 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inPP["inputs · 4 slices"]:1
  tmp["FP32 temp · 1 slice"]:1
  outPP["output · 2 slices"]:1
```

##### 字节账与复用顺序

```text
UB_total=28u<=U; workspace core reserve=16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| temp | x Cast/Mul/Muls | next node | last use后 | PIPE_V mutex | key41/42 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | FP32 chain + RINT | inputs | CopyOut | temp/output | mask保持FP32；输出低精度；tail len |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `masked_scale_tiling_arch35.cpp` | key41/42 | half等宽资源模板 |
| Kernel entry/router | `masked_scale_apt.cpp` | key41/42 | half/bfloat body |
| Device body/helper | `masked_scale_dag.h` | `MaskCastOne<Tx,float>` | mask CastNoOp |

### Route R-006：FP32 x + FP32 mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling/Router | `masked_scale_tiling_arch35.cpp` / `masked_scale_apt.cpp` | key43 | x=F32, mask=F32 | `MaskCastOne<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
CopyIn<float>(x,mask) -> Mul<float> -> Muls<float>(scale) -> CopyOut<float>
```

##### 源码等价伪码

```text
for each block and valid tile: y[offset:offset+len]=(x*mask)*scale
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/mask/y GM | `b*blockFormer+off` | len | F32 ND | CopyIn/CopyOut | Vector/caller | caller/executor | exact tail |
| tensorPool | `bufId*u*4` | `28u` | F32 slices | scheduler | DAG | Kernel | B=7，只有Mul结果temp |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inPP["x+mask ping/pong · 4×4u B"]:1
  mul["Mul temp · 1×4u B"]:1
  outPP["Muls/output ping/pong · 2×4u B"]:1
```

##### 字节账与复用顺序

```text
UB_total=28u<=U; dynamic_ub=N/A; core workspace reserve=16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Mul temp | Mul | Muls | Muls消费后 | PIPE_V mutex/ReleaseTensor | 不将Muls提前到两输入就绪前 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn→Mul→Muls→CopyOut | MTE2/V/MTE3 mutex | 前一阶段 | 后一阶段 | 编译期buffer id | 全程FP32；最后tile只处理tailNum |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `masked_scale_tiling_arch35.cpp` | key43 | bmin=32、B=7 |
| Kernel entry/router | `masked_scale_apt.cpp` | key43 | float/float实例 |
| Device body/helper | `masked_scale_dag.h` + opbase `vec.h` | Mul/Muls/CopyIn/CopyOut | 最简FP32管线 |

## 算子特定优化

### Trick T-001：按 mask dtype选择单/双 Cast DAG，并消除同类型 Cast

#### 触发与源码依据

- 适用 Route：R-001～R-006。
- 触发条件：8-bit mask用 `MaskedScaleMaskCastTwo`；F16/F32 mask用 `MaskedScaleMaskCastOne`；其中源目标同 dtype 的 Cast由 `DAGSch` 的 CastNoOp分析删除。
- Golden 来源：[`math/masked_scale/op_kernel/arch35/masked_scale_dag.h:25-55`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/math/masked_scale/op_kernel/arch35/masked_scale_dag.h#L25-L55)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
using Cast1 = Bind<Vec::Cast<half, Tmask, 0>, OpCopyIn1>;
using Cast2 = Bind<Vec::Cast<float, half, 0>, Cast1>;
using OpMul = Bind<Vec::Mul<float>, Cast0, Cast2>;
using OpMuls = Bind<Vec::Muls<float>, OpMul, Placeholder::Var<float, 0>>;
using Cast3 = Bind<Vec::Cast<Tx, float, 1>, OpMuls>;
```

#### 机制与设计原因

8-bit mask先走 half再到float，保持其全部整数值精确可表示，同时复用通用 Vector Cast；F16/F32 mask避免多余一级。DAG在FP32输入/输出处继续消除 no-op Cast，使峰值temp从两块降为一块，Host以同一 `OpDag::BufferNum` 反推更大的合法tile，Device也用该值分配tensor pool。该机制改变了指令数和UB容量，而不仅是模板封装。

#### 迁移边界

- 适用：两个输入最终在FP32逐元素计算、8-bit输入值域可由FP16精确表示的融合算子。
- 前提：Host和Device必须实例化资源等价的同一DAG；Cast round mode、dtype和buffer liveness必须一致。
- 失效：更宽整数或要求直接整数→FP32特殊舍入时不能复用双Cast路径；增删分支后必须重新由DAG计算B。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得为所有key硬编码同一 `BufferNum`，否则可能低估R-001/R-003的两块同时存活temp。

### Trick T-002：Host/Device共用DAG元数据闭合UB上界与物理复用

#### 触发与源码依据

- 适用 Route：R-001～R-006。
- 触发条件：`MemOptCfg<LEVEL_2>`，Host `DoTiling<OpDag>` 与 Device `ElementwiseSch<...,OpDag>` 使用相同 `BufferNum/MaxDtypeBytes`。
- Golden 来源：[`pkg_inc/op_common/atvoss/elewise/elewise_sch.h:41-51`](https://gitcode.com/cann/opbase/blob/d574f5571391390f8b619c6a134b65c819842318/pkg_inc/op_common/atvoss/elewise/elewise_sch.h#L41-L51)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
RUN_LOG("BufferNum: %d, Mte2Num: %d, Mte3Num: %d, BufLevel: %d",
        ElemDag::BufferNum, ElemDag::Mte2Num,
        ElemDag::Mte3Num, ElemDag::BufLevel);
pipePtr->InitBuffer(buf, tilingData->ubFormer * ElemDag::MaxDtypeBytes * ElemDag::BufferNum);
tensorPool = buf.Get<uint8_t>();
blockLen = tilingData->ubFormer * ElemDag::MaxDtypeBytes;
```

#### 机制与设计原因

Host用 `U/(B*4)` 计算tile，Device用完全相同的乘积一次分配pool；每个DAG节点只持有编译期分配的slice，生命周期结束后由后继节点或下一tile复用。这样避免为每个Cast/Mul/Muls永久物化一块UB，同时容量证明不依赖手工估算。tile的 `i&1` 在MTE2/MTE3 slice上形成ping/pong，PIPE mutex保证跨搬运/计算管线的可见性。

#### 迁移边界

- 适用：可静态展开的线性/小分叉Vector DAG。
- 前提：Host与entry模板、CastNoOp集合、LEVEL_2策略和最大dtype必须完全一致。
- 失效：新增高级API临时区、持久buffer或改变执行图时，旧B与tile公式失效，应重新取 `OpDag` 元数据。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得把逻辑Tensor有效字节相加代替 `B*ubFormer*MaxDtypeBytes`，也不得忘记MTE2/MTE3 ping/pong倍数。

## 模板与实现对比

| 对比维度 | R-001 | R-002 | R-003 | R-004 | R-005 | R-006 | 设计选择依据 |
|---|---|---|---|---|---|---|---|
| 命中条件/优先级 | low+8bit | F32+8bit | low+F16 | F32+F16 | low+F32 | F32+F32 | dtype key直接映射，无运行时fallback |
| 模板与 dtype | CastTwo low | CastTwo float | CastOne low/half | CastOne float/half | CastOne low/float | CastOne float/float | mask所需Cast级数 |
| 循环与分核 | 公共公式，bmin=8 | bmin=8 | bmin=16 | bmin=16 | bmin=16 | bmin=32 | 最小dtype控制4KiB/核与256B对齐 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | TBuf B=8 | B=7 | B=8 | B=7 | B=7 | B=7 | 无显式TQue/dynamic UB/通信 |
| 内存总占用与复用 | 32u | 28u | 32u | 28u | 28u | 28u | 两路Cast同时存活时多1 temp |
| 搬运/Vector/SIMT/NDDMA/通信 API | Pad+Cast×3+Mul/Muls | Cast mask×2 | Cast×2+Mul/Muls | Cast mask | Cast x+out | Mul/Muls | 全为AIV Vector，无SIMT/NDDMA |
| 同步、精度与 tail | FP32数学，RINT输出，mutex，tail len | 全FP32输出 | FP32数学/RINT | 全FP32输出 | FP32数学/RINT | 全FP32 | scheduler统一tail字段 |
| fallback/失败边界 | unknown key失败/无写出 | 同左 | 同左 | 同左 | 同左 | 同左 | 无第二Kernel fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档明确不支持 | 公开文档明确不支持 | Ascend 950PR/950DT支持 |
| Host/Launcher | 不发布 | 不发布 | RegBase dtype表；OpDef `ascend950`；arch35 Tiling |
| Kernel/资源 | 无可发布闭包 | 无可发布闭包 | AIV_ONLY，12个binary dtype项，ATVOSS LEVEL_2 pool |

## 限制与失败边界

- 只支持公开 dtype组合、ND、三 Tensor exact same shape，`self/y` storage format相同，`self/mask` rank≤8；不执行广播。
- 文档的 mask“非连续Tensor”栏为未承诺，而 header注释与实现会调用 `Contiguous(mask)`；本卡按较窄公开契约，不把实现行为推广为保证。
- `dim0` 各维乘积、`coreNum/ubFormer` 向 int32窄化及 `count*sizeof(T)` 向 uint32窄化的极端溢出保护在固定仓不可见。
- entry遇到 null workspace/null user workspace/未知key直接返回而不写输出；Host正常路径通过Tiling和workspace约束避免这些终点。
- 16 MiB是核心节点声明的system workspace，不是所有输入形态下ACLNN总 workspace；framework offsets、events与复用属于executor边界。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `math/masked_scale/docs/aclnnMaskedScale.md` | 产品支持/接口表 | 950公开性、dtype/shape/连续性契约 |
| 入口实现 | `math/masked_scale/op_api/aclnn_masked_scale.cpp` | `aclnnMaskedScaleGetWorkspaceSize/aclnnMaskedScale` | 校验、empty、Contiguous、ViewCopy、executor |
| 入口实现 | `math/masked_scale/op_api/masked_scale.cpp` | `l0op::MaskedScale` | 临时输出与AICore launcher |
| Host/Tiling/Launcher | `math/masked_scale/op_host/arch35/masked_scale_tiling_arch35.cpp` | `Tiling4MaskedScale/Tiling4DoTilingDag` | key、DAG、POD、blockDim、workspace |
| Host/Tiling/Launcher | `math/masked_scale/op_kernel/arch35/masked_scale_tiling_struct.h` | `MaskedScaleTilingData` | POD外层结构 |
| Kernel entry/body | `math/masked_scale/op_kernel/arch35/masked_scale_dag.h` | `MaskedScaleMaskCastOne/Two` | 两个计算DAG |
| Kernel entry/body | `math/masked_scale/op_kernel/masked_scale_apt.cpp` | `masked_scale` | AIV_ONLY、12-key router |
| 构建接线 | `math/masked_scale/op_host/masked_scale_def.cpp` | `MaskedScale::MaskedScale` | 950 OpDef/dtype/AutoContiguous |
| 构建接线 | `math/masked_scale/op_host/config/ascend950/masked_scale_binary.json` | `op_list` | 12个950 binary组合 |
| 构建接线 | `math/masked_scale/CMakeLists.txt` | `add_all_modules_sources` | arch35/950活动接线及非950遗留信号 |
| Host/Tiling/Launcher | `opbase: pkg_inc/op_common/atvoss/elewise/elewise_base_struct.h` | `EleBaseTilingData` | POD字段/ABI |
| Kernel entry/body | `opbase: pkg_inc/op_common/atvoss/elewise/elewise_sch.h` | `ElementwiseSch` | block/tile、pool、调用链 |
| Kernel entry/body | `opbase: pkg_inc/op_common/atvoss/util/dag.h` | `DAGSch` | CastNoOp、BufferNum、buffer IDs |
| Kernel entry/body | `opbase: pkg_inc/op_common/atvoss/util/vec.h` | `Cast/Mul/Muls/CopyIn/CopyOut` | 实际AscendC primitives与参数 |
| Kernel entry/body | `opbase: pkg_inc/op_common/atvoss/util/sync.h` | `GetTensor/ReleaseTensor` | PIPE mutex协议 |
| Host/Tiling/Launcher | `opbase: src/op_common/atvoss/elewise/elewise_tiling.cpp` | `ElewiseTiling/GetMaxElemNum` | 分核、tile、tail与UB公式 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 80B base + offset80 scale，外层88B | 字段顺序、int32窄化、scale位置 | `EleBaseTilingData`; `MaskedScaleTilingData` |
| 调用顺序 | Contiguous两输入→MaskedScale→ViewCopy | producer失败必须阻断consumer | `aclnnMaskedScaleGetWorkspaceSize` |
| workspace/内存 | 核心16MiB；UB=`B*u*4` | null guard、executor总offset不可臆测、B按key | `Tiling4MaskedScale`; `ElementwiseSch::Init` |
| 同步 | MTE2/V/MTE3 mutex + tile parity | buffer id/liveness与DAG一致 | `GetTensor/ReleaseTensor`; `GetBufferIds` |
| 精度与尾块 | 8bit→half→float；FP32数学；RINT输出；tail len | 不更改Cast顺序/round mode/有效len | `MaskedScaleMaskCastOne/Two`; `Process` |
| 失败边界 | empty无Kernel；dtype/key/UB失败；entry null无写出 | 不把未知key当fallback | `CheckParams`; `Tiling4DoTilingDag`; `masked_scale` |
