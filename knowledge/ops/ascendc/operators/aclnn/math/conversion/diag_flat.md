---
type: Operator
title: "DiagFlat A2/A3 ACLNN 算子知识"
description: "说明 aclnnDiagFlat 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/docs/aclnnDiagFlat.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_host/op_api/aclnn_diag_flat.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_host/op_api/aclnn_diag_flat.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_host/op_api/diag_flat.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_host/diag_flat_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_host/diag_flat_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_v2/op_host/diag_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_kernel/diag_flat.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_kernel/diag_flat_nd_to_2d_with_few.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_host/config/ascend910b/diag_flat_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_host/config/ascend910_93/diag_flat_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-24T23:51:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# DiagFlat A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnDiagFlatGetWorkspaceSize` + `aclnnDiagFlat`，公开 ACLNN 两段式接口；固定仓只存在该版本族。 |
| 计算语义 | 将 `self` 按逻辑顺序展平为长度 `N` 的向量，在宽度 `W=N+abs(diagonal)` 的方阵指定对角线上写入该向量，其余元素为零。 |
| 输入输出 | `self` 为 1～8 维 ND、可非连续；公开文档列出 FP16/FP32/FP64、整数、BOOL、COMPLEX64、BF16，`out` 应与 `self` 同 dtype，shape 为 `[W,W]`。 |
| 执行类型 | ACLNN executor 可能包含框架 `Contiguous`/`Cast`/`ViewCopy` 节点；算子自有节点只有一个 `diag_flat`，执行族为 `AIV_VECTOR`。 |
| Kernel 路线 | R-001：`N<=64`、key 101、小输入块内标量散写；R-002：`N>64`、key 103、64×64 辅助矩阵与按位 `And` 向量化。公开 ACLNN dtype 不会产生 16 字节 key 102/104。 |
| 核心资源 | GM 输入/输出；depth=1 的 VECIN/VECOUT Queue；R-002 额外 VECCALC 辅助矩阵；workspace 前 `32*C_total` 字节用于核间同步，Tiling 另加 16 MiB 系统 workspace。 |
| 来源闭包 | `source_closed`：公开入口、L0 launcher、Tiling/POD、key 101/103 router、两条 AIV body、同步和 A2/A3 binary 接线已闭合；框架通用节点以明确边界保留。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `v=flatten(self)`、`N=numel(self)`、`d=diagonal`、`W=N+abs(d)`。输出 `y` 为 `W×W` 方阵：当 `d>=0` 时 `y[i,i+d]=v[i]`；当 `d<0` 时 `y[i-d,i]=v[i]`；其余位置为零，`0<=i<N`。L0 `DiagFlat` 按该公式分配 `[W,W]` 临时输出。该逻辑只描述 DiagFlat，不包含 `Diag` 的二维输入取对角线语义，也不包含 Grad、原地或量化变体。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 文档为 1～8 维 ND；允许非连续，进入算子前调用 `Contiguous` | A2/A3 API 检查表：FLOAT16、FLOAT、DOUBLE、INT64、INT32、INT16、INT8、UINT8、BOOL、COMPLEX64、BFLOAT16 | 最大 rank 由 `MAX_SUPPORT_DIMS_NUMS` 检查；代码未检查最小 rank。空 Tensor 在 ACLNN 第一段提前成功返回，不创建 DiagFlat 节点。 |
| `diagonal` | Host 标量 | `int64_t` | `0` 为主对角线，正数向上、负数向下；L0 与 Tiling 均使用 `abs(diagonal)` 计算 `W`，固定源码未见对 `abs(INT64_MIN)` 或 `W*W` 溢出的保护。 |
| `out` | 文档为 ND、可非连续，预期 shape `[W,W]` | 文档要求与 `self` 一致；实现只分别校验二者各自在支持列表内 | `ViewCopy` 把临时结果写入调用者对象；固定 API 实现未显式检查 `out` shape 与 `self` dtype 相等，dtype 不等时会插入 `Cast`。 |
| `workspaceSize` | Host 输出 | `uint64_t*` | 返回 executor 汇总 workspace；固定实现未显式检查该输出指针或 `executor` 二级指针，交给框架宏/对象边界。 |
| `executor` | Host 输出/第二段输入 | `aclOpExecutor**` / `aclOpExecutor*` | 第一段持有并组装任务，`ReleaseTo` 后由调用者传给第二段。 |
| `workspace`、`stream` | Device 地址、ACL stream | `void*`、`aclrtStream` | 第二段由 `CommonOpExecutorRun` 在指定 stream 上执行整个任务图。 |

### 入口与版本

公开声明为 `aclnnDiagFlatGetWorkspaceSize` 与 `aclnnDiagFlat`。第一段创建 executor、校验、连续化输入、调用 L0 `DiagFlat`、调用 `Cast` 和 `ViewCopy` 并返回 workspace；第二段调用 `CommonOpExecutorRun`。固定仓未发现更高版本公开符号或旧版本 body 复用关系，因此 A2/A3 采用该唯一版本。公开 ACLNN 文档明确 A2/A3 支持、Ascend 950 不支持；950 的 OpDef、binary config 和通用 README 接线与公开 ACLNN 表冲突，因此本卡不把这些实现信号解释为 950 公开支持。

## Host/Launcher 执行流程

### 参数校验与前处理

调用顺序为：`CREATE_EXECUTOR` → `CheckParams` → 空输入短路 → `l0op::Contiguous(self)` → `l0op::DiagFlat` → `l0op::Cast` → `l0op::ViewCopy` → `GetWorkspaceSize`/`ReleaseTo`。`CheckParams` 先检查 `self/out` 非空，再分别检查 dtype，再检查二者最大 rank，最后只对 `FRACTAL_NZ` 输入记录精度告警。接口文档要求 dtype 相同，但源码没有等值检查；OpDef 列表也未包含 ACLNN 检查表中的 BOOL，因此 BOOL 会通过 ACLNN 前置检查、但算子注册层是否接受存在固定源码内的契约冲突。任一节点返回空指针即返回内部空指针错误。`self->IsEmpty()` 时 `workspaceSize=0`，executor 为空任务图并直接成功；此路径不会显式清零 `abs(d)×abs(d)` 的非空输出。

### 参数、Workspace 与生命周期

L0 根据 `N=self->Numel()` 和 `d` 分配临时 `[N+abs(d),N+abs(d)]`，dtype 沿用 `self`。Tiling 从平台读取 `C_total=GetCoreNumAiv()` 与 `UB_platform`，设置 DiagFlat 节点 workspace：`userSize=32*C_total` 字节，`sysWorkspaceSize=16*1024*1024` 字节，节点总量为二者之和；kernel 只把起始 `32*C_total` 字节绑定成 `int32_t` 同步区。ACLNN 返回的是包含 `Contiguous/Cast/ViewCopy` 在内的 executor 汇总值，其额外切片、offset 与对齐由通用框架管理，固定算子源码不可见。executor 在第一段构造并转交调用者，第二段执行期间持有临时 Tensor 和 workspace；stream 完成前不得复用。

### 启动图与失败传播

```text
aclnnDiagFlatGetWorkspaceSize
  -> [framework boundary] Contiguous(self)          # 非连续输入可能产生节点
  -> l0op::DiagFlat
       -> DiagFlatAiCore / ADD_TO_LAUNCHER_LIST_AICORE
          -> TilingDiagFlat
          -> diag_flat(input, tempOut, workspaceSlice, tiling)
  -> [framework boundary] Cast(tempOut, out.dtype)  # 合法文档合同下 dtype 相同
  -> [framework boundary] ViewCopy(castOut, out)    # 支持非连续 out
aclnnDiagFlat(workspace, workspaceSize, executor, stream)
  -> CommonOpExecutorRun
```

算子自有 launch graph 是单 Kernel：Host 节点 `DiagFlatAiCore` 启动 entry `diag_flat`，`blockDim=C_total`，POD 为 `DiagV2TilingData`，输入为连续 `self`，输出为 L0 临时方阵，workspace slice 的前 `32*C_total` 字节用于 `SyncAll`。Tiling 设置 `ScheduleMode(1)`，且构建配置 `auto_sync:true`。节点内所有 AIV 先并行清零输出与同步区，再核间 `SyncAll`，随后写对角线。executor stream 保证 DiagFlat 临时输出在后续 Cast/ViewCopy 读取前可见；通用节点的内部 kernel、额外 workspace 边和 no-op 判定固定算子源码不可见。任何 `ADD_TO_LAUNCHER_LIST_AICORE`、Contiguous、DiagFlat、Cast 或 ViewCopy 组图失败都会阻断后续节点；第二段直接传播 `CommonOpExecutorRun` 状态。

## Route 与参数设计

### 分核与资源计算

令 `s=sizeof(self.dtype)`、`N=inputNum`。Tiling 总是以 `C_total=GetCoreNumAiv()` 启动。R-001 设 `C_used=1`、`H_normal=N`、`H_last=0`。R-002 先算 `H=ceil_align(max(ceil_div(N,C_total),64),64)`，再设 `C_used=ceil_div(N,H)`、`H_normal=H`、`H_last=N-(C_used-1)H`。R-002 Device 以 64 个元素为一 tile；尾核不足 64 时把 `inputIdx` 回退到一个完整 64 元素块并令 `curHandleNum=ceil_align(H_last,64)`，所以尾核可能与前核幂等地重复写相同对角线元素。

Host 还写入 `ubOutputSize=64*64*sizeof(dataType)`、`ubInputSize=UB_platform-2*64*64*sizeof(dataType)`；这里源码表达式对 `ge::DataType` 变量取 `sizeof`，且两字段不被 DiagFlat Device body 读取，不能把它们当作实际 UB 分配。实际 UB 由各 Route 的 `InitBuffer` 常量决定。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 小输入散写 | key 101；`DiagFlatNDTo2DWithFew<T>` | 非空且 `N<=64`；`ge::GetSizeByDataType(dtype)!=16`。公开 A2/A3 ACLNN dtype 均落入 1/2/4/8 字节实例。 | `N>64` 转 R-002；16 字节转源码 dormant key 102，但公开入口检查表不允许该 dtype。 |
| 2 | R-002 64×64 向量块 | key 103；`DiagFlatNDTo2D<T>` | `N>64`；`ge::GetSizeByDataType(dtype)!=16`。router 再按 `sizeof(DTYPE_X)` 选择 `int8_t/int16_t/int32_t/int64_t` 位宽实例。 | 16 字节转源码 dormant key 104，但公开入口不可达；未知 key 无 body、无显式错误。 |

```text
if self.IsEmpty(): ACLNN success, no DiagFlat launch
else if N <= 64:
    if dtype_size != 16: key=101 -> R-001
    else: key=102 -> dormant for this ACLNN contract
else:
    H = ceil_align(max(ceil_div(N, C_total), 64), 64)
    if dtype_size != 16: key=103 -> R-002
    else: key=104 -> dormant for this ACLNN contract
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 小输入散写 | 101 | A2/A3；`1<=N<=64`；公开 dtype byte width `s∈{1,2,4,8}` | 单 `diag_flat` AIV 节点；所有 `C_total` 核清零，只有 block 0 写对角线 | `DiagFlatNDTo2DWithFew<T>::Process`；router 对 4/8 字节改用同位宽整数实例，其余用 `DTYPE_X` | input/output Queue depth 1；workspace 同步区；`SyncAll` 后标量 `GetValue/SetValue` 构造 tile | `tile=32/s`；最后 tile 用 DataCopyPad 编译分支或回退回滚分支；A2/A3 实际编译宏值固定仓未直接列出。 |
| R-002 64×64 向量块 | 103 | A2/A3；`N>64`；公开 dtype byte width `s∈{1,2,4,8}` | 单 `diag_flat` AIV 节点；所有核清零，前 `C_used` 核计算 | `DiagFlatNDTo2D<int8_t/int16_t/int32_t/int64_t>::Process` | input/output Queue + assist TBuf；`SyncAll`；把数据重解释为 int16 后以对角 `-1` mask 做 `And` | 每次固定 64 元素；尾核回退形成完整块；CopyOut 按行宽对齐/stride 上限选择二维 DMA、DataCopyPad 或逐行 DMA。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空输入 | `self->IsEmpty()` | 避免创建任何算子节点 | 全部 | ACLNN 成功、workspace 0；输出写入语义见限制 | `aclnn_diag_flat.cpp::aclnnDiagFlatGetWorkspaceSize` |
| 64 元素阈值 | `N<=SCALAR_THRESHOLD_NUM(64)` | 小输入直接在 UB 方块散写；大输入用固定 64×64 Vector tile | R-001/R-002 | 另一 Route | `diag_flat_tiling.cpp::TilingDiagFlat` |
| 16 字节 route | `dtypeSize==16` | 独立的双 `int64_t` body | 非公开 dormant key 102/104 | ACLNN dtype 检查先拒绝；不进入正式 Route | `CalcScalarTiling` / `CalcAuxMatrixTiling` |
| R-002 核粒度 | `H=ceil_align(max(ceil_div(N,C_total),64),64)` | 确保每核处理数与固定 tile 64 对齐 | R-002 | 无 fallback；平台核数/UB 读取失败时 Tiling 失败 | `CalcAuxMatrixTiling` |
| DataCopyPad 编译能力 | `__CCE_AICORE__==220 \|\| ==310 \|\| __NPU_ARCH__∈{3003,3113}` | 非对齐 GM 搬运与无 workspace 参数的 `SyncAll()` 可用性 | R-001/R-002 CopyOut 和清零同步 | 编译为逐行 DataCopy + `SyncAll(syncGlobal,workLocal)`；A2/A3 精确宏映射固定仓不可见 | `IsDataCopyPadSupport` |
| R-002 二维 DMA | `W*s % 32==0 && dstStride<65535` | `DataCopyParams` 的 stride 为 32 字节 block、`uint16_t` 宽度 | R-002 | 能力分支且 `s<8` 时 `DataCopyPad`；否则逐行 DataCopy | `DiagFlatNDTo2D::CopyOut` |
| R-001 输出 tile | `ONCE_HANDLE_NUM=32/s` | 一行正好一个 32B block；tile 为 `(32/s)^2` 元素 | R-001 | 不存在资源 fallback | `DiagFlatNDTo2DWithFew::Init` |
| workspace 非空 | `workspace==nullptr` | kernel 初始化同步区依赖 workspace | 全部 | entry 直接返回，不写输出，Host 不接收 Device 错误码 | `diag_flat` entry |

### POD/ABI

`DiagV2TilingData` 由固定仓中的 tiling-data 宏声明 18 个连续 `int64_t` 字段。字段顺序和类型可见；宏 `BEGIN_TILING_DATA_DEF/TILING_DATA_FIELD_DEF` 的实现与生成后的 `kernel_tiling.h` 不在固定仓，因此精确 `sizeof/alignof/offsetof` 统一标为固定仓不可见，不能仅按普通 C struct 推测。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `DiagV2TilingData` | `xWidth` | `int64_t`/64 | 声明顺序 0；精确布局固定仓不可见 | 元素 | DiagFlat Host 未赋值 | 活动 body 未读取 | DiagV2 遗留字段，inactive |
| `DiagV2TilingData` | `xHeight` | `int64_t`/64 | 声明顺序 1；精确布局固定仓不可见 | 元素 | 未赋值 | 未读取 | inactive |
| `DiagV2TilingData` | `gmOffset` | `int64_t`/64 | 声明顺序 2；精确布局固定仓不可见 | 元素 | 未赋值 | 未读取 | inactive |
| `DiagV2TilingData` | `numOut` | `int64_t`/64 | 声明顺序 3；精确布局固定仓不可见 | 元素 | 未赋值 | 未读取 | inactive |
| `DiagV2TilingData` | `realCoreNum` | `int64_t`/64 | 声明顺序 4；精确布局固定仓不可见 | core | 未赋值 | 未读取 | inactive |
| `DiagV2TilingData` | `numPerCore` | `int64_t`/64 | 声明顺序 5；精确布局固定仓不可见 | 元素/core | 未赋值 | 未读取 | inactive |
| `DiagV2TilingData` | `tailNum` | `int64_t`/64 | 声明顺序 6；精确布局固定仓不可见 | 元素 | 未赋值 | 未读取 | inactive |
| `DiagV2TilingData` | `tilingKey` | `int64_t`/64 | 声明顺序 7；精确布局固定仓不可见 | key | `CalcScalarTiling` 写 101/102；`CalcAuxMatrixTiling` 写 103/104 | body 不直接读取；entry 用框架 `TILING_KEY_IS` | 同值另传给 `context->SetTilingKey` |
| `DiagV2TilingData` | `matrixRowLength` | `int64_t`/64 | 声明顺序 8；精确布局固定仓不可见 | 元素 | 未赋值 | 未读取 | inactive |
| `DiagV2TilingData` | `inputNum` | `int64_t`/64 | 声明顺序 9；精确布局固定仓不可见 | 元素 | `inputShape.GetShapeSize()` | 两类 `ParseTilingData`；循环、宽度、地址 | 非空正式 Route 为正 |
| `DiagV2TilingData` | `totalCoreNum` | `int64_t`/64 | 声明顺序 10；精确布局固定仓不可见 | core | `DiagV2CompileInfo.totalCoreNum=GetCoreNumAiv()` | 清零分片、sync 区与 workQueue 容量 | `<=0` 时 TilingPrepare 失败 |
| `DiagV2TilingData` | `usedCoreNum` | `int64_t`/64 | 声明顺序 11；精确布局固定仓不可见 | core | R-001=1；R-002=`ceil_div(N,H)` | `Process` 早退、尾核识别 | 不大于 `totalCoreNum` |
| `DiagV2TilingData` | `normalCoreHandleNum` | `int64_t`/64 | 声明顺序 12；精确布局固定仓不可见 | 元素/core | R-001=`N`；R-002=`H` | `inputIdx=blockIdx*normalCoreHandleNum`、主核 loops | R-001 body 不读取 |
| `DiagV2TilingData` | `lastCoreHandleNum` | `int64_t`/64 | 声明顺序 13；精确布局固定仓不可见 | 元素 | R-001=0；R-002=`N-(C_used-1)H` | R-002 尾核回退/对齐 | R-001 body 不读取 |
| `DiagV2TilingData` | `ubInputSize` | `int64_t`/64 | 声明顺序 14；精确布局固定仓不可见 | 字节（Host 意图） | `UB_platform-2*64*64*sizeof(dataType)` | 活动 body 未读取 | Host 对 enum 变量取 `sizeof`；inactive |
| `DiagV2TilingData` | `ubOutputSize` | `int64_t`/64 | 声明顺序 15；精确布局固定仓不可见 | 字节（Host 意图） | `64*64*sizeof(dataType)` | 活动 body 未读取 | inactive |
| `DiagV2TilingData` | `diagonal` | `int64_t`/64 | 声明顺序 16；精确布局固定仓不可见 | 行/列偏移 | `*attrs->GetAttrPointer<int64_t>(0)` | `offset_`，宽度和 GM 地址公式 | `abs(INT64_MIN)` 未保护 |
| `DiagV2TilingData` | `workspaceSize` | `int64_t`/64 | 声明顺序 17；精确布局固定仓不可见 | 字节 | `32*C_total+16MiB` | 活动 body 未读取 | kernel 实际只绑定前 `32*C_total` 字节 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 小输入散写 | 101 | `conversion/diag_flat/op_host/diag_flat_tiling.cpp::TilingDiagFlat/CalcScalarTiling` | `DiagV2TilingData`：`N,d,C_total,C_used=1` | `conversion/diag_flat/op_kernel/diag_flat.cpp::diag_flat` key 101 | `diag_flat_nd_to_2d_with_few.h::DiagFlatNDTo2DWithFew<T>` | A2/A3 binary JSON 四种 DtypeByte 实例；`ascendc_config.json` 的 `DiagFlat` compute_units/auto_sync |
| R-002 64×64 向量块 | 103 | `conversion/diag_flat/op_host/diag_flat_tiling.cpp::TilingDiagFlat/CalcAuxMatrixTiling` | `DiagV2TilingData`：`N,d,C_total,C_used,H_normal,H_last` | `conversion/diag_flat/op_kernel/diag_flat.cpp::diag_flat` key 103 + byte宽 router | `diag_flat_nd_to_2d.h::DiagFlatNDTo2D<T>` | 同上 |

## Kernel 实现

### 公共执行流程

两条 Route 都由单 entry `diag_flat(input,output,workspace,tiling)` 执行：workspace 为空时静默返回；创建 `TPipe`，解析 POD/key，实例化 byte-width 等价的整数模板以保持 FP/复数的原始 bit pattern。每个 AIV block 在 `Init` 先把同步区和分配到的输出 GM 片段清零，再进行 `SyncAll`，保证任何核写对角线前整张输出已清零。之后绑定输入/输出 GM、初始化 depth=1 Queue；R-002 还构造 VECCALC 对角辅助矩阵。`Process` 对 `blockIdx>=C_used` 早退，活动核按 tile 执行 CopyIn→Compute→CopyOut。没有 SIMT、NDDMA、通信、Cube/AIC 或 MIX_CV 节点。

### Route R-001：小输入散写

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/diag_flat/op_host/op_api/aclnn_diag_flat.cpp` | `aclnnDiagFlatGetWorkspaceSize` | 非空、dtype/rank 校验通过 | 连续化后加入 DiagFlat |
| Tiling | `conversion/diag_flat/op_host/diag_flat_tiling.cpp` | `TilingDiagFlat` | `inputNum<=64` | 调用 `CalcScalarTiling` |
| Tiling | 同上 | `CalcScalarTiling` | `dtypeSize!=16` | key 101，`usedCoreNum=1` |
| Router | `conversion/diag_flat/op_kernel/diag_flat.cpp` | `diag_flat` | `TILING_KEY_IS(101)`；4B→`int32_t`，8B→`int64_t`，否则 `DTYPE_X` | `DiagFlatNDTo2DWithFew<T>::Init/Process` |

#### Device 数据流

##### 函数级调用链

```text
DiagFlatAiCore -> ADD_TO_LAUNCHER_LIST_AICORE(DiagFlat)
  -> TilingDiagFlat -> CalcScalarTiling -> key 101, blockDim=C_total
  -> diag_flat(input, output, workspace, tiling)
     -> DiagFlatNDTo2DWithFew<T>::Init
        -> ParseTilingData
        -> InitGm -> MemSetZero(sync) -> GmSetZeroPad/GmSetZero(output) -> SyncAll
        -> InitBuffer(inputQueue/outputQueue)
     -> Process
        -> CopyIn -> Compute(ClearOuput + diagonal SetValue) -> CopyOut/CopyOutTail
```

##### 源码等价伪码

```text
# 伪码；s=sizeof(T), B=32/s, W=N+abs(d)
all blocks:
  zero workspace[0 : 32*C_total]
  split W*W output elements across C_total, zero each assigned slice
  SyncAll()
  if blockIdx >= 1: return

loopSize = ceil_div(N, B)
for tile in [0, loopSize):
  if last tile and old-platform branch and N>B and N%B!=0:
      rollback = B-(N%B)
  else rollback = 0
  ubIn[0:B] = GM input[tile*B-rollback : +B]
  Duplicate ubOut[B,B] = 0
  for i in [0,B): ubOut[i,i] = ubIn[i]
  row/column base incorporates sign(d)
  normal tile: copy each of B rows, one 32B block per row
  tail:
    DataCopyPad only N%B valid rows with byte blockLen=(N%B)*s
    or old-platform rollback and B full row copies
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `tile*B-rollback` | 逻辑 `N`，单搬运 `B=32/s` 元素 | 原 dtype bit pattern | Contiguous | `CopyIn` | executor 临时/调用输入，kernel 全程 | 每次 32B；尾部能力分支仍固定搬入 B |
| GM output | base row `(d<=0?abs(d):0)`；元素 `(row*W)+(d>0?d:0)+col` | `W*W` 元素 | `T`, ND row-major | 清零 + `CopyOut` | Cast/ViewCopy | L0 临时，后继节点消费前有效 | 清零分片按 B 对齐；对角 tile 按行写 |
| workspace sync | byte 0 | `32*C_total` | `int32_t` | 所有 block `MemSetZero` | `SyncAll` fallback | executor workspace，barrier 后可不再读取 | 每核协议 32B；各核写同一全区为幂等零值 |
| `workQueue_` | UB allocator | `C_total*8*sizeof(int32_t)=32*C_total` bytes，depth 1 | int32 | `InitBuffer` | old-platform `SyncAll` | Init→barrier 后 free local tensor | A2/A3宏映射不可见；能力分支不分配 local tensor但 Queue 仍初始化 |
| `inputQueue_` | UB allocator | `B*s=32` bytes，depth 1 | `T[B]` | `CopyIn` | `Compute` | Alloc→EnQue→DeQue→Free | 固定 32B |
| `outputQueue_` | UB allocator | `B*B*s=1024/s` bytes，depth 1 | `T[B,B]` | `ClearOuput/Compute` | `CopyOut` | Alloc→EnQue→DeQue→Free | 每行 32B；同一 tensor 先清零后对角散写 |
| `PopStackBuffer` | LCM stack | primitive 返回容量；固定仓不可见 | int16 | `MemSetZero` Duplicate | workspace/GM 清零 DMA | 仅 `MemSetZero` | 容量决定分轮和 tail，非 TPipe 显式分配 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  workQ["workQueue · depth=1 · 32*C_total B"]:1
  inQ["inputQueue · depth=1 · 32 B"]:1
  outQ["outputQueue · depth=1 · 1024/s B"]:2
```

##### 字节账与复用顺序

```text
TPipe_UB = 32*C_total + 32 + 1024/s bytes
temporary_LCM = PopStackBuffer capacity（固定仓不可见，只在清零阶段存活）
UB_total = TPipe_UB + temporary_LCM
UB_total <= UB_platform：Host 没有按实际式做 IsCapable；能否满足由目标平台编译/运行资源保证，固定仓无显式失败分支
dynamic_ub = N/A（AIV Queue/TBuf 路径）
workspace_total_for_DiagFlat = 32*C_total + 16*1024*1024 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `outputQueue_` tensor | `ClearOuput` 先 Duplicate 零，再 `Compute` 对角 SetValue | `CopyOut/CopyOutTail` | 下一 tile `AllocTensor` | EnQue/DeQue/FreeTensor | depth 1，必须完成 CopyOut 后再覆盖 |
| `inputQueue_` tensor | `CopyIn` | `Compute::GetValue` | 下一 tile | EnQue/DeQue/FreeTensor | 单核串行 tile 循环 |
| output GM 清零区 | `InitGm` 各 block | 后续对角写默认零背景 | 不再整体覆盖 | `SyncAll` | 所有核必须参加 barrier，故 Process 早退在 barrier 之后 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全局清零→对角写 | `SyncAll()` 或 `SyncAll(syncGlobal,workLocal)` | 所有 block 的同步区/输出清零 | 各 block `Process` | 整张 output GM | 只搬运 bit pattern，无浮点运算；避免清零覆盖已写对角线 |
| GM→VEC | Queue EnQue/DeQue | `DataCopy` | `Compute` | inputQueue | 一次固定 32B；尾 tile 可能回退读前一段 |
| UB 清零→标量散写 | `Duplicate`，随后 `SetFlag/WaitFlag<V_S>` | Vector 清零 | Scalar `GetValue/SetValue` | outputQueue tile | 不做 cast/round；每个 T 值原位复制 |
| VEC→GM | Queue + 每行 `PipeBarrier<PIPE_MTE3>` | `Compute` | `CopyOut` DataCopy | outputQueue/GM row | 最后 tile 的能力分支只提交 `N%B` 行；回退分支写完整重叠块 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/diag_flat/op_host/diag_flat_tiling.cpp` | `TilingDiagFlat/CalcScalarTiling` | N 阈值、key 101、单计算核、总 AIV blockDim 和 workspace |
| Kernel entry/router | `conversion/diag_flat/op_kernel/diag_flat.cpp` | `diag_flat`, key 101 | dtype byte-width 到 `DiagFlatNDTo2DWithFew<T>` 的映射 |
| Device body/helper | `conversion/diag_flat/op_kernel/diag_flat_nd_to_2d_with_few.h` | `InitGm/Process/ClearOuput/CopyOutTail/MemSetZero` | 输出并行清零、barrier、32B tile、标量对角散写、tail 分支 |

### Route R-002：64×64 向量块

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/diag_flat/op_host/op_api/aclnn_diag_flat.cpp` | `aclnnDiagFlatGetWorkspaceSize` | 非空、dtype/rank 校验通过 | 连续化后加入 DiagFlat |
| Tiling | `conversion/diag_flat/op_host/diag_flat_tiling.cpp` | `TilingDiagFlat` | `inputNum>64` | 调用 `CalcAuxMatrixTiling` |
| Tiling | 同上 | `CalcAuxMatrixTiling` | `dtypeSize!=16`；`H=ceil_align(max(ceil_div(N,C_total),64),64)` | key 103、计算 `C_used/H_last` |
| Router | `conversion/diag_flat/op_kernel/diag_flat.cpp` | `diag_flat` | key 103；按 `sizeof(DTYPE_X)` 为 4/8/2/else | `DiagFlatNDTo2D<int32_t/int64_t/int16_t/int8_t>` |

#### Device 数据流

##### 函数级调用链

```text
DiagFlatAiCore -> ADD_TO_LAUNCHER_LIST_AICORE(DiagFlat)
  -> TilingDiagFlat -> CalcAuxMatrixTiling -> key 103, blockDim=C_total
  -> diag_flat(input, output, workspace, tiling)
     -> DiagFlatNDTo2D<T>::Init
        -> ParseTilingData (含尾核 rollback)
        -> InitGm -> MemSetZero(sync/output) -> ConstructAssistMatrix -> SyncAll
        -> InitBuffer(inputQueue/outputQueue)
     -> Process
        -> for each 64-element tile: CopyIn -> Compute(And) -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；s=sizeof(T), W=N+abs(d)
inputIdx = blockIdx*H_normal
if blockIdx == C_used-1 and H_last%64 != 0:
    inputIdx -= 64-(H_last%64)
curHandle = tail ? ceil_align(H_last,64) : H_normal

all blocks:
  zero workspace[0 : 32*C_total]
  split W*W output into ceil_align(...,64) chunks and zero assigned chunk
  build UB assist[64,64]: all zero, diagonal raw bits = all ones
  SyncAll()
  if blockIdx >= C_used: return

for tile in [0, curHandle/64):
  ubIn[0:64] = input[inputIdx + tile*64 : +64]
  reinterpret ubIn/ubOut/assist as int16 lanes
  for each <=128-lane segment:
      And(dst, broadcast(src row via src0RepStride=0), diagonal-mask row, mask, repeat=64)
  gmBase = (inputIdx + (d>0?0:abs(d)))*W
  if W*s is 32B aligned and dstStride<65535:
      one 64-row strided DataCopy
  else if DataCopyPad-capable and s<8:
      one byte-stride DataCopyPad
  else:
      64 one-row DataCopy operations
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input | `inputIdx+tile*64` | 逻辑 N；单次 64 元素 | 原 dtype bit pattern | Contiguous | `CopyIn` | executor 输入/临时，kernel 全程 | 尾核通过回退保证完整 64 元素 |
| GM output | base `(inputIdx+(d>0?0:abs(d)))*W`，tile 内 `row*W+(d>0?d:0)+column` | `W*W` 元素 | `T`, ND row-major | 清零 + `CopyOut` | Cast/ViewCopy | L0 临时 | 清零尾片可向前回退到 64 元素边界；对角写可能在核间重叠但值相同 |
| workspace sync | byte 0 | `32*C_total` | int32 | `MemSetZero` | `SyncAll` fallback | executor workspace | 同 R-001 |
| `workQueue_` | UB allocator | `32*C_total` bytes，depth 1 | int32 | `InitBuffer` | fallback barrier | Init→barrier | local tensor barrier 后释放 |
| `inputQueue_` | UB allocator | `64*s` bytes，depth 1 | `T[64]` | `CopyIn` | `Compute` | 每 tile Alloc→Free | 64 元素 |
| `outputQueue_` | UB allocator | `4096*s` bytes，depth 1 | `T[64,64]` | `And` | `CopyOut` | 每 tile Alloc→Free | 64 行，每行 `64*s` bytes |
| `assistBuf_` | UB allocator | `4096*s` bytes | `T[64,64]`/重解释 int16 | `ConstructAssistMatrix` | 每次 `Compute` | Init 后只读至 kernel 结束 | 零矩阵对角为全 1 bit mask，跨 tile 复用 |
| `PopStackBuffer` | LCM stack | 容量固定仓不可见 | int16 | `MemSetZero` | 清零 DMA | 仅初始化 | 同 R-001 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  workQ["workQueue · depth=1 · 32*C_total B"]:1
  inQ["inputQueue · depth=1 · 64*s B"]:1
  assist["assistBuf · 4096*s B"]:1
  outQ["outputQueue · depth=1 · 4096*s B"]:1
```

##### 字节账与复用顺序

```text
TPipe_UB = 32*C_total + 64*s + 4096*s + 4096*s
         = 32*C_total + 8256*s bytes
temporary_LCM = PopStackBuffer capacity（固定仓不可见，只在清零阶段存活）
UB_total = TPipe_UB + temporary_LCM
UB_total <= UB_platform：Host 的 ubInputSize/ubOutputSize 不参与 body 分配，也没有实际 IsCapable fallback
dynamic_ub = N/A（AIV Queue/TBuf 路径）
workspace_total_for_DiagFlat = 32*C_total + 16*1024*1024 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `assistBuf_` | `ConstructAssistMatrix` | 所有 tile 的 `And` src1 | kernel 结束 | 初始化事件 `MTE3_V`、标量写前 `V_S`，之后只读 | 任何 tile 不得原地修改 mask |
| `inputQueue_` | `CopyIn` | 当前 tile `And` src0 | 下一 tile | EnQue/DeQue/FreeTensor | `src0RepStride=0` 广播当前输入块，不跨 tile |
| `outputQueue_` | `And` | `CopyOut` | 下一 tile | EnQue/DeQue/FreeTensor | depth 1，GM 提交后再复用 |
| 尾核输入/输出覆盖域 | 尾核回退的完整 64 块 | CopyIn/CopyOut | 无 | 不使用 atomic；写入值与前核相同 | 仅回退到仍在 `[0,N)` 的最后完整 64 元素块 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全局清零→计算 | `SyncAll` | 所有 block 清零 | `Process` | output GM | 防止对角线写被其他核清零覆盖 |
| assist 初始化 | `SetFlag/WaitFlag<MTE3_V>` 与 `<V_S>` | UB 分配/Vector Duplicate | 标量写 `-1` 对角 | assistBuf | mask 的全 1 bit pattern 保留所有输入位；无数值计算 |
| CopyIn→And→CopyOut | Queue + `SetFlag/WaitFlag<S_V>` | DMA/标量参数 | Vector `And` | input/output Queue | 将任意 dtype 重解释成 int16，仅做 bitwise 复制/置零，无 cast/round |
| 尾核 | 地址 rollback | 前核与尾核 | 各自 GM 写 | 最后 64 元素 | 重叠写相同 bit pattern；固定仓未提供 race 检测或 atomic |
| CopyOut | 二维 DataCopy/DataCopyPad/逐行 DataCopy | outputQueue | GM | 64×64 tile | `blockLen=64*s/32` blocks，`dstStride=W*s/32-blockLen`；非对齐分支使用字节 stride |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/diag_flat/op_host/diag_flat_tiling.cpp` | `CalcAuxMatrixTiling/TilingDiagFlat` | 64 对齐分核、key 103、workspace、ScheduleMode |
| Kernel entry/router | `conversion/diag_flat/op_kernel/diag_flat.cpp` | `diag_flat`, key 103 | DTYPE_X byte-width 到整数位宽模板 |
| Device body/helper | `conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h` | `ParseTilingData/ConstructAssistMatrix/Compute/CopyOut` | 尾核回退、辅助矩阵、int16 And 参数和三种 CopyOut 策略 |

## 算子特定优化

### Trick T-001：用全 1 对角 mask 将散写转为 64 行 Vector And

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：公开 dtype byte width 非 16、`N>64` 命中 key 103；每核/尾核均形成完整 64 元素 tile。
- Golden 来源：[`conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h:181-202`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h#L181-L202)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    uint64_t countOfInt16 = ONCE_HANDLE_NUM * sizeof(T) / sizeof(int16_t);
    constexpr uint64_t INT16_MAX_MASK = 128;
    uint64_t loops = (countOfInt16 + INT16_MAX_MASK - 1) / INT16_MAX_MASK;
    uint64_t mask = countOfInt16 > INT16_MAX_MASK ? INT16_MAX_MASK : countOfInt16;
    uint8_t repeatTime = ONCE_HANDLE_NUM;
    uint8_t dstBlkStride = 1;
    uint8_t src0BlkStride = 1;
    uint8_t src1BlkStride = 1;
    uint8_t dstRepStride = ONCE_HANDLE_NUM * sizeof(T) / ONE_BLK_SIZE;
    uint8_t src0RepStride = 0;
    uint8_t src1RepStride = ONCE_HANDLE_NUM * sizeof(T) / ONE_BLK_SIZE;
    BinaryRepeatParams params = {dstBlkStride, src0BlkStride, src1BlkStride,
                                 dstRepStride, src0RepStride, src1RepStride};

    LocalTensor<int16_t> ubInputCast = ubInput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> ubOutputCast = ubOutput.template ReinterpretCast<int16_t>();
    LocalTensor<int16_t> ubAssistCast = ubAssist.template ReinterpretCast<int16_t>();
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for (int i = 0; i < loops; i++) {
        And(ubOutputCast[mask * i], ubInputCast[mask * i], ubAssistCast[mask * i], mask, repeatTime, params);
    }
```

#### 机制与设计原因

`ConstructAssistMatrix` 预先构造 64×64 mask：非对角线为 0、对角线为与 T 等宽的全 1。`Compute` 把输入、输出、mask 都重解释为 int16 lane，`src0RepStride=0` 使每次 repeat 复用同一输入行段，`src1RepStride=dstRepStride=64*s/32` 逐行推进；64 次 repeat 一次产生 64×64 对角 tile。这样把逐元素 scalar GM scatter 改为 UB 内 Vector 运算和成块/跨行 DMA，且对浮点、整数、复数都只搬运 bit pattern，不引入数值 cast。

#### 迁移边界

- 适用：把一维块嵌入固定大小二维稀疏模式，且填充值可由 bit mask 选择的 Vector kernel。
- 前提：tile 固定 64；Queue/TBuf 能容纳 `64*s + 2*4096*s` 字节；目标 dtype 可安全按 int16 lane 重解释；Host 必须把尾块回退为完整 64 元素。
- 失效：`N<=64` 改用 R-001；16 字节 dtype 需源码 key 102/104 的专用双 int64 实现，但该 dtype 不在本 ACLNN 公开合同内；UB 不足时固定源码没有 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。迁移时若把 `src0RepStride` 从 0 改为逐行推进，将不再把每个输入元素扩展到对应 mask 行；若辅助矩阵对角不是全 1 bit pattern，浮点/复数原始位也会被破坏，这是由当前源码参数关系直接决定的约束，不是实测性能结论。

## 模板与实现对比

| 对比维度 | R-001 小输入散写 | R-002 64×64 向量块 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `1<=N<=64`，key 101 | `N>64`，key 103 | Host 以 64 为唯一 shape 阈值；16B key 不属于公开 ACLNN dtype |
| 模板与 dtype | `WithFew<T>`，B=`32/s` | `NDTo2D<T>`，T 为同 byte-width 整数 | 都保持原始 bit pattern；大输入固定 64 元素并行 |
| 循环与分核 | 所有核清零，仅 block0；`ceil_div(N,32/s)` tile | `C_used=ceil_div(N,H)`；每核 `H/64` tile；尾核回退 | 小输入避免 64×64 大辅助矩阵；大输入扩展 AIV 并行度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | input 32B，output `1024/s`B，无 assist | input `64s`B，output/assist 各 `4096s`B | 均无 dynamic UB/通信；R-002 用容量换 Vector 化 |
| 内存总占用与复用 | `32C+32+1024/s + LCM临时` | `32C+8256s + LCM临时` | LCM PopStack 容量固定仓不可见 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy、Duplicate、scalar Get/Set、DataCopyPad | DataCopy、Duplicate、And、二维 DMA/DataCopyPad | 均为 AIV_VECTOR，无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | SyncAll；bit-copy；B 尾 tile pad/rollback | SyncAll；int16 bitwise And；64 元素尾核 rollback | 先全局清零再写对角线；无 FP 算术/round |
| fallback/失败边界 | DataCopyPad 编译 guard；未知 key 无动作 | CopyOut 三分支；UB 无 IsCapable fallback | workspace null 静默返回；Tiling 平台资源读取失败返回失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档支持；映射 `ascend910b` | 公开 ACLNN 文档支持；映射 `ascend910_93` | 公开 ACLNN 文档明确不支持；通用 README、OpDef、binary config 又接入 950，发布证据冲突，不生成 950 卡 |
| Host/Launcher | 共用 API/L0/Tiling；binary JSON 提供 1/2/4/8-byte DtypeByte 实例 | 与 A2 共用实现和等价 binary JSON | `IsRegBase()` 会选择含 BF16 的 dtype 表，OpDef/AddConfig 和构建表存在，但不能覆盖公开“不支持” |
| Kernel/资源 | `GetCoreNumAiv/GetCoreMemSize(UB)` 运行时取平台值；同一 key 101/103 body | 同 A2；`C_total/UB_platform` 随平台实际值变化 | 实现文件存在且 compute_units 含 ascend950；因 ACLNN 冲突不发布 |

## 限制与失败边界

- 公开文档要求 `out` 与 `self` dtype 一致、shape 为语义输出；实现只做支持集合和最大 rank 检查，没有 dtype 相等或输出 shape 等式检查。超出文档合同的 Cast/ViewCopy 行为不作为稳定语义承诺。
- ACLNN dtype 检查表与文档包含 BOOL，但 `diag_flat_def.cpp` 的 OpDef dtype 集合不含 BOOL；BOOL 路线存在注册契约冲突，固定仓没有消解该冲突的代码。
- 空 `self` 在第一段直接返回成功且不创建清零节点；当 `abs(diagonal)>0` 时语义 shape 非空，固定源码未显示由谁写零。
- `abs(INT64_MIN)`、`N+abs(d)`、`W*W`、workspace `32*C_total` 与各种乘法没有显式溢出保护；POD 全为 int64，但部分 DMA 参数窄化为 uint8/uint16。
- R-002 实际 UB 分配不消费 Host 写入的 `ubInputSize/ubOutputSize`，没有 UB IsCapable/fallback；`PopStackBuffer` 容量、tiling-data 宏最终 ABI、A2/A3 的精确编译宏值均为固定仓不可见。
- entry 遇到 `workspace==nullptr` 直接返回，无 Device 状态回传；unknown key 也没有显式错误分支。
- 950 的公开文档与实现接线冲突，所以 `diag_flat_950.md` 不得创建。源码 key 102/104 的 16B body 对公开 ACLNN 不可达，未纳入 Route 正文。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/diag_flat/docs/aclnnDiagFlat.md` | 产品支持表、函数原型 | A2/A3 支持、950 不支持、公开参数与两段式调用 |
| 入口声明/约束 | `conversion/diag_flat/README.md` | 产品支持情况 | 通用算子层 950 支持信号及 dtype/语义 |
| 入口声明/约束 | `conversion/diag_flat/op_host/op_api/aclnn_diag_flat.h` | `aclnnDiagFlat*` | C ABI 声明 |
| 入口实现 | `conversion/diag_flat/op_host/op_api/aclnn_diag_flat.cpp` | `CheckParams/aclnnDiagFlatGetWorkspaceSize/aclnnDiagFlat` | dtype/rank、空输入、Contiguous→DiagFlat→Cast→ViewCopy、executor |
| Host/Launcher | `conversion/diag_flat/op_host/op_api/diag_flat.cpp` | `DiagFlat/DiagFlatAiCore` | 输出 shape 与 AICore launcher |
| Host/Launcher | `conversion/diag_flat/op_host/diag_flat_def.cpp` | `ops::DiagFlat` | OpDef dtype 和 A2/A3/950 config，BOOL 冲突 |
| Host/Tiling/Launcher | `conversion/diag_flat/op_host/diag_flat_tiling.cpp` | `TilingDiagFlat` | 分核、key、workspace、blockDim、ScheduleMode |
| POD/参数 | `conversion/diag_v2/op_host/diag_v2_tiling.h` | `DiagV2TilingData/DiagV2CompileInfo` | POD 字段、常量、key 定义 |
| Kernel entry/body | `conversion/diag_flat/op_kernel/diag_flat.cpp` | `diag_flat` | key router 与 dtype byte-width 模板 |
| Kernel entry/body | `conversion/diag_flat/op_kernel/diag_flat_nd_to_2d_with_few.h` | `DiagFlatNDTo2DWithFew` | R-001 完整 body |
| Kernel entry/body | `conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h` | `DiagFlatNDTo2D` | R-002 完整 body |
| 构建接线 | `conversion/diag_flat/op_host/config/ascend910b/diag_flat_binary.json` | `DiagFlat op_list` | A2 DtypeByte binary 实例 |
| 构建接线 | `conversion/diag_flat/op_host/config/ascend910_93/diag_flat_binary.json` | `DiagFlat op_list` | A3 DtypeByte binary 实例 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `DiagFlat` | A2/A3/950 compute_units 与 auto_sync |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 可复用 18 字段声明顺序、活动字段 producer/consumer | 不能假定普通 struct offset；inactive 遗留字段仍在序列中；key 同时写 POD 与 context | `DiagV2TilingData`, `DiagFlatSetTilingData` |
| 调用顺序 | Contiguous→DiagFlat→Cast→ViewCopy；kernel Init 清零→SyncAll→Process | 不得把 Process 早退移到 SyncAll 前；后处理必须等临时输出可见 | `aclnnDiagFlatGetWorkspaceSize`, `InitGm` |
| workspace/内存 | 前 `32*C_total` 字节为 sync；R-001/R-002 Queue/TBuf 公式可直接移植 | 保留 16MiB 系统 workspace、owner/lifetime；PopStack 临时容量外部决定 | `TilingDiagFlat`, `MemSetZero`, `InitBuffer` |
| 同步 | ScheduleMode 1、auto_sync、SyncAll 和 Queue 生命周期 | 整张输出清零完成前不得写对角线；编译能力分支的两种 SyncAll ABI 不得混用 | `TilingDiagFlat`, `IsDataCopyPadSupport`, `InitGm` |
| 精度与尾块 | 以同位宽整数/`int16` 重解释并复制 bit pattern；R-002 尾核回退 | 不得引入数值 cast；mask 对角必须全 1；rollback 必须保持完整 64 元素在界内 | `diag_flat` router, `ConstructAssistMatrix`, `Compute`, `ParseTilingData` |
| 失败边界 | 平台核数/UB 读取失败、节点组图失败和 API 前置错误可复用 | workspace null/unknown key 是静默终点；空输入、BOOL 和 950 冲突必须显式处理 | `TilingPrepareForDiagFlat`, `diag_flat`, `CheckParams` |
