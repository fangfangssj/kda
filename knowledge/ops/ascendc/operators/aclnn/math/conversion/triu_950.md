---
type: Operator
title: "Triu 950 ACLNN 算子知识"
description: "说明 aclnnTriu 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/docs/aclnnTriu&aclnnInplaceTriu.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_api/aclnn_triu.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_api/aclnn_triu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_api/triu.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/triu_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triu_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triu_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/triu_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/triangulator_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_all.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_tiny_shape.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_medium_shape.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_row_and_col.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/config/ascend950/triu_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T09:51:13Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Triu 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnTriuGetWorkspaceSize` + `aclnnTriu`，公开 ACLNN 两段式接口，无带版本后缀的新入口。 |
| 计算语义 | 对最后两维的每个 `(i,j)`，`i + diagonal <= j` 时保留 `self`，否则置零；前导维视为批量维。 |
| 输入输出 | `self -> out`，shape/dtype/storage format 一致；950 公开列表为 1/2/4/8-byte 的整数、FP16/BF16/FP32/FP64、BOOL 和 COMPLEX32/64，Host 传入 Kernel 时按字节宽度分组。 |
| 执行类型 | 算子自有节点为单 `AIV_VECTOR` Kernel `triu`；ACLNN 前后的 `Contiguous`/`ViewCopy` 为固定仓不可见的框架边界。 |
| Kernel 路线 | R-001 全零；R-002 原样拷贝；R-003 tiny 预计算 mask；R-004 medium 整矩阵批处理；R-005 normal 行列分块。 |
| 核心资源 | GM + AIV UB（`TBuf`/`TQue`/`TQueBind`）+ MicroAPI 寄存器；Tiling 为算子节点申报 16 MiB workspace，Kernel 只用它做非空门控。 |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 launcher→950 OpDef/Tiling/POD→key router→五个 AIV body→950 构建配线均在固定 Commit 闭合；框架 executor/Contiguous/ViewCopy 按允许边界标记。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设输入 shape 最后两维为 `row=R` 和 `col=C`，所有前导维之积为 `M`，对 `m∈[0,M)`、`i∈[0,R)`、`j∈[0,C)`：

```text
out[m,i,j] = self[m,i,j],  if j >= i + diagonal
out[m,i,j] = 0,            otherwise
```

这是非原地 `Triu`：ACLNN 要求独立 `out`，不把 `aclnnInplaceTriu` 的别名/生命周期语义合并进本卡。所有 Route 都只做原始 bit-pattern 拷贝、生成全零 bit-pattern 和 mask 选择，不对浮点/复数做算术。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 公开文档列 ND/NCHW/NHWC/HWCN/NDHWC/NCDHW；API 拒绝 private storage format，内部 `Contiguous` 生成连续 storage | UINT8/INT8/UINT16/INT16/UINT32/INT32/UINT64/INT64/FLOAT16/BFLOAT16/FLOAT/DOUBLE/BOOL/COMPLEX32/COMPLEX64 | 非空输入 rank 为 2–8；空 Tensor 在 API 层直接成功返回，不启动 Triu Kernel。 |
| `diagonal` | scalar ACLNN attribute | `int64_t` | 默认语义为 0；L0 用 `OP_ATTR(diagonal)` 传给 Host Tiling。 |
| `out` | shape/storage format 必须与 `self` 相同 | 必须与 `self` 相同 | `ViewCopy(triuResult, out)` 把连续中间结果写入用户指定 view/storage。 |
| `workspaceSize` | Host 标量出参 | `uint64_t` | 来自 executor 全图 workspace 汇总；其中 Triu 节点的 Tiling 固定申报 16 MiB。 |
| `executor` | Host 执行图 | `aclOpExecutor**` | 由第一段接口创建并移交，第二段在指定 stream 上执行。 |

### 入口与版本

固定 Commit 只声明无版本后缀的 `aclnnTriuGetWorkspaceSize`/`aclnnTriu`，未出现 `V2` 等更新公开入口。`aclnnTriuGetWorkspaceSize` 进入 `doTriuGetWorkspaceSize`，而 `aclnnTriu` 调用 `CommonOpExecutorRun`。950 公开支持由文档的 Ascend 950PR/950DT 表格、`Triu::AICore().AddConfig("ascend950", ...)`、`conversion/triu/CMakeLists.txt` 的 `SUPPORT_COMPUTE_UNIT "ascend950"` 及 `ascend950/triu_binary.json` 交叉支撑。

## Host/Launcher 执行流程

### 参数校验与前处理

`doTriuGetWorkspaceSize` 按如下顺序建图：

1. `CREATE_EXECUTOR()`，失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
2. `CheckNotNull`；然后按 `IsRegBase()` 选 `DTYPE_SUPPORT_LIST_REGBASE`，校验 input/output dtype 相同。
3. `CheckFormat` 拒绝 private storage format，并要求 `self/out` storage format 相同。
4. `CheckShape` 要求 rank≤8；只有 `self->Size()>0` 时才要求 rank≥2；始终要求 `self/out` shape 相同。
5. `self->IsEmpty()` 则直接返回 executor workspace，不添加 `Contiguous/Triu/ViewCopy`。
6. 否则依次调用 `l0op::Contiguous(self)`、`l0op::Triu(selfContiguous, diagonal)`、`l0op::ViewCopy(triuResult, out)`；任一返回空指针即终止建图。

### 参数、Workspace 与生命周期

L0 `Triu` 按 `self` storage shape/dtype/format 由 executor 分配连续中间 `out`，再用 `ADD_TO_LAUNCHER_LIST_AICORE(Triu, op::AI_CORE, OP_INPUT(self), OP_ATTR(diagonal), OP_OUTPUT(out))` 登记节点。Tiling `PostTiling` 把 `workspace[0]` 设为 `16,777,216` 字节。Kernel entry 调用 `GetUserWorkspace(workspace)` 并在返回空指针时直接退出，但五个 body 都不读写 `userWS`；因此这 16 MiB 在固定实现中是启动门控所需的保留 slice，不是算法中间张量。

ACLNN 返回的总 `workspaceSize` 还可包含框架 `Contiguous/ViewCopy` 需求，executor 的汇总/对齐算法在固定仓不可见。workspace 由调用者在第二段调用期间持有；executor 保存中间 tensor 和 launcher 的生命周期。

### 启动图与失败传播

```text
aclnnTriuGetWorkspaceSize
  -> [framework boundary] l0op::Contiguous(self, executor)
  -> l0op::Triu -> ADD_TO_LAUNCHER_LIST_AICORE(Triu, AI_CORE, selfContiguous, diagonal, triuResult)
       -> Tiling4Trilu<true,false> -> TriangulatorTiling::RunTriangulatorTilingAscendC
       -> triu<<<blockDim=usedCoreNum>>>(x, y, workspace, tiling)  # AIV_VECTOR, single owned Kernel
  -> [framework boundary] l0op::ViewCopy(triuResult, out, executor)

aclnnTriu(workspace, workspaceSize, executor, stream)
  -> CommonOpExecutorRun(..., stream)  # executor 内顺序为数据依赖顺序
```

| 节点/边 | 参数与介质 | ordering / owner / lifetime | 失败传播 |
|---|---|---|---|
| Contiguous→Triu | executor 中的连续 `selfContiguous`，shape/dtype 不变 | executor 持有；同一执行图数据依赖 | `Contiguous==nullptr` 时不登记 Triu |
| Triu 节点 | input/output GM；POD 依 Route 变化；`blockDim=usedCoreNum`；workspace slice=16 MiB | 单 AIV Kernel；输出在 Kernel 完成后才可被 ViewCopy 消费 | Host Tiling/POD capacity 失败阻断 launch；entry workspace 空时无声返回 |
| Triu→ViewCopy | executor 中的连续 `triuResult` GM | executor 持有到 ViewCopy 消费完成 | `Triu` launcher 登记失败或 `ViewCopy==nullptr` 时第一段返错 |
| 空 Tensor | 无 Device 节点 | executor 立即移交 | API 成功返回 |

## Route 与参数设计

### 分核与资源计算

Tiling 从 origin shape 得到 `row=dim[-2]`、`col=dim[-1]`、`fusedFrontAxis=Πdim[0:-2]`，从 dtype 得到 `dtypeBytes∈{1,2,4,8}`，从平台得到 AIV 核数 `C` 与 UB 容量 `UB_hw`，然后使用 `UB=UB_hw-8192` 和运行时 `V=GetVRegSize(context)`。逐 Route 参数如下：

- R-001/R-002：`N=fusedFrontAxis*row*col`。若 `N*dtypeBytes<1024`，`usedCoreNum=1`；否则 `normal=max(ceil(N/C), floor(1024/dtypeBytes))`，`used=ceil(N/normal)`，`tail=N-normal*(used-1)`。`bufferSize=floorAlign(UB/2,32)`。
- R-003：`bufferSize=floorAlign(UB/4,32)`；`highInner` 使每个 block 包含整数个矩阵且单矩阵不超过一个 VReg；`baseBlockNum=highOuter`。
- R-004：`bufferSize=floorAlign(UB/4,32)`；每行 UB stride 按 `min(V/dtypeBytes, aligned col)` 对齐，`highInner=floor((bufferSize/dtypeBytes)/(row*alignedCol))`，再把前导矩阵分成 `highOuter` 个 block。
- R-005：`bufferSize=floorAlign(UB/8,32)`；根据 `col<=V/dtypeBytes`、`row<=128` 的优先级计算 `rowInner/colInner`，`baseBlockNum=fusedFrontAxis*ceil(row/rowInner)*ceil(col/colInner)`。分母 8 是保守上界，Device 实际分配 `2+2+2+1=7` 份 `bufferSize`。

R-003/R-004/R-005 通用分核为 `used=min(baseBlockNum,C)`、`normal=floor(baseBlockNum/used)`、`remainder=baseBlockNum%used`；Device 使前 `remainder` 个核处理 `normal+1` 个 block，其余核处理 `normal` 个。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 全零 | `1/2/4/8`; `SplitAll<T,OUTPUT_ZERO_MODE>` | Triu 且 `diagonal >= col` | R-002 |
| 2 | R-002 原样拷贝 | `11/12/14/18`; `SplitAll<T,OUTPUT_INPUT_MODE>` | `diagonal <= 1-row` | R-003 |
| 3 | R-003 tiny mask | `31/32/34/38`; `SplitTinyShape<T>` | `row*col*dtypeBytes <= V` | R-004 |
| 4 | R-004 medium 整矩阵 | `41/42/44/48`; `SplitMediumShape<T,false>` | `row*CeilAlign(col*dtypeBytes,V)*4 <= UB` | R-005 |
| 5 | R-005 normal 分块 | `21/22/24/28`; `SplitRowAndCol<T,false>` | 以上均不命中 | Host 无其它 fallback |

```text
if diagonal >= col: mode=0, key=0*10+dtypeBytes
else if diagonal <= 1-row: mode=1, key=1*10+dtypeBytes
else if row*col*dtypeBytes <= V: mode=3, key=3*10+dtypeBytes
else if row*CeilAlign(col*dtypeBytes,V)*4 <= UB: mode=4, key=4*10+dtypeBytes
else: mode=2, key=2*10+dtypeBytes
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1/2/4/8 | 950 + supported dtype byte width + `d>=C` | 单 `triu` AIV | `SplitAll<int{8,16,32,64}_t,0>::Process` | 单 TBuf，一次 Duplicate 后反复 MTE3 | 末 loop 用 `tailLoopBlockNum` |
| R-002 | 11/12/14/18 | 950 + supported dtype byte width + `d<=1-R` | 单 `triu` AIV | `SplitAll<int{8,16,32,64}_t,1>::Process` | 双 TBuf ping-pong，MTE2↔MTE3 event | 末 loop 精确字节长 |
| R-003 | 31/32/34/38 | 950 + interior diagonal + plane bytes≤V | 单 `triu` AIV | `SplitTinyShape<int{8,16,32,64}_t>::Process` | 2-depth input/output queue + 32B mask TBuf，MicroAPI mask | `highTail`，DMA blockCount 只覆盖有效矩阵 |
| R-004 | 41/42/44/48 | 950 + interior + non-tiny + 4-buffer whole-matrix UB guard | 单 `triu` AIV | `SplitMediumShape<T,false>::Process` | 2-depth input/output queue，行对齐批量 DMA + Select | `highTail`，每行 rowMask |
| R-005 | 21/22/24/28 | 950 + 其余内部对角线 | 单 `triu` AIV | `SplitRowAndCol<T,false>::Process` | input/output/inOut 双深队列 + zero TBuf；整块 copy/zero 短路 | row/col tail + `rowMask/selectMask` |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 极值对角线 | `d>=C` / `d<=1-R` | 输出可化简为全零/全 copy | R-001/R-002 | tiny→medium→normal | `TriangulatorTiling::ComputeTilingMode` |
| VReg 整平面 | `R*C*dtypeBytes<=V` | 一个矩阵 mask 能在一次 VReg 操作内表达 | R-003 | R-004 | `ComputeTilingMode`, `ComputeMask` |
| 4-buffer 整矩阵 | `R*CeilAlign(C*dtypeBytes,V)*4<=UB_hw-8192` | medium 的双深 in/out queue 总计四份 buffer | R-004 | R-005 | `ComputeTilingMode`, `ComputeMediumInfo` |
| normal UB 保守分母 | `bufferSize=floorAlign((UB_hw-8192)/8,32)` | Device 分配 7 份，Host 保留 1 份余量 | R-005 | 无 fallback；Tiling 算术必须产生正的 inner | `ComputeNormalInfo`, `SplitRowAndCol::Init` |
| DMA 字段窄化 | `blockCount/highActual*row` 强转 `uint16_t`；`blockLen/stride` 强转 `uint32_t` | `DataCopyExtParams` ABI 宽度 | R-001–R-005 | 固定 Host 未显式做溢出 fallback | 各 `CopyIn/CopyOut/ProcessPerLoop` |
| key byte-width | `key=mode*10+dtypeBytes`，`dtypeBytes∈{1,2,4,8}` | router 只实例化四种 raw integer width | 全部 | unknown key 在 entry 中无分支、无输出 | `ComputeTilingKey`, `triangulatorEntry` |
| workspace 非空 | `GetUserWorkspace(workspace)!=nullptr` | entry 在 router 前做门控 | 全部 | 直接 return | `triangulatorEntry` |

### POD/ABI

`BEGIN_TILING_DATA_DEF`/`TILING_DATA_FIELD_DEF` 的宏展开定义不在固定仓，所以表中只能确认声明顺序、字段类型和 Host/Device 同名结构；精确 `sizeof/alignof/offsetof` 标为固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `TriangulatorTilingData` | `bufferSize` | `int64_t/64` | 第1字段；精确 offset/size/align 不可见 | 字节 | `SaveToTilingData: baseInfoOp.bufferSize` | `SplitAll::ParseTilingData` | zero route 再乘 2；copy route 原值 |
| 同上 | `usedCoreNum` | `int64_t/64` | 第2字段；精确布局不可见 | core | `splitCoreOp.usedCoreNum` | `SplitAll::Init` | `blockIdx>=used` 返回 |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第3字段；精确布局不可见 | element | `splitCoreOp.normalCoreProcessNum` | `SplitAll::Init` | 非末核数据量 |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第4字段；精确布局不可见 | element | `splitCoreOp.tailCoreProcessNum` | `SplitAll::Init` | 末核数据量 |
| `TriangulatorNormalTilingData` | `bufferSize` | `int64_t/64` | 第1字段；精确布局不可见 | 字节 | `baseInfoOp.bufferSize` | `SplitRowAndCol::Init` | 每个 queue slot/TBuf 容量 |
| 同上 | `row` | `int64_t/64` | 第2字段；精确布局不可见 | element | `baseInfoOp.row` | `ParseTilingData` | 最后第2维 |
| 同上 | `col` | `int64_t/64` | 第3字段；精确布局不可见 | element | `baseInfoOp.col` | `ParseTilingData`, DMA stride | 最后维 |
| 同上 | `diagOffset` | `int64_t/64` | 第4字段；精确布局不可见 | column | `baseInfoOp.diagOffset` | `ComputeBlockInfo` | 有符号 |
| 同上 | `usedCoreNum` | `int64_t/64` | 第5字段；精确布局不可见 | core | `splitCoreOp.usedCoreNum` | `Init/Process` | 超出核返回 |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第6字段；精确布局不可见 | tile | `splitCoreOp.normalCoreProcessNum` | `Process` | 基准 tile 数 |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第7字段；精确布局不可见 | core-count remainder | `splitCoreOp.tailCoreProcessNum` | `Process` | 前该数个核多 1 tile |
| 同上 | `rowInner` | `int64_t/64` | 第8字段；精确布局不可见 | row | `splitCoreOp.rowInner` | `ParseTilingData` | 末 tile 用 `rowTail` |
| 同上 | `colInner` | `int64_t/64` | 第9字段；精确布局不可见 | column | `splitCoreOp.colInner` | `ParseTilingData`, `regCount` | 末 tile 用 `colTail` |
| `TriangulatorTinyTilingData` | `bufferSize` | `int64_t/64` | 第1字段；精确布局不可见 | 字节 | `baseInfoOp.bufferSize` | queue `InitBuffer` | 每 slot |
| 同上 | `planeArea` | `int64_t/64` | 第2字段；精确布局不可见 | element/matrix | `row*col` | DMA blockLen/GM offset | plane bytes≤V |
| 同上 | `usedCoreNum` | `int64_t/64` | 第3字段；精确布局不可见 | core | split result | `Init/Process` | 超出核返回 |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第4字段；精确布局不可见 | block | split result | `Process` | 基准 block 数 |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第5字段；精确布局不可见 | core-count remainder | split result | `Process` | 前该数个核多 1 block |
| 同上 | `highInner` | `int64_t/64` | 第6字段；精确布局不可见 | matrix/block | `ComputeTinyInfo` | `ComputeBlockInfo` | 正常 block |
| 同上 | `highOuter` | `int64_t/64` | 第7字段；精确布局不可见 | block | `ComputeTinyInfo` | `ComputeBlockInfo` | 最后 block 判定 |
| 同上 | `highTail` | `int64_t/64` | 第8字段；精确布局不可见 | matrix | `ComputeTinyInfo` | `ComputeBlockInfo` | 最后 block |
| 同上 | `highMask` | `uint64_t[4]/256` | 第9字段（array）；精确布局不可见 | bit/element | `ComputeMask` | `maskLocal_`/MicroAPI mask | 仅 R-003 读取 |
| `TriangulatorMediumTilingData` | `bufferSize` | `int64_t/64` | 第1字段；精确布局不可见 | 字节 | `baseInfoOp.bufferSize` | queue `InitBuffer` | 每 slot |
| 同上 | `row` | `int64_t/64` | 第2字段；精确布局不可见 | row | `baseInfoOp.row` | loops/DMA count | 全矩阵 |
| 同上 | `col` | `int64_t/64` | 第3字段；精确布局不可见 | column | `baseInfoOp.col` | stride/regCount | 全矩阵 |
| 同上 | `copyCols` | `int64_t/64` | 第4字段；精确布局不可见 | element | `ComputeMediumInfo` | `copyEleNum=copyCols+i` | Triu interior 为每行前置零数起点 |
| 同上 | `usedCoreNum` | `int64_t/64` | 第5字段；精确布局不可见 | core | split result | `Init/Process` | 超出核返回 |
| 同上 | `normalCoreProcessNum` | `int64_t/64` | 第6字段；精确布局不可见 | block | split result | `Process` | 基准 block 数 |
| 同上 | `tailCoreProcessNum` | `int64_t/64` | 第7字段；精确布局不可见 | core-count remainder | split result | `Process` | 前该数个核多 1 block |
| 同上 | `highInner` | `int64_t/64` | 第8字段；精确布局不可见 | matrix/block | `ComputeMediumInfo` | `ComputeBlockInfo` | 正常 block |
| 同上 | `highOuter` | `int64_t/64` | 第9字段；精确布局不可见 | block | `ComputeMediumInfo` | `ComputeBlockInfo` | 最后 block 判定 |
| 同上 | `highTail` | `int64_t/64` | 第10字段；精确布局不可见 | matrix | `ComputeMediumInfo` | `ComputeBlockInfo` | 最后 block |
| 同上 | `headRows` | `int64_t/64` | 第11字段；精确布局不可见 | row | `max(-d,0)` | head loop | Triu 头部行全 copy |
| 同上 | `midRows` | `int64_t/64` | 第12字段；精确布局不可见 | row | `R-headRows-tailRows` | Select loop | 对角线穿过行 |
| 同上 | `tailRows` | `int64_t/64` | 第13字段；精确布局不可见 | row | `max(R-C+d,0)` | tail loop | Triu 尾部行全零 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 1/2/4/8 | `triangulator_tiling.cpp::ComputeTilingMode/DoTiling` | `TriangulatorTilingData` | `triu_apt.cpp::triu` → `triangulatorEntry<false>` | `split_all.h::SplitAll<T,0>` | `CMakeLists.txt` + `triu_def.cpp` + `ascend950/triu_binary.json` |
| R-002 | 11/12/14/18 | 同上 | `TriangulatorTilingData` | 同上 | `split_all.h::SplitAll<T,1>` | 同上 |
| R-003 | 31/32/34/38 | `ComputeTinyInfo/ComputeMask/PostTiling` | `TriangulatorTinyTilingData` | 同上 | `split_tiny_shape.h::SplitTinyShape<T>` | 同上 |
| R-004 | 41/42/44/48 | `ComputeMediumInfo/PostTiling` | `TriangulatorMediumTilingData` | 同上 | `split_medium_shape.h::SplitMediumShape<T,false>` | 同上 |
| R-005 | 21/22/24/28 | `ComputeNormalInfo/PostTiling` | `TriangulatorNormalTilingData` | 同上 | `split_row_and_col.h::SplitRowAndCol<T,false>` | 同上 |

## Kernel 实现

### 公共执行流程

`triu` entry 标注 `KERNEL_TYPE_AIV_ONLY`，以 `triangulatorEntry<false>` 锁定上三角语义。entry 先检查 user workspace，再用 key 的个位选 raw dtype 宽度、十位选执行族，解析对应 POD，在栈上创建 `TPipe` 和 Route 类后调用 `Init`/`Process`。四种 dtype key 使用 `int8_t/int16_t/int32_t/int64_t` 只是 raw word 载体；`triu_binary.json` 以 `DtypeByte` 模式把同字节数的公开 dtype 匹配到同一二进制实现。

各 Route 的 GM 域互不重叠，无跨核 atomic/barrier；核内顺序由 queue 生命周期、hard event 或 MicroAPI 顺序保证。workspace 不是节点间边，也不存在多 Kernel 中间结果。unknown key 没有 `else` 错误分支，entry 直接结束且不写输出。

### Route R-001：全零

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_triu.cpp` | `doTriuGetWorkspaceSize` | 非空且校验通过 | `l0op::Triu` |
| Tiling | `op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | `operatorType_ && diagOffset>=col` | mode 0 |
| Tiling | 同上 | `ComputeTilingKey` | `0*10+dtypeBytes` | key 1/2/4/8 |
| Router | `op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<false>` | `TILING_KEY_IS(OUTPUT_ZERO_INT*)` | `SplitAll<T,0>` |

#### Device 数据流

##### 函数级调用链

```text
triu(x,y,workspace,tiling)
  -> triangulatorEntry<false>
    -> invokeTemplateSplitAll<T,OUTPUT_ZERO_MODE>
      -> SplitAll::Init -> SplitAll::Process
        -> Duplicate(UB,0) -> SplitAll::ProcessPerLoop -> DataCopyPad(UB->GM)
```

##### 源码等价伪码

```text
# 伪码
N = fusedFrontAxis*row*col
core_span = (blockIdx==used-1 ? tailCoreProcessNum : normalCoreProcessNum)
gm_base = blockIdx*normalCoreProcessNum
B = 2*tiling.bufferSize                  # 只分配一个 B-byte TBuf
zero_ub[0:B/sizeof(T)] = 0; wait V->MTE3
for off in [0, core_span) step B/sizeof(T):
    valid = min(B/sizeof(T), core_span-off)
    DataCopyPad(y[gm_base+off], zero_ub, blockCount=1, blockLen=valid*sizeof(T))
PipeBarrier(MTE3)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `y` | `blockIdx*normal + loop*normalLoop` | `loopBlockLength` elements | raw T, flat ND | MTE3 | API out/ViewCopy | user/executor | 各核不重叠 |
| `ubBuf1_` | UB allocator | `2*tiling.bufferSize` bytes | T | `Duplicate(0)` | 所有 CopyOut loop | `TPipe`, Kernel lifetime | 单次初始化、只读复用 |
| workspace | user slice | 16 MiB | opaque | caller | entry null check | caller | body 不读写 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  zero["ubBuf1_ · depth=1 · 2*bufferSize bytes"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*floorAlign((UB_hw-8192)/2,32) <= UB_hw-8192
dynamic_ub = N/A
workspace_total = 16,777,216 bytes (reserved, no algorithmic reads/writes)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `ubBuf1_` | 启动后单次 `Duplicate` | 每个 MTE3 CopyOut | 无 | `V_MTE3` flag/wait，末尾 `PipeBarrier<PIPE_MTE3>` | CopyOut 不得早于 Duplicate |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| zero init | `Duplicate` + `V_MTE3` | Vector | MTE3 | zero UB | raw all-zero bits，无 cast |
| loop output | `DataCopyPad` | UB | GM | 当前 loop | `blockLen=valid*sizeof(T)`，末 loop 不多写 |
| completion | `PipeBarrier<PIPE_MTE3>` | 所有 MTE3 | Kernel return | GM y | 确保搬出完成 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode`, `OutputZeroAndInputTiling` | 极值对角线及最小 1024-byte/core 分核 |
| Kernel entry/router | `op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitAll<T,0>` | key→raw-width template |
| Device body/helper | `op_kernel/arch35/split_all.h` | `SplitAll::Process/ProcessPerLoop` | 单 zero buffer 和有效字节 CopyOut |

### Route R-002：原样拷贝

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 第一分支失败后 `diagOffset<=1-row` | mode 1 |
| Tiling | 同上 | `ComputeTilingKey` | `10+dtypeBytes` | key 11/12/14/18 |
| Router | `op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<false>` | `OUTPUT_INPUT_INT*` | `SplitAll<T,1>` |

#### Device 数据流

##### 函数级调用链

```text
triu -> triangulatorEntry<false> -> invokeTemplateSplitAll<T,OUTPUT_INPUT_MODE>
  -> SplitAll::Init -> SplitAll::Process
    -> ProcessPerLoop: wait MTE3->MTE2 -> GM->UB -> MTE2->MTE3 -> UB->GM -> signal reusable
```

##### 源码等价伪码

```text
# 伪码
B = tiling.bufferSize; allocate buf0[B], buf1[B]
for each core-owned loop, alternating buf0/buf1:
    wait previous MTE3 use of this buffer
    DataCopyPad(buf, x[coreBase+off], bytes=valid*sizeof(T))
    signal/wait MTE2->MTE3
    DataCopyPad(y[coreBase+off], buf, bytes=valid*sizeof(T))
    signal MTE3->MTE2 for next reuse
wait both buffers' final MTE3 completion
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x/y` | `blockIdx*normal + loop*normalLoop` | valid elements | raw T flat | caller/MTE3 | MTE2/caller | executor/user | disjoint per core |
| `ubBuf1_` | UB | `bufferSize` bytes | T | MTE2 | MTE3 | `TPipe` | ping |
| `ubBuf2_` | UB | `bufferSize` bytes | T | MTE2 | MTE3 | `TPipe` | pong |
| workspace | user slice | 16 MiB | opaque | caller | null check | caller | no body access |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  ping["ubBuf1_ · depth=1 · bufferSize bytes"]:1
  pong["ubBuf2_ · depth=1 · bufferSize bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*floorAlign((UB_hw-8192)/2,32) <= UB_hw-8192
dynamic_ub = N/A
workspace_total = 16,777,216 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ping/pong 各自 | MTE2 CopyIn | MTE3 CopyOut | 同 parity 的下个 loop | `MTE3_MTE2` + `MTE2_MTE3` event | 重写前等待上次 CopyOut |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| buffer reuse | `MTE3_MTE2` event | 前次 CopyOut | 下次 CopyIn | ping/pong slot | bitwise copy |
| stage order | `MTE2_MTE3` event | CopyIn | CopyOut | 当前 slot | 无 cast/算术 |
| tail | `DataCopyPadExtParams` | Host length | DMA | GM bounds | 精确 `tailLoopBlockNum*sizeof(T)` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode`, `OutputZeroAndInputTiling` | 选 copy route 并分核 |
| Kernel entry/router | `op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitAll<T,1>` | key→copy template |
| Device body/helper | `op_kernel/arch35/split_all.h` | `SplitAll::ProcessPerLoop` | 双 buffer DMA 事件网络 |

### Route R-003：tiny 预计算 mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `triangulator_tiling.cpp` | `ComputeTilingMode` | 非极值且 `R*C*dtypeBytes<=V` | mode 3 |
| Tiling | 同上 | `ComputeMask` | `c>=r+diagOffset` 置 POD bit | `highMask[4]` |
| Router | `triangulator_base.h` | `triangulatorEntry<false>` | key 31/32/34/38 | `SplitTinyShape<T>` |

#### Device 数据流

##### 函数级调用链

```text
triu -> invokeTemplateSplitTinyShape<T> -> SplitTinyShape::Init
  -> Process -> ComputeBlockInfo -> CopyIn
  -> DoVectorCopy (MicroAPI mask load/unpack, zero-fill, masked source overwrite)
  -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
load POD highMask[4] into 32-byte maskLocal
for each core-owned high block:
    highActual = last ? highTail : highInner
    GM offset = block*highInner*planeArea
    DMA highActual rows-of-matrices into VReg-strided UB queue
    for matrix in block:
        vreg = next VReg-sized source plane
        dst VReg = 0
        overwrite dst lanes selected by expanded highMask with source
    DMA only planeArea bytes per matrix back to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `block*highInner*planeArea` | `highActual*planeArea` elements | batched ND | caller/Kernel | Kernel/caller | executor/user | last uses highTail |
| `inQueue_` | UB | depth 2 × `bufferSize` | T, each plane VReg-strided | MTE2 | MicroAPI | TPipe | producer prefetches next block |
| `outQueue_` | UB | depth 2 × `bufferSize` | T, same stride | MicroAPI | MTE3 | TPipe | queue lifetime |
| maskBuf | UB | 32 bytes | `uint64[4]` | POD load | mask register load | TPipe/Kernel | read-only |
| registers | VReg | `vreg/dreg/sreg` + masks | T/mask | MicroAPI | MicroAPI | vector scope | fixed source does not expose register bytes |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inq["inQueue · depth=2 · 2*bufferSize"]:2
  mask["maskBuf · 32B"]:1
  outq["outQueue · depth=2 · 2*bufferSize"]:2
```

##### 字节账与复用顺序

```text
B = floorAlign((UB_hw-8192)/4,32)
UB_total = 2B + 2B + 32 = 4B+32 <= UB_hw-8192+32 <= UB_hw
dynamic_ub = N/A
workspace_total = 16,777,216 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input queue slot | `CopyIn` | `DoVectorCopy::DeQue` | `FreeTensor` 后 | queue EnQue/DeQue/Free | 深度 2 允许 next-block prefetch |
| output queue slot | `DoVectorCopy` | `CopyOut::DeQue` | `FreeTensor` 后 | queue protocol | CopyOut 完成才复用 |
| maskBuf | Init/POD load | 每个 vector block | 无 | Kernel 内只读 | 所有 matrix 共用相同 R/C/d mask |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad`, queue | MTE2 | `DeQue` | input slot | `blockCount=highActual`, `blockLen=planeArea*sizeof(T)` |
| mask | `MaskUnPack` 0–3 次 | maskBuf | Select-like masked write | element lanes | 按 T 字节宽展开，无数值 cast |
| UB→GM | queue + `DataCopyPad` | vector output | MTE3 | output slot | last block 用 `highTail` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/triangulator_tiling.cpp` | `ComputeTinyInfo`, `ComputeMask` | VReg guard、batch 分块和 256-bit mask |
| Kernel entry/router | `op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitTinyShape<T>` | tiny POD 解析 |
| Device body/helper | `op_kernel/arch35/split_tiny_shape.h` | `DoVectorCopy` | zero-first + masked raw source overwrite |

### Route R-004：medium 整矩阵

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `triangulator_tiling.cpp` | `ComputeTilingMode` | 前三条件失败且 `R*CeilAlign(C*dtypeBytes,V)*4<=UB` | mode 4 |
| Tiling | 同上 | `ComputeMediumInfo` | 计算 high split 与 head/mid/tail rows | medium POD |
| Router | `triangulator_base.h` | `triangulatorEntry<false>` | key 41/42/44/48 | `SplitMediumShape<T,false>` |

#### Device 数据流

##### 函数级调用链

```text
triu -> invokeTemplateSplitMediumShape<T,false> -> SplitMediumShape::Init
  -> Process -> ComputeBlockInfo -> CopyIn
  -> DoVectorCopy(head copy, mid Select, tail zero) -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
for each high block containing highActual complete matrices:
    DMA highActual*row rows, each blockLen=col*sizeof(T), into aligned UB rows
    for each matrix:
        headRows=max(-d,0): copy all source lanes
        for i in [0,midRows):
            zero_prefix = copyCols+i
            for each VReg segment: dst = Select(zero,src,UpdateMask(zero_prefix))
        tailRows=max(row-col+d,0): write zero lanes
    DMA valid col bytes of every row back to contiguous GM rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `block*highInner*row*col` | `highActual*row*col` | ND | caller/Kernel | Kernel/caller | executor/user | block disjoint |
| `inQueue_` | UB | 2×B | T, aligned row stride | MTE2 | MicroAPI | TPipe | depth 2 |
| `outQueue_` | UB | 2×B | T, aligned row stride | MicroAPI | MTE3 | TPipe | depth 2 |
| registers | VReg | three T RegTensor + masks | raw T | load/zero | Select/store | vector scope | bytes not visible |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inq["inQueue · depth=2 · 2B"]:2
  outq["outQueue · depth=2 · 2B"]:2
```

##### 字节账与复用顺序

```text
B = floorAlign((UB_hw-8192)/4,32)
UB_total = 2B+2B = 4B <= UB_hw-8192
dynamic_ub = N/A
workspace_total = 16,777,216 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input slot | `CopyIn` | `DoVectorCopy` | `FreeTensor` 后 | queue protocol | prefetch 最多占两 slot |
| output slot | `DoVectorCopy` | `CopyOut` | `FreeTensor` 后 | queue protocol | MTE3 消费完才归还 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| row DMA | `DataCopyPad` | GM | input queue | aligned UB rows | valid `blockLen=C*sizeof(T)`，padding lane 不写 GM |
| vector | `UpdateMask`, `Select` | source/zero regs | output reg/store | 对角线前缀 | bitwise source/zero，`rowMask` 限制存储 |
| high tail | `highActual` | POD | DMA/vector loops | last block | 不处理不存在的 matrix |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/triangulator_tiling.cpp` | `ComputeMediumInfo` | UB guard、high split、head/mid/tail |
| Kernel entry/router | `op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitMediumShape<T,false>` | medium key 映射 |
| Device body/helper | `op_kernel/arch35/split_medium_shape.h` | `CopyIn/DoVectorCopy/CopyOut` | 整行 DMA 和 mask Select |

### Route R-005：normal 行列分块

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `triangulator_tiling.cpp` | `ComputeTilingMode` | R-001–R-004 条件均失败 | mode 2 |
| Tiling | 同上 | `ComputeNormalInfo` | `col<=V/T`，否则 `row<=128`，否则 normal | `rowInner/colInner` |
| Router | `triangulator_base.h` | `triangulatorEntry<false>` | key 21/22/24/28 | `SplitRowAndCol<T,false>` |

#### Device 数据流

##### 函数级调用链

```text
triu -> invokeTemplateSplitRowAndCol<T,false> -> SplitRowAndCol::Init
  -> Process: Duplicate(zeroBuf)
    -> ComputeBlockInfo -> CopyIn
    -> DoCopyVector(only ON_LINE tiles)
    -> CopyOut(copy queue / zeroBuf / inOutQueue)
```

##### 源码等价伪码

```text
# 伪码
initialize zeroBuf once; wait V->MTE3
for each owned tile (batch,rowTile,colTile):
    compute actual rows/cols and GM base
    headRows = colStart-(rowStart+d)
    tailRows = rowActual-colActual-headRows
    position = UPPER_RIGHT if headRows>=rowActual
               LOWER_LEFT if tailRows>=rowActual
               ON_LINE otherwise
    if LOWER_LEFT: skip GM input; CopyOut zeroBuf
    else if UPPER_RIGHT: GM->inOutQueue->GM (no vector)
    else:
        GM->inQueue
        per row/VReg: keep lanes j>=i+d, zero earlier lanes with Select
        outQueue->GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM tile | `batch*R*C + rowTile*rowInner*C + colTile*colInner` | `rowActual×colActual` | strided ND tile | caller/Kernel | Kernel/caller | executor/user | `srcStride=(C-colActual)*sizeof(T)` |
| `inQueue_` | UB | 2B | aligned tile | MTE2 | vector | TPipe | ON_LINE only |
| `outQueue_` | UB | 2B | aligned tile | vector | MTE3 | TPipe | ON_LINE only |
| `inOutQueue_` | UB | 2B | aligned tile | MTE2 | MTE3 | TPipe | UPPER_RIGHT raw copy |
| `zeroBuf_` | UB | B | T zeros | Duplicate | MTE3 | TPipe | all LOWER_LEFT tiles share |
| registers | VReg | zero/src/dst + masks | raw T | vector | vector | vector scope | bytes invisible |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  inq["inQueue · depth=2 · 2B"]:2
  outq["outQueue · depth=2 · 2B"]:2
  bind["inOutQueue · depth=2 · 2B"]:2
  zero["zeroBuf · B"]:1
```

##### 字节账与复用顺序

```text
B = floorAlign((UB_hw-8192)/8,32)
UB_total = 2B+2B+2B+B = 7B <= 7/8*(UB_hw-8192)
dynamic_ub = N/A
workspace_total = 16,777,216 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output queue slots | MTE2/vector | vector/MTE3 | `FreeTensor` 后 | queue protocol | 只在 ON_LINE 成对使用 |
| `inOutQueue_` slot | MTE2 | MTE3 | `FreeTensor` 后 | bound queue protocol | 只在 UPPER_RIGHT 使用，不进 vector |
| `zeroBuf_` | 单次 Duplicate | 所有 LOWER_LEFT CopyOut | 无 | `V_MTE3` event 后只读 | 每次 DMA 只读 `rowActual×colActual` 有效区 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| zero init | `FetchEventID(V_MTE3)` + flag/wait | Duplicate | zero-tile MTE3 | `zeroBuf_` | all-zero raw bits |
| ON_LINE | queue + `UpdateMask/Select` | MTE2/vector | vector/MTE3 | tile slots | `rowMask=regCount` 限制 store，`selectMask=copyCols+i` 推进对角线 |
| row/col tail | `rowActual/colActual` + DMA strides | `ComputeBlockInfo` | CopyIn/Out | GM bounds | 末 tile 不读写越界 |
| whole-tile routes | position enum | address classifier | CopyIn/Out | 整 tile | UPPER_RIGHT bitwise copy，LOWER_LEFT no input read |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/triangulator_tiling.cpp` | `ComputeNormalInfo` | 3 级 row/col inner 决策和 UB 上界 |
| Kernel entry/router | `op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitRowAndCol<T,false>` | normal key 映射 |
| Device body/helper | `op_kernel/arch35/split_row_and_col.h` | `ComputeBlockInfo/CopyIn/DoCopyVector/CopyOut` | tile 分类、DMA 短路、Select 和 tail |

## 算子特定优化

### Trick T-001：极值对角线直接化简为全零/纯 DMA

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：Triu 的 `diagonal>=col` 或 `diagonal<=1-row`
- Golden 来源：[`conversion/triu/op_host/arch35/triangulator_tiling.cpp:198-216`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.cpp#L198-L216)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if ((!operatorType_ && baseInfoOp.diagOffset <= -baseInfoOp.row) ||
    (operatorType_ && baseInfoOp.diagOffset >= baseInfoOp.col)) {
    splitCoreOp.tilingMode = OUTPUT_ZERO_MODE;
} else if (
    (!operatorType_ && baseInfoOp.diagOffset >= baseInfoOp.col - 1) ||
    (operatorType_ && baseInfoOp.diagOffset <= 1 - baseInfoOp.row)) {
    splitCoreOp.tilingMode = OUTPUT_INPUT_MODE;
} else if (baseInfoOp.col * baseInfoOp.row * baseInfoOp.dtypeBytes <= baseInfoOp.vRegSize) {
    splitCoreOp.tilingMode = TINY_SHAPE_MODE;
} else if (
    baseInfoOp.row * CeilAlign(baseInfoOp.col * baseInfoOp.dtypeBytes, baseInfoOp.vRegSize) * FOUR_BUFFER <=
    baseInfoOp.ubSize) {
    splitCoreOp.tilingMode = MEDIUM_SHAPE_MODE;
} else {
    splitCoreOp.tilingMode = OUTPUT_NORMAL_MODE;
}
```

#### 机制与设计原因

边界对角线使得上三角 mask 恒为假或恒为真。Host 在 Kernel 前专门化：R-001 只在 UB 中生成一次零并反复搬出，完全消除 GM input 读取；R-002 只做双 buffer GM→UB→GM，消除逐元素 mask 和 Vector 计算。`dtypeBytes` 只决定 DMA 字节数与 raw template 宽度。

#### 迁移边界

- 适用：任何能在 Host 证明输出 mask 全真/全假的三角、band 或 mask-fill 算子。
- 前提：对角线与 R/C 已知；raw zero bit-pattern 对目标 dtype 表示数值零。
- 失效：`1-row<diagonal<col` 时必须进 R-003/R-004/R-005。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的真实 fallback 是在边界条件不成立时进入 tiny/medium/normal；若错把内部对角线送入 R-001/R-002，会将部分本应保留/置零的区域整体写错。

### Trick T-002：normal 路线的整 tile 分类与无输入 zero 短路

#### 触发与源码依据

- 适用 Route：R-005
- 触发条件：行列 tile 整体位于上三角保留区或下三角置零区
- Golden 来源：[`conversion/triu/op_kernel/arch35/split_row_and_col.h:185-220`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_row_and_col.h#L185-L220)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
info.headRows = colIndex * colInner_ - (rowIndex * rowInner_ + diagOffset_);
info.tailRows = info.rowActual - info.colActual - info.headRows;
info.position = ON_LINE;
if (info.headRows >= info.rowActual) {
    info.position = UPPER_RIGHT;
} else if (info.tailRows >= info.rowActual) {
    info.position = LOWER_LEFT;
}

info.gmOffset = highAxisIndex * planeArea_ + rowIndex * rowInner_ * col_ + colIndex * colInner_;
info.stride = colInner_ * sizeof(T) / GetUbBlockSize() -
              CeilDiv(static_cast<uint32_t>(info.colActual * sizeof(T)), static_cast<uint32_t>(GetUbBlockSize()));
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::CopyIn(NormalBlockInfo& info)
{
    if (info.position == UPPER_RIGHT && IS_LOWER || info.position == LOWER_LEFT && !IS_LOWER) {
        return;
    }
```

#### 机制与设计原因

`headRows/tailRows` 用 tile 起点和对角线关系将 tile 分为 `UPPER_RIGHT/LOWER_LEFT/ON_LINE`。对 Triu（`IS_LOWER=false`），`LOWER_LEFT` 的 `CopyIn` 直接 return，`CopyOut` 从一次初始化的 `zeroBuf_` 搬出；`UPPER_RIGHT` 使用 `TQueBind` 输入直接搬出；只有 `ON_LINE` 进入 MicroAPI Select。这同时减少全零 tile 的 GM 读取和两类整 tile 的 Vector 指令。

#### 迁移边界

- 适用：mask 边界在 tile 空间单调、能用几个整数比较证明整 tile 类型的算子。
- 前提：分块覆盖域不重叠，GM stride 与 `rowActual/colActual` 尾块一致。
- 失效：对角线穿过 tile 时必须进 `ON_LINE` Select，不能套用整块 copy/zero。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不得只根据 tile 左上角判定整块类型；官方实现同时检查 `headRows>=rowActual` 与 `tailRows>=rowActual`，否则对角线穿过的 tile 会被误分类。

### Trick T-003：tiny 路线一次 Host mask 复用整个 batch

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：`row*col*dtypeBytes<=VRegSize`，且不是全零/全 copy
- Golden 来源：[`conversion/triu/op_kernel/arch35/split_tiny_shape.h:180-201`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_tiny_shape.h#L180-L201)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::Duplicate(sreg, 0);

if constexpr (sizeof(T) / sizeof(int8_t) == DOUBLE) {
    AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::MaskDist::DIST_US>(maskRegO, maskUbAddr);
} else if constexpr (sizeof(T) / sizeof(int8_t) == QUADRUPLE) {
    AscendC::MicroAPI::DataCopy(maskReg, maskUbAddr);
    AscendC::MicroAPI::MaskUnPack(maskRegD, maskReg);
    AscendC::MicroAPI::MaskUnPack(maskRegO, maskRegD);
} else if constexpr (sizeof(T) / sizeof(int8_t) == OCTA) {
    AscendC::MicroAPI::DataCopy(maskReg, maskUbAddr);
    AscendC::MicroAPI::MaskUnPack(maskRegD, maskReg);
    AscendC::MicroAPI::MaskUnPack(maskRegQ, maskRegD);
    AscendC::MicroAPI::MaskUnPack(maskRegO, maskRegQ);
} else {
    AscendC::MicroAPI::DataCopy(maskRegO, maskUbAddr);
}

for (uint16_t i = 0; i < repeatTimes; i++) {
    AscendC::MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vreg, srcAddr, dataCount);
    AscendC::MicroAPI::DataCopy(dstAddr, sreg, moveMask);
    AscendC::MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, vreg, dataCount, maskRegO);
}
```

#### 机制与设计原因

Host 把单矩阵的上三角保留集编码到 `uint64_t highMask[4]`。Device 只在 `Init` 把它放入 32-byte UB，然后根据 raw T 宽度做 0–3 次 `MaskUnPack`，对每个 batch matrix 先全 VReg 写零、再用同一 mask 覆盖保留 lane。这避免 Device 对每个元素重复计算行列坐标和比较。

#### 迁移边界

- 适用：小固定平面、batch 间 mask 相同的二维选择。
- 前提：平面 bit 数不超过 256，且 plane bytes 不超过 VReg；必须按 dtype 宽度做正确 mask 扩展。
- 失效：plane bytes 超过 VReg 时改用 R-004/R-005，不能读越 `highMask[4]`。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不能把 256-bit element mask 不经展开直接用于 16/32/64-bit T；官方路线按 `sizeof(T)` 选 `DIST_US` 或连续 `MaskUnPack`，省略会使 mask lane 与元素错位。

## 模板与实现对比

| 对比维度 | R-001 全零 | R-002 原样拷贝 | R-003 tiny | R-004 medium | R-005 normal | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | `d>=C` | `d<=1-R` | plane bytes≤V | whole-matrix 4B≤UB | fallback | 先利用语义恒等，再按 VReg/UB 容量选专门化 |
| 模板与 dtype | `SplitAll<T,0>` | `SplitAll<T,1>` | `SplitTinyShape<T>` | `SplitMediumShape<T,false>` | `SplitRowAndCol<T,false>` | T 只取 1/2/4/8-byte raw integer carrier |
| 循环与分核 | flat element | flat element | batch-plane block | complete-matrix block | batch×rowTile×colTile | 与 Route 的最小独立处理域一致 |
| Queue/TBuf/寄存器 | 1 TBuf | 2 TBuf | 2-depth in/out + mask | 2-depth in/out | 3×2-depth queues + zero | 资源随计算复杂度递增 |
| 内存总占用与复用 | 约 UB-8KiB | 约 UB-8KiB | 4B+32 | 4B | 7B | Host 预留 8KiB；normal 额外保守 1B |
| 搬运/Vector API | zero-only MTE3 | MTE2/MTE3 ping-pong | strided DMA + mask copy | row DMA + Select | tile DMA/zero/copy + Select | 尽量在粗粒度证明后跳过 vector |
| 同步、精度与 tail | V_MTE3, exact bytes | hard events, exact bytes | queue/highTail | queue/highTail/row mask | queue+event/row+col tail | 全部 bitwise，无 cast/round |
| fallback/失败边界 | 条件失败去 R-002 | 去 R-003 | 去 R-004 | 去 R-005 | unknown key 无输出 | Host 没有 Device 通用 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档声明支持，但本任务固定 `conversion/triu/` 实现范围没有 A2 Host/POD/Kernel/构建闭包 | 公开文档声明支持，但本任务固定实现范围没有 A3 Host/POD/Kernel/构建闭包 | 支持；文档、OpDef、binary config 和 CMake 一致 |
| Host/Launcher | ACLNN 公共层可见，平台 Tiling 固定范围不可见 | 同左 | `arch35` `TriangulatorTiling` 五 Route |
| Kernel/资源 | 固定范围不可见 | 固定范围不可见 | AIV_ONLY，VReg+UB，16 MiB 保留 workspace |

## 限制与失败边界

- 公开文档的 shape 栏写“2–8 维和 0 维”，但 API `CheckShape` 对任何 `Size()>0` 的 rank<2 输入返参数错；只有 empty tensor 绕过最小 rank 检查并在 API 层短路。本卡以实现分支为可复用边界，不推测文档中“0 维”的其它含义。
- Host 的 `matrixNum*row*col`、`row*col*dtypeBytes`、对齐乘法使用 `int64_t`，没有显式溢出检查；Device 还把 DMA `blockCount` 窄化为 `uint16_t`、长度/步长窄化为 `uint32_t`，固定仓未给出超限 fallback。
- `PostTiling` 的 raw POD capacity 不足会返图失败；平台 AIV 核数/UB 非正、dtype size 为 0、rank<2 或必需 context/shape/desc 缺失也会失败。
- Kernel entry 对 null user workspace 或 unknown key 只直接返回，固定 Device 源码不写错误码；调用方必须保持 Host Tiling 产生的 key/workspace 协议。
- executor 如何汇总、对齐和切分 Triu/Contiguous/ViewCopy 的总 workspace，以及通用 queue/MicroAPI 内部指令级行为，在固定仓不可见。
- `BEGIN_TILING_DATA_DEF` 的宏展开和序列化 ABI 实现在固定仓不可见；不得仅凭字段表自行重排或改位宽。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/triu/docs/aclnnTriu&aclnnInplaceTriu.md` | `aclnnTriuGetWorkspaceSize`, `aclnnTriu` | 公开原型、平台、dtype/shape 和错误语义 |
| 入口声明/约束 | `conversion/triu/op_api/aclnn_triu.h` | 两段式声明 | API 计算图与参数 |
| 入口实现 | `conversion/triu/op_api/aclnn_triu.cpp` | `CheckParams`, `doTriuGetWorkspaceSize`, `aclnnTriu` | API 校验、empty 短路、Contiguous/Triu/ViewCopy/executor |
| Host/Launcher | `conversion/triu/op_api/triu.cpp` | `l0op::Triu` | 中间输出分配与 AI_CORE launcher |
| Host/Launcher | `conversion/triu/op_host/triu_def.cpp` | `ops::Triu` | 950 dtype/format 与 `triu_apt` 配置 |
| Host/Tiling/Launcher | `conversion/triu/op_host/arch35/triu_tiling.cpp` | `TilingPrepare4Trilu`, `IMPL_OP_OPTILING(Triu)` | AIV 核数/UB producer 和 Tiling 注册 |
| Host/Tiling/Launcher | `conversion/triu/op_host/arch35/triu_tiling.h` | `Tiling4Trilu<true,false>` | shape/dtype/diagonal 解析与上三角语义 |
| Host/Tiling/Launcher | `conversion/triu/op_host/arch35/triangulator_tiling.h` | POD declarations | 四种 POD 与 Host 状态 |
| Host/Tiling/Launcher | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `TriangulatorTiling::*` | 五 Route、分核、UB、POD、workspace、key |
| Kernel entry/body | `conversion/triu/op_kernel/triu_apt.cpp` | `triu` | AIV_ONLY entry |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<false>` | 20 个 key 到 5 个 body 族 |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_all.h` | `SplitAll` | 全零/全 copy |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_tiny_shape.h` | `SplitTinyShape` | tiny mask 路线 |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_medium_shape.h` | `SplitMediumShape<T,false>` | medium 路线 |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_row_and_col.h` | `SplitRowAndCol<T,false>` | normal 路线 |
| 构建接线 | `conversion/triu/op_host/config/ascend950/triu_binary.json` | `DtypeByte` | 950 raw byte-width binary matching |
| 构建接线 | `conversion/triu/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `SUPPORT_TILING_DIR` | ascend950 使用 arch35 Tiling，仅公开范围内 950 闭包 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `name=Triu` | 活动 AscendC binary 配置列出 `ascend950`/`mc62`，950 编译选项闭合 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 四个 POD 按 Route 保留最小参数集 | 字段顺序/位宽、key 对 POD 的映射和 raw serialization 必须一致 | `Triangulator*TilingData`, `PostTiling`, router |
| 调用顺序 | API 先 Contiguous，再 Triu，后 ViewCopy | 不得让 ViewCopy 早于 Triu 或让非连续 storage 直接套用连续 GM 公式 | `doTriuGetWorkspaceSize` |
| workspace/内存 | 16 MiB slice 可作为现有 entry 协议的保留空间 | `GetUserWorkspace` 不得返回 null；UB 总账和 depth 不得超 Host guard | `PostTiling`, `triangulatorEntry`, 各 `InitBuffer` |
| 同步 | queue protocol、R-002 hard events、zero buffer V_MTE3 是完整复用协议 | 不得在 CopyOut 完成前重写 slot，不得在 Duplicate 完成前 MTE3 读 zero buffer | `SplitAll::ProcessPerLoop`, `SplitRowAndCol::Process` |
| 精度与尾块 | raw-width copy/zero/select 支持同字节 dtype 共用 body | 不得引入数值 cast；必须保留 `highTail/rowActual/colActual/tailLoopBlockNum` 和 mask 展开 | `triu_binary.json`, 各 `CopyOut/DoVectorCopy` |
| 失败边界 | Host 校验和 Tiling capacity 检查可直接复用 | unknown key/null workspace 的 Device 无声返回不能被视为成功输出 | `CheckParams`, `PostTiling`, `triangulatorEntry` |