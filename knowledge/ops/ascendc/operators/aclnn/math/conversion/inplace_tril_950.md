---
type: Operator
title: "InplaceTril 950 ACLNN 算子知识"
description: "说明 aclnnInplaceTril 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/docs/aclnnTril&aclnnInplaceTril.md"
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/op_api/aclnn_tril.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/op_api/aclnn_tril.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/op_api/tril.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/op_host/tril_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/op_host/arch35/tril_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/op_host/config/ascend950/tril_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/op_kernel/tril_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triu_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.cpp
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
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/tril/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake
    role: implementation
created_at: "2026-08-25T09:50:31Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# InplaceTril 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnInplaceTrilGetWorkspaceSize` + `aclnnInplaceTril`，公开两段式 ACLNN，无版本后缀。 |
| 计算语义 | 对最后两维的每个 `(i,j)`，当 `j <= i + diagonal` 时保留输入，否则写零；结果经 `ViewCopy` 写回 `selfRef`。 |
| 输入输出 | `selfRef` 是 2～8 维 ND、允许非连续；950 支持 1/2/4/8 字节宽的整数、浮点、布尔和复数公开 dtype；接口没有独立输出对象。 |
| 执行类型 | 五条 Route 均为单个算子自有 `AIV_VECTOR` 节点；entry 显式 `KERNEL_TYPE_AIV_ONLY`。前后可能有框架 `Contiguous` 与 `ViewCopy` 节点。 |
| Kernel 路线 | R-001 全零、R-002 全拷贝、R-003 tiny 整矩阵寄存器掩码、R-004 medium 整行分段、R-005 normal 行列分块；各 Route 按 dtype 字节宽展开 4 个 key。 |
| 核心资源 | GM、AIV UB、`TBuf`/depth=2 `TQue`、MicroAPI `RegTensor/MaskReg`；Tiling 声明 16 MiB user workspace，body 只检查其非空而不读写。executor 还持有连续中间输入、独立 Tril 输出与框架复制资源。 |
| 来源闭包 | `source_closed`：公开 Inplace ACLNN 经普通 Tril executor 图复用到 950 arch35 Tiling、4 类 POD、20 个 key、AIV router、5 个 body 族和构建接线；`Contiguous`/`ViewCopy`、executor 内部 workspace 切片以及 AscendC primitive 实体是明确框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

把任意前导维折叠为 `matrixNum`，令最后两维分别为 `row`、`col`。对每个批次矩阵：

```text
y[..., i, j] = x[..., i, j],  j <= i + diagonal
y[..., i, j] = 0,             j >  i + diagonal
selfRef <- ViewCopy(y, selfRef)
```

`aclnnInplaceTrilGetWorkspaceSize` 并不让 Tril kernel 的 `x` 与 `y` 指针直接别名：它把 `selfRef` 同时作为普通 `aclnnTrilGetWorkspaceSize` 的输入和最终 `out`，普通路径先 `Contiguous(selfRef)`，再由 `l0op::Tril` 分配独立连续输出，最后 `ViewCopy(trilResult, selfRef)`。因此真正的原地提交发生在框架复制节点，算子自有 kernel 仍按独立输入/输出实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | 2～8 维、ND；公开文档允许非连续 | 950：DOUBLE、FLOAT、FLOAT16、BFLOAT16、INT8/16/32/64、UINT8/16/32/64、BOOL、COMPLEX32/64 | 非空指针；既是逻辑输入也是最终写回目标。空 tensor 在 ACLNN Host 成功短路。 |
| `diagonal` | Host 标量 | `int64_t` | 对角线偏移；正值向右上移动，负值向左下移动，无默认参数重载。 |
| `workspaceSize` | Host 输出标量 | `uint64_t*` | 第一段返回完整 executor 图所需 Device workspace；必须非空。 |
| `executor` | Host 输出句柄 | `aclOpExecutor**` | 第一段创建并移交；第二段消费，必须非空。 |
| `workspace` | Device 地址 | 原始字节区 | 第二段传给 `CommonOpExecutorRun`；算子 Tiling 的 user workspace 固定为 16 MiB。 |
| `stream` | ACL runtime stream | `aclrtStream` | executor 内各节点在该 stream 上按框架顺序提交。 |

公开 API 的 dtype 先按 SoC 选择：950 的 `IsRegBase()` 分支返回 `DTYPE_SUPPORT_LIST_REGBASE`。binary config 用 `DtypeByte` 匹配，仅保留 int8/int16/int32/int64 四个代表二进制；Device 将同字节宽 dtype 按 `int8_t/int16_t/int32_t/int64_t` 位模式搬运、选择或清零，不做数值转换。

### 入口与版本

公开文档同时定义 `aclnnTril` 与 `aclnnInplaceTril`；本卡只覆盖原地逻辑算子。`aclnnInplaceTrilGetWorkspaceSize` 在 `aclnn_tril.cpp` 中经 `const_cast` 取得同一 `selfRef` 的输出 view，然后完整复用无版本后缀的 `aclnnTrilGetWorkspaceSize`；第二段 `aclnnInplaceTril` 直接调用 `CommonOpExecutorRun`。固定仓不存在更高版本后缀。

950 公开产品表、`Tril::AICore().AddConfig("ascend950", ...)`、`ascend950/tril_binary.json`、`SUPPORT_COMPUTE_UNIT "ascend950"` 与 arch35 AIV entry 一致。文档也声称 A2/A3 支持，但指定的 `conversion/tril/` 构建只接入 arch35，OpDef 只注册 `ascend950`；固定范围没有 A2/A3 的 Host/Tiling/Kernel 闭包，所以本卡不把 950 实现外推到 A2/A3。

## Host/Launcher 执行流程

### 参数校验与前处理

真实顺序如下：

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize` 与 `executor`。
2. Inplace 包装把 `out = const_cast<aclTensor*>(selfRef)`，进入普通 `aclnnTrilGetWorkspaceSize`。
3. `CheckNotNull` 同时检查 `self`/`out`，等价为检查 `selfRef`；`CheckDtypeValid` 取 950 RegBase 列表并检查输入、输出 dtype 一致；同一对象天然满足一致性。
4. `CheckFormat` 检查 storage format 一致；`CheckShape` 要求维数大于 1、输入输出维数不超过 8且 view shape 相等。同一对象天然满足 format/shape 相等，但维数约束仍生效。
5. 创建 executor。`selfRef->IsEmpty()` 时直接返回 executor workspace，不创建任何 Device 节点，也不写数据。
6. 非空时加入框架 `Contiguous(selfRef)`，加入 `l0op::Tril(selfContiguous, diagonal)`，最后加入 `ViewCopy(trilResult, selfRef)`；任一返回空指针即终止第一段。
7. 第二段以调用者提供的 workspace、size、executor、stream 执行整张图并返回框架状态。

### 参数、Workspace 与生命周期

`l0op::Tril` 按输入 storage shape/dtype/format 由 executor 分配独立 `out`，再用 `ADD_TO_LAUNCHER_LIST_AICORE(Tril, AI_CORE, input, attr, output)` 加入算子节点。950 Tiling 从 origin shape 生成 `row=dim[-2]`、`col=dim[-1]`，把其余维连乘为 `matrixNum`，属性缺失时 `diagonal=0`。`TilingPrepare4Trilu` 读取 AIV core 数和物理 UB，Tiling 再预留 8192 B。

`PostTiling` 对所有 Route 都把第一个 user workspace slice 设成 `16,777,216` B，并设置 `blockDim=usedCoreNum`、key 与 Route 对应 POD。kernel entry 调用 `GetUserWorkspace(workspace)`，空指针立即返回；其后 body 不读取 `userWS`。该固定 16 MiB 的 owner 是 executor/调用者，至少覆盖该节点执行期。完整 ACLNN `workspaceSize` 还包含 `Contiguous`、独立 Tril 输出及 `ViewCopy` 的框架资源，其切片公式和复用时点在固定仓不可见，只能以 `uniqueExecutor->GetWorkspaceSize()` 为边界。

### 启动图与失败传播

非空输入的逻辑启动图：

```text
selfRef
  -> [框架 Contiguous，必要时物化连续输入]
  -> [Tril AIV node：tril(x, y, workspace, tiling)，blockDim=usedCoreNum]
  -> [框架 ViewCopy：独立连续 y 写回 selfRef view]
```

算子自有节点始终只有一个：Host `l0op::Tril` 添加 `Tril`，entry 是 `conversion/tril/op_kernel/tril_apt.cpp::tril`，POD 由 `Tiling4Trilu<false,false>` 生成。它读取连续 `x`、写独立连续 `y`，不以 workspace 交接中间数据。框架在同一 executor/stream 中维护 `Contiguous→Tril→ViewCopy` 顺序与中间 tensor 生命周期；固定仓未展开 executor 的 stream event、每个框架节点 blockDim、workspace offset 和失败清理。第一段任何检查/分配/加节点失败均不返回可执行图；第二段由 `CommonOpExecutorRun` 传播执行状态，Tril 节点失败时后续 `ViewCopy` 是否提交及清理细节固定仓不可见。

## Route 与参数设计

### 分核与资源计算

共同变量（单位显式列出）：

```text
B = dtypeBytes ∈ {1,2,4,8}                           # 字节/元素
M = product(originShape[0:-2])                       # 矩阵数
R = originShape[-2], C = originShape[-1]             # 元素
Uphys = availableUBSize                              # 字节
U = Uphys - 8192                                     # Tiling 可用字节
V = GetVRegSize(context)                             # 字节
key = 10 * tilingMode + B                            # mode: 0/1/2/3/4
```

R-001/R-002：`baseBlockNum=M*R*C` 个元素。若 `baseBlockNum*B < 1024`，只用 1 核；否则 `normal=max(ceil(baseBlockNum/totalCoreNum), floor(1024/B))`，`used=ceil(baseBlockNum/normal)`，最后一核处理 `baseBlockNum-normal*(used-1)`。Host `bufferSize=floor_align(U/2,32)`；R-001 Device 把它乘 2 后建一个 buffer，R-002 建两个该大小的 ping-pong buffer。

R-003：`bufferSize=floor_align(U/4,32)`；每个整矩阵占 `R*C*B <= V`，`highInner` 是每个 base block 的矩阵数，`highOuter=ceil(M/highInner)`，最后块用 `highTail`。`used=min(highOuter,totalCoreNum)`，前 `highOuter % used` 个核各多处理一个 base block。

R-004：同样四 buffer 预算。整行在 UB 中按 `alignCount=(C<=V/B ? 32/B : V/B)` 对齐；Tiling 令每个 base block 包含 `highInner` 个整矩阵，`highOuter=ceil(M/highInner)`，分核同 R-003。

R-005：`bufferSize=floor_align(U/8,32)`，源码常量名虽为 `SEVEN_BUFFER`，值实际是 8。Tiling 求 `colInner,rowInner` 使 `rowInner*colInner <= bufferSize/B`，其中列按 32 B 或 VReg 元素数对齐；`rowOuter=ceil(R/rowInner)`、`colOuter=ceil(C/colInner)`、`baseBlockNum=M*rowOuter*colOuter`，再按前余数核多一块分配。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 全零 | `1/2/4/8`; `SplitAll<T,0>` | Tril 且 `diagonal <= -R` | R-002 |
| 2 | R-002 全拷贝 | `11/12/14/18`; `SplitAll<T,1>` | 前项不命中且 `diagonal >= C-1` | R-003 |
| 3 | R-003 tiny | `31/32/34/38`; `SplitTinyShape<T>` | 前两项不命中且 `R*C*B <= V` | R-004 |
| 4 | R-004 medium | `41/42/44/48`; `SplitMediumShape<T,true>` | 前三项不命中且 `R*ceil_align(C*B,V)*4 <= U` | R-005 |
| 5 | R-005 normal | `21/22/24/28`; `SplitRowAndCol<T,true>` | 其余非空合法输入 | 无；key 必须属于四种字节宽 |

```text
if diagonal <= -R:                 mode=0, R-001
else if diagonal >= C-1:           mode=1, R-002
else if R*C*B <= V:                mode=3, R-003
else if R*ceil_align(C*B,V)*4 <= U: mode=4, R-004
else:                              mode=2, R-005
key = mode*10 + B
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 全零 | `1,2,4,8` | 950；`d<=-R`；key 个位等于 B | 单 AIV `tril` | `SplitAll<int{8,16,32,64}_t, OUTPUT_ZERO_MODE>` | 一个扩大为 `2*bufferSize` 的 TBuf；一次 Duplicate 后反复 MTE3 | 最后一核、最后 UB loop 按真实元素数写；无下一 Route |
| R-002 全拷贝 | `11,12,14,18` | 950；`d>=C-1` | 单 AIV `tril` | `SplitAll<T, OUTPUT_INPUT_MODE>` | 两个 TBuf ping-pong；每区 MTE2→MTE3→再次 MTE2 event | 最后一核与最后 loop 截断；无下一 Route |
| R-003 tiny | `31,32,34,38` | 950；非极值；`R*C*B<=V` | 单 AIV `tril` | `SplitTinyShape<T>` | 输入/输出 depth=2；32 B mask TBuf；整矩阵 VReg mask | `highTail` 截断批次；二维 mask 精确保护 |
| R-004 medium | `41,42,44,48` | 950；非极值/非 tiny；整矩阵四 buffer 可放 UB | 单 AIV `tril` | `SplitMediumShape<T,true>` | 输入/输出 depth=2；整行 DMA；head/mid/tail 行分段 | `highTail`；每行 `rowMask` 屏蔽对齐 lane |
| R-005 normal | `21,22,24,28` | 950 其余合法输入 | 单 AIV `tril` | `SplitRowAndCol<T,true>` | 三组 depth=2 queue/bind + zero TBuf；块按位置跳读/直拷/置零 | `rowActual/colActual`、former/tail core；无未知 key body |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 空 tensor | `selfRef->IsEmpty()` | 避免 `baseBlockNum=0` 后除以 `usedCoreNum=0`，也无需写回 | 全部 | Host 成功返回，无 kernel | `aclnnTrilGetWorkspaceSize` |
| dtype 字节宽 | `GetSizeByDataType(xDtype)!=0` 且 API RegBase 列表 | key 仅路由 1/2/4/8 B 的四种整数模板 | 全部 | API/tiling 失败或未知 key 无 body | `GetDtypeSupportList`; `Tiling4Trilu`; `triangulatorEntry` |
| 物理资源 | `AIV coreNum>0`, `UB>0`, 然后 `U=UB-8192` | blockDim 和所有 buffer 公式依赖正资源 | 全部 | Tiling 失败 | `TilingPrepareTrilForAscendC`; `GetPlatformInfo` |
| 全量搬运粒度 | `M*R*C*B<1024` 时单核，否则每核至少 `floor(1024/B)` 元素 | 避免全零/全拷贝 Route 产生过小的分核搬运块 | R-001/002 | Route 内调整核数，不换 Route | `OutputZeroAndInputTiling` |
| tiny VReg | `R*C*B<=V` | 每个矩阵用一个寄存器宽的数据区和预计算 mask | R-003 | R-004 | `ComputeTilingMode`; `SplitTinyShape::DoVectorCopy` |
| medium UB | `R*ceil_align(C*B,V)*4<=U` | depth=2 输入和输出共四份整行对齐 buffer | R-004 | R-005 | `ComputeTilingMode`; `ComputeMediumInfo` |
| normal 对齐 | `colInner` 按 `32/B` 或 `V/B` 对齐且不超 `bufferSize/B` | 保证二维 DMA stride 与 MicroAPI repeat 的块/VReg 粒度 | R-005 | 在 R-005 内缩小 tile | `ComputeNormalInfo`; `SplitRowAndCol::Init` |
| user workspace 非空 | `GetUserWorkspace(workspace)!=nullptr` | entry 的显式前置 guard，尽管 body 不消费 workspace 内容 | 全部 | Device 直接返回，不写 `y` | `triangulatorEntry` |

### POD/ABI

下表的字段顺序和类型来自 `BEGIN_TILING_DATA_DEF`。序列化宏 `TILING_DATA_FIELD_DEF`/`REGISTER_TILING_DATA_CLASS` 的定义不在固定仓，因此 `sizeof/alignof/offsetof` 不能由本仓逐字确认；实现复用必须保持声明顺序，不能仅按自然 C++ 布局猜测。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `TriangulatorTilingData` | `bufferSize` | `int64_t`/64 | 第1字段；精确 offset 固定仓不可见 | 字节 | `SaveToTilingData: baseInfoOp.bufferSize` | `SplitAll::ParseTilingData`，R-001 乘2、R-002原值 | 32 B floor-align |
| 同上 | `usedCoreNum` | `int64_t`/64 | 第2字段；offset 不可见 | core | `DoTiling`/`OutputZeroAndInputTiling` | `SplitAll::Init/Process` | `1..totalCoreNum` |
| 同上 | `normalCoreProcessNum` | `int64_t`/64 | 第3字段；offset 不可见 | 元素/core | 同上 | `SplitAll::Init` | 除最后核外 |
| 同上 | `tailCoreProcessNum` | `int64_t`/64 | 第4字段；offset 不可见 | 元素 | 同上 | `SplitAll::Init` | 最后一核 |
| `TriangulatorTinyTilingData` | `bufferSize` | `int64_t`/64 | 第1字段；offset 不可见 | 字节/queue slot | `ComputeTinyInfo` | `SplitTinyShape::InitBuffer` | depth=2，两队列 |
| 同上 | `planeArea` | `int64_t`/64 | 第2字段；offset 不可见 | 元素/矩阵 | `row*col` | `ComputeBlockInfo`、DMA | `planeArea*B<=V` |
| 同上 | `usedCoreNum` | `int64_t`/64 | 第3字段；offset 不可见 | core | `DoTiling` | `Init/Process` | 前余数核多一块 |
| 同上 | `normalCoreProcessNum` | `int64_t`/64 | 第4字段；offset 不可见 | base block/core | `baseBlockNum/used` | `Process` | floor 值 |
| 同上 | `tailCoreProcessNum` | `int64_t`/64 | 第5字段；offset 不可见 | core 数 | `baseBlockNum%used` | `Process` | 此处表示“多一块的核数” |
| 同上 | `highInner` | `int64_t`/64 | 第6字段；offset 不可见 | 矩阵/base block | `ComputeTinyInfo` | `ComputeBlockInfo` | 正值 |
| 同上 | `highOuter` | `int64_t`/64 | 第7字段；offset 不可见 | base block | `ComputeTinyInfo` | 尾块判断 | `ceil(M/highInner)` |
| 同上 | `highTail` | `int64_t`/64 | 第8字段；offset 不可见 | 矩阵 | `M%highInner` 或 `highInner` | 最后 base block DMA/repeat | `1..highInner` |
| 同上 | `highMask` | `uint64_t[4]`/256 | 第9字段；数组 offset 不可见 | 256 lane bit mask | `ComputeMask`，满足 `c<=r+d` 的 bit 置1 | `SplitTinyShape::Init`→`MaskReg` | 最大覆盖 256 bit；tiny 条件受 VReg 限制 |
| `TriangulatorMediumTilingData` | `bufferSize` | `int64_t`/64 | 第1字段；offset 不可见 | 字节/queue slot | `ComputeMediumInfo` | 两个 `InitBuffer` | 四份总预算 |
| 同上 | `row` | `int64_t`/64 | 第2字段；offset 不可见 | 行 | 输入 shape | DMA blockCount、head/mid/tail | 正值非空 |
| 同上 | `col` | `int64_t`/64 | 第3字段；offset 不可见 | 元素/行 | 输入 shape | DMA blockLen、VReg loop | 正值非空 |
| 同上 | `copyCols` | `int64_t`/64 | 第4字段；offset 不可见 | 元素 | `(headRows_raw>0?1:-headRows_raw+1)`（Tril 不减1） | 每个 mid row 的 select mask 起点 | 随行号递增 |
| 同上 | `usedCoreNum` | `int64_t`/64 | 第5字段；offset 不可见 | core | `DoTiling` | `Process` | 正值 |
| 同上 | `normalCoreProcessNum` | `int64_t`/64 | 第6字段；offset 不可见 | base block/core | `base/used` | `Process` | floor 值 |
| 同上 | `tailCoreProcessNum` | `int64_t`/64 | 第7字段；offset 不可见 | core 数 | `base%used` | `Process` | 前若干核多一块 |
| 同上 | `highInner` | `int64_t`/64 | 第8字段；offset 不可见 | 矩阵/base block | `ComputeMediumInfo` | `ComputeBlockInfo` | 正值 |
| 同上 | `highOuter` | `int64_t`/64 | 第9字段；offset 不可见 | base block | 同上 | 尾块判断 | `ceil(M/highInner)` |
| 同上 | `highTail` | `int64_t`/64 | 第10字段；offset 不可见 | 矩阵 | 同上 | 最后 block | `1..highInner` |
| 同上 | `headRows` | `int64_t`/64 | 第11字段；offset 不可见 | 行 | `max(-d,0)` | 全零头行 loop | 可能0 |
| 同上 | `midRows` | `int64_t`/64 | 第12字段；offset 不可见 | 行 | `R-headRows-tailRows` | 对角相交行 loop | 正常为正 |
| 同上 | `tailRows` | `int64_t`/64 | 第13字段；offset 不可见 | 行 | `max(R-C+ d,0)` | 全拷贝尾行 loop | 可能0 |
| `TriangulatorNormalTilingData` | `bufferSize` | `int64_t`/64 | 第1字段；offset 不可见 | 字节/slot | `ComputeNormalInfo` | 四类 UB 对象 | Host 按 U/8 保守切分 |
| 同上 | `row` | `int64_t`/64 | 第2字段；offset 不可见 | 行 | 输入 shape | tile 网格/地址 | 正值 |
| 同上 | `col` | `int64_t`/64 | 第3字段；offset 不可见 | 元素/行 | 输入 shape | tile 网格/DMA stride | 正值 |
| 同上 | `diagOffset` | `int64_t`/64 | 第4字段；offset 不可见 | 列偏移 | ACLNN attr | `ComputeBlockInfo` | 非极值 Route |
| 同上 | `usedCoreNum` | `int64_t`/64 | 第5字段；offset 不可见 | core | `DoTiling` | `Process` | 正值 |
| 同上 | `normalCoreProcessNum` | `int64_t`/64 | 第6字段；offset 不可见 | base block/core | `base/used` | `Process` | floor 值 |
| 同上 | `tailCoreProcessNum` | `int64_t`/64 | 第7字段；offset 不可见 | core 数 | `base%used` | `Process` | 前若干核多一块 |
| 同上 | `rowInner` | `int64_t`/64 | 第8字段；offset 不可见 | 行/tile | `ComputeNormalInfo` | tile 网格、地址、tail | `1..R` |
| 同上 | `colInner` | `int64_t`/64 | 第9字段；offset 不可见 | 元素/tile row | `ComputeNormalInfo` | tile 网格、stride、VReg loop | `1..C`，按块/VReg 对齐 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 全零 | `1/2/4/8` | `triangulator_tiling.cpp::ComputeTilingMode/DoTiling` | `TriangulatorTilingData` | `tril_apt.cpp::tril`→`triangulatorEntry<true>` | `split_all.h::SplitAll<T,0>` | `conversion/tril/CMakeLists.txt`; `tril_binary.json` |
| R-002 全拷贝 | `11/12/14/18` | 同上、`OutputZeroAndInputTiling` | `TriangulatorTilingData` | 同上 | `split_all.h::SplitAll<T,1>` | 同上 |
| R-003 tiny | `31/32/34/38` | `ComputeTinyInfo/ComputeMask` | `TriangulatorTinyTilingData` | 同上→`invokeTemplateSplitTinyShape` | `split_tiny_shape.h::SplitTinyShape<T>` | 同上；Tiling/body 由 `DEPENDENCIES triu` 复用 |
| R-004 medium | `41/42/44/48` | `ComputeMediumInfo` | `TriangulatorMediumTilingData` | 同上→`invokeTemplateSplitMediumShape<T,true>` | `split_medium_shape.h::SplitMediumShape<T,true>` | 同上 |
| R-005 normal | `21/22/24/28` | `ComputeNormalInfo` | `TriangulatorNormalTilingData` | 同上→`invokeTemplateSplitRowAndCol<T,true>` | `split_row_and_col.h::SplitRowAndCol<T,true>` | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 的单个 Device 节点都是：`tril` 把 task type 标为 `AIV_ONLY`，调用 `triangulatorEntry<true>`；entry 先取得 user workspace 并要求非空，再按 key 选择 `T=int8/16/32/64_t` 与实现类。模板参数 `IS_LOWER=true` 固化 Tril 方向。各类都以 `GetBlockIdx()` 映射 Host 的 `usedCoreNum`，从 GM `x` 搬到 UB/寄存器，按位保留或置零，再写 GM `y`。没有 Cube/AIC/MIX_CV、SIMT、NDDMA、跨 kernel workspace 边或数值 cast。

### Route R-001：全零 SplitAll

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/tril/op_api/aclnn_tril.cpp` | `aclnnInplaceTrilGetWorkspaceSize` | 非空、合法 dtype、2～8维；复用普通 Tril 图 | `l0op::Tril` |
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | `operatorType_=false && d<=-R` | mode=0，key=`B` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<true>` | key `1/2/4/8` | `SplitAll<T,OUTPUT_ZERO_MODE>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::Tril -> ADD_TO_LAUNCHER_LIST_AICORE(Tril)
  -> tril(x,y,workspace,tiling)
    -> triangulatorEntry<true>
      -> invokeTemplateSplitAll<T,OUTPUT_ZERO_MODE>
        -> SplitAll::Init
        -> SplitAll::Process
          -> Duplicate(zero buffer)
          -> SplitAll::ProcessPerLoop -> DataCopyPad(UB zero -> GM y)
```

##### 源码等价伪码

```text
# 伪码
bufBytes = 2 * pod.bufferSize
coreElems = (blockIdx == usedCoreNum-1 ? tailCoreProcessNum : normalCoreProcessNum)
gmBase = blockIdx * normalCoreProcessNum
loopCap = bufBytes / B
zeroBuf[0:loopCap] = 0
wait V -> MTE3 once
for loop in [0, ceil(coreElems/loopCap)):
    n = min(loopCap, coreElems-loop*loopCap)
    DataCopyPad(y[gmBase+loop*loopCap:], zeroBuf, blockCount=1, blockLen=n*B)
PipeBarrier(MTE3)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `y` | `blockIdx*normalCoreProcessNum + loop*loopCap` | 当前 `n` 元素 | T/flat ND | MTE3 | ViewCopy | executor 至写回 | 最后 loop 精确字节数 |
| `ubBuf1_` | UB base | `2*pod.bufferSize` B | T | `Duplicate(0)` | 每个 MTE3 loop | `TPipe`/kernel | 全循环复用，不重写 |
| user workspace | 框架首 slice，offset 不可见 | 16 MiB | bytes | executor | entry 仅判空 | 节点执行期 | 内容未使用 |

##### 分层内存布局

```mermaid
block-beta
  columns 1
  zeroBuf["ubBuf1 · TBuf · 2*bufferSize 字节 · 一次清零后只读"]
```

##### 字节账与复用顺序

```text
UB_total = 1 * (2*floor_align((Uphys-8192)/2,32))
UB_total <= Uphys-8192
dynamic_ub = N/A
workspace_total = 16,777,216 B（算子 Tiling slice）+ 框架不可见部分
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 整个 `ubBuf1_` | `Duplicate` | 所有 `ProcessPerLoop` MTE3 | kernel 结束前不覆盖 | 初始 `V_MTE3` flag/wait；结束 `PIPE_MTE3` barrier | buffer 始终保持零位模式 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零→写出 | `Duplicate`; `V_MTE3` flag/wait | Vector | 首次 MTE3 | zero TBuf | 按 T 位宽生成全零，无 cast |
| 多次写出完成 | `PipeBarrier<PIPE_MTE3>` | 每次 MTE3 | kernel 退出 | GM `y` | 每次 `blockLen=n*B`，最后 loop 不越界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode`; `OutputZeroAndInputTiling` | 极端 diagonal、1024 B 分核与 buffer 公式 |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<true>` key 1/2/4/8 | 字节宽路由到全零模板 |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_all.h` | `SplitAll<T,0>::Process/ProcessPerLoop` | UB 一次清零并分段写出 |

### Route R-002：全拷贝 SplitAll

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `conversion/tril/op_api/aclnn_tril.cpp` | `aclnnInplaceTrilGetWorkspaceSize` | 非空合法输入 | Tril node |
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | R-001 不命中且 `d>=C-1` | mode=1，key=`10+B` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry<true>` | `11/12/14/18` | `SplitAll<T,OUTPUT_INPUT_MODE>` |

#### Device 数据流

##### 函数级调用链

```text
tril -> triangulatorEntry<true>
  -> invokeTemplateSplitAll<T,OUTPUT_INPUT_MODE>
    -> SplitAll::Init(two TBuf)
    -> SplitAll::Process
      -> alternating ProcessPerLoop(buf1,event1) / ProcessPerLoop(buf2,event2)
        -> DataCopyPad(GM x -> UB) -> DataCopyPad(UB -> GM y)
```

##### 源码等价伪码

```text
# 伪码
loopCap = pod.bufferSize/B
allocate buf[2], each pod.bufferSize bytes
initialize two MTE3->MTE2 reuse flags
for loop in core range, alternate p=loop%2:
    wait MTE3->MTE2(event[p])
    copy x[coreBase+loop*loopCap : +n] -> buf[p]
    signal/wait MTE2->MTE3(event[p])
    copy buf[p] -> y[same range]
    signal MTE3->MTE2(event[p])
wait both reuse flags before exit; release event IDs
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x/y` | `blockIdx*normalCoreProcessNum + loop*loopCap` | `n` 元素 | T/flat ND | caller/MTE3 | MTE2/ViewCopy | executor | 同 offset 位拷贝 |
| `ubBuf1_` | UB slot0 | `bufferSize` B | T | MTE2 | MTE3 | `TPipe` | 偶数 loop，event 防提前覆盖 |
| `ubBuf2_` | UB slot1 | `bufferSize` B | T | MTE2 | MTE3 | `TPipe` | 奇数 loop，event 防提前覆盖 |
| user workspace | offset 不可见 | 16 MiB | bytes | executor | 仅判空 | 节点期 | 不读写 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  ping["ubBuf1 · TBuf · bufferSize 字节"]
  pong["ubBuf2 · TBuf · bufferSize 字节"]
```

##### 字节账与复用顺序

```text
UB_total = 2*floor_align((Uphys-8192)/2,32) <= Uphys-8192
dynamic_ub = N/A
workspace_total = 16,777,216 B + 框架不可见部分
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ping/pong 各自整区 | 当前 loop MTE2 | 同 loop MTE3 | 同 parity 下一 loop MTE2 | `MTE3_MTE2` 与 `MTE2_MTE3` flag/wait | 每个 event 与物理 buffer 一一对应 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB→GM | 两组 hard event | MTE2/MTE3 | 对向搬运引擎 | ping/pong | 纯位拷贝；最后 `blockLen=tail*B` |
| kernel 退出 | wait + release 两 event | 最后两次 MTE3 | 退出 | GM 写完成/UB 生命周期 | 不做 cast/round |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode`; `DoTiling` | 全拷贝极值和分核 |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | keys 11/12/14/18 | 路由到 mode 1 |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_all.h` | `SplitAll<T,1>` | 双 TBuf、双 event ping-pong |

### Route R-003：tiny 整矩阵掩码

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 非极值且 `R*C*B<=V` | mode=3 |
| Tiling | 同文件 | `ComputeTinyInfo`; `ComputeMask` | 按 UB 求 `highInner/highOuter/highTail`；`c<=r+d` 置 mask bit | Tiny POD/key `30+B` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | keys 31/32/34/38 | 字节宽 | `SplitTinyShape<T>` |

#### Device 数据流

##### 函数级调用链

```text
tril -> triangulatorEntry<true> -> invokeTemplateSplitTinyShape<T>
  -> SplitTinyShape::Init(mask TBuf + in/out depth=2 queues)
  -> Process
    -> ComputeBlockInfo -> CopyIn(next base block)
    -> DoVectorCopy(current): mask load/unpack -> zero -> masked source overwrite
    -> CopyOut(current)
```

##### 源码等价伪码

```text
# 伪码
maskBits = Host precomputed [bit(i*C+j)=1 iff j<=i+d]
each base block = highActual matrices, each padded to V bytes in UB
pipeline: prefetch first; for each subsequent block prefetch next before compute/copy current
for matrix in base block:
    vreg = load V/B lanes from padded x
    dstReg = 0
    masked_store(dstReg, vreg, expanded maskBits)  # only lower triangle overwrites zero
copy highActual matrices UB -> GM, dropping VReg padding by srcStride
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM base block | `offset*highInner*planeArea` | `highActual*planeArea` 元素 | T/连续矩阵 | caller/kernel | CopyIn/ViewCopy | executor | 最后一块 `highTail` |
| `inQueue_` | depth=2 | `2*bufferSize` B | 每矩阵 V 字节 padded | MTE2 | MicroAPI | `TPipe` | blockLen=`planeArea*B`; dstStride=`V/32-ceil(blockLen/32)` |
| `outQueue_` | depth=2 | `2*bufferSize` B | 同上 | MicroAPI | MTE3 | `TPipe` | srcStride 同上，GM 紧凑 |
| mask TBuf | 独立 32 B | 4×uint64 | bit mask | Host POD→Init | MaskReg load/unpack | kernel 全程 | 1/2/4/8 B dtype 逐级扩展 |
| RegTensor | VReg | `V/B` lanes | T | load/zero | masked store | 单矩阵迭代 | padding lane 不写 GM |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  mask["maskBuf · 32 B"]:1
  inQ["inQueue · depth=2 · 2*bufferSize B"]:2
  outQ["outQueue · depth=2 · 2*bufferSize B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*bufferSize + 2*bufferSize + 32
bufferSize = floor_align((Uphys-8192)/4,32)
UB_total <= Uphys-8160（8192 B 保留区吸收额外 32 B mask）
dynamic_ub = N/A
workspace_total = 16,777,216 B + 框架不可见部分
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | `CopyIn` | `DoVectorCopy` | `FreeTensor` 后下一 base block | `EnQue/DeQue/FreeTensor` | depth=2 支持 next/current 重叠 |
| outQueue slot | `DoVectorCopy` | `CopyOut` | `FreeTensor` 后 | queue 协议 | GM 写完才释放 |
| maskBuf | `Init::SetValue` | 每个 VEC scope | 不覆盖 | kernel 生命周期 | 所有矩阵共享同 R×C mask |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→Vector | inQueue | `CopyIn` | `DeQue` | padded input slot | `highActual` 控制 repeat；每矩阵逻辑区≤V |
| Vector select | `Duplicate` + mask load/unpack + masked `DataCopy` | source/zero regs | output queue | 每个矩阵 | 位保留/位零化，无算术精度变化 |
| Vector→MTE3 | outQueue | `EnQue` | `DeQue` | padded output slot | DMA `blockLen=planeArea*B` 丢弃 padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTinyInfo`; `ComputeMask` | 整矩阵批次与 256-bit mask |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitTinyShape<T>` | Tiny POD 与 body |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_tiny_shape.h` | `CopyIn/DoVectorCopy/CopyOut` | VReg padding、mask 扩展、双缓冲 |

### Route R-004：medium 整行分段

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 非极值、非 tiny，且 `R*ceil_align(C*B,V)*4<=U` | mode=4 |
| Tiling | 同文件 | `ComputeMediumInfo` | 求 `high*` 与 `headRows/midRows/tailRows/copyCols` | key `40+B` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitMediumShape<T,true>` | keys 41/42/44/48 | Tril medium body |

#### Device 数据流

##### 函数级调用链

```text
tril -> triangulatorEntry<true> -> invokeTemplateSplitMediumShape<T,true>
  -> SplitMediumShape::Init(in/out depth=2)
  -> Process
    -> ComputeBlockInfo -> CopyIn(highActual*row full rows)
    -> DoVectorCopy(head zero / mid Select / tail copy)
    -> CopyOut(compact rows)
```

##### 源码等价伪码

```text
# 伪码
for each assigned base block of highActual matrices:
    DMA every row: C*B GM bytes -> align(C, alignCount)*B UB bytes
    for each matrix:
        emit headRows rows as zero
        for i in [0,midRows):
            keep = copyCols+i
            for each VReg segment of row:
                src=load; dst=Select(src, zero, UpdateMask(keep)); store with rowMask
        emit tailRows rows as source
    DMA each UB row's C*B valid bytes -> compact GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM block | `offset*(R*C*highInner)` | `highActual*R*C` 元素 | T/连续矩阵 | caller/kernel | CopyIn/ViewCopy | executor | 最后 highTail |
| inQueue | depth=2 | `2*bufferSize` B | 行 padded UB | MTE2 | VEC | `TPipe` | `blockCount=highActual*R`, `blockLen=C*B` |
| outQueue | depth=2 | `2*bufferSize` B | 同 padding | VEC | MTE3 | `TPipe` | `srcStride=dstStride` |
| VRegs | VReg | zero/src/dst + masks | T | MicroAPI | UB store | 单 inner loop | `regCount=align(C,32/B)` if C≤V/B else V/B |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueue · depth=2 · 2*bufferSize B"]:2
  outQ["outQueue · depth=2 · 2*bufferSize B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*bufferSize + 2*bufferSize
bufferSize = floor_align((Uphys-8192)/4,32)
UB_total <= Uphys-8192
dynamic_ub = N/A
workspace_total = 16,777,216 B + 框架不可见部分
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | `CopyIn(next)` | `DoVectorCopy(current)` | free 后 | queue 协议 | depth=2；当前/下一 base block 分槽 |
| outQueue slot | `DoVectorCopy` | `CopyOut` | free 后 | queue 协议 | 有效行写回后释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 整行 DMA | in/out queue | MTE2/VEC | DeQue | padded row | 只提交 C 个有效元素 |
| head/mid/tail | `Duplicate`, `Select`, masked DataCopy | VRegs | output queue | 每行 | `rowMask` 抑制最后 VReg 无效 lane；无 cast |
| 批次尾 | `highActual=highTail` | Tiling | blockCount/loop | 最后 base block | 不读写超出 M |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeMediumInfo` | 四 buffer 上界和三段行数 |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitMediumShape<T,true>` | lower 模板固化 |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_medium_shape.h` | `DoVectorCopy` | 整行 DMA、head/mid/tail MicroAPI |

### Route R-005：normal 行列分块

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeTilingMode` | 前四条件均不满足 | mode=2 |
| Tiling | 同文件 | `ComputeNormalInfo` | 按列宽、R≤128 分支和 UB 求 `rowInner/colInner` | key `20+B` |
| Router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitRowAndCol<T,true>` | keys 21/22/24/28 | normal body |

#### Device 数据流

##### 函数级调用链

```text
tril -> triangulatorEntry<true> -> invokeTemplateSplitRowAndCol<T,true>
  -> SplitRowAndCol::Init(three depth=2 queue groups + zero TBuf)
  -> Process
    -> Duplicate(zeroBuf)
    -> ComputeBlockInfo(position, offsets, actual row/col)
    -> CopyIn: skip-zero / inQueue-diagonal / boundQueue-copy
    -> DoCopyVector: only diagonal-intersecting tiles
    -> CopyOut: zeroBuf / outQueue / boundQueue
```

##### 源码等价伪码

```text
# 伪码
zeroBuf[:] = 0; wait V->MTE3
for each assigned (matrix,rowTile,colTile):
    rowActual/colActual = tail-aware sizes
    gmOffset = matrix*R*C + rowTile*rowInner*C + colTile*colInner
    headRows = colTile*colInner - (rowTile*rowInner+d)
    tailRows = rowActual-colActual-headRows
    position = upper-right if headRows>=rowActual
               lower-left if tailRows>=rowActual else on-line
    if upper-right: skip CopyIn; CopyOut zeroBuf
    elif lower-left: DMA x -> bound inOutQueue -> DMA unchanged to y
    else:
        DMA x tile -> inQueue
        for head rows: zero
        for diagonal rows: Select(src,zero,keepMask(copyCols+i))
        for tail rows: copy src
        outQueue -> y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM tile | `matrix*R*C + rowTile*rowInner*C + colTile*colInner` | `rowActual*colActual` 有效元素 | T/strided rows | caller/kernel | CopyIn/ViewCopy | executor | `srcStride=(C-colActual)*B` |
| `inQueue_` | depth=2 | `2*bufferSize` B | diagonal tile | MTE2 | VEC | `TPipe` | only ON_LINE |
| `outQueue_` | depth=2 | `2*bufferSize` B | diagonal tile | VEC | MTE3 | `TPipe` | only ON_LINE |
| `inOutQueue_` | bound depth=2 | `2*bufferSize` B | full-copy tile | MTE2 | MTE3 | `TPipe` | lower-left tile 不经 VEC |
| `zeroBuf_` | TBuf | `bufferSize` B | zero tile | Duplicate | MTE3 | kernel | upper-right tiles共享 |
| VRegs | VReg | zero/src/dst/masks | T | MicroAPI | outQueue | 当前 tile/row | `regCount` 为 block/VReg 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 7
  inQ["inQueue · depth=2 · 2*bufferSize"]:2
  outQ["outQueue · depth=2 · 2*bufferSize"]:2
  bindQ["inOutQueue · depth=2 · 2*bufferSize"]:2
  zero["zeroBuf · bufferSize"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*bufferSize + 2*bufferSize + 2*bufferSize + bufferSize
         = 7*floor_align((Uphys-8192)/8,32)
         <= 7/8*(Uphys-8192)
dynamic_ub = N/A
workspace_total = 16,777,216 B + 框架不可见部分
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| inQueue slot | ON_LINE `CopyIn` | `DoCopyVector` | free 后 | queue | 只用于相交 tile |
| outQueue slot | `DoCopyVector` | ON_LINE `CopyOut` | free 后 | queue | Vector 完成后 EnQue |
| inOutQueue slot | lower-left `CopyIn` | lower-left `CopyOut` | free 后 | bound queue | 不经过计算，位模式不变 |
| zeroBuf | 初始 Duplicate | 所有 upper-right `CopyOut` | 不覆盖 | 一次 `V_MTE3` wait | 只读共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| zero 初始化 | `Duplicate` + `V_MTE3` flag/wait | VEC | MTE3 | zeroBuf | 位零化，无精度转换 |
| diagonal tile | in/out queue + `Select` | MTE2/VEC | DeQue | tile | `rowMask` 与 `selectMask`；`rowActual/colActual` 控制 DMA tail |
| full-copy tile | `TQueBind` | MTE2 | MTE3 | 同一 queue slot | 无 VEC、无 cast |
| 分核 tail | 前 `tailCoreProcessNum` 核多一 tile | Host POD | `Process` | base block 域 | 精确覆盖 M×rowOuter×colOuter |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `ComputeNormalInfo` | 行列 tile 与 1/8 UB 预算 |
| Kernel entry/router | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `invokeTemplateSplitRowAndCol<T,true>` | lower normal 路由 |
| Device body/helper | `conversion/triu/op_kernel/arch35/split_row_and_col.h` | `ComputeBlockInfo/CopyIn/DoCopyVector/CopyOut` | 空读跳过、直通队列、对角 Select |

## 算子特定优化

### Trick T-001：极端 diagonal 退化为纯清零或纯搬运

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：Tril 的 `diagonal<=-row` 或 `diagonal>=col-1`
- Golden 来源：[`conversion/triu/op_host/arch35/triangulator_tiling.cpp:198-216`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.cpp#L198-L216)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
void TriangulatorTiling::ComputeTilingMode()
{
    OP_LOGD(nodeName_, "[Triangulator] ComputeTilingMode start running.");
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
}
```

#### 机制与设计原因

Host 在任何 shape/UB 细分之前识别数学退化。全零 body 不读 `x`，只清零一个大 TBuf 后多次写 GM；全拷贝 body 不执行逐元素比较或掩码，只用两个 TBuf 搬运。两者还用 `OutputZeroAndInputTiling` 保证大输入每核常规块至少约 1024 B，避免把纯搬运切成过碎任务。源码可证明减少输入搬运（全零）或 Vector 计算（全拷贝），不代表已实测性能幅度。

#### 迁移边界

- 适用：输出能由属性与二维边界静态判定为全常量或原输入的三角类操作。
- 前提：偏移比较使用同一 `row/col` 定义，空 tensor 已在 Host 短路，dtype 零位模式符合语义。
- 失效：偏移位于 `(-row, col-1)` 时必须进入 R-003/004/005，不能近似成极值 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码用严格的边界互补条件区分两种退化；把 `diagonal==-row+1` 错纳入全零或把 `diagonal==col-2` 错纳入全拷贝会改变仍与矩阵相交的对角线元素。

### Trick T-002：tiny Route 由 Host 预计算一次二维 mask

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：非极端 diagonal 且 `row*col*dtypeBytes<=vRegSize`
- Golden 来源：[`conversion/triu/op_host/arch35/triangulator_tiling.cpp:238-251`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_host/arch35/triangulator_tiling.cpp#L238-L251)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
void TriangulatorTiling::ComputeMask()
{
    OP_LOGD(nodeName_, "[Triangulator] ComputeMask start running.");
    uint8_t idx = 0;
    for (auto r = 0; r < baseInfoOp.row; ++r) {
        for (auto c = 0; c < baseInfoOp.col; ++c) {
            if ((!operatorType_ && c <= r + baseInfoOp.diagOffset) ||
                (operatorType_ && c >= r + baseInfoOp.diagOffset)) {
                splitCoreOp.highMask[idx / MASK_BIT_WIDTH] |= 1UL << (idx % MASK_BIT_WIDTH);
            }
            ++idx;
        }
    }
}
```

#### 机制与设计原因

同一 `(R,C,d)` mask 对所有 `matrixNum` 复用。Host 把 256 bit mask 放进 POD，Device 每核在 `Init` 时只复制到 32 B mask TBuf；MicroAPI 根据 T 的字节宽加载或逐级 `MaskUnPack`，每个矩阵先写零，再让 mask 为真的 lane 用源值覆盖。这样 Device 内层不重复计算行列坐标与 diagonal 比较，且整矩阵通过 VReg padding 保持固定后移地址。

#### 迁移边界

- 适用：单个二维平面不超过一个 VReg、批次共享同一对角线属性的位选择问题。
- 前提：mask 容量、lane 顺序和 dtype 扩展规则保持一致；GM 只写 `planeArea` 有效元素。
- 失效：二维平面超过 VReg 时转 R-004/005；不能继续使用固定 4×uint64 mask。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码的 Tiny body对 2/4/8 字节元素分别执行 1/2/3 次 mask unpack；直接把 1-byte mask lane 布局用于更宽 dtype 会让元素与 mask bit 错位。

### Trick T-003：normal Route 按 tile 相对对角线位置跳读、直通或计算

#### 触发与源码依据

- 适用 Route：R-005
- 触发条件：前四 Route 不命中；每个 tile 经 `ComputeBlockInfo` 分类为 `UPPER_RIGHT`、`ON_LINE` 或 `LOWER_LEFT`
- Golden 来源：[`conversion/triu/op_kernel/arch35/split_row_and_col.h:199-220`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/triu/op_kernel/arch35/split_row_and_col.h#L199-L220)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::CopyIn(NormalBlockInfo& info)
{
    if (info.position == UPPER_RIGHT && IS_LOWER || info.position == LOWER_LEFT && !IS_LOWER) {
        return;
    }

    DataCopyExtParams copyInParam = {
        static_cast<uint16_t>(info.rowActual), static_cast<uint32_t>(info.colActual * sizeof(T)),
        static_cast<uint32_t>((col_ - info.colActual) * sizeof(T)), static_cast<uint32_t>(info.stride),
        static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> copyInPadParam = {false, 0, 0, 0};
    if (info.position == ON_LINE) {
        LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam, copyInPadParam);
        inQueue_.EnQue(xLocal);
    } else {
        LocalTensor<T> xLocal = inOutQueue_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam, copyInPadParam);
        inOutQueue_.EnQue(xLocal);
    }
}
```

#### 机制与设计原因

对 Tril，完整位于对角线右上的 tile 直接跳过 GM→UB，CopyOut 从一次清零的 `zeroBuf_` 写零；完整位于左下的 tile 经 `TQueBind` 原样 MTE2→MTE3，不进入 Vector；只有与对角线相交的 tile 才分配普通 input/output queue 并运行 `Select`。该分类同时减少不必要输入搬运、Vector 指令和中间物化，而且三类 tile 的 queue 生命周期彼此独立。

#### 迁移边界

- 适用：规则边界可把二维 tile 明确分类为全保留、全丢弃、边界相交的掩码算子。
- 前提：`headRows/tailRows` 与 tile 的 GM 行 stride、对角偏移定义一致；zero buffer 在首个 MTE3 前完成 Vector 清零。
- 失效：相交 tile 仍必须走 `inQueue→Select→outQueue`，不能仅按 tile 左上角分类。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码只对“整个 tile 位于清零区”的位置跳过 CopyIn；若把 `ON_LINE` tile 也跳读，保留区的原值将不可恢复。

## 模板与实现对比

| 对比维度 | R-001 全零 | R-002 全拷贝 | R-003 tiny | R-004 medium | R-005 normal | 设计选择依据 |
|---|---|---|---|---|---|---|
| 命中条件/优先级 | `d<=-R`，最高 | `d>=C-1`，第二 | 平面≤VReg | 整矩阵四 buffer 可驻 UB | fallback | 先数学退化，再按 VReg/UB 资源选实现 |
| 模板与 dtype | `SplitAll<T,0>` | `SplitAll<T,1>` | `SplitTinyShape<T>` | `SplitMediumShape<T,true>` | `SplitRowAndCol<T,true>` | T 仅按 1/2/4/8 B 位宽实例化 |
| 循环与分核 | 元素/UB loop，最后核 tail | 同左 | base block×整矩阵 | base block×矩阵×行 | matrix×rowTile×colTile | 工作域随可驻留粒度变化 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 1 TBuf | 2 TBuf | 两个 depth=2 queue+maskBuf+VRegs | 两个 depth=2 queue+VRegs | 三个 depth=2 queue/bind+zeroBuf+VRegs | 全部 dynamic UB/通信 window=N/A |
| 内存总占用与复用 | 约 U，一个 buffer只读 | 约 U，ping-pong | `4*buf+32` | `4*buf` | `7*buf`，Host按 U/8 | 8192 B Host reserve 覆盖额外对象 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Duplicate+MTE3 | MTE2/MTE3 | padded DMA+masked MicroAPI | 整行 DMA+Select | 2D DMA+skip/bind/Select | 均 AIV；无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | V→MTE3；最后 loop | 双 event；最后 loop | queue；highTail+mask | queue；highTail+rowMask | queue/event；row/col/core tail | 全部位拷贝/零化，无 cast/round |
| fallback/失败边界 | 无 | 无 | 超 VReg→R-004 | 超 UB→R-005 | unknown key 无 body | 空 workspace entry直接返回；资源/tiling失败阻断 launch |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档写支持，但指定 `conversion/tril/` 固定范围缺少 A2 Host/Tiling/Kernel 构建闭包，未发布卡 | 同左 | 支持；公开表、`ascend950` OpDef/config/CMake 与 arch35 AIV 实现闭合 |
| Host/Launcher | 固定范围不可见 | 固定范围不可见 | `Tiling4Trilu<false,false>`，五 Route，20 key，单 AIV node |
| Kernel/资源 | 固定范围不可见 | 固定范围不可见 | VReg/UB 自适应；`AIV_ONLY`；16 MiB user workspace只判空 |

## 限制与失败边界

- `selfRef` 必须非空指针、dtype 属于 950 RegBase 列表、维数为 2～8；空 tensor 是成功短路，不运行 Device。
- shape 前导维连乘、`row*col*B`、`M*R*C`、workspace 和 tile 计算都使用 `int64_t`，固定源码未提供显式乘法溢出检查；超大 shape 的行为不得外推。
- API 文档声明非连续输入支持，但 `Contiguous`、`ViewCopy` 的具体搬运 kernel、workspace slice、stream event 和异常清理属于固定仓外框架边界。
- `TILING_DATA_FIELD_DEF` 的宏定义不在固定仓，四种 POD 的精确 `sizeof/alignof/offsetof` 不可见；只能确认字段声明顺序、类型和 producer/consumer。
- `DataCopyExtParams`、MicroAPI mask/DataCopy 的库内默认字段、硬件 lane 细节和 primitive 实体不在固定仓；本卡只陈述算子源码显式赋值及调用顺序。
- entry 要求 `GetUserWorkspace(workspace)` 非空，否则无错误码地直接返回且不写输出；16 MiB slice 本体不被 body 使用，不能擅自删除该 ABI/生命周期要求。
- router 对未知 key 没有显式 error/fallback 分支；正确性依赖 Host 只产生 mode `{0,1,2,3,4}` 与字节宽 `{1,2,4,8}` 的组合。
- 950 Complex32/Complex64、浮点、整数和 bool 通过 `DtypeByte` 共用整数位宽模板；实现是位选择/位零化，无数值 cast。primitive 对非标准位模式的更深语义固定仓不可见。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/tril/docs/aclnnTril&aclnnInplaceTril.md` | `aclnnInplaceTril*` | 公开原地语义、产品、dtype、shape、两段式接口 |
| 入口声明/约束 | `conversion/tril/op_api/aclnn_tril.h` | `aclnnInplaceTrilGetWorkspaceSize`; `aclnnInplaceTril` | C ABI 原型 |
| 入口实现 | `conversion/tril/op_api/aclnn_tril.cpp` | `GetDtypeSupportList`; `CheckParams`; 两个 Inplace 函数 | 950 dtype、校验、executor 图复用、第二段 |
| 入口实现 | `conversion/tril/op_api/tril.cpp` | `l0op::Tril` | 独立输出分配与 AICore launcher 添加 |
| Host/Tiling/Launcher | `conversion/tril/op_host/tril_def.cpp` | `ops::Tril` | dtype/format、diagonal attr、ascend950 config |
| Host/Tiling/Launcher | `conversion/tril/op_host/arch35/tril_tiling.cpp` | `IMPL_OP_OPTILING(Tril)` | Tril 固化为 `<false,false>`，复用 Triu 公共 Tiling |
| Host/Tiling/Launcher | `conversion/triu/op_host/arch35/triu_tiling.h` | `Tiling4Trilu`; `GetDiagonalValue` | shape/dtype/diagonal producer |
| Host/Tiling/Launcher | `conversion/triu/op_host/arch35/triangulator_tiling.h` | 四个 TilingData | POD 字段与注册 key |
| Host/Tiling/Launcher | `conversion/triu/op_host/arch35/triangulator_tiling.cpp` | `TriangulatorTiling::*` | Route、tile/core、UB/workspace、POD producer |
| Kernel entry/body | `conversion/tril/op_kernel/tril_apt.cpp` | `tril`; `triangulatorEntry<true>` | AIV_ONLY entry 与 lower 参数 |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/triangulator_base.h` | `triangulatorEntry`; invoke helpers | 20 key 到模板/body 的 router |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_all.h` | `SplitAll<T,MODE>` | 全零与全拷贝实现 |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_tiny_shape.h` | `SplitTinyShape<T>` | Tiny mask、VReg、DMA 与双缓冲 |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_medium_shape.h` | `SplitMediumShape<T,true>` | 整行 medium pipeline |
| Kernel entry/body | `conversion/triu/op_kernel/arch35/split_row_and_col.h` | `SplitRowAndCol<T,true>` | normal tile 分类、搬运、Select |
| 构建接线 | `conversion/tril/op_host/config/ascend950/tril_binary.json` | `DtypeByte` entries | 950 四种字节宽二进制匹配 |
| 构建接线 | `conversion/tril/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`; `add_all_modules_sources` | ascend950/arch35、依赖 triu、ACLNN sources |
| 构建接线 | `cmake/func.cmake` | `add_all_modules_sources`; `add_dependent_ops` | op_api/tiling/opdef 加载及 triu 依赖接入 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 4 个 Route POD、mode×10+B key 编码可复用 | 字段顺序/64位类型、Tiny `highMask[4]`、不同 mode 的 POD 类型必须与 router 一致 | `triangulator_tiling.h`; `ComputeTilingKey`; `triangulatorEntry` |
| 调用顺序 | `Contiguous→Tril→ViewCopy` 与单 AIV node 可复用 | 原地写回不能改成未经别名证明的 x/y 同址 kernel；空 tensor Host 短路 | `aclnnTrilGetWorkspaceSize`; `l0op::Tril` |
| workspace/内存 | 五 Route 的 buffer 预算与 16 MiB user slice可复用 | 8192 B reserve、depth 乘数、Tiny额外32 B mask、entry非空检查不能遗漏 | `GetPlatformInfo`; `PostTiling`; 各 body `InitBuffer` |
| 同步 | SplitAll events、queue EnQue/DeQue/FreeTensor、zeroBuf V→MTE3顺序可复用 | buffer 未消费前不得覆盖；bound queue 只用于直通 tile | `SplitAll::Process`; 各 `Process/Copy*` |
| 精度与尾块 | 同字节宽位搬运/选择、`highTail`、row/col/core tail 可复用 | 不得加入数值 cast；必须屏蔽 VReg/UB padding，GM 只提交有效字节 | `ComputeBlockInfo`; `DoVectorCopy`; `DataCopyPad` 参数 |
| 失败边界 | Host检查、Tiling资源检查、unknown key无 body、workspace空直接返回 | 不得把固定仓不可见的 executor/primitive行为推测成成功 fallback | `CheckParams`; `RunTriangulatorTilingAscendC`; `triangulatorEntry` |