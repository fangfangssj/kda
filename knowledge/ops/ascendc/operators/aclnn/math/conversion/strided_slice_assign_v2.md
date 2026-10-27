---
type: Operator
title: "StridedSliceAssignV2 A2/A3 ACLNN 算子知识"
description: "说明 aclnnStridedSliceAssignV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/docs/aclnnStridedSliceAssignV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/docs/zh/context/两段式接口.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/cmake/gen_ops_info.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/scripts/util/tiling_data_def_build.py
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/config/ascend910b/strided_slice_assign_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/config/ascend910_93/strided_slice_assign_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_host/config/ascend950/strided_slice_assign_v2_binary.json
    role: implementation
created_at: "2026-08-25T09:33:46Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# StridedSliceAssignV2 A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnStridedSliceAssignV2GetWorkspaceSize` / `aclnnStridedSliceAssignV2`，CANN 9.1.0；固定仓未出现更高版本入口。 |
| 计算语义 | 将 `inputValue` 写入 `varRef` 由 `begin/end/strides/axesOptional` 指定的切片，切片外元素保持原值；不支持广播。 |
| 输入输出 | `varRef` 与 `inputValue` 为 ND、同 dtype；支持 FP16/FP32/BF16/INT32/INT64/DOUBLE/INT8，`varRef` rank 为 1–8；索引数组公开类型为 INT64。 |
| 执行类型 | 单节点 `AIV_VECTOR` 启动图；实际 body 只用 AIV 的 MTE2/MTE3 `DataCopyPad` 和硬事件，不执行 Vector 算术，也不含 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 原生元素宽度搬运（FP16/FP32/BF16/INT32/INT8）；R-002 64-bit 元素的 `int32_t` 视图等字节搬运（INT64/DOUBLE）。两者均为 key 1。 |
| 核心资源 | 输入/输出 GM、单个 195584-byte `TBuf<VECCALC>`；无 `TQue`、无动态 UB、无计算中间量；Tiling 声明 16 MiB 系统 workspace，但 Kernel 的 `workspace` 参数未被读取。 |
| 来源闭包 | `source_closed`：公开 ACLNN 契约、ACLNN OpDef 接线、Tiling/POD、key 1 entry、两条编译期 dtype body、A2/A3 binary config 与构建链已闭合；自动生成 ACLNN bridge 和框架原地输出绑定是明确的框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `V` 为 `varRef`，`X` 为 `inputValue`。若 `axesOptional` 缺省，则每个维度都由对应的 `begin[i]`、`end[i]`、`strides[i]` 描述；若提供 `axesOptional`，Host 先把未列出的轴补成 `[0, varDim[i], 1]`，再把列表项散射到指定轴。对每一维，负 `begin/end` 先加该维长度，再截断到 `[0, varDim[i]]`。有效切片形状必须满足：

```text
inputValueDim[i] = max(ceil_div(endAdjusted[i] - beginAdjusted[i], stridesAdjusted[i]), 0)
```

Kernel 将 `X[k0,...,k(r-1)]` 写到：

```text
V[begin[0] + k0*strides[0], ..., begin[r-2] + k(r-2)*strides[r-2], begin[r-1] + k(r-1)]
```

最后一维步长被 Host 强制为 1，因此每个外层坐标对应一段连续 GM 搬运。Kernel 不读取 `var` 输入，只覆盖输出 GM 的切片区；公开接口把 `varRef` 声明为输入/输出，但自动生成 bridge 如何把该可变对象绑定为 `var_out` 在固定仓不可见。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `varRef` | ND，rank 1–8；可非连续 | FP16/FP32/BF16/INT32/INT64/DOUBLE/INT8 | 可变输入/输出；每一维必须大于 0；切片外保持原值。OpDef 标记 `AutoContiguous()`。 |
| `inputValue` | ND，与 `varRef` rank 相同；可非连续 | 与 `varRef` 相同 | shape 必须精确等于归一化切片 shape，不广播。OpDef 标记 `AutoContiguous()`。 |
| `begin` | 一维索引列表 | ACLNN 为 INT64 | 负值按维长归一化并截断；固定 Host 还能读取内部 INT32 Tensor，但该能力不扩展公开 ACLNN 契约。 |
| `end` | 一维索引列表 | ACLNN 为 INT64 | 与 `begin` 同样归一化并截断；仅参与 Host 校验，不进入 Device POD。 |
| `strides` | 一维索引列表 | ACLNN 为 INT64 | 公开约束要求全部为正；Host 仅显式检查最后一维为 1，其他正值约束依赖公开入口校验。 |
| `axesOptional` | 可选一维索引列表 | ACLNN 为 INT64 | 缺省时直接按全维列表解释；存在时散射到对应轴，未列出轴取完整范围和步长 1。 |
| `workspaceSize` / `executor` | 标量输出 / opaque handle | `uint64_t` / `aclOpExecutor**` | 第一段返回；executor 包含一次执行流程。 |
| `workspace` / `stream` | Device 临时内存 / stream | opaque | 第二段使用；通用两段式契约禁止复用同一个 executor 重复执行。 |

### 入口与版本

固定 Commit 只公开 `aclnnStridedSliceAssignV2GetWorkspaceSize` 与 `aclnnStridedSliceAssignV2` 这一版本族。公开产品表明确支持 Atlas A2、A3，且 `ascend910b`、`ascend910_93` 的 OpDef 与 binary config 均有接线；两个 config 对七种数据 dtype 给出相同的专用 binary 列表。`op_host/CMakeLists.txt` 以 `ACLNNTYPE aclnn` 注册 OpDef，仓级 `add_modules_sources` 将其接到 ACLNN OpDef 对象；具体生成的 C API bridge 源文件不在固定仓，按框架边界处理。

公开 ACLNN 文档明确将 Ascend 950PR/950DT 标为不支持，而 README、OpDef 和 `ascend950` binary config 又存在支持信号；依 VV Skill 的平台冲突规则，本卡不覆盖 950，也不生成 `_950` 正式卡。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段公开接口接收可变 `varRef`、只读 `inputValue`、三个必选索引数组和可选 `axesOptional`，返回 workspace 大小和一次性 executor。公开文档列出空指针与非法 dtype 错误，并声明非连续 Tensor 可用；自动生成 bridge 中更精细的检查、`Contiguous/ViewCopy` 节点和返回码映射固定仓不可见。
2. OpDef 对所有输入、输出标记 `AutoContiguous()`；`begin/end/strides/axes` 同时标记 `ValueDepend(REQUIRED)`，使 Tiling 在 Host 侧取得常量值。
3. `InferShape4Elewise` 令输出 shape 等于 `var` shape，未知 rank 原样传播；`InferDataType4SameAsInput` 令输出 dtype 等于输入 0。
4. `Tiling4StridedSliceAssignV2` 取得 `var` 和 `input_value` storage shape，要求 rank 相同、rank 位于 `[1,8]` 且每个 `var` 维度大于 0。
5. `GetConstInputData` 读取 `begin/end/strides`；有 axes 时，`AdjustWithAxes` 先构造全切片，再按 `axes[i]` 写回；随后 `AdjustBegin`、`AdjustEnd` 做负索引和上下界归一化。
6. `CheckInputShape` 逐维比较 `max(CeilDiv(end-begin,stride),0)` 与 `inputValueDim`，然后强制 `strides[rank-1] == 1`。任一步失败都返回 `GRAPH_FAILED`，不产生 launch。
7. Tiling 计算累计 shape、写入 key 1、POD、AIV blockDim 和 workspace。第二段接口在传入 stream 上执行 executor；生成式 launcher 的精确函数名固定仓不可见。

### 参数、Workspace 与生命周期

`TilingPrepare4StridedSliceAssignV2` 从 `PlatformAscendC` 读取 `GetCoreNumAiv()` 和 UB 大小，二者小于等于 0 均失败。Tiling 使用 `totalCoreNum` 作为 `blockDim`，但只保存 UB 大小到 compile info，未据其裁剪 tile。Kernel 自身固定分配 `UB_SIZE = (192-1)*1024 = 195584` bytes。

POD 由栈上零初始化数组生成；有效 rank 以外的槽保持 0。`varCumShape[rank-1]` 与 `inputCumShape[rank-1]` 先写最后一维，循环仅计算索引 `rank-2 ... 1`，因为 Device 地址公式只读取 `cum[n+1]`。Host 将 `currentWorkspace[0]` 固定为 `16777216` bytes。该空间由执行框架在第二段调用期间持有；单一 Kernel entry 虽接收 `workspace`，body 不读取它，因此没有算子数据在其中生产或消费。ACLNN 层因非连续物化可能增加的 workspace、各 slice offset 以及 `varRef`/`var_out` 原地别名建立过程固定仓不可见。

### 启动图与失败传播

```text
aclnnStridedSliceAssignV2GetWorkspaceSize
  -> 生成式 ACLNN bridge（固定仓不可见）
    -> StridedSliceAssignV2 OpDef / InferShape / Tiling4StridedSliceAssignV2
      -> executor + workspaceSize
aclnnStridedSliceAssignV2(workspace, workspaceSize, executor, stream)
  -> 单节点 K-001 strided_slice_assign_v2
       grid/blockDim = GetCoreNumAiv()
       tilingKey = 1
       POD = StridedSliceAssignV2TilingData
       input = input_value GM；output = var_out GM
       workspace slice = [0, 16777216)，system workspace，Kernel 未读
```

仅有一个 Device 节点，因此没有 Kernel 间 workspace/GM 交接边。Host/Tiling 返回失败时 executor 不应启动 K-001；第二段执行错误按 `aclnnStatus` 传播，精确 launcher 返回链固定仓不可见。entry 对非 key 1 没有 `else`，会直接结束且不写输出；正式 Host 始终设置 key 1。

## Route 与参数设计

### 分核与资源计算

令 `r=dimNum`，`C=GetCoreNumAiv()`，`B=195584 bytes`，元素字节数为 `s=sizeof(T)`：

```text
outerRows = product(inputValueDim[0:r-1])              # 外层连续行数，单位：row
blockDim = C                                            # 单位：AIV core
core c handles row = c, c+C, c+2C, ... < outerRows
ubFactor = floor(B / s)                                 # 单位：逻辑 T 元素
innerRepTimes = ceil(inputValueDim[r-1] / ubFactor)     # 单位：DMA chunk
tailNum = inputValueDim[r-1] - (innerRepTimes-1)*ubFactor
copyBytes(j) = (j is last ? tailNum : ubFactor) * s
```

不存在 former/tail core POD；循环以 `totalIdx += usedCoreNum` 自然处理余数。每个 row 的 input 起点是 `sum(inputIdx[n]*inputCumShape[n+1])`，输出起点是 `begin[r-1] + sum((begin[n]+inputIdx[n]*strides[n])*varCumShape[n+1])`。最后一维 stride 为 1，所以 chunk 内地址连续。算子源码只显式赋值 `DataCopyExtParams.blockCount=1` 与逐 chunk 的 `blockLen=copyBytes`；`srcStride/dstStride` 以及两个 `DataCopyPadExtParams` 的默认初始化语义来自仓外 `kernel_operator.h`，固定仓不可见。tail 的有效搬运长度由 `blockLen` 精确缩短。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 原生元素宽度搬运 | binary dtype specialization + key 1；`KernelStridedSliceAssignV2<DTYPE_VAR>` | A2/A3 binary config 选中 FP16/FP32/BF16/INT32/INT8，Tiling 无条件设置 key 1；实例内 `sizeof(T) != 8` | 若 dtype 为 INT64/DOUBLE，编译期进入 R-002；不支持 dtype 在 ACLNN/OpDef 阶段失败。 |
| 2 | R-002 64-bit 等字节搬运 | binary dtype specialization + key 1；`KernelStridedSliceAssignV2<DTYPE_VAR>` | A2/A3 binary config 选中 INT64/DOUBLE，Tiling 无条件设置 key 1；实例内 `sizeof(T) == 8` | 无下一 Route；不支持 dtype 失败。 |

```text
A2/A3 公开 ACLNN dtype
  -> binary config 选择 DTYPE_VAR
    -> Tiling4StridedSliceAssignV2: key = 1
      -> entry TILING_KEY_IS(1)
        -> if constexpr(sizeof(DTYPE_VAR) == 8)
             true  -> R-002 int32_t 视图 + DILATION=2
             false -> R-001 原生 T 视图
  -> 其他 dtype：ACLNN/OpDef 拒绝
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 原生元素宽度搬运 | key 1 + `sizeof(T)!=8` | A2/A3；FP16/FP32/BF16/INT32/INT8；rank 1–8；shape/stride 校验通过 | 单节点 K-001 AIV | `KernelStridedSliceAssignV2<T>::Process` 的 `GlobalTensor<T>` 分支 | 195584-byte `valueBuf.Get<T>()`；每 chunk MTE2→MTE3，再 MTE3→MTE2 复用 | 最后 chunk 用 `tailNum*sizeof(T)`；无 Device fallback。 |
| R-002 64-bit 等字节搬运 | key 1 + `sizeof(T)==8` | A2/A3；INT64/DOUBLE；其余 shape/stride 条件同 R-001 | 单节点 K-001 AIV | 同一模板的 `GlobalTensor<int32_t>` 编译期分支，逻辑 offset 乘 2 | 同一 195584-byte TBuf，以 int32 LocalTensor 访问；同一硬事件序列 | 逻辑尾元素数乘 8 得字节数；无 Device fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| rank 与正维 | `rank(var)==rank(inputValue)`、`1<=rank<=8`、每个 `varDim[i]>0` | POD 仅有 8 槽；地址函数使用 `rank-1` 和累计 shape | R-001/R-002 | `GRAPH_FAILED` | `strided_slice_assign_v2_tiling.cpp::Tiling4StridedSliceAssignV2` |
| shape 精确匹配 | 对每维 `max(CeilDiv(end-begin,stride),0)==inputValueDim[i]` | Kernel 不广播，只逐 row 等长覆盖 | R-001/R-002 | `GRAPH_FAILED` | `strided_slice_assign_v2_tiling.cpp::CheckInputShape` |
| 最后一维连续 | `strides[rank-1] == 1` | 每次 `DataCopyPad` 把最后一维 chunk 当连续字节段 | R-001/R-002 | `GRAPH_FAILED` | `strided_slice_assign_v2_tiling.cpp::Tiling4StridedSliceAssignV2` |
| AIV 核数 | `GetCoreNumAiv()>0` | blockDim 与 core-cyclic row 映射要求非零 | R-001/R-002 | Tiling prepare 失败 | `strided_slice_assign_v2_tiling.cpp::TilingPrepare4StridedSliceAssignV2` |
| UB 固定容量 | Host 只检查平台 UB `>0`；源码实际隐含要求 `platform UB >= 195584` | `InitBuffer(valueBuf,195584)`；Host 读取的 `ubSize` 未参与 selector | R-001/R-002 | 固定仓无 fallback，资源不足行为不可由源码闭合 | `strided_slice_assign_v2.h::Init` |
| 64-bit 视图 | `sizeof(T)==8`，逻辑 GM offset 乘 `DILATION=2` | MTE 按 int32 地址索引，但 `blockLen` 仍为逻辑元素数乘 8 bytes | R-002 | 编译期 false 时进入 R-001 | `strided_slice_assign_v2.h::Init/Process` |
| key | Host 恒写 `tilingKey=1` | entry 只实例化 key 1 | R-001/R-002 | 未知 key 静默退出、不写输出 | `strided_slice_assign_v2.cpp::strided_slice_assign_v2` |

### POD/ABI

`scripts/util/tiling_data_def_build.py::gen_tiling` 将宏定义按声明顺序生成 `#pragma pack(1)` Device struct。全部字段均为 `int64_t`，因此序列化 blob 共 392 bytes；Device struct `alignof=1`。Host 宏类本身的 C++ `sizeof/alignof` 定义位于仓外运行时头，固定仓不可见，但 `SaveToBuffer/GetDataSize` 与生成脚本共同约束下列 wire offsets。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `StridedSliceAssignV2TilingData` | `dimNum` | `int64_t`/64 | offset 0，extent 8；packed struct sizeof 392 | 维 | `tiling.set_dimNum(static_cast<int64_t>(dimNum))` | `Init`: 控制数组索引、循环和地址计算 | `[1,8]` |
| 同上 | `varDim[8]` | `int64_t[8]`/512 | offset 8，extent 64 | 元素/维 | `tiling.set_varDim(varDim)` | `Init` 绑定 `varShape`，body 后续未读取 | 有效 `[0,dimNum)`；其余为 0 |
| 同上 | `inputValueDim[8]` | `int64_t[8]`/512 | offset 72，extent 64 | 元素/维 | `tiling.set_inputValueDim(inputValueDim)` | `Init/Process/CalcIdx`：outerRows、最后一维 tile、坐标反解 | 有效 `[0,dimNum)`；其余为 0 |
| 同上 | `begin[8]` | `int64_t[8]`/512 | offset 136，extent 64 | 元素索引 | `AdjustWithAxes`、`AdjustBegin` 后 `tiling.set_begin(begin)` | `CalcVarOffset` | 已截断到 `[0,varDim[i]]`；其余为 0 |
| 同上 | `strides[8]` | `int64_t[8]`/512 | offset 200，extent 64 | 元素步长 | 可选 axes 展开后 `tiling.set_strides(strides)` | `CalcVarOffset`，只读取前 `dimNum-1` 维 | 公开约束为正；最后一维必须 1；其余槽为 0 |
| 同上 | `varCumShape[8]` | `int64_t[8]`/512 | offset 264，extent 64 | 元素 | `varCumShape[last]=varDim[last]`；向前计算到索引 1；`set_varCumShape` | `CalcVarOffset` 读取 `[n+1]` | offset 0 不参与消费；未写槽为 0；乘法溢出未检查 |
| 同上 | `inputCumShape[8]` | `int64_t[8]`/512 | offset 328，extent 64 | 元素 | 与 `varCumShape` 同法；`set_inputCumShape` | `CalcInputOffset` 读取 `[n+1]` | offset 0 不参与消费；未写槽为 0；乘法溢出未检查 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 原生元素宽度搬运 | binary dtype + key 1 + `sizeof(T)!=8` | `op_host/strided_slice_assign_v2_tiling.cpp::Tiling4StridedSliceAssignV2` | `StridedSliceAssignV2TilingData`，`DTYPE_VAR` | `op_kernel/strided_slice_assign_v2.cpp::strided_slice_assign_v2` | `op_kernel/strided_slice_assign_v2.h::KernelStridedSliceAssignV2<T>::Init/Process/Calc*` | `op_host/CMakeLists.txt::add_modules_sources`；A2/A3 binary JSON；`cmake/gen_ops_info.cmake::prepare_compile_from_config` |
| R-002 64-bit 等字节搬运 | binary dtype + key 1 + `sizeof(T)==8` | 同 R-001 | 同一 POD；`DTYPE_VAR` 为 INT64/DOUBLE，`DILATION=2` | 同一 entry/key 1 | 同一模板的 `GlobalTensor<int32_t>` / `LocalTensor<int32_t>` 分支 | 同 R-001，binary 分别为 `Int64_Int64`、`Double_Int64` |

## Kernel 实现

### 公共执行流程

所有 Route 共用单节点 K-001。entry 构造 `TPipe`、以 `GET_TILING_DATA` 解析 392-byte POD，只在 key 1 构造 `KernelStridedSliceAssignV2<DTYPE_VAR>` 并依次调用 `Init`、`Process`。`Init` 取得实际 `GetBlockNum()`，绑定 POD 指针和输入/输出 GM，分配单个 195584-byte `TBuf<VECCALC>`，根据元素宽度计算最后一维 chunk。`Process` 把最后一维之前的笛卡尔积展平为 row，AIV 核循环领取 row；每个 chunk 执行 GM→TBuf，等待 MTE2→MTE3，再 TBuf→GM，等待 MTE3→MTE2 后复用。没有 Vector 算术、跨核通信或多 Kernel 边。

### Route R-001：原生元素宽度搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnStridedSliceAssignV2.md` | `aclnnStridedSliceAssignV2GetWorkspaceSize` | dtype 属于 FP16/FP32/BF16/INT32/INT8 且其余公开约束成立 | 进入 StridedSliceAssignV2 executor |
| Host config | `op_host/config/ascend910b\|ascend910_93/strided_slice_assign_v2_binary.json` | 对应 dtype `bin_filename` | A2/A3 + var/input 同 dtype + indices INT64 | 编译/装载相应 `DTYPE_VAR` binary |
| Tiling | `op_host/strided_slice_assign_v2_tiling.cpp` | `Tiling4StridedSliceAssignV2` | shape 校验通过；无 Route 分支 | `key=1`、`blockDim=totalCoreNum` |
| Router | `op_kernel/strided_slice_assign_v2.cpp` | `strided_slice_assign_v2` | `TILING_KEY_IS(1)` | 实例化 `KernelStridedSliceAssignV2<DTYPE_VAR>` |
| Body | `op_kernel/strided_slice_assign_v2.h` | `Init/Process` | `if constexpr(sizeof(T)==sizeof(int64_t))` 为 false | 原生 `GlobalTensor<T>` / `LocalTensor<T>` 路径 |

#### Device 数据流

##### 函数级调用链

```text
Tiling4StridedSliceAssignV2 -> key=1, blockDim=GetCoreNumAiv(), POD
  -> strided_slice_assign_v2(..., input_value, ..., var_out, workspace, tiling)
    -> GET_TILING_DATA
    -> KernelStridedSliceAssignV2<T>::Init
       -> GlobalTensor<T>::SetGlobalBuffer
       -> TPipe::InitBuffer(valueBuf, 195584)
    -> KernelStridedSliceAssignV2<T>::Process
       -> CalcIdx -> CalcInputOffset -> CalcVarOffset
       -> DataCopyPad(GM input -> LocalTensor<T>)
       -> SetFlag/WaitFlag<MTE2_MTE3>
       -> DataCopyPad(LocalTensor<T> -> GM output)
       -> SetFlag/WaitFlag<MTE3_MTE2>
```

##### 源码等价伪码

```text
# 源码等价伪码
C = GetBlockNum()
B = 195584
ubFactor = B / sizeof(T)
rows = product(inputShape[0:r-1])
for row = GetBlockIdx(); row < rows; row += C:
    idx = unravel(row, inputShape[0:r-1])
    inputOffset = sum(idx[n] * inputCumShape[n+1], n=0..r-2)
    varOffset = begin[r-1] + sum((begin[n] + idx[n]*strides[n]) * varCumShape[n+1], n=0..r-2)
    for j = 0 .. ceil_div(inputShape[r-1], ubFactor)-1:
        count = tailNum if j is last else ubFactor
        copyParams.blockCount = 1
        copyParams.blockLen = count*sizeof(T)
        # srcStride/dstStride 与 pad 参数未在算子 body 显式赋值，依赖仓外类型默认初始化
        valueBuf<T>[0:count] = DataCopyPad(inputValueGm[inputOffset+j*ubFactor], copyParams)
        wait MTE2 -> MTE3
        DataCopyPad(varOutGm[varOffset+j*ubFactor], valueBuf<T>[0:count], copyParams)
        wait MTE3 -> MTE2 before next overwrite
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `inputValueGm` | `inputOffset+j*ubFactor` | `copyNum*sizeof(T)` bytes | T/ND row-major | ACLNN input/框架 contiguous 边界 | MTE2 `DataCopyPad` | caller；本次 executor 执行期 | 非对齐 tail 由 `DataCopyPad` 处理 |
| UB `valueBuf` | offset 0 | 固定 195584 bytes；逻辑容量 `floor(195584/sizeof(T))` | `TBuf<VECCALC>`，以 `LocalTensor<T>` 访问 | MTE2 | MTE3 | 每个 AIV 的 `TPipe`，覆盖到下个 chunk | depth 不适用；单物理 TBuf 跨 chunk 复用 |
| GM `varOutGm` | `varOffset+j*ubFactor` | `copyNum*sizeof(T)` bytes | T/ND row-major | MTE3 `DataCopyPad` | ACLNN 可变输出 | executor/调用者；切片写入后有效 | 只写有效字节，不物化整张 var |
| system workspace | offset 0 | 16777216 bytes | opaque | framework | Kernel 未消费 | 第二段调用期 | 无算子数据复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  gmIn["inputValue GM · 当前连续 row/chunk"]:1
  value["valueBuf · TBuf<VECCALC> · 195584 bytes · 单缓冲"]:1
  gmOut["varOut GM · 切片目标 row/chunk"]:1
```

##### 字节账与复用顺序

```text
TQue_bytes = 0
valueBuf_bytes = 195584
UB_total = 195584 bytes
UB_total <= platform_ub_size             # 必要条件；Host 仅检查 platform_ub_size > 0
dynamic_ub = N/A
workspace_total = 16777216 bytes system workspace；Kernel 数据 workspace = 0
=> ubFactor = floor(195584 / sizeof(T))，每个 DMA 的 blockLen <= 195584 bytes
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `valueBuf[0:copyBytes)` | 当前 chunk 的 MTE2 `DataCopyPad` | 当前 chunk 的 MTE3 `DataCopyPad` | 下一 `j` 或下一 row 的 MTE2 | 写出前 `MTE2_MTE3`；再写入前 `MTE3_MTE2` | 两个 `WaitFlag` 均完成，且 `blockLen<=195584` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB 完成 | `SetFlag/WaitFlag<HardEvent::MTE2_MTE3>` | MTE2 `DataCopyPad` | MTE3 写回前 | `valueBuf` 当前有效字节 | 纯字节搬运，无 cast/round；tail=`tailNum*sizeof(T)` |
| UB→GM 完成 | `SetFlag/WaitFlag<HardEvent::MTE3_MTE2>` | MTE3 `DataCopyPad` | 下一次 MTE2 覆盖前 | 同一 `valueBuf` | 保证单缓冲复用；只覆盖切片有效范围 |
| 跨核 | row 的 core-cyclic 映射 | 每个 AIV 独占不同 `totalIdx` | N/A | 不同输出 row | 无 atomic/barrier；shape 映射保证 row 不重叠 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_tiling.cpp` | `Tiling4StridedSliceAssignV2` | 归一化索引、校验 shape/last stride、生成累计 shape、blockDim/key/POD/workspace |
| Kernel entry/router | `conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.cpp` | `strided_slice_assign_v2` | key 1 映射到 `KernelStridedSliceAssignV2<DTYPE_VAR>`，其他 key 无 body |
| Device body/helper | `conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.h` | `Init/Process/CalcIdx/CalcInputOffset/CalcVarOffset` | AIV row 分核、地址公式、固定 UB chunk、MTE 硬事件与精确 tail |

### Route R-002：64-bit 等字节搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnStridedSliceAssignV2.md` | `aclnnStridedSliceAssignV2GetWorkspaceSize` | dtype 为 INT64 或 DOUBLE 且其余公开约束成立 | 进入 StridedSliceAssignV2 executor |
| Host config | `op_host/config/ascend910b\|ascend910_93/strided_slice_assign_v2_binary.json` | `StridedSliceAssignV2_Int64_Int64` / `Double_Int64` | A2/A3 + var/input 同 dtype + indices INT64 | 装载 64-bit `DTYPE_VAR` binary |
| Tiling | `op_host/strided_slice_assign_v2_tiling.cpp` | `Tiling4StridedSliceAssignV2` | shape 校验通过 | `key=1`、`blockDim=totalCoreNum` |
| Router | `op_kernel/strided_slice_assign_v2.cpp` | `strided_slice_assign_v2` | `TILING_KEY_IS(1)` | 同一 Kernel 模板 |
| Body | `op_kernel/strided_slice_assign_v2.h` | `Init/Process` | `sizeof(T)==sizeof(int64_t)` 为 true | GM/UB 均改用 `int32_t` 视图；逻辑 offset 乘 2 |

#### Device 数据流

##### 函数级调用链

```text
Tiling4StridedSliceAssignV2 -> key=1, DTYPE_VAR=INT64/DOUBLE
  -> strided_slice_assign_v2
    -> KernelStridedSliceAssignV2<T>::Init
       -> reinterpret input_value/var_out as GlobalTensor<int32_t>
       -> TPipe::InitBuffer(valueBuf, 195584)
    -> KernelStridedSliceAssignV2<T>::Process
       -> CalcIdx -> CalcInputOffset -> CalcVarOffset
       -> DataCopyPad(GM int32 view at logicalOffset*2 -> LocalTensor<int32_t>, blockLen=logicalCount*8)
       -> MTE2_MTE3 event
       -> DataCopyPad(LocalTensor<int32_t> -> GM int32 view at logicalOffset*2)
       -> MTE3_MTE2 event
```

##### 源码等价伪码

```text
# 源码等价伪码；地址仍以逻辑 64-bit 元素推导
ubFactor = 195584 / 8 = 24448 logical elements
for each core-cyclic outer row:
    logicalInputOffset = CalcInputOffset()
    logicalVarOffset = CalcVarOffset()
    for each chunk j:
        logicalCount = tailNum if last else 24448
        copyParams.blockLen = logicalCount * 8 bytes
        int32BaseIn  = (logicalInputOffset + j*24448) * 2
        int32BaseOut = (logicalVarOffset   + j*24448) * 2
        valueBuf<int32>[0:logicalCount*2] = DataCopyPad(inputGmInt32[int32BaseIn], blockLen)
        wait MTE2 -> MTE3
        DataCopyPad(varOutGmInt32[int32BaseOut], valueBuf<int32>, blockLen)
        wait MTE3 -> MTE2
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `inputValueGmInt32` | `(inputOffset+j*24448)*2` int32 单元 | `copyNum*8` bytes = `copyNum*2` int32 | 64-bit payload 的 int32 bit view | caller | MTE2 | caller/执行期 | 不解释数值，仅保持字节 |
| UB `valueBuf` | offset 0 | 195584 bytes = 48896 int32 = 24448 logical 64-bit 元素 | `TBuf<VECCALC>` / `LocalTensor<int32_t>` | MTE2 | MTE3 | 每 AIV 单缓冲 | 与 R-001 同一物理布局，跨 chunk 复用 |
| GM `varOutGmInt32` | `(varOffset+j*24448)*2` int32 单元 | `copyNum*8` bytes | 64-bit payload 的 int32 bit view | MTE3 | ACLNN 输出 | executor/调用者 | 等字节写入，不做 FP/整数运算 |
| system workspace | offset 0 | 16777216 bytes | opaque | framework | Kernel 未消费 | 第二段调用期 | 无算子数据复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  gmIn["64-bit input GM · int32 view · offset×2"]:1
  value["valueBuf · 195584 bytes · 48896 int32 lanes"]:1
  gmOut["64-bit output GM · int32 view · offset×2"]:1
```

##### 字节账与复用顺序

```text
TQue_bytes = 0
valueBuf_bytes = 195584
UB_total = 195584 bytes
UB_total <= platform_ub_size
dynamic_ub = N/A
workspace_total = 16777216 bytes system workspace；Kernel 数据 workspace = 0
=> ubFactor = 195584 / 8 = 24448 logical 64-bit elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `valueBuf[0:copyNum*8)` | MTE2 从 int32 GM 视图搬入 | MTE3 等字节写出 | 下一 chunk/row | `MTE2_MTE3` 与 `MTE3_MTE2` 成对事件 | GM 基址乘 2，`blockLen` 保持 logicalCount×8，且写出完成后才复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 64-bit GM→UB | `DataCopyPad<int32_t>` + `MTE2_MTE3` | MTE2 | MTE3 前 `WaitFlag` | int32 视图下的完整 64-bit payload | 无 cast、无舍入；尾部仍以 `tailNum*8` bytes 搬运 |
| UB→64-bit GM | `DataCopyPad<int32_t>` + `MTE3_MTE2` | MTE3 | 下一 MTE2 前 `WaitFlag` | 同一 payload 与 TBuf | INT64/DOUBLE 位模式原样保留 |
| 跨核 | core-cyclic row | 各 AIV | N/A | 不同 row | 无 atomic/barrier，地址域互斥 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_tiling.cpp` | `Tiling4StridedSliceAssignV2` | 与 R-001 共用 shape、分核、key、POD 与 workspace |
| Kernel entry/router | `conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.cpp` | `strided_slice_assign_v2<DTYPE_VAR=INT64/DOUBLE>` | key 1 进入同一模板实例 |
| Device body/helper | `conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.h` | `Init/Process` 的 `if constexpr(sizeof(T)==8)` | 64-bit GM 重解释、offset×2、字节长度不变、事件同步与 tail |

## 算子特定优化

### Trick T-001：把外层坐标按核轮转、把连续最后一维切成满 UB DMA

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：Host 已保证最后一维 stride 为 1；`inputShape[last]>0`；`GetBlockNum()>0`。
- Golden 来源：[`conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.h:68-84`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.h#L68-L84)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t totalLoopNum = 1;
for (size_t n = 0; n < dimNum - 1; n++) {
    totalLoopNum *= inputShape[n];
}

for (uint64_t totalIdx = GetBlockIdx(); totalIdx < totalLoopNum; totalIdx += usedCoreNum) {
    if (dimNum > 2) {
        CalcIdx(totalIdx);
    } else {
        inputIdx[0] = totalIdx;
    }
    int64_t inputOffset = CalcInputOffset();
    int64_t varOffset = CalcVarOffset();

    for (int64_t j = 0; j < innerRepTimes; j++) {
        uint64_t copyNum = (j == innerRepTimes - 1) ? tailNum : ubFactor;
        copyParams.blockLen = copyNum * sizeof(T);
```

#### 机制与设计原因

Host 对最后一维 stride 的强制条件把每个外层坐标转化为一段连续输入和连续输出。Device 不逐元素执行索引写，而是把 `r-1` 个外层维度展平为 `totalLoopNum`，以 `GetBlockIdx()+k*usedCoreNum` 在 AIV 核之间无重叠轮转；核内再按 `ubFactor=195584/sizeof(T)` 切成尽可能大的连续 `DataCopyPad`。这样源码可证明地把索引计算降到每 row 一次，并以单个固定 TBuf 完成分块搬运，最后块才缩短 `blockLen`。R-002 的 `ubFactor` 仍按逻辑 64-bit `T` 计算，因此每个满块恰好是 195584 bytes。

#### 迁移边界

- 适用：目标切片的内层维连续、外层 row 可独立覆盖、算子仅需等字节赋值的场景。
- 前提：最后一维 stride 为 1；输出 row 不重叠；平台至少能提供 195584-byte UB；shape 累乘与地址计算不溢出 `uint64_t/int64_t`。
- 失效：最后一维非连续时本实现直接在 Host 失败，没有 gather fallback；UB 不足也没有替代 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；可确认的失败终点仅是 Host 对最后一维非 1 stride 返回 `GRAPH_FAILED`，而不是切换到逐元素 Kernel。

## 模板与实现对比

| 对比维度 | R-001 原生元素宽度搬运 | R-002 64-bit 等字节搬运 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | FP16/FP32/BF16/INT32/INT8，`sizeof(T)!=8` | INT64/DOUBLE，`sizeof(T)==8` | binary config 先选 dtype specialization，body 用 `if constexpr`，两者均 key 1，无运行时 fallback 优先级 |
| 模板与 dtype | `KernelStridedSliceAssignV2<T>`，GM/Local 均为 T | 同一模板，GM/Local 改为 `int32_t` bit view | 8-byte 类型走 DILATION=2 地址换算 |
| 循环与分核 | outer row 按 AIV 核轮转，inner chunk 按 `195584/sizeof(T)` | 完全相同；满块为 24448 个 64-bit 元素 | 共享 shape/POD 与 row 地址公式 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 单 195584-byte TBuf；无 Queue/动态 UB/通信 | 同一物理 TBuf，以 48896 个 int32 单元查看 | 两条 Route 的字节容量相同 |
| 内存总占用与复用 | UB 195584 bytes，事件后跨 chunk 复用 | UB 195584 bytes，事件后跨 chunk 复用 | 单缓冲降低中间物化；Host UB 值未参与裁剪 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 两次 `DataCopyPad<T>`；无算术 | 两次 `DataCopyPad<int32_t>`，blockLen 仍为 logicalCount×8 | 均为 AIV MTE 搬运，不含 SIMT/NDDMA/Cube |
| 同步、精度与 tail | MTE2_MTE3、MTE3_MTE2；纯复制；tail 字节数精确 | 同步相同；INT64/DOUBLE 位模式不变 | 不做 cast/round/accumulate |
| fallback/失败边界 | 未知 key 静默结束；shape/stride/资源失败无 Device fallback | 同左 | 唯一 key 与固定 UB 实现 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 ACLNN 文档支持；`ascend910b` config 接线 | 公开 ACLNN 文档支持；`ascend910_93` config 接线 | 公开 ACLNN 文档明确不支持；README/OpDef/`ascend950` config 却有实现信号，构成发布冲突，未建卡 |
| Host/Launcher | 与 A3 共用同一 Tiling/OpDef；对应 binary JSON 内容相同 | 与 A2 共用同一 Tiling/OpDef；对应 binary JSON 内容相同 | 实现目录存在同一 Host/config，但公开 ACLNN gate 阻断发布 |
| Kernel/资源 | 共用 key 1、固定 195584-byte TBuf、AIV row-cyclic body | 同 A2 | 源码 body 共用信号不能覆盖公开“不支持”结论 |

## 限制与失败边界

- 公开 ACLNN 只支持 A2/A3；950 公开文档与实现接线冲突，因此没有 `_950` 正式卡。
- 自动生成的 ACLNN C API bridge、`AutoContiguous` 的真实物化/回写节点、ACLNN workspace 总公式、`varRef` 到 `var_out` 的原地别名和精确 launcher 符号在固定仓不可见；不可据此推断额外 Kernel。
- Host 读取平台 UB 大小却不参与 `ubFactor`，Kernel 固定申请 195584 bytes；源码没有 `ubSize>=195584` guard 或低 UB fallback。
- 公开契约要求所有 stride 为正，但算子 Tiling 仅显式校验最后一维等于 1；非末维零/负 stride 在进入 `CeilDiv` 前没有本地防护，依赖 ACLNN 前置校验，生成式校验源码固定仓不可见。
- `GetConstInputData` 按输入 shape size 复制到固定 8 槽数组，`AdjustWithAxes` 直接用 `axes[i]` 下标；本地 Tiling 没有独立验证 begin/end/strides 长度、axes 长度与轴范围。公开 bridge 是否补充这些检查固定仓不可见。
- `DataCopyExtParams` 除 `blockCount/blockLen` 外的字段以及 `DataCopyPadExtParams` 均未在算子 body 显式赋值；其仓外类型默认值和 padding 细节固定仓不可见，复用时不能自行假定具体数值。
- `outerRows`、累计 shape、输入/输出 offset 都存在 64-bit 乘加，固定源码没有溢出检查。
- inputValue 某维为 0 的公开行为未明确；Kernel 对最后一维为 0 时的 `innerRepTimes/tailNum` 推导没有独立 guard，不应从源码外推空 Tensor 支持。
- entry 对未知 key 只跳过 body，不报告 Device 错误；正式 Host 只产生 key 1。
- Tiling 固定声明 16 MiB system workspace，但 Kernel 不读取；该内存的框架用途固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/strided_slice_assign_v2/docs/aclnnStridedSliceAssignV2.md` | 两段式原型、产品支持表、参数/约束 | ACLNN 公开入口、A2/A3 支持、950 不支持、dtype/rank/stride/确定性 |
| 入口声明/约束 | `docs/zh/context/两段式接口.md` | 两段式接口说明 | workspace/executor 顺序与 executor 不可重复执行 |
| 入口声明/约束 | `conversion/strided_slice_assign_v2/README.md` | 产品支持与语义 | 算子层 950 支持信号，与 ACLNN 文档形成冲突 |
| 入口实现 | `conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_def.cpp` | `ops::StridedSliceAssignV2` | 输入输出、ValueDepend、AutoContiguous、A2/A3/950 compute-unit 接线 |
| 入口实现 | `conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_infershape.cpp` | `InferShape4Elewise`、`InferDataType4SameAsInput` | 输出 shape/dtype 继承 var |
| Host/Tiling/Launcher | `conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_tiling.h` | `StridedSliceAssignV2TilingData`、compile info | POD 字段与 AIV/UB compile info |
| Host/Tiling/Launcher | `conversion/strided_slice_assign_v2/op_host/strided_slice_assign_v2_tiling.cpp` | `Adjust*`、`CheckInputShape`、`Tiling4*`、`TilingPrepare4*` | 索引归一化、校验、分核、key、POD、workspace |
| Kernel entry/body | `conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.cpp` | `strided_slice_assign_v2` | entry ABI、key 1 router、模板实例 |
| Kernel entry/body | `conversion/strided_slice_assign_v2/op_kernel/strided_slice_assign_v2.h` | `KernelStridedSliceAssignV2` | 两条 dtype body、地址、MTE、事件、UB、tail |
| 构建接线 | `conversion/strided_slice_assign_v2/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn)` | ACLNN OpDef 分类 |
| 构建接线 | `cmake/func.cmake` | `add_modules_sources` | OpDef/Tiling/Infer 源加入对应构建对象 |
| 构建接线 | `cmake/gen_ops_info.cmake` | `get_op_type_and_validate`、`prepare_compile_from_config` | binary JSON 与 `op_kernel` 编译接线 |
| POD 生成 | `scripts/util/tiling_data_def_build.py` | `gen_tiling` | packed Device POD、字段顺序和 `GET_TILING_DATA` 生成 |
| 构建接线 | `conversion/strided_slice_assign_v2/op_host/config/ascend910b/strided_slice_assign_v2_binary.json` | `op_list` | A2 七种 dtype binary |
| 构建接线 | `conversion/strided_slice_assign_v2/op_host/config/ascend910_93/strided_slice_assign_v2_binary.json` | `op_list` | A3 七种 dtype binary |
| 构建接线 | `conversion/strided_slice_assign_v2/op_host/config/ascend950/strided_slice_assign_v2_binary.json` | `op_list` | 950 实现接线冲突信号，不作为公开支持证明 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 392-byte、7 字段的顺序序列化；数组均 8 槽 | 字段次序、int64 位宽、wire offset、有效 rank 与零填充槽不得改变 | `StridedSliceAssignV2TilingData`；`gen_tiling` |
| 调用顺序 | 先归一化/shape 校验，再 key/POD/workspace；Device `Init` 后 `Process` | key 必须为 1；生成式 ACLNN bridge 细节不可伪造 | `Tiling4StridedSliceAssignV2`；`strided_slice_assign_v2` |
| workspace/内存 | 每 AIV 单 195584-byte TBuf；system workspace 16 MiB 且 Kernel 未读 | 不得用逻辑有效长度替代固定分配；平台 UB 必须容纳 195584 bytes | `KernelStridedSliceAssignV2::Init`；`Tiling4StridedSliceAssignV2` |
| 同步 | 每 chunk 严格 MTE2→MTE3、MTE3→下一 MTE2 | 单缓冲未等写出完成不得覆盖；无跨核共享/atomic | `KernelStridedSliceAssignV2::Process` |
| 精度与尾块 | 所有 dtype 都是等字节复制；64-bit 用 int32 视图；尾部按实际 bytes | 64-bit GM offset 必须乘 2，blockLen 必须仍为 `copyNum*8`；不得引入 cast/round | `KernelStridedSliceAssignV2::Init/Process` |
| 失败边界 | rank/shape/last-stride/core/UB-positive 失败均阻断 launch；未知 key 无输出 | 不得假设非末维 stride、axes 范围、空 Tensor 或低 UB fallback 已在本地实现 | `Tiling4*`、`TilingPrepare4*`、entry key guard |

- [ValueDepend](../../../../apis/utils_api/prototype_register_management/op_param_def/value_depend.md)
