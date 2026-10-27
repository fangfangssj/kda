---
type: Operator
title: "ExpandIntoJaggedPermute A2/A3 ACLNN 算子知识"
description: "说明 aclnnExpandIntoJaggedPermute 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/docs/aclnnExpandIntoJaggedPermute.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_host/expand_into_jagged_permute_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_host/expand_into_jagged_permute_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_host/expand_into_jagged_permute_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_kernel/expand_into_jagged_permute.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_kernel/expand_into_jagged_permute.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T16:35:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ExpandIntoJaggedPermute A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnExpandIntoJaggedPermuteGetWorkspaceSize` + `aclnnExpandIntoJaggedPermute`，公开 ACLNN 两段式接口，无其它公开版本 |
| 计算语义 | 对每个表项 `i`，以 `permute[i]` 选择输入段起点，并生成长度为 `outputOffset[i+1]-outputOffset[i]` 的连续整数段 |
| 输入输出 | `permute[P]`、`inputOffset[P+1]`、`outputOffset[P+1]`、`outputPermuteOut[outputSize]`；均为一维 ND、`int32` Tensor |
| 执行类型 | 单 Kernel；节点 K-001 为 `AIV_VECTOR`（`DataCopyPad`、`CreateVecIndex`、`Sub`、`Adds`），无可达 Cube/AIC 或 `MIX_CV` body |
| Kernel 路线 | R-001：key 0，输入偏移表预载到 UB；R-002：key 1，输入偏移从 GM 标量读取 |
| 核心资源 | GM + 5 个 depth-1 `TQue` + 1 个 `TBuf`；Kernel 不使用 workspace；Host 固定声明 16 MiB system workspace |
| 来源闭包 | `source_closed`：公开 ACLNN 文档与 `ACLNNTYPE aclnn` 接线，经 OpDef/自动连续化框架边界、Tiling/POD、router 到两个 AIV_VECTOR body 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

记 `P = len(permute)`。对 `0 <= i < P`：

```text
L_i = outputOffset[i + 1] - outputOffset[i]
s_i = inputOffset[permute[i]]
outputPermuteOut[outputOffset[i] : outputOffset[i + 1]]
    = [s_i, s_i + 1, ..., s_i + L_i - 1]
```

公开接口要求 `permute[i]` 位于 `[0, P-1]`，两个 offset Tensor 严格递增且末值等于 `outputSize`。Kernel 只写每段的有效 `L_i` 个元素；每个 128 元素 tile 中超过尾长的 UB lane 不写入 GM。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `permute` | `[P]` / ND | `int32` | 必选，一维；值域公开约束为 `[0, P-1]`；OpDef 标记 `AutoContiguous()` |
| `inputOffset` | `[P+1]` / ND | `int32` | 必选，一维、严格递增，末值为 `outputSize`；按 `permute[i]` 索引 |
| `outputOffset` | `[P+1]` / ND | `int32` | 必选，一维、严格递增，末值为 `outputSize`；相邻差产生段长 |
| `outputSize` | 标量属性 | ACLNN 原型 `int64_t`；OpDef `Int` | 必选，决定输出逻辑长度；Tiling 以 `GetAttrPointer<int32_t>(0)` 读取，固定源码未给出超出 `int32` 的处理 |
| `outputPermuteOut` | `[outputSize]` / ND | `int32` | 必选输出；文档允许非连续 Tensor，具体 ViewCopy/连续化桥接由生成框架负责 |

### 入口与版本

公开文档只声明无版本后缀的 `aclnnExpandIntoJaggedPermute`。第一段接口完成校验、生成 executor 并返回 workspace 大小；第二段把 workspace、executor 和 stream 提交执行。固定仓未保存自动生成的 ACLNN inner bridge 源文，但 `op_host/CMakeLists.txt` 将 `expand_into_jagged_permute` 标为 `ACLNNTYPE aclnn`，OpDef 的三个输入均启用 `AutoContiguous()`；这是允许保留为框架边界的生成层，算子自有 Tiling、POD、router 和 body 均已闭合。

公开产品表明确支持 Atlas A2/A3，OpDef 和 binary config 同时接入 `ascend910b`、`ascend910_93`。公开表明确不支持 Ascend 950PR/950DT，故本卡不覆盖 950。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段接口按公开文档检查必选指针、dtype/format、Tensor 间 shape 与属性关系；生成代码在固定仓不可见。
2. OpDef 将三个输入限制为 `int32/ND` 并标记 `AutoContiguous()`；非连续输入的实际物化属于框架边界。
3. `InputParamCheck` 获取三个 shape、Tensor 与 descriptor，检查 descriptor 非空、三个输入为一维。源码中的 dtype 条件是 `a != b && a != c && b != c`，只会拒绝三者两两不同；不过 OpDef 已把三者均限制为 `int32`。
4. `InputParamCheck` 在一维检查后第 75 行直接返回成功，因此第 77–81 行的 `len(inputOffset)=len(outputOffset)=P+1` 检查不可达；该关系由公开接口约束，但不能归因于当前 Tiling 的动态检查。
5. `Init` 读取平台 AIV 核数、UB 大小、三个一维长度及 `output_size`，计算 key、blockDim 和 POD。查询到的 UB 大小只保存到 `param.maxCoreMemery`，没有进入 Route 或 tile 上界检查。

任一步返回 `GRAPH_FAILED` 时 Tiling 失败，Kernel 不应启动；生成 ACLNN executor 的具体错误传播为框架边界。

### 参数、Workspace 与生命周期

设 `P = permute.shape[0]`、`S = sizeof(int32)=4`：

```text
SIZE = cacheLine = 512 bytes
oneTaskOffsetLen = 512 elements
oneTaskLen = cacheLine / S = 128 elements
taskNum = ceil(P / 512)
lastTaskLen = P - 512 * (taskNum - 1)
blockDim = platform.GetCoreNumAiv()
```

`Init` 通过 `context->GetWorkspaceSizes(1)[0] = 16 * 1024 * 1024` 声明一个 16 MiB system workspace。Kernel entry 接收 `workspace`，但 router/body 未读取它；非连续 Tensor 可能产生的框架 staging 与 executor 元数据不在固定仓生成源码中，故 ACLNN 第一段返回的最终总大小公式不可见。workspace 由调用方按第一段结果分配，在第二段 stream 执行结束前保持有效。

### 启动图与失败传播

```text
aclnnExpandIntoJaggedPermuteGetWorkspaceSize(...)       # 生成框架边界
  -> ExpandIntoJaggedPermute OpDef / AutoContiguous
  -> TilingForExpandIntoJaggedPermute
       -> InputParamCheck -> Init -> SetTilingKey -> SetTilingData
  -> executor（单节点 K-001）

aclnnExpandIntoJaggedPermute(workspace, size, executor, stream)
  -> K-001 expand_into_jagged_permute<<<blockDim=GetCoreNumAiv()>>>
       -> key 0: ExpandIntoJaggedPermute<int32_t, false>
       -> key 1: ExpandIntoJaggedPermute<int32_t, true>
```

| 节点/边 | producer | consumer | 数据与范围 | ordering、owner/lifetime | 失败传播 |
|---|---|---|---|---|---|
| K-001 输入 | ACLNN executor | `expand_into_jagged_permute` | 3 个只读 GM Tensor、1 个输出 GM Tensor、POD；ND/int32 | 调用方/框架持有至 stream 完成；同一 stream 提交 | Tiling 或 launch 失败应阻断执行；细节属生成框架边界 |
| system workspace | Tiling `Init` | runtime | offset 0、16 MiB；Kernel 不消费 | 调用方按 GetWorkspaceSize 结果持有 | 分配或提交失败由 ACLNN 状态码返回 |
| K-001 输出 | Kernel `CopyOut` | `outputPermuteOut` | 每个 tile 仅 `nowlen` 个 int32 | 单 Kernel 内 TQue 顺序；stream 完成后对调用方可见 | 无后继 Kernel |

启动图没有 Kernel 间 workspace 边、event 或多节点生命周期。

## Route 与参数设计

### 分核与资源计算

Tiling 固定启动全部可用 AIV 核。每个核都扫描全部 `P` 个表项及其输出 tile，并用全局 tile 序号 `taskidA` 的模分配决定写入者：仅当 `taskidA % realCoreNum == blockIdx` 时计算和 CopyOut。地址并不按连续 core 区间切分；所有核的 `sCoreOffset` 都保持成员初值 0，而每段通过 `outputOffsetLocal[i]` 重置 `outoffsetStart`。

每个表批次最多装入 512 项；每个输出 tile 固定计算 128 个 int32，尾 tile 只提交 `nowlen = L_i - 128*(tileCount-1)` 个。`coreLen`、`frontCoreNum`、`tailCoreBlockFactor` 等传统 former/tail 字段不参与实际写址或循环边界。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 UB 预载输入偏移 | key 0；`ExpandIntoJaggedPermute<int32_t, false>` | `inputoffset = inputOffset.shape[0] <= 512` | 条件为假进入 R-002 |
| 2 | R-002 GM 标量输入偏移 | key 1；`ExpandIntoJaggedPermute<int32_t, true>` | `inputoffset > 512` | 无其它 fallback；未知 key 的 entry 无操作 |

```text
if inputOffset.shape[0] <= 512:
    key = 0  -> T0=false -> 整张 inputOffset 预载到 inputOffsetsQueue
else:
    key = 1  -> T0=true  -> 每段以 inputOffsetGm.GetValue(permuteId) 读取
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 UB 预载 | key 0 | A2/A3，OpDef int32/ND，三个输入一维，`inputOffset.len <= 512` | 单 K-001 AIV_VECTOR | `ExpandIntoJaggedPermute<DTYPE_PERMUTE,false>::Process` | 先把 `P+1` 个 inputOffset 搬入 depth-1 VECIN Queue；其余按 512/128 分块；Queue 隐式同步 | 每段尾 tile 仍 `Adds` 128 元素，只 CopyOut `nowlen`；Tiling 失败则不启动 |
| R-002 GM 标量 | key 1 | A2/A3，OpDef int32/ND，三个输入一维，`inputOffset.len > 512` | 单 K-001 AIV_VECTOR | `ExpandIntoJaggedPermute<DTYPE_PERMUTE,true>::Process` | 不执行预载；按 `permuteId` 从 GM 标量取起点；但 Init 仍分配整张 `inputOffsetsQueue` | 同 R-001；未知 key 无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 输入偏移预载阈值 | `param.inputoffset <= 512` | key 0 在 Process 前一次性搬运整张 inputOffset，后续 UB 随机读 | R-001 | R-002 GM 标量读取 | `expand_into_jagged_permute_tiling.cpp::SetTilingKey` |
| 每批表项上限 | 固定 `oneTaskOffsetLen=512` | 三个分块 Queue 每个请求 512 个 int32 | 两 Route | 外层 `taskNum=ceil(P/512)` 增加批次 | `Init` / `Process` |
| 输出 tile 上限 | 固定 `oneTaskLen=512/4=128` | `rangeQueue`、`outputPermuteQue` 容量和 `Adds` 长度均为 128 | 两 Route | 内层 `ceil(L_i/128)` 增加 tile，末次 CopyOut 截尾 | `Init` / `initRange` / `Process` |
| UB 可行性 | 源码查询 `maxCoreMemery`，但没有比较 `7168 + 4*(P+1)` 请求字节 | 两 Route 的 Init 都按 `offsetLen=P+1` 分配 `inputOffsetsQueue`，R-002 也未消除此 buffer | 两 Route，长输入尤甚 | 固定源码没有 fallback/失败 guard | `Init` / `ExpandIntoJaggedPermute::Init` |
| 空输入 | `P=0` 时 `taskNum=0`，仍启动平台全部 AIV 核 | Process 外层循环为 0；只初始化 range/Queue | 两 Route（key 0） | 无输出写入 | `GetCeilInt` / `Process` |
| dtype 一致性 | Tiling 条件使用三个“不等”关系的逻辑与 | 该表达式不是通常的“任意两者不同即失败” | 两 Route | OpDef 的 int32 限制先行；固定 Tiling 无额外 fallback | `InputParamCheck` |

### POD/ABI

`ExpandIntoJaggedPermuteTilingData` 按下表声明 10 个 `int64_t` 字段。宏 `BEGIN_TILING_DATA_DEF` 的物理 `sizeof/alignof/offsetof` 实现在仓外依赖中，固定仓只能证明字段声明顺序和 64 位类型，不能把自然布局推测成 ABI。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ExpandIntoJaggedPermuteTilingData` | `realCoreNum` | `int64_t`/64 | 声明序号 0；物理 offset 固定仓不可见 | core | 未调用 setter | `Init` 未读取（改用 `GetBlockNum()`） | inactive |
| 同上 | `frontCoreNum` | `int64_t`/64 | 序号 1；物理 offset 不可见 | core | 未调用 setter | `Init` 读入成员，但后续不使用 | inactive |
| 同上 | `blockFactor` | `int64_t`/64 | 序号 2；物理 offset 不可见 | 512-entry batch 数 | `set_blockFactor(ceil(P/512))` | `Init -> coreTaskNum`，控制所有核扫描批次数 | `P=0` 时 0 |
| 同上 | `oneTaskLen` | `int64_t`/64 | 序号 3；物理 offset 不可见 | int32 元素 | `set_oneTaskLen(512/dtypeSize)=128` | Queue/TBuf 容量、`CreateVecIndex`、`Adds`、tile 循环 | 依赖 dtypeSize 非 0；OpDef 固定 int32 |
| 同上 | `tailCoreBlockFactor` | `int64_t`/64 | 序号 4；物理 offset 不可见 | task | 未调用 setter | `Init` 读入成员，但后续不使用 | inactive |
| 同上 | `lastTaskLen` | `int64_t`/64 | 序号 5；物理 offset 不可见 | 表项 | `set_lastTaskLen(P-512*(taskNum-1))` | 只用于计算未消费的 `coreLen` | `P=0` 得 512，但循环由 blockFactor=0 抑制 |
| 同上 | `oneTaskOffsetLen` | `int64_t`/64 | 序号 6；物理 offset 不可见 | 表项 | `set_oneTaskOffsetLen(512)` | 三个分块 Queue 容量与 `copyOffsetLen` 上限 | 固定 512 |
| 同上 | `inputLen` | `int64_t`/64 | 序号 7；物理 offset 不可见 | 表项 | `set_inputLen(P)` | `copyOffsetLen=min(P-nowoutputOffsets,512)` | 非负 shape |
| 同上 | `offsetLen` | `int64_t`/64 | 序号 8；物理 offset 不可见 | int32 元素 | `set_offsetLen(P+1)` | `inputOffsetsQueue` 容量；R-001 CopyIn 长度 | 源码直接按 `P+1`，不读取真实 offset shape 长度 |
| 同上 | `outputSize` | `int64_t`/64 | 序号 9；物理 offset 不可见 | int32 元素 | `set_outputSize(*GetAttrPointer<int32_t>(0))` | `Init` 读入成员，但 Process 不使用 | inactive 于地址边界检查 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 UB 预载 | key 0 | `op_host/expand_into_jagged_permute_tiling.cpp::{InputParamCheck,Init,SetTilingKey,SetTilingData}` | `ExpandIntoJaggedPermuteTilingData` | `op_kernel/expand_into_jagged_permute.cpp::expand_into_jagged_permute` key-0 branch | `op_kernel/expand_into_jagged_permute.h::ExpandIntoJaggedPermute<T,false>::{Init,Process,CopyInPermute,CopyInOffset,initRange,CopyOut}` | `op_host/CMakeLists.txt` + `scripts/kernel/binary_config/ascendc_config.json` |
| R-002 GM 标量 | key 1 | 同上 | 同一 POD；key 决定 `T0=true` | 同一 entry 的 key-1 branch | `ExpandIntoJaggedPermute<T,true>` 同组方法；input 起点走 `inputOffsetGm.GetValue` | 同上 |

## Kernel 实现

### 公共执行流程

两条 Route 共用单 Kernel entry、相同分块与输出 tile 管线。entry 解析 POD 后按 key 构造模板实例。每个 AIV 核创建 `[0,127]` 的 int32 range；随后所有核都按 512 个表项一批扫描：搬入 `outputOffset[i]`、`outputOffset[i+1]` 与 `permute[i]`，用 `Sub` 把后一 offset 原地变为段长；逐段、逐 128 元素 tile 增加全局 tile 序号。命中 `taskidA % GetBlockNum() == GetBlockIdx()` 的核执行 `Adds(range, inputStart)` 并写出有效尾长，否则只推进地址。

Kernel 参数中的 `workspace` 未使用；未知 key 没有 `else` 错误分支，因此 entry 直接结束且不写输出。两 Route 唯一计算差异是 `inputStart` 来自预载 UB 或 GM 标量。

### Route R-001：UB 预载输入偏移

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnExpandIntoJaggedPermute.md` | 两段式公开原型 | 三个 int32/ND 一维输入；A2/A3 支持 | 进入生成 executor |
| Host/Tiling | `op_host/expand_into_jagged_permute_tiling.cpp` | `SetTilingKey` | `param.inputoffset <= 512` | key 0 |
| Router | `op_kernel/expand_into_jagged_permute.cpp` | `expand_into_jagged_permute` | `TILING_KEY_IS(0)` | `ExpandIntoJaggedPermute<DTYPE_PERMUTE,false>` |

#### Device 数据流

##### 函数级调用链

```text
TilingForExpandIntoJaggedPermute
  -> SetBlockDim(GetCoreNumAiv), key=0, SaveToBuffer(POD)
expand_into_jagged_permute(permute, input_offsets, output_offsets, output_permute, workspace, tiling)
  -> ExpandIntoJaggedPermute<int32_t,false>::Init
  -> ExpandIntoJaggedPermute<int32_t,false>::Process
       -> initRange
       -> CopyInPermute(offsetLen, 0)          # 实际搬入 input_offsets
       -> foreach 512-entry batch: CopyInOffset
       -> Sub(nextOutputOffset, nextOutputOffset, outputOffset)
       -> foreach segment/tile: Adds -> EnQue -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；每个 AIV 核均执行
range[0:128] = 0..127
inputOffsetLocal[0:P+1] = inputOffsetGm[0:P+1]
nextEntry = 0; globalTile = 0
for batch in 0 .. ceil(P/512)-1:
    n = min(P-nextEntry, 512)
    outBeg[0:n] = outputOffsetGm[nextEntry:nextEntry+n]
    outEnd[0:n] = outputOffsetGm[nextEntry+1:nextEntry+n+1]
    perm[0:n]   = permuteGm[nextEntry:nextEntry+n]
    len[0:n] = outEnd - outBeg
    nextEntry += n
    for j in 0 .. n-1:
        dst = outBeg[j]
        src = inputOffsetLocal[perm[j]]
        for tile in 0 .. ceil(len[j]/128)-1:
            valid = (tile is last) ? len[j]-128*(tiles-1) : 128
            if globalTile % GetBlockNum() == GetBlockIdx():
                ubOut[0:128] = range[0:128] + int32(src)
                outputPermuteGm[dst:dst+valid] = ubOut[0:valid]
            globalTile += 1
            src += valid; dst += valid
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `permute` | batch base `nowoutputOffsets` | 每批 `min(P-base,512)` | int32/ND | caller | `CopyInOffset` | caller 至 stream 完成 | `DataCopyPad`, blockCount=1, blockLen=`n*4` bytes, stride=0 |
| GM `inputOffset` | base 0 | `P+1` | int32/ND | caller | `CopyInPermute` | caller 至 stream 完成 | 单次 `DataCopyPad`, blockLen=`(P+1)*4` |
| GM `outputOffset` | base 与 base+1 | 各 `n` | int32/ND | caller | `CopyInOffset` | caller 至 stream 完成 | 两次 `DataCopyPad`，形成相邻窗口 |
| `rangeQueue` TBuf | 独占 UB 子区 | 128 int32 = 512 B | int32 | `CreateVecIndex` | 每次 `Adds` | Kernel 全程 | 不入 Queue，跨全部 tile 只初始化一次 |
| `permuteQueue` VECIN | 独占 UB 子区 | depth 1 × 512 int32 = 2048 B | int32 | `CopyInOffset` | Process 标量 `GetValue` | 每 batch Alloc→EnQue→DeQue→Free | 下一 batch 复用 |
| `inputOffsetsQueue` VECIN | 独占 UB 子区 | depth 1 × `(P+1)` int32 | int32 | `CopyInPermute` | Process 标量 `GetValue` | Process 前 Alloc，末尾 Free | 全程驻留，R-001 有效 |
| `outputOffsetQueue` VECIN | 独占 UB 子区 | depth 1 × 512 int32 = 2048 B | int32 | `CopyInOffset` | `Sub` 与地址读取 | 每 batch 生命周期 | 最后 entry 读取前释放调用发生在源码循环中 |
| `outputOffsetNextQueue` VECIN | 独占 UB 子区 | depth 1 × 512 int32 = 2048 B | int32 | `CopyInOffset` | `Sub` 后段长读取 | 每 batch 生命周期 | `Sub` 原地覆盖 next-offset 为 length |
| `outputPermuteQue` VECOUT | 独占 UB 子区 | depth 1 × 128 int32 = 512 B | int32 | `Adds` | `CopyOut` | 每个本核 tile Alloc→EnQue→DeQue→Free | 尾 tile 仍算满 128，只搬有效前缀 |
| GM `outputPermute` | `outputOffset[i] + tile*128` | `valid` int32 | int32/ND | `CopyOut` | caller | caller 持有 | UB→GM `DataCopyPad`, blockLen=`valid*4` |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  range["rangeQueue TBuf · 128×4 = 512 B"]:1
  perm["permuteQueue · d=1 · 512×4 = 2048 B"]:1
  inoff["inputOffsetsQueue · d=1 · (P+1)×4 B"]:1
  out0["outputOffsetQueue · d=1 · 512×4 = 2048 B"]:1
  out1["outputOffsetNextQueue · d=1 · 512×4 = 2048 B"]:1
  outq["outputPermuteQue · d=1 · 128×4 = 512 B"]:1
```

##### 字节账与复用顺序

```text
UB_requested = 512 + 2048 + 4*(P+1) + 2048 + 2048 + 512
             = 7168 + 4*(P+1) bytes
R-001: P+1 <= 512, so UB_requested <= 9216 bytes
UB_requested <= platform_UB 是运行必要条件，但 Host 未执行该比较；
TPipe 内部对齐后的实际占用属于固定仓外 primitive 实现，不能据此虚报等号。
dynamic_ub = N/A
workspace_total(operator-owned) = 0；system workspace declaration = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `outputOffsetNextQueue` | `CopyInOffset` | `Sub` 与段长 `GetValue` | 下一 batch | EnQue→DeQue，最后 entry 时 Free | 当前 batch 所有段长已读完 |
| `permuteQueue` | `CopyInOffset` | `permuteLocal.GetValue` | 下一 batch | EnQue→DeQue，最后 entry 时 Free | 当前 batch 所有 permute 值已读完 |
| `outputPermuteQue` | `Adds` | `CopyOut` | 下一属本核 tile | EnQue→DeQue→Free | CopyOut 完成后才复用 depth-1 Queue |
| `rangeQueue` | `initRange` | 所有 `Adds` | 不覆盖 | TBuf Kernel 全生命周期 | 只读 range 不被其它 API 改写 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| offset/permute 搬入 | depth-1 TQue EnQue/DeQue | `DataCopyPad` | Process `DeQue` | 三路 VECIN local Tensor | blockCount=1、byte blockLen=`n*4`、stride=0；pad 参数关闭 padding |
| 段长 | `Sub(dst=next,dst,begin,n)` | 两个 output offset Queue | 标量 `GetValue` | `outputOffsetNextLocal[0:n]` | int32 减法；公开约束递增，固定 Kernel 无负长 guard |
| range 生成 | `CreateVecIndex(range,0,128)` | `initRange` | `Adds` | range TBuf | int32 起值 0；生成 128 lane |
| 输出生成 | `Adds(output,range,int32(inputoffset),128)` | range + 输入起点 | output Queue | 128 个 int32 | 始终满 128；scalar 显式转 `int32_t`，无升精度 |
| 输出提交 | output TQue + `DataCopyPad` | `Adds` | GM CopyOut | 当前 tile | 仅 `nowlen` 个元素写 GM；无效 lane 不提交 |
| 跨核排他 | `globalTile % realCoreNum` | 所有核相同扫描序 | 匹配 blockIdx 的唯一核 | 输出 tile | 无 atomic/barrier；正确性依赖所有核生成完全一致的 globalTile 顺序 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/expand_into_jagged_permute_tiling.cpp` | `Init` / `SetTilingKey` / `SetTilingData` | 512-entry batch、128-output tile、全 AIV blockDim、key 0、POD 与 16 MiB workspace |
| Kernel entry/router | `op_kernel/expand_into_jagged_permute.cpp` | `expand_into_jagged_permute` key-0 branch | key 0 映射到 `<DTYPE_PERMUTE,false>` |
| Device body/helper | `op_kernel/expand_into_jagged_permute.h` | `Init` / `CopyInPermute` / `CopyInOffset` / `initRange` / `Process` / `CopyOut` | UB 预载输入 offset、三路分块 DMA、原地 Sub、模分核、Adds 与有效尾写回 |

### Route R-002：GM 标量输入偏移

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnExpandIntoJaggedPermute.md` | 两段式公开原型 | 三个 int32/ND 一维输入；A2/A3 支持 | 进入生成 executor |
| Host/Tiling | `op_host/expand_into_jagged_permute_tiling.cpp` | `SetTilingKey` | `param.inputoffset > 512` | key 1 |
| Router | `op_kernel/expand_into_jagged_permute.cpp` | `expand_into_jagged_permute` | `TILING_KEY_IS(1)` | `ExpandIntoJaggedPermute<DTYPE_PERMUTE,true>` |

#### Device 数据流

##### 函数级调用链

```text
TilingForExpandIntoJaggedPermute
  -> SetBlockDim(GetCoreNumAiv), key=1, SaveToBuffer(POD)
expand_into_jagged_permute(...)
  -> ExpandIntoJaggedPermute<int32_t,true>::Init
  -> ExpandIntoJaggedPermute<int32_t,true>::Process
       -> initRange
       -> foreach 512-entry batch: CopyInOffset
       -> Sub(nextOutputOffset, nextOutputOffset, outputOffset)
       -> inputOffsetGm.GetValue(permuteId)
       -> foreach segment/tile: Adds -> EnQue -> CopyOut
```

##### 源码等价伪码

```text
# 与 R-001 相同的全核扫描和模分核；差异如下
range[0:128] = 0..127
for each 512-entry batch and each segment j:
    src = inputOffsetGm.GetValue(permuteLocal[j])
    for each 128-entry output tile:
        if globalTile % GetBlockNum() == GetBlockIdx():
            outputLocal[0:128] = range + int32(src)
            copy only valid lanes to outputOffset[j] + tile*128
        advance src/dst by valid; globalTile += 1
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `inputOffset` | scalar address `permuteId` | 每段 1 int32 读取 | int32/ND | caller | `inputOffsetGm.GetValue` | caller 至 stream 完成 | 非合并随机标量路径，固定源码无 cache helper |
| 三路 batch GM/Queue | 同 R-001 | 各至多 512 int32 | int32/ND | `CopyInOffset` | Process | 每 batch | 同 R-001 |
| `inputOffsetsQueue` | 独占 UB 子区 | depth 1 × `(P+1)` int32 | int32 | 没有 `AllocTensor/EnQue` | 无消费者 | Init 分配至 Kernel 结束 | R-002 未条件消除此 buffer；不能从字节账扣除 |
| `rangeQueue` / `outputPermuteQue` | 同 R-001 | 各 512 B | int32 | `CreateVecIndex` / `Adds` | `Adds` / `CopyOut` | 同 R-001 | 同 R-001 |
| GM `outputPermute` | `outputOffset[i]+tile*128` | `valid` int32 | int32/ND | `CopyOut` | caller | caller 持有 | 尾部有效前缀写回 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  range["rangeQueue TBuf · 128×4 = 512 B"]:1
  perm["permuteQueue · d=1 · 512×4 = 2048 B"]:1
  inoff["inputOffsetsQueue · d=1 · (P+1)×4 B · 未消费"]:1
  out0["outputOffsetQueue · d=1 · 512×4 = 2048 B"]:1
  out1["outputOffsetNextQueue · d=1 · 512×4 = 2048 B"]:1
  outq["outputPermuteQue · d=1 · 128×4 = 512 B"]:1
```

##### 字节账与复用顺序

```text
UB_requested = 7168 + 4*(P+1) bytes
R-002: P+1 > 512，且源码没有 UB_requested <= platform_UB 的 selector/guard；
查询到的 maxCoreMemery 未参与计算，因此固定源码不能证明任意长 P 的 UB 可行性。
dynamic_ub = N/A
workspace_total(operator-owned) = 0；system workspace declaration = 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `outputOffsetNextQueue` | `CopyInOffset` | `Sub` / 段长读取 | 下一 batch | EnQue→DeQue→Free | 同 R-001 |
| `permuteQueue` | `CopyInOffset` | `permuteId` 读取 | 下一 batch | EnQue→DeQue→Free | 同 R-001 |
| `outputPermuteQue` | `Adds` | `CopyOut` | 下一属本核 tile | EnQue→DeQue→Free | 同 R-001 |
| `inputOffsetsQueue` | 无写入者 | 无消费者 | N/A | 仅 InitBuffer 分配 | 不得误报为可供其它 Queue alias 的空闲区 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| input 起点 | GM `GetValue` | caller GM | 当前段计算 | 单个 inputOffset | int32 标量；无 Queue 同步、无越界 guard，依赖 `permute` 值域 |
| batch 搬入、Sub | depth-1 TQue | `DataCopyPad` | DeQue / `Sub` | permute/output offset UB | 同 R-001 |
| 输出生成与提交 | `Adds` + output TQue + `DataCopyPad` | range + GM 标量 | GM output | 当前 tile | 计算 128 lane，只写 `nowlen`；int32 无升精度 |
| 跨核排他 | tile 序号模核数 | 全核冗余扫描 | 唯一匹配核 | 输出 tile | 无 atomic/barrier；所有核必须保持相同扫描序 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/expand_into_jagged_permute_tiling.cpp` | `SetTilingKey` | `inputoffset>512` 产生 key 1 |
| Kernel entry/router | `op_kernel/expand_into_jagged_permute.cpp` | key-1 branch | key 1 映射 `<DTYPE_PERMUTE,true>` |
| Device body/helper | `op_kernel/expand_into_jagged_permute.h` | `Init` / `Process` | 跳过 `CopyInPermute`，以 `inputOffsetGm.GetValue(permuteId)` 取段起点；其余循环、Queue、tail 与 R-001 相同 |

## 算子特定优化

### Trick T-001：预生成 range 并用模分核复用同一生成管线

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：任意 `L_i>0` 的输出段；每个 128 元素 tile 仅由满足 `taskidA % realCoreNum == blockIdx` 的核写回。
- Golden 来源：[`op_kernel/expand_into_jagged_permute.h:193-213`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_kernel/expand_into_jagged_permute.h#L193-L213)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t tasklen = Ceil(outLen, this->oneTaskLen);
for(int32_t taskid = 0; taskid < tasklen; taskid++){
    if(taskid == tasklen-1){
      this->nowlen = outLen - this->oneTaskLen*(tasklen-1);
    }else{
      this->nowlen = this->oneTaskLen;
    }
  if (taskidA % this->realCoreNum == this->blockIdx){
      auto outputPermuteLocal = outputPermuteQue.AllocTensor<T>();
      Adds(outputPermuteLocal, rangeLocal, static_cast<int32_t>(inputoffset), oneTaskLen);
      outputPermuteQue.EnQue<T>(outputPermuteLocal);
      CopyOut(outoffsetStart, nowlen);
```

#### 机制与设计原因

`initRange` 在每核启动时只执行一次 `CreateVecIndex(rangeLocal, 0, 128)`，所有后续段/tile 复用该只读 TBuf；每 tile 只需一次 `Adds` 把段起点加到 `[0,127]`，避免逐元素标量构造。全局 tile 序号取模把不重叠的输出 tile 唯一映射到 AIV 核，不需要 atomic；尾 tile 保持相同的 128-lane Vector 调用，只缩短 GM CopyOut，避免单独的尾计算模板。

#### 迁移边界

- 适用：输出由“连续整数 range + 每段标量基址”构成，且所有核能以同一确定顺序枚举 tile 的场景。
- 前提：A2/A3、int32、输出段互不重叠、`realCoreNum=GetBlockNum()>0`、尾部 CopyOut 支持字节长度。
- 失效：若各核枚举顺序不同、输出段重叠或需要每 lane 不同变换，模分核不再保证唯一写者；固定仓没有替代 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：短输入偏移表驻留 UB，长表改为 GM 标量读取

#### 触发与源码依据

- 适用 Route：R-001（UB 驻留）和 R-002（GM 标量 fallback）。
- 触发条件：`inputOffset.shape[0] <= 512` 选择 R-001，否则选择 R-002。
- Golden 来源：[`op_host/expand_into_jagged_permute_tiling.cpp:118-126`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/expand_into_jagged_permute/op_host/expand_into_jagged_permute_tiling.cpp#L118-L126)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
static ge::graphStatus SetTilingKey(ExpandIntoJaggedPermuteParam& param) {
    uint64_t tilingkey = 0;
    int64_t limitedSize = 512;
    if(param.inputoffset <= limitedSize){
        param.tilingKey = tilingkey;
    }else{
        param.tilingKey = tilingkey + 1;
    }
    return ge::GRAPH_SUCCESS;
}
```

#### 机制与设计原因

R-001 在每核开始时用一次 `DataCopyPad` 把完整 inputOffset 搬入 UB，后续按 `permuteId` 使用 local `GetValue`；R-002 不执行这次搬运，直接用 GM `GetValue`，避免把大表内容搬入后再随机取值。key 只改变输入起点的读取介质，段长、range 生成和 CopyOut 路径保持一致。需要注意：当前 `InitBuffer` 对 R-002 仍分配 `(P+1)*4` 字节的 inputOffsetsQueue，故源码只证明减少一次实际搬运，不能声称消除了对应 UB 容量。

#### 迁移边界

- 适用：随机索引的小查找表可整表驻留，大表只需稀疏读取的场景。
- 前提：A2/A3、int32；短表满足 `len<=512`；两 Route 均须满足实际 UB 分配可行性。
- 失效：大表 Route 若仍按完整长度 InitBuffer，可能受 UB 容量限制；固定源码没有基于 `maxCoreMemery` 的进一步 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码本身提供了边界：不能把 key 1 描述为“完全不占 inputOffset UB”，因为 `ExpandIntoJaggedPermute::Init` 对两个模板无条件调用 `InitBuffer(inputOffsetsQueue, ..., offsetLen*sizeof(T))`。

## 模板与实现对比

| 对比维度 | R-001 UB 预载输入偏移 | R-002 GM 标量输入偏移 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `inputOffset.len<=512`，key 0 | `inputOffset.len>512`，key 1 | 固定 512 项阈值，单层 if/else |
| 模板与 dtype | `<int32,false>` | `<int32,true>` | `T0` 编译期选择输入起点介质 |
| 循环与分核 | 全核扫描、tile 序号模核数 | 相同 | 循环拓扑无差异 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 5 TQue + 1 TBuf；inputOffset Queue 有效驻留 | 同样分配 5 TQue + 1 TBuf；inputOffset Queue 不消费 | 无 SIMT dynamic UB、NDDMA 或通信 window |
| 内存总占用与复用 | 请求 `7168+4*(P+1)` B，且 `P+1<=512` | 同公式但 `P+1>512`，无 Host UB guard | key 1 减少搬运，不减少当前源码的分配容量 |
| 搬运/Vector/SIMT/NDDMA/通信 API | inputOffset 一次 `DataCopyPad`，local `GetValue`；公共 `Sub/Adds` | GM `GetValue`；公共 `Sub/Adds` | 输入查表位置不同 |
| 同步、精度与 tail | Queue 顺序、int32、满 128 Adds + 有效尾 CopyOut | 相同；GM 标量无需 Queue | 无 cast/round/atomic |
| fallback/失败边界 | 条件不满足进入 R-002 | 无后续 fallback；未知 key 无输出 | 两 Route 都缺少 UB 上界 guard |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` config | 公开支持；`ascend910_93` config | 公开文档明确不支持；未生成 `_950` 卡 |
| Host/Launcher | 共用同一 OpDef、Tiling 和两条 key Route | 共用 | N/A |
| Kernel/资源 | binary config 启用同一 `ExpandIntoJaggedPermute`，blockDim/UB 从平台运行信息读取 | 同左 | binary config 无 `ascend950`，OpDef 无对应 AddConfig |

## 限制与失败边界

- 文档要求 `len(inputOffset)=len(outputOffset)=P+1`，但 `InputParamCheck` 中对应源码位于提前 `return GRAPH_SUCCESS` 之后而不可达；不能依赖当前 Tiling 动态执行该校验。
- Tiling dtype 判断使用 `a!=b && a!=c && b!=c`；任意两者不同的通常校验语义并未由该表达式实现。OpDef 的三个输入均固定 int32，生成框架是否在更早处拒绝异常 descriptor 不在固定仓可见范围。
- `outputSize` 的公开 ACLNN 类型为 `int64_t`，但 Tiling 以 `int32_t*` 读取属性；超出 int32 范围时的转换/拒绝语义固定仓不可见，Kernel 也不以该字段做写边界检查。
- 两 Route 都无条件为 `inputOffsetsQueue` 请求 `(P+1)*4` 字节，且 Host 虽查询 UB 容量却不作比较。尤其 R-002 没有源码可证明的最大 P、fallback 或错误终点；复用实现时必须补做目标平台实际 UB 可行性分析，不能从本卡推定任意长输入都安全。
- `realCoreNum`、`frontCoreNum`、`tailCoreBlockFactor` 三个 POD 字段未由 Host setter 写入；实际核数取 `GetBlockNum()`，后两者后续不使用，因此有效循环不依赖它们。不得复制这些 inactive 字段作为新逻辑依据。
- entry 对未知 TilingKey 没有报错分支，会直接无输出返回；固定 Host 仅产生 0/1。
- Kernel 不校验 `permuteId`、offset 单调性、段边界或 int32 `Adds` 溢出，依赖公开接口前置约束；生成 ACLNN bridge 的值级检查不可见。
- `TPipe::InitBuffer` 的内部对齐粒度与实际物理占用属于仓外 primitive 实现，本卡字节账是源码请求容量，不推测隐藏 padding。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/expand_into_jagged_permute/docs/aclnnExpandIntoJaggedPermute.md` | `aclnnExpandIntoJaggedPermuteGetWorkspaceSize`、`aclnnExpandIntoJaggedPermute` | 两段式原型、语义、参数、错误与 A2/A3/950 产品表 |
| 入口实现 | `index/expand_into_jagged_permute/op_host/expand_into_jagged_permute_def.cpp` | `ops::ExpandIntoJaggedPermute` / `OP_ADD` | int32/ND、AutoContiguous、属性及 ascend910b/910_93 注册；生成 bridge 边界 |
| Host/Tiling/Launcher | `index/expand_into_jagged_permute/op_host/expand_into_jagged_permute_tiling.h` | `ExpandIntoJaggedPermuteParam` / `ExpandIntoJaggedPermuteTilingData` | Host 参数与 POD 字段 |
| Host/Tiling/Launcher | `index/expand_into_jagged_permute/op_host/expand_into_jagged_permute_tiling.cpp` | `InputParamCheck` / `Init` / `SetTilingKey` / `SetTilingData` / `TilingForExpandIntoJaggedPermute` | 校验、workspace、分块、key、POD 与 blockDim |
| Kernel entry/body | `index/expand_into_jagged_permute/op_kernel/expand_into_jagged_permute.cpp` | `expand_into_jagged_permute` | key 0/1 router 与模板实例 |
| Kernel entry/body | `index/expand_into_jagged_permute/op_kernel/expand_into_jagged_permute.h` | `ExpandIntoJaggedPermute<T,T0>` 全部方法 | Queue/TBuf、循环、搬运、Vector、模分核与 tail |
| 构建接线 | `index/expand_into_jagged_permute/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn)` | 公开 ACLNN OpDef 接线 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `ExpandIntoJaggedPermute` 条目 | 仅 `ascend910b`、`ascend910_93` Kernel binary 接线及 auto_sync |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 10 个 int64 声明字段，实际活跃字段为 blockFactor、oneTaskLen、lastTaskLen、oneTaskOffsetLen、inputLen、offsetLen、outputSize | 字段顺序、Host/Device 同一宏结构；三个未 setter 字段不可作为有效逻辑 | `ExpandIntoJaggedPermuteTilingData` / `SetTilingData` / `Init` |
| 调用顺序 | range 初始化→可选 inputOffset 预载→batch 三路搬入→Sub→段/tile→Adds→CopyOut | 全核必须以同序推进 `taskidA`，即使当前核不写也必须推进 src/dst | `Process` |
| workspace/内存 | Kernel 无 operator-owned workspace；6 个 UB buffer 的请求账可复用 | depth=1、固定 512/128 容量、R-002 未消除 inputOffsetsQueue、UB 上界需外部补证 | Tiling `Init` / Kernel `Init` |
| 同步 | TQue 的 Alloc/EnQue/DeQue/Free 顺序连接 DMA、Vector 和 CopyOut | depth-1 buffer 必须消费后再覆盖；模分核依赖输出 tile 不重叠 | `CopyInPermute` / `CopyInOffset` / `CopyOut` / `Process` |
| 精度与尾块 | int32 range + int32 scalar Adds；尾 tile 满算、只写有效前缀 | scalar cast、128-lane Adds、`nowlen` CopyOut 顺序不能随意改 | `initRange` / `Process` / `CopyOut` |
| 失败边界 | Host 只产生 key 0/1；公开约束承担值域与单调性 | 不可假定不可达 shape 检查、UB guard、未知 key fallback 或 950 支持存在 | `InputParamCheck` / `SetTilingKey` / entry router / API 产品表 |
