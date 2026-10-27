---
type: Operator
title: "ForeachAddScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachAddScalarV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/docs/aclnnForeachAddScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_host/op_api/foreach_add_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_host/foreach_add_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_host/config/ascend950/foreach_add_scalar_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:30:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAddScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddScalarV2GetWorkspaceSize` + `aclnnForeachAddScalarV2`，公开 ACLNN V2；底层 launcher op type 为 `ForeachAddScalar`。 |
| 计算语义 | `out[i][j] = x[i][j] + scalar`，逐 TensorList、逐线性元素。 |
| 输入输出 | 最多 50 个 FP16/FP32/INT32/BF16 ND Tensor；V2 scalar 是 `aclScalar`，进入 Kernel 前转为 FP16/FP32/INT32 单元素 Tensor。 |
| 执行类型 | 框架前处理可含 `Contiguous`；核心是单个 arch35 `AIV_VECTOR` Kernel。调用链只有 `DataCopyPad/Cast/Adds`，没有可达 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP16 FP32 staging；R-002 FP32 direct；R-003 INT32 direct；R-004 BF16 FP32 staging。 |
| 核心资源 | `inQueue/outQueue` 均 depth2；FP16/BF16 另有 FP32 `castBuf`；Host 固定预留 32 B workspace，Device body 不使用 workspace。 |
| 来源闭包 | `source_closed`：公开 V2→ACLNN 实现→L0 launcher→regbase selector/Tiling/POD→四 key router→基类/派生 body→ascend950 binary/CMake 全闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `x=[x_0,...,x_{n-1}]` 的每个有效线性元素计算 `y_i[j]=x_i[j]+s`。FP32 与 INT32 在原 dtype 上执行 `Adds`。950 的 FP16 和 BF16 都先以 `CAST_NONE` 升为 FP32，在同一 FP32 `castBuf` 原地加 `float(scalarVal)`，再以 `CAST_RINT` 降回输出 dtype。这一精度路线与 A2/A3 的 FP16 direct 不同。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList，0–8 维/项，公开 ND | FP16、FP32、INT32、BF16 | 最多 50 项；列表内 dtype 一致；支持空 Tensor 和空列表。公开文档不支持非连续 Tensor，API 实现仍经过框架 Contiguous。 |
| `scalar` | `aclScalar` | FP16、FP32、DOUBLE、INT32、INT64，按 x 约束 | x=FP16 时允许 FP16/FP32/DOUBLE；x=FP32/BF16 时允许 FP32/DOUBLE；x=INT32 时允许 INT32/INT64。 |
| `out` | 与 x 等长；API 实际要求逐项 shape 相等；公开 ND | 与 x 相同 | 最多50项，不支持非连续输出。 |
| `workspaceSize` | 标量输出 | `uint64_t` | 空列表为0；核心节点请求32 B，executor 还可能计入框架前处理。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建并持有转换结果/launcher，第二段在传入 stream 执行。 |

### 入口与版本

固定仓同时公开旧 `aclnnForeachAddScalar` 与 V2；V2 把 scalar 从单元素 `aclTensor` 改为 `aclScalar`，950 公开支持表明确为支持。本卡按版本族最新入口使用 V2。V2 Host 将 scalar 物化成底层 Tensor：BF16 x 一律转 FP32；FP16 x 且 scalar 为 FP32/DOUBLE 时在 regbase 平台转 FP32；其余转 x dtype。五个 ascend950 binary 条目闭合 FP16/FP16、FP16/FP32、FP32/FP32、INT32/INT32、BF16/FP32 组合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 创建 executor；检查 `x/scalar/out` 非空与两列表等长。
2. `IsRegbase()` 使 950 进入 FP16/FP32/INT32/BF16 支持集；空 TensorList 在 dtype check 中直接成功。
3. 非空列表检查所有 x dtype 相同、所有 out dtype 与 x 相同；检查 scalar 组合。950 特有地允许 FP16 x + FP32 scalar。
4. 检查 x 维数不超过8、每对 x/out shape 完全相等，并拒绝 private format。
5. 空列表直接返回 `workspaceSize=0`；非空时逐 x 调用 `l0op::Contiguous`，组装新 TensorList。
6. 按前述规则 `ConvertToTensor` 物化 scalar；然后 `l0op::ForeachAddScalarV2` 把 `ForeachAddScalar` AICore 节点加入 executor。
7. 第一段取 executor 总 workspace 并转交 executor；第二段 `CommonOpExecutorRun` 在指定 stream 上执行。

### 参数、Workspace 与生命周期

executor 持有每个 Contiguous 结果、scalar Tensor 和 Foreach launcher，直到第二段执行结束。核心 regbase Tiling 的 `PostTiling` 写 `currentWorkspace[0]=32`；arch35 entry 把 workspace 原样传给派生 `Init`，但 `ForeachRegbaseUnary::Init` 不读该参数，因此没有 Kernel 间 workspace edge。输入 Contiguous 的实际 workspace/别名决策属于框架边界，固定仓不可见；公开文档本身不承诺非连续输入支持。

### 启动图与失败传播

```text
V2 phase-1
  -> [framework Contiguous(x_i)]*n
  -> ConvertToTensor(scalar, FP16/FP32/INT32)
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachAddScalar)
V2 phase-2 on caller stream
  -> [optional framework copy nodes]
  -> foreach_add_scalar<<<numBlocks, AIV>>>(ListTensorDesc inputs, scalarTensor,
                                            ListTensorDesc outputs, reservedWS, POD)
```

核心语义是单 Kernel。所有 executor 节点在同一 stream 中保持 producer→consumer 顺序；任一参数转换或 launcher 添加失败会在第一段返回错误，不会提交核心节点。Kernel entry 无返回值，unknown key 不显式报错；公开路径依赖 Host selector 与 binary config 保证 key/实例匹配。

## Route 与参数设计

### 分核与资源计算

设 `s=sizeof(T)`、总元素数 `D=sum numel(x_i)`，固定单核分配粒度为 `E=4096/s` 元素：

```text
numBlocks = min(platform_AIV_cores, 80, ceil(D/E))
if D==0: numBlocks=1, start=0, end=0, endOffset=-1
else:
  blockCount=ceil(D/E)
  base=floor(blockCount/numBlocks)*E
  early cores (< blockCount%numBlocks) process base+E elements
```

`AssignDataToEachCore` 把连续域映射成每核的 Tensor 起止索引和闭区间 offset。设平台 UB 为 `Uplat`、每 tile 元素数为 `N=inputsTensorUbSize`：FP32/INT32 的逐元素 UB 字节为 `4*s=16`；FP16/BF16 为 `4*s+4=12`（两个 depth2 Queue 加一个 FP32 cast buffer）。Host 先 `N0=floor(Uplat/bytesPerElement)`，再按 `floorAlign(N0*s,32)/s` 得 N。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 FP32 staging | key10001；`ForeachAddScalarRegbase<half,DTYPE_SCALAR,...>` | priority10000 membase 拒绝 950 后，regbase template priority30000 的 `IsRegbaseSocVersion=true`，且 x=FP16；binary 选择 `DTYPE_SCALAR=half/float` | dtype 不同按后续 key |
| 2 | R-002 FP32 direct | key10002；`<float,float>` | 同 selector，x=FP32，scalar materialized FP32 | 后续 key |
| 3 | R-003 INT32 direct | key10003；`<int,int>` | 同 selector，x=INT32，scalar materialized INT32 | 后续 key |
| 4 | R-004 BF16 FP32 staging | key10004；`<bfloat16_t,float>` | 同 selector，x=BF16，scalar materialized FP32 | unknown key 无 body |

```text
TilingRegistry 按 priority 数值升序尝试（数值越小越先）
priority 10000 ForeachAddScalarMembaseTiling
  GetShapeAttrsInfo(no-op) -> GetPlatformInfo
  IsCapable = !IsRegbaseSocVersion(A5) -> false -> GRAPH_PARAM_INVALID
priority 30000 ForeachRegbaseTilingUnaryScalar
  GetShapeAttrsInfo -> Check x/list/shape + CheckScalar
  GetPlatformInfo
  IsCapable = IsRegbaseSocVersion(A5) -> true
  DoOpTiling -> partition + CheckOutput + UB split
  x FP16  -> 10001 -> R-001
  x FP32  -> 10002 -> R-002
  x INT32 -> 10003 -> R-003
  x BF16  -> 10004 -> R-004
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 10001 | 950；x/out FP16；scalar FP16/FP32/DOUBLE 经转换为 FP16或FP32；1–50列表项 | 单 arch35 AIV node | `ForeachRegbaseUnary<half>` + `ForeachAddScalarRegbase<half,DTYPE_SCALAR>` | in/out Queue depth2 + FP32 castBuf；Queue ordering | 每 tile `min(N,remaining)`；DataCopyPad valid bytes |
| R-002 FP32 | 10002 | x/out FP32；scalar FP32/DOUBLE→FP32 | 同上 | `<float,float>` direct Adds | 两个 depth2 Queue，无 castBuf | 同 tail |
| R-003 INT32 | 10003 | x/out INT32；scalar INT32/INT64→INT32 | 同上 | `<int,int>` direct Adds | 两个 depth2 Queue | 同 tail；溢出策略固定仓未另述 |
| R-004 BF16 | 10004 | x/out BF16；scalar FP32/DOUBLE→FP32 | 同上 | `<bfloat16_t,float>` Cast/Adds/Cast | 两个 depth2 Queue + FP32 castBuf | 同 R-001 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| regbase selector | priority10000 membase 先返回 `GRAPH_PARAM_INVALID`，随后 priority30000 的 `IsRegbaseSocVersion(context_)` 成立 | arch35 使用 `ListTensorDesc`、regbase POD 与 key1000x | 全部 | 若 regbase 也不匹配则 tiling失败 | `TilingRegistry::DoTilingImpl`、两个 `IsCapable` |
| Tensor 数 | `1<=count<=50`（非空 launch） | POD tensor数组50；文档也给950最大50 | 全部 | Tiling GRAPH_FAILED；空列表在 ACLNN 层提前成功 | `GetShapeAttrsInfo` |
| core 上限 | `numBlocks<=80` | per-core POD数组长度80 | 全部 | 截断到80 | `DoOpTiling` |
| 4KB分核 | `elementsPerBlock=4096/s` | 每核至少按固定字节粒度分配 | 全部 | ceil后尾核只处理有效域 | `ForeachRegbaseTiling::DoOpTiling` |
| scalar lower组合 | `{F32/F32,F16/F16,F16/F32,BF16/F32,I32/I32}` | 决定 `DTYPE_SCALAR` 与安全读取类型 | 全部 | Tiling失败 | `SUPPORT_DTYPE_COMB`、`CheckScalar` |
| 32B UB对齐 | `N=floorAlign(floor(Uplat/B)*s,32)/s` | DataCopyPad/Queue容量按block对齐 | 全部 | 没有 fallback；N=0 会使 Device `CEIL_DIV` 除零，Host未显式拒绝 | `ForeachRegbaseTilingUnaryScalar::DoOpTiling` |
| POD/计数窄化 | Host 的 `inputsTensorUbSize` 为 `int64_t` 计算结果但 POD 字段为 `uint32_t`；`ceil(totalDataCount/E)` 存入 `uint32_t tempCoreNum` | Device N 与 blockDim 依赖32位值 | 全部 | 无显式溢出 fallback | unary scalar/base `DoOpTiling` |
| 空元素域 | `D==0` | 保持有效 blockDim/POD | 全部 | 1 core，`endOffset=-1`，Device计算 `dataCount=0`，tile循环0次 | base `DoOpTiling`/`Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSoloTilingDataRegbase` | `inputsTensorUbSize` | `uint32_t`/32 | 第1字段；宏最终 `sizeof/alignof/offsetof` 固定仓不可见 | 元素/Queue slot | `ForeachRegbaseTilingUnaryScalar::DoOpTiling` | `ForeachRegbaseUnary::ParseTilingData` | 使 `N*s` 32B对齐 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 第2；400 B | 元素/Tensor | `GetShapeAttrsInfo`→base `DoOpTiling` | `Process` | 0允许空Tensor |
| 同上 | `tensorStartList` | `uint16_t[80]` | 第3；160 B | Tensor索引/core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | `numBlocks<=80` |
| 同上 | `tensorEndList` | `uint16_t[80]` | 第4；160 B | Tensor索引/core | 同上 | 同上 | 闭区间 |
| 同上 | `tensorStartOffsetList` | `int64_t[80]` | 第5；640 B | 元素/core | 同上 | 同上 | 首Tensor offset |
| 同上 | `tensorEndOffsetList` | `int64_t[80]` | 第6；640 B；显式字段宽度合计2004 B，宏最终 padding/offset固定仓不可见 | 元素/core | 同上 | 同上 | 闭区间；空域为-1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 10001 | `ForeachRegbaseTilingUnaryScalar` | `ForeachSoloTilingDataRegbase`、N/数组 | arch35 `foreach_add_scalar` key10001 | `ForeachRegbaseUnary` + `ForeachAddScalarRegbase<half,DTYPE_SCALAR>` | ascend950 binary 两个FP16组合 + arch35 CMake |
| R-002 | 10002 | 同上 | 同上 | key10002 | `<float,float>` | FP32 binary + arch35 CMake |
| R-003 | 10003 | 同上 | 同上 | key10003 | `<int,int>` | INT32 binary + arch35 CMake |
| R-004 | 10004 | 同上 | 同上 | key10004 | `<bfloat16_t,float>` | BF16 binary + arch35 CMake |

## Kernel 实现

### 公共执行流程

entry 在栈上创建 `TPipe`，通过 `GET_TILING_DATA_WITH_STRUCT` 解析 regbase POD，按 key 调 `ForeachAddScalarImpl<T,ScalarT>`，结束时 `pipeOp.Destroy()`。`ForeachRegbaseUnary::Init` 用 `ListTensorDesc` 解析输入/输出地址，初始化两个 depth2 Queue；派生类绑定单元素 scalar GM，并按类型条件初始化 FP32 `castBuf`。每核遍历其 `[tensorStart,tensorEnd]`，把首尾 offset 夹出的有效段按 N 元素切 tile，依次 `CopyIn(DataCopyPad) -> Compute -> CopyOut(DataCopyPad)`。

### Route R-001：FP16 FP32 staging

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.cpp` | dtype check + conversion | regbase、x FP16、scalar FP16/FP32/DOUBLE | lower scalar Tensor FP16或FP32 |
| Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `CheckScalar`/`GetTilingKey` | `{F16,F16}`或`{F16,F32}` | key10001 |
| Router | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp` | key10001 | `DTYPE_SCALAR`由binary特化 | half/ScalarT body |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_scalar
  -> ForeachAddScalarImpl<half,DTYPE_SCALAR>
    -> ForeachAddScalarRegbase::Init
      -> ForeachRegbaseUnary::Init/ParseTilingData
    -> ForeachRegbaseUnary::Process/SingleTensorProcess
      -> CopyIn
      -> ForeachRegbaseUnary::Compute -> ForeachAddScalarRegbase::Compute
        -> Cast(half->float, CAST_NONE) -> Adds<float> in-place -> Cast(float->half, CAST_RINT)
      -> CopyOut
  -> TPipe::Destroy
```

##### 源码等价伪码

```text
# 伪码
scalarVal = scalarGM[0]  # ScalarT=half 或 float
for tensor in core_owned_closed_range:
  begin/end = POD offsets
  for tile in ceil(validCount/N):
    count = min(N, remaining)
    xh = DataCopyPad(GM[begin+tile*N], blockLen=count*2, no_pad)
    f[0:count] = Cast(xh, CAST_NONE)
    f[0:count] = Adds(f, float(scalarVal), count)
    yh = Cast(f, CAST_RINT, count)
    DataCopyPad(GM_out, yh, blockLen=count*2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `ListTensorDesc[i]+cursorStart+tile*N` | count elements | FP16 ND | caller/Kernel | Kernel/caller | caller | blockLen=`count*2` B |
| `inQueue` | pipe slot | depth2 × `N*2` B | FP16 | CopyIn | Compute | TPipe/tile | double buffer |
| `outQueue` | pipe slot | depth2 × `N*2` B | FP16 | Compute | CopyOut | TPipe/tile | double buffer |
| `castBuf` | one TBuf | `N*4` B | FP32 | up-Cast/Adds | down-Cast | TPipe/Kernel | same subregion in-place reused |
| scalar GM | element0 | 2或4 B | ScalarT | executor | Init | executor/kernel | read once/core |
| workspace | unused | 0 B Device; 32 B Host reserved | N/A | Host | none | executor | no cross-node edge |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  in0["inQueue slot0 · N·2 B"]:1
  in1["inQueue slot1 · N·2 B"]:1
  cast["castBuf · N·4 B"]:1
  out0["outQueue slot0 · N·2 B"]:1
  out1["outQueue slot1 · N·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*N*2 + 2*N*2 + N*4 = 12N bytes
bytesPerElement(host)=2*2 + 2*2 + 4 = 12
N=floorAlign(floor(Uplat/12)*2,32)/2
UB_total<=Uplat
dynamic_ub=N/A; workspace_total=32 B reserved/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `castBuf[0:N)` | up-Cast | in-place Adds、down-Cast | 下一 tile up-Cast | Compute调用顺序；Queue在Compute返回后EnQue | down-Cast完成前不可覆盖 |
| Queue slot0/1 | CopyIn/Compute | Compute/CopyOut | 双缓冲轮转 | Alloc/EnQue/DeQue/Free | consumer释放后复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | inQueue | DataCopyPad | Compute DeQue | FP16 tile | no pad；count可小于N/为0 |
| Vector chain | 同一Compute顺序 | Cast/Adds | 下一Cast | castBuf | FP32 add，回写`CAST_RINT` |
| UB→GM | outQueue | Compute EnQue | CopyOut DeQue | FP16 result | 仅写`count*2` B |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | unary scalar `DoOpTiling` | 12N字节、key10001、blockDim |
| Kernel entry/router | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp` | half/DTYPE_SCALAR | binary scalar specialization |
| Device body/helper | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar_regbase.h`、`foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary.h` | Init/Compute/Process | Queue、Cast/Adds、tail |

### Route R-002：FP32 direct

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.cpp` | dtype/conversion | x FP32，scalar FP32/DOUBLE | lower FP32 |
| Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `GetTilingKey` | `DT_FLOAT` | 10002 |
| Router | arch35 `foreach_add_scalar.cpp` | key10002 | true | `<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_scalar -> ForeachAddScalarImpl<float,float>
  -> ForeachRegbaseUnary::Init/Process
    -> CopyIn -> ForeachAddScalarRegbase::Compute -> Adds<float> -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
for each owned slice, for each N-element tile:
  count=min(N,remaining)
  xLocal=DataCopyPad(blockLen=count*4)
  yLocal=Adds<float>(xLocal, scalar_fp32, count)
  DataCopyPad(out, yLocal, blockLen=count*4)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | tensor base + offsets | count FP32 | ND | caller/Kernel | Kernel/caller | caller | valid bytes=count*4 |
| in/out Queue | pipe slots | each depth2×N×4 B | FP32 | CopyIn/Adds | Adds/CopyOut | TPipe/tile | double buffered |
| castBuf | N/A | 0 | N/A | N/A | N/A | N/A | compile-time not allocated |
| workspace | unused | Device0/Host32 B | N/A | Host | none | executor | reserved |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["inQueue slot0 · N·4 B"]:1
  in1["inQueue slot1 · N·4 B"]:1
  out0["outQueue slot0 · N·4 B"]:1
  out1["outQueue slot1 · N·4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=2*N*4+2*N*4=16N
N=floorAlign(floor(Uplat/16)*4,32)/4
dynamic_ub=N/A; workspace=32 B reserved/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| in/out Queue slot | CopyIn/Adds | Adds/CopyOut | double-buffer轮转 | Queue lifecycle | consumer释放后 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| copy→Adds→copy | Queue protocol | each stage | next stage | tile slot | FP32 direct；tail count controls DMA/Adds |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `DoOpTiling/GetTilingKey` | 16N、key10002 |
| Kernel entry/router | arch35 `foreach_add_scalar.cpp` | float branch | direct body |
| Device body/helper | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar_regbase.h` | non-16bit `Compute` | `Adds(out,in,scalar,count)` |

### Route R-003：INT32 direct

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.cpp` | `ForeachAddScalarV2CheckDtypeValid`/conversion | x INT32，scalar INT32/INT64 | lower INT32 |
| Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `CheckScalar/GetTilingKey` | `{I32,I32}` | 10003 |
| Router | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp` | `foreach_add_scalar` key10003 | true | `<int,int>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_scalar -> ForeachAddScalarImpl<int,int>
  -> ForeachRegbaseUnary::Process -> CopyIn -> Adds<int> -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
for each core-owned N-element tile:
  count=min(N,remaining)
  y_i32[0:count]=Adds<int32>(x_i32[0:count], scalar_i32)
  commit count*4 bytes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | TensorDesc base + offset | count INT32 | ND | caller/Kernel | Kernel/caller | caller | valid bytes=count*4 |
| in/out Queue | slots | each depth2×N×4 B | INT32 | CopyIn/Adds | Adds/CopyOut | TPipe/tile | double buffer |
| castBuf/workspace | N/A | 0 Device | N/A | Host | none | executor | Host32 B reserved |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  in0["inQueue slot0 · N·4 B"]:1
  in1["inQueue slot1 · N·4 B"]:1
  out0["outQueue slot0 · N·4 B"]:1
  out1["outQueue slot1 · N·4 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=16N; N=floorAlign(floor(Uplat/16)*4,32)/4
dynamic_ub=N/A; workspace=32 B reserved/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue slot0/1 | CopyIn/Adds | Adds/CopyOut | next tile | Alloc/EnQue/DeQue/Free | consumer completed |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| INT32 pipeline | Queue protocol | CopyIn | Adds/CopyOut | tile | direct INT32；overflow semantics未额外定义；tail按count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | INT dtype map | key10003、16N |
| Kernel entry/router | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp` | int branch | `<int,int>` |
| Device body/helper | derived/base headers | non-16bit Compute/Copy | direct integer vector add |

### Route R-004：BF16 FP32 staging

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.cpp` | BF16 conversion branch | x BF16、scalar FP32/DOUBLE | scalar FP32 Tensor |
| Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | `CheckScalar/GetTilingKey` | `{BF16,F32}` | 10004 |
| Router | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp` | `foreach_add_scalar` key10004 | true | `<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_add_scalar -> ForeachAddScalarImpl<bfloat16_t,float>
  -> Init (bind scalar FP32 + allocate castBuf)
  -> base Process -> CopyIn
  -> derived Compute: Cast(BF16->FP32) -> Adds<float> -> Cast(FP32->BF16,RINT)
  -> CopyOut -> pipe.Destroy
```

##### 源码等价伪码

```text
# 伪码
for tensor slice, for tile <=N:
  count=min(N,remaining)
  xb=DataCopyPad(blockLen=count*2)
  f=Cast(xb, CAST_NONE, count)
  f=Adds(f, scalar_fp32, count)
  yb=Cast(f, CAST_RINT, count)
  DataCopyPad(yb, blockLen=count*2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | ListTensorDesc+offset | count BF16 | ND | caller/Kernel | Kernel/caller | caller | count*2B |
| in/out Queue | slots | each depth2×N×2B | BF16 | CopyIn/Cast | Cast/CopyOut | pipe/tile | double buffer |
| castBuf | offset0 | N×4B | FP32 | Cast/Adds | Adds/downCast | pipe/Kernel | in-place Adds |
| scalar GM | 0 | 4B FP32 | scalar | executor | Init | executor/kernel | read once/core |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  in0["inQueue slot0 · N·2 B"]:1
  in1["inQueue slot1 · N·2 B"]:1
  cast["castBuf · N·4 B"]:1
  out0["outQueue slot0 · N·2 B"]:1
  out1["outQueue slot1 · N·2 B"]:1
```

##### 字节账与复用顺序

```text
UB_total=12N
N=floorAlign(floor(Uplat/12)*2,32)/2
dynamic_ub=N/A; workspace=32 B reserved/0 B consumed
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| castBuf | up-Cast | Adds/down-Cast | next tile | same Compute call order | down-Cast before overwrite |
| queue slots | CopyIn/down-Cast | Compute/CopyOut | double-buffer rotation | Queue lifecycle | release after consumer |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue copy | EnQue/DeQue | CopyIn | Compute | BF16 tile | tail count controls blockLen |
| FP32 staging | sequential Vector APIs | Cast/Adds | down-Cast | castBuf | `CAST_NONE` up，`CAST_RINT` down |
| output | outQueue | Compute | CopyOut | BF16 tile | exact count bytes |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | scalar check/UB formula/key | BF16/F32、12N、10004 |
| Kernel entry/router | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp` | BF16 branch | bfloat16/float |
| Device body/helper | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar_regbase.h` | 16-bit `Init/Compute` | castBuf、FP32 add、RINT |

## 算子特定优化

### Trick T-001：编译期条件分配与 FP32 castBuf 原地复用

#### 触发与源码依据

- 适用 Route：R-001、R-004。
- 触发条件：`T` 为 `half` 或 `bfloat16_t`；FP32/INT32 Route 在编译期消除 castBuf 初始化与 Cast 链。
- Golden 来源：[`foreach_add_scalar_regbase.h:31-49`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar_regbase.h#L31-L49)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Base::Init(inputs, outputs, workspace, tilingData, tPipe);
inScalarGM.SetGlobalBuffer((__gm__ ScalarT*)scalar, 1);
if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
    scalarVal = inScalarGM.GetValue(0);
    tPipe->InitBuffer(castBuf, tilingData->inputsTensorUbSize * sizeof(float));
    castLocal = castBuf.Get<float>();
} else {
    scalarVal = T(inScalarGM.GetValue(0));
}
```

#### 机制与设计原因

`if constexpr` 让 FP32/INT32 实例完全不分配 castBuf，UB 只承担两个 depth2 Queue（16N B）。FP16/BF16 才额外分配 N×4 B，且同一 `castLocal` 先接收升精度结果，再由 `Adds(castLocal,castLocal,...)` 原地覆盖，最后直接作为降精度输入，避免第二个 FP32 临时区。Host 的 `ubSizePerNumber` 同步从 16 B/elem 改为 12 B/elem，和 Device 实际分配闭合。

#### 迁移边界

- 适用：16bit 输入需要 FP32 逐元素计算，而 32bit 路线可直接计算的 unary-scalar 算子。
- 前提：Host 以 Queue depth2 和 castBuf 逐项计费；同一 tile 的 Cast/Adds/Cast 无跨元素依赖；scalar 类型与 binary config 一致。
- 失效：若计算同时需要原 FP32 输入与结果，不能原地 Adds，必须增加独立 TBuf 并重算 Host N。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | 10001 | 10002 | 10003 | 10004 | priority10000 membase 先拒绝 950，随后 priority30000 regbase；x dtype决定key |
| 模板与 dtype | half/half或float scalar | float/float | int/int | bf16/float | ACLNN物化 + binary DTYPE_SCALAR |
| 循环与分核 | E=2048；tile N | E=1024 | E=1024 | E=2048 | 固定4096B/core粒度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | depth2 in/out + castBuf | depth2 in/out | depth2 in/out | depth2 in/out + castBuf | 16bit FP32 staging |
| 内存总占用与复用 | 12N | 16N | 16N | 12N | conditional allocation；castBuf in-place |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad/Cast/Adds/Cast | DataCopyPad/Adds | 同 | 同FP16 | 全部AIV Vector |
| 同步、精度与 tail | FP32 + RINT | FP32 direct | INT32 direct | FP32 + RINT | Queue protocol + count tail |
| fallback/失败边界 | scalar binary缺失则不可启动 | unknown key no body | 同 | 同 | Host/API/binary必须一致 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持V2 | 支持V2 | 支持V2 |
| Host/Launcher | membase key1–4、最多50 core | 同A2 | regbase key10001–10004、最多80 core、列表最多50 |
| Kernel/资源 | FP16 direct；Queue depth1；BF16分层cast tile | 同A2 | FP16/BF16均FP32 staging；Queue depth2；ListTensorDesc |

## 限制与失败边界

- 950 公开文档明确 TensorList 最大50；Host regbase Tiling 对非空 launch 强制 `1..50`。空列表由 ACLNN 层提前成功，不进入该限制。
- 文档宣称 out shape size 可大于等于 x，但 V2 API C++ 先执行 shape 完全相等检查；公开 V2 实际有效域取实现的严格条件。
- 公开格式是 ND 且 x/out 非连续不支持；虽然 API 对输入逐项调用 Contiguous，也不能据此扩展公开契约，输出从不做 ViewCopy/Contiguous。
- `ForeachSoloTilingDataRegbase` 的宏实现位于外部依赖；固定仓只确认字段顺序/宽度合计2004 B，最终 padding、`sizeof/alignof/offsetof` 不可见。
- Host 以 `int64_t` 计算 N 后写入 `uint32_t` POD，并把 `ceil(totalDataCount/E)` 存入 `uint32_t`；总元素累计也没有显式溢出检查。工程复用不得扩大到超过这些位宽的域。
- 当 `N=0` 时 Device 的 `CEIL_DIV(dataCount,N)` 不安全；Host UB公式未显式拒绝极小UB，但实际950平台UB应由平台信息提供。本卡不推测平台最小UB。
- core workspace 预留32 B但未消费；executor总 workspace 受框架Contiguous影响，固定仓没有闭合其精确公式。
- arch35 entry 对 unknown key 没有显式错误或 fallback；`pipeOp.Destroy()`仍执行，但不产生输出。正常路径必须保持 selector、POD、binary config与entry一致。
- INT32 `Adds` 的溢出/饱和细节未在本算子固定源码声明；不得外推。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_add_scalar/docs/aclnnForeachAddScalarV2.md` | 支持表/参数表 | 950公开支持、dtype/数量/shape约束 |
| 入口声明/约束 | `foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.h` | V2 APIs | C ABI |
| 入口实现 | `foreach/foreach_add_scalar/op_host/op_api/aclnn_foreach_add_scalar_v2.cpp` | check/Exec/API | 950 scalar组合、转换、executor |
| 入口实现 | `foreach/foreach_add_scalar/op_host/op_api/foreach_add_scalar_v2.cpp` | L0 function | launcher添加 |
| Host注册 | `foreach/foreach_add_scalar/op_host/foreach_add_scalar_def.cpp` | `regbaseCfg` | ascend950 dtype签名/compute unit |
| 构建接线 | `foreach/foreach_add_scalar/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN Host源与`foreach_utils`依赖 |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachSoloTilingDataRegbase` | 字段及key注册 |
| Tiling入口 | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachAddScalarTiling` | registry入口/membase对照 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.h` | class declarations | selector/状态/POD owner |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_regbase_tiling.cpp` | base/unary scalar functions | checks、分核、UB、key、PostTiling |
| Host/Tiling | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | 数值升序、模板 fallthrough |
| Host/Tiling | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | GetShape→Platform→IsCapable→DoOp→DoLib→Workspace→Post→SetKey 顺序 |
| 构建接线 | `foreach/foreach_utils/op_host/CMakeLists.txt` | `add_modules_sources` | 公共Tiling/InferShape源进入Host模块 |
| Kernel entry | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar.cpp` | entry/impl | key10001–10004 router |
| Kernel body | `foreach/foreach_add_scalar/op_kernel/arch35/foreach_add_scalar_regbase.h` | derived Init/Compute | scalar、castBuf、precision |
| Kernel helper | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_common.h` | key macros | Host/Device key一致性 |
| Kernel helper | `foreach/foreach_utils/op_kernel/arch35/foreach_regbase_unary.h` | Init/Process/Copy methods | ListTensorDesc、Queue、DMA、tail |
| 构建接线 | `foreach/foreach_add_scalar/op_host/config/ascend950/foreach_add_scalar_binary.json` | five binary entries | dtype/ScalarT实例 |
| 构建接线 | `foreach/foreach_add_scalar/op_kernel/CMakeLists.txt` | arch35 source/ascend950 | 活动Kernel选择与编译option |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段regbase TensorList分核POD | N单位是元素，不是字节；Tensor数组50/core数组80；offset闭区间 | POD definition/ParseTilingData |
| 调用顺序 | V2 check→Contiguous→scalar Tensor→launcher→stream run | 空列表必须提前短路；scalar目标dtype须与binary一致 | `ExecForeachAddScalarV2GetWorkspaceSize` |
| workspace/内存 | direct 16N；16bit 12N | Queue depth2必须乘入；castBuf只在half/BF16分配；workspace不参与算法 | Host `DoOpTiling`、Device `InitBuffer` |
| 同步 | Queue lifecycle与单Compute内Vector顺序 | castBuf在down-Cast前不得覆盖，Queue slot在Free前不得复用 | base/derived Compute |
| 精度与尾块 | FP16/BF16 FP32 staging；FP32/INT32 direct | 16bit回写`CAST_RINT`；所有DMA blockLen必须用有效count | derived Compute/base Copy methods |
| 失败边界 | API/tiling checks + priority selector | 不得把>50、unknown key、非连续输出或不匹配ScalarT当作fallback | check functions/router/binary config |