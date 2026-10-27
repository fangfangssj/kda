---
type: Operator
title: "ChunkCat A2/A3 ACLNN 算子知识"
description: "说明 aclnnChunkCat 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/docs/aclnnChunkCat.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_api/aclnn_chunk_cat.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_api/aclnn_chunk_cat.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_api/chunk_cat.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_host/chunk_cat_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_host/chunk_cat_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_host/chunk_cat_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_host/config/ascend910b/chunk_cat_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_host/config/ascend910_93/chunk_cat_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T23:49:48Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ChunkCat A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 最新且唯一公开版本 `aclnnChunkCatGetWorkspaceSize` / `aclnnChunkCat`，两段式 ACLNN |
| 计算语义 | 把每个输入沿 `dim=0` 切为 `numChunks` 个等宽上界块，逐块横向拼接并把短尾块补零，最后转换到 `out` dtype |
| 输入输出 | 非空输入均为同 dtype 的 1～8 维 tensor；FP16/BF16 可输出 FP16/BF16/FP32，FP32 只输出 FP32；内部统一成二维 ND |
| 执行类型 | 每个算子自有节点均为 `AIV_VECTOR`；入口可形成一层单 Kernel 或按 512 路归并的多 Kernel 图 |
| Kernel 路线 | R-001～R-004 同 dtype；R-005～R-008 FP16/BF16 互转；R-009～R-012 FP16/BF16→FP32；每组三条/四条分别覆盖单行、全对齐、半对齐转置和一般转置 |
| 核心资源 | 输入/输出 GM、两个 `VECCALC TBuf`、MTE2/V/MTE3 event；节点间由 executor 持有中间 GM tensor；Device 不读取 workspace 参数 |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 多层 launcher→固定 key 0 Tiling/POD→AIV entry→classic Vector/transpose body→A2/A3 OpDef 与 binary config 闭合；Contiguous/ViewCopy 与 SDK primitive 为明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对第 `i` 个输入，把 `dim=0` 后所有维合为 `c_i=prod(shape_i[1:])`，令 `s_i=shape_i[0]`、`q_i=ceil(s_i/numChunks)`。内部输出是 `[numChunks, sum_i(q_i*c_i)]`：第 `r` 行依次放置每个输入的扁平区间 `x_i[r*q_i:min((r+1)q_i,s_i), :]`，不足 `q_i*c_i` 的尾部置零，再转换为 `out` dtype。公开 API 和 Host/Tiling 都只接受 `dim=0`；结果随后用 View 改回调用者格式并 `ViewCopy` 到 `out`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `tensors` | tensor-list；每项 view rank 1～8，公开格式 ND，API 接受非 private format 并先 `Contiguous` | 列表内一致；FP16/BF16/FP32 | 空列表在建 executor 后直接成功返回；非空列表的 null、dtype、format、rank 由 `CheckParams` 检查 |
| `dim` | scalar | int64 | 公开与 Tiling 仅支持 0；API 行 224～226 只记录错误而未立即返回，后续 infer/Tiling 仍以失败终止非零值 |
| `numChunks` | scalar | int64 | Tiling 要求 `>0`；决定输出行数和每个输入的 `q_i` |
| `out` | 调用者给定 shape/format；内部 L0 输出 ND `[numChunks,sum_i(q_i*c_i)]` | FP16/BF16/FP32 | 输入 FP32 时必须输出 FP32；其余 6 个输入/输出组合均有 binary；最终必须与推导 shape/scalar 一致 |
| `workspaceSize/executor` | 两段式接口输出 | uint64 / opaque | executor 保存连续化、各层 ChunkCat 和最终 ViewCopy；workspace 大小由 executor 汇总 |

### 入口与版本

固定提交只公开无版本后缀的上述接口，未见 V2/V3，因此它是 A2/A3 的最新公开版本。公开文档同时声明 Atlas A2 与 A3 支持；OpDef 注册 `ascend910b`、`ascend910_93`，两平台 binary config 都列出 7 个 dtype 组合且内容一致，平台证据无冲突。API 的 `GetSupportDtypeList` 对 `DAV_2201` 接受 FP32/FP16/BF16；`IsRegBase` 平台也复用同表，但 A2/A3 entry 由 `__CCE_AICORE__ != 310` 选择本卡的 classic body。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段先创建 executor。非空列表随后依次执行 `CheckNotNull→CheckDtypeValid→CheckFormat→CheckShape`；每个非空输入经 `l0op::Contiguous`，`MergeLastDims` 保留 `[0..dim]` 并把后续维乘成末维。`SplitToChunkCat` 每次最多把 512 个 tensor 组成一个 L0 `ChunkCat`；如果一层产生多个中间输出，就把它们作为下一层输入继续归并，直到只剩一个。最后校验内部结果与 `out` shape/scalar 一致，建立与 `out` format 相同的 View，再 `ViewCopy`。任一 Contiguous、InferShape、launcher 入队、shape 检查或 ViewCopy 失败均停止构图。

### 参数、Workspace 与生命周期

每个 L0 节点由 `INFER_SHAPE(ChunkCat)` 分配内部 ND 输出，并由 `ADD_TO_LAUNCHER_LIST_AICORE` 入队。Tiling 把 `GetLibApiWorkSpaceSize()` 写入该节点唯一 workspace size；`chunk_cat(x,y,workspace,tiling)` 不读取 `workspace`，所以这是系统库 workspace 而非节点间数据边。executor 拥有连续化结果、中间 ChunkCat 输出、最终 View 与节点描述，第二段通过同一 `stream` 执行；固定仓未公开 executor 对多节点 workspace 的具体 offset/复用策略和释放时点。

### 启动图与失败传播

设非空输入数为 `N`。若 `N<=512`，自有图只有一个 `ChunkCat` AIV 节点；若更大，第 `k` 层节点数为 `N_{k+1}=ceil(N_k/512)`，各节点消费最多 512 个同层 tensor 并产生内部 GM tensor，直到 `N_L=1`。同层节点互不依赖；下一层只在其所有直接 producer 输出可用后消费。executor/stream 提供框架 ordering，固定仓未暴露独立 event。最后一张内部 GM tensor 经框架 `ViewCopy` 写 `out`；任一 producer 构图/launch 失败应阻断依赖节点，具体运行时 cleanup 属于 `CommonOpExecutorRun` 外部边界。

| 边 | offset/extent/dtype/layout | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|
| 输入→首层节点 | 每个连续化 tensor 全部有效元素，ND | executor 同 stream | executor 持有至运行完成 | Contiguous/L0 失败停止构图 |
| 第 `k` 层→第 `k+1` 层 | producer 输出 `[numChunks,sum(q_i*c_i)]`，dtype 已为 `out` dtype，ND；整 tensor 交接 | executor 依赖图/同 stream，固定仓未见显式 event | executor 内部 tensor；consumer 完成前不可覆盖 | producer 失败阻断 consumer |
| 末层→`out` | 完整内部结果，PostProcess 只改 format metadata | ViewCopy 在末层之后 | executor View 与调用者 out 覆盖异步执行期 | shape/ViewCopy 失败不执行第二段 |

## Route 与参数设计

### 分核与资源计算

Tiling 先取 AIV `coreNum`、`ubSize` 与系统 workspace。令输入字节数 `s=sizeof(T1)`、`E=32/s`。对每个输入计算 `q_i=ceil(shape_i[0]/numChunk)`、`C_i=q_i*prod(shape_i[1:])`，累加 `outputCol=sum C_i`，`outputRow=numChunk`；`allAlign=∀i(C_i%E==0)`，`halfAlign=∀i(C_i%(E/2)==0)`，`oneConcat=(outputRow==1&&!allAlign)`。classic 平台令 `reserve=256`；同 dtype 时 `inUb=(ubSize+reserve)/2`，转换时为 `/3`，`outUb=ubSize+reserve-inUb`。

列/行 UB 因子按源码三分支求得：全对齐或单行先以 `colLimit=inUb/s`（单行再减 `32*(E-1)`）切列，再在只需一列 tile 时扩行；半/非对齐先取 `rowLimit=32`（半对齐）或 `16*E`（一般），再取 `colLimit=inUb/rowLimit/s-2E`。最后 `blockRowFactor=align_up(ceil(outputRow/coreNum),ubRowFactor)`，`blockRowNum=ceil(outputRow/blockRowFactor)`，剩余核数 `leftCore=coreNum/blockRowNum`，再同法切列；`blockDim=blockRowNum*blockColNum`。每核二维坐标为 `(blockIdx/blockColNum, blockIdx%blockColNum)`，最后行/列核使用 tail factor。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001/R-005/R-009 单行拼接 | key 0；按 dtype binary 实例 | `numChunk==1 && !allAlign`；分别为同 dtype、16-bit 互转、16→FP32 | 否则检查全对齐 |
| 2 | R-002/R-006/R-010 全对齐拼接 | key 0；按 dtype binary 实例 | `allAlign` | 否则检查半对齐 |
| 3 | R-003/R-007/R-011 半对齐转置 | key 0；按 dtype binary 实例 | `!allAlign && !oneConcat && halfAlign` | 否则一般转置 |
| 4 | R-004/R-008/R-012 一般转置 | key 0；按 dtype binary 实例 | `!allAlign && !oneConcat && !halfAlign` | Tiling 参数非法则失败 |

```text
binary selector: (T1,T2) ∈ {(fp16,fp16),(bf16,bf16),(fp32,fp32),
                            (fp16,bf16),(bf16,fp16),(fp16,fp32),(bf16,fp32)}
  -> Tiling key = 0
  -> if oneConcat: 单行 Route
     else if allAlign: 全对齐 Route
     else if halfAlign: 半对齐转置 Route
     else: 一般转置 Route
  -> 每个 tile 内仍按 isAllZero、count==1 且 tile 列对齐的优先级走 fast path
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001..R-004 | 0 / same binary | `T1==T2`；四行布局谓词依次为 one/all/half/general | 每个归并节点单 AIV；入口可多层 | `ChunkCat<T1,T1,false>`；Copy/Concat/两次 transpose | `inUb≈1/2`、`outUb≈1/2`；MTE2↔V↔MTE3 event | 行列核/tile tail；零域与单输入 fast path |
| R-005..R-008 | 0 / cross-16 binary | `(fp16,bf16)` 或 `(bf16,fp16)`；同四个布局谓词 | 同上 | `ChunkCat<T1,T2,true>`；`DoCast` 经 FP32、`CAST_RINT` | `inUb≈1/3`、`outUb≈2/3`；dstBuf 的 FP32/T2 view 原地复用 | 同上 |
| R-009..R-012 | 0 / widen binary | `(fp16,fp32)` 或 `(bf16,fp32)`；同四个布局谓词 | 同上 | `ChunkCat<T1,float,true>`；直接 `CAST_NONE` 到 FP32 | `inUb≈1/3`、`outUb≈2/3` | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 32B 输入块 | `E=32/sizeof(T1)` | DMA block、padding、splitColAlign 的共同元素粒度 | 全部 | 半/一般非对齐路线 | `ChunkCatTiling::CalculateOutputInfo` |
| 单行 UB 预留 | `oneConcat` 时 `colLimit=inUb/s-32*(E-1)` | 最多 32 个输入段各自向上补齐造成局部偏移膨胀 | 单行 routes | colLimit 非正的保护在固定仓不可见 | `ChunkCatTiling::DoUbSplit` |
| 转置 repeat | `repeatTimes=totalUbColAlign*colRepeatNum/E` 存入 `uint8_t` | `TransDataTo5HD` repeat 参数位宽 | 半/一般 routes | Host 未显式检查溢出；依赖 tile 公式 | `ChunkCat::Trans1/Trans2` |
| DMA blockCount | 多行 split 时 `currentUbRowFactor` 窄化为 `uint16_t` | `DataCopyExtParams.blockCount` 位宽 | 全部 | Host 未显式检查窄化 | `ChunkCatCommon::DoRowsCopy` |
| 32 输入段阶段化 | `ubLoopInfo.count>31` | 本地 `inputCol[32]` 与 concat 参数表容量 | 全部 | 单行立即 Compute+CopyOut；其余先部分重排回 srcBuf | `ChunkCat::ComputeOver32` |
| all-zero tile | 当前 tile 与所有输入有效行均无交集 | 跳过 MTE2，使用已清零 srcBuf 形成零输出 | 多行 routes | 非零 tile 走正常 CopyIn | `ChunkCat::Compute` |

### POD/ABI

`ChunkCatTilingData` 以声明顺序被 Host 逐字段赋值并由 `REGISTER_TILING_DEFAULT/GET_TILING_DATA_WITH_STRUCT` 解析。固定仓没有 `static_assert(sizeof/offsetof)`，因此精确 padding、`sizeof`、`alignof` 和各字段 `offsetof` 标记为固定仓不可见；三 bool 后接 15 个 int64 字段的声明顺序不得改变。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ChunkCatTilingData` | `isAllAlign` | bool/实现 ABI | 声明序 0 | flag | `SetTilingData=isAllAlign_` | `ChunkCat::Init` | 所有 `C_i` 均按 E 对齐 |
| 同上 | `isHalfAlign` | bool/实现 ABI | 声明序 1 | flag | `=isHalfAlign_` | `ChunkCat::Init` | 所有 `C_i` 均按 E/2 对齐 |
| 同上 | `isOneConcat` | bool/实现 ABI | 声明序 2 | flag | `=isOneConcat_` | `ChunkCat::Init` | `outputRow==1&&!isAllAlign` |
| 同上 | `inputNum` | int64/64 | 声明序 3 | tensor 个数 | `=inputNum_` | CopyIn tensor 循环 | `>0` |
| 同上 | `dim` | int64/64 | 声明序 4 | 轴 | `=dim_` | `GetChunkInfo/GetShape` | 固定 0 |
| 同上 | `numChunk` | int64/64 | 声明序 5 | chunk 个数 | `=numChunk_` | `chunkCol=ceil(chunkDimSize/numChunk)` | `>0` |
| 同上 | `outputRow` | int64/64 | 声明序 6 | 行元素域 | `=outputRow_` | 分核/地址 | 等于 numChunk |
| 同上 | `outputCol` | int64/64 | 声明序 7 | 每行元素 | `=outputCol_` | dst stride/地址 | `sum C_i` |
| 同上 | `inUbSize` | int64/64 | 声明序 8 | 字节 | `=inUbSize_` | `InitBuffer(srcBuf_)` | 同 dtype 约半 UB，转换约 1/3 |
| 同上 | `outUbSize` | int64/64 | 声明序 9 | 字节 | `=outUbSize_` | `InitBuffer(dstBuf_)` | 与 inUb 和为 ubSize+256 |
| 同上 | `blockRowNum` | int64/64 | 声明序 10 | core group 数 | `=blockRowNum_` | blockIdx 二维映射 | `>=1` |
| 同上 | `blockColNum` | int64/64 | 声明序 11 | core group 数 | `=blockColNum_` | blockIdx 二维映射 | `>=1` |
| 同上 | `ubRowFactor` | int64/64 | 声明序 12 | 行/tile | `=ubRowFactor_` | Process rowLoop | tail 可更小 |
| 同上 | `ubColFactor` | int64/64 | 声明序 13 | 元素/tile 行 | `=ubColFactor_` | Process colLoop | E 对齐上界 |
| 同上 | `blockRowFactor` | int64/64 | 声明序 14 | 行/core | `=blockRowFactor_` | block row base | ubRowFactor 倍数 |
| 同上 | `blockColFactor` | int64/64 | 声明序 15 | 元素/core 行 | `=blockColFactor_` | block col base | ubColFactor 倍数 |
| 同上 | `tailBlockRowFactor` | int64/64 | 声明序 16 | 行 | `=tailBlockRowFactor_` | 最后 row group | 1..blockRowFactor |
| 同上 | `tailBlockColFactor` | int64/64 | 声明序 17 | 元素 | `=tailBlockColFactor_` | 最后 col group | 1..blockColFactor |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001..R-004 | key 0 + same-dtype binary | `chunk_cat_tiling.cpp::Tiling4ChunkCat` | `ChunkCatTilingData` 全字段 | `chunk_cat.cpp::chunk_cat→ChunkCat<T1,T1>` | `chunk_cat.h` + `chunk_cat_common.h` | OpDef + ascend910b/910_93 binary JSON + module CMake |
| R-005..R-008 | key 0 + 16-bit cross binary | 同上 | 同上；in/out UB 为 1/3:2/3 | `chunk_cat→ChunkCat<T1,T2,true>` | 同上；`DoCast` 两步 FP32 | 同上 |
| R-009..R-012 | key 0 + widen binary | 同上 | 同上 | `chunk_cat→ChunkCat<T1,float,true>` | 同上；`DoCast` 直接 widen | 同上 |

## Kernel 实现

### 公共执行流程

每个节点入口都声明 `KERNEL_TYPE_AIV_ONLY`，解析同一 POD 并创建 `TPipe`。`InitCommon` 将 blockIdx 映射为行/列 group，绑定当前核输出基址和 list-tensor 描述，按字节分配 `srcBuf` 与 `dstBuf`。`Process` 遍历本核二维 UB tile；每 tile 先把整个 srcBuf 清零，再扫描与当前列窗相交的输入，按有效行使用 `DataCopyPad` 搬入。每累计 32 个输入段就阶段化处理，最终按 route 做 concat/transpose/cast，等待 V→MTE3 后逐行搬出。所有 route 共用行/列 tail、短尾 chunk 的零填充以及 MTE3→MTE2 的跨 tile 复用协议。

### Route R-001～R-012：经典拼接、转置与转换实现族

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_chunk_cat.cpp` | `CheckDtypeValid/SplitToChunkCat` | 7 个合法 dtype pair；最多 512 输入一节点 | 建立一个或多层 ChunkCat 节点 |
| Tiling | `op_host/chunk_cat_tiling.cpp` | `CalculateOutputInfo/DoUbSplit` | `oneConcat→allAlign→halfAlign→general` 资源公式 | R-001..R-012 对应布局组 |
| Binary | `op_host/config/ascend910{b,_93}/chunk_cat_binary.json` | `op_list` | dtype pair 精确匹配 | same/cross16/widen 模板实例 |
| Router | `op_kernel/chunk_cat.cpp` | `chunk_cat` | `ORIG_DTYPE_X==ORIG_DTYPE_Y` 否则 `NEED_CAST=true` | `ChunkCat<DTYPE_X,DTYPE_Y[,true]>` |

#### Device 数据流

##### 函数级调用链

```text
l0op::ChunkCat -> ADD_TO_LAUNCHER_LIST_AICORE(ChunkCat)
  -> chunk_cat(x-list, y, system-workspace, tiling)
    -> ChunkCatCommon::InitCommon
    -> ChunkCat::Init
    -> ChunkCat::Process
      -> dupToZero
      -> CopyIn -> GetChunkInfo/SplitTensorDim0/CopyInChunk
        -> DoRowsCopy|DoLastRowsCopy -> ExecuteDataCopy(DataCopyPad)
      -> Compute
        -> ComputeOneConcat
         | UBRearrange4Concat
         | UBRearrange4Trans -> Trans1 -> UBRearrange4TransConcat -> Trans2
         | all-zero/single-segment fast path
        -> DoCast (NEED_CAST only)
      -> CopyOut(DataCopyPad)
  -> internal GM edge to next reduction layer, or ViewCopy to out
```

##### 源码等价伪码

```text
# 源码等价伪码；每个 block 负责二维矩形
rowGroup = blockIdx / blockColNum; colGroup = blockIdx % blockColNum
for tile in rowLoop * colLoop:
  rowStart = rowGroup*blockRowFactor + tileRow*ubRowFactor
  colStart = colGroup*blockColFactor + tileCol*ubColFactor
  currentRows/currentCols = normal factor or row/col tail
  srcBuf[:] = 0; totalCol=localOffset=count=0
  for each input i until currentCols covered:
    C_i = ceil(shape_i[0]/numChunk) * prod(shape_i[1:])
    if [totalCol,totalCol+C_i) intersects [colStart,colStart+currentCols):
      splitCol/startOffset = intersection
      copy valid rows GM[rowStart*C_i+startOffset] -> srcBuf[localOffset]
      # DataCopyPad preserves 32B local alignment; absent chunk rows remain zero
      remember signed splitCol and advance padded localOffset
    if count > 31: flush/partially concatenate and reuse srcBuf
  if oneConcat: compact each segment, optional cast
  elif no valid source: copy zeroed srcBuf, optional cast
  elif count==1 and currentCols%E==0: direct UB copy, optional cast
  elif allAlign: row-wise UB concat, optional cast
  else: rearrange -> transpose -> concat -> inverse transpose -> optional cast
  wait V->MTE3; DataCopyPad dstBuf rows -> GM output rectangle
  wait MTE3->MTE2 before next reuse
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input-list GM | `rowStart*tensorCol+startOffset` | 有效行×`splitCol` | T1/各输入二维 ND | caller/Contiguous | `ExecuteDataCopy` | caller/executor，运行期有效 | 每行用 byte stride；local 右补齐 |
| `srcBuf/srcLocal` | 各段 `localOffset` | `inUbSize` 字节 | T1/packed-or-transpose staging | Duplicate、MTE2 | concat/transpose/cast | TPipe；每 tile 复用 | `TBuf<VECCALC>`；reinterpret T2 view alias |
| `dstBuf/dstLocal` | 0 或 concat/transpose offset | `outUbSize` 字节 | T2；也 alias 为 T1/FP32 | V/UB copy/cast | MTE3 CopyOut | TPipe；MTE3 完成后复用 | `TBuf<VECCALC>`；三种 LocalTensor 共享物理区 |
| output GM | 核 base + `tileRow*ubRowFactor*outputCol+tileCol*ubColFactor` | `currentRows*currentCols` 有效元素 | T2/二维 ND | MTE3 | 下层节点或 ViewCopy | executor/caller | `dstStride=(outputCol-currentCols)*sizeof(T2)` |
| system workspace | 固定仓未见 Device offset | `GetLibApiWorkSpaceSize()`/节点 | opaque | platform/executor | SDK runtime；body 不读 | executor | 非数据边；具体复用固定仓不可见 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  src["srcBuf · VECCALC TBuf · inUbSize bytes"]:1
  dst["dstBuf · VECCALC TBuf · outUbSize bytes"]:2
```

##### 字节账与复用顺序

```text
same dtype: inUbSize=floor((ubSize+256)/2), outUbSize=ubSize+256-inUbSize
cast dtype: inUbSize=floor((ubSize+256)/3), outUbSize=ubSize+256-inUbSize
UB_total = inUbSize + outUbSize = ubSize + 256 bytes
dynamic_ub = N/A（AIV classic TBuf）
workspace_total/node = GetLibApiWorkSpaceSize(); Device body 0-byte semantic slice
=> classic Tiling 把接口预留的 256B 加回预算；该平台 API 的物理 UB 口径属于固定仓外部 SDK
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| srcBuf 全区 | `dupToZero`、MTE2 CopyIn | concat/transpose/DoCast | 下一 tile Duplicate/MTE2 | `V_MTE2` 与 tile 末 `MTE3_MTE2` | 不在 V 或 MTE3 仍读取时覆盖 |
| dstBuf 全区 | concat/transpose/DoCast | MTE3 CopyOut | 下一 tile Vector 写 | `V_MTE3`，CopyOut 后 `MTE3_MTE2` | output DMA 完成后才能复用 |
| dstBuf FP32/T2 alias | `Cast(...dstLocalFP32_)` | 第二次 Cast/DataCopy | 同次 DoCast 后续或下一 tile | Vector 程序序与 `PipeBarrier` | 同一物理区按源码顺序覆写 |
| srcBuf 阶段化区 | 首 32 段 MTE2 | `ComputeOver32` 部分 concat | 把 compact 结果写回 srcBuf 后继续追加 | `MTE2_V`、`V_MTE2` | `inputCol[0]` 重置为已 compact 总列 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 清零→MTE2 | `Set/Wait V_MTE2` | `Duplicate(srcBuf,0)` | CopyIn | srcBuf | 未搬入的尾 chunk 保持 0 |
| MTE2→V | `MTE2_V` | DataCopyPad | Compute | 有效输入段 | split/last row 由 blockLen/blockCount 限制 |
| V 内 | `PipeBarrier<PIPE_V>` | rearrange/transpose/cast | 下一 Vector 阶段 | src/dst alias | repeat/stride 取自 tile 与 E |
| V→MTE3 | `V_MTE3` | Compute | CopyOut | dstBuf | 只提交 currentRows×currentCols |
| MTE3→下一 tile | `MTE3_MTE2` | CopyOut | 下一 CopyIn | 两 TBuf 生命周期 | 防止 DMA 与复用冲突 |
| 16-bit cross | FP32 `CAST_NONE` 后 `CAST_RINT` | T1→FP32 | FP32→T2 | dstBuf alias | FP16↔BF16 经 FP32 舍入 |
| widen | `Cast(...CAST_NONE)` | FP16/BF16 | FP32 out | dstBuf | 无降精度阶段 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/chunk_cat_tiling.cpp` | `GetInputInfo/CalculateOutputInfo/DoUbSplit/DoBlockSplit/SetTilingData` | 完整 shape、alignment、UB、block、key 0 与 POD producer |
| Kernel entry/router | `op_kernel/chunk_cat.cpp` | `chunk_cat` | AIV-only；按平台和 `ORIG_DTYPE` 选择 classic 同 dtype/cast 实例 |
| Device body/helper | `op_kernel/chunk_cat_common.h` | `InitCommon/GetChunkInfo/CopyInChunk/ExecuteDataCopy` | list descriptor、二维 core 地址与 tail DMA |
| Device body/helper | `op_kernel/chunk_cat.h` | `Process/Compute/Trans1/Trans2/DoCast` | 四布局族、阶段化、同步和精度顺序 |

## 算子特定优化

### Trick T-001：32 路阶段化压缩避免输入数扩大局部参数表

#### 触发与源码依据

- 适用 Route：R-001～R-012
- 触发条件：单 tile 已收集的相交输入段 `ubLoopInfo.count>31`
- Golden 来源：[`conversion/chunk_cat/op_kernel/chunk_cat.h:180-220`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat.h#L180-L220)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (isOneConcat_ && ubLoopInfo.count > NUM_THIRTY_ONE) {
    Compute(ubLoopInfo);
    CopyOut(ubLoopInfo);
    localOffset = 0;
    ubLoopInfo.preCatCol += ubLoopInfo.totalUbCol;
    ubLoopInfo.count = 0;
    ubLoopInfo.totalUbCol = 0;
    ubLoopInfo.totalUbColAlign = 0;
}
else if (ubLoopInfo.count > NUM_THIRTY_ONE) {
    // 提前做部分concat
```

#### 机制与设计原因

本地只保留 `inputCol[32]`。单行 route 直接计算并把已收集列提交到 GM，以 `preCatCol` 推进；多行 route 则先在 UB 中把 32 段压成一段，写回 srcBuf，并把参数表重置为一个已拼接段后继续扫描。这样输入列表可以远大于 32，而无需按总输入数扩大 UB 元数据或重复扫描已处理段。

#### 迁移边界

- 适用：大量小 tensor 在同一 tile 相交的拼接/重排。
- 前提：阶段结果能表示成后续流程接受的一段，且 MTE2/V/MTE3 ordering 与 `preCatCol` 一致。
- 失效：若中间结果不能保持目标顺序/对齐，必须扩大参数表或使用独立中间节点。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时不得省略 count/total/aligned-total/preCatCol 的同步重置。

### Trick T-002：非对齐拼接用两次块转置把跨段 padding 移出有效列

#### 触发与源码依据

- 适用 Route：R-003/R-004/R-007/R-008/R-011/R-012
- 触发条件：`!isAllAlign && !isOneConcat`
- Golden 来源：[`conversion/chunk_cat/op_kernel/chunk_cat.h:414-433`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat.h#L414-L433)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
UBRearrange4Trans(ubLoopInfo, this->srcLocal_, dstLocalT1_);
Trans1(ubLoopInfo, dstLocalT1_, this->srcLocal_);
UBRearrange4TransConcat(ubLoopInfo, this->srcLocal_, dstLocalT1_);
Trans2(ubLoopInfo, dstLocalT1_, this->srcLocal_);
```

#### 机制与设计原因

每段的 DMA 右 padding 会打断逐行拼接。实现先按 16 行块重排，再以 `TransDataTo5HD` 交换行/块维，使每段有效列成为可连续复制区；拼接后第二次转置恢复 row-major。半对齐 route 用 `colRepeatNum=2`，一般 route 用 `E`，因此 Host 的 rowLimit 和 Device repeat/stride 必须成对迁移。

#### 迁移边界

- 适用：多行、段宽非 32B 对齐且需去除段间 padding 的拼接。
- 前提：tile 行数、`uint8_t repeatTimes`、TBuf 容量和 E/2/E 对齐满足源码上界。
- 失效：全对齐改用直接 concat；单行改用 R-001/R-005/R-009 的逐段提交。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；只做一次转置会留下转置布局，忽略 signed `inputCol` 会把 split 段的 padding 当有效数据。

## 模板与实现对比

| 对比维度 | R-001/005/009 单行 | R-002/006/010 全对齐 | R-003/007/011 半对齐 | R-004/008/012 一般 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | `numChunk=1&&!allAlign` | `allAlign` | `!all&&!one&&halfAlign` | 其余 | Tiling flag 决策 |
| 模板与 dtype | same/cross16/widen 三组 | 同左 | 同左 | 同左 | binary pair 决定 NEED_CAST/DoCast |
| 循环与分核 | 列优先，逐最多 32 段提交 | 列优先，可扩多行 | rowLimit=32 | rowLimit=16E | 避免段间 padding |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 两 TBuf，无 queue/dynamic UB/window | 同左 | 同左 | 同左 | classic AIV |
| 内存总占用与复用 | same 1:1，cast 1:2 | 同左 | 同左 | 同左 | src/dst 分区按 dtype 是否相同 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+DataCopy/Cast | 加 UBRearrange | 加双 TransDataTo5HD | 同左但 colRepeat=E | 无 SIMT/NDDMA/通信 |
| 同步、精度与 tail | MTE2/V/MTE3；逐段 tail | 行列 tail | transpose repeat tail | transpose repeat tail | 所有输出只写有效矩形 |
| fallback/失败边界 | colLimit/窄化未显式检查 | 资源非法失败 | repeat/stride 依赖 Host 上界 | 同左 | 无另一个 key fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持（ascend910b） | 支持（ascend910_93） | 支持，见独立 `_950` 卡 |
| Host/Launcher | 与 A3 共用 classic Tiling，`reserveUb=256` | 与 A2 共用 | `isRegBase=true`、reserve=0，入口图相同 |
| Kernel/资源 | `__CCE_AICORE__!=310`，classic Vector/transpose | 同 A2 | `__CCE_AICORE__==310`，MicroAPI per-tensor scatter/direct |

## 限制与失败边界

- 公开只支持 `dim=0`、`numChunks>0`、非空项 rank 1～8、同输入 dtype；输入 FP32 不能输出 16-bit。API 对 `dim!=0` 仅打日志，真正失败由 infer/Tiling 保证。
- 第一段在 `CheckParams` 前调用 `tensors->Size()`；因此源码实现与公开文档对 null `tensors` 返回 `ACLNN_ERR_PARAM_NULLPTR` 的承诺存在可见差异，固定仓未展示框架是否在进入函数前拦截。
- 空 tensor-list 路径在写 `workspaceSize` 前返回成功；`out`、`workspaceSize` 的值和是否写输出固定仓不可见，不应依赖该路径生成数据。
- 非空 list 中的 empty tensor 会被过滤；若全部项都 empty，`SplitToChunkCat` 也直接成功返回且不建立 ViewCopy，空输出是否需要实际写入由固定仓外框架语义决定。
- `MergeLastDims`、shape 乘积、`outputCol` 使用 int64 且没有显式溢出检查；DMA 的 `blockCount/blockLen/stride` 又窄化到 uint16/uint32，超界去向固定仓未闭合为显式错误。
- A2/A3 `UB_total=ubSize+256` 依赖外部平台 API 的 256B 预留口径；SDK primitive、list-tensor descriptor ABI、executor workspace offset/释放和运行时 cleanup 在固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/chunk_cat/docs/aclnnChunkCat.md` | 产品表、原型、参数表 | 公开性、A2/A3 支持与接口语义 |
| 入口声明/约束 | `conversion/chunk_cat/op_api/aclnn_chunk_cat.h` | 两段式声明 | 实际符号 |
| 入口实现 | `conversion/chunk_cat/op_api/aclnn_chunk_cat.cpp` | `CheckParams/SplitToChunkCat/aclnnChunkCat*` | 校验、512 路归并、executor 图 |
| 入口实现 | `conversion/chunk_cat/op_api/chunk_cat.cpp` | `l0op::ChunkCat` | infer 与 AICore launcher 节点 |
| Host/Tiling/Launcher | `conversion/chunk_cat/op_host/chunk_cat_infershape.cpp` | `InferShape4ChunkCat` | 二维输出 shape |
| Host/Tiling/Launcher | `conversion/chunk_cat/op_host/chunk_cat_tiling.cpp` | `ChunkCatTiling/Tiling4ChunkCat` | route、POD、UB、blockDim、workspace |
| Host/Tiling/Launcher | `conversion/chunk_cat/op_kernel/chunk_cat_tiling_data.h` | `ChunkCatTilingData` | Host/Device ABI 声明 |
| Kernel entry/body | `conversion/chunk_cat/op_kernel/chunk_cat.cpp` | `chunk_cat` | AIV-only entry 与模板选择 |
| Kernel entry/body | `conversion/chunk_cat/op_kernel/chunk_cat_common.h` | `ChunkCatCommon` | 公共 GM/UB、DMA、tail |
| Kernel entry/body | `conversion/chunk_cat/op_kernel/chunk_cat.h` | `ChunkCat` | classic route body、同步、cast、transpose |
| 构建接线 | `conversion/chunk_cat/op_host/chunk_cat_def.cpp` | `ChunkCat::ChunkCat` | A2/A3/950 config 与 dtype pair |
| 构建接线 | `conversion/chunk_cat/op_host/config/ascend910b/chunk_cat_binary.json` | `op_list` | A2 七个 binary 实例 |
| 构建接线 | `conversion/chunk_cat/op_host/config/ascend910_93/chunk_cat_binary.json` | `op_list` | A3 七个 binary 实例 |
| 构建接线 | `conversion/chunk_cat/CMakeLists.txt` | `add_all_modules_sources` | API/Host 模块收集 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 单 key、18 字段同一 POD | 字段顺序、bool ABI、字节/元素单位不得混用 | `ChunkCatTilingData/SetTilingData` |
| 调用顺序 | 每 512 路一层递归归并 | 下一层必须等待对应上一层 GM 输出 | `SplitToChunkCat` |
| workspace/内存 | 两个 TBuf 按 same/cast 比例切分 | 256B 口径、alias 生命周期、32 路阶段化必须一致 | `GetInputInfo/InitCommon/ComputeOver32` |
| 同步 | V→MTE2、MTE2→V、V→MTE3、MTE3→MTE2 | 不可在 DMA/Vector consumer 完成前覆盖 TBuf | `dupToZero/Compute/CopyOut` |
| 精度与尾块 | cross16 经 FP32+RINT，widen 直接 Cast；短 chunk 零填充 | cast 顺序、splitColAlign、有效 CopyOut 矩形不可改变 | `DoCast/DoLastRowsCopy` |
| 失败边界 | dim/dtype/shape/Tiling/launcher 失败均停止 | 不得把 key 0 无 fallback 误写成自动降级 | `CheckParams/Tiling4ChunkCat/chunk_cat` |