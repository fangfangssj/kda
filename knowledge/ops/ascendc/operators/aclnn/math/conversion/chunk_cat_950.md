---
type: Operator
title: "ChunkCat 950 ACLNN 算子知识"
description: "说明 aclnnChunkCat 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
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
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_host/chunk_cat_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_host/config/ascend950/chunk_cat_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T23:49:48Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ChunkCat 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 最新且唯一公开版本 `aclnnChunkCatGetWorkspaceSize` / `aclnnChunkCat`，两段式 ACLNN |
| 计算语义 | 每个输入沿 `dim=0` 切为 `numChunks` 个上界等宽块，逐块横向拼接，短尾块补零，并转换到 `out` dtype |
| 输入输出 | 同 dtype FP16/BF16/FP32 输入列表，内部归一成二维 ND；FP16/BF16 可输出 FP16/BF16/FP32，FP32 只输出 FP32 |
| 执行类型 | 全部算子自有节点为 `AIV_VECTOR`；每节点在 `__VEC_SCOPE__` 内使用 MicroAPI，入口可形成单 Kernel 或多层 512 路归并图 |
| Kernel 路线 | R-001 同 dtype；R-002 FP16↔BF16；R-003 FP16/BF16→FP32；每条 route 内按段宽/行数选择 scatter 或顺序 VF 子路径 |
| 核心资源 | 输入/输出 GM、`srcBuf`/`dstBuf` 两个 VECCALC TBuf、RegTensor/MaskReg/UnalignReg、MTE2/V/MTE3 event；Device 不读取 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→一层/多层 L0 launcher→950 Tiling/POD→AIV entry→Arch35 MicroAPI body→ascend950 OpDef/binary config 闭合；SDK primitive 与 executor 壳为明确框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对第 `i` 个输入，令 `s_i=shape_i[0]`、`c_i=prod(shape_i[1:])`、`q_i=ceil(s_i/numChunks)`。内部结果为 `[numChunks,sum_i(q_i*c_i)]`；第 `r` 行按输入顺序拼接 `x_i[r*q_i:min((r+1)q_i,s_i),:]`，并把每段不足 `q_i*c_i` 的尾部写零。Arch35 body 在每个 tensor 段搬入后直接把有效值/零值写到目标 UB 行位置，可在写入时完成 dtype 转换；最终 ViewCopy 到调用者 `out`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `tensors` | tensor-list，每项 view rank 1～8；公开 ND，API 支持非连续并先 Contiguous | 列表内同 dtype FP16/BF16/FP32 | 非空项必须非 null、非 private format；L0 每节点最多消费 512 项 |
| `dim` | scalar | int64 | 仅 0；API 对非零只打日志，InferShape/Tiling 返回失败 |
| `numChunks` | scalar | int64 | Tiling 要求 `>0`；决定输出行数和每输入块高 |
| `out` | 内部为二维 ND，外部 shape/format 必须匹配 | FP16/BF16/FP32 | FP32 输入只允许 FP32 输出；其余 6 种组合有 950 binary |
| `workspaceSize/executor` | 两段式输出 | uint64 / opaque | executor 保存连续化、多层节点、中间 GM tensor、ViewCopy 和 stream 执行图 |

### 入口与版本

固定提交的公开文档和头文件只声明无版本后缀入口，因此它是 950 最新公开版本。文档明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，对应 binary config 列出 7 个 dtype pair，Host 以 `IsRegbaseSocVersion(context_)` 选择 RegBase UB 公式，Kernel 编译宏 `__CCE_AICORE__==310` 选择 `ChunkCatArch35`。三层证据一致，无平台发布冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段创建 executor 后，对非空列表依次检查 null、dtype、format、rank；逐项 `Contiguous`，再把 `dim=0` 后的维度乘到末维。`SplitToChunkCat` 每 512 项构造一个 `l0op::ChunkCat`；一层结果数大于 1 时继续以同一 `numChunks` 归并，直至剩一个。最后 `CheckShapeAndScalarSame`、PostProcess format View 和 `l0op::ViewCopy`。任一连续化、InferShape、AICore launcher 入队、shape 或 ViewCopy 失败即停止。

### 参数、Workspace 与生命周期

每个 L0 节点分配内部 ND 输出并调用 `ADD_TO_LAUNCHER_LIST_AICORE`。Tiling 把平台 `GetLibApiWorkSpaceSize()` 写入唯一 workspace size；entry 虽接收 `workspace`，Arch35 body 不读取它。连续化张量和每层中间输出由 executor 持有，作为下一层 GM 输入；用户 workspace、输入、输出和 executor 至少覆盖异步 stream 执行期。多节点 workspace 的 offset/alias 与 executor 精确销毁时点属于固定仓外框架边界。

### 启动图与失败传播

若非空输入数 `N<=512`，只有一个自有 AIV 节点；否则第 `k` 层有 `ceil(N_k/512)` 个节点，下一层把这些 `[numChunks,cols]` 中间 tensor 再按 `dim=0` 切成单行 chunk 并横向合并，直到一个输出。节点 entry 都是 `chunk_cat`，blockDim 都来自各自 Tiling。executor 在同一 stream/依赖图中保证 producer→consumer 可见性；固定仓未见算子自有 stream event。producer 构图或运行失败必须阻断 consumer，具体 cleanup 由 `CommonOpExecutorRun` 管理。

| 边 | offset/extent/dtype/layout | ordering | owner/lifetime | 失败传播 |
|---|---|---|---|---|
| 连续输入→首层 | 每输入全部有效元素，二维 ND View | executor/同 stream | executor 或 caller | 前处理失败停止构图 |
| 层间 GM | producer 完整二维输出，dtype 为 out dtype；consumer 读取整 tensor | producer→consumer 依赖 | executor 中间 tensor，consumer 完成前有效 | producer 失败阻断下一层 |
| 末层→out | 完整内部结果，format View 后 ViewCopy | 末层→ViewCopy | executor View + caller out | shape/ViewCopy 失败不执行完整图 |

## Route 与参数设计

### 分核与资源计算

Host 公共推导与 A2/A3 相同：`E=32/sizeof(T1)`，每输入 `C_i=ceil(shape_i[0]/numChunk)*prod(shape_i[1:])`，`outputRow=numChunk`，`outputCol=sum C_i`。950 的 `isRegBase=true`，所以 `reserveUb=0`；同 dtype `inUb=ubSize/2`，转换 dtype `inUb=ubSize/3`，`outUb=ubSize-inUb`。RegBase `DoUbSplit` 总是先取 `colLimit=inUb/sizeof(T1)-32E`，令 `ubColFactor` 为不超过 colLimit 的 E 对齐均分值；只有单列 tile 时 `rowLimit=colLimit/ubColFactor`，否则行 factor 为 1。

block/core 计算为 `blockRowFactor=align_up(ceil(outputRow/coreNum),ubRowFactor)`、`blockRowNum=ceil(outputRow/blockRowFactor)`、`leftCore=coreNum/blockRowNum`、`blockColFactor=align_up(ceil(outputCol/leftCore),ubColFactor)`，最终 `blockDim=blockRowNum*blockColNum`。最后行/列 group 使用 POD tail factor。Device 对每个相交 tensor 进一步选择 `scatter = splitCol<=GetVRegSize()/sizeof(T2)/2 && currentUbRowFactor>=2`；否则走顺序 VF。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 同 dtype | key 0；fp16→fp16、bf16→bf16、fp32→fp32 binary | `T1==T2` | 检查 16-bit cross |
| 2 | R-002 16-bit 互转 | key 0；fp16→bf16 或 bf16→fp16 binary | `sizeof(T1)==sizeof(T2)` 且 `T1!=T2` | 检查 widen |
| 3 | R-003 widen | key 0；fp16→fp32 或 bf16→fp32 binary | `sizeof(T1)<sizeof(T2)` | 非配置 pair 失败 |

```text
(T1,T2) binary -> key 0 -> __CCE_AICORE__==310 -> ChunkCatArch35<T1,T2>
for each UB tile and each intersecting input segment:
  if splitCol > VRegBytes/sizeof(T2)/2 OR currentRows < 2:
    CopyCatVF<false>  # sequential unaligned load/store
  else:
    CopyCatVF<true>   # generated scatter indices
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 同 dtype | 0 + same binary | 950，三个同 dtype pair | 每归并节点单 AIV，入口可多层 | `ChunkCatArch35<T,T>::CopyCatVF`；MicroAPI copy/scatter | 两 TBuf 各约半 UB；RegTensor/Mask/Unalign；MTE2/V/MTE3 | scatter guard 失败转 sequential；行/列/VReg tail |
| R-002 16-bit cross | 0 + cross binary | 950，FP16↔BF16 | 同上 | 同模板；`castTraithalf2bf/castTraitbf2half` | 1/3:2/3 TBuf；RINT、ZEROING | 同上 |
| R-003 widen | 0 + widen binary | 950，FP16/BF16→FP32 | 同上 | 同模板；sequential 双 layout Cast+Interleave，scatter 先 Interleave 再 Cast | 1/3:2/3 TBuf；多个 FP32 RegTensor | FP32 tail 最多拆两次 store |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| RegBase 列预留 | `colLimit=inUb/s-32E` | 最多 32 段的 local 32B padding/阶段提交 | 全部 | 非正 colLimit 未显式拦截 | `ChunkCatTiling::DoUbSplit` |
| scatter 上界 | `splitCol<=V/sizeof(T2)/2 && rows>=2` | index RegTensor 需容纳至少两行，`rowNum=V/sizeof(T2)/splitCol` 不得为 0 | 全部 | `CopyCatVF<false>` | `ChunkCatArch35::Compute` |
| index dtype | T2 为 FP16/BF16 用 `uint16_t`，否则 `uint32_t` | scatter byte/element index 位宽与 FP32 lane 匹配 | 全部 | 编译期选择，无 runtime fallback | `CopyCatVF` |
| 32 段提交 | `count>31` | 控制每批输出段数量与 localOffset | 全部 | 先 CopyOut，再从零 localOffset 继续 | `CopyInAndCompute` |
| rowNum/tail | `rowNum=V/sizeof(T2)/srcLen` 仅在 scatter guard 内 | 避免除后为 0并控制 mask | scatter | sequential | `CopyCatVF` |
| DMA 位宽 | blockCount uint16、blockLen/stride uint32 | DataCopyExtParams ABI | 全部 | Host 未显式检查窄化 | `ExecuteDataCopy/CopyOut` |

### POD/ABI

Host 和 Device 共享 `ChunkCatTilingData` 声明。固定仓没有 `sizeof/alignof/offsetof` 断言；精确 padding 固定仓不可见，声明序和单位如下，三 bool 后的所有 int64 顺序不得变化。Arch35 不读取三个 alignment flag，但它们仍属于 ABI。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ChunkCatTilingData` | `isAllAlign` | bool/实现 ABI | 声明序 0 | flag | `=isAllAlign_` | Arch35 inactive | 仍序列化 |
| 同上 | `isHalfAlign` | bool/实现 ABI | 声明序 1 | flag | `=isHalfAlign_` | Arch35 inactive | 仍序列化 |
| 同上 | `isOneConcat` | bool/实现 ABI | 声明序 2 | flag | `=isOneConcat_` | Arch35 inactive | 仍序列化 |
| 同上 | `inputNum` | int64/64 | 声明序 3 | tensor 数 | `=inputNum_` | input loop | `>0` |
| 同上 | `dim` | int64/64 | 声明序 4 | axis | `=dim_` | `GetChunkInfo` | 固定 0 |
| 同上 | `numChunk` | int64/64 | 声明序 5 | chunks | `=numChunk_` | chunkCol | `>0` |
| 同上 | `outputRow` | int64/64 | 声明序 6 | 行 | `=outputRow_` | 地址/分核 | numChunk |
| 同上 | `outputCol` | int64/64 | 声明序 7 | 元素/行 | `=outputCol_` | row stride | `sum C_i` |
| 同上 | `inUbSize` | int64/64 | 声明序 8 | 字节 | `=inUbSize_` | `InitBuffer(srcBuf_)` | same 1/2，cast 1/3 |
| 同上 | `outUbSize` | int64/64 | 声明序 9 | 字节 | `=outUbSize_` | `InitBuffer(dstBuf_)` | 与 inUb 和为 ubSize |
| 同上 | `blockRowNum` | int64/64 | 声明序 10 | group 数 | `=blockRowNum_` | block mapping | `>=1` |
| 同上 | `blockColNum` | int64/64 | 声明序 11 | group 数 | `=blockColNum_` | block mapping | `>=1` |
| 同上 | `ubRowFactor` | int64/64 | 声明序 12 | 行/tile | `=ubRowFactor_` | Process | tail 可小 |
| 同上 | `ubColFactor` | int64/64 | 声明序 13 | 元素/行/tile | `=ubColFactor_` | Process/rowStride | E 对齐上界 |
| 同上 | `blockRowFactor` | int64/64 | 声明序 14 | 行/core | `=blockRowFactor_` | row base | ubRow 倍数 |
| 同上 | `blockColFactor` | int64/64 | 声明序 15 | 元素/core 行 | `=blockColFactor_` | col base | ubCol 倍数 |
| 同上 | `tailBlockRowFactor` | int64/64 | 声明序 16 | 行 | `=tailBlockRowFactor_` | last row group | 有效 tail |
| 同上 | `tailBlockColFactor` | int64/64 | 声明序 17 | 元素 | `=tailBlockColFactor_` | last col group | 有效 tail |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 0 + same binary | `Tiling4ChunkCat` | `ChunkCatTilingData` | `chunk_cat→ChunkCatArch35<T,T>` | `chunk_cat_common.h` + `chunk_cat_arch35.h` copy/scatter | ascend950 OpDef/binary JSON/CMake |
| R-002 | key 0 + cross16 binary | 同上 | 同上，UB 1/3:2/3 | `chunk_cat→ChunkCatArch35<half,bfloat16_t>` 或反向 | 同上 + RINT CastTraits | 同上 |
| R-003 | key 0 + widen binary | 同上 | 同上，UB 1/3:2/3 | `chunk_cat→ChunkCatArch35<T,float>` | 同上 + Interleave/Cast | 同上 |

## Kernel 实现

### 公共执行流程

`chunk_cat` 声明 AIV-only，解析 POD 后在 950 构造 `ChunkCatArch35<T1,T2>`。`InitCommon` 绑定 list descriptor、输出核矩形和两个 TBuf。`Process` 遍历本核 UB tile，`CopyInAndCompute` 对每个与列窗相交的输入计算 chunk/有效行、用 MTE2 搬入可用部分，并立即在 `__VEC_SCOPE__` 把该段写入 dstBuf 的最终行 stride；短 chunk 的三段分别是有效整行 `rowLoop0`、部分有效且需列尾补零的 `rowLoop1`、纯零 `rowLoop2`。超过 31 段时先 CopyOut，推进 `preCatCol` 后复用局部 offset。tile 结束再提交剩余段。

### Route R-001：同 dtype MicroAPI 搬运

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Binary | `op_host/config/ascend950/chunk_cat_binary.json` | same dtype entries | fp16/fp16、bf16/bf16、fp32/fp32 | R-001 |
| Tiling | `chunk_cat_tiling.cpp` | `isRegBase/GetInputInfo` | same dtype→inUb=ub/2 | POD/key 0 |
| Router | `chunk_cat.cpp` | `__CCE_AICORE__==310` | 平台宏 | `ChunkCatArch35<T,T>` |
| Device | `chunk_cat_arch35.h` | `Compute` | scatter guard | scatter 或 sequential 子路径 |

#### Device 数据流

##### 函数级调用链

```text
l0op::ChunkCat -> chunk_cat -> ChunkCatArch35<T,T>::Init/Process
  -> CopyInAndCompute -> ChunkCatCommon::CopyInChunk(DataCopyPad)
  -> Compute -> CopyCatVF<true|false>
    -> DoScatterCatVF/DoScatterPadCatVF
     | DoCopyCatVF/DoPadCatVF
  -> CopyOut(DataCopyPad) -> GM layer edge/ViewCopy
```

##### 源码等价伪码

```text
# 源码等价伪码
for tile; for intersecting tensor segment:
  compute valid/full/partial/zero row groups
  if any valid source: wait MTE2->V
  if splitCol<=V/sizeof(T)/2 and rows>=2:
    index = lane%splitCol + floor(lane/splitCol)*alignedDstRowStride
    scatter source lanes to final dst rows; scatter zero for missing rows
  else:
    sequential unaligned load T and unaligned store T per row; write zero RegTensor for padding
  wait V->MTE3
after 32 segments or tile end: strided DataCopyPad dstBuf -> output GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `rowStart*tensorCol+startOffset` | valid rows×splitCol | T/2D ND | caller/previous layer | MTE2 | caller/executor | DataCopyPad local 32B alignment |
| srcBuf | `localOffset` per segment | `inUbSize=ubSize/2` bytes | T/packed segments | MTE2 | MicroAPI load | TPipe/tile | segment extent align-up to E |
| dstBuf | `dstOffset=totalCol+startOffset-colStart-preCatCol` | `outUbSize=ubSize-inUbSize` bytes | T/rowStride=`align(currentCols,32/sizeof(T))` | MicroAPI | MTE3 | TPipe/tile/batch | final positions, no second reorder |
| RegTensor/index | register-local | `GetVRegSize()` lane domain | T + uint16/uint32 index | MicroAPI load/arange | store/scatter | V scope | fixed仓外部 API 决定确切寄存器字节 |
| output GM | tile base + preCatCol | currentRows×totalUbCol | T/2D ND | MTE3 | next layer/ViewCopy | executor/caller | srcStride skips dstBuf unused columns |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  src["srcBuf · inUbSize=ubSize/2"]:2
  dst["dstBuf · outUbSize=ubSize-inUbSize"]:2
```

##### 字节账与复用顺序

```text
UB_total = inUbSize + outUbSize = ubSize
register_bytes = 固定仓不可见；VReg lane 上界由 GetVRegSize() 提供
dynamic_ub = N/A（AIV TBuf）
workspace_total/node = GetLibApiWorkSpaceSize(); Device semantic slice = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| srcBuf segment | MTE2 | MicroAPI load | 后续 tensor/localOffset reset | `MTE2_V` | valid copy 完成后读取 |
| dstBuf batch | MicroAPI store/scatter | MTE3 CopyOut | 32 段提交后或下一 tile | `V_MTE3`、`MTE3_MTE2/V` | CopyOut 完成后复用 |
| RegTensor | MicroAPI load/copy | immediate store | 下一 V loop | V-scope program order/LocalMemBar | widen 之外无双 store |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| MTE2→V | `MTE2_V` | valid DataCopyPad | Compute | src segment | zero-only 段不等待 MTE2 |
| V store | `LocalMemBar<VEC_STORE,VEC_STORE>` | partial copy | zero-pad store | overlapping partial row | 保证有效值先于尾零覆盖序列 |
| V→MTE3 | `V_MTE3` | CopyCatVF | CopyOut | dstBuf | mask/unaligned post 限制 tail |
| MTE3→复用 | `MTE3_MTE2` + `MTE3_V` | CopyOut | 下一 batch | src/dst buffers | 防提前覆盖 |
| 数值 | compile-time same type copy | source | destination | value lanes | 无算术转换；pad 精确为 0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/chunk_cat_tiling.cpp` | `DoUbSplit/SetTilingData` | RegBase UB、block、key/POD |
| Kernel entry/router | `op_kernel/chunk_cat.cpp` | `chunk_cat/ChunkCatArch35<T,T>` | 950 AIV-only 路由 |
| Device body/helper | `op_kernel/chunk_cat_arch35.h` | `CopyInAndCompute/CopyCatVF/DoCopyCatVF/DoScatterCatVF` | per-segment direct final-layout write |

### Route R-002：FP16/BF16 互转

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Binary | `op_host/config/ascend950/chunk_cat_binary.json` | cross16 entries | fp16→bf16 或 bf16→fp16 | R-002 |
| Tiling | `GetInputInfo` | `inputDataType!=outputDataType` | inUb=ub/3 | POD/key 0 |
| Device | `DoCopyCatVF/DoScatterCatVF` | `is_same_v<T1,half>...` | compile-time pair | RINT CastTrait |

#### Device 数据流

##### 函数级调用链

```text
chunk_cat -> ChunkCatArch35<T1,T2>::Process
  -> CopyInAndCompute -> CopyInChunk
  -> CopyCatVF<scatter|sequential>
    -> MicroAPI::Cast<T2,T1,castTraithalf2bf|castTraitbf2half>
    -> unaligned store or DataCopyScatter
  -> CopyOut
```

##### 源码等价伪码

```text
# 与 R-001 地址/循环相同
srcReg = load valid T1 lanes
dstReg = Cast<T2,T1, {NO_SAT/UNKNOWN, ZEROING, CAST_RINT}>(srcReg, mask)
store/scatter dstReg to aligned final row positions
generate T2 zero RegTensor for partial/absent chunk lanes
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM/srcBuf | 同 R-001 | `inUbSize=floor(ubSize/3)` | T1 | MTE2 | Cast | TPipe/tile | E1 对齐 |
| dstBuf | final row offset | `outUbSize=ubSize-inUbSize` | T2 | Cast/store | MTE3 | TPipe/tile | E2 row stride |
| src/dst RegTensor | VReg lanes | GetVRegSize domain | T1→T2 | load/Cast | store/scatter | V scope | ZEROING mask，RINT |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  src["srcBuf · floor(ubSize/3)"]:1
  dst["dstBuf · ubSize-inUbSize"]:2
```

##### 字节账与复用顺序

```text
UB_total = floor(ubSize/3) + (ubSize-floor(ubSize/3)) = ubSize
register_bytes = 固定仓不可见
dynamic_ub = N/A
workspace_total/node = GetLibApiWorkSpaceSize(); body 不读
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| srcBuf | MTE2 | Cast load | 下一段/tile | MTE2_V | 不提前覆盖 |
| dstBuf | Cast store/scatter | MTE3 | 下一 batch | V_MTE3/MTE3_V | DMA 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load→cast | MTE2_V | MTE2 | V Cast | srcBuf | mask 覆盖 main/tail |
| cast→store | V program order | Cast | scatter/unaligned store | RegTensor | `CAST_RINT`、ZEROING，bf16→half 为 NO_SAT |
| store→copyout | V_MTE3 | V store | MTE3 | dstBuf | 只搬有效 totalUbCol |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/chunk_cat_tiling.cpp` | `GetInputInfo` | dtype 不同触发 1/3:2/3 UB |
| Kernel entry/router | `op_kernel/chunk_cat.cpp` | `ChunkCatArch35<DTYPE_X,DTYPE_Y>` | 950 不使用 classic NEED_CAST 参数 |
| Device body/helper | `op_kernel/chunk_cat_arch35.h` | CastTraits/`DoCopyCatVF/DoScatterCatVF` | 两个方向的 RINT 转换 |

### Route R-003：FP16/BF16 扩展到 FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Binary | `ascend950/chunk_cat_binary.json` | fp16_fp32/bf16_fp32 | 输入 16-bit、输出 FP32 | R-003 |
| Tiling | `GetInputInfo` | dtype 不同 | inUb=ub/3 | POD/key 0 |
| Device | `DoCopyCatVF/DoScatterCatVF` | final compile-time else | layout Cast+Interleave | FP32 store/scatter |

#### Device 数据流

##### 函数级调用链

```text
chunk_cat -> ChunkCatArch35<T1,float>::Process
  -> CopyInAndCompute -> CopyCatVF<false>
    -> DoCopyCatVF: Cast ZERO/ONE -> Interleave -> two FP32 unaligned stores
  | CopyCatVF<true>
    -> DoScatterCatVF: Interleave source halves -> Cast ZERO -> FP32 scatter
  -> DoPadCatVF/DoScatterPadCatVF -> CopyOut
```

##### 源码等价伪码

```text
# sequential
src16 = unaligned_load(main16 lanes)
fp32a = Cast<float,T1,RegLayout::ZERO>(src16)
fp32b = Cast<float,T1,RegLayout::ONE>(src16)
Interleave(fp32a,fp32b); store fp32a; LocalMemBar; store fp32b
# scatter
Interleave(srcReg1,srcReg2,src16,src16)
fp32 = Cast<float,T1,RegLayout::ZERO>(srcReg1)
scatter fp32 by generated uint32 index
# tail uses tailFP32Fir=min(tail,V/4), tailFP32Sec=tail-tailFP32Fir
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM/srcBuf | 同 R-001 | floor(ubSize/3) bytes | FP16/BF16 | MTE2 | MicroAPI load | TPipe | 32B local pad |
| dstBuf | final row offset | ubSize-inUbSize bytes | FP32 | Cast/Interleave store | MTE3 | TPipe | rowStride align 8 elements |
| RegTensor pair | VReg | main16=V/2，mainFP32=V/4 | T1 + two FP32 layouts | Cast/Interleave | two stores/scatter | V scope | store 间 LocalMemBar |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  src["srcBuf · floor(ubSize/3) · 16-bit"]:1
  dst["dstBuf · ubSize-inUbSize · FP32"]:2
```

##### 字节账与复用顺序

```text
UB_total = floor(ubSize/3) + (ubSize-floor(ubSize/3)) = ubSize
register_bytes = 固定仓不可见；sequential 同时声明 src16 + 两个 FP32 RegTensor
dynamic_ub = N/A
workspace_total/node = GetLibApiWorkSpaceSize(); body 不读
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| srcBuf | MTE2 | src16 RegTensor load | 后续段 | MTE2_V | valid source 才读 |
| dstBuf FP32 row | first/second FP32 store | MTE3 | 下一 batch | LocalMemBar + V_MTE3 | 两半 store 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 两半 FP32 store | `LocalMemBar<VEC_STORE,VEC_STORE>` | first store | second store | 同行连续 FP32 区 | ZERO/ONE layout 共同还原全部 16-bit lanes |
| tail | `tailFP32Fir/Sec` | tail Cast | two stores | 末段 | 第二长度可为 0 |
| partial row | store valid→LocalMemBar→store zero | valid Copy | pad | 同一行 | `colLen0/colLen1/tailPad` 防无效 lane 写有效值 |
| batch提交 | V_MTE3/MTE3_V | V | MTE3/next V | dstBuf | 有效 `totalUbCol`，未提交对齐 padding |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/chunk_cat_tiling.cpp` | `GetInputInfo/DoUbSplit` | widen UB 与 tile 上界 |
| Kernel entry/router | `op_kernel/chunk_cat.cpp` | `ChunkCatArch35<T,float>` | 950 router |
| Device body/helper | `op_kernel/chunk_cat_arch35.h` | `DoCopyCatVF/DoScatterCatVF/CopyCatVF` | ZERO/ONE layout、Interleave、tail 与 scatter |

## 算子特定优化

### Trick T-001：按段直接散射到最终 UB 行布局，消除 classic 双转置

#### 触发与源码依据

- 适用 Route：R-001～R-003
- 触发条件：`splitCol<=GetVRegSize()/sizeof(T2)/2 && currentUbRowFactor>=2`
- Golden 来源：[`conversion/chunk_cat/op_kernel/chunk_cat_arch35.h:134-144`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat_arch35.h#L134-L144)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (tensorInfo.splitCol > GetVRegSize() / sizeof(T2) / NUM_TWO || ubLoopInfo.currentUbRowFactor < HALF) {
    CopyCatVF<false>(ubLoopInfo, tensorInfo, totalCol, localOffset);
} else {
    CopyCatVF<true>(ubLoopInfo, tensorInfo, totalCol, localOffset);
}
```

#### 机制与设计原因

小段且至少两行时，VF 用 `Arange/Div/Muls/Sub/Add` 生成 `lane→rowStride` scatter index，把每个输入段直接放到 dstBuf 的最终 row-major 位置，同时对缺失行 scatter 零；这避免了 classic body 为去掉段间 padding而做的重排、转置、拼接、逆转置。段太宽或只有一行时改用连续 unaligned copy，避免 `rowNum` 太小和索引生成开销。

#### 迁移边界

- 适用：多行小列段拼接，目标 UB 行 stride 可由寄存器索引表达。
- 前提：`rowNum=V/sizeof(T2)/splitCol>=2`，index dtype/Mask 与目标 dtype 一致，dstBuf 容量覆盖 aligned row stride。
- 失效：宽段或单行必须走 sequential VF，不能强行 scatter。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若在 guard 外使用 scatter，`rowNum` 可能不足以覆盖两行，tail/index mask 关系不再成立。

### Trick T-002：FP16/BF16→FP32 用 ZERO/ONE layout 与 Interleave 完整展开寄存器

#### 触发与源码依据

- 适用 Route：R-003 sequential 子路径
- 触发条件：输入 16-bit、输出 FP32，且 scatter guard 不成立
- Golden 来源：[`conversion/chunk_cat/op_kernel/chunk_cat_arch35.h:194-200`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/conversion/chunk_cat/op_kernel/chunk_cat_arch35.h#L194-L200)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
AscendC::MicroAPI::Cast<T2, T1, castTraitZero>(dstReg0, srcReg0, mask);
AscendC::MicroAPI::Cast<T2, T1, castTraitOne>(dstReg1, srcReg0, mask);
AscendC::MicroAPI::Interleave(dstReg0, dstReg1, dstReg0, dstReg1);
AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg0, uReg, mainFP32);
AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_STORE>();
AscendC::MicroAPI::DataCopyUnAlign(curDstAddr, dstReg1, uReg, mainFP32);
```

#### 机制与设计原因

一个 VReg 的 16-bit 输入扩成 FP32 后需要两个寄存器容量。ZERO/ONE layout 分别取偶/奇布局，再 Interleave 成连续输出，分两次 unaligned store；tail 也拆为不超过 `V/sizeof(float)` 的两段。该顺序在扩宽同时保留原 lane 次序，并避免在 UB 中额外物化中间 FP32 tensor。

#### 迁移边界

- 适用：16-bit→32-bit 的顺序搬运式转换。
- 前提：MicroAPI 支持相同 CastTrait/layout，两个 store 之间保留 VEC_STORE barrier，tail 两段长度准确。
- 失效：同 dtype或 16-bit 互转使用各自单寄存器路径；scatter 路径有不同 Interleave/Cast 顺序。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；省略 ONE layout 或第二次 store 会丢失一半输入 lane。

## 模板与实现对比

| 对比维度 | R-001 同 dtype | R-002 16-bit cross | R-003 widen | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 三个 same binary | 两个 cross binary | 两个 widen binary | binary config 精确 pair |
| 模板与 dtype | `Arch35<T,T>` | `Arch35<half,bf16>`/反向 | `Arch35<T,float>` | entry 同模板，不用 NEED_CAST bool |
| 循环与分核 | 相同二维 core/tile/per-tensor loop | 同左 | 同左 | dtype 只改变 UB 比例与 VF |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2 TBuf + copy regs | 2 TBuf + Cast regs | 2 TBuf + 双 FP32 regs | 无 queue/dynamic UB/通信 |
| 内存总占用与复用 | 1/2:1/2 | 1/3:2/3 | 1/3:2/3 | 输出扩展/转换预留更大 dstBuf |
| 搬运/Vector/SIMT/NDDMA/通信 API | unaligned copy/scatter | 加 RINT Cast | ZERO/ONE Cast+Interleave | 都是 AIV MicroAPI |
| 同步、精度与 tail | mask/unaligned tail | ZEROING+RINT | 双 store 与两段 FP32 tail | 保留有效 lane，pad 零 |
| fallback/失败边界 | scatter→sequential | 同左 | 同左 | 无另一个 Tiling key |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持（950PR/950DT） |
| Host/Launcher | classic reserve=256；入口图相同 | 同 A2 | RegBase reserve=0，始终用 RegBase UB split |
| Kernel/资源 | classic DataCopy/transpose/DoCast | 同 A2 | Arch35 MicroAPI per-segment direct/scatter，UB 总和=ubSize |

## 限制与失败边界

- 公开仅支持 `dim=0`、`numChunks>0`、rank 1～8、输入同 dtype；FP32 输入不得输出 16-bit。固定 binary 之外没有 fallback。
- API 在 null 检查前访问 `tensors->Size()`；公开 null 错误语义与该实现顺序存在差异。空列表又在写 `workspaceSize` 前成功返回，固定仓未定义此路径的输出数据。
- 非空 list 中的 empty tensor 全部被过滤；若过滤后为空，第一段不创建 ChunkCat/ViewCopy，空结果的写入语义属于固定仓外框架边界。
- shape/列数乘法没有显式 int64 溢出检查；DMA 参数会窄化到 uint16/uint32。`colLimit=inUb/s-32E`、core/tile 除法也缺少所有极端零/负上界检查，非法极端 shape 的统一失败终点固定仓不可见。
- Arch35 每逢 `count>31` 提交后重置 `localOffset/totalUbCol` 但不重置 `count`，所以此后每新增一段都会提交；功能顺序由 `preCatCol` 保持，精确性能影响未验证。若最后一次提交恰好清零 `totalUbCol`，Process 仍执行一次零长度 CopyOut 调用，其 SDK 行为属于固定仓外部边界。
- `GetVRegSize`、MicroAPI primitive、list tensor descriptor ABI、executor workspace offsets/cleanup 是固定仓不可见的 SDK/框架边界；POD 精确 `sizeof/offsetof` 亦无静态断言。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `conversion/chunk_cat/docs/aclnnChunkCat.md` | 产品表、原型、约束 | 950 公开支持、API 语义 |
| 入口声明/约束 | `conversion/chunk_cat/op_api/aclnn_chunk_cat.h` | 两段式声明 | 实际入口 |
| 入口实现 | `conversion/chunk_cat/op_api/aclnn_chunk_cat.cpp` | `SplitToChunkCat/aclnnChunkCat*` | 校验、归一化、多层图、ViewCopy |
| 入口实现 | `conversion/chunk_cat/op_api/chunk_cat.cpp` | `l0op::ChunkCat` | infer/launcher 入队 |
| Host/Tiling/Launcher | `conversion/chunk_cat/op_host/chunk_cat_infershape.cpp` | `InferShape4ChunkCat` | 输出 shape |
| Host/Tiling/Launcher | `conversion/chunk_cat/op_host/chunk_cat_tiling.cpp` | `ChunkCatTiling/Tiling4ChunkCat` | RegBase UB、core、POD、workspace |
| Host/Tiling/Launcher | `conversion/chunk_cat/op_kernel/chunk_cat_tiling_data.h` | `ChunkCatTilingData` | POD ABI |
| Kernel entry/body | `conversion/chunk_cat/op_kernel/chunk_cat.cpp` | `chunk_cat` | AIV-only/950 路由 |
| Kernel entry/body | `conversion/chunk_cat/op_kernel/chunk_cat_common.h` | `ChunkCatCommon` | 公共 GM/UB、DMA、chunk tail |
| Kernel entry/body | `conversion/chunk_cat/op_kernel/chunk_cat_arch35.h` | `ChunkCatArch35` | MicroAPI scatter/direct/cast/pad |
| 构建接线 | `conversion/chunk_cat/op_host/chunk_cat_def.cpp` | `ChunkCat::ChunkCat` | ascend950 config/dtype |
| 构建接线 | `conversion/chunk_cat/op_host/config/ascend950/chunk_cat_binary.json` | `op_list` | 七个 950 binary |
| 构建接线 | `conversion/chunk_cat/CMakeLists.txt` | `add_all_modules_sources` | API/Host 模块收集 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 单 key、公共 18 字段 POD | 字段顺序、inactive bool、字节/元素单位 | `ChunkCatTilingData/SetTilingData` |
| 调用顺序 | 512 路分层归并 | 下一层必须消费完整上一层 GM 结果 | `SplitToChunkCat` |
| workspace/内存 | 两 TBuf 总和严格为 ubSize；Device 不读 workspace | dtype UB 比例、aligned rowStride、32 段提交 | `GetInputInfo/InitCommon/CopyInAndCompute` |
| 同步 | MTE2_V、V_MTE3、MTE3_MTE2/V、LocalMemBar | producer 完成前不得覆盖/store 重排 | `Compute/CopyOut/CopyCatVF` |
| 精度与尾块 | cross16 RINT；widen ZERO/ONE+Interleave；短 chunk 显式零 | CastTrait、mask、tailFP32Fir/Sec、partial-row 次序 | `DoCopyCatVF/DoScatterCatVF` |
| 失败边界 | 无 key fallback，scatter 仅内部回退 sequential | 不得把 SDK 外部行为推测为成功/错误 | `Compute/Tiling4ChunkCat` |