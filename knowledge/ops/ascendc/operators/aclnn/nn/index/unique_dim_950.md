---
type: Operator
title: "UniqueDim 950 ACLNN 算子知识"
description: "说明 aclnnUniqueDim 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_with_counts_ext2/op_host/op_api/aclnn_unique_dim.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_with_counts_ext2/docs/aclnnUniqueDim.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_with_counts_ext2/op_host/op_api/aclnn_unique_dim.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_with_counts_ext2/op_host/op_api/unique_with_counts_ext2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_with_counts_ext2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_host/op_tiling/unique_dim_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_host/op_tiling/unique_dim_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_host/unique_dim_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_host/unique_dim_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/unique_dim_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_kernel.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_constant.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_host/config/ascend950/unique_dim_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T17:12:38Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# UniqueDim 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnUniqueDimGetWorkspaceSize` + `aclnnUniqueDim`，公开两段式 ACLNN API；950 非空输入由 `l0op::UniqueWithCountsExt2` 选择 `UniqueDim` AICore。 |
| 计算语义 | 沿归一化轴 `k` 把每个切片视为长度 `rowLen=prod(shape)/shape[k]` 的行，按行字典序排序并去重；输出唯一行、原行到唯一行的逆映射和计数。 |
| 输入输出 | `self/valueOut`：ND，13 种公开 dtype；`inverseOut/countsOut`：INT64。输入 rank 1–8；`dim∈[-rank,rank)`。 |
| 执行类型 | 单 Host Kernel 节点；entry 是 `KERNEL_TYPE_MIX_AIV_1_0` 的 AIV carrier，实际主体由多次 `asc_vf_call` 启动 SIMT，另有 AIV 标量/shape UB 与 `DataCopyPad`；无 Cube/AIC、无 `MIX_CV`。 |
| Kernel 路线 | R-001：`returnInverse=true`；R-002：`returnInverse=false`。公开入口固定 `returnCounts=true`，两条均为 key 1。空输入在 ACLNN Host 返回，不到达 key 0。 |
| 核心资源 | GM 输入/输出；user workspace 的 uint32 索引、排序、flag、position、scan 区与可选 transpose 区；AIV `ubBuf_` 双半区及 `shapeBuf_`；SIMT 256/2048 线程；`SyncAll`/`PipeBarrier`/cache flush。 |
| 来源闭包 | source_closed：公开声明与校验、950 selector、Tiling/POD、entry、SIMT sort/scan/gather body、workspace 和构建接线均闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入 shape 为 `(d0,…,d{r-1})`，`k=dim<0?dim+r:dim`，`N=dk`，`rowLen=prod(di)/N`。实现先在 `k!=0` 时把 `[outer,N,inner]` 转为 `[N,outer,inner]`，其中 `outer=prod(i<k)di`、`inner=prod(i>k)di`；随后按每个长度为 `rowLen` 的切片作稳定字典序排序。相邻排序行不等则置 boundary flag，inclusive prefix 给出唯一组号，首行索引被 compact 后 gather 到 `valueOut`；`inverseOut[原索引]=组号`，`countsOut[g]=boundaryPos[g+1]-boundaryPos[g]`。`k!=0` 时再把唯一行转回原轴次序。

浮点排序中 BF16/DOUBLE 走位模式 total-order 映射，`-0` 归一为 `+0`；FP16/FP32 用标量 `<`/`>`。去重比较额外检测 NaN，使含 NaN 的行彼此不合并。公开入口固定请求 counts；`returnInverse=false` 时 inverse body 被编译期裁掉。`sorted` 传入 `UniqueDim` 属性，但 Tiling/Kernel 不读取它，Device 总执行排序去重；这仍满足 `sorted=false` 时不要求输出有序的接口边界。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | ND，rank 1–8，允许非连续 | UINT8/INT8/UINT16/INT16/UINT32/INT32/UINT64/INT64/DOUBLE/FLOAT/FLOAT16/BFLOAT16/BOOL | ACLNN 先 `Contiguous`；950 selector 对上述全表均走 AICore。 |
| `sorted` | 标量属性 | bool | 表示是否要求排序；实现始终排序。 |
| `returnInverse` | 标量属性 | bool | true 写 `[N]` inverse；false 不执行 scatter，shapeOut 把该输出形状编码为 `[1]`。 |
| `dim` | 标量属性 | INT64 | `[-rank,rank)`，Tiling 归一化为 `k`。 |
| `valueOut` | 与输入同 rank，仅第 `k` 维变为 `numOut` | 与 `self` 相同 | 去重行；ACLNN 写回 storage/original shape，运行时 shape 由 outshape buffer 返回。 |
| `inverseOut` | true 时 `[N]`；false 时运行时 shape `[1]` | INT64 | 原轴行号到唯一行号。 |
| `countsOut` | `[numOut]` | INT64 | 每个唯一行出现次数；公开入口固定启用。 |

### 入口与版本

公开文档列出 Ascend 950PR/950DT、A2、A3 支持；本卡只覆盖 950 的最新公开 `aclnnUniqueDim`。`aclnnUniqueDimGetWorkspaceSize` 建 executor、校验参数、对空输入直接返回、插入 `Contiguous`，再调用 `l0op::UniqueWithCountsExt2`。950 上 `IsRegbase()` 为真且 dtype 命中全量 `X_DTYPE_SUPPORT_LIST_ASCEND950`，因此选择 `UniqueDimAiCore`；它把公开的 `returnInverse` 透传、把 `return_counts` 固定为 true。第二段 `aclnnUniqueDim` 仅调用 `CommonOpExecutorRun`。

## Host/Launcher 执行流程

### 参数校验与前处理

按源码顺序：`OP_CHECK_COMM_INPUT` 检查 workspace/executor 出参；`CREATE_EXECUTOR`；`CheckParams` 依次检查四个 tensor 非空、self dtype 在平台表、valueOut dtype 等于 self、inverse/counts 均 INT64、rank 不超过 8、`dim` 在合法区间。`self->IsEmpty()` 时 workspace=0 并直接交出 executor，不添加 Contiguous 或 UniqueDim 节点。非空输入先 `l0op::Contiguous`；随后把 value/count 的 view shape 回写为 storage/original shape，再进入 selector。任一步失败立即返回对应 ACLNN 错误，不构造后续节点。

### 参数、Workspace 与生命周期

ACLNN executor 拥有 contiguous 临时量、27 个 INT64 的 `outShapeTensor` 和 launch 描述；第一段用 `GetWorkspaceSize()` 汇总框架与算子 workspace，第二段要求用户传入该地址/大小并在同一 stream 执行。Tiling 的 `currentWorkspace[0]=sysWorkspaceSize+workspaceSize`；entry 用 `GetUserWorkspace` 跳过系统区。

算子 user workspace 均按 128 B 对齐：`indices[N]`、`sortBuf[N]`、`flags[N]` 均为 uint32；counts 开启时有 `positions[N+1]`；`partialSum` 为 `coreNum*128 B`；`globalPrefix` 为 `(coreNum+1)*4 B`；shape 区为 `27*8 B`；`dim!=0` 再追加 `totalElements*sizeof(T)` 的 transpose 区。user workspace 从 executor 执行开始有效，Kernel 完成后才可释放；transpose 区先存重排输入，gather 后复用为回转后的 valueOut。

### 启动图与失败传播

```text
aclnnUniqueDimGetWorkspaceSize
  -> [框架边界] Contiguous(self)                     # 仅非连续时物化
  -> l0op::UniqueWithCountsExt2
     -> CheckSupport4UniqueDim
     -> UniqueDimAiCore / ADD_TO_LAUNCHER_LIST_AICORE(UniqueDim)
        -> K1 unique_dim(x,value,inverse,counts,shapeOut,userWorkspace,tiling)
aclnnUniqueDim(workspace,size,executor,stream)
  -> CommonOpExecutorRun：按 executor 顺序在同一 stream 执行
```

K1 的 `blockDim=usedCoreNum=aivCoreNum`，POD 为 `UniqueDimTilingData`，输入为 contiguous self，三个正式输出加框架 outshape buffer；user workspace 是唯一跨 SIMT phase 的 GM 交接介质。所有 phase 都在同一 Kernel entry 内，不是多 Host Kernel。`ADD_TO_LAUNCHER_LIST_AICORE` 失败阻断 executor 构造；Tiling 任一 getter 失败返回 `GRAPH_FAILED`；运行错误由第二段返回。Contiguous 的具体 framework workspace slice 与内部 Kernel 固定仓不可见。

## Route 与参数设计

### 分核与资源计算

非空时 `usedCoreNum=aivCoreNum`，`elementsPerCore=ceil(N/usedCoreNum)`，末核 `tail=N-(C-1)*elementsPerCore`，超出 N 的核由 Device clamp 为 0。固定 `tileSize=512`、`numTiles=ceil(N/512)`、普通 SIMT `blockDimX=256`；transpose、copy、gather 使用 `GM_BLOCK_DIM=2048`。每核排序区为 `[coreId*elementsPerCore, +coreLen)`；其他 flat 域使用 `ceil(total/C)`。Host 读取 UB 大小但不据此选 Route；Device 固定分配 `2*align(512*8,blockSize)+align(27*8,blockSize)`，可移植实现必须自行保证该和不超过 950 UB。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 inverse+counts | key 1；`UniqueDimKernel<T,true,true>` | 非空、950、dtype 在 13 项表且 `returnInverse=true`；公开 launcher 固定 counts=true | `returnInverse=false` 去 R-002；非 950 去 AICPU（不属于本卡） |
| 2 | R-002 counts-only | key 1；`UniqueDimKernel<T,false,true>` | 非空、950、dtype 在 13 项表且 `returnInverse=false`；公开 launcher 固定 counts=true | selector 不满足去 AICPU；未知 key 无 body |

```text
if self.empty: ACLNN success, no Device launch
else if IsRegbase && dtype in X_DTYPE_SUPPORT_LIST_ASCEND950:
    launch UniqueDim(return_counts=true)
    if returnInverse: R-001
    else: R-002
else:
    launch AICPU UniqueWithCountsExt2  # A2/A3/fallback，非本卡
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 inverse+counts | 1 / flags `1,1` | 950、非空、rank≤8、合法 dim、13 dtype、inverse=true | 单 K1，AIV carrier 内 SIMT sort→scan→inverse→compact/gather→counts | `UniqueDimKernel<DTYPE_X,true,true>::Process` | 全 workspace slice；phase 间 `SyncAll`；merge 另有 `PipeBarrier<PIPE_ALL>` | 核尾 clamp、tileLen、merge tail；不支持去 AICPU |
| R-002 counts-only | 1 / flags `0,1` | 同上但 inverse=false | 单 K1，跳过 inverse phase | `UniqueDimKernel<DTYPE_X,false,true>::Process` | inverse GM 不写；其余同 R-001 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 selector | `IsRegbase() && CheckType(dtype,13项表)` | 只有 ascend950 binary/config 接线 | R-001/R-002 | AICPU fallback | `CheckSupport4UniqueDim` |
| 非空 | ACLNN `!self->IsEmpty()` | 空输入 Host 直接完成，无 key0 launch | 两 Route | success/no launch | `aclnnUniqueDimGetWorkspaceSize` |
| rank/dim | `rank<=8 && -rank<=dim<rank` | POD 仅保存 8 维；Tiling 直接索引归一化 dim | 两 Route | ACLNN 参数错误 | `CheckShapeValid/CheckDimValue/GetShapeInfo` |
| uint32 索引 | `N` 被写入 uint32 workspace，但源码无 `N<=UINT32_MAX` 检查 | indices/flags/positions/prefix 均 uint32，窄化发生在 Device | 两 Route | 固定仓无 fallback，属于失败边界 | `ComputeWorkspaces/SimtInitIndices` |
| 固定 UB tile | `tileSize=512` | 双半区各 512 个 int64，另有 shape buffer | 两 Route | 无替代 Route | `DoBlockTiling/UniqueDimKernel::Init` |
| 128 B scan stride | `psStride=128/4=32` | 每核 partial sum 隔页，避免相邻核共享 cache line/page | 两 Route | 无替代 Route | `ComputeWorkspaces/Phase4_PrefixSum` |
| return flags | public `returnCounts=true`，inverse 运行时 bool | template 裁掉不需要的 scatter/count body | 两 Route | 四模板中 counts=false 对公开入口不可达 | `UniqueDimAiCore/DispatchByFlags` |

### POD/ABI

`UniqueDimTilingData` 的每个字段均由 `TILING_DATA_FIELD_DEF(int64_t,...)` 定义。宏展开后的 `sizeof/alignof/offsetof` 位于外部 tiling 框架，固定仓不可见；因此不把源码声明顺序推测为具体 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `UniqueDimTilingData` | `numInp` | int64/64 | 固定仓不可见 | 行 | `GetShapeInfo: shape[dim]` | `Init/Process` 总行数 | 非空 Route >0 |
| 同上 | `rowLen` | int64/64 | 同上 | 元素/行 | `totalElements/numInp` | 行比较、gather | 乘积溢出未检查 |
| 同上 | `returnInverse` | int64/64 | 同上 | bool(0/1) | `ReadAttrs` attr2 | `DispatchByFlags` | 决定 R-001/R-002 |
| 同上 | `returnCounts` | int64/64 | 同上 | bool(0/1) | `ReadAttrs` attr3；公开固定 true | `DispatchByFlags` | 公开恒 1 |
| 同上 | `usedCoreNum` | int64/64 | 同上 | core | `aivCoreNum` | 仅 ABI 保留；Device 取 `GetBlockNum` | 非空用全核 |
| 同上 | `elementsPerCore` | int64/64 | 同上 | 行/core | `ceil(N/C)` | `Init` coreStart/coreLen | 末核另字段 |
| 同上 | `tailCoreElements` | int64/64 | 同上 | 行 | `N-(C-1)*perCore` clamp 0 | final coreLen | 空核再 clamp |
| 同上 | `blockDimX` | int64/64 | 同上 | thread | 常量 256 | 多数 `asc_vf_call` | gather/transpose 改用 2048 常量 |
| 同上 | `inputDtypeSize` | int64/64 | 同上 | byte/element | `GetSizeByDataType` | Device 未读取 | workspace producer 专用镜像 |
| 同上 | `tileSize` | int64/64 | 同上 | 行 | 常量 512 | sort UB/loop | 固定 tile |
| 同上 | `numTiles` | int64/64 | 同上 | tile | `ceil(N/512)` | Device 未读取 | Host 诊断字段 |
| 同上 | `indicesOffset` | int64/64 | 同上 | byte | 0 | `indicesGm_` bind | uint32[N] |
| 同上 | `sortBufOffset` | int64/64 | 同上 | byte | `align(N*4,128)` | `sortBufGm_` bind | uint32[N] |
| 同上 | `flagsOffset` | int64/64 | 同上 | byte | `align(sortOff+N*4,128)` | `flagsGm_` bind | uint32[N] |
| 同上 | `positionsOffset` | int64/64 | 同上 | byte | counts? `align(flagsEnd,128)`:0 | `positionsGm_` bind | counts=false 时不可读；公开 counts=true |
| 同上 | `partialSumOffset` | int64/64 | 同上 | byte | `align(next,128)` | local/global scan | `C*128 B` |
| 同上 | `globalPrefixOffset` | int64/64 | 同上 | byte | `align(partial+C*128,128)` | prefix/gather | uint32[C+1] |
| 同上 | `shapeOutOffset` | int64/64 | 同上 | byte | `align(global+(C+1)*4,128)` | Device 未绑定此 user slice | 预留 216 B 后计算 transpose offset |
| 同上 | `workspaceSize` | int64/64 | 同上 | byte | `align` 后 user 总量 | Device 未读取 | executor workspace 使用 |
| 同上 | `dim` | int64/64 | 同上 | axis | 负轴归一化 | transpose、shapeOut | `[0,rank)` |
| 同上 | `outerSize` | int64/64 | 同上 | 元素乘积 | `prod(shape[0:k])` | transpose | 乘积溢出未检查 |
| 同上 | `innerSize` | int64/64 | 同上 | 元素乘积 | `rowLen/outer` | transpose | N>0 |
| 同上 | `transposeDstOffset` | int64/64 | 同上 | byte | `align(shapeOff+216,128)` | transpose input/back | dim=0 不分配但不读取 |
| 同上 | `inputDimNum` | int64/64 | 同上 | dim | rank | shapeOut | ≤8 |
| 同上 | `inputDim0` | int64/64 | 同上 | 元素 | shape[0] | shapeOut | rank<1 时默认 0；公开 rank≥1 |
| 同上 | `inputDim1` | int64/64 | 同上 | 元素 | shape[1] 或初始化 0 | shapeOut | rank≤1 时不读 |
| 同上 | `inputDim2` | int64/64 | 同上 | 元素 | shape[2] 或 0 | shapeOut | rank≤2 时不读 |
| 同上 | `inputDim3` | int64/64 | 同上 | 元素 | shape[3] 或 0 | shapeOut | rank≤3 时不读 |
| 同上 | `inputDim4` | int64/64 | 同上 | 元素 | shape[4] 或 0 | shapeOut | rank≤4 时不读 |
| 同上 | `inputDim5` | int64/64 | 同上 | 元素 | shape[5] 或 0 | shapeOut | rank≤5 时不读 |
| 同上 | `inputDim6` | int64/64 | 同上 | 元素 | shape[6] 或 0 | shapeOut | rank≤6 时不读 |
| 同上 | `inputDim7` | int64/64 | 同上 | 元素 | shape[7] 或 0 | shapeOut | rank≤7 时不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 inverse+counts | key1 / flags 1,1 | `unique_with_counts_ext2.cpp::UniqueDimAiCore`; `unique_dim_tiling_arch35.cpp::Tiling4UniqueDim` | `UniqueDimTilingData` | `unique_dim_apt.cpp::unique_dim/DispatchByFlags<T>` | `UniqueDimKernel<T,true,true>`；`unique_dim_simt*.h` | `unique_dim_def.cpp` + ascend950 binary/config |
| R-002 counts-only | key1 / flags 0,1 | 同上 | 同一 POD，inverse=0 | 同一 entry | `UniqueDimKernel<T,false,true>`；同一 helpers | 同上 |

## Kernel 实现

### 公共执行流程

两 Route 共用单 K1 launch。entry 解析 direct-struct POD、取得 user workspace，并按两个 flags 实例化模板。`Init` 绑定 GM，按 block 切行，分配 UB 双半区和 shape buffer。`Process` 顺序为：初始化索引；可选输入转置；每核 UB tile sort 与核内 GM merge；全核 merge-path；相邻差分；局部/全局 prefix；可选 inverse scatter；compact key + gather；可选回转；counts；末核写三路动态 shape。每个 SIMT VF 内用 `__syncthreads` 和前后 D-cache clean/invalidate；跨 AIV core 的 phase 依靠 `SyncAll`，merge pair 还加 `PipeBarrier<PIPE_ALL>`。

### Route R-001：inverse+counts

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `index/unique_with_counts_ext2/op_host/op_api/aclnn_unique_dim.cpp` | `aclnnUniqueDimGetWorkspaceSize` | 非空、参数合法 | 调用 l0op |
| Host | `.../unique_with_counts_ext2.cpp` | `CheckSupport4UniqueDim/UniqueDimAiCore` | `IsRegbase && dtype∈13项表`；attr `(dim,sorted,true,true)` | AICore `UniqueDim` |
| Tiling | `index/unique_dim/op_host/op_tiling/unique_dim_tiling_arch35.cpp` | `SetTilingData` | `N>0` | key1、blockDim=C、scheduleMode=1 |
| Router | `index/unique_dim/op_kernel/unique_dim_apt.cpp` | `DispatchByFlags<T>` | inverse && counts | `UniqueDimKernel<T,true,true>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(UniqueDim)
  -> unique_dim key1 -> DispatchByFlags<T>
    -> UniqueDimKernel<T,true,true>::Init/Process
      -> Phase0_InitIndices -> [Phase_transpose]
      -> Phase1_BlockSort -> Phase2_CrossCoreMerge
      -> Phase3_AdjacentDiff -> Phase4_PrefixSum
      -> Phase5_ScatterInverse
      -> Phase6_UniqueDetectAndGather -> [Phase_transpose_back]
      -> Phase7_ComputeCounts -> Phase8_OutputShape
```

##### 源码等价伪码

```text
# 伪码；C=blockNum, s=coreId*ceil(N/C), len=clamp(...)
indices[s + tid : s+len : 256] = original row id; SyncAll
if dim != 0: parallel_transpose_2048(input -> ws.transpose); SyncAll
for each local tile<=512:
    each of 256 threads creates 2 identity ids; compare-swap
    ping-pong merge in two UB halves; store indices
merge local GM tiles by doubling width
for width=ceil(N/C); width<N; width*=2:
    all C*256 threads merge each pair by merge-path, indices<->sortBuf
    copy unmatched tail; barrier+SyncAll
flags[i] = i==0 || row(indices[i-1]) != row(indices[i]); SyncAll
thread0/core inclusive-scans flags; core0 scans partial sums; add prefix; SyncAll
inverse[indices[i]] = flags[i]-1; SyncAll
for each group boundary: compactKey[group]=indices[i], positions[group]=i
positions[numOut]=N; gather compact rows with 2048 threads
if dim != 0: transpose value to ws.transpose, then flat-copy back; SyncAll
counts[g]=positions[g+1]-positions[g]
final core writes 27-word runtime shape via DataCopyPad
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM self | row `i`: `i*rowLen` after optional transpose | `N*rowLen` | T / row-major | ACLNN contiguous/transpose | compare/gather | executor/input lifetime | dim!=0 改指向 transpose |
| ws indices | 0 | `4N` B | uint32[N] | init/sort ping-pong | diff/inverse/compact | K1 whole life | 128 B next slice |
| ws sortBuf | `align(4N,128)` | `4N` B | uint32[N] | merge/compactKeys | merge/gather | K1 whole life | sort后复用为 compact keys |
| ws flags | `align(sortOff+4N,128)` | `4N` B | uint32[N] | adjacent diff/scan | inverse/compact | phase3–6 | 原地从 boundary 变 prefix |
| ws positions | `align(flagsOff+4N,128)` | `4(N+1)` B | uint32[N+1] | compact + sentinel | counts | phase6–7 | 仅 counts Route |
| ws partialSum | aligned next | `C*128` B | uint32，stride 32 | local scan | global scan | phase4 | 每核仅首 uint32 有效 |
| ws globalPrefix | aligned partial end | `4(C+1)` B | uint32[C+1] | global scan | add-prefix/numOut | phase4–8 | 最后元素复写/保持 numOut |
| ws transpose | `transposeDstOffset` | `totalElements*sizeof(T)` | T | transpose input/back | sort/gather/copyback | dim!=0 | 输入转置消费后复用 |
| UB `ubBuf_` | half0/half1 | `2*align(512*8,blockSize)` | int64[1024] | tile sort round | next merge round | 每核 K1 | ping-pong halves |
| UB `shapeBuf_` | 独立 TBuf | `align(27*8,blockSize)` | uint64[27] | final core scalar/vector | shapeOut | phase8 | 不与 sort alias |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  ubA["ubBuf half A · 512×8 B aligned"]:1
  ubB["ubBuf half B · 512×8 B aligned"]:1
  shape["shapeBuf · 27×8 B aligned"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*align(512*8, blockSize) + align(27*8, blockSize)
dynamic_ub = N/A（asc_vf_call 未传 dynamic UB，使用 ubBuf_ 的 local_mem 指针）
workspace_total = sysWorkspaceSize + (
  align 链：3*uint32[N] + uint32[N+1] + C*128 + uint32[C+1] + 216B
  + (dim!=0 ? T[totalElements] : 0))
=> Host 未用 ubSize 做 route guard；固定 tile 的 UB 上界由上述分配给出
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ub half A/B | sort/merge round | 下一 round | 后续 round 反向写 | `__syncthreads` | round 完成后切换读写 half |
| sortBuf | merge dst | 下一 merge src | phase6 compact key | cross-core `SyncAll` | 排序和 diff/inverse 已消费 indices |
| flags | adjacent diff | local scan | scan 原地改写 prefix | VF barrier+`SyncAll` | boundary 完成后才 scan |
| transpose | input transpose | sort/gather | output transpose back | phase 间 `SyncAll` | gather 完成后原输入转置不再需要 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT VF | `__syncthreads` + `dcci` | VF GM/local writes | 同 VF 后续读/返回 | cache/线程块 | grid-stride `<end` |
| phase 间 | `SyncAll` | 各 core GM 写 | 下一 phase | workspace/global output | 空核仍参与同步 |
| merge pair | `PipeBarrier<PIPE_ALL>` + `SyncAll` | merge-path dst | 下一 pair/round | indices/sortBuf | `mid<end`，未配对 tail 单独 copy |
| compare | dtype-specific scalar order | row loads | sort/dedup | T 值 | -0 合并；NaN 排序相等但去重不等 |
| shape write | V_S、S_MTE3 hard event | Duplicate/scalar store | DataCopyPad | shapeBuf | 216 B exact copy |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/unique_dim/op_host/op_tiling/unique_dim_tiling_arch35.cpp` | `DoBlockTiling/ComputeWorkspaces/SetTilingData` | 全核、512 tile、128 B workspace 链、key1 |
| Kernel entry/router | `index/unique_dim/op_kernel/unique_dim_apt.cpp` | `unique_dim/DispatchByFlags` | flags 1,1 到 `<T,true,true>` |
| Device body/helper | `index/unique_dim/op_kernel/arch35/unique_dim_kernel.h` | `UniqueDimKernel<T,true,true>::Process` | phase 顺序、workspace bind、同步、shape |
| Device body/helper | `index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h` | `SimtPhase1AllInOne/SimtMergePathImpl` | UB tile sort 和 GM merge-path |
| Device body/helper | `index/unique_dim/op_kernel/arch35/unique_dim_simt.h` | `SimtAdjacentDiff/SimtLocalScan/SimtGatherRows` | dedup、scan、gather、counts |

### Route R-002：counts-only

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `index/unique_with_counts_ext2/op_host/op_api/aclnn_unique_dim.cpp` | `aclnnUniqueDimGetWorkspaceSize` | 非空且 `returnInverse=false` | 调用 l0op |
| Host | `.../unique_with_counts_ext2.cpp` | `UniqueDimAiCore` | 950 selector；attr `(dim,sorted,false,true)` | AICore `UniqueDim` |
| Tiling | `.../unique_dim_tiling_arch35.cpp` | `SetTilingData` | N>0、inverse=0、counts=1 | key1 |
| Router | `index/unique_dim/op_kernel/unique_dim_apt.cpp` | `DispatchByFlags<T>` | `!returnInverse && returnCounts` | `<T,false,true>` |

#### Device 数据流

##### 函数级调用链

```text
unique_dim key1 -> DispatchByFlags<T>
  -> UniqueDimKernel<T,false,true>::Init/Process
    -> init/transpose/sort/cross-merge/diff/prefix
    -> [compile-time skip Phase5_ScatterInverse]
    -> compact+gather/transpose-back/counts/shapeOut
```

##### 源码等价伪码

```text
# 伪码；排序、merge、scan 与 R-001 完整相同
build/sort indices; boundary flags; inclusive prefix -> numOut
# if constexpr(RETURN_INVERSE) 为 false，不读写 inverseOut
compact boundary indices to sortBuf; positions[numOut]=N
gather unique rows; optional transpose back
counts[g]=positions[g+1]-positions[g]
shape inverse=[1], count=[numOut]; final core DataCopyPad
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices/sortBuf/flags | 与 R-001 相同 | 各 `4N` B | uint32 | sort/scan | compact/gather | K1 | 128 B slice |
| positions | 与 R-001 相同 | `4(N+1)` B | uint32 | compact/sentinel | counts | phase6–7 | counts 固定 true |
| inverseOut | 正式输出指针仍传入 | 0 个元素被写 | INT64 | N/A | N/A | executor | shapeOut 编码 `[1]` |
| transpose/UB/shape | 与 R-001 同公式 | 同 R-001 | T/int64/uint64 | 同 R-001 | 同 R-001 | K1 | 相同复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  ubA["ubBuf half A · 512×8 B aligned"]:1
  ubB["ubBuf half B · 512×8 B aligned"]:1
  shape["shapeBuf · 27×8 B aligned"]:2
```

##### 字节账与复用顺序

```text
UB_total = 2*align(512*8, blockSize) + align(27*8, blockSize)
dynamic_ub = N/A
workspace_total = 与 R-001 相同（counts=true）；inverse=false 不减少 workspace，只省 scatter 指令和 inverse GM 写
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| ub half A/B | tile merge | next merge | ping-pong round | `__syncthreads` | 同 R-001 |
| sortBuf | merge | merge/diff后 compact | compact key | `SyncAll` | 排序已完成 |
| transpose | input transpose | sort/gather | output transpose | `SyncAll` | dim!=0 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| sort/scan/gather | VF barrier+dcci、`SyncAll` | 同 R-001 | 同 R-001 | workspace | 同一 dtype order/tail |
| inverse branch | `if constexpr(false)` | N/A | 直接进入 phase6 | inverseOut | 无写入、无同步 |
| counts/shape | `SyncAll` + hard event | positions/counts | shape submit | count/output shape | count尾段 `cntStart<cntEnd` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/unique_with_counts_ext2/op_host/op_api/unique_with_counts_ext2.cpp` | `UniqueDimAiCore` | returnCounts 固定 true，inverse 透传 |
| Kernel entry/router | `index/unique_dim/op_kernel/unique_dim_apt.cpp` | `DispatchByFlags<T>` | flags 0,1 到 `<T,false,true>` |
| Device body/helper | `index/unique_dim/op_kernel/arch35/unique_dim_kernel.h` | `Process/Phase6/Phase7/Phase8` | 编译期跳过 inverse，其余完整执行 |

## 算子特定优化

### Trick T-001：单次 VF 内完成 UB tile sort 与核内 GM merge

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：非空 key1 且当前 core `coreLen>1`；每个 local tile 最多 512 行。
- Golden 来源：[`index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h:287-303`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h#L287-L303)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    // Step 1: Sort each tile cooperatively in UB
    for (int64_t t = 0; t < numLocalTiles; t++) {
        int64_t tileOff = t * tileSize;
        int64_t tileLen = (tileOff + tileSize <= coreLen) ? tileSize : (coreLen - tileOff);
        int64_t tileGlobalStart = coreStart + tileOff;
        SimtSortOneTile<T>(tid, threadNum, tileLen, tileGlobalStart,
                           indices, inputFlat, rowLen, ubBuf);
    }

    // Ensure all tiles written to GM before merge reads them
    __syncthreads();

    // Step 2: Multi-round merge from GM
    if (numLocalTiles > 1) {
        SimtMergeTilesGM<T>(tid, threadNum, coreStart, coreLen, tileSize,
                            indices, sortBuf, inputFlat, rowLen);
    }
```

#### 机制与设计原因

256 个 SIMT thread 各先处理 2 个 identity index，512 行 tile 在 `ubBuf_` 两个 512×int64 半区中 ping-pong 合并；tile 落到 GM 后，仍在同一次 `SimtPhase1AllInOne` VF 调用内做宽度翻倍的核内 merge。源码注释明确该组合避免跨 VF_CALL 的 GM cache 可见性问题；因此不是单纯封装，而是把同步域限定在一次 VF 的 `__syncthreads`/dcci 范围，同时保留跨 tile 的 GM 容量。`tileSize` 来自 POD 固定 512，`threadNum=blockDim.x=256`，双半区容量来自 `InitBuffer`。

#### 迁移边界

- 适用：需要对 GM 间接 key 做分块排序、单 tile 可放入固定 local buffer 的 950 SIMT 实现。
- 前提：单 VF 内可见的 local_mem 双缓冲、256 thread×2 item、GM ping-pong buffer，以及相同 cache flush/barrier 协议。
- 失效：tile 容量、线程模型或 GM cache 协议不同必须重做 UB 字节账和同步；本算子没有另一路 Host fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码注释给出的真实风险是把 tile sort 与 GM merge 拆成多个 VF_CALL，却不补齐跨调用的 GM cache 可见性；可能让 merge 读取旧数据。固定仓未提供独立的替代实现。

## 模板与实现对比

| 对比维度 | R-001 inverse+counts | R-002 counts-only | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | inverse=true | inverse=false | runtime flag router |
| 模板与 dtype | `<DTYPE_X,true,true>`，13 dtype binary | `<DTYPE_X,false,true>` | bool 模板裁枝 |
| 循环与分核 | 全核、512 tile、256/2048 threads | 相同 | 主排序不依赖 inverse |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 双半区 ubBuf + shapeBuf，无 Queue/通信 | 相同 | workspace/UB 固定 |
| 内存总占用与复用 | 写 inverseOut | 不写 inverseOut，workspace 不变 | counts 始终开 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT GM load/store、local_mem merge、shape DataCopyPad | 同上但无 scatter inverse | 单 Kernel Hybrid |
| 同步、精度与 tail | inverse 后多一次 SyncAll | 编译期跳过该 phase/同步 | 保持 phase 可见性 |
| fallback/失败边界 | selector 失败去 AICPU | 同上 | 950 AICore 无二级 Route |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；固定仓仅见 AICPU launcher，body 不可见 | 同左 | 公开支持且 AICore source_closed |
| Host/Launcher | `UniqueWithCountsExt2` AICPU | 同左 | `UniqueDim` AICore |
| Kernel/资源 | AICPU ABI/workspace/同步固定仓不可见 | 同左 | 本卡 SIMT + AIV shape、uint32 workspace |

## 限制与失败边界

- ACLNN 限制 rank≤8、合法 dim、dtype/output matching；空输入由 Host success/no-launch。Contiguous 的框架实现和具体 workspace slice固定仓不可见。
- 950 Tiling 对归一化 dim、shape 乘积、workspace offset 乘加没有显式溢出检查；内部索引、flag、position、prefix 均 uint32，源码也没有 `N<=UINT32_MAX` guard，不能推断超范围行为。
- 公开 API 与 950 Tiling/Kernel 接受任意合法轴，但注册的 `InferShape4UniqueDim` 源码仍在 `dim!=0` 时返回 `GRAPH_FAILED`；eager ACLNN launcher 已提供三个 `OP_OUTSHAPE`，固定仓外的 launcher 框架是否调用该图模式 InferShape 不可见。复用到图模式时必须保留这一冲突，不能直接宣称非零轴一定可用。
- `UniqueDimTilingData` 的宏展开 ABI（sizeof/alignof/offsetof）属于外部 tiling 框架，固定仓不可见。
- `shapeOutOffset` 对应的 user workspace 216 B 预留区未被 Kernel 绑定；实际 outshape 通过独立 `shapeOut` GM 参数写入。不得将预留区误当作输出 shape 交接。
- 公开 selector 的错误日志写“only bf16/fp16/fp32”，但实际 `X_DTYPE_SUPPORT_LIST_ASCEND950`、OpDef 和 ascend950 binary config 均列 13 dtype；本卡按实际谓词与构建接线记录。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/unique_with_counts_ext2/op_host/op_api/aclnn_unique_dim.h` | `aclnnUniqueDimGetWorkspaceSize/aclnnUniqueDim` | 公开原型与两段式入口 |
| 入口声明/约束 | `index/unique_with_counts_ext2/docs/aclnnUniqueDim.md` | 产品表、原型、约束 | A2/A3/950 公开性、dtype/rank/dim |
| 入口实现 | `index/unique_with_counts_ext2/op_host/op_api/aclnn_unique_dim.cpp` | `aclnnUniqueDimGetWorkspaceSize/aclnnUniqueDim` | 校验、empty、Contiguous、executor |
| Host/Launcher | `index/unique_with_counts_ext2/op_host/op_api/unique_with_counts_ext2.cpp` | `CheckSupport4UniqueDim/UniqueDimAiCore` | 950/AICPU selector、attrs、outshape |
| 构建接线 | `index/unique_with_counts_ext2/op_host/CMakeLists.txt` | `add_modules_sources` | ACLNN/l0op Host 源文件接线 |
| Host/Tiling/Launcher | `index/unique_dim/op_host/op_tiling/unique_dim_tiling_arch35.h` | `UniqueDimTilingData` | POD 字段与 helper 状态 |
| Host/Tiling/Launcher | `index/unique_dim/op_host/op_tiling/unique_dim_tiling_arch35.cpp` | `Tiling4UniqueDim` | shape/core/workspace/key producer |
| Host/Tiling/Launcher | `index/unique_dim/op_host/unique_dim_def.cpp` | `UniqueDim::UniqueDim` | ascend950 AICore 注册、13 dtype、opFile |
| Host/Tiling/Launcher | `index/unique_dim/op_host/unique_dim_infershape.cpp` | `InferShape4UniqueDim` | 动态 shape、dtype 与图模式 dim=0 guard |
| Kernel entry/body | `index/unique_dim/op_kernel/unique_dim_apt.cpp` | `unique_dim/DispatchByFlags` | entry、key、模板 router |
| Kernel entry/body | `index/unique_dim/op_kernel/arch35/unique_dim_kernel.h` | `UniqueDimKernel` | phase graph、资源、同步、shape |
| Kernel entry/body | `index/unique_dim/op_kernel/arch35/unique_dim_constant.h` | `SHAPE_LEN/MAGIC_GM_PAGE_SIZE` | shape ABI 索引与 scan stride 常量 |
| Kernel entry/body | `index/unique_dim/op_kernel/arch35/unique_dim_simt.h` | `Simt*` | compare/transpose/scan/gather/counts |
| Kernel entry/body | `index/unique_dim/op_kernel/arch35/unique_dim_simt_sort.h` | `SimtPhase1AllInOne/SimtMergePathImpl` | tile sort 与 merge |
| 构建接线 | `index/unique_dim/op_host/config/ascend950/unique_dim_binary.json` | `UniqueDim` op_list | 13 dtype ascend950 binaries |
| 构建接线 | `index/unique_dim/op_host/CMakeLists.txt` | `add_modules_sources/target_sources` | Host/Tiling 与 OpDef 编译目标 |
| 构建接线 | `scripts/kernel/binary_config/ascendc_config.json` | `UniqueDim` | compute_units ascend950、no dcache flush |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 32 个 int64 字段、offset/shape/core 参数 | 字段顺序/位宽/单位及 inactive 规则；具体 offsetof 不可推测 | `UniqueDimTilingData` |
| 调用顺序 | 单 K1 内 phase0→8 | merge/scan/gather 的顺序及 conditional inverse/counts | `UniqueDimKernel::Process` |
| workspace/内存 | uint32 中间量、128 B 对齐、transpose 复用 | slice offset/extent，`positions[N+1]`，partial stride 128 B | `ComputeWorkspaces/Init` |
| 同步 | VF barrier+dcci、跨核 SyncAll、merge PipeBarrier | 不能仅以源码书写顺序替代可见性 | `SimtPhase1AllInOne/Phase2_CrossCoreMerge` |
| 精度与尾块 | dtype compare、-0/NaN 规则、tile/core clamp | NaN dedup 不等、merge tail copy、grid-stride bounds | `SimtScalarCompare/SimtRowCompareDedup` |
| 失败边界 | 950 selector，不满足落 AICPU | AICPU body 不可拿本卡 950 实现代替；uint32 上界无 guard | `CheckSupport4UniqueDim` |