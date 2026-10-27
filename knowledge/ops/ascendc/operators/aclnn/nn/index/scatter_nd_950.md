---
type: Operator
title: "ScatterNd 950 ACLNN 算子知识"
description: "说明 aclnnScatterNd 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd/docs/aclnnScatterNd.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_api/aclnn_scatter_nd.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_api/scatter_nd_update.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_host/arch35/scatter_nd_update_tiling_regbase.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_host/arch35/scatter_nd_update_tiling_regbase.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_host/arch35/scatter_nd_update_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/scatter_nd_update_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_common.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_deterministic_simd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_deterministic_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simd.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simd_mask.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simt_sort.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_host/scatter_nd_update_def.cpp
    role: implementation
created_at: "2026-08-25T17:01:37Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ScatterNd 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnScatterNdGetWorkspaceSize` / `aclnnScatterNd`；真实公开实现位于 `scatter_nd_update` 闭合依赖。 |
| 计算语义 | 复制 `data` 后以 `updates` 覆盖 `indices` 指定切片；确定性上下文用全局最大原始行号保证重复索引取最后一条。 |
| 输入输出 | ND；`data/updates/out` 为 FP16/FP32/BOOL/BF16，`indices` 为 INT32/INT64；公开 shape 约束与 A2/A3 相同。 |
| 执行类型 | 单 AIV Kernel launch；body 可能为 AIV_VECTOR、SIMT，或 AIV 预处理+SIMT 内嵌线程函数；无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 SIMD mask；R-002 deterministic SIMT；R-003 deterministic SIMD；R-004 non-deterministic SIMD；R-005 SIMT dynamic-sort；R-006 plain SIMT。 |
| 核心资源 | GM/UB/TQue/TBuf/RegTensor；确定性路线使用 16MiB 保留区后的 `mask` 与 `varIdx` workspace；`SyncAll`、event、atomic max。 |
| 来源闭包 | source_closed：ACLNN→L0→arch35 Tiling/POD→key 100/200 router→六个可达 body/helper→ascend950 构建已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `K=indices.shape[-1]`、`N=prod(indices.shape[:-1])`、`S=prod(data.shape[K:])`。第 `r` 行索引的元素偏移为 `O[r]=sum(indices[r,d]*stride[d])`，其更新切片覆盖 `out[O[r]:O[r]+S]`。确定性路线先以 atomic max 计算每个目标行对应的最大原始 `r`，再只提交该行更新，因此重复索引选择最后一条。非确定性路线用于框架关闭 deterministic 时，重复目标的最终值依具体路线与核间写入顺序。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `data` | ND，rank 1~8，可非连续 | FP16/FP32/BOOL/BF16 | 先 Contiguous，作为 `ScatterNdUpdate` 的原地 `var`。 |
| `indices` | ND，rank 1~8，可非连续 | INT32/INT64 | `K<=rank(data)`，公开要求非负且不越界；arch35 核心要求实际 indices rank>=2、`K` 1~7。 |
| `updates` | `indices[:-1]+data[K:]`，ND | 同 `data` | Contiguous 后传给核心。 |
| `out` | 与 `data` 同 shape | 同 `data` | 核心更新临时 `self`，最后 ViewCopy 到 out。 |
| `workspaceSize/executor` | 标量/句柄 | uint64/opaque | 第一段产生，第二段在指定 stream 消费。 |

### 入口与版本

固定提交只有无版本后缀公开入口。文档明确支持 Ascend 950PR/950DT；`scatter_nd_update/CMakeLists.txt`、OpDef 与 ascend950 binary config 均接入 arch35。API 实现依次调用 Contiguous、`l0op::ScatterNdUpdate(...,false)`、ViewCopy。公共文档声明默认确定性；Host 的具体 selector 以 `TilingContext::GetDeterministic()` 为 producer，框架如何设置该值在固定仓不可见，但关闭 deterministic 后仍可达的四条路线也在本卡闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

API 按 null→dtype→shape 顺序校验；BF16 在 `DAV_2201` 或 RegBase(950) 接受。任一 `data/indices/updates` empty 时返回 workspace 0，不添加核心节点。非空时三路 Contiguous；因此本公开入口到核心的 `var/indices/updates` 均连续，arch35 支持 view 的字段对该入口 inactive。L0 `IsAiCoreSupport` 在 RegBase 对四种公开 dtype 全部返回 true，不落 AiCPU。

### 参数、Workspace 与生命周期

`ScatterNdUpdateTiling::GetPlatformInfo` 取得 AIV 核数，把平台 UB 减去 32KiB DCache/栈保留后设置 local memory size。核心 workspace 基线固定为 16MiB；确定性路线追加 `(varStorageInAxis+N+1)*sizeof(TYPE_T)`，其中 `TYPE_T` 由地址范围选择 uint32/uint64。Device `maskGm=workspace`，`varIdxGm=workspace+(varStorageInAxis+1)`（TYPE_T 元素偏移）。executor 拥有所有节点与 workspace，第二段统一在 stream 上运行。

### 启动图与失败传播

```text
Contiguous(data/indices/updates)
  -> ScatterNdUpdateAiCore -> scatter_nd_update_apt [单 AIV launch]
     -> router(key 100/200 + POD flags)
        -> one of R-001...R-006
  -> ViewCopy(updated temporary, out)
```

确定性 body 内部是单 launch 三阶段图：`InitMaskGm` → `SyncAll` → `CalcMask`（AIV offset/sort + SIMT atomic max）→ `SyncAll` → update commit。workspace owner 始终是 executor；两个 `SyncAll` 分别保证 mask 清零完成以及 atomic winner/varIdx 对提交阶段可见。API/L0/tiling/launch 任一返回失败即阻断后续 ViewCopy。

## Route 与参数设计

### 分核与资源计算

`N=indiceShapeSize/K`，`S=updateShapeSize/N`，`varInAxis=outputShapeSize/S`。地址 key：当 `indiceShapeSize<UINT32_MAX && updateShapeSize<UINT32_MAX && outputStorageShapeSize<INT32_MAX` 为 100，否则 200；key 200 且 output 仍小于 INT32_MAX 时 OFFSET_T 沿用 indices dtype，否则强制 INT64。确定性：`deterministic==1 && N>1`，再以 `S*sizeof(T)<128` 选 SIMT，否则 SIMD。非确定性且行字节>=128 选 SIMD；其中 mask 条件为 `varInAxis<0.1*UB && N>5*varInAxis && N>1000*usedCore && S>19`。窄行且 `N/varInAxis>=3` 选 SIMT-sort，否则 plain SIMT。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 SIMD mask | 100/200 + `isMask=1` | 非确定性、`S*sizeof(T)>=128` 且四项 mask 谓词全真 | R-004 |
| 2 | R-002 deterministic SIMT | 100/200 + deterministic flags | `GetDeterministic()==1 && N>1 && S*sizeof(T)<128` | R-003 |
| 3 | R-003 deterministic SIMD | 100/200 + deterministic flags | `GetDeterministic()==1 && N>1 && S*sizeof(T)>=128` | 非确定性树 |
| 4 | R-004 non-deterministic SIMD | 100/200 + `isSimdNonDeterminstic=1` | 非确定性且 `S*sizeof(T)>=128`，R-001 未命中 | R-005/R-006 |
| 5 | R-005 SIMT dynamic-sort | 100/200 + `isSimtWithSort=1` | 非确定性、`S*sizeof(T)<128`、`N/varInAxis>=3` | R-006 |
| 6 | R-006 plain SIMT | 100/200 fallback | 其余非空情况，包括 `N<=1` | unknown key 无 body |

```text
key = all address extents small ? 100 : 200
if isMask: R-001
else if deterministic && rowBytes<128: R-002
else if deterministic: R-003
else if rowBytes>=128: R-004
else if N/varInAxis>=3: R-005
else: R-006
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | mask flag | 950、非确定性、宽行、高重复 mask 谓词 | 单 AIV node | `ScatterNdUpdateSimdMask<T,U,OFFSET_T>` | 每核 UB mask；无 GM atomic | one-line/multi-line 子路 |
| R-002 | det+SIMT | 950、det、N>1、rowBytes<128 | AIV init/mask + SIMT commit | `DeterministicSimt<T,U,TYPE,OFFSET>` | workspace atomic max；2 次 SyncAll；1024 threads | update tile tail |
| R-003 | det+SIMD | 950、det、N>1、rowBytes>=128 | AIV mask + AIV DMA commit | `ScatterNdUpdateDeterministicSimd<T,U,TYPE,OFFSET_T>` | 同 workspace；double-buffer update queue | row/slice tail |
| R-004 | SIMD flag | 950、non-det、rowBytes>=128、mask false | AIV Vector/DMA | `ScatterNdUpdateSimd` | split-after 或 split-indices；可动态 sort | tail core/row/col |
| R-005 | SIMT sort flag | 950、non-det、窄行、高重复 | AIV offset/sort + SIMT store | `ScatterNdUpdateSimtSort` | UB radix sort；1024 threads | 每 UB row tile动态 sort/no-sort |
| R-006 | fallback | 950、其它非空 | SIMT | `ScatterNdUpdateSimt` | updates/indices double buffer；1024 threads | rank1 快路、通用 K 路 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| UB 保留 | `usableUB=platformUB-32KiB` | DCache/编译栈占用 | 全部 | Tiling fail 若平台 UB<=32KiB | `GetPlatformInfo` |
| 地址位宽 | 三个 extent 阈值 | router 的 TYPE/OFFSET 模板避免窄化 | 全部 | key200/int64 offset | `GetTilingKey/scatter_nd_update_apt` |
| deterministic row | `S*sizeof(T)<128` | 小行用 SIMT 线程铺元素，大行用 DMA tile | R-002/R-003 | 互转 | `CalcDeterministicCoreSplit` |
| mask | 四项完整谓词 | 每核 UB mask 可容纳目标行并值得去重 | R-001 | R-004 | `CalculateMask` |
| SIMT sort | `N/varInAxis>=3` | 高重复时 UB 排序/unique 减少重复 store | R-005 | R-006 | `CalculateDerivedParams` |
| dynamic sort | `maxScore>0.01` | tile 内重复统计决定 radix sort | R-004/R-005/确定性 mask | no-sort inner branch | `IndexStatistic*` consumers |
| UB double buffer | `(UB-256)/2` | indices/update/sort 子区需同时闭合 | R-001~R-004 | factors 递减/切列 | `DoOpTiling*` |
| empty | API 任一输入 empty | 无核心工作 | 全部 | workspace 0 | ACLNN impl |

### POD/ABI

`ScatterNdUpdateRegBaseTilingData` 是显式标准字段类，但固定仓未提供编译器 ABI 的 `sizeof/alignof/offsetof` 断言；下表保留声明顺序，数值 offset 标为固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ScatterNdUpdateRegBaseTilingData` | `blockNum` | uint64/64 | 固定仓不可见 | core/block | Block/Sort Tiling；PostTiling 可覆盖 launch dim | SIMT bodies | det/SIMD routes launch 用 totalCore |
| 同上 | `rankSize` | uint32/32 | 同上 | 维 | indices 最后一维 | 全部 offset 计算 | 1~7 |
| 同上 | `blockTilingSize` | uint64/64 | 同上 | 更新元素/索引行 | BlockTiling/SortTiling | SIMT bodies | route dependent |
| 同上 | `tailBlockTilingSize` | uint64/64 | 同上 | 元素/行 | Host 尾差 | SIMT tail core | 可为 0 |
| 同上 | `ubTilingSize` | uint32/32 | 同上 | 元素/行 | UbTiling/SortTiling | SIMT UB loop | 非空 >0 |
| 同上 | `sliceSize` | uint64/64 | 同上 | T 元素 | `S` | 全部 | empty 可 0 |
| 同上 | `outPutShape[8]` | uint64[8] | 同上 | 元素 | var origin shape | plain SIMT bounds | 仅 rank 内 active |
| 同上 | `strideList[8]` | uint64[8] | 同上 | T 元素 | contiguous 逆向积/实际 view stride | offset helpers | ACLNN 核心为连续 |
| 同上 | `outputStorageShapeSize` | uint64/64 | 同上 | T 元素 | storage size | bounds/key/router | >0 |
| 同上 | `varInAxis` | int64/64 | 同上 | 目标行 | output/S | mask/SIMT | >0 |
| 同上 | `varStorageInAxis` | int64/64 | 同上 | storage 行 | storage/S | deterministic workspace | ACLNN 等于 varInAxis |
| 同上 | `indexRankSize` | int64/64 | 同上 | 维 | K | SIMD bodies | 1~7 |
| 同上 | `afterAxis` | int64/64 | 同上 | T 元素/行 | S | route/loops | >0 |
| 同上 | `updateLoopSize` | int64/64 | 同上 | loop | ceilDiv after/factor | update loops | route dependent |
| 同上 | `updateTailNum` | int64/64 | 同上 | T 元素 | Host 尾差 | last update loop | 可等于 factor |
| 同上 | `indicesLoopSize` | int64/64 | 同上 | loop | ceilDiv N/factor | split-after | inactive 为 0 |
| 同上 | `indiceTailNum` | int64/64 | 同上 | 索引行 | Host 尾差 | split-after | 可为 0 |
| 同上 | `usedCoreNumBefore` | int64/64 | 同上 | core | indices/after 分核 | early return | <=AIV core |
| 同上 | `usedCoreNumAfter` | int64/64 | 同上 | core | after 分核 | 保留字段 | 当前 body 基本不读 |
| 同上 | `indicesFactor` | int64/64 | 同上 | 索引行/tile | UB 方程 | SIMD/mask | >=1 |
| 同上 | `afterAxisFactor` | int64/64 | 同上 | T 元素/tile | UB 方程 | update DMA | UB block 对齐 |
| 同上 | `eachCoreAfterAxisCount` | int64/64 | 同上 | T 元素 | ceilDiv S/core | split-after | 主核 |
| 同上 | `tailCoreAfterAxisCount` | int64/64 | 同上 | T 元素 | 尾差 | split-after | 尾核 |
| 同上 | `tailUpdateLoopSize` | int64/64 | 同上 | loop | ceilDiv tail/factor | split-after 尾核 | 可 0 |
| 同上 | `tailUpdateAxisNum` | int64/64 | 同上 | T 元素 | 尾核尾差 | split-after | 可 0 |
| 同上 | `ubQuantaIndxFactor` | int64/64 | 同上 | 索引行 | Host 保留 | 固定 body 未见读取 | inactive |
| 同上 | `ubRowFactor` | int64/64 | 同上 | 行 | Host 保留 | 固定 body 未见读取 | inactive |
| 同上 | `eachCoreIndexCount` | int64/64 | 同上 | 索引行 | ceilDiv N/core | SIMD/det/mask | 主核 |
| 同上 | `tailCoreIndexCount` | int64/64 | 同上 | 索引行 | 尾差 | 尾核 | 可 0 |
| 同上 | `eachCoreVarCount` | int64/64 | 同上 | 行 | Host 保留 | 固定 body 未见读取 | inactive |
| 同上 | `tailCoreVarCount` | int64/64 | 同上 | 行 | Host 保留 | 固定 body 未见读取 | inactive |
| 同上 | `isSplitAfterAxis` | int64/64 | 同上 | bool | route heuristic | SIMD Process | 0/1 |
| 同上 | `isDeterminstic` | int64/64 | 同上 | bool | context deterministic && N>1 | router | 0/1 |
| 同上 | `isSimtWithSort` | int64/64 | 同上 | bool | `N/varInAxis>=3` | router | 0/1 |
| 同上 | `isSimdWithSort` | int64/64 | 同上 | bool | highDup && batch/core>64 | SIMD dynamic sort | 0/1 |
| 同上 | `isSimdNonDeterminstic` | int64/64 | 同上 | bool | non-det && rowBytes>=128 | router | 0/1 |
| 同上 | `isMask` | int64/64 | 同上 | bool | four-part predicate | router/mask body | 0/1，最高优先级 |
| 同上 | `isSplitOneLine` | int64/64 | 同上 | bool | mask 行太宽 | mask body | 0/1 |
| 同上 | `IsContiguous` | int64/64 | 同上 | bool | input view 判断 | 路线审计 | ACLNN 为 1 |
| 同上 | `normCoreHandleIdx` | int64/64 | 同上 | 索引行 | deterministic mask split | CalcMask | 主核 |
| 同上 | `tailCoreHandleIdx` | int64/64 | 同上 | 索引行 | 尾差 | CalcMask | 尾核 |
| 同上 | `calcMaskUsedCoreNum` | int32/32 | 同上 | core | min(AIV,ceilDiv(N,128)) | mask phase early return | >=1 |
| 同上 | `maskNormBlockLen` | int64/64 | 同上 | 行 | varStorage/core | 固定 body 未见读取 | 保留 |
| 同上 | `maskTailBlockLen` | int64/64 | 同上 | 行 | 尾差 | 固定 body 未见读取 | 保留 |
| 同上 | `isDeterminSimt` | int64/64 | 同上 | bool | `rowBytes<128` | router | 0/1 |
| 同上 | `indicesUbFactor` | int64/64 | 同上 | 索引行 | deterministic UB 二分 | CalcMask buffers | >=1 |
| 同上 | `normBlockLoop` | int64/64 | 同上 | loop | ceilDiv norm/factor | CalcMask | 主核 |
| 同上 | `tailBlockLoop` | int64/64 | 同上 | loop | ceilDiv tail/factor | CalcMask | 尾核 |
| 同上 | `normBlockTail` | int64/64 | 同上 | 索引行 | 尾差 | CalcMask | 主核末 loop |
| 同上 | `tailBlockTail` | int64/64 | 同上 | 索引行 | 尾差 | CalcMask | 尾核末 loop |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | 100/200+mask | `CalculateMask/DoOpTilingForSimdMask` | RegBase POD | `scatter_nd_update_apt` mask first branch | `scatter_nd_update_simd_mask.h` + common | ascend950 arch35 |
| R-002 | 100/200+det simt | `DoOpTilingForDeterministic` | 同上 | deterministic SIMT branch | deterministic_simt + deterministic common | 同上 |
| R-003 | 100/200+det simd | 同上 | 同上 | deterministic SIMD branch | deterministic_simd + common | 同上 |
| R-004 | 100/200+simd | `DoOpTilingForSimdNonDetermin` | 同上 | SIMD branch | scatter_nd_update_simd + common | 同上 |
| R-005 | 100/200+simt sort | `SortTiling` | 同上 | SIMT-sort branch | simt_sort + indices_sort_utils + common | 同上 |
| R-006 | 100/200 fallback | `BlockTiling/UbTiling` | 同上 | final else | simt + common | 同上 |

## Kernel 实现

### 公共执行流程

entry 固定 `KERNEL_TYPE_AIV_ONLY`，但 `asc_vf_call` 在 AIV 内启动 SIMT 线程函数，执行族按真实 body 标注。8-bit `DTYPE_UPDATES` 通过 `GetComputeType` 变为 int8。key100 用 uint32 地址类型；key200 根据 output extent 选择 indices dtype 或 int64 offset。`ScatterNdUpdateBase::ComputeOutOfset` 以 MicroAPI gather/multiply/add 在寄存器中生成线性元素偏移，tail 用 `UpdateMask` zeroing；排序 helper 用 radix sort、前置 -1 padding 与 unique boundary。

### Route R-001：SIMD mask

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_regbase.cpp` | `CalculateMask` | `varInAxis<0.1UB && N>5varInAxis && N>1000usedCore && S>19` | `isMask=1` |
| Router | `scatter_nd_update_apt.cpp` | first branch | `tilingData.isMask==1` | mask body |

#### Device 数据流

##### 函数级调用链

```text
ScatterNdUpdateSimdMask::Init -> UB mask zero + stride init
  -> Process -> ProcessMaskOneLine or ProcessMaskMultiLine
    -> CopyIndiceIn -> ComputeOutOfset -> CopyUpdatesIn -> CopyOut*
```

##### 源码等价伪码

```text
per core: zero local mask[varStorageInAxis]                    # 伪码
for owned index rows in original order:
  offset = vector_linearize(indices)
  if valid and localMask[target]==0:
    DMA updates slice(s) to output[target]; localMask[target]=1
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `maskBuf` | UB | `ceilAlign(varInAxis,UB_block)` | INT8 | Duplicate | duplicate filter | kernel | local per core |
| `indicesBuf/outOfstBuf` | UB | `indicesFactor*K*sizeof(U)` / `indicesFactor*sizeof(OFFSET)` | ND/1D | indices/MicroAPI | scalar copyout | kernel | factor tile |
| `dataQueue` | UB | `2*indicesFactor*afterAxisFactor*sizeof(T)` | T | updates DMA | output DMA | tile | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  mask["maskBuf · ceilAlign(varInAxis,ubBlock)"]:1
  idx["indicesBuf · factor*K*sizeof(U)"]:1
  off["offsetBuf · factor*sizeof(OFFSET)"]:1
  data["dataQueue · depth=2 · factor*afterFactor*sizeof(T)"]:1
```

##### 字节账与复用顺序

```text
UB_total = align(varInAxis,ubBlock)+indicesFactor*K*sizeof(U)+indicesFactor*sizeof(OFFSET)
         +8*sizeof(U)+2*indicesFactor*afterAxisFactor*sizeof(T)
workspace_total = 16MiB (non-deterministic baseline)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataQueue slot | CopyUpdatesIn | CopyOut | next tile | EnQue/DeQue/FreeTensor | depth 2 |
| mask lane | Duplicate/commit | later row check | never before end | scalar V_S event | target in range |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| offset | MTE2_V + V_S | DMA/MicroAPI | scalar mask | index/offset | key controls width |
| update | queue | DMA in | DMA out | data slot | last row/col use tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/scatter_nd_update_tiling_regbase.cpp` | `CalculateMask/DoOpTilingForSimdMask` | 完整 mask 与 UB 谓词 |
| Kernel entry/router | `op_kernel/scatter_nd_update_apt.cpp` | mask branch | 最高优先级 |
| Device body/helper | `arch35/scatter_nd_update_simd_mask.h` | `ProcessMask*` | 每核 local mask 去重与两种切法 |

### Route R-002：deterministic SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_regbase.cpp` | `CalcDeterministicCoreSplit` | det、N>1、`S*sizeof(T)<128` | `isDeterminSimt=1` |
| Router | `scatter_nd_update_apt.cpp` | deterministic SIMT branch | det && flag | body |

#### Device 数据流

##### 函数级调用链

```text
InitBase -> InitMaskGm -> SyncAll -> CalcMask(dynamic sort/unsort atomic-max) -> SyncAll
  -> InitUpdateBuffer -> CopyInUpdate -> asc_vf_call<SimtComputeData>(1024 threads)
```

##### 源码等价伪码

```text
zero mask workspace; barrier                                       # 伪码
for index tiles: vector-linearize; atomicMax(mask[targetRow], originalRow); save varIdx[row]
barrier
for owned update tiles in 1024-thread SIMT:
  if varIdx[row] valid and mask[targetRow]==row: output[target+col]=update[row,col]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| mask workspace | user+0 | `(varStorageInAxis+1)*sizeof(TYPE)` | uint32/64 | atomic max | commit predicate | launch | initialized 0 |
| varIdx workspace | after mask | `N*sizeof(TYPE)` | uint32/64 | CalcMask | SimtComputeData | launch | per index row |
| mask-phase UB | separate buffers | aligned index/offset/sort/origin/count/hash | mixed | AIV/MicroAPI | SIMT mask helpers | before Reset | tiling binary-search fit |
| `inQueX` | UB | `2*align(afterAxisFactor*sizeof(T),32)` | T | updates DMA | SIMT | commit phase | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  idx["indicesQueue · align(indicesUbFactor*K*sizeof(U))"]:1
  off["offset/sort/origin/count"]:2
  hash["hash · 128*sizeof(float)"]:1
  upd["after Reset: inQueX · depth=2 · align(afterFactor*sizeof(T))"]:4
```

##### 字节账与复用顺序

```text
mask_phase_UB = align(F*K*sizeof(U))+8*sizeof(U)+align(F*sizeof(OFFSET))+128*4
              +align(F*sizeof(OFFSET))+64+align(F*4)+align(F*4)+64
commit_UB = 2*align(afterAxisFactor*sizeof(T),32)
workspace_total = 16MiB+(varStorageInAxis+N+1)*sizeof(TYPE)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| whole UB | CalcMask | mask helpers | `pipe.Reset` 后 update queue | second SyncAll | mask workspace 已可见 |
| update queue | DMA | SIMT threads | next tile | EnQue/DeQue/Free | asc_vf_call 返回 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| init→mask | `SyncAll` | `InitGlobalMemory` | CalcMask | workspace zero | TYPE width by key |
| winner | `asc_atomic_max` | all cores | second `SyncAll` | mask/varIdx | 最大原始行号 |
| commit | 1024-thread SIMT | UB updates | GM store | output | thread stride; tail count guarded |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_regbase.cpp` | `DoOpTilingForDeterministic` | factors/workspace/core split |
| Kernel entry/router | `scatter_nd_update_apt.cpp` | deterministic SIMT branch | key/offset specialization |
| Device body/helper | `deterministic_simt.h` + `common.h` | `CalcMask/SimtComputeData` | atomic winner、barrier、thread mapping |

### Route R-003：deterministic SIMD

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_regbase.cpp` | `CalcDeterministicCoreSplit` | det、N>1、rowBytes>=128 | flag 0 |
| Router | `scatter_nd_update_apt.cpp` | deterministic SIMD branch | det && !simt | body |

#### Device 数据流

##### 函数级调用链

```text
InitBase -> SyncAll -> CalcMask -> SyncAll -> pipe.Reset
  -> inQueue Alloc -> CopyInUpdate -> EnQue/DeQue -> CopyOutUpdate -> Free
```

##### 源码等价伪码

```text
build identical global last-writer mask                                # 伪码
for each owned original row:
  skip unless mask[targetRow]==originalRow
  for slice tiles: DataCopyPad updates->UB->output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| workspace | 同 R-002 | 同 R-002 | TYPE 32/64 | CalcMask | scalar predicate | launch | two barriers |
| mask-phase UB | 同 R-002 | 同 R-002 | mixed | mask | mask | phase | Reset 后释放 |
| `inQueue_` | UB | `2*align(afterAxisFactor*sizeof(T),32)` | T | DMA | DMA | update phase | depth 2 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  maskPhase["mask-phase buffers · same as R-002"]:4
  update["after Reset: inQueue · depth=2 · align(afterFactor*sizeof(T),32)"]:4
```

##### 字节账与复用顺序

```text
UB_total_by_phase = max(mask_phase_UB, 2*align(afterAxisFactor*sizeof(T),32))
workspace_total = 16MiB+(varStorageInAxis+N+1)*sizeof(TYPE)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| UB whole | mask phase | mask helpers | update phase | `SyncAll` + Reset | GM mask closed |
| queue slot | update DMA | output DMA | next tile | queue lifecycle | tail bytes only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| mask | atomic+SyncAll | all cores | scalar winner check | workspace | last-writer exact integer |
| copy | TQue | MTE2 | MTE3 | update slice | final `updateTailNum` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_regbase.cpp` | `CalcDeterministicUpdateSplit` | wide-row factors |
| Kernel entry/router | `scatter_nd_update_apt.cpp` | deterministic SIMD | route ordering |
| Device body/helper | `deterministic_simd.h` | `Process/CopyOutUpdate` | winner filtering与DMA commit |

### Route R-004：non-deterministic SIMD

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_regbase.cpp` | `CalculateDerivedParams/DoOpTilingForSimdNonDetermin` | !det、rowBytes>=128、mask false | SIMD |
| Router | `scatter_nd_update_apt.cpp` | SIMD branch | flag==1 | body |

#### Device 数据流

##### 函数级调用链

```text
ScatterNdUpdateSimd::Init -> InitBaseBuffer
  -> Process -> ProcessSplitAfter or ProcessSplitIndices
    -> CopyIndice* -> ComputeOutOfset -> optional SortAndComputeUniqueIdx
    -> CopyUpdates* -> CopyOut*(sorted or direct)
```

##### 源码等价伪码

```text
if split-after: each core owns output columns and scans all index rows   # 伪码
else: each core owns index rows
per tile compute offsets; if isSimdWithSort and histogram score>0.01:
  radix-sort offsets and keep selected original positions
DMA strided update rows to UB, then valid offsets to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| base buffers | UB | stride + indices/offset/hash/sort/origin/count | mixed | DMA/vector | scalar DMA | kernel | conditional sort data |
| `dataQueue` | UB | `2*indicesFactor*afterFactor*sizeof(T)` | row-major padded | strided DMA | CopyOut | tile | depth 2 |
| workspace | N/A user data | 16MiB baseline | N/A | framework | library reserve | launch | no operator slices |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  idx["indices/offset/stride/hash"]:2
  sort["sort/origin/unique"]:1
  data["dataQueue · depth=2"]:1
```

##### 字节账与复用顺序

```text
UB_total = 8*sizeof(U)+indicesFactor*K*sizeof(U)+indicesFactor*sizeof(OFFSET)+128*4
         +align(indicesFactor*sizeof(OFFSET)+64,32)+2*indicesFactor*4
         +2*align((indicesFactor+1)*4,32)+2*indicesFactor*afterFactor*sizeof(T)
workspace_total = 16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| sort queues | radix/unique | sorted CopyOut | next index tile | DeQue/Free/EnQue protocol | only dynamic-sort branch |
| dataQueue | strided DMA | output DMA | next col tile | queue lifecycle | padded row stride used |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| offset | MTE2_V/V_S | indices/Vector | scalar CopyOut | offsets | key200 int64 path |
| sort | S_V/V_S | radix/unique | CopyOut | sort buffers | threshold 0.01 |
| DMA | TQue | CopyIn | CopyOut | update rows | row/col/core tails |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_regbase.cpp` | `DoOpTilingForSimdNonDetermin` | split heuristic与factor |
| Kernel entry/router | `scatter_nd_update_apt.cpp` | SIMD branch | router priority |
| Device body/helper | `scatter_nd_update_simd.h` | `ProcessSplitAfter/Indices` | 两种并行域、动态排序 |

### Route R-005：SIMT dynamic-sort

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_regbase.cpp` | `CalculateDerivedParams/SortTiling` | !det、rowBytes<128、`N/varInAxis>=3` | SIMT sort flag |
| Router | `scatter_nd_update_apt.cpp` | simt-sort branch | prior flags false | body |

#### Device 数据流

##### 函数级调用链

```text
ScatterNdUpdateSimtSort::Init -> UB buffers
  -> Process -> Compute -> ComputeSortDimensionOther
  -> IndexStatistic -> SortDataMove or NoSortDataMove
  -> asc_vf_call<SortSimtCompute/NoSortSimtCompute>(1024 threads)
```

##### 源码等价伪码

```text
per UB index-row tile: vector-linearize offsets                    # 伪码
if histogram score>0.01: radix-sort+unique; SIMT stores selected rows
else: SIMT stores all rows directly
each thread strides by 1024 over rows*sliceSize
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| indices queue | UB | `align(ubRows*K*sizeof(U),32)` | U | GM | offset | tile | depth1 |
| offset/sort/origin/unique/hash/stride | UB | formulas in Init | mixed | AIV | SIMT args | tile | sort conditional |
| updates/output | GM direct | `N*S` / output | T | input | SIMT | executor | no update UB materialization |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  idx["indicesQueue · depth=1"]:1
  off["offset + sort"]:1
  aux["origin + unique + hash"]:1
  stride["strideBuf"]:1
```

##### 字节账与复用顺序

```text
UB_total = align(UbRows*K*sizeof(U),32)+align(UbRows*sizeof(OFFSET),32)
         +align(UbRows*sizeof(OFFSET),32)+64+align(UbRows*4,32)
         +align((UbRows+1)*4,32)+128*4+8*sizeof(U)
workspace_total = 16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| sort aux | Sort/unique | SIMT call | next Compute | `asc_vf_call` returns then Free | current tile only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| indices | MTE2_V/V_S | DMA/offset | statistic | UB | currUbTilingSize |
| sort | S_V | radix | unique/SIMT | aux | -1 padding sentinel |
| store | SIMT threads | GM updates | GM output | output | `index<uniqueIdNum*S` guard |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_regbase.cpp` | `SortTiling` | UB 二分和 block 分配 |
| Kernel entry/router | `scatter_nd_update_apt.cpp` | simt-sort branch | key specialization |
| Device body/helper | `scatter_nd_update_simt_sort.h` | `Compute/SortDataMove` | 动态排序与 SIMT 地址映射 |

### Route R-006：plain SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `tiling_regbase.cpp` | `BlockTiling/UbTiling` | prior route flags false | ordinary tiling |
| Router | `scatter_nd_update_apt.cpp` | final else | key100/200 | plain SIMT |

#### Device 数据流

##### 函数级调用链

```text
ScatterNdUpdateSimt::Init -> Process -> ComputeData
  -> ComputeDimensionOther: CopyIn indices+updates -> SimtCompute
  -> or ComputeDimensionOne: CopyIn updates -> SimtComputeDimensionOne
```

##### 源码等价伪码

```text
for each core update tile:                                      # 伪码
  load updates and, if K>=2, required indices into double-buffer UB
  1024 threads stride over update elements
  row=element/S; col=element-row*S; compute target from K indices
  if valid: output[target+col]=update[element]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| `inQueX` | UB | `2*roundUp32(ubTilingSize*sizeof(T))` | T | GM updates | SIMT | tile | depth2 |
| `inQueIdx` | UB | `2*roundUp32((ceil(ub/S)*K+2)*sizeof(U))` | U | indices | SIMT | K>=2 | depth2 |
| stride/outputShape bufs | UB | `8*sizeof(TYPE)` each | TYPE | POD scalar writes | SIMT | K>=2 | barrier before call |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  upd["inQueX · depth=2 · roundUp32(ubTile*sizeof(T))"]:2
  idx["inQueIdx · depth=2 · roundUp32(indexTile*sizeof(U))"]:1
  meta["stride + outputShape · 16*sizeof(TYPE)"]:1
```

##### 字节账与复用顺序

```text
K==1: UB_total = 2*roundUp32(ubTilingSize*sizeof(T))
K>=2: add 2*roundUp32(indiceUbTilingSize*sizeof(U))+16*sizeof(TYPE)
workspace_total = 16MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input queues | DataCopyPad | SIMT | next UB loop | EnQue/DeQue/FreeTensor | asc_vf_call completes |
| metadata TBuf | scalar init | SIMT | kernel end | UB DSB | K>=2 only |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| metadata | `DataSyncBarrier<UB>` | scalar stores | SIMT | stride/shape | exact integer TYPE |
| data | queue + SIMT | DMA | threads | inputs | thread bound and final UB tile |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `tiling_regbase.cpp` | `BlockTiling/UbTiling` | core/UB/tail formulas |
| Kernel entry/router | `scatter_nd_update_apt.cpp` | final else | plain SIMT fallback |
| Device body/helper | `scatter_nd_update_simt.h` | `SimtCompute*` | thread→element映射与rank1快路 |

## 算子特定优化

### Trick T-001：确定性 last-writer 的 mask/varIdx 两阶段协议

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：`TilingContext::GetDeterministic()==1 && N>1`
- Golden 来源：[`scatter_nd_update_deterministic_simd.h:53-67`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_deterministic_simd.h#L53-L67)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    SyncAll();
    this->CalcMask();
    SyncAll();

    if (this->blockIdx >= this->tiling_.usedCoreNumBefore) {
        return;
    }
    this->pipe_.Reset();
    this->pipe_.InitBuffer(inQueue_, 2, Ops::Base::CeilAlign(this->tiling_.afterAxisFactor * sizeof(PARAMS_T), UB_AGLIN_VALUE));
```

#### 机制与设计原因

第一次 barrier 等待 workspace 清零，`CalcMask` 以 original row 为 atomic-max value 竞争每个目标行，同时保存每行线性地址；第二次 barrier 后只有 winner 搬运 updates。于是重复索引无需对值做原子操作，且 SIMD/SIMT commit 共用同一 last-writer 协议；`pipe.Reset` 让 mask 阶段 UB 物理空间复用为双缓冲更新队列。

#### 迁移边界

- 适用：覆盖语义且需“输入中最后一条”确定性的 scatter。
- 前提：950、全核 barrier 可用、workspace 按 TYPE 宽度分配、original row 可由 atomic max 表示。
- 失效：关闭 deterministic 时转 R-001/R-004/R-005/R-006，不得声称重复目标确定。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

省略任一 `SyncAll` 会分别导致未清零 mask 被 atomic 消费，或提交阶段读取尚未完成的 winner/varIdx；源码给出的两道 barrier 是协议组成部分。

## 模板与实现对比

| 对比维度 | R-001 mask | R-002 det SIMT | R-003 det SIMD | R-004 SIMD | R-005 SIMT sort | R-006 SIMT | 设计选择依据 |
|---|---|---|---|---|---|---|---|
| 命中条件/优先级 | 宽行高重复 | det窄行 | det宽行 | non-det宽行 | non-det窄行高重复 | fallback | router 固定优先级 |
| 模板与 dtype | SIMD T/U/OFFSET | SIMT+TYPE | SIMD+TYPE | SIMD | SIMT+sort | SIMT | key100/200 控制地址宽度 |
| 循环与分核 | index/core | index/core+threads | index/core+slice | split-after/indices | row tile+threads | element tile+threads | 行宽与重复度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | local mask+dataQ | mask buffers→updateQ | 同左 | base buffers+dataQ | sort aux | input queues | 无通信 window |
| 内存总占用与复用 | UB local mask | 额外 GM workspace，UB Reset | 同左 | baseline workspace | baseline | baseline | deterministic 付出 mask/varIdx |
| 搬运/Vector/SIMT/NDDMA/通信 API | DMA/MicroAPI | MicroAPI+atomic+SIMT | MicroAPI+DMA | DMA/MicroAPI/Sort | Sort+SIMT GM | DMA+SIMT | 无 Cube/NDDMA/通信 |
| 同步、精度与 tail | event | 2 SyncAll | 2 SyncAll+queue | event/queue | event+SIMT | UB barrier | 全部显式 tail |
| fallback/失败边界 | →R-004 | →R-003 by row | non-det tree | →SIMT by row | →plain SIMT | unknown key no body | flags由Host产生 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 另见无后缀卡 | 另见无后缀卡 | 公开支持 Ascend 950PR/DT |
| Host/Launcher | arch32 key10/11/20/21/30 | 同 A2 | arch35 key100/200+flags |
| Kernel/资源 | AIV 两阶段/大索引 | 同 A2 | AIV+内嵌 SIMT；det workspace mask协议 |

## 限制与失败边界

- 公开 API 要求索引非负且不越界；Device 多数路线保留范围检查，但不能替代 API 约束。通用 SIMT 的多维判断使用 `indiceVal > outputShape[dim]`，固定源码未解释等于维长的边界行为，复用时仍须按公开接口预校验。
- ACLNN 已将三个输入 Contiguous，因此 arch35 非连续 view route 对本入口 inactive；不要据此省略最终 `ViewCopy` 对非连续 `out` 的处理。
- `GetDeterministic()` 的框架级设置实现、16MiB reserved workspace 内部用途、Tiling class 的 ABI offset 固定仓不可见。
- R-001/R-004/R-005/R-006 在重复索引时不提供跨核 last-writer 证明；只有 R-002/R-003 可承诺确定性。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/scatter_nd/docs/aclnnScatterNd.md` | public API | 平台/语义/约束 |
| 入口实现 | `index/scatter_nd_update/op_api/aclnn_scatter_nd.cpp` | `aclnnScatterNdGetWorkspaceSize` | 校验与 executor graph |
| L0 selector | `index/scatter_nd_update/op_api/scatter_nd_update.cpp` | `ScatterNdUpdate` | 950 AICore route |
| Host/Tiling/Launcher | `index/scatter_nd_update/op_host/arch35/scatter_nd_update_tiling_regbase.h` | `ScatterNdUpdateTiling` | 状态/方法 |
| Host/Tiling/Launcher | `index/scatter_nd_update/op_host/arch35/scatter_nd_update_tiling_regbase.cpp` | `DoOpTiling` | selector、资源、workspace、POD |
| Host/Tiling/Launcher | `index/scatter_nd_update/op_host/arch35/scatter_nd_update_tiling_arch35.cpp` | `Tiling4ScatterNdUpdate` | 注册与平台 compile info |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/scatter_nd_update_apt.cpp` | `scatter_nd_update` | key/flag router |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_struct.h` | POD class | ABI 声明 |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_common.h` | base/deterministic common | MicroAPI、sort、atomic、workspace |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_deterministic_simt.h` | deterministic SIMT | thread commit |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_deterministic_simd.h` | deterministic SIMD | DMA commit |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simd.h` | SIMD | split routes |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simd_mask.h` | SIMD mask | UB local dedup |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simt_sort.h` | SIMT sort | dynamic sort |
| Kernel entry/body | `index/scatter_nd_update/op_kernel/arch35/scatter_nd_update_simt.h` | plain SIMT | fallback |
| 构建接线 | `index/scatter_nd_update/CMakeLists.txt` | ascend950→arch35 | active build |
| 构建接线 | `index/scatter_nd_update/op_host/scatter_nd_update_def.cpp` | AICore ascend950 | OpDef/binary filename |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 单 RegBase class按声明顺序 | uint32/uint64混排、8项数组、inactive字段 | `ScatterNdUpdateRegBaseTilingData` |
| 调用顺序 | Contiguous→core→ViewCopy | deterministic 的 init→barrier→mask→barrier→commit | API impl / deterministic Process |
| workspace/内存 | 16MiB baseline与 mask/varIdx 公式 | TYPE 宽度、`varStorageInAxis+1` offset、owner lifetime | `GetWorkspaceSize/InitBase` |
| 同步 | queue/event/UB barrier/SyncAll/atomic | 两道全核 barrier不可省略 | common + bodies |
| 精度与尾块 | key100/200与 OFFSET specialization | MicroAPI tail mask、SIMT thread guards、DMA有效字节 | `GetTilingKey/router` |
| 失败边界 | Host shape/UB/view校验与 unknown key | 不得把非确定性 route 宣称确定 | API/Tiling/router |