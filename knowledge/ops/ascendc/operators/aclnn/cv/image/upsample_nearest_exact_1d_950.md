---
type: Operator
title: "UpsampleNearestExact1d 950 ACLNN 算子知识"
description: "说明 aclnnUpsampleNearestExact1d 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest/op_host/op_api/aclnn_upsample_nearest_exact1d.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/upsample_nearest/op_host/op_api/aclnn_upsample_nearest_exact1d.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_api/resize_nearest_neighbor_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_host/arch35/resize_nearest_neighbor_v2_tiling_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_host/arch35/resize_nearest_neighbor_v2_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_kernel/resize_nearest_neighbor_v2_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_host/resize_nearest_neighbor_v2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_host/config/ascend950/resize_nearest_neighbor_v2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T20:54:48Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# UpsampleNearestExact1d 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnUpsampleNearestExact1dGetWorkspaceSize` / `aclnnUpsampleNearestExact1d`，公开 ACLNN 两段式接口。 |
| 计算语义 | `out[n,c,l_dst] = self[n,c,min(floor((l_dst+0.5)/scales),L-1)]`。 |
| 输入输出 | 3D NCL/ND；FP16、FP32、BF16；输出 N/C 与输入一致，L=`outputSize[0]`。 |
| 执行类型 | 单 Kernel `SIMT`，950 入口最终调用 `ResizeNearestNeighborV2`，entry 显式指定 `KERNEL_TYPE_AIV_ONLY`。 |
| Kernel 路线 | R-001/R-002 等尺寸直拷贝，R-003/R-004 单点广播，R-005/R-006 通用 exact 坐标；每种模式分别使用 uint32/1024 线程或 uint64/512 线程索引。 |
| 核心资源 | 输入/输出 GM、SIMT 标量寄存器；无 AIV Queue/TBuf、无 dynamic UB；Tiling 声明 16 MiB user workspace，但 SIMT body 不读写该区。 |
| 来源闭包 | `source_closed`：公开 ACLNN→950 `IsRegBase` 分支→`ResizeNearestNeighborV2` launcher/Tiling/POD→AIV-only entry→全部可达 SIMT body 与构建接线已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入为 `(N,C,L)`、输出为 `(N,C,L_o)`。公开约束给出 `L_o=floor(L*scales)`；每个输出位置读取 `l_src=min(floor((l_dst+0.5)/scales),L-1)`。950 Host 把 3D 张量 view 为 `(N,C,1,L)`，向底层传入 `alignCorners=false`、`halfPixelCenters=true` 和 scales `[1, scales]`；Tiling 再生成 `scaleH=1`、`scaleW=1/scales`，SIMT 公式与公开 exact 公式一致。数据只做索引搬运，不发生 dtype 转换或算术舍入。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | 3D NCL 或 ND（按 NCL） | FP16/FP32/BF16 | N/C/L 均大于 0；可非连续；950 每维不超过 `2^20`。 |
| `outputSize` | 长度 1 的 `aclIntArray` | INT64 | `outputSize[0]>0`，决定输出 L；底层转换为 `[1,L_o]` INT32 tensor。 |
| `scales` | 标量 | double | 正值时 Host 以 double 计算倒数后窄化为 float，并另存 float 原比例；公开约束要求 `L_o=floor(L*scales)`。 |
| `out` | 3D NCL/ND | 与 `self` 相同 | 非空、N/C 与输入一致；可非连续，最终由 `ViewCopy` 写回。 |
| `workspaceSize/executor` | Host 输出 | uint64 / executor | 由第一段返回，必须原样交给第二段。 |

### 入口与版本

固定仓只公开名为 `aclnnUpsampleNearestExact1d` 的版本。公开产品表明确支持 Ascend 950PR/950DT；950 由 `IsRegBase()` 分支选择 `ResizeNearestNeighborV2`，而不是 A2/A3 的 `UpsampleNearest` body。底层 OpDef、binary config 和 CMake 均接线 `ascend950`，因此本卡覆盖 950 的实际最新公开入口。

## Host/Launcher 执行流程

### 参数校验与前处理

顺序为：检查 `workspaceSize/executor`、创建 executor；检查 `self/outputSize/out` 非空、self dtype 在 FP32/FP16/BF16 且 out 同型、N/C/L 与 `outputSize[0]` 均正、self 为 3D、outputSize 长度为 1。空 tensor 分支在元素检查之后，正常有效输入不会命中。随后 `Contiguous(self)` 与 `Contiguous(out)`；以 double 计算 `1.0/scales` 后转 float，同时把正 scales 转 float；950 将 self/out `UnsqueezeNd(axis=2)` 并 `ReFormat(NCHW)`，构造 size `[1,L_o]` INT32 tensor、scales `[1,originScales]` float 数组，调用 `ResizeNearestNeighborV2(..., false, true, ...)`，最后 `SqueezeNd(axis=2)`、恢复原 storage format 并 `ViewCopy`。

### 参数、Workspace 与生命周期

executor 拥有连续化临时 tensor、view 描述、size/scales 常量和 4D 临时输出。`ResizeNearestNeighborV2` Tiling 把 workspace 固定为 `16*1024*1024` 字节；entry 要求系统 workspace 与 `GetUserWorkspace` 都非空，但六条 SIMT body 只绑定输入/输出 GM，不消费 user workspace。第一段返回的是整个 executor 的 workspace，调用者不得用底层固定值替代。第二段 `CommonOpExecutorRun` 在传入 stream 上执行，任何 `CHECK_RET` 或 Tiling 失败均阻断后续节点。

### 启动图与失败传播

```text
Contiguous + UnsqueezeNd/ReFormat（框架前处理）
  -> ResizeNearestNeighborV2 单 Device 节点
     Host: ResizeNearestNeighborV2AICORE / ADD_TO_LAUNCHER_LIST_AICORE
     entry: resize_nearest_neighbor_v2<schId, NCHW, 0, 1, idxInt32>
     blockDim: realCoreNum=min(N*C*L_o, AIV coreNum)
     POD: ResizeNearestNeighborV2TilingData
     workspace slice: user workspace 16 MiB，SIMT body 不访问
  -> SqueezeNd/ReFormat + ViewCopy（框架后处理）
```

前处理、Kernel、后处理由同一 executor/stream 保序。Kernel 只有一个 Device 节点，无跨 Kernel workspace 边；每个 block 负责互斥的输出线性区间，失败由 launcher/CommonOpExecutorRun 返回。

## Route 与参数设计

### 分核与资源计算

设 `E=N*C*L_o`。Tiling 取 `realCoreNum=min(E, coreNumAiv)`，`splitBlockFactor=floor(E/realCoreNum)`，`splitBlockTailFactor=E-splitBlockFactor*realCoreNum`。block `b<tail` 处理 `factor+1` 个元素，起点 `b*factor+b`；其余处理 `factor` 个，起点 `b*factor+tail`。若输入输出总元素、H/W 轴均满足 32 位上界则使用 uint32 和 1024 个 SIMT threads，否则使用 uint64 和 512 threads；线程 `t` 以 `idx=t; idx<blkProcessNum; idx+=blockDim.x` 遍历。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 等尺寸/32 位 | `(schId=4,NCHW,align=0,half=1,idx32=1)` | `L==L_o` 且 32 位索引 guard 成立 | R-002 |
| 2 | R-002 等尺寸/64 位 | `(4,NCHW,0,1,0)` | `L==L_o` 且 32 位 guard 不成立 | 失败仅来自 Tiling/launch |
| 3 | R-003 单点广播/32 位 | `(5,NCHW,0,1,1)` | `L!=L_o && L==1` 且 32 位 guard 成立 | R-004 |
| 4 | R-004 单点广播/64 位 | `(5,NCHW,0,1,0)` | `L!=L_o && L==1` 且 32 位 guard 不成立 | 失败仅来自 Tiling/launch |
| 5 | R-005 通用 exact/32 位 | `(3,NCHW,0,1,1)` | `L!=L_o && L!=1` 且 32 位 guard 成立 | R-006 |
| 6 | R-006 通用 exact/64 位 | `(3,NCHW,0,1,0)` | `L!=L_o && L!=1` 且 32 位 guard 不成立 | 失败仅来自 Tiling/launch |

```text
format=NCHW；NHWC 两个特化均 false
NCHW Gather 因 originalScaleH=1 非零而 false
if L == L_o: schId=4
else if H==1 && L==1: schId=5
else: schId=3
idx32 = xElems,yElems < UINT32_MAX 且 H/W 各轴 <= INT32_MAX
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | schId 4/idx32 1 | 950、FP16/FP32/BF16、`L=L_o`、32 位 guard | 单 SIMT | `ResizeNearestNeighborV2Simt<T,uint32_t,NCHW,4,false,true>` | GM→GM，1024 threads | block 余数前置分配 |
| R-002 | schId 4/idx32 0 | 同上但 32 位 guard false | 单 SIMT | 同模板、索引 uint64 | GM→GM，512 threads | 同上 |
| R-003 | schId 5/idx32 1 | `L!=L_o,L=1`、32 位 guard | 单 SIMT | `...<T,uint32_t,NCHW,5,false,true>` | 单输入点广播，1024 threads | 同上 |
| R-004 | schId 5/idx32 0 | 同上但 32 位 guard false | 单 SIMT | uint64 版本 | 单输入点广播，512 threads | 同上 |
| R-005 | schId 3/idx32 1 | `L!=L_o,L!=1`、32 位 guard | 单 SIMT | `...<T,uint32_t,NCHW,3,false,true>` | 每线程坐标计算+一次 GM 读写 | clamp 到 `L-1` |
| R-006 | schId 3/idx32 0 | 同上但 32 位 guard false | 单 SIMT | uint64 版本 | 每线程坐标计算+一次 GM 读写 | clamp 到 `L-1` |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 索引位宽 | x/y 元素数 `<UINT32_MAX` 且 H/W 四轴 `<=INT32_MAX` | 允许 uint32 magic division 与 1024-thread callee | R-001/003/005 | uint64、512-thread Route | `MatchTilingStrategyAndSetTilingKey` |
| 直接复制 | `lenSrcH==lenDesH && lenSrcW==lenDesW` | 消除坐标解码和比例计算 | R-001/002 | 单点或通用 | 同上 |
| 单点广播 | 直接复制未命中且 `lenSrcH==1 && lenSrcW==1` | 源空间坐标恒为 `(0,0)` | R-003/004 | 通用 | 同上 |
| workspace 非空 | entry 的 `workspace/userWS != nullptr` | entry 在 dispatch 前设置系统 workspace | 全 Route | 直接 return，无输出 | `resize_nearest_neighbor_v2` |
| block 有效域 | `blockIdx<realCoreNum` | 防止额外 block 越界 | 全 Route | block 无操作 | `ResizeNearestNeighborV2Simt::Process` |

### POD/ABI

`ResizeNearestNeighborV2TilingData` 由注册宏生成；固定仓能确认字段声明顺序和源码类型，但宏展开后的 `sizeof/alignof/offsetof` 不可见。以下每个字段单列，inactive 表示六条 SIMT Route 不读取。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ResizeNearestNeighborV2TilingData` | `realCoreNum` | int64/64 | 宏展开不可见 | block | `min(E,coreNum_)` | `Process` block guard | `1..coreNumAiv` |
| 同上 | `ubSize` | int64/64 | 同上 | byte | 平台 UB size | inactive | SIMT 不分配 UB |
| 同上 | `alignCorners` | int64/64 | 同上 | 布尔槽 | `lenDstHwcNum_`（此路线为 0） | inactive；模板参数为 false | 不可改作真实 attr 读取 |
| 同上 | `halfPixelCenters` | int64/64 | 同上 | 布尔 | attr true | inactive；模板参数为 true | exact 路线固定 1 |
| 同上 | `lenN` | int64/64 | 同上 | 元素轴长 | 4D N | inactive（线性索引反推出 N） | `>0` |
| 同上 | `lenC` | int64/64 | 同上 | 元素轴长 | 4D C | `GetEachDimIdx/CalOutputIdx` | `>0` |
| 同上 | `lenSrcH` | int64/64 | 同上 | 元素 | 1 | `CalSrcIdx` | 固定 1 |
| 同上 | `lenSrcW` | int64/64 | 同上 | 元素 | 输入 L | `CalSrcIdx/CalOutputIdx` | `>0` |
| 同上 | `lenDesH` | int64/64 | 同上 | 元素 | 1 | `GetEachDimIdx` | 固定 1 |
| 同上 | `lenDesW` | int64/64 | 同上 | 元素 | 输出 L | `GetEachDimIdx` | `>0` |
| 同上 | `condition` | int64/64 | 同上 | 枚举 | 默认 0 | inactive | SIMT inactive |
| 同上 | `switchParams` | int64/64 | 同上 | 枚举 | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitBlockFactor` | int64/64 | 同上 | 输出元素/block | `floor(E/realCoreNum)` | `Process` | `>=1` |
| 同上 | `splitBlockTailFactor` | int64/64 | 同上 | block 数 | `E-factor*realCoreNum` | `Process` | `<realCoreNum` |
| 同上 | `lenCAlign` | int64/64 | 同上 | 元素 | C 的 UB block 对齐 | inactive | SIMT inactive |
| 同上 | `hwcNum` | int64/64 | 同上 | 元素 | `H*W*C` | inactive | SIMT inactive |
| 同上 | `dstHwcNum` | int64/64 | 同上 | 元素 | `Ho*Wo*C` | inactive | SIMT inactive |
| 同上 | `wcNum` | int64/64 | 同上 | 元素 | `W*C` | inactive | SIMT inactive |
| 同上 | `dstWcNum` | int64/64 | 同上 | 元素 | `Wo*C` | inactive | SIMT inactive |
| 同上 | `nLoop` | int64/64 | 同上 | loop | 默认 0 | inactive | SIMT inactive |
| 同上 | `nLoopTimesBefore` | int64/64 | 同上 | loop | 默认 0 | inactive | SIMT inactive |
| 同上 | `nLoopTimesLast` | int64/64 | 同上 | loop | 默认 0 | inactive | SIMT inactive |
| 同上 | `nLoopTailLast` | int64/64 | 同上 | 元素 | 默认 0 | inactive | SIMT inactive |
| 同上 | `wcLoop` | int64/64 | 同上 | loop | 默认 0 | inactive | SIMT inactive |
| 同上 | `wcLoopTimesBefore` | int64/64 | 同上 | loop | 默认 0 | inactive | SIMT inactive |
| 同上 | `wcLoopTailBefore` | int64/64 | 同上 | 元素 | 默认 0 | inactive | SIMT inactive |
| 同上 | `wcLoopTimesLast` | int64/64 | 同上 | loop | 默认 0 | inactive | SIMT inactive |
| 同上 | `wcLoopTailLast` | int64/64 | 同上 | 元素 | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitBlockFullCount` | int64/64 | 同上 | block | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitFactorDesH` | int64/64 | 同上 | 元素 | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitFactorTailDesH` | int64/64 | 同上 | 元素 | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitCountDesH` | int64/64 | 同上 | count | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitFactorDesW` | int64/64 | 同上 | byte/元素（按非 SIMT 模板） | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitFactorTailDesW` | int64/64 | 同上 | byte/元素 | 默认 0 | inactive | SIMT inactive |
| 同上 | `splitCountDesW` | int64/64 | 同上 | count | 默认 0 | inactive | SIMT inactive |
| 同上 | `scaleW` | float/32 | 同上 | 输入/输出元素 | `1/originalScaleW` 或 `L/L_o` | `CalSrcIdx` | FP32 坐标 |
| 同上 | `scaleH` | float/32 | 同上 | 输入/输出元素 | `1/originalScaleH=1` | `CalSrcIdx` | 固定 1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/002 | schId 4 + idx 位 | `MatchTilingStrategyAndSetTilingKey` | POD active 字段 | `resize_nearest_neighbor_v2` schId 4 | `ResizeNearestNeighborV2Simt::Process/SimtCompute` | ascend950 OpDef/config/CMake |
| R-003/004 | schId 5 + idx 位 | 同上 | 同上 | entry schId 5 | 同上，INPUT_ONE_MODE | 同上 |
| R-005/006 | schId 3 + idx 位 | 同上 | 同上 | entry schId 3 | 同上，`CalSrcIdx` | 同上 |

## Kernel 实现

### 公共执行流程

entry 检查 workspace，解析 POD，并以编译期 `schId/format/alignCorners/halfPixelCenters/idxInt32` 实例化 SIMT 类；`Init` 仅绑定 GM 和 blockIdx。`Process` 生成 C/H/W 除法 magic 数，按前余数分核确定线性输出区间，并用 `asc_vf_call` 启动 1024 或 512 线程。线程互不写同一输出，无 queue、event、barrier 或 atomic。

### Route R-001：等尺寸直拷贝/uint32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_upsample_nearest_exact1d.cpp` | `upsampleNearestExact1dCompute` | `IsRegBase()`，NCL/ND | NCHW 4D Resize |
| Tiling | `resize_nearest_neighbor_v2_tiling_base.cpp` | `MatchTilingStrategyAndSetTilingKey` | H/W 输入输出均相等且 idx32 guard true | schId 4/idx1 |
| Router | `resize_nearest_neighbor_v2_apt.cpp` | `resize_nearest_neighbor_v2` | schId 4、idx1 | SIMT uint32 |

#### Device 数据流

##### 函数级调用链

```text
ResizeNearestNeighborV2AICORE -> resize_nearest_neighbor_v2<4,NCHW,0,1,1>
  -> ResizeNearestNeighborV2Simt<uint32_t>::Init
  -> Process -> calleeInt32(1024) -> SimtCompute -> GM[y]=GM[x]
```

##### 源码等价伪码

```text
range = balanced_output_range(blockIdx)
for idx = threadIdx.x; idx < range.count; idx += 1024:
  y[range.start + idx] = x[range.start + idx]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `blkStart+idx` | 1 element/thread/iteration | T/NCHW | input GM | output GM | executor/Kernel | 无别名证据 |
| SIMT 标量 | thread-local | 索引与区间 | uint32 | Process | callee | thread | 寄存器布局固定仓不可见 |
| user workspace | framework offset | 16 MiB | byte | executor | entry 仅判空，body 不访问 | executor/本 Kernel | 不与 GM 输入输出别名 |

##### 分层内存布局

N/A：该 SIMT Route 不初始化 AIV UB、shared/dynamic UB；数据直接 GM→标量 load/store→GM。

##### 字节账与复用顺序

```text
UB_total = 0
dynamic_ub = 0（launch 未传 dynamic UB）
workspace_total = 16 MiB（Tiling 声明；body 访问 0 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | 无本地 buffer | 直接 GM | N/A | 每输出唯一线程 | block 区间互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分核 | 互斥线性区间 | Tiling | block/thread | output GM | 前 `tail` blocks 多 1 元素；无数值计算 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling_base.cpp` | `TilingStrategy` | 生成 schId 4 和均衡区间 |
| Kernel entry/router | `...resize_nearest_neighbor_v2_apt.cpp` | schId 4 | 选择 SIMT 模板 |
| Device body/helper | `...resize_nearest_neighbor_v2_simt.h` | `SimtCompute` ALL_COPY_MODE | 同 offset 直拷贝 |

### Route R-002：等尺寸直拷贝/uint64

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_upsample_nearest_exact1d.cpp` | `upsampleNearestExact1dCompute` | `IsRegBase()`，NCL/ND | NCHW 4D Resize |
| Tiling | `...tiling_base.cpp` | `MatchTilingStrategyAndSetTilingKey` | 等尺寸且 idx32 guard false | schId 4/idx0 |
| Router | `...apt.cpp` | `resize_nearest_neighbor_v2` | idx0 | uint64/512 threads |

#### Device 数据流

##### 函数级调用链

```text
entry<4,NCHW,0,1,0> -> Process -> calleeInt64(512) -> SimtCompute -> GM copy
```

##### 源码等价伪码

```text
同 R-001，但所有域索引为 uint64，线程步长为 512。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `blkStart+idx` | 1 T | NCHW | input | output | Kernel | uint64 地址 |
| user workspace | framework offset | 16 MiB | byte | executor | entry 判空 | executor/本 Kernel | body 访问 0 B |

##### 分层内存布局

N/A：无 UB/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=16 MiB（body 访问 0 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| N/A | N/A | GM | N/A | 唯一线程 | 区间互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 512-thread grid-stride | Process | calleeInt64 | 输出区间 | uint64 防溢出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling_base.cpp` | idx guard | 选择 uint64 |
| Kernel entry/router | `...apt.cpp` | idxInt32=false | 实例化 uint64 |
| Device body/helper | `...simt.h` | `calleeInt64` | 512 threads |

### Route R-003：单点广播/uint32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_upsample_nearest_exact1d.cpp` | `upsampleNearestExact1dCompute` | `IsRegBase()`，NCL/ND | NCHW 4D Resize |
| Tiling | `...tiling_base.cpp` | selector | 非等尺寸且 H=W=1、idx32 | schId 5/idx1 |
| Router | `...apt.cpp` | schId 5 | INPUT_ONE_MODE | uint32 SIMT |

#### Device 数据流

##### 函数级调用链

```text
entry<5,...,1> -> Process -> calleeInt32 -> SimtCompute
  -> GetEachDimIdx -> CalOutputIdx(N,C,0,0) -> GM broadcast
```

##### 源码等价伪码

```text
for owned y index: decode (N,C,H,W); y[index]=x[N*C + C]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `n*lenC+c` | 1 T | NCHW | source | thread | executor | 多线程只读复用 |
| output GM | `blkStart+idx` | 1 T | NCHW | thread | result | executor | 唯一写者 |
| user workspace | framework offset | 16 MiB | byte | executor | entry 判空 | executor/本 Kernel | body 访问 0 B |

##### 分层内存布局

N/A：直接 GM 广播，无 UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=16 MiB（body 访问 0 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 输入 GM 单点 | API/input | 多线程 | 不覆盖 | stream input lifetime | 只读 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| broadcast | 普通 GM load/store | 输入 | thread | 输入单点 | 无 cast；输出尾由区间 count 限制 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling_base.cpp` | schId 5 | H=W=1 特化 |
| Kernel entry/router | `...apt.cpp` | schId 5 | SIMT |
| Device body/helper | `...simt.h` | INPUT_ONE_MODE | 固定源空间 0,0 |

### Route R-004：单点广播/uint64

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_upsample_nearest_exact1d.cpp` | `upsampleNearestExact1dCompute` | `IsRegBase()`，NCL/ND | NCHW 4D Resize |
| Tiling | `...tiling_base.cpp` | selector | 单点且 idx32 false | schId 5/idx0 |
| Router | `...apt.cpp` | idx0 | uint64 | 512 threads |

#### Device 数据流

##### 函数级调用链

```text
entry<5,...,0> -> Process -> calleeInt64 -> INPUT_ONE_MODE broadcast
```

##### 源码等价伪码

```text
同 R-003，域索引为 uint64、步长 512。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | uint64 `N*C+C` / output index | 1 T | NCHW | thread | output | Kernel | 只读源/唯一写者 |
| user workspace | framework offset | 16 MiB | byte | executor | entry 判空 | executor/本 Kernel | body 访问 0 B |

##### 分层内存布局

N/A：无 UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=16 MiB（body 访问 0 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 输入 GM | API | 多线程 | 不覆盖 | stream lifetime | 只读 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 512-thread grid-stride | Tiling | threads | 大索引 | uint64 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling_base.cpp` | idx guard | uint64 |
| Kernel entry/router | `...apt.cpp` | schId 5 | 单点模式 |
| Device body/helper | `...simt.h` | `calleeInt64` | 512 threads |

### Route R-005：通用 exact/uint32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_upsample_nearest_exact1d.cpp` | `upsampleNearestExact1dCompute` | `IsRegBase()`，NCL/ND | NCHW 4D Resize |
| Tiling | `...tiling_base.cpp` | selector | 非等尺寸、L!=1、idx32 | schId 3/idx1 |
| Router | `...apt.cpp` | schId 3 | COMMON | uint32 SIMT |

#### Device 数据流

##### 函数级调用链

```text
entry<3,NCHW,0,1,1> -> Process -> calleeInt32 -> SimtCompute
  -> GetEachDimIdx -> CalSrcIdx(floor half-pixel) -> CalOutputIdx -> GM load/store
```

##### 源码等价伪码

```text
for owned yIndex:
  decode N,C,H,W using uint32 magic division
  srcW=min(floor((W+0.5)*scaleW), L-1); srcH=0
  xIndex=n*(lenC*L)+c*L+srcW
  y[yIndex]=x[xIndex]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `n*(lenC*L)+c*L+srcW` | 1 T | NCHW | input | thread | executor | 非连续 gather load |
| output GM | `blkStart+idx` | 1 T | NCHW | thread | result | executor | 连续唯一写 |
| thread scalar | local | magic/indices/scale | uint32+float | Process | CalSrcIdx | thread | 无共享 |
| user workspace | framework offset | 16 MiB | byte | executor | entry 判空 | executor/本 Kernel | body 访问 0 B |

##### 分层内存布局

N/A：无 UB/dynamic UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=16 MiB（body 访问 0 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | 当前迭代 | 地址计算 | 下一迭代 | thread 顺序 | 不跨线程 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 坐标 | `floorf/min` | FP32 scale | GM load | src index | half-pixel；clamp `L-1` |
| 写回 | 唯一线程 | load | store | output element | 无 cast/atomic；count 防越界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling_base.cpp` | `SetScales/TilingStrategy` | reciprocal scale、区间 |
| Kernel entry/router | `...apt.cpp` | schId 3 | 通用 SIMT |
| Device body/helper | `...simt.h` | `CalSrcIdx/SimtCompute` | exact 坐标和 GM gather |

### Route R-006：通用 exact/uint64

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_upsample_nearest_exact1d.cpp` | `upsampleNearestExact1dCompute` | `IsRegBase()`，NCL/ND | NCHW 4D Resize |
| Tiling | `...tiling_base.cpp` | selector | 通用且 idx32 false | schId 3/idx0 |
| Router | `...apt.cpp` | idx0 | uint64 COMMON | 512 threads |

#### Device 数据流

##### 函数级调用链

```text
entry<3,NCHW,0,1,0> -> Process -> calleeInt64 -> SimtCompute -> exact GM gather
```

##### 源码等价伪码

```text
同 R-005，但域索引、magic division 和地址为 uint64，线程步长为 512。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | uint64 exact 索引 | 1 T read + 1 T write | NCHW | input/thread | output | Kernel | 无本地 buffer |
| user workspace | framework offset | 16 MiB | byte | executor | entry 判空 | executor/本 Kernel | body 访问 0 B |

##### 分层内存布局

N/A：无 UB。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=16 MiB（body 访问 0 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | 当前迭代 | 地址计算 | 下一迭代 | thread 顺序 | 不跨线程 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| SIMT | 512-thread loop | Tiling | calleeInt64 | 大域 | uint64 地址；坐标仍 FP32 floor/clamp |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `...tiling_base.cpp` | idx guard | 大域 fallback |
| Kernel entry/router | `...apt.cpp` | idxInt32=false | uint64 模板 |
| Device body/helper | `...simt.h` | `calleeInt64` | 512 threads、exact gather |

## 算子特定优化

### Trick T-001：按语义特化消除通用坐标计算

#### 触发与源码依据

- 适用 Route：R-001/R-002、R-003/R-004。
- 触发条件：输入输出 H/W 完全相同，或未命中等尺寸且输入 H=W=1。
- Golden 来源：[`image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_simt.h:126-148`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_simt.h#L126-L148)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (mode == ALL_COPY_MODE) {
    outputGm[yGmIdx] = inputGm[yGmIdx];
    continue;
}
...
if constexpr (mode == INPUT_ONE_MODE) {
    outputGm[yGmIdx] = inputGm[CalOutputIdx<T, T_IDX, format, mode, align_corner, half_pixel>(N, C, 0, 0, lenC,
        1, 1, 1, lenC)];
    continue;
}
```

#### 机制与设计原因

`schId` 是编译期模板参数；等尺寸分支直接同地址复制，省去 N/C/H/W 解码、FP32 scale、`floorf` 和 clamp。单点分支仍需从输出索引得到 N/C，但源 H/W 固定为零，避免比例坐标计算。`if constexpr` 令无关通用代码不进入对应实例，而不是运行时逐元素分支。

#### 迁移边界

- 适用：纯 nearest 搬运且能在 Host/Tiling 证明空间轴恒等或源空间为 1。
- 前提：输入输出 dtype/layout 一致；每个输出只有一个写者；特化谓词必须先于通用路线。
- 失效：非等尺寸且源 L 不为 1 时必须使用 R-005/R-006 的 exact 坐标。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不能在 `L!=L_o` 时使用同 offset 复制，也不能把 `L==1` 广播推广到多点输入。

### Trick T-002：索引位宽决定线程规模

#### 触发与源码依据

- 适用 Route：全部；R-001/003/005 为 32 位，R-002/004/006 为 64 位。
- 触发条件：Tiling 的元素数/轴长 guard。
- Golden 来源：[`image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_simt.h:201-211`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_simt.h#L201-L211)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (sizeof(T_IDX) == sizeof(uint64_t)) {
    asc_vf_call<calleeInt64<T, T_IDX, format, mode, align_corner, half_pixel>>(dim3(THREAD_NUM_MIDDLE),
        scaleH, scaleW, lenC, lenDesH, lenDesW, lenSrcH, lenSrcW, shiftC, mC, shiftH, mH, shiftW, mW, lenSrcWH,
        lenSrcWC, (__gm__ T *)(inputGm_.GetPhyAddr()), (__gm__ T *)(outputGm_.GetPhyAddr()), blkStartOffset,
        blkProcessNum);
} else {
    asc_vf_call<calleeInt32<T, T_IDX, format, mode, align_corner, half_pixel>>(dim3(THREAD_NUM), scaleH,
```

#### 机制与设计原因

可证明安全的小域使用 uint32 magic division 并启动 1024 threads；大域切换 uint64 地址和 512 threads，避免索引溢出。线程数与 index template 绑定，Route 不能只按 schId 合并。

#### 迁移边界

- 适用：线性域可在 Host 精确证明 32 位安全、且 Device 同时提供 32/64 位实现。
- 前提：所有参与乘加的 shape 与总元素 guard 同时成立。
- 失效：任一 guard 不成立必须使用 uint64/512-thread Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得仅检查输出元素数而忽略输入总量和各空间轴上界。

## 模板与实现对比

| 对比维度 | R-001/002 等尺寸 | R-003/004 单点 | R-005/006 通用 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | 第一优先：L 相等 | 第二：L=1 | 最终 fallback | selector 顺序不可交换 |
| 模板与 dtype | schId4，T 为原 dtype | schId5 | schId3 | dtype 均不转换 |
| 循环与分核 | 同一均衡线性域 | 同左 | 同左 | E 决定 block 区间 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅 thread scalar | 同左 | 增加坐标 scalar | 无 UB/通信 |
| 内存总占用与复用 | 0 B UB | 0 B UB，源只读复用 | 0 B UB | workspace body 均不访问 |
| 搬运/Vector/SIMT/NDDMA/通信 API | 同 offset GM copy | 固定源点 GM load | indexed GM gather | 语义特化 |
| 同步、精度与 tail | 无算术；1024/512 | 无算术；1024/512 | FP32 scale+floor+clamp | index guard 决定线程数 |
| fallback/失败边界 | 不等尺寸→下一类 | L!=1→通用 | Tiling/launch 失败 | 无额外 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，但本任务未发布：可达 Transpose 核心实现固定仓不可见 | 同 A2 | 公开支持并闭合 |
| Host/Launcher | 非 RegBase：Transpose→UpsampleNearest→Transpose | 同 A2 | RegBase：view→ResizeNearestNeighborV2→view |
| Kernel/资源 | UpsampleNearest AIV Vector；完整 ACLNN 图缺 Transpose body | 同 A2 | AIV-only entry 内 SIMT，0 B UB、16 MiB 声明 workspace |

## 限制与失败边界

- 950 每个维度不超过 `2^20`，公开文档还要求 `N*ceil(C/16)*16*(L+L_o)*sizeof(dtype)<60 GiB`，且 `L_o=floor(L*scales)`。
- API 实现先要求各轴正数，因此其后“空 tensor 成功”分支对常规 0 轴输入不可达；不得据此宣称支持空 tensor。
- `outputSize` 在底层窄化为 INT32 tensor；公开每维 `<=2^20` 保证 950 文档域内安全。
- 坐标比例由 double 倒数窄化为 float，再由 Device `floorf` 使用；极端比例的数值边界只按该顺序复用。
- 固定仓无法展开 Tiling 注册宏，因此 POD 的 `sizeof/alignof/offsetof` 不可见。
- A2/A3 的 ACLNN 图包含两次 `l0op::Transpose`，本仓仅有 stub/声明、没有其 Host/Tiling/Kernel body，故未生成无后缀正式卡。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `image/upsample_nearest/docs/aclnnUpsampleNearestExact1d.md` | 产品表/公式/约束 | 950 公开支持与语义 |
| 入口声明/约束 | `image/upsample_nearest/op_host/op_api/aclnn_upsample_nearest_exact1d.h` | 两段式函数 | ABI 声明 |
| 入口实现 | `image/upsample_nearest/op_host/op_api/aclnn_upsample_nearest_exact1d.cpp` | `GetWorkspaceSize/upsampleNearestExact1dCompute` | 校验、950 分支、view、scale |
| 入口实现 | `image/resize_nearest_neighbor_v2/op_api/resize_nearest_neighbor_v2.cpp` | `ResizeNearestNeighborV2/AICORE` | dtype selector 与 launcher |
| Host/Tiling/Launcher | `image/resize_nearest_neighbor_v2/op_host/arch35/resize_nearest_neighbor_v2_tiling_base.h` | `ResizeNearestNeighborV2TilingData` | POD 声明 |
| Host/Tiling/Launcher | `image/resize_nearest_neighbor_v2/op_host/arch35/resize_nearest_neighbor_v2_tiling_base.cpp` | selector/Tiling/Fill | 六 Route 与参数 |
| Kernel entry/body | `image/resize_nearest_neighbor_v2/op_kernel/resize_nearest_neighbor_v2_apt.cpp` | entry | AIV-only router |
| Kernel entry/body | `image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_tiling_key.h` | template key | schId/format/attr/index key |
| Kernel entry/body | `image/resize_nearest_neighbor_v2/op_kernel/arch35/resize_nearest_neighbor_v2_simt.h` | SIMT class/callees | 线程、索引、搬运、tail |
| 构建接线 | `image/resize_nearest_neighbor_v2/op_host/resize_nearest_neighbor_v2_def.cpp` | `AddConfig("ascend950")` | 950 dtype/format/entry 文件 |
| 构建接线 | `image/resize_nearest_neighbor_v2/op_host/config/ascend950/resize_nearest_neighbor_v2_binary.json` | NCHW binaries | FP16/FP32/BF16 950 binary |
| 构建接线 | `image/resize_nearest_neighbor_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` | ascend950 arch35 接线 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | active shape/scale/分核字段 | 注册宏布局不可猜；inactive 字段仍须保持声明顺序 | `ResizeNearestNeighborV2TilingData` |
| 调用顺序 | view→Resize→view/ViewCopy | `align=false,half=true` 与 scales `[1,s]` | `upsampleNearestExact1dCompute` |
| workspace/内存 | SIMT body 0 B UB | entry 仍要求非空 user workspace；第一段 size 原样传递 | entry/Tiling |
| 同步 | 每输出唯一线程、stream 节点保序 | 不引入跨线程共享写 | `Process/SimtCompute` |
| 精度与尾块 | double reciprocal→float scale→`floorf`→clamp | 顺序、half pixel 与索引位宽 guard | API + `CalSrcIdx` |
| 失败边界 | 校验/Tiling/launcher 任一失败阻断后继 | 不把 entry 的早退当成功输出 | `CHECK_RET/CommonOpExecutorRun` |
