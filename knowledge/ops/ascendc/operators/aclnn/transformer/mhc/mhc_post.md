---
type: Operator
title: "MhcPost A2/A3 ACLNN 算子知识"
description: "说明 aclnnMhcPost 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/docs/aclnnMhcPost.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/op_api/aclnn_mhc_post.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/op_api/aclnn_mhc_post.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/op_api/mhc_post.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/mhc_post_def.cpp
    role: constraint
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/mhc_post_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/mhc_post_tiling_base.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/mhc_post_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/arch35/mhc_post_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/arch35/mhc_post_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/mhc_post_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/mhc_post_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/mhc_post.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/mhc_post.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T15:10:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# MhcPost A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMhcPostGetWorkspaceSize` / `aclnnMhcPost`，公开两段式 ACLNN；固定仓没有更高版本后缀入口 |
| 计算语义 | `y[q,i,d]=hOut[q,d]*hPost[q,i]+sum(j=0..n-1,x[q,j,d]*hRes[q,j,i])`，`q=T` 或 `q=b*S+s` |
| 输入输出 | TND/BSND、ND；`x/hOut/y` 为 FP16/BF16，`hRes/hPost` 为 FP32；A2/A3 要求 `n=4, 0<D<100000, D%128=0` |
| 执行类型 | `AIV_VECTOR` |
| Kernel 路线 | R-001 流式 X（key 0）；R-002 常驻 X（key 1） |
| 核心资源 | 五个 GM 张量、三条 depth=2 TQue、三个 FP32 TBuf、标量寄存器；Kernel 不消费 workspace |
| 来源闭包 | `source_closed`：ACLNN→Contiguous/L0→Tiling/POD/key→arch22 AIV entry/body→注册与构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

把 TND 的 `T` 或 BSND 的 `B*S` 统一记为 `Q`。每个 `q,i,d` 先以 FP32 标量 `hPost[q,i]` 缩放 `hOut[q,d]`，再按 `j` 顺序用 `hRes[q,j,i]` 对 `x[q,j,d]` 做 `Axpy`。索引 `j*n+i` 实现 `hRes` 转置乘法；输入先转 FP32，累加保持 FP32，最终以 `CAST_RINT` 转回 FP16/BF16，只提交真实 D 域。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[T,n,D]` 或 `[B,S,n,D]`，ND | FP16/BF16 | 必选；A2/A3 Tiling 要求所有维正、`n=4`、`D<100000` 且 128 整除 |
| `hRes` | `[T,n,n]` 或 `[B,S,n,n]`，ND | FP32 | 必选；按 `[q,j,i]` 读取 |
| `hOut` | `[T,D]` 或 `[B,S,D]`，ND | 与 `x` 相同 | 必选；向输出流维 i 广播 |
| `hPost` | `[T,n]` 或 `[B,S,n]`，ND | FP32 | 必选；每个输出流的缩放标量 |
| `out` | 与 `x` 同 shape，ND | 与 `x` 相同 | 调用方提供；ACLNN 显式拒绝 `x/out` private storage format |
| 属性 | N/A | N/A | 公开接口没有属性 |

### 入口与版本

固定源码只有无版本后缀的两段式入口。第一段校验并记录执行图，第二段在调用方 stream 执行。`mhc_post_def.cpp` 把 `mhc_post` 注册给 `ascend910b`（A2）和 `ascend910_93`（A3），两代共用最新 arch22 body；950 映射到独立 `mhc_post_apt`，不属于本卡。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckNotNull` 依次拒绝五个 tensor 空指针；`CheckFormat` 拒绝 `x/out` private format；`CheckDtype` 限定 `x∈{FP16,BF16}`、`hRes/hPost=FP32`、`hOut/out` 与 x 同 dtype；`CheckShape` 逐维核对 TND/BSND。executor 创建后，任一输入为空立即返回 `workspaceSize=0`，不创建 Kernel/ViewCopy。非空时依次对 `x,hRes,hOut,hPost` 调用 `Contiguous`，任一步失败即终止；L0 按 x 的 view shape/dtype/format 分配内部 output，最后 `ViewCopy` 到 out。

### 参数、Workspace 与生命周期

`MhcPostAICore` 用 `ADD_TO_LAUNCHER_LIST_AICORE` 按 `(x,hRes,hOut,hPost)->output` 记录任务。Tiling 查询 AIV 核数、UB 容量和 `GetLibApiWorkSpaceSize()`，后者写 `workspaceSizes[0]`；Device `Init` 虽接收 `workspace` 却不读取。executor 持有 contiguous 临时量、内部输出和任务图；调用方须保留 executor 并按第一段大小提供 Device workspace，供第二段在指定 stream 提交。

### 启动图与失败传播

启动链为 `aclnnMhcPostGetWorkspaceSize→l0op::MhcPost→MhcPostAICore→ADD_TO_LAUNCHER_LIST_AICORE`。`Contiguous` 与末尾 `ViewCopy` 是框架节点；算子拥有的核心 launch graph 只有一个 Device 节点：

| 节点 | Host 添加/启动 | entry/执行族 | blockDim 与 POD | 输入→输出 | workspace slice |
|---|---|---|---|---|---|
| N1 MhcPost | `l0op::MhcPostAICore:ADD_TO_LAUNCHER_LIST_AICORE` | `mhc_post<0/1>` / `AIV_VECTOR` | `usedCoreNum`；`MhcPostTilingData` 96 B；bool key 0/1 | `x,hRes,hOut,hPost→output` | Device `Init` 接收但不读取 workspace；无算子数据 slice |

```text
Contiguous(x,hRes,hOut,hPost) [框架边界]
  -> N1 MhcPost AIV_VECTOR [单 Kernel]
  -> ViewCopy(internal output, out) [框架边界]
```

运行时由 `TilingForMhcPost→DoTilingImpl` 产生 blockDim、key、POD 与 LibApi workspace 大小。Host Tiling 校验失败返回 `GRAPH_FAILED`，阻断 N1；launcher 失败返回 nullptr，阻断 ViewCopy；第二段 `CommonOpExecutorRun` 非成功统一返回 `ACLNN_ERR_INNER`。N1 无跨 Kernel 边、event 或 owner 交接；executor 拥有框架临时量与任务图，stream 保证本执行图的提交顺序，固定仓没有本算子 fallback。

## Route 与参数设计

### 分核与资源计算

令 `Q=T` 或 `B*S`、`coreNum=GetCoreNumAiv()`、`UB=GetCoreMemSize(UB)`。初值为 `bsOuter=Q, bsInner=bsTail=1, dOuter=1, dInner=D`。基础路线每个 D 元素预算 `3*(2*2B+1*4B)=24B`，`maxTileD=UB/24`；当 `Q*dOuter<=coreNum/2` 或 `dInner>=maxTileD` 时，若 `dInner>256` 则 `dOuter*=2,dInner=D/dOuter`。随后 `dInner=alignUp(dInner,16)`、`dOuter=ceilDiv(D,dInner)`、`dTail=D-(dOuter-1)*dInner`、`dTailAlign=alignUp(dTail,16)`。令 `totalCount=Q*dOuter`，先 `used=min(totalCount,coreNum)`，再 `normal=ceilDiv(totalCount,used)`、`used=ceilDiv(totalCount,normal)`、`tail=totalCount-(used-1)*normal`；末核处理 tail 项，其余核处理 normal 项。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 流式 X | key 0 / `MhcPostKernel<T,0>` | 完成共同 Tiling 后 `8*(n+2)*dInner>UB`，`usePermanentX_` 保持 0 | 否则 R-002 |
| 2 | R-002 常驻 X | key 1 / `MhcPostKernel<T,1>` | `8*(n+2)*dInner<=UB` 时置 `usePermanentX_=1`；A2/A3 已先约束 `n=4` | 否则 R-001；无第三路线 |

```text
ACLNN 校验成功且非空
  -> A2/A3: n==4 && 0<D<100000 && D%128==0，否则 GRAPH_FAILED
  -> ComputeTiling
  -> if 8*(n+2)*dInner <= UB: key 1 / R-002
     else: key 0 / R-001
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 流式 X | key 0 / `USE_PERMANENT_X=0` | FP16/BF16；TND/BSND shape 一致；A2/A3 上 `n=4, 0<D<100000, D%128=0`；且 `8*(n+2)*dInner>UB` | 单节点：`mhc_post<0>`，`blockDim=usedCoreNum` | `MhcPostKernel<DTYPE_X,0>::Process→ComputeCopyOut` | 每个 `(q,dTile,i,j)` 逐行搬 X；三条 depth=2 TQue、三个 FP32 TBuf；TQue 协议同步 | 末核用 `tailCoreProcessNum`，末 D tile 用 `dTail`；条件不满足转 R-002 |
| R-002 常驻 X | key 1 / `USE_PERMANENT_X=1` | 同上公共条件；且 `8*(n+2)*dInner<=UB` | 单节点：`mhc_post<1>`，`blockDim=usedCoreNum` | `MhcPostKernel<DTYPE_X,1>::Process→ComputeCopyOutAllX` | 每 item 一次多 block 搬 n 行 X 并在 UB 复用；arch22 每次 `Axpy` 前后 `PIPE_V` barrier | 末 tile local 行距用 `dTailAlign`、只写 `dTail`；无第三路线 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| D 切片资源上界 | 循环条件 `Q*dOuter<=coreNum/2 \|\| dInner>=UB/24`；只有 `dInner>256` 才继续二分 | R-001 的三条 depth=2 TQue 与三个 FP32 TBuf合计 `24*dInner` B；同时尝试增加可并行 item | 两条 Route 共用 | `dInner<=256` 时停止二分，随后 16 元素对齐并重算 tail | `mhc_post_tiling.cpp:MhcPostTilingBase::ComputeTiling` |
| 常驻 X UB 上界 | `8*(n+2)*dInner<=UB` | R-002 的 X queue 与 X FP32 TBuf按 n 倍扩容，总 UB 为 `8*(n+2)*dInner` B | R-002 | 条件为假使用 R-001 | `mhc_post_tiling.cpp:ComputeTiling`；`mhc_post.h:Init` |
| arch22 规格 | `socVer!=4` 时 `n==4 && 0<D<100000 && D%128==0` | 地址/搬运路线只对该公开规格发布，D 的 128 对齐也保证完整 tile 基础对齐 | R-001/R-002 | `GRAPH_FAILED`，无 fallback | `mhc_post_tiling.cpp:CheckSpecConstraints` |
| 多 block 搬运窄化 | R-002 将 `n` 强转 `uint16_t`、`dNum*sizeof(T)` 与 src stride 强转 `uint32_t` | `DataCopyExtParams` 字段位宽 | R-002 | 固定 Host 未给额外 guard；属于不可见的超界边界 | `mhc_post.h:CopyInX` |
| AIV-only 编译选择 | key 头只声明 `ASCENDC_TPL_AIV_ONLY`，bool key 只允许 0/1 | 两条实例都必须落到 Vector 核 | R-001/R-002 | 不存在 AIC/Cube fallback | `mhc_post_tiling_key.h:ASCENDC_TPL_SEL` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `MhcPostTilingData` | `n` | int64/64 | `sizeof=96, alignof=8, offsetof=0` | 元素 | `PostTiling:n_` | `Init/Compute*` 循环与容量 | A2/A3 为 4 |
| `MhcPostTilingData` | `d` | int64/64 | `sizeof=96, alignof=8, offsetof=8` | 元素 | `PostTiling:d_` | GM 行跨度 | `(0,100000)` 且 128 整除 |
| `MhcPostTilingData` | `usedCoreNum` | int64/64 | `sizeof=96, alignof=8, offsetof=16` | 核 | `PostTiling:usedCoreNum_` | `Init/Process` guard | 同时作为 blockDim |
| `MhcPostTilingData` | `normalCoreProcessNum` | int64/64 | `sizeof=96, alignof=8, offsetof=24` | 项/核 | `PostTiling:normalCoreProcessNum_` | 普通核 count/itemStart | ceilDiv 结果 |
| `MhcPostTilingData` | `tailCoreProcessNum` | int64/64 | `sizeof=96, alignof=8, offsetof=32` | 项 | `PostTiling:tailCoreProcessNum_` | 末核 count | 差值公式 |
| `MhcPostTilingData` | `bsInner` | int64/64 | `sizeof=96, alignof=8, offsetof=40` | 项 | `PostTiling:1` | arch22 body 不读取 | inactive 但保留 |
| `MhcPostTilingData` | `bsOuter` | int64/64 | `sizeof=96, alignof=8, offsetof=48` | q 项 | `PostTiling:Q` | arch22 body 不读取 | inactive 但保留 |
| `MhcPostTilingData` | `bsTail` | int64/64 | `sizeof=96, alignof=8, offsetof=56` | 项 | `PostTiling:1` | arch22 body 不读取 | inactive 但保留 |
| `MhcPostTilingData` | `dInner` | int64/64 | `sizeof=96, alignof=8, offsetof=64` | 元素/tile | `PostTiling:alignUp(...,16)` | UB 容量/地址 | 16 元素对齐 |
| `MhcPostTilingData` | `dOuter` | int64/64 | `sizeof=96, alignof=8, offsetof=72` | tile/q | `PostTiling:ceilDiv(D,dInner)` | item 解码 | 至少 1 |
| `MhcPostTilingData` | `dTail` | int64/64 | `sizeof=96, alignof=8, offsetof=80` | 有效元素 | `PostTiling:D-(dOuter-1)*dInner` | 末 tile dNum | `1..dInner` |
| `MhcPostTilingData` | `dTailAlign` | int64/64 | `sizeof=96, alignof=8, offsetof=88` | 对齐元素 | `PostTiling:alignUp(dTail,16)` | R-002 X local 行距 | 16 元素对齐 |

Host 的 `mhc_post_tiling.h` 显式包含 `arch35/mhc_post_tiling_data.h` 与 `arch35/mhc_post_tiling_key.h`，A2/A3 Device body 则包含根目录同名头；两份 `MhcPostTilingData` 都是同序 12×int64，两个 key 头也都只声明 AIV-only 的 bool 0/1，因此 Host 序列化与 arch22 Device 解包 ABI 闭合。`usePermanentX` 不在 POD 内，而由 `GET_TPL_TILING_KEY(usePermanentX_)` 编为模板布尔 key。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 流式 X | key 0 / `USE_PERMANENT_X=0` | `mhc/mhc_post/op_host/mhc_post_tiling.cpp:ComputeTiling/GetTilingKey` | Host arch35 POD/key 头→Device 根目录同布局 POD/key；xFactor=1 | `mhc/mhc_post/op_kernel/mhc_post.cpp:mhc_post<0>` | `mhc/mhc_post/op_kernel/mhc_post.h:Process/ComputeCopyOut/CopyInX/CopyOutTile` | `mhc/mhc_post/op_host/CMakeLists.txt:add_modules_sources` |
| R-002 常驻 X | key 1 / `USE_PERMANENT_X=1` | `mhc/mhc_post/op_host/mhc_post_tiling.cpp:ComputeTiling/GetTilingKey` | 同一跨目录同布局 ABI；xFactor=n | `mhc/mhc_post/op_kernel/mhc_post.cpp:mhc_post<1>` | `mhc/mhc_post/op_kernel/mhc_post.h:Process/ComputeCopyOutAllX/CopyInX/CopyOutTile` | `mhc/mhc_post/op_host/CMakeLists.txt:add_modules_sources` |

## Kernel 实现

### 公共执行流程

entry 声明 `KERNEL_TYPE_AIV_ONLY`，解包 POD，构造 `TPipe` 和模板 Kernel，执行 `Init→Process`。有效核把线性 item 解成 `bsIdx=global/dOuter` 与 `dIdx=global-bsIdx*dOuter`，先搬一个 hOut D tile；两路线均在 FP32 TBuf 中执行 `Cast→Muls→n次Axpy→CAST_RINT`，经 VECOUT 队列写 `y[q,i,dTile]`。hRes/hPost 用 `GetValue` 标量读取；workspace、L1、L0、跨核同步均未用于计算。

### Route R-001：流式 X

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.cpp | `aclnnMhcPostGetWorkspaceSize/CheckParam` | 非空且 dtype/shape/format 成功；空输入提前成功 | L0 或终止 |
| Tiling | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `ComputeTiling/GetTilingKey` | `8*(n+2)*dInner>UB` | key 0 |
| Router | mhc/mhc_post/op_kernel/mhc_post.cpp | `mhc_post<usePermanentX>` | 模板实参 0、AIV-only | `ComputeCopyOut` |

#### Device 数据流

##### 函数级调用链

```text
mhc_post<0> -> MhcPostKernel<T,0>::Init -> Process
  -> CopyInHOut -> ComputeCopyOut
    -> CopyInX -> Cast -> Axpy -> CopyOutTile
```

##### 源码等价伪码

```text
伪码：
blockIdx=GetBlockIdx(); if blockIdx>=usedCoreNum return
count=(blockIdx<usedCoreNum-1)?normalCoreProcessNum:tailCoreProcessNum
for itemIdx in [0,count):
  global=blockIdx*normalCoreProcessNum+itemIdx
  bsIdx=global/dOuter; dIdx=global-bsIdx*dOuter
  dNum=(dIdx<dOuter-1)?dInner:dTail; dStart=dIdx*dInner
  CopyInHOut(offset=bsIdx*d+dStart, bytes=dNum*sizeof(T)); Cast -> hOutF32
  for i in [0,n):
    outF32=hOutF32*hPost[bsIdx*n+i]
    for j in [0,n):
      CopyInX(offset=bsIdx*n*d+j*d+dStart, bytes=dNum*sizeof(T))
      Cast -> xF32; outF32+=xF32*hRes[bsIdx*n*n+j*n+i]; Free xTile
    CAST_RINT(outF32,dNum); CopyOut(offset=bsIdx*n*d+i*d+dStart, bytes=dNum*sizeof(T))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `bsIdx*n*d+j*d+dIdx*dInner` | 总 `Q*n*d*sizeof(T)`；每搬 `dNum*sizeof(T)` | T/ND | caller | CopyInX | launch | 单 block，末块 dTail |
| GM `hOut` | `bsIdx*d+dIdx*dInner` | 总 `Q*d*sizeof(T)`；每搬 `dNum*sizeof(T)` | T/ND | caller | CopyInHOut | launch | 每 item 一次 |
| GM `hRes/hPost` | `bsIdx*n*n+j*n+i` / `bsIdx*n+i` | `Q*n*n*4B` / `Q*n*4B` | FP32/ND | caller | GetValue→Axpy/Muls | launch | arch22 对四输入禁用 L2 hint |
| GM `out` | `bsIdx*n*d+i*d+dIdx*dInner` | 总 `Q*n*d*sizeof(T)`；每写 `dNum*sizeof(T)` | T/ND | CopyOutTile | caller | launch | 不写 padding |
| TQue `hOutTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECIN | GM copy | Cast | item | depth=2；DeQue/Free |
| TQue `xTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECIN | 每 j GM copy | Cast | j | depth=2；Axpy 后 Free |
| TQue `outputTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECOUT | CAST_RINT | GM copy | i | depth=2；EnQue/DeQue/Free |
| 三个 FP32 TBuf | 独立 TPipe 区 | 各 `dInner*4B`，合计 `12*dInner B` | FP32/VECCALC | Cast/Muls/Axpy | Vector | 核内 | 单缓冲覆盖复用 |
| 寄存器/标量 | `hPostBase+i` / `hResBase+j*n+i` | 每次一个 FP32 加循环/地址标量 | FP32/int64 | GetValue/控制流 | Muls/Axpy | 指令/循环 | 无 RegTensor |
| workspace/L1/L0 | N/A；workspace 未读 | Host 返回 LibApi workspace；Device 无 L1/L0 | N/A | runtime | framework | executor/stream | 不得作为中间输出 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  hOutQ["hOutTileQueue · depth=2 · 4*dInner B"]:1
  xQ["xTileQueue · depth=2 · 4*dInner B"]:1
  outQ["outputTileQueue · depth=2 · 4*dInner B"]:1
  hOutF32["hOutF32Buf · 4*dInner B"]:1
  xF32["xF32Buf · 4*dInner B"]:1
  outF32["outF32Buf · 4*dInner B"]:1
```

##### 字节账与复用顺序

```text
queue_bytes = 3 * (depth 2 * dInner * sizeof(T=2B)) = 12*dInner B
tbuf_bytes  = 3 * (dInner * sizeof(float)) = 12*dInner B
UB_total(R-001) = 24*dInner B
必要条件：24*dInner <= UB。ComputeTiling 在 dInner>=UB/24 时尝试二分，但 dInner<=256 会先停止；固定仓没有对该 256 元素下限再做 UB 后置检查，因此还要求目标平台实际 UB 能容纳最终布局。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `xTileQueue_` 当前 slot | `CopyInX` 的 `DataCopyPad` | `ComputeCopyOut` 中 `Cast` | 当前 j 的 `FreeTensor` 后下一 j/item `AllocTensor` | `EnQue→DeQue→FreeTensor` | 每个 j 必须消费并释放后才能复用 slot |
| `xF32Buf_` | 当前 j 的 `Cast` | 紧随其后的 `Axpy` | 下一 j 的 `Cast` | 同一函数数据依赖；x queue 已在 Axpy 后释放 | 不跨 j 保留 xF32 |
| `outF32Buf_` | 每个 i 的 `Muls`，随后 n 次 `Axpy` 原地更新 | 最终 `Cast(...CAST_RINT)` | 下一 i 的 `Muls` | Vector 数据依赖；输出先 EnQue 并 CopyOut | 必须完成当前 i 的全部 j 累加和降精度 |
| `hOutF32Buf_` | item 开始的 `Cast` | 所有 i 的 `Muls` | 下一 item | `hOutTile` 在全部 i 完成后 `FreeTensor` | 当前 item 的所有 i 共用同一 hOut tile |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VECIN | `Alloc→DataCopyPad→EnQue→DeQue` | MTE 搬 hOut/x | Cast | TQue slot | `blockCount=1,blockLen=dNum*sizeof(T),stride=0`，pad disabled |
| Vector 顺序 | 数据依赖，无显式 barrier | Cast/Muls/前一 Axpy | 后一 Vector 指令 | 三个 FP32 TBuf | CAST_NONE→FP32，按 j 累加，元素计数 dNum |
| VECOUT→GM | `EnQue→DeQue→DataCopyPad→Free` | CAST_RINT | MTE3 | output slot | round 后只写 dNum |
| 复用/tail | `FreeTensor` 与核/item guard | 当前 j/i | 下次 Alloc | queue slot | 末核 tailCoreProcessNum；末 D tile dTail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `ComputeTiling/PostTiling/GetTilingKey` | 分核、D tile、POD、blockDim 和 key 0 |
| Kernel entry/router | mhc/mhc_post/op_kernel/mhc_post.cpp | `mhc_post<0>` | 解包 POD，实例化流式 AIV body |
| Device body/helper | mhc/mhc_post/op_kernel/mhc_post.h | `MhcPostKernel<T,0>::ComputeCopyOut` | 每 i 初始化 outF32，每 j 搬 X/Cast/Axpy |
| Device body/helper | mhc/mhc_post/op_kernel/mhc_post.h | `CopyInHOut/CopyInX/CopyOutTile` | GM offset、DataCopyExt 字节参数与 queue 生命周期 |

### Route R-002：常驻 X

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.cpp | `aclnnMhcPostGetWorkspaceSize/CheckParam` | 与 R-001 相同非空前置条件 | L0/Tiling |
| Tiling | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `ComputeTiling/GetTilingKey` | `8*(n+2)*dInner<=UB` | key 1 |
| Router | mhc/mhc_post/op_kernel/mhc_post.cpp | `mhc_post<usePermanentX>` | 模板实参 1 | `ComputeCopyOutAllX` |

#### Device 数据流

##### 函数级调用链

```text
mhc_post<1> -> MhcPostKernel<T,1>::Init(xFactor=n) -> Process
  -> CopyInHOut -> CopyInX(n blocks) -> ComputeCopyOutAllX
    -> Cast(all X) -> PipeBarrier/Axpy -> CopyOutTile
```

##### 源码等价伪码

```text
伪码：
按 R-001 的 block/item/offset 公式得到 bsIdx,dIdx,dNum,dStart
CopyInHOut(...)
CopyInX(xOffset=bsIdx*n*d+dStart,
  blockCount=n, blockLen=dNum*sizeof(T), srcStride=(d-dNum)*sizeof(T), dstStride=0)
dNumAlign=(dIdx<dOuter-1)?dNum:dTailAlign
Cast hOut[0:dNum] -> FP32; Cast xLocal[0:n*dNumAlign] -> FP32
for i in [0,n):
  outF32=hOutF32*hPost[bsIdx*n+i]
  for j in [0,n):
    PipeBarrier<PIPE_V>(); outF32+=xF32[j*dNumAlign]*hRes[bsIdx*n*n+j*n+i]
    PipeBarrier<PIPE_V>()
  CAST_RINT 并写 outputOffset=bsIdx*n*d+i*d+dStart 的 dNum 元素
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | 起点 `bsIdx*n*d+dIdx*dInner`，第 j 行 `+j*d` | n blocks×`dNum*sizeof(T)`；src gap `(d-dNum)*sizeof(T)` | T/ND | caller | CopyInX | launch | 一次搬 n 行 |
| GM `hOut/hRes/hPost/out` | 与 R-001 相同 | `Q*d*sizeof(T)` / `Q*n*n*4B` / `Q*n*4B` / `Q*n*d*sizeof(T)` | T/FP32/ND | caller/Kernel | Vector/caller | launch | 输入禁用 L2 hint |
| TQue `xTileQueue_` | TPipe 分配 | `2*n*dInner*sizeof(T)` | T/VECIN | n-block copy | 一次 DeQue 后全部 i/j | item | depth=2，xFactor=n |
| TQue `hOutTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECIN | GM | Cast | item | depth=2 |
| TQue `outputTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECOUT | CAST_RINT | GM | i | depth=2 |
| TBuf `xF32Buf_` | TPipe 分配 | `n*dInner*4B` | FP32/VECCALC | 一次 Cast | 所有 Axpy | item | 第 j 行地址 `j*dNumAlign` |
| TBuf `hOutF32/outF32` | 两个独立区 | 各 `dInner*4B` | FP32/VECCALC | Cast/Muls/Axpy | Vector | item/i | PIPE_V 保护 outF32 链 |
| 寄存器/标量 | GM scalar offset | 一个 FP32/次加 int64 控制量 | FP32/int64 | GetValue | Muls/Axpy | 指令 | 无 RegTensor |
| workspace/L1/L0 | N/A；workspace 未读 | LibApi workspace；无 L1/L0 | N/A | runtime | framework | executor/stream | GM→UB→GM |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  hOutQ["hOutTileQueue · depth=2 · 4*dInner B"]:1
  xQ["xTileQueue · depth=2 · 4*n*dInner B"]:1
  outQ["outputTileQueue · depth=2 · 4*dInner B"]:1
  hOutF32["hOutF32Buf · 4*dInner B"]:1
  xF32["xF32Buf · 4*n*dInner B"]:1
  outF32["outF32Buf · 4*dInner B"]:1
```

##### 字节账与复用顺序

```text
queue_bytes = 4*(n+2)*dInner B
tbuf_bytes  = 4*(n+2)*dInner B
UB_total(R-002) = 8*(n+2)*dInner B
UB_total <= UB <=> Host 选择 key 1 的 fullyBytesPerTileD*dInner <= UB。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `xTileQueue_` n 行 slot | `CopyInX` 的单次多 block `DataCopyPad` | `Cast(xF32, xTile, n*dNumAlign)` | item 末 `FreeTensor` 后下一 item | `EnQue→DeQue→FreeTensor` | tail 行距必须用 `dTailAlign`，且全部 i/j 完成前不得释放 |
| `xF32Buf_` n 行常驻区 | item 开始的一次 `Cast` | 所有 i 的全部 j `Axpy` | 下一 item 的 Cast | 每次 Axpy 前后 `PipeBarrier<PIPE_V>`；item 末释放源 queue | j 地址必须为 `j*dNumAlign`，无效 padding lane 不得进入 dNum |
| `outF32Buf_` | 当前 i 的 `Muls` 和 n 次原地 `Axpy` | `CAST_RINT` | 下一 i | 每次 Axpy 前后 `PIPE_V`；输出 queue 完成交接 | n 次累加顺序与 barrier 不可删改 |
| `hOutF32Buf_` | 一次 `Cast` | 所有 i 的 `Muls` | 下一 item | hOut queue 在所有 i 完成后释放 | 保留整个 item 生命周期 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 多行 X 搬入 | TQue `EnQue→DeQue` | `DataCopyPad(blockCount=n)` | 全 X Cast | x queue slot | 每行有效 dNum；末 tile local 行距 dTailAlign |
| FP32 常驻 | Cast 数据依赖 | `Cast(...,n*dNumAlign)` | 所有 i/j Axpy | xF32 | CAST_NONE；padding 不进入 dNum Axpy |
| Axpy 累加 | 每个 j 前后 `PipeBarrier<PIPE_V>` | Muls/前一 Axpy | 当前/后一 Axpy | outF32/x slice | FP32 按 j 顺序，mask/元素数 dNum |
| 输出/复用 | output TQue，末尾 Free 两输入 | CAST_RINT/DataCopyPad | GM/下一 item | queue/TBuf | 只写真实 dNum；末核规则同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `ComputeTiling/PostTiling/GetTilingKey` | 用 UB 公式选择 key 1 并产生 dTailAlign |
| Kernel entry/router | mhc/mhc_post/op_kernel/mhc_post.cpp | `mhc_post<1>` | 实例化常驻 X AIV body |
| Device body/helper | mhc/mhc_post/op_kernel/mhc_post.h | `MhcPostKernel<T,1>::ComputeCopyOutAllX` | 一次 Cast n 行，按 j*dNumAlign 复用并显式 PIPE_V 同步 |
| Device body/helper | mhc/mhc_post/op_kernel/mhc_post.h | `CopyInX/CopyOutTile` | 多 block 搬运参数与有效输出提交 |

## 算子特定优化

### Trick T-001：UB 充足时一次常驻全部 X 行

#### 触发与源码依据

- 适用 Route：R-002 常驻 X。
- 触发条件：共同 Tiling 完成后 `8*(n+2)*dInner<=UB`，Host 把 `usePermanentX_` 置 1；模板令 `xFactor_=n`，每 item 一次搬入并 Cast n 行 X。
- Golden 来源：[`mhc/mhc_post/op_kernel/mhc_post.h:203-227`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/mhc_post.h#L203-L227)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__aicore__ inline void MhcPostKernel<TEMPLATE_ARGS>::ComputeCopyOutAllX(int64_t bsIdx, int64_t dIdx, int64_t dNum)
{
    int64_t hPostBase = bsIdx * tilingData_->n;
    int64_t hResBase = bsIdx * tilingData_->n * tilingData_->n;
    int64_t dNumAlign = (dIdx < tilingData_->dOuter - 1) ? dNum : tilingData_->dTailAlign;
    LocalTensor<T> hOutTile = hOutTileQueue_.DeQue<T>();
    LocalTensor<T> xTile = xTileQueue_.DeQue<T>();

    LocalTensor<float> hOutF32 = hOutF32Buf_.Get<float>();
    LocalTensor<float> xF32 = xF32Buf_.Get<float>();
    LocalTensor<float> outF32 = outF32Buf_.Get<float>();

    Cast(hOutF32, hOutTile, RoundMode::CAST_NONE, dNum);
    Cast(xF32, xTile, RoundMode::CAST_NONE, tilingData_->n * dNumAlign);

    for (int64_t i = 0; i < tilingData_->n; i++) {
        LocalTensor<T> outputTile = outputTileQueue_.AllocTensor<T>();

        Muls(outF32, hOutF32, hPostGm_.GetValue(hPostBase + i), dNum);
        for (int64_t j = 0; j < tilingData_->n; j++) {
            PipeBarrier<PIPE_V>();
            Axpy(outF32, xF32[j * dNumAlign], hResGm_.GetValue(hResBase + j * tilingData_->n + i), dNum);
            PipeBarrier<PIPE_V>();
        }
```

#### 机制与设计原因

R-001 对每个输出流 i 都为每个 j 重搬并重 Cast 一次 X；R-002 先以 `DataCopyPad(blockCount=n, blockLen=dNum*sizeof(T), srcStride=(d-dNum)*sizeof(T))` 聚合搬入 n 行，再一次 `Cast(..., n*dNumAlign)` 到 FP32，之后所有 i/j 都按 `j*dNumAlign` 复用。它把 X 的 GM 搬运与 Cast 从每个 i 重复降为每 item 一次；代价是 X queue 与 X FP32 TBuf 均扩大 n 倍，因此 Host 用精确 UB 字节账做选择。arch22 在每次原地 `Axpy` 前后插入 `PIPE_V` barrier，保护 outF32 累加链和下一片 X 读取。

#### 迁移边界

- 适用：同一 X tile 被多个输出流 i 反复用于线性组合的纯 Vector 算子。
- 前提：A2/A3 公开规格、FP16/BF16 输入、FP32 累加、`8*(n+2)*dInner<=UB`，并保持 `dTailAlign` 作为末 tile 的 local 行距。
- 失效：UB 条件不成立必须使用 R-001 逐行流式 X；不得缩小 buffer 后仍实例化 key 1。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码可验证的失败边界是 UB 条件为假时选择 R-001，而不是强行常驻。

## 模板与实现对比

| 对比维度 | R-001 流式 X | R-002 常驻 X | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `8*(n+2)*dInner>UB` | `8*(n+2)*dInner<=UB` | `ComputeTiling` 先算共同 tile，再以完整常驻 UB 账选择 bool key |
| 模板与 dtype | `MhcPostKernel<FP16/BF16,0>` | `MhcPostKernel<FP16/BF16,1>` | dtype 共用 body 模板；bool 改变 X 容量和计算函数 |
| 循环与分核 | 每 item→i→j，j 内搬 X | 每 item 先搬 n 行 X，再 i→j 复用 | block/item 映射相同，X 生命周期不同 |
| Queue/TBuf/寄存器 | X queue/TBuf 各 `1*dInner` 容量；标量 hRes/hPost | X queue/TBuf 各 `n*dInner` 容量；标量相同 | key 1 用 UB 换跨 i 复用 |
| UB 总占用与复用 | `24*dInner` B；xF32 逐 j 覆盖 | `8*(n+2)*dInner` B；xF32 整 item 常驻 | Host guard 与 `InitBuffer` 完全对应 |
| 搬运/Vector API | 每 i、j 单 block CopyIn；每 j Cast | 每 item 一次 n-block CopyIn 与一次全 X Cast | 常驻路线减少重复搬运和转换 |
| 同步、精度与 tail | TQue；FP32 按 j 累加；只处理 dNum | TQue+每次 Axpy 前后 `PIPE_V`；末 tile 行距 dTailAlign | 两者均 `CAST_RINT`，但同步图与 local layout 不同 |
| fallback/失败边界 | UB 不足的最终实现；没有后继 fallback | UB 不足转 R-001 | 未知 key、规格失败均无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持：OpDef `ascend910b` | 支持：OpDef `ascend910_93` | 支持；另见 `_950` 卡 |
| Host/Launcher | 与 A3 共用；强制 `n=4,D%128=0,0<D<100000` | 与 A2 共用；核/UB/workspace 运行期查询 | 共用 ACLNN/Tiling，但 soc code 4 绕过上述规格检查 |
| Kernel/资源 | arch22 `mhc_post.cpp/.h`；四输入禁用 L2 hint；R-002 有 PIPE_V barrier | 同 arch22 body | arch35 `mhc_post_apt.cpp/.h`；无 L2 hint 设置且 R-002 无显式 PIPE_V barrier |

## 限制与失败边界

- ACLNN 先做 shape 一致性再判空；合法空输入返回 0 workspace 且不 launch，固定源码没有写 out 内容，不能假设清零。
- A2/A3 Tiling 要求所有维正、`n=4,0<D<100000,D%128=0`，失败为 `GRAPH_FAILED`，无 fallback。
- ACLNN 仅显式拒绝 x/out private format；OpDef 声明全 ND，不能推定其它 storage format 直达 Kernel。
- `B*S`、`Q*dOuter` 与 GM offset 用 64 位，而 DataCopy `blockLen/srcStride` 为 uint32、R-002 `blockCount` 为 uint16；B/S/总地址溢出的显式检查固定仓不可见。
- R-001 的真实 UB 为 `24*dInner` B；Host 在 `dInner<=256` 时停止继续切分且没有再次检查 `24*dInner<=UB`。固定仓未给目标平台 UB 数值下界，迁移到不同 UB 规格时必须补证这一必要条件。
- key 0/1 的 x buffer 容量、搬运次数、局部行距和同步图不同，不得互换。底层 Vector repeat/stride 分解固定仓不可见；源码只提供元素计数、DataCopyExt 参数与显式 barrier。
- LibApi workspace 内部公式和 executor 切片固定仓不可见；Device 不读取 workspace。未进行编译、真机和数值验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 公开说明/平台 | mhc/mhc_post/README.md | `产品支持情况/功能说明` | A2/A3/950 公开支持与公式 |
| API 文档/约束 | mhc/mhc_post/docs/aclnnMhcPost.md | `产品支持情况/约束说明` | 两段式入口、shape/dtype 与 A2/A3 n/D 规格 |
| 入口声明/约束 | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.h | `aclnnMhcPostGetWorkspaceSize/aclnnMhcPost` | 两段式签名 |
| 入口实现 | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.cpp | `CheckParam/aclnnMhcPost*` | 校验、空域、Contiguous、executor、失败传播 |
| L0 Launcher | mhc/mhc_post/op_host/op_api/mhc_post.cpp | `MhcPost/MhcPostAICore` | 输出分配与 launch 参数顺序 |
| OpDef/平台 | mhc/mhc_post/op_host/mhc_post_def.cpp | `ops::MhcPost/OP_ADD` | dtype/format 与平台文件映射 |
| Host/Tiling | mhc/mhc_post/op_host/mhc_post_tiling_base.cpp | `TilingForMhcPost/IMPL_OP_OPTILING` | Tiling registry 接线 |
| Host/Tiling ABI | mhc/mhc_post/op_host/mhc_post_tiling.h | `MhcPostCompileInfo/include` | Host 显式选择 arch35 POD/key 头 |
| Host/Tiling | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `MhcPostTilingBase::*` | 校验、资源、分核、workspace、POD、key |
| Host POD | mhc/mhc_post/op_kernel/arch35/mhc_post_tiling_data.h | `MhcPostTilingData` | Host producer 使用的 12×int64 ABI |
| Host Key | mhc/mhc_post/op_kernel/arch35/mhc_post_tiling_key.h | `ASCENDC_TPL_ARGS_DECL/SEL` | Host 生成 bool 0/1 AIV-only key |
| Device POD | mhc/mhc_post/op_kernel/mhc_post_tiling_data.h | `MhcPostTilingData` | arch22 entry 解包的同布局 ABI |
| Device Key | mhc/mhc_post/op_kernel/mhc_post_tiling_key.h | `ASCENDC_TPL_ARGS_DECL/SEL` | arch22 两个 AIV-only 实例 |
| Kernel entry | mhc/mhc_post/op_kernel/mhc_post.cpp | `mhc_post<usePermanentX>` | arch22 entry |
| Kernel body | mhc/mhc_post/op_kernel/mhc_post.h | `MhcPostKernel` | 地址、Vector、同步、tail |
| 构建接线 | mhc/mhc_post/op_host/CMakeLists.txt | `add_modules_sources/add_ops_compile_options` | 模块与 arch22 编译选项 |
| 类别接线 | mhc/CMakeLists.txt | `add_subdirectory` | mhc_post 进入类别构建 |
| 总仓接线 | CMakeLists.txt | `ASCEND_ALL_COMPUTE_UNIT/add_subdirectory(mhc)` | SoC/arch 和一级目录接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Host arch35 头与 Device 根目录头均为 96 B、12×int64 同布局 POD | 两份字段顺序/位宽/单位必须保持一致；inactive bs 字段不可删；key 不属于 POD | `mhc_post_tiling.h`、两份 `MhcPostTilingData`、`PostTiling` |
| 调用顺序 | ACLNN→Contiguous→L0→ViewCopy→第二段可复用 | 空域位置、executor/workspace 生命周期和 stream 不可改 | `aclnnMhcPostGetWorkspaceSize`、`MhcPostAICore` |
| workspace/内存 | 分核/D tile 与三队列三 TBuf 模型可复用 | xFactor 1/n、dTailAlign 行距和 32 B 对齐不可互换 | `ComputeTiling`、`Init` |
| 同步 | TQue 协议可复用 | arch22 R-002 每次 Axpy 前后 PIPE_V barrier 及 output queue 交接不可删 | `ComputeCopyOutAllX`、`CopyOutTile` |
| 精度与尾块 | FP32 累加、CAST_RINT 可复用 | hRes[j,i]、j 顺序、dTail/dTailAlign 和有效写回不可改 | `ComputeCopyOut/ComputeCopyOutAllX` |
| 失败边界 | ACLNN/Tiling 返回链可复用 | A2/A3 n/D 规格、private format、空域和无 fallback 不可放宽 | `CheckParam/CheckSpecConstraints/CommonOpExecutorRun` |