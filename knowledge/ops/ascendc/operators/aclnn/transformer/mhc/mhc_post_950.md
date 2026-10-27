---
type: Operator
title: "MhcPost 950 ACLNN 算子知识"
description: "说明 aclnnMhcPost 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
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
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/mhc_post_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/arch35/mhc_post.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T15:30:00Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# MhcPost 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnMhcPostGetWorkspaceSize` / `aclnnMhcPost`，公开两段式 ACLNN；固定仓没有版本后缀入口 |
| 计算语义 | `y[q,i,d]=hOut[q,d]*hPost[q,i]+sum(j=0..n-1,x[q,j,d]*hRes[q,j,i])`，`q=T` 或 `q=b*S+s` |
| 输入输出 | TND/BSND、ND；`x/hOut/y` 为 FP16/BF16，`hRes/hPost` 为 FP32；非空 950 Tiling 仅显式要求正维与 shape/dtype 一致 |
| 执行类型 | `AIV_VECTOR` |
| Kernel 路线 | R-001 arch35 流式 X（key 0）；R-002 arch35 常驻 X（key 1） |
| 核心资源 | 五个 GM 张量、三条 depth=2 TQue、三个 FP32 TBuf、标量寄存器；无 workspace/L1/L0 数据路径 |
| 来源闭包 | `source_closed`：ACLNN→Contiguous/L0→950 Tiling/POD/key→`mhc_post_apt` arch35 AIV body→950 注册与构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `Q=T`（TND）或 `Q=B*S`（BSND）。对每个 `q,i,d`，Device 把 `hOut[q,d]` 和 `x[q,j,d]` 从 FP16/BF16 转为 FP32，以 `hPost[q,i]` 做 `Muls` 初始化，再依次以 `hRes[q,j,i]` 做 n 次 `Axpy`。`hRes` 的 `[j,i]` 地址表达转置乘法；末尾 `CAST_RINT` 到输入 dtype，GM 只写 `dNum` 个有效元素。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[T,n,D]` 或 `[B,S,n,D]`，ND | FP16/BF16 | 必选；非空 Tiling 要求所有维严格正；950 不执行 A2/A3 的 n=4/D 对齐检查 |
| `hRes` | `[T,n,n]` 或 `[B,S,n,n]`，ND | FP32 | 必选；按 `[q,j,i]` 标量读取 |
| `hOut` | `[T,D]` 或 `[B,S,D]`，ND | 与 x 相同 | 必选；沿输出流 i 广播 |
| `hPost` | `[T,n]` 或 `[B,S,n]`，ND | FP32 | 必选；每个 i 一个缩放标量 |
| `out` | 与 x 同 shape，ND | 与 x 相同 | 调用方提供；x/out private storage format 在 ACLNN 被拒绝 |
| 属性 | N/A | N/A | 公开接口没有属性 |

### 入口与版本

固定仓只有无版本后缀的公开两段式 ACLNN，A2/A3/950 共享 API 与 L0。OpDef 对 `ascend950` 设置 `opFile.value=mhc_post_apt`，动态 format/rank/shape 标志及 `PrecisionReduceFlag(true)`；因此 950 最新可达 Device 文件是 `mhc_post_apt.cpp` 与 `arch35/mhc_post.h`，不能用 arch22 body 替代。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段先做五个 tensor 非空指针检查，然后检查 x/out 非 private format、dtype 组合、TND/BSND rank 与全部对应维。创建 executor 后，任一输入为空时直接返回 `workspaceSize=0` 与空任务 executor。非空路径依次 `Contiguous(x,hRes,hOut,hPost)`，随后 L0 分配与 x 同 view shape/dtype/format 的内部 output，加入 MhcPost AICore 任务，最后 `ViewCopy` 到调用方 out；任一步返回 nullptr 都立即失败。

### 参数、Workspace 与生命周期

L0 launcher 的 ABI 顺序固定为四输入和一个输出。Tiling 从 950 平台对象读取 AIV 核数、UB 字节数及 LibApi workspace；`workspaceSizes[0]=GetLibApiWorkSpaceSize()`。`MhcPostKernel::Init` 的 workspace 形参未被读取，故算子数据没有 workspace slice；但 executor 可能需要框架 LibApi workspace，调用方仍必须按第一段结果分配，并保留 executor/临时 tensor 所有权直到指定 stream 上第二段完成。

### 启动图与失败传播

链路为 `aclnnMhcPostGetWorkspaceSize→MhcPost→MhcPostAICore→ADD_TO_LAUNCHER_LIST_AICORE`。`Contiguous` 和 `ViewCopy` 属框架边界；950 核心 launch graph 只有一个 Device 节点：

| 节点 | Host 添加/启动 | entry/执行族 | blockDim 与 POD | 输入→输出 | workspace slice |
|---|---|---|---|---|---|
| N1 MhcPost 950 | `l0op::MhcPostAICore:ADD_TO_LAUNCHER_LIST_AICORE` | `mhc_post_apt.cpp:mhc_post<0/1>` / `AIV_VECTOR` | `usedCoreNum`；arch35 `MhcPostTilingData` 96 B；bool key 0/1 | `x,hRes,hOut,hPost→output` | Device 不读 workspace；无算子数据 slice |

```text
Contiguous(x,hRes,hOut,hPost) [框架边界]
  -> N1 MhcPost arch35 AIV_VECTOR [单 Kernel]
  -> ViewCopy(internal output, out) [框架边界]
```

运行期 `TilingForMhcPost→DoTilingImpl→PostTiling` 写 blockDim、POD、workspace 和 key。950 非空维度≤0、shape/dtype 不一致或平台信息缺失均 `GRAPH_FAILED` 并阻断 N1；launcher 失败使 L0 返回 nullptr 并阻断 ViewCopy；第二段失败统一映射为 `ACLNN_ERR_INNER`。N1 无跨 Kernel 边/event；executor 持有框架临时量与任务图，stream 保证本执行图提交顺序，没有后继 fallback Kernel。

## Route 与参数设计

### 分核与资源计算

令 `Q=T` 或 `B*S`、`coreNum=A5 AIV数`、`UB=A5 UB字节数`。初始 `dOuter=1,dInner=D`，基础每 D 元素预算为三条 TQue 的 `depth2*2B` 加三个 FP32 TBuf 的 `4B`，即 `24B`，所以 `maxTileD=UB/24`。当 `Q*dOuter<=coreNum/2` 或 `dInner>=maxTileD`，且 `dInner>256` 时，将 dOuter 加倍并重算 `dInner=D/dOuter`。最终 `dInner=alignUp(...,16)`、`dOuter=ceilDiv(D,dInner)`、`dTail=D-(dOuter-1)*dInner`、`dTailAlign=alignUp(dTail,16)`。`totalCount=Q*dOuter`，经 `used=min(totalCount,coreNum)`、`normal=ceilDiv(totalCount,used)`、`used=ceilDiv(totalCount,normal)`、`tail=totalCount-(used-1)*normal` 收紧核数；末核处理 tail 项。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 arch35 流式 X | key 0 / `MhcPostKernel<T,0>` | 共同 Tiling 后若 `8*(n+2)*dInner>UB`，`usePermanentX_` 保持 0 | 否则 R-002 |
| 2 | R-002 arch35 常驻 X | key 1 / `MhcPostKernel<T,1>` | 若 `8*(n+2)*dInner<=UB`，Host 置 1；950 的 n 仅受正维校验 | 否则 R-001；无第三路线 |

```text
ACLNN dtype/shape/format 成功
  -> 任一输入为空: workspaceSize=0，不 launch
  -> A5 非空: 所有维 > 0；socVer==4 绕过 A2/A3 n/D 规格检查
  -> ComputeTiling
  -> if 8*(n+2)*dInner <= UB: key 1 / R-002
     else: key 0 / R-001
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 arch35 流式 X | key 0 / `USE_PERMANENT_X=0` | FP16/BF16；TND/BSND shape 一致且全维正；950 `socVer==4`；`8*(n+2)*dInner>UB` | 单节点：`mhc_post_apt:mhc_post<0>`，`blockDim=usedCoreNum` | `MhcPostKernel<DTYPE_X,0>::Process→ComputeCopyOut` | 每个 `(q,dTile,i,j)` 逐行搬 X；三条 depth=2 TQue、三个 FP32 TBuf | 末核用 `tailCoreProcessNum`，末 D tile 用 `dTail`；不满足转 R-002 |
| R-002 arch35 常驻 X | key 1 / `USE_PERMANENT_X=1` | 同上公共条件；`8*(n+2)*dInner<=UB` | 单节点：`mhc_post_apt:mhc_post<1>`，`blockDim=usedCoreNum` | `MhcPostKernel<DTYPE_X,1>::Process→ComputeCopyOutAllX` | 每 item 一次多 block 搬 n 行 X 并在 UB 复用；arch35 源码无显式 `PIPE_V` barrier | 末 tile local 行距用 `dTailAlign`、只写 `dTail`；无第三路线 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| D 切片资源上界 | `Q*dOuter<=coreNum/2 \|\| dInner>=UB/24`；只有 `dInner>256` 继续二分 | R-001 三条 depth=2 TQue 与三个 FP32 TBuf合计 `24*dInner` B；并增加可并行 item | 两条 Route | `dInner<=256` 时停止二分，16 元素对齐并重算 tail | `mhc_post_tiling.cpp:MhcPostTilingBase::ComputeTiling` |
| 常驻 X UB 上界 | `8*(n+2)*dInner<=UB` | key 1 的 X queue 与 X FP32 TBuf按 n 倍扩容，总 UB 为 `8*(n+2)*dInner` B | R-002 | 条件为假使用 R-001 | `mhc_post_tiling.cpp:ComputeTiling`；`arch35/mhc_post.h:Init` |
| 950 规格分支 | `socVer==4` 直接返回成功；其余仍检查 `n=4,D` 规格 | 950 公开实现允许任意正 n/D，但 Device 搬运字段仍有固定位宽 | R-001/R-002 | 正维、shape/dtype 失败为 `GRAPH_FAILED` | `mhc_post_tiling.cpp:CheckSpecConstraints` |
| 多 block 搬运窄化 | R-002：`blockCount=(uint16)n`、`blockLen=(uint32)(dNum*sizeof(T))`，src stride 初始化为 int64 字段 | `DataCopyExtParams` ABI 与 arch35 overload | R-002 | 固定 Host 未给额外超界 guard | `arch35/mhc_post.h:CopyInX` |
| AIV-only 编译选择 | arch35 key 头只声明 `ASCENDC_TPL_AIV_ONLY`，bool 只允许 0/1 | 两个可达实例都是纯 Vector | R-001/R-002 | 不存在 AIC/Cube fallback | `arch35/mhc_post_tiling_key.h:ASCENDC_TPL_SEL` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `MhcPostTilingData` | `n` | int64/64 | `sizeof=96, alignof=8, offsetof=0` | 元素 | `PostTiling:n_` | Init/Compute 的循环和容量 | 950 非空时 >0 |
| `MhcPostTilingData` | `d` | int64/64 | `sizeof=96, alignof=8, offsetof=8` | 元素 | `PostTiling:d_` | GM 行跨度 | 950 非空时 >0 |
| `MhcPostTilingData` | `usedCoreNum` | int64/64 | `sizeof=96, alignof=8, offsetof=16` | AIV 核 | `PostTiling:usedCoreNum_` | Init/Process guard | 同 blockDim |
| `MhcPostTilingData` | `normalCoreProcessNum` | int64/64 | `sizeof=96, alignof=8, offsetof=24` | 项/核 | `PostTiling:normalCoreProcessNum_` | 普通核 count/itemStart | ceilDiv |
| `MhcPostTilingData` | `tailCoreProcessNum` | int64/64 | `sizeof=96, alignof=8, offsetof=32` | 项 | `PostTiling:tailCoreProcessNum_` | 末核 count | 差值公式 |
| `MhcPostTilingData` | `bsInner` | int64/64 | `sizeof=96, alignof=8, offsetof=40` | 项 | `PostTiling:1` | arch35 body 不读 | inactive，保留 ABI |
| `MhcPostTilingData` | `bsOuter` | int64/64 | `sizeof=96, alignof=8, offsetof=48` | q 项 | `PostTiling:Q` | arch35 body 不读 | inactive，保留 ABI |
| `MhcPostTilingData` | `bsTail` | int64/64 | `sizeof=96, alignof=8, offsetof=56` | 项 | `PostTiling:1` | arch35 body 不读 | inactive，保留 ABI |
| `MhcPostTilingData` | `dInner` | int64/64 | `sizeof=96, alignof=8, offsetof=64` | 元素/tile | `PostTiling:alignUp(...,16)` | UB 容量、地址、搬运 | 16 元素对齐 |
| `MhcPostTilingData` | `dOuter` | int64/64 | `sizeof=96, alignof=8, offsetof=72` | tile/q | `PostTiling:ceilDiv(D,dInner)` | item 解码 | 至少 1 |
| `MhcPostTilingData` | `dTail` | int64/64 | `sizeof=96, alignof=8, offsetof=80` | 有效元素 | `PostTiling:D-(dOuter-1)*dInner` | 末 tile dNum | `1..dInner` |
| `MhcPostTilingData` | `dTailAlign` | int64/64 | `sizeof=96, alignof=8, offsetof=88` | 对齐元素 | `PostTiling:alignUp(dTail,16)` | R-002 local X 行距 | 16 元素对齐 |

模板量 `usePermanentX` 由 `GET_TPL_TILING_KEY` 传递，不占 POD 字段；arch35 key 头只允许 AIV-only 的布尔 0/1。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 arch35 流式 X | key 0 / `USE_PERMANENT_X=0` | `mhc/mhc_post/op_host/mhc_post_tiling.cpp:ComputeTiling/GetTilingKey` | arch35 12 字段 POD；xFactor=1 | `mhc/mhc_post/op_kernel/mhc_post_apt.cpp:mhc_post<0>` | `mhc/mhc_post/op_kernel/arch35/mhc_post.h:Process/ComputeCopyOut/CopyInX/CopyOutTile` | `mhc/mhc_post/op_host/CMakeLists.txt:ascend950 compile options/add_modules_sources` |
| R-002 arch35 常驻 X | key 1 / `USE_PERMANENT_X=1` | `mhc/mhc_post/op_host/mhc_post_tiling.cpp:ComputeTiling/GetTilingKey` | 同 POD；xFactor=n、dTailAlign | `mhc/mhc_post/op_kernel/mhc_post_apt.cpp:mhc_post<1>` | `mhc/mhc_post/op_kernel/arch35/mhc_post.h:Process/ComputeCopyOutAllX/CopyInX/CopyOutTile` | `mhc/mhc_post/op_host/CMakeLists.txt:ascend950 compile options/add_modules_sources` |

## Kernel 实现

### 公共执行流程

950 entry 声明 `KERNEL_TYPE_AIV_ONLY`，解包 arch35 POD，构造 TPipe 与 `MhcPostKernel<DTYPE_X,key>`。每核按普通/末核 item 数把线性 item 解成 `(bsIdx,dIdx)`；共同路径搬 hOut，FP32 执行 Muls 与 Axpy，CAST_RINT 后通过 VECOUT TQue 回写。hRes/hPost 为 GM FP32 标量访问；950 body 没有 arch22 的 `SetL2CacheHint(DISABLE)`，也没有 L1/L0、workspace、atomic 或跨核交接。

### Route R-001：arch35 流式 X

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.cpp | `aclnnMhcPostGetWorkspaceSize/CheckParam` | 非空、TND/BSND、dtype/format 成功 | L0/Tiling |
| Tiling | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `CheckSpecConstraints/ComputeTiling/GetTilingKey` | `socVer==4` 跳过 n/D 规格；`8*(n+2)*dInner>UB` | key 0 |
| Router | mhc/mhc_post/op_kernel/mhc_post_apt.cpp | `mhc_post<usePermanentX>` | 模板实参 0、AIV-only | arch35 `ComputeCopyOut` |

#### Device 数据流

##### 函数级调用链

```text
mhc_post_apt::mhc_post<0> -> MhcPostKernel<T,0>::Init -> Process
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
  CopyInHOut(offset=bsIdx*d+dStart, blockLen=dNum*sizeof(T)); Cast -> FP32
  for i in [0,n):
    outF32=hOutF32*hPost[bsIdx*n+i]
    for j in [0,n):
      CopyInX(offset=bsIdx*n*d+j*d+dStart, blockLen=dNum*sizeof(T))
      Cast -> xF32; outF32+=xF32*hRes[bsIdx*n*n+j*n+i]; Free xTile
    CAST_RINT 并写 outputOffset=bsIdx*n*d+i*d+dStart 的 dNum 元素
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | `bsIdx*n*d+j*d+dIdx*dInner` | 总 `Q*n*d*sizeof(T)`；每搬 `dNum*sizeof(T)` | T/ND | caller | CopyInX | launch | 每 j 一个 block |
| GM `hOut` | `bsIdx*d+dIdx*dInner` | 总 `Q*d*sizeof(T)`；每搬 dNum | T/ND | caller | CopyInHOut | launch | 每 item 一次 |
| GM `hRes/hPost` | `bsIdx*n*n+j*n+i` / `bsIdx*n+i` | `Q*n*n*4B` / `Q*n*4B` | FP32/ND | caller | GetValue→Axpy/Muls | launch | 950 body 未覆盖 L2 hint |
| GM `out` | `bsIdx*n*d+i*d+dIdx*dInner` | 总 `Q*n*d*sizeof(T)`；每写 dNum | T/ND | CopyOutTile | caller | launch | 只写有效前缀 |
| TQue `hOutTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECIN | GM | Cast | item | depth=2、DeQue/Free |
| TQue `xTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECIN | GM | Cast | j | depth=2、逐 j 复用 |
| TQue `outputTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECOUT | CAST_RINT | GM | i | depth=2、EnQue/DeQue/Free |
| 三个 FP32 TBuf | 三个独立区 | 各 `dInner*4B`，总 `12*dInner B` | FP32/VECCALC | Cast/Muls/Axpy | Vector | 核内 | 单缓冲覆盖 |
| 寄存器/标量 | hPost/hRes offset | 每次一个 FP32 与 int64 控制量 | FP32/int64 | GetValue | Muls/Axpy | 指令 | 无 RegTensor |
| workspace/L1/L0 | N/A；workspace 未读 | LibApi workspace；Device 无 L1/L0 | N/A | runtime | framework | executor/stream | 实际 GM→UB→GM |

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
必要条件：24*dInner <= UB。ComputeTiling 在 dInner>=UB/24 时尝试二分，但 dInner<=256 会先停止；固定仓没有对该下限再做 UB 后置检查，因此仍需目标 A5 平台实际 UB 容纳最终布局。
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `xTileQueue_` 当前 slot | `CopyInX:DataCopyPad` | `ComputeCopyOut:Cast` | 当前 j `FreeTensor` 后下一 j/item | `EnQue→DeQue→FreeTensor` | 每个 j 必须先完成 Cast/Axpy |
| `xF32Buf_` | 当前 j 的 Cast | 紧随其后的 Axpy | 下一 j 的 Cast | 源码数据依赖；x queue 在 Axpy 后释放 | 不跨 j 保留 xF32 |
| `outF32Buf_` | 每个 i 的 Muls，随后 n 次原地 Axpy | 最终 `CAST_RINT` | 下一 i 的 Muls | 源码顺序/数据依赖，arch35 无显式 `PIPE_V` | 当前 i 全部 j 累加完成后才降精度 |
| `hOutF32Buf_` | item 开始的一次 Cast | 所有 i 的 Muls | 下一 item | hOut queue 在所有 i 完成后释放 | 保留整个 item 生命周期 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VECIN | `Alloc→DataCopyPad→EnQue→DeQue` | MTE 搬 hOut/x | Cast | queue slot | blockCount=1，blockLen=dNum*sizeof(T)，pad disabled |
| Vector | 源码顺序/数据依赖；无显式 PIPE_V | Cast/Muls/Axpy | 后继 Vector | FP32 TBuf | CAST_NONE→FP32，按 j 累加，元素计数 dNum |
| VECOUT→GM | `EnQue→DeQue→DataCopyPad→Free` | CAST_RINT | MTE3 | output slot | round 后仅写 dNum |
| tail/复用 | 核/item guard 与 FreeTensor | 当前 j/i | 下一次 Alloc | queue/TBuf | 末核 tailCoreProcessNum，末 D tile dTail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `CheckSpecConstraints/ComputeTiling/PostTiling/GetTilingKey` | 950 规格分支、分核、POD、blockDim、key 0 |
| Kernel entry/router | mhc/mhc_post/op_kernel/mhc_post_apt.cpp | `mhc_post<0>` | 解包 POD 并实例化 arch35 流式 body |
| Device body/helper | mhc/mhc_post/op_kernel/arch35/mhc_post.h | `MhcPostKernel<T,0>::ComputeCopyOut` | 每 i 初始化 FP32，每 j 逐次 CopyIn/Cast/Axpy |
| Device body/helper | mhc/mhc_post/op_kernel/arch35/mhc_post.h | `CopyInHOut/CopyInX/CopyOutTile` | GM offset、DataCopyExt 参数、queue 生命周期 |

### Route R-002：arch35 常驻 X

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.cpp | `aclnnMhcPostGetWorkspaceSize/CheckParam` | 与 R-001 相同非空前置条件 | L0/Tiling |
| Tiling | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `CheckSpecConstraints/ComputeTiling/GetTilingKey` | 950 正维；`8*(n+2)*dInner<=UB` | key 1 |
| Router | mhc/mhc_post/op_kernel/mhc_post_apt.cpp | `mhc_post<usePermanentX>` | 模板实参 1 | arch35 `ComputeCopyOutAllX` |

#### Device 数据流

##### 函数级调用链

```text
mhc_post_apt::mhc_post<1> -> MhcPostKernel<T,1>::Init(xFactor=n) -> Process
  -> CopyInHOut -> CopyInX(n blocks) -> ComputeCopyOutAllX
    -> Cast(all X) -> Muls/Axpy -> CopyOutTile
```

##### 源码等价伪码

```text
伪码：
按 R-001 的 block/item/offset 公式得到 bsIdx,dIdx,dNum,dStart
CopyInX 起点=bsIdx*n*d+dStart，参数：
  blockCount=(uint16)n; blockLen=(uint32)(dNum*sizeof(T))
  srcStride=(int64)((d-dNum)*sizeof(T)); dstStride=0
dNumAlign=(dIdx<dOuter-1)?dNum:dTailAlign
Cast hOut[0:dNum] 与连续 xLocal[0:n*dNumAlign] 到 FP32
for i in [0,n):
  outF32=hOutF32*hPost[bsIdx*n+i]
  for j in [0,n):
    outF32+=xF32[j*dNumAlign:j*dNumAlign+dNum]*hRes[bsIdx*n*n+j*n+i]
  CAST_RINT 并写 outputOffset=bsIdx*n*d+i*d+dStart 的 dNum 元素
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x` | 起点 `bsIdx*n*d+dIdx*dInner`，第 j 行 `+j*d` | n blocks×`dNum*sizeof(T)`，src gap `(d-dNum)*sizeof(T)` | T/ND | caller | CopyInX | launch | 一次搬 n 行 |
| GM `hOut/hRes/hPost/out` | 与 R-001 相同 | `Q*d*sizeof(T)` / `Q*n*n*4B` / `Q*n*4B` / `Q*n*d*sizeof(T)` | T/FP32/ND | caller/Kernel | Vector/caller | launch | 950 body未设 L2 hint |
| TQue `xTileQueue_` | TPipe 分配 | `2*n*dInner*sizeof(T)` | T/VECIN | 多 block GM copy | 全部 i/j | item | depth=2，xFactor=n |
| TQue `hOutTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECIN | GM | Cast | item | depth=2 |
| TQue `outputTileQueue_` | TPipe 分配 | `2*dInner*sizeof(T)` | T/VECOUT | CAST_RINT | GM | i | depth=2 |
| TBuf `xF32Buf_` | TPipe 分配 | `n*dInner*4B` | FP32/VECCALC | 一次 Cast | 所有 i/j Axpy | item | 第 j 行 `j*dNumAlign` |
| TBuf `hOutF32/outF32` | 两个独立区 | 各 `dInner*4B` | FP32/VECCALC | Cast/Muls/Axpy | Vector | item/i | 源码无显式 PIPE_V barrier |
| 寄存器/标量 | hRes/hPost offset | 一个 FP32/次及 int64 控制量 | FP32/int64 | GetValue | Muls/Axpy | 指令 | 无 RegTensor |
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
| `xTileQueue_` n 行 slot | 单次多 block `DataCopyPad` | `Cast(...,n*dNumAlign)` | item 末 Free 后下一 item | `EnQue→DeQue→FreeTensor` | tail 行距用 `dTailAlign`，全部 i/j 完成前不得释放 |
| `xF32Buf_` n 行常驻区 | item 开始的一次 Cast | 所有 i/j 的 Axpy | 下一 item Cast | 源码顺序/数据依赖；arch35 无显式 barrier | j 地址 `j*dNumAlign`，只处理 dNum 有效 lane |
| `outF32Buf_` | 当前 i 的 Muls 与 n 次原地 Axpy | `CAST_RINT` | 下一 i | 源码顺序/数据依赖 | 必须保持 j 累加顺序；不能用 arch22 barrier 事实补写 |
| `hOutF32Buf_` | 一次 Cast | 所有 i 的 Muls | 下一 item | hOut queue 最后释放 | 保留整个 item 生命周期 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 多行 X 搬入 | TQue `EnQue→DeQue` | `DataCopyPad(blockCount=n)` | 全 X Cast | x queue slot | 每行 dNum 有效；末 tile local 行距 dTailAlign |
| FP32 常驻 | Cast 数据依赖 | `Cast(...,n*dNumAlign)` | 所有 Axpy | xF32 | padding 只占 local stride，不进入 dNum 计算 |
| Axpy 累加 | 源码顺序/数据依赖，无显式 barrier | Muls/前一 Axpy | 后一 Axpy | outF32/x slice | FP32 按 j 顺序，元素计数 dNum |
| 输出/复用 | output TQue，末尾 Free 两输入 | CAST_RINT/DataCopyPad | GM/下一 item | queue/TBuf | 只写 dNum；末核同 R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `CheckSpecConstraints/ComputeTiling/PostTiling/GetTilingKey` | 950 分支、UB 条件、dTailAlign、key 1 |
| Kernel entry/router | mhc/mhc_post/op_kernel/mhc_post_apt.cpp | `mhc_post<1>` | 实例化 arch35 常驻 X body |
| Device body/helper | mhc/mhc_post/op_kernel/arch35/mhc_post.h | `MhcPostKernel<T,1>::ComputeCopyOutAllX` | 一次 Cast n 行并按 j*dNumAlign 复用 |
| Device body/helper | mhc/mhc_post/op_kernel/arch35/mhc_post.h | `CopyInX/CopyOutTile` | arch35 srcStride 类型、多 block 搬运与有效输出 |

## 算子特定优化

### Trick T-001：950 上用 UB 容量门控 X 全行常驻

#### 触发与源码依据

- 适用 Route：R-002 arch35 常驻 X。
- 触发条件：950 非空 shape 全维正，共同 Tiling 后 `8*(n+2)*dInner<=UB`；Host 置 `usePermanentX_=1`，模板使 X queue/TBuf 容量乘 n。
- Golden 来源：[`mhc/mhc_post/op_kernel/arch35/mhc_post.h:199-220`](../../../../../../../cann-ops-raw/ascendc/ops-transformer/be33d95bb755cedec3108f70805c1874d1792540/mhc/mhc_post/op_kernel/arch35/mhc_post.h#L199-L220)

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
            Axpy(outF32, xF32[j * dNumAlign], hResGm_.GetValue(hResBase + j * tilingData_->n + i), dNum);
        }
```

#### 机制与设计原因

R-002 的 `CopyInX` 以 `blockCount=(uint16)n` 一次搬入所有 X 行，`Cast` 的元素数为 `n*dNumAlign`，之后每个输出流 i 都复用 `xF32[j*dNumAlign]`。相对 R-001 在每个 i/j 内重复 CopyIn+Cast，这一机制减少 X 的重复 GM 搬运和重复升精度转换；代价是 X queue 与 X TBuf 同时扩大 n 倍，所以 Host 以 `8*(n+2)*dInner` 的完整 UB 账作为开关。950 body 的该循环没有显式 `PIPE_V` barrier，知识迁移必须保留这一平台事实，不能从 arch22 补写同步。

#### 迁移边界

- 适用：950 上同一 X tile 被多个 i 重复线性组合的 FP16/BF16→FP32 Vector 路线。
- 前提：全维正、shape/dtype 一致、`8*(n+2)*dInner<=UB`，且 `n` 与 DMA 字段窄化仍在实际可表达范围内。
- 失效：UB 条件不成立转 R-001；若无法证明 `uint16 blockCount`、`uint32 blockLen` 等窄化安全，不得外推到超大 n/D。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码可验证的边界是 UB 不足时选择 R-001，并且不得把 A2/A3 的显式 `PIPE_V` barrier 当作 950 源码事实。

## 模板与实现对比

| 对比维度 | R-001 arch35 流式 X | R-002 arch35 常驻 X | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `8*(n+2)*dInner>UB` | `8*(n+2)*dInner<=UB` | 共同 tile 后以完整常驻 UB 账选 key |
| 模板与 dtype | `MhcPostKernel<FP16/BF16,0>` | `MhcPostKernel<FP16/BF16,1>` | bool 模板量改变 X 容量和实际 helper |
| 循环与分核 | item→i→j，j 内逐行搬 X | item 先搬 n 行，再 i→j 复用 | block/item 映射相同，X 生命周期不同 |
| Queue/TBuf/寄存器 | X queue/TBuf 各一行容量 | X queue/TBuf 各 n 行容量 | 用 UB 换跨 i 复用；hRes/hPost 都是 GM 标量 |
| UB 总占用与复用 | `24*dInner` B；xF32 逐 j 覆盖 | `8*(n+2)*dInner` B；xF32 整 item 常驻 | Host guard 与 `InitBuffer` 对应 |
| 搬运/Vector API | 每 i/j 单 block CopyIn、Cast、Axpy | 每 item n-block CopyIn、一次全 X Cast、复用 Axpy | 常驻路线减少重复搬运/转换 |
| 同步、精度与 tail | TQue，源码顺序/依赖；dNum 有效域 | TQue，源码无显式 `PIPE_V`；末 tile local 行距 dTailAlign | 两者 FP32 累加、`CAST_RINT`，但 local layout 不同 |
| fallback/失败边界 | UB 不足最终路线 | UB 不足转 R-001 | 规格或窄化风险无额外 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持：OpDef `ascend950` 与 arch35 body |
| Host/Launcher | 强制 `n=4,D%128=0,0<D<100000` | 同 A2 | `socVer==4` 绕过该规格检查；其余 ACLNN/Tiling 共用 |
| Kernel/资源 | arch22 body 禁用四输入 L2 hint；R-002 显式 PIPE_V barrier | 同 A2 | arch35 body不设置 L2 hint；R-002 无显式 PIPE_V barrier，srcStride 初始化为 int64 |

## 限制与失败边界

- ACLNN 在空域短路前先做 shape/dtype/format 一致性；合法空输入返回 0 workspace 且不 launch，固定源码不写 out 内容。
- 950 非空路径只显式要求所有维正及 shape/dtype 一致；源码通过 `socVer==4` 直接绕过 A2/A3 的 `n=4,D<100000,D%128=0`，不能把文档中的旧限制移植到本卡。
- 这也暴露窄化边界：R-002 `blockCount` 强转 uint16，所有路线 `blockLen` 强转 uint32；n、D 或乘积超界的额外 Host 检查固定仓不可见。GM offset、`B*S`、`Q*dOuter` 的溢出检查同样不可见。
- R-001 的真实 UB 是 `24*dInner` B；Host 在 `dInner<=256` 时停止继续切分，未再检查 `24*dInner<=UB`。固定仓没有给出 950 UB 数值下界，移植资源模型时必须单独证明该必要条件。
- ACLNN 只显式检查 x/out private storage format，OpDef 声明全 ND；不能据动态 format 标志推定任意 layout 与本 Kernel 地址公式等价。
- key 0/1 的 X queue/TBuf 容量、搬运网络、dTailAlign 行距不同。950 R-002 没有显式 PIPE_V barrier；其依赖保障超出 TQue 交接与源码顺序的底层实现固定仓不可见，不得凭 arch22 补写。
- Vector repeat/stride/mask 的底层展开及 LibApi workspace 公式固定仓不可见；Device workspace 参数未使用。未完成编译、真机或数值验证。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 公开说明/平台 | mhc/mhc_post/README.md | `产品支持情况/功能说明` | 950 与 A2/A3 公开支持、公式 |
| API 文档/约束 | mhc/mhc_post/docs/aclnnMhcPost.md | `产品支持情况/约束说明` | 两段式入口、shape/dtype，n/D 规格仅标 A2/A3 |
| 入口声明/约束 | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.h | `aclnnMhcPostGetWorkspaceSize/aclnnMhcPost` | 公开两段式签名 |
| 入口实现 | mhc/mhc_post/op_host/op_api/aclnn_mhc_post.cpp | `CheckParam/aclnnMhcPost*` | 校验、空域、Contiguous、executor、失败传播 |
| L0 Launcher | mhc/mhc_post/op_host/op_api/mhc_post.cpp | `MhcPost/MhcPostAICore` | 输出分配与参数顺序 |
| OpDef/平台 | mhc/mhc_post/op_host/mhc_post_def.cpp | `ops::MhcPost/OP_ADD` | ascend950→mhc_post_apt 映射 |
| Host/Tiling | mhc/mhc_post/op_host/mhc_post_tiling_base.cpp | `TilingForMhcPost/IMPL_OP_OPTILING` | Tiling registry 接线 |
| Host/Tiling ABI | mhc/mhc_post/op_host/mhc_post_tiling.h | `MhcPostCompileInfo/include` | 950 Host 选择 arch35 POD/key 头 |
| Host/Tiling | mhc/mhc_post/op_host/mhc_post_tiling.cpp | `MhcPostTilingBase::*` | 950 校验分支、分核、workspace、POD、key |
| POD | mhc/mhc_post/op_kernel/arch35/mhc_post_tiling_data.h | `MhcPostTilingData` | arch35 ABI |
| Key | mhc/mhc_post/op_kernel/arch35/mhc_post_tiling_key.h | `ASCENDC_TPL_ARGS_DECL/SEL` | 两个 arch35 AIV-only key |
| Kernel entry | mhc/mhc_post/op_kernel/mhc_post_apt.cpp | `mhc_post<usePermanentX>` | 950 entry |
| Kernel body | mhc/mhc_post/op_kernel/arch35/mhc_post.h | `MhcPostKernel` | 地址、搬运、Vector、tail |
| 构建接线 | mhc/mhc_post/op_host/CMakeLists.txt | `ascend950/add_modules_sources` | 950 编译选项与模块接入 |
| 类别接线 | mhc/CMakeLists.txt | `add_subdirectory` | mhc_post 进入类别构建 |
| 总仓接线 | CMakeLists.txt | `ASCEND_ALL_COMPUTE_UNIT/add_subdirectory(mhc)` | ascend950→arch35 与一级目录接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | arch35 96 B、12×int64 POD | 字段顺序/单位/inactive 字段不可改；模板 key 不在 POD | `MhcPostTilingData`、`PostTiling` |
| 调用顺序 | 共用 ACLNN→Contiguous→L0→ViewCopy→第二段 | 空域位置、executor/workspace/stream 生命周期不可改 | `aclnnMhcPostGetWorkspaceSize`、`MhcPostAICore` |
| workspace/内存 | 分核、D tile、三队列三 TBuf 可复用 | key 0/1 的 xFactor、容量、dTailAlign 与多行搬运不可互换 | `ComputeTiling`、`Init/CopyInX` |
| 同步 | TQue EnQue/DeQue/Free 协议可复用 | 950 无显式 PIPE_V 的事实不得用 arch22 barrier 改写；output 交接不可删 | `ComputeCopyOutAllX/CopyOutTile` |
| 精度与尾块 | FP32 Muls/Axpy 与 CAST_RINT 可复用 | hRes[j,i]、j 顺序、dTailAlign 和 dNum 有效写不可改 | `ComputeCopyOut/ComputeCopyOutAllX` |
| 失败边界 | ACLNN/Tiling 失败传播可复用 | 950 只正维的真实边界、窄化缺口、空输入不 launch 和无 fallback 不可伪造补齐 | `CheckParam/CheckSpecConstraints/CommonOpExecutorRun` |