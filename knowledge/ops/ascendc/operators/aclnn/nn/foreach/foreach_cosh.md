---
type: Operator
title: "ForeachCosh A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachCosh 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 AIV_VECTOR 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/docs/aclnnForeachCosh.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/foreach_cosh_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/arch22/foreach_cosh_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/foreach_cosh.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/config/ascend910b/foreach_cosh_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cosh/op_host/config/ascend910_93/foreach_cosh_binary.json
    role: implementation
created_at: "2026-08-25T15:31:16Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachCosh A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnForeachCoshGetWorkspaceSize` → `aclnnForeachCosh`；固定仓未保存自动生成的 inner bridge。 |
| 计算语义 | 对 TensorList 中每个张量的每个输入元素执行 `y[t][i] = cosh(x[t][i])`。 |
| 输入输出 | `x/out` 均为 ND TensorList；元素 dtype 同为 FLOAT16、FLOAT32 或 BFLOAT16；输入允许非连续并由 OpDef `AutoContiguous` 处理，输出必须连续。 |
| 执行类型 | 单 Kernel、单 AIV_VECTOR 节点；FP16/FP32 调用对应类型的 AscendC `Cosh`，BF16 先升到 FP32。 |
| Kernel 路线 | R-001 FP16 AIV；R-002 FP32 AIV；R-003 BF16→FP32 AIV。 |
| 核心资源 | TensorList GM 指针表、每核输入/输出双缓冲 `TQue`、BF16 专用 FP32 `TQue`；算子用户 workspace 不参与 Device 计算。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档 → ACLNN 自动生成边界 → `ForeachCosh` OpDef/arch22 Tiling → `ForeachCommonTilingData` → `foreach_cosh` router → `ForeachTriangle`/`KernelForeachUnary` → A2/A3 build/config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 TensorList 长度为 `L`，第 `t` 个输入的 storage element count 为 `N_t`。对 `0 <= t < L`、`0 <= i < N_t`：

```text
out[t][i] = cosh(x[t][i])
```

Host 允许 `out[t]` 的 storage element count 大于 `x[t]`；Kernel 只写前 `N_t` 个元素，额外输出区间未由本算子写入。公开文档声明默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND，rank 0–8 | BFLOAT16、FLOAT16、FLOAT32，列表内一致 | 支持空 Tensor；非连续输入由 `AutoContiguous` 框架边界转连续。A2/A3 公共 Tiling 最多枚举 50 个张量。 |
| `out` | TensorList；每项 ND，`numel(out[t]) >= numel(x[t])` | 与 `x` 及列表内其它项一致 | 输出不支持非连续 Tensor；Kernel 只提交与输入 `numel` 相同的前缀。 |
| `workspaceSize` | 标量输出 | `uint64_t` | 两段式第一段返回。Tiling 写系统 workspace 槽 32 B；Kernel 明确把用户 workspace 置空且不读取。自动生成 ACLNN 层如何呈现该系统槽在固定仓不可见。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段创建、第二段消费；内部对象布局和释放协议属于自动生成框架边界。 |
| `workspace/stream` | Device 地址/运行时 stream | `void*` / `aclrtStream` | 第二段按同一 stream 启动单 Kernel；没有 Kernel 间边。 |

### 入口与版本

固定版本只有公开 `aclnnForeachCosh` 版本族，没有更高版本号入口。文档给出 `GetWorkspaceSize(x, out, ...)` 与执行接口原型，OpDef 的 `ACLNNTYPE aclnn` 构建接线生成 inner bridge；该生成体不在固定仓内，作为明确框架边界，不臆造其内部函数名。README 与 ACLNN 文档均明确支持 Atlas A2、A3，`ForeachCosh` OpDef 注册 `ascend910b`、`ascend910_93`，Host 与 Kernel CMake 同时接线这两个 compute unit，无公开支持冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段按公开文档检查空指针、dtype 集合、`x/out` dtype 一致性、列表内 dtype 一致性、shape 约束和 rank≤8；自动生成的检查代码固定仓不可见。
2. OpDef 对动态输入和输出注册 ND、FP16/FP32/BF16，并为两端设置 `AutoContiguous`；公开约束仍要求用户输出连续。
3. arch22 `Tiling4ForeachCoshTiling` 构造 `ForeachCommonTiling`，以 `FOREACH_COSH_OP_CODE=13` 初始化；`Init` 从输入 0 顺序读取至首个空实例，记录每项 storage `numel`、总元素数、dtype、dtype 字节数与每 32 B 元素数。
4. `CheckOpInfo` 检查动态列表数量、每项 rank、输出 dtype/容量和列表内 dtype；失败返回 `GRAPH_FAILED`，不启动 Kernel。
5. TilingKey 仅由 dtype 决定：FP16→1、FP32→2、BF16→4。其它 dtype 得到 0，但公开 OpDef/ACLNN 已将其拒绝。

### 参数、Workspace 与生命周期

令 `E=ΣN_t`，`B=32/sizeof(T)`：

```text
blockCount32 = ceil(E / B)
blockDim = min(platformAivCoreNum, blockCount32, 50), E=0 时取 1
baseBlocks = blockCount32 / blockDim
remainderBlocks = blockCount32 % blockDim
core c elements = (baseBlocks + (c < remainderBlocks ? 1 : 0)) * B
```

`AssignDataToEachCore` 沿 TensorList 逻辑拼接域推进 cursor，将每核区间编码成起止 tensor index 与闭区间 offset。分界按 32 B block 计算，但最后一个张量以真实末元素收尾。`ForeachCommonTilingData` 由 Tiling context 持有并在 launch 时只读；Kernel 期间有效。

Tiling 写 `workspaces[0]=32`，但 Kernel entry 将 `userWS=nullptr`，`ForeachTriangle::Init` 的 workspace 参数没有被读取；因此没有算子拥有的 workspace slice 或跨 Kernel生命周期。ACLNN executor/workspace 的框架所有权在固定仓不可见。

### 启动图与失败传播

```text
aclnnForeachCoshGetWorkspaceSize  [自动生成框架边界]
  -> ForeachCosh OpDef / InferShape / Tiling4ForeachCoshTiling
  -> ForeachCommonTiling::RunBigKernelTiling
  -> executor 保存单节点 launch 参数

aclnnForeachCosh(workspace, workspaceSize, executor, stream)
  -> node K0: foreach_cosh<<<blockDim, stream>>>(xList, yList, workspace, tiling)
       AIV_VECTOR; POD=ForeachCommonTilingData; 无用户 workspace 读写
```

| 节点/边 | producer | consumer | 介质与范围 | ordering / owner / failure |
|---|---|---|---|---|
| K0 | ACLNN executor | `foreach_cosh` | `x/y` TensorList GM；POD；`blockDim` | 同一调用 stream；executor/运行时拥有 launch 参数。Tiling 或 launch 失败即返回，后续没有 Device 节点。 |
| 跨节点边 | N/A | N/A | 单 Kernel，无中间 workspace/GM edge | N/A。 |

## Route 与参数设计

### 分核与资源计算

对 FP16/BF16，`B=16` 元素/32 B；FP32 为 `B=8`。各核拼接域由 POD 的四个 start/end 数组恢复；核内再以 `maxDataCount` 分 tile。

令 `P=tilingData.GetDataSize()`，`UB` 为平台 UB bytes，`U=inputsTensorUbSize`：

```text
FP16: reserveMath=6*2*1024*8=98304 B
      U=floor32((UB-P-reserveMath)/4)
FP32: reserveMath=2*4*1024*8=65536 B
      U=floor32((UB-P-reserveMath)/4)
BF16: reserveMath=98304 B
      U=floor64(((UB-P-reserveMath)/10)/4)
```

`P` 由注册宏的 `GetDataSize()` 动态取得；固定仓没有该宏定义，不能把推导的自然布局值冒充序列化 ABI。Host 未在减法前显式检查 `UB > P+reserveMath`，依赖受支持平台资源满足该前提。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 AIV | key 1；`ForeachTriangle<half,half,Cosh<half,false>>` | `dataType == DT_FLOAT16` | 检查 key 2。 |
| 2 | R-002 FP32 AIV | key 2；`ForeachTriangle<float,float,Cosh<float,false>>` | `dataType == DT_FLOAT` | 检查 key 4。 |
| 3 | R-003 BF16→FP32 AIV | key 4；`ForeachTriangle<bfloat16_t,float,Cosh<float,false>>` | `dataType == DT_BF16` 且非 `__NPU_ARCH__` 3003/3113 编译 guard；A2/A3 build 命中 | 未知 key/被 guard 排除：entry 无 body、无输出提交。 |

```text
Host: dtype FP16 ? key=1 : dtype FP32 ? key=2 : dtype BF16 ? key=4 : GRAPH_FAILED/框架拒绝
Device: key==1 ? R-001 : key==2 ? R-002 : key==4 && BF16 guard ? R-003 : no-op
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 AIV | 1 | FP16；列表≤50；rank≤8；输出容量足够；A2/A3 | K0 单 AIV_VECTOR | `ForeachTriangle<half,half,Cosh<half,false>>::Process` | 输入/输出 `TQue` 各 depth2×U；V pipe barrier 包围 `Cosh` | 尾 tile 用 byte-count `DataCopyPad`；未知 key no-op。 |
| R-002 FP32 AIV | 2 | FP32；其余同上 | K0 单 AIV_VECTOR | `ForeachTriangle<float,float,Cosh<float,false>>::Process` | 同为 4U 显式 Queue，但 reserveMath 为 64 KiB | 同上。 |
| R-003 BF16→FP32 AIV | 4 | BF16；其余同上；编译 guard 可达 | K0 单 AIV_VECTOR | BF16 special `InnerComputer` 调 `Cosh<float,false>` | 输入/输出各 depth2×9U，FP32 queue 1×2U；Cast/Cosh/Cast 间 V barrier | 外 tile 与内 cast tile 双层 tail；最终 `CAST_RINT`。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 列表上限 | `Init` 只枚举 `i<50` | POD 每个 tensor 数组容量 50 | 全部 | ACLNN/Tiling 应拒绝超限；公共实现本身不会编码第 51 项 | `foreach_tiling_def.h::MAX_TENSOR_CONT`; `ForeachCommonTiling::Init` |
| 核数上限 | `min(GetNeedCoreNum, 50)` | POD 每核数组容量 50 | 全部 | 使用最多 50 AIV 核 | `RunBigKernelTiling` |
| 32 B 分核 | `B=32/sizeof(T)` | 非尾块 `DataCopy` 走对齐元素数 | 全部 | 末端真实长度由 end offset/tail copy 修正 | `GetNeedCoreNum`; `AssignDataToEachCore` |
| UB reserve | 上述 `U` 公式 | 双缓冲与 `Cosh` 数学临时量必须容纳 | 全部 | 固定源码无 fallback；算术下溢没有显式保护 | `DivideUbMemory4` |
| BF16 编译 guard | `__CCE_AICORE__>=220` 且 arch 非3003/3113 | 需要 BF16 与 FP32 Cast 路径 | R-003 | guard 不成立则 key4 entry 无分支 | `foreach_cosh.cpp`; `foreach_triangle.h` |
| 输出前缀 | `numel(out)>=numel(x)` | Kernel 地址与循环均以输入计数 | 全部 | 容量更小 Tiling 失败；更大部分不写 | `CheckOutputShapeAndDtype`; `KernelForeachUnary::Process` |
| 空域 | `E==0 => blockDim=1` | 保持合法 launch；每项 dataCount=0 时循环次数0 | 全部 | 无有效输出写入 | `GetNeedCoreNum`; `SingleTensorProcess` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | `BEGIN_TILING_DATA_DEF` 宏布局；`sizeof/alignof/offsetof` 固定仓不可见 | bytes/单个 U | `DivideUbMemory4` → `FillTilingData::set_inputsTensorUbSize(U)` | `KernelForeachBase::ParseTilingData`; 决定 Queue capacity/tile | 必须≥32 B 对齐；BF16 为64 B对齐。 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 同上 | elements/每 tensor | `Init`: `storageShape.GetShapeSize()` | `Process`: tensor end 与 dataCount | 未使用槽初始化为0。 |
| 同上 | `tensorStartList[50]` | `uint16_t[50]` | 同上 | tensor index/每 core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | blockIdx<blockDim≤50。 |
| 同上 | `tensorEndList[50]` | `uint16_t[50]` | 同上 | tensor index/每 core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | 闭区间。 |
| 同上 | `tensorStartOffsetList[50]` | `int64_t[50]` | 同上 | elements in start tensor | `AssignDataToEachCore` | `Process` 设置首 tensor cursor | 闭区间起点。 |
| 同上 | `tensorEndOffsetList[50]` | `int64_t[50]` | 同上 | elements in end tensor | `AssignDataToEachCore` | `Process` 设置末 tensor cursor | 闭区间终点；真实尾长度。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | `GetTilingKeyByDtypeOnly`→1 | `arch22::Tiling4ForeachCoshTiling`; `ForeachCommonTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_cosh.cpp::foreach_cosh`, key1 | `ForeachTriangle`; `KernelForeachUnary`; `KernelForeachBase`; AscendC `Cosh<half,false>` 边界 | `op_host/CMakeLists.txt`; `op_kernel/CMakeLists.txt`; `ascend910b/ascend910_93 binary.json` |
| R-002 FP32 | 同上→2 | 同上 | 同上 | 同上，key2 | 同上，`Cosh<float,false>` | 同上 |
| R-003 BF16 | 同上→4 | 同上 | 同上 | 同上，guarded key4 | BF16 `InnerComputer`; `Cast`; `Cosh<float,false>` | 同上 |

## Kernel 实现

### 公共执行流程

Kernel 从 TensorList 首地址读一个 byte offset，经 `offset>>3` 定位 uint64 地址表，再以 tensor index 取真实 GM 指针。每个 AIV core 从 POD 恢复一个横跨若干 Tensor 的闭区间；对每个相交 Tensor 建立 `GM base + cursorStart`，按 Queue 容量切 tile，执行 `CopyIn → Compute → CopyOut`。单节点直接写最终 `out`，无 workspace 边。

### Route R-001：FP16 AIV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key1 |
| Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling` | 非 copy op，dtype-only key | blockDim/POD 完成 |
| Router | `foreach/foreach_cosh/op_kernel/foreach_cosh.cpp` | `foreach_cosh` | `TILING_KEY_IS(1)` | `ForeachTriangle<half,half,Cosh<half,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_cosh(xList,yList,workspace,tiling)
  -> ForeachTriangle<half,half,Cosh<half,false>>::Init
     -> KernelForeachBase<half>::ParseTilingData
     -> InitBuffer(dataQueue,2,U); InitBuffer(outQueue,2,U)
  -> KernelForeachUnary::Process
     -> per tensor/core slice -> SingleTensorProcess
        -> CopyIn -> ForeachTriangle::Compute -> Cosh<half,false> -> CopyOut
```

##### 源码等价伪码

```text
core = GetBlockIdx()
[t0,o0,t1,o1] = POD per-core closed interval
for t in t0..t1:
  begin = (t==t0 ? o0 : 0); end = (t==t1 ? o1 : N[t]-1)
  bind xGM=tensorAddr(xList,t)+begin; yGM=tensorAddr(yList,t)+begin
  for tile=0; tile<ceil((end-begin+1)/(U/2)); ++tile:
    n = min(U/2, remaining)
    inputUB = copy GM[tile*U/2 : +n]       # tail uses byte DataCopyPad
    PIPE_V barrier; outputUB = cosh(inputUB,n); PIPE_V barrier
    copy outputUB[0:n] -> GM[tile*U/2 : +n] # tail byte copy
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `tensorAddr(x,t)+cursorStart+tile*U/2` | `n` elements | half/ND flat storage | caller | `CopyIn` | caller；Kernel只读 | full tile U bytes；tail精确 `n*2` bytes |
| `dataQueue` | Queue slot | depth2×U bytes | half | `CopyIn` | `Cosh` | `TPipe`; tile从 Alloc 到 Free | double buffer；32 B aligned U |
| `outQueue` | Queue slot | depth2×U bytes | half | `Cosh` | `CopyOut` | `TPipe`; Alloc→EnQue→DeQue→Free | double buffer |
| output GM | 与 input slice 同逻辑 offset | `n` elements | half/ND flat storage | `CopyOut` | caller | caller；最终结果 | 只写输入长度前缀 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U bytes"]:2
  outQ["outQueue · depth=2 · 2U bytes"]:2
```

##### 字节账与复用顺序

```text
UB_visible = 2U + 2U = 4U
reserveMath = 98304 B
P + reserveMath + 4U <= UB（由 U 的向下取整成立）
dynamic_ub = N/A；AIV TPipe 静态队列
workspace_total = 0 B（Kernel 不读取 user workspace；Tiling 系统槽写32 B）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | `CopyIn` | `ForeachTriangle::Compute` | `FreeTensor` 后的后续 tile | EnQue/DeQue/FreeTensor + V barrier | depth2，消费前不得复用。 |
| `outQueue` slot | `Cosh` | `CopyOut` | `FreeTensor` 后的后续 tile | EnQue/DeQue/FreeTensor | GM copy 完成前不得覆盖。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue protocol | `DataCopy`/`DataCopyPad` | DeQue | input slot | half 保持；tail按 `n*sizeof(half)` bytes。 |
| Compute | `PipeBarrier<PIPE_V>` | DeQue/input ready | `Cosh` 前后 | Vector pipeline | 直接 half `Cosh`，count=`n`，无显式升精度。 |
| UB→GM | Queue protocol | EnQue | `CopyOut` DeQue | output slot | 仅提交 n 个有效元素。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `GetNeedCoreNum`, `AssignDataToEachCore`, `DivideUbMemory4` | 32 B 分核、每核拼接域、U 公式。 |
| Kernel entry/router | `foreach/foreach_cosh/op_kernel/foreach_cosh.cpp` | key1 branch | 模板实参为 half/half/`Cosh<half,false>`。 |
| Device body/helper | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `Process/CopyIn/CopyOut` | tile地址、tail DMA、Queue生命周期。 |
| Device body/helper | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | generic `InnerComputer::Compute` | V barrier 与 count-aware `Cosh`。 |

### Route R-002：FP32 AIV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key2 |
| Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory4` | `dataTypeSize==4` | reserveMath=65536 B，U 32 B对齐 |
| Router | `foreach/foreach_cosh/op_kernel/foreach_cosh.cpp` | `foreach_cosh` | `TILING_KEY_IS(2)` | `ForeachTriangle<float,float,Cosh<float,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_cosh -> ForeachTriangle<float,float,Cosh<float,false>>::Init/Process
  -> per-core tensor slice -> per tile CopyIn<float>
  -> generic InnerComputer::Compute -> Cosh<float,false>
  -> CopyOut<float>
```

##### 源码等价伪码

```text
与 R-001 相同，但 maxDataCount=U/4，full tile 搬 U bytes；
tail copy blockLen=n*4 bytes；Cosh 的 dst/src 均为 float LocalTensor，count=n。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `tensorAddr(list,t)+cursorStart+tile*U/4` | n elements | float/ND flat | caller/CopyOut | CopyIn/caller | caller | tail `n*4` B |
| `dataQueue` | slot | depth2×U | float | CopyIn | Cosh | TPipe per tile | 32 B aligned |
| `outQueue` | slot | depth2×U | float | Cosh | CopyOut | TPipe per tile | 32 B aligned |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2U bytes"]:2
  outQ["outQueue · depth=2 · 2U bytes"]:2
```

##### 字节账与复用顺序

```text
UB_visible=4U
reserveMath=65536 B
P+reserveMath+4U<=UB
dynamic_ub=N/A
workspace_total=0 B（用户 workspace）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input slot | CopyIn | Cosh | FreeTensor后 | Queue+V barrier | 同 R-001。 |
| output slot | Cosh | CopyOut | FreeTensor后 | Queue | 同 R-001。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn | DataCopy/Pad | GM | Queue consumer | input slot | FP32；full U bytes，tail n*4 B。 |
| Compute | V barriers + `Cosh<float,false>` | input Queue | output Queue | Vector data | FP32直接计算；count=n 排除无效 lane。 |
| CopyOut | DataCopy/Pad | output Queue | GM | output slice | 精确写n元素。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory4` | FP32 reserve/U公式。 |
| Kernel entry/router | `foreach/foreach_cosh/op_kernel/foreach_cosh.cpp` | key2 branch | float body映射。 |
| Device body/helper | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | generic `InnerComputer<float,float,...>` | 直接 FP32 Cosh。 |

### Route R-003：BF16→FP32 AIV

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key4 |
| Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory4` | BF16 | `/10` attenuation、`/4` queues、64 B向下对齐 |
| Router | `foreach/foreach_cosh/op_kernel/foreach_cosh.cpp` | guarded key4 | BF16 guard成立 | `ForeachTriangle<bfloat16_t,float,Cosh<float,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_cosh -> ForeachTriangle<bfloat16_t,float,Cosh<float,false>>::Init
  -> dataQueue/outQueue depth2×9U; float32Queue depth1×2U
  -> Process outer tile (maxDataCount=9U/2)
     -> BF16 CopyIn
     -> InnerComputer BF16::Compute
        -> per cast chunk (maxCastDataCount=U/4)
           -> Cast BF16->FP32(first half float32Queue)
           -> Cosh<float,false>(second half, first half, n)
           -> Cast FP32->BF16(CAST_RINT)
     -> BF16 CopyOut
```

##### 源码等价伪码

```text
for each core-owned tensor slice:
  for outer tile of at most 9U/2 BF16 elements:
    copy n BF16 GM -> dataQueue slot
    for j in 0..ceil(n/(U/4))-1:
      m=min(U/4,n-j*U/4)
      f32[0:m]=cast_bf16_to_f32(input[j*U/4:+m])
      PIPE_V barrier
      f32[U/4:U/4+m]=cosh(f32[0:m])
      PIPE_V barrier
      output[j*U/4:+m]=cast_rint_f32_to_bf16(f32[U/4:+m])
    copy n BF16 -> GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | outer offset `tile*9U/2` | n BF16 | BF16/ND flat | caller/CopyOut | CopyIn/caller | caller | tail n*2 B |
| `dataQueue` | slot | depth2×9U | BF16 | CopyIn | Cast | TPipe outer tile | U is64 B aligned |
| `outQueue` | slot | depth2×9U | BF16 | final Cast | CopyOut | TPipe outer tile | 同上 |
| `float32Queue` first half | offset0 | U bytes=`U/4` float | | Cast in | Cosh src | depth1，整个Process期间持有 | 与second half同一2U slot |
| `float32Queue` second half | offset U bytes | U bytes | | Cosh dst | final Cast | 同上 | 非原地Cosh，避免覆盖输入 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["dataQueue · depth=2 · 18U B"]:2
  outQ["outQueue · depth=2 · 18U B"]:2
  fIn["f32 src · U B"]:1
  fOut["f32 dst · U B"]:1
```

##### 字节账与复用顺序

```text
UB_visible = 18U + 18U + 2U = 38U
reserveMath = 98304 B
Host uses U=floor64((UB-P-reserveMath)/40)
=> P+reserveMath+40U<=UB，覆盖显式38U并保留2U余量
dynamic_ub=N/A
workspace_total=0 B（用户 workspace）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| f32 first U | BF16→FP32 Cast | Cosh src | 下一 inner chunk Cast | V barrier | 当前Cosh消费完成后覆盖。 |
| f32 second U | Cosh dst | FP32→BF16 Cast | 下一 inner chunk Cosh | V barrier | final Cast完成后覆盖。 |
| Queue slots | Copy/compute | downstream stage | FreeTensor后 | Queue协议 | 同R-001。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Cast in | `Cast(...,CAST_NONE,m)` + V barrier | BF16 input | Cosh | f32 first half | 升到FP32；inner tail=m。 |
| Cosh | `Cosh<float,false>(dst,src,m)` + V barrier | f32 src | Cast out | f32 halves | FP32计算。 |
| Cast out | `Cast(...,CAST_RINT,m)` + V barrier | f32 result | BF16 Queue | BF16 output | RINT降精度；outer/inner双tail。 |
| GM copy | DataCopy/Pad | Queue | GM | exact prefix | n*2 B tail。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `DivideUbMemory4` | BF16 attenuate/align U。 |
| Kernel entry/router | `foreach/foreach_cosh/op_kernel/foreach_cosh.cpp` | guarded key4 | BF16输入、FP32计算模板。 |
| Device body/helper | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | BF16 `Init` | 9U Queue capacity、U/4 cast count。 |
| Device body/helper | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | BF16 `InnerComputer` | 分块Cast/Cosh/Cast与RINT。 |

## 算子特定优化

### Trick T-001：跨 Tensor 拼接域按 32 B block 均衡分核

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：任意非空 TensorList；`blockDim=min(AIV核数,ceil(E/B),50)`。
- Golden 来源：[`foreach/foreach_utils/op_host/foreach_tiling_class.h:650-695`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h#L650-L695)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
int64_t tempPerCoreCount = blockCount / needCoreNum * static_cast<int64_t>(elementsPerBlock);
int64_t remainderCount = blockCount % needCoreNum;
```

#### 机制与设计原因

先把整个 TensorList 视为逻辑拼接域，以 32 B block 均衡给 AIV 核，再把每核区间反解成 tensor index/offset。这样小 Tensor 可以共享同一核，大 Tensor 可跨核，无需为每个 Tensor 单独 launch；remainder 核各多一个 block，真实末尾再由 end offset 限定。

#### 迁移边界

- 适用：能以扁平 storage 元素顺序独立计算的 TensorList unary 算子。
- 前提：每个元素无跨元素依赖；GM 指针表格式相同；POD 数组容量覆盖 Tensor 与 core 数。
- 失效：要求整 Tensor 独占核、跨元素归约或输出布局不同的算法必须重做分核。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时若只按 Tensor 数均分，会忽略 Tensor 间元素数差异，但该后果未做性能实测。

### Trick T-002：BF16 用双层 tile 在固定 2U FP32 区内升精度

#### 触发与源码依据

- 适用 Route：R-003。
- 触发条件：dtype BF16、key4 且编译 guard 成立。
- Golden 来源：[`foreach/foreach_utils/op_kernel/foreach_triangle.h:51-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h#L51-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t castTimes = dataCount / maxCastDataCount;
uint32_t castTimesRemainder = dataCount % maxCastDataCount;
```

#### 机制与设计原因

外层 Queue tile 可达 `9U/2` 个 BF16 元素，内部按 `U/4` 个元素切成 FP32 chunk；2U FP32 Queue 一半放源、一半放 `Cosh` 结果。循环复用这两个子区，避免为整个 BF16 outer tile 物化 FP32，同时将数学运算保持在 FP32，最后用 `CAST_RINT` 回 BF16。

#### 迁移边界

- 适用：低精度输入、FP32 primitive、可独立分块的逐元素算子。
- 前提：内外 tile 的 count/offset 单位保持一致；两半之间不别名；V barrier 保持。
- 失效：需要跨 chunk 状态或 primitive 不能按 count 处理 tail 时不可直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16→FP32 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key1 | key2 | guarded key4 | dtype-only Host key。 |
| 模板与 dtype | half→half Cosh | float→float Cosh | BF16→float Cosh→BF16 | BF16显式升精度。 |
| 循环与分核 | B=16，tile U/2 | B=8，tile U/4 | B=16，outer 9U/2，inner U/4 | dtype字节数与buffer布局。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 4U Queue | 4U Queue | 38U Queue | 无显式TBuf/dynamic UB/通信。 |
| 内存总占用与复用 | reserve 98304 +4U | reserve 65536 +4U | reserve 98304 +38U，Host按40U预算 | 数学primitive reserve与Cast区。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy/Pad + Cosh half | DataCopy/Pad + Cosh float | DataCopy/Pad + Cast/Cosh/Cast | 均为AIV Vector。 |
| 同步、精度与 tail | V barrier，half | V barrier，float | 多个V barrier，FP32，RINT | count与精度路线不同。 |
| fallback/失败边界 | unknown key no-op | 同 | guard不成立 no-op | 固定entry无default错误上报。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，`ascend910b` | 公开支持，`ascend910_93` | 公开支持，但见独立 `_950` 卡。 |
| Host/Launcher | arch22 `ForeachCommonTiling` | 与A2共用arch22 | arch35独立Tiling。 |
| Kernel/资源 | 公共 AIV_VECTOR `ForeachTriangle` | 与A2共用 | SIMT VF、无AIV Queue；不在本卡展开。 |

## 限制与失败边界

- 公开接口支持 FP16/FP32/BF16、rank 0–8、空 Tensor；输出必须连续，输入非连续转换属于自动生成/OpDef框架边界。
- A2/A3 公共 POD 固定最多 50 个 Tensor、50 个 core；这是源码隐藏上限，公开文档未列出 TensorList 长度上限。
- 输出可以比输入大，但本 Kernel 只写与输入元素数相同的前缀；额外区间未定义为本算子输出。
- Host 的 UB 减法没有显式资源不足 guard；`P+reserveMath<=UB` 由受支持平台隐含保证。`P` 的精确宏序列化布局在固定仓不可见。
- unknown TilingKey 和被编译 guard 排除的 BF16 key 没有 Device error/fallback，而是 entry 不执行 body；正常公开链路必须阻止这种组合。
- 自动生成 ACLNN inner bridge、executor 具体布局和系统 workspace 对外 size 的换算固定仓不可见；不推测补齐。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_cosh/docs/aclnnForeachCosh.md` | `aclnnForeachCoshGetWorkspaceSize`, `aclnnForeachCosh` | 公开原型、dtype/shape/连续性/错误语义、A2/A3支持。 |
| 入口实现 | `foreach/foreach_cosh/op_host/foreach_cosh_def.cpp` | `ops::ForeachCosh`, `OP_ADD` | OpDef dtype/layout/AutoContiguous和平台注册；inner bridge为生成边界。 |
| 构建接线 | `foreach/foreach_cosh/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `SUPPORT_TILING_DIR` | A2/A3到arch22 Tiling及ACLNN生成接线。 |
| Host/Tiling/Launcher | `foreach/foreach_cosh/op_host/arch22/foreach_cosh_tiling_arch22.cpp` | `Tiling4ForeachCoshTiling` | ForeachCosh到公共Tiling的入口。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、workspace与POD producer。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD字段、容量、ForeachCosh注册。 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | key与dtype映射。 |
| Kernel entry/body | `foreach/foreach_cosh/op_kernel/foreach_cosh.cpp` | `foreach_cosh` | key router与模板实例。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、循环、DMA/tail。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD消费、TensorList地址解析、BF16容量换算。 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | `ForeachTriangle`, `InnerComputer` | Cosh调用、BF16精度与同步。 |
| 构建接线 | `foreach/foreach_cosh/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3使用公共Kernel source。 |
| 构建接线 | `foreach/foreach_cosh/op_host/config/ascend910b/foreach_cosh_binary.json` | `op_list` | A2三dtype binary条目。 |
| 构建接线 | `foreach/foreach_cosh/op_host/config/ascend910_93/foreach_cosh_binary.json` | `op_list` | A3三dtype binary条目。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段的Tensor/core区间编码 | 数组容量50、字段位宽/顺序、闭区间单位 | `ForeachCommonTilingData`; `ParseTilingData` |
| 调用顺序 | Init→per tensor→per tile CopyIn/Compute/CopyOut | TensorList地址表协议与cursorStart | `KernelForeachUnary::Process` |
| workspace/内存 | 用户workspace不使用；U按dtype预算 | double buffer depth2；BF16 9U/2U布局；reserve公式 | `RunBigKernelTiling`; `DivideUbMemory4`; `Init` |
| 同步 | Queue协议与Cosh/Cast间V barrier | 不得提前覆盖Queue和FP32两半 | `InnerComputer::Compute` |
| 精度与尾块 | count-aware Cosh；BF16 FP32化/RINT | tail byte数、inner chunk offset与round mode | `CopyIn`; `CopyOut`; BF16 `ComputePerCast` |
| 失败边界 | Host校验失败阻断launch | 不得让unknown key或guard冲突进入Device | `CheckOpInfo`; `foreach_cosh` router |
