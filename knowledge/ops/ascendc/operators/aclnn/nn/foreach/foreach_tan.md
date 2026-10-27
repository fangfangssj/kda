---
type: Operator
title: "ForeachTan A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachTan 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/docs/aclnnForeachTan.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_host/foreach_tan_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_host/arch22/foreach_tan_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/foreach_tan.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tan/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:22:21Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachTan A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachTanGetWorkspaceSize` + `aclnnForeachTan`，公开 ACLNN 两段式接口；固定仓未给接口版本后缀，故采用该公开版本。 |
| 计算语义 | 对 TensorList 中每个输入张量逐元素计算 `y_i[j] = tan(x_i[j])`。 |
| 输入输出 | `x`/`out` 均为 ND TensorList；元素 dtype 为 FP16、FP32 或 BF16，列表内及输入输出 dtype 一致；输出每项容量不少于对应输入。 |
| 执行类型 | 单 Kernel、单个 `AIV_VECTOR` 节点；不存在 Cube/AIC 或 MIX_CV 可达节点。 |
| Kernel 路线 | R-001 FP16 原生向量；R-002 FP32 原生向量；R-003 BF16→FP32→BF16 向量。 |
| 核心资源 | ListTensor GM 描述符、每核 GM 分段、`dataQueue`/`outQueue` 双缓冲；BF16 另有单深度 `float32Queue`；算子 Kernel 不读 workspace。 |
| 来源闭包 | `source_closed`：公开 API 约束→OpDef/arch22 Tiling→`ForeachCommonTilingData`→dtype router→`KernelForeachUnary`/`ForeachTriangle` 已闭合；生成式 ACLNN bridge 与 AscendC `Tan` primitive 内部属于框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入列表 `x=[x_0,...,x_(n-1)]`，输出列表 `y=[y_0,...,y_(n-1)]`。对每个 `i` 及 `0 <= j < numel(x_i)`，写 `y_i[j]=tan(x_i[j])`。Host 允许 `numel(y_i) >= numel(x_i)`；Device 仅写输入有效元素数，因此输出多出的区域不在该算子写入域内。公开约束给出的功能精度值域是 `[-65504, 65504]`，并声明默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 每个 Tensor 为 ND，公开文档维数 0–8 | FP16/FP32/BF16 | 非空 TensorList 指针；支持列表中的空 Tensor；所有成员 dtype 一致；输入可非连续，OpDef 配置 `AutoContiguous()`。 |
| `out` | 每个 Tensor 为 ND，维数 0–8；对应项 `numel(out_i) >= numel(x_i)` | 与 `x` 相同 | 非空 TensorList 指针；输出不支持非连续 Tensor；只写前 `numel(x_i)` 个元素。 |
| `workspaceSize`/`executor` | 标量输出/Host 对象 | `uint64_t*`/`aclOpExecutor**` | 第一段接口返回；executor 保存计算流程。 |
| `workspace`/`stream` | Device 地址/运行时流 | N/A | 第二段接口消费第一段结果并在指定 stream 上提交执行。 |

### 入口与版本

公开文档只给出无后缀 `aclnnForeachTan` 两段式版本，未出现更高版本；A2、A3 产品表均明确支持。`ForeachTan` OpDef 为 `ascend910b` 与 `ascend910_93` 注册 FP16/FP32/BF16 动态 ND 输入输出，构建把二者映射到 `arch22`。固定仓不包含生成的 `aclnnop/aclnn_foreach_tan.h` 或 ACLNN inner bridge 源文件；接口到 OpDef 的 executor 组装、输入 Contiguous 临时量和 ACLNN 总 workspace 合并公式属于固定仓不可见的生成框架边界，未被推测补齐。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口先拒绝 `x`/`out` 空指针、非支持 dtype、输入输出 dtype 不一致、shape 不满足、列表内 dtype 不一致或成员维数大于 8。可见 OpDef 对输入和输出都声明 `AutoContiguous()`；公开 API 仅承诺输入可非连续、输出必须连续。arch22 Tiling 的 `Init(FOREACH_TAN_OP_CODE)` 最多枚举 50 个动态输入，取得存储 shape 元素数并锁定 dtype；`CheckOpInfo()` 再检查 TensorList 数量、每个输入维数、输出 dtype 与容量以及列表 dtype 一致性，任何失败返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

`ForeachCommonTiling::RunBigKernelTiling()` 从平台读取 AIV 核数与 UB 容量，生成分核边界、UB tile 容量和 POD。它向算子节点报告 32 字节 workspace，但 Device entry 显式令 `userWS=nullptr`，所以此 Kernel 不消费该地址。ACLNN 第一段可能还为框架输入连续化分配临时空间；由于生成 bridge 不在固定仓，调用者最终 `workspaceSize` 的合并公式、临时量 offset 与 owner 细节固定仓不可见。executor 在第二段调用前有效，workspace 由调用者分配并至少维持到 stream 上任务完成。

### 启动图与失败传播

单节点启动图为：ACLNN executor（可能先做框架 Contiguous）→Host 调用 `ForeachTan` Tiling→在同一 stream 启动 `foreach_tan(xListDesc,yListDesc,workspace,tiling)`。节点 `blockDim=needCoreNum`，POD 为 `ForeachCommonTilingData`，输入/输出边是各自 ListTensor 描述符；没有 Kernel 间 workspace 边。Tiling 任一空描述符、shape/dtype 校验、workspace 数组获取或 POD 序列化失败都阻断启动。Kernel entry 对未知 key 没有 `else` 报错分支，会无输出返回；正常 Host 只产生 key 1/2/4。

## Route 与参数设计

### 分核与资源计算

设 dtype 字节数 `d`，每 32 字节块元素数 `e=32/d`，全部输入元素数 `N=sum_i numel(x_i)`。`needCoreNum=min(platformAivCores, max(1, ceil(N/e)))`，且上限 50。分配以 32 字节块为单位：`blockCount=ceil(N/e)`，基准每核元素数 `floor(blockCount/needCoreNum)*e`，前 `blockCount mod needCoreNum` 个核多处理 `e` 个元素；每核用 `tensorStart/End` 与首尾 offset 表示可能跨多个列表成员的连续逻辑段。

设平台 UB 为 `U`，POD 序列化大小为 `P=tilingData.GetDataSize()`。Tan 临时 reserve 初值：FP16/BF16 为 `10*2*1024*8=163840` 字节，FP32 为 `4*4*1024*8=131072` 字节；当 `U <= P+reserve` 时反复减半。FP16/FP32 的 `S=floor32((U-P-reserve)/4)`；BF16 的 `S=floor64(((U-P-reserve)/10)/4)`。`S` 写入 `inputsTensorUbSize`，是每个基础 UB 子区的字节容量。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 原生向量 | key 1；`ForeachTriangle<half,half,Tan<half,false>>` | `dataType==DT_FLOAT16`；Host `GetTilingKeyByDtypeOnly` 返回 1 | 检查下一 dtype；不支持 dtype 产生 key 0，公开校验应先失败 |
| 2 | R-002 FP32 原生向量 | key 2；`ForeachTriangle<float,float,Tan<float,false>>` | `dataType==DT_FLOAT`，返回 2 | 检查下一 dtype |
| 3 | R-003 BF16 升精度向量 | key 4；`ForeachTriangle<bfloat16_t,float,Tan<float,false>>` | `dataType==DT_BF16`，返回 4；A2/A3 编译不命中排除 BF16 的 Kirin guard | 未知 key 无输出返回 |

```text
公开校验与 ForeachCommonTiling::Init/CheckOpInfo 成功
  if dtype == FP16: key=1 -> R-001
  else if dtype == FP32: key=2 -> R-002
  else if dtype == BF16: key=4 -> R-003
  else: key=0/校验失败；entry 无匹配分支
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 原生向量 | 1 | A2/A3，FP16，ND，成员维数≤8，输出容量足够，列表 dtype 一致 | 单 AIV Kernel | `ForeachTriangle<half,half,Tan<half,false>>`→`KernelForeachUnary` | 双深度输入/输出 Queue；Queue 与 `PipeBarrier<PIPE_V>` | 整 tile 用 `DataCopy`；尾 tile 用字节长度 `DataCopyPad` |
| R-002 FP32 原生向量 | 2 | 同上但 FP32 | 单 AIV Kernel | `ForeachTriangle<float,float,Tan<float,false>>` | 与 R-001 同结构，tile 字节公式按 FP32 | 同 R-001 |
| R-003 BF16 升精度向量 | 4 | 同上但 BF16，且非 Kirin 3003/3113 编译目标 | 单 AIV Kernel | `ForeachTriangle<bfloat16_t,float,Tan<float,false>>`→BF16 `InnerComputer` | 输入/输出 Queue 每 buffer 为 `9S`，另有 `2S` FP32 Queue；Cast 间 V barrier | tile 内按 `S/4` 个元素分段 Cast；末段受有效 `dataCount` 保护 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 50 项列表/POD上限 | `Init` 循环 `i<MAX_TENSOR_CONT(50)` | POD 各 tensor/core 数组均为 50 项 | 全部 | 固定仓未见显式 `>50` 报错；生成 ACLNN bridge 是否预先限制不可见 | `foreach_tiling_def.h::MAX_TENSOR_CONT`；`ForeachCommonTiling::Init` |
| 32 字节分核 | `elementsPerBlock=32/d` | 保证非尾核按 DMA block 粒度切分 | 全部 | 尾部由 `DataCopyPad` 处理 | `GetNeedCoreNum`；`AssignDataToEachCore` |
| Tan reserve 衰减 | `while (U <= P+extraBuffer) extraBuffer/=2` | 为 AscendC Tan primitive 临时需求留空间，同时避免 reserve 超过 UB | 全部 | reserve 逐次减半；源码未给零 UB 独立失败 | `DivideUbMemory4` |
| BF16 10 倍缩放与 64B 对齐 | dtype BF16 | 覆盖 `9S` copy 区与 FP32 cast 区，且 BF16 基础块对齐 | R-003 | 非 BF16 不走该缩放 | `DivideUbMemory4`；`KernelForeachBase::Init` |
| BF16 编译 guard | `__CCE_AICORE__>=220` 且 arch 非 3003/3113 | BF16 Cast/body 仅在该条件编译 | R-003 | A2/A3 active config 可达；Kirin 不生成该分支 | `foreach_tan.cpp::foreach_tan`；`foreach_triangle.h::InnerComputer` |
| 输出仅按输入计数写 | `numel(out_i)>=numel(x_i)` | Device loop 的 `dataCount` 来自输入 POD | 全部 | 额外输出元素保持算子外既有值 | `CheckOutputShapeAndDtype`；`KernelForeachUnary::Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 宏生成结构的 `sizeof/alignof/offsetof` 固定仓不可见 | 字节 | `FillTilingData`: `set_inputsTensorUbSize(S)` | `KernelForeachBase::ParseTilingData`，计算 Queue 容量与 tile 元素数 | FP16/FP32 32B 对齐；BF16 64B 对齐 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 同上 | 元素 | `Init`: 每项 `storageShape.GetShapeSize()` | `KernelForeachUnary::Process` 取得成员元素数 | 未活动槽由对象默认值/宏序列化；不应读取 |
| 同上 | `tensorStartList[50]` | `uint16_t[50]` | 同上 | TensorList 索引 | `AssignDataToEachCore` | 当前 `blockIdx` 的首成员 | blockDim≤50 |
| 同上 | `tensorEndList[50]` | `uint16_t[50]` | 同上 | TensorList 索引 | `AssignDataToEachCore` | 当前核末成员，循环为闭区间 | 同上 |
| 同上 | `tensorStartOffsetList[50]` | `int64_t[50]` | 同上 | 元素 | `AssignDataToEachCore` | 首成员 GM 起始偏移 | 相对对应 Tensor 起点 |
| 同上 | `tensorEndOffsetList[50]` | `int64_t[50]` | 同上 | 元素 | `AssignDataToEachCore` | 末成员闭区间终点 | `dataCount=end-start+1` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | `foreach_tan_tiling_arch22.cpp::Tiling4ForeachTanTiling`→`ForeachCommonTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_tan.cpp::foreach_tan` key 1 | `ForeachTriangle<half,...>`；`KernelForeachUnary` | `op_kernel/CMakeLists.txt::add_kernel_sources(ascend910b,ascend910_93)` |
| R-002 FP32 | 2 | 同上 | 同上 | 同 entry key 2 | `ForeachTriangle<float,...>`；`KernelForeachUnary` | 同上 |
| R-003 BF16 | 4 | 同上 | 同上 | 同 entry key 4 + arch guard | BF16 `InnerComputer`；`KernelForeachUnary`/`KernelForeachBase` | 同上 |

## Kernel 实现

### 公共执行流程

entry 解析 POD 后按 key 构造一个 `ForeachTriangle`，调用 `Init` 与 `Process`。`Init` 读取当前 `blockIdx` 的 TensorList 首尾范围，绑定 x/y ListTensor 描述符并初始化 Queue。`Process` 遍历本核覆盖的列表成员，利用描述符中的地址表解析实际 GM 指针；成员被核边界截断时叠加 start/end offset。每个成员再按 `maxDataCount` 分 tile，执行 CopyIn→Tan→CopyOut。Queue 的 EnQue/DeQue/FreeTensor 建立搬运与 Vector 消费顺序；Tan/Cast 相邻步骤由 `PipeBarrier<PIPE_V>` 隔离。

### Route R-001：FP16 原生向量

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachTan.md` | `aclnnForeachTanGetWorkspaceSize` | x/out 为 FP16 且满足公开约束 | 创建 executor |
| Host/Tiling | `foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT16` | key 1 |
| Router | `foreach_tan/op_kernel/foreach_tan.cpp` | `foreach_tan` | `TILING_KEY_IS(1)` | `ForeachTriangle<half,half,Tan<half,false>>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachCommonTiling::RunBigKernelTiling -> foreach_tan(x,y,workspace,tiling)
  -> ForeachTriangle<half,...>::Init
  -> KernelForeachUnary::Process
    -> SingleTensorProcess -> CopyIn
    -> ForeachTriangle::Compute -> InnerComputer<half,half>::Compute -> Tan<half,false>
    -> KernelForeachUnary::CopyOut
```

##### 源码等价伪码

```text
# 源码等价伪码
core = GetBlockIdx(); load this core's [tensorStart,tensorEnd] and offsets
for tensor i in closed interval:
  begin = (i==start ? startOffset : 0)
  end   = (i==end ? endOffset : numel[i]-1)
  bind xGM/yGM to list_address[i] + begin
  for tile k over count=end-begin+1, capacity=S/2 elements:
    n = min(capacity, remaining)
    CopyIn xGM[k*capacity : +n] (tail uses byte-count DataCopyPad)
    PipeBarrier(V); Tan<half,false>(outUB,inUB,n); PipeBarrier(V)
    CopyOut exactly n elements (tail uses byte-count DataCopyPad)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x GM | `GetTensorAddr(i,x)+cursorStart+k*(S/2)` | `n` elements | FP16/ND | ACLNN caller/framework Contiguous | `CopyIn` | caller/framework；stream 完成前有效 | 非尾按元素 DMA；尾按字节 CopyPad |
| `dataQueue` | Queue 管理 | depth 2，每 buffer `S` 字节 | FP16 | `CopyIn` | Tan | `TPipe`，FreeTensor 后复用 | 双缓冲 |
| `outQueue` | Queue 管理 | depth 2，每 buffer `S` 字节 | FP16 | Tan | `CopyOut` | `TPipe`，CopyOut 后释放 | 双缓冲 |
| y GM | `GetTensorAddr(i,y)+cursorStart+k*(S/2)` | `n` elements | FP16/ND | `CopyOut` | caller | caller owner | 只写有效 n |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2S bytes"]:2
  outQ["outQueue · depth=2 · 2S bytes"]:2
```

`extraBuffer` 是 Host 选择 tile 时为 `Tan` primitive 留出的资源裕量，不是该 Kernel 通过 `InitBuffer` 声明的可定位子区；其物理布局固定仓不可见，因此未画入显式 UB 布局。

##### 字节账与复用顺序

```text
explicit_UB = 2S + 2S = 4S
reserve = attenuate(163840) until U > P + reserve
4S <= U - P - reserve, S=floor32((U-P-reserve)/4)
dynamic_ub = N/A（AIV Queue 路径）
workspace_total = Kernel 报告 32 字节；entry 不读取
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | `CopyIn` | `Compute` DeQue | `FreeTensor` 后下一 tile | EnQue→DeQue→FreeTensor | depth=2 且消费前不得覆盖 |
| `outQueue` slot | Tan | `CopyOut` DeQue | `FreeTensor` 后下一 tile | EnQue→DeQue→FreeTensor | 仅拷贝有效 `n` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | VECIN Queue | DMA | Compute DeQue | 输入 slot | 尾 tile 用 `blockCount=1, blockLen=n*2` 字节 |
| Tan | `PipeBarrier<PIPE_V>` + `Tan<half,false>` | input Queue | output EnQue | Vector 读写 | FP16 原生计算；公开精度值域限制仍适用 |
| UB→GM | VECOUT Queue | Compute EnQue | CopyOut DeQue | 输出 slot | 仅写 n 个元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling`/`DivideUbMemory4` | key、blockDim、S、POD、workspace |
| Kernel entry/router | `foreach_tan/op_kernel/foreach_tan.cpp` | key 1 branch | 绑定 FP16 Triangle 模板 |
| Device body/helper | `foreach_utils/op_kernel/foreach_triangle.h`、`kernel_foreach_unary.h` | `InnerComputer`/`SingleTensorProcess` | Tan 顺序、双 Queue、tail CopyPad |

### Route R-002：FP32 原生向量

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachTan.md` | 第一段接口 | x/out 为 FP32 | executor |
| Host/Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `op_kernel/foreach_tan.cpp` | `foreach_tan` | `TILING_KEY_IS(2)` | `ForeachTriangle<float,float,Tan<float,false>>` |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> foreach_tan key 2
  -> ForeachTriangle<float,...>::Init/Process
    -> KernelForeachUnary::CopyIn
    -> InnerComputer<float,float>::Compute -> Tan<float,false>
    -> KernelForeachUnary::CopyOut
```

##### 源码等价伪码

```text
# 源码等价伪码
按 POD 的闭区间分核遍历 TensorList；每项以 capacity=S/4 元素切 tile
CopyIn(n) -> barrier(V) -> Tan<float,false>(n) -> barrier(V) -> CopyOut(n)
最后不足 capacity 的 tile 用 n*4 字节 DataCopyPad，地址步长仍为 k*capacity
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | list ptr + `cursorStart+k*(S/4)` | n elements | FP32/ND | caller/Kernel | Kernel/caller | stream 生命周期 | 尾 n*4 字节 |
| `dataQueue` | Queue slot | depth 2 × S | FP32 | CopyIn | Tan | TPipe | FreeTensor 后复用 |
| `outQueue` | Queue slot | depth 2 × S | FP32 | Tan | CopyOut | TPipe | FreeTensor 后复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2S bytes"]:2
  outQ["outQueue · depth=2 · 2S bytes"]:2
```

FP32 的 `extraBuffer` 同样只是 Host selector 裕量，初值 131072 字节并可二分衰减；固定仓没有与之对应的显式 `InitBuffer`，故不把它伪造成 UB 子区。

##### 字节账与复用顺序

```text
explicit_UB=4S; S=floor32((U-P-reserve)/4)
reserve=attenuate(131072); dynamic_ub=N/A
workspace_total=32 bytes reported, not read by entry
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| input/output Queue slot | CopyIn/Tan | Tan/CopyOut | 对应 FreeTensor 后 | Queue 协议 | 与 R-001 相同，dtype 宽度改为 4 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn/Out | Queue + DataCopy/Pad | GM/UB | Compute/GM | tile slot | FP32；尾部精确 n 个元素 |
| Compute | V barrier + `Tan<float,false>` | input UB | output Queue | Vector pipeline | 不发生显式 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | dtype key/`DivideUbMemory4` | FP32 reserve 与 tile |
| Kernel entry/router | `foreach_tan.cpp` | key 2 | FP32 模板 |
| Device body/helper | `foreach_triangle.h`/`kernel_foreach_unary.h` | generic `InnerComputer` | 原生 FP32 Tan 与 tail |

### Route R-003：BF16 升精度向量

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachTan.md` | 第一段接口 | BF16 | executor |
| Host/Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_BF16` | key 4 |
| Router | `op_kernel/foreach_tan.cpp` | guarded key 4 | arch 非 3003/3113 | BF16 input + FP32 Tan body |

#### Device 数据流

##### 函数级调用链

```text
RunBigKernelTiling -> foreach_tan key 4
  -> ForeachTriangle<bfloat16_t,float,...>::Process
    -> CopyIn BF16
    -> InnerComputer BF16 specialization::Compute
      -> ComputePerCast -> Cast BF16→FP32 -> Tan<float,false> -> Cast FP32→BF16(CAST_RINT)
    -> CopyOut BF16
```

##### 源码等价伪码

```text
# 源码等价伪码
capacity = maxDataCount = 9S/2 BF16 elements
castCapacity = maxCastDataCount = S/4 FP32 elements
for each GM tile of at most capacity:
  CopyIn BF16
  for each cast segment q of at most castCapacity:
    Cast fp32Buf[0:qn] <- bf16In[q*castCapacity:] (CAST_NONE)
    Tan fp32Buf[castCapacity:castCapacity+qn] <- fp32Buf[0:qn]
    Cast bf16Out[q*castCapacity:] <- fp32 result (CAST_RINT)
  CopyOut valid BF16 count
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | list ptr + cursor/tile offset | n BF16 elements | BF16/ND | caller/Kernel | Kernel/caller | stream | tail n*2 bytes |
| `dataQueue` | Queue slot | depth 2，每 buffer `9S` | BF16 | CopyIn | Cast | TPipe | 双缓冲 |
| `outQueue` | Queue slot | depth 2，每 buffer `9S` | BF16 | down-Cast | CopyOut | TPipe | 双缓冲 |
| `float32Queue` | 单 slot 内 offset 0 与 `S/4` 元素 | depth 1，总 `2S` | FP32 | up-Cast/Tan | Tan/down-Cast | 整个 Process DeQue 后持有 | 同一物理 tensor 分输入/输出两段 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=2 · 18S"]:2
  fp32["float32Queue · depth=1 · 2S"]:1
  outQ["outQueue · depth=2 · 18S"]:2
```

##### 字节账与复用顺序

```text
explicit_UB = 2*9S + 2*9S + 2S = 38S
S=floor64(((U-P-reserve)/10)/4), hence 38S <= 0.95*(U-P-reserve) ignoring alignment loss
reserve attenuates from 163840 bytes; dynamic_ub=N/A
workspace_total=32 bytes reported, unused by entry
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 `[0,S)` | BF16→FP32 Cast | Tan source | 下一 cast segment | V barrier after Cast/Tan | Tan 完成前不覆盖 |
| FP32 `[S,2S)` | Tan | FP32→BF16 Cast | 下一 segment | V barrier after Tan/Cast | `paramsCount=2` 保证不与 source 重叠 |
| BF16 Queue slots | DMA/Cast | Cast/DMA | FreeTensor 后 | Queue 生命周期 | 与 n/capacity 一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| up-Cast | `Cast(...CAST_NONE)` + V barrier | BF16 Queue | Tan | FP32 source 子区 | cast segment 尾由 `castTimesRemainder` 限定 |
| Tan | `Tan<float,false>` + V barrier | FP32 source | down-Cast | FP32 result 子区 | FP32 计算 |
| down-Cast | `Cast(...CAST_RINT)` + V barrier | FP32 result | CopyOut | BF16 out Queue | 明确 round-to-nearest integer 模式标记 `CAST_RINT` |
| GM tail | DataCopyPad | Queue | GM | 最后 tile | 只搬运 `n*2` 字节 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory4` | 10 倍缩放、64B 对齐 |
| Kernel entry/router | `foreach_tan.cpp` | guarded key 4 | BF16→FP32 模板 |
| Device body/helper | `foreach_triangle.h`/`kernel_foreach_unary.h`/`kernel_foreach_base.h` | BF16 `InnerComputer` | 9S copy buffer、2S FP32 buffer、分段 Cast |

## 算子特定优化

### Trick T-001：按 Tan 临时需求自适应衰减 reserve

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：`opCode==FOREACH_TAN_OP_CODE`；初始 `extraBuffer` 与 dtype 宽度/计算过程数确定，且平台 UB 不足以同时容纳 POD 与该 reserve 时。
- Golden 来源：[`foreach_tiling_class.h:921-938`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h#L921-L938)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t calcPro = TAN_HALF_CALC_PROC;
if (dataTypeSize == BYTE_LEN_4) {
    calcPro = TAN_FLOAT_CALC_PROC;
}
uint32_t extraBuffer = calcPro * static_cast<uint32_t>(dataTypeSize) * TAN_BASIC_BLOCK_SIZE * 8U;
while (static_cast<uint32_t>(ubSizePlatForm) <= static_cast<uint32_t>(tilingData.GetDataSize() + extraBuffer)) {
    extraBuffer = extraBuffer / BUFFER_ATTENUATION;
}
```

#### 机制与设计原因

Host 不用固定最大 tile 挤占 Tan primitive 临时空间，而是先按 FP16/BF16 的 10 个计算过程或 FP32 的 4 个过程预留，并在平台 UB 无法容纳时二分衰减。剩余空间再按输入/输出双 Queue（BF16 还含 cast 扩展）反推 `S`。这使资源选择随真实 UB 容量收缩，改变 tile 大小和每次 GM 搬运量，同时保证显式 Queue 与预留区不会按 Host 公式重叠；primitive 内部如何消费 reserve 固定仓不可见。

#### 迁移边界

- 适用：具有显著内部临时需求、又允许缩小 tile 的单 Kernel AIV 一元算子。
- 前提：临时需求能由 dtype、基础块与过程数给出保守字节式，Queue 总量能从剩余 UB 严格反推。
- 失效：若 primitive 的最小临时空间不可继续衰减，应增加失败/fallback；当前固定源码未提供另一 Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 1 | key 2 | key 4 + arch guard | dtype-only Host key |
| 模板与 dtype | `half/half/Tan<half>` | `float/float/Tan<float>` | `bf16/float/Tan<float>` | BF16 缺少直接 body，显式升精度 |
| 循环与分核 | 32B 块；tile S/2 | 32B 块；tile S/4 | 32B 块；GM tile 9S/2，cast tile S/4 | dtype 宽度和 cast buffer |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 双 Queue | 双 Queue | 双 Queue + FP32 Queue | 均无通信/dynamic UB |
| 内存总占用与复用 | 4S | 4S | 38S | BF16 Host 先除 10 再除 4 |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy/Pad + Tan half | DataCopy/Pad + Tan float | DataCopy/Pad + Cast/Tan/Cast | 无 SIMT/NDDMA |
| 同步、精度与 tail | V barrier；FP16 | V barrier；FP32 | 每次 Cast/Tan 后 V barrier；CAST_RINT | 所有 Route 尾块精确字节搬运 |
| fallback/失败边界 | 无 Device fallback | 无 Device fallback | 编译 guard 外不可达 | 未知 key 静默无输出 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持 | 支持 | 支持，但见独立 `_950` 卡 |
| Host/Launcher | `ascend910b`→arch22 公共 Tiling | `ascend910_93`→arch22 公共 Tiling | arch35 专用 Tiling，POD 与 key 不同 |
| Kernel/资源 | AIV `ForeachTriangle` | 与 A2 共用源码 | SIMT 512 线程、直接 GM `tanf`，不使用 AIV Queue |

## 限制与失败边界

- 公开接口仅支持 FP16/FP32/BF16、ND、成员维数≤8、输入输出 dtype 一致、输出容量不小于输入；输出非连续不支持，输入非连续由框架 `AutoContiguous` 边界处理。
- 公共 arch22 POD 只有 50 个 Tensor 与 50 个 core 槽；固定源码未看到对 TensorList 长度大于 50 的显式错误，生成 ACLNN bridge 是否另行限制固定仓不可见。
- `totalDataCount` 与 shape size 使用 `int64_t`，分核中的 `ceil(a+b-1)` 未见显式溢出保护；POD 宏生成的精确 `sizeof/alignof/offsetof` 固定仓不可见。
- 算子 Kernel 报告 32 字节 workspace 但不读取；ACLNN 总 workspace 中可能存在的输入连续化临时量布局、executor 内部 launch API 与 stream 错误传播细节固定仓不可见。
- AscendC `Tan<T,false>` primitive 的微指令、内部临时区精确布局和误差界固定仓不可见；仅能闭合其调用参数、外部 reserve 和显式同步。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_tan/docs/aclnnForeachTan.md` | 两段式原型/产品表 | ACLNN 语义、平台、错误与值域 |
| 入口实现 | `foreach/foreach_tan/op_host/foreach_tan_def.cpp` | `ForeachTan` | dtype/layout、AutoContiguous、平台配置 |
| Host/Tiling/Launcher | `foreach/foreach_tan/op_host/arch22/foreach_tan_tiling_arch22.cpp` | `Tiling4ForeachTanTiling` | opCode 到公共 Tiling |
| Host/Tiling/Launcher | `foreach/foreach_tan/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`/`SUPPORT_TILING_DIR` | A2/A3→arch22 Host 接线 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD、workspace |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | POD 字段与 ForeachTan 注册 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | dtype→key |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `IMPL_OP_INFERSHAPE(ForeachTan)` | 公共图侧 shape/dtype 推导注册；ACLNN 显式 out 仍按接口约束 |
| Kernel entry/body | `foreach/foreach_tan/op_kernel/foreach_tan.cpp` | `foreach_tan` | key→模板 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | `ForeachTriangle`/`InnerComputer` | Tan/Cast 管线 |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、循环、CopyIn/Out、tail |
| Kernel entry/body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD consumer、ListTensor 地址、BF16 容量 |
| 构建接线 | `foreach/foreach_tan/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 active arch22 源 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段分核描述可复用 | 数组上限 50、字段类型/顺序、闭区间 end 语义 | `ForeachCommonTilingData`/`ParseTilingData` |
| 调用顺序 | Init→按 Tensor→按 tile→CopyIn→Compute→CopyOut | key 必须与模板 dtype 一致；输出地址沿输入元素数推进 | `foreach_tan`/`KernelForeachUnary::Process` |
| workspace/内存 | 32B Kernel workspace 不参与 body；UB 由 S 反推 | reserve、Queue depth、BF16 的 9S/2S 配比不可擅改 | `RunBigKernelTiling`/`InitBuffer` |
| 同步 | Queue 协议和 V barrier | Cast/Tan/复用前 barrier 与 FreeTensor 顺序 | `InnerComputer`/`KernelForeachUnary` |
| 精度与尾块 | FP16/FP32 原生，BF16 经 FP32 | BF16 down-cast `CAST_RINT`；尾部只提交有效字节 | `ComputePerCast`/`CopyIn`/`CopyOut` |
| 失败边界 | Host 校验失败阻断 launch | 未知 key、>50 列表、宏 ABI 不可用推测补齐 | `CheckOpInfo`/entry router |