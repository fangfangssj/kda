---
type: Operator
title: "ForeachNorm A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachNorm 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/docs/aclnnForeachNorm.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_host/foreach_norm_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_util.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_reduce_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_reduce_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/foreach_norm.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/foreach_norm.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:58:55Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachNorm A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachNormGetWorkspaceSize` / `aclnnForeachNorm`，公开 ACLNN 两段式接口；固定仓无版本后缀，故为该平台实际公开版本。 |
| 计算语义 | 对列表中每个 Tensor 独立全量归约；实际 Device selector 为 `int(scalar)==1` 时 `sum(abs(x))`，否则 `sqrt(sum(x*x))`。 |
| 输入输出 | `x`: 动态 TensorList，ND，FP16/FP32/BF16；`scalar`: FLOAT/INT64（ACLNN 文档还声明 DOUBLE，生成桥接不可见）；`out`: 与 `x` 等长、同 dtype，每个 Tensor 含 1 个元素。 |
| 执行类型 | 单 Kernel `foreach_norm`，`KERNEL_TYPE_MIX_AIV_1_0` 中的 AIV_VECTOR body；无可达 Cube/AIC 或 MIX_CV 节点。 |
| Kernel 路线 | R-001 16-bit L1；R-002 16-bit L2；R-003 FP32 L1；R-004 FP32 L2。 |
| 核心资源 | GM TensorList 描述、双缓冲 VECIN/VECOUT、FP32 cast Queue（16-bit 路线）、1 KiB calc TBuf、用户 workspace 中的分片归约值、全核同步。 |
| 来源闭包 | source_closed：公开 API 文档 → OpDef/自动 ACLNN 构建边界 → `ForeachNormMembaseTiling` → `ForeachReduceTilingData` → `foreach_norm` → `ForeachNormND` 已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令输入列表为 `x=[X_0,...,X_{m-1}]`，每个输出 `Y_j` 是单元素 Tensor。固定 Kernel 实际实现为：

```text
int(float(scalar)) == 1: Y_j = cast_T(sum_k abs(float(X_j[k])))
otherwise:               Y_j = cast_T(sqrt(sum_k float(X_j[k])^2))
empty X_j:               Y_j = 0
```

FP16/BF16 在逐片归约前升为 FP32；FP32 直接归约；最终结果写回输入 dtype。公开文档给出一般 p-norm 公式，但固定 Device body 只含 L1/L2 两个模型，非 1 值统一进入 L2，不能按一般 p 次幂复用。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 为 ND，维数 0–8 | FLOAT16/FLOAT32/BFLOAT16，列表内一致 | 支持空 Tensor；A2/A3 Tiling 最多读取 256 项；OpDef 对输入启用 `AutoContiguous()`。 |
| `scalar` | 标量，固定 Kernel 读取 1 个元素 | OpDef: FLOAT32/INT64；ACLNN 文档: FLOAT32/DOUBLE/INT64 | Device 先转成 `float`，再以 `static_cast<int>(scalarVal)==1` 选 L1，否则 L2。 |
| `out` | 与 `x` 等长；每项 storage shape size 必须为 1，ND | 与对应 `x` 同 dtype | 不支持非连续输出；OpDef 对输出启用 `AutoContiguous()`；空输入 Tensor 输出 0。 |

### 入口与版本

公开文件声明两段式 `aclnnForeachNormGetWorkspaceSize(x, scalar, out, workspaceSize, executor)` 和 `aclnnForeachNorm(workspace, workspaceSize, executor, stream)`，产品表明确支持 Atlas A2/A3。`op_host/CMakeLists.txt` 以 `ACLNNTYPE aclnn` 接入自动生成 ACLNN inner bridge；固定仓没有该生成实现源码，因此空指针/基础 dtype 校验和 executor 封装以文档为框架边界，算子拥有的 OpDef、Tiling、POD、Kernel 与构建均已继续闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段接口按公开契约检查 `x/scalar/out` 非空、支持 dtype、`x/out` dtype 一致、列表内 dtype 一致、每个 Tensor 维数不超过 8、`out` 每项 shape size 为 1。OpDef 对动态输入和输出声明 `AutoContiguous()`；输入非连续处理属于生成框架边界，输出公开约束为不支持非连续。Tiling 的 `ForeachReduceTiling::Init` 再枚举最多 256 个动态输入、记录 storage 元素数并校验列表 dtype；`CheckShapeInfo` 校验列表长度、维数、输出数量/dtype/shape。

### 参数、Workspace 与生命周期

`Tiling4ForeachNormTiling` 进入按 priority 递增的模板注册表。A2/A3 上 `ForeachNormMembaseTiling::IsCapable()==true`，产生 key `1/2/4`、`blockDim=needCoreNum`、schedule mode 1 和 `ForeachReduceTilingData`。workspace 为 `(64+256)*sizeof(float) + 16 MiB = 1,280 B + 16 MiB`；Kernel 通过 `SetSysWorkspace` 后取得 user workspace，前至多 320 个 FP32 槽承载 Stage-1 分片值。workspace 与 executor 由第一段返回，调用者在第二段执行结束前保持有效；stream 的具体 task 封装属于生成框架边界。

### 启动图与失败传播

```text
aclnnForeachNormGetWorkspaceSize
  -> 自动 ACLNN executor（固定仓不可见）
  -> ForeachNorm / Tiling4ForeachNormTiling
  -> ForeachNormMembaseTiling(priority=10000)
  -> 单节点 foreach_norm<<<needCoreNum, scheduleMode=1>>>(inputs, scalar, output, workspace, tiling)
       Stage 1: 每核对负责的 Tensor 片段归约 -> userWS[coreMiddleOffset + localTensor]
       CrossCoreSetFlag/WaitFlag
       Stage 2: 核 blockIdx 以 needCoreNum 为步长归约各 Tensor 的分片值 -> output
```

单 Kernel 内的 userWS 边为 FP32、连续 1 元素/分片；producer 是 `SingleTensorProcess`，consumer 是 `CopyInStage2`，全核 `CrossCoreWaitFlag` 保证可见性。Tiling 任一校验、raw tiling/workspace 指针获取失败即返回 `GRAPH_FAILED` 并阻断 launch；Device 的 `workspace==nullptr` 或系统 workspace 解析失败则直接返回且不写输出；未知 key 无 body 分支。

## Route 与参数设计

### 分核与资源计算

对 dtype 大小 `s`，`elementsPerBlock=32/s`，每个 Tensor 贡献 `ceil(numel/elementsPerBlock)` 个 32B block；`needCoreNum=min(max(totalBlockCount,1), AIVCoreNum)`。Host 按 block 数近似均分，记录每核起止 Tensor 与元素 offset。每个 Tensor 被切成的片段数写入 `tensorMiddleCountList`，其 workspace 前缀和写入 `tensorMiddleStartList`；每核 workspace 写入起点为 `coreMiddleOffsetList`。

设平台 UB 为 `U`，POD 序列化大小为 `D`，`A=U-D-16384`：FP32 的 `I=floor((A/2)/32)*32` 字节；FP16 的 `I=floor((A/10/2)/32)*32` 字节；BF16 改为 64B 向下对齐。16-bit Kernel 的每片最大元素数为 `9I/2`，FP32 为 `I/4`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 16-bit L1 | key 1/4；`ForeachNormND<half\|bfloat16_t,float,1>` | 非 regbase；x=FP16/BF16；Device `static_cast<int>(scalarVal)==1` | 同 dtype R-002 |
| 2 | R-002 16-bit L2 | key 1/4；`ForeachNormND<half\|bfloat16_t,float,2>` | 非 regbase；x=FP16/BF16；Device 上述条件为 false | 无 fallback |
| 3 | R-003 FP32 L1 | key 2；`ForeachNormND<float,float,1>` | 非 regbase；x=FP32；Device `static_cast<int>(scalarVal)==1` | R-004 |
| 4 | R-004 FP32 L2 | key 2；`ForeachNormND<float,float,2>` | 非 regbase；x=FP32；Device 上述条件为 false | 无 fallback |

```text
priority 10000 ForeachNormMembaseTiling
  if IsRegbaseSocVersion: GRAPH_PARAM_INVALID -> priority 30000（A2/A3 不发生）
  else dtype FP16 -> key 1; FP32 -> key 2; BF16 -> key 4; other -> key 0
foreach_norm
  if key in {1,4}: int(scalarVal)==1 ? R-001 : R-002
  if key == 2:      int(scalarVal)==1 ? R-003 : R-004
  otherwise: no body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 1 FP16；4 BF16；model 1 | A2/A3，合法 TensorList，`int(scalar)==1` | 单 AIV `foreach_norm`，两阶段跨核归约 | `ForeachNormND<T,float,1>`；16-bit `InnerComputer` | 9I-byte 双 VECIN + 2I-byte FP32 Queue + 双 32B VECOUT + 1KiB TBuf；userWS + 全核 flag | `DataCopyPad` + 有效 `dataCount`；空 Tensor 写 0 |
| R-002 | 1/4；model 2 | 同上但 `int(scalar)!=1` | 同上 | `ForeachNormND<T,float,2>` | 同 R-001 | 尾片平方和后 sqrt；无一般 p fallback |
| R-003 | 2；model 1 | A2/A3，x=FP32，`int(scalar)==1` | 同上 | `ForeachNormND<float,float,1>` specialization | I-byte双 VECIN + 双32B VECOUT + 1KiB TBuf；无 cast Queue | 尾片按有效长度归约 |
| R-004 | 2；model 2 | A2/A3，x=FP32，`int(scalar)!=1` | 同上 | `ForeachNormND<float,float,2>` specialization | 同 R-003 | 平方和后 sqrt；无一般 p fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台执行族 | `!IsRegbaseSocVersion(context_)` | membase body 使用传统 Vector API 与显式 PipeBarrier | 全部 | `GRAPH_PARAM_INVALID` 后尝试 regbase | `ForeachNormMembaseTiling::IsCapable` |
| 32B 分核 | 每 Tensor block=`ceil(numel/(32/s))` | 避免按未对齐片段分核，并给 `DataCopyPad` 尾块 | 全部 | Tiling 重新分配；无 Device fallback | `AssignDataToEachCore` |
| UB 保留 | `U-D-16384`，16-bit 再 `/10`，最终 `/2` | 覆盖双 Queue、cast scratch、TBuf 与保留量 | 全部 | tile 变小；若算术下溢固定仓无显式保护 | `DivideUbMemory` |
| dtype key | dtype 仅 FP16/FP32/BF16 | router 只实例化 key 1/2/4 | 全部 | key 0 / 无 body；正常应由 OpDef 拒绝 | `GetTilingKeyVal`, `foreach_norm` |
| 16-bit BF16 guard | 非 NPU arch 3003/3113 才编译 BF16 body | Kirin 不支持 BF16；A2/A3 body可达 | R-001/R-002 | 被编译 guard 排除的平台无执行 | `foreach_norm` `#if` |
| calc partial cache | `cachedPartialCountMax=1024/4-32/4=248` | 分批缓存 partial，满时先 ReduceSum 再加到 slot 0 | 全部 | 达上限即折叠，不溢出 TBuf | `SingleTensorProcess` |

### POD/ABI

`BEGIN_TILING_DATA_DEF` 的底层宏来自仓外 CANN 头 `register/tilingdata_base.h`，故固定仓可确认字段顺序、类型与数组长度，但 `sizeof/alignof/offsetof` 不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachReduceTilingData` | `inputsTensorUbSize` | uint64_t/64 | 第1字段；精确 offset 不可见 | 字节 | `DivideUbMemory` | `Init` 分配 Queue、求 maxDataCount | 32B/64B 向下对齐 |
| 同上 | `needCoreNum` | uint32_t/32 | 第2字段 | core | `GetNeedCoreNum` | Stage2 步长 | 至少1，不超过 AIV 核数 |
| 同上 | `totalTensorCount` | uint32_t/32 | 第3字段 | Tensor | `Init` 枚举数 | Stage2 上界 | 0–256 |
| 同上 | `tensorDataCountList[256]` | uint64_t[] | 第4字段 | 元素/Tensor | storage shape size | Stage1/空域/地址 | 空 Tensor 为0 |
| 同上 | `tensorStartList[64]` | uint16_t[] | 第5字段 | Tensor index/core | `AssignDataToEachCore` | `ParseTilingData` | active core 项有效 |
| 同上 | `tensorEndList[64]` | uint16_t[] | 第6字段 | Tensor index/core | 同上 | 同上 | 闭区间 |
| 同上 | `tensorStartOffsetList[64]` | uint64_t[] | 第7字段 | 元素 | 同上 | 首 Tensor cursor | active core 项有效 |
| 同上 | `tensorEndOffsetList[64]` | uint64_t[] | 第8字段 | 元素 | 同上 | 尾 Tensor cursor | 闭区间尾 |
| 同上 | `tensorMiddleCountList[256]` | uint16_t[] | 第9字段 | partial/Tensor | `AssignTensorMiddleCountList` | Stage2搬运/归约长度 | 每 Tensor 至少1计数槽 |
| 同上 | `tensorMiddleStartList[256]` | uint16_t[] | 第10字段 | FP32 slot | partial 前缀和 | Stage2 workspace offset | 总槽≤core+Tensor |
| 同上 | `coreMiddleOffsetList[64]` | uint16_t[] | 第11字段 | FP32 slot | 每核覆盖 Tensor 数前缀和 | Stage1 workspace offset | active core 项有效 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/R-002 | 1/4 + scalar model | `foreach_reduce_tiling_func.cpp`: `ForeachReduceTiling` | `ForeachReduceTilingData` | `op_kernel/foreach_norm.cpp`: `foreach_norm` | `op_kernel/foreach_norm.h`: `ForeachNormND`, 16-bit `InnerComputer`, `NormAdapter` | `op_kernel/CMakeLists.txt`: A2/A3 compute units |
| R-003/R-004 | 2 + scalar model | 同上 | 同上 | 同上 | `ForeachNormND<float,float,...>` specialization | 同上 |

## Kernel 实现

### 公共执行流程

entry 先建立系统/user workspace，读取 scalar 并按 key 与 modelCode 实例化模板。每核 Stage 1 遍历 Tiling 分配的 Tensor 闭区间，将输入切成不超过 `maxDataCount` 的 tile；每 tile 变换并归约成 FP32 partial，多个 tile 在 `calcBuf` 中累加，最终每个“核×Tensor片段”写一个 FP32 到 userWS。全核 flag 后，核以 `blockIdx + n*needCoreNum` 领取 Tensor，搬入该 Tensor 全部 partial，二次求和并在 L2 路线开方，cast 后写单元素输出。

### Route R-001：16-bit L1

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_reduce_tiling_def.h` | `ForeachNormMembaseTiling::IsCapable` | `!IsRegbaseSocVersion(context_)` | membase |
| Tiling | `foreach_reduce_tiling_func.cpp` | `GetTilingKeyVal` | FP16→1，BF16→4 | entry dtype branch |
| Router | `foreach_norm.cpp` | `foreach_norm` | `static_cast<int>(scalarVal)==1` | model 1 |
| Body | `foreach_norm.h` | `ForeachNormND<T,float,1>` | T=half/bfloat16_t | L1 body |

#### Device 数据流

##### 函数级调用链

```text
foreach_norm -> ForeachNormND<T,float,1>::Init/Process
  -> SingleTensorProcess -> CopyInStage1 -> InnerComputer::SquareAndReduceRound1Compute
  -> NormAdapter<float,1>::AbsAndPowerAdapt -> ReduceSum -> userWS partial
  -> CrossCoreSetFlag/WaitFlag
  -> CopyInStage2 -> ReduceRound2AndSqrtCompute -> no sqrt -> Cast -> output
```

##### 源码等价伪码

```text
for tensor in [tensorStart,tensorEnd]:
  if numel==0: continue
  segment=[startOffset/endOffset adjusted]
  for tile of at most 9*I/sizeof(T):
    DataCopyPad GM->dataQueue; cast chunks of I/4 elements to FP32
    abs; ReduceSum; accumulate partials in calcBuf slot0
  DataCopyPad calcBuf[0] -> userWS[coreMiddleOffset+localTensor]
cross_core_barrier()
for tensor=blockIdx; tensor<totalTensorCount; tensor+=needCoreNum:
  if empty: output=0
  else: copy its FP32 partials; ReduceSum; Cast RINT to T; copy sizeof(T) to output
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | TensorList地址 + cursor + tile*maxDataCount | 当前 tile `dataCount` | T/ND | caller | CopyInStage1 | caller；Kernel期有效 | DataCopyPad尾块 |
| `dataQueue` | UB Queue槽 | depth2 × 9I B | T，Stage2按FP32视图 | MTE2 | Vector | TPipe | 双缓冲；Stage1/2复用 |
| `float32Queue` | UB独立槽 | depth1 × 2I B | FP32 | Cast | Reduce | TPipe；Process末释放 | 16-bit路线专用 |
| `calcBuf` | slot0；partial从8起 | 1024 B | FP32 | tile reduce | workspace copy | TPipe | slot0运行累加，248 partial批折叠 |
| userWS | `coreMiddleOffset+localTensor` / `tensorMiddleStart` | ≤320×4 B | FP32 | Stage1 | Stage2 | executor workspace | 全核 barrier后读取 |
| outQueue | UB Queue槽 | depth2×32 B | T | final cast | MTE3 | TPipe | 每输出释放 |
| output GM | TensorList第i项首元素 | sizeof(T) | T/ND | MTE3 | caller | caller | 只写1元素 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 18I B"]:2
  castQ["float32Queue · depth=1 · 2I B"]:1
  outQ["outQueue · depth=2 · 64 B"]:1
  calc["calcBuf · 1024 B"]:4
```

##### 字节账与复用顺序

```text
UB_total = 2*(9I) + 2I + 2*32 + 1024 = 20I + 1088 B
I <= floor((U-D-16384)/20) 后按 dtype 对齐
dynamic_ub = N/A（AIV 固定 TPipe 分配）
workspace_total = 1280 B user + 16 MiB system
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataQueue槽 | CopyInStage1 | SquareAndReduceRound1 | FreeTensor后下一tile | EnQue/DeQue/FreeTensor | depth2容量按9I分配 |
| calcBuf slot0/8..255 | tile归约 | partial批归约/最终MTE3 | 折叠后下一批 | PipeBarrier + V_MTE3 | cachedPartial≤248 |
| userWS partial槽 | Stage1 MTE3 | Stage2 MTE2 | Kernel结束后 | MTE3_MTE2 + CrossCoreWaitFlag | 唯一核×Tensor片段槽 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| tile队列 | EnQue/DeQue/FreeTensor | MTE2 | Vector | dataQueue槽 | DataCopyPad仅以dataCount参与归约 |
| cast/reduce | PIPE_V、V_S、S_V | Vector reduce | scalar GetValue/SetValue | FP32 partial | T升FP32；无效尾 lane 不进入dataCount |
| workspace写 | V_MTE3、MTE3_MTE2 | Vector | MTE3/MTE2 | calcBuf/userWS | 每次仅sizeof(float) |
| 跨核 | CrossCore flag 1 | 所有Stage1 | 所有Stage2 | userWS全体partial | scheduleMode=1 |
| 输出 | Cast `CAST_RINT` | FP32 L1和 | MTE3 | 单元素 | 输出降到T；空Tensor显式0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_reduce_tiling_func.cpp` | `RunBigKernelTiling`, `DivideUbMemory` | 分核、UB、workspace、key |
| Kernel entry/router | `foreach_norm.cpp` | `foreach_norm` key1/key4 | scalar与dtype到模板 |
| Device body/helper | `foreach_norm.h` | `ForeachNormND<T,float,1>`, `InnerComputer`, `NormAdapter` | cast→abs→两级sum→RINT输出 |

### Route R-002：16-bit L2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_reduce_tiling_func.cpp` | `GetTilingKeyVal` | FP16→1，BF16→4 | 16-bit entry |
| Router | `foreach_norm.cpp` | `foreach_norm` | `static_cast<int>(scalarVal)!=1` | model 2 |
| Body | `foreach_norm.h` | `ForeachNormND<T,float,2>` | T=half/bfloat16_t | L2 body |

#### Device 数据流

##### 函数级调用链

```text
foreach_norm -> ForeachNormND<T,float,2>::Process
  -> CopyInStage1 -> Cast FP32 -> Mul(x,x) -> ReduceSum -> userWS
  -> CrossCore barrier -> CopyInStage2 -> ReduceSum -> Sqrt -> Cast RINT -> output
```

##### 源码等价伪码

```text
按R-001相同分核/切片/地址；逐tile将T升FP32，原地平方后求和；
partial写userWS；全核同步；二级求和后sqrt；RINT降回T；空Tensor写0。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/dataQueue | 同R-001 | depth2×9I B | T | MTE2 | Cast/Mul | TPipe | DataCopyPad tail |
| float32Queue | 独立UB | depth1×2I B | FP32 | Cast/Mul | ReduceSum | TPipe | tile间复用 |
| calcBuf/userWS | slot0与partial槽；workspace前缀 | 1024 B；≤1280 B | FP32 | ReduceSum | 二级ReduceSum/Sqrt | Kernel/workspace | barrier保护 |
| outQueue/output | TensorList第i项 | depth2×32 B；sizeof(T)有效 | T | Cast RINT | MTE3 | TPipe/caller | 单元素 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 18I B"]:2
  castQ["float32Queue · depth=1 · 2I B"]:1
  outQ["outQueue · depth=2 · 64 B"]:1
  calc["calcBuf · 1024 B"]:4
```

##### 字节账与复用顺序

```text
UB_total = 20I + 1088 B
dynamic_ub = N/A
workspace_total = 1280 B + 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| float32Queue tensor | Cast/Mul | ReduceSum | 下一cast chunk | Queue生命周期+PIPE_V | chunk≤I/4 |
| calcBuf partial cache | tile ReduceSum | 批ReduceSum/Add slot0 | 清零计数后的下一批 | PIPE_V | ≤248缓存项 |
| userWS | Stage1 | Stage2 | Kernel结束 | CrossCoreWaitFlag | offset前缀不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Stage1 | Cast NONE→Mul→ReduceSum | MTE2 | Vector | tile | FP32平方/累加；有效dataCount |
| 跨核 | CrossCore flag | partial写 | Stage2 | userWS | 所有partial可见 |
| Stage2 | ReduceSum→Sqrt→Cast RINT | MTE2 | MTE3 | 单输出 | FP32开方后降精度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_reduce_tiling_func.cpp` | `ForeachReduceTiling` | 同R-001参数布局 |
| Kernel entry/router | `foreach_norm.cpp` | modelCode默认2 | 非1值走L2 |
| Device body/helper | `foreach_norm.h` | `NormAdapter<float,2>` | 原地Mul、二级sum、Sqrt |

### Route R-003：FP32 L1

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_reduce_tiling_func.cpp` | `GetTilingKeyVal` | dtype=FLOAT | key2 |
| Router | `foreach_norm.cpp` | `foreach_norm` | `int(scalarVal)==1` | model1 |
| Body | `foreach_norm.h` | `ForeachNormND<float,float,1>` | FP32 specialization | 无cast L1 |

#### Device 数据流

##### 函数级调用链

```text
foreach_norm -> ForeachNormND<float,float,1>::Process
  -> CopyInStage1 -> InnerComputer<float>::SquareAndReduceRound1Compute
  -> Abs -> ReduceSum -> userWS -> barrier -> ReduceSum -> Copy(no sqrt) -> output
```

##### 源码等价伪码

```text
逐tile CopyIn FP32；在dataLocal原地Abs并ReduceSum；标量partial写calcBuf；
跨tile/跨核二级求和；最终Copy一个FP32结果；tail仅按dataCount参与。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dataQueue | UB Queue槽 | depth2×I B | FP32 | MTE2 | Abs/Reduce | TPipe | 双缓冲 |
| calcBuf | slot0与8..255 | 1024 B | FP32 | Reduce | workspace copy | TPipe | partial批复用 |
| userWS | partial前缀 | ≤1280 B | FP32 | Stage1 | Stage2 | executor | barrier |
| outQueue | UB Queue槽 | depth2×32 B | FP32 | Copy | MTE3 | TPipe | 4B有效 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2I B"]:2
  calc["calcBuf · 1024 B"]:1
  outQ["outQueue · depth=2 · 64 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2I + 64 + 1024 = 2I + 1088 B
I <= floor((U-D-16384)/2) aligned to 32B
dynamic_ub = N/A
workspace_total = 1280 B + 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataLocal | CopyIn | Abs/ReduceSum原地 | FreeTensor后 | Queue+PIPE_V | dataCount≤I/4 |
| calcBuf/userWS | Reduce | Stage2 | 批折叠/Kernel结束 | barriers | 同R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Stage1 | PIPE_V、V_S/S_V | Abs/Reduce | 标量partial | dataLocal/calcBuf | 全程FP32；有效dataCount |
| Stage1→2 | MTE3_MTE2 + CrossCore | userWS写 | Stage2搬入 | partial | 跨核可见 |
| 输出 | V_MTE3 | Copy结果 | output | 1个FP32 | 无降精度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_reduce_tiling_func.cpp` | key2、DivideUbMemory | FP32 tile字节 |
| Kernel entry/router | `foreach_norm.cpp` | key2/model1 | FP32 L1实例 |
| Device body/helper | `foreach_norm.h` | `InnerComputer<float,P,modelCode>` | 避免cast Queue |

### Route R-004：FP32 L2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_reduce_tiling_func.cpp` | `GetTilingKeyVal` | dtype=FLOAT | key2 |
| Router | `foreach_norm.cpp` | `foreach_norm` | `int(scalarVal)!=1` | model2 |
| Body | `foreach_norm.h` | `ForeachNormND<float,float,2>` | FP32 specialization | L2 |

#### Device 数据流

##### 函数级调用链

```text
foreach_norm -> ForeachNormND<float,float,2>::Process
  -> CopyIn -> Mul in-place -> ReduceSum -> userWS -> barrier -> ReduceSum -> Sqrt -> output
```

##### 源码等价伪码

```text
与R-003相同分片和内存；将Abs替换为原地Mul(x,x)，二级和后对1元素Sqrt，再写FP32。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| dataQueue | tile*I/4 | depth2×I B | FP32 | MTE2 | Mul/Reduce | TPipe | 原地平方 |
| calcBuf | slot0/partial cache | 1024 B | FP32 | Reduce | workspace | TPipe | 248项批折叠 |
| userWS/outQueue | partial前缀/输出首元素 | ≤1280 B / depth2×32B | FP32 | Stage1/Stage2 | Stage2/MTE3 | executor/TPipe | barrier保护 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["dataQueue · depth=2 · 2I B"]:2
  calc["calcBuf · 1024 B"]:1
  outQ["outQueue · depth=2 · 64 B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2I + 1088 B
dynamic_ub = N/A
workspace_total = 1280 B + 16 MiB
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| dataLocal | CopyIn | Mul/ReduceSum | FreeTensor后 | PIPE_V/Queue | 仅有效dataCount |
| userWS | Stage1 | Stage2 | Kernel结束 | CrossCoreWaitFlag | 唯一partial槽 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Stage1 | Mul+ReduceSum/PIPE_V | input tile | partial | UB | FP32；tail=dataCount |
| Stage2 | ReduceSum+Sqrt | workspace | output | UB/outQueue | FP32全过程 |
| 跨核/输出 | CrossCore + V_MTE3 | partial/final | Stage2/GM | userWS/output | empty显式0 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_reduce_tiling_func.cpp` | `ForeachReduceTiling` | key2、分核、workspace |
| Kernel entry/router | `foreach_norm.cpp` | key2 else | model2 |
| Device body/helper | `foreach_norm.h` | FP32 `NormAdapter<...,2>` | 原地平方、两级sum、sqrt |

## 算子特定优化

### Trick T-001：1 KiB TBuf 中的分批 partial 折叠

#### 触发与源码依据

- 适用 Route：R-001–R-004。
- 触发条件：一个核负责的 Tensor 片段超过单 tile，且产生多个 tile partial。
- Golden 来源：[`foreach/foreach_norm/op_kernel/foreach_norm.h:321-353`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_norm/op_kernel/foreach_norm.h#L321-L353)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint16_t tempLocalCount = byteLen / sizeof(P);
uint16_t partialStartOffset = BYTE_BLOCK / sizeof(P);
uint16_t cachedPartialCountMax = tempLocalCount - partialStartOffset;
uint16_t cachedPartialCount = 0;
bool hasAccumulator = false;
LocalTensor<P> tempLocal = calcBuf.Get<P>(tempLocalCount);
```

#### 机制与设计原因

slot 0 保存跨批运行累加器，slot 8 起缓存 tile partial；缓存达到 248 项或到最后一 tile 时，先对缓存做一次 `ReduceSum`，再用单元素 `Add` 合入 slot 0。这样 calc TBuf 固定为 1 KiB，而不随超长 Tensor 的 tile 数增长；关键边界来自 `byteLen=1024`、`BYTE_BLOCK=32` 和 `P=float`。

#### 迁移边界

- 适用：多 tile 的单 Tensor 分片归约。
- 前提：partial 为 FP32、slot 0 与 32B 后缓存区布局保持不变，折叠前保留 `PIPE_V`。
- 失效：若 partial dtype/算子不是可结合的求和，必须重新证明布局和归约次序，不能直接套用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；从源码边界可知，若不在 248 项处折叠会越过 1 KiB `calcBuf`，但本次未做运行验证。

## 模板与实现对比

| 对比维度 | R-001 16-bit L1 | R-002 16-bit L2 | R-003 FP32 L1 | R-004 FP32 L2 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key1/4，int(s)=1 | key1/4，其余 | key2，int(s)=1 | key2，其余 | dtype由Host key，范数模型由Device scalar |
| 模板与 dtype | T=half/bf16, model1 | T=half/bf16, model2 | float, model1 | float, model2 | 16-bit升FP32；FP32 specialization |
| 循环与分核 | 两级归约 | 同左 | 同左 | 同左 | 32B block均分和workspace partial |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 20I+1088B | 同左 | 2I+1088B | 同左 | 16-bit需要cast Queue |
| 内存总占用与复用 | 1KiB partial cache | 同左 | 同左 | 同左 | 固定cache批折叠 |
| 搬运/Vector/SIMT/NDDMA/通信 API | Pad/Abs/ReduceSum/Cast | Pad/Mul/ReduceSum/Sqrt/Cast | Pad/Abs/ReduceSum | Pad/Mul/ReduceSum/Sqrt | 仅AIV Vector |
| 同步、精度与 tail | FP32累加，RINT输出 | 同左+sqrt | FP32 | FP32+sqrt | queue/pipe/全核flag |
| fallback/失败边界 | 空Tensor→0 | 非1统一到此 | 空Tensor→0 | 非1统一到此 | 无一般p实现 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持 | 公开支持 | 公开支持，但见独立 `_950` 卡 |
| Host/Launcher | 共用 membase priority 10000 | 共用 membase priority 10000 | regbase priority 30000 |
| Kernel/资源 | `foreach_norm.h`，最多64核POD，传统Vector API | 同A2 | arch35 MicroAPI，最多80核POD，UB公式不同 |

## 限制与失败边界

- 固定 Kernel 只实现 L1/L2：`int(float(scalar))==1` 走 L1，其余走 L2；这比公开一般 p-norm 公式窄，不能推测其他 p 的结果。
- ACLNN 文档声明 scalar 可为 DOUBLE，但 OpDef/binary config 仅实例化 FLOAT32/INT64；DOUBLE 到实际 scalar Tensor 的生成桥接固定仓不可见。
- 文档称支持空 Tensor；A2/A3 Tiling 允许列表长度 0，并以一个核启动，但生成 ACLNN 对空列表的封装细节不可见。
- `U-D-16384` 使用无符号算术且无显式下溢检查；实际平台 UB 足够性只由平台配置保证，固定仓未给出数值下限证明。
- Device 遇到 null workspace、系统 workspace 解析失败或未知 key 直接返回，不写错误码到 Host；失败传播依赖 runtime launch 边界。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_norm/docs/aclnnForeachNorm.md` | 两段式原型、产品表 | 公开性、参数、错误契约 |
| 入口实现 | `foreach/foreach_norm/op_host/CMakeLists.txt` | `ACLNNTYPE aclnn` | 自动生成 ACLNN inner bridge 的活动构建边界 |
| 入口实现 | `foreach/foreach_norm/op_host/foreach_norm_def.cpp` | `ForeachNorm`, `OP_ADD` | dtype/layout/平台和AutoContiguous；生成ACLNN边界 |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority升序尝试与fallthrough |
| Host/Tiling/Launcher | `common/inc/op_host/tiling_util.h` | `IsRegbaseSocVersion` | A2/A3与RegBase执行族判定 |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_reduce_tiling_def.h` | POD、`ForeachNormMembaseTiling` | ABI字段、平台selector |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_reduce_tiling_func.cpp` | `ForeachReduceTiling`, registrations | 校验、分核、UB、workspace、key |
| Kernel entry/body | `foreach/foreach_norm/op_kernel/foreach_norm.cpp` | `foreach_norm` | workspace、scalar、key router |
| Kernel entry/body | `foreach/foreach_norm/op_kernel/foreach_norm.h` | `ForeachNormND`, helpers | 完整Device body、内存与同步 |
| 构建接线 | `foreach/foreach_norm/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3活动Kernel选择 |

每行只写一个文件，并确保该文件存在于 Frontmatter `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 两阶段 foreach reduce 的起止Tensor/offset与partial前缀 | 字段顺序、64核/256 Tensor数组和元素/slot单位 | `ForeachReduceTilingData` |
| 调用顺序 | Stage1→全核flag→Stage2 | 不得提前读userWS | `ForeachNormND::Process` |
| workspace/内存 | 320个FP32 partial槽+16MiB系统区 | partial offset唯一，UB depth必须计入 | `RunBigKernelTiling`, `Init` |
| 同步 | Queue、PipeBarrier、硬事件与CrossCore flag | MTE3写后方可Stage2 MTE2读 | `SingleTensorProcess`, `Process` |
| 精度与尾块 | 16-bit升FP32，两级sum，RINT回写；Pad尾仅按有效数归约 | 模型1 Abs、模型2 Mul+Sqrt顺序 | `InnerComputer`, `NormAdapter` |
| 失败边界 | 校验失败阻断；null workspace/未知key无输出 | 不得把非1 scalar解释为一般p | `CheckShapeInfo`, `foreach_norm` |
