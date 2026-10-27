---
type: Operator
title: "ForeachMaximumScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachMaximumScalarV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/docs/aclnnForeachMaximumScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_host/op_api/aclnn_foreach_maximum_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_host/op_api/foreach_maximum_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_host/foreach_maximum_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_host/arch35/foreach_maximum_scalar_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_host/config/ascend950/foreach_maximum_scalar_binary.json
    role: implementation
created_at: "2026-08-25T15:47:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMaximumScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnForeachMaximumScalarV2GetWorkspaceSize` / `aclnnForeachMaximumScalarV2`；V2 是 950 实际支持的最新公开版本。 |
| 计算语义 | 对列表中每个张量逐元素计算 `y_i[j] = max(x_i[j], scalar)`；浮点路径显式传播 NaN 并处理有符号零。 |
| 输入输出 | `x/out` 为 ND TensorList，FLOAT16/FLOAT32/INT32/BFLOAT16；V2 `aclScalar` 经 API 转为单元素 Tensor。 |
| 执行类型 | 单 Kernel、SIMT；未发现 Cube/AIC 或 MIX_CV 可达节点。 |
| Kernel 路线 | R-001 FLOAT32；R-002 FLOAT16/BFLOAT16 升至 FP32；R-003 INT32。 |
| 核心资源 | GM TensorList、GM POD、每 block 1024 SIMT threads；无显式 AIV Queue/TBuf、无 operator workspace。 |
| 来源闭包 | source_closed：ACLNN V2→框架前处理→L0 launcher→arch35 Tiling/POD/key→SIMT entry/body→ascend950 构建/config 闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`，每个输出有效元素为 `y_i[j]=maximum(x_i[j], scalar)`。FLOAT16/BFLOAT16 先转换为 FP32 比较再降回存储 dtype；FLOAT32 原精度比较；INT32 使用 `a>=b?a:b`。浮点 `SimtMaximum` 若任一操作数为 NaN 就返回该 NaN；相等且 `a==0` 时返回 `a+b`，从而由加法结果决定 ±0。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每 Tensor 0–8 维、ND | FLOAT16/FLOAT32/INT32/BFLOAT16，列表内一致 | TensorList 可为空；公开 V2 文档将非连续输入标为不支持，API 实现仍建立 `Contiguous` 节点。950 Tiling 最多 256 个 Tensor。 |
| `scalar` | `aclScalar` | FLOAT/DOUBLE、FLOAT16/DOUBLE 或 INT32/INT64，依 x dtype | 必须落入目标 dtype 数值范围；BF16 x 转为 FP32 scalar Tensor，其它转为 x dtype。 |
| `out` | TensorList；API 源码要求与 x 对应 shape 相等；ND | 与 x 一致 | 非连续输出不支持；Tiling/Kernel 仅按 x 的每 Tensor 元素数写有效域。 |

### 入口与版本

旧版 `aclnnForeachMaximumScalar` 的公开产品表明确 950 不支持；V2 产品表明确 Ascend 950PR/950DT 支持，且 V2 将 scalar 从 `aclTensor` 更新为 `aclScalar`，因此 950 只提取 V2。第一段生成 executor/workspace，第二段用 `CommonOpExecutorRun` 提交到用户 stream。

## Host/Launcher 执行流程

### 参数校验与前处理

API 顺序为：创建 executor；检查 `x/scalar/out` 非空；`GetDtypeSupportList` 接受 regbase 平台；校验 TensorList dtype、scalar dtype 配对、输入/输出 shape、维度和非私有格式。空列表直接返回 0 workspace 且不添加 Kernel。非空时逐 x Tensor 调 `l0op::Contiguous`，检查 scalar 值范围，以 `ConvertToTensor` 生成单元素 Tensor，再由 `l0op::ForeachMaximumScalarV2` 添加 `ForeachMaximumScalar` AICore launcher。

### 参数、Workspace 与生命周期

arch35 Tiling 写 `workspace[0]=0`，Kernel 的 workspace 参数未读取。ACLNN 返回的聚合 workspace 仍可能包含前置 Contiguous/标量转换框架节点，其公式和 slice offset 在固定仓不可见。Tiling 另设 `localMemorySize=ubSize-128 KiB`；SIMT body 没有显式 shared/dynamic UB 对象或对该空间的读取。executor、转换 Tensor 和前处理结果由框架持有到同一 executor 在 stream 上执行。

### 启动图与失败传播

核心启动图是单节点：L0 `ADD_TO_LAUNCHER_LIST_AICORE`→`ForeachMaximumScalarTilingFunc`→`foreach_maximum_scalar<schMode>`。Host 以 `blockDim=needCoreNum` 启动；每个 AIV block 内 `VF_CALL` 产生 1024 个 SIMT threads。前置 Contiguous/ConvertToTensor 与主节点由 executor 保持顺序；任一 API 检查、平台信息、Tensor 数、dtype、POD、workspace 指针或 launcher 添加失败都会在相应阶段返回并阻断 Kernel。单 Kernel 没有节点间 workspace/事件边。

## Route 与参数设计

### 分核与资源计算

Tiling 读取 `aivCoreNum` 与 UB 大小，枚举动态输入实例数 `tensorNum`，要求 `tensorNum<=256`；将每个 storage shape 元素数写入 `tensorElements[i]` 并累加 `totalElements`。若总元素为 0，`needCoreNum=blockDim=1`；否则 `needCoreNum=min(aivCoreNum, ceil(totalElements/1024))`。每 block 固定 `THREAD_NUM=1024`，全局线程号 `tid=blockIdx*threadNum+threadIdx`，总步长 `stride=threadNum*blockNum`。每个线程按同一 tid/stride 扫描列表中每个 Tensor。

workspace 为 0；local-memory launch 参数为 `ubSize-131072` B。Kernel body 只从 GM 读取 scalar、POD 和 TensorList 指针，不调用 `InitBuffer`，故显式 SIMT dynamic/shared UB 占用为 N/A。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FLOAT32 | schMode 0→`Process<float,float>` | dataType=DT_FLOAT；Host 设置 template key FLOAT | 其它 dtype 继续 Host `else if` |
| 2 | R-002 低精度升 FP32 | schMode 1→`Process<half,half>`；schMode 3→`Process<bfloat16_t,float>` | dataType=DT_FLOAT16 或 DT_BF16 | INT32 进入 R-003；其它 dtype 失败 |
| 3 | R-003 INT32 | schMode 2→`Process<int32_t,int32_t>` | dataType=DT_INT32 | 其它 dtype Host 返回 GRAPH_FAILED |

```text
DT_FLOAT   -> template key schMode=0 -> Process<float,float>
DT_FLOAT16 -> template key schMode=1 -> Process<half,half>
DT_INT32   -> template key schMode=2 -> Process<int32_t,int32_t>
DT_BF16    -> template key schMode=3 -> Process<bfloat16_t,float>
其它       -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FLOAT32 | schMode 0 | 950、FLOAT32、tensorNum≤256、API/Tiling 成功 | 单 SIMT Kernel；blockDim×1024 threads | `Process<float,float>`→`OpForeachMaximumScalarSimt<float,float>` | 直接 GM load/compute/store；无显式 UB/barrier | `idx<count` 线程边界；空 Tensor `continue` |
| R-002 低精度升 FP32 | schMode 1/3 | 950、FLOAT16 或 BF16；对应 scalar Tensor half 或 float | 同上 | `Process<half,half>` / `Process<bfloat16_t,float>`；`ComputeType<T>=float` | GM→寄存器 FP32→GM；无显式 UB/barrier | 同一 grid-stride tail；无其它 fallback |
| R-003 INT32 | schMode 2 | 950、INT32 | 同上 | `Process<int32_t,int32_t>`→int overload | GM int32 寄存器路径 | 同一 tail；无其它 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数 | `tensorNum<=256` | POD `tensorElements[256]` 固定长度 | 全部 | GRAPH_FAILED | `ForeachMaximumScalarTilingFunc` |
| 最少每核元素 | `ceil(totalElements/1024)` | 每 block 有 1024 threads，避免总元素很少时启动所有 AIV | 全部 | 非 fallback；直接缩小 blockDim | `MIN_PER_CORE` / core split |
| 零域 | `totalElements==0`→blockDim 1 | 保证 launch 参数非零；Device 对每个 count=0 的 Tensor continue | 全部 | N/A | `ForeachMaximumScalarTilingFunc` / SIMT body |
| local memory | `SetLocalMemorySize(ubSize-128KiB)`，仅检查 `ubSize!=0` | 预留 128 KiB DCache | 全部 | `ubSize<128KiB` 未见显式下溢 guard | `DCACHE_SIZE` / `SetLocalMemorySize` |
| 线程数 | `THREAD_NUM=1024`；`LAUNCH_BOUND(1024)` | VF_CALL 固定一维 1024 threads/block | 全部 | 无其它线程配置 Route | `OpForeachMaximumScalarSimt` / `Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachMaximumScalarTilingData` | `needCoreNum` | int32_t/32 | offset 0；struct align 8，sizeof 2064 B | core | Host 设为 1 或 `min(coreNum,ceil(total/1024))` | body 不读；其值同时用于 Host `SetBlockDim` | int64→int32；被 AIV core 数约束 |
| 同上 | `tensorCount` | int32_t/32 | offset 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT 外层循环 | 0..256 |
| 同上 | `totalElements` | int64_t/64 | offset 8 | 元素 | Host 累加各 numel | body 不读；Host 用于分核 | 加法溢出未显式检查 |
| 同上 | `tensorElements` | int64_t[256] | offset 16；2048 B | 每 Tensor 元素 | Host `storageShape.GetShapeSize()` | SIMT `count=tensorElements[t]` | count=0 时跳过 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FLOAT32 | schMode 0 | `op_host/arch35/foreach_maximum_scalar_tiling.cpp::ForeachMaximumScalarTilingFunc` | `ForeachMaximumScalarTilingData` | `op_kernel/arch35/foreach_maximum_scalar.cpp::foreach_maximum_scalar<0>` | `foreach_maximum_scalar_simt.h::Process<float,float>` / `SimtMaximum(float,float)` | `op_kernel/CMakeLists.txt` ascend950；ascend950 binary config FLOAT32 entry |
| R-002 低精度升 FP32 | schMode 1/3 | 同上 | 同上 | entry `<1>/<3>` | `Process<half,half>` / `Process<bfloat16_t,float>` / `ComputeType` | 同上 FLOAT16/BF16 entries |
| R-003 INT32 | schMode 2 | 同上 | 同上 | entry `<2>` | `Process<int32_t,int32_t>` / int overload | 同上 INT32 entry |

## Kernel 实现

### 公共执行流程

entry 用编译期 `schMode` 选择 T/S 模板；`Process` 从 scalar GM 读取一个 S，直接以 `offsetof(ForeachMaximumScalarTilingData,tensorElements)` 构造 GM 数组指针，读取 `tensorCount`，再 `VF_CALL<OpForeachMaximumScalarSimt<T,S>>(Dim3(1024), ...)`。每个 SIMT thread 对所有 Tensor 使用同一 global tid 和 grid stride，TensorList 地址由首字偏移定位 uint64 指针表。无 Queue、TBuf、workspace、barrier 或 atomic。

### Route R-001：FLOAT32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_maximum_scalar_v2.cpp` | `CheckDtypeValid` | x=FLOAT，scalar=FLOAT/DOUBLE | FP32 scalar Tensor |
| Tiling | `op_host/arch35/foreach_maximum_scalar_tiling.cpp` | `ForeachMaximumScalarTilingFunc` | `dataType==ge::DT_FLOAT` | schMode 0 |
| Router | `op_kernel/arch35/foreach_maximum_scalar.cpp` | `foreach_maximum_scalar<schMode>` | `if constexpr(schMode==0)` | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachMaximumScalar launcher
  -> foreach_maximum_scalar<0>(x, scalarTensor, y, workspace, tiling)
    -> NsForeachMaximumScalar::Process<float,float>
      -> VF_CALL<OpForeachMaximumScalarSimt<float,float>>(Dim3(1024), ...)
        -> SimtGetTensorAddr -> SimtMaximum(float,float) -> GM store
```

##### 源码等价伪码

```text
scalarVal = *(float*)scalarGM
globalTid = blockIdx * 1024 + threadIdx
stride = 1024 * blockNum
for t in [0,tensorCount):
  if tensorElements[t] == 0: continue
  xData/yData = DecodeTensorList(x/y,t)
  for idx = globalTid; idx < tensorElements[t]; idx += stride:
    yData[idx] = SimtMaximum(xData[idx], scalarVal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `Decode(list,t)+idx` | `tensorElements[t]` | FP32/ND | API input、SIMT store | SIMT load、API output | caller | 单元素 load/store；idx guard |
| scalar GM | element 0 | 4 B | FP32 | ConvertToTensor | 每 block `Process` 读取 | executor | 读取一次并按值传入 VF |
| tiling GM | struct base + offset | 2064 B | POD | Host Tiling | Process/VF | runtime | `tensorElements` 指针常驻 VF 调用 |
| SIMT 寄存器值 | thread-local | xVal/sVal/result | FP32 | GM load/cast | SimtMaximum/store | 单线程迭代 | 无共享/别名 |

##### 分层内存布局

N/A：该 SIMT body 未调用 AIV `InitBuffer`，也未声明 shared/dynamic UB；实际替代路径是 GM 单元素 load→thread-local 值→GM 单元素 store。Host 虽设置 `localMemorySize=ubSize-128KiB`，固定 body 不显式消费该区域。

##### 字节账与复用顺序

```text
UB_total = N/A（无 AIV Queue/TBuf）
dynamic_ub = N/A（VF_CALL 未传 dynUBufSize，body 无 shared/dynamic UB）
workspace_total = 0 B operator workspace + executor 框架临时量（固定仓公式不可见）
grid_threads = needCoreNum * 1024
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `xVal/sVal/result` | 当前迭代 load/compute | 当前迭代 store | 下一 `idx+=stride` | 单线程程序顺序 | 每 idx 由唯一 global tid 命中 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM load→compute→store | 单线程程序顺序 | 当前 thread | 同一 thread | 当前 idx | FP32；`idx<count` 保护 tail |
| 线程间 | N/A，无 barrier/atomic | 唯一 idx 映射 | N/A | 各自 y 元素 | grid-stride 不重叠 |
| 浮点最大值 | `isnan` + 比较 + zero 加法 | x/scalar | store | result | 显式 NaN 传播与 ±0 路径 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_maximum_scalar_tiling.cpp` | `ForeachMaximumScalarTilingFunc` | blockDim、POD、key、workspace/local memory |
| Kernel entry/router | `op_kernel/arch35/foreach_maximum_scalar.cpp` | `foreach_maximum_scalar<0>` | schMode→float/float |
| Device body/helper | `op_kernel/arch35/foreach_maximum_scalar_simt.h` | `OpForeachMaximumScalarSimt<float,float>` | grid-stride、地址、显式 float maximum |

### Route R-002：FLOAT16/BFLOAT16 升至 FP32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_maximum_scalar_v2.cpp` | `CheckDtypeValid` / `ConvertToTensor` | x=FLOAT16 或 BF16 | half scalar Tensor 或 FP32 scalar Tensor |
| Tiling | `op_host/arch35/foreach_maximum_scalar_tiling.cpp` | dtype branch | DT_FLOAT16 / DT_BF16 | schMode 1/3 |
| Router | `op_kernel/arch35/foreach_maximum_scalar.cpp` | entry `<1>/<3>` | compile-time branch | `Process<half,half>` / `<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_maximum_scalar<1 or 3>
  -> Process<T,S>: read scalar S -> VF_CALL Dim3(1024)
    -> OpForeachMaximumScalarSimt<T,S>
      -> xVal=static_cast<float>(xData[idx])
      -> sVal=static_cast<float>(scalarVal)
      -> SimtMaximum(float,float)
      -> yData[idx]=static_cast<T>(result)
```

##### 源码等价伪码

```text
for each Tensor t and each grid-stride idx<count:
  fp32 xVal = (fp32)xData[idx]
  fp32 sVal = (fp32)scalarVal
  fp32 result = NaN-propagating maximum(xVal,sVal)
  yData[idx] = (T)result  // T=half or bfloat16_t
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `Decode(list,t)+idx` | count | half/BF16 ND | API/SIMT | SIMT/API | caller | 单元素，idx guard |
| scalar GM | element 0 | 2 B half 或 4 B float | S | ConvertToTensor | Process | executor | 每 block 一次 |
| thread-local | 当前 idx | xVal/sVal/result | FP32 | casts | SimtMaximum/降精度 store | thread | 下一 idx 覆盖 |

##### 分层内存布局

N/A：无 AIV UB Queue/TBuf 或 SIMT shared/dynamic UB；升精度值位于 thread-local 标量生命周期。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total = 0 B + executor 框架临时量
thread-local live values = xVal + sVal + result（固定仓不提供寄存器字节分配）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local FP32 scalars | casts/SimtMaximum | 当前 store | 下一 grid-stride idx | 单线程顺序 | 无跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | `static_cast<C>`，C=float | GM load/scalar arg | SimtMaximum | thread-local | half/BF16→FP32 |
| 比较 | `SimtMaximum(float,float)` | FP32 values | store | result | NaN/±0 显式路径 |
| 降精度/tail | `static_cast<T>` + `idx<count` | result | GM y | 单元素 | rounding mode 未在源码显式指定；无越界写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_maximum_scalar_tiling.cpp` | dtype key branch | schMode 1/3 |
| Kernel entry/router | `op_kernel/arch35/foreach_maximum_scalar.cpp` | entry `<1>/<3>` | T/S 模板映射 |
| Device body/helper | `op_kernel/arch35/foreach_maximum_scalar_simt.h` | `ComputeType<half/bfloat16_t>` / VF body | FP32 compute 和 thread tail |

### Route R-003：INT32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_maximum_scalar_v2.cpp` | dtype/scalar checks | x=INT32；scalar=INT32/INT64 且在 int32 范围 | int32 scalar Tensor |
| Tiling | `op_host/arch35/foreach_maximum_scalar_tiling.cpp` | dtype branch | DT_INT32 | schMode 2 |
| Router | `op_kernel/arch35/foreach_maximum_scalar.cpp` | entry `<2>` | compile-time branch | `Process<int32_t,int32_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_maximum_scalar<2>
  -> Process<int32_t,int32_t>
    -> VF_CALL<OpForeachMaximumScalarSimt<int32_t,int32_t>>(Dim3(1024), ...)
      -> SimtMaximum(int32_t,int32_t) -> GM store
```

##### 源码等价伪码

```text
for each Tensor t and each grid-stride idx<count:
  int32 xVal = xData[idx]
  int32 sVal = scalarVal
  yData[idx] = (xVal >= sVal) ? xVal : sVal
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `Decode(list,t)+idx` | count | int32/ND | API/SIMT | SIMT/API | caller | 单元素、idx guard |
| scalar GM | element 0 | 4 B | int32 | ConvertToTensor | Process | executor | 每 block 一次 |
| thread-local | current idx | xVal/sVal/result | int32 | load/compare | store | thread | 下一 idx 覆盖 |

##### 分层内存布局

N/A：无 AIV UB 与 SIMT shared/dynamic UB，只有 GM 和 thread-local 标量。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total = 0 B + executor 框架临时量
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local int32 scalars | load/compare | store | 下一 idx | 单线程顺序 | 唯一 idx 映射 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load/compare/store | 单线程程序顺序 | GM load | GM store | idx | int32 精确比较；`idx<count` tail |
| 线程间 | N/A | 唯一 grid-stride 映射 | N/A | y 元素 | 无 atomic/barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_maximum_scalar_tiling.cpp` | DT_INT32 branch | schMode 2 |
| Kernel entry/router | `op_kernel/arch35/foreach_maximum_scalar.cpp` | entry `<2>` | int32 模板 |
| Device body/helper | `op_kernel/arch35/foreach_maximum_scalar_simt.h` | `SimtMaximum(int32_t,int32_t)` | `a>=b?a:b` 与 grid tail |

## 算子特定优化

### Trick T-001：以全列表统一 grid-stride 避免每 Tensor 单独启动

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：单次 TensorList `tensorCount<=256`；Host 已按总元素数选择 blockDim；每 block 固定 1024 threads。
- Golden 来源：[`foreach_maximum_scalar_simt.h:109-120`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar_simt.h#L109-L120)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
    AscendC::Simt::GetThreadIdx());
uint64_t stride = static_cast<uint64_t>(
    AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    C xVal = static_cast<C>(xData[idx]);
    C sVal = static_cast<C>(scalarVal);
    C result = SimtMaximum(xVal, sVal);
    yData[idx] = static_cast<T>(result);
}
```

#### 机制与设计原因

一个 Kernel 启动覆盖整个 TensorList；每个 Tensor 复用同一全局线程网格，线程按 `blockNum*1024` 步长遍历。这样无需为不同 shape 的每个 Tensor 构造独立 launch，且 `idx<count` 自然处理尾部。scalar 在 `Process` 中每 block 从 GM 读取一次，再按值传入 VF body。

#### 迁移边界

- 适用：列表元素之间独立、每个 Tensor 都是连续一维可寻址元素域的 elementwise 算子。
- 前提：TensorList 指针表 ABI 一致；无跨元素依赖；Tensor 数可装入 POD；每个输出元素只有一个 global tid 写入。
- 失效：需要跨 Tensor 归约、广播形状变换或非连续 Device 索引时必须更换地址/同步策略；超过 256 Tensor 时 Host 直接失败。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FLOAT32 | R-002 FLOAT16/BFLOAT16 | R-003 INT32 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1/3 | schMode 2 | Host dtype branch |
| 模板与 dtype | T/S/C=float | T=half 或 BF16；C=float；S=half/float | T/S/C=int32 | 存储与计算精度 |
| 循环与分核 | 统一 grid-stride | 相同 | 相同 | 总元素/1024 选 blockDim |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread-local FP32；无 explicit UB | thread-local FP32；无 explicit UB | thread-local int32；无 explicit UB | SIMT 直接 GM 路径 |
| 内存总占用与复用 | operator workspace 0 | 同左 | 同左 | 无中间物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT GM load/store | SIMT load/cast/store | SIMT GM load/store | dtype 路径 |
| 同步、精度与 tail | 程序顺序；显式 float maximum | FP32 maximum 后降精度 | int32 compare | 均以 idx<count tail |
| fallback/失败边界 | 无 | 无 | 无 | 其它 dtype GRAPH_FAILED |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 公开支持，但不属于本卡 | V2 公开支持，但不属于本卡 | V2 公开支持 |
| Host/Launcher | arch22 公共 Tiling | arch22 公共 Tiling | arch35 自定义 Tiling、schMode 模板 key |
| Kernel/资源 | AIV Vector 双 Queue | 同 A2 | SIMT 1024 threads/block、workspace 0、无显式 UB buffer |

## 限制与失败边界

- 950 仅 V2 公开支持；旧 `aclnnForeachMaximumScalar` 产品表明确 950 不支持，不得把旧接口迁入本卡。
- API 源码对 `x/out` shape 做精确相等校验，严于公开文档的“out shape size 大于等于 x”描述。
- `tensorNum` 最大 256；`totalElements` 的 int64 累加、`needCoreNum` 向 int32 窄化未见溢出检查。
- Host 只检查 `ubSize!=0`，随后执行 `ubSize-128KiB`；固定仓未给 `ubSize>=128KiB` guard。body 没有显式使用该 local-memory 区域。
- FLOAT16/BFLOAT16 的 `static_cast<T>` 降精度 rounding mode 未在源码显式指定；固定仓不可见编译器/ISA 细节。
- executor 的 Contiguous/ConvertToTensor 临时 workspace、stream 内部事件和 cleanup 由框架管理，固定仓未展开；operator 自身 workspace 明确为 0。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_maximum_scalar/docs/aclnnForeachMaximumScalarV2.md` | V2 产品表/原型 | 950 支持、V2 语义与约束 |
| 入口实现 | `foreach/foreach_maximum_scalar/op_host/op_api/aclnn_foreach_maximum_scalar_v2.cpp` | `ExecForeachMaximumScalarV2GetWorkspaceSize` | API 校验、前处理、executor |
| 入口实现 | `foreach/foreach_maximum_scalar/op_host/op_api/foreach_maximum_scalar_v2.cpp` | `ForeachMaximumScalarV2` | AICore launcher 添加 |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar/op_host/foreach_maximum_scalar_def.cpp` | `ForeachMaximumScalar` OpDef | ascend950 注册、dtype/layout |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar/op_host/arch35/foreach_maximum_scalar_tiling.cpp` | `ForeachMaximumScalarTilingFunc` | 分核、POD、key、workspace/local memory |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar_tiling_data.h` | `ForeachMaximumScalarTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 候选 |
| Kernel entry/body | `foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar.cpp` | `foreach_maximum_scalar<schMode>` | key router |
| Kernel entry/body | `foreach/foreach_maximum_scalar/op_kernel/arch35/foreach_maximum_scalar_simt.h` | `Process` / VF body / `SimtMaximum` | SIMT 全实现 |
| 构建接线 | `foreach/foreach_maximum_scalar/op_kernel/CMakeLists.txt` | ascend950 source selection | arch35 kernel 活动接线 |
| 构建接线 | `foreach/foreach_maximum_scalar/op_host/config/ascend950/foreach_maximum_scalar_binary.json` | 4 dtype binary entries | ascend950 dtype/scalar Tensor 组合 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B POD、256 个元素计数 | 字段顺序、offset、位宽与 TensorList 指针表偏移协议 | `ForeachMaximumScalarTilingData` / `SimtGetTensorAddr` |
| 调用顺序 | scalar read→VF_CALL→每 Tensor grid-stride | 1024 threads、同一 blockNum/stride | `Process` / VF body |
| workspace/内存 | operator workspace 0、直接 GM 路径 | 不得假设存在 Queue/shared UB；localMemory 预留 128 KiB DCache | Tiling func / VF body |
| 同步 | 每元素单线程程序顺序 | 唯一 idx 映射；无须也没有 atomic/barrier | `OpForeachMaximumScalarSimt` |
| 精度与尾块 | half/BF16 升 FP32；浮点显式 NaN/±0；idx guard | ComputeType、SimtMaximum 顺序、降精度 cast | `ComputeType` / `SimtMaximum` |
| 失败边界 | tensorNum≤256、四 dtype key | 其它 dtype、零 core/UB、POD/workspace 获取失败必须终止 | `ForeachMaximumScalarTilingFunc` |