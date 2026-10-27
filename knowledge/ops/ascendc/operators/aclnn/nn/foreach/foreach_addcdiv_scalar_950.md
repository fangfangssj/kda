---
type: Operator
title: "ForeachAddcdivScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachAddcdivScalarV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/docs/aclnnForeachAddcdivScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_host/op_api/aclnn_foreach_addcdiv_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_host/op_api/foreach_addcdiv_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_host/config/ascend950/foreach_addcdiv_scalar_binary.json
    role: implementation
created_at: "2026-08-25T15:30:35Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAddcdivScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddcdivScalarV2GetWorkspaceSize` / `aclnnForeachAddcdivScalarV2`，公开 ACLNN V2 两段式接口。 |
| 计算语义 | `y_i[j] = x1_i[j] + scalar * (x2_i[j] / x3_i[j])`。 |
| 输入输出 | ND TensorList，FP16/FP32/BF16；同列表长度、同位置 shape、同 dtype。V2 scalar 为 `aclScalar`，内部转为单元素 Tensor。 |
| 执行类型 | 单 Kernel；外层 entry 标记 `KERNEL_TYPE_AIV_ONLY`，实际计算节点是每个 Tensor 片段一次 SIMT VF，无 Cube/AIC/MIX_CV。 |
| Kernel 路线 | R-001 FP16 SIMT；R-002 FP32 SIMT；R-003 BF16 SIMT。每条路线内部按片段元素数选择 32 位/64 位索引 VF。 |
| 核心资源 | GM TensorList、GM scalar、GM y；SIMT 标量寄存器；无 AIV Queue/TBuf、无 shared/dynamic UB、无 atomic/barrier；Host 预留 `UB-128 KiB` local-memory size，但 body 未显式消费；workspace 0。 |
| 来源闭包 | source_closed：公开 V2 → executor/launcher → arch35 Tiling/POD → dtype 模板 router → SIMT VF body → ascend950 构建接线已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

三个输入 TensorList 长度为 `n`，同位置 Tensor 的元素数为 `N_i`。对 `0<=i<n`、`0<=j<N_i`：

```text
y_i[j] = x1_i[j] + scalar * (x2_i[j] / x3_i[j])
```

SIMT body 的精确表达式为 `x1Val + scalarVal * (x2Val / x3Val)`。FP16/BF16 数据先逐元素转 FP32，完成除法、乘法、加法后转回低精度；FP32 直接计算。固定源码没有对零除数、NaN 或 Inf 添加特殊分支。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | 每项 0～8 维，公开格式 ND | FP16/FP32/BF16 | V2 文档不支持空 Tensor；列表内 dtype 一致。空 TensorList 由 API 成功短路。 |
| `x2` | 与 `x1` 同位置 shape | 同 `x1` | 列表长度、shape、dtype 匹配。 |
| `x3` | 与 `x1` 同位置 shape | 同 `x1` | 除数列表；约束同 x2。 |
| `scalar` | `aclScalar` | `x1=FP32/BF16` 时 FP32/DOUBLE；`x1=FP16` 时 FP16/DOUBLE | executor 对 BF16 数据显式生成 FP32 scalar Tensor，其余生成数据 dtype Tensor。 |
| `out` | 与 `x1` 同位置 shape；ND | 同 `x1` | 由 SIMT 线程直接写 GM。 |

### 入口与版本

公开版本族含 `aclnnForeachAddcdivScalar` 和更新的 `aclnnForeachAddcdivScalarV2`；V2 将 scalar 从 `aclTensor` 改为 `aclScalar`。V2 文档明确支持 Ascend 950PR/950DT，OpDef 注册 `ascend950`，Host CMake 将 `ascend950` 接到 `arch35`，Kernel CMake 将其接到 `arch35/foreach_addcdiv_scalar.cpp`。API 架构检查以 `AclnnUtil::IsRegbase(curArch)` 接纳 950。

## Host/Launcher 执行流程

### 参数校验与前处理

V2 API 与 A2/A3 共用：创建 executor；检查空指针、dtype 组合、四个列表的 dtype 一致性、三个输入列表长度与同位置 shape、输出 shape、最大 8 维及非私有格式。空 TensorList 设置 workspace 0 并不添加 Device 节点。非空时逐项调用框架 `Contiguous`，组装三份 executor TensorList；BF16 数据将 scalar 转成 FP32 Tensor，FP16/FP32 数据将其转成相同 dtype Tensor；最后只添加一个 `ForeachAddcdivScalar` AICore launcher 节点。任一步失败立即返回。

### 参数、Workspace 与生命周期

arch35 Tiling 从 compile info 或平台取得 AIV 核数与 UB 字节数；要求 `UB > 128 KiB`，设置 `localMemorySize=UB-128 KiB`，但 SIMT body 没有 shared/dynamic UB 声明或使用。节点 `workspace[0]=0`。ACLNN 第一段仍必须以 executor 的 `GetWorkspaceSize()` 为准，因为框架 Contiguous 可能引入自己的任务/空间；executor 拥有连续化结果、内部 scalar Tensor和 launch 描述，第二段在给定 stream 上运行。

### 启动图与失败传播

```text
V2 GetWorkspaceSize
  -> framework Contiguous(x1/x2/x3 items)
  -> ConvertToTensor(scalar)
  -> K0 foreach_addcdiv_scalar<schMode>
       outer blockDim = needCoreNum
       localMemorySize = platform UB - 128 KiB
       POD = ForeachAddcdivScalarTilingData
       workspace = 0
       per assigned tensor segment -> one SIMT VF_CALL
  -> y GM
```

K0 是唯一算子 Kernel。外层 AIV block 根据 POD 访问不重叠的 Tensor 片段；每片段的 VF 线程直接从 x1/x2/x3 GM 读取并写 y GM，因此无 Kernel 间 workspace 边。executor/stream 保证框架前处理完成后启动 K0；固定仓未展开框架内部事件。前处理或 launcher 失败都会阻断 K0。

## Route 与参数设计

### 分核与资源计算

设 `N=sum_i N_i`、`s=2/4` B、`E=32/s`、平台 AIV 核数 `C`：

```text
needCoreNum = (N == 0) ? 0 : clamp(ceil(N / 1024), 1, C)
blockCount = max(ceil(N / E), 1)
perCoreBlocks = floor(blockCount / needCoreNum)
remainder = blockCount % needCoreNum
```

`AssignDataToEachCore` 按 TensorList 顺序切连续区间，前 `remainder` 个核多一个 32 B block，并写入每核首尾 Tensor 索引及闭区间 offset。API 非空路线的 Tensor 数必须 `<=256`；每核数组固定 80 项，固定仓依赖 Ascend 950 的 AIV 核数不超过该数组容量，源码没有额外 `needCoreNum<=80` 检查。

每个分到的 Tensor 片段独立选择：

```text
dataCount <= INT32_MAX -> INDEX_T=uint32_t, threads=1024
dataCount >  INT32_MAX -> INDEX_T=int64_t,  threads=512
```

VF 内 `threadIdx` 从 0 开始，以 `threadCount` 为步长覆盖 `[0,dataCount)`；外层 `Process` 已分核，因此 VF 不再加入 `blockIdx/blockNum` 步长。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 SIMT | key/schMode 0；`Process<half>` | x1 dtype `DT_FLOAT16` | 下一 dtype；未知模板值不调用 body |
| 2 | R-002 FP32 SIMT | key/schMode 1；`Process<float>` | x1 dtype `DT_FLOAT` | 下一 dtype |
| 3 | R-003 BF16 SIMT | key/schMode 2；`Process<bfloat16_t>` | x1 dtype `DT_BF16` | 不支持 dtype 在 API 失败 |

```text
API arch/dtype/shape guards
  -> empty list: success, no K0
  -> dtype FP16: schMode 0 -> R-001
  -> dtype FP32: schMode 1 -> R-002
  -> dtype BF16: schMode 2 -> R-003
each R:
  per core, per assigned tensor segment:
    dataCount <= INT32_MAX ? VF<uint32_t,1024> : VF<int64_t,512>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 SIMT | 0 | 950、非空列表、FP16、检查通过 | 单 K0；每片段一 VF | `Process<half>`；`ForeachAddcdivScalarSimtKernel<half,INDEX_T>` | 直接 GM；1024/512 线程；无 UB/atomic/barrier | `i<dataCount` 线程循环自然保护尾部 |
| R-002 FP32 SIMT | 1 | 950、非空列表、FP32、检查通过 | 同上 | `Process<float>`；SIMT float VF | 同上 | 同上 |
| R-003 BF16 SIMT | 2 | 950、非空列表、BF16、检查通过 | 同上 | `Process<bfloat16_t>`；SIMT BF16 VF | 同上；固定源码的 scalar GM 类型读取与 Host producer 不一致 | 同上；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数 | `tensorNum <= 256` | POD 长度数组 256 项 | 全部 | Tiling 返回失败 | `ForeachAddcdivScalarTilingFunc` |
| UB/DCACHE | `ubSize > 128*1024` | Host 扣除 DCACHE 后设置 local memory | 全部 | Tiling 返回失败 | `ForeachAddcdivScalarTilingFunc` |
| 核数组 | `needCoreNum` 必须落入 80 项数组 | POD 首尾数组各 80 项 | 全部 | 源码无额外 guard；依赖平台 AIV 核数 | `AssignDataToEachCore` |
| 32 B 分核 | `E=32/dataTypeSize` | 让非最后分片按 DMA block 粒度切分 | 全部 | 最后余量仍以真实 Tensor 末 offset 收口 | `AssignDataToEachCore` |
| 索引宽度 | `dataCount <= INT32_MAX` | 32 位索引采用 1024 线程；64 位索引采用 512 线程以区分寄存器压力路径 | 全部 | 自动进入另一 VF 实例 | `Process`; `THREADS<INDEX_T>` |
| 空域 | API 空列表提前返回；Device 再检查 `totalDataCount<=0` | 防止读取无效 POD/TensorList | 全部 | Host 无 K0；Device 直接 return | API exec；`Process` |
| scalar ABI | BF16 数据时 Host producer 为 FP32 Tensor，但 VF 以 `T=bfloat16_t` 重解释 scalar | producer/consumer 类型必须一致才有定义明确的标量值 | R-003 | 固定源码无 fallback；属于已知闭包冲突 | API scalar conversion；SIMT VF `scalarPtr` |

### POD/ABI

Host `ForeachAddcdivScalarTilingDataHost` 与 Device `ForeachAddcdivScalarTilingData` 字段顺序、类型和数组长度逐项相同。按固定成员声明与普通 C++ 自然对齐，结构 alignof 为 8 B、sizeof 为 3664 B；offset 如下。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAddcdivScalarTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；struct align 8/size 3664 | 核 | `needCoreNum` cast 写入 | Device 不直接读取 | blockDim 使用同值；0 仅空域 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4 | Tensor 个数 | `tensorNum` cast 写入 | Device 不直接读取 | `0..256` |
| 同上 | `totalDataCount` | `int64_t`/64 | offset 8 | 元素 | 所有 Tensor numel 之和 | `Process` 空域检查 | 加法溢出未显式检查 |
| 同上 | `tensorDataCountList[256]` | `int64_t[256]` | offset 16，2048 B | 元素/每 Tensor | shape size | `Process` 非末片段终点 | 仅前 tensorCount 项有效 |
| 同上 | `tensorStartList[80]` | `uint16_t[80]` | offset 2064，160 B | Tensor 索引 | `AssignDataToEachCore` | `Process` | 仅已启动 core 项有效 |
| 同上 | `tensorEndList[80]` | `uint16_t[80]` | offset 2224，160 B | Tensor 索引，闭区间 | 同上 | `Process` | 同上 |
| 同上 | `tensorStartOffsetList[80]` | `int64_t[80]` | offset 2384，640 B | 元素 | 同上 | `Process` | 同上 |
| 同上 | `tensorEndOffsetList[80]` | `int64_t[80]` | offset 3024，640 B | 元素，闭区间 | 同上 | `Process` | 同上 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | schMode 0 | `arch35/...tiling_arch35.cpp::ForeachAddcdivScalarTilingFunc` | Host/Device POD；key 0；blockDim | `arch35/foreach_addcdiv_scalar.cpp::foreach_addcdiv_scalar<0>` | `Process<half>`；`ForeachAddcdivScalarSimtKernel<half,INDEX_T>` | `op_kernel/CMakeLists.txt` ascend950 arch35 |
| R-002 | schMode 1 | 同上 | key 1 | `foreach_addcdiv_scalar<1>` | `Process<float>`；float VF | 同上 |
| R-003 | schMode 2 | 同上 | key 2 | `foreach_addcdiv_scalar<2>` | `Process<bfloat16_t>`；BF16 VF | 同上 |

## Kernel 实现

### 公共执行流程

entry 用模板 `schMode` 编译期选择数据类型，并声明 `KERNEL_TYPE_AIV_ONLY`。`Process<T>` 读取 `blockIdx` 对应的 POD 区间，逐 Tensor 解析 `ListTensorDesc` 中的 GM 数据地址，加上 `cursorStart`，按片段长度选择 `uint32_t/1024 threads` 或 `int64_t/512 threads` VF。VF 的每个线程执行 `for(i=threadIdx;i<dataCount;i+=threadCount)`，直接加载三个 GM 元素和 scalar，再直接写对应 y 元素。各外层 block 的区间互斥，各线程的 i 互斥，因此无需 atomic；无共享数据，因此无 cooperative group/barrier。

### Route R-001：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_addcdiv_scalar_v2.cpp` | dtype check/conversion | self FP16；scalar FP16/DOUBLE | 内部 scalar FP16 |
| Tiling | `op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | key 0 |
| Router | `op_kernel/arch35/foreach_addcdiv_scalar.cpp` | `foreach_addcdiv_scalar<schMode>` | `schMode==0` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
launcher -> foreach_addcdiv_scalar<0>
  -> Process<half>
    -> GetTensorAddrFromList<half>(x1/x2/x3/y)
    -> dataCount<=INT32_MAX
         ? VF_CALL<ForeachAddcdivScalarSimtKernel<half,uint32_t>>(1024 threads)
         : VF_CALL<ForeachAddcdivScalarSimtKernel<half,int64_t>>(512 threads)
```

##### 源码等价伪码

```text
if totalDataCount<=0: return
core = blockIdx
for tensor in POD[core].start..end:
  begin/end = POD boundary offsets; count=end-begin+1
  if count<=0: continue
  bind four GM pointers at begin
  choose INDEX_T and threads from count
  parallel for i=threadIdx; i<count; i+=threadCount:
    a=float(x1[i]); b=float(x2[i]); c=float(x3[i]); s=float(scalar_fp16[0])
    y[i]=half(a + s*(b/c))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/x3 GM | TensorList data pointer + `cursorStart+i` | core/Tensor 片段 | FP16/ND | caller/Contiguous | SIMT thread | caller/executor | 每线程标量 load |
| scalar GM | element 0 | 1×FP16 | FP16 | ConvertToTensor | 全部 VF 线程 | executor | 线程各自加载/转 FP32 |
| y GM | TensorList data pointer + `cursorStart+i` | 同输入片段 | FP16/ND | SIMT thread | caller | caller | 每线程唯一地址 |
| 寄存器 | thread-local `x1Val/x2Val/x3Val/scalarVal/i/step` | 固定仓未给寄存器字节数 | FP32/INDEX_T | SIMT thread | 同线程 | VF 调用 | 线程私有 |

##### 分层内存布局

N/A：该 SIMT body 不创建 AIV Queue/TBuf、shared/dynamic UB 或通信 window；数据经线程标量 GM load/store 与线程私有寄存器流动。Host 虽设置 `localMemorySize=UB-128 KiB`，固定 body 没有对应显式分配。

##### 字节账与复用顺序

```text
UB_total = 0 B（无显式 AIV Queue/TBuf/shared UB）
dynamic_ub = N/A（VF_CALL 未传 dynUBufSize）
local_memory_reservation = platform_UB - 128 KiB（Host 设置，body 未显式消费）
workspace_total = 0 B（算子节点）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程寄存器 | 当前循环 GM loads | 当前表达式/store | 同线程下一 i | 顺序语义 | i 写地址在线程间不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 线程覆盖 | grid-stride loop | threadIdx/threadCount | 无等待 | `[0,dataCount)` | `i<dataCount` 保护任意尾部 |
| 数值计算 | 线程顺序表达式 | 三个 FP16 load | y store | 线程私有值 | 全部转 FP32计算，再 `__float2half` |
| 跨线程 | N/A | N/A | N/A | 每线程唯一 y[i] | 无 atomic/barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_addcdiv_scalar/op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp` | TilingFunc / AssignDataToEachCore | blockDim、POD、local memory、key |
| Kernel entry/router | `foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar.cpp` | `schMode==0` | `Process<half>` |
| Device body/helper | `foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_simt.h` | `Process<half>` / SIMT VF | 线程数、索引、GM 地址与 FP32 计算 |

### Route R-002：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_addcdiv_scalar_v2.cpp` | dtype check/conversion | self FP32；scalar FP32/DOUBLE | 内部 scalar FP32 |
| Tiling | `op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT` | key 1 |
| Router | `op_kernel/arch35/foreach_addcdiv_scalar.cpp` | `schMode==1` | true | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
launcher -> foreach_addcdiv_scalar<1> -> Process<float>
  -> per Tensor segment VF_CALL<ForeachAddcdivScalarSimtKernel<float,uint32_t|int64_t>>
```

##### 源码等价伪码

```text
for each assigned segment and each thread-stride index i:
  y[i] = x1[i] + scalar_fp32[0] * (x2[i] / x3[i])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/x3/y GM | TensorList pointer + core segment + i | `dataCount` FP32 | FP32/ND | caller/thread | thread/caller | caller | 直接标量 load/store |
| scalar GM | 0 | 1×FP32 | FP32 | executor | VF threads | executor | 无类型转换 |
| 寄存器 | thread-local values/index | 固定仓不可见 | FP32/INDEX_T | thread | thread | VF | 下一循环覆盖 |

##### 分层内存布局

N/A：与 R-001 相同，实际路径是 GM↔线程寄存器，无 AIV UB/shared/dynamic UB。

##### 字节账与复用顺序

```text
UB_total = 0 B
dynamic_ub = N/A
local_memory_reservation = platform_UB - 128 KiB（未显式消费）
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程寄存器 | GM load | 当前表达式 | 下一 i | 线程顺序 | 唯一 y 索引 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 索引循环 | threadIdx/threadNum | 线程调度 | 无等待 | 输出索引 | `i<dataCount` |
| 计算/写回 | 顺序标量算术 | GM load | GM store | 线程私有 | 全程 FP32；无显式 round |
| 同步 | N/A | N/A | N/A | 互斥地址 | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp` | `GetTilingKeyByDtype` | FP32→1 |
| Kernel entry/router | `op_kernel/arch35/foreach_addcdiv_scalar.cpp` | `schMode==1` | `Process<float>` |
| Device body/helper | `op_kernel/arch35/foreach_addcdiv_scalar_simt.h` | float VF | 直接 FP32 算术与写回 |

### Route R-003：BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_addcdiv_scalar_v2.cpp` | dtype check/conversion | self BF16；scalar FP32/DOUBLE；转换目标 FP32 | 内部 scalar Tensor 为 FP32 |
| Tiling | `op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_BF16` | key 2 |
| Router | `op_kernel/arch35/foreach_addcdiv_scalar.cpp` | `schMode==2` | true | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
launcher -> foreach_addcdiv_scalar<2> -> Process<bfloat16_t>
  -> VF_CALL<ForeachAddcdivScalarSimtKernel<bfloat16_t,uint32_t|int64_t>>
     -> scalar is reinterpreted as __gm__ bfloat16_t* in fixed body
```

##### 源码等价伪码

```text
for each assigned i:
  scalarVal = float(reinterpret_cast<bf16*>(scalarGM)[0])
  a,b,c = bf16_to_float(x1[i],x2[i],x3[i])
  y[i] = float_to_bf16(a + scalarVal*(b/c))
```

注意：Host/API 和 binary config 为 BF16 数据路线生产 FP32 scalar Tensor，而上述 Device body 以 BF16 指针读其首 16 bit。该 producer/consumer 类型冲突是固定源码事实，不能把预期 FP32 scalar 值当成当前 body 已闭合实现。

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/x3/y GM | TensorList pointer + segment + i | `dataCount` BF16 | BF16/ND | caller/thread | thread/caller | caller | 直接 GM |
| scalar GM | byte offset 0 | Host: 1×FP32；Device 实际读 1×BF16 | ABI 冲突 | executor FP32 Tensor | VF `reinterpret_cast<__gm__ T*>` | executor | 无转换桥接 |
| 寄存器 | a/b/c/scalar/result/index | 固定仓不可见 | FP32/INDEX_T | thread | thread | VF | 下一 i 覆盖 |

##### 分层内存布局

N/A：无 AIV Queue/TBuf/shared/dynamic UB；GM scalar 存在上表所述类型冲突。

##### 字节账与复用顺序

```text
UB_total = 0 B
dynamic_ub = N/A
local_memory_reservation = platform_UB - 128 KiB（未显式消费）
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程寄存器 | 当前元素 GM load | 当前计算/store | 下一 i | 线程顺序 | 输出地址唯一；scalar ABI 冲突另见限制 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 元素覆盖 | thread-stride loop | threadIdx | 无等待 | `[0,dataCount)` | `i<dataCount` |
| 数据计算 | BF16→FP32→BF16 helper | GM load | y store | 线程私有 | `__float2bfloat16`；round mode 未在固定源码显式命名 |
| scalar 读取 | 指针重解释 | FP32 Host producer | BF16 Device load | scalar 首 2 B | 类型不一致，无 fallback |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/op_api/aclnn_foreach_addcdiv_scalar_v2.cpp` | `ConvertToTensor(...DT_FLOAT)` | BF16 路线 scalar 为 FP32 |
| Kernel entry/router | `op_kernel/arch35/foreach_addcdiv_scalar.cpp` | `schMode==2` | `Process<bfloat16_t>` |
| Device body/helper | `op_kernel/arch35/foreach_addcdiv_scalar_simt.h` | BF16 VF | scalar 按 T 重解释；数据升 FP32计算 |

## 算子特定优化

### Trick T-001：分核后 VF 只保留线程级步长，并按索引宽度选择线程数

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：每个外层 core 的每个非空 Tensor 片段；片段 `<=INT32_MAX` 走 32 位索引，否则走 64 位索引。
- Golden 来源：[`foreach_addcdiv_scalar_simt.h:137-145`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_simt.h#L137-L145)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (dataCount <= static_cast<int64_t>(INT32_MAX)) {
    Simt::VF_CALL<ForeachAddcdivScalarSimtKernel<T, uint32_t>>(
        Simt::Dim3(THREADS<uint32_t>), static_cast<uint32_t>(dataCount), scalar,
        x1_gm, x2_gm, x3_gm, y_gm);
} else {
    Simt::VF_CALL<ForeachAddcdivScalarSimtKernel<T, int64_t>>(
        Simt::Dim3(THREADS<int64_t>), dataCount, scalar,
        x1_gm, x2_gm, x3_gm, y_gm);
}
```

#### 机制与设计原因

外层 Tiling 已把全局域分成互斥 core 区间，`Process` 又把当前 core 的区间拆到各 Tensor 片段。因此 VF 内只用 `threadIdx/threadNum`，不重复乘入 block 维；这避免非零 core 对已经偏移的局部片段再次跳过数据。32 位索引实例使用 1024 线程，64 位索引实例使用 512 线程；源码注释把差异归因于索引宽度的寄存器压力，并通过编译期 `THREADS<INDEX_T>` 固化。

#### 迁移边界

- 适用：Host/外层 Kernel 已完成不重叠分块、VF 参数已偏移到局部片段起点的纯逐元素计算。
- 前提：`dataCount` 分支与 INDEX_T 的合法范围一致；每线程输出地址互斥。
- 失效：若 VF 参数仍指向全局起点或多个 block 共同处理同一域，必须重新引入 block 维地址映射或同步，而不能照搬线程级步长。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码注释明确记录：在已分核的 `Process` 之后再次在 VF 使用 `GetBlockIdx/GetBlockNum` 会令非零 core 跳过全部数据；不得恢复该二次 block 步长。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 2 | dtype 唯一映射 |
| 模板与 dtype | `Process<half>` | `Process<float>` | `Process<bfloat16_t>` | 编译期消除 dtype 分支 |
| 循环与分核 | `ceil(N/1024)` cores；线程 stride | 同左 | 同左 | 片段按 32 B block 分配 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅线程寄存器；dyn UB N/A | 同左 | 同左 | 直接 GM SIMT |
| 内存总占用与复用 | 显式 UB 0；寄存器循环复用 | 同左 | 同左 | body 无局部数组/Queue |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM load/store；SIMT VF | 同左 | 同左 | 无 Vector/NDDMA/通信 |
| 同步、精度与 tail | FP32 中间→FP16；i<count | FP32；i<count | FP32 中间→BF16；scalar ABI 冲突 | 地址互斥故无 atomic/barrier |
| fallback/失败边界 | 32/64 位索引内部切换 | 同左 | 同左，另有 scalar 冲突 | 未知 schMode 无 body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 公开支持，见无后缀卡 | V2 公开支持，见无后缀卡 | V2 公开支持 |
| Host/Launcher | arch22 公共 foreach Tiling | 同 A2 | 独立 arch35 Tiling/POD；workspace 0 |
| Kernel/资源 | AIV Vector、UB Queue | 同 A2 | SIMT VF、直接 GM、无显式 UB |

## 限制与失败边界

- R-003 存在固定源码 producer/consumer 冲突：API 与 binary config 为 BF16 数据路线提供 FP32 scalar Tensor，SIMT VF 却以 `bfloat16_t*` 读取 scalar。复用实现时必须先解决 ABI 一致性；本卡不推测正确值或替代代码。
- Host POD 每核数组仅 80 项，Tiling 没有独立检查 `needCoreNum<=80`，依赖固定平台 AIV 核数约束。
- `totalDataCount` 使用 int64 累加且无溢出检查；极端总元素数不可外推。
- Host 要求 UB 大于 128 KiB 并设置剩余 local memory，但 body 没有显式消费；不能把该保留空间移作他用而不重审 launch ABI。
- V2 文档不支持空 Tensor；空 TensorList 仅是 API 的无 Kernel 成功短路。公开格式按 ND，而非 API 检查实际接受的所有非私有格式。
- 未知 `schMode` 没有错误/fallback body；API/Tiling 应阻止该状态。
- DOUBLE scalar 的转换/舍入和 executor 内 stream 事件属于固定仓不可见框架边界。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_addcdiv_scalar/docs/aclnnForeachAddcdivScalarV2.md` | 产品表/V2 原型 | 公开性、版本、950 支持与约束 |
| 入口实现 | `foreach/foreach_addcdiv_scalar/op_host/op_api/aclnn_foreach_addcdiv_scalar_v2.cpp` | `ExecForeachAddcdivScalarV2GetWorkspaceSize` | 校验、前处理、scalar producer |
| 入口实现 | `foreach/foreach_addcdiv_scalar/op_host/op_api/foreach_addcdiv_scalar_v2.cpp` | `l0op::ForeachAddcdivScalarV2` | 单节点 launcher |
| Host/Tiling/Launcher | `foreach/foreach_addcdiv_scalar/op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp` | TilingFunc/AssignDataToEachCore | 分核、key、local memory、workspace |
| Host/Tiling/Launcher | `foreach/foreach_addcdiv_scalar/op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.h` | Host POD | Host ABI |
| Kernel entry/body | `foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar.cpp` | templated entry | schMode→T |
| Kernel entry/body | `foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_simt.h` | `Process` / SIMT VF | 线程、地址、精度、tail |
| Kernel entry/body | `foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_tiling_data.h` | Device POD | Device ABI |
| Kernel entry/body | `foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_tiling_key.h` | template args | 三个 schMode 实例 |
| 构建接线 | `foreach/foreach_addcdiv_scalar/op_kernel/CMakeLists.txt` | ascend950 `add_kernel_sources` | arch35 活动 Kernel |
| 构建接线 | `foreach/foreach_addcdiv_scalar/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` / `SUPPORT_TILING_DIR` | ascend950→arch35 |
| 构建接线 | `foreach/foreach_addcdiv_scalar/op_host/config/ascend950/foreach_addcdiv_scalar_binary.json` | `op_list` | 950 三组 dtype ABI，BF16 数据的 scalar 为 FP32 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 3664 B、8 B 对齐的分核 POD | 字段顺序、256/80 数组长度、闭区间 offset | Host/Device tiling data |
| 调用顺序 | API 前处理→单 K0→每片段 VF | 不在 VF 再乘 block stride | API exec；`Process` |
| workspace/内存 | 节点 workspace 0；直接 GM；Host localMemorySize=UB-128KiB | executor 总 workspace仍由第一段返回；local memory 设置不可静默删除 | TilingFunc |
| 同步 | 无 atomic/barrier；靠互斥分核/线程索引 | 复用前必须保持输出地址互斥 | AssignDataToEachCore；VF loop |
| 精度与尾块 | 低精度升 FP32；thread loop 自然 tail | FP16/BF16 cast 顺序；32/64 位索引阈值 | SIMT VF；`Process` |
| 失败边界 | tensorNum 256、UB>128KiB、已知 BF16 scalar ABI 冲突 | 不把冲突路线当作已验证正确 | TilingFunc；API conversion；VF scalarPtr |