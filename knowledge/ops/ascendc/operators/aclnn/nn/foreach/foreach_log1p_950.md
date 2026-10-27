---
type: Operator
title: "ForeachLog1p 950 ACLNN 算子知识"
description: "说明 aclnnForeachLog1p 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/docs/aclnnForeachLog1p.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_host/foreach_log1p_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_host/arch35/foreach_log1p_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_kernel/arch35/foreach_log1p.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_host/config/ascend950/foreach_log1p_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:46:05Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachLog1p 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnForeachLog1pGetWorkspaceSize` / `aclnnForeachLog1p`，固定仓未公开版本后缀 |
| 计算语义 | 对列表中每个张量逐元素计算 `y_i[j] = ln(x_i[j] + 1)` |
| 输入输出 | 动态 TensorList；ND；FP16、FP32、BF16；对应输入输出 dtype 相同 |
| 执行类型 | 单 Kernel `SIMT`；每个 outer block 以 512 个 SIMT thread 执行 grid-stride loop |
| Kernel 路线 | R-001：FP16/FP32/BF16 统一 SIMT 模板，key 0/1/2 选择 `T` |
| 核心资源 | GM TensorList 指针表、Tiling GM、线程标量寄存器；无 AIV Queue/TBuf、无 shared/dynamic UB 数据区、无 workspace 数据 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档→`ACLNNTYPE aclnn` 生成边界→950 OpDef/config→arch35 Tiling/POD/template key→SIMT entry/body→ascend950 构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`。Host 将所有列表项按原顺序展平为长度 `totalElements` 的逻辑一维域；线程由累计 offset 找到所属 Tensor 和局部下标，然后执行 `static_cast<T>(log1pf(static_cast<float>(x_i[j])))`。因此三种 storage dtype 都以 FP32 `log1pf` 计算，FP16/BF16 再窄化到输出 dtype。源码不额外处理定义域或特殊值。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 0–8 维；ND；输入可非连续并由 `AutoContiguous` 处理 | FP16、FP32、BF16 | 所有 Tensor dtype 一致；支持零元素 Tensor；950 Tiling 显式限制列表项数≤128 |
| `out` | TensorList；ND | 与 `x` 相同 | 输出不支持非连续；对应输出元素容量不得小于输入；Kernel 只写输入元素范围 |
| `workspaceSize` / `executor` | Host 输出 | `uint64_t*` / executor | 第一段接口返回；950 Tiling 把算子节点 workspace 设为 0；非连续输入的框架 Contiguous workspace 公式固定仓不可见 |
| `workspace` / `stream` | Device地址/runtime stream | — | 第二段接口提交；Kernel 的 `Process` 参数含 workspace 但不读取 |

### 入口与版本

固定 Commit 的最新公开入口是无版本后缀 `aclnnForeachLog1p`。公开文档明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 将其映射到 `arch35`，binary config 覆盖三种 dtype，Kernel CMake 单独编译 `arch35/foreach_log1p.cpp`。生成后的 ACLNN inner bridge 不在仓内，其函数生成由 `ACLNNTYPE aclnn` 构建声明界定；算子拥有的后续链路已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段接口校验空指针、dtype、输入输出 dtype、shape、列表内 dtype 与维数；OpDef 对输入输出声明 ND、动态列表、`AutoContiguous`。
2. `ForeachLog1pTilingFunc` 读取输入动态实例数；`static_cast<int32_t>(tensorNum) > 128` 时失败。
3. 逐 Tensor 读取 storage shape size，累加 `totalElements` 并写 `cumulativeOffsets[i]`；首项描述符决定 key。固定 950 Tiling 没有重复执行公共 arch22 的输出 shape/dtype 校验代码，接口/OpDef/框架校验仍是可见上游边界。
4. 平台信息优先来自 compileInfo，其次 context platform；两者不可用时使用 `24 cores, 1 MiB UB` fallback。原始 UB 必须大于 128 KiB DCache 预留。

### 参数、Workspace 与生命周期

设平台 AIV 核数为 `C`，`N=totalElements`：`needCoreNum=clamp(ceil(N/1024),1,C)`，`perCoreElements=ceil(N/needCoreNum)`。Host 将 `needCoreNum` 同时写入 POD 并设为 `blockDim`；Device 不读取这两个分核字段，而是由 SIMT 的 `GetBlockNum()`/`GetBlockIdx()`使用实际 launch grid。Host 将可用 local memory 设为 `UB-128 KiB`，但当前 SIMT body 不声明或寻址任何 dynamic/shared UB。算子节点 workspace slice 为 0 B；`AutoContiguous` 可能在生成 executor 中增加框架 workspace。Tiling GM、x/y 指针表和输出 GM 必须保持到 stream 上 Kernel 完成。

### 启动图与失败传播

```text
aclnnForeachLog1pGetWorkspaceSize(x, out)
  -> ACLNN生成边界 / ForeachLog1p OpDef
  -> ForeachLog1pTilingFunc
     -> blockDim=clamp(ceil(totalElements/1024),1,AIV cores)
     -> key(dtype), POD, localMemory=UB-128KiB, workspace=0
  -> executor
aclnnForeachLog1p(..., stream)
  -> [同一 stream，单节点]
     foreach_log1p<schMode><<<blockDim>>>(xList,yList,null,tiling)
       -> VF_CALL<OpForeachLog1pSimtKernel<T>>(Dim3(512), ...)
```

这是算子拥有的单节点图，无跨 Kernel workspace 边；非连续输入可能经过生成 executor 的框架 `Contiguous` 节点，其实现未进入固定源码集合。Tiling 获取平台、实例信息、shape、dtype 或 local-memory 设置失败时返回 `GRAPH_FAILED` 并阻断 launch；Device 无 fallback/cleanup 节点。

## Route 与参数设计

### 分核与资源计算

```text
N = Σ shapeSize(x[i])
cumulativeOffsets[i] = Σ(k=0..i) shapeSize(x[k])
needCoreNum = max(min(ceil(N/1024), platformAivCoreNum), 1)
perCoreElements = ceil(N/needCoreNum)  # POD中保存但Device不读
threadsPerBlock = 512
globalThread = blockIdx * threadNum + threadIdx
gridStride = threadNum * blockNum
flatIdx = globalThread, globalThread+gridStride, ... < N
```

Tiling 的累计表最多 128 项。每个线程对当前 `flatIdx` 线性扫描 `cumulativeOffsets[0:tensorCount)` 找首个大于它的累计终点，再计算 `localIdx=flatIdx-prevCumSum`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 统一 SIMT | schMode 0/1/2，分别 `Process<half/float/bfloat16_t>` | Host `GetTilingKeyByDtype`: FP16→0，FP32→1，BF16→2；编译期 `if constexpr` 仅保留所选 T | unsupported dtype 正常由 OpDef拒绝；异常值在 Host default 到0 |

```text
dtype == FP16 -> schMode 0 -> Process<half>
dtype == FP32 -> schMode 1 -> Process<float>
dtype == BF16 -> schMode 2 -> Process<bfloat16_t>
otherwise -> Host helper default 0（正常公开入口不可达）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 统一 SIMT | 0/1/2 | FP16/FP32/BF16；TensorList≤128；950 | 单 SIMT Kernel，outer grid × 512 threads | `foreach_log1p<schMode>`→`Process<T>`→`OpForeachLog1pSimtKernel<T>`→`Log1pCompute<T>` | 每线程直接 GM load/store；仅标量寄存器；无 Queue/shared/workspace；线程写地址唯一 | grid-stride 的 `flatIdx<N` 是统一 tail；无 Device fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数 | `static_cast<int32_t>(tensorNum) <= 128` | POD 累计表固定128项；每线程线性查表 | R-001 | `GRAPH_FAILED`；极大 `uint64_t` 到 `int32_t` 窄化未额外防护 | `ForeachLog1pTilingFunc` |
| DCache 预留 | `ubSize > 128*1024` | Host 从 local memory 中扣除 DCache | R-001 | `GRAPH_FAILED` | `ForeachLog1pTilingFunc` |
| 每核最小工作量 | `ceil(N/1024)` 决定 blocks | 避免以超过约每block 1024元素所需的核数启动 | R-001 | 至少1 block，即使N=0 | `MIN_PER_CORE_ELEMENTS` |
| 线程数 | 编译期 `THREAD_NUM=512`，`LAUNCH_BOUND(512)`，`VF_CALL Dim3(512)` | thread body 与 launch bound 一致 | R-001 | 固定源码无其它线程数 Route | `OpForeachLog1pSimtKernel` / `Process` |
| 空域 | `N=0` 时 blockDim仍为1，所有线程初值≥0且 `flatIdx<N` 为假 | 避免 GM/TensorList 解引用 | R-001 | 正常无输出 | SIMT for-loop guard |
| dtype default | `GetTilingKeyByDtype` default 0 | Host helper没有失败返回 | R-001 | 正常 OpDef不可达；若绕过则按half解释 GM，属于错误边界 | `GetTilingKeyByDtype` |

### POD/ABI

Host 直接包含 Kernel 侧 `ForeachLog1pTilingData`，因此 producer/consumer 使用同一 C++ 定义。自然对齐下结构 size=1048 B、align=8；固定仓未打印 `sizeof/offsetof`，下列 offset 由字段顺序推得。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachLog1pTilingData` | `needCoreNum` | `int32_t`/32 | offset 0 | block | `static_cast<int32_t>(needCoreNum)` | 当前 Device body不读；相同值另用于 `SetBlockDim` | 1..平台AIV核数 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4 | Tensor项数 | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT 查表上界 | 0..128 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | 元素 | shape size累加 | SIMT grid-stride终点 | 累加无溢出检查 |
| 同上 | `perCoreElements` | `int64_t`/64 | offset 16 | 元素/block | `ceil(N/needCoreNum)` | 当前 Device body不读 | 仅诊断/保留字段 |
| 同上 | `cumulativeOffsets` | `int64_t[128]` | offset 24，1024 B；结构size1048 | 累计元素终点 | 每项累加后写 `[i]` | `flatIdx`→tensorId/localIdx 映射 | `[0,tensorCount)`有效；单调非降 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 统一 SIMT | schMode 0/1/2 | `op_host/arch35/foreach_log1p_tiling_arch35.cpp::ForeachLog1pTilingFunc` | `ForeachLog1pTilingData`、512 threads | `op_kernel/arch35/foreach_log1p.cpp::foreach_log1p<schMode>` | `foreach_log1p_simt.h::Process<T>` / `OpForeachLog1pSimtKernel<T>` / `Log1pCompute<T>` / `SimtGetTensorAddr<T>` | binary config三dtype；`op_kernel/CMakeLists.txt` ascend950 source |

## Kernel 实现

### 公共执行流程

编译期 `schMode` 选择 storage 类型。entry 解析 POD（局部 `tilingData` 变量未继续使用），转给 `Process<T>`；`Process` 从 Tiling GM 读取总元素数、列表项数和累计 offset 指针，然后以 `VF_CALL` 启动 512-thread SIMT body。SIMT body 将整个 TensorList 视为一条展平域，每线程以全 grid 步长循环，线性定位 Tensor，直接从输入 GM 取一个标量，以 `log1pf` 计算并写唯一输出地址。

### Route R-001：FP16/FP32/BF16 统一 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `foreach_log1p_tiling_arch35.cpp` | `GetTilingKeyByDtype` | switch FP16/FP32/BF16→0/1/2 | schMode |
| Tiling | 同上 | `ForeachLog1pTilingFunc` | Tensor≤128、UB>128KiB；blockDim/clamp公式 | 产生POD、localMemory、0 workspace |
| Router | `op_kernel/arch35/foreach_log1p.cpp` | `foreach_log1p<schMode>` | 三个独立 `if constexpr` | `Process<T>` |
| Body | `foreach_log1p_simt.h` | `Process<T>` | `VF_CALL(... Dim3(512), totalElements, tensorCount, cumOffsets,x,y)` | SIMT body |

#### Device 数据流

##### 函数级调用链

```text
foreach_log1p<schMode>(xList,yList,workspace,tiling)
  -> Process<T>(xList,yList,workspace,tiling)
     -> VF_CALL<OpForeachLog1pSimtKernel<T>>(Dim3(512), N, tensorCount, cumOffsets, xList, yList)
        -> grid-stride flatIdx loop
           -> linear cumulative-offset search
           -> SimtGetTensorAddr<T>(xList/yList,tensorId)
           -> Log1pCompute<T>::Compute
              -> static_cast<float>(val) -> log1pf -> static_cast<T>
           -> outputPtr[localIdx] = result
```

##### 源码等价伪码

```text
flatIdx = blockIdx * 512 + threadIdx
stride = 512 * blockNum
while flatIdx < totalElements:
    tensorId = tensorCount - 1
    prev = 0
    for t in [0, tensorCount):
        if flatIdx < cumulativeOffsets[t]:
            tensorId = t
            prev = (t > 0 ? cumulativeOffsets[t-1] : 0)
            break
    localIdx = flatIdx - prev
    inputPtr  = tensorAddress(xList, tensorId)
    outputPtr = tensorAddress(yList, tensorId)
    v = inputPtr[localIdx]
    outputPtr[localIdx] = cast<T>(log1pf(cast<float>(v)))
    flatIdx += stride
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| Tiling GM | struct base + offsets 8/4/24 | 1048 B POD | packed fields/自然对齐 | Host | `Process`/SIMT | framework，Kernel期间只读 | cumulative仅前tensorCount项有效 |
| TensorList指针表 | `base + (*base >> 3) + tensorId` | tensorCount个64-bit地址 | runtime list layout | ACLNN/framework | `SimtGetTensorAddr` | framework | `>>3` 把字节offset转uint64索引 |
| 输入 GM元素 | `inputPtr[flatIdx-prev]` | 1×sizeof(T)/迭代 | T/ND展平 | 调用者 | 当前thread | 调用者 | 标量直接读取，无显式合并DMA |
| 输出 GM元素 | `outputPtr[flatIdx-prev]` | 1×sizeof(T)/迭代 | T/ND展平 | 当前thread | 调用者 | 调用者 | 每个flatIdx唯一，无atomic |
| 线程寄存器 | `flatIdx,tensorId,prev,localIdx,val,ptrs` | 编译器决定；固定仓不可见字节数 | 标量 | 当前thread | 当前thread | 单次/循环迭代 | 无跨线程共享 |
| AIV UB/shared/dynamic UB | N/A | body分配0 B；Host仅设置可用local-memory上限 `U-128KiB` | — | — | — | — | 无寻址/复用 |
| workspace | N/A | 0 B | — | Host | 无 | 调用者 | `workspace` 参数未读 |

##### 分层内存布局

N/A：该 SIMT 节点不创建 AIV `TQue/TBuf`，也不声明 shared/dynamic UB 数据区；实际路径是每线程 GM 标量读取→寄存器 `log1pf`→GM 标量写回。

##### 字节账与复用顺序

```text
UB_total = 0 B（无 InitBuffer/TQue/TBuf/shared 对象）
dynamic_ub = 0 B 被 body 寻址；Host localMemorySize = platformUB - 128 KiB 仅作为 launch/local-memory 配置
operator_workspace = 0 B
executor_total_workspace = framework preprocessing workspace（生成公式固定仓不可见）
thread-local live state = 固定仓可证明为标量变量，寄存器具体字节数由编译器决定而不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程标量变量 | 当前loop迭代 | 同一thread的查表/计算/写回 | 下一flatIdx迭代 | 程序顺序 | 写回完成后才推进flatIdx；无跨线程别名 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 线程映射 | grid-stride + `flatIdx<N` | outer grid/512 threads | 当前thread | 输出元素唯一性 | 覆盖任意N；N=0不读GM |
| Tensor定位 | 线程内程序顺序 | cumOffsets线性查找 | localIdx/地址读取 | 当前flatIdx映射 | 零元素Tensor因累计值不增长而自然跳过 |
| 数值计算 | `log1pf(float(val))` | GM load | GM store | 当前标量 | FP16/BF16先升FP32再降T；FP32保持FP32 |
| 跨线程同步 | N/A | — | — | — | 无共享状态、无atomic/barrier需求 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_log1p/op_host/arch35/foreach_log1p_tiling_arch35.cpp` | `GetPlatformInfoFallback` / `ForeachLog1pTilingFunc` | 平台fallback、累计offset、blockDim、key、localMemory、0 workspace |
| Kernel entry/router | `foreach/foreach_log1p/op_kernel/arch35/foreach_log1p.cpp` | `foreach_log1p<schMode>` | 0/1/2到T的编译期映射 |
| Device body/helper | `foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_simt.h` | `Process<T>` | 512-thread VF_CALL参数 |
| Device body/helper | 同上 | `OpForeachLog1pSimtKernel<T>` | grid-stride、累计表查找、GM读写 |
| Device body/helper | 同上 | `Log1pCompute<T>` / `SimtGetTensorAddr<T>` | FP32 log1pf与TensorList地址解析 |

## 算子特定优化

### Trick T-001：把 TensorList 展平成一个全局 grid-stride 域

#### 触发与源码依据

- 适用 Route：R-001 的全部 dtype key
- 触发条件：Host 已产生单调 `cumulativeOffsets`，Device 以 `blockIdx*threadNum+threadIdx` 为起点、全 grid thread 数为步长。
- Golden 来源：[`foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_simt.h:64-85`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_simt.h#L64-L85)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int64_t flatIdx = static_cast<int64_t>(
         AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum()
         + AscendC::Simt::GetThreadIdx());
     flatIdx < totalElements;
     flatIdx += static_cast<int64_t>(
         AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum())) {
    int32_t tensorId = tensorCount - 1;
    int64_t prevCumSum = 0;
    for (int32_t t = 0; t < tensorCount; t++) {
        if (flatIdx < cumOffsets[t]) {
            tensorId = t;
            prevCumSum = (t > 0) ? cumOffsets[t - 1] : 0;
            break;
        }
    }
```

#### 机制与设计原因

Host 不为每个 Tensor 启动独立 Kernel，也不为不同长度 Tensor 生成 per-Tensor block 表；累计 offset 将整个列表变成连续的逻辑索引域。512-thread grid-stride loop 自动覆盖尾部和超大域，每个 `flatIdx` 只由一个 thread 处理，所以无需 atomic 或 barrier。代价是每个元素要对至多128个累计 offset 做线性查找；固定源码没有二分或按 Tensor 分组的另一 Route。

#### 迁移边界

- 适用：TensorList 各项相互独立、逐元素计算、不同 Tensor 可共享统一 dtype/公式的场景。
- 前提：累计 offset 单调且不溢出，列表项数≤128，输出对应项至少容纳输入元素，TensorList 指针表 ABI保持一致。
- 失效：跨 Tensor 依赖、需要 Tensor 级同步、不同项使用不同 dtype/公式，或列表过大使线性查找不可接受时，需要不同 Route/索引结构。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若省略 `flatIdx < totalElements` 尾部保护，最后一个不满 grid 的波次会越界；固定仓未提供其它可验证反例。

## 模板与实现对比

| 对比维度 | R-001 统一 SIMT | 设计选择依据 |
|---|---|---|
| 命中条件/优先级 | key 0/1/2，FP16/FP32/BF16 | Host dtype switch；无 shape 子Route |
| 模板与 dtype | `Process<T>`，统一 `Log1pCompute<T>` | storage T变化，计算均为FP32 log1pf |
| 循环与分核 | ceil(N/1024) outer blocks；512 threads；grid-stride | 覆盖展平列表与尾部 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅线程标量；无Queue/shared/workspace | 直接GM逐元素路径 |
| 内存总占用与复用 | body局部内存0 B；寄存器大小不可见 | 无中间物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT GM scalar load/store + `log1pf` | 逐元素公式 |
| 同步、精度与 tail | 无跨线程同步；FP32计算；`flatIdx<N` tail | 独立唯一输出地址 |
| fallback/失败边界 | Tensor>128、UB≤128KiB失败；unknown dtype helper default half | 单一路线，无Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 Foreach 区间POD | 同A2 | arch35累计offset POD、每约1024元素一个outer block |
| Kernel/资源 | AIV Vector双缓冲，BF16有FP32 scratch | 同A2 | SIMT 512 threads、直接GM、0 workspace |

## 限制与失败边界

- 950 Tiling 显式限制 TensorList≤128；每个元素线性扫描累计表，固定源码没有其它查找 Route。
- `totalElements` 与累计 offset 使用 `int64_t`，累加未做溢出检查；`tensorNum` 在上限检查前窄化成 `int32_t`，固定源码未覆盖超 `INT32_MAX` 的异常实例数。
- `GetTilingKeyByDtype` 对未知 dtype 默认 key 0。公开入口正常受 OpDef 限制，但绕过校验会使 Kernel 按 half 解释地址，不能作为 fallback 使用。
- Host 计算并序列化的 `needCoreNum`、`perCoreElements` 当前不被 Device body读取；实际并行度依赖 `SetBlockDim` 和 SIMT `GetBlockNum()`，移植时不能只复制 POD 字段而漏掉 launch 设置。
- 固定仓未保存自动生成 ACLNN bridge；executor/Contiguous 细节是框架边界。本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_log1p/docs/aclnnForeachLog1p.md` | `aclnnForeachLog1p*` | 公开接口、950支持、dtype/shape/错误语义 |
| 入口实现 | `foreach/foreach_log1p/op_host/CMakeLists.txt` | `add_modules_sources(... ACLNNTYPE aclnn ...)` | ACLNN生成边界、ascend950→arch35 |
| Host/OpDef | `foreach/foreach_log1p/op_host/foreach_log1p_def.cpp` | `ops::ForeachLog1p` | dtype、ND、AutoContiguous、ascend950配置 |
| Host/infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon` / `InferDataType4ForeachCommon` | 输出项数、shape与dtype传播 |
| Host/Tiling | `foreach/foreach_log1p/op_host/arch35/foreach_log1p_tiling_arch35.cpp` | `ForeachLog1pTilingFunc` | selector、POD、blockDim、localMemory、workspace |
| POD | `foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_tiling_data.h` | `ForeachLog1pTilingData` | ABI字段与128项累计表 |
| 模板选择 | `foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_tiling_key.h` | `ForeachLog1p` template args | schMode 0/1/2编译选择 |
| Kernel entry/body | `foreach/foreach_log1p/op_kernel/arch35/foreach_log1p.cpp` | `foreach_log1p<schMode>` | key→Process<T> |
| Kernel body | `foreach/foreach_log1p/op_kernel/arch35/foreach_log1p_simt.h` | `Process` / SIMT body/helper | 512 threads、grid-stride、地址、log1pf |
| 构建接线 | `foreach/foreach_log1p/op_host/config/ascend950/foreach_log1p_binary.json` | `op_list` | 950 三dtype binary |
| 构建接线 | `foreach/foreach_log1p/op_kernel/CMakeLists.txt` | `add_kernel_sources` | arch35 source仅为ascend950编译 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 累计offset表达TensorList展平域 | 五字段顺序、128长度、int64元素单位 | `ForeachLog1pTilingData` |
| 调用顺序 | Tiling→entry→Process→VF_CALL→thread body | schMode/T映射、512-thread launch | `foreach_log1p<schMode>` / `Process<T>` |
| workspace/内存 | 0 workspace、直接GM、无Queue/shared | Host localMemory扣128KiB但body不寻址；TensorList ABI `>>3` | `ForeachLog1pTilingFunc` / `SimtGetTensorAddr` |
| 同步 | 每flatIdx唯一thread，无barrier/atomic | grid-stride必须基于全blockNum，输出映射不得重叠 | `OpForeachLog1pSimtKernel` |
| 精度与尾块 | 三dtype统一FP32 log1pf，FP16/BF16降T；flatIdx guard | cast顺序与 `flatIdx<N` | `Log1pCompute<T>` |
| 失败边界 | Tensor≤128、UB>128KiB、OpDef dtype | unknown dtype default不可当fallback；累计和不溢出 | `ForeachLog1pTilingFunc` / `GetTilingKeyByDtype` |

- [Log1pf](../../../../apis/simt_api/math_functions/float_math_functions/log1pf.md)
