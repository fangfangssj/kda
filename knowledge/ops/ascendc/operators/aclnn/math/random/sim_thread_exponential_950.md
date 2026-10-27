---
type: Operator
title: "SimThreadExponential 950 ACLNN 算子知识"
description: "说明 aclnnSimThreadExponential 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 SIMT 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/docs/aclnnSimThreadExponential.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_host/sim_thread_exponential_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_host/arch35/sim_thread_exponential_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/random_common/op_host/arch35/random_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_kernel/sim_thread_exponential.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/op_kernel/arch35/sim_thread_exponential_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/random_common/op_kernel/arch35/random_kernel_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/sim_thread_exponential/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:23:17Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# SimThreadExponential 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnSimThreadExponentialGetWorkspaceSize` + `aclnnSimThreadExponential`，公开 ACLNN 两段式接口 |
| 计算语义 | Philox4x32-10 生成 `u = r*2^-32 + 2^-33`，写 `x=-log(u)/lambda`；近 1 端点以 `-epsilon/2` 保护 |
| 输入输出 | `selfRef`：ND、最多 8 维、FP16/BF16/FP32；950 要求 count 等于元素数、offset 是 4 的倍数、lambda>0 |
| 执行类型 | 单外层 Kernel，实际计算节点为 SIMT VF；entry 标记 `KERNEL_TYPE_MIX_AIV_1_0`，不是 Cube/AIC 或 MIX_CV |
| Kernel 路线 | R-001 FP32 SIMT；R-002 FP16/BF16 SIMT |
| 核心资源 | GM 直接写、每线程 key/counter/results 寄存器数组、128 KiB Dcache 预留；无 AIV TQue/TBuf、无 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN/生成桥边界 → arch35 私有配置 → `RandomTilingArch35` → unified SIMT POD → entry/router → split-block launcher → Philox SIMT body → CMake |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

每个输出消耗 Philox4x32-10 的一个 uint32 lane。`ExponentialTransform<T>` 将其变换为严格大于 0 且小于 1 的 FP32 `u`，当 `u >= 1 - epsilon/2` 时直接令 `logVal=-epsilon/2`，否则调用 `AscendC::Simt::Log(u)`；随后以 FP32 计算 `-logVal/lambda`，最后 `static_cast<T>` 写 GM。FP16/BF16 的最终转换细节由 SIMT 编译器实现，固定源码没有显式 round mode。

公开公式写 `u∈(0,1]`，固定 950 body 的半格偏移实际避免 0，并用近 1 分支避免端点对数行为。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `selfRef` | ND，文档最多 8 维 | FP16/BF16/FP32 | 输入/输出；OpDef `AutoContiguous()`，950 entry 实际写第一个 `self` GM 参数而忽略第二个 `self_ref`，依赖原地 alias 契约 |
| `count` | 标量 | INT64 | 必须非负且等于输出 storage shape 元素数；作为 `outputSize` |
| `lambda` / `lambd` | 标量 | ACLNN DOUBLE，Op 属性/Tiling FLOAT | `attrCheckRules` 明确要求 `>0`；写入 POD `prob` |
| `seed` | 标量 | INT64 | POD 原样保存；SIMT 初始化 Philox key 的低/高 32 位 |
| `offset` | 标量 | INT64 | 必须是 4 的倍数；作为 Philox counter offset，分片时进一步生成 `kernelOffset` |

### 入口与版本

固定仓公开的最新入口是无版本后缀的 ACLNN 两段式接口。自动生成 ACLNN inner bridge、executor 和连续化实现未落在算子目录，作为框架边界；公开文档、OpDef 与全部算子拥有的 Host/POD/Kernel 链路均可定位。

README/API 明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，CMake 将 `ascend950` 映射到 `arch35`。固定仓没有更高版本入口或文档不支持/实现支持的平台注册冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段校验空指针、dtype、维数与 `lambda>0`；OpDef 对 `self` 执行自动连续化。
2. `RandomTilingArch35::CheckInputsOutputsAndAttrs` 依据私有 `BuildOpConfig` 校验输出 dtype 与 `lambd>0`。
3. `getOutputSize` 读取 `count` 并要求其等于输出 storage shape 元素数。
4. `getSeedAndOffset` 读取 seed/offset 并拒绝 `offset % 4 != 0`。
5. 公共基类取得 AIV core/UB，扣除 128 KiB Dcache 后计算 local-memory 配置。
6. `DoSimtBlockTiling` 计算外层 `blockDim=usedCoreNum`；若输出不能用 32 位索引，最多分成 8 个 contiguous split block，再为每个 block 计算 logical grid、totalThreads 与独立 counter offset。
7. `UniqueProcess` 写 `prob=lambda` 并按输出 dtype 选择 key 1/2/3；`WriteBackToContext` 写回零 workspace、blockDim、key、local memory 与 POD。

### 参数、Workspace 与生命周期

`workspaceSize_=0`，所以本算子没有用户 workspace slice 或 Kernel 间 GM 交接。第一段仍遵循 ACLNN 通用 executor 生命周期；第二段在指定 stream 启动单个 entry。

`config.DcacheSize=128*1024`；公共 Host 将平台 UB 减去该值，并通过 `SetLocalMemorySize(ubSize_)` 设置剩余 local memory。SIMT body 不声明 shared/dynamic UB 数组，也不读取 POD 的 `ubSize`（该字段保持值初始化的 0）；实际活跃数据是每线程局部 `key[2]`、`counter[4]`、`counterTmp[4]`、`results[4]` 和标量。

### 启动图与失败传播

```text
ACLNN GetWorkspaceSize
  -> generated executor / AutoContiguous boundary
  -> Tiling4SimThreadExponentialArch35
     -> RandomTilingArch35::DoTiling
        -> validate -> platform -> FillUnifiedSimtTilingData
        -> CalcSplitBlocks -> CalcExecutionPoliciesForBlocks
        -> UniqueProcess -> WriteBackToContext
  -> sim_thread_exponential(self, self_ref, workspace, tiling)
     [outer blockDim=usedCoreNum, key 1/2/3]
     -> SimThreadExponential::Process<T>(self, POD)
        -> for each split block: VF_CALL<PhiloxSimtKernelDiscontinuous>(512 threads)
           -> ExponentialTransform<T> -> self GM
```

这是一个外层 Device entry，不是 Host 多 Kernel graph。每个 split block 是同一 entry 内顺序发出的 SIMT VF 调用；它们写互不重叠的 `self + gmOffset` 区间，counter 使用各自 `kernelOffset`。没有 workspace 边、event、atomic 或跨 entry 生命周期。任一 Host 校验、平台、分片或写回错误返回 `GRAPH_FAILED` 并阻断 launch；未知 key 在 Device router 静默无输出。

## Route 与参数设计

### 分核与资源计算

令 `N=count`、平台 AIV 数 `A`：

```text
avgPerCore = ceil(N/A)
numOfPerCore = ceil_align(avgPerCore, 4)
usedCoreNum = max(1, min(A, ceil(N/numOfPerCore)))
```

对每个 32 位可索引 split block，令元素数 `n`：

```text
grid = min(max(ceil(n/256), 1), 78*(2048/256)) = min(max(ceil(n/256),1),624)
totalThreads = grid*256                         # logical thread domain
threadsPerRound = totalThreads*4               # Philox UNROLL=4
roundsNeeded = ceil(n/threadsPerRound)
counterOffset = roundsNeeded*4
```

单分片的 `kernelOffset=offset`；多分片先额外跳过一次按整个 `outputSize` 计算的 `counterOffset`，然后逐片累加各自 counter extent，避免分片 PRNG counter 域重叠。split 判断同时要求 `numel<=INT32_MAX` 且最大字节 offset `<=INT32_MAX`；否则按最大 extent 维度二分，最多 8 片。

SIMT VF 编译期 `blockDim.x=512`。每个物理线程从 `idx=blockIdx.x*blockDim.x+threadIdx.x` 出发，并以实际 `blockDim.x*gridDim.x` 跨步遍历逻辑 `linearIndex<totalThreads`，所以 Host 的 logical grid 不要求等于外层 AIV `usedCoreNum`。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | key 3，`Process<float>` | output dtype `DT_FLOAT`（UniqueProcess 的 final else，已由 dtype rule 限定） | router 再检查 key 1 |
| 2 | R-002 低精度 SIMT | key 1/2，`Process<half/bfloat16_t>` | FP16→1，BF16→2 | 非法 dtype 在 Host 失败；未知 key 无输出 |

```text
validate dtype in {FP16,BF16,FP32}
if FP16: key=1 -> Process<half>
else if BF16: key=2 -> Process<bfloat16_t>
else: key=3 -> Process<float>
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 SIMT | 3 | FP32/ND/950；count=elements；lambda>0；offset%4=0 | 单 entry→每 split block 一个顺序 VF call | `Process<float>`→`PhiloxSimtKernelDiscontinuous<float,ExponentialTransform<float>>` | GM 直写；线程私有寄存器；无 shared UB/atomic/barrier | `li<numel` 逐元素 guard；未知 key 无输出 |
| R-002 低精度 SIMT | 1/2 | FP16/BF16，其余同 R-001 | 同上 | half/BF16 模板 | 同上；FP32 transform 后 static_cast | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Dcache 预留 | `platformUB - 128KiB > 0` | SIMT VF 配置要求 Dcache，剩余作为 local memory | 全部 | Host `GRAPH_FAILED` | `GetPlatformInfo`, `BuildOpConfig` |
| 32 位索引片 | `numel<=2^31-1` 且 max byte offset<=2^31-1 | SIMT 当前 slice 的索引/地址域 | 全部 | 最大 extent 维二分，最多 8 片 | `Is32bitIndexable/CalcSplitBlocks` |
| offset 对齐 | `offset%4==0` | 每 Philox counter 产生 4 个结果 | 全部 | Host `GRAPH_FAILED` | `getSeedAndOffset` |
| logical grid cap | `grid<=624` | 78 cluster×每 AIC 8 个 256-thread group 的策略上界 | 全部 | 每物理线程循环覆盖更多 logical index | `CalcExecutionPoliciesForBlocks` |
| 线程展开 | `UNROLL=4`, VF threads=512 | 一轮 Philox 的 4 lanes 对应四个跨 `totalThreads` 输出 | 全部 | 编译期固定，无 fallback | `PhiloxSimtKernelDiscontinuous` |
| 空域 | `N>0` 才能安全形成 `numOfPerCore` 分母 | `DoSimtBlockTiling` 对空域可能计算 `ceil(0/0)` | 全部 | 固定源码未提供可靠 fallback | `DoSimtBlockTiling` |

### POD/ABI

该结构由普通 C++ 成员构成；按源码声明与自然对齐，整体为 8 字节对齐、408 B，split 数组从 offset 88 开始，每项 40 B。固定源码没有显式 `static_assert`，移植时应以同一 ABI 编译器复核。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `RandomUnifiedSimtTilingDataStruct` | `usedCoreNum` | int64/64 | off 0 | AIV core | `DoSimtBlockTiling` | `Process` early return | 至少1（空域公式例外见限制） |
| 同上 | `outputSize` | int64/64 | off 8 | element | `getOutputSize=count` | 间接用于 split policies | 非负 |
| 同上 | `seed` | int64/64 | off 16 | seed bits | `getSeedAndOffset` | `ExponentialLauncher`→Philox | 全 Route active |
| 同上 | `offset` | int64/64 | off 24 | random result | `getSeedAndOffset` | 仅 Host 派生 `kernelOffset`；body 不直接读 | 4 对齐 |
| 同上 | `ubSize` | int64/64 | off 32 | byte | 值初始化 0，无专属赋值 | 未读取 | inactive |
| 同上 | `extraInt64Param1` | int64/64 | off 40 | N/A | 值初始化 0 | 未读取 | inactive |
| 同上 | `prob` | float/32 | off 48 | rate | `UniqueProcess: prob=*lambdAttr` | `ExponentialTransform(lambda)` | >0 |
| 同上 | `extraFloat32Param1` | float/32 | off 52 | N/A | 值初始化 0 | 未读取 | inactive |
| 同上 | `from` | int64/64 | off 56 | N/A | 值初始化 0 | 未读取 | inactive |
| 同上 | `range` | uint64/64 | off 64 | N/A | 值初始化 0 | 未读取 | inactive |
| 同上 | `fromFp32` | float/32 | off 72 | N/A | 值初始化 0 | 未读取 | inactive |
| 同上 | `toFp32` | float/32 | off 76 | N/A | 值初始化 0 | 未读取 | inactive |
| 同上 | `splitBlockCount` | int64/64 | off 80 | split | `CalcSplitBlocks` | `ProcessWithSplitBlocks` loop | 1..8 |
| 同上 | `splitBlocks[i].numel` | int64/64 | off `88+40i` | element | slice state | SIMT `outputSize` | 32-bit indexable slice |
| 同上 | `splitBlocks[i].gmOffset` | int64/64 | off `96+40i` | element | `ReduceDimExtent` | launcher GM base | contiguous ND offset |
| 同上 | `splitBlocks[i].grid` | int64/64 | off `104+40i` | logical 256-thread group | execution policy | launcher 参数存在但函数体不直接用 `grid` | 1..624 |
| 同上 | `splitBlocks[i].totalThreads` | int64/64 | off `112+40i` | logical thread | `grid*256` | repeat/linear index domain | 正数 |
| 同上 | `splitBlocks[i].kernelOffset` | int64/64 | off `120+40i` | Philox result offset | accumulated offset | `PhiloxAlgParsInit` | 分片互不重叠 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 SIMT | key 3 | `sim_thread_exponential_tiling_arch35.cpp::BuildOpConfig/UniqueProcess` + `random_tiling_arch35.cpp` | `RandomUnifiedSimtTilingDataStruct` | `sim_thread_exponential.cpp::sim_thread_exponential` | `sim_thread_exponential_simt.h::Process<float>` + `random_kernel_base.h::PhiloxSimtKernelDiscontinuous` | `CMakeLists.txt` ascend950→arch35 |
| R-002 低精度 SIMT | key 1/2 | 同上 | 同上 | 同上 | half/BF16 模板实例 + 同一 common body | 同上 |

## Kernel 实现

### 公共执行流程

entry 注册 unified SIMT POD，声明 `KERNEL_TYPE_MIX_AIV_1_0`，按 key 调用 `Process<T>(self, &tilingData)`。`Process` 过滤多余外层 core，然后构造 launcher。每个外层 core 都按 POD 顺序遍历 split block，计算除法 magic/shift，向 SIMT VF 发起 512-thread 调用。SIMT 线程用 Philox key/counter 私有数组生成四个结果，四个 lane 分别映射到相隔 `totalThreads` 的输出位置；每次写前以 `li<numel` 保护。所有写位置唯一，不需要 atomic、shared memory 或 barrier。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnSimThreadExponential.md` | 两段式接口 | FP32、lambda>0、950 offset/count 约束 | executor |
| Host | `op_host/arch35/sim_thread_exponential_tiling_arch35.cpp` | `UniqueProcess` | output dtype neither FP16 nor BF16 after dtype rule | key 3 |
| Router | `op_kernel/sim_thread_exponential.cpp` | `sim_thread_exponential` | `TILING_KEY_IS(3)` | `Process<float>` |
| SIMT | `op_kernel/arch35/sim_thread_exponential_simt.h` | `ExponentialLauncher<float>` | compile-time T=float | discontinuous Philox VF |

#### Device 数据流

##### 函数级调用链

```text
sim_thread_exponential(self,...)
  -> Process<float>(self,POD)
     -> ProcessWithSplitBlocks
        -> GetUintDivMagicAndShift(totalThreads)
        -> ExponentialLauncher<float>::operator()
           -> VF_CALL<PhiloxSimtKernelDiscontinuous<float,ExponentialTransform<float>>>(512 threads)
              -> PhiloxAlgParsInit
              -> FlashCounter -> PhiloxRandomSimt -> 10*Philox4x32Round
              -> ExponentialTransform<float>::operator() -> GM write
```

##### 源码等价伪码

```text
if outerCore >= usedCoreNum: return
for each split block b in POD order:
  policy = magic_shift(b.totalThreads)
  gm = self + b.gmOffset
  SIMT_VF<512>:
    key = low_high(seed); counter = skip(offset=ceil(b.kernelOffset/4))
    physicalIdx = blockIdx.x*blockDim.x + threadIdx.x
    repeats = ceil(b.numel/(b.totalThreads*4))
    for loop in [0,repeats):
      for linear = physicalIdx; linear < b.totalThreads; linear += blockDim.x*gridDim.x:
        counterTmp = counter; FlashCounter(linear,loop)
        results[4] = Philox4x32_10(key,counterTmp)
        for lane in 0..3:
          li = linear + loop*b.totalThreads*4 + b.totalThreads*lane
          if li < b.numel:
            u = results[lane]*2^-32 + 2^-33
            logv = (u >= 1-eps/2) ? -eps/2 : simt_log(u)
            gm[li] = -logv/lambda
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `self` split | `self + split.gmOffset + li` | `split.numel` | FP32/contiguous ND slice | transform | caller | caller/executor，stream 完成后可读 | 单元素 volatile write |
| thread `key[2]` | thread local | 8 B logical | uint32 | seed split | Philox rounds | SIMT thread/call | 编译器寄存器布局不可见 |
| `counter/counterTmp[4]` | thread local | 各16 B logical | uint32 | offset/thread mapping | Philox | SIMT thread/iteration | copy then mutate tmp |
| `results[4]` | thread local | 16 B logical | uint32 | Philox | transform four lanes | inner iteration | 四路复用 |
| local memory | platform UB - 128 KiB | Host `SetLocalMemorySize` | opaque | framework | SIMT VF/compiler | entry | body 无显式 shared/dynamic allocation |

##### 分层内存布局

N/A：该 SIMT Route 不创建 AIV TQue/TBuf 或 shared/dynamic UB。实际层级是每线程寄存器/栈样式局部数组，编译器最终寄存器与 spill 字节数在固定源码不可见；Host 仅预留 128 KiB Dcache 并配置剩余 local memory。

##### 字节账与复用顺序

```text
UB_total = N/A（无 InitBuffer/TQue/TBuf）
dynamic_ub = N/A（VF_CALL 未传动态 UB；SetLocalMemorySize = platform_UB - 128KiB）
workspace_total = 0
thread_logical_locals = key 8B + counter 16B + counterTmp 16B + results 16B + scalars
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `counterTmp[4]` | CopyArray(counter) + FlashCounter | `PhiloxRandomSimt` | 下一 logical linear index | 单线程顺序 | 不跨线程共享 |
| `results[4]` | Philox | 四次 transform | 下一 inner iteration | 单线程顺序 | lane 写完后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread mapping | 无 barrier；私有状态 | FlashCounter | Philox | counterTmp | magic/shift 快除只产映射 |
| Philox | 10 rounds，私有数组 | counter/key | transform | results[4] | uint32 deterministic |
| transform | `Simt::Log` + scalar FP32 | uniform | GM write | thread local scalars | eps/2 近1保护 |
| output | `if(li<numel)` | transform | GM visibility by stream completion | unique GM element | 精确逐元素 tail，无 atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `random_common/op_host/arch35/random_tiling_arch35.cpp` | `CalcExecutionPoliciesForBlocks/WriteBackToContext` | split policy、blockDim、zero workspace |
| Kernel entry/router | `sim_thread_exponential/op_kernel/sim_thread_exponential.cpp` | key 3 | FP32 route |
| Device body/helper | `sim_thread_exponential_simt.h` + `random_kernel_base.h` | `ExponentialTransform<float>`, `PhiloxSimtKernelDiscontinuous` | 线程循环、Philox、FP32 GM write |

### Route R-002：FP16/BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnSimThreadExponential.md` | 两段式接口 | FP16/BF16 与 950 公共约束 | executor |
| Host | `sim_thread_exponential_tiling_arch35.cpp` | `UniqueProcess` | FP16→1；BF16→2 | dtype key |
| Router | `sim_thread_exponential.cpp` | key 1/2 branches | key 1 优先于 key 2 | half/BF16 Process |
| SIMT | `sim_thread_exponential_simt.h` | `ExponentialTransform<T>` | T=half/BF16 | FP32 transform 后 static_cast |

#### Device 数据流

##### 函数级调用链

```text
sim_thread_exponential
  -> Process<half|bfloat16_t>
  -> ProcessWithSplitBlocks
  -> VF_CALL<PhiloxSimtKernelDiscontinuous<T,ExponentialTransform<T>>>
  -> FP32 u/log/div -> static_cast<T> -> volatile GM write
```

##### 源码等价伪码

```text
execute exactly the same split, thread mapping, Philox and li<numel guard as R-001
x_fp32 = -protected_log(u)/lambda
gm_T[li] = static_cast<T>(x_fp32)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `self` split | `self + gmOffset + li` | split.numel | FP16/BF16 ND | transformed/cast scalar | caller | caller/executor | 单元素 volatile write |
| thread locals | 同 R-001 | key/counters/results + FP32 scalars | uint32/float | Philox/transform | static_cast/write | thread | 不共享 |
| workspace/shared UB | N/A | 0 | N/A | N/A | N/A | N/A | 无 |

##### 分层内存布局

N/A：与 R-001 相同，仅每线程最终输出类型变化；没有 AIV UB buffer、shared memory 或通信 window。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = N/A
workspace_total = 0
thread-local logical arrays = 56 B plus scalars；编译器物理寄存器占用不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread counter/results | 同 R-001 | Philox/transform | 下一 inner iteration | 单线程程序序 | 无跨线程 alias |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Philox/transform | 无共享同步 | private counter | private result | thread locals | FP32 运算 |
| 降精度写回 | C++ `static_cast<T>` | FP32 x | GM | scalar | 固定源码未指定 round mode |
| tail | `li<numel` | lane result | write | GM extent | 逐元素安全保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `sim_thread_exponential_tiling_arch35.cpp` | `UniqueProcess` | key 1/2 与 lambda |
| Kernel entry/router | `sim_thread_exponential.cpp` | key 1/2 | half/BF16 route |
| Device body/helper | `sim_thread_exponential_simt.h` | `ExponentialTransform<half/bfloat16_t>` | FP32 transform、static_cast、tail write |

## 算子特定优化

### Trick T-001：超过 32 位索引范围时切成最多 8 个独立 PRNG slice

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：整个输出的元素数或最大字节 offset 超过 `INT32_MAX`
- Golden 来源：[`random_common/op_host/arch35/random_tiling_arch35.cpp:145-177`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/random_common/op_host/arch35/random_tiling_arch35.cpp#L145-L177)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (state.Is32bitIndexable()) {
    simtTilingData.splitBlocks[0].numel = state.numel;
    simtTilingData.splitBlocks[0].gmOffset = state.gmOffset;
    simtTilingData.splitBlockCount = 1;
} else {
    TensorSliceState stack[MAX_SPLIT_BLOCKS];
    int32_t top = 0;
    stack[top++] = state;
```

#### 机制与设计原因

Host 先折叠连续维度，再沿字节 extent 最大的维度二分，直至每片的元素数与最大 byte offset 都可由 32 位索引表达。Device 仍用同一个 SIMT body，只改变 GM base、numel 与 counter offset；这避免在热循环中为所有索引付出宽地址路径，同时保留大张量可达性。各片 counter extent 累加，防止随机序列重叠。

#### 迁移边界

- 适用：连续或可明确描述 stride 的逐元素 counter-based 随机算子。
- 前提：slice 必须覆盖且不重叠，`gmOffset` 单位必须与 T 指针一致，counter extent 必须独立。
- 失效：超过 8 片仍不能 32 位索引时 Host 返回失败或无法形成完整覆盖，不能静默截断。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓的真实失败边界是 stack 容量不足时 `GRAPH_FAILED`；不能只切 GM 地址而复用相同 `kernelOffset`，否则不同片会复用随机数。

### Trick T-002：每线程一次 Philox 生成四个跨 stride 输出

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：`UNROLL=4`，所有 950 SIMT route
- Golden 来源：[`random_common/op_kernel/arch35/random_kernel_base.h:650-667`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/random_common/op_kernel/arch35/random_kernel_base.h#L650-L667)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
uint64_t repeatTime = (outputSize + totalThreads * UNROLL - 1) / (totalThreads * UNROLL);

for (uint64_t loopIdx = 0; loopIdx < repeatTime; loopIdx++) {
    for (uint64_t linearIndex = idx; linearIndex < totalThreads; linearIndex += blockDim.x * gridDim.x) {
        uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);

        FlashCounter(linearIndex, loopIdx, counterTmp);

        uint32_t results[SIMT_STEP];
        PhiloxRandomSimt(key, counterTmp, results);
```

#### 机制与设计原因

一轮 Philox4x32-10 的四个结果映射到 `linearIndex + iStep*totalThreads`，所以昂贵的十轮 counter 混合由四个输出摊销。物理线程数小于 Host logical `totalThreads` 时，内层循环以 `blockDim.x*gridDim.x` 跨步补齐逻辑域；`li<numel` 独立保护四个 lane 的尾部。

#### 迁移边界

- 适用：每个输出只消耗一个 32 位随机 lane 的逐元素变换。
- 前提：UNROLL 不得超过 `SIMT_STEP=4`，counter mapping 与 `totalThreads` 必须一致。
- 失效：每输出消耗多个随机数时要重算 lane 分配和 counter 跳步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若把四个 lane 写成连续索引却仍使用 discontinuous counter mapping，会改变确定性序列布局。

### Trick T-003：Philox 乘法结果复用 `uint2` 取高低 32 位

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：每次 `Philox4x32Round`
- Golden 来源：[`random_common/op_kernel/arch35/random_kernel_base.h:402-414`](../../../../../../../cann-ops-raw/ascendc/ops-math/165d41f02a43a62324b434cb6bda7c533b14da3e/random/random_common/op_kernel/arch35/random_kernel_base.h#L402-L414)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t res0_u64 = __mul_i32toi64(PHILOX_M4X32_A, counter[0]);
uint64_t res1_u64 = __mul_i32toi64(PHILOX_M4X32_B, counter[IDX_2]);

// 这里使用uint2和uint32功能等价，但是编译器的编译结果不一样，导致性能差异很大，使用性能更好的uint2
uint2& res0 = reinterpret_cast<uint2&>(res0_u64);
uint2& res1 = reinterpret_cast<uint2&>(res1_u64);
```

#### 机制与设计原因

64 位乘积一次产生高低两个 32 位词，源码通过 `uint2` 视图直接供 round 混合使用。注释明确这是功能等价但编译结果更优的形式；机制改变的是十轮 Philox 热路径的指令生成，不改变算法位级结果。

#### 迁移边界

- 适用：同一 950 SIMT 编译环境中的 32×32→64 乘积拆分。
- 前提：`uint2` lane 与目标 ABI 的低/高 word 顺序保持一致。
- 失效：跨编译器/端序/ABI 迁移必须先证明布局，不能直接假定。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码明确指出改用功能等价的逐 uint32 形式会生成不同且较差的代码；未给出量化性能数据。

## 模板与实现对比

| 对比维度 | R-001 FP32 SIMT | R-002 FP16/BF16 SIMT | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | key 3 | key 1/2 | output dtype |
| 模板与 dtype | `T=float` | `T=half/bfloat16_t` | Host/router 一致 |
| 循环与分核 | split + logical thread + unroll4 | 相同 | dtype 不改变 PRNG mapping |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程私有数组；无 shared UB | 相同 | SIMT GM 直写 |
| 内存总占用与复用 | workspace 0；128KiB Dcache | 相同 | 公共 config |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT VF，volatile FP32 write | SIMT VF，static_cast 后 write | 无 NDDMA/通信 |
| 同步、精度与 tail | 无 barrier；FP32；li guard | 无 barrier；FP32→T；li guard | 唯一地址写 |
| fallback/失败边界 | Host split/资源失败；未知 key无输出 | 同左 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，另见无后缀卡 | 支持，另见无后缀卡 | 支持 |
| Host/Launcher | arch32 专用 Tiling | arch32 专用 Tiling | arch35 `RandomTilingArch35` + split policy |
| Kernel/资源 | AIV_VECTOR、178176 B TBuf | 同左 | SIMT VF、线程私有状态、workspace 0 |

## 限制与失败边界

- 950 强制 `count == output storage numel`、`offset % 4 == 0`、`lambda > 0`；不满足即 Host `GRAPH_FAILED`。
- `count==0` 通过非负与 shape equality 后，`DoSimtBlockTiling` 可能以 `numOfPerCore=0` 调用 CeilDiv；代码中的“空 tensor 至少一核”注释不足以闭合除零前置条件，固定源码无可靠空域 Route。
- 分片最多 8 个；stack 容量不足时明确失败。while 因片数上限退出但仍有未处理 stack 的终点没有额外完整性检查，超大/复杂 stride 情形不可推测为成功。
- `splitBlocks[i].grid` 由 Host 写入并传给 launcher 形参，但 launcher body 不直接读取 `grid`；真正使用的是 `totalThreads`、VF 的 512 threads 和物理 `gridDim.x`。
- entry 将第一个 GM 参数 `self` 传给 950 body，第二个 `self_ref` 未使用；复用 direct launcher 时必须维持 ACLNN 原地 alias/参数绑定，不能假定输出总是第二个参数。
- `prob` 之外的多数字段为公共 POD inactive 字段；不得赋予本算子不存在的语义。POD ABI 的 408 B/offset 是按标准自然对齐推导，跨 ABI 应复核。
- 低精度写回只出现 `static_cast<T>`，固定源码没有显式舍入模式；不能声称与 arch32 `CAST_RINT` 完全相同。
- 自动生成 ACLNN bridge、框架连续化、SIMT 编译器寄存器/spill 与 stream 内部调度不可见；本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `random/sim_thread_exponential/docs/aclnnSimThreadExponential.md` | ACLNN 两段式原型 | 公开接口、950 约束与平台 |
| Op 注册 | `random/sim_thread_exponential/op_host/sim_thread_exponential_def.cpp` | `SimThreadExponential` | dtype/ND/AutoContiguous/ascend950 |
| 私有 Host/Tiling | `random/sim_thread_exponential/op_host/arch35/sim_thread_exponential_tiling_arch35.cpp` | `BuildOpConfig/UniqueProcess` | lambda/count/offset、SIMT config、key |
| 公共 Host/Tiling | `random/random_common/op_host/arch35/random_tiling_arch35.cpp` | `RandomTilingArch35` | 平台、split、policy、workspace、blockDim |
| POD | `random/random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h` | `RandomUnifiedSimtTilingDataStruct` | 字段与 split ABI |
| Kernel entry | `random/sim_thread_exponential/op_kernel/sim_thread_exponential.cpp` | `sim_thread_exponential` | task type 与 dtype router |
| 算子 Device body | `random/sim_thread_exponential/op_kernel/arch35/sim_thread_exponential_simt.h` | `Process/ExponentialLauncher/Transform` | launch 与指数变换 |
| 公共 Device helper | `random/random_common/op_kernel/arch35/random_kernel_base.h` | `ProcessWithSplitBlocks/PhiloxSimtKernelDiscontinuous` | 线程映射与 Philox |
| 构建接线 | `random/sim_thread_exponential/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | unified SIMT POD 与最多 8 个 split descriptors | 字段顺序、64 位单位、inactive 字段 | `RandomUnifiedSimtTilingDataStruct` |
| 调用顺序 | outer entry→按序 split VF→thread Philox→transform | 每片 gm/counter offset 与顺序 | `ProcessWithSplitBlocks` |
| workspace/内存 | workspace 0、Dcache 128KiB、线程私有状态 | 不虚构 shared UB；保留 local-memory 配置 | `BuildOpConfig/WriteBackToContext` |
| 同步 | 无 barrier/atomic，依赖唯一地址与单线程序 | split 不重叠、li guard | `PhiloxSimtKernelDiscontinuous` |
| 精度与尾块 | FP32 log/div、近1保护、static_cast、逐元素 tail | epsilon、lane mapping、`li<numel` | `ExponentialTransform` |
| 失败边界 | Host constraint/split/platform 失败阻断 | 空域、8片上限、未知 key、参数 alias | `DoTiling`, entry router |

- [__mul_i32toi64](../../../../apis/simt_api/math_functions/integer_math_functions/_mul_i32toi64.md)
