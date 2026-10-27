---
type: Operator
title: "ForeachLog10 950 ACLNN 算子知识"
description: "说明 aclnnForeachLog10 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/docs/aclnnForeachLog10.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_host/foreach_log10_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_host/arch35/foreach_log10_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_kernel/arch35/foreach_log10_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_kernel/arch35/foreach_log10_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_kernel/arch35/foreach_log10.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_kernel/arch35/foreach_log10_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_host/config/ascend950/foreach_log10_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/scripts/kernel/binary_config/ascendc_config.json
    role: implementation
created_at: "2026-08-25T15:45:37Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachLog10 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachLog10GetWorkspaceSize` + `aclnnForeachLog10`，CANN 9.1.0 的无版本后缀公开 ACLNN 两段式接口 |
| 计算语义 | 对列表中每个张量逐元素计算 `y_i[j] = log10(x_i[j])` |
| 输入输出 | 动态 TensorList；ND；FLOAT16/FLOAT32/BFLOAT16；输入可非连续，输出不支持非连续 |
| 执行类型 | 单 Kernel，SIMT（AIV outer entry 内 `asc_vf_call`） |
| Kernel 路线 | R-001 FLOAT `log10f`；R-002 FLOAT16/BFLOAT16 `hlog10` |
| 核心资源 | GM TensorList 指针表、GM POD；1024 threads/block；无 AIV Queue/TBuf、无 workspace、无共享/动态 UB 数据结构 |
| 来源闭包 | `source_closed`：公开 ACLNN 契约→OpDef/公共 infer→arch35 Tiling/POD/template key→SIMT entry/body→950 构建及 binary config 已闭合；自动生成 ACLNN bridge 和 `asc_vf_call` runtime 属于框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`、`y=[y_0,...,y_(n-1)]`。对每个 `i` 与 `0<=j<numel(x_i)`，结果为 `y_i[j]=log10(x_i[j])`。FLOAT 路线调用 `log10f`；FLOAT16 和 BFLOAT16 路线分别以自身类型调用 `hlog10`。算子不对零、负数、NaN 或 Inf 做额外分支。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 0–8 维，ND | FLOAT16/FLOAT32/BFLOAT16 | 非空指针；列表内 dtype 一致；支持空 Tensor；输入非连续由 OpDef `AutoContiguous()` 的生成框架边界处理。 |
| `out` | TensorList；每个 Tensor 0–8 维，ND | 与 `x` 相同 | 非空指针；输出不支持非连续；公开 ACLNN 契约要求容量不小于对应输入，Kernel 只写 `numel(x_i)`。 |
| `workspaceSize`/`executor` | 标量/执行器 | `uint64_t*`/`aclOpExecutor**` | 第一段返回；950 Tiling 设置 workspace 为 0。 |
| `workspace`/`stream` | Device 地址/Stream | N/A | 第二段消费；Kernel entry 参数含 workspace 但 body 不读。 |

### 入口与版本

固定仓只公开无版本后缀入口，未发现同语义的更高版本。ACLNN 文档和 README 明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 把它映射到 `arch35`，Kernel CMake 选择 `arch35/foreach_log10.cpp`，950 binary config 和全局 binary config 都列出 `ascend950` 三种 dtype。生成的 ACLNN inner bridge 未提交，公开文档中的第一段校验与 executor 生命周期作为框架边界；算子自有 Host/Tiling、POD、SIMT router/body 和构建均已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段按公开契约检查空指针、dtype、输入输出一致性、shape 容量、列表内一致性与最多 8 维；错误为 `161001`、`161002` 或 `561002`。
2. OpDef 声明动态 ND TensorList 和 `AutoContiguous()`；公共 infer 要求输入实例数等于输出数，并逐项传播输入 shape/dtype。
3. `ForeachLog10TilingFunc` 获取 AIV 核数和 UB 大小；二者为 0 或平台对象为空即失败。它从动态输入实例信息取得 `tensorNum`，从第一个输入描述取得统一 dtype。
4. Tiling 先清零 `ForeachLog10TilingData`；若 `tensorNum>256` 显式失败。它逐项写入 storage shape 元素数并累加 `totalElements`，随后计算 blockDim、workspace、local-memory budget 和 dtype template key。

### 参数、Workspace 与生命周期

设 `N=sum_i numel(x_i)`，平台 AIV 核数为 `C`，UB 容量为 `U`。

- `tensorCount=tensorNum<=256`，`tensorElements[i]=numel(x_i)`，`totalElements=N`。
- `N==0` 时 `needCoreNum=blockDim=1`；否则 `needCoreNum=blockDim=min(C,ceil(N/1024))`。这里 1024 的单位是元素/目标 core，不是线程数，恰与 `THREAD_NUM=1024` 数值相同但来源独立。
- `workspace[0]=0`。
- `SetLocalMemorySize(uint32_t(U-128*1024))` 设置 outer launch 的 local-memory 大小；SIMT body 没有声明或索引共享/dynamic UB，因此该容量不进入数据字节账。
- executor 持有编排并在传入 stream 上启动一个 outer AIV Kernel；每个 outer block 调用 `asc_vf_call` 创建 1024-thread SIMT 执行。生成 bridge 的临时连续化对象 owner/cleanup 固定仓不可见。

### 启动图与失败传播

```text
aclnnForeachLog10GetWorkspaceSize(x,out)
  -> [生成框架：校验/可能的输入 Contiguous/executor]
  -> ForeachLog10TilingFunc
     -> POD + blockDim + localMemorySize + key; workspaceSize=0
aclnnForeachLog10(nullptr,0,executor,stream)
  -> foreach_log10<schMode><<<blockDim,stream>>>(xList,yList,nullptr,tiling)
     -> NsForeachLog10::Process<T>
        -> asc_vf_call<OpForeachLog10Simt<T>>(dim3(1024), tensorCount, elemCounts, x, y)
```

只有一个 Device 节点，无 Kernel 间 workspace 边。Host `SetBlockDim(needCoreNum)` 决定 SIMT 可见的 `gridDim.x/numBlocks`；`asc_vf_call(dim3(1024))` 决定每个 block 的 1024 threads。Host 平台、POD、dtype 或 tensorCount 检查失败时不 launch；entry 是编译期 `schMode` 分发，不存在运行时 unknown-key fallback。

## Route 与参数设计

### 分核与资源计算

Host 只用列表总元素数选择 outer block 数，不预先把张量切给特定 block。Device 对每个张量依次执行同一 global-stride 循环：`tid=blockIdx.x*blockDim.x+threadIdx.x`，`stride=blockDim.x*gridDim.x=1024*needCoreNum`，线程处理 `idx=tid,tid+stride,...<tensorElements[t]`。因此每个张量均由全部 blocks/threads 协作，张量之间串行推进，但各线程在每个张量内访问互不重叠的元素。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FLOAT | `GET_TPL_TILING_KEY(0)`；`foreach_log10<schMode=0>` | `dataType==DT_FLOAT` | FLOAT16/BFLOAT16 继续后续 dtype 分支；其它失败 |
| 2 | R-002 16-bit | `GET_TPL_TILING_KEY(1/2)`；`foreach_log10<1/2>` | 前者 `DT_FLOAT16`，后者 `DT_BF16`；按源码 else-if 顺序 | 其它 dtype `GRAPH_FAILED` |

```text
DT_FLOAT   -> schMode 0 -> R-001 Process<float> -> log10f
DT_FLOAT16 -> schMode 1 -> R-002 Process<half> -> hlog10
DT_BF16    -> schMode 2 -> R-002 Process<bfloat16_t> -> hlog10
otherwise  -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FLOAT | schMode 0 | 950；FLOAT32；tensorCount≤256；ACLNN 校验通过 | 单 outer AIV entry→单 SIMT VF body | `Process<float>`→`OpForeachLog10Simt<float>`→`Log10Func<float>/log10f` | GM load→register scalar→GM store；无共享缓冲、barrier 或 atomic | `idx<count` 保护；空 tensor `continue`；无 fallback |
| R-002 16-bit | schMode 1/2 | 950；FLOAT16 或 BFLOAT16；其余同上 | 同一单节点 topology | `Process<half/bfloat16_t>`→`OpForeachLog10Simt<T>`→`Log10Func<T>/hlog10` | 同上，标量保持输入 dtype | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList 上限 | `tensorNum<=256` | POD `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `ForeachLog10TilingFunc`; `ForeachLog10TilingData` |
| 最小每核元素 | `needCoreNum=min(C,ceil(N/1024))`，N=0 特判1 | 避免总元素较少时启动所有 core；SIMT block 固定 1024 threads | 全部 | 无另一 Route | `MIN_PER_CORE`; `ForeachLog10TilingFunc` |
| 固定 thread block | `THREAD_NUM=1024`; `LAUNCH_BOUND(1024)` | `tid/stride` 映射依赖 `blockDim.x=1024` | 全部 | 无不同 threadsPerBlock fallback | `OpForeachLog10Simt`; `Process` |
| UB/DCACHE 预留 | 仅检查 `U!=0`，然后 `uint32_t(U-128KiB)` | launch local-memory budget 保留 128 KiB DCACHE | 全部 | 对 `U<128KiB` 无显式失败，可能无符号下溢 | `GetPlatformInfo`; `SetLocalMemorySize` |
| 空张量与尾线程 | `count==0` skip；循环条件 `idx<count` | 防止空指针数据访问和最后 stride 越界 | 全部 | skip/线程自然退出 | `OpForeachLog10Simt` |

### POD/ABI

该 POD 是普通 C++ struct；按字段声明和标准自然对齐，`alignof=8`、`sizeof=2064`。Kernel 还用 `offsetof(ForeachLog10TilingData,tensorElements)` 取得数组地址，Host/Device 共享同一头文件。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachLog10TilingData` | `needCoreNum` | `int32_t`/32 | offset 0；struct align 8/sizeof 2064 | outer blocks | N=0→1，否则 `int32_t(min(C,ceil(N/1024)))` | Device 未读取；block count 由 `SetBlockDim` 生效 | 非活动字段仍被写入 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4 | 张量数 | `int32_t(tensorNum)` | `Process` 读后传给 SIMT tensor loop | 0–256；空列表在读取首 inputDesc 前的行为不闭合 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | 元素 | 所有 `numel` 累加 | Device 未读取；只支撑 Host 分核 | 累加溢出未检查 |
| 同上 | `tensorElements` | `int64_t[256]` | offset 16；extent 2048 B | 每张量元素 | `storageShape.GetShapeSize()` | `offsetof` 得数组 GM 指针；SIMT `tensorElements[t]` | 未使用尾项由 `memset_s` 清零 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FLOAT | schMode 0 | `op_host/arch35/foreach_log10_tiling.cpp::ForeachLog10TilingFunc` | `ForeachLog10TilingData` | `op_kernel/arch35/foreach_log10.cpp::foreach_log10<0>` | `foreach_log10_simt.h::Process<float>/OpForeachLog10Simt<float>/Log10Func<float>` | Host/Kernel CMake；ascend950 binary JSON；global binary config |
| R-002 16-bit | schMode 1/2 | 同上 dtype branches | 同上 | `foreach_log10<1/2>` | `Process<half/bfloat16_t>`；`Log10Func<half/bfloat16_t>` | 同上 FP16/BF16 条目 |

## Kernel 实现

### 公共执行流程

模板 entry 用编译期 `schMode` 选择 dtype，然后把原始 tiling GM 指针传给 `Process<T>`。`Process` 不复制 POD 数组：它用 `offsetof` 直接构造 `tensorElements` 的 GM 指针，只把 `tensorCount` 读为标量，随后 `asc_vf_call` 以 1024 threads/block 进入 SIMT body。SIMT body逐 Tensor 取 GM 实际地址，按 global stride 对各元素做一次 load、一次 log10、一次 store。`workspace`、`needCoreNum`、`totalElements` 在 Device body 中均不读取。

### Route R-001：FLOAT log10f

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_log10_def.cpp` | `ForeachLog10` | FLOAT32、950 config | arch35 Host |
| Host/Tiling | `op_host/arch35/foreach_log10_tiling.cpp` | `ForeachLog10TilingFunc` | `dataType==ge::DT_FLOAT` | schMode 0 |
| Router | `op_kernel/arch35/foreach_log10.cpp` | `foreach_log10<schMode>` | `if constexpr schMode==TILING_KEY_FLOAT` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log10<0>(xList,yList,workspace,tiling)
  -> NsForeachLog10::Process<float>
     -> offsetof tensorElements + read tensorCount
     -> asc_vf_call<OpForeachLog10Simt<float>>(dim3(1024),...)
        -> for each tensor -> SimtGetTensorAddr<float>
        -> global-stride loop -> Log10Func<float> -> log10f -> GM store
```

##### 源码等价伪码

```text
# 伪码，每个 SIMT thread 执行
tid = blockIdx.x * 1024 + threadIdx.x
stride = 1024 * gridDim.x
for t in [0,tensorCount):
  count = tensorElements[t]
  if count == 0: continue
  xData = tensor_addr<float>(xList,t); yData = tensor_addr<float>(yList,t)
  for idx = tid; idx < count; idx += stride:
    value = xData[idx]
    yData[idx] = log10f(value)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM pointer list | `base + (*uint64(base)>>3)`，再取 `[t]` | `tensorCount` 个 64-bit 地址 | TensorList descriptor | ACLNN framework | `SimtGetTensorAddr` | executor/Kernel launch | `>>3` 要求首 offset 按 8 B 编码 |
| GM `x_t` | `xPtr[t]+idx` | `tensorElements[t]` | FLOAT/ND linear | caller | SIMT thread | caller；只读 | global stride 保证 thread 间不重叠 |
| thread scalar | register/local scalar `val` | 1 FLOAT | FLOAT | GM load | `log10f`/store | 当前线程当前迭代 | 固定仓不暴露寄存器字节数 |
| GM `y_t` | `yPtr[t]+idx` | 输入有效元素数 | FLOAT/ND linear | SIMT thread | caller | caller；stream 完成后有效 | 每元素单写者 |

##### 分层内存布局

N/A：该 SIMT body 没有 `TQue/TBuf/LocalTensor`、shared/dynamic UB 或中间 workspace；真实替代路径是 GM scalar load→线程标量→GM scalar store。Host 虽设置 `localMemorySize=U-128KiB`，body 未索引该区域。

##### 字节账与复用顺序

```text
AIV_UB_total = 0 B（无 InitBuffer）
SIMT shared/dynamic data buffer = 0 B（无声明/索引）
launch_local_memory_budget = uint32_t(U - 128 KiB)
workspace_total = 0 B
register_bytes = 固定仓不可见；每线程仅能证明 val 的单迭代生命周期
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `val` | 当前 idx 的 GM load | `log10f` | 同线程下一 idx | 单线程程序顺序 | 不跨线程共享；无需 barrier |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Tensor loop | `count==0 continue` | POD count | 地址解析 | 空 Tensor | 不访问空 Tensor 数据地址 |
| Element compute | SIMT program order | GM load | `log10f`→GM store | 单元素 | FLOAT32 输入/输出；无 cast |
| Tail | `idx<count` | tid/stride | loop body | 最后不满 stride 的元素 | 越界线程不执行；无 padding lane |
| Cross-thread | N/A | 每线程独立 idx | N/A | GM output | 单写者，无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_log10_tiling.cpp` | `ForeachLog10TilingFunc` | tensorCount/numel、blockDim、local memory、key/workspace |
| Kernel entry/router | `op_kernel/arch35/foreach_log10.cpp` | `foreach_log10<0>` | FLOAT key 到 `Process<float>` |
| Device body/helper | `op_kernel/arch35/foreach_log10_simt.h` | `Process<float>`; `OpForeachLog10Simt<float>`; `Log10Func<float>` | 1024 threads、global stride、log10f、地址与 tail |

### Route R-002：FLOAT16/BFLOAT16 hlog10

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_log10_def.cpp` | `ForeachLog10` | FLOAT16 或 BFLOAT16、950 config | arch35 Host |
| Host/Tiling | `op_host/arch35/foreach_log10_tiling.cpp` | `ForeachLog10TilingFunc` | `DT_FLOAT16`→mode1；`DT_BF16`→mode2 | 对应 compile-time key |
| Router | `op_kernel/arch35/foreach_log10.cpp` | `foreach_log10<1/2>` | mode1/mode2 `if constexpr` | `Process<half/bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log10<1 or 2>
  -> Process<half or bfloat16_t>
     -> asc_vf_call<OpForeachLog10Simt<T>>(dim3(1024),...)
        -> tensor loop + SimtGetTensorAddr<T>
        -> global-stride loop -> Log10Func<T> -> hlog10 -> GM store
```

##### 源码等价伪码

```text
# 伪码；T=half 或 bfloat16_t
tid = blockIdx.x*1024 + threadIdx.x
stride = 1024*gridDim.x
for t in [0,tensorCount):
  count = tensorElements[t]
  if count==0: continue
  for idx=tid; idx<count; idx+=stride:
    T val = xData[t][idx]
    yData[t][idx] = hlog10(val)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM pointer list | 与 R-001 相同 | `tensorCount` 个地址 | descriptor | framework | `SimtGetTensorAddr<T>` | executor/launch | 8 B offset 协议 |
| GM `x_t/y_t` | `tensorPtr[t]+idx` | `tensorElements[t]` | FLOAT16 或 BFLOAT16/ND | caller/线程 | 线程/caller | caller | global-stride 单写者 |
| thread scalar | `T val` | 1 个 16-bit 逻辑值 | half/bfloat16_t | GM load | hlog10/store | 单迭代 | 实际寄存器分配固定仓不可见 |

##### 分层内存布局

N/A：无 AIV UB Queue、SIMT shared/dynamic UB 或 workspace；路径为 GM→线程标量→`hlog10`→GM。Host local-memory budget 未被 body 寻址。

##### 字节账与复用顺序

```text
AIV_UB_total = 0 B
SIMT shared/dynamic data buffer = 0 B
launch_local_memory_budget = uint32_t(U - 128 KiB)
workspace_total = 0 B
register_bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `val` | GM load | `hlog10`/GM store | 下一 idx | 单线程程序顺序 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Compute | SIMT program order | 16-bit GM load | `hlog10`→16-bit GM store | 单元素 | 源码未显式升到 FP32；结果类型与输入相同 |
| Empty/tail | `count==0`; `idx<count` | POD | loop | GM 范围 | 空 Tensor skip，最后 stride 越界线程不执行 |
| Cross-thread | N/A | 独立 idx | N/A | output | 无 barrier、atomic、共享状态 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_log10_tiling.cpp` | dtype branches；`ForeachLog10TilingFunc` | mode1/2、blockDim/POD |
| Kernel entry/router | `op_kernel/arch35/foreach_log10.cpp` | `foreach_log10<1/2>` | half/BF16 模板映射 |
| Device body/helper | `op_kernel/arch35/foreach_log10_simt.h` | `Log10Func<half>`; `Log10Func<bfloat16_t>`; `OpForeachLog10Simt<T>` | `hlog10`、global stride、empty/tail |

## 算子特定优化

### Trick T-001：单次 SIMT 启动覆盖整个异长 TensorList

#### 触发与源码依据

- 适用 Route：R-001、R-002
- 触发条件：`tensorCount<=256`，Host 已用总元素数选定 `gridDim.x`，每个 block 固定 1024 threads。
- Golden 来源：[`foreach/foreach_log10/op_kernel/arch35/foreach_log10_simt.h:82-91`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log10/op_kernel/arch35/foreach_log10_simt.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    blockIdx.x * blockDim.x +
    threadIdx.x);
uint64_t stride = static_cast<uint64_t>(
    blockDim.x * gridDim.x);

for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    T val = xData[idx];
    yData[idx] = Log10Func<T>(val);
}
```

#### 机制与设计原因

同一组 blocks/threads 对每个 Tensor 复用 global-stride 映射，异长张量无需 Host 为每项生成独立 tile/POD 区间或多个 Kernel launch。`idx<count` 同时承担每个 Tensor 的尾保护；每个输出索引只有一个线程写，因此不需要 atomic 或 barrier。源码可证明它减少了逐 Tensor 启动和中间物化，但没有真机性能数据。

#### 迁移边界

- 适用：TensorList 中各元素完全独立、每个元素成本近似且无需跨元素归约的 unary 算子。
- 前提：TensorList 指针表协议一致，POD 提供逐 Tensor 元素数，threadsPerBlock 固定值与 `tid/stride` 一致。
- 失效：需要跨 Tensor 同步、共享状态、归约或不同输出布局时，必须增加相应 Route/同步与内存设计。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FLOAT | R-002 16-bit | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | mode0 / FLOAT | mode1 FP16、mode2 BF16 | dtype else-if selector |
| 模板与 dtype | `Process<float>` | `Process<half/bfloat16_t>` | binary dtype specialization |
| 循环与分核 | `ceil(total/1024)` blocks，1024 threads，global stride | 完全相同 | 与 dtype 无关 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/shared UB；线程 FLOAT scalar | 无 Queue/shared UB；线程16-bit scalar | scalar SIMT path |
| 内存总占用与复用 | workspace/显式 UB 0 | 同左 | local-memory budget 未消费 |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM load/store + `log10f` | scalar GM load/store + `hlog10` | math overload |
| 同步、精度与 tail | 无 cast/barrier；`idx<count` | 无显式升精度；`idx<count` | 元素独立 |
| fallback/失败边界 | 无 fallback | 无 fallback | 其它 dtype Host 失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 foreach Tiling，workspace 32 B | 与 A2 共用 | arch35 专用 Tiling，workspace 0，local memory `U-128KiB` |
| Kernel/资源 | AIV Vector，POD/Queue，最多闭合 50 张量 | 与 A2 共用 | SIMT，1024 threads/block，无显式 UB buffer，最多 256 张量 |

## 限制与失败边界

- 950 显式拒绝 `tensorCount>256`；整个空列表在读取 `GetDynamicInputDesc(0,0)` 时的行为不闭合，不能由“支持空 Tensor”推导支持空 TensorList。
- `totalElements` 的 `int64_t` 累加、`ceil(N/1024)` 与后续窄化没有溢出检查；极端超大 shape 不属于已验证域。
- 仅检查 UB 非零，没有检查 `U>=128KiB`；`U-DCACHE_SIZE` 是无符号运算后窄化为 `uint32_t`，极小 UB 平台会下溢，且无 fallback。
- `needCoreNum` 与 `totalElements` 在 Device 中不读取；修改 POD 时仍不能删除或重排，因为 Host/Device共享完整 struct ABI。
- `asc_vf_call` 的底层 cooperative-group/寄存器分配与 stream 可见性由 runtime primitive 保证，固定仓不包含其实现；算子 body 本身无 shared/dynamic UB、barrier 或 atomic。
- FLOAT16/BFLOAT16 路线没有显式 FP32 升精度；`hlog10` 的内部近似与异常值处理属于 SIMT math 库边界。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_log10/docs/aclnnForeachLog10.md` | 两个 ACLNN 原型 | 参数、错误、executor/stream |
| 平台/语义 | `foreach/foreach_log10/README.md` | 产品支持表、公式 | 950 公开支持、dtype/layout |
| OpDef | `foreach/foreach_log10/op_host/foreach_log10_def.cpp` | `ForeachLog10` | dtype、AutoContiguous、ascend950 config |
| infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`; `InferDataType4ForeachCommon` | 输出数/shape/dtype |
| Host/Tiling | `foreach/foreach_log10/op_host/arch35/foreach_log10_tiling.cpp` | `GetPlatformInfo`; `ForeachLog10TilingFunc` | 校验、POD、blockDim、local memory、workspace/key |
| POD | `foreach/foreach_log10/op_kernel/arch35/foreach_log10_tiling_data.h` | `ForeachLog10TilingData` | Host/Device ABI |
| template key | `foreach/foreach_log10/op_kernel/arch35/foreach_log10_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个 schMode 实例 |
| Kernel entry | `foreach/foreach_log10/op_kernel/arch35/foreach_log10.cpp` | `foreach_log10<schMode>` | compile-time dtype router |
| Kernel body | `foreach/foreach_log10/op_kernel/arch35/foreach_log10_simt.h` | `Process`; `OpForeachLog10Simt`; `Log10Func`; `SimtGetTensorAddr` | VF launch、地址、线程循环、math/tail |
| Host 构建 | `foreach/foreach_log10/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`; `SUPPORT_TILING_DIR`; `ACLNNTYPE aclnn` | ascend950→arch35 与 ACLNN 接线 |
| Kernel 构建 | `foreach/foreach_log10/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_log10.cpp` | 950 Kernel 选择、AUTO_SYNC false |
| 950 binary | `foreach/foreach_log10/op_host/config/ascend950/foreach_log10_binary.json` | 三个 op_list 条目 | 三 dtype 二进制实例 |
| 全局 binary | `scripts/kernel/binary_config/ascendc_config.json` | `ForeachLog10` | ascend950 活动 compute unit 与 auto_sync=false |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B POD、4 个字段、256 项 numel 数组 | offset 0/4/8/16、字段位宽/顺序 | `ForeachLog10TilingData`; `offsetof` consumer |
| 调用顺序 | outer entry→`Process<T>`→`asc_vf_call`→逐 Tensor/global stride | 1024 threads 与 tid/stride 公式 | `foreach_log10`; `Process`; `OpForeachLog10Simt` |
| workspace/内存 | workspace 0，无显式 UB/shared buffer | local memory `U-128KiB` 及 TensorList GM 地址协议 | `ForeachLog10TilingFunc`; `SimtGetTensorAddr` |
| 同步 | 每元素单写者、线程程序顺序 | 不能在需要共享状态时沿用无 barrier/atomic 设计 | `OpForeachLog10Simt` |
| 精度与尾块 | FLOAT `log10f`；FP16/BF16 `hlog10`；`idx<count` | dtype specialization与空 Tensor skip | `Log10Func`; `OpForeachLog10Simt` |
| 失败边界 | 平台/UB/core/dtype/tensorCount检查 | 不得为 >256、U<128KiB、未知 dtype 添加无证据 fallback | `GetPlatformInfo`; `ForeachLog10TilingFunc` |

- [Log10f](../../../../apis/simt_api/math_functions/float_math_functions/log10f.md)
