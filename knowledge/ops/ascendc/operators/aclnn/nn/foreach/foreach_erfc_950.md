---
type: Operator
title: "ForeachErfc 950 ACLNN 算子知识"
description: "说明 aclnnForeachErfc 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/docs/aclnnForeachErfc.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_host/foreach_erfc_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_host/arch35/foreach_erfc_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_host/arch35/foreach_erfc_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_kernel/arch35/foreach_erfc.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_host/config/ascend950/foreach_erfc_binary.json
    role: implementation
created_at: "2026-08-25T15:31:07Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachErfc 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnForeachErfcGetWorkspaceSize` + `aclnnForeachErfc`；950 内部算子类型 `ForeachErfc` |
| 计算语义 | TensorList 中每个元素独立计算 `erfc` |
| 输入输出 | ND 动态 TensorList；FP16、FP32、BF16；每项 0–8 维；输出 dtype 与输入一致、storage size 不小于输入 |
| 执行类型 | 单 Kernel；arch35 outer entry 内调用一个 `SIMT` VF 节点，无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001/002/003：FP16/FP32/BF16 + int32 索引；R-004/005/006：FP16/FP32/BF16 + int64 索引 |
| 核心资源 | TensorList 指针表 GM、tiling GM 内 cumulative offsets、SIMT thread 私有标量；无显式 AIV UB/TQue/TBuf/shared/atomic/workspace |
| 来源闭包 | `source_closed`：公开 ACLNN 约束→950 OpDef/config→arch35 Tiling/POD/key→templated entry→SIMT grid-stride body→arch35 构建接线已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 TensorList 第 `i` 项的所有有效元素 `j` 执行：

`y_i[j] = erfc(x_i[j]) = 1 - erf(x_i[j]) = 2/sqrt(pi) * integral(x_i[j], +inf, exp(-t^2) dt)`。

Host 把列表各项按顺序展平为长度 `totalElements=sum_i numel(x_i)` 的逻辑域；Device 用前缀和 `cumulativeOffsets` 将扁平索引还原为 `(tensorId, localIdx)`。输出比输入多出的元素不写。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 0–8 维；ND | FP16、FP32、BF16 | 输入可非连续并由 `AutoContiguous` 框架处理；公开接口支持空 Tensor；同一列表 dtype 一致 |
| `out` | TensorList；每项 0–8 维；ND | 与 `x` 一致 | 不支持非连续输出；对应项 storage size `>= x_i`；列表 dtype 一致 |
| `workspaceSize`/`workspace` | Device workspace | `uint64_t`/地址 | arch35 Tiling 将算子 workspace 设为 0，Kernel 参数保留但不读写；框架 Contiguous 额外 workspace 固定仓不可见 |
| `executor`/`stream` | Host executor/运行时流 | N/A | 第一段生成，第二段在指定 stream 发射单 Kernel |

### 入口与版本

固定仓公开文档明确支持 Ascend 950PR/950DT，OpDef 注册 `ascend950`，Host CMake 将其映射到 `arch35`，Kernel CMake显式选择 `arch35/foreach_erfc.cpp`，binary config 为三种 dtype 各发布一个 ND 动态输入/输出 binary。这是 950 实际支持的最新公开版本。生成后的 ACLNN C bridge/header 不在固定仓中，属于框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口负责空指针、dtype、输入输出 dtype、shape size、列表内 dtype 和维数检查。arch35 Tiling 自身按以下顺序执行：

1. 从平台取 AIV core 数与 UB；任一非正失败。
2. 要求 `UB > 128 KiB`，随后把 128 KiB 作为 DCache 保留，`localMemorySize=UB-128 KiB`。
3. 读取动态输入 0 的实例数，要求 `tensorNum<=256`。
4. 逐项读取 storage shape，累计 `totalElements` 并写包含式前缀终点 `cumulativeOffsets[i]=sum_{k<=i} numel(x_k)`；dtype 只从第 0 项描述符读取。
5. OpDef 的 `AutoContiguous` 处理位于生成 executor 边界；确切节点与 workspace 固定仓不可见。

### 参数、Workspace 与生命周期

设 `M=totalElements`、平台 AIV core 数为 `C`：

- `maxNeededCoreNum=ceil(M/1024)`。
- `needCoreNum=max(min(C,maxNeededCoreNum),1)`，并作为 outer Kernel `blockDim`。
- `perCoreElements=ceil(M/needCoreNum)` 写入 POD，但当前 Device body不读取。
- `SetLocalMemorySize(UB-128 KiB)`；SIMT body未显式声明 dynamic/shared UB，固定仓不能把该 local-memory 配额细分到 `erfcf` 内部。
- `workspace[0]=0`，Kernel 的 `workspace` 形参未使用。
- Tiling GM、TensorList 元数据及每项 input/output GM 在异步 stream 执行完成前保持有效；没有跨 Kernel 中间边。

### 启动图与失败传播

```text
aclnnForeachErfc executor
  -> outer foreach_erfc<schMode><<<needCoreNum>>>(xList,yList,workspace,tiling)
    -> Process<T>
      -> asc_vf_call<OpForeachErfcSimtKernel<T,IDX_T>>(threads)
```

这是一个 Host Kernel 节点；其语义计算在内嵌 SIMT VF 完成。`needCoreNum` 决定 `gridDim.x`；32 位索引 VF 每 block 1024 threads，64 位索引 VF 每 block 512 threads。Tiling 的平台/UB/元数据错误返回 `GRAPH_FAILED` 并阻断 launch；`totalElements==0` 时已发射的 outer entry 在 `Process` 内直接返回；不支持的 `schMode` 没有 body/fallback。

## Route 与参数设计

### 分核与资源计算

outer block 数按每核至少约 1024 个逻辑元素限制：`grid=min(C,ceil(M/1024))`，再下限 1。每个 SIMT thread 的初始扁平索引为 `blockIdx.x*blockDim.x+threadIdx.x`，步长为 `blockDim.x*gridDim.x`，因此所有 outer blocks 共同覆盖 `[0,M)`。`M<=INT32_MAX` 选 int32 索引并用 1024 threads/block；否则用 int64 与 512 threads/block。前缀数组最多 256 项，每个元素通过线性扫描找到首个 `flatIdx<cumulativeOffsets[t]` 的 Tensor。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16/i32 | schMode 0；`<half,int32_t>` | dtype FP16 且 `0<M<=INT32_MAX` | FP16 超阈值→R-004 |
| 2 | R-002 FP32/i32 | schMode 1；`<float,int32_t>` | dtype FP32 且 `0<M<=INT32_MAX` | FP32 超阈值→R-005 |
| 3 | R-003 BF16/i32 | schMode 2；`<bfloat16_t,int32_t>` | dtype BF16 且 `0<M<=INT32_MAX` | BF16 超阈值→R-006 |
| 4 | R-004 FP16/i64 | schMode 0；`<half,int64_t>` | dtype FP16 且 `M>INT32_MAX` | 无 fallback |
| 5 | R-005 FP32/i64 | schMode 1；`<float,int64_t>` | dtype FP32 且 `M>INT32_MAX` | 无 fallback |
| 6 | R-006 BF16/i64 | schMode 2；`<bfloat16_t,int64_t>` | dtype BF16 且 `M>INT32_MAX` | 无 fallback |

```text
dtype -> FP16:key0 / FP32:key1 / BF16:key2
  -> Process<T>: M==0 -> return
  -> M<=INT32_MAX -> IDX_T=int32_t, threads=1024
  -> M> INT32_MAX -> IDX_T=int64_t, threads=512
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 0 + int32 guard | FP16、950、`0<M<=INT32_MAX` | 单 outer Kernel→SIMT VF | `OpForeachErfcSimtKernel<half,int32_t>` | 1024 threads；GM direct load/store；无 barrier | grid-stride `flatIdx<M` |
| R-002 | 1 + int32 guard | FP32、其余同上 | 同上 | `<float,int32_t>` | 同上 | 同上 |
| R-003 | 2 + int32 guard | BF16、其余同上 | 同上 | `<bfloat16_t,int32_t>` | 同上 | 同上 |
| R-004 | 0 + int64 guard | FP16、950、`M>INT32_MAX` | 单 outer Kernel→SIMT VF | `<half,int64_t>` | 512 threads；64 位地址/步长 | grid-stride tail |
| R-005 | 1 + int64 guard | FP32、其余同上 | 同上 | `<float,int64_t>` | 同上 | 同上 |
| R-006 | 2 + int64 guard | BF16、其余同上 | 同上 | `<bfloat16_t,int64_t>` | 同上 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| DCache 保留 | `UB>128 KiB` | `SetLocalMemorySize(UB-128 KiB)` 留出 DCache | 全部 | `GRAPH_FAILED` | `ForeachErfcTilingFunc` |
| Tensor 数上限 | `tensorNum<=256` | POD `cumulativeOffsets[256]` 固定 | 全部 | `GRAPH_FAILED` | `MAX_TENSOR_NUM_FOREACH_ERFC` |
| 每核最少元素 | `grid<=ceil(M/1024)` | 避免为少量元素启动过多 outer blocks | 全部 | 无另一 Route | `SINGLE_CORE_MIN_ELEMENTS` |
| 索引位宽 | `M<=INT32_MAX` | 32 位索引允许 1024 thread launch bound；大域防溢出改 64 位 | R-001–006 | i32→i64 | `Process/THREAD_NUM_VF` |
| dtype key | FP16→0、FP32→1、BF16→2 | 编译期 `schMode` 实例化 `T` | 全部 | 其他 dtype 在接口校验失败；未知 mode 无输出 | `GetTilingKeyByDtype/foreach_erfc` |
| 空域 | `M==0` | 避免 VF 与 TensorList 地址解引用 | 全部 | `Process` return | `NsForeachErfc::Process` |

### POD/ABI

Host struct 与 Device class 以相同字段顺序和原生标量类型声明；固定仓没有 `static_assert(sizeof/alignof/offsetof)`，也不包含目标编译器 ABI 定义，因此只能证明字段顺序、类型、位宽和数组长度一致，精确 offset、alignof 与 sizeof 均标为固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachErfcTilingData` | `needCoreNum` | `int32_t`/32 | 第 1 字段；精确 offset/alignof/sizeof 固定仓不可见 | core | `needCoreNum` 窄化后赋值 | 当前 body不读取；blockDim由 context单独消费 | ≥1；平台 core 数需可装入 int32 |
| 同上 | `tensorCount` | `int32_t`/32 | 第 2 字段；精确 offset 固定仓不可见 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | SIMT 线性查前缀循环上界 | 0–256 |
| 同上 | `totalElements` | `int64_t`/64 | 第 3 字段；精确 offset 固定仓不可见 | 元素 | shape size 累加 | `Process` 空域/索引 guard与 VF上界 | 累加溢出未显式检查 |
| 同上 | `perCoreElements` | `int64_t`/64 | 第 4 字段；精确 offset 固定仓不可见 | 元素/core | `ceil(totalElements/needCoreNum)` | 当前 body不读取 | 仅保留 ABI |
| 同上 | `cumulativeOffsets[256]` | `int64_t[256]` | 第 5 字段；精确 offset 固定仓不可见 | 元素前缀终点 | 循环逐项写累计值 | VF 找 tensorId/localIdx | `[tensorCount,256)` inactive且不读取 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001/R-004 | schMode 0 + M guard | `ForeachErfcTilingFunc` | `ForeachErfcTilingData` | `foreach_erfc<0>` | `Process<half>`→SIMT `<half,int32/int64>` | ascend950→arch35 entry |
| R-002/R-005 | schMode 1 + M guard | 同上 | 同上 | `foreach_erfc<1>` | `Process<float>`→SIMT `<float,int32/int64>` | 同上 |
| R-003/R-006 | schMode 2 + M guard | 同上 | 同上 | `foreach_erfc<2>` | `Process<bfloat16_t>`→SIMT `<bfloat16_t,int32/int64>` | 同上 |

## Kernel 实现

### 公共执行流程

templated outer entry解析 POD并按 `schMode` 编译期选择 dtype。`Process<T>`直接从 tiling GM读取 `M/tensorCount/cumulativeOffsets`；非空时根据 `INT32_MAX` 选择索引类型和 thread 数并调用 VF。每线程按 grid stride遍历 flat index，线性扫描前缀和定位 Tensor，再通过 TensorList GM 的“首个 uint64 字节偏移→指针表→第 tensorId 个地址”ABI取得输入输出地址，load一个元素、调用 dtype overload、store一个元素。没有跨线程共享数据、atomic或 barrier。

### Route R-001：FP16 + int32 索引 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_erfc_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | key 0 |
| Router | `op_kernel/arch35/foreach_erfc.cpp` | `foreach_erfc<schMode>` | `schMode==0` | `Process<half>` |
| Device | `foreach_erfc_simt.h` | `Process<half>` | `0<M<=INT32_MAX` | `<half,int32_t>`，1024 threads |

#### Device 数据流

##### 函数级调用链

```text
foreach_erfc<0> -> Process<half>
  -> asc_vf_call<OpForeachErfcSimtKernel<half,int32_t>>(dim3(1024),...)
    -> SimtGetTensorAddr<half> -> ErfcCompute::Calc(half) -> GM store
```

##### 源码等价伪码

```text
for flat=int32(blockIdx*blockDim+threadIdx); flat<M; flat+=blockDim*gridDim:
  scan t=0..tensorCount-1 until flat<cumulativeOffsets[t]
  prev = t>0 ? cumulativeOffsets[t-1] : 0
  local = flat-prev
  f = half_to_float(xPtr[t][local])
  yPtr[t][local] = float_to_half(erfcf(f))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| tiling GM | `tilingGM->cumulativeOffsets + t`；数值字节 offset 固定仓不可见 | `tensorCount`×`sizeof(int64_t)` active | int64 prefix | Host | thread scan | executor | read-only |
| input/output GM | `tensorPtr[t]+local` | 1 element/thread/iteration | FP16/ND | caller/thread | thread/caller | caller | direct scalar access |
| thread private | registers | flat/t/local/f | int32/int32/FP32 | thread | same thread | iteration | no shared alias |

##### 分层内存布局

N/A：该 SIMT body没有显式 AIV UB/TQue/TBuf/shared或 dynamic UB；实际替代路径是 GM scalar load/store与线程私有标量。Host设置的 `UB-128 KiB` local-memory配额内部细分固定仓不可见。

##### 字节账与复用顺序

```text
显式 AIV UB_total = 0
dynamic_ub = N/A（asc_vf_call 未传动态 UB 字节参数）
workspace_total = 0
thread-private lifetime = 单次 grid-stride iteration
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread private scalars | 当前 thread | 当前 thread | 下一 flat iteration | 程序顺序 | 每个 flat index唯一写输出 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dispatch | grid-stride loop | launch geometry | each thread | `[0,M)` | `flat<M`保护尾部 |
| compute | `__half2float→erfcf→__float2half` | GM load | GM store | 单元素 | FP32中间；转换舍入细节由SIMT intrinsic定义 |
| synchronization | N/A | N/A | N/A | 无共享写冲突 | 每个 flat只由一个线程访问 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_erfc_tiling_arch35.cpp` | `ForeachErfcTilingFunc` | blockDim/key/prefix |
| Kernel entry/router | `op_kernel/arch35/foreach_erfc.cpp` | `foreach_erfc<0>` | half dispatch |
| Device body/helper | `op_kernel/arch35/foreach_erfc_simt.h` | `<half,int32_t>`、`ErfcCompute::Calc(half)` | 1024 threads、FP32 erfcf |

### Route R-002：FP32 + int32 索引 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_erfc_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT` | key 1 |
| Router | `arch35/foreach_erfc.cpp` | `schMode==1` | true | `Process<float>` |
| Device | `foreach_erfc_simt.h` | `Process<float>` | `0<M<=INT32_MAX` | `<float,int32_t>`，1024 threads |

#### Device 数据流

##### 函数级调用链

```text
foreach_erfc<1> -> Process<float> -> VF<float,int32_t>
  -> SimtGetTensorAddr<float> -> erfcf(float) -> GM store
```

##### 源码等价伪码

```text
与 R-001 相同的 int32 grid-stride和前缀扫描；计算改为 y=erfcf(x)，无显式 cast。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| prefix GM | `tilingGM->cumulativeOffsets+t`；数值字节 offset 固定仓不可见 | active前缀 | int64 | Host | thread | executor | read-only |
| input/output GM | `[t][flat-prev]` | 4 B each | FP32/ND | caller/thread | thread/caller | caller | direct scalar |
| private | registers | flat/t/local/value | int32/FP32 | thread | thread | iteration | 无共享 |

##### 分层内存布局

N/A：同 R-001，无显式 AIV UB/shared/dynamic UB。

##### 字节账与复用顺序

```text
显式 UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread registers | thread | thread | 下一 iteration | 程序顺序 | flat 唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | `erfcf` | FP32 load | FP32 store | 单元素 | FP32；`flat<M` tail guard |
| sync | N/A | N/A | N/A | 无共享状态 | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_erfc_tiling_arch35.cpp` | `GetTilingKeyByDtype` | FP32 key 1 |
| Kernel entry/router | `arch35/foreach_erfc.cpp` | `foreach_erfc<1>` | float dispatch |
| Device body/helper | `foreach_erfc_simt.h` | `<float,int32_t>` | direct erfcf |

### Route R-003：BF16 + int32 索引 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_erfc_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_BF16` | key 2 |
| Router | `arch35/foreach_erfc.cpp` | `schMode==2` | true | `Process<bfloat16_t>` |
| Device | `foreach_erfc_simt.h` | `Process<bfloat16_t>` | `0<M<=INT32_MAX` | `<bfloat16_t,int32_t>`，1024 threads |

#### Device 数据流

##### 函数级调用链

```text
foreach_erfc<2> -> Process<bfloat16_t> -> VF<bfloat16_t,int32_t>
  -> BF16-to-float -> erfcf -> float-to-BF16 -> GM
```

##### 源码等价伪码

```text
复用 int32 grid-stride与前缀扫描；f=__bfloat162float(x); y=__float2bfloat16(erfcf(f))。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| prefix GM | `tilingGM->cumulativeOffsets+t`；数值字节 offset 固定仓不可见 | active前缀 | int64 | Host | thread | executor | read-only |
| input/output GM | `[t][local]` | 2 B each | BF16/ND | caller/thread | thread/caller | caller | direct scalar |
| private | registers | FP32 `f/result` | FP32 | conversion/erfcf | conversion | iteration | 每线程私有 |

##### 分层内存布局

N/A：无显式 AIV UB/shared/dynamic UB，升精度值在线程私有标量中。

##### 字节账与复用顺序

```text
显式 UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 private scalar | BF16 conversion/erfcf | erfcf/BF16 conversion | 下一 flat | 程序顺序 | thread private |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | BF16→FP32→`erfcf`→BF16 | GM load | GM store | 单元素 | FP32中间；`flat<M` tail |
| sync | N/A | N/A | N/A | 无冲突域 | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_erfc_tiling_arch35.cpp` | key 2 producer | BF16 selector |
| Kernel entry/router | `arch35/foreach_erfc.cpp` | `foreach_erfc<2>` | BF16 dispatch |
| Device body/helper | `foreach_erfc_simt.h` | `ErfcCompute::Calc(bfloat16_t)` | FP32中间 |

### Route R-004：FP16 + int64 索引 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Router | `arch35/foreach_erfc.cpp` | `schMode==0` | FP16 | `Process<half>` |
| Device | `foreach_erfc_simt.h` | `Process<half>` | `M>INT32_MAX` | `<half,int64_t>`，512 threads |

#### Device 数据流

##### 函数级调用链

```text
foreach_erfc<0> -> Process<half> -> VF<half,int64_t>(dim3(512))
  -> prefix scan -> half/float erfcf conversion -> GM
```

##### 源码等价伪码

```text
与 R-001 相同，但 flat/prev/local/stride 全为 int64，threadsPerBlock=512。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM/prefix | 与R-001相同 | M>INT32_MAX | FP16/int64 | Host/caller | SIMT | executor/caller | direct scalar |
| private | 64位flat/prev/local | 每thread一组 | int64 | thread | thread | iteration | 无共享 |

##### 分层内存布局

N/A：无显式 UB/shared；64位线程私有索引替代。

##### 字节账与复用顺序

```text
显式 UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| private scalars | thread | thread | 下一 flat | 程序顺序 | flat唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| dispatch | int64 grid-stride | launch | thread | 大于2^31-1元素域 | `flat<M` |
| compute | half→float→erfcf→half | thread | GM | 单元素 | 同R-001 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_erfc_tiling_arch35.cpp` | `totalElements` | int64累计 |
| Kernel entry/router | `arch35/foreach_erfc.cpp` | `foreach_erfc<0>` | FP16 |
| Device body/helper | `foreach_erfc_simt.h` | `<half,int64_t>` | 512 threads与64位步长 |

### Route R-005：FP32 + int64 索引 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Router | `arch35/foreach_erfc.cpp` | `schMode==1` | FP32 | `Process<float>` |
| Device | `foreach_erfc_simt.h` | `Process<float>` | `M>INT32_MAX` | `<float,int64_t>`，512 threads |

#### Device 数据流

##### 函数级调用链

```text
foreach_erfc<1> -> Process<float> -> VF<float,int64_t> -> erfcf -> GM
```

##### 源码等价伪码

```text
int64 grid-stride + cumulativeOffsets线性扫描 + FP32 erfcf直接写回。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `[t][flat-prev]` | 4 B/元素 | FP32/ND | caller/thread | thread/caller | caller | scalar |
| private | int64索引+FP32值 | thread local | scalar | thread | thread | iteration | 无共享 |

##### 分层内存布局

N/A：无显式 AIV UB/shared/dynamic UB。

##### 字节账与复用顺序

```text
显式 UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread private | thread | thread | 下一iteration | 程序顺序 | flat唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | FP32 `erfcf` | GM | GM | 单元素 | int64 `flat<M` guard |
| sync | N/A | N/A | N/A | 无共享 | 无 barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_erfc_tiling_arch35.cpp` | `totalElements` | int64域 |
| Kernel entry/router | `arch35/foreach_erfc.cpp` | `foreach_erfc<1>` | FP32 |
| Device body/helper | `foreach_erfc_simt.h` | `<float,int64_t>` | 512-thread direct erfcf |

### Route R-006：BF16 + int64 索引 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Router | `arch35/foreach_erfc.cpp` | `schMode==2` | BF16 | `Process<bfloat16_t>` |
| Device | `foreach_erfc_simt.h` | `Process<bfloat16_t>` | `M>INT32_MAX` | `<bfloat16_t,int64_t>`，512 threads |

#### Device 数据流

##### 函数级调用链

```text
foreach_erfc<2> -> Process<bfloat16_t> -> VF<bfloat16_t,int64_t>
  -> BF16/FP32 conversions + erfcf -> GM
```

##### 源码等价伪码

```text
int64 grid-stride + prefix scan；每元素 BF16→FP32→erfcf→BF16。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM | `[t][flat-prev]` | 2 B/元素 | BF16/ND | caller/thread | thread/caller | caller | scalar |
| private | int64索引+FP32中间 | thread local | scalar | thread | thread | iteration | 无共享 |

##### 分层内存布局

N/A：无显式 AIV UB/shared/dynamic UB。

##### 字节账与复用顺序

```text
显式 UB_total=0; dynamic_ub=N/A; workspace_total=0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread private | conversion/erfcf | same thread | 下一flat | 程序顺序 | flat唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| compute | BF16→FP32→`erfcf`→BF16 | GM | GM | 单元素 | int64 tail guard |
| sync | N/A | N/A | N/A | 无共享 | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_erfc_tiling_arch35.cpp` | key/prefix producer | BF16大域 |
| Kernel entry/router | `arch35/foreach_erfc.cpp` | `foreach_erfc<2>` | BF16 |
| Device body/helper | `foreach_erfc_simt.h` | `<bfloat16_t,int64_t>` | 512 threads、FP32中间 |

## 算子特定优化

### Trick T-001：按总元素规模切换 32/64 位索引与 thread 数

#### 触发与源码依据

- 适用 Route：R-001–R-006
- 触发条件：`M<=INT32_MAX` 选 int32，否则选 int64
- Golden 来源：[`foreach_erfc_simt.h:108-123`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_simt.h#L108-L123)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (totalElements <= static_cast<int64_t>(INT32_MAX)) {
    using IDX_T = int32_t;
    asc_vf_call<OpForeachErfcSimtKernel<T, IDX_T>>(
        dim3(THREAD_NUM_VF<IDX_T>),
        static_cast<IDX_T>(totalElements), tensorCount,
        cumOffsets, x, y);
} else {
    using IDX_T = int64_t;
```

#### 机制与设计原因

常见规模使用 int32 索引，模板常量把 launch bound 提至 1024 threads；超过 `INT32_MAX` 后才承担 int64 索引并降为 512 threads，避免扁平索引、步长与 local offset 溢出。选择发生在 Device wrapper，Host POD始终保留 int64 总量与前缀。

#### 迁移边界

- 适用：扁平元素域可由总元素数判断索引安全性的 SIMT unary 算子。
- 前提：所有地址差与累计前缀在 int64 内，32 位分支只在 `M<=INT32_MAX` 使用。
- 失效：单个地址偏移还受字节乘法溢出、或 primitive要求固定 thread 数时需重新证明。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓提供的真实规避边界是：大于 `INT32_MAX` 仍强制窄化会破坏 `flatIdx`/步长覆盖，因此官方切到 int64；固定仓未给出错误实现代码。

### Trick T-002：用累计前缀统一调度整个 TensorList

#### 触发与源码依据

- 适用 Route：R-001–R-006
- 触发条件：`tensorCount<=256` 且 `M>0`
- Golden 来源：[`foreach_erfc_simt.h:74-94`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_simt.h#L74-L94)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (IDX_T flatIdx = static_cast<IDX_T>(
        blockIdx.x * blockDim.x + threadIdx.x);
     flatIdx < totalElements;
     flatIdx += static_cast<IDX_T>(
        blockDim.x * gridDim.x)) {
    int32_t tensorId = tensorCount - 1;
```

#### 机制与设计原因

Host只发射一个 Kernel，把所有 Tensor拼为一个逻辑元素域；thread通过前缀和还原Tensor地址。这样小Tensor共享同一grid，不需要按列表项创建多Kernel或中间workspace。代价是每个元素最多线性扫描256个前缀，源码没有二分或缓存路线。

#### 迁移边界

- 适用：TensorList 各项同 dtype、同 unary语义、输出一一对应。
- 前提：前缀和有序且不溢出，TensorList指针表ABI一致。
- 失效：不同Tensor使用不同属性、需要跨项通信或列表规模超过256时不能复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001/002/003 int32 | R-004/005/006 int64 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | `0<M<=INT32_MAX` | `M>INT32_MAX` | 先空域 return，再按总元素阈值 |
| 模板与 dtype | half/float/bfloat16_t × int32 | 同三 dtype × int64 | schMode决定T，规模决定IDX_T |
| 循环与分核 | 1024 threads，int32 grid-stride | 512 threads，int64 grid-stride | `THREAD_NUM_VF<IDX_T>` |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程私有标量；无显式UB/shared | 64位线程私有索引；无显式UB/shared | SIMT direct GM路径 |
| 内存总占用与复用 | workspace 0 | workspace 0 | 仅POD prefix与GM Tensor |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar load/store、erfcf | 同左 | 无AIV Vector/NDDMA/通信 |
| 同步、精度与 tail | 无barrier；flat<M；half/bf16经FP32 | 同左 | 每flat唯一thread |
| fallback/失败边界 | 超阈值切int64 | 无更宽fallback | totalElements int64累计未检查溢出 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持 |
| Host/Launcher | arch22公共foreach Tiling | arch22公共foreach Tiling | arch35独立前缀和/1024元素分核 |
| Kernel/资源 | AIV_VECTOR双queue | 同A2 | SIMT direct GM；无显式queue/workspace |

## 限制与失败边界

- TensorList 上限为256；`totalElements`使用int64累加但没有显式加法溢出检查。
- arch35 Tiling只读取第0项dtype，列表内dtype一致性依赖公开ACLNN/OpDef框架校验；生成桥固定仓不可见。
- `needCoreNum/perCoreElements`写入POD但Device body不读取；不得误用它们替代真实`gridDim.x`和grid-stride公式。
- 每个元素线性扫描最多256个 cumulative offsets；固定仓无二分或共享缓存fallback。
- 不支持的schMode没有Device fallback；空域固定发射1个outer block后在`Process`返回。
- `erfcf`、half/BF16 conversion intrinsic的内部近似、舍入细节和Host设置local memory的内部消耗固定仓不可见。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_erfc/docs/aclnnForeachErfc.md` | 两段式ACLNN符号 | 950平台、原型、dtype/shape/错误 |
| 入口声明/约束 | `foreach/foreach_erfc/README.md` | 产品支持表 | 950公开支持 |
| 入口实现 | `foreach/foreach_erfc/op_host/foreach_erfc_def.cpp` | `ops::ForeachErfc` | 950 OpDef/dtype/ND/AutoContiguous |
| Host/Tiling/Launcher | `foreach/foreach_erfc/op_host/arch35/foreach_erfc_tiling_arch35.h` | `ForeachErfcTilingData` | Host POD声明 |
| Host/Tiling/Launcher | `foreach/foreach_erfc/op_host/arch35/foreach_erfc_tiling_arch35.cpp` | `ForeachErfcTilingFunc` | 平台、前缀、分核、key、workspace |
| Kernel entry/body | `foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_tiling_data.h` | `ForeachErfcTilingData` | Device POD声明 |
| Kernel entry/body | `foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_tiling_key.h` | `ForeachErfc` template args | schMode 0/1/2构建选择 |
| Kernel entry/body | `foreach/foreach_erfc/op_kernel/arch35/foreach_erfc.cpp` | `foreach_erfc<schMode>` | dtype router |
| Kernel entry/body | `foreach/foreach_erfc/op_kernel/arch35/foreach_erfc_simt.h` | `Process/OpForeachErfcSimtKernel/ErfcCompute` | SIMT线程、地址、精度、tail |
| 构建接线 | `foreach/foreach_erfc/op_host/CMakeLists.txt` | `ascend950→arch35` | 950 Host选择 |
| 构建接线 | `foreach/foreach_erfc/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_erfc.cpp` | 950 Kernel选择 |
| 构建接线 | `foreach/foreach_erfc/op_host/config/ascend950/foreach_erfc_binary.json` | `ForeachErfc` op_list | 三dtype binary接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | Host/Device同序同类型五字段、256项前缀 | 字段顺序与位宽、inactive项不读；不得假定固定仓未证明的 offset/sizeof | 两份`ForeachErfcTilingData` |
| 调用顺序 | outer entry→Process→一个SIMT VF | 空域先return；dtype再索引分支 | `foreach_erfc/Process` |
| workspace/内存 | workspace 0、direct GM | UB必须大于128KiB；local memory保留公式 | `ForeachErfcTilingFunc` |
| 同步 | 无共享状态，不需barrier/atomic | 每个flat index只能唯一映射一个thread | `OpForeachErfcSimtKernel` |
| 精度与尾块 | FP32 erfcf；half/BF16经FP32 | `INT32_MAX`位宽边界、`flat<M`、dtype转换顺序 | `ErfcCompute/Process` |
| 失败边界 | 平台/UB/tensor count失败阻断launch | 不得扩大256或为未知schMode自造fallback | `ForeachErfcTilingFunc/foreach_erfc` |