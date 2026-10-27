---
type: Operator
title: "ForeachAcos 950 ACLNN 算子知识"
description: "说明 aclnnForeachAcos 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/docs/aclnnForeachAcos.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_host/foreach_acos_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_host/arch35/foreach_acos_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_host/arch35/foreach_acos_tiling_arch35.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_kernel/arch35/foreach_acos.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_kernel/arch35/foreach_acos_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_kernel/arch35/foreach_acos_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_kernel/arch35/foreach_acos_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_host/config/ascend950/foreach_acos_binary.json
    role: implementation
created_at: "2026-08-25T15:28:42Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAcos 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAcosGetWorkspaceSize` + `aclnnForeachAcos`，公开 ACLNN，两段式且无编号版本 |
| 计算语义 | 对列表中每个张量逐元素计算 `y_i[j] = acos(x_i[j])` |
| 输入输出 | 动态 TensorList；各项为 ND；FP16/FP32/BF16；对应输入输出 dtype 一致，输出容量不得小于输入 |
| 执行类型 | 单节点 `SIMT`；没有可达 Cube/AIC/MIX_CV 节点 |
| Kernel 路线 | R-001 FP16、R-002 FP32、R-003 BF16 |
| 核心资源 | TensorList descriptor 与数据均在 GM；POD 中保存 257 个 `int64_t` 前缀和；每核 512 SIMT 线程；workspace=0；无显式 AIV UB/TQue/TBuf |
| 来源闭包 | `source_closed`：公开 API 文档→`ForeachAcos` OpDef→arch35 Tiling/POD→三 dtype 模板 key→SIMT entry/body→ascend950 构建与 binary config 已闭合；ACLNN inner bridge 为仓内未落盘的生成边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

输入为列表 `x=[x_0,...,x_(n-1)]`，输出为同长度列表 `y`；对每个张量有效元素 `j` 执行 `y_i[j]=cos^{-1}(x_i[j])`。Kernel 按所有张量的拼接平坦索引分工，再由前缀和恢复 `(tensorIdx, localIdx)`。源码没有对定义域 `[-1,1]` 做显式裁剪或报错，数值行为交由 `acosf`。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList，各 Tensor 为 ND，0–8 维 | FP16/FP32/BF16，列表内一致 | 可含空 Tensor；输入可非连续，OpDef 的 `AutoContiguous()` 是框架前处理边界 |
| `out` | TensorList，各 Tensor 为 ND，0–8 维 | 与 `x` 一致 | 列表内一致；输出不支持非连续；同位置输出 shape size 不小于输入 |
| `workspaceSize` | 标量 | `uint64_t*` | 第一段接口输出 executor 总 workspace；950 ForeachAcos 节点 Tiling 置 0，生成式连续化前处理是否追加空间在固定仓不可见 |
| `executor` | 执行器 | `aclOpExecutor**` | 第一段接口输出并拥有已编排计算流程；第二段消费 |
| `stream` | Stream | `aclrtStream` | 第二段接口指定异步执行顺序 |

### 入口与版本

固定仓公开且仅公开无编号的 `aclnnForeachAcosGetWorkspaceSize`/`aclnnForeachAcos`，因此它是 950 的实际最新版本。文档明确列出 Ascend 950PR/950DT 支持；`ForeachAcos` OpDef 的 `AddConfig("ascend950")`、Host `arch35` 选择、ascend950 binary config 与 Kernel CMake 同时闭合。生成的 ACLNN C++ inner bridge 未签入固定仓；其参数校验与 executor 组装以公开文档及 `ACLNNTYPE aclnn` 生成接线为框架边界，不外推不可见实现。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口检查 `x/out` 非空、dtype 属于 FP16/FP32/BF16 且彼此一致、列表内 dtype 一致、维数不超过 8、输出容量约束。OpDef 为输入和输出声明 `AutoContiguous()`；文档只允许输入非连续而禁止输出非连续。随后 arch35 `ForeachAcosTilingFunc` 获取平台 AIV 核数和 UB，要求 `UB > 128 KiB`，读取动态输入数量（最多 256），遍历 storage shape 计算总元素与前缀和，并从首 Tensor 获取 dtype。

### 参数、Workspace 与生命周期

令 `N=tensorNum`、`E=sum(numel_i)`、`Cmax=GetCoreNumAiv()`：`needCoreNum = E==0 ? 1 : min(Cmax, ceil(E/1024))`；`perCoreElements = E==0 ? 0 : align_up(ceil(E/needCoreNum),32)`，单位均为元素。Tiling 将 `blockDim=needCoreNum`、local-memory size 设为 `UB-128 KiB`、ForeachAcos 节点 workspace[0] 设为 0；ACLNN 总 workspace 仍可能包含生成式连续化前处理。POD、TensorList descriptor、输入输出数据在 Kernel 完成前必须有效；executor 与用户 stream 由 ACLNN 框架持有/消费。

### 启动图与失败传播

单节点图：ACLNN executor 在给定 stream 上启动 `foreach_acos<schMode>(xList,yList,workspace,tiling)`，grid/blockDim 为 `needCoreNum`，每个 AIV block 内由 `Process<T>` 发起 `asc_vf_call`，SIMT block 使用固定 512 线程。无跨 Kernel 边；workspace slice 为 `[0,0)`。平台信息、UB、POD 指针、动态输入 shape/desc、Tensor 数上限或 `SetLocalMemorySize` 任一失败都会令 Tiling 返回 `GRAPH_FAILED`，阻断 launch；SIMT `E==0` 直接返回。

## Route 与参数设计

### 分核与资源计算

Host 先按列表顺序构造 `prefixSums[0]=0`、`prefixSums[i+1]=prefixSums[i]+numel_i`。每核负责 `[coreIdx*perCoreElements, min((coreIdx+1)*perCoreElements,E))`；512 线程以 `threadIdx.x` 为起点、步长 `blockDim.x=512` 遍历。`perCoreElements` 的 32 元素对齐可能使最后若干核的区间在 `E` 处截断，但 `flatIdx<coreEnd` 防止越界。实现不分配 dynamic UB，Host 设置的 local memory 仅是 launch 资源参数。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 | `schMode=0` / `Process<half>` | 首个输入 dtype 为 `DT_FLOAT16` | 检查 FP32 |
| 2 | R-002 FP32 | `schMode=1` / `Process<float>` | 首个输入 dtype 为 `DT_FLOAT` | 检查 BF16 |
| 3 | R-003 BF16 | `schMode=2` / `Process<bfloat16_t>` | 首个输入 dtype 为 `DT_BF16` | 其它 dtype 由公开接口/OpDef 拒绝；Tiling helper 默认 1 只是不应从受支持入口命中的防御值 |

```text
dtype == FP16 -> key 0 -> foreach_acos<0> -> Process<half>
else dtype == FP32 -> key 1 -> foreach_acos<1> -> Process<float>
else dtype == BF16 -> key 2 -> foreach_acos<2> -> Process<bfloat16_t>
else -> 公开入口不支持
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 0 | 950，列表内 FP16，N≤256，约束校验通过 | 单 SIMT 节点 | `ForeachAcosSimtKernel<half>` + `AcosCompute<half>` | GM 直接 load/store；每线程私有标量；无 barrier/atomic | FP32 升精度；`flatIdx<coreEnd` 截尾 |
| R-002 FP32 | 1 | 950，列表内 FP32，N≤256，约束校验通过 | 单 SIMT 节点 | `ForeachAcosSimtKernel<float>` + 主模板 | GM 直接 load/store；无显式 UB | 直接 `acosf`；同一尾保护 |
| R-003 BF16 | 2 | 950，列表内 BF16，N≤256，约束校验通过 | 单 SIMT 节点 | `ForeachAcosSimtKernel<bfloat16_t>` + BF16 特化 | GM 直接 load/store；每线程 float 中间量 | FP32 升精度；同一尾保护 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList 上限 | `static_cast<int32_t>(tensorNum) <= 256` | POD 仅有 257 项前缀和 | 全部 | `GRAPH_FAILED` | `foreach_acos_tiling_arch35.cpp::ForeachAcosTilingFunc` |
| 平台内存 | `ubSize > 128*1024` | 128 KiB 被当作 DCache 保留后传给 local-memory size | 全部 | `GRAPH_FAILED` | `ForeachAcosTilingFunc` |
| 每核最低工作量 | `ceil(E/1024)` 再与 `Cmax` 取小 | 避免为少量元素启用过多 block | 全部 | 非 fallback，只改变 grid | `ForeachAcosTilingFunc` |
| 32 元素对齐 | `perCoreElements=align_up(ceil(E/C),32)` | 连续区间边界规整，末端由 `min` 截止 | 全部 | N/A | `ForeachAcosTilingFunc`、`ForeachAcosSimtKernel` |
| 空域 | `E==0` | 避免前缀和二分和 descriptor 解引用 | 全部 | Kernel 直接返回 | `Process`、`ForeachAcosSimtKernel` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAcosTilingData` | `needCoreNum` | `int32_t`/32 | offset 0 | core/block | `needCoreNum` | `Process`→VF 参数 | ≥1，≤平台核数 |
| 同上 | `tensorNum` | `int32_t`/32 | offset 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | 二分上界 `n` | 0..256 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | 元素 | `sum(GetShapeSize())` | 空域和 coreEnd | 非负；加法溢出未显式检查 |
| 同上 | `perCoreElements` | `int64_t`/64 | offset 16 | 元素/core | 32 对齐公式 | coreStart/coreEnd | E=0 时 0 |
| 同上 | `prefixSums` | `int64_t[257]` | offset 24；2056 B | 累计元素 | 循环写 `[0..N]` | `PrefixSumUpperBound` | `[N+1..256]` inactive，不读取 |

结构为自然 8 字节对齐的普通 class，Host 与 Device 字段顺序一致；由字段可得 `sizeof=2080`、`alignof=8`。仓内没有 `static_assert`，因此复用时仍须保持该布局。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | key 0 | `ForeachAcosTilingFunc` | `ForeachAcosTilingData` | `foreach_acos<0>` | `Process<half>`→`ForeachAcosSimtKernel<half>` | Kernel CMake ascend950 + ascend950 binary JSON |
| R-002 FP32 | key 1 | 同上 | 同上 | `foreach_acos<1>` | `Process<float>`→`ForeachAcosSimtKernel<float>` | 同上 |
| R-003 BF16 | key 2 | 同上 | 同上 | `foreach_acos<2>` | `Process<bfloat16_t>`→SIMT BF16 特化 | 同上 |

## Kernel 实现

### 公共执行流程

三条 Route 共用单 Kernel graph。entry 解析 POD，按编译期 `schMode` 选择 `Process<T>`；`Process` 读五组参数并在非空域调用 512-thread VF。每线程生成全局平坦索引，二分 `prefixSums` 定位 Tensor，按 TensorList descriptor 的首字偏移找到指针表，直接 GM load→`acosf`→GM store。不同线程写唯一 `flatIdx`，故无 atomic/barrier；workspace 参数不读取。

### Route R-001：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_acos_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | key 0 |
| Router | `op_kernel/arch35/foreach_acos.cpp` | `foreach_acos<schMode>` | `schMode==KEY_FP16` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_acos<0>(xList,yList,workspace,tiling)
  -> Process<half>
    -> asc_vf_call<ForeachAcosSimtKernel<half>>(threads=512,...)
      -> PrefixSumUpperBound
      -> SimtGetTensorAddr<half>(xList) -> AcosCompute<half>::Compute -> SimtGetTensorAddr<half>(yList)
```

##### 源码等价伪码

```text
if E==0: return
start = blockIdx.x * perCoreElements
end = min(start + perCoreElements, E)
for flat = start + threadIdx.x; flat < end; flat += 512:
  t = upper_bound(prefixSums[0..N], flat) - 1
  local = flat - prefixSums[t]
  y[t][local] = float_to_half(acosf(half_to_float(x[t][local])))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM prefix | POD+24 | `(N+1)*8` 有效，257*8 分配 | int64 连续 | Host | 每线程二分 | POD/本次 launch | 8 B 对齐 |
| GM input | descriptor→ptr[t]+local | 1 元素/迭代 | FP16 ND storage | 用户/contiguous 边界 | SIMT thread | 调用方至 stream 完成 | 无共享复用 |
| GM output | descriptor→ptr[t]+local | 1 元素/迭代 | FP16 ND storage | SIMT thread | 后续 stream 节点 | 调用方 | 每 flat 唯一写者 |
| thread scalar | 寄存器实现固定仓不可见 | `val/fval/fresult/result` 生命周期 | half/float | 当前线程 | 当前线程 | 单迭代 | 无跨线程共享 |

##### 分层内存布局

N/A：该 SIMT 路线不创建 AIV UB/TQue/TBuf；数据经 GM 标量 load/store，每线程中间标量位于编译器管理的寄存器，具体寄存器字节固定仓不可见。

##### 字节账与复用顺序

```text
UB_total = 0（无显式 AIV UB 分配）
dynamic_ub = 0（asc_vf_call 未传动态 UB）
workspace_total = 0
POD allocation = 2080 B，其中 prefixSums 有效区=(N+1)*8 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程标量 | 当前循环迭代 | `AcosCompute`/GM store | 下一 flat 迭代 | 单线程程序序 | 不跨线程暴露 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→计算→GM | 单线程程序序 | GM load | `acosf` 与 store | 当前元素 | half→float→`acosf`→half；`flat<end` 保护尾部 |
| 跨线程 | N/A | 每线程独占 flat | N/A | 输出元素 | 无 barrier/atomic；唯一写者 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_acos_tiling_arch35.cpp` | `ForeachAcosTilingFunc` | grid、前缀和、每核区间、workspace=0 |
| Kernel entry/router | `op_kernel/arch35/foreach_acos.cpp` | `foreach_acos<0>` | key 0→half |
| Device body/helper | `op_kernel/arch35/foreach_acos_simt.h` | `ForeachAcosSimtKernel<half>`、`AcosCompute<half>` | 二分映射与 FP32 升精度 |

### Route R-002：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_acos_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT` | key 1 |
| Router | `op_kernel/arch35/foreach_acos.cpp` | `foreach_acos<schMode>` | `schMode==KEY_FP32` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_acos<1> -> Process<float> -> ForeachAcosSimtKernel<float>
  -> PrefixSumUpperBound -> SimtGetTensorAddr<float> -> acosf -> GM store
```

##### 源码等价伪码

```text
for each assigned flat index guarded by flat<min(coreStart+perCoreElements,E):
  (tensor,local)=prefix_binary_search(flat)
  y[tensor][local]=acosf(x[tensor][local])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | descriptor pointer + local | 各 1 FP32/迭代 | FP32 ND | GM/thread | thread/GM | launch | 唯一 flat 写入 |
| POD prefix | POD+24 | `(N+1)*8` 有效 | int64 | Host | binary search | launch | 同 R-001 |

##### 分层内存布局

N/A：无显式 AIV UB 或 dynamic UB，只有 GM 与线程私有标量。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=0; POD=2080 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程 FP32 标量 | GM load | `acosf`/store | 下一循环 | 程序序 | thread 独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 元素计算 | `acosf` | FP32 load | FP32 store | 当前元素 | 无 cast；flat/end 截尾 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_acos_tiling_arch35.cpp` | `ForeachAcosTilingFunc` | key 1 参数生产 |
| Kernel entry/router | `op_kernel/arch35/foreach_acos.cpp` | `foreach_acos<1>` | FP32 route |
| Device body/helper | `op_kernel/arch35/foreach_acos_simt.h` | `AcosCompute<float>` | 直接 `acosf` |

### Route R-003：BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `op_host/arch35/foreach_acos_tiling_arch35.cpp` | `GetTilingKeyByDtype` | `DT_BF16` | key 2 |
| Router | `op_kernel/arch35/foreach_acos.cpp` | `foreach_acos<schMode>` | `schMode==KEY_BF16` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_acos<2> -> Process<bfloat16_t> -> ForeachAcosSimtKernel<bfloat16_t>
  -> PrefixSumUpperBound -> BF16 GM load -> __bfloat162float -> acosf -> __float2bfloat16 -> GM store
```

##### 源码等价伪码

```text
for assigned flat index with flat<coreEnd:
  locate tensor/local by prefix binary search
  f=float(x_bf16[tensor][local]); r=acosf(f); y_bf16[tensor][local]=bf16(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | descriptor pointer + local | 各 1 BF16/迭代 | BF16 ND | GM/thread | thread/GM | launch | 唯一 flat 写入 |
| thread float | 编译器分配 | `fval/fresult` | FP32 | cast/acosf | BF16 cast | 单迭代 | 寄存器字节不可见 |

##### 分层内存布局

N/A：无显式 AIV UB 或 dynamic UB；BF16 升精度值是线程私有标量。

##### 字节账与复用顺序

```text
UB_total=0; dynamic_ub=0; workspace_total=0; POD=2080 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程 float 标量 | BF16→float | `acosf`/float→BF16 | 下一循环 | 程序序 | thread 独占 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 元素计算 | 显式 cast + `acosf` | BF16 load | BF16 store | 当前元素 | BF16→FP32→BF16；转换舍入细节由 intrinsic，固定仓未展开 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_acos_tiling_arch35.cpp` | `ForeachAcosTilingFunc` | key 2 参数生产 |
| Kernel entry/router | `op_kernel/arch35/foreach_acos.cpp` | `foreach_acos<2>` | BF16 route |
| Device body/helper | `op_kernel/arch35/foreach_acos_simt.h` | `AcosCompute<bfloat16_t>` | FP32 升精度 |

## 算子特定优化

### Trick T-001：连续平坦分核配合前缀和二分定位

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003
- 触发条件：`E>0` 的所有 950 路线
- Golden 来源：[`foreach_acos_simt.h:120-145`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_kernel/arch35/foreach_acos_simt.h#L120-L145)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t coreStart = static_cast<int64_t>(coreIdx) * perCoreElements;
int64_t coreEnd = (coreStart + perCoreElements > totalElements)
                      ? totalElements
                      : (coreStart + perCoreElements);

for (int64_t flatIdx = coreStart + static_cast<int64_t>(threadIdx.x);
     flatIdx < coreEnd;
     flatIdx += static_cast<int64_t>(blockDim.x))
{
    int32_t tensorIdx = PrefixSumUpperBound(prefixSums, tensorNum, flatIdx);
    int64_t localIdx = flatIdx - prefixSums[tensorIdx];
```

#### 机制与设计原因

Host 将异形 TensorList 统一为长度 `E` 的平坦域，每核得到连续区间，线程再以 512 为步长。元素到 Tensor 的转换通过前缀和 O(log N) 二分完成，不需要为每个 Tensor 单独 launch，也不让每个元素线性扫描列表；这改变的是调度和索引计算，而非数学结果。

#### 迁移边界

- 适用：元素独立、TensorList 项地址可从 descriptor 读取的逐元素算子。
- 前提：N≤256、前缀和单调且不溢出、每个 flat 索引只有一个写者。
- 失效：需要跨元素/跨 Tensor 归约、广播或 alias 协议时不能直接迁移；固定实现无其它 Route fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若删除 `flatIdx < coreEnd` 或把二分条件从 `prefixSums[mid] <= target` 改为严格小于，会分别导致尾部越界或边界元素映射到错误 Tensor。固定仓提供了正确实现，未提供独立错误版本。

### Trick T-002：低精度输入显式提升到 FP32 调用 acosf

#### 触发与源码依据

- 适用 Route：R-001、R-003
- 触发条件：模板 T 为 `half` 或 `bfloat16_t`
- Golden 来源：[`foreach_acos_simt.h:50-69`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_acos/op_kernel/arch35/foreach_acos_simt.h#L50-L69)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
float fval = __half2float(val);
float fresult = acosf(fval);
return __float2half(fresult);
```

#### 机制与设计原因

低精度路线不直接在 half/BF16 上求反余弦，而是显式转 FP32、调用 `acosf` 后降回原 dtype。这样固定了计算精度顺序，并复用同一 FP32 transcendent 实现；BF16 特化使用对应的 `__bfloat162float/__float2bfloat16`。

#### 迁移边界

- 适用：输出 dtype 必须保持低精度、但超越函数以 FP32 计算的逐元素路线。
- 前提：目标 SIMT 支持相应转换 intrinsic 与 `acosf`。
- 失效：要求其它舍入模式或更高精度结果时需改实现；本算子无其它 950 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 0 | key 1 | key 2 | dtype 唯一决定 |
| 模板与 dtype | `half` 特化 | `float` 主模板 | BF16 特化 | 保持 API dtype |
| 循环与分核 | 连续平坦域、512 threads | 同左 | 同左 | 共用 Tiling/SIMT body |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程 half+float 标量；dynUB=0 | 线程 float 标量；dynUB=0 | 线程 BF16+float 标量；dynUB=0 | 无 AIV queue |
| 内存总占用与复用 | workspace 0；POD 2080 B | 同左 | 同左 | 前缀和内嵌 POD |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM 标量 + cast + acosf | GM 标量 + acosf | GM 标量 + cast + acosf | dtype 精度路径不同 |
| 同步、精度与 tail | 无 barrier；FP32 升精度 | 无 cast | 无 barrier；FP32 升精度 | 每 flat 唯一写者 |
| fallback/失败边界 | 无 Device fallback | 无 | 无 | Host 先拒绝不支持条件 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；见无后缀卡 | 支持；见无后缀卡 | 支持 |
| Host/Launcher | arch22 公共 foreach Tiling | arch22 公共 foreach Tiling | arch35 专用前缀和 Tiling，workspace=0 |
| Kernel/资源 | AIV_VECTOR，TQue/TBuf/高级 Acos API | 同 A2 | SIMT 512-thread，GM 标量访问，无显式 UB |

## 限制与失败边界

- 950 明确限制 TensorList 项数≤256；`totalElements` 的 `int64_t` 累加没有显式溢出检查。
- `GetTilingKeyByDtype` 的 default 返回 FP32 key，但受支持 ACLNN/OpDef 只允许三种 dtype；不得把该防御 default 解释为其它 dtype 支持。
- TensorList descriptor 的首个 `uint64_t` 被解释为到指针表的字节偏移，Device 右移 3 后索引；ABI 不可擅改。
- API inner bridge、executor 内部 launch 调用及 `acosf` 底层实现是固定仓未落盘的框架/库边界。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_acos/docs/aclnnForeachAcos.md` | 两段式原型与产品表 | API、错误、平台和 dtype |
| OpDef/生成接线 | `foreach/foreach_acos/op_host/foreach_acos_def.cpp` | `ops::ForeachAcos` | 动态列表、AutoContiguous、ascend950 |
| Host/Tiling | `foreach/foreach_acos/op_host/arch35/foreach_acos_tiling_arch35.cpp` | `ForeachAcosTilingFunc` | 前缀和、grid、key、workspace |
| POD | `foreach/foreach_acos/op_host/arch35/foreach_acos_tiling_arch35.h` | `ForeachAcosTilingData` | Host ABI |
| Kernel entry | `foreach/foreach_acos/op_kernel/arch35/foreach_acos.cpp` | `foreach_acos<schMode>` | 三 key router |
| Kernel body | `foreach/foreach_acos/op_kernel/arch35/foreach_acos_simt.h` | `Process`、`ForeachAcosSimtKernel` | SIMT 完整 body |
| Device POD | `foreach/foreach_acos/op_kernel/arch35/foreach_acos_tiling_data.h` | `ForeachAcosTilingData` | Device ABI 镜像 |
| 模板 key | `foreach/foreach_acos/op_kernel/arch35/foreach_acos_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个 schMode 实例 |
| Host 构建接线 | `foreach/foreach_acos/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950 选择 arch35 Tiling |
| Kernel 构建接线 | `foreach/foreach_acos/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 选择 arch35 entry |
| binary config | `foreach/foreach_acos/op_host/config/ascend950/foreach_acos_binary.json` | `op_list` | 三 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 前缀和内嵌 POD | 字段顺序、257 项、8 B 对齐 | `ForeachAcosTilingData` |
| 调用顺序 | entry→Process→VF→二分→load/compute/store | E==0 必须在二分前返回 | `Process` |
| workspace/内存 | workspace=0，GM 直接访问 | descriptor 指针表算法和 POD 生命周期 | `SimtGetTensorAddr` |
| 同步 | 元素唯一写者，无 barrier/atomic | flat 域不得重叠 | `ForeachAcosSimtKernel` |
| 精度与尾块 | low precision→FP32→low precision | cast 次序和 `flat<coreEnd` | `AcosCompute` |
| 失败边界 | N≤256、UB>128 KiB、平台信息有效 | 失败必须阻断 launch | `ForeachAcosTilingFunc` |
