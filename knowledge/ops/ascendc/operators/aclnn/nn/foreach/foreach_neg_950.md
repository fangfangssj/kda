---
type: Operator
title: "ForeachNeg 950 ACLNN 算子知识"
description: "说明 aclnnForeachNeg 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/docs/aclnnForeachNeg.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_host/foreach_neg_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_host/arch35/foreach_neg_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_kernel/arch35/foreach_neg_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_kernel/arch35/foreach_neg_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_kernel/arch35/foreach_neg.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_kernel/arch35/foreach_neg_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_host/config/ascend950/foreach_neg_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:59:19Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachNeg 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachNegGetWorkspaceSize` + `aclnnForeachNeg`，公开 ACLNN 两段式接口，无显式版本后缀 |
| 计算语义 | 对 TensorList 中每个元素执行 `y[i][j] = -x[i][j]` |
| 输入输出 | 动态 ND TensorList；FP16/FP32/INT32/BF16；输出逐项继承输入 shape/dtype |
| 执行类型 | 单 Kernel：`SIMT`（AIV entry 内 `VF_CALL`） |
| Kernel 路线 | R-001 四 dtype 统一 SIMT grid-stride 路线，`schMode` 0/1/2/3 只改变模板类型 |
| 核心资源 | GM TensorList 指针表、GM tiling POD；1024 threads/block；无 AIV Queue/TBuf、无 shared/dynamic UB、无 workspace、无 atomic |
| 来源闭包 | `source_closed`：公开 ACLNN 文档与 `ACLNNTYPE aclnn` OpDef 生成边界，闭合到 arch35 SIMT Tiling、模板 key/POD、`foreach_neg` router、SIMT VF body 和 ascend950 构建/binary config |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

输入 `x=[x_0,...,x_(n-1)]`，输出 `y=[y_0,...,y_(n-1)]`。每个 SIMT 线程以全局 `tid` 为起点、全网格线程数为 stride，逐 Tensor 写 `y_t[idx]=-x_t[idx]`。Tensor 元素数为 0 时跳过；非空 Tensor 的每个索引恰由一个线程处理。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 0～8 维、ND | FLOAT16、FLOAT32、INT32、BFLOAT16 | 非空指针；支持非连续输入并进入 `AutoContiguous` 生成边界；列表内 dtype 一致 |
| `out` | TensorList；每个 Tensor 0～8 维、ND | 与 `x` 一致 | 非空指针；公开接口不支持非连续输出；同位置输出 storage 容量不得小于输入，Kernel 写输入 `numel` |
| `workspaceSize` | 标量 | `uint64_t` | 核心 950 Tiling 固定返回 0；生成 executor 的框架临时区公式固定仓不可见 |
| `executor` | 不适用 | `aclOpExecutor**` / `aclOpExecutor*` | 第一段创建、第二段消费 |
| `workspace` | Device 地址 | `void*` | entry 保留参数但不读取 |
| `stream` | Runtime stream | `aclrtStream` | executor 在该 stream 上提交单 Kernel |

### 入口与版本

固定仓只存在无版本后缀公开入口。`foreach_neg_def.cpp` 把 ascend950 添加到 OpDef config，`op_host/CMakeLists.txt` 将 ascend950 映射到 arch35 并标记 `ACLNNTYPE aclnn`；`op_kernel/CMakeLists.txt` 对 ascend950 独占选择 `arch35/foreach_neg.cpp`。公开 README/API 文档明确支持 Ascend 950PR/950DT，没有平台冲突。ACLNN 两段式实现由 opbuild 从 OpDef 生成，生成 cpp 不在固定源码树中，作为框架边界。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段公开接口完成文档声明的空指针、dtype、shape、列表一致性检查；生成桥源码固定仓不可见。
2. OpDef 约束动态 ND TensorList 和四种 dtype，并声明 `AutoContiguous()`；公共 infer-shape/dtype 注册逐项复制输入 shape/dtype。
3. `ForeachNegTilingFunc` 读取 AIV core 数和 UB，二者为 0 均失败；取得输入实例数 `tensorNum` 和首个输入 dtype。
4. Tiling 清零 POD，逐 Tensor 读取 storage `numel` 写入 `tensorElements[i]` 并累加 `totalElements`。
5. dtype 依次映射到 `schMode` 0/1/2/3；其它 dtype 记录错误并返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

设 `N=sum_i numel(x_i)`、平台 AIV core 数为 `C`：

```text
N == 0: blockDim = needCoreNum = 1
N > 0 : blockDim = needCoreNum = min(C, ceil(N/1024))
threadsPerBlock = 1024
total SIMT threads = blockDim * 1024
workspace[0] = 0
localMemorySize = UB_size - 128 KiB
```

`tensorElements` 和 TensorList 指针表在 K0 整个执行期有效。POD 由 Host runtime 持有到 launch 完成；输入/输出由调用者/executor 持有。`localMemorySize` 为 launch 配置，当前 body 不分配 dynamic/shared UB。公开 ACLNN 若为非连续输入建立临时 Tensor，其 owner/lifetime 属于生成 executor 边界。

### 启动图与失败传播

```text
aclnnForeachNegGetWorkspaceSize
  -> generated executor / contiguous boundary
  -> ForeachNegTilingFunc
aclnnForeachNeg(..., stream)
  -> K0 foreach_neg<schMode><<<blockDim, stream>>>(xList,yList,null,tiling)
     -> NsForeachNeg::Process<T>
        -> VF_CALL<OpForeachNegSimt<T>>(Dim3(1024), tensorCount, elemCounts, x, y)
```

单节点 K0 无 workspace slice、无节点间 GM 中间量、无 event 边。Tiling 任一空指针、平台信息、dtype 或 POD 初始化失败都返回 `GRAPH_FAILED` 并阻断 K0；launch/runtime 错误由第二段 ACLNN 状态传播。stream 只需保证框架前处理先于 K0。

## Route 与参数设计

### 分核与资源计算

Tiling 用总元素数只决定 `blockDim`，不把 Tensor 分给特定 block。所有 block/线程都遍历全部 Tensor；对 Tensor `t`，线程坐标为：

```text
tid = blockIdx * 1024 + threadIdx
stride = 1024 * blockDim
for idx = tid; idx < tensorElements[t]; idx += stride:
    y_t[idx] = -x_t[idx]
```

因此单个大 Tensor 被整个 grid 切分，小 Tensor 只激活 `idx<count` 的线程；每个 Tensor 都从 `tid` 重新开始。没有 tile、former/tail core、UB 容量计算或 workspace 切片；尾部完全由循环条件保护。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 SIMT grid-stride | `schMode=0/1/2/3` → `float/half/int32_t/bfloat16_t` | Host `if/else if` 顺序：FLOAT、FLOAT16、INT32、BF16；OpDef/公开 API 约束成立且 ascend950 | 非四 dtype 直接 `GRAPH_FAILED`；模板值之外无 body |

```text
FLOAT   -> GET_TPL_TILING_KEY(0) -> foreach_neg<0> -> Process<float>
FLOAT16 -> GET_TPL_TILING_KEY(1) -> foreach_neg<1> -> Process<half>
INT32   -> GET_TPL_TILING_KEY(2) -> foreach_neg<2> -> Process<int32_t>
BF16    -> GET_TPL_TILING_KEY(3) -> foreach_neg<3> -> Process<bfloat16_t>
other   -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 SIMT grid-stride | `schMode` 0/1/2/3 | FP32/FP16/INT32/BF16；ND；公开维度≤8；950/ascend950 | 单 K0，AIV entry→SIMT VF | `foreach_neg<schMode>`→`Process<T>`→`OpForeachNegSimt<T>` | 直接 GM load/neg/store；1024 threads/block；无 UB/shared/atomic/barrier/workspace | `idx<count` 循环保护；空 Tensor `continue`；其它 dtype Host 失败 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台资源 | `coreNum!=0 && ubSize!=0` | blockDim 与 local-memory launch 配置需要平台值 | R-001 | `GRAPH_FAILED` | `GetPlatformInfo` |
| 每核最少 1024 元素启发式 | `blockDim=min(C,ceil(N/1024))` | 对应固定 1024 threads/block，避免少量元素启动全部 core | R-001 | N=0 特判 1 core | `ForeachNegTilingFunc` |
| DCache 预留 | `SetLocalMemorySize(uint32_t(ubSize-128KiB))` | launch 为 DCache 保留 128 KiB | R-001 | `ubSize<128KiB` 无显式保护，存在无符号下溢边界 | `ForeachNegTilingFunc` |
| POD Tensor 数组 | `tensorElements[256]` | Device 逐 `t<tensorCount` 读取固定数组 | R-001 | Host 未检查 `tensorNum<=256`，超界无 fallback | `ForeachNegTilingData`、`ForeachNegTilingFunc` |
| 空 Tensor | `count==0` | 避免解析数据地址和无效循环 | R-001 | `continue` | `OpForeachNegSimt` |
| SIMT tail | `idx<count`，`idx+=1024*blockDim` | 防止末轮越过 Tensor 元素域 | R-001 | 线程自然退出 | `OpForeachNegSimt` |
| dtype | FLOAT→FLOAT16→INT32→BF16 的 `if/else if` | router 只有四个模板 specialization | R-001 | 其它 dtype `GRAPH_FAILED` | `ForeachNegTilingFunc`、`foreach_neg<schMode>` |

### POD/ABI

`ForeachNegTilingData` 是普通标准布局结构；按字段声明与自然对齐，`needCoreNum` offset 0、`tensorCount` offset 4、`totalElements` offset 8、`tensorElements` offset 16，`sizeof=2064`、`alignof=8`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachNegTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；sizeof 4 | core | `ForeachNegTilingFunc: 1 或 min(C,ceil(N/1024))` | body 不读取；与 blockDim 重复记录 | N=0 时 1 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4；sizeof 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process` 传给 SIMT；`for(t<tensorCount)` | 数组容量 256；uint64→int32 无检查 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8；sizeof 8 | 元素 | `sum(numel)` | body 不读取；仅 Host 分核 | 求和溢出无检查 |
| 同上 | `tensorElements` | `int64_t[256]` | offset 16；sizeof 2048 | 每 Tensor 元素 | 循环 `tiling->tensorElements[i]=numel` | GM tiling offset 指针；SIMT `tensorElements[t]` | `i<tensorNum` 未显式限制到 256 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 SIMT grid-stride | template `schMode` 0/1/2/3 | `op_host/arch35/foreach_neg_tiling.cpp::ForeachNegTilingFunc` | `ForeachNegTilingData`；blockDim、localMemorySize | `op_kernel/arch35/foreach_neg.cpp::foreach_neg<schMode>` | `op_kernel/arch35/foreach_neg_simt.h::Process/OpForeachNegSimt/SimtGetTensorAddr` | Host ascend950→arch35；Kernel ascend950→arch35 cpp；ascend950 binary config 四 dtype |

## Kernel 实现

### 公共执行流程

四个模板 key 只改变 `T`。entry 注册/解析 `ForeachNegTilingData`，直接把 GM tiling 指针交给 `Process<T>`。`Process` 用 `offsetof(ForeachNegTilingData,tensorElements)` 得到 GM 数组指针，并以 `VF_CALL` 启动 1024 threads/block 的 SIMT body。body 对每个 Tensor 解析 TensorList 指针表，以全网格 grid-stride 访问 GM 并直接写输出。没有 AIV Vector primitive、Queue/TBuf、NDDMA、通信、atomic 或多 Kernel 边。

### Route R-001：四 dtype SIMT grid-stride

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach/foreach_neg/op_host/foreach_neg_def.cpp` | `ForeachNeg::ForeachNeg` | ascend950；dtype 四选一；ND dynamic list | 进入 arch35 Tiling |
| Host/Tiling | `foreach/foreach_neg/op_host/arch35/foreach_neg_tiling.cpp` | `ForeachNegTilingFunc` | dtype `if/else if`；`N` 与 coreNum | schMode 0..3、blockDim、POD |
| Template key | `foreach/foreach_neg/op_kernel/arch35/foreach_neg_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | `schMode` 候选 0,1,2,3 | 编译四个实例 |
| Router | `foreach/foreach_neg/op_kernel/arch35/foreach_neg.cpp` | `foreach_neg<schMode>` | `if constexpr schMode==0/1/2/3` | `Process<float/half/int32_t/bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachNegTilingFunc -> blockDim / ForeachNegTilingData / schMode
Host launch K0: foreach_neg<schMode>(xList,yList,workspace,tiling)
  -> NsForeachNeg::Process<T>
     -> elemCounts = tiling + offsetof(tensorElements)
     -> VF_CALL<OpForeachNegSimt<T>>(Dim3(1024), tensorCount, elemCounts, xList, yList)
        -> SimtGetTensorAddr<T>(xList/yList,t)
        -> grid-stride GM load -> unary minus -> GM store
```

##### 源码等价伪码

```text
# 伪码
blocks = blockDim                     # Host: 1 or min(coreNum,ceil(totalElements/1024))
threads = 1024
parallel for (blockIdx, threadIdx):
    tid = blockIdx*threads + threadIdx
    stride = threads*blocks
    for t in [0,tensorCount):
        count = tensorElements[t]
        if count == 0: continue
        xData = TensorListAddress(xList,t)
        yData = TensorListAddress(yList,t)
        for idx = tid; idx < count; idx += stride:
            yData[idx] = -xData[idx]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM TensorList pointer table | `base + ((*uint64(base))>>3) + t` | 每项 8 B 地址 | uint64 | ACLNN marshal | `SimtGetTensorAddr` | executor/调用者，整个 K0 | 首字相对偏移按 8 B 缩放 |
| GM tiling POD | `tiling` | 2064 B | standard-layout POD | Host Tiling | entry/Process/SIMT | runtime，整个 K0 | `tensorElements` offset 16 |
| GM input element | `xData[idx]` | 1×sizeof(T) | T/ND storage | 调用者 | 当前 SIMT thread | 调用者，K0 只读 | grid-stride，不做显式向量对齐 |
| thread-local value | 寄存器，精确物理分配固定仓不可见 | 一个 T load/neg/store 值及循环标量 | T | GM load | unary minus/GM store | 单线程单迭代 | 下一 idx 覆盖 |
| GM output element | `yData[idx]` | 1×sizeof(T) | T/ND storage | 当前线程 | 下游 | 调用者 | 每元素单写，无冲突 |
| AIV UB/shared/dynamic UB | N/A | 0 B 显式分配 | N/A | 无 | 无 | N/A | launch 设置 localMemorySize，但 body 不分配 |
| workspace | N/A | 0 B | N/A | Host | 无 | executor | entry 参数未读 |

##### 分层内存布局

N/A：此 Route 不使用 AIV `TQue/TBuf`、SIMT shared/dynamic UB 或中间 workspace；数据从 GM 直接经线程寄存器写回 GM。

##### 字节账与复用顺序

```text
UB_total = 0 B（无显式 Queue/TBuf/shared 分配）
dynamic_ub = 0 B（VF_CALL 未提供 dynamic UB 参数）
workspace_total = 0 B
localMemorySize launch cap = UB_size - 128 KiB（当前 body 不消费）
thread register bytes = 固定仓/编译器不可见，只能确认单迭代生命周期
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `xData[idx]` 值 | GM load | unary minus、GM store | 同线程下一 `idx` | 顺序语句依赖 | 每个 `idx` 仅一个线程命中 |
| 无共享子区 | N/A | N/A | N/A | 无 barrier/atomic | grid-stride 索引必须唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Tensor 空域 | `if(count==0) continue` | POD count | 地址解析 | 空 Tensor | 不访问其 Tensor 数据地址 |
| 元素分工 | `idx=tid; idx<count; idx+=stride` | block/thread ids | GM load/store | 输出元素唯一性 | `idx<count` 是全部 tail 保护 |
| 计算 | C++ unary `-` | GM load | GM store | 当前元素 | T 原 dtype；无 Cast/round/accumulate |
| 线程/块同步 | N/A | 每线程独立 | 无等待点 | 无共享写 | 无 barrier/atomic；单 Kernel stream 完成提供对下游可见性 |
| 数值边界 | 无额外 guard | unary minus | 输出 | 标量值 | INT32_MIN、NaN payload 等位级行为固定源码未单独约束 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_neg/op_host/arch35/foreach_neg_tiling.cpp` | `GetPlatformInfo`、`ForeachNegTilingFunc` | blockDim、POD、workspace=0、localMemorySize、dtype key |
| Kernel entry/router | `foreach/foreach_neg/op_kernel/arch35/foreach_neg.cpp` | `foreach_neg<0/1/2/3>` | schMode→T mapping |
| Device body/helper | `foreach/foreach_neg/op_kernel/arch35/foreach_neg_simt.h` | `Process<T>` | tiling GM offset 与 `VF_CALL(Dim3(1024))` |
| Device body/helper | `foreach/foreach_neg/op_kernel/arch35/foreach_neg_simt.h` | `OpForeachNegSimt<T>`、`SimtGetTensorAddr<T>` | TensorList 地址、block/thread 映射、grid-stride、空域与 direct GM store |

## 算子特定优化

### Trick T-001：跨整个 TensorList 复用统一 grid-stride 网格

#### 触发与源码依据

- 适用 Route：R-001
- 触发条件：所有四种 dtype；`tensorCount` 与 POD 数组合法；单 Kernel SIMT launch
- Golden 来源：[`foreach/foreach_neg/op_kernel/arch35/foreach_neg_simt.h:57-74`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_neg/op_kernel/arch35/foreach_neg_simt.h#L57-74)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    for (int32_t t = 0; t < tensorCount; t++) {
        int64_t count = tensorElements[t];
        if (count == 0) {
            continue;
        }

        __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
        __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);

        uint64_t tid = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
            AscendC::Simt::GetThreadIdx());
        uint64_t stride = static_cast<uint64_t>(
            AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            yData[idx] = -xData[idx];
        }
```

#### 机制与设计原因

Host 只按总元素数选择一次 blockDim，Device 在同一 Kernel 中让同一网格依次覆盖所有 Tensor。每个 Tensor 以相同 `tid/stride` 重新分工，省去逐 Tensor launch、workspace 描述符和跨节点同步；`idx<count` 同时处理任意长度尾部，空 Tensor 在地址解析前跳过。该结论只说明源码减少了 launch/中间物化，不包含未实测性能比例。

#### 迁移边界

- 适用：TensorList 中各 Tensor 可独立做同一种纯逐元素运算，且无需跨 Tensor 归约或共享状态。
- 前提：POD Tensor 数组容量覆盖列表；每个输出索引只能由一个 grid-stride 线程命中；TensorList 地址表 ABI 保持一致。
- 失效：每 Tensor 需要不同属性/网格、需要 barrier/atomic、或列表超过 256 项时必须改造 Tiling/POD/body，固定仓没有 fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 SIMT grid-stride | R-002 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | 四 dtype，schMode 0..3 | N/A（无第二 Route） | dtype 只改变模板类型 |
| 模板与 dtype | `Process<float/half/int32_t/bfloat16_t>` | N/A | 原 dtype unary minus |
| 循环与分核 | blockDim=`min(C,ceil(N/1024))`；1024 threads/block；逐 Tensor grid-stride | N/A | 总元素决定网格，单 Tensor count 决定线程尾部 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/shared/dynamic UB；线程寄存器物理字节不可见 | N/A | direct GM 路径 |
| 内存总占用与复用 | UB 0、workspace 0；仅线程局部值跨迭代覆盖 | N/A | 无中间物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar GM load/neg/store | N/A | 不使用 Vector/NDDMA/通信 |
| 同步、精度与 tail | 无 barrier/atomic；原 dtype；`idx<count` | N/A | 输出索引无冲突 |
| fallback/失败边界 | 非四 dtype失败；数组>256、UB<128KiB 无 fallback | N/A | 固定单实现族 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，独立无后缀卡覆盖 | 公开支持，独立无后缀卡覆盖 | 公开支持 |
| Host/Launcher | 本卡不覆盖 | 本卡不覆盖 | ascend950→arch35，自定义 SIMT Tiling |
| Kernel/资源 | 本卡不覆盖 | 本卡不覆盖 | 单 SIMT Kernel；1024 threads/block；UB/workspace 0 |

## 限制与失败边界

- POD `tensorElements` 固定 256 项，但 `ForeachNegTilingFunc` 未检查 `tensorNum<=256`，还会把 `uint64_t tensorNum` 窄化为 `int32_t tensorCount`；公开文档未给 TensorList 长度上限，因此不得外推超过 256 的安全性。
- `totalElements` 为有符号 64 位累加，固定源码未做求和溢出检查；负/溢出结果会破坏 blockDim 计算。
- `SetLocalMemorySize(static_cast<uint32_t>(ubSize-128KiB))` 未先检查 `ubSize>=128KiB`；目标 950 平台满足资源是实现前提，无 fallback。
- 核心 workspace 为 0；公开 ACLNN 对非连续输入的框架临时区公式和 executor owner 细节固定仓不可见。
- SIMT 原 dtype unary minus 不做 Cast；固定源码未单独规定 INT32_MIN、NaN payload、异常状态或浮点位级行为。
- 输出只写输入 `numel`；若公开允许输出 storage size 更大，多余区域保持调用前内容，不能视为已初始化。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_neg/docs/aclnnForeachNeg.md` | 两段式 ACLNN | 公开接口、平台、dtype/shape、错误与确定性 |
| 入口实现 | `foreach/foreach_neg/op_host/foreach_neg_def.cpp` | `ForeachNeg`、ascend950 config | ACLNN codegen 输入、接口 dtype/layout/AutoContiguous |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`、ForeachNeg 注册 | 输出 shape/dtype 逐项继承 |
| Host/Tiling/Launcher | `foreach/foreach_neg/op_host/arch35/foreach_neg_tiling.cpp` | `ForeachNegTilingFunc` | 平台、POD、blockDim、workspace、local memory、key |
| Host/Tiling/Launcher | `foreach/foreach_neg/op_kernel/arch35/foreach_neg_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 候选与模板选择 |
| Host/Tiling/Launcher | `foreach/foreach_neg/op_kernel/arch35/foreach_neg_tiling_data.h` | `ForeachNegTilingData` | POD ABI |
| Kernel entry/body | `foreach/foreach_neg/op_kernel/arch35/foreach_neg.cpp` | `foreach_neg<schMode>` | key router |
| Kernel entry/body | `foreach/foreach_neg/op_kernel/arch35/foreach_neg_simt.h` | `Process`、`OpForeachNegSimt`、`SimtGetTensorAddr` | VF launch、线程映射、地址、循环、tail |
| 构建接线 | `foreach/foreach_neg/op_host/CMakeLists.txt` | ascend950→arch35、`ACLNNTYPE aclnn` | 950 Host/ACLNN 接线 |
| 构建接线 | `foreach/foreach_neg/op_host/config/ascend950/foreach_neg_binary.json` | 四个 `op_list` 项 | 950 四 dtype binary config |
| 构建接线 | `foreach/foreach_neg/op_kernel/CMakeLists.txt` | ascend950 `KERNEL_SRC arch35/foreach_neg.cpp` | 950 Kernel 构建选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B 四字段 POD 与 TensorList 指针表 | offset 0/4/8/16、数组 256、字段单位与窄化边界 | `ForeachNegTilingData`、`SimtGetTensorAddr` |
| 调用顺序 | Tiling→单 `foreach_neg<schMode>`→Process→VF body | schMode→T mapping、VF_CALL 1024 threads 不得错配 | `ForeachNegTilingFunc`、`foreach_neg`、`Process` |
| workspace/内存 | direct GM、workspace 0、无显式 UB | DCache 128 KiB 预留和 Tensor/POD lifetime | `SetLocalMemorySize`、`VF_CALL` |
| 同步 | 无共享写、无需 barrier/atomic | 必须保持每个 `idx` 唯一和单 stream launch ordering | `OpForeachNegSimt` |
| 精度与尾块 | 原 dtype unary minus、`idx<count` tail | 不得引入隐式 Cast/round；空 Tensor 必须在地址使用前跳过 | `OpForeachNegSimt` |
| 失败边界 | 非四 dtype Host 失败 | 256 Tensor 容量、int32 窄化、totalElements 与 UB 减法边界 | `ForeachNegTilingFunc` |