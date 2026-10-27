---
type: Operator
title: "ForeachAtan 950 ACLNN 算子知识"
description: "说明 aclnnForeachAtan 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/docs/aclnnForeachAtan.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_host/foreach_atan_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_host/arch35/foreach_atan_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_kernel/arch35/foreach_atan_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_kernel/arch35/foreach_atan_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_kernel/arch35/foreach_atan.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_kernel/arch35/foreach_atan_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_host/config/ascend950/foreach_atan_binary.json
    role: implementation
created_at: "2026-08-25T15:31:06Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAtan 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAtanGetWorkspaceSize` + `aclnnForeachAtan`，公开两段式 ACLNN，无更高版本入口 |
| 计算语义 | 对 TensorList 每项逐元素计算 `out_i[j]=atan(x_i[j])` |
| 输入输出 | 动态 TensorList、ND、FP16/FP32/BF16；同一列表及输入输出 dtype 一致；每项输出容量不小于输入 |
| 执行类型 | 单 Host Kernel，节点内部为 `SIMT` VF |
| Kernel 路线 | R-001 FP32 `atanf`；R-002 FP16→FP32→`atanf`→FP16；R-003 BF16→FP32→`atanf`→BF16 |
| 核心资源 | TensorList GM 指针表、GM tiling POD、每 block 1024 SIMT threads、标量寄存器；无 AIV Queue/TBuf、无算子 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN、Ascend950 OpDef/构建接线、arch35 Tiling/POD/template key、Kernel entry 与三个 SIMT VF body 闭合；生成 ACLNN bridge 和 SIMT math intrinsic 内部作为框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

输入为 `x=[x_0,...,x_(L-1)]`。对每个列表项 `i` 和 storage-shape 的线性元素域：

```text
0 <= j < numel(x_i): out_i[j] = atan(x_i[j])
```

arch35 Kernel 直接按 TensorList 中的 GM 地址和 `tensorElements[i]` 访问，不依赖原始维度或 stride。若输出容量大于输入，只有前 `numel(x_i)` 个元素被写；额外输出区间固定源码未初始化。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 0–8 维；ND | FP16、FP32、BF16 | 非空指针；支持空 Tensor；同一列表 dtype 一致；公开 ACLNN 支持非连续输入，OpDef 配置 `AutoContiguous()` |
| `out` | TensorList；每项 0–8 维；ND | 与 `x` 相同 | 非空指针；列表项一一对应；每项元素容量不小于输入；公开接口不支持非连续输出 |
| `workspaceSize` | 标量输出 | `uint64_t*` | arch35 Tiling 写 `0`，无需算子 workspace |
| `executor` | Host 对象输出 | `aclOpExecutor**` | 第一段接口生成，第二段消费 |
| `workspace` | Device 地址 | `void*` | workspaceSize 为 0 时可不分配；Kernel 形参存在但不读取 |
| `stream` | Runtime stream | `aclrtStream` | 第二段在该 stream 提交单 Kernel |

无算子属性；公开文档声明默认确定性实现。

### 入口与版本

固定 Commit 只有 `aclnnForeachAtanGetWorkspaceSize`/`aclnnForeachAtan` 这一公开版本族。ACLNN 生成 bridge 的函数体在固定仓不可见；`op_host/CMakeLists.txt` 的 `ACLNNTYPE aclnn`、`ForeachAtan` OpDef、公开文档和后续完整源码共同闭合该框架边界。

公开产品表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 将其映射到 `arch35`，Kernel CMake指定 `arch35/foreach_atan.cpp`，Ascend950 binary config 列出三种 dtype。因此 950 使用本卡的 arch35 SIMT 实现，而不复用 A2/A3 的 arch22 Vector body。

## Host/Launcher 执行流程

### 参数校验与前处理

公开 ACLNN 错误表约束 `x/out` 非空、dtype受支持且一致、shape满足要求、列表内部 dtype 一致、rank不超过8。生成式 ACLNN 校验/连续化的具体函数固定仓不可见。可见 arch35 流程为：

1. `ForeachAtanTilingFunc` 读取 AIV core 数和 UB 字节数；二者为 0 均返回 `GRAPH_FAILED`。
2. 从动态输入实例信息读取 `tensorNum`，要求 `tensorNum<=256`；从第 0 个输入描述取 dtype。
3. POD 先整块 `memset_s(...,0)`，再写每项 storage-shape 元素数和总元素数。
4. 根据总元素数生成 block 数，workspace 设 0，local memory 设为 `ubSize-128KiB`，最后按 dtype 选择 `schMode`。

输出列表项/shape/dtype 的可见 Tiling 检查不在 arch35 文件内；它们由公开 ACLNN/OpDef/框架推导约束。`ForeachAtanTilingFunc` 读取第 0 个动态输入描述，因此“TensorList 零项”不由本源码证明可用；公开文档只明确“空 Tensor”，不是空列表。

### 参数、Workspace 与生命周期

令 `N=sum_i numel(x_i)`，平台可用 AIV core 数为 `C_hw`：

```text
if N == 0:
    C = 1
else:
    C = min(C_hw, ceil(N / 1024))
blockDim = C
threadsPerBlock = 1024
globalStride = C * 1024 threads
```

Host 将 `needCoreNum=C` 同时写 POD 和 `SetBlockDim(C)`；Device 只通过 `GetBlockNum()`获取 grid 大小，未读取 POD 的 `needCoreNum`。每个 block 都启动 `THREAD_NUM=1024` 的 VF，所有线程用全局 `tid=blockIdx*1024+threadIdx` 对每个 Tensor 做 grid-stride 循环。

`workspace[0]=0`；`SetLocalMemorySize(uint32_t(ubSize-128*1024))` 配置 launch local memory。三个 VF body 没有显式 dynamic-UB/shared-memory 数组或 `TQue/TBuf`，只保留该 launch 配置；其 runtime 物理布局固定仓不可见。

executor 由第一段拥有并传给第二段；输入、输出及 executor 必须保持到 launch 提交，输入/输出 GM 至少保持到 stream 任务完成。无 workspace owner/lifetime。

### 启动图与失败传播

```text
aclnnForeachAtanGetWorkspaceSize
  -> generated ACLNN bridge（固定仓不可见）
  -> ForeachAtanTilingFunc
     -> POD + blockDim + localMemorySize + schMode
  -> executor records K0

aclnnForeachAtan(..., stream)
  -> K0 foreach_atan<schMode><<<C, stream>>>(xList,yList,null,tilingGM)
     -> VF_CALL<dtype body>(Dim3(1024), tensorCount, elemCounts, x, y)
```

| 节点/边 | entry/介质 | extent、layout | ordering 与 owner/lifetime | 失败传播 |
|---|---|---|---|---|
| Node K0 | `foreach_atan<schMode>` | C blocks × 1024 threads；POD `ForeachAtanTilingData` | executor 在调用 stream 提交；`AUTO_SYNC false`，不插入算子级自动同步 | Tiling 任一空指针、平台0值、>256项或 unsupported dtype 失败即不形成 launch；runtime launch错误由 ACLNN 返回 |
| tiling edge | GM POD | 2064B、自然对齐8 | executor/runtime持有到 K0 完成读取 | 无后继节点 |
| x/y edge | TensorList GM对象 | pointer table + 每项线性数据 | 调用者持有至 stream 完成 | K0失败时输出不可用 |
| workspace edge | N/A | 0B | 无 owner | 无 producer/consumer |

这是单 Host Kernel。`VF_CALL` 是 K0 内部 SIMT 执行体调用，不形成第二个 Host launch 或 workspace 边。

## Route 与参数设计

### 分核与资源计算

Host 只按总元素数估算 core 数，每核目标至少约 1024 个总元素；Device 不静态分配 Tensor 区间，而是所有 block/thread 对每个 Tensor 重用同一 `tid/stride`。因此 Tensor `i` 的线程 `tid` 处理：

```text
idx = tid, tid + C*1024, tid + 2*C*1024, ... < tensorElements[i]
```

小 Tensor 中 `tid>=count` 的线程自然不进入内层循环；空 Tensor 由 `count==0` 直接跳过。没有 former/tail core 表，也没有 DMA 对齐或 tile loop；尾部由 `idx<count` 逐元素保护。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `schMode=0` / FLOAT | `dataType==ge::DT_FLOAT` | 检查 FP16 |
| 2 | R-002 FP16 升精度 SIMT | `schMode=1` / FLOAT16 | `dataType==ge::DT_FLOAT16` | 检查 BF16 |
| 3 | R-003 BF16 升精度 SIMT | `schMode=2` / BF16 | `dataType==ge::DT_BF16` | unsupported dtype→`GRAPH_FAILED` |

```text
if dtype == FLOAT:   GET_TPL_TILING_KEY(0) -> foreach_atan<0> -> Process<float>
else if dtype == FLOAT16: GET_TPL_TILING_KEY(1) -> foreach_atan<1> -> Process<half>
else if dtype == BF16:    GET_TPL_TILING_KEY(2) -> foreach_atan<2> -> Process<bfloat16_t>
else: GRAPH_FAILED
```

`ASCENDC_TPL_ARGS_DECL/SEL` 仅生成这三个 `schMode` specialization；不存在 shape/资源 fallback Route。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | FP32、ND、公开约束满足、Tensor数<=256、950 | 单 K0→SIMT VF | `foreach_atan<0>`→`Process<float>`→`OpForeachAtanSimtFloat` | 直接 GM load/`atanf`/store；无 barrier/atomic/shared UB | `idx<count`；无 fallback |
| R-002 FP16 | schMode 1 | 同上但 FP16 | 单 K0→SIMT VF | `foreach_atan<1>`→`Process<half>`→`OpForeachAtanSimtHalf` | 每线程标量 half→float→half；无同步 | 同上 |
| R-003 BF16 | schMode 2 | 同上但 BF16 | 单 K0→SIMT VF | `foreach_atan<2>`→`Process<bfloat16_t>`→`OpForeachAtanSimtBfloat16` | 每线程标量 BF16→float→BF16；无同步 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList 容量 | `tensorNum<=256` | POD `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `ForeachAtanTilingFunc` |
| 每核最少总工作量 | `C=min(C_hw,ceil(N/1024))`，N>0 | 每 block 1024 threads，避免总线程数显著超过总元素 | 全部 | 不是 fallback；N=0 强制1 block | `MIN_PER_CORE`, core split |
| 线程上限 | `THREAD_NUM=1024`, `LAUNCH_BOUND(1024)` | VF body 编译/launch specialization | 全部 | 无其它线程数 Route | `foreach_atan_simt.h` |
| DCache local-memory reserve | `localMemorySize=uint32_t(ubSize-128KiB)` | 从平台 UB 扣除 `DCACHE_SIZE` | 全部 | 固定源码只检查 `ubSize!=0`，未检查 `ubSize>=128KiB`；若不满足会无符号下溢 | `ForeachAtanTilingFunc` |
| dtype | 三个精确分支 | 决定 VF template 与标量转换 | 各 Route | unsupported→`GRAPH_FAILED` | `ForeachAtanTilingFunc`, template selector |
| 空 Tensor | `count==0` | 避免解析指针和内层循环 | 全部 | `continue` | 三个 `OpForeachAtanSimt*` |
| 64位索引 | `numel`/idx/stride 为 int64/uint64 | 支持单 Tensor 大元素数和跨 block stride | 全部 | `totalElements` 求和无显式溢出检查 | Tiling + VF loops |

### POD/ABI

`ForeachAtanTilingData` 是普通结构，按固定字段顺序自然对齐：`sizeof=2064`、`alignof=8`。Device 通过 GM 指针与 `offsetof(tensorElements)` 访问，避免把大数组作为本地栈对象使用。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAtanTilingData` | `needCoreNum` | `int32_t`/32 | offset 0，size 4 | block/core 数 | N=0写1；否则 `int32_t(min(C_hw,ceil(N/1024)))` | body不直接读取；真实值由 `SetBlockDim`→`GetBlockNum`消费 | 初始化0；coreNum>0后写 |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4，size 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`读取并传给 VF；VF外层 `t<tensorCount` | Host先保证<=256 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8，size 8 | 元素 | 累加每项 numel | body不读取；仅Host分核使用 | 无显式加法溢出检查 |
| 同上 | `tensorElements` | `int64_t[256]`/每项64 | offset 16，size 2048 | 元素/每Tensor | `storageShape.GetShapeSize()` | `elemCounts[t]`; VF的count与尾部判断 | 未使用项保持memset后的0；负值未显式检查 |

TensorList ABI由 `SimtGetTensorAddr`解释：对象首 `uint64_t` 是到指针数组的字节 offset，右移3转成64位项数，再取第 `idx` 个 GM地址。容器构造代码固定仓不可见。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode0 | `arch35/foreach_atan_tiling.cpp::ForeachAtanTilingFunc` | `ForeachAtanTilingData`, C/1024/local memory | `arch35/foreach_atan.cpp::foreach_atan<0>` | `Process<float>`, `OpForeachAtanSimtFloat`, `SimtGetTensorAddr` | Host CMake→arch35；Kernel CMake→arch35；ascend950 binary JSON FP32 |
| R-002 FP16 | schMode1 | 同上 | 同 POD | `foreach_atan<1>` | `Process<half>`, `OpForeachAtanSimtHalf` | 同上，FP16 binary |
| R-003 BF16 | schMode2 | 同上 | 同 POD | `foreach_atan<2>` | `Process<bfloat16_t>`, `OpForeachAtanSimtBfloat16` | 同上，BF16 binary |

## Kernel 实现

### 公共执行流程

三个 specialization 都把 `tiling` 保留为 GM POD 指针，从 `offsetof(ForeachAtanTilingData,tensorElements)` 得到元素数数组，读取 `tensorCount`，以 `VF_CALL(...Dim3(1024),...)` 进入对应 SIMT VF。VF 外层逐 Tensor 循环；非空项解析 x/y GM 地址，计算全局 tid/stride，再逐元素直接 load、计算、store。不同线程处理互斥索引，没有中间物化、共享内存、atomic、barrier 或跨 block通信。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | docs + `foreach_atan_def.cpp` | dtype表 | FP32，950 | 支持 |
| Tiling | `arch35/foreach_atan_tiling.cpp` | `ForeachAtanTilingFunc` | `dataType==DT_FLOAT` | template key schMode0 |
| Router | `arch35/foreach_atan.cpp` | `foreach_atan<schMode>` | `schMode==TILING_KEY_FLOAT(0)` | `Process<float>` |
| VF | `foreach_atan_simt.h` | `Process<float>` | `is_same_v<T,float>` | `VF_CALL<OpForeachAtanSimtFloat>(Dim3(1024),...)` |

#### Device 数据流

##### 函数级调用链

```text
ForeachAtanTilingFunc -> executor launch foreach_atan<0>
  -> Process<float>
    -> VF_CALL<OpForeachAtanSimtFloat>(Dim3(1024), tensorCount, elemCounts, x, y)
      -> for each Tensor: SimtGetTensorAddr<float>(x/y)
      -> grid-stride loop: yData[idx] = atanf(xData[idx])
```

##### 源码等价伪码

```text
# 伪码；每个 SIMT thread
for t in [0, tensorCount):
    count = tensorElements[t]
    if count == 0: continue
    xData = resolveTensorList<float>(x,t)
    yData = resolveTensorList<float>(y,t)
    tid = blockIdx*1024 + threadIdx
    stride = blockNum*1024
    for idx=tid; idx<uint64(count); idx+=stride:
        yData[idx] = atanf(xData[idx])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM POD | tiling+16 | tensorCount×8有效字节（结构总2064B） | int64 array | Host Tiling | every thread | executor/runtime至K0完成 | 8B自然对齐；只读 |
| GM x_i | pointer[i]+idx×4 | 4B/thread/iteration | FP32 linear ND | caller | current thread | caller至stream完成 | 标量直接load |
| scalar x/y | thread-local | 2×FP32语义值；精确寄存器字节固定仓不可见 | FP32 | load/atanf | atanf/store | 当前迭代 | 后续迭代覆盖 |
| GM out_i | pointer[i]+idx×4 | 4B/thread/iteration | FP32 linear ND | current thread | caller | caller至stream完成 | 每元素唯一writer |
| dynamic local memory | launch配置 | `ubSize-128KiB` | runtime local memory | Host | VF runtime | K0 | body无显式子区，物理布局固定仓不可见 |

##### 分层内存布局

N/A：此 SIMT Route 未创建 AIV `TQue/TBuf` 或显式 dynamic-UB/shared-memory数组；真实数据路径是 `GM scalar load → thread-local scalar → atanf → GM scalar store`。Host仍设置 `localMemorySize=ubSize-128KiB`，其 runtime内部用途固定仓不可见。

##### 字节账与复用顺序

```text
UB_total(AIV Queue/TBuf) = 0
dynamic_ub launch request = ubSize - 131072 B
explicit SIMT shared/dynamic suballocation = 0 B in operator body
workspace_total = 0 B
register_bytes = 固定仓不可见；可见生命周期为每线程每次循环的 x/y/tid/stride 标量
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local x/y scalar | current iteration load/atanf | atanf/store | same thread next idx | 单线程程序顺序 | 不跨线程共享 |
| GM out[idx] | unique tid序列 | caller | K0完成后才可覆盖 | stream completion | grid-stride索引集合互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM load→compute→store | SIMT thread program order | same thread | next statement | x/y scalar | FP32 `atanf`; `idx<count`保护尾部 |
| threads/blocks | N/A，无barrier/atomic | 每线程独立 | 无等待 | 互斥out索引 | count=0整项跳过 |
| Host可见性 | runtime stream ordering | K0 | caller同步/后续stream任务 | GM out | `AUTO_SYNC false`，不自动Host同步 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_atan/op_host/arch35/foreach_atan_tiling.cpp` | `ForeachAtanTilingFunc` | C、POD、workspace0、localMemory、schMode0 |
| Kernel entry/router | `foreach_atan/op_kernel/arch35/foreach_atan.cpp` | `foreach_atan<0>` | GM POD指针→`Process<float>` |
| Device body/helper | `foreach_atan/op_kernel/arch35/foreach_atan_simt.h` | `OpForeachAtanSimtFloat` | 1024线程、grid-stride、atanf直写 |

### Route R-002：FP16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `arch35/foreach_atan_tiling.cpp` | dtype selector | `DT_FLOAT16` | schMode1 |
| Router | `arch35/foreach_atan.cpp` | `foreach_atan<1>` | FLOAT16 enum | `Process<half>` |
| VF | `foreach_atan_simt.h` | `Process<half>` | `is_same_v<T,half>` | half VF |

#### Device 数据流

##### 函数级调用链

```text
foreach_atan<1> -> Process<half>
  -> VF_CALL<OpForeachAtanSimtHalf>(Dim3(1024),...)
    -> SimtGetTensorAddr<half>
    -> __half2float(load) -> atanf -> __float2half -> store
```

##### 源码等价伪码

```text
# 伪码；Tensor/idx循环同R-001
float xVal = __half2float(xData[idx])
float yVal = atanf(xVal)
yData[idx] = __float2half(yVal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | pointer[i]+idx×2 | 2B load/store | FP16 | caller/thread | thread/caller | stream 生命周期 | 标量访问 |
| thread-local `xVal/yVal` | register allocation fixed repo invisible | 2个FP32语义值 | FP32 | conversion/atanf | atanf/conversion | current iteration | next idx覆盖 |
| dynamic local memory | launch配置 | ubSize-128KiB | runtime | Host | runtime | K0 | body无显式子区 |

##### 分层内存布局

N/A：无 AIV UB Queue/TBuf；数据直接在 GM FP16 与每线程 FP32 标量之间转换。

##### 字节账与复用顺序

```text
UB_total(AIV)=0; explicit shared/dynamic allocation=0
dynamic_ub launch request=ubSize-131072 B
workspace_total=0
register_bytes=固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-thread FP32 scalars | conversion/atanf | atanf/final conversion | next idx | thread program order | thread-private |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| half→float | `__half2float` | GM load | atanf | xVal | 精确转换语义由intrinsic定义 |
| compute | `atanf` | xVal | final conversion | yVal | FP32域计算 |
| float→half | `__float2half` | yVal | GM store | out[idx] | rounding mode在固定源码未显式给出；`idx<count`尾保护 |
| inter-thread | N/A | unique thread | none | disjoint idx | 无barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `arch35/foreach_atan_tiling.cpp` | FLOAT16 branch | schMode1 |
| Kernel entry/router | `arch35/foreach_atan.cpp` | `foreach_atan<1>` | half specialization |
| Device body/helper | `arch35/foreach_atan_simt.h` | `OpForeachAtanSimtHalf` | 两次转换夹FP32 atanf |

### Route R-003：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `arch35/foreach_atan_tiling.cpp` | dtype selector | `DT_BF16` | schMode2 |
| Router | `arch35/foreach_atan.cpp` | `foreach_atan<2>` | BF16 enum | `Process<bfloat16_t>` |
| VF | `foreach_atan_simt.h` | `Process<bfloat16_t>` | type match | BF16 VF |

#### Device 数据流

##### 函数级调用链

```text
foreach_atan<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachAtanSimtBfloat16>(Dim3(1024),...)
    -> SimtGetTensorAddr<bfloat16_t>
    -> __bfloat162float(load) -> atanf -> __float2bfloat16 -> store
```

##### 源码等价伪码

```text
# 伪码；Tensor/idx循环同R-001
float xVal = __bfloat162float(xData[idx])
float yVal = atanf(xVal)
yData[idx] = __float2bfloat16(yVal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | pointer[i]+idx×2 | 2B load/store | BF16 | caller/thread | thread/caller | stream 生命周期 | scalar |
| per-thread scalars | compiler register allocation | 2个FP32语义值 | FP32 | convert/atanf | atanf/convert | iteration | next idx覆盖 |
| dynamic local memory | launch配置 | ubSize-128KiB | runtime | Host | runtime | K0 | explicit operator subregion 0 |

##### 分层内存布局

N/A：无 AIV UB Queue/TBuf；GM BF16经每线程FP32标量计算后直接回写。

##### 字节账与复用顺序

```text
UB_total(AIV)=0; explicit shared/dynamic allocation=0
dynamic_ub launch request=ubSize-131072 B
workspace_total=0
register_bytes=固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| per-thread FP32 scalars | BF16 conversion/atanf | atanf/BF16 conversion | next idx | thread order | thread-private |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `__bfloat162float` | GM | atanf | xVal | FP32域输入 |
| compute | `atanf` | xVal | final convert | yVal | FP32域计算 |
| FP32→BF16 | `__float2bfloat16` | yVal | GM | out[idx] | rounding mode固定仓未显式给出；`idx<count`保护 |
| inter-thread | N/A | unique thread | none | disjoint idx | 无同步/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `arch35/foreach_atan_tiling.cpp` | BF16 branch | schMode2 |
| Kernel entry/router | `arch35/foreach_atan.cpp` | `foreach_atan<2>` | BF16 specialization |
| Device body/helper | `arch35/foreach_atan_simt.h` | `OpForeachAtanSimtBfloat16` | BF16/FP32转换与atanf |

## 算子特定优化

### Trick T-001：单次 launch 覆盖 TensorList 的直接 GM grid-stride 计算

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：任意非空 Tensor；`blockDim=C`，每 block 固定1024线程
- Golden 来源：[`foreach/foreach_atan/op_kernel/arch35/foreach_atan_simt.h:58-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_kernel/arch35/foreach_atan_simt.h#L58-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t t = 0; t < tensorCount; t++) {
    int64_t count = tensorElements[t];
    if (count == 0) {
        continue;
    }

    __gm__ float* xData = SimtGetTensorAddr<float>(xList, t);
    __gm__ float* yData = SimtGetTensorAddr<float>(yList, t);

    uint64_t tid = static_cast<uint64_t>(
        AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
        AscendC::Simt::GetThreadIdx());
    uint64_t stride = static_cast<uint64_t>(
        AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

    for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
        yData[idx] = atanf(xData[idx]);
    }
}
```

#### 机制与设计原因

同一个 K0/VF 依次遍历 TensorList，所有 block/thread 对当前 Tensor 用全局 grid-stride 划分索引。它避免为每个列表项单独 launch，也不把输入搬入 AIV Queue或物化中间 Tensor；每个有效元素只发生一次标量 GM load、一次数学计算和一次 GM store。`idx<count`同时处理任意尾长，空 Tensor 直接跳过。FP16/BF16 Route只在每线程寄存器中增加两次标量转换，不改变该搬运拓扑。

#### 迁移边界

- 适用：各元素互不依赖、TensorList项dtype一致、可直接GM标量访问的逐元素算子。
- 前提：SIMT math intrinsic可用；每元素仅一个writer；TensorList不超过256项；线程数固定1024可合法launch。
- 失效：需要DMA合并、跨元素/跨Tensor归约、共享状态或严格对齐Vector primitive时，应设计另一实现族；本算子没有fallback。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：大 POD 保持在 GM，仅传递元素数数组指针

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：全部 arch35 launch；POD 含256项 `int64_t` 数组
- Golden 来源：[`foreach/foreach_atan/op_kernel/arch35/foreach_atan_simt.h:149-166`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_atan/op_kernel/arch35/foreach_atan_simt.h#L149-L166)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
const __gm__ ForeachAtanTilingData* tilingGm)
{
    // Extract tensorElements array pointer from GM tiling data
    __gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
        reinterpret_cast<__gm__ char*>(
            const_cast<__gm__ ForeachAtanTilingData*>(tilingGm)) +
        offsetof(ForeachAtanTilingData, tensorElements));

    int32_t tensorCount = tilingGm->tensorCount;

    if constexpr (std::is_same_v<T, float>) {
        AscendC::Simt::VF_CALL<OpForeachAtanSimtFloat>(
            AscendC::Simt::Dim3(THREAD_NUM),
            tensorCount,
            elemCounts,
```

#### 机制与设计原因

2064B POD 的大头是 `tensorElements[256]`。入口不把它复制成线程本地数组，而以 `offsetof=16` 构造 GM 指针，仅把标量 `tensorCount` 和该指针传给 VF。这避免在 Kernel栈或显式UB中物化整个元素数表；各线程按需要读取当前 `t` 的 count。ABI偏移因此是不可改变的实现约束。

#### 迁移边界

- 适用：大只读POD数组、Device只需顺序/按索引读取的SIMT算子。
- 前提：POD在整个K0期间位于GM且自然对齐；Host/Device结构定义完全一致。
- 失效：若POD生命周期不足、布局改变或访问需要共享缓存/同步，必须重做传参和内存设计。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode0 | schMode1 | schMode2 | dtype单层决策树 |
| 模板与 dtype | `Process<float>` | `Process<half>` | `Process<bfloat16_t>` | 三个binary specialization |
| 循环与分核 | C×1024 grid-stride | 同 | 同 | 总元素只决定C，各Tensor独立重用grid |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无Queue；FP32 scalar | 无Queue；FP32 x/y scalar | 无Queue；FP32 x/y scalar | 直接GM，低精度升精度 |
| 内存总占用与复用 | explicit local 0 | explicit local 0 | explicit local 0 | launch local-memory request相同，body无子分配 |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar load+atanf+store | half convert+atanf | BF16 convert+atanf | 全部SIMT，无NDDMA/通信 |
| 同步、精度与 tail | FP32；idx guard | FP32计算，降FP16 | FP32计算，降BF16 | 每元素唯一writer，无barrier |
| fallback/失败边界 | 无 | 无 | 无 | unsupported dtype/Tensor>256直接失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22公共Tiling，最多50项/50核，workspace32B | 同A2 | arch35独立Tiling，最多256项，workspace0，C=min(core,ceil(N/1024)) |
| Kernel/资源 | AIV_VECTOR、双缓冲Queue；BF16有FP32 Queue | 同A2 | SIMT 1024 threads/block，直接GM，无AIV Queue |

## 限制与失败边界

- 公开ACLNN约束：仅FP16/FP32/BF16 ND，列表内部及输入输出dtype一致，rank<=8，输出不支持非连续Tensor；支持空Tensor。
- arch35显式限制TensorList项数<=256；第0个动态输入描述必须存在，空列表可用性固定仓不可见。
- 输出若比输入大，仅前输入元素数区域被写入；其余区域未定义。
- `totalElements` 使用有符号64位累加但无显式溢出检查；shape产生负numel的路径也未在此Tiling显式拒绝。
- `SetLocalMemorySize(uint32_t(ubSize-128KiB))` 仅在 `ubSize!=0` 后执行，缺少 `ubSize>=128KiB` guard；移植到更小UB平台时必须补充失败边界，不能依赖无符号下溢。
- Kernel无显式barrier/atomic；正确性依赖每个grid-stride索引唯一writer和stream完成后的Host可见性。
- `__float2half`/`__float2bfloat16` 的确切rounding mode、`atanf`近似、VF_CALL/runtime local memory内部、自动生成ACLNN bridge/executor以及TensorList容器构造在固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_atan/docs/aclnnForeachAtan.md` | `aclnnForeachAtanGetWorkspaceSize`, `aclnnForeachAtan` | 原型、约束、错误、950公开支持 |
| 入口实现 | `foreach/foreach_atan/op_host/foreach_atan_def.cpp` | `ForeachAtan` | dtype/layout、AutoContiguous、ascend950注册 |
| Host/Tiling/Launcher | `foreach/foreach_atan/op_host/arch35/foreach_atan_tiling.cpp` | `ForeachAtanTilingFunc` | 平台参数、POD、C、workspace、local memory、key |
| Host/Tiling/Launcher | `foreach/foreach_atan/op_kernel/arch35/foreach_atan_tiling_data.h` | `ForeachAtanTilingData` | 2064B POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_atan/op_kernel/arch35/foreach_atan_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个schMode specialization |
| Kernel entry/body | `foreach/foreach_atan/op_kernel/arch35/foreach_atan.cpp` | `foreach_atan<schMode>` | entry、POD GM指针、dtype router |
| Kernel entry/body | `foreach/foreach_atan/op_kernel/arch35/foreach_atan_simt.h` | `Process`, `OpForeachAtanSimt*`, `SimtGetTensorAddr` | VF launch、线程映射、精度、tail |
| 构建接线 | `foreach/foreach_atan/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35与ACLNN接线 |
| 构建接线 | `foreach/foreach_atan/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_atan.cpp` | 950 Kernel target，AUTO_SYNC false |
| 构建接线 | `foreach/foreach_atan/op_host/config/ascend950/foreach_atan_binary.json` | `op_list` | 950 FP16/FP32/BF16 binaries |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064B，offset 0/4/8/16 | `tensorElements[256]`、8B对齐、GM pointer访问不变 | `ForeachAtanTilingData`, `Process` |
| 调用顺序 | 单K0→VF_CALL→Tensor loop→grid-stride loop | VF threads=1024；不是多Host-Kernel | `foreach_atan`, `Process` |
| workspace/内存 | workspace0；直接GM；launch local memory=M-128KiB | 不得虚构Queue/shared buffer；需保证减法不下溢 | Tiling local-memory设置，三个VF body |
| 同步 | 每元素唯一writer，无barrier/atomic | 输出仅在stream完成后Host可见 | grid-stride loop, AUTO_SYNC false |
| 精度与尾块 | FP32 atanf；FP16/BF16先升FP32再降；idx<count | 转换顺序、64位tid/stride和count guard不变 | `OpForeachAtanSimt*` |
| 失败边界 | >256、平台0值、unsupported dtype失败 | 不得补成fallback；空列表/溢出/rounding缺失不得推测 | `ForeachAtanTilingFunc` |