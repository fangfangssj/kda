---
type: Operator
title: "ForeachSigmoid 950 ACLNN 算子知识"
description: "说明 aclnnForeachSigmoid 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/docs/aclnnForeachSigmoid.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_host/foreach_sigmoid_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_host/arch35/foreach_sigmoid_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_host/config/ascend950/foreach_sigmoid_binary.json
    role: implementation
created_at: "2026-08-25T16:20:27Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSigmoid 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSigmoidGetWorkspaceSize` + `aclnnForeachSigmoid`，公开、无版本后缀的两段式 ACLNN 接口 |
| 计算语义 | 对 TensorList 中每个张量逐元素执行 `1/(1+exp(-x))` |
| 输入输出 | ND TensorList；FP16、FP32、BF16；输入可非连续、输出必须连续；rank 0～8；950 Tiling 最多 256 个 Tensor |
| 执行类型 | 单节点 `SIMT`；AICore 外层 block 内对每个 Tensor 发起 512-thread VF，所有可达节点均无 Cube/AIC 或 MIX_CV |
| Kernel 路线 | R-001 FP16 SIMT；R-002 FP32 SIMT；R-003 BF16 SIMT |
| 核心资源 | TensorList GM 地址表；SIMT 寄存器标量；无算子拥有的 AIV TQue/TBuf/shared/dynamic UB；16 MiB AscendC tools workspace 由框架持有但 Kernel 不读 |
| 来源闭包 | `source_closed`：公开 ACLNN 约束→框架生成两段式边界→950 OpDef/arch35 Tiling与POD→模板 key router→SIMT body→ascend950 构建/binary config 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对输入列表 `x=[x_0,...,x_{n-1}]` 的每个 storage element：

```text
y_i[j] = 1 / (1 + exp(-x_i[j])), 0 <= j < numel(x_i)
```

各 Tensor 依次发起 VF，但所有元素无数据依赖。输出允许比对应输入拥有更大的 shape size，Kernel 只写前 `numel(x_i)` 个元素，额外后缀不写。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 ND，rank 0～8 | FP16、FP32、BF16 | 非空指针；同列表 dtype 一致；单个空 Tensor 支持；OpDef `.AutoContiguous()` 处理非连续输入 |
| `out` | TensorList；每项 ND，rank 0～8 | 与 x 相同 | 非空指针；输出连续；同列表 dtype 一致；对应 shape size≥输入 |
| `workspaceSize` | 标量输出 | `uint64_t` | arch35 Tiling 固定返回 `16*1024*1024` B |
| `executor` | Host 对象 | `aclOpExecutor**` | 第一段生成，包含 launch 流程 |
| `workspace` | Device byte region | 无 | tools workspace；entry/body 参数保留但不解引用 |
| `stream` | runtime stream | `aclrtStream` | 第二段提交唯一 Kernel |

### 入口与版本

固定源码只有无版本后缀的公开 ACLNN 版本。自动生成的 ACLNN inner bridge/executor 不在算子目录，作为框架边界；算子拥有的 OpDef、Tiling、POD、entry、body 与构建均可见。公开 API 文档与 README 明确支持 Ascend 950PR/950DT；OpDef `AddConfig("ascend950")`、Host `arch35` 选择、Kernel CMake 与 ascend950 binary config 一致，无平台冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段按公开契约检查空指针、支持 dtype、输入输出 dtype、列表内 dtype、shape size 和 rank，并将非连续输入通过框架连续化；输出不允许非连续。
2. OpDef 对 x/y 声明动态 ND、FP16/FP32/BF16、动态 rank/shape，并为 ascend950 配置 `DynamicCompileStaticFlag(true)`、`PrecisionReduceFlag(true)` 等。
3. `ForeachSigmoidTilingFunc` 要求平台信息非空、AIV core 数大于 0、UB size 非 0；读取 x 动态实例数并限制为 `[0,256]`。
4. 非空列表以第 0 个输入描述得到 dtype；空列表默认 `DT_FLOAT`。Tiling 对每个输入 storage shape 取 `GetShapeSize()`，记录最大元素数和逐 Tensor 元素数；不支持 dtype 返回 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

令 `M=max_i numel(x_i)`、`C=GetCoreNumAiv()`：

```text
perCoreElements = ceilAlign(max(1024, ceil(M/C)), 32)  # 元素
needCoreNum = max(1, min(C, ceil(M/perCoreElements)))
workspaceSize = 16 MiB
localMemorySize = ubSize - 128 KiB
```

`SetBlockDim(needCoreNum)` 产生 SIMT grid 的 block 数。`needCoreNum` 也写入 POD，但 Device body 不读取该字段；真正的 block 数来自 `Simt::GetBlockNum()`。16 MiB workspace 由调用者/executor 拥有并覆盖 launch 生命周期，Kernel entry 没有消费。`SetLocalMemorySize` 声明每核可用 local-memory budget，但固定 SIMT body 没有声明 shared/dynamic UB 对象。

### 启动图与失败传播

```text
aclnnForeachSigmoidGetWorkspaceSize(x,out)
  -> ACLNN 校验/连续化框架边界
  -> ForeachSigmoidTilingFunc
     -> POD + template key + blockDim + localMemorySize + 16 MiB workspace
  -> executor
aclnnForeachSigmoid(workspace,16 MiB,executor,stream)
  -> foreach_sigmoid<schMode><<<needCoreNum blocks, stream>>>
     -> NsForeachSigmoid::Process<T>
        -> 对每个非空 Tensor 执行 VF_CALL<OpForeachSigmoidSimt<T>>(512 threads)
```

这是单 Kernel launch graph，无 Kernel 间 workspace 边。Tiling 任一空指针、平台资源、Tensor 数或 dtype 检查失败均不 launch；第二段/launch 错误沿 ACLNN status 返回。所有 Tensor 的 VF call 位于同一 Kernel 的源程序顺序中；它们写不同输出地址，不需要跨 Tensor barrier。

## Route 与参数设计

### 分核与资源计算

分核只看最大 Tensor `M`，使每个 Tensor 都复用相同 grid。每个 VF 固定 `THREAD_NUM=512`；线程到元素的映射为：

```text
globalThread = blockIdx * 512 + threadIdx
gridThreads = 512 * blockNum
idx = globalThread; idx < tensorElements[i]; idx += gridThreads
```

因此每个 Tensor 的所有元素被恰好一个 grid-stride 序列覆盖；短 Tensor 只激活前部线程，长 Tensor 每线程处理多个元素。`ALIGN_SIZE=32` 的单位是元素，仅用于 `perCoreElements`，不是 DMA 字节对齐。固定 body 直接 GM load/store，没有 tile、former/tail core、UB 搬运或 workspace slice。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 SIMT | `GET_TPL_TILING_KEY(0)`；`foreach_sigmoid<0>` | `tensorNum==0` 时默认 `DT_FLOAT`，否则首输入 dtype 为 FP16 时选择 mode 0 | 非 FP16 继续 dtype 分支 |
| 2 | R-002 FP32 SIMT | `GET_TPL_TILING_KEY(1)`；`foreach_sigmoid<1>` | 首输入 dtype FP32；空列表也因默认 `DT_FLOAT` 走 mode 1 | 非 FP32 继续 dtype 分支 |
| 3 | R-003 BF16 SIMT | `GET_TPL_TILING_KEY(2)`；`foreach_sigmoid<2>` | 首输入 dtype BF16 | 其它 dtype 返回 `GRAPH_FAILED` |

```text
tensorNum == 0 -> xDtype=DT_FLOAT -> mode 1 -> R-002（Process 循环 0 次）
else DT_FLOAT16 -> mode 0 -> R-001
else DT_FLOAT   -> mode 1 -> R-002
else DT_BF16    -> mode 2 -> R-003
else GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | mode 0 | FP16、ND、rank≤8、输出 size≥输入、tensorNum≤256、950 | 单 SIMT Kernel；每 Tensor 一次 512-thread VF | `Process<half>`→`OpForeachSigmoidSimt<half>`→`ComputeSigmoid<half>` | GM direct load/store；线程寄存器中 float32 中间量；无显式 shared/dynamic UB/barrier/atomic | `idx<count` 保护；count=0 跳过 |
| R-002 FP32 | mode 1 | FP32 或空 TensorList，其余同上 | 同上 | `Process<float>`/float body | 同上 | 同上 |
| R-003 BF16 | mode 2 | BF16，其余同上 | 同上 | `Process<bfloat16_t>`/BF16 body | GM 为 BF16，中间标量统一 FP32 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `0<=tensorNum<=256` | POD `tensorElements[256]` 固定容量 | 全部 | `GRAPH_FAILED` | `ForeachSigmoidTilingFunc` |
| 最小每核域 | `perCoreElements>=1024` 且向上对齐 32 元素 | 避免小 Tensor 启动过多 block；与 SIMT grid-stride 匹配 | 全部 | 不是 fallback，直接影响 blockDim | 同上 |
| DCACHE 保留 | `SetLocalMemorySize(uint32(ubSize-128KiB))` | Host 从 UB 总量扣除固定 DCACHE_SIZE | 全部 | `ubSize==0` 失败；未显式检查 `ubSize>=128KiB` | 同上 |
| 空 Tensor | `count==0` | 避免无效 VF call/GM 解引用 | 全部 | `continue` 到下一个 Tensor | `Process<T>` |
| 线程尾部 | `idx<count`，步长=`512*blockNum` | grid-stride 覆盖任意元素数 | 全部 | 超界线程不访存 | `OpForeachSigmoidSimt` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSigmoidTilingData` | `tensorNum` | `int32_t`/32 | offset 0；结构自然对齐 8 | Tensor 个数 | `tiling->tensorNum=tensorNum` | `Process` 的 Tensor 循环上界 | `[0,256]` |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | block/core 个数 | `tiling->needCoreNum=needCoreNum` | 固定 Device body不读取；launch 使用相同值设置 blockDim | `>=1`，`<=C` |
| 同上 | `tensorElements[256]` | `int64_t[256]` | offset 8；结构 `sizeof=2056`、`alignof=8`（由显式标准布局字段可确定） | 每 Tensor 元素数 | 循环写 `shape.GetShapeSize()` | `Process`: `count=tensorElements[i]` | 未活动项先由 `memset_s` 置 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | mode 0 | `arch35/foreach_sigmoid_tiling.cpp::ForeachSigmoidTilingFunc` | `ForeachSigmoidTilingData` | `arch35/foreach_sigmoid.cpp::foreach_sigmoid<0>` | `foreach_sigmoid_simt.h::Process<half>/OpForeachSigmoidSimt/ComputeSigmoid` | ascend950 Host/Kernel CMake + binary config |
| R-002 FP32 | mode 1 | 同上 | 同上 | `foreach_sigmoid<1>` | `Process<float>` 及同 helper | 同上 |
| R-003 BF16 | mode 2 | 同上 | 同上 | `foreach_sigmoid<2>` | `Process<bfloat16_t>` 及同 helper | 同上 |

## Kernel 实现

### 公共执行流程

entry 使用编译期 `schMode` 路由到唯一 dtype template。`Process<T>` 构造 `ListTensorDesc`，顺序遍历 `tensorNum`；空 Tensor 直接跳过，非空 Tensor 解析 x/y 第 i 项数据地址并发起 `VF_CALL`，线程块大小恒 512。VF 每线程以 `blockIdx*512+threadIdx` 为首元素，以整个 grid 的线程数为步长直接读写 GM。`ComputeSigmoid` 无论 T 为 half、float 还是 BF16，都先转 float，按 `-x -> expf -> +1 -> reciprocal` 计算，再 cast 回 T。

### Route R-001：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnForeachSigmoid.md` | 两段式 API | FP16、ND、shape/dtype/rank 合法 | executor 或错误 |
| Tiling | `op_host/arch35/foreach_sigmoid_tiling.cpp` | `ForeachSigmoidTilingFunc` | `xDtype==DT_FLOAT16` | template mode 0 |
| Router | `op_kernel/arch35/foreach_sigmoid.cpp` | `foreach_sigmoid<schMode>` | `schMode==TILING_KEY_FLOAT16(0)` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sigmoid<0>(xList,yList,workspace,tiling)
  -> Process<half>
     -> ListTensorDesc::GetDataPtr<half>(i)
     -> VF_CALL<OpForeachSigmoidSimt<half>>(Dim3(512),xPtr,yPtr,count)
        -> ComputeSigmoid<half>
```

##### 源码等价伪码

```text
# 源码等价伪码
for i=0..tensorNum-1:
  count=tensorElements[i]; if count==0: continue
  xPtr=xList.data<half>(i); yPtr=yList.data<half>(i)
  for idx=blockIdx*512+threadIdx; idx<count; idx+=blockNum*512:
    f=float(xPtr[idx]); yPtr[idx]=half(1.0f/(1.0f+expf(-f)))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList x GM | 地址表第 i 项 + `idx` | 1 half/线程/迭代 | half/ND | 调用者 | 当前 SIMT thread | 调用者，launch 有效 | 无显式 DMA 对齐要求 |
| 线程寄存器 | thread-private `val/fval/negVal/expVal/denom/result` | 1 标量各；确切寄存器字节固定仓不可见 | half + float32 | 当前线程 | 当前线程 | 单次循环迭代 | 无跨线程共享 |
| TensorList y GM | 地址表第 i 项 + `idx` | 1 half/线程/迭代 | half/ND | 当前线程 | 调用者后续任务 | 调用者 | `idx<count` 防越界 |
| tools workspace | base 0 | 16 MiB | bytes | executor | Kernel 不消费 | 调用者/executor 至 stream 完成 | 无算子 slice |

##### 分层内存布局

N/A：该 SIMT 节点没有算子拥有的 AIV UB Queue/TBuf，也未声明 shared/dynamic UB；数据经线程标量寄存器从 x GM 直接到 y GM。Host 的 local-memory budget 为 `ubSize-128 KiB`，固定 body 未显式消费。

##### 字节账与复用顺序

```text
UB_total(operator-owned explicit) = 0 B
dynamic_ub = 0 B（launch 未生产 dynUBufSize，body 无 shared/dynamic UB）
workspace_total = 16 MiB tools reserve; operator body consumed = 0 B
register_bytes = 固定仓不可见，仅能确认每线程标量生命周期
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程局部标量 | `ComputeSigmoid` 当前迭代 | 同线程表达式后继 | 下一 grid-stride 迭代 | 程序顺序 | 不跨线程/不跨迭代暴露 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM load→compute→store | 单线程程序顺序 | `xPtr[idx]` | `ComputeSigmoid`/`yPtr[idx]` | 单元素 | half 先 cast FP32，`expf` 等均 FP32，结果 cast half |
| grid tail | `idx<count` | grid-stride index | load/store guard | Tensor 有效域 | 超界线程不访问；无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_sigmoid/op_host/arch35/foreach_sigmoid_tiling.cpp` | `ForeachSigmoidTilingFunc` | mode 0、blockDim、POD/workspace |
| Kernel entry/router | `foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid.cpp` | `foreach_sigmoid<0>` | half route |
| Device body/helper | `foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_simt.h` | `Process<half>`、`OpForeachSigmoidSimt<half>`、`ComputeSigmoid<half>` | Tensor loop、VF、grid-stride、FP32 opmath |

### Route R-002：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnForeachSigmoid.md` | 两段式 API | FP32 或空列表，契约合法 | executor |
| Tiling | `ForeachSigmoidTilingFunc` | dtype selector | `xDtype==DT_FLOAT`；空列表默认 float | mode 1 |
| Router | `foreach_sigmoid<schMode>` | constexpr branch | mode 1 | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sigmoid<1> -> Process<float>
  -> per nonempty tensor VF_CALL<OpForeachSigmoidSimt<float>>(512)
     -> ComputeSigmoid<float>
```

##### 源码等价伪码

```text
# 与 R-001 相同；T=float，最终 cast 回 float
for idx=globalThread; idx<count; idx+=512*blockNum:
  y[idx]=float(1.0f/(1.0f+expf(-float(x[idx]))))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | Tensor i base + idx | 1 float/线程/迭代 | float/ND | 调用者/线程 | 线程/调用者 | launch | `idx<count` |
| 寄存器 | thread-private | 若干 float 标量 | float | 当前线程 | 当前线程 | 单次迭代 | 程序顺序复用 |
| workspace | 0..16 MiB | 16 MiB reserve | bytes | executor | 未消费 | stream 生命周期 | 无 slice |

##### 分层内存布局

N/A：与 R-001 相同，只有 GM 与线程标量寄存器。

##### 字节账与复用顺序

```text
explicit UB=0 B; dynamic_ub=0 B; workspace reserve=16 MiB/body use=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local scalars | 当前迭代 | 当前线程 | 下一迭代 | 程序顺序 | 无跨线程别名 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全流水 | thread program order | GM load | expf/加法/倒数/GM store | 单元素 | 全程 FP32；`idx<count` 是唯一尾保护 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sigmoid_tiling.cpp` | `ForeachSigmoidTilingFunc` | mode 1 与空列表默认 route |
| Kernel entry/router | `op_kernel/arch35/foreach_sigmoid.cpp` | `foreach_sigmoid<1>` | float route |
| Device body/helper | `op_kernel/arch35/foreach_sigmoid_simt.h` | float instantiations | direct GM grid-stride |

### Route R-003：BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `docs/aclnnForeachSigmoid.md` | 两段式 API | BF16 且契约合法 | executor |
| Tiling | `ForeachSigmoidTilingFunc` | dtype selector | `xDtype==DT_BF16` | mode 2 |
| Router | `foreach_sigmoid<schMode>` | constexpr branch | mode 2 | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sigmoid<2> -> Process<bfloat16_t>
  -> per nonempty tensor VF_CALL<OpForeachSigmoidSimt<bfloat16_t>>(512)
     -> ComputeSigmoid<bfloat16_t>
```

##### 源码等价伪码

```text
# 源码等价伪码
for idx=globalThread; idx<count; idx+=512*blockNum:
  f=float(x_bf16[idx])
  r=1.0f/(1.0f+expf(-f))
  y_bf16[idx]=bfloat16_t(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | Tensor i base + idx | 1 BF16/线程/迭代 | BF16/ND | 调用者/线程 | 线程/调用者 | launch | `idx<count` |
| 寄存器 | thread-private | BF16 val + FP32 intermediates | scalar | 当前线程 | 当前线程 | 单次迭代 | 程序顺序复用 |
| workspace | 0..16 MiB | 16 MiB reserve | bytes | executor | 未消费 | stream 生命周期 | 无 slice |

##### 分层内存布局

N/A：无 AIV UB、shared/dynamic UB；BF16 仅在每线程寄存器中升为 FP32。

##### 字节账与复用顺序

```text
explicit UB=0 B; dynamic_ub=0 B; workspace reserve=16 MiB/body use=0 B
register bytes 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 thread scalars | BF16 cast/算术 | final BF16 cast | 下一 idx | 程序顺序 | thread-private |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32→BF16 | C++ scalar cast + `expf` | GM load | GM store | 单元素 | 核心算术 FP32；回 BF16 的具体舍入语义由编译器/类型实现决定，固定仓未显式命名 round mode |
| tail | `idx<count` | index | load/store | 有效域 | count 0 在 VF 前跳过 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sigmoid_tiling.cpp` | `ForeachSigmoidTilingFunc` | mode 2、POD/grid |
| Kernel entry/router | `op_kernel/arch35/foreach_sigmoid.cpp` | `foreach_sigmoid<2>` | BF16 route |
| Device body/helper | `op_kernel/arch35/foreach_sigmoid_simt.h` | BF16 instantiations | thread-local FP32 opmath |

## 算子特定优化

### Trick T-001：一个固定 grid 以 grid-stride 循环复用于整个 TensorList

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：Tensor 非空；blockDim 由最大 Tensor 计算，VF 固定 512 threads。
- Golden 来源：[`foreach_sigmoid_simt.h:53-64`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_simt.h#L53-L64)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint64_t idx = static_cast<uint64_t>(
         Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
     idx < static_cast<uint64_t>(elemCount);
     idx += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
    T val = xPtr[idx];
    yPtr[idx] = ComputeSigmoid(val);
}
```

#### 机制与设计原因

Host 只根据最大 Tensor 选择一次 blockDim；每个 Tensor 复用该 grid，线程用总 grid 线程数作为步长。它无需为不同 Tensor 生成 per-tensor tiling 或 UB tile 表，同时 `idx<elemCount` 自动适配短 Tensor 与尾部。该结论仅说明源码减少参数和分支物化，不给出未实测性能幅度。

#### 迁移边界

- 适用：TensorList 上无跨元素依赖、每元素成本近似固定的 pointwise SIMT 算子。
- 前提：每个 Tensor 均能由同一 dtype template 处理，地址由 `ListTensorDesc` 安全取得。
- 失效：需要 per-tensor 不同 grid/shared memory、跨元素同步或 reduction 时需独立 route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP16/BF16 统一用线程局部 FP32 opmath

#### 触发与源码依据

- 适用 Route：R-001、R-003（R-002 也经过相同 float 表达式但无升精度）
- 触发条件：`ComputeSigmoid<T>` 的 T 为 half 或 bfloat16_t。
- Golden 来源：[`foreach_sigmoid_simt.h:39-47`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_simt.h#L39-L47)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
float fval = static_cast<float>(val);
float negVal = -fval;
float expVal = expf(negVal);
float denom = 1.0f + expVal;
float result = 1.0f / denom;
return static_cast<T>(result);
```

#### 机制与设计原因

每线程只对一个标量升精度，不创建 GM/UB 中间 Tensor；指数、加法和倒数统一用 FP32，最后一次 cast 返回目标 dtype。这同时闭合了精度顺序和中间值生命周期，并避免 BF16/FP16 的整 tile FP32 scratch 物化。

#### 迁移边界

- 适用：可逐元素独立升精度的 transcendental 函数。
- 前提：目标平台提供 `expf`，线程寄存器资源可容纳这些标量，最终 cast 误差符合 API 预期。
- 失效：若必须规定特定 BF16/FP16 round mode，需显式 primitive；这里的 cast 未在源码中命名 round mode。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | mode 0 | mode 1；空列表默认 | mode 2 | dtype 单轴分发 |
| 模板与 dtype | `Process<half>` | `Process<float>` | `Process<bfloat16_t>` | GM 指针类型不同 |
| 循环与分核 | 512-thread grid-stride | 同左 | 同左 | body 完全同构 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread-local FP32 scalars | thread-local FP32 scalars | thread-local FP32 scalars | 无显式 UB/shared memory |
| 内存总占用与复用 | explicit UB 0 | explicit UB 0 | explicit UB 0 | workspace 均不消费 |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM + SIMT `expf` | 同左 | 同左 | 无 DMA staging |
| 同步、精度与 tail | FP32 opmath→half；idx guard | FP32；idx guard | FP32 opmath→BF16；idx guard | 仅最终 cast 不同 |
| fallback/失败边界 | 非命中继续 selector | 空列表 route | 其它 dtype 失败 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；见无后缀卡 | 公开支持；见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 common tiling | arch22 common tiling | arch35 专用 tiling；256 Tensor、16 MiB tools workspace |
| Kernel/资源 | AIV Vector/TQue/UB tile | AIV Vector/TQue/UB tile | SIMT direct GM、512 threads、无显式 UB |

## 限制与失败边界

- 950 算子 Tiling 最多编码 256 个 Tensor；超过直接 `GRAPH_FAILED`。公开文档未单独声明该数值上限。
- `ForeachSigmoidTilingFunc` 只读取首输入 dtype，列表内 dtype、输出 dtype/shape/rank 等依赖 ACLNN/框架契约；若绕过公开入口，固定 950 Tiling 不重复这些校验。
- `SetLocalMemorySize(uint32_t(ubSize-128KiB))` 未见 `ubSize>=128KiB` 的显式 guard；仅检查 `ubSize!=0`，不能外推下溢平台的行为。
- `needCoreNum` POD 字段在固定 Device body 中未读取；真正的并行规模由 launch blockDim 和 SIMT intrinsic 获得。
- TensorList 描述符布局与 `ListTensorDesc` 内部地址解码是 AscendC framework boundary；SIMT VF_CALL 的内部调度/隐式同步也不在固定仓展开。
- BF16/FP16 最终 `static_cast<T>` 的确切 round mode 固定源码未显式给出。未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sigmoid/docs/aclnnForeachSigmoid.md` | 两段式 ACLNN | 参数、错误、公开 950 支持 |
| 入口声明/约束 | `foreach/foreach_sigmoid/README.md` | 产品支持/功能/参数 | 950PR/DT 与三 dtype |
| 入口实现 | `foreach/foreach_sigmoid/op_host/foreach_sigmoid_def.cpp` | `ForeachSigmoid` | ascend950 OpDef、dtype/layout/连续化 |
| Host/Tiling/Launcher | `foreach/foreach_sigmoid/op_host/arch35/foreach_sigmoid_tiling.cpp` | `ForeachSigmoidTilingFunc` | selector、POD、blockDim、workspace/local memory |
| Host/Tiling/Launcher | `foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_tiling_data.h` | `ForeachSigmoidTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_tiling_key.h` | template args/select | mode 候选与 key 生成 |
| Kernel entry/body | `foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid.cpp` | `foreach_sigmoid<schMode>` | constexpr dtype router |
| Kernel entry/body | `foreach/foreach_sigmoid/op_kernel/arch35/foreach_sigmoid_simt.h` | `Process`、`OpForeachSigmoidSimt`、`ComputeSigmoid` | 完整 SIMT body |
| 构建接线 | `foreach/foreach_sigmoid/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35 Host |
| 构建接线 | `foreach/foreach_sigmoid/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_sigmoid.cpp` | ascend950 Kernel source |
| 构建接线 | `foreach/foreach_sigmoid/op_host/config/ascend950/foreach_sigmoid_binary.json` | 三个 binary entry | 950 FP16/FP32/BF16 binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | tensorNum + coreNum + 256 个元素计数 | 字段顺序、位宽、256 上限；inactive 元素清零 | `ForeachSigmoidTilingData` |
| 调用顺序 | 单 Kernel 内依次遍历 Tensor、每项 VF_CALL | 空 Tensor 必须在取地址/VF 前跳过 | `Process<T>` |
| workspace/内存 | 直接 GM，显式 UB/dynamic UB 为 0 | 16 MiB tools reserve 与 local-memory 设置不得误当中间 Tensor | `ForeachSigmoidTilingFunc`、entry |
| 同步 | thread-private 程序顺序；无 atomic/barrier | 不能引入跨线程共享状态而仍沿用此协议 | `OpForeachSigmoidSimt` |
| 精度与尾块 | FP32 opmath；grid-stride `idx<count` | `expf` 顺序、最终 cast 和 count guard | `ComputeSigmoid` |
| 失败边界 | 平台/core/UB/Tensor 数/dtype 失败不 launch | 不把首 dtype 选择当完整列表校验 | `GetPlatformInfo`、`ForeachSigmoidTilingFunc` |

- [数学函数expf](../../../../apis/simt_api/math_functions/float_math_functions/expf.md)
