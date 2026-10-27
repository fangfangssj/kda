---
type: Operator
title: "ForeachExp 950 ACLNN 算子知识"
description: "说明 aclnnForeachExp 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 SIMT 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/docs/aclnnForeachExp.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_host/foreach_exp_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_host/arch35/foreach_exp_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_kernel/arch35/foreach_exp_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_kernel/arch35/foreach_exp_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_kernel/arch35/foreach_exp.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_host/config/ascend950/foreach_exp_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:32:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachExp 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnForeachExpGetWorkspaceSize` → `aclnnForeachExp`；固定仓的生成式 inner bridge 不可见，实际可见 Host 边界为 `ForeachExp` OpDef。 |
| 计算语义 | 对 TensorList 每项逐元素计算 `y_i[j]=exp(x_i[j])`。 |
| 输入输出 | `x`/`out`：ND TensorList，FLOAT16/FLOAT32/BFLOAT16，每 Tensor 0–8 维；输入可由框架连续化，输出必须连续。 |
| 执行类型 | 单 Kernel：SIMT。 |
| Kernel 路线 | R-001 FLOAT32 直接 `expf`；R-002 FLOAT16/BFLOAT16 升到 FLOAT32 `expf` 后转回原 dtype。 |
| 核心资源 | TensorList GM 指针表、2064 B Tiling POD、每线程标量/寄存器；无显式 AIV Queue/TBuf、shared buffer、Atomic 或 Kernel workspace；Host 设置 `localMemorySize=UB-128 KiB`。 |
| 来源闭包 | `source_closed`：公开 ACLNN → 生成式框架边界 → 950 OpDef/arch35 Tiling/POD/template selector → templated `foreach_exp` → `Process<T>`/SIMT VF body → ascend950 build/config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `x=[x_0,...,x_(n-1)]`，输出 `y_i[j]=e^(x_i[j])`。单次 Kernel 启动形成 `needCoreNum` 个 SIMT block，每 block 1024 threads；每个线程对每个 Tensor 使用全 grid-stride 遍历。FLOAT32 值直接传给 `expf`；FLOAT16/BFLOAT16 通过显式 specialization 转为 FLOAT32，调用 `expf`，再 `static_cast` 回原 dtype。只写输入元素数覆盖的输出前缀。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 动态 TensorList；每项 ND、0–8 维 | FLOAT16、FLOAT32、BFLOAT16，列表内一致 | 支持空 Tensor；公开接口允许非连续输入，OpDef 使用 `AutoContiguous()`。950 Tiling 最多接受 256 个 Tensor。 |
| `out` | 动态 TensorList；每项 ND、0–8 维，shape size≥对应输入 | 与输入相同，列表内一致 | 支持空 Tensor；公开接口不支持非连续输出。Kernel 只写 `tensorElements[t]` 个元素。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | 第一段返回完整 Executor workspace。算子 Tiling slot 为 0 B；框架输入连续化所需空间公式在固定仓不可见。 |
| `executor` | Host 句柄输出 | `aclOpExecutor**` | 第一段生产，第二段消费。 |
| `workspace` | Device 地址 | `void*` | 传入第二段；950 Kernel entry 不读取该参数。 |
| `stream` | Runtime stream | `aclrtStream` | Executor 在指定 stream 提交前处理与单个 SIMT Kernel。 |

### 入口与版本

固定提交公开入口没有 V2 等新版本，最新实际版本即 `aclnnForeachExpGetWorkspaceSize/aclnnForeachExp`。公开产品表明确支持 Ascend 950PR/950DT；OpDef 添加 `ascend950`，Host CMake 映射到 arch35，binary config 含三种 dtype，Kernel CMake 对 ascend950 选择 `arch35/foreach_exp.cpp`，无发布冲突。ACLNN inner bridge、Executor 与 `AutoContiguous` 实现由框架生成，是固定仓允许的外部边界。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开第一段接口检查空指针、支持 dtype、输入输出 dtype 一致、列表内 dtype 一致、shape 约束和维度≤8；失败返回文档定义的 ACLNN 参数或 Tiling 错误。
2. `ForeachExp` OpDef 声明动态 ND 输入/输出、FLOAT16/FLOAT32/BFLOAT16 与 `AutoContiguous`；输出连续性由公开接口约束。
3. arch35 `ForeachExpTilingFunc` 读取 AIV core 数与 UB；任一为 0 失败。它取动态输入实例数 `tensorNum` 和第 0 个输入描述的 dtype。
4. POD 先整体 `memset_s(...,0)`；`tensorNum>256` 失败；随后枚举全部输入 storage shape，填 `tensorElements[i]` 与 `totalElements`。
5. 生成式 bridge 的逐节点 Contiguous、Executor cleanup 与 ACLNN→GE 返回码细节固定仓不可见。

### 参数、Workspace 与生命周期

令总元素数 `N=sum_i tensorElements[i]`，平台 AIV core 数为 `P`。当 `N=0`，`needCoreNum=blockDim=1`；否则 `needCoreNum=blockDim=min(P,ceil(N/1024))`。Tiling workspace 为 0 B。Host 把 `localMemorySize` 设置为 `UB_size-128 KiB`，其中 128 KiB 常量名为 `DCACHE_SIZE`；固定 Kernel 未显式声明 shared/dynamic UB buffer，因此这只是 launcher 的本地内存预算，实际 runtime 映射在固定仓不可见。POD、TensorList 指针表、输入和输出 GM 必须在 stream 中 Kernel 完成前有效。

### 启动图与失败传播

```text
aclnnForeachExpGetWorkspaceSize(x,out)
  -> [生成式 ACLNN 校验/Contiguous/Executor：固定仓不可见]
  -> ForeachExpTilingFunc
     -> GetPlatformInfo(AIV cores, UB)
     -> fill ForeachExpTilingData
     -> SetBlockDim(needCoreNum)
     -> SetLocalMemorySize(UB-128KiB)
     -> SetTilingKey(GET_TPL_TILING_KEY(schMode))
aclnnForeachExp(...,stream)
  -> K-001 foreach_exp<schMode><<<needCoreNum blocks, 1024 threads/block, stream>>>
     -> VF_CALL<OpForeachExpSimt<T>>(Dim3(1024), ...)
```

唯一节点 K-001 输入 `xList`、输出 `yList`、POD；workspace 参数未读取，没有跨 Kernel 边。Executor 在同一 stream 的前处理必须先于 K-001；Tiling 的 platform/null/tensorNum/dtype 检查失败会阻断 launch。SIMT body 不使用 event/barrier/atomic，各线程写互不重叠的 `idx=tid+k*stride`。

## Route 与参数设计

### 分核与资源计算

`MIN_PER_CORE=1024` 元素，故 `blockDim=1`（`N=0`）或 `min(P,ceil(N/1024))`。每 block 固定 `THREAD_NUM=1024`，全网格线程数 `G=blockDim*1024`。对每个 Tensor `t`，线程线性 id 为 `tid=blockIdx*1024+threadIdx`，处理 `idx=tid,tid+G,...<tensorElements[t]`。这不是预先把不同 Tensor 分给不同 block：所有线程网格逐 Tensor 重用，元素索引用 guard 限定。

`ForeachExpTilingData` 字段按自然 8 B 对齐：offset 0/4/8/16，总大小 `4+4+8+256*8=2064 B`。Kernel 无显式 AIV UB 分配、无 workspace slice、无 shared buffer；每线程活跃标量为 `xVal:T`、`xFloat:float`、`yFloat:float` 及索引变量，精确寄存器字节数由编译器决定，固定仓不可见。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FLOAT32 直接 SIMT Exp | `GET_TPL_TILING_KEY(FOREACH_EXP_TPL_SCH_MODE_FLOAT=0)`；`foreach_exp<0>` | `dataType==DT_FLOAT` | 否则依次检查 FLOAT16、BF16 |
| 2 | R-002 低精度升精度 SIMT Exp | schMode 1：FLOAT16；schMode 2：BF16；`foreach_exp<1/2>` | `dataType==DT_FLOAT16`，否则 `dataType==DT_BF16` | 其它 dtype：Tiling 记录 error 并 `GRAPH_FAILED` |

```text
dtype FLOAT32 -> schMode 0 -> foreach_exp<0> -> Process<float> -> R-001
else dtype FLOAT16 -> schMode 1 -> foreach_exp<1> -> Process<half> -> R-002
else dtype BFLOAT16 -> schMode 2 -> foreach_exp<2> -> Process<bfloat16_t> -> R-002
else -> GRAPH_FAILED, no launch
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FLOAT32 | schMode 0 | 950；FLOAT32；ND；Tensor count≤256；每项≤8维；输出约束通过 | 单 SIMT K-001；`B=needCoreNum`,`T=1024` | `foreach_exp<0>`→`Process<float>`→`OpForeachExpSimt<float>` | GM load→线程标量→`expf`→GM store；无 barrier/atomic/shared | `idx<count` grid-stride guard；空 Tensor `continue`；非法 dtype Host 失败 |
| R-002 FP16/BF16 | schMode 1/2 | 同上但 FLOAT16/BFLOAT16 | 单 SIMT K-001；相同网格 | `foreach_exp<1/2>`→`Process<half/bfloat16_t>`→相应 VF | GM→T 标量→FLOAT32→`expf`→T→GM；无显式共享内存/同步 | 同；cast 行为由 dtype specialization 区分 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台资源 | `coreNumAiv!=0 && ubSize!=0` | launch grid 和 localMemorySize 需要有效平台值 | 两 Route | `GRAPH_FAILED` | `GetPlatformInfo` |
| Tensor 数 | `tensorNum<=256` | POD `tensorElements[256]` 固定容量 | 两 Route | `GRAPH_FAILED` | `ForeachExpTilingFunc`; `ForeachExpTilingData` |
| 每核最小工作量 | `needCoreNum=min(P,ceil(N/1024))` | 以约 1024 总元素/core 限制 block 数 | 两 Route | `N=0` 特判 1 block | `ForeachExpTilingFunc` |
| 线程上限 | `THREAD_NUM=1024`; `LAUNCH_BOUND(1024)` | VF_CALL 与编译 launch bound 必须一致 | 两 Route | 无 runtime fallback | `OpForeachExpSimt`; `Process` |
| 本地内存预算 | `localMemorySize=ubSize-128*1024` | 为 DCACHE 保留 128 KiB；SIMT Kernel 本身不显式消费 buffer | 两 Route | `ubSize==0` 失败；固定源码未检查 `ubSize<128KiB` 下的无符号下溢 | `ForeachExpTilingFunc::SetLocalMemorySize` |
| 元素 tail | `idx<count` | 防止最后 grid-stride 超出 Tensor | 两 Route | 线程无工作，非错误 | `OpForeachExpSimt` |
| dtype selector | FLOAT→FP16→BF16 的 `if/else if` | 只构建 3 个 schMode 实例 | 两 Route | 其它 dtype `GRAPH_FAILED` | `ForeachExpTilingFunc`; `foreach_exp_tiling_key.h` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachExpTilingData` | `needCoreNum` | `int32_t` / 32 | sizeof=2064 B，alignof=8，offsetof=0 | block/core 数 | `needCoreNum` 或 1；同时 `SetBlockDim` | body 未读取；launch blockDim 使用同值 | `1..platform AIV cores` |
| `ForeachExpTilingData` | `tensorCount` | `int32_t` / 32 | offsetof=4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process` 读取并传给 VF；VF 外层 Tensor loop | ABI 可表达 0..256；Tiling 在此前要求第 0 个动态输入描述存在，实际可达为 1..256 |
| `ForeachExpTilingData` | `totalElements` | `int64_t` / 64 | offsetof=8 | 元素 | shape numel 累加 | body 未读取；Host 用来算核数 | 固定源码未显式检查累加溢出 |
| `ForeachExpTilingData` | `tensorElements` | `int64_t[256]` | offsetof=16，extent=2048 B | 元素/每 Tensor | `tensorElements[i]=storageShape.GetShapeSize()`；预先清零 | GM offset 指针传入 VF，逐 `t` 读 `count` | inactive slots 为 0；有效 `i<tensorCount` |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `GET_TPL_TILING_KEY(0)` | `op_host/arch35/foreach_exp_tiling.cpp::ForeachExpTilingFunc` | `ForeachExpTilingData`; blockDim/localMemorySize | `op_kernel/arch35/foreach_exp.cpp::foreach_exp<0>` | `foreach_exp_simt.h::Process<float>/OpForeachExpSimt<float>` | `op_kernel/CMakeLists.txt` ascend950 source；ascend950 binary json FLOAT32 entry |
| R-002 | `GET_TPL_TILING_KEY(1/2)` | 同上 | 同 POD | `foreach_exp<1/2>` | `Process<half/bfloat16_t>`；`PromoteToFloat/CastFromFloat` specializations；SIMT VF | 同 build；binary json 含 FLOAT16/BFLOAT16 |

## Kernel 实现

### 公共执行流程

arch35 entry 是 `template<uint32_t schMode>` 的 `__aicore__` Kernel。构建期模板 selector 只生成 schMode 0/1/2；entry 以 `if constexpr` 映射到 `Process<T>`。`Process` 不复制整个 POD到栈，而将 tiling GM 地址强转为 `const __gm__ ForeachExpTilingData*`，再通过 `offsetof` 取得 `tensorElements` 数组 GM 指针，读取 `tensorCount`，以 `VF_CALL(...Dim3(1024))` 启动 SIMT VF。VF 的所有线程依次遍历所有 Tensor，各自用唯一 grid-stride 索引直接读写 GM，无跨线程通信和跨 Kernel workspace。

### Route R-001：FLOAT32 直接 SIMT Exp

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `foreach/foreach_exp/docs/aclnnForeachExp.md` | 两段式入口 | 950、FLOAT32、ND、0–8维 | ForeachExp Executor |
| Tiling | `foreach/foreach_exp/op_host/arch35/foreach_exp_tiling.cpp` | `ForeachExpTilingFunc` | `dataType==ge::DT_FLOAT` | `GET_TPL_TILING_KEY(schMode=0)` |
| Router | `foreach/foreach_exp/op_kernel/arch35/foreach_exp.cpp` | `foreach_exp<schMode>` | `if constexpr(schMode==TILING_KEY_FLOAT)` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_exp<0>(xList,yList,workspace,tiling)
  -> NsForeachExp::Process<float>
     -> elemCounts = tiling + offsetof(tensorElements)
     -> VF_CALL<OpForeachExpSimt<float>>(Dim3(1024),tensorCount,elemCounts,xList,yList)
        -> for each Tensor: SimtGetTensorAddr<float>(x/y)
           -> grid-stride loop -> PromoteToFloat<float> -> expf -> CastFromFloat<float> -> GM store
```

##### 源码等价伪码

```text
# 源码等价伪码
G = GetThreadNum() * GetBlockNum()      # 1024 * needCoreNum
tid = GetBlockIdx()*GetThreadNum() + GetThreadIdx()
for t in [0, tensorCount):
    count = tensorElements[t]
    if count == 0: continue
    xData = tensor_list_addr<float>(xList,t)
    yData = tensor_list_addr<float>(yList,t)
    for idx = tid; idx < count; idx += G:
        xFloat = xData[idx]
        yFloat = expf(xFloat)
        yData[idx] = yFloat
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x TensorList pointer table | `dataAddr + (*dataAddr >> 3)`, slot `t` | `tensorCount` addresses | uint64 table | framework/caller | `SimtGetTensorAddr` | caller，Kernel 完成前有效 | 8 B 地址索引 |
| x Tensor GM | `xData[idx]`, `idx=tid+kG` | `tensorElements[t]` | FLOAT32/ND flattened | caller | SIMT thread | caller | guard `idx<count` |
| per-thread values | compiler-selected registers/local scalar | `xVal`, `xFloat`, `yFloat`, indices | FLOAT32/integers | GM load/arithmetic | expf/GM store | 当前 loop iteration | 精确寄存器字节固定仓不可见 |
| y Tensor GM | `yData[idx]` | 输入有效元素数 | FLOAT32/ND flattened | SIMT thread | caller/downstream | caller | 不同 thread 的 idx 不重叠 |
| Host local memory budget | runtime-defined | `UB_size-131072` B | launcher local memory | Tiling | runtime/SIMT launch | 单 Kernel | Kernel 无显式 shared/dynamic buffer consumer |
| workspace | N/A | 0 B | N/A | N/A | entry 不读取 | N/A | 无 workspace edge |

##### 分层内存布局

N/A：该 SIMT Kernel 未使用 AIV `TQue/TBuf/InitBuffer`，也未声明 shared/dynamic UB 数组。实际替代路径是 GM 标量 load → 每线程标量/寄存器 → GM 标量 store；Host 仅设置 `localMemorySize=UB-128 KiB` 的 launch 预算。

##### 字节账与复用顺序

```text
UB_total(AIV queues/tbuf) = 0 B
dynamic/local_memory_budget = platform_UB - 128*1024 B
explicit_shared_buffer = 0 B
per_thread_register_bytes = 固定仓不可见（由编译器决定）
workspace_total = 0 B
grid_threads = needCoreNum * 1024
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程 `xVal/xFloat/yFloat` 标量 | GM load / expf | expf / GM store | 同线程下一 `idx` | 单线程程序顺序；无跨线程共享 | 每个 `idx` 只归属一个线性 tid 模 G |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 线程索引 | grid-stride | launch grid | GM load/store | 唯一元素 idx | `idx<count` 保护尾部 |
| 计算 | `expf(float)` | FLOAT32 GM load | FLOAT32 GM store | 当前线程标量 | `PromoteToFloat<float>`/`CastFromFloat<float>` 均为 identity |
| 跨线程/跨 block | N/A | 各线程独立 | N/A | 无共享写地址 | 不需要 barrier/atomic |
| 空 Tensor | `if(count==0) continue` | POD count | 下一 Tensor | 空域 | 不解引用 Tensor data address |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_exp/op_host/arch35/foreach_exp_tiling.cpp` | `ForeachExpTilingFunc` | blockDim、localMemorySize、POD 与 schMode 0 |
| Kernel entry/router | `foreach/foreach_exp/op_kernel/arch35/foreach_exp.cpp` | `foreach_exp<0>` | compile-time dtype mapping |
| Device body/helper | `foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h` | `Process<float>`; `OpForeachExpSimt<float>` | 1024-thread VF、Tensor loop、grid-stride、expf 与 direct GM store |

### Route R-002：FLOAT16/BFLOAT16 升精度 SIMT Exp

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `foreach/foreach_exp/docs/aclnnForeachExp.md` | 两段式入口 | 950、FLOAT16/BFLOAT16 | ForeachExp Executor |
| Tiling | `foreach/foreach_exp/op_host/arch35/foreach_exp_tiling.cpp` | `ForeachExpTilingFunc` | FLOAT16 else-if BF16 | schMode 1 或 2 |
| Router | `foreach/foreach_exp/op_kernel/arch35/foreach_exp.cpp` | `foreach_exp<1/2>` | compile-time schMode | `Process<half>` / `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_exp<1/2> -> Process<half/bfloat16_t>
  -> VF_CALL<OpForeachExpSimt<T>>(1024 threads,...)
     -> SimtGetTensorAddr<T>
     -> xVal=T GM load
     -> PromoteToFloat<T>(xVal)
     -> expf(float)
     -> CastFromFloat<T>(float)
     -> yData[idx] GM store
```

##### 源码等价伪码

```text
# 源码等价伪码；Tensor 与 grid-stride 索引同 R-001
for idx = tid; idx < tensorElements[t]; idx += G:
    xLow = xData[idx]
    x32 = static_cast<float>(xLow)
    y32 = expf(x32)
    yData[idx] = static_cast<T>(y32)  # T=half or bfloat16_t
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList/GM | 同 R-001 | `tensorElements[t]` | FP16 或 BF16/ND | caller | SIMT | caller | `idx<count`；地址表 8 B slot |
| per-thread low value | 当前 `idx` | 1 元素 | T | GM load | `PromoteToFloat<T>` | loop iteration | 下一 idx 覆盖 |
| per-thread FLOAT32 | 当前 `idx` | `xFloat`,`yFloat` | FLOAT32 | cast/expf | expf/downcast | loop iteration | 无共享 materialization |
| y GM | `yData[idx]` | 输入有效元素数 | T | `CastFromFloat<T>` 后 store | caller | caller | threads 写地址互斥 |
| Host local memory/workspace | 同 R-001 | budget=`U-128KiB`; workspace=0 | runtime | Tiling | launch | 单 Kernel | 无显式 buffer consumer |

##### 分层内存布局

N/A：无 AIV UB Queue/TBuf、无 SIMT shared 数组。低精度值和 FLOAT32 中间量仅以每线程标量存在；Host local memory budget 的具体 runtime 布局固定仓不可见。

##### 字节账与复用顺序

```text
UB_total(AIV/shared explicit) = 0 B
dynamic/local_memory_budget = platform_UB - 131072 B
workspace_total = 0 B
per_thread = one T input scalar + two FLOAT32 scalars + index state
exact register/local spill bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程低精度/F32 标量 | GM load / cast / expf | cast / expf / downcast | 下一 grid-stride iteration | 单线程程序顺序 | 无跨线程别名；idx 唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load/promote | `static_cast<float>` specialization | FP16/BF16 GM load | `expf` | per-thread scalar | 输入先升到 FLOAT32 |
| compute/downcast | `expf` + `static_cast<T>` | FLOAT32 x | T GM store | per-thread scalar | 固定源码未给显式 round mode；使用类型 cast 语义 |
| tail | grid-stride guard | tid/G | loop exit | GM bound | `idx<count`，不访问无效 lane |
| 同步 | N/A | 单线程链 | N/A | 无共享数据 | 无 barrier、atomic、event |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_exp/op_host/arch35/foreach_exp_tiling.cpp` | dtype selector | schMode 1/2、相同 grid/POD |
| Kernel entry/router | `foreach/foreach_exp/op_kernel/arch35/foreach_exp.cpp` | `foreach_exp<1/2>` | half/BF16 template dispatch |
| Device body/helper | `foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h` | `PromoteToFloat<half/bfloat16_t>`; `CastFromFloat`; `OpForeachExpSimt<T>` | 升精度、expf、降精度和 grid-stride tail |

## 算子特定优化

### Trick T-001：一个 SIMT 网格顺序覆盖整个 TensorList

#### 触发与源码依据

- 适用 Route：R-001、R-002。
- 触发条件：Tensor count≤256；所有 Tensor dtype 一致；单个 `foreach_exp` launch。
- Golden 来源：[`foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h:108-112`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h#L108-112)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    for (int32_t t = 0; t < tensorCount; t++) {
        int64_t count = tensorElements[t];
        if (count == 0) {
            continue;
        }
```

#### 机制与设计原因

Host 不为每个 Tensor 单独生成 launch；一个 VF 内所有线程逐 Tensor 重用相同 grid，并以 `idx=tid+k*(threads*blocks)` 覆盖元素。由源码可证明这避免了 TensorList 项级中间 workspace 与多 Kernel 启动图；空 Tensor 直接跳过，长短 Tensor 都由同一 grid-stride 公式处理。`blockDim=min(P,ceil(totalElements/1024))` 根据列表总量选择网格，而非受单个 Tensor shape 限制。

#### 迁移边界

- 适用：TensorList 各项可独立逐元素处理、dtype 一致且无跨 Tensor 归约的 unary 算子。
- 前提：POD 必须保存每项精确元素数，TensorList 指针表 ABI 不变；线程写地址必须互斥。
- 失效：Tensor 间存在依赖、每项需要不同模板/资源或必须独立同步时，需拆 launch 或增加分发。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；若用总元素扁平 offset 跨 Tensor 直接寻址，会破坏 TensorList 非连续 GM 指针语义。

### Trick T-002：低精度只在每线程标量上升精度，不物化 FLOAT32 Tensor

#### 触发与源码依据

- 适用 Route：R-002。
- 触发条件：T 为 `half` 或 `bfloat16_t`。
- Golden 来源：[`foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h:123-134`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h#L123-134)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
        for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
            // Read input value
            T xVal = xData[idx];

            // Promote to float32 for precision
            float xFloat = PromoteToFloat<T>(xVal);

            // Compute exp using high-precision math API
            float yFloat = expf(xFloat);

            // Cast back to original type
            yData[idx] = CastFromFloat<T>(yFloat);
        }
```

#### 机制与设计原因

每线程只持有一个低精度输入和两个 FLOAT32 标量，指数计算使用 `expf`，结果立即降回目标 dtype 并写 GM。与物化整 Tensor 的 FLOAT32 临时副本相比，固定源码明确没有额外 GM/workspace/shared buffer；精度提升局限在当前元素生命周期内。

#### 迁移边界

- 适用：元素独立、可用 FLOAT32 标量数学函数、输出仍为原低精度 dtype 的 unary 算子。
- 前提：低精度到 FLOAT32 及返回的 cast 语义满足接口；无需跨元素共享中间量。
- 失效：需要高精度输出、跨元素归约或显式指定 round mode 时，必须调整存储/数值协议。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；直接对低精度值使用另一套近似函数不等价于固定源码的 `promote→expf→cast` 顺序。

## 模板与实现对比

| 对比维度 | R-001 FLOAT32 | R-002 FLOAT16/BFLOAT16 | 设计选择依据 |
|---|---|---|---|
| 命中条件/优先级 | dtype FLOAT32，schMode 0 | dtype FLOAT16/BFLOAT16，schMode 1/2 | Host `if/else if` 顺序 |
| 模板与 dtype | `Process<float>` | `Process<half>` / `Process<bfloat16_t>` | compile-time selector 消除 runtime dtype branch |
| 循环与分核 | `min(P,ceil(N/1024))` blocks，1024 threads，逐 Tensor grid-stride | 完全相同 | dtype 不改变调度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无显式 Queue/shared；每线程 FLOAT32 标量 | 无显式 Queue/shared；低精度+FLOAT32 标量 | localMemory budget 都为 U−128KiB |
| 内存总占用与复用 | workspace 0；标量下一 idx 复用 | workspace 0；标量下一 idx 复用 | 无 Tensor 级中间物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT GM scalar load/store + expf | SIMT GM scalar load/store + casts + expf | 单 SIMT Kernel |
| 同步、精度与 tail | 无同步；FLOAT32；idx guard | 无同步；F32 compute 后回 cast；idx guard | threads 写地址互斥 |
| fallback/失败边界 | 非该 dtype 尝试 R-002 | 非三种 dtype Host 失败 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，详见无后缀卡 | 支持，详见无后缀卡 | 支持；公开文档、OpDef、config/build 一致 |
| Host/Launcher | arch22 公共 Tiling，最多 50 Tensor/core | 同 A2 | arch35 自定义 Tiling，最多 256 Tensor，按总量/1024 选 block |
| Kernel/资源 | AIV_VECTOR + TQue，Tiling workspace 32 B | 同 A2 | SIMT 1024 threads/block，无显式 Queue/shared/workspace |

## 限制与失败边界

- 950 Tiling 明确拒绝 `tensorNum>256`；POD `tensorCount` 为 int32，逐项元素数和总元素数为 int64。`totalElements += numel` 未见显式溢出检查。
- Tiling 在读取 `tensorNum` 后立即要求 `GetDynamicInputDesc(0,0)` 非空，因此空 TensorList 会在 dtype 获取处失败；“支持空 Tensor”是指列表项可为零元素 Tensor。
- `localMemorySize` 用无符号 `ubSize-128KiB` 后窄化到 uint32；固定源码只检查 UB 非 0，未显式防止 `ubSize<128KiB` 的下溢，也未检查大于 uint32 上限。
- Tiling 仅从动态输入第 0 项取得 dtype；列表内 dtype 一致、输出 dtype/shape 和维度约束依赖公开 ACLNN/OpDef 框架校验，arch35 Tiling 本体不重复验证。
- Kernel 的 `needCoreNum` 与 `totalElements` POD 字段不由 Device body读取；真实 block 数依赖 Host `SetBlockDim`，复用时两者必须保持一致。
- 无显式 barrier/atomic 的安全性依赖 `idx=tid+kG` 对每个 Tensor 形成互斥写集合；不得改成会让多个线程写同一元素的映射。
- 输出比输入更大时仅输入元素前缀被写；额外输出区域语义固定源码未定义。生成式 ACLNN bridge/Executor workspace、cleanup 与 runtime 错误映射固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_exp/docs/aclnnForeachExp.md` | `aclnnForeachExpGetWorkspaceSize/aclnnForeachExp` | 公开平台、两段式 API、dtype/shape/错误语义 |
| 入口实现 | `foreach/foreach_exp/op_host/foreach_exp_def.cpp` | `ops::ForeachExp`; ascend950 config | 动态输入输出、AutoContiguous、950 注册 |
| Host/Tiling/Launcher | `foreach/foreach_exp/op_host/arch35/foreach_exp_tiling.cpp` | `GetPlatformInfo`; `ForeachExpTilingFunc` | POD、blockDim、workspace、localMemory 与 dtype selector |
| Host/Tiling/Launcher | `foreach/foreach_exp/op_kernel/arch35/foreach_exp_tiling_data.h` | `ForeachExpTilingData` | POD 字段和 256 项容量 |
| Host/Tiling/Launcher | `foreach/foreach_exp/op_kernel/arch35/foreach_exp_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 0/1/2 build selector |
| Kernel entry/body | `foreach/foreach_exp/op_kernel/arch35/foreach_exp.cpp` | `foreach_exp<schMode>` | compile-time router 与 GM POD 访问 |
| Kernel entry/body | `foreach/foreach_exp/op_kernel/arch35/foreach_exp_simt.h` | `Process`; `OpForeachExpSimt`; cast helpers | VF launch、指针表、线程映射、精度与 tail |
| 构建接线 | `foreach/foreach_exp/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950 Host 选择 arch35 Tiling |
| 构建接线 | `foreach/foreach_exp/op_host/config/ascend950/foreach_exp_binary.json` | `ForeachExp` op_list | 950 三 dtype binary 条目 |
| 构建接线 | `foreach/foreach_exp/op_kernel/CMakeLists.txt` | `add_kernel_sources(KERNEL_SRC arch35/foreach_exp.cpp)` | ascend950 Kernel source 选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B POD 保存 tensorCount 与 256 个 numel | 字段 offset/位宽、256 上限、GM 直接访问 | `ForeachExpTilingData`; `Process` |
| 调用顺序 | Host 填 POD/selector→单 Kernel→VF_CALL | schMode 与 T 必须一一对应；VF 线程数固定 1024 | `ForeachExpTilingFunc`; `foreach_exp`; `Process` |
| workspace/内存 | workspace 0；每线程标量，无显式 shared buffer | localMemorySize 的 128KiB 保留与 TensorList GM owner | `SetLocalMemorySize`; `OpForeachExpSimt` |
| 同步 | 独占 idx 消除跨线程同步 | grid-stride 不得产生地址冲突；同 stream ordering 不得改变 | `OpForeachExpSimt` |
| 精度与尾块 | FP32 `expf`; FP16/BF16 promote→expf→cast；`idx<count` | cast 顺序、dtype specialization、空 Tensor continue | `PromoteToFloat`; `CastFromFloat`; VF loop |
| 失败边界 | platform/null/tensorNum/dtype 失败阻断 launch | 不得越过 256；不得把不支持 dtype 映射到任一 schMode | `GetPlatformInfo`; `ForeachExpTilingFunc` |

- [数学函数expf](../../../../apis/simt_api/math_functions/float_math_functions/expf.md)
