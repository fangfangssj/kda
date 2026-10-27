---
type: Operator
title: "ForeachErf 950 ACLNN 算子知识"
description: "说明 aclnnForeachErf 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/docs/aclnnForeachErf.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_host/foreach_erf_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_host/arch35/foreach_erf_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_kernel/arch35/foreach_erf_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_kernel/arch35/foreach_erf_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_kernel/arch35/foreach_erf.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_kernel/arch35/foreach_erf_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_host/config/ascend950/foreach_erf_binary.json
    role: implementation
created_at: "2026-08-25T15:31:37Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachErf 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnForeachErfGetWorkspaceSize` + `aclnnForeachErf`；固定仓未出现其它公开版本。 |
| 计算语义 | 对 TensorList 中每个张量逐元素计算 erf；950 Device 实体采用 clamp 后的 FP32 有理式近似。 |
| 输入输出 | ND TensorList；FLOAT16/FLOAT32/BFLOAT16；输出与输入同 dtype/对应元素域。 |
| 执行类型 | 单 Kernel：模板 entry `foreach_erf<schMode>`，执行族为 SIMT。 |
| Kernel 路线 | R-001 FLOAT32；R-002 FLOAT16→FLOAT32；R-003 BFLOAT16→FLOAT32。 |
| 核心资源 | TensorList GM 地址表、2064 B POD、每 block 1024 SIMT threads、寄存器标量；无 source-visible AIV Queue/TBuf、无中间 workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN→生成式 bridge/OpDef→`ForeachErfTilingFunc`→`ForeachErfTilingData`/模板 key→`foreach_erf<schMode>`→`Process/OpForeachErfSimt/ComputeErf`→950 构建与 binary config。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

公开语义为 `y_i[j]=erf(x_i[j])`。950 Kernel 的实际有限精度实现先把输入转换为 FLOAT32（输入本来是 FLOAT32 时省略转换），将其 clamp 到 `[-3.92,3.92]`，令 `z=x²`，用 Horner 顺序计算 5 次多项式形成分子、分母并返回二者之比。NaN 通过 `x!=x` 检测并以 `0.0f/0.0f` 产生 NaN；FLOAT16/BFLOAT16 最后 `static_cast<T>` 降精度。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 为 ND，0–8 维；允许零元素 Tensor；公开支持非连续输入 | FLOAT16/FLOAT32/BFLOAT16 | 非空指针；列表内 dtype 一致；950 tiling 显式要求 Tensor 数 ≤256。 |
| `out` | TensorList；ND，0–8 维；公开不支持非连续输出 | 与 `x` 相同 | 非空指针；输出容量不小于对应输入；Kernel 只写输入 numel。 |
| 属性 | N/A | N/A | 无算子属性。 |

### 入口与版本

公开 API、确定性声明与 A2/A3 相同。README/API 产品表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 将其映射到 `arch35`，Kernel CMake 显式选择 `arch35/foreach_erf.cpp`，binary config 覆盖三种 dtype，因此 950 公开支持无冲突。ACLNN header/inner bridge 由 `ACLNNTYPE aclnn` 生成，实体不在固定仓，作为框架边界；`ForeachErf` 自有 Host/POD/Kernel 全部闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段按公开文档检查空指针、dtype、列表内部 dtype、一致的输入输出 dtype、shape 容量与最大 8 维；输入连续化的生成细节固定仓不可见。
2. `ForeachErfTilingFunc` 获取 AIV core 数和 UB 字节数，任一为 0 则失败。
3. 从动态输入实例信息取得 `tensorNum`，大于 256 返回 `GRAPH_FAILED`；读取第 0 个动态输入描述以选 dtype，因此空 TensorList 无有效 dtype 描述时失败。
4. 循环每个输入 storage shape，写 `tensorElements[i]` 并累加 `totalElements`；获取 shape/POD/workspace 指针失败均阻断 launch。

### 参数、Workspace 与生命周期

Host 令 `C=AIVCoreNum`、`N=totalElements`：若 `N==0`，`blockDim=needCoreNum=1`；否则 `blockDim=needCoreNum=min(C,ceil(N/1024))`。每个 Kernel block 内 `VF_CALL` 创建 1024 个 SIMT thread，故全局逻辑 stride 为 `1024*blockDim`。

算子 tiling workspace slice 为 0 B，Kernel `workspace` 参数未读取；ACLNN 总 workspace 仍可能包含生成式输入连续化等框架 slice，其公式固定仓不可见。Host 另调用 `SetLocalMemorySize(uint32_t(ubSize-128*1024))`；SIMT body 没有 source-visible shared/dynamic UB 对象或地址使用，因此该值只可记录为 launch local-memory budget，不能虚构 buffer。POD 与输入/输出 TensorList 地址至少在当前 stream 节点执行完成前有效。

### 启动图与失败传播

```text
aclnnForeachErfGetWorkspaceSize  # 生成式 bridge
  -> ForeachErf OpDef/公开校验
  -> ForeachErfTilingFunc
     -> POD + blockDim + localMemorySize + symbolic template key
  -> workspaceSize = framework preprocess slice + 0 B operator slice; executor
aclnnForeachErf(workspace,workspaceSize,executor,stream)
  -> [optional generated contiguous node(s), fixed仓不可见]
  -> foreach_erf<schMode><<<blockDim, stream>>>(xList,yList,operatorWorkspace,tiling)
     -> NsForeachErf::Process<T>
        -> VF_CALL<OpForeachErfSimt<T>>(Dim3(1024),...)
```

算子自有部分为单 Device 节点，无算子自有的跨 Kernel workspace/GM 边；生成式连续化可能在其前增加框架节点与临时边。Host/tiling 任一检查失败则不 launch；Kernel 没有运行时 dtype fallback，因为 `schMode` 是编译期模板选择。stream 保证该节点与同 stream 前后任务的 ordering。

## Route 与参数设计

### 分核与资源计算

`MIN_PER_CORE=1024` 只用于选 blockDim，不把 Tensor 切成 Host per-core slice。每个 SIMT block 的 thread 仍遍历全部 Tensor，针对 Tensor `t` 使用全局 `tid=blockIdx*threadNum+threadIdx` 与 `stride=threadNum*blockNum` 处理 `idx=tid,tid+stride,...`；不同 block/thread 写互斥元素。零元素 Tensor 被 `continue` 跳过。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FLOAT32 SIMT | `GET_TPL_TILING_KEY(SCH_MODE_FLOAT)`；`foreach_erf<0>` | `dataType==ge::DT_FLOAT` | 检查 FLOAT16 |
| 2 | R-002 FLOAT16 SIMT | `GET_TPL_TILING_KEY(SCH_MODE_FLOAT16)`；`foreach_erf<1>` | `dataType==ge::DT_FLOAT16` | 检查 BF16 |
| 3 | R-003 BFLOAT16 SIMT | `GET_TPL_TILING_KEY(SCH_MODE_BF16)`；`foreach_erf<2>` | `dataType==ge::DT_BF16` | 其它 dtype `GRAPH_FAILED` |

```text
DT_FLOAT   -> symbolic schMode 0 -> Process<float>
DT_FLOAT16 -> symbolic schMode 1 -> Process<half>
DT_BF16    -> symbolic schMode 2 -> Process<bfloat16_t>
else -> tiling error, no launch
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FLOAT32 | schMode 0 | FLOAT32、ND、≤8 维、Tensor 数≤256、950 | 单 SIMT Kernel | `foreach_erf<0>`→`Process<float>`→`OpForeachErfSimt<float>` | 直接 GM load/store；寄存器 FP32；无 barrier/atomic | `idx<count`；无 fallback |
| R-002 FLOAT16 | schMode 1 | FLOAT16，其余同上 | 单 SIMT Kernel | `foreach_erf<1>`→`Process<half>`→SIMT body | GM half→寄存器 FP32→GM half | 同上 |
| R-003 BFLOAT16 | schMode 2 | BFLOAT16，其余同上 | 单 SIMT Kernel | `foreach_erf<2>`→`Process<bfloat16_t>`→SIMT body | GM BF16→寄存器 FP32→GM BF16 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 最少 1024 元素/核 | `blockDim=min(C,ceil(N/1024))`；N=0 特判 1 | 配合每 block 1024 threads，避免为很小总域启动过多 block | 全部 | 无替代 Route | `ForeachErfTilingFunc` |
| Tensor 数 | `tensorNum<=256` | POD 固定 `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `ForeachErfTilingFunc` |
| local memory 扣 DCache | `SetLocalMemorySize(uint32_t(U-128KiB))` | 为 arch35 local memory launch 元数据扣除 DCache | 全部 | 固定源码无 `U>=128KiB` guard，减法下溢风险需外部平台保证 | `ForeachErfTilingFunc` |
| dtype 模板闭集 | FLOAT/FLOAT16/BF16 | 编译期 entry 只有 3 个 `schMode` | 各 Route | `GRAPH_FAILED` | `ForeachErfTilingFunc`; `foreach_erf<schMode>` |
| clamp | `x<-3.92` 或 `x>3.92` | 限制有理式求值域 | 全部 | 饱和到边界后仍求近似，无独立 fallback | `Clamp`; `ErfApproximation` |
| 空 Tensor | `count==0` | 避免取得/访问无元素 Tensor 数据 | 全部 | `continue` 到下一 Tensor | `OpForeachErfSimt` |

### POD/ABI

`ForeachErfTilingData` 是按源码字段顺序的标准布局聚合体：offset 0/4/8/16，总大小 `16+256*8=2064` B，alignment 8 B。固定仓未给出显式 `static_assert`，复用时仍应以同编译 ABI 校验。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachErfTilingData` | `needCoreNum` | int32/32 | offset 0 | block | N=0 写1，否则 `min(C,ceil(N/1024))` | Device 未读取；blockDim 同值 | inactive POD 字段 |
| 同上 | `tensorCount` | int32/32 | offset 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT 外层 `t` 循环 | 0..256；空列表会先因 dtype desc 缺失失败 |
| 同上 | `totalElements` | int64/64 | offset 8 | 元素 | storage numel 累加 | Device 未读取 | 仅 Host 分核；累加溢出无独立 guard |
| 同上 | `tensorElements` | int64[256]/每项64 | offset 16，总 2048 B | 元素/张量 | `tensorElements[i]=storageShape.GetShapeSize()` | `OpForeachErfSimt: count=tensorElements[t]` | 仅前 tensorCount 项活动 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | schMode 0 | `arch35/foreach_erf_tiling.cpp:ForeachErfTilingFunc` | `ForeachErfTilingData`; 1024 threads | `arch35/foreach_erf.cpp:foreach_erf<0>` | `Process<float>`; `OpForeachErfSimt<float>`; `ComputeErf/ErfApproximation` | `op_kernel/CMakeLists.txt`; ascend950 binary config |
| R-002 | schMode 1 | 同上 | 同上 | `foreach_erf<1>` | half specialization，公共 SIMT helpers | 同上 |
| R-003 | schMode 2 | 同上 | 同上 | `foreach_erf<2>` | BF16 specialization，公共 SIMT helpers | 同上 |

## Kernel 实现

### 公共执行流程

模板 entry 将 POD 作为 GM 指针使用，避免把含 256 项数组的结构复制到栈；`Process<T>` 通过 `offsetof` 取得 `tensorElements` 地址，并以 `VF_CALL(...Dim3(1024))` 启动 SIMT body。所有 thread 按相同顺序遍历 TensorList；每个 Tensor 内以 global thread id/stride 划分元素。每个有效元素只有一次 GM load、寄存器计算和一次 GM store，无 Queue、DMA、barrier、atomic 或 workspace。

### Route R-001：FLOAT32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/foreach_erf_tiling.cpp` | `ForeachErfTilingFunc` | `dataType==DT_FLOAT` | template key schMode 0 |
| Router | `op_kernel/arch35/foreach_erf.cpp` | `foreach_erf<schMode>` | `schMode==TILING_KEY_FLOAT(0)` | `Process<float>` |
| Body | `op_kernel/arch35/foreach_erf_simt.h` | `ComputeErf<float>` | `is_same_v<T,float>` | 直接 `HandleSpecialValues` |

#### Device 数据流

##### 函数级调用链

```text
foreach_erf<0>
  -> NsForeachErf::Process<float>
     -> VF_CALL<OpForeachErfSimt<float>>(1024 threads)
        -> SimtGetTensorAddr<float>
        -> ComputeErf<float> -> HandleSpecialValues -> ErfApproximation -> Clamp
```

##### 源码等价伪码

```text
# 伪码
for t in [0,tensorCount):
  count=tensorElements[t]; if count==0: continue
  xData/yData = TensorListAddr(t)
  tid=blockIdx*1024+threadIdx; stride=1024*blockNum
  for idx=tid; idx<count; idx+=stride:
    x=xData[idx]
    if x is NaN: y=NaN
    else: xc=clamp(x,-3.92,3.92); z=xc*xc
          y=HornerNumerator(z,xc)/HornerDenominator(z)
    yData[idx]=y
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList address table | `base + ((*base)>>3) + t` | tensorCount 个 uint64 pointer | uint64 table | framework | `SimtGetTensorAddr` | executor/launch | 8 B pointer |
| GM x | `xData[idx]` | 1 float/thread iteration | FLOAT32/ND | caller | thread register | caller | idx tail guard |
| registers | thread-local `x,xClamped,x2,numer,denom` | 源码标量；物理寄存器数固定仓不可见 | float | loads/Horner | Horner/store | thread iteration | 值生命周期内复用 numer/denom |
| GM y | `yData[idx]` | 1 float/thread iteration | FLOAT32/ND | thread | caller | caller | 无重叠写 |

##### 分层内存布局

N/A：该 SIMT body 无 source-visible AIV UB、shared/dynamic UB 或中间 buffer；实际路径为 GM→thread-local registers→GM。Host 仍设置 `localMemorySize=U-128KiB` 的 launch budget，但 body 没有对应地址消费。

##### 字节账与复用顺序

```text
UB_total = N/A（无 InitBuffer/Queue/TBuf）
dynamic/local_memory_launch_budget = U - 128 KiB（body source-visible usage = 0 B）
workspace_total = 0 B
register_bytes = 固定仓不可见；仅能证明每线程标量生命周期
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `numer` | Horner step | 下一 Horner step/最终 division | 同一线程下一赋值 | SIMT 单线程程序顺序 | 不跨 thread 共享 |
| thread-local `denom` | Horner step | 最终 division | 下一元素迭代 | 单线程程序顺序 | 同上 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分工 | global tid/stride | launch geometry | each thread | GM element | `idx<count` 保护尾部；无 barrier |
| 特殊值 | `x!=x`; clamp | x register | approximation/store | scalar | NaN→NaN；±Inf 被 clamp 后走边界近似，并非显式 ±1 常量 |
| 多项式 | Horner + division | FP32 registers | GM store | scalar | 全程 FP32；固定系数 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_erf/op_host/arch35/foreach_erf_tiling.cpp` | `ForeachErfTilingFunc` | blockDim、POD、key、workspace/local memory |
| Kernel entry/router | `foreach_erf/op_kernel/arch35/foreach_erf.cpp` | `foreach_erf<0>` | FLOAT32 模板分发 |
| Device body/helper | `foreach_erf/op_kernel/arch35/foreach_erf_simt.h` | `OpForeachErfSimt<float>`; `ErfApproximation` | grid stride、直接 GM、Horner 近似 |

### Route R-002：FLOAT16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `arch35/foreach_erf_tiling.cpp` | `ForeachErfTilingFunc` | `dataType==DT_FLOAT16` | schMode 1 |
| Router | `arch35/foreach_erf.cpp` | `foreach_erf<1>` | schMode 1 | `Process<half>` |
| Body | `arch35/foreach_erf_simt.h` | `ComputeErf<half>` | non-float branch | `static_cast<float>`→FP32近似→`static_cast<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_erf<1> -> Process<half> -> VF_CALL<OpForeachErfSimt<half>>
  -> ComputeErf<half> -> HandleSpecialValues(float) -> ErfApproximation -> cast half
```

##### 源码等价伪码

```text
# 伪码；tid/stride 与 R-001 相同
x16 = xData[idx]
xf = float(x16)
rf = HandleSpecialValues(xf)
yData[idx] = half(rf)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `TensorListAddr(t)+idx` | 1 half load/store | FLOAT16/ND | caller/thread | thread/caller | launch | idx guard |
| registers | `xVal` half；`xFloat/resultFloat` FP32；Horner 标量 | physical bytes 不可见 | half+float | load/cast | approximation/cast | iteration | 全部 thread-local |

##### 分层内存布局

N/A：无 AIV UB、shared/dynamic UB、中间 GM；GM→half register→FP32 registers→half register→GM。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic/local_memory_launch_budget = U - 128 KiB（source-visible usage 0 B）
workspace_total = 0 B
register_bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 thread scalars | cast/Horner | subsequent scalar ops | next idx | program order | thread-private |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load/promote | C++ scalar cast | GM half | FP32 approximation | scalar | half 精确转换到 FP32 |
| compute | FP32 Horner/division | FP32 x | cast back | scalar | clamp/NaN 同 R-001 |
| store | `static_cast<half>` | FP32 result | GM | scalar | 舍入模式未在源码显式命名；idx<count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `arch35/foreach_erf_tiling.cpp` | FLOAT16 branch | schMode 1 |
| Kernel entry/router | `arch35/foreach_erf.cpp` | `foreach_erf<1>` | half body |
| Device body/helper | `arch35/foreach_erf_simt.h` | `ComputeErf<half>` | FP32 promotion/compute/demotion |

### Route R-003：BFLOAT16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `arch35/foreach_erf_tiling.cpp` | `ForeachErfTilingFunc` | `dataType==DT_BF16` | schMode 2 |
| Router | `arch35/foreach_erf.cpp` | `foreach_erf<2>` | schMode 2 | `Process<bfloat16_t>` |
| Body | `arch35/foreach_erf_simt.h` | `ComputeErf<bfloat16_t>` | non-float branch | BF16→FP32→BF16 |

#### Device 数据流

##### 函数级调用链

```text
foreach_erf<2> -> Process<bfloat16_t> -> VF_CALL<OpForeachErfSimt<bfloat16_t>>
  -> ComputeErf<bfloat16_t> -> HandleSpecialValues(float) -> ErfApproximation -> cast BF16
```

##### 源码等价伪码

```text
# 伪码；线程映射同 R-001
xb = xData[idx]
xf = float(xb)
rf = rational_erf(clamp_or_nan(xf))
yData[idx] = bfloat16_t(rf)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | `TensorListAddr(t)+idx` | 1 BF16 load/store | BF16/ND | caller/thread | thread/caller | launch | idx guard |
| registers | BF16 x/y + FP32 approximation scalars | physical bytes 不可见 | BF16+float | load/cast | compute/store | iteration | thread-local |

##### 分层内存布局

N/A：无 AIV UB、shared/dynamic UB、中间 GM；使用 GM 与 thread-local registers。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic/local_memory_launch_budget = U - 128 KiB（source-visible usage 0 B）
workspace_total = 0 B
register_bytes = 固定仓不可见
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 thread scalars | BF16 cast/Horner | downstream scalar op | next idx | program order | thread-private |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| promote | `static_cast<float>` | BF16 GM value | approximation | scalar | BF16→FP32 |
| compute | FP32 Horner/division | FP32 | demote | scalar | 与 R-001 同系数/clamp |
| demote/store | `static_cast<bfloat16_t>` | FP32 | GM | scalar | 舍入模式固定仓未命名；idx<count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `arch35/foreach_erf_tiling.cpp` | BF16 branch | schMode 2 |
| Kernel entry/router | `arch35/foreach_erf.cpp` | `foreach_erf<2>` | BF16 body |
| Device body/helper | `arch35/foreach_erf_simt.h` | `ComputeErf<bfloat16_t>` | FP32 promotion/compute/demotion |

## 算子特定优化

### Trick T-001：跨 block 网格步进直接覆盖 TensorList，省去中间物化

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：950 arch35 SIMT body；Host 已给出 `blockDim`，每 block 的 `VF_CALL` 固定 1024 threads。
- Golden 来源：[`foreach/foreach_erf/op_kernel/arch35/foreach_erf_simt.h:145-164`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_erf/op_kernel/arch35/foreach_erf_simt.h#L145-L164)

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
            T xVal = xData[idx];
            T yVal = ComputeErf<T>(xVal);
            yData[idx] = yVal;
        }
```

#### 机制与设计原因

所有 block/thread 用全局 `tid` 和总线程数作为 stride，在每个 Tensor 内形成互斥覆盖；每元素直接 GM load→寄存器近似→GM store，不生成扁平化 workspace，也不需要 Host 为每核写 Tensor 起止 offset。零元素 Tensor 在取数据地址之后的工作前被跳过。此设计减少中间物化与 UB 搬运，代价是所有 block 都串行遍历 Tensor 元数据。

#### 迁移边界

- 适用：逐元素、无跨元素依赖、输出与输入一一对应的 TensorList 算子。
- 前提：TensorList 地址表 ABI、每元素独立、各 thread 的 global tid 唯一，输出无 alias 冲突。
- 失效：归约、跨元素窗口、需要共享中间量或输出元素被多个 thread 更新时，必须引入 barrier/atomic/分阶段实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；但若把 stride 错写成单 block 的 thread 数，多 block 会重复写同一输出元素。

## 模板与实现对比

| 对比维度 | R-001 FLOAT32 | R-002 FLOAT16 | R-003 BFLOAT16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 2 | Host dtype 分支 |
| 模板与 dtype | `Process<float>` | `Process<half>` | `Process<bfloat16_t>` | compile-time specialization |
| 循环与分核 | blockDim=min(C,ceil(N/1024)); 1024 threads；grid stride | 同左 | 同左 | dtype 不改变拓扑 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | FP32 thread scalars | half+FP32 scalars | BF16+FP32 scalars | 无 Queue/TBuf/通信；local budget未消费 |
| 内存总占用与复用 | workspace 0；寄存器不可量化 | 同左 | 同左 | 直接 GM |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM + SIMT `VF_CALL` | 同左 | 同左 | 单 SIMT 节点 |
| 同步、精度与 tail | FP32近似；idx guard | FP32近似后 half cast | FP32近似后 BF16 cast | 无 barrier/atomic |
| fallback/失败边界 | unsupported dtype Host失败 | 同左 | 同左 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 tiling | arch22 公共 tiling | arch35 独立 POD/key 与 local memory budget |
| Kernel/资源 | AIV Vector + UB Queue + AscendC Erf primitive | 同左 | SIMT 1024 threads，直接 GM，源码内 FP32 有理式 |

## 限制与失败边界

- 950 最多 256 个 Tensor；每个 Tensor 公开限制 0–8 维；空 Tensor 可跳过，但空 TensorList 因无法取得第 0 个 dtype 描述而失败。
- `totalElements` 的 int64 累加、`ceil(total/1024)`、`needCoreNum` 到 int32/uint32 的窄化没有独立溢出 guard。
- `ubSize-128KiB` 在 uint64 上计算后窄化为 uint32，固定源码未提供 UB 小于 128 KiB 或差值超过 uint32 的 guard；依赖已注册 950 平台属性。
- 实现把有限/无限输入 clamp 到 ±3.92 后计算有理式；NaN 显式传播，±Inf 没有显式返回 ±1，而是边界近似。不要把它描述为数学库精确值。
- FLOAT16/BFLOAT16 的最终 `static_cast<T>` 舍入模式、生成式 ACLNN bridge 与 VF_CALL 的底层 launch ABI 在固定仓不可见。
- `needCoreNum` 与 `totalElements` 写入 POD 但 Device 不消费；迁移 ABI 时不得因为 inactive 而擅自删除字段。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_erf/docs/aclnnForeachErf.md` | 两段式 ACLNN | API 原型、dtype/shape/错误 |
| 平台/语义 | `foreach/foreach_erf/README.md` | 产品表与公式 | 950 公开支持、数学语义 |
| 入口实现边界 | `foreach/foreach_erf/op_host/foreach_erf_def.cpp` | `ops::ForeachErf` | OpDef dtype/layout/平台 |
| Host/Tiling | `foreach/foreach_erf/op_host/arch35/foreach_erf_tiling.cpp` | `ForeachErfTilingFunc` | 校验、blockDim、POD、workspace、local memory、key |
| POD | `foreach/foreach_erf/op_kernel/arch35/foreach_erf_tiling_data.h` | `ForeachErfTilingData` | ABI 字段 |
| key | `foreach/foreach_erf/op_kernel/arch35/foreach_erf_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 闭集 |
| Kernel entry | `foreach/foreach_erf/op_kernel/arch35/foreach_erf.cpp` | `foreach_erf<schMode>` | key→dtype body |
| Kernel body | `foreach/foreach_erf/op_kernel/arch35/foreach_erf_simt.h` | `Process`; `OpForeachErfSimt`; `ComputeErf`; `ErfApproximation` | SIMT launch、线程映射、GM、近似与精度 |
| ACLNN/Host 构建接线 | `foreach/foreach_erf/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`; `SUPPORT_TILING_DIR`; `add_modules_sources` | ascend950→arch35 与生成式 ACLNN 接线 |
| 构建接线 | `foreach/foreach_erf/op_kernel/CMakeLists.txt` | `add_kernel_sources` ascend950 | arch35 Kernel 选择 |
| binary config | `foreach/foreach_erf/op_host/config/ascend950/foreach_erf_binary.json` | `ForeachErf` 三 dtype entries | 950 binary 发布覆盖 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B 四字段 POD | 字段顺序、offset、256 项、inactive 字段 | `ForeachErfTilingData`; `Process` |
| 调用顺序 | entry→Process→VF_CALL→逐 Tensor/逐 element | 1024 threads、全局 tid/stride | `foreach_erf<schMode>`; `OpForeachErfSimt` |
| workspace/内存 | workspace 0，直接 GM；local budget U-128KiB | TensorList address ABI、stream lifetime | `ForeachErfTilingFunc`; `SimtGetTensorAddr` |
| 同步 | 元素互斥，无 barrier/atomic | stride 必须含 blockNum | `OpForeachErfSimt` |
| 精度与尾块 | FP32 Horner；16-bit 路线升/降精度；idx guard | clamp、系数、求值顺序、NaN 处理 | `ComputeErf`; `ErfApproximation` |
| 失败边界 | >256/unsupported dtype/平台信息失败不 launch | 不添加未证实 fallback | `ForeachErfTilingFunc` |