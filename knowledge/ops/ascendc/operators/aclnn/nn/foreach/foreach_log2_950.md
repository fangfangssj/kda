---
type: Operator
title: "ForeachLog2 950 ACLNN 算子知识"
description: "说明 aclnnForeachLog2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/docs/aclnnForeachLog2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_host/foreach_log2_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_host/arch35/foreach_log2_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_kernel/arch35/foreach_log2_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_kernel/arch35/foreach_log2_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_kernel/arch35/foreach_log2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_kernel/arch35/foreach_log2_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_host/config/ascend950/foreach_log2_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:47:38Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachLog2 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachLog2GetWorkspaceSize` + `aclnnForeachLog2`，无版本后缀公开 ACLNN 两段式接口 |
| 计算语义 | 对 TensorList 中每个张量逐元素计算 `log2` |
| 输入输出 | 动态 TensorList；ND；FP16/FP32/BF16；列表内及输入输出 dtype 一致；每个 Tensor 0–8 维 |
| 执行类型 | 单 Kernel，唯一 Device 节点为 SIMT；无 Cube/AIC/MIX_CV |
| Kernel 路线 | R-001 FP32 SIMT；R-002 FP16→FP32 SIMT；R-003 BF16→FP32 SIMT |
| 核心资源 | GM TensorList 指针表、每 AIV block 1024 SIMT threads、线程私有标量/寄存器；无显式 AIV UB Queue、shared buffer、Atomic 或 workspace |
| 来源闭包 | `source_closed`：公开 ACLNN 文档/OpDef → arch35 Tiling/POD/template key → templated Kernel entry → SIMT body → ascend950 build/config 已闭合；自动生成 ACLNN bridge 与 `log2f` 数学库内部作为框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`，对每个 Tensor `i` 的线性存储下标 `j` 计算 `y_i[j]=log2(x_i[j])`。950 三条路线最终都调用 SIMT `log2f`：FP32 直接计算，FP16/BF16 先转换成 FP32，结果再转换回存储 dtype。固定仓没有额外定义负数、零、NaN 或无穷值处理，边界行为由 950 SIMT `log2f` 和标量 cast 决定。公开文档声明默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 0–8 维、ND | FP16/FP32/BF16 | 非空指针；支持空 Tensor；所有项 dtype 一致；输入可非连续，OpDef `AutoContiguous()` 接入框架连续化 |
| `out` | TensorList；每项 0–8 维、ND | 与 `x` 相同 | 非空指针；输出不支持非连续；每项容量不小于对应输入；Kernel 只写输入元素数 |
| `workspaceSize` | 标量输出 | `uint64_t*` | 第一段返回 executor 总 workspace；950 ForeachLog2 节点自身请求 0 B，框架连续化聚合量固定仓不可见 |
| `executor` | 执行器输出 | `aclOpExecutor**` | 第一段生成，第二段消费 |
| `workspace` | Device 地址 | `void*` | 传给 entry，但 950 body 完全不读取 |
| `stream` | Runtime stream | `aclrtStream` | 第二段提交执行；调用者负责需要的同步 |

### 入口与版本

固定 Commit 只公开无版本后缀 `aclnnForeachLog2`，故为 950 实际支持的最新公开版本。README/API 产品表明确列出 Ascend 950PR/950DT 支持；OpDef 注册 `ascend950`，host/kernel CMake 将其映射至 arch35，binary config 发布三种 dtype。自动生成 ACLNN bridge 不在仓内，作为框架边界；算子自有的 Host、POD、template selector、entry 和 SIMT body 均已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

公开第一段接口校验 `x/out` 非空、dtype 属于 FP16/FP32/BF16、输入输出 dtype 一致；接口文档还把列表内 dtype 不一致、shape 不满足和维数超过 8 归为 Tiling 错误。OpDef 为输入输出动态列表注册 ND 与 `AutoContiguous()`。arch35 Tiling 获取 AIV 核数和 UB 容量并拒绝 0；从输入 0 的 instance info 取得 `tensorNum`，拒绝大于 256；读取首个动态输入 descriptor 的 dtype，逐 Tensor 读取 storage shape 元素数，填充 `tensorElements[]` 并求和。若列表本身为 0 项，`GetDynamicInputDesc(0,0)` 返回空时会失败；“空 Tensor”不等于“空 TensorList”。

### 参数、Workspace 与生命周期

设总元素数 `N=sum_i n_i`、平台 AIV 核数 `C`。Tiling 选择 `B=1`（`N==0`）或 `B=min(C,ceil(N/1024))`，同时写 `needCoreNum=B` 并 `SetBlockDim(B)`。节点 workspace 为 0。Tiling 还调用 `SetLocalMemorySize(uint32_t(UB-128KiB))`；SIMT body 没有显式 dynamic/shared UB 参数或分配，因此该值属于 launch 的 local-memory budget，固定 body 未展示消费子区。POD 与 executor 在 launch 完成前有效；`x/y` 指针表和各 Tensor GM 地址由调用者/框架保持有效。

### 启动图与失败传播

```text
aclnnForeachLog2GetWorkspaceSize(x,out)
  -> 自动生成 ACLNN bridge（固定仓不可见）
  -> ForeachLog2 OpDef / AutoContiguous
  -> ForeachLog2TilingFunc
     -> ForeachLog2TilingData + GET_TPL_TILING_KEY(schMode)
  -> executor 保存单节点 launch
aclnnForeachLog2(workspace,workspaceSize,executor,stream)
  -> foreach_log2<schMode><<<blockDim=B,stream>>>(xList,yList,workspace,tiling)
     -> NsForeachLog2::Process<T,float>
        -> asc_vf_call<OpForeachLog2Simt<T,float>>(dim3(1024),...)
```

唯一 Kernel 节点直接从 `xList` GM 读、向 `yList` GM 写，workspace 无 slice，单节点内由全局 thread id 划分元素；不存在 Kernel 间边、event 或 cleanup。平台信息/POD/内存设置/dtype 任一检查失败即返回 `GRAPH_FAILED`，阻止 launch plan；unsupported dtype 在 Host 明确失败。

## Route 与参数设计

### 分核与资源计算

Host 使用 `MIN_PER_CORE=1024` 元素：

```text
N = Σ tensorElements[t]
B = (N==0) ? 1 : min(AIVCoreNum, ceil(N/1024))
threadsPerBlock = 1024
globalThreads = B*1024
localMemorySize = uint32_t(UBSize - 128*1024)
workspace = 0
```

每个 SIMT thread 的全局 id `tid=blockIdx.x*blockDim.x+threadIdx.x`，步长 `stride=blockDim.x*gridDim.x=B*1024`。所有 thread 都顺序遍历 `t=0..tensorCount-1`；对每个非空 Tensor，thread 处理 `idx=tid,tid+stride,...<tensorElements[t]`。所以 B 由跨列表总元素数决定，但元素分配在每个 Tensor 内重新从 `tid` 起算。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `GET_TPL_TILING_KEY(0)`；`foreach_log2<schMode=0>` | `dataType==DT_FLOAT` | FP16 判断 |
| 2 | R-002 FP16→FP32 SIMT | `GET_TPL_TILING_KEY(1)`；`foreach_log2<schMode=1>` | `dataType==DT_FLOAT16` | BF16 判断 |
| 3 | R-003 BF16→FP32 SIMT | `GET_TPL_TILING_KEY(2)`；`foreach_log2<schMode=2>` | `dataType==DT_BF16` | Host `GRAPH_FAILED` |

```text
if dtype == FP32: schMode=0 -> Process<float,float>
else if dtype == FP16: schMode=1 -> Process<half,float>
else if dtype == BF16: schMode=2 -> Process<bfloat16_t,float>
else: GRAPH_FAILED
```

`GET_TPL_TILING_KEY` 的最终编码由固定仓之外的 template-argument 基础设施生成；源码闭合的是 `schMode` 取值 0/1/2 及其编译期实例，不把 encoded key 猜成字面 0/1/2。

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | FP32、950、tensorNum≤256、平台参数有效 | 单 SIMT `foreach_log2<0>`；B blocks × 1024 threads | `Process<float,float>` → `OpForeachLog2Simt<float,float>` | 直接 GM load/store；线程局部 FP32 标量；无 barrier/Atomic/shared UB | `idx<count` 保护；空 Tensor `continue`；unsupported dtype Host 失败 |
| R-002 FP16 | schMode 1 | FP16，其余同上 | 单 SIMT `foreach_log2<1>` | `Process<half,float>` → SIMT half/float | GM FP16→线程 FP32→GM FP16；无共享状态 | 同上 |
| R-003 BF16 | schMode 2 | BF16，其余同上 | 单 SIMT `foreach_log2<2>` | `Process<bfloat16_t,float>` → SIMT BF16/float | GM BF16→线程 FP32→GM BF16；无共享状态 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList 容量 | `tensorNum<=256` | POD `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `ForeachLog2TilingFunc`; `MAX_TENSOR_NUM` |
| 平台 AIV 核 | `GetCoreNumAiv()!=0` | blockDim producer 必须非零 | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| UB/local memory | `UBSize!=0`，随后 `UBSize-128KiB` 窄化到 uint32 | 为 arch35 kernel 设置 local memory budget | 全部 | 仅 0 被拒绝；`UBSize<128KiB` 下溢无显式 Guard | `GetPlatformInfo`; `SetLocalMemorySize` |
| 最小每核工作 | `B=min(C,ceil(N/1024))` | 避免总元素很少时启动所有 AIV block | 全部 | 无另一 Route；B 至少 1 |
| 固定 thread 数 | `THREAD_NUM=1024`，`LAUNCH_BOUND(1024)` | SIMT body 与 `asc_vf_call(dim3(1024))` 一致 | 全部 | 固定仓无 threads fallback | `OpForeachLog2Simt`; `Process` |
| 空 Tensor | `count==0` | 避免解引用/计算空 storage | 全部 | 当前 Tensor `continue`，处理下一 Tensor |
| 末尾保护 | `idx<count` | global thread stride 通常不能整除 Tensor 元素数 | 全部 | 该 thread 不再访问本 Tensor |
| dtype | 仅 FP32/FP16/BF16 | 仅实例化三种 schMode | 全部 | Host `GRAPH_FAILED` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachLog2TilingData` | `needCoreNum` | `int32_t`/32 | offset 0；结构自然对齐推导 `alignof=8`,`sizeof=2064` | AIV block | `N==0?1:min(C,ceil(N/1024))` | body 未读取；同值传给 `SetBlockDim` | 正值；窄化自 int64，无额外溢出检查 |
| `ForeachLog2TilingData` | `tensorCount` | `int32_t`/32 | offset 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`→`OpForeachLog2Simt` 外层循环上界 | 0..256；0 项在更早 descriptor 读取处通常失败 |
| `ForeachLog2TilingData` | `totalElements` | `int64_t`/64 | offset 8 | 元素 | 逐 Tensor `totalElements+=numel` | body 未读取；Host 仅用于 B | 累加无显式溢出检查 |
| `ForeachLog2TilingData` | `tensorElements[256]` | `int64_t[256]` | offset 16，长度 2048 B，结构结束于 2064 | 元素/张量 | `storageShape.GetShapeSize()` | SIMT `count=tensorElements[t]`；地址以 `offsetof` 从 GM POD 取得 | 仅前 `tensorCount` 项活动；其余由 `memset_s` 置 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | `ForeachLog2TilingFunc`, dtype FP32 分支 | `ForeachLog2TilingData`; B、count、element array | `arch35/foreach_log2.cpp::foreach_log2<0>` | `Process<float,float>`; `OpForeachLog2Simt<float,float>`; `SimtGetTensorAddr<float>` | arch35 kernel CMake + ascend950 FP32 binary item |
| R-002 FP16 | schMode 1 | 同函数，dtype FP16 分支 | 同上 | `foreach_log2<1>` | `Process<half,float>`; SIMT body | ascend950 FP16 binary item |
| R-003 BF16 | schMode 2 | 同函数，dtype BF16 分支 | 同上 | `foreach_log2<2>` | `Process<bfloat16_t,float>`; SIMT body | ascend950 BF16 binary item |

## Kernel 实现

### 公共执行流程

arch35 entry 通过 `REGISTER_TILING_DEFAULT/GET_TILING_DATA_WITH_STRUCT` 注册并取得 POD，同时保留指向 GM POD 的 `tilingGm`。编译期 `if constexpr` 把 schMode 0/1/2 映射到三组 `<T,ComputeT>`。`Process` 不把 2064 B POD复制到 UB，只通过 `offsetof(tensorElements)` 构造 GM 数组指针，读取 `tensorCount`，再以 `asc_vf_call` 启动 1024 个 SIMT thread。每个 thread 对列表所有 Tensor 重复 TensorList 指针表解析，并按全局 thread stride 独立 load→`log2f`→store；线程间无共享数据、barrier 或 Atomic。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach/foreach_log2/op_host/foreach_log2_def.cpp` | `ForeachLog2` | 950 config，dtype FP32 | arch35 Tiling |
| Host/Tiling | `foreach/foreach_log2/op_host/arch35/foreach_log2_tiling.cpp` | `ForeachLog2TilingFunc` | `dataType==DT_FLOAT` | `GET_TPL_TILING_KEY(schMode=0)` |
| Router | `foreach/foreach_log2/op_kernel/arch35/foreach_log2.cpp` | `foreach_log2<schMode>` | `if constexpr(schMode==0)` | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log2<0>(xList,yList,workspace,tiling)
  -> NsForeachLog2::Process<float,float>
     -> elemCounts = tilingGm + offsetof(tensorElements)
     -> asc_vf_call<OpForeachLog2Simt<float,float>>(dim3(1024),...)
        -> SimtGetTensorAddr<float>(xList/yList,t)
        -> xVal=float(xData[idx]) -> yVal=log2f(xVal) -> yData[idx]=float(yVal)
```

##### 源码等价伪码

```text
# 伪码，所有 B 个 AIV block 各运行 1024 SIMT threads
tid = blockIdx.x*1024 + threadIdx.x
stride = 1024*gridDim.x
for t in [0,tensorCount):
  count=tensorElements[t]
  if count==0: continue
  xData=TensorListPointerTable(xList)[t]
  yData=TensorListPointerTable(yList)[t]
  for idx=tid; idx<count; idx+=stride:
    yData[idx]=log2f(xData[idx])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM TensorList header/table | `dataAddr`; `table=dataAddr+(*dataAddr>>3)` | `tensorCount` 个 64-bit Tensor 地址 | framework TensorList ABI | framework | 每个 SIMT thread 的 `SimtGetTensorAddr` | executor/框架；Kernel 期间有效 | offset 右移 3 转为 uint64 索引 |
| GM `x_t` | `TensorPtr[t]+idx` | `count` FP32 | ND 线性 storage | caller/框架 | thread scalar load | caller | `idx<count` |
| thread `xVal/yVal` | 寄存器分配固定仓不可见 | 两个概念 FP32 scalar 生命周期 | FP32 | load/`log2f` | `log2f`/store | 当前 thread/iteration | 无共享/alias；实际寄存器字节不可见 |
| GM `y_t` | `TensorPtr[t]+idx` | `count` FP32 | ND | thread store | caller | caller；stream 完成后可见 | 每个 idx 单 writer，无 Atomic |
| local memory/workspace | 无显式子区 | launch localMemory=`UB-128KiB`; workspace 0 | N/A | Host | body 无显式 consumer | launch/executor | 不臆造 dynamic UB 布局 |

##### 分层内存布局

N/A：该 SIMT body 没有 `InitBuffer`、TQue/TBuf、shared/dynamic UB 或通信 window。实际替代路径是 GM 标量 load/store与线程局部 `xVal/yVal`；Host 仅设置未被源码显式切片的 local-memory budget。

##### 字节账与复用顺序

```text
AIV_UB_queue_total = 0 B
SIMT shared/dynamic UB explicit allocation = 0 B
thread register bytes = fixed repo invisible (xVal/yVal logical lifetime only)
launch localMemorySize = uint32_t(UBSize-128KiB), no body suballocation
workspace_total(node)=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `xVal/yVal` | 当前 idx load/log2f | 当前 idx log2f/store | 当前 thread 下一 idx | 单线程程序顺序 | 每个 idx 在覆盖前已 store；线程间地址互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load→compute→store | SIMT 单线程程序顺序 | GM load | `log2f`/GM store | 当前 idx | FP32 输入与输出，无显式 round |
| 线程划分 | grid-stride loop | `tid/stride` | 每次 idx | GM element | `idx<count` 尾保护；每元素唯一 thread |
| 跨线程 | N/A | N/A | N/A | 无共享状态 | 无 barrier/Atomic 需求 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_log2/op_host/arch35/foreach_log2_tiling.cpp` | `GetPlatformInfo`, `ForeachLog2TilingFunc` | B、local memory、schMode 0、workspace 0 |
| Kernel entry/router | `foreach/foreach_log2/op_kernel/arch35/foreach_log2.cpp` | `foreach_log2<0>` | FP32 template mapping |
| Device body/helper | `foreach/foreach_log2/op_kernel/arch35/foreach_log2_simt.h` | `Process<float,float>`, `OpForeachLog2Simt`, `SimtGetTensorAddr` | 1024 threads、grid-stride、GM 直算 |

### Route R-002：FP16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_log2_def.cpp` | `ForeachLog2` | 950、FP16 | arch35 Tiling |
| Host/Tiling | `arch35/foreach_log2_tiling.cpp` | `ForeachLog2TilingFunc` | `dataType==DT_FLOAT16` | schMode 1 |
| Router | `op_kernel/arch35/foreach_log2.cpp` | `foreach_log2<1>` | `if constexpr(schMode==1)` | `Process<half,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log2<1> -> Process<half,float>
  -> asc_vf_call<OpForeachLog2Simt<half,float>>(1024 threads)
     -> SimtGetTensorAddr<half>
     -> static_cast<float>(xData[idx])
     -> log2f
     -> static_cast<half>(yVal) -> GM store
```

##### 源码等价伪码

```text
# 伪码；Tensor/thread 循环与 R-001 相同
for idx=tid; idx<count; idx+=B*1024:
  fp32 = float(xHalf[idx])
  fp32Result = log2f(fp32)
  yHalf[idx] = half(fp32Result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x_t/y_t` | `TensorPtr[t]+idx` | count FP16 | ND | caller/thread store | thread/caller | caller | `idx<count`；每元素独立 |
| thread scalar | 当前 idx | 概念 FP32 `xVal/yVal` | FP32 | FP16 cast/log2f | log2f/FP16 cast | thread iteration | 实际寄存器数量不可见 |
| local/workspace | N/A | explicit 0 B；launch budget 同 R-001 | N/A | Host | body 无 consumer | launch | 不使用 Queue/shared/Atomic |

##### 分层内存布局

N/A：无显式 AIV UB 或 SIMT shared/dynamic buffer；GM FP16 标量在寄存器级转换为 FP32，计算后直接转换并写回 GM。

##### 字节账与复用顺序

```text
explicit UB/shared bytes=0
workspace=0
thread scalar storage=fixed repo invisible
global traffic per valid element=2 B load + 2 B store (TensorList table访问除外)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread `xVal/yVal` | FP16 load/cast、log2f | log2f、FP16 store | 下一 idx | thread program order | store 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| FP16→FP32 | `static_cast<ComputeT>` | GM half | `log2f` | thread scalar | FP32 计算 |
| FP32→FP16 | `static_cast<T>` | `log2f` | GM store | output scalar | round mode未在源码显式命名 |
| tail | `idx<count` | tid/stride | load/store | Tensor 末尾 | 无越界 lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_log2/op_host/arch35/foreach_log2_tiling.cpp` | FP16 branch | schMode 1，公共 B |
| Kernel entry/router | `foreach/foreach_log2/op_kernel/arch35/foreach_log2.cpp` | `foreach_log2<1>` | half storage/float compute |
| Device body/helper | `foreach/foreach_log2/op_kernel/arch35/foreach_log2_simt.h` | `OpForeachLog2Simt<half,float>` | 标量升精度/降精度与 tail |

### Route R-003：BF16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_log2_def.cpp` | `ForeachLog2` | 950、BF16；ascend950 config 使用全 dtype 列表 | arch35 Tiling |
| Host/Tiling | `arch35/foreach_log2_tiling.cpp` | `ForeachLog2TilingFunc` | `dataType==DT_BF16` | schMode 2 |
| Router | `op_kernel/arch35/foreach_log2.cpp` | `foreach_log2<2>` | `if constexpr(schMode==2)` | `Process<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_log2<2> -> Process<bfloat16_t,float>
  -> asc_vf_call<OpForeachLog2Simt<bfloat16_t,float>>(1024 threads)
     -> SimtGetTensorAddr<bfloat16_t>
     -> BF16 scalar cast to float -> log2f -> cast to BF16 -> GM store
```

##### 源码等价伪码

```text
# 伪码；线程映射同 R-001
for idx=tid; idx<count; idx+=B*1024:
  fp32=float(xBf16[idx])
  fp32Result=log2f(fp32)
  yBf16[idx]=bfloat16_t(fp32Result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x_t/y_t` | `TensorPtr[t]+idx` | count BF16 | ND | caller/thread | thread/caller | caller | idx guard；single writer |
| thread scalar | 当前 idx | 概念 FP32 `xVal/yVal` | FP32 | BF16 cast/log2f | log2f/BF16 cast | thread iteration | 寄存器 ABI 不可见 |
| local/workspace | N/A | explicit 0 B；launch budget 同公共路径 | N/A | Host | 无显式 body consumer | launch | 无 Queue/shared/Atomic |

##### 分层内存布局

N/A：无显式 AIV UB 或 SIMT shared/dynamic buffer；BF16 元素按线程直接从 GM 读取，以 FP32 scalar 计算后写回。

##### 字节账与复用顺序

```text
explicit UB/shared bytes=0
workspace=0
thread scalar storage=fixed repo invisible
global traffic per valid element=2 B load + 2 B store (TensorList table访问除外)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread `xVal/yVal` | BF16 cast/log2f | log2f/BF16 store | next idx | thread program order | store before reuse |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `static_cast<float>` | GM BF16 | `log2f` | thread scalar | 以 FP32 计算 |
| FP32→BF16 | `static_cast<bfloat16_t>` | log2f | GM store | output scalar | 源码未命名 round mode |
| tail/empty | `count==0 continue`; `idx<count` | POD count | address/load | Tensor domain | 空 Tensor不解址；末尾不越界 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach/foreach_log2/op_host/arch35/foreach_log2_tiling.cpp` | BF16 branch | schMode 2 |
| Kernel entry/router | `foreach/foreach_log2/op_kernel/arch35/foreach_log2.cpp` | `foreach_log2<2>` | BF16 storage/float compute |
| Device body/helper | `foreach/foreach_log2/op_kernel/arch35/foreach_log2_simt.h` | `OpForeachLog2Simt<bfloat16_t,float>` | GM direct scalar pipeline |

## 算子特定优化

### Trick T-001：按总工作量裁剪 block、按 Tensor 复用全局 grid-stride

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：`tensorCount<=256`，Host 以 `ceil(totalElements/1024)` 裁剪 blockDim；Device 固定 1024 threads/block，并对每个 Tensor使用相同的全局 `tid/stride`。
- Golden 来源：[`foreach/foreach_log2/op_kernel/arch35/foreach_log2_simt.h:64-74`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_log2/op_kernel/arch35/foreach_log2_simt.h#L64-L74)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    blockIdx.x * blockDim.x +
    threadIdx.x);
uint64_t stride = static_cast<uint64_t>(
    blockDim.x * gridDim.x);

for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    ComputeT xVal = static_cast<ComputeT>(xData[idx]);
    ComputeT yVal = log2f(xVal);
    yData[idx] = static_cast<T>(yVal);
}
```

#### 机制与设计原因

Host 不无条件启动平台全部 AIV block，而以跨列表总元素数每 1024 元素分配一个 block；Device 再将 `B*1024` 个 thread 作为统一 grid。每个 Tensor 都从相同 `tid` 重启，使长 Tensor 自动被整个 grid 分摊，短 Tensor 的多余 thread 通过 `idx<count` 退出，无需为异长 Tensor生成逐项分核 POD。每个 thread 直接 GM load/store，避免 UB tile 和跨线程同步。FP16/BF16 通过模板 `ComputeT=float` 在寄存器标量层升精度，不物化整块 FP32 buffer。

#### 迁移边界

- 适用：TensorList 逐元素、元素间独立、每项可用相同 unary 函数的 SIMT 算子。
- 前提：TensorList 指针表 ABI 相同；每元素唯一 writer；固定 1024 threads/block 合法；总元素数足以作为 blockDim 的近似负载信号。
- 失效：需要跨元素归约、邻域访问、跨 Tensor 状态或共享缓存时，必须增加 barrier/共享内存/不同分核；不能直接复用此 body。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若把 `stride` 误写为单 block 的 `blockDim.x`，不同 block 会处理相同 idx 并发生重复写；源码使用 `blockDim.x*gridDim.x` 保证全 grid 唯一覆盖。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 2 | Host dtype if/else 唯一选择 |
| 模板与 dtype | `<float,float>` | `<half,float>` | `<bfloat16_t,float>` | 存储 dtype不同，计算统一用 float（FP32 route 本身也是 float） |
| 循环与分核 | B blocks×1024，逐 Tensor grid-stride | 相同 | 相同 | shape/dtype 不改变调度 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无显式 Queue；FP32 scalar | 无显式 Queue；FP32 scalar | 无显式 Queue；FP32 scalar | 直接 GM 标量流水 |
| 内存总占用与复用 | workspace 0；寄存器不可见 | 同 | 同 | body 无共享中间量 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT GM load/log2f/store | 加 FP16 cast | 加 BF16 cast | 精度转换差异 |
| 同步、精度与 tail | thread order、FP32、idx guard | FP32 compute/FP16 store | FP32 compute/BF16 store | 每元素独立无需 barrier |
| fallback/失败边界 | unsupported dtype Host 失败 | 同 | 同 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，详见无后缀 A2/A3 卡 | 公开支持，详见无后缀 A2/A3 卡 | 公开支持 Ascend 950PR/950DT |
| Host/Launcher | arch22 公共 Foreach Tiling，32 B workspace | 同 A2 | arch35 独立 Tiling，POD 256 项，workspace 0，local memory=`UB-128KiB` |
| Kernel/资源 | AIV_VECTOR，UB Queue/事件 | AIV_VECTOR，UB Queue/事件 | SIMT，1024 threads/block，GM direct，无显式 UB Queue/Atomic |

## 限制与失败边界

- 950 Tiling 明确限制 Tensor 数不超过 256；公开文档没有写该数值，复用实现时必须保留 Host Guard。
- `ForeachLog2TilingFunc` 在读取 dtype 前要求 `GetDynamicInputDesc(0,0)` 非空，因此固定实现不能证明 0 项 TensorList 可用；单个空 Tensor通过 `count==0` 安全跳过。
- `totalElements` 为 `int64_t` 累加，`needCoreNum` 再窄化到 `int32_t`；固定源码无 shape-size 溢出 Guard。
- 仅检查 `UBSize!=0`，随后无条件计算无符号 `UBSize-128KiB` 并窄化为 `uint32_t`；若平台 UB 不满足该前提，源码没有 fallback。
- `needCoreNum` 与 `totalElements` 写入 POD 但 Device 未读取；真正的 gridDim 来自 `SetBlockDim`。修改 launch 时不能假设 Device 会用 POD 修正 grid。
- 输出容量可大于输入时只写输入元素数，额外区域不清零。输入连续化和 executor 总 workspace 聚合位于自动生成 ACLNN bridge 框架边界，精确切片不可见。
- Template key 的 encoded 数值由仓外宏实现生成；只能复用 `schMode=0/1/2` 语义，不应硬编码未经证实的最终 key。
- `log2f` 内部近似、特殊值和 FP16/BF16 降精度 round mode未在固定仓展开；不得推测。本文未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_log2/docs/aclnnForeachLog2.md` | 两段式 API | 原型、约束、错误码、950 公开支持 |
| 入口声明/约束 | `foreach/foreach_log2/README.md` | 产品支持表、公式 | Ascend 950PR/DT 与语义 |
| 入口实现 | `foreach/foreach_log2/op_host/foreach_log2_def.cpp` | `ops::ForeachLog2` | dtype/layout/AutoContiguous/ascend950 注册 |
| Host/Tiling/Launcher | `foreach/foreach_log2/op_host/arch35/foreach_log2_tiling.cpp` | `GetPlatformInfo`, `ForeachLog2TilingFunc` | POD、B、workspace、local memory、dtype selector |
| POD | `foreach/foreach_log2/op_kernel/arch35/foreach_log2_tiling_data.h` | `ForeachLog2TilingData` | 四字段 ABI |
| Host selector | `foreach/foreach_log2/op_kernel/arch35/foreach_log2_tiling_key.h` | `ForeachLog2` template args | schMode 0/1/2 声明与选择 |
| Kernel entry/body | `foreach/foreach_log2/op_kernel/arch35/foreach_log2.cpp` | `foreach_log2<schMode>` | POD 注册、三模板 router |
| Kernel entry/body | `foreach/foreach_log2/op_kernel/arch35/foreach_log2_simt.h` | `Process`, `OpForeachLog2Simt`, `SimtGetTensorAddr` | 1024-thread SIMT、指针解址、grid-stride、精度/tail |
| 构建接线 | `foreach/foreach_log2/op_host/config/ascend950/foreach_log2_binary.json` | 三个 `op_list` 条目 | FP16/FP32/BF16 binary 发布 |
| 构建接线 | `foreach/foreach_log2/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `SUPPORT_TILING_DIR` | ascend950→arch35 |
| 构建接线 | `foreach/foreach_log2/op_kernel/CMakeLists.txt` | `KERNEL_SRC arch35/foreach_log2.cpp` | 950 kernel source 选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 以 256 项 int64 数组传每 Tensor 元素数 | 字段顺序/位宽/offset、tensorCount≤256、POD 在 GM 可按 `offsetof` 读取 | `ForeachLog2TilingData`; `Process` |
| 调用顺序 | entry→Process→asc_vf_call→SIMT body | `asc_vf_call` threads=1024 与 `LAUNCH_BOUND(1024)` 一致 | `foreach_log2`; `Process`; `OpForeachLog2Simt` |
| workspace/内存 | 节点 workspace 0；无显式共享/UB buffer | `SetLocalMemorySize(UB-128KiB)` 的 launch 约定、TensorList GM ABI不可随意改 | `ForeachLog2TilingFunc`; `SimtGetTensorAddr` |
| 同步 | 每元素独立，依靠单线程顺序 | stride 必须为完整 grid；若引入共享状态需新增同步 | `OpForeachLog2Simt` |
| 精度与尾块 | FP16/BF16 以 FP32 `log2f`；`idx<count` | storage cast顺序、空 Tensor `continue`、round mode不可臆造 | `OpForeachLog2Simt<T,ComputeT>` |
| 失败边界 | Host拒绝 tensorNum>256/无平台资源/unsupported dtype | 0 项列表、UB<128KiB、累加溢出没有 fallback | `GetPlatformInfo`; `ForeachLog2TilingFunc` |