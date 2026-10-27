---
type: Operator
title: "ForeachPowScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachPowScalarV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/docs/aclnnForeachPowScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/op_api/foreach_pow_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/foreach_pow_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/arch35/foreach_pow_scalar_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_host/config/ascend950/foreach_pow_scalar_binary.json
    role: implementation
created_at: "2026-08-25T15:47:57Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachPowScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachPowScalarV2GetWorkspaceSize` / `aclnnForeachPowScalarV2`，公开 ACLNN V2 |
| 计算语义 | 对 TensorList 的每个有效元素计算 `y_i[j]=pow(x_i[j], exponent)` |
| 输入输出 | 公开契约：x/out 为 FLOAT16/FLOAT32/BFLOAT16 ND TensorList；scalar 为 `aclScalar` |
| 执行类型 | 单 Kernel，SIMT（`__simt_vf__`，每 block 1024 threads）；无 Cube/AIC 或 MIX_CV |
| Kernel 路线 | R-001 FP32 SIMT；R-002 FP16→FP32 SIMT；R-003 BF16→FP32 SIMT |
| 核心资源 | GM TensorList 指针表、GM POD、SIMT thread grid；无显式 TQue/TBuf/shared/dynamic UB；workspace 0 |
| 来源闭包 | source_closed：公开 V2 API→ACLNN 前处理→L0 launcher→arch35 Tiling/POD/template key→SIMT body→ascend950 构建/config 闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对 `x=[x_0,...,x_(n-1)]` 的每个 tensor 和每个线性元素执行 `y_i[j]=x_i[j]^exponent`。950 实现不拼接或物化列表数据；所有 SIMT block 依次遍历 tensor 列表，每个线程用全 grid stride 只处理自己的元素下标。公开文档声明默认确定性实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；各 tensor ND，秩 0–8 | FLOAT32/FLOAT16/BFLOAT16 | 列表内 dtype 一致；支持空 Tensor；公开文档标记非连续 Tensor 不支持 |
| `scalar` | `aclScalar` | x=FP32/BF16 时 FLOAT/DOUBLE；x=FP16 时 FP16/DOUBLE | `abs(exponent)<=2^31-1`；ACLNN 将其转换为一元素 tensor，DOUBLE 窄化到计算 dtype |
| `out` | TensorList；ND | 与 x 相同 | ACLNN 实现逐位置要求 shape 精确相等；公开文档较弱地写 shape size≥输入 |
| `workspaceSize/executor` | Host 输出 | 框架类型 | 第一段接口生成；源码有空列表返回 0 的分支，但 element-0 dtype 读取发生在它之前，零长度列表安全性不能闭合 |
| `workspace/workspaceSize/executor/stream` | 第二段参数 | 框架类型 | `CommonOpExecutorRun` 在 stream 上执行 |

### 入口与版本

版本族包含旧 `aclnnForeachPowScalar`（tensor scalar，公开文档明确 950 不支持）和 V2（`aclScalar`，公开产品表支持 Ascend 950PR/950DT）。因此 950 只取最新且实际接线的 V2。V2 将 scalar 转成 tensor 后调用内部 `ForeachPowScalar`，950 使用独立 arch35 Tiling 与 SIMT Kernel。

V2 文档/header 只公开 FP16/FP32/BF16。ACLNN dtype 检查、OpDef、binary config 与 arch35 router 还包含 INT32，且 Device 实现了 `PowiImpl`；由于它没有进入 V2 公开 dtype 契约，本卡只将其记为未公开实现边界，不提升为正式 Route。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecForeachPowScalarV2GetWorkspaceSize` 创建 executor 后依次检查非空指针、当前架构、列表 dtype 一致性、out 与 x dtype、scalar 对应 dtype、输入秩≤8、输入输出 shape 精确相等以及非私有 format。参数检查成功后有空 TensorList 返回分支，但 `CheckDtypePowScalar` 已先读取 `(*x)[0]`，故固定仓不能证明零长度列表安全。非空时逐 x 调用框架 `l0op::Contiguous` 并重新构造 TensorList；`ConvertToTensor` 对 FP32 生成 FLOAT scalar tensor、对 FP16 生成 FLOAT16、对 BF16 生成 FLOAT；然后 L0 `ADD_TO_LAUNCHER_LIST_AICORE(ForeachPowScalar, OP_INPUT(x,scalar), OP_OUTPUT(out))` 添加唯一核心节点。Contiguous、scalar tensor 物化和 executor 内部对象是明确的框架边界。

### 参数、Workspace 与生命周期

ACLNN 第一段返回整个 executor 的 workspace，可能包含前置框架节点所需空间。arch35 Tiling 对核心算子写 `workspace[0]=0`，Device entry 不读取 workspace。Host 还调用 `SetLocalMemorySize(ubSize-128KiB)`；固定 body 没有显式 local/shared/dynamic UB 分配，因此这是 launch 资源上限而非可见的数据缓冲。contiguous inputs、scalar tensor、POD 与 launcher 归 executor 所有，在第二段调用及传入 stream 执行完成前有效。

### 启动图与失败传播

```text
aclnnForeachPowScalarV2GetWorkspaceSize
  -> [每个 x] l0op::Contiguous（框架节点，按需）
  -> ConvertToTensor(scalar)
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachPowScalar)
     -> ForeachPowScalarTilingFunc
     -> foreach_pow_scalar<schMode><<<needCoreNum, 1024 threads/block, stream>>>
        -> Process<T,S> -> VF_CALL<OpForeachPowScalarSimt<T,S>>
aclnnForeachPowScalarV2
  -> CommonOpExecutorRun(..., stream)
```

核心是单 SIMT Kernel，无 Kernel 间 workspace edge。Tiling/launcher 失败或 API `CHECK_RET` 失败会阻断 executor 生成/执行；同一 executor/stream 保证前置框架节点先于核心节点。核心节点直读 x/scalar/POD 并直写 y。

## Route 与参数设计

### 分核与资源计算

Host 遍历至多 256 个输入 tensor，记录 `tensorElements[i]=storageShape.GetShapeSize()`，并累加 `E=totalElements`。若 `E=0`，`needCoreNum=blockDim=1`；否则 `needCoreNum=blockDim=min(AIV_core_num,ceil(E/1024))`。每个 SIMT block 固定 `THREAD_NUM=1024`，全局线程号 `tid=blockIdx*threadNum+threadIdx`，总 stride=`threadNum*blockNum`。每个 tensor 内执行 `idx=tid; idx<count; idx+=stride`。

Tiling 取得平台 UB `U`，要求 `U!=0`，并设置 local memory size 为 `uint32(U-128KiB)`。源码没有 `U>=128KiB` guard；950 平台资源必须保证减法不下溢。Kernel 没有显式 UB/shared 分配，workspace 为 0。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | schMode 0；`Process<float,float>` | 公开 x/out FLOAT32，scalar FLOAT/DOUBLE，API 约束通过 | 检查下一 dtype |
| 2 | R-002 FP16 SIMT | schMode 1；`Process<half,half>` | 公开 x/out FLOAT16，scalar FLOAT16/DOUBLE，API 约束通过 | 检查下一 dtype |
| 3 | R-003 BF16 SIMT | schMode 3；`Process<bfloat16_t,float>` | 公开 x/out BF16，scalar FLOAT/DOUBLE，API 约束通过 | 不支持 dtype 时 Tiling 失败 |

```text
API 校验失败 -> ACLNN 错误
else dataType==DT_FLOAT   -> GET_TPL_TILING_KEY(0) -> R-001
else dataType==DT_FLOAT16 -> GET_TPL_TILING_KEY(1) -> R-002
else dataType==DT_BF16    -> GET_TPL_TILING_KEY(3) -> R-003
else -> 不属于 V2 公开契约；Tiling 另有实现级 INT32 schMode 2
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | 950；FP32；n≤256；API 约束通过 | 单 SIMT Kernel；blockDim×1024 threads | `Process<float,float>`→`OpForeachPowScalarSimt<float,float>` | 直接 GM load/powf/store；无显式共享区或 barrier | `idx<count` 保护；stride 覆盖尾部 |
| R-002 FP16 | schMode 1 | 950；FP16；n≤256；API 约束通过 | 同上 | `Process<half,half>`；`ComputeType<half>=float` | 每线程寄存器升到 float，powf 后降回 half | 同上 |
| R-003 BF16 | schMode 3 | 950；BF16；n≤256；API 约束通过 | 同上 | `Process<bfloat16_t,float>`；`ComputeType<BF16>=float` | scalar 已是 float；base 寄存器升精度 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| tensor 数≤256 | `tensorNum > 256 -> GRAPH_FAILED` | POD `tensorElements[256]` 定长 | 全部 | Tiling 失败 | `ForeachPowScalarTilingFunc` |
| 每核至少约 1024 元素 | `blockDim=min(coreNum,ceil(E/1024))` | 控制 SIMT block 数；每 block 1024 threads | 全部 | E=0 特判为 1 block | `MIN_PER_CORE/ForeachPowScalarTilingFunc` |
| 固定 1024 threads | `VF_CALL(Dim3(THREAD_NUM))`，`THREAD_NUM=1024` | `LAUNCH_BOUND(1024)` 与 thread 映射一致 | 全部 | 无可选 threads fallback | `Process/OpForeachPowScalarSimt` |
| local memory 扣 128KiB | `SetLocalMemorySize(uint32(U-128KiB))` | 预留 DCache；Kernel 本体不显式分配 local buffer | 全部 | 源码无 U<128KiB 失败 guard | `DCACHE_SIZE/ForeachPowScalarTilingFunc` |
| dtype key | FLOAT/FLOAT16/BF16 分别 0/1/3 | 编译期实例消除运行时 dtype 分支 | 各 Route | 其他公开不支持 dtype；Tiling else 失败 | `ForeachPowScalarTilingFunc` |
| 空 tensor | `count==0 -> continue` | 避免地址取用和循环 | 全部 | 跳到下一 tensor | `OpForeachPowScalarSimt` |

### POD/ABI

`ForeachPowScalarTilingData` 是普通标准布局字段序列：offset 0/4/8/16，按最大 8 字节对齐，`sizeof=2064`、`alignof=8`。Device 直接从 GM tiling 指针读取，避免把 2KiB 数组复制到栈。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachPowScalarTilingData` | `needCoreNum` | int32/32 | offset 0 | block 数 | 0 域写 1；否则 `int32(min(core,ceil(E/1024)))` | 固定 Device body 不读取；blockDim 使用同一 Host 值 | inactive POD 字段仍须保留 ABI |
| 同上 | `tensorCount` | int32/32 | offset 4 | tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT tensor 循环 | 0–256 |
| 同上 | `totalElements` | int64/64 | offset 8 | 元素 | 各 numel 累加 | 固定 Device body不读取 | 累加无显式溢出检查 |
| 同上 | `tensorElements` | `int64[256]` | offset 16；2048 bytes | 每 tensor 元素 | `tiling->tensorElements[i]=numel` | SIMT `count=tensorElements[t]` | 未用项由 memset 置 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | `arch35/foreach_pow_scalar_tiling_arch35.cpp::ForeachPowScalarTilingFunc` | `ForeachPowScalarTilingData` | `arch35/foreach_pow_scalar.cpp::foreach_pow_scalar<0>` | `Process<float,float>/OpForeachPowScalarSimt` | ascend950 arch35 CMake + binary config |
| R-002 FP16 | schMode 1 | 同上 | 同上 | `foreach_pow_scalar<1>` | `Process<half,half>`；`ComputeType<half>` | 同上 |
| R-003 BF16 | schMode 3 | 同上 | 同上 | `foreach_pow_scalar<3>` | `Process<bfloat16_t,float>`；`ComputeType<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

entry 通过 `GET_TILING_DATA_WITH_STRUCT` 注册/解析 POD，同时把原始 GM tiling 指针 cast 为 `const __gm__ ForeachPowScalarTilingData*`，避免大结构入栈。schMode 编译期选择 `<T,S>`。`Process` 每 block 从 scalar GM 读一个 `S`，用 `offsetof(...,tensorElements)` 构造 GM 数组指针，再 `VF_CALL` 1024 threads。每个线程对每个非空 tensor 获取列表地址，按全 grid stride 直接 GM load、寄存器计算、GM store。线程写集互斥，无 atomic、barrier 或跨线程共享状态。

### Route R-001：FP32 SIMT powf

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachPowScalarV2.md` | dtype 表 | x/out FLOAT32；scalar FLOAT/DOUBLE | 公开支持 |
| Tiling | `arch35/foreach_pow_scalar_tiling_arch35.cpp` | `dataType==DT_FLOAT` | `GET_TPL_TILING_KEY(0)` | schMode 0 |
| Router | `arch35/foreach_pow_scalar.cpp` | `schMode==TILING_KEY_FLOAT` | 编译期 true | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachPowScalarTilingFunc -> foreach_pow_scalar<0>
  -> NsForeachPowScalar::Process<float,float>
  -> VF_CALL<OpForeachPowScalarSimt<float,float>>(Dim3(1024), ...)
     -> SimtGetTensorAddr<float> -> powf -> GM store
```

##### 源码等价伪码

```text
scalarVal = *(float*)scalarGM
tid = blockIdx*1024 + threadIdx
stride = 1024*blockNum
for t in [0,tensorCount):
  if tensorElements[t]==0: continue
  xData = tensor_list_address<float>(x,t); yData = tensor_list_address<float>(y,t)
  for idx=tid; idx<tensorElements[t]; idx+=stride:
    yData[idx] = (float)powf((float)xData[idx], (float)scalarVal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList 地址表 | `base + (*base >> 3) + t` | tensorCount 个 64-bit pointer | pointer table | framework | `SimtGetTensorAddr` | executor/Kernel | 首字节偏移须为 8B 倍数 |
| GM x/y | `tensorAddr[t]+idx` | count 元素 | FP32/ND linear | caller/SIMT | SIMT/caller | caller | `idx<count`；线程写集互斥 |
| scalar GM | offset 0 | 1 float | FP32 | ConvertToTensor | `Process` | executor | 每 block 读取一次 |
| POD GM | offset 16 + 8t | 2064 bytes 总结构 | standard-layout | Tiling | Process/SIMT | launcher | 直接 GM 读取，不复制入栈 |
| dynamic/shared UB | N/A | 无显式分配 | N/A | N/A | N/A | N/A | 以寄存器局部值替代 |

##### 分层内存布局

N/A：该 SIMT body 无 `InitBuffer`、shared/dynamic UB 数组或 Kernel 间 buffer；每线程只持有 `xVal/xCompute/sCompute/result` 等寄存器生命周期局部值。Host 仅设置 local memory 上限 `U-128KiB`。

##### 字节账与复用顺序

```text
UB_explicit = 0
dynamic_ub_explicit = N/A
local_memory_limit = uint32(U - 128KiB)
workspace_total_for_core_kernel = 0
GM traffic per valid FP32 element = 4-byte load + 4-byte store
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 每线程寄存器局部值 | 当前 idx load/powf | 当前 idx store | 下一 idx | 单线程程序顺序 | 不跨线程共享；`idx` 写集互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 元素计算 | 单线程程序顺序 | GM load | powf/store | 当前元素 | FP32 base/scalar/result |
| 线程并行 | 无 barrier/atomic | 各 tid | Kernel 完成 | GM y | grid-stride 的 `idx<count` 覆盖任意尾数且无重叠 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_pow_scalar/op_host/arch35/foreach_pow_scalar_tiling_arch35.cpp` | `ForeachPowScalarTilingFunc` | blockDim、POD、key、workspace/local memory |
| Kernel entry/router | `foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar.cpp` | `foreach_pow_scalar<0>` | FP32 route |
| Device body/helper | `foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_simt.h` | `Process<float,float>/OpForeachPowScalarSimt` | 1024 threads、地址、stride、powf、tail |

### Route R-002：FP16 升精度 SIMT powf

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachPowScalarV2.md` | dtype 表 | x/out FLOAT16；scalar FLOAT16/DOUBLE | 公开支持 |
| Tiling | `ForeachPowScalarTilingFunc` | `dataType==DT_FLOAT16` | template key 1 | schMode 1 |
| Router | `foreach_pow_scalar<schMode>` | `schMode==TILING_KEY_FLOAT16` | compile-time | `Process<half,half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_pow_scalar<1>
  -> Process<half,half>
  -> VF_CALL<OpForeachPowScalarSimt<half,half>>(1024 threads)
     -> ComputeType<half>::type=float -> powf -> cast to half -> GM store
```

##### 源码等价伪码

```text
scalarVal = *(half*)scalarGM
for each tensor and idx assigned by global grid stride:
  xCompute = (float)xData[idx]
  sCompute = (float)scalarVal
  result = powf(xCompute,sCompute)
  yData[idx] = (half)result
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | tensor pointer + idx | count half elements | FP16/ND | caller/SIMT | SIMT/caller | caller | tail 用 idx guard，无 DMA padding |
| per-thread registers | 当前 idx | | half x/scalar；float x/scalar/result | scalar/GM load | powf/store | 当前迭代 | 下一 idx 覆盖 |
| UB/workspace | N/A | 0 显式字节 | N/A | N/A | N/A | N/A | 直接 GM+寄存器 |

##### 分层内存布局

N/A：没有 AIV Queue 或 SIMT shared/dynamic UB；FP16→FP32→FP16 仅在每线程局部值中完成。

##### 字节账与复用顺序

```text
UB_explicit = 0
workspace_core_kernel = 0
GM traffic per valid element = 2-byte load + 2-byte store
temporary lifetime = one thread iteration
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local float result | powf | cast/store | 下一 idx | 程序顺序 | 不逃逸线程 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| promote/pow/demote | C++ cast + powf | half load | half store | thread-local value | base/scalar 升 FP32，结果转 half；具体舍入模式固定仓不可见 |
| tail | `idx<count` | thread mapping | store | GM y | 任意 tensor 尾部，不写越界 lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_pow_scalar_tiling_arch35.cpp` | dtype branch | schMode 1 |
| Kernel entry/router | `op_kernel/arch35/foreach_pow_scalar.cpp` | `Process<half,half>` | 模板映射 |
| Device body/helper | `op_kernel/arch35/foreach_pow_scalar_simt.h` | `ComputeType<half>`、SIMT body | FP32 powf 与 grid-stride |

### Route R-003：BF16 升精度 SIMT powf

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `aclnn_foreach_pow_scalar_v2.cpp` | `ConvertToTensor` | x[0] dtype BF16 | scalar tensor 转 FLOAT |
| Tiling | `ForeachPowScalarTilingFunc` | `dataType==DT_BF16` | template key 3 | schMode 3 |
| Router | `foreach_pow_scalar<schMode>` | schMode BF16 | compile-time | `Process<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_pow_scalar<3>
  -> Process<bfloat16_t,float>
  -> VF_CALL<OpForeachPowScalarSimt<bfloat16_t,float>>(1024 threads)
     -> ComputeType<bfloat16_t>::type=float -> powf -> cast to bfloat16_t
```

##### 源码等价伪码

```text
scalarVal = *(float*)scalarGM
for each tensor and global-grid-stride idx:
  xCompute = (float)xData[idx]
  result = powf(xCompute, scalarVal)
  yData[idx] = (bfloat16_t)result
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | tensor pointer + idx | count BF16 elements | BF16/ND | caller/SIMT | SIMT/caller | caller | `idx<count` |
| scalar GM | 0 | 1 float | FP32 | ACLNN conversion | Process | executor | 每 block 读一次 |
| per-thread registers | 当前 idx | | BF16 x/y；FP32 x/scalar/result | load/powf | powf/store | 单迭代 | 下一 idx 覆盖 |
| UB/workspace | N/A | 0 显式字节 | N/A | N/A | N/A | N/A | 直接 GM+寄存器 |

##### 分层内存布局

N/A：没有显式 UB/shared buffer；BF16 输入的 FP32 计算值只存在于线程局部寄存器语义中。

##### 字节账与复用顺序

```text
UB_explicit = 0
workspace_core_kernel = 0
GM traffic per valid element = 2-byte load + 2-byte store
POD GM = 2064 bytes, read-only during Kernel
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local FP32 result | powf | BF16 cast/store | 下一 idx | 单线程程序顺序 | 无跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| promote/pow/demote | C++ cast + powf | BF16 load/FP32 scalar | BF16 store | thread local | base 升 FP32；结果降 BF16，具体舍入不可见 |
| tensor tail | grid-stride loop guard | thread mapping | GM store | 输出边界 | 空 tensor continue，非空严格 `idx<count` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_pow_scalar_tiling_arch35.cpp` | BF16 dtype branch | schMode 3、blockDim/POD |
| Kernel entry/router | `op_kernel/arch35/foreach_pow_scalar.cpp` | `Process<bfloat16_t,float>` | storage/compute dtype |
| Device body/helper | `op_kernel/arch35/foreach_pow_scalar_simt.h` | `ComputeType<bfloat16_t>`、SIMT body | FP32 powf、直接 GM store |

## 算子特定优化

### Trick T-001：跨整个 TensorList 复用同一全局 grid-stride 映射

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：任意非空 tensor；`blockDim=min(coreNum,ceil(totalElements/1024))`，每 block 1024 threads。
- Golden 来源：[`foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_simt.h:104-120`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_simt.h#L104-L120)

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
```

#### 机制与设计原因

每个 tensor 都复用同一个全局线程编号和 grid stride，直接从指针表取独立 GM 基址。这样不需要 Host 为每个 tensor 生成独立 launch，也不需要把列表拼接进 workspace；小 tensor 自然只激活前 `count` 个线程，大 tensor 由同一线程跨 stride 继续处理。各线程 idx 同余类互斥，因此输出无需 atomic 或 barrier。

#### 迁移边界

- 适用：TensorList 各项可独立逐元素计算，且可线性化为 count。
- 前提：地址表 ABI、tensorElements 与 tensorCount 一致；block/thread producer 保持；输出无跨元素依赖。
- 失效：tensor 间有归约/顺序依赖、输出 alias 导致跨线程冲突或需要布局变换时必须更换路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；迁移时若把 stride 错写成单 block 的 threadNum，会导致不同 block 写同一元素。

### Trick T-002：低精度只在寄存器语义中升到 FP32

#### 触发与源码依据

- 适用 Route：R-002、R-003
- 触发条件：`T` 为 `half` 或 `bfloat16_t`，`ComputeType<T>::type` 特化为 float。
- Golden 来源：[`foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_simt.h:47-53`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_simt.h#L47-L53)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
struct ComputeType { using type = T; };
template <> struct ComputeType<half> { using type = float; };
template <> struct ComputeType<bfloat16_t> { using type = float; };
```

#### 机制与设计原因

SIMT body 对每个元素 load 后立刻 cast 到 `C=float`，调用 `powf`，再只在 store 前降回 T。没有为整 tile 分配 FP32 GM/UB 中间张量，既锁定低精度 route 的计算顺序，也避免中间物化；BF16 scalar 在 API 层已经转成 FLOAT，FP16 scalar 在线程内再升到 FLOAT。

#### 迁移边界

- 适用：每元素中间值数量少、寄存器可容纳、输出仍为低精度的 SIMT unary 算子。
- 前提：目标数学函数接受 FP32；最终 cast 的舍入行为满足语义。
- 失效：需要跨线程共享中间值、多个结果或更高精度函数时需重新评估寄存器/UB 与精度路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；不得把 `ComputeType` 特化删除后声称仍保持相同 FP16/BF16 数值路径。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 3 | Host dtype selector |
| 模板与 dtype | T=float,S=float,C=float | T=half,S=half,C=float | T=BF16,S=float,C=float | scalar tensor producer 与 `ComputeType` |
| 循环与分核 | blockDim×1024 全局 stride | 同左 | 同左 | 总元素决定 blockDim |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程局部 FP32 | half+FP32 局部 | BF16+FP32 局部 | 无显式 Queue/shared/dynUB |
| 内存总占用与复用 | 显式 UB 0 | 显式 UB 0 | 显式 UB 0 | 直接 GM→寄存器→GM |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT load/powf/store | SIMT cast/powf/cast/store | 同左 | 无 DMA/layout/通信节点 |
| 同步、精度与 tail | FP32；idx guard | FP32 计算后 half | FP32 计算后 BF16 | 无 barrier/atomic |
| fallback/失败边界 | 不支持 dtype Tiling 失败 | 同左 | 同左 | INT32 是未公开实现，不是公共 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 支持，见无后缀卡 | V2 支持，见无后缀卡 | V2 文档及 ascend950 接线支持 |
| Host/Launcher | arch22 公共 foreach Tiling | 同 A2 | arch35 专用 Tiling；≤256 tensor；每核目标≥1024元素 |
| Kernel/资源 | AIV Vector + Queue/UB tile | 同 A2 | SIMT 1024 threads，直接 GM/寄存器，无显式 UB |

## 限制与失败边界

- V2 公开契约不含 INT32；源码中的 schMode 2、binary config 和 `PowiImpl` 只是未公开实现信号，不应作为 API 兼容承诺。
- `tensorNum` 最大 256；超过时 Tiling 明确失败。`tensorCount` 窄化到 int32 前已有该上限。
- 零长度 TensorList 的 0-workspace 分支晚于 `(*x)[0]` dtype 读取，固定仓不能证明该输入安全；非空列表内的零元素 tensor 则由 SIMT `count==0` 明确跳过。
- `totalElements` int64 累加没有显式溢出检查；shape size 负值也没有在 arch35 Tiling 单独检查。
- Host 检查 out 与 x shape 精确相等，比 V2 文档的“shape size 不小于”更严格；实现复用按精确相等处理。
- 文档标记非连续 x 不支持，源码仍调用框架 Contiguous；框架内部行为不在固定算子仓闭包内。
- `SetLocalMemorySize(uint32(U-128KiB))` 没有 `U>=128KiB` guard；不得迁移到不满足此前提的平台。
- `powf` 与低精度 cast 的底层近似、异常值和舍入细节属于编译器/数学库边界，固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_pow_scalar/docs/aclnnForeachPowScalarV2.md` | 产品支持表、接口参数表 | 950 公开性与 V2 契约 |
| 入口声明/约束 | `foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.h` | 两段式接口声明 | C ABI |
| 入口实现 | `foreach/foreach_pow_scalar/op_host/op_api/aclnn_foreach_pow_scalar_v2.cpp` | `CheckParams/Exec...` | 校验、Contiguous、scalar 转换、executor |
| 入口实现 | `foreach/foreach_pow_scalar/op_host/op_api/foreach_pow_scalar_v2.cpp` | `ForeachPowScalarV2` | 核心 launcher 节点 |
| Host/Tiling/Launcher | `foreach/foreach_pow_scalar/op_host/foreach_pow_scalar_def.cpp` | `ForeachPowScalar` OpDef | ascend950 注册与 dtype |
| Host/Tiling/Launcher | `foreach/foreach_pow_scalar/op_host/arch35/foreach_pow_scalar_tiling_arch35.cpp` | `ForeachPowScalarTilingFunc` | POD、blockDim、workspace、local memory、key |
| Host/Tiling/Launcher | `foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_tiling_data.h` | `ForeachPowScalarTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_tiling_key.h` | template key 声明/选择 | schMode 0–3 编译实例 |
| Kernel entry/body | `foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar.cpp` | `foreach_pow_scalar<schMode>` | entry/router |
| Kernel entry/body | `foreach/foreach_pow_scalar/op_kernel/arch35/foreach_pow_scalar_simt.h` | `Process/OpForeachPowScalarSimt/SimtGetTensorAddr` | SIMT body、地址、精度、tail |
| 构建接线 | `foreach/foreach_pow_scalar/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/TILING_DIR` | ascend950→arch35 |
| 构建接线 | `foreach/foreach_pow_scalar/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 Kernel 选择 |
| 构建接线 | `foreach/foreach_pow_scalar/op_host/config/ascend950/foreach_pow_scalar_binary.json` | `ForeachPowScalar` op_list | 950 binary dtype 接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064B 结构、直接 GM 消费 | 字段 offset、数组 256、tensorCount 与数组一致 | `ForeachPowScalarTilingData` |
| 调用顺序 | scalar/POD 读取→VF_CALL→tensor/idx 循环 | 1024 threads、全 grid stride 公式 | `Process/OpForeachPowScalarSimt` |
| workspace/内存 | 核心 workspace 0，无显式 UB | local memory 扣 128KiB；TensorList 指针 ABI | `ForeachPowScalarTilingFunc/SimtGetTensorAddr` |
| 同步 | 写集互斥，不用 barrier/atomic | `tid` 与 `stride=threadNum*blockNum` 不得改坏 | `OpForeachPowScalarSimt` |
| 精度与尾块 | FP16/BF16 升 FP32 powf；idx guard | scalar dtype producer、最终 cast、空 tensor continue | `ComputeType/Exec.../OpForeachPowScalarSimt` |
| 失败边界 | API 校验、tensorNum/平台/UB/dtype Tiling 检查 | 不把 INT32 实现信号扩张成 V2 契约 | `CheckParams/GetPlatformInfo/ForeachPowScalarTilingFunc` |