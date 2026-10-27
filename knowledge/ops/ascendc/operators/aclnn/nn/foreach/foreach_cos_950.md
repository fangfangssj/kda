---
type: Operator
title: "ForeachCos 950 ACLNN 算子知识"
description: "说明 aclnnForeachCos 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/docs/aclnnForeachCos.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_host/foreach_cos_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_host/arch35/foreach_cos_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_kernel/arch35/foreach_cos_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_kernel/arch35/foreach_cos_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_kernel/arch35/foreach_cos.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_kernel/arch35/foreach_cos_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_host/config/ascend950/foreach_cos_binary.json
    role: implementation
created_at: "2026-08-25T15:30:56Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachCos 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachCosGetWorkspaceSize` + `aclnnForeachCos`，公开两段式 ACLNN；固定仓只公开该版本。 |
| 计算语义 | 对 TensorList 中每个 Tensor 逐元素计算 `y[t][i]=cos(x[t][i])`。 |
| 输入输出 | 动态 ND TensorList；`FLOAT32/FLOAT16/BFLOAT16`；输出逐项与输入同 dtype，storage 容量至少覆盖输入。 |
| 执行类型 | 单 Kernel：模板 entry `foreach_cos<schMode>` 内启动 `SIMT` VF；无 Cube/AIC/MIX_CV 可达节点。 |
| Kernel 路线 | R-001 FP32 SIMT；R-002 FP16→FP32 SIMT；R-003 BF16→FP32 SIMT。 |
| 核心资源 | GM TensorList 指针表与 GM POD；每 block 512 SIMT threads；local memory size 设为 `UB-128 KiB`，body 无显式 shared/dynamic UB buffer；workspace 0。 |
| 来源闭包 | source_closed：公开 ACLNN 文档→`ForeachCos` OpDef/公共 infer→arch35 Tiling/POD/template key→模板 entry→SIMT VF body→ascend950 构建与 binary config 闭合；生成式 ACLNN inner bridge 作为框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

输入为 `x=[x_0,...,x_{m-1}]`。对每个列表下标 `t` 和元素下标 `i<numel(x_t)`，SIMT body 计算 `y_t[i]=cosf(float(x_t[i]))`，再转换为目标 `T`；当 `T=float` 时前后转换为恒等。公共 infer 逐项复制输入 shape/dtype；若调用者提供更大 output storage，超出输入元素数的区域不写。公开文档声明默认确定性，且只对输入值域 `[-65504,65504]` 保证功能精度。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每项 0–8 维、ND；可含空 Tensor；输入可非连续，OpDef `AutoContiguous()` | `FLOAT32/FLOAT16/BFLOAT16` | 非空指针；列表内 dtype 一致；950 config 三 dtype 均接线。 |
| `out` | TensorList；每项 0–8 维、ND；不可非连续 | 与 `x` 相同 | 非空指针；列表项数与输入一致；每项 storage 元素数至少为对应输入元素数。 |
| `workspaceSize` | 标量输出 | `uint64_t*` | arch35 Tiling 写 0。 |
| `executor` | Host 对象输出 | `aclOpExecutor**` | 第一段创建、第二段消费。 |
| `workspace` | Device 地址 | `void*` | 由于 size 0 可为空；entry 参数存在但 body 未读取。 |
| `stream` | 运行时 stream | `aclrtStream` | 第二段接口提交单 Kernel；生成式 launcher 的精确函数固定仓不可见。 |

### 入口与版本

固定源码只公开无版本后缀的 `aclnnForeachCosGetWorkspaceSize/aclnnForeachCos`。API 产品表明确列出 Ascend 950PR/950DT 支持；OpDef 注册 `ascend950`，Host CMake 将其映射到 `arch35`，Kernel CMake 仅为 `ascend950` 选择 `arch35/foreach_cos.cpp`，binary config 包含三种公开 dtype，证据一致。手写 ACLNN bridge 不在仓内；其参数封送、executor 和 launch 壳按 Skill 作为框架生成边界。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 第一段公开接口检查 `x/out` 空指针、支持 dtype、输入输出 dtype 一致、列表内部 dtype 一致、每项维数不超过 8、shape 容量约束，错误码为 `161001/161002/561002`。
2. OpDef 将输入/输出注册为动态 ND Tensor 并启用 `AutoContiguous()`；输入非连续的具体 Contiguous 节点由框架生成，输出不接受非连续 Tensor。
3. `InferShape4ForeachCommon` 要求输入实例数等于输出数，逐项复制 shape；`InferDataType4ForeachCommon` 逐项复制 dtype。
4. `ForeachCosTilingFunc` 获取 AIV core 数和 UB 字节数，两者为 0 均失败；读取动态输入实例数，超过 256 失败；读取第 0 项 dtype，逐项记录 storage `numel` 并求和。
5. Tiling 将 POD 先清零后填充；根据总元素数计算 blockDim，设置 workspace 0、local memory size 和 dtype template key。空列表会在读取第 0 个 input descriptor 时失败；列表中可包含 `numel=0` 的 Tensor。

### 参数、Workspace 与生命周期

记 `N=sum(tensorElements[t])`，平台 AIV core 数为 `C`。当 `N=0` 时 `needCoreNum=blockDim=1`；否则 `needCoreNum=blockDim=min(C,ceil(N/1024))`。每个 Device block 内 `THREAD_NUM=512`，全局线程数为 `blockDim*512`。Tiling 设置 `workspace[0]=0`，因此无用户 workspace slice。它还设置 `localMemorySize=uint32_t(UBSize-128 KiB)`；源码只检查 UB 非零而没有显式检查 `UBSize>=128 KiB`。

POD 与 TensorList descriptor 由 executor 持有并在 Kernel 执行期间有效。Device 直接从 GM POD 读取 `tensorCount/tensorElements[]`；`needCoreNum/totalElements` 不被 body 读取，blockDim 已通过 launch metadata 传递。

### 启动图与失败传播

```text
aclnnForeachCosGetWorkspaceSize(x,out)
  -> 生成式 ACLNN bridge/executor（固定仓不可见）
    -> common infer
    -> ForeachCosTilingFunc
       -> POD + template key + blockDim + localMemorySize + workspaceSize=0
aclnnForeachCos(nullptr,0,executor,stream)
  -> foreach_cos<schMode><<<blockDim, stream>>>(xList,yList,nullptr,tiling)
     -> NsForeachCos::Process<T>
        -> VF_CALL<OpForeachCosSimt<T>>(Dim3(512), ...)
```

这是一个 Device 节点，没有中间 workspace/GM 边。每个 block/thread 对每个 Tensor 以全局 grid stride 分片，写入互不重叠的 `y_t[idx]`。Tiling 中任一平台、指针、数量、dtype 或 workspace descriptor 检查失败都返回 `GRAPH_FAILED` 并阻断 launch；生成式 executor 的 cleanup/错误码映射固定仓不可见。

## Route 与参数设计

### 分核与资源计算

Host 仅以总元素数决定 block 数，不产生 former/tail core 字段：`B=1` if `N=0` else `min(C,ceil(N/1024))`。Device 对每个 Tensor 重新使用相同线程映射：`tid=blockIdx*threadNum+threadIdx`，`stride=threadNum*blockNum=B*512`，线程处理 `idx=tid,tid+stride,...<tensorElements[t]`。因此 Tensor 之间没有扁平 prefix sum，也不需要 Host 生成 per-core offset；小 Tensor 的高 `tid` 线程自然不进入内层循环。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `GET_TPL_TILING_KEY(FOREACH_COS_TPL_SCH_MODE_FLOAT)`；`schMode=0` | `dataType==DT_FLOAT` | 判断 FP16 |
| 2 | R-002 FP16 升精度 SIMT | `GET_TPL_TILING_KEY(...FLOAT16)`；`schMode=1` | 前项未命中且 `dataType==DT_FLOAT16` | 判断 BF16 |
| 3 | R-003 BF16 升精度 SIMT | `GET_TPL_TILING_KEY(...BF16)`；`schMode=2` | 前两项未命中且 `dataType==DT_BF16` | 其他 dtype 返回 `GRAPH_FAILED` |

```text
DT_FLOAT   -> template key(schMode=0) -> Process<float>
DT_FLOAT16 -> template key(schMode=1) -> Process<half>
DT_BF16    -> template key(schMode=2) -> Process<bfloat16_t>
otherwise  -> GRAPH_FAILED, no launch
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | FP32、ND、列表≤256、shape/接口约束通过、950 | 单 Kernel→SIMT VF | `foreach_cos<0>`→`Process<float>`→`OpForeachCosSimt<float>` | GM load/cosf/store；无显式 UB、barrier、atomic | `idx<count` grid-stride guard；空 Tensor skip |
| R-002 FP16 | schMode 1 | FP16，其余同 R-001 | 同上 | `foreach_cos<1>`→`Process<half>` | 每元素 `__half2float→cosf→__float2half` | 同上 |
| R-003 BF16 | schMode 2 | BF16，其余同 R-001 | 同上 | `foreach_cos<2>`→`Process<bfloat16_t>` | 每元素 `static_cast<float>→cosf→static_cast<bf16>` | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 最小每核工作量 | `B=min(C,ceil(N/1024))`，`N>0` | 避免为少量元素启动全部 AIV blocks | 全部 | `N=0` 使用 1 block | `ForeachCosTilingFunc` |
| 固定线程数 | `THREAD_NUM=512`，`LAUNCH_BOUND(512)`，`VF_CALL Dim3(512)` | thread mapping 与 stride 都依赖 512 | 全部 | 无线程数 fallback | `OpForeachCosSimt/Process` |
| TensorList 上限 | `tensorNum<=256` | POD `tensorElements[256]` | 全部 | 超过则 `GRAPH_FAILED` | `ForeachCosTilingFunc` |
| local memory 扣减 | `SetLocalMemorySize(uint32_t(ubSize-128KiB))` | 预留 `DCACHE_SIZE` 后把剩余值传给 launch metadata | 全部 | `ubSize==0` 失败；`0<ubSize<128KiB` 无显式 guard | `GetPlatformInfo/ForeachCosTilingFunc` |
| 空 Tensor | `count==0` | 避免解析数据指针并进入 loop | 全部 | `continue` 下一 Tensor | `OpForeachCosSimt` |
| dtype | 仅三分支 | entry 只实例化三种 `T` | 全部 | Host `GRAPH_FAILED`；非法 schMode 编译实例无 body 分支 | `ForeachCosTilingFunc/foreach_cos<schMode>` |

### POD/ABI

`ForeachCosTilingData` 是直接声明的自然对齐 C++ 结构：字段顺序可推导 `needCoreNum@0`、`tensorCount@4`、`totalElements@8`、`tensorElements@16`，8 B 对齐、总大小 `16+256*8=2064 B`。固定文件未声明 `packed` 或自定义序列化；迁移时必须维持该声明和目标编译 ABI。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCosTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；struct align 8；sizeof 2064 | block | `1` 或 `int32_t(min(C,ceil(N/1024)))` | body 不读取；blockDim 独立由 `SetBlockDim` 传递 | `C>0` |
| `ForeachCosTilingData` | `tensorCount` | `int32_t`/32 | offset 4 | Tensor 个数 | `static_cast<int32_t>(tensorNum)` | `Process`→VF 参数→外层 `t` 循环 | `[0,256]`；空列表先因 descriptor null 失败 |
| `ForeachCosTilingData` | `totalElements` | `int64_t`/64 | offset 8 | 元素 | 逐 Tensor `numel` 求和 | body 不读取；只供 Host 分核前留档 | 求和无显式 overflow 检查 |
| `ForeachCosTilingData` | `tensorElements` | `int64_t[256]`/每项64 | offset 16；extent 2048 B | 元素/张量 | `tensorElements[i]=storageShape.GetShapeSize()` | VF `count=tensorElements[t]` | inactive `[tensorCount,256)` 清零且不读 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | `ForeachCosTilingFunc` | `ForeachCosTilingData`、B、512 threads | `foreach_cos<0>` | `Process<float>`、`OpForeachCosSimt<float>`、`SimtGetTensorAddr` | 950 kernel/host CMake；ascend950 config |
| R-002 FP16 | schMode 1 | 同上 | 同上 | `foreach_cos<1>` | `Process<half>`、转换 helper、SIMT body | 同上 |
| R-003 BF16 | schMode 2 | 同上 | 同上 | `foreach_cos<2>` | `Process<bfloat16_t>`、转换 helper、SIMT body | 同上 |

## Kernel 实现

### 公共执行流程

模板 entry 解析 `ForeachCosTilingData`，并保留其 GM 指针；`schMode` 编译期选择 `T`。`Process<T>` 通过 `offsetof(ForeachCosTilingData,tensorElements)` 生成 GM 元素计数数组指针，读取 `tensorCount`，然后用 `VF_CALL` 启动 512-thread SIMT body。VF 对每个 Tensor 解析 TensorList 指针表；所有 blocks/threads 用全局 grid stride 对该 Tensor 的元素分片，执行 load→转 FP32→`cosf`→转回 T→store。没有第二个 Kernel、workspace 交接或显式同步节点。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachCos.md` | `aclnnForeachCos*` | FP32 且接口校验通过 | `ForeachCos` executor |
| Host/Tiling | `op_host/arch35/foreach_cos_tiling.cpp` | `ForeachCosTilingFunc` | `dataType==DT_FLOAT` | FLOAT template key |
| Router | `op_kernel/arch35/foreach_cos.cpp` | `foreach_cos<schMode>` | `schMode==TILING_KEY_FLOAT(0)` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachCosTilingFunc -> foreach_cos<0>(xList,yList,null,tiling), blockDim=B
  -> NsForeachCos::Process<float>
    -> VF_CALL<OpForeachCosSimt<float>>(Dim3(512),tensorCount,counts,x,y)
      -> SimtGetTensorAddr<float> -> ToFloat32<float> -> cosf -> FromFloat32<float>
```

##### 源码等价伪码

```text
tid = blockIdx*512 + threadIdx
stride = 512*blockNum
for t in [0,tensorCount):
  count = tensorElements[t]
  if count==0: continue
  xData = tensorListPointer(xList,t); yData = tensorListPointer(yList,t)
  for idx=tid; idx<count; idx+=stride:
    yData[idx] = cosf(xData[idx])
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList descriptor GM | `dataAddr + ((*dataAddr)>>3) + t` | 1 pointer/Tensor | `uint64_t` pointer table | framework | `SimtGetTensorAddr` | executor/kernel duration | first word is byte offset, shift 3 converts to entries |
| input GM | `xData[idx]` | 1 element/thread iteration | FP32/ND | caller | thread | caller | no staged tile |
| thread scalar | private | one `T` + two float temporaries | registers，精确字节固定仓不可见 | load/helper | cosf/store | one loop iteration | no cross-thread alias |
| output GM | `yData[idx]` | 1 element/thread iteration | FP32/ND | thread | caller | caller | unique idx per grid-stride sequence |
| local memory metadata | N/A | `U-131072` B | launch local memory | Host | body 无显式引用 | Kernel launch | fixed body does not define shared layout |
| workspace | N/A | 0 B | N/A | Host | none | none | entry parameter unused |

##### 分层内存布局

N/A：该 SIMT body 不声明 AIV `TQue/TBuf` 或 dynamic/shared UB 数组；数据从 GM 进入线程私有标量后直接写 GM。Host 仍把 local memory size 设置为 `U-128 KiB`，但固定 body 未给出其子区布局。

##### 字节账与复用顺序

```text
UB_total = N/A（无显式 AIV Queue/TBuf）
dynamic_ub/local_memory_limit = U - 131072 B
thread_private = 固定仓不可见（编译器寄存器分配）
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private `xVal/xF32/yF32` | 当前线程 load/cosf | 当前线程下一表达式 | 下一 `idx` 迭代 | 单线程程序顺序 | 不跨线程共享 |
| output `yData[idx]` | 唯一 grid-stride thread | caller | Kernel 完成后 caller 可覆盖 | stream completion | `idx=tid+k*stride` 集合互不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 线程分片 | grid stride | Host blockDim + VF 512 threads | each thread loop | output index ownership | `idx<count` 是尾部保护 |
| 数学 | `cosf` | FP32 GM load | FP32 store | thread scalar | 全程 FP32 |
| 同步 | N/A | each thread | N/A | disjoint output indices | 无 barrier/atomic；stream 完成提供最终可见性 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_cos/op_host/arch35/foreach_cos_tiling.cpp` | `GetPlatformInfo/ForeachCosTilingFunc` | B、POD、workspace/local memory、FLOAT key |
| Kernel entry/router | `foreach_cos/op_kernel/arch35/foreach_cos.cpp` | `foreach_cos<0>` | 编译期映射 `Process<float>` |
| Device body/helper | `foreach_cos/op_kernel/arch35/foreach_cos_simt.h` | `Process<float>/OpForeachCosSimt<float>` | 512 threads、指针表、grid stride、cosf、tail |

### Route R-002：FP16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachCos.md` | `aclnnForeachCos*` | FP16 且接口校验通过 | executor |
| Host/Tiling | `foreach_cos_tiling.cpp` | `ForeachCosTilingFunc` | FLOAT 分支未命中且 `DT_FLOAT16` | FLOAT16 template key |
| Router | `arch35/foreach_cos.cpp` | `foreach_cos<schMode>` | `schMode==1` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_cos<1> -> Process<half>
  -> VF_CALL<OpForeachCosSimt<half>>(512 threads)
    -> SimtGetTensorAddr<half>
    -> ToFloat32<half>(__half2float)
    -> cosf
    -> FromFloat32<half>(__float2half)
```

##### 源码等价伪码

```text
for each tensor t on every thread:
  skip if count==0
  for idx=globalTid; idx<count; idx+=B*512:
    h = xData[idx]
    f = __half2float(h)
    r = cosf(f)
    yData[idx] = __float2half(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `xData[idx]`/`yData[idx]` | one FP16 each iteration | FP16/ND | caller/thread | thread/caller | caller | disjoint grid-stride indices |
| thread scalar | private | one half + FP32 intermediates | registers，精确字节不可见 | load/conversion/cosf | conversion/store | one iteration | overwritten next idx |
| local memory metadata | N/A | `U-131072` B | launch local memory | Host | no explicit body consumer | launch | no declared shared layout |
| workspace | N/A | 0 B | N/A | Host | none | none | unused |

##### 分层内存布局

N/A：SIMT 标量直通 GM，不使用 AIV UB Queue；local memory 仅由 Host 设置上限，body 没有可列出的子区。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub/local_memory_limit = U - 131072 B
thread_private = 固定仓不可见
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private FP32 scalar | `__half2float`/`cosf` | next expression/`__float2half` | next idx | thread program order | each thread owns its scalars |
| `yData[idx]` | unique thread | caller | post-stream | stream completion | stride partition disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load/convert | `__half2float` | FP16 GM | `cosf` | thread scalar | 提升到 FP32 |
| compute/store | `cosf`→`__float2half` | FP32 scalar | FP16 GM | output element | 转回 FP16 的具体 round mode 固定 helper 未显式命名 |
| tail/sync | `idx<count`；无 barrier/atomic | thread mapping | store | bounds/index ownership | 空 Tensor `continue` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_cos/op_host/arch35/foreach_cos_tiling.cpp` | `ForeachCosTilingFunc` | FP16 key、B 和 POD |
| Kernel entry/router | `foreach_cos/op_kernel/arch35/foreach_cos.cpp` | `foreach_cos<1>` | `Process<half>` |
| Device body/helper | `foreach_cos/op_kernel/arch35/foreach_cos_simt.h` | `ToFloat32<half>/FromFloat32<half>/OpForeachCosSimt` | conversion、cosf 与 grid stride |

### Route R-003：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `docs/aclnnForeachCos.md` | `aclnnForeachCos*` | BF16 且接口校验通过 | executor |
| Host/Tiling | `foreach_cos_tiling.cpp` | `ForeachCosTilingFunc` | 前两分支未命中且 `DT_BF16` | BF16 template key |
| Router | `arch35/foreach_cos.cpp` | `foreach_cos<schMode>` | `schMode==2` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_cos<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachCosSimt<bfloat16_t>>(512 threads)
    -> SimtGetTensorAddr<bfloat16_t>
    -> ToFloat32<bfloat16_t>(static_cast<float>)
    -> cosf
    -> FromFloat32<bfloat16_t>(static_cast<bfloat16_t>)
```

##### 源码等价伪码

```text
for t=0..tensorCount-1:
  if tensorElements[t]==0: continue
  for idx=globalTid; idx<tensorElements[t]; idx+=B*512:
    f = static_cast<float>(xData[idx])
    r = cosf(f)
    yData[idx] = static_cast<bfloat16_t>(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `xData[idx]`/`yData[idx]` | one BF16 each iteration | BF16/ND | caller/thread | thread/caller | caller | grid-stride ownership |
| thread scalar | private | BF16 value + FP32 intermediates | registers，精确字节不可见 | load/casts/cosf | cast/store | one iteration | no sharing |
| local memory metadata | N/A | `U-131072` B | launch local memory | Host | no explicit body consumer | launch | no subregion |
| workspace | N/A | 0 B | N/A | Host | none | none | unused |

##### 分层内存布局

N/A：与 R-002 相同，数据不经 AIV Queue/TBuf；固定 body 只出现 GM 与线程私有标量。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub/local_memory_limit = U - 131072 B
thread_private = 固定仓不可见
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 scalar | BF16 cast/cosf | cosf/BF16 cast | next idx | thread program order | thread private |
| output index | unique thread | caller | post-stream | stream completion | grid-stride index sets disjoint |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| convert | `static_cast<float>` | BF16 load | cosf | thread scalar | FP32 computation |
| downconvert | `static_cast<bfloat16_t>` | cosf | GM store | output element | 具体 rounding 语义由仓外 BF16 类型实现决定，固定仓不可见 |
| tail/sync | `idx<count`；无 barrier/atomic | thread mapping | store | bounds/index ownership | empty Tensor skip |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_cos/op_host/arch35/foreach_cos_tiling.cpp` | `ForeachCosTilingFunc` | BF16 key、B、counts |
| Kernel entry/router | `foreach_cos/op_kernel/arch35/foreach_cos.cpp` | `foreach_cos<2>` | `Process<bfloat16_t>` |
| Device body/helper | `foreach_cos/op_kernel/arch35/foreach_cos_simt.h` | BF16 conversion helpers/VF | FP32 cosf、grid stride、bounds |

## 算子特定优化

### Trick T-001：每个 Tensor 复用统一全局 grid-stride 映射

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：`count>0`；Host 已产生 `B=blockDim`，每 block 固定 512 threads。
- Golden 来源：[`foreach_cos_simt.h:88-107`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_kernel/arch35/foreach_cos_simt.h#L88-L107)

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
```

#### 机制与设计原因

Host 无需为 256 个 Tensor 生成 prefix sum 或 per-core 首尾 offset；每个 Tensor 都重新以 `tid` 和 `B*512` 划分自己的线性元素。每个输出 index 只属于一个线程，省去 atomic 与 block barrier；空 Tensor 在解析数据地址前跳过。代价是所有线程都串行遍历 TensorList，极小 Tensor 上只有低 tid 线程工作，这是源码可见的调度边界而非实测性能结论。

#### 迁移边界

- 适用：TensorList 内各元素完全独立、每项可线性寻址、无需跨 Tensor 归约的 unary 运算。
- 前提：Host 提供准确的 `tensorCount/counts`，TensorList 指针表编码一致，所有线程使用同一 `blockNum/threadNum`。
- 失效：存在跨元素状态、输出地址别名或需要 Tensor 间同步时必须更换分片方案。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：低精度标量只在寄存器中升到 FP32

#### 触发与源码依据

- 适用 Route：R-002、R-003。
- 触发条件：模板 `T` 为 `half` 或 `bfloat16_t`。
- Golden 来源：[`foreach_cos_simt.h:103-107`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_cos/op_kernel/arch35/foreach_cos_simt.h#L103-L107)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    T xVal = xData[idx];
    float xF32 = ToFloat32<T>(xVal);
    float yF32 = cosf(xF32);
    yData[idx] = FromFloat32<T>(yF32);
}
```

#### 机制与设计原因

每个线程 load 一个低精度元素后在私有标量中转换、调用 FP32 `cosf` 并立刻降回目标类型，没有将整 tile 的 FP32 中间量物化到 UB 或 GM。它同时固定了精度顺序为“升精度→cosf→降精度”；转换 helper 的确切 rounding 语义不在当前文件中显式给出。

#### 迁移边界

- 适用：可逐元素计算且目标数学函数提供 FP32 标量实现的 FP16/BF16 unary 算子。
- 前提：转换 helper 和 `cosf` 在目标 SIMT 工具链可用，输出元素与线程一一独占。
- 失效：需要向量级共享中间量、特定 rounding mode 或跨元素计算时不能直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0，第一 dtype 分支 | schMode 1，第二分支 | schMode 2，第三分支 | Host dtype selector |
| 模板与 dtype | `Process<float>` | `Process<half>` | `Process<bfloat16_t>` | compile-time specialization |
| 循环与分核 | B blocks×512，per-Tensor grid stride | 同左 | 同左 | shape 不改变 body |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程标量；无 Queue/TBuf | half + FP32 标量 | BF16 + FP32 标量 | local memory metadata相同；寄存器字节不可见 |
| 内存总占用与复用 | GM 直通 | GM 直通 | GM 直通 | 不物化 tile |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT load/cosf/store | SIMT load/cast/cosf/cast/store | 同左，BF16 casts | 三者均 SIMT |
| 同步、精度与 tail | 无 barrier；FP32 | 无 barrier；FP32 compute | 无 barrier；FP32 compute | `idx<count`；空 Tensor skip |
| fallback/失败边界 | unsupported dtype Host fail | 同左 | 同左 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持，见无后缀卡 | 支持，见无后缀卡 | 支持；公开文档、OpDef、CMake/config 一致 |
| Host/Launcher | arch22 common Tiling，workspace 32 | 同 A2 | arch35 专用 Tiling，workspace 0，B≈每 1024 元素一 core |
| Kernel/资源 | AIV Vector Queue/Cos primitive | 同 A2 | SIMT 512-thread VF，GM→线程标量→GM |

## 限制与失败边界

- TensorList 最多 256 项；超过即 `GRAPH_FAILED`。列表可包含空 Tensor并跳过，但零项空列表因 `GetDynamicInputDesc(0,0)` 返回空而失败。
- `totalElements` 的 `int64_t` 求和无显式溢出检查；`needCoreNum`/blockDim 由总元素数计算，但 Device 只使用实际 `GetBlockNum()`，不读取 POD 同名字段。
- Host 只检查 `ubSize!=0`，随后执行无符号 `ubSize-128 KiB` 并窄化为 `uint32_t`；固定源码假设平台 UB 足够，未提供不足时的 fallback。
- TensorList 指针表格式要求首个 `uint64_t` 是到地址数组的字节 offset，Device 以 `offset>>3` 定位表；改变 descriptor ABI 会破坏寻址。
- FP16 的 `__float2half` 和 BF16 `static_cast` 的确切 rounding/异常语义、线程寄存器占用、SIMT `VF_CALL` 库内实现、生成式 ACLNN bridge/executor 均在固定仓不可见。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_cos/docs/aclnnForeachCos.md` | `aclnnForeachCosGetWorkspaceSize/aclnnForeachCos` | 公开原型、950 支持、接口约束 |
| 入口实现 | `foreach/foreach_cos/op_host/foreach_cos_def.cpp` | `ops::ForeachCos` | ascend950 注册、I/O dtype/layout/AutoContiguous |
| Host/Tiling/Launcher | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon/InferDataType4ForeachCommon` | 列表输出推导 |
| Host/Tiling/Launcher | `foreach/foreach_cos/op_host/arch35/foreach_cos_tiling.cpp` | `GetPlatformInfo/ForeachCosTilingFunc` | 校验、POD、B、workspace/local memory、key |
| Host/Tiling/Launcher | `foreach/foreach_cos/op_kernel/arch35/foreach_cos_tiling_data.h` | `ForeachCosTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_cos/op_kernel/arch35/foreach_cos_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 模板选择域 |
| Kernel entry/body | `foreach/foreach_cos/op_kernel/arch35/foreach_cos.cpp` | `foreach_cos<schMode>` | template router |
| Kernel entry/body | `foreach/foreach_cos/op_kernel/arch35/foreach_cos_simt.h` | `Process/OpForeachCosSimt/SimtGetTensorAddr` | 512-thread SIMT、grid stride、转换与 cosf |
| 构建接线 | `foreach/foreach_cos/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend950 选择 arch35 entry |
| 构建接线 | `foreach/foreach_cos/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35 Host |
| 构建接线 | `foreach/foreach_cos/op_host/config/ascend950/foreach_cos_binary.json` | `op_list` | 950 三 dtype binary |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B direct struct，counts 内嵌 256 项 | 字段顺序、自然对齐、`offsetof(tensorElements)=16`、GM lifetime | `ForeachCosTilingData`、`Process` |
| 调用顺序 | template entry→Process→VF_CALL→per-Tensor grid stride | `schMode` 与 `T` 映射、512 threads | `foreach_cos<schMode>`、`Process<T>` |
| workspace/内存 | workspace 0；GM 直通线程标量 | local memory 扣 128 KiB；不可臆造 Queue/shared buffer | `ForeachCosTilingFunc`、SIMT body |
| 同步 | disjoint grid-stride 输出无需 atomic/barrier | 不能让线程输出 index 集合重叠；最终由 stream 完成同步 | `OpForeachCosSimt` |
| 精度与尾块 | half/BF16→FP32→cosf→原 dtype | 保留转换顺序、`idx<count` 和 empty skip | `ToFloat32/FromFloat32/OpForeachCosSimt` |
| 失败边界 | Host 拒绝 >256、unsupported dtype、空平台资源 | 空列表、UB 扣减假设、外部 bridge/VF_CALL 边界 | `ForeachCosTilingFunc` |