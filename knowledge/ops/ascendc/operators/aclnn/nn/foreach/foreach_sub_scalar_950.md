---
type: Operator
title: "ForeachSubScalar 950 ACLNN 算子知识"
description: "说明 aclnnForeachSubScalarV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/docs/aclnnForeachSubScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/op_api/foreach_sub_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/foreach_sub_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/arch35/foreach_sub_scalar_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_host/config/ascend950/foreach_sub_scalar_binary.json
    role: implementation
created_at: "2026-08-25T16:22:25Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSubScalar 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSubScalarV2GetWorkspaceSize` + `aclnnForeachSubScalarV2`，公开 ACLNN 两段式接口；旧 `aclnnForeachSubScalar` 的文档明确不支持 Ascend 950，950 只取 V2。 |
| 计算语义 | 对 TensorList 中每个有效元素执行 `y_i[j]=x_i[j]-scalar`。 |
| 输入输出 | `x/out`：ND TensorList，FP32/FP16/INT32/BF16；`scalar`：`aclScalar`，executor 转为单元素 Tensor。 |
| 执行类型 | 单 Kernel `SIMT`；每个 AIV block 内 1024 threads，所有可达节点无 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | R-001 FP32；R-002 FP16（FP32计算）；R-003 INT32；R-004 BF16（FP32计算）。 |
| 核心资源 | TensorList GM 指针表、POD 内 `tensorElements[256]`、SIMT grid-stride；无显式 AIV Queue/TBuf、shared/dynamic UB、atomic 或 device workspace。 |
| 来源闭包 | `source_closed`：V2 API→L0 launcher→arch35 tiling/POD/template key→SIMT entry/body→ascend950 binary/CMake 闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_(n-1)]`，对每个 `0<=i<n` 和 `0<=j<numel(x_i)` 写 `y_i[j]=x_i[j]-s`。FP32 使用 FP32 计算；INT32 使用 INT32 计算；FP16/BF16 都通过 `ComputeType<T>::type=float` 在标量表达式中升至 FP32，再 cast 为 T 写 GM。Tensor 数为 0 时 ACLNN 第一段直接成功返回；单个零元素 Tensor 在 SIMT 中跳过。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList，每项 0–8 维，ND | FP32、FP16、INT32、BF16，列表内一致 | V2 公开契约不支持非连续；实现第一段仍逐项调用框架 `Contiguous`。950 Tiling最多 256 项。 |
| `scalar` | `aclScalar` | x=FP32/BF16 时 FP32/DOUBLE；x=FP16 时 FP16/DOUBLE；x=INT32 时 INT32/INT64 | 转为目标 Tensor：BF16 route用 FP32，其余用 x dtype。 |
| `out` | TensorList，逐项 shape/storage format 与 x 相等 | 与 x 一致 | 必须连续；列表长度由 shape逐项校验的访问隐式要求与 x 对应。 |
| `workspaceSize` | 标量输出 | `uint64_t` | arch35 核心算子 `workspace[0]=0`；executor 仍可能为框架前处理分配空间，具体汇总公式固定仓不可见。 |
| `executor` | Host 对象 | `aclOpExecutor**` | 持有前处理 Tensor、单 Kernel任务与生命周期。 |

### 入口与版本

`aclnnForeachSubScalarV2` 将旧入口的 Tensor scalar 改为 `aclScalar`。公开 V2 产品表列 Ascend 950PR/950DT 为支持，而旧 `aclnnForeachSubScalar` 表中为不支持；因此 950 只有 V2 可发布。V2 经 `ConvertToTensor` 复用内部 op type `ForeachSubScalar`，其 OpDef、arch35 tiling、kernel与 ascend950 binary均在固定 commit 接通。

## Host/Launcher 执行流程

### 参数校验与前处理

`CheckParams` 顺序为 null→平台/dtype→shape→format。950 由 `IsRegbase(curArch)` 进入 dtype支持列表；非空列表要求所有 x/out dtype相同，按 x dtype限制 scalar；逐个 x/out shape相等、x维数≤8、storage format一致且非私有。空 TensorList 返回 `workspaceSize=0`。非空时，框架为每个 x 建 `Contiguous` 结果并组成新列表，再把 scalar 转为单元素 Tensor；任一步失败都不会添加核心 Kernel。

### 参数、Workspace 与生命周期

L0 使用 `ADD_TO_LAUNCHER_LIST_AICORE(ForeachSubScalar, OP_INPUT(x, scalar), OP_OUTPUT(out))` 添加一个节点。arch35 tiling令核心算子 workspace为 0；entry也完全忽略 `workspace`。第一段返回 executor 汇总 workspace，第二段由 `CommonOpExecutorRun` 按调用者提供的 stream 执行；TensorList指针描述、scalar Tensor和out必须保持有效至 stream完成。Contiguous内部切片、scalar Tensor存储及自动生成 launcher ABI是框架边界，固定仓不可见。

### 启动图与失败传播

```text
aclnnForeachSubScalarV2GetWorkspaceSize
  -> N × l0op::Contiguous(x_i) [框架边界]
  -> ConvertToTensor(scalar, FP32/FP16/INT32) [框架边界]
  -> ADD_TO_LAUNCHER_LIST_AICORE(ForeachSubScalar)
       -> ForeachSubScalarTilingFunc
       -> foreach_sub_scalar<schMode><<<needCoreNum, stream>>>
            -> VF_CALL<OpForeachSubScalarSimt<T,S>>(Dim3(1024), ...)
```

唯一 Device节点的 `blockDim=needCoreNum`，线程数固定 1024；POD、x/scalar/y直接传入。无 Kernel间边或 workspace slice。框架保证同一 executor/stream 中前处理先于核心节点；任何 API校验、前处理、launcher添加、tiling失败都阻断启动。Tiling检查平台信息、TensorList metadata、POD指针、`tensorNum<=256`、workspace指针及 dtype；失败返回 `GRAPH_FAILED`。

## Route 与参数设计

### 分核与资源计算

Tiling从动态输入0读取 `tensorNum` 与每项 storage shape size，累加 `totalElements`。若总元素数为 0，则 `needCoreNum=blockDim=1`；否则 `needCoreNum=min(aivCoreNum, ceil(totalElements/1024))`。这里的每核最少1024元素只决定 block数，Device没有静态每核切片：SIMT thread使用全局 `tid=blockIdx*threadNum+threadIdx`，步长 `threadNum*blockNum`，对每个 Tensor独立 grid-stride。

Tiling将 `SetLocalMemorySize(ubSize-128 KiB)` 写入 launch metadata，并拒绝 `ubSize==0`；SIMT body没有显式 dynamic/shared UB指针或局部数组，故该预算在固定 body中无可定位的 consumer。workspace为0。dtype经单函数决策树映射到模板参数 `schMode` 0/1/2/3。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 | schMode 0；`Process<float,float>` | `dataType==DT_FLOAT` | else检查 FP16 |
| 2 | R-002 FP16 | schMode 1；`Process<half,half>` | `dataType==DT_FLOAT16` | else检查 INT32 |
| 3 | R-003 INT32 | schMode 2；`Process<int32_t,int32_t>` | `dataType==DT_INT32` | else检查 BF16 |
| 4 | R-004 BF16 | schMode 3；`Process<bfloat16_t,float>` | `dataType==DT_BF16` | 其余 dtype `GRAPH_FAILED` |

```text
if FP32: schMode=0 -> R-001
else if FP16: schMode=1 -> R-002
else if INT32: schMode=2 -> R-003
else if BF16: schMode=3 -> R-004
else: tiling failure
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode0 | 950、FP32、tensorNum≤256、API校验通过 | 单 SIMT节点，blockDim×1024 threads | `Process<float,float>`→`OpForeachSubScalarSimt<float,float>` | 仅GM标量load/元素load/store；无barrier/atomic | `idx<count` grid-stride；无 fallback |
| R-002 FP16 | schMode1 | 950、FP16、其余同上 | 同上 | `Process<half,half>`，`ComputeType<half>=float` | GM读写half，表达式FP32 | cast回half；尾线程由循环条件屏蔽 |
| R-003 INT32 | schMode2 | 950、INT32 | 同上 | `Process<int32_t,int32_t>` | INT32 GM直算 | 无宽化/饱和；边界条件同上 |
| R-004 BF16 | schMode3 | 950、BF16 | 同上 | `Process<bfloat16_t,float>`，compute FP32 | scalar GM为FP32，x/y为BF16 | cast回BF16；边界条件同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor数上限 | `tensorNum<=256` | POD数组固定256项 | 全部 | Tiling失败 | `MAX_TENSOR_NUM`、`ForeachSubScalarTilingData::tensorElements` |
| 每block目标工作量 | 非空时 `ceil(totalElements/1024)` 再与AIV核数取min | 每block固定1024 threads，减少超过元素数的block | 全部 | 总元素0固定1 block | `ForeachSubScalarTilingFunc` |
| DCache预留 | `SetLocalMemorySize(ubSize-128*1024)` | Host为arch35 local-memory metadata预留128 KiB DCache | 全部 | `ubSize==0`失败；源码未检查 `ubSize<128KiB` 的无符号下溢 | `DCACHE_SIZE`、`SetLocalMemorySize` |
| dtype key封闭 | 仅四种 `dataType` | entry只有4个 `if constexpr`实例 | 全部 | unsupported dtype直接Tiling失败 | `ForeachSubScalarTilingFunc` |
| 空域 | `totalElements==0`→1 block；每 Tensor `count==0`→continue | 保持合法launch且避免无效GM访问 | 全部 | N/A | Tiling与`OpForeachSubScalarSimt` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSubScalarTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；struct align 8、sizeof 2064 | block | 0域写1；否则`min(coreNum,ceil(total/1024))` | body不直接读取；同值写入`SetBlockDim` | 必须≥1，窄化前≤AIV coreNum |
| 同上 | `tensorCount` | `int32_t`/32 | offset 4 | Tensor | `static_cast<int32_t>(tensorNum)` | `Process`→SIMT外层t循环 | 0–256 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | element | 逐shape累加 | body不直接读取，仅Host分核 | 累加溢出未显式检查 |
| 同上 | `tensorElements` | `int64_t[256]` | offset 16，extent 2048 B | element/Tensor | 每项storage shape size；memset先清零 | `tensorElements[t]` | 非活动尾项为0且不读取 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode0 | `ForeachSubScalarTilingFunc` | `ForeachSubScalarTilingData` | `foreach_sub_scalar<0>` | `Process<float,float>`/SIMT body | ascend950 CMake + binary config + OpDef |
| R-002 FP16 | schMode1 | 同上 | 同上 | `foreach_sub_scalar<1>` | `Process<half,half>` | 同上 |
| R-003 INT32 | schMode2 | 同上 | 同上 | `foreach_sub_scalar<2>` | `Process<int32_t,int32_t>` | 同上 |
| R-004 BF16 | schMode3 | 同上 | 同上 | `foreach_sub_scalar<3>` | `Process<bfloat16_t,float>` | 同上 |

## Kernel 实现

### 公共执行流程

entry以模板参数 `schMode` 编译期选择 `(T,S)`。`Process` 从 scalar GM读取一次 `S scalarVal`，通过 POD内 `tensorElements` 的 GM offset得到元素数数组，再发起固定1024线程的 `VF_CALL`。每个SIMT thread对所有 Tensor循环；TensorList描述首个64位值给出指针表相对偏移，`(offset>>3)`定位指针数组，再取第t个数据地址。对于每个非空 Tensor，global thread id从自身位置起以全grid线程数为步长读 `xData[idx]`、计算并写 `yData[idx]`。没有跨thread共享数据、barrier、atomic或跨Kernel边。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | `CheckDtypeValid` | x=FP32，scalar=FLOAT/DOUBLE | scalar转FP32 Tensor |
| Tiling | `op_host/arch35/foreach_sub_scalar_tiling_arch35.cpp` | `ForeachSubScalarTilingFunc` | `dataType==DT_FLOAT` | schMode0 |
| Router | `op_kernel/arch35/foreach_sub_scalar.cpp` | `foreach_sub_scalar<0>` | constexpr float key | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar<0> -> Process<float,float>
  -> VF_CALL<OpForeachSubScalarSimt<float,float>>(Dim3(1024), tensorCount, elemCounts, x, y, scalarVal)
    -> SimtGetTensorAddr<float> -> grid-stride load/sub/store
```

##### 源码等价伪码

```text
scalarVal = scalarGM[0]
tid = blockIdx*1024 + threadIdx; stride = 1024*blockNum
for t in [0,tensorCount):
  count=tensorElements[t]; if count==0: continue
  xData/yData = tensorListPointer(t)
  for idx=tid; idx<count; idx+=stride:
    yData[idx] = float(xData[idx]) - float(scalarVal)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | pointer[t]+idx | one FP32/thread/iteration | ND contiguous | caller/executor | SIMT thread | stream完成前 | idx<count尾保护 |
| scalar GM/register | scalar[0] | one FP32/block Process值 | FP32 | executor/Process | all threads via VF_CALL参数 | Kernel | 无GM重复按元素加载 |
| POD GM | offset16+8t | 2064 B struct | standard layout | Tiling | Process/SIMT | launch | 256项固定数组 |

##### 分层内存布局

N/A：该 SIMT body不声明 AIV Queue/TBuf、shared/dynamic UB或局部数组；实际路径是逐thread GM load→寄存器表达式→GM store。Host仅设置 `localMemorySize=U-128KiB` 元数据，固定 body中无显式子区。

##### 字节账与复用顺序

```text
UB_total = N/A（无显式 AIV UB）
dynamic_ub = N/A（VF_CALL仅给 Dim3(1024)，无dynUB参数）
workspace_total = 0 B（核心算子）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread寄存器值 | GM load | 当前Sub/store | 下一idx迭代 | 单线程程序顺序 | 各线程idx序列互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| load-sub-store | 单线程程序顺序；无barrier/atomic | x/scalar load | y store | 当前idx | FP32；`idx<count`保护尾部；grid-stride确保唯一写者 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sub_scalar_tiling_arch35.cpp` | `ForeachSubScalarTilingFunc` | blockDim/POD/schMode0 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_scalar.cpp` | `foreach_sub_scalar<0>` | FP32实例 |
| Device body/helper | `op_kernel/arch35/foreach_sub_scalar_simt.h` | `OpForeachSubScalarSimt<float,float>` | 1024线程grid-stride |

### Route R-002：FP16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | scalar conversion | x=FP16，scalar=FP16/DOUBLE | scalar转FP16 Tensor |
| Tiling | arch35 tiling | dtype tree | `DT_FLOAT16` | schMode1 |
| Router | arch35 entry | `foreach_sub_scalar<1>` | constexpr key | `Process<half,half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar<1> -> Process<half,half>
  -> VF_CALL<OpForeachSubScalarSimt<half,half>> -> ComputeType<half>::type=float
```

##### 源码等价伪码

```text
同 R-001 线程映射；每个idx执行
y[idx] = half(float(x[idx]) - float(scalarVal));
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | pointer[t]+idx | one half | FP16 | caller/SIMT | SIMT/store | iteration | idx guard |
| scalar/register | scalar[0] | one half读值，表达式cast FP32 | FP16/FP32 compute | Process | SIMT expression | Kernel | 线程共享值参数 |

##### 分层内存布局

N/A：无显式 AIV UB/shared/dynamic UB；每线程只保留当前 half、FP32表达式和输出值。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread临时 | half load/cast | FP32 Sub/cast/store | 下一idx | 单线程顺序 | idx唯一 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 表达式 | C++ static_cast链 | half load | half store | 当前idx | x/scalar升FP32相减，再static_cast<half>；源码未指定独立RoundMode；idx<count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 tiling | `ForeachSubScalarTilingFunc` | FP16→schMode1 |
| Kernel entry/router | arch35 entry | `foreach_sub_scalar<1>` | half/half |
| Device body/helper | arch35 simt | `ComputeType<half>`、SIMT body | FP32表达式与tail guard |

### Route R-003：INT32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | V2实现 | `CheckDtypeValid` | x=INT32，scalar=INT32/INT64 | scalar转INT32 Tensor |
| Tiling | arch35 tiling | dtype tree | `DT_INT32` | schMode2 |
| Router | arch35 entry | `foreach_sub_scalar<2>` | constexpr key | `Process<int32_t,int32_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar<2> -> Process<int32_t,int32_t>
  -> VF_CALL<OpForeachSubScalarSimt<int32_t,int32_t>> -> grid-stride int32 load/sub/store
```

##### 源码等价伪码

```text
同 R-001；y[idx]=int32_t(int32_t(x[idx])-int32_t(scalarVal))。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y/scalar GM与寄存器 | pointer[t]+idx/scalar[0] | 每thread一个INT32 | INT32/ND | executor | SIMT | iteration/Kernel | idx guard、无共享缓冲 |

##### 分层内存布局

N/A：无显式 AIV UB/shared/dynamic UB，INT32直接在thread表达式中计算。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread寄存器 | GM load | Sub/store | 下一idx | 程序顺序 | 唯一写者 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | 无barrier/atomic | INT32 load | INT32 store | 当前idx | 无宽化/饱和逻辑；idx<count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 tiling | dtype tree | INT32→2 |
| Kernel entry/router | arch35 entry | `foreach_sub_scalar<2>` | INT32实例 |
| Device body/helper | arch35 simt | generic `ComputeType<int32_t>` | C=int32_t、grid-stride |

### Route R-004：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | V2实现 | scalar conversion | x=BF16、scalar=FLOAT/DOUBLE | scalar转FP32 Tensor |
| Tiling | arch35 tiling | dtype tree | `DT_BF16` | schMode3 |
| Router | arch35 entry | `foreach_sub_scalar<3>` | constexpr key | `Process<bfloat16_t,float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sub_scalar<3> -> Process<bfloat16_t,float>
  -> VF_CALL<OpForeachSubScalarSimt<bfloat16_t,float>> -> ComputeType<bf16>::type=float
```

##### 源码等价伪码

```text
同 R-001 线程映射；scalarVal已是FP32：
y[idx] = bfloat16_t(float(x[idx]) - float(scalarVal));
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | pointer[t]+idx | 一个BF16 | BF16/ND | caller | SIMT | iteration | idx guard |
| scalar GM/register | scalar[0] | 一个FP32 | FP32 | executor/Process | all threads | Kernel | 单次Process读取 |

##### 分层内存布局

N/A：无显式 AIV UB/shared/dynamic UB；当前 BF16在thread表达式内升FP32后回写。

##### 字节账与复用顺序

```text
UB_total=N/A; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread临时 | BF16 load/cast | FP32 Sub/BF16 store | 下一idx | 程序顺序 | 唯一idx |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 表达式 | static_cast链，无barrier | BF16/FP32 load | BF16 store | 当前idx | FP32相减后cast BF16；具体硬件round mode固定仓未显式命名；idx<count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 tiling | dtype tree | BF16→3 |
| Kernel entry/router | arch35 entry | `foreach_sub_scalar<3>` | BF16/FP32 scalar实例 |
| Device body/helper | arch35 simt | `ComputeType<bfloat16_t>` | FP32表达式 |

## 算子特定优化

### Trick T-001：一份 Tensor 元数据驱动全 grid 的跨列表 grid-stride

#### 触发与源码依据

- 适用 Route：R-001–R-004。
- 触发条件：`tensorCount<=256` 且 blockDim由总元素数产生。
- Golden 来源：[`foreach_sub_scalar_simt.h:54-70`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar_simt.h#L54-L70)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t t = 0; t < tensorCount; t++) {
    int64_t count = tensorElements[t];
    if (count == 0) {
        continue;
    }

    __gm__ T* xData = SimtGetTensorAddr<T>(xList, t);
    __gm__ T* yData = SimtGetTensorAddr<T>(yList, t);
```

#### 机制与设计原因

一次 Kernel launch覆盖整个 TensorList；所有thread按相同 Tensor顺序读取 POD元素数，但每个 Tensor内用 global tid/全grid stride分摊元素。这样不需要为每个 Tensor单独启动 Kernel，也不需要 Host生成每核起止表。空 Tensor直接跳过，非整除尾部由 `idx<count`自然保护。每个 `(t,idx)`只被一个thread访问，因此无需 atomic或barrier。

#### 迁移边界

- 适用：TensorList各项执行相同独立逐元素运算，元数据可放入固定POD。
- 前提：Tensor数≤256；各Tensor地址可由统一TensorList描述解析；不同Tensor之间无依赖。
- 失效：跨Tensor归约、需要逐Tensor不同属性或POD超限时应拆任务/扩展ABI。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：scalar 在 VF_CALL 前单次读取并按值传入线程体

#### 触发与源码依据

- 适用 Route：R-001–R-004。
- 触发条件：单一 scalar Tensor、任一支持schMode。
- Golden 来源：[`foreach_sub_scalar_simt.h:80-96`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar_simt.h#L80-L96)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__gm__ S* scalarGm = reinterpret_cast<__gm__ S*>(scalar);
S scalarVal = *scalarGm;

__gm__ int64_t* elemCounts = reinterpret_cast<__gm__ int64_t*>(
    reinterpret_cast<__gm__ char*>(
        const_cast<__gm__ ForeachSubScalarTilingData*>(tilingGm)) +
    offsetof(ForeachSubScalarTilingData, tensorElements));
```

#### 机制与设计原因

`Process` 在进入 `VF_CALL` 前把 scalar从GM读成值参数，SIMT线程的内层元素循环不再逐元素解引用 scalar GM。BF16路线令S=float；FP16路线S=half但表达式再cast到`ComputeType<half>=float`。元素数数组则直接以 POD GM offset传给线程体，避免复制256项元数据到UB。

#### 迁移边界

- 适用：整个 launch共享一个标量、元数据只读。
- 前提：scalar值在Kernel期间不变；值参数ABI支持S；POD保持standard-layout及offset 16。
- 失效：逐Tensor/逐元素scalar时必须传数组并在相应索引加载。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | sch0 | sch1 | sch2 | sch3 | dtype决策树 |
| 模板与 dtype | T/S/C=float | half/half/float | int32/int32/int32 | bf16/float/float | `ComputeType`决定C |
| 循环与分核 | blockDim×1024 grid-stride | 同 | 同 | 同 | 总元素决定block数 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread寄存器；无显式UB | 同 | 同 | 同 | SIMT直访GM |
| 内存总占用与复用 | workspace0 | 同 | 同 | 同 | 无中间物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT load/store | 同 | 同 | 同 | 无Vector/NDDMA/通信 |
| 同步、精度与 tail | FP32，无barrier，idx guard | FP32 compute→half | INT32直算 | FP32 compute→BF16 | 每idx唯一写者 |
| fallback/失败边界 | 无 | 无 | 无 | 无 | unsupported dtype tiling失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2公开支持 | V2公开支持 | V2公开支持；旧API明确不支持 |
| Host/Launcher | arch22公共tiling | arch22公共tiling | arch35专用tiling与POD |
| Kernel/资源 | AIV Vector Queue/cast buffer | 同 | SIMT 1024 threads，无显式UB/workspace |

## 限制与失败边界

- `tensorNum>256`、平台信息/shape/POD/workspace指针为空、AIV core数或UB大小为0、dtype不在四种集合时Tiling失败；空总域以1 block合法启动并由`count==0`跳过。
- `totalElements`为`int64_t`逐项累加且固定源码未见溢出检查；`SetLocalMemorySize(static_cast<uint32_t>(ubSize-128KiB))`也未显式检查UB小于128KiB的下溢，复用时必须保留平台前提。
- FP16/BF16通过C++ `static_cast<T>`降精度，固定源码未命名硬件round mode；不得擅自声明为与arch22 `CAST_RINT`完全相同。
- executor内部Contiguous/scalar Tensor workspace布局与自动生成模板key编码属于框架边界，固定仓不可见；核心SIMT节点workspace为0。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sub_scalar/docs/aclnnForeachSubScalarV2.md` | V2两段接口 | 950公开支持、dtype/shape契约 |
| 入口实现 | `foreach/foreach_sub_scalar/op_host/op_api/aclnn_foreach_sub_scalar_v2.cpp` | `ExecForeachSubScalarV2GetWorkspaceSize` | API校验、前处理、executor |
| 入口实现 | `foreach/foreach_sub_scalar/op_host/op_api/foreach_sub_scalar_v2.cpp` | `l0op::ForeachSubScalarV2` | 单节点launcher |
| Host/Tiling/Launcher | `foreach/foreach_sub_scalar/op_host/foreach_sub_scalar_def.cpp` | `ForeachSubScalar` | ascend950 OpDef接线与dtype |
| Host/Tiling/Launcher | `foreach/foreach_sub_scalar/op_host/arch35/foreach_sub_scalar_tiling_arch35.cpp` | `ForeachSubScalarTilingFunc` | POD、blockDim、key、workspace/local memory |
| Host/Tiling/Launcher | `foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar_tiling_data.h` | `ForeachSubScalarTilingData` | POD ABI |
| Kernel entry/body | `foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar.cpp` | `foreach_sub_scalar<schMode>` | compile-time router |
| Kernel entry/body | `foreach/foreach_sub_scalar/op_kernel/arch35/foreach_sub_scalar_simt.h` | `Process/OpForeachSubScalarSimt` | 线程映射、地址、精度、tail |
| 构建接线 | `foreach/foreach_sub_scalar/op_kernel/CMakeLists.txt` | `arch35/foreach_sub_scalar.cpp` | ascend950 Kernel source |
| 构建接线 | `foreach/foreach_sub_scalar/op_host/config/ascend950/foreach_sub_scalar_binary.json` | 四个 binary entries | 950 FP16/FP32/INT32/BF16实例 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 4个标量字段族+256元素数 | offsets 0/4/8/16、sizeof2064、tensorCount≤256 | `ForeachSubScalarTilingData` |
| 调用顺序 | Contiguous→scalar conversion→单SIMT Kernel | 失败阻断与stream顺序 | V2 API/L0 |
| workspace/内存 | 核心workspace0、直接GM访问 | TensorList指针表偏移解析、localMemory metadata | Tiling/`SimtGetTensorAddr` |
| 同步 | 每idx单thread，无barrier/atomic | grid-stride必须保持互斥 | `OpForeachSubScalarSimt` |
| 精度与尾块 | half/BF16在float表达式中计算，`idx<count` | S/C模板关系、不可臆测round mode | `ComputeType`/store expression |
| 失败边界 | 4 dtype封闭、256 Tensor、空域1 block | unsupported不得fallback | dtype tree/guards |
