---
type: Operator
title: "ForeachSinh 950 ACLNN 算子知识"
description: "说明 aclnnForeachSinh 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/docs/aclnnForeachSinh.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_host/foreach_sinh_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_host/arch35/foreach_sinh_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/arch35/foreach_sinh.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_host/config/ascend950/foreach_sinh_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:20:50Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSinh 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 ACLNN：`aclnnForeachSinhGetWorkspaceSize` → `aclnnForeachSinh`；固定仓未出现更高版本入口。 |
| 计算语义 | 对 TensorList 每个 Tensor 逐元素计算 `sinh`；SIMT body 统一以 FP32 `sinhf` 求值。 |
| 输入输出 | ND TensorList；FP16、FP32、BF16；每 Tensor 0–8 维；输出连续、dtype 与输入一致且容量不小于输入。 |
| 执行类型 | 单 Kernel、单个 `SIMT` 节点；无 Cube/AIC 与 `MIX_CV` 可达节点。 |
| Kernel 路线 | R-001 FP16 SIMT；R-002 FP32 SIMT；R-003 BF16 SIMT。 |
| 核心资源 | GM ListTensor 描述符；每 block 512 threads；grid-stride 直接 GM 读写；无显式 TQue/TBuf/shared UB；16 MiB AscendC tools workspace。 |
| 来源闭包 | `source_closed`：公开 ACLNN 文档 → 自动生成 ACLNN/OpDef 框架边界 → `ForeachSinhTilingFunc` → schMode 模板 entry → `NsForeachSinh::Process/OpForeachSinhSimt/ComputeSinh`。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`、`y=[y_0,...,y_{n-1}]`。对每个输入有效元素，`y_i[j]=sinh(x_i[j])=(exp(x_i[j])-exp(-x_i[j]))/2`。950 实体先把输入值 `static_cast<float>`，调用 `sinhf`，再转换成输出模板类型；FP32 路线的前后 cast 为恒等类型转换，FP16/BF16 路线在 FP32 中求值后降精度。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每 Tensor ND、0–8 维 | FLOAT16/FLOAT32/BFLOAT16 | 非空指针；同一列表 dtype 一致；文档允许空 Tensor；OpDef 启用 `AutoContiguous()`。 |
| `out` | TensorList；每 Tensor ND、0–8 维 | 与 `x` 相同 | 非空指针；输出不支持非连续；每项容量不少于对应输入。 |
| `workspaceSize` | 标量 | `uint64_t*` | 950 Tiling 固定返回 16 MiB AscendC tools workspace。 |
| `executor` | 框架对象 | `aclOpExecutor**` | 第一段接口产生；内部布局固定仓不可见。 |
| `workspace`/`stream` | Device 地址/执行流 | — | 第二段接口使用；Kernel 参数包含 workspace，但 body 不解引用它。 |

### 入口与版本

公开文档同时将 Ascend 950PR/950DT 标为支持。`foreach_sinh_def.cpp` 为 `ascend950` 配置动态 rank/shape、ND 与三种 dtype；`op_host/CMakeLists.txt` 选择 arch35 Tiling，`op_kernel/CMakeLists.txt` 为 `ascend950` 选择 `arch35/foreach_sinh.cpp`，binary config 也列出三种 dtype。自动生成 ACLNN inner bridge 不在固定仓中，其 Contiguous/executor 装箱是明确框架边界，不补造成手写入口实现。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 文档定义空指针、dtype、输入输出 dtype 一致、shape 容量、列表内 dtype 一致和维数上限错误；OpDef 将输入输出声明为动态 ND 并启用 `AutoContiguous()`。
2. arch35 `ForeachSinhTilingFunc` 读取平台 AIV core 数与 UB；core 数 `<=0` 或 UB 为 0 直接失败。
3. 它从动态输入 0 的 instance 数窄化成 `int32_t tensorNum`，随后检查 `[0,256]`；若非空，以第一个输入描述符 dtype 选择 schMode。
4. 两次遍历输入 shape：第一次取得最大 Tensor element count，第二次写 POD 每项 element count。固定 arch35 Tiling 本身不读取输出 desc/shape；这部分依赖生成 ACLNN/OpDef 校验边界。

### 参数、Workspace 与生命周期

设最大单 Tensor 元素数 `M=max_i n_i`、平台核数 `Cmax`。Host 计算：

```text
perCore = align_up(max(1024, ceil(M/Cmax)), 32) elements
needCoreNum = max(1, min(Cmax, ceil(M/perCore)))
blockDim = needCoreNum
threadsPerBlock = 512
localMemorySize = uint32(ubSize - 128 KiB)
workspaceSize = 16 MiB
```

所有 block 都遍历整个 TensorList；每个 Tensor 内由 `(blockIdx,threadIdx)` 的全 grid stride 分工，因此核心数按最大 Tensor 而非总元素数选择。workspace 由调用方持有并覆盖第二段执行；固定 body 不读 `workspace`，注释只表明其为 AscendC tools 预留。Host 写入 POD 的 `needCoreNum` 不被 Device 读取，真实 consumer 是 `SetBlockDim(needCoreNum)`。

### 启动图与失败传播

```text
aclnnForeachSinhGetWorkspaceSize
  -> [自动生成 ACLNN/OpDef 边界]
  -> ForeachSinhTilingFunc
     -> GetPlatformInfo
     -> tensorNum/maxTensorElements/perCore/needCoreNum
     -> dtype -> schMode 0/1/2
     -> POD + workspace/localMemorySize/blockDim
  -> workspaceSize=16 MiB + executor

aclnnForeachSinh(workspace, workspaceSize, executor, stream)
  -> K0 foreach_sinh<schMode><<<needCoreNum blocks, 512 SIMT threads/block>>>
```

仅有 K0，无 workspace 中间边。K0 在同一 stream 直接读取每个输入 Tensor GM 并写对应输出。任一 Host 空指针、平台或 dtype 检查失败均返回 `GRAPH_FAILED` 并阻断 launch；固定仓未保存生成 bridge 到 ACLNN status 的精确映射。模板 entry 只有 schMode 0/1/2 三个 `if constexpr` 分支，构建 selector 不会实例化其它值。

## Route 与参数设计

### 分核与资源计算

每个 Tensor 的逻辑一维域为 `[0,n_i)`。线程全局索引 `g=blockIdx*threadNum+threadIdx`，全 grid stride `S=threadNum*blockNum=512*needCoreNum`；线程处理 `g,g+S,...<n_i`。`needCoreNum` 至少为 1，故空 TensorList 仍发射一个 block；单个空 Tensor 在 `Process` 中因 `count==0` 跳过。

没有 `InitBuffer`、TQue、TBuf、shared/dynamic UB 显式分配。Host 将平台 UB 减去 128 KiB DCache 后设为 launch local memory size，但 body 没有可见的显式 consumer。workspace 固定 16 MiB，也没有算子 body slice。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 SIMT | schMode 0 | `xDtype==DT_FLOAT16` | 检查 FP32 |
| 2 | R-002 FP32 SIMT | schMode 1 | `xDtype==DT_FLOAT` | 检查 BF16 |
| 3 | R-003 BF16 SIMT | schMode 2 | `xDtype==DT_BF16` | `GRAPH_FAILED` |

```text
tensorNum==0 -> xDtype defaults DT_FLOAT -> schMode 1 -> R-002 (body loops zero tensors)
otherwise FLOAT16 -> 0 -> R-001
else FLOAT32 -> 1 -> R-002
else BF16 -> 2 -> R-003
else -> Host failure
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | schMode 0 | FP16、TensorNum `[0,256]`、950 | 单 K0 SIMT | `Process<half>` → `OpForeachSinhSimt<half>` → `ComputeSinh<half>` | 直接 GM load/store，无显式共享内存/barrier/atomic | `idx<count` grid-stride guard；空 Tensor skip |
| R-002 | schMode 1 | FP32，其余同上；空列表默认此 key | 单 K0 SIMT | `Process<float>` → `OpForeachSinhSimt<float>` | 同上 | 同上 |
| R-003 | schMode 2 | BF16，其余同上 | 单 K0 SIMT | `Process<bfloat16_t>` → `OpForeachSinhSimt<bfloat16_t>` | 同上；FP32 计算后 cast 回 BF16 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `0<=tensorNum<=256` | POD `tensorElements[256]` 固定容量 | 全部 | Host `GRAPH_FAILED` | `ForeachSinhTilingFunc` |
| core/UB 可用 | `coreNum>0 && ubSize!=0` | blockDim 和 local memory producer 必需 | 全部 | Host failure | `GetPlatformInfo` |
| 每核至少 1024 元素目标 | `perCore=max(1024,ceil(M/Cmax))` 后 32 元素对齐 | 避免为最大 Tensor 的很小分片发射过多 block | 全部 | 非 selector fallback；直接改变 blockDim | `ForeachSinhTilingFunc` |
| local memory 扣减 | `SetLocalMemorySize(uint32(ubSize-128KiB))` | 给 DCache 预留 128KiB | 全部 | 仅检查 ubSize 非 0，未检查 `>=128KiB`，存在下溢边界 | 同上 |
| 空 Tensor | `count==0` | 防止空域 GM 访问/VF launch | 全部 | continue 下一 Tensor | `NsForeachSinh::Process` |
| 线程 tail | `idx<count` | grid 大小一般不整除 element count | 全部 | 超界线程不访问 GM | `OpForeachSinhSimt` |

### POD/ABI

`ForeachSinhTilingData` 是普通结构体：自然布局下固定字段声明可证明 offsets 为 0、4、8；最大对齐 8，`sizeof=2056`、`alignof=8`。Host 与 Device include 同一 header。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSinhTilingData` | `tensorNum` | `int32_t`/32 | offset 0；struct sizeof 2056/align 8 | tensors | `tiling->tensorNum=tensorNum` | `Process` 循环上界 | `[0,256]` |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | blocks | `tiling->needCoreNum=needCoreNum` | Device 不读取；Host 同值传 `SetBlockDim` | 至少 1；POD 字段 inactive |
| 同上 | `tensorElements` | `int64_t[256]` | offset 8 | elements/Tensor | 逐项 `shape.GetShapeSize()` | `Process` 的 `count` 与 VF 参数 | 未使用项由 `memset_s` 清零 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | schMode 0 | `foreach_sinh_tiling_arch35.cpp::ForeachSinhTilingFunc` | `ForeachSinhTilingData` | `foreach_sinh<0>` | `Process<half>/OpForeachSinhSimt<half>/ComputeSinh<half>` | arch35 kernel CMake + ascend950 binary config |
| R-002 | schMode 1 | 同上 | 同上 | `foreach_sinh<1>` | float 实例 | 同上 |
| R-003 | schMode 2 | 同上 | 同上 | `foreach_sinh<2>` | bfloat16 实例 | 同上 |

## Kernel 实现

### 公共执行流程

模板 entry 使用 `REGISTER_TILING_DEFAULT` 与 `GET_TILING_DATA_WITH_STRUCT` 解析 POD，再以编译期 schMode 选择 dtype。每个 block 构造输入/输出 `ListTensorDesc`，顺序遍历全部 Tensor；空 Tensor 跳过，非空项取得类型化 GM 指针并以 `VF_CALL(...Dim3(512),xPtr,yPtr,count)` 进入 SIMT body。每个线程对唯一的 grid-stride 索引序列执行 load→FP32 `sinhf`→cast→store；索引序列在同一次 launch 中不重叠，因此不需要 atomic 或 block barrier。

### Route R-001：FP16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_sinh_tiling_arch35.cpp` | `ForeachSinhTilingFunc` | `xDtype==DT_FLOAT16` | schMode 0 |
| Router | `op_kernel/arch35/foreach_sinh.cpp` | `foreach_sinh<schMode>` | `schMode==TILING_KEY_FLOAT16` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sinh<0>(x,y,workspace,tiling)
  -> NsForeachSinh::Process<half>
     -> ListTensorDesc::GetDataPtr<half>(i)
     -> VF_CALL<OpForeachSinhSimt<half>>(Dim3(512),xPtr,yPtr,count)
        -> ComputeSinh<half> -> sinhf(float(val)) -> half(result)
```

##### 源码等价伪码

```text
for tensor i in [0,tensorNum):
  count=tensorElements[i]; if count==0: continue
  xPtr=xList.GetDataPtr<half>(i); yPtr=yList.GetDataPtr<half>(i)
  parallel for tid=blockIdx*512+threadIdx; tid<count; tid+=512*blockNum:
    fval=float(xPtr[tid]); result=sinhf(fval); yPtr[tid]=half(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| ListTensor descriptor | `x/y` entry 参数 | `tensorNum` pointers | descriptor | ACLNN | `ListTensorDesc` | caller | 框架格式 |
| GM input | `xPtr[idx]` | 1 element/iteration | FP16 flat | caller | thread | caller | 无向量搬运要求 |
| thread scalar | register/local | one FP16 + FP32 intermediates | scalar | load/cast/sinhf | store | thread iteration | 固定仓不可见寄存器字节数 |
| GM output | `yPtr[idx]` | 1 element/iteration | FP16 flat | thread | caller | caller | `idx<count` |
| workspace | 未寻址 | 16 MiB | AscendC tools opaque | caller | body 无 consumer | caller | 单一预留，无 slice |

##### 分层内存布局

N/A：该 SIMT body 未调用 `InitBuffer`，未声明 shared/dynamic UB、TQue 或 TBuf；实际替代路径是 GM 标量 load/store 与线程私有标量。Host 仍设置 `localMemorySize=uint32(ubSize-128KiB)`，固定 body 无显式消费点。

##### 字节账与复用顺序

```text
UB_explicit = 0 B
dynamic/shared_ub_explicit = 0 B
launch_local_memory = uint32(ubSize - 128 KiB)  # Host 设置，body 无显式分配
workspace_total = 16 MiB AscendC tools reserve; operator body slice = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程私有 `val/fval/result` | 当前 iteration | `sinhf`/store | 同线程下一 idx | 程序顺序 | store 完成后覆盖；线程间不共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 并行分工 | grid-stride 索引 | Host grid + thread ids | 当前 thread | GM element | `idx<count` 消除尾部越界 |
| 数值 | `static_cast<float>`→`sinhf`→`static_cast<half>` | GM load | GM store | thread scalar | FP32 求值，降精度 cast 具体舍入细节固定仓未展开 |
| 同步 | N/A | 每线程 | 同线程 store | 唯一 GM index | 无 atomic/barrier；索引域不重叠 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sinh_tiling_arch35.cpp` | `ForeachSinhTilingFunc` | schMode、blockDim、workspace/local memory |
| Kernel entry/router | `op_kernel/arch35/foreach_sinh.cpp` | `foreach_sinh<0>` | FP16 模板映射 |
| Device body/helper | `op_kernel/arch35/foreach_sinh_simt.h` | `Process<half>`、`OpForeachSinhSimt<half>`、`ComputeSinh<half>` | Tensor loop、512 threads、grid stride、FP32 math |

### Route R-002：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_sinh_tiling_arch35.cpp` | `ForeachSinhTilingFunc` | `xDtype==DT_FLOAT`；空列表也默认此 dtype | schMode 1 |
| Router | `op_kernel/arch35/foreach_sinh.cpp` | `foreach_sinh<schMode>` | schMode 1 | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sinh<1> -> Process<float>
  -> for each nonempty Tensor: VF_CALL<OpForeachSinhSimt<float>>(512,...)
     -> ComputeSinh<float>: float -> sinhf -> float
```

##### 源码等价伪码

```text
对每个非空 Tensor，线程 idx=blockIdx*512+threadIdx；
while idx<count: y[idx]=sinhf(x[idx]); idx+=512*blockNum。
空 TensorList 时 tensorNum=0，body 不进入循环。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `[idx]` | 1 FP32/iteration | flat | caller/thread | thread/caller | caller | tail guard |
| thread scalar | 私有 | FP32 val/result | scalar | load/sinhf | store | iteration | 无共享 |
| workspace | 未寻址 | 16 MiB | opaque | caller | 无 | caller | body 0 B |

##### 分层内存布局

N/A：无 AIV Queue/TBuf、无显式 SIMT shared/dynamic UB；GM→线程标量→GM。

##### 字节账与复用顺序

```text
UB_explicit = 0 B; shared_ub_explicit = 0 B
launch_local_memory = uint32(ubSize - 128 KiB)
workspace_total = 16 MiB reserve / 0 B body slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | load/sinhf | store | next idx | thread program order | 当前 store 完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Compute | `sinhf` | FP32 load | FP32 store | thread scalar | 全程 FP32 |
| Tail/sync | grid-stride + `idx<count` | grid | thread | GM range | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_sinh_tiling_arch35.cpp` | dtype branch | schMode 1 |
| Kernel entry/router | `op_kernel/arch35/foreach_sinh.cpp` | `foreach_sinh<1>` | float body |
| Device body/helper | `foreach_sinh_simt.h` | float instances | 直接 FP32 `sinhf`、grid stride |

### Route R-003：BF16 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Tiling | `foreach_sinh_tiling_arch35.cpp` | dtype branch | `xDtype==DT_BF16` | schMode 2 |
| Router | `op_kernel/arch35/foreach_sinh.cpp` | `foreach_sinh<schMode>` | schMode 2 | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_sinh<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachSinhSimt<bfloat16_t>>(512,...)
     -> ComputeSinh<bfloat16_t>
        -> static_cast<float> -> sinhf -> static_cast<bfloat16_t>
```

##### 源码等价伪码

```text
for every nonempty Tensor and each thread-owned idx:
  bf16 val=x[idx]
  float fval=float(val)
  float result=sinhf(fval)
  y[idx]=bfloat16_t(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM input/output | `[idx]` | 1 BF16/iteration | flat | caller/thread | thread/caller | caller | idx guard |
| thread scalar | private | BF16 val + FP32 fval/result | scalar | load/cast/sinhf | cast/store | iteration | no shared alias |
| workspace | not addressed | 16 MiB | opaque | caller | none | caller | body 0 B |

##### 分层内存布局

N/A：无显式 UB/shared buffer；BF16 与 FP32 中间量在线程私有标量中生存。

##### 字节账与复用顺序

```text
UB_explicit = 0 B; shared_ub_explicit = 0 B
launch_local_memory = uint32(ubSize - 128 KiB)
workspace_total = 16 MiB reserve / 0 B body slice
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread private fval/result | cast/sinhf | cast/store | next grid-stride idx | thread program order | store 后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度/计算 | cast→`sinhf` | BF16 load | FP32 result | private scalar | FP32 求值 |
| 降精度 | `static_cast<bfloat16_t>` | FP32 result | GM store | private scalar | 具体 rounding 语义由 SIMT cast 实现，固定仓未展开 |
| Tail/sync | `idx<count` | launch grid | thread | GM range | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_sinh_tiling_arch35.cpp` | BF16 dtype branch | schMode 2 |
| Kernel entry/router | `op_kernel/arch35/foreach_sinh.cpp` | `foreach_sinh<2>` | BF16 body |
| Device body/helper | `foreach_sinh_simt.h` | BF16 instances | FP32 sinhf 中间精度与 tail |

## 算子特定优化

### Trick T-001：以最大 Tensor 定 grid、全部 Tensor 复用同一 grid-stride 分工

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003。
- 触发条件：950 arch35 路线；每个非空 Tensor 调用同一 512-thread VF。
- Golden 来源：[`foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_simt.h:49-58`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_simt.h#L49-L58)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
inline void OpForeachSinhSimt(
    __gm__ T* xPtr, __gm__ T* yPtr, int64_t elemCount)
{
    for (uint64_t idx = static_cast<uint64_t>(
             Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
         idx < static_cast<uint64_t>(elemCount);
         idx += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
        T val = xPtr[idx];
        yPtr[idx] = ComputeSinh(val);
    }
}
```

#### 机制与设计原因

Host 只按最大 Tensor 计算 block 数；对每个 Tensor，所有 block/thread 复用同一个 grid-stride 公式。这样单个 Tensor 大于 grid 时每线程继续跨 stride 工作，小 Tensor 则由 `idx<elemCount` 自动裁掉多余线程。每个索引只属于一个线程，不需要 atomic、barrier 或中间 UB；源码可证明它避免中间物化，但没有真机性能数据。

#### 迁移边界

- 适用：TensorList 中 Tensor 相互独立、逐元素、每个输出只写一次的算子。
- 前提：所有 Tensor dtype 相同；最大 Tensor 能代表所需 grid 上界；每次 VF 调用共享 blockDim。
- 失效：跨元素归约、邻域依赖、原地读写冲突或需要 Tensor 间同步的算法。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若把 stride 错写为仅 `threadNum`，不同 block 将覆盖相同输出；固定仓未提供其它可验证反例。

### Trick T-002：三种 dtype 统一 FP32 `sinhf` 中间计算

#### 触发与源码依据

- 适用 Route：R-001/R-002/R-003。
- 触发条件：`ComputeSinh<T>` 被 SIMT body 调用。
- Golden 来源：[`foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_simt.h:36-42`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_simt.h#L36-L42)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T>
__simt_callee__ inline T ComputeSinh(T val)
{
    float fval = static_cast<float>(val);
    float result = sinhf(fval);
    return static_cast<T>(result);
}
```

#### 机制与设计原因

单一 helper 将 FP16/BF16 提升到 FP32 后调用 `sinhf`，避免各低精度 Route 维护不同近似流程；FP32 Route 也复用同一 body。低精度仅在最终 store 前转换，计算阶段保持 FP32。具体降精度 rounding 由固定仓外的 SIMT 类型转换语义决定。

#### 迁移边界

- 适用：低精度输入、允许 FP32 opmath 中间语义的逐元素超越函数。
- 前提：API 数值契约允许该升精度顺序，SIMT 提供 `sinhf` 与目标类型转换。
- 失效：要求严格低精度逐指令复现或不同 rounding mode 的接口。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 2 | Host dtype 顺序分支 |
| 模板与 dtype | `T=half` | `T=float` | `T=bfloat16_t` | binary config 三项 |
| 循环与分核 | 512 threads、grid stride | 同 | 同 | 最大 Tensor 统一定 grid |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程标量；无显式 UB | 同 | BF16+FP32 线程标量 | 无 Queue/通信 |
| 内存总占用与复用 | body UB 0 | body UB 0 | body UB 0 | launch local memory 无显式 consumer |
| 搬运/Vector/SIMT/NDDMA/通信 API | GM scalar+SIMT VF | 同 | 同 | 单 SIMT 节点 |
| 同步、精度与 tail | FP32 sinhf→FP16 | FP32 sinhf | FP32 sinhf→BF16 | idx guard；无 barrier/atomic |
| fallback/失败边界 | 无 Device fallback | 空列表默认 key | 无 | unsupported dtype Host failure |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 Foreach Tiling | 同 A2 | arch35 独立 Tiling，最多 256 Tensor |
| Kernel/资源 | AIV Vector、双 Queue | 同 A2 | SIMT 512 threads、直接 GM、16 MiB tools workspace |

## 限制与失败边界

- `tensorNum` 在检查 256 上限前由 instance 数 `static_cast<int32_t>`；固定源码没有窄化前范围检查。正常框架输入仍必须满足 `[0,256]`。
- `ubSize` 只检查非 0，却直接计算无符号 `ubSize-128KiB` 再窄化到 `uint32_t`；复用实现必须补足 `ubSize>=128KiB` 的平台前提。
- arch35 Tiling 不读取输出描述或输出 shape，依赖生成 ACLNN/OpDef 边界落实文档约束；该生成 bridge 在固定仓不可见。
- workspace 固定 16 MiB，但算子 body 不读 `workspace`；其 AscendC tools 内部用途与切片在固定仓不可见，不能改写成已证明的 0 workspace API。
- FP16/BF16 的最终 `static_cast<T>` rounding 细节不在固定仓展开；只可确认 FP32 `sinhf` 发生在降精度之前。
- 输出容量大于输入时只写前 `tensorElements[i]` 个元素；多出区域未定义。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sinh/docs/aclnnForeachSinh.md` | ACLNN 原型/产品表 | 公开性、950 支持与接口约束 |
| 入口实现 | `foreach/foreach_sinh/op_host/foreach_sinh_def.cpp` | `ops::ForeachSinh` | ascend950 OpDef、dtype/ND/AutoContiguous |
| Host/Tiling/Launcher | `foreach/foreach_sinh/op_host/arch35/foreach_sinh_tiling_arch35.cpp` | `ForeachSinhTilingFunc` | 校验、grid、POD、workspace/local memory |
| Host/Tiling/Launcher | `foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_tiling_data.h` | `ForeachSinhTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | schMode 实例集合 |
| Kernel entry/body | `foreach/foreach_sinh/op_kernel/arch35/foreach_sinh.cpp` | `foreach_sinh<schMode>` | 编译期 router |
| Kernel entry/body | `foreach/foreach_sinh/op_kernel/arch35/foreach_sinh_simt.h` | `Process/OpForeachSinhSimt/ComputeSinh` | 全部 SIMT body |
| 构建接线 | `foreach/foreach_sinh/op_host/config/ascend950/foreach_sinh_binary.json` | 三项 `op_list` | 950 三 dtype binary |
| 构建接线 | `foreach/foreach_sinh/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 kernel source 选择 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2056B 三字段结构 | offset 0/4/8、256 项、Host/Device 同 header | `ForeachSinhTilingData` |
| 调用顺序 | Tensor loop→VF_CALL→grid-stride | 空 Tensor 必须 continue | `NsForeachSinh::Process` |
| workspace/内存 | 16MiB tools reserve；body 直接 GM | local memory 先扣 128KiB；不得宣称 workspace 可删 | `ForeachSinhTilingFunc` |
| 同步 | 唯一索引免 atomic/barrier | stride 必须含 blockNum | `OpForeachSinhSimt` |
| 精度与尾块 | 统一 FP32 `sinhf`；idx guard | cast 顺序与有效域不可变 | `ComputeSinh` |
| 失败边界 | Host 拒绝平台/dtype/tensorNum 错误 | 不存在 Device fallback | `GetPlatformInfo/ForeachSinhTilingFunc` |

- [Sinhf](../../../../apis/simt_api/math_functions/float_math_functions/sinhf.md)
