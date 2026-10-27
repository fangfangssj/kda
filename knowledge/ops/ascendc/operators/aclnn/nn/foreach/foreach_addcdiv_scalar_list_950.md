---
type: Operator
title: "ForeachAddcdivScalarList 950 ACLNN 算子知识"
description: "说明 aclnnForeachAddcdivScalarList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/docs/aclnnForeachAddcdivScalarList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_host/foreach_addcdiv_scalar_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_host/arch35/foreach_addcdiv_scalar_list_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_host/arch35/foreach_addcdiv_scalar_list_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_host/config/ascend950/foreach_addcdiv_scalar_list_binary.json
    role: implementation
created_at: "2026-08-25T15:30:32Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAddcdivScalarList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAddcdivScalarListGetWorkspaceSize` + `aclnnForeachAddcdivScalarList`，无版本后缀的公开 ACLNN 两段式接口 |
| 计算语义 | `y_i[j] = x1_i[j] + scalars[i] * (x2_i[j] / x3_i[j])` |
| 输入输出 | 三个等长动态 TensorList + 标量 Tensor → 动态 TensorList；ND；FP16/FP32/BF16；逐位置 shape/dtype 一致 |
| 执行类型 | 单 Kernel：`SIMT` 的 `__simt_vf__` 线程 body；无 Cube/AIC、MIX_CV、AIV Queue、NDDMA 或通信节点 |
| Kernel 路线 | R-001 FP16→FP32 SIMT；R-002 FP32 直算 SIMT；R-003 BF16→FP32 SIMT |
| 核心资源 | ListTensor GM 指针表；每 block 512 threads；直接 GM load/compute/store；无显式 shared/dynamic UB、Queue、atomic、barrier 或 Device workspace |
| 来源闭包 | source_closed：公开接口契约、生成式 ACLNN→OpDef 框架边界、公共 InferShape、arch35 Tiling/POD、三种 dtype 的 SIMT entry/body 与 950 构建接线已闭合。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

列表长度为 `n`。每个列表位置共享一个 `scalars[i]`，对 `x1_i.numel` 个元素先求 `x2_i/x3_i`，再乘标量并加 `x1_i`。Device 没有广播、除零、NaN/Inf 或饱和特判；输出有效域由 `perTensorElementNum[i]` 限定。该卡不涵盖原地、Grad 或量化变体。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | 动态 TensorList；每项 0–8 维，ND | FP16/FP32/BF16，列表内一致 | 可含空 Tensor；非连续输入通过 OpDef `AutoContiguous` 框架边界。 |
| `x2` | 与 `x1` 等长且逐项同 shape | 与 `x1` 一致 | 除法分子。 |
| `x3` | 与 `x1` 等长且逐项同 shape | 与 `x1` 一致 | 除法分母。 |
| `scalars` | `numel == tensorNum`，0–8 维 ND | 与 `x1` 一致 | 每 tensor 一个标量；公开文档要求非空。 |
| `out` | 动态 TensorList；InferShape 逐项复制 `x1` | 与 `x1` 一致 | 公开接口不支持非连续输出。 |

### 入口与版本

固定仓公开的最新且唯一版本是无后缀 `aclnnForeachAddcdivScalarList`。第一段返回总 workspace 与 executor，第二段接收 workspace、executor、stream 执行。仓内没有生成后的 `aclnnop` 头/实现，故 inner bridge 的逐语句参数检查、连续化节点和 executor 总 workspace 公式固定仓不可见；Skill 允许将其作为框架边界。OpDef 明确注册 `ascend950`，公开产品表明确支持 Ascend 950PR/950DT，Host CMake 选择 arch35、Kernel CMake 选择 arch35 SIMT entry、binary JSON 覆盖三种 dtype，平台证据无冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

1. 公开 ACLNN 第一段要求五个主参数非空，dtype 为 FP16/FP32/BF16 且一致，三个列表等长/逐位置 shape 相同，标量数等于列表长度，维数不超过 8；错误经 `161001/161002/561002` 返回。
2. OpDef 对所有输入执行 `AutoContiguous`；公共 InferShape 要求输出实例数等于 `x1`，逐项复制 shape/dtype。
3. arch35 Tiling 自身读取平台 core/UB、`x1` 实例数、逐项 storage numel 和首项 dtype；它显式拒绝 `tensorNum>256`，但不重复检查 `x2/x3/scalars/out`，因此这些约束依赖 ACLNN/图框架校验边界。
4. Tiling 选择 dtype key、计算 blockDim、设置 LocalMemorySize、序列化 POD 并设置 kernel workspace 为 0；任一步失败则不 launch。

### 参数、Workspace 与生命周期

令 `M=max_i numel(x1_i)`、`C=GetCoreNumAiv()`、`UB=GetCoreMemSize(UB)`。Tiling 首先要求 `UB>128 KiB`，令 `L=UB-128 KiB`；`needCoreNum=min(max(1,ceil(M/1024)),C)`，既写 POD 又作为 `blockDim`。key 是 FP16→0、FP32→1、BF16→2。`SetLocalMemorySize(L)` 把 `L` 交给 launch/runtime，但 SIMT body 不声明或寻址 dynamic/shared UB；因此它是可见的 launch local-memory 参数，不是可证明已消费的用户缓冲。

算子 kernel workspace 明确为 0，entry 和所有 Process 都忽略 `workspace`。公开 ACLNN workspace 仍可能因非连续输入转换而非零，精确公式固定仓不可见。executor、调用方 workspace 和 stream 的 owner/lifetime 与 A2/A3 相同：必须保持到已入队工作不再依赖，精确 runtime 释放点仓内不可见。

### 启动图与失败传播

```text
aclnn...GetWorkspaceSize
  -> [生成式 bridge / AutoContiguous，固定仓不可见]
  -> ForeachAddcdivScalarListTilingFunc
     -> key, blockDim=needCoreNum, localMemory=L, workspace[0]=0, POD
  -> executor 记录单节点 N1
aclnnForeachAddcdivScalarList(..., stream)
  -> N1: foreach_addcdiv_scalar_list<schMode><<<needCoreNum, stream>>>
     -> 每个外层 block 对每个非空 tensor 发起 512-thread VF_CALL
```

N1 直接从 `x1/x2/x3/scalars` GM 读并写 `y` GM，没有 Kernel 间 workspace edge。Tiling 获取平台信息、容量检查、`SetLocalMemorySize` 任一失败均返回 `GRAPH_FAILED`；entry 的 `if constexpr` 仅包含 key 0/1/2，未知模板值没有 body。stream ordering 与 launch 错误传播由生成式 ACLNN runtime 保证，固定仓不可见。

## Route 与参数设计

### 分核与资源计算

分核只取最大单 tensor 元素数 `M`，而非列表总和，因为所有 block 会逐个遍历整张列表，并对每个 tensor 用同一全局线程网格分摊元素。对 tensor `i`，线程线性下标：

```text
index0 = blockIdx * 512 + threadIdx
stride = blockNum * 512
for index=index0; index<perTensorElementNum[i]; index+=stride
```

空 tensor 在 Process 层 `continue`。因此没有 former/tail core POD；尾部由线程循环的 `index<numel` 判断保护。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16→FP32 SIMT | key/schMode 0 | `GetTilingKeyByDtype(DT_FLOAT16)==0` | dtype 非 FP16 时选其他编译实例 |
| 2 | R-002 FP32 直算 SIMT | key/schMode 1 | `DT_FLOAT` | dtype 非 FP32 时选其他实例 |
| 3 | R-003 BF16→FP32 SIMT | key/schMode 2 | `DT_BF16` | dtype 非 BF16 时不选；非法 dtype 的 Host default 为 0，须由上游 dtype guard 阻断 |

```text
dtype FP16 -> schMode 0 -> ProcessFp16
dtype FP32 -> schMode 1 -> Process<float,float>
dtype BF16 -> schMode 2 -> ProcessBf16
other dtype -> Host helper default 0（正常公开入口不可达）
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16→FP32 | 0 | FP16；列表约束成立；950 | 单 SIMT N1 | `ProcessFp16`→`OpForeachAddcdivScalarListFp16SimtKernel` | 每线程 direct GM load；标量与三输入转 FP32；无 barrier/atomic | `index<numel`；空 tensor skip |
| R-002 FP32 直算 | 1 | 同上，FP32 | 单 SIMT N1 | `Process<float,float>`→`OpForeachAddcdivScalarListSimtKernel<float,float>` | direct GM load/FMA-friendly expression/store | 同上 |
| R-003 BF16→FP32 | 2 | 同上，BF16 | 单 SIMT N1 | `ProcessBf16`→`OpForeachAddcdivScalarListBf16SimtKernel` | BF16 inputs/scalar 转 FP32，三步 math，结果降 BF16 | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| tensor 数上限 | `static_cast<int32_t>(tensorNum)>256` | POD `perTensorElementNum[256]` | 全部 | `GRAPH_FAILED` | `ForeachAddcdivScalarListTilingFunc` |
| UB reserve | `UB>128 KiB`; `L=UB-128 KiB` | Host 为 local memory 扣除 DCache | 全部 | `GRAPH_FAILED` | `DCACHE_SIZE`; `SetLocalMemorySize` |
| 核数 | `ceil(M/1024)` 截到平台 core，至少 1 | 保证每最大 tensor 约每核至少 1024 元素目标 | 全部 | 无替代 Route | `SINGLE_CORE_MIN_ELEMENTS`; Tiling Func |
| threads | `THREAD_NUM=512`; `LAUNCH_BOUND(512)` | SIMT 线程网格固定 | 全部 | 无动态 thread fallback | SIMT header |
| dtype | OpDef/ACLNN 限 FP16/FP32/BF16 | Host helper default 对非法 dtype 返回 0 | 全部 | 必须上游失败；不能让非法 dtype 落入 FP16 | `GetTilingKeyByDtype`; OpDef |
| 空域/tail | `elementNum==0` skip；线程 `index<elementNum` | 防止 GM 越界 | 全部 | 自然无输出或线程退出 | `Process*`; SIMT kernels |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachAddcdivScalarListTilingData` | `tensorNum` | `int32_t`/32 | offset 0；结构 align 8，sizeof 2056 B | tensor 个数 | `static_cast<int32_t>(tensorNum)` | 三个 `Process*` 的 tensor 循环上界 | 0..256 |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | block/core 数 | `static_cast<int32_t>(needCoreNum)` | Device body未读取；launch 使用独立 `SetBlockDim(needCoreNum)` | inactive POD 字段仍须保持 ABI |
| 同上 | `perTensorElementNum[256]` | `int64_t[256]` | offset 8；2048 B | 元素 | 逐项 `shape.GetShapeSize()` | `Process*` 取 `elementNum` | `[tensorNum,256)` 未初始化且不读取 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | schMode 0 | `arch35/...tiling.cpp::ForeachAddcdivScalarListTilingFunc` | arch35 Host/Device 同布局 POD；blockDim/L | arch35 entry `if constexpr schMode==0` | `ProcessFp16`; FP16 SIMT kernel; `SimtGetTensorAddr` | Host/Kernel CMake + ascend950 binary JSON |
| R-002 FP32 | schMode 1 | 同上 | 同上 | entry schMode 1 | `Process<float,float>`; generic SIMT kernel | 同上 |
| R-003 BF16 | schMode 2 | 同上 | 同上 | entry schMode 2 | `ProcessBf16`; BF16 SIMT kernel | 同上 |

## Kernel 实现

### 公共执行流程

entry 使用编译期 `schMode` 选择一个 dtype body。每个外层 AIC block 的 Process 都遍历 `tensorNum`；空 tensor 跳过，非空 tensor 通过 ListTensor 描述取四个 GM 指针，再调用 `VF_CALL`，线程维度固定 `Dim3(512)`。VF 内每个线程按全 grid stride 处理互不重叠元素，所以无共享内存、atomic 或 barrier。标量从 `scalars_gm[tensorIdx]` 读取；每个线程独立加载，生命周期只在寄存器表达式内。`workspace` 和 entry 局部解析出的 `tilingData` 对象均未参与数学计算；Process 直接从 `tiling` GM 读取 POD。

### Route R-001：FP16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/arch35/...tiling.cpp` | `GetTilingKeyByDtype` | `DT_FLOAT16` | key 0 |
| Template | `op_kernel/arch35/...tiling_key.h` | `schMode` list | value 0 | 编译 FP16 实例 |
| Router | `op_kernel/arch35/foreach_addcdiv_scalar_list.cpp` | entry | `if constexpr(schMode==0)` | `ProcessFp16` |

#### Device 数据流

##### 函数级调用链

```text
foreach_addcdiv_scalar_list<0>
  -> ProcessFp16
     -> SimtGetTensorAddr<half> ×4
     -> VF_CALL<OpForeachAddcdivScalarListFp16SimtKernel>(Dim3(512), ...)
        -> direct GM load -> FP32 cast/math -> half cast -> direct GM store
```

##### 源码等价伪码

```text
for tensor i in [0,tensorNum):
  if numel[i]==0: continue
  resolve x1_i,x2_i,x3_i,y_i from ListTensor pointer tables
  parallel 512-thread VF across blockDim blocks:
    for j=blockIdx*512+threadIdx; j<numel[i]; j+=blockDim*512:
      s=float(scalars[i])
      a=float(x1_i[j]); b=float(x2_i[j]); c=float(x3_i[j])
      y_i[j]=half(a + s*(b/c))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| ListTensor descriptor GM | base + first-u64 offset→pointer table `[i]` | tensorNum pointers | uint64 table | framework | `SimtGetTensorAddr` | caller/executor | pointer offset按 8 B (`>>3`) |
| x1/x2/x3 GM | pointer[i] + `j` | one half/thread/iteration | FP16 flat ND | prior op | thread registers | caller | 无合并参数源码；直接标量 load |
| scalar GM | `scalars+i` | one half/thread | FP16 | caller | thread | caller | 每线程读取并转 float |
| y GM | pointer[i] + `j` | one half/thread | FP16 | thread | next op | caller | `j<numel` 保证有效 |
| launch local memory | N/A（body无地址） | Host `L=UB-128KiB` | opaque launch resource | Tiling/runtime | 固定仓无 consumer | runtime | 不得虚构 dynamic UB 子区 |
| workspace | N/A | 0 B | N/A | Host | 无 | caller | 未使用 |

##### 分层内存布局

N/A：该 SIMT body 直接 GM→线程标量→GM，不分配 AIV UB Queue、shared/dynamic UB 或中间 workspace。Host 虽设置 local-memory size `L`，固定 body 没有可映射的地址/子区。

##### 字节账与复用顺序

```text
UB_total = N/A（无 InitBuffer/TQue/TBuf）
dynamic_ub = 0 B 可证明被 body 寻址；launch local-memory 参数 L=UB_platform-128KiB
workspace_total = 0 B（kernel tiling）
per-thread live values = scalar, x1, x2, x3, result（寄存器字节数固定仓不可见）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程局部 FP32 标量 | casts/loads | 单表达式 | 下一循环迭代 | 线程顺序语义 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 线程分工 | grid-stride loop | block/thread id | 本线程 | y[j] 独占 | `j<numel` 保护 tail |
| 计算 | N/A（线程内顺序） | FP32 casts | expression/store | thread locals | FP16→FP32；单表达式可供编译器融合；结果 cast half |
| 跨线程 | 无 barrier/atomic | 各线程 | N/A | 不相交 y[j] | 无冲突域 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_addcdiv_scalar_list_tiling.cpp` | Tiling Func | key0、blockDim、L、POD |
| Kernel entry/router | `op_kernel/arch35/foreach_addcdiv_scalar_list.cpp` | `schMode==0` | FP16 Process 路由 |
| Device body/helper | `op_kernel/arch35/foreach_addcdiv_scalar_list_simt.h` | `ProcessFp16`; FP16 SIMT kernel | pointer resolve、512 threads、FP32 math、tail |

### Route R-002：FP32 直算 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | arch35 Tiling | `GetTilingKeyByDtype` | `DT_FLOAT` | key 1 |
| Template | tiling key header | schMode list | value 1 | FP32 instance |
| Router | arch35 entry | `schMode==1` | compile-time true | `Process<float,float>` |

#### Device 数据流

##### 函数级调用链

```text
entry<1> -> Process<float,float>
  -> SimtGetTensorAddr<float> ×4
  -> VF_CALL<OpForeachAddcdivScalarListSimtKernel<float,float>>(512)
  -> direct FP32 GM loads -> expression -> FP32 GM store
```

##### 源码等价伪码

```text
沿用 R-001 的 tensor 循环和 grid-stride 地址。
每线程: scalar=scalars[i]; y[j]=x1[j] + scalar*(x2[j]/x3[j]);
所有量保持 FP32，无显式中间 buffer或同步。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/scalar GM | pointer[i]+j / scalars+i | 4 B per load | FP32 | caller | thread | caller | direct load |
| y GM | pointer[i]+j | 4 B per store | FP32 | thread | next op | caller | tail predicate |
| local/workspace | N/A | L launch parameter / 0 B workspace | opaque/N/A | Host | body无显式 consumer | runtime/caller | 无 UB 布局 |

##### 分层内存布局

N/A：与 R-001 相同，只有 GM 和线程局部标量。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = 0 B body-addressable；launch L=UB_platform-128KiB
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread locals | GM loads/expression | GM store | 下一 j | 线程内程序序 | 无跨线程 alias |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| grid-stride | `GetBlockIdx/GetThreadIdx` | index formula | load/store | 独占元素 | `j<numel` |
| math | 单 C++ 表达式 | loads | store | thread locals | FP32；源码注释说明有利于 FMA fusion，但未承诺生成指令 |
| synchronization | N/A | N/A | N/A | 不共享 | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | key1 path | FP32 selector |
| Kernel entry/router | arch35 entry cpp | schMode1 | generic Process |
| Device body/helper | SIMT header | `Process<float,float>`; generic SIMT kernel | direct FP32 expression |

### Route R-003：BF16→FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | arch35 Tiling | `GetTilingKeyByDtype` | `DT_BF16` | key 2 |
| Template | tiling key header | schMode list | value 2 | BF16 instance |
| Router | arch35 entry | `schMode==2` | compile-time true | `ProcessBf16` |

#### Device 数据流

##### 函数级调用链

```text
entry<2> -> ProcessBf16
  -> SimtGetTensorAddr<bfloat16_t> ×4
  -> VF_CALL<OpForeachAddcdivScalarListBf16SimtKernel>(512)
  -> BF16 GM loads -> FP32 div/mul/add -> BF16 GM store
```

##### 源码等价伪码

```text
for each assigned j:
  s=float(bf16 scalars[i])
  a=float(x1[j]); b=float(x2[j]); c=float(x3[j])
  divResult=b/c
  scaledResult=s*divResult
  result=a+scaledResult
  y[j]=bfloat16_t(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| BF16 input/scalar GM | pointer[i]+j / scalars+i | 2 B each | BF16 | caller | thread cast | caller | direct load |
| thread FP32 locals | registers（字节数不可见） | scalar,a,b,c,div,scaled,result | FP32 | casts/math | next expression/store | one iteration | compiler可复用，物理寄存器不可见 |
| BF16 y GM | pointer[i]+j | 2 B | BF16 | cast/store | next op | caller | tail predicate |
| local/workspace | N/A | L / 0 B | opaque/N/A | Host | 无显式 consumer | runtime/caller | 无 UB 子区 |

##### 分层内存布局

N/A：直接 GM 与线程 FP32 标量路径，无 AIV UB/shared memory/workspace。

##### 字节账与复用顺序

```text
UB_total = N/A
dynamic_ub = 0 B body-addressable；launch L=UB_platform-128KiB
workspace_total = 0 B
register_bytes = 固定仓不可见，仅能证明局部值不跨迭代
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 thread locals | casts/div/mul/add | BF16 cast/store | 下一 j | thread sequential semantics | 不共享、不跨 j |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | C++ cast | GM loads | div/mul/add | thread locals | 显式升精度 |
| FP32→BF16 | C++ cast | FP32 result | GM store | y[j] | 具体 rounding mode 固定仓未显式命名 |
| tail/sync | index predicate / N/A | grid | thread | valid output | 空 tensor skip；`j<numel`；无 barrier |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling cpp | key2 path | BF16 selector |
| Kernel entry/router | arch35 entry cpp | schMode2 | BF16 Process |
| Device body/helper | SIMT header | `ProcessBf16`; BF16 SIMT kernel | FP32 intermediate, grid-stride tail |

## 算子特定优化

### Trick T-001：用最大单 Tensor 定 blockDim，并让同一网格复用到整张列表

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：所有合法输入；`needCoreNum=min(ceil(maxTensorElements/1024),coreNum)`，至少 1 核。
- Golden 来源：[`foreach_addcdiv_scalar_list_tiling.cpp:100-123`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_host/arch35/foreach_addcdiv_scalar_list_tiling.cpp#L100-L123)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    for (uint64_t i = 0; i < tensorNum; i++) {
        auto idxTensorShapePtr = context->GetDynamicInputShape(INPUT_IDX_X1, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, idxTensorShapePtr);
        auto idxTensorShape = idxTensorShapePtr->GetStorageShape();
        int64_t numel = idxTensorShape.GetShapeSize();
        tilingData->perTensorElementNum[i] = numel;
        maxTensorElements = std::max(maxTensorElements, numel);

        if (i == 0) {
            auto idxTensorDtypePtr = context->GetDynamicInputDesc(INPUT_IDX_X1, i);
            OP_CHECK_NULL_WITH_CONTEXT(context, idxTensorDtypePtr);
            dataType = idxTensorDtypePtr->GetDataType();
        }
    }

    int64_t needCoreNum = std::max(static_cast<int64_t>(1),
        (maxTensorElements + SINGLE_CORE_MIN_ELEMENTS - 1) / SINGLE_CORE_MIN_ELEMENTS);
    needCoreNum = std::min(needCoreNum, coreNum);

    tilingData->tensorNum = static_cast<int32_t>(tensorNum);
    tilingData->needCoreNum = static_cast<int32_t>(needCoreNum);

    context->SetBlockDim(needCoreNum);
    context->SetTilingKey(GetTilingKeyByDtype(dataType));
```

#### 机制与设计原因

Host 不按列表总元素数扩大 grid，而按最大 tensor 选择 grid；Device 中每个 block 复用这张网格逐个处理所有 tensor。这样无需把不等长 tensor 拼接或生成逐核 start/end POD，POD 只保存每项 numel。每个线程用 `blockNum*512` 步长覆盖任意较大 tensor，较小 tensor 由边界判断自然减少活跃线程。源码证明减少了分核元数据与地址映射复杂度；未执行性能测量。

#### 迁移边界

- 适用：列表项彼此独立、每项相同逐元素表达式、允许所有 block 顺序遍历列表。
- 前提：列表最多 256；所有 block/threads 对 y 元素映射互斥；空 tensor skip。
- 失效：若 tensor 间有依赖、归约/atomic 或每项需要不同 launch 资源，不能直接复用该网格。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；把 `needCoreNum` 改为列表总元素推导而不改变 Device 网格，会让每个 tensor 都使用过大的相同 grid，改变既有调度边界。

### Trick T-002：SIMT grid-stride 直接 GM 计算，消除显式 UB 搬运与 tail DMA

#### 触发与源码依据

- 适用 Route：R-002；R-001/R-003 使用同一 index 网络并增加 cast。
- 触发条件：非空 tensor；512-thread VF_CALL。
- Golden 来源：[`foreach_addcdiv_scalar_list_simt.h:57-72`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list_simt.h#L57-L72)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    MidT scalar_val = scalars_gm[tensorIdx];

    for (uint64_t index = static_cast<uint64_t>(
            AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum()
            + AscendC::Simt::GetThreadIdx());
         index < static_cast<uint64_t>(perTensorElementNum);
         index += static_cast<uint32_t>(AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum())) {
        MidT x1_val = x1_i_gm[index];
        MidT x2_val = x2_i_gm[index];
        MidT x3_val = x3_i_gm[index];

        // Single expression: enables compiler FMA fusion for x1 + scalar*(x2/x3)
        MidT result = x1_val + scalar_val * (x2_val / x3_val);

        y_i_gm[index] = result;
    }
```

#### 机制与设计原因

每线程直接处理唯一 GM 元素，循环步长等于全网格线程数；`index<numel` 同时承担 tail 保护，所以不需要 AIV `DataCopyPad`、UB Queue 或尾块 mask。FP32 路线把表达式保持为单句，源码意图是给编译器融合机会；是否实际生成 FMA 固定仓和本次验证均不能证明。

#### 迁移边界

- 适用：无跨元素依赖、直接 GM 访问可接受的 pointwise 算子。
- 前提：输出索引互斥；不需要共享缓存、barrier 或 atomic；保留 64-bit index 与边界判断。
- 失效：需要邻域复用、归约、复杂广播或受 DMA 合并约束时，应设计 UB/shared-memory 路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；删除 `index<perTensorElementNum` 或把 stride 缩成 block 内线程数会造成越界或跨 block 重复写。

## 模板与实现对比

| 对比维度 | R-001 FP16→FP32 | R-002 FP32 | R-003 BF16→FP32 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode0 | schMode1 | schMode2 | dtype selector |
| 模板与 dtype | 专用 FP16 body | generic `<float,float>` | 专用 BF16 body | 低精度显式升 FP32 |
| 循环与分核 | 相同 512-thread grid-stride | 同 | 同 | dtype 不改变 index 网络 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue；thread FP32 locals | thread FP32 locals | thread FP32 locals | 直接 GM SIMT |
| 内存总占用与复用 | 无可寻址 UB/workspace | 同 | 同 | Host L 未被 body 显式寻址 |
| 搬运/Vector/SIMT/NDDMA/通信 API | direct load/store SIMT | 同 | 同 | 无 DMA 网络 |
| 同步、精度与 tail | FP16→FP32→FP16；predicate tail | FP32 single expression | BF16→FP32→BF16；round模式未命名 | 精度 body 不同故拆 Route |
| fallback/失败边界 | 非法 dtype 不能落入默认 key0 | 无 fallback | 无 fallback | 上游 dtype guard 必须生效 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 Foreach Tiling，POD 50 tensor/core，workspace 32 B | 同 A2 | arch35 独立 Tiling，256 tensor，workspace 0，设置 `L` |
| Kernel/资源 | 单 AIV Vector，Queue/UB tile | 同 A2 | 单 SIMT VF，512 threads，direct GM |

## 限制与失败边界

- 公开约束：FP16/FP32/BF16；0–8 维；三个 TensorList 等长且逐位置 shape/dtype 一致；`scalars.numel==tensorNum`；输出不支持非连续。
- arch35 Tiling 显式上限 256 个 tensor；公开文档未写该数值。Tiling 只重读 `x1` 与首项 dtype，`x2/x3/scalars/out` 的一致性依赖 ACLNN/框架校验，不得绕过公开入口直接复用 Host。
- `UB<=128KiB` 时 Tiling 失败；`SetLocalMemorySize(L)` 的 runtime 映射和真实消耗固定仓不可见，Device body没有显式 UB consumer。
- FP16/BF16 最终 C++ cast 的具体硬件 rounding mode未在源码命名；不能从 A2/A3 的 `CAST_RINT` 推断到 950。
- API bridge、executor 总 workspace、底层 VF launch 细节、寄存器字节数与 stream/event 实现固定仓不可见。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_addcdiv_scalar_list/docs/aclnnForeachAddcdivScalarList.md` | ACLNN 两段式接口、产品/错误表 | 公开性、950 支持、约束 |
| 入口实现 | `foreach/foreach_addcdiv_scalar_list/op_host/foreach_addcdiv_scalar_list_def.cpp` | `ForeachAddcdivScalarList`; `OP_ADD` | bridge 落点、dtype/format/AutoContiguous/950 注册 |
| Host Infer | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | common infer + AddcdivScalarList 注册 | 输出 shape/dtype |
| Host/POD | `foreach/foreach_addcdiv_scalar_list/op_host/arch35/foreach_addcdiv_scalar_list_tiling.h` | Host POD/CompileInfo | ABI 字段、平台信息 |
| Host/Tiling | `foreach/foreach_addcdiv_scalar_list/op_host/arch35/foreach_addcdiv_scalar_list_tiling.cpp` | Tiling Func/Parse | key、blockDim、L、workspace、POD producer |
| Kernel entry/body | `foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list.cpp` | templated entry | schMode router |
| Kernel body | `foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list_simt.h` | `Process*`; three VF kernels; pointer helper | threads、GM、精度、tail |
| POD | `foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list_tiling_data.h` | Device POD | Host/Device ABI 对照 |
| selector | `foreach/foreach_addcdiv_scalar_list/op_kernel/arch35/foreach_addcdiv_scalar_list_tiling_key.h` | `schMode` template list | 0/1/2 实例 |
| 构建接线 | `foreach/foreach_addcdiv_scalar_list/op_host/CMakeLists.txt` | 950→arch35 | Host 选择 |
| 构建接线 | `foreach/foreach_addcdiv_scalar_list/op_kernel/CMakeLists.txt` | arch35 kernel source/ascend950 | Kernel 选择 |
| 构建接线 | `foreach/foreach_addcdiv_scalar_list/op_host/config/ascend950/foreach_addcdiv_scalar_list_binary.json` | dtype op_list | 三种 950 binary 签名 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2056 B、align8，字段 offset 0/4/8 | 256 长度、int32/int64 位宽、inactive needCoreNum仍保留 | Host/Device POD 定义 |
| 调用顺序 | entry→Process→每 tensor VF_CALL | 每个外层 block 遍历同一列表；空 tensor先 skip | entry；`Process*` |
| workspace/内存 | kernel workspace 0；direct GM | 不得把 L 臆造成 body buffer；ListTensor 指针表协议不变 | Tiling Func；`SimtGetTensorAddr` |
| 同步 | 无共享写、无 atomic/barrier | grid-stride 必须保证 y[j] 唯一 | three SIMT kernels |
| 精度与尾块 | FP16/BF16 在 FP32 计算；FP32直算；predicate tail | 低精度 cast顺序、512 threads、64-bit index | SIMT header |
| 失败边界 | tensor≤256、UB>128KiB、公开 dtype guard | 非法 dtype default0、仓外 ACLNN校验不能省略 | Host Tiling; OpDef/docs |