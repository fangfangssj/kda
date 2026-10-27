---
type: Operator
title: "ForeachTanh 950 ACLNN 算子知识"
description: "说明 aclnnForeachTanh 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/docs/aclnnForeachTanh.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_host/foreach_tanh_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_host/arch35/foreach_tanh_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_kernel/arch35/foreach_tanh_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_kernel/arch35/foreach_tanh_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_kernel/arch35/foreach_tanh.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_kernel/arch35/foreach_tanh_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_host/config/ascend950/foreach_tanh_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:22:14Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachTanh 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式 `aclnnForeachTanhGetWorkspaceSize` / `aclnnForeachTanh`，无其它公开版本 |
| 计算语义 | 对 TensorList 每项逐元素计算 `tanh` |
| 输入输出 | 动态 TensorList；ND；FLOAT32/FLOAT16/BFLOAT16；每张量 0--8 维；输入可非连续、输出连续 |
| 执行类型 | 单 Kernel；唯一 Device 节点为 SIMT，外层 AscendC entry 内逐张量 `VF_CALL` |
| Kernel 路线 | R-001 FP32；R-002 FP16→FP32→FP16；R-003 BF16→FP32→BF16 |
| 核心资源 | TensorList GM 描述符、每线程标量寄存器、512 threads/block；无 TQue/TBuf、无 shared/dynamic UB 数据区、workspace 0 |
| 来源闭包 | `source_closed`：公开 ACLNN 约束→生成式 executor/连续化框架边界→950 OpDef/config→arch35 Tiling/POD/template key→entry→SIMT body→构建接线闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `x=[x_0,...,x_{n-1}]`，对每个 `i` 和 `0<=j<numel(x_i)` 写 `y_i[j]=tanh(x_i[j])`。FP32 直接调用 `tanhf`；FP16/BF16 显式升为 FP32，调用 `tanhf` 后降回原 dtype。空张量在外层循环中直接跳过。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每张量 0--8 维、ND | FLOAT32/FLOAT16/BFLOAT16 | 列表指针非空、列表内 dtype 一致；支持空张量与非连续输入，连续化属于生成式 ACLNN 框架边界 |
| `out` | TensorList；每项容量不小于输入、ND | 与 `x` 相同 | 输出列表 dtype 一致；公开接口不支持非连续输出；仅输入元素对应前缀被写 |
| `workspaceSize` | 标量 | `uint64_t*` | 950 算子 Tiling 写 0；框架连续化若需额外空间，其公式固定仓不可见 |
| `executor` | 不适用 | `aclOpExecutor**` | 第一段接口产生，第二段前保持有效 |
| `workspace` / `stream` | Device 地址 / stream | 不适用 | entry 保留 workspace 参数但不读取；stream 承载单 Kernel launch |

### 入口与版本

固定提交仅公开无版本后缀 API。公开产品表明确支持 Ascend 950PR/950DT；OpDef `AddConfig("ascend950")`、ascend950 binary config 的三个 dtype 条目、Host CMake 的 `ascend950→arch35`、Kernel CMake 的 `arch35/foreach_tanh.cpp` 共同证明 950 接线。自动生成 ACLNN inner bridge 不在固定仓，作为明确框架边界；operator-owned selector、POD、router 和 SIMT body 已闭合。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段公开错误语义覆盖空指针、非法/不一致 dtype、输出容量不足、列表内 dtype 不一致和维数大于 8。OpDef 对输入输出声明动态 ND、三种 dtype 与 `AutoContiguous()`。进入 arch35 Tiling 后，`GetPlatformInfo` 要求 AIV 核数和 UB 非 0；随后读取输入实例数、首项 dtype以及每项 storage shape 元素数。任一 context 指针为空或 dtype 不支持立即 `GRAPH_FAILED`。

### 参数、Workspace 与生命周期

设 `M=max_i numel(x_i)`、平台 AIV 核数 `Cmax`。Host 计算 `perCore=align_up(max(1024,ceil(M/Cmax)),32)` elements，`C=max(1,min(Cmax,ceil(M/perCore)))`。POD 仅保存列表项数、`C` 和每项元素数；`perCore` 不进入 Device。Host 设置 `blockDim=C`、`localMemorySize=uint32(UB-128KiB)`、workspace 0。SIMT body没有显式使用该 local-memory 区；实际数据直接从 GM 读入线程标量并写回 GM。

### 启动图与失败传播

```text
aclnnForeachTanhGetWorkspaceSize(x,out)
  -> 生成式校验/连续化/executor（固定仓框架边界）
  -> ForeachTanhTilingFunc -> POD/key/blockDim/localMemorySize/workspace=0
aclnnForeachTanh(...,stream)
  -> K-001 foreach_tanh<schMode><<<blockDim=C, localMemory=UB-128KiB, stream>>>
     -> NsForeachTanh::Process<T>
        -> 对每个非空张量执行 VF_CALL<dtype body>(threads=512)
```

逻辑上只有 K-001 一个 Device launch，无跨 Kernel 边。K-001 内的 `VF_CALL` 是 SIMT 向量函数调用，不是 Host 侧第二个 Kernel 节点。失败的 Tiling/launch 阻断输出；固定仓未显示 `VF_CALL` 内部异常传播接口。

## Route 与参数设计

### 分核与资源计算

`C` 由最大张量而非总元素数决定；所有张量复用同一 `C×512` 线程网格。每个线程全局起点为 `blockIdx*threadNum+threadIdx`，步长为 `threadNum*blockNum`，并以 `idx<elemCount` 守护。`localMemorySize=UB-128KiB` 是 launch 参数，但三条 body 都没有 shared/dynamic UB 分配、barrier 或 atomic。workspace 总量为 0。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 | schMode 0 / `Process<float>` | `dataType == DT_FLOAT` | 检查 FP16 |
| 2 | R-002 FP16 升精度 | schMode 1 / `Process<half>` | `dataType == DT_FLOAT16` | 检查 BF16 |
| 3 | R-003 BF16 升精度 | schMode 2 / `Process<bfloat16_t>` | `dataType == DT_BF16` | 其它 dtype Tiling 失败 |

```text
DT_FLOAT -> GET_TPL_TILING_KEY(0) -> foreach_tanh<0> -> R-001
else DT_FLOAT16 -> GET_TPL_TILING_KEY(1) -> foreach_tanh<1> -> R-002
else DT_BF16 -> GET_TPL_TILING_KEY(2) -> foreach_tanh<2> -> R-003
else -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | FP32、合法列表、950 | K-001 SIMT | `Process<float>`→`OpForeachTanhSimt<float>` | GM→线程 float→GM；无 UB/barrier | grid-stride `idx<count`；空张量跳过 |
| R-002 FP16 | schMode 1 | FP16、合法列表、950 | K-001 SIMT | `Process<half>`→`OpForeachTanhHalfSimt` | GM half→线程 float→GM half；无 UB/barrier | 同上 |
| R-003 BF16 | schMode 2 | BF16、合法列表、950 | K-001 SIMT | `Process<bfloat16_t>`→`OpForeachTanhBf16Simt` | GM BF16→线程 float→GM BF16；无 UB/barrier | 同上 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台资源 | `coreNum!=0 && ubSize!=0` | launch 需要有效 grid 和 local-memory 参数 | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| local memory 扣减 | `uint32(ubSize-128KiB)` | 为 DCache 留 128KiB；body不消费余下 UB | 全部 | 无显式下溢检查或 fallback | `ForeachTanhTilingFunc` |
| TensorList 容量 | `tensorElements[256]` | POD 固定 256 项 | 全部 | Host 未在写数组前显式拒绝 `tensorNum>256`；安全复用上界 256 | `ForeachTanhTilingData` / Tiling loop |
| dtype | 三分支精确匹配 | entry 仅实例化三种模板 | 对应 Route | 不支持 dtype失败 | `ForeachTanhTilingFunc` |
| 零元素 | `count==0` | 无需发出 VF_CALL | 全部 | continue | `NsForeachTanh::Process` |
| 线程数 | 编译常量 `THREAD_NUM=512` 与 `LAUNCH_BOUND(512)` | 每个 VF_CALL 固定 block 线程数 | 全部 | 固定仓无其它线程配置 | `foreach_tanh_simt.h` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachTanhTilingData` | `tensorNum` | `int32_t`/32 | offset 0；struct align 8 | tensor count | `int32(GetInstanceNum())` | `Process` 外层循环上界 | 安全上界 256；size_t→int32 窄化未检查 |
| 同上 | `needCoreNum` | `int32_t`/32 | offset 4 | AIV/SIMT blocks | Host 公式 `C` | entry launch 的 blockDim由 context消费；body通过 `GetBlockNum()`观察 | `>=1` |
| 同上 | `tensorElements[256]` | `int64_t[256]` | offset 8；2048 bytes；struct `sizeof=2056` | elements/item | 每项 `shape.GetShapeSize()` | `Process` 的 count 与线程循环上界 | 未使用项由 memset 置 0 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | schMode 0 | `arch35/foreach_tanh_tiling.cpp::ForeachTanhTilingFunc` | `ForeachTanhTilingData` | `foreach_tanh<0>` | `Process<float>` / `OpForeachTanhSimt<float>` | arch35 Kernel CMake + ascend950 binary json |
| R-002 FP16 | schMode 1 | 同上 | 同上 | `foreach_tanh<1>` | `Process<half>` / `OpForeachTanhHalfSimt` | 同上 |
| R-003 BF16 | schMode 2 | 同上 | 同上 | `foreach_tanh<2>` | `Process<bfloat16_t>` / `OpForeachTanhBf16Simt` | 同上 |

## Kernel 实现

### 公共执行流程

编译期 `schMode` 选择 `T`，entry 解析 plain struct POD 后调用 `Process<T>`。`ListTensorDesc` 将输入/输出 TensorList 描述符解析为每项 GM 指针。所有 active blocks 都按相同列表次序访问；对每个非空张量调用一个 512-thread SIMT body。每线程只持有当前标量及 FP16/BF16 路线的 FP32 中间值，无 shared memory、barrier、atomic 或 Queue。不同线程写互斥 `idx`，跨张量调用按当前 entry 的程序次序发生。

### Route R-001：FP32 直接 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/arch35/foreach_tanh_tiling.cpp` | `ForeachTanhTilingFunc` | `dataType==DT_FLOAT` | template key schMode 0 |
| Router | `op_kernel/arch35/foreach_tanh.cpp` | `foreach_tanh<schMode>` | `schMode==0` | `Process<float>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_tanh<0> -> Process<float>
  -> ListTensorDesc::GetDataPtr<float>(i)
  -> VF_CALL<OpForeachTanhSimt<float>>(Dim3(512), xPtr, yPtr, count)
     -> tanhf(float)
```

##### 源码等价伪码

```text
for tensor i in [0,tensorNum):
  count=tensorElements[i]; if count==0: continue
  xPtr=xList[i]; yPtr=yList[i]
  parallel block b in [0,C), thread t in [0,512):
    for idx=b*512+t; idx<count; idx+=512*C:
      val=xPtr[idx]
      yPtr[idx]=tanhf(val)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `x_i` | `xList.GetDataPtr<float>(i)+idx` | count elements | FP32/ND | caller | thread | caller，launch期间 | 标量 load |
| thread scalar `val` | register，物理布局不可见 | 1 float | FP32 | GM load | `tanhf` | 单次 loop iteration | 下一 idx覆盖 |
| GM `y_i` | `yList.GetDataPtr<float>(i)+idx` | count elements | FP32/ND | `tanhf` | caller | caller | 标量 store，线程地址互斥 |
| launch local memory | N/A body未引用 | `UB-128KiB` launch 参数 | 未使用 | Host | N/A | launch | 不分配数据区 |
| workspace | N/A | 0 | N/A | N/A | N/A | N/A | N/A |

##### 分层内存布局

N/A：该 SIMT 路线没有 AIV UB Queue/TBuf 或 shared/dynamic UB 数据布局；真实替代路径是 GM 标量 load→线程寄存器→GM 标量 store。

##### 字节账与复用顺序

```text
UB_total = 0（body 显式分配）
dynamic_ub = 0（body未声明动态/共享数组；Host仅设置 localMemorySize=UB-128KiB）
workspace_total = 0
register_bytes = 固定仓不可见；源码可见生命周期为每线程 1 个 float val/result
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread scalar | GM load | `tanhf`/GM store | 下一 grid-stride iteration | 单线程程序次序 | `idx<count` |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| thread mapping | grid-stride loop | Host grid + thread id | load/store | 输出索引 | `idx<count` 处理任意尾数 |
| compute | N/A barrier；线程私有 | load | `tanhf`→store | scalar | FP32直接计算 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tanh/op_host/arch35/foreach_tanh_tiling.cpp` | `ForeachTanhTilingFunc` | schMode 0、C、POD、local memory、workspace |
| Kernel entry/router | `foreach_tanh/op_kernel/arch35/foreach_tanh.cpp` | `foreach_tanh<0>` | float模板映射 |
| Device body/helper | `foreach_tanh/op_kernel/arch35/foreach_tanh_simt.h` | `OpForeachTanhSimt<float>` | 512-thread grid-stride 与 tanhf |

### Route R-002：FP16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/arch35/foreach_tanh_tiling.cpp` | `ForeachTanhTilingFunc` | `dataType==DT_FLOAT16` | schMode 1 |
| Router | `op_kernel/arch35/foreach_tanh.cpp` | `foreach_tanh<1>` | `schMode==1` | `Process<half>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_tanh<1> -> Process<half>
  -> VF_CALL<OpForeachTanhHalfSimt>(Dim3(512),...)
     -> __half2float -> tanhf -> __float2half
```

##### 源码等价伪码

```text
对每个非空张量和每个 grid-stride idx：
  h=xPtr[idx]
  f=__half2float(h)
  r=tanhf(f)
  yPtr[idx]=__float2half(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | list item + idx | count | FP16/ND | caller/thread | thread/caller | caller | 标量 load/store |
| thread scalars | register，布局不可见 | 1 half + FP32 val/result | mixed | load/conversion/tanhf | next stage/store | iteration内 | 下一 idx覆盖 |
| local memory/workspace | N/A body未用 | launch `UB-128KiB` / workspace 0 | N/A | Host | N/A | launch | 无共享复用 |

##### 分层内存布局

N/A：SIMT body只使用线程私有标量寄存器，未分配 AIV UB、dynamic UB 或 shared buffer。

##### 字节账与复用顺序

```text
UB_total=0（显式）
dynamic_ub=0（body显式）；workspace_total=0
register_bytes=固定仓不可见；可见生命周期为 half val、float valF32/resultF32
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 scalars | conversion/tanhf | tanhf/downcast | 下一 idx | 单线程程序次序 | idx guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升/降精度 | `__half2float` / `__float2half` | GM half / FP32 result | tanhf / GM store | thread scalar | tanh在FP32执行；转换舍入模式由 intrinsic 定义，固定仓未展开 |
| tail | grid-stride guard | thread ids | load/store | output index | 任意 count，无 padding lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tanh/op_host/arch35/foreach_tanh_tiling.cpp` | dtype branch | schMode 1 |
| Kernel entry/router | `foreach_tanh/op_kernel/arch35/foreach_tanh.cpp` | `foreach_tanh<1>` | half route |
| Device body/helper | `foreach_tanh/op_kernel/arch35/foreach_tanh_simt.h` | `OpForeachTanhHalfSimt` | FP32升精度 tanhf |

### Route R-003：BF16 升精度 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host/Tiling | `op_host/arch35/foreach_tanh_tiling.cpp` | `ForeachTanhTilingFunc` | `dataType==DT_BF16` | schMode 2 |
| Router | `op_kernel/arch35/foreach_tanh.cpp` | `foreach_tanh<2>` | `schMode==2` | `Process<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_tanh<2> -> Process<bfloat16_t>
  -> VF_CALL<OpForeachTanhBf16Simt>(Dim3(512),...)
     -> __bfloat162float -> tanhf -> __float2bfloat16
```

##### 源码等价伪码

```text
对每个非空张量和每个 grid-stride idx：
  b=xPtr[idx]
  f=__bfloat162float(b)
  r=tanhf(f)
  yPtr[idx]=__float2bfloat16(r)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM x/y | list item + idx | count | BF16/ND | caller/thread | thread/caller | caller | 标量 load/store |
| thread scalars | register，布局不可见 | 1 BF16 + FP32 val/result | mixed | load/conversion/tanhf | next stage/store | iteration内 | 下一 idx覆盖 |
| local memory/workspace | N/A body未用 | launch `UB-128KiB` / workspace 0 | N/A | Host | N/A | launch | 无共享复用 |

##### 分层内存布局

N/A：SIMT body只使用线程私有标量寄存器，未分配 AIV UB、dynamic UB 或 shared buffer。

##### 字节账与复用顺序

```text
UB_total=0（显式）
dynamic_ub=0（body显式）；workspace_total=0
register_bytes=固定仓不可见；可见生命周期为 BF16 val、float valF32/resultF32
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread FP32 scalars | conversion/tanhf | tanhf/downcast | 下一 idx | 单线程程序次序 | idx guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升/降精度 | `__bfloat162float` / `__float2bfloat16` | GM BF16 / FP32 result | tanhf / GM store | thread scalar | tanh在FP32执行；intrinsic舍入细节固定仓未展开 |
| tail | grid-stride guard | thread ids | load/store | output index | 任意 count，无 padding lane |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tanh/op_host/arch35/foreach_tanh_tiling.cpp` | dtype branch | schMode 2 |
| Kernel entry/router | `foreach_tanh/op_kernel/arch35/foreach_tanh.cpp` | `foreach_tanh<2>` | BF16 route |
| Device body/helper | `foreach_tanh/op_kernel/arch35/foreach_tanh_simt.h` | `OpForeachTanhBf16Simt` | FP32升精度 tanhf |

## 算子特定优化

### Trick T-001：所有张量复用按最大张量定宽的 grid-stride 网格

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：任一非空 TensorList 项；Host 已令 `blockDim=needCoreNum`，每次 `VF_CALL` 固定 512 threads
- Golden 来源：[`foreach_tanh_simt.h:87-104`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_tanh/op_kernel/arch35/foreach_tanh_simt.h#L87-L104)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (int32_t i = 0; i < tilingData->tensorNum; i++) {
    int64_t count = tilingData->tensorElements[i];
    if (count == 0) {
        continue;
    }

    __gm__ T* xPtr = xList.GetDataPtr<T>(i);
    __gm__ T* yPtr = yList.GetDataPtr<T>(i);

    if constexpr (std::is_same_v<T, float>) {
        Simt::VF_CALL<OpForeachTanhSimt<float>>(
            Simt::Dim3(THREAD_NUM), xPtr, yPtr, count);
```

#### 机制与设计原因

Host 只依据最大张量确定一次 blockDim；Device 对每个列表项复用这张网格，SIMT body再用 `idx += threadNum*blockNum` 覆盖任意元素数。这样 POD 不需要为每个张量保存 tile/core 分段，也不物化中间 workspace；空张量在发出 `VF_CALL` 前被跳过。固定源码只证明减少参数与中间物化，未提供实测性能数据。

#### 迁移边界

- 适用：列表项彼此独立、逐元素、无跨元素归约或共享状态的 SIMT unary 算子。
- 前提：每线程地址互斥；最大张量足以决定共享 grid；每项元素数可由 64-bit loop guard表达。
- 失效：需要按张量定制 block 数、共享内存、跨线程归约或跨项同步时必须更换调度结构。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | schMode 0 | schMode 1 | schMode 2 | dtype 决定编译期实例 |
| 模板与 dtype | float direct | half→float→half | BF16→float→BF16 | 低精度路线用 FP32 tanhf |
| 循环与分核 | `C×512` grid-stride | 同左 | 同左 | 共用最大张量 grid |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程 float scalar；无 UB | half+FP32 scalars；无 UB | BF16+FP32 scalars；无 UB | 无跨线程数据 |
| 内存总占用与复用 | workspace/显式 UB 0 | 同左 | 同左 | 直接 GM 标量路径 |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM + tanhf | GM + half convert + tanhf | GM + BF16 convert + tanhf | dtype conversion |
| 同步、精度与 tail | 无 barrier；FP32；idx guard | 无 barrier；FP32 compute；idx guard | 同左 | 线程私有 |
| fallback/失败边界 | 非匹配去下一 dtype | 同左 | 非三 dtype失败 | 无运行时 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 Foreach Tiling | arch22 公共 Foreach Tiling | arch35 专用最大张量分核、workspace 0 |
| Kernel/资源 | AIV Vector Queue body | 同 A2 | SIMT 512-thread、线程标量、无显式 UB |

## 限制与失败边界

- 仅支持 ND 的 FLOAT32/FLOAT16/BFLOAT16，单张量 0--8 维；输出必须连续且容量不小于输入，超出输入元素数的输出尾部不写。
- POD 固定 `tensorElements[256]`；Host 在循环写入前没有 `tensorNum<=256` 检查，故实现复用必须把 256 当安全上界，不能把更长列表推定为支持。
- Tiling 从首个动态输入 desc 取 dtype；空 TensorList 会在该处失败，公开“支持空 tensor”不等价于支持零项列表。
- `tensorNum` 从 `size_t` 窄化为 `int32_t`，`localMemorySize` 从 `ubSize-128KiB` 窄化为 `uint32_t`，固定源码未作溢出/下溢检查。
- `ListTensorDesc` 的描述符内部 ABI、`VF_CALL` 的底层调度与浮点转换 intrinsic 的舍入细节属于框架/库边界；算子拥有的参数和调用关系已闭合。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_tanh/docs/aclnnForeachTanh.md` | ACLNN 原型、产品表 | 950 公开支持与接口约束 |
| 入口实现 | `foreach/foreach_tanh/op_host/foreach_tanh_def.cpp` | `ops::ForeachTanh` | ascend950 注册、dtype、AutoContiguous |
| Host/Tiling/Launcher | `foreach/foreach_tanh/op_host/arch35/foreach_tanh_tiling.cpp` | `ForeachTanhTilingFunc` | grid、key、POD、local memory、workspace |
| Host/Tiling/Launcher | `foreach/foreach_tanh/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT` / `SUPPORT_TILING_DIR` | ascend950 到 arch35 的活动映射 |
| Host/Tiling/Launcher | `foreach/foreach_tanh/op_kernel/arch35/foreach_tanh_tiling_data.h` | `ForeachTanhTilingData` | plain POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_tanh/op_kernel/arch35/foreach_tanh_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个 schMode 实例 |
| Kernel entry/body | `foreach/foreach_tanh/op_kernel/arch35/foreach_tanh.cpp` | `foreach_tanh<schMode>` | 编译期 router |
| Kernel entry/body | `foreach/foreach_tanh/op_kernel/arch35/foreach_tanh_simt.h` | `Process` / 三个 SIMT body | 逐张量 launch、线程映射、精度与 tail |
| 构建接线 | `foreach/foreach_tanh/op_host/config/ascend950/foreach_tanh_binary.json` | `ForeachTanh` 三 dtype条目 | 950 binary config |
| 构建接线 | `foreach/foreach_tanh/op_kernel/CMakeLists.txt` | arch35 `add_kernel_sources` | ascend950 选择专用 Kernel |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 两个 int32 + 256个 int64 | offset、2056-byte大小、256上界 | `ForeachTanhTilingData` |
| 调用顺序 | entry→Process→逐非空项 VF_CALL | 同一列表次序、每项先取 x/y 指针 | `Process<T>` |
| workspace/内存 | workspace 0；显式 UB 0 | localMemorySize扣 128KiB仍是 launch ABI | `GetWorkspaceSize` / `SetLocalMemorySize` |
| 同步 | 线程私有、无 barrier/atomic | 每线程互斥索引和单线程程序次序 | 三个 SIMT body |
| 精度与尾块 | FP32直算；FP16/BF16升到 FP32 | 转换顺序、512 threads、grid-stride guard | `OpForeachTanh*Simt` |
| 失败边界 | context/dtype/resource失败即终止 | 256项、非空列表、窄化边界不可外推 | `ForeachTanhTilingFunc` |