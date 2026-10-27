---
type: Operator
title: "ForeachMulList 950 ACLNN 算子知识"
description: "说明 aclnnForeachMulList 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/docs/aclnnForeachMulList.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_host/foreach_mul_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_host/arch35/foreach_mul_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_host/config/ascend950/foreach_mul_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:47:11Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachMulList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMulListGetWorkspaceSize` + `aclnnForeachMulList`，无版本后缀公开 ACLNN |
| 计算语义 | `y_i[j]=x1_i[j]*x2_i[j]`，同位置 Tensor、同位置元素一一对应 |
| 输入输出 | 动态 TensorList；ND；FP32/FP16/BF16/INT32；单 Tensor 0～8 维 |
| 执行类型 | 单节点 `SIMT`：一个 AICore entry 内按 Tensor 交集调用 512-thread VF；无 Cube/AIC 计算或 `MIX_CV` |
| Kernel 路线 | R-001 FP32；R-002 FP16 升 FP32；R-003 BF16 升 FP32；R-004 INT32 |
| 核心资源 | TensorList GM 描述符、SIMT thread-local 标量；无显式 AIV UB Queue/TBuf、无 shared/dynamic UB、无 atomic；算子 workspace 0 B |
| 来源闭包 | `source_closed`：公开 ACLNN 约束→OpDef/InferShape→arch35 Tiling/POD/key→SIMT entry/body→Ascend 950 构建接线闭合；生成式 ACLNN bridge 是框架边界 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

对长度为 `n` 的两个 TensorList，逐项逐元素相乘。Host 将 `x1` 每个 Tensor 的 storage shape 元素数记录为 `N_i`，并以逻辑拼接域 `[0,N)`（`N=ΣN_i`）分配 AICore block；Device 再把每核区间与各 Tensor 的逻辑区间求交，GM 本身不拼接、不重排。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；每 Tensor 0～8 维；ND | FP32/FP16/BF16/INT32 | 支持空 Tensor；列表内 dtype 一致；OpDef `AutoContiguous()` |
| `x2` | 列表长度及逐 Tensor shape 与 `x1` 相同；ND | 与 `x1` 相同 | 支持空 Tensor；列表内 dtype 一致；OpDef `AutoContiguous()` |
| `out` | 对应输出 shape size 至少为 `x1`；ND | 与 `x1` 相同 | 不支持用户传入非连续输出；Kernel 只写 `x1` 有效元素域；InferShape 默认复制 `x1` shape/dtype |

无属性。arch35 Tiling 明确拒绝 `tensorNum>256`；空总域仍设置 `blockDim=1`，Kernel 随即因 `coreStart>=coreEnd` 返回。

### 入口与版本

固定仓仅公开无后缀版本 `aclnnForeachMulList`，没有更高同语义版本。公开产品表明确支持 Ascend 950PR/950DT；OpDef 注册 `ascend950`，Host CMake 选择 arch35 Tiling，Kernel CMake 只为 `ascend950` 选择 `arch35/foreach_mul_list.cpp`，证据一致。ACLNN 两段式 inner bridge 的生成实现不在固定仓中，但算子自有链路完整可见。

## Host/Launcher 执行流程

### 参数校验与前处理

1. ACLNN 第一段按公开契约检查空指针、四种 dtype、列表/shape/dtype/维数关系；OpDef 还把输入标为 `AutoContiguous()`。生成 bridge 的逐行检查与临时连续化实现固定仓不可见。
2. 公共 InferShape 要求输出动态实例数等于 `x1` 实例数，并逐项复制 `x1` shape/dtype。
3. arch35 `ForeachMulListTilingFunc` 获取 AIV core 数与 UB；两者为 0 即失败。
4. 从 `x1` 的 instance info 取得 `tensorNum`，大于 256 即失败；逐 Tensor 写 `tensorSizes[i]` 并求和 `totalElements`。
5. dtype 只读取 `x1[0]` descriptor；不属于 FP32/FP16/BF16/INT32 即失败。x2/out 的一致性依赖 API/OpDef/InferShape 框架契约，arch35 Tiling 本身不重复检查。

### 参数、Workspace 与生命周期

Tiling 将算子 workspace 写为 0 B；Kernel 的 `workspace` 参数未读。第一段 ACLNN 若因输入非连续而产生框架临时 workspace，其精确公式固定仓不可见，调用者仍必须按第一段返回值分配并保活到 stream 工作完成。

`SetLocalMemorySize(ubSize-128KiB)` 把扣除 DCache 后的 local memory 值交给 launcher；当前 Kernel body 没有显式 UB/shared/dynamic buffer 或对应消费符号，不能把该值虚构成某个数组。POD 和 TensorList 描述符在单次 launch 内只读，输出 GM 在完成后归调用者。

### 启动图与失败传播

```text
aclnnForeachMulListGetWorkspaceSize
  -> 框架校验/必要连续化
  -> ForeachMulListTilingFunc
     -> GetPlatformInfo
     -> 统计 tensorSizes/totalElements
     -> 产生 blockDim/POD/localMemorySize/key/workspace=0
  -> executor 记录单一 foreach_mul_list launch

aclnnForeachMulList(..., stream)
  -> node-1 foreach_mul_list<schMode><<<needCoreNum, stream>>>
     -> Process<T,schMode>
        -> 对每个 core×Tensor 交集调用 VF_CALL<leaf>(Dim3(512), cnt, x1P, x2P, yP)
```

没有节点间 workspace/GM 中间边。Tiling 任一步失败就不生成 node；第二段 launch 失败向 ACLNN 返回。每个输出元素仅由一个 core 的一个 thread 写，不需要 atomic；源码无显式跨 block 同步。

## Route 与参数设计

### 分核与资源计算

设平台 AIV core 数为 `Cplat`，总元素数 `N`：

```text
if N == 0:
    P=0; C=1; last=0
else:
    P0 = min(65536, ceil(N/Cplat))
    P  = ceilAlign(P0, 32)       # 元素，warp-size 对齐
    C  = ceil(N/P)
    last = N - P*(C-1)
core k covers [k*P, min((k+1)*P,N))
```

每个交集 VF 固定 `threadsPerBlock=512`；thread `t` 处理 `t, t+512, ... <cnt`。`dynUBufSize` 没有传给 `VF_CALL`，显式 dynamic/shared UB 为 N/A。Host 设置的 local memory 为 `L=U-131072` B，但 body 不分配该空间。workspace 为 0 B。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP32 SIMT | `schMode=0` | `dataType==DT_FLOAT` | 检查下一 dtype |
| 2 | R-002 FP16→FP32 SIMT | `schMode=1` | `dataType==DT_FLOAT16` | 检查下一 dtype |
| 3 | R-003 BF16→FP32 SIMT | `schMode=2` | `dataType==DT_BF16` | 检查下一 dtype |
| 4 | R-004 INT32 SIMT | `schMode=3` | `dataType==DT_INT32` | 其它 dtype Tiling 失败 |

```text
FLOAT   -> GET_TPL_TILING_KEY(mode 0) -> Process<float,0>    -> SimtFp32
FLOAT16 -> GET_TPL_TILING_KEY(mode 1) -> Process<half,1>     -> SimtFp16
BF16    -> GET_TPL_TILING_KEY(mode 2) -> Process<bf16,2>     -> SimtBf16
INT32   -> GET_TPL_TILING_KEY(mode 3) -> Process<int32_t,3>  -> SimtInt32
else -> GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 | mode 0 | FP32、合法 TensorList、950 | 单 SIMT node | `Process<float,0>`→`ForeachMulListSimtFp32` | 直接 GM；线程私有 float；无 barrier/atomic | thread-stride `idx<count` |
| R-002 FP16 | mode 1 | FP16、合法 TensorList、950 | 单 SIMT node | `Process<half,1>`→`ForeachMulListSimtFp16` | 直接 GM；两个 FP32 thread-local 输入与 FP32 product | 同上；转换后写 half |
| R-003 BF16 | mode 2 | BF16、合法 TensorList、950 | 单 SIMT node | `Process<bfloat16_t,2>`→`ForeachMulListSimtBf16` | 直接 GM；两个 FP32 thread-local 输入与 FP32 product | 同上；转换后写 BF16 |
| R-004 INT32 | mode 3 | INT32、合法 TensorList、950 | 单 SIMT node | `Process<int32_t,3>`→`ForeachMulListSimtInt32` | 直接 GM；thread-local int；无 barrier/atomic | thread-stride guard |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| Tensor 数上限 | `tensorNum<=256` | POD 的 `tensorSizes[256]` | 全部 | `GRAPH_FAILED` | `ForeachMulListTilingFunc` |
| 非零平台资源 | `coreNum!=0 && ubSize!=0` | blockDim 和 local memory producer 需要有效平台信息 | 全部 | `GRAPH_FAILED` | `GetPlatformInfo` |
| 每核上限与 warp 对齐 | `P=min(65536,ceil(N/Cplat)); P=ceilAlign(P,32)` | 控制每 block 覆盖域并与 32-thread warp 对齐 | 全部 | 无另一 Route | `ForeachMulListTilingFunc` |
| DCache 扣减 | `localMemorySize=ubSize-128KiB` | launcher 为 DCache 预留常量空间 | 全部 | 源码只检查 UB 非 0，未显式检查 `ubSize>=128KiB` | 同上 |
| 空域 | `N==0`→`blockDim=1`、所有 count=0 | 保持合法 launch，Device 立即返回 | 全部 | 空操作成功路径 | Tiling + `Process` |
| 线程数 | `THREAD_NUM=512`、`LAUNCH_BOUND(512)` | 每个 Tensor 交集以固定 512 线程步进 | 全部 | 无动态 thread fallback | `foreach_mul_list_simt.h` |

### POD/ABI

`ForeachMulListTilingData` 是显式 standard-layout C++ struct；按固定仓字段顺序与自然 8 B 对齐，其 `sizeof=2088`、`alignof=8`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachMulListTilingData` | `needCoreNum` | `int32_t`/32 | offset 0；其后 padding 4 B | core/block | 空域写 1；非空 `ceil(N/P)` | `Process` 判断最后 core | 正数；blockDim 同值 |
| 同上 | `totalElements` | `int64_t`/64 | offset 8 | 元素 | `ΣtensorSizes[i]` | 最后 core 的 `coreEnd` | shape size 求和溢出未额外检查 |
| 同上 | `perCoreElements` | `int64_t`/64 | offset 16 | 元素/core | 公式 `P` | `coreStart/coreEnd` | 非空时 32 对齐且上限对齐后最多 65536 |
| 同上 | `lastCoreElements` | `int64_t`/64 | offset 24 | 元素 | `N-P*(C-1)` | 当前 body 不读取 | inactive consumer；仍序列化 |
| 同上 | `tensorNum` | `int32_t`/32 | offset 32；其后 padding 4 B | Tensor 个数 | `static_cast<int32_t>(instanceNum)` | Tensor 遍历上界 | 0～256 |
| 同上 | `tensorSizes` | `int64_t[256]` | offset 40；extent 2048 B；struct end 2088 | 元素/每 Tensor | `storageShape.GetShapeSize()` | 累积 `tStart/tEnd`、求交 | 未用槽由 `memset_s` 清零 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP32 | mode 0 | `ForeachMulListTilingFunc` FP32 branch | 显式 POD + blockDim/localMemorySize | `foreach_mul_list<0>` | `Process<float,0>`、`ForeachMulListSimtFp32` | `op_kernel/CMakeLists.txt` ascend950 arch35 source |
| R-002 FP16 | mode 1 | 同上 FP16 branch | 同一 POD | `foreach_mul_list<1>` | `Process<half,1>`、`ForeachMulListSimtFp16` | 同上 |
| R-003 BF16 | mode 2 | 同上 BF16 branch | 同一 POD | `foreach_mul_list<2>` | `Process<bfloat16_t,2>`、`ForeachMulListSimtBf16` | 同上 |
| R-004 INT32 | mode 3 | 同上 INT32 branch | 同一 POD | `foreach_mul_list<3>` | `Process<int32_t,3>`、`ForeachMulListSimtInt32` | 同上 |

## Kernel 实现

### 公共执行流程

四条 Route 共享 `Process<T,schMode>`：block `b` 得到 `[bP,(b+1)P)`，最后 block 以 `N` 截断；遍历 `tensorSizes` 形成 Tensor 逻辑区间，计算交集 `[oS,oE)`，将其变换为 Tensor 内 `localOff` 和 `cnt`，从三个 `ListTensorDesc` 取相同位置指针。`schMode` 编译期选择一个 leaf VF；每个 leaf 以 512 个 thread 做 grid-stride 循环。所有访问直达 GM，不经过显式 UB，且元素写域互斥。

### Route R-001：FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | docs + `foreach_mul_list_def.cpp` | dtype 表 | FP32 + 950 | 支持 |
| Tiling | `foreach_mul_list_tiling_arch35.cpp` | `ForeachMulListTilingFunc` | `dataType==DT_FLOAT` | mode 0 |
| Router | `arch35/foreach_mul_list.cpp` | `foreach_mul_list<schMode>` | `schMode==0` | `Process<float,0>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_list<0> -> Process<float,0>
  -> ListTensorDesc::GetDataPtr<float>(t)+localOff
  -> VF_CALL<ForeachMulListSimtFp32>(Dim3(512),cnt,x1P,x2P,yP)
     -> thread-stride GM load -> float Mul -> GM store
```

##### 源码等价伪码

```text
# 伪码
[coreStart,coreEnd)=blockIdx*P .. min((blockIdx+1)*P,N); empty -> return
for each Tensor t with logical [tStart,tEnd):
    [oS,oE)=intersection(core range,tensor range)
    if oS<oE:
        ptrs = GetDataPtr<float>(t)+(oS-tStart); cnt=oE-oS
        for idx=threadIdx; idx<cnt; idx+=512:
            y[idx]=x1[idx]*x2[idx]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | `GetDataPtr<float>(t)+oS-tStart+idx` | 1 element/thread iteration | FP32/ND contiguous | framework/leaf | leaf/user | 调用者；launch 内 | 无 UB；thread-stride唯一覆盖 |
| thread-local | VF registers，固定源码未给物理寄存器数 | x1、x2、product 标量 | FP32 | GM load | GM store | 单 thread/迭代 | 编译器分配固定仓不可见 |
| local/dynamic UB | N/A：body 无分配 | launcher 设置 `U-128KiB`，无显式 consumer | N/A | Host | 无 | runtime | 不臆造布局 |
| workspace | N/A | 0 B | N/A | Host | 无 | runtime | 未读 |

##### 分层内存布局

N/A：该 SIMT 路线直接 GM load/compute/store；未使用 AIV `TQue/TBuf`、shared/dynamic UB 或中间 workspace。

##### 字节账与复用顺序

```text
UB_total(explicit) = 0 B
dynamic_ub = N/A（VF_CALL 未传 dynUBufSize）
launcher_local_memory = U - 131072 B（body 无显式消费）
workspace_total(operator) = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM y 元素 | 唯一 `(block,t,thread,iteration)` | 用户 | 后续外部算子 | stream 顺序；本 Kernel 无别名复用 | 分核区间与线程 idx 不重叠 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| leaf | 无显式 barrier/atomic | GM x1/x2 | 同 thread Mul | thread-local | FP32 原 dtype；`idx<cnt` 保护尾部 |
| store | 唯一 writer | Mul | stream 后续工作 | GM y element | 最后 thread iteration 自然截断 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_mul_list_tiling_arch35.cpp` | `ForeachMulListTilingFunc` | mode 0、blockDim/POD |
| Kernel entry/router | `op_kernel/arch35/foreach_mul_list.cpp` | `foreach_mul_list<0>` | mode 到 `Process<float,0>` |
| Device body/helper | `op_kernel/arch35/foreach_mul_list_simt.h` | `Process<float,0>`/`ForeachMulListSimtFp32` | Tensor 交集与直接 GM FP32 Mul |

### Route R-002：FP16 升 FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | docs + OpDef | | FP16 | 支持 |
| Tiling | arch35 Tiling | `dataType==DT_FLOAT16` | mode 1 | entry specialization |
| Router | arch35 entry | `schMode==1` | `Process<half,1>` | FP16 leaf |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_list<1> -> Process<half,1>
  -> VF_CALL<ForeachMulListSimtFp16>(512,cnt,...)
     -> __half2float(x1/x2) -> FP32 Mul -> __float2half -> GM y
```

##### 源码等价伪码

```text
# core/Tensor overlap same as common flow
for idx=threadIdx; idx<cnt; idx+=512:
    val1=float(x1[idx]); val2=float(x2[idx])
    y[idx]=half(val1*val2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | `GetDataPtr<half>(t)+localOff+idx` | 1 element/iteration | FP16/ND | framework/leaf | leaf/user | launch | direct GM |
| thread-local | `val1,val2` + product | 3 FP32 logical scalars；物理寄存器固定仓不可见 | FP32 | conversion/Mul | half conversion | thread iteration | 下一 iteration 覆盖 |
| UB/workspace | N/A | explicit 0 B / workspace 0 B | N/A | N/A | N/A | runtime | direct GM |

##### 分层内存布局

N/A：SIMT 直接 GM，只有 thread-local FP32 标量，无显式 UB/shared buffer。

##### 字节账与复用顺序

```text
UB_total(explicit)=0 B; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local `val1/val2` | half conversion | FP32 Mul | 下一 idx | 单线程程序顺序 | 不跨线程共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| convert/Mul/store | 单线程程序顺序 | half GM load | FP32 Mul→half store | 当前 idx | `__half2float`→FP32 Mul→`__float2half`；具体 rounding 契约来自外部 intrinsic，固定仓未展开；`idx<cnt` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | `ForeachMulListTilingFunc` FP16 branch | mode 1 |
| Kernel entry/router | arch35 entry | `foreach_mul_list<1>` | `Process<half,1>` |
| Device body/helper | arch35 SIMT header | `ForeachMulListSimtFp16` | FP16→FP32→FP16 与 thread tail |

### Route R-003：BF16 升 FP32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | docs + OpDef | | BF16 + 950 | 支持 |
| Tiling | arch35 Tiling | `dataType==DT_BF16` | mode 2 | entry specialization |
| Router | arch35 entry | `schMode==2` | `Process<bfloat16_t,2>` | BF16 leaf |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_list<2> -> Process<bfloat16_t,2>
  -> VF_CALL<ForeachMulListSimtBf16>(512,cnt,...)
     -> __bfloat162float(x1/x2) -> FP32 Mul -> __float2bfloat16 -> GM y
```

##### 源码等价伪码

```text
for each core/Tensor overlap:
  for idx=threadIdx; idx<cnt; idx+=512:
    val1=float(x1[idx]); val2=float(x2[idx]); y[idx]=bf16(val1*val2)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | `GetDataPtr<bfloat16_t>(t)+localOff+idx` | 1 element/iteration | BF16/ND | framework/leaf | leaf/user | launch | direct GM |
| thread-local | `val1,val2` 与 product | 3 FP32 logical scalars | FP32 | conversion/Mul | BF16 conversion | thread iteration | 下一 idx 覆盖 |
| UB/workspace | N/A | explicit 0 B / workspace 0 B | N/A | N/A | N/A | runtime | 无中间物化 |

##### 分层内存布局

N/A：SIMT 直接 GM，仅用 thread-local FP32 标量。

##### 字节账与复用顺序

```text
UB_total(explicit)=0 B; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-local scalars | BF16 conversion | FP32 Mul/BF16 conversion | 下一 idx | 单线程程序顺序 | 不共享 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| convert/Mul/store | 单线程程序顺序 | BF16 load | FP32 Mul→BF16 store | 当前 idx | `__bfloat162float`→FP32 Mul→`__float2bfloat16`；rounding 细节固定仓未展开；`idx<cnt` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | BF16 branch | mode 2 |
| Kernel entry/router | arch35 entry | `foreach_mul_list<2>` | BF16 Process |
| Device body/helper | arch35 SIMT header | `ForeachMulListSimtBf16` | BF16 升精度计算与 tail |

### Route R-004：INT32 SIMT

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | docs + OpDef | | INT32 | 支持 |
| Tiling | arch35 Tiling | `dataType==DT_INT32` | mode 3 | entry specialization |
| Router | arch35 entry | `schMode==3` | `Process<int32_t,3>` | INT32 leaf |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_list<3> -> Process<int32_t,3>
  -> VF_CALL<ForeachMulListSimtInt32>(512,cnt,...)
     -> int32 GM load -> integer Mul -> GM store
```

##### 源码等价伪码

```text
for each core/Tensor overlap:
  for idx=threadIdx; idx<cnt; idx+=512:
    y[idx]=x1[idx]*x2[idx]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | `GetDataPtr<int32_t>(t)+localOff+idx` | 1 element/iteration | INT32/ND | framework/leaf | leaf/user | launch | direct GM；唯一 writer |
| thread-local | load/product | INT32 logical scalars | INT32 | GM load/Mul | store | thread iteration | 物理寄存器固定仓不可见 |
| UB/workspace | N/A | 0 B explicit / 0 B workspace | N/A | N/A | N/A | runtime | 无 buffer |

##### 分层内存布局

N/A：SIMT 直接 GM，无显式 AIV UB、dynamic UB 或 workspace。

##### 字节账与复用顺序

```text
UB_total(explicit)=0 B; dynamic_ub=N/A; workspace_total=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM y element | 唯一 thread | 用户 | 外部后续算子 | stream 顺序 | 分核与 idx guard 唯一覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| integer Mul/store | 无显式 barrier/atomic | GM load | 同 thread store | 当前 idx | INT32 溢出契约固定仓未声明；`idx<cnt` tail |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | arch35 Tiling | INT32 branch | mode 3 |
| Kernel entry/router | arch35 entry | `foreach_mul_list<3>` | INT32 Process |
| Device body/helper | arch35 SIMT header | `ForeachMulListSimtInt32` | 直接整数 Mul 与 tail |

## 算子特定优化

### Trick T-001：扁平分核后在 Device 求 Tensor 交集

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：当前 core 区间与 Tensor 逻辑区间满足 `oS<oE`。
- Golden 来源：[`foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_simt.h:104-118`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_simt.h#L104-L118)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    // Iterate tensors, find overlap with core range, compute
    int64_t tStart = 0;
    for (int32_t t = 0; t < tilingData->tensorNum; t++) {
        int64_t tSize = tilingData->tensorSizes[t];
        int64_t tEnd = tStart + tSize;

        int64_t oS = (coreStart > tStart) ? coreStart : tStart;
        int64_t oE = (coreEnd < tEnd) ? coreEnd : tEnd;

        if (oS < oE) {
            int64_t localOff = oS - tStart;
            int64_t cnt = oE - oS;
            __gm__ T* x1P = x1List.GetDataPtr<T>(t) + localOff;
            __gm__ T* x2P = x2List.GetDataPtr<T>(t) + localOff;
            __gm__ T* yP = yList.GetDataPtr<T>(t) + localOff;
```

#### 机制与设计原因

Host 只需传 256 个 Tensor 大小与统一的扁平 core 切片，无需构造每核的 Tensor start/end/offset 数组。Device 通过区间交集恢复 Tensor 内 offset，空 Tensor 或无交集自然跳过。该机制减少 POD 的 per-core mapping 字段，并保持跨 Tensor 的总体负载分配。

#### 迁移边界

- 适用：TensorList 元素间无依赖、可在逻辑拼接域任意切片的点运算。
- 前提：每个 Tensor 的 GM 指针可通过 `ListTensorDesc` 独立取得，输入输出列表位置一一对应。
- 失效：操作需要整 Tensor 归约、跨 Tensor 状态或不能在 Tensor 内任意切分时不可照搬。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：512-thread 直接 GM thread-stride 计算，避免 UB 中间物化

#### 触发与源码依据

- 适用 Route：R-001；其余 Route 复用相同线程步进，仅多 dtype 转换。
- 触发条件：某 core 与 Tensor 的交集 `cnt>0`。
- Golden 来源：[`foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_simt.h:36-43`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_simt.h#L36-L43)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ForeachMulListSimtFp32(
    int64_t count, __gm__ float* x1, __gm__ float* x2, __gm__ float* y)
{
    for (int64_t idx = static_cast<int64_t>(Simt::GetThreadIdx());
         idx < count;
         idx += static_cast<int64_t>(Simt::GetThreadNum())) {
        y[idx] = x1[idx] * x2[idx];
    }
}
```

#### 机制与设计原因

每个 thread 直接加载两路 GM 标量、计算并写回，没有显式 GM→UB→GM tile staging，也没有中间 workspace。`idx+=GetThreadNum()` 让任意 `cnt` 都由固定 512 线程覆盖，`idx<count` 同时完成尾部保护；互斥 idx 消除 atomic 和 block barrier。

#### 迁移边界

- 适用：无邻域依赖、单元素计算量小、支持 SIMT 直接 GM 标量访问的点运算。
- 前提：每个输出元素只有一个 writer，TensorList 数据已连续化，VF launch 支持 512 threads。
- 失效：需要数据复用、邻域访问、归约或共享中间结果时应设计 UB/shared staging 与同步。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | R-004 INT32 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | mode 0 | mode 1 | mode 2 | mode 3 | Host dtype if/else 顺序 |
| 模板与 dtype | `float,0` | `half,1` | `bfloat16_t,2` | `int32_t,3` | 编译期消除 dtype 分支 |
| 循环与分核 | 共同扁平分核+512 thread stride | 同 | 同 | 同 | shape 不参与 Route |
| Queue/TBuf/寄存器/dynamic UB/通信 window | thread-local float | thread-local FP32 转换值 | thread-local FP32 转换值 | thread-local int | 无显式 UB/shared/window |
| 内存总占用与复用 | explicit UB 0 | explicit UB 0 | explicit UB 0 | explicit UB 0 | 直接 GM |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT scalar GM Mul | SIMT convert+Mul | SIMT convert+Mul | SIMT integer Mul | 无 NDDMA/通信 |
| 同步、精度与 tail | 无显式同步；FP32 | FP32 Mul→FP16 | FP32 Mul→BF16 | INT32 | 全部 `idx<cnt` |
| fallback/失败边界 | 无 | 无 | 无 | 其它 dtype 失败 | 无 runtime fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持，见无后缀卡 | 公开支持，见无后缀卡 | 公开支持 |
| Host/Launcher | arch22 公共 Tiling | arch22 公共 Tiling | arch35 专用 Tiling，Tensor 上限 256，workspace 0 |
| Kernel/资源 | AIV Vector Queue pipeline | 同 | SIMT 512 threads，直接 GM，无显式 UB queue |

## 限制与失败边界

- `tensorNum` 最大 256；超过时 arch35 Tiling 显式失败。单 Tensor 维数、列表/shape/dtype 一致性由 ACLNN/OpDef/InferShape 契约负责，arch35 Tiling 自身只枚举 x1。
- `totalElements` 与逐 Tensor 累积均为 `int64_t`，固定源码没有显式求和溢出检查。
- `SetLocalMemorySize(ubSize-128KiB)` 仅检查 `ubSize!=0`，没有显式 `ubSize>=128KiB` guard；Ascend 950 实际平台值由 runtime 提供，不能把其它平台资源值代入。
- FP16/BF16 conversion intrinsic 的精确 rounding 规范位于仓外 SIMT API；固定仓只能证明转换顺序，不能补写 rounding mode。
- INT32 乘法溢出语义未由公开文档定义。
- 自动连续化 bridge 的临时 workspace、event 与 cleanup 细节固定仓不可见；算子 Tiling 自有 workspace 是 0 B。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_mul_list/docs/aclnnForeachMulList.md` | 两段式 API、产品/参数表 | 公开性、950、接口约束 |
| 入口实现 | `foreach/foreach_mul_list/op_host/foreach_mul_list_def.cpp` | `ForeachMulList` OpDef | dtype、TensorList、AutoContiguous、ascend950 |
| 入口实现 | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon` | 输出实例/shape/dtype |
| Host/Tiling/Launcher | `foreach/foreach_mul_list/op_host/arch35/foreach_mul_list_tiling_arch35.cpp` | `ForeachMulListTilingFunc` | 平台、分核、workspace/local memory、key |
| Host/Tiling/Launcher | `foreach/foreach_mul_list/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`/`SUPPORT_TILING_DIR` | ascend950 选择 arch35 Tiling |
| Host/Tiling/Launcher | `foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_tiling_data.h` | `ForeachMulListTilingData` | POD ABI |
| Host/Tiling/Launcher | `foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 四个 schMode |
| Kernel entry/body | `foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list.cpp` | `foreach_mul_list<schMode>` | entry/router |
| Kernel entry/body | `foreach/foreach_mul_list/op_kernel/arch35/foreach_mul_list_simt.h` | `Process` 与四个 leaf | Tensor 交集、512-thread SIMT、精度/tail |
| 构建接线 | `foreach/foreach_mul_list/op_kernel/CMakeLists.txt` | ascend950 `KERNEL_SRC` | 950 专用 Kernel |
| 构建接线 | `foreach/foreach_mul_list/op_host/config/ascend950/foreach_mul_list_binary.json` | `ForeachMulList` op_list | 950 四 dtype 二进制实例 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段显式 struct；`sizeof=2088` | 字段 offset、`tensorSizes[256]`、元素单位 | `ForeachMulListTilingData` |
| 调用顺序 | Host 产 key/POD→entry specialization→Process→逐交集 VF_CALL | schMode 与 T/leaf 对应不可错配 | arch35 entry + `Process` |
| workspace/内存 | 算子 workspace 0；直接 GM；无显式 UB | 不得把 localMemorySize 虚构成 body buffer | Tiling `SetLocalMemorySize`、SIMT body |
| 同步 | 每元素唯一 writer，无显式 barrier/atomic | 分核区间、Tensor 交集和 thread idx 必须互斥 | `Process`、leaf loop |
| 精度与尾块 | FP16/BF16 升 FP32；FP32/INT32 直算；`idx<cnt` | conversion 顺序与 dtype leaf | 四个 SIMT leaf |
| 失败边界 | 256 Tensor、平台资源、四 dtype | 其它 dtype 不得静默映射；空域仍一 block | `ForeachMulListTilingFunc` |