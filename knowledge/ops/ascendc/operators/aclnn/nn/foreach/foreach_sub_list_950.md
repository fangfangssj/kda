---
type: Operator
title: "ForeachSubList 950 ACLNN 算子知识"
description: "说明 aclnnForeachSubListV2 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 SIMT 实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/docs/aclnnForeachSubListV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_host/op_api/aclnn_foreach_sub_list_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_host/op_api/foreach_sub_list_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_host/foreach_sub_list_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_host/arch35/foreach_sub_list_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_host/config/ascend950/foreach_sub_list_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:22:53Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachSubList 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachSubListV2GetWorkspaceSize` / `aclnnForeachSubListV2`，公开 ACLNN V2；同族旧入口 `aclnnForeachSubList` 的公开表明确不支持 950。 |
| 计算语义 | 对列表位置 `i` 和该 Tensor 的元素 `j` 计算 `y_i[j] = x1_i[j] - alpha * x2_i[j]`。 |
| 输入输出 | `x1/x2/out` 为等长 TensorList；每个位置 shape 相同；Tensor dtype 为 FP32/FP16/INT32/BF16，公开格式 ND、维数 0–8；`alpha` 是 Scalar。 |
| 执行类型 | 单 Kernel：`foreach_sub_list<schMode>` 外层 AIV entry 内以 `VF_CALL` 启动 SIMT body；无 Cube/AIC、无 MIX_CV。 |
| Kernel 路线 | R-001 FP32/INT32 直接算术；R-002 FP16 升 FP32；R-003 BF16 升 FP32但 alpha ABI 冲突。 |
| 核心资源 | TensorList GM 指针表、各 Tensor GM 数据、2064 B tiling POD；每 block 1024 个 SIMT thread；无算子 workspace、无 AIV `TQue/TBuf`。 |
| 来源闭包 | `source_closed`：公开 V2 ACLNN → 参数校验/Scalar 转 Tensor → `ForeachSubList` launcher → arch35 tiling/template key → arch35 entry → SIMT grid-stride body → ascend950 binary/CMake。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设列表长度为 `n`，`E_i = numel(x1_i)`。有效域为 `0 <= i < n`、`0 <= j < E_i`：

```text
y_i[j] = x1_i[j] - alpha * x2_i[j]
```

每个 Tensor 按 storage shape 展平成一维元素域执行；Host 要求 `x1_i` 与 `x2_i` shape 相等，ACLNN V2 还要求 `out_i` shape 相等。输出 Tensor 之间没有归约或交叉依赖。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | TensorList；每 Tensor 0–8 维；公开格式 ND | FP32/FP16/INT32/BF16 | 可含空 Tensor；列表内 dtype 一致；V2 文档不承诺非连续 Tensor。 |
| `x2` | 与 `x1` 等长且逐位置 shape 相同 | 与 `x1` 相同 | 逐元素减数。 |
| `alpha` | `aclScalar` | `x1` 为 FP32/BF16 时接受 FP32/DOUBLE；FP16 时接受 FP16/DOUBLE；INT32 时接受 INT32/INT64 | 第一段接口将其转为设备 Tensor：BF16 输入路线固定转 FP32，其余转为 `x1` dtype。 |
| `out` | TensorList；V2 API 实现要求逐位置 shape 与 `x1` 相同 | 与 `x1` 相同 | 输出列表；公开文档不承诺非连续输出。 |
| `workspaceSize/executor` | 标量 Host 输出 | `uint64_t*` / `aclOpExecutor**` | 第一段接口返回；空列表时 workspace 为 0 且 executor 中不加入算子节点。 |

### 入口与版本

固定 Commit 同时公开 `aclnnForeachSubList` 与 `aclnnForeachSubListV2`。V2 把 `alpha` 从 `aclTensor` 改为 `aclScalar`；V2 产品表明确支持 Ascend 950PR/950DT，而旧接口产品表明确 950 不支持，因此 950 取最新且实际支持的 `aclnnForeachSubListV2`。`GetDtypeSupportList` 接受 `IsRegbase(curArch)`，OpDef 注册 `ascend950`，binary config 含四种 Tensor dtype，Host/CMake 将 ascend950 绑定 arch35 实现，公开性与实现接线无冲突。

## Host/Launcher 执行流程

### 参数校验与前处理

`ExecForeachSubListV2GetWorkspaceSize` 的固定顺序如下：

1. 创建 `aclOpExecutor`；创建失败返回 `ACLNN_ERR_INNER_CREATE_EXECUTOR`。
2. `CheckNotNull` 检查 `x1/x2/alpha/out`。
3. `CheckDtypeValid` 检查 950 Regbase 平台、四种 Tensor dtype、列表内及三列表间 dtype 一致性，以及 Scalar dtype 对应关系。
4. `CheckShape` 检查 `x1.Size == x2.Size`、逐位置 `x1/x2` shape 相等、`x1` 维数不超过 8、逐位置 `x1/out` shape 相等。
5. `CheckFormat` 拒绝三个列表中任一 Tensor 的私有 storage format。
6. 若 `x1` 或 `x2` 为空列表，返回 workspace 0，不加入 Device node。
7. 对 `x1/x2` 每个 Tensor 调用框架 `l0op::Contiguous` 并重建 TensorList。该实现属于框架边界；公开 V2 契约本身不承诺非连续输入。
8. `ConvertToTensor` 将 Scalar 物化：BF16 Tensor 路线用 FP32 Tensor，其余用 Tensor dtype。
9. `l0op::ForeachSubListV2` 以 `ADD_TO_LAUNCHER_LIST_AICORE(ForeachSubList, ...)` 加入唯一算子节点；任一步失败立即返回，不执行后续节点。

### 参数、Workspace 与生命周期

API 返回的 `workspaceSize = executor->GetWorkspaceSize()`，其 owner 是调用方，需存活至第二段接口在指定 stream 上完成。算子自身 arch35 tiling 将 `workspace[0]` 设为 0；若框架 `Contiguous` 产生额外节点，其临时空间由 executor 汇总，固定算子源码未公开该框架空间公式。

Tiling 从平台取得 `coreNum = GetCoreNumAiv()` 与 UB 大小。`totalElements = sum_i storage_numel(x1_i)`；非空元素域时 `needCoreNum = min(coreNum, ceil(totalElements / 1024))`，全空元素域时取 1。`SetBlockDim(needCoreNum)` 决定 SIMT block 数；`SetLocalMemorySize(uint32_t(ubSize - 128 KiB))` 写 launch local-memory 上限，但当前 body 没有 dynamic/shared UB 分配。

### 启动图与失败传播

```text
aclnnForeachSubListV2GetWorkspaceSize
  -> [框架边界：逐输入 Contiguous；支持域内通常为连续视图]
  -> ForeachSubList launcher node
       Host: ForeachSubListTilingFunc
       entry: foreach_sub_list<schMode>
       grid/blockDim: needCoreNum
       threadsPerBlock: 1024（VF_CALL）
       dynamic UB: 0（body 未申请）
       workspace slice: 0 B
       POD: ForeachSubListTilingData
  -> y TensorList
```

唯一算子 Device 节点直接从 `x1/x2/alpha` GM 读并向 `y` GM 写，无跨 Kernel workspace 边。executor 保证 launcher list 在第二段传入 stream 上按序执行；任一参数检查、Contiguous、launcher 添加或 tiling 失败都会阻断该节点。Device body 无返回码与 cleanup 分支。

## Route 与参数设计

### 分核与资源计算

```text
tensorCount = x1 的实例数，要求 <= 256
tensorElements[i] = storage_shape_size(x1[i])               # 元素
totalElements = Σ tensorElements[i]                          # 元素，int64
needCoreNum = totalElements == 0
            ? 1
            : min(aivCoreNum, ceil(totalElements / 1024))
blockDim = needCoreNum
threadsPerBlock = 1024
globalStride = blockDim * 1024                               # 元素线程步长
workspace = 0 B
localMemorySize = uint32_t(UB_bytes - 128 KiB)
```

每个 block 的 thread 在每个 Tensor 内从 `tid = blockIdx * 1024 + threadIdx` 开始，以 `globalStride` 递增。所有 block 重复遍历 TensorList 元数据，但各线程写不同元素。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 直接算术 | `schMode=FLOAT` 或 `INT32` | `x1[0].dtype == DT_FLOAT`；否则 `== DT_INT32`；Host 按 FP32→FP16→INT32→BF16 顺序判断 | 下一 dtype 分支；非四种 dtype 返回 `GRAPH_FAILED`。 |
| 2 | R-002 FP16 升精度 | `schMode=FLOAT16` | `x1[0].dtype == DT_FLOAT16` | 下一 dtype 分支。 |
| 3 | R-003 BF16 升精度/ABI 缺口 | `schMode=BF16` | 前三分支均不命中且 `x1[0].dtype == DT_BF16` | 非四种 dtype 返回 `GRAPH_FAILED`。 |

```text
if dtype == FP32: schMode = FLOAT      -> R-001
else if dtype == FP16: schMode = FLOAT16 -> R-002
else if dtype == INT32: schMode = INT32  -> R-001
else if dtype == BF16: schMode = BF16    -> R-003
else: GRAPH_FAILED
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 直接算术 | `schMode=0(FLOAT)`、`2(INT32)`；Host 用 `GET_TPL_TILING_KEY` 编码 | 950；dtype FP32 或 INT32；TensorList `<=256`；API 校验通过 | 单节点 `foreach_sub_list<schMode>` → `Process<T>` → `VF_CALL<OpForeachSubListSimt<T>>` | `T=float` 或 `int32_t`；`y=x1-alpha*x2` | GM 直读直写；无 UB、queue、barrier、atomic、workspace | `idx<count` 自然保护尾部；未知 dtype Host 失败。 |
| R-002 FP16 升精度 | `schMode=1(FLOAT16)` | 950；dtype FP16；其余同上 | 同一单节点拓扑 | `T=half`；元素和 alpha 转 FP32，FP32 乘减，转回 half | GM 直读直写；线程私有标量；无共享同步 | `idx<count`。 |
| R-003 BF16 升精度/ABI 缺口 | `schMode=3(BF16)` | 950；dtype BF16；其余同上 | 同一单节点拓扑 | `T=bfloat16_t`；body 计划升 FP32，但先把 FP32 alpha Tensor 按 BF16 读取 | GM 直读直写；线程私有标量；无共享同步 | `idx<count`；无 ABI 修复/fallback。 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| TensorList 上限 | `tensorNum <= 256` | POD 仅有 `tensorElements[256]` | 全部 | `GRAPH_FAILED` | `foreach_sub_list_tiling_arch35.cpp::ForeachSubListTilingFunc` |
| 每核最少目标元素 | `needCoreNum=min(aivCoreNum, ceil(totalElements/1024))` | 与每 block 1024 thread 配合，避免 block 数超过按 1024 元素估算的需求 | 全部 | 无备用 Route | 同上 |
| 空元素域 | `totalElements==0 -> blockDim=1` | 保持合法 launch；每个 Tensor 的 `count==0` 被跳过 | 全部 | 无输出写入 | 同上、`OpForeachSubListSimt` |
| local-memory 扣减 | `uint32_t(ubSize-128 KiB)`，源码只检查 `ubSize!=0` | 为 DCache 预留 128 KiB；当前 SIMT body 不消费剩余 UB | 全部 | 固定源码无 `ubSize>=128 KiB` guard | `GetPlatformInfo`、`ForeachSubListTilingFunc` |
| dtype 编译选择 | 仅四个 `schMode` | 入口是编译期 `if constexpr`，无 runtime unknown-key body | 全部 | Host 非支持 dtype 失败 | `foreach_sub_list_tiling_key.h`、arch35 `foreach_sub_list` |

### POD/ABI

`ForeachSubListTilingData` 是标准 C++ struct；按声明顺序与自然对齐，其 `sizeof=2064`、`alignof=8`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachSubListTilingData` | `needCoreNum` | `int32_t/32` | offset 0；size 4 | core | 全空写 1，否则 `int32(min(coreNum,ceil(total/1024)))` | body 不读取；同值另由 `SetBlockDim` 传 launch | `coreNum>0`；转换前值不超过平台核数。 |
| 同上 | `tensorCount` | `int32_t/32` | offset 4；size 4 | Tensor 个数 | `int32_t(tensorNum)` | `Process` 读后传入 SIMT 循环上界 | `0..256`。 |
| 同上 | `totalElements` | `int64_t/64` | offset 8；size 8 | 元素 | `sum_i numel` | Device 不读取 | 求和无显式溢出检查。 |
| 同上 | `tensorElements` | `int64_t[256]` | offset 16；size 2048 | 每 Tensor 元素 | `tensorElements[i]=storageShape.GetShapeSize()`；先整体清零 | `elemCounts[t]` 决定 `idx<count` | 仅 `[0,tensorCount)` 活跃。 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 直接算术 | `schMode FLOAT/INT32` | `op_host/arch35/foreach_sub_list_tiling_arch35.cpp::ForeachSubListTilingFunc` | `ForeachSubListTilingData` | `op_kernel/arch35/foreach_sub_list.cpp::foreach_sub_list<schMode>` | `foreach_sub_list_simt.h::Process<float/int32_t>`、`OpForeachSubListSimt` | `op_kernel/CMakeLists.txt` ascend950；`config/ascend950/...json` |
| R-002 FP16 升精度 | `schMode FLOAT16` | 同上 | 同上 | 同上 | `Process<half>`、`OpForeachSubListSimt` | 同上 |
| R-003 BF16 升精度/ABI 缺口 | `schMode BF16` | 同上 | 同上 | 同上 | `Process<bfloat16_t>`、`OpForeachSubListSimt` | 同上 |

## Kernel 实现

### 公共执行流程

arch35 entry 用模板参数 `schMode` 在编译期选 `T`，从 tiling GM 取得 POD 指针，然后 `Process<T>` 通过 `offsetof(..., tensorElements)` 构造元素数数组指针并以 `VF_CALL(... Dim3(1024))` 启动 SIMT 函数。每个 SIMT thread 对所有 Tensor 顺序循环；TensorList GM 对象的首个 `uint64_t` 是地址表相对偏移，右移 3 转成 `uint64_t` 索引，再取第 `t` 个真实 Tensor GM 地址。每个 Tensor 内使用全局 grid-stride 切分，不需要 block 间同步。

### Route R-001：FP32/INT32 直接算术

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_list_v2.cpp` | `GetDtypeSupportList` / `CheckDtypeValid` | Regbase 且 dtype FP32/INT32 | 加入 `ForeachSubList`。 |
| Tiling | `op_host/arch35/foreach_sub_list_tiling_arch35.cpp` | `ForeachSubListTilingFunc` | `DT_FLOAT` 或第三分支 `DT_INT32` | 生成对应 `schMode` key。 |
| Router | `op_kernel/arch35/foreach_sub_list.cpp` | `foreach_sub_list<schMode>` | `if constexpr schMode==FLOAT/INT32` | `Process<float/int32_t>`。 |

#### Device 数据流

##### 函数级调用链

```text
ForeachSubListTilingFunc -> launch foreach_sub_list<schMode>(x1,x2,alpha,y,workspace,tiling)
  -> NsForeachSubList::Process<float|int32_t>
    -> VF_CALL<OpForeachSubListSimt<T>>(threads=1024, tensorCount, elemCounts, ...)
      -> SimtGetTensorAddr<T>(x1/x2/y, t)
      -> yData[idx] = x1Data[idx] - alphaVal * x2Data[idx]
```

##### 源码等价伪码

```text
# 伪码；blockDim 来自 needCoreNum，每 block 1024 threads
alphaVal = reinterpret_cast<gm T*>(alpha)[0]
for t in [0, tensorCount):
    count = tensorElements[t]
    if count == 0: continue
    x1Data, x2Data, yData = decode_tensor_list_pointer(t)
    tid = blockIdx * 1024 + threadIdx
    stride = blockNum * 1024
    for idx = tid; idx < count; idx += stride:
        yData[idx] = x1Data[idx] - alphaVal * x2Data[idx]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList 地址表 | `base + ((*base)>>3) + t` | `tensorCount` 个 64-bit 指针 | `uint64_t` table | ACLNN/framework | `SimtGetTensorAddr` | 输入/输出 TensorList；Kernel 生命周期 | 偏移按 8 B 索引。 |
| `x1_i/x2_i` GM | `decoded_ptr + idx` | `tensorElements[i]` | `T`，展平 storage layout | 调用方 | SIMT thread | 调用方持有至 stream 完成 | 每线程标量 load。 |
| `y_i` GM | `decoded_ptr + idx` | 同上 | `T` | SIMT thread | 调用方后续节点 | 调用方 | 每元素唯一 writer，无 atomic。 |
| tiling GM | POD base + offset | 2064 B | 标准 struct | Host tiling | entry/Process | runtime | `alignof=8`。 |

##### 分层内存布局

N/A：本 Route 未调用 `InitBuffer`，未使用 AIV UB、dynamic/shared UB、`TQue` 或 `TBuf`；线程只持有标量寄存器和 GM 指针。

##### 字节账与复用顺序

```text
UB_total = 0 B（算子显式分配）
dynamic_ub = 0 B（固定源码无 dynamic/shared UB）
workspace_total = 0 B（ForeachSubList 节点）
tiling_payload = 2064 B（runtime 管理，不计入算子 UB/workspace）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程私有 `alphaVal/x1Val/x2Val` | 当前 thread | 当前表达式 | 下一元素迭代 | SIMT 单线程程序序 | 不跨 thread 共享。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM 读写 | 每元素唯一 thread | GM load | 同 thread 表达式/store | 单个 `y[idx]` | FP32 在 FP32 执行；INT32 直接乘减，固定源码未定义溢出策略。 |
| 跨 thread | N/A | 各 thread | N/A | 无共享数据 | `idx<count` 保证尾元素；空 Tensor 直接跳过。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sub_list_tiling_arch35.cpp` | `ForeachSubListTilingFunc` | 生产 blockDim、POD、workspace 0 和 dtype key。 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_list.cpp` | `foreach_sub_list<schMode>` | FLOAT/INT32 映射到 `Process<float/int32_t>`。 |
| Device body/helper | `op_kernel/arch35/foreach_sub_list_simt.h` | `OpForeachSubListSimt<T>` / `SimtGetTensorAddr` | 地址表解码、grid-stride 与直接算术。 |

### Route R-002：FP16 升 FP32 算术

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_list_v2.cpp` | `CheckDtypeValid` / `ConvertToTensor` | Tensor dtype FP16 | alpha 转 FP16 Tensor。 |
| Tiling | `op_host/arch35/foreach_sub_list_tiling_arch35.cpp` | `ForeachSubListTilingFunc` | `DT_FLOAT16` | 生成 `schMode=FLOAT16` key。 |
| Router | `op_kernel/arch35/foreach_sub_list.cpp` | `foreach_sub_list<schMode>` | `schMode==FLOAT16` | `Process<half>`。 |

#### Device 数据流

##### 函数级调用链

```text
ForeachSubListTilingFunc -> foreach_sub_list<FLOAT16>
  -> Process<half>
    -> VF_CALL<OpForeachSubListSimt<T>>(threads=1024, ...)
      -> GM T load -> cast each operand and alpha to float
      -> float result = x1 - alpha*x2
      -> cast result to T -> GM y
```

##### 源码等价伪码

```text
# 伪码；列表/线程循环与 R-001 相同
alphaVal = reinterpret_cast<gm T*>(alpha)[0]
for each assigned idx:
    x1f = float(x1Data[idx])
    x2f = float(x2Data[idx])
    alphaf = float(alphaVal)
    result = x1f - alphaf * x2f
    yData[idx] = T(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| TensorList/GM | 与 R-001 相同 | 每 Tensor `count` 元素 | FP16 | framework/调用方 | SIMT thread | stream 完成前有效 | 标量 load/store。 |
| 线程寄存器 | thread 私有 | `x1Val,x2Val,alphaF,result` 各一个 FP32 标量 | FP32 | 当前 thread cast/算术 | 当前 thread store | 单次元素迭代 | 下一 `idx` 复用。 |
| tiling GM | 同 R-001 | 2064 B | POD | Host | Process | runtime | 无别名。 |

##### 分层内存布局

N/A：无 AIV UB、dynamic/shared UB 或跨线程 buffer；升精度中间量为线程私有标量。

##### 字节账与复用顺序

```text
UB_total = 0 B
dynamic_ub = 0 B
workspace_total = 0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 线程私有标量 | cast/乘减 | 当前 `y` store | 下一 grid-stride 元素 | 单线程程序序 | 不跨线程发布。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 升精度 | C++ `static_cast<float>` | FP16 GM load | FP32 乘减 | 当前元素 | 乘法与减法在 FP32；结果一次转回 half；源码未指定显式 round mode。 |
| tail | `idx<count` | thread id | GM store | 有效元素域 | 无 padding lane；空 Tensor 跳过。 |
| 同步 | N/A | 当前 thread | N/A | 无共享状态 | 无 barrier/atomic。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/foreach_sub_list_tiling_arch35.cpp` | `ForeachSubListTilingFunc` | FP16 key 与 blockDim。 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_list.cpp` | `foreach_sub_list<FLOAT16>` | 映射 `half`。 |
| Device body/helper | `op_kernel/arch35/foreach_sub_list_simt.h` | `OpForeachSubListSimt<half>` | FP16 输入/alpha 升 FP32、乘减、降精度。 |

### Route R-003：BF16 升 FP32算术与 alpha ABI 缺口

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_host/op_api/aclnn_foreach_sub_list_v2.cpp` | `CheckDtypeValid` / `ConvertToTensor` | Tensor dtype BF16；Scalar FP32/DOUBLE | 明确生成 `DT_FLOAT` alpha Tensor。 |
| Binary/OpDef | `op_host/config/ascend950/foreach_sub_list_binary.json` / `foreach_sub_list_def.cpp` | BF16 binary / `DtypeScalarToTensor2` | x1/x2/y BF16 | alpha 参数 ABI 为 FP32。 |
| Tiling | `op_host/arch35/foreach_sub_list_tiling_arch35.cpp` | `ForeachSubListTilingFunc` | `dataType==DT_BF16` | `schMode=BF16`。 |
| Router/body | `op_kernel/arch35/foreach_sub_list.cpp` / `foreach_sub_list_simt.h` | `Process<bfloat16_t>` / `OpForeachSubListSimt<T>` | `T=bfloat16_t` | `reinterpret_cast<__gm__ T*>(alpha)[0]` 按 BF16 读取 FP32 参数。 |

#### Device 数据流

##### 函数级调用链

```text
API ConvertToTensor(alpha, DT_FLOAT)
  -> ForeachSubList BF16 binary (alpha dtype=float32)
  -> foreach_sub_list<BF16>
    -> Process<bfloat16_t>
      -> OpForeachSubListSimt<bfloat16_t>
        -> reinterpret_cast<gm bfloat16_t*>(alpha)[0]   # ABI 不一致点
        -> float(alphaVal), float(x1), float(x2)
        -> result FP32 -> bfloat16_t y
```

##### 源码等价伪码

```text
# 伪码；固定源码实际行为，不把缺失修正为推测
alphaTensorABI = FP32
alphaVal = load_bfloat16_from_first_2_bytes(alphaTensorABI)
for each Tensor and grid-stride idx:
    result = float(x1BF16[idx]) - float(alphaVal) * float(x2BF16[idx])
    yBF16[idx] = bfloat16_t(result)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x1/x2/y GM | TensorList 解码指针 + `idx` | `count` 元素 | BF16 | caller / SIMT | SIMT / caller | stream 完成前 | 每元素唯一 writer。 |
| alpha GM | base byte 0 | API/OpDef 为 4 B FP32 | FP32 ABI | `ConvertToTensor` | body 按 BF16 读取前 2 B | executor | 类型不一致，无 cast node。 |
| 线程寄存器 | 当前元素 | 4 个 FP32 标量 + BF16 alpha raw | 标量 | 当前 thread | 当前表达式 | 单次迭代 | 无共享。 |

##### 分层内存布局

N/A：与 R-002 相同，无显式 UB/workspace；ABI 缺口发生在 alpha GM 的读取类型，不涉及中间 buffer。

##### 字节账与复用顺序

```text
UB_total = 0 B
dynamic_ub = 0 B
workspace_total = 0 B
alpha_extent = producer 4 B FP32；consumer 读取 2 B BF16（不一致）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程私有 FP32 标量 | 当前 thread | 当前表达式/store | 下一 idx | 单线程程序序 | 不解决 alpha producer/consumer dtype 不一致。 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| alpha 读取 | GM scalar load | FP32 Tensor producer | BF16 reinterpret load | alpha 首 2 B | 无转换语义可证明；这是正式失败边界。 |
| x1/x2 算术 | BF16→FP32 cast | GM load | FP32 乘减 | 当前元素 | 计划在 FP32 计算，后转 BF16；alpha 值先已失去 ABI 保证。 |
| tail/同步 | `idx<count` / N/A | thread id | GM store | 有效元素域 | 尾部安全；无 barrier/atomic。 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| API producer | `op_host/op_api/aclnn_foreach_sub_list_v2.cpp` | `ExecForeachSubListV2GetWorkspaceSize` | BF16 路线把 Scalar 转为 `DT_FLOAT` Tensor。 |
| Host/Tiling producer | `op_host/arch35/foreach_sub_list_tiling_arch35.cpp` | `ForeachSubListTilingFunc` | BF16 key、POD、blockDim。 |
| Kernel entry/router | `op_kernel/arch35/foreach_sub_list.cpp` | `foreach_sub_list<BF16>` | 映射 `Process<bfloat16_t>`。 |
| Device body/helper | `op_kernel/arch35/foreach_sub_list_simt.h` | `OpForeachSubListSimt<bfloat16_t>` | alpha 按 `T*` 读取，形成 ABI 缺口。 |

## 算子特定优化

### Trick T-001：以全局 grid-stride 直接遍历每个 Tensor

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：950 arch35 单 Kernel 路线；`blockDim=needCoreNum`、每 block 1024 thread。
- Golden 来源：[`foreach_sub_list_simt.h:66-82`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_simt.h#L66-L82)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint64_t tid = static_cast<uint64_t>(
    AscendC::Simt::GetBlockIdx() * AscendC::Simt::GetThreadNum() +
    AscendC::Simt::GetThreadIdx());
uint64_t stride = static_cast<uint64_t>(
    AscendC::Simt::GetThreadNum() * AscendC::Simt::GetBlockNum());

for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        float x1Val = static_cast<float>(x1Data[idx]);
        float x2Val = static_cast<float>(x2Data[idx]);
        float alphaF = static_cast<float>(alphaVal);
        float result = x1Val - alphaF * x2Val;
        yData[idx] = static_cast<T>(result);
    } else {
        yData[idx] = x1Data[idx] - alphaVal * x2Data[idx];
    }
}
```

#### 机制与设计原因

线程直接从输入 GM 标量读取并写输出 GM，不物化拼接后的列表数据，也不申请 UB/workspace。`tid` 和 `stride` 由实际 block/thread 数产生，因此任意元素数都由同一循环覆盖，尾部只需 `idx<count`。所有 block 对每个 Tensor 使用相同映射，元素 writer 唯一，不需要 atomic 或 barrier。

#### 迁移边界

- 适用：TensorList 中各 Tensor 可独立逐元素处理、无跨元素状态的 SIMT 算子。
- 前提：地址表 ABI 一致；Host 提供准确的逐 Tensor 元素数和合法 blockDim；每元素表达式无跨线程依赖。
- 失效：需要归约、共享缓存、跨 Tensor 广播或复杂 layout 时必须补充同步/中间存储路线。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：FP16/BF16 元素只在寄存器中升 FP32

#### 触发与源码依据

- 适用 Route：R-002、R-003；R-003 仅描述计划的低精度计算机制，受 alpha ABI 缺口限制。
- 触发条件：`T` 为 `half` 或 `bfloat16_t`。
- Golden 来源：[`foreach_sub_list_simt.h:72-80`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_simt.h#L72-L80)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint64_t idx = tid; idx < static_cast<uint64_t>(count); idx += stride) {
    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
        float x1Val = static_cast<float>(x1Data[idx]);
        float x2Val = static_cast<float>(x2Data[idx]);
        float alphaF = static_cast<float>(alphaVal);
        float result = x1Val - alphaF * x2Val;
        yData[idx] = static_cast<T>(result);
```

#### 机制与设计原因

输入与输出仍保持 16-bit GM 流量，只有当前元素在线程私有 FP32 标量中完成乘减，避免为升精度建立 UB/GM 中间 Tensor。模板 `if constexpr` 消除 FP32/INT32 路线的运行时类型分支；关键条件来自 compile-time `T`，元素索引来自 grid-stride 循环。

#### 迁移边界

- 适用：低精度逐元素算术，且每线程可用少量标量寄存器完成全部中间计算。
- 前提：降精度的语言/硬件转换满足目标误差要求；`alpha` 的实际 Tensor ABI 必须与 Device 读取类型一致。
- 失效：需要大量中间值、向量级复用或显式 rounding mode 时应采用 UB/Vector 路线；当前 BF16 alpha ABI 不一致不得直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓存在真实边界：BF16 V2 API 将 `alpha` 转为 FP32 Tensor，而此 body 以 `bfloat16_t*` 读取，直接迁移前必须修正该 ABI 对齐问题。

## 模板与实现对比

| 对比维度 | R-001 直接算术 | R-002 FP16 升精度 | R-003 BF16/ABI 缺口 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | FP32 或 INT32 | FP16 | BF16 | Host dtype 决策树，均无 fallback。 |
| 模板与 dtype | `Process<float/int32_t>` | `Process<half>` | `Process<bfloat16_t>` | 编译期 `schMode`。 |
| 循环与分核 | 每 Tensor 全局 grid-stride | 相同 | 相同 | `ceil(total/1024)` 限制 block 数。 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 线程标量；dynamic UB 0 | 额外 FP32 线程标量；dynamic UB 0 | 同 R-002 | 逐元素 GM 直通。 |
| 内存总占用与复用 | 算子 UB/workspace 0 | 算子 UB/workspace 0 | 算子 UB/workspace 0 | 无中间物化。 |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT 标量 GM load/store | 同上并 cast | 同上，但 alpha load dtype 错配 | 无 Vector/NDDMA/通信。 |
| 同步、精度与 tail | 无同步；原 dtype；`idx<count` | FP32 乘减后降 FP16；`idx<count` | 计划 FP32 乘减后降 BF16；alpha 无有效 ABI | 低精度路线按参数 ABI 拆分。 |
| fallback/失败边界 | 非支持 dtype Host 失败 | 同上 | 无 ABI 修复/fallback | 固定源码无备用 Kernel。 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | V2 公开支持；见无后缀卡 | V2 公开支持；见无后缀卡 | V2 公开支持；本卡。 |
| Host/Launcher | arch22 公共 Foreach tiling，最多 50 Tensor | 同 A2 | arch35 专用 POD，最多 256 Tensor，workspace 0。 |
| Kernel/资源 | AIV Vector，双缓冲 Queue；BF16 使用 FP32 UB | 同 A2 | SIMT 1024 thread/block，GM 直通，无显式 UB。 |

## 限制与失败边界

- 950 正式入口仅取 `aclnnForeachSubListV2`；旧 `aclnnForeachSubList` 的公开产品表明确 950 不支持。
- arch35 tiling 明确拒绝 TensorList 长度大于 256；API 层未提前检查该上限，失败表现为 tiling `GRAPH_FAILED`。
- BF16 存在固定源码 ABI 不一致：API `ConvertToTensor(scalar, DT_FLOAT)`、OpDef/binary config 也声明 FP32 `alpha`，但 `Process<bfloat16_t>` 中 `reinterpret_cast<__gm__ T*>(alpha)` 按 BF16 读取。固定源码未提供 cast/fallback；因此不能从源码证明该组合数值正确。
- `totalElements` 的 int64 求和没有显式溢出检查；`ubSize-128 KiB` 在转 `uint32_t` 前没有 `ubSize>=128 KiB` guard。
- API 显式比较 `x1.Size` 与 `x2.Size`，但没有独立比较 `out.Size`；随后按 `x1.Size` 索引 `out`，调用方必须满足公开列表对应关系。
- TensorList 地址表布局由框架提供；本卡闭合算子自己的解码公式，但不扩展框架分配实现。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_sub_list/docs/aclnnForeachSubListV2.md` | 产品支持、函数原型、参数表 | 950 公开性、V2 语义与 dtype/shape 契约。 |
| 入口实现 | `foreach/foreach_sub_list/op_host/op_api/aclnn_foreach_sub_list_v2.cpp` | `CheckParams`、`ExecForeachSubListV2GetWorkspaceSize` | 校验、Contiguous、Scalar 转 Tensor、executor/workspace。 |
| 入口实现 | `foreach/foreach_sub_list/op_host/op_api/foreach_sub_list_v2.cpp` | `l0op::ForeachSubListV2` | 唯一 AICore launcher node。 |
| OpDef | `foreach/foreach_sub_list/op_host/foreach_sub_list_def.cpp` | `ops::ForeachSubList` | dtype、AutoContiguous、ascend950 注册。 |
| Host/Tiling/Launcher | `foreach/foreach_sub_list/op_host/arch35/foreach_sub_list_tiling_arch35.cpp` | `ForeachSubListTilingFunc` | blockDim、POD、workspace、key、local memory。 |
| POD | `foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_tiling_data.h` | `ForeachSubListTilingData` | ABI 字段。 |
| 模板 key | `foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_tiling_key.h` | `ForeachSubList` template args | 四个 `schMode`。 |
| Kernel entry/body | `foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list.cpp` | `foreach_sub_list<schMode>` | dtype router。 |
| Kernel entry/body | `foreach/foreach_sub_list/op_kernel/arch35/foreach_sub_list_simt.h` | `Process`、`OpForeachSubListSimt` | VF launch、地址、线程循环、精度。 |
| binary config | `foreach/foreach_sub_list/op_host/config/ascend950/foreach_sub_list_binary.json` | `ForeachSubList` 四项 binary | 950 dtype 与 alpha Tensor dtype。 |
| 构建接线 | `foreach/foreach_sub_list/op_kernel/CMakeLists.txt` | `add_kernel_sources(... ascend950)` | arch35 entry 的活动构建。 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 2064 B POD 与 TensorList 地址表解码 | 字段顺序、`tensorElements[256]`、首地址偏移右移 3 | `ForeachSubListTilingData`、`SimtGetTensorAddr` |
| 调用顺序 | Scalar 转 Tensor后加入单一 ForeachSubList node | 空列表短路；launcher 添加失败须阻断执行 | `ExecForeachSubListV2GetWorkspaceSize` |
| workspace/内存 | 算子 workspace/UB 为 0，GM 直通 | executor 仍可能包含框架 Contiguous workspace | `ForeachSubListTilingFunc`、`OpForeachSubListSimt` |
| 同步 | 元素唯一 writer，无 barrier/atomic | 只能复用于无跨元素依赖表达式 | `OpForeachSubListSimt` |
| 精度与尾块 | FP16/BF16 升 FP32；grid-stride `idx<count` | BF16 alpha 读取类型必须先对齐；降精度模式固定源码未显式命名 | 同上 |
| 失败边界 | `tensorNum<=256`、四 dtype、core/UB 非零 | 无未知 key fallback；tiling 失败必须向上返回 | `GetPlatformInfo`、`ForeachSubListTilingFunc` |