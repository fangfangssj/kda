---
type: Operator
title: "ScatterMul 950 ACLNN 算子知识"
description: "说明 aclnnScatterMul 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与 SIMT 原子乘法实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_mul/docs/aclnnScatterMul.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_mul/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_mul/op_host/scatter_mul_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_mul/op_host/arch35/scatter_mul_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_mul/op_kernel/arch35/scatter_mul.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_mul/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T16:59:07Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ScatterMul 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 公开两段式接口 `aclnnScatterMulGetWorkspaceSize` + `aclnnScatterMul`，本仓只有这一公开版本族 |
| 计算语义 | 对每个索引条目 `m` 和切片内偏移 `s`，原地执行 `var[indices[m], s] *= updates[m, s]`；越界索引由 Device 跳过 |
| 输入输出 | `varRef/updates`：ND、`float16/float32/int32/int8/uint8` 且 dtype 相同；`indices`：ND、`int32/int64`；输出与 `varRef` 共享内存 |
| 执行类型 | 单 Kernel、单 SIMT 节点；外层 entry 标记 `KERNEL_TYPE_AIV_ONLY`，核内通过 `asc_vf_call` 启动 512-thread VF；无 Cube/AIC、无 MIX_CV |
| Kernel 路线 | R-001：`TilingKey=0` 的 GM-direct SIMT CAS 路线；按元素宽度在 body 内编译期分为 32-bit CAS 与包围字 CAS |
| 核心资源 | `var/indices/updates` GM；寄存器标量与 CAS 状态；无 AIV UB Queue/TBuf、无 dynamic UB、无算子数据 workspace；Tiling 固定申请 16 MiB system workspace |
| 来源闭包 | `source_closed`：公开 ACLNN 文档 → `add_modules_sources(... ACLNNTYPE aclnn)` 自动 bridge 边界 → `ScatterMul` OpDef/tiling 注册 → 公共 POD/tiling → `scatter_mul` entry → `ScatterReduceSimt<..., MODE_MUL>` 与 CAS helper → ascend950 构建接线 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `D0 = varRef.shape[0]`，`S = product(varRef.shape[1:])`，`M = numel(indices)`。公开 shape 约束是 `updates.shape = indices.shape + varRef.shape[1:]`，所以 `numel(updates) = M * S`。Device 将 `updates` 展平，对每个 `m ∈ [0, M)`、`s ∈ [0, S)` 执行：

```text
if 0 <= indices[m] < D0:
    varRef[indices[m] * S + s] =
        varRef[indices[m] * S + s] * updates[m * S + s]
```

更新是原地的，Kernel 的 `y` 参数不被读取或写入。重复索引通过 CAS 重试串行化相同目标元素的 read-modify-write；不同线程的成功次序不固定，因此浮点重复乘法的舍入次序也不固定。负索引或 `indices[m] >= D0` 时源码直接 `continue`，不会更新任何元素。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `varRef` | ND，rank 1–8 | `FLOAT16/FLOAT32/INT32/INT8/UINT8` | 输入/输出原地对象；公开 API 要求非空，允许非连续 Tensor |
| `indices` | ND，rank 1–8 | `INT32/INT64` | `numel(indices)=M`；Device 对每个值做 `[0,D0)` 边界检查 |
| `updates` | ND，rank 1–8 | 与 `varRef` 相同 | `updates.shape = indices.shape + varRef.shape[1:]`；展平后第 `m*S+s` 项乘到目标切片 |
| `useLocking` | 标量属性 | `BOOL` | 默认 `false`；固定源码的 OpDef 声明该属性，但 tiling/POD/kernel 均不消费它，原子 CAS 路线不随其值变化 |
| `workspaceSize` | 标量输出 | `uint64_t` | 第一段接口返回；算子 tiling 自有请求为 16 MiB system workspace，生成式 ACLNN executor 的总量计算属于框架边界 |
| `executor` | opaque | `aclOpExecutor *` | 第一段接口创建，第二段接口在给定 stream 上执行 |

### 入口与版本

公开入口是 `aclnnScatterMulGetWorkspaceSize(varRef, indices, updates, useLocking, workspaceSize, executor)` 和 `aclnnScatterMul(workspace, workspaceSize, executor, stream)`。固定仓未提供手写 ACLNN inner 实现；`index/scatter_mul/CMakeLists.txt::add_modules_sources` 以 `OPTYPE scatter_mul ACLNNTYPE aclnn` 接入生成式 bridge，这是本卡明确保留的框架边界。算子拥有的 OpDef、Tiling、POD、Kernel router/body 和 build target 均已继续追到实际实现。

README 与 API 文档均只把 Ascend 950PR/950DT 标为支持，把 Atlas A2/A3 明确标为不支持；OpDef 的 `AICore().AddConfig("ascend950", ...)`、顶层 `SUPPORT_COMPUTE_UNIT "ascend950"` 和 kernel `COMPUTE_UNITS ascend950` 与公开支持表一致。因此本卡仅覆盖 950，不将源码注释中提到的历史 A2/A3 TBE 行为视作当前公开支持或当前实现版本。

## Host/Launcher 执行流程

### 参数校验与前处理

公开文档声明第一段接口完成以下校验：三个 Tensor 指针非空；dtype 在各自支持集合中；`varRef` 与 `updates` dtype 相同；`updates.shape` 满足拼接约束。不满足时分别返回 `ACLNN_ERR_PARAM_NULLPTR` 或 `ACLNN_ERR_PARAM_INVALID`。非连续 Tensor 被公开标为支持，但固定仓未包含生成式 bridge 的连续化/视图处理源码，因此具体 copy/view 节点、临时区和失败点属于 ACLNN 框架边界，不能从本 Commit 推断。

注册后的 `ScatterMul` 输出 shape/dtype 由 `InferShape4ScatterMul` 与 `InferDataType4ScatterMul` 复制输入 `var`。`use_locking` 不进入 `ScatterReduceCommonTiling` 或 Kernel POD；当前实现始终依赖 CAS 保证单元素原子更新。

### 参数、Workspace 与生命周期

`TilingPrepareForScatterReduce` 从 `PlatformAscendC::GetCoreNumAiv()` 生成 `ScatterReduceCompileInfo::coreNum`。`ResolveCoreNum` 优先读取 compile info；ACLNN runtime 未携带 compile info 时改从 `TilingContext::GetPlatformInfo()` 读取 AIV 核数；结果为 0 时归一为 1。

算子 tiling 通过 `GetWorkspaceSizes(1)` 写 `workspaces[0] = 16 * 1024 * 1024` bytes。`scatter_mul` entry 虽接收 `workspace`，实际 body 不访问它，所以这 16 MiB 是运行时 system workspace 请求，不是算子节点间数据边。公开 executor 负责 workspace owner/lifetime：调用者按第一段返回总量申请 Device 内存，并保证其至少持续到第二段异步任务在指定 stream 上完成。

### 启动图与失败传播

```text
aclnnScatterMulGetWorkspaceSize
  -> 生成式 ACLNN bridge（固定仓不可见的框架边界）
  -> ScatterMul OpDef / InferShape / Tiling4ScatterMul
  -> ScatterReduceCommonTiling
       产生 TilingKey=0、blockDim、ScatterReduceSimtTilingData、16 MiB system workspace

aclnnScatterMul(workspace, workspaceSize, executor, stream)
  -> 同一 stream 上启动唯一 Device 节点 K-001 scatter_mul
       inputs: var, indices, updates
       output: 原地 var（形式参数 y 不使用）
       workspace: 不使用数据 slice
       blockDim: tiling.blockNum 个 AIV core
       per-core VF: asc_vf_call(dim3(512), ...)
```

K-001 没有前后 Kernel 节点，也没有 workspace/GM 中间边。第一段参数校验、tiling 空指针检查或 platform 解析失败时 executor 构造失败，不应启动 K-001；第二段执行状态由 ACLNN 框架传播。固定仓未显式展示生成 bridge 内部的 stream enqueue 与 cleanup 代码，不补造额外 event 或同步。

## Route 与参数设计

### 分核与资源计算

Tiling 使用 storage shape，单位如下：

```text
D0 = var rank == 0 ? 1 : var.shape[0]                    # 元素维度
V  = numel(var)                                           # 元素
S  = D0 == 0 ? 0 : V / D0                                # 元素/索引切片
M  = numel(indices)                                       # 索引条目
C  = resolved AIV core count, 至少 1                      # core

B0 = M == 0 ? 1 : min(M, C)                              # 初选 core
P  = B0 == 0 ? 0 : ceil_div(M, B0)                       # 索引条目/前核
B  = P == 0 ? 1 : ceil_div(M, P)                         # 实际 blockNum
T  = M - (B - 1) * P                                     # 尾核索引条目

blockTilingSize     = P * S                               # update 元素/前核
tailBlockTilingSize = T * S                               # update 元素/尾核
blockDim            = B                                  # AIV core 数
```

每核拿到完整索引切片，因此 `xBlockOffset = blockIdx * P * S`。核内没有 tile 或 UB 上限；512 个 SIMT 线程以步长 512 遍历该核的 `blockElems`。所有量与 POD 字段均为 `uint64_t`，但固定 tiling 源码对 shape-size 乘法、`P*S`、`T*S` 没有显式溢出检查；合法 shape 与可表示范围依赖上游框架。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 GM-direct SIMT CAS | `TilingKey=0`; `ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MUL>` | 所有通过 ACLNN/OpDef dtype 与 shape 校验、可进入 tiling 的 950 请求；没有第二 selector | 参数/tiling 失败则不启动；`S==0` 或核号越界则 Device 早退 |

```text
公开 A5 ScatterMul 请求
  -> 参数合法？否：ACLNN 参数错误
  -> ResolveCoreNum 成功？否：GRAPH_FAILED
  -> SetTilingKey(0), SetBlockDim(B)
  -> R-001
       sizeof(DTYPE_VAR) == 4 ? AtomicReduce32 : AtomicReduceSubWord
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 GM-direct SIMT CAS | key `0`; 元素宽度为 body 内 `if constexpr` | 950；`var/updates ∈ {fp16,fp32,int32,int8,uint8}` 同 dtype；`indices ∈ {int32,int64}`；shape 通过公开约束 | 单节点 K-001 `scatter_mul`；`B` 个 AIV core，每核一个 512-thread VF | `ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MUL>::Init/Process` → `SimtComputeReduce` → `AtomicReduce32` 或 `AtomicReduceSubWord` | GM direct；无 UB；CAS 对同一 32-bit 地址线性化；不同地址无 barrier | 尾核使用 `tailBlockTilingSize`；线程 grid-stride 尾部由 `index < blockElems` 保护；越界索引跳过；`S==0` 早退 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 平台公开支持 | 文档为 950PR/950DT √ 且 `SUPPORT_COMPUTE_UNIT=ascend950` | 只有 arch35 Host/Kernel 与 ascend950 target | R-001 | A2/A3 不发布、不进入此实现 | `README.md::产品支持情况`; `CMakeLists.txt`; `ScatterMul::ScatterMul` |
| 有效 AIV 核数 | `coreNum==0` 时强制为 1 | 防止分核除零，保证即便空 indices 也有合法 blockDim | R-001 | 始终归一为 1，不存在 fallback | `ResolveCoreNum` |
| 空索引 | `M==0` 时 `B=1,P=0,T=0` | 保持合法 launch，VF 循环零次 | R-001 | 空操作 | `ScatterReduceCommonTiling`; `SimtComputeReduce` |
| 空切片 | `D0==0 -> S=0`，或合法 shape 得到 `S==0` | `Init/Process` 不绑定或访问 GM，避免除以零 `globalIdx/S` | R-001 | Kernel 早退 | `ScatterReduceSimt::Init/Process` |
| 索引边界 | `idxVal < 0 \|\| uint64(idxVal) >= varFirstDim` | 避免目标 GM 越界；先检查负数再做无符号转换 | R-001 | 跳过该索引的当前元素 | `SimtComputeReduce` |
| 元素宽度 | `sizeof(PARAMS_T)==4` 编译期判断 | CAS API 直接支持 `float/int32`，不直接支持 16/8-bit | R-001 内部分支 | 2/1-byte 类型走包围 32-bit word CAS | `SimtComputeReduce` |
| 包围字对齐 | `var` GM 基址至少 4-byte 对齐；源码注释给出实际 GM tensor ≥32-byte 对齐 | `reinterpret_cast<uint32_t*>` 后按 `byteOff>>2` 定位原子字 | R-001 subword | 固定源码没有运行时 fallback 或对齐错误返回 | `AtomicReduceSubWord` |
| 固定线程数 | `THREAD_NUM=512`，`asc_vf_call(dim3(THREAD_NUM),...)` | 用 grid-stride 循环覆盖任意 `blockElems`，无 UB tile cap | R-001 | 无其他线程配置 | `THREAD_NUM`; `ScatterReduceSimt::Process` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ScatterReduceCompileInfo` | `coreNum` | `uint64_t` / 64-bit | 单字段；固定仓无 `sizeof/alignof/offsetof` 断言 | AIV core | `TilingPrepareForScatterReduce`: `GetCoreNumAiv()` | `ResolveCoreNum`: 决定 `C` | 0 被归一为 1；ACLNN 无 compile info 时不读取 |
| `ScatterReduceSimtTilingData` | `blockNum` | `uint64_t` / 64-bit | 第 1 字段；精确 offset/总 `sizeof` 固定仓不可见 | core/block | `td->blockNum = B` | `Init`: 尾核判断；`Process`: 核号 guard | 至少 1 |
| `ScatterReduceSimtTilingData` | `blockTilingSize` | `uint64_t` / 64-bit | 第 2 字段；精确 offset 固定仓不可见 | update 元素/core | `td->blockTilingSize = P*S` | `Init`: `xBlockOffSet_` 与前核 `currBlockTilingSize_` | `P*S` 无显式溢出 guard |
| `ScatterReduceSimtTilingData` | `tailBlockTilingSize` | `uint64_t` / 64-bit | 第 3 字段；精确 offset 固定仓不可见 | update 元素 | `td->tailBlockTilingSize = T*S` | `Init`: 最后一个 block 的 `currBlockTilingSize_` | 空 indices 时为 0 |
| `ScatterReduceSimtTilingData` | `sliceSize` | `uint64_t` / 64-bit | 第 4 字段；精确 offset 固定仓不可见 | 元素/index | `td->sliceSize = S` | `Init/Process` 空域 guard；VF 计算 `m/sliceOff/dstIndex` | 0 时早退 |
| `ScatterReduceSimtTilingData` | `varFirstDim` | `uint64_t` / 64-bit | 第 5 字段；精确 offset/总 `sizeof` 固定仓不可见 | 元素 | `td->varFirstDim = D0` | VF 索引上界检查 | rank 0 在 tiling 内视作 1；公开 API rank 为 1–8 |

Host 通过 `context->GetTilingData<ScatterReduceSimtTilingData>()` 直接写结构，Kernel 用 `REGISTER_TILING_DEFAULT` 和 `GET_TILING_DATA_WITH_STRUCT` 读取同名结构。字段顺序与位宽源码闭合；目标 ABI 对齐数值没有在固定仓以 `static_assert` 或序列化元数据显式锁定，移植时必须保持 Host/Device 同一结构定义与工具链 ABI。

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 GM-direct SIMT CAS | `SetTilingKey(0)`；body 内 `sizeof(PARAMS_T)` | `Tiling4ScatterMul` → `ScatterReduceCommonTiling` / `ResolveCoreNum` | `ScatterReduceCompileInfo`; `ScatterReduceSimtTilingData` 五字段 | `scatter_mul`: 注册 POD，实例化 `ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MUL>` | `ScatterReduceSimt::Init/Process`; `SimtComputeReduce`; `AtomicReduce32`; `AtomicReduceSubWord`; `ReduceAcc<...,MODE_MUL>` | `index/scatter_mul/CMakeLists.txt`; `index/scatter_mul/op_kernel/CMakeLists.txt`; shared Host dependency `scatter_reduce_common` |

## Kernel 实现

### 公共执行流程

只有 R-001。Host 启动一个 `scatter_mul` 节点；entry 创建 `TPipe` 但 body 明确不使用它，标记 AIV-only carrier，解析公共 POD，并把 `DTYPE_VAR/DTYPE_INDICES`、64-bit 地址类型和 `MODE_MUL` 固化进模板。每个 AIV core 计算自己的连续 flattened-update 区间，再以 `asc_vf_call` 让 512 个 SIMT thread 直接从 GM 读取 indices/updates、对原地 var 做 CAS。没有 `CopyIn/CopyOut`、Queue、TBuf、event 或跨 Kernel 边。

### Route R-001：GM-direct SIMT CAS

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `index/scatter_mul/docs/aclnnScatterMul.md` | `aclnnScatterMulGetWorkspaceSize` / `aclnnScatterMul` | 支持 dtype、相同 var/update dtype、shape 拼接约束 | 合法请求进入 executor；非法请求返回 ACLNN 错误 |
| OpDef | `index/scatter_mul/op_host/scatter_mul_def.cpp` | `ScatterMul::ScatterMul` | dtype tuple 与 `ascend950` config | 注册 `ScatterMul` 950 实现 |
| Tiling wrapper | `index/scatter_mul/op_host/arch35/scatter_mul_tiling.cpp` | `Tiling4ScatterMul` | 无分支，直接调用公共 tiling | 进入 `ScatterReduceCommonTiling` |
| Tiling | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `ScatterReduceCommonTiling` | shape 指针、POD、workspace 指针非空且 core 解析成功；`SetTilingKey(0)` | 产生 R-001 POD 与 `blockDim=B` |
| Router | `index/scatter_mul/op_kernel/arch35/scatter_mul.cpp` | `scatter_mul` | 固定 `ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MUL>` | 单一 body，无 unknown-key 分支 |
| Body | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `SimtComputeReduce` | `sizeof(PARAMS_T)==4` | 32-bit 直 CAS，否则包围字 CAS |

#### Device 数据流

##### 函数级调用链

```text
Host K-001 launch: scatter_mul(var, indices, updates, y, workspace, tiling)
  -> REGISTER/GET ScatterReduceSimtTilingData
  -> ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MUL>::Init
       -> GetBlockIdx / 计算 xBlockOffSet_ 与 currBlockTilingSize_
       -> 绑定 indices、updates、var 三个 GlobalTensor
  -> ScatterReduceSimt<...>::Process
       -> asc_vf_call<SimtComputeReduce<...>>(dim3(512), GM pointers, core range, S, D0)
          -> grid-stride loop
          -> ReduceAcc<ACC, MODE_MUL>
          -> AtomicReduce32 或 AtomicReduceSubWord
```

##### 源码等价伪码

```text
# 源码等价伪码；所有 offset 单位均为元素，除 byteOff/wordAddr 外
if S == 0: return
b = GetBlockIdx()
begin = blockTilingSize * b
n = (b == blockNum - 1) ? tailBlockTilingSize : blockTilingSize
if b >= blockNum: return

vf_launch(threads=512):
  for local = threadIdx.x; local < n; local += blockDim.x:  # VF blockDim.x=512
    global = begin + local
    m = global / S
    s = global - m * S
    idx = indices[m]
    if idx < 0 or uint64(idx) >= D0: continue
    dst = uint64(idx) * S + s

    if sizeof(VAR_T) == 4:
      old = var[dst]
      do:
        expected = old
        desired = VAR_T(expected * updates[global])
        old = atomic_cas(&var[dst], expected, desired)
      while old != expected
    else:
      byteOff = dst * sizeof(VAR_T)
      word = reinterpret<uint32*>(var) + (byteOff >> 2)
      shift = (byteOff & 3) * 8
      keepMask = ~(subMask << shift)
      updateAcc = ACC(updates[global])                 # fp16->fp32, int8/uint8->int32
      oldWord = *word
      do:
        expectedWord = oldWord
        oldElem = extract(expectedWord, shift, subMask)
        newElem = VAR_T(ACC(oldElem) * updateAcc)      # 回写前降回原 dtype
        desiredWord = preserve_neighbors_and_insert(expectedWord, newElem)
        oldWord = atomic_cas(word, expectedWord, desiredWord)
      while oldWord != expectedWord
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM `indices` | `m = (begin+local)/S` | `M` elements | `int32/int64`, ND flatten | 调用者 | VF `idxGmAddr[m]` | 调用者；Kernel 期间只读 | 无 staging；逐 update element 会重复读同一 slice 的 index |
| GM `updates` | `global = begin+local` | `M*S` elements | same as var, ND flatten | 调用者 | VF `xGmAddr[global]` | 调用者；Kernel 期间只读 | 无 staging；线程 grid-stride |
| GM `varRef` | `dst=idx*S+s` | `D0*S` elements | supported value dtype, ND flatten | 调用者初值/CAS 前次结果 | CAS helper | 调用者，原地输入输出 | 32-bit 元素按元素 CAS；subword 共享 32-bit 原子字并保留邻元素 |
| 寄存器/线程标量 | `global,m,s,idx,dst` 与 CAS `cur/prev/new` | 固定仓未提供编译后寄存器字节数 | `int64_t` 地址量、dtype/ACC、`uint32_t` CAS word | VF/thread | 地址计算与 CAS retry | 单线程单迭代或 retry-loop 生命周期 | 不跨线程共享 |
| AIV UB Queue/TBuf | N/A | 0 bytes（源码未 `InitBuffer`） | N/A | N/A | N/A | `TPipe` 仅构造，body `(void)pipe` | GM-direct，无复用 |
| dynamic UB/shared | N/A | 0 bytes（launch 未传 dynamic UB） | N/A | N/A | N/A | N/A | 实际替代路径为寄存器 + GM CAS |
| system workspace | Kernel 未形成地址 | 16 MiB tiling request | runtime opaque | ACLNN caller/runtime | Kernel 不消费 | executor/stream 生命周期 | 非节点间数据区，不能从源码扣作 UB |

##### 分层内存布局

本 Route 未使用 AIV UB，不伪造 `block-beta`。真实布局是 GM `indices`/`updates` 只读、GM `varRef` 原地原子更新；每个 VF thread 只持有地址计算与 CAS retry 的寄存器标量。相邻 subword 元素可能共享一个 32-bit 原子字，但 `keepMask` 在每次 CAS retry 中基于最新成功候选重建未更新 lane。

##### 字节账与复用顺序

```text
AIV_queue_bytes = 0
AIV_tbuf_bytes = 0
dynamic_ub = 0 / N/A（asc_vf_call 未配置 dynamic UB）
UB_total = 0
operator_data_workspace = 0
tiling_system_workspace_request = 16 * 1024 * 1024 = 16,777,216 bytes
public_executor_workspace_total = 固定生成式 ACLNN bridge 不可见，由第一段接口返回
=> blockElems 不受 UB tile 限制；由 512-thread grid-stride 循环直接遍历 GM
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| GM `varRef[dst]` 32-bit 元素 | 成功的 `AtomicReduce32` CAS | 后续冲突线程的 CAS load/retry；最终调用者 | 下一次同地址成功 CAS | `asc_atomic_cas` 比较失败后返回当前值并重试 | 所有更新必须只通过同一原子协议 |
| GM 包围 32-bit word | 成功的 `AtomicReduceSubWord` CAS | 更新同字任意 sub-element 的线程 | 下一次成功 word CAS | 每次失败都以返回的新 word 重新 extract/reduce/merge | 基址至少 4-byte 对齐；`keepMask` 必须保留非目标 bits |
| 寄存器 `cur/prev` | 当前线程 load/CAS 返回 | 当前 retry 判断 | 下一 retry 覆盖 | `cur != prev` 循环条件 | 不跨线程共享，无显式释放 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 核间/线程冲突更新 | `asc_atomic_cas` retry | 任意命中同一目标的 thread/core | CAS 失败线程读取返回当前值后重算 | 32-bit GM 原子地址 | 成功次序不固定；无全局 barrier |
| 32-bit value 计算 | `ReduceAcc<T,MODE_MUL>` | `float` 或 `int32` 当前值与 update | 目标 CAS | 单元素 | 在原 dtype 中乘；整数溢出行为未由仓内额外约束 |
| FP16 subword | `half -> float`，float 乘，`float -> half` 后 CAS | 提取的 half 与 update half | 包围字 CAS | 16-bit lane | 每个成功更新都会降精度回 half；重复索引成功次序影响舍入 |
| INT8/UINT8 subword | `int8/uint8 -> int32`，int32 乘，降回原类型 | 提取元素与 update | 包围字 CAS | 8-bit lane | 窄化行为由 C++ cast/目标工具链决定；源码无饱和指令 |
| 相邻 subword 冲突 | 32-bit word CAS + `keepMask` | 同 word 不同 lane 的 threads | 失败者 retry | 整个 32-bit word | 增加冲突域但保持其他 lane 的最新 bits |
| 前核/尾核 | `currBlockTilingSize_` 条件选择 | POD `blockTilingSize/tailBlockTilingSize` | VF loop | 每核连续 update 区间 | 最后核精确取 `T*S`；无 padding lane |
| 线程尾部 | `local < blockElems` | 512-thread VF | 每线程 grid-stride loop | 当前核范围 | 非 512 倍数自然由循环谓词保护 |
| 空域 | `S==0` 或 `blockIdx>=blockNum` 早退 | POD/核号 | `Init/Process` | 所有 GM | 不读取 GM，不写输出 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `TilingPrepareForScatterReduce`; `ResolveCoreNum`; `ScatterReduceCommonTiling` | 生成 core split、五字段 POD、key 0、blockDim 与 16 MiB workspace |
| Kernel entry/router | `index/scatter_mul/op_kernel/arch35/scatter_mul.cpp` | `scatter_mul`; `ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MUL>` | 固定乘法 mode、64-bit 地址类型、AIV-only carrier 与单 Route |
| Device body/helper | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `ScatterReduceSimt::Init/Process`; `SimtComputeReduce`; `AtomicReduce32`; `AtomicReduceSubWord`; `ReduceAcc` | 分核地址、512-thread grid-stride、边界跳过、两类 CAS、精度转换与 retry |

## 算子特定优化

### Trick T-001：GM-direct 512-thread grid-stride 消除 UB tile 上限

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：所有进入 950 唯一 Route 的请求；每个 core 的 `currBlockTilingSize_` 可为任意非负元素数。
- Golden 来源：[`index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h:163-174`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L163-L174)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename PARAMS_T, typename INDICES_T, typename ADDR_T, uint8_t Mode>
__aicore__ inline void ScatterReduceSimt<PARAMS_T, INDICES_T, ADDR_T, Mode>::Process()
{
    if (tiling_.sliceSize == 0 || blockIdx_ >= tiling_.blockNum) {
        return;
    }
    // One SIMT launch over this core's whole element range; the grid-stride loop and direct GM reads
    // handle any size without a UB tile, so there is no per-tile element cap.
    asc_vf_call<SimtComputeReduce<INDICES_T, PARAMS_T, ADDR_T, Mode>>(
        dim3(THREAD_NUM), (__gm__ INDICES_T*)idxGm_.GetPhyAddr(), (__gm__ PARAMS_T*)xGm_.GetPhyAddr(),
        (__gm__ PARAMS_T*)outputGm_.GetPhyAddr(), xBlockOffSet_, currBlockTilingSize_,
        static_cast<ADDR_T>(tiling_.sliceSize), static_cast<ADDR_T>(tiling_.varFirstDim));
}
```

#### 机制与设计原因

`THREAD_NUM=512` 是编译期线程数，`xBlockOffSet_` 与 `currBlockTilingSize_` 来自 Host 的按完整 slice 分核结果；VF 内线程以 `blockDim.x` 为步长遍历。indices、updates 和 var 都以 GM 指针直接传入，不做 UB `CopyIn/CopyOut`，因此 Host 无需根据 UB 容量再切 tile，超大 slice 也沿同一 Route 运行。代价是同一 `indices[m]` 会被 slice 内不同元素线程重复读取；固定源码没有 index cache 路线。

#### 迁移边界

- 适用：目标操作可以逐元素直接从 GM 完成，且冲突写能用目标平台原子协议闭合的 scatter/reduce。
- 前提：950 SIMT/VF；合法 GM 地址；线程迭代互不依赖，重复目标由 CAS 处理；地址量保持 64-bit。
- 失效：需要跨元素归约、UB 数据复用或平台没有所需原子能力时，必须设计新的 Route，不能沿用此 GM-direct body。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：包围 32-bit word CAS 支持 FP16/INT8/UINT8 原地原子乘

#### 触发与源码依据

- 适用 Route：R-001 的 `sizeof(PARAMS_T) != 4` 分支。
- 触发条件：`varRef/updates` 为 `float16/int8/uint8`；GM 基址满足源码声明的至少 4-byte 对齐。
- Golden 来源：[`index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h:63-95`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h#L63-L95)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
template <typename T, uint8_t Mode>
__simt_callee__ inline void AtomicReduceSubWord(__gm__ uint32_t* base, uint64_t elemIdx, T val)
{
    using ACC = typename Conditional<AscendC::IsSameType<T, half>::value, float, int32_t>::type;
    constexpr uint32_t SUBMASK = (sizeof(T) == 1) ? 0xFFU : 0xFFFFU;
    uint64_t byteOff = elemIdx * sizeof(T);
    __gm__ uint32_t* wordAddr = base + (byteOff >> 2);
    uint32_t shift = (static_cast<uint32_t>(byteOff) & 3U) * 8U;
    uint32_t keepMask = ~(SUBMASK << shift);
    ACC valAcc = static_cast<ACC>(val);
    uint32_t cur = *wordAddr;
    uint32_t prev;
    do {
        prev = cur;
        uint32_t subBits = (prev >> shift) & SUBMASK;
        T curVal;
        if constexpr (AscendC::IsSameType<T, half>::value) {
            uint16_t hb = static_cast<uint16_t>(subBits);
            curVal = *reinterpret_cast<half*>(&hb);
        } else if constexpr (AscendC::IsSameType<T, int8_t>::value) {
            curVal = static_cast<int8_t>(subBits);
        } else {
            curVal = static_cast<uint8_t>(subBits);
        }
        T newVal = static_cast<T>(ReduceAcc<ACC, Mode>(static_cast<ACC>(curVal), valAcc));
        uint32_t newBits;
        if constexpr (AscendC::IsSameType<T, half>::value) {
            newBits = static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&newVal));
        } else {
            newBits = static_cast<uint32_t>(static_cast<uint8_t>(newVal));
        }
        uint32_t newWord = (prev & keepMask) | ((newBits & SUBMASK) << shift);
        cur = asc_atomic_cas(wordAddr, prev, newWord);
    } while (cur != prev);
}
```

#### 机制与设计原因

硬件 helper 没有 8/16-bit 指针 CAS overload，代码把目标元素映射到其包围 32-bit word：`byteOff>>2` 定位 word，`(byteOff&3)*8` 定位 lane，`keepMask` 保留其他 lane。每次 CAS 失败都使用返回的最新 word 重新提取、乘法和合并，所以即使相邻元素由其他线程更新也不会用旧 word 覆盖它。FP16 在 float 中做本次乘法、INT8/UINT8 在 int32 中做本次乘法，再降回存储类型。

#### 迁移边界

- 适用：原子 API 不直接覆盖的 8/16-bit 独立元素 read-modify-write。
- 前提：对象基址和 allocation 边界能安全容纳包围 32-bit word；lane 位布局与源码一致；所有竞争写遵循 word CAS；窄化语义可接受。
- 失效：未对齐/不足 4-byte 可访问范围、bit layout 不同、需要饱和乘法或必须保持更高精度累积时不能直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只原子写目标 subword 而不保留同一 word 的其他 lane，会丢失相邻线程的更新；源码通过 `keepMask` 和失败后完整重算避免这一风险。

### Trick T-003：按完整索引切片分核，地址区间连续且无需跨核 partial 合并

#### 触发与源码依据

- 适用 Route：R-001。
- 触发条件：`S=product(var.shape[1:])`；Host 按 `P` 个索引条目而非任意元素边界分核。
- Golden 来源：[`index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp:76-90`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp#L76-L90)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
    // split index entries across cores; each core handles whole slices, read directly from GM in the VF
    uint64_t blockNum = (indicesNum == 0) ? 1 : (indicesNum < coreNum ? indicesNum : coreNum);
    uint64_t perCoreIndices = (blockNum == 0) ? 0 : (indicesNum + blockNum - 1) / blockNum;
    blockNum = (perCoreIndices == 0) ? 1 : (indicesNum + perCoreIndices - 1) / perCoreIndices;
    uint64_t tailCoreIndices = indicesNum - (blockNum - 1) * perCoreIndices;

    auto* td = context->GetTilingData<ScatterReduceCommon::ScatterReduceSimtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    td->blockNum = blockNum;
    td->blockTilingSize = perCoreIndices * sliceSize;
    td->tailBlockTilingSize = tailCoreIndices * sliceSize;
    td->sliceSize = sliceSize;
    td->varFirstDim = varFirstDim;

    context->SetBlockDim(blockNum);
```

#### 机制与设计原因

`blockTilingSize=P*S` 保证 core 边界落在 index slice 边界，故每核 updates 区间是连续的，并且 `(global/S, global%S)` 始终能恢复 index entry 与 slice offset。不同 core 仍可能因重复 `indices` 命中同一 var slice，但 CAS 已处理写冲突；无需产生 per-core partial workspace 或第二个 merge Kernel。重算 `blockNum=ceil(M/P)` 避免最后出现无数据 core。

#### 迁移边界

- 适用：updates 的前缀与 indices 一一对应，尾维组成等长切片的 scatter-reduce。
- 前提：`updates.numel=M*S`，每个 slice 长度统一，`P*S` 可在 POD 64-bit 范围表示。
- 失效：变长 slice、ND index tuple 或需要按 slice 内维度做特殊布局时，必须重做地址映射与分核。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 GM-direct SIMT CAS | 设计选择依据 |
|---|---|---|
| 命中条件/优先级 | 950 所有公开合法 dtype/shape；唯一 key 0 | 无 Host fallback 或第二模板 |
| 模板与 dtype | `DTYPE_VAR × DTYPE_INDICES × int64_t × MODE_MUL`；32-bit/subword 编译期分支 | dtype tuple 由 OpDef/build specialization，乘法 mode 在 entry 固化 |
| 循环与分核 | 按完整 index slice 分 `B` 个 core；每核 512 threads grid-stride | 连续 updates 区间，无 UB tile cap |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 无 Queue/TBuf/dynamic UB；仅线程寄存器；无通信 | `TPipe` 不使用，GM-direct |
| 内存总占用与复用 | UB 0；system workspace 16 MiB；var 原地 CAS | 不物化 partial/output；subword 复用包围 32-bit word |
| 搬运/Vector/SIMT/NDDMA/通信 API | SIMT `asc_vf_call` + GM load + `asc_atomic_cas` | 唯一执行族是 SIMT，无 Vector/NDDMA/通信节点 |
| 同步、精度与 tail | CAS retry；FP16/小整数升精度单步计算后降回；尾核和线程谓词 | 原子化冲突写，浮点成功次序不固定 |
| fallback/失败边界 | 无其他 Route；非法参数/tiling 失败，空域早退，越界 index 跳过 | 固定源码的唯一终点 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 README/API 明确不支持 | 公开 README/API 明确不支持 | 950PR/950DT 公开支持 |
| Host/Launcher | 不发布；当前仓无对应支持接线 | 不发布；当前仓无对应支持接线 | arch35 公共 tiling，key 0，按 AIV core 分索引条目 |
| Kernel/资源 | 不发布；`scatter_mul_def.cpp` 注释提到历史 TBE 行为不能作为当前实现证据 | 同 A2 | ascend950 单 SIMT Kernel；GM-direct CAS；无 UB；16 MiB system workspace |

## 限制与失败边界

- 仅覆盖公开 `aclnnScatterMul` 的 950/Ascend 950PR、950DT 路线；A2/A3 被公开文档明确标为不支持，不能凭注释或 arch35 文件名外推。
- 固定仓不含生成式 ACLNN inner bridge 源码；非连续 Tensor 的具体 contiguous/view-copy 节点、executor 总 workspace 构成、stream enqueue/cleanup 是明确框架边界。算子自有 tiling 请求仅能证明固定 16 MiB system workspace，Kernel 本身不使用该指针。
- `useLocking` 在公开接口与 OpDef 中存在，但固定 Host/POD/Kernel 不消费；不能把它解释为 Route 开关。CAS 始终使用。
- 重复索引的 CAS 成功顺序不固定；FP32 与每步回写 FP16 的乘法舍入结果可能受此顺序影响。固定仓没有确定性排序或高精度跨更新累积 Route。
- Device 对负值和 `>=D0` 的索引静默跳过；公开 ACLNN 错误表未声明索引值域错误。迁移实现不得未经接口约定改为报错或 wraparound。
- subword 路线把 FP16 提升为 FP32、INT8/UINT8 提升为 INT32 做单次乘法后立即降回；源码无饱和 cast。整数溢出/窄化与浮点特殊值行为未由固定仓进一步约束。
- `P*S`、`T*S`、`idx*S+s` 和 shape size 到 `uint64_t` 的路径无显式 overflow guard；依赖上游合法 shape 与地址范围。POD 精确 `sizeof/alignof/offsetof` 没有仓内断言，Host/Device 必须共享结构与 ABI。
- 包围字 CAS 假定 var 基址至少 4-byte 对齐，源码注释称 GM tensor 实际至少 32-byte 对齐；没有运行时对齐 fallback。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `index/scatter_mul/docs/aclnnScatterMul.md` | `aclnnScatterMulGetWorkspaceSize`; `aclnnScatterMul` | 公开两段式接口、dtype/shape、非连续支持、错误码和 950/A2/A3 支持表 |
| 入口实现 | `index/scatter_mul/CMakeLists.txt` | `add_modules_sources(... OPTYPE scatter_mul ACLNNTYPE aclnn ...)` | 生成式 ACLNN bridge 接线、ascend950 与 shared Host dependency |
| OpDef | `index/scatter_mul/op_host/scatter_mul_def.cpp` | `ScatterMul::ScatterMul`; `OP_ADD` | dtype tuple、属性、ascend950 AICore config |
| Host/Tiling/Launcher | `index/scatter_mul/op_host/arch35/scatter_mul_tiling.cpp` | `Tiling4ScatterMul`; `IMPL_OP_OPTILING` | ScatterMul 到公共 tiling 的真实委托 |
| Host/Tiling/Launcher | `index/scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.cpp` | `TilingPrepareForScatterReduce`; `ResolveCoreNum`; `ScatterReduceCommonTiling` | core、slice、POD、key、blockDim、workspace producer |
| POD | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_struct.h` | `ScatterReduceSimtTilingData`; `MODE_MUL` | 五字段 ABI 顺序与乘法 mode |
| Kernel entry/body | `index/scatter_mul/op_kernel/arch35/scatter_mul.cpp` | `scatter_mul` | entry 参数、POD 解析、AIV-only 标记和模板实例 |
| Kernel entry/body | `index/scatter_reduce_common/op_kernel/arch35/scatter_reduce_common_simt.h` | `ScatterReduceSimt`; `SimtComputeReduce`; atomic helpers | 512-thread GM-direct 循环、地址、边界、CAS、精度与 tail |
| 构建接线 | `index/scatter_mul/op_kernel/CMakeLists.txt` | `add_kernel_sources` | arch35 kernel 源与 `COMPUTE_UNITS ascend950` |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 五个 `uint64_t` 字段直接结构传递 | 字段顺序、位宽、元素单位；Host/Device 同一结构与 ABI | `ScatterReduceSimtTilingData`; `GET_TILING_DATA_WITH_STRUCT` |
| 调用顺序 | 单节点 entry → `Init` → `Process` → VF → CAS helper | `S==0` 早退必须先于 VF 中的除法；先 Init 后 Process | `scatter_mul`; `ScatterReduceSimt::Init/Process` |
| workspace/内存 | GM-direct，无数据 workspace/UB tile；system workspace 16 MiB | 不把 system workspace 当 partial buffer；保持 var 原地 owner/lifetime | `ScatterReduceCommonTiling`; `ScatterReduceSimt` constructor |
| 同步 | 同目标通过 CAS retry，同字 subword 每次失败完整重算 | 不能用非原子 load/store；`keepMask` 必须保留邻 lane | `AtomicReduce32`; `AtomicReduceSubWord` |
| 精度与尾块 | 32-bit 原 dtype；subword 用 FP32/INT32 单步计算后降回；尾核 + grid-stride predicate | FP16 不得跨多次更新一直保留 FP32；尾核必须用 `tailBlockTilingSize` | `ReduceAcc`; `AtomicReduceSubWord`; `ScatterReduceSimt::Init` |
| 失败边界 | 参数失败不 launch；空 slice 早退；越界 index 跳过 | 不得把越界 index wrap；不得为 A2/A3 复用 950 卡 | ACLNN 文档错误表；`SimtComputeReduce`; README 支持表 |

- [asc_atomic_cas](../../../../apis/simt_api/atomic_operations/asc_atomic_cas.md)
