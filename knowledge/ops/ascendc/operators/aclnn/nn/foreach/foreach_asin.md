---
type: Operator
title: "ForeachAsin A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachAsin 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单/多 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/docs/aclnnForeachAsin.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/README.md
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/foreach_asin_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/arch22/foreach_asin_tiling_arch22.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_infershape.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/foreach_asin.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/config/ascend910b/foreach_asin_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_asin/op_host/config/ascend910_93/foreach_asin_binary.json
    role: implementation
created_at: "2026-08-25T15:31:05Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachAsin A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachAsinGetWorkspaceSize` + `aclnnForeachAsin`，公开 ACLNN 两段式接口；固定仓仅公开无版本后缀版本。 |
| 计算语义 | 对 tensor list 中每个有效元素执行 `y[t][i] = asin(x[t][i])`。 |
| 输入输出 | `x`/`out` 均为 ND 动态 tensor list；FP16、FP32、BF16；输入可非连续并由 `AutoContiguous` 处理，输出必须连续。 |
| 执行类型 | 单 Kernel `AIV_VECTOR`；R-001 FP16、R-002 FP32、R-003 BF16 均只到 Vector `Asin`，无 Cube/AIC/MIX_CV 节点。 |
| Kernel 路线 | R-001：key 1/FP16 直接计算；R-002：key 2/FP32 直接计算；R-003：key 4/BF16→FP32→BF16。 |
| 核心资源 | GM tensor-list 描述、`dataQueue`/`outQueue` 双缓冲、BF16 专用 `float32Queue`、Vector pipe barrier；声明 32 B workspace 但 Kernel 不读取。 |
| 来源闭包 | source_closed：公开 ACLNN 文档→OpDef/自动连续化→arch22 Tiling→`ForeachCommonTilingData`→key router→`ForeachTriangle`/`KernelForeachUnary`→构建与 binary config 已闭合；ACLNN inner bridge 与 `Asin` primitive 内部为框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 list 长度为 `L`，第 `t` 个输入的存储元素数为 `N_t`。对 `0 <= t < L`、`0 <= i < N_t`，输出有效区为 `out[t][i] = asin(x[t][i])`。实数域内 `|x| <= 1` 时为反正弦主值；超出该域的具体 NaN 行为由固定 CANN `Asin` primitive 决定，primitive 内部在本仓不可见。各 tensor 独立，Host 仅把所有 `N_t` 串成逻辑一维域用于分核，不改变 tensor 的 ND 逻辑布局。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | tensor list；每个 tensor 为 ND，0~8 维 | FP16、FP32、BF16，list 内一致 | 文档允许空 tensor和非连续输入；OpDef 的 `.AutoContiguous()` 在执行器内形成连续输入边界。空指针、非法 dtype、list 内 dtype 不一致或维度大于 8 失败。 |
| `out` | tensor list；ND，0~8 维 | 与 `x` 一致 | 必须连续；公开文档要求对应输出 shape size 不小于输入。公共 infer-shape 实际把对应输出 shape 设为输入 shape，并要求动态输入数等于输出数。 |
| 属性 | N/A | N/A | 无算子属性。 |

### 入口与版本

公开接口是无版本后缀的 `aclnnForeachAsinGetWorkspaceSize`/`aclnnForeachAsin`，第一段完成校验并返回 workspace 字节数与 executor，第二段把 executor 提交到调用者给定 `aclrtStream`。固定仓没有 `_10001` 等更高公开版本，也没有手写 ACLNN inner bridge；该 bridge 由 `ACLNNTYPE aclnn` 构建框架生成，是允许的框架边界。README 和接口文档均明确支持 Atlas A2、A3；OpDef、`SUPPORT_COMPUTE_UNIT`、arch22 Tiling、Kernel target 和两个 binary config 彼此一致。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段 ACLNN 接口检查 `x`/`out` 非空、dtype 属于 FP16/FP32/BF16 且输入输出一致。OpDef 对动态输入、输出都声明 ND 与 `.AutoContiguous()`；公开约束仍明确输出不支持非连续 tensor。公共 infer-shape 要求动态输入实例数等于输出数，逐项复制 shape/dtype。arch22 `Tiling4ForeachAsinTiling` 构造 `ForeachCommonTiling`，以 `FOREACH_ASIN_OP_CODE=13` 初始化：遍历最多 `MAX_TENSOR_CONT=50` 个输入实例，确定统一 dtype、元素字节数 `d`、每 32 B 元素数 `e=32/d`、`N_t` 与总元素数 `E`。随后 `CheckOpInfo` 验证 list 数、0~8 维、输出 dtype 和对应 shape size；任何 null 或不一致返回 `GRAPH_FAILED`，阻断 launch。

### 参数、Workspace 与生命周期

`aclnnForeachAsinGetWorkspaceSize` 返回的 executor 持有连续化、Tiling 和单 Kernel launch 流程；调用者持有 executor 与 Device workspace，直至第二段提交。Tiling 将 `workspaces[0]` 固定为 `WORK_SPACE_SIZE=32` B，但 Kernel entry 将 `workspace` 丢弃并把 `userWS=nullptr` 传入实现，因此算子算法 workspace 消费为 0 B，32 B 只属于框架声明。输入/输出 GM 由调用者持有，在 stream 上 Kernel 完成前均须有效。

### 启动图与失败传播

单节点启动图：executor 在调用 stream 上启动 `foreach_asin(xListDesc, yListDesc, workspace, tiling)`，`blockDim=needCoreNum`；POD 是 `ForeachCommonTilingData`。GM `x`/`y` 是 tensor-list 描述对象，Kernel 通过描述头中的偏移定位指针表，再取第 `t` 个真实 tensor 地址。无节点间 workspace 边。第一段校验、Tiling、POD 保存或 workspace 获取失败均不产生 Kernel；第二段 launch 错误按 ACLNN status 返回。stream 提供调用前后可见性，固定仓未公开 executor 内部 launch 包装符号。

## Route 与参数设计

### 分核与资源计算

设平台 AIV 核数为 `C`，dtype 字节数为 `d`，`e=32/d`，总元素数 `E=sum(N_t)`：`blockCount=ceil(E/e)`，`needCoreNum=min(max(blockCount,1), C, 50)`。令 `q=floor(blockCount/needCoreNum)`、`r=blockCount mod needCoreNum`，前 `r` 核处理 `(q+1)*e` 个逻辑元素，其余核处理 `q*e`；`AssignDataToEachCore` 把每核区间编码为起止 tensor 索引与两个闭区间 offset。

令 `U` 为平台 UB 字节数，`P=ForeachCommonTilingData::GetDataSize()`。FP16/BF16 的 `extra=6*2*1024*8=98304` B，FP32 的 `extra=2*4*1024*8=65536` B。FP16/FP32 取 `S=floor32((U-P-extra)/4)`；BF16 先将 `(U-P-extra)` 除 10，再除 4，最后按 64 B 向下对齐得到 `S`。`S` 写入 `inputsTensorUbSize`；A2/A3 Tiling 没有在减法前显式检查 `U >= P+extra`，因此资源不足时的无符号下溢是固定源码边界。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 直接 Vector | key `1`; `ForeachTriangle<half, half, Asin<half,false>>` | 统一输入 dtype 为 `DT_FLOAT16`，`GetTilingKeyByDtypeOnly` 返回 1；router 首分支 `TILING_KEY_IS(1)` | 检查 key 2 |
| 2 | R-002 FP32 直接 Vector | key `2`; `ForeachTriangle<float, float, Asin<float,false>>` | 统一输入 dtype 为 `DT_FLOAT`，返回 2；router 第二分支 | 检查 key 4 |
| 3 | R-003 BF16 升精度 Vector | key `4`; `ForeachTriangle<bfloat16_t, float, Asin<float,false>>` | 统一输入 dtype 为 `DT_BF16`，返回 4，且编译目标不是 `__NPU_ARCH__` 3003/3113 | 未知/被编译 guard 排除时 Kernel 无分支执行 |

```text
Init 收集统一 dtype；任一不一致 -> GRAPH_FAILED
  DT_FLOAT16 -> key 1 -> R-001
  DT_FLOAT   -> key 2 -> R-002
  DT_BF16    -> key 4 -> R-003（3003/3113 编译目标排除）
  其它 dtype -> key 0；公开 OpDef/API 应先拒绝，router 无 fallback
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | FP16、ND、每项 0~8 维、A2/A3 | 单 `foreach_asin` AIV_VECTOR | `ForeachTriangle<half,half,Asin<half,false>>`→`KernelForeachUnary` | `dataQueue`/`outQueue` 均 depth 2，Vector pipe barrier 包围 `Asin` | 尾 tile 用按字节 `DataCopyPad`; 无 key fallback |
| R-002 FP32 | 2 | FP32，其余同上 | 同上 | `ForeachTriangle<float,float,Asin<float,false>>` | 同布局；`S` 与 primitive 预留不同 | 同上 |
| R-003 BF16 | 4 | BF16、A2/A3，且非 3003/3113 编译 guard | 同上 | `ForeachTriangle<bfloat16_t,float,Asin<float,false>>` | 两个 9S 双缓冲 Queue + 2S FP32 Queue；Cast/Asin/Cast 间 barrier | BF16 tile 内再按 `S/4` 个 FP32 元素分段；无 fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 32 B 分核粒度 | `e=32/d`，`blockCount=ceil(E/e)` | 使非尾核逻辑区间按 DMA block 元素数分配 | 全部 | 最后 tensor/tile 走 tail CopyPad | `GetNeedCoreNum`, `AssignDataToEachCore` |
| tensor/core 静态容量 | `MAX_TENSOR_CONT=50`, `MAX_CORE_CONT=50` | POD 数组固定长 | 全部 | 核数显式截到 50；输入超过 50 没有显式报错且初始化只遍历前 50，禁止据此扩展 | `ForeachCommonTiling::Init`, `RunBigKernelTiling` |
| Asin primitive UB 预留 | FP16/BF16 `98304` B；FP32 `65536` B | Host 为 Vector math primitive 预留临时空间，内部布局不在固定仓 | 全部 | 无资源 fallback；减法前无显式充足性检查 | `DivideUbMemory4` |
| BF16 编译 guard | `__CCE_AICORE__>=220` 且 arch 非 3003/3113 | BF16 需 FP32 Queue 与 Cast 路径 | R-003 | 不满足时 key 4 router/body 不存在 | `foreach_asin`, `KernelForeachUnary::Init/Process` |
| tail 搬运 | `dataCount % maxDataCount != 0` | 非整 tile 可能不足 32 B | 全部 | 使用单 block、精确字节数的 `DataCopyPad` | `SingleTensorProcess`, `CopyIn`, `CopyOut` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 宏生成结构的 `sizeof/alignof/offsetof` 固定仓不可见；声明为首字段 | 字节/单 buffer 基准 `S` | `FillTilingData: set_inputsTensorUbSize(S)` | `KernelForeachBase::ParseTilingData`; 推导 tile/Queue 容量 | 32 B 或 BF16 64 B 向下对齐 |
| 同上 | `tensorDataCountList[50]` | `int64_t[50]` | 精确 offset 固定仓不可见 | 元素 | `Init: storageShape.GetShapeSize()` | `Process` 计算每个 tensor 当前核有效长度 | 只初始化已发现实例；最多 50 |
| 同上 | `tensorStartList[50]` | `uint16_t[50]` | 精确 offset 固定仓不可见 | tensor 索引 | `AssignDataToEachCore` | `ParseTilingData`→循环起点 | 核数最大 50 |
| 同上 | `tensorEndList[50]` | `uint16_t[50]` | 精确 offset 固定仓不可见 | tensor 索引（含） | `AssignDataToEachCore` | `Process`→循环终点 | 含端点 |
| 同上 | `tensorStartOffsetList[50]` | `int64_t[50]` | 精确 offset 固定仓不可见 | tensor 内元素 offset | `AssignDataToEachCore` | 首 tensor 的 `cursorStart` | 含端点 |
| 同上 | `tensorEndOffsetList[50]` | `int64_t[50]` | 精确 offset 固定仓不可见 | tensor 内元素 offset | `AssignDataToEachCore` | 尾 tensor 的 `cursorEnd` | 含端点；空 tensor 可形成 -1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 FP16 | key 1 | `arch22/foreach_asin_tiling_arch22.cpp::Tiling4ForeachAsinTiling`; `foreach_tiling_class.h::RunBigKernelTiling` | `ForeachCommonTilingData`, `S`, core ranges | `op_kernel/foreach_asin.cpp::foreach_asin`, first branch | `ForeachTriangle`; `KernelForeachUnary`; `KernelForeachBase` | `op_kernel/CMakeLists.txt` A2/A3 targets；FP16 binary entry |
| R-002 FP32 | key 2 | 同上 | 同上 | 同 entry，second branch | 同上，FP32 实例 | 同 target；FP32 binary entry |
| R-003 BF16 | key 4 | 同上 | 同上 | 同 entry，guarded third branch | BF16 `InnerComputer` + FP32 `Asin` | 同 target；BF16 binary entry |

## Kernel 实现

### 公共执行流程

单 Kernel entry 解析 POD，按 key 构造 `ForeachTriangle`，传入 x/y tensor-list GM、`nullptr` user workspace 并调用 `Process`。`KernelForeachBase` 用 `GetBlockIdx` 读取本核的起止 tensor/offset；`GetTensorAddr` 依据 list 描述头中的字节偏移找到 GM 指针表。对每个相交 tensor，建立从 `cursorStart` 开始的 GM view，再以 `maxDataCount` 切 tile：`CopyIn`→Queue→`ForeachTriangle::Compute`→Queue→`CopyOut`。所有 Route 都是同一 Device 节点，没有跨 Kernel 边。

### Route R-001：FP16 直接 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_asin_def.cpp` | `ForeachAsin::ForeachAsin` | dtype 集含 `DT_FLOAT16`，A2/A3 config | 进入 arch22 Tiling |
| Host/Tiling | `foreach_tiling_class.h` | `GetTilingKeyByDtypeOnly`, `DivideUbMemory4` | `dataType==DT_FLOAT16`, `d=2` | key 1，`S=floor32((U-P-98304)/4)` |
| Router | `op_kernel/foreach_asin.cpp` | `foreach_asin` | `TILING_KEY_IS(1)` | `ForeachTriangle<half,half,Asin<half,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_asin(x, y, workspace, tiling)
  -> ForeachTriangle<half,half,Asin<half,false>>::Init
    -> KernelForeachUnary::Init -> KernelForeachBase::ParseTilingData
  -> KernelForeachUnary::Process
    -> SingleTensorProcess -> CopyIn -> ForeachTriangle::Compute
      -> InnerComputer::Compute -> Asin<half,false>
    -> CopyOut
```

##### 源码等价伪码

```text
# 伪码
core = GetBlockIdx(); 读取本核 [tensorStart..tensorEnd] 与边界 offset
for t in 本核覆盖的 tensor:
  count = end(t) - start(t) + 1
  inGM/outGM = list_ptr(t) + start(t)
  for tile in ceil(count / (S/2)):
    n = min(S/2, remain)
    input = tail ? DataCopyPad(n*2 bytes) : DataCopy(n elements)
    PipeBarrier(PIPE_V); Asin<half,false>(output, input, n); PipeBarrier(PIPE_V)
    output = tail ? DataCopyPad(n*2 bytes) : DataCopy(n elements)
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `list_ptr(t)+cursorStart+tile*S/2` | `n<=S/2` 元素 | FP16/ND 连续存储 | 调用者/CopyIn | Asin/CopyOut | 调用者；stream 完成前有效 | tail 精确 `n*2` B |
| `dataQueue` | TPipe 管理 | depth 2 × 每 buffer `S` B | FP16 UB | CopyIn | Compute DeQue | Kernel | DeQue 后 Free，下一 tile 可覆盖 |
| `outQueue` | TPipe 管理 | depth 2 × 每 buffer `S` B | FP16 UB | Compute | CopyOut | Kernel | CopyOut 后 Free |
| Asin 临时区 | primitive 内部固定仓不可见 | Host 预留 98304 B | Vector 临时 UB | `Asin` | `Asin` | 单次 primitive | 与显式 Queue 不得重叠，具体布局不可见 |
| workspace | Kernel 不寻址 | Host 声明 32 B；算法 0 B | N/A | executor | N/A | 调用者 | 不复用为算法数据 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=2 · 2S B"]:2
  outQ["outQueue · depth=2 · 2S B"]:2
  math["Asin primitive reserve · 98304 B"]:1
```

##### 字节账与复用顺序

```text
S = floor32((U - P - 98304) / 4)
UB_explicit = 2S + 2S = 4S
UB_total_bound = 4S + 98304 + P <= U
dynamic_ub = N/A
workspace_total = 32 B declared; 0 B read by Kernel
=> maxDataCount = S / 2 FP16 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` 当前 slot | `CopyIn` | `Compute::DeQue`/Asin | `FreeTensor` 后的后续 tile | EnQue→DeQue→FreeTensor | 不在 Asin 完成前释放 |
| `outQueue` 当前 slot | Asin | `CopyOut::DeQue` | GM CopyOut 后 | EnQue→DeQue→FreeTensor | 只写 `n` 个有效元素 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | TQue EnQue/DeQue | `CopyIn` | `Compute` | input slot | full tile 用元素数 DataCopy；tail 用精确字节 CopyPad |
| 计算 | `PipeBarrier<PIPE_V>` | input/Asin | Asin/Queue commit | Vector pipe | FP16 原生 `Asin<half,false>`，有效长度 `n` |
| UB→GM | outQueue | Compute | CopyOut | output slot | 仅提交 `n` 元素，padding lane 不写 GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigKernelTiling`, `DivideUbMemory4` | key、核区间、S 与 workspace |
| Kernel entry/router | `foreach_asin/op_kernel/foreach_asin.cpp` | `foreach_asin`, key 1 | key→FP16 template |
| Device body/helper | `foreach_utils/op_kernel/foreach_triangle.h`; `kernel_foreach_unary.h` | `InnerComputer<half,...>`, `SingleTensorProcess` | Queue、Asin、tail 与 barrier |

### Route R-002：FP32 直接 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_asin_def.cpp` | `ForeachAsin::ForeachAsin` | dtype 集含 `DT_FLOAT` | arch22 Tiling |
| Host/Tiling | `foreach_tiling_class.h` | `GetTilingKeyByDtypeOnly`, `DivideUbMemory4` | `DT_FLOAT`, `d=4` | key 2，`S=floor32((U-P-65536)/4)` |
| Router | `op_kernel/foreach_asin.cpp` | `foreach_asin` | `TILING_KEY_IS(2)` | `ForeachTriangle<float,float,Asin<float,false>>` |

#### Device 数据流

##### 函数级调用链

```text
foreach_asin -> ForeachTriangle<float,float,Asin<float,false>>::Init/Process
  -> KernelForeachUnary::SingleTensorProcess
    -> CopyIn -> InnerComputer::Compute -> Asin<float,false> -> CopyOut
```

##### 源码等价伪码

```text
# 伪码；core/tensor 循环同 R-001
tile_elems = S/4
for tile: n=min(tile_elems, remain)
  CopyIn FP32[n]
  barrier; Asin<float,false>(out, in, n); barrier
  CopyOut FP32[n]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `list_ptr(t)+cursorStart+tile*S/4` | `n<=S/4` | FP32/ND | 调用者/CopyIn | Asin/CopyOut | stream 生命周期 | tail `n*4` B |
| `dataQueue` | TPipe | depth 2 × `S` B | FP32 UB | CopyIn | Asin | Kernel | Queue 生命周期 |
| `outQueue` | TPipe | depth 2 × `S` B | FP32 UB | Asin | CopyOut | Kernel | Queue 生命周期 |
| Asin 临时区 | 不可见 | Host 预留 65536 B | Vector UB | primitive | primitive | 调用期 | 固定仓不可见 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["dataQueue · depth=2 · 2S B"]:2
  outQ["outQueue · depth=2 · 2S B"]:2
  math["Asin primitive reserve · 65536 B"]:1
```

##### 字节账与复用顺序

```text
S = floor32((U - P - 65536) / 4)
UB_total_bound = 4S + 65536 + P <= U
dynamic_ub = N/A
workspace_total = 32 B declared; 0 B read
=> maxDataCount = S/4 FP32 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | CopyIn | Asin | FreeTensor 后 | TQue | Asin 完成后释放 |
| `outQueue` slot | Asin | CopyOut | FreeTensor 后 | TQue | GM copy 完成后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| CopyIn/Out | TQue + DataCopy/Pad | GM/UB | Compute/GM | Queue slot | FP32 tail 精确 `n*4` B |
| Compute | PIPE_V barrier | CopyIn/Asin | Asin/CopyOut | Vector result | 全程 FP32，无显式 cast |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory4` | FP32 extra 与 S |
| Kernel entry/router | `foreach_asin.cpp` | key 2 branch | FP32 template |
| Device body/helper | `foreach_triangle.h`; `kernel_foreach_unary.h` | `InnerComputer<float,...>` | 直接 FP32 Asin 与 tail |

### Route R-003：BF16 升精度 Vector

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API/OpDef | `foreach_asin_def.cpp` | `ForeachAsin::ForeachAsin` | dtype 集含 `DT_BF16`，A2/A3 config | arch22 Tiling |
| Host/Tiling | `foreach_tiling_class.h` | `DivideUbMemory4` | `DT_BF16`, `d=2`，总可用量先 `/10` | key 4，`S=floor64(((U-P-98304)/10)/4)` |
| Router | `foreach_asin.cpp` | guarded key 4 branch | key 4 且 arch 非 3003/3113 | BF16 storage + FP32 compute template |

#### Device 数据流

##### 函数级调用链

```text
foreach_asin -> ForeachTriangle<bfloat16_t,float,Asin<float,false>>::Init/Process
  -> KernelForeachUnary::SingleTensorProcess
    -> CopyIn BF16 tile
    -> InnerComputer<bfloat16_t,float>::Compute
      -> ComputePerCast: Cast BF16->FP32 -> Asin<float,false> -> Cast FP32->BF16(CAST_RINT)
    -> CopyOut BF16 tile
```

##### 源码等价伪码

```text
# 伪码
tile_elems = 9S/2; cast_chunk = S/4
for each BF16 tile with n elements:
  CopyIn BF16[n]
  for chunk in ceil(n/cast_chunk):
    m=min(cast_chunk, remain)
    Cast(fp32_src, bf16_chunk, CAST_NONE, m)
    Asin<float,false>(fp32_dst, fp32_src, m)
    Cast(bf16_out_chunk, fp32_dst, CAST_RINT, m)
  CopyOut BF16[n]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | BF16 list pointer + element offset | `n<=9S/2` | BF16/ND | 调用者 | CopyIn/Out | stream | tail `n*2` B |
| `dataQueue` | TPipe | depth 2 × `9S` B | BF16 UB | CopyIn | Cast | Kernel | slot 释放后复用 |
| `outQueue` | TPipe | depth 2 × `9S` B | BF16 UB | final Cast | CopyOut | Kernel | slot 释放后复用 |
| `float32Queue` | 同一 LocalTensor 的两段 | depth 1 × `2S` B；src `[0,S)`，dst `[S,2S)` | FP32 UB | first Cast/Asin | Asin/final Cast | 整个 Process 持有 | 两个不重叠子区；每 chunk 覆盖 |
| Asin 临时区 | 不可见 | Host 预留 98304 B | FP32 primitive UB | Asin | Asin | primitive | 布局不可见 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  inQ["dataQueue · depth=2 · 18S B"]:2
  outQ["outQueue · depth=2 · 18S B"]:2
  fp32["float32Queue · depth=1 · 2S B"]:1
  math["Asin reserve · 98304 B"]:1
```

##### 字节账与复用顺序

```text
S = floor64(((U - P - 98304) / 10) / 4)
UB_explicit = 2*(9S) + 2*(9S) + 2S = 38S
Host budget before alignment provides 40S <= U-P-98304
UB_total_bound = 38S + 98304 + P <= U
workspace_total = 32 B declared; 0 B read
=> maxDataCount=9S/2 BF16 elements; maxCastDataCount=S/4 FP32 elements
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 src `[0,S)` | BF16→FP32 Cast | FP32 Asin | 下一 chunk Cast | PIPE_V barrier | Asin 已读完 |
| FP32 dst `[S,2S)` | FP32 Asin | FP32→BF16 Cast | 下一 chunk Asin | PIPE_V barrier | final Cast 已读完 |
| Queue slots | CopyIn/Compute | Compute/CopyOut | FreeTensor 后 | TQue | 不跨 tile 提前覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `Cast(...CAST_NONE)` + barrier | BF16 UB | Asin | FP32 src | 每 chunk 最多 `S/4` 元素 |
| Asin | `Asin<float,false>` + barrier | FP32 src | final Cast | FP32 dst | 以 FP32 计算 |
| FP32→BF16 | `Cast(...CAST_RINT)` + barrier | Asin | outQueue | BF16 output | RINT 降精度；最后 chunk 用实际 `m` |
| GM tail | DataCopyPad | Queue | GM | 有效 BF16 元素 | 只读写实际 `n` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory4` | `/10`、`/4`、64 B 对齐 |
| Kernel entry/router | `foreach_asin.cpp` | key 4 guarded branch | BF16→FP32 template |
| Device body/helper | `kernel_foreach_base.h`; `kernel_foreach_unary.h`; `foreach_triangle.h` | BF16 specializations | 9S Queue、2S FP32 Queue、chunked Cast/Asin/Cast |

## 算子特定优化

### Trick T-001：跨 tensor 的 32 B 块级均衡分核

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003
- 触发条件：输入 list 完成统一 dtype/shape 校验后，按所有 tensor 总元素数分核。
- Golden 来源：[`foreach_tiling_class.h:653-691`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h#L653-L691)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
int64_t tempPerCoreCount = blockCount / needCoreNum * static_cast<int64_t>(elementsPerBlock);
int64_t remainderCount = blockCount % needCoreNum;
```

#### 机制与设计原因

Host 先把 list 视为一个连续逻辑域，再以 32 B 元素块把余数优先分给前几个核；POD 只保存每核首尾 tensor 与 offset。这样一个核可跨越多个小 tensor，避免按 tensor 固定分核造成小 tensor 核利用不足，同时保持绝大多数核边界满足 DMA block 粒度。

#### 迁移边界

- 适用：元素级、各 tensor 之间无依赖且共享 dtype 的 tensor-list 算子。
- 前提：能用首尾 tensor/offset 表示连续逻辑区间，tensor 数和核数不超过 POD 容量。
- 失效：tensor 间存在归约/通信、不同 dtype/layout，或超过 50-tensor 固定容量时必须重做 POD/selector。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：BF16 分块升精度且复用单个 FP32 Queue

#### 触发与源码依据

- 适用 Route：R-003
- 触发条件：dtype 为 BF16 且编译目标支持该 specialization。
- Golden 来源：[`foreach_triangle.h:68-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_triangle.h#L68-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
op(float32Tensor[offset], float32Tensor, dataCount);
Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
```

#### 机制与设计原因

每次只把 `S/4` 个 BF16 元素提升为 FP32，`float32Tensor` 的前后 `S` B 分别承载输入与输出；Barrier 后原地重复覆盖这两个子区。它避免为整个 `9S/2` 元素 BF16 tile 物化 FP32 副本，同时把 `Asin` 固定在 FP32 路径并以 RINT 降精度。

#### 迁移边界

- 适用：低精度存储、FP32 math primitive、可独立分块的逐元素算子。
- 前提：两个 FP32 子区不重叠，chunk 长度不超过 `maxCastDataCount`，Cast/Asin 间维持 PIPE_V barrier。
- 失效：primitive 需要跨 chunk 状态或目标平台没有 BF16 Cast specialization 时不能沿用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 1，首分支 | key 2，次分支 | key 4，guarded 末分支 | dtype-only key |
| 模板与 dtype | half→half | float→float | bfloat16_t→float→bfloat16_t | BF16 使用 FP32 primitive |
| 循环与分核 | `e=16`，tile `S/2` | `e=8`，tile `S/4` | `e=16`，tile `9S/2` + cast chunk `S/4` | DMA 粒度与 Cast 容量 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 两个 depth-2 Queue | 同左 | 两个 9S depth-2 Queue + 2S FP32 Queue | BF16 分块升精度 |
| 内存总占用与复用 | `4S+98304+P` | `4S+65536+P` | `38S+98304+P` | Host 公式与实际 Queue |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy/Pad + half Asin | DataCopy/Pad + float Asin | DataCopy/Pad + Cast/float Asin/Cast | 全部 AIV_VECTOR |
| 同步、精度与 tail | 两次 math barrier，FP16 | 两次 math barrier，FP32 | 每 Cast/Asin 间 barrier，RINT | tail 都按有效字节 |
| fallback/失败边界 | 未知 key 无输出 | 同左 | arch guard 不满足则无实现 | 无 Device fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 支持；README/API、ascend910b config | 支持；README/API、ascend910_93 config | 支持，但使用独立 `_950` 卡 |
| Host/Launcher | arch22 `ForeachCommonTiling` | 与 A2 共用 arch22 | arch35 独立 Tiling/POD |
| Kernel/资源 | 通用 AIV Vector body，平台核数/UB 运行时获取 | 与 A2 共用 | SIMT VF，512 threads，16 MiB workspace 声明 |

## 限制与失败边界

- 公开接口允许空 tensor；A2/A3 核区间采用闭区间 offset，空 tensor 由零长度 tile 自然跳过。空 tensor-list 在 Kernel 中的完整行为固定仓不可见，不作推测。
- 公共 Tiling 仅收集 50 个 tensor，POD 也仅有 50 项；源码没有对第 51 个实例显式报错。工程复用必须在入口限制 list 长度或扩展并同步修改 Host/Device ABI。
- `totalDataCount`、各 shape size 和 offset 为 `int64_t`，源码没有加法溢出检查；POD tensor/core 索引为 `uint16_t`，当前 50 上限内安全。
- UB 公式使用无符号 `ubSizePlatForm`，减去 `P+extra` 前没有下界检查；不应在未知 UB 平台直接复用。
- 未知 TilingKey 没有显式错误/回退分支；依赖 Host dtype 校验保证只生成 1/2/4。
- ACLNN 自动生成 inner bridge、executor launch 包装、tensor-list 描述 ABI 和 `Asin` primitive 内部临时布局为固定仓不可见边界。
- 卡内 32 B 是 ForeachAsin Device 节点的 Tiling workspace 声明；若输入需要 `AutoContiguous`，第一段 ACLNN 返回的 executor 总 workspace 是否还包含框架连续化开销，固定仓不可见。
- 本次知识提取未执行编译、单元测试、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_asin/docs/aclnnForeachAsin.md` | 两段式原型、参数表、错误表 | ACLNN 公开性、dtype/shape/连续性与错误 |
| 平台/语义 | `foreach/foreach_asin/README.md` | 产品支持表、公式 | A2/A3 公开支持和 asin 语义 |
| OpDef | `foreach/foreach_asin/op_host/foreach_asin_def.cpp` | `ForeachAsin`, `OP_ADD` | dtype、ND、AutoContiguous、平台注册 |
| Host/Tiling | `foreach/foreach_asin/op_host/arch22/foreach_asin_tiling_arch22.cpp` | `Tiling4ForeachAsinTiling` | code 13 与公共 Tiling 入口 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、key、workspace、POD producer |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData` | 字段、数组容量与注册 |
| dtype selector | `foreach/foreach_utils/op_host/common_dtype.h` | `GetDataTypeSize`, `GetTilingKeyByDtypeOnly` | dtype 字节数与 key 1/2/4 |
| infer-shape | `foreach/foreach_utils/op_host/foreach_infershape.cpp` | `InferShape4ForeachCommon`, `InferDataType4ForeachCommon` | 输出实例数、shape 与 dtype 推导 |
| Kernel entry/body | `foreach/foreach_asin/op_kernel/foreach_asin.cpp` | `foreach_asin` | key router 与模板实例 |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD consumer 与 list 地址解析 |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、tile、CopyIn/Out 与 tail |
| Kernel body | `foreach/foreach_utils/op_kernel/foreach_triangle.h` | `ForeachTriangle`, `InnerComputer` | Asin、BF16 Cast、barrier |
| 构建接线 | `foreach/foreach_asin/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 Kernel target |
| 构建接线 | `foreach/foreach_asin/op_host/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`, `SUPPORT_TILING_DIR` | A2/A3 compute unit 到 arch22 Host Tiling |
| 构建接线 | `foreach/foreach_asin/op_host/config/ascend910b/foreach_asin_binary.json` | `op_list` | A2 三 dtype binary config |
| 构建接线 | `foreach/foreach_asin/op_host/config/ascend910_93/foreach_asin_binary.json` | `op_list` | A3 三 dtype binary config |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段 core-range POD | 字段顺序、50 长度、元素/字节单位、Host/Device 同步更新 | `ForeachCommonTilingData`, `ParseTilingData` |
| 调用顺序 | Init→Process→逐 tensor→逐 tile | key 分支与 Queue stage 顺序 | `foreach_asin`, `SingleTensorProcess` |
| workspace/内存 | 算法 workspace 0 B；显式 UB 由 S 推导 | 32 B Host 声明、Queue depth、BF16 9S/2S 布局、primitive 预留 | `RunBigKernelTiling`, `KernelForeachUnary::Init` |
| 同步 | TQue + PIPE_V barrier | Cast/Asin/Cast 与 FreeTensor 先后 | `InnerComputer::Compute`, `ComputePerCast` |
| 精度与尾块 | FP16/FP32 直接；BF16 FP32 升精度、RINT | tail 精确字节数，不能把 padding 写回 | `CopyIn`, `CopyOut`, BF16 `InnerComputer` |
| 失败边界 | Host 校验失败阻断 launch | 未知 key、>50 tensor、UB 下溢无 fallback | `CheckOpInfo`, `DivideUbMemory4`, router |