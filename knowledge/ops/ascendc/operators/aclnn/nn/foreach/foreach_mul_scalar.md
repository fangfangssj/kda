---
type: Operator
title: "ForeachMulScalar A2/A3 ACLNN 算子知识"
description: "说明 aclnnForeachMulScalarV2 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/docs/aclnnForeachMulScalarV2.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/op_api/foreach_mul_scalar_v2.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/foreach_mul_scalar_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_func.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_class.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/foreach_tiling_def.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_host/common_dtype.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_kernel/foreach_mul_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_base.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_kernel/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/config/ascend910b/foreach_mul_scalar_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_mul_scalar/op_host/config/ascend910_93/foreach_mul_scalar_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_templates_registry.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/common/inc/op_host/tiling_base.h
    role: implementation
created_at: "2026-08-25T15:47:32Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# ForeachMulScalar A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnForeachMulScalarV2GetWorkspaceSize` / `aclnnForeachMulScalarV2`，公开 ACLNN 两段式 V2 入口 |
| 计算语义 | 对列表中每个张量逐元素计算 `out[i][j] = x[i][j] * scalar` |
| 输入输出 | `x/out` 为长度相同的 ND TensorList，元素 dtype 为 FP16/FP32/BF16/INT32；`scalar` 是 `aclScalar`，API 层按平台和 `x` dtype 转为单元素 Tensor |
| 执行类型 | 单 Kernel：`foreach_mul_scalar`，`AIV_VECTOR` |
| Kernel 路线 | R-001 FP16 直乘；R-002 FP32 直乘；R-003 INT32 直乘；R-004 BF16→FP32 分段直乘→BF16 |
| 核心资源 | TensorList 描述符与 GM 数据、AIV UB `dataQueue/outQueue`；BF16 另有 `float32Queue`；算子 Kernel 不消费 workspace |
| 来源闭包 | `source_closed`：公开 V2 ACLNN→L0 launcher→membase Tiling/POD→dtype router→公共 foreach body→Vector `Muls` 与构建配置闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设列表长度为 `N`，第 `i` 个输入张量有效元素数为 `L_i`。对 `0 <= i < N`、`0 <= j < L_i`：

```text
out[i][j] = x[i][j] * scalar
```

各张量独立，输出 dtype 与对应输入相同。公开文档声明默认确定性实现。API 允许列表内张量为空；当 TensorList 自身长度为 0 时，第一段接口直接返回 `workspaceSize=0` 且不加入 Device 节点。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | TensorList；每个 Tensor 为 ND，0–8 维 | FP16/FP32/BF16/INT32，列表内一致 | 非空指针；与 `out` 列表长度及逐项 shape 相同；公开 V2 文档标注不支持非连续 Tensor，源码仍通过框架 `Contiguous` 边界准备内部输入 |
| `scalar` | `aclScalar`，无 Tensor shape | FP32/FP16/INT32/DOUBLE/INT64 | FP32/BF16 输入配 FP32/DOUBLE；FP16 配 FP16/DOUBLE；INT32 配 INT32/INT64；随后转换成内部单元素 Tensor |
| `out` | TensorList；逐项 shape 与 `x` 相同，ND | 与 `x` 相同 | 非空指针；列表内 dtype 一致；固定实现的 Tiling/POD 容量为 50 个 Tensor |
| `workspaceSize` | 标量输出 | `uint64_t` | 由 executor 汇总；空 TensorList 为 0 |
| `executor` | Host 对象 | `aclOpExecutor**` | 持有 `Contiguous`、标量转换和单个 AICore launcher 的生命周期 |

### 入口与版本

同一语义存在 `aclnnForeachMulScalar` 与 `aclnnForeachMulScalarV2`。V2 将公开 `scalar` 入口从单元素 `aclTensor` 改为 `aclScalar`；A2、A3 和 Ascend 950 的公开产品表均支持 V2，因此按“逐架构最新公开版本”选择 V2。`ExecForeachMulScalarV2GetWorkspaceSize` 把 `aclScalar` 转成内部 Tensor，随后仍启动算子类型 `ForeachMulScalar`，没有改写为旧公开入口。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckNotNull` 检查 `x/scalar/out`。
2. `CheckDtype` 仅接受 `DAV_2201` 或 regbase；本卡平台走 `DAV_2201`。它检查列表首项 dtype、`x/out` 列表内一致性及 scalar 组合。
3. `CheckShape` 要求两个列表等长、每个 `x` 不超过 8 维、逐项 shape 相同；`CheckFormat` 排除私有 storage format。
4. 若 `x->Size()==0`，返回空 executor，`workspaceSize=0`，不启动 Kernel。
5. 否则逐项调用框架 `l0op::Contiguous`，再由 executor 建立内部 TensorList。该函数与潜在搬运 workspace 属于固定仓允许的框架边界。
6. scalar 转换：BF16 输入转 FP32 Tensor；其余 A2/A3 路线转为 `x[0]` dtype，因此 DOUBLE/INT64 在 Host 侧窄化为 Kernel 所需 FP16/FP32/INT32。
7. `l0op::ForeachMulScalarV2` 通过 `ADD_TO_LAUNCHER_LIST_AICORE(ForeachMulScalar, OP_INPUT(x, scalar), OP_OUTPUT(out))` 加入单个 AICore 节点。

任一步失败立即返回相应 ACLNN 错误，不会加入后续节点。

### 参数、Workspace 与生命周期

`uniqueExecutor` 拥有连续化结果、内部 scalar Tensor 和 launcher；第一段以 `GetWorkspaceSize()` 返回框架汇总大小并把 owner 转交给调用方。第二段 `CommonOpExecutorRun(workspace, workspaceSize, executor, stream)` 在指定 stream 上执行。

Foreach Tiling 将节点 workspace 设为固定 32 B，但 Kernel entry 明确把 `userWS=nullptr` 传给 body，公共 unary body也不读取 workspace；因此这 32 B 是 launcher/Tiling ABI 预留，不是计算中间数据。API 总 workspace 还可能包含框架 `Contiguous` 需要的空间，其 offset 与公式在固定仓生成框架边界内不可见，不能把节点的 32 B 当成整个 ACLNN workspace。

### 启动图与失败传播

```text
aclnnForeachMulScalarV2GetWorkspaceSize
  -> [框架边界] 每个 x 的 Contiguous + aclScalar→单元素 Tensor
  -> L0 ForeachMulScalarV2
  -> node-1 ForeachMulScalar / entry foreach_mul_scalar
       blockDim = needCoreNum
       POD = ForeachCommonTilingData
       inputs = TensorList 描述符 + scalar GM
       output = TensorList 描述符
       semantic workspace = 0 B

aclnnForeachMulScalarV2
  -> CommonOpExecutorRun(node-1, stream)
```

只有一个 Device 节点，不存在 Kernel 间 workspace 边。executor/stream 保证 launcher 的提交顺序；Tiling 或 launcher 失败会阻断该节点，第二段返回执行状态。固定仓未展开通用 launcher 对 executor workspace slice 的具体 offset。

## Route 与参数设计

### 分核与资源计算

令 `s=sizeof(T)`，`B=32/s` 个元素/32 B，`L=sum_i L_i`，平台 AIV 核数为 `C`：

```text
needCoreNum = min(C, ceil(L / B), 50)
若 L == 0，则 GetNeedCoreNum 将 0 修正为 1
blockCount = ceil(L / B)
basePerCore = floor(blockCount / needCoreNum) * B
前 blockCount % needCoreNum 个 core 再多处理 B 个元素
```

每核由 `tensorStart/End` 与首尾 offset 表示跨 TensorList 的连续逻辑区间。对于 FP16/FP32/INT32：

```text
U = floor32((UB_platform - sizeof(POD)) / 2) bytes
maxDataCount = U / s
```

对于 BF16：

```text
U = floor64(((UB_platform - sizeof(POD)) / 10) / 2) bytes
data/out queue capacity each = 9U bytes
float32 queue capacity = U bytes
maxDataCount = 9U / 2 elements
maxCastDataCount = U / 4 elements
```

所有算术先发生于源码的 `uint32_t/uint64_t/int64_t` 变量；固定仓没有独立的 `UB_platform <= sizeof(POD)` 防护，故迁移时不能忽略无符号减法前提。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 FP16 直乘 | key `1`; `ForeachOneScalarBinary<half,half,MulsAdapter<half>,1>` | membase `IsCapable = !IsRegbaseSocVersion(context)` 且 `x=FP16`；注册 priority 10000 | dtype 不符继续 key router；平台为 regbase 时该 Tiling 返回 `GRAPH_PARAM_INVALID` |
| 2 | R-002 FP32 直乘 | key `2`; `ForeachOneScalarBinary<float,float,MulsAdapter<float>,1>` | 同一 membase selector 且 `x=FP32` | 继续 R-003 |
| 3 | R-003 INT32 直乘 | key `3`; `ForeachOneScalarBinary<int,int,MulsAdapter<int>,1>` | 同一 membase selector、`x=INT32`、`__CCE_AICORE__ >= 220` | 继续 R-004；编译 guard 不满足则无 body |
| 4 | R-004 BF16 升精度 | key `4`; `ForeachOneScalarBinary<bfloat16_t,float,MulsAdapter<float>,1>` | 同一 membase selector、`x=BF16`、`__CCE_AICORE__ >= 220` 且架构不是 3003/3113 | 未知 key 无计算分支，Host 支持集正常不会产生 |

```text
TilingRegistry 按 priority 从小到大：
  priority 10000 ForeachMulScalarMembaseTiling
    if !IsRegbaseSocVersion:
      key = {FP16:1, FP32:2, INT32:3, BF16:4}
    else: fallthrough
  priority 30000 ForeachRegbaseTilingUnaryScalar  # A5 卡覆盖
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 | 1 | A2/A3；FP16 TensorList；scalar 经 API 转 FP16 | 单 `foreach_mul_scalar` AIV | `ForeachOneScalarBinary<half,half,...,1>`→`InnerComputer<half,half>` | depth=1 的输入/输出 Queue；`PipeBarrier` 包围 `Muls` | 完整 tile 用 `DataCopy`，末 tile 用 `DataCopyPad`；无 dtype fallback |
| R-002 FP32 | 2 | A2/A3；FP32；scalar 经 API 转 FP32 | 同上 | `ForeachOneScalarBinary<float,float,...,1>` | 同 R-001，元素 4 B | 同上 |
| R-003 INT32 | 3 | A2/A3；INT32；scalar 经 API 转 INT32；编译核版本 >=220 | 同上 | `ForeachOneScalarBinary<int,int,...,1>` | 同 R-002；整数 `Muls` | 同上 |
| R-004 BF16 | 4 | A2/A3；BF16；scalar 内部 FP32；编译核版本 >=220 且非 3003/3113 | 同上 | `ForeachOneScalarBinary<bfloat16_t,float,...,1>`→BF16 specialization | 9U 输入 Queue + 9U 输出 Queue + U FP32 Queue；每个 cast/mul 阶段 `PipeBarrier` | GM tail 用 `DataCopyPad`；tile 内再按 `maxCastDataCount` 分段 |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| membase 平台 | `!IsRegbaseSocVersion(context_)` | POD、key 与 Kernel body属于非 regbase ABI | 全部 | priority 30000 regbase 模板 | `foreach_tiling_def.h::MEMBASE_TILING::IsCapable` |
| 50 Tensor/50 core | Init 最多扫描 50；`needCoreNum=min(...,50)` | POD 数组边界固定 | 全部 | API/Host 校验或 Tiling 容量边界；不扩容 | `ForeachCommonTiling::Init/RunBigScalarKernelTiling` |
| dtype key | `GetTilingKeyByDtypeOnly(dataType)` | router 仅实例化 1–4 | 全部 | Host dtype 校验失败或 unknown-key 空分支 | `common_dtype.h::GetTilingKeyByDtypeOnly` |
| BF16 UB 缩放 | `dataType==DT_BF16` 时先 `/10`，再 `/2`、64 B 向下对齐 | Device 实际分配 `9U+9U+U=19U` | R-004 | 无更小 Route；依赖有效 UB 前提 | `ForeachCommonTiling::DivideUbMemory7` |
| tail | `dataCount % maxDataCount != 0` | 非整 tile 的 DMA 长度未必满足普通 `DataCopy` 路径 | 全部 | `DataCopyPad` 精确字节搬运 | `KernelForeachUnary::SingleTensorProcess/CopyIn/CopyOut` |
| INT/BF16 编译 guard | `__CCE_AICORE__ >= 220`；BF16 另排除 arch 3003/3113 | 对应模板/API 可用性 | R-003/R-004 | 无 Device fallback；构建配置必须匹配 Host | `op_kernel/foreach_mul_scalar.cpp::foreach_mul_scalar` |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ForeachCommonTilingData` | `inputsTensorUbSize` | `uint64_t`/64 | 宏生成结构的 `sizeof/alignof/offsetof` 在固定仓依赖中不可见 | 字节 | `DivideUbMemory7`→`FillTilingData` | `KernelForeachBase::Init`，决定 Queue 容量与 tile 上限 | 非 BF16 32 B 对齐；BF16 64 B 对齐 |
| 同上 | `tensorDataCountList` | `int64_t[50]` | 同上 | 元素/张量 | `Init`: storage shape size | `Process`: 各 Tensor 的末 offset | 可为 0 |
| 同上 | `tensorStartList` | `uint16_t[50]` | 同上 | Tensor 索引/core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | `blockIdx < needCoreNum <= 50` |
| 同上 | `tensorEndList` | `uint16_t[50]` | 同上 | Tensor 索引/core | `AssignDataToEachCore` | `ParseTilingData[blockIdx]` | 闭区间 |
| 同上 | `tensorStartOffsetList` | `int64_t[50]` | 同上 | 元素 | `AssignDataToEachCore` | 每核首 Tensor GM base 偏移 | 首元素从 0 开始 |
| 同上 | `tensorEndOffsetList` | `int64_t[50]` | 同上 | 元素 | `AssignDataToEachCore` | 每核末 Tensor 有效末元素 | 闭区间；空域可为 -1 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | key 1 | `foreach_tiling_func.cpp::Tiling4ForeachMulScalarTiling`; `ForeachMulScalarMembaseTiling` | `ForeachCommonTilingData` | `op_kernel/foreach_mul_scalar.cpp::foreach_mul_scalar` | `KernelForeachUnary`; `ForeachOneScalarBinary<half,...>` | `op_kernel/CMakeLists.txt`; ascend910b/910_93 binary JSON |
| R-002 | key 2 | 同上 | 同上 | 同上 | `ForeachOneScalarBinary<float,...>` | 同上 |
| R-003 | key 3 | 同上 | 同上 | 同上 | `ForeachOneScalarBinary<int,...>` | 同上 |
| R-004 | key 4 | 同上 | 同上 | 同上 | BF16 `InnerComputer` specialization | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 共享单 AIV entry。router 解析 `ForeachCommonTilingData`，按 key 构造 dtype 特化的 `ForeachOneScalarBinary`；`KernelForeachBase` 将 `blockIdx` 映射到 POD 中的 Tensor 起止索引/offset。每个 core 依次遍历自己的 Tensor 片段，每个片段再按 UB tile 循环：GM→`dataQueue`→`Muls`→`outQueue`→GM。scalar 在 `Init` 中从 GM 读取一次后保存在成员标量中。entry 把 workspace 改为 `nullptr`，确认单节点没有语义 workspace。

### Route R-001：FP16 直乘

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_foreach_mul_scalar_v2.cpp` | `CheckDtype` / `Exec...` | `x=DT_FLOAT16`；scalar 为 FP16/DOUBLE | scalar 转 FP16 Tensor |
| Tiling | `foreach_tiling_class.h` | `GetTilingKeyByDtypeOnly` 调用 | `dataType=DT_FLOAT16` | key 1，membase POD |
| Router | `op_kernel/foreach_mul_scalar.cpp` | `foreach_mul_scalar` | `TILING_KEY_IS(1)` | `ForeachOneScalarBinary<half,half,...,1>` |

#### Device 数据流

##### 函数级调用链

```text
ForeachMulScalarV2 launcher
  -> foreach_mul_scalar
    -> ForeachOneScalarBinary<half,...>::Init
      -> KernelForeachUnary::Init -> KernelForeachBase::Init/ParseTilingData
    -> KernelForeachUnary::Process
      -> SingleTensorProcess -> CopyIn -> InnerComputer::Compute -> CopyOut
        -> MulsAdapter<half> -> AscendC::Muls
```

##### 源码等价伪码

```text
scalarVal = (half)scalarGM[0]
for tensor in [tensorStart, tensorEnd]:
  begin = tensor==tensorStart ? startOffset : 0
  end   = tensor==tensorEnd ? endOffset : tensorSize[tensor]-1
  bind inputGM/outputGM at tensor address + begin
  for tile in ceil((end-begin+1)/maxDataCount):
    n = min(maxDataCount, remaining)
    CopyIn n half; dataQueue.EnQue
    in = dataQueue.DeQue; out = outQueue.Alloc
    PipeBarrier(V); Muls(out, in, scalarVal, n); PipeBarrier(V)
    dataQueue.Free; outQueue.EnQue
    CopyOut n half; outQueue.Free
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `tensorAddr(i)+cursorStart+tile*maxDataCount` | `n` 元素 | half/ND | API input | CopyIn | caller；整个执行期 | full tile 普通 copy；tail 按字节 copy pad |
| scalar GM | element 0 | 1 | half | API ConvertToTensor | Init | executor | 每 core 读取一次 |
| `dataQueue` | Queue slot | `U` bytes，depth 1 | half | CopyIn | Compute | `TPipe`/tile | 消费后 `FreeTensor` |
| `outQueue` | Queue slot | `U` bytes，depth 1 | half | Compute | CopyOut | `TPipe`/tile | 写回后释放 |
| output GM | 与 input 相同逻辑 offset | `n` 元素 | half/ND | CopyOut | caller | caller | 只写有效 `n` |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | launcher 预留 32 B 不被 Kernel 读 | 不复用 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = U + U = 2U
U = floor32((UB_platform - sizeof(ForeachCommonTilingData)) / 2)
UB_total <= UB_platform - sizeof(POD)
dynamic_ub = N/A
workspace_total(node semantic) = 0 B; Tiling ABI reservation = 32 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `dataQueue` slot | CopyIn | Compute `DeQue` | 下一 tile CopyIn | `EnQue/DeQue/FreeTensor` | 必须先 Free |
| `outQueue` slot | `Muls` | CopyOut `DeQue` | 下一 tile Compute | `EnQue/DeQue/FreeTensor` | CopyOut 完成后 Free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→V | TQue `EnQue/DeQue` | CopyIn | Compute | input UB slot | tail 用 `blockLen=n*sizeof(half)` 的 `DataCopyPad` |
| V compute | `PipeBarrier<PIPE_V>` | input ready/Muls | Muls/CopyOut | Vector pipeline | half 直接乘，无 FP32 提升 |
| V→GM | TQue | Compute | CopyOut | output UB slot | 只搬 `n` 个有效元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_utils/op_host/foreach_tiling_class.h` | `RunBigScalarKernelTiling/DivideUbMemory7` | key、blockDim、U 和 POD |
| Kernel entry/router | `foreach_mul_scalar/op_kernel/foreach_mul_scalar.cpp` | key 1 branch | 映射到 half/half 模板 |
| Device body/helper | `foreach_utils/op_kernel/kernel_foreach_unary.h`; `foreach_one_scalar_binary.h` | `Process/CopyIn/Compute/CopyOut`; `InnerComputer` | 跨列表循环、Queue 生命周期、`Muls` 与 tail |

### Route R-002：FP32 直乘

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_foreach_mul_scalar_v2.cpp` | `CheckDtype` / `Exec...` | `x=DT_FLOAT`；scalar FP32/DOUBLE | scalar 转 FP32 Tensor |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_FLOAT` | key 2 |
| Router | `op_kernel/foreach_mul_scalar.cpp` | `foreach_mul_scalar` | `TILING_KEY_IS(2)` | float/float 模板 |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar -> ForeachOneScalarBinary<float,float,...>::Init/Process
  -> KernelForeachUnary::CopyIn -> InnerComputer<float,float>::Compute
    -> MulsAdapter<float> -> Muls -> CopyOut
```

##### 源码等价伪码

```text
与 R-001 相同；T=FP32、scalarVal=float，n 的 GM/UB 字节数为 4*n。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | `tensorAddr+cursorStart+tile*maxDataCount` | `n` | FP32/ND | caller/CopyOut | CopyIn/caller | caller | tail 精确 4n B |
| scalar GM | 0 | 1 | FP32 | Host conversion | Init | executor | 每 core 一次 |
| `dataQueue` / `outQueue` | Queue slot | 各 U B，depth 1 | FP32 | CopyIn/Compute | Compute/CopyOut | tile | 各 `U/4` 元素 |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | N/A | 32 B ABI 预留未消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2U; U=floor32((UB_platform-sizeof(POD))/2)
dynamic_ub=N/A; workspace_total(node semantic)=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 两个 Queue slot | CopyIn/Compute | Compute/CopyOut | 下一 tile | `FreeTensor` | 与 R-001 相同 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy/compute/copy | TQue + `PipeBarrier<PIPE_V>` | CopyIn/Muls | Compute/CopyOut | UB slots/Vector pipe | FP32 直接乘；tail 4n B |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `RunBigScalarKernelTiling` | key 2、分核、U |
| Kernel entry/router | `foreach_mul_scalar.cpp` | key 2 branch | float 模板 |
| Device body/helper | `foreach_one_scalar_binary.h` | generic `InnerComputer` | barrier 包围 FP32 `Muls` |

### Route R-003：INT32 直乘

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_foreach_mul_scalar_v2.cpp` | `CheckDtype` / `Exec...` | `x=DT_INT32`；scalar INT32/INT64 | scalar 转 INT32 Tensor |
| Tiling | `common_dtype.h` | `GetTilingKeyByDtypeOnly` | `DT_INT32` | key 3 |
| Router | `op_kernel/foreach_mul_scalar.cpp` | `foreach_mul_scalar` | `__CCE_AICORE__>=220 && key==3` | int/int 模板 |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar -> ForeachOneScalarBinary<int,int,...>::Init/Process
  -> CopyIn -> InnerComputer<int,int>::Compute -> MulsAdapter<int> -> CopyOut
```

##### 源码等价伪码

```text
与 R-002 相同；T=int32，scalarVal=(int)scalarGM[0]，Vector Muls 直接写 int32 输出。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | 同 R-002 | `n` | INT32/ND | caller/CopyOut | CopyIn/caller | caller | tail 4n B |
| scalar GM | 0 | 1 | INT32 | Host conversion | Init | executor | INT64 已在 Host 窄化 |
| `dataQueue` / `outQueue` | Queue slot | 各 U B，depth 1 | INT32 | CopyIn/Compute | Compute/CopyOut | tile | 同 R-002 |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | N/A | ABI 预留未消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 2
  inQ["dataQueue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total=2U; U=floor32((UB_platform-sizeof(POD))/2)
dynamic_ub=N/A; workspace_total(node semantic)=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 两个 Queue slot | CopyIn/Compute | Compute/CopyOut | 下一 tile | TQue + `FreeTensor` | 同 R-001 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Copy/compute/copy | TQue + Vector barrier | CopyIn/Muls | Compute/CopyOut | UB/Vector pipe | INT32 `Muls`；源码未定义额外饱和或升宽；tail 4n B |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | dtype key/UB split | key 3、U |
| Kernel entry/router | `foreach_mul_scalar.cpp` | guarded key 3 | int 模板及编译 guard |
| Device body/helper | `foreach_one_scalar_binary.h` | generic `InnerComputer<int,int>` | INT32 直乘 |

### Route R-004：BF16 分段升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| API | `op_api/aclnn_foreach_mul_scalar_v2.cpp` | `CheckDtype` / `Exec...` | `x=DT_BF16`；scalar FP32/DOUBLE | scalar 转 FP32 Tensor |
| Tiling | `DivideUbMemory7` | BF16 分支 | `(UB-POD)/10/2`，64 B 对齐 | key 4，U |
| Router | `op_kernel/foreach_mul_scalar.cpp` | key 4 branch | `__CCE_AICORE__>=220` 且非 arch 3003/3113 | BF16/float specialization |

#### Device 数据流

##### 函数级调用链

```text
foreach_mul_scalar
  -> ForeachOneScalarBinary<bfloat16_t,float,...>::Init
  -> KernelForeachUnary::Process/SingleTensorProcess
    -> CopyIn BF16 tile
    -> InnerComputer<bfloat16_t,float>::Compute
      -> ComputePerCast: Cast BF16→FP32 -> Muls FP32 in-place -> Cast FP32→BF16 RINT
    -> CopyOut BF16 tile
```

##### 源码等价伪码

```text
scalarVal = scalarFP32GM[0]
for each core-owned tensor slice:
  for tile of at most 9U/2 BF16 elements:
    CopyIn tile
    for k in ceil(tileCount/(U/4)):
      n = min(U/4, remaining)
      PipeBarrier(V)
      Cast(fp32Buf, bf16In[k], CAST_NONE, n)
      PipeBarrier(V)
      Muls(fp32Buf, fp32Buf, scalarVal, n)
      PipeBarrier(V)
      Cast(bf16Out[k], fp32Buf, CAST_RINT, n)
    CopyOut valid tileCount BF16 elements
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | 公共 tensor/tile offset | `n` | BF16/ND | caller/CopyOut | CopyIn/caller | caller | tail 2n B |
| scalar GM | 0 | 1 | FP32 | Host conversion | Init | executor | 每 core 一次 |
| `dataQueue` | Queue slot | `9U` B，depth 1 | BF16 | CopyIn | Compute | tile | 容纳大 GM tile |
| `outQueue` | Queue slot | `9U` B，depth 1 | BF16 | Compute | CopyOut | tile | 与输入相同 tile extent |
| `float32Queue` | slot 0 | `U` B，depth 1 | FP32 | Cast/Muls | Muls/Cast | 整个 Process | 同一子区原地 `Muls`，逐 cast chunk 覆盖 |
| workspace | N/A | 0 B semantic | N/A | N/A | N/A | N/A | 不消费 |

##### 分层内存布局

```mermaid
block-beta
  columns 3
  inQ["dataQueue · depth=1 · 9U bytes"]:1
  fp32["float32Queue · depth=1 · U bytes"]:1
  outQ["outQueue · depth=1 · 9U bytes"]:1
```

##### 字节账与复用顺序

```text
UB_total = 9U + U + 9U = 19U
U = floor64(((UB_platform-sizeof(POD))/10)/2)
19U <= 19/20 * (UB_platform-sizeof(POD))  # 向下取整只会继续减小左侧
maxCastDataCount=U/4; maxDataCount=9U/2
dynamic_ub=N/A; workspace_total(node semantic)=0 B
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| FP32 slot | BF16→FP32 Cast | in-place `Muls`、FP32→BF16 Cast | 下一 cast chunk | 三个 `PipeBarrier<PIPE_V>` | 降精度 Cast 完成后才覆盖 |
| BF16 Queue slots | CopyIn/Compute | Compute/CopyOut | 下一 tile | TQue 生命周期 | 各自 `FreeTensor` 后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `PipeBarrier<PIPE_V>` | Cast | FP32 Muls | FP32 slot | `CAST_NONE` |
| FP32 Muls | in-place + barrier | Muls | downcast | FP32 slot | scalar FP32，乘法 FP32 |
| FP32→BF16 | `CAST_RINT` | FP32 result | CopyOut | BF16 out slot | round-to-nearest integer mode；只转换当前 chunk |
| GM tail | `DataCopyPad` | GM/UB | UB/GM | 有效元素 | 精确 2n B，不写逻辑尾外 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `foreach_tiling_class.h` | `DivideUbMemory7` | BF16 `/10` 与 64 B 对齐 |
| Kernel entry/router | `foreach_mul_scalar.cpp` | guarded key 4 | BF16/float 模板 |
| Device body/helper | `kernel_foreach_base.h`; `kernel_foreach_unary.h`; `foreach_one_scalar_binary.h` | BF16 specializations | 9U tile、U FP32 chunk、cast/mul/cast 与 barrier |

## 算子特定优化

### Trick T-001：单 Kernel 跨 TensorList 融合

#### 触发与源码依据

- 适用 Route：R-001～R-004
- 触发条件：TensorList 非空，Tiling 已用 `tensorStart/End/Offset` 为每核分配跨张量逻辑区间。
- Golden 来源：[`foreach_utils/op_kernel/kernel_foreach_unary.h:111-130`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/kernel_foreach_unary.h#L111-L130)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
for (uint16_t i = Base::tensorStart; i <= Base::tensorEnd; i++) {
    int64_t cursorStart = 0;
    int64_t cursorEnd = Base::tensorDataCountList[i] - 1;
    int64_t dataCount = 0;
    if (i == Base::tensorStart) {
        cursorStart = Base::tensorStartOffset;
    }
    if (i == Base::tensorEnd) {
        cursorEnd = Base::tensorEndOffset;
    }

    dataCount = cursorEnd - cursorStart + 1;

    inTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, inTensorsPtr) + cursorStart);
    outTensorsGM.SetGlobalBuffer(Base::GetTensorAddr(i, outTensorsPtr) + cursorStart);
    ProcessPlusInLoop(i, cursorStart);
    SingleTensorProcess(dataCount, float32Tensor);
}
```

#### 机制与设计原因

Host 把整个列表视为按 32 B block 串联的逻辑域，再用首尾 Tensor 索引和 offset 表达每核区间。Device 在一次 launch 内切换 TensorList 数据指针，避免为每个 Tensor 单独构造/提交 Kernel；Tensor 内仍按 UB 上限切 tile。每次绑定的地址由 `GetTensorAddr` 解 TensorList 描述符得到，实际搬运量只覆盖本核的首尾有效区间。

#### 迁移边界

- 适用：相同 dtype、逐 Tensor 独立且无跨 Tensor 归约的 foreach 一元/标量算子。
- 前提：列表长度/POD 数组不超过 50；Host 与 Device 使用同一闭区间 offset 约定；TensorList 描述符 ABI 可用。
- 失效：跨 Tensor 有依赖、每 Tensor 需要不同计算图，或列表超过固定 POD 容量时必须重做分发/POD，不能直接复用。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

### Trick T-002：BF16 大搬运 tile 与小 FP32 计算片复用

#### 触发与源码依据

- 适用 Route：R-004
- 触发条件：`T=bfloat16_t`，Host 已按 BF16 路线生成 U。
- Golden 来源：[`foreach_utils/op_kernel/foreach_one_scalar_binary.h:51-75`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h#L51-L75)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
uint32_t castTimes = dataCount / maxCastDataCount;
uint32_t castTimesRemainder = dataCount % maxCastDataCount;
for (uint32_t i = 0; i < castTimes; i++) {
    ComputePerCast(dataLocal, outLocal, float32Tensor, scalarVal, maxCastDataCount, i, maxCastDataCount);
}

if (castTimesRemainder > 0) {
    ComputePerCast(
        dataLocal, outLocal, float32Tensor, scalarVal, maxCastDataCount, castTimes, castTimesRemainder);
}
```

#### 机制与设计原因

GM→UB 一次可搬入 `9U` 字节 BF16 数据，而 FP32 中间区仅 `U` 字节。Device 把大 tile 再切成 `U/4` 元素的小片，反复覆盖同一 FP32 Queue；这避免为整块 BF16 tile 同时物化等长 FP32 输入/输出。每片中 `Muls` 原地覆盖 FP32 buffer，关键实参分别来自 scalar GM、`maxCastDataCount=U/4` 和当前 remainder。

#### 迁移边界

- 适用：低精度 GM 数据需要短暂 FP32 逐元素计算，且每片计算后即可降精度输出。
- 前提：Host 的 `/10`、`/2` 与 64 B 对齐必须和 Device 的 `9U+9U+U` 分配一致；每片之间无跨片依赖。
- 失效：算法需要整 tile FP32 中间量、跨片归约或多份并存中间量时必须增大 buffer 或减小 GM tile。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例。

## 模板与实现对比

| 对比维度 | R-001 FP16 | R-002 FP32 | R-003 INT32 | R-004 BF16 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | key1 | key2 | key3+CCE>=220 | key4+CCE>=220+非3003/3113 | dtype-only key，membase priority 10000 |
| 模板与 dtype | half/half | float/float | int/int | bf16/float specialization | scalar 内部 Tensor dtype |
| 循环与分核 | 32 B block，UB tile | 同左 | 同左 | 同 core 分配；tile 内另 cast chunk | 搬运与计算容量不同 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 2×U Queue | 同左 | 同左 | 9U+9U BF16 Queue + U FP32 Queue | BF16 升精度 |
| 内存总占用与复用 | 2U | 2U | 2U | 19U，FP32 slot 分片复用 | Host UB divider |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopy(Pad)+Muls | 同左 | 同左 | DataCopy(Pad)+Cast/Muls/Cast | 全部 AIV，无其它执行族 |
| 同步、精度与 tail | Queue+2次 V barrier；half | FP32 | INT32 | 每片 3次 V barrier；RINT 降精度 | specialization |
| fallback/失败边界 | unknown key 无 body | 同左 | 编译 guard | 编译/架构 guard | Host/构建必须一致 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开 V2 支持 | 公开 V2 支持 | 公开支持，见独立 `_950` 卡 |
| Host/Launcher | `ascend910b` membase；key 1–4 | `ascend910_93` membase；key 1–4 | regbase priority 30000、key 10001–10004 |
| Kernel/资源 | 公共 membase Kernel；最多 50 AIV core | 同一源码与 POD | arch35 Kernel；最多 80 core；FP16/BF16 均用 FP32 cast buffer |

## 限制与失败边界

- 公开 V2 API：`x/out` dtype 仅 FP16/FP32/BF16/INT32，逐项 shape 相同，维数不超过 8；scalar dtype 组合必须满足平台表。
- POD 仅容纳 50 个 Tensor 和 50 个 core；公开文档给出最大 50 个输出 Tensor，不能把固定数组扩展为无界列表。
- A2/A3 FP16 直接以 half 执行 `Muls`，与 950 的 FP32 中间计算顺序不同；迁移不能只换 key。
- API executor 总 workspace 的框架 `Contiguous` 部分和 launcher slice offset固定仓不可见；仅能证明 foreach Kernel 本身不读取 workspace，Tiling 固定预留 32 B。
- 生成式 Tiling ABI 宏的确切 `sizeof/alignof/offsetof` 在固定仓依赖中不可见；字段类型、顺序和 consumer 已闭合，但不能臆造数值 offset。
- unknown key router 没有显式错误写回；正确性依赖 Host dtype/key 和构建 binary 配置一致。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `foreach/foreach_mul_scalar/docs/aclnnForeachMulScalarV2.md` | 产品表、原型、参数表 | V2 公开性、A2/A3 支持、dtype/shape |
| 入口声明/约束 | `foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.h` | 两段式声明 | C ABI |
| 入口实现 | `foreach/foreach_mul_scalar/op_host/op_api/aclnn_foreach_mul_scalar_v2.cpp` | `CheckParams`, `Exec...`, phase-2 | 校验、Contiguous、scalar 转换、executor |
| 入口实现 | `foreach/foreach_mul_scalar/op_host/op_api/foreach_mul_scalar_v2.cpp` | `ForeachMulScalarV2` | L0 单节点 launcher |
| Host 注册 | `foreach/foreach_mul_scalar/op_host/foreach_mul_scalar_def.cpp` | `ForeachMulScalar` OpDef | ascend910b/910_93/950 dtype 与平台配置 |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_func.cpp` | `Tiling4ForeachMulScalarTiling`, membase 注册 | selector、priority、Tiling parse |
| Host/Tiling | `foreach/foreach_utils/op_host/foreach_tiling_class.h` | `ForeachCommonTiling` | 校验、分核、UB、POD、workspace |
| POD | `foreach/foreach_utils/op_host/foreach_tiling_def.h` | `ForeachCommonTilingData`, `MEMBASE_TILING` | ABI 字段和 membase capable |
| dtype key | `foreach/foreach_utils/op_host/common_dtype.h` | `GetTilingKeyByDtypeOnly` | key 1–4 |
| Kernel entry/body | `foreach/foreach_mul_scalar/op_kernel/foreach_mul_scalar.cpp` | `foreach_mul_scalar`, `MulsAdapter` | AIV 标记、router、模板实例 |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_base.h` | `KernelForeachBase` | POD 解析、TensorList 地址、BF16 tile |
| Kernel body | `foreach/foreach_utils/op_kernel/kernel_foreach_unary.h` | `KernelForeachUnary` | Queue、循环、DMA、tail |
| Kernel body | `foreach/foreach_utils/op_kernel/foreach_one_scalar_binary.h` | `ForeachOneScalarBinary`, `InnerComputer` | scalar load、Muls、BF16 cast |
| 构建接线 | `foreach/foreach_mul_scalar/op_kernel/CMakeLists.txt` | `add_kernel_sources` | ascend910b/910_93 构建与 950 分源 |
| 构建接线 | `foreach/foreach_mul_scalar/op_host/config/ascend910b/foreach_mul_scalar_binary.json` | `ForeachMulScalar` op_list | A2 binary dtype 实例 |
| 构建接线 | `foreach/foreach_mul_scalar/op_host/config/ascend910_93/foreach_mul_scalar_binary.json` | `ForeachMulScalar` op_list | A3 binary dtype 实例 |
| Tiling框架 | `common/inc/op_host/tiling_templates_registry.h` | `TilingRegistry::DoTilingImpl` | priority从小到大遍历与param-invalid fallthrough |
| Tiling框架 | `common/inc/op_host/tiling_base.h` | `TilingBaseClass::DoTiling` | `IsCapable`及Tiling阶段失败传播 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 六字段描述 TensorList 分核区间 | 数组长度50、字段类型/顺序、offset 为闭区间 | `ForeachCommonTilingData` |
| 调用顺序 | 一次 launcher 内逐 Tensor、逐 tile | scalar/连续化必须先于 launcher；empty list 不 launch | `Exec...`; `KernelForeachUnary::Process` |
| workspace/内存 | Kernel semantic workspace=0；Queue按 U 分配 | 不能把预留32 B作为中间量；BF16必须按19U账 | `GetWorkspaceSize`; entry `userWS`; `InitBuffer` |
| 同步 | TQue 生命周期与 Vector barrier | Queue slot Free 前不得覆盖；BF16 cast片 barrier 不可省略 | `CopyIn/Compute/CopyOut`; `ComputePerCast` |
| 精度与尾块 | direct dtype Muls；BF16 FP32升精度/RINT；tail DataCopyPad | FP16不升精度；tail只写有效字节 | `InnerComputer`; `KernelForeachUnary::CopyOut` |
| 失败边界 | selector 不适用返回 param invalid；校验失败阻断 | unknown key 无 Device fallback；Host/build key必须一致 | `MEMBASE_TILING::IsCapable`; router |