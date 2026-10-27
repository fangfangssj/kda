---
type: Operator
title: "HardsigmoidBackward 950 ACLNN 算子知识"
description: "说明 aclnnHardsigmoidBackward 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/docs/aclnnHardsigmoidBackward.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_api/aclnn_hardsigmoid_backward.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_api/aclnn_hardsigmoid_backward.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_api/hardsigmoid_grad.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_host/hard_sigmoid_grad_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_kernel/arch35/hard_sigmoid_grad_tiling_data.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_kernel/arch35/hard_sigmoid_grad_tiling_key.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_kernel/hard_sigmoid_grad_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_kernel/arch35/hard_sigmoid_grad.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/CMakeLists.txt
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/cmake/func.cmake
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/hard_sigmoid_grad_v3/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/experimental/activation/hard_sigmoid_grad_v3/op_host/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T15:00:14Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# HardsigmoidBackward 950 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `aclnnHardsigmoidBackwardGetWorkspaceSize` + `aclnnHardsigmoidBackward`，公开 ACLNN 两段式接口；固定仓未出现更高版本入口。 |
| 计算语义 | 对每个元素，令 `z = self / 6 + 0.5`；当且仅当 `0 < z < 1` 时输出 `gradOutput / 6`，否则输出 `0`，即在 `self = -3`、`3` 两端点取零。 |
| 输入输出 | `gradOutput`、`self` 与 `out` shape 相同、ND、0–8 维，允许空 Tensor 与非连续 Tensor；950 支持 `FLOAT`、`FLOAT16`、`BFLOAT16`，两输入可经类型提升后计算。 |
| 执行类型 | 核心语义节点为单个 `AIV_VECTOR` Kernel；ACLNN executor 外围还可包含框架 `Contiguous`、`Cast`、`ViewCopy` 节点。无可达 Cube/AIC 或 `MIX_CV` body。 |
| Kernel 路线 | R-001 FP32 原生计算；R-002 FP16→FP32→FP16；R-003 BF16→FP32→BF16。 |
| 核心资源 | GM 输入/输出，三条 depth=2 的 `TQue`，`tmpBuf`、`maskBuf`，半精度路线另有 `gradFp32Buf`、`selfFp32Buf`；核心 Kernel 系统 workspace 为 0。 |
| 来源闭包 | `source_closed`：公开 ACLNN、L0 launcher、950 OpDef、arch35 Tiling/POD/key、Kernel entry/body 与 CMake 接线均在固定 Commit 闭合；外围通用张量转换是明确的框架边界。 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

设 `g = gradOutput[i]`、`x = self[i]`，公开 ACLNN 固定产生 `alpha = 1.0f / 6.0f` 与 `beta = 0.5f`。Kernel 按严格比较实现：

```text
z = alpha * x + beta
out[i] = alpha * g,  if z > 0 and z < 1
         0,          otherwise
```

这是 `Hardsigmoid` 的逐元素反向传播。两个比较分别生成并立即消费 bit mask；相等于边界时比较为假。FP16/BF16 路线先将两输入扩展为 FP32，完成 `z`、缩放和两次筛选后再降精度；FP32 路线直接计算。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `gradOutput` | 与 `self/out` 完全相同；ND；0–8 维 | `FLOAT`、`FLOAT16`、`BFLOAT16` | 上游梯度；支持空 Tensor 与非连续 Tensor。 |
| `self` | 与 `gradOutput/out` 完全相同；ND；0–8 维 | `FLOAT`、`FLOAT16`、`BFLOAT16` | 前向输入；支持空 Tensor 与非连续 Tensor。 |
| `out` | 与两输入完全相同；ND；0–8 维 | `FLOAT`、`FLOAT16`、`BFLOAT16` | 必须可由两输入的 `PromoteType` 转换；通过 `ViewCopy` 写入用户视图。 |
| `alpha` | 标量属性，非 ACLNN 显式参数 | FP32 | ACLNN 固定为 `1.0f / 6.0f`；Tiling 还检查有限且非零。 |
| `beta` | 标量属性，非 ACLNN 显式参数 | FP32 | ACLNN 固定为 `0.5f`；Tiling 检查有限。 |

### 入口与版本

公开头文件声明唯一两段式版本。第一段创建 executor、校验参数并构建计算图，第二段以用户提供的 workspace 和 stream 调用 `CommonOpExecutorRun`。950 由 `IsRegbase()` 分支获得 `{FLOAT, BFLOAT16, FLOAT16}` 支持表。稳定目录的 OpDef 与 CMake 仅接线 `ascend950/arch35`。候选 `experimental/activation/hard_sigmoid_grad_v3` 虽公开声明 A2/A3，并带独立 API、Host/Tiling 与 Kernel，但仓库根选项 `ENABLE_EXPERIMENTAL` 默认是 `OFF`，`add_category_subdirectory` 在默认构建中只扫描稳定目录；因此它不是本次采用的活动实现。本卡只发布稳定链路闭合的 950。

## Host/Launcher 执行流程

### 参数校验与前处理

第一段接口按以下顺序执行：

1. `OP_CHECK_COMM_INPUT` 校验 `workspaceSize` 与 `executor` 出参；创建 executor。
2. `CheckNotNull` 校验 `gradOutput/self/out`；失败返回 `ACLNN_ERR_PARAM_NULLPTR`。
3. `CheckDtypeValid` 逐个校验两输入 dtype，计算 `PromoteType(gradOutput, self)`，再校验该类型可转换到 `out`；失败返回 `ACLNN_ERR_PARAM_INVALID`。
4. `CheckShape` 校验三者不超过框架最大维数并且 shape 完全相等；公开文档把最大维数具体化为 8。
5. 任一 Tensor 为空时返回 `workspaceSize=0` 和空计算 executor，不添加核心 launcher。
6. 非空时依次建立 `Contiguous(gradOutput)`→`Cast(promoteType)`、`Contiguous(self)`→`Cast(promoteType)`，再添加 `HardSigmoidGrad`，最后 `Cast(out dtype)`→`ViewCopy(out)`。

`Contiguous/Cast/ViewCopy` 的内部 Kernel 是仓外框架边界；算子自有的核心 launcher、Tiling、POD 与 Kernel 已继续闭合。

### 参数、Workspace 与生命周期

`l0op::HardSigmoidGrad` 以 `x->GetViewShape()` 和提升后的 dtype 分配中间输出，并用 `ADD_TO_LAUNCHER_LIST_AICORE` 添加核心节点。Tiling 为该节点写 `WS_SYS_SIZE=0`；`hard_sigmoid_grad` 的 `workspace` 形参未读取。ACLNN 返回的总 `workspaceSize` 则由 executor 对外围转换及中间 Tensor 统一计算，固定范围没有展开该框架分配公式，调用者必须使用第一段返回值原样分配并保持至第二段 stream 工作完成。

核心节点的 GM 输入、中间输出和最终 `ViewCopy` 目标均由 executor 持有；第二段在同一 `stream` 提交整条列表。源码对 launcher 添加失败逐节点检查并立即返回，不会继续构建后续节点。

### 启动图与失败传播

```text
gradOutput -> Contiguous -> Cast(promoteType) --+
                                                 +-> HardSigmoidGrad[AIV_VECTOR] -> Cast(out dtype) -> ViewCopy(out)
self       -> Contiguous -> Cast(promoteType) --+
```

| 节点/边 | producer → consumer | 介质与范围 | ordering / owner / lifetime | 失败传播 |
|---|---|---|---|---|
| 输入规范化 | 用户 Tensor → `Contiguous` | executor 中间 Tensor；逻辑 shape 不变 | executor 按添加顺序绑定同一执行流程；executor 持有 | 返回 `nullptr`，第一段立即报内部错误 |
| 类型提升 | `Contiguous` → `Cast(promoteType)` | 同 shape、提升 dtype | executor 持有至消费者完成 | 返回 `nullptr`，不添加核心节点 |
| 核心节点 | 两个提升后 Tensor → `hard_sigmoid_grad` | 输入/输出 GM，元素数 `totalLength`；无节点 workspace slice | `ADD_TO_LAUNCHER_LIST_AICORE` 添加，第二段在用户 stream 执行 | launcher 添加失败返回错误；Kernel runtime 失败由 executor 返回 |
| 输出适配 | 核心中间输出 → `Cast(out dtype)` → `ViewCopy(out)` | executor 中间 Tensor 与用户输出视图 | 同一 executor/stream；完成前不得释放 workspace、executor、输入输出 | 任一返回空，第一段立即失败 |

核心语义为单 Kernel，不存在 Kernel 间 workspace 边、event 或 atomic 交接。

## Route 与参数设计

### 分核与资源计算

Tiling 从两输入 storage shape 取元素数并要求相等，标量 shape 被归一为 `{1}`。非空时：

```text
actualCoreNum = totalLength < 1024 ? 1 : platformAivCoreNum
blockFactor   = ceil(totalLength / actualCoreNum)        # 元素/core
usedCoreNum   = ceil(totalLength / blockFactor)          # blockDim
alignBytes    = max(GetUbBlockSize(context), 256)        # 字节
alignElem     = alignBytes / 4                            # FP32 等价元素
bufferNum     = dtype in {FP16,BF16} ? 7 : 8
ubFactor      = floor_align(floor((ubSize / 4) / bufferNum), alignElem)
```

`ubFactor<=0`、平台 AIV 核数/UB 大小为零或 `GetUbBlockSize()==0` 都失败。每个 block 的有效长度为 `min(blockFactor, totalLength-blockFactor*blockIdx)`；每核 tile 数为 `ceil(blockLength/ubFactor)`，最后 tile 用真实 `currentNum`。没有单独 former/tail POD。

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 0 | 空 Tensor 快返 | 无核心 key | API 校验后任一 Tensor `IsEmpty()` | `workspaceSize=0`，成功返回，不启动 Kernel |
| 1 | R-001 FP32 原生 | `schMode=0` / `HardSigmoidGrad<float>` | 非空且提升后核心输入 dtype 为 `DT_FLOAT` | 其它 dtype 继续按下两行选择 |
| 2 | R-002 FP16 升精度 | `schMode=1` / `HardSigmoidGrad<half>` | 非空且提升后核心输入 dtype 为 `DT_FLOAT16` | 其它 dtype 继续 |
| 3 | R-003 BF16 升精度 | `schMode=2` / `HardSigmoidGrad<bfloat16_t>` | 非空且提升后核心输入 dtype 为 `DT_BF16` | ACLNN dtype 校验拒绝其它类型；Tiling 源码对意外其它 dtype 会落入 key 0 |

```text
ACLNN: invalid/null/shape -> error
       empty             -> success, no launch
       nonempty          -> Contiguous + Cast(promoteType)
Tiling: dtype == BF16    -> schMode 2
        dtype == FP16    -> schMode 1
        otherwise        -> schMode 0
Kernel entry: 0 -> float; 1 -> half; 2 -> bfloat16_t; unknown -> no body
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 FP32 原生 | 0 | 950、非空、提升 dtype `FLOAT`、shape 校验通过、`ubFactor>0` | 单 `hard_sigmoid_grad<0>` AIV_VECTOR | `HardSigmoidGrad<float>`；无 Kernel 内 Cast | 3×depth-2 FP32 Queue + FP32 tmp + mask；Queue 同步 | core remainder + tile `currentNum`；无 Device fallback |
| R-002 FP16 升精度 | 1 | 950、非空、提升 dtype `FLOAT16`、其余资源 guard 通过 | 单 `hard_sigmoid_grad<1>` AIV_VECTOR | `HardSigmoidGrad<half>`；`CAST_NONE`→FP32→`CAST_RINT` | 3×depth-2 FP16 Queue + 3×FP32 TBuf + mask；`selfFp32Buf` 原地复用 | 同上；无 Device fallback |
| R-003 BF16 升精度 | 2 | 950、非空、提升 dtype `BFLOAT16`、其余资源 guard 通过 | 单 `hard_sigmoid_grad<2>` AIV_VECTOR | `HardSigmoidGrad<bfloat16_t>`；`CAST_NONE`→FP32→`CAST_ROUND` | 3×depth-2 BF16 Queue + 3×FP32 TBuf + mask；`selfFp32Buf` 原地复用 | 同上；无 Device fallback |

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 小任务单核 | `totalLength < 1024` | 避免把小逐元素域切到全部 AIV 核 | 全部 Route | `>=1024` 使用平台 AIV 核数 | `FillTilingData` |
| Compare 对齐 | `alignBytes=max(ubBlockSize,256)`，`ubFactor` 按 `alignBytes/4` 元素下对齐 | `CompareScalar` 读取数按 256B/64 FP32 元素上对齐，必须留在 `tmpBuf` | 全部 Route | `ubFactor<=0` 时 Tiling 失败 | `FillTilingData`、`HardSigmoidGrad::Compute` |
| UB 槽预算 | FP32 `bufferNum=8`；FP16/BF16 `bufferNum=7`，`typeSize=4` | 分别覆盖 7 或 6 个 FP32 等价槽及 mask/tail 余量 | 分别影响 R-001 与 R-002/R-003 | 无更小 tile fallback；失败 | `GetBufferConfig` |
| dtype specialization | `BF16→2`，`FP16→1`，其余→0 | 决定模板类型、Cast 与降精度 round mode | 全部 Route | 公开 ACLNN 先拒绝不支持 dtype；未知 key entry 不执行 body | `FillTilingData`、`hard_sigmoid_grad` |
| 空域 | ACLNN `IsEmpty()` | 避免生成零元素核心 launcher | 全部 Route 之前 | 成功快返 | `aclnnHardsigmoidBackwardGetWorkspaceSize` |
| 属性有限性 | `isfinite(alpha) && alpha!=0 && isfinite(beta)` | 防止退化或 NaN/Inf 进入算术与比较 | 全部 Route | Tiling `GRAPH_FAILED` | `GetShapeAttrsInfo` |

### POD/ABI

`HardSigmoidGradTilingData` 仅由三个连续 `int64_t` 和两个连续 `float` 构成且无 packing 属性；按该目标的自然布局，整体 `sizeof=32`、`alignof=8`，字段 offset 依次为 `0/8/16/24/28`。固定仓没有额外静态断言，复用时仍应让 Host 与 Device 直接共享此定义。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `HardSigmoidGradTilingData` | `totalLength` | `int64_t`/64 | `sizeof=32, alignof=8, offsetof=0` | 元素 | `tiling->totalLength=totalLength` | `ComputePerCoreLength` 计算 remainder | 非空为正；空 Tiling helper 清零但 ACLNN 空域不启动核心 |
| 同上 | `blockFactor` | `int64_t`/64 | `offsetof=8` | 元素/core | `CeilDiv(totalLength,actualCoreNum)` | GM core base 与每核长度 | 非空为正 |
| 同上 | `ubFactor` | `int64_t`/64 | `offsetof=16` | 元素/tile | UB 公式下对齐后写入 | Queue/TBuf 分配、tile loop 和 GM tile offset | 必须 `>0`；至少按 256B FP32 等价粒度对齐 |
| 同上 | `alpha` | `float`/32 | `offsetof=24` | 无量纲 | ACLNN attr `1.0f/6.0f`→`GetShapeAttrsInfo`→字段 | `Muls` 计算 `z` 和梯度 | 有限且非零 |
| 同上 | `beta` | `float`/32 | `offsetof=28` | 无量纲 | ACLNN attr `0.5f`→`GetShapeAttrsInfo`→字段 | `Adds` 计算 `z` | 有限 |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001 | `schMode=0` | `hard_sigmoid_grad_tiling_arch35.cpp::FillTilingData` | `HardSigmoidGradTilingData` | `hard_sigmoid_grad_apt.cpp::hard_sigmoid_grad<0>` | `hard_sigmoid_grad.h::HardSigmoidGrad<float>` | `hard_sigmoid_grad_def.cpp` + `CMakeLists.txt` 的 `ascend950/arch35` |
| R-002 | `schMode=1` | 同上 | 同上 | `hard_sigmoid_grad<1>` | `HardSigmoidGrad<half>`、`CastDownRound<half>` | 同上 |
| R-003 | `schMode=2` | 同上 | 同上 | `hard_sigmoid_grad<2>` | `HardSigmoidGrad<bfloat16_t>`、`CastDownRound<bfloat16_t>` | 同上 |

## Kernel 实现

### 公共执行流程

三个 Route 共享单节点启动图。entry 用 `GET_TILING_DATA_WITH_STRUCT` 把 GM tiling 读为局部 POD，再按编译期 `schMode` 实例化模板。`Init` 计算当前 block 的连续切片，绑定三路 GM，按 route 分配 UB；`Process` 顺序执行每个 tile 的 `CopyIn→Compute→CopyOut`。

两路 `DataCopyPad` 的参数均为 `blockCount=1`、`blockLen=currentNum*sizeof(T)` 字节、两个 stride 为 0，且禁用输入 padding；输出同样单 block、零 stride，只提交 `currentNum` 个有效元素。Queue 的 `EnQue/DeQue/FreeTensor` 建立 GM↔Vector 阶段的生产消费与复用顺序。核心节点没有 dynamic UB、NDDMA、通信、atomic、跨核归约或额外 workspace。

### Route R-001：FP32 原生计算

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardsigmoid_backward.cpp` | `GetDtypeSupportList` / `CheckDtypeValid` | 950 `IsRegbase()` 支持 FP32，提升类型可转为 out | 建立 FP32 核心输入 |
| Tiling | `op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp` | `FillTilingData` | dtype 非 BF16 且非 FP16 | `schMode=0` |
| Router | `op_kernel/hard_sigmoid_grad_apt.cpp` | `hard_sigmoid_grad<schMode>` | `schMode==0` | `HardSigmoidGrad<float>` |

#### Device 数据流

##### 函数级调用链

```text
ADD_TO_LAUNCHER_LIST_AICORE(HardSigmoidGrad)
  -> hard_sigmoid_grad<0>(gradOutput, self, gradInput, workspace_unused, tiling)
    -> HardSigmoidGrad<float>::Init
    -> HardSigmoidGrad<float>::Process
      -> RunTileLoop
        -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
# 源码等价伪码
b = GetBlockIdx()
L = min(blockFactor, totalLength - blockFactor*b)
bind GM[b*blockFactor : b*blockFactor+L]
for i in [0, ceil(L/U)):
    n = min(U, L-i*U)
    copy g[i*U : +n], x[i*U : +n] GM->VECIN
    a = align_up(n, 64)
    tmp[0:a] = -1
    tmp[0:n] = alpha*x[0:n] + beta
    result[0:n] = alpha*g[0:n]
    result[0:n] = select(tmp>0, result, 0)
    result[0:n] = select(tmp<1, result, 0)
    copy result[0:n] VECOUT->GM[i*U : +n]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三路 GM | `b*blockFactor+i*U` 元素 | `currentNum` 有效，core extent `L` | FP32/线性 ND | executor/CopyOut | CopyIn/用户 | executor；stream 完成前有效 | core 区间不重叠；tail 精确 |
| `gradOutputQueue` | Queue slot | `2*U*4` B | FP32 | CopyIn | Compute | `TPipe`，Compute 后释放 | depth=2 |
| `selfQueue` | Queue slot | `2*U*4` B | FP32 | CopyIn | Compute | 同上 | depth=2 |
| `gradInputQueue` | Queue slot | `2*U*4` B | FP32 | Compute | CopyOut | CopyOut 后释放 | depth=2 |
| `tmpBuf` | 独立 TBuf | `U*4` B | FP32 | Duplicate/Muls/Adds | CompareScalar | Kernel core 全生命周期 | 尾区预填后有效区覆盖 |
| `maskBuf` | 独立 TBuf | `M=max(32,align_up(ceil(U/8),32))` B | uint8 bitmask | CompareScalar | 紧随其后的 Select | 两次比较间原地覆盖 | 32B 对齐 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  gq["gradOutputQueue · depth=2 · 8U B"]:1
  xq["selfQueue · depth=2 · 8U B"]:1
  yq["gradInputQueue · depth=2 · 8U B"]:1
  tmp["tmpBuf · 4U B"]:1
  mask["maskBuf · M B"]:4
```

##### 字节账与复用顺序

```text
U = ubFactor, M = max(32, align_up(ceil(U/8), 32))
UB_total = 8U + 8U + 8U + 4U + M = 28U + M
U 按至少 64 元素对齐，因此 M <= 4U；UB_total <= 32U
U <= floor((ubSize/4)/8)，故 UB_total <= ubSize
dynamic_ub = N/A
workspace_total(core kernel) = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `maskBuf` | 第一次 `CompareScalar(GT)` | 第一次 `Select` | 第二次 `CompareScalar(LT)` | 源码顺序；Select 在覆盖前完成 | 不交换两次 Compare/Select 顺序 |
| Queue slot | CopyIn/Compute | Compute/CopyOut | 后续 tile | `EnQue→DeQue→FreeTensor` | Free 后才能复用 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | `DataCopyPad` + input Queue | CopyIn | Compute `DeQue` | 两路输入 slot | `blockLen=n*4` B，无 padding |
| 算术 | Vector `Duplicate/Muls/Adds` | 当前 tile | Compare | `tmp/result` | 全 FP32；`a=align_up(n,64)`，仅前 n 元素输出 |
| 两级筛选 | `CompareScalar` + `Select` | mask | 紧随 Select | mask 与 result | strict GT/LT；Compare 读 a，Select 只写 n |
| UB→GM | output Queue + `DataCopyPad` | Compute `EnQue` | CopyOut `DeQue` | result slot | 仅复制 n 个元素 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp` | `GetBufferConfig`、`FillTilingData` | FP32 取 8 槽、key 0、block/tile 公式 |
| Kernel entry/router | `op_kernel/hard_sigmoid_grad_apt.cpp` | `hard_sigmoid_grad<0>` | key 0 映射 `HardSigmoidGrad<float>` |
| Device body/helper | `op_kernel/arch35/hard_sigmoid_grad.h` | `Init/RunTileLoop/CopyIn/Compute/CopyOut` | FP32 原生 Vector 管线、Queue 与 tail |

### Route R-002：FP16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardsigmoid_backward.cpp` | `GetDtypeSupportList` / `PromoteType` | 950 支持 FP16，提升结果为 FP16 | 建立 FP16 核心输入 |
| Tiling | `op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp` | `FillTilingData` | `dataType==DT_FLOAT16` | `schMode=1` |
| Router | `op_kernel/hard_sigmoid_grad_apt.cpp` | `hard_sigmoid_grad<schMode>` | `schMode==1` | `HardSigmoidGrad<half>` |

#### Device 数据流

##### 函数级调用链

```text
HardSigmoidGrad launcher -> hard_sigmoid_grad<1>
  -> HardSigmoidGrad<half>::Init
  -> Process -> RunTileLoop
    -> CopyIn
    -> Compute: Cast(CAST_NONE) -> FP32 Muls/Adds -> Compare/Select x2 -> Cast(CAST_RINT)
    -> CopyOut
```

##### 源码等价伪码

```text
# core/tile 地址与 R-001 相同
copy half g,x GM->UB; a=align_up(n,64); tmp[0:a]=-1
grad32[0:n]=cast_none(g); self32[0:n]=cast_none(x)
tmp[0:n]=alpha*self32+beta
self32[0:n]=alpha*grad32                 # self32 原地转作结果
self32=select(tmp>0,self32,0); self32=select(tmp<1,self32,0)
result_half=cast_rint(self32); copy n 个 half 到 GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| 三路 GM | `b*blockFactor+i*U` | n 有效 | FP16/线性 ND | executor/CopyOut | CopyIn/用户 | executor | tail 精确 |
| 三条 Queue | 各自 slot | 每条 `2*U*2=4U` B | FP16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth=2 |
| `gradFp32Buf` | TBuf | `4U` B | FP32 | Cast grad | Muls | core 生命周期 | 结果缩放后可复用但源码未复用 |
| `selfFp32Buf` | TBuf | `4U` B | FP32 | Cast self，随后 Muls 覆盖 | tmp Muls；两次 Select；downcast | core 生命周期 | 同一物理 buffer 从 self 转作 result |
| `tmpBuf` | TBuf | `4U` B | FP32 | Duplicate/Muls/Adds | CompareScalar | core 生命周期 | tail 预填 |
| `maskBuf` | TBuf | `M` B | bitmask | 两次 Compare | 各自 Select | core 生命周期 | 两次比较间覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  qs["3 Queues · each depth=2 · total 12U B"]:2
  grad32["gradFp32Buf · 4U B"]:1
  self32["selfFp32Buf · 4U B"]:1
  tmp["tmpBuf · 4U B"]:2
  mask["maskBuf · M B"]:2
```

##### 字节账与复用顺序

```text
UB_total = 3*(2*U*2) + 3*(U*4) + M = 24U + M
M <= 4U，故 UB_total <= 28U
U <= floor((ubSize/4)/7)，故 UB_total <= ubSize
dynamic_ub = N/A; workspace_total(core kernel) = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `selfFp32Buf` | `Cast(self)` | `Muls(tmp,selfFp32)` | `Muls(selfFp32,gradFp32,alpha)` | 同一 Compute 内源码顺序 | 必须先完成 `tmp` |
| `maskBuf` | GT Compare | GT Select | LT Compare | 源码顺序 | 每次 mask 必须先消费 |
| Queue slot | CopyIn/Compute | Compute/CopyOut | 后续 tile | Queue 生命周期 | Free 后覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入 | `DataCopyPad` + Queue | CopyIn | Compute | half 输入 | n×2 B |
| 升精度 | `Cast(...CAST_NONE,n)` | half Queue | FP32 算术 | 两个 FP32 TBuf | widening，只处理 n |
| 比较 | `Duplicate(a)`、Compare(a)、Select(n) | tmp/mask | Select | padded compare window | a 为 64 元素倍数，padding 值 -1 不进入有效输出 |
| 降精度 | `CastDownRound<half>::CAST_RINT` | FP32 result | output Queue | result | n 个元素，round-to-nearest-even 语义由枚举锁定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp` | `GetBufferConfig`、`FillTilingData` | 7 槽预算、key 1 |
| Kernel entry/router | `op_kernel/hard_sigmoid_grad_apt.cpp` | `hard_sigmoid_grad<1>` | 映射 half 模板 |
| Device body/helper | `op_kernel/arch35/hard_sigmoid_grad.h` | `HardSigmoidGrad<half>`、`CastDownRound<half>` | FP32 中间计算、buffer 复用、CAST_RINT |

### Route R-003：BF16 升精度

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| ACLNN | `op_api/aclnn_hardsigmoid_backward.cpp` | `GetDtypeSupportList` / `PromoteType` | 950 支持 BF16，提升结果为 BF16 | 建立 BF16 核心输入 |
| Tiling | `op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp` | `FillTilingData` | `dataType==DT_BF16` | `schMode=2` |
| Router | `op_kernel/hard_sigmoid_grad_apt.cpp` | `hard_sigmoid_grad<schMode>` | `schMode==2` | `HardSigmoidGrad<bfloat16_t>` |

#### Device 数据流

##### 函数级调用链

```text
HardSigmoidGrad launcher -> hard_sigmoid_grad<2>
  -> HardSigmoidGrad<bfloat16_t>::Init
  -> Process -> RunTileLoop
    -> CopyIn
    -> Compute: Cast(CAST_NONE) -> FP32 Muls/Adds -> Compare/Select x2 -> Cast(CAST_ROUND)
    -> CopyOut
```

##### 源码等价伪码

```text
# 与 R-002 相同的 core/tile、FP32 中间计算和 selfFp32 原地复用
result_bf16 = Cast(result_fp32, CAST_ROUND, n)
copy result_bf16[0:n] to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM 与三条 Queue | 与 R-002 相同 | GM n；Queues 合计 12U B | BF16 | CopyIn/Compute/CopyOut | 各下一阶段 | executor/TPipe | depth=2、tail 精确 |
| 三个 FP32 TBuf | 独立区域 | 各 `4U` B | FP32 | Cast/算术 | 算术/downcast | core 生命周期 | `selfFp32Buf` 原地复用 |
| `maskBuf` | 独立区域 | `M` B | bitmask | Compare | Select | core 生命周期 | 32B 对齐、两次覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  qs["3 Queues · each depth=2 · total 12U B"]:2
  fp32["3 FP32 TBuf · total 12U B"]:1
  mask["maskBuf · M B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 24U + M <= 28U <= ubSize
dynamic_ub = N/A
workspace_total(core kernel) = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `selfFp32Buf` | self widening Cast | tmp Muls | grad scaling Muls | 单一 Compute 顺序 | tmp 先完成 |
| `maskBuf` | 当前 Compare | 当前 Select | 下一 Compare/tile | 源码顺序 | Select 先消费 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入/搬出 | Queue + `DataCopyPad` | CopyIn/Compute | Compute/CopyOut | BF16 tile | 只传 n×2 B |
| 计算 | widening Cast + FP32 Vector | BF16 input | downcast | FP32 intermediates | strict boundary 在 FP32 中判断 |
| 降精度 | `CastDownRound<bfloat16_t>::CAST_ROUND` | FP32 result | output Queue | BF16 result | 仅 n 元素 |
| 尾块 | `Duplicate(a)` + Compare(a)/Select(n) | tmp | result | padded lanes | 无效 lane 不写 GM |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp` | `GetBufferConfig`、`FillTilingData` | 7 槽预算、key 2 |
| Kernel entry/router | `op_kernel/hard_sigmoid_grad_apt.cpp` | `hard_sigmoid_grad<2>` | 映射 BF16 模板 |
| Device body/helper | `op_kernel/arch35/hard_sigmoid_grad.h` | `HardSigmoidGrad<bfloat16_t>`、`CastDownRound<bfloat16_t>` | FP32 中间计算与 CAST_ROUND |

## 算子特定优化

### Trick T-001：固定 `1.0f/6.0f` 保持边界语义

#### 触发与源码依据

- 适用 Route：R-001、R-002、R-003。
- 触发条件：公开 ACLNN 构建 `HardSigmoidGrad` 核心节点。
- Golden 来源：[`op_api/aclnn_hardsigmoid_backward.cpp:130-136`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_api/aclnn_hardsigmoid_backward.cpp#L130-136)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
float alpha = 1.0f / 6.0f;
float beta = 0.5f;
auto hardsigmoidGradOut =
    l0op::HardSigmoidGrad(gradOutputContiguousCasted, selfContiguousCasted, alpha, beta, uniqueExecutor.get());
```

#### 机制与设计原因

常量表达式进入 FP32 attr 与 POD，Kernel 用它先算 `alpha*self+beta`，再做严格 `GT/LT`。这避免用截短十进制字面量改变 `self=-3` 附近的分支；机制改变的是边界精度语义，而不是宣称实测性能收益。

#### 迁移边界

- 适用：需要与该 Hardsigmoid 严格端点规则一致的 backward。
- 前提：Host 常量、Tiling POD 和 Kernel FP32 算术顺序同时保持。
- 失效：若前向使用不同 `alpha/beta` 或不同边界定义，必须成对修改公式与比较，不可只复制常量。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

源码注释明确指出用 `0.16666666f` 可能令 `x=-3` 落入非零 mask；固定仓未提供独立运行反例。

### Trick T-002：对齐比较窗口预填哨兵

#### 触发与源码依据

- 适用 Route：全部 Route 的最后 tile，亦统一用于完整 tile。
- 触发条件：`alignedCount=align_up(currentNum,64)` 可能大于有效 `currentNum`。
- Golden 来源：[`op_kernel/arch35/hard_sigmoid_grad.h:203-219`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_kernel/arch35/hard_sigmoid_grad.h#L203-219)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t alignedCount = (currentNum + COMPARE_ALIGN - 1) / COMPARE_ALIGN * COMPARE_ALIGN;
if constexpr (kCastNeeded) {
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    Duplicate(tmp, -1.0f, alignedCount);
```

#### 机制与设计原因

Host 先把 `ubFactor` 按至少 256B 的 FP32 等价元素数对齐，因此 Device 的上取整窗口不越过 `tmpBuf`。Device 将整个窗口预填为区间 `(0,1)` 外的 `-1`，再仅覆盖前 `currentNum` 个真实值，使比较尾 bit 确定；`Select` 仍只处理有效元素。这一组合闭合了对齐读与逻辑 tail。

#### 迁移边界

- 适用：比较 primitive 的访问粒度大于逻辑尾长的逐元素筛选。
- 前提：Host tile 对齐、UB 实际容量、Device `COMPARE_ALIGN` 三者一致，且哨兵落在两个比较构成的有效域之外。
- 失效：删去 Host 对齐会造成窗口越界风险；删去预填会让 padding mask 依赖旧 UB 内容。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不得仅把 `CompareScalar` count 上取整而既不扩容也不初始化尾区；源码以 Host 对齐和 Device `Duplicate` 同时规避该路径。

### Trick T-003：编译期消除 Cast buffer，并在半精度路线原地复用

#### 触发与源码依据

- 适用 Route：R-001 条件消除；R-002/R-003 原地复用。
- 触发条件：`kCastNeeded = !is_same<T,float>`。
- Golden 来源：[`op_kernel/arch35/hard_sigmoid_grad.h:157-169`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/hard_sigmoid_grad/op_kernel/arch35/hard_sigmoid_grad.h#L157-169)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (kCastNeeded) {
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(gradFp32Buf, ubLength_ * sizeof(float));
    pipe.InitBuffer(selfFp32Buf, ubLength_ * sizeof(float));
} else {
    pipe.InitBuffer(tmpBuf, ubLength_ * sizeof(T));
}
```

#### 机制与设计原因

FP32 实例在编译期不分配两个 widening scratch，也不发出 Cast。FP16/BF16 实例需要 scratch，但在 `tmp=alpha*self+beta` 消费完 `selfFp32` 后，用同一 buffer 写 `alpha*gradFp32` 并承载两次 Select，避免第四个 FP32 结果 TBuf。Tiling 的 8/7 个 FP32 等价槽分别匹配两种实际分配。

#### 迁移边界

- 适用：同一逐元素模板同时覆盖原生计算 dtype 与需要升精度的 dtype。
- 前提：编译期类型已知；覆盖 `selfFp32` 前，所有依赖原 self 的计算必须完成。
- 失效：若后续步骤仍读取原 `selfFp32`，必须增加 buffer 或调整生命周期；若增加 buffer，也必须同步修改 Host 槽预算。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

在 `tmp` 计算前覆盖 `selfFp32` 会把边界判断输入替换成梯度；固定仓以明确的 Muls/Adds→覆盖顺序规避此错误。

## 模板与实现对比

| 对比维度 | R-001 FP32 | R-002 FP16 | R-003 BF16 | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | key 0 | key 1 | key 2 | Tiling 按 dtype 三分支，无运行时优先级竞争 |
| 模板与 dtype | `HardSigmoidGrad<float>` | `<half>` | `<bfloat16_t>` | entry 编译期选择 |
| 循环与分核 | 相同 | 相同 | 相同 | 与 dtype 无关的元素域切分 |
| Queue/TBuf | 3 Queue + tmp + mask | 3 Queue + 3 FP32 TBuf + mask | 同 R-002 | 半精度 widening |
| 内存总占用与复用 | `28U+M`；result Queue 原地筛选 | `24U+M`；selfFp32 转 result | 同 R-002 | Host 分别预留 8/7 FP32 等价槽 |
| 搬运/Vector API | FP32 DataCopy，Muls/Adds/Compare/Select | 半精度 DataCopy + Cast + FP32 管线 | BF16 DataCopy + Cast + FP32 管线 | 精度与 GM 带宽类型不同 |
| 同步、精度与 tail | Queue；原生 FP32；64 元素 compare tail | Queue；FP32 中间；CAST_RINT | Queue；FP32 中间；CAST_ROUND | round trait 是两条半精度 Route 的不可合并差异 |
| fallback/失败边界 | 无 Device fallback | 无 | 无 | Host 非法输入失败；未知 key 无 body |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开文档标为支持；稳定活动构建未接线 A2。候选 V3 仅在默认关闭的 experimental 构建中可选，故不发布卡 | 同 A2：稳定活动构建未接线 A3，experimental V3 不作为默认活动版本 | 公开文档支持，且稳定 `ascend950/arch35` 全链路闭合 |
| Host/Launcher | 稳定目录只有公共 ACLNN 前端，没有 A2 Tiling 接线；V3 Host 仅属 experimental | 同 A2 | `hard_sigmoid_grad_tiling_arch35.cpp` |
| Kernel/资源 | experimental V3 有 arch32 AIV 实现，但默认构建不扫描 | 同 A2 | 单 AIV_VECTOR Kernel，本文三 Route |

## 限制与失败边界

- 输入/输出空指针、输入 dtype 不支持、提升 dtype 无法转到输出、shape 不同或超过 8 维，由 ACLNN 第一段拒绝。
- Tiling 还拒绝空平台信息、AIV 核数或 UB 大小为零、输入 storage 元素数不等、非法 `alpha/beta`、`ubBlockSize==0`、`ubFactor<=0`；公开 ACLNN 固定属性不会触发属性失败。
- `CeilDiv`、`currentNum*sizeof(T)` 到 `DataCopyParams::blockLen` 的目标字段位宽与超大 Tensor 上界未在指定源码中展开；不可据此承诺超大元素数不会窄化或溢出。
- Tiling 对非 BF16/FP16 dtype 默认产生 key 0，但公开 ACLNN 已先做 dtype 校验；Kernel 对未知 `schMode` 没有 else/failure body，不能绕过 Host selector 直接传任意 key。
- `op_kernel/arch35/hard_sigmoid_grad_tiling_key.h` 第 20 行把 mode 1 注释为 FP16 native precision，但真实 body 明确执行 FP16→FP32→FP16；本文以 entry/body 为准。
- 核心 Kernel 的 workspace 为 0；ACLNN 总 workspace 仍可能由外围 `Contiguous/Cast/ViewCopy` 与 executor 中间 Tensor 决定，精确公式属于固定范围外框架边界。
- A2/A3 的公开支持声明与 experimental V3 源码均可见，但稳定活动目录没有 A2/A3 构建接线；根构建的 `ENABLE_EXPERIMENTAL=OFF` 使 V3 默认不活动，因此未生成 `hardsigmoid_backward.md`。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `docs/aclnnHardsigmoidBackward.md` | 产品表、函数原型、参数表 | 950 公开支持、shape/dtype/layout、空/非连续 Tensor、两段式接口 |
| 入口声明/约束 | `op_api/aclnn_hardsigmoid_backward.h` | 两个 ACLNN 符号 | 导出原型 |
| 入口实现 | `op_api/aclnn_hardsigmoid_backward.cpp` | `CheckParams`、两段式函数 | 校验、空域、前后处理、workspace/executor |
| Launcher | `op_api/hardsigmoid_grad.cpp` | `l0op::HardSigmoidGrad` | 输出分配与 AICore launcher 添加 |
| 平台注册 | `op_host/hard_sigmoid_grad_def.cpp` | `ops::HardSigmoidGrad` | dtype/format/attrs 与 `ascend950` config |
| Host/Tiling | `op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp` | `HardSigmoidGradTilingFunc` 及 helpers | 分核、UB、workspace、key 与 POD producer |
| POD | `op_kernel/arch35/hard_sigmoid_grad_tiling_data.h` | `HardSigmoidGradTilingData` | 五字段 Host/Device 契约 |
| Key | `op_kernel/arch35/hard_sigmoid_grad_tiling_key.h` | `ASCENDC_TPL_ARGS_DECL/SEL` | 三个模板 key 实例 |
| Kernel entry | `op_kernel/hard_sigmoid_grad_apt.cpp` | `hard_sigmoid_grad<schMode>` | key→类型模板 router |
| Kernel body | `op_kernel/arch35/hard_sigmoid_grad.h` | `HardSigmoidGrad<T>` 及 helpers | GM/UB、循环、Vector、同步、精度、tail |
| 构建接线 | `CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT`、`SUPPORT_TILING_DIR` | 仅 `ascend950/arch35` 活动接线 |
| 构建选择 | 根 `CMakeLists.txt` | `ENABLE_EXPERIMENTAL` | experimental 默认关闭 |
| 构建选择 | `cmake/func.cmake` | `add_category_subdirectory` | 默认只扫描稳定类别目录 |
| 候选版本审计 | `experimental/activation/hard_sigmoid_grad_v3/README.md` | 产品表、版本说明 | V3 声明 A2/A3，但属于 experimental |
| 候选构建接线 | `experimental/activation/hard_sigmoid_grad_v3/op_host/CMakeLists.txt` | `add_modules_sources` | V3 仅在 experimental 类别被选中时参与构建 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 五字段控制总长、core、tile 与公式常量 | 字段声明顺序、类型、元素单位；Host/Device 必须使用同一 struct | `HardSigmoidGradTilingData` |
| 调用顺序 | `Contiguous→Cast→HardSigmoidGrad→Cast→ViewCopy`；Kernel 内 `CopyIn→Compute→CopyOut` | 空域快返、两路输入先统一 dtype、输出最终写回视图 | ACLNN 第一段、`RunTileLoop` |
| workspace/内存 | 核心 workspace 0；UB 以 8/7 FP32 等价槽推 tile | mask 余量、Queue depth=2、半精度三个 FP32 TBuf | `GetWorkspaceSize`、`GetBufferConfig`、`Init` |
| 同步 | Queue 管理 tile 生命周期，mask 按 Compare→Select 顺序复用 | `FreeTensor` 前不得覆盖；第二次 Compare 前先消费第一次 mask | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | FP32 原生；半精度 widening；64 元素 compare 窗口 + 哨兵 | strict GT/LT、half `CAST_RINT`、BF16 `CAST_ROUND`、Select/CopyOut 仅 n | `Compute`、`CastDownRound` |
| 失败边界 | Host 校验与 Tiling guard 均失败闭合；空域成功无 launch | 不允许绕开 selector 使用未知 key；不得把默认关闭的 experimental V3 当作稳定 A2/A3 活动 body | `CheckParams`、`HardSigmoidGradTilingFunc`、根 `ENABLE_EXPERIMENTAL` |
