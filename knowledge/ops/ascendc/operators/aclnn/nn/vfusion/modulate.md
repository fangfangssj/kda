---
type: Operator
title: "Modulate A2/A3 ACLNN 算子知识"
description: "说明 aclnnModulate 在 A2/A3 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/docs/aclnnModulate.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/op_api/aclnn_modulate.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/op_api/aclnn_modulate.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/op_api/modulate.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/modulate_tiling.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/modulate_tiling.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_kernel/modulate.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/modulate_def.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/config/ascend910b/modulate_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/config/ascend910_93/modulate_binary.json
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-25T18:38:02Z"
updated_at: '2026-09-08T06:54:09Z'
platforms: [a2, a3]
cann_versions: ["9.1.0"]
---

# Modulate A2/A3 ACLNN 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | 最新且唯一公开版本 `aclnnModulateGetWorkspaceSize` + `aclnnModulate`，两段式 ACLNN |
| 计算语义 | `out[b,l,d] = self[b,l,d] * (1 + scale[b,d]) + shift[b,d]`；缺少 scale 时省略乘法，缺少 shift 时省略加法 |
| 输入输出 | `self/out: [B,L,D]`；可选 scale/shift: `[B,D]` 或 `[B,1,D]`；同 dtype，ND，FLOAT/FLOAT16/BFLOAT16 |
| 执行类型 | Modulate 计算节点为单 Kernel `AIV_VECTOR`；公开 API 还可编排框架 `Contiguous`/`ViewCopy` 节点 |
| Kernel 路线 | R-001～R-003：切 B，分别 both/shift-only/scale-only；R-004～R-006：切 L；R-007～R-009：切 D |
| 核心资源 | GM；`QueueX/QueueY` depth=2；按可选项条件分配 `QueueScale/QueueShift` depth=1；框架 workspace 与 LibApi workspace |
| 来源闭包 | `source_closed`：公开 ACLNN→L0 `Modulate`→OpDef/Tiling→key 0/1/2 router→`ModulateB/L/D` body→A2/A3 binary/config/CMake 已闭合 |
| 验证状态 | 已完成源码闭包检查；本次知识提取未执行编译、真机、数值或性能测试。 |

## 接口与语义

### 计算语义

令 `self∈R[B,L,D]`，scale/shift 在 L 维广播。公开文档和 Kernel 实体均实现 `self * (1 + scale) + shift`。`aclnn_modulate.h` 的注释写成 `self * scale + shift`，与文档及 `PreProcess/ComputeWithoutShift` 中先对 scale 执行 `Adds(..., 1)` 的实现冲突；本卡以实际 Device body 为准。输出有效域与 `self` 完全相同。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `self` | `[B,L,D]` / ND | FLOAT、FLOAT16、BFLOAT16 | 必需；API 要求 rank=3 |
| `scaleOptional` | `[B,D]` 或 `[B,1,D]` / ND | 同 `self` | 可选；存在时参与 `1+scale`；B、D 必须匹配 |
| `shiftOptional` | `[B,D]` 或 `[B,1,D]` / ND | 同 `self` | 可选；存在时参与平移；B、D 必须匹配 |
| `out` | 同 `self` / ND | 同 `self` | 必需；shape 与 dtype 均需匹配 `self` |

API 声明允许非连续 tensor，第一段接口通过 `Contiguous` 形成连续输入，再以 `ViewCopy` 写入用户的 `out`。实现只对非 ND 发 warning，不直接报错，但 `CheckShape` 在判空前无条件读取两个可选 tensor 的 storage format；因此任一可选指针为 null 时存在空指针解引用边界，详见“限制与失败边界”。

### 入口与版本

固定 Commit `955f8b33985b6d72989ebff341bb9c2c564a51bc` 仅公开无版本后缀的 `aclnnModulate`，没有 V2 等更高版本。README 与 API 文档明确列出 A2、A3 支持；`Modulate` OpDef 分别注册 `ascend910b`、`ascend910_93`，两份 binary config 覆盖三种 dtype，Kernel CMake 也对两计算单元编译同一 `op_kernel/modulate.cpp`，公开性和实现接线一致。

## Host/Launcher 执行流程

### 参数校验与前处理

`aclnnModulateGetWorkspaceSize` 的顺序是：创建 executor；`CheckNotNull(self,out)`；按 `ASCEND910B ≤ SocVersion ≤ ASCEND910E` 选择三 dtype 支持表并检查所有存在的输入及输出同 dtype；`CheckShape` 检查 `self` rank=3、可选项 rank=2/3、B/D 匹配、rank=3 时中间维为 1、`out` shape 等于 `self`。随后有三条 API 编排路径：

1. 任一已提供 tensor 为空：workspace 置 0，释放 executor，不添加计算节点。
2. scale 和 shift 都未提供：`Contiguous(self) -> ViewCopy(out)`，workspace 置 0，不启动 Modulate Kernel。
3. 至少一个可选项存在：分别对存在的输入 `Contiguous`，由 `l0op::Modulate` 分配临时输出并加入 `ADD_TO_LAUNCHER_LIST_AICORE(Modulate, ...)`，最后 `ViewCopy` 到用户输出。

固定实现中 `CheckShape` 先解引用两个可选指针，使第 2 条与“仅提供一个可选项”在空指针场景无法由源码证明安全可达；这些分支仍是明确的 Host 意图，但不是可安全复用的调用保证。

### 参数、Workspace 与生命周期

executor 拥有 `Contiguous` 结果、`l0op::Modulate` 分配的临时输出和 `ViewCopy` 任务；第一段接口返回 `uniqueExecutor->GetWorkspaceSize()`，第二段以调用者给出的 workspace、大小和 stream 执行整个任务图。Modulate Tiling 自身把 `workspace[0]` 设为 `PlatformAscendC::GetLibApiWorkSpaceSize()`；其精确数值在固定仓的框架实现边界外，Kernel 的 `workspace` 参数未被 body 读取。任务图中临时 tensor 和 workspace 至 executor/stream 执行完成前有效，用户负责 workspace 生命周期。

### 启动图与失败传播

正常计算路径：

```text
aclnnModulateGetWorkspaceSize
  -> Contiguous(self[, scale][, shift])                 # 框架节点
  -> ADD_TO_LAUNCHER_LIST_AICORE(Modulate)              # 单 AIV_VECTOR 节点
       entry modulate(x, scale, shift, tempY, workspace, tiling)
  -> ViewCopy(tempY, out)                               # 框架节点
aclnnModulate(workspace, size, executor, stream)
  -> CommonOpExecutorRun                                # 同一 stream 按 executor 图执行
```

Modulate 节点的输入/输出均为连续 ND GM tensor；blockDim=`frontNum+tailNum`，POD=`ModulateTilingData`，没有算子自有跨 Kernel workspace 边。executor 内同 stream 的 producer→consumer 顺序保证临时输出在 `ViewCopy` 前可见；任一 `CHECK_RET` 失败立即返回并阻断后续节点。框架 `Contiguous/ViewCopy` 的内部 Kernel/POD 不属于该算子拥有源码，按框架边界处理。

## Route 与参数设计

### 分核与资源计算

设 `C=GetCoreNumAiv()`、`s=sizeof(T)`、`U=平台 UB 字节数`、`N=6`（scale+shift）或 `N=5`（仅一个可选项）。Host 计算：

```text
ubLength = floor(floor(U / N / s) / 64) * 64          # 元素
frontLength = floor(totalElements / C) + 1
frontNum = totalElements mod C
tailLength = floor(totalElements / C)
tailNum = tailLength == 0 ? 0 : C - frontNum
blockDim = frontNum + tailNum
```

`totalElements` 分别为 B、B×L、D。前 `frontNum` 个核各处理 `frontLength` 单元，后续核各处理 `tailLength` 单元。BF16 的 Queue 仍按 Host `ubLength` 分配字节，构造后把 Device 可处理 `ubLength` 减半，以在同一 Queue 中同时容纳 BF16 输入和 FP32 reinterpret/cast 区。

策略判断保持源码优先级：

```text
if B*L < C:
  tileD = floor(D/C)
  choose B if B*L*tileD*s < 32 KiB else D
else if B >= C:
  choose B
else if B >= floor(C/2):
  tileL = floor(B*L/C)
  choose B if D*tileL*s < 32 KiB else L
else:
  choose L
```

### Route 选择

| 顺序 | Route | Key/模板 | 精确选择条件与优先级 | 不适用去向 |
|---:|---|---|---|---|
| 1 | R-001 切B/both | key 0 / `ModulateB<T>` | 策略为 B 且 scale、shift 均存在 | 同策略按参数状态到 R-002/R-003 |
| 2 | R-002 切B/shift-only | key 0 / `ModulateB<T>` | 策略为 B 且 scale 缺失、shift 存在 | R-003 或失败 |
| 3 | R-003 切B/scale-only | key 0 / `ModulateB<T>` | 策略为 B 且 scale 存在、shift 缺失 | 两者皆缺走框架 ViewCopy；见空指针边界 |
| 4 | R-004 切L/both | key 1 / `ModulateL<T>` | 策略为 L 且 scale、shift 均存在 | R-005/R-006 |
| 5 | R-005 切L/shift-only | key 1 / `ModulateL<T>` | 策略为 L 且 scale 缺失、shift 存在 | R-006 或失败 |
| 6 | R-006 切L/scale-only | key 1 / `ModulateL<T>` | 策略为 L 且 scale 存在、shift 缺失 | 两者皆缺走框架 ViewCopy；见空指针边界 |
| 7 | R-007 切D/both | key 2 / `ModulateD<T>` | 策略为 D 且 scale、shift 均存在 | R-008/R-009 |
| 8 | R-008 切D/shift-only | key 2 / `ModulateD<T>` | 策略为 D 且 scale 缺失、shift 存在 | R-009 或失败 |
| 9 | R-009 切D/scale-only | key 2 / `ModulateD<T>` | 策略为 D 且 scale 存在、shift 缺失 | 两者皆缺走框架 ViewCopy；见空指针边界 |

```text
parameterStatus = both ? 0 : (!scale ? 1 : 2)
SelectStrategy() -> key 0(B) / key 1(L) / key 2(D)
key + parameterStatus -> R-001 ... R-009
dtype BF16 -> 各 Route 的 BF16 FP32-cast 子实现
dtype FLOAT/FLOAT16 -> 各 Route 的原 dtype Vector 子实现
unknown key -> entry 不创建 op、不写输出
```

### 模板/实现执行矩阵

| Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|
| R-001 | 0/B | A2/A3；B 策略；status=0 | 单 `modulate` | `ModulateB<T>::Process -> Compute` | X/Y 双缓冲+Scale/Shift 单缓冲；PIPE_V | B 前后核、D 尾段、L 尾组 |
| R-002 | 0/B | A2/A3；B 策略；status=1 | 同上 | `ModulateB<T>::Process -> ComputeWithoutScale` | X/Y 双缓冲+Shift 单缓冲 | 同上 |
| R-003 | 0/B | A2/A3；B 策略；status=2 | 同上 | `ModulateB<T>::Process -> ComputeWithoutShift` | X/Y 双缓冲+Scale 单缓冲 | 同上 |
| R-004 | 1/L | A2/A3；L 策略；status=0 | 单 `modulate` | `ModulateL<T>::Process -> Compute` | 6份 UB；跨 batch 最多两段 | L 前后核、D 尾段、行组尾 |
| R-005 | 1/L | A2/A3；L 策略；status=1 | 同上 | `ModulateL<T>::Process -> ComputeWithoutScale` | 5份 UB | 同上 |
| R-006 | 1/L | A2/A3；L 策略；status=2 | 同上 | `ModulateL<T>::Process -> ComputeWithoutShift` | 5份 UB | 同上 |
| R-007 | 2/D | A2/A3；D 策略；status=0 | 单 `modulate` | `ModulateD<T>::Process -> Compute` | 6份 UB；每核一段 D | D 前后核、UB 尾段、逐 L |
| R-008 | 2/D | A2/A3；D 策略；status=1 | 同上 | `ModulateD<T>::Process -> ComputeWithoutScale` | 5份 UB | 同上 |
| R-009 | 2/D | A2/A3；D 策略；status=2 | 同上 | `ModulateD<T>::Process -> ComputeWithoutShift` | 5份 UB | 同上 |

各行中 `T∈{float,half,bfloat16_t}`。BF16 子实现先 `Cast(..., CAST_NONE)` 到 FP32，FP32 运算后 `CAST_RINT` 回 BF16；FLOAT/FLOAT16 直接按 T 运算。该差异由编译期 `std::is_same<T,bfloat16_t>` 分支完全列明，不另改变 Host key、POD 或启动图。

### 隐藏性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| UB 份数 | status=0 用 N=6，否则 N=5 | X/Y 各 depth2，两个或一个可选 Queue 各 depth1 | 全部 | 缩小 `ubLength`，不换 key | `ModulateTilingOp::Init` |
| 64 元素对齐 | `ubLength=floor(U/N/s/64)*64` | Queue 容量及 BF16 分区 | 全部 | 向下取整 | `ModulateTilingOp::Init` |
| B/D 选择阈值 | `B*L<C && B*L*floor(D/C)*s >= 32KiB` | 只有 D 分片达到最小工作量才切 D | R-007～009 | 切 B | `SelectStrategy` |
| B/L 选择阈值 | `B<C && B>=C/2 && D*floor(BL/C)*s >=32KiB` | 每核 L 分片工作量足够才切 L | R-004～006 | 切 B | `SelectStrategy` |
| 多行合并 | `!useDTiling && DId==0 && opCopyLength*s<8192` | 小 D 时一次 DMA/Vector 处理多行 | R-001～006 | `handleL=1` | `Compute*` |
| DMA 字段窄化 | `handleL -> uint16_t`, `opCopyLength*s -> uint32_t` | `DataCopyExtParams` ABI | 全部 | 固定仓没有显式溢出 fallback | `CopyIn/CopyOut/CopyCommom` |
| BF16 半 UB | `T=bfloat16_t` | 同一 Queue 的 BF16 源区与 FP32 计算区 | 全部 | 非 BF16 保持 Host ubLength | `ModulateBase` |
| 空域 | API 任一已提供 tensor empty | 避免启动 | 全部 | workspace=0，直接成功 | `aclnnModulateGetWorkspaceSize` |
| unknown key | key 非 0/1/2 | router 没有 else body | 全部 | 不写输出 | `modulate` |

### POD/ABI

`ModulateTilingData` 的声明顺序是十个 `int64_t` 字段，但它由 `BEGIN_TILING_DATA_DEF/TILING_DATA_FIELD_DEF` 宏生成，宏定义不在固定仓内；因此不能把普通 C++ struct 的布局推定成最终序列化 ABI。下表保留声明槽位顺序，`sizeof/alignof/offsetof` 统一标为固定仓不可见。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ModulateTilingData` | `inputB` | int64/64 | 槽位1；size/align/offset固定仓不可见 | 元素维 | `Init: xShape.dim(0)` | B/L/D 循环、offset | 非空计算应 >0 |
| 同上 | `inputL` | int64/64 | 槽位2；同上 | 元素维 | `Init: xShape.dim(1)` | 行/batch 边界、offset | 非空计算应 >0 |
| 同上 | `inputD` | int64/64 | 槽位3；同上 | 元素维 | `Init: xShape.dim(2)` | D 循环、广播 offset | 非空计算应 >0 |
| 同上 | `ubLength` | int64/64 | 槽位4；同上 | T 元素 | `floor(U/N/s/64)*64` | Queue 容量、tile；BF16 再除2 | Host 未显式检查 0 |
| 同上 | `frontNum` | int64/64 | 槽位5；同上 | core | `totalElements%C` | 核类型、base offset | `[0,C-1]` |
| 同上 | `frontLength` | int64/64 | 槽位6；同上 | B/L/D 单元 | `floor(total/C)+1` | 前核长度 | total<C 时为1 |
| 同上 | `tailNum` | int64/64 | 槽位7；同上 | core | `tailLength==0?0:C-frontNum` | blockDim/后核 | 可为0 |
| 同上 | `tailLength` | int64/64 | 槽位8；同上 | B/L/D 单元 | `floor(total/C)` | 后核长度 | 可为0但该核不启动 |
| 同上 | `useDTiling` | int64/64 | 槽位9；同上 | bool值 | D 策略写1，默认0 | offset 与多行 guard | B/L inactive=0 |
| 同上 | `parameterStatus` | int64/64 | 槽位10；同上 | 枚举 | both=0/no-scale=1/no-shift=2 | 条件分配及 `CopyAndCompute` switch | 两者皆缺会写1，但公开意图绕过 Kernel |

### Route 源码闭包

| Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|
| R-001～003 | 0 / `TilingB` | `modulate_tiling.cpp: SelectStrategy/CalcTilingParam` | `ModulateTilingData` + status 0/1/2 | `op_kernel/modulate.cpp: modulate`, key0 | `ModulateB::Init/Process`, `ModulateBase::Compute*` | `op_kernel/CMakeLists.txt`, ascend910b/910_93 configs |
| R-004～006 | 1 / `TilingL` | 同上 | 同上 | 同文件 key1 | `ModulateL::Init/Process`, `Compute*` | 同上 |
| R-007～009 | 2 / `TilingD` | 同上 | 同上 | 同文件 key2 | `ModulateD::Init/Process`, `Compute*` | 同上 |

## Kernel 实现

### 公共执行流程

入口 `modulate(x,scale,shift,y,workspace,tiling)` 创建单个 `TPipe`，解析 POD，并按 key 构造 B/L/D 类。公共基类先缓存全部 POD 字段，初始化 X/Y 双缓冲与条件可选 Queue。每个 tile 的管线为：可选参数 `GM→QueueScale/Shift`；X `GM→QueueX`；按 status 调用 `Compute/ComputeWithoutScale/ComputeWithoutShift`；结果入 `QueueY` 并 `QueueY→GM`。Queue 的 `EnQue/DeQue/FreeTensor` 建立 MTE/V 的生产消费与复用生命周期，Vector 指令间以 `PipeBarrier<PIPE_V>` 保护原地 scale 加 1、X 乘法与后续加法/cast。

多行模式仅在非 D 切分、当前为完整 D 段且单行小于 8192 字节时启用：`handleL=min(剩余行,ubLength/alignedD)`，`alignedD=ceil(opCopyLength/(32/s))*(32/s)`。`DataCopyPad` 以 `blockCount=handleL`、`blockLen=opCopyLength*s`、src/dst stride=0 搬连续整行，padding lane 只驻留 UB；Vector 有效长度仍为 `opCopyLength`，CopyOut 也只写有效字节。

下列各 Route 的资源表使用统一符号：`u=Host ubLength`、`e=Device 有效 ubLength（BF16 为 u/2，否则 u）`、`s=sizeof(T)`、`a=ceil(n/(32/s))*(32/s)`。FLOAT/FLOAT16 原 dtype 计算；BF16 的 GM 输入落在 Queue 后半区，前半区 reinterpret 为 FP32，并在写回前 `CAST_RINT`。

### Route R-001：切 B、scale+shift

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | B 策略决策树 | key 0 |
| Tiling | 同上 | `Init` | `scaleShape && shiftShape` | status=0，N=6 |
| Router | `op_kernel/modulate.cpp` | `modulate` | `TILING_KEY_IS(0)` | `ModulateB<DTYPE_X>` |

#### Device 数据流

##### 函数级调用链

```text
ModulateAiCore -> modulate -> ModulateB<T>::Init/Process
  -> CopyScaleIn + CopyShiftIn -> Compute
  -> CopyIn -> [BF16 Cast] -> Mul -> Add -> [BF16 Cast] -> CopyOut
```

##### 源码等价伪码

```text
bind core's contiguous B slice; for each local b:
  for each D tile n=min(e, remaining D):
    load scale[b,n], shift[b,n] once
    for L rows, optionally grouped by handleL:
      load x[rows,n]; scale=scale+1; y=x*scale+shift; store valid n per row
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| core GM X/Y | 前核 `frontLength*L*D*blockIdx`，后核加 tail offset | `currentB*L*D` | T/ND | Host/user, Kernel | Kernel, ViewCopy | 当前 Kernel | B 连续片 |
| Scale/Shift GM | 对应 B 片 `*D` | `currentB*D` | T/[B,D] | Host | CopyScale/ShiftIn | 当前 Kernel | 每 D tile 重载一次/批 |
| QueueX | tile offset | `2*u*s` | T；BF16 后半源/前半FP32 | CopyIn | Compute | TPipe；Free 后复用 | depth2 |
| QueueY | tile offset | `2*u*s` | T/同上 | Compute | CopyOut | TPipe；Free 后复用 | depth2 |
| QueueScale | D tile | `u*s` | T/同上 | CopyScaleIn | Compute | 当前 D tile | depth1，原地 `+1` |
| QueueShift | D tile | `u*s` | T/同上 | CopyShiftIn | Compute | 当前 D tile | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  qx["QueueX · depth=2 · 2*u*s"]:2
  qy["QueueY · depth=2 · 2*u*s"]:2
  qs["QueueScale · depth=1 · u*s"]:1
  qh["QueueShift · depth=1 · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total = (2+2+1+1)*u*s = 6*u*s <= U
dynamic_ub = N/A
workspace_total = GetLibApiWorkSpaceSize() + executor framework workspace（固定仓内数值不可见）
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| QueueX slot | CopyIn | Compute | 下一行组 | EnQue→DeQue→FreeTensor | Compute 完成后释放 |
| QueueY slot | Compute | CopyOut | 下一行组 | EnQue→DeQue→FreeTensor | GM 写完后释放 |
| Scale/Shift | CopyScale/ShiftIn | 所有当前 tile 行组 | 下一 D tile | tile 末 FreeTensor | 不得在 L 循环内覆盖 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→UB | Queue | DataCopyPad | DeQue | X/scale/shift | blockLen=n*s；多行仅完整D段 |
| scale+1→Mul→Add | PIPE_V barrier | Adds/Mul | Mul/Add | 原地 scale、x | BF16升FP32；native其余dtype |
| BF16回写 | PIPE_V barrier + Cast RINT | Add | CopyOut | yLocal | 有效 `handleL*a` cast，GM仅写n |
| UB→GM | Queue | EnQue y | CopyOut | y slot | D余数 `n=D%e` 单独处理 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `SelectStrategy/CalcTilingParam/SetTiling` | key0、B分核、blockDim |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate`, `ModulateB<T>` | key0映射B body |
| Device body/helper | 同上 | `ModulateB::Process`, `Compute`, `PreProcess` | D tile、多行、scale+1、Mul/Add、BF16 cast |

### Route R-002：切 B、仅 shift

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | B 策略 | key0 |
| Tiling | 同上 | `Init` | `!scaleShape` 且 shift 存在 | status=1，N=5 |
| Router | `op_kernel/modulate.cpp` | `CopyAndCompute` | `NO_SCALE` | `ComputeWithoutScale` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateB<T>::Process -> CopyShiftIn -> ComputeWithoutScale
  -> CopyIn -> [BF16 Cast] -> Add(x,shift) -> [BF16 Cast] -> CopyOut
```

##### 源码等价伪码

```text
for core B slice, b, D tile: load shift once; for grouped L rows load x; y=x+shift; store valid D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM X/Y | 前核基址 `frontLength*L*D*blockIdx`；后核基址 `frontLength*L*D*frontNum+tailLength*L*D*(blockIdx-frontNum)` | `currentB*L*D` | T/ND | Host/Kernel | Kernel/ViewCopy | Kernel | 连续 |
| QueueX/QueueY | 当前 D tile | 各 `2*u*s` | T | CopyIn/Compute | Compute/CopyOut | TPipe | depth2 |
| QueueShift | 当前 D tile | `u*s` | T | CopyShiftIn | ComputeWithoutScale | D tile | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qh["QueueShift · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total = 5*u*s <= U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y slots | CopyIn/Compute | Compute/CopyOut | 下一行组 | Queue FreeTensor | 消费完成 |
| Shift slot | CopyShiftIn | 当前 D tile 所有行 | 下一 D tile | tile末 FreeTensor | L循环完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 搬入/搬出 | Queue+DataCopyPad | GM/Compute | Compute/GM | slots | D尾段、行组尾 |
| BF16 | Cast+PIPE_V | BF16源 | FP32 Add/回转 | X/shift/Y | CAST_NONE→Add→CAST_RINT |
| native | Vector Add | X/shift | Y | 有效n lane | FLOAT/FLOAT16不升精度 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `Init/SelectStrategy` | status1、N5、key0 |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/CopyAndCompute` | key0+NO_SCALE |
| Device body/helper | 同上 | `ComputeWithoutScale` | shift广播、Add、tail |

### Route R-003：切 B、仅 scale

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | B 策略 | key0 |
| Tiling | 同上 | `Init` | scale存在且 `!shiftShape` | status=2，N=5 |
| Router | `op_kernel/modulate.cpp` | `CopyAndCompute` | `NO_SHIFT` | `ComputeWithoutShift` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateB<T>::Process -> CopyScaleIn -> ComputeWithoutShift
  -> scale+1 -> CopyIn -> [BF16 Cast] -> Mul -> [BF16 Cast] -> CopyOut
```

##### 源码等价伪码

```text
for core B slice, b, D tile: load scale once; scale+=1; for grouped L rows load x; y=x*scale; store valid D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM X/Y | 前核基址 `frontLength*L*D*blockIdx`；后核基址 `frontLength*L*D*frontNum+tailLength*L*D*(blockIdx-frontNum)` | `currentB*L*D` | T/ND | Host/Kernel | Kernel/ViewCopy | Kernel | 连续 |
| QueueX/QueueY | 当前 tile | 各 `2*u*s` | T | CopyIn/Compute | Compute/CopyOut | TPipe | depth2 |
| QueueScale | 当前 tile | `u*s` | T | CopyScaleIn/Adds | Mul | D tile | depth1、原地加1 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qs["QueueScale · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total = 5*u*s <= U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y slots | CopyIn/Compute | Compute/CopyOut | 下一行组 | Queue FreeTensor | 消费完成 |
| Scale slot | CopyScaleIn/Adds | 所有当前 tile 行 | 下一 D tile | PIPE_V；tile末FreeTensor | Adds先于Mul且L循环完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scale预处理 | Adds+PIPE_V | scale load | Mul | scale slot | BF16在FP32加1 |
| x计算 | Queue/Vector | CopyIn | Mul/CopyOut | X/Y | BF16回转RINT；D余数有效写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `Init/SelectStrategy` | status2、N5、key0 |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/CopyAndCompute` | key0+NO_SHIFT |
| Device body/helper | 同上 | `ComputeWithoutShift` | scale+1、Mul、tail |

### Route R-004：切 L、scale+shift

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | 最后分支 L，或半核区且 `tileLSize>=32KiB` | key1 |
| Tiling | 同上 | `Init` | 两可选 shape 均存在 | status0，N6 |
| Router | `op_kernel/modulate.cpp` | `modulate` | `TILING_KEY_IS(1)` | `ModulateL<T>` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateL<T>::Init/GetBatchEndId/GetSecondFirstL/Process
  -> per covered batch: CopyScaleIn + CopyShiftIn -> Compute
  -> CopyIn grouped L rows -> Mul/Add -> CopyOut
```

##### 源码等价伪码

```text
bind core's flattened [B,L] row slice
split it at batch boundary into one or two segments
for each segment and D tile: load that batch's scale/shift; process segment rows in handleL groups
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| core GM X/Y | 前核 `frontLength*D*idx`；后核 `frontLength*D*frontNum+tailLength*D*relative` | `currentRows*D` | T/flattened BL,D | Host/Kernel | Kernel/ViewCopy | Kernel | 连续行片 |
| Scale/Shift GM | `batchStartId*D`，段内 BId 经 `isTilingL?0:BId` 修正为局部基址 | 每批D | T/[B,D] | Host | Compute | 当前 batch/D tile | 广播 |
| QueueX/Y | 当前行组/D tile | 各 `2*u*s` | T | CopyIn/Compute | Compute/CopyOut | TPipe | depth2 |
| QueueScale/Shift | 当前 batch/D tile | 各 `u*s` | T | Copy*In | Compute | D tile | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qs["QueueScale · u*s"]:1
  qh["QueueShift · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total=6*u*s<=U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y slots | CopyIn/Compute | Compute/CopyOut | 下一行组 | Queue协议 | 当前组消费完成 |
| Scale/Shift | 每 batch/D tile CopyIn | 该段所有 L 行 | 下一段/tile | FreeTensor | batch 分段结束 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| batch 分段 | 地址算术 | Init | Process | scale/shift batch 归属 | 核片最多按源码拆为两段 |
| 向量管线 | Queue+PIPE_V | DMA/Adds/Mul | Vector/CopyOut | UB slots | 同 R-001；末行组缩短handleL |
| D tail | `remainLength` | Process | CopyAndCompute | 最后D段 | 只搬/算/写有效n |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `SelectStrategy/CalcTilingParam` | key1、BL分核 |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/ModulateL<T>` | key1映射L body |
| Device body/helper | 同上 | `ModulateL::Init/Process`, `GetBatchEndId/GetSecondFirstL`, `Compute` | batch边界、广播、多行、精度 |

### Route R-005：切 L、仅 shift

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | L 策略 | key1 |
| Tiling | 同上 | `Init` | `!scaleShape` 且 shift存在 | status1，N5 |
| Router | `op_kernel/modulate.cpp` | `CopyAndCompute` | NO_SCALE | `ComputeWithoutScale` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateL<T>::Process -> per batch/D tile CopyShiftIn
  -> grouped CopyIn -> ComputeWithoutScale(Add) -> CopyOut
```

##### 源码等价伪码

```text
split flattened BL core slice at batch boundary; load shift per batch/D tile; y=x+shift for grouped rows
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM X/Y | 前核 `frontLength*D*blockIdx`；后核 `frontLength*D*frontNum+tailLength*D*(blockIdx-frontNum)` | `currentRows*D` | T/ND | Host/Kernel | Kernel/ViewCopy | Kernel | 连续 |
| QueueX/Y | 行组/D tile | 各 `2*u*s` | T | CopyIn/Compute | Compute/CopyOut | TPipe | depth2 |
| QueueShift | batch/D tile | `u*s` | T | CopyShiftIn | Add | tile | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qh["QueueShift · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total=5*u*s<=U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y slots | CopyIn/Compute | Add/CopyOut | 下一组 | Queue协议 | 消费完成 |
| Shift | CopyShiftIn | 当前batch/tile各行 | 下一tile | FreeTensor | 行循环完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| Queue | EnQue/DeQue/Free | DMA/Vector | Vector/DMA | slots | 行组与D尾缩短 |
| BF16/native | Cast+Add / Add | X、shift | Y | 有效lane | BF16 FP32后RINT；其余原dtype |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `Init/SelectStrategy` | status1、N5、key1 |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/CopyAndCompute` | L+NO_SCALE |
| Device body/helper | 同上 | `ModulateL::Process/ComputeWithoutScale` | batch分段、shift广播、Add |

### Route R-006：切 L、仅 scale

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | L 策略 | key1 |
| Tiling | 同上 | `Init` | scale存在且 `!shiftShape` | status2，N5 |
| Router | `op_kernel/modulate.cpp` | `CopyAndCompute` | NO_SHIFT | `ComputeWithoutShift` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateL<T>::Process -> CopyScaleIn -> ComputeWithoutShift
  -> scale+1 -> grouped CopyIn -> Mul -> CopyOut
```

##### 源码等价伪码

```text
for each batch segment/D tile: load scale; scale+=1; for grouped rows y=x*scale; write valid D
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM X/Y | 前核 `frontLength*D*blockIdx`；后核 `frontLength*D*frontNum+tailLength*D*(blockIdx-frontNum)` | `currentRows*D` | T/ND | Host/Kernel | Kernel/ViewCopy | Kernel | 连续 |
| QueueX/Y | 行组/D tile | 各 `2*u*s` | T | CopyIn/Compute | Compute/CopyOut | TPipe | depth2 |
| QueueScale | batch/D tile | `u*s` | T | CopyScaleIn/Adds | Mul | tile | depth1、原地加1 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qs["QueueScale · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total=5*u*s<=U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y | CopyIn/Compute | Mul/CopyOut | 下一组 | Queue协议 | 消费完成 |
| Scale | CopyScaleIn/Adds | 当前batch/tile各行 | 下一tile | PIPE_V+FreeTensor | Adds完成且行循环结束 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scale+1 | Adds+PIPE_V | load | Mul | scale | BF16用FP32 |
| 输出 | Queue+Cast | Mul | CopyOut | Y | BF16 RINT；D/行尾只写有效域 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `Init/SelectStrategy` | status2、N5、key1 |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/CopyAndCompute` | L+NO_SHIFT |
| Device body/helper | 同上 | `ModulateL::Process/ComputeWithoutShift` | batch广播、scale+1、Mul |

### Route R-007：切 D、scale+shift

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | `B*L<C && B*L*floor(D/C)*s>=32KiB` | key2，useDTiling=1 |
| Tiling | 同上 | `Init` | 两可选均存在 | status0，N6 |
| Router | `op_kernel/modulate.cpp` | `modulate` | `TILING_KEY_IS(2)` | `ModulateD<T>` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateD<T>::Init/Process
  -> for each B and local D sub-tile: CopyScaleIn+CopyShiftIn -> Compute
  -> for each L: CopyIn -> Mul/Add -> CopyOut
```

##### 源码等价伪码

```text
core owns D interval [coreD,coreD+dNum)
for b: for local D chunks n<=e: load scale/shift[b,coreD+chunk]
  for l=0..L-1: load x[b,l,D-slice]; y=x*(1+scale)+shift; store
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM X/Y | `b*L*D+l*D + front/tail core D start + chunk` | 每次n，跨B/L | T/ND | Host/Kernel | Kernel/ViewCopy | Kernel | D片非连续跨行 |
| Scale/Shift GM | `b*D + core D start + chunk` | n | T/[B,D] | Host | Compute | B/chunk | L广播 |
| QueueX/Y | 单行D chunk | 各 `2*u*s` | T | CopyIn/Compute | Compute/CopyOut | TPipe | depth2；handleL=1 |
| QueueScale/Shift | B/D chunk | 各 `u*s` | T | Copy*In | Compute | 当前B/chunk | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 6
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qs["QueueScale · u*s"]:1
  qh["QueueShift · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total=6*u*s<=U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y slots | 每L CopyIn/Compute | Compute/CopyOut | 下一L | Queue协议 | 每行完成 |
| Scale/Shift | 每B/chunk CopyIn | 全部L | 下一chunk/B | FreeTensor | L循环完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分核 | front/tail地址式 | Tiling | Init/CalcOffset | D无重叠分片 | `dNum`前后核不同 |
| 向量 | Queue+PIPE_V | DMA/Adds/Mul | Compute/CopyOut | slots | useDTiling禁用多行；BF16升FP32 |
| chunk tail | remainLength | Process | CopyAndCompute | 本核最后D段 | n有效写 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `SelectStrategy/CalcTilingParam` | key2、D分核、useDTiling |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/ModulateD<T>` | key2到D body |
| Device body/helper | 同上 | `ModulateD::Process`, `CalcOffset*`, `Compute` | D前后核地址、B/L循环、广播 |

### Route R-008：切 D、仅 shift

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | D 策略 | key2 |
| Tiling | 同上 | `Init` | scale缺失、shift存在 | status1，N5 |
| Router | `op_kernel/modulate.cpp` | `CopyAndCompute` | NO_SCALE | `ComputeWithoutScale` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateD<T>::Process -> CopyShiftIn -> per L CopyIn/Add/CopyOut
```

##### 源码等价伪码

```text
for b and core-owned D chunks: load shift once; for every l y=x+shift; write only chunk length
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM X/Y | `b*L*D+l*D+coreD+chunk`；`coreD=frontLength*idx` 或 `frontLength*frontNum+tailLength*(idx-frontNum)` | 每行n | T/ND | Host/Kernel | Kernel/ViewCopy | Kernel | 跨行stride=D |
| QueueX/Y | 单行chunk | 各 `2*u*s` | T | CopyIn/Compute | Add/CopyOut | TPipe | depth2 |
| QueueShift | B/chunk | `u*s` | T | CopyShiftIn | Add | B/chunk | depth1 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qh["QueueShift · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total=5*u*s<=U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y | per-L CopyIn/Compute | Add/CopyOut | 下一L | Queue协议 | 行完成 |
| Shift | CopyShiftIn | 全L | 下一chunk | FreeTensor | L循环完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| DMA/Vector | Queue | CopyIn | Add/CopyOut | slots | handleL固定1，D chunk尾缩短 |
| BF16/native | Cast/Add/Cast或Add | X/shift | Y | lane mask/有效长度 | BF16 FP32+RINT |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `Init/SelectStrategy` | status1、N5、key2 |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/CopyAndCompute` | D+NO_SCALE |
| Device body/helper | 同上 | `ModulateD::Process/ComputeWithoutScale` | D分片、shift广播、Add |

### Route R-009：切 D、仅 scale

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host | `op_host/modulate_tiling.cpp` | `SelectStrategy` | D 策略 | key2 |
| Tiling | 同上 | `Init` | scale存在、shift缺失 | status2，N5 |
| Router | `op_kernel/modulate.cpp` | `CopyAndCompute` | NO_SHIFT | `ComputeWithoutShift` |

#### Device 数据流

##### 函数级调用链

```text
modulate -> ModulateD<T>::Process -> CopyScaleIn -> scale+1
  -> per L CopyIn/Mul/CopyOut
```

##### 源码等价伪码

```text
for b and core D chunks: load scale; scale+=1; for every l y=x*scale; write chunk tail exactly
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| GM X/Y | `b*L*D+l*D+coreD+chunk`；`coreD=frontLength*idx` 或 `frontLength*frontNum+tailLength*(idx-frontNum)` | 每行n | T/ND | Host/Kernel | Kernel/ViewCopy | Kernel | D分片 |
| QueueX/Y | 单行chunk | 各 `2*u*s` | T | CopyIn/Compute | Mul/CopyOut | TPipe | depth2 |
| QueueScale | B/chunk | `u*s` | T | CopyScaleIn/Adds | Mul | B/chunk | depth1、原地加1 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  qx["QueueX · 2*u*s"]:2
  qy["QueueY · 2*u*s"]:2
  qs["QueueScale · u*s"]:1
```

##### 字节账与复用顺序

```text
UB_total=5*u*s<=U; dynamic_ub=N/A; workspace_total=LibApi+executor framework workspace
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| X/Y | per-L CopyIn/Compute | Mul/CopyOut | 下一L | Queue协议 | 行完成 |
| Scale | CopyScaleIn/Adds | 全L | 下一chunk | PIPE_V+FreeTensor | 加1完成且L循环完成 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| scale预处理 | Adds+PIPE_V | load | Mul | Scale | BF16 FP32处理 |
| 计算输出 | Queue+Mul+Cast | X/scale | GM Y | X/Y | BF16 RINT；D尾有效n |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/modulate_tiling.cpp` | `Init/SelectStrategy` | status2、N5、key2 |
| Kernel entry/router | `op_kernel/modulate.cpp` | `modulate/CopyAndCompute` | D+NO_SHIFT |
| Device body/helper | 同上 | `ModulateD::Process/ComputeWithoutShift` | D分片、scale+1、Mul |

## 算子特定优化

### Trick T-001：小 D 合并多行搬运与计算

#### 触发与源码依据

- 适用 Route：R-001～R-006
- 触发条件：`!useDTiling && DId==0 && opCopyLength*sizeof(T)<8192`
- Golden 来源：[`vfusion/modulate/op_kernel/modulate.cpp:196-208`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_kernel/modulate.cpp#L196-208)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t alignedLength = ALIGNED_SIZE / sizeof(T);
this->alignedD = (opCopyLength + alignedLength - 1) & ~(alignedLength - 1);
bool isDSmall = (!this->useDTiling) && (DId == 0) && (opCopyLength * sizeof(T) < MIN_DLENGTH);
int64_t handleL = isDSmall ? min(loopLEnd - loopLStart, this->ubLength / this->alignedD) : 1;
```

#### 机制与设计原因

完整 D 行较小时，单次 `DataCopyPad` 的 `blockCount` 改为 `handleL`，X/Y Queue 同时承载多行；scale/shift 只搬一次并跨这些行广播。`alignedD` 按 32 字节换算为元素后向上对齐，`ubLength/alignedD` 保证多行布局不超出单 Queue slot。机制减少小行场景的逐行 DMA/Queue 调度次数；源码没有给出实测收益。

#### 迁移边界

- 适用：按行广播同一参数、行连续且完整 D 被当前 tile 覆盖的 Vector 元素算子。
- 前提：非 D 分核、`D*sizeof(T)<8192`、行 padding stride 使用 `alignedD`、有效 GM 写仍用原 D。
- 失效：D 分核或大行时回到 `handleL=1`，不得跨不连续 D 片合并。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

若在 D 分核时沿用多行且假设 GM 行片连续，会跨过每行未归当前核的 D 区间；源码用 `!useDTiling` 明确排除此做法。

### Trick T-002：BF16 在同一 Queue 内分区完成升精度计算

#### 触发与源码依据

- 适用 Route：R-001～R-009 的 BF16 子实现
- 触发条件：`std::is_same<T,bfloat16_t>::value`
- Golden 来源：[`vfusion/modulate/op_kernel/modulate.cpp:47-56`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_kernel/modulate.cpp#L47-56)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
this->pipe->InitBuffer(this->QueueX, DOUBLE_BUFFER, this->ubLength * sizeof(T));
this->pipe->InitBuffer(this->QueueY, DOUBLE_BUFFER, this->ubLength * sizeof(T));
if(this->parameterStatus != NO_SCALE) {
    this->pipe->InitBuffer(this->QueueScale, 1, this->ubLength * sizeof(T));
}
if(this->parameterStatus != NO_SHIFT) {
    this->pipe->InitBuffer(this->QueueShift, 1, this->ubLength * sizeof(T));
}
this->ubLength = std::is_same<T, bfloat16_t>::value ? this->ubLength / HALF_UBLENGTH : this->ubLength;
```

#### 机制与设计原因

Host 按 Queue 的完整物理字节数做 UB 预算；BF16 body 随后将有效 tile 元素数减半，`DataCopyPad` 把 BF16 源放到 `local[ubLength]`，再把同一物理 Queue 的起始地址 reinterpret 为 FP32 目标。scale/shift、X、Y 都以该模式完成 `CAST_NONE→FP32 Adds/Mul/Add→CAST_RINT`，无需额外 TBuf。Queue 的字节账不变，而有效 BF16 tile 容量让出空间给 FP32 中间态。

#### 迁移边界

- 适用：可证明 cast 源/目标在一个足够大的 LocalTensor 内按官方原地转换约定安全布置的 BF16 元素管线。
- 前提：有效长度减半；所有 CopyIn 使用后半区；计算与 CopyOut 遵循 Queue/PIPE_V 顺序。
- 失效：若仍按 Host `ubLength` 个 BF16 元素计算，FP32 目标容量不足；非 BF16 不做减半与后半区加载。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定源码以 `ubLength/2` 约束展示了反例边界：不能把整个 BF16 Queue 都当作输入，同时还在同一物理区物化等长 FP32 结果。

### Trick T-003：Host 以工作量阈值选择切 B/L/D

#### 触发与源码依据

- 适用 Route：全部，决定 R-001～R-009 的 B/L/D 组
- 触发条件：`SelectStrategy` 的有序决策树
- Golden 来源：[`vfusion/modulate/op_host/modulate_tiling.cpp:168-185`](https://gitcode.com/cann/ops-nn/blob/955f8b33985b6d72989ebff341bb9c2c564a51bc/vfusion/modulate/op_host/modulate_tiling.cpp#L168-185)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (this->tilingData.inputB * this->tilingData.inputL < this->coreNum) {
    int64_t tileD = this->tilingData.inputD / this->coreNum;
    int64_t tileDSize = this->tilingData.inputB * this->tilingData.inputL * tileD * this->dataTypeSize;
    return tileDSize < MIN_TILE_SIZE ? TilingStrategy::TilingB : TilingStrategy::TilingD;
}
```

#### 机制与设计原因

当 BL 不足以填满核时，只有预计每个 D 分片跨全部 BL 累计达到 32KiB 才切 D，否则保持切 B；BL 足够时又根据 B 是否已能铺核以及 L 分片字节量选择 B/L。该 specialization 避免为了增加核数而把每核有效工作切得过小，并让 Device body 使用无运行时跨核协调的连续分片。

#### 迁移边界

- 适用：B/L/D 三维、参数沿 L 广播、三种切分都有无重叠地址式的算子。
- 前提：阈值按 `sizeof(T)` 计算并保持源码判断优先级；D 路只在 `B*L<C` 分支可达。
- 失效：阈值不满足时回到 B 或 L；不得仅按最大核利用率强制 D 路。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

固定仓未提供可验证反例；源码只给出阈值分流与 fallback，没有性能数据。

## 模板与实现对比

| 对比维度 | R-001～003 切B | R-004～006 切L | R-007～009 切D | 设计选择依据 |
|---|---|---|---|---|
| 命中条件/优先级 | B足以铺核，或细分工作量不足 | BL铺核且L分片足够，或B很小 | BL不足且D分片累计≥32KiB | `SelectStrategy` 有序树 |
| 模板与 dtype | `ModulateB<T>` | `ModulateL<T>` | `ModulateD<T>` | T 三 dtype；BF16升FP32 |
| 循环与分核 | 核→B→D tile→L组 | 核BL片→batch段→D tile→L组 | 核D片→B→D chunk→L | 广播参数归属决定循环顺序 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | X/Y depth2，可选Queue depth1；无TBuf/dynamic UB | 同左 | 同左 | status0为6份，status1/2为5份 |
| 内存总占用与复用 | 5或6×u×s；参数跨L复用 | 同，参数跨batch段的L复用 | 同，参数跨全部L复用 | Queue末尾FreeTensor |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad；Adds/Mul/Add/Cast | 同；可合并多行 | 同；固定单行 | 全部 AIV_VECTOR，无SIMT/NDDMA/通信 |
| 同步、精度与 tail | Queue+PIPE_V；B/D/L组尾 | Queue+PIPE_V；BL核片、batch、D/行组尾 | Queue+PIPE_V；D核片/chunk尾 | BF16 CAST_NONE→FP32→RINT |
| fallback/失败边界 | unknown key不写；DMA窄化无guard | 同 | 同 | Host 无通用 fallback Kernel |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公开支持；`ascend910b` config/OpDef/CMake闭合 | 公开支持；`ascend910_93` config/OpDef/CMake闭合 | 公开支持；另见 `_950` 卡 |
| Host/Launcher | 共用非 regbase `ModulateTilingOp` | 共用非 regbase `ModulateTilingOp` | 使用 `ModulateTilingForRegbase`，optional 进入模板 key |
| Kernel/资源 | 共用 `op_kernel/modulate.cpp` AIV_VECTOR | 共用该 Kernel | 独立 arch35 MicroAPI AIV_VECTOR body |

## 限制与失败边界

- `CheckShape` 在验证 optional 指针是否为空前执行 `scaleOptional->GetStorageFormat()`、`shiftOptional->GetStorageFormat()`；任一 optional 为 null 时存在空指针解引用。其后“两个都缺则 ViewCopy”和单 optional Kernel 分支因此不能被固定源码证明为安全的公开 ACLNN 路径。
- `self` 为空时的条件也在短路右侧调用 optional 的 `IsEmpty()`；仅当 optional 非 null 才安全。空 tensor 成功路径不写输出数据。
- API 文档/Kernel 语义是 `self*(1+scale)+shift`，但公开头注释遗漏 `1+`；复用实现时不得按头注释改写数学语义。
- Host 将 `handleL` 窄化为 DMA 的 `uint16_t blockCount`、将 `opCopyLength*sizeof(T)` 窄化为 `uint32_t blockLen`，固定仓没有显式上界校验或 fallback。
- `dataType` switch 的 default 不报错且 `dataTypeSize` 保持0；公开 ACLNN dtype 检查阻断该情况，绕过 ACLNN 的内部调用不得依赖此行为。
- exact LibApi workspace、executor 的 `Contiguous/ViewCopy` workspace 公式和内部 Kernel 由框架拥有，固定仓不可见；Modulate body 不读取 workspace。
- unknown key、未知 `parameterStatus` 都没有报错：Device 分支不执行或 switch default 不计算，输出不保证被写。
- 本次知识提取未执行编译、真机、数值或性能测试。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `vfusion/modulate/docs/aclnnModulate.md` | `aclnnModulate*` | 平台、shape/dtype、两段式公开接口 |
| 入口声明/约束 | `vfusion/modulate/op_host/op_api/aclnn_modulate.h` | `aclnnModulate*` | C ABI 参数顺序 |
| 入口实现 | `vfusion/modulate/op_host/op_api/aclnn_modulate.cpp` | `CheckParams`, `aclnnModulateGetWorkspaceSize`, `aclnnModulate` | 校验、Contiguous/ViewCopy、executor 与失败路径 |
| Launcher | `vfusion/modulate/op_host/op_api/modulate.cpp` | `ModulateAiCore`, `l0op::Modulate` | 临时输出与 AICore launcher 节点 |
| Host/Tiling | `vfusion/modulate/op_host/modulate_tiling.h` | `ModulateTilingData` | POD 字段与 CompileInfo |
| Host/Tiling | `vfusion/modulate/op_host/modulate_tiling.cpp` | `ModulateTilingOp`, `ModulateTilingFunc` | 平台分支、策略、分核、UB/workspace、key |
| Kernel entry/body | `vfusion/modulate/op_kernel/modulate.cpp` | `modulate`, `ModulateBase/B/L/D` | router、Queue、搬运、Vector、精度、tail |
| 平台注册 | `vfusion/modulate/op_host/modulate_def.cpp` | `ops::Modulate` | A2/A3/950 OpDef 与 dtype |
| binary config | `vfusion/modulate/op_host/config/ascend910b/modulate_binary.json` | `Modulate` op_list | A2 三 dtype 二进制接线 |
| binary config | `vfusion/modulate/op_host/config/ascend910_93/modulate_binary.json` | `Modulate` op_list | A3 三 dtype二进制接线 |
| 构建接线 | `vfusion/modulate/op_kernel/CMakeLists.txt` | `add_kernel_sources` | A2/A3 共用 Kernel、950 独立 arch35 源 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 10个int64声明槽位 | 字段顺序、元素单位、status/useDTiling语义；宏展开布局不可推定 | `ModulateTilingData` |
| 调用顺序 | Contiguous→Modulate→ViewCopy；Kernel 参数→X→Vector→Y | executor stream 顺序；参数 Queue 必须覆盖完整行循环 | `aclnnModulateGetWorkspaceSize`, `CopyAndCompute` |
| workspace/内存 | status0 6份、status1/2 5份 UB；X/Y depth2 | depth 必须计入；BF16有效u减半；LibApi workspace由Host设置 | `ModulateTilingOp::Init`, `ModulateBase` |
| 同步 | Queue 生命周期+PIPE_V barrier | scale原地+1、Mul/Add、BF16 cast之间不得重排 | `PreProcess`, `Compute*` |
| 精度与尾块 | BF16 FP32计算并RINT；其余原dtype；DataCopyPad有效字节写 | 多行仅完整D；D/行组/核尾长度必须保持 | `Compute*`, `CopyIn/CopyOut` |
| 失败边界 | 参数检查失败阻断任务；unknown key不写 | optional null解引用缺陷不可隐藏；不得假设fallback | `CheckShape`, `modulate` |