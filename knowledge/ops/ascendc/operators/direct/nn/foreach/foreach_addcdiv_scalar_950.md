---
type: Operator
title: "ForeachAddcdivScalar 950 Direct Invoke 算子知识"
description: "说明 cann_bench::foreach_addcdiv_scalar 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_host/arch35/foreach_addcdiv_scalar_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/arch35/foreach_addcdiv_scalar_simt.h
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/foreach/foreach_addcdiv_scalar/op_kernel/CMakeLists.txt
    role: implementation
created_at: "2026-08-24T19:33:16Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ForeachAddcdivScalar 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | `cann_bench::foreach_addcdiv_scalar`；PrivateUse1 实现 `foreach_addcdiv_scalar_npu`；另有六个 Host 可调用的 `launch_foreach_addcdiv_scalar_{supplemental,adapted}_{fp16,fp32,bf16}`。 |
| 计算语义 | 对列表中每组同形 Tensor 逐元素计算 `y=x1+scalar*(x2/x3)`，先除后乘加。 |
| 输入输出 | `x1/x2/x3/y` 为等长 ND TensorList，对应 shape 相同，统一 FP16/FP32/BF16；scalar 在 Host 收窄为 FP32 值。 |
| 执行类型 | 单 Device Kernel；R-001～R-003 为 AIV-only RegBase/MicroAPI；R-004～R-009 为 AIV wrapper 内 SIMT VF；无 Cube/AIC 或 MIX_CV。 |
| Kernel 路线 | P0 active：默认 RegBase（R-001/R-002/R-003）；P1 conditional：显式 adapted SIMT32（R-004/R-006/R-008）；P2 conditional：同 dtype 的 SIMT64 宽索引 fallback（R-005/R-007/R-009）。dtype 叶互斥同级，不作跨 dtype 排序。 |
| 核心资源 | RegBase 使用四个 depth=2 TQue 与 FP32 RegTensor；SIMT 直访 GM、使用线程寄存器；workspace 均为 0。 |
| 来源闭包 | `direct_implementation_closed`：PrivateUse1、descriptor、tiling/POD、六个 launcher、两执行族 body 与构建接线闭合；官方固定 Commit 仅作 950/arch35 相关映射。 |
| 证据边界 | 默认/适配 family、九个可用叶和 dormant faithful entry 均由 Direct 源码闭合；固定官方源码只支撑 950/arch35 逻辑映射。 |

## 接口与语义

### 计算语义

对 `0<=i<N`、`0<=j<E_i`：

```text
s = float32(scalar)                  // Host 恰好收窄一次
q = float32(x2[i][j]) / float32(x3[i][j])
r = float32(x1[i][j]) + s * q
y[i][j] = cast<T>(r)                 // T=float 时该 cast 为恒等
```

运算次序契约是先除法、再以商为 `Axpy` 的乘数做加法，不能改写成 `(x1*x3+s*x2)/x3`。FP16/BF16 路线把三输入逐元素提升到 FP32，计算后只在最终写回时降精度；scalar 始终是 FP32，不先量化为输入 dtype。实现没有对零分母、NaN、Inf、subnormal 或 scalar 有限性增加特殊分支，结果遵循 Device 浮点原语。

这里唯一的广播是 scalar 的逻辑广播：一个 Host 属性值供所有列表项、所有元素和所有 AIV block 复用。`x1/x2/x3` 之间没有 shape broadcast、dtype promote 或跨列表项配对；第 `i` 个输出只读取三个列表的第 `i` 项。scalar 不进入 TensorList descriptor、不占 GM workspace，也不按 tensor 展开。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x1` | 列表长 1～64；每项 rank 1～8、`1<=numel<=64*1024*1024`；NPU 路线要求 contiguous ND | 整个三列表统一为 FP16/FP32/BF16 之一 | 加法左输入；以 `x1[0]` 的 device/dtype 为全局基准。 |
| `x2` | 与 x1 等长；`shape(x2[i])==shape(x1[i])`；NPU 路线 contiguous | 同 x1 | 除法分子；不允许广播。 |
| `x3` | 与 x1 等长；`shape(x3[i])==shape(x1[i])`；NPU 路线 contiguous | 同 x1 | 除法分母；不允许广播。 |
| `scalar` | 无 shape，是 schema 属性而非 Tensor | dispatcher 以 C++ `double` 接收，再显式 `static_cast<float>` | 全列表共享，无默认值；不检查范围和有限性。 |
| `y` | 与 x1 等长；逐项 `empty_like(x1[i])` | 同 x1 | 新分配、非原地；NPU 有效域被完整写出。 |

Meta 路线执行同样的列表长度、统一 dtype、对应 shape、rank、numel 和 device 校验，但不要求 contiguous；PrivateUse1 在取地址前追加三输入 contiguous 检查。输出没有复用任何输入地址，因此 Kernel 可按每元素唯一 writer 推理，不需要处理输入输出别名。

### TensorList 的双轨元数据契约

Device 侧同时依赖两套互补元数据，不能只复制其中一套：

| 元数据 | 携带内容 | 不携带内容 | 消费位置 |
|---|---|---|---|
| 四份 `ListTensorDesc` | x1/x2/x3/y 各自按列表顺序排列的 N 个 GM 首地址 | dtype、shape、stride、numel、分核边界 | `ListTensorDesc::GetDataPtr<T>(i)` |
| `ForeachAddcdivScalarTilingData` | N、每项 `numel`、总元素数、每核的列表起止项和 inclusive offset、RegBase tile 容量 | 各 Tensor GM 地址、shape/stride、scalar | AIV block 的遍历与 tile 划分 |

对任一 `i`，四份 descriptor 的第 `i` 个地址必须与 POD 的 `tensorDataCountList[i]` 属于同一 zip 槽位；descriptor 自身无法发现错序、错 dtype 或错长度。四份 descriptor 必须有相同 N，Kernel 模板 suffix 必须与实际 GM dtype 相同，POD 的 N/长度必须与 descriptor 匹配。违反这些不变量不是可恢复输入错误，而是直接调用 launcher 时的 ABI 越界风险。

pointer-only descriptor 的 `3+N` 个 64-bit word 为：

```text
word[0] = 24                         // 地址数组相对 descriptor 首址的 byte offset
word[1] = uint64(N) << 32            // 低 32 位 dim=0，高 32 位 count=N
word[2] = 0x00000000ffffffff         // SDK 要求的 sentinel
word[3+i] = uint64(data_ptr(list[i]))
```

Host 把四份等长 descriptor 连续放在同一 UInt8 Device Tensor 中；若单份字节数 `D=8*(3+N)`，其首址偏移依次为 `0,D,2D,3D`。descriptor 没有 stride 信息，因此 contiguous 校验是把线性 `base+j` 解释成逻辑元素 `j` 的必要前置条件。

### 入口与版本

`TORCH_LIBRARY_FRAGMENT` 声明 `cann_bench::foreach_addcdiv_scalar`；Meta 只校验并构造输出元信息，PrivateUse1 绑定 `foreach_addcdiv_scalar_npu`。这不是 ACLNN 入口：官方固定 Commit 的 scalar 是 Tensor、950 arch35 body 为 SIMT；转换交付把 scalar 改成按值 FP32，并增加 supplemental RegBase。官方 README 明确 Ascend 950PR/950DT 支持该逻辑算子，official `op_kernel/CMakeLists.txt` 把 arch35 接到 `ascend950`，没有平台冲突；这些事实不把转换 ABI 记作官方 ABI。

## Host/Launcher 执行流程

### 参数校验与前处理

默认入口的顺序必须保持为：

```text
ValidateLists
  -> OptionalDeviceGuard(x1[0].device)
  -> current NPU stream
  -> dtypeTag: fp16=0, fp32=1, bf16=2
  -> 逐项 empty_like + 收集 numel + 打包四份 descriptor
  -> calc_tiling(elementCounts, N, dtypeTag)
  -> 分配 descriptorDevice，计算四个 slice 首址
  -> scalar double -> float
  -> RunOpApi callback: H2D descriptor copy -> supplemental dtype launcher
```

校验、tiling、`blockDim>=1` 检查都发生在 callback/launch 前；H2D copy 返回非成功状态时在 callback 内抛错，因此 launcher 不执行。dtype selector 只在已知的 0/1/2 上运行，最终 `else` 才可解释为 BF16，不能把未知 tag 也当成 BF16。

### Host checked arithmetic 与容量证明

| 表达式 | 默认入口可证明范围 | 实现中的检查/缺口 |
|---|---|---|
| `N` | `[1,64]` | 入口显式检查；低于 POD 的 256 槽容量。 |
| `descriptorWords=3+N` | `[4,67]`；四份共不超过 268 word | 由 N 上界间接证明，`N<<32` 不溢出。 |
| 单项 `E_i` | `[1,2^26]` | 入口显式检查；tiler 自身仅检查正数。 |
| `E=sum(E_i)` | `[1,2^32]` | 默认入口由 N 与单项上界证明 int64 安全；通用 tiler 的逐项累加没有 checked-add。 |
| `needCoreNum=clamp(ceil(E/1024),1,C)` | 正数且不超过平台 AIV 核数 C | `C>0` 已检查；固定数组只有 80 槽，代码未显式检查 `C<=80`。 |
| `M=floor((U/(8s))*s/32)*32/s` | 32B 对齐的正元素数 | 检查 `U>128KiB`、`M>0`、`M<=UINT32_MAX`；未显式检查 `M*s` 转为 DMA uint32 字节数。950 的实际 UB 规模使该窄化可行，但移植时应重新证明。 |
| POD 值传 | 自然布局 `sizeof=3672,alignof=8` | 编译参数上限 4096B；必须以编译期 `static_assert`/同编译器布局复核，不能采用 CMake 注释中的 3664B。 |

`totalDataCount==0` 的 tiler 分支对默认入口不可达，因为每项必须非空；若将 tiler 独立复用，空域会得到 `blockDim=0`，调用方必须阻断 launch。

### 参数、Workspace 与生命周期

总辅助 GM 只有四份 descriptor 的 `4D=32*(3+N)` bytes；workspace 为 0。Host descriptor vector 和 Device descriptor Tensor 被 callback 捕获，H2D copy 与 Kernel 使用同一 stream，因而 copy 先于 descriptor 读取。`OpCommand::RunOpApi` 对异步完成期的对象保活是框架边界：若脱离该框架直接复用 launcher，调用方必须显式保证四份 descriptor 与所有输入/输出 Tensor 至 Kernel 完成仍有效。

launcher 解引用 Host POD，再把 3672B 结构按值送入 entry；Kernel 不从 workspace 或 GM tiling pointer 取 POD。POD、launcher 声明和 Kernel entry 必须用同一字段顺序、类型和 ABI 对齐编译。scalar 同样按值进入 entry，因此没有 scalar Tensor 的生命周期边。

### 启动图与失败传播

```text
PrivateUse1
  -> validate/build outputs and 4 descriptors
  -> calc_tiling
  -> H2D descriptor copy (current stream)
  -> launch_supplemental_{dtype}
     -> one AIV grid(blockDim) -> RegBase Process<T>

独立 Host Direct Invoke
  -> launch_adapted_{dtype}
     -> one AIV grid(blockDim) -> Process<T> -> per-segment SIMT VF_CALL<T,INDEX_T>
```

| 边 | 交接介质 | ordering/owner/lifetime | 失败传播 |
|---|---|---|---|
| Host descriptor→Device | 单 GM Tensor 的四个 D-byte slice | 同 stream；callback 捕获对象，框架完成期保活 | copy 失败抛出，Kernel 不启动 |
| tiling→entry | 3672B POD 值副本 | Host 先清零并填充，launcher 解引用后按值传 | tiling/零 blockDim 阻断默认入口 |
| scalar→entry/VF | 一个 FP32 值 | entry 参数按值；RegBase 保存为成员，adapted 继续按值传入 VF | 无 GM 生命周期；非有限值不拦截 |
| Kernel→y | 每个输出 Tensor 的 `[0,E_i)` | 当前 stream；PyTorch Tensor owner | launcher 为 void，无显式 Device 状态回传 |

六个 `launch_*` 均为 Host-visible `extern "C"`。PrivateUse1 只选三个 supplemental；三个 adapted 仅是可由其它 Host 接线显式调用的并列入口。三个 faithful Device entry 虽参与编译，却没有 Host-visible launcher，而且仍要求 scalar GM Tensor ABI，不属于本卡 attribute 接口的可达 Route。

## Route 与参数设计

### 分核与资源计算

Host 只在分核时把 TensorList 看成长度为 `E=sum(E_i)` 的逻辑拼接流；Device 仍按每个 Tensor 的独立 GM 首址访问，绝不能把不同 Tensor 的物理地址视作连续。令 `s=sizeof(T)`、`B=32/s`、`C=GetCoreNumAiv()`：

```text
Q = ceil(E/B)                                      // 逻辑 32B block 数
K = min(ceil(E/1024), C)，且默认入口下 K>=1
q = floor(Q/K), r = Q mod K
nominal(k) = (q + (k<r ? 1 : 0))*B                // 每核名义元素容量
global_begin(k) = sum_{h<k} nominal(h)
global_end(k) = min(E, global_begin(k)+nominal(k)) // exclusive
```

`AssignDataToEachCore` 再把每个 `[global_begin,global_end)` 映射回：`tensorStartList[k]`、`tensorStartOffsetList[k]`、`tensorEndList[k]`、`tensorEndOffsetList[k]`。四个 offset 都以 element 为单位，POD 的 end offset 是 inclusive；Device 恢复段长时统一使用 `cursorEnd-cursorStart+1`。

这个映射允许两种情况同时存在：一个大 Tensor 被相邻 core 切开；一个 core 顺次处理多个小 Tensor。每个逻辑元素只落入一个 core 区间，因此不需要 atomic 或跨核同步。前 `r` 个 core 多一个 32B block；只允许最后一个有效区间短于名义容量，尾部 padding 不是可写元素。PrivateUse1 排除空 Tensor，POD tiler 容量 256，而入口进一步限为 64。

RegBase 令平台 UB 为 U：

```text
M0=floor(U/(2*4*s))
M=floor(M0*s/32)*32/s
UB_total=4 queues*depth2*M*s=8*M*s<=U
```

tiler 只检查 `U>128KiB`，未从 U 扣除该常量；SIMT 不消费 M。

### 核内任务映射

```text
block k
  for tensor i in [tensorStart[k], tensorEnd[k]]:
    begin = (i==tensorStart[k]) ? tensorStartOffset[k] : 0
    end   = (i==tensorEnd[k])   ? tensorEndOffset[k]   : E_i-1
    len   = end-begin+1
    resolve x1/x2/x3/y pointer i from four descriptors
    RegBase: for tile t in [0,ceil(len/M)) -> DMA/MicroAPI/DMA
    adapted: one VF_CALL for this segment -> thread-stride GM loop
```

block-level 分片已体现在 `begin/end`；adapted VF 内索引必须从 0 到该 segment 的 `len`，只做 thread stride，不能再次叠加 `blockIdx` stride，否则非零 block 会跳数。RegBase 的 GM offset 是 `begin+t*M`，并在每次切换 tensor 时重新解析四个 descriptor 地址。

### Route 选择

Route 先按生产可达性和 fallback 价值分 family；同一 family 内的 dtype 是互斥模板实例，不存在源码支持的 FP16/FP32/BF16 重要性全序。

| 优先级 | family/状态 | 覆盖与价值 | selector 与退化关系 |
|---:|---|---|---|
| P0 | supplemental RegBase / active | PrivateUse1 对整个合法 dtype 域的唯一生产主干；使用 UB/MicroAPI | 已校验 `dtypeTag` 互斥选择 R-001/R-002/R-003；任一路失败即报错，不自动退到 adapted |
| P1 | adapted SIMT32 / conditional | Host 显式调用 adapted 时的常规正长度 segment；GM 直访、uint32 索引 | 每个 dtype 内 `0<dataCount<=INT32_MAX` 选择 R-004/R-006/R-008；更大 segment 转 P2 同 dtype 叶 |
| P2 | adapted SIMT64 / conditional fallback | adapted 的宽索引安全兜底，避免 `dataCount` 窄化 | 每个 dtype 内 `dataCount>INT32_MAX` 选择 R-005/R-007/R-009；无更宽 fallback |

PrivateUse1 源码中的 `if fp16 / else if fp32 / else bf16` 只是互斥分支书写顺序。adapted 也先由 launcher suffix 固定 dtype，再在单个 segment 内判断索引位宽；两层 selector 不能重排为跨 dtype 优先级。

### 九个 active/conditional 叶的完整分发闭包

| Route | 状态 | 精确触发谓词 | Host producer/selector | launch 与 Device entry | body/template 与字段 delta | tail/precision/fallback |
|---|---|---|---|---|---|---|
| R-001 FP16 RegBase | active | PrivateUse1；统一 dtype=FP16；validation、tiling、copy 成功 | `dtypeTag=0`；`dataTypeSize=2` 生成 M/core range | `launch_supplemental_fp16`→`foreach_addcdiv_scalar_supplemental_fp16` | `RegBase<half>`；descriptor 指向 half；DMA=`count*2B`；POD schema 不变 | FP32 promotion，`CAST_RINT` 回 half；Vector mask+精确 DMA；失败终止 |
| R-002 FP32 RegBase | active | PrivateUse1；统一 dtype=FP32；validation、tiling、copy 成功 | `dtypeTag=1`；`dataTypeSize=4` 生成 M/core range | `launch_supplemental_fp32`→`foreach_addcdiv_scalar_supplemental_fp32` | `RegBase<float>`；descriptor 指向 float；DMA=`count*4B`；POD schema 不变 | 全 FP32；Vector mask+精确 DMA；失败终止 |
| R-003 BF16 RegBase | active | PrivateUse1；统一 dtype=BF16；validation、tiling、copy 成功 | 已校验 tag 的最终分支 `dtypeTag=2`；`dataTypeSize=2` 生成 M/core range | `launch_supplemental_bf16`→`foreach_addcdiv_scalar_supplemental_bf16` | `RegBase<bfloat16_t>`；descriptor 指向 BF16；DMA=`count*2B`；POD schema 不变 | FP32 promotion，`CAST_RINT` 回 BF16；Vector mask+精确 DMA；失败终止 |
| R-004 FP16 SIMT32 | conditional | 显式 adapted_fp16；当前 segment `0<len<=INT32_MAX` | 调用方生产合法 descriptor/POD；suffix 固定 FP16，Process 检查 len | `launch_adapted_fp16`→`foreach_addcdiv_scalar_adapted_fp16` | `SimtKernel<half,uint32_t>`；1024 threads；int64 POD range 保持，M inactive | FP32 promotion 后回 half；`i<len`；len 更大→R-005 |
| R-006 FP32 SIMT32 | conditional | 显式 adapted_fp32；当前 segment `0<len<=INT32_MAX` | 调用方生产合法 descriptor/POD；suffix 固定 FP32，Process 检查 len | `launch_adapted_fp32`→`foreach_addcdiv_scalar_adapted_fp32` | `SimtKernel<float,uint32_t>`；1024 threads；int64 POD range 保持，M inactive | 全 FP32；`i<len`；len 更大→R-007 |
| R-008 BF16 SIMT32 | conditional | 显式 adapted_bf16；当前 segment `0<len<=INT32_MAX` | 调用方生产合法 descriptor/POD；suffix 固定 BF16，Process 检查 len | `launch_adapted_bf16`→`foreach_addcdiv_scalar_adapted_bf16` | `SimtKernel<bfloat16_t,uint32_t>`；1024 threads；int64 POD range 保持，M inactive | FP32 promotion 后回 BF16；`i<len`；len 更大→R-009 |
| R-005 FP16 SIMT64 | conditional fallback | 显式 adapted_fp16；当前 segment `len>INT32_MAX` | 调用方以 fp16 suffix 固定 T；Process 的 uint32 谓词未命中 | `launch_adapted_fp16`→`foreach_addcdiv_scalar_adapted_fp16` | `SimtKernel<half,int64_t>`；512 threads；POD range 保持 int64，M inactive | FP32 promotion 后回 half；int64 `i<len`；无更宽 fallback |
| R-007 FP32 SIMT64 | conditional fallback | 显式 adapted_fp32；当前 segment `len>INT32_MAX` | 调用方以 fp32 suffix 固定 T；Process 的 uint32 谓词未命中 | `launch_adapted_fp32`→`foreach_addcdiv_scalar_adapted_fp32` | `SimtKernel<float,int64_t>`；512 threads；POD range 保持 int64，M inactive | 全 FP32；int64 `i<len`；无更宽 fallback |
| R-009 BF16 SIMT64 | conditional fallback | 显式 adapted_bf16；当前 segment `len>INT32_MAX` | 调用方以 bf16 suffix 固定 T；Process 的 uint32 谓词未命中 | `launch_adapted_bf16`→`foreach_addcdiv_scalar_adapted_bf16` | `SimtKernel<bfloat16_t,int64_t>`；512 threads；POD range 保持 int64，M inactive | FP32 promotion 后回 BF16；int64 `i<len`；无更宽 fallback |

默认入口把每项限制到 `2^26` 元素，所以其 POD 即使被显式交给 adapted 也只命中三个 SIMT32 叶；三个 SIMT64 叶服务于能够生产合法宽 segment、并补齐 checked arithmetic/lifetime 的独立 Host 调用方。它们是受控 fallback，不是 PrivateUse1 自动退路。

### Fallback 与跨 Route 不变量

- 默认 RegBase 的 validation、tiling、copy 或 launch 前置条件失败时终止；没有 adapted、框架或 CPU 自动 fallback。
- adapted 的唯一 fallback 边是同 dtype 的 SIMT32→SIMT64。dtype、scalar FP32 值 ABI、四 descriptor zip 顺序、POD core range 都不改变。
- 所有九叶保持先除后乘加、B16 的 FP32 promotion、每元素唯一 writer 和只写有效域。MicroAPI `Axpy`、SIMT 表达式收缩或最终 cast 原语可能带来末位差异，不能承诺 family 间 bitwise 等价。
- SIMT64 已是最宽索引叶；RegBase repeat/DMA 窄化也没有 Device 补救。越过 Host 已证明范围必须拒绝，不能截断。

### Dormant faithful family

`foreach_addcdiv_scalar_faithful_{fp16,fp32,bf16}` 三个 entry 参与编译，但没有 Host-visible launcher，生产链不可达；其 scalar 参数还是 `GM_ADDR`，Device 按 T 读取 `scalarPtr[0]`，与本卡 attribute 的共享 FP32 值 ABI 不同。重新启用至少需要：新增匹配 launcher、构造并保活 T dtype scalar GM Tensor、保持 descriptor/POD ABI，并为 selector 定义与 active RegBase 的明确关系。不能把 FP32 值地址强转为 faithful scalar 参数。

### 性能与安全 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 950 build | `ascend950->dav-3510` | arch35/MicroAPI/SIMT 目标 | 全部 | 配置失败 | top CMake |
| POD 上限 | 4096 | 值传 POD 推导 3672B | 全部 | 编译器拒绝 | operator CMake |
| UB | `coreNum>0,U>128KiB,M>0,M<=UINT32_MAX` | 四 Queue 双缓冲 | R-001/R-002/R-003 | tiling 失败 | `calc_*_tiling` |
| 容量 | PrivateUse1 N<=64；POD N<=256；per-core array 80 | 固定数组索引 | 全部 | 前两者失败；`coreNum<=80` 需平台保证，代码未检查 | Validate/POD |
| repeat 窄化 | repeats→uint16 | MicroAPI 循环 | R-001/R-002/R-003 | 无显式 guard | RegBase Compute |
| SIMT 位宽 | dataCount 与 INT32_MAX 比较 | u32 1024 threads/i64 512 threads | R-004/R-006/R-008→R-005/R-007/R-009 | 同 dtype i64 | adapted Process |

### POD/ABI

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| tiling | `needCoreNum` | int32 | off 0 | core | clamp core 公式 | grid 间接消费 | 数组仅 80 |
| tiling | `tensorCount` | int32 | off 4 | tensor | N | body 不读 | 入口 64/POD 256 |
| tiling | `totalDataCount` | int64 | off 8 | element | sum | SIMT 空域 guard | 求和未查溢出 |
| tiling | `inputsTensorUbSize` | uint32 | off 16 | element | M | RegBase tile | SIMT inactive |
| tiling | `reserved` | uint32 | off 20 | N/A | 清零 | 不读 | 保持 0 |
| tiling | `tensorDataCountList[256]` | int64[] | off 24, 2048B | element | numel | tensor tail | 未用项 0 |
| tiling | `tensorStartList[80]` | uint16[] | off 2072, 160B | tensor/core | core splitter | blockIdx 索引 | inclusive range |
| tiling | `tensorEndList[80]` | uint16[] | off 2232, 160B | tensor/core | 同上 | 同上 | inclusive |
| tiling | `tensorStartOffsetList[80]` | int64[] | off 2392, 640B | element/core | 同上 | 首段 | inclusive |
| tiling | `tensorEndOffsetList[80]` | int64[] | off 3032, 640B | element/core | 同上 | 尾段 | inclusive |
| descriptor | word0/word1/word2 | uint64×3 | off 0/8/16 | byte/count/sentinel | 24、N<<32、0xffffffff | SDK parser | 固定 header |
| descriptor | `data[i]` | uint64 | off `24+8i` | GM addr | `data_ptr` | `GetDataPtr<T>` | 生命周期覆盖 launch |

POD 以 `{}` 初始化并在 tiler 内再次 `memset`，所以 reserved 与所有未用数组槽为 0；Device 只允许以 `blockIdx<K` 读取前 K 个 core 槽。`tensorCount` 主要用于 Host 生成数组，Device 遍历以每核 start/end 为准；它不能替代 descriptor 内 N。descriptor header、POD `tensorCount`、四份地址槽数三者应视作一组 ABI 原子更新。

### 对齐、DMA 与尾块

对齐存在三个不同层次，不应混为“所有 Tensor 地址/长度都 32B 对齐”：

| 层次 | 对齐事实 | 尾部处理 |
|---|---|---|
| Host 分核 | 逻辑拼接流按 `B=32/s` 元素分 block，除总尾外每核名义容量为 32B 倍数 | 最后一个 core 只记录真实 inclusive end，不把 padding 写入任何 Tensor |
| RegBase tile | `M*s` 向下对齐到 32B；完整 tile 是 32B 倍数 | 最后 tile 的 `currentCount=min(M,remaining)` 可为任意正数 |
| Tensor segment | Tensor 边界可落在任意 element offset；一个 core 跨 Tensor 时每段独立取 GM base | `DataCopyPad` 以 `currentCount*s` 的真实字节数搬入/搬出；不对 GM 越界补齐 |
| Vector repeat | 每 repeat 覆盖 `VECTOR_REG_WIDTH/4` 个 FP32 lane，B16 也以 FP32 lane 数推进 | `UpdateMask<float>(remaining)` 逐 repeat 产生有效 lane mask；load/cast/div/axpy/store 共用 mask |
| SIMT | 不依赖 32B tile；每个 segment 单独发 VF | `for(i=tid;i<len;i+=threadNum)` 的 `i<len` 是唯一尾保护 |

RegBase 的 Queue 顺序是三次 `AllocTensor/DataCopyPad/EnQue`，随后三次 `DeQue` 与一次 y allocation，计算完成后释放输入并把 y `EnQue`，最后 `DeQue/DataCopyPad/FreeTensor`。Queue event 建立 DMA→Vector→DMA 的核内顺序；双缓冲 depth=2 提供资源槽，但当前 `CopyIn->Compute->CopyOut` 调用序列没有显式软件流水重叠。SIMT 只有线程程序序和唯一 writer，不使用 barrier、atomic、L1/L0 或跨核通信。

### Family 源码闭包

| family | Host/tiling producer | launch/entry | body | 构建接线 |
|---|---|---|---|---|
| P0 active RegBase | plugin validation/descriptor + `calc_*_tiling` | 三个 supplemental launcher/entry | 一个 `RegBase<T>`；三 dtype 实例 | operator CMake + 4096B 参数上限 |
| P1/P2 conditional adapted | 独立调用方提供同 ABI descriptor/POD/stream；`Process<T>` 做索引分支 | 三个 adapted launcher/entry | 一个 `SimtKernel<T,INDEX_T>`；3×2 实例 | 同上 |
| dormant faithful | 无 Host producer/launcher | 三个 faithful entry，仅 Device 符号 | faithful SIMT，scalar GM ABI | 编译接入但生产不可达 |

## Kernel 实现

### Family P0（active）：supplemental RegBase

#### Selector、entry 与字段实例化

PrivateUse1 完成统一 dtype 校验、POD 生成和 descriptor H2D 后，按互斥 tag 选择一个 entry：

| dtype 叶 | tag | entry | T | s/B | load/store delta |
|---|---:|---|---|---|---|
| R-001 | 0 | `foreach_addcdiv_scalar_supplemental_fp16` | `half` | 2B / 16 element | unpack half→FP32；`CAST_RINT`+pack 回 half |
| R-002 | 1 | `foreach_addcdiv_scalar_supplemental_fp32` | `float` | 4B / 8 element | FP32 register load/store，无 dtype cast |
| R-003 | 2 | `foreach_addcdiv_scalar_supplemental_bf16` | `bfloat16_t` | 2B / 16 element | unpack BF16→FP32；`CAST_RINT`+pack 回 BF16 |

三叶共享同一 POD schema、四份 descriptor 和 `RegBase<T>` body；`s` 同时决定 Host 分核的 `B=32/s`、tiler 的 M、Queue 字节数和 DMA 字节数。entry 创建 `TPipe`，再把按值 scalar/POD 交给 `Process<T>`。

#### Body、地址与循环骨架

```text
k = blockIdx
for i = tensorStart[k] .. tensorEnd[k]:
  begin = (i == tensorStart[k]) ? startOffset[k] : 0
  end   = (i == tensorEnd[k])   ? endOffset[k]   : E_i-1
  len   = end-begin+1
  x1/x2/x3/y_gm = descriptor[i] + begin

  for t = 0 .. ceil(len/M)-1:
    count = min(M, len-t*M)
    DMA exact count*s bytes: x1/x2/x3 GM -> three VECIN queues
    for repeat over ceil(count/(VECTOR_REG_WIDTH/4)):
      mask = UpdateMask<float>(remaining)
      a,b,c = LoadAsFloat<T>(..., mask)
      b = Div(b, c, mask)
      a = Axpy(a, b, scalarVal, mask)
      StoreFromFloat<T>(y, a, mask)
    DMA exact count*s bytes: VECOUT queue -> y GM
```

`cursorEnd` 是 inclusive，故段长必须加 1；切换 Tensor 时重新解析 descriptor 地址，不能跨不同 GM allocation 连续搬运。`UpdateMask<float>` 的 remaining 由 API 逐 repeat 推进，load/cast/div/axpy/store 共用同一 mask。

#### GM/UB/L1/L0、同步与生命周期

- 四个 depth=2 Queue 各占 `2*M*s`，总显式 UB 为 `8*M*s<=U`；三个输入从 GM 经 `DataCopyPad` 到 VECIN，FP32 RegTensor 计算后经 VECOUT 回 GM。
- 无 L1/L0、dynamic UB、workspace、atomic 或跨核通信。Queue event 建立 DMA→Vector→DMA 顺序；代码按 `CopyIn->Compute->CopyOut` 串行调用，没有显式跨 tile 软件流水重叠。
- scalar 与 POD 都按值进入 Kernel；四份 descriptor 和 Tensor storage 必须活到同 stream Kernel 完成。默认框架路径依赖 `RunOpApi`/allocator 的异步资源语义，独立 launcher 调用方必须自行保活。

#### Tail、精度与 fallback

- 完整 tile 的 `M*s` 为 32B 倍数；Tensor 边界和最后 tile 可以任意 element 对齐。`DataCopyPad` 只搬真实字节，Vector mask 再限制尾 lane，二者共同禁止尾后读写。
- FP16/BF16 只在 load/store 边界转换，Div 与 Axpy 均为 FP32；scalar 不量化为 T。FP32 叶全程 FP32。零分母、NaN/Inf/subnormal 不设特判。
- P0 没有 Device 或 Host 自动 fallback。validation、tiling、copy 或 launch 前置条件失败即终止，不能静默转 adapted。

### Family P1/P2（conditional）：adapted SIMT

#### Selector、entry 与九叶中的六个实例

adapted 不是 PrivateUse1 的候补分支；调用方先以 launcher suffix 固定 T，再由共享 `Process<T>` 对每个正长度 segment 选择 INDEX_T：

| dtype | launcher/entry | P1 SIMT32 | P2 SIMT64 | dtype 精度 delta |
|---|---|---|---|---|
| FP16 | `launch_adapted_fp16` / `foreach_addcdiv_scalar_adapted_fp16` | R-004：`SimtKernel<half,uint32_t>` | R-005：`SimtKernel<half,int64_t>` | half→FP32，最终回 half |
| FP32 | `launch_adapted_fp32` / `foreach_addcdiv_scalar_adapted_fp32` | R-006：`SimtKernel<float,uint32_t>` | R-007：`SimtKernel<float,int64_t>` | 全 FP32 |
| BF16 | `launch_adapted_bf16` / `foreach_addcdiv_scalar_adapted_bf16` | R-008：`SimtKernel<bfloat16_t,uint32_t>` | R-009：`SimtKernel<bfloat16_t,int64_t>` | BF16→FP32，最终回 BF16 |

INDEX_T 不改变 POD 字段：`tensorDataCountList`、start/end offset 始终为 int64；它只改变 VF 的 `dataCount/i/step` 类型、launch bound 和线程寄存器宽度。`inputsTensorUbSize=M` 在 adapted 中存在于共享 POD，但不消费。

#### Body、索引 fallback 与任务骨架

```text
Process<T>(descriptor4, scalarVal, pod):
  k = blockIdx
  for each tensor segment assigned to k:
    len = inclusiveEnd-begin+1
    if len <= 0: continue
    resolve four typed GM pointers at descriptor[i] + begin

    if len <= INT32_MAX:
      VF_CALL<SimtKernel<T,uint32_t>>(threads=1024, uint32(len), scalarVal, ptrs)  // P1
    else:
      VF_CALL<SimtKernel<T,int64_t>>(threads=512, int64(len), scalarVal, ptrs)    // P2

SimtKernel<T,INDEX_T>:
  for i = INDEX_T(threadIdx); i < len; i += INDEX_T(threadNum):
    a,b,c = cast_to_fp32(x1[i],x2[i],x3[i])
    y[i] = cast_to_T(a + scalarVal * (b/c))
```

block-level 分片已经体现在 GM base 与 len 中，VF 只能使用 thread stride；再次叠加 block stride 会让非零 block 跳过数据。默认入口的单 Tensor 上界是 `2^26`，因此其 POD 只会触发 P1；P2 要求独立 Host 对更宽合法 segment 补齐 descriptor、checked sum、core 数组容量和生命周期证明。

#### GM/UB/L1/L0、同步、tail 与精度

- 四输入/输出均为 GM 直访，只有线程私有 FP32 value 和 INDEX_T 索引寄存器；无显式 UB Queue/TBuf、L1/L0、dynamic UB、workspace、barrier 或 atomic。
- 每个 segment 由一个 block 的一次 VF_CALL 处理，`i<len` 是唯一尾保护；32/64 位叶都不依赖 32B DMA 对齐。
- FP16/BF16 的三个输入先转 FP32，按 `a+s*(b/c)` 计算后最终转回 T；FP32 直接计算。与 RegBase 的 Div+Axpy 语义顺序一致，但不同原语不提供 bitwise 等价保证。
- P1 的唯一 fallback 是同 dtype P2，selector 先证明 `len<=INT32_MAX` 才窄化。P2 无更宽 fallback，不得把 int64 len 截断。

### 两个 active/conditional family 的共同不变量

| 契约 | RegBase | adapted SIMT |
|---|---|---|
| Host/POD | 默认 plugin 生产；POD 3672B 按值 | 独立调用方生产同 ABI POD；M 字段 inactive |
| block→Tensor 映射 | 同一组 inclusive start/end arrays | 同左 |
| descriptor | 四份 pointer-only descriptor，zip 顺序和 dtype suffix 必须一致 | 同左 |
| 数学 | 显式 Div 后 Axpy | C 表达式保持 `x1+s*(x2/x3)` |
| writer/sync | 每元素唯一 writer；Queue 核内排序 | 每元素唯一 writer；线程程序序 |
| 输出域 | mask+真实 DMA 字节 | thread guard |
| family fallback | 无 | SIMT32→同 dtype SIMT64 |

## 模板与实现对比

| 对比维度 | P0 R-001/R-002/R-003 RegBase | P1 R-004/R-006/R-008 SIMT32 | P2 R-005/R-007/R-009 SIMT64 | 设计选择依据 |
|---|---|---|---|---|
| 状态/命中条件 | active；PrivateUse1 默认、互斥 dtype 分支 | conditional；显式 adapted+常规段 | conditional fallback；显式 adapted+宽段 | 先按生产覆盖/适配/fallback 分 family；dtype 同级 |
| 模板与 dtype | RegBase T | SIMT T,u32 | SIMT T,i64 | dtype 控制 cast，段宽控制索引 |
| 循环与分核 | core→tensor→tile→repeat | core→tensor→1024-thread stride | 同左、512 threads | UB tile 与 GM direct |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 四 depth2 Queue+FP32 regs | thread regs | thread regs+i64 | 均无通信/workspace |
| 内存总占用与复用 | `8*M*s` | 无显式 UB | 无显式 UB | Queue 生命周期或 thread program order |
| 搬运/Vector/SIMT/NDDMA/通信 API | DataCopyPad+MicroAPI | VF_CALL+GM | VF_CALL+GM | 无 NDDMA/通信 |
| 同步、精度与 tail | Queue+mask；B16 promotion | unique writer+thread guard | 同左、i64 guard | 无 atomic/barrier |
| fallback/失败边界 | 任一前置失败终止 | 同 dtype 大段转 i64 | 无更宽 fallback | void launcher 无状态回传 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方逻辑算子支持；本直调未闭合 | 官方逻辑算子支持；本直调未闭合 | 官方支持，直调交付明确 ascend950/dav-3510 |
| Host/Launcher | 不发布本卡 Route | 不发布本卡 Route | PrivateUse1+六 launcher |
| Kernel/资源 | 其它架构实现不作本卡事实 | 同左 | arch35 AIV RegBase与AIV/SIMT，POD 3672B |

## 限制与失败边界

- PrivateUse1 列表上限 64，POD/tiler 为 256；每 Tensor 连续、rank 1～8、1～64M 元素。绕过入口直接调用 launcher 时，调用方负责 descriptor、dtype、range 与生命周期。
- 独立复用 tiler 时的 int64 总和、`coreNum<=80`、RegBase repeats→uint16 没有显式 guard；默认入口只对总和给出了范围证明。
- RegBase 容量使用完整 U，虽检查 U>128KiB 却不扣该常量；迁移必须保持口径或重做 UB 证明。
- CMake 的 3664B 注释与转换 POD 推导 3672B 不一致；4096 上限覆盖后者，但不能混用官方 POD。
- descriptor copy 与 Kernel 同 stream；框架完成期保活不可见。launcher 返回 void，无显式 launch 状态传播。
- adapted 是显式 Host 调用才可达的 conditional family，不是 PrivateUse1 fallback；faithful entry 无 Host launcher，属于 dormant family。
- Frontmatter 只保留转换所参考的固定官方 HTTP 源码；attribute scalar、RegBase、POD 扩展和 PrivateUse1 接线属于 Direct Invoke 转换事实，不声称来自官方 Commit。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | conversion plugin cpp | schema、Validate、PrivateUse1 | 接口、descriptor、默认 Route |
| Host/Launcher | conversion launch header | calc 与六 launcher | Host 表面 |
| Host/Tiling/entry | conversion kernel cpp | calc、Assign、entries | core/UB/POD/launch |
| POD | conversion tiling data header | tiling class | ABI |
| Kernel body | conversion supplemental header | RegBase | AIV/MicroAPI |
| Kernel body | conversion adapted header | Process/SIMT VF | SIMT |
| 构建接线 | conversion operator CMake | register、4096 | direct 950 build |
| 官方平台 | official README | 支持表 | 950 相关性 |
| 官方 Host | official arch35 tiling cpp | core split | arch35 映射 |
| 官方 Kernel | official arch35 kernel cpp/SIMT header | dtype dispatch、VF | AIV/SIMT 映射 |
| 官方构建 | official op_kernel CMake | ascend950 | 950 接线 |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 3672B POD、pointer-only descriptor | 字段顺序/offset、三字 header、scalar 按值 | tiling/InitDescriptor |
| 调用顺序 | validate→tiling→H2D→launch | 同 stream、失败阻断 | PrivateUse1 |
| workspace/内存 | workspace 0；RegBase `8*M*s` | 32B alignment、四 Queue depth2 | calc/Init |
| 同步 | Queue protocol；SIMT unique writer | Queue 释放顺序；VF 无二次 block stride | RegBase/SIMT |
| 精度与尾块 | B16→FP32；先 Div 后乘加 | CAST_RINT、mask/thread guard、inclusive range | Compute/VF |
| 失败边界 | tiling/copy/validation 失败 | void launcher 不是状态；不混用 POD | plugin/calc/launch |