---
type: Operator
title: "Mish 950 Direct Invoke 算子知识"
description: "说明 cann_bench::mish、mish_npu 与 launch_mish_kernel_{float,half,bfloat16} 在 950 上的接口语义、Host/Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/README.md
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/op_host/arch35/mish_tiling_arch35.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/op_kernel/arch35/mish_struct.h
    role: header
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/op_kernel/mish_apt.cpp
    role: implementation
  - resource: cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/op_kernel/arch35/mish_dag.h
    role: implementation
created_at: "2026-08-25T19:29:31Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# Mish 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | dispatcher `cann_bench::mish` → `mish_npu` → `launch_mish_kernel_float` / `launch_mish_kernel_half` / `launch_mish_kernel_bfloat16`；入口类型为 Direct Invoke。 |
| 计算语义 | `y=x*tanh(softplus(x))`；Direct Device body 使用等价式 `u=exp(min(x,cap))`、`v=u*(u+2)`、`y=x*v/(v+2)`。 |
| 输入输出 | 调用契约为 950 NPU 上连续 FP32/FP16/BF16 Tensor；任意 rank 扁平为 `numel`，输出 shape、dtype 与输入相同。 |
| 执行类型 | 五条 Route 的唯一 Device 节点均为 `AIV_VECTOR`；单 Kernel，无 Cube/AIC、MIX_CV、SIMT、NDDMA 或通信节点。 |
| Kernel 路线 | 重要性顺序：P0 R-001 FP32/PD4；P1 R-002 FP16/PD2；P2 R-004 BF16/PD2；P3 R-005 BF16/PD4；P4 R-003 FP16/PD4。先放无二级 selector 的 FP32，再放两个低 UB 短跨度主干，最后放两个长跨度资源特化；BF16/PD4 因精度转换和最大 UB 风险排在 FP16/PD4 前。Route ID 保持 launcher 映射，不随阅读顺序改号。 |
| 核心资源 | 输入/输出 GM、`VECIN`/`VECOUT` TQue、2～3 个 `VECCALC` TBuf；workspace 为 0，最大静态 UB 为 BF16/PD4 的 229376B。 |
| 来源闭包 | `direct_implementation_closed`：dispatcher、输出分配、分核、dtype/PD selector、按值 launch ABI、五个模板实例、Kernel body、UB/queue/tail 与算子构建注册均已闭合。 |
| 证据边界 | Host、selector、C ABI、五个模板实例与 Device body 已做静态闭包；正文不从构建或运行记录外推额外的数值、时延或吞吐结论。 |

## 接口与语义

### 计算语义

官方相关来源将 Mish 定义为：

```text
softplus(x) = ln(1 + exp(x))
mish(x) = x * tanh(softplus(x))
```

本次 Direct 转换没有直接求 `log` 和 `tanh`，而是逐元素计算：

```text
c = 20.0                 # FP32、BF16；BF16 先升到 FP32
c = half(5.5)            # FP16 原生计算
u = exp(min(x, c))
v = u * (u + 2)
y = x * v / (v + 2)
```

不截断时，令 `u=exp(x)`，则 `tanh(log(1+u))=((1+u)^2-1)/((1+u)^2+1)=u(u+2)/(u(u+2)+2)`，因此上述表达式与 Mish 等价。截断只作用于指数的输入，最后仍乘原始 `x`。FP16 的 5.5 上限使 `exp(x)*(exp(x)+2)` 留在 half 的有限范围；FP32/BF16 计算路径使用 20.0 上限。这里的单 Exp/单 Div、两个 cap 和原生 FP16 路线均是 Direct 转换事实，不是固定官方 `mish_dag.h` 原样实现。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | 任意非空 shape；Direct body 按一维连续数组解释 | FP32/FP16/BF16 | 必须位于 NPU 且连续；实现不执行 `Contiguous`，也不校验 layout。 |
| `y` | `empty_like(x)`，shape/options 与 `x` 相同 | 与 `x` 相同 | 每个有效元素由唯一 AIV core 写一次；无原地别名。 |
| 属性 | N/A | N/A | 无 beta、threshold、axis 等属性。 |

固定官方 README 支持 Ascend 950PR/950DT，声明 ND 的 FLOAT/FLOAT16/BFLOAT16 输入输出和 0～8 维约束。Direct 入口不是该 ACLNN/图入口：它只读取 `numel`，不检查 rank，且依赖调用方保证连续与合法 dtype。

### 入口与版本

插件通过 `TORCH_LIBRARY_FRAGMENT` 声明 `mish(Tensor x) -> Tensor y`，Meta 与 NPU 实现都以 `empty_like(x)` 创建输出；PrivateUse1 绑定到 `mish_npu`。`mish_npu` 安装输入 device guard、取得当前 NPU stream，并用三个 C ABI launcher 直接提交 Ascend C Kernel。`OpCommand::RunOpApi("Mish", acl_call)` 只是提交包装，数值计算不经过 ACLNN 或框架 fallback。

固定提交中的 README、`MishTiling`、`mish` AIV entry 和 `MishDAG` 是数学语义、950 支持、dtype 选择和 AIV 实现族的相关官方来源。本文的 `calc_mish_tiling_params`、三个 `launch_mish_kernel_*`、五个 `KernelMish<T,PD>` 实例、cap、队列深度和 UB 布局均来自最终 Direct 转换实现；二者不得互换归因。

最终 Direct 源码中的 `op_kernel/arch35/mish_dag.h` 是转换审计留档；活动算子 CMake 只把 `op_kernel/mish_kernel.cpp` 加入 kernel source，后者也不 include 该 DAG。因此该审计头不可达，不进入本文 Route、UB 或数值 body 闭包。

## Host/Launcher 执行流程

### 参数校验与前处理

1. dispatcher 进入 `mish_npu(x)`，`OptionalDeviceGuard` 选择 `x.device()` 的上下文。
2. `mish_meta(x)` 调用 `torch::empty_like(x)` 分配 `y`；没有 Contiguous、View、shape 重排或 workspace 分配。
3. 取得当前 NPU stream，读取 `totalLength=x.numel()`，再调用 `calc_mish_tiling_params(totalLength)`。
4. 取 `x.data_ptr()`、`y.data_ptr()`，在 `acl_call` 内按 `x.scalar_type()` 分发到 FP32、FP16 或 BF16 launcher。
5. FP32 固定启动 `mish_kernel<float,4>`；FP16/BF16 再按 `blockLength>=131072` 选择 PD4，否则选 PD2。

Direct 插件没有显式 NPU、连续性、dtype 或非空校验。输入非空且 dtype 为三种受支持类型是正常启动的前置条件；不支持 dtype 会跳过全部分支但 `acl_call` 仍返回 0，空输入会使分核公式走到 `numBlocks=(totalLength+blockLength-1)/blockLength` 的零除路径。

### 参数、Workspace 与生命周期

令 `N=totalLength>0`、`C=max(GetCoreNumAiv(),1)`：

```text
B0 = max(min(C, ceil(N/2048)), 1)
L0 = ceil(N/B0)
L  = align_up(L0, 4096)        # blockLength，元素
B  = ceil(N/L)                 # numBlocks，重新收缩空核
tileSize_host = 4096           # 按值传递，但 Device body 不读取
```

Host tuple `(B,L,4096)` 先保存为三个 `int64_t`，到 C ABI 时把 `tileSize` 窄化为 `uint32_t`；Device 参数在 launch 时按值封送。`y` 由返回 Tensor 持有，`x` 被 callable 按值捕获到提交结束；源码没有 completion event 或显式 `recordStream`，所以输入 storage 覆盖异步 Kernel 完成期仍依赖框架 allocator/current-stream 契约，不能仅由 lambda 生命周期推出。Direct entry 没有 tiling GM、POD workspace 或 dynamic UB 参数：

```text
workspace_total = 0
launch = mish_kernel<T,PD><<<B, nullptr, current_stream>>>(x, y, N, L, 4096)
```

`L` 是元素数而非字节。4096 元素对齐使 FP32 的非末核工作量总是整 tile；窄类型的 8192 tile 则只有在 `L%8192==0` 时整除。若 `L%8192==4096`，每个非末核也会执行一个 4096 元素尾 tile，而不只是最后一核有 tail。重算后的 `B=ceil(N/L)` 保证所有已启动核满足 `b*L<N`，每核实际元素数为 `n_b=min(N-b*L,L)>0`。

### 启动图与失败传播

```text
cann_bench::mish
  -> mish_npu
    -> empty_like + current stream + calc_mish_tiling_params
    -> dtype launcher + optional narrow PD selector
      -> K1: mish_kernel<T,PD><<<numBlocks,nullptr,stream>>>
        -> output GM
```

| 节点 | Host 启动函数 | Kernel entry | blockDim/参数 | 输入→输出 | workspace |
|---|---|---|---|---|---|
| K1 | `launch_mish_kernel_float/half/bfloat16` | `mish_kernel<T,PD>` | `blockDim=B`；`x,y,N,L,tileSize` 按值/指针传入 | `x` GM → `y` GM | N/A，0B |

这是单节点图，不存在 Kernel 间 workspace 边。Host 的 dtype/PD 判断在 launch 前完成；源码没有检查 `<<<>>>` 的异步提交状态，lambda 固定返回 0，因此 Device 异步失败不会由该返回值同步传播。

## Route 与参数设计

### 分核与资源计算

Device 模板由 `T` 与编译期 `PD` 决定：

```text
tile(T) = 4096, T=float
tile(T) = 8192, T=half|bfloat16_t
ComputeT = half, T=half
ComputeT = float, T=float|bfloat16_t
tileNum_b = floor(n_b/tile(T))
tail_b = n_b - tileNum_b*tile(T)
```

每个完整 tile 与尾 tile 都严格执行 `CopyIn→Compute→CopyOut`。Host 返回的 `tileSize=4096` 在 `KernelMish::Init` 的形参中被标记 unused；Device 始终使用 `kInternalTile`，所以复用时不能把 Host tile 当成实际窄类型 tile。

各路线静态 UB 字节数：

```text
queue_pair = 2 * PD * tile * sizeof(T)
scratch = 2 * tile * sizeof(ComputeT)
bf16_extra = tile * sizeof(float) if T=BF16 else 0
UB_total = queue_pair + scratch + bf16_extra
```

### Route 特征与重要性

P0～P4 表示本卡的阅读、维护与复用重要性，不改变 Route ID，也不改变 Host 的 dtype 分发、depth 判断、短路或 fallback。后续 Route 矩阵、源码闭包、逐 Route 章节和实现对比均按此顺序展开。

| 重要性 | Route | Route 特征 | 适用域 | 实现前提 | Host 顺序、短路与 fallback |
|---|---|---|---|---|---|
| P0 | R-001 FP32 PD4 | FP32 主干；4096 元素 tile、固定 depth 4，输入、算术与输出全程 FP32 | 950、连续 FP32、`N>0` | 160KiB 静态 UB 可用 | H1 dtype 首分支；命中后跳过 H2/H3，未命中继续 H2 |
| P1 | R-002 FP16 PD2 | 短跨度 FP16 主干；8192 元素 tile、原生 half 算术、低开销 depth 2 | 950、连续 FP16、`N>0`、`L<131072` | FP16 cap 保持 half(5.5)；96KiB 静态 UB 可用 | H2→H2.1 false 的 `else`；true 短路到 R-003 |
| P2 | R-004 BF16 PD2 | 短跨度 BF16 主干；BF16 搬运、FP32 中间计算、RINT 回写、depth 2 | 950、连续 BF16、`N>0`、`L<131072` | `xFloatBuf_`、`spBuf_`、`tmpBuf_` 三块 FP32 scratch 闭合；160KiB 静态 UB 可用 | H3→H3.1 false 的 `else`；true 短路到 R-005 |
| P3 | R-005 BF16 PD4 | 长跨度 BF16 资源特化；保持 BF16→FP32→BF16 路径并把 Queue 容量从 2 slot 提到 4 slot | 950、连续 BF16、`N>0`、`L>=131072` | 目标 UB 可容纳 224KiB；Host 无资源降级 guard | H3→H3.1 true；命中后跳过 R-004，false 进入 R-004；无 Device fallback |
| P4 | R-003 FP16 PD4 | 长跨度 FP16 资源特化；与 P1 同算术和 tile，仅把 Queue 容量从 2 slot 提到 4 slot | 950、连续 FP16、`N>0`、`L>=131072` | launcher 与 entry 的 PD 同为 4；160KiB 静态 UB 可用 | H2→H2.1 true；命中后跳过 R-002，false 进入 R-002；无 Device fallback |

### Route 选择

下表按重要性降序排列；`Host 顺序`独立记录源码中 dtype 外层分发和窄类型 depth 内层判断的真实位置，因此 P0～P4 不代表 `if/else if` 的执行次序。

| 重要性 | Route | Key/模板 | 精确选择条件 | Host 顺序、短路与 fallback |
|---|---|---|---|---|
| P0 | R-001 FP32 PD4 | `KernelMish<float,4>` | `dtype==torch::kFloat32` | H1：dtype 第一分支，命中后跳过 H2/H3；PD 固定 4。未命中继续 H2；其余 dtype 最终无启动 |
| P1 | R-002 FP16 PD2 | `KernelMish<half,2>` | `dtype==torch::kFloat16 && blockLength<16*8192` | H2：仅在 H1 未命中后进入 FP16 launcher；H2.1 先判断 `L>=131072`，false 才由 `else` 命中本 Route；true 短路到 R-003 |
| P2 | R-004 BF16 PD2 | `KernelMish<bfloat16_t,2>` | `dtype==torch::kBFloat16 && blockLength<16*8192` | H3：仅在 H1/H2 均未命中后进入 BF16 launcher；H3.1 先判断 `L>=131072`，false 才由 `else` 命中本 Route；true 短路到 R-005 |
| P3 | R-005 BF16 PD4 | `KernelMish<bfloat16_t,4>` | `dtype==torch::kBFloat16 && blockLength>=16*8192` | H3.1：BF16 launcher 的首个 depth 分支，true 后跳过 R-004；false 进入 R-004。无 Device fallback |
| P4 | R-003 FP16 PD4 | `KernelMish<half,4>` | `dtype==torch::kFloat16 && blockLength>=16*8192` | H2.1：FP16 launcher 的首个 depth 分支，true 后跳过 R-002；false 进入 R-002。无 Device fallback |

```text
N>0 -> calc B,L -> dtype
  float32  -> P0/R-001
  float16  -> L>=131072 ? P4/R-003 : P1/R-002
  bfloat16 -> L>=131072 ? P3/R-005 : P2/R-004
  otherwise -> no launch, lambda returns 0
```

### active、conditional 与 dormant 分类

| 类别 | Route/分支 | 可达性与处理原则 |
|---|---|---|
| active | R-001、R-002、R-003、R-004、R-005 | 五个模板实例都由生产 launcher 直接接线。FP16/BF16 的 `L>=131072` 与 `L<131072` 是互斥 active selector 子域；每次调用恰好启动其中一个 Route，均须保留完整闭包。 |
| conditional | 无 | 生产 selector 没有只在额外运行条件成立后才启用的 Route；L 阈值只是 active Route 之间的静态分区。 |
| dormant | `op_kernel/arch35/mish_dag.h` 中的 `MishDag1::MishCustom/MishDAG` | 该文件虽随转换源码留存，但算子 CMake 只编译 `mish_kernel.cpp`，活动 Kernel 也未 include 它，生产链不可达。只有把该头接入编译源并由 launcher/entry 显式路由后才可能重新启用；当前不得把它当作 fallback。 |

合法输入域的 Route partition 无重叠也无遗漏：FP32 全部进入 R-001；FP16 按阈值分为 R-002/R-003；BF16 按同一阈值分为 R-004/R-005。空输入、非连续 storage 与其它 dtype 不在合法 partition 内，也没有备用 Route。

### 模板/实现执行矩阵

| 重要性 | Route | Host 顺序 | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/fallback |
|---|---|---|---|---|---|---|---|---|
| P0 | R-001 FP32 PD4 | H1 | FP32 branch | 950，FP32，连续，`N>0` | 单 K1/AIV | `mish_kernel<float,4>` → `KernelMish<float,4>` | 160KiB UB；TQue producer/consumer 同步；FP32 | 4096 tile，`tail>0` 才提交尾块；dtype 未命中转 H2 |
| P1 | R-002 FP16 PD2 | H2→H2.1 false | FP16 + `L<131072` | 950，FP16，连续，`N>0` | 单 K1/AIV | `mish_kernel<half,2>` → `KernelMish<half,2>` | 96KiB UB；原生 FP16 | 8192 tile；H2.1 true 短路到 R-003 |
| P2 | R-004 BF16 PD2 | H3→H3.1 false | BF16 + `L<131072` | 950，BF16，连续，`N>0` | 单 K1/AIV | `mish_kernel<bfloat16_t,2>` → `KernelMish<bfloat16_t,2>` | 160KiB UB；BF16→FP32→BF16 | 8192 tile；H3.1 true 短路到 R-005 |
| P3 | R-005 BF16 PD4 | H3→H3.1 true | BF16 + `L>=131072` | 同上且每核名义跨度至少 16 个窄 tile | 单 K1/AIV | `mish_kernel<bfloat16_t,4>` → `KernelMish<bfloat16_t,4>` | 224KiB UB；BF16→FP32→BF16；4-slot Queue | 8192 tile；false 进入 R-004，无 Device fallback |
| P4 | R-003 FP16 PD4 | H2→H2.1 true | FP16 + `L>=131072` | 950，FP16，连续，`N>0`，且每核名义跨度至少 16 个窄 tile | 单 K1/AIV | `mish_kernel<half,4>` → `KernelMish<half,4>` | 160KiB UB；原生 FP16；4-slot Queue | 8192 tile；false 进入 R-002，无 Device fallback |

### 关键性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 非空域 | `totalLength>0`（调用契约，源码未检查） | `blockLength=0` 后重算 `numBlocks` 会除零 | 全部 | 未定义 Host 失败，不启动合法 Route | `calc_mish_tiling_params` |
| 最小工作量 | `B0<=ceil(N/2048)` | 避免为不足 2048 元素的名义工作创建更多核 | 全部 | 减少 `B0` | `MIN_ELEMS_PER_CORE` |
| core span 对齐 | `L=align_up(ceil(N/B0),4096)` | FP32 非末核无 partial tile；窄类型核起点至少 4096 对齐 | 全部 | 无备用路线 | `CORE_SPAN_ALIGN_ELEMS` |
| 窄类型 PD4 | `L>=16*8192` | 仅长窄工作负载承担更深输入/输出 Queue 的 UB 与设置成本 | P1 R-002、P2 R-004、P3 R-005、P4 R-003 | 保持 PD2 | `NARROW_PD4_MIN_BLOCK_LENGTH` |
| BF16 Vector 能力 | `T` 既非 float 也非 half | Direct 950 `Exp/Div` 路线使用 FP32 operand，故需 `xFloatBuf_` | P2 R-004、P3 R-005 | 无 BF16 原生 fallback | `KernelMish::Compute` |
| UB 上界 | 模板静态最大 229376B；Host 不读取 UB 大小 | 五实例必须适配目标 950 UB；没有运行时缩 tile fallback | 全部 | 资源不足则启动/编译失败 | `KernelMish::Init` |
| DMA 长度窄化 | `count<=8192` | `count*sizeof(T)` 转 `uint32_t`；当前 tile 上界保证安全 | 全部 | 无备用路线 | `CopyIn/CopyOut` |
| dtype guard | FP32/FP16/BF16 三分支 | 仅五个模板实例被编译和启动 | 全部 | 其他 dtype 不启动但返回 0 | `mish_npu::acl_call` |

### POD/ABI

本实现没有 tiling POD 结构和 tiling GM；launcher 将标量逐项按值传给 Kernel。固定源码没有给出编译器最终参数包的总 `sizeof/alignof/offsetof`，因此下表只记录源码类型、顺序和 consumer，不臆造封送 offset。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `mish_kernel` 参数列 | `x` | `GM_ADDR`/指针位宽 | 第 1 参数；总布局固定源码不可见 | GM 地址 | `x.data_ptr()` | `Init` 绑定 `GlobalTensor<T>` | 必须指向连续的至少 `N` 个 `T` |
| 同上 | `y` | `GM_ADDR`/指针位宽 | 第 2 参数 | GM 地址 | `y.data_ptr()` | `Init` 绑定输出 GM | 与 x 不应别名 |
| 同上 | `totalLength` | `int64_t`/64 | 第 3 参数 | 元素 | `x.numel()` | `Init` 算本核有效长度 | 必须 >0；乘加需不溢出 int64 |
| 同上 | `blockLength` | `int64_t`/64 | 第 4 参数 | 元素/core | `L=align_up(ceil(N/B0),4096)` | GM base、本核长度、PD selector（Host） | 正数、4096 对齐 |
| 同上 | `tileSize` | Host tuple 为 `int64_t`，C ABI/entry 为 `uint32_t`/32 | 第 5 参数 | 元素 | Host tuple 固定 4096，调用 launcher 前显式窄化 | `Init` 形参未读取 | inactive；真实 tile 由模板决定 |
| launch config | `numBlocks` | `int64_t`/64 | `<<<>>>` 第 1 项，不进入 Device 参数列 | AIV block | `B=ceil(N/L)` | `GetBlockIdx()` 可达域 | `1<=B<=max(AIV cores,1)` |
| launch config | `stream` | `void*`/指针位宽 | `<<<>>>` 第 3 项 | stream handle | `getCurrentNPUStream().stream(false)` | runtime ordering | 异步错误未在 lambda 返回值传播 |

### Route 源码闭包

| 重要性 | Route | Host 顺序 | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|---|---|
| P0 | R-001 | H1 | FP32 | `mish_npu`、`calc_mish_tiling_params`、`launch_mish_kernel_float` | 5 项 entry 参数，无结构体 | `mish_kernel<float,4>` | `KernelMish<float,4>::Init/Process/CopyIn/Compute/CopyOut` | 算子 `CMakeLists.txt::register_direct_launch_op` 注册 kernel/plugin |
| P1 | R-002 | H2→H2.1 false | FP16/PD2 | 同上、`launch_mish_kernel_half` 的 `<131072` 分支 | 同上 | `mish_kernel<half,2>` | `KernelMish<half,2>` 全方法 | 同上 |
| P2 | R-004 | H3→H3.1 false | BF16/PD2 | 同上、`launch_mish_kernel_bfloat16` 的 `<131072` 分支 | 同上 | `mish_kernel<bfloat16_t,2>` | `KernelMish<bfloat16_t,2>` 全方法 | 同上 |
| P3 | R-005 | H3→H3.1 true | BF16/PD4 | 同上、BF16 launcher 的 `>=131072` 分支 | 同上 | `mish_kernel<bfloat16_t,4>` | `KernelMish<bfloat16_t,4>` 全方法 | 同上 |
| P4 | R-003 | H2→H2.1 true | FP16/PD4 | 同上、half launcher 的 `>=131072` 分支 | 同上 | `mish_kernel<half,4>` | `KernelMish<half,4>` 全方法 | 同上 |

## Kernel 实现

### 公共执行流程

所有 Route 共享一个单 Kernel launch graph 和 `KernelMish` 方法序列。core `b=GetBlockIdx()` 绑定：

```text
xGm = x + b*L
yGm = y + b*L
n_b = clamp(N-b*L, 0, L)
full = floor(n_b/tile)
tail = n_b-full*tile
for i in [0,full): CopyIn(i*tile,tile) -> Compute(tile) -> CopyOut(i*tile,tile)
if tail>0: CopyIn(full*tile,tail) -> Compute(tail) -> CopyOut(full*tile,tail)
```

`CopyIn` 与 `CopyOut` 均用 `DataCopyExtParams{blockCount=1, blockLen=count*sizeof(T), srcStride=0, dstStride=0, rsv=0}`。输入 `DataCopyPad` 关闭显式 padding；Vector API 的 `calCount=count` 和回写字节数共同保护尾部。输入 Queue 的 `EnQue→DeQue→FreeTensor`、输出 Queue 的 `AllocTensor→EnQue→DeQue→FreeTensor` 形成 producer/consumer 与复用生命周期；无显式 event、barrier、atomic 或跨核通信。

`Process` 对每个 tile 按 `CopyIn→Compute→CopyOut` 的程序顺序发射命令，完成本轮 Queue 生命周期后才进入下一轮；源码没有显式的多 tile 预取或分阶段 loop。PD2/PD4 可直接证明的是 Queue slot 容量和 UB 布局差异；TQue/runtime 可以利用不同 pipe 的异步命令形成跨轮重叠，但仅凭模板 depth 不能推出恰有四个在途 tile 或固定吞吐收益。tail 的位置也由 dtype 决定：FP32 只有最后一核可能出现非整 tile；FP16/BF16 在 `L%8192==4096` 时，每个非末核都会有 4096 元素 tail，最后一核还可有任意 `1..8191` 元素 tail。

以下逐 Route 章节按 P0→P4 展开：R-001、R-002、R-004、R-005、R-003。R-ID 保持不变，真实 Host 顺序仍以“Route 选择”中的 H1～H3.1 为准。

### P0 / Route R-001：FP32 PD4

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | `op_plugin/mish_plugin.cpp` | `mish_npu::acl_call` | `dtype==torch::kFloat32`，第一分支 | float launcher |
| Launcher | `op_kernel/mish_kernel.cpp` | `launch_mish_kernel_float` | 无 shape/PD 分支 | `mish_kernel<float,4>` |
| Entry | 同上 | `mish_kernel<float,4>` | `tile=4096,ComputeT=float` | `KernelMish<float,4>::Process` |

#### Device 数据流

##### 函数级调用链

```text
launch_mish_kernel_float
  -> mish_kernel<float,4>
    -> KernelMish<float,4>::Init
    -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
for each owned tile with valid count:
  xLocal = copy xGm[offset : offset+count]
  sp = min(xLocal, 20.0f)
  sp = exp(sp)
  tmp = sp + 2.0f
  sp = sp * tmp
  tmp = sp + 2.0f
  sp = sp / tmp
  yLocal = xLocal * sp
  copy yLocal[0:count] -> yGm[offset:]
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `b*L+i*4096` | `count<=4096` | FP32/flat | caller/CopyOut | CopyIn/caller | Tensor | 单连续 byte block |
| `inQueueX_` | Queue slot | `4*4096*4=65536B` | FP32 | CopyIn | Compute | `TPipe`; xLocal 用后 free | depth 4 |
| `outQueueY_` | Queue slot | `4*4096*4=65536B` | FP32 | Compute | CopyOut | `TPipe`; copy 后 free | depth 4 |
| `spBuf_` | TBuf | `4096*4=16384B` | FP32 | Mins/Exp/Mul/Div | 后继 Vector op | Kernel lifetime | 原地覆盖 |
| `tmpBuf_` | TBuf | `4096*4=16384B` | FP32 | Adds | Mul/Div | Kernel lifetime | 两次 Adds 覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueueX · depth=4 · 65536B"]:1
  outQ["outQueueY · depth=4 · 65536B"]:1
  sp["spBuf · 16384B"]:1
  tmp["tmpBuf · 16384B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4*4096*4 + 4*4096*4 + 4096*4 + 4096*4 = 163840B = 160KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `spBuf_` | Mins | Exp→Mul→Div | 下一 tile Mins | 同一依赖链；下一 tile 在本 tile CopyOut 后 | 保持源码调用顺序 |
| `tmpBuf_` | 第一次 Adds | Mul | 第二次 Adds | Mul 消费完成后覆盖 | 不可提前执行第二次 Adds |
| Queue slot | CopyIn/Compute | Compute/CopyOut | 后续 AllocTensor | `FreeTensor` | EnQue 后才 DeQue，消费后才 free |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VEC | input TQue | CopyIn EnQue | Compute DeQue | x tile | blockLen=`count*4`B |
| Vector | 数据依赖，无显式 barrier | Mins/Exp/Adds/Mul/Div | 下一 Vector op | sp/tmp | 全 FP32，cap=20，`calCount=count` |
| VEC→GM | output TQue | Compute EnQue | CopyOut DeQue | y tile | 只回写 `count*4`B；tail 0 时不执行 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_plugin/mish_plugin.cpp`；`op_kernel/mish_kernel.cpp` | `mish_npu`；`calc_mish_tiling_params` | N、B、L、stream 与 dtype 分支 |
| Kernel entry/router | `op_kernel/mish_kernel.cpp` | `launch_mish_kernel_float`；`mish_kernel<float,4>` | FP32 固定 PD4 单 AIV Kernel |
| Device body/helper | 同上 | `KernelMish<float,4>` | 4096 tile、FP32 稳定等价式、Queue tail |

### P1 / Route R-002：FP16 PD2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | `op_plugin/mish_plugin.cpp` | `acl_call` | `dtype==torch::kFloat16` | half launcher |
| Launcher | `op_kernel/mish_kernel.cpp` | `launch_mish_kernel_half` | `blockLength<16*8192` | `mish_kernel<half,2>` |
| Entry | 同上 | `mish_kernel<half,2>` | `tile=8192,ComputeT=half` | native-half body |

#### Device 数据流

##### 函数级调用链

```text
launch_mish_kernel_half -> mish_kernel<half,2>
  -> KernelMish<half,2>::Init
  -> Process -> CopyIn -> Compute(native half) -> CopyOut
```

##### 源码等价伪码

```text
for each count<=8192 tile:
  sp=exp(min(xLocal,half(5.5)))
  tmp=sp+half(2); sp=sp*tmp
  tmp=sp+half(2); sp=sp/tmp
  yLocal=xLocal*sp
  copy only count elements to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `b*L+i*8192` | `count<=8192` | FP16/flat | caller/CopyOut | CopyIn/caller | Tensor | tail byte copy |
| `inQueueX_`/`outQueueY_` | Queue slots | each `2*8192*2=32768B` | FP16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth 2 |
| `spBuf_`/`tmpBuf_` | TBuf | each `8192*2=16384B` | FP16 | Vector body | Vector body | Kernel | 原地/阶段覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueueX · depth=2 · 32768B"]:1
  outQ["outQueueY · depth=2 · 32768B"]:1
  sp["spBuf · 16384B"]:1
  tmp["tmpBuf · 16384B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*8192*2 + 2*8192*2 + 8192*2 + 8192*2 = 98304B = 96KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `spBuf_`/`tmpBuf_` | 同 R-001 的对应 half API | 后继 half API | 下一阶段/下一 tile | 源码依赖顺序 | cap 必须保持 half(5.5) |
| Queue slots | CopyIn/Compute | Compute/CopyOut | 后续 tile | `FreeTensor` | depth 与模板实例一致 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| GM→VEC→GM | TQue | EnQue | DeQue/FreeTensor | x/y tile | `count*2`B 搬运 |
| Vector | 原生 half 数据依赖 | Mins/Exp/Adds/Mul/Div | 下一 op | sp/tmp | cap=5.5；无升降精度 cast |
| tail | `tailTileElementNum_>0` | Init | Process | 末 tile | `calCount=count<=8192` |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/mish_kernel.cpp` | `calc_mish_tiling_params` | 4096 对齐 L |
| Kernel entry/router | 同上 | `launch_mish_kernel_half` | `L<131072` 选择 PD2 |
| Device body/helper | 同上 | `KernelMish<half,2>::Compute` | 8192 tile、native half、cap 5.5 |

### P2 / Route R-004：BF16 PD2

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | `op_plugin/mish_plugin.cpp` | `acl_call` | BF16 | BF16 launcher |
| Launcher | `op_kernel/mish_kernel.cpp` | `launch_mish_kernel_bfloat16` | `blockLength<131072` | `mish_kernel<bfloat16_t,2>` |
| Entry | 同上 | `mish_kernel<bfloat16_t,2>` | `tile=8192,ComputeT=float` | widen-compute-narrow body |

#### Device 数据流

##### 函数级调用链

```text
launch_mish_kernel_bfloat16 -> mish_kernel<bfloat16_t,2>
  -> KernelMish<bfloat16_t,2>::Init
  -> Process -> CopyIn -> Compute(Cast/FP32/Cast_RINT) -> CopyOut
```

##### 源码等价伪码

```text
xLocal_bf16 = copy GM tile
xF = cast_bf16_to_fp32(xLocal, CAST_NONE)
sp = exp(min(xF,20.0f)); tmp=sp+2; sp=sp*tmp
tmp=sp+2; sp=sp/tmp; sp=xF*sp
yLocal_bf16 = cast_fp32_to_bf16(sp, CAST_RINT)
copy only count BF16 elements to GM
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | `b*L+i*8192` | `count<=8192` | BF16/flat | caller/CopyOut | CopyIn/caller | Tensor | `count*2`B |
| `inQueueX_`/`outQueueY_` | Queue slots | each `2*8192*2=32768B` | BF16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth 2 |
| `xFloatBuf_` | TBuf | `8192*4=32768B` | FP32 | input Cast | arithmetic/final Mul | Kernel | 每 tile 覆盖 |
| `spBuf_`/`tmpBuf_` | TBuf | each 32768B | FP32 | arithmetic | arithmetic/cast | Kernel | 原地/阶段覆盖 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inQueueX · depth=2 · 32768B"]:1
  outQ["outQueueY · depth=2 · 32768B"]:1
  xF["xFloatBuf · 32768B"]:1
  sp["spBuf · 32768B"]:1
  tmp["tmpBuf · 32768B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 2*8192*2 + 2*8192*2 + 3*8192*4 = 163840B = 160KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| `xFloatBuf_` | input Cast | Mins 与最终 Mul | 下一 tile Cast | 当前 arithmetic 和 output cast 完成后 | 不得在最终 Mul 前覆盖 |
| `spBuf_`/`tmpBuf_` | FP32 ops | 后继 FP32 ops | 下一阶段/tile | 数据依赖 | 输出只在末尾 RINT cast |
| Queue slots | CopyIn/Compute | Compute/CopyOut | 后续 tile | FreeTensor | BF16 Queue 容量不按 FP32 计算 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| BF16→FP32 | `Cast(...CAST_NONE,count)` | x Queue | FP32 arithmetic | `xFloatBuf_` | 仅有效 count |
| arithmetic | FP32 数据依赖 | Mins/Exp/Adds/Mul/Div | 后继 op | xF/sp/tmp | cap=20 |
| FP32→BF16 | `Cast(...CAST_RINT,count)` | final FP32 sp | y Queue | yLocal | RINT 后只回写 count |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/mish_kernel.cpp` | `calc_mish_tiling_params` | L 与 B |
| Kernel entry/router | 同上 | `launch_mish_kernel_bfloat16` | `L<131072` 选择 PD2 |
| Device body/helper | 同上 | `KernelMish<bfloat16_t,2>::Compute` | BF16 widening、FP32 等价式、RINT narrowing |

### P3 / Route R-005：BF16 PD4

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | `op_plugin/mish_plugin.cpp` | `acl_call` | BF16 | BF16 launcher |
| Launcher | `op_kernel/mish_kernel.cpp` | `launch_mish_kernel_bfloat16` | `blockLength>=131072` | `mish_kernel<bfloat16_t,4>` |
| Entry | 同上 | `mish_kernel<bfloat16_t,4>` | `tile=8192,PD=4` | widen-compute-narrow body |

#### Device 数据流

##### 函数级调用链

```text
launch_mish_kernel_bfloat16 -> mish_kernel<bfloat16_t,4>
  -> KernelMish<bfloat16_t,4>::Init -> Process
    -> CopyIn -> Compute(BF16->FP32->BF16) -> CopyOut
```

##### 源码等价伪码

```text
precondition from launcher: blockLength >= 16*8192
execute R-004's 8192-element FP32-compute tiles and per-core tail
allocate input/output queues with four slots instead of two
# command order remains CopyIn -> Compute -> CopyOut per loop iteration
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | 同 R-004 | `count<=8192` | BF16 | caller/CopyOut | CopyIn/caller | Tensor | 同 R-004 |
| `inQueueX_`/`outQueueY_` | Queue slots | each `4*8192*2=65536B` | BF16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth 4 |
| `xFloatBuf_`/`spBuf_`/`tmpBuf_` | 三个 TBuf | each 32768B | FP32 | Cast/Vector | Vector/Cast | Kernel | 同 R-004 |

##### 分层内存布局

```mermaid
block-beta
  columns 5
  inQ["inQueueX · depth=4 · 65536B"]:1
  outQ["outQueueY · depth=4 · 65536B"]:1
  xF["xFloatBuf · 32768B"]:1
  sp["spBuf · 32768B"]:1
  tmp["tmpBuf · 32768B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4*8192*2 + 4*8192*2 + 3*8192*4 = 229376B = 224KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue/TBuf | 同 R-004 | 同 R-004 | 后续 tile | Queue lifecycle/数据依赖 | 目标 950 UB 必须容纳 224KiB；Host 无降级 guard |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | 与 R-004 相同 | 同上 | 同上 | 同上 | 唯一差异为 Queue 容量 4；cast、cap、tail 不变；实际跨轮重叠度由 TQue/runtime 决定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/mish_kernel.cpp` | `calc_mish_tiling_params` | L 与 B |
| Kernel entry/router | 同上 | `launch_mish_kernel_bfloat16` | `L>=131072` 选择 PD4 |
| Device body/helper | 同上 | `KernelMish<bfloat16_t,4>` | 最大 224KiB UB；FP32 compute + RINT |

### P4 / Route R-003：FP16 PD4

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Plugin | `op_plugin/mish_plugin.cpp` | `acl_call` | FP16 | half launcher |
| Launcher | `op_kernel/mish_kernel.cpp` | `launch_mish_kernel_half` | `blockLength>=131072` | `mish_kernel<half,4>` |
| Entry | 同上 | `mish_kernel<half,4>` | `tile=8192,PD=4` | native-half body |

#### Device 数据流

##### 函数级调用链

```text
launch_mish_kernel_half -> mish_kernel<half,4>
  -> KernelMish<half,4>::Init -> Process -> CopyIn -> Compute -> CopyOut
```

##### 源码等价伪码

```text
precondition from launcher: blockLength >= 16*8192
execute the same 8192-element native-half tiles and per-core tail as R-002
only inQueueX/outQueueY compile-time capacity changes from 2 to 4
# command order remains CopyIn -> Compute -> CopyOut per loop iteration
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| x/y GM | 同 R-002 | `count<=8192` | FP16 | caller/CopyOut | CopyIn/caller | Tensor | 相同 tail |
| `inQueueX_`/`outQueueY_` | Queue slots | each `4*8192*2=65536B` | FP16 | CopyIn/Compute | Compute/CopyOut | `TPipe` | depth 4 |
| `spBuf_`/`tmpBuf_` | TBuf | each 16384B | FP16 | Vector | Vector | Kernel | 同 R-002 |

##### 分层内存布局

```mermaid
block-beta
  columns 4
  inQ["inQueueX · depth=4 · 65536B"]:1
  outQ["outQueueY · depth=4 · 65536B"]:1
  sp["spBuf · 16384B"]:1
  tmp["tmpBuf · 16384B"]:1
```

##### 字节账与复用顺序

```text
UB_total = 4*8192*2 + 4*8192*2 + 8192*2 + 8192*2 = 163840B = 160KiB
dynamic_ub = N/A
workspace_total = 0
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| Queue/TBuf | 同 R-002 | 同 R-002 | 后续 tile | Queue lifecycle/数据依赖 | launcher 与 entry 的 PD 必须同为 4 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 全管线 | 与 R-002 相同 | 同上 | 同上 | 同上 | 唯一差异为 Queue 容量 4；数学与 tail 不变；实际跨轮重叠度由 TQue/runtime 决定 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_kernel/mish_kernel.cpp` | `calc_mish_tiling_params` | 产生 L |
| Kernel entry/router | 同上 | `launch_mish_kernel_half` | `L>=131072` 选择 PD4 |
| Device body/helper | 同上 | `KernelMish<half,4>` | 与 R-002 同算术，Queue depth 4 |

## 算子特定优化

### Trick T-001：稳定单 Exp/单 Div 等价式与 dtype cap

#### 触发与源码依据

- 适用 Route：P0 R-001、P1 R-002、P2 R-004、P3 R-005、P4 R-003。
- 触发条件：所有合法非空 FP32/FP16/BF16 Direct 输入；cap 按模板 dtype 固定。
- 相关官方来源：[`mish_dag.h:81-100`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/op_kernel/arch35/mish_dag.h#L81-L100)。
- 归因边界：以下摘录是固定官方四 Exp/两 Div 分段基线；单 Exp/单 Div 与 cap 是 Direct 转换事实。

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
MicroAPI::Muls(vregInputNegNumerator, vregInput, FP32_NEG_ONE, mask);             // -x
MicroAPI::Muls(vregInputNegDenominator, vregInput, FP32_NEG_TWO, mask);           // -2x
MicroAPI::Exp(vregInputNegNumerator, vregInputNegNumerator, mask);                // e^-x
MicroAPI::Exp(vregInputNegDenominator, vregInputNegDenominator, mask);            // e^-2x
MicroAPI::Muls(vregInputNegNumerator, vregInputNegNumerator, FP32_TWO, mask);     // 2e^-x
MicroAPI::Adds(vregInputNegNumerator, vregInputNegNumerator, FP32_ONE, mask);     // 2e^-x + 1
```

#### 机制与设计原因

Direct body 把官方分段表达式代数化为 `u=exp(min(x,cap)); v=u*(u+2); y=x*v/(v+2)`，每元素只调用一次 `Exp` 和一次 `Div`，同时只需 `sp`、`tmp` 两条计算缓冲；BF16 额外需要 `xF`。`Mins` 的 cap 在 Exp 前执行：FP32/BF16 为 20，FP16 为 half(5.5)。源码可直接证明指令序列和中间值上界的变化；它本身不证明时延或吞吐增益。

#### 迁移边界

- 适用：逐元素 Mish，输出与输入一一对应且可接受相同 cap 语义的实现。
- 前提：目标 Vector API 支持相应 dtype 的 `Mins/Exp/Adds/Mul/Div`，并验证 cap 后的大值舍入行为与精度口径。
- 失效：若必须逐位复现未截断公式、不同特殊值语义或目标 dtype 在 cap 处误差不合格，应恢复经验证的稳定分段公式或另建 Route。
- 证据边界：复用时应单独核对 cap 邻域、极值、NaN/Inf 与目标精度口径；本卡不从静态公式推导这些输入上的逐位一致性。

#### 错误做法

不能在 FP16 路线沿用 cap=20：`exp(20)` 以及后续乘积超出 half 有限范围。固定官方仓未提供 Direct cap 路线的可验证反例；这里的失败依据来自 Direct 数据类型与表达式上界，而非官方 fallback。

### Trick T-002：FP16 原生计算，BF16 仅在寄存/UB 路径升 FP32

#### 触发与源码依据

- 适用 Route：P1 R-002 与 P4 R-003 使用原生 FP16；P2 R-004 与 P3 R-005 使用 BF16 widening。
- 触发条件：模板 `kIsHalf` 决定 `ComputeT=half`；BF16 命中 `!kIsFloat&&!kIsHalf` 时分配 `xFloatBuf_`。
- 相关官方来源：[`mish_dag.h:73-80`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/op_kernel/arch35/mish_dag.h#L73-L80)。
- 归因边界：官方摘录显示 16-bit 输入升 FP32 的相关基线；Direct 仅 BF16 升精度、FP16 保持原生是转换特化。

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if constexpr (std::is_same_v<T, float>) {
    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
        vregInput, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
} else {
    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
        vregInput16, (__ubuf__ T*)(srcAddr + loopIdx * vlSize));
    MicroAPI::Cast<float, T, castTrait0>(vregInput, vregInput16, mask);
}
```

#### 机制与设计原因

FP16 Route 让 `sp/tmp` 都是 half，省去输入、输出 Cast 和一个 32KiB `xFloatBuf_`；cap=5.5 控制 half 中间乘积。BF16 Route 因 Direct 950 `Exp/Div` operand 路径采用 FP32，先 `CAST_NONE` 到 `xF`，全程 FP32 计算，最后 `CAST_RINT` 回 BF16。编译期 `if constexpr` 消除不适用分支和 BF16 专用 TBuf 的条件分配。

#### 迁移边界

- 适用：目标硬件可原生执行 FP16 Exp/Div，而 BF16 需要 FP32 operand 的逐元素算子。
- 前提：分别审计 FP16 cap、BF16 RINT 和最大 UB；不能只按元素字节数合并两条 Route。
- 失效：目标平台原生支持 BF16 或 FP16 精度口径要求 FP32 中间值时，应重做模板、UB 与误差闭包。
- 证据边界：本卡只固化转换 body 的 cast、round mode 与 UB 关系；目标平台若改变 BF16 Vector operand 能力，必须重新建立精度和资源闭包。

#### 错误做法

不要为 BF16 仅把 Queue dtype 改成 BF16 而继续对其直接调用该 Direct 路线的 `Exp/Div`；也不要删除末尾 `CAST_RINT`。固定官方仓未提供该 Direct 特化的可验证 fallback。

### Trick T-003：按每核跨度编译期选择 PD2/PD4

#### 触发与源码依据

- 适用 Route：P1 R-002、P2 R-004、P3 R-005、P4 R-003。
- 触发条件：窄类型 `blockLength<131072` 选 PD2，`>=131072` 选 PD4；FP32 固定 PD4。
- 相关官方来源：[`mish_tiling_arch35.cpp:110-136`](../../../../../../../cann-ops-raw/ascendc/ops-nn/955f8b33985b6d72989ebff341bb9c2c564a51bc/activation/mish/op_host/arch35/mish_tiling_arch35.cpp#L110-L136)。
- 归因边界：官方摘录使用 `ElewiseBaseTiling` 和官方 tiling key；显式 4096 core span、16-tile 阈值及 PD2/PD4 模板是 Direct 转换事实。

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
if (this->outputDtype == ge::DT_FLOAT16) {
    dType = TPL_FP16;
    baseTilingResult = elewiseBaseTiling.DoTiling<MishDAG<half>::OpDag>(*tiling);
} else if (this->outputDtype == ge::DT_BF16) {
    dType = TPL_BF16;
    baseTilingResult = elewiseBaseTiling.DoTiling<MishDAG<bfloat16_t>::OpDag>(*tiling);
} else if (this->outputDtype == ge::DT_FLOAT) {
    dType = TPL_FP32;
```

#### 机制与设计原因

Direct launcher 使用已经按 4096 元素对齐的每核 `L` 作为工作量代理；只有 `L` 至少覆盖 16 个 8192 元素 tile 时才实例化 PD4。短窄任务因此保留 96KiB/160KiB 的 PD2 布局，长窄任务使用 160KiB/224KiB 的 PD4 布局。PD 是模板参数，Kernel 内没有运行时 Queue-depth 分支。当前 `Process` 仍按每轮 `CopyIn→Compute→CopyOut` 发射，没有显式多 tile 预取；因此可迁移的硬事实是“阈值控制静态 Queue 容量与 UB，并为 runtime 跨 pipe 调度提供更多 slot”，不是“固定四个 tile 在途”或必然获得某个收益。

#### 迁移边界

- 适用：tile 固定、每核工作量可在 Host 获知、Queue depth 可做编译期实例化的单 Kernel elementwise 实现。
- 前提：阈值必须结合 tile、dtype、UB 上限和目标硬件重新审计；BF16/PD4 至少需要 229376B 静态 UB。
- 失效：可用 UB 小于路线字节账、tile 改变或分核不再以 `L` 表示时，必须改阈值、降为 PD2 或增加明确 fallback。
- 证据边界：迁移时需要结合目标 TQue/event 语义核对真实跨轮重叠；若改成显式预取/计算/回写交错，还必须重新审计共享 TBuf 的并发复用。

#### 错误做法

不要在 Device 内用普通运行时 `if` 切换同一组 `TQue` 的 depth；`InitBuffer` 容量和模板类型必须与 launcher 选择的实例一致。固定官方仓未提供该 Direct 阈值的可验证反例。

## 模板与实现对比

| 对比维度 | P0 R-001 FP32/PD4 | P1 R-002 FP16/PD2 | P2 R-004 BF16/PD2 | P3 R-005 BF16/PD4 | P4 R-003 FP16/PD4 | 设计选择依据 |
|---|---|---|---|---|---|---|
| Host 顺序/命中条件 | H1：FP32 第一分支 | H2→H2.1 false | H3→H3.1 false | H3→H3.1 true | H2→H2.1 true | dtype 先分发，窄类型 launcher 再先判 PD4、否则 PD2 |
| 模板与 dtype | `<float,4>` | `<half,2>` | `<bfloat16_t,2>`，`ComputeT=float` | `<bfloat16_t,4>`，`ComputeT=float` | `<half,4>` | Vector operand 能力和精度路径 |
| 循环与分核 | 4096 tile | 8192 tile | 8192 tile | 8192 tile | 8192 tile | core span 统一 4096 对齐 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | Q4 + 2 FP32 TBuf | Q2 + 2 half TBuf | Q2 + 3 FP32 TBuf | Q4 + 3 FP32 TBuf | Q4 + 2 half TBuf | BF16 额外保存 widened x；无 dynamic UB/通信 |
| 内存总占用与复用 | 160KiB | 96KiB | 160KiB | 224KiB | 160KiB | PD 与 ComputeT 共同决定 |
| 搬运/Vector/SIMT/NDDMA/通信 API | byte DataCopyPad；FP32 Vector | 同搬运；native half Vector | BF16 byte copy + Cast + FP32 Vector + RINT | 同 P2，Queue 容量 4 | 同 P1，Queue 容量 4 | 全部为 AIV；其余执行族 N/A；PD4 无显式多 tile 预取 |
| 同步、精度与 tail | TQue；FP32；4096 tail | TQue；FP16；8192 tile/per-core tail | TQue；FP32 中间；8192 tile/per-core tail | 同 P2 | 同 P1 | `calCount` 与 byte copy 限定有效域；窄类型非末核也可能有 4096 tail |
| fallback/失败边界 | dtype 未命中继续 H2 | depth true 短路到 P4 | depth true 短路到 P3 | depth false 到 P2；无 Device fallback | depth false 到 P1；无 Device fallback | 空输入、非法 dtype、UB 不足均无合法 fallback |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 公共工程暴露 A2 编译目标并接线同一 Mish 源码，但 canonical 仅闭合 950；A2 不计入本卡平台范围 | 公共工程暴露 A3 编译目标并接线同一 Mish 源码，但 canonical 仅闭合 950；A3 不计入本卡平台范围 | 固定官方 README 支持 Ascend 950PR/950DT；本卡目标为 950/950 |
| Host/Launcher | 公共构建接线存在；A2 的 Host、launcher 与 launch ABI 闭包未建立 | 公共构建接线存在；A3 的 Host、launcher 与 launch ABI 闭包未建立 | `GetCoreNumAiv` + 4096 span + 三 dtype launcher + 窄类型 PD selector |
| Kernel/资源 | A2 的编译、Vector API、UB 容量与运行语义未闭合，不可复用 950 假设 | A3 的编译、Vector API、UB 容量与运行语义未闭合，不可复用 950 假设 | 单 AIV Kernel；五模板；最大 224KiB UB |

## 限制与失败边界

- Direct 正常域是非空、连续、NPU 上的 FP32/FP16/BF16 Tensor；插件未强制这些检查。非连续 Tensor 会被当作一维连续 GM，语义不闭合。
- `totalLength==0` 会使对齐后的 `blockLength==0`，随后重算 `numBlocks` 时除零；没有 empty fast path。
- 不支持 dtype 不启动 Kernel，但 `acl_call` 仍返回 0，返回的 `empty_like` 内容未定义；没有错误或数值 fallback。
- `tileSize` ABI 字段固定传 4096 但 Device 不读取；真实 tile 是模板常量。修改 dtype tile 时还必须同步审计 core span、PD 阈值、UB 与 tail。
- Host 不查询 UB 大小；R-005 静态占用 229376B。迁移到其他产品前必须按目标 UB 重新证明，不得仅凭 `arch35` 目录名扩平台。
- `totalLength`、`blockLength` 和地址乘加使用 int64；超大 `numel` 的 Host 加法与 GM offset 溢出未显式防护。
- 没有 workspace、跨 Kernel edge、atomic、显式 event/barrier、SIMT、NDDMA、通信或 Cube fallback。
- 固定官方实现的 rank/dtype/shape 校验属于相关官方 Host/Tiling，不会自动出现在 Direct 插件中。
- 本卡闭包限于固定源码与转换实现的静态关系；运行环境、精度阈值与性能收益需由消费方在目标配置上另行证明。

## 源码索引

### 链路索引

下表只列固定提交的相关官方来源；Direct 文件名在正文符号表中用于说明转换事实，不作为 frontmatter source，也不伪称官方 blob。

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `activation/mish/README.md` | Mish 公式、产品表、参数表 | 950/950 支持、官方数学语义、ND/dtype/rank 约束 |
| Host/Tiling | `activation/mish/op_host/arch35/mish_tiling_arch35.cpp` | `MishTiling::RunTiling`、`TilingPrepareForMish` | 官方 dtype/shape 校验、Elewise tiling、AIV core/UB 信息、blockDim/key |
| 模板参数 | `activation/mish/op_kernel/arch35/mish_struct.h` | `Mish` template args | 官方 schMode 与 FP16/BF16/FP32 模板 dtype 枚举 |
| Kernel entry | `activation/mish/op_kernel/mish_apt.cpp` | `mish<schMode,dType>` | 官方 AIV-only entry、dtype 到 DAG scheduler 映射 |
| Kernel body | `activation/mish/op_kernel/arch35/mish_dag.h` | `MishCustom`、`MishDAG` | 官方相关 Vector 算术、cast/mask、copy DAG |

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 无 POD；复用五项 entry 参数和 launch config | 参数顺序、64/32 位标量、`tileSize` inactive 事实 | `mish_launch.h`；`mish_kernel` |
| 调用顺序 | dtype→PD selector→单 Kernel→逐 tile C/I/O | 先按 dtype，再仅对窄类型判断 L；不得启动多个 Route | `mish_npu`；三个 launcher |
| workspace/内存 | workspace=0；Route 字节账可单独复用 | Queue depth、tile、ComputeT、BF16 extra TBuf 必须成套 | `KernelMish::Init` |
| 同步 | TQue 生命周期与 Vector 数据依赖 | EnQue/DeQue/FreeTensor 顺序；无跨核共享 | `CopyIn/Compute/CopyOut` |
| 精度与尾块 | 稳定等价式、dtype cap、BF16 RINT、`calCount=count` | FP16 cap=5.5；BF16 先升 FP32；只写有效 tail | `KernelMish::Compute/Process` |
| 失败边界 | 非空、连续、合法 dtype/950 UB 是调用前提 | 不得把无启动返回 0 当成合法 fallback；空输入需另加分支 | `mish_npu::acl_call`；`calc_mish_tiling_params` |
