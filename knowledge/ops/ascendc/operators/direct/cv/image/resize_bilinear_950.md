---
type: Operator
title: "ResizeBilinear 950 Direct Invoke 算子知识"
description: "说明 cann_bench::resize_bilinear、calc_resize_bilinear_config 与 launch_resize_bilinear_* 在 950 上的接口语义、Launcher 参数生成、Route 选择、单 Kernel 启动图与逐节点实现。"
tags: [task.implementation, task.lookup, topic.data_movement, topic.memory, topic.precision, topic.programming_model]
status: stable
sources:
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/README.md"
    role: header
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_api/resize_bilinear_v2.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_host/arch35/resize_bilinear_v2_tiling_arch35.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_kernel/resize_bilinear_v2_apt.cpp"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_base.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_simt_hw.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_simt_nchw.h"
    role: implementation
  - resource: "cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/CMakeLists.txt"
    role: implementation
created_at: "2026-08-25T19:31:34Z"
updated_at: "2026-09-11T03:05:58Z"
platforms: ["950"]
cann_versions: ["9.1.0"]
---

# ResizeBilinear 950 Direct Invoke 算子知识

## 算子速览

| 项目 | 内容 |
|---|---|
| 实际入口 | PyTorch custom op `cann_bench::resize_bilinear`，PrivateUse1 实现 `resize_bilinear_npu`，配置函数 `calc_resize_bilinear_config`，按 dtype 分开的 C ABI launcher `launch_resize_bilinear_{float32,float16,bfloat16}`；入口类型为 `Direct Invoke`。 |
| 计算语义 | 对连续 NCHW 四维张量的 H/W 两轴做双线性插值，输出保持 N、C 和 dtype，空间大小来自 `output_size` 或 `scale_factor`。 |
| 输入输出 | `x:[N,C,inH,inW]`、`y:[N,C,outH,outW]`，连续 NCHW；FP16、FP32、BF16，同 dtype 输入输出；所有维非空。 |
| 执行类型 | R-001～R-004 都是单个 `KERNEL_TYPE_AIV_ONLY` entry 内 `asc_vf_call` 的 SIMT 节点；无 Cube/AIC、MIX_CV、NDDMA 或通信节点。 |
| Kernel 路线 | 四条 terminal Route 均为 `active`：R-001 `SIMT_HW/i32`（key 30004）、R-002 `SIMT_NCHW/i32`（30001）、R-003 `SIMT_HW/i64`（30005）、R-004 `SIMT_NCHW/i64`（30003）。每条路线各有 3 dtype × 2 坐标模式的显式 entry；没有独立 `conditional` terminal Route。 |
| 核心资源 | 输入/输出 GM、线程私有标量/寄存器、1024 或 512 个 SIMT thread；无 AIV `TQue/TBuf`、UB、L1、L0A/L0B/L0C、dynamic UB 或 workspace，也无 atomic/barrier。 |
| 来源闭包 | `direct_implementation_closed`：最终直调载荷闭合 schema/Meta/PrivateUse1、配置/POD、显式 entry/launcher、两套 SIMT body 和直调构建；官方固定提交闭合其转换基线。 |

## 接口与语义

### 计算语义

对输出元素 `(n,c,h,w)`，先产生源坐标：

```text
if align_corners:
  scaleH = outH>1 ? (inH-1)/(outH-1) : inH/outH
  scaleW = outW>1 ? (inW-1)/(outW-1) : inW/outW
  srcH = h*scaleH; srcW = w*scaleW
else:
  scaleH = hasScaleFactor ? 1/scaleFactorH : inH/outH
  scaleW = hasScaleFactor ? 1/scaleFactorW : inW/outW
  srcH = max((h+0.5)*scaleH-0.5, 0)
  srcW = max((w+0.5)*scaleW-0.5, 0)

left   = min(floor(srcW), inW-1)
right  = left + (left < inW-1)
top    = min(floor(srcH), inH-1)
bottom = top + (top < inH-1)
dx = clamp(srcW-left, 0, 1); dy = clamp(srcH-top, 0, 1)
y = (1-dy)*((1-dx)*x[top,left] + dx*x[top,right])
  + dy*((1-dx)*x[bottom,left] + dx*x[bottom,right])
```

所有坐标和插值标量均用 FP32；FP16/BF16 输入读入 `float`，四点加权后赋回同 dtype 输出。`right/bottom` 在非边界处总保留相邻上邻点，即使对应权重恰为 0；这是最终 Direct 转换为匹配调用语义而从官方 `ceil(source)` 派生出的精度边界，不是原官方 arch35 body 的逐字复制。

### 输入、输出与属性

| 参数 | shape/layout | dtype | 约束、默认值与语义 |
|---|---|---|---|
| `x` | `[N,C,inH,inW]`，物理/逻辑连续 NCHW | FP16、FP32、BF16 | rank 必须为 4，四维均大于 0；拒绝非连续存储。 |
| `output_size` | 可选长度 2 数组 | `int64` 元数据 | 与 `scale_factor` 必须恰有一个存在；值为正，直接给出 `outH/outW`。 |
| `align_corners` | 标量，默认 false | bool | true 时两端中心对齐并令 `halfPixelCenters=0`；false 时令其为 1。 |
| `scale_factor` | 可选长度 2 数组 | Host double，POD 前降为 float | 两值必须有限且大于 0；代码检查 double 乘积有限、至少为 1 且 `<=static_cast<double>(INT64_MAX)`，随后 `floor` 并转为 int64；该检查本身不证明严格 int64 可表示性。 |
| `y` | `[N,C,outH,outW]`，连续 NCHW | 与 `x` 相同 | 由 `torch::empty` 分配；每个逻辑元素恰写一次。 |
| `stream` | 当前 NPU stream | `void*` | `resize_bilinear_npu` 从当前 NPU stream 取得并传入显式 kernel launch。 |

### 入口与版本

`TORCH_LIBRARY_FRAGMENT` 声明 `cann_bench::resize_bilinear(Tensor x, int[]? output_size=None, bool align_corners=False, float[]? scale_factor=None)`；Meta 路线只检查合同并分配元信息输出，PrivateUse1 绑定 `resize_bilinear_npu`。后者设置 device guard、要求连续 NCHW、分配输出、调用 `calc_resize_bilinear_config`，再在 `OpCommand::RunOpApi("ResizeBilinearDirect", ...)` 中按 dtype 调用三个 launcher 之一。

这是面向 950/Ascend 950、CANN 9.1.0 的 Direct 转换，不是 ACLNN 包装。官方固定提交的 `ResizeBilinearV2With4d` 提供 `half_pixel_centers=!align_corners`、scale 转发与 arch35 实现基线；Direct 转换删除 registry `gert::TilingContext`、`GET_TILING_DATA/TILING_KEY_IS` 和 `size/workspace/tiling` GM 参数，改为 Host 直接生成同序 POD、显式 dtype/half-pixel/index-width entry 以及按值 POD launch。最终载荷只保留 NCHW 的 SIMT_HW/SIMT_NCHW terminal family，没有保留官方 selector 中先于 SIMT 的特化 family。其中 all-copy 与 NCHW-broadcast 和 Direct 合法输入域重叠；point-copy、NHWC-broadcast、C-parallel 与 SIMT-NHWC 则要求 Direct 合同不接受的 NHWC。所有未保留 family 都不在本入口生产图中，也不是运行时 fallback。

## Host/Launcher 执行流程

### 参数校验与前处理

1. `CheckInputContract` 检查 rank=4、dtype 属于 FP16/FP32/BF16、N/C/H/W 全部为正。
2. `ResolveOutput` 强制 `output_size` 与 `scale_factor` 异或；前者检查长度和正值，后者检查长度、有限正值，并对 double 乘积执行 `>=1` 与 `<=static_cast<double>(INT64_MAX)` 检查后再 `floor`/转 int64；不把该 double 上界检查提升为严格可表示性证明。
3. `resize_bilinear_npu` 设置输入设备 guard，拒绝非连续输入，调用 `AllocateOutput` 创建同 dtype 的 `[N,C,outH,outW]`。
4. `calc_resize_bilinear_config` 再检查六个维度与显式 scale 正值；`CheckedProduct` 逐级防止 `H*W`、`N*C`、总输入/输出元素数发生 int64 溢出。
5. 从 `PlatformAscendCManager` 读取 AIV core 数和 UB 字节数；平台对象为空或 AIV core 数为 0 时抛错。UB 值写入 POD，但四条 SIMT 路线不消费它。
6. 配置函数生成 scale、index 宽度、Route、key、分核商余和 `blockDim`；plugin 按 torch dtype 选择 launcher，launcher 再按 `route/useIndex64/halfPixelCenters` 选择唯一显式 entry。

注意输出在 `calc_resize_bilinear_config` 的总元素溢出检查之前已请求分配；极端 shape 可能先由框架分配失败，而不是到达配置函数的异常。

### 参数、Workspace 与生命周期

`ResizeBilinearLaunchConfig` 在 Host 栈上零初始化；其中 `ResizeBilinearV2TilingData` 以值传给 kernel entry。配置和指针被 lambda 按值捕获，`x/y` 的存储由 torch tensor 持有，当前 stream 负责 launch ordering。显式 launch 的第二配置参数是 `nullptr`，四条 Route 均无 workspace 和 dynamic UB：`workspace_total=0`、`dynamic_ub=N/A`。官方 registry Host 原本固定声明 32B workspace，但两套 SIMT body 从不读取它，Direct 转换因此不保留该未消费边。

### 启动图与失败传播

```text
cann_bench::resize_bilinear
  -> resize_bilinear_npu
    -> AllocateOutput
    -> calc_resize_bilinear_config
    -> launch_resize_bilinear_{float32|float16|bfloat16}(x,y,config,stream)
      -> exactly one resize_bilinear_{hw|nchw}_{f32|f16|bf16}_{hp|corner}_{i32|i64}
         <<<config.blockDim, nullptr, stream>>>(x,y,tiling-by-value)
        -> ResizeBilinearV2Simt{HW|NCHW}<T,T,halfPixel,0,T_IDX>::Init
        -> Process
        -> asc_vf_call<callee...>(1024 or 512 threads)
        -> SimtCompute -> y
```

这是单 Kernel 图，没有 kernel 间 GM/workspace 边。Host 校验、乘积、平台查询失败会在 launch 前抛出；`RunOpApi` lambda 固定返回 0，源码没有把异步 kernel fault 编码进 lambda 返回值。正常 plugin 只能产生合法 enum；若绕过配置函数把非法 `route` 强制塞入 config，launcher 的 `else` 会按 NCHW 处理而非报错。所有正常调用的执行顺序由同一 stream 保证。

## Route 与参数设计

### 分核与资源计算

记 `I=N*C*inH*inW`、`O=N*C*outH*outW`、`OHW=outH*outW`、`Cores=GetCoreNumAiv()`。配置先用 int64 有保护乘法得到 `I/O/OHW`。索引宽度谓词为：

```text
useIdx32 = I < UINT32_MAX && O < UINT32_MAX
        && inH,inW,outH,outW <= INT32_MAX
threads  = useIdx32 ? 1024 : 512
needHWSplit = I != 1 && spatial_shape_changed
           && OHW >= Cores*threads
```

HW 路线固定 `realCoreNum=Cores`，以 `q=floor(OHW/Cores)`、`r=OHW-q*Cores` 分空间点；前 `r` 个 block 处理 `q+1` 点，其余处理 `q` 点。NCHW 路线取 `realCoreNum=min(O,Cores)`，同样以商余分整个扁平输出域。两者的 `blockDim=realCoreNum`，每个 block 内线程以 `threadIdx.x` 起步并按内部线程数步进。

### 生产可达性分类与穷尽证明

令 `A=useIdx32`、`H=needHWSplit(A)`；注意 `H` 的阈值依赖 `A`：i32 用 `Cores*1024`，i64 用 `Cores*512`。`calc_resize_bilinear_config` 对所有通过合同与 int64 乘积检查的输入都恰好产生下列一个 config；四个 config 值都有 typed launcher 分支、显式全局 entry 和 Device body，所以四条都是 `active` terminal Route。shape 谓词只是 active Route 的互斥选择条件，不把 i64 Route 降为 `conditional`。

```text
A  &&  H(A)  -> R-001 / SIMT_HW   / key 30004 / u32 / 1024 threads
A  && !H(A)  -> R-002 / SIMT_NCHW / key 30001 / u32 / 1024 threads
!A &&  H(A)  -> R-003 / SIMT_HW   / key 30005 / u64 /  512 threads
!A && !H(A)  -> R-004 / SIMT_NCHW / key 30003 / u64 /  512 threads
```

| Route | 类别 | Host producer 与 launch 闭包 | terminal 出边 |
|---|---|---|---|
| R-001 | `active` | `A&&H` 写 `route=SIMT_HW,useIndex64=0,key=30004`；typed launcher 命中六个 HW/u32 entry 之一 | selector 终止；Device 无失败重试。`H` 为假才在 launch 前产生 R-002 |
| R-002 | `active` | `A&&!H` 写 `route=SIMT_NCHW,useIndex64=0,key=30001`；typed launcher 命中六个 NCHW/u32 entry 之一 | selector 终止；Device 无后续 Route。`A` 为假会用 512 阈值重新计算 `H`，再产生 R-003/R-004 |
| R-003 | `active` | `!A&&H` 写 `route=SIMT_HW,useIndex64=1,key=30005`；typed launcher 命中六个 HW/u64 entry 之一 | selector 终止；Device 无失败重试。`H` 为假才在 launch 前产生 R-004 |
| R-004 | `active` | `!A&&!H` 写 `route=SIMT_NCHW,useIndex64=1,key=30003`；typed launcher 命中六个 NCHW/u64 entry 之一 | selector 的安全终点；合法 config 没有下一条 Route |

本实现没有独立 `conditional` terminal Route。`halfPixelCenters` 与 dtype 只是每条 Route 内的编译期 entry 轴；它们不改变 ownership、分核域或 key，不能拆成额外 Route。

### Dormant source 与官方图外 family

最终 NCHW header 保留三个 `if constexpr(mode==...)` 分支，但全部 12 个 Direct NCHW entry 都实例化 `mode=0`；它们是生产不可达的 dormant source，不是 terminal Route：

| dormant 分支 | 保留的 Device 语义 | 生产链不可达原因 | 重新启用条件 |
|---|---|---|---|
| D-001 `SIMT_SCENE_ALL_COPY=1` | 按解码后的 N/C/H/W 从输入同位置复制 | entry 宏固定模板参数 `mode=0`，Host/launch config 没有 mode 字段，launcher 也没有 mode 分支 | 增加互斥 Host 谓词、mode producer 或独立 route、显式 entry/launcher；证明同 shape、两种坐标模式、dtype 以及零权重 NaN/Inf 的 identity 语义 |
| D-002 `SIMT_SCENE_INPUT_ONE=2` | 每个输出读取 `input[N*C+C]` | 同上；当前 `1x1` 输入仍由 R-001/R-002 的通用插值 body处理 | 增加 producer/entry/launcher，并闭合 `inH=inW=1` 的索引、分核及零权重 NaN/Inf 语义 |
| D-003 `SIMT_SCENE_OUTPUT_ONE=3` | 每个输出读取每个 N/C 平面的首元素 | 同上；当前 `outH=outW=1` 仍走通用 body | 增加 producer/entry/launcher，并证明 align/scale、相邻样本与零权重 NaN/Inf 下首元素直拷仍等价 |

官方固定 selector 还有下列六个 family，但最终 Direct ZIP 没有打包其 body，也没有对应 enum、config producer、launcher 或 entry，因此它们属于“官方图外 family”，不能仅凭官方 key 称为 Direct dormant Route。表中“此前未命中”严格指官方 selector 的行序短路：

| 官方 family/key | 精确 selector（按官方优先级） | 官方 entry/body 与数据依赖 | Direct 状态 | 若要纳入 Direct 必须补齐 |
|---|---|---|---|---|
| all-copy / 40000 | 首项；`dtypeX==dtypeY && inH==outH && inW==outW`，格式不限 | `ResizeBilinearV2AllCopy<DTYPE_X>`；`TPipe + TQueBind<VECIN,VECOUT>`，消费 `realCoreNum/splitBlockFactor/splitBlockTailFactor/ubCFactor` 做 GM↔UB 双缓冲复制 | NCHW 同 shape 与 Direct 合法域重叠，但 Direct 让它进入 R-002/mode 0 | 打包 body，补 Host producer、entry/launcher 与 UB queue；证明直拷不会改变零权重 NaN/Inf、block/UB 尾块语义 |
| point-copy / 40001 | all-copy 未命中；`dtypeX==dtypeY && format==NHWC && C*sizeof(X)>=128 && scaleH>=1 && scaleW>=1`；令 `qH=int64(scaleH+1e-6),qW=int64(scaleW+1e-6)`，还要求 `abs(scaleH-qH)<1e-6 && abs(scaleW-qW)<1e-6`，且 `alignCorners!=0 \|\| halfPixelCenters!=1 \|\| (qH,qW均为奇数)` | `ResizeBilinearV2PointCopy<DTYPE_X>`；`TPipe` 与 point-copy UB/分核参数，消费 `n/h/w/cFactor` 及 `ubN/H/W/CFactor` | Direct 固定 NCHW，生产不可达 | 扩展 schema/layout 到 NHWC；移植整数 scale 与奇偶 guard、完整 factor producer、body/entry/launcher、UB/尾块和精度边界 |
| NCHW-broadcast / 40002 | 前两项未命中；`dtypeX==dtypeY && format==NCHW && outH*outW*sizeof(Y)>=128 && inH==1 && inW==1` | `ResizeBilinearV2BroadcastNCHW<DTYPE_X>`；`xQue/yQue`、`DataCopyPad`、Vector `Duplicate`、MTE2_S/S_V event，消费 N/C/HW 与 UB factor | 与 Direct 的 `1x1` 输入子域重叠；Direct 仍选择 R-001 或 R-002 | 打包 body，恢复 N/C/HW 与 UB tiling producer、POD consumer、entry/launcher；审计重复样本参与 FP32 表达式的 NaN/Inf 边界 |
| NHWC-broadcast / 40003 | 前三项未命中；`dtypeX==dtypeY && format==NHWC && C*sizeof(X)>=128 && inH==1 && inW==1` | `ResizeBilinearV2BroadcastNHWC<DTYPE_X>`；`TPipe`、NHWC GM↔UB broadcast queues 与 N/HW/C factor | Direct 固定 NCHW，生产不可达 | 扩展 NHWC storage 解释；恢复 factor/UB producer、body/entry/launcher、同步与 C/HW tail |
| C-parallel / 10000 | 前四项未命中；`format==NHWC && C*sizeof(X)>=128 && scaleH>=2 && scaleW>=2`；此 predicate 不要求输入输出 dtype 相同 | key 先进入二级分派：`cFactor<lenC` 调 `ResizeBilinearV2Nc<DTYPE_X,DTYPE_Y>`，否则调 `ResizeBilinearV2CParallel<DTYPE_X,DTYPE_Y>`；两者消费 C-parallel 的 N/HW/C 与 UB factors | Direct 固定 NCHW 且同 dtype，生产不可达 | 扩展 NHWC/dtype 合同；移植 `DoTilingCParallel`、两套 body、`cFactor<lenC` 二级 router、entry/launcher、UB 账与分派闭包 |
| SIMT-NHWC / 30000、30002 | 前五项未命中且 `format==NHWC`；`useIdx32` 真取 30000/u32/1024 threads，假取 30002/u64/512 threads | `ResizeBilinearV2SimtNHWC<DTYPE_X,DTYPE_Y,halfPixel,mode,T_IDX>`；`halfPixelCenters>0` 固定 `halfPixel=true,mode=0`，否则按同 shape、输入 1x1、输出 1x1、其余依次选 mode 1/2/3/0 | Direct 固定 NCHW，生产不可达 | 扩展 NHWC schema/layout；打包 SIMT-NHWC body，补 u32/u64、half-pixel/mode 显式 entry 与 launcher，并闭合 NHWC 地址解码和尾块 |

### Route 重要性

P0 为最高优先级，以下顺序用于组织实现知识，不表示 Host 会按表格顺序逐条尝试 Route。

| 优先级 | Route | 状态 | Route 特征 | 适用范围 | 实现前提 |
|---|---|---|---|---|---|
| P0 | R-001 SIMT_HW/i32 | active | 用 `uint32_t` 索引切分 OHW，每个空间点在同一 thread 内循环 N/C；1024 threads | 输入输出总元素可安全用 i32 寻址，空间尺寸改变且 OHW 足以支撑全核空间并行 | `useIdx32 && needHWSplit`；连续 NCHW、同 dtype、AIV core 数有效 |
| P1 | R-002 SIMT_NCHW/i32 | active safety | 用 `uint32_t` 索引切分完整 NCHW 输出域；1024 threads | i32 寻址安全，但空间不变、单元素输入或 OHW 不足以启用 HW ownership | `useIdx32 && !needHWSplit`；连续 NCHW、同 dtype、正的完整输出域 |
| P2 | R-003 SIMT_HW/i64 | active | 用 `uint64_t` 索引的 HW ownership；512 threads | i32 寻址不安全，且空间尺寸改变、OHW 达到 i64 特化的全核阈值 | `!useIdx32 && needHWSplit`；所有 Host int64 乘积检查成功 |
| P3 | R-004 SIMT_NCHW/i64 | active safety | 用 `uint64_t` 索引切分完整 NCHW 输出域；512 threads | i32 寻址不安全，且 HW ownership 谓词为假的大索引域 | `!useIdx32 && !needHWSplit`；所有 Host int64 乘积检查成功 |

### Host selector 顺序、短路与未命中备选

Host 的真实决策顺序是：先完成合同、乘积和平台检查；再计算 `useIdx32`，由它选择 1024 或 512 的 `threadThreshold`；随后只计算一次 `needHWSplit`，真则进入 HW 分支，假则进入 NCHW 分支；最后在分支内用 `useIdx32` 定 key 和 index specialization。因此 R-002/R-004 是 `needHWSplit` 的 `else` 结果，R-003/R-004 是 index-width 决策结果，不是前一 Route 启动失败后再试的运行时 fallback。

| Host 步骤 | 判定/短路 | 结果或未命中时 selector 备选 |
|---:|---|---|
| 1 | 合同、正维度、int64 乘积、platform 和 AIV core 检查 | 任一失败立即抛错，不生成 Route，也没有 Device fallback |
| 2 | `useIdx32 = I<UINT32_MAX && O<UINT32_MAX && H/W extents<=INT32_MAX` | 真时选 1024 threads 和 i32 specialization；假时直接选 512 threads 和 i64 specialization |
| 3 | `needHWSplit = I!=1 && spatial_shape_changed && OHW>=Cores*threads` | 真时选 HW，假时选 NCHW；两分支都直接生成唯一 Route |
| 4 | plugin 先按 dtype 调 typed launcher；launcher 再按 `route`、`useIndex64`、`halfPixelCenters` 短路 | 合法 config 命中唯一 entry；绕过 config 传入非 `SIMT_HW` route 时，launcher 的 `else` 落到 NCHW；非零 index/HP 标志分别落到 i64/half-pixel |

```text
validate/products/platform
  -> useIdx32?
     -> yes: needHWSplit ? R-001(key 30004) : R-002(key 30001)
     -> no : needHWSplit ? R-003(key 30005) : R-004(key 30003)
  -> dtype {f32,f16,bf16}
  -> halfPixelCenters ? *_hp_* : *_corner_*
  -> one explicit entry
```

### 模板/实现执行矩阵

| 优先级 | Route | key/selector | dtype/shape/platform 完整命中条件 | 执行拓扑与 Kernel 节点 | 实际模板/body | 内存与同步特征 | tail/去向 |
|---|---|---|---|---|---|---|---|
| P0 | R-001 SIMT_HW/i32 | 30004 + `route=SIMT_HW,useIndex64=0` | 3 dtype；正的连续 NCHW；i32 安全；空间改变；`OHW>=Cores*1024`；950 | 单 AIV entry→SIMT | `ResizeBilinearV2SimtHW<T,T,HP,0,uint32_t>::Process/SimtCompute` | 无 UB/L1/L0/workspace；1024 threads；每线程固定 H/W 后串行 N、C | block 商余 + loop-bound tail；`H` 为假时在 launch 前生成 R-002 |
| P1 | R-002 SIMT_NCHW/i32 | 30001 + `route=SIMT_NCHW,useIndex64=0` | 同 dtype/平台/index，且 HW 谓词为假 | 单 AIV entry→SIMT | `ResizeBilinearV2SimtNCHW<T,T,HP,0,uint32_t>::Process/SimtCompute` | 无 UB/L1/L0/workspace；1024 threads；扁平 NCHW 解码 | block 商余 + loop-bound tail；index 不安全时以 512 阈值重算 `H`，选 R-003/R-004 |
| P2 | R-003 SIMT_HW/i64 | 30005 + `route=SIMT_HW,useIndex64=1` | 3 dtype；i32 谓词失败；空间改变；`OHW>=Cores*512`；950 | 单 AIV entry→SIMT | `ResizeBilinearV2SimtHW<T,T,HP,0,uint64_t>::Process/SimtCompute` | 无 UB/L1/L0/workspace；512 threads；64-bit magic/index | 同 R-001；`H` 为假时在 launch 前生成 R-004 |
| P3 | R-004 SIMT_NCHW/i64 | 30003 + `route=SIMT_NCHW,useIndex64=1` | i32 谓词失败且 HW 谓词为假 | 单 AIV entry→SIMT | `ResizeBilinearV2SimtNCHW<T,T,HP,0,uint64_t>::Process/SimtCompute` | 无 UB/L1/L0/workspace；512 threads；64-bit 扁平解码 | 同 R-002；正常 config 无后续 Route |

其中 `T` 分别是 `float/half/bfloat16_t`，`HP=true/false` 分别来自 `halfPixelCenters!=0/==0`。key 写入 POD 用于保持官方映射和诊断；最终 launcher 实际按 `route/useIndex64/halfPixelCenters` 分支，不读取 `tilingKey` 做 router。

### 关键性能 Guard

| Guard | Host/Tiling 条件 | Device/资源原因 | 影响 Route | 不满足时去向 | 源码符号 |
|---|---|---|---|---|---|
| 合同域 | rank=4、连续、3 种 dtype、各维正、两种 size producer 异或 | body 只实现连续 NCHW 同 dtype寻址且无空域保护 | 全部 | Host 抛错 | Direct `CheckInputContract/ResolveOutput/resize_bilinear_npu` |
| int64 乘积 | `a>0 && b>0 && a<=INT64_MAX/b` | 防止 Host 分核和 GM extent 回绕 | 全部 | Host 抛错 | Direct `CheckedProduct` |
| i32 index | `I,O<UINT32_MAX` 且四个 H/W extent `<=INT32_MAX` | `T_IDX=uint32_t` 的扁平索引、乘法与 magic division 必须可表示 | R-001/R-002 | i64 Route | `calc_resize_bilinear_config::useIdx32`；官方 `MatchTilingStrategyAndSetTilingKey` |
| HW 阈值 | `spatial changed && I!=1 && OHW>=Cores*threads` | HW body 每空间点复用一次坐标解码并循环 N/C；阈值还保证每个 core 至少有充足 thread 域 | R-001/R-003 | 对应 NCHW | `needHWSplit` |
| SIMT thread 数 | i32=1024、i64=512 | 与 `LAUNCH_BOUND` 和索引宽度 specialization 一致 | 四 Route | 无动态替代 | `THREAD_NUM/THREAD_NUM_MIDDLE`、`calleeInt*` |
| block tail | `blockIdx < splitBlockTailFactor` | 将不能整除的前 `r` 个输出块各增 1 个元素，保证无遗漏/重叠 | 四 Route | N/A | 两个 `Process` |
| 半像素编译期分支 | `halfPixelCenters=!alignCorners` | 消除每线程运行时坐标模式分支 | 四 Route | 选择对应 `corner` entry | `LAUNCH_RESIZE_TYPED`、模板参数 `halfPixel` |
| dtype 显式 entry | plugin 仅调用三个 typed launcher | 每个 entry 固化 GM 元素类型和输出转换 | 四 Route | 非法 dtype 在 plugin 先失败 | `launch_resize_bilinear_*` |
| UB 查询非 selector | 无条件读取 UB 并写 `td.ubSize` | 当前 SIMT body 不分配/消费 UB；该值不限制 Route | 四 Route | N/A | `calc_resize_bilinear_config`、两个 body |

### POD/ABI

`ResizeBilinearV2TilingData` 是 standard-layout、trivially-copyable、无 packing 的声明；按声明自然对齐为 `sizeof=192, alignof=8`。最终载荷只断言了两个类型性质，未用 `static_assert` 锁死 `sizeof/offsetof`；下列 offset 是该声明在目标 C++ ABI 下的机械推导，跨 ABI 复用应补尺寸断言。`ResizeBilinearLaunchConfig` 相应为 `sizeof=208, alignof=8`。

| 结构 | 字段 | 类型/位宽 | ABI 布局 | 单位 | Host/launcher producer | Device consumer | 边界 |
|---|---|---|---|---|---|---|---|
| `ResizeBilinearV2TilingData` | `tilingKey` | int64/64 | offset 0 | key | config 选 Route 时写 30001/3/4/5 | Device 不读；保留官方映射 | 不驱动最终 launcher |
| 同上 | `realCoreNum` | int64/64 | offset 8 | AIV block | HW=`Cores`；NCHW=`min(O,Cores)` | HW `Process` 读入未使用局部量；其余不读 | 必须大于 0 |
| 同上 | `ubSize` | int64/64 | offset 16 | byte | 平台 UB query | 不读 | 非 Route guard |
| 同上 | `alignCorners` | int64/64 | offset 24 | bool-as-int | `alignCorners?1:0` | 不读 | 与下一字段互补 |
| 同上 | `halfPixelCenters` | int64/64 | offset 32 | bool-as-int | `alignCorners?0:1` | launcher 选 `HP` entry；body 为编译期参数 | 只允许 0/1 producer |
| 同上 | `lenN` | int64/64 | offset 40 | element extent | `n` | HW N 循环；NCHW 解码/extent | `>0`；进入 T_IDX |
| 同上 | `lenC` | int64/64 | offset 48 | element extent | `c` | HW C 循环；NCHW 解码/stride | `>0`；进入 T_IDX |
| 同上 | `lenSrcH` | int64/64 | offset 56 | element extent | `inH` | 邻点 clamp、输入 stride | `>0` |
| 同上 | `lenSrcW` | int64/64 | offset 64 | element extent | `inW` | 邻点 clamp、输入 stride | `>0` |
| 同上 | `lenDesH` | int64/64 | offset 72 | element extent | `outH` | magic division、输出 stride | `>0` |
| 同上 | `lenDesW` | int64/64 | offset 80 | element extent | `outW` | magic division、输出 stride | `>0` |
| 同上 | `nFactor` | int64/64 | offset 88 | element | `n` | 不读 | Direct 保留字段 |
| 同上 | `hFactor` | int64/64 | offset 96 | element | `outH` | 不读 | Direct 保留字段 |
| 同上 | `wFactor` | int64/64 | offset 104 | element | `outW` | 不读 | Direct 保留字段 |
| 同上 | `cFactor` | int64/64 | offset 112 | element | `c` | 不读 | Direct 保留字段 |
| 同上 | `hwFactor` | int64/64 | offset 120 | element | `OHW` | 不读 | Direct 保留字段 |
| 同上 | `ubNFactor` | int64/64 | offset 128 | element | `n` | 不读 | Direct 保留字段 |
| 同上 | `ubHFactor` | int64/64 | offset 136 | element | `outH` | 不读 | Direct 保留字段 |
| 同上 | `ubWFactor` | int64/64 | offset 144 | element | `outW` | 不读 | Direct 保留字段 |
| 同上 | `ubCFactor` | int64/64 | offset 152 | element | `c` | 不读 | Direct 保留字段 |
| 同上 | `splitBlockFactor` | int64/64 | offset 160 | output element 或 HW point/block | domain/realCoreNum | 两个 `Process` 的 block 基数 | `>=1` |
| 同上 | `splitBlockTailFactor` | int64/64 | offset 168 | block count | domain-q*cores | 两个 `Process` 的前缀余数 | `[0,realCoreNum)` |
| 同上 | `scaleW` | float/32 | offset 176 | source pixel/output pixel | scale 公式 | `SimtCompute` 坐标 | 正有限合同下非负 |
| 同上 | `scaleH` | float/32 | offset 180 | source pixel/output pixel | scale 公式 | `SimtCompute` 坐标 | 同上 |
| 同上 | `ubHWFactor` | int64/64 | offset 184 | element | `OHW` | 不读 | 由 4B padding 后 8B 对齐 |
| `ResizeBilinearLaunchConfig` | `tiling` | struct/192B | offset 0 | POD | `calc_resize_bilinear_config` | entry 按值接收内层 POD | 必须与其他三字段一致 |
| 同上 | `blockDim` | uint32/32 | offset 192 | AIV block | `uint32_t(realCoreNum)` | kernel launch grid | core 数来自 uint32 query |
| 同上 | `route` | enum:uint32/32 | offset 196 | selector | `SIMT_HW/NCHW` | launcher | 非 HW 值落到 NCHW else |
| 同上 | `useIndex64` | uint32/32 | offset 200 | bool-as-int | `useIdx32?0:1` | launcher 选 i32/i64 entry | 正常 producer 只写 0/1 |

### Route 源码闭包

| 优先级 | Route | Key/selector | Host/Tiling producer | POD/参数 | Kernel entry/router 节点 | Kernel body/直接依赖 | 构建接线 |
|---|---|---|---|---|---|---|---|
| P0 | R-001 SIMT_HW/i32 | 30004 / `useIdx32&&needHWSplit` | 官方 SIMT 子路径 `MatchTilingStrategyAndSetTilingKey/DoTilingSIMT_HW` → Direct `calc_resize_bilinear_config`；Direct 不保留官方前置特化 | 同序 `ResizeBilinearV2TilingData` + config | `resize_bilinear_hw_{f32,f16,bf16}_{hp,corner}_i32` | `ResizeBilinearV2SimtHW<...,uint32_t>::Process/SimtCompute` | 官方 `SUPPORT_COMPUTE_UNIT=ascend950`；Direct registration 单 kernel TU |
| P1 | R-002 SIMT_NCHW/i32 | 30001 / `useIdx32&&!needHWSplit` | 官方 SIMT 子路径/`DoTilingSIMT` → Direct config；Direct 将官方前置特化覆盖域并入此路 | 同上 | `resize_bilinear_nchw_*_{hp,corner}_i32` | `ResizeBilinearV2SimtNCHW<...,uint32_t>` + quick-div helper | 同上 |
| P2 | R-003 SIMT_HW/i64 | 30005 / `!useIdx32&&needHWSplit` | 同 R-001，线程阈值 512 | 同上 | `resize_bilinear_hw_*_{hp,corner}_i64` | `ResizeBilinearV2SimtHW<...,uint64_t>` | 同上 |
| P3 | R-004 SIMT_NCHW/i64 | 30003 / `!useIdx32&&!needHWSplit` | 同 R-002，index/magic 改为 u64 | 同上 | `resize_bilinear_nchw_*_{hp,corner}_i64` | `ResizeBilinearV2SimtNCHW<...,uint64_t>` | 同上 |

## Kernel 实现

### 公共执行流程

每个显式 entry 接收 `(x,y,tilingData-by-value)`，声明 `KERNEL_TYPE_AIV_ONLY`，构造对应模板类并执行 `Init→Process`。`Init` 绑定输入/输出 GM 并保存外层 AIV `blockIdx`。`Process` 计算 magic/shift 整除参数和本 block 的商余区间，再以 i32 路线 1024 threads、i64 路线 512 threads 调用 SIMT callee。每线程只处理本 block 区间中的 `idx=threadIdx.x+k*threads`；block 商余闭合核尾，循环条件 `idx<blkProcessNum` 闭合线程尾，代码没有额外 Vector mask。无共享内存、atomic 或 barrier。坐标 clamp、四点 GM load、FP32 插值及同 dtype store 是所有 Route 的公共计算核心。

### P0 · Route R-001：SIMT_HW/i32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host baseline | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | `MatchTilingStrategyAndSetTilingKey` | 官方 all-copy/point/broadcast/C-parallel 均未先命中，且 i32 安全、NCHW、空间改变、`OHW>=coreNum*1024` | 官方 key 30004；前置特化命中时不是此 key |
| Direct selector | Direct 派生层（最终载荷） | `calc_resize_bilinear_config` | `useIdx32&&needHWSplit`；限定连续 NCHW/同 dtype，但不执行官方前置特化 selector | `route=SIMT_HW,key=30004,blockDim=Cores`；较大 1x1 输入广播子域也可由本 Route 承接 |
| Router | Direct 派生层（最终载荷） | `LAUNCH_RESIZE_TYPED` | dtype + HW + `useIndex64=0` + half-pixel 标志 | 六个 `*_i32` HW entry 之一 |

#### Device 数据流

##### 函数级调用链

```text
resize_bilinear_hw_T_{hp|corner}_i32
  -> ResizeBilinearV2SimtHW<T,T,HP,0,uint32_t>::Init
  -> Process
    -> GetUintDivMagicAndShift(outW/outH)
    -> asc_vf_call<calleeInt32>(1024 threads)
      -> SimtCompute
```

##### 源码等价伪码

```text
q=OHW/Cores; r=OHW-q*Cores
count=q+(block<r); start=block*q+min(block,r)
for hw=start+threadIdx; hw<start+count; hw+=1024:
  (H,W)=quick_divmod(hw,outH,outW)
  derive source coordinate and four neighbors
  for n in [0,N):
    for c in [0,C):
      base=n*C*inH*inW+c*inH*inW
      out=n*C*OHW+c*OHW+hw
      output[out]=bilinear(float(input[four neighbors]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `n*C*inHW+c*inHW+srcH*inW+srcW` | `I*sizeof(T)` | T/NCHW | caller | 四次 scalar load | input tensor/整次 launch | 无 DMA tile；邻点重合时源码仍可重复 load |
| output GM | `n*C*OHW+c*OHW+hw` | `O*sizeof(T)` | T/NCHW | SIMT thread | caller/后续 stream op | output tensor | 由唯一 HW owner 写，无重叠 |
| thread-private | 当前 H/W、四邻点、delta、magic 参数 | 固定标量集合，编译器寄存器数量不可见 | uint32/float | thread | 同 thread | 单次 loop iteration | 无跨 thread 复用 |

##### 分层内存布局

N/A：该 SIMT 节点没有 AIV `TQue/TBuf`、shared/dynamic UB 或 workspace；实际路径是线程私有标量加 GM scalar load/store。

##### 字节账与复用顺序

```text
UB_total = 0; L1_total = 0; L0A/L0B/L0C_total = 0
dynamic_ub = N/A (asc_vf_call launch 参数不含动态 UB)
workspace_total = 0
GM_live = I*sizeof(T) input + O*sizeof(T) output
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程私有坐标/权重标量 | 当前 `hw` iteration | 同 iteration 四点插值 | 下一 `hw` | 线程程序顺序 | 不逃逸到其他 thread |
| 无共享子区 | N/A | N/A | N/A | 无 barrier/queue | 每个输出地址只有一个 HW owner |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| block 切分 | quotient/remainder | Host POD | `Process` | HW 点域 | 前 `r` block 多 1 点；thread loop 再覆盖 block 内 tail |
| 坐标解码 | `Simt::UintDiv` + magic/shift | `Process` | `SimtCompute` | H/W | uint32；outH/outW 正值 |
| 插值 | 线程程序顺序 | 四次 GM load | 单次 GM store | 一个输出元素 | scale/坐标/权重/累加均 FP32，写 T 时降精度 |
| 可见性 | 当前 stream | kernel | 后续同 stream work | output GM | 无显式 atomic/barrier；唯一写地址 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | `MatchTilingStrategyAndSetTilingKey/DoTilingSIMT_HW` | 官方 HW/i32 阈值、key 与商余公式基线 |
| Kernel entry/router | `op_kernel/resize_bilinear_v2_apt.cpp` | key 30004 branch | 官方模板为 `SimtHW<...,true\|false,0,uint32_t>`；Direct 改为显式 entry |
| Device body/helper | `op_kernel/arch35/resize_bilinear_v2_simt_hw.h` | `Process/calleeInt32/SimtCompute` | 1024 threads、HW ownership、N/C 内循环与四点插值 |

### P1 · Route R-002：SIMT_NCHW/i32

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host baseline | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | `MatchTilingStrategyAndSetTilingKey` | 官方 all-copy/point/broadcast/C-parallel 均未先命中，且 i32 安全、NCHW、HW 谓词为假 | 官方 key 30001；前置特化命中时不是此 key |
| Direct selector | Direct 派生层（最终载荷） | `calc_resize_bilinear_config` | `useIdx32&&!needHWSplit`；不执行官方前置特化 selector | `realCoreNum=min(O,Cores)`、key 30001；同 shape/1x1 等子域也由本 Route 承接 |
| Router | Direct 派生层（最终载荷） | `LAUNCH_RESIZE_TYPED` | dtype + NCHW + i32 + half-pixel | 六个 NCHW i32 entry 之一 |

#### Device 数据流

##### 函数级调用链

```text
resize_bilinear_nchw_T_{hp|corner}_i32
  -> ResizeBilinearV2SimtNCHW<T,T,HP,0,uint32_t>::Init
  -> Process
    -> GetUintDivMagicAndShift(outW,outH,C,N)
    -> asc_vf_call<calleeInt32nchw>(1024 threads)
      -> SimtCompute -> QuickDivForSimtComputenchw
```

##### 源码等价伪码

```text
cores=min(O,Cores); q=O/cores; r=O-q*cores
count=q+(block<r); start=block*q+min(block,r)
for linear=start+threadIdx; linear<start+count; linear+=1024:
  (N,C,H,W)=quick_divmod_NCHW(linear)
  derive source coordinate and four neighbors
  base=N*Cdim*inHW+C*inHW
  output[linear]=bilinear(float(input[base+four neighbors]))
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input GM | `N*Cdim*inHW+C*inHW+srcH*inW+srcW` | `I*sizeof(T)` | T/NCHW | caller | 当前 output thread | tensor | 每输出最多四次 scalar load |
| output GM | `linear` | `O*sizeof(T)` | T/NCHW | 唯一 SIMT thread | caller | tensor | 扁平连续域，无重叠 |
| thread-private | N/C/H/W、邻点、权重 | 固定标量，寄存器数量不可见 | uint32/float | thread | 同 thread | iteration | 下一元素覆盖 |

##### 分层内存布局

N/A：无 AIV UB、dynamic UB 或 workspace；使用线程私有标量和 GM。

##### 字节账与复用顺序

```text
UB_total=0; L1_total=0; L0A/L0B/L0C_total=0
dynamic_ub=N/A; workspace_total=0
GM_live=I*sizeof(T)+O*sizeof(T)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| 线程私有解码/插值标量 | 当前 linear | 当前 store | 下一 linear | 程序顺序 | thread 独占 |
| 无共享子区 | N/A | N/A | N/A | 无 barrier/atomic | 扁平输出切分互斥 |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| block/thread tail | 商余 + 1024 stride | Host/Process | callee loop | O 个输出 | `linear<start+count` 保护尾部 |
| NCHW 解码 | 四组 magic division | `Process` | quick-div helper | N/C/H/W | uint32 |
| 插值/写回 | 程序顺序 | GM loads | GM store | 单输出 | FP32 计算后转 T；相邻上邻点规则保留 0 权重项 |
| stream ordering | stream | launch | 后续同 stream op | output GM | 无跨 block 同步需求 |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | `DoTilingSIMT` | `min(O,Cores)` 与整输出商余 |
| Kernel entry/router | `op_kernel/resize_bilinear_v2_apt.cpp` | key 30001 branch | 官方 i32 NCHW 模板映射；Direct 展开为显式 entry |
| Device body/helper | `op_kernel/arch35/resize_bilinear_v2_simt_nchw.h` | `QuickDivForSimtComputenchw/Process/SimtCompute` | 扁平输出 ownership、1024 threads、四维快速解码 |

### P2 · Route R-003：SIMT_HW/i64

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host baseline | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | `MatchTilingStrategyAndSetTilingKey` | 官方前置特化均未命中，且 i32 谓词失败、NCHW、空间改变、`OHW>=coreNum*512` | key 30005 |
| Direct selector | Direct 派生层（最终载荷） | `calc_resize_bilinear_config` | `!useIdx32&&needHWSplit`；不执行官方前置特化 selector | HW/i64，`blockDim=Cores` |
| Router | Direct 派生层（最终载荷） | `LAUNCH_RESIZE_TYPED` | dtype + HW + i64 + half-pixel | 六个 HW i64 entry 之一 |

#### Device 数据流

##### 函数级调用链

```text
resize_bilinear_hw_T_{hp|corner}_i64
  -> ResizeBilinearV2SimtHW<T,T,HP,0,uint64_t>::Init/Process
  -> asc_vf_call<calleeInt64>(512 threads)
  -> SimtCompute<uint64_t>
```

##### 源码等价伪码

```text
与 R-001 相同的 HW 商余、H/W ownership 和 N/C 内循环；
所有 domain/index/magic 变量改为 uint64，thread stride 改为 512。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | 同 R-001，地址算术为 uint64 | `I/O*sizeof(T)` | T/NCHW | caller/thread | thread/caller | tensor | 唯一 HW owner |
| thread-private | 同 R-001 | 固定标量，寄存器数量不可见 | uint64/float | thread | thread | iteration | 512-thread specialization |

##### 分层内存布局

N/A：无 UB/dynamic UB/workspace；GM + thread-private 路径。

##### 字节账与复用顺序

```text
UB_total=0; L1_total=0; L0A/L0B/L0C_total=0
dynamic_ub=N/A; workspace_total=0
GM_live=I*sizeof(T)+O*sizeof(T)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private uint64/float 标量 | iteration | 同 iteration | 下一 iteration | 程序顺序 | 不跨 thread |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分核/tail | 同 R-001 | Host POD | Process/thread loop | OHW | uint64 index，512 stride |
| 插值 | 同 R-001 | scalar GM load | scalar GM store | 单输出 | FP32 math；T store |
| 同步 | stream + 唯一地址 | kernel | 后续 stream op | output | 无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | i64 `needHWSplit`/`DoTilingSIMT_HW` | 512 阈值、key 30005、HW 商余 |
| Kernel entry/router | `op_kernel/resize_bilinear_v2_apt.cpp` | key 30005 branch | `SimtHW<...,uint64_t>` 映射 |
| Device body/helper | `op_kernel/arch35/resize_bilinear_v2_simt_hw.h` | `calleeInt64/Process/SimtCompute` | 512 threads 与 uint64 路径 |

### P3 · Route R-004：SIMT_NCHW/i64

#### 适用条件

##### 选择证据

| 层级 | 文件 | 符号 | 源码谓词/参数 | 结果/下一跳 |
|---|---|---|---|---|
| Host baseline | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | `MatchTilingStrategyAndSetTilingKey` | 官方前置特化均未命中，且 i32 谓词失败、NCHW、i64 HW 谓词为假 | key 30003 |
| Direct selector | Direct 派生层（最终载荷） | `calc_resize_bilinear_config` | `!useIdx32&&!needHWSplit`；不执行官方前置特化 selector | NCHW/i64 final route |
| Router | Direct 派生层（最终载荷） | `LAUNCH_RESIZE_TYPED` | dtype + NCHW + i64 + half-pixel | 六个 NCHW i64 entry 之一 |

#### Device 数据流

##### 函数级调用链

```text
resize_bilinear_nchw_T_{hp|corner}_i64
  -> ResizeBilinearV2SimtNCHW<T,T,HP,0,uint64_t>::Init/Process
  -> asc_vf_call<calleeInt64nchw>(512 threads)
  -> SimtCompute<uint64_t> -> QuickDivForSimtComputenchw<uint64_t>
```

##### 源码等价伪码

```text
与 R-002 相同地切分并解码完整 O 域；
T_IDX/magic/offset 为 uint64，线程步长为 512。
```

#### 内存与数据搬运

##### 资源与搬运表

| 层级/缓冲区 | offset/地址公式 | extent/容量公式 | dtype/layout | producer | consumer | owner/lifetime | 对齐/复用 |
|---|---|---|---|---|---|---|---|
| input/output GM | 同 R-002，uint64 地址算术 | `I/O*sizeof(T)` | T/NCHW | caller/thread | thread/caller | tensor | 每输出唯一 owner |
| thread-private | N/C/H/W、magic、邻点、权重 | 固定标量，寄存器数量不可见 | uint64/float | thread | thread | iteration | 512-thread specialization |

##### 分层内存布局

N/A：无 UB/dynamic UB/workspace；GM + thread-private 路径。

##### 字节账与复用顺序

```text
UB_total=0; L1_total=0; L0A/L0B/L0C_total=0
dynamic_ub=N/A; workspace_total=0
GM_live=I*sizeof(T)+O*sizeof(T)
```

| 复用子区 | 写入者 | 读取者/消费点 | 下一次覆盖 | 同步或释放依据 | 安全条件 |
|---|---|---|---|---|---|
| thread-private uint64/float 标量 | iteration | 同 iteration | 下一 iteration | 程序顺序 | 不跨 thread |

#### 同步、精度与尾块

##### 同步与数值边界表

| 阶段 | primitive/机制 | producer | consumer/等待点 | 保护对象 | 精度/尾块 |
|---|---|---|---|---|---|
| 分核/tail | 商余 + 512 stride | Host/Process | callee loop | O | `linear<start+count` |
| 解码 | uint64 magic division | Process | helper | N/C/H/W | extent 先从 int64 转 uint64 |
| 插值/同步 | FP32 scalar + stream | thread/kernel | store/后续 op | output | 同 R-002；无 barrier/atomic |

#### 关键实现

##### 符号证据表

| 阶段 | 文件 | 符号/模板实例 | 实现事实 |
|---|---|---|---|
| Host/Tiling producer | `op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | i64 selector/`DoTilingSIMT` | key 30003、整输出商余 |
| Kernel entry/router | `op_kernel/resize_bilinear_v2_apt.cpp` | key 30003 branch | `SimtNCHW<...,uint64_t>` 映射 |
| Device body/helper | `op_kernel/arch35/resize_bilinear_v2_simt_nchw.h` | `calleeInt64nchw/QuickDivForSimtComputenchw` | 512 threads、uint64 四维解码 |

## 算子特定优化

### Trick T-001：大空间域切换为 HW ownership，复用坐标解码

#### 触发与源码依据

- 适用 Route：R-001、R-003。
- 触发条件：NCHW、输入不为单元素、输入输出空间不同，且 `outH*outW >= coreNum*(i32?1024:512)`。
- 固定源码：[`op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp:607-615`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
int64_t hwSizeThreshold = useIdx32 ? coreNum_ * SIMT_DEFAULT_THREAD_NUM :
                                     coreNum_ * SIMT_DEFAULT_THREAD_NUM_IDX64;
bool needHWSplit = xShape_.GetShapeSize() != 1 && (lenSrcW_ != lenDesW_ || lenSrcH_ != lenDesH_) &&
                   lenDesH_ * lenDesW_ >= hwSizeThreshold;
if (useIdx32) {
    tilingKey_ = (needHWSplit) ? TILING_KEY_SIMT_HW : TILING_KEY_SIMT_NCHW;
} else {
    tilingKey_ = (needHWSplit) ? TILING_KEY_SIMT_HW_IDX64 : TILING_KEY_SIMT_NCHW_IDX64;
}
```

#### 机制与设计原因

NCHW body 对每个输出元素解码 N/C/H/W；HW body 只对每个空间点解码一次 H/W，然后在同 thread 内循环全部 N/C。对大的 `OHW`，这减少坐标/magic decode 次数并保持足够的 `Cores*threads` 空间并行度。Direct selector 机械保留该阈值与 1024/512 specialization，但没有保留官方前置特化；all-copy 与 NCHW-broadcast 的官方覆盖域仍与 Direct 合法输入相交，只是在最终载荷中改由通用 SIMT Route 承接。

#### 迁移边界

- 适用：连续 NCHW、所有 N/C 平面共享相同 resize 坐标、空间点数足以覆盖线程域。
- 前提：同一 `(H,W)` thread 写各 N/C 平面的互不重叠地址；索引宽度与 thread 数配套。
- 失效：空间域不足、空间不变或单元素输入时改用对应 NCHW Route。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不可只凭“空间 shape 改变”强制 HW 路线；官方固定阈值同时要求 `OHW` 覆盖 `coreNum*threads`，否则会减少可用并行域。固定仓未提供其它可验证反例。

### Trick T-002：商余分核让尾元素只落在前缀 block

#### 触发与源码依据

- 适用 Route：R-001～R-004。
- 触发条件：Route 已选定，domain 不能被 `realCoreNum` 整除时 `splitBlockTailFactor>0`。
- 固定源码：[`op_kernel/arch35/resize_bilinear_v2_simt_hw.h:133-140`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_simt_hw.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
T_IDX blkProcessNum = tiling_->splitBlockFactor;
T_IDX blkStartOffset = blockIdx_ * tiling_->splitBlockFactor;
if (blockIdx_ < tiling_->splitBlockTailFactor) {
    blkProcessNum += 1;
    blkStartOffset += blockIdx_;
} else {
    blkStartOffset += tiling_->splitBlockTailFactor;
}
```

#### 机制与设计原因

Host 只传商 `q` 和余数 `r`；Device 以 `start=block*q+min(block,r)` 重建连续、无洞、无重叠的区间。它避免额外 per-block offset 表和 workspace，也使每个 block 的负载差最多一个 domain 元素。NCHW body使用同一公式，区别仅是 domain 为 `O` 而非 `OHW`。

#### 迁移边界

- 适用：可线性编号且每个输出独立的 SIMT 域。
- 前提：`0<=r<realCoreNum`、`q>=1`；本 Direct selector 对两条路线均保证。
- 失效：各元素开销高度不规则或输出间存在依赖时，应改用显式 prefix/调度数据。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

只给前 `r` 个 block 增加 count 而不同时在后续 block 的 start 上加 `r`，会造成区间重叠/漏写；源码同时更新 count 和 start，不能拆开迁移。

### Trick T-003：magic division 解码扁平输出坐标

#### 触发与源码依据

- 适用 Route：R-002、R-004；R-001/R-003 也对 H/W 使用两级版本。
- 触发条件：正的 N/C/outH/outW，`T_IDX` 与 Route index 宽度一致。
- 固定源码：[`op_kernel/arch35/resize_bilinear_v2_simt_nchw.h:60-75`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_simt_nchw.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
T_IDX tmpRes = Simt::UintDiv(tmp, mW, shiftW);
W = tmp - tmpRes * lenDesW;
tmp = tmpRes;

tmpRes = Simt::UintDiv(tmp, mH, shiftH);
H = tmp - tmpRes * lenDesH;
tmp = tmpRes;

tmpRes = Simt::UintDiv(tmp, mC, shiftC);
C = tmp - tmpRes * lenC;
tmp = tmpRes;

tmpRes = Simt::UintDiv(tmp, mN, shiftN);
N = tmp - tmpRes * lenN;
```

#### 机制与设计原因

`Process` 每个外层 AIV block 为常量 extent 生成 `m/shift`，线程用 `Simt::UintDiv` 和乘减恢复 quotient/remainder，避免在每个输出元素上直接执行四次普通整数除模。i32 与 i64 分别绑定 1024/512-thread callee，防止把 32-bit magic 参数错误复用到超范围地址域。

#### 迁移边界

- 适用：row-major 扁平域到固定正 extent 的重复解码。
- 前提：先调用 `GetUintDivMagicAndShift`，且 producer/consumer 使用同一无符号宽度。
- 失效：动态 extent 在单 thread 循环内改变，或 extent 为 0 时不可复用当前参数。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

不要在 i64 Route 中继续实例化 `uint32_t` magic/index；selector 的阈值、callee thread 数和 `T_IDX` 是同一 specialization 契约。

### Trick T-004：Direct 派生保留零权重上邻点的精度语义

#### 触发与源码依据

- 适用 Route：R-001～R-004 的 `mode=0` 插值 body。
- 触发条件：源坐标落在整数点或半像素负边界，尤其输入邻点含 NaN/Inf 时差异可观察。
- 固定源码（被转换的官方基线）：[`op_kernel/arch35/resize_bilinear_v2_simt_hw.h:79-86`](../../../../../../../cann-ops-raw/ascendc/ops-cv/bb717d075d129fd1fc7b96b2886824a1d153ed2f/image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_simt_hw.h)

**官方源码（固定 Commit 连续逐字摘录）**：

```cpp
origHeight = static_cast<float>((H + offset) * scaleH) - offset;
origWidth = static_cast<float>((W + offset) * scaleW) - offset;
leftX = max(min(static_cast<T_IDX>(floorf(origWidth)), lenSrcW - 1), static_cast<T_IDX>(0));
deltaX = origWidth - floorf(origWidth);
rightX = max(min(static_cast<T_IDX>(ceilf(origWidth)), lenSrcW - 1), static_cast<T_IDX>(0));
topY = max(min(static_cast<T_IDX>(floorf(origHeight)), lenSrcH - 1), static_cast<T_IDX>(0));
deltaY = origHeight - floorf(origHeight);
bottomY = max(min(static_cast<T_IDX>(ceilf(origHeight)), lenSrcH - 1), static_cast<T_IDX>(0));
```

#### 机制与设计原因

最终 Direct body明确派生为：先 clamp 负坐标，再令 `upper=lower+(lower<last)`，并把 delta clamp 到 `[0,1]`。与官方 `ceil(source)` 相比，整数非末端位置仍读取右/下邻点，即使其权重为 0；因此 IEEE `0*NaN` 的可观察传播与调用语义一致。此修改改变精度边界而不改变一般有限、非整数坐标的双线性加权顺序，并同时应用于 HW/NCHW 与 i32/i64。

#### 迁移边界

- 适用：要求保留相邻采样参与 IEEE 表达式语义的 bilinear 实现。
- 前提：四点都可合法寻址；末端仍令 upper=lower，delta clamp 后不越界。
- 失效：若上层明确规定整数点只读取一个样本或规定忽略零权重 NaN，则需按该合同另建实现。
- 未验证：本次知识提取未执行编译、真机、数值或性能测试。

#### 错误做法

机械保留官方 `upper=ceil(source)` 会在整数非末端把 upper 折叠到 lower；对普通有限值结果相同，但对零权重邻点中的 NaN/Inf 可能改变结果。不能仅用有限随机输入证明两者等价。

## 模板与实现对比

| 对比维度 | P0 · R-001 HW/i32 | P1 · R-002 NCHW/i32 | P2 · R-003 HW/i64 | P3 · R-004 NCHW/i64 | 设计选择依据 |
|---|---|---|---|---|---|
| 命中条件/优先级 | i32、HW 阈值真 | i32、HW 阈值假 | i64、HW 阈值真 | i64、HW 阈值假 | 先 index 宽度，后 domain ownership |
| 模板与 dtype | `SimtHW<T,T,HP,0,u32>` | `SimtNCHW<T,T,HP,0,u32>` | `SimtHW<T,T,HP,0,u64>` | `SimtNCHW<T,T,HP,0,u64>` | 每 Route 均覆盖 3 dtype×2 HP |
| 循环与分核 | OHW 分核；thread→HW→N→C | O 分核；thread→NCHW | 同 R-001，512 stride | 同 R-002，512 stride | 大空间复用坐标，否则完整输出并行 |
| Queue/TBuf/寄存器/dynamic UB/通信 window | 仅线程标量 | 同左 | uint64 标量更多 | 同左 | 四者 UB/workspace/通信均为 0/N/A |
| 内存总占用与复用 | GM input/output；坐标跨 N/C 复用 | GM input/output | 同 R-001 | 同 R-002 | 无中间物化 |
| 搬运/Vector/SIMT/NDDMA/通信 API | scalar GM + `asc_vf_call(1024)` | 同左 + 四维 quick-div | scalar GM + `asc_vf_call(512)` | 同左 + 四维 quick-div | 无 Vector DMA/NDDMA/通信 |
| 同步、精度与 tail | 唯一 HW owner；FP32；商余 | 唯一 linear owner；FP32；商余 | 同 R-001，u64 | 同 R-002，u64 | 无 atomic/barrier，stream ordering |
| 未命中时 selector 备选 | `H` 假则 R-002 | `A` 假后以 512 阈值重算 `H`，选 R-003/R-004 | `H` 假则 R-004 | 无后续 Route | 这些箭头只发生在 launch 前；四条 Route 均无运行时 fallback，Host 合同或平台异常直接失败 |

## 平台差异

| 项目 | A2 | A3 | 950 |
|---|---|---|---|
| 支持状态 | 官方 README 声明 ResizeBilinearV2 支持；本 Direct 转换未生成 A2 载荷 | 官方 README 声明支持；本 Direct 转换未生成 A3 载荷 | 官方 README 与 `SUPPORT_COMPUTE_UNIT=ascend950` 均支持；本卡目标为 950/dav-3510 |
| Host/Launcher | Direct 入口固定仓不可见 | Direct 入口固定仓不可见 | `PlatformAscendCManager` 直查 AIV/UB，显式 launcher/entry |
| Kernel/资源 | 不得从 950 SIMT 参数外推 | 不得从 950 SIMT 参数外推 | arch35 SIMT HW/NCHW，i32=1024 threads、i64=512 threads，无 UB/workspace |

## 限制与失败边界

- Direct 合同只覆盖 rank-4、连续 NCHW、非空、输入输出同 dtype的 FP16/FP32/BF16；不接受 NHWC、非连续 view、空 tensor、dtype 转换或独立 `size` tensor。
- `output_size` 与 `scale_factor` 必须恰给一个。scale-derived 输出先检查 double 乘积有限、`>=1` 且 `<=static_cast<double>(INT64_MAX)`，随后 `floor`/转 int64；由于该上界本身是 double 转换，本文不宣称它严格证明 int64 可表示性。factor 再降为 float 进入 POD，坐标与插值也均为 FP32。
- `CheckedProduct` 保护 Host int64 乘积；i32 Route 使用严格 `<UINT32_MAX`，等于上限会走 i64。`lenN/lenC` 没有单独 int32 guard，但总元素数 guard 覆盖其在合法正 shape 下的 i32 乘法域。
- launcher 返回 `void`，`RunOpApi` lambda 固定返回 0；异步设备错误不由该返回值表达。绕过配置函数传入非法 route/config 没有完整防御性校验。
- `ubSize`、多个 factor 和 `tilingKey` 是为保持官方同序 POD/映射而保留的 inactive 字段；当前 body 不应据此推导 UB 占用或 Device router 行为。
- 官方固定提交提供 950 支持、POD、selector 和 kernel 基线；邻点选择是最终 Direct 载荷的明确派生事实。本文未把该派生误写成官方原始实现。

## 源码索引

### 链路索引

| 链路阶段 | 文件 | 关键符号 | 支撑内容 |
|---|---|---|---|
| 入口声明/约束 | `image/resize_bilinear_v2/README.md` | 产品表、参数表 | 950 支持、三 dtype 与双线性语义 |
| 入口实现 | `image/resize_bilinear_v2/op_api/resize_bilinear_v2.cpp` | `ResizeBilinearV2AICORE/ResizeBilinearV2With4d` | `half_pixel_centers=!align_corners` 与 scales 转发基线 |
| POD | `image/resize_bilinear_v2/op_host/arch35/resize_bilinear_v2_tiling_arch35.h` | `ResizeBilinearV2TilingData` | 字段类型与固定顺序 |
| Host/Tiling | `image/resize_bilinear_v2/op_host/arch35/resize_bilinear_v2_tiling_arch35.cpp` | `SetDimsByFormat/MatchTilingStrategyAndSetTilingKey/DoTilingSIMT*` | scale、index 宽度、HW/NCHW selector、商余分核 |
| Kernel router | `image/resize_bilinear_v2/op_kernel/resize_bilinear_v2_apt.cpp` | `resize_bilinear_v2` | key→HW/NCHW→HP→index 模板映射 |
| Kernel common | `image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_base.h` | `THREAD_NUM/THREAD_NUM_MIDDLE/HALF_PIXEL` | 1024/512 threads 与 0.5 常量 |
| Kernel body | `image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_simt_hw.h` | `ResizeBilinearV2SimtHW::Process/SimtCompute` | HW ownership、N/C loop、SIMT 插值基线 |
| Kernel body | `image/resize_bilinear_v2/op_kernel/arch35/resize_bilinear_v2_simt_nchw.h` | `QuickDivForSimtComputenchw/Process/SimtCompute` | 扁平 NCHW ownership、magic division、SIMT 插值基线 |
| 构建接线 | `image/resize_bilinear_v2/CMakeLists.txt` | `SUPPORT_COMPUTE_UNIT/SUPPORT_TILING_DIR` | ascend950→arch35 构建选择 |

每行对应的文件都存在于 Frontmatter `sources`；Direct wrapper/POD/显式 entry 是最终载荷派生事实，按约束不使用本机路径、交付件伪协议或非官方 URL 充当 `sources`。

### 实现复用检查

| 检查项 | 可复用事实 | 不得改变的约束 | 源码符号 |
|---|---|---|---|
| ABI/POD | 官方 25 字段同序 POD可机械改为按值直调结构 | 字段顺序、int64/float 位宽、float 后对齐、inactive 字段不能误读 | `ResizeBilinearV2TilingData`；Direct 同名 struct |
| 调用顺序 | validate→resolve/allocate→config→typed launcher→one entry→SIMT | dtype、HP、index width、route 四轴必须一致 | `resize_bilinear_npu/calc_resize_bilinear_config/LAUNCH_RESIZE_TYPED` |
| workspace/内存 | 两套 SIMT body只需 input/output GM 与线程标量 | 不得伪造官方 32B workspace 依赖、UB queue 或 dynamic UB | `DoTiling`、两个 `Init/Process` |
| 同步 | 每输出唯一 owner，无 atomic/barrier；依赖同 stream ordering | 不得让 HW block 重叠同一空间点，或破坏 NCHW 扁平区间互斥 | 两个 `Process/SimtCompute` |
| 精度与尾块 | FP32 坐标/权重/插值；商余 block tail；Direct 相邻上邻点派生 | scale 公式、half-pixel clamp、upper 规则、加权顺序、T store 都需保持 | `SetDimsByFormat`、两个 `SimtCompute`、Direct 派生 body |
| 失败边界 | Host 拒绝合同、溢出和平台异常 | 直调 config 非法值没有 Device fallback；lambda 0 不等于同步成功 | `ResolveOutput/CheckedProduct/calc_resize_bilinear_config` |
