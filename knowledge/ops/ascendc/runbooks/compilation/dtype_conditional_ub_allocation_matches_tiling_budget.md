---
type: "Runbook"
title: "按 dtype 分配 UB 必须匹配 tiling 预算"
description: "SPLIT_D 的 fp32 路径直接使用输入张量，却仍分配仅低精度转换需要的 xFp32Buf，使实际 UB 占用超过 tiling 预算；按 dtype 守护该缓冲后恢复大 D 路径。R2（950/9.1.0）：布局分支（useRawCopy）切换与 dtype 分支同理——tiling 的 bufferDivisor 必须按 kernel 实际布局同步，且两侧阈值判定式一致。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-14T03:54:37Z"
updated_at: '2026-10-20T12:33:33Z'
platforms: ["a3", "950"]
---

# 按 dtype 分配 UB 必须匹配 tiling 预算

## 现象

RmsNorm 的 NORMAL 路径和小 D 用例已通过，但较长归约维进入 SPLIT_D 后，设备同步报 ACL `507035`，并明确给出 “VEC instruction read/write UB is out of bounds”。故障发生在 fp32 大 D 路径；这与一般的数值 mismatch 或 GM 索引错误不同。

R2（Ascend950PR，arch35，CANN 9.1.0，SwiGlu 算子 BenchNoHarnessSwiGlu）：kernel 侧为小 innerSize 引入 `useRawCopy_` 布局分支（`inQueueX0` 单队列存 `2*ubFormer_` 个 raw 元素，另加 `outQueue` 与 3 个 FP32 `TBuf`）后，host tiling 的 UB 切分仍按**旧布局**估算每输出元素字节占用（FP32：`bufferDivisor = 7 * sizeof(float)`）。修完小 innerSize 混合路径后重跑 ST Real：FP16 的 `[4,8] dim=-1` 用例全部转 PASS，但 `builtin_fp32_2d_dim-1_[4,8]` 仍然 `流同步失败: 507035`。以下仅按原卡列出的布局复算：raw 路径的 FP32 分配账目为 36B/输出元素，旧除数只计 28B/输出元素；不能由这项静态账目认定该次 Ascend950PR 的实际 UB 容量。

## 适用场景

- 工程形态：classic Ascend C 直调 kernel 的编译、链接或运行工程；R2 为 AscendC 自定义算子 host tiling + kernel 协同开发（UB 切分/InitBuffer 布局账目）。
- API 路径：`KernelRmsNormSplitD`、`xFp32Buf_`、`ACL 507035`、`ubFactor` 所在的 kernel、host 或构建接口链；R2 增补 `TilingFunc` 内 `ubFormer` 计算（`ubSize / bufferDivisor` 对齐）、kernel `TPipe::InitBuffer`。
- 验证平台与版本：平台 a3；CANN：unknown。R2 在 Ascend950PR（arch35）+ CANN 9.1.0 实测。
- 适用粒度：`same_operator`，迁移前须重算同算子各 dtype/template 的 live-buffer 预算；R2 表明该规则同样适用于**布局分支**（不限 dtype 分支），粒度推广到 `operator_family`——所有 kernel 内存在多套 UB buffer 布局（按 tiling/shape/布局分支切换）的算子。
- 路线优先级：`recommended_fastpath`。
- 不适用/停止边界：错误信号、API 契约或数据路径与本卡不一致时停止套用；修复后未闭合原信号则回到根因诊断；单布局 kernel（无 useRawCopy 类分支）不涉及 R2 形态。

## 根因

SPLIT_D 的 fp32 计算直接对 `xLocal` 做平方，不使用 `xFp32Buf_`；但 `Init` 仍无条件为它分配 `ubFactor * sizeof(float)`。令 `B_tiling(dtype, tile)` 为 host 预算、`B_live(dtype, tile)` 为 kernel 所有 live queue/TBuf 与固定 scratch 之和，则该分支出现 `B_live = B_tiling + tile * sizeof(float)`，并在 `B_live > UB_available` 时触发 VEC UB 越界。

根因只有一个：kernel 的 dtype 条件分配与 tiling 的逐元素预算不一致。修复不是减小测试规模或改异常处理，而是让 fp32 的实际 live-buffer 集合与 host 预算使用同一公式。

R2 账目失配（触发维度从 dtype 分支换成布局分支）：按原卡所列 `useRawCopy_` 路径，FP32 分配从旧布局的 28B/输出元素变成 36B/输出元素，而 host tiling 的 `bufferDivisor` 仍按旧布局计账。是否超过实际可用 UB，应另用来源已绑定的 `B_available` 比较：

- 旧布局（分块 x0/x1）：3 个 `TQue`（×2 双缓冲）+ 1 个 `TBuf`，全 FP32 → **7 × 4B = 28B/输出元素**。
- raw 布局：`inQueueX0`（×2 双缓冲、每份 `2*ubFormer_` raw 元素，即 4×ubFormer_ 元素 → 16B/元素）+ `outQueue`（×2 → 8B/元素）+ 3 个 `TBuf`（12B/元素）→ **9 × 4B = 36B/输出元素**。

仅按原卡给出的 `T=ubFormer_=8192` 做静态账目演算：raw 布局列出的分配为 `inQueueX0 = 2×16384×4 = 131072B` + `outQueue = 65536B` + 3×`32768B` = **294912B**，即 `B_raw=36*T`。原卡在此采用 `196608B（192KB）` 比较，但没有给出与该次950PR运行绑定的容量查询或显式预留依据；本轮不将它认作该平台实测容量或已证实的受限预算，也不以 R3 的容量替换这段历史记录。是否超额须比较该次来源明确的 `B_raw` 与 `B_available`。FP16/BF16 在本卡所列两套布局中均为 24B/输出元素，FP32 则需从 28B 改按 36B 计账；每套布局应单独核算。

## 解决配方

### before

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                int64_t numRow, int64_t numCol, int64_t numColAlign,
                                int64_t blockLength, uint32_t ubFactor, float epsilon)
    {
        numCol_ = numCol;
        numColAlign_ = numColAlign;
        ubFactor_ = static_cast<int64_t>(ubFactor);
        epsilon_ = epsilon;
        blockElems_ = 32 / sizeof(T);

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        gammaGm_.SetGlobalBuffer((__gm__ T *)gamma);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        int64_t blk = AscendC::GetBlockIdx();
        coreStart_ = blockLength * blk;
        coreEnd_ = coreStart_ + blockLength;
        if (coreEnd_ > numRow) coreEnd_ = numRow;
        if (coreStart_ > numRow) coreStart_ = numRow;

        pipe_.InitBuffer(xQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(gammaQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(xFp32Buf_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(sqxBuf_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, 256);
        pipe_.InitBuffer(tmpBuf_, 256);
        pipe_.InitBuffer(scalarBuf_, 256);
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(gammaFp32Buf_, ubFactor_ * sizeof(float));
        }
    }
```

### after

```cpp
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                int64_t numRow, int64_t numCol, int64_t numColAlign,
                                int64_t blockLength, uint32_t ubFactor, float epsilon)
    {
        numCol_ = numCol;
        numColAlign_ = numColAlign;
        ubFactor_ = static_cast<int64_t>(ubFactor);
        epsilon_ = epsilon;
        blockElems_ = 32 / sizeof(T);

        xGm_.SetGlobalBuffer((__gm__ T *)x);
        gammaGm_.SetGlobalBuffer((__gm__ T *)gamma);
        yGm_.SetGlobalBuffer((__gm__ T *)y);

        int64_t blk = AscendC::GetBlockIdx();
        coreStart_ = blockLength * blk;
        coreEnd_ = coreStart_ + blockLength;
        if (coreEnd_ > numRow) coreEnd_ = numRow;
        if (coreStart_ > numRow) coreStart_ = numRow;

        pipe_.InitBuffer(xQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(gammaQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, DEPTH, ubFactor_ * sizeof(T));
        pipe_.InitBuffer(sqxBuf_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, 256);
        pipe_.InitBuffer(tmpBuf_, 256);
        pipe_.InitBuffer(scalarBuf_, 256);
        if constexpr (!std::is_same<T, float>::value) {
            pipe_.InitBuffer(xFp32Buf_, ubFactor_ * sizeof(float));
            pipe_.InitBuffer(gammaFp32Buf_, ubFactor_ * sizeof(float));
        }
    }
```

### R2 解决配方（布局分支切换的除数同步）

host tiling 用与 kernel **完全相同的阈值**判定 `useRawCopy`，并按布局切换 `bufferDivisor`（verbatim 取自该会话 tiling 写入事件）：

```cpp
    // 5. UB 切分
    // 判断是否使用整块原始数据拷贝策略 (kernel 中 innerSize < 32/sizeof(T) 时启用)
    // useRawCopy 时 UB 布局变化: inQueueX0 存 2*tileSize 元素 (raw data)
    constexpr uint32_t MIN_BURST_BYTES = 32;
    bool useRawCopy = (innerSize * dtypeSize < MIN_BURST_BYTES) && (innerSize > 0);

    // FP32 路径:
    //   非 rawCopy: 3 TQue(×2) + 1 TBuf = 7 buffers, all FP32 → 28 bytes/elem
    //   rawCopy:    1 TQue(×2, 2x size) + 1 TQue(×2) + 3 TBuf = (4+2+3) = 9 → 36 bytes/elem
    // FP16/BF16 路径:
    //   非 rawCopy: 3 TQue(×2, orig) + 3 TBuf(FP32) → 6*dtypeSize + 3*4 = 24 bytes/elem
    //   rawCopy:    1 TQue(×2, 2x orig) + 1 TQue(×2, orig) + 3 TBuf(FP32) → (4+2)*dtypeSize + 3*4 = 24 bytes/elem
    uint64_t bufferDivisor;
    if (dataType == ge::DT_FLOAT) {
        bufferDivisor = useRawCopy ? (9 * sizeof(float)) : (7 * sizeof(float));
    } else {
        bufferDivisor = 6 * dtypeSize + 3 * sizeof(float);  // 24 bytes/elem (FP16/BF16, rawCopy same)
    }
```

要点：`useRawCopy` 判定式（`innerSize * dtypeSize < 32 && innerSize > 0`）必须与 kernel 侧 `useRawCopy_ = (innerSize_ < MIN_BURST_BYTES / sizeof(T)) && (innerSize_ > 0)` 一致，两处阈值漂移会造成 tiling 账目与 kernel 实际布局错位。为什么不削减 kernel buffer 而是改除数：raw 布局是上一卡（小 innerSize 混合路径）的正确性修复，不动；UB 切分本来就该反映真实布局，`ubFormer_` 变小只影响每 tile 元素数，不影响正确性。为什么 FP16/BF16 除数不变：raw 布局下 `(4+2)*2+3*4 = 24B/元素` 与旧布局 `6*2+12 = 24B/元素` 相等，属巧合而非规律——每套布局都应单独记账，不能默认相等。

## 设计决策

优先移除 fp32 不使用的缓冲，而不是把无用项补进 `perElem`。前者使 kernel 的实际资源需求与已有计算路径一致，也保留更大的 `ubFactor`；后者会为从不读取的存储付费并缩小切片。

## 性能影响

修复减少 fp32 SPLIT_D 的 UB 占用，理论上允许保持既有切片宽度；轨迹没有提供同平台、同负载的配对性能数据，因此只证明功能与资源闭环，不宣称加速。R2 同理：`ubFormer_` 变小只影响每 tile 元素数，无配对性能数据。

## 预防

- 为每个模板、每个 dtype 列出所有存活的 `TQue/TBuf`，逐项核对 `depth * bytes`；
- host 的 `perElem` 公式必须与 kernel 的 `if constexpr` 分配条件同源核算；
- 在模板切换点两侧使用参数化归约维 probe，至少覆盖 fp32 与低精度分支；
- 资源估算除逐元素缓冲外，还要预留 reduce、tmp、scalar 等固定开销。
- R2：kernel 内任何改变 UB buffer 数量/大小的分支（useRawCopy、双缓冲切换、额外 workspace buffer）都必须在同一变更里同步 host tiling 的 `bufferDivisor`/UB 切分账目，并在注释里写清"每输出元素字节占用 = Σ(buffer 数 × dtype 字节)"的推导；每套布局单独记账，不能因某 dtype 两套布局恰好相等就默认相等。
- 复查 UB 资源时用乘法显式列账（如本卡 raw FP32 的 `B_raw=36*T`），再与来源明确的 `B_available` 比较，不要只看"元素数没变"。

## 验证信号

以下实际构建、运行和评测信号沿用原开发会话记录；单独标明的 R2 资源账目是静态推导。本轮未重放这些历史算子验证。

- 修复前：大 SPLIT_D 路径触发 UB 越界后的 ACL 507035 vector core exception。（来源平台：a3；证据等级: tool_observed）。
- 修复后：大 SPLIT_D 定向矩阵全部输出 OK。（来源平台：a3；证据等级: tool_observed）。
- 修复后：干净构建 exit=0，随后 32 passed。（来源平台：a3；证据等级: tool_observed）。

R2 轮（Ascend950PR / CANN 9.1.0，SwiGlu 迭代一联调）增量信号：

- R2 修复前（raw 布局 + 旧除数 28B/元素）：`builtin_fp32_2d_dim-1_[4,8]` `流同步失败: 507035`（来源平台：950；证据等级：tool_observed）。
- R2 静态资源账目：按原卡所列 `ubFormer_=8192` 和 buffer 集合，`131072+65536+32768×3 = 294912B`。此处仅复算分配需求，不认证该次 `ubFormer_` 的实测绑定或950PR可用容量，不作为本轮设备 `tool_observed` 信号。
- R2 除数同步后：`builtin_fp32_2d_dim-1_[4,8]` PASS（`MERE=1.84e-09, MARE=1.32e-08`），12 pass 1 fail（仅剩 bf16，另卡根因）（来源平台：950；证据等级：tool_observed）。
- R2 最终回归：ST Real 13/13 PASS、UT 24/24（来源平台：950；证据等级：tool_observed）。

## 失效边界

本卡只覆盖 `KernelRmsNormSplitD` 的这组缓冲及 tiling 预算。若 `DEPTH`、队列集合、dtype 转换路线、UB 容量或固定 scratch 发生变化，必须重新计算 `B_live(dtype, tile) <= UB_available`，不得复用源任务的逐元素常数。其他平台未在本闭环验证。

R2 补充：本轮只复算原卡列出的 buffer 账目，尚未取得与该次运行绑定的 UB 查询、预留规则和实际 tiling 回执；`ubFormer_=8192` 仅作上述原卡示例，不据此重建该环境的容量或分核。采用时须按来源明确的 `B_available`、实际 tile 与对齐规则重算；不能复用未绑定的容量常量，也不能以 R3 的独立场景代替 R2 环境。本卡保留"布局账目失配"这一机制；`useRawCopy` 阈值、32B 约束、bf16 转换仍分别见原独立卡。

## 坏实践

- 看到大 D 越界就只把 `ubFactor` 乘固定折扣；这会掩盖 host/kernel 资源模型不一致；
- 为所有 dtype 无条件分配转换缓冲，期待编译器消除；`InitBuffer` 已经占用 UB，不因后续未读而自动安全；
- 只跑 NORMAL 或小 D 用例后宣布全部模板通过。
- R2：kernel 改了 buffer 布局、tiling 除数沿用旧值——FP32 分配账与旧预算脱节，原记录仍报 507035，容易误判为"上一修复没修干净"；是否超过该次实际可用 UB，还须绑定 B_available 与实际 tiling。
- R2：用 dtype 推断布局而不传递统一的 `useRawCopy` 判定——两处阈值一旦漂移即错位。

## 误诊

如果缩小 `ubFactor` 后故障位置不变，或错误日志指向 GM 地址、尾块 DMA、队列同步，应检查地址和生命周期，不要把所有 ACL 507035 都归为容量溢出。只有"模板切换相关 + 实际分配可算出超过 UB + 移除无用缓冲后闭环"才支持本根因。

R2：原记录首轮按"混合路径对 FP32 无效"排查，并检查 DataCopyPad 参数和 CopyOut 大小；随后定位到布局切换与 tiling 预算失配。按本轮可见材料，36B/输出元素的静态账目可复算，但是否超过该次实际容量仍需 B_available 与 tiling 绑定，不能仅由 507035 认定。

## R3 增量（反向脱节：预算虚高致 ubFactor 缩水，按 kernel 实际 InitBuffer 反推）

R3 并入 Gather 算子（bench_no_harness_gather，Ascend950PR/arch35/CANN 9.1.0，UB=253952B）迭代三全功能审查的实例，补齐本卡契约的**反向**形态——主卡与 R2 是"kernel 实际分配超过 tiling 预算"（超额→507035），R3 是"tiling 预算为 kernel 从未分配的缓冲预留"（虚高→ubFactor 缩水、UB 利用率低下）：

**1. 现象**：kernel Init 阶段只在 UB 分配 index 双缓冲（各 `CeilAlign(ubFactor × indexDtypeSize, 32)` 字节）加一个 32 字节 tempBuf——x 数据逐元素从 GM 直取、从不整体缓存进 UB。但 tiling 的 UB 预算公式按"要把整个 x 缓存进 UB"计算：`xBufSize = CeilAlign(xNumel × xDtypeSize, ...)`（x.numel=2M 的 fp32 case 即 8MB，远超 253952B 的 UB）→ 恒走"回退 ubSize/2"分支 → 再被 bufferNum=4 和 `xDtypeSize + indexDtypeSize` 双重稀释，ubFactor 只有 3968，kernel 实际 UB 占用 31776B/253952B ≈ 12.5%——批量搬运粒度被无谓压小到约 1/7。

**2. 修复（预算 = Σ(kernel 实际 InitBuffer) + 显式余量）**：

```cpp
    // 6、UB 切分（基于 kernel 实际 UB 使用量计算 ubFactor）
    // Kernel 实际 UB 分配（参见 <op>.h::Init）：
    //   - indexBufA: CeilAlign(ubFactor × indexDtypeSize, 32) 字节
    //   - indexBufB (双缓冲): CeilAlign(ubFactor × indexDtypeSize, 32) 字节
    //   - tempBuf: 32 字节（单元素中转缓冲区，MTE2→MTE3 搬运中转）
    // 总 UB = BUFFER_NUM × CeilAlign(ubFactor × idxSize, 32) + 32 ≤ ubSize
    int64_t xNumel = xShape.GetShapeSize();
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? 2 : 1;  // index 缓冲区数量（单/双缓冲）
    // 安全余量：系统/框架 UB 开销 + tempBuf + 每缓冲区 32B 对齐填充
    constexpr int64_t UB_SAFETY_MARGIN = 16384;   // 16KB 系统/框架预留
    constexpr int64_t TEMP_BUF_SIZE = 32;          // 单元素中转缓冲区
    constexpr int64_t ALIGN_OVERHEAD = 32;         // 每缓冲区 32B 对齐预留
    int64_t availableUb = static_cast<int64_t>(ubSize) - UB_SAFETY_MARGIN - TEMP_BUF_SIZE
                          - bufferNum * ALIGN_OVERHEAD;
    int64_t ubFactor = FloorAlign(
        FloorDiv(FloorDiv(availableUb, bufferNum), indexDtypeSize),
        ubBlockSize);
    if (ubFactor < 1) {
        ubFactor = 1;  // 保证至少处理 1 个元素/批
    }
```

配套 UT 断言与 tiling 公式同源（公式漂移在 UT 暴露）：

```cpp
    // 验证 ubFactor 合理性：基于 kernel 实际 UB 分配
    int64_t indexDtypeSize = GetDtypeSize(param.indexDtype);
    int64_t bufferNum = param.expectDoubleBuffer ? 2 : 1;
    int64_t indexUbSize = (tiling->ubFactor * indexDtypeSize + 31) & ~31;  // 32B 对齐
    int64_t ubEstimate = bufferNum * indexUbSize + 32;  // index 缓冲区 + tempBuf
    EXPECT_LE(ubEstimate, static_cast<int64_t>(PLATFORM_UB_SIZE))
        << "ubFactor UB budget exceeded for case: " << param.caseName
        << " (ubEstimate=" << ubEstimate << ", ubSize=" << PLATFORM_UB_SIZE << ")";
```

要点：① 预算公式以 kernel `InitBuffer` 的实际分配为唯一事实源，并把该分配公式以注释契约写进 tiling；② `bufferNum` 语义修正为"index 缓冲区数量（单/双缓冲）= 1/2"——旧公式按"2 queues × BUFFER_NUM"算成 2/4，把队列深度与缓冲个数混为一谈；③ 安全余量取固定 16KB（覆盖系统/框架预留，经验值非实测下界）。

**3. 误诊教训（R3）**：首版修复只留 96 字节余量（UB 用满 99.96%）联调出现 507035 流同步错误时，第一反应是"UB 用太满"，一度想直接加大余量或封顶 ubFactor 了事；回退对照 + 头文件查证后确认真因是 `DataCopyParams.blockLen` uint16 截断（搬运参数截断，另卡）。教训：tiling 优化后出现的运行时错误，先隔离"新参数是否触发 kernel 既有缺陷"，再怀疑预算本身；余量大小与该错误无因果关系，已通过对照实验区分。

**R3 验证信号**（同一次 Gather 迭代三开发会话，修复=改 tiling 公式 + 同步 UT 断言 + 重编译重装 + 重跑 UT/ST）：

- 修复前结构性信号：旧公式下双缓冲 case ubFactor=3968，kernel UB 占用 31776B/253952B ≈ 12.5%（来源平台：950；证据等级：tool_observed）。
- 修复后 UT：38/38 通过（tiling 31 + infershape 7），后扩展到 62/62；单缓冲 int32 case 日志观测 ubFactor=59360、双缓冲 int64 ubFactor=14816（预算解释按上方代码使用公式 `FloorAlign(FloorDiv(FloorDiv(availableUb(bufferNum), bufferNum), indexDtypeSize), ubBlockSize)`，不将双缓冲预算常量套用于单缓冲）（来源平台：950；证据等级：tool_observed）。
- 修复后 ST：54/54 通过，含三个 ~1M 输出元素大 shape 用例全部 Bitwise Match；Golden 自测 16/16（来源平台：950；证据等级：tool_observed）。
- 数值核验：blockLen 最大 237440 < DataCopyExtParams 上限（来源平台：950；证据等级：tool_observed）；预算按上方代码静态演算为 `availableUb(bufferNum) = 253952 - 16384 - 32 - bufferNum * 32`，单缓冲 `availableUb(1)=237504B`、双缓冲 `availableUb(2)=237472B`，均大于 0；此处仅澄清预算分支，不新增设备验证。

**R3 边界与坏实践**：本配方假设 kernel 不缓存 x；若 kernel 改为整体缓存 x 或新增其他 UB 缓冲，预算公式必须同步重写，否则反向溢出（回到主卡/R2 形态）。`UB_SAFETY_MARGIN=16384` 是经验值，最小安全值未实测；ubSize=253952B 为 Ascend950PR/arch35 的值，其他平台取本平台 `GetCoreMemSize(UB)` 后机制相同（未验证，推断可迁移）。ubFactor 提升约 7.5 倍的结构性批量粒度收益未做 profiling，不能据此宣称具体加速比。坏实践补充：tiling 与 kernel 各写各的 UB 假设互不对照（本例漂移存在两个迭代才被审查发现）；预算公式里混入语义不明的倍数（"2 queues × BUFFER_NUM"）而不回到 kernel 的 `InitBuffer` 分配式反推；用"回退到 ubSize/2"这类粗粒度兜底长期运行（兜底掩盖预算公式本身的错误）。

## 静态源码增量：生命周期复用后仍逐项核算分配

本节提供两个独立固定源码的**静态账目和语法数据流**，不是 R4 设备复现。原卡 a3／950 的平台、运行结果与身份保持不变；本节不新增平台验证，不把包名当实际运行硬件。未重新编译或执行以下原源码，也未核验生成的依赖指令。

### ApplyAdamW：两个临时区与条件转换区

固定源码为 具有下列 SHA-256 的原内包 内的 `csrc/ops/apply_adam_w/op_kernel/apply_adam_w_kernel.cpp`，源码 SHA256 `12a6a42d920c4c667b1ee17a6dc5dea8d6483373c4d1bba6699eb2e6f80d326f`，内 ZIP SHA256 `ccb03290872b315ddb0ee8597916915cfab7dc8eb06cc1daf50271b9cd9e8bc4`。以下行号均指该完整文件。

L85–102 的所有显式 `InitBuffer` 可按 tile 元素数 `T`、输入字节数 `d` 列账：

| 分配项 | 字节 | 条件 |
|---|---:|---|
| 四个输入队列、一个输出队列，每个 depth=2 | 10×T×d | 所有 dtype |
| bufTmp1、bufTmp2 | 2×T×4 | 所有 dtype |
| var、grad、m、v、result 的 FP32 转换区 | 5×T×4 | 仅非 float |

因此本源 float 与 2B 输入都为 **48×T**，不是旧注释仍残留的 52／72 倍数。L50–53 的实际 device 上限为 3840；L318、L341–343 的 host 上限也为 3840，并生成 256 元素倍数。`T=3840` 时显式分配合计 **184320B**，所有独立分配均为 32B 倍数。该 Init 只夹取范围，不自动为任意外部 tile 对齐：例如传入 17 会保留 17。因此上述对齐结论绑定原 host 路由，不能只依据 device clamp。

| 复用存储 | 旧值最后语法读者 | 新值写入 | 管线与证据边界 |
|---|---|---|---|
| tmp1 | Sqrt(tmp2,tmp1)，L181／221 | Muls(tmp1,mL/mF,...)，L188／227 | 都是 Vector；源码存在读后复用，但没有在本轮验证实际 WAR 完成关系 |
| tmp2 | Div(...,tmp2)，L191／229 | weight_decay 非零时 Muls(tmp2,varL/varF,...)，L195／233 | 都是 Vector；不能由 C++ 调用先后直接证明设备完成 |

该数据流可解释为何只分配两个临时区；它不授权去掉同步、改变算术／舍入顺序，或声称复用本身提高性能。源码同管线顺序仍应结合实际编译器依赖生成与设备证据另查；不要把“最后一次发射读取”误写成“最后读者已完成”。

### DynamicQuant：复用之外还有每条路径的固定与变长项

固定源码为 具有下列 SHA-256 的原内包 内的 `csrc/ops/dynamic_quant/op_kernel/dynamic_quant_kernel.cpp`，源码 SHA256 `3afa30971be24a7e3a03b2e99816597eb25c2fe467fef03f610b5a0d1426727e`，内 ZIP SHA256 `2166e8e9a43dd2965b775c5653b564ee324786796d2f906783f770e4382723d0`。

L247–254 的八项实际分配统一写为：`B = 2*I*d + 2*O + 2*32*4 + 4*F + 4*A + W + 4*D + 2*H`。依次对应 input queue、output queue、scale queue、FP32 work、abs work、reduce work、reduce destination、half work；`d=2`。对齐后输入／输出行跨度分别为 `S=Align16(lastDim)`、`Q=Align32(lastDim)`（L103–104）。

| 路径 | I、O、F、A、W、D、H 的取值 |
|---|---|
| small | I=O=F=H=rows*Q；tiny bf16 abs 分支 A=F，否则 A=8；D=rows*8；W 为传入 reduceBufSize |
| medium | I=F=Q；O=H=8192；A=8；W=8192；D=ceil(lastDim/8192)*8 |
| large | I=O=F=H=8192；A=8；W=max(原 W,8192*4+256)；D=ceil(lastDim/8192)*8 |

不能只按“12B／16B 每元素”估计 small：scale、reduce work、destination 和最小 abs 区也要加入。L136–177 的 rows 估算先夹至 1..32，L184–195 再按八项总量比较源内预算 `184*1024`；超额转到后续分支。L108–130 的初始路由还受 dtype、对齐行宽与 token 数影响；half 的 S>=4096、bf16 的 S>=8192 或 (S>=4096 且 tokens<1024)，以及 S>11200，都会先进入后续路径。这里列阈值仅为复算原 Init 的分配集合，不将其性能调度经验作为贡献内容。

后续 L204–240 按 medium 总账决定是否驻留；否则使用 large。**large 的 D 仍随行宽增长，源码没有最后一次总账拒绝。** 合成极大行宽 10^9 时模型得到 4037888B，超过源预算；这说明不能从路径名推导无界安全，不证明该非评测形状在原设备上实际运行过或必现某错误。迁移时必须给出输入范围并重算最后路径。

复用例限于 small 的 tiny-bf16 abs 存储：Stage A 的 ReduceMax 为旧 abs 值的语法读者，后续 Stage C BroadCast 将同区改写为广播临时量（L625，后接 L626 的 PIPE_V barrier）。其间有标量结果读取，但本轮不把源码注释视作完成关系证明，也不认为一个位于覆写之后的 barrier 自然证明覆写前 WAR 已满足。其他 dtype／分支不能直接套用；数值异常处理与量化转换策略不在本节采用范围。

### 本节已做检查及不可继承的结论

随审阅材料给出三个完整来源身份（本节两个，加配套驻留 Guide 的 GroupNorm）、35 个窄分配语句的原行绑定、许可声明及原创 CPU 模型。CPU 实际执行 16 个合成输入、30 个对齐 tile 交叉核算和 2 个抽象生命周期完成关系，保留超额及未对齐的负例；它们只验证账目，不是设备精度、同步或性能测试。

已公开 `split_optimization/source_receipts.json#/sources/1` 与 `/sources/2` 保存这两个源码内包的整提交历史派生摘要；不是原报告或原始单条记录，也没有单机制消融。本节不将历史结果归因于两缓冲复用，不扩展原卡的平台，不新增运行复现次数。完整分配、最后读者及实际完成关系仍须分别核对。

## 公开证据定位

[已公开审阅附件](https://gitcode.com/GitCodeVoyager/cannbot-knowledge/releases/download/pr136-evidence-20261020/cannbot_non_level4_split_review_20261019.zip)的ZIP SHA-256为`15662e91640ee0851b9056a3f2589e17a4baa539374b93ae612bcf810f55b556`。本卡对应：`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/source_receipts.json#/sources/1`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/source_receipts.json#/sources/2`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/source_structure.json#/allocation_statements/ApplyAdamW`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/source_structure.json#/allocation_statements/DynamicQuant`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/source_structure.json#/lifetime_observations`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/applyadamw_allocation_excerpt.txt`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/dynamicquant_allocation_excerpt.txt`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/verify_budget_models.py`；`cannbot_non_level4_split_review_20261019/evidence_public/split_optimization/expected_output.json`。

这些是外部ZIP内路径，不是本知识仓目录；技术依据为上述公开附件，不以本 PR 自述作为技术证明。各成员原件/公开投影的SHA及转换范围按包内manifest、映射或README核对。 此入口只支持本卡“静态源码增量”节，不作为主卡或R2/R3历史运行记录的新证据。
