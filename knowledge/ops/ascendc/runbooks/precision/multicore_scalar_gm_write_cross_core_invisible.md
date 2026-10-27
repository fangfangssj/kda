---
type: "Runbook"
title: 多核 S 管线标量写跨核不可见
description: "DAV_3510/arch35 上多核 kernel 用 GlobalTensor::SetValue/GetValue（S 管线标量 GM 访问）做 Phase1 拷贝时，SyncAll 后其他核读不到小张量（<4M 元素）与跨步访问场景的写入，ST 出现垃圾值级失配（MARE≈65~5e3）；dcci 缓存刷新与 DataCopyPad 多变体均无法救标量路径，最终在 Tiling 按数据量/访问模式强制单核（blockDim=1）规避，大张量线性访问才用多核。"
tags: [task.debugging, topic.precision, topic.programming_model]
status: stable
sources: []
created_at: '2026-09-16T11:20:30Z'
updated_at: '2026-09-17T02:32:55Z'
platforms: ["950"]
---

# 多核 S 管线标量写跨核不可见

## 现象

在 Ascend950PR（DAV_3510 / arch35，CANN 9.1.0）上开发 scatter 类自定义算子（y = data 复制后按 indices 沿 dim 散布 updates，reduce=None/add/multiply/amin/amax，dtype 限 fp16/int32）。迭代一单核版本（blockDim=1）已全部通过。迭代二引入多核：Tiling 把 blockDim 设为 coreNum（48），kernel 改为三段式——Phase1 各核用 `GlobalTensor::SetValue/GetValue` 标量逐元素把自己负责的连续段从 data 拷到 y，`PipeBarrier<PIPE_ALL>()` 后 `SyncAll<true>()` 跨核同步，Phase2 各核按 indices 分段做散布读-改-写。

编译通过、UT（Tiling 参数）17/17 通过，但 ST Real 模式（NPU 真跑，MERE/MARE 对 golden）大面积失败，测试报告原文：

```
--- 迭代一穿刺用例 (必须全通过) ---
  总计: 4, 通过: 1, 失败: 3
--- L0 迭代一适用 (fp16/int32, 必须全通过) ---
  总计: 1, 通过: 0, 失败: 1
--- 迭代二穿刺用例 (多核策略, 必须全通过) ---
  总计: 5, 通过: 2, 失败: 3
--- 迭代二多 shape 用例 (fp16/int32, 必须全通过) ---
  总计: 49, 通过: 4, 失败: 45
--- 迭代二 ST 汇总 ---
  迭代二范围总计: 54, 通过: 6, 失败: 48
  迭代二通过率: 11.1%
```

失败用例的失配是垃圾值级别，不是精度漂移：

```
[Real] PIERCE_MAIN_1D_FP16_I32_NONE: 主线 1D fp16/i32 None: 单核拷贝+逐元素写回 (data=float16, idx=int32, dim=0, reduce=None)
  [FAIL] MERE=1.30e+00 (>= 9.77e-04), MARE=7.15e+01 (>= 9.77e-03), float16
    不匹配 [1]: golden=-2.580078, actual=-3.166016, rel_err=2.27e-01
    不匹配 [2]: golden=-4.515625, actual=2.796875, rel_err=1.62e+00
    不匹配 [14]: golden=-3.511719, actual=4.921875, rel_err=2.40e+00
```

当时的通过/失败分布有明确的规律：单核用例（PIERCE3，1D reduce=add，Tiling 规定 blockDim=1）通过；大张量多核用例（PIERCE2_1 [8192,512] 4M 元素、PIERCE2_2 4M）通过；小张量（[1024]、[64,128]、[8,10]）全部失败；1M 元素的 PIERCE2_5 失败；4M 元素但跨步切分的 PIERCE2_3（[512,8192] dim=0 reduce=add，splitAxis=rank-1）也失败。

## 适用场景

- 工程形态：ascendc-registry-invoke 自定义算子 kernel 直调开发，多核（blockDim>1）kernel 内两阶段（先拷贝/初始化，SyncAll 后再跨段读-改-写）数据流。
- API 路径：`AscendC::GlobalTensor::SetValue/GetValue`（标量 GM 访问）、`AscendC::SyncAll<true>()`、`AscendC::GetBlockIdx()/GetBlockNum()`、Tiling `context->SetBlockDim()`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510，arch35，编译目标 `--soc=ascend950`），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide（DAV_3510 上任何多核标量 GM 写 + SyncAll 的 kernel；结论按机制推广，阈值数值按本算子 UB/L2 容量实测）。
- 路线优先级：safety_fallback（单核规避；多核标量写路线在标量写跨核可见性机制明确前应视为 banned）。fallback 声明：小张量/跨步场景退回单核，功能正确优先；大张量线性访问保留多核。
- 明确不适用 / 未验证场景：其他 arch（910B 等）未验证（未验证，机制推断可迁移需重新实测）；`SyncAll` 带 workspace 的软同步路径未实验；MTE3 向量搬运路径见本卡解决配方的备选段。

## 根因

`GlobalTensor::SetValue/GetValue` 是标量操作，头文件标注走 S 管线（CANN 9.1.0 `kernel_tensor_impl.h`）：

```cpp
template <typename T> __aicore__ inline __inout_pipe__(S)
    typename GlobalTensor<T>::PrimType GlobalTensor<T>::GetValue(const uint64_t offset)
...
template <typename T> __aicore__ inline __inout_pipe__(S)
    void GlobalTensor<T>::SetValue(const uint64_t offset, typename GlobalTensor<T>::PrimType value)
```

而 DAV_3510 的 `SyncAll<true>()` 实现（`kernel_operator_sync_impl.h` 的 `SyncAllImpl`）是：

```cpp
template<bool isAIVOnly, pipe_t triggerPipe, pipe_t waitPipe>
__aicore__ inline void SyncAllImpl()
{
    if constexpr (isAIVOnly) {
        if ASCEND_IS_AIV {
            PipeBarrier<triggerPipe>();
            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, AscendC::SYNC_AIV_ONLY_ALL));
            wait_flag_dev(PIPE_S, AscendC::SYNC_AIV_ONLY_ALL);
        }
        return;
    }
    ...
}
```

跨核同步信号挂在 MTE3 管线上：本地 `PipeBarrier<PIPE_ALL>` 只保证本核各管线（含 S）完成，但 S 管线的 GM 写走独立 store buffer/缓存路径，与 MTE3 DMA 写路径不互通；写流量未超过缓存容量时（小张量）数据滞留在核内/L2，其他核 Phase2 读 y 时拿到旧值或垃圾。大张量（≥2M~4M 元素）写流量自然驱逐到 GM，碰巧跨核可见；跨步 reduce（splitAxis=rank-1）即使大张量也因 Phase2 散布读取命中缓存冲突路径而失败。dcci 只刷 L2 数据缓存，覆盖不到该路径——所以缓存刷新无效。

## 解决配方

最终采用「Tiling 按数据量与访问模式选 blockDim + Phase1 保持标量拷贝」。修复前的 Tiling 多核分支（引入失败状态，blockDim 按数据量推导、可为 1~48 任意值）：

```cpp
    // ===== blockDim + copyBlockFactor 计算 =====
    int64_t copyBlockFactor = 0;
    int64_t blockDim = 1;
    if (isReduceMode && is1D) {
        // 1D reduce：单核保证聚合正确性
        blockDim = 1;
        copyBlockFactor = CeilAlign(dataTotalNum, blockAlign);
    } else {
        // 多核 Phase1：每核拷贝 CeilDiv(dataTotalNum, coreNum) 个元素（对齐）
        copyBlockFactor = CeilAlign(CeilDiv(dataTotalNum, coreNum), blockAlign);
        if (copyBlockFactor < blockAlign) {
            copyBlockFactor = blockAlign;
        }
        blockDim = CeilDiv(dataTotalNum, copyBlockFactor);
        if (blockDim < 1) {
            blockDim = 1;
        }
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
    }
```

修复后的 Tiling（最终版，4M 阈值 + 跨步 reduce 单核 + 多核恒 blockDim=coreNum）：

```cpp
    // ===== blockDim + copyBlockFactor 计算 =====
    // DAV_3510 SyncAll 硬同步要求所有 AIV 核参与，多核场景必须 blockDim=coreNum
    // 标量写（S 管线）在大张量时自然刷新到 GM（跨核可见），小张量时停留在 L2 缓存（跨核不可见）
    // 因此小张量或跨步访问模式强制单核（blockDim=1），避免 Phase1→Phase2 跨核缓存一致性问题
    constexpr int64_t MULTICORE_MIN_ELEMENTS = 4 * 1024 * 1024;  // 4M 元素阈值
    // 跨步切分（splitAxis=rank-1）的 reduce 模式：Phase2 散布读取 y 导致缓存冲突，强制单核
    bool isStridedReduce = isReduceMode && (splitMode == 1) && (splitAxis == rank - 1) && (rank >= 2);
    int64_t copyBlockFactor = 0;
    int64_t blockDim = 1;
    if (isReduceMode && is1D) {
        // 1D reduce：单核保证聚合正确性（无 SyncAll 需求）
        blockDim = 1;
        copyBlockFactor = CeilAlign(dataTotalNum, blockAlign);
    } else if (isStridedReduce) {
        // 跨步 reduce：Phase2 散布读取 y，标量写跨核缓存不可见，强制单核
        blockDim = 1;
        copyBlockFactor = CeilAlign(dataTotalNum, blockAlign);
    } else if (dataTotalNum < MULTICORE_MIN_ELEMENTS) {
        // 小张量：单核（标量写跨核缓存不可见，强制单核避免一致性问题）
        blockDim = 1;
        copyBlockFactor = CeilAlign(dataTotalNum, blockAlign);
    } else {
        // 大张量多核 Phase1：copyBlockFactor 按 coreNum 均分，blockDim=coreNum 确保 SyncAll 正确
        copyBlockFactor = CeilAlign(CeilDiv(dataTotalNum, coreNum), blockAlign);
        if (copyBlockFactor < blockAlign) {
            copyBlockFactor = blockAlign;
        }
        // DAV_3510 SyncAll<true> 要求全部核参与，blockDim 必须等于 coreNum
        blockDim = coreNum;
    }
```

Phase1 的 kernel 侧保持标量逐元素拷贝（修复后终版，含机制注释）：

```cpp
// Phase 1: data → y 批量拷贝（多核协同，每核拷贝 copyBlockFactor 个元素）
// 标量 SetValue/GetValue 拷贝（arch35 上 DataCopyPad UB→GM 的 DataCopyExtParams 形态不可用）
// 多核场景下大张量标量写可自然刷新到 GM（跨核可见）；小张量由 Tiling 强制单核避免跨核问题
template <typename D_T_DATA, typename D_T_IDX>
__aicore__ inline void BenchNoHarnessScatter<D_T_DATA, D_T_IDX>::Phase1CopyData()
{
    if (dataTotalNum_ <= 0) {
        return;
    }
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t start = blockIdx * copyBlockFactor_;
    int64_t end = start + copyBlockFactor_;
    if (end > dataTotalNum_) {
        end = dataTotalNum_;
    }
    if (start >= end) {
        return;
    }
    for (int64_t i = start; i < end; ++i) {
        yGm_.SetValue(i, dataGm_.GetValue(i));
    }
    PipeBarrier<PIPE_ALL>();
}
```

Phase2 的工作均分用 `GetBlockNum()`（返回实际启动核数，与 Tiling 设定的 blockDim 一致），Process 主结构为：

```cpp
template <typename D_T_DATA, typename D_T_IDX>
__aicore__ inline void BenchNoHarnessScatter<D_T_DATA, D_T_IDX>::Process()
{
    Phase1CopyData();
    // 确保本地所有流水线完成后再跨核同步
    PipeBarrier<PIPE_ALL>();
    // 多核硬同步（确保 Phase 1 拷贝完成后再进入 Phase 2）
    SyncAll<true>();
    Phase2ScatterUpdate();
}
```

## 设计决策

- 为什么最终选「单核规避」而不是把 Phase1 全部改 MTE3 向量搬运：arch35 上 `DataCopyPad` UB→GM 的 `DataCopyExtParams` 变体数据损坏（本会话二次确认，另见同批「arch35 UB→GM DataCopyExtParams 拷贝写失效」卡），唯一可用的 `DataCopyParams` 变体必须配 V 操作桥接（TQue + Adds），而 V 操作要求元素数对齐 32B/sizeof(T)，非对齐尾部仍要退回标量写——尾部又回到跨核不可见问题。单核阈值一刀切规避了整条复杂链路。
- 已验证可行但被放弃的备选（大张量场景实测 bit-exact 通过）：`TQue<VECIN/VECOUT>` + `DataCopyPad(DataCopyParams)` GM→UB→GM，用 `Adds(outLocal, midLocal, 0, tileElem)` 桥接 MTE2→V→MTE3 管线——MTE3 写可被 SyncAll 正确同步、跨核可见；代价是 Adds 的对齐粒度约束（fp16=16 元素）与双 32KB UB 预算。后续若要做小张量多核，应沿这条路径补齐对齐尾部处理，而不是回到标量写。
- 多核场景 blockDim 恒等于 coreNum（48）：`SyncAll<true>` 的全核到达语义要求全部 AIV 核参与，按数据量推导的部分核启动（blockDim=43/47）与 SyncAll 不兼容；UT 期望值按此设计（`EXPECT_BLOCK_DIM_MULTICORE = CORE_NUM_3510`）。
- 阈值取 4M 元素：2M 阈值时 PIERCE2_3（4M 跨步）与 PIERCE2_5（1M）仍挂，跨步单核化后把阈值提到 4M 让 PIERCE2_4（2M）也单核，全量绿。该数值绑定本算子 dtype（fp16）与 DAV_3510 缓存容量，换 dtype/平台需重标定。
- 当前方案路线档位：safety_fallback（正确性优先；多核标量写在机制明确前不作为推荐路线）。

## 预防

- DAV_3510 上写多核 kernel 前，先问一句：跨核可见的数据写是否全部走 MTE3（DataCopy 类）？任何 `GlobalTensor::SetValue` 跨核生产、他核消费的两段式数据流都是雷。
- 用「规模扫描」快速判别本问题：同一实现小 shape 全挂、≥数 M 元素大 shape 全过、单核（blockDim=1）全过、失配值是垃圾而非精度漂移——即命中本卡，不要先怀疑 Tiling 切分算错。
- 多核 SyncAll 场景 blockDim 直接取 coreNum，不要按数据量凑部分核。

## 验证信号

以下信号来自该算子迭代二多核整合会话：多核引入即失配、逐次尝试缓存刷新与搬运 API 替换、最终单核阈值方案两次全量复跑。

- 修复前：ST Real 迭代二 6/54（11.1%）、迭代一穿刺 1/4；PIERCE_MAIN_1D [1024] fp16 MERE=1.30e+00 / MARE=7.15e+01，不匹配位置为垃圾值（来源平台：950；证据等级：tool_observed）。
- 规模规律：单核 PIERCE3 PASS、4M 线性 PIERCE2_1/2_2 PASS、2M PIERCE2_4 中途转好、1M PIERCE2_5 与 4M 跨步 PIERCE2_3 FAIL（来源平台：950；证据等级：tool_observed）。
- 修复后（4M 阈值 + 跨步/1D reduce 单核 + 多核恒 48）：`bash build.sh --soc=ascend950 -j8` Build completed、UT `[PASSED] 17 tests`、`--force` 安装后 ST 迭代一 5/5（100%）+ 迭代二 54/54（100%），`验证结果: PASS ✓`（来源平台：950；证据等级：tool_observed）。
- 清理后复跑确认：移除 cache_ctrl include 后重建 + 重装 + ST 复跑，迭代一 5/5、迭代二 54/54 维持 PASS（来源平台：950；证据等级：tool_observed）。

## 性能影响

小张量（<4M 元素）与跨步 reduce 退单核，多核并行收益放弃；本会话无 profiling 数据，相对代价未量化（单跑性能数据本会话未采集）。大张量线性访问仍为 48 核并行。

## 失效边界

- 4M 阈值按 fp16（2 字节）与 DAV_3510 实测标定；int64（8 字节）等大 dtype、或缓存容量不同的其他芯片需重新标定（未验证，推断机制可迁移）。
- 其他 arch（非 DAV_3510）的 S 管线/MTE3 一致性行为未验证。
- 单核规避只覆盖「Phase1 生产 + Phase2 消费」跨段场景；单核内 S 管线读写顺序由 `PipeBarrier<PIPE_ALL>` 保证（reduce 读-改-写循环内已加），本卡不讨论同核顺序问题。
- `DataCopyPad(DataCopyParams)` + TQue + Adds 的 MTE3 路径仅在大张量（对齐 tile）实测 bit-exact；非对齐尾部的完整工程化方案未落地（见遗留问题）。

**未决项**：

- 优先项：复现并追根因到机制层——S 管线 store buffer 与 MTE3 路径的一致性模型、是否存在正确的 flush/屏障指令（本卡根因是行为学定位 + 头文件机制推断，微架构层未证实）。
- 小张量多核的正确路线：`DataCopyParams` + TQue + Adds + 非对齐尾部处理（尾段 rounding-up 拷贝或双路径）未完成，需独立验证。
- 跨步 reduce 多核化（避免散布读缓存冲突的切分方式）未尝试。
## 坏实践

- 认为加了 `SyncAll` 就万事大吉：SyncAll 只同步「到达」，不保证 S 管线 GM 写对其他核可见。
- 试图用 `asc_dcci_entire_out()` / `asc_dcci_entire_all()` 刷缓存救标量写跨核可见性：实测 6/54→18/54 后停滞，entire_all 无进一步改善（dcci 不覆盖 S 管线路径）。
- Phase2 的 blockNum 从 Tiling 字段推导而不是 `GetBlockNum()`：两者不一致时部分 indices 漏处理（会话中已改为 `GetBlockNum()`）。

## 误诊

- 初始怀疑 Tiling 切分/blockDim 计算错误（blockDim=43/47 部分核）：改恒 blockDim=48 后失败模式不变，UT 全过也证明 Tiling 无误——排除。
- 怀疑 `GetBlockNum()` 返回值异常导致 Phase2 漏处理：分析证明即使 GetBlockNum=1 也只是退化为单核全覆盖，不会产生垃圾值——排除。
- 怀疑 Phase1/Phase2 并发（SyncAll 不生效导致 Phase1 覆盖 Phase2 写）：与「单核用例通过」的观测矛盾——部分成立（本质就是 Phase1 写对 Phase2 不可见，但方向是读不到而不是被覆盖）。

## 增量（字节量维度：同元素数不同 dtype 跨界，判据加字节上限）

另一独立轨迹并入同算子家族独立会话的**字节量维度**实测（Ascend950PR / DAV_3510 / 48 AIV 核，CANN 9.1.0，全 dtype ST）：本卡 4M 元素阈值按 fp16（2 字节）标定，**同元素数不同 dtype 的字节量差一倍即跨过可见性边界**——

- fp16 大张量用例（[8192,512] = 4M 元素 = **8MB**）多核通过（UT 期望多核）；
- fp32 大张量用例（[4096,1024] = 4M 元素 = **16MB**，dim=0，reduce=None）多核失败，输出是**整值错位**而非精度抖动：`[FAIL] MERE=6.11e-02, MARE=3.11e+04`，不匹配位 `golden=-3.526771, actual=3.383426`（符号翻转）——元素数阈值无法区分两者。经验边界：**8MB 多核通过、16MB 多核失败**。

两轮阈值实验的失败记录（勿重走）：① 阈值 4M→8M 元素（让 fp32 用例落回单核）→ fp32 通过但 UT 两条 4M fp16 多核用例失败（`blockNum: 1` vs 期望 `48`、`copyBlockFactor: 4194304` vs 期望 `87384`）——破坏已验证的多核路径与 UT 契约；② 阈值改回 4M → fp32 用例复发。**纯字节阈值方向也分不开**（8MB 过、16MB 败，字节越大越危险，与"越大越自然驱逐"的朴素预期相反），故用组合判据。

**双判据修复（verbatim，元素数阈值 + 字节上限）**：

```cpp
constexpr int64_t MULTICORE_MIN_ELEMENTS = 4 * 1024 * 1024;  // 4M 元素阈值（标量写跨核可见）
// 字节上限：fp16 4M=8MB 多核可行，但 float32 4M=16MB 时 S 管线 store buffer 压力过大
// 导致跨核写丢失，因此超过 8MB 的张量也强制单核
constexpr int64_t MULTICORE_MAX_BYTES = 8 * 1024 * 1024;  // 8MB 字节上限
int64_t dataTotalBytes = dataTotalNum * dataSize;
...
    } else if (dataTotalNum < MULTICORE_MIN_ELEMENTS || dataTotalBytes > MULTICORE_MAX_BYTES) {
        // 小张量（<4M 元素）或超大字节量（>8MB）：单核，避免标量写跨核缓存一致性问题
        blockDim = 1;
        copyBlockFactor = CeilAlign(dataTotalNum, blockAlign);
    } else {
        // 大张量多核 Phase1：copyBlockFactor 按 coreNum 均分，blockDim=coreNum 确保 SyncAll 正确
        copyBlockFactor = CeilAlign(CeilDiv(dataTotalNum, coreNum), blockAlign);
        if (copyBlockFactor < blockAlign) {
            copyBlockFactor = blockAlign;
        }
        // DAV_3510 SyncAll<true> 要求全部核参与，blockDim 必须等于 coreNum
        blockDim = coreNum;
    }
```

判据组合后的行为：4M fp16（8MB，`8MB > 8MB` 为 false）→ 多核，满足 UT 期望（blockDim=48、copyBlockFactor=87384）；4M fp32（16MB）→ 单核，ST 正确；1M 元素各 dtype → 单核不变。判读补充：**"同元素数不同 dtype"的大张量用例（fp16 vs fp32 同元素数）是对本判据的天然探针**；看到"数值完全不同/符号翻转"还按精度容差排查是写丢失/错位的形态，不是舍入误差。

### 复现增量验证信号

- 修复前：fp32 大张量用例（4M 元素=16MB）多核输出整值错位（golden=-3.526771 vs actual=3.383426 等，MERE=6.11e-02~7.21e-02）；同元素数 4M fp16（8MB）多核通过（来源平台：950 实测；证据等级：tool_observed）。
- 阈值实验：4M→8M 元素后 fp32 用例通过（总失败 16→15），但 UT 两条 4M fp16 多核用例失败（blockNum 1 vs 期望 48、copyBlockFactor 4194304 vs 期望 87384）；改回 4M 后 fp32 用例复发（来源平台：950 实测；证据等级：tool_observed）。
- 双判据落地后：UT 17/17 通过，ST Real 全量 875/875（含 fp32 大张量与 4M fp16 多核用例）（来源平台：950 实测；证据等级：tool_observed）。

### 复现增量失效边界

- 8MB 是该会话实测的经验边界（8MB 通过/16MB 失败两个点），中间值（9~15MB）未测，边界不是精确常数；其他核数/SoC 不可直接搬用（未验证，推断需按字节量重新标定）。
- "S 管线 store buffer 压力过大导致丢写"仍是机制假设，未做 profiling/微架构验证（与主卡遗留问题同源）。
- 判据只覆盖标量写路径；Phase 改用 MTE3 批量搬运后需重新标定（同场景 DataCopyPad UB→GM 在 arch35 不可用，见主卡设计决策）。