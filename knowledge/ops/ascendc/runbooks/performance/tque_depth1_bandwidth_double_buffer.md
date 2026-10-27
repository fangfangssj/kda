---
type: "Runbook"
title: TQue 深度 1 带宽减半：fp32 双缓冲修复
description: "逐元素 tile-loop kernel 的 TQue 队列深度 1 串行化 CopyIn/Compute/CopyOut，有效带宽约为推断 HBM 峰值一半（≈49.7%）；fp32 路径启用深度 2 双缓冲并让 tiling bufferDivisor 同步乘 depthFactor 重算 ubFormer 后，带宽仅升至推断峰值的 ≈52~53%（纯拷贝变体 ≈57%），≥80% 目标未达成——双缓冲是必要修改但非充分解，剩余缺口根因待查。"
tags: [task.optimization, topic.data_movement, topic.tiling]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# TQue 深度 1 带宽减半：fp32 双缓冲修复

## 现象

在 Ascend950PR（CANN 9.1.0）上为自定义逐元素算子 softplus_v2_grad 实现完 tile-loop 分支 kernel 并通过全部 25 个精度用例后，进行带宽验收：对 L1 用例 `[4096,4096]` fp32（16,777,216 元素，2 读 + 1 写 = 12 B/elem，共 192 MiB 有效流量），用 aclrtEvent 计时 20 次取最优/中位，测得端到端有效带宽 best/median 均为推断 HBM 峰值的 ≈49.7%（峰值口径未验证，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)）。该算子按设计属访存带宽受限（设计文档要求有效带宽利用率 ≥ 80% HBM 峰值），而按本机硬件参数推断的 ~3.28 TB/s HBM 峰值（HBM3 3200 MT/s × 1024 bit 总线宽度推得，推断口径非实测）下，利用率只有 ≈49.7%，带宽验收失败。

当时的 kernel 状态：三个队列（两个 VECIN 输入 + 一个 VECOUT 输出）全部是 `AscendC::TQue<..., 1>` 深度 1，每个 tile 走 `CopyIn → Compute → CopyOut` 的同步序列。判别性证据来自 copy-only 探针：把 VF 计算链整体替换为单条 `Adds(dxF, dyF, 0.0f, mask)`（即 dxF = dyF + 0，几乎零计算）后重测，带宽只比完整计算链提升约 5%（copy-only 探针相对完整计算链，同平台同用例成对对照）——这说明 kernel 不是计算受限，而是搬运路径受限；当时的初步归因是"深度 1 使 MTE2 读、VF 计算、MTE3 写串行等待，把带宽压到峰值一半"，该归因后来被修后复测数据部分证伪（串行只占约 3 个百分点，见根因与误诊）。

## 适用场景

- 工程形态：aclnn 自定义算子开发（AscendC tile-loop kernel + host tiling 联动），性能验收阶段。
- API 路径：`AscendC::TQue` 模板深度参数 / `TPipe::InitBuffer(que, bufferNum, bufLen)` / host 侧 TilingFunc 的 UB 切分公式（bufferDivisor）。
- 验证过的平台与 CANN 版本：Ascend950PR（A310-50 卡，AICore 28 / AIV 56，HBM3 3200 MT/s 128 GB），CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有"逐元素、2 读 1 写、tile-loop 切分"的带宽受限 AscendC kernel。
- 路线优先级：recommended_fastpath（fp32 路径 UB 预算允许时的标准做法、修后 ST/UT 全过的必要修改；但本轨迹实测它不足以把利用率提到 ≥80%，见验证信号与遗留问题）。
- 明确不适用 / 未验证场景：
  - fp16 升精度路径（K=3 份 fp32 中间量占 UB）本卡未启用双缓冲，保持深度 1（见失效边界）。
  - 修后带宽为推断峰值的 ≈52~53%，未达 ≥80% 验收线；剩余约 47 个百分点缺口的根因未定位（见验证信号与遗留问题）。
  - 非 tile-loop 的 full-load 单 tile kernel：tile 间无循环，深度 2 的收益机制不同，未验证。

## 根因

TQue 队列深度 1（无 double-buffer）使每个 tile 的 CopyIn（MTE2）、Compute（VF）、CopyOut（MTE3）串行执行，搬运与计算无法重叠。

机制链：深度 1 时第 i 个 tile 的写回必须等第 i 个 tile 的计算完成、且第 i+1 个 tile 的读入必须等第 i 个的缓冲释放——单 tile 时间 ≈ 读时间 + 算时间 + 写时间（三段相加）。启用深度 2 后，第 i 个 tile 的 CopyOut 与第 i+1 个 tile 的 CopyIn/Compute 在双份 UB 缓冲间流水重叠，读写在时间上交错。但修后实测证伪了"重叠后逼近峰值（max(读, 写)）"的预期：带宽只从 ≈49.7% 升到 ≈52~53%（推断峰值口径），与修复前 copy-only 探针的 ~52% 相当——说明深度 1 串行只解释了约 3 个百分点的损耗（把计算藏进搬运），剩余约 47 个百分点缺口另有主导瓶颈且根因未定位（候选方向：MTE 发射粒度/频率、tile 大小、核数调度、访存模式，见遗留问题）。"深度 1 串行 → 带宽减半"不是完整机制，只是与 ~50% 数值巧合的部分解释。

另一个被实测推翻的原始设计判定：kernel 首版注释称"当前 tiling 的 ubFormer 已填满 fp32 的 UB（bufferDivisor=12B），UB 余量不足 → 深度 1"。该判定把"深度 1 口径下 ubFormer 填满 UB"误当成"无余量"——正确做法是启用深度 2 的同时把 tiling 的 I/O 预算翻倍（ubFormer 减半），UB 总占用不变而流水重叠生效。

## 解决配方

修复分两侧，kernel 与 tiling 必须同步改，否则 UB 预算失配：

1. kernel 侧：TQue 模板深度参数 1→2（模板参数取最大深度），实际使用的 bufferNum 由编译期常量 `DEPTH` 按 dtype 静态选择（fp32=2，fp16=1），`InitBuffer` 传 `DEPTH`。

修复前（深度 1，节选自统一 tile-loop kernel 的 private 段与 Init 尾部，verbatim）：

```cpp
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueDY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDX_;
```

```cpp
        uint32_t bufBytes = static_cast<uint32_t>(ubFormer_ * sizeof(DT));
        pipe_.InitBuffer(inQueueX_, 1, bufBytes);
        pipe_.InitBuffer(inQueueDY_, 1, bufBytes);
        pipe_.InitBuffer(outQueueDX_, 1, bufBytes);
```

修复后（verbatim）：

```cpp
private:
    // DESIGN-BRANCH-2 §3/§4: fp32 (DT=float) UB 余量充足 → TQue 深度 2 (double-buffer)
    // 重叠 CopyIn/Compute/CopyOut；fp16 (DT=half) 保 K=3 中间量 → 深度 1。
    // Tiling 侧 bufferDivisor 按相同 depthFactor 算 ubFormer，保证不溢出 UB。
    static constexpr bool isHalf_ = (sizeof(DT) == sizeof(half));
    static constexpr int DEPTH = isHalf_ ? 1 : 2;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inQueueDY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueueDX_;
```

```cpp
        // UB buffers sized to ubFormer (single-tile capacity). For full-load
        // branch ubFormer >= blockFormer so the whole per-block set fits;
        // for tile-loop ubFormer < blockFormer and we loop per tile.
        // Tiling guarantees depth*ubFormer*elemBytes*3 (+K*4 for fp16) <= ubSize.
        // depth = DEPTH (fp32 double-buffer=2, fp16=1); Tiling bufferDivisor matches.
        uint32_t bufBytes = static_cast<uint32_t>(ubFormer_ * sizeof(DT));
        pipe_.InitBuffer(inQueueX_, DEPTH, bufBytes);
        pipe_.InitBuffer(inQueueDY_, DEPTH, bufBytes);
        pipe_.InitBuffer(outQueueDX_, DEPTH, bufBytes);
```

2. tiling 侧（`softplus_v2_grad_tiling_host.cpp`，公共切分公式、full-load 分支、tile-loop 分支三处同构修改，verbatim 以 tile-loop 为例）：

修复前：

```cpp
    // —— UB 切分（extraSize = 0；bufferDivisor 字节口径）——
    // fp32 直算 K=0：bufferDivisor = bufferNum * elemBytes
    // fp16 升精度 K=3：bufferDivisor = bufferNum * elemBytes + K * 4（K 份 fp32 中间）
    int64_t bufferDivisor = static_cast<int64_t>(in.bufferNum) * in.elemBytes
                            + static_cast<int64_t>(in.kBuf) * 4;
    int64_t maxElemNum = in.ubSize / bufferDivisor;
    int64_t alignFactor = kAlign256 / in.elemBytes;  // fp32→64 元素；fp16→128 元素
    int64_t ubFormer = AlignDown(maxElemNum, alignFactor);
    if (ubFormer < 1) {
        ubFormer = alignFactor;
    }
```

修复后：

```cpp
    // —— UB 切分（extraSize = 0；bufferDivisor 字节口径）——
    // fp32 直算路径（K=0）开启 double-buffer (TQue 深度 2)；
    // fp16 升精度路径（K=3）保 K 份中间量，深度 1。depthFactor 由 kBuf 推导。
    int32_t depthFactor = (in.kBuf == 0) ? 2 : 1;
    int64_t bufferDivisor = static_cast<int64_t>(depthFactor) * in.bufferNum * in.elemBytes
                            + static_cast<int64_t>(in.kBuf) * 4;
    int64_t maxElemNum = in.ubSize / bufferDivisor;
    int64_t alignFactor = kAlign256 / in.elemBytes;  // fp32→64 元素；fp16→128 元素
    int64_t ubFormer = AlignDown(maxElemNum, alignFactor);
    if (ubFormer < 1) {
        ubFormer = alignFactor;
    }
```

关键联动算式：fp32 时 `bufferDivisor` 从 `3×4=12` 变为 `2×3×4=24` 字节/元素，`ubFormer = AlignDown(ubSize/24, 64)` 约减半；kernel 侧 UB 总占用 `DEPTH × ubFormer × elemBytes × 3` 保持 ≤ ubSize 不溢出，tile 数翻倍但每 tile 更小，流水重叠生效（但修后实测只收回约 3 个百分点带宽，见验证信号）。

3. UT oracle 同步：三个 TilingUT 的独立 oracle 函数按相同 depthFactor 公式更新；因 fp32 的 ubFormer 减半，钉死边界值的用例必须重算——公共 UT `boundary_equal_fullload` 的 ubSize 从 6144 提到 12288（bufferDivisor 12→24，保持 ubFormer=512 使 dim0=512 仍归 full-load）；full-load UT `boundary_equal_ubFormer` 的 dim0 从 16384 改 8192（ubFormer 16384→8192，保持 blockFormer==ubFormer 边界）。oracle 更新（verbatim，三个文件同构）：

```cpp
    // UB 切分（extraSize = 0）
    // fp32 (kBuf=0) 开 double-buffer → I/O Queue 翻倍 (depthFactor=2)；
    // fp16 (kBuf=3) 保 K 份中间量 → depthFactor=1。depthFactor 由 kBuf 推导。
    int32_t depthFactor = (kBuf == 0) ? 2 : 1;
    int64_t bufferDivisor = static_cast<int64_t>(depthFactor) * bufferNum * elemBytes
                            + static_cast<int64_t>(kBuf) * 4;
```

```cpp
INSTANTIATE_TEST_SUITE_P(Section55, BranchBoundaryTest, testing::Values(
    // ubSize=12288, fp32 (depthFactor=2, bufferDivisor=24) -> ubFormer=512;
    // dim0=512 -> blockFormer=512 == ubFormer -> full-load(1)
    FormulaCase{"boundary_equal_fullload", 512, 32, 12288, 4, 32, 3, 0},
    // dim0=513 -> blockFormer=1024 > ubFormer=512 -> tile-loop(2), ubTailOfTailBlock=1
    FormulaCase{"boundary_over_tileloop",  513, 32, 12288, 4, 32, 3, 0}
));
```

## 设计决策

- 为什么深度由 kernel 模板参数静态选择而不是 TilingData 字段：dtype 已由编译期 `-DDTYPE_GRAD_OUTPUT=float/half` 注入模板参数，深度判定（fp32=2/fp16=1）只依赖 dtype，进 TilingData 会引入运行时分支与 tiling/kernel 两侧的一致性负担。
- 为什么模板参数写 2 而实际 InitBuffer 传 DEPTH：TQue 模板参数是编译期最大深度，InitBuffer 的 bufferNum 是实际启用深度；fp16 实例化时 DEPTH=1，模板仍是 2（与 CANN ElemwiseOpBase 的 `TQue<..., QUE_MAX_DEPTH=2>` + `InitBuffer(que, pingpongNum, ...)` 模式一致）。
- 为什么 tiling 必须同改：只改 kernel 不改 tiling 的话，ubFormer 仍按深度 1 口径计算，kernel 会申请 2 倍 UB 而 TilingData 的不变式 `3×ubFormer×elemBytes ≤ ubSize` 失配，可能溢出 UB。
- 为什么 fp16 不启用：fp16 路径的 K=3 份 fp32 中间量预算已计入 bufferDivisor（`+K*4`），再翻倍 I/O 会使 ubFormer 过小、tile 过碎；本轨迹未验证 fp16 双缓冲的收益与代价。
- 当前方案路线档位：recommended_fastpath（fp32 路径的标准做法与必要修改，但非充分解——修后 ≈52~53% 仍未达 ≥80%，见验证信号）。

## 系统信息

- 硬件/软件：Ascend950PR 单卡（npu-smi：Product A310-50-C00MM304A1、HBM 131072 MB @3200 MHz、Aicore Count 28 / AIV 56），CANN 9.1.0，算子包构建 `cd operators && ./build.sh softplus_v2_grad_package`。
- 目录布局（算子包内）：
  - `op_kernel/softplus_v2_grad_kernel.cpp`——统一 tile-loop kernel（TQue 深度、InitBuffer、Process/ProcessTile/CopyIn/Compute/CopyOut、单 `__simd_vf__` VF 链）。
  - `op_host/softplus_v2_grad_tiling_host.cpp`——公共切分公式 + full-load/tile-loop TilingFunc（depthFactor 公式）。
  - `tests/ut/test_softplus_v2_grad_tiling.cpp`、`test_softplus_v2_grad_branch1_tiling.cpp`、`test_softplus_v2_grad_branch2_tiling.cpp`——独立 oracle TilingUT。
  - `tests/aclnn/`——ST 框架（run.sh / test_aclnn.cpp / golden 生成 manifest）。
- 带宽测量：aclrtEvent 计时（warmup 5 + 20 iter，取 best/median），有效流量按 2R+1W=12 B/elem 计算；HBM 峰值按硬件参数推断（3200 MT/s × 1024 bit），非实测值。

## 验证信号

来源说明：以下信号来自该算子 tile-loop 分支带宽验收与双缓冲修复会话：先由事件计时与 copy-only 探针量化带宽瓶颈，再实施 kernel+tiling+UT 三侧联动修复，最后重跑构建、TilingUT、全量 ST 与带宽复测（修后 ≈52~53%，≥80% 未达成）。

- 修复前（深度 1，完整计算链）：L1 `[4096,4096]` fp32 有效带宽 best/median/avg 与修复会话复测 baseline 均稳定在推断 3.28 TB/s 峰值的 ≈49.7%（20 iter，bitwise 2 次运行一致；峰值口径未验证，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)）（来源平台：950；证据等级：tool_observed）。
- 修复前（深度 1，copy-only 探针 `Adds(dxF, dyF, 0.0f, mask)`）：相对完整计算链仅快 ~5%（约为推断峰值的 ~52%），证明带宽/流水受限而非计算受限（来源平台：950；证据等级：tool_observed）。
- 修复后：`./build.sh softplus_v2_grad_package` 编译成功 exit 0（来源平台：950；证据等级：tool_observed）。
- 修复后：三个 TilingUT 全过——公共 22/22、full-load 11/11、tile-loop 9/9（含 depthFactor 公式更新后的 oracle 与重算的边界用例）（来源平台：950；证据等级：tool_observed）。
- 修复后（fp32 双缓冲）带宽复测：L1 `[4096,4096]` fp32 有效带宽为推断 3.28 TB/s 峰值的 ≈52%（事件计时 30 iter 两次复测一致），kernel-only aclrtEvent（N=100）口径 ≈53%——相对修复前 ≈49.7% 仅提升约 2~3 个百分点，设计文档 ≥80% 验收线未达成（峰值口径未验证，见 [950PR HBM 峰值判定：GetCoreMemBw=31 非带宽](getcoremembw_hbm_ini_rate_not_bandwidth.md)）（来源平台：950；证据等级：tool_observed）。
- 修复后纯拷贝变体（同 tiling、零计算，kernel-only aclrtEvent）：约为推断峰值的 ≈57%；计算版达纯拷贝变体的 92.6%（与 [带宽受限 kernel 计算链优化无带宽收益](bandwidth_bound_compute_optimization_no_gain.md) 同一修后基线）——修后完整计算链已与修复前 copy-only 探针水平（~52%）相当，剩余缺口在搬运路径本身（来源平台：950；证据等级：tool_observed）。
- 修复后全量 ST 复测：total=25 PASSED=25 FAILED=0 ERROR=0（Branch-1 15/15、Branch-2 10/10），RUN_EXIT=0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- fp16 升精度路径（K=3 中间量）本卡保持深度 1；fp16 双缓冲未验证，推断可迁移但须先重算 K 份中间量与 I/O 翻倍的 UB 预算。
- 双缓冲使 fp32 的 ubFormer 约减半：极小 shape 或 tile 数本就很少的场景，深度 2 的第二份缓冲纯属浪费；UB 本就紧张的 kernel（多输入/大中间量）启用前须复核 `depth×ubFormer×elemBytes×队列数 ≤ ubSize`。
- 本卡修后实测止于：带宽 ≈52~53%（推断峰值口径，kernel-only ≈53%）、纯拷贝变体 ≈57%、全量 ST 25/25——≥80% 未达成，剩余约 47 个百分点缺口根因未定位；采用双缓冲后仍须按验收线复测，并另行排查缺口（见遗留问题）。
- HBM 峰值 ~3.28 TB/s 为硬件参数推断口径，非实测；利用率百分比随峰值口径变化，判读时先核对峰值来源。
- 仅在 Ascend950PR + CANN 9.1.0 实测；其他平台的队列/流水机制可迁移但未验证（推断可迁移）。

**未决项**：

- 优先项：定位修后仍存在的约 47 个百分点带宽缺口（推断峰值口径下 ≈52~53% vs ≥80% 目标）的根因——深度 1 串行已被双缓冲消除、计算已被纯拷贝对照证明隐藏（计算版为纯拷贝变体的 92.6%），剩余瓶颈须从搬运路径本身（MTE 发射粒度/频率、tile 大小、核数调度、访存模式）排查；未定位前不得把本卡修后数值当达标口径引用。
- fp16 路径启用双缓冲的可行性与收益未尝试。
- 双缓冲后 tile 数翻倍对尾块（ubTail 很小）场景的效率影响未单独测量。
## 坏实践

- 只改 kernel 的 TQue 深度而不同步改 tiling 的 bufferDivisor——UB 预算失配，深度 2 会申请超出 tiling 不变式保证的 UB。
- 以"当前 tiling 的 ubFormer 已填满 UB"为由拒绝双缓冲——那是深度 1 口径的自我循环论证；正确判据是实测带宽是否达标 + 深度 2 口径下重算 ubFormer 是否仍为正。
- 把 DMA 拷贝带宽或未优化的参考算子带宽当 HBM 峰值来算利用率。

## 误诊

- 修复前的主流解释"深度 1 串行使读/写时间相加、带宽被压到峰值一半，双缓冲后应逼近 max(读, 写) 即峰值"——修后实测仅 +2~3 个百分点（≈52~53%），该机制预期被证伪：串行只占小头，主导瓶颈另有其因（未定位）；修复前 ~50% 的数值与"串行减半"公式巧合相符是本卡最初的误判来源（实测推翻它的修后带宽数据为 tool_observed）。
- kernel 开发者自评时曾假设一个无依据的低 HBM 峰值分母（仅为推断 3.28 TB/s 峰值的约 0.37x），按该分母得出 ≈128% 利用率"远超 80%"而宣称带宽验收通过——该假设无任何依据，被多探针推断的 ~3.28 TB/s 峰值推翻，实际利用率约 50%（推翻它的实测带宽（约为推断峰值的 49.7%）与峰值推断为 tool_observed）。
- 验收排查初期曾把系统级 HBM 采样与核内带宽查询接口的返回值当候选峰值来源，均不可用（采样频率过低/语义不符）。
