---
type: "Runbook"
title: "Group Phase2读y别名workspace与CopyOut竞态"
description: "多核 group 路径把 m_part/s_part workspace 别名到输出 y 后，Phase 2 先读 workspace 再 CopyOut 写 y，A 分块数>1 时早块输出覆盖晚块待读数据，bf16 大 A 用例 mere 千级错乱；修复为 Phase 2 直接从 GM x 三遍重算（复用非分组路径），不读 workspace。"
tags: [task.debugging, topic.memory, topic.compute.normalization, topic.compute.reduction]
status: stable
sources: []
created_at: '2026-09-16T11:13:37Z'
updated_at: '2026-09-16T11:13:37Z'
platforms: ["950"]
---

# Group Phase2读y别名workspace与CopyOut竞态

## 现象

在 Ascend950PR（CANN 9.2.0）上对自研 L2 softmax 基准算子（AscendC，arch35）做 tilingKey=4（GroupTailA，R 外层/A 内层转置多核分组路径）白盒全量测试（TTK kernel 模式，24 条）时，2 条 bf16 用例灾难性失败：

- 用例 A：bfloat16，shape `(2049, 2047)`，`dim=-2`，输入范围 `(-0.01, 0.01)`，`mere=1707.48`、`mare=234661.64`，阈值 mere=2^-7=0.0078125 / mare=0.078125——超阈 6~7 个数量级；`reason: 'threshold exceeded: mere(1.71e+03>=0.0078125), mare(2.35e+05>=0.078125)'`。
- 用例 B：bfloat16，同 shape，`dim=0`（对 rank=2 与 dim=-2 折叠等价），`mere=1707.13`、`mare=235470.78`——与用例 A 几乎相同，同根因。

softmax 合法输出 ∈ [0,1] 且每条归约轴和为 1，mere 达千量级只可能是输出根本不是有效 softmax 数值。两条失败用例 OOB/内存检查全 PASS（`dyn_oob_result="OK,OK"`、`memory_oob_status=PASS`），排除越界写；同 shape 的 fp16/fp32 2D 用例 mere≈1.18e-5 全 PASS，失败组合收敛到「bf16 + key=4 + A 轴 2047（触发 A 分块切分）」。该 shape 在 key=4 路径下 A 需切 19 个分块（aSplitChunkCnt=19，nColsAlign=112），而通过的 key=3 用例全部 aSplitChunkCnt=1。

## 适用场景

- 工程形态：AscendC 多核分组（group 两阶段 Phase1/Phase2）kernel 的白盒/ST 精度修复；workspace 与输出别名复用的算子。
- API 路径：`AscendC::DataCopyPad` GM workspace 读写 + `PipeBarrier<PIPE_ALL>` + `SyncAll` 两阶段同步；`asc_vf_call` 重缩放 VF。
- 验证过的平台与 CANN 版本：Ascend950PR（TTK SOC 串 Ascend950PR_9579），CANN 9.2.0。
- 适用粒度：operator_family——所有「分组归约 + workspace 别名输出 + A/R 多分块」的 softmax/归约类 kernel。
- 路线优先级：recommended_fastpath（Phase 2 直算重构一次解决竞态与重缩放误差，代价仅 Phase 2 多一遍 GM 读）。
- 明确不适用 / 未验证场景：aSplitChunkCnt=1 的形状（无竞态，本开发会话中 key=3 全部用例即此类）；框架可正常传独立 workspace GM_ADDR 的调用路径（本路径已验证不可用，见失效边界）。

## 根因

kernel `Init()` 在 group 模式（tilingKey=3/4）把 m_part/s_part 的 GM workspace **别名到输出张量 y**（`gmWs_.SetGlobalBuffer((__gm__ float*)y)`，因为 aclnn 框架不向该 kernel 传递有效的独立 workspace GM_ADDR）；Phase 2 对每个 A 分块「先读 workspace 算 final_m/final_s，再 Pass 3 CopyOut 写 y」。当 aSplitChunkCnt>1 时，**早分块的 CopyOut 覆盖了后续分块尚未读取的 workspace（即 y 的同一段内存）**，后续分块用被输出数据污染的 m/s 计算出完全错误的 softmax 值。bf16 (2049,2047) 是首个 aSplitChunkCnt=19 的用例形状，故灾难性失败首次在此暴露；同路径 fp32 大 R 用例 aSplitChunkCnt=1 不触发竞态，其失败另有原因。

单句根因：**key=4 GroupTailA 的 Phase 2 从别名到输出 y 的 workspace 读 m/s，与自身 CopyOut 写 y 形成读-写竞态（aSplitChunkCnt>1 时早块覆盖晚块待读数据）。**

## 解决配方

**before**（`op_kernel/arch35/bench_harness_softmax_kernel.h` 原始 `ProcessGroupPhase2TailA`，节选关键段；Phase 1 先把各 rGroup 的 m_part/s_part 写入 workspace[y]，Phase 2 逐 aSub 读回合并，随后同函数内 Pass 3 CopyOut 写 y）：

```cpp
    __aicore__ inline void ProcessGroupPhase2TailA(int64_t aLoopIdx)
    {
        int64_t aSubCnt = ValidASubTailA(aLoopIdx);
        int64_t validACols = ValidAColsTailA(aLoopIdx);
        int64_t nColsAlign = TileAColsAlign();
        int64_t aTotal = GroupATotalTailA();

        for (int64_t aSub = 0; aSub < aSubCnt; ++aSub) {
            int64_t aOff = WsAOffTailA(aLoopIdx, aSub);

            // ── Pass A: final_m = max over rGroups of m_part ──
            InitMaxCacheTailA();
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int64_t rGroupIdx = 0; rGroupIdx < td_->rGroupCnt; ++rGroupIdx) {
                int64_t wsOffM = rGroupIdx * aTotal + aOff;
                auto dst = tmpBuf0_.Get<float>();
                AscendC::DataCopyExtParams extParams;
                extParams.blockCount = 1;
                extParams.blockLen = static_cast<uint32_t>(validACols * sizeof(float));
                extParams.srcStride = 0;
                extParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<float> padParams;
                padParams.isPad = true;
                padParams.paddingValue = BitsToFp32(0xFF800000u);  // -inf
                AscendC::DataCopyPad(dst[0], gmWs_[wsOffM], extParams, padParams);
                AscendC::PipeBarrier<PIPE_ALL>();
                AccumulateMaxTailA(validACols);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // ── Pass B: final_s = Σ s_part × exp(m_part − final_m) ──
            InitSumCacheTailA();
            AscendC::PipeBarrier<PIPE_ALL>();
            __ubuf__ float* cacheBase = (__ubuf__ float*)cacheBuf_.Get<float>().GetPhyAddr();
            __ubuf__ float* finalMPtr = cacheBase;
            __ubuf__ float* finalSPtr = cacheBase + nColsAlign;
            for (int64_t rGroupIdx = 0; rGroupIdx < td_->rGroupCnt; ++rGroupIdx) {
                int64_t wsOffM = rGroupIdx * aTotal + aOff;
                int64_t wsOffS = td_->rGroupCnt * aTotal + rGroupIdx * aTotal + aOff;
                auto dstM = tmpBuf0_.Get<float>();
                auto dstS = tmpBuf1_.Get<float>();
                AscendC::DataCopyExtParams extParams;
                extParams.blockCount = 1;
                extParams.blockLen = static_cast<uint32_t>(validACols * sizeof(float));
                extParams.srcStride = 0;
                extParams.dstStride = 0;
                AscendC::DataCopyPadExtParams<float> padM;
                padM.isPad = true;
                padM.paddingValue = BitsToFp32(0xFF800000u);  // -inf (max identity)
                AscendC::DataCopyPadExtParams<float> padS;
                padS.isPad = true;
                padS.paddingValue = 0.0f;  // 0 (sum identity)
                AscendC::DataCopyPad(dstM[0], gmWs_[wsOffM], extParams, padM);
                AscendC::DataCopyPad(dstS[0], gmWs_[wsOffS], extParams, padS);
                AscendC::PipeBarrier<PIPE_ALL>();
                __ubuf__ float* mPartPtr = (__ubuf__ float*)dstM.GetPhyAddr();
                __ubuf__ float* sPartPtr = (__ubuf__ float*)dstS.GetPhyAddr();
                asc_vf_call<BHSoftmaxRescaleSumVf<VL_F32_U16>>(
                    finalSPtr, mPartPtr, sPartPtr, finalMPtr,
                    static_cast<uint16_t>(validACols));
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // ── Pass 3: Sub+Exp+Div+Cast → CopyOut path 2/3 (re-read x) ──
            for (int64_t rIdx = 0; rIdx < td_->rLoopCntTotal; ++rIdx) {
                int64_t validR = ValidRTailA(rIdx);
                CopyInTailA(aLoopIdx, aSub, rIdx);
                AscendC::PipeBarrier<PIPE_ALL>();
                CastToFp32TailA();
                AscendC::PipeBarrier<PIPE_ALL>();
                DoSubExpDivCastTailA(validR);
                AscendC::PipeBarrier<PIPE_ALL>();
                CopyOutTailA(aLoopIdx, aSub, rIdx, validR);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
```

（注：原 `Init()` 中 `if (td_->rGroupCnt > 0) { gmWs_.SetGlobalBuffer((__gm__ float*)y); } else { gmWs_.SetGlobalBuffer((__gm__ float*)workspace); }` 保持不变——独立 workspace 绑定不可用，见坏实践。）

**after**（终版修复，单文件 25 insertions / 74 deletions；Phase 2 整体改走与 tilingKey=1 NormalTailA 相同的三遍直算，不读 workspace）：

```cpp
    __aicore__ inline void ProcessGroupPhase2TailA(int64_t aLoopIdx)
    {
        // Phase 2 computes final_m / final_s / y directly from GM x via the same
        // 3-pass algorithm as key=1 NormalTailA (ProcessThreePassTailA), rather
        // than merging per-rGroup m_part/s_part from workspace.
        //
        // Rationale (DESIGN-BRANCH-4 §7 精度处理 + §8 workspace):
        //  1. The framework does not reliably pass the dedicated workspace GM_ADDR
        //     to the kernel (verified: using workspace param causes 507035 crash).
        //     Workspace is aliased to y (same as key=3).  Phase 2 reading workspace
        //     (aliased to y) races with Phase 2 CopyOut when aSplitChunkCnt > 1:
        //     early A-chunks' CopyOut overwrites workspace needed by later A-chunks.
        //     This caused case00005/06 catastrophic failure (bf16, aSplitChunkCnt=19).
        //  2. The rescaling trick (final_s = Σ s_part × exp(m_part − final_m))
        //     introduces multi-level accumulation error growing with
        //     rGroupCnt × rLoopsPerGroup, causing case00010/11 precision failure.
        //
        // By computing final_m = max(x) and final_s = Σ exp(x − final_m) directly
        // from GM x (3-pass: max → sum → div+cast), Phase 2 never reads workspace,
        // eliminating both the race and the rescaling error.  The precision matches
        // key=1 (NormalTailA) exactly.  Cost: one extra GM read pass over R in
        // Phase 2 (max pass), but correctness > performance.
        //
        // Phase 1 (R-cores) still writes m_part/s_part to workspace (aliased to y);
        // this is harmless — Phase 2 Pass 3 CopyOut overwrites all of y with the
        // correct output, and SyncAll guarantees Phase 1 completes before Phase 2.
        ProcessThreePassTailA(aLoopIdx, 1);
    }
```

`ProcessThreePassTailA(aStart, aLoops)` 即既有的非分组三遍实现（每个 aSub：`InitMaxCacheTailA` + 全 R `CopyInTailA/CastToFp32TailA/DoReduceMaxTailA/AccumulateMaxTailA` 求 max；`InitSumCacheTailA` + 全 R `DoSubExpTailA/DoReduceSumTailA/AccumulateSumTailA` 求和；再全 R `DoSubExpDivCastTailA + CopyOutTailA` 输出），未改动。Phase 1 保留原样（R 核仍写 workspace[y]）：`SyncAll` 保证其先于 Phase 2 完成，Phase 2 Pass 3 的 CopyOut 会把 y 全量覆盖为正确输出，因此 Phase 1 的写入无害。

## 设计决策

- **为何不修 Init 绑定到独立 workspace**：host 侧 `GetWorkspaceSize` 对 key=3/4 确实计算并返回了非零 userWs（`rGroupCnt × aTotal × sizeof(float) × 2`），但把 kernel 的 `gmWs_` 绑到该 workspace 后 C++ ST 立即大量 `ACL error 507035`（aclrtSynchronizeStream 同步错误，17/40 aclnn-FAIL），回退绑定后全过——aclnn 框架未把独立 workspace 有效传给 kernel，此路不通。
- **为何不做「先读全部 workspace 再统一 CopyOut」的两阶段方案**：需要把所有 aSub 的 final_m/final_s 同时留在 UB。失败形状需 `2 × nColsAlign × aSplitChunkCnt × 4B = 2×112×19×4 = 17024B`，超过 cacheBuf 的 16KB；tmpBuf0/1 在 Pass 3 中被 ReduceSum 占用，不能复用。方案不成立。
- **为何选 Phase 2 三遍直算**：一次改动同时消除 workspace 竞态与 `Σ s_part × exp(m_part − final_m)` 重缩放的多级累加误差；精度与已验证的 key=1 NormalTailA 路径完全一致。代价是 Phase 2 多一遍 GM 读（max 遍），correctness > performance。
- **路线档位**：recommended_fastpath；分组并行的性能收益在 Phase 2 侧被放弃（R 核 Phase 1 的部分积变成白写），属已知代价。

## 系统信息

- SOC：Ascend950PR（TTK SOC 串 `Ascend950PR_9579`），CANN 9.2.0，vendor 部署 `opp/vendors/BenchHarnessSoftmax`。
- 工程布局：`operators/BenchHarnessSoftmax_package/op_kernel/arch35/bench_harness_softmax_kernel.h`（修复文件）；`op_host/arch35/bench_harness_softmax_tiling_arch35.cpp`（tiling，未改）；白盒 `tests/whitebox/branch-4/ttk/`（whitebox_all.csv 24 条 / result.csv / golden_plugin.py）；C++ ST `tests/aclnn/`（test_aclnn.cpp + run.sh）。
- 相关文件清单：读 `bench_harness_softmax_kernel.h`（修复前全文）、`tiling_data.h`、`bench_harness_softmax_tiling_arch35.cpp`、DESIGN-BRANCH-4.md、whitebox_failures.md；写 `bench_harness_softmax_kernel.h`（唯一代码修复）+ `docs/.../LOG.md`（记录）。

## 验证信号

以下信号来自该算子 tilingKey=4 分支白盒失败修复会话（修复 + 验收两轮独立复跑），均为上板实测：

- 修复前白盒全量：total=24 pass=20 fail=4；case00005 mere=1707.4779 / mare=234661.64、case00006 mere=1707.1310 / mare=235470.78（来源平台：950；证据等级：tool_observed）。
- 修复后 C++ aclnn ST 全量：181/181 精度 PASS（Branch-0 46/46 + Branch-1 37/37 + Branch-2 24/24 + Branch-3 34/34 + Branch-4 40/40），无任何分支回归（来源平台：950；证据等级：tool_observed）。
- 修复后 TTK 白盒单例复跑：case00005 mere=1.5228e-5 PASS（1707→1.5e-5）、case00006 mere=1.4933e-5 PASS；此前通过的 case00008 mere=4.559e-5、case00009 mere=4.612e-5 无回归（来源平台：950；证据等级：tool_observed）。
- 验收轮独立全量复跑：total=24 pass=22 fail=2（残留 2 条为 fp32 大 R 求和顺序问题，另卡归因，与本竞态无关）（来源平台：950；证据等级：tool_observed）。
- dtype 交叉对照：case00003（fp16 `[8191,8192]` dim=0）/ case00004（fp32 `[8191,8192]` dim=-2）mere=1.18e-5 PASS，唯 bf16 + 多 A 分块组合灾难性失败（来源平台：950；证据等级：tool_observed）。
- `./build.sh BenchHarnessSoftmax_package` 构建成功退出 0（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 竞态只在 `aSplitChunkCnt > 1` 的形状触发；aSplitChunkCnt=1 时读 workspace 与 CopyOut 天然串行，本修复前后行为一致（该结论由 tiling 参数分析 + key=3 全过推出，key=3 改不改未验证）。
- 本修复依赖「独立 workspace GM_ADDR 不可用」这一 aclnn 调用路径事实；若框架侧修复了 workspace 传递，别名方案本身仍不安全（竞态依旧），但可重新评估两阶段合并方案。
- Phase 2 三遍直算带来一遍额外 GM 读（R 轴全量），对超大 R 用例的耗时影响未做性能对比（本开发会话无 profiling 信号）。
- bf16 输入范围 `(-0.01, 0.01)` 与 `(2049,2047)` 之外的组合未逐一验证；机制（CopyOut 覆盖 workspace）与数值范围无关，推断可迁移但未验证。

**未决项**：

- aclnn 框架为何不把独立 workspace 传给 kernel（507035 的框架侧机制）未追明——见同会话另一张 507035 卡；根因深挖到 opapi `NnopbaseRunWithWorkspace` 传递链是优先接续点。
- Phase 1 在最终方案下成为纯浪费功（R 核写的部分积不再被读）；可评估让 Phase 1 空转或改做他用，未尝试。
- key=3（GroupTailR）路径存在同构竞态隐患（现用例 aSplitChunkCnt=1 未触发），是否同样改直算未验证。
## 坏实践

- **把 gmWs_ 直绑 host 分配的独立 workspace**：host 明明算了 userWs>0，但 kernel 侧绑定后立即 ACL 507035 崩溃（`aclrtSynchronizeStream` 同步错误，#411-415 等 17 条 aclnn-FAIL）；`git stash` 回基线全过证实由该绑定引入。在查清框架 workspace 传递链之前，不要假设「host 分配了 workspace ⇒ kernel 第 3 参可用」。
- **见 mere 千级就往 bf16 Cast/reduction 精度方向查**：本例失败组合（仅 bf16 + 特定 shape）确实指向 dtype 路径，但真正的判别信号是「mere 达到 10^3 量级」——softmax 输出有界 [0,1]，任何数值精度问题都不可能产生千级相对误差，必是数据被覆盖/未算/错轴一类结构性问题。
- **修复后只跑失败用例**：本修复同时改了 group 路径公共逻辑，必须全分支回归（本例 C++ ST 181/181 + 白盒全量 + 此前通过用例单例复跑三层验证）。

## 误诊

- 首轮失败报告初判「疑似 NaN/Inf、归约轴错误或 bf16 Cast/reduction 路径缺陷」——方向过宽；fp16/fp32 同形通过 + bf16 独败的组合应更快收敛到「bf16 专属路径的内存/边界行为」而非数值精度。

## 判别补充（同 trace 白盒失败报告的分流探针）

与修复会话互补的失败组合收敛方法与单例复现：

- **量级判别先行**：softmax 合法输出 ∈ [0,1] 且每条归约轴和为 1，相对误差达千量级只可能是输出根本不是有效 softmax 值（NaN/Inf/归约轴错配/数据被覆盖）——先排除"精度漂移"，按结构性错误排查。
- **OOB/内存字段交叉**：`dyn_oob_result` / `memory_oob_status` 全过 → 排除越界写坏。
- **dtype 交叉对照**：同形/邻形 fp16、fp32 用例通过（case00003 fp16 `[8191,8192]` dim=0 与 case00004 fp32 `[8191,8192]` dim=-2 均 mere=1.18e-5 PASS）→ 嫌疑收敛到 dtype×shape 边界联合条件（本例 bf16 + A=2047 非对齐 + 多 A 分块），单变量排除不充分。
- **dim 折叠对照**：rank=2 下 dim=-2 与 dim=0 等价，两例 mere 几乎相同 → 同根因，排除 dim 解析问题。
- **单例复现命令**（TTK kernel 模式；`--ti` 为 1-based 索引，先按 `testcase_name` 在 CSV 中定位行号；`<OUTDIR>` = 白盒产物目录，`$TTK`/`$PY`/`$TBE` 取自 env.sh）：

```
cd $TTK
env PYTHONPATH="$TTK:<OUTDIR>:$ASCEND_TOOLKIT_HOME/python/site-packages:$TBE" \
    TTK_FULL_SOC_VERSION=Ascend950PR_9579 \
    LD_LIBRARY_PATH="<vendor>/op_api/lib:$LD_LIBRARY_PATH" \
    $PY -m ttk kernel -i <OUTDIR>/whitebox_all.csv --ti=<N> --pc=1 \
    --plugin <OUTDIR>/golden_plugin.py -o <OUTDIR>/repro_<case>.csv
```