---
type: "Runbook"
title: "2 字节逐元素 GM RMW 大规模丢写判别（mare≈1）"
description: "2 字节元素（fp16/bf16）的逐元素 GM 读-改-写在超大规模下偶发丢写，特征是 mere 极小而 mare≈1；用失配分类探针区分丢写与舍入，规避方法是改走 4 字节 RMW。R2 实证：每元素三事件、跨批/每元素 PipeBarrier、S-pipe 直写四条同步路线全部无效（~2M 次重复仅 512 次生效，指标逐位不变）。"
tags: [task.debugging, topic.precision, topic.data_movement.memory_access, topic.compute.indexing]
status: stable
sources: []
created_at: '2026-09-14T02:36:53Z'
updated_at: '2026-09-14T12:20:54Z'
platforms: ["950"]
---

# 2 字节逐元素 GM RMW 大规模丢写判别（mare≈1）
## 现象

在 Ascend950PR（CANN 9.2.0）上为 BenchHarnessScatter（scatter-reduce 算子）修黑盒失败用例时，bf16 add 超大用例（1D 33,582,839 元素，data 全部为 -1.1754943508e-38 = -2^-126，16,791,419 个 updates 值域 (-1,-0.01)，dim=0）以"逐元素 GM 读-改-写"实现 reduce：每个 update 用 MTE2 DataCopyPad 读 `y[target]`（blockLen=2 字节）、S-pipe 计算、MTE3 DataCopyPad 写回，元素间用 MTE2_S / S_MTE3 / MTE3_MTE2 事件同步。TTK 指标呈现强烈反差：

- mere=7.752292731311172e-06（平均相对误差极小，绝大多数位置正确）
- mare=0.9997577667236328（最大相对误差 ≈100%），threshold=0.0078125，FAIL（`reason: 'threshold exceeded: mare(1.00e+00>=0.078125)'`）

而同一实现的 2M 规模自建复现（1D、一致 golden）却显示：mism=33129/2000000、**lostUpd=0**、rounding=33129、maxRel=0.0101——2M 规模没有任何丢写，只有小幅舍入差异。

## 适用场景

- 工程形态：aclnn 自定义算子 kernel 中对 GM 做逐元素（blockLen=元素宽度）DataCopyPad 读-改-写的 reduce/scatter 路径，TTK 黑盒精度评测。
- API 路径：`AscendC::DataCopyPad`（MTE2 读 / MTE3 写，`DataCopyExtParams.blockLen = sizeof(T)`）+ `SetFlag/WaitFlag`（MTE2_S / S_MTE3 / MTE3_MTE2）、`PipeBarrier<PIPE_ALL>`、S-pipe `GetValue/SetValue`。
- 验证过的平台与 CANN 版本：Ascend950PR（arch3510），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——所有以 2 字节元素逐元素 GM RMW 实现累加/选择语义的算子（scatter-reduce / index-put 等）。
- 路线优先级：recommended_fastpath（判别方法本身；其推荐的规避路线见下）。
- 明确不适用 / 未验证场景：
  - 4 字节（int32/fp32/int64 拆两次 4 字节）元素：同算子 int32 add 34,125,328 元素、fp32 大 shape 用例全部 PASS，未观察到丢写。
  - 批量搬运（blockLen 为整块）路径：本卡只针对逐元素 RMW。
  - 覆盖（update）模式：MTE3 写 FIFO 序即 golden 语义，无累加依赖。
  - 2M 以下规模：实测无丢写，无需处理。
  - 其他 NPU 架构 / CANN 版本：未验证。

## 判别方法

**触发**：2 字节 dtype 的 reduce 类精度失败，且指标形态为"mere 极小 + mare 接近 1"，或自建复现在中规模全绿、只在超大 shape 失败。

**探针**（自建 aclnn 诊断程序 `tests/aclnn/repro_diag.cpp` TEST B 的逐字失配分类循环——bf16 add、N=2,000,000，host golden 为 fp32 累加单次舍回，与官方 golden 同 dtype 策略；`out[i]==data[i]` 即"该位置的 update 整个丢失"分类为 LOSTUPD）：

```cpp
        int64_t mism=0, lostUpd=0, rounding=0; double maxRel=0;
        for(int64_t i=0;i<N;i++){
            if(out[i]!=gold[i]){
                mism++;
                float g=bf16_to_f32(gold[i]), a=bf16_to_f32(out[i]);
                if(out[i]==data[i]) lostUpd++; else rounding++;
                double rel=std::abs(a-g)/(std::abs(g)+1e-7f); if(rel>maxRel)maxRel=rel;
                if(mism<=8) fprintf(stderr,"  [B %ld] got=%g(0x%04X) gold=%g(0x%04X) data=%g rel=%.4g%s\n",
                    i,a,out[i],g,gold[i],bf16_to_f32(data[i]),rel, out[i]==data[i]?" LOSTUPD":"");
            }
        }
```

判别算式：
- 舍入上界：bf16 1 ulp 相对步长 2^-8≈0.39%，k 步逐步舍入漂移量级 k×ulp（本例 data≈-2^-126 对 sum 无贡献，golden≈u）。舍入类失配的 rel 上界约百分之几，**不可能达到 0.9998**。
- 丢写特征值：位置被 1 个 update 命中时 golden=bf16(u)≈-0.5，kernel 丢写后保持 data=-2^-126（或 0），rel = |−2^-126−(−0.5)|/(0.5+1e-7) ≈ 1.0。
- 因此 mere 极小 + mare≈1 ⇒ 极少数位置发生整 update 丢失（丢写），而非舍入分歧；lostUpd 计数为 0 而指标仍 mare≈1 时，说明丢写只出现在自建复现未覆盖的更大规模。

**观测**（本会话实测数据）：
- bf16 add 33.6M（TTK，逐元素 2 字节 RMW）：mere=7.75e-6、mare=0.99976。
- bf16 add 2M（自建复现，同实现）：lostUpd=0、rounding=33129、maxRel=0.0101——规模以下无丢写。
- int32 add 34,125,328 元素（同逐元素 RMW 实现，4 字节）：PASS。
- fp16/bf16 amax 116,792,947 元素（2 字节逐元素 RMW，选择语义）：PASS。
- 改走 fp32 workspace（4 字节 RMW + fp32 累加）后：bf16 add 33.6M 用例 PASS（精度模型对齐见 [FP16 scatter 归约逐次回写放大相消误差卡](low_precision_scatter_reduction_needs_fp32_accumulator.md)）。

**解释**：2 字节元素的逐元素 GM RMW 存在规模相关的偶发丢写（本例 2M 无、33.6M 出现，mere 表明仅极少数位置受影响）；同一算子的 4 字节逐元素 RMW 在 34M+ 全 PASS，说明危害与元素宽度绑定。推测机制是 DataCopyPad 以 2 字节 blockLen 写 GM 时按 32 字节块做读-改-写，同块内相邻目标的连续 RMW 在极端规模下竞争（未验证，见遗留问题）。规避：把累加路径换到 4 字节载体（如 fp32 workspace）上做逐元素 RMW，或按 target 分组批量归约（未尝试）。

### R2 尝试与失败：四条同步路线全部无效（banned 路线实证）

同一用例在姊妹会话（kernel 已单核、逐元素"GM 读→S 归约→GM 写"）上先后叠加四种同步方案，指标**逐位不变**（mare=1.0，dump 热点位置值恒为 -256）：随机索引下两个热点位置各被约 200 万次更新命中，期望 ≈ -1056768（约 2M 次 × 平均 -0.5 的累加），实际 -256 ≈ 512 次 × -0.5——**约 2M 次更新只生效了 512 次**：

```
Index: 000 RealIndex: 000000 Expected: -1056768.000000000000000000 Actual: -256.000000000000000000 Diff: -1.000000
Index: 001 RealIndex: 33582838 Expected: -1056768.000000000000000000 Actual: -256.000000000000000000 Diff: -1.000000
```

**方案 A：每元素三事件同步（MTE2_S / S_MTE3 / MTE3_MTE2，verbatim 取自会话终态版本）**

```cpp
    } else {
        // Reduce: per-element MTE2 read → S reduce → MTE3 write.
        for (int64_t i = 0; i < aISeg; i++) {
            I idxVal = idxBuf.GetValue(i);
            int64_t flatPos = chunkBase + i;
            UnravelFlatToMultiIndex(flatPos, td_->indicesShape, td_->rank, multiIdx);
            int64_t targetOff = CalcTargetOffset(multiIdx, td_->dim, idxVal,
                                                   td_->dataStrides, td_->rank);
            // MTE2: read y[targetOff] → B2[0]
            AscendC::DataCopyPad(yBuf, gmY_[targetOff], p1, pad1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evMTE2_S_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evMTE2_S_);
            // S-pipe: reduce B2[0] with B1[i]
            T oldVal = yBuf.GetValue(0);
            T updVal = updBuf.GetValue(i);
            T newVal = ReduceScalar<T>(td_->reduceMode, oldVal, updVal);
            yBuf.SetValue(0, newVal);
            // S→MTE3 sync: B2[0] written by S-pipe visible to MTE3
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(evS_MTE3_);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(evS_MTE3_);
            // MTE3: write B2[0] → y[targetOff]
            AscendC::DataCopyPad(gmY_[targetOff], yBuf, p1);
            // MTE3→MTE2 sync: B2 reusable + GM write visible to next MTE2 read
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3_MTE2_);
        }
    }
```

**方案 B：跨批 PipeBarrier（批内事件 + 批间全屏障）**

```cpp
        ScatterChunk(chunkBase, aISeg);
        // Cross-batch GM coherency: ensure this batch's MTE3 writes are fully
        // committed to GM before the next batch's MTE2 reads (duplicate indices
        // spanning batch boundaries).  Per-element MTE3_MTE2 event within a batch
        // handles same-batch duplicates; this handles cross-batch duplicates.
        if (flat < totalTiles - 1) AscendC::PipeBarrier<PIPE_ALL>();
```

**方案 C：每元素 3×PipeBarrier<PIPE_ALL>（verbatim）**

```cpp
    } else {
        // Reduce: per-element MTE2 read → S reduce → MTE3 write, synced via
        // PipeBarrier<PIPE_ALL> (simplest correct sync; avoids event/barrier
        // interference).  Each barrier ensures the prior pipe's GM/UB access
        // completes before the next pipe starts.
        for (int64_t i = 0; i < aISeg; i++) {
            I idxVal = idxBuf.GetValue(i);
            int64_t flatPos = chunkBase + i;
            UnravelFlatToMultiIndex(flatPos, td_->indicesShape, td_->rank, multiIdx);
            int64_t targetOff = CalcTargetOffset(multiIdx, td_->dim, idxVal,
                                                   td_->dataStrides, td_->rank);
            // MTE2: read y[targetOff] → B2[0]
            AscendC::DataCopyPad(yBuf, gmY_[targetOff], p1, pad1);
            AscendC::PipeBarrier<PIPE_ALL>();   // MTE2 done → S-pipe can read B2
            // S-pipe: reduce B2[0] with B1[i]
            T oldVal = yBuf.GetValue(0);
            T updVal = updBuf.GetValue(i);
            T newVal = ReduceScalar<T>(td_->reduceMode, oldVal, updVal);
            yBuf.SetValue(0, newVal);
            AscendC::PipeBarrier<PIPE_ALL>();   // S done → MTE3 can read B2
            // MTE3: write B2[0] → y[targetOff]
            AscendC::DataCopyPad(gmY_[targetOff], yBuf, p1);
            AscendC::PipeBarrier<PIPE_ALL>();   // MTE3 done → next iter can read GM
        }
    }
```

**方案 D：S-pipe 直写（GetValue/SetValue + 每元素 PipeBarrier，方案 C 之前的早期版本）**——同样无效。

每轮修改后重编译、单用例重跑，输出指标（mare=1.0）与修改前逐位一致。结论：丢更新不发生在"指令顺序"层面（否则屏障应有效），疑似 DMA/写缓冲对 2 字节写的合并（coalescing）导致后续写覆盖未落地的中间值，或按固定粒度聚合丢失；确切微架构机制未验证。**对 bf16/fp16 大规模重复目标走"2 字节逐元素 GM RMW + 任何已试同步原语"的组合标记为 banned**。

## 设计决策

- 判别顺序选择"先分类失配、再下结论"而不是直接改实现：丢写与舍入两类根因的修法完全不同（前者要换载体/加同步，后者要对齐精度模型），lostUpd/rounding 分类是零成本的第一手证据。
- 规避路线选择 4 字节 fp32 workspace 而非继续在 2 字节路径上加强同步：MTE3_MTE2 事件在本实现里逐元素都已存在（见现象中的同步清单），仍出现规模性丢写；而 4 字节 RMW 有同算子 34M+ 全 PASS 的实证，代价只是额外 workspace。
- 当前方案路线档位：recommended_fastpath（4 字节载体 RMW）；原 2 字节逐元素 RMW + 逐元素事件同步在该场景下不可靠。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0；TTK aclnn 评测（--pc=1）。
- 相关文件：`op_kernel/arch35/bench_harness_scatter_kernel.h`（`ScatterChunk` reduce 分支＝受影响的 2 字节逐元素 RMW 实现；fp32 路径＝规避实现）、`tests/aclnn/repro_diag.cpp`（失配分类探针）。

## 预防

- 2 字节 dtype 的 reduce 语义优先考虑 4 字节载体（fp32 workspace / 预转换）而不是直接在原生 dtype 上逐元素 RMW；确需 2 字节 RMW 时，评测必须覆盖目标最大规模（本例 2M 全绿、33.6M 才暴露）。
- 精度失败先看 mere/mare 的形态组合再定位：mere 极小 + mare≈1（超出 dtype 舍入上界）⇒ 怀疑丢写/写错位置；mere 与 mare 同量级小幅超限 ⇒ 怀疑舍入模型。
- 诊断程序按 `actual==data` 分类丢写，能把"丢写 vs 舍入"在分钟级分开。

- 与多核非对齐写回边界覆盖区分：本卡是单 kernel 逐元素 RMW 的规模相关丢写（2M 无、33.6M 出现），不是跨核边界竞态（见 [多核非对齐 DataCopyPad 写回边界互相覆盖卡](multicore_non_aligned_datacopypad_rmw_overlap.md)）；若失配集中在相邻核共享的 32B 边界位置，优先怀疑跨核覆盖而非本卡机制。
- 多核把 indices 分片做非原子 RMW 是另一独立丢更新根因（重复索引互相覆盖），判别与修复见 [多核 scatter 同目标非原子 RMW 丢更新卡](multicore_nonatomic_rmw_same_target_lost_updates.md)。

## 验证信号

以下信号来自该算子 blackbox-fix 会话的 TTK 指标与自建 aclnn 复现（bf16 add 用例及其对照）：

- 2 字节逐元素 GM RMW（bf16 add 33.6M，TTK）：mere=7.75e-6、mare=0.99976 超限 FAIL（来源平台：950；证据等级：tool_observed）。
- 同实现 2M 规模（自建复现）：mism=33129/2000000、lostUpd=0、rounding=33129、maxRel=0.0101——无丢写、只有舍入（来源平台：950；证据等级：tool_observed）。
- 4 字节对照（int32 add 34,125,328 元素，同逐元素 RMW）：TTK PASS（来源平台：950；证据等级：tool_observed）。
- 2 字节选择语义对照（fp16/bf16 amax 116,792,947 元素）：TTK PASS（是否被容差掩盖未验证）（来源平台：950；证据等级：tool_observed）。
- 规避生效（bf16 add 33.6M 改走 fp32 workspace 4 字节 RMW）：TTK PASS（来源平台：950；证据等级：tool_observed）。

- 独立复现 R2（姊妹会话，同步路线证伪）：bf16 add 16.79M 索引、热点位置 ~2M 次更新，expected -1056768 vs actual -256（≈512/2M 生效），mare=1.0；方案 A/B/C/D 依次上板，mare=1.0 逐位不变、dump 值恒为 -256（来源平台：950；证据等级：tool_observed）。
- 独立复现 R2 对照：int32 10000 重复单批 add 0/10000 失配（万级 4 字节重复正确）；fp16 单元素块写 0/32 失配（排除 DMA burst 越界写腐蚀邻居）；受控 564,480 元素 fp16 amax -Inf（含次正规全谱 data）0 失配（小规模 2 字节读写正确）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- "丢写"结论由 mere/mare 形态 + 舍入上界算式 + 规模对照推断，未在 33.6M 规模上直接抓到单次丢写现场；若复现他算子，建议先跑失配分类探针确认 actual==data 形态再迁移结论。
- amax/amin 2 字节大 shape PASS 可能是丢写被容差掩盖（丢失的 update 常非最大/最小值），不能反证无丢写（未验证）。
- 仅在 950 + CANN 9.2.0 实测；其他架构/版本未验证，推断机制（元素宽度绑定）可迁移但阈值规模会不同（未验证，推断可迁移）。
- R2 的"512 次生效"是否稳定（换 seed/规模是否仍是 512）未验证；与批量参数（aI=10920 或 21840）的关系未验证。
- 本卡不解决舍入模型对齐问题（见同批 fp32 累加卡）；两类根因常叠加出现。

**未决项**：

- 优先项：复现并追根因到机制层——在 33.6M 规模直接抓到丢写现场（dump actual/golden/data 三方对比），并验证"DataCopyPad 2 字节 blockLen 写 GM 按 32 字节块 RMW、同块竞争"的具体微架构机制；R2 建议构造可调规模/可调字节数的最小复现（单目标 N 次更新，N 从 10^3 扫到 2×10^6），定位丢更新开始的规模阈值与 512 的来源（DMA 写合并缓冲？L2 行为？）。
- 丢写概率随规模的定量关系（2M 无、33.6M 出现）未测量；不确定是否存在安全规模阈值。
- 按 target 分组批量归约的替代实现未尝试。
## 坏实践

- 看到 mare≈1 就归因"精度不够"去放宽容差或改 dtype——丢写不是容差问题。
- 只用中规模自建复现验证 2 字节 RMW 路径就认为安全（本例 2M 全绿、33.6M 翻车）。
- 在 2 字节逐元素 RMW 上靠不断加密同步事件"撞"通过——本实现的逐元素 MTE3_MTE2 同步已属最重，仍不解决；R2 进一步实证每元素三事件、跨批/每元素 PipeBarrier、S-pipe 直写四条路线全部无效，浪费多轮 17 分钟级全量回归。
- 用万级重复的自建用例代表百万级真实负载（R2：int32 万级对照 PASS 而 bf16 真实用例是百万级重复）。

## 误诊

- 自建复现首版的"lostUpd=154965/244000（63% 丢写）"假象：host golden 按 2D 语义算 target 而给 kernel 的是 1D 张量，两侧 target 不一致制造了假丢失更新；修正 golden 语义后 lostUpd=0。教训：丢写结论必须建立在 golden 语义与 kernel 输入完全一致的前提下。
- 曾据"TTK mare 与修复前逐位相同"怀疑新代码未生效/内核缓存过期；实际是另一独立根因（FloatToHalf 次正规阈值，见 [FloatToHalf 次正规阈值漏 exp102 卡](floattohalf_subnormal_threshold_exp102.md)）叠加，新路径确实在运行。
- R2 早期把该用例与 fp16 amax 的次正规失配混为一谈，按"GM 写可见性"统一加屏障——后者实为转换编码缺陷，屏障方向对两者都无效；"DMA burst 越界写"假设被邻接对照实验直接否定。
