---
type: "Runbook"
title: "MTE2占比高≠带宽bound：发射开销判别"
description: "逐行 256B NdDma 的融合 kernel 相对非融合五 kernel 链约 2.43x 慢；用 msprof op_summary PipeUtilization 算\"每描述符耗时/有效带宽\"，判定为 MTE2 发射/描述符开销 bound 而非带宽 bound，据此选 DMA 合并杠杆——只看泳道占比会误判为带宽问题。"
tags: ["task.optimization", "topic.data_movement.data_copy"]
status: stable
sources: []
created_at: "2026-09-14T12:14:53Z"
updated_at: "2026-09-14T12:14:53Z"
platforms: ["950"]
---

# MTE2占比高≠带宽bound：发射开销判别

## 现象

在 Ascend950PR（AIV=56、UB 253952B、L2 128MiB、AICore 额定 1650MHz，CANN 9.2.0）上为融合算子 FusedRmsNormSigmoidMul（bf16，x=(128,6,128)，gamma=(128,)，other 同形，RMSNorm 归一化后乘 Sigmoid(other)）做性能优化：先把注册工程转为 `<<<>>>` kernel 直调基线，再按采集协议（10 样本取中位、aclnnArgMax 256MiB fp16 框架算子预热稳频、warmup 不计入）采集基线与非融合链参考。结果出现反直觉现象——**融合单 kernel 比非融合五 kernel 链还慢**：同协议 10 样本中位下，融合基线（block_num=24）相对链参考（RmsNorm→Cast→Sigmoid→Cast→Mul 五 kernel 逐组和，block_num 55+48+56+30）约 2.43x 慢；链内目标为链参考/1.2，基线距目标需 2.92x 提升。基线 kernel 的搬运形态是"逐行小 DMA"：每行 R=128 个 bf16 = 256B 一条 NdDma 5D 描述符，reduce 段每核 32 条、post 段每核 32×3 条（x/gamma/other 各一），合计每核 128 条 MTE2 描述符 + 32 条 MTE3。

## 适用场景

- 工程形态：AIV 向量 kernel 的性能分析与优化选型（行归约 + 逐元素融合类算子；msprof 采集 + op_summary 指标解读）。
- API 路径：msprof `--aic-metrics=PipeUtilization` → `op_summary_*.csv` 的 `aiv_*_time/aiv_*_ratio` 列；kernel 侧 NdDma 5D `DataCopy` / `DataCopyPad` / `asc_vf_call`。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：platform_wide——"发射开销 vs 带宽"判别方法适用于该平台所有 AIV kernel；具体派生数值仅对本 kernel 形态成立。
- 路线优先级：recommended_fastpath（判别先行，再选 DMA 合并/常驻/复用杠杆；避免盲目做带宽优化）。
- 明确不适用 / 未验证场景：无 msprof op_summary 数据时（只有端到端时延）无法做本判别；MTE3/vec 主导的 kernel 需换对应泳道分析。
- 精度影响：本卡为诊断规则，未改动任何代码，无精度回归对象（未验证精度影响不适用——无代码变更）。

## 判别方法

对 op_summary 的目标 kernel 行做泳道占比 + 每描述符耗时 + 有效带宽三层推导（区别于"只看哪个 ratio 大"）：

1. **泳道占比**（10 样本）：MTE2 ratio 0.769–0.780（中位 0.776）绝对主导；scalar 0.319–0.335（中位 0.333）次级；vec 0.105、MTE3 0.094；icache miss 0.005 可忽略。重叠因子 `Σratio ≈ 1.308`，理想上界 `Σ/max ≈ 1.686`，达成率 77.6%——单泳道（MTE2）>0.75 显著占优，属访存 bound 路由。
2. **每描述符耗时**：由代码读出每核 MTE2 描述符数 128 条（reduce 32 + post 32×3）× 256B；"MTE2 泳道时间 / 128" 得到的每描述符耗时，比 256B 数据在 GB/s 级带宽下的传输时间高出一个量级以上——时间主要花在描述符发射/装配而非数据传输。
3. **有效带宽对照**：由泳道时间折算的单核与 24 核聚合有效带宽，距平台 HBM 能力低 1~2 个数量级 → **MTE2 泳道时间 ≈ 纯发射开销，不是带宽瓶颈**。
4. **roofline 校验**（判"修得好不好"）：由 unique 流量与链内目标（链参考/1.2）折算所需聚合有效带宽仅为当前有效带宽的约 1.7x——消除逐条发射开销后（描述符 128→每核 ≤4 条、单条 4–8KiB 连续块）结构上可达，说明**主要障碍是搬运组织方式，不是平台能力**。

探针（op_summary PipeUtilization 提取，verbatim）：

```bash
cd "<PROF_DIR>/mindstudio_profiler_output" && python3 -c "
import csv
rows = list(csv.DictReader(open('op_summary_*.csv')))
tgt = [r for r in rows if 'fused_rms_norm' in r['Op Name'] or 'fused_rms_norm' in r.get('Kernel Name','')]
print('total rows:', len(rows), 'target rows:', len(tgt))
if tgt:
    keys = ['Op Name','Task Duration(us)','Block Num','aiv_time(us)','aiv_total_cycles','aiv_vec_time(us)','aiv_vec_ratio','aiv_scalar_time(us)','aiv_scalar_ratio','aiv_mte2_time(us)','aiv_mte2_ratio','aiv_mte3_time(us)','aiv_mte3_ratio','aiv_icache_miss_rate']
    for i, r in enumerate(tgt):
        print('--- sample', i)
        for k in keys:
            print(f'  {k} = {r.get(k)}')
"
```

被判别的代码形态（基线 kernel 的逐行搬运，verbatim 节选——`CopyInPost` 每 post 行发 3 条逐行搬运，`CopyLogical` 每次一条 256B NdDma 5D 描述符）：

```cpp
    __aicore__ inline void CopyInPost(int64_t a, int64_t rOffset, uint32_t count)
    {
        CopyLogical(buf0_.Get<bfloat16_t>(), xGm_, a * td_->rTotal + rOffset,
            count, td_->xShape, td_->xStride, td_->xStorageOffset);
        CopyLogical(buf1_.Get<bfloat16_t>(), gammaGm_, rOffset, count,
            td_->gammaShape, td_->gammaStride, td_->gammaStorageOffset);
        CopyOther(buf2_.Get<bfloat16_t>(), a * td_->rTotal + rOffset, count);
    }
```

```cpp
    __aicore__ inline void CopyLogical(LocalTensor<bfloat16_t> dst,
        GlobalTensor<bfloat16_t> src, uint64_t logicalBegin, uint32_t count,
        const int64_t* shape, const int64_t* stride, int64_t storageOffset)
    {
        uint32_t done = 0;
        while (done < count) {
            uint64_t flat = logicalBegin + done;
            uint64_t coord[8] = {};
            for (int32_t d = 7; d >= 0; --d) {
                coord[d] = flat % static_cast<uint64_t>(shape[d]);
                flat /= static_cast<uint64_t>(shape[d]);
            }
            uint32_t seg = (count - done < static_cast<uint32_t>(shape[7] - coord[7]) ? count - done : static_cast<uint32_t>(shape[7] - coord[7]));
            uint64_t srcOffset = 0;
            (void)storageOffset;
            for (int32_t d = 0; d < 8; ++d) {
                srcOffset += coord[d] * static_cast<uint64_t>(stride[d]);
            }
            NdDmaLoopInfo<5> info;
            for (uint32_t d = 0; d < 5; ++d) {
                info.loopLpSize[d] = 0;
                info.loopRpSize[d] = 0;
                info.loopSize[d] = 1;
                info.loopSrcStride[d] = 0;
                info.loopDstStride[d] = 0;
            }
            info.loopSize[0] = seg;
            info.loopSrcStride[0] = stride[7];
            info.loopDstStride[0] = 1;
            NdDmaDci();
            NdDmaParams<bfloat16_t, 5> params = {info, 0};
            DataCopy<bfloat16_t, 5>(dst[done], src[srcOffset], params);
            done += seg;
        }
    }
```

（同形态的次级问题：gamma 每核被逐行重复搬入 32 次——全 launch 768 次搬入中 744 次冗余，L2 兜底不产生 HBM 流量但全额消耗 MTE2 发射与标量装配；x 在 reduce 与 post 两段各读一遍；逐行 `SetFlag/WaitFlag` 焊点使 MTE2→V→MTE3 深度 1 串行；tiling 用魔数 tileLimit 解出 au=32 → 24/56 核欠填。）

## 解释（结论）

MTE2 泳道占比高 ≠ 带宽瓶颈：把"泳道时间"除以"该泳道发出的描述符数"得到每条开销，再折算有效带宽与平台能力对照（本例低 1~2 个数量级），即可把"带宽 bound"与"发射/描述符开销 bound"区分开。前者要减流量/换布局，后者要**合并搬运**（tile 级单条连续 DMA）、**不变量常驻**（gamma 一次载入 UB）与**消除二次读**（x tile 复用）。据此制定的优化杠杆排序：L1 搬运重组（描述符 128→≤4 条/核、gamma 32→1 次、x 读 2→1 遍）> L2 tiling 填核（au 重解，24→48~56 核）> L3 计算/标量通路（VF 调用收拢、常量外提）。

## 设计决策

- 判别顺序定为"泳道占比 → 每描述符耗时 → 有效带宽 → roofline"：只看占比会把本例误判为带宽问题（0.776 的 MTE2 占比很像搬不动）；每描述符耗时是区分发射开销与数据时间的钥匙。
- 每核描述符数从代码静态读出（128 条/核）而非估算——判别精度依赖该计数，代码与采集身份（逐文件 SHA + 二进制 SHA）已先行核验。
- 历史归因交叉验证（同 kernel 上一版本 CANN 采集）：MTE2 0.79→0.776 一致、scalar 0.51→0.333 下降、vec/MTE3 基本不变——MTE2 主导结论被独立采集确认。
- 本卡只产出"判别 + 杠杆排序"；三条候选方案（tile 级合并搬运快路 / VF 收拢加严变体 / host-only tiling 再平衡）的实施与修后实测分别落在 [tile_merged_dma_eliminate_mte2_issue_bound](tile_merged_dma_eliminate_mte2_issue_bound.md)、[vf_lane_accumulate_row_loop_consolidation](vf_lane_accumulate_row_loop_consolidation.md)、[host_tiling_magic_au_cap_ub_equation_rebalance](host_tiling_magic_au_cap_ub_equation_rebalance.md) 三卡，不在本卡闭环内。

## 系统信息

- 平台：Ascend950PR_9579，CANN 9.2.0；AIV=56、UB=253952B、L2=128MiB、AICore 额定 1650MHz（来源 950PR_957x.ini）。
- 采集协议：msprof `--aic-metrics=PipeUtilization`；collector 10 样本取中位、warmup（aclnnArgMax 256MiB fp16 + Cast）不计入；目标 kernel 以 Itanium mangled 符号（`_Z26fused_rms_norm_sigmoid_mul...`）在 op_summary 中匹配，Block Num=24 与 host tiling usedCoreNum 一致。
- 被分析代码：直调基线 `csrc/op_kernel/kernel/arch35/fused_rms_norm_sigmoid_mul_kernel.h`（BaseKernel：CopyLogical/CopyOther/CopyInReduce/CopyInPost/ProcessTile/ProcessPost）、host tiling（tileLimit=4096 元素魔数 → au=32 → blockDim=24）。

## 验证信号

以下信号来自同一次性能优化会话的基线采集与分析（单轮采集，10 样本，Ascend950PR / CANN 9.2.0）：

- 同协议 10 样本中位：融合基线（block_num=24，采集 validation 30/30 映射通过）相对非融合五 kernel 链参考（block_num 55+48+56+30）约 2.43x 慢；距链内目标（链参考/1.2）需 2.92x 提升（来源平台：950；证据等级：tool_observed）。
- op_summary PipeUtilization（10 样本中位）：MTE2 ratio 0.776、scalar 0.333、vec 0.105、MTE3 0.094；`aiv_total_cycles` 按核数与额定频率折算与 `aiv_time` 自洽（排除频率读错/样本错位类假信号）（来源平台：950；证据等级：tool_observed）。
- 派生判别量：每 MTE2 描述符耗时（MTE2 泳道时间/每核 128 条）远大于 256B 数据在 GB/s 级带宽下的传输时间；折算单核/24 核聚合有效带宽距平台 HBM 能力低 1~2 个数量级 → 发射/描述符开销 bound（来源平台：950；证据等级：tool_observed，派生自上述实测值）。
- roofline：达到链内目标（链参考/1.2）所需聚合有效带宽约为当前有效带宽的 1.7x，消除逐条发射后结构上可达（来源平台：950；证据等级：tool_observed 派生）。

## 性能影响

本卡为诊断（判别规则），无修后对比。相对代价表述：融合基线相对非融合五 kernel 链参考（参照路线：RmsNorm+Cast+Sigmoid+Cast+Mul 链参考，来源平台：950）约 2.43x 慢；该差距的主构成被判定为 MTE2 逐条发射开销而非算力或带宽上限——修后对比见后续优化轮次三卡（本卡未含）。

## 失效边界

- "每描述符耗时"推导依赖从代码静态数出每核描述符数（本例 128 条）；代码改动后需重数，否则判别失真。
- 每描述符耗时与折算带宽等派生数值绑定本 kernel 形态（256B/条 NdDma 5D、24 核）；其它块大小/核数的 kernel 需按同方法重算，不可套用数值。
- 单轮采集（10 样本）为 single_run 口径；scalar 占比与历史参考（0.51）存在差异，方向性结论（MTE2 主导）有一致性佐证但幅度未跨轮复现。
- 优化方案（合并搬运快路/VF 收拢/host tiling 再平衡）的实施效果不在本卡验证范围。
- 泳道占比判 bound 的朴素读法（"占比最高者即 bound"）在不做每描述符/带宽推导时会把本例误判为带宽瓶颈——同平台"理论 bound 落空"的另一实证见 [theory_bound_missing_scalar_pipe_candidate](theory_bound_missing_scalar_pipe_candidate.md)。

**未决项**：

- 三条优化方案（S1 tile 级合并搬运快路 / S2 S1+VF 收拢 / S3 host-only tiling 再平衡）已契约化派发，实施与修后 perf/精度回归为下一步（本卡切片内未完成，不能宣称收益；后续轮次已完成，见三张优化卡）。
- launch 参数拷贝开销因无实测占比证据被 deferred，待后续轮次量化。
## 坏实践

- 只看 `aiv_mte2_ratio` 高就下"带宽瓶颈、要减数据量"的结论——本例减流量空间很小（unique 流量仅 576KiB），真正杠杆是合并描述符。
- 用 task_time_*.csv 找 kernel 级数据（该文件常只有 PLACE_HOLDER_SQE/N/A 行）；kernel 级时延与 PipeUtilization 在 op_summary_*.csv。
- 采集前后不核对代码/二进制身份（SHA）与 block_num 一致性——无法排除"测的不是这份代码"。

## 误诊

- 无（本例判别链一次到位；记录一个易混点：`aiv_total_cycles/核数 @额定频率` 与 `aiv_time` 的自洽校验可排除"频率读错/样本错位"类假信号）。
