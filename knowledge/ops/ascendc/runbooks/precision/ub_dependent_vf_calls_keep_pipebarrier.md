---
type: "Runbook"
title: "有数据依赖的 VF 调用间仍需 PipeBarrier"
description: "同步文档声明 arch35/Ascend 950 连续 VF 之间硬件保证顺序、不需要 PipeBarrier；在'前 VF StoreAlign 写出的 UB 是后 VF LoadAlign 读入源'的依赖形态上被单变量证伪——仅移除依赖 VF 对间 PipeBarrier 后出现小比例非全零 bitwise 失配（F2/E1/E8）。950+CANN 9.2.0 实测依赖 VF 对之间屏障不可删。"
tags: [task.debugging, topic.implementation.synchronization]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# 有数据依赖的 VF 调用间仍需 PipeBarrier

## 现象

在为 Ascend950PR（CANN 9.2.0，arch35）开发的 Ascend C 融合算子 kernel 中，每个 `asc_vf_call` 之后都插有 `AscendC::PipeBarrier<PIPE_ALL>()`。仓库内同步文档（sync-and-consistency §2.3/§2.5）声明："arch35/Ascend 950 上连续 VF 之间硬件保证顺序，不需要 PipeBarrier"、"PipeBarrier<PIPE_ALL>() 会等待所有流水线完成，对性能影响大"。按该声明做了单变量修改：仅移除分块搬运宏（CHUNKED_*）、门控计算链（GateSafeAbsSqrt→GateSignMul→Sigmoid）与卷积循环（MulRow→AddRow）中 VF 之间的 `PipeBarrier<PIPE_ALL>`，保留分块粒度（每调用 64 元素）、手动计数递减与全部跨流水线 barrier。重新构建后 23 个 ST 用例从 16/23 跌到 13/23：新增 3 个用例出现**小比例、非全零**的 bitwise 失配——F2_ReduceD1 8/8、E1_NaNInject 6/64、E8_InfInject 7/64 元素不一致，典型"消费者读到旧/部分数据"形态。同时证明 507035 崩溃与 barrier 数量无关（该实验中崩溃用例依旧崩溃）。

## 适用场景

- 工程形态：Ascend C regbase 范式 kernel（`__simd_vf__` + `asc_vf_call`），ACLNN 算子仓工程。
- API 路径：`asc_vf_call`、`AscendC::PipeBarrier<PIPE_ALL>`、`AscendC::Reg::LoadAlign/StoreAlign`（VF 内 UB 读写）。
- 验证过的平台与 CANN 版本：Ascend950PR（950）+ CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（任何"前一个 VF 的输出 UB 是后一个 VF 输入"的依赖调用链）。
- 路线优先级：banned（在 950/CANN 9.2.0 上按文档声明批量删除依赖 VF 对之间的 barrier 已实测失败，禁止采用）。
- 明确不适用/未验证：无数据依赖的相邻 VF 调用是否可免 barrier 未验证；`LocalMemBar<VEC_STORE, VEC_LOAD>` 替代方案未验证；跨流水线（MTE2→V、V→SCALAR、V→MTE3）同步点不在本实验范围内。

## 根因

同步文档"连续 VF 之间硬件保证顺序"的声明在"前一个 VF 经 StoreAlign 写出的 UB 区域是后一个 VF 经 LoadAlign 读入的源"这一依赖形态上不成立（或不足以覆盖 VF 内 Load/计算/Store + 掩码状态的组合）：移除屏障后消费者 VF 可在后继数据未就绪时读取 UB，产生小比例脏读失配。确切微架构机制（顺序违例发生在 VF 间还是 VF 内部阶段间）未在本会话剖析。

## 坏实践

被证伪的"文档合规"改法（verbatim，已回退）——门控链从"每步一 barrier"改为"三连 VF 无 barrier"：

```cpp
        // S9: gate (3 VF steps — hardware guarantees VF ordering, no inter-VF PipeBarrier)
        uint16_t g1=1;
        asc_vf_call<GateSafeAbsSqrtVF>((__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),tmpP,1,g1);
        asc_vf_call<GateSignMulVF>(tmpP,tmpP,(__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),1,g1);
        asc_vf_call<SigmoidVF>(tmpP,tmpP,1,g1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float gateVal=tmpBuf_.Get<float>().GetValue(0);
```

分块搬运宏同样删掉了每次 `asc_vf_call` 之后的 barrier（还原前的失败形态）：

```cpp
#define CHUNKED_CAST_IN_BF16_F32(dst, src) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<CastBf16ToF32RowVF>((dst)+_co, (src)+_co, _ck, _cr); \
    _co+=_ck; _cr2-=_ck; } } while(0)
```

恢复后的正确形态是每个 `asc_vf_call` 后跟 `AscendC::PipeBarrier<PIPE_ALL>();`（原始版本）。

## 解决配方

- 在 950/CANN 9.2.0 上，凡"VF-A 写 → VF-B 读同一 UB 区域"的相邻 `asc_vf_call` 对，保留其间的 `PipeBarrier<PIPE_ALL>()`。
- 不要按文档声明做批量删除；如确需削减同步开销，必须逐对单变量回归（每删一对 barrier 跑全量 ST），并可评估 VF 编程规则 §6 提到的更轻量 `LocalMemBar<VEC_STORE, VEC_LOAD>` 作为 V→V 同步替代（本卡未验证其有效性；`LocalMemBar` 只能在 `__simd_vf__` 调用上下文内使用、从 `__aicore__` 直接调用会 507035 的约束见 [arch35 __aicore__ 调 LocalMemBar 报 507035 卡](../compilation/arch35_aicore_localmembar_507035.md)——采用前先核对该调用上下文边界）。
- 失败形态速查：小比例、非全零 bitwise 失配（个位数/几十个元素错）≈ 脏读（本卡）；输出近全零 ≈ 计数器/掩码问题（区分见 [UpdateMask 手动递减移除致输出全零卡](updatemask_manual_decrement_required_despite_doc.md)）；大面积（约 95%+）失配 ≈ rep 循环覆盖截断（区分见 [__simd_vf__ 多 rep 截断致大 D 尾部为零卡](simd_vf_multi_rep_truncation_large_d.md)）。

相邻同步边界核对（排查时一并划界）：同一 VF 内部 store→load 须拆调用加屏障，见 [Reg 同 wrapper 回读 UB 引发尺寸相关漂移卡](split_multi_pass_vf_at_ub_producer_consumer_boundaries.md)——区分条件：单调用内跨 pass 回读 vs 本卡跨调用依赖；固定地址累加器的逐块 StoreAlign→LoadAlign 同样需逐 VF barrier，见 [Reg 固定地址跨块累加需逐 VF PipeBarrier 卡](fixed_address_vf_accumulator_needs_per_call_barrier.md)；跨块循环上提到 `__aicore__` wrapper 后调用间加 Vector barrier 的完整形态，见 [AR 跨块循环放在 __simd_vf__ 只执行首块卡](simd_vf_chunk_loop_must_be_driven_by_aicore_wrapper.md)。

## 系统信息

- 硬件/软件：Ascend950PR 单卡，CANN 9.2.0。
- 相关文件：operators/<pkg>/op_kernel/arch35/bench_harness_engram_gate_fusion_kernel.h（连续 asc_vf_call 段与 PipeBarrier 插入处）（路径按语料脱敏策略泛化）。

## 验证信号

以下信号来自同一算子 kernel 修复会话的 23 用例 ST 回归（aclnn 直调，Ascend950PR 实机），单变量实验顺序"基线 → 仅移除 VF 间 barrier → 回退"：

- 基线（barrier 保留）：16/23 通过（来源平台：950；证据等级：tool_observed）。
- 仅移除 VF 间 PipeBarrier（分块粒度、手动递减、跨流水线 barrier 全保留）：13/23，新增 F2_ReduceD1 8/8、E1_NaNInject 6/64、E8_InfInject 7/64 元素 bitwise 失配；同轮中原 507035 崩溃用例依旧崩溃，证明崩溃与 barrier 数量无关（来源平台：950；证据等级：tool_observed）。
- 回退 barrier 后的回归中 F2/E1/E8 恢复 bitwise 通过（16/23，同轮 507035 由另一处守卫修复消除）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论限定：Ascend950PR + CANN 9.2.0、UB 数据依赖的相邻 `asc_vf_call` 链；无数据依赖的相邻 VF、以及跨流水线（MTE2→V、V→SCALAR、V→MTE3）同步点不在本实验范围内。
- `LocalMemBar<VEC_STORE, VEC_LOAD>` 作为轻量替代未验证，不能据本卡直接换用。
- 其他 CANN 版本上文档声明是否成立未验证。

**未决项**：

- 优先项：剖析依赖 VF 对之间到底需要什么强度的同步（PipeBarrier<PIPE_ALL> 是否过重、LocalMemBar<VEC_STORE,VEC_LOAD> 是否足够），在保持 bitwise 通过的前提下削减同步开销。
- 文档"sync-and-consistency §2.3/§2.5"的适用边界应补充"有 UB 数据依赖的 VF 链"例外说明。
## 误诊

- "507035 崩溃源于每个 VF 后的 PipeBarrier<PIPE_ALL> 数量过多（大 L 用例累计数十万次）导致硬件资源耗尽"：本单变量实验证伪——barrier 大幅减少后崩溃不消失；真正根因是多核分发未拦截多余核，见 [507035 未守卫多余核越界分发卡](../runtime/507035_unguarded_extra_cores_oob_dispatch.md)。
- "小比例失配是随机噪声/测试不稳定"：回退 barrier 即恢复通过、重加即复现失配，可稳定复现，非噪声。
