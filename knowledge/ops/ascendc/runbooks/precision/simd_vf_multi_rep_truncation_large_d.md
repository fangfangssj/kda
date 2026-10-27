---
type: "Runbook"
title: "__simd_vf__多rep截断致大D尾部为零"
description: "arch35 上单次 asc_vf_call 传入 rep>1 时 __simd_vf__ 内部 rep 循环覆盖不可靠（D=1024 即 rep=16 时实测只执行前 8、8 rep 分块仍截断、2 rep 不足），大 D 输入 bf16→fp32 cast 后尾部元素保持零/旧值，归约值约为 golden 一半；用 partials/cast 探针与量级判读定位，修复为每调用 1 rep（64 元素）+ 外层 CHUNKED 分块。"
tags: [task.debugging, topic.programming_model.execution_model.simd, topic.programming_model.execution_unit.vector, topic.compute.reduction, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# __simd_vf__多rep截断致大D尾部为零

## 现象

融合算子 kernel（arch35，`asc_vf_call<VF>` 调用 `__simd_vf__` 行函数，行长度 D 按 64 元素一个 rep 切分，`dRep_=(dAlignElems_+63)/64`）在权重覆写根因修复后通过数停在 12/23，剩余失败集中于大 D：D=1024 的 M1/M2/M3 用例近 100% 元素 bitwise 不一致、D=769 的 F7 5941/6152 元素错、D=512 相关用例部分错；小 D（D≤128，rep≤2）用例全部通过。

kernel PRINTF dump 与 golden 对比：M1（D=1024）l=0 行 `statK=1.769626` vs golden `3.376005`、`statQ=1.658331` vs golden `3.315531`——归约统计量约为 golden 的一半（3.315531/2=1.657766≈1.658331），而 keys 原始值逐元素正确。继续探针：

- 逐 rep 部分和（partial）打印：D=1024 应有 16 个 rep 的部分和，实测 `p0..p7` 非零、`p8..p15` 全为 0——归约只吃到了前 8 个 rep；
- cast 前后探针：`bf16_src[512]=-0.052734 bf16_src[1023]=0.065430`（DataCopyPad 搬运正确），但 `fp32_dst[512]=0.000000 fp32_dst[1023]=0.000000`（cast 后为零）；缓冲复用的第二行 `fp32_dst[512]=0.086250` 与本行源值 `0.084473` 不符——是上一行的 stale 值。

即：`CastBf16ToF32RowVF` 单次调用传 `rep=16` 只执行了前 8 个 rep，其余元素保持缓冲原值（初始为零、复用时为旧值）。

后续维护中两次把单调用 rep 上限调回 8（一次为削减 barrier 次数、一次为排查大 shape 崩溃时减少调用数），均立即重踩本缺陷，其中一轮的用例级日志（verbatim，唯一变量为 `MAX_VF_REP` 常量）：

```text
Test F7_D769Prime: FAILED (bitwise mismatch, output 5979/6152 elems differ)   # 改前 PASSED
Test M2_DeepSeekDecode: FAILED (bitwise mismatch, output 3931/4096 elems differ)  # 改前主输出 0/4096 differ
Test M3_LargeBatch: FAILED (bitwise mismatch, output 8245134/8388608 elems differ)  # 改前 296/8388608
Test M5_Dil1Model: FAILED (bitwise mismatch, output 4191068/4194304 elems differ)
Test D1_DecodeL1Dynamic: FAILED (bitwise mismatch, output 3916/4096 elems differ)  # 改前 PASSED
==== summary: 12/23 cases PASSED ====   # 改前 14/23
```

回退该常量后（其余代码不动）精确恢复：F7/D1 回到 PASSED、M3 回到 296/8388608、总数回到 14/23。

## 适用场景

- 工程形态：aclnn 算子仓 kernel 直调（operators/<pkg>/op_kernel/arch35），regbase `Reg::` API + `asc_vf_call<__simd_vf__>` 行函数风格。
- API 路径：`asc_vf_call<VF>(dst, src, n, rep)`、`__simd_vf__ inline void`（内部 `for(i<rep)` 循环 + `Reg::LoadAlign/Cast/StoreAlign`）、`Reg::UpdateMask`。
- 验证过的平台与 CANN 版本：950（arch35）+ CANN 9.2.0。
- 适用粒度：operator_family——所有用 rep 参数表达行长（n = rep×64）的 VF 行函数调用，以及把 rep 当性能旋钮调节的维护改动。
- 路线优先级：recommended_fastpath（分块调用，每调用 1 rep）。banned：在该环境把单调用 rep 增大到 >1（8 rep 已实测必坏）。
- 明确不适用/未验证场景：rep=1 的小 D 已天然安全；D≤64（单 rep 即覆盖）的算子无此风险；非 VF/SIMD 主链不适用；无 `UpdateMask` 计数器或改用其他循环范式的 VF 函数 rep>1 行为未单独验证（见失效边界）。

## 根因

单句归因：arch35 上单次 `asc_vf_call` 传入较大 rep（D=1024 时 rep=16）时，`__simd_vf__` 函数内部的 rep 循环被截断（实测只执行前 ~8 个 rep），未处理的尾部元素保持缓冲原值，导致大 D 输入的 fp32 化尾部为零/旧值、全部下游归约与逐元素计算错。

机制算式（为何归约值约为 golden 一半）：`statQ = Σx_i²`（D=1024 求和）；只累加前 512 个元素时，对随机输入 `Σ_{i<512}x_i² ≈ (1/2)·Σ_{i<1024}x_i² = statQ_golden/2`。实测 1.658331 vs 3.315531/2=1.657766，吻合。

部分归约链在截断下的放大机制：`ReduceSumSqPartialsVF` 每个 rep 迭代向 `work+i` 写一个 partial，`MergeReduceVF` 再把 `work[0..rep-1]` 一次性归并——前段迭代若因计数器提前归零而掩码为空，对应 partial 缺失，整行统计量错误，进而整行输出错误。

截断的指令级机制未定界。一个候选解释是 `UpdateMask` 引用传参语义与手动递减 `r=(r>VL)?r-VL:0` 叠加后计数器每迭代实际递减 2×VL（512 元素 chunk 只覆盖约 256）；但该假设与 [UpdateMask 手动递减不可移除卡](updatemask_manual_decrement_required_despite_doc.md)在 rep=1 分块形态下的观测相互矛盾（那边移除手动递减才全零、保留时 rep=1 全对），两卡机制联动**未定界**，此处仅记录为候选，不作为结论。确切的 `__simd_vf__` rep 截断机制（编译器循环展开上限 / VF 指令迭代立即数位宽 / 计数器语义）待最小复现 kernel 剖析。

## 判别方法

1. 信号形态：小 D 全对、大 D（rep 增大）全错；归约统计量 ≈ golden 的 1/2（rep 减半处截断时）或更小比例。
2. partials 探针：在两阶段归约的 work[] 里逐 rep 打印部分和——尾部 rep 部分和为 0 即输入侧缺数据。
3. cast 前后探针：同一下标打印 `bf16_src[k]` 与 `fp32_dst[k]`——源非零、目的为零（或为上一行 stale 值）即 cast 截断。
4. **量级判读**：约 95–98% 元素失配（F7 5979/6152、M3 8245134/8388608）是覆盖性截断的量级，远超"舍入/归约顺序差异"（同一 kernel 修复崩溃后的残余精度差异仅 270/8388608≈0.003%）；小比例、非全零的个位数/几十元素失配更可能是 VF 间脏读（区分见 [有数据依赖的 VF 调用间仍需 PipeBarrier 卡](ub_dependent_vf_calls_keep_pipebarrier.md)——选用条件：失败集中在移除了 VF 间 barrier 之后）；输出近全零则是计数器/掩码问题（区分见 [UpdateMask 手动递减移除致输出全零卡](updatemask_manual_decrement_required_despite_doc.md)）。
5. **A/B/A 单变量对照**：仅改 `MAX_VF_REP` 常量 1→8→回退，全量 ST 回归——回退即恢复证明 rep 是唯一变量，排除 barrier/递减/其他改动干扰。
6. 判别口诀：归约值≈golden 整数分之一且部分和尾部为零时，先查输入缓冲填充完整性，不要先怀疑归约算法。

## 解决配方

把所有行 VF 调用改为分块调用：每次 `asc_vf_call` 只处理不超过 `MAX_VF_ELEMS = MAX_VF_REP*64` 个元素（rep=ceil(chunk/64)），块间加 `PipeBarrier<PIPE_ALL>`。`MAX_VF_REP` 实测定为 1（每调用 64 元素）。

Before（原始单次大 rep 调用，D=1024 时 dRep_=16）：

```cpp
// S1: CopyIn keys, Cast→fp32
CopyInRowBf16((b*L_+l)*HC_*D_+hc*D_, 0);
AscendC::PipeBarrier<PIPE_ALL>();
asc_vf_call<CastBf16ToF32RowVF>((__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),
    (__ubuf__ bfloat16_t*)bufB0_.Get<T>().GetPhyAddr(),(uint16_t)dAlignElems_,dRep_);
AscendC::PipeBarrier<PIPE_ALL>();
// S4: stat_k
asc_vf_call<ReduceSumSqVF>(tmpP,(__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),(uint16_t)dAlignElems_,dRep_);
```

其中被截断的 cast 行函数（内部 rep 循环）：

```cpp
__simd_vf__ inline void CastBf16ToF32RowVF(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint16_t n, uint16_t rep) {
    constexpr uint32_t VL=64; AscendC::Reg::RegTensor<bfloat16_t> rs; AscendC::Reg::RegTensor<float> rd; AscendC::Reg::MaskReg m;
    uint32_t r=n; for(uint16_t i=0;i<rep;++i){uint16_t o=i*VL; m=AscendC::Reg::UpdateMask<float>(r);
    AscendC::Reg::LoadAlign<bfloat16_t,AscendC::Reg::LoadDist::DIST_UNPACK_B16>(rs,src+o);
    AscendC::Reg::Cast<float,bfloat16_t,kCastB16ToF32>(rd,rs,m); AscendC::Reg::StoreAlign<float>(dst+o,rd,m); r=(r>VL)?r-VL:0;}
}
```

After（CHUNKED 宏分块，逐调用小 rep；常量与宏定义 verbatim）：

```cpp
static constexpr uint16_t MAX_VF_REP = 1;  // __simd_vf__ loops truncate beyond ~8 reps; chunk to 1 rep per call
static constexpr uint32_t VF_VL = 64;
static constexpr uint16_t MAX_VF_ELEMS = MAX_VF_REP * VF_VL;  // 64

// Chunked VF call macros — split large D into MAX_VF_REP-sized chunks to avoid
// __simd_vf__ loop truncation beyond 8 reps (observed on Ascend 950 / DAV 3510).
#define CHUNKED_CALL1(VF, dst, src) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, (src)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_CAST_IN(VF, dst, src) do { \
    uint32_t _co=0; uint16_t _cr2=(uint16_t)dAlignElems_; \
    while(_cr2>0) { uint16_t _ck=_cr2>MAX_VF_ELEMS?MAX_VF_ELEMS:_cr2; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((dst)+_co, (src)+_co, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _co+=_ck; _cr2-=_ck; } } while(0)
#define CHUNKED_REDUCE_P(VF, work, src) do { \
    uint32_t _so=0; uint16_t _rr=(uint16_t)dAlignElems_; uint16_t _wi=0; \
    while(_rr>0) { uint16_t _ck=_rr>MAX_VF_ELEMS?MAX_VF_ELEMS:_rr; uint16_t _cr=(_ck+VF_VL-1)/VF_VL; \
    asc_vf_call<VF>((work)+_wi, (src)+_so, _ck, _cr); AscendC::PipeBarrier<PIPE_ALL>(); \
    _so+=_ck; _rr-=_ck; _wi+=_cr; } } while(0)
// （另有 CHUNKED_CALL0/2/1S/1SW、CHUNKED_CAST_OUT、CHUNKED_REDUCE_DOT_P 同模式，
//   分别覆盖无 src、双 src、标量+src、cast 输出、dot 部分和等调用形态。）
```

调用点批量替换（示例，sed 逐形态替换）：

```bash
cd operators/<pkg>/op_kernel/arch35 && sed -i \
  -e 's/callCastIn<\(CastBf16ToF32RowVF\)>(/CHUNKED_CAST_IN(\1, /g' \
  -e 's/callCastOut<\(CastF32ToBf16RowVF\)>(/CHUNKED_CAST_OUT(\1, /g' \
  -e 's/callReduceP<\(ReduceSumSqPartialsVF\)>(/CHUNKED_REDUCE_P(\1, /g' \
  ... bench_harness_<op>_kernel.h
```

配套要点：

- 两阶段归约用 `ReduceSumSqPartialsVF`（逐 rep partial store）+ `MergeReduceVF`（单次掩码归并）拆成两次调用、中间 `PipeBarrier`——"单 VF 内 store→load 无内存序"的同款根因与拆分屏障模式见 [Reg 同 wrapper 回读 UB 引发尺寸相关漂移卡](split_multi_pass_vf_at_ub_producer_consumer_boundaries.md)——选用条件：中间结果必须经 UB 中转时。
- `MergeReduceVF` 本身无 for 循环、以 rep=13 传参在基线中工作正常——说明失败与 `for(i<rep)` 循环范式相关而非 rep 参数本身；给无循环的合并函数传大 rep 不受本卡限制。
- 每块调用后的 `PipeBarrier` 对"固定地址累加器反复 StoreAlign→LoadAlign"的形态同样不可省，见 [Reg 固定地址跨块累加需逐 VF PipeBarrier 卡](fixed_address_vf_accumulator_needs_per_call_barrier.md)——区分条件：跨块同址读改写 vs 本卡的逐块不相交覆盖。
- 外层分块后注意 wrapper 与 VF callee 不可重复叠加 chunk offset（asc_vf_call 已传前移指针时 callee 不要再加同一 offset），见 [分块 wrapper 与 VF callee 不可重复叠加 offset 卡](wrapper_and_vf_callee_must_not_double_apply_chunk_offset.md)。

## 设计决策

- 分块粒度实测扫描：MAX_VF_REP=8 → 12/23（元素 512 正确但 1023 仍为零，8 rep 的块仍有尾部截断）；4 → 12/23；2 → 12/23；1 → 14/23（F7_D769Prime、D1_DecodeL1Dynamic 转 PASSED）。最终取 1（每调用 64 元素，稳定可验证）。
- 备选（被放弃）：参照内置 add_rms_norm（arch35）的两阶段归约模式重写 reduce——不解决本根因（错在上游 cast 填充，不在归约算法），仅作为归约实现的稳健模式保留。
- 参考证据：CANN 内置算子（add_rms_norm 等）普遍采用 per-rep partial store + 合并的两阶段模式，回避了单次多 rep 的寄存器内累加——与本卡"不信任大 rep 单次调用"的结论一致。
- 后续两次把 rep 调回 8 的动机回顾：一次为削减 barrier 次数、一次为排查大 shape 507035 崩溃时减少调用数——实验证明该方向直接破坏正确性，且**并未**缓解 507035（崩溃用例在 8 rep 下依旧 sync error），两个问题独立；如需减少调用数应从合并 VF 链入手，而不是加 rep。2..7 rep 档位未逐档实测，不默认可用。

## 预防

- 写 VF 行函数调用时，rep 直接按 1（64 元素）规划，长度用外层循环分块；不要把 dRep 整个传进去，也不要为省调用数调大 rep。
- 代码注释里的截断警告（`// __simd_vf__ loops truncate beyond ~8 reps; chunk to 1 rep per call`）不要当过期注释删除——它是前轮实测的护栏。
- 评审检查清单：grep 所有 `asc_vf_call<`，检查第 rep 实参是否有 >1 的可能路径（大 D、对齐后的 dAlignElems）。
- 任何触碰 rep/VL 常量的改动，必须跑全量 ST 回归（重踩轮次正是靠 F7/D1 两个原 PASSED 用例翻车暴露的）。
- 大 D 用例（D≥512、非 2 幂 D=769）纳入 ST 必测集——本缺陷小 D 全对、只有大 D 暴露。

## 验证信号

本次验证来自同一融合算子 kernel 精度修复会话及后续维护轮次：PRINTF 探针定位 + 分块修复后重跑同一 ACLNN ST 测试集；重踩轮次为仅改 `MAX_VF_REP` 常量的 A/B/A 对照。

- partials 探针：D=1024 的 16 个 rep 部分和中 `p8..p15=0.000000`（来源平台：950；证据等级：tool_observed）。
- cast 前后探针：`bf16_src[512]=-0.052734 bf16_src[1023]=0.065430` 而 `fp32_dst[512]=0.000000 fp32_dst[1023]=0.000000`；缓冲复用行的 `fp32_dst[512]=0.086250` 与本行源值不符（来源平台：950；证据等级：tool_observed）。
- 分块粒度扫描：MAX_VF_REP=8/4/2 均 12/23，MAX_VF_REP=1 达 14/23 且 F7_D769Prime、D1_DecodeL1Dynamic PASSED（来源平台：950；证据等级：tool_observed）。
- 清理全部调试代码后复跑确认 14/23（相比本轮起点 7/23）（来源平台：950；证据等级：tool_observed）。
- 重踩轮次 A（基线 14/23）：rep=1 时 `summary: 14/23 cases PASSED`，F7_D769Prime PASSED、D1_DecodeL1Dynamic PASSED、M3_LargeBatch 296/8388608 elems differ；仅改 rep=8 后 `summary: 12/23 cases PASSED`，F7 翻为 5979/6152 differ、D1 翻为 3916/4096 differ、M2 主输出 0/4096→3931/4096、M3 296→8245134/8388608、M5 4191068/4194304；回退 rep=1 后恢复 14/23、F7/D1 恢复 PASSED、M3 恢复 296/8388608（来源平台：950；证据等级：tool_observed）。
- 重踩轮次 B（基线 16/23）：仅 MAX_VF_REP 1→8（barrier、手动递减等其余不动）后 14/23，新增 F7_D769Prime 失配 5979/6152、D1_DecodeL1Dynamic 失配 3916/4096；已失败大 D 用例失配数放大（大批量用例 296/8388608 → 8245134/8388608）；D≤64 用例（F1/F2/F5/F6/F8、E1–E8）全部通过；回退 MAX_VF_REP=1 后的后续两轮回归中 F7 与 D1 均恢复 bitwise 通过（来源平台：950；证据等级：tool_observed）。
- 同轮证 507035 崩溃与 rep 无关：rep=8 下崩溃用例依旧 sync error，两问题独立（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 截断的精确阈值与机制未定界：16 rep 的调用执行了前 8 个；8 rep 的分块仍出现尾部截断；2 rep 不足；1 rep 稳定。保守取 1，不要据此外推"≤8 安全"，更不要按注释字面把 8 当安全上界。
- 结论限定"UpdateMask 计数器 + 手动递减 + for(i<rep)"这一具体 VF 行处理范式（950 + CANN 9.2.0）：无计数器、单一递减源或其他循环范式的 VF 函数 rep>1 行为未验证；`MergeReduceVF`（无 for 循环）以 rep=13 传参正常，说明失败与 for 循环范式相关而非 rep 参数本身。
- 与 UpdateMask 手动递减卡的机制联动（计数器是否每迭代递减 2×VL）未定界，两卡结论各自独立成立、互不依赖该假设。
- 大 D + 大 L 用例（M1/M5/D3）在本修复后仍 100% 失败（疑似 kernel 指令数上限等其他根因），后续修复轮次继续，不在本卡范围。
- 507035 大 shape 崩溃与本卡无关（rep=8 下崩溃用例未变化），不要把本卡当成崩溃修复。
- 每调用 1 rep 的性能代价未测量；性能敏感场景需另行权衡。VL≠64 的 `__simd_vf__`（如 VL=128/256）未验证，截断行为可能按元素数或 rep 数计，需单独实测。
- 仅验证 arch35（950）+ CANN 9.2.0；其他 arch 未验证（未验证，推断可迁移，迁移前必须复测）。

**未决项**：

- 优先项：用最小复现 kernel 定界 `__simd_vf__` rep 截断的精确机制与上限（编译器循环展开上限 / VF 指令迭代立即数位宽 / 计数器在 `UpdateMask` 与手动递减叠加下的真实步进），给出可解释的安全 rep 上界；2..7 rep 档位的正确性未测，如需压缩调用数可先补这组实验。
- 与 UpdateMask 计数器语义的联动（2×VL 双递减假设 vs rep=1 观测的矛盾）待指令级剖析定界。
- 大 D 大 L 用例剩余失败（M1/M5/D3 等）的根因追踪（疑似 kernel 指令数上限）属后续修复轮次。
- 分块对性能的影响未测量。
## 坏实践

- 把 `dRep_`（行长/64）作为 rep 实参一次性传给 `asc_vf_call`，依赖 `__simd_vf__` 内部 for 循环处理任意大 rep——大 D 下静默截断，无任何编译期/运行期告警。
- 依据注释里 "truncate beyond ~8 reps" 的字面意思把 rep 恰好设为 8——注释口径是前轮粗略记录，实测 8 已越界；安全边界要按实测而不是按注释取整。
- 为减少调用/`PipeBarrier` 总数而增大单调用 rep——实测破坏正确性且不缓解 507035；减少调用数应从合并 VF 链入手。
- 在排查一个未定位根因（507035）的过程中顺手做性能向常量调整——引入第二变量，污染对照，还破坏了原本正确的用例。
- 发现归约值约为 golden 一半时，先重写归约算法（本例先按内置算子两阶段模式重写，无效且引入过新问题）——应先用 partials 探针验证输入完整性。

## 误诊

- 第一归因：怀疑 `Reg::Reduce` 多 rep 寄存器内累加错误，遂参照 CANN 内置 add_rms_norm 重写两阶段归约——修后 statK 仍为 1.769626 不变。partials 探针显示"归约没错、输入就只有前一半"，才回溯到上游 cast 截断。
- 重踩轮次的即时归因"更大数据块改变了归约顺序（部分和分组不同）"：与失配量级不符——舍入/顺序差异在本 kernel 的实测残余是小比例（270/8388608≈0.003% 量级），而 rep=8 实验中同一用例失配率约 98%，是覆盖性截断而非舍入重排；该解释已按截断机制修正。
- "barrier 减少导致失配"：同会话单变量实验（仅移除 barrier、rep 保持 1）产生的是小比例失配且用例集合不同（F2/E1/E8），可与本卡现象区分（见 [有数据依赖的 VF 调用间仍需 PipeBarrier 卡](ub_dependent_vf_calls_keep_pipebarrier.md)）。
