---
type: "Runbook"
title: "VF乘法原位覆盖源buffer致bitwise失配"
description: "conv 累加循环里 MulRowVF 以 B0(normed_vg) 为目的地址原地覆盖，k=K-1 分支再读 B0 时读到上轮乘积，输出 bitwise 全错；改用独立 temp buffer 保存乘积即修复。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.arithmetic]
status: stable
sources: []
created_at: '2026-09-14T08:44:28Z'
updated_at: '2026-09-14T08:44:28Z'
platforms: ["950"]
---

# VF乘法原位覆盖源buffer致bitwise失配

## 现象

自定义 ACLNN 算子包 Normal 分支（tilingKey=0）kernel 的 507035 崩溃修掉之后（kernel 可完整跑完所有用例；崩溃由 [多余核无守卫越界分发](../runtime/507035_unguarded_extra_cores_oob_dispatch.md) 与 [环形索引负模回绕](../runtime/507035_negative_modulo_ring_index_ub_offset.md) 两个独立根因先后闭合），ST 黑盒数值用例仍**全部** bitwise 失配：`./test_aclnn` 输出 `Test F1_StandardPrefill: FAILED (bitwise mismatch, output 60/64 elems differ)`、`F2_ReduceD1: FAILED (8/8 elems differ)`、`F6_DecodeL1ConvState: FAILED (16/16 elems differ)`、`E3_AllZero: FAILED (8/64 elems differ)` 等，仅最小调用通路 OK。当时代码状态：kernel 的 lookback conv1d 累加循环用 `asc_vf_call<MulRowVF>` 逐 tap 乘累加，`B0` 缓冲区此时保存着上游归一化结果 normed_vg，`B1` 是累加器 conv_raw。

## 适用场景

- 工程形态：aclnn 算子包 ST 黑盒精度联调（`tests/aclnn` 下 `./test_aclnn` 逐用例比对 golden）。
- API 路径：`__simd_vf__` 自定义 VF 函数 + `asc_vf_call` 调用（MulRowVF/AddRowVF 类乘加），UB 多 buffer 复用（B0/B1/B2 + temp）。
- 验证过的平台与 CANN 版本：950 + CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family——同类"多 UB buffer + VF 乘加链"的 kernel 通用；本例验证于含 lookback conv 的融合算子。
- 路线优先级：recommended_fastpath（独立 temp 保存乘积，源 buffer 只读）。
- 明确不适用 / 未验证场景：失配若来自 dtype/容差口径（如 bf16 舍入模式、MERE/MARE 口径）或 host golden 本身错误，不属本卡；本卡只覆盖"kernel 内数据流别名破坏"这一根因。

## 根因

conv1d 循环每一 tap 执行 `MulRowVF(b0P, xs, cwP+k*dAlignElems_)`——**以 B0 为目的地址**把乘积写回 B0，随后 `AddRowVF(b1P, b1P, b0P)` 累加进 B1。但循环的 `k=K-1` 分支里 `xs=b0P`，期望 B0 仍保存**原始 normed_vg**；此时读到的却是上一轮迭代写入的乘积。即同一 UB buffer 在循环中同时充当"乘法输出"与"后续源"，别名破坏了数据依赖——设计约定 B0=normed_vg（源）、B1=conv_raw（累加）、乘积应落独立 temp，实现把 temp 放进了源 buffer。崩溃期被掩盖（507035 先崩在别处），崩溃修掉后即以全量 bitwise 失配显形。

与库内『写入破坏后续读取』族卡同属一个抽象类、但根因各自独立分卡：[FP16 half 就地覆写 x1 破坏 y_second 读依赖](fp16_half_x1_y_second.md) 与 [Abs 原地取绝对值后量化丢符号致 int8 坍缩](abs_inplace_overwrites_signed_source_before_quant.md) 是同一条指令 dst==src 的逐元素别名；本卡是 conv tap 循环里乘法目的地址覆盖 k=K-1 特判分支要读的源 buffer，且被崩溃掩盖、崩溃修复后以全量 bitwise 失配显形。

## 解决配方

两处配套修改（`op_kernel/arch35/*_kernel.h`）。

**① Init() 无条件分配 convStateBuf_，兼作 conv 循环乘积 temp**（修复前 convStateBuf_ 仅在 `stateLen_>0` 时分配，其余场景无可用 temp）：

```cpp
    // convStateBuf_ doubles as conv-loop temp (fp32, 1 row) AND conv_state loader (bf16, D*SL)
    int64_t convStateBufSize=(dAlignElems_*4+31)&~31;
    if(stateLen_>0){
        int64_t csSize=(D_*stateLen_*sizeof(T)+31)&~31;
        if(csSize>convStateBufSize) convStateBufSize=csSize;
    }
    pipe_->InitBuffer(convStateBuf_,convStateBufSize);
```

**② conv 循环乘积写入独立 temp `ctP`，B0 全程只读**：

修复前：

```cpp
        // S15-16: Conv1d→B1
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<ZeroFillRowVF>((__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),(uint16_t)dAlignElems_,dRep_);
        auto b0P=(__ubuf__ float*)bufB0_.Get<float>().GetPhyAddr();
        auto b1P=(__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr();
        for(int64_t k=0;k<K_;k++) {
            __ubuf__ float* xs;
            if(k==K_-1) { xs=b0P; }
            else { int64_t n=l+k*dil_; int64_t lr=l%stateLen_;
                if(stateLen_>0 && (n%stateLen_)==lr) xs=svP; else xs=lbP+(n%stateLen_)*dAlignElems_; }
            asc_vf_call<MulRowVF>(b0P,xs,cwP+k*dAlignElems_,(uint16_t)dAlignElems_,dRep_);
            asc_vf_call<AddRowVF>(b1P,b1P,b0P,(uint16_t)dAlignElems_,dRep_);
        }
```

修复后：

```cpp
        // S15-16: Conv1d→B1 (B0=normed_vg preserved as source, convStateBuf_ as MulRowVF temp)
        AscendC::PipeBarrier<PIPE_ALL>();
        asc_vf_call<ZeroFillRowVF>((__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr(),(uint16_t)dAlignElems_,dRep_);
        auto b0P=(__ubuf__ float*)bufB0_.Get<float>().GetPhyAddr();
        auto b1P=(__ubuf__ float*)bufB1_.Get<float>().GetPhyAddr();
        auto ctP=(__ubuf__ float*)convStateBuf_.Get<float>().GetPhyAddr();
        for(int64_t k=0;k<K_;k++) {
            __ubuf__ float* xs;
            if(k==K_-1) { xs=b0P; }
            else { int64_t n=l+k*dil_; int64_t lr=l%stateLen_;
                if(stateLen_>0 && (n%stateLen_)==lr) xs=svP; else xs=lbP+(n%stateLen_)*dAlignElems_; }
            asc_vf_call<MulRowVF>(ctP,xs,cwP+k*dAlignElems_,(uint16_t)dAlignElems_,dRep_);
            asc_vf_call<AddRowVF>(b1P,b1P,ctP,(uint16_t)dAlignElems_,dRep_);
        }
```

改动语义：`MulRowVF` 目的地址 `b0P → ctP`，`AddRowVF` 第三操作数 `b0P → ctP`；B0 自始至终只读。验证：`bash operators/build.sh <pkg>` 重编译后在 `tests/aclnn` 复跑 `./test_aclnn` 比对逐用例 mismatch 计数。

## 设计决策

- 选"新增独立 temp"而不是"重排循环让 k=K-1 先读 B0"：重排会改变累加顺序（conv 累加对顺序敏感），且把特判逻辑摊到每个 tap；独立 temp 只多占一行 fp32 UB（`dAlignElems_` 元素，32B 对齐），数据流清晰。
- temp 复用 convStateBuf_（单行 fp32 大小取 max(D*SL bf16, 一行 fp32)）而不是再开新 TBuf：UB 预算紧张（B0/B1/B2 主 buffer 已占大头），conv_state 装载与 conv 乘法 temp 在时序上不重叠，复用安全。
- 当前方案属 recommended_fastpath。

## 预防

- 写 VF/向量乘加链时先画 buffer 读写时序表：任一 buffer 若在某步后仍作为源被读取，期间禁止作为任何 VF 指令的目的地址。
- review 高危信号：`asc_vf_call<XxxVF>(b0P, xs, ...)` 中目的指针与后续源指针同名、或目的指针所属 buffer 在同循环其他分支作为源出现。
- 崩溃修复后立即全量跑数值用例：被崩溃掩盖的数据流错误会以"全量 bitwise 失配"显形，不要误判为精度口径问题。

## 验证信号

以下为同轮 kernel 精度修复会话中的上板实测（`./test_aclnn` 逐用例 mismatch 计数）：

- 修复前（崩溃已消除、conv 循环仍原位覆盖）：F1 60/64 differ、F2 8/8 differ、F6 16/16 differ、E3 8/64 differ，数值用例 0 通过（来源平台：950；证据等级：tool_observed）。
- 修复后（乘积落独立 temp）：F6_DecodeL1ConvState PASSED，E3_AllZero/E4_RawGateZero/E5_RawGatePos/E6_RawGateNeg/E7_RMSNormZero 全部 PASSED，总通过 7/23（此前仅最小通路）；F1 60/64→38/64、F2 8/8→3/8、F8 62/64→25/64（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 本卡修复针对"乘法原位覆盖后续要读的源 buffer"；若源 buffer 之后不再被读，原位写是合法优化，不适用本卡。
- UB 紧张到连一行 temp 都挤不出的场景，需改为重排读取顺序（会改变累加顺序，需重验精度），本卡未走该路线、未验证。
- 剩余未通过的 15 例（大 shape 全量 differ 等）与本根因无关（见遗留问题）。
- 验证仅覆盖 950 + CANN 9.2.0；其他平台推断同样适用（VF 语义相同），未验证，推断可迁移。

**未决项**：

- 修复后仍有 15/23 用例失配（M1/M3 等 8/8 全量 differ）。会话内的归因是"VF 指令精度与 host fp32 的固有差异、测试侧尚未实现设计规定的 MERE/MARE 容差口径"——该归因仅为 agent 陈述（agent_reported），本轮未做数据级验证，下个接续点：按算子设计的精度口径实现 MERE/MARE 比对后复跑，区分"容差口径缺失"与"真实精度缺陷"。
- 链式衔接：本卡是本 trace 修复链的中间环——前驱 [卷积环形 lookback 写早于读覆盖历史状态](conv_ring_lookback_write_before_read_overwrites_history.md)（修环形历史覆盖；其修后仍 ~22 用例全红的状态即本卡修前状态）；后继 [多权重共用加载 buffer 后加载覆写](fused_kernel_shared_weight_ub_buffer_later_load_overwrites.md)（本卡修后 7/23、F1 38/64 的状态与其修前基线逐字吻合）。三张卡是三个不同代码位置、不同机制的独立根因闭环，不合并。
## 坏实践

- 在循环里让 VF 指令目的地址覆盖"后面还要读"的源 buffer，靠"当前用例碰巧没触发"过关——短循环 K 小时可能侥幸，K=K-1 特判分支必然踩中。
- 崩溃修复后不复跑数值用例就转下一任务，让数据流错误潜伏到验收阶段。

## 误诊

- 崩溃修掉后首轮全量 bitwise 失配时，曾怀疑连续 `asc_vf_call` 之间缺 V-pipeline 同步（kernel 侧 PRINTF 观测到中间值异常——两打印点间未被写的 UB 值变化实为 PRINTF 扰动制造的幽灵异常，判别闭环见 [kernel PRINTF 时序扰动：无 PRINTF 基线验证](../runtime/printf_perturbs_timing_verify_without_printf.md)）；插入全链 PipeBarrier 后结果逐用例不变，排除同步假设，真因即本卡 buffer 别名。
