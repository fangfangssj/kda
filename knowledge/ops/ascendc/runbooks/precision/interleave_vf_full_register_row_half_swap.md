---
type: "Runbook"
title: "Reg::Interleave 自定义重组 D=128 行两半互换"
description: "用 Reg::Interleave+StoreAlign 自写 InterleaveVF 做行内交错重组，整寄存器行（H=64=fp32 寄存器宽度）输出前后两半互换且换 store 顺序无效；flat 行改 UB 级双输出 AscendC::Interleave 后全绿。"
tags: [task.debugging, topic.precision, topic.data_movement.layout_transform]
status: stable
sources: []
created_at: '2026-09-17T03:25:37Z'
updated_at: '2026-09-17T03:25:37Z'
platforms: ["950"]
---

# Reg::Interleave 自定义重组 D=128 行两半互换

## 现象

在 Ascend950 上为 RoPE 类算子 interleaved 分支调试 kernel：按"DeInterleave 拆奇偶 → fp32 域 Mul/MulAdd → Interleave 重组"实现 rotate_half，重组步用的是自写 `__simd_vf__ InterleaveVF`（`Reg::LoadAlign` 两个源寄存器 → `Reg::Interleave` 产两个目的寄存器 → 两次 `Reg::StoreAlign` 分别落低/高段）。padded 平面布局的向量核异常修掉之后，11 个数值用例过了 10 个，唯独大 shape 用例（[8,1024,64,128] fp16，D=128、H=64，fp32 计算域下一次 repeat 恰好两寄存器全量落盘）失败：

```
FAILED: queryOut mismatch at idx=0 actual=-0.0193939 expected=0.496094 (dtype=fp16 …)
```

用独立 numpy 探针对比输出与期望 .bin：`act[0..15] == exp[64..79]` 为 **True**、`act[0..15] == exp[0..15]` 为 False——输出的前 64 与后 64 元素整段互换；且 `act != 输入 q`（非搬运直通，RoPE 数值本身是对的）。所有小 D 用例（D=2/4/6，H<16，第二寄存器完全被 mask 掉、仅第一寄存器部分写入有效）全部正确。

## 适用场景

- 工程形态：AscendC regbase kernel（dav-3510），`__simd_vf__` + `asc_vf_call` 自定义 Vector 计算链。
- API 路径：`AscendC::Reg::Interleave` / `AscendC::Reg::DeInterleave`（寄存器级，`reg_compute/kernel_reg_compute_vec_duplicate_intf.h`）；UB 级 `AscendC::Interleave`（`kernel_operator_vec_duplicate_intf.h`，双输出）。
- 验证过的平台与 CANN 版本：Ascend950（Ascend950PR），CANN 9.2.0（与 frontmatter `platforms` 一致）。
- 适用粒度：operator_family（症状绑定"自写 Interleave 重组 + 行长达到整寄存器"这一组合，不绑定具体算子）。
- 路线优先级：recommended_fastpath（能整块用 UB 级 API 就不要手写寄存器级重组）。
- 明确不适用 / 未验证场景：行元素数小于一个寄存器宽度且尾块被 mask 的场景 InterleaveVF 实测正确，本卡不否定该用法；`Reg::Interleave` 本体在其他调用形态（如 CANN 内部 gather 实现）中的语义本卡未测。

## 根因

自写 `InterleaveVF` 中 `Reg::Interleave(dstReg0, dstReg1, src0, src1)` 的两个输出寄存器到"交错结果低半/高半"的映射，与两次 `StoreAlign` 目标段（`CreateAddrReg(2i, N)` 低段 / `CreateAddrReg(2i+1, N)` 高段）的对应关系，在"行元素恰好填满寄存器、两段都全量写入"（H=64=fp32 寄存器宽度 64）的场景下与预期相反：低段实际落到了交错结果的**高半**。而且这不是简单的输出寄存器序反了——实验证明把两次 `StoreAlign` 的寄存器对调后，D=128 的输出**逐位不变**（仍是高半在前），小 D 用例却全部翻转出错；即两种 store 顺序在整寄存器行上产生相同结果、在尾块掩码行上只有原顺序正确。微架构层的确切寄存器排布机制未定位（诚实记录，见遗留问题）。UB 级 `AscendC::Interleave`（内部 impl 自带正确的段映射与尾块处理）在同样数据上无此问题。

## 解决配方

重组步改为条件式：**flat 布局（`paddedH == H`，即 H 为 16 的倍数 → outEven/outOdd 无行间 padding、可当连续整块）时用 UB 级双输出 `AscendC::Interleave` 整块重组；仅 paddedH>H 的小 D 行保留 per-row `InterleaveVF`**。

修复前（无条件 per-row InterleaveVF，fp32 路径，fp16 路径同构）：

```cpp
            // ③ Interleave per-row: outFull[2i]=outEven[i], outFull[2i+1]=outOdd[i]（复用 bufQ_ 槽位）
            rep = static_cast<uint16_t>(CeilDivU32L(H, kVecLen));
            for (uint32_t r = 0; r < cnt; ++r) {
                asc_vf_call<InterleaveVF<ComputeT>>(
                    (__ubuf__ ComputeT*)q.GetPhyAddr() + static_cast<int64_t>(r) * paddedD_,
                    (__ubuf__ ComputeT*)oE.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                    (__ubuf__ ComputeT*)oO.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                    D, kVecLen, rep);
            }
```

修复后（fp32 路径条件式，fp16 路径把 `q` 换 `qFp` 同构修改）：

```cpp
            // ③ Interleave: outFull[2i]=outEven[i], outFull[2i+1]=outOdd[i]（复用 bufQ_ 槽位）
            //    paddedH==H（H 为 16 倍数 → outEven/outOdd flat 无行间 padding）→ whole-chunk
            //    AscendC::Interleave 双输出；否则 paddedH>H → per-row InterleaveVF（每行 D 元素）。
            if (paddedH_ == H_) {
                uint32_t padCount = PadUp(static_cast<uint32_t>(cnt) * H_, 8u);
                AscendC::Interleave(q, q[padCount], oE, oO, static_cast<int32_t>(padCount));
            } else {
                uint16_t repI = static_cast<uint16_t>(CeilDivU32L(H_, kVecLen));
                for (uint32_t r = 0; r < cnt; ++r) {
                    asc_vf_call<InterleaveVF<ComputeT>>(
                        (__ubuf__ ComputeT*)q.GetPhyAddr() + static_cast<int64_t>(r) * paddedD_,
                        (__ubuf__ ComputeT*)oE.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                        (__ubuf__ ComputeT*)oO.GetPhyAddr() + static_cast<int64_t>(r) * paddedH_,
                        D, kVecLen, repI);
                }
            }
```

flat 分支的正确性前提（实现时已逐项核对）：`paddedH==H` ⟹ H 为 16 倍数 ⟹ D 为 32 倍数 ⟹ `paddedD==D`（outFull 亦 flat）；`padCount=PadUp(cnt*H, 8)` 为 8 的倍数保证第二输出基址 `q[padCount]` 32B 对齐且 count 为偶；两输出共写 `2*padCount ≤ 2*nH+16 ≤ nD+16`，在带尾块余量的缓冲内。

自写 InterleaveVF（终版，两种 store 顺序实验后保留的原始顺序）：

```cpp
template <typename T>
__simd_vf__ inline void InterleaveVF(
    __ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr,
    uint32_t dCount, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> srcReg0, srcReg1, dstReg0, dstReg1;
    AscendC::Reg::MaskReg mask0, mask1;
    AscendC::Reg::AddrReg aSrc, aDst0, aDst1;
    uint32_t dstCount = dCount;  // UpdateMask 按引用递减
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aSrc = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::Reg::LoadAlign(srcReg0, src0Addr, aSrc);
        AscendC::Reg::LoadAlign(srcReg1, src1Addr, aSrc);
        AscendC::Reg::Interleave(dstReg0, dstReg1, srcReg0, srcReg1);
        // dstReg0/dstReg1 各持交错结果的低/高半（各 oneRepeatSize 元素），分别 StoreAlign 到
        // dst 的 [2i*N .. +N) 与 [2i*N+N .. +2N)。用独立 AddrReg/MaskReg 变量避免复用 race。
        aDst0 = AscendC::Reg::CreateAddrReg<T>(static_cast<uint16_t>(2u * i), oneRepeatSize);
        mask0 = AscendC::Reg::UpdateMask<T>(dstCount);
        AscendC::Reg::StoreAlign(dstAddr, dstReg0, aDst0, mask0);
        aDst1 = AscendC::Reg::CreateAddrReg<T>(static_cast<uint16_t>(2u * i + 1u), oneRepeatSize);
        mask1 = AscendC::Reg::UpdateMask<T>(dstCount);
        AscendC::Reg::StoreAlign(dstAddr, dstReg1, aDst1, mask1);
    }
}
```

## 设计决策

- 为什么不继续修 InterleaveVF 而是分叉：两种 store 顺序在 D=128 上输出相同，说明"对调寄存器"不是正确的修复方向，继续盲试寄存器序成本高；UB 级 `AscendC::Interleave` 的内部 impl（`InterleaveImplNormal`，按 `VecMicroGetRepeatTimes`/掩码尾块处理）在同样数据上已被工程内验证正确，flat 场景直接整块调用最省。
- 分叉条件选 `paddedH_ == H_` 而非 `H >= 某阈值`：它精确刻画"outEven/outOdd 是否 flat"这一 UB API 的前提，而不是碰运气选形状阈值。
- InterleaveVF 保留在小 D 路径：小 D 行（H 远小于寄存器宽度）实测正确（第二寄存器被 UpdateMask 全掩，只有第一寄存器部分写入有效），且 UB 级 API 处理不了"行间有 padding 缝隙"的非 flat 源。

## 预防

- 需要 Interleave/DeInterleave 语义时优先用 UB 级 `AscendC::Interleave`/`AscendC::DeInterleave`（`kernel_operator_vec_duplicate_intf.h`）——它藏在 "duplicate" 头文件里、按名字搜不到，先用 `grep -r "Interleave" $ASCEND_HOME_PATH/…/include/basic_api/` 定位再手写。
- 手写寄存器级重组时，测试集必须包含"行元素数 ≥ 一个寄存器宽度（两段全量写入）"的用例——只用小 shape 测会漏掉本卡的两半互换。
- 判别指纹：`actual[0..k] == expected[N/2..N/2+k]`（前后半互换）且小 shape 全对，优先怀疑双输出寄存器重组的段映射。

## 验证信号

以下信号来自该算子 interleaved 分支 kernel 的精度调试会话（10/11 基线上对大 shape 用例做 .bin 级探针对比、两种修复尝试对照、最终条件式修复全量回归）：

- 失败签名：大 shape 用例（D=128）`actual=-0.0193939 expected=0.496094`，探针证实 `act[0..15] == exp[64..79]` 为 True（两半互换）、`act[0..15] == q[0..15]` 为 False（非直通）（来源平台：950；证据等级：tool_observed）。
- 反证实验：对调两次 `StoreAlign` 的寄存器后重建重跑，D=128 输出逐位不变（仍 `actual=-0.0193939`），小 D 用例从全对翻转为 numeric 0/11（来源平台：950；证据等级：tool_observed）。
- 修复后：条件式重组重建重跑，branch-1 numeric 11/11 + returns_empty 1/1 = 12/12 全绿；UT 57/57 不回归（来源平台：950；证据等级：tool_observed）。

## 失效边界

- 结论覆盖 `Reg::Interleave` 双输出 + 两次 `StoreAlign` 的手写组合在 dav-3510/CANN 9.2.0 上的行为；UB 级 `AscendC::Interleave` 的整块用法在 flat 布局下验证正确，非 flat（行间 padding）场景不支持（分叉保留 InterleaveVF）。
- `paddedH == H` 分叉依赖"16 元素对齐"的布局常量；若布局常量改为非 16 倍数，flat 判定与 32B 对齐前提需要重新推导。
- 其他 CANN 版本的 `Reg::Interleave` 实现若修正/变更了输出寄存器排布，症状可能不复现（未验证，机制以目标版本 impl 为准）。
- 相邻未定论现象：同算子族 interleaved 链相对 half 链的白盒精度劣化（含 D=88 边界、机制未定论）见 [RoPE interleaved 链白盒精度劣于 half 链卡](rope_interleaved_whitebox_precision_regression.md)——与本卡不同根因，勿合并归因。

**未决项**：

- **优先项：复现并追根因到机制层**——在 dav-3510 上用最小用例（单行 H=64 fp32，Reg::Interleave 双输出逐寄存器 dump）定位两种 store 顺序产生相同输出的确切机制（寄存器洪泛/写合并/`CreateAddrReg` 段寻址语义任一假设均未验证），给出 `Reg::Interleave` 输出排布的权威结论。
- UB 级 `AscendC::Interleave` 对非偶数 count、count 小于寄存器宽度等边角的尾块行为未单独压测（本工程 count 恒为 8 的倍数）。
## 坏实践

- 用两个小 shape 用例验收自写寄存器级交错重组（第一版测试组合的盲区）。
- 发现两半互换就直接对调输出寄存器序而不做对照实验——本例对调后大 shape 无变化、小 shape 全挂，说明该方向是误诊。

## 误诊

- "Reg::Interleave 把低半放 dstReg1、高半放 dstReg0，对调两次 StoreAlign 即可修复"——对照实验证伪：对调后 D=128 输出逐位不变（仍互换），小 D 全部翻转（numeric 0/11），已回退（该假设仅推断，无工具证据支撑其成立）。
