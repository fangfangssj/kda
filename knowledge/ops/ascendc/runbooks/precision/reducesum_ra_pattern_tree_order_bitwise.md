---
type: "Runbook"
title: ReduceSum RA(c310)累加序：16宽平衡寄存器树
description: dav_c310（Ascend950PR/NpuArch 3510）上ReduceSum<float,RA>的逐位累加序可从CANN impl头文件精确推导：dimA分档调度、CalculateMainR取最大2幂、tail行先并main行、16宽平衡寄存器树交错配对；4行小shape可位级锚定验证。
tags: [task.debugging, topic.compute.reduction, topic.implementation.api_layer.high_level_api, topic.precision]
status: stable
sources: []
created_at: '2026-09-09T11:49:41Z'
updated_at: '2026-09-09T11:49:41Z'
platforms: ["950"]
---
# ReduceSum RA(c310)累加序：16宽平衡寄存器树


## 现象

在 Ascend950PR 上排查某融合反向算子 bias 归约的精度问题时，需要回答"硬件 `ReduceSum<float, RA>`（对 UB 内 (dimR, dimA) tile 沿第一轴归约）内部按什么顺序累加"——这决定了能否在 host 侧逐位预测 kernel 的归约结果（位级 golden、归约序仿真、ULP 级差异定界都需要）。用 32 行真实 kernel 输出盲试 15 种候选序（顺序 / 严格折半树 / 2~16 行分组×顺序/树合并 / stride-16 配对树）全部不匹配（mismatch 3788~4675/6144 列），说明硬件树不属于任何常规候选；随后读 CANN impl 头文件破解了精确树，并用 B=4 的最小 shape 用例位级锚定验证。

## 适用场景

- 工程形态：AscendC kernel 开发/精度调试中需要逐位预测或复现 ReduceSum 结果的场景（位级 golden、归约链路仿真、ULP 级差异定界）。
- API 路径：`AscendC::ReduceSum<T, Pattern::Reduce::RA, isReuseSource>`（adv_api 高阶 API）；impl 头文件在 `$ASCEND_HOME_PATH/x86_64-linux/asc/impl/adv_api/detail/reduce/` 下。
- 验证过的平台与 CANN 版本：Ascend950PR（950PR_9589，NpuArch=3510，AIV=dav-c310-vec → 生效变体目录 dav_c310）+ CANN cann-9.0.0-beta.2（与 frontmatter `platforms` 一致）。
- 适用粒度（transfer_scope）：operator_family——dav_c310 变体上所有使用 ReduceSum RA 的 kernel（严格绑定该变体，非全平台）。
- 路线优先级：recommended_fastpath（需要位级预测时的唯一可靠路径：读生效变体 impl + 小 shape 锚定）。
- 明确不适用/未验证：其他 dav_* 变体（树可能不同）；lane 字节数非 32B 对齐的 UnAligned 分支；fp16/bf16（vlSize 不同）；AR pattern（沿内轴归约）另一套实现；dimA≤32 的 Concat 树与 dimR>64 的多轮折叠未实测。

## 判别方法

1. **先确认生效变体**：SoC 前缀 Ascend950PR 会匹配 8 个 ini（950z/9579/957b/957c/957d/9589/958b/9599，均 NpuArch=3510）→ 生效变体目录 `dav_c310`（AIV=dav-c310-vec）。**只读 dav_c310 目录下的 impl，其他变体目录的断言不构成证据**（可用 lookup_arch_variant 脚本确认）。
2. **读调度层**（`reduce_sum_c310_impl.h` 的 `ReduceSumRAReuseSourceCompute`）：按 `(srcShape[1]·sizeof(T)) % 32 == 0` 分 Aligned/UnAligned 两路；Aligned 走 `ReduceRAImpl<...>(dstAddr, srcAddr, tmpAddr, srcShape[1], srcShape[0])`——**注意实参交换**：dimA=srcShape[1]（行宽 lane）、dimR=srcShape[0]（行数）。
3. **dimA 分档**（ReduceRAImpl 内部派发）：`dimA ≤ 32 || dimA > 65535` → Concat 实现；`32 < dimA ≤ 64` → LessThanVL；`dimA > 64` → OverVL。（本算子 lane=6·aUbFactor：384/576 → OverVL；48 → LessThanVL。）
4. **关键常量**（`reduce_common_util_c310_impl.h`）：fp32 的 `vlSize = GetVecLen()/sizeof(T) = 256/4 = 64`；`BASE_FOLD=4`；`FOLD_ONE/TWO/THREE = 1/2/3`；`CalculateMainR(dimR)` = 不超过 dimR 的最大 2 幂（48→32、32→32、4→4）；`CalculateFolds(n)` = log2(n)；`FindClosestPowerOfTwo(n) = 63−clz(n) = floor(log2(n))`。
5. **OverVL / isReuseSource=true 的精确树**（`reduce_common_ra_reuse_align_c310_impl.h`，见"完整代码"）：
   - a. `tailR = dimR − mainR > 0` 时先做 **tail-add**：`rows[r] := x[r] + x[mainR+r]`（r<tailR，逐行向量加，就地写回 src）；
   - b. **主折叠**：每轮 `loopRNum >>= 4`（BASE_FOLD），16 个寄存器 `vreg_k = 行(loopR + k·loopRNum)`，四级平衡树 L1:(k,k+8) → L2:(k,k+4) → L3:(k,k+2) → L4:(0,1)，结果写回行 loopR；
   - c. **尾折叠**按 `folds % 4` 选：foldOne（row0+row1）/ foldTwo（(r0+r2)+(r1+r3)）/ foldThree（8 行变体树）/ foldZero（直拷）。
6. **位级锚定验证**：用最小 shape 的真实 kernel 输出验证推导——**4 行 RA reduce = (x0+x2)+(x1+x3)**（交错配对，不是 (x0+x1)+(x2+x3)！），与 LessThanVL/foldTwo 推导一致，24 列位级全对。
7. **机制要点（为什么是交错）**：OverVL 主折叠中 vreg_k 装的是行 `loopR + k·loopRNum`（k=0..15）；dimR=32 时 loopRNum=2，即 vreg_k=行(loopR+2k)，L1 配对 (k,k+8) 等价于 (行 r, 行 r+16)。对 32 行：主折叠后 row0=偶行平衡树（配对序 {0,16},{8,24},{4,20},{12,28} 逐级合并）、row1=奇行平衡树，foldOne 再 row0+row1——即"偶/奇位分开、各自平衡树、最后相加"的交错结构。dimR=48 时先 tail-add（rows[0..16) := x[r]+x[r+32]），再对得到的 32 行跑同一棵树。

## 系统信息

- 工具链：CANN cann-9.0.0-beta.2（x86_64-linux host），目标 SoC Ascend950PR_9589（NpuArch=3510，AIV=dav-c310-vec，生效变体 dav_c310）。
- 关键文件：`$ASCEND_HOME_PATH/x86_64-linux/asc/impl/adv_api/detail/reduce/reduce_sum/reduce_sum_c310_impl.h`（调度）；`.../reduce/reduce_common_ra_reuse_align_c310_impl.h`（RA 树实现）；`.../reduce/reduce_common_util_c310_impl.h`（常量）；`.../reduce/reduce_common_util_impl.h`（FindClosestPowerOfTwo）。
- 验证脚本（已随证据提交仓库 `docs/<OP>/develop/checks/branch0_bias_mare_dumps/`）：`probe_hw_order.py`（候选序位级比对）、`sim_kernel.py`（hw_ra 函数族——树的可执行复刻）。

## 完整代码

CANN impl 头文件核心段（verbatim，`reduce_common_ra_reuse_align_c310_impl.h`，dav_c310；`Binaryfunc` 为加法、`isReuseSource=true` 路径）：

```cpp
// Process mainR and tailR（tail-add 阶段：tail 行并入 main 行）
for (uint16_t i = 0; i < needInplaceAdd; i++) {
    for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
        mask = Reg::UpdateMask<T, Trait>(inplaceA);
        for (uint16_t loopR = 0; loopR < static_cast<uint16_t>(tailR); loopR++) {
            Reg::LoadAlign(vregMain, srcAddr + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vregTail, srcAddr + loopA * vlSize + aTailOffset + loopR * dimA);
            Binaryfunc(vregMain, vregMain, vregTail, mask);
            Reg::StoreAlign(addr + loopA * vlSize + loopR * dimA, vregMain, mask);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// MainFolds need 16 register
Reg::RegTensor<T, Trait> vreg0;  // ... vreg1 ~ vreg15 同款声明

// Process main folds（16 宽平衡寄存器树）
uint16_t loopRNum = mainR;
for (uint16_t loopMain = 0; loopMain < mainTimes; loopMain++) {
    loopRNum = loopRNum >> avgFolds;
    uint16_t offsetR = loopRNum * dimA;
    uint32_t mainA = dimA;
    for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
        mask = Reg::UpdateMask<T, Trait>(mainA);
        for (uint16_t loopR = 0; loopR < loopRNum; loopR++) {
            // L0
            Reg::LoadAlign(vreg0, addr + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg1, addr + offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg2, addr + 2 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg3, addr + 3 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg4, addr + 4 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg5, addr + 5 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg6, addr + 6 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg7, addr + 7 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg8, addr + 8 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg9, addr + 9 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg10, addr + 10 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg11, addr + 11 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg12, addr + 12 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg13, addr + 13 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg14, addr + 14 * offsetR + loopA * vlSize + loopR * dimA);
            Reg::LoadAlign(vreg15, addr + 15 * offsetR + loopA * vlSize + loopR * dimA);
            // L1
            Binaryfunc(vreg0, vreg0, vreg8, mask);
            Binaryfunc(vreg1, vreg1, vreg9, mask);
            Binaryfunc(vreg2, vreg2, vreg10, mask);
            Binaryfunc(vreg3, vreg3, vreg11, mask);
            Binaryfunc(vreg4, vreg4, vreg12, mask);
            Binaryfunc(vreg5, vreg5, vreg13, mask);
            Binaryfunc(vreg6, vreg6, vreg14, mask);
            Binaryfunc(vreg7, vreg7, vreg15, mask);
            // L2
            Binaryfunc(vreg0, vreg0, vreg4, mask);
            Binaryfunc(vreg1, vreg1, vreg5, mask);
            Binaryfunc(vreg2, vreg2, vreg6, mask);
            Binaryfunc(vreg3, vreg3, vreg7, mask);
            // L3
            Binaryfunc(vreg0, vreg0, vreg2, mask);
            Binaryfunc(vreg1, vreg1, vreg3, mask);
            // L4
            Binaryfunc(vreg0, vreg0, vreg1, mask);
            Reg::StoreAlign(addr + loopA * vlSize + loopR * dimA, vreg0, mask);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// Process tail folds
for (uint16_t i = 0; i < foldOne; i++) {          // folds%4==1: row0+row1
    for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
        mask = Reg::UpdateMask<T, Trait>(tailA);
        Reg::LoadAlign(vreg0, addr + loopA * vlSize);
        Reg::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
        Binaryfunc(vreg0, vreg0, vreg1, mask);
        Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
    }
}

for (uint16_t i = 0; i < foldTwo; i++) {          // folds%4==2: (r0+r2)+(r1+r3)
    for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
        mask = Reg::UpdateMask<T, Trait>(tailA);
        Reg::LoadAlign(vreg0, addr + loopA * vlSize);
        Reg::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
        Reg::LoadAlign(vreg2, addr + 2 * dimA + loopA * vlSize);
        Reg::LoadAlign(vreg3, addr + 3 * dimA + loopA * vlSize);
        Binaryfunc(vreg0, vreg0, vreg2, mask);
        Binaryfunc(vreg1, vreg1, vreg3, mask);
        Binaryfunc(vreg0, vreg0, vreg1, mask);
        Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
    }
}

for (uint16_t i = 0; i < foldThree; i++) {        // folds%4==3: 8 行变体树
    for (uint16_t loopA = 0; loopA < loopANum; loopA++) {
        mask = Reg::UpdateMask<T, Trait>(tailA);
        Reg::LoadAlign(vreg0, addr + loopA * vlSize);
        Reg::LoadAlign(vreg1, addr + dimA + loopA * vlSize);
        Reg::LoadAlign(vreg2, addr + 2 * dimA + loopA * vlSize);
        Reg::LoadAlign(vreg3, addr + 3 * dimA + loopA * vlSize);
        Reg::LoadAlign(vreg4, addr + 4 * dimA + loopA * vlSize);
        Reg::LoadAlign(vreg5, addr + 5 * dimA + loopA * vlSize);
        Reg::LoadAlign(vreg6, addr + 6 * dimA + loopA * vlSize);
        Reg::LoadAlign(vreg7, addr + 7 * dimA + loopA * vlSize);
        Binaryfunc(vreg0, vreg0, vreg4, mask);
        Binaryfunc(vreg1, vreg1, vreg5, mask);
        Binaryfunc(vreg2, vreg2, vreg6, mask);
        Binaryfunc(vreg3, vreg3, vreg7, mask);
        Binaryfunc(vreg0, vreg0, vreg2, mask);
        Binaryfunc(vreg1, vreg1, vreg3, mask);
        Binaryfunc(vreg0, vreg0, vreg1, mask);
        Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
    }
}

// Reduce to 1（folds%4==0: 直拷）
for (uint16_t i = 0; i < foldZero; i++) {
    for (uint16_t loopA = 0; loopA < loopANumFinal; loopA++) {
        mask = Reg::UpdateMask<T, Trait>(processA);
        Reg::LoadAlign(vreg0, addr + loopA * vlSize);
        Reg::StoreAlign(dstAddr + loopA * vlSize, vreg0, mask);
    }
}
```

OverVL 包装层参数推导（同文件 `ReduceRAOverVLImpl`，verbatim 要点）：

```cpp
uint32_t mainR = ReduceOpInternal::CalculateMainR(dimR, false, vlSize);
uint32_t tailR = dimR - mainR;
// mainR==1 时直接 ReduceCopyOutImpl 返回
// 注意：仅 !isReuseSource 时才有 tailR==0 的 mainR 减半特例：
if constexpr (!isReuseSource) {
    if (tailR == 0 && mainR > 1) { mainR = mainR / 2; tailR = mainR; }
}
uint16_t folds = ReduceOpInternal::CalculateFolds(mainR);
uint16_t avgFolds = ReduceOpInternal::BASE_FOLD;          // = 4
uint16_t tailFolds = folds % avgFolds;
uint16_t foldZero = (tailFolds == 0) ? 1 : 0;
uint16_t foldOne = (tailFolds == ReduceOpInternal::FOLD_ONE) ? 1 : 0;    // ==1
uint16_t foldTwo = (tailFolds == ReduceOpInternal::FOLD_TWO) ? 1 : 0;    // ==2
uint16_t foldThree = (tailFolds == ReduceOpInternal::FOLD_THREE) ? 1 : 0; // ==3
```

Python 可执行复刻（hw_ra 函数族，verbatim 取自证据脚本 sim_kernel.py；逐位语义用 fp32 强转保持）：

```python
def hw_ra(x, dimA, dimR):
    """Hardware ReduceSum RA, isReuseSource=true, T=float (4B), dav_c310.
    x: (dimR, cols) fp32. Returns (cols,) fp32 per exact register tree."""
    x = x.copy()
    if dimA <= 32 or dimA > 65535:
        return _concat(x, dimA, dimR)
    if dimA <= 64:
        return _less_than_vl(x, dimA, dimR)
    return _over_vl(x, dimA, dimR)


def _tail_add(x, mainR, tailR):
    # rows[0..tailR) := x[r] + x[mainR+r]  (in place, fp32)
    for r in range(tailR):
        x[r] = (x[r] + x[mainR + r]).astype(np.float32)


def _reg16_tree(v):
    # L1: (k,k+8); L2: (k,k+4); L3: (k,k+2); L4: (0,1)
    v = list(v)
    for k in range(8):
        v[k] = (v[k] + v[k + 8]).astype(np.float32)
    for k in range(4):
        v[k] = (v[k] + v[k + 4]).astype(np.float32)
    for k in range(2):
        v[k] = (v[k] + v[k + 2]).astype(np.float32)
    return (v[0] + v[1]).astype(np.float32)


def _over_vl(x, dimA, dimR):
    if dimR == 1:
        return x[0].copy()
    mainR = 1 << (dimR.bit_length() - 1)
    tailR = dimR - mainR
    if tailR > 0:
        _tail_add(x, mainR, tailR)
    folds = mainR.bit_length() - 1
    avg = 4
    mainTimes, tailFolds = divmod(folds, avg)
    loopRNum = mainR
    for _ in range(mainTimes):
        loopRNum >>= avg
        for loopR in range(loopRNum):
            v = [x[loopR + k * loopRNum] for k in range(16)]
            x[loopR] = _reg16_tree(v)
    return _tail_folds(x, tailFolds)


def _less_than_vl(x, dimA, dimR):
    if dimR == 1:
        return x[0].copy()
    mainR = 1 << (dimR.bit_length() - 1)
    tailR = dimR - mainR
    if tailR > 0:
        _tail_add(x, mainR, tailR)
    folds = mainR.bit_length() - 1
    avg = 4
    mainTimes, tailFolds = divmod(folds, avg)
    loopRNum = mainR
    for _ in range(mainTimes):
        loopRNum >>= avg
        for loopR in range(loopRNum):
            base = loopR * 16
            v = [x[base + k] for k in range(16)]
            x[loopR] = _reg16_tree(v)
    return _tail_folds(x, tailFolds)


def _concat(x, dimA, dimR):
    if dimR == 1:
        return x[0].copy()
    if dimR == 2:
        return (x[0] + x[1]).astype(np.float32)
    foldTime = dimR.bit_length() - 1  # floor(log2(dimR))
    mainR = 1 << foldTime
    foldTime -= 1
    tailR = dimR - mainR
    if tailR > 0:
        _tail_add(x, mainR, tailR)
    for _ in range(foldTime):
        mainR >>= 1
        for j in range(mainR):
            x[j] = (x[j] + x[j + mainR]).astype(np.float32)
    return (x[0] + x[1]).astype(np.float32)


def _tail_folds(x, tailFolds):
    if tailFolds == 0:
        return x[0].copy()
    if tailFolds == 1:
        return (x[0] + x[1]).astype(np.float32)
    if tailFolds == 2:
        a = (x[0] + x[2]).astype(np.float32)
        b = (x[1] + x[3]).astype(np.float32)
        return (a + b).astype(np.float32)
    v = [x[k] for k in range(8)]
    return _reg16_tree(v)
```

## 设计决策

- 推导路径选"读生效变体 impl 源码"而非继续盲试候选：15 个候选全不匹配后候选空间发散（交错/蝶形/分组×组合爆炸），读码是收敛路径；B=4 小 shape 锚定使推导可信（4 行只有 5 种配对方式，位级判定唯一）。
- 仿真只建模 isReuseSource=true 路径（本算子调用形态）；头文件中 `!isReuseSource` 的 tailR==0 减半特例保留在包装层摘录中但未验证。
- 该知识属 recommended_fastpath：位级预测 ReduceSum 时先读码后锚定，不要猜。

## 验证信号

来源说明：以下为实机 kernel dump 与推导序的位级比对（分析在 x86 host 用 numpy 逐位复刻硬件树；脚本随证据提交仓库）。

- 4 行 RA reduce = (x0+x2)+(x1+x3)：与 kernel 输出位级相等（24 列全对；其余 5 种候选 mismatch 8~11/24）（来源平台：950；证据等级：tool_observed）
- 32 行 RA：15 个常规候选全部 mismatch（3788~4675/6144），排除顺序/折半树/2~16 分组/stride-16 配对类候选（来源平台：950；证据等级：tool_observed）
- 推导树（tail-add + 16 宽平衡树 + fold 尾处理）作为全链路仿真器组件，在 4 个失败用例 × 2 路 bias 输出上与 kernel 位级一致 8/8（来源平台：950；证据等级：tool_observed）
- 生效变体确认：Ascend950PR → NpuArch=3510 → AIV=dav-c310-vec → variant_dir=dav_c310（来源平台：950 工程环境；证据等级：tool_observed）

## 失效边界

- 绑定 CANN 9.0.0-beta.2 的 dav_c310 impl（NpuArch 3510）；CANN 升级或换变体（dav_c220 等）须重读 impl 并重新锚定。
- 只验证了 isReuseSource=true + Aligned（lane 字节数 32B 对齐）+ fp32（vlSize=64）；UnAligned 分支、fp16/bf16（vlSize 不同）、dimA≤32 的 Concat 树、AR pattern 均未验证。
- dimR>64（需要多轮 mainTimes 折叠）的树未实测（本例 dimR≤48 单轮折叠）；`CalculateMainR` 对 dimR≥2^16 的行为（MAX_OFFSET=16 截断）未验证。

## 坏实践

- 不确认生效变体直接 grep `$ASCEND_HOME_PATH`，读到别的 dav_* 头文件就下结论。
- 拿 kernel 注释或单一 impl 目录的写法当"硬件不支持某模式"的证据。
- 盲试候选序（本例 15 个候选全败）——位级判定只有 0/1，"最接近"（如 stride16 配对树 mismatch 3788/6144）不等于正确。

## 误诊

- 初期假设硬件树是 stride-16 蝶形（stride16_pair__tree 是 15 个候选中"最接近"的）——位级比对证伪；正确树来自 impl 源码推导 + 4 行锚定。

## 预防

- 需要对 ReduceSum 做位级预测的场景（位级 golden/归约仿真/ULP 定界），把"读生效变体 impl + 4 行小 shape 锚定"固化为标准两步。
- 引用该知识时绑定 CANN 版本（impl 头文件路径含版本目录）。

## 未决事项

- UnAligned 分支（lane 非 32B 对齐）与 Concat 分支（dimA≤32）的精确树未推导验证。
- fp16/bf16 dtype 的 vlSize 与树形未验证。
