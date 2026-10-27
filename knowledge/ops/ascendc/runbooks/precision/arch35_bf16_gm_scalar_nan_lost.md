---
type: "Runbook"
title: "arch35 bf16 标量 GM 读写丢 nan 位模式"
description: "arch35 上 GlobalTensor<bfloat16_t>::GetValue/SetValue（__bf16 标量 load/store）把 bf16 nan 位模式 0x7FC0 读成 0（0x0000/0x8000），bf16 add 特殊值用例 nan 传播全部失效而 amax/amin 因比较恒 false 侥幸通过——修复是 bf16 数据全程 GlobalTensor<uint16_t> 位模式读写（.template ReinterpretCast），kernel 内不物化 __bf16 标量。"
tags: [task.debugging, topic.precision, topic.compute.elementwise.type_conversion]
status: stable
sources: []
created_at: '2026-09-17T00:48:21Z'
updated_at: '2026-09-17T00:48:21Z'
platforms: ["950"]
---

# arch35 bf16 标量 GM 读写丢 nan 位模式

## 现象

scatter 类自定义算子（Ascend950PR / arch35，CANN 9.1.0，三段式 reduce=add/multiply/amin/amax，data/updates/y 逐元素读-改-写）在全 dtype ST 实测中，bf16 + reduce=add + 特殊值注入用例（CSV 用例 L1_112：data/updates 按 [inf, -inf, nan, 0, -0, inf, nan] 周期注入，shape [16,20]/[10,12]，dim=0）失败，失败模式非常有辨识度：

```
[Real] L1_112: special_val: bfloat16 reduce=add (inf/nan/0 in data) ...
  [FAIL] MERE=2.50e+08 (>= 7.81e-03), MARE=1.00e+10 (>= 7.81e-02), bfloat16
    不匹配 [3]: golden=nan(0x7fc0), actual=-0.000000(0x8000), rel_err=1.00e+10
    不匹配 [84]: golden=nan(0x7fc0), actual=-0.000000(0x8000), rel_err=1.00e+10
    不匹配 [91]: golden=nan(0x7fc0), actual=0.000000(0x0000), rel_err=1.00e+10
    不匹配 [140]: golden=nan(0x7fc0), actual=-0.000000(0x8000), rel_err=1.00e+10
    不匹配 [142]: golden=nan(0x7fc0), actual=0.030762(0x3cfc), rel_err=1.00e+10
```

对照关系（hex 位模式来自测试脚手架按 2 字节 memcpy 读出）：

- golden 全是 0x7fc0（bf16 quiet nan）；
- actual 是 0x8000（-0）、0x0000（+0）或 0x3cfc（普通值）——nan 位模式在 kernel 数据通路上变成了零/普通值；
- 同一算子 bf16 常规值用例全通过，bf16 amax/amin 特殊值用例也通过（只有 add 失败）；
- fp16/fp32 的 add/amax/amin 特殊值用例全部通过。

当时 kernel 的 bf16 路径（经位级转换修复编译问题后的版本）是：`GlobalTensor<bfloat16_t>` 直接 `GetValue/SetValue` 读改写，标量侧用位级 `Bf16ToFloat/FloatToBf16`（`__builtin_memcpy` 位复制）做算术。

## 适用场景

- 工程形态：AscendC kernel 对 GM 上的 bf16 张量做逐元素读-改-写（scatter/scatter_add 类、任何需要 nan/inf 位模式精确传递的算子），数据含特殊值（nan/inf/±0）且 golden 按位或按值严格比对。
- API 路径：`GlobalTensor<T>::GetValue/SetValue`（arch35 上走 `__inout_pipe__(S)` 标量管线，bf16 走 `*addr` 直接 `__bf16` load/store 分支）；`GlobalTensor<T>::ReinterpretCast<uint16_t>()`。
- 验证过的平台与 CANN 版本：Ascend950PR（DAV_3510，arch35）实测，CANN 9.1.0。
- 适用粒度：platform_wide——arch35 上一切经 `GlobalTensor<bfloat16_t>` 标量通路的 nan 传递。
- 路线优先级：recommended_fastpath（位模式通路是验证过的正确路径）。
- 明确不适用 / 未验证场景：fp16/fp32/int 标量通路（实测 nan 传递正常）；LocalTensor 向量通路；非特殊值数据（无 nan 可丢，直接通路即可）。

## 根因

arch35 上 `GlobalTensor<bfloat16_t>::GetValue/SetValue` 的 `else` 分支直接 `__gm__ PrimType* addr ...; return *addr;` / `*addr = value;`，即硬件 `__bf16` 标量 load/store。实测该通路会把 bf16 nan 位模式（0x7FC0）变成 0——失配位置的 golden=0x7fc0 而 actual=0x8000/0x0000 直接证明了位模式在 kernel 数据通路中被改写。由于 amax/amin 走比较（`(updF > oldF)` 对 nan 恒 false，返回另一个操作数）恰好能在部分分布下与 golden 巧合吻合，而 add 必须 `x + nan = nan` 精确传播，所以只有 add 用例暴露此问题。

需要如实说明：nan 具体在哪一环被改写（`__bf16` 标量 load、寄存器传递、还是 store）未做指令级隔离；被证伪路径与被证实路径的差别是——被证实的最终配置让 bf16 数据全程以 `uint16_t` 位模式流动，kernel 内从不物化 `__bf16` 标量。

## 解决配方

对 bf16 模板实例，Phase1 拷贝与 Phase2 读改写全部改用 `uint16_t` 位模式视图（helper 的位级转换见同域 bf16 标量位布局卡）：

```cpp
// Phase 1: data → y 拷贝，bf16 用 uint16_t 读写位模式
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
    if constexpr (std::is_same_v<D_T_DATA, bfloat16_t>) {
        // bf16: 用 uint16_t 读写位模式，绕过 __bf16 标量加载/存储
        auto dataGmU16 = dataGm_.template ReinterpretCast<uint16_t>();
        auto yGmU16 = yGm_.template ReinterpretCast<uint16_t>();
        for (int64_t i = start; i < end; ++i) {
            yGmU16.SetValue(i, dataGmU16.GetValue(i));
        }
    } else {
        for (int64_t i = start; i < end; ++i) {
            yGm_.SetValue(i, dataGm_.GetValue(i));
        }
    }
    PipeBarrier<PIPE_ALL>();
}
```

位级转换与聚合 helper（在 float 上运算，amax/amin 比较顺序与 `std::max/std::min` 的 NaN 语义一致）：

```cpp
static __aicore__ inline float Bf16BitsToFloat(uint16_t bits)
{
    uint32_t fBits = static_cast<uint32_t>(bits) << 16;
    float f;
    __builtin_memcpy(&f, &fBits, sizeof(float));
    return f;
}
static __aicore__ inline uint16_t FloatToBf16Bits(float f)
{
    uint32_t fBits;
    __builtin_memcpy(&fBits, &f, sizeof(uint32_t));
    uint16_t bits = static_cast<uint16_t>(fBits >> 16);
    // round-to-nearest-even
    uint32_t roundBit = (fBits >> 15) & 1U;
    uint32_t sticky = (fBits & 0x7FFFU) != 0U ? 1U : 0U;
    if (roundBit && (sticky || (bits & 1U))) {
        bits = static_cast<uint16_t>(bits + 1U);
    }
    return bits;
}
static __aicore__ inline float ScalarOpFloat(float oldF, float updF, uint8_t mode)
{
    switch (mode) {
        case 1: return oldF + updF;                              // add
        case 2: return oldF * updF;                              // multiply
        case 3: return (updF < oldF) ? updF : oldF;             // amin (等价 std::min(oldF, updF))
        case 4: return (updF > oldF) ? updF : oldF;             // amax (等价 std::max(oldF, updF))
        default: return updF;                                    // None
    }
}
```

`ProcessElement` 的 bf16 分支同样先 `updatesGm_.template ReinterpretCast<uint16_t>()` / `yGm_.template ReinterpretCast<uint16_t>()` 取位模式，再 `Bf16BitsToFloat → ScalarOpFloat → FloatToBf16Bits → SetValue(uint16_t)`（reduce=None 时直接 `yGmU16.SetValue(targetOffset, updateBits)` 拷位模式）。交付版还在 add/multiply 分支加了显式 nan 位检查（任一操作数是 nan 位模式则直接写 0x7FC0）作为双保险。

两个配套要点：

1. 模板类内必须写 `.template ReinterpretCast<uint16_t>()`（缺 `template` 关键字会编译失败，且被增量构建掩盖成"修复无效"，本卡的修复曾因此空转四轮）；
2. 修复后必须 `--make_clean` 重建并核对 kernel `.o` 时间戳，确认新代码进了二进制。

## 设计决策

- 为什么选"全程 uint16 位模式"而不是只换算术层：hex 证据表明 nan 丢失发生在 bf16 标量数据通路（load/store），仅改 `Bf16ToFloat` 输入侧（union 双关）不足以保证 GM 进出的位模式完整；让 bf16 在 kernel 内自始至终以 uint16 流动是把"物化 __bf16 标量"这个风险点整体移除。
- 备选方案：LocalTensor + DataCopyPad 批量搬运 bf16——该算子 Phase1 曾实测 arch35 上 DataCopyPad UB→GM 不可用（前序迭代 issue），且散布写路径天然是标量，未采用。
- 备选方案：显式 `x != x` nan 检测（`oldF != oldF`）加在 float 层——方向正确但当时被陈旧二进制掩盖未获验证，最终交付代码保留为双保险。
- 当前方案路线档位：recommended_fastpath。

## 系统信息

- 硬件/软件：Ascend950PR（DAV_3510，arch35）实测，CANN 9.1.0，bisheng 编译器；ST 在真实 NPU 上运行（npu-smi 显示 Ascend950PR）。
- 相关文件：`op_kernel/arch35/bench_no_harness_scatter.h`（Phase1CopyData/ProcessElement/ScalarOp/Bf16BitsToFloat/FloatToBf16Bits）；`tests/st/test_aclnn_scatter.cpp`（hex 位模式打印脚手架）；`tests/st/testcases/L1_test_cases.csv`（L1_112/113/114 用例定义）。

## 预防

- arch35 上凡 bf16 数据需要特殊值语义（nan/inf 精确传递）的 kernel，默认按 uint16 位模式通路实现，`GlobalTensor<bfloat16_t>` 标量 GetValue/SetValue 只用于确认无特殊值的数据。
- 精度比对脚手架对 fp16/bf16 失配位置打印原始 hex 位模式（2 字节 memcpy），把"数值错"与"位模式错"区分开——这是本问题能在两轮内锁定的关键。

## 验证信号

以下信号来自 bf16 特殊值用例的修复会话：hex 位模式对照锁定"nan 变 0"，切到位模式通路（含 template 关键字修复与干净重建）后三类特殊值用例全部转绿。

- 修复前：L1_112 golden=nan(0x7fc0) vs actual=0x8000/0x0000/0x3cfc；bf16 常规值用例与 bf16 amax/amin 特殊值用例通过、fp16/fp32 全通过（来源平台：950 实测；证据等级：tool_observed）。
- 修复过程：kernel 内 y[0] 探针返回 0x0001/0x0004 证明 reduceMode 属性传递无误，排除 executor 缓存假设（来源平台：950 实测；证据等级：tool_observed）。
- 修复后：L1_112/113/114（bf16 add/amax/amin 特殊值）全部 `MERE=0.00e+00, MARE=0.00e+00`；全量 ST Real 875/875、Mock 875/875、UT 17/17（来源平台：950 实测；证据等级：tool_observed）。

## 失效边界

- nan 丢失的精确环节（`__bf16` 标量 load / 寄存器 / store）未隔离，结论以"整条 bf16 标量通路不可信、uint16 通路已验证"的形式成立。
- 仅 arch35 实测；其他架构 `GlobalTensor<bfloat16_t>` 标量通路的 nan 行为未验证（未验证，推断可用同法位模式通路规避）。
- 位模式通路对"值语义正确但位模式不同"的合法场景（如多个 nan payload）会保位而不是规范化——golden 若做 nan 规范化比较需另行适配。

**未决项**：

- 优先项：复现并指令级定位 nan 改写点（反汇编最小 kernel：GM→`__bf16` 寄存器→store 各插桩），确认是硬件 load/store 行为还是编译器降级。
- 向工具链侧确认 `GlobalTensor<bfloat16_t>` 标量通路对特殊值是否有官方结论/替代 API。
## 坏实践

- 在 arch35 上用 `GlobalTensor<bfloat16_t>::GetValue/SetValue` 传递 nan/inf 语义数据。
- 用"amax/amin 通过"推断 bf16 通路完好——比较类运算对 nan 不敏感，会掩盖通路缺陷（本例正是只有 add 暴露）。
- 修复后不做干净重建就下结论"方案无效"。

## 误诊

- 依次尝试过并被（错误地）判定无效的方案：union 类型双关读位、`ReinterpretCast` 位模式通路、float 层 `oldF != oldF` nan 检测、uint16 位模式判 nan——四轮"无效"的真实原因是 kernel 编译失败被陈旧二进制掩盖（`ReinterpretCast` 缺 `template` 关键字），方案本身方向正确。
- 一度怀疑 executor 缓存使 reduce 属性退化为 None（覆盖写），被 kernel 内 reduceMode 探针（0x0001）排除。
