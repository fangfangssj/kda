---
type: "Runbook"
title: DataCopyPad blockLen uint16_t 窄化截断风险
description: DataCopyParams 四字段均为 uint16_t，blockLen=count*sizeof(T) 由 64 位乘法隐式窄化；-Wall/-Wextra 不告警、-Wconversion 才报，需按 UB 规划分母演算 blockLen 是否超 65535，超限应换 DataCopyExtParams。
tags: [task.testing, topic.data_movement.data_copy, topic.memory]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T01:47:15Z'
platforms: ["950"]
---

# DataCopyPad blockLen uint16_t 窄化截断风险

## 现象

对已通过 36/36 NPU 精度用例的自定义逐元素算子做 C++ 安全检视时，在 kernel 头文件 `op_kernel/arch35/<op>.h` 的 `CopyIn`/`CopyOut` 里发现如下模式（`count` 形参为 `uint64_t`，`sizeof(T)` 为 `size_t`）：

```cpp
template <typename T, int COMPUTE_BRANCH>
__aicore__ inline void KernelBenchNoHarnessExp<T, COMPUTE_BRANCH>::CopyIn(
    uint64_t progress, uint64_t count)
{
    LocalTensor<T> xLocal = inQueueX_.template AllocTensor<T>();
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = count * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[progress * ubFormer_], copyParams, {false, 0, 0, 0});
    inQueueX_.EnQue(xLocal);
}
```

`CopyOut` 同型（`DataCopyPad(yGm_[progress * ubFormer_], yLocal, copyParams)`）。查 API 文档确认 `DataCopyParams` 的 `blockLen` 是 `uint16_t`（上限 65535），而 `count * sizeof(T)` 是 64 位运算——存在隐式窄化；一旦乘积超 65535 会静默截断，得到错误的搬运长度。用编译器探针复核：`-Wall -Wextra` 下无任何告警，`-Wconversion` 下才报出。数值演算表明该算子当前两条路径的 blockLen 均在安全范围内，属**当前未触发的潜在风险**（本卡如实标注：截断在本次开发中未实际发生）。

## 适用场景

- 工程形态：AscendC kernel 用 `DataCopyPad` + `DataCopyParams` 做 GM↔UB 搬运，`blockLen` 由 `count * sizeof(T)`（64 位）计算；代码检视 / 新写 kernel 自查 / Buffer 规划调整后回归。
- API 路径：`DataCopyParams`（四字段全 `uint16_t`）与 `DataCopyExtParams`（`blockLen` 为 `uint32_t`，文档范围 [1, 2097151]）两套搬运参数结构；`DataCopyPad` 的 Local↔GM 重载（单位语义区分见[DataCopy blockLen 以 32B 块计非字节卡](../precision/datacopyparams_block_len_32b_block_units.md)——`DataCopyPad` 搭配 `DataCopyParams` 时 blockLen 按字节解释）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致；探针为宿主机 g++ 13.3.0 编译检查 + 演算，被检 kernel 在 950 上板运行通过 36/36 用例）。
- 适用粒度：platform_wide——参数结构宽度是 API 定义层约束；但 65535 边界的具体数值演算依赖平台 UB 容量（本卡演算基于 950 的 253952B UB）。
- 路线优先级：safety_fallback（当前 `DataCopyParams` + 演算证明安全为已验证可用路线；`DataCopyExtParams` 替换为建议的更稳路线，未实施）。
- 明确不适用 / 未验证场景：`DataCopyExtParams`（blockLen uint32_t）路径无此边界；blockCount/srcStride/dstStride 同为 uint16_t 但通常为常数或小值，未逐一演算；非 950 平台的 UB 容量不同，数值结论不直接搬用。

## 判别方法

**触发条件**：kernel 里 `DataCopyParams.blockLen`（或 blockCount/stride）的赋值右侧是更宽类型（`uint64_t count * sizeof(T)`、`size_t` 表达式），且 `count` 上界由 UB Buffer 规划分母决定。

**可执行探针一（编译期确认窄化存在，宿主机 g++ 即可）**：

```bash
cat << 'EOF' | g++ -x c++ -std=c++17 -fsyntax-only -Wconversion - 2>&1 || true
#include <cstdint>
#include <cstddef>
struct DataCopyParams {
    uint16_t blockCount;
    uint16_t blockLen;
    uint16_t srcStride;
    uint16_t dstStride;
};
int main() {
    uint64_t count = 12672;
    size_t sizeof_T = 4;
    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = count * sizeof_T;  // narrowing: uint64_t -> uint16_t
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    return 0;
}
EOF
```

原始观测（`-Wall -Wextra` 同型片段**不报**，`-Wconversion` 报）：

```
<stdin>:14:33: warning: conversion from 'uint64_t' {aka 'long unsigned int'} to 'uint16_t' {aka 'short unsigned int'} may change value [-Wconversion]
```

**可执行探针二（数值边界演算：blockLen 是否可能超 65535）**——用 kernel 的 UB 规划分母推 `ubFormer` 上界再乘元素大小（本算子实测：UB=253952B，FP32 路径分母 20、8 对齐；FP16/BF16 路径分母 16、16 对齐）：

```bash
python3 -c "
ub_size = 253952
# FP32 path
buffer_divisor_fp32 = 20
ub_former_fp32 = (ub_size // buffer_divisor_fp32) // 8 * 8  # FloorAlign to 8
blocklen_fp32 = ub_former_fp32 * 4
print(f'FP32: ubFormer={ub_former_fp32}, blockLen={blocklen_fp32}, uint16_max=65535, safe={blocklen_fp32 < 65535}')

# FP16/BF16 path
buffer_divisor_fp16 = 16
ub_former_fp16 = (ub_size // buffer_divisor_fp16) // 16 * 16  # FloorAlign to 16
blocklen_fp16 = ub_former_fp16 * 2
print(f'FP16: ubFormer={ub_former_fp16}, blockLen={blocklen_fp16}, uint16_max=65535, safe={blocklen_fp16 < 65535}')
"
```

原始观测：

```
FP32: ubFormer=12696, blockLen=50784, uint16_max=65535, safe=True
FP16: ubFormer=15872, blockLen=31744, uint16_max=65535, safe=True
```

**解释**：`DataCopyParams.blockLen` 上限 65535；当前两条路径分别用到 50784 / 31744，余量约 1.29× / 2.07×。风险来自 Buffer 规划变更的方向性：分母变小（如去掉 DoubleBuffer 或合并临时缓冲）会线性放大 `ubFormer`——FP32 路径分母从 20 降到 16 时 `ubFormer` ≈ 15872、blockLen ≈ 63504 已贴近 65535；再降即越界静默截断，搬运长度错误（数据损坏/越界的方向取决于截断值与实际长度的关系）。`DataCopyExtParams.blockLen` 为 `uint32_t`（文档范围 [1, 2097151]），无此边界。

## 解决配方

本轨迹未实施修复（检视产出 issue，状态"待修复"）。检视给出的两条修复路线，按优先序：

方案一（推荐，换宽类型结构）：

```cpp
// 修改前
DataCopyParams copyParams;
copyParams.blockCount = 1;
copyParams.blockLen = count * sizeof(T);
copyParams.srcStride = 0;
copyParams.dstStride = 0;
DataCopyPad(xLocal, xGm_[progress * ubFormer_], copyParams, {false, 0, 0, 0});

// 修改后
DataCopyExtParams copyParams;
copyParams.blockCount = 1;
copyParams.blockLen = count * sizeof(T);
copyParams.srcStride = 0;
copyParams.dstStride = 0;
copyParams.rsv = 0;
DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
DataCopyPad(xLocal, xGm_[progress * ubFormer_], copyParams, padParams);
```

方案二（保留 `DataCopyParams`，加范围校验后显式收窄）：

```cpp
constexpr uint32_t MAX_BLOCK_LEN = 65535;
OP_CHECK_IF(count * sizeof(T) > MAX_BLOCK_LEN,
            /* error handling */, /* return */);
copyParams.blockLen = static_cast<uint16_t>(count * sizeof(T));
```

## 设计决策

- 为什么"当前不改"也成立：两条路径的 blockLen 均有实测余量，且 Buffer 规划分母是显式常量（改动需要人主动为之）；本次按检视流程落 issue 跟踪而非热修。
- 为什么推荐 `DataCopyExtParams` 而不是只加校验：宽类型从结构上消除边界，校验方案把失败路径留给运行期错误处理（kernel 侧错误处理本身有代价）；代价是 Ext 参数结构与 Pad 参数要配对使用（`DataCopyPadExtParams<T>`），写法略长。
- 探针为什么用独立 g++ 片段而不是直接编 kernel：窄化告警只依赖结构定义与赋值表达式，宿主机片段即可复现，且 `-Wconversion` 不在生产编译 flags 里（这正是它长期沉默的原因）。
- 当前方案路线档位：safety_fallback（`DataCopyParams` + 演算安全 = 已验证可用；`DataCopyExtParams` = 推荐升级路线，未实施）。

## 预防

- kernel 里所有 `DataCopyParams` 字段赋值检查右侧表达式宽度；CI/自查用 `-Wconversion` 编译（`-Wall -Wextra` 抓不到这类窄化）。
- Buffer 规划调整（分母变化）后重跑 blockLen 边界演算：`ubFormer × sizeof(T) < 65535` 必须作为不变量维持。
- 大块搬运（分母小、元素大）直接选 `DataCopyExtParams`。

## 验证信号

来源说明：本结论来自该算子上板通过后的 C++ 安全检视：API 文档核对 + 宿主机编译探针 + 数值演算，结论为"窄化存在、当前值安全"。

- 探针一：`-Wconversion` 报 `conversion from 'uint64_t' to 'uint16_t' may change value`；`-Wall -Wextra` 同型片段无告警（来源平台：950（kernel 目标平台，探针在宿主机 g++ 13.3.0 执行）；证据等级：tool_observed）。
- 探针二：FP32 blockLen=50784 < 65535、FP16/BF16 blockLen=31744 < 65535，当前安全（来源平台：950（UB=253952B 平台参数）；证据等级：tool_observed）。
- 被检 kernel 同期在 NPU 上 36/36 精度用例通过，佐证当前 blockLen 未截断（来源平台：950；证据等级：tool_observed）。

修复方案（DataCopyExtParams 替换）在本轨迹未实施、未验证，不作为已验证信号。

## 失效边界

- 本卡是**诊断规则**：截断故障在本次开发中未实际发生，所有"安全"结论只覆盖检视时的 Buffer 规划分母（FP32=20、FP16=16）与 950 的 253952B UB；规划变更后必须重算。
- `DataCopyExtParams.blockLen` 文档范围 [1, 2097151]，超过 2097151 的问题超出本卡讨论。
- 演算假设 `count` 上界 = `ubFormer`（整块搬运）；若存在按尾块（tail）传更小 count 的调用路径，上界仍被整块路径决定，结论不变。
- 其他平台 UB 容量与对齐粒度不同，数值结论不可直接搬用（未验证，推断约束形式一致）。

**未决项**：

- 优先项：在受控用例里实际触发一次越界（放大 `ubFormer` 或直接构造超长 blockLen），验证截断后的故障形态（错误搬运长度/越界），把风险从推断变为实测。
- `DataCopyExtParams` 替换后的性能与行为回归未做（搬运语义等价性按文档推断）。
## 坏实践

- 依赖 `-Wall -Wextra` 判定"无窄化"：默认告警集不含此类转换，本例实测沉默。
- 在 `blockLen` 赋值处直接 `static_cast<uint16_t>(...)` 而不加上界校验：只是把隐式截断变显式截断，错误长度照样进 DMA。

## 误诊

- 无。

## 复现增量：真实触发实例（关闭本卡优先遗留项）——87380 截断为 21844

另一独立轨迹并入 gather 算子（Ascend950PR/arch35，CANN 9.1.0，另一 trace）的**实际触发**实例，把本卡"截断在本次开发中未实际发生"的潜在风险升级为实测故障，并验证了本卡推荐的 `DataCopyExtParams` 修复路线：

**触发链（全部实测/源码查证）**：tiling 把 ubFactor（每批元素数）从约 3968 提升到约 31728 后，大 shape 用例（index 1048576 元素）的 `CopyIndexBatch` 中 `blockLen = currentNum × indexDtypeSize = 21845 × 4 = 87380` 字节，赋给 `uint16_t` 字段时静默截断为 `87380 % 65536 = 21844`（实际只搬运约 1/4 的 index 字节）；UB 中剩余 index 是未初始化垃圾值，kernel 用垃圾 index 计算 `srcOffset` 导致 GM 越界访问——部分用例 `流同步失败: 507035`，未崩溃的用例表现为大面积 mismatch（910230/1048576，约 87% 元素错）。`static_cast<uint32_t>(currentNum * indexDtypeSize)` 不阻止向 `uint16_t` 字段赋值时的截断——cast 只改表达式类型，反而让代码"看起来"安全。

**隐蔽性与隔离对照**：迭代一/二没暴露是因为旧 tiling 预留了不存在的 x 缓冲导致 ubFactor 只有 ~3968（blockLen=15872 < 65535 恰好不溢出）——保守的 tiling 掩盖了 API 宽度缺陷，任何 tiling 优化都会引爆它。隔离验证：把 tiling 临时回退到旧公式后，同批 54 条用例全部通过（证明失败由 ubFactor 提升触发，而非用例本身或 UB 总量溢出）。

**修复（本卡推荐路线的落地，verbatim）**：

```cpp
template <typename D_T_X, int BUFFER_MODE>
__aicore__ inline void BenchNoHarnessGather<D_T_X, BUFFER_MODE>::CopyIndexBatch(
    const LocalTensor<uint8_t>& indexLocal, int64_t batchIdx, int64_t currentNum,
    int64_t indexDtypeSize)
{
    // CopyIn: index GM → UB (MTE2, 连续读)
    // 使用 DataCopyExtParams（blockLen 为 uint32_t），避免 DataCopyParams.blockLen
    // （uint16_t, max 65535）在大 batch 场景下溢出
    DataCopyExtParams idxCopyParams;
    idxCopyParams.blockCount = 1;
    idxCopyParams.blockLen = static_cast<uint32_t>(currentNum * indexDtypeSize);
    idxCopyParams.srcStride = 0;
    idxCopyParams.dstStride = 0;
    DataCopyPad(indexLocal, indexGm[batchIdx * ubFactor_ * indexDtypeSize],
                idxCopyParams, {true, 0, 0, 0});
}
```

修复后核算 `DataCopyExtParams.blockLen` 自身上限 2097151 字节：本例最大 `blockLen = ubFactor × indexDtypeSize ≈ 237440 < 2097151`，安全（检视阶段复核：单缓冲 int32 ubFactor=59360 → 237440、双缓冲 int64 ubFactor=14816 → 118528，均低于上限）。修复策略：只改批量搬运点（`CopyIndexBatch`），不动逐元素搬运（`blockLen = sizeof(T)` 最大 8 字节，`DataCopyParams` 安全）——按最小改动面修复；选 `DataCopyExtParams` 而非"限制 ubFactor ≤ 65535/indexDtypeSize"（后者把 tiling 能力绑死在搬运 API 的字段宽度上）。

**独立 trace 新增验证信号**（均为 tool_observed）：

- 修复前：大 shape 用例 `[FAIL] 流同步失败: 507035`、`[FAIL] 发现 910230 个不匹配（共 1048576 元素）`（来源平台：950；证据等级：tool_observed）。
- 隔离对照：回退旧 tiling 后同批 54 条用例全部 PASS（含三个 ~1M 输出的大 shape 用例）（来源平台：950；证据等级：tool_observed）。
- 修复后：UT 38/38（tiling 31 + infershape 7）；ST 54/54；大 shape 用例 `1048576 elems, bitwise=YES`、`MERE=0.00e+00, MARE=0.00e+00`；小 shape 无回归（来源平台：950；证据等级：tool_observed）。
- 源码锚点：`kernel_struct_data_copy.h` 中 `struct DataCopyParams` 的 blockLen 为 uint16_t、`struct DataCopyExtParams`（:188）为 uint32_t；`DataCopyPad` 的 `DataCopyExtParams` 重载存在于 arch35 头文件（来源平台：950；证据等级：tool_observed）。

**独立 trace 扩展边界与坏实践**：若单批字节数超过 `DataCopyExtParams.blockLen` 上限 2097151（约 2MB）仍会溢出/被拒，需在 tiling 或 kernel 侧分批；uint16 截断本身不报编译错/告警（连 `-Wall/-Wextra` 也不报，`-Wconversion` 才报，见本卡探针一），不能依赖编译期发现。坏实践补充：`static_cast<uint32_t>(value)` 后赋给 `uint16_t` 字段；依赖旧 tiling 的保守性"恰好不溢出"；流同步 507035 出现时先怀疑 UB 预算满（本例第一假设"UB 用到 99.96% 太满"，回退对照 + 头文件查证后确认真因是 blockLen 截断——先查搬运参数合法性，再怀疑内存预算）。