---
type: "Runbook"
title: "507035 — Host 按逻辑行宽预算导致 UB 溢出"
description: "Host tiling 用逻辑 D 估算 UB，而 Device 按对齐 stride 分配；统一预算公式后 20 个实测用例通过。"
tags: ["task.debugging", "topic.implementation.tiling", "topic.memory"]
status: stable
sources: []
created_at: "2026-08-14T03:13:06Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# 507035 — Host 按逻辑行宽预算导致 UB 溢出

## 现象

完成 UB 行对齐修复后，大逻辑行宽、较小 tile 因子的一组非对齐输入已经通过，但另一组逻辑行宽较小、tile 因子较大的输入仍在 NPU 同步时报 507035，并使后续用例级联跳过。这表明行首对齐已正确，但 Host 选出的 tile 容量仍可超过 Device 的真实 UB 分配。

## 适用场景

- 工程形态：Host 动态 tiling 决定 `ubFactor`，Device 按对齐 stride 初始化深度队列；
- API 路径：`PlatformAscendCManager::GetCoreMemSize` + `TPipe::InitBuffer`；
- 验证过的平台与 CANN 版本：950（Ascend950PR_957c / dav-3510），CANN 9.0.1；
- 适用粒度：`operator_family`；
- 路线优先级：`recommended_fastpath`；
- 明确不适用：Host 与 Device 没有行对齐扩张、或 UB 分配完全固定且不由输入 shape 决定的实现。

## 根因

Host 旧公式用逻辑行宽计算 `perRow_old = 4*D*elementSize + 64`，Device 实际使用 `strideElems = ceil(D/VL)*VL`，因此真实逐行开销是 `perRow_real = 4*strideElems*elementSize + 64`，还要加上各 buffer 的 tail padding。每 tile 的低估量至少为 `4*(strideElems-D)*elementSize*ubFactor`；当 D 刚越过 VL 边界且 `ubFactor` 较大时，累计差额把实际分配推过 UB 上限并触发 507035。

## 解决配方

Host 与 Device 共用同一对齐公式；预算同时计入 x/y 深度队列、两个 32B scalar slot、gamma 与每个 buffer 的 tail padding。以下为完整 tiling 函数 before/after。

#### before

- revision/state: 行对齐修复后、UB 预算修复前的 worktree
- path: `operators/rms_norm/rms_norm.asc`
- symbol: `std::tuple<int64_t, int64_t, int64_t> calc_rms_norm_tiling_params(`
- line span in that trace snapshot: 352-375
- supported claim: 本函数完整呈现该根因修复前或修复后的实际代码形态。

```cpp
std::tuple<int64_t, int64_t, int64_t> calc_rms_norm_tiling_params(
    int64_t numRow, int64_t D, int64_t elementSize)
{
    constexpr static int64_t RESERVED = 8192;  // UB slack for padding + overhead
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    int64_t numBlocks = std::min(coreNum, numRow);
    if (numBlocks <= 0) numBlocks = 1;
    int64_t blockRows = (numRow + numBlocks - 1) / numBlocks;

    int64_t ubBudget = static_cast<int64_t>(ubSize) - RESERVED;
    int64_t gammaBytes = D * elementSize;                 // loaded once per core
    int64_t perRow = 4 * D * elementSize + 64;            // xQ+yQ (depth 2 => 4 row-copies) + two 32B slots
    int64_t ubFactor = (perRow > 0 && (ubBudget - gammaBytes) > 0)
                           ? (ubBudget - gammaBytes) / perRow
                           : 1;
    if (ubFactor < 1) ubFactor = 1;
    if (ubFactor > blockRows) ubFactor = blockRows;
    return std::make_tuple(numBlocks, blockRows, ubFactor);
}
```

#### after

- revision/state: eba9af3（轨迹可见 Git 前缀）
- path: `operators/rms_norm/rms_norm.asc`
- symbol: `std::tuple<int64_t, int64_t, int64_t> calc_rms_norm_tiling_params(`
- line span in that trace snapshot: 359-391
- supported claim: 本函数完整呈现该根因修复前或修复后的实际代码形态。

```cpp
std::tuple<int64_t, int64_t, int64_t> calc_rms_norm_tiling_params(
    int64_t numRow, int64_t D, int64_t elementSize)
{
    constexpr static int64_t RESERVED = 8192;  // UB slack for overhead
    // Ascend950PR/950DT SIMD register VL = 256B -> 64 float elements (matches the
    // device's `AscendC::VECTOR_REG_WIDTH / sizeof(float)`, which is a device-only
    // intrinsic the host cannot name). strideElems/padBytes here mirror the device
    // UB allocation exactly so ubFactor never overflows UB.
    constexpr static int64_t VL = 256 / static_cast<int64_t>(sizeof(float));  // 64
    constexpr static int64_t padBytes = VL * static_cast<int64_t>(sizeof(float));  // 256

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    int64_t numBlocks = std::min(coreNum, numRow);
    if (numBlocks <= 0) numBlocks = 1;
    int64_t blockRows = (numRow + numBlocks - 1) / numBlocks;

    int64_t strideElems = ((D + VL - 1) / VL) * VL;          // matches device
    int64_t strideBytes = strideElems * elementSize;
    int64_t perRow = 4 * strideBytes + 64;                    // 4 row-copies + 2x32B slots
    int64_t fixed = 5 * padBytes + D * elementSize;           // per-buffer padding + gamma
    int64_t ubAvail = static_cast<int64_t>(ubSize) - RESERVED;
    int64_t ubFactor = (perRow > 0 && (ubAvail - fixed) > 0)
                           ? (ubAvail - fixed) / perRow
                           : 1;
    if (ubFactor < 1) ubFactor = 1;
    if (ubFactor > blockRows) ubFactor = blockRows;
    return std::make_tuple(numBlocks, blockRows, ubFactor);
}
```

## 验证代码

下列验证函数来自同一轨迹的实际测试文件；参数矩阵由测试模块提供，函数本身对每个 shape/dtype 调用 NPU 自定义算子并与 CPU reference 比较。测试侧容差在本故障修复前后未放宽，解决配方只改 Host tiling。

### reference helper

- revision/state: bed8346（轨迹可见 Git 前缀，最终版本未改此函数）
- path: `operators/rms_norm/test_rms_norm.py`
- symbol: `_ref_rms_norm`
- line span in that trace snapshot: 55-56
- supported claim: 生成与被测自定义 kernel 独立的 CPU RMSNorm 参考结果。

```python
def _ref_rms_norm(x, gamma, epsilon):
    return torch.nn.functional.rms_norm(x, normalized_shape=(x.shape[-1],), weight=gamma, eps=epsilon)
```

### tolerance helper

- revision/state: bed8346（轨迹可见 Git 前缀，最终版本未改此函数）
- path: `operators/rms_norm/test_rms_norm.py`
- symbol: `_tol`
- line span in that trace snapshot: 59-62
- supported claim: 修复前后使用同一 dtype 容差，未通过放宽测试掩盖 507035。

```python
def _tol(dtype):
    if dtype == torch.float32:
        return dict(rtol=1e-5, atol=1e-5)
    return dict(rtol=1e-2, atol=1e-2)
```

### NPU probe

- revision/state: bed8346（轨迹可见 Git 前缀，最终版本未改此函数）
- path: `operators/rms_norm/test_rms_norm.py`
- symbol: `test_rms_norm_operator`
- line span in that trace snapshot: 65-77
- supported claim: 在物理 NPU 上执行自定义算子并逐 dtype/shape 与 CPU reference 比较。

```python
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_operator(shape, dtype):
    D = shape[-1]
    x = torch.randn(*shape, dtype=dtype)
    gamma = torch.randn(D, dtype=dtype)
    expected = _ref_rms_norm(x, gamma, 1e-6)
    result = torch.ops.op_rms_norm.rms_norm(x.npu(), gamma.npu(), 1e-6).cpu()
    tol = _tol(dtype)
    assert torch.allclose(result, expected, equal_nan=True, **tol), \
        f"rms_norm failed for shape {shape} dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result.float() - expected.float())):.6f}"
```

## 设计决策

- 预算依据物理 `strideElems` 而不是逻辑 D，因为 `InitBuffer` 消耗的是物理布局；
- fixed 与 per-row 开销分开：gamma/tail padding 不随行数线性增长，x/y depth 与 scalar slots 随行数增长；
- 保留一段 RESERVED 余量，但不再用余量掩盖可精确计算的对齐差额；
- 当前方案属于 `recommended_fastpath`。

## 系统信息

- toolchain：CANN 9.0.1；
- target：Ascend950PR_957c / dav-3510；
- Host API：`GetCoreMemSize(CoreMemType::UB)`、`GetCoreNumAiv()`；
- 相关文件：`operators/rms_norm/rms_norm.asc`。

## 预防

- Host tiling 预算必须从 Device 的实际 `InitBuffer` 表达式反推，而不是另写近似公式；
- 对齐后的 stride、queue depth、buffer 数量、scalar slots、tail padding 都要逐项计入；
- 回归至少覆盖 D 刚越过 VL 边界且 `ubFactor` 较大的组合；只测 aligned D 会隐藏低估。

## 验证信号

来源为 Ascend950 上行对齐修复后的再次评测，以及 Host 预算修复后的完整评测。

- 修复前：评测前一组非对齐多行输入已通过，但较小逻辑行宽/较大 `ubFactor` 的输入仍触发 507035（来源平台：950；证据等级：tool_observed）；
- 修复后：目标重新构建成功，pytest 45 项通过，cann-bench 正确性 20/20 通过且进程退出码为 0（来源平台：950；证据等级：tool_observed）；
- 干净重建后再次得到 `[100%] Built target` 与 pytest 45 项通过（来源平台：950；证据等级：tool_observed）。

## 失效边界

预算常量中的 VL 与 tail padding 必须随目标平台/API 重新确认。A2/A3 未验证，推断可迁移的只有 Host 必须按 Device 物理布局预算这一机制，不是具体常量。

## 坏实践

- 用逻辑元素数 D 直接估 UB，却在 Device 端按更大的对齐 stride 分配；
- 只留一个经验 RESERVED 值，不核算 queue depth 和每 buffer padding；
- aligned D 用例通过后就认为 UB 预算公式正确。

## 误诊

该 507035 与前一个行首不对齐根因共享错误码，但触发区间相反：行首问题在多行非对齐时出现，预算问题在对齐扩张比例与 tile 因子乘积较大时出现。不能按错误码合并根因。