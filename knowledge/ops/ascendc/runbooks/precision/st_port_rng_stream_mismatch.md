---
type: "Runbook"
title: PyTorch 移植 ST 随机流不一致：逐位移植 RNG
description: 把 C++ aclnn ST 移植到 PyTorch 时用 torch.randn 生成输入，即使种子相同随机流也不同：near-zero 输出采样不同使 MARE 判 FAIL 而 C++ 同用例 PASS。需逐位移植 C++ 的 LCG+Box-Muller RNG（含 float32 截断）与量化流程。
tags: [task.testing, topic.precision, topic.compute.random_sampling]
status: stable
sources: []
created_at: '2026-09-17T00:58:06Z'
updated_at: '2026-09-17T00:58:06Z'
platforms: ["950"]
---

# PyTorch 移植 ST 随机流不一致：逐位移植 RNG

## 现象

在 Ascend950PR（CANN 9.1.0）上把自定义算子 BenchNoHarnessConv2d 的 C++ ST 测试移植为 PyTorch 测试工程（golden 用 torch 实现、输入用 `torch.randn` + `torch.Generator` 按用例名播种）。首轮单用例验证：L0-001（x[1,64,56,56] filter[64,64,3,3] fp16）MERE=2.03e-6 通过、**MARE=1.48e-2 超阈**（阈值 9.77e-3）失败，最大差异位置 Golden=-75.3125 / Actual=-75.375。而 C++ ST 同用例日志是 `[PASS] MERE=1.76e-04, MARE=4.87e-04`——同一算子、同一 NPU，两个测试工程判定相反。

对齐 C++ 比较器的自适应 epsilon（fp16: epsilon=threshold）后 MARE 降为 1.04e-2 仍超阈；golden 改为 fp32 计算/输出后 MERE 与 C++ 完全一致（1.77e-4 vs 1.76e-4）但 MARE 仍 1.04e-2。也就是说平均误差已对齐、最大相对误差仍采样到超阈的 near-zero 元素——差异只能来自**输入数据本身**：PyTorch 与 C++ 用的不是同一份随机输入。

## 适用场景

- 工程形态：aclnn 算子 ST 测试的跨框架移植/复刻（C++ golden+harness → Python/PyTorch），要求两侧判定结论一致。
- API 路径：C++ 自定义 RNG（LCG + Box-Muller + QuantizeToDType）与 `torch.randn`/`torch.Generator` 的不可互换性；MERE/MARE 比较器公式（epsilon 语义、threshold 表）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0（与 frontmatter `platforms` 一致）。
- 适用粒度：workflow_only——测试工程移植方法论（数据流复现），不涉及 kernel 行为本身。
- 路线优先级：recommended_fastpath（移植 RNG 是两侧判定对齐的最短路径）。
- 明确不适用 / 未验证场景：不要求与 C++ ST 判定一致的独立 PyTorch 测试（可自定随机流，但 near-zero 用例的 MARE 波动需自行承受）；C++ 侧换用标准库 mt19937 的工程需按其 RNG 重写移植。

## 根因

C++ ST 的输入生成是**自定义确定性随机流**，与 `torch.randn` 的生成器算法完全不同——即使种子相同，产生的数据序列也不同：

```cpp
class RNG {
public:
    explicit RNG(uint64_t seed) : state_(seed) {}

    //均匀分布 [0, 1)
    float Uniform() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float)((double)(state_ >> 11) * (1.0 / 9007199254740992.0));
    }

    // 标准正态分布 (Box-Muller)
    float Normal() {
        float u1 = Uniform();
        float u2 = Uniform();
        while (u1 < 1e-10f) u1 = Uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)PI * u2);
    }

    void FillNormal(std::vector<float>& data) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = Normal();
        }
    }

private:
    uint64_t state_;
};
```

数据流水线还有两个必须逐位复现的环节：① 填充顺序固定 x→filter→bias 连续消耗同一 RNG 流；② 生成 fp32 正态后先量化到目标 dtype 再转回 fp32（`QuantizeToDType`：`dst[i] = Half2Float(Float2Half(src[i]))` 等）模拟设备端数据精度。MARE 是**最大**相对误差，对 near-zero 输出元素（golden≈1e-3 以下）的采样极其敏感——随机流不同意味着 near-zero 样本不同，MARE 随之漂移，于是出现"C++ PASS / PyTorch FAIL"的矛盾判定。种子推导本身两侧一致（`seed = Σ case_id[i]*31^...`，再 `*2654435761 + 12345`），但种子相同救不了生成器算法不同。

## 解决配方

在 Python 侧逐位移植 C++ RNG（含 uint64 溢出模拟与 float32 截断模拟）并复现量化流程：

```python
class CppRNG:
    """复现 C++ ST 测试的 RNG (LCG + Box-Muller)

    C++ 源码: test_aclnn_bench_no_harness_conv_2d.cpp → class RNG
    算法:
      - LCG: state = state * 6364136223846793005 + 1442695040888963407 (uint64)
      - Uniform: (float)((double)(state >> 11) * (1.0 / 2^53))
      - Normal: Box-Muller (sqrt(-2*ln(u1)) * cos(2*pi*u2))
    """

    MASK64 = (1 << 64) - 1
    _PI = 3.14159265358979323846

    def __init__(self, seed):
        self.state = seed & self.MASK64

    def _to_float32(self, val):
        """模拟 C++ 的 (float) 转换：double → float32 → double"""
        return struct.unpack('f', struct.pack('f', val))[0]

    def uniform(self):
        """均匀分布 [0, 1)"""
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & self.MASK64
        shifted = self.state >> 11
        result = float(shifted) * (1.0 / 9007199254740992.0)
        return self._to_float32(result)

    def normal(self):
        """标准正态分布 (Box-Muller)"""
        u1 = self.uniform()
        u2 = self.uniform()
        while u1 < 1e-10:
            u1 = self.uniform()
        result = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * self._PI * u2)
        return self._to_float32(result)

    def fill_normal(self, n):
        """生成 n 个正态分布随机数 (float32)"""
        data = [self.normal() for _ in range(n)]
        return torch.tensor(data, dtype=torch.float32)
```

```python
def generate_inputs(tc, device="cpu", seed=None):
    """生成随机输入数据（与 C++ ST 测试完全一致）

    使用 C++ RNG (LCG + Box-Muller) 生成标准正态分布数据，
    然后量化到目标 dtype（模拟设备端精度）。
    每条用例基于 case_id 生成确定性种子，保证可复现。
    """
    if seed is None:
        # 基于 case_id 生成确定性种子（与 C++ ST 测试一致）
        seed = 0
        for c in tc["case_id"]:
            seed = seed * 31 + ord(c)
        seed = (seed * 2654435761 + 12345) & 0xFFFFFFFF

    rng = CppRNG(seed)

    # 1. 生成原始 fp32 随机数据（标准正态分布）
    # 顺序与 C++ 一致: x → filter → bias
    x_raw = rng.fill_normal(tc["N"] * tc["C_in"] * tc["H"] * tc["W"])
    filter_raw = rng.fill_normal(tc["C_out"] * tc["C_in"] * tc["K_h"] * tc["K_w"])
    bias_raw = rng.fill_normal(tc["C_out"])

    # 2. reshape
    x_raw = x_raw.reshape(tc["N"], tc["C_in"], tc["H"], tc["W"])
    filter_raw = filter_raw.reshape(tc["C_out"], tc["C_in"], tc["K_h"], tc["K_w"])
    bias_raw = bias_raw.reshape(tc["C_out"])

    # 3. 量化到目标 dtype（模拟设备端数据精度，与 C++ QuantizeToDtype 一致）
    dt = tc["dtype"]
    x = x_raw.to(dt)
    filter = filter_raw.to(dt)
    bias = bias_raw.to(dt)

    if device != "cpu":
        x = x.to(device)
        filter = filter.to(device)
        bias = bias.to(device)

    return x, filter, bias
```

配套对齐（前置必要条件，但单独不足以通过）：

- 比较器 epsilon 语义对齐 C++：fp16/bf16 用 `epsilon = threshold`（fp16: 2^-10，bf16: 2^-7），fp32 用 `epsilon = 0.1`，而不是需求文档字面的固定 1e-7——near-zero 值（|golden| < threshold）的比较从相对误差退化为绝对误差/threshold；
- golden 在 fp32 下计算并**保持 fp32 输出**（不 cast 回 fp16），NPU 的 fp16 输出 cast 到 fp32 后比对（C++ `ComputeConv2dGolden` 即 fp32 累加/输出）。

```python
def _get_epsilon(dtype: torch.dtype) -> float:
    """根据 dtype 获取自适应 epsilon（与 C++ ST 测试保持一致）

    - fp16/bf16: 使用 threshold 作为 epsilon，使 near-zero 值的比较从相对误差
      退化为绝对误差/threshold，避免 near-zero 值因微小绝对差异而产生过高的相对误差。
    - fp32: 使用 0.1 以吸收 float32 累加精度与 CPU double golden 的差异。
    """
    threshold = _get_threshold(dtype)
    if dtype == torch.float32 or dtype == torch.float64:
        return 0.1  # fp32: 吸收 fp32 vs double 大 C_in 累加精度差异
    else:
        return threshold  # fp16/bf16: 使用 threshold 作为 epsilon
```

## 设计决策

- 为何逐位移植 RNG 而不是"各自随机、各自判定"：本任务的验收标准是 PyTorch 测试与已验收的 C++ ST **结论一致**（187 用例同判）；随机流不同则 near-zero 采样不同，MARE 判定无法对齐。
- 为何用 `struct.pack/unpack('f')` 模拟 float32 截断：C++ `Uniform()/Normal()` 返回 `float`（32 位），Python 浮点是 double；不截断会让量化前的数据在最低位偏离，量化后偶发不同样本。数据最终量化到 fp16 时该差异多数被吸收，但为确定性仍保留截断。
- 为何选"对齐 C++ 实现"而不是"改需求文档公式"：C++ ST 是已通过验收的事实标准；需求文档的 `|golden| + 1e-7` 是理论公式，实测 near-zero 值按 1e-7 分母会把相对误差放大到不可用（golden=0.001、actual=0.0015 → rel_err=50%）。**这不是放宽容差**：对齐的是两个比较器的语义，对齐后仍有 10 条用例因真实差异失败（见坏实践）。

## 验证信号

来源说明：以下信号来自该算子 PyTorch ST 移植的单用例判定对齐闭环。

- 移植前：L0-001 MARE=1.48e-2（固定 1e-7 epsilon）FAIL；对齐自适应 epsilon 后 MARE=1.04e-2 仍 FAIL；golden 改 fp32 后 MERE=1.77e-4 与 C++ 日志 1.76e-4 一致但 MARE=1.04e-2 仍 FAIL（来源平台：950；证据等级：tool_observed）。
- 移植 CppRNG 后（epsilon/golden 对齐已就位）：L0-001 单用例 PASS（1/1，通过率 100%，Golden 自测 PASS）（来源平台：950；证据等级：tool_observed）。
- C++ 侧参照（同用例）：`[Real] L0-001 ... [PASS] MERE=1.76e-04, MARE=4.87e-04 (threshold=9.77e-04)`（来源平台：950；证据等级：tool_observed）。

## 失效边界

- CppRNG 移植以该 C++ 测试的 RNG 实现为准；其它工程的 RNG（mt19937、random_device 等）需按各自源码重写，本卡代码不可直接照搬。
- `_to_float32` 截断假设 C++ 侧 `sqrtf/cosf/logf` 与 Python `math.sqrt/cos/log` 在 double 中间精度下截断到 float32 后逐位一致；本例 187 用例验证通过，但极端舍入边界未做位级证明。
- 随机流对齐只解决"判定一致性"；它不能修复真实精度缺陷（对齐后仍失败的用例是真差异，需另查根因——本例为 golden 累加顺序差异，见[golden 归约顺序与 kernel 不一致卡](golden_reduce_order_near_zero_mare.md)）。

**未决项**：

- 无（随机流对齐闭环完整；对齐后剩余失败用例由 golden 累加顺序根因解释，另见独立卡片）。
## 坏实践

- 用"放宽容差到恰好通过"代替随机流对齐——本会话中 epsilon 对齐后 MARE 仍超阈，说明真实差异仍在；若当时直接再放大 epsilon 掩盖 1.04e-2，会把后续 10 条用例的真实 golden 差异一起掩盖。
- 只对齐 MERE 不看 MARE：MERE 是均值、对 near-zero 不敏感，两侧 MERE 一致（1.77e-4≈1.76e-4）不代表判定一致。

## 误诊

- "算子精度问题"初判：MARE 超阈首先被当作算子精度损失；但 C++ ST 同用例通过、且 PyTorch 侧 MERE 已与 C++ 一致，反证是测试工程侧的数据/比较语义差异。
- epsilon 对齐即通过的预期：对齐后仍 1.04e-2 失败——单因修复未闭环时不要下结论，继续比对数据流。
