---
type: "Runbook"
title: "PhiloxRandom 默认 7 rounds 与原生 10 不匹配"
description: "自定义 AscendC exponential_ kernel 沿用 CANN philox.h 模板默认 Rounds=7，与原生 aclnnSimThreadExponential 的 Philox-4x32-10（10 rounds）不一致，单次调用即 0% 逐元素匹配；探针用 PhiloxRandom<10> 构建后单次调用 100% 一致，据此判别 rounds 数是单次调用错位的根因。"
tags: ["task.debugging", "topic.precision", "topic.compute.random_sampling"]
status: stable
sources: []
created_at: "2026-08-18T08:42:25Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# PhiloxRandom 默认 7 rounds 与原生 10 不匹配

## 现象

自定义 AscendC `StatelessRandomUniformV3` kernel 接入 deepseek-v4 模型替换 `aten::exponential_`（用于 `model_infer.py:102` 的 Gumbel-max 采样 `probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)`）后，推理能跑通、`[ATEN-REPLACE]` 与 `[KERNEL-REPLACE]` 打印次数也对（260/257），但 decode 输出与基线不一致。

为定位差异，编写对比脚本 `compare_exponential.py`：在两个独立进程里分别加载自定义 `.so` 与原生 `exponential_`，固定 `torch.manual_seed(42)` + `torch.npu.manual_seed_all(42)`，对同一 shape `(1,128)` seed=42 offset=0 各调用一次 `exponential_`，逐元素对比。结果（qrun 输出原文）：

```
Shape=(1, 128), seed=42
  Native first 10: [1.8045962  2.4804842  0.6825031  0.73629117 1.2446185  0.60963553
 0.12777337 0.9405528  1.856717   0.05927224]
  Custom first 10: [0.8961101  0.58456844 1.5179725  1.0605239  1.1845177  1.065725
 0.8438821  1.5797583  0.10162541 0.5730974 ]
  Exact match: False
  Matching elements: 0/128 (0.0%)
```

Custom 与 Native 的前 10 个值完全不同、0% 匹配——这说明 RNG 算法参数本身就不对，而非多核切分或 offset 推进问题（那些只在多次调用时才暴露）。

## 适用场景

- **工程形态**：kernel 直调（自定义 AscendC kernel 通过 PrivateUse1 dispatch 拦截 `aten::exponential_`，替换原生 `aclnnSimThreadExponential`）。
- **API 路径**：`AscendC::PhiloxRandom<Rounds, T>`（`adv_api/math/philox.h`）+ 自定义 host `exponential_custom()`。
- **验证平台与版本**：Ascend950（模型 config 路径 `config/ci_950/deepseek_v4.yaml`；kernel 打印 `[AIV Block 0/56]`/`[AIV Block 0/32]`）；CANN 版本未从 trace 恢复。
- **适用粒度**：`operator_family`（所有用 `PhiloxRandom` 实现 RNG 算子、且需与原生 `exponential_`/`uniform` 逐 bit 对齐的自定义 kernel）。
- **路线优先级**：`recommended_fastpath`（用 10 rounds 是与原生对齐的必要条件，非可选优化）。
- **不适用或未验证场景**：非 RNG 类算子；不需要与原生逐 bit 对齐的场景（如只要求统计分布正确）；a2/a3 未验证（见 ## 失效边界）。

## 根因

自定义 kernel 调用 `AscendC::PhiloxRandom` 时沿用了 CANN 公共头文件 `philox.h` 的模板默认值 `Rounds=7`，但原生 `aclnnSimThreadExponential` 实际使用 PyTorch 标准 **Philox-4x32-10**（10 rounds）。rounds 数直接决定 Philox 单轮轮函数的迭代次数，7 vs 10 产出的 32 位输出完全不同，因此即使 seed/offset/counter 完全相同，单次调用也 0% 匹配。

CANN 公共头文件 `philox.h`（`/Ascend/cann/x86_64-linux/asc/include/adv_api/math/philox.h`）的模板声明 verbatim 如下（默认 `Rounds = 7`，这是误导自定义实现者的根源）：

```cpp
template <uint16_t Rounds = 7, typename T>
__aicore__ inline void PhiloxRandom(
    const LocalTensor<T>& dstLocal, const PhiloxKey& philoxKey, const PhiloxCounter& philoxCounter, uint16_t count)
{
    PhiloxRandomImpl<Rounds>(dstLocal, philoxKey, philoxCounter, count);
}

template <uint16_t Rounds = 7, typename T>
__aicore__ inline void PhiloxRandom(const LocalTensor<T>& dstLocal, const PhiloxKey& philoxKey,
    const PhiloxCounter& philoxCounter, const PhiloxRandomParams& params)
{
    PhiloxRandomImpl<Rounds>(dstLocal, philoxKey, philoxCounter, params);
}
```

## 判别方法

- **触发（trigger）**：自定义 `exponential_` kernel 接入模型后 decode 结果与基线不一致；单进程对比脚本对同一 seed=42/offset=0 单次调用 `exponential_`，Custom 与 Native 输出 0% 逐元素匹配（`Matching elements: 0/128 (0.0%)`）。tool_observed（compare_exponential.py qrun 输出）。
- **探针（probe）**：编写 `compare_exponential.py`，两进程（一加载自定义 `.so`、一不加载）固定 `torch.manual_seed(42)` + `torch.npu.manual_seed_all(42)`，对 `shape=(1,128)` offset=0 调用 `torch.empty(shape).exponential_(1)`，逐元素对比前若干值与匹配率；在自定义 kernel 上分别用 `PhiloxRandom<7>` 与 `PhiloxRandom<10>` 构建 `.so` 跑同一对比。可执行检查：同 seed/offset 单次调用匹配率为 0% 即命中本判别。
- **观测（observation）**：`PhiloxRandom<7>`（CANN 默认）`Matching elements: 0/128 (0.0%)`，Custom 与 Native first 10 完全不同；`PhiloxRandom<10>` `Exact match: True`、`Matching 128/128`，Native 与 Custom first 16 逐元素完全相同：

```
Native first 16: [1.8045962  2.4804842  0.6825031  0.73629117 1.2446185  0.60963553
 0.12777337 0.9405528  1.856717   0.05927224 0.9286345  0.0783492
 0.8092845  0.45824048 0.27046707  0.11940057]
Custom first 16: [1.8045962  2.4804842  0.6825031  0.73629117 1.2446185  0.60963553
 0.12777337 0.9405528  1.856717   0.05927224 0.9286345  0.0783492
 0.8092845  0.45824048 0.27046707  0.11940057]
```

  10 rounds 多 shape 验证：`128/1024/129280/387840` 元素全部 `Exact match: True (100.0%)`。tool_observed（compare_exponential.py qrun 输出）。
- **判读（interpretation）**：原生 `aclnnSimThreadExponential` 是闭源 L0 op，无公开文档说明其 Philox rounds 数，但 PyTorch/CUDA 侧的 `Philox4x32` 标准是 10 rounds；CANN `philox.h` 把模板默认值设为 7（可能服务于其他 L0 op），自定义实现者若直接沿用默认 7 就会与原生 `exponential_` 错位。改为显式 `PhiloxRandom<10, uint32_t>` 后单次调用逐元素一致。此判读是 agent 的根因分析（interpretation），由 7→10 对比结果反推得到；单次调用一致只证明 RNG 算法参数对齐，**不**证明多次调用的 generator state 推进也对齐（多次调用 offset 推进与 counter 布局对齐需另卡诊断）。

## 验证信号

- 单次调用 `PhiloxRandom<7>`：`Matching elements: 0/128 (0.0%)`，Custom 与 Native first 10 完全不同（来源平台:Ascend950；证据等级:tool_observed）。
- 单次调用 `PhiloxRandom<10>`：`Exact match: True`，`Matching 128/128`，first 16 逐元素相同（来源平台:Ascend950；证据等级:tool_observed）。
- 10 rounds 多 shape（128/1024/129280/387840）：全部 `Exact match: True (100.0%)`，`STATUS: ALL PASSED`（来源平台:Ascend950；证据等级:tool_observed）。
- CANN `philox.h` 模板默认 `Rounds = 7`（来源平台:Ascend950；证据等级:artifact_verified）。

## 失效边界

- 本结论在 Ascend950 + 当前 CANN 版本上验证；a2/a3 未在本 trace 内实测，**未验证，推断可迁移**（Philox-4x32-10 是 PyTorch 跨平台标准，理论上 a2/a3 的原生 `exponential_` 也应一致，但需用同款对比脚本实测确认）。
- 仅证明"单次调用 RNG 算法参数对齐"；多次连续调用的 offset 推进与 counter 布局对齐需另卡诊断（increment=12 与 counter[2:3] 布局），二者未同时满足时多次调用仍发散。
- 若原生实现未来切换 RNG 算法（非 Philox 或非 10 rounds），本对齐失效，需重新诊断。

## 误诊

早期一度以为 decode 不一致是"基线本身乱码/temperature=1.0 随机性高"的正常表现（基线 3 次运行输出完全相同但内容是多语言乱码）。这是 `agent_reported` 的误判——基线 3 次一致正说明 seed 固定下序列稳定，替换后输出不同只能是 RNG 序列差异，不是"随机性的正常表现"。该误判未进入验证信号。