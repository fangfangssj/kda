---
type: "Runbook"
title: "philox offset 推进量须固定 12 而非元素数"
description: "自定义 exponential_ host 用 philox_engine_inputs(totalElements) 推进 generator offset，但原生 aclnnSimThreadExponential 每次调用固定推进 offset 12（与 tensor size 无关），导致多次连续调用从第 2 次起 0% 匹配。"
tags: ["task.debugging", "topic.precision", "topic.compute.random_sampling"]
status: stable
sources: []
created_at: "2026-08-18T16:41:25Z"
updated_at: "2026-08-24T00:00:00Z"
platforms: ["950"]
---

# philox offset 推进量须固定 12 而非元素数

## 现象

在用 `PhiloxRandom<10>` 修复单次调用一致性后，单次调用逐元素一致。但模型中 `exponential_` 在一次推理里被连续调用约 260 次（Gumbel-max 采样每步调用），写出 `compare_multi_call.py` 对 `shape=(1,387840)` seed=42 连续调用 5 次 `exponential_` 对比 Native vs Custom，qrun 输出原文：

```
Multi-call Comparison: shape=(1, 387840), seed=42, calls=5
Call 1: exact_match=True, matching=387840/387840
Call 2: exact_match=False, matching=0/387840
  Max diff: 13.794339
  First mismatch at index 0
    Native: 2.4778473377227783
    Custom: 0.5191850066184998
Call 3: exact_match=False, matching=0/387840
...
STATUS: DIVERGENCE DETECTED
```

第 1 次匹配、第 2 次起 0% 匹配——这是 generator state（offset）在调用间推进不一致：单次 RNG 算法对齐了，但每次调用后 generator 内部 offset 的推进量不同，导致第 2 次起 seed 相同但起始 offset 不同，序列完全错位。

## 适用场景

- **工程形态**：kernel 直调（自定义 host `exponential_custom()` 通过 `torch_npu` 的 NPUGeneratorImpl 取 seed/offset 后 launch 自定义 kernel）。
- **API 路径**：`torch_npu::NPUGeneratorImpl::philox_engine_inputs(uint64_t increment)`（`torch_npu/csrc/aten/NPUGeneratorImpl.h`）。
- **验证平台与版本**：Ascend950；CANN 版本未恢复；torch_npu 的 NPUGeneratorImpl 头文件路径 `/root/miniconda3/envs/wsy_dsv4_test/lib/python3.10/site-packages/torch_npu/include/torch_npu/csrc/aten/NPUGeneratorImpl.h`。
- **适用粒度**：`operator_family`（所有从 NPU Generator 取 Philox state 的自定义 RNG 算子 host 侧）。
- **路线优先级**：`recommended_fastpath`（与原生 offset 推进对齐是多次调用一致的必要条件）。
- **不适用或未验证场景**：单次调用场景（不涉及 offset 推进）；不使用 NPU Generator 的 stateless RNG；a2/a3 未实测（见 `## 失效边界`）。

## 判别方法

**触发**：单次调用逐元素一致（RNG rounds 已对齐），但连续多次调用从第 2 次起 0% 匹配（`STATUS: DIVERGENCE DETECTED`），且第 1 次总是匹配——指向 generator state 在调用间的推进量不一致。

**探针**：编写 `diag_offset.py`，在调用 `exponential_` 前后各读一次 `torch.npu.default_generators[0].get_offset()`（或等价的 `default_generator` offset 接口），打印每次调用的 offset 变化；再写 `diag_offset_formula.py` 对不同 tensor size 各调用一次，记录 increment 与 size 的关系表。

**原始观测**（`diag_offset.py` 对 `shape=(1,387840)` 连续 5 次，qrun 输出原文）：

```
Call 1: offset 0 -> 12, increment=12
  totalElements=387840, (n+3)//4*4=387840, n//4=96960
  Output first 5: [1.8045962  2.4804842  0.6825031  0.73629117 1.2446185 ]
Call 2: offset 12 -> 24, increment=12
  totalElements=387840, (n+3)//4*4=387840, n//4=96960
  Output first 5: [2.4778473  0.5126816  0.34211105 0.8688191  0.18528074]
Call 3: offset 24 -> 36, increment=12
Call 4: offset 36 -> 48, increment=12
Call 5: offset 48 -> 60, increment=12
```

`diag_offset_formula.py` 的 size→increment 表（qrun 输出原文）证明 increment 与 size 无关：

```
       size  increment   size/inc     size/4    inc*4
-------------------------------------------------------
           4         12       0.33          1       48
           8         12       0.67          2       48
          16         12       1.33          4       48
         128         12      10.67         32       48
        1024         12      85.33        256       48
       32768         12    2730.67       8192       48
      387840         12   32320.00      96960       48
   529530880         12 44127573.33  132382720       48
```

increment 恒为 12（`inc*4=48` 列也恒定），size 从 4 到 529530880 都不变。

**解释（root-cause interpretation）**：自定义 host 侧 `exponential_custom` 用 `philox_engine_inputs(totalElements)` 向 generator 申请 Philox engine inputs，把 `totalElements`（如 387840）当作 increment 传入。但 `philox_engine_inputs(increment)` 的参数是 **generator offset 的推进量**，不是元素总数——原生 `aclnnSimThreadExponential` 每次调用推进的 increment 是**固定值 12**，与 tensor size 完全无关。于是 Custom 每次把 offset 推进 387840，Native 推进 12，第 2 次调用起始 offset 就不同，序列完全错位。原生 `aclnnSimThreadExponential` 是闭源 L0 op，其 offset 推进策略无公开文档；`philox_engine_inputs` 的参数名虽叫 increment，但原生传入的是固定 12（可能是某种 thread/subsequence 维度的固定步长，而非元素数）。这是根因解释，明确为 interpretation，非单点 verbatim 实测。注意：increment=12 只对齐了 offset 推进量，**不**保证 kernel 内部 counter 的 offset 放置位置正确——后者单独修正前 offset=12 的第 2 次调用仍 0% 匹配。

## 验证信号

- `compare_multi_call.py`：Call 1 `exact_match=True 387840/387840`，Call 2-5 全部 `0/387840`，`STATUS: DIVERGENCE DETECTED`（来源平台:Ascend950；证据等级:tool_observed）。
- `diag_offset.py`：原生 5 次调用 offset 推进 `0->12->24->36->48->60`，每次 increment=12（来源平台:Ascend950；证据等级:tool_observed）。
- `diag_offset_formula.py`：size 4..529530880 increment 恒为 12（`inc*4=48` 恒定）（来源平台:Ascend950；证据等级:tool_observed）。
- 将 host `philox_engine_inputs` 参数对齐原生固定步长后，`get_offset()` 观测 offset 推进 `0/12/24/36/48` 与原生逐次一致（来源平台:Ascend950；证据等级:tool_observed）。

## 失效边界

- 本结论在 Ascend950 + 当前 CANN/torch_npu 版本上验证；a2/a3 未实测，**未验证，推断可迁移**（increment=12 可能依赖 torch_npu 的 NPUGeneratorImpl 实现，跨版本/跨平台需用 `gen.get_offset()` 重新诊断确认）。
- increment=12 只解决 offset 推进量；不解决 kernel 内 counter 放置位置。本卡单独成立时多次调用仍发散——offset 对齐后 Call 2 仍 0% 匹配。
- 早期误把 increment 改为 `(totalElements + 3) & ~3u`（向上对齐到 4 的倍数），依据 `philox_offset_per_thread_ % 4 == 0` 约束——但 387840 本就是 4 的倍数，该"对齐"对 387840 无变化，offset 推进仍是 387840 而非 12，多次调用仍发散。真正 increment 是与 size 无关的固定 12，只能靠 `get_offset()` 诊断得出。
- 若 torch_npu 升级后 NPUGeneratorImpl 的 `philox_engine_inputs` 语义改变（如改为按元素数推进），本对齐失效。