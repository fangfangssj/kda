---
type: "Runbook"
title: "torch.unique ±0 符号不稳定致 bit-exact 失败"
description: "PyTorch CPU torch.unique 对 ±0 合并后的符号位选择依输入顺序不稳定（可能输出 -0.0），NPU kernel 归一为 +0.0，bit-exact 比对失败；golden 用 y + 0.0 归一化（IEEE 754 RN 下 (-0)+(+0)=+0，保 dtype 保 ±inf）后全量 151/151 通过。"
tags: [task.testing, topic.precision]
status: stable
sources: []
created_at: '2026-09-17T02:30:14Z'
updated_at: '2026-09-17T02:30:14Z'
platforms: ["950"]
---

# torch.unique ±0 符号不稳定致 bit-exact 失败

## 现象

为去重算子（对标 `torch.unique(x, return_inverse)`，ascend950，CANN 9.1.0）搭建 PyTorch ST 测试（torch_adapter + golden.py 用 CPU `torch.unique` 做参考、阈值 0 的 bit-exact 比对）。golden 自测 21/21 通过、首个 NPU 冒烟用例通过后跑 L0 全量（34 例），唯一失败的是含 `pos_neg_zero` 输入池的 fp16 用例：

```
[FAIL] L0-029 dtype=float16 ri=True ... special=pos_neg_zero ...
  golden y bits: [-32768, 15360, ...]  # -0.0 (0x8000) 在首位
  npu     y bits: [0, 15360, ...]      # +0.0 (0x0000)
```

即 golden（CPU torch.unique）对这组输入输出了 **-0.0**（bit 0x8000），而 NPU kernel 输出 **+0.0**（bit 0x0000），数值相等但位模式不同，bit-exact 判失败。探针复现确认：同一 fp16 输入同时含 +0（bit 0）与 -0（bit -32768）时，CPU `torch.unique` 的 y 里零值的符号位取决于输入顺序/排序稳定性——**实现细节决定、不保证**；换一组顺序就可能是 +0。而算子需求明确 ±0 必须合并为同一唯一值、C++ 版 golden 也显式做了 -0→+0 归一化（NormalizeZero），NPU 输出 +0.0 与之一致。也就是说：**PyTorch golden 在"±0 合并后保留哪个符号"这一实现未定义维度上过度指定了**，golden 需要显式归一化。

## 适用场景

- 工程形态：浮点去重/unique/排序类算子的 bit-exact ST 测试（PyTorch golden + NPU 实现对照）。
- API 路径：`torch.unique(x, return_inverse=True)`（CPU 参考）；golden 输出归一化 `y + 0.0`；位级比对（`view(torch.int16/int32)`）。
- 验证过的平台与 CANN 版本：Ascend950PR，CANN 9.1.0；PyTorch 2.7.1+cpu + torch_npu 2.7.1.post4。
- 适用粒度：operator_family——任何对浮点输入做去重/合并、且测试采用 bit-exact 阈值的算子。
- 路线优先级：recommended_fastpath——golden 显式归一化符号零是应采用的做法。
- 明确不适用 / 未验证场景：数值容差（非 bit-exact）比对的测试不受影响（±0 数值相等）；torch.unique 的 ±inf、NaN 语义不在本卡范围（同轨迹探针确认 ±inf 独立保留，NaN 未测）。

## 根因/解决配方

**根因（单句归因）**：CPU `torch.unique` 对 ±0 合并后的符号位选择依输入顺序不稳定（可能保留 -0.0），而 NPU kernel 与算子需求/C++ golden 均归一为 +0.0，golden 缺少符号零归一化导致 bit-exact 比对在 ±0 混合输入上失败。

**机制（算式）**：IEEE 754 round-to-nearest 模式下 `(-0.0) + (+0.0) = +0.0`；`(+0.0) + (+0.0) = +0.0`；正常值 `x + 0.0 = x`；`±inf + 0.0 = ±inf` 不变。因此对 golden 的浮点输出统一做 `y + 0.0` 即可把 -0 归一为 +0 且不动其他任何位模式；PyTorch 标量加法不提升 tensor dtype（fp16/bf16/fp32 均保持原 dtype）。

**修复（before/after，verbatim——golden.py）**：

修复前：

```python
    y, inverse = torch.unique(x, return_inverse=True)
    # torch.unique 对 ND 输入返回原 shape 的 inverse, 需展平为 1D
    inverse = inverse.flatten()
    if return_inverse:
        return y, inverse
    return y
```

修复后：

```python
    y, inverse = torch.unique(x, return_inverse=True)
    # torch.unique 对 ND 输入返回原 shape 的 inverse, 需展平为 1D
    inverse = inverse.flatten()
    # 浮点 -0.0 归一化为 +0.0 (对标 C++ golden NormalizeZero + NPU 行为)
    # IEEE 754: (-0.0) + (+0.0) = +0.0 (round-to-nearest 模式)
    # torch.unique 的 ±0 sign 选择不稳定, 需显式归一化保证 bit-exact 一致
    if y.dtype.is_floating_point:
        y = y + 0.0
    if return_inverse:
        return y, inverse
    return y
```

并补一个专门触发 -0 的 golden 自测（该输入下 torch.unique 原始输出含 -0.0，验证归一化后为 +0.0）：

```python
    # -0 归一化验证: 此输入下 torch.unique 原始输出含 -0.0 (bit 0x8000),
    # golden 需归一化为 +0.0 (bit 0x0000) 以匹配 NPU 行为
    x_fp16_negzero = torch.tensor(
        [-0.0, 0.0, 1.0, 1.0, 0.0, -0.0, 1.0, 1.0,
         2.0, 3.0, 3.0, 1.0, 0.0, 2.0, -0.0, -0.0], dtype=torch.float16)
    check("fp16 -0归一化", x_fp16_negzero, True,
          expected_y=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float16),
          expected_inv=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 2, 3, 3, 1, 0, 2, 0, 0],
                                    dtype=torch.int64))
```

**这是测试脚手架修复而非测试放松的边界**：修改的是 golden 在"实现未定义维度（±0 符号）"上的过度指定，使其与算子需求（±0 合并、+0 规范形）和 C++ golden 一致；NPU 实现未被改动也未被放宽阈值（仍是 bit-exact、阈值 0）。若反过来 NPU 输出 -0 而 golden 归一为 +0 之外再去改期望值迁就错误实现，则属禁止的测试侧放松。

## 设计决策

- **为何选 `y + 0.0` 而不是位操作清符号位**：一行、dtype 安全（标量加法不升 dtype）、语义就是 IEEE 规范的 +0 合成；位操作（`view(int16) & 0x7FFF`）要按 dtype 分支且易错。
- **为何归一化 golden 而不是改比对器把 ±0 视为相等**：bit-exact 是本算子验收标准（int 全系 bit 级、浮点值也要求位一致）；在比对器里开"±0 等价"特例会把规则埋进比较逻辑，而 golden 归一化把规范形（+0）显式声明在参考实现里，与 C++ golden 行为对齐。
- 当前方案属 recommended_fastpath。

## 验证信号

本卡结论来自该算子 PyTorch ST 工程开发会话（ascend950，CANN 9.1.0，torch 2.7.1+cpu / torch_npu 2.7.1.post4）：

- 修复前：该 fp16 pos_neg_zero 用例 bit-exact 失败，golden y 首位 bit -32768（-0.0）vs NPU bit 0（+0.0）（来源平台：950；证据等级：tool_observed）
- 归一化性质探针（CPU）：fp16 [-32768,15360,31744,-1024] → y+0.0 后 [0,15360,31744,-1024]；bf16 [-32768,16256,32640,-128] → [0,16256,32640,-128]；fp32 [-2147483648,...] → [0,...]——三种浮点 dtype 均保 dtype、-0→+0、±inf 位不变（来源平台：950（CPU 参考侧）；证据等级：tool_observed）
- 修复后：golden 自测 22/22（含新增 -0 归一化用例）、L0 34/34、全量 L0+L1 151/151 bit-exact 通过（7 dtype 全覆盖，pos_neg_zero/pos_neg_inf 等特殊值用例在内），复跑一次仍 151/151（来源平台：950；证据等级：tool_observed）

## 失效边界

- 只解决 ±0 符号维度；torch.unique 在其他实现未定义维度（如相等元素的去重顺序对 y 内容无影响但 inverse 映射稳定）上的行为不在本卡范围。
- `y + 0.0` 依赖 round-to-nearest 舍入模式（PyTorch 默认）；若环境改全局舍入模式（非默认）需重新验证。
- NaN 输入未测（本算子验收不含 NaN 用例）。
- 其他 NPU 世代 kernel 的 ±0 规范形未验证，推断可迁移（规范形属算子需求层）。

**未决项**：

- 无高优先项。可选：golden 自测为三种浮点 dtype 各加一个 -0 触发用例（当前仅 fp16）。
## 坏实践

- 用 CPU 参考实现直接当 bit-exact golden 而不审视其实现未定义维度——±0 符号这类"看运气"的位模式会让测试随机红。
- 发现 golden 与 NPU 位不一致时先怀疑实现、直接改 NPU 代码迁就 CPU 输出——应先判断哪个才是规范形（本例需求文档与 C++ golden 均指明 +0）。

## 误诊

- 无明显误诊轮次。排查中两段探针脚本自身的 Python 错误（`torch.int16` 当函数调用、变量名 NameError）即时修正，不影响结论。
