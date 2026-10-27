---
type: "Runbook"
title: torch.cummin平局索引为last-occurrence
description: 实测 torch.cummin 等值时索引更新为后出现位置（[3,1,4,1]→indices[0,1,1,3]）；与 first-occurrence kernel 对标时平局用例索引必然失配。复现增量（独立复现）：向量化 first-occurrence golden 修正实现（values 复用 torch.cummin、indices 用 changed 掩码+cummax 修正，含 NaN 双 NaN 排除）、NaN 感知自测比较与 indices bitwise exact 口径；教训——CPU 自测期望先被改成 torch 末次语义凑绿，NPU 仍 FAIL 才定位到 golden 语义与规格不符。
tags: [task.testing, topic.precision, topic.compute.ordering]
status: stable
sources: []
created_at: '2026-09-14T12:18:28Z'
updated_at: '2026-09-17T02:20:40Z'
platforms: ["950"]
---

# torch.cummin平局索引为last-occurrence

## 现象

为 Cummin 自定义算子开发 PyTorch ST 测试（`tests/st/torch/golden.py`，以 `torch.cummin` 为 CPU golden 参考实现）时，golden 自测的期望索引按"first-occurrence"（等值保留最先出现的索引）人工构造，结果 3 个自测用例失败，`torch.cummin` 的真实输出与预期不同：

- 输入 `[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]`，期望 indices `[0, 1, 1, 1, 1, 1, 1, 1]`（1.0 首次出现在位置 1），torch.cummin 实际输出 `tensor([0, 1, 1, 3, 3, 3, 3, 3])`——位置 3 出现等值的 1.0 后索引被更新为 3；
- 全零输入 `torch.zeros(16)`，期望全 0，实际 `tensor([0, 1, 2, ..., 15])`；
- 2D dim=0 含等值列的用例同型失配。

即 torch.cummin 的平局（tie）语义是 last-occurrence：等值元素出现时索引随之更新（等效 `<=` 比较更新规则）。而自研 AscendC kernel 按 spec 用 `<`（first-occurrence，全零输入输出全 0 索引）——NPU 实测也确认了这一分歧：平局用例（全零）经 PyTorch 适配层跑 NPU，values 全对、indices 与 torch.cummin 不一致。

## 适用场景

- 工程形态：算子仓开发（ops-registry-invoke）的精度对标——以 torch.cummin/torch.cummax 等 PyTorch 参考实现作为自定义算子 golden 的评测场景。
- API 路径：`torch.cummin`（PyTorch 参考实现）；自定义 kernel 的等值更新规则（`<` vs `<=`）。
- 验证过的平台与 CANN 版本：torch.cummin 行为为 CPU 侧实测（PyTorch/torch_npu 环境）；kernel 侧分歧在 Ascend 950 + CANN 9.1.0 NPU 实测确认。
- 适用粒度：`operator_family`（所有含平局索引语义的最值/扫描类算子对标：cummin/cummax/argmin/argmax 类）。
- 路线优先级：recommended_fastpath（golden 语义显式化）。
- 明确不适用/未验证场景：无等值输入的用例（两种语义等价）；torch.cummax 是否同为 last-occurrence 未实测（同型风险，推断一致但未验证）。

## 判别方法

**触发**：以 torch.cummin 为 golden 的测试在含等值输入（全零、重复值）用例上 indices 大面积失配而 values 全对。

**探针**（可执行，CPU 即可）：

```python
import torch
x = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
print(torch.cummin(x, 0).indices)   # -> tensor([0, 1, 1, 3, 3, 3, 3, 3])
print(torch.cummin(torch.zeros(16), 0).indices)  # -> tensor([0, 1, 2, ..., 15])
```

**原始观测**：如上两条实际输出；NPU 侧 kernel 对全零输入返回全 0 索引（first-occurrence）。

**解释**：torch.cummin 平局时索引更新为后出现位置（last-occurrence，等效 `<=` 更新）；first-occurrence kernel 用 `<` 更新。两者在含等值的输入上 indices 必然分歧，与 kernel 正确性无关——是 golden 与被测实现的语义差异，必须在 golden 设计层显式处理。同族的 TopK 平局对标判别（tie 值索引顺序与 CPU 不一致、按值/索引集合比对）见 [TopK BF16 tie 索引顺序与 CPU 不一致卡](topk_bf16_tie_cpu.md)——那卡是 tie 顺序不可依赖，本卡是 cummin 的具体更新规则实测，两者都要求 golden 层显式处理平局。

## 解决配方

两条正交处理：

1. **以 torch.cummin 为 golden 时**：期望值按 torch 实际语义构造（修正后 verbatim）：

```python
{
    "name": "1D 基本 cummin (float32)",
    "input": torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]),
    "dim": 0,
    "expected_values": torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    # torch.cummin uses <= (last-occurrence for ties)
    "expected_indices": torch.tensor([0, 1, 1, 3, 3, 3, 3, 3], dtype=torch.int64),
},
{
    "name": "Tie 等值 (torch.cummin last-occurrence 语义)",
    "input": torch.tensor([2.0, 2.0, 2.0, 2.0]),
    "dim": 0,
    "expected_values": torch.tensor([2.0, 2.0, 2.0, 2.0]),
    # torch.cummin: <= comparison, last occurrence wins
    "expected_indices": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
},
{
    "name": "全零",
    "input": torch.zeros(16),
    "dim": 0,
    "expected_values": torch.zeros(16),
    # All zeros are equal -> torch.cummin last-occurrence: indices increment
    "expected_indices": torch.arange(16, dtype=torch.int64),
},
```

2. **对标 first-occurrence kernel 时**：不能直接拿 torch.cummin 的 indices 当 oracle——需自写与 kernel 语义一致的 golden（严格 `<` 更新），或把平局用例的 indices 比对改为语义感知（明确标注两套期望）。C++ ST 的 CPU golden 即是自写的 first-occurrence 实现（`val < current_min` 才更新），其平局自测（Tie case - all same values (first-occurrence)）通过。

## 设计决策

- 评测基准选择：算子 spec 若声明 first-occurrence（如本例"相等最小值时保留最先出现的索引"），验收 golden 必须自写同语义实现；torch.cummin 只能作为 values 的参考与无平局输入的 indices 参考。
- 放弃的候选：在比较器里跳过平局用例的 indices 比对（测试放松，掩盖语义分歧，拒绝）。
- 当前方案属 recommended_fastpath。

## 系统信息

- 相关文件：`tests/st/torch/golden.py`（torch.cummin 参考 + 期望值构造）、`tests/st/test_aclnn_cummin.cpp`（自写 first-occurrence CPU golden）、`docs/spec.yaml`/`docs/REQUIREMENTS.md`（first-occurrence 语义契约）。
- 环境：PyTorch + torch_npu（CPU golden 自测）、Ascend 950 NPU（语义分歧实测）。

## 验证信号

验证来自同一开发会话：torch.cummin 的 last-occurrence 输出由 CPU 探针直接实测（[3,1,4,1,...]→[0,1,1,3,3,3,3,3]、全零→arange）（来源平台：CPU；证据等级：tool_observed）；期望值修正后 PyTorch mock 自测 40/40 通过（来源平台：950 环境的 CPU golden 路径；证据等级：tool_observed）；NPU 上 kernel（first-occurrence）与 torch.cummin（last-occurrence）在全零平局用例的 indices 分歧实测确认（values 不受影响）（来源平台：950；证据等级：tool_observed）。

## 失效边界

- torch.cummin 的平局语义实测自当前 PyTorch/torch_npu 版本；不同版本行为如无文档保证可能变化，用前跑探针确认。
- torch.cummax 及带 dim 的其他归约（argmin 等）未逐一实测，同型风险。
- 本卡不评判两种语义孰"正确"——语义选择权在算子 spec，本卡解决的是对标层的语义显式化。

**未决项**：

- torch.cummax/argmin 类接口的平局语义未逐一探针确认（下个 agent 建立对标 golden 前补齐）。
## 坏实践

- 拿 torch.cummin 的 indices 直接当 first-occurrence kernel 的 oracle 而不核对平局语义——所有含等值的用例都会伪失配，误导 kernel 调试方向。
- 在比较器里对 indices 比对做"豁免/跳过"来让测试通过——掩盖语义分歧（测试放松红线）。

## 误诊

- 开发者最初按 PyTorch 文档字面（"index location of each ..."）推断 first-occurrence 并手写期望（agent_reported）——被 torch 实际输出证伪；API 语义以实测探针为准。

## 增量（2026-09-17，并行策展批：独立复现——向量化 first-occurrence golden 修正实现与 NaN/int32 处理）

另一 Cummin 类算子（BenchNoHarnessCummin，Ascend950PR + CANN 9.1.0，torch_npu PyTorch ST）独立复现本卡现象并补齐可复用的 golden 修正实现：`golden.py` 照搬评测任务自带参考（`return torch.cummin(input, dim=dim)`）后，CPU golden 自测 3 FAIL（stable 语义/NaN 传播/INT32 整数），NPU 全相等值用例（shape `(100)`、fp32、data range `(5.0,5.0)`）indices 逐位失配（报错原文 `[indices] Golden[1] = 1` / `[indices] Actual[1] = 0`）。探针再证：`torch.cummin([5.0]*10)` indices=`[0..9]`（末次）；`torch.cummin([10,5,8,3,12,3], int32)` indices=`[0,1,1,3,3,5]`（末尾相等的 3 取位置 5）。算子需求文档规定"相等最小值处理：indices 保留最先出现的索引位置"、设计文档规定比较条件用严格 `<`；需求文档括号里"与 torch.cummin 一致（stable 语义）"的表述本身是错误断言——文档内矛盾时以"严格 `<`"的规格条文和 C++ ST golden（同源实现）为准。

**1. 向量化 first-occurrence golden（终版，verbatim）**——values 复用 torch.cummin（累积最小值与 tie-breaking 无关），indices 用 changed 掩码 + cummax 修正：

```python
def compute_golden_cummin(input: torch.Tensor, dim: int):
    """计算沿 dim 轴的累积最小值 golden 结果。

    使用严格小于 (<) 比较保证 stable 语义 (相等值保留首个索引)。
    NaN 传播行为与 torch.cummin 一致。

    Args:
        input: 输入张量 (CPU, float16/float32/int32/bfloat16)
        dim: 计算累积最小值的轴, 支持负数索引

    Returns:
        (values, indices): 累积最小值张量和索引张量
          - values: 同 input 形状/dtype
          - indices: 同 input 形状, int64 dtype
    """
    # Step 1: 使用 torch.cummin 获取 values (values 与 < / <= 无关)
    v, _ = torch.cummin(input, dim=dim)

    # Step 2: 修正 indices 为 < 语义 (首个出现位置)
    n = v.shape[dim]
    if n <= 1:
        # 单元素或空: indices 全 0
        indices = torch.zeros(v.shape, dtype=torch.int64)
        return v, indices

    # 计算 "changed" 掩码: True 表示该位置出现了新的最小值
    v_prev = v.roll(1, dim)
    if v.is_floating_point():
        # NaN 处理: 两个 NaN 不算 change (NaN != NaN 为 True, 需要排除)
        both_nan = torch.isnan(v) & torch.isnan(v_prev)
        changed = (v != v_prev) & ~both_nan
    else:
        changed = v != v_prev
    # 第一个位置永远是 "change"
    changed.select(dim, 0).fill_(True)

    # 构建 positions 张量: changed 位置填入沿 dim 的索引, 否则填 0
    indices_along_dim = torch.arange(n, dtype=torch.int64)
    # reshape 为可广播形状
    reshape_shape = [1] * v.dim()
    reshape_shape[dim] = n
    indices_along_dim = indices_along_dim.reshape(reshape_shape).expand_as(v)

    positions = indices_along_dim * changed.to(torch.int64)

    # 沿 dim 的累积最大值 = 每个位置对应的首次出现索引
    corrected_idx = positions.cummax(dim=dim).values

    return v, corrected_idx
```

设计要点：dim 轴最长百万级（本工程 1M 元素用例），Python 逐位置循环不可接受——"changed 位置填索引、否则填 0、沿 dim 取累积最大值"一步得到每个位置的首次出现索引；changed 掩码必须 `& ~both_nan`（`NaN != NaN` 为真，不排除会把 NaN 段每个位置都判成"新最小值"）。NaN 语义两者一致：NaN 一旦出现其后 values 全 NaN、indices 指向首个 NaN。

**2. NaN 感知的自测比较（verbatim）**——含 NaN 的期望值不能用 `torch.equal`（NaN != NaN 恒假）：

```python
def _tensors_equal_with_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """比较两个张量是否相等, 支持 NaN (NaN == NaN 视为相等)。"""
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        # NaN 位置必须一致
        nan_a = torch.isnan(a)
        nan_b = torch.isnan(b)
        if not torch.equal(nan_a, nan_b):
            return False
        # 非 NaN 位置精确匹配
        if nan_a.any():
            return torch.equal(a[~nan_a], b[~nan_b])
        return torch.equal(a, b)
    else:
        return torch.equal(a, b)
```

NPU vs golden 的正式比对中，浮点 values 的 MERE/MARE 同样要做 NaN 处理：两边都 NaN 记误差 0、只有一边 NaN 记 inf、golden 为 NaN 时分母改用 `|actual| + eps`（本工程 compare.py 实现，全部 NaN 用例通过）。indices 输出一律 bitwise exact 比对（int64），不放进 MERE/MARE 浮点容差——int32 等按 bitwise exact 验收的 dtype，任何相等元素都会暴露该分歧（无容差掩盖）。

**3. 预防与误诊补充**：为"argmin 类 + 首次出现"算子接 torch 原语 golden 前，先用三个 10 元素以内探针钉死语义（全相等值、含重复值、含 NaN，重点看 indices）；需求文档写"与 torch.xxx 一致"时视为待验证断言。误诊教训（本轨迹实际发生）：第一轮把 CPU 自测 FAIL 归因为"自测期望值写错"，把期望改成 torch.cummin 的末次行为（indices `[0..9]`）使 CPU 自测全绿——直到 NPU 上全相等值用例仍 FAIL 才定位到 golden 语义与算子规格不符。**CPU 自测只能证明 golden 与自身期望一致，不能证明 golden 与算子规格一致**。

**4. 复现增量验证信号（来源：该算子 PyTorch ST 开发与最终精度验收会话，golden 语义修正前后同用例对照）**：

- 修复前：CPU golden 自测 stable/INT32 两条 FAIL（torch.cummin 末次语义）；NPU 全相等值用例 L0_022 indices 失配 `Golden[1]=1 / Actual[1]=0`（来源平台：950；证据等级：tool_observed）。
- 修复前探针：`torch.cummin([5.0]*10)` indices `[0..9]`；`torch.cummin([10,5,8,3,12,3], int32)` indices `[0,1,1,3,3,5]`（末次出现）（来源平台：CPU golden 环境（torch CPU 探针）；证据等级：tool_observed）。
- 修复后：golden 自测 8/8 PASS（含 stable 全相等 / NaN 传播 / INT32 含相等值）（来源平台：CPU golden 环境；证据等级：tool_observed）。
- 修复后：NPU L0_022（(100) fp32 range (5.0,5.0)）PASS；NPU 9 用例 dtype/dim 扫描（fp16/int32/bf16/fp32、负 dim、1M 元素、NaN 输入）全 PASS；验收阶段 L0 30/30 PASS（来源平台：950；证据等级：tool_observed）。
- 修复后：CPU 全量 630 用例 484 通过 / 0 失败 / 146 跳过（>2M 元素用例因 CPU golden 耗时上限跳过，与语义无关）（来源平台：CPU golden 环境；证据等级：tool_observed）。

**5. 复现增量边界**：

- torch.cummin 相等值取末次是否为 PyTorch 文档化稳定行为未查证（仅本环境实测）；torch 版本升级后须重跑探针，若行为变为首次出现则本修正退化为恒等变换（无害）。
- 修正函数对 dim 轴长度 ≤1 的输入直接返回全 0 索引；空 tensor 由上层跳过。
- torch.cummax 自身的索引语义未探针（本修正只依赖 cummax 的数值语义，不涉及其索引输出）。